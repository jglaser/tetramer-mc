#!/usr/bin/env python3
"""Audit active masks, reverse densities, and every native transition.

Native formation/breakage is evaluated at every accepted physical update.
Every hard-valid global candidate is also classified, including rejections.
The native catalogue and atom-union checks are independent post-hoc tools.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from analyze_free_tetramer_campaign import read, sha, save


class AtlasDensityAudit:
    """Independent NumPy reconstruction of the specified atlas fit and density."""

    def __init__(self, model, config, body_bound):
        import numpy as np
        from scipy.special import logsumexp, gammaln
        self.np = np
        self.logsumexp = logsumexp
        self.cfg = config['atlas_transport']
        self.mask_cfg = config['atlas_mask']
        self.means = np.asarray(model['means'])
        self.weights = np.asarray(model['weights'])
        self.weights /= self.weights.sum()
        self.k = len(self.weights)
        self.lower = np.linalg.cholesky(model['covariances'])
        self.inverse_lower = np.linalg.inv(self.lower)
        self.anchor_t = np.asarray([a['position'] for a in model['anchors']])
        self.anchor_r = np.asarray([a['rotation'] for a in model['anchors']])
        self.angular_length = model['angular_length']
        self.cube = 2*(config['boundary']['radius']+body_bound)
        self.uniform_weight = config['learned_uniform_weight']
        self.tril = np.tril_indices(6)
        self.mean_indices = np.arange(27*self.k).reshape(self.k, 27)[:, :6].flatten()
        self.covariance_indices = np.arange(27*self.k).reshape(self.k, 27)[:, 6:].flatten()
        self.gain = np.full(28*self.k-1, self.cfg['weight_gain'])
        self.noise = np.full(28*self.k-1, self.cfg['weight_noise'])
        for indices, name in ((self.mean_indices, 'mean'), (self.covariance_indices, 'covariance')):
            self.gain[indices] = self.cfg[name+'_gain']
            self.noise[indices] = self.cfg[name+'_noise']
        k = np.arange(self.k+1)
        logits = k*np.log(self.mask_cfg['activity'])-gammaln(k+1.)
        upper = self.mask_cfg['max_components']
        upper = self.k if upper is None else upper
        self.support = self.mask_cfg['min_components'], upper
        logits[(k < self.support[0]) | (k > upper)] = -np.inf
        self.log_count = logits-logsumexp(logits)
        self.log_label = (np.log(self.weights) if self.mask_cfg['label_law'] == 'atlas_weight'
                          else np.zeros(self.k))
        self.log_e = np.full(self.k+1, -np.inf)
        self.log_e[0] = 0.
        for weight in self.log_label:
            self.log_e[1:] = np.logaddexp(self.log_e[1:], self.log_e[:-1]+weight)
        self.max_density_error = self.max_fit_error = self.max_prior_error = 0.
        self.density_checks = self.fit_checks = self.prior_checks = 0
        self.fit_calls = 0

    def arrays(self, poses):
        from scipy.spatial.transform import Rotation
        np = self.np
        p = np.asarray([pose['position'] for pose in poses])
        q = np.asarray([pose['orientation'] for pose in poses])
        return p, Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()

    def latents(self, t, r):
        from scipy.spatial.transform import Rotation
        np = self.np
        matrices = r[:, None]@self.anchor_r[None].swapaxes(-1, -2)
        q = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat().reshape(len(t), self.k, 4)
        with np.errstate(divide='ignore', invalid='ignore'):
            c = q[:, :, :3]/q[:, :, 3, None]
        values = np.empty((len(t), self.k, 6))
        values[:, :, :3] = t[:, None]-self.anchor_t
        values[:, :, 3:] = self.angular_length*c
        return values, c

    def fit(self, poses):
        np = self.np
        self.fit_calls += 1
        p, r = self.arrays(poses)
        pair_mask = ~np.eye(len(p), dtype=bool)
        t = np.einsum('aji,abj->abi', r, p[None]-p[:, None])[pair_mask]
        rotations = np.einsum('api,bpj->abij', r, r)[pair_mask]
        latent, _ = self.latents(t, rotations)
        with np.errstate(invalid='ignore', over='ignore'):
            residual = np.einsum('kij,mkj->mki', self.inverse_lower, latent-self.means)
            distances = np.linalg.norm(residual, axis=2)
        distances[~np.isfinite(distances)] = np.inf
        assignment = np.argmin(distances, axis=1)
        selected = distances[np.arange(len(t)), assignment] <= self.cfg['assignment_cutoff']
        coordinates = np.zeros(28*self.k-1)
        counts = np.zeros(self.k, dtype=int)
        for label in range(self.k):
            rows = np.clip(residual[selected & (assignment == label), label],
                           -self.cfg['residual_clip'], self.cfg['residual_clip'])
            counts[label] = len(rows)
            if not len(rows):
                continue
            shrinkage = self.cfg['shrinkage']
            denominator = len(rows)+shrinkage
            mean = rows.sum(axis=0)/denominator
            centered = rows-mean
            covariance = (shrinkage*(np.eye(6)+np.outer(mean, mean))+
                          centered.T@centered)/denominator
            lower = np.linalg.cholesky(covariance)
            lower[np.diag_indices(6)] = np.log(np.diag(lower))
            coordinates[27*label:27*label+6] = mean
            coordinates[27*label+6:27*(label+1)] = lower[self.tril]
        weight_coordinates = np.log(counts/self.weights+self.cfg['shrinkage'])
        coordinates[27*self.k:] = weight_coordinates[:-1]-weight_coordinates[-1]
        return dict(coordinates=coordinates, component_counts=counts,
                    data_count=len(t), assigned_count=int(counts.sum()))

    def check_fit(self, fit, recorded):
        np = self.np
        assert recorded['data_count'] == fit['data_count']
        assert recorded['assigned_count'] == fit['assigned_count']
        assert np.array_equal(recorded['component_counts'], fit['component_counts'])
        error = float(np.max(np.abs(np.asarray(recorded['coordinates'])-fit['coordinates'])))
        assert error < 2e-7, ('fit', error)
        self.max_fit_error = max(self.max_fit_error, error)
        self.fit_checks += 1

    def model(self, fit, eta):
        np = self.np
        values = self.gain*fit['coordinates']+self.noise*np.asarray(eta)
        blocks = values[:27*self.k].reshape(self.k, 27)
        shift = 3*np.tanh(blocks[:, :6]/3.)
        means = self.means+np.einsum('kij,kj->ki', self.lower, shift)
        transform = np.zeros((self.k, 6, 6))
        transform[:, self.tril[0], self.tril[1]] = .5*np.tanh(blocks[:, 6:]/.5)
        for i in range(6):
            coordinate = blocks[:, 6+i*(i+1)//2+i]
            transform[:, i, i] = np.exp(np.log(2)*np.tanh(coordinate/np.log(2)))
        lower = self.lower@transform
        # Rust rebuilds from the full covariance; match that factorization step.
        lower = np.linalg.cholesky(lower@lower.swapaxes(-1, -2))
        logits = np.log(self.weights)
        logits[:-1] += 4*np.tanh(values[27*self.k:]/4.)
        weights = np.exp(logits-self.logsumexp(logits))
        return means, lower, weights

    def log_density(self, pose, anchor, model, labels):
        np = self.np
        p, r = self.arrays([anchor, pose])
        uniform_weight = self.uniform_weight if len(labels) else 1.
        uniform = (np.log(uniform_weight)-3*np.log(self.cube)
                   if np.all(p[1] >= -.5*self.cube) and np.all(p[1] < .5*self.cube) else -np.inf)
        if not len(labels):
            return float(uniform)
        latent, c = self.latents((r[0].T@(p[1]-p[0]))[None], (r[0].T@r[1])[None])
        latent, c = latent[0, labels], c[0, labels]
        means, lower, weights = (x[labels] for x in model)
        weights = weights/weights.sum()
        with np.errstate(invalid='ignore', over='ignore'):
            residual = np.linalg.solve(lower, (latent-means)[..., None])[..., 0]
            logg = (-3*np.log(2*np.pi)-np.log(np.diagonal(lower, axis1=1, axis2=2)).sum(axis=1)
                    -.5*np.sum(residual**2, axis=1)+3*np.log(self.angular_length)
                    +2*np.log(np.pi)+2*np.log1p(np.sum(c*c, axis=1)))
        logg[~np.isfinite(logg)] = -np.inf
        learned = np.log1p(-uniform_weight)+self.logsumexp(np.log(weights)+logg)
        return float(np.logaddexp(uniform, learned))

    def check_density(self, actual, recorded, name):
        error = abs(actual-recorded)
        assert error < 2e-6, (name, actual, recorded, error)
        self.max_density_error = max(self.max_density_error, error)
        self.density_checks += 1

    def check_mask(self, state, recorded=None):
        np = self.np
        labels = state['labels']
        assert labels == sorted(set(labels))
        assert all(0 <= label < self.k for label in labels)
        assert self.support[0] <= len(labels) <= self.support[1]
        logp = self.log_count[len(labels)]+self.log_label[labels].sum()-self.log_e[len(labels)]
        if recorded is not None:
            error = abs(float(logp)-recorded)
            assert error < 2e-10, ('mask prior', error)
            self.max_prior_error = max(self.max_prior_error, error)
            self.prior_checks += 1

    def joint_inclusion_probability(self, labels):
        np = self.np
        required = sorted(set(labels))
        other = [i for i in range(self.k) if i not in required]
        log_e = np.full(len(other)+1, -np.inf)
        log_e[0] = 0.
        for label in other:
            log_e[1:] = np.logaddexp(log_e[1:], log_e[:-1]+self.log_label[label])
        terms = [self.log_count[k]+self.log_label[required].sum()+log_e[k-len(required)]-self.log_e[k]
                 for k in range(len(required), self.k+1) if k-len(required) < len(log_e)]
        return float(np.exp(self.logsumexp(terms)))


def native_bond_transitions(changes):
    """Separate changes of registration label from loss of a body-pair bond."""
    active = set()
    formations = breakages = registry_changes = 0
    for event in changes:
        before = {key[:2] for key in active}
        active -= {tuple(key) for key in event['broken']}
        active |= {tuple(key) for key in event['formed']}
        after = {key[:2] for key in active}
        event['formed_body_pairs'] = sorted(after-before)
        event['broken_body_pairs'] = sorted(before-after)
        event['changed_registry_body_pairs'] = sorted(
            {tuple(key[:2]) for key in event['formed']} &
            {tuple(key[:2]) for key in event['broken']} & before & after)
        formations += len(after-before)
        breakages += len(before-after)
        registry_changes += len(event['changed_registry_body_pairs'])
    return dict(native_bond_formations=formations, native_bond_breakages=breakages,
                native_registry_changes=registry_changes)


def one(task):
    reference, campaign, out, job = task
    reference, campaign, out = map(Path, (reference, campaign, out))
    sys.path.insert(0, str(reference/'scripts'))
    import numpy as np
    from scipy.spatial.transform import Rotation
    from audit_tetramer_assembly import AtomicAssembly, pose_arrays, rotations
    from tetramer_order import TetramerOrder, graph_summary
    from analyze_precursor_exchange import prepare_templates

    directory = Path(job['directory'])
    cfg, summary, provenance = (read(directory/name) for name in ('config.json', 'summary.json', 'manifest.json'))
    manifest = read(campaign/'manifest.json')
    assert summary['complete'] and summary['completed_sweeps'] == manifest['sweeps']
    assert provenance['executable_sha256'] == manifest['binary_sha256']
    assert provenance['model_sha256'] == manifest['model_sha256'] == sha(directory/'provenance/frozen-relative-model.json')
    assert provenance['config_sha256'] == job['config_sha256'] == sha(job['config'])
    assert provenance['shape_sha256'] == sha(directory/'provenance/shape.json')
    assert not cfg['seed_labels'] and not cfg['fixed_body_indices']
    shape = read(directory/'provenance/shape.json')
    atoms = np.asarray([a['center'] for a in shape['atoms']])
    radii = np.asarray([a['radius'] for a in shape['atoms']])
    bound = float(np.max(np.linalg.norm(atoms, axis=1)+radii))
    radius, rd = cfg['boundary']['radius'], cfg['depletant_radius']
    analysis_cfg = copy.deepcopy(cfg)
    analysis_cfg.update(shape=str(directory/'provenance/shape.json'), rigid_members=shape['rigid_members'],
                        box_lengths=[8*(radius+bound)]*3)
    templates = prepare_templates(cfg['monomer_shape'], reference/'results/c1c3-scaffold/motifs.json',
                                  reference/'results/native-neighbor-classes/classification.json')
    atomic = AtomicAssembly(analysis_cfg, directory)
    order = TetramerOrder(analysis_cfg, directory, templates=templates)
    frames = [json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
    assert frames[0]['sweep'] == 0 and frames[-1]['sweep'] == manifest['sweeps']
    n = len(frames[0]['poses'])
    state = copy.deepcopy(frames[0]['poses'])
    all_pairs = list(itertools.combinations(range(n), 2))
    saved = {frame['sweep']: frame for frame in frames}
    active, changes, rows = set(), [], []
    counts, accepted = Counter(), Counter()
    branches, native_proposals = defaultdict(Counter), []
    correction_samples = defaultdict(list)
    atlas_refreshes = []
    atlas_state = frames[0].get('atlas_state')
    mask_state = frames[0]['atlas_mask_state']
    audit = AtlasDensityAudit(read(directory/'provenance/frozen-relative-model.json'), cfg, bound)
    audit.check_mask(mask_state)
    mask_histogram, mask_label_histogram = Counter(), Counter()
    mask_refreshes = []
    reverse_support = Counter()
    reciprocal_pairs = [(14, 19), (21, 22), (7, 7)]
    reciprocal_inclusions = Counter()
    component_counts = defaultdict(Counter)
    fit_cache = None
    maximum_position_error = maximum_rotation_error = 0.

    def current_fit():
        nonlocal fit_cache
        if fit_cache is None:
            fit_cache = audit.fit(state)
        return fit_cache

    def classify(poses, pairs=None):
        result = order.classify({'poses': poses}, pair_filter=pairs)
        entry = {(*r['bodies'], r['motif_id']) for r in result['registered_tetramer_motifs'] if r['entry']}
        stay = {(*r['bodies'], r['motif_id']) for r in result['registered_tetramer_motifs']}
        return entry, stay

    def compare(actual, expected):
        nonlocal maximum_position_error, maximum_rotation_error
        p, q = pose_arrays({'poses': actual})
        ep, eq = pose_arrays({'poses': expected})
        pe = float(np.max(np.abs(p-ep)))
        re = float(np.max(np.abs(rotations(q)-rotations(eq))))
        maximum_position_error = max(maximum_position_error, pe)
        maximum_rotation_error = max(maximum_rotation_error, re)
        assert pe < 5e-8 and re < 5e-10, (job['id'], pe, re)

    def observe(frame):
        nonlocal state, fit_cache
        compare(state, frame['poses'])
        entry, stay = classify(state)
        assert active == (active & stay) | entry
        p, q = pose_arrays(frame)
        matrix = rotations(q)
        world = np.einsum('bij,aj->bai', matrix, atoms)+p[:, None, :]
        wall = float(np.min(radius-np.linalg.norm(world, axis=2)-radii[None, :]))
        assert wall >= -2e-8
        result = atomic.frame(p, q)
        assert result['hard_valid'], (job['id'], frame['sweep'], result)
        near = {tuple(e) for e in result['depletion_edges']}
        native = {key[:2] for key in active}
        assert native <= near
        if frame['sweep'] == 0:
            distance = np.linalg.norm(p[:, None]-p[None, :], axis=2)
            assert np.min(distance[np.triu_indices(n, 1)]) > 2*(bound+rd)
            assert not near and not native
        row = dict(sweep=frame['sweep'], native=graph_summary(n, native),
                   nonspecific=graph_summary(n, near), registered_keys=sorted(active),
                   minimum_atomic_wall_clearance_A=wall, hard_valid=True)
        if atlas_state is not None:
            assert frame['atlas_state'] == atlas_state
            assert frame['atlas_mask_state'] == mask_state
            audit.check_fit(current_fit(), frame['atlas_fit'])
            eta = np.asarray(atlas_state['eta'])
            assert np.isfinite(eta).all()
            row['atlas'] = dict(eta_dimension=eta.size, eta_square_mean=float(np.mean(eta**2)),
                                fit=frame.get('atlas_fit'))
        rows.append(row)
        # Restart replay from exact serialized poses to avoid accumulated roundoff.
        state = copy.deepcopy(frame['poses'])
        fit_cache = None

    observe(frames[0])
    current_sweep, selected = 0, set()
    for serial, line in enumerate((directory/'moves.jsonl').open()):
        move = json.loads(line)
        sweep, kind = move['sweep'], move['kind']
        if sweep != current_sweep:
            if current_sweep:
                assert selected == set(range(n))
                if current_sweep in saved:
                    observe(saved[current_sweep])
            current_sweep, selected = sweep, set()
        counts[kind] += 1
        previous = set(active)
        changed_pairs = []
        accepted_entry_stay = None
        candidate_fit = None
        proposal = move.get('proposal') or {}
        source = proposal.get('branch', kind)
        if kind in ('local', 'global'):
            i = move['moving_index']
            assert i not in selected
            selected.add(i)
            compare([state[i]], [move['old_pose']])
            branch_counts = branches[source]
            branch_counts['attempted'] += 1
            branch_counts['hard_valid'] += int(move['hard_valid'])
            branch_counts['accepted'] += int(move['accepted'])
            affected = [pair for pair in all_pairs if i in pair]
            if kind == 'global':
                labels = mask_state['labels']
                assert proposal['atlas_mask_labels'] == labels
                selected_label = proposal.get('component_index')
                assert proposal['atlas_component_label'] == (None if selected_label is None else labels[selected_label])
                label = proposal['atlas_component_label']
                if label is not None:
                    component_counts[label]['attempted'] += 1
                    component_counts[label]['hard_valid'] += int(move['hard_valid'])
                    component_counts[label]['accepted'] += int(move['accepted'])
                forward_model = audit.model(current_fit(), atlas_state['eta'])
                anchor = state[proposal['anchor_index']]
                old_forward = audit.log_density(state[i], anchor, forward_model, labels)
                audit.check_density(old_forward, proposal['old_log_density'], 'old forward')
                if move['proposed_pose'] is not None:
                    forward = audit.log_density(move['proposed_pose'], anchor, forward_model, labels)
                    audit.check_density(forward, proposal['new_log_density'], 'new forward')
            if kind == 'global' and move['hard_valid']:
                candidate = list(state)
                candidate[i] = move['proposed_pose']
                candidate_fit = audit.fit(candidate)
                reverse_model = audit.model(candidate_fit, atlas_state['eta'])
                reverse = audit.log_density(state[i], anchor, reverse_model, labels)
                audit.check_density(reverse, proposal['transported_reverse_log_density'], 'candidate reverse')
                audit.check_density(reverse-forward, proposal['log_reverse_forward'], 'reverse/forward ratio')
                expected_log_acceptance = min(0., reverse-forward+move['gate']['log_weight'])
                audit.check_density(expected_log_acceptance, move['log_acceptance'], 'acceptance')
                reverse_full = audit.log_density(state[i], anchor, reverse_model, list(range(audit.k)))
                uniform_log = np.log(audit.uniform_weight if labels else 1.)-3*np.log(audit.cube)
                uniform_full_log = np.log(audit.uniform_weight)-3*np.log(audit.cube)
                masked_floor_fraction = float(np.exp(uniform_log-reverse))
                full_floor_fraction = float(np.exp(uniform_full_log-reverse_full))
                lost_reverse_support = full_floor_fraction < .5 and masked_floor_fraction > .99
                reverse_support['hard_valid_global'] += 1
                reverse_support['full_atlas_supported'] += int(full_floor_fraction < .5)
                reverse_support['mask_atlas_supported'] += int(masked_floor_fraction < .5)
                reverse_support['lost_to_uniform_floor'] += int(lost_reverse_support)
                correction_samples[f'{source}_masked_minus_full_reverse'].append(reverse-reverse_full)
                candidate_entry, candidate_stay = classify(candidate, affected)
                novel = candidate_entry-active
                branch_counts['native_candidate'] += int(bool(candidate_entry))
                branch_counts['new_native_candidate'] += int(bool(novel))
                branch_counts['accepted_native_candidate'] += int(bool(candidate_entry) and move['accepted'])
                branch_counts['accepted_new_native_candidate'] += int(bool(novel) and move['accepted'])
                if candidate_entry:
                    native_proposals.append(dict(sweep=sweep, moving_index=i, source=source,
                        atlas_component_label=proposal.get('atlas_component_label'),
                        accepted=move['accepted'], entry=sorted(candidate_entry), novel=sorted(novel),
                        log_acceptance=move['log_acceptance'],
                        log_reverse_forward=proposal.get('log_reverse_forward'),
                        gate_log_weight=(move.get('gate') or {}).get('log_weight'),
                        reverse_uniform_fraction=masked_floor_fraction,
                        full_reverse_uniform_fraction=full_floor_fraction,
                        lost_reverse_atlas_support=bool(lost_reverse_support)))
                    reverse_support['native_candidates_lost_to_uniform_floor'] += int(lost_reverse_support)
                if move['accepted']:
                    accepted_entry_stay = candidate_entry, candidate_stay
                for key, value in [('proposal', proposal.get('log_reverse_forward')),
                                   ('gate', (move.get('gate') or {}).get('log_weight')),
                                   ('acceptance', move.get('log_acceptance'))]:
                    if value is not None:
                        correction_samples[f'{source}_{key}'].append(value)
                        if candidate_entry:
                            correction_samples[f'{source}_native_{key}'].append(value)
                        if move['accepted']:
                            correction_samples[f'{source}_accepted_{key}'].append(value)
            if move['accepted']:
                accepted[kind] += 1
                state[i] = copy.deepcopy(move['retained_pose'])
                fit_cache = candidate_fit
                assert state[i] == move['proposed_pose']
                changed_pairs = affected
        elif kind == 'gca':
            if move.get('accepted', True):
                accepted[kind] += 1
                flipped = set(move['result']['flipped_indices'])
                u = np.asarray(move['axis'])
                u /= np.linalg.norm(u)
                transform = 2*np.outer(u, u)-np.eye(3)
                for i in flipped:
                    p, q = pose_arrays({'poses': [state[i]]})
                    matrix = transform@rotations(q)[0]
                    state[i] = dict(position=(transform@p[0]).tolist(),
                        orientation=Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]].tolist())
                changed_pairs = [(a, b) for a, b in all_pairs if (a in flipped) != (b in flipped)]
                fit_cache = None
        elif kind == 'center_shift':
            if move.get('accepted', True):
                accepted[kind] += 1
                for pose in state:
                    pose['position'] = (np.asarray(pose['position'])+move['result']['displacement']).tolist()
                fit_cache = None
        elif kind == 'atlas_refresh':
            assert atlas_state is not None
            assert move['old_state'] == atlas_state
            atlas_state = copy.deepcopy(move['state'])
            audit.check_fit(current_fit(), move['fit'])
            eta = np.asarray(atlas_state['eta'])
            atlas_refreshes.append(dict(sweep=sweep, eta_dimension=eta.size,
                eta_square_mean=float(np.mean(eta**2)), fit=move.get('fit')))
        elif kind == 'atlas_mask_refresh':
            assert move['old_state'] == mask_state
            mask_state = copy.deepcopy(move['state'])
            audit.check_mask(mask_state, move['log_probability'])
            mask_histogram[len(mask_state['labels'])] += 1
            mask_label_histogram.update(mask_state['labels'])
            for pair in reciprocal_pairs:
                reciprocal_inclusions[str(pair)] += int(set(pair) <= set(mask_state['labels']))
            mask_refreshes.append(dict(sweep=sweep, labels=mask_state['labels'],
                                      log_probability=move['log_probability']))
        else:
            raise AssertionError((job['id'], 'unknown move', kind))
        if changed_pairs:
            entry, stay = accepted_entry_stay or classify(state, changed_pairs)
            subset = set(changed_pairs)
            active = {key for key in active if key[:2] not in subset or key in stay} | entry
        if previous != active:
            changes.append(dict(sweep=sweep, serial=serial, kind=kind, source=source,
                formed=sorted(active-previous), broken=sorted(previous-active),
                moving_index=move.get('moving_index'), component_index=proposal.get('component_index'),
                atlas_component_label=proposal.get('atlas_component_label')))
    assert selected == set(range(n))
    observe(saved[current_sweep])
    checkpoint = read(directory/'checkpoint.json')
    compare(state, checkpoint['poses'])
    assert checkpoint.get('atlas_state') == atlas_state
    assert checkpoint['atlas_mask_state'] == mask_state
    assert len(mask_refreshes) == summary['counts']['atlas_mask_refreshes']
    for kind in ('local', 'global'):
        assert counts[kind] == summary['counts'][kind]['attempted']
        assert accepted[kind] == summary['counts'][kind]['accepted']
    if cfg.get('atlas_transport') is not None:
        assert len(atlas_refreshes) == manifest['sweeps']
    formations = Counter()
    breakages = Counter()
    for change in changes:
        formations[change['source']] += len(change['formed'])
        breakages[change['source']] += len(change['broken'])
    frozen_reproduction = None
    if job['variant'] == 'full' and manifest.get('source_campaign'):
        source_manifest = read(Path(manifest['source_campaign'])/'manifest.json')
        source_job = next(j for j in source_manifest['jobs']
                          if j['replicate'] == job['replicate'] and j.get('variant', j.get('mode')) == 'covariance')
        source_path = Path(source_job['directory'])/'trajectory.jsonl'
        source_frames = {f['sweep']: f for f in map(json.loads, source_path.read_text().splitlines())}
        common = [f for f in frames if f['sweep'] in source_frames]
        exact = all(f['poses'] == source_frames[f['sweep']]['poses'] for f in common)
        assert exact, (job['id'], 'Full-mask trajectory changed from prior covariance control')
        frozen_reproduction = dict(exact=True, compared_frames=len(common),
            all_new_frames_compared=len(common) == len(frames),
            source_trajectory_sha256=sha(source_path))
    result = dict(id=job['id'], replicate=job['replicate'], variant=job['variant'], passed=True,
        **native_bond_transitions(changes),
        counts=summary['counts'], branch_counts={k: dict(v) for k, v in branches.items()},
        cpu_seconds=summary['sampler_cpu_seconds'], cost=summary['cost'], rows=rows,
        final_native=rows[-1]['native'], final_nonspecific=rows[-1]['nonspecific'],
        motif_formations=sum(formations.values()), motif_breakages=sum(breakages.values()),
        formations_by_source=dict(formations), breakages_by_source=dict(breakages),
        first_native_association_sweep=min((c['sweep'] for c in changes if c['formed']), default=None),
        changes=changes, native_proposals=native_proposals, atlas_refreshes=atlas_refreshes,
        atlas_mask=dict(refreshes=mask_refreshes, count_histogram=dict(mask_histogram),
            label_histogram=dict(mask_label_histogram),
            mean_k=sum(k*v for k, v in mask_histogram.items())/sum(mask_histogram.values()),
            prior_count_probabilities=np.exp(audit.log_count).tolist(),
            reciprocal_pair_inclusions={str(pair): dict(
                observed_count=reciprocal_inclusions[str(pair)],
                observed_fraction=reciprocal_inclusions[str(pair)]/len(mask_refreshes),
                prior_probability=audit.joint_inclusion_probability(pair)) for pair in reciprocal_pairs}),
        original_component_counts={str(k): dict(v) for k, v in component_counts.items()},
        independent_density_audit=dict(fit_checks=audit.fit_checks,
            density_checks=audit.density_checks, prior_checks=audit.prior_checks,
            fit_calls=audit.fit_calls, maximum_log_density_error=audit.max_density_error,
            maximum_fit_error=audit.max_fit_error, maximum_prior_error=audit.max_prior_error),
        reverse_support=dict(reverse_support),
        full_reference_reproduction=frozen_reproduction,
        correction_quantiles={k: dict(zip(('q10', 'median', 'q90'),
            np.quantile(v, [.1, .5, .9]).tolist())) for k, v in correction_samples.items()},
        replay=dict(passed=True, records=sum(counts.values()), frames=len(frames),
            maximum_position_error_A=maximum_position_error, maximum_rotation_error=maximum_rotation_error),
        protocol=order.protocol, source_sha256={name: sha(directory/name) for name in
            ('config.json', 'manifest.json', 'summary.json', 'moves.jsonl', 'trajectory.jsonl')})
    save(out/'runs'/job['id']/'analysis.json', result)
    print(json.dumps(dict(id=job['id'], passed=True, formations=result['motif_formations'],
        breakages=result['motif_breakages'], native_bonds=result['final_native']['bonds'],
        cpu_seconds=result['cpu_seconds'])), flush=True)
    return result


def report(out, results, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    arms = list(dict.fromkeys(r['variant'] for r in results))
    aggregates = {}
    for arm in arms:
        subset = [r for r in results if r['variant'] == arm]
        learned = Counter()
        for row in subset:
            learned.update(row['branch_counts'].get('learned', {}))
        mask_histogram = Counter()
        support = Counter()
        for row in subset:
            mask_histogram.update({int(k): v for k, v in row['atlas_mask']['count_histogram'].items()})
            support.update(row['reverse_support'])
        aggregates[arm] = dict(runs=len(subset), learned=dict(learned),
            cpu_seconds=sum(r['cpu_seconds'] for r in subset),
            motif_formations=sum(r['motif_formations'] for r in subset),
            motif_breakages=sum(r['motif_breakages'] for r in subset),
            native_bond_formations=sum(r['native_bond_formations'] for r in subset),
            native_bond_breakages=sum(r['native_bond_breakages'] for r in subset),
            native_registry_changes=sum(r['native_registry_changes'] for r in subset),
            final_native_bonds=[r['final_native']['bonds'] for r in subset],
            first_native_association_sweeps=[r['first_native_association_sweep'] for r in subset],
            mean_k=sum(k*v for k, v in mask_histogram.items())/sum(mask_histogram.values()),
            count_histogram=dict(mask_histogram), reverse_support=dict(support),
            reciprocal_pair_inclusions={key: dict(
                observed_count=sum(r['atlas_mask']['reciprocal_pair_inclusions'][key]['observed_count'] for r in subset),
                observed_fraction=sum(r['atlas_mask']['reciprocal_pair_inclusions'][key]['observed_count'] for r in subset)/sum(mask_histogram.values()),
                prior_probability=subset[0]['atlas_mask']['reciprocal_pair_inclusions'][key]['prior_probability'])
                for key in subset[0]['atlas_mask']['reciprocal_pair_inclusions']})
    save(out/'aggregates.json', aggregates)
    lines = ['# Active atlas mask pilot', '',
        f"{len(manifest['preparations'])} reused dispersed starts, {manifest['sweeps']} sweeps each. Every arm retains the same 28-component native-informed atlas and full Gaussian auxiliary state, with identical mean/covariance transport, initial poses, MC seeds, rd=1.5 Å, z=0.035 Å⁻³, local moves, spherical GCA, and center shifts. Masks compress active proposal evaluation only. Component parameters and current-geometry fits remain full size.", '',
        '| Arm | Mean active K | Gaussian accepted/attempted | Native candidates accepted/proposed | Native bonds formed/broken | Final native bonds by start | CPU s |',
        '|---|---:|---:|---:|---:|---|---:|']
    for arm, record in aggregates.items():
        c = record['learned']
        lines.append(f"| {arm} | {record['mean_k']:.3f} | {c.get('accepted', 0)}/{c.get('attempted', 0)} | {c.get('accepted_native_candidate', 0)}/{c.get('native_candidate', 0)} | {record['native_bond_formations']}/{record['native_bond_breakages']} | {record['final_native_bonds']} | {record['cpu_seconds']:.2f} |")
    lines += ['', 'The candidate classification includes every hard-valid global candidate, including rejected proposals. Native transitions are reconstructed after every accepted local, global, or GCA update, with the unchanged registered-pose and external residue-patch criterion. Bond formation and breakage count body pairs. A switch of native motif label on a continuously bonded pair is reported separately as a registry change.', '',
        'Registry changes by arm: '+str({k: v['native_registry_changes'] for k, v in aggregates.items()})+'. These are correlated proposal and transition counts, not independent trials.', '',
        'Every physical update and mask refresh is replayed to every saved frame and the final checkpoint. NumPy independently reconstructs current and candidate fits, both proposal densities with the same mask, and each refreshed mask probability. Saved frames independently pass atom-union hard-core and wall checks. The full-mask control must reproduce the previous covariance-arm physical configurations exactly. The supplied atlas contains native information in every arm; this tests retention of accessibility, not native discovery. Reusing these starts is a paired control, not four additional independent preparations. A short pilot cannot establish equilibrium, a mixing speedup, or crystal growth.', '',
        '| Arm | Full-atlas reverse support | Lost to uniform floor with mask | Native candidates with lost reverse support |',
        '|---|---:|---:|---:|']
    for arm, record in aggregates.items():
        support = record['reverse_support']
        lines.append(f"| {arm} | {support.get('full_atlas_supported', 0)} | {support.get('lost_to_uniform_floor', 0)} | {support.get('native_candidates_lost_to_uniform_floor', 0)} |")
    lines += ['', 'Reverse support is counted over hard-valid global proposals. “Full-atlas supported” means that Gaussians supply more than half of the reverse density using the candidate fit. “Lost to uniform floor” means that the corresponding masked reverse density is more than 99% uniform. These thresholds are diagnostics only.', '',
        '| Arm | Pair (14,19) observed/prior | Pair (21,22) observed/prior | Label 7 observed/prior |',
        '|---|---:|---:|---:|']
    for arm, record in aggregates.items():
        cells = [f"{v['observed_fraction']:.3f}/{v['prior_probability']:.3f}"
                 for v in record['reciprocal_pair_inclusions'].values()]
        lines.append('| '+arm+' | '+' | '.join(cells)+' |')
    lines += ['', 'The reciprocal label pairs are post-hoc diagnostics from the preceding atlas audit; they do not alter the mask law. Inclusion probabilities are calculated from the exact elementary-symmetric-polynomial law. Observations use post-refresh masks, excluding the deliberately full initialization when allowed.', '']
    (out/'report.md').write_text('\n'.join(lines))
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    x = np.arange(len(arms))
    for ax, field, label in zip(axes, ('native_bond_formations', 'native_registry_changes', 'cpu_seconds'),
                               ('Native bonds formed', 'Native registry changes', 'Sampler CPU seconds')):
        for i, arm in enumerate(arms):
            values = [r[field] for r in results if r['variant'] == arm]
            ax.plot([i]*len(values), values, 'o', alpha=.7)
            ax.plot(i, np.mean(values), '_', color='black', ms=18, mew=2)
        ax.set_xticks(x, arms, rotation=20)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.2)
    figure.suptitle('Same atlas and parameter transport; active-component compression')
    figure.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        figure.savefig(out/f'atlas-mask-comparison.{ext}', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--reference', type=Path, default=Path('/home/xvg/protein-nucleation'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    out = (args.out or campaign/'assessment').resolve()
    manifest = read(campaign/'manifest.json')
    assert read(campaign/'summary.json')['complete']
    tasks = [(str(args.reference), str(campaign), str(out), job) for job in manifest['jobs']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, tasks))
    save(out/'analysis.json', dict(passed=True, results=results, manifest=manifest, analyzer_sha256=sha(__file__)))
    report(out, results, manifest)
    print(json.dumps(dict(passed=True, out=str(out))), flush=True)


if __name__ == '__main__':
    main()
