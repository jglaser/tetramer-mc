#!/usr/bin/env python3
"""Independent docking replay, chart-map checks, and completed core roundtrips."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from analyze_free_tetramer_campaign import read, sha, save


class CorePassages:
    """Hysteretic core visits; two consecutive passages complete one roundtrip."""

    def __init__(self, native=.8, other=2.):
        self.native, self.other = native, other
        self.first = self.last = None
        self.visits, self.transitions, self.roundtrips = [], [], []

    def observe(self, q, cycle, attempt, cpu, branch, contact):
        label = ('native' if q <= self.native else 'other' if q >= self.other else None) if contact else None
        if label is None or label == self.last:
            return
        visit = dict(core=label, q=q, cycle=cycle, attempt=attempt, cpu_seconds=cpu, branch=branch)
        self.visits.append(visit)
        if self.first is None:
            self.first = label
        else:
            self.transitions.append(dict(source=self.last, target=label, **visit))
            if len(self.transitions) % 2 == 0:
                initial = self.visits[-3]
                self.roundtrips.append(dict(start_core=initial['core'],
                    first_cycle=initial['cycle'], last_cycle=cycle,
                    elapsed_cycles=cycle-initial['cycle'],
                    elapsed_cpu_seconds=cpu-initial['cpu_seconds']))
        self.last = label

    def result(self, cpu):
        return dict(native_core_q=self.native, other_core_q=self.other,
                    both_cores_require_depletion_contact=True,
                    visits=self.visits, transitions=self.transitions, roundtrips=self.roundtrips,
                    completed_roundtrips=len(self.roundtrips),
                    completed_roundtrips_per_cpu_second=len(self.roundtrips)/cpu,
                    native_to_other=sum(v['source'] == 'native' for v in self.transitions),
                    other_to_native=sum(v['source'] == 'other' for v in self.transitions))


class ChartAudit:
    def __init__(self, model):
        import numpy as np
        from prepare_smc_normalizer_atlas import unwrap_proposal_model, reciprocal_virtual_branches
        self.np = np
        self.proposal_model = model
        model, self.reciprocal_components = unwrap_proposal_model(model)
        self.base_model = model
        self.base_indices, self.inverted, self.weights = reciprocal_virtual_branches(model, self.reciprocal_components)
        self.mean = np.asarray(model['means'])[self.base_indices]
        covariance = np.asarray(model['covariances'])[self.base_indices]
        covariance = .5*(covariance+covariance.swapaxes(-1, -2))
        # An ill-conditioned atlas amplifies different valid Cholesky roundoff
        # into visibly different extreme-tail coordinates. Replay the specified
        # scalar factorization, and independently check its covariance residual.
        # Coordinate maps, rotations, densities and Jacobians below remain
        # independently reconstructed with NumPy/SciPy.
        self.lower = np.zeros_like(covariance)
        for component in range(len(covariance)):
            for i in range(6):
                for j in range(i+1):
                    remainder = float(covariance[component, i, j])
                    for k in range(j):
                        remainder -= float(self.lower[component, i, k])*float(self.lower[component, j, k])
                    self.lower[component, i, j] = np.sqrt(remainder) if i == j else remainder/self.lower[component, j, j]
        residual = np.abs(self.lower@self.lower.swapaxes(-1, -2)-covariance)
        self.cholesky_backward_error = float(np.max(residual)/np.max(np.abs(covariance)))
        assert self.cholesky_backward_error <= 1e-14
        self.condition_numbers = np.linalg.cond(covariance)
        self.logdet = np.log(np.diagonal(self.lower, axis1=1, axis2=2)).sum(axis=1)
        self.anchor_t = np.asarray([a['position'] for a in model['anchors']])[self.base_indices]
        self.anchor_r = np.asarray([a['rotation'] for a in model['anchors']])[self.base_indices]
        self.weights /= self.weights.sum()
        self.ell = model['angular_length']
        self.checks = Counter()
        self.max_errors = defaultdict(float)

    def arrays(self, pose):
        from scipy.spatial.transform import Rotation
        return self.np.asarray(pose['position']), Rotation.from_quat(self.np.asarray(pose['orientation'])[[1, 2, 3, 0]]).as_matrix()

    def pose(self, p, r):
        from scipy.spatial.transform import Rotation
        return dict(position=p.tolist(), orientation=Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]].tolist())

    def relative(self, pose, anchor):
        p, r = self.arrays(pose)
        a, ar = self.arrays(anchor)
        return self.pose(ar.T@(p-a), ar.T@r)

    def absolute(self, pose, anchor):
        p, r = self.arrays(pose)
        a, ar = self.arrays(anchor)
        return self.pose(a+ar@p, ar@r)

    def reciprocal(self, pose):
        p, r = self.arrays(pose)
        return self.pose(-r.T@p, r.T)

    def encode(self, label, pose):
        from scipy.spatial.transform import Rotation
        np = self.np
        p, r = self.arrays(pose)
        if self.inverted[label]:
            p, r = -r.T@p, r.T
        q = Rotation.from_matrix(r@self.anchor_r[label].T).as_quat()
        if q[3] == 0.:
            raise ValueError('Cayley half-turn seam')
        value = np.r_[p-self.anchor_t[label], self.ell*q[:3]/q[3]]
        return np.linalg.solve(self.lower[label], value-self.mean[label])

    def decode(self, label, z):
        from scipy.spatial.transform import Rotation
        np = self.np
        value = self.mean[label]+self.lower[label]@z
        u = value[3:]/self.ell
        quaternion = np.r_[u, 1.]
        quaternion /= np.linalg.norm(quaternion)
        rotation = Rotation.from_quat(quaternion).as_matrix()@self.anchor_r[label]
        volume = self.logdet[label]-3*np.log(self.ell)-2*np.log(np.pi)-2*np.log1p(u@u)
        position = value[:3]+self.anchor_t[label]
        if self.inverted[label]:
            position, rotation = -rotation.T@position, rotation.T
        return self.pose(position, rotation), float(volume)

    def gaussian(self, label, pose):
        np = self.np
        try:
            z = self.encode(label, pose)
        except ValueError:
            return float('-inf')
        _, volume = self.decode(label, z)
        return float(-3*np.log(2*np.pi)-.5*(z@z)-volume)

    def full_gaussian(self, pose):
        from scipy.special import logsumexp
        logs = [self.np.log(w)+self.gaussian(i, pose) for i, w in enumerate(self.weights)]
        return float(logsumexp(logs))

    def close(self, name, actual, recorded, atol=2e-6, rtol=2e-10):
        np = self.np
        error = float(np.max(np.abs(np.asarray(actual)-recorded)))
        magnitude = float(max(1., np.max(np.abs(actual)), np.max(np.abs(recorded))))
        assert error <= atol+rtol*magnitude, (name, error, magnitude, actual, recorded)
        self.max_errors[name] = max(self.max_errors[name], error)
        self.checks[name] += 1

    def compare_pose(self, actual, recorded, name='pose'):
        p, r = self.arrays(actual)
        ep, er = self.arrays(recorded)
        self.close(name+'_position', p, ep, atol=2e-7, rtol=2e-10)
        self.close(name+'_rotation', r, er, atol=2e-8, rtol=0.)

    def involution(self, old_relative, proposal, correlation):
        np = self.np
        trace, step = proposal['trace'], proposal['step']
        a, b, noise = trace['source'], trace['target'], np.asarray(trace['noise'])
        self.check_branch_metadata(proposal, a, b)
        encoded = self.encode(a, old_relative)
        z = np.asarray(step['source_latent'])
        # Inverting an almost-pi rotation amplifies tiny implementation-level
        # quaternion differences. Check the independent inverse with a stated
        # tail tolerance, then validate the supplied latent by stable decoding.
        # Cholesky forward error scales with covariance condition number even
        # away from the Cayley seam. Keep the stable decoded-pose audit strict.
        factor_tolerance = 64*np.finfo(float).eps*self.condition_numbers[a]
        tail = np.linalg.norm(z) >= 1e5
        self.close('source_latent_tail' if tail else 'source_latent', encoded, z,
                   rtol=max(factor_tolerance, 2e-7 if tail else 2e-10))
        reconstructed, old_volume = self.decode(a, z)
        self.compare_pose(reconstructed, old_relative, 'source_decode')
        sine = np.sqrt((1-correlation)*(1+correlation))
        target = correlation*z+sine*noise
        inverse_noise = sine*z-correlation*noise
        new_pose, new_volume = self.decode(b, target)
        jacobian = new_volume-old_volume
        auxiliary = .5*(noise@noise-inverse_noise@inverse_noise)
        self.close('target_latent', target, step['target_latent'])
        self.close('inverse_noise', inverse_noise, step['inverse_trace']['noise'])
        assert step['inverse_trace']['source'] == b and step['inverse_trace']['target'] == a
        self.compare_pose(new_pose, step['pose'], 'mapped_pose')
        self.close('extended_jacobian', jacobian, step['log_extended_jacobian'])
        self.close('auxiliary_ratio', auxiliary, step['log_auxiliary_ratio'])
        self.close('map_correction', jacobian+auxiliary, step['log_correction'])
        source_log = self.gaussian(a, old_relative)
        target_log = self.gaussian(b, new_pose)
        self.close('selected_source_density', source_log, proposal['selected_source_log_density'],
                   rtol=max(2*factor_tolerance, 2e-7 if abs(source_log) > 1e10 else 2e-10))
        self.close('selected_target_density', target_log, proposal['selected_target_log_density'],
                   rtol=max(128*np.finfo(float).eps*self.condition_numbers[b],
                            2e-7 if abs(target_log) > 1e10 else 2e-10))
        # Use the correlated latent trace to avoid subtracting two huge,
        # independently reconstructed log densities near a chart seam.
        self.close('selected_density_ratio', .5*(target@target-z@z)+jacobian,
                   step['log_correction'])
        if proposal.get('full_old_gaussian_log_density') is not None:
            self.close('full_old_gaussian', self.full_gaussian(old_relative), proposal['full_old_gaussian_log_density'])
            self.close('full_new_gaussian', self.full_gaussian(new_pose), proposal['full_new_gaussian_log_density'])
        return new_pose

    def check_branch_metadata(self, proposal, source, target):
        """Trace labels index virtual charts; optional metadata gives base IDs."""
        for label in (source, target):
            assert type(label) is int and 0 <= label < len(self.weights), 'Invalid virtual branch label'
        for role, label in (('source', source), ('target', target)):
            expected = {role+'_component_index': int(self.base_indices[label]), role+'_inverted': bool(self.inverted[label])}
            for name, value in expected.items():
                if self.inverted.any():
                    assert name in proposal and proposal[name] is not None, ('Missing reciprocal branch metadata', name)
                if proposal.get(name) is not None:
                    assert type(proposal[name]) is type(value) and proposal[name] == value, ('Incorrect reciprocal branch metadata', name)


def one(task):
    reference, campaign, out, job, rejected_tail_warnings = task
    reference, campaign, out = map(Path, (reference, campaign, out))
    sys.path.insert(0, str(reference/'scripts'))
    import numpy as np
    from audit_coordination_test import Coordinate
    from audit_tetramer_assembly import AtomicAssembly, pose_arrays

    manifest = read(campaign/'manifest.json')
    cfg = read(job['config'])
    directory = Path(job['directory'])
    summary = read(directory/'summary.json')
    assert summary['complete']
    assert sha(job['config']) == job['config_sha256']
    assert sha(campaign/'provenance/model.json') == manifest['model_sha256']
    assert sha(campaign/'provenance/docking-mc') == manifest['binary_sha256']
    input_hashes = dict(config=job['config_sha256'], model=manifest['model_sha256'], binary=manifest['binary_sha256'])
    source_hashes = {name: sha(directory/name) for name in ('moves.jsonl', 'trajectory.jsonl', 'summary.json', 'checkpoint.json')}
    analyzer_hash = sha(__file__)
    cached_path = out/'runs'/job['id']/'analysis.json'
    if cached_path.exists():
        cached = read(cached_path)
        if (cached.get('analyzer_sha256') == analyzer_hash and cached.get('input_sha256') == input_hashes
                and cached['source_sha256'] == source_hashes
                and (cached['full_map_audit_passed'] or rejected_tail_warnings)):
            print(json.dumps(dict(id=job['id'], passed=True, reused_verified_replay=True)), flush=True)
            return cached
    env = read(cfg['metadata']['environment'])
    coordinate = Coordinate(cfg['metadata'], env)
    atomic_cfg = dict(cfg, box_lengths=cfg['metadata']['source_periodic_box'], rigid_members=env['rigid_members'])
    atomic = AtomicAssembly(atomic_cfg, directory)
    # A nonzero periodic image is farther away than both exclusion bounds at
    # every center in the capture ball. Open and source-periodic interactions agree.
    fixed_distance = max(np.linalg.norm(np.asarray(p['position'])-cfg['capture_center']) for p in cfg['fixed_poses'])
    image_clearance = min(atomic.lengths)-fixed_distance-cfg['capture_radius']-2*(atomic.bound+atomic.rd)
    assert image_clearance > 0.
    audit = ChartAudit(read(campaign/'provenance/model.json'))
    frames = [json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
    assert [f['cycle'] for f in frames] == list(range(manifest['cycles']+1))
    state = copy.deepcopy(frames[0]['pose'])
    audit.compare_pose(state, cfg['initial_pose'], 'initial')
    q, _, distance = coordinate.evaluate(state)
    contact = frames[0]['depletion_contact']
    assert (q <= 1.) == (job['start'] == 'native')
    assert distance <= cfg['capture_radius']+1e-9
    core = CorePassages(other=2.)
    strict = CorePassages(other=5.)
    for tracker in (core, strict):
        tracker.observe(q, 0, -1, 0., 'initial', contact)
    initial_relative = audit.relative(state, cfg['fixed_poses'][0])
    initial_norms = np.asarray([np.linalg.norm(audit.encode(i, initial_relative)) for i in range(len(audit.weights))])
    from scipy.special import logsumexp
    component_logs = np.asarray([audit.gaussian(i, initial_relative) for i in range(len(audit.weights))])
    full_initial_log = float(logsumexp(np.log(audit.weights)+component_logs))
    initial_responsibilities = np.exp(np.log(audit.weights)+component_logs-full_initial_log)
    best_source = int(np.argmax(initial_responsibilities))
    initial_coverage = dict(full_gaussian_log_density=full_initial_log,
        maximum_responsibility_label=best_source,
        maximum_responsibility=float(initial_responsibilities[best_source]),
        maximum_responsibility_reference_weight=float(audit.weights[best_source]),
        source_probability_norm_le6=float(audit.weights[initial_norms <= 6].sum()),
        minimum_source_latent_norm=float(initial_norms.min()),
        source_latent_norms=initial_norms.tolist(), responsibilities=initial_responsibilities.tolist())
    exact_native = audit.relative(env['native_pose'], cfg['fixed_poses'][0])
    exact_logs = np.asarray([audit.gaussian(i, exact_native) for i in range(len(audit.weights))])
    exact_full = float(logsumexp(np.log(audit.weights)+exact_logs))
    exact_best = int(np.argmax(np.log(audit.weights)+exact_logs))
    initial_coverage['exact_native_reference'] = dict(full_gaussian_log_density=exact_full,
        maximum_responsibility_label=exact_best,
        maximum_responsibility=float(np.exp(np.log(audit.weights[exact_best])+exact_logs[exact_best]-exact_full)),
        source_latent_norm=float(np.linalg.norm(audit.encode(exact_best, exact_native))))
    counts = defaultdict(Counter)
    penalties = defaultdict(list)
    pair_counts = Counter()
    qrows = [dict(cycle=0, q=q, native=q <= 1., contact=contact, cpu_seconds=0.)]
    events, map_data, numerical_warnings, core_candidates = [], [], [], []
    native_last = q <= 1.
    current_cycle = 1
    attempts = 0
    atomic_cycles = set(np.linspace(0, manifest['cycles'], 8, dtype=int).tolist())
    for serial, line in enumerate((directory/'moves.jsonl').open()):
        move = json.loads(line)
        cycle = move['cycle']
        if cycle != current_cycle:
            assert attempts == cfg['local_attempts_per_cycle']+1
            audit.compare_pose(state, frames[current_cycle]['pose'], 'replay')
            assert contact == frames[current_cycle]['depletion_contact']
            qrows.append(dict(cycle=current_cycle, q=q, native=q <= 1., contact=contact,
                              cpu_seconds=frames[current_cycle]['sampler_cpu_seconds']))
            current_cycle, attempts = cycle, 0
        attempts += 1
        assert move['cycle'] == current_cycle
        audit.compare_pose(state, move['old_pose'], 'old_pose')
        assert move['depletion_contact_before'] == contact
        branch = move.get('branch', move['kind'])
        counter = counts[branch]
        counter['attempted'] += 1
        counter['accepted'] += int(move['accepted'])
        counter['hard_valid'] += int(move['hard_valid'])
        counter['capture_valid'] += int(move['capture_valid'])
        counter['capture_rejected'] += int(not move['capture_valid'] and move['proposed_pose'] is not None)
        counter['hard_rejected'] += int(move['capture_valid'] and not move['hard_valid'])
        counter['null'] += int(move['proposed_pose'] is None)
        proposal = move.get('proposal') or {}
        if proposal.get('trace') is not None:
            trace = proposal['trace']
            pair_counts[f"{trace['source']},{trace['target']}"] += 1
        candidate = move['proposed_pose']
        if candidate is not None:
            candidate_q, _, candidate_distance = coordinate.evaluate(candidate)
            assert move['capture_valid'] == (candidate_distance <= cfg['capture_radius'])
            if move['kind'] == 'global' and contact and ((q <= .8 and candidate_q >= 2.) or (q >= 2. and candidate_q <= .8)):
                core_candidates.append(dict(cycle=cycle, branch=branch, old_q=q, candidate_q=candidate_q,
                    source_core='native' if q <= .8 else 'other', old_depletion_contact=True,
                    candidate_depletion_contact_if_accepted=move['depletion_contact'] if move['accepted'] else None,
                    capture_valid=move['capture_valid'], hard_valid=move['hard_valid'], accepted=move['accepted'],
                    log_correction=proposal.get('log_reverse_forward', 0.),
                    poisson_log_factor=(move.get('gate') or {}).get('log_weight'),
                    log_acceptance=move.get('log_acceptance'),
                    source_label=(proposal.get('trace') or {}).get('source'),
                    target_label=(proposal.get('trace') or {}).get('target')))
            if move['capture_valid'] and move['hard_valid']:
                counter['valid_native_candidate'] += int(candidate_q <= 1.)
                counter['accepted_native_candidate'] += int(candidate_q <= 1. and move['accepted'])
            if proposal.get('step') is not None:
                anchor = cfg['fixed_poses'][proposal.get('anchor_index', 0)]
                old_relative = audit.relative(state, anchor)
                try:
                    calculated = audit.involution(old_relative, proposal, job['correlation'])
                except AssertionError as error:
                    # Preserve a failed strict audit explicitly for an old
                    # mismatch smoke; never waive an accepted/valid map check.
                    # The subsequent physical replay and target decode still
                    # run. Full campaigns require the default strict mode.
                    assert rejected_tail_warnings and not move['accepted'] and not move['hard_valid']
                    assert error.args[0][0] == 'source_decode_rotation'
                    assert np.linalg.norm(proposal['step']['source_latent']) >= 1e5
                    numerical_warnings.append(dict(cycle=cycle, branch=branch,
                        source=proposal['trace']['source'], target=proposal['trace']['target'],
                        rejected=True, hard_valid=False, check=error.args[0][0],
                        maximum_rotation_matrix_error=float(error.args[0][1])))
                    calculated, _ = audit.decode(proposal['trace']['target'], proposal['step']['target_latent'])
                audit.compare_pose(audit.absolute(calculated, anchor), candidate, 'candidate_lab')
                trace, step = proposal['trace'], proposal['step']
                for key in ('log_extended_jacobian', 'log_auxiliary_ratio', 'log_correction'):
                    penalties[key].append(step[key])
                map_data.append(dict(cycle=cycle, source=trace['source'], target=trace['target'],
                    accepted=move['accepted'], hard_valid=move['hard_valid'], capture_valid=move['capture_valid'],
                    source_latent_norm=float(np.linalg.norm(step['source_latent'])),
                    target_latent_norm=float(np.linalg.norm(step['target_latent'])),
                    correction=step['log_correction'], candidate_q=candidate_q))
            elif job['method'] == 'mixture' and 'old_log_density' in proposal:
                anchor = cfg['fixed_poses'][proposal.get('anchor_index', 0)]
                epsilon = cfg['uniform_probability']
                def complete_density(pose):
                    uniform = (np.log(epsilon)-3*np.log(2*cfg['capture_radius'])
                        if np.all(np.abs(np.asarray(pose['position'])-cfg['capture_center']) < cfg['capture_radius'])
                        else float('-inf'))
                    gaussian = np.log1p(-epsilon)+audit.full_gaussian(audit.relative(pose, anchor))
                    return float(np.logaddexp(uniform, gaussian))
                old_log, new_log = complete_density(state), complete_density(candidate)
                audit.close('complete_old_density', old_log, proposal['old_log_density'])
                audit.close('complete_new_density', new_log, proposal['new_log_density'])
                audit.close('complete_density_ratio', old_log-new_log, proposal['log_reverse_forward'])
            if move.get('gate') is not None:
                counter['gate_evaluations'] += 1
                counter['gate_rejected'] += int(not move['accepted'])
                correction = proposal.get('log_reverse_forward', 0.)
                expected = min(0., correction+move['gate']['log_weight'])
                audit.close('acceptance', expected, move['log_acceptance'])
                penalties['gate_log_weight'].append(move['gate']['log_weight'])
                penalties['valid_log_correction'].append(correction)
                if move['accepted']:
                    penalties['accepted_gate_log_weight'].append(move['gate']['log_weight'])
                    penalties['accepted_log_correction'].append(correction)
        if move['accepted']:
            assert move['hard_valid'] and move['capture_valid'] and candidate is not None
            pold, rold = audit.arrays(state)
            pnew, rnew = audit.arrays(candidate)
            changed = np.linalg.norm(pnew-pold) > 1e-10 or np.max(np.abs(rnew-rold)) > 1e-12
            counter['accepted_pose_changes'] += int(changed)
            counter['accepted_identity'] += int(not changed)
            if proposal.get('trace') is not None:
                label_change = proposal['trace']['source'] != proposal['trace']['target']
                counter['accepted_label_changes'] += int(changed and label_change)
                counter['accepted_same_label_changes'] += int(changed and not label_change)
            state = copy.deepcopy(move['retained_pose'])
            audit.compare_pose(state, candidate, 'accepted')
            q, _, distance = coordinate.evaluate(state)
            assert distance <= cfg['capture_radius']+1e-9
        else:
            audit.compare_pose(state, move['retained_pose'], 'rejected')
        contact = move['depletion_contact']
        if (q <= 1.) != native_last:
            events.append(dict(cycle=cycle, attempt=move['attempt'], source='native' if native_last else 'other',
                               target='native' if q <= 1. else 'other', q=q, branch=branch))
            native_last = q <= 1.
        for tracker in (core, strict):
            tracker.observe(q, cycle, move['attempt'], move['sampler_cpu_seconds'], branch, contact)
    assert attempts == cfg['local_attempts_per_cycle']+1 and current_cycle == manifest['cycles']
    audit.compare_pose(state, frames[current_cycle]['pose'], 'replay')
    assert contact == frames[current_cycle]['depletion_contact']
    qrows.append(dict(cycle=current_cycle, q=q, native=q <= 1., contact=contact,
                     cpu_seconds=frames[current_cycle]['sampler_cpu_seconds']))
    checkpoint = read(directory/'checkpoint.json')
    audit.compare_pose(state, checkpoint['pose'], 'checkpoint')
    for index, branches in ((0, ['local']), (1, [b for b in counts if b != 'local'])):
        for key in ('attempted', 'accepted', 'accepted_pose_changes'):
            assert sum(counts[b][key] for b in branches) == summary['counts'][index][key]
    for tracker in (core, strict):
        atomic_cycles.update(v['cycle'] for v in tracker.visits[:6])
    atomic_rows = []
    for cycle in sorted(atomic_cycles):
        p, qv = pose_arrays({'poses': [frames[cycle]['pose']]+cfg['fixed_poses']})
        checked = atomic.frame(p, qv)
        assert checked['hard_valid'], (job['id'], cycle, checked)
        expected_contact = bool(checked['depletion_edges'])
        assert expected_contact == frames[cycle]['depletion_contact'], (job['id'], cycle, 'contact mismatch', checked)
        atomic_rows.append(dict(cycle=cycle, hard_valid=True, minimum_gap_A=checked['minimum_interbody_gap_A']))
    cpu = summary['sampler_cpu_seconds']
    half = manifest['cycles']//2
    result = dict(id=job['id'], site=job['site'], start=job['start'], replicate=job['replicate'], mode=job['mode'],
        passed=True, full_map_audit_passed=not numerical_warnings,
        cpu_seconds=cpu, counts={k: dict(v) for k, v in counts.items()}, cost=summary.get('cost'),
        initial_q=qrows[0]['q'], final_q=qrows[-1]['q'],
        initial_atlas_coverage=initial_coverage,
        unbound_fraction=float(np.mean([not r['contact'] for r in qrows[1:]])),
        first_half_native_fraction=float(np.mean([r['native'] for r in qrows[1:half+1]])),
        last_half_native_fraction=float(np.mean([r['native'] for r in qrows[half+1:]])),
        core=core.result(cpu), strict_core=strict.result(cpu), raw_boundary_crossings=events,
        rows=qrows, pair_counts=dict(pair_counts), map_diagnostics=map_data,
        core_candidate_diagnostics=core_candidates,
        penalty_quantiles={k: dict(zip(('q10', 'median', 'q90'), np.quantile(v, [.1, .5, .9]).tolist()))
                           for k, v in penalties.items()},
        independent_audit=dict(checks=dict(audit.checks), maximum_errors=dict(audit.max_errors),
            cholesky_relative_backward_error=audit.cholesky_backward_error,
            covariance_condition_numbers=audit.condition_numbers.tolist(), numerical_warnings=numerical_warnings,
            atomic_frames=atomic_rows, periodic_image_exclusion_clearance_A=float(image_clearance)),
        source_sha256=source_hashes, input_sha256=input_hashes, analyzer_sha256=analyzer_hash)
    save(out/'runs'/job['id']/'analysis.json', result)
    print(json.dumps(dict(id=job['id'], passed=True, roundtrips=result['core']['completed_roundtrips'],
                         strict_roundtrips=result['strict_core']['completed_roundtrips'], cpu_seconds=cpu)), flush=True)
    return result


def report(out, results, manifest):
    import numpy as np
    modes = manifest['modes']
    groups = []
    for site in manifest['sites']:
        for start in manifest['starts']:
            for mode in modes:
                rows = [r for r in results if (r['site'], r['start'], r['mode']) == (site, start, mode)]
                cpu = sum(r['cpu_seconds'] for r in rows)
                completed = sum(r['core']['completed_roundtrips'] for r in rows)
                groups.append(dict(site=site, start=start, mode=mode, trajectories=len(rows), cpu_seconds=cpu,
                    completed_roundtrips=completed, completed_roundtrips_per_cpu_second=completed/cpu,
                    strict_roundtrips=sum(r['strict_core']['completed_roundtrips'] for r in rows),
                    native_to_other=sum(r['core']['native_to_other'] for r in rows),
                    other_to_native=sum(r['core']['other_to_native'] for r in rows),
                    first_half_native=[r['first_half_native_fraction'] for r in rows],
                    last_half_native=[r['last_half_native_fraction'] for r in rows]))
    totals = []
    for mode in modes:
        rows = [r for r in results if r['mode'] == mode]
        counter = Counter()
        for row in rows:
            for branch, count in row['counts'].items():
                if branch != 'local':
                    counter.update(count)
        cpu = sum(r['cpu_seconds'] for r in rows)
        totals.append(dict(mode=mode, cpu_seconds=cpu, global_counts=dict(counter),
            useful_globals_per_cpu_second=counter['accepted_pose_changes']/cpu,
            native_to_other=sum(r['core']['native_to_other'] for r in rows),
            other_to_native=sum(r['core']['other_to_native'] for r in rows),
            completed_roundtrips=sum(r['core']['completed_roundtrips'] for r in rows),
            strict_roundtrips=sum(r['strict_core']['completed_roundtrips'] for r in rows),
            unbound_fraction=float(np.mean([r['unbound_fraction'] for r in rows])),
            competing_start_native_entries=sum(r['start']=='other' and r['core']['other_to_native']>0 for r in rows),
            maximum_map_source_norm=max((m['source_latent_norm'] for r in rows for m in r['map_diagnostics']), default=None)))
    save(out/'groups.json', groups)
    save(out/'totals.json', totals)
    candidate_groups = []
    for mode in modes:
        for cutoff in (2., 5.):
            for source_core in ('native', 'other'):
                rows = [v for r in results if r['mode']==mode for v in r['core_candidate_diagnostics']
                        if v['source_core']==source_core and
                        (v['candidate_q'] if source_core=='native' else v['old_q']) >= cutoff]
                valid = [v for v in rows if v['capture_valid'] and v['hard_valid']]
                gated = [v for v in valid if v['poisson_log_factor'] is not None]
                quantiles = {}
                for key in ('log_correction', 'poisson_log_factor', 'log_acceptance'):
                    values = [v[key] for v in gated if v[key] is not None]
                    quantiles[key] = dict(zip(('q10','median','q90'),np.quantile(values,[.1,.5,.9]).tolist())) if values else None
                candidate_groups.append(dict(mode=mode, source_core=source_core, other_q_cutoff=cutoff,
                    proposed=len(rows), hard_and_capture_valid=len(valid), gate_evaluated=len(gated),
                    accepted=sum(v['accepted'] for v in rows),
                    accepted_bound_target=sum(v['accepted'] and v['candidate_depletion_contact_if_accepted'] for v in rows),
                    quantiles=quantiles))
    save(out/'core-candidate-diagnostics.json',candidate_groups)
    conditional = manifest.get('conditional_coverage_control', False)
    initial_description = ('Two selected native SMC endpoints and two previously inspected competing-basin snapshots are retained as separate starts. This is a retrospective coverage control, not a blind test or newly learned atlas.' if conditional else
        'Selected native and competing SMC endpoints remain separate starts. This assembly atlas is being tested outside its fitted conditional environment.')
    component_count = manifest.get('component_count', 28)
    lines = ['# Conditional docking with involutive proposals', '',
        f"{len(results)} runs, {manifest['cycles']} cycles each. One mobile rigid tetramer remains in the fixed 18 Å capture sphere with one fixed neighbor, rd=1.5 Å and z=0.035 Å⁻³. All modes use the same frozen {component_count}-component atlas and two local attempts plus one global attempt per cycle. {initial_description}", '',
        '| Mode | Global attempts | Useful global accepts | Changed labels | Identity accepts | N→O / O→N | Completed roundtrips | CPU s | Useful globals / CPU s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for t in totals:
        c = t['global_counts']
        changed_labels = str(c.get('accepted_label_changes',0)) if t['mode'] != 'mixture' else 'n/a'
        lines.append(f"| {t['mode']} | {c['attempted']} | {c.get('accepted_pose_changes',0)} | {changed_labels} | {c.get('accepted_identity',0)} | {t['native_to_other']} / {t['other_to_native']} | {t['completed_roundtrips']} | {t['cpu_seconds']:.2f} | {t['useful_globals_per_cpu_second']:.4f} |")
    lines += ['', 'Useful means a changed physical pose: displacement >10⁻¹⁰ Å or maximum rotation-matrix change >10⁻¹². In particular, c=1 with identical source and target labels is an exact identity and is counted separately. Useful acceptance need not change a contact environment.', '',
        '| Mode | Capture rejected | Hard rejected after capture | Gate evaluated | Gate rejected | Unbound frame fraction |',
        '|---|---:|---:|---:|---:|---:|']
    for t in totals:
        c = t['global_counts']
        lines.append(f"| {t['mode']} | {c.get('capture_rejected',0)} | {c.get('hard_rejected',0)} | {c.get('gate_evaluations',0)} | {c.get('gate_rejected',0)} | {t['unbound_fraction']:.4f} |")
    lines += ['', '| Site | Initial arm | Mode | N→O / O→N | Completed roundtrips | Strict roundtrips | CPU s | Roundtrips / CPU s |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['site']} | {g['start']} | {g['mode']} | {g['native_to_other']} / {g['other_to_native']} | {g['completed_roundtrips']} | {g['strict_roundtrips']} | {g['cpu_seconds']:.2f} | {g['completed_roundtrips_per_cpu_second']:.5f} |")
    lines += ['', 'A primary passage connects the native core q≤0.8 with the competing core q≥2. Both endpoints require exact overlap of the depletant-inflated atomic sphere unions. Unbound transit is allowed between core visits and its occupancy is reported separately. Two consecutive passages returning to the first visited core form one nonoverlapping completed roundtrip. The stricter diagnostic uses q≥5. These are registration and contact criteria, not independently identified thermodynamic basins. Every attempted move is replayed, so visits between saved cycle frames are included.', '',
        '| Mode | Proposed N→O | Hard/capture valid | Gate evaluated | Accepted | Median proposal log ratio | Median Poisson log factor |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in candidate_groups:
        if g['source_core']!='native' or g['other_q_cutoff']!=2.:
            continue
        medians = [('—' if g['quantiles'][key] is None else f"{g['quantiles'][key]['median']:.2f}")
                   for key in ('log_correction','poisson_log_factor')]
        lines.append(f"| {g['mode']} | {g['proposed']} | {g['hard_and_capture_valid']} | {g['gate_evaluated']} | {g['accepted']} | {medians[0]} | {medians[1]} |")
    lines += ['', 'These candidate diagnostics require the old native core to be bound, but use q≥2 alone as the competing-candidate proxy: contact is not recomputed for every rejected proposal. Actual accepted core passages above require contact at both ends. Medians condition on hard/capture validity and an evaluated gate. The Poisson log factor is a sampled acceptance factor, not a measured free-energy difference. The machine-readable diagnostic includes both directions, q≥5 controls, and 10/50/90% quantiles.', '',
        '| Site | Start | Replica | Initial q | Full atlas log density | Dominant label | Its reference weight | Static source probability ‖z‖≤6 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in results:
        if row['mode'] != modes[0]:
            continue
        c = row['initial_atlas_coverage']
        lines.append(f"| {row['site']} | {row['start']} | {row['replicate']} | {row['initial_q']:.3f} | {c['full_gaussian_log_density']:.3f} | {c['maximum_responsibility_label']} | {c['maximum_responsibility_reference_weight']:.4f} | {c['source_probability_norm_le6']:.4f} |")
    lines += ['', 'The static source label is sampled from the reference weights, not from responsibilities conditioned on the old pose. Thus a well-covered pose can still receive an unsuitable source label. All five Gaussian charts, including the broad component, are retained in the conditional control; the separate uniform pose branch has probability 0.1. At c=0, the selected-component auxiliary ratio differs from the full-mixture independence ratio, so those arms are distinct controls.', '',
        'The audit reconstructs all retained physical moves, latent/noise transformations, selected and full Gaussian densities, extended Haar Jacobians, and the gate/proposal acceptance decomposition. It separately reconstructs the complete mixture proposal density and cross-checks useful move counts against the runner. Selected frames receive independent atom-union hard and depletion-contact checks. A geometric clearance bound rules out interactions with omitted remote periodic images throughout the capture domain.', '',
        'Scalar Cholesky replay is checked by its covariance residual; independent inverse-coordinate and selected-density tolerances account for covariance conditioning and the Cayley half-turn seam. Physical candidate and retained-pose checks remain separate. The report does not treat floating-point agreement as a proof of equilibrium.', '']
    warnings = [dict(run=r['id'], **w) for r in results for w in r['independent_audit']['numerical_warnings']]
    if warnings:
        lines += [f"**Incomplete strict map audit for {len(warnings)} rejected, hard-invalid extreme-tail traces.** Their reconstructed source rotations exceeded the strict tolerance. These traces were not accepted; physical replay and candidate reconstruction still pass. They are explicitly recorded in analysis.json, and this campaign does not receive a blanket numerical-map validation claim.", '']
    lines += ['Initialization is not an equilibrium sample within either starting basin. Counts describe finite-horizon algorithmic MC trajectories. First native entry, a complete return, and opposite-start agreement are different diagnostics. No zero-denominator speedup ratio is reported, and no observed event count establishes physical attachment kinetics or equilibrium.', '']
    (out/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(modes, plt.get_cmap('tab10').colors))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(modes))
    ax = axes[0, 0]
    ax.bar(x, [t['useful_globals_per_cpu_second'] for t in totals], color=[colors[m] for m in modes], alpha=.65)
    for i, mode in enumerate(modes):
        values = [sum(c.get('accepted_pose_changes',0) for b,c in r['counts'].items() if b!='local')/r['cpu_seconds']
                  for r in results if r['mode']==mode]
        ax.plot(np.full(len(values), i), values, 'o', ms=4, color='black', alpha=.65)
    ax.set_xticks(x, modes)
    ax.set_ylabel('Nonidentity global accepts / CPU s')
    ax.set_title('Accepted physical proposals')
    ax = axes[0, 1]
    for offset, key, label, color in ((-.24,'native_to_other','Native → competing','#486ea1'),
            (0,'other_to_native','Competing → native','#d4872b'),
            (.24,'completed_roundtrips','Complete roundtrip','#4c956c')):
        ax.bar(x+offset, [t[key]/t['cpu_seconds'] for t in totals], width=.23, label=label, color=color)
    ax.set_xticks(x, modes)
    ax.set_ylabel('Bound core passages / CPU s')
    ax.set_title('Primary core passages (q≤0.8 ↔ q≥2)', pad=27)
    strict_exits = sum(r['strict_core']['native_to_other'] for r in results)
    roundtrips = sum(r['core']['completed_roundtrips'] for r in results)
    annotation = ('No native→q≥5 exits; no completed returns' if strict_exits==0 and roundtrips==0 else
                  f'Strict native→q≥5 exits: {strict_exits}; completed returns: {roundtrips}')
    ax.text(.5, 1.025, annotation, transform=ax.transAxes, ha='center', fontsize=9)
    ax.legend(fontsize=8)
    site = manifest['sites'][0]
    for index, start in enumerate(('native','other')):
        ax = axes[1,index]
        for mode in modes:
            row = next((r for r in results if (r['site'],r['start'],r['replicate'],r['mode'])==(site,start,0,mode)), None)
            if row is None:
                continue
            ax.plot([v['cycle'] for v in row['rows']], [max(v['q'],1e-3) for v in row['rows']],
                    color=colors[mode], label=mode, lw=1.1, alpha=.85)
        ax.axhline(.8,color='#4c956c',ls='--',lw=.9)
        ax.axhline(2.,color='#d4872b',ls='--',lw=.9)
        if index == 1:
            ax.axhline(5.,color='#727272',ls=':',lw=1.)
            ax.text(.015, 5.15, 'Strict competing core: q≥5', transform=ax.get_yaxis_transform(),
                    fontsize=8, color='#555555')
        ax.set_yscale('log')
        ax.set_xlabel('MC cycle')
        ax.set_ylabel('Native registration error q')
        ax.set_title(f'{site}, {start} start, replica0')
        ax.legend(fontsize=8, ncol=3)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.18)
    figure.suptitle(f'Frozen {component_count}-component atlas: {manifest["cycles"]} cycles per start and method')
    figure.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        figure.savefig(out/f'involution-docking.{ext}', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--reference', type=Path, default=Path('/home/xvg/protein-nucleation'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--record-rejected-tail-warnings', action='store_true',
        help='Record an incomplete strict map audit for rejected invalid source tail reconstructions')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    out = (args.out or campaign/'assessment').resolve()
    manifest = read(campaign/'manifest.json')
    assert read(campaign/'summary.json')['complete']
    tasks = [(str(args.reference), str(campaign), str(out), job, args.record_rejected_tail_warnings)
             for job in manifest['jobs']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, tasks))
    save(out/'analysis.json', dict(passed=True, physical_replay_passed=True,
        full_map_audit_passed=all(r['full_map_audit_passed'] for r in results),
        manifest=manifest, results=results, analyzer_sha256=sha(__file__)))
    report(out, results, manifest)
    print(json.dumps(dict(passed=True, out=str(out))), flush=True)


if __name__ == '__main__':
    main()
