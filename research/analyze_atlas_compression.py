#!/usr/bin/env python3
"""Offline, native-blind compression audit of the fixed reference atlas.

Selection uses saved, geometrically classified contact poses, never native labels.
Native candidates and formations are post-hoc diagnostics. All densities include
every component, its Cayley-to-Haar Jacobian, and the unchanged uniform floor.
Transported runs are evaluated against the reference atlas, not an approximate
reconstruction of their changing proposals. This is support diagnosis, not a
counterfactual simulation or an equilibrium estimate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp


def read(path):
    return json.loads(Path(path).read_text())


def matrices(poses):
    return Rotation.from_quat(np.asarray([p['orientation'] for p in poses])[:, [1, 2, 3, 0]]).as_matrix()


def relative(moving, anchor):
    r = matrices([moving, anchor])
    return r[1].T @ (np.asarray(moving['position'])-anchor['position']), r[1].T @ r[0]


class Atlas:
    def __init__(self, model, config, shape):
        self.model = model
        self.w = np.asarray(model['weights']); self.w /= self.w.sum()
        self.means = np.asarray(model['means'])
        self.cov = np.asarray(model['covariances'])
        self.lower = np.linalg.cholesky(self.cov)
        self.inv_lower = np.linalg.inv(self.lower)
        self.logdet = np.log(np.diagonal(self.lower, axis1=1, axis2=2)).sum(axis=1)*2
        self.anchors = model['anchors']
        self.angular = model['angular_length']
        self.eps = config['learned_uniform_weight']
        bound = max(np.linalg.norm(a['center'])+a['radius'] for a in shape['atoms'])
        self.logu = np.log(self.eps)-3*np.log(2*(config['boundary']['radius']+bound))

    def log_components(self, poses):
        t = np.asarray([p[0] for p in poses]); r = np.asarray([p[1] for p in poses])
        out = np.full((len(poses), len(self.w)), -np.inf)
        for k, anchor in enumerate(self.anchors):
            q = Rotation.from_matrix(r @ np.asarray(anchor['rotation']).T).as_quat()
            valid = abs(q[:, 3]) > 1e-15
            c = q[valid, :3]/q[valid, 3, None]
            x = np.concatenate((t[valid]-anchor['position'], self.angular*c), axis=1)-self.means[k]
            y = x @ self.inv_lower[k].T
            log_jac = -2*np.log(np.pi)-2*np.log1p((c*c).sum(axis=1))
            out[valid, k] = (np.log(self.w[k])-.5*(6*np.log(2*np.pi)+self.logdet[k]+(y*y).sum(axis=1))
                             +3*np.log(self.angular)-log_jac)
        return out

    def log_density(self, lc, labels):
        labels = list(labels)
        if not labels:
            # Empty model's global branch is entirely uniform.
            return np.full(len(lc), self.logu-np.log(self.eps))
        return np.logaddexp(self.logu, np.log1p(-self.eps)+logsumexp(lc[:, labels], axis=1)
                           -np.log(self.w[labels].sum()))


def collect(campaign, atlas):
    contact, native, formation, accepted, metadata, replay = [], [], [], [], [], []
    for directory in sorted((campaign/'runs').glob('free-r*-*')):
        name = directory.name; arm = name.split('-')[-1]; replicate = int(name.split('-')[1][1:])
        audit = read(campaign/'assessment'/'runs'/name/'analysis.json')
        frames = [json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
        frame_map = {f['sweep']: f for f in frames}
        audit_rows = {r['sweep']: r for r in audit['rows']}
        base = dict(run=name, arm=arm, replicate=replicate)
        for frame in frames[1:]:
            row = audit_rows[frame['sweep']]
            native_edges = {tuple(e) for e in row['native']['edges']}
            for i, j in row['nonspecific']['edges']:
                for a, b in ((i, j), (j, i)):
                    contact.append(dict(**base, sweep=frame['sweep'], moving=a, anchor=b,
                                        native=tuple(sorted((a, b))) in native_edges,
                                        pose=relative(frame['poses'][a], frame['poses'][b])))
        native_map = {(p['sweep'], p['moving_index']): p for p in audit['native_proposals']}
        changes = {c['serial']: c for c in audit['changes']}
        state = copy.deepcopy(frames[0]['poses']); sweep = 0
        active_native_edges = set()
        counts = [Counter() for _ in atlas.w]
        log_density_errors = []
        for serial, line in enumerate((directory/'moves.jsonl').open()):
            move = json.loads(line)
            if move['sweep'] != sweep:
                if sweep in frame_map:
                    target = frame_map[sweep]['poses']
                    assert np.max(abs(np.asarray([p['position'] for p in state])-np.asarray([p['position'] for p in target]))) < 1e-7
                    assert np.max(abs(matrices(state)-matrices(target))) < 1e-8
                    state = copy.deepcopy(target)
                sweep = move['sweep']
            kind = move['kind']; proposal = move.get('proposal') or {}
            k = proposal.get('component_index')
            if kind == 'global' and k is not None:
                count = counts[k]
                count['attempted'] += 1
                count['hard_valid'] += int(move['hard_valid'])
                count['accepted'] += int(move['accepted'])
                candidate = move.get('proposed_pose')
                if candidate:
                    item = dict(**base, sweep=sweep, serial=serial, component=k, accepted=move['accepted'],
                                pose=relative(candidate, state[proposal['anchor_index']]),
                                old_native_bonded=any(move['moving_index'] in edge for edge in active_native_edges),
                                log_reverse_forward=proposal.get('log_reverse_forward'),
                                gate_log_weight=(move.get('gate') or {}).get('log_weight'),
                                log_acceptance=move.get('log_acceptance'))
                    if arm == 'frozen' and move['hard_valid']:
                        predicted = atlas.log_density(atlas.log_components([item['pose']]), range(len(atlas.w)))[0]
                        log_density_errors.append(abs(predicted-proposal['new_log_density']))
                    if (sweep, move['moving_index']) in native_map:
                        count['native_candidate'] += 1
                        count['accepted_native_candidate'] += int(move['accepted'])
                        native.append(item)
                    if move['accepted']:
                        accepted.append(item)
                    change = changes.get(serial)
                    if change and change['formed_body_pairs']:
                        count['native_bonds_formed'] += len(change['formed_body_pairs'])
                        formation.append(dict(**item, bonds=len(change['formed_body_pairs'])))
            if kind in ('local', 'global') and move['accepted']:
                state[move['moving_index']] = copy.deepcopy(move['retained_pose'])
            elif kind == 'gca' and move.get('accepted', True):
                axis = np.asarray(move['axis']); axis /= np.linalg.norm(axis)
                transform = 2*np.outer(axis, axis)-np.eye(3)
                for i in move['result']['flipped_indices']:
                    r = transform @ matrices([state[i]])[0]
                    state[i] = dict(position=(transform @ state[i]['position']).tolist(),
                                    orientation=Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]].tolist())
            elif kind == 'center_shift' and move.get('accepted', True):
                for pose in state:
                    pose['position'] = (np.asarray(pose['position'])+move['result']['displacement']).tolist()
            if serial in changes:
                active_native_edges.difference_update(tuple(p) for p in changes[serial]['broken_body_pairs'])
                active_native_edges.update(tuple(p) for p in changes[serial]['formed_body_pairs'])
        error = max(log_density_errors, default=0.)
        assert error < 1e-5, (name, error)
        metadata.append(dict(**base, counts=[dict(c) for c in counts]))
        replay.append(dict(run=name, maximum_reference_log_density_error=error))
    return contact, native, formation, accepted, metadata, replay


def greedy_path(atlas, lc):
    """Backward maximum empirical likelihood. Native labels never enter."""
    active = list(range(len(atlas.w))); path = {len(active): active.copy()}
    while len(active) > 1:
        options = [(float(np.mean(atlas.log_density(lc, [j for j in active if j != k]))), k) for k in active]
        _, remove = max(options, key=lambda pair: (pair[0], -pair[1]))
        active.remove(remove); path[len(active)] = active.copy()
    return path


def diagnose(args):
    campaign = Path(args.campaign).resolve(); out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    model = read(campaign/'provenance/model.json')
    cfg = read(campaign/'runs/free-r00-frozen/config.json')
    atlas = Atlas(model, cfg, read(campaign/'provenance/shape.json'))
    contact, native, formation, accepted, metadata, replay = collect(campaign, atlas)
    pools = {'contacts': contact, 'native_candidates': native, 'native_formations': formation, 'accepted_gaussian': accepted}
    lc = {key: atlas.log_components([r['pose'] for r in rows]) for key, rows in pools.items()}
    full = {key: atlas.log_density(values, range(len(atlas.w))) for key, values in lc.items()}
    responsibility = {key: np.exp(values-logsumexp(values, axis=1)[:, None]) for key, values in lc.items()}
    reciprocal = {}
    modes = responsibility['contacts'].argmax(axis=1)
    for arm in ('frozen', 'covariance'):
        counts = Counter()
        for i in range(0, len(contact), 2):
            assert contact[i]['moving'] == contact[i+1]['anchor']
            assert contact[i]['anchor'] == contact[i+1]['moving']
            if contact[i]['arm'] == arm:
                counts[tuple(sorted((int(modes[i]), int(modes[i+1]))))] += 1
        reciprocal[arm] = [dict(components=list(pair), saved_pair_observations=count)
                           for pair, count in counts.most_common()]
    records = []
    for k, meta in enumerate(model['components']):
        row = dict(component=k, weight=float(atlas.w[k]), defensive=meta['defensive'], chart=meta['chart_name'],
                   translation_std_A=np.sqrt(np.diag(atlas.cov[k])[:3]).tolist(), arms={})
        for arm in ('frozen', 'means', 'covariance', 'weights'):
            counts = Counter()
            for run in metadata:
                if run['arm'] == arm: counts.update(run['counts'][k])
            indices = [i for i, r in enumerate(contact) if r['arm'] == arm]
            row['arms'][arm] = dict(**counts, contact_responsibility=float(responsibility['contacts'][indices, k].sum()))
            candidates = [r for r in native if r['arm'] == arm and r['component'] == k]
            row['arms'][arm]['native_candidate_diagnostics'] = {}
            for bonded in (False, True):
                subset = [r for r in candidates if r['old_native_bonded'] == bonded]
                row['arms'][arm]['native_candidate_diagnostics'][str(bonded).lower()] = dict(
                    old_native_bonded=bonded, proposed=len(subset), accepted=sum(r['accepted'] for r in subset),
                    **{field+'_median': float(np.median([r[field] for r in subset])) if subset else None
                       for field in ('log_reverse_forward', 'gate_log_weight', 'log_acceptance')})
        labels = [j for j in range(len(atlas.w)) if j != k]
        row['deletion'] = {}
        for key in ('contacts', 'native_candidates', 'native_formations'):
            delta = full[key]-atlas.log_density(lc[key], labels)
            row['deletion'][key] = dict(mean_log_density_loss=float(delta.mean()), maximum_log_density_loss=float(delta.max()),
                                       below_tenth_density=int(np.sum(delta > np.log(10))))
        records.append(row)
    folds = []
    for arm in ('frozen', 'covariance'):
        for heldout in range(4):
            train = np.asarray([i for i, r in enumerate(contact) if r['arm'] == arm and r['replicate'] != heldout])
            test = np.asarray([i for i, r in enumerate(contact) if r['arm'] == arm and r['replicate'] == heldout])
            native_test = np.asarray([i for i, r in enumerate(native) if r['arm'] == arm and r['replicate'] == heldout])
            path = greedy_path(atlas, lc['contacts'][train])
            ranking = np.argsort(-atlas.w).tolist()
            for k in (28, 24, 20, 16, 12, 8, 4, 1):
                for method, labels in (('likelihood', path[k]), ('weight', ranking[:k])):
                    loss = full['contacts'][test]-atlas.log_density(lc['contacts'][test], labels)
                    nloss = full['native_candidates'][native_test]-atlas.log_density(lc['native_candidates'][native_test], labels)
                    folds.append(dict(arm=arm, heldout=heldout, method=method, k=k, labels=labels,
                                      heldout_contact_count=len(test), mean_contact_log_density_loss=float(loss.mean()),
                                      max_contact_log_density_loss=float(loss.max()),
                                      native_candidates=len(native_test), native_below_tenth=int(np.sum(nloss > np.log(10))),
                                      native_max_log_density_loss=float(nloss.max())))
    covers = {}
    for key in ('native_candidates', 'native_formations'):
        # Diagnostic only: native labels used here, never for the pruning folds.
        resp = responsibility[key]
        sol = milp(np.ones(len(atlas.w)), integrality=np.ones(len(atlas.w)), bounds=Bounds(0, 1),
                   constraints=LinearConstraint(resp, .9, np.inf), options={'time_limit': 30})
        assert sol.success, sol.message
        labels = np.flatnonzero(sol.x > .5).tolist()
        covers[key] = dict(labels=labels, k=len(labels), fraction_reference_gaussian_responsibility=.9,
                           minimum_retained_responsibility=float(resp[:, labels].sum(axis=1).min()),
                           proof='binary linear program, all collected poses; post-hoc, not a selection rule')
    result = dict(schema='atlas-compression-diagnosis-v1', campaign=str(campaign),
                  protocol=__doc__, sample_counts={k: len(v) for k, v in pools.items()}, components=records,
                  cross_validation=folds, diagnostic_minimum_support=covers, density_replay=replay,
                  reciprocal_contact_component_modes=reciprocal,
                  limitation='Correlated irreversible assembly starts; neither contact frequencies nor compression estimate equilibrium basin probabilities.',
                  source_sha256=hashlib.sha256((campaign/'provenance/model.json').read_bytes()).hexdigest())
    (out/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Reference atlas compression diagnosis', '',
             'Selection uses unlabeled geometric contacts from three runs and evaluates the fourth. Native registry labels enter diagnostics only. Each deletion renormalizes retained Gaussian weights and preserves the 10% uniform floor. Covariance-arm poses are scored under the fixed reference atlas; these are support tests, not counterfactual trajectories.', '',
             '| Component | Weight | Translation std range Å | Frozen native bonds | Covariance native bonds | Defensive |',
             '|---:|---:|---:|---:|---:|:---:|']
    for row in records:
        std = row['translation_std_A']
        lines.append(f"| {row['component']} | {row['weight']:.4f} | {min(std):.3g}–{max(std):.3g} | {row['arms']['frozen'].get('native_bonds_formed',0)} | {row['arms']['covariance'].get('native_bonds_formed',0)} | {row['defensive']} |")
    lines += ['', '## Leave-one-replicate-out pruning', '',
              'Loss is full-model minus compressed-model mean log density (nats/contact); negative values improve empirical density. Native candidate loss counts use density less than 10% of the original mixture. Positive mean improvement does not ensure rare-basin coverage.', '',
              '| Arm | K | Method | Mean held-out loss | Worst held-out pose loss | Native candidates losing >90% density |',
              '|---|---:|---|---:|---:|---:|']
    for arm in ('frozen', 'covariance'):
        for k in (28, 24, 20, 16, 12, 8, 4, 1):
            for method in ('likelihood', 'weight'):
                rows = [r for r in folds if r['arm'] == arm and r['k'] == k and r['method'] == method]
                lines.append(f"| {arm} | {k} | {method} | {np.mean([r['mean_contact_log_density_loss'] for r in rows]):.3f} | {max(r['max_contact_log_density_loss'] for r in rows):.3f} | {sum(r['native_below_tenth'] for r in rows)}/{sum(r['native_candidates'] for r in rows)} |")
    lines += ['', '## Diagnostic native support', '']
    for key, cover in covers.items():
        lines.append(f"- {key}: {cover['k']} components {cover['labels']} suffice and are necessary to retain ≥90% of the reference Gaussian responsibility at every observed pose in this set (binary optimization).")
    lines += ['', '## Native entry from a previously native-unbonded mover', '',
              'These are proposal diagnostics, not free energies: the gate is stochastic and the old mover may retain nonspecific contacts. Restricting to native-unbonded movers avoids pooling new entry with attempts to detach an already registered oligomer.', '',
              '| Component | Accepted / native candidates | Median log proposal ratio | Median Poisson gate log weight |',
              '|---:|---:|---:|---:|']
    for row in records:
        sample = row['arms']['frozen']['native_candidate_diagnostics']['false']
        if sample['proposed']:
            lines.append(f"| {row['component']} | {sample['accepted']}/{sample['proposed']} | {sample['log_reverse_forward_median']:.2f} | {sample['gate_log_weight_median']:.2f} |")
    lines += ['', '## Interpretation and limits', '',
              'The native-support set is an intentionally label-informed diagnosis; it must not be used as a template-free training target. Weight-ranked pruning can discard low-weight but indispensable registration basins. Native-blind contact likelihood can also discard unvisited or low-occupancy basins, and repeated residence can dominate its mean. A protected per-basin coverage floor, broad exploration channel, and held-out lower-tail density diagnostic are better safeguards than average likelihood alone.', '',
              'None of these trajectories shows native bond detachment. Observed contact occupancies are not equilibrium weights; this study can separate empirical proposal redundancy from registry support, but cannot yet distinguish equilibrium contacts from metastable trapping. The supplied atlas is itself native-informed through its training configurations.', '',
              'Generating-label counts also miss reciprocal support: a directed pair and its inverse can occupy different atlas components. The most common paired dominant components in frozen saved contacts are '+str(reciprocal['frozen'][:5])+'. Components with no accepted forward proposal may still be needed to evaluate reverse densities or other neighbor representations.', '',
              f"Frozen proposal log-density reconstruction maximum error: {max(r['maximum_reference_log_density_error'] for r in replay):.3g} nats. Full mixture responsibilities are used rather than attributing a pose only to its selected generating label.", '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    shown = records
    xs = np.arange(len(shown))
    axes[0].bar(xs-.15, [r['arms']['frozen'].get('native_candidate', 0) for r in shown], width=.65,
                color='#b5b7db', label='Valid native candidate')
    axes[0].bar(xs+.15, [r['arms']['frozen'].get('accepted_native_candidate', 0) for r in shown], width=.45,
                color='#5037a0', label='Accepted native candidate')
    axes[0].set(title='Few accepted routes; more geometric routes', xlabel='Reference component', ylabel='Frozen candidate count')
    axes[0].set_xticks([1, 5, 9, 11, 19, 22, 24]); axes[0].legend(fontsize=7)
    for method, color in [('likelihood', '#a43955'), ('weight', '#318084')]:
        ks = (4, 8, 12, 16, 20, 24, 28)
        values = [[r for r in folds if r['arm'] == 'frozen' and r['method'] == method and r['k'] == k] for k in ks]
        axes[1].plot(ks, [np.mean([r['mean_contact_log_density_loss'] for r in rows]) for rows in values], 'o-', color=color, label=method)
        axes[2].plot(ks, [1-sum(r['native_below_tenth'] for r in rows)/sum(r['native_candidates'] for r in rows) for rows in values], 'o-', color=color, label=method)
    axes[1].axhline(0, color='black', linewidth=.6)
    axes[1].set(title='Better occupied-contact likelihood…', xlabel='Retained components K', ylabel='Held-out log-density loss (nat)')
    axes[2].set(title='…can delete unoccupied native routes', xlabel='Retained components K', ylabel='Native candidate coverage', ylim=(0, 1.04))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Offline pruning: four frozen runs; leave one replicate out', fontsize=12)
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(out/f'compression-diagnosis.{ext}', dpi=160)
    plt.close(fig)
    lines += ['![Offline compression diagnostic](compression-diagnosis.png)', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(out=str(out), sample_counts=result['sample_counts'], minimum_support=covers), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', default='runs/atlas-transport-400')
    parser.add_argument('--out', default='results/atlas-compression-diagnosis')
    diagnose(parser.parse_args())
