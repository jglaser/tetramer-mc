#!/usr/bin/env python3
"""Replay the matched atlas campaign and classify every native transition.

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
    maximum_position_error = maximum_rotation_error = 0.

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
        nonlocal state
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
            eta = np.asarray(atlas_state['eta'])
            assert np.isfinite(eta).all()
            row['atlas'] = dict(eta_dimension=eta.size, eta_square_mean=float(np.mean(eta**2)),
                                fit=frame.get('atlas_fit'))
        rows.append(row)
        # Restart replay from exact serialized poses to avoid accumulated roundoff.
        state = copy.deepcopy(frame['poses'])

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
            if kind == 'global' and move['hard_valid']:
                candidate = list(state)
                candidate[i] = move['proposed_pose']
                candidate_entry, candidate_stay = classify(candidate, affected)
                novel = candidate_entry-active
                branch_counts['native_candidate'] += int(bool(candidate_entry))
                branch_counts['new_native_candidate'] += int(bool(novel))
                branch_counts['accepted_native_candidate'] += int(bool(candidate_entry) and move['accepted'])
                branch_counts['accepted_new_native_candidate'] += int(bool(novel) and move['accepted'])
                if candidate_entry:
                    native_proposals.append(dict(sweep=sweep, moving_index=i, source=source,
                        accepted=move['accepted'], entry=sorted(candidate_entry), novel=sorted(novel),
                        log_acceptance=move['log_acceptance'],
                        log_reverse_forward=proposal.get('log_reverse_forward'),
                        gate_log_weight=(move.get('gate') or {}).get('log_weight')))
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
        elif kind == 'center_shift':
            if move.get('accepted', True):
                accepted[kind] += 1
                for pose in state:
                    pose['position'] = (np.asarray(pose['position'])+move['result']['displacement']).tolist()
        elif kind == 'atlas_refresh':
            assert atlas_state is not None
            assert move['old_state'] == atlas_state
            atlas_state = copy.deepcopy(move['state'])
            eta = np.asarray(atlas_state['eta'])
            atlas_refreshes.append(dict(sweep=sweep, eta_dimension=eta.size,
                eta_square_mean=float(np.mean(eta**2)), fit=move.get('fit')))
        else:
            raise AssertionError((job['id'], 'unknown move', kind))
        if changed_pairs:
            entry, stay = accepted_entry_stay or classify(state, changed_pairs)
            subset = set(changed_pairs)
            active = {key for key in active if key[:2] not in subset or key in stay} | entry
        if previous != active:
            changes.append(dict(sweep=sweep, serial=serial, kind=kind, source=source,
                formed=sorted(active-previous), broken=sorted(previous-active),
                moving_index=move.get('moving_index'), component_index=proposal.get('component_index')))
    assert selected == set(range(n))
    observe(saved[current_sweep])
    checkpoint = read(directory/'checkpoint.json')
    compare(state, checkpoint['poses'])
    assert checkpoint.get('atlas_state') == atlas_state
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
    if job['variant'] == 'frozen' and manifest.get('source_campaign'):
        source_manifest = read(Path(manifest['source_campaign'])/'manifest.json')
        source_job = next(j for j in source_manifest['jobs']
                          if j['replicate'] == job['replicate'] and j.get('mode', j.get('variant')) == 'frozen')
        source_path = Path(source_job['directory'])/'trajectory.jsonl'
        source_frames = {f['sweep']: f for f in map(json.loads, source_path.read_text().splitlines())}
        common = [f for f in frames if f['sweep'] in source_frames]
        exact = all(f['poses'] == source_frames[f['sweep']]['poses'] for f in common)
        assert exact, (job['id'], 'Frozen trajectory changed from supplied reference')
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
        frozen_reference_reproduction=frozen_reproduction,
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
        aggregates[arm] = dict(runs=len(subset), learned=dict(learned),
            cpu_seconds=sum(r['cpu_seconds'] for r in subset),
            motif_formations=sum(r['motif_formations'] for r in subset),
            motif_breakages=sum(r['motif_breakages'] for r in subset),
            native_bond_formations=sum(r['native_bond_formations'] for r in subset),
            native_bond_breakages=sum(r['native_bond_breakages'] for r in subset),
            native_registry_changes=sum(r['native_registry_changes'] for r in subset),
            final_native_bonds=[r['final_native']['bonds'] for r in subset],
            first_native_association_sweeps=[r['first_native_association_sweep'] for r in subset])
    save(out/'aggregates.json', aggregates)
    lines = ['# Matched atlas transport pilot', '',
        f"{len(manifest['preparations'])} reused dispersed starts, {manifest['sweeps']} sweeps each. Every arm retains the same 28-component native-informed atlas, initial poses, MC seeds, rd=1.5 Å, z=0.035 Å⁻³, local moves, spherical GCA, and center shifts. Parameter transport is added in stages; the component count remains fixed.", '',
        '| Arm | Gaussian accepted/attempted | Native candidates accepted/proposed | Native bonds formed/broken | Final native bonds by start | CPU s |',
        '|---|---:|---:|---:|---|---:|']
    for arm, record in aggregates.items():
        c = record['learned']
        lines.append(f"| {arm} | {c.get('accepted', 0)}/{c.get('attempted', 0)} | {c.get('accepted_native_candidate', 0)}/{c.get('native_candidate', 0)} | {record['native_bond_formations']}/{record['native_bond_breakages']} | {record['final_native_bonds']} | {record['cpu_seconds']:.2f} |")
    lines += ['', 'The candidate classification includes every hard-valid global candidate, including rejected proposals. Native transitions are reconstructed after every accepted local, global, or GCA update, with the unchanged registered-pose and external residue-patch criterion. Bond formation and breakage count body pairs. A switch of native motif label on a continuously bonded pair is reported separately as a registry change.', '',
        'Registry changes by arm: '+str({k: v['native_registry_changes'] for k, v in aggregates.items()})+'. These are correlated proposal and transition counts, not independent trials.', '',
        'Every physical update is replayed to every saved frame and the final checkpoint. Saved frames independently pass atom-union hard-core and wall checks. The same supplied atlas contains native information in every arm; this tests retention of that accessibility, not native discovery. Reusing these starts is a paired control, not four additional independent preparations. A short assembly pilot cannot establish equilibrium, a mixing speedup, or crystal growth.', '']
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
    figure.suptitle('Same native-informed atlas; progressive parameter transport')
    figure.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        figure.savefig(out/f'atlas-transport-comparison.{ext}', dpi=180)
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
