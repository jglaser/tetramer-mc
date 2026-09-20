#!/usr/bin/env python3
"""Audit a conditional pilot's counts, costs, and saved-frame assembly.

Atomic checks and registered motif entry use the existing independent reference
classifier. Saved-frame entry is not a continuous-time formation/residence audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def one(task):
    reference, campaign, out, job = task
    reference, campaign, out = map(Path, (reference, campaign, out))
    sys.path.insert(0, str(reference/'scripts'))
    import numpy as np
    from audit_tetramer_assembly import AtomicAssembly, pose_arrays, rotations
    from tetramer_order import TetramerOrder, graph_summary
    from analyze_precursor_exchange import prepare_templates

    directory = Path(job['directory'])
    cfg, summary, provenance = (read(directory/name) for name in ('config.json', 'summary.json', 'manifest.json'))
    campaign_manifest = read(campaign/'manifest.json')
    assert summary['complete'] and summary['completed_sweeps'] == campaign_manifest['sweeps']
    assert provenance['executable_sha256'] == campaign_manifest['binary_sha256']
    assert provenance['config_sha256'] == job['config_sha256'] == sha(job['config'])
    assert provenance.get('model_sha256') is None
    assert provenance['shape_sha256'] == sha(directory/'provenance/shape.json')
    assert provenance['source_bundle_sha256'] == sha(directory/'provenance/source-bundle.json')
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
    assert frames[0]['sweep'] == 0 and frames[-1]['sweep'] == summary['completed_sweeps']
    n = len(frames[0]['poses'])
    actual_hash = hashlib.sha256(json.dumps(frames[0]['poses'], sort_keys=True).encode()).hexdigest()
    assert actual_hash == cfg['metadata']['initial_poses_sha256']
    rows = []
    count_histogram = Counter()
    for frame in frames:
        p, q = pose_arrays(frame)
        matrix = rotations(q)
        result = atomic.frame(p, q)
        assert result['hard_valid'], (job['id'], frame['sweep'], result)
        world = np.einsum('bij,aj->bai', matrix, atoms)+p[:, None, :]
        clearance = float(np.min(radius-np.linalg.norm(world, axis=2)-radii[None, :]))
        assert clearance >= -2e-8
        classified = order.classify(frame)
        entry = {tuple(r['bodies']) for r in classified['registered_tetramer_motifs'] if r['entry']}
        near = {tuple(e) for e in result['depletion_edges']}
        assert entry <= near
        if frame['sweep'] == 0:
            distances = np.linalg.norm(p[:, None]-p[None, :], axis=2)
            assert np.min(distances[np.triu_indices(n, 1)]) > 2*(bound+rd)
            assert not entry and not near
        row = dict(sweep=frame['sweep'], hard_valid=True,
                   minimum_atomic_wall_clearance_A=clearance,
                   minimum_interbody_gap_A=result['minimum_interbody_gap_A'],
                   native_entry=graph_summary(n, entry), nonspecific=graph_summary(n, near))
        state = frame.get('conditional_state')
        if state is not None:
            k, eta = state['k'], state['eta']
            assert len(eta) == (0 if k == 0 else 28*k-1)
            assert np.isfinite(eta).all()
            count_histogram[k] += 1
            row['conditional'] = dict(k=k, eta_dimension=len(eta),
                                      eta_square_mean=float(np.mean(np.square(eta))) if eta else None)
        rows.append(row)

    counts = defaultdict(Counter)
    global_sources = Counter()
    selected = Counter()
    nonzero_count_corrections = Counter()
    refresh_counts = Counter()
    refresh_rows = []
    for line in (directory/'moves.jsonl').read_text().splitlines():
        move = json.loads(line)
        kind = move['kind']
        counts[kind]['attempted'] += 1
        if kind in ('global', 'local'):
            selected[move['moving_index']] += 1
            counts[kind]['hard_valid'] += int(move['hard_valid'])
            counts[kind]['accepted'] += int(move['accepted'])
            counts[kind]['proposal_nulls'] += int(move['proposed_pose'] is None)
            if move['accepted'] and kind == 'global':
                global_sources[move['proposal'].get('branch', 'unknown')] += 1
        elif kind in ('gca', 'center_shift'):
            counts[kind]['accepted'] += int(move.get('accepted', True))
        if 'conditional' in kind:
            result = move.get('result', move)
            state = result.get('state', result.get('conditional_state', {}))
            if isinstance(state, dict) and 'k' in state:
                refresh_counts[state['k']] += 1
                fit = result.get('fit')
                if fit is not None:
                    probabilities = np.exp(fit['log_probabilities'])
                    assert abs(float(np.sum(probabilities))-1.) < 1e-10
                    assert 0 <= state['k'] < len(probabilities)
                    refresh_rows.append(dict(sweep=move['sweep'], k=state['k'],
                                             data_count=fit['data_count'],
                                             conditional_mean_k=float(np.dot(np.arange(len(probabilities)), probabilities)),
                                             probabilities=probabilities.tolist()))
        # Keep this diagnostic tolerant of per-kernel naming while retaining
        # full summary/move counts below; the canonical ratio is in production.
        containers = (move, move.get('proposal', {}), move.get('result', {}))
        for container in containers:
            values = [v for key, v in container.items() if 'count' in key and ('correction' in key or 'log_ratio' in key)]
            if any(isinstance(v, (int, float)) and abs(v) > 1e-12 for v in values):
                nonzero_count_corrections[kind] += 1
                break
    assert sum(selected.values()) == n*summary['completed_sweeps']
    assert set(selected) == set(range(n))
    assert all(v == summary['completed_sweeps'] for v in selected.values())
    for kind in ('local', 'global'):
        for key in ('attempted', 'hard_valid', 'accepted', 'proposal_nulls'):
            assert counts[kind][key] == summary['counts'][kind][key]
    assert sum(refresh_counts.values()) == summary['counts'].get('conditional_refreshes', 0)
    result = dict(id=job['id'], replicate=job['replicate'], variant=job['variant'], passed=True,
                  frames=rows, cpu_seconds=summary['sampler_cpu_seconds'], cost=summary['cost'],
                  counts=summary['counts'], move_counts={k: dict(v) for k, v in counts.items()},
                  accepted_global_sources=dict(global_sources), saved_frame_k_histogram=dict(count_histogram),
                  refresh_k_histogram=dict(refresh_counts), nonzero_count_corrections=dict(nonzero_count_corrections),
                  conditional_refreshes=refresh_rows,
                  final_native_entry=rows[-1]['native_entry'], final_nonspecific=rows[-1]['nonspecific'],
                  maximum_saved_native_bonds=max(r['native_entry']['bonds'] for r in rows),
                  maximum_saved_nonspecific_bonds=max(r['nonspecific']['bonds'] for r in rows),
                  first_saved_native_entry_sweep=next((r['sweep'] for r in rows if r['native_entry']['bonds']), None),
                  protocol=order.protocol,
                  classifier_sha256={name: sha(reference/'scripts'/name) for name in
                                     ('tetramer_order.py', 'audit_tetramer_assembly.py', 'analyze_precursor_exchange.py')},
                  source_sha256={name: sha(directory/name) for name in ('trajectory.jsonl', 'moves.jsonl', 'summary.json')})
    write(out/'runs'/job['id']/'analysis.json', result)
    print(json.dumps(dict(id=job['id'], passed=True, cpu_seconds=result['cpu_seconds'],
                          final_native_bonds=result['final_native_entry']['bonds'],
                          final_nonspecific_bonds=result['final_nonspecific']['bonds'])), flush=True)
    return result


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
    write(out/'analysis.json', dict(passed=True, results=results, manifest=manifest, analyzer_sha256=sha(__file__)))
    lines = ['# Conditional closure pilot', '',
             f"{len(manifest['preparations'])} independent dispersed starts, paired across four controls; {manifest['sweeps']} sweeps each. R=354.5082 Å, rd=1.5 Å, z=0.035 Å⁻³. All bodies move, with GCA and center shifts after each sweep. Production uses no external model or native docking data.", '',
             '| Start | Arm | Local accepted/attempted | Global accepted/attempted | Sampled K | Final native bonds | Final nonspecific bonds | CPU s |',
             '|---|---|---:|---:|---|---:|---:|---:|']
    for r in results:
        local, glob = r['counts']['local'], r['counts']['global']
        lines.append(f"| {r['replicate']} | {r['variant']} | {local['accepted']}/{local['attempted']} | {glob['accepted']}/{glob['attempted']} | {r['saved_frame_k_histogram']} | {r['final_native_entry']['bonds']} | {r['final_nonspecific']['bonds']} | {r['cpu_seconds']:.2f} |")
    lines += ['', 'Every saved frame passes independent atom-union overlap and wall checks. The initial separation certificate and zero external contacts are independently checked. Native entry uses the existing registered tetramer motif and external residue-patch classifier; native templates enter only this analysis.', '',
              'K occupancies are correlated observations. Native/contact counts are sampled-frame descriptors; this analysis does not reconstruct transient formation or residence between frames. These short runs do not establish equilibrium, crystal growth, a sampling speedup, or an unbiased physical marginal; the latter is assessed separately with exact-start stationarity and deterministic balance tests.', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(passed=True, out=str(out))), flush=True)


if __name__ == '__main__':
    main()
