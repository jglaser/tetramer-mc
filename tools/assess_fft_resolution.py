#!/usr/bin/env python3
"""Stream a frozen FFT peak cache to assess nested angular thinning.

This is a candidate-recovery diagnostic at unchanged translation spacing, not
a proof of basin recovery or a mathematical minimum. Native references, when
requested, are held-out labels and never affect the selected budget.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from fft_depletion_docking import super_fibonacci


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def nested_ranks(n, seed):
    if n <= 0:
        raise ValueError('rotation count must be positive')
    order = np.random.default_rng(seed).permutation(n)
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    return rank


def load_poses(path):
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and 'slots' in raw:
        poses = [x['optimized_pose'] for x in raw['slots']]
    else:
        poses = [x.get('pose', x) for x in raw]
    if not poses:
        raise ValueError('reference pose list is empty')
    for p in poses:
        t, q = np.asarray(p['position']), np.asarray(p['orientation'])
        if t.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(np.r_[t, q])):
            raise ValueError('invalid reference pose')
        if not np.isclose(np.linalg.norm(q), 1., atol=1e-6):
            raise ValueError('reference quaternion must be normalized')
    return poses


def expanded_reference(poses, translation_tolerance, angle_degrees):
    if translation_tolerance <= 0 or not 0 < angle_degrees <= 180:
        raise ValueError('invalid pose tolerances')
    chord = 2*np.sin(np.radians(angle_degrees)/4)
    features, owners = [], []
    for slot, pose in enumerate(poses):
        t = np.asarray(pose['position'], float)
        q = np.asarray(pose['orientation'], float)
        q /= np.linalg.norm(q)
        matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        for tt, qq in ((t, q), (-matrix.T@t, q*np.array([1., -1., -1., -1.]))):
            for sign in (1., -1.):
                features.append(np.r_[tt/translation_tolerance, sign*qq/chord])
                owners.append(slot)
    return np.asarray(features), np.asarray(owners), chord


def accumulate(rows, ranks, budgets, ref_features, owners, tree, best,
               translation_tolerance, chord, tracked_rows=None, tracked_budget_index=None):
    """Best grid score within BOTH translation and orientation tolerances."""
    rows = np.asarray(rows)
    if not len(rows):
        return
    ids = rows[:, 0].astype(np.int64)
    if np.any(ids < 0) or np.any(ids >= len(ranks)) or np.any(ids != rows[:, 0]):
        raise ValueError('invalid rotation index in cache')
    q = rows[:, 2:6].astype(float)
    q /= np.linalg.norm(q, axis=1)[:, None]
    feature = np.column_stack((rows[:, 6:9]/translation_tolerance, q/chord))
    # sqrt(2) ball conservatively encloses the two unit balls. Filter each.
    neighbors = tree.query_ball_point(feature, np.sqrt(2.)+1e-10, workers=1)
    counts = np.fromiter((len(x) for x in neighbors), dtype=np.int64, count=len(neighbors))
    if not np.any(counts):
        return
    row_ids = np.repeat(np.arange(len(rows)), counts)
    ref_ids = np.concatenate([x for x in neighbors if len(x)]).astype(np.int64)
    delta = feature[row_ids]-ref_features[ref_ids]
    keep = ((np.einsum('ij,ij->i', delta[:, :3], delta[:, :3]) <= 1.+1e-10)
            & (np.einsum('ij,ij->i', delta[:, 3:], delta[:, 3:]) <= 1.+1e-10))
    row_ids, ref_ids = row_ids[keep], ref_ids[keep]
    for i, budget in enumerate(budgets):
        keep = ranks[ids[row_ids]] < budget
        selected_rows = row_ids[keep]
        selected_slots = owners[ref_ids[keep]]
        selected_scores = rows[selected_rows, 9]
        if tracked_rows is not None and i == tracked_budget_index and len(selected_rows):
            order = np.lexsort((-selected_scores, selected_slots))
            first = np.r_[True, selected_slots[order][1:] != selected_slots[order][:-1]]
            winners = order[first]
            improve = selected_scores[winners] > best[i, selected_slots[winners]]
            winners = winners[improve]
            tracked_rows[selected_slots[winners]] = rows[selected_rows[winners]]
        np.maximum.at(best[i], selected_slots, selected_scores)


def recommendation(best, budgets, score_fraction=.9, required_fraction=.95,
                   required_stratum=.9):
    full = best[-1]
    measurable = np.isfinite(full)
    if not np.any(measurable):
        return {'budget': None, 'reason': 'no reference basin has a cached nearby candidate'}, []
    indices = np.flatnonzero(measurable)
    order = indices[np.argsort(full[indices], kind='stable')]
    strata = [x for x in np.array_split(order, 4) if len(x)]
    rows, selected = [], None
    for i, budget in enumerate(budgets):
        recovered = np.isfinite(best[i]) & (best[i] >= score_fraction*full)
        fraction = float(np.mean(recovered[measurable]))
        groups = [float(np.mean(recovered[s])) for s in strata]
        passed = fraction >= required_fraction and min(groups) >= required_stratum
        if selected is None and passed:
            selected = int(budget)
        rows.append(dict(rotations=int(budget), nearby_basins=int(np.isfinite(best[i]).sum()),
                         full_cache_nearby_basins=int(measurable.sum()), reference_basins=len(full),
                         retained_score_fraction=fraction, score_quartile_retention=groups,
                         passes_candidate_criterion=passed))
    return dict(budget=selected, score_fraction=score_fraction,
                required_fraction=required_fraction, required_score_quartile_fraction=required_stratum,
                rule='First nested budget meeting native-blind criteria, conditional on full-cache coverage.'), rows


def angular_diagnostic(n, ranks, budgets, probes, seed, lever_radius):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(probes, 4)); q /= np.linalg.norm(q, axis=1)[:, None]
    base = super_fibonacci(n)
    results = []
    for budget in budgets:
        centers = base[ranks < budget]
        distances = cKDTree(np.vstack((centers, -centers))).query(q, workers=1)[0]
        angles = 4*np.arcsin(np.clip(distances/2, 0, 1))
        displacement = 2*lever_radius*np.sin(angles/2)
        results.append(dict(rotations=int(budget), random_probes=probes,
            nearest_angle_degrees=dict(zip(('median', 'p95', 'p99', 'sample_max'),
                np.degrees(np.quantile(angles, [.5, .95, .99, 1])).tolist())),
            atom_center_displacement_bound_A=dict(zip(('median', 'p95', 'p99', 'sample_max'),
                np.quantile(displacement, [.5, .95, .99, 1]).tolist()))))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[10000, 30000, 75000, 150000, 300000])
    parser.add_argument('--translation-tolerance', type=float, default=4.)
    parser.add_argument('--angle-tolerance', type=float, default=10.)
    parser.add_argument('--score-fraction', type=float, default=.9)
    parser.add_argument('--required-fraction', type=float, default=.95)
    parser.add_argument('--required-stratum', type=float, default=.9)
    parser.add_argument('--seed', type=int, default=20260928001)
    parser.add_argument('--chunk', type=int, default=100000)
    parser.add_argument('--probes', type=int, default=8192)
    parser.add_argument('--export-budget', type=int, help='Export best raw peak per reference from this tested subset; keep null slots')
    parser.add_argument('--native-definition', type=Path,
                        help='Optional post-hoc labels; cannot alter selected budget')
    args = parser.parse_args()
    plans = sorted(args.scan.glob('plan-gpu*.json'))
    if not plans and (args.scan/'plan.json').exists():
        plans = [args.scan/'plan.json']
    if not plans:
        raise ValueError('no scan plan found')
    data = [json.loads(p.read_text()) for p in plans]
    frozen = data[0]
    keys = ('rotations', 'spacing', 'rd', 'shape_sha256')
    if any(any(p[k] != frozen[k] for k in keys) for p in data):
        raise ValueError('inconsistent scan plans')
    manifests = sorted(args.scan.glob('manifest*.json'))
    if not manifests or not all(json.loads(p.read_text()).get('complete') for p in manifests):
        raise ValueError('cache is incomplete')
    n = frozen['rotations']
    budgets = sorted(set(args.budgets))
    if not budgets or min(budgets) <= 0 or max(budgets) > n:
        raise ValueError('budgets must be in 1..scan rotations')
    budgets = sorted(set(budgets+[n]))
    if args.export_budget is not None and args.export_budget not in budgets:
        raise ValueError('export budget must be one of the tested budgets')
    if args.chunk <= 0 or args.probes <= 0 or any(not 0 < v <= 1 for v in
            (args.score_fraction, args.required_fraction, args.required_stratum)):
        raise ValueError('invalid allocation or fraction')
    poses = load_poses(args.references)
    ref_features, owners, chord = expanded_reference(poses, args.translation_tolerance, args.angle_tolerance)
    tree = cKDTree(ref_features)
    ranks = nested_ranks(n, args.seed)
    shape_path = Path(frozen['shape'])
    if sha(shape_path) != frozen['shape_sha256']:
        raise ValueError('shape hash mismatch')
    shape = json.loads(shape_path.read_text())
    lever = float(max(np.linalg.norm(a['center']) for a in shape['atoms']))
    args.out.mkdir(parents=True, exist_ok=False)
    plan = dict(schema='fft-angular-resolution-audit-v1', arguments={k:str(v) if isinstance(v, Path) else v
                 for k,v in vars(args).items()}, scan_plan_sha256={p.name:sha(p) for p in plans},
                scan_manifest_sha256={p.name:sha(p) for p in manifests}, reference_sha256=sha(args.references),
                script_sha256=sha(__file__), native_selection=False,
                scope='Nested thinning at unchanged translation spacing. No rescore or basin-convergence proof.')
    (args.out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (args.out/'executed-assess_fft_resolution.py').write_text(Path(__file__).read_text())
    best = np.full((len(budgets), len(poses)), -np.inf)
    tracked_rows = np.full((len(poses), 11), np.nan) if args.export_budget is not None else None
    tracked_index = budgets.index(args.export_budget) if args.export_budget is not None else None
    started, count, shard_hashes = time.time(), 0, {}
    expected = {}
    for p in manifests:
        m = json.loads(p.read_text())
        if 'shard_sha256' in m:
            expected[p.name.replace('manifest-', 'shard-').replace('.json', '.npy')] = m['shard_sha256']
        for sh in m.get('shards', []):
            expected[f"shard-{sh['worker']:04d}.npy"] = sh['sha256']
    shards = sorted(args.scan.glob('shard-*.npy'))
    if set(expected) != {p.name for p in shards}:
        raise ValueError('shards differ from completed manifest')
    for path in shards:
        shard_hashes[path.name] = sha(path)
        if shard_hashes[path.name] != expected[path.name]:
            raise ValueError('shard hash mismatch: '+str(path))
        rows = np.load(path, mmap_mode='r')
        if rows.ndim != 2 or rows.shape[1] != 11:
            raise ValueError('unexpected cache shape')
        for start in range(0, len(rows), args.chunk):
            block = rows[start:start+args.chunk]
            accumulate(block, ranks, budgets, ref_features, owners, tree, best,
                       args.translation_tolerance, chord, tracked_rows, tracked_index)
            count += len(block)
        print(json.dumps(dict(shard=path.name, attempted_rows=count, elapsed=time.time()-started)), flush=True)
    chosen, table = recommendation(best, budgets, args.score_fraction, args.required_fraction, args.required_stratum)
    geometry = angular_diagnostic(n, ranks, budgets, args.probes, args.seed+1, lever)
    if chosen.get('budget') is not None:
        np.save(args.out/'selected-rotation-indices.npy', np.flatnonzero(ranks < chosen['budget']))
    np.save(args.out/'best-grid-overlap.npy', best)
    if tracked_rows is not None:
        slots = []
        for slot, row in enumerate(tracked_rows):
            present = bool(np.isfinite(row[9]))
            pose = None
            if present:
                q = row[2:6]/np.linalg.norm(row[2:6])
                pose = dict(position=row[6:9].tolist(), orientation=q.tolist())
            slots.append(dict(slot=slot, present=present, raw_peak=row.tolist() if present else None,
                              pose=pose, hard_validity='not_checked' if present else 'missing'))
        export = dict(schema='fft-subset-basin-starts-v1', budget=args.export_budget,
                      full_rotation_grid=n, nested_seed=args.seed, reference_sha256=sha(args.references),
                      scan_shard_sha256=shard_hashes, native_selection=False, slots=slots,
                      scope='Raw grid peaks, not hard-valid starts. Exact repair and independent refinement remain required.')
        (args.out/'subset-basin-starts.json').write_text(json.dumps(export, indent=2)+'\n')
        np.save(args.out/'exported-subset-rotation-indices.npy', np.flatnonzero(ranks < args.export_budget))
    heldout = None
    if args.native_definition:
        from native_contact_regions import NativeContactRegions
        from posthoc_native_recall import recall
        labels = recall(poses, NativeContactRegions(args.native_definition))
        slots = [i for i,r in enumerate(labels) if r['complete_native_entry']]
        full = best[-1]
        heldout = dict(definition_sha256=sha(args.native_definition), native_reference_slots=slots,
                       labels=[dict(slot=i, **labels[i]) for i in slots],
                       budgets=[dict(rotations=int(b), nearby_slots=[i for i in slots if np.isfinite(best[k,i])],
                           score_retained_slots=[i for i in slots if np.isfinite(best[k,i]) and best[k,i] >= args.score_fraction*full[i]])
                           for k,b in enumerate(budgets)], selection_used_labels=False)
    report = dict(complete=True, recommendation=chosen, candidate_table=table, angular_diagnostics=geometry,
                  heldout_native=heldout, processed_rows=count, shard_sha256=shard_hashes,
                  cached_spacing_A=frozen['spacing'], maximum_translation_rounding_A=np.sqrt(3)*frozen['spacing']/2,
                  maximum_atom_center_lever_A=lever, elapsed_seconds=time.time()-started,
                  limitations=['Full-cache missing basins remain missing; selection conditions on measurable references.',
                    'A nearby high-scoring raw peak need not repair into the reference basin.',
                    'Random-probe maximum angle is not a covering-radius proof.',
                    'No coarser translation FFT was computed; spacing cannot be chosen by angular thinning.',
                    'One nested sequence is a diagnostic, not replicated discovery.',
                    'Reference basins came from the full cache; this is retrospective candidate retention.',
                    'A newly generated lower-count super-Fibonacci grid differs from the audited subset.'])
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(recommendation=chosen, candidate_table=table), indent=2))


if __name__ == '__main__':
    main()
