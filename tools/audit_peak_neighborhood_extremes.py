#!/usr/bin/env python3
"""Independently check atomic gaps at the largest local-reference weights."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from analyze_native_region_reference import native_q
from audit_shoulder_mis_independently import chart_values, near
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_native_confirmation_atlas import AtomUnionAudit
from run_shoulder_mis_campaign import local_dependencies


def audit(analysis_path, out, count=8):
    require(not out.exists(), 'Use a fresh selected-atom audit directory')
    require(type(count) is int and count > 0, 'Positive integer selection count required')
    source = read(analysis_path); require(source['complete'], 'Incomplete local reference audit')
    old_path = analysis_path.parent/'provenance/old-chart-model.json'
    require(sha(old_path) == source['archived_sha256'][old_path.name], 'Old partition chart changed')
    old_model = read(old_path); reports = []
    for campaign in source['campaigns']:
        original = campaign['original_reference_audit']; root = Path(campaign['root'])
        require(sha(root/'manifest.json') == original['manifest_sha256'], 'Reference manifest changed')
        master = read(root/'manifest.json'); cfg = read(root/'provenance/config.json')
        require(sha(root/'provenance/shape.json') == original['shape_sha256'], 'Shape changed')
        atom = AtomUnionAudit(read(root/'provenance/shape.json'), cfg['fixed_poses'])
        heap = []; n = 0; valid = 0; hashes = {}
        for job in master['jobs']:
            path = Path(job['directory'])/'samples.jsonl'; digest = hashlib.sha256(); local_n = 0
            with path.open('rb') as handle:
                for line in handle:
                    digest.update(line); row = json.loads(line)
                    require(row['draw'] == local_n, 'Row order changed'); local_n += 1; n += 1
                    if row['log_importance_weight'] is None: continue
                    valid += 1
                    point = dict(population=job['id'], seed=job['seed'], source_samples=str(path), **row)
                    entry = (row['log_importance_weight'], n, point)
                    if len(heap) < count: heapq.heappush(heap, entry)
                    elif entry[:2] > heap[0][:2]: heapq.heapreplace(heap, entry)
            require(local_n == job['samples'] and digest.hexdigest() == original['sample_sha256'][str(path)],
                    'Original sample rows or unconditional count changed')
            hashes[str(path)] = digest.hexdigest()
        require(n == campaign['physical']['full']['draws'] and valid == campaign['physical']['full']['nonzero'],
                'Audited full count mismatch')
        points = []
        for _, _, row in sorted(heap, reverse=True):
            q = native_q(cfg['metadata'], row['pose']); near(q, row['q'])
            gaps = atom.gaps(row['pose']); require(min(gaps) >= 0, 'Selected weight fails independent atom-union hard check')
            rotation = Rotation.from_quat(np.asarray(row['pose']['orientation'])[[1, 2, 3, 0]]).as_matrix()
            _, old_radius, _ = chart_values(old_model, np.asarray([row['pose']['position']]), np.asarray([rotation]), cfg['fixed_poses'][0])
            logf = float(logsumexp([cloud['log_weight'] for cloud in row['clouds']]))-math.log(2)
            near(logf+row['log_hard_weight'], row['log_importance_weight'])
            points.append(dict(population=row['population'], seed=row['seed'], draw=row['draw'], source_samples=row['source_samples'],
                pose=row['pose'], original_q=q, new_chart_radius=row['latent_radius'], old_chart_radius=float(old_radius[0, 0]),
                minimum_atomic_gap_by_neighbor_A=gaps, original_clouds=row['clouds'], log_boltzmann_mean=logf,
                original_log_importance_weight=row['log_importance_weight'], log_mean_contribution=row['log_importance_weight']-math.log(n),
                fraction_of_local_mass=math.exp(row['log_importance_weight']-math.log(n)-campaign['physical']['full']['logQ'])))
        if points: near(points[0]['fraction_of_local_mass'], campaign['physical']['full']['maximum_fraction'])
        reports.append(dict(root=str(root), radius_A=campaign['radius_A'], draws=n, valid=valid,
                            sample_sha256=hashes, top_poses=points))
        print(json.dumps(dict(radius_A=campaign['radius_A'], checked=len(points),
            minimum_checked_gap_A=min(min(point['minimum_atomic_gap_by_neighbor_A']) for point in points) if points else None)), flush=True)
    (out/'provenance').mkdir(parents=True)
    sources = {**local_dependencies([Path(__file__)]), 'local-analysis.json': analysis_path,
               'old-chart-model.json': old_path}
    for name, path in sources.items(): (out/'provenance'/name).write_bytes(path.read_bytes())
    result = dict(complete=True, source_analysis_sha256=sha(analysis_path), top_count_per_campaign=count, campaigns=reports,
        archived_sha256={p.name: sha(p) for p in (out/'provenance').iterdir()},
        scope='Independent atom-union hard checks at selected largest original weights only. Unchanged full-N estimators; no new physical draws, depletant clouds, fitting, or neighborhood/tail bounds.')
    write(out/'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=8); args = parser.parse_args()
    audit(args.analysis.resolve(), args.out.resolve(), args.count)
