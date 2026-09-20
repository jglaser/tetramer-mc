#!/usr/bin/env python3
"""Evaluate frozen shoulder guides on existing intermediate contacts, without fitting."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from audit_shoulder_mis_independently import chart_values, independent_densities, near, read, rotation, sha
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_smc_normalizer_atlas import Density, relative_poses

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve(); assert not out.exists()
    start = time.process_time()
    root = ROOT/'runs/ab-intermediate-cayley-reference-4x131072-l64-20260920'
    prep = ROOT/'runs/ab-intermediate-cayley-cover-preparation-20260920'
    frozen = ROOT/'runs/ab-shoulder-guide-preparation-20260920'
    manifest = read(root/'manifest.json'); cfg = read(root/'provenance/config.json')
    input_cfg = read(root/'provenance/input-config.json'); region = read(root/'provenance/region.json')
    original_cfg = read(frozen/'config.json')
    assert {k: cfg[k] for k in KEYS} == {k: input_cfg[k] for k in KEYS} == {k: original_cfg[k] for k in KEYS}
    assert region['physical_fixed_neighbors'] == cfg['fixed_poses'] and region['physical_metric'] == cfg['metadata']
    assert region['minimum_original_q'] == 2 and region['maximum_original_q'] == 5
    assert region['minimum_original_q_inclusive'] and not region['maximum_original_q_inclusive']
    assert region['mahalanobis_radius'] == 10
    for filename, digest in manifest['archive_sha256'].items(): assert sha(root/'provenance'/filename) == digest
    shape_path = root/'provenance/shape.json'; shape_hash = sha(shape_path)
    assert shape_hash == region['shape_sha256'] == sha(frozen/'provenance/shape.json')
    models = {name: read(frozen/f'model-{name}.json') for name in ('mixture', 'geometry')}
    assert all(model['shape_sha256'] == shape_hash for model in models.values())
    prior = read(ROOT/'runs/ab-shoulder-mixture-confirmation-4x65536-l64-20260920/assessment-streaming.json')
    guide = dict(prior['guide'])
    assert guide['weight'] == .75 and guide['uniform_probability'] == .05 and guide['anchor_index'] == 0
    assert guide['anchor_pose'] == cfg['fixed_poses'][0]
    proof = read(prep/'report.json')['proof']
    angle = proof['full_product_cover_angle_cap']; radius = 5*cfg['metadata']['member_error_scale']
    node = dict(reference=cfg['metadata']['native_poses'][0], centroid=proof['centroid_A'], ball_radius=radius,
                angle_cap=angle, volume=4*radius**3/3*(angle-math.sin(angle)))
    product = dict(covers=[node], weights=[1.], scales=[1.])
    historical_path = prep/'source-intermediate-poses.json'; historical = read(historical_path)
    assert len(historical) == 16
    assert sha(historical_path) == read(prep/'report.json')['source_pose_support'][0]['sha256']
    known_protocol = read(prep/'protocol.json')
    for source in known_protocol['known_pose_sources']:
        assert sha(source['path']) == source['sha256']
    valid = []; source_files = []; total = 0; seeds = set()
    for job in manifest['jobs']:
        path = Path(job['directory']); run, summary = read(path/'manifest.json'), read(path/'summary.json')
        assert summary['complete'] and summary['manifest'] == run
        assert run['region_sha256'] == manifest['region_sha256'] and run['shape_sha256'] == shape_hash
        assert run['config_sha256'] == sha(root/'provenance/config.json')
        assert run['seed'] == job['seed'] and run['seed'] not in seeds; seeds.add(run['seed'])
        assert run['physical_fixed_neighbors'] == cfg['fixed_poses'] and run['activity'] == cfg['reservoir_density']
        digest = hashlib.sha256(); count = 0
        with (path/'samples.jsonl').open('rb') as handle:
            for line in handle:
                digest.update(line); row = json.loads(line)
                assert row['draw'] == count; count += 1
                if row['log_importance_weight'] is None: continue
                assert row['hard_valid'] and row['region_valid'] and row['capture_valid'] and 2 <= row['q'] < 5
                assert len(row['clouds']) == 2
                valid.append(dict(population=job['id'], seed=job['seed'], source=str(path/'samples.jsonl'), source_row=row, pose=row['pose'],
                                  source_draw=row['draw'], original_q=row['q']))
        assert count == job['samples'] == run['samples'] == summary['samples']
        assert digest.hexdigest() == summary['samples_sha256']
        source_files.append(dict(path=str(path/'samples.jsonl'), sha256=digest.hexdigest(), seed=job['seed'], draws=count))
        total += count
    largest = sorted(valid, key=lambda row: row['source_row']['log_importance_weight'], reverse=True)[:16]
    # Cross-evaluate every valid existing cover row for a geometric-only hit diagnostic.
    raw = [row['source_row'] for row in valid]; observed = {}; densities = {}
    for name, model in models.items():
        lg, lc, radii, jac, lv, q, capture = independent_densities(raw, cfg, model, region, guide, product)
        near(q, [row['q'] for row in raw]); assert capture.all() and np.isfinite(lc).all()
        near(lc, [-lv-row['log_physical_jacobian'] for row in raw])
        weights = np.exp(lg-lc); estimate = weights.sum()/total
        variance = (weights@weights-weights.sum()**2/total)/(total-1)/total
        observed[name] = dict(valid_cover_rows=len(raw), all_cover_draws=total,
                              estimated_full_guide_hard_capture_window_probability=float(estimate),
                              observed_relative_SE=float(math.sqrt(variance)/estimate),
                              weight_ESS=float(weights.sum()**2/(weights@weights)), maximum_weight_fraction=float(weights.max()/weights.sum()),
                              hypothetical_4x16384_valid_count=float(65536*estimate), hypothetical_4x32768_valid_count=float(131072*estimate),
                              scope='Retrospective geometric importance diagnostic only; not a physical normalizer or a coverage guarantee. Observed weights are diffuse, but unvisited high-guide-probability subregions can invalidate extrapolated counts.')
        densities[name] = (lg, lc)
    selected = [('historical', p) for p in historical]+[('direct_top', p) for p in largest]
    selected_raw = [dict(pose=p['pose']) for _, p in selected]
    values = {}
    max_density_error = 0.
    for name, model in models.items():
        lg, lc, radii, jac, lv, q, capture = independent_densities(selected_raw, cfg, model, region, guide, product)
        assert np.isfinite(lg).all() and np.isfinite(lc).all() and capture.all()
        gaussian, norms, _ = Density(model).evaluate(relative_poses([p['pose'] for _, p in selected], cfg['fixed_poses'][0]))
        base = np.logaddexp(math.log(.25/node['volume']), math.log(.75*.05/np.prod(guide['cube_lengths'])))
        direct_log = np.logaddexp(base, math.log(.75*.95)+gaussian)
        max_density_error = max(max_density_error, near(lg, direct_log))
        values[name] = (lg, lc, norms, q)
    atom = AtomUnionAudit(read(shape_path), cfg['fixed_poses']); rows = []
    for index, (kind, source) in enumerate(selected):
        q = float(values['mixture'][3][index]); near(q, source['original_q'])
        gaps = atom.gaps(source['pose']); assert all(gap >= 0 for gap in gaps)
        row = dict(kind=kind, source=source['source'], source_draw=source['source_draw'], pose=source['pose'], original_q=q,
                   minimum_atomic_gap_by_neighbor_A=gaps, all_atom_hard_valid=True,
                   depletant_contact_by_neighbor=[gap < 2*cfg['depletant_radius'] for gap in gaps],
                   source_log_importance_weight=source['source_row']['log_importance_weight'],
                   cloud_log_weights=[cloud['log_weight'] for cloud in source['source_row']['clouds']], guides={})
        for name in models:
            lg, lc, norms, _ = values[name]
            row['guides'][name] = dict(full_hybrid_log_density=float(lg[index]), complete_cover_log_density=float(lc[index]),
                                      guide_to_cover_density_ratio=float(math.exp(lg[index]-lc[index])),
                                      mahalanobis_radii=norms[index].tolist())
        rows.append(row)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), complete=True, physical={k: cfg[k] for k in KEYS},
                  target='2<=original q<5, both AB neighbors, unchanged hard union and bath',
                  guide_law='.25 q_max=5 product cover + .75(.95 frozen model + .05 unchanged cube/Haar)',
                  product_cover=product, guide_description=guide, rows=rows, observed_geometric_hit_diagnostic=observed,
                  minimum_atomic_gap_A=min(min(row['minimum_atomic_gap_by_neighbor_A']) for row in rows),
                  maximum_independent_density_error=max_density_error, source_files=source_files,
                  input_sha256=dict(historical_poses=sha(historical_path), config=sha(root/'provenance/config.json'),
                                    region=sha(root/'provenance/region.json'), shape=shape_hash,
                                    **{f'model-{name}': sha(frozen/f'model-{name}.json') for name in models}),
                  helper_sha256=sha(__file__), independent_density_helper_sha256=sha(ROOT/'tools/audit_shoulder_mis_independently.py'),
                  CPU_seconds=time.process_time()-start,
                  scope='Existing poses only. No fit, pose generation, depletant insertions, changed physical weights, physical normalizer estimate, or historical/fresh pooling. Ratios at selected points cannot guarantee finite-budget basin coverage.')
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name in ('diagnose_intermediate_frozen_guides.py', 'audit_shoulder_mis_independently.py', 'prepare_native_confirmation_atlas.py', 'prepare_smc_normalizer_atlas.py'):
        shutil.copy2(ROOT/'tools'/name, out/name)
    print(json.dumps(dict(observed=observed, min_gap=result['minimum_atomic_gap_A'], max_error=max_density_error,
                         selected=[dict(kind=r['kind'], q=r['original_q'], ratios={n:r['guides'][n]['guide_to_cover_density_ratio'] for n in models},
                                        radii={n:r['guides'][n]['mahalanobis_radii'] for n in models}, gaps=r['minimum_atomic_gap_by_neighbor_A']) for r in rows]), indent=2))


if __name__ == '__main__':
    main()
