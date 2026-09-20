#!/usr/bin/env python3
"""Audit existing inner-reference extremes against frozen guides and atom unions.

This diagnostic does not draw depletants, refit, replace weights, or estimate
a new physical integral. It retains the source importance weights verbatim.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import math
from pathlib import Path
import shutil

import numpy as np

from analyze_native_region_reference import GaussianGuide, native_q, proposal_log_density
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_smc_normalizer_atlas import Density, ROOT, read, relative_poses, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=ROOT/'runs/ab-inner-shoulder-matching-comparison-20260920/comparison.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    assert not out.exists(), 'Preserve previous audits with a fresh output directory'
    comparison = read(args.comparison); assert comparison['complete']
    selected = comparison['focused_band']['top_16']; assert len(selected) == 16
    prep = ROOT/'runs/ab-shoulder-guide-preparation-20260920'
    cfg = read(prep/'config.json'); shape_path = prep/'provenance/shape.json'
    physical = comparison['focused_band']['physical_signature']
    # The reference comparison binds the input configurations to this physical
    # identity; all source-configuration fields are checked again here.
    source_root = Path(comparison['focused_band']['root'])
    source_cfg = read(source_root/'provenance/config.json')
    keys = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')
    assert {k: cfg[k] for k in keys} == {k: source_cfg[k] for k in keys}
    assert sha(shape_path) == sha(source_root/'provenance/shape.json')
    models = {name: read(prep/f'model-{name}.json') for name in ('weighted', 'geometry', 'mixture')}
    reference_campaign = ROOT/'runs/ab-shoulder-mixture-confirmation-4x65536-l64-20260920'
    assessment = read(reference_campaign/'assessment-streaming.json')
    assert assessment['complete'] and assessment['all_rows_and_hashes_validated']
    campaign_cfg = read(reference_campaign/'provenance/config.json')
    assert {k: cfg[k] for k in keys} == {k: campaign_cfg[k] for k in keys}
    guide_description = assessment['guide']; cover = assessment['cover_mixture']
    assert guide_description['anchor_index'] == 0 and guide_description['weight'] == .75
    assert guide_description['uniform_probability'] == .05
    poses = [row['pose'] for row in selected]
    relative = relative_poses(poses, cfg['fixed_poses'][0])
    values = {name: Density(model).evaluate(relative) for name, model in models.items()}
    scalar = {name: GaussianGuide(dict(guide_description, model_sha256=sha(prep/f'model-{name}.json')),
                                model, cfg, sha(shape_path)) for name, model in models.items()}
    atom_audit = AtomUnionAudit(read(shape_path), cfg['fixed_poses'])
    rows = []; density_error = 0.; q_error = 0.
    for index, source in enumerate(selected):
        row = copy.deepcopy(source)
        q = native_q(cfg['metadata'], source['pose']); q_error = max(q_error, abs(q-source['q']))
        assert 1 < q < 1.1 and abs(q-source['q']) < 2e-8
        gaps = atom_audit.gaps(source['pose']); assert all(gap >= 0 for gap in gaps)
        assert all(gap < 2*cfg['depletant_radius'] for gap in gaps)
        assert math.dist(source['pose']['position'], cfg['capture_center']) <= cfg['capture_radius']
        logcover, membership = proposal_log_density(cover, source['pose']); assert any(membership)
        results = {}
        for name in models:
            log_gaussian = float(values[name][0][index])
            log_scalar_guide, inside_cube, component_logs = scalar[name].log_density(source['pose'])
            assert inside_cube
            log_uniform = -3*math.log(2*cfg['capture_radius'])
            log_vector_guide = float(np.logaddexp(math.log(.95)+log_gaussian, math.log(.05)+log_uniform))
            error = abs(log_scalar_guide-log_vector_guide); density_error = max(density_error, error)
            assert error < 2e-7
            log_full = float(np.logaddexp(math.log(.25)+logcover, math.log(.75)+log_scalar_guide))
            results[name] = {'gaussian_log_density': log_gaussian,
                'mahalanobis_radii': values[name][1][index].tolist(),
                'gaussian_plus_cube_log_density': log_scalar_guide,
                'full_q2_campaign_log_density': log_full,
                'scalar_component_log_densities': component_logs}
        row.update(original_q_independent=q, minimum_atomic_gap_by_neighbor_A=gaps,
            depletant_contact_by_neighbor=[gap < 2*cfg['depletant_radius'] for gap in gaps],
            all_atom_hard_valid=True, frozen_guide_diagnostics=results,
            cloud_log_weight_difference=source['clouds'][0]['log_weight']-source['clouds'][1]['log_weight'])
        rows.append(row)
    report = {'complete': True, 'physical_signature': physical, 'rows': rows,
        'maximum_original_q_error': q_error, 'maximum_scalar_vector_guide_log_density_error': density_error,
        'minimum_atomic_gap_A': min(min(row['minimum_atomic_gap_by_neighbor_A']) for row in rows),
        'input_sha256': {'comparison': sha(args.comparison), 'config': sha(prep/'config.json'), 'shape': sha(shape_path),
                        **{f'model-{name}': sha(prep/f'model-{name}.json') for name in models}},
        'analysis_sha256': sha(__file__),
        'scope': 'Existing top16 only. Original physical weights and poses retained verbatim. Independent atomic checks and proposal-density/radius diagnostics do not define new regions, new importance weights, a refit or an equilibrium claim.'}
    out.mkdir(parents=True); write(out/'analysis.json', report)
    for file in ('analyze_inner_shoulder_top_poses.py', 'analyze_native_region_reference.py',
                 'prepare_native_confirmation_atlas.py', 'prepare_smc_normalizer_atlas.py'):
        shutil.copy2(ROOT/'tools'/file, out/file)
    for row in rows:
        print(row['population'], row['draw'], 'q', row['q'], 'gaps', row['minimum_atomic_gap_by_neighbor_A'],
              'radii', {name: d['mahalanobis_radii'] for name, d in row['frozen_guide_diagnostics'].items()},
              'clouds', [c['log_weight'] for c in row['clouds']])


if __name__ == '__main__':
    main()
