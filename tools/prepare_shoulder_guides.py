#!/usr/bin/env python3
"""Freeze complementary shoulder guides; fit and probe geometry, never physical mass.

The original 1 < q < 2 partition is retained. All observed shoulder poses and
their original full importance weights enter the weighted fit without clipping.
An equal-weight geometry fit is a separate proposal control; their 50/50 mixture
is the prespecified primary guide. New independent integration remains required.
"""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from analyze_native_region_reference import native_q as original_q
from prepare_native_confirmation_atlas import (
    AtomUnionAudit, ELL, chart_log_density, coordinates, geometric_floor,
    laboratory_model, model_from_fit, to_world, weighted_fit,
)
from prepare_smc_normalizer_atlas import Density, arrays, read, relative_poses, sha, write

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/ab-global-widths-2-4-4x16384-l64-20260920'
GUIDES = ('weighted', 'geometry', 'mixture')


def shoulder(q):
    return 1. < q < 2.


def combine_models(weighted, geometry):
    """Normalized equal allocation, independent of estimated physical masses."""
    keys = ('schema', 'angular_length', 'coordinate_convention', 'shape_sha256')
    assert all(weighted[k] == geometry[k] for k in keys)
    result = {k: copy.deepcopy(weighted[k]) for k in keys}
    for key in ('anchors', 'means', 'covariances'):
        result[key] = copy.deepcopy(weighted[key] + geometry[key])
    result['weights'] = [.5*w for w in weighted['weights']] + [.5*w for w in geometry['weights']]
    assert abs(sum(result['weights'])-1) < 1e-14
    return result


def fit_guides(poses, log_weights, fixed, native, shape_hash):
    relative = relative_poses(poses, fixed)
    t, _, r = arrays(relative_poses([native], fixed))
    anchor = {'position': t[0].tolist(), 'rotation': r[0].tolist()}
    x = coordinates(relative, anchor, ELL)
    maximum_angle = float(np.degrees(2*np.arctan(np.linalg.norm(x[:, 3:]/ELL, axis=1))).max())
    assert maximum_angle < 60, 'Broad or near-seam shoulder cloud requires an explicitly revised chart protocol'
    floor = geometric_floor()
    fits = {'weighted': weighted_fit(x, np.asarray(log_weights), floor),
            'geometry': weighted_fit(x, np.zeros(len(x)), floor)}
    models = {name: model_from_fit(fit, anchor, shape_hash, 'single') for name, fit in fits.items()}
    models['mixture'] = combine_models(models['weighted'], models['geometry'])
    audits = {}
    for name, model in models.items():
        observed = Density(model).evaluate(relative)[0]
        if name == 'mixture':
            expected = np.logaddexp(Density(models['weighted']).evaluate(relative)[0],
                                   Density(models['geometry']).evaluate(relative)[0]) - math.log(2)
        else:
            expected = chart_log_density(x, fits[name], 'single')
        world = Density(laboratory_model(model, fixed)).evaluate(poses)[0]
        audits[name] = {'maximum_independent_chart_density_error': float(np.max(np.abs(observed-expected))),
                        'maximum_world_body_density_error': float(np.max(np.abs(observed-world)))}
        assert max(audits[name].values()) < 2e-8
    return models, fits, {'maximum_chart_angle_deg': maximum_angle, 'floor': floor.tolist(),
                          'density_checks': audits}


def load_source(campaign, shoulder_file):
    manifest = read(campaign/'manifest.json')
    audit_path = campaign/'assessment/contact-geometry-audit/results.json'
    audit = read(audit_path)
    assert audit['complete'] and audit['input_sha256'][str(campaign/'manifest.json')] == sha(campaign/'manifest.json')
    extracted = read(shoulder_file)
    by_key = {(r['population'], r['draw']): r for r in extracted}
    assert len(by_key) == len(extracted) == 66
    jobs = [j for j in manifest['jobs'] if j['covariance_std_scale'] == 4.]
    assert len(jobs) == 4 and len({j['seed'] for j in jobs}) == 4
    rows, sources, counts = [], [], set()
    total_cpu = total_valid = total_cloud_cpu = total_envelope_cpu = 0.
    for job in jobs:
        directory = Path(job['directory'])
        summary = read(directory/'summary.json')
        assert summary['complete'] and summary['numerical_nulls'] == 0
        assert summary['manifest']['seed'] == job['seed'] and summary['manifest']['proposal_anchor_index'] == 0
        assert audit['input_sha256'][str(directory/'summary.json')] == sha(directory/'summary.json')
        digest = hashlib.sha256(); selected = 0; seen = 0
        with (directory/'samples.jsonl').open('rb') as handle:
            for line in handle:
                digest.update(line); row = json.loads(line)
                assert row['draw'] == seen; seen += 1
                valid = row.get('capture_valid', False) and row.get('hard_valid', False)
                if valid and shoulder(row['q']):
                    item = by_key[(job['id'], row['draw'])]
                    assert all(item[key] == value for key, value in row.items()), 'Extracted pose or original weight changed'
                    assert item['all_atom_hard_valid'] and item['selected_as_all_shoulder']
                    assert item['independent_contact_neighbors'] == [True, True]
                    assert item['seed'] == job['seed'] and item['width'] == 4.
                    assert math.isfinite(item['log_importance_weight'])
                    rows.append(item); selected += 1
        assert digest.hexdigest() == audit['input_sha256'][str(directory/'samples.jsonl')]
        assert seen == job['samples'] == summary['samples']
        counts.add(seen)
        total_cpu += summary['sampler_cpu_seconds']; total_valid += summary['hard_valid']
        total_cloud_cpu += summary['cost']['cloud_cpu_seconds']
        total_envelope_cpu += summary['cost']['envelope_cpu_seconds']
        sources.append({'population': job['id'], 'seed': job['seed'], 'draws': seen, 'selected': selected,
                        'samples_path': str(directory/'samples.jsonl'), 'samples_sha256': digest.hexdigest(),
                        'summary_sha256': sha(directory/'summary.json')})
    assert len(rows) == len(extracted) and len(counts) == 1, 'All shoulder rows from equal-sized populations are required'
    cfg = read(campaign/'provenance/config.json')
    shape_path = campaign/'provenance/shape.json'
    assert sha(campaign/'provenance/config.json') == manifest['archive_sha256']['config.json']
    assert sha(shape_path) == manifest['archive_sha256']['shape.json']
    assert len(cfg['fixed_poses']) == 2 and len(cfg['metadata']['native_poses']) == 1
    assert cfg['capture_radius'] == 18. and cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035
    cost = {'unconditional_draws': sum(s['draws'] for s in sources), 'valid_draws': int(total_valid),
            'CPU_seconds': total_cpu, 'CPU_seconds_per_valid': total_cpu/total_valid,
            'cloud_plus_envelope_CPU_seconds_per_valid': (total_cloud_cpu+total_envelope_cpu)/total_valid}
    return rows, cfg, shape_path, sources, cost


def geometry_probes(model, cfg, audit, count, seed):
    """Independent draws from the actual normalized mixture, without retry."""
    rng = np.random.default_rng(seed); density = Density(model)
    components = rng.choice(len(model['weights']), size=count, p=model['weights'])
    records = []
    for index, component in enumerate(components):
        relative = density.draw_component(rng, int(component), 1)
        pose = to_world(relative, cfg['fixed_poses'][0])[0]
        q = original_q(cfg['metadata'], pose)
        gaps = audit.gaps(pose)
        hard = all(gap >= 0 for gap in gaps)
        capture = np.linalg.norm(np.asarray(pose['position'])-cfg['capture_center']) <= cfg['capture_radius']
        records.append({'draw': index, 'component': int(component), 'pose': pose, 'original_q': q,
                        'capture_valid': bool(capture), 'hard_valid_all_neighbors': hard,
                        'minimum_gap_by_neighbor_A': gaps,
                        'depletion_contact_by_neighbor': [gap < 2*cfg['depletant_radius'] for gap in gaps],
                        'shoulder': shoulder(q), 'valid_shoulder': bool(hard and capture and shoulder(q))})
    fraction = float(np.mean([r['valid_shoulder'] for r in records]))
    return {'seed': seed, 'draws': count, 'valid_shoulder': sum(r['valid_shoulder'] for r in records),
            'valid_shoulder_fraction': fraction, 'binomial_standard_error': math.sqrt(fraction*(1-fraction)/count),
            'hard_valid': sum(r['hard_valid_all_neighbors'] for r in records),
            'capture_valid': sum(r['capture_valid'] for r in records),
            'shoulder': sum(r['shoulder'] for r in records),
            'component_draws': np.bincount(components, minlength=len(model['weights'])).tolist()}, records


def prepare(campaign, shoulder_file, out, count=256):
    assert count > 0
    assert not out.exists(), 'Use a fresh frozen output directory'
    started = time.monotonic()
    (out/'provenance').mkdir(parents=True)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_campaign': str(campaign),
                'source_shoulder_file': str(shoulder_file), 'source_shoulder_sha256': sha(shoulder_file),
                'proposal_anchor_index': 0, 'angular_length_A': ELL, 'original_q_interval': [1., 2.],
                'interval_endpoints': 'open; original metric unchanged',
                'fit_rows': 'All 66 independently checked shoulder poses; no clipping, trimming, retries, or per-population renormalization',
                'weighted_fit': 'Normalized original full importance weights; maximum-likelihood full 6x6 scatter plus additive floor',
                'geometry_fit': 'Equal weight for every one of the same 66 poses; maximum-likelihood full 6x6 scatter plus additive floor',
                'primary_mixture_weights': [.5, .5], 'primary_components': ['weighted', 'geometry'],
                'floor': geometric_floor().tolist(),
                'floor_rule': 'Add diag(.05^2 I3, [ell*tan(.1 degree/2)]^2 I3) in A body frame',
                'chart_rule': 'Original native reference expressed in A body frame; left Cayley residual. Require maximum observed chart angle below 60 degrees.',
                'probe_counts': {name: count for name in GUIDES},
                'probe_seeds': {name: 99441010+1009*i for i, name in enumerate(GUIDES)},
                'selection': 'Primary 50/50 allocation fixed before probes; geometry outcomes do not change the models',
                'no_physical_production': True, 'fitter_sha256': sha(__file__)}
    write(out/'protocol.json', protocol)
    rows, cfg, shape_path, sources, cost = load_source(campaign, shoulder_file)
    poses = [r['pose'] for r in rows]
    for row in rows:
        assert abs(original_q(cfg['metadata'], row['pose'])-row['q']) < 2e-8
    models, fits, fit_audit = fit_guides(poses, [r['log_importance_weight'] for r in rows],
                                       cfg['fixed_poses'][0], cfg['metadata']['native_poses'][0], sha(shape_path))
    physical = {k: copy.deepcopy(cfg[k]) for k in ('fixed_poses', 'capture_center', 'capture_radius',
                                                  'depletant_radius', 'reservoir_density', 'metadata')}
    for name, model in models.items():
        model['proposal_provenance'] = {'kind': 'Frozen original-q shoulder proposal', 'guide': name,
            'source_rows': len(rows), 'protocol_sha256': sha(out/'protocol.json'), 'physical': physical,
            'proposal_anchor_index': 0, 'inference': 'Proposal fit only; fresh independent integration with a complete geometric cover required.'}
        write(out/f'model-{name}.json', model)
    cfg_copy = copy.deepcopy(cfg); cfg_copy['shape'] = str(out/'provenance/shape.json')
    write(out/'config.json', cfg_copy)
    inputs = {'input-shoulder-poses.json': shoulder_file, 'source-campaign-manifest.json': campaign/'manifest.json',
              'source-assessment.json': campaign/'assessment/analysis.json',
              'source-geometry-audit.json': campaign/'assessment/contact-geometry-audit/results.json',
              'input-config.json': campaign/'provenance/config.json', 'shape.json': shape_path}
    for name in ('prepare_shoulder_guides.py', 'prepare_native_confirmation_atlas.py',
                 'prepare_smc_normalizer_atlas.py', 'analyze_native_region_reference.py'):
        inputs[name] = ROOT/'tools'/name
    for name, path in inputs.items(): shutil.copy2(path, out/'provenance'/name)
    atom_audit = AtomUnionAudit(read(shape_path), cfg['fixed_poses'])
    training_checks = []
    for row in rows:
        gaps = atom_audit.gaps(row['pose']); assert all(gap >= 0 for gap in gaps)
        assert all(gap < 2*cfg['depletant_radius'] for gap in gaps)
        training_checks.append({'population': row['population'], 'draw': row['draw'], 'gaps_A': gaps})
    probes = {}
    for name in GUIDES:
        report, records = geometry_probes(models[name], cfg, atom_audit, count, protocol['probe_seeds'][name])
        report['conditional_CPU_seconds_for_4x8192_pure_guide'] = 32768*report['valid_shoulder_fraction']*cost['CPU_seconds_per_valid']
        probes[name] = report
        with (out/f'geometry-probes-{name}.jsonl').open('w') as handle:
            for record in records: handle.write(json.dumps(record, allow_nan=False)+'\n')
    report = {'complete': True, 'source_rows': len(rows), 'source_populations': sources,
        'fits': {name: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fit.items()}
                 for name, fit in fits.items()}, 'fit_audit': fit_audit, 'physical': physical,
        'training_geometry_checks': training_checks, 'geometry_probes': probes, 'cost_reference': cost,
        'model_sha256': {name: sha(out/f'model-{name}.json') for name in GUIDES},
        'config_sha256': sha(out/'config.json'), 'protocol_sha256': sha(out/'protocol.json'),
        'archived_sha256': {name: sha(out/'provenance'/name) for name in inputs},
        'probe_sha256': {name: sha(out/f'geometry-probes-{name}.jsonl') for name in GUIDES},
        'wall_seconds': time.monotonic()-started,
        'cost_caveat': 'Pure-guide conditional forecast from old mean CPU per valid pose. It excludes additional complete-cover proposals and changes in cloud volume and proposal overhead. No runtime guarantee.',
        'inference': 'The 66 poses are independent proposal draws, not equilibrium samples. Original weighted ESS is small. The geometric covariance and 50/50 allocation define alternative proposal coverage only. No fresh depletant clouds, physical normalizer, or equilibrium covariance estimate is produced.'}
    write(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=SOURCE)
    parser.add_argument('--shoulder-poses', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probes-per-guide', type=int, default=256)
    args = parser.parse_args(); campaign = args.campaign.resolve()
    path = args.shoulder_poses or campaign/'assessment/contact-geometry-audit/width-4.0-shoulder-poses.json'
    result = prepare(campaign, path.resolve(), args.out.resolve(), args.probes_per_guide)
    print(json.dumps({k: result[k] for k in ('complete', 'model_sha256', 'config_sha256', 'fit_audit', 'geometry_probes', 'wall_seconds')}, indent=2))


if __name__ == '__main__':
    main()
