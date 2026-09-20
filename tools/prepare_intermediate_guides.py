#!/usr/bin/env python3
"""Freeze [2,5) intermediate guides, then probe their geometry without clouds.

All valid poses from one complete independent reference campaign enter the fit.
The fitted density is a proposal, not a newly estimated physical normalizer.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
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
from scipy.special import logsumexp

from analyze_native_region_reference import native_q
from prepare_native_confirmation_atlas import AtomUnionAudit, ELL, geometric_floor, to_world
from prepare_shoulder_guides import fit_guides
from prepare_smc_normalizer_atlas import Density, read, sha, write

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs/ab-intermediate-cayley-reference-4x131072-l64-20260920'
GUIDES = ('weighted', 'geometry', 'mixture')
WINDOW = dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False)
PHYSICAL = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')


def in_window(q):
    return 2 <= q < 5


def load_source(campaign):
    manifest = read(campaign/'manifest.json'); cfg = read(campaign/'provenance/config.json')
    region = read(campaign/'provenance/region.json'); shape_path = campaign/'provenance/shape.json'
    for name, digest in manifest['archive_sha256'].items(): assert sha(campaign/'provenance'/name) == digest
    assert region['shape_sha256'] == sha(shape_path)
    assert region['physical_fixed_neighbors'] == cfg['fixed_poses'] and region['physical_metric'] == cfg['metadata']
    assert region['minimum_original_q'] == 2 and region['maximum_original_q'] == 5
    assert region['minimum_original_q_inclusive'] and not region['maximum_original_q_inclusive']
    assert len(cfg['fixed_poses']) == 2 and len(cfg['metadata']['native_poses']) == 1
    assert cfg['capture_radius'] == 18 and cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035
    rows = []; sources = []; seeds = set(); budgets = set(); cpu = 0.
    for job in manifest['jobs']:
        path = Path(job['directory']); run, summary = read(path/'manifest.json'), read(path/'summary.json')
        assert summary['complete'] and run == summary['manifest']
        assert run['seed'] == job['seed'] and run['seed'] not in seeds; seeds.add(run['seed'])
        assert run['config_sha256'] == sha(campaign/'provenance/config.json')
        assert run['shape_sha256'] == sha(shape_path) and run['region_sha256'] == manifest['region_sha256']
        assert run['physical_fixed_neighbors'] == cfg['fixed_poses']
        assert run['activity'] == .035 and run['lambda_ratio'] == 64 and run['lambda'] == 2.24 and run['cloud_replicates'] == 2
        assert run['minimum_original_q'] == 2 and run['maximum_original_q'] == 5
        assert run['minimum_original_q_inclusive'] and not run['maximum_original_q_inclusive']
        for filename, field in [('input-config.json', 'config_sha256'), ('shape.json', 'shape_sha256'), ('region.json', 'region_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
            assert sha(path/'provenance'/filename) == run[field]
        digest = hashlib.sha256(); count = selected = 0
        with (path/'samples.jsonl').open('rb') as handle:
            for line in handle:
                digest.update(line); row = json.loads(line)
                assert row['draw'] == count; count += 1
                valid = row['region_valid'] and row['capture_valid'] and row['hard_valid']
                assert valid == (row['log_importance_weight'] is not None)
                if not valid: continue
                q = native_q(cfg['metadata'], row['pose'])
                assert abs(q-row['q']) < 2e-8 and in_window(q)
                assert math.dist(row['pose']['position'], cfg['capture_center']) <= cfg['capture_radius']
                clouds = row['clouds']; assert len(clouds) == 2
                for c in clouds:
                    assert type(c['overlap_points']) is int and 0 <= c['overlap_points'] <= c['raw_points']
                    assert abs(c['log_weight']-(.035*c['lower_volume']+c['overlap_points']*math.log1p(1/64))) < 2e-8
                logf = float(logsumexp([c['log_weight'] for c in clouds])-math.log(2))
                assert abs(logf+row['log_hard_weight']-row['log_importance_weight']) < 2e-8
                assert abs(row['log_hard_weight']-run['log_latent_shell_volume']-row['log_physical_jacobian']) < 2e-8
                rows.append(dict(population=job['id'], seed=job['seed'], **row)); selected += 1
        assert count == job['samples'] == run['samples'] == summary['samples']
        assert digest.hexdigest() == summary['samples_sha256']
        budgets.add(count); cpu += summary['sampler_cpu_seconds']
        sources.append(dict(population=job['id'], seed=job['seed'], unconditional_draws=count, selected_valid_rows=selected,
                            samples_path=str(path/'samples.jsonl'), samples_sha256=digest.hexdigest(),
                            manifest_sha256=sha(path/'manifest.json'), summary_sha256=sha(path/'summary.json'),
                            sampler_CPU_seconds=summary['sampler_cpu_seconds']))
    assert len(sources) == 4 and len(budgets) == 1 and sum(p['unconditional_draws'] for p in sources) == 524288
    assert len(rows) == 3335
    logw = np.array([row['log_importance_weight'] for row in rows]); normalized = np.exp(logw-logsumexp(logw))
    assessment = read(campaign/'assessment/analysis.json')
    assert assessment['original_q_window'] == WINDOW and assessment['estimate']['draws'] == 524288
    assert abs(1/(normalized@normalized)-assessment['estimate']['ess']) < 2e-8
    assert abs(normalized.max()-assessment['estimate']['max_fraction']) < 2e-8
    cost = dict(unconditional_draws=524288, valid_draws=len(rows), sampler_CPU_seconds=cpu,
                CPU_seconds_per_valid=cpu/len(rows), scope='Source total generation CPU divided by valid rows, including source rejection overhead; future geometry/cloud costs may differ.')
    return rows, cfg, shape_path, sources, cost


def geometry_probe(model, cfg, atom, count, seed):
    rng = np.random.default_rng(seed); density = Density(model)
    components = rng.choice(len(model['weights']), size=count, p=model['weights'])
    records = []
    for index, component in enumerate(components):
        pose = to_world(density.draw_component(rng, int(component), 1), cfg['fixed_poses'][0])[0]
        q = native_q(cfg['metadata'], pose); gaps = atom.gaps(pose)
        hard = all(gap >= 0 for gap in gaps)
        capture = math.dist(pose['position'], cfg['capture_center']) <= cfg['capture_radius']
        records.append(dict(draw=index, component=int(component), pose=pose, original_q=q,
                            capture_valid=capture, hard_valid_all_neighbors=hard, in_original_window=in_window(q),
                            valid_original_window=hard and capture and in_window(q), minimum_gap_by_neighbor_A=gaps,
                            depletant_contact_by_neighbor=[gap < 2*cfg['depletant_radius'] for gap in gaps]))
    successes = sum(row['valid_original_window'] for row in records); p = successes/count
    z = 1.959963984540054; midpoint = (p+z*z/(2*count))/(1+z*z/count)
    half = z/(1+z*z/count)*math.sqrt(p*(1-p)/count+z*z/(4*count*count))
    report = dict(seed=seed, draws=count, valid_original_window=successes, valid_fraction=p,
                  binomial_standard_error=math.sqrt(p*(1-p)/count), Wilson_95_interval=[midpoint-half, midpoint+half],
                  hard_valid=sum(row['hard_valid_all_neighbors'] for row in records), capture_valid=sum(row['capture_valid'] for row in records),
                  in_original_window=sum(row['in_original_window'] for row in records),
                  component_draws=np.bincount(components, minlength=len(model['weights'])).tolist(),
                  scope='Independent raw-model draws, all failures retained. No product/cube branch, no depletant cloud, no physical acceptance/normalizer claim.')
    return report, records


def prepare(campaign, out):
    assert not out.exists(), 'Use a fresh immutable preparation directory'
    started = time.process_time(); rows, cfg, shape_path, sources, cost = load_source(campaign)
    out.joinpath('provenance').mkdir(parents=True)
    inputs = {'input-config.json': campaign/'provenance/config.json', 'shape.json': shape_path,
              'source-campaign-manifest.json': campaign/'manifest.json', 'source-assessment.json': campaign/'assessment/analysis.json',
              'source-region.json': campaign/'provenance/region.json'}
    for name in ('prepare_intermediate_guides.py', 'prepare_shoulder_guides.py', 'prepare_native_confirmation_atlas.py',
                 'prepare_smc_normalizer_atlas.py', 'analyze_native_region_reference.py'):
        inputs[name] = ROOT/'tools'/name
    for name, path in inputs.items(): shutil.copy2(path, out/'provenance'/name)
    with (out/'source-valid-poses.jsonl').open('w') as handle:
        for row in rows: handle.write(json.dumps(row, allow_nan=False)+'\n')
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_campaign=str(campaign), q_window=WINDOW,
                    source_unconditional_draws=524288, source_valid_fit_rows=len(rows), source_populations=sources,
                    source_valid_poses_sha256=sha(out/'source-valid-poses.jsonl'),
                    source_selection='All hard/capture/original-window-valid reference rows. No historical poses, clipping, trimming, per-population normalization or retrospective removal.',
                    weighted_fit='Normalize the original full importance weights once across all equal-budget source populations. Full 6x6 MLE scatter plus additive floor.',
                    geometry_fit='Equal weight for each of the same source poses. Full 6x6 MLE scatter plus additive floor.',
                    primary_mixture=dict(components=['weighted', 'geometry'], weights=[.5, .5]),
                    proposal_anchor_index=0, angular_length_A=ELL,
                    chart='Original native reference in A body frame; left Cayley residual; maximum observed angle must be below 60 degrees.',
                    additive_covariance_floor=geometric_floor().tolist(),
                    floor_rule='diag(.05^2 I3, [ell*tan(.1 degree/2)]^2 I3)',
                    geometry_probe_counts={name: 256 for name in GUIDES},
                    geometry_probe_seeds={name: 99941010+1009*i for i, name in enumerate(GUIDES)},
                    probe_law='Independent raw model draws, all invalids retained; original [2,5), capture, and both AB atomic gap checks. No product/cube branch or clouds.',
                    frozen_before_probes=True, no_physical_sampling=True,
                    archived_sha256={name: sha(out/'provenance'/name) for name in inputs})
    write(out/'protocol.json', protocol)
    models, fits, fit_audit = fit_guides([row['pose'] for row in rows], [row['log_importance_weight'] for row in rows],
                                       cfg['fixed_poses'][0], cfg['metadata']['native_poses'][0], sha(shape_path))
    physical = {key: copy.deepcopy(cfg[key]) for key in PHYSICAL}
    for name, model in models.items():
        model['proposal_provenance'] = dict(kind='Frozen intermediate-region proposal', guide=name, source_rows=len(rows),
                                           q_window=WINDOW, protocol_sha256=sha(out/'protocol.json'), physical=physical,
                                           proposal_anchor_index=0, inference='Proposal only; low source physical ESS requires fresh independent validation with complete support.')
        write(out/f'model-{name}.json', model)
    new_cfg = copy.deepcopy(cfg); new_cfg['shape'] = str(out/'provenance/shape.json'); write(out/'config.json', new_cfg)
    # Record all model/input hashes before independent probe RNGs are initialized.
    freeze = dict(protocol_sha256=sha(out/'protocol.json'), config_sha256=sha(out/'config.json'),
                  model_sha256={name: sha(out/f'model-{name}.json') for name in GUIDES},
                  archived_sha256=protocol['archived_sha256'], source_valid_poses_sha256=sha(out/'source-valid-poses.jsonl'))
    write(out/'freeze.json', freeze)
    atom = AtomUnionAudit(read(shape_path), cfg['fixed_poses']); source_checks = []
    for row in rows:
        gaps = atom.gaps(row['pose']); assert all(gap >= 0 for gap in gaps)
        source_checks.append(dict(population=row['population'], draw=row['draw'], minimum_gap_by_neighbor_A=gaps))
    write(out/'source-atomic-checks.json', source_checks)
    probes = {}
    for name in GUIDES:
        report, records = geometry_probe(models[name], cfg, atom, 256, protocol['geometry_probe_seeds'][name])
        report['conditional_pure_guide_cost_forecasts'] = [dict(unconditional_draws=n,
                expected_valid_rows=n*report['valid_fraction'], source_cost_CPU_seconds=n*report['valid_fraction']*cost['CPU_seconds_per_valid'])
                for n in (65536, 131072)]
        probes[name] = report
        with (out/f'geometry-probes-{name}.jsonl').open('w') as handle:
            for record in records: handle.write(json.dumps(record, allow_nan=False)+'\n')
        assert sha(out/f'model-{name}.json') == freeze['model_sha256'][name]
    assert sha(out/'protocol.json') == freeze['protocol_sha256']
    result = dict(complete=True, fit_rows=len(rows), source_populations=sources, physical=physical, q_window=WINDOW,
                  fits={name: {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in fit.items()} for name, fit in fits.items()},
                  fit_audit=fit_audit, source_atomic_checks_sha256=sha(out/'source-atomic-checks.json'),
                  minimum_source_atomic_gap_A=min(min(row['minimum_gap_by_neighbor_A']) for row in source_checks),
                  source_atomic_rows_checked=len(source_checks), geometry_probes=probes, cost_reference=cost,
                  frozen=freeze, freeze_sha256=sha(out/'freeze.json'),
                  probe_sha256={name: sha(out/f'geometry-probes-{name}.jsonl') for name in GUIDES}, CPU_seconds=time.process_time()-started,
                  forecast_scope='Raw-model conditional geometry forecast only; source CPU per valid pose includes source overhead and need not match new cloud volumes. Complete-cover proposals, cube branch, density evaluation, and rejected-draw overhead are not forecast.',
                  inference='No new physical samples, normalizer, equilibrium covariance, or physical convergence claim. Weighted ESS refers to noisy source importance weights; equal geometry and 50/50 mixture define proposal controls only.')
    write(out/'report.json', result)
    print(json.dumps({key: result[key] for key in ('complete', 'fit_rows', 'minimum_source_atomic_gap_A', 'fit_audit', 'geometry_probes', 'CPU_seconds')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); prepare(args.campaign.resolve(), args.out.resolve())
