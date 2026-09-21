#!/usr/bin/env python3
"""Fit three frozen inner-shoulder guides from original whole-ball references.

This is deterministic postprocessing only. Each fit uses all positive original
importance weights from that center's R=.5 reference law, retaining every zero
in the original physical-normalizer denominator. Different radii, historical
discovery streams, and priority-assigned masks never enter these fits.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigvalsh
from scipy.special import logsumexp

from analyze_peak_neighborhood import geometry_model_audit
from audit_shoulder_mis_independently import near
from compare_intermediate_local_reference import read_batches
from fit_peak_reference_proposal import population_diagnostics, serial
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_native_confirmation_atlas import (coordinates, weighted_fit, geometric_floor,
    model_from_fit, chart_log_density, laboratory_model)
from prepare_smc_normalizer_atlas import Density, relative_poses
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
ASSESSMENT = ROOT/'runs/ab-shoulder-guide-peak-reference-assessment-20260921/analysis.json'
ASSESSMENT_SHA256 = '100f67ef74289bee1c3f0a38651c2b1c211e83afb9e50d0c9c516521fd44e3ef'
CENTERS = ('direct', 'mixture', 'geometry')
WINDOW = dict(minimum=1., maximum=1.1, lower_inclusive=False, upper_inclusive=False)
PHYSICAL = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')
FIT_RULE = ('Original unclipped importance weights pooled once across all 16 equal-budget populations; '
            'no equal-population renormalization, priority assignment, or pooling of different radius laws. '
            'Weighted raw scatter plus additive geometric floor; one globally normalized Gaussian per center.')


def select_campaigns(assessment, direct):
    """Select whole radius-.5 laws, explicitly excluding independent shell sums."""
    require(assessment['complete'] and direct['complete'], 'Completed source assessments required')
    require(assessment['original_q_window'] == direct['original_q_window'] == WINDOW,
            'Require original strict 1<q<1.1 target')
    selected = {}
    for name in CENTERS:
        source = direct['campaigns'] if name == 'direct' else assessment['campaigns']
        candidates = [c for c in source if c.get('radius_A') == .5 and
                      (name == 'direct' or c.get('owner') == name)]
        require(len(candidates) == 1, f'Need exactly one whole radius-.5 campaign for {name}')
        selected[name] = candidates[0]
    require(len({c['root'] for c in selected.values()}) == 3, 'Source campaigns repeat')
    return selected


def check_region(region, cfg, shape_hash):
    require(region['mahalanobis_radius'] == .5 and region.get('minimum_mahalanobis_radius', 0.) == 0.,
            'Only the original whole radius-.5 ball can supply a fit')
    window = dict(minimum=region['minimum_original_q'], maximum=region['maximum_original_q'],
                  lower_inclusive=region.get('minimum_original_q_inclusive', True),
                  upper_inclusive=region.get('maximum_original_q_inclusive', True))
    require(window == WINDOW, 'Source original-q window changed')
    require(region['fixed_neighbor'] == cfg['fixed_poses'][0] and
            region['physical_fixed_neighbors'] == cfg['fixed_poses'], 'Source AB frame or neighbors changed')
    for rkey, ckey in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'),
        ('activity', 'reservoir_density'), ('depletant_radius', 'depletant_radius'), ('physical_metric', 'metadata')]:
        require(region[rkey] == cfg[ckey], f'Source physical field changed: {rkey}')
    require(region['shape_sha256'] == region['gaussian_chart']['shape_sha256'] == shape_hash,
            'Source chart and physical shape differ')


def collect_rows(master, expected_hashes, expected, population_count=16, samples_per_population=16384):
    """Keep original weights verbatim; every original draw retains its denominator."""
    jobs = master['jobs']
    require(len(jobs) == population_count, 'Wrong number of original source populations')
    require(len({j['seed'] for j in jobs}) == population_count, 'Repeated original random streams')
    require(len({j['id'] for j in jobs}) == population_count, 'Repeated source population IDs')
    rows, labels, populations, hashes = [], [], [], {}
    draws = 0
    for job in jobs:
        require(job['samples'] == samples_per_population, 'Original source budget changed')
        path = (Path(job['directory'])/'samples.jsonl').resolve()
        require(str(path) in expected_hashes, 'No audited hash for original source rows')
        digest, count, logs = hashlib.sha256(), 0, []
        zero_flags = dict(hard_invalid=0, capture_invalid=0, region_invalid=0, out_of_inner_window=0)
        for raw_lines, batch in read_batches(path):
            for line in raw_lines:
                digest.update(line)
            for row in batch:
                require(row['draw'] == count, 'Source draw sequence changed')
                count += 1
                inner = 1. < row['q'] < 1.1
                weight = row['log_importance_weight']
                if weight is None:
                    for key, invalid in [('hard_invalid', not row['hard_valid']),
                        ('capture_invalid', not row['capture_valid']), ('region_invalid', not row['region_valid']),
                        ('out_of_inner_window', not inner)]:
                        zero_flags[key] += int(invalid)
                    continue
                require(math.isfinite(weight), 'Nonfinite original fitting weight')
                require(row['hard_valid'] and row['capture_valid'] and row['region_valid'] and inner,
                        'Positive source weight violates original physical masks')
                require(row['latent_radius'] <= .5+1e-10, 'Positive point is outside original radius-.5 support')
                rows.append(row)
                labels.append(job['id'])
                logs.append(weight)
        require(count == samples_per_population and logs, 'Truncated or empty source population')
        require(digest.hexdigest() == expected_hashes[str(path)], 'Source rows changed since pinned audit')
        hashes[str(path)] = digest.hexdigest()
        draws += count
        lw = np.asarray(logs)
        norm = np.exp(lw-logsumexp(lw))
        populations.append(dict(population=job['id'], seed=job['seed'], unconditional_draws=count,
            positive_rows=len(logs), zero_rows=count-len(logs), zero_mask_counts_nonexclusive=zero_flags,
            original_logQ=float(logsumexp(logs)-math.log(count)), original_weight_ESS=float(1/(norm@norm)),
            largest_normalized_weight=float(norm.max()), samples_path=str(path), samples_sha256=digest.hexdigest()))
    require(draws == expected['draws'] and len(rows) == expected['nonzero'], 'Original source N or zero counts differ')
    logw = np.asarray([r['log_importance_weight'] for r in rows], float)
    logq = float(logsumexp(logw)-math.log(draws))
    near(logq, expected['logQ'])
    weights = np.exp(logw-logsumexp(logw))
    ess = float(1/(weights@weights)); maximum = float(weights.max())
    near(ess, expected['weight_ESS']); near(maximum, expected['maximum_fraction'])
    for population, reported in zip(populations, expected.get('population_logQ_values', [])):
        near(population['original_logQ'], reported)
    reconstruction = dict(original_logQ=logq, original_weight_ESS=ess,
        original_maximum_fraction=maximum, original_full_N=draws, positive_rows=len(rows),
        zero_rows=draws-len(rows), reference_logQ=expected['logQ'],
        absolute_logQ_error=abs(logq-expected['logQ']))
    return rows, np.asarray(labels), logw, populations, hashes, reconstruction


def conditional_fit(rows, labels, logw, model, cfg):
    poses = [row['pose'] for row in rows]
    relative = relative_poses(poses, cfg['fixed_poses'][0])
    ell, anchor = model['angular_length'], model['anchors'][0]
    x = coordinates(relative, anchor, ell)
    floor = geometric_floor(ell, translation_std=.05, angle_axis_scale_deg=.1)
    fitted = weighted_fit(x, logw, floor)
    models = {'weighted': model_from_fit(fitted, anchor, model['shape_sha256'], 'single', ell)}
    for sd in (.1, .2):
        comparison = copy.deepcopy(model)
        comparison['covariances'] = (np.asarray(model['covariances'])*sd**2).tolist()
        models[f'geometry_sd_{sd}'] = comparison
    densities = {name: Density(m).evaluate(relative)[0] for name, m in models.items()}
    independent = chart_log_density(x, fitted, 'single', ell)
    errors = dict(chart=near(independent, densities['weighted']),
        laboratory=near(independent, Density(laboratory_model(models['weighted'], cfg['fixed_poses'][0])).evaluate(poses)[0]))
    robustness = population_diagnostics(x, logw, labels, floor, ell,
        np.asarray(model['covariances'][0]), fitted, densities)
    require(len(robustness) == 16, 'All 16 leave-one-population-out folds required')
    inverse = np.linalg.inv(np.linalg.cholesky(model['covariances'][0]))
    weights = np.exp(logw-logsumexp(logw))
    result = dict(angular_length=ell, floor=floor, fit=fitted,
        geometric_anchor=anchor, geometric_covariance=model['covariances'][0],
        geometry_scaled_mean=inverse@fitted['mean'],
        geometry_scaled_covariance=inverse@fitted['covariance']@inverse.T,
        geometry_scaled_covariance_eigenvalues=eigvalsh(fitted['covariance'], np.asarray(model['covariances'][0])),
        pooled_conditional_expected_log_proposal_density={k:float(weights@v) for k,v in densities.items()},
        population_robustness=robustness, independent_density_errors=errors,
        leave_one_population_out_gain_ranges={name:dict(minimum=min(p['leave_one_population_out_gain_vs_geometry'][name] for p in robustness),
            maximum=max(p['leave_one_population_out_gain_vs_geometry'][name] for p in robustness))
            for name in ('geometry_sd_0.1','geometry_sd_0.2')})
    return models, result


def fit(out):
    out = Path(out).resolve()
    require(not out.exists(), 'Fresh fit output required')
    started = time.process_time()
    require(sha(ASSESSMENT) == ASSESSMENT_SHA256, 'Pinned shoulder assessment changed')
    assessment = read(ASSESSMENT)
    direct_path = Path(assessment['direct_reference']['path'])
    require(sha(direct_path) == assessment['direct_reference']['sha256'], 'Pinned direct assessment changed')
    direct = read(direct_path)
    selected = select_campaigns(assessment, direct)
    input_hashes = {str(ASSESSMENT):ASSESSMENT_SHA256, str(direct_path):sha(direct_path)}
    inputs = {'source-assessment.json':ASSESSMENT, 'direct-assessment.json':direct_path}
    for label, path, audited in [('source', ASSESSMENT, assessment), ('direct', direct_path, direct)]:
        for name, digest in audited['archived_sha256'].items():
            archived = path.parent/'provenance'/name
            require(sha(archived) == digest, f'Pinned {label} assessment archive changed: {name}')
            input_hashes[str(archived)] = digest
            inputs[f'{label}-audit-archive__{name}'] = archived
    per_center, all_models, sample_hashes = {}, {}, {}
    physical = None
    all_seeds = []
    for name in CENTERS:
        record = selected[name]
        reference = Path(record['root'])
        print(f'Fitting {name} from original whole R=.5 stream', flush=True)
        # Verify all consumed campaign artifacts, including the archived
        # executable and configuration; historical and R=.25 rows are unused.
        for path, digest in record['input_sha256'].items():
            require(sha(path) == digest, 'Pinned source campaign input changed: '+path)
            input_hashes[str(Path(path).resolve())] = digest
        master = read(reference/'manifest.json')
        require(master['lambda_ratio'] == 64. and master['cloud_replicates'] == 2, 'Original cloud allocation changed')
        for filename, digest in master['archive_sha256'].items():
            path = reference/'provenance'/filename
            require(sha(path) == digest, 'Original execution archive changed: '+str(path))
            input_hashes[str(path)] = digest
            inputs[f'{name}-execution__{filename}'] = path
        cfg = read(reference/'provenance/config.json')
        region = read(reference/'provenance/region.json')
        model = region['gaussian_chart']
        shape_hash = sha(reference/'provenance/shape.json')
        current = {key:cfg[key] for key in PHYSICAL}
        if physical is None:
            physical = current
        require(current == physical and len(cfg['fixed_poses']) == 2 and cfg['depletant_radius'] == 1.5
                and cfg['reservoir_density'] == .035 and cfg['capture_radius'] == 18., 'AB physical target changed')
        check_region(region, cfg, shape_hash)
        require(master['region_sha256'] == sha(reference/'provenance/region.json'), 'Original source region changed')
        archived_model = ASSESSMENT.parent/'provenance'/f'model-{name}.json'
        require(model == read(archived_model), 'Fitting chart differs from pinned geometric chart')
        if name == 'direct':
            peak_path = direct_path.parent/'provenance/selected-pose.json'
            require(sha(peak_path) == direct['selection']['selected_pose_sha256'], 'Direct selected peak changed')
            peak = read(peak_path)['pose']
            inputs['selected-direct.json'] = peak_path
        else:
            peak = assessment['selection'][name]['pose']
        geometry, _ = geometry_model_audit(model, region, cfg, peak)
        rows, labels, logw, populations, hashes, reconstructed = collect_rows(master, record['input_sha256'], record['physical']['full'])
        sample_hashes.update(hashes); input_hashes.update(hashes)
        all_seeds.extend(p['seed'] for p in populations)
        models, details = conditional_fit(rows, labels, logw, model, cfg)
        near(details['fit']['normalized_weight_ESS'], reconstructed['original_weight_ESS'])
        near(details['fit']['largest_normalized_weight'], reconstructed['original_maximum_fraction'])
        all_models[name] = models
        per_center[name] = dict(source=str(reference), source_full_N=reconstructed['original_full_N'],
            positive_rows=len(rows), zero_rows=reconstructed['zero_rows'], source_populations=populations,
            source_physical_full=record['physical']['full'], original_weight_reconstruction=reconstructed,
            chart_geometry=geometry, geometric_model_sha256=sha(archived_model),
            source_manifest_sha256=sha(reference/'manifest.json'), source_region_sha256=sha(reference/'provenance/region.json'),
            **details)
        inputs[f'{name}-source-manifest.json'] = reference/'manifest.json'
        inputs[f'{name}-source-all-row-audit.json'] = Path(record['original_audit_path'])
    require(len(set(all_seeds)) == 48, 'Original random streams repeat across centers')
    inputs.update(local_dependencies([Path(__file__), ROOT/'tools/test_shoulder_local_guides.py']))
    for path in inputs.values():
        input_hashes[str(path.resolve())] = sha(path)
    (out/'provenance').mkdir(parents=True)
    for name, path in inputs.items():
        (out/'provenance'/name).write_bytes(path.read_bytes())
    model_hashes, comparison_hashes = {}, {}
    for name, models in all_models.items():
        write(out/f'model-{name}.json', models['weighted'])
        model_hashes[name] = sha(out/f'model-{name}.json')
        per_center[name]['model_sha256'] = model_hashes[name]
        per_center[name]['model_path'] = str(out/f'model-{name}.json')
        comparison_hashes[name] = {}
        for comparison in ('geometry_sd_0.1', 'geometry_sd_0.2'):
            path = out/f'model-{name}-{comparison}.json'
            write(path, models[comparison])
            comparison_hashes[name][comparison] = sha(path)
        per_center[name]['comparison_model_sha256'] = comparison_hashes[name]
    report = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        no_new_physical_draws=True, no_new_geometry_draws=True, deterministic_fit=True,
        physical=physical, original_q_window=WINDOW, source_ball_radius_A=.5,
        center_order=list(CENTERS), shape_sha256=all_models['direct']['weighted']['shape_sha256'],
        fit_rule=FIT_RULE, family='single', floor_rule='diag(.05^2 I3, [ell*tan(.1 degree/2)]^2 I3)',
        geometry_comparison_sds=[.1,.2], per_center=per_center,
        predictive_scope='All 16 leave-one-population-out weighted log-density scores condition on each original whole ball and physical masks. Training sets overlap. These are proposal diagnostics, not physical errors, unbiased likelihood estimates, equilibrium evidence, or unseen-tail bounds.',
        scope='Each Gaussian is globally normalized, but its fitted moments describe only its own original radius-.5 ball intersected with strict 1<q<1.1, capture and AB hard/depletion masks. Fits do not describe entire basins. Priority-assigned-region masses may allocate a later atlas but do not select or reweight fitting rows.',
        input_sha256=input_hashes, source_sample_sha256=sample_hashes,
        archived_sha256={name:sha(out/'provenance'/name) for name in inputs},
        output_model_sha256=model_hashes, comparison_model_sha256=comparison_hashes,
        fitting_CPU_seconds=time.process_time()-started)
    for path, digest in input_hashes.items():
        require(sha(path) == digest, 'Input changed during deterministic fit: '+path)
    write(out/'report.json', serial(report))
    write(out/'freeze.json', dict(report_sha256=sha(out/'report.json'), model_sha256=model_hashes,
        comparison_model_sha256=comparison_hashes, archived_sha256=report['archived_sha256'],
        source_sample_sha256=sample_hashes, source_sha256=input_hashes))
    print(dict(output=str(out), fit_ESS={name:per_center[name]['fit']['normalized_weight_ESS'] for name in CENTERS}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    fit(parser.parse_args().out)
