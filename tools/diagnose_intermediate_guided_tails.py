#!/usr/bin/env python3
"""Read-only branch, density and paired-cloud diagnostics for fresh guided tails."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import logsumexp

from audit_shoulder_mis_independently import independent_densities, near, read, sha
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_smc_normalizer_atlas import Density, relative_poses

ROOT = Path(__file__).resolve().parents[1]
EDGES = (2., 2.5, 3., 4., 5.)
BRANCHES = ('weighted', 'geometry', 'product', 'cube')
WINDOW = dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False)
PHYSICAL = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'metadata')


def branch(row, name):
    if row['proposal_family'] == 'cover':
        assert row['guide_branch'] is None and row['guide_component'] is None
        return 'product'
    assert row['proposal_family'] == 'guide'
    if row['guide_branch'] == 'uniform':
        assert row['guide_component'] is None
        return 'cube'
    assert row['guide_branch'] == 'learned'
    if name == 'geometry':
        assert row['guide_component'] == 0
        return 'geometry'
    assert row['guide_component'] in (0, 1)
    return ('weighted', 'geometry')[row['guide_component']]


def summarize(selected, n):
    """Every excluded row remains an exact zero in the common full N."""
    if not selected:
        return dict(draws=n, nonzero=0, logQ=None, weight_ESS=0., maximum_fraction=None,
                    row_RSE=None, paired_cloud_variance_fraction=None)
    logs = np.array([r['log_importance_weight'] for r in selected]); offset = float(logs.max())
    weights = np.exp(logs-offset); total = float(weights.sum())
    variance = (float(weights@weights)-total*total/n)/(n-1)/n
    clouds = np.array([r['cloud_log_weights'] for r in selected])
    cloud_values = np.exp(clouds-np.array([r['log_proposal_density'] for r in selected])[:, None]-offset)
    noise = float(np.sum((cloud_values[:, 0]-cloud_values[:, 1])**2)/4/n**2)
    return dict(draws=n, nonzero=len(selected), logQ=float(math.log(total/n)+offset),
                weight_ESS=float(total**2/(weights@weights)), maximum_fraction=float(weights.max()/total),
                row_RSE=float(math.sqrt(variance)/(total/n)), log_variance_of_mean=float(math.log(variance)+2*offset),
                paired_cloud_variance_fraction=float(noise/variance),
                maximum_log_boltzmann_mean=max(r['log_boltzmann_mean'] for r in selected),
                scope='Observed iid moments over full unconditional N. Random branch labels are not fixed-quota strata. Unvisited tails are not bounded.')


def load_guided(name, prep, region, seen_seeds):
    root = ROOT/f'runs/ab-intermediate-{name}-4x16384-l64-20260920'
    master = read(root/'manifest.json'); cfg = read(root/'provenance/config.json'); model = read(root/'provenance/guide-model.json')
    assert sha(root/'provenance/guide-model.json') == sha(prep/f'model-{name}.json')
    for filename, digest in master['archive_sha256'].items(): assert sha(root/'provenance'/filename) == digest
    for key in PHYSICAL: assert cfg[key] == read(prep/'config.json')[key]
    assert master['q_window'] == WINDOW and master['total_unconditional_draws'] == 65536
    valid = []; proposed = []; sources = []; total = 0; cpu = 0.; density_error = q_error = 0.
    for job in master['jobs']:
        path = Path(job['output']); run, summary = read(path/'manifest.json'), read(path/'summary.json')
        assert summary['complete'] and run['samples'] == summary['samples'] == 16384
        assert run['seed'] == job['seed'] and run['seed'] not in seen_seeds; seen_seeds.add(run['seed'])
        assert run['config_sha256'] == sha(root/'provenance/config.json')
        assert run['shape_sha256'] == master['shape_sha256'] == sha(root/'provenance/shape.json')
        assert run['executable_sha256'] == master['binary_sha256'] == sha(root/'provenance/native-region-normalizer')
        assert run['guide']['model_sha256'] == sha(prep/f'model-{name}.json')
        assert run['guide']['weight'] == .75 and run['guide']['uniform_probability'] == .05 and run['guide']['anchor_index'] == 0
        assert run['lambda_ratio'] == 64 and run['cloud_replicates'] == 2 and run['activity'] == .035 and run['depletant_radius'] == 1.5
        assert run['q_window'] == WINDOW
        digest = hashlib.sha256(); count = 0
        with (path/'samples.jsonl').open('rb') as handle:
            while True:
                raw = []
                for _ in range(4096):
                    line = handle.readline()
                    if not line: break
                    digest.update(line); raw.append(json.loads(line))
                if not raw: break
                lg, lc, radii, jac, lv, q, capture = independent_densities(raw, cfg, model, region, run['guide'], run['cover_mixture'])
                density_error = max(density_error, near(lg, [r['log_proposal_density'] for r in raw]))
                q_error = max(q_error, near(q, [r['q'] for r in raw]))
                for index, row in enumerate(raw):
                    assert row['draw'] == count; count += 1
                    label = branch(row, name); inside = 2 <= q[index] < 5
                    proposed.append(dict(branch=label, q=float(q[index])))
                    reason = row.get('zero'); assert reason in (None, 'q', 'capture', 'hard')
                    assert (reason == 'q') == (not inside)
                    if inside: assert (reason == 'capture') == (not capture[index])
                    if reason is not None:
                        assert row.get('log_importance_weight') is None
                        continue
                    assert inside and capture[index] and math.isfinite(lc[index])
                    clouds = row['cloud_log_weights']; assert len(clouds) == 2
                    near(clouds, .035*row['lower_volume']+np.array(row['cloud_overlap_counts'])*math.log1p(1/64))
                    logf = float(np.logaddexp(*clouds)-math.log(2))
                    near(logf, row['log_boltzmann_mean']); near(logf-lg[index], row['log_importance_weight'])
                    near(-lg[index], row['log_hard_weight'])
                    valid.append(dict(row, source_kind=name, population=path.name, seed=job['seed'],
                                      actual_source_branch=label, source_samples_path=str(path/'samples.jsonl')))
        assert count == 16384 and digest.hexdigest() == summary['samples_sha256']
        total += count; cpu += summary['cpu_seconds']
        sources.append(dict(population=path.name, seed=job['seed'], draws=count, samples_sha256=digest.hexdigest(),
                            samples_path=str(path/'samples.jsonl'), manifest_sha256=sha(path/'manifest.json'),
                            summary_sha256=sha(path/'summary.json'), sampler_CPU_seconds=summary['cpu_seconds']))
    assert total == 65536
    result = dict(root=str(root), manifest_sha256=sha(root/'manifest.json'), all_rows_audited=total,
                  source_density_max_error=density_error, original_q_max_error=q_error, sampler_CPU_seconds=cpu,
                  raw_sources=sources, all=summarize(valid, total), branches={}, q_bands=[])
    for label in BRANCHES:
        rows = [row for row in valid if row['actual_source_branch'] == label]; summary = summarize(rows, total)
        result['branches'][label] = dict(unconditional_branch_draws=sum(p['branch'] == label for p in proposed),
                                         valid_rows=len(rows), observed_total_weight_fraction=math.exp(summary['logQ']-result['all']['logQ']) if rows else 0.,
                                         weight_summary=summary)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        rows = [row for row in valid if lo <= row['q'] < hi]; summary = summarize(rows, total)
        record = dict(lower=lo, upper=hi, all=summary, branches={})
        for label in BRANCHES:
            sub = [row for row in rows if row['actual_source_branch'] == label]; stats = summarize(sub, total)
            record['branches'][label] = dict(proposed_in_band=sum(p['branch'] == label and lo <= p['q'] < hi for p in proposed),
                valid_rows=len(sub), observed_band_weight_fraction=math.exp(stats['logQ']-summary['logQ']) if sub else 0.,
                observed_total_weight_fraction=math.exp(stats['logQ']-result['all']['logQ']) if sub else 0., fullN_log_contribution=stats['logQ'])
        result['q_bands'].append(record)
    result['populations'] = [dict(population=source['population'], all=summarize([row for row in valid if row['population'] == source['population']], source['draws'])) for source in sources]
    return result, valid, cfg, run


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, default=ROOT/'runs/ab-intermediate-guided-comparison-qualified-20260920/comparison.json')
    args = parser.parse_args(); out = args.out.resolve(); assert not out.exists()
    started = time.process_time(); prep = ROOT/'runs/ab-intermediate-guide-preparation-20260920'
    freeze = read(prep/'freeze.json'); assert sha(prep/'source-valid-poses.jsonl') == freeze['source_valid_poses_sha256']
    region = read(prep/'provenance/source-region.json'); models = {name: read(prep/f'model-{name}.json') for name in ('mixture', 'geometry')}
    assert all(sha(prep/f'model-{name}.json') == freeze['model_sha256'][name] for name in models)
    training = [json.loads(line) for line in (prep/'source-valid-poses.jsonl').read_text().splitlines()]
    assert len(training) == 3335
    direct_top = []
    for row in sorted(training, key=lambda r: r['log_importance_weight'], reverse=True)[:16]:
        clouds = [c['log_weight'] for c in row['clouds']]
        direct_top.append(dict(row, cloud_log_weights=clouds, log_proposal_density=-row['log_hard_weight'],
                               log_boltzmann_mean=float(np.logaddexp(*clouds)-math.log(2)), actual_source_branch='direct_cover', source_kind='training_reference'))
    results = {}; selected = list(direct_top); seen_seeds = set()
    for name in models:
        result, valid, cfg, run = load_guided(name, prep, region, seen_seeds)
        results[name] = result
        selected += sorted(valid, key=lambda row: row['log_importance_weight'], reverse=True)[:16]
    # Pointwise responsibilities differ from the actual randomly selected branch.
    poses = [row['pose'] for row in selected]; raw = [dict(pose=pose) for pose in poses]
    values = {}; maximum_cross_density_error = 0.
    for name, model in models.items():
        lg, lc, radii, jac, lv, q, capture = independent_densities(raw, cfg, model, region, run['guide'], run['cover_mixture'])
        gaussian, norms, component_logs = Density(model).evaluate(relative_poses(poses, cfg['fixed_poses'][0]))
        product_term = np.full(len(poses), math.log(.25/run['cover']['volume']))
        cube_term = np.full(len(poses), math.log(.75*.05/np.prod(run['guide']['cube_lengths'])))
        learned_terms = math.log(.75*.95)+component_logs
        terms = np.column_stack((learned_terms, product_term, cube_term))
        maximum_cross_density_error = max(maximum_cross_density_error, near(lg, logsumexp(terms, axis=1)))
        assert np.isfinite(lc).all() and capture.all() and np.all((q >= 2) & (q < 5))
        values[name] = dict(log_density=lg, cover_log_density=lc, radii=norms,
                            responsibilities=np.exp(terms-lg[:, None]))
    atom = AtomUnionAudit(read(prep/'provenance/shape.json'), cfg['fixed_poses']); audited = []
    for index, row in enumerate(selected):
        gaps = atom.gaps(row['pose']); assert all(gap >= 0 for gap in gaps)
        result = dict(source_kind=row['source_kind'], population=row['population'], seed=row['seed'], draw=row['draw'], q=row['q'],
                      pose=row['pose'], actual_source_branch=row['actual_source_branch'],
                      original_log_importance_weight=row['log_importance_weight'], log_boltzmann_mean=row['log_boltzmann_mean'],
                      original_cloud_log_weights=row['cloud_log_weights'], minimum_atomic_gap_by_neighbor_A=gaps,
                      guide_densities={})
        for name, value in values.items():
            labels = ['weighted', 'geometry', 'product', 'cube'] if name == 'mixture' else ['geometry', 'product', 'cube']
            result['guide_densities'][name] = dict(log_full_density=float(value['log_density'][index]),
                log_guide_to_complete_cover_ratio=float(value['log_density'][index]-value['cover_log_density'][index]),
                mahalanobis_radii=value['radii'][index].tolist(),
                conditional_branch_density_fractions={label: float(x) for label, x in zip(labels, value['responsibilities'][index])})
        if row['source_kind'] != 'training_reference':
            result['observed_weight_fraction_in_source_campaign'] = math.exp(row['log_importance_weight']-math.log(65536)-results[row['source_kind']]['all']['logQ'])
        audited.append(result)
    comparison_path = args.comparison.resolve(); comparison = read(comparison_path)
    assert comparison['physical'] == {key: cfg[key] for key in PHYSICAL}
    for name, result in results.items():
        primary = comparison['guided_campaigns'][Path(result['root']).name]
        for key in ('logQ', 'weight_ESS', 'maximum_fraction', 'row_RSE', 'paired_cloud_variance_fraction'):
            near(result['all'][key], primary['physical']['all'][key])
        for index, record in enumerate(result['q_bands']): near(record['all']['logQ'], primary['physical'][str(index)]['logQ'])
    report = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(), physical={key: cfg[key] for key in PHYSICAL},
                  q_window=WINDOW, bands=EDGES, campaigns=results, top_pose_density_checks=audited,
                  selected_poses=len(audited), maximum_cross_density_error=maximum_cross_density_error,
                  minimum_selected_atomic_gap_A=min(min(row['minimum_atomic_gap_by_neighbor_A']) for row in audited),
                  input_sha256=dict(freeze=sha(prep/'freeze.json'), training_poses=sha(prep/'source-valid-poses.jsonl'),
                                    source_comparison=sha(comparison_path), shape=sha(prep/'provenance/shape.json'),
                                    **{f'model-{name}': sha(prep/f'model-{name}.json') for name in models}),
                  helper_sha256=sha(__file__), CPU_seconds=time.process_time()-started,
                  scope='No new poses, clouds, fitting, replacement importance weights, or pooled normalizers. All branch contributions retain full N and the full original hybrid density. Random branch statistics are proposal diagnostics, not separately normalized physical basin masses. Training-reference extremes are in-sample checks. Pointwise density and finite-sample cloud decomposition do not establish integrated coverage or convergence.')
    out.mkdir(parents=True); (out/'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for filename in ('diagnose_intermediate_guided_tails.py', 'audit_shoulder_mis_independently.py', 'prepare_native_confirmation_atlas.py', 'prepare_smc_normalizer_atlas.py'):
        shutil.copy2(ROOT/'tools'/filename, out/filename)
    print(json.dumps(dict(campaigns={name: {key: value[key] for key in ('all', 'branches', 'q_bands')} for name, value in results.items()},
                         extrema=[{key: row[key] for key in ('source_kind', 'population', 'draw', 'q', 'actual_source_branch', 'original_cloud_log_weights', 'guide_densities')} for row in audited[:18]],
                         maximum_cross_density_error=maximum_cross_density_error, CPU_seconds=report['CPU_seconds']), indent=2))


if __name__ == '__main__':
    main()
