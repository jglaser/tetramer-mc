#!/usr/bin/env python3
"""Extend a hash-bound finite-region comparison; reuse saved audits, never rerun them."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD_COMPARISON = ROOT/'runs/mobile-competing-reference-comparison-20260921'
OLD_ANALYSIS_SHA = 'd6957829ec621b735c87921140026c2049d0d2642dc938c6e121a484a4fda3d9'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def check_partition(regions):
    require([(r.get('minimum_mahalanobis_radius', 0.), r['mahalanobis_radius']) for r in regions]
            == [(0., 3.), (3., 5.), (5., 8.), (8., 12.)], 'Expected exact four disjoint strata')
    keys = ('gaussian_chart', 'fixed_neighbor', 'physical_fixed_neighbors', 'capture_center',
            'capture_radius', 'shape_sha256', 'activity', 'depletant_radius', 'physical_metric',
            'minimum_original_q', 'minimum_original_q_inclusive', 'maximum_original_q_inclusive')
    for region in regions:
        require('maximum_original_q' not in region, 'Unexpected upper q cutoff')
        require(all(region[key] == regions[0][key] for key in keys), 'Stratum physical/chart law changed')
        require(region['minimum_original_q'] == 1. and region['minimum_original_q_inclusive'] is False,
                'Competitor q domain changed')


def archived_math(old):
    require(sha(old/'analysis.json') == OLD_ANALYSIS_SHA, 'Previous finite comparison changed')
    prior = read(old/'analysis.json')
    require(prior['complete'] and sha(old/'analyzer.py') == prior['analyzer_sha256'], 'Previous analyzer mismatch')
    for name, digest in read(old/'freeze.json').items():
        require(sha(old/name) == digest, 'Previous comparison freeze changed')
    for region in prior['regions']:
        for name, digest in region['source_sha256'].items():
            require(sha(name) == digest, 'Previously audited input changed')
    spec = importlib.util.spec_from_file_location('frozen_competing_region_math', old/'analyzer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, prior


def load_new_region(root, protocol, status, old_config, math_module):
    root = Path(root).resolve()
    spec = next(c for c in protocol['commands'] if c['region'] == root.name)
    terminal = next(r for r in status['jobs'] if r['region'] == root.name)
    require(terminal['exit_code'] == terminal['audit_exit_code'] == 0, 'Physical run or saved audit failed')
    require(terminal['assessment_sha256'] == sha(root/'assessment/analysis.json'), 'Saved assessment changed')
    manifest, assessed = read(root/'manifest.json'), read(root/'assessment/analysis.json')
    for name, digest in manifest['archive_sha256'].items():
        require(sha(root/'provenance'/name) == digest, 'Archived population input changed')
    region = read(root/'provenance/region.json')
    require(sha(root/'provenance/region.json') == assessed['region_sha256'] == spec['region_sha256'], 'Frozen region differs')
    config = read(root/'provenance/config.json')
    stripped = {k: v for k, v in config.items() if k != 'shape'}
    require(stripped == {k: v for k, v in old_config.items() if k != 'shape'}, 'Physical configuration differs')
    require(sha(root/'provenance/shape.json') == region['shape_sha256'], 'Hard shape differs')
    require(manifest['cloud_replicates'] == 2 and manifest['lambda_ratio'] == 64., 'Cloud law changed')
    require(len(manifest['jobs']) == len(spec['expected_jobs']) == 4, 'Missing independent population')
    populations, all_z, all_h, hashes = [], [], [], {}
    for job, expected in zip(manifest['jobs'], spec['expected_jobs']):
        require(all(job[k] == expected[k] for k in ('id', 'seed', 'samples')), 'Population budget or seed changed')
        directory = Path(job['directory'])
        require(directory.resolve() == root/'runs'/job['id'], 'Population path escaped campaign')
        summary = read(directory/'summary.json')
        pmanifest = read(directory/'manifest.json')
        require(summary['manifest'] == pmanifest and summary['complete'], 'Population not complete')
        require(pmanifest['executable_sha256'] == protocol['source_sha256']['latent-region-normalizer'] and
                pmanifest['source_bundle_sha256'] == protocol['source_sha256']['source-bundle.json'], 'Compiled engine changed')
        require(pmanifest['seed'] == job['seed'] and pmanifest['samples'] == job['samples'], 'Unconditional count changed')
        require(pmanifest['config_sha256'] == sha(root/'provenance/config.json') and
                pmanifest['region_sha256'] == spec['region_sha256'], 'Population physical target changed')
        audited = next(a for a in assessed['populations'] if a['id'] == job['id'])
        require(sha(directory/'samples.jsonl') == audited['samples_sha256'] == summary['samples_sha256'], 'Saved audited rows changed')
        z, h = [], []
        with (directory/'samples.jsonl').open() as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                require(row['draw'] == i, 'Missing unconditional row')
                z.append(-math.inf if row['log_importance_weight'] is None else row['log_importance_weight'])
                h.append(-math.inf if row['log_hard_weight'] is None else row['log_hard_weight'])
        require(len(z) == job['samples'], 'Invalid unconditional denominator')
        values = math_module.paired_moments(z, h)
        math_module.check_estimate(values, audited['estimate'], audited['hard_region'])
        populations.append(dict(id=job['id'], seed=job['seed'], **values))
        all_z.extend(z); all_h.extend(h)
        for name in ('summary.json', 'manifest.json', 'samples.jsonl'):
            hashes[str(directory/name)] = sha(directory/name)
    combined = math_module.paired_moments(all_z, all_h)
    math_module.check_estimate(combined, assessed['estimate'], assessed['hard_region'])
    population = math_module.paired_moments([p['log_Qz'] for p in populations], [p['log_Q0'] for p in populations])
    return dict(name=root.name, campaign=str(root), region_sha256=sha(root/'provenance/region.json'),
                row_uncertainty=combined, population_uncertainty=population, populations=populations,
                sampler_cpu_seconds=assessed['sampler_cpu_seconds'], audited_weight_concentration=assessed['estimate'],
                source_sha256=hashes, assessment_sha256=sha(root/'assessment/analysis.json'))


def analyze(campaign, old, out):
    campaign, old, out = [Path(p).resolve() for p in (campaign, old, out)]
    require(not out.exists(), 'Use a fresh comparison directory')
    math_module, previous = archived_math(old)
    protocol, status = read(campaign/'protocol.json'), read(campaign/'status.json')
    require(status['complete'] and not status['running'], 'New campaign incomplete')
    require(status['protocol_sha256'] == sha(campaign/'protocol.json') and
            status['driver_sha256'] == sha(campaign/'driver.py'), 'Frozen protocol or driver changed')
    for name, digest in protocol['source_sha256'].items():
        require(sha(campaign/'provenance'/name) == digest, 'Frozen dependency changed')
    for name, digest in protocol['input_sha256'].items():
        require(sha(campaign/name) == digest, 'Frozen campaign input changed')
    prior = {r['name']: r for r in previous['regions']}
    old_config = read(Path(prior['competitor-r3']['campaign'])/'provenance/config.json')
    regions = copy.deepcopy(previous['regions'])
    regions.extend(load_new_region(campaign/c['region'], protocol, status, old_config, math_module)
                   for c in protocol['commands'])
    seeds = [p['seed'] for r in regions for p in r['populations']]
    require(len(seeds) == len(set(seeds)), 'Independent populations reuse seeds')
    by_name = {r['name']: r for r in regions}
    names = ['competitor-r3', 'competitor-shell-3-5', 'competitor-shell-5-8', 'competitor-shell-8-12']
    check_partition([read(Path(by_name[n]['campaign'])/'provenance/region.json') for n in names])
    sums = [math_module.sum_independent_regions([by_name[n] for n in names[:i]], f'competitor-r{r}')
            for i, r in ((2, 5), (3, 8), (4, 12))]
    for i, combined in enumerate(sums, start=2):
        combined['stratum_fractions_Qz'] = {n: math.exp(by_name[n]['row_uncertainty']['log_Qz']-combined['log_Qz']) for n in names[:i]}
    comparisons = [math_module.contrast(by_name['native-r4'], r) for r in sums]
    result = dict(schema='mobile-competing-outer-comparison-v1', complete=True,
                  previous_comparison_sha256=sha(old/'analysis.json'),
                  protocol_sha256=sha(campaign/'protocol.json'), status_sha256=sha(campaign/'status.json'),
                  regions=regions, derived_disjoint_sums=sums, finite_region_contrasts=comparisons,
                  scope=protocol['scope'], physical_audits_rerun=False,
                  warning='Observed row/population uncertainties cannot certify unseen high-weight contributions. '
                  'A finite radius extension is not whole-basin coverage or a bound on its omitted complement.')
    out.mkdir()
    shutil.copy2(Path(__file__), out/'analyzer.py')
    shutil.copy2(old/'analyzer.py', out/'reference_math.py')
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['Conditional finite-region extension, using already audited fixed-N weights.\n',
             '| Region | log Qz | row RSE | population RSE | log Q0 |',
             '|---|---:|---:|---:|---:|']
    for r in regions+sums:
        a, b = r['row_uncertainty'], r['population_uncertainty']
        lines.append(f"| {r['name']} | {a['log_Qz']:.6f} | {a['Qz_relative_SE']:.2%} | {b['Qz_relative_SE']:.2%} | {a['log_Q0']:.6f} |")
    lines.append('')
    for contrast in comparisons:
        c = contrast['row_uncertainty']
        lines.append(f"Native R4 versus {contrast['denominator_region']}: log mass ratio {c['log_Qz_ratio']:.6f} ± {c['log_Qz_ratio_SE']:.6f} observed row SE.\n")
    lines += [result['scope'], '', result['warning']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'freeze.json').write_text(json.dumps({p.name: sha(p) for p in out.iterdir()}, indent=2)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--old-comparison', type=Path, default=OLD_COMPARISON)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.campaign, args.old_comparison, args.out)
