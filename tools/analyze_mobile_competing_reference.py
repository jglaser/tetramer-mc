#!/usr/bin/env python3
"""Compare already audited finite-region integrals with paired ratio errors.

This reaggregates hash-bound saved weights; it never reruns a geometry/cloud
auditor or launches a physical calculation.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT/'runs/mobile-competing-reference-preparation-20260921'
PREPARATION_SHA = '6fee3d56e0f577a1e1d2e0ab869001adfd51f1a412a84d75d681bcd185b30de0'
PILOT = ROOT/'runs/mobile-competing-reference-pilot-20260921'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def paired_moments(log_z, log_zero):
    """Delta variance of paired fixed-N sample means, keeping invalid zeros."""
    z, h = np.asarray(log_z, dtype=float), np.asarray(log_zero, dtype=float)
    require(z.ndim == h.ndim == 1 and z.shape == h.shape and len(z) >= 2, 'Require paired records with at least two draws')
    require(not np.isnan(z).any() and not np.isnan(h).any() and not np.isposinf(z).any() and not np.isposinf(h).any(), 'Invalid log weights')
    require(np.array_equal(np.isfinite(z), np.isfinite(h)), 'Physical and hard weights must share invalid zeros')
    n = len(z)
    if not np.isfinite(z).any():
        return dict(draws=n, nonzero=0, log_Qz=None, log_Q0=None, log_enhancement=None,
                    covariance_relative=None, log_enhancement_SE=None,
                    unresolved='No nonzero observations; neither zero mass nor an upper bound is established.')
    total_z, total_h = logsumexp(z), logsumexp(h)
    wz, wh = np.exp(z-total_z), np.exp(h-total_h)
    rz, rh = wz-1/n, wh-1/n
    factor = n/(n-1)
    covariance = factor*np.array([[rz@rz, rz@rh], [rh@rz, rh@rh]])
    residual_variance = float(factor*np.sum((wz-wh)**2))
    log_z_mean, log_h_mean = float(total_z-math.log(n)), float(total_h-math.log(n))
    return dict(draws=n, nonzero=int(np.isfinite(z).sum()), log_Qz=log_z_mean, log_Q0=log_h_mean,
        Qz_relative_SE=float(math.sqrt(covariance[0, 0])), Q0_relative_SE=float(math.sqrt(covariance[1, 1])),
        Qz_ESS=float(1/(wz@wz)), Q0_ESS=float(1/(wh@wh)),
        largest_Qz_fraction=float(wz.max()), largest_Q0_fraction=float(wh.max()),
        covariance_relative=covariance.tolist(), log_enhancement=log_z_mean-log_h_mean,
        log_enhancement_SE=math.sqrt(residual_variance), enhancement_relative_SE=math.sqrt(residual_variance),
        paired_residual_variance=residual_variance,
        formula='Var_delta[log(Qz/Q0)] = N/(N-1) sum_i (Y_i/sum(Y)-H_i/sum(H))^2; both original unconditional denominators retained.',
        uncertainty_scope='Observed delta-method SE of a correlated ratio of means; not the mean of per-draw ratios and not a tail-coverage guarantee.')


def check_estimate(calculated, recorded_z, recorded_h):
    for actual, wanted in ((calculated['log_Qz'], recorded_z['logQ']), (calculated['log_Q0'], recorded_h['logQ'])):
        require(actual is None and wanted is None or actual is not None and wanted is not None and abs(actual-wanted) < 2e-10, 'Saved audited integral differs from hash-bound weights')
    require(calculated['draws'] == recorded_z['draws'] == recorded_h['draws'], 'Unconditional denominator changed')


def sum_independent_regions(regions, name):
    """Add integrals on disjoint strata, with independent estimator covariance."""
    require(len(regions) >= 2, 'A stratified sum requires at least two regions')
    values = [r['row_uncertainty'] for r in regions]
    require(all(v['log_Qz'] is not None for v in values), 'Cannot identify an unobserved stratum as zero mass')
    lz, lh = float(logsumexp([v['log_Qz'] for v in values])), float(logsumexp([v['log_Q0'] for v in values]))
    fz = np.exp([v['log_Qz']-lz for v in values])
    fh = np.exp([v['log_Q0']-lh for v in values])
    result = dict(name=name, source_regions=[r['name'] for r in regions], log_Qz=lz, log_Q0=lh, log_enhancement=lz-lh,
                  scope='Sum of independent disjoint finite-stratum integrals. Sample rows are not concatenated or treated as one uniform-ball sample.')
    for field in ('row_uncertainty', 'population_uncertainty'):
        cov = np.zeros((2, 2))
        for region, a, b in zip(regions, fz, fh):
            c = np.asarray(region[field]['covariance_relative'])
            cov += c*np.outer([a, b], [a, b])
        variance = float(cov[0, 0]+cov[1, 1]-2*cov[0, 1])
        result[field] = dict(log_Qz=lz, log_Q0=lh, log_enhancement=lz-lh, covariance_relative=cov.tolist(),
            Qz_relative_SE=math.sqrt(max(0., float(cov[0, 0]))), Q0_relative_SE=math.sqrt(max(0., float(cov[1, 1]))),
            log_enhancement_SE=math.sqrt(max(0., variance)), enhancement_relative_SE=math.sqrt(max(0., variance)))
    return result


def contrast(native, competitor):
    result = dict(numerator_region=native['name'], denominator_region=competitor['name'])
    for field in ('row_uncertainty', 'population_uncertainty'):
        a, b = native[field], competitor[field]
        if a['log_Qz'] is None or b['log_Qz'] is None:
            result[field] = dict(unresolved='At least one finite region has no nonzero observations.')
            continue
        result[field] = dict(log_Qz_ratio=a['log_Qz']-b['log_Qz'],
            log_Qz_ratio_SE=math.hypot(a['Qz_relative_SE'], b['Qz_relative_SE']),
            log_Q0_ratio=a['log_Q0']-b['log_Q0'],
            log_Q0_ratio_SE=math.hypot(a['Q0_relative_SE'], b['Q0_relative_SE']),
            log_enhancement_difference=a['log_enhancement']-b['log_enhancement'],
            log_enhancement_difference_SE=math.hypot(a['log_enhancement_SE'], b['log_enhancement_SE']))
    result['scope'] = 'Ratio of the named finite integrals only; regional sampling seeds are disjoint. The volume and enhancement contrasts are correlated, so their separate SEs must not be added to obtain the Qz-ratio SE.'
    return result


def verify_preparation(preparation):
    require(sha(preparation/'plan.json') == PREPARATION_SHA, 'Unexpected frozen preparation')
    plan, frozen = read(preparation/'plan.json'), read(preparation/'freeze.json')
    for name, digest in frozen.items():
        require(sha(preparation/name) == digest, 'Frozen preparation file changed: '+name)
    for name, digest in plan['input_sha256'].items():
        require(sha(preparation/'provenance'/name) == digest, 'Preparation input changed: '+name)
    cfg, shape = read(preparation/'config.json'), read(preparation/'provenance/shape.json')
    bound = max(np.linalg.norm(a['center'])+a['radius'] for a in shape['atoms'])
    require(cfg['capture_center'] == [0., 0., 0.] and cfg['capture_radius']+bound < cfg['metadata']['physical_sphere_radius_A'], 'Lab capture does not certify the original protein wall')
    require(cfg['fixed_poses'] == plan['physical_scaffold'], 'Exact fixed scaffold changed')
    require(sha(preparation/'provenance/shape.json') == plan['shape_sha256'], 'Shape differs from plan')
    return plan, cfg, float(bound)


def terminal_binding(root, name, preparation, shell_journal):
    assessment = root/'assessment/analysis.json'
    if shell_journal is not None:
        journal = Path(shell_journal).resolve()
        status = read(journal/'status.json' if journal.is_dir() else journal)
        require(status['complete'] and not status['running'] and status['physical_exit_code'] == status['audit_exit_code'] == 0, 'Shell calculation or its saved audit is incomplete')
        require(status['analysis_sha256'] == sha(assessment), 'Shell saved assessment changed')
        require(status['preparation_plan_sha256'] == sha(preparation/'plan.json'), 'Shell preparation binding differs')
        require(status['region_sha256'] == sha(preparation/f'region-{name}.json'), 'Shell region binding differs')
        require(status['source_protocol_sha256'] == sha(PILOT/'protocol.json') and status['first_stage_status_sha256'] == sha(PILOT/'status.json'), 'Shell first-stage provenance changed')
        argv = status['argv']
        require(Path(argv[argv.index('--out')+1]).resolve() == root, 'Shell journal belongs to another campaign')
        return dict(path=str(journal), status_sha256=sha(journal/'status.json' if journal.is_dir() else journal),
                    assessment_sha256=sha(assessment), expected_seeds=[int(argv[argv.index('--seed')+1])+1009*i for i in range(4)])
    parent = root.parent
    status, protocol = read(parent/'status.json'), read(parent/'protocol.json')
    require(status['complete'] and not status['running'], 'Parent campaign is incomplete')
    require(status['protocol_sha256'] == sha(parent/'protocol.json') and status['driver_sha256'] == sha(parent/'driver.py'), 'Parent protocol/driver changed')
    require(protocol['preparation_sha256'] == sha(preparation/'plan.json'), 'Parent preparation differs')
    require(status['assessments'][name] == sha(assessment), 'Saved completed assessment changed')
    statuses = [job for job in status['jobs'] if job['region'] == name]
    require(len(statuses) == 1 and statuses[0]['exit_code'] == statuses[0]['audit_exit_code'] == 0, 'Region or auditor failed')
    for filename, digest in protocol['source_sha256'].items():
        require(sha(parent/'provenance'/filename) == digest, 'Campaign source changed: '+filename)
    command = next(c for c in protocol['commands'] if c['region'] == name)
    return dict(path=str(parent), status_sha256=sha(parent/'status.json'), protocol_sha256=sha(parent/'protocol.json'),
                assessment_sha256=sha(assessment), expected_seeds=[j['seed'] for j in command['expected_jobs']])


def load_region(root, preparation, plan, cfg, bound, shell_journal=None):
    root = Path(root).resolve()
    manifest, assessed = read(root/'manifest.json'), read(root/'assessment/analysis.json')
    for filename, digest in manifest['archive_sha256'].items():
        require(sha(root/'provenance'/filename) == digest, 'Campaign archive changed: '+filename)
    region_path = root/'provenance/region.json'
    matches = [p for p in preparation.glob('region-*.json') if sha(p) == sha(region_path)]
    require(len(matches) == 1, 'Region is not exactly one frozen prepared region')
    name = matches[0].stem.removeprefix('region-')
    binding = terminal_binding(root, name, preparation, shell_journal)
    region = read(region_path)
    require(manifest['region_sha256'] == assessed['region_sha256'] == sha(region_path), 'Region identity differs')
    require(read(root/'provenance/input-config.json') == cfg, 'Input physical config differs from frozen preparation')
    expected = copy.deepcopy(cfg)
    expected['shape'] = str(root/'provenance/shape.json')
    require(read(root/'provenance/config.json') == expected, 'Resolved config changed beyond shape relocation')
    require(region['physical_fixed_neighbors'] == cfg['fixed_poses'] and region['physical_metric'] == cfg['metadata'], 'Region physical scaffold or metric differs')
    require(sha(root/'provenance/shape.json') == plan['shape_sha256'] and manifest['archive_sha256']['latent-region-normalizer'] == plan['reference_binary_sha256'], 'Physical shape or binary changed')
    require(manifest['lambda_ratio'] == 64. and manifest['cloud_replicates'] == 2, 'Unplanned cloud law')
    model = region['gaussian_chart']; fixed = region['fixed_neighbor']
    r = Rotation.from_quat(np.array(fixed['orientation'])[[1, 2, 3, 0]]).as_matrix()
    center = r@(np.array(model['anchors'][0]['position'])+model['means'][0][:3])+fixed['position']
    upper = np.linalg.norm(center)+region['mahalanobis_radius']*np.linalg.norm(np.linalg.cholesky(model['covariances'][0])[:3], ord=2)
    require(upper < cfg['capture_radius'], 'Finite region extends beyond the wall-safe capture')
    jobs = manifest['jobs']
    require(len(jobs) == 4 and len({j['id'] for j in jobs}) == 4 and all(j['samples'] == 8192 for j in jobs), 'Missing or unexpected populations')
    require(sorted(j['seed'] for j in jobs) == sorted(binding['expected_seeds']), 'Population seeds differ from the terminally bound plan')
    require(len(assessed['populations']) == 4 and {p['id'] for p in assessed['populations']} == {j['id'] for j in jobs}, 'Saved audit omits populations')
    populations, all_z, all_h, files, cpu = [], [], [], {}, 0.
    for job in jobs:
        directory = Path(job['directory'])
        require(directory.resolve() == root/'runs'/job['id'], 'Population directory is outside its campaign')
        status = read(root/f"{job['id']}-status.json")
        require(status['id'] == job['id'] and status['returncode'] == 0, 'Population did not complete successfully')
        summary, pop_manifest = read(directory/'summary.json'), read(directory/'manifest.json')
        require(summary['complete'] and summary['samples'] == job['samples'] and summary['manifest'] == pop_manifest, 'Population summary/manifest disagree')
        require(pop_manifest['seed'] == job['seed'] and pop_manifest['physical_fixed_neighbors'] == cfg['fixed_poses'], 'Population seed or scaffold differs')
        require(pop_manifest['region_sha256'] == sha(region_path) and pop_manifest['config_sha256'] == sha(root/'provenance/config.json'), 'Population region/config differs')
        require(pop_manifest['shape_sha256'] == plan['shape_sha256'] and pop_manifest['executable_sha256'] == plan['reference_binary_sha256'], 'Population physical provenance differs')
        require(pop_manifest['source_bundle_sha256'] == plan['reference_source_bundle_sha256'], 'Population compiled sources differ')
        require(pop_manifest['lambda_ratio'] == 64. and pop_manifest['cloud_replicates'] == 2 and pop_manifest['activity'] == cfg['reservoir_density'], 'Population bath/cloud law differs')
        for filename, key in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'), ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
            require(sha(directory/'provenance'/filename) == pop_manifest[key], 'Population archived input changed')
        audited = next(p for p in assessed['populations'] if p['id'] == job['id'])
        digest = sha(directory/'samples.jsonl')
        require(digest == audited['samples_sha256'] == summary['samples_sha256'] and audited['seed'] == job['seed'], 'Previously audited rows changed')
        z, h = [], []
        with (directory/'samples.jsonl').open() as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                require(row['draw'] == i, 'Missing/repeated unconditional draw')
                z.append(-math.inf if row['log_importance_weight'] is None else row['log_importance_weight'])
                h.append(-math.inf if row['log_hard_weight'] is None else row['log_hard_weight'])
        require(len(z) == job['samples'], 'Draw denominator differs')
        values = paired_moments(z, h)
        check_estimate(values, audited['estimate'], audited['hard_region'])
        populations.append(dict(id=job['id'], seed=job['seed'], **values))
        all_z.extend(z); all_h.extend(h)
        cpu += summary['sampler_cpu_seconds']
        for filename in ('manifest.json', 'summary.json', 'samples.jsonl', 'config.json'):
            files[str(directory/filename)] = sha(directory/filename)
    combined = paired_moments(all_z, all_h)
    check_estimate(combined, assessed['estimate'], assessed['hard_region'])
    population = paired_moments([-math.inf if p['log_Qz'] is None else p['log_Qz'] for p in populations],
                                [-math.inf if p['log_Q0'] is None else p['log_Q0'] for p in populations])
    require(combined['log_Qz'] is None or abs(combined['log_Qz']-population['log_Qz']) < 2e-10, 'Unequal population normalization')
    population['interpretation'] = 'SE from four equal-budget independent population means, retaining their paired numerator/denominator covariance; only four populations, so dispersion is itself uncertain.'
    return dict(name=name, campaign=str(root), region_sha256=sha(region_path), region_definition=region['definition'],
        binding=binding, manifest_sha256=sha(root/'manifest.json'), row_uncertainty=combined,
        population_uncertainty=population, populations=populations, sampler_cpu_seconds=cpu,
        audited_weight_concentration=assessed['estimate'], audited_hard_region=assessed['hard_region'],
        wall_proof=dict(maximum_region_center_radius_A=float(upper), capture_radius_A=cfg['capture_radius'],
                        shape_bound_A=bound, physical_sphere_radius_A=cfg['metadata']['physical_sphere_radius_A'],
                        minimum_guaranteed_wall_clearance_A=cfg['metadata']['physical_sphere_radius_A']-cfg['capture_radius']-bound),
        source_sha256=files)


def report_text(result):
    rows = ['Finite-region weights on the exact final two-body scaffold; all invalid zeros retained.\n',
            '| Region | log Qz | row RSE | population RSE | log Q0 | row RSE |',
            '|---|---:|---:|---:|---:|---:|']
    regions = result['regions']+result['derived_disjoint_sums']
    for region in regions:
        a, b = region['row_uncertainty'], region['population_uncertainty']
        if a['log_Qz'] is None:
            rows.append(f"| {region['name']} | unresolved | — | — | unresolved | — |")
        else:
            rows.append(f"| {region['name']} | {a['log_Qz']:.6f} | {a['Qz_relative_SE']:.2%} | {b['Qz_relative_SE']:.2%} | {a['log_Q0']:.6f} | {a['Q0_relative_SE']:.2%} |")
    rows += ['', '| Region | log(Qz/Q0) | paired row SE | paired population SE |', '|---|---:|---:|---:|']
    for region in regions:
        a, b = region['row_uncertainty'], region['population_uncertainty']
        if a['log_enhancement'] is not None:
            rows.append(f"| {region['name']} | {a['log_enhancement']:.6f} | {a['log_enhancement_SE']:.6f} | {b['log_enhancement_SE']:.6f} |")
    for comparison in result['finite_region_contrasts']:
        a = comparison['row_uncertainty']
        if 'unresolved' not in a:
            rows += ['', f"Native R4 versus {comparison['denominator_region']}: log finite-mass ratio {a['log_Qz_ratio']:.6f} ± {a['log_Qz_ratio_SE']:.6f} observed row SE. It decomposes into log accessible-volume ratio {a['log_Q0_ratio']:.6f} and regional depletion-enhancement difference {a['log_enhancement_difference']:.6f}. These two contributions are correlated."]
    for combined in result['derived_disjoint_sums']:
        rows += ['', f"The competitor 3–5 shell contributes {combined['outer_shell_Qz_fraction']:.2%} of the measured R5 mass and {combined['outer_shell_Q0_fraction']:.2%} of its accessible pose volume. {combined['coverage_warning']}"]
    rows += ['', 'Q0 measures accessible pose volume. Qz/Q0 is the ratio of paired means, with residual-based uncertainty; it is not exp(z times mean overlap). Population errors use the four separate equal-budget populations. All SEs describe observed contributions and cannot certify unseen high-weight poses.', '', result['scope'], '']
    return '\n'.join(rows)


def analyze(preparation, region_roots, out, shell_journal=None):
    preparation, out = Path(preparation).resolve(), Path(out).resolve()
    require(not out.exists(), 'Use a fresh comparison output directory')
    plan, cfg, bound = verify_preparation(preparation)
    roots = [Path(r).resolve() for r in region_roots]
    regions = []
    for root in roots:
        is_shell = read(root/'provenance/region.json').get('minimum_mahalanobis_radius', 0) > 0
        regions.append(load_region(root, preparation, plan, cfg, bound, shell_journal if is_shell else None))
    names = [r['name'] for r in regions]
    require(len(names) == len(set(names)) and 'native-r4' in names and 'competitor-r3' in names, 'Require unique native R4 and competitor R3 regions')
    seeds = [p['seed'] for r in regions for p in r['populations']]
    require(len(seeds) == len(set(seeds)), 'Regional population seeds overlap')
    by_name = {r['name']: r for r in regions}
    sums = []
    if 'competitor-shell-3-5' in by_name:
        core, shell = [read(preparation/f'region-{name}.json') for name in ('competitor-r3', 'competitor-shell-3-5')]
        require(core['gaussian_chart'] == shell['gaussian_chart'] and core['mahalanobis_radius'] == shell['minimum_mahalanobis_radius'] == 3. and shell['mahalanobis_radius'] == 5., 'Competitor strata are not the frozen disjoint partition')
        combined = sum_independent_regions([by_name['competitor-r3'], by_name['competitor-shell-3-5']], 'competitor-r5-from-disjoint-strata')
        combined['outer_shell_Qz_fraction'] = math.exp(by_name['competitor-shell-3-5']['row_uncertainty']['log_Qz']-combined['log_Qz'])
        combined['outer_shell_Q0_fraction'] = math.exp(by_name['competitor-shell-3-5']['row_uncertainty']['log_Q0']-combined['log_Q0'])
        combined['coverage_warning'] = 'The 3–5 shell carries additional observed mass. Its contribution does not bound the mass beyond radius5 or other chart neighborhoods; low between-population scatter does not override large row concentration/error.'
        sums.append(combined)
    result = dict(schema='mobile-competing-finite-region-comparison-v1', complete=True,
        preparation=str(preparation), preparation_plan_sha256=sha(preparation/'plan.json'),
        analyzer_sha256=sha(__file__), physical_scaffold=cfg['fixed_poses'], physical_metric=cfg['metadata'],
        shape_sha256=plan['shape_sha256'], regions=regions, derived_disjoint_sums=sums,
        finite_region_contrasts=[contrast(by_name['native-r4'], r) for r in regions+sums if r['name'] != 'native-r4'],
        scope='Conditional finite-region comparison on one exact observed scaffold. Native original-chart R4 is only part of q<=1; no old scaffold coverage percentage is transferred. The competitor regions exclude other attachments, chart tails and scaffold motion. No overall native/competitor basin ratio, equilibrium population, stationary efficiency, global free-energy gap, physical residence rate or assembly-cost claim. The original 18-Angstrom capture calculation excluded this competitor and is not its reference. Uniform direct-region weights do not contain the mobile proposal atlas density.')
    out.mkdir(parents=True)
    shutil.copy2(__file__, out/'analyzer.py')
    write(out/'analysis.json', result)
    (out/'report.md').write_text(report_text(result))
    write(out/'freeze.json', {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(out=str(out), regions=names, complete=True, analysis_sha256=sha(out/'analysis.json'))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, default=PREPARATION)
    parser.add_argument('--regions', type=Path, nargs='+', default=[PILOT/'native-r4', PILOT/'competitor-r3'])
    parser.add_argument('--shell-journal', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.preparation, args.regions, args.out, args.shell_journal)
