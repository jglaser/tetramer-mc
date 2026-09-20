#!/usr/bin/env python3
"""Freeze and run a small fixed-pose conditional Poisson-noise control.

The standalone Rust driver calls the existing public overlap-weight API.
There are no pose proposals, new physical kernels, fits or regional integrals.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2

from prepare_smc_normalizer_atlas import read, sha, write

ROOT = Path(__file__).resolve().parents[1]
CARGO = Path('/home/xvg/.cargo/bin/cargo')
RUSTC = Path('/home/xvg/.cargo/bin/rustc')
KERNEL_FILES = ('src/overlap_weight.rs', 'src/geometry.rs', 'src/math.rs', 'src/depletion.rs', 'src/native_region.rs', 'Cargo.toml', 'Cargo.lock')


def command(args, output):
    result = subprocess.run([str(a) for a in args], cwd=ROOT, capture_output=True, text=True)
    output.write_text(result.stdout+'\n'+result.stderr)
    assert result.returncode == 0, result.stderr
    return result.stdout


def original_cases(diagnostic):
    selected = []
    for name, kind, rank in [('training_maximum', 'training_reference', 0),
                             ('fresh_mixture_maximum', 'mixture', 0), ('fresh_mixture_second', 'mixture', 1)]:
        point = [p for p in diagnostic['top_pose_density_checks'] if p['source_kind'] == kind][rank]
        if kind == 'training_reference':
            path = ROOT/f"runs/ab-intermediate-cayley-reference-4x131072-l64-20260920/runs/{point['population']}/samples.jsonl"
        else:
            path = ROOT/f"runs/ab-intermediate-mixture-4x16384-l64-20260920/runs/{point['population']}/samples.jsonl"
        summary = read(path.parent/'summary.json'); assert sha(path) == summary['samples_sha256']
        row = None
        with path.open() as handle:
            for line in handle:
                candidate = json.loads(line)
                if candidate['draw'] == point['draw']:
                    row = candidate; break
        assert row and row['pose'] == point['pose'] and abs(row['q']-point['q']) < 2e-8
        logs = [c['log_weight'] for c in row['clouds']] if kind == 'training_reference' else row['cloud_log_weights']
        assert logs == point['original_cloud_log_weights']
        lower = row['clouds'][0]['lower_volume'] if kind == 'training_reference' else row['lower_volume']
        upper = row['clouds'][0]['upper_volume'] if kind == 'training_reference' else row['upper_volume']
        selected.append(dict(id=name, pose=point['pose'], original_q=point['q'], source_path=str(path),
                             source_samples_sha256=sha(path), source_draw=point['draw'], source_seed=point['seed'],
                             original_cloud_log_weights=logs, original_lower_volume=lower, original_upper_volume=upper,
                             original_row=row, clouds=256, seed=100101010+1009*len(selected)))
    return selected


def summarize_clouds(case, rows):
    assert len(rows) == 256 and [r['cloud'] for r in rows] == list(range(256))
    for row in rows:
        assert row['pose'] == case['pose'] and row['seed'] == case['seed']
        assert row['activity'] == .035 and row['lambda'] == 2.24 and row['lambda_ratio'] == 64
        assert abs(row['original_q']-case['original_q']) < 2e-8
        weight = row['weight']
        assert weight['lower_volume'] == case['original_lower_volume']
        assert abs(weight['upper_volume']-case['original_upper_volume']) < 1e-8
        assert type(weight['overlap_points']) is int and 0 <= weight['overlap_points'] <= weight['raw_points']
        assert abs(weight['log_weight']-(.035*weight['lower_volume']+weight['overlap_points']*math.log1p(1/64))) < 2e-8
    n = len(rows); logs = np.array([r['weight']['log_weight'] for r in rows]); counts = np.array([r['weight']['overlap_points'] for r in rows])
    offset = float(logs.max()); scaled = np.exp(logs-offset); mean = float(scaled.mean()); variance = float(scaled.var(ddof=1))
    oldlogs = case['original_cloud_log_weights']; oldmean = float(np.exp(np.array(oldlogs)-offset).mean())
    # Unordered distinct-cloud pairs are a descriptive empirical two-cloud law.
    # They are dependent pairs, not 32,640 extra independent physical samples.
    i, j = np.triu_indices(n, 1); pairmeans = (scaled[i]+scaled[j])/2
    percentile = float((np.sum(pairmeans < oldmean)+.5*np.sum(pairmeans == oldmean))/len(pairmeans))
    combined_se = math.sqrt(variance*(.5+1/n))
    k = int(counts.sum()); lam = 2.24; z = .035; lower = case['original_lower_volume']
    low = 0. if k == 0 else .5*chi2.ppf(.025, 2*k)
    high = .5*chi2.ppf(.975, 2*(k+1))
    # Sum K is Poisson(n*lambda*(C-L)); transform a count interval to log E[W].
    log_count_interval = [float(z*(lower+bound/(n*lam))) for bound in (low, high)]
    return dict(id=case['id'], clouds=n, log_arithmetic_mean_W=float(logsumexp(logs)-math.log(n)),
                arithmetic_mean_W=float(mean*math.exp(offset)), observed_relative_SE=math.sqrt(variance/n)/mean,
                original_log_two_cloud_mean=float(logsumexp(oldlogs)-math.log(2)), original_cloud_log_weights=oldlogs,
                original_mean_over_fresh_mean=float(oldmean/mean),
                original_minus_fresh_in_observed_combined_SE=float((oldmean-mean)/combined_se),
                original_two_cloud_mean_empirical_percentile=percentile,
                percentile_scope='Descriptive empirical convolution of distinct fresh clouds. Pairs are dependent, and original extremes were selected; not a calibrated significance test.',
                sum_overlap_counts=k, mean_overlap_count=float(counts.mean()), count_Fano_factor=float(counts.var(ddof=1)/counts.mean()),
                log_conditional_mean_W_Poisson_count_95_interval=log_count_interval,
                interval_formula='K_total~Poisson(n*lambda*(C-L)); Garwood bounds on its mean, transformed through log E[W]=z*(L+mu/(n*lambda)). Conditional on the fixed numerical geometry.',
                per_cloud_log_weight_range=[float(logs.min()), float(logs.max())], raw_points=sum(r['weight']['raw_points'] for r in rows),
                sampler_CPU_seconds=sum(r['sampling_CPU_seconds'] for r in rows), envelope_CPU_seconds=sum(r['envelope_CPU_seconds'] for r in rows),
                scope='Arithmetic W mean is the unbiased conditional estimator. Its logarithm is not unbiased. No proposal density, pose-volume integral, physical mass or regional convergence is inferred.')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(); out = args.out.resolve(); assert not out.exists()
    archive = out/'provenance'; archive.mkdir(parents=True)
    diagnostic_path = ROOT/'runs/ab-intermediate-guided-tail-diagnostic-qualified-20260920/analysis.json'
    diagnostic = read(diagnostic_path); cases = original_cases(diagnostic)
    production = ROOT/'runs/ab-intermediate-mixture-4x16384-l64-20260920'
    old_bundle_path = production/'runs/r00/provenance/source-bundle.json'; old = read(old_bundle_path)
    source_checks = {}
    for name in KERNEL_FILES:
        source_checks[name] = dict(archived_sha256=old['files'][name]['sha256'], current_sha256=sha(ROOT/name))
        assert source_checks[name]['archived_sha256'] == source_checks[name]['current_sha256'], name
    changed = {}
    for name, record in old['files'].items():
        if sha(ROOT/name) != record['sha256']:
            changed[name] = ''.join(difflib.unified_diff(record['text'].splitlines(True), (ROOT/name).read_text().splitlines(True), fromfile='archived/'+name, tofile='current/'+name))
    assert set(changed) <= {'src/lib.rs', 'src/latent_region.rs'}, 'Inspect new non-kernel changes before proceeding'
    write(archive/'source-identity-audit.json', dict(kernel_files=source_checks, other_changed_files=changed,
            scope='Current source is not globally identical to the production binary. Existing overlap, geometry, math, envelope and dependency files used by this driver are byte-identical. The latent region and module-list changes do not enter fixed-pose sampling.'))
    for source, target in [(old_bundle_path, 'production-source-bundle.json'), (diagnostic_path, 'input-diagnostic.json'),
                           (production/'provenance/config.json', 'production-config.json'), (production/'provenance/shape.json', 'shape.json'),
                           (ROOT/'tools/fixed_pose_clouds.rs', 'fixed_pose_clouds.rs'), (Path(__file__), 'run_intermediate_fixed_pose_clouds.py'),
                           (ROOT/'tests/overlap_weight.rs', 'existing-api-tests.rs')]:
        shutil.copy2(source, archive/target)
    cfg = read(production/'provenance/config.json'); cfg['shape'] = str(archive/'shape.json'); write(archive/'config.json', cfg)
    assert cfg['endpoint_gate'] == dict(max_cells=2047, max_depth=14, min_width=.5)
    stdout = command([CARGO, 'build', '--offline', '--locked', '--release', '--lib', '--message-format=json'], archive/'build.log')
    artifacts = {}; bundle_path = None
    for line in stdout.splitlines():
        item = json.loads(line)
        if item.get('reason') == 'compiler-artifact':
            libs = [path for path in item['filenames'] if path.endswith('.rlib')]
            # Host build-script libraries can have the same crate name with
            # different features/serde identities. Link the optimized runtime
            # dependency, including production float_roundtrip JSON parsing.
            if libs and item['profile']['opt_level'] == '3':
                if item['target']['name'] == 'serde_json': assert 'float_roundtrip' in item['features']
                artifacts[item['target']['name']] = libs[0]
        if item.get('reason') == 'build-script-executed' and 'tetramer-mc' in item['package_id']:
            bundle_path = Path(item['out_dir'])/'source-bundle.json'
    assert bundle_path and bundle_path.exists()
    compiled_bundle = read(bundle_path)
    for name in KERNEL_FILES: assert compiled_bundle['files'][name]['sha256'] == old['files'][name]['sha256']
    shutil.copy2(bundle_path, archive/'compiled-library-source-bundle.json')
    binary = archive/'fixed-pose-clouds'
    rustc = [RUSTC, '--edition=2024', '-O', archive/'fixed_pose_clouds.rs', '-L', f'dependency={ROOT}/target/release/deps', '-o', binary]
    for name in ('tetramer_mc', 'anyhow', 'serde_json', 'rand', 'sha2'):
        rustc += ['--extern', f'{name}={artifacts[name]}']
    command(rustc, archive/'harness-build.log')
    command([CARGO, 'test', '--offline', '--locked', '--release', '--test', 'overlap_weight'], archive/'api-validation.log')
    write(out/'original-cases.json', cases)
    control = dict(config=str(archive/'config.json'), lambda_ratio=64., cases=[{key:value for key,value in case.items() if key != 'original_row'} for case in cases])
    write(out/'control.json', control)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), target='Fixed-pose conditional overlap weights only; original [2,5) poses unchanged.',
                    source_diagnostic_sha256=sha(diagnostic_path), original_cases_sha256=sha(out/'original-cases.json'),
                    fixed_pose_count=3, clouds_per_pose=256, cloud_RNG='sha256(fixed-pose-overlap-weight-control-v1, case seed LE64, cloud index LE64) -> pinned rand StdRng',
                    seeds=[case['seed'] for case in cases], lambda_ratio=64., activity=.035, lambda_intensity=2.24,
                    endpoint_gate=cfg['endpoint_gate'], fixed_poses=cfg['fixed_poses'], capture_center=cfg['capture_center'], capture_radius=cfg['capture_radius'],
                    shape_sha256=sha(archive/'shape.json'), control_sha256=sha(out/'control.json'), binary_sha256=sha(binary),
                    archived_sha256={p.name:sha(p) for p in archive.iterdir()}, physical_kernel_changed=False,
                    no_pose_proposals=True, frozen_before_clouds=True, no_integrated_mass_claim=True)
    write(out/'protocol.json', protocol)
    command([binary, out/'control.json', out/'clouds.jsonl'], out/'execution.log')
    rows = [json.loads(line) for line in (out/'clouds.jsonl').read_text().splitlines()]
    assert len(rows) == 768
    reports = [summarize_clouds(case, [row for row in rows if row['case'] == case['id']]) for case in cases]
    write(out/'analysis.json', dict(complete=True, points=reports, physical=dict(fixed_poses=cfg['fixed_poses'], depletant_radius=1.5,
                  activity=.035, lambda_ratio=64., endpoint_gate=cfg['endpoint_gate']), protocol_sha256=sha(out/'protocol.json'),
                  cloud_rows_sha256=sha(out/'clouds.jsonl'), original_cases_sha256=sha(out/'original-cases.json'),
                  scope='Independent fixed-N clouds at three previously selected fixed poses. Originals are selection-biased extremes. No regional integration, refitting, density weighting, or assembly inference.'))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
