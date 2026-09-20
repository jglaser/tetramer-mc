#!/usr/bin/env python3
"""Fresh fixed-pose clouds at contact-atlas extremes; no regional estimator."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from prepare_cayley_rms_cover import read, write, sha, require
from prepare_intermediate_local_region import ROOT
from run_intermediate_fixed_pose_clouds import summarize_clouds, KERNEL_FILES
from run_shoulder_mis_campaign import local_dependencies


def run(diagnostic_path, out):
    require(not out.exists(), 'Use a fresh output directory')
    diagnostic = read(diagnostic_path)
    require(diagnostic['complete'], 'Incomplete source audit')
    previous = ROOT/'runs/ab-intermediate-fixed-pose-clouds-20260920'
    prior = read(previous/'protocol.json'); prior_analysis = read(previous/'analysis.json')
    require(sha(previous/'protocol.json') == prior_analysis['protocol_sha256'], 'Driver protocol changed')
    for name, digest in prior['archived_sha256'].items():
        require(sha(previous/'provenance'/name) == digest, 'Frozen driver source changed')
    binary = previous/'provenance/fixed-pose-clouds'
    require(sha(binary) == prior['binary_sha256'], 'Frozen driver binary changed')
    driver_cfg = read(previous/'provenance/config.json')
    compiled = read(previous/'provenance/compiled-library-source-bundle.json')
    shape_path = previous/'provenance/shape.json'
    require(sha(shape_path) == prior['shape_sha256'], 'Frozen driver shape changed')
    cases = []; source_files = {}; kernel_checks = {}
    for arm, seed in (('narrow', 101601010), ('broad', 101602019)):
        campaign = next(c for c in diagnostic['campaigns'] if c['arm'] == arm)
        point = campaign['top_poses'][0]
        require(point['original_log_importance_weight'] == max(p['original_log_importance_weight'] for p in campaign['top_poses']), 'Not the selected maximum')
        root = Path(campaign['root']); path = root/'runs'/point['population']
        for name in ('samples.jsonl', 'summary.json', 'manifest.json'):
            require(sha(path/name) == campaign['input_sha256'][str(path/name)], 'Audited source changed')
        summary = read(path/'summary.json'); manifest = read(path/'manifest.json')
        require(summary['complete'] and summary['samples_sha256'] == sha(path/'samples.jsonl'), 'Source population incomplete or changed')
        require(manifest['seed'] == point['seed'], 'Wrong source seed')
        source_cfg = read(root/'provenance/config.json')
        for key in ('fixed_poses', 'metadata', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density', 'endpoint_gate'):
            require(source_cfg[key] == driver_cfg[key], 'Driver physical configuration differs: '+key)
        require(sha(root/'provenance/shape.json') == sha(shape_path), 'Source shape differs')
        bundle = read(path/'provenance/source-bundle.json')
        kernel_checks[arm] = {}
        for name in KERNEL_FILES:
            expected = compiled['files'][name]['sha256']; observed = bundle['files'][name]['sha256']
            require(expected == observed, 'Compiled fixed-pose and original production kernel differ: '+name)
            kernel_checks[arm][name] = expected
        row = None
        with (path/'samples.jsonl').open() as handle:
            for line in handle:
                candidate = json.loads(line)
                if candidate['draw'] == point['draw']:
                    row = candidate; break
        require(row is not None and row['pose'] == point['pose'] and row['q'] == point['q'], 'Selected pose changed')
        require(row['cloud_log_weights'] == point['cloud_log_weights'] and row['cloud_overlap_counts'] == point['cloud_overlap_counts'], 'Selected cloud weights changed')
        require(row['log_importance_weight'] == point['original_log_importance_weight'], 'Selected original importance weight changed')
        require(row['lower_volume'] == point['lower_volume'], 'Selected lower envelope changed')
        cases.append(dict(id=arm, pose=row['pose'], original_q=row['q'],
            original_cloud_log_weights=row['cloud_log_weights'], original_lower_volume=row['lower_volume'],
            original_upper_volume=row['upper_volume'], clouds=256, seed=seed,
            source_path=str(path/'samples.jsonl'), source_samples_sha256=sha(path/'samples.jsonl'),
            source_seed=point['seed'], source_draw=point['draw'], original_row=row, source_point=point))
        for filename in ('manifest.json', 'summary.json'):
            source_files[arm+'-'+filename] = path/filename
        source_files[arm+'-source-bundle.json'] = path/'provenance/source-bundle.json'
        source_files[arm+'-config.json'] = root/'provenance/config.json'
    archive = out/'provenance'; archive.mkdir(parents=True)
    sources = {**local_dependencies([Path(__file__)]), **source_files,
        'input-diagnostic.json': diagnostic_path, 'driver-protocol.json': previous/'protocol.json',
        'fixed-pose-clouds': binary, 'fixed_pose_clouds.rs': previous/'provenance/fixed_pose_clouds.rs',
        'compiled-library-source-bundle.json': previous/'provenance/compiled-library-source-bundle.json',
        'driver-config.json': previous/'provenance/config.json', 'shape.json': shape_path}
    for name, path in sources.items(): shutil.copy2(path, archive/name)
    cfg = dict(driver_cfg); cfg['shape'] = str(archive/'shape.json'); write(archive/'config.json', cfg)
    write(out/'cases.json', cases)
    write(out/'control.json', dict(config=str(archive/'config.json'), lambda_ratio=64., cases=cases))
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), input_sha256=sha(diagnostic_path),
        cases_sha256=sha(out/'cases.json'), control_sha256=sha(out/'control.json'),
        original_driver_protocol_sha256=sha(previous/'protocol.json'), binary_sha256=sha(binary),
        config_sha256=sha(archive/'config.json'), shape_sha256=sha(archive/'shape.json'),
        seeds=[case['seed'] for case in cases], clouds_per_pose=256, activity=.035,
        depletant_radius=1.5, lambda_ratio=64., physical_kernel_identity=kernel_checks,
        archived_sha256={p.name:sha(p) for p in archive.iterdir()},
        scope='Fresh independent clouds at two previously selected fixed poses, using identical recorded envelopes and the frozen physical kernel. No new poses, importance weighting, corrected regional estimates, basin mass, or unbiased-log claim.')
    write(out/'protocol.json', protocol)
    result = subprocess.run([str(archive/'fixed-pose-clouds'), str(out/'control.json'), str(out/'clouds.jsonl')], capture_output=True, text=True)
    (out/'execution.log').write_text(result.stdout+'\n'+result.stderr)
    require(result.returncode == 0, result.stderr)
    rows = [json.loads(line) for line in (out/'clouds.jsonl').read_text().splitlines()]
    require(len(rows) == 512, 'Wrong fixed cloud budget')
    estimates = [summarize_clouds(case, [r for r in rows if r['case'] == case['id']]) for case in cases]
    write(out/'analysis.json', dict(complete=True, points=estimates, protocol_sha256=sha(out/'protocol.json'),
        cloud_rows_sha256=sha(out/'clouds.jsonl'), cases_sha256=sha(out/'cases.json'), scope=protocol['scope']))
    print(json.dumps(estimates, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); run(args.diagnostic.resolve(), args.out.resolve())
