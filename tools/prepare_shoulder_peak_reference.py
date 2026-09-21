#!/usr/bin/env python3
"""Freeze independent finite-neighborhood checks of the inner shoulder peak."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from analyze_native_region_reference import native_q
from prepare_cayley_rms_cover import read, write, sha, require, uniform_draws, check_coordinates
from prepare_far_atlas_repeat import validate_command_options
from prepare_intermediate_local_region import BINARY, BINARY_SHA
from prepare_peak_neighborhood import geometry_model, verify_member_geometry
from prepare_native_confirmation_atlas import AtomUnionAudit
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT/'runs/ab-inner-shoulder-local-candidates-20260921/analysis.json'
CANDIDATES_SHA = 'cb776845bd955b78d1e93cbbc5206c4f59c8d9d32350b911f8314ad6833c9881'
GUIDES = ROOT/'runs/ab-shoulder-guide-preparation-20260920'
DIRECT = ROOT/'runs/ab-inner-shoulder-matching-comparison-20260920/comparison.json'
GUIDED = ROOT/'runs/ab-shoulder-confirmation-with-cover-comparison-20260920/comparison.json'
WINDOW = dict(minimum=1., maximum=1.1, lower_inclusive=False, upper_inclusive=False)
RADII, SEEDS = (.25, .5), (109001010, 109101010)
POPULATIONS, SAMPLES, PROBES = 16, 16384, 256
PHYSICAL = ('metadata', 'fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density')


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Fresh preparation directory required')
    require(sha(CANDIDATES) == CANDIDATES_SHA, 'Discovery diagnostic changed')
    diagnostic = read(CANDIDATES); require(diagnostic['complete'] and diagnostic['selected_peak'] == 0, 'Explicit original maximum required')
    for path, digest in diagnostic['source_sha256'].items(): require(sha(Path(path)) == digest, 'Discovery source changed')
    selected = diagnostic['candidates'][0]
    require((selected['population'], selected['seed'], selected['draw']) == ('r02', 99633028, 211501), 'Selected stream or draw changed')
    source_path = Path(selected['source_samples_path'])
    require(sha(source_path) == selected['source_samples_sha256'], 'Selected sample changed')
    original = None
    with source_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['draw'] == selected['draw']: original = row; break
    require(original == selected['row'] and original['pose'] == selected['pose'], 'Original selected row changed')
    require(original['log_importance_weight'] == selected['original_log_importance_weight'], 'Do not replace selected weight')
    require(original['hard_valid'] and original['capture_valid'] and original['region_valid'] and 1 < original['q'] < 1.1, 'Selected point outside original inner target')
    cfg = read(GUIDES/'config.json'); shape = Path(cfg['shape']); fixed = cfg['fixed_poses'][0]
    require({k:cfg[k] for k in PHYSICAL} == diagnostic['source_physical_signature'], 'Physical target differs')
    require(cfg['capture_radius'] == 18. and cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and len(cfg['fixed_poses']) == 2, 'Wrong fixed AB bath')
    require(cfg['metadata']['member_error_scale'] == 2. and cfg['metadata']['angle_error_scale_deg'] == 15., 'Changed registration metric')
    atom = AtomUnionAudit(read(shape), cfg['fixed_poses']); gaps = atom.gaps(selected['pose'])
    require(min(gaps) >= 0 and abs(native_q(cfg['metadata'], selected['pose'])-selected['q']) < 2e-8, 'Selected geometry differs')
    model, geometry = geometry_model(cfg['metadata'], fixed, selected['pose'], sha(shape), diagnostic['geometric_model']['angular_length'])
    require(model == diagnostic['geometric_model'], 'Do not refit the discovered chart')
    # The generic helper's display radii are not the present protocol's radii.
    eigen_min = min(geometry['A_eigenvalues_A2'])
    geometry['maximum_rotation_angle_deg_by_radius'] = {str(r):math.degrees(2*math.atan(r/(2*math.sqrt(eigen_min)))) for r in RADII}
    require(sha(BINARY) == BINARY_SHA, 'Reviewed latent executable changed')
    direct, guided = read(DIRECT), read(GUIDED)
    require(direct['complete'] and guided['q_window'] == dict(minimum=1., maximum=2., lower_inclusive=False, upper_inclusive=False), 'Historical windows differ')
    history = [dict(name='direct-inner', root=direct['focused_band']['root'], comparison_path=str(DIRECT), comparison_sha256=sha(DIRECT),
        original_q_window=WINDOW, original_N=diagnostic['source_full_N'])]
    for key in ('mixture-confirmation', 'geometry-confirmation'):
        c = guided['campaigns'][key]
        history.append(dict(name=key, root=c['root'], comparison_path=str(GUIDED), comparison_sha256=sha(GUIDED),
            comparison_key=key, original_q_window=guided['q_window'], original_N=c['physical']['draws']))
    prior_seeds = set()
    for h in history:
        master = read(Path(h['root'])/'manifest.json'); h['manifest_sha256'] = sha(Path(h['root'])/'manifest.json')
        prior_seeds.update(j['seed'] for j in master['jobs'])
    new_seeds = [base+1009*i for base in SEEDS for i in range(POPULATIONS)]
    require(len(set(new_seeds)) == len(new_seeds) and not prior_seeds.intersection(new_seeds), 'Production streams reused')
    archive = out/'provenance'; archive.mkdir(parents=True)
    runner = ROOT/'tools/run_latent_region_campaign.py'
    runner_files = [ROOT/'tools'/name for name in ('analyze_latent_region.py', 'prepare_smc_normalizer_atlas.py',
        'analyze_basin_normalizers.py', 'analyze_native_region_reference.py', 'analyze_latent_region_shells.py')]
    sources = {'discovery.json': CANDIDATES, 'direct-comparison.json': DIRECT, 'guided-comparison.json': GUIDED,
        'input-config.json': GUIDES/'config.json', 'shape.json': shape,
        **local_dependencies([Path(__file__), runner, *runner_files])}
    for name, path in sources.items(): shutil.copy2(path, archive/name)
    frozen_cfg = copy.deepcopy(cfg); frozen_cfg['shape'] = str(archive/'shape.json')
    write(out/'config.json', frozen_cfg); write(out/'model.json', model)
    write(out/'selected-pose.json', selected); write(out/'geometry.json', geometry)
    campaigns, commands = [], {}
    for radius, base in zip(RADII, SEEDS):
        label = f'{radius:g}'.replace('.', 'p')
        region = dict(fixed_neighbor=fixed, physical_fixed_neighbors=cfg['fixed_poses'],
            capture_center=cfg['capture_center'], capture_radius=cfg['capture_radius'], activity=cfg['reservoir_density'],
            depletant_radius=cfg['depletant_radius'], physical_metric=cfg['metadata'], shape_sha256=sha(shape),
            gaussian_chart=model, minimum_original_q=1., maximum_original_q=1.1,
            minimum_original_q_inclusive=False, maximum_original_q_inclusive=False,
            minimum_mahalanobis_radius=0., mahalanobis_radius=radius,
            definition='Frozen geometric chart ball at one discovered inner-shoulder pose, intersected with unchanged strict1<q<1.1, capture and both AB hard neighbors. Finite local region, not complete shoulder.')
        path = out/f'region-r{label}.json'; write(path, region)
        output = ROOT/f'runs/ab-inner-shoulder-peak-reference-20260921/r{label}'
        require(not output.exists(), 'Preserve existing production outputs')
        campaigns.append(dict(radius_A=radius, region=str(path), region_sha256=sha(path), output=str(output),
            populations=POPULATIONS, samples_per_population=SAMPLES, workers=16, seed_base=base,
            seeds=[base+1009*i for i in range(POPULATIONS)]))
        commands[label] = [sys.executable, str(archive/runner.name), '--out', str(output), '--config', str(out/'config.json'),
            '--region', str(path), '--binary', str(BINARY), '--samples', str(SAMPLES), '--replicates', str(POPULATIONS),
            '--workers', '16', '--seed', str(base), '--lambda-ratio', '64', '--cloud-replicates', '2']
    help_text = subprocess.run([sys.executable, str(archive/runner.name), '--help'], capture_output=True, text=True, check=True).stdout
    for command in commands.values(): validate_command_options(command, help_text)
    (archive/'runner-help.txt').write_text(help_text); write(out/'commands.json', commands)
    protocol = dict(schema='AB-inner-shoulder-geometric-peak-reference-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        physical={k:cfg[k] for k in PHYSICAL}, original_q_window=WINDOW,
        selection_source=dict(path=str(CANDIDATES), sha256=CANDIDATES_SHA, selected_peak=0),
        selection='Recorded direct-inner maximum. Selection and its top-three mass fraction are discovery diagnostics, not independent finite-region weights.',
        sampling='Uniform latent six-ball with exact translation/normalized-Haar Jacobian; all target/capture/hard zeros retained; no retries, refits, early stop or population replacement.',
        analysis_plan=dict(radial_edges_A=[0.,.25,.5], disjoint_shell_sources=[dict(radius_A=.25, shell=[0.,.25], lower_inclusive=True), dict(radius_A=.5, shell=[.25,.5], lower_inclusive=False)],
            rule='Sum only the independent [0,.25] and(.25,.5] estimates with their independent variances. Do not add overlapping whole-ball estimates. Historical remasks retain entire original N, full original density, inner-window complement and same-row covariance. No history/fresh pooling.'),
        executable=str(BINARY), executable_sha256=BINARY_SHA, lambda_ratio=64., cloud_replicates=2,
        campaigns=campaigns, historical_campaigns=history, probe_count_per_radius=PROBES,
        probe_seeds=[109201010+1009*i for i in range(len(RADII))], selected_atomic_gaps_A=gaps,
        config_sha256=sha(out/'config.json'), model_sha256=sha(out/'model.json'), selected_pose_sha256=sha(out/'selected-pose.json'),
        commands_sha256=sha(out/'commands.json'), input_sha256={str(p):sha(p) for p in sources.values()},
        archived_sha256={p.name:sha(p) for p in archive.iterdir()},
        scope='Local calibration at a data-selected contact. This neither replaces the complete shoulder nor establishes unseen-tail convergence, mixing or assembly.')
    write(out/'protocol.json', protocol); write(out/'freeze.json', {p.name:sha(p) for p in out.glob('*.json')})
    reports = []
    for radius, seed in zip(RADII, protocol['probe_seeds']):
        poses, latent, x, logj = uniform_draws(model, fixed, radius, PROBES, seed)
        checks = check_coordinates(model, fixed, poses, latent, x, logj)
        checks.update(verify_member_geometry(cfg['metadata'], fixed, selected['pose'], poses, latent, x, model, geometry, logj))
        rows = []
        for index, pose in enumerate(poses):
            q = native_q(cfg['metadata'], pose); local_gaps = atom.gaps(pose)
            capture = bool(np.linalg.norm(np.asarray(pose['position'])-cfg['capture_center']) <= cfg['capture_radius'])
            rows.append(dict(draw=index, pose=pose, q=q, capture_valid=capture, atomic_gaps_A=local_gaps,
                valid=bool(capture and min(local_gaps) >= 0 and 1 < q < 1.1), log_jacobian=float(logj[index])))
        path = out/f'geometry-probes-r{radius:g}.jsonl'
        with path.open('x') as handle:
            for row in rows: handle.write(json.dumps(row,allow_nan=False)+'\n')
        reports.append(dict(radius_A=radius, seed=seed, valid=sum(r['valid'] for r in rows), unconditional_draws=PROBES,
            geometry_checks=checks, probe_path=str(path), probe_sha256=sha(path)))
    write(out/'report.json', dict(complete=True, no_physical_weight_draws=True, frozen_protocol_sha256=sha(out/'protocol.json'),
        freeze_sha256=sha(out/'freeze.json'), probes=reports, production_allocation_unchanged_after_probes=True))
    print(dict(complete=True, prepared_only=True, protocol=str(out/'protocol.json'), protocol_sha256=sha(out/'protocol.json'),
        probes=[dict(radius_A=r['radius_A'], valid=r['valid'], N=r['unconditional_draws']) for r in reports]))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);prepare(p.parse_args().out)
