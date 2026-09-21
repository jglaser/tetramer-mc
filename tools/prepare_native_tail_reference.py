#!/usr/bin/env python3
"""Prepare a complete native reference; old fitted radii are reporting masks only."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction as F
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.stats import beta

from analyze_native_region_reference import native_q
from audit_shoulder_mis_independently import chart_values
from prepare_cayley_rms_cover import derive_model, uniform_draws, check_coordinates, read, write, sha, require
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_smc_normalizer_atlas import arrays
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT/'runs/native-ab-old-chart-support-certificate-20260921'
BINARY = ROOT/'runs/ab-shoulder-cayley-reference-4x262144-l64-20260920/provenance/latent-region-normalizer'
BINARY_SHA = 'd5c112c8c9a93fb3914d935d087acb2a3b0cf222d89f3bd2614a1d63826bab0d'
MODEL_SHA = '86aedd9218381a47a4efef756bab82e58f03aee78f31d58873f10ab9cdc2b667'
CONFIG_SHA = '010616557e8afb6880a75fad64df3ce485c35afd47baaa21be3b3bc09383c8ad'
WINDOW = dict(minimum=0., maximum=1., lower_inclusive=True, upper_inclusive=True)
SEED, PROBE_SEED, PROBES, SAMPLES, POPULATIONS = 113501010, 113601010, 2048, 32768, 4
MASKS = ('old_r_le_4', 'old_4_lt_r_le_5', 'old_5_lt_r_le_8', 'old_8_lt_r_le_12', 'old_r_gt_12')


def old_radius_bin(radius):
    require(math.isfinite(radius) and radius >= 0, 'Invalid old-chart radius')
    return next(name for name, upper in zip(MASKS, (4., 5., 8., 12., math.inf)) if radius <= upper)


def inverse(matrix):
    n = len(matrix)
    a = [list(row)+[F(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = a[i][i]; require(pivot != 0, 'Zero rational inverse pivot')
        a[i] = [v/pivot for v in a[i]]
        for j in range(n):
            if j != i:
                factor = a[j][i]; a[j] = [x-factor*y for x, y in zip(a[j], a[i])]
    return [row[n:] for row in a]


def positive_pivots(matrix):
    require(all(matrix[i][j] == matrix[j][i] for i in range(len(matrix)) for j in range(len(matrix))),
            'Rational matrix is not symmetric')
    a = [list(row) for row in matrix]; pivots = []
    for k in range(len(a)):
        pivot = a[k][k]; require(pivot > 0, 'Rational matrix is not positive definite')
        pivots.append(pivot)
        for i in range(k+1, len(a)):
            for j in range(k+1, len(a)):
                a[i][j] -= a[i][k]*a[k][j]/pivot
    return pivots


def exact_cover_check(config_json, model):
    """Certify the emitted decimal-input ellipsoid contains every exact q<=1 pose.

    A>160I and RMS<=2 imply sin(theta/2)^2<1/160, so k=159/160.
    Cov_cover^-1 <= diag(I,4*k*A_chart/ell^2) then gives rho_cover<=2.
    Only the angular block needs strict LDL; the translation gap is exactly zero.
    """
    cfg = json.loads(config_json, parse_float=F, parse_int=F)
    m = json.loads(json.dumps(model, allow_nan=False), parse_float=F, parse_int=F)
    metric = cfg['metadata']; members = [p['position'] for p in metric['rigid_members']]
    require(metric['member_error_scale'] == 2 and metric['angle_error_scale_deg'] == 15,
            'Certificate requires the original registration scales')
    require(metric['native_poses'] == [dict(position=[0, 0, 0], orientation=[1, 0, 0, 0])]
            and cfg['fixed_poses'][0]['orientation'] == [0, 1, 0, 0], 'Certificate requires exact native/A frames')
    require(all(sum(p[i] for p in members) == 0 for i in range(3)), 'Member centroid is not exactly zero')
    n = len(members)
    moment = [[sum(p[i]*p[j] for p in members)/n for j in range(3)] for i in range(3)]
    trace = sum(moment[i][i] for i in range(3))
    a = [[(trace if i == j else F(0))-moment[i][j] for j in range(3)] for i in range(3)]
    ap = positive_pivots([[a[i][j]-(160 if i == j else 0) for j in range(3)] for i in range(3)])
    signs = [1, -1, -1]
    require(m['weights'] == [1] and m['means'] == [[0]*6] and len(m['anchors']) == 1,
            'Require one zero-mean geometric chart')
    require(m['anchors'][0] == dict(position=[-signs[i]*cfg['fixed_poses'][0]['position'][i] for i in range(3)],
                                   rotation=[[signs[i] if i == j else 0 for j in range(3)] for i in range(3)]),
            'Native cover anchor is not exact in the A frame')
    covariance = m['covariances'][0]; ell = m['angular_length']
    require(ell > 0 and len(covariance) == 6 and all(len(row) == 6 for row in covariance), 'Invalid cover covariance')
    require(all(covariance[i][j] == F(i == j) for i in range(3) for j in range(3))
            and all(covariance[i][j] == covariance[j][i] == 0 for i in range(3) for j in range(3, 6)),
            'Cover translation block changed')
    angular_precision = inverse([row[3:] for row in covariance[3:]])
    k = F(159, 160)
    gap = [[4*k*signs[i]*a[i][j]*signs[j]/ell**2-angular_precision[i][j] for j in range(3)] for i in range(3)]
    gp = positive_pivots(gap)
    return dict(complete=True, arithmetic='Exact Fraction arithmetic on archived/emitted JSON decimal literals',
                A_minus_160I_positive_LDL_pivots_approx=[float(x) for x in ap],
                k_exact='159/160', certified_precision_gap_positive_LDL_pivots_approx=[float(x) for x in gp],
                cover_radius=2., conclusion='Every exact original q<=1 pose belongs to the emitted geometry-chart rho<=2 ball.',
                argument='RMS²=|t|²+4*cᵀA_chart*c/(1+|c|²)<=4; A>160I implies 1/(1+|c|²)>=159/160. The exact precision comparison bounds rho² by this expression.',
                limitation='A certificate for real arithmetic on archived decimal geometry, not a bound on runtime floating-point encoder error or physical statistical weight.')


def prepare(out, campaign_out):
    out, campaign_out = Path(out).resolve(), Path(campaign_out).resolve()
    require(not out.exists() and not campaign_out.exists(), 'Preparation and proposed campaign paths must be fresh')
    certificate = read(CERTIFICATE/'analysis.json'); certified = certificate['result']
    require(certificate['complete'] and certificate['exit_code'] == 0 and certified['exact_U_less_2704'], 'Incomplete old support certificate')
    require(sha(CERTIFICATE/'certificate.py') == certificate['script_sha256']
            and sha(CERTIFICATE/'stdout.json') == certificate['stdout_sha256'], 'Changed exact certificate')
    require(read(CERTIFICATE/'stdout.json') == certified, 'Certificate output mismatch')
    config_path, old_model_path = Path(certified['config']), Path(certified['model'])
    require(sha(config_path) == certified['config_sha256'] == CONFIG_SHA, 'Changed native config')
    require(sha(old_model_path) == certified['model_sha256'] == MODEL_SHA, 'Changed old chart')
    require(sha(BINARY) == BINARY_SHA, 'Changed executable')
    cfg, old = read(config_path), read(old_model_path); shape = Path(cfg['shape'])
    if not shape.is_absolute(): shape = config_path.parent/shape
    require(sha(shape) == old['shape_sha256'] == certified['shape_sha256'], 'Changed shape')
    require(cfg['capture_radius'] == 18. and cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035
            and len(cfg['fixed_poses']) == 2, 'Changed native physical target')
    fixed = cfg['fixed_poses'][0]
    model, proof = derive_model(cfg['metadata'], fixed, old['shape_sha256'], q_max=1., ell=old['angular_length'])
    exact = exact_cover_check(config_path.read_text(), model)
    archive = out/'provenance'; archive.mkdir(parents=True)
    sources = {'input-config.json':config_path, 'old-model.json':old_model_path, 'shape.json':shape,
               'support-certificate.json':CERTIFICATE/'analysis.json', 'certificate-stdout.json':CERTIFICATE/'stdout.json',
               'certificate.py':CERTIFICATE/'certificate.py', 'latent-region-normalizer':BINARY,
               'run_latent_region_campaign.py':ROOT/'tools/run_latent_region_campaign.py',
               **local_dependencies([Path(__file__), ROOT/'tools/analyze_native_tail_reference.py',
                                      ROOT/'tools/test_native_tail_reference.py',
                                      # The runner copies this by filename; it is not a Python import.
                                      ROOT/'tools/analyze_latent_region_shells.py'])}
    for name, path in sources.items(): shutil.copy2(path, archive/name)
    frozen_cfg = copy.deepcopy(cfg); frozen_cfg['shape'] = str(archive/'shape.json')
    write(out/'config.json', frozen_cfg); write(out/'model.json', model)
    write(out/'old-model.json', old); write(out/'complete-cover-proof.json', dict(numerical_construction=proof, exact_check=exact))
    region = dict(fixed_neighbor=copy.deepcopy(fixed), physical_fixed_neighbors=copy.deepcopy(cfg['fixed_poses']),
                  capture_center=cfg['capture_center'], capture_radius=cfg['capture_radius'],
                  shape_sha256=old['shape_sha256'], activity=cfg['reservoir_density'], depletant_radius=cfg['depletant_radius'],
                  physical_metric=copy.deepcopy(cfg['metadata']), gaussian_chart=model,
                  minimum_mahalanobis_radius=0., mahalanobis_radius=2.,
                  minimum_original_q=0., maximum_original_q=1.,
                  minimum_original_q_inclusive=True, maximum_original_q_inclusive=True,
                  definition='Complete original native q<=1 under uniform geometric six-ball sampling; fitted-chart radii enter reporting only.')
    write(out/'region.json', region)
    command = [str(archive/'run_latent_region_campaign.py'), '--out', str(campaign_out), '--config', str(out/'config.json'),
               '--region', str(out/'region.json'), '--binary', str(archive/'latent-region-normalizer'),
               '--samples', str(SAMPLES), '--replicates', str(POPULATIONS), '--workers', '4', '--seed', str(SEED),
               '--lambda-ratio', '64', '--cloud-replicates', '2']
    protocol = dict(schema='complete-native-tail-reference-preparation-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    production_launched=False, original_q_window=WINDOW, old_chart_reporting_edges=[4., 5., 8., 12.],
                    reporting_masks=MASKS, old_chart_tail_rule='r_old>12 with no finite cutoff; r_old<52 is a certificate diagnostic only.',
                    sampling='Uniform standard six-ball radius 2; physical measure d³t times normalized Haar. Every original draw remains in the complete denominator; invalid and off-mask draws contribute zero.',
                    weight='V6 * J * 1_capture * 1_hard_AB * 1_(0<=q<=1) * mean(W1,W2); tail/finite-shell masks multiply this contribution directly.',
                    old_chart_role='Fixed coordinate transform for predeclared reporting masks only; no Gaussian weighting, refit, or hard-validity conditioning.',
                    completeness='The exact emitted-decimal precision certificate proves complete q<=1 coverage; shape/capture/AB constraints only reduce that domain.',
                    proposed_campaign=dict(output=str(campaign_out), samples_per_population=SAMPLES, populations=POPULATIONS,
                                           workers=4, seed_base=SEED, seeds=[SEED+1009*i for i in range(POPULATIONS)],
                                           lambda_ratio=64., cloud_replicates=2, command_argv_after_python=command),
                    analysis_command_argv_after_python=[str(archive/'analyze_native_tail_reference.py'),
                        '--preparation', str(out), '--campaign', str(campaign_out),
                        '--out', str(campaign_out.parent/'native-tail-reference-audit-20260921')],
                    geometry_probe_count=PROBES, geometry_probe_seed=PROBE_SEED,
                    geometry_gate='Review cost and support after fixed probes. Probe failures are retained; zero hits do not establish zero mass or justify truncating support.',
                    executable_sha256=BINARY_SHA, config_sha256=sha(out/'config.json'), region_sha256=sha(out/'region.json'),
                    model_sha256=sha(out/'model.json'), old_model_sha256=sha(out/'old-model.json'),
                    exact_cover_proof_sha256=sha(out/'complete-cover-proof.json'),
                    input_sha256={str(path):sha(path) for path in sources.values()},
                    archived_sha256={name:sha(archive/name) for name in sources})
    write(out/'protocol.json', protocol); write(out/'freeze.json', {p.name:sha(p) for p in out.glob('*.json')})
    started = time.process_time()
    poses, latent, x, jacobian = uniform_draws(model, fixed, 2., PROBES, PROBE_SEED)
    checks = check_coordinates(model, fixed, poses, latent, x, jacobian)
    positions, _, rotations = arrays(poses)
    old_radii = chart_values(old, positions, rotations, fixed)[1][:, 0]
    atom = AtomUnionAudit(read(shape), cfg['fixed_poses'])
    counts = {key:0 for key in ('capture', 'hard', 'native', 'valid', *MASKS)}
    rows = []
    for i, pose in enumerate(poses):
        q = native_q(cfg['metadata'], pose); gaps = atom.gaps(pose)
        capture = math.dist(pose['position'], cfg['capture_center']) <= cfg['capture_radius']
        hard = min(gaps) >= 0; native = 0 <= q <= 1; valid = capture and hard and native
        key = old_radius_bin(float(old_radii[i]))
        if native: require(old_radii[i] < 52, 'Probe violates exact old-chart support certificate; stop without censoring')
        for flag, value in (('capture',capture), ('hard',hard), ('native',native), ('valid',valid)):
            counts[flag] += int(value)
        if valid: counts[key] += 1
        rows.append(dict(draw=i, pose=pose, q=q, cover_radius=float(np.linalg.norm(latent[i])),
                         old_radius=float(old_radii[i]), log_physical_jacobian=float(jacobian[i]),
                         minimum_gap_by_neighbor_A=gaps, capture_valid=capture, hard_valid=hard,
                         native_valid=native, target_valid=valid, reporting_bin=key))
    with (out/'geometry-probes.jsonl').open('w') as handle:
        for row in rows: handle.write(json.dumps(row, allow_nan=False)+'\n')
    def interval(k):
        return [float(beta.ppf(.025, k, PROBES-k+1)) if k else 0.,
                float(beta.ppf(.975, k+1, PROBES-k)) if k < PROBES else 1.]
    report = dict(complete=True, physical_production=False, no_depletant_clouds=True, counts=counts,
                  draws=PROBES, seed=PROBE_SEED, checks=checks,
                  geometry_fraction_intervals_95={key:interval(k) for key,k in counts.items()},
                  projected_counts_at_proposed_fixed_N={key:k/PROBES*SAMPLES*POPULATIONS for key,k in counts.items()},
                  valid_old_radius_range=[min((r['old_radius'] for r in rows if r['target_valid']), default=None),
                                          max((r['old_radius'] for r in rows if r['target_valid']), default=None)],
                  complete_cover_physical_volume_bounds_A3=proof['physical_cover_volume_bounds_A3'],
                  geometry_CPU_seconds=time.process_time()-started,
                  scope='Geometry counts forecast occupied rows only, not physical-weight variance or convergence. The executable still evaluates clouds for every valid native pose; old-radius masks are poststratification.',
                  protocol_sha256=sha(out/'protocol.json'), probe_sha256=sha(out/'geometry-probes.jsonl'))
    for name,digest in read(out/'freeze.json').items(): require(sha(out/name) == digest, 'Preparation changed during probes')
    write(out/'report.json', report); print(json.dumps(report, indent=2)); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--campaign-out', type=Path, required=True)
    args = parser.parse_args(); prepare(args.out, args.campaign_out)
