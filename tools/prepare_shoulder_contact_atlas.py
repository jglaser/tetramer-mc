#!/usr/bin/env python3
"""Freeze complete-shoulder proposals using independently calibrated contacts."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
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
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

from analyze_native_region_reference import (GaussianGuide, hybrid_log_density,
    native_q, theta_minus_sin, validate_window_cover, window_cover_metric)
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_far_atlas_repeat import validate_command_options
from prepare_far_capture_control import BINARY, EXPECTED_BINARY
from prepare_native_confirmation_atlas import AtomUnionAudit, to_world
from prepare_smc_normalizer_atlas import Density, arrays, relative_poses
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'runs/ab-shoulder-guide-peak-reference-assessment-20260921/analysis.json'
REFERENCE_SHA = '100f67ef74289bee1c3f0a38651c2b1c211e83afb9e50d0c9c516521fd44e3ef'
LOCAL = ROOT/'runs/ab-shoulder-guide-peak-reference-preparation-20260921'
DIRECT = ROOT/'runs/ab-inner-shoulder-peak-reference-preparation-20260921'
LEGACY = ROOT/'runs/ab-shoulder-guide-preparation-20260920'
FITS = ROOT/'runs/ab-shoulder-local-guides-20260921'
CENTERS = ('direct', 'mixture', 'geometry')
WINDOW = dict(minimum=1., maximum=2., lower_inclusive=False, upper_inclusive=False)
INNER = dict(minimum=1., maximum=1.1, lower_inclusive=False, upper_inclusive=False)
PHYSICAL = ('metadata', 'fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density')
ARMS = (('narrow', 1., 112101010), ('broad', 4., 112201010))
BETA, EPSILON = .99, .05


def assemble(legacy, fitted, log_masses, covariance_scale):
    """Half legacy, half calibrated contacts; full g sums every component."""
    require(set(fitted) == set(log_masses) == set(CENTERS), 'Need three fixed contacts')
    require(covariance_scale in (1., 4.), 'Predeclared covariance controls only')
    keys = ('schema', 'angular_length', 'coordinate_convention', 'shape_sha256')
    require(len(legacy['weights']) == 2 and all(w > 0 for w in legacy['weights']) and
            abs(sum(legacy['weights'])-1) < 1e-12, 'Unchanged normalized legacy mixture required')
    require(all(math.isfinite(log_masses[name]) for name in CENTERS), 'Finite independently measured masses required')
    mass_weights = np.exp(np.array([log_masses[name] for name in CENTERS])-logsumexp(list(log_masses.values())))
    model = {k: copy.deepcopy(legacy[k]) for k in keys}
    for k in ('anchors', 'means', 'covariances'):
        model[k] = copy.deepcopy(legacy[k])
    model['weights'] = [.5*w for w in legacy['weights']]
    for name, weight in zip(CENTERS, mass_weights):
        one = fitted[name]
        require(all(one[k] == legacy[k] for k in keys) and one['weights'] == [1.], 'Fitted conventions/components differ')
        require(all(len(one[k]) == 1 for k in ('anchors', 'means', 'covariances')), 'Exactly one component per local fit')
        model['anchors'].extend(copy.deepcopy(one['anchors']))
        model['means'].extend(copy.deepcopy(one['means']))
        model['covariances'].append((covariance_scale*np.asarray(one['covariances'][0])).tolist())
        model['weights'].append(.5*float(weight))
    require(abs(sum(model['weights'])-1.) < 1e-12 and len(model['weights']) == 5, 'Atlas normalization changed')
    return model, dict(zip(CENTERS, mass_weights.tolist()))


def cover_model(cfg):
    """Product cover from the original maximum-q member and angle bounds."""
    metric = {k: cfg['metadata'][k] for k in ('native_poses', 'rigid_members', 'member_error_scale', 'angle_error_scale_deg')}
    require(len(metric['native_poses']) == 1, 'This cover assumes one native reference')
    enclosing = window_cover_metric(metric, WINDOW)
    members = np.asarray([m['position'] for m in metric['rigid_members']])
    require(np.array_equal(members.mean(axis=0), np.zeros(3)), 'Exactly centered members required')
    covariance = members.T@members/len(members)
    magnitude = float(np.sum(members*members)/len(members))
    slack = 4096*math.ulp(1.)*(1+magnitude)*len(members)
    upper = min(float(np.linalg.norm(covariance)), float(np.max(np.sum(abs(covariance), axis=1))))+slack
    lower = max(0., float(np.trace(covariance))-slack-upper)
    radius = enclosing['member_error_scale']
    cap = min(math.radians(enclosing['angle_error_scale_deg']),
              2*math.asin(min(1., radius/(2*math.sqrt(lower)))) if lower else math.pi)
    cover = dict(reference=metric['native_poses'][0], centroid=[0., 0., 0.],
                 member_covariance=covariance.tolist(), moment_trace=float(np.trace(covariance)),
                 lambda_max_upper=upper, l_lower=lower,
                 nominal_angle_cap=math.radians(enclosing['angle_error_scale_deg']),
                 ball_radius=radius, angle_cap=cap, volume=4*radius**3/3*theta_minus_sin(cap))
    validate_window_cover(dict(q_window=WINDOW, metric=metric, cover_metric=enclosing), cover, cfg)
    return dict(scales=[1.], weights=[1.], covers=[cover]), dict(
        metric=metric, enclosing_metric=enclosing, member_covariance_A2=covariance.tolist(),
        lower_rotational_moment_bound_A2=lower, outward_slack_A2=slack,
        cover=cover, scope='Complete cover of original q<=2, with checked outward floating-point slack; capture and strict target masks remain separate. Not a bound on statistical mass.')


def probe(model, cfg, cover, atom, seed, count=256):
    rng = np.random.default_rng(seed); density = Density(model)
    components = rng.choice(len(model['weights']), count, p=model['weights'])
    poses = [to_world(density.draw_component(rng, int(k), 1), cfg['fixed_poses'][0])[0] for k in components]
    raw = density.evaluate(relative_poses(poses, cfg['fixed_poses'][0]))[0]
    desc = dict(weight=BETA, uniform_probability=EPSILON, anchor_index=0, anchor_pose=cfg['fixed_poses'][0],
                cube_lengths=[36.]*3, capture_center=cfg['capture_center'])
    guide = GaussianGuide(desc, model, cfg, model['shape_sha256'])
    t, _, rotations = arrays(poses); reference = cover['covers'][0]
    rt, _, rr = arrays([reference['reference']])
    radii = np.linalg.norm(t-rt[0], axis=1)
    angles = Rotation.from_matrix(rr[0].T@rotations).magnitude()
    covered = (radii <= reference['ball_radius']) & (angles <= reference['angle_cap'])
    cube = np.all((t-np.asarray(cfg['capture_center']) >= -18.) & (t-np.asarray(cfg['capture_center']) < 18.), axis=1)
    expected_guide = np.logaddexp(math.log1p(-EPSILON)+raw, np.where(cube, math.log(EPSILON)-3*math.log(36.), -math.inf))
    expected = np.logaddexp(math.log(BETA)+expected_guide, np.where(covered, math.log1p(-BETA)-math.log(reference['volume']), -math.inf))
    observed = np.array([hybrid_log_density(cover, guide, pose)[0] for pose in poses])
    error = float(np.max(abs(observed-expected))); require(error < 2e-9, 'Independent full mixture density differs')
    rows = []
    for i, pose in enumerate(poses):
        q = native_q(cfg['metadata'], pose); gaps = atom.gaps(pose)
        capture = math.dist(pose['position'], cfg['capture_center']) <= 18.
        require(q > 2. or bool(covered[i]), 'A q<=2 probe escaped its complete cover')
        rows.append(dict(draw=i, component=int(components[i]), pose=pose, q=q, atomic_gaps_A=gaps,
            capture_valid=capture, hard_valid=min(gaps) >= 0, shoulder_valid=capture and min(gaps) >= 0 and 1 < q < 2))
    return dict(seed=seed, draws=count, shoulder_valid=sum(r['shoulder_valid'] for r in rows),
        hard_valid=sum(r['hard_valid'] for r in rows), full_density_error=error,
        component_counts=np.bincount(components, minlength=5).tolist(),
        scope='Pure-atlas proposal probes only; no physical weights, full-hybrid acceptance estimate or equilibrium claim.'), rows


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Fresh preparation directory required')
    require(sha(REFERENCE) == REFERENCE_SHA, 'Independent local calibration changed')
    reference = read(REFERENCE); direct_path = Path(reference['direct_reference']['path']); direct = read(direct_path)
    require(reference['complete'] and reference['original_q_window'] == INNER and sha(direct_path) == reference['direct_reference']['sha256'], 'Wrong finite reference')
    for result, path in ((reference, REFERENCE), (direct, direct_path)):
        for name, digest in result['archived_sha256'].items(): require(sha(path.parent/'provenance'/name) == digest, 'Reference source archive changed')
        for name, digest in result['source_sha256'].items(): require(sha(name) == digest, 'Reference input changed')
    cfg = read(LOCAL/'config.json'); shape = Path(cfg['shape'])
    require(cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and cfg['capture_radius'] == 18. and len(cfg['fixed_poses']) == 2, 'Wrong AB baseline')
    fit = read(FITS/'report.json'); seal = read(FITS/'freeze.json')
    require(fit['complete'] and fit['no_new_physical_draws'] and fit['original_q_window'] == INNER and fit['source_ball_radius_A'] == .5, 'Wrong local fit scope')
    require(all(fit['physical'][k] == cfg[k] for k in PHYSICAL) and sha(FITS/'report.json') == seal['report_sha256'], 'Fit target/report changed')
    for name, digest in seal['archived_sha256'].items(): require(sha(FITS/'provenance'/name) == digest, 'Fit archive changed')
    for name, digest in fit['input_sha256'].items(): require(sha(name) == digest, 'Fit input changed')
    fitted = {name: read(FITS/f'model-{name}.json') for name in CENTERS}
    require(all(sha(FITS/f'model-{n}.json') == seal['model_sha256'][n] for n in CENTERS), 'Fit model changed')
    legacy_report = read(LEGACY/'report.json'); legacy = read(LEGACY/'model-mixture.json')
    require(sha(LEGACY/'model-mixture.json') == legacy_report['model_sha256']['mixture'] and legacy['shape_sha256'] == sha(shape), 'Legacy guide/shape changed')
    require(all(legacy_report['physical'][k] == cfg[k] for k in PHYSICAL), 'Legacy physical baseline differs')
    require(sha(BINARY) == EXPECTED_BINARY, 'Reviewed executable changed')
    mass = {name: reference['independent_assigned_region_sums']['physical'][name]['logQ'] for name in CENTERS}
    models = {}; allocation = None
    for name, scale, _ in ARMS: models[name], allocation = assemble(legacy, fitted, mass, scale)
    prior_seeds = {p['seed'] for c in reference['campaigns']+reference['histories']+direct['campaigns'] for p in c['populations']}
    seeds = [base+1009*i for _, _, base in ARMS for i in range(16)]+[112001010, 112002019]
    require(len(seeds) == len(set(seeds)) and not prior_seeds.intersection(seeds), 'New streams overlap')
    archive = out/'provenance'; archive.mkdir(parents=True)
    inputs = {'finite-reference.json': REFERENCE, 'direct-reference.json': direct_path,
        'input-config.json': LOCAL/'config.json', 'shape.json': shape,
        'fit-report.json': FITS/'report.json', 'fit-freeze.json': FITS/'freeze.json',
        'legacy-model.json': LEGACY/'model-mixture.json', 'legacy-report.json': LEGACY/'report.json',
        **{f'fitted-{n}.json': FITS/f'model-{n}.json' for n in CENTERS},
        **local_dependencies([Path(__file__), ROOT/'tools/run_native_region_reference.py'])}
    for name, path in inputs.items(): shutil.copy2(path, archive/name)
    frozen_cfg = copy.deepcopy(cfg); frozen_cfg['shape'] = str(archive/'shape.json'); write(out/'config.json', frozen_cfg)
    centers = []
    for name in CENTERS:
        source = DIRECT/'model.json' if name == 'direct' else LOCAL/f'model-{name}.json'
        pose = read(DIRECT/'selected-pose.json')['pose'] if name == 'direct' else reference['selection'][name]['pose']
        shutil.copy2(source, out/f'geometric-{name}.json')
        centers.append(dict(name=name, model_path=str(out/f'geometric-{name}.json'), model_sha256=sha(source), pose=pose))
    cover, proof = cover_model(cfg); write(out/'cover-proof.json', proof)
    for name, model in models.items(): write(out/f'model-{name}.json', model)
    arms = [dict(name=name, new_covariance_scale=scale, output=str(ROOT/f'runs/ab-shoulder-contact-atlas-{name}-16x32768-l64-20260921'),
        model_path=str(out/f'model-{name}.json'), model_sha256=sha(out/f'model-{name}.json'), components=5,
        populations=16, samples_per_population=32768, workers=16, seeds=[base+1009*i for i in range(16)]) for name, scale, base in ARMS]
    require(all(not Path(a['output']).exists() for a in arms), 'Existing output must be preserved')
    commands = {a['name']: [sys.executable, str(archive/'run_native_region_reference.py'), '--root', a['output'], '--config', str(out/'config.json'),
        '--binary', str(BINARY), '--expected-binary-sha256', EXPECTED_BINARY, '--replicates', '16', '--samples', '32768', '--workers', '16',
        '--seed-base', str(a['seeds'][0]), '--q-min', '1', '--q-max', '2', '--q-lower-open', '--q-upper-open', '--cover-scales', '1',
        '--model', a['model_path'], '--model-weight', str(BETA), '--model-uniform-probability', str(EPSILON), '--model-anchor-index', '0'] for a in arms}
    help_text = subprocess.run([sys.executable, str(archive/'run_native_region_reference.py'), '--help'], capture_output=True, text=True, check=True).stdout
    for command in commands.values(): validate_command_options(command, help_text)
    (archive/'runner-help.txt').write_text(help_text); write(out/'commands.json', commands)
    protocol = dict(schema='complete-shoulder-calibrated-contact-atlas-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        physical={k: cfg[k] for k in PHYSICAL}, config_sha256=sha(out/'config.json'), shape_sha256=sha(shape), q_window=WINDOW, inner_q_window=INNER,
        executable=str(BINARY), executable_sha256=EXPECTED_BINARY, lambda_ratio=64., cloud_replicates=2, proposal_anchor_index=0,
        hybrid=dict(model_weight=BETA, uniform_probability=EPSILON, cover_scales=[1.]),
        atlas=dict(legacy_probability=.5, fitted_contact_probability=.5, local_allocation=allocation,
                   allocation_source='Independent priority-assigned region weights normalized within the three contacts; proposal allocation only.',
                   fit_rule='One original-weight Gaussian for each whole R.5 source, unchanged additive floor; no clipping or per-population reweighting.',
                   width_control='Only three new fitted covariances multiplied by1 or4; legacy family unchanged.'),
        prefix_counts=[8192, 16384, 32768], arms=arms, maximum_concurrent_workers=32, total_unconditional_draws=1048576,
        analysis=dict(centers=centers, finite_reference_path=str(REFERENCE), finite_reference_sha256=REFERENCE_SHA,
            direct_reference_path=str(direct_path), direct_reference_sha256=sha(direct_path), priority_order=list(CENTERS), radii=[.25, .5],
            rule='Full=inner+outer. Inner=priority union+outside union. Original full N and same-row covariance; no prior/fresh stitching.'),
        cover_proof_sha256=sha(out/'cover-proof.json'), probes=dict(count=256, seeds=[112001010, 112002019]),
        commands_sha256=sha(out/'commands.json'), input_sha256={str(p): sha(p) for p in inputs.values()},
        archived_sha256={p.name: sha(p) for p in archive.iterdir()},
        scope='Frozen complete-shoulder independent importance sampling. Three local fits are truncated-neighborhood proposal models, not entire basin densities. Fresh physical estimates must retain full mixture densities and remaining support. No MCMC or assembly claim.')
    write(out/'protocol.json', protocol)
    write(out/'freeze.json', dict(protocol_sha256=sha(out/'protocol.json'), config_sha256=sha(out/'config.json'),
        model_sha256={name: sha(out/f'model-{name}.json') for name in models}, commands_sha256=sha(out/'commands.json'), archived_sha256=protocol['archived_sha256']))
    atom = AtomUnionAudit(read(shape), cfg['fixed_poses']); probes = {}
    for i, (name, _, _) in enumerate(ARMS):
        probes[name], rows = probe(models[name], cfg, cover, atom, protocol['probes']['seeds'][i])
        with (out/f'geometry-probes-{name}.jsonl').open('x') as handle:
            for row in rows: handle.write(json.dumps(row, allow_nan=False)+'\n')
    write(out/'report.json', dict(complete=True, prepared_only=True, no_physical_draws=True, protocol_sha256=sha(out/'protocol.json'),
        freeze_sha256=sha(out/'freeze.json'), commands_sha256=sha(out/'commands.json'), probes=probes,
        production_allocation_unchanged_after_probes=True, scope=protocol['scope']))
    print(dict(complete=True, prepared_only=True, protocol_sha256=sha(out/'protocol.json'), probes=probes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    prepare(parser.parse_args().out)
