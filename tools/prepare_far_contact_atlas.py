#!/usr/bin/env python3
"""Freeze a calibrated guide and two complete-far importance campaigns."""
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

from analyze_far_peak_reference import PHYSICAL, WINDOW
from analyze_native_region_reference import GaussianGuide, hybrid_log_density, native_q
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_far_capture_control import BINARY, EXPECTED_BINARY, capture_proof
from prepare_native_confirmation_atlas import AtomUnionAudit, to_world
from prepare_peak_neighborhood import geometry_model
from prepare_smc_normalizer_atlas import Density, arrays, relative_poses
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT/'runs/ab-competing-frozen-atlas-20260920'
CENTERS = ROOT/'runs/ab-far-contact-centers-20260921/analysis.json'
FIT = ROOT/'runs/ab-far-local-guide-20260921'
LOCAL = ROOT/'runs/ab-far-peak-reference-preparation-20260920'
ARMS = (('narrow', .2, 107101010), ('broad', .4, 107201010))
SAMPLES, POPULATIONS = 32768, 8
PREFIXES = (8192, 16384, 32768)
BETA, EPSILON = .99, .05


def assemble(cfg, centers, old, fitted, width):
    require(centers and math.isfinite(width) and width > 0, 'Need geometric centers and positive width')
    keys = ('schema', 'angular_length', 'coordinate_convention', 'shape_sha256')
    require(all(old[k] == fitted[k] for k in keys), 'Old and new model conventions differ')
    require(fitted['weights'] == [1.] and len(fitted['anchors']) == len(fitted['means']) == len(fitted['covariances']) == 1,
            'One new local fitted component required')
    require(abs(sum(old['weights'])-1.) < 2e-12 and all(w > 0 for w in old['weights']), 'Normalized positive old model required')
    model = {k: copy.deepcopy(old[k]) for k in keys}
    for k in ('anchors', 'means', 'covariances'): model[k] = copy.deepcopy(old[k])
    model['weights'] = [.2*w for w in old['weights']]
    for center in centers:
        g, _ = geometry_model(cfg['metadata'], cfg['fixed_poses'][0], center['pose'], model['shape_sha256'], model['angular_length'])
        model['anchors'].extend(g['anchors']); model['means'].extend(g['means'])
        model['covariances'].append((width**2*np.asarray(g['covariances'][0])).tolist())
        model['weights'].append(.6/len(centers))
    for scale, weight in ((1., .16), (4., .04)):
        model['anchors'].extend(copy.deepcopy(fitted['anchors'])); model['means'].extend(copy.deepcopy(fitted['means']))
        model['covariances'].append((scale*np.asarray(fitted['covariances'][0])).tolist()); model['weights'].append(weight)
    require(len({len(model[k]) for k in ('anchors', 'means', 'covariances', 'weights')}) == 1, 'Mixture arrays differ')
    require(abs(sum(model['weights'])-1.) < 2e-12, 'Unnormalized atlas')
    return model


def probe(model, cfg, atom, count, seed):
    rng = np.random.default_rng(seed); density = Density(model)
    components = rng.choice(len(model['weights']), size=count, p=model['weights'])
    rows = []
    for i, component in enumerate(components):
        pose = to_world(density.draw_component(rng, int(component), 1), cfg['fixed_poses'][0])[0]
        q = native_q(cfg['metadata'], pose); gaps = atom.gaps(pose)
        capture = math.dist(pose['position'], cfg['capture_center']) <= cfg['capture_radius']
        rows.append(dict(draw=i, component=int(component), pose=pose, original_q=q, minimum_AB_gaps_A=gaps,
            capture_valid=capture, hard_valid=min(gaps) >= 0, in_far_window=5 <= q < 37,
            far_valid=capture and min(gaps) >= 0 and 5 <= q < 37))
    poses = [r['pose'] for r in rows]; positions = np.asarray([p['position'] for p in poses])-cfg['capture_center']
    raw = density.evaluate(relative_poses(poses, cfg['fixed_poses'][0]))[0]
    desc = dict(weight=BETA, uniform_probability=EPSILON, anchor_index=0, anchor_pose=cfg['fixed_poses'][0],
        cube_lengths=[36.]*3, capture_center=cfg['capture_center'])
    guide = GaussianGuide(desc, model, cfg, model['shape_sha256'])
    cube = np.all((positions >= -18.) & (positions < 18.), axis=1)
    ball = np.linalg.norm(positions, axis=1) <= 74.; volume = 4*math.pi/3*74**3
    expected_guide = np.logaddexp(math.log1p(-EPSILON)+raw, np.where(cube, math.log(EPSILON)-3*math.log(36.), -math.inf))
    expected = np.logaddexp(math.log(BETA)+expected_guide, np.where(ball, math.log1p(-BETA)-math.log(volume), -math.inf))
    cover = dict(covers=[dict(reference=cfg['metadata']['native_poses'][0], centroid=[0.]*3,
        ball_radius=74., angle_cap=math.pi, volume=volume)], weights=[1.], scales=[1.])
    actual_guide = np.array([guide.log_density(p)[0] for p in poses])
    actual = np.array([hybrid_log_density(cover, guide, p)[0] for p in poses])
    errors = dict(guide=float(np.max(np.abs(actual_guide-expected_guide))), full_hybrid=float(np.max(np.abs(actual-expected))))
    require(max(errors.values()) < 2e-9, 'Independent full-density reconstruction failed')
    return dict(seed=seed, unconditional_draws=count, far_valid=sum(r['far_valid'] for r in rows),
        hard_valid=sum(r['hard_valid'] for r in rows), capture_valid=sum(r['capture_valid'] for r in rows),
        in_far_window=sum(r['in_far_window'] for r in rows), density_max_errors=errors,
        component_counts=np.bincount(components, minlength=len(model['weights'])).tolist(),
        scope='Raw atlas draws and independent geometry/full-density checks. No physical clouds or acceptance-rate estimate.'), rows


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Fresh preparation required')
    selection = read(CENTERS); require(selection['complete'], 'Completed geometric center selection required')
    center_protocol = read(CENTERS.parent/'protocol.json')
    require(sha(CENTERS.parent/'protocol.json') == selection['protocol_sha256'], 'Center protocol changed')
    for name, digest in selection['archived_sha256'].items(): require(sha(CENTERS.parent/'provenance'/name) == digest, 'Center archive changed')
    for path, digest in selection['source_sha256'].items(): require(sha(path) == digest, 'Center input changed')
    fitted = read(FIT/'model-weighted.json'); fit = read(FIT/'report.json'); seal = read(FIT/'freeze.json')
    require(fit['complete'] and fit['no_new_geometry_or_physical_draws'], 'Completed local fit required')
    require(sha(FIT/'report.json') == seal['report_sha256'] and sha(FIT/'model-weighted.json') == seal['model_sha256']['weighted'], 'Frozen fit changed')
    for name,digest in seal['archived_sha256'].items(): require(sha(FIT/'provenance'/name) == digest, 'Fit archive changed')
    for path,digest in fit['input_sha256'].items(): require(sha(path) == digest, 'Fit input changed')
    old = read(PREVIOUS/'model-mixture.json'); oldseal = read(PREVIOUS/'report.json')
    require(sha(PREVIOUS/'model-mixture.json') == oldseal['outputs']['model-mixture.json'] and len(old['weights']) == 119, 'Old 119-component model changed')
    cfg = read(LOCAL/'config.json'); shape = Path(cfg['shape'])
    require(all(cfg[k] == fit['physical'][k] == center_protocol['physical'][k] for k in PHYSICAL), 'Different physical target in inputs')
    require(sha(shape) == old['shape_sha256'] == fitted['shape_sha256'], 'Shape changed')
    require(sha(BINARY) == EXPECTED_BINARY, 'Reviewed physical executable changed')
    proof = capture_proof(cfg); require(cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035 and len(cfg['fixed_poses']) == 2, 'Different baseline')
    centers = selection['centers']; candidates = selection['candidates']; indexes = selection['center_candidate_indices']
    members, _, _ = arrays(cfg['metadata']['rigid_members']); t, _, rotations = arrays([p['pose'] for p in candidates])
    world = np.einsum('nij,mj->nmi', rotations, members)+t[:, None, :]
    distance = np.sqrt(np.mean(np.sum((world[:,None]-world[np.asarray(indexes)][None,:])**2, axis=3), axis=2))
    require(float(distance.min(axis=1).max()) <= .5, 'Observed candidate cover incomplete')
    require([p['pose'] for p in centers] == [candidates[i]['pose'] for i in indexes], 'Center index mismatch')
    require(centers[0]['pose'] == read(LOCAL/'selected-pose.json')['pose'], 'Peak must be first center')
    models = {name: assemble(cfg, centers, old, fitted, width) for name,width,_ in ARMS}
    archive = out/'provenance'; archive.mkdir(parents=True)
    sources = dict(input_config=LOCAL/'config.json', shape=shape, center_selection=CENTERS,
        local_fit_model=FIT/'model-weighted.json', local_fit_report=FIT/'report.json', local_fit_freeze=FIT/'freeze.json',
        old_model=PREVIOUS/'model-mixture.json', old_model_seal=PREVIOUS/'report.json',
        local_reference=ROOT/'runs/ab-far-peak-reference-assessment-20260920/analysis.json',
        geometric_model=LOCAL/'model.json', old_region=PREVIOUS/'region-competitor-r3.json')
    sources = {k+'.json': v for k,v in sources.items()}
    sources.update(local_dependencies([Path(__file__)]))
    for name,path in sources.items(): shutil.copy2(path, archive/name)
    frozen_cfg = copy.deepcopy(cfg); frozen_cfg['shape'] = str(archive/'shape.json'); write(out/'config.json', frozen_cfg)
    write(out/'capture-proof.json', proof)
    protocol = dict(schema='complete-far-calibrated-contact-atlas-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        physical={k: cfg[k] for k in PHYSICAL}, q_window=WINDOW, executable=str(BINARY), executable_sha256=EXPECTED_BINARY,
        lambda_ratio=64., cloud_replicates=2, proposal_anchor_index=0,
        atlas='20% unchanged old119 +60% equal geometric centers +16% local weighted fit +4% local fit with covariance times4.',
        hybrid=dict(model_weight=BETA, uniform_probability=EPSILON, cover_scales=[1.],
            description='1% complete q37 product ball/Haar plus99%*(95% atlas+5% capture cube/Haar); full mixture denominator for every draw.'),
        center_selection=dict(path=str(CENTERS), sha256=sha(CENTERS), count=len(centers), candidates=len(candidates),
            maximum_nearest_member_RMS_A=float(distance.min(axis=1).max())),
        local_fit=dict(path=str(FIT/'model-weighted.json'), sha256=sha(FIT/'model-weighted.json'), source_ball_radius_A=.5),
        analysis=dict(local_model_path=str(LOCAL/'model.json'), local_model_sha256=sha(LOCAL/'model.json'),
            local_reference_path=str(sources['local_reference.json']), local_reference_sha256=sha(sources['local_reference.json']),
            masks=['full', 'ball0p5', 'ball1', 'ball2', 'radial_0', 'radial_1', 'radial_2', 'outside2'],
            rule='Fixed full-N positive masks with same-row covariance. Disjoint radial shells and outside2 sum to complete far mass. Nested balls are not added.'),
        arms=[dict(name=name, geometric_latent_SD_A=width, components=len(models[name]['weights']),
            populations=POPULATIONS, samples_per_population=SAMPLES, seeds=[base+1009*i for i in range(POPULATIONS)],
            output=str(ROOT/f'runs/ab-far-contact-atlas-{name}-8x32768-l64-20260921')) for name,width,base in ARMS],
        prefix_counts=list(PREFIXES), prefix_rule='First8192,16384,32768 rows per population; prefixes are correlated and full-stream covariance is retained. No refitting, stopping or replacement.',
        probes=dict(draws_per_arm=256, seeds=[107001010,107002019]), capture_proof_sha256=sha(out/'capture-proof.json'),
        validation='All-row original-density/q/Poisson audits; matched finite-region calibration, independent population and width errors, contribution concentration, prefixes, and all remaining-space weights. Observed errors do not bound unseen tails.',
        scope='Frozen independent importance sampling. Local source rows calibrate the proposal; future rows are independent. This does not measure MCMC mixing or template-free assembly.',
        input_sha256={str(p):sha(p) for p in sources.values()}, archived_sha256={n:sha(archive/n) for n in sources})
    for arm in protocol['arms']: require(not Path(arm['output']).exists(), 'Production path already exists')
    write(out/'protocol.json', protocol)
    for name,m in models.items():
        m['proposal_provenance'] = dict(protocol_sha256=sha(out/'protocol.json'), arm=name, scope=protocol['scope'])
        write(out/f'model-{name}.json', m)
    write(out/'freeze.json', dict(protocol_sha256=sha(out/'protocol.json'), config_sha256=sha(out/'config.json'),
        model_sha256={name:sha(out/f'model-{name}.json') for name in models}, archived_sha256=protocol['archived_sha256']))
    atom = AtomUnionAudit(read(shape), cfg['fixed_poses']); probes = {}
    for i,(name,_,_) in enumerate(ARMS):
        check, rows = probe(models[name], cfg, atom, 256, protocol['probes']['seeds'][i]); probes[name] = check
        with (out/f'geometry-probes-{name}.jsonl').open('x') as f:
            for row in rows: f.write(json.dumps(row, allow_nan=False)+'\n')
        print(dict(arm=name, components=len(models[name]['weights']), **check), flush=True)
    commands = {}
    for arm in protocol['arms']:
        command = [sys.executable, str(ROOT/'tools/run_native_region_reference.py'), '--root', arm['output'], '--config', str(out/'config.json'),
            '--binary', str(BINARY), '--expected-binary-sha256', EXPECTED_BINARY, '--replicates', str(POPULATIONS), '--samples', str(SAMPLES),
            '--workers', '8', '--seed-base', str(arm['seeds'][0]), '--q-min', '5', '--q-max', '37', '--q-upper-open', '--cover-scales', '1',
            '--model', str(out/f"model-{arm['name']}.json"), '--model-weight', str(BETA), '--model-uniform-probability', str(EPSILON), '--model-anchor-index', '0']
        commands[arm['name']] = command
    write(out/'commands.json', commands)
    write(out/'report.json', dict(complete=True, probes=probes, freeze_sha256=sha(out/'freeze.json'),
        probe_sha256={name:sha(out/f'geometry-probes-{name}.jsonl') for name in models}, commands_sha256=sha(out/'commands.json'),
        no_physical_draws=True, scope='Frozen model and geometry/full-density probes only. Runner commands use independent predeclared production streams.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    prepare(p.parse_args().out)
