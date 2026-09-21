#!/usr/bin/env python3
"""Freeze an inert, native-informed, three-mobile-tetramer posterior pilot.

No executable is selected, built or launched. Proposal models are combined
without fitting; their allocation weights are not physical occupancies.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import math
from pathlib import Path
import shutil

from prepare_smc_normalizer_atlas import Density, arrays, read, sha, write
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
RADIUS_12 = 354.50820786337056
RADIUS = RADIUS_12*(3/12)**(1/3)
PREPARATION_SEED = 115401010
DENSITY_CHECK_SEED = 115401011
SEED_BASE, SEED_STRIDE = 115501010, 1009
SWEEPS, BURN, WORKERS = 2000, 400, 12
MODES = ('capture_only', 'c0', 'c09')
STARTS = ('dispersed', 'preassociated')
PENDING_BINARY = '<REVIEWED_BINARY_PENDING>'
MODELS = (
    ('native', 'examples/frozen-relative-mixture.json', .5, 28,
     '2e534634e6e2fe83da969a2867504c293af8e90a0cf3c4330cdc9e38a6770064'),
    ('outside_r8', 'runs/outside-r8-atlas-extension-20260920/site0/model.json', .4, 117,
     '54423aef3a1317e2fe8369f9cc8e6dad8effad4c72ff18a76fdcc5f58566458d'),
    ('shoulder', 'runs/ab-shoulder-contact-atlas-preparation-20260921/model-broad.json', .1, 5,
     'daf2d3adb096453b6e91f1dfe8f614a614fb053f139e2bde386b36d9b8b2c901'),
)
INPUTS = {
    'tetramer-shape.json': ('examples/tetramer-shape.json', 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'),
    'monomer-shape.json': ('examples/monomer-shape.json', '2e5296635dd09f3d445476bf8a39fd6be2f9cbdf4d3de9905e14f4b5a7848afe'),
    'native-pair-motifs.json': ('examples/native-pair-motifs.json', '4c0309b21bf7c31e12bb2f11fb103b20c5772a8472bfc59c2cc0ff8c6ff51282'),
    'dispersed-12-source.json': ('examples/spherical-free.json', 'b8e27a8c08edbb9d658d3e37fce92585d8ab2d91eb11385eacfc92a52d38721b'),
    'AB-source-config.json': ('runs/ab-shoulder-contact-atlas-preparation-20260921/config.json',
                              '55aab739e2d7d9cdd8a19a0eb30c81715d2ead394b3282baff87f14f30d64d4d'),
}
FORBIDDEN = {'capture_center', 'capture_radius', 'target_region', 'fixed_poses',
             'auxiliary_transport', 'reversible_jump', 'contact_memory',
             'conditional_closure', 'atlas_transport', 'atlas_mask'}
SCOPE = ('Native-informed sampling control with three fully mobile rigid tetramers. '
         'The physical target is hard atomic unions inside a spherical protein wall times '
         'exp(-z times total expanded-sphere union volume); the depletant bath is wall-permeable. '
         'No q/capture restriction, equilibrium claim, template-free assembly claim, '
         'or physical kinetic-rate interpretation. Frozen proposal masses are not physical occupancies.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_model(model, shape_hash):
    require(model['schema'] == 'weighted-pose-mixture-v1' and model['coordinate_convention'] == 'anchor-body-relative',
            'Require a frozen anchor-body-relative Gaussian mixture')
    require(model['shape_sha256'] == shape_hash, 'Model physical shape differs')
    count = len(model['weights'])
    require(count > 0 and all(len(model[k]) == count for k in ('anchors', 'means', 'covariances')), 'Missing model component')
    ell = model['angular_length']
    require(math.isfinite(ell) and ell > 0, 'Invalid angular length')
    weights = np.asarray(model['weights'], dtype=float)
    require(np.all(np.isfinite(weights)) and np.all(weights > 0) and abs(weights.sum()-1.) < 1e-12,
            'Require positive normalized component weights')
    means, covariance = np.asarray(model['means'], dtype=float), np.asarray(model['covariances'], dtype=float)
    require(means.shape == (count, 6) and covariance.shape == (count, 6, 6)
            and np.all(np.isfinite(means)) and np.all(np.isfinite(covariance)), 'Invalid model coordinates')
    require(np.allclose(covariance, covariance.swapaxes(1, 2), rtol=1e-12, atol=1e-12), 'Covariance is not symmetric')
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError('Covariance must be strictly positive definite; no repair permitted') from error
    for anchor in model['anchors']:
        t, r = np.asarray(anchor['position']), np.asarray(anchor['rotation'])
        require(t.shape == (3,) and r.shape == (3, 3) and np.isfinite(t).all() and np.isfinite(r).all(), 'Invalid chart anchor')
        require(np.allclose(r.T@r, np.eye(3), atol=1e-10) and abs(np.linalg.det(r)-1.) < 1e-10,
                'Anchor must be a proper rotation')
    return count


def combine_models(models, masses, shape_hash, angular_length=None):
    """Change only angular coordinate units and predeclared group allocations."""
    require(len(models) == len(masses) > 0 and all(math.isfinite(x) and x > 0 for x in masses)
            and abs(sum(masses)-1.) < 1e-12, 'Invalid model allocation masses')
    ell = models[0]['angular_length'] if angular_length is None else angular_length
    require(math.isfinite(ell) and ell > 0, 'Invalid common angular length')
    result = dict(schema='weighted-pose-mixture-v1', coordinate_convention='anchor-body-relative',
                  shape_sha256=shape_hash, angular_length=ell, anchors=[], means=[], covariances=[], weights=[])
    groups = []
    for model, mass in zip(models, masses):
        count = validate_model(model, shape_hash)
        start = len(result['weights'])
        scale = np.diag([1.]*3+[ell/model['angular_length']]*3)
        result['anchors'].extend(copy.deepcopy(model['anchors']))
        result['means'].extend((np.asarray(model['means'])@scale.T).tolist())
        result['covariances'].extend((scale@np.asarray(model['covariances'])@scale.T).tolist())
        result['weights'].extend((mass*np.asarray(model['weights'])).tolist())
        groups.append(dict(first_component=start, components=count, proposal_mass=mass,
                           original_angular_length=model['angular_length'], angular_coordinate_scale=ell/model['angular_length']))
    validate_model(result, shape_hash)
    return result, groups


def verify_density_identity(combined, models, masses, seed=DENSITY_CHECK_SEED):
    """Check the full physical G=sum mass*G_group, including normalized Haar."""
    rng = np.random.default_rng(seed)
    densities = [Density(model) for model in models]
    poses = []
    for model, density in zip(models, densities):
        for component in range(len(model['weights'])):
            poses.extend(density.draw_component(rng, component, 2))
    # Independent broad off-center orientations supplement every source component.
    quaternions = rng.normal(size=(32, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
    poses.extend(dict(position=t.tolist(), orientation=q.tolist())
                 for t, q in zip(rng.normal(size=(32, 3))*60., quaternions))
    wanted = logsumexp(np.array([density.evaluate(poses)[0]+math.log(mass)
                                for density, mass in zip(densities, masses)]), axis=0)
    actual = Density(combined).evaluate(poses)[0]
    require(np.isfinite(wanted).all() and np.isfinite(actual).all(), 'Nonfinite physical mixture density')
    error = float(np.max(np.abs(actual-wanted)))
    require(np.allclose(actual, wanted, rtol=2e-12, atol=2e-9), 'Full physical G mixture identity failed')
    return dict(passed=True, poses=len(poses), seed=seed, maximum_log_density_difference=error,
                definition='G_new(x)=sum_group mass_group G_group(x), relative to d3t times normalized Haar; no defensive uniform mixed into G.',
                includes_all_components=True, fitting_performed=False)


def atomic_start_check(poses, shape, radius, rd=1.5, require_no_contacts=False):
    """Independently check every interbody atomic core pair and every wall atom."""
    require(math.isfinite(radius) and radius > 0 and math.isfinite(rd) and rd >= 0, 'Invalid start-check geometry')
    t, _, rotations = arrays(poses)
    atoms = np.asarray([a['center'] for a in shape['atoms']], dtype=float)
    radii = np.asarray([a['radius'] for a in shape['atoms']], dtype=float)
    require(atoms.ndim == 2 and atoms.shape[1] == 3 and len(atoms) > 0 and np.isfinite(atoms).all()
            and np.isfinite(radii).all() and np.all(radii > 0), 'Invalid atomic union')
    world = np.einsum('bij,aj->bai', rotations, atoms)+t[:, None, :]
    clearance = (radius-np.linalg.norm(world, axis=2)-radii).min(axis=1)
    pairs = []
    for i, j in itertools.combinations(range(len(poses)), 2):
        gap, exclusion_pairs = math.inf, 0
        for first in range(0, len(atoms), 256):
            distances = cdist(world[i, first:first+256], world[j])-radii[first:first+256, None]-radii[None, :]
            gap = min(gap, float(distances.min()))
            exclusion_pairs += int(np.count_nonzero(distances < 2*rd))
        pairs.append(dict(bodies=[i, j], minimum_atomic_core_gap_A=gap,
                          overlapping_exclusion_sphere_pairs=exclusion_pairs))
    hard = all(p['minimum_atomic_core_gap_A'] >= 0 for p in pairs)
    wall = bool(np.all(clearance >= 0))
    no_contacts = all(p['overlapping_exclusion_sphere_pairs'] == 0 for p in pairs)
    return dict(passed=hard and wall and (no_contacts or not require_no_contacts),
                bodies=len(poses), atoms_per_body=len(atoms), hard_valid=hard, wall_valid=wall,
                no_initial_exclusion_contacts=no_contacts, required_no_contacts=require_no_contacts,
                minimum_atomic_wall_clearance_by_body_A=clearance.tolist(), pairs=pairs,
                interpretation='All atomic sphere pairs between distinct bodies and all wall atoms checked. Intrinsic overlapping spheres form a rigid union, not additional mobile bodies. No repair or minimization.')


def prepare_starts(source, ab, shape, radius=RADIUS):
    require(len(source['initial_poses']) == 12 and source['boundary'] == dict(kind='spherical', radius=RADIUS_12),
            'Dispersed source is not the frozen 12-body sphere')
    dispersed = copy.deepcopy(source['initial_poses'][:3])
    for pose in dispersed:
        pose['position'] = (np.asarray(pose['position'])*(radius/RADIUS_12)).tolist()
    first_check = atomic_start_check(dispersed, shape, radius, require_no_contacts=True)
    generation = dict(source_indices=[0, 1, 2], position_scale=radius/RADIUS_12,
                      orientations_unchanged=True, first_scaled_candidate_check=first_check,
                      fallback_used=False, preparation_seed=PREPARATION_SEED)
    if not first_check['passed']:
        # Generate a fresh separated preparation, never move overlapping bodies
        # incrementally or repair the frozen preassociated configuration.
        centers = np.asarray([a['center'] for a in shape['atoms']])
        radii = np.asarray([a['radius'] for a in shape['atoms']])
        bound = float(np.max(np.linalg.norm(centers, axis=1)+radii))
        inner = radius-bound-1e-6
        require(inner > 0, 'Sphere cannot hold the rigid atomic bound')
        rng = np.random.default_rng(PREPARATION_SEED)
        accepted, draws = [], 0
        while len(accepted) < 3 and draws < 100000:
            direction = rng.normal(size=3)
            point = direction/np.linalg.norm(direction)*inner*rng.random()**(1/3)
            draws += 1
            if all(np.linalg.norm(point-previous) > 2*(bound+1.5)+1e-6 for previous in accepted):
                accepted.append(point)
        require(len(accepted) == 3, 'Fixed-seed independent separated preparation failed')
        for pose, position in zip(dispersed, accepted):
            pose['position'] = position.tolist()
        generation.update(fallback_used=True, proposals=draws, preparation_law='Sequential uniform inner-ball positions conditioned on exclusion-bound separation; orientations from frozen indices 0–2; not equilibrium.')
    require(len(ab['fixed_poses']) == 2 and ab['metadata']['native_poses'][0] ==
            dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]), 'Original AB/native start differs')
    associated = copy.deepcopy(ab['fixed_poses']+[ab['metadata']['native_poses'][0]])
    centroid = np.mean([p['position'] for p in associated], axis=0)
    for pose in associated:
        pose['position'] = (np.asarray(pose['position'])-centroid).tolist()
    starts = dict(dispersed=dispersed, preassociated=associated)
    checks = {name: atomic_start_check(poses, shape, radius, require_no_contacts=name == 'dispersed')
              for name, poses in starts.items()}
    require(all(check['passed'] for check in checks.values()), 'Initial atomic wall/core/contact checks failed; no repair permitted')
    return starts, dict(checks=checks, dispersed_generation=generation,
                        preassociated_generation=dict(body_order=['released_A', 'released_B', 'native_identity'],
                            subtracted_centroid=centroid.tolist(), orientations_unchanged=True,
                            original_bodies_all_mobile=True, initial_pose_repair=False), equilibrated=False)


def analysis_plan():
    return dict(schema='mobile-posterior-pilot-analysis-plan-v1', sweeps=SWEEPS, burn_sweeps=BURN,
        retain='Every sweep endpoint 401–2000 including rejected/repeated states; every postburn move for event attribution.',
        primary_comparison='c0 versus c09: same .5 posterior fraction conditional on global slot and same .25 unconditional capture probability.',
        contextual_control='capture_only has .5 unconditional capture probability; its global allocation differs from either posterior arm.',
        native_graph=dict(definition='Frozen native-pair-motifs plus external member-pair native residue-patch registration; intrinsic contacts excluded.',
            maximum_member_error_entry_A=2., body_rotation_entry_degrees=15.,
            maximum_member_error_retention_A=3., body_rotation_retention_degrees=20.,
            external_member_pose_entry_A=3., external_member_rotation_entry_degrees=20.,
            external_member_pose_retention_A=5., external_member_rotation_retention_degrees=30.,
            native_reference_patch_gap_A=1., shared_patch_entry_gap_A=2., shared_patch_retention_gap_A=3.,
            minimum_shared_native_residue_pairs=1),
        nonspecific_contact_graph='Unordered interbody pair with minimum atomic hard-surface gap <= 2 rd = 3 A, reported separately from native registration and with native overlap flagged.',
        metrics=['native and nonspecific contact edge/degree/component occupancies',
                 'formation and detachment events at every attempted update, including GCA and common shifts',
                 'partner exchange: loss of an active partner and subsequent gain of a different partner for the same mobile body; report both body/partner identities and intervening states',
                 'residence in sweeps and attempted updates, with left/right censoring retained and no physical-time conversion',
                 'individual label/contact apparent ESS per full postburn sampler CPU, unresolved for constant descriptors',
                 'initialization discrepancies and first/last retained-half occupancies'],
        cpu='Full postburn cumulative sampler CPU frame difference includes all slots, rejections, gate work, GCA, center shift and recorded overhead. Observer costs remain separate.',
        event_attribution='Retain per-run events by local, full-mixture-capture, frozen-posterior-Gaussian, frozen-posterior-uniform, GCA and center-shift kernels; no boundary concatenation.',
        rules=['All 12 runs and all retained repeats; no best-run selection, thinning or optional stopping.',
               'No fitted weights, covariance updates or adaptation from production.',
               'Native-informed proposal and preassociated start are explicit; not template-free discovery.',
               'No equilibrium, stationary efficiency, thermodynamic assembly or physical kinetics inferred from apparent ESS or a short trajectory.'], scope=SCOPE)


def validate_configs(jobs):
    require(len(jobs) == len({j['id'] for j in jobs}) == 12, 'Need the exact 12-run design')
    expected = [(start, replica, mode) for start in STARTS for replica in (0, 1) for mode in MODES]
    poses_by_start = {}
    for index, (job, design) in enumerate(zip(jobs, expected)):
        require(tuple(job[k] for k in ('start', 'replicate', 'mode')) == design, 'Changed factorial allocation')
        cfg = read(job['config'])
        require(not FORBIDDEN.intersection(cfg), 'Forbidden auxiliary or physical support option')
        require(cfg['fixed_body_indices'] == [] and cfg['seed_labels'] == [] and len(cfg['initial_poses']) == 3, 'Not three fully mobile bodies')
        require(cfg['boundary'] == dict(kind='spherical', radius=RADIUS), 'Physical wall changed')
        physical = dict(box_lengths=[2*RADIUS]*3, depletant_radius=1.5, reservoir_density=.035,
            poisson_lambda_ratio=64., global_probability=.5, learned_uniform_weight=.1,
            local_translation_std_A=.2, local_small_angle_std_degrees=1.,
            gca_probability=1., center_shift_probability=1.,
            endpoint_gate=dict(max_cells=2047, max_depth=14, min_width=.5))
        require(all(cfg.get(key) == value for key, value in physical.items()), 'Physical or matched proposal law changed')
        require(cfg['seed'] == job['seed'] == SEED_BASE+SEED_STRIDE*index, 'Changed independent stream')
        require(cfg['initial_poses'] == poses_by_start.setdefault(job['start'], cfg['initial_poses']), 'Arms do not share the frozen start')
        expected_options = None if job['mode'] == 'capture_only' else dict(probability=.5, correlation=.9 if job['mode'] == 'c09' else 0.)
        require(('frozen_posterior' not in cfg) if expected_options is None else cfg.get('frozen_posterior') == expected_options,
                'Changed posterior arm allocation')
        require(sha(job['config']) == job['config_sha256'], 'Configuration hash differs')
        require(job['command'][0] == PENDING_BINARY and '--no-moves' not in job['command'], 'Preparation must remain inert with complete future move logging')


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), 'Fresh preparation directory required')
    sources = {name: ROOT/relative for name, (relative, _) in INPUTS.items()}
    for name, (_, digest) in INPUTS.items():
        require(sha(sources[name]) == digest, 'Frozen preparation input changed: '+name)
    models, masses = [], []
    for name, relative, mass, count, digest in MODELS:
        path = ROOT/relative
        require(sha(path) == digest, 'Frozen model changed: '+name)
        model = read(path)
        require(len(model['weights']) == count, 'Frozen model component count changed')
        sources['model-'+name+'.json'] = path
        models.append(model)
        masses.append(mass)
    shape_hash = sha(sources['tetramer-shape.json'])
    combined, groups = combine_models(models, masses, shape_hash)
    require(len(combined['weights']) == 150, 'Expected all 150 frozen components')
    density = verify_density_identity(combined, models, masses)
    starts, start_checks = prepare_starts(read(sources['dispersed-12-source.json']),
        read(sources['AB-source-config.json']), read(sources['tetramer-shape.json']))
    for filename in ('prepare_mobile_posterior_pilot.py', 'prepare_smc_normalizer_atlas.py', 'test_prepare_mobile_posterior_pilot.py'):
        sources[filename] = ROOT/'tools'/filename
    reference = Path('/home/xvg/protein-nucleation')
    for filename in ('tetramer_order.py', 'audit_tetramer_assembly.py', 'analyze_precursor_exchange.py',
                     'analyze_depletion_mirror_benchmark.py', 'analyze_fluid_sampling.py'):
        sources['native-observer/'+filename] = reference/'scripts'/filename
    sources['native-observer/motifs.json'] = reference/'results/c1c3-scaffold/motifs.json'
    sources['native-observer/classification.json'] = reference/'results/native-neighbor-classes/classification.json'
    provenance = out/'provenance'
    provenance.mkdir(parents=True)
    for name, source in sources.items():
        (provenance/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, provenance/name)
        require(sha(source) == sha(provenance/name), 'Input changed during preparation')
    write(out/'model.json', combined)
    write(out/'initial-states.json', dict(poses=starts, validation=start_checks))
    write(out/'density-identity.json', density)
    write(out/'analysis-plan.json', analysis_plan())
    jobs = []
    (out/'configs').mkdir()
    (out/'runs').mkdir()
    (out/'logs').mkdir()
    for start in STARTS:
        for replica in (0, 1):
            for mode in MODES:
                index = len(jobs)
                identity = f'mobile3-{start}-r{replica:02d}-{mode}'
                seed = SEED_BASE+SEED_STRIDE*index
                config = dict(shape=str(provenance/'tetramer-shape.json'), monomer_shape=str(provenance/'monomer-shape.json'),
                    box_lengths=[2*RADIUS]*3, boundary=dict(kind='spherical', radius=RADIUS),
                    initial_poses=copy.deepcopy(starts[start]), fixed_body_indices=[], seed_labels=[], seed=seed,
                    depletant_radius=1.5, reservoir_density=.035, poisson_lambda_ratio=64.,
                    global_probability=.5, learned_uniform_weight=.1, local_translation_std_A=.2,
                    local_small_angle_std_degrees=1., gca_probability=1., center_shift_probability=1.,
                    endpoint_gate=dict(max_cells=2047, max_depth=14, min_width=.5),
                    metadata=dict(start=start, replicate=replica, mode=mode, preparation_equilibrated=False,
                        native_informed_proposal=True, training_feedback=False,
                        native_pair_motifs=str(provenance/'native-pair-motifs.json'),
                        model_sha256=sha(out/'model.json'), analysis_plan_sha256=sha(out/'analysis-plan.json'), scope=SCOPE))
                if mode != 'capture_only':
                    config['frozen_posterior'] = dict(probability=.5, correlation=.9 if mode == 'c09' else 0.)
                config_path = out/'configs'/(identity+'.json')
                write(config_path, config)
                command = [PENDING_BINARY, 'run', '--config', str(config_path), '--model', str(out/'model.json'),
                    '--method', 'learned', '--out', str(out/'runs'/identity), '--sweeps', str(SWEEPS), '--sample-every', '1']
                jobs.append(dict(id=identity, index=index, start=start, replicate=replica, mode=mode, seed=seed,
                    config=str(config_path), config_sha256=sha(config_path), directory=str(out/'runs'/identity),
                    command=command, log=str(out/'logs'/(identity+'.log'))))
    validate_configs(jobs)
    manifest = dict(schema='mobile-frozen-posterior-pilot-preparation-v1', preparation_only=True, launched=False,
        binary_supplied=False, binary_sha256=None, source_bundle_pending=True,
        binary_requirement='Freeze a reviewed executable and runtime source bundle into a separate new production campaign; this archive is not runnable.',
        sweeps=SWEEPS, burn_sweeps=BURN, sample_every=1, workers=WORKERS, bodies=3, runs=12,
        single_body_attempts_per_run=3*SWEEPS, collective_attempts_per_run=2*SWEEPS,
        total_single_body_attempts=12*3*SWEEPS, total_collective_attempts=12*2*SWEEPS,
        retained_endpoints_per_run=SWEEPS-BURN, seeds=[job['seed'] for job in jobs],
        sphere_radius_A=RADIUS, radius_scaling=dict(original_bodies=12, new_bodies=3, original_radius_A=RADIUS_12,
            formula='R3 = R12 * (3/12)^(1/3)', equal_number_density=True),
        proposal_groups=[dict(group=name, source_path=str(ROOT/relative), source_sha256=digest, **group)
                         for (name, relative, _, _, digest), group in zip(MODELS, groups)],
        unconditional_single_body_kernel_probabilities=dict(capture_only=dict(local=.5, full_mixture_capture=.5),
            c0=dict(local=.5, full_mixture_capture=.25, posterior_gaussian=.225, posterior_uniform=.025),
            c09=dict(local=.5, full_mixture_capture=.25, posterior_gaussian=.225, posterior_uniform=.025)),
        model=str(out/'model.json'), model_sha256=sha(out/'model.json'), shape_sha256=shape_hash,
        analysis_plan_sha256=sha(out/'analysis-plan.json'), initial_states_sha256=sha(out/'initial-states.json'),
        density_identity_sha256=sha(out/'density-identity.json'), jobs=jobs,
        input_sha256={name: sha(provenance/name) for name in sources},
        source_paths={name: str(source) for name, source in sources.items()}, scope=SCOPE)
    write(out/'manifest.json', manifest)
    write(out/'plan.json', manifest)
    write(out/'commands.json', dict(executable_pending=True, launch_authorized=False, jobs=jobs))
    (out/'commands.sh').write_text('#!/usr/bin/env bash\nset -eu\necho "Inert preparation: reviewed binary/source bundle must be frozen in a fresh production campaign." >&2\nexit 2\n')
    (out/'README.md').write_text('# Three-mobile-tetramer frozen posterior pilot — preparation only\n\n'
        'No simulation or executable build was launched. The binary and runtime source bundle remain pending for a separate production campaign. '
        '`commands.sh` always exits before any run; `commands.json` contains nonexecutable command templates.\n\n'
        f'Twelve jobs: two starts × two replicas × three arms, {SWEEPS:,} sweeps each, fixed burn {BURN}, every sweep retained, at most {WORKERS} workers. '
        f'All three tetramers move inside a radius {RADIUS:.12f} Å atomic wall; rd=1.5 Å, z=0.035 Å⁻³.\n\n'
        'The primary comparison is c=0 versus c=0.9 with the same global-slot allocation. Capture-only is contextual because its capture probability is twice as large. '
        'The 150-component proposal combines 0.5 native, 0.4 outside-r8 and 0.1 shoulder model masses with no refit or dropped component. '
        'The source models are native-informed and include previously identified competitors. Proposal allocations are not physical populations.\n\n'
        'Both starts are shared across arms and replicas and are not equilibrated: dispersed positions use scaled archived coordinates if valid, otherwise a fixed-seed separated preparation; '
        'the preassociated start releases A, B and the original identity-native body and recenters the centroid. Every interbody atomic core pair and every wall atom is checked without repairs.\n\n'
        'Read `analysis-plan.json` for the frozen native/nonspecific contact, partner-exchange, residence, initialization and kernel/CPU diagnostics. '
        'All repeats, censoring and independent runs must remain visible. No equilibrium, template-free discovery, or physical kinetics claim follows from this pilot.\n')
    write(out/'freeze.json', dict(schema='immutable-mobile-posterior-preparation-v1', launched=False,
        files={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.out)
    print(f"Prepared {result['runs']} inert jobs; model {result['model_sha256']}; no binary or launch.")
