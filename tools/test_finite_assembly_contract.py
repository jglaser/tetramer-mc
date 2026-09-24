"""Synthetic finite-assembly design tests; no trajectories or geometry calls."""
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import finite_assembly_contract as campaign
from contact_benchmark_contract import ATTEMPTS, SCHEMA as BENCHMARK_SCHEMA, digest


def contract_fixture():
    """A complete, deliberately unpopulated matrix, independently constructed."""
    common = dict(local_translation_std_A=.2, local_small_angle_std_degrees=1.,
        gca_probability=.1, center_shift_probability=.1, poisson_lambda_ratio=64.,
        endpoint_gate=dict(max_cells=2047, max_depth=14, min_width=.5))
    window = dict(burn_sweep=100, end_sweep=400, cadence_sweeps=1)
    models = {family: dict(model=dict(path=family+'.json', sha256=campaign.MODEL_SHA[family]),
        provenance=(dict(manifest=dict(path='blind/manifest.json', sha256=campaign.BLIND_MANIFEST_SHA),
                         freeze=dict(path='blind/freeze.json', sha256=campaign.BLIND_FREEZE_SHA))
                    if family == 'geometry-only' else None))
        for family in ('geometry-only', 'native-informed')}
    result = dict(schema=campaign.SCHEMA,
        physical=dict(shape=dict(path='shape.json', sha256=campaign.SHAPE_SHA),
            concentration_uM=106.8, concentration_tolerance_uM=.05,
            volume_relative_tolerance=1e-12, depletant_radius=1.5, reservoir_density=.035),
        models=models, primary_common_schedule=common, window=window, blocks=[])
    seed = 10000
    for study in ('primary-spherical', 'boundary-comparison'):
        for bodies in (12, 24):
            volume = bodies*1e27/(106.8e-6*6.02214076e23)
            radius = (3*volume/(4*math.pi))**(1/3)
            side = volume**(1/3)
            for boundary in (('spherical',) if study == 'primary-spherical' else ('spherical', 'periodic')):
                for family in ('geometry-only', 'native-informed'):
                    identity = dict(shape_sha256=campaign.SHAPE_SHA, bodies=bodies,
                        boundary=(dict(kind=boundary, radius=radius) if boundary == 'spherical' else dict(kind=boundary)),
                        box_lengths=[2*radius]*3 if boundary == 'spherical' else [side]*3,
                        depletant_radius=1.5, reservoir_density=.035, fixed_body_indices=[], measure=campaign.MEASURE)
                    schedule = copy.deepcopy(common)
                    if study == 'boundary-comparison':
                        schedule.update(gca_probability=0., center_shift_probability=0.)
                    model_hash = campaign.MODEL_SHA[family]
                    benchmark = dict(schema=BENCHMARK_SCHEMA, physical_identity_sha256=digest(identity),
                        observer_definition_sha256='b'*64, region_definition_sha256='c'*64,
                        common_schedule=schedule, window=copy.deepcopy(window),
                        preparations=list(campaign.PREPARATIONS), streams_per_preparation=4,
                        arms=dict(
                            local=dict(role='local', method='local-uniform', model_sha256=None,
                                frozen_posterior=None, global_probability=0., learned_uniform_weight=.1),
                            redraw=dict(role='independent-redraw', method='learned', model_sha256=model_hash,
                                frozen_posterior=None, global_probability=.5, learned_uniform_weight=.1),
                            transport=dict(role='correlated-transport', method='learned', model_sha256=model_hash,
                                frozen_posterior=dict(probability=.9, correlation=.9), global_probability=.5,
                                learned_uniform_weight=.1)), single_body_attempt_budget=ATTEMPTS)
                    block_id = f'{study}-{bodies}-{boundary}-{family}'
                    block = dict(id=block_id, study=study, bodies=bodies, boundary=boundary,
                        model_family=family, physical_identity=identity, benchmark_contract=benchmark, jobs=[])
                    for arm in benchmark['arms']:
                        for preparation in campaign.PREPARATIONS:
                            for stream in range(4):
                                seed += 1
                                block['jobs'].append(dict(id=f'{block_id}-{arm}-{preparation}-{stream}',
                                    arm=arm, preparation=preparation, stream=stream, seed=seed,
                                    config=None, initial_poses_sha256=None,
                                    invocation=dict(method=benchmark['arms'][arm]['method'],
                                        sweeps=window['end_sweep'], sample_every=window['cadence_sweeps'],
                                        model_sha256=benchmark['arms'][arm]['model_sha256'])))
                    result['blocks'].append(block)
    return result


def rebind_physical(block):
    block['benchmark_contract']['physical_identity_sha256'] = digest(block['physical_identity'])


def configuration_fixture(block, job):
    identity = block['physical_identity']
    arm = block['benchmark_contract']['arms'][job['arm']]
    poses = [dict(position=[float(i), 0., 0.], orientation=[1., 0., 0., 0.])
             for i in range(block['bodies'])]
    job['initial_poses_sha256'] = digest(poses)
    return dict(shape='shape.json', initial_poses=poses, seed=job['seed'],
        boundary=copy.deepcopy(identity['boundary']), box_lengths=list(identity['box_lengths']),
        depletant_radius=1.5, reservoir_density=.035, fixed_body_indices=[],
        global_probability=arm['global_probability'], learned_uniform_weight=arm['learned_uniform_weight'],
        frozen_posterior=copy.deepcopy(arm['frozen_posterior']),
        **copy.deepcopy(block['benchmark_contract']['common_schedule']))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def synthetic_files(slot_mode=None, archive_escape=False):
    """Mock external trust anchors only; exercise actual file/hash/config logic.

    This deliberately says nothing about authenticity of scientific assets.
    Production has no trust-anchor override argument and no files are modified.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); archive = root/'blind'
        shape_sha = write_json(root/'shape.json', dict(atoms=[dict(center=[0.,0.,0.],radius=.3)]))
        geometry_sha = write_json(archive/'model.json', dict(synthetic='geometry'))
        native_sha = write_json(root/'native.json', dict(synthetic='native'))
        slots = [dict(replicate=i, slot=j) for i in range(4) for j in range(16)]
        if slot_mode == 'omitted': slots.pop()
        elif slot_mode == 'duplicated': slots[-1] = slots[0]
        elif slot_mode == 'reordered': slots.reverse()
        source = dict(model_sha256=geometry_sha, shape_sha256=shape_sha, source_slot_order=slots,
            base_components=64, virtual_components=128)
        manifest_sha = write_json(archive/'manifest.json', source)
        files = {'manifest.json': manifest_sha, 'model.json': geometry_sha}
        if archive_escape:
            files['../outside.json'] = write_json(root/'outside.json', dict(synthetic=True))
        freeze_sha = write_json(archive/'freeze.json', dict(files=files))
        with patch.multiple(campaign, SHAPE_SHA=shape_sha,
                MODEL_SHA={'geometry-only': geometry_sha, 'native-informed': native_sha},
                BLIND_MANIFEST_SHA=manifest_sha, BLIND_FREEZE_SHA=freeze_sha):
            plan = contract_fixture()
            plan['models']['geometry-only']['model']['path'] = 'blind/model.json'
            plan['models']['native-informed']['model']['path'] = 'native.json'
            block = plan['blocks'][0]; job = block['jobs'][0]
            config = configuration_fixture(block, job); config['shape'] = '../shape.json'
            config_path = root/'configs/first.json'
            job['config'] = dict(path='configs/first.json', sha256=write_json(config_path, config))
            yield root, plan, config_path, config


class AssemblyContractTests(unittest.TestCase):
    def test_complete_matrix_and_independent_volume_calculation(self):
        manifest = contract_fixture()
        report = campaign.validate_contract(manifest)
        self.assertIsInstance(report, dict)
        self.assertTrue(report['complete_design'])
        self.assertFalse(report['input_bindings_complete'])
        self.assertFalse(report['input_files_verified'])
        self.assertFalse(report['geometry_validated'])
        self.assertFalse(report['production_authorized'])
        self.assertEqual(len(manifest['blocks']), 12)
        self.assertEqual(sum(len(b['jobs']) for b in manifest['blocks']), 432)
        volumes = {}
        for block in manifest['blocks']:
            identity = block['physical_identity']; n = block['bodies']
            if block['boundary'] == 'spherical':
                volume = 4*math.pi*identity['boundary']['radius']**3/3
            else:
                volume = math.prod(identity['box_lengths'])
            concentration = n*1e27/(volume*6.02214076e23)*1e6
            self.assertAlmostEqual(concentration, 106.8, places=10)
            volumes[(n, block['boundary'])] = volume
            self.assertEqual(identity['fixed_body_indices'], [])
            self.assertEqual({(j['arm'], j['preparation'], j['stream']) for j in block['jobs']},
                {(arm, prep, stream) for arm in ('local', 'redraw', 'transport')
                 for prep in campaign.PREPARATIONS for stream in range(4)})
        self.assertAlmostEqual(volumes[(24, 'spherical')]/volumes[(12, 'spherical')], 2., places=12)
        for n in (12, 24):
            self.assertAlmostEqual(volumes[(n, 'spherical')]/volumes[(n, 'periodic')], 1., places=12)

    def test_missing_extra_duplicate_or_unplanned_matrix_entries_fail(self):
        for mode in ('missing-block', 'extra-block', 'missing-job', 'extra-job', 'stream', 'preparation', 'arm'):
            manifest = contract_fixture()
            if mode == 'missing-block': manifest['blocks'].pop()
            elif mode == 'extra-block': manifest['blocks'].append(copy.deepcopy(manifest['blocks'][0]))
            elif mode == 'missing-job': manifest['blocks'][0]['jobs'].pop()
            elif mode == 'extra-job': manifest['blocks'][0]['jobs'].append(copy.deepcopy(manifest['blocks'][0]['jobs'][0]))
            elif mode == 'stream': manifest['blocks'][0]['jobs'][0]['stream'] = 4
            elif mode == 'preparation': manifest['blocks'][0]['jobs'][0]['preparation'] = 'other'
            else: manifest['blocks'][0]['jobs'][0]['arm'] = 'other'
            with self.subTest(mode=mode), self.assertRaises(ValueError): campaign.validate_contract(manifest)

    def test_wrong_concentration_size_or_boundary_volume_is_rejected(self):
        for mode in ('concentration', 'looser-concentration-tolerance', 'looser-volume-tolerance', 'sphere', 'box', 'size'):
            manifest = contract_fixture()
            if mode == 'concentration': manifest['physical']['concentration_uM'] = 107.
            elif mode == 'looser-concentration-tolerance': manifest['physical']['concentration_tolerance_uM'] = 100.
            elif mode == 'looser-volume-tolerance': manifest['physical']['volume_relative_tolerance'] = .1
            else:
                block = next(b for b in manifest['blocks'] if b['boundary'] == ('periodic' if mode == 'box' else 'spherical'))
                if mode == 'sphere': block['physical_identity']['boundary']['radius'] *= 1.01
                elif mode == 'box': block['physical_identity']['box_lengths'][0] *= 1.01
                else: block['bodies'] = block['physical_identity']['bodies'] = 16
                rebind_physical(block)
            with self.subTest(mode=mode), self.assertRaises(ValueError): campaign.validate_contract(manifest)

    def test_fixed_shape_bath_and_all_mobile_requirement(self):
        for mode in ('shape', 'radius', 'activity', 'fixed-seed', 'measure'):
            manifest = contract_fixture(); block = manifest['blocks'][0]
            if mode == 'shape': block['physical_identity']['shape_sha256'] = 'a'*64
            elif mode == 'radius': block['physical_identity']['depletant_radius'] = 1.4
            elif mode == 'activity': block['physical_identity']['reservoir_density'] = .04
            elif mode == 'fixed-seed': block['physical_identity']['fixed_body_indices'] = [0, 1, 2, 3]
            else: block['physical_identity']['measure'] = 'different rotational measure'
            rebind_physical(block)
            with self.subTest(mode=mode), self.assertRaises(ValueError): campaign.validate_contract(manifest)

    def test_globally_independent_seeds_and_job_identity(self):
        for mode in ('reused-seed', 'negative', 'overflow', 'boolean', 'duplicate-id'):
            manifest = contract_fixture(); left = manifest['blocks'][0]['jobs'][0]; right = manifest['blocks'][-1]['jobs'][-1]
            if mode == 'reused-seed': right['seed'] = left['seed']
            elif mode == 'negative': right['seed'] = -1
            elif mode == 'overflow': right['seed'] = 2**64
            elif mode == 'boolean': right['seed'] = True
            else: right['id'] = left['id']
            with self.subTest(mode=mode), self.assertRaises(ValueError): campaign.validate_contract(manifest)

    def test_invocation_matches_declared_method_model_and_observation_budget(self):
        for key, bad in [('method', 'local-uniform'), ('model_sha256', None), ('sweeps', 401),
                         ('sample_every', 2), ('sweeps', True), ('sample_every', 1.)]:
            plan = contract_fixture()
            job = next(j for j in plan['blocks'][0]['jobs'] if j['arm'] == 'transport')
            job['invocation'][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                campaign.validate_contract(plan)

    def test_independent_redraw_is_not_zero_correlation_transport(self):
        for role in ('redraw', 'transport'):
            manifest = contract_fixture(); arms = manifest['blocks'][0]['benchmark_contract']['arms']
            arms[role]['frozen_posterior'] = dict(probability=.9, correlation=0.)
            with self.subTest(role=role), self.assertRaises(ValueError): campaign.validate_contract(manifest)

    def test_common_sizes_schedule_and_boundary_safe_collective_moves(self):
        for mode in ('local-size', 'boundary-gca', 'boundary-shift', 'window', 'global-frequency', 'uniform-support'):
            manifest = contract_fixture()
            block = next(b for b in manifest['blocks'] if b['study'] == 'boundary-comparison')
            benchmark = block['benchmark_contract']
            if mode == 'local-size': benchmark['common_schedule']['local_translation_std_A'] = .3
            elif mode == 'boundary-gca': benchmark['common_schedule']['gca_probability'] = .1
            elif mode == 'boundary-shift': benchmark['common_schedule']['center_shift_probability'] = .1
            elif mode == 'window': benchmark['window']['end_sweep'] += 1
            elif mode == 'global-frequency': benchmark['arms']['redraw']['global_probability'] = .6
            else: benchmark['arms']['redraw']['learned_uniform_weight'] = 0.
            with self.subTest(mode=mode), self.assertRaises(ValueError): campaign.validate_contract(manifest)

    def test_geometry_only_provenance_is_bound_and_cannot_be_omitted(self):
        for mode in ('missing', 'manifest', 'freeze', 'model'):
            manifest = contract_fixture(); geometry = manifest['models']['geometry-only']
            if mode == 'missing': geometry['provenance'] = None
            elif mode == 'model': geometry['model']['sha256'] = 'a'*64
            else: geometry['provenance'][mode]['sha256'] = 'a'*64
            with self.subTest(mode=mode), self.assertRaises(ValueError): campaign.validate_contract(manifest)

    def test_physical_hash_observers_and_partial_config_binding_fail(self):
        for mode in ('physical-hash', 'observer', 'region', 'config-only', 'initial-only'):
            manifest = contract_fixture(); block = manifest['blocks'][0]
            if mode == 'physical-hash': block['benchmark_contract']['physical_identity_sha256'] = 'a'*64
            elif mode == 'observer': block['benchmark_contract']['observer_definition_sha256'] = 'a'*64
            elif mode == 'region': block['benchmark_contract']['region_definition_sha256'] = 'a'*64
            elif mode == 'config-only': block['jobs'][0]['config'] = dict(path='config.json', sha256='a'*64)
            else: block['jobs'][0]['initial_poses_sha256'] = 'a'*64
            with self.subTest(mode=mode), self.assertRaises(ValueError): campaign.validate_contract(manifest)


    def test_effective_configuration_matches_each_kernel_size_and_boundary(self):
        plan = contract_fixture()
        for block in plan['blocks']:
            for arm in ('local', 'redraw', 'transport'):
                job = copy.deepcopy(next(j for j in block['jobs'] if j['arm'] == arm))
                config = configuration_fixture(block, job)
                with self.subTest(block=block['id'], arm=arm):
                    self.assertTrue(campaign.validate_configuration(config, block, job))

    def test_configuration_cannot_change_physics_move_law_or_initialization(self):
        for mode in ('seed', 'frozen-body', 'resume', 'body-count', 'pose', 'quaternion',
                     'boundary', 'activity', 'local-size', 'global-rate', 'rho-zero-redraw'):
            plan = contract_fixture(); block = plan['blocks'][0]
            job = copy.deepcopy(next(j for j in block['jobs'] if j['arm'] == 'redraw'))
            config = configuration_fixture(block, job)
            if mode == 'seed': config['seed'] += 1
            elif mode == 'frozen-body': config['fixed_body_indices'] = [0]
            elif mode == 'resume': config['resume'] = 'old-checkpoint.json'
            elif mode == 'body-count':
                config['initial_poses'].pop(); job['initial_poses_sha256'] = digest(config['initial_poses'])
            elif mode == 'pose': config['initial_poses'][0]['position'][0] += 1.
            elif mode == 'quaternion':
                config['initial_poses'][0]['orientation'] = [2.,0.,0.,0.]
                job['initial_poses_sha256'] = digest(config['initial_poses'])
            elif mode == 'boundary': config['boundary']['radius'] += 1.
            elif mode == 'activity': config['reservoir_density'] = .04
            elif mode == 'local-size': config['local_translation_std_A'] *= 2.
            elif mode == 'global-rate': config['global_probability'] = .6
            else: config['frozen_posterior'] = dict(probability=.9, correlation=0.)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                campaign.validate_configuration(config, block, job)

    def test_adaptive_auxiliary_or_bias_modes_are_explicitly_outside_contract(self):
        for mode in campaign.UNSUPPORTED:
            plan = contract_fixture(); block = plan['blocks'][0]; job = copy.deepcopy(block['jobs'][0])
            config = configuration_fixture(block, job); config[mode] = {}
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                campaign.validate_configuration(config, block, job)

    def test_bound_initial_poses_match_across_arms_and_model_families(self):
        for across in ('arms', 'model-families'):
            plan = contract_fixture(); first = plan['blocks'][0]; left = first['jobs'][0]
            if across == 'arms':
                right = next(j for j in first['jobs'] if j['arm'] == 'redraw'
                             and j['preparation'] == left['preparation'] and j['stream'] == left['stream'])
            else: right = plan['blocks'][1]['jobs'][0]
            left['config'] = dict(path='left.json', sha256='a'*64)
            right['config'] = dict(path='right.json', sha256='b'*64)
            left['initial_poses_sha256'] = right['initial_poses_sha256'] = 'e'*64
            self.assertEqual(campaign.validate_contract(plan)['bound_configurations'], 2)
            right['initial_poses_sha256'] = 'f'*64
            with self.subTest(across=across), self.assertRaisesRegex(ValueError, 'same declared initial poses'):
                campaign.validate_contract(plan)

    def test_real_hash_checks_do_not_promote_partial_input_design(self):
        with synthetic_files() as (root, plan, path, config):
            report = campaign.validate_files(plan, root)
            self.assertTrue(report['bound_files_verified'])
            self.assertFalse(report['input_files_verified'])
            self.assertTrue(report['native_blind_provenance_verified'])
            self.assertEqual(report['bound_configurations'], 1)
            self.assertFalse(report['input_bindings_complete'])
            self.assertFalse(report['geometry_validated'])
            self.assertFalse(report['production_authorized'])
            self.assertIn(str(path), report['verified_file_sha256'])

    def test_all_bound_inputs_are_required_for_file_complete_flag(self):
        with synthetic_files() as (root, plan, first_path, first_config):
            for block in plan['blocks']:
                for job in block['jobs']:
                    config = configuration_fixture(block, job); config['shape'] = '../shape.json'
                    relative = 'configs/'+job['id']+'.json'
                    job['config'] = dict(path=relative, sha256=write_json(root/relative, config))
            report = campaign.validate_files(plan, root)
            self.assertEqual(report['bound_configurations'], 432)
            self.assertTrue(report['input_bindings_complete'])
            self.assertTrue(report['bound_files_verified'])
            self.assertTrue(report['input_files_verified'])
            self.assertFalse(report['geometry_validated'])
            self.assertFalse(report['production_authorized'])

    def test_portable_file_hash_does_not_depend_on_newer_python_file_digest(self):
        # Python 3.9 lacks hashlib.file_digest. Poison it even on newer Python
        # so this checks the implementation, not merely interpreter availability.
        payload = bytes(range(256))*4101 + b'final partial block'
        with tempfile.TemporaryDirectory() as tmp:
            for name, contents in [('empty', b''), ('multiple-blocks', payload)]:
                path = Path(tmp)/name; path.write_bytes(contents)
                with patch.object(campaign.hashlib, 'file_digest', create=True,
                        side_effect=AssertionError('Python 3.9 cannot use file_digest')) as unsupported:
                    self.assertEqual(campaign.file_sha(path), hashlib.sha256(contents).hexdigest())
                    unsupported.assert_not_called()

    def test_file_hash_and_effective_seed_are_checked_separately(self):
        for rebind in (False, True):
            with synthetic_files() as (root, plan, path, config):
                config['seed'] += 1; changed = write_json(path, config)
                if rebind: plan['blocks'][0]['jobs'][0]['config']['sha256'] = changed
                with self.subTest(rebind=rebind), self.assertRaisesRegex(ValueError, 'seed differs' if rebind else 'hash mismatch'):
                    campaign.validate_files(plan, root)

    def test_all_native_blind_slots_and_order_are_mandatory(self):
        for mode in ('omitted', 'duplicated', 'reordered'):
            with synthetic_files(slot_mode=mode) as (root, plan, path, config):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'slot inventory'):
                    campaign.validate_files(plan, root)

    def test_native_blind_archive_cannot_escape_its_directory(self):
        with synthetic_files(archive_escape=True) as (root, plan, path, config):
            with self.assertRaisesRegex(ValueError, 'archived provenance path'):
                campaign.validate_files(plan, root)


if __name__ == '__main__':
    unittest.main()
