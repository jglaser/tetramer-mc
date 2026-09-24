"""Window-free preparation controls; synthetic files only, no physical draws."""
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_finite_assembly_config_bank as bank


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')
    return bank.sha(path)


def freeze(root):
    return save(root/'freeze.json', dict(files={p.relative_to(root).as_posix(): bank.sha(p)
        for p in root.rglob('*') if p.is_file() and p.name != 'freeze.json'}))


@contextmanager
def fixture():
    """Mock trust anchors and the completed geometry audit, never scientific data."""
    with tempfile.TemporaryDirectory() as temp:
        repo = Path(temp)
        starts, observer = repo/'runs/starts', repo/'runs/observer'
        blind = repo/'runs/native-blind-memory-proposal-preparation-20260921'
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.1)])
        shape_sha = save(starts/'inputs/shape.json', shape)
        model_shas = dict(geometry_only=save(blind/'model.json', dict(synthetic='blind')),
            native_informed=save(repo/'examples/frozen-coverage-reciprocal-mixture.json', dict(synthetic='native')))
        blind_manifest = dict(source_slot_order=[dict(replicate=i, slot=j) for i in range(4) for j in range(16)],
            base_components=64, virtual_components=128, shape_sha256=shape_sha)
        blind_sha = save(blind/'manifest.json', blind_manifest)
        blind_freeze = freeze(blind)
        states, entries = {}, []
        for n in (12, 24):
            volume = n*1e33/(106.8*6.02214076e23)
            radius, side = (3*volume/(4*math.pi))**(1/3), volume**(1/3)
            for prep in bank.contract.PREPARATIONS:
                for stream in range(4):
                    for boundary in ('spherical', 'periodic'):
                        poses = [dict(position=[float(i), 0., 0.], orientation=[1., 0., 0., 0.]) for i in range(n)]
                        state = dict(initial_poses=poses, initial_poses_sha256=bank.digest(poses),
                            preparation_seed=stream+100, seed_labels=list(range(8)) if prep=='native-seeded' else [],
                            boundary=dict(kind=boundary, **({'radius': radius} if boundary=='spherical' else {})),
                            box_lengths=[2*radius if boundary=='spherical' else side]*3)
                        states[n, prep, stream, boundary] = state
                        ident = f'N{n}-{prep}-r{stream:02d}-{boundary}'
                        save(starts/'starts'/f'{ident}.json', state)
                        save(starts/'starts'/f'{ident}-validation.json', dict(passed=True, synthetic=True))
                        entries.append(dict(id=ident))
        save(starts/'manifest.json', dict(synthetic=True, states=entries))
        save(starts/'plan.json', dict(preparation_seeds=[100, 101, 102, 103]))
        freeze(starts)
        save(observer/'starting-states-manifest.json', bank.read(starts/'manifest.json'))
        (observer/'source').mkdir(parents=True)
        (observer/'source/observer.py').write_text('# Synthetic observer source.\n')
        definitions = []
        with patch.object(bank.contract, 'SHAPE_SHA', shape_sha):
            for n in (12, 24):
                for boundary in ('spherical', 'periodic'):
                    identity = bank.state_identity(states[n, 'dispersed', 0, boundary])
                    relative = f'definitions/N{n}-{boundary}.json'
                    definition = dict(physical_identity=identity, physical_identity_sha256=bank.digest(identity),
                        regions=[dict(id='synthetic-only')],
                        source_sha256={'../source/observer.py': bank.sha(observer/'source/observer.py')})
                    definitions.append(dict(bodies=n, boundary=boundary, path=relative,
                        sha256=save(observer/relative, definition)))
        observer_sha = save(observer/'manifest.json', dict(complete=True, timing_windows=None,
            production_ready=False, starting_states=48, physical_draws=0, definitions=definitions))
        observer_freeze = freeze(observer)
        source = repo/'current-source.rs'
        source.write_text('// Synthetic validated source.\n')
        save(repo/'runs/old/plan.json', dict(master_seed=91, allocation=[dict(seed=92)], probe_seeds=[93]))
        def fake_authenticated(root):
            root = Path(root)
            # Return source states, but validate each copied state's bytes against
            # the synthetic freeze so bank copying/corruption still gets tested.
            bank.archive_files(root)
            return root, bank.read(root/'manifest.json'), bank.read(root/'plan.json'), {}, states, {}, [], []
        with patch.multiple(bank.contract, SHAPE_SHA=shape_sha,
                MODEL_SHA={'geometry-only': model_shas['geometry_only'], 'native-informed': model_shas['native_informed']},
                BLIND_MANIFEST_SHA=blind_sha, BLIND_FREEZE_SHA=blind_freeze), \
             patch.multiple(bank, OBSERVER_MANIFEST_SHA=observer_sha, OBSERVER_FREEZE_SHA=observer_freeze), \
             patch.object(bank, 'authenticated_inputs', side_effect=fake_authenticated), \
             patch.object(bank, 'source_bindings', return_value={str(source): bank.sha(source)}):
            yield repo, starts, observer, states, source


class ConfigBankTests(unittest.TestCase):
    def test_full_bank_configurations_and_independent_seeds(self):
        with fixture() as (repo, starts, observers, states, _):
            out = repo/'new-bank'
            result = bank.prepare(repo, starts, observers, out)
            verified = bank.validate_bank(out)
            self.assertEqual(verified['configs'], 432)
            self.assertEqual(len({j['block'] for j in result['jobs']}), 12)
            self.assertEqual(len(set(result['seeds'])), 432)
            self.assertFalse(result['production_ready'])
            self.assertIsNone(result['window'])
            self.assertIsNone(result['executable'])
            self.assertEqual(result['physical_draws'], 0)
            for job in result['jobs']:
                config = bank.read(out/job['config']['path'])
                state = states[job['bodies'], job['preparation'], job['stream'], job['boundary']]
                self.assertEqual(config['initial_poses'], state['initial_poses'])
                self.assertNotIn('sweeps', config)
                self.assertNotIn('sample_every', config)
                self.assertNotIn('method', config)
                self.assertNotIn('--resume', job['argv_template'])
                self.assertEqual(config['gca_probability'], 1 if job['study']=='primary-spherical' else 0)
                self.assertEqual(config['center_shift_probability'], config['gca_probability'])
                self.assertEqual(config['frozen_posterior'] is not None, job['arm']=='transport')
                self.assertEqual(job['model'] is None, job['arm']=='local')
                self.assertEqual(config['seed'], job['seed'])
                definition = bank.read(out/job['observer']['path'])
                self.assertEqual(job['observer_definition_sha256'], bank.digest(definition))
                self.assertNotEqual(job['observer_definition_sha256'], job['observer']['sha256'])

    def test_config_corruption_is_rejected_even_if_outer_freeze_updated(self):
        with fixture() as (repo, starts, observers, _, _):
            out = repo/'bank'; result = bank.prepare(repo, starts, observers, out)
            path = out/result['jobs'][0]['config']['path']
            config = bank.read(path); config['reservoir_density'] = .04; save(path, config)
            freeze(out)
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                bank.validate_bank(out)

    def test_rebound_config_cannot_change_the_physical_law(self):
        with fixture() as (repo, starts, observers, _, _):
            out = repo/'bank'; result = bank.prepare(repo, starts, observers, out)
            job = result['jobs'][0]; path = out/job['config']['path']
            config = bank.read(path); config['fixed_body_indices'] = [0]; save(path, config)
            job['config']['sha256'] = bank.sha(path); job['argv_template'] = bank.argv_template(job)
            save(out/'manifest.json', result); freeze(out)
            with self.assertRaisesRegex(ValueError, 'Config differs'):
                bank.validate_bank(out)

    def test_no_overwrite_and_external_source_mutation(self):
        with fixture() as (repo, starts, observers, _, source):
            out = repo/'bank'; bank.prepare(repo, starts, observers, out)
            before = bank.sha(out/'freeze.json')
            with self.assertRaisesRegex(ValueError, 'Fresh'):
                bank.prepare(repo, starts, observers, out)
            self.assertEqual(before, bank.sha(out/'freeze.json'))
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'External source changed'):
                bank.validate_bank(out)
            self.assertTrue(bank.validate_bank(out, check_external=False)['complete'])

    def test_history_inventory_and_collision_stop_before_creation(self):
        with fixture() as (repo, starts, observers, _, _):
            inventory = bank.seed_inventory(repo)
            self.assertTrue({91, 92, 93, 100, 101, 102, 103} <= set(inventory['seeds']))
            first = next(bank.job_layout())
            seed = bank.runtime_seed(152101010, first['id'])
            save(repo/'runs/reserved/manifest.json', dict(runtime_seeds=[seed]))
            out = repo/'bank'
            with self.assertRaisesRegex(ValueError, 'collide'):
                bank.prepare(repo, starts, observers, out)
            self.assertFalse(out.exists())

    def test_archive_escape_and_edited_observer_fail_before_creation(self):
        with fixture() as (repo, starts, observers, _, _):
            definition = observers/'definitions/N12-periodic.json'
            definition.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Archived bytes changed'):
                bank.prepare(repo, starts, observers, repo/'bank')
            self.assertFalse((repo/'bank').exists())
            with self.assertRaisesRegex(ValueError, 'escapes'):
                bank.inside(repo, '../escape.json')

    def test_failed_write_is_preserved_and_frozen(self):
        with fixture() as (repo, starts, observers, _, _):
            out = repo/'bank'
            original = bank.write
            def fail_config(path, value):
                if Path(path).parent.name == 'configs':
                    raise OSError('Synthetic disk failure')
                return original(path, value)
            with patch.object(bank, 'write', side_effect=fail_config):
                with self.assertRaisesRegex(OSError, 'disk failure'):
                    bank.prepare(repo, starts, observers, out)
            self.assertTrue((out/'failure.json').exists())
            self.assertTrue((out/'freeze.json').exists())
            with self.assertRaisesRegex(ValueError, 'Failed bank'):
                bank.validate_bank(out)

    def test_schedule_is_exhaustive_and_retains_independent_capture(self):
        for mutation in ('missing', 'extra', 'zero-global', 'zero-floor', 'no-gca', 'no-capture', 'zero-rho'):
            value = copy.deepcopy(bank.DEFAULT_SCHEDULE)
            if mutation == 'missing': value.pop('frozen_posterior')
            elif mutation == 'extra': value['sweeps'] = 100
            elif mutation == 'zero-global': value['global_probability'] = 0
            elif mutation == 'zero-floor': value['learned_uniform_weight'] = 0
            elif mutation == 'no-gca': value['common']['gca_probability'] = 0
            elif mutation == 'no-capture': value['frozen_posterior']['probability'] = 1
            else: value['frozen_posterior']['correlation'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bank.validate_schedule(value)

    def test_layout_and_seed_derivation_are_order_stable_and_uint64(self):
        jobs = list(bank.job_layout())
        self.assertEqual(len(jobs), 432)
        self.assertEqual(len({j['id'] for j in jobs}), 432)
        left = {j['id']: bank.runtime_seed(1, j['id']) for j in jobs}
        right = {j['id']: bank.runtime_seed(1, j['id']) for j in reversed(jobs)}
        self.assertEqual(left, right)
        self.assertTrue(all(0 <= s < 2**64 for s in left.values()))
        for invalid in (-1, 2**64, True):
            with self.assertRaises(ValueError): bank.runtime_seed(invalid, jobs[0]['id'])


if __name__ == '__main__':
    unittest.main()
