"""Synthetic metadata tests only: no protein, proposal or geometry calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_native_class_support_pilot as prepare


def guide_fixture():
    original = dict(schema=prepare.GUIDE_SCHEMA, region_sha256='r'*64,
        defensive_uniform_shell_probability=.5, conditional_probability=1.,
        raw_translation_axes=[0, 1, 2], minimum_conditional_mass=1e-12,
        class_channels=copy.deepcopy(prepare.CHANNELS),
        compiled_native=dict(path='/old/compiled.json', sha256='c'*64), fixed_poses=['sentinel'],
        gaussian_components=[dict(weight=float(i+1), mean=[i*.01]*6,
            covariance=[[float(a == b) for b in range(6)] for a in range(6)]) for i in range(92)])
    rows = [dict(id=identity, training=True, latent=[float(i), .1, .2, .3, .4, .5])
            for i, identity in enumerate(prepare.TRAINING_IDS)]
    added = [dict(training_id=row['id'], latent_sigma=width, weight=.25/24, mean=row['latent'],
        covariance=[[width**2 if a == b else 0. for b in range(6)] for a in range(6)])
        for row in rows for width in prepare.WIDTHS]
    diagnosis = dict(schema='native-class-saved-support-diagnosis-v1', complete=True, passed=True,
                     rows=rows, new_components=added)
    return original, diagnosis


def context_fixture(directory):
    source = directory/'source'; source.mkdir()
    bindings = prepare.Bindings()
    def put(name, value):
        path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
        prepare.write(path, value); bindings.bind(path); return path
    original, diagnosis = guide_fixture()
    refs = {}
    for key, value in [('region', {}), ('shape', {}), ('compiled_native', {}), ('producer_bundle', {}), ('producer', {})]:
        refs[key] = prepare.bound(put(key+'.json', value))
    definition = dict(input_sha256={})
    refs['definition'] = prepare.bound(put('native-definition/definition.json', definition))
    inputs = dict(inputs=refs, seed_namespace=prepare.NAMESPACE)
    ip = put('inputs.json', inputs); vp = put('validation.json', dict(complete=True))
    code = {name: put('code/'+name, {}) for name in prepare.ENTRIES}
    return dict(inputs=inputs, inputs_path=ip, validation_path=vp,
        data=dict(config=dict(shape='oldshape'), original_guide=original, diagnosis=diagnosis,
                  definition=definition, producer_bundle=dict(files={})), bindings=bindings,
        source=code, runtime=dict(file_sha256={})), dict(seeds=[], files={})


class PreparationTests(unittest.TestCase):
    def test_mixture_preserves_original_law_and_exact_center_width_inventory(self):
        original, diagnosis = guide_fixture(); saved = copy.deepcopy(original)
        compiled = dict(path='/new/compiled.json', sha256='c'*64)
        candidate, inventory = prepare.build_guide(original, diagnosis, compiled)
        self.assertEqual(original, saved)
        self.assertEqual(len(candidate['gaussian_components']), 116)
        for before, after in zip(original['gaussian_components'], candidate['gaussian_components']):
            self.assertEqual(after['mean'], before['mean'])
            self.assertEqual(after['covariance'], before['covariance'])
        self.assertAlmostEqual(sum(c['weight'] for c in candidate['gaussian_components'][:92]), .75)
        self.assertAlmostEqual(sum(c['weight'] for c in candidate['gaussian_components'][92:]), .25)
        self.assertEqual({k:v for k,v in candidate.items() if k not in ('gaussian_components', 'compiled_native')},
                         {k:v for k,v in original.items() if k not in ('gaussian_components', 'compiled_native')})
        self.assertEqual([x['training_id'] for x in inventory[92::3]], prepare.TRAINING_IDS)
        self.assertEqual(inventory[-1]['latent_sigma'], .45)

    def test_center_covariance_or_fallback_changes_are_rejected(self):
        for mutate in [lambda g,d: d['new_components'][0]['mean'].__setitem__(0, 123.),
                       lambda g,d: d['new_components'][0]['covariance'][0].__setitem__(0, .5),
                       lambda g,d: g.__setitem__('minimum_conditional_mass', 1e-8),
                       lambda g,d: g['class_channels'][2].__setitem__('orthant', 23)]:
            original, diagnosis = guide_fixture()
            # Keep each saved row independent of the component object, as JSON does.
            diagnosis = json.loads(json.dumps(diagnosis)); mutate(original, diagnosis)
            with self.assertRaises(ValueError): prepare.build_guide(original, diagnosis, original['compiled_native'])

    def test_binary64_retention_uses_sequential_parser_normalization(self):
        original, diagnosis = guide_fixture()
        candidate, _ = prepare.build_guide(original, diagnosis, original['compiled_native'])
        result = prepare.binary64_retention(original, candidate)
        before = 0.; after = 0.
        for row in original['gaussian_components']: before += row['weight']
        for row in candidate['gaussian_components']: after += row['weight']
        ratios = [(new['weight']/after)/(old['weight']/before)
                  for old,new in zip(original['gaussian_components'], candidate['gaussian_components'])]
        self.assertEqual(result['original_component_retention_min'], min(ratios))
        self.assertEqual(result['original_component_retention_max'], max(ratios))
        self.assertIn('no IEEE', result['scope'])

    def test_seeds_and_unconditional_selection_are_frozen_and_collision_checked(self):
        first = prepare.population_design('/tmp/prospective', prepare.NAMESPACE, [])
        self.assertEqual(first, prepare.population_design('/tmp/prospective', prepare.NAMESPACE, []))
        self.assertEqual(len({p[key] for p in first for key in ('seed', 'audit_seed')}), 8)
        self.assertTrue(all(len(set(p['selected_ids'])) == 16 and min(p['selected_ids']) >= 0
                            and max(p['selected_ids']) < 128 for p in first))
        for key in ('seed', 'audit_seed'):
            with self.assertRaises(ValueError):
                prepare.population_design('/tmp/prospective', prepare.NAMESPACE, [first[0][key]])

    def test_materialization_has_seventeen_bounded_jobs_and_preserves_venv_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve(); context, inventory = context_fixture(base)
            out = base/'prepared'; out.mkdir()
            with patch.object(prepare, 'observer_setup_inventory', return_value=dict(contacts=37)):
                receipt = prepare.materialize(out, context, inventory)
            protocol = prepare.read(out/'protocol.json'); plan = prepare.read(out/'execution-plan.json')
            self.assertTrue(receipt['complete']); self.assertFalse(receipt['launched'])
            self.assertEqual(plan['maximum_workers'], 1); self.assertEqual(len(plan['jobs']), 17)
            self.assertEqual(sum(j['cpu_limit_seconds'] for j in plan['jobs']), 12300)
            self.assertEqual(sum(j['wall_limit_seconds'] for j in plan['jobs']), 24600)
            for i in range(4):
                group = plan['jobs'][4*i:4*i+4]
                self.assertEqual([j['phase'] for j in group], ['producer', 'algebra', 'labels', 'geometry'])
                self.assertNotIn('--probes', group[0]['argv']); self.assertNotIn('--cloud-replicates', group[0]['argv'])
                self.assertEqual(group[0]['argv'][group[0]['argv'].index('--samples')+1], '128')
                self.assertEqual(group[1]['argv'][0], protocol['python'])
            self.assertEqual(protocol['selected_geometry_rows'], 64)
            self.assertEqual(protocol['new_Poisson_clouds'], 0)
            self.assertFalse(protocol['launch_review_complete'])
            self.assertEqual(receipt['execution_plan'], prepare.bound(out/'execution-plan.json'))
            for path,digest in plan['files'].items(): self.assertEqual(prepare.sha(path), digest)

    def test_copy_divergence_stops_before_execution_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve(); context, inventory = context_fixture(base)
            out = base/'prepared'; out.mkdir()
            def bad_copy(source, destination): Path(destination).write_text('divergent')
            with patch.object(prepare.shutil, 'copy2', side_effect=bad_copy):
                with self.assertRaisesRegex(ValueError, 'Copied bytes'):
                    prepare.materialize(out, context, inventory)
            self.assertFalse((out/'execution-plan.json').exists())

    def test_preparation_failure_is_preserved_and_directory_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve(); context, inventory = context_fixture(base)
            context['inputs']['seed_inventory_roots'] = []
            out = base/'prepared'
            with patch.object(prepare, 'validate', return_value=context), \
                 patch.object(prepare, 'seed_inventory', return_value=inventory), \
                 patch.object(prepare, 'materialize', side_effect=InterruptedError('synthetic interrupt')):
                with self.assertRaises(InterruptedError): prepare.prepare(out, 'unused', 'unused')
                failure = prepare.read(out/'preparation-failure.json')
                self.assertFalse(failure['complete']); self.assertFalse(failure['launched'])
                self.assertTrue((out/'preparation-attempt.json').is_file())
                with self.assertRaisesRegex(ValueError, 'fresh directory'):
                    prepare.prepare(out, 'unused', 'unused')


if __name__ == '__main__': unittest.main()
