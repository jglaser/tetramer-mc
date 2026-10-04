"""Synthetic completed metadata only; no native, atom or proposal evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import prepare_singleton_fusion_diagnostic as prepare


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return prepare.reference(path)


def fixture(root):
    source = [dict(position=[float(i), 0., 0.], orientation=[1., 0., 0., 0.]) for i in range(264)]
    shape = save(root/'shape.json', dict(atoms=[dict(center=[0., 0., 0.], radius=.1)]))
    model = save(root/'atlas.json', dict(shape_sha256=shape['sha256']))
    config = dict(shape=shape['path'], boundary=dict(kind='spherical', radius=600.), box_lengths=[1200.]*3,
        initial_poses=source, coordinate_frame_convention='Stored poses use sphere-centered coordinates; toy fixture',
        depletant_radius=1.4, reservoir_density=.0275, seed=1)
    config_ref = save(root/'source-config.json', config)
    frame = dict(poses=source, boundary='spherical', spherical_wall_radius=600., coordinate_wall_center=[100., 200., 300.])
    frame_ref = save(root/'source-frame.json', frame)
    reference = dict(source_config=config_ref, source_frame=frame_ref, shape=shape)
    original = dict(schema='evolving-dimer-benchmark-v1', contexts=[dict(prepare.CONTEXT)]+[{}]*3,
        physical=dict(wall_radius=600., activity=.0275, depletant_radius=1.4), source_config=config_ref,
        source_frame=frame_ref, reference_config=save(root/'reference.json', reference), shape=shape, atlas=model,
        source_freeze_manifest=save(root/'freeze.json', dict(frame_sha256=frame_ref['sha256'])))
    original_ref = save(root/'original.json', original)
    starts = []
    for stream in range(4):
        selected = copy.deepcopy([source[27], source[132]])
        selected[0]['position'][1] = stream+.1; selected[1]['position'][2] = stream+.2
        value = dict(context_index=0, stream=stream, status='prepared', is_equilibrium_sample=False,
            source=[source[27], source[132]], selected=selected)
        starts.append(dict(context_index=0, stream=stream, members=[27,132], record=save(root/f'start-0-{stream}.json',value), value=value))
    initial = dict(states=[dict(state_id='source', reference_native_keys=[])]+[
        dict(state_id=f'context-{i}-stream-{j}', reference_native_keys=[]) for i in range(4) for j in range(4)])
    return dict(original=original, original_ref=original_ref, config=config, frame=frame, reference=reference,
                starts=starts, initial=initial)


def build_fixture(root):
    source = root/'repository/src/bin/singleton-fusion-diagnostic.rs'
    source.parent.mkdir(parents=True); source.write_text('// synthetic compiled source\n')
    binary = root/'singleton-fusion-diagnostic'; binary.write_bytes(b'synthetic binary, never execute'); binary.chmod(0o755)
    bundle = save(root/'bundle.json', dict(schema=1, files={'src/bin/singleton-fusion-diagnostic.rs':
        dict(text=source.read_text(), sha256=prepare.sha(source))}))
    validation = save(root/'build-validation.json', dict(complete=True, passed=True))
    witness = dict(schema='singleton-fusion-build-witness-v1', complete=True, passed=True,
        executable=prepare.reference(binary), source_bundle=bundle, cli_source=prepare.reference(source),
        validation=validation, production_executable_unchanged=True)
    return witness, save(root/'build-witness.json', witness)


class PreparationTests(unittest.TestCase):
    def test_exact_four_substitutions_keep_spectators_and_sphere_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = fixture(Path(tmp))
            states = prepare.initial_states(f['original'], f['reference'], f['config'], f['frame'], f['initial'], f['starts'])
            self.assertEqual([s['id'] for s in states], ['source']+[f'prepared-{i}' for i in range(4)])
            self.assertEqual(states[0]['poses'], f['frame']['poses'])
            for stream, state in enumerate(states[1:]):
                self.assertEqual([state['poses'][i] for i in (27,132)], f['starts'][stream]['value']['selected'])
                self.assertTrue(all(state['poses'][i] == f['frame']['poses'][i] for i in range(264) if i not in (27,132)))
            self.assertEqual(states[0]['poses'][0]['position'], [0.,0.,0.])  # nonzero lab wall center is not subtracted

    def test_context_native_evidence_inventory_and_source_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for field in ('context', 'source', 'native', 'start_count', 'prepared_source', 'quaternion'):
                f = fixture(Path(tmp)/field)
                if field == 'context': f['original']['contexts'][0]['root'] = 9
                elif field == 'source': f['config']['initial_poses'] = list(reversed(f['config']['initial_poses']))
                elif field == 'native': f['initial']['states'][0]['reference_native_keys'] = [[27,132,2]]
                elif field == 'start_count': f['starts'].pop()
                elif field == 'prepared_source': f['starts'][0]['value']['source'] = []
                else: f['starts'][0]['value']['selected'][0]['orientation'] = [2.,0.,0.,0.]
                with self.subTest(field=field), self.assertRaises(ValueError):
                    prepare.initial_states(f['original'], f['reference'], f['config'], f['frame'], f['initial'], f['starts'])

    def test_build_witness_binds_compiled_source_and_production_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); witness, ref = build_fixture(root)
            bindings = prepare.Bindings()
            self.assertEqual(prepare.bind_build(bindings, ref), witness)
            self.assertIn(witness['cli_source']['path'], bindings.files)
            Path(witness['cli_source']['path']).write_text('// changed\n')
            with self.assertRaisesRegex(ValueError, 'Changed bound input'): prepare.bind_build(prepare.Bindings(), ref)
            witness['production_executable_unchanged'] = False
            ref = save(root/'witness-invalid.json', witness)
            with self.assertRaisesRegex(ValueError, 'isolated build witness'): prepare.bind_build(prepare.Bindings(), ref)

    def test_preparer_validation_rejects_stale_missing_or_mutated_test_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = root/'prepare.py'; a.write_text('a'); b = root/'test.py'; b.write_text('b')
            sources = {str(p):prepare.sha(p) for p in (a,b)}
            value = dict(complete=True, passed=True, source_before=sources, source_after=dict(sources))
            ref = save(root/'validation.json', value)
            prepare.validate_tests(prepare.Bindings(), ref, [a,b])
            a.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Changed bound input'): prepare.validate_tests(prepare.Bindings(), ref, [a,b])
            value['source_before'] = {}; value['source_after'] = {}
            ref = save(root/'missing.json', value)
            with self.assertRaisesRegex(ValueError, 'omitted source'): prepare.validate_tests(prepare.Bindings(), ref, [a,b])

    def context(self, root):
        f = fixture(root/'inputs'); witness, witness_ref = build_fixture(root/'build')
        b = prepare.Bindings()
        for path in root.rglob('*'):
            if path.is_file(): b.bind(path)
        states = prepare.initial_states(f['original'], f['reference'], f['config'], f['frame'], f['initial'], f['starts'])
        sources = {Path(prepare.__file__).name:Path(prepare.__file__).resolve(),
                   Path(prepare.driver.__file__).name:Path(prepare.driver.__file__).resolve()}
        for path in sources.values(): b.bind(path)
        initial_summary = save(root/'initial-summary.json', dict(complete=True,passed=True)); b.load(initial_summary)
        return dict(bindings=b, original=f['original'], original_ref=f['original_ref'], source_config=f['config'], states=states,
            initial=dict(bindings=dict(summary=initial_summary)),
            witness=witness, witness_ref=witness_ref, validation=witness['validation'], sources=sources,
            python=str(Path(sys.executable).absolute()))

    def test_materialized_single_job_frozen_manifest_and_no_output_before_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); context = self.context(root); out = root/'prepared'
            with mock.patch.object(prepare, 'admit', side_effect=ValueError('metadata rejected')):
                with self.assertRaisesRegex(ValueError, 'metadata rejected'): prepare.prepare(None,None,None,None,out=out)
            self.assertFalse(out.exists())
            with mock.patch.object(prepare, 'admit', return_value=context): result = prepare.prepare(None,None,None,None,out=out)
            self.assertTrue(result['complete']); self.assertFalse(result['launched']); self.assertFalse((out/'diagnostic').exists())
            manifest = prepare.read(out/'manifest.json'); execution = prepare.read(out/'execution-plan.json')
            self.assertEqual(manifest['caps'],prepare.CAPS); self.assertEqual(len(manifest['constructions']),10)
            self.assertEqual(manifest['constructions'][0]['neighbors'],[132,228]); self.assertEqual(manifest['constructions'][1]['neighbors'],[27,228])
            self.assertEqual(len(execution['jobs']),1); self.assertEqual(execution['jobs'][0]['cpu_limit_seconds'],600)
            self.assertEqual(execution['jobs'][0]['address_space_limit_bytes'],8*2**30)
            self.assertEqual(prepare.driver.verify_plan(out/'execution-plan.json',execution,prepare.sha(out/'execution-plan.json'),fresh=True),out)
            actual_config = prepare.read(manifest['config']['path']); expected = copy.deepcopy(context['source_config'])
            expected['shape'] = manifest['shape']['path']; self.assertEqual(actual_config,expected)
            for item in context['states'][1:]: self.assertIn(item['provenance']['path'],execution['files'])
            with self.assertRaisesRegex(ValueError, 'Fresh preparation'): prepare.prepare(None,None,None,None,out=out)

    def test_partial_metadata_failure_is_retained_and_not_dispatchable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); context = self.context(root); out = root/'partial'
            with mock.patch.object(prepare, 'admit', return_value=context), mock.patch.object(prepare, 'materialize', side_effect=ValueError('copy failed')):
                with self.assertRaisesRegex(ValueError, 'copy failed'): prepare.prepare(None,None,None,None,out=out)
            self.assertTrue((out/'preparation-attempt.json').exists()); self.assertFalse((out/'preparation.json').exists())
            failure = prepare.read(out/'preparation-failure.json'); self.assertFalse(failure['complete']); self.assertEqual(failure['retries'],0)


if __name__ == '__main__': unittest.main()
