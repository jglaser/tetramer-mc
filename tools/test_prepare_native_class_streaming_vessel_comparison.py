"""Pure scheduling/input-contract controls; never invoke a scientific process."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import prepare_native_class_streaming_vessel_comparison as prep


def plan_for(root):
    hashes = {name: digest for name, (_, digest) in prep.SOURCES.items() if name != 'guide.json'}
    hashes.update({'compiled-native.json': prep.COMPILED_SHA, 'shape-compatibility.json': prep.WITNESS_SHA,
                   'source-guide.json': prep.GUIDE[1], 'native-region/definition.json': prep.NATIVE[1]})
    return dict(schema=prep.SCHEMA, preparation_only=True, physical_jobs_launched=0, dispatch_ready=False,
        jobs=prep.jobs_for(root), total_unconditional_draws=prep.TOTAL,
        stages=[dict(name=s, draws_per_population=n, independent_populations_per_arm=4) for s, n in prep.STAGES],
        arms=list(prep.ARMS), physical=prep.physical(), proposal_contracts=prep.proposal_contracts(),
        maximum_physical_workers=8, maximum_audit_workers=4, maximum_all_workers=32,
        thread_environment=prep.THREADS, native_definition_sha256=prep.NATIVE[1], input_sha256=hashes)


class PreparationTests(unittest.TestCase):
    def test_declared_job_outputs_are_allowed_but_unfrozen_input_and_other_outputs_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'common').mkdir(); (root/'inputs').mkdir(); (root/'plan.json').write_text('{}')
            (root/'inputs/shape.json').write_text('{}'); (root/'common/code.py').write_text('# frozen')
            frozen = {str(p.relative_to(root)):prep.sha(p) for p in root.rglob('*') if p.is_file()}
            jobs = prep.jobs_for(root); prep.validate_inventory(root,frozen,jobs)
            target = Path(jobs[0]['directory']); target.mkdir(parents=True); (target/'summary.json').write_text('{}')
            prep.validate_inventory(root,frozen,jobs)
            (root/'inputs/unfrozen.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'immutable'): prep.validate_inventory(root,frozen,jobs)
            (root/'inputs/unfrozen.json').unlink(); (root/'unprepared-output.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Undeclared'): prep.validate_inventory(root,frozen,jobs)

    def test_exact_fresh_allocations_seeds_and_explicit_wall_proposal(self):
        root = Path('/tmp/native-class-inert-test-only')
        plan = plan_for(root); prep.validate_contract(plan, root)
        self.assertEqual(len(plan['jobs']), 16)
        self.assertEqual(sum(j['samples'] for j in plan['jobs']), 2621440)
        self.assertEqual(len({j['seed'] for j in plan['jobs']}), 16)
        for job in plan['jobs']:
            self.assertIn('--wall-uniform-envelope', job['command'])
            self.assertNotIn('--synthetic', job['audit_command'])
            self.assertEqual(job['samples'], dict(prep.STAGES)[job['stage']])
            self.assertEqual('--definition' in job['audit_command'], job['arm']=='half_mixture')
            self.assertEqual('--latent-guide' in job['command'], job['arm']=='half_mixture')
            self.assertIn('--native-definition', job['partition_command'])
        self.assertFalse(plan['dispatch_ready'])

    def test_contract_rejects_allocation_physics_proposal_native_and_gate_changes(self):
        root = Path('/tmp/native-class-inert-test-only'); base = plan_for(root)
        for field, value in [('physical_jobs_launched', 1), ('dispatch_ready', True),
                             ('maximum_physical_workers', 9), ('schema', 'renamed-legacy')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                prep.validate_contract(dict(base, **{field:value}), root)
        for defect in ('seed','draws','activity','floor','components','native','shape','region'):
            changed = copy.deepcopy(base)
            if defect == 'seed': changed['jobs'][1]['seed'] = changed['jobs'][0]['seed']
            elif defect == 'draws': changed['jobs'][0]['samples'] -= 1
            elif defect == 'activity': changed['physical']['activity'] = .04
            elif defect == 'floor': changed['proposal_contracts']['half_mixture']['latent_defensive_uniform_probability'] = .2
            elif defect == 'components': changed['proposal_contracts']['half_mixture']['latent_gaussian_component_count'] = 92
            elif defect == 'native': changed['native_definition_sha256'] = '0'*64
            elif defect == 'shape': changed['input_sha256']['shape.json'] = '0'*64
            else: changed['input_sha256']['current_R4.json'] = '0'*64
            with self.subTest(defect=defect), self.assertRaises(ValueError): prep.validate_contract(changed, root)

    def test_guide_only_relocates_compiled_path_without_refitting(self):
        source = dict(schema='defensive-native-class-line-guide-v1', gaussian_components=[{'weight':1/116} for _ in range(116)],
            class_channels=prep.CHANNELS, raw_translation_axes=[0,1,2], conditional_probability=1.,
            defensive_uniform_shell_probability=.5, compiled_native=dict(path='/old/compiled.json', sha256=prep.COMPILED_SHA),
            region_sha256=prep.SOURCES['current_R4.json'][1])
        candidate = copy.deepcopy(source); candidate['compiled_native']['path'] = '/tmp/new/compiled.json'
        prep.validate_guide(candidate, source, '/tmp/new/compiled.json')
        for defect in ('weight','channel','path','count','floor'):
            changed = copy.deepcopy(candidate)
            if defect == 'weight': changed['gaussian_components'][0]['weight'] = .9
            elif defect == 'channel': changed['class_channels'][0]['class'] = 'native'
            elif defect == 'path': changed['compiled_native']['path'] = '/other/compiled.json'
            elif defect == 'count': changed['gaussian_components'].pop()
            else: changed['defensive_uniform_shell_probability'] = .2
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                prep.validate_guide(changed, source, '/tmp/new/compiled.json')

    def test_embedded_source_identity_and_escape_paths_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); text = 'saved source, never executed\n'
            value = dict(schema=1, files={'src/main.rs':dict(text=text,sha256=hashlib.sha256(text.encode()).hexdigest())})
            bundle = root/'bundle'; binary = root/'binary'; prep.write(bundle,value)
            binary.write_bytes(b'inert prefix'+bundle.read_bytes())
            self.assertEqual(prep.verify_embedded(binary,bundle),value)
            for defect in ('not_embedded','text','escape'):
                changed = copy.deepcopy(value)
                if defect == 'text': changed['files']['src/main.rs']['text'] += 'tamper'
                elif defect == 'escape': changed['files']['../escape'] = changed['files'].pop('src/main.rs')
                prep.write(bundle,changed)
                binary.write_bytes(b'inert' if defect=='not_embedded' else bundle.read_bytes())
                with self.subTest(defect=defect), self.assertRaises(ValueError): prep.verify_embedded(binary,bundle)


if __name__ == '__main__': unittest.main()
