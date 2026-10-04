"""Synthetic metadata only: no Rust, classifiers, trajectories or sampling."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_partner_atlas_benchmark as p
import test_prepare_flexible_surrogate_benchmark as fixtures

save, opaque, replace, python_receipt = fixtures.save, fixtures.opaque, fixtures.replace, fixtures.python_receipt


def build_fixture(root):
    with patch.object(fixtures.p, 'RUST_TESTS', p.RUST_TESTS), patch.object(fixtures.p, 'REQUIRED_COMMANDS', p.REQUIRED_COMMANDS):
        project, ref, bundle = fixtures.build_fixture(root)
    value = p.read(ref['path']); value['schema'] = 'partner-atlas-benchmark-validation-v1'
    for k in ('source_before', 'source_after'):
        value[k][str(project/'build.rs')] = p.sha(project/'build.rs')
    return project, replace(ref['path'], value), bundle


def reference_fixture(root, project, build, bundle, change=None):
    directory = root/'reference-run'; output = directory/'reference'
    sources = {str(Path(n).relative_to(project)):h for n,h in p.read(build['path'])['source_after'].items()}
    protocol = save(output/'protocol.json', dict(schema='partner-atlas-cached-reference-protocol-v1',
        streams=4, sources_per_stream=2048, total_outer_calls=32768, total_candidate_attempts=147456,
        new_source_draws=0, arms=[[1,1.,True,'direct'],[1,1.,False,'m1_guided'],[8,1.,False,'m8_guided'],[8,0.,False,'flat8']],
        reference_source_sha256=sources[p.RUST_TESTS[-1]], compiled_library_bundle_sha256=bundle['sha256']))
    summary = save(output/'summary.json', dict(schema='partner-atlas-cached-reference-summary-v1',
        complete=True, passed=True, statistical_checks_passed=True, completed_sources=8192,
        checks=[dict(passed=True) for _ in range(292)]))
    hashes = {'protocol.json':protocol['sha256'], 'summary.json':summary['sha256'], 'kernel-attempts.jsonl':'a'*64}
    receipt = dict(schema='partner-atlas-cached-reference-receipt-v1', complete=True, passed=change != 'scientific',
        numerical_allocation_complete=True, statistical_checks_passed=True, error=None,
        retries=0, replacement_draws=0, new_source_draws=0, output_sha256=hashes.copy())
    scientific = save(output/'receipt.json', receipt); hashes['receipt.json'] = scientific['sha256']
    validation = dict(schema='partner-atlas-reference-validation-v1', complete=True, passed=True,
        returncode=0, child_started=True, child_drained=True, error=None, source_before=copy.deepcopy(sources),
        source_after=copy.deepcopy(sources), production_before='0'*64, production_after='0'*64,
        numerical_allocation_complete=True, scientific_receipt_passed=True, receipt_bindings_valid=True,
        outputs={'reference/'+n:h for n,h in hashes.items()})
    if change in ('passed','child_started','child_drained'): validation[change] = False
    if change == 'returncode': validation['returncode'] = 1
    if change == 'source':
        validation['source_before']['src/lib.rs'] = validation['source_after']['src/lib.rs'] = 'c'*64
    vr = save(directory/'validation.json', validation)
    audit = dict(schema='partner-atlas-reference-journal-audit-v1', complete=True, passed=True,
        reference_statistical_checks_passed=True, validation_sha256=vr['sha256'],
        inventory=dict.fromkeys(['direct','m1_guided','m8_guided','flat8'],8192), candidate_attempts=147456,
        new_physical_samples=0, input_sha256={vr['path']:vr['sha256'], **{str(output/n):h for n,h in hashes.items()}})
    if change == 'inventory': audit['inventory']['direct'] -= 1
    if change == 'audit_join': audit['input_sha256'][str(output/'summary.json')] = 'b'*64
    ar = save(root/'reference-audit/audit.json', audit)
    av = dict(schema='partner-atlas-reference-audit-execution-v1', complete=True, passed=True,
        child_started=True, child_drained=True, returncode=0, error=None,
        source_before={'auditor.py':'d'*64}, source_after={'auditor.py':'d'*64},
        audit_sha256=ar['sha256'], new_physical_samples=0)
    if change == 'audit_drain': av['child_drained'] = False
    if change == 'audit_source': av['source_after'] = {}
    if change == 'audit_hash': av['audit_sha256'] = 'e'*64
    avr = save(root/'reference-audit/validation.json', av)
    return vr, ar, avr, sources


def native_fixture(root, original):
    directory = root/'native'; archive = directory/'definition'
    dependencies = {f'input-{i}.json':opaque(archive/'inputs'/f'input-{i}.json') for i in range(14)}
    dependencies['source/native_contact_regions.py'] = opaque(archive/'inputs/source/native_contact_regions.py', '# never imported\n')
    hashes = {k:v['sha256'] for k,v in dependencies.items()}
    config = p.read(original/'config.json')
    definition = save(archive/'definition.json', dict(shape_sha256=config['shape']['sha256'],
        input_sha256=hashes, criteria={'frozen':'synthetic'}))
    compiled = save(directory/'compiled-native.json', dict(source_definition_sha256=definition['sha256'],
        source_input_sha256=hashes, criteria={'frozen':'synthetic'}))
    witness = save(directory/'witness.json', dict(native_definition_sha256=definition['sha256'],
        shape_sha256=config['shape']['sha256'], compiled_native_sha256=compiled['sha256'],
        native_shape_compatibility=dict(compatible=True, hard_valid_implication_within_tolerance=True,
            native_atoms=4004, physical_atoms=4004, matched_atoms=4004,
            physical_index_by_native_atom=list(range(4004)), matched_max_center_error_a=0.,
            matched_max_radius_error_a=0., pair_overlap_slack_bound_a=0.)))
    refs = dict(definition=definition, compiled_native=compiled, witness=witness,
                shape=config['shape'], source_frame=config['source_frame'])
    plan = save(directory/'initial/audit-plan.json', dict(schema='native-pair-initial-audit-plan-v1', **refs))
    report = dict(complete=True, passed=True,
        input_sha256={v['path']:v['sha256'] for v in [plan,*refs.values(),*dependencies.values()]},
        classifier=dict(definition=definition['path'], definition_sha256=definition['sha256'],
            input_sha256=hashes, criteria={'frozen':'synthetic'}, runtime_sha256=hashes['source/native_contact_regions.py']))
    summary = save(directory/'analysis/summary.json',report)
    return directory, summary, report, {summary['path']:summary['sha256']}


def arguments(destination, campaign, analysis, build, python, refs, native, bundle):
    vr, ar, avr, _ = refs
    return (destination,campaign,analysis,build['path'],python['path'],vr['path'],vr['sha256'],
            ar['path'],ar['sha256'],avr['path'],avr['sha256'],native[0],native[1]['sha256'],bundle['path'])


class PreparationTests(unittest.TestCase):
    def test_allocation_and_all_forty_cached_identities_without_outcome_filtering(self):
        with tempfile.TemporaryDirectory() as temp:
            original,campaign,analysis = fixtures.inherited_fixture(Path(temp))
            inputs = p.Inputs(); config,_,controls,_ = p.inherited(inputs,campaign,analysis)
            self.assertEqual(config,p.read(original/'config.json'))
            self.assertEqual(len(controls['observations']),40)
            self.assertEqual({p.identity(x['job']) for x in controls['observations']},
                {(0,a,i,s) for a in p.CONTROL_ARMS for i in p.old.INITIALIZATIONS for s in range(4)})
            self.assertFalse(any(Path(n).name == 'trajectory.jsonl' for n in inputs.files))
            a=p.allocation();self.assertEqual(len(p.jobs()),32)
            self.assertEqual((a['chains'],a['combined_analysis_chains']),(32,72))
            self.assertEqual((a['local_attempts'],a['dimer_attempts'],a['maximum_inner_steps']),(589824,147456,663552))
            self.assertEqual(a['maximum_journal_rows'],32*(1+4608*7))
            self.assertEqual(a['retained_initial_observations'],147488)
            self.assertEqual((a['new_cloud_banks'],a['new_preparation_attempts']),(0,0))
            self.assertEqual(a['primary_control_arm'],'local')

    def test_incomplete_or_changed_cached_control_authority_rejected(self):
        for change in ('incomplete','cache','cloud'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                original,campaign,analysis=fixtures.inherited_fixture(Path(temp))
                if change=='incomplete':
                    for name in ('status','summary'):
                        path=analysis/f'execution/{name}.json';v=p.read(path);v['complete']=False;replace(path,v)
                else:
                    path=(next((analysis/'analysis').glob('*observations.jsonl')) if change=='cache'
                          else original/'prepared/cloud-0-source-0.bin')
                    path.write_text('changed')
                with self.assertRaises(ValueError):p.inherited(p.Inputs(),campaign,analysis)

    def test_build_commands_sources_bundle_and_protected_production_fail_closed(self):
        for change in (None,'source','production','drain','command','example','bundle','executable'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                project,ref,bundle=build_fixture(Path(temp));v=p.read(ref['path'])
                if change in ('source','production','drain','command'):
                    if change=='source':v['source_after'].pop(str(project/'build.rs'))
                    elif change=='production':v['production_after']='1'*64
                    elif change=='drain':v['runs'][0]['child_drained']=False
                    else:v['runs'][0]['argv']=['true']
                    replace(ref['path'],v)
                elif change=='example':(project/p.old.EXAMPLE).write_text('// changed')
                elif change=='bundle':Path(bundle['path']).write_text('{}')
                elif change=='executable':Path(v['executable']['path']).write_bytes(b'changed')
                with patch.object(p,'ROOT',project):
                    if change is None:
                        _,witness,compiled=p.validate_build(p.Inputs(),ref['path'],bundle['path'])
                        self.assertEqual(witness['new_arms'],p.ARMS);self.assertIn('build.rs',compiled)
                    else:
                        with self.assertRaises(ValueError):p.validate_build(p.Inputs(),ref['path'],bundle['path'])

    def test_reference_admission_requires_both_drained_receipts_and_full_source_match(self):
        for change in (None,'pin','passed','returncode','child_started','child_drained','inventory','scientific',
                       'source','audit_drain','audit_source','audit_hash','audit_join','bundle'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);project,build,bundle=build_fixture(root)
                vr,ar,avr,sources=reference_fixture(root,project,build,bundle,change)
                args=(p.Inputs(),vr['path'],'f'*64 if change=='pin' else vr['sha256'],ar['path'],ar['sha256'],
                      avr['path'],avr['sha256'],sources,'f'*64 if change=='bundle' else bundle['sha256'])
                if change is None:
                    result=p.validate_reference(*args)
                    self.assertFalse(result['journal_rescan']);self.assertFalse(result['reference_rerun'])
                    self.assertFalse(any(n.endswith('kernel-attempts.jsonl') for n in args[0].files))
                else:
                    with self.assertRaises(ValueError):p.validate_reference(*args)

    def test_native_authority_exact_definition_witness_and_shape_without_geometry(self):
        for change in (None,'summary','shape','witness','classifier','dependency','lifecycle'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);original,_,_=fixtures.inherited_fixture(root)
                directory,summary,report,lifecycle=native_fixture(root,original)
                config=p.read(original/'config.json')
                if change=='shape':config['shape']=dict(config['shape'],sha256='f'*64)
                if change=='classifier':report['classifier']['runtime_sha256']='a'*64
                if change=='witness':(directory/'witness.json').write_text('{}')
                if change=='dependency':(directory/'definition/inputs/input-0.json').write_text('changed')
                with patch.object(p.native_completed,'authenticate',return_value=(report,lifecycle),
                                  side_effect=ValueError('incomplete lifecycle') if change=='lifecycle' else None):
                    args=(p.Inputs(),directory,'b'*64 if change=='summary' else summary['sha256'],config)
                    if change is None:
                        value=p.validate_native_authority(*args)
                        self.assertTrue(value['definition_frozen']);self.assertTrue(value['no_geometry_queries'])
                        self.assertEqual(p.native_plan(value)['queries_per_retained_endpoint'],1)
                    else:
                        with self.assertRaises(ValueError):p.validate_native_authority(*args)

    def test_fresh_freeze_complete_execution_closure_without_inherited_mutation(self):
        sources=p.source_paths()
        for name in [p.SELF,p.TEST,'tools/analyze_partner_atlas_benchmark.py',
                     'tests/test_analyze_partner_atlas_benchmark.py','tools/plot_surrogate_internal_native.py']:
            self.assertIn(name,sources)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original,campaign,analysis=fixtures.inherited_fixture(root)
            project,build,bundle=build_fixture(root);refs=reference_fixture(root,project,build,bundle)
            native=native_fixture(root,original);python=python_receipt(root,sources);destination=root/'new'
            before={str(f):p.sha(f) for f in original.rglob('*') if f.is_file()}
            args=arguments(destination,campaign,analysis,build,python,refs,native,bundle)
            with patch.object(p,'ROOT',project),patch.object(p,'source_paths',return_value=sources),\
                    patch.object(p.native_completed,'authenticate',return_value=native[2:]):
                receipt=p.prepare(*args);config=p.verify(destination);plan=p.read(destination/'execution-plan.json')
                self.assertTrue(receipt['complete']);self.assertFalse(receipt['launched'])
                self.assertEqual(len(plan['jobs']),32);self.assertEqual((plan['maximum_workers'],plan['threads']),(1,1))
                self.assertEqual({j['terminal']['success_contract'] for j in plan['jobs']},{'complete'})
                for job in plan['jobs']:
                    self.assertEqual({k:job[k] for k in p.LIMITS},p.LIMITS)
                    self.assertEqual(job['phase'],'producer')
                for key in ('physical','local','factorized','cloud','contexts','source_frame','source_config','master_seed','preparation_output'):
                    self.assertEqual(config[key],p.read(original/'config.json')[key])
                self.assertEqual(config['partner_atlas_policy'],p.policy())
                self.assertEqual(len(config['control_analysis']['observations']),40)
                self.assertFalse(p.read(destination/'native-observer-plan.json')['dispatched'])
                self.assertFalse((destination/'execution').exists());self.assertFalse((destination/'chains').exists())
                self.assertFalse(any(Path(n).name in ('trajectory.jsonl','kernel-attempts.jsonl') for n in plan['files']))
                self.assertEqual(before,{str(f):p.sha(f) for f in original.rglob('*') if f.is_file()})
                with self.assertRaises(ValueError):p.prepare(*args)
                (destination/'common/source/src/flexible_surrogate_chain.rs').write_text('// changed')
                with self.assertRaises(ValueError):p.verify(destination)

    def test_midcopy_source_mutation_preserves_failed_destination_and_prevents_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original,campaign,analysis=fixtures.inherited_fixture(root)
            project,build,bundle=build_fixture(root);refs=reference_fixture(root,project,build,bundle)
            native=native_fixture(root,original)
            probe=root/'probe.py';probe.write_text('TESTED = True\n')
            sources={'tools/probe.py':probe,p.DRIVER:p.ROOT/p.DRIVER}
            python=python_receipt(root,sources);destination=root/'failed'
            copyfile=p.shutil.copyfile
            def changing_copy(source,target):
                if Path(source)==probe:probe.write_text('TESTED = False\n')
                return copyfile(source,target)
            args=arguments(destination,campaign,analysis,build,python,refs,native,bundle)
            with patch.object(p,'ROOT',project),patch.object(p,'source_paths',return_value=sources),\
                    patch.object(p.native_completed,'authenticate',return_value=native[2:]),\
                    patch.object(p.shutil,'copyfile',side_effect=changing_copy):
                with self.assertRaisesRegex(ValueError,'Tested source changed'):p.prepare(*args)
                self.assertFalse(p.read(destination/'preparation-failure.json')['complete'])
                self.assertFalse((destination/'preparation.json').exists())
                with self.assertRaises(ValueError):p.prepare(*args)


if __name__ == '__main__':
    unittest.main()
