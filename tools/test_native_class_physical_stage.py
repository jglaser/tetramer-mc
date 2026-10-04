"""Synthetic metadata tests for deterministic post-production plan materialization.

No scientific producer or geometry observer executes. Existing independently
tested workers are replaced at their public boundary; these tests cover the
new handoff, input identities, immutable plans and complete inventory.
"""
from __future__ import annotations
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import native_class_physical_stage as stage


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stage.write(path, value)


def reference(path): return dict(path=str(Path(path).resolve()), sha256=stage.sha(path))


def fixture(base, *, source_schema=None):
    root = Path(base).resolve(); source_schema = source_schema or stage.streaming.full.SCHEMA
    (root/'analysis/class-r00').mkdir(parents=True); (root/'populations').mkdir()
    metadata = root/'common'; metadata.mkdir()
    for name in ('region', 'old', 'definition', 'strata', 'compiled', 'witness', 'guide'):
        save(metadata/(name+'.json'), dict(synthetic=name))
    pop = dict(id='r00', seed=120, audit_seed=121, directory=str(root/'populations/class-r00'))
    save(metadata/'preselection.json', stage.selected.preselection('r00', 32, pop['audit_seed']))
    pop['preselection'] = reference(metadata/'preselection.json')
    arm = dict(id='class', samples=32, alpha=.5, lambda_ratio=128., source_schema=source_schema,
               guide=reference(metadata/'guide.json'), populations=[pop])
    target = {k: reference(metadata/(n+'.json')) for k, n in
              [('region', 'region'), ('old_r5_region', 'old'), ('native_definition', 'definition')]}
    limits = dict(cpu_limit_seconds=120, wall_limit_seconds=240, address_space_limit_bytes=2**30,
                  max_record_bytes=2**20)
    source = stage.local_sources(stage.__file__)
    protocol = dict(schema=stage.PROTOCOL_SCHEMA, root=str(root), stage_materialization_policy=stage.POLICY,
        code_directory=str(Path(stage.__file__).resolve().parent), runtime=stage.streaming.runtime_identity(),
        source_sha256={n:stage.sha(p) for n,p in source.items()}, files={},
        arms=[arm], target=target, target_and_regions_sha256='a'*64,
        native_identity=dict(compiled_native=reference(metadata/'compiled.json'),
                             shape_compatibility=reference(metadata/'witness.json')),
        strata_file=reference(metadata/'strata.json'), observer_setup={'synthetic':True},
        phase_limits={name:dict(limits) for name in ('labels','selection','geometry','statistics')},
        strata=stage.statistics.STRATA, gates=stage.statistics.GATES, comparisons=[])
    save(root/'protocol.json', protocol)
    population = Path(pop['directory']); population.mkdir()
    (population/'samples.jsonl').write_text('synthetic stream, never parsed\n')
    (population/'attempts.jsonl').write_text('synthetic attempts, never parsed\n')
    manifest = dict(schema=source_schema, samples=32, seed=120, region_sha256=target['region']['sha256'],
        activity=.035, cloud_replicates=2, importance_uniform_probability=.5, lambda_ratio=128.,
        importance_guide_sha256=arm['guide']['sha256'])
    save(population/'manifest.json', manifest)
    save(population/'summary.json', dict(complete=True,manifest=manifest,samples=32,
        samples_sha256=stage.sha(population/'samples.jsonl'), attempts_sha256=stage.sha(population/'attempts.jsonl')))
    output=stage.paths(root,'class','r00')
    for name in ('algebra','labels'): save(output[name],dict(complete=True,passed=True,synthetic=name))
    jobs=[]
    for phase, terminal in [('producer',population/'summary.json'),('algebra',output['algebra']),
                            ('labels',output['labels']),('selection',output['selection_review']),('geometry',output['geometry'])]:
        jobs.append(dict(id='class-r00-'+phase,population='class-r00',phase=phase,
            argv=[str(Path(sys.executable).absolute()),'-B','-c','pass'],
            cpu_limit_seconds=120,wall_limit_seconds=240,address_space_limit_bytes=2**30,
            terminal=dict(path=str(terminal),success_contract='complete' if phase=='producer' else 'complete_and_passed')))
    execution=dict(schema='native-class-physical-execution-v1',root=str(root),maximum_workers=1,threads=1,
        files={str(root/'protocol.json'):stage.sha(root/'protocol.json'),
               pop['preselection']['path']:pop['preselection']['sha256'],
               str(Path(stage.controller.__file__).resolve()):stage.sha(stage.controller.__file__),
               str(Path(sys.executable).resolve()):stage.sha(sys.executable)},jobs=jobs,
        executable_resolutions={str(Path(sys.executable).absolute()):str(Path(sys.executable).resolve())})
    save(root/'execution-plan.json',execution)
    save(root/'execution/claim.json',dict(schema=stage.controller.SCHEMA,
        plan_sha256=stage.sha(root/'execution-plan.json'),maximum_workers=1,threads=1,retries=0,replacements=0))
    for ordinal in range(3): complete(root,execution,ordinal)
    return root,protocol,execution,arm,pop,output


def complete(root,execution,ordinal):
    job=execution['jobs'][ordinal]
    folder=root/'execution/jobs'/f'{ordinal:03d}-{job["id"]}'
    receipt=dict(id=job['id'],population=job['population'],phase=job['phase'],argv=job['argv'],
        pid=999,birth_ticks=111,returncode=0,child_started=True,child_drained=True,error=None,
        success=True,timeout=False,retries=0,replacements=0,success_contract=job['terminal']['success_contract'],
        terminal=reference(job['terminal']['path']))
    save(folder/'attempt.json',dict(job=job))
    save(folder/'process.json',{key:receipt[key] for key in ('id','pid','birth_ticks','argv')})
    save(folder/'exit.json',receipt); save(folder/'success.json',receipt)


class MaterializationTests(unittest.TestCase):
    def test_population_preserves_schema_allocation_and_all_attempt_hashes(self):
        for schema in (stage.streaming.full.SCHEMA,stage.streaming.hard_input.SCHEMA):
            with self.subTest(schema=schema),tempfile.TemporaryDirectory() as temp:
                root,p,e,arm,pop,out=fixture(temp,source_schema=schema)
                value=stage.population_binding(p,e,arm,pop,stage.Inputs())
                self.assertEqual(value['samples'],32); self.assertEqual(value['seed'],120)
                self.assertEqual(value['samples_sha256'],stage.sha(Path(pop['directory'])/'samples.jsonl'))
                self.assertEqual(stage.read(Path(pop['directory'])/'manifest.json')['schema'],schema)

    def test_changed_producer_allocation_and_terminal_binding_rejected(self):
        for field,value in [('samples',33),('seed',121),('alpha',.2),('lambda_ratio',64.)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                root,p,e,arm,pop,out=fixture(temp)
                (pop if field=='seed' else arm)[field]=value
                with self.assertRaises(ValueError): stage.population_binding(p,e,arm,pop,stage.Inputs())
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            (Path(pop['directory'])/'samples.jsonl').write_text('changed\n')
            with self.assertRaises(ValueError): stage.population_binding(p,e,arm,pop,stage.Inputs())

    def test_predecessor_requires_successful_drained_owned_terminal(self):
        for change in ('missing_success','changed_terminal','failed','different_argv'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                root,p,e,arm,pop,out=fixture(temp)
                folder=root/'execution/jobs/001-class-r00-algebra'
                if change=='missing_success': (folder/'success.json').unlink()
                elif change=='changed_terminal': out['algebra'].write_text('{}\n')
                else:
                    value=stage.read(folder/'exit.json')
                    if change=='failed': value.update(returncode=1,child_drained=False)
                    else: value['argv']=['/different']
                    for name in ('exit','success'):
                        (folder/(name+'.json')).unlink(); save(folder/(name+'.json'),value)
                with self.assertRaises((ValueError,FileNotFoundError)):
                    stage.predecessor(root,e,'class-r00-algebra',out['algebra'],stage.Inputs())

    def test_label_plan_uses_frozen_limits_native_identity_and_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            path=stage.materialize_labels(p,e,arm,pop,stage.Inputs()); value=stage.read(path)
            self.assertEqual(path,out['labels_plan'])
            self.assertEqual(value['compiled_native'],p['native_identity']['compiled_native'])
            self.assertEqual(value['limits']['max_atomic_queries'],32)
            self.assertEqual(value['limits']['max_native_queries'],32)
            self.assertEqual(value['source_sha256'],stage.source_map(p,stage.endpoints))
            with self.assertRaises(FileExistsError): stage.materialize_labels(p,e,arm,pop,stage.Inputs())

    def test_selection_keeps_exact_object_and_binds_deterministic_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            selection=dict(schema=stage.selected.SELECTION_SCHEMA,synthetic=True,rows=[])
            with mock.patch.object(stage.selected,'select',return_value=selection) as select:
                review=stage.materialize_selection(p,e,arm,pop,stage.Inputs())
            self.assertEqual(select.call_count,1); self.assertEqual(stage.read(out['selection']),selection)
            self.assertEqual(review['protocol'],reference(root/'protocol.json'))
            self.assertEqual(review['execution_plan'],reference(root/'execution-plan.json'))
            self.assertEqual(review['preselection'],pop['preselection'])
            self.assertTrue(review['predeclared_before_first_draw'])
            self.assertEqual(review['limits']['max_full_geometry_queries'],20)
            complete(root,e,3)
            path=stage.materialize_geometry(p,e,arm,pop,stage.Inputs())
            plan=stage.read(path)
            self.assertEqual(plan['selection'],reference(out['selection']))
            self.assertEqual(plan['review'],reference(out['selection_review']))
            self.assertEqual(plan['limits']['max_axis_queries'],60)

    def test_selection_rejects_unfrozen_ids_before_scan_and_prior_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp); e['files'].pop(pop['preselection']['path'])
            with mock.patch.object(stage.selected,'select',side_effect=AssertionError('must not scan')):
                with self.assertRaises(ValueError): stage.materialize_selection(p,e,arm,pop,stage.Inputs())
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp); save(out['geometry'],dict(prior=True))
            with mock.patch.object(stage.selected,'select',return_value={}):
                with self.assertRaises(ValueError): stage.materialize_selection(p,e,arm,pop,stage.Inputs())
            self.assertFalse(out['selection_review'].exists())

    def test_geometry_rejects_changed_selection_after_successful_predecessor(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            with mock.patch.object(stage.selected,'select',return_value={'synthetic':True}):
                stage.materialize_selection(p,e,arm,pop,stage.Inputs())
            complete(root,e,3); out['selection'].write_text('{}\n')
            with self.assertRaises(ValueError): stage.materialize_geometry(p,e,arm,pop,stage.Inputs())

    def test_statistics_waits_for_every_geometry_terminal_and_preserves_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            with self.assertRaises(FileNotFoundError): stage.materialize_statistics(p,e,stage.Inputs())
            save(out['geometry'],dict(complete=True,passed=True,synthetic=True)); complete(root,e,4)
            path=stage.materialize_statistics(p,e,stage.Inputs()); value=stage.read(path)
            self.assertEqual(value['comparisons'],p['comparisons'])
            self.assertEqual(value['arms'][0]['populations'][0]['seed'],pop['seed'])
            self.assertEqual(value['arms'][0]['populations'][0]['labels'],reference(out['labels']))
            self.assertEqual(value['gates'],p['gates'])

    def test_protocol_claim_runtime_source_and_controller_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            loaded,execution=stage.load_protocol(root/'protocol.json',stage.sha(root/'protocol.json'),stage.Inputs())
            self.assertEqual(loaded,p); self.assertEqual(execution,e)
            with mock.patch.object(stage.streaming,'runtime_identity',return_value={'changed':True}):
                with self.assertRaises(ValueError): stage.load_protocol(root/'protocol.json',stage.sha(root/'protocol.json'),stage.Inputs())
            claim=root/'execution/claim.json'; claim.unlink(); save(claim,dict(plan_sha256='0'*64))
            with self.assertRaises(ValueError): stage.load_protocol(root/'protocol.json',stage.sha(root/'protocol.json'),stage.Inputs())

    def test_dispatch_invokes_only_chosen_worker_and_all_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            with mock.patch.object(stage.endpoints,'run',return_value={'complete':True}) as worker, \
                    mock.patch.object(stage,'verify_live_authority'):
                stage.run(root/'protocol.json',protocol_sha256=stage.sha(root/'protocol.json'),population='class-r00',phase='labels')
            worker.assert_called_once_with(out['labels_plan'],plan_sha256=stage.sha(out['labels_plan']))
            with mock.patch.object(stage.statistics,'analyze',side_effect=AssertionError('no partial statistics')), \
                    mock.patch.object(stage,'verify_live_authority'):
                with self.assertRaises(ValueError):
                    stage.run(root/'protocol.json',protocol_sha256=stage.sha(root/'protocol.json'),population='class-r00',phase='statistics')

    def test_live_authority_checks_parent_birth_and_current_frozen_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            claim=stage.read(root/'execution/claim.json'); (root/'execution/claim.json').unlink()
            parent=os.getppid(); stat=Path(f'/proc/{parent}/stat').read_text().rsplit(')',1)[1].split()
            claim.update(pid=parent,birth_ticks=int(stat[19])); save(root/'execution/claim.json',claim)
            save(root/'execution/status.json',dict(plan_sha256=claim['plan_sha256'],failure=None,
                active=dict(ordinal=2,id='class-r00-labels',population='class-r00',phase='labels')))
            stage.verify_live_authority(p,e,'class-r00','labels')
            with self.assertRaises(ValueError): stage.verify_live_authority(p,e,'class-r00','geometry')
            claim['birth_ticks']+=1; (root/'execution/claim.json').unlink(); save(root/'execution/claim.json',claim)
            with self.assertRaises(ValueError): stage.verify_live_authority(p,e,'class-r00','labels')

    def test_admission_is_post_controller_and_binds_complete_population_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            with self.assertRaises(FileNotFoundError): stage.materialize_admission(p,e,stage.Inputs())
            save(root/'execution/summary.json',dict(complete=True,passed=True,active=None,unstarted=[],failure=None,
                plan_sha256=stage.sha(root/'execution-plan.json'),completed=[{} for _ in e['jobs']]))
            for path in [out['geometry_plan'],out['geometry'],stage.paths(root)['statistics_plan'],stage.paths(root)['statistics']]:
                save(path,dict(synthetic=True))
            plan_path=stage.materialize_admission(p,e,stage.Inputs()); value=stage.read(plan_path)
            self.assertEqual(len(value['populations']),1)
            self.assertEqual(value['populations'][0]['selected_receipt'],reference(out['geometry']))
            self.assertEqual(value['execution_status'],reference(root/'execution/summary.json'))

    def test_admission_refuses_circular_execution_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            root,p,e,arm,pop,out=fixture(temp)
            save(root/'execution/summary.json',dict(complete=True,passed=True,active=None,unstarted=[],failure=None,
                plan_sha256=stage.sha(root/'execution-plan.json'),completed=[{} for _ in e['jobs']]))
            e['jobs'][-1]['phase']='admission'
            with self.assertRaises(ValueError): stage.materialize_admission(p,e,stage.Inputs())


if __name__=='__main__': unittest.main()
