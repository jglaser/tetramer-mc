#!/usr/bin/env python3
"""Synthetic metadata only; opaque files stand in for scientific inputs."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_two_root_dimer_benchmark as p


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    p.write(path,value);return p.record(path)


def opaque(path,text='not parsed as scientific data'):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text);return p.record(path)


def fixture(root):
    original=root/'original';analysis=root/'control';original.mkdir();(analysis/'analysis').mkdir(parents=True)
    physical=opaque(original/'shape.json')
    contexts=[dict(root=a,child=b,anchor=228+c) for c,(a,b) in enumerate(p.old.EXPECTED_PAIRS)]
    config=dict(schema='evolving-dimer-benchmark-v1',jobs=p.old.jobs(),allocation=p.old.allocation(),
        master_seed=p.old.MASTER,contexts=contexts,shape=physical,source_frame=physical,source_config=physical,
        physical=dict(depletant_radius=1.4,activity=.0275,lambda_ratio=64.,wall_radius=593.742500239952),
        limits=dict(cpu_seconds=1800.,raw_per_leg=20000000),local=dict(member_order=[0,1,0,1]),
        factorized=dict(root_cap=32,internal_cap=32,joint_cap=1,order='root_first'),cloud=dict(raw_count=16384),
        compiled_source_sha256={'src/old.rs':'f'*64},output=str(original/'execution'),preparation_output=str(original/'prepared'))
    config_ref=save(original/'config.json',config)
    source=opaque(original/'common/source/tools/historical_observer.py','historical = True\n')
    protocol_ref=save(original/'protocol.json',{'scope':'original frozen synthetic protocol'})
    freeze_ref=save(original/'freeze.json',dict(schema='evolving-dimer-preparation-freeze-v1',complete=True,
        scientific_execution_started=False,files={'config.json':config_ref['sha256'],
        'protocol.json':protocol_ref['sha256'],'common/source/tools/historical_observer.py':source['sha256']},
        original_input_bindings={'shape':physical}))
    binding_ref=save(original/'binding.json',dict(config_sha256=config_ref['sha256'],freeze_sha256=freeze_ref['sha256'],
        protocol_sha256=protocol_ref['sha256']))
    files=[];banks=[];starts=[]
    for c in range(4):
        for initialization in p.old.INITIALIZATIONS:
            for s in range(4):
                raw=opaque(original/'prepared'/f'cloud-{c}-{initialization}-{s}.bin')
                metadata=opaque(original/'prepared'/f'cloud-{c}-{initialization}-{s}.json')
                files.extend([raw,metadata]);banks.append(dict(context_index=c,initialization=initialization,stream=s,raw=raw,metadata=metadata))
        for s in range(4):
            record=opaque(original/'prepared'/f'start-{c}-{s}.json');ledger=opaque(original/'prepared'/f'start-{c}-{s}.jsonl')
            files.extend([record,ledger]);starts.append(dict(context_index=c,stream=s,record=record,ledger=ledger))
    prepared=save(original/'prepared/manifest.json',dict(complete=True,passed=True,all_attempts_retained=True,
        config_sha256=config_ref['sha256'],binding_sha256=binding_ref['sha256'],files=files,cloud_banks=banks,alternative_starts=starts))
    run=save(original/'run-binding.json',dict(complete=True,config_sha256=config_ref['sha256'],prelaunch_binding=binding_ref,
        prepared_manifest=prepared,prepared_files={x['path']:x['sha256'] for x in files}))
    ap=save(original/'preparation-audit-v2/protocol.json',{'scope':'toy independent audit'})
    save(original/'preparation-audit-v2/result.json',dict(schema='evolving-dimer-preparation-independent-audit-v1',complete=True,
        passed=True,physical=config['physical'],source_contexts=contexts,protocol=ap,
        input_sha256={v['path']:v['sha256'] for v in (config_ref,binding_ref,prepared)}))
    completed=[];chains=[];execution_files={v['path']:v['sha256'] for v in (config_ref,run)};cached={}
    for job in config['jobs']:
        ident=job['id'];directory=original/'execution'/f'job-{ident:03}'
        # Deliberately nonexistent: no old scientific journal may be read.
        trajectory=dict(path=str(directory/'trajectory.jsonl'),sha256='a'*64)
        terminal=save(directory/'terminal.json',dict(complete=True,conditional_target=True,job=job,blocks=4608,
            config_sha256=config_ref['sha256'],binding_sha256=run['sha256'],trajectory=trajectory,cpu_seconds=10.))
        execution_files[terminal['path']]=terminal['sha256'];execution_files[trajectory['path']]=trajectory['sha256']
        completed.append(dict(job=job,success=True,terminal_sha256=terminal['sha256']))
        chains.append(dict(job=job,trajectory=trajectory,metrics=dict(full_sampler_cpu_seconds=10.)))
        if job['context_index']==0 and job['arm'] in ('local','m4'):
            name=f'job-{ident:03}-observations.jsonl';cached[name]=opaque(analysis/'analysis'/name)['sha256']
    save(original/'dispatch/status.json',dict(complete=True,passed=True,active=[],unstarted=[],completed=completed))
    report=save(analysis/'analysis/analysis.json',dict(complete=True,chains=chains))
    ib=save(analysis/'analysis/input-binding.json',dict(config_sha256=config_ref['sha256'],run_binding_sha256=run['sha256'],
        source_files={'tools/historical_observer.py':source['sha256']}))
    manifest=save(analysis/'analysis/manifest.json',dict(complete=True,files={'analysis.json':report['sha256'],
        'input-binding.json':ib['sha256'],**cached}))
    execution=save(analysis/'execution-plan.json',dict(files=execution_files))
    save(analysis/'summary.json',dict(complete=True,passed=True,chains=96,analysis=report,
        manifest_sha256=manifest['sha256'],plan_sha256=execution['sha256']))
    save(analysis/'exit.json',dict(returncode=0,child_started=True,child_drained=True,error=None))
    return original,analysis


def build_fixture(root):
    example=opaque(root/'example.rs','// synthetic source, not compiled\n')
    executable=opaque(root/'executable','#!/bin/sh\nexit 99\n')
    name='src/toy.rs';text='// synthetic library\n';digest=hashlib.sha256(text.encode()).hexdigest()
    bundle=save(root/'bundle.json',dict(schema=1,files={name:dict(text=text,sha256=digest)}))
    source=dict(path=str(root/name),sha256=digest)
    sources={p.old.EXAMPLE:example,name:source}
    validation=save(root/'validation.json',dict(complete=True,passed=True,sources=sources,source_after=copy.deepcopy(sources),
        compiled_source_bundle=bundle,production_executable_unchanged=True,new_arm=p.ARM,
        synthetic_kernel_tests_executed=True,runs=[dict(returncode=0),dict(returncode=0)]))
    raw=save(root/'raw-build.json',dict(complete=True,passed=True))
    witness=save(root/'witness.json',dict(schema='two-root-example-build-witness-v1',complete=True,passed=True,
        new_arm=p.ARM,production_executable_unchanged=True,production_kernel_unchanged=True,
        canonical_observations=True,retained_prior_root=True,executable=executable,example_source=example,
        source_bundle=bundle,validation=validation,raw_build_witness=raw))
    return witness


def validation_fixture(root):
    hashes={str(v):p.sha(v) for v in p.source_paths().values()}
    return save(root/'python-validation.json',dict(complete=True,passed=True,source_before=hashes,source_after=hashes))


class PreparationTests(unittest.TestCase):
    def test_exact_inventory_and_paired_mapping_without_old_journal_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original,analysis=fixture(root);inputs=p.Inputs()
            config,authority,controls,files,mapping=p.inherited(inputs,original,analysis)
            self.assertEqual(len(controls['observations']),16);self.assertEqual(len(mapping),8)
            self.assertEqual({j['id'] for j in p.jobs()},set(range(8)))
            self.assertEqual(len({(j['initialization'],j['stream']) for j in p.jobs()}),8)
            self.assertTrue(all(set(m['controls'])=={'local','m4'} for m in mapping))
            self.assertEqual(p.allocation()['dimer_attempts'],36864)
            self.assertEqual(p.allocation()['local_attempts'],147456)
            self.assertFalse(any(Path(v).name=='trajectory.jsonl' for v in inputs.files))
            self.assertEqual(files,p.read(authority['run_binding']['path'])['prepared_files'])

    def test_missing_inventory_stale_cache_and_undrained_observer_rejected(self):
        for change in ('cache','inventory','exit'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                original,analysis=fixture(Path(temp))
                if change=='cache': next((analysis/'analysis').glob('*observations.jsonl')).write_text('changed')
                else:
                    path=original/'dispatch/status.json' if change=='inventory' else analysis/'exit.json'
                    value=p.read(path)
                    if change=='inventory':value['completed'].pop()
                    else:value['child_drained']=False
                    path.unlink();p.write(path,value)
                with self.assertRaises(ValueError):p.inherited(p.Inputs(),original,analysis)

    def test_preparation_binding_and_archive_integrity_required(self):
        for change in ('prepared','archive'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                original,analysis=fixture(Path(temp))
                if change=='prepared':(original/'prepared/cloud-0-source-0.bin').write_text('changed')
                else:(original/'common/source/tools/historical_observer.py').write_text('changed')
                with self.assertRaises(ValueError):p.inherited(p.Inputs(),original,analysis)

    def test_build_pin_tests_and_compiled_text_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);ref=build_fixture(root)
            with patch.object(p,'WITNESS_SHA',ref['sha256']):
                _,witness,compiled=p.validate_build(p.Inputs(),ref['path'])
                self.assertEqual(set(compiled),{'src/toy.rs'})
                Path(witness['executable']['path']).write_text('different build')
                with self.assertRaises(ValueError):p.validate_build(p.Inputs(),ref['path'])
            with self.assertRaises(ValueError):p.validate_build(p.Inputs(),ref['path'])
            tested=validation_fixture(root);p.validate_tests(p.Inputs(),tested['path'],p.source_paths())
            data=p.read(tested['path']);data['source_after'].pop(next(iter(data['source_after'])))
            Path(tested['path']).unlink();p.write(tested['path'],data)
            with self.assertRaises(ValueError):p.validate_tests(p.Inputs(),tested['path'],p.source_paths())

    def test_metadata_prepare_full_roundtrip_no_launch_and_frozen_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original,analysis=fixture(root);witness=build_fixture(root/'build');validation=validation_fixture(root)
            destination=root/'new'
            with patch.object(p,'WITNESS_SHA',witness['sha256']):
                receipt=p.prepare(destination,original,analysis,witness['path'],validation['path'])
                self.assertTrue(receipt['complete']);self.assertFalse(receipt['launched'])
                config=p.verify(destination);execution=p.read(destination/'execution-plan.json')
                self.assertEqual(len(execution['jobs']),8);self.assertEqual(execution['maximum_workers'],1)
                self.assertTrue(all(j['phase']=='producer' and j['terminal']['success_contract']=='complete' for j in execution['jobs']))
                self.assertTrue(all(j['cpu_limit_seconds']==3600 and j['wall_limit_seconds']==7200 for j in execution['jobs']))
                self.assertEqual(config['two_root_policy'],p.policy());self.assertEqual(config['master_seed'],p.old.MASTER)
                self.assertEqual(config['physical'],p.read(original/'config.json')['physical'])
                self.assertEqual(p.read(destination/'run-binding.json')['prepared_manifest'],p.record(original/'prepared/manifest.json'))
                self.assertFalse((destination/'execution').exists());self.assertFalse((destination/'chains').exists())
                self.assertFalse(any(Path(v).name=='trajectory.jsonl' for v in execution['files']))
                self.assertFalse(any(v.startswith(str(p.ROOT/'src')) for v in execution['files']))
                self.assertIn(str(destination/'common/source'/p.DRIVER),execution['files'])
                with self.assertRaises(ValueError):p.prepare(destination,original,analysis,witness['path'],validation['path'])
                config['two_root_policy']['root_probabilities']=[1.,0.]
                (destination/'config.json').unlink();p.write(destination/'config.json',config)
                with self.assertRaises(ValueError):p.verify(destination)

    def test_exclusive_failure_prefix_is_not_retryable(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);original,analysis=fixture(root);witness=build_fixture(root/'build');validation=validation_fixture(root)
            destination=root/'failed'
            with patch.object(p,'WITNESS_SHA',witness['sha256']),patch.object(p.shutil,'copyfile',side_effect=OSError('injected copy failure')):
                with self.assertRaises(OSError):p.prepare(destination,original,analysis,witness['path'],validation['path'])
            failure=p.read(destination/'preparation-failure.json')
            self.assertFalse(failure['complete']);self.assertIn('injected',failure['error'])
            self.assertFalse((destination/'preparation.json').exists())
            with self.assertRaises(ValueError):p.prepare(destination,original,analysis,witness['path'],validation['path'])

    def test_tested_source_mutation_during_or_after_archive_is_rejected(self):
        for stage in ('during_copy','before_publication'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);original,analysis=fixture(root);witness=build_fixture(root/'build')
                probe=root/'probe.py';probe.write_text('TESTED = True\n')
                sources={'tools/probe.py':probe,p.DRIVER:p.ROOT/p.DRIVER}
                destination=root/'new';copyfile=p.shutil.copyfile;verify=p.verify
                def changing_copy(source,target):
                    if Path(source)==probe and stage=='during_copy':probe.write_text('TESTED = False\n')
                    return copyfile(source,target)
                def changing_verify(base):
                    result=verify(base)
                    if stage=='before_publication':probe.write_text('TESTED = False\n')
                    return result
                with patch.object(p,'source_paths',return_value=sources):
                    validation=validation_fixture(root)
                    with patch.object(p,'WITNESS_SHA',witness['sha256']),patch.object(p.shutil,'copyfile',side_effect=changing_copy),\
                            patch.object(p,'verify',side_effect=changing_verify):
                        with self.assertRaisesRegex(ValueError,'Tested source changed'):
                            p.prepare(destination,original,analysis,witness['path'],validation['path'])
                self.assertFalse((destination/'preparation.json').exists())
                self.assertFalse(p.read(destination/'preparation-failure.json')['complete'])


if __name__=='__main__':unittest.main()
