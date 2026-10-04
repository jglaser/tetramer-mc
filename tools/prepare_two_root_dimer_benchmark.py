#!/usr/bin/env python3
"""Freeze eight two-root chains; reuse completed controls, starts and clouds.

Metadata/hash operations only. Historical Rust is authenticated in its original
archive, never against today's workspace. No old trajectory is parsed or copied.
The generated generic-controller plan contains producers only; analysis waits
for the complete eight-chain inventory.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys

import prepare_evolving_dimer_benchmark as old
import prepare_evolving_dimer_root_extension as shared
import run_native_class_physical_campaign as driver

ROOT = Path(__file__).resolve().parents[1]
require, read, sha, record, write = shared.require, shared.read, shared.sha, shared.record, shared.write
ARM = 'two_root_m4'
SELF = 'tools/prepare_two_root_dimer_benchmark.py'
TEST = 'tools/test_prepare_two_root_dimer_benchmark.py'
DRIVER = 'tools/run_native_class_physical_campaign.py'
WITNESS_SHA = '5b9d736839796cdd365e23bfa5c6c497f34fba7354e1b212c5455f7866e97a32'
LIMITS = dict(cpu_limit_seconds=3600, wall_limit_seconds=7200, address_space_limit_bytes=16*2**30)
ROLES = dict(root_order='two_root_m4/root_order', proposal='m4/proposal',
             threshold='m4/threshold', bath='m4/bath', accept='m4/accept',
             local='local/{attempt}/{proposal,bath,accept}')
SCOPE = ('Context0 conditional two-mobile contact-sampling control at rd1.4,z.0275. '
         'No assembly, native-registry or original-condition thermodynamic conclusion. '
         'Four independent streams per start; shared RNG roles make arm comparisons paired.')


def policy():
    return dict(schema='two-root-factorized-m4-policy-v1', root_probabilities=[.5,.5],
                selection='once_per_global_slot', internal_threshold_m=4)


def jobs():
    result=[]
    for initialization in old.INITIALIZATIONS:
        for stream in range(4):
            key=f'{old.MASTER}|two-root-m4|0|{initialization}|{stream}'
            result.append(dict(id=len(result),context_index=0,initialization=initialization,
                stream=stream,arm=ARM,seed_family=dict(context_index=0,initialization=initialization,stream=stream),
                queue_key=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')))
    return result


def allocation():
    return dict(schema='two-root-dimer-allocation-v1',contexts=1,context_indices=[0],arms=[ARM],
        initializations=old.INITIALIZATIONS,streams=4,chains=8,warmup_blocks=512,production_blocks=4096,
        blocks_per_chain=4608,local_attempts_per_block=4,local_member_order=[0,1,0,1],
        extra_dimer_per_block={ARM:1},local_attempts=147456,dimer_attempts=36864,
        retained_block_observations=36864,initial_observations=8,retained_initial_observations=36872,
        production_observations=32768,inherited_cloud_banks_used=8,inherited_alternative_starts_used=4,
        new_cloud_banks=0,new_preparation_attempts=0,new_raw_cloud_points=0,master_seed=old.MASTER,
        replacement=False,retry=False,native_classification=False,python_geometry_evaluations=0,
        physical_draws_during_python_preparation=0,maximum_workers=1,threads=1,per_chain_limits=LIMITS)


class Inputs:
    def __init__(self): self.files={}
    def bind(self,path,digest=None):
        ref=record(path)
        require(digest is None or ref['sha256']==digest,'Changed input: '+ref['path'])
        require(self.files.get(ref['path'],ref['sha256'])==ref['sha256'],'Conflicting input binding')
        self.files[ref['path']]=ref['sha256']; return ref
    def load(self,ref): return read(self.bind(ref['path'],ref['sha256'])['path'])
    def recheck(self):
        for path,digest in self.files.items(): require(sha(path)==digest,'Input changed during preparation: '+path)


def inherited(inputs,original,analysis_root):
    """Reuse frozen original authority and completed all-row observer evidence.

    verify_freeze checks common/source archives and original physical inputs,
    not workspace Rust. Only the sixteen consumed observation caches are hashed;
    old raw journals remain evidence inherited through the completed observer.
    """
    original,analysis_root=Path(original).resolve(),Path(analysis_root).resolve()
    config=old.verify_freeze(original)
    paths=dict(config='config.json',binding='binding.json',run_binding='run-binding.json',
        freeze='freeze.json',protocol='protocol.json',prepared_manifest='prepared/manifest.json',
        preparation_audit='preparation-audit-v2/result.json',dispatch_status='dispatch/status.json')
    authority={k:inputs.bind(original/v) for k,v in paths.items()}
    binding,run,prepared,audit,status=[read(authority[k]['path']) for k in
        ('binding','run_binding','prepared_manifest','preparation_audit','dispatch_status')]
    require(binding['config_sha256']==authority['config']['sha256'] and binding['freeze_sha256']==authority['freeze']['sha256']
        and binding['protocol_sha256']==authority['protocol']['sha256'],'Original prelaunch identity differs')
    require(run['complete'] is True and run['config_sha256']==authority['config']['sha256']
        and run['prelaunch_binding']==authority['binding'] and run['prepared_manifest']==authority['prepared_manifest'],
        'Original run binding differs')
    require(prepared['complete'] is True and prepared['passed'] is True and prepared['all_attempts_retained'] is True
        and prepared['config_sha256']==authority['config']['sha256'] and prepared['binding_sha256']==authority['binding']['sha256'],
        'Original preparation identity differs')
    require(audit['schema']=='evolving-dimer-preparation-independent-audit-v1' and audit['complete'] is True
        and audit['passed'] is True and audit['physical']==config['physical'] and audit['source_contexts']==config['contexts'],
        'Independent preparation audit differs')
    for key in ('config','binding','prepared_manifest'):
        ref=authority[key];require(audit['input_sha256'].get(ref['path'])==ref['sha256'],'Preparation audit binding differs')
    inputs.load(audit['protocol'])
    files={r['path']:r['sha256'] for r in prepared['files']}
    require(len(files)==len(prepared['files']) and files==run['prepared_files'],'Prepared file inventory differs')
    for path,digest in files.items(): inputs.bind(path,digest)
    require(len(prepared['cloud_banks'])==32 and {(v['context_index'],v['initialization'],v['stream']) for v in prepared['cloud_banks']}
        =={(c,i,s) for c in range(4) for i in old.INITIALIZATIONS for s in range(4)}
        and len(prepared['alternative_starts'])==16 and {(v['context_index'],v['stream']) for v in prepared['alternative_starts']}
        =={(c,s) for c in range(4) for s in range(4)},'Original preparation families differ')
    for group,keys in ((prepared['cloud_banks'],('raw','metadata')),(prepared['alternative_starts'],('record','ledger'))):
        for item in group:
            for key in keys: require(files.get(item[key]['path'])==item[key]['sha256'],'Unbound preparation component')
    require(status['complete'] is True and status['passed'] is True and not status['active'] and not status['unstarted'],
            'Original sampler is incomplete or undrained')
    done={v['job']['id']:v for v in status['completed']}
    require(len(done)==len(status['completed'])==96 and set(done)==set(range(96))
        and all(v['success'] is True and v['job']==config['jobs'][v['job']['id']] for v in done.values()),
        'Original completed job inventory differs')
    controls={k:inputs.bind(analysis_root/v) for k,v in dict(summary='summary.json',exit='exit.json',manifest='analysis/manifest.json',
        analysis='analysis/analysis.json',input_binding='analysis/input-binding.json',execution_plan='execution-plan.json').items()}
    summary,manifest,report,ib,execution=[read(controls[k]['path']) for k in
        ('summary','manifest','analysis','input_binding','execution_plan')]
    exit_record=read(controls['exit']['path'])
    require(exit_record['returncode']==0 and exit_record['child_started'] is True and exit_record['child_drained'] is True
        and exit_record['error'] is None and not (analysis_root/'failure.json').exists(),'Prior observer failed or undrained')
    require(summary['complete'] is True and summary['passed'] is True and summary['chains']==96
        and summary['analysis']==controls['analysis'] and summary['manifest_sha256']==controls['manifest']['sha256']
        and summary['plan_sha256']==controls['execution_plan']['sha256'] and manifest['complete'] is True
        and report['complete'] is True,
        'Completed control observer identity differs')
    require(manifest['files']['analysis.json']==controls['analysis']['sha256']
        and manifest['files']['input-binding.json']==controls['input_binding']['sha256']
        and ib['config_sha256']==authority['config']['sha256'] and ib['run_binding_sha256']==authority['run_binding']['sha256'],
        'Control observer input authority differs')
    for key in ('config','run_binding'):
        ref=authority[key];require(execution['files'].get(ref['path'])==ref['sha256'],'Prior observer did not bind source authority')
    # These historical sources are held in the old immutable archive, even if
    # a later workspace version has changed. No old arithmetic is replayed.
    for name,digest in ib['source_files'].items():
        inputs.bind(original/'common/source'/name,digest)
    selected=[j for j in config['jobs'] if j['context_index']==0 and j['arm'] in ('local','m4')]
    require(len(selected)==16 and {(j['arm'],j['initialization'],j['stream']) for j in selected}
        =={(a,i,s) for a in ('local','m4') for i in old.INITIALIZATIONS for s in range(4)},'Control family inventory differs')
    reported={c['job']['id']:c for c in report['chains']}
    require(len(reported)==len(report['chains'])==96,'Original observer chain inventory differs')
    controls['observations']={}; controls['terminals']={}
    for job in selected:
        ident=job['id']; terminal=inputs.bind(original/'execution'/f'job-{ident:03}'/'terminal.json',done[ident]['terminal_sha256'])
        value=read(terminal['path']); chain=reported[ident]
        require(value['complete'] is True and value['conditional_target'] is True and value['job']==job==chain['job']
            and value['blocks']==4608 and value['config_sha256']==authority['config']['sha256']
            and value['binding_sha256']==authority['run_binding']['sha256']
            and chain['trajectory']==value['trajectory'] and execution['files'].get(terminal['path'])==terminal['sha256']
            and execution['files'].get(value['trajectory']['path'])==value['trajectory']['sha256']
            and value['cpu_seconds']==chain['metrics']['full_sampler_cpu_seconds']
            and not (Path(terminal['path']).parent/'failure.json').exists(),'Cached control terminal/replay differs')
        name=f'job-{ident:03}-observations.jsonl'
        controls['observations'][str(ident)]=inputs.bind(analysis_root/'analysis'/name,manifest['files'][name])
        controls['terminals'][str(ident)]=terminal
    mapping=[]
    for job in jobs():
        controls_by_arm={j['arm']:j['id'] for j in selected if j['initialization']==job['initialization'] and j['stream']==job['stream']}
        mapping.append(dict(job_id=job['id'],context_index=0,initialization=job['initialization'],stream=job['stream'],controls=controls_by_arm))
    return config,authority,controls,files,mapping


def validate_build(inputs,path):
    ref=inputs.bind(path,WITNESS_SHA); witness=read(ref['path'])
    require(witness['schema']=='two-root-example-build-witness-v1' and witness['complete'] is True and witness['passed'] is True
        and witness['new_arm']==ARM and all(witness[k] is True for k in
        ('production_executable_unchanged','production_kernel_unchanged','canonical_observations','retained_prior_root')),
        'Wrong isolated two-root build witness')
    for key in ('executable','source_bundle','example_source','validation','raw_build_witness'):
        inputs.bind(witness[key]['path'],witness[key]['sha256'])
    bundle=read(witness['source_bundle']['path']); validation=read(witness['validation']['path'])
    require(validation['complete'] is True and validation['passed'] is True and validation['sources']==validation['source_after']
        and validation['compiled_source_bundle']==witness['source_bundle'] and validation['production_executable_unchanged'] is True
        and validation['new_arm']==ARM and validation['synthetic_kernel_tests_executed'] is True
        and len(validation['runs'])==2 and all(v['returncode']==0 for v in validation['runs']), 'Unsuccessful isolated build validation')
    require(validation['sources'][old.EXAMPLE]==witness['example_source'],'Example source/test binding differs')
    compiled={}
    for name,item in bundle['files'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'Unsafe compiled source entry')
        require(hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256']
            and validation['sources'][name]['sha256']==item['sha256'],'Compiled source/test mismatch')
        compiled[name]=item['sha256']
    return ref,witness,compiled


def source_paths():
    from analyze_two_root_dimer_benchmark import SOURCE_FILES
    return {str(p.relative_to(ROOT)):p for p in old.source_dependencies([ROOT/n for n in [SELF,TEST,DRIVER]+SOURCE_FILES])}


def validate_tests(inputs,path,sources):
    ref=inputs.bind(path); receipt=read(ref['path'])
    require(receipt.get('complete') is True and receipt.get('passed') is True
        and receipt.get('source_before')==receipt.get('source_after') and isinstance(receipt.get('source_after'),dict),
        'Incomplete or changed Python validation')
    tested=receipt['source_after']; admitted={}
    for name,p in sources.items():
        admitted[name]=tested.get(str(p),tested.get(name))
        require(admitted[name]==sha(p),'Untested/changed Python source: '+name)
    return ref,admitted


def recheck_sources(sources,admitted):
    require(set(sources)==set(admitted),'Source inventory changed during preparation')
    for name,path in sources.items():
        require(sha(path)==admitted[name],'Tested source changed during preparation: '+name)


def configuration(original,root,authority,controls,mapping,compiled):
    config=copy.deepcopy(original)
    config.update(scientific_allocation=record(root/'common/scientific-allocation.json'),protocol=record(root/'protocol.json'),
        jobs=jobs(),allocation=allocation(),compiled_source_sha256=compiled,output=str(root/'chains'),
        inherited_campaign=authority,control_analysis=controls,control_mapping=mapping,two_root_policy=policy())
    config['limits']['cpu_seconds']=LIMITS['cpu_limit_seconds']
    return config


def verify(base):
    base=Path(base).resolve();f=read(base/'freeze.json')
    require(f['schema']=='two-root-dimer-freeze-v1' and f['complete'] is True and f['scientific_execution_started'] is False,'Invalid freeze')
    for name,digest in f['files'].items():
        path=(base/name).resolve();require(path.is_relative_to(base) and sha(path)==digest,'Frozen artifact changed: '+name)
    for path,digest in f['input_sha256'].items(): require(sha(path)==digest,'Inherited input changed: '+path)
    config=read(base/'config.json');original=read(old.checked_file(config['inherited_campaign']['config']))
    expected=copy.deepcopy(original)
    for key in ('scientific_allocation','protocol','jobs','allocation','compiled_source_sha256','output',
                'inherited_campaign','control_analysis','control_mapping','two_root_policy'): expected[key]=config[key]
    expected['limits']['cpu_seconds']=LIMITS['cpu_limit_seconds']
    require(config==expected and config['jobs']==jobs() and config['allocation']==allocation()
        and config['two_root_policy']==policy() and config['output']==str(base/'chains'),'Undeclared matched-control change')
    protocol=read(old.checked_file(config['protocol']))
    require(protocol['roles']==ROLES and protocol['two_root_policy']==policy() and protocol['allocation']==allocation(),
            'Changed root/RNG/schedule policy')
    return config


def prepare(base,original,control_analysis,build_witness,validation):
    base=Path(base).resolve();require(not base.exists(),'Fresh campaign required; no overwrite/retry')
    inputs=Inputs(); original_config,authority,controls,prepared,mapping=inherited(inputs,original,control_analysis)
    witness_ref,witness,compiled=validate_build(inputs,build_witness)
    sources=source_paths(); validation_ref,admitted_sources=validate_tests(inputs,validation,sources)
    from analyze_two_root_dimer_benchmark import analysis_plan
    analysis=analysis_plan()
    # All physical/config references are consumed unchanged by the Rust worker.
    for value in original_config.values():
        if isinstance(value,dict) and set(value)=={'path','sha256'}: inputs.bind(value['path'],value['sha256'])
    inputs.recheck();base.mkdir(parents=True)
    try:
        (base/'common').mkdir()
        for name,path in sources.items():
            target=base/'common/source'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);require(sha(target)==admitted_sources[name],'Tested source changed during copy: '+name)
        example=base/'common/source'/old.EXAMPLE;example.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(witness['example_source']['path'],example)
        require(sha(example)==witness['example_source']['sha256'],'Example archive differs')
        for key,name in (('executable','evolving_dimer_benchmark'),('source_bundle','source-bundle.json')):
            shutil.copyfile(witness[key]['path'],base/'common'/name)
            require(sha(base/'common'/name)==witness[key]['sha256'],'Compiled archive differs')
        executable=base/'common/evolving_dimer_benchmark';executable.chmod(0o755)
        write(base/'common/scientific-allocation.json',allocation());write(base/'analysis-plan.json',analysis)
        source_hashes=dict(admitted_sources);source_hashes[old.EXAMPLE]=witness['example_source']['sha256']
        protocol=dict(schema='two-root-dimer-protocol-v1',allocation=allocation(),two_root_policy=policy(),roles=ROLES,
            inherited_campaign=authority,control_analysis=controls,control_mapping=mapping,build_witness=witness_ref,
            validation=validation_ref,analysis_plan=record(base/'analysis-plan.json'),source_files=source_hashes,
            scope=SCOPE,all_attempts_retained=True,analysis_after_complete_inventory=True,retry=False,replacement=False,
            inherited_evidence='Completed original all-row/contact observer and independent preparation audit. Old raw journals not reread; only the sixteen consumed observation caches are rehashed.',
            selection='Context0 chosen after completed diagnostic development; all eight start/stream families fixed without trajectory-outcome filtering.')
        write(base/'protocol.json',protocol)
        config=configuration(original_config,base,authority,controls,mapping,compiled);write(base/'config.json',config)
        files={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()}
        write(base/'freeze.json',dict(schema='two-root-dimer-freeze-v1',complete=True,scientific_execution_started=False,
            files=files,input_sha256=dict(inputs.files)))
        write(base/'binding.json',dict(schema='evolving-dimer-binding-v1',config_sha256=sha(base/'config.json'),
            protocol_sha256=sha(base/'protocol.json'),freeze_sha256=sha(base/'freeze.json'),example_source_sha256=sha(example),
            compiled_source_bundle_sha256=sha(base/'common/source-bundle.json'),executable_sha256=sha(executable),
            isolated_build_witness=witness_ref))
        write(base/'run-binding.json',dict(schema='evolving-dimer-run-binding-v1',complete=True,
            config_sha256=sha(base/'config.json'),protocol_sha256=sha(base/'protocol.json'),prelaunch_binding=record(base/'binding.json'),
            prepared_manifest=authority['prepared_manifest'],prepared_files=prepared,cloud_banks=32,alternative_starts=16,
            inherited_preparation_authority=authority))
        execution_jobs=[]
        for job in sorted(jobs(),key=lambda v:(v['queue_key'],v['id'])):
            execution_jobs.append(dict(id=f"job-{job['id']:03}",population=f"context0-{job['initialization']}-{job['stream']}",
                phase='producer',argv=[str(executable),'--mode','run','--config',str(base/'config.json'),
                    '--binding',str(base/'run-binding.json'),'--job',str(job['id'])],
                terminal=dict(path=str(base/'chains'/f"job-{job['id']:03}"/'terminal.json'),success_contract='complete'),**LIMITS))
        consumed=dict(inputs.files)
        consumed.update({str(p):sha(p) for p in base.rglob('*') if p.is_file()})
        execution=dict(schema=driver.SCHEMA,root=str(base),maximum_workers=1,threads=1,files=consumed,jobs=execution_jobs,
            executable_resolutions={str(executable):str(executable)},preparation_receipt=str(base/'preparation.json'),scope=SCOPE)
        write(base/'execution-plan.json',execution);inputs.recheck();verify(base);recheck_sources(sources,admitted_sources)
        receipt=dict(schema='two-root-dimer-preparation-v1',complete=True,launched=False,protocol=record(base/'protocol.json'),
            execution_plan=record(base/'execution-plan.json'),config=record(base/'config.json'),allocation=allocation(),
            new_physical_draws=0,new_geometry_queries=0,source_archived=True)
        write(base/'preparation.json',receipt)
        spec=importlib.util.spec_from_file_location('two_root_frozen_driver',base/'common/source'/DRIVER)
        frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        frozen.verify_plan(base/'execution-plan.json',execution,receipt['execution_plan']['sha256'],fresh=True)
        return receipt
    except BaseException as error:
        write(base/'preparation-failure.json',dict(complete=False,launched=False,error=f'{type(error).__name__}: {error}'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','original','control-analysis','build-witness','validation'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(__import__('json').dumps(prepare(args.root,args.original,args.control_analysis,args.build_witness,args.validation),indent=2))
