#!/usr/bin/env python3
"""Freeze matched fused/unfused singleton chains without scientific queries."""
from __future__ import annotations
import argparse
import copy
import hashlib
from pathlib import Path
import shutil

import prepare_evolving_dimer_benchmark as old
import prepare_evolving_dimer_root_extension as prior

ROOT, EXAMPLE = prior.ROOT, prior.EXAMPLE
require, read, sha, record, runtime, write = (prior.require, prior.read, prior.sha,
                                           prior.record, prior.runtime, prior.write)
ARMS = ('singleton_two_neighbor', 'singleton_two_neighbor_unfused')
LIMITS = dict(max_workers=3, cpu_seconds=3600, wall_seconds=7200, address_space_bytes=16*1024**3)
ROLE_MAP = dict(proposal='singleton_two_neighbor/proposal', bath='singleton_two_neighbor/bath',
                accept='singleton_two_neighbor/accept', local='local/{attempt}/{proposal,bath,accept}')


def singleton_policy():
    return dict(schema='two-neighbor-singleton-policy-v1', uniform_half_width=160.,
        uniform_probability=.5, trial_cap=32, member_schedule='alternating_0_first',
        oligomer=dict(multi_contact_mass=.8, max_mismatch=12., pair_distance_A=8.,
            pair_angle_degrees=60., max_candidates=4096, max_hard_checks=256, max_components=32))


def jobs():
    values=[]
    for context in range(4):
        for initialization in old.INITIALIZATIONS:
            for stream in range(4):
                for arm in ARMS:
                    key=f'{old.MASTER}|two-neighbor-singleton|{context}|{initialization}|{stream}|{arm}'
                    values.append(dict(id=len(values),context_index=context,initialization=initialization,
                        stream=stream,arm=arm,
                        seed_family=dict(context_index=context,initialization=initialization,stream=stream),
                        queue_key=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')))
    return sorted(values,key=lambda v:(v['queue_key'],v['id']))


def allocation():
    return dict(schema='two-neighbor-singleton-allocation-v1',contexts=4,arms=list(ARMS),
        initializations=old.INITIALIZATIONS,streams=4,chains=64,warmup_blocks=512,
        production_blocks=4096,blocks_per_chain=4608,local_attempts_per_block=4,
        local_member_order=[0,1,0,1],extra_singleton_per_block={a:1 for a in ARMS},
        singleton_member_schedule='(block-1)%2',local_attempts=1179648,singleton_attempts=294912,
        retained_block_observations=294912,initial_observations=64,
        retained_initial_observations=294976,production_observations=262144,
        inherited_alternative_starts=16,new_cloud_banks=0,new_preparation_attempts=0,
        guidance_clouds_used=0,new_raw_guidance_points=0,master_seed=old.MASTER,
        replacement=False,native_classification=False,python_geometry_evaluations=0,
        physical_draws_during_python_preparation=0,execution_limits=LIMITS,
        limits_scope='Per chain, with all attempts retained; no refill, restart or replacement.')


def inherited(original,analysis_root):
    config,_,authority,controls,prepared_files=prior.inherited(original,analysis_root)
    original,analysis_root=Path(original).resolve(),Path(analysis_root).resolve()
    status=read(original/'dispatch/status.json');manifest=read(analysis_root/'analysis/manifest.json')
    selected=[j for j in config['jobs'] if j['arm'] in ('local','m4')]
    require(len(selected)==64,'Exactly64 cached local/m4 controls required')
    for job in selected:
        terminal=original/'execution'/f"job-{job['id']:03}"/'terminal.json'
        done=next(x for x in status['completed'] if x['job']['id']==job['id'])
        require(done['success'] is True and sha(terminal)==done['terminal_sha256']
            and read(terminal)['job']==job,'Cached control terminal differs')
        name=f"job-{job['id']:03}-observations.jsonl";item=record(analysis_root/'analysis'/name)
        require(manifest['files'][name]==item['sha256'],'Cached observation binding differs')
        controls['observations'][str(job['id'])]=item
    require(len(controls['observations'])==64,'Cached control inventory differs')
    return config,authority,controls,prepared_files


def validation_bindings(receipts):
    require(set(receipts)=={'primitive','example','observer','controller','build'},
        'Four validation receipts and the isolated build receipt required')
    result={};names=set()
    for label,path in receipts.items():
        value=read(path);require(value.get('passed') is True,'Failed validation '+label)
        require(value.get('source_unchanged',True) is True,'Changed tested source '+label)
        sources=value.get('source_after',value.get('source_sha256',
            value.get('source_sha256_recorded_after_test')))
        if 'source_before' in value: require(value['source_before']==value['source_after'],'Tested source changed')
        require(isinstance(sources,dict) and sources,'Missing tested source closure')
        for name,digest in sources.items():
            p=Path(name) if Path(name).is_absolute() else ROOT/name
            p=p.resolve();require(p.is_relative_to(ROOT),'Tested source outside repository')
            require(sha(p)==digest,'Changed tested source '+str(p));names.add(str(p.relative_to(ROOT)))
        if label=='build':
            require(value.get('production_unchanged') is True and value.get('source_unchanged') is True,
                'Isolated build changed production or source')
            require(value.get('commands') and all(c['exit_code']==0 for c in value['commands']),
                'Isolated build did not complete')
            old.checked_file(value['executable']);old.checked_file(value['compiled_source_bundle'])
        result[label]=record(path)
    required={'src/two_neighbor_singleton.rs','src/evolving_dimer.rs','src/basin_involution.rs',
        'src/oligomer_proposal.rs','tests/two_neighbor_singleton.rs',EXAMPLE,
        'tools/prepare_two_neighbor_singleton_benchmark.py','tools/run_two_neighbor_singleton_benchmark.py',
        'tools/test_two_neighbor_singleton_benchmark.py','tools/analyze_two_neighbor_singleton_benchmark.py',
        'tools/test_analyze_two_neighbor_singleton_benchmark.py'}
    require(required<=names,'Missing directly validated implementation or tests')
    return result,names


def prepare(base,original,control_analysis,receipts):
    base=Path(base).resolve();require(not base.exists(),'Fresh campaign required')
    original_config,authority,controls,prepared=inherited(original,control_analysis)
    validations,validated=validation_bindings(receipts)
    from analyze_two_neighbor_singleton_benchmark import analysis_plan
    analysis=analysis_plan()
    compiled_paths=[ROOT/p for p in ('Cargo.toml','Cargo.lock','build.rs','vendor/README.md')]+sorted((ROOT/'src').rglob('*.rs'))
    compiled={str(p.relative_to(ROOT)):sha(p) for p in compiled_paths}
    toolnames=['tools/prepare_two_neighbor_singleton_benchmark.py','tools/run_two_neighbor_singleton_benchmark.py',
        'tools/test_two_neighbor_singleton_benchmark.py']+analysis['source_files']
    closure=compiled_paths+[ROOT/EXAMPLE]+old.source_dependencies([ROOT/p for p in toolnames])
    closure += [ROOT/p for p in validated]+sorted(p for p in (ROOT/'vendor').rglob('*') if p.is_file())
    sources={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(closure)}
    base.mkdir(parents=True);(base/'common').mkdir()
    write(base/'common/scientific-allocation.json',allocation());write(base/'analysis-plan.json',analysis)
    for name,digest in sources.items():
        path=base/'common/source'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,path);require(sha(path)==digest,'Source changed during archival')
    protocol=dict(schema='two-neighbor-singleton-protocol-v1',allocation=allocation(),
        singleton_policy=singleton_policy(),effective_fusion_mass={ARMS[0]:.8,ARMS[1]:0.},
        inherited_campaign=authority,control_analysis=controls,roles=ROLE_MAP,
        validation_receipts=validations,analysis_plan=record(base/'analysis-plan.json'),
        source_files=sources,runtime=runtime(),retry=False,replacement=False,
        interpretation='Two mobile labels,262 fixed spectators,rd1.4,z.0275. Sampling diagnostic, not assembly or original-condition thermodynamics.',
        physical_target_unchanged=True,preparation='Reuse original16 displaced starts and source; zero point-cloud guidance.',
        correction='Complete .5 uniform +.5 fused/unfused density ratio; shared finite-cap factor cancels. One many-body bath and one MH decision.',
        evidence_boundary='Synthetic normalization, cap/replay, strict scoring and bath/restart integration; all-row scalar plus endpoint observer checks. No independent full protein catalogue-fit or physical bath point reconstruction claimed.')
    write(base/'protocol.json',protocol)
    config=copy.deepcopy(original_config)
    config.update(scientific_allocation=record(base/'common/scientific-allocation.json'),
        protocol=record(base/'protocol.json'),jobs=jobs(),allocation=allocation(),
        compiled_source_sha256=compiled,output=str(base/'execution'),inherited_campaign=authority,
        control_analysis=controls,singleton_policy=singleton_policy())
    config['limits']['cpu_seconds']=LIMITS['cpu_seconds']
    write(base/'config.json',config)
    bindings={}
    def collect(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256'}:bindings[value['path']]=value['sha256']
            else:
                for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(config);collect(validations);bindings.update(prepared)
    files={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()}
    write(base/'freeze.json',dict(schema='two-neighbor-singleton-freeze-v1',complete=True,
        scientific_execution_started=False,files=files,input_sha256=bindings,runtime=runtime()))
    verify(base)
    return dict(root=str(base),config=record(base/'config.json'),freeze=record(base/'freeze.json'),chains=64,launched=False)


def verify(base):
    base=Path(base).resolve();f=read(base/'freeze.json')
    require(f['schema']=='two-neighbor-singleton-freeze-v1' and f['complete'] is True
        and f['scientific_execution_started'] is False,'Invalid freeze')
    for name,digest in f['files'].items():
        p=(base/name).resolve();require(p.is_relative_to(base) and sha(p)==digest,'Frozen source/input changed '+name)
    for path,digest in f['input_sha256'].items():require(sha(path)==digest,'Inherited input changed '+path)
    config=read(base/'config.json');original=read(old.checked_file(config['inherited_campaign']['config']))
    expected=copy.deepcopy(original)
    for key in ('scientific_allocation','protocol','jobs','allocation','compiled_source_sha256',
                'output','inherited_campaign','control_analysis','singleton_policy'):
        expected[key]=config[key]
    expected['limits']['cpu_seconds']=LIMITS['cpu_seconds']
    require(config==expected and config['jobs']==jobs() and config['allocation']==allocation()
        and config['singleton_policy']==singleton_policy() and config['output']==str(base/'execution'),
        'New config changes exceed matched scope')
    p=read(base/'protocol.json');require(p['roles']==ROLE_MAP and p['singleton_policy']==singleton_policy(),
        'Proposal policy or RNG roles differ')
    return config


def bind(base,executable,source_bundle=None):
    base=Path(base).resolve();config=verify(base)
    require(not(base/'binding.json').exists() and not(base/'run-binding.json').exists(),'Already bound')
    paths=[Path(source_bundle)] if source_bundle else sorted((ROOT/'target-validation-line-guide/release/build').glob('tetramer-mc-*/out/source-bundle.json'))
    matches=[]
    for path in paths:
        bundle=read(path)
        require(all(hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256'] for v in bundle['files'].values()),'Corrupt bundle')
        if {k:v['sha256'] for k,v in bundle['files'].items()}==config['compiled_source_sha256']:matches.append(path)
    require(matches and len({sha(p) for p in matches})==1,'No unique matching compiled bundle')
    require(sha(ROOT/EXAMPLE)==sha(base/'common/source'/EXAMPLE),'Example changed since freeze')
    protocol=read(base/'protocol.json')
    build=read(old.checked_file(protocol['validation_receipts']['build']))
    require(record(executable)==build['executable']
        and sha(matches[0])==build['compiled_source_bundle']['sha256']
        and build['source_sha256'][EXAMPLE]==sha(base/'common/source'/EXAMPLE),
        'Executable, compiled bundle or example differs from validated isolated build')
    target=base/'common/evolving_dimer_benchmark';require(not target.exists(),'Partial binding')
    digest=sha(executable);shutil.copyfile(executable,target);require(sha(target)==digest,'Executable changed');target.chmod(0o755)
    shutil.copyfile(matches[0],base/'common/source-bundle.json')
    write(base/'binding.json',dict(schema='evolving-dimer-binding-v1',config_sha256=sha(base/'config.json'),
        protocol_sha256=sha(base/'protocol.json'),freeze_sha256=sha(base/'freeze.json'),
        example_source_sha256=sha(base/'common/source'/EXAMPLE),
        compiled_source_bundle_sha256=sha(base/'common/source-bundle.json'),executable_sha256=digest,
        isolated_build_receipt=protocol['validation_receipts']['build']))
    authority=config['inherited_campaign'];inherited_run=read(old.checked_file(authority['run_binding']))
    write(base/'run-binding.json',dict(schema='evolving-dimer-run-binding-v1',complete=True,
        config_sha256=sha(base/'config.json'),protocol_sha256=sha(base/'protocol.json'),
        prelaunch_binding=record(base/'binding.json'),prepared_manifest=authority['prepared_manifest'],
        prepared_files=inherited_run['prepared_files'],guidance_clouds_used=0,alternative_starts=16,
        inherited_preparation_authority=authority))
    return dict(binding=record(base/'binding.json'),run_binding=record(base/'run-binding.json'),executable=record(target),launched=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--original',type=Path);p.add_argument('--control-analysis',type=Path)
    p.add_argument('--validation-receipt',action='append',default=[])
    p.add_argument('--bind-executable',type=Path);p.add_argument('--source-bundle',type=Path);a=p.parse_args()
    if a.bind_executable:result=bind(a.root,a.bind_executable,a.source_bundle)
    else:
        if a.original is None or a.control_analysis is None:p.error('--original and --control-analysis required')
        result=prepare(a.root,a.original,a.control_analysis,dict(v.split('=',1) for v in a.validation_receipt))
    print(__import__('json').dumps(result,indent=2))
