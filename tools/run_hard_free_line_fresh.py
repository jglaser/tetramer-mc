#!/usr/bin/env python3
"""Freeze and drain eight proposal/audit/observer triples, without Poisson work."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS'):os.environ[key]='1'
from prepare_hard_free_line_fresh import ARMS,jobs,read,sha,write,require,REGION_SHA,SHAPE_SHA,NATIVE_SHA
from report_hard_free_line_fresh import report,read_rows
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_confirmation import worker_environment,runtime
from run_contact_tail_pilot import verify_frozen,copy_frozen
from run_full_vessel_comparison import execute_group
from run_mobile_posterior_pilot import verify_bundle

SCHEMA='hard-free-line-fresh-controller-v1'
BINARY_SHA='937a8f999759eae16db0541344f632d461ee92b251c682a4d617907d56d8736d'
BUNDLE_SHA='6436087198eb1ca6ce067c21cb4cf04529c2fb787d0e9c4dc083dee0cea84346'
REFERENCE_SOURCES={'hard_free_line_reference.py':'7731cf2b64d9c501a679315d8c451d67d6704db234ee4fd5ef5981c74aee9156',
 'analyze_contact_line_audit.py':'b123db87aa843c83f18f50bb72443d4b0624960a77fb92fc24126011313b6a93'}


def validate_design(plan):
    expected=dict(arms=ARMS,jobs=jobs(),populations_per_arm=4,draws_per_population=256,total_attempted_draws=2048,
        probes_per_population=0,alpha=.5,beta={'baseline':0.,'xyz':1.},axes=[0,1,2],minimum_conditional_mass=1e-12,
        maximum_CPU_workers=1,new_Poisson_clouds=0,new_physical_mass_estimates=0,native_definition_sha256=NATIVE_SHA,
        all_attempts_retained=True,no_retries=True,no_optional_stopping=True,no_autoextension=True,
        physical_gates_unchanged=True,physical_campaign_ready=False)
    require(all(plan.get(k)==v for k,v in expected.items()),'Fixed fresh-proposal allocation changed')


def steps(out,python):
    out=Path(out).resolve();c=out/'common';result=[]
    for j in jobs():
        directory=out/j['arm']/j['id'];name=j['arm']+'-'+j['id']
        commands={
            'proposal':[str(c/'contact-line-guide-audit'),'--config',str(c/'config.json'),'--region',str(c/'region.json'),
                '--importance-guide',str(c/(j['arm']+'.json')),'--samples','256','--seed',str(j['seed']),'--out',str(directory)],
            'audit':[python,'-B','-E',str(c/'reference/hard_free_line_reference.py'),'--root',str(directory),
                '--out',str(directory/'independent-audit.json')],
            'observer':[python,'-B','-E',str(c/'observe_hard_free_line_fresh.py'),'--root',str(directory),
                '--frozen',str(c/'preparation/classifier'),'--config',str(c/'config.json'),'--shape',str(c/'shape.json'),
                '--out',str(directory/'observer'),'--arm',j['arm'],'--population',j['id'],'--seed',str(j['seed'])]}
        for kind,command in commands.items():
            destination=directory if kind=='proposal' else directory/('independent-audit.json' if kind=='audit' else 'observer')
            result.append(dict(id=name+'-'+kind,arm=j['arm'],population=j['id'],seed=j['seed'],kind=kind,
                status='pending',directory=str(destination),population_directory=str(directory),
                log=str(out/'logs'/(name+'-'+kind+'.log')),command=command))
    return result


def freeze(out,preparation,archived_score,references):
    out,preparation,archived_score=map(lambda p:Path(p).resolve(),(out,preparation,archived_score))
    require(not out.exists(),'Fresh execution directory required');verify_frozen(preparation);verify_frozen(archived_score)
    plan=read(preparation/'plan.json');validate_design(plan)
    old=archived_score/'common';binary=old/'contact-line-guide-audit';bundle=old/'source-bundle.json'
    require(sha(binary)==BINARY_SHA and sha(bundle)==BUNDLE_SHA,'Use the completed score executable, not a new build')
    require(read(archived_score/'status.json')['complete'] and read(archived_score/'independent-audit.json')['complete'],
            'Archived scoring did not complete its independent validation')
    source,rust_hashes=verify_bundle(binary,bundle,old/'rust-source')
    for name,h in REFERENCE_SOURCES.items():require(sha(old/name)==h,'Archived independent reference changed')
    dependencies=local_dependencies([Path(__file__)])
    require('observe_hard_free_line_fresh.py' in dependencies and 'report_hard_free_line_fresh.py' in dependencies,
        'Observer/report source closure missing')
    references=[Path(p).resolve() for p in references]
    require(references and all(p.is_file() for p in references),'Validation receipts required')
    for path in references:
        receipt=read(path);require(receipt.get('complete') is True and receipt.get('passed',True) is True,
            'Failed/incomplete validation: '+str(path))
    out.mkdir();common=out/'common';common.mkdir();(out/'logs').mkdir()
    for arm in ARMS:(out/arm).mkdir()
    copy_frozen(preparation,common/'preparation')
    for name,path in dependencies.items():shutil.copy2(path,common/name)
    shutil.copy2(binary,common/'contact-line-guide-audit');shutil.copy2(bundle,common/'source-bundle.json')
    for name,entry in source['files'].items():
        target=common/'rust-source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(entry['text'])
    (common/'reference').mkdir()
    for name in REFERENCE_SOURCES:shutil.copy2(old/name,common/'reference'/name)
    for name in ('shape.json','region.json'):shutil.copy2(preparation/'common'/name,common/name)
    config=read(preparation/'common/config.json');config['shape']=str(common/'shape.json');write(common/'config.json',config)
    for arm in ARMS:shutil.copy2(preparation/'guides'/f'{arm}.json',common/f'{arm}.json')
    receipts=[]
    for i,path in enumerate(references):
        target=common/f'reference-receipt-{i:02}-{path.name}';shutil.copy2(path,target)
        receipts.append(dict(original=str(path),archive=target.name,sha256=sha(path)))
    protocol=dict(plan,schema=SCHEMA,execution_ready=True,preparation=str(preparation),
        preparation_plan_sha256=sha(preparation/'plan.json'),preparation_freeze_sha256=sha(preparation/'freeze.json'),
        archived_score=str(archived_score),archived_score_protocol_sha256=sha(archived_score/'protocol.json'),
        binary_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,rust_sources=rust_hashes,
        reference_sources=REFERENCE_SOURCES,reference_receipts=receipts,
        python_sources={name:sha(path) for name,path in dependencies.items()},controller_sha256=sha(__file__),
        runtime=runtime(),python_executable=sys.executable,steps=steps(out,sys.executable),
        scope='Exactly eight sequential 256-draw proposal populations. Each is independently audited before its valid poses are classified once. No probes or Poisson sampling.')
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    validate(out,sha(out/'protocol.json'));return sha(out/'protocol.json')


def validate(out,expected):
    out=Path(out).resolve();verify_frozen(out);require(sha(out/'protocol.json')==expected,'Protocol changed')
    p=read(out/'protocol.json');validate_design(p)
    require(p['schema']==SCHEMA and runtime()==p['runtime'] and sys.flags.optimize==0 and sys.executable==p['python_executable'],
        'Controller runtime changed')
    require(sha(__file__)==p['controller_sha256'] and
        {n:sha(v) for n,v in local_dependencies([Path(__file__)]).items()}==p['python_sources'],'Controller source closure changed')
    c=out/'common';verify_bundle(c/'contact-line-guide-audit',c/'source-bundle.json',c/'rust-source')
    require(sha(c/'contact-line-guide-audit')==BINARY_SHA and sha(c/'source-bundle.json')==BUNDLE_SHA,'Executable changed')
    require(p['reference_sources']==REFERENCE_SOURCES and all(sha(c/'reference'/n)==h for n,h in REFERENCE_SOURCES.items()),
        'Independent source closure changed')
    require(p['steps']==steps(out,sys.executable),'Exact step commands changed')
    prep=Path(p['preparation']);verify_frozen(prep)
    require(sha(prep/'plan.json')==p['preparation_plan_sha256'] and sha(prep/'freeze.json')==p['preparation_freeze_sha256'],
        'Preparation binding changed')
    return p


def verify_output(out,p,step):
    directory=Path(step['population_directory']);common=out/'common';manifest=read(directory/'manifest.json');summary=read(directory/'summary.json')
    expected=dict(schema='hard-free-line-guide-audit-v1',samples=256,seed=step['seed'],physical_jobs=0,
        executable_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,config_sha256=sha(common/'config.json'),
        region_sha256=REGION_SHA,shape_sha256=SHAPE_SHA,guide_sha256=sha(common/(step['arm']+'.json')),probes_sha256=None)
    require(all(manifest.get(k)==v for k,v in expected.items()),'Fresh output manifest changed')
    require(summary['complete'] and summary['samples']==256 and summary['probes']==0 and summary['manifest']==manifest,
        'Incomplete fresh population')
    for name in ('samples','probes','attempts'):
        require(sha(directory/(name+'.jsonl'))==summary[name+'_sha256'],'Output changed: '+name)
    require((directory/'probes.jsonl').read_bytes()==b'','Unexpected saved-query scoring')
    records=read_rows(directory/'samples.jsonl');attempts=read_rows(directory/'attempts.jsonl')
    require(len(records)==len(attempts)==256,'Missing attempted draw')
    for i,(r,a) in enumerate(zip(records,attempts)):
        require(type(r['id']) is int and r['id']==i and r['kind']=='fresh' and r['draw'] is not None,'Draw identity changed')
        require(a==dict(ordinal=i,kind='fresh',id=i,state='begin'),'Attempt journal changed')
    return summary


def run(out,expected):
    out=Path(out).resolve();p=validate(out,expected)
    require(not (out/'status.json').exists(),'No repeated execution')
    with (out/'launch-claim.json').open('x') as stream:json.dump(dict(protocol_sha256=expected),stream)
    state=dict(complete=False,phase='starting',protocol_sha256=expected,steps=p['steps'],started=time.time())
    def snapshot():
        (out/'status.tmp').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');(out/'status.tmp').replace(out/'status.json')
    snapshot()
    def interrupted(signum,frame):raise InterruptedError('Fresh pilot signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        for step in state['steps']:
            state['phase']=step['id'];snapshot();before=resource.getrusage(resource.RUSAGE_CHILDREN)
            execute_group([step],snapshot,1,out/'common',worker_environment(),before_launch=lambda:verify_frozen(out))
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            step['child_CPU_seconds']=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
            directory=Path(step['population_directory'])
            if step['kind']=='proposal':verify_output(out,p,step)
            elif step['kind']=='audit':
                receipt=read(directory/'independent-audit.json');require(receipt['complete'] and receipt['probes']==0,'Independent audit incomplete')
            else:
                receipt=read(directory/'observer/summary.json')
                require(receipt['complete'] and receipt['attempted']==256 and receipt['native_classifier_calls']==receipt['valid'],
                    'Native/contact observer incomplete')
            snapshot()
        state['phase']='report';snapshot();analysis=report(out)
        analysis['child_CPU_seconds']={arm:{kind:sum(s['child_CPU_seconds'] for s in state['steps'] if s['arm']==arm and s['kind']==kind)
            for kind in ['proposal','audit','observer']} for arm in ARMS}
        write(out/'analysis.json',analysis);verify_frozen(out)
        state.update(complete=True,phase='complete',finished=time.time(),analysis_sha256=sha(out/'analysis.json'));snapshot()
    except BaseException as error:
        state.update(complete=False,error=repr(error),finished=time.time())
        for step in state['steps']:
            if step['status']=='pending':step['status']='not_started'
        snapshot();raise
    finally:signal.signal(signal.SIGTERM,previous)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze')
    for name in ('out','preparation','archived-score'):f.add_argument('--'+name,type=Path,required=True)
    f.add_argument('--reference',type=Path,action='append',required=True)
    for name in ('preflight','run'):
        a=sub.add_parser(name);a.add_argument('--out',type=Path,required=True);a.add_argument('--expected-protocol-sha256',required=True)
    a=p.parse_args()
    if a.action=='freeze':print(freeze(a.out,a.preparation,a.archived_score,a.reference))
    elif a.action=='preflight':validate(a.out,a.expected_protocol_sha256);print('preflight passed')
    else:print(run(a.out,a.expected_protocol_sha256)['complete'])
