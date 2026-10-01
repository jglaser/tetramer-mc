#!/usr/bin/env python3
"""Freeze one no-draw xyz scoring job, then audit and derive its axis laws."""
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
            'VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
from prepare_hard_free_line_score import ARMS,PROBES_SHA,REGION_SHA,SHAPE_SHA,read,rows,sha,write,require
from report_hard_free_line_score import report
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_confirmation import worker_environment,runtime
from run_contact_tail_pilot import verify_frozen,copy_frozen
from run_full_vessel_comparison import execute_group
from run_mobile_posterior_pilot import verify_bundle

SCHEMA='hard-free-line-score-controller-v1'


def validate_design(plan):
    expected=dict(arms=ARMS,primary_arm='xyz',secondary_arms=['x','y','z'],scored_guide='guides/xyz.json',
        queries=206,full_mixture_density_evaluations=206,derived_axis_densities=618,total_reported_candidate_densities=824,
        declared_axes_per_query=[0,1,2],axis_geometry_requests=618,Gaussian_component_axis_branches=56856,
        critical_queries=78,critical_original_sources={'baseline':4,'expanded':74},breadth_queries=128,
        breadth_classes={'native_R5':32,'native_complement':32,'competing':32,'invalid':32},
        alpha=.5,beta=1.,minimum_conditional_mass=1e-12,contact_widths=None,contact_labels=None,
        new_pose_draws=0,new_Poisson_clouds=0,new_physical_mass_estimates=0,maximum_CPU_workers=1,
        no_optional_stopping=True,no_retries=True,no_autoextension=True,physical_campaign_ready=False)
    require(all(plan.get(k)==v for k,v in expected.items()),'Fixed feasibility score allocation changed')


def job(out):
    common=out/'common'
    return dict(id='xyz',kind='geometry',status='pending',directory=str(out/'scored'),log=str(out/'score.log'),
        command=[str(common/'contact-line-guide-audit'),'--config',str(common/'config.json'),
            '--region',str(common/'region.json'),'--importance-guide',str(common/'xyz.json'),
            '--probes',str(common/'score-probes.jsonl'),'--samples','0','--seed','0','--out',str(out/'scored')])


def freeze(out,preparation,binary,bundle,repository,references):
    out,preparation,binary,bundle,repository=map(lambda p:Path(p).resolve(),(out,preparation,binary,bundle,repository))
    require(not out.exists(),'Fresh score execution required');verify_frozen(preparation)
    plan=read(preparation/'plan.json');validate_design(plan)
    require(sha(preparation/'score-probes.jsonl')==PROBES_SHA,'Saved probe allocation changed')
    source,rust_hashes=verify_bundle(binary,bundle,repository)
    from hard_free_line_reference import audit  # archive dependency closure; no scoring here
    dependencies=local_dependencies([Path(__file__)])
    references=[Path(p).resolve() for p in references]
    require(references and all(p.is_file() for p in references),'Reviewed reference receipts required')
    for p in references:
        receipt=read(p)
        require(receipt.get('complete') is True and receipt.get('passed',True) is True,
                'Incomplete or failed validation receipt: '+str(p))
    out.mkdir();common=out/'common';common.mkdir()
    copy_frozen(preparation,common/'preparation')
    for name,path in dependencies.items():shutil.copy2(path,common/name)
    shutil.copy2(binary,common/'contact-line-guide-audit');shutil.copy2(bundle,common/'source-bundle.json')
    for name,entry in source['files'].items():
        path=common/'rust-source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(entry['text'])
    for name in ('shape.json','region.json'):shutil.copy2(preparation/'common'/name,common/name)
    config=read(preparation/'common/config.json');config['shape']=str(common/'shape.json');write(common/'config.json',config)
    shutil.copy2(preparation/'score-probes.jsonl',common/'score-probes.jsonl')
    shutil.copy2(preparation/'guides/xyz.json',common/'xyz.json')
    archived=[]
    for i,path in enumerate(references):
        target=common/f'reference-{i:02}-{path.name}';shutil.copy2(path,target)
        archived.append(dict(original=str(path),archive=target.name,sha256=sha(path)))
    protocol=dict(plan,schema=SCHEMA,preparation=str(preparation),preparation_plan_sha256=sha(preparation/'plan.json'),
        preparation_freeze_sha256=sha(preparation/'freeze.json'),binary_sha256=sha(binary),
        source_bundle_sha256=sha(bundle),rust_sources=rust_hashes,reference_receipts=archived,
        python_sources={name:sha(path) for name,path in dependencies.items()},controller_sha256=sha(__file__),
        runtime=runtime(),repository=str(repository),jobs=[job(out)],execution_ready=True,
        unused_cli_seed=0,samples=0,scope='One fixed saved-query score job, independent audit, algebraic report; zero pose or Poisson draws.')
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    validate(out,sha(out/'protocol.json'));return sha(out/'protocol.json')


def validate(out,expected):
    out=Path(out).resolve();verify_frozen(out);require(sha(out/'protocol.json')==expected,'Protocol hash changed')
    p=read(out/'protocol.json');validate_design(p)
    require(p['schema']==SCHEMA and runtime()==p['runtime'] and sys.flags.optimize==0,'Controller runtime changed')
    require(sha(__file__)==p['controller_sha256'] and
        {n:sha(v) for n,v in local_dependencies([Path(__file__)]).items()}==p['python_sources'],
        'Controller/reference source closure changed')
    common=out/'common';verify_bundle(common/'contact-line-guide-audit',common/'source-bundle.json',common/'rust-source')
    require(sha(common/'contact-line-guide-audit')==p['binary_sha256'],'Frozen executable changed')
    prep=Path(p['preparation']);verify_frozen(prep)
    require(sha(prep/'plan.json')==p['preparation_plan_sha256'] and sha(prep/'freeze.json')==p['preparation_freeze_sha256'],
        'Preparation binding changed')
    require(p['jobs']==[job(out)] and p['unused_cli_seed']==0 and p['samples']==0,'Exact score command changed')
    return p


def verify_output(out,p):
    directory=out/'scored';common=out/'common';manifest=read(directory/'manifest.json');summary=read(directory/'summary.json')
    expected=dict(schema='hard-free-line-guide-audit-v1',samples=0,seed=0,physical_jobs=0,
        executable_sha256=p['binary_sha256'],source_bundle_sha256=p['source_bundle_sha256'],
        config_sha256=sha(common/'config.json'),region_sha256=REGION_SHA,shape_sha256=SHAPE_SHA,
        guide_sha256=sha(common/'xyz.json'),probes_sha256=PROBES_SHA)
    require(all(manifest.get(k)==v for k,v in expected.items()),'Output manifest differs from frozen score inputs')
    require(summary['complete'] and summary['samples']==0 and summary['probes']==206 and summary['manifest']==manifest,
        'Output score allocation incomplete')
    for name in ('samples','probes','attempts'):
        require(sha(directory/(name+'.jsonl'))==summary[name+'_sha256'],'Saved output changed: '+name)
    require((directory/'samples.jsonl').read_bytes()==b'','Score-only job drew a new pose')
    scored=rows(directory/'probes.jsonl');attempts=rows(directory/'attempts.jsonl');probes=rows(common/'score-probes.jsonl')
    require(len(scored)==len(attempts)==len(probes)==206,'Missing attempted query')
    for i,(r,a,b) in enumerate(zip(scored,attempts,probes)):
        require(r['id']==b['id'] and r['latent']==b['latent'] and r['kind']=='probe' and r['draw'] is None,
                'Saved query changed or resampled')
        require(a==dict(ordinal=i,kind='probe',id=b['id'],state='begin'),'Attempt journal changed')
    return summary


def run(out,expected):
    out=Path(out).resolve();p=validate(out,expected)
    require(not (out/'status.json').exists(),'No repeated execution')
    with (out/'launch-claim.json').open('x') as stream:json.dump(dict(protocol_sha256=expected),stream)
    state=dict(complete=False,phase='scoring',protocol_sha256=expected,jobs=p['jobs'],started=time.time())
    def snapshot():
        (out/'status.tmp').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');(out/'status.tmp').replace(out/'status.json')
    snapshot()
    def interrupted(signum,frame):raise InterruptedError('Feasibility score signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        before=resource.getrusage(resource.RUSAGE_CHILDREN)
        execute_group(state['jobs'],snapshot,1,Path(p['repository']),worker_environment(),before_launch=lambda:verify_frozen(out))
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        state['jobs'][0]['child_CPU_seconds']=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        verify_output(out,p);state['phase']='independent_audit';snapshot()
        from hard_free_line_reference import audit
        receipt=audit(out/'scored');require(receipt['complete'],'Independent feasibility audit failed')
        write(out/'independent-audit.json',receipt);state['phase']='report';snapshot()
        result=report(Path(p['preparation']),out/'scored',out/'independent-audit.json');write(out/'analysis.json',result)
        verify_frozen(out)
        state.update(complete=True,phase='complete',finished=time.time(),output_sha256={name:sha(out/name) for name in
            ['scored/probes.jsonl','scored/samples.jsonl','scored/attempts.jsonl','scored/summary.json','independent-audit.json','analysis.json']})
        snapshot()
    except BaseException as error:
        state.update(complete=False,error=repr(error),finished=time.time())
        for j in state['jobs']:
            if j['status']=='pending':j['status']='not_started'
        snapshot();raise
    finally:signal.signal(signal.SIGTERM,previous)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze')
    for name in ('out','preparation','binary','bundle','repository'):f.add_argument('--'+name,type=Path,required=True)
    f.add_argument('--reference',type=Path,action='append',required=True)
    for name in ('preflight','run'):
        a=sub.add_parser(name);a.add_argument('--out',type=Path,required=True);a.add_argument('--expected-protocol-sha256',required=True)
    a=parser.parse_args()
    if a.action=='freeze':print(freeze(a.out,a.preparation,a.binary,a.bundle,a.repository,a.reference))
    elif a.action=='preflight':validate(a.out,a.expected_protocol_sha256);print('preflight passed')
    else:print(run(a.out,a.expected_protocol_sha256)['complete'])
