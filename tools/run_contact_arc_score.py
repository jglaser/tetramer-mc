#!/usr/bin/env python3
"""Freeze and execute one Rust score job plus independent Python reconstruction."""
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

from prepare_contact_arc_score import ARMS, MINIMUM_ARC_MASS, REGION_SHA, SHAPE_SHA, read, rows, sha, require, write
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_confirmation import worker_environment, runtime
from run_contact_tail_pilot import verify_frozen, copy_frozen
from run_full_vessel_comparison import execute_group
from run_mobile_posterior_pilot import verify_bundle
from report_contact_arc_score import report

SCHEMA='contact-arc-score-controller-v1'


def validate_design(plan):
    expected=dict(arms=ARMS,unique_saved_queries=206,candidate_density_evaluations=412,
        critical_queries=78,critical_original_sources={'baseline':4,'expanded':74},breadth_queries=128,
        breadth_classes={'native_R5':32,'native_complement':32,'competing':32,'invalid':32},
        minimum_arc_mass=MINIMUM_ARC_MASS,defensive_uniform_probability=.5,conditional_probability=.5,
        widths_A=[.02,.1,.5],maximum_CPU_workers=1,new_pose_draws=0,new_Poisson_clouds=0,
        guides_unchanged=True,labels_unchanged=True,all_components_and_widths_in_density=True,
        no_optional_stopping=True,no_retries=True,no_autoextension=True)
    require(all(plan.get(k)==v for k,v in expected.items()),'Fixed score-only allocation changed')


def make_jobs(out, common):
    base=['--config',str(common/'config.json'),'--region',str(common/'region.json'),
          '--probes',str(common/'score-probes.jsonl'),'--minimum-arc-mass','1e-12']
    rust=[str(common/'contact-arc-density-audit'),*base]
    python=[sys.executable,'-B',str(common/'contact_arc_density_reference.py'),
            '--shape',str(common/'shape.json'),*base]
    for arm in ARMS:
        rust+=['--importance-guide',str(common/(arm+'.json'))]
        python+=['--guide',str(common/(arm+'.json'))]
    rust+=['--out',str(out/'rust')];python+=['--out',str(out/'python.json')]
    return [dict(id=name,command=cmd,kind='geometry',status='pending',log=str(out/(name+'.log')),
        directory=str(out/('rust' if name=='rust' else 'python.json')))
        for name,cmd in [('rust',rust),('python',python)]]


def freeze(out, preparation, binary, bundle, repository, references):
    out,preparation,binary,bundle,repository=map(lambda p:Path(p).resolve(),(out,preparation,binary,bundle,repository))
    require(not out.exists(),'Fresh score execution directory required')
    verify_frozen(preparation);plan=read(preparation/'plan.json');validate_design(plan)
    source,rust_hashes=verify_bundle(binary,bundle,repository)
    from contact_arc_density_reference import audit  # source closure; no execution while freezing
    dependencies=local_dependencies([Path(__file__)])
    references=[Path(p).resolve() for p in references]
    require(references and all(p.is_file() for p in references),'Reviewed production-validation receipts required')
    out.mkdir();common=out/'common';common.mkdir()
    copy_frozen(preparation,common/'preparation')
    for name,path in dependencies.items():shutil.copy2(path,common/name)
    shutil.copy2(binary,common/'contact-arc-density-audit');shutil.copy2(bundle,common/'source-bundle.json')
    for name,entry in source['files'].items():
        target=common/'rust-source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(entry['text'])
    for name in ('shape.json','region.json'):shutil.copy2(preparation/'common'/name,common/name)
    cfg=read(preparation/'common/config.json');cfg['shape']=str(common/'shape.json');write(common/'config.json',cfg)
    shutil.copy2(preparation/'score-probes.jsonl',common/'score-probes.jsonl')
    for arm in ARMS:shutil.copy2(preparation/'guides'/f'{arm}.json',common/f'{arm}.json')
    receipts=[]
    for i,p in enumerate(references):
        target=common/f'reference-{i:02}-{p.name}';shutil.copy2(p,target)
        receipts.append(dict(original=str(p),sha256=sha(p),archive=target.name))
    protocol=dict(plan,schema=SCHEMA,preparation=str(preparation),preparation_plan_sha256=sha(preparation/'plan.json'),
        preparation_freeze_sha256=sha(preparation/'freeze.json'),binary_sha256=sha(binary),
        source_bundle_sha256=sha(bundle),rust_sources=rust_hashes,reference_receipts=receipts,
        python_sources={name:sha(path) for name,path in dependencies.items()},controller_sha256=sha(__file__),
        runtime=runtime(),repository=str(repository),jobs=make_jobs(out,common))
    # Preserve the immutable earlier plan, but name this scientific gate
    # correctly in new records: autonomous work already has user authorization.
    protocol.pop('physical_campaign_authorized',None)
    protocol['physical_campaign_ready']=False
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    validate(out,sha(out/'protocol.json'))
    return sha(out/'protocol.json')


def validate(out,expected):
    out=Path(out).resolve();verify_frozen(out)
    require(sha(out/'protocol.json')==expected,'Protocol hash mismatch')
    p=read(out/'protocol.json');validate_design(p)
    require(p['schema']==SCHEMA and runtime()==p['runtime'] and sys.flags.optimize==0,'Controller runtime changed')
    require(sha(__file__)==p['controller_sha256'] and
        {n:sha(v) for n,v in local_dependencies([Path(__file__)]).items()}==p['python_sources'],
        'Controller or source closure changed')
    common=out/'common';verify_bundle(common/'contact-arc-density-audit',common/'source-bundle.json',common/'rust-source')
    require(sha(common/'contact-arc-density-audit')==p['binary_sha256'],'Executable changed')
    prep=Path(p['preparation']);verify_frozen(prep)
    require(sha(prep/'plan.json')==p['preparation_plan_sha256'] and sha(prep/'freeze.json')==p['preparation_freeze_sha256'],
        'Saved score preparation changed')
    require(p['jobs']==make_jobs(out,common),'Exact sequential score commands changed')
    return p


def verify_output(out,p):
    common=out/'common';directory=out/'rust'
    manifest=read(directory/'manifest.json');summary=read(directory/'summary.json')
    expected=dict(schema='contact-arc-density-score-v1',queries=206,arms=2,minimum_arc_mass=MINIMUM_ARC_MASS,
        config_sha256=sha(common/'config.json'),region_sha256=REGION_SHA,shape_sha256=SHAPE_SHA,
        probes_sha256=sha(common/'score-probes.jsonl'),guide_sha256=[sha(common/(a+'.json')) for a in ARMS],
        executable_sha256=p['binary_sha256'],source_bundle_sha256=p['source_bundle_sha256'],new_pose_draws=0,new_Poisson_clouds=0)
    require(all(manifest.get(k)==v for k,v in expected.items()),'Rust score manifest differs from frozen inputs')
    require(summary['complete'] and summary['manifest']==manifest and summary['queries']==206 and summary['arms']==2,
            'Rust score output incomplete')
    for name in ('scores','attempts'):
        require(sha(directory/(name+'.jsonl'))==summary[name+'_sha256'],'Score/journal changed')
    probes=rows(common/'score-probes.jsonl');scored=rows(directory/'scores.jsonl');journal=rows(directory/'attempts.jsonl')
    require(len(scored)==len(journal)==206,'Attempted-query count changed')
    for i,(probe,row,attempt) in enumerate(zip(probes,scored,journal)):
        require(row['ordinal']==i and row['id']==probe['id'] and row['latent']==probe['latent'] and
            attempt==dict(ordinal=i,id=probe['id'],state='begin'),'Query identity/order/journal differs')
        require(len(row['arms'])==2 and [a['arm_index'] for a in row['arms']]==[0,1],'Guide order differs')
    return summary


def run(out,expected):
    out=Path(out).resolve();p=validate(out,expected)
    require(not (out/'status.json').exists(),'No repeated score launch')
    with (out/'launch-claim.json').open('x') as stream:json.dump(dict(protocol_sha256=expected),stream)
    state=dict(complete=False,protocol_sha256=expected,jobs=p['jobs'],started=time.time())
    def snapshot():
        (out/'status.tmp').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
        (out/'status.tmp').replace(out/'status.json')
    snapshot()
    def interrupted(signum,frame):raise InterruptedError('Arc-score controller signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        for job in state['jobs']:
            before=resource.getrusage(resource.RUSAGE_CHILDREN)
            execute_group([job],snapshot,1,Path(p['repository']),worker_environment(),before_launch=lambda:verify_frozen(out))
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            job['child_CPU_seconds']=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime;snapshot()
        verify_output(out,p)
        from contact_arc_density_reference import audit
        receipt=audit(out/'rust',out/'python.json')
        require(receipt['complete'],'Independent arc density audit failed')
        write(out/'independent-audit.json',receipt)
        review=report(Path(p['preparation']),out);write(out/'analysis.json',review)
        verify_frozen(out)
        state.update(complete=True,finished=time.time(),output_sha256={name:sha(out/name) for name in
            ['rust/scores.jsonl','rust/attempts.jsonl','rust/summary.json','python.json','independent-audit.json','analysis.json']})
        snapshot()
    except BaseException as error:
        state.update(complete=False,error=repr(error),finished=time.time())
        for job in state['jobs']:
            if job['status']=='pending':job['status']='not_started'
        snapshot();raise
    finally:signal.signal(signal.SIGTERM,previous)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze')
    for name in ('out','preparation','binary','bundle','repository'):f.add_argument('--'+name,type=Path,required=True)
    f.add_argument('--reference',type=Path,action='append',required=True)
    for action in ('preflight','run'):
        p=sub.add_parser(action);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-protocol-sha256',required=True)
    a=parser.parse_args()
    if a.action=='freeze':print(freeze(a.out,a.preparation,a.binary,a.bundle,a.repository,a.reference))
    elif a.action=='preflight':validate(a.out,a.expected_protocol_sha256);print('preflight passed')
    else:print(run(a.out,a.expected_protocol_sha256)['complete'])
