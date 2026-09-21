#!/usr/bin/env python3
"""Run the frozen full-D170 reciprocal normalizer once; drain before audit/failure."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from prepare_mobile_full_capture_campaign import (
    ANALYZER_NAME,BINARY_SHA,BUNDLE_SHA,CLOUDS,LAMBDA_RATIO,REPLICATES,SAMPLES,WORKERS,
    read,require,sha,validate,write,
)
from run_mobile_posterior_pilot import exclusive_write,inside


def check(campaign,journal):
    campaign,journal=Path(campaign).resolve(),Path(journal).resolve()
    require(not inside(campaign,journal) and not inside(journal,campaign),'Campaign and journal must be separate')
    protocol=validate(campaign)
    require(sha(__file__)==protocol['controller_sha256'],'Use the frozen controller')
    require(not journal.exists() and not (campaign/'status.json').exists(),'No retry, resume or overwrite')
    for entry in protocol['campaigns']:
        folder=Path(entry['path'])
        require(not (folder/'status.json').exists() and not (folder/'assessment').exists(),'Existing campaign status/audit')
        require(not any((folder/'runs').iterdir()) and not any((folder/'logs').iterdir()),'Existing population outputs/logs')
        require(not list(folder.glob('r??-status.json')),'Existing population terminal status')
    return protocol


def verify_population(folder,manifest,job):
    directory=Path(job['directory']);summary=read(directory/'summary.json');population=read(directory/'manifest.json')
    require(summary['complete'] is True and summary['samples']==SAMPLES and summary['numerical_nulls']==0,'Incomplete/null physical population')
    require(summary['manifest']==population,'Summary and population manifests differ')
    wanted=dict(schema=3,samples=SAMPLES,seed=job['seed'],cloud_replicates=CLOUDS,activity=.035,lambda_value=.035*64.,
        config_sha256=manifest['config_sha256'],model_sha256=manifest['model_sha256'],shape_sha256=manifest['shape_sha256'],
        source_bundle_sha256=BUNDLE_SHA,executable_sha256=BINARY_SHA,uniform_probability=manifest['uniform_probability'],
        proposal_model_kind='reciprocal-pose-mixture-v1',base_component_count=150,virtual_component_count=300,
        reciprocal_components=[True]*150,covariance_scale=1.,proposal_anchor_index=None,physical_fixed_neighbor_count=2)
    wanted['lambda']=wanted.pop('lambda_value')
    require(all(population.get(k)==v for k,v in wanted.items()),'Population law/input binding differs')
    for name,field in [('input-config.json','config_sha256'),('model.json','model_sha256'),('shape.json','shape_sha256'),
                       ('source-bundle.json','source_bundle_sha256')]:
        require(sha(directory/'provenance'/name)==population[field],'Actual population input differs: '+name)
    require(summary['estimates']['total']['unconditional_draws']==SAMPLES,'Population conditioned away invalid draws')
    effective=read(directory/'config.json');submitted=read(folder/'provenance/config.json')
    for key in ('fixed_poses','capture_center','capture_radius','depletant_radius','reservoir_density','poisson_lambda_ratio','metadata'):
        require(effective[key]==submitted[key],'Effective physical target changed: '+key)
    return dict(summary_sha256=sha(directory/'summary.json'),manifest_sha256=sha(directory/'manifest.json'),
        samples_sha256=sha(directory/'samples.jsonl'),sampler_cpu_seconds=summary['sampler_cpu_seconds'])


def verify_assessment(folder,manifest,analysis,records):
    require(not analysis['pending'],'Audit left a population pending')
    require(set(analysis['groups'])=={'1.0'},'Unexpected covariance-scale group')
    group=analysis['groups']['1.0']
    require(group['unconditional_draws']==REPLICATES*SAMPLES and group['populations']==REPLICATES,'Audit discarded unconditional draws')
    require(group['model_sha256']==manifest['model_sha256'] and group['config_sha256']==manifest['config_sha256'],'Audit model/config differs')
    populations=analysis['populations'];jobs=manifest['jobs']
    require(len(populations)==4 and {p['job']['id'] for p in populations}=={j['id'] for j in jobs},'Audit omitted a population')
    by_id={p['job']['id']:p for p in populations};terminal={r['id']:r for r in records}
    for job in jobs:
        result=by_id[job['id']]
        require(result['job']==job and result['estimates']['total']['draws']==SAMPLES,'Audit population allocation differs')
        audit=result['proposal_audit']
        require(audit['checked_actual_poses']==SAMPLES and audit['proposal_anchor_indices']==[0,1]
            and audit['reciprocal_components']==[True]*150 and audit['virtual_component_count']==300,'Audit proposal law differs')
        for name,key in [('summary.json','summary_sha256'),('manifest.json','manifest_sha256'),('samples.jsonl','samples_sha256')]:
            path=Path(job['directory'])/name
            require(analysis['provenance'][str(path)]==terminal[job['id']]['output'][key]==sha(path),'Audit physical output binding differs')
    require(analysis['provenance'][str(folder/'manifest.json')]==sha(folder/'manifest.json'),'Audit campaign binding differs')
    auditor=folder/'provenance'/ANALYZER_NAME
    require(analysis['provenance'][str(auditor)]==manifest['observer_sha256']==sha(auditor),'Audit source binding differs')


def run(campaign,journal):
    campaign,journal=Path(campaign).resolve(),Path(journal).resolve()
    protocol=check(campaign,journal)
    journal.mkdir(parents=True);shutil.copy2(__file__,journal/'runner.py')
    state=dict(schema='mobile-full-capture-status-v1',complete=False,phase='physical',
        started=time.time(),protocol_sha256=sha(campaign/'protocol.json'),jobs=[],audits={})
    exclusive_write(campaign/'status.json',state)
    entries={e['arm']:e for e in protocol['campaigns']}
    manifests={arm:read(Path(e['path'])/'manifest.json') for arm,e in entries.items()}
    statuses={arm:dict(running=True,complete=False,jobs=[dict(id=j['id'],seed=j['seed'],samples=j['samples'],status='pending',
        exit_code=None) for j in manifest['jobs']]) for arm,manifest in manifests.items()}
    reserved=[]
    try:
        for arm,status in statuses.items():
            exclusive_write(Path(entries[arm]['path'])/'status.json',status);reserved.append(arm)
    except BaseException as error:
        for arm in reserved:
            statuses[arm].update(running=False,complete=False)
            for record in statuses[arm]['jobs']:record.update(status='not_started',reason='Reservation failed; no child launched')
            write(Path(entries[arm]['path'])/'status.json',statuses[arm])
        state.update(phase='reservation_failed',exception=repr(error),finished=time.time())
        write(campaign/'status.json',state);write(journal/'status.json',state);raise
    jobs=[(arm,index,job) for arm,m in manifests.items() for index,job in enumerate(m['jobs'])]
    active={};next_index=0;failed=False;exception=None
    def snapshot():
        state['jobs']=[dict(arm=arm,**r) for arm,status in statuses.items() for r in status['jobs']]
        write(campaign/'status.json',state);write(journal/'status.json',state)
        for arm,status in statuses.items():write(Path(entries[arm]['path'])/'status.json',status)
    def finish(key,child):
        arm,index=key;code=child.wait();record=statuses[arm]['jobs'][index]
        record.update(status='complete' if code==0 else 'failed',exit_code=code,finished=time.time())
        write(Path(entries[arm]['path'])/(record['id']+'-status.json'),
            dict(id=record['id'],returncode=code,completed_unix=record['finished'],seed=record['seed'],samples=record['samples']))
        print('Finished',arm,record['id'],'exit',code,flush=True)
        return code
    try:
        while active or (next_index<len(jobs) and not failed):
            while not failed and next_index<len(jobs) and len(active)<WORKERS:
                arm,index,job=jobs[next_index];next_index+=1
                folder=Path(entries[arm]['path']);manifest=manifests[arm]
                require(sha(folder/'manifest.json')==entries[arm]['manifest_sha256'],'Manifest changed at launch')
                for name in ('basin-normalizer','config.json','model.json','shape.json'):
                    require(sha(folder/'provenance'/name)==manifest['archive_sha256'][name],'Input changed at launch: '+name)
                require(not Path(job['directory']).exists(),'Population output appeared before launch')
                record=statuses[arm]['jobs'][index];record.update(status='launching',started=time.time(),argv=job['command'],log=job['log'])
                snapshot()
                try:
                    with Path(job['log']).open('xb') as log:child=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
                except BaseException as error:
                    record.update(status='launch_failed',exception=repr(error),finished=time.time());raise
                active[arm,index]=child;record.update(status='running',pid=child.pid);snapshot()
                print('Started',arm,job['id'],'PID',child.pid,flush=True)
            for key,child in list(active.items()):
                if child.poll() is None:continue
                failed=finish(key,child)!=0 or failed;del active[key];snapshot()
            if active:time.sleep(.5)
    except BaseException as error:
        failed,exception=True,error;state['exception']=repr(error)
    finally:
        for key,child in active.items():failed=finish(key,child)!=0 or failed
        for status in statuses.values():
            for record in status['jobs']:
                if record['status']=='pending':record.update(status='not_started',reason='Stopped after failure; no retry')
            status.update(running=False,complete=all(r['status']=='complete' and r['exit_code']==0 for r in status['jobs']))
        snapshot()
    if failed or not all(s['complete'] for s in statuses.values()):
        state.update(phase='physical_failed',complete=False,finished=time.time());snapshot()
        if exception is not None:raise exception
        raise RuntimeError('A physical job failed; all started children drained, no retry or audit')
    try:
        state['phase']='physical_validation';snapshot();validate(campaign)
        for arm,index,job in jobs:
            statuses[arm]['jobs'][index]['output']=verify_population(Path(entries[arm]['path']),manifests[arm],job)
        snapshot()
    except BaseException as error:
        state.update(phase='physical_validation_failed',complete=False,exception=repr(error),finished=time.time());snapshot();raise
    try:
        state['phase']='audit';snapshot()
        for arm,manifest in manifests.items():
            folder=Path(entries[arm]['path'])
            require(not (folder/'assessment').exists(),'Audit output already exists; no retries')
            require(sha(folder/'provenance'/ANALYZER_NAME)==manifest['observer_sha256'],'Frozen auditor changed')
            argv=[sys.executable,'-B',str(folder/'provenance'/ANALYZER_NAME),'--root',str(folder)]
            state['audits'][arm]=dict(argv=argv,started=time.time(),manifest_sha256=sha(folder/'manifest.json'));snapshot()
            with (journal/(arm+'-audit.log')).open('xb') as log:
                result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            state['audits'][arm].update(returncode=result.returncode,finished=time.time());snapshot();result.check_returncode()
            analysis=read(folder/'assessment/analysis.json')
            verify_assessment(folder,manifest,analysis,statuses[arm]['jobs'])
            state['audits'][arm]['analysis_sha256']=sha(folder/'assessment/analysis.json');snapshot()
    except BaseException as error:
        state.update(phase='audit_failed',complete=False,exception=repr(error),finished=time.time());snapshot();raise
    state.update(phase='complete',complete=True,finished=time.time());snapshot()
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('preflight','run'))
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--journal',type=Path,required=True)
    args=parser.parse_args()
    result=check(args.campaign,args.journal) if args.action=='preflight' else run(args.campaign,args.journal)
    print(__import__('json').dumps(dict(action=args.action,complete=result.get('complete'),
        physical_launches=0 if args.action=='preflight' else len(result['jobs'])),indent=2))


if __name__=='__main__':main()
