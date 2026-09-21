#!/usr/bin/env python3
"""Run the frozen eight-population shell control once; drain before audit/failure."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from prepare_mobile_competing_importance_campaign import (
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
    require(summary['complete'] is True and summary['samples']==SAMPLES,'Incomplete physical population')
    require(summary['manifest']==population,'Summary and population manifests differ')
    wanted=dict(schema='importance-latent-region-normalizer-v1',samples=SAMPLES,seed=job['seed'],
        cloud_replicates=CLOUDS,activity=manifest['physical_activity'],lambda_ratio=LAMBDA_RATIO,
        config_sha256=manifest['config_sha256'],region_sha256=manifest['region_sha256'],
        shape_sha256=manifest['shape_sha256'],source_bundle_sha256=BUNDLE_SHA,executable_sha256=BINARY_SHA,
        importance_guide_sha256=manifest['importance_guide_sha256'],importance_uniform_probability=.5,importance_component_count=32)
    require(all(population.get(k)==v for k,v in wanted.items()),'Population law/input binding differs')
    for name,field in [('input-config.json','config_sha256'),('region.json','region_sha256'),('shape.json','shape_sha256'),
                       ('source-bundle.json','source_bundle_sha256'),('importance-guide.json','importance_guide_sha256')]:
        require(sha(directory/'provenance'/name)==population[field],'Actual population input differs: '+name)
    sample_hash=sha(directory/'samples.jsonl')
    require(summary['samples_sha256']==sample_hash,'Population sample stream changed')
    require(summary['estimates']['region']['draws']==summary['estimates']['hard_region']['draws']==SAMPLES,
        'Population denominator differs from unconditional draw count')
    return dict(summary_sha256=sha(directory/'summary.json'),manifest_sha256=sha(directory/'manifest.json'),
        samples_sha256=sample_hash,sampler_cpu_seconds=summary['sampler_cpu_seconds'])


def verify_assessment(folder,manifest,analysis,records):
    require(analysis['region_sha256']==manifest['region_sha256'],'Audit region differs')
    require(analysis['estimate']['draws']==analysis['hard_region']['draws']==REPLICATES*SAMPLES,'Audit discarded unconditional draws')
    importance=analysis['importance_sampling']
    require(importance['guide_sha256']==manifest['importance_guide_sha256'] and importance['uniform_shell_probability']==.5
        and importance['gaussian_component_count']==32 and importance['draws']==REPLICATES*SAMPLES,'Audit guide law differs')
    populations=analysis['populations'];jobs=manifest['jobs']
    require(len(populations)==4 and {p['id'] for p in populations}=={j['id'] for j in jobs},'Audit omitted a population')
    by_id={p['id']:p for p in populations};terminal={r['id']:r for r in records}
    for job in jobs:
        result=by_id[job['id']]
        require(result['seed']==job['seed'] and result['estimate']['draws']==result['hard_region']['draws']==SAMPLES,'Audit population/seed differs')
        require(result['samples_sha256']==terminal[job['id']]['output']['samples_sha256']==sha(Path(job['directory'])/'samples.jsonl'),
            'Audit sample binding differs')
    require(analysis['physical_fixed_neighbors']==read(folder/'provenance/config.json')['fixed_poses'],'Audit scaffold differs')


def run(campaign,journal):
    campaign,journal=Path(campaign).resolve(),Path(journal).resolve()
    protocol=check(campaign,journal)
    journal.mkdir(parents=True);shutil.copy2(__file__,journal/'runner.py')
    state=dict(schema='mobile-competing-guided-outer-status-v1',complete=False,phase='physical',
        started=time.time(),protocol_sha256=sha(campaign/'protocol.json'),jobs=[],audits={})
    exclusive_write(campaign/'status.json',state)
    entries={e['region']:e for e in protocol['campaigns']}
    manifests={region:read(Path(e['path'])/'manifest.json') for region,e in entries.items()}
    statuses={region:dict(running=True,complete=False,jobs=[dict(id=j['id'],seed=j['seed'],samples=j['samples'],status='pending',
        exit_code=None) for j in manifest['jobs']]) for region,manifest in manifests.items()}
    reserved=[]
    try:
        for region,status in statuses.items():
            exclusive_write(Path(entries[region]['path'])/'status.json',status);reserved.append(region)
    except BaseException as error:
        for region in reserved:
            statuses[region].update(running=False,complete=False)
            for record in statuses[region]['jobs']:record.update(status='not_started',reason='Reservation failed; no child launched')
            write(Path(entries[region]['path'])/'status.json',statuses[region])
        state.update(phase='reservation_failed',exception=repr(error),finished=time.time())
        write(campaign/'status.json',state);write(journal/'status.json',state);raise
    jobs=[(region,index,job) for region,m in manifests.items() for index,job in enumerate(m['jobs'])]
    active={};next_index=0;failed=False;exception=None
    def snapshot():
        state['jobs']=[dict(region=region,**r) for region,status in statuses.items() for r in status['jobs']]
        write(campaign/'status.json',state);write(journal/'status.json',state)
        for region,status in statuses.items():write(Path(entries[region]['path'])/'status.json',status)
    def finish(key,child):
        region,index=key;code=child.wait();record=statuses[region]['jobs'][index]
        record.update(status='complete' if code==0 else 'failed',exit_code=code,finished=time.time())
        write(Path(entries[region]['path'])/(record['id']+'-status.json'),
            dict(id=record['id'],returncode=code,completed_unix=record['finished'],seed=record['seed'],samples=record['samples']))
        print('Finished',region,record['id'],'exit',code,flush=True)
        return code
    try:
        while active or (next_index<len(jobs) and not failed):
            while not failed and next_index<len(jobs) and len(active)<WORKERS:
                region,index,job=jobs[next_index];next_index+=1
                folder=Path(entries[region]['path']);manifest=manifests[region]
                require(sha(folder/'manifest.json')==entries[region]['manifest_sha256'],'Manifest changed at launch')
                for name in ('latent-region-normalizer','config.json','region.json','importance-guide.json','shape.json'):
                    require(sha(folder/'provenance'/name)==manifest['archive_sha256'][name],'Input changed at launch: '+name)
                require(not Path(job['directory']).exists(),'Population output appeared before launch')
                record=statuses[region]['jobs'][index];record.update(status='launching',started=time.time(),argv=job['command'],log=job['log'])
                snapshot()
                try:
                    with Path(job['log']).open('xb') as log:child=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
                except BaseException as error:
                    record.update(status='launch_failed',exception=repr(error),finished=time.time());raise
                active[region,index]=child;record.update(status='running',pid=child.pid);snapshot()
                print('Started',region,job['id'],'PID',child.pid,flush=True)
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
        for region,index,job in jobs:
            statuses[region]['jobs'][index]['output']=verify_population(Path(entries[region]['path']),manifests[region],job)
        snapshot()
    except BaseException as error:
        state.update(phase='physical_validation_failed',complete=False,exception=repr(error),finished=time.time());snapshot();raise
    try:
        state['phase']='audit';snapshot()
        for region,manifest in manifests.items():
            folder=Path(entries[region]['path'])
            require(not (folder/'assessment').exists(),'Audit output already exists; no retries')
            require(sha(folder/'provenance'/ANALYZER_NAME)==manifest['observer_sha256'],'Frozen auditor changed')
            argv=[sys.executable,'-B',str(folder/'provenance'/ANALYZER_NAME),'--root',str(folder)]
            state['audits'][region]=dict(argv=argv,started=time.time(),manifest_sha256=sha(folder/'manifest.json'));snapshot()
            with (journal/(region+'-audit.log')).open('xb') as log:
                result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            state['audits'][region].update(returncode=result.returncode,finished=time.time());snapshot();result.check_returncode()
            analysis=read(folder/'assessment/analysis.json')
            verify_assessment(folder,manifest,analysis,statuses[region]['jobs'])
            state['audits'][region]['analysis_sha256']=sha(folder/'assessment/analysis.json');snapshot()
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
