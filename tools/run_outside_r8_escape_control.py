#!/usr/bin/env python3
"""Run the prespecified frozen-parent/extension contact-escape control."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from prepare_smc_normalizer_atlas import ROOT,read,sha,write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,default=ROOT/'runs/outside-r8-atlas-extension-20260920/site0/benchmark-plan.json')
    ap.add_argument('--out',type=Path,default=ROOT/'runs/outside-r8-escape-2000')
    args=ap.parse_args();plan=read(args.plan)
    assert plan['cycles']==2000 and plan['jobs']==8 and plan['workers']==4
    out=args.out.resolve()
    if out.exists():raise ValueError('Refusing existing production output')
    original=ROOT/'runs/posterior-docking-mis-5000'
    oldmanifest=read(original/'manifest.json')
    binary=original/'provenance/docking-mc'
    assert sha(binary)==oldmanifest['binary_sha256']
    cfg=read(plan['physical_configuration'])
    assert cfg['poisson_lambda_ratio']==16 and cfg['local_attempts_per_cycle']==2 and cfg['uniform_probability']==.1
    out.mkdir(parents=True);archive=out/'provenance';archive.mkdir()
    inputs={'benchmark-plan.json':args.plan,'docking-mc':binary,'launcher.py':Path(__file__),
        'source-config.json':Path(plan['physical_configuration']),'shape.json':Path(cfg['shape']),
        'environment.json':Path(cfg['metadata']['environment']),
        'density-reference.py':ROOT/'tools/prepare_smc_normalizer_atlas.py',
        'registration-reference.py':ROOT/'tools/prepare_deep_far_normalizer_atlas.py',
        'chart-reference.py':ROOT/'tools/analyze_involution_docking_campaign.py',
        'posterior-analyzer.py':ROOT/'tools/analyze_posterior_docking_pilot.py',
        'old-region.json':ROOT/'runs/outside-r8-local-region-20260920/site0/provenance/old-region.json',
        'new-region.json':ROOT/'runs/outside-r8-local-region-20260920/site0/region-r3.json'}
    for name,path in inputs.items():shutil.copy2(path,archive/name)
    for name in ('docking.rs','depletion.rs','proposal.rs'):
        shutil.copy2(original/'provenance'/name,archive/name)
    alljobs=[]
    for modelname,sourcepath in plan['models'].items():
        assert sha(sourcepath)==plan['model_sha256'][modelname]
        campaign=out/modelname
        for sub in ('provenance','configs','runs','logs'):(campaign/sub).mkdir(parents=True)
        for name in ('shape.json','environment.json'):
            shutil.copy2(archive/name,campaign/'provenance'/name)
        shutil.copy2(sourcepath,campaign/'provenance/model.json')
        jobs=[]
        for rep,start in enumerate(plan['starts']):
            for mode,correlation in (('c0',0.),('c09',.9)):
                identifier=f'{modelname}-cohort{start["cohort_index"]:02d}-{mode}'
                conf=copy.deepcopy(cfg)
                conf.update(shape=str(campaign/'provenance/shape.json'),initial_pose=start['pose'],seed=plan['seeds_by_start'][rep])
                conf['metadata'].update(environment=str(campaign/'provenance/environment.json'),
                    start_basin='cohort',replicate=rep,mode=mode,atlas_frozen=True,
                    component_count=len(read(sourcepath)['weights']),initialization_equilibrated=False,
                    initial_pose_source=start,model_variant=modelname)
                path=campaign/'configs'/f'{identifier}.json';write(path,conf)
                job=dict(id=identifier,start='cohort',replicate=rep,mode=mode,seed=conf['seed'],
                    correlation=correlation,method='posterior-involution',config=str(path),config_sha256=sha(path),
                    directory=str(campaign/'runs'/identifier),campaign=str(campaign),model_variant=modelname,
                    model=str(campaign/'provenance/model.json'),model_sha256=sha(sourcepath),cohort_index=start['cohort_index'])
                jobs.append(job);alljobs.append(job)
        write(campaign/'manifest.json',dict(schema=1,jobs=jobs,cycles=2000,sample_every=1,
            model_sha256=sha(sourcepath),binary_sha256=sha(archive/'docking-mc'),
            scope='Matched nonstationary contact-escape control; no equilibrium-rate claim.'))
    manifest=dict(schema=1,created_unix_time=time.time(),jobs=alljobs,cycles=2000,workers=4,
        input_sha256={p.name:sha(p) for p in archive.iterdir()},binary_sha256=sha(archive/'docking-mc'),
        plan_sha256=sha(args.plan),per_job_wall_guard_seconds=300,
        scope='Eight prespecified jobs only; original parent and all old results unchanged. Freeze before launch; no refitting.')
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(prepared=True,out=str(out),jobs=8,binary_sha256=manifest['binary_sha256'])),flush=True)
    def run(job):
        command=[str(archive/'docking-mc'),'--config',job['config'],'--model',job['model'],
            '--out',job['directory'],'--cycles','2000','--sample-every','1',
            '--method','posterior-involution','--correlation',str(job['correlation'])]
        start=time.monotonic();log=Path(job['campaign'])/'logs'/f'{job["id"]}.log';timed_out=False
        with log.open('w') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            write(log.with_suffix('.process.json'),dict(pid=process.pid,command=command))
            try:code=process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid,signal.SIGTERM);code=process.wait();timed_out=True
        path=Path(job['directory'])/'summary.json';summary=read(path) if path.exists() else {}
        return dict(id=job['id'],returncode=code,timed_out=timed_out,complete=summary.get('complete',False),
            cpu_seconds=summary.get('sampler_cpu_seconds'),wall_seconds=time.monotonic()-start)
    records=[];begun=time.monotonic()
    write(out/'status.json',dict(running=True,completed=0,total=8,supervisor_pid=os.getpid()))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run,j) for j in alljobs]):
            records.append(future.result());print(json.dumps(records[-1]),flush=True)
            write(out/'status.json',dict(running=True,completed=len(records),total=8,records=records))
    for name,digest in manifest['input_sha256'].items():assert sha(archive/name)==digest
    for job in alljobs:
        assert sha(job['config'])==job['config_sha256'] and sha(job['model'])==job['model_sha256']
    result=dict(complete=all(r['returncode']==0 and r['complete'] and not r['timed_out'] for r in records),
        records=records,wall_seconds=time.monotonic()-begun,total_sampler_cpu_seconds=sum(r['cpu_seconds'] or 0 for r in records))
    write(out/'summary.json',result);write(out/'status.json',dict(result,running=False,completed=len(records),total=8))
    if not result['complete']:raise SystemExit('Incomplete/failing jobs preserved; do not treat as a completed benchmark')


if __name__=='__main__':main()
