#!/usr/bin/env python3
"""Separately audit preserved full-capture rows after the diagnosed factor mismatch.

No physical executable is invoked. Original inputs, statuses and failed audit
remain untouched. A frozen corrected observer is run once per original arm.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

SCHEMA='mobile-full-capture-audit-recovery-v1'
PROTOCOL_SHA='53c6cad8eb785840fa11789eead83a52ad83986d484e8af8b12779c4ac39c32e'
DIAGNOSIS_SHA='e1c764509ec8cf99c0da5b57192bf170a62074a7254aaaa18982841036a06b21'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(value,message):
    if not value:raise ValueError(message)


def original_snapshot(campaign):
    campaign=Path(campaign).resolve();protocol=read(campaign/'protocol.json');status=read(campaign/'status.json')
    require(sha(campaign/'protocol.json')==PROTOCOL_SHA==status['protocol_sha256'],'Original protocol differs')
    require(status['phase']=='audit_failed' and status['complete']is False,'Expected preserved original audit failure')
    require(len(status['jobs'])==8 and all(j['status']=='complete' and j['exit_code']==0 for j in status['jobs']),
            'Every original physical population must be terminal and successful')
    require(set(status['audits'])=={'epsilon-0p1'} and status['audits']['epsilon-0p1']['returncode']==1,
            'Original observer failure history differs')
    frozen=read(campaign/'freeze.json')['files']
    for name,digest in frozen.items():require(sha(campaign/name)==digest,'Original frozen input changed: '+name)
    hashes={}
    for entry in protocol['campaigns']:
        folder=Path(entry['path']);require(folder==campaign/entry['arm'],'Original arm path differs')
        require(sha(folder/'manifest.json')==entry['manifest_sha256'],'Original arm manifest differs')
        manifest=read(folder/'manifest.json')
        for job in manifest['jobs']:
            term=next(j for j in status['jobs']if j['arm']==entry['arm'] and j['id']==job['id'])
            directory=Path(job['directory']);require(directory==folder/'runs'/job['id'],'Original population path differs')
            for name,key in [('samples.jsonl','samples_sha256'),('summary.json','summary_sha256'),('manifest.json','manifest_sha256')]:
                path=directory/name;require(sha(path)==term['output'][key],'Physical output changed: '+str(path));hashes[str(path)]=sha(path)
            summary=read(directory/'summary.json');pm=read(directory/'manifest.json')
            require(summary['complete'] and summary['manifest']==pm and pm['samples']==8192 and pm['seed']==job['seed'],
                    'Incomplete original population')
    return dict(protocol_sha256=sha(campaign/'protocol.json'),status_sha256=sha(campaign/'status.json'),
                freeze_sha256=sha(campaign/'freeze.json'),physical_output_sha256=hashes)


def freeze(campaign,out,diagnosis):
    from prepare_shoulder_docking_benchmark import local_dependencies
    campaign,out,diagnosis=Path(campaign).resolve(),Path(out).resolve(),Path(diagnosis).resolve()
    require(not out.exists(),'Use a fresh recovery directory')
    snapshot=original_snapshot(campaign);require(sha(diagnosis)==DIAGNOSIS_SHA,'Reviewed factor diagnosis changed')
    tools=Path(__file__).resolve().parent
    dependencies=local_dependencies([tools/'analyze_basin_normalizers.py',Path(__file__)])
    require('normalizer_proposal_density.py'in dependencies,'Missing corrected factor observer')
    archive=out/'provenance';archive.mkdir(parents=True)
    for name,path in dependencies.items():shutil.copy2(path,archive/name)
    shutil.copy2(diagnosis,archive/'diagnosis.json')
    manifest=dict(schema=SCHEMA,campaign=str(campaign),original=snapshot,diagnosis_sha256=DIAGNOSIS_SHA,
        archive_sha256={p.name:sha(p)for p in archive.iterdir()},
        scope='Observer correction only: reconstruct the actual scalar Cholesky factor used consistently for Gaussian draws and densities, independently check its covariance backward error, and retain the original 2e-8 full-density tolerance. All physical poses, clouds, weights, regions, budgets and original failed audit remain unchanged.',
        physical_jobs_launched=0)
    write(out/'manifest.json',manifest);return manifest


def validate_inputs(out):
    out=Path(out).resolve();m=read(out/'manifest.json');require(m['schema']==SCHEMA,'Wrong recovery schema')
    require(original_snapshot(m['campaign'])==m['original'],'Original experiment changed')
    for name,digest in m['archive_sha256'].items():require(sha(out/'provenance'/name)==digest,'Recovery observer changed: '+name)
    require(m['diagnosis_sha256']==DIAGNOSIS_SHA==sha(out/'provenance/diagnosis.json'),'Recovery diagnosis differs')
    return m


def verify_audit(out,entry,record,analysis):
    folder=Path(entry['path']);manifest=read(folder/'manifest.json')
    require(not analysis['pending'] and len(analysis['populations'])==4,'Recovery audit did not cover all populations')
    require(analysis['provenance'][str(folder/'manifest.json')]==entry['manifest_sha256'],'Recovered arm binding differs')
    require(analysis['provenance'][record['observer_path']]==record['observer_sha256'],'Recovered observer identity differs')
    for result,job in zip(analysis['populations'],manifest['jobs']):
        require(result['job']==job,'Recovered population identity differs')
        audit=result['proposal_audit']
        require(audit['checked_actual_poses']==8192 and audit['proposal_anchor_indices']==[0,1]
                and audit['maximum_log_density_error']<2e-8,'Incomplete or inaccurate proposal reconstruction')
        require(audit.get('factor_audit')is not None,'No audited effective Gaussian factors')
        for name in ('samples.jsonl','summary.json','manifest.json'):
            path=Path(job['directory'])/name
            require(analysis['provenance'][str(path)]==sha(path),'Recovered physical data binding differs')


def run(out):
    out=Path(out).resolve();m=validate_inputs(out)
    require(sha(__file__)==m['archive_sha256'][Path(__file__).name],'Run the frozen recovery script')
    require(not(out/'status.json').exists(),'Recovery cannot retry or overwrite')
    state=dict(schema=SCHEMA,complete=False,phase='audit',manifest_sha256=sha(out/'manifest.json'),
               started=time.time(),audits={},physical_jobs_launched=0)
    write(out/'status.json',state)
    try:
        for entry in read(Path(m['campaign'])/'protocol.json')['campaigns']:
            arm=entry['arm'];assessment=out/arm;observer=out/'provenance/analyze_basin_normalizers.py'
            require(not assessment.exists(),'Assessment already exists')
            argv=[sys.executable,'-B',str(observer),'--root',entry['path'],'--out',str(assessment)]
            record=dict(argv=argv,assessment_path=str(assessment/'analysis.json'),observer_path=str(observer),
                        observer_sha256=sha(observer),manifest_sha256=entry['manifest_sha256'],started=time.time())
            state['audits'][arm]=record;write(out/'status.json',state)
            with(out/(arm+'.log')).open('x')as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode,finished=time.time());write(out/'status.json',state);result.check_returncode()
            analysis=read(assessment/'analysis.json');verify_audit(out,entry,record,analysis)
            record['analysis_sha256']=sha(assessment/'analysis.json');write(out/'status.json',state)
        validate_inputs(out);state.update(complete=True,phase='complete',finished=time.time())
    except BaseException as error:
        state.update(phase='failed',exception=repr(error),finished=time.time());write(out/'status.json',state);raise
    write(out/'status.json',state);return state


def validate_completed(campaign,out):
    out=Path(out).resolve();m=validate_inputs(out);state=read(out/'status.json')
    require(Path(m['campaign'])==Path(campaign).resolve(),'Recovery targets another campaign')
    require(state['schema']==SCHEMA and state['complete'] and state['phase']=='complete'
            and state['manifest_sha256']==sha(out/'manifest.json') and state['physical_jobs_launched']==0,'Recovery incomplete')
    protocol=read(Path(campaign)/'protocol.json')
    require(set(state['audits'])=={e['arm']for e in protocol['campaigns']},'Recovered arm set differs')
    for entry in protocol['campaigns']:
        record=state['audits'][entry['arm']];path=out/entry['arm']/'analysis.json'
        require(record['observer_path']==str(out/'provenance/analyze_basin_normalizers.py')
                and record['observer_sha256']==m['archive_sha256']['analyze_basin_normalizers.py']
                and record['manifest_sha256']==entry['manifest_sha256'],'Recovered observer/arm identity differs')
        require(record['assessment_path']==str(path) and record['returncode']==0
                and record['analysis_sha256']==sha(path),'Recovered assessment changed')
        verify_audit(out,entry,record,read(path))
    return m,state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--campaign',type=Path);parser.add_argument('--diagnosis',type=Path)
    args=parser.parse_args()
    if args.action=='freeze':
        require(args.campaign is not None and args.diagnosis is not None,'Freeze requires campaign and diagnosis')
        freeze(args.campaign,args.out,args.diagnosis)
    else:run(args.out)
