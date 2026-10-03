#!/usr/bin/env python3
"""Classify every accepted width-probe endpoint with the frozen native observer.

Read-only post-hoc labeling; no native information enters proposals or MH.
No rejected endpoint is classified or counted as a sampled native event.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from analyze_refined_atlas_assembly import native_frame
from native_contact_regions import NativeContactRegions
from prepare_shoulder_docking_benchmark import local_dependencies

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def main(run,definition):
    run=Path(run).resolve();definition=Path(definition).resolve();started=time.process_time()
    review=read(run/'completed-review.json');analysis=read(run/'analysis.json')
    assert review['complete'] and review['passed'] and analysis['complete'] and analysis['passed']
    assert review['physical_exit_code']==review['audit_exit_code']==0
    files={str(run/name):digest for name,digest in review['output_hashes'].items()}
    files[str(run/'completed-review.json')]=sha(run/'completed-review.json')
    d=read(definition);files[str(definition)]=sha(definition)
    files.update({str(definition.parent/'inputs'/name):digest for name,digest in d['input_sha256'].items()})
    state=read(run/'execution/source-state.json')
    plan=read(run/'config.json');reference=read(plan['reference_config']['path'])
    assert d['shape_sha256']==reference['shape']['sha256']
    files[plan['reference_config']['path']]=plan['reference_config']['sha256']
    files[reference['shape']['path']]=reference['shape']['sha256']
    sources=list(local_dependencies([Path(__file__)]).values())
    files.update({str(p):sha(p) for p in sources})
    assert all(sha(p)==h for p,h in files.items())
    binding=run/'accepted-native-binding.json'
    write(binding,dict(schema='fft-width-accepted-native-binding-v1',input_sha256=files,
        allocation='One full initial-state observer evaluation and one per accepted endpoint; every accepted row, no sampling',
        criteria=d['criteria'],scope='Instantaneous complete entry and global catalogue-cycle consistency; no hysteresis, native selection or physical reweighting. These reset endpoints are not a trajectory.'))
    archive=run/'accepted-native-source';archive.mkdir()
    for p in sources:shutil.copyfile(p,archive/Path(p).name)
    classifier=NativeContactRegions(definition)
    initial=native_frame(dict(poses=state,sweep=0,sampler_cpu_seconds=0.),classifier,set())
    before=set(map(tuple,initial['native_keys']));rows=[];n=0
    with (run/'execution/attempts.jsonl').open() as f:
        for line in f:
            raw=json.loads(line);n+=1
            if not raw['accepted']:continue
            observed=native_frame(dict(poses=raw['retained_state'],sweep=raw['index'],sampler_cpu_seconds=0.),classifier,set())
            after=set(map(tuple,observed['native_keys']))
            cached=raw['cached'];members={cached['case']['root'],cached['case']['child']}
            assert all(i in members or j in members for i,j,k in before.symmetric_difference(after))
            rows.append(dict(index=raw['index'],atlas=cached['atlas'],method=cached['method'],case=cached['case']['name'],attempt=cached['attempt'],
                gained_native_keys=sorted(after-before),lost_native_keys=sorted(before-after),observer=observed))
    assert n==analysis['summary']['outer_attempts']==1536
    assert [r['index'] for r in rows]==[r['index'] for r in analysis['rows'] if r['accepted']]
    assert len(rows)==analysis['summary']['accepted']
    assert all(sha(p)==h for p,h in files.items())
    result=dict(schema='fft-width-accepted-native-analysis-v1',complete=True,passed=True,binding_sha256=sha(binding),
        unconditional_outer_attempts=n,accepted_endpoints=len(rows),initial_observer=initial,rows=rows,
        accepted_native_gains=sum(len(r['gained_native_keys']) for r in rows),accepted_native_losses=sum(len(r['lost_native_keys']) for r in rows),
        classification_cpu_seconds=time.process_time()-started,
        limitation='Only accepted reset endpoints are labeled. Does not estimate equilibrium weights, ESS, assembly stability or missed near-native basins.')
    output=run/'accepted-native-analysis.json';write(output,result)
    write(run/'accepted-native-review.json',dict(complete=True,passed=True,input_files_verified=len(files),
        output_sha256=sha(output),binding_sha256=sha(binding),source_sha256={p.name:sha(p) for p in archive.iterdir()}))
    print(json.dumps({k:result[k] for k in ['complete','passed','accepted_endpoints','accepted_native_gains','accepted_native_losses','classification_cpu_seconds']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'results/fft-width-physical-20261003')
    p.add_argument('--definition',type=Path,default=ROOT/'runs/mobile-native-region-definition-20260921/definition.json');a=p.parse_args();main(a.run,a.definition)
