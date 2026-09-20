#!/usr/bin/env python3
"""Run the fixed minimal empty/B/AB factorial screen after independent preflight checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rotation(pose):
    return Rotation.from_quat(np.array(pose['orientation'])[[1,2,3,0]]).as_matrix()


def stripped(config):
    result=copy.deepcopy(config)
    result.pop('fixed_poses')
    result['metadata'].pop('cooperative_plan',None)
    return result


def preflight(plan_root, a_root):
    plan=json.loads((plan_root/'manifest.json').read_text())
    base_path=Path(plan['source_config'])
    assert sha(base_path)==plan['source_sha256']
    base=json.loads(base_path.read_text())
    a_manifest=json.loads((a_root/'manifest.json').read_text())
    first=Path(a_manifest['jobs'][0]['output'])
    physical=json.loads((first/'manifest.json').read_text())
    cover=json.loads((first/'cover.json').read_text())
    assert a_manifest['config_sha256']==sha(base_path)
    binary=a_root/'provenance/native-region-normalizer'
    assert sha(binary)==a_manifest['binary_sha256']
    assert sha(base['shape'])==physical['shape_sha256']
    assert base['depletant_radius']==physical['depletant_radius']==1.5
    assert base['reservoir_density']==physical['activity']==.035
    assert physical['lambda_ratio']==64 and physical['cloud_replicates']==2
    for key,value in physical['metric'].items():
        assert base['metadata'][key]==value
    cfgs={}
    for label in ['empty','A','B','AB']:
        item=plan['configs'][label];path=Path(item['path']);assert sha(path)==item['sha256']
        config=json.loads(path.read_text());assert stripped(config)==stripped(base)
        cfgs[label]=config
    assert cfgs['empty']['fixed_poses']==[]
    assert cfgs['A']['fixed_poses']==base['fixed_poses']
    assert cfgs['AB']['fixed_poses']==cfgs['A']['fixed_poses']+cfgs['B']['fixed_poses']
    assert len(cfgs['A']['fixed_poses'])==len(cfgs['B']['fixed_poses'])==1

    shape=json.loads(Path(base['shape']).read_text())
    atoms=np.array([atom['center'] for atom in shape['atoms']])
    radii=np.array([atom['radius'] for atom in shape['atoms']])
    def gap(p,q):
        moving=atoms@rotation(p).T+p['position'];fixed=atoms@rotation(q).T+q['position']
        minimum=np.inf
        for radius in np.unique(radii):
            tree=cKDTree(fixed[radii==radius]);distance=tree.query(moving,k=1,workers=1)[0]
            minimum=min(minimum,float(np.min(distance-radii-radius)))
        return minimum
    reference=base['metadata']['native_poses'][0]
    geometric_checks={}
    for label,config in cfgs.items():
        checks=[]
        for j,fixed in enumerate(config['fixed_poses']):
            value=gap(reference,fixed);assert value>=0, (label,'native',j,value)
            checks.append({'pair':['native',j],'minimum_atomic_gap_A':value})
            for i in range(j):
                value=gap(config['fixed_poses'][i],fixed);assert value>=0, (label,i,j,value)
                checks.append({'pair':[i,j],'minimum_atomic_gap_A':value})
        geometric_checks[label]=checks
    body_bound=float(np.max(np.linalg.norm(atoms,axis=1)+radii))
    box=np.array(base['metadata']['source_periodic_box']);center=np.array(base['capture_center'])
    image_gaps=[]
    for fixed in cfgs['AB']['fixed_poses']:
        for image in itertools.product((-1,0,1),repeat=3):
            if image!=(0,0,0):
                image_gaps.append(np.linalg.norm(np.array(fixed['position'])+np.array(image)*box-center)
                                  -base['capture_radius']-2*body_bound)
    # Exclude both hard interactions and contacts between inflated spheres.
    assert min(image_gaps)>2*base['depletant_radius']
    return plan,cfgs,binary,{'passed':True,'physical_fields_exactly_equal_except_fixed_poses':True,
        'metadata_only_change':'cooperative_plan annotation', 'shape_sha256':physical['shape_sha256'],
        'reference_cover':cover,'identical_cover_argument':'same frozen binary and bit-identical complete native metric',
        'fixed_fixed_and_reference_pose_checks':geometric_checks,
        'nonzero_periodic_image_gap_lower_A':float(min(image_gaps)),
        'reused_A_manifest_sha256':sha(a_root/'manifest.json'),'frozen_binary_sha256':sha(binary)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--A-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();plan_root=args.plan.resolve();a_root=args.A_root.resolve();root=args.out.resolve()
    assert not root.exists(),'Refuse existing execution directory'
    plan,cfgs,binary,checks=preflight(plan_root,a_root)
    (root/'provenance').mkdir(parents=True)
    frozen=root/'provenance/native-region-normalizer';shutil.copy2(binary,frozen)
    shutil.copy2(__file__,root/'provenance/run_native_factorial_screen.py')
    shutil.copy2(plan_root/'manifest.json',root/'provenance/prepared-plan.json')
    (root/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
    jobs=[];groups={}
    for label,samples,seed_base in [('empty',262144,98781010),('B',1048576,98791010),('AB',1048576,98801010)]:
        target=root/label;(target/'provenance').mkdir(parents=True);(target/'runs').mkdir();(target/'logs').mkdir()
        config=target/'provenance/config.json';shutil.copy2(plan['configs'][label]['path'],config)
        subset=[]
        for i in range(4):
            output=target/'runs'/f'r{i:02d}';seed=seed_base+1009*i
            command=[str(frozen),'--config',str(config),'--out',str(output),'--samples',str(samples),
                     '--seed',str(seed),'--lambda-ratio','64','--cloud-replicates','2']
            job={'subset':label,'replicate':i,'seed':seed,'output':str(output),'samples':samples,'command':command}
            jobs.append(job);subset.append(job)
        group={'protocol':'native-factorial-fixed-budget-v1','subset':label,'jobs':subset,
               'config_sha256':sha(config),'binary_sha256':sha(frozen),'total_unconditional_draws':4*samples}
        (target/'manifest.json').write_text(json.dumps(group,indent=2)+'\n');groups[label]=group
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'groups':groups,
              'reused_A_root':str(a_root),'total_new_unconditional_draws':sum(j['samples'] for j in jobs),
              'maximum_workers':8,'frozen_binary_sha256':sha(frozen),'preflight_sha256':sha(root/'preflight.json'),
              'runner_sha256':sha(__file__),'no_replacement':'Fixed budget; no early stop, zero replacement or outcome-selected rerun'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    state={'runner_pid':os.getpid(),'complete':False,'jobs':{}};lock=threading.Lock()
    def persist():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();tmp=root/'runner-status.tmp'
        tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'runner-status.json')
    def run(job):
        label=f"{job['subset']}-r{job['replicate']:02d}";started=time.monotonic()
        with (root/job['subset']/'logs'/f"r{job['replicate']:02d}.log").open('x') as log:
            process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
            with lock:state['jobs'][label]={'pid':process.pid,'status':'running'};persist()
            while process.poll() is None:
                with lock:
                    progress=Path(job['output'])/'progress.json'
                    if progress.exists():
                        try:state['jobs'][label]['progress']=json.loads(progress.read_text())
                        except json.JSONDecodeError:pass
                    persist()
                time.sleep(2)
            with lock:
                result=state['jobs'][label];result.update(status='complete' if process.returncode==0 else 'failed',
                    exit_code=process.returncode,wall_seconds=time.monotonic()-started)
                summary=Path(job['output'])/'summary.json'
                if summary.exists():result['summary_sha256']=sha(summary)
                persist()
            return process.returncode
    with ThreadPoolExecutor(max_workers=8) as pool:codes=list(pool.map(run,jobs))
    state['complete']=True;state['success']=all(code==0 for code in codes);persist()
    print(json.dumps(state),flush=True)
    if not state['success']:raise SystemExit('Retained failed job: diagnosis required, no replacement')


if __name__=='__main__':main()
