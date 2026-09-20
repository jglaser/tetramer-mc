#!/usr/bin/env python3
"""Prepare or run the prespecified four-target nested-cover precision screen.

Default is preparation only. --execute requires a tested release executable and
a fresh output directory. Every target uses four fixed 16,384-draw populations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time


SCALES=[.1,.2,.4,1.]
WEIGHTS=[.25]*4
SAMPLES=16384
REPLICATES=4
SEED_BASES={'empty':98911010,'A':98921010,'B':98931010,'AB':98941010}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(args):
    design=json.loads(args.plan.read_text())
    configs={label:design['configs'][label] for label in SEED_BASES}
    for entry in configs.values():assert sha(entry['path'])==entry['sha256']
    return {'protocol':'nested-native-cover-fixed-budget-v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'original_design':str(args.plan),'original_design_sha256':sha(args.plan),
        'configs':configs,'reference_A_root':str(args.A_root),
        'reference_factorial_root':str(args.factorial_root),'output':str(args.out),
        'scales':SCALES,'weights':WEIGHTS,'samples_per_population':SAMPLES,
        'populations_per_target':REPLICATES,'seed_bases':SEED_BASES,
        'replicate_seed_stride':1009,'maximum_workers':8,
        'cloud_replicates':2,'lambda_ratio':64,
        'total_unconditional_draws':len(SEED_BASES)*REPLICATES*SAMPLES,
        'target_rule':'Original native metric and target remain unchanged; every weight divides by the full normalized cover mixture density.',
        'stopping_rule':'All prescribed populations retained. No early stopping, selected replacement, or pooling with prior experiments.',
        'runner_sha256':sha(__file__)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('runs/native-cooperativity-plan-20260920/manifest.json'))
    parser.add_argument('--A-root',type=Path,default=Path('runs/native-region-reference-8x4194304-l64-20260920'))
    parser.add_argument('--factorial-root',type=Path,default=Path('runs/native-factorial-screen-20260920'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--prepare-output',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    for field in ['plan','A_root','factorial_root','out']:
        setattr(args,field,getattr(args,field).resolve())
    prepared=plan(args)
    if not args.execute:
        if args.prepare_output:
            assert not args.prepare_output.exists(),'Keep the previous declaration immutable'
            args.prepare_output.write_text(json.dumps(prepared,indent=2)+'\n')
        print(json.dumps(prepared,indent=2));return
    assert args.binary is not None,'A tested release executable is required'
    assert not args.out.exists(),'Refuse an existing execution directory; no implicit restart'
    binary=args.binary.resolve()
    # Reuse the independent shape/frame/bath/native/fixed-pair checks. This
    # validates the old frozen baseline too; the new executable is frozen below.
    from run_native_factorial_screen import preflight
    _,_,_,checks=preflight(args.plan.parent,args.A_root)
    help_text=subprocess.run([str(binary),'--help'],check=True,capture_output=True,text=True).stdout
    assert '--cover-scales' in help_text and '--cover-weights' in help_text
    root=args.out;(root/'provenance').mkdir(parents=True)
    frozen=root/'provenance/native-region-normalizer';shutil.copy2(binary,frozen)
    for source in [Path(__file__),Path(__file__).with_name('analyze_native_region_reference.py'),
                   Path(__file__).with_name('run_native_factorial_screen.py')]:
        shutil.copy2(source,root/'provenance'/source.name)
    (root/'prepared-protocol.json').write_text(json.dumps(prepared,indent=2)+'\n')
    (root/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
    jobs=[]
    for label,seed_base in SEED_BASES.items():
        target=root/label;(target/'provenance').mkdir(parents=True)
        (target/'runs').mkdir();(target/'logs').mkdir()
        config=target/'provenance/config.json';shutil.copy2(prepared['configs'][label]['path'],config)
        subset=[]
        for replicate in range(REPLICATES):
            seed=seed_base+1009*replicate;output=target/'runs'/f'r{replicate:02d}'
            command=[str(frozen),'--config',str(config),'--out',str(output),'--samples',str(SAMPLES),
                '--seed',str(seed),'--cloud-replicates','2','--lambda-ratio','64',
                '--cover-scales','0.1,0.2,0.4,1','--cover-weights','1,1,1,1']
            job={'target':label,'replicate':replicate,'seed':seed,'output':str(output),
                'samples':SAMPLES,'command':command};subset.append(job);jobs.append(job)
        group={'protocol':prepared['protocol'],'target':label,'jobs':subset,'config_sha256':sha(config),
            'binary_sha256':sha(frozen),'total_unconditional_draws':REPLICATES*SAMPLES}
        (target/'manifest.json').write_text(json.dumps(group,indent=2)+'\n')
    manifest=dict(prepared,jobs=jobs,binary_sha256=sha(frozen),preflight_sha256=sha(root/'preflight.json'))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    state={'runner_pid':os.getpid(),'complete':False,'jobs':{}};lock=threading.Lock()
    def persist():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temporary=root/'runner-status.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(root/'runner-status.json')
    def run(job):
        key=f"{job['target']}-r{job['replicate']:02d}";started=time.monotonic()
        with (root/job['target']/'logs'/f"r{job['replicate']:02d}.log").open('x') as log:
            process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
            with lock:state['jobs'][key]={'pid':process.pid,'status':'running','seed':job['seed']};persist()
            while process.poll() is None:
                with lock:
                    progress=Path(job['output'])/'progress.json'
                    if progress.exists():
                        try:state['jobs'][key]['progress']=json.loads(progress.read_text())
                        except json.JSONDecodeError:pass
                    persist()
                time.sleep(2)
            with lock:
                state['jobs'][key].update(status='complete' if process.returncode==0 else 'failed',
                    exit_code=process.returncode,wall_seconds=time.monotonic()-started)
                persist()
            return process.returncode
    with ThreadPoolExecutor(max_workers=8) as pool:codes=list(pool.map(run,jobs))
    state['simulation_complete']=True;state['success']=all(code==0 for code in codes);persist()
    if not state['success']:
        state['complete']=True;persist();raise SystemExit('Failed outputs retained without replacement')
    analyzer=root/'provenance/analyze_native_region_reference.py'
    state['analyses']={}
    for label in SEED_BASES:
        log_path=root/label/'analysis.log'
        with log_path.open('x') as log:
            result=subprocess.run([os.sys.executable,str(analyzer),str(root/label),'--workers','4'],
                stdout=log,stderr=subprocess.STDOUT)
        state['analyses'][label]={'exit_code':result.returncode};persist()
        if result.returncode:
            state.update(complete=True,success=False);persist()
            raise SystemExit(f'{label} streaming audit failed; original outputs retained')
    state['complete']=True;persist()
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
