#!/usr/bin/env python3
"""Compare gate cost at identical retained reference endpoints, not different runs.

Requires the existing Python reference environment (NumPy/SciPy), explicitly
provided by --reference. This validation tool is not a runtime dependency.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cases-per-arm',type=int,default=8)
    parser.add_argument('--repetitions',type=int,default=3)
    args=parser.parse_args()
    assert args.cases_per_arm>0 and args.repetitions>0
    sys.path.insert(0,str(args.reference/'scripts'))
    import numpy as np
    from scipy.spatial.transform import Rotation
    from run_learned_tetramer_pilot import PeriodicEndpointEnvironment
    from endpoint_cloud_gate import EndpointCloudGate
    production=args.reference/'results/learned-tetramer-assembly/production'
    selected=[]
    for arm in ('seeded','seed-free'):
        folder=production/'runs'/f'{arm}-r00-learned'
        with gzip.open(folder/'moves.jsonl.gz','rt') as stream:
            rows=[json.loads(line) for line in stream]
        eligible=[r for r in rows if r.get('hard_valid') and r.get('gate',{}).get('raw_points',0)>0]
        for index in np.linspace(0,len(eligible)-1,min(args.cases_per_arm,len(eligible)),dtype=int):
            row=eligible[index];poses=[None]*(len(row['spectator_poses'])+1)
            poses[row['moving_index']]=row['old_pose']
            for s in row['spectator_poses']:poses[s['body_index']]={k:s[k] for k in ('position','orientation')}
            selected.append(dict(label=f'{arm}-s{row["sweep"]}-u{row["update_in_sweep"]}',poses=poses,moving_index=row['moving_index'],new_pose=row['proposed_pose']))
    cfg=json.loads((production/'runs/seeded-r00-learned/config.json').read_text())
    document=dict(shape=cfg['shape'],box_lengths=cfg['box_lengths'],depletant_radius=cfg['depletant_radius'],activity=cfg['reservoir_density'],lambda_ratio=16.,cases=selected)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'cases.json').write_text(json.dumps(document,indent=2)+'\n')
    root=Path(__file__).resolve().parents[1]
    binary=root/'target/release/gate_benchmark'
    binary_hash=hashlib.sha256(binary.read_bytes()).hexdigest()
    subprocess.run([str(root/'target/release/gate_benchmark'),'--input',str(args.out/'cases.json'),'--out',str(args.out/'rust.json'),'--repetitions',str(args.repetitions)],check=True)
    shape=json.loads(Path(cfg['shape']).read_text())
    body=np.array([a['center'] for a in shape['atoms']]);radii=np.array([a['radius'] for a in shape['atoms']])
    builder=PeriodicEndpointEnvironment(body,radii,np.array(cfg['box_lengths']),cfg['depletant_radius'])
    def R(pose):return Rotation.from_quat(np.array(pose['orientation'])[[1,2,3,0]]).as_matrix()
    results=[]
    for number,case in enumerate(selected):
        positions=np.array([p['position'] for p in case['poses']]);rotations=np.array([R(p) for p in case['poses']])
        i=case['moving_index'];new=case['new_pose'];new_t=np.array(new['position']);new_R=R(new)
        start=time.process_time();env,_=builder.build(positions,rotations,i,positions[i],new_t)
        assert env.hard_valid(positions[i],rotations[i]) and env.hard_valid(new_t,new_R)
        setup=time.process_time()-start;gate=EndpointCloudGate(env);samples=[];rng=np.random.default_rng(313001+number)
        start=time.process_time()
        for _ in range(args.repetitions):samples.append(gate.sample(rng,16*cfg['reservoir_density'],cfg['reservoir_density'],positions[i],rotations[i],new_t,new_R))
        results.append(dict(label=case['label'],setup_cpu_seconds=setup,gate_cpu_seconds=time.process_time()-start,repetitions=args.repetitions,samples=samples))
    (args.out/'python.json').write_text(json.dumps(dict(cases=results),indent=2)+'\n')
    rust=json.loads((args.out/'rust.json').read_text())['cases']
    totals={name:dict(gate_cpu_seconds=sum(row['gate_cpu_seconds'] for row in rows),setup_cpu_seconds=sum(row['setup_cpu_seconds'] for row in rows),raw_points=sum(s['raw_points'] for row in rows for s in row['samples'])) for name,rows in [('python',results),('rust',rust)]}
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==binary_hash
    sources={name:hashlib.sha256((args.reference/'scripts'/name).read_bytes()).hexdigest() for name in ('endpoint_cloud_gate.py','contact_triangle_cloud.py','run_learned_tetramer_pilot.py')}
    report=dict(totals=totals,gate_speedup=totals['python']['gate_cpu_seconds']/totals['rust']['gate_cpu_seconds'],rust_executable_sha256=binary_hash,python_source_sha256=sources,cases_sha256=hashlib.sha256((args.out/'cases.json').read_bytes()).hexdigest(),scope='Same endpoints and physical parameters, independent RNGs. Gate speedup only; different conservative partitions may change raw work. Excludes training, compilation, trajectories and mixture evaluation.')
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
