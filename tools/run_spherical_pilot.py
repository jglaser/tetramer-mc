#!/usr/bin/env python3
"""Matched spherical tests from previously assembled periodic oligomers.

Unwrap connected exclusion-bound neighborhoods and independently place those
rigid fragments inside an equal-volume sphere. This is a new initialization,
not an equilibrium map between periodic and spherical ensembles.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import argparse, copy, hashlib, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from spherical_ensemble import SphericalSystem, pose, save, sha


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    inputs=out/'inputs'; inputs.mkdir(exist_ok=True)
    model=ROOT/'examples/frozen-relative-mixture.json'
    jobs=[]; rng=np.random.default_rng(9827131)
    for arm in ('seeded','seed-free'):
        source=ROOT/'runs/rust-validation-400/runs'/f'{arm}-r00-learned'
        cfg=json.loads((source/'config.json').read_text())
        checkpoint=json.loads((source/'checkpoint.json').read_text())
        shape_path=ROOT/'examples/tetramer-shape.json'; shape=json.loads(shape_path.read_text())
        body=np.array([a['center'] for a in shape['atoms']]); radii=np.array([a['radius'] for a in shape['atoms']])
        lengths=np.array(cfg['box_lengths']); radius=float((3*np.prod(lengths)/(4*np.pi))**(1/3))
        system=SphericalSystem(body,radii,radius,cfg['depletant_radius'],cfg['reservoir_density'])
        old=np.array([p['position'] for p in checkpoint['poses']]); q=np.array([p['orientation'] for p in checkpoint['poses']]); rotations=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        delta=old[None]-old[:,None]; delta-=lengths*np.floor(delta/lengths+.5)
        adjacency=np.linalg.norm(delta,axis=2)<2*system.ebound
        unseen=set(range(len(old))); groups=[]; relative=old.copy()
        while unseen:
            initial=min(unseen); unseen.remove(initial); group=[initial]; relative[initial]=0.
            for i in group:
                for j in sorted(unseen.copy()):
                    if adjacency[i,j]:
                        unseen.remove(j); group.append(j); relative[j]=relative[i]+delta[i,j]
            relative[group]-=relative[group].mean(axis=0)
            groups.append(group)
        groups.sort(key=lambda g:-len(g)); placed=[]; positions=old.copy()
        for group in groups:
            for attempt in range(10000):
                center=rng.uniform(-radius,radius,3)
                trial=relative[group]+center
                if all(system.inside(t,rotations[i]) for t,i in zip(trial,group)) and not any(
                    system.overlap(t,rotations[i],positions[j],rotations[j]) for t,i in zip(trial,group) for j in placed):
                    positions[group]=trial; placed+=group; break
            else: raise RuntimeError('Could not place fragment')
        assert system.valid(positions,rotations)
        cfg.update(shape=str(shape_path),monomer_shape=str(ROOT/'examples/monomer-shape.json'),
                   boundary={'kind':'spherical','radius':radius},spherical_radius=radius,box_lengths=[2*radius]*3,
                   initial_poses=[pose(t,R) for t,R in zip(positions,rotations)],seed=518910+(arm=='seed-free'),
                   fixed_body_indices=[])
        # Preserve a stable explicit motif path for independent post-hoc analysis.
        motifs=ROOT/'examples/native-pair-motifs.json'
        if motifs.exists(): cfg.setdefault('metadata',{})['native_pair_motifs']=str(motifs)
        cfg.setdefault('metadata',{}).update(spherical_initialization='unwrapped preassembled fragments independently placed in equal-volume sphere',
            source_checkpoint=str(source/'checkpoint.json'),source_sha256=sha(source/'checkpoint.json'),initial_fragments=groups,
            sphere_volume_matches_original_periodic_volume=True)
        for name,gca,shift,aux in [('frozen',0,0,False),('frozen-gca-shift',1,1,False),('transport-gca-shift',1,1,True)]:
            config=copy.deepcopy(cfg);config.update(gca_probability=float(gca),center_shift_probability=float(shift))
            if aux:config['auxiliary_transport']={}
            path=inputs/f'{arm}-{name}.json';save(path,config)
            jobs.append(dict(id=f'{arm}-{name}',config=str(path),model=str(model),gca=gca,shift=shift,auxiliary=aux,
                             source_fragments=groups))
    save(out/'manifest.json',dict(jobs=jobs,model_sha256=sha(model),preparation_sha256=sha(Path(__file__)),
        scope='Matched starting fragments; finite validation pilot, not equilibrium or speedup measurement'))
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sweeps',type=int,default=40);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--backend',choices=['python','rust'],default='python');p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();a.out=a.out.resolve()
    jobs=json.loads((a.out/'manifest.json').read_text())['jobs'] if (a.out/'manifest.json').exists() else prepare(a.out)
    if a.prepare_only:return
    def one(job):
        dest=a.out/a.backend/job['id'];dest.parent.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():
            existing=json.loads((dest/'summary.json').read_text())
            if not existing.get('complete') or existing['completed_sweeps'] != a.sweeps:
                raise RuntimeError(f'Existing {job["id"]} has a different sweep budget; use a new output directory')
            return dict(id=job['id'],status='existing')
        if a.backend=='python':
            command=[sys.executable,str(ROOT/'research/spherical_ensemble.py'),'--config',job['config'],'--model',job['model'],
                     '--out',str(dest),'--sweeps',str(a.sweeps),'--sample-every','5','--gca-every',str(job['gca']),'--shift-every',str(job['shift'])]
            if job['auxiliary']:command+=['--auxiliary']
        else:
            command=[str(ROOT/'target/release/tetramer-mc'),'run','--config',job['config'],'--model',job['model'],
                     '--out',str(dest),'--sweeps',str(a.sweeps),'--sample-every','5']
        with (dest.parent/(job['id']+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row=dict(id=job['id'],returncode=result.returncode);print(json.dumps(row),flush=True)
        if result.returncode:raise RuntimeError(f'Failed {job["id"]}; see log')
        return row
    with ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,jobs))
    save(a.out/f'{a.backend}-completion.json',dict(jobs=results,sweeps=a.sweeps))


if __name__=='__main__':main()
