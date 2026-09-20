#!/usr/bin/env python3
"""Audit selected endpoints of the new dominant explicit-far SMC population."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import resource
import shutil
import subprocess
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from audit_old_new_geometry import aabb, read, save, sha, rotation
from audit_previous_smc_regions import recompute_q

ROOT=Path(__file__).resolve().parents[1]

def symmetry_audit(env,shape,particles):
    members=env['rigid_members'];centers=np.array([p['position'] for p in members]);center=centers.mean(axis=0)
    coords=np.array([a['center'] for a in shape['atoms']]);radii=np.array([a['radius'] for a in shape['atoms']])
    groups=[np.flatnonzero(radii==r) for r in np.unique(radii)]
    candidates=[];valid=[]
    for perm in itertools.permutations(range(4)):
        target=centers[list(perm)];u,s,vt=np.linalg.svd((centers-center).T@(target-center))
        row=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
        rot=Rotation.from_matrix(row.T);translation=center-rot.apply(center)
        err=np.linalg.norm(rot.apply(centers)+translation-target,axis=1)
        angular=[(rotation(members[perm[i]]).inv()*rot*rotation(members[i])).magnitude() for i in range(4)]
        moved=rot.apply(coords)+translation;maxatom=0.;bijection=True
        for indices in groups:
            distance,nearest=cKDTree(coords[indices]).query(moved[indices])
            maxatom=max(maxatom,float(distance.max()));bijection &= len(np.unique(nearest))==len(indices)
        accepted=bool(err.max()<1e-7 and max(angular)<1e-7 and maxatom<1e-7 and bijection)
        q=rot.as_quat();pose=dict(position=translation.tolist(),orientation=q[[3,0,1,2]].tolist())
        item=dict(permutation=perm,member_max_error_A=float(err.max()),member_rotation_max_error_deg=float(np.degrees(max(angular))),
            radius_matched_atom_max_distance_A=maxatom,radius_matched_nearest_is_bijective=bool(bijection),exact_symmetry=accepted,pose=pose)
        candidates.append(item)
        if accepted:valid.append(pose)
    equivalents=[]
    for ref in env['native_poses']:
        for symmetry in valid:
            r=rotation(ref)*rotation(symmetry);q=r.as_quat()
            equivalents.append(dict(position=(rotation(ref).apply(symmetry['position'])+ref['position']).tolist(),orientation=q[[3,0,1,2]].tolist()))
    equivalence_env=dict(env,native_poses=equivalents)
    qeq=[recompute_q(p['pose'],equivalence_env,{}) for p in particles]
    nearby=env['native_capture_certificate']['nearby_placements']
    crystal_env=dict(env,native_poses=[dict(position=p['position'],orientation=p['orientation']) for p in nearby])
    crystalq=[recompute_q(p['pose'],crystal_env,{}) for p in particles]
    return dict(permutations=candidates,exact_symmetry_count=len(valid),
        equivalent_native_q_range=[min(qeq),max(qeq)],listed_crystallographic_q_range=[min(crystalq),max(crystalq)],
        listed_crystallographic_placements=len(nearby),inside_capture_placements=[p for p in nearby if p['center_distance_A']<=env['capture_radius']],
        nearest_alternative_center_A=env['native_capture_certificate']['nearest_alternative_center_distance_A'],
        scope='All 24 proper Kabsch fits of ordered member-center permutations; a body symmetry additionally requires member orientations and radius-preserving full-atom bijection to agree within 1e-7. Nearby crystallographic placements are from the archived capture certificate, not a fresh exhaustive crystallographic enumeration.')

def main(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);archive=out/'provenance';archive.mkdir()
    source=args.population.resolve();summary=read(source/'summary.json');counts=read(source/'final_overlap_counts.json')
    config_path=source.parents[1]/'configs'/(source.name+'.json');config=read(config_path);env=read(config['environment']);shape=read(config['shape'])
    assert summary['complete'];particles=summary['final_particles'];physical_q=np.array([recompute_q(p['pose'],env,{}) for p in particles])
    assert np.max(np.abs(physical_q-5*np.array([p['q'] for p in particles])))<1e-10
    distances=np.array([np.linalg.norm(np.array(p['pose']['position'])-env['capture_center']) for p in particles])
    assert max(distances)<=env['capture_radius'] and min(physical_q)>5
    count=np.array([p['count'] for p in counts['samples']]);indices=[]
    for i in [*np.argsort(count)[-3:][::-1],np.argmin(physical_q),np.argmax(physical_q),np.argsort(physical_q)[len(physical_q)//2]]:
        if int(i) not in indices:indices.append(int(i))
    reference=ROOT/'runs/old-new-geometry-audit-v2';prior_input=read(reference/'input.json')
    shutil.copy2(config['shape'],archive/'shape.json')
    cases=[]
    for i,index in enumerate(indices):
        p=particles[index];lower,upper=aabb(shape,p['pose'],1.5);fixedbounds=[aabb(shape,f,1.5) for f in env['fixed_poses']]
        lower=np.maximum(lower,np.min([b[0] for b in fixedbounds],axis=0));upper=np.minimum(upper,np.max([b[1] for b in fixedbounds],axis=0))
        bound=max(np.linalg.norm(a['center'])+a['radius']+1.5 for a in shape['atoms'])
        for fixed in env['fixed_poses']:assert min(config['box_lengths'])-np.linalg.norm(np.array(fixed['position'])-p['pose']['position'])>2*bound
        cases.append(dict(id=source.name+f'-p{index:03d}',pose=p['pose'],fixed=env['fixed_poses'],box_lengths=config['box_lengths'],bounds_lo=lower.tolist(),bounds_hi=upper.tolist(),
            seed=89770131+i*101,source_particle=index,physical_q=physical_q[index],capture_distance_A=distances[index],saved_overlap_count=int(count[index]),saved_count_intensity=counts['count_intensity']))
    save(out/'input.json',dict(shape=str(archive/'shape.json'),rd=1.5,activity=.035,common_points=1048576,cloud_repeats=32,lambda_ratio=64,gate_options=prior_input['gate_options'],cases=cases))
    symmetry=symmetry_audit(env,shape,particles);save(out/'symmetry.json',symmetry)
    save(out/'population.json',dict(logZ=summary['logZ'],family_ess=summary['family_ess'],distinct_families=summary['distinct_families'],
        physical_q_range=[float(min(physical_q)),float(max(physical_q))],capture_distance_range_A=[float(min(distances)),float(max(distances))],
        saved_count_intensity=counts['count_intensity'],saved_count_mean=float(count.mean()),saved_count_range=[int(count.min()),int(count.max())],
        saved_population_mean_C_A3=float(count.mean()/counts['count_intensity']),
        scope='Independent conditional counts are noisy per endpoint. Final endpoints descend from one family and do not provide independent equilibrium replication.'))
    binary=reference/'provenance/old-new-geometry-probe';shutil.copy2(binary,archive/binary.name)
    used=[Path(__file__),source/'summary.json',source/'final_overlap_counts.json',config_path,Path(config['environment']),Path(config['shape']),binary,reference/'provenance.json']
    for name in ['summary.json','final_overlap_counts.json']:shutil.copy2(source/name,archive/name)
    shutil.copy2(config_path,archive/'config.json');shutil.copy2(config['environment'],archive/'environment.json');shutil.copy2(__file__,archive/Path(__file__).name)
    save(out/'provenance.json',dict(source_sha256={str(p):sha(p) for p in used},selection='Top three saved noisy overlap counts plus minimum, maximum and median physical q. Subsequent independent volumes remove ranking noise from measurements. All original data unchanged.'))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
    with (out/'probe.log').open('w') as log:subprocess.run([str(archive/binary.name),str(out/'input.json'),str(out/'results.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    after=resource.getrusage(resource.RUSAGE_CHILDREN);save(out/'timing.json',dict(wall_seconds=time.monotonic()-start,cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime))
    save(out/'artifacts.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifacts.json'})
    print(json.dumps(read(out/'results.json'),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'runs/explicit-far-endpoint-audit')
    p.add_argument('--population',type=Path,default=ROOT/'runs/explicit-far-smc-plan-20260920/production/site0-far5-r1.5-z0.035-n512-r1')
    main(p.parse_args())
