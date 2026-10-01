#!/usr/bin/env python3
"""Freeze deterministic toy arc-score inputs; never executes a scorer."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import numpy as np
from contact_arc_density_reference import require,sha,read,dependency_sha256


def prepare(source,out):
    source=Path(source).resolve();out=Path(out).resolve();require(not out.exists(),'Reference inputs must be new')
    frozen=read(source/'manifest.json')
    for name,h in frozen['files'].items():require(sha(source/name)==h,'Source reference input changed')
    original_config=read(source/'config.json');original_region=read(source/'region.json')
    original_guides=[read(source/'guides'/f'{name}.json') for name in ['uniform_phi','localized_phi']]
    out.mkdir(parents=True);(out/'shape.json').write_bytes((source/'shape.json').read_bytes());shape_hash=sha(out/'shape.json')
    write=lambda path,data:path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    jobs=[]
    for name,separation,third in [('two_spheres',2.4,None),('partial_spectator',2.4,[0.,0.,3.]),('blocked_spectator',4.02,[0.,0.,0.])]:
        directory=out/name;directory.mkdir();config=copy.deepcopy(original_config);region=copy.deepcopy(original_region)
        pose=lambda p:dict(position=p,orientation=[1.,0.,0.,0.])
        config.update(shape=str(out/'shape.json'),fixed_poses=[pose([-separation/2,0.,0.]),pose([separation/2,0.,0.])],initial_pose=pose([0.,5.,0.]))
        if third is not None:config['fixed_poses'].append(pose(third))
        config['metadata']=dict(reference='deterministic-complete-contact-arc-score-v1',scenario=name,no_pose_draws=True,no_physical_weights=True)
        region.update(fixed_neighbor=config['fixed_poses'][0],physical_fixed_neighbors=config['fixed_poses'],physical_metric=config['metadata'],
            shape_sha256=shape_hash,definition='Deterministic sphere queries for complete normalized arc-density validation; no draws or physical estimates.')
        region['gaussian_chart']['anchors'][0]['position']=[separation/2,0.,0.];region['gaussian_chart']['shape_sha256']=shape_hash
        write(directory/'config.json',config);write(directory/'region.json',region)
        guide_paths=[]
        for label,old in zip(['uniform_phi','localized_phi'],original_guides):
            guide=copy.deepcopy(old);guide['region_sha256']=sha(directory/'region.json');path=directory/(label+'.json');write(path,guide);guide_paths.append(str(path))
        L=np.linalg.cholesky(region['gaussian_chart']['covariances'][0]);probes=[]
        for r in [2.015,2.2]:
            rho=math.sqrt(r*r-(separation/2)**2)
            for angle_id,a in enumerate([[0.,.2,-.1],[.3,-.2,.1]]):
                for pi,phi in enumerate([0.,1.3,3.4]):
                    raw=np.array([0.,-rho*math.sin(phi),rho*math.cos(phi),*a])
                    probes.append(dict(id=f'{name}/r{r}/angle{angle_id}/phi{pi}',latent=np.linalg.solve(L,raw).tolist()))
        for label,t in [('axis',[0.,0.,0.]),('outside_width',[0.,6.,0.]),('axis_far',[3.,0.,0.])]:
            probes.append(dict(id=f'{name}/{label}',latent=np.linalg.solve(L,np.r_[t,[.2,-.1,.3]]).tolist()))
        (directory/'probes.jsonl').write_text(''.join(json.dumps(p,allow_nan=False)+'\n' for p in probes))
        job=dict(name=name,config=str(directory/'config.json'),shape=str(out/'shape.json'),region=str(directory/'region.json'),guides=guide_paths,
            probes=str(directory/'probes.jsonl'),minimum_arc_mass=1e-12,queries=len(probes),density_evaluations=2*len(probes))
        jobs.append(job)
        if name=='partial_spectator':jobs.append(dict(job,name='partial_spectator_floor1',minimum_arc_mass=1.))
    allocation=dict(complete=True,schema='contact-arc-deterministic-reference-allocation-v1',jobs=jobs,
        queries=sum(j['queries'] for j in jobs),density_evaluations=sum(j['density_evaluations'] for j in jobs),
        new_pose_draws=0,new_Poisson_clouds=0,source_reference_inputs=str(source),source_manifest_sha256=sha(source/'manifest.json'),
        scope='Fixed toy queries for two spheres, a partially blocking spectator, fully blocked circles, null-axis/outside-width queries and floor1 old-law identity.')
    write(out/'allocation.json',allocation);archive=out/'provenance';archive.mkdir()
    for path in [Path(__file__),*map(Path,dependency_sha256())]:shutil.copy2(path,archive/path.name)
    files=[p for p in out.rglob('*') if p.is_file()];write(out/'manifest.json',dict(complete=True,files={str(p.relative_to(out)):sha(p) for p in files}))
    return allocation


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    print(json.dumps(prepare(a.source,a.out),indent=2))
