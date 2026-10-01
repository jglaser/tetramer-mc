#!/usr/bin/env python3
"""Freeze small sphere/tiny-union guide inputs, without executing a sampler."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=a.out.resolve();assert not out.exists();out.mkdir(parents=True)
    source=root/'results/contact-arc-reference-inputs-20261001/two_spheres'
    basecfg=read(source/'config.json');basereg=read(source/'region.json');baseguide=read(source/'uniform_phi.json')
    modes=[('sphere_x',[0],1.,.5,1e-12),('sphere_xyz',[0,1,2],1.,.5,1e-12),('sphere_beta0',[0,1,2],0.,.5,1e-12),
        ('sphere_uniform',[0,1,2],1.,1.,1e-12),('sphere_floor',[0,1,2],1.,.5,.999999999999),('tiny_union_tightcapture',[0,1,2],1.,.5,1e-12)]
    probes=[([0,0,0],[0,0,0]),([.04,0,0],[0,0,0]),([0,3.,0],[0,0,0]),([0,0,3.],[0,0,0]),
        ([0,2.,1.],[.3,-.2,.1]),([-.5,0,0],[0,0,0]),([.5,0,0],[0,0,0]),([8.,0,0],[0,0,0]),
        ([0,8.,0],[0,0,0]),([0,0,0],[10.,0,0]),([.01,0,0],[.1,.2,-.1])]
    jobs=[]
    for mi,(name,axes,beta,alpha,floor) in enumerate(modes):
        path=out/name;path.mkdir();cfg=copy.deepcopy(basecfg);region=copy.deepcopy(basereg)
        shape=dict(atoms=[dict(center=[0.,0.,0.],radius=1.)])
        tight=name=='tiny_union_tightcapture'
        if tight:shape['atoms'] += [dict(center=[.4,.1,0.],radius=.7),dict(center=[-.2,.1,.3],radius=.5)]
        write(path/'shape.json',shape);cfg['shape']=str(path/'shape.json');separation=1.2 if tight else 2.05
        cfg['fixed_poses'][0]['position']=[-separation,0.,0.];cfg['fixed_poses'][1]['position']=[separation,0.,0.]
        cfg['capture_radius']=.1 if tight else 10.;cfg['initial_pose']['position']=[0.,0.,0.] if tight else [0.,3.,0.]
        cfg['metadata']=dict(reference='hard-free-line-sphere-and-tiny-union-v1',scenario=name,no_physical_weights=True)
        write(path/'config.json',cfg)
        region['fixed_neighbor']=copy.deepcopy(cfg['fixed_poses'][0]);region['physical_fixed_neighbors']=copy.deepcopy(cfg['fixed_poses'])
        region['gaussian_chart']['anchors'][0]['position']=[separation,0.,0.]
        region['gaussian_chart']['shape_sha256']=sha(path/'shape.json')
        region.update(capture_radius=cfg['capture_radius'],physical_metric=cfg['metadata'],shape_sha256=sha(path/'shape.json'))
        write(path/'region.json',region)
        guide=dict(schema='defensive-hard-free-line-guide-v1',region_sha256=sha(path/'region.json'),defensive_uniform_shell_probability=alpha,
            conditional_probability=beta,minimum_conditional_mass=floor,raw_translation_axes=axes,
            gaussian_components=[] if alpha==1 else baseguide['gaussian_components'])
        write(path/'guide.json',guide)
        L=np.linalg.cholesky(region['gaussian_chart']['covariances'][0]);mean=np.array(region['gaussian_chart']['means'][0])
        rows=[dict(id=f'{name}/fixed-{i:02}',latent=np.linalg.solve(L,np.r_[position,angles]-mean).tolist()) for i,(position,angles) in enumerate(probes)]
        (path/'probes.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
        jobs.append(dict(name=name,config=str(path/'config.json'),region=str(path/'region.json'),guide=str(path/'guide.json'),probes=str(path/'probes.jsonl'),samples=128,seed=610020100+mi,probes_count=len(rows)))
    allocation=dict(schema='hard-free-line-reference-allocation-v1',jobs=jobs,samples=128*len(jobs),probes=len(probes)*len(jobs),new_Poisson_clouds=0,protein_jobs=0,
        scope='Six fixed toy arms;128 proposal-only draws and11 deterministic queries each. No protein pose or physical-weight sampling.',preparer_sha256=sha(__file__))
    write(out/'allocation.json',allocation);write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}));print(out/'allocation.json')

if __name__=='__main__':main()
