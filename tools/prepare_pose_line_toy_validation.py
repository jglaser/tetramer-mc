#!/usr/bin/env python3
"""Freeze small reference inputs; no jobs, protein poses, or cloud sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quat(c):
    q = np.r_[1., c]
    return q/np.linalg.norm(q)


def rotation(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def prepare(out):
    if out.exists():raise ValueError('Fresh reference directory required')
    out.mkdir(parents=True)
    sources = out/'preparation-source.py'
    sources.write_bytes(Path(__file__).read_bytes())
    shapes = {
        'sphere':[(np.zeros(3),.4)],
        'dumbbell':[(np.array([x,0.,0.]),.4) for x in (-.8,.8)],
        'asymmetric':[(np.array([.45*i,.35*j,.3*k]),.22+.025*((i+j+k)%3))
            for i in (-1,0,1) for j in (-1,1) for k in (0,1)],
    }
    fixed = dict(position=[7.,-4.,3.], orientation=quat([.2,-.1,.3]).tolist())
    frame = rotation(fixed['orientation'])
    anchor = np.array([1.8,.15,0.])
    center = np.array(fixed['position'])+frame@anchor
    other = dict(position=(np.array(fixed['position'])+frame@np.array([3.4,.2,.1])).tolist(),
        orientation=quat([-.3,.2,.1]).tolist())
    lower = np.diag([.6,.7,.65,.8,.9,.75])
    lower[1,0]=.15; lower[3,0]=.25; lower[4,1]=-.2; lower[5,2]=.3
    component_lower = np.eye(6)
    component_lower[0,0]=.8; component_lower[3,1]=.2; component_lower[4,2]=-.15
    components = [dict(weight=.6,mean=[0.]*6,covariance=np.eye(6).tolist()),
        dict(weight=.4,mean=[.25,-.2,.1,.35,-.1,.2],covariance=(component_lower@component_lower.T).tolist())]
    jobs=[]
    for index,(name,atoms) in enumerate(shapes.items()):
        root=out/name;root.mkdir()
        write(root/'shape.json',dict(name=name,volume=1.,atoms=[dict(center=p.tolist(),radius=r)for p,r in atoms]))
        shape_sha=digest(root/'shape.json')
        capture=1.7 if name=='asymmetric' else 5.
        metadata=dict(native_poses=[fixed],rigid_members=[dict(position=[0.]*3,orientation=[1.,0.,0.,0.])],
            member_error_scale=3.,angle_error_scale_deg=90.)
        config=dict(shape='shape.json',fixed_poses=[fixed,other],initial_pose=dict(position=center.tolist(),orientation=[1.,0.,0.,0.]),
            capture_center=center.tolist(),capture_radius=capture,depletant_radius=.5,reservoir_density=.4,
            poisson_lambda_ratio=16.,translation_steps=[.1],rotation_steps_deg=[5.],rotation_probability=.5,
            local_attempts_per_cycle=1,uniform_probability=.5,seed=71620+index,metadata=metadata)
        write(root/'config.json',config)
        chart=dict(shape_sha256=shape_sha,coordinate_convention='anchor-body-relative',angular_length=1.3,
            weights=[1.],anchors=[dict(position=anchor.tolist(),rotation=rotation(quat([-.15,.25,.1])).tolist())],
            means=[[0.]*6],covariances=[(lower@lower.T).tolist()])
        region=dict(shape_sha256=shape_sha,fixed_neighbor=fixed,physical_fixed_neighbors=[fixed,other],
            capture_center=center.tolist(),capture_radius=capture,activity=.4,depletant_radius=.5,
            physical_metric=metadata,minimum_original_q=0.,mahalanobis_radius=4.,gaussian_chart=chart)
        write(root/'region.json',region)
        guide=dict(schema='defensive-hard-free-pose-line-guide-v1',region_sha256=digest(root/'region.json'),
            defensive_uniform_shell_probability=.5,gaussian_components=components,raw_pose_axes=list(range(6)),
            conditional_probability=1.,minimum_conditional_mass=1e-12)
        write(root/'guide.json',guide)
        jobs.append(dict(id=name,draws=512,seed=71620+index,config=str(root/'config.json'),region=str(root/'region.json'),
            guide=str(root/'guide.json'),out=str(root/'draws')))
    inputs={str(p.relative_to(out)):digest(p) for p in out.glob('*/*.json')}
    write(out/'allocation.json',dict(schema='pose-line-toy-validation-v1',total_draws=1536,jobs=jobs,input_sha256=inputs,
        preparation_sha256=digest(sources),new_protein_draws=0,new_Poisson_clouds=0,max_workers=1,
        scope='Fixed toy allocation before execution; no retries or adaptive extension. Independent density/geometry audit required.'))
    return jobs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(dict(jobs=prepare(a.out.resolve()))))
