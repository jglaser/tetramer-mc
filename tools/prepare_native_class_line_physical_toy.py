#!/usr/bin/env python3
"""Freeze analytic-sphere physical validation inputs; no sampling."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from native_contact_regions import CRITERIA
from native_class_line_physical_reference import analytic_sphere_mass


def write(path, data):
    with Path(path).open('x') as out: out.write(json.dumps(data, indent=2, allow_nan=False)+'\n')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    root = Path(root).resolve()
    if root.exists(): raise ValueError('Fresh physical toy directory required')
    root.mkdir(); common = root/'common'; common.mkdir()
    identity = np.eye(3).tolist(); pose = lambda x:dict(position=x, orientation=[1.,0.,0.,0.])
    origin = pose([0.,0.,0.]); fixed = [origin]; members = [copy.deepcopy(origin) for _ in range(4)]
    shape = dict(name='analytic-sphere-four-coincident-rigid-members', volume=4*np.pi/3,
                 atoms=[dict(center=[0.,0.,0.], radius=1.) for _ in range(4)])
    write(common/'shape.json', shape); shape_sha = sha(common/'shape.json')
    compiled = dict(schema='native-entry-compiled-v1', source_definition_sha256='0'*64,
        source_input_sha256={'tetramer-shape.json':shape_sha}, criteria=CRITERIA, fixed_poses=fixed,
        members=[dict(position=[0.,0.,0.], rotation=identity) for _ in range(4)],
        monomer_atoms=[dict(center=[0.,0.,0.],radius=1.,residue=0)], residue_count=1,
        references=[dict(label='toy',family='toy',position=[2.,0.,0.],rotation=identity,native_residue_pairs=[0])],
        motifs=[dict(id=0,position=[2.,0.,0.],rotation=identity,
                     member_contacts=[dict(member_i=i,member_j=i,directed_class='toy') for i in range(4)])])
    write(common/'compiled-native.json', compiled)
    metadata = dict(native_poses=[pose([2.,0.,0.])],rigid_members=members,member_error_scale=2.,angle_error_scale_deg=15.)
    chart = dict(shape_sha256=shape_sha,coordinate_convention='anchor-body-relative',angular_length=8.,weights=[1.],
                 anchors=[dict(position=[0.,0.,0.],rotation=identity)],means=[[0.]*6],covariances=[np.eye(6).tolist()])
    channels = [{'class':'hard_free','probability':.2}, {'class':'contact_without_native','probability':.2},
                {'class':'contact_without_native','probability':.2,'orthant':22},
                {'class':'contact_without_native','probability':.2,'orthant':62},
                {'class':'native','probability':.2,'orthant':55}]
    components = [dict(weight=.6,mean=[2.5,0.,0.,0.,0.,0.],covariance=np.eye(6).tolist()),
                  dict(weight=.4,mean=[-2.5,0.,0.,0.,0.,0.],covariance=np.eye(6).tolist())]
    jobs, analytic = [], {}
    for ai, activity in enumerate([0.,.3]):
        folder = root/f'z{activity:g}'; folder.mkdir()
        config = dict(shape=str(common/'shape.json'),fixed_poses=fixed,initial_pose=pose([3.,0.,0.]),
            capture_center=[0.,0.,0.],capture_radius=10.,depletant_radius=.5,reservoir_density=activity,
            poisson_lambda_ratio=64.,translation_steps=[.1],rotation_steps_deg=[1.],rotation_probability=.5,
            local_attempts_per_cycle=1,uniform_probability=.5,seed=1,metadata=metadata,
            endpoint_gate=dict(max_cells=127,max_depth=6,min_width=0.))
        write(folder/'config.json',config)
        region = dict(shape_sha256=shape_sha,fixed_neighbor=origin,physical_fixed_neighbors=fixed,
            capture_center=[0.,0.,0.],capture_radius=10.,activity=activity,depletant_radius=.5,
            physical_metric=metadata,minimum_original_q=0.,mahalanobis_radius=4.,gaussian_chart=chart)
        write(folder/'region.json',region)
        guide = dict(schema='defensive-native-class-line-guide-v1',region_sha256=sha(folder/'region.json'),
            defensive_uniform_shell_probability=.5,gaussian_components=components,raw_translation_axes=[0,1,2],
            conditional_probability=1.,minimum_conditional_mass=1e-12,class_channels=channels,
            compiled_native=dict(path=str(common/'compiled-native.json'),sha256=sha(common/'compiled-native.json')),
            shape_sha256=shape_sha,fixed_poses=fixed,capture_center=[0.,0.,0.],capture_radius=10.,depletant_radius=.5)
        for mi, method in enumerate(['hard-free','class']):
            spec=copy.deepcopy(guide)
            if method=='hard-free':spec['class_channels']=[{'class':'hard_free','probability':1.}]
            write(folder/f'{method}-guide.json',spec)
            for pop in range(4):
                ident=f'z{activity:g}-{method}-p{pop}'
                jobs.append(dict(id=ident,activity=activity,method=method,population=pop,samples=2048,
                    seed=6100403001+1000*ai+100*mi+pop,config=str(folder/'config.json'),region=str(folder/'region.json'),
                    guide=str(folder/f'{method}-guide.json'),out=str(root/'populations'/ident),
                    cloud_replicates=2,lambda_ratio=64.,cpu_limit_seconds=1200,wall_limit_seconds=2400))
        analytic[str(activity)] = analytic_sphere_mass(1.,.5,activity,translation_scale=1.,angular_scale=1.,angular_length=8.,capture_radius=10.)
    analytic['hard'] = analytic_sphere_mass(1.,.5,0.,translation_scale=1.,angular_scale=1.,angular_length=8.,capture_radius=10.)
    write(root/'analytic-reference.json', dict(parameters=dict(core_radius=1.,depletant_radius=.5,latent_radius=4.,
        translation_scale=1.,angular_scale=1.,angular_length=8.,capture_radius=10.), references=analytic,
        geometry='Four coincident monomer spheres form exactly one physical union sphere; no inter-member self-core checks apply. Native labels only partition the proposal.',
        target='Full R4 in translation/Cayley chart; original q>=0, capture contains the complete r<=4 translation ball; no wall restriction.',
        measure='J=1/(8^3*pi^2)*(1+||u_rot||^2/8^2)^-2; orientation cap depends on translation radius through the six-ball.'))
    files={str(p):sha(p) for p in root.rglob('*.json')}
    for name in ['prepare_native_class_line_physical_toy.py','native_class_line_physical_reference.py','native_class_line_reference.py']:
        p=Path(__file__).with_name(name).resolve();files[str(p)]=sha(p)
    allocation=dict(schema='native-class-line-physical-sphere-validation-v1',jobs=jobs,total_attempts=32768,
        independent_populations_per_arm=4,draws_per_population=2048,maximum_workers=1,protein_queries=0,
        analytic_reference=str(root/'analytic-reference.json'),source_and_input_sha256=files,
        criteria=dict(reference_mass_combined_SE_limit=5.,proposal_difference_combined_SE_limit=5.,
            hard_depletion_difference_combined_SE_limit=5.,cloud_count_or_weight_z_limit=5.,
            confidence_note='Reference-limit validation at fixed allocation; stochastic discrepancies remain visible, no adaptive extension or retry.'),
        execution_ready=False,scope='Independent analytic sphere hard-only/depletion integration; no native thermodynamic or assembly inference.')
    write(root/'allocation.json',allocation)
    return allocation


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=prepare(a.out);print(json.dumps(dict(jobs=len(result['jobs']),total_attempts=result['total_attempts'],physical_draws_launched=0)))
