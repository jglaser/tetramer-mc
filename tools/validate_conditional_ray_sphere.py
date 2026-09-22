#!/usr/bin/env python3
"""End-to-end sphere/AO and Haar quadrature control of the conditional-ray guide.

Two fixed-N toy runs at z=0 and z=2; these are validation, not protein data.
Every saved Rust proposal density is reconstructed with the independent Python
implementation before comparison against one-dimensional physical quadrature.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from scipy.integrate import quad
from analyze_latent_region import analyze
from prepare_shoulder_docking_benchmark import local_dependencies
from run_mobile_posterior_pilot import read,write,sha,require,verify_bundle

ROOT=Path(__file__).resolve().parents[1]
BINARY=ROOT/'target/conditional-ray-review/release/latent-region-normalizer'
BUNDLE=ROOT/'target/conditional-ray-review/release/build/tetramer-mc-356a8e0b913327fe/out/source-bundle.json'
RADIUS=2.4


def reference(activity):
    def f(t):
        angle=math.sqrt(max(0.,RADIUS*RADIUS-t*t))
        primitive=.5*(math.atan(angle)-angle/(1+angle*angle))
        overlap=math.pi*(2.8+t)*(1.4-t)**2/12 if t<1.4 else 0.
        return 16*t*t*primitive*math.exp(activity*overlap)
    a,ea=quad(f,.6,1.4,epsabs=1e-11,epsrel=1e-11)
    b,eb=quad(f,1.4,RADIUS,epsabs=1e-11,epsrel=1e-11)
    return a+b,ea+eb


def validate(out):
    out=Path(out).resolve();require(not out.exists(),'Fresh validation directory required')
    bundle,_=verify_bundle(BINARY,BUNDLE,ROOT);out.mkdir(parents=True)
    archive=out/'provenance';archive.mkdir()
    for name,path in {'latent-region-normalizer':BINARY,'source-bundle.json':BUNDLE,
        **local_dependencies([Path(__file__)])}.items():shutil.copy2(path,archive/name)
    for name,item in bundle['files'].items():
        p=archive/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(item['text'])
    write(out/'validation-plan.json',dict(samples_per_run=16384,seeds=[131100011,131100012],activities=[0.,2.],
        binary_sha256=sha(BINARY),bundle_sha256=sha(BUNDLE),tests='Same sphere geometry; analytic AO volume with normalized SO(3) Haar. Two clouds, lambda/z64. Fixed N, no retries.'))
    results=[]
    for index,z in enumerate((0.,2.)):
        root=out/f'z{index}';prov=root/'provenance';prov.mkdir(parents=True)
        shape=dict(name='sphere',volume=4*math.pi*.3**3/3,atoms=[dict(center=[0.,0.,0.],radius=.3)])
        write(prov/'shape.json',shape)
        fixed=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]);native=dict(position=[1.,0.,0.],orientation=[1.,0.,0.,0.])
        metadata=dict(native_poses=[native],rigid_members=[fixed],member_error_scale=1.,angle_error_scale_deg=15.)
        config=dict(shape=str(prov/'shape.json'),fixed_poses=[fixed],initial_pose=native,capture_center=[0.,0.,0.],capture_radius=4.,
            depletant_radius=.4,reservoir_density=z,poisson_lambda_ratio=64.,translation_steps=[.1],rotation_steps_deg=[1.],
            rotation_probability=.5,local_attempts_per_cycle=1,uniform_probability=.1,seed=1,metadata=metadata)
        write(prov/'config.json',config)
        region=dict(fixed_neighbor=fixed,physical_fixed_neighbors=[fixed],capture_center=[0.,0.,0.],capture_radius=4.,
            shape_sha256=sha(prov/'shape.json'),activity=z,depletant_radius=.4,physical_metric=metadata,
            minimum_original_q=0.,minimum_original_q_inclusive=True,minimum_mahalanobis_radius=0.,mahalanobis_radius=RADIUS,
            gaussian_chart=dict(shape_sha256=sha(prov/'shape.json'),angular_length=1.,coordinate_convention='anchor-body-relative',
                anchors=[dict(position=[0.,0.,0.],rotation=np.eye(3).tolist())],means=[[0.]*6],covariances=[np.eye(6).tolist()],weights=[1.]))
        write(prov/'region.json',region)
        guide=dict(schema='defensive-conditional-ray-guide-v1',region_sha256=sha(prov/'region.json'),defensive_uniform_shell_probability=.5,
            inner_radius=.6,widths=[.1,.8,1.8],interfaces=[dict(moving_members=[[0.,0.,0.]],target_world_members=[[0.,0.,0.]])])
        write(prov/'importance-guide.json',guide)
        shutil.copy2(BINARY,prov/'latent-region-normalizer');shutil.copy2(BUNDLE,prov/'source-bundle.json')
        job=dict(id='r00',seed=131100011+index,samples=16384,directory=str(root/'runs/r00'))
        manifest=dict(schema='importance-latent-region-campaign-v3',jobs=[job],workers=1,lambda_ratio=64.,cloud_replicates=2,
            region_sha256=sha(prov/'region.json'),importance_guide_sha256=sha(prov/'importance-guide.json'),
            archive_sha256={p.name:sha(p)for p in prov.iterdir()if p.is_file()})
        write(root/'manifest.json',manifest)
        argv=[str(prov/'latent-region-normalizer'),'--config',str(prov/'config.json'),'--region',str(prov/'region.json'),
            '--importance-guide',str(prov/'importance-guide.json'),'--out',job['directory'],'--samples','16384',
            '--seed',str(job['seed']),'--cloud-replicates','2','--lambda-ratio','64']
        with(root/'physical.log').open('xb')as stream:physical=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
        write(root/'status.json',dict(returncode=physical.returncode,command=argv));physical.check_returncode()
        # The just-frozen auditor closure, not a later source edit, is executed.
        with(root/'audit.log').open('xb')as stream:
            audited=subprocess.run([sys.executable,'-B',str(archive/'analyze_latent_region.py'),'--root',str(root)],stdout=stream,stderr=subprocess.STDOUT)
        audited.check_returncode();analysis=read(root/'assessment/analysis.json')
        checks=[]
        for label,activity,key in [('Qz',z,'estimate'),('Q0',0.,'hard_region')]:
            exact,error=reference(activity);estimate=analysis[key];mean=math.exp(estimate['logQ']);se=mean*estimate['relative_se']
            require(abs(mean-exact)<6.5*se+1e-7,'Sphere physical quadrature disagreement')
            checks.append(dict(estimand=label,estimate=mean,quadrature=exact,quadrature_error=error,observed_SE=se,
                standardized_error=(mean-exact)/se,passed=True))
        sampling=analysis['importance_sampling'];require(sampling['shell_rejected']==0 and sampling['branch_counts']['conditional-ray']>0,'Missing in-ball or guided probes')
        results.append(dict(activity=z,samples=16384,physical_returncode=physical.returncode,audit_returncode=audited.returncode,
            analysis_sha256=sha(root/'assessment/analysis.json'),checks=checks,proposal_audit=sampling,
            sampler_cpu_seconds=analysis['sampler_cpu_seconds']))
    record=dict(schema='conditional-ray-sphere-crosslanguage-v1',complete=True,results=results,binary_sha256=sha(BINARY),
        source_bundle_sha256=sha(BUNDLE),total_unconditional_draws=32768,
        source_sha256={p.relative_to(archive).as_posix():sha(p)for p in archive.rglob('*')if p.is_file()},
        scope='Analytic sphere depletion and Haar-measure validation of the new proposal, with exact independent full-density audit. No protein estimates or assembly claims.')
    write(out/'validation.json',record)
    print(json.dumps(dict(complete=True,out=str(out),validation_sha256=sha(out/'validation.json'),results=results)))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();validate(args.out)
