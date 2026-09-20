#!/usr/bin/env python3
"""Exercise original-q window launcher and audit against a sphere/Haar integral."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pose(position):return dict(position=position,orientation=[1.,0.,0.,0.])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=args.out.resolve();binary=args.binary.resolve()
    assert not root.exists();root.mkdir(parents=True)
    here=Path(__file__).resolve().parent
    shape=root/'sphere.json';save(shape,dict(name='q-window reference sphere',volume=4*math.pi/3,
        atoms=[dict(center=[0.,0.,0.],radius=1.)]))
    center=[3.,-2.,1.];anchor=pose([50.,0.,0.]);a=.5;angle=math.pi/4
    cfg=dict(shape=str(shape),fixed_poses=[anchor,pose([0.,50.,0.])],capture_center=center,capture_radius=2.,
        depletant_radius=.5,reservoir_density=0.,poisson_lambda_ratio=64.,
        metadata=dict(native_poses=[pose(center)],rigid_members=[pose([0.,0.,0.])],
            member_error_scale=a,angle_error_scale_deg=45.))
    config=root/'config.json';save(config,cfg);config_sha=sha(config)
    covariance=[[float(i==j)*(.5**2 if i<3 else .25**2) for j in range(6)] for i in range(6)]
    model=root/'model.json';save(model,dict(coordinate_convention='anchor-body-relative',angular_length=1.,
        shape_sha256=sha(shape),anchors=[dict(position=[center[i]-anchor['position'][i] for i in range(3)],
            rotation=[[float(i==j) for j in range(3)] for i in range(3)])],means=[[0.]*6],
        covariances=[covariance],weights=[1.]))
    base=[sys.executable,str(here/'run_native_region_reference.py'),'--config',str(config),'--binary',str(binary),
        '--replicates','2','--samples','512','--workers','2','--seed-base','7362201','--expected-binary-sha256',sha(binary)]
    subprocess.run(base+['--root',str(root/'default-prepared'),'--prepare-only'],check=True,capture_output=True,text=True)
    default=json.loads((root/'default-prepared/manifest.json').read_text())
    assert 'q_window' not in default and all('--q-min' not in j['command'] for j in default['jobs'])
    window=dict(minimum=1.,maximum=2.,lower_inclusive=False,upper_inclusive=False)
    controls={}
    for label,extra in [('uniform',[]),('guided-mixture',['--model',str(model),'--cover-scales','.5,1'])]:
        campaign=root/label
        command=base+['--root',str(campaign),'--q-min','1','--q-max','2','--q-lower-open','--q-upper-open']+extra
        with (root/f'{label}-runner.log').open('x') as log:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        analyzer=campaign/'provenance/analyze_native_region_reference.py'
        with (root/f'{label}-analysis.log').open('x') as log:
            subprocess.run([sys.executable,str(analyzer),str(campaign),'--workers','2'],check=True,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((campaign/'assessment-streaming.json').read_text())
        assert result['q_window']==window and result['samples']==1024 and result['independent_q_and_density_rows']==1024
        assert result['raw_cloud_points']==0 and result['regions']==result['hard_regions']
        assert result['regions']['region_core']['nonzero']==0 and 'native' not in result['regions']
        exact=4*(2*a)**3/3*(2*angle-math.sin(2*angle))-4*a**3/3*(angle-math.sin(angle))
        observed=result['regions']['region'];value=math.exp(observed['log_normalizer']);se=value*observed['relative_SE']
        assert abs(value-exact)<6*se, (label,value,exact,se)
        for population in result['populations']:
            assert population['physical_signature']['physical_fixed_neighbors']==cfg['fixed_poses']
            assert population['physical_signature']['metric']==cfg['metadata']
            assert population['independent_complete_cover_validated']
        controls[label]=dict(exact_hard_volume=exact,estimate=value,observed_SE=se,
            samples=1024,assessment_sha256=sha(campaign/'assessment-streaming.json'))
    assert sha(config)==config_sha
    report=dict(passed=True,total_fixture_draws=2048,controls=controls,binary_sha256=sha(binary),driver_sha256=sha(__file__),
        original_config_unchanged=True,default_prepare_command_unchanged=True,
        scope='Independent analytic difference of sphere translation-ball times SO(3) cap volumes; no production protein jobs.')
    save(root/'smoke-result.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
