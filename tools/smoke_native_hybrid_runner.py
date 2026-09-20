#!/usr/bin/env python3
"""Exercise frozen-guide launcher and streaming audit on a tiny shifted sphere."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from analyze_native_region_reference import rotate


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pose(position,orientation=None):return {'position':position,'orientation':orientation or [1.,0.,0.,0.]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();root=args.out.resolve();binary=args.binary.resolve()
    assert not root.exists();root.mkdir(parents=True)
    tools=Path(__file__).resolve().parent
    shape=root/'sphere.json';save(shape,{'name':'hybrid smoke sphere','volume':4*math.pi/3,
        'atoms':[{'center':[0.,0.,0.],'radius':1.}]})
    center=[3.,-2.,1.];anchor=pose([50.,0.,0.],[math.cos(.2),0.,math.sin(.2),0.])
    cfg=dict(shape=str(shape),fixed_poses=[anchor,pose([0.,50.,0.])],capture_center=center,capture_radius=1.,
        depletant_radius=.5,reservoir_density=0.,poisson_lambda_ratio=64.,
        metadata={'native_poses':[pose(center)],'rigid_members':[pose([0.,0.,0.])],
            'member_error_scale':.5,'angle_error_scale_deg':45.})
    config=root/'config.json';save(config,cfg)
    inverse=[anchor['orientation'][0],0.,-anchor['orientation'][2],0.]
    relative=rotate(inverse,[center[i]-anchor['position'][i] for i in range(3)])
    columns=[rotate(inverse,[float(i==j) for i in range(3)]) for j in range(3)]
    rotation=[[columns[j][i] for j in range(3)] for i in range(3)]
    covariance=[[0.]*6 for _ in range(6)]
    for i in range(6):covariance[i][i]=.2**2 if i<3 else .15**2
    model=root/'model.json';save(model,dict(coordinate_convention='anchor-body-relative',angular_length=1.,
        shape_sha256=sha(shape),anchors=[dict(position=relative,rotation=rotation)],means=[[0.]*6],
        covariances=[covariance],weights=[1.]))
    base=[sys.executable,str(tools/'run_native_region_reference.py'),'--config',str(config),'--binary',str(binary),
        '--replicates','2','--samples','256','--workers','2','--seed-base','7341201','--expected-binary-sha256',sha(binary)]
    # Preparation-only checks also cover the default command path: no guide
    # flags or model validation are introduced into existing invocations.
    subprocess.run(base+['--root',str(root/'default-prepared'),'--prepare-only'],check=True,capture_output=True,text=True)
    default=json.loads((root/'default-prepared/manifest.json').read_text())
    assert 'guide_override' not in default
    assert all('--model' not in job['command'] for job in default['jobs'])
    assert not any((root/'default-prepared/runs').iterdir())
    campaign=root/'hybrid'
    command=base+['--root',str(campaign),'--model',str(model),'--model-weight','.75',
        '--model-uniform-probability','.05','--model-anchor-index','0']
    with (root/'runner.log').open('w') as output:subprocess.run(command,check=True,stdout=output,stderr=subprocess.STDOUT)
    manifest=json.loads((campaign/'manifest.json').read_text())
    frozen=campaign/'provenance/guide-model.json'
    assert sha(frozen)==sha(model)==manifest['guide_override']['model_sha256']
    for job in manifest['jobs']:
        assert job['command'][job['command'].index('--model')+1]==str(frozen)
    with (root/'analyzer.log').open('w') as output:
        subprocess.run([sys.executable,str(tools/'analyze_native_region_reference.py'),str(campaign),'--workers','2'],
            check=True,stdout=output,stderr=subprocess.STDOUT)
    result=json.loads((campaign/'assessment-streaming.json').read_text())
    expected=4*.5**3/3*(math.pi/4-math.sin(math.pi/4))
    hard=result['hard_regions']['native'];estimate=math.exp(hard['log_normalizer']);se=estimate*hard['relative_SE']
    assert result['all_rows_and_hashes_validated'] and result['samples']==512 and result['raw_cloud_points']==0
    assert result['regions']['native']==hard
    assert abs(estimate-expected)<6*se
    # In particular, guide plus a SINGLE cover must use weighted Q0. The old
    # volume*valid/N formula is wrong for this nonuniform hybrid proposal.
    assert abs(result['zero_activity_volume_estimate_A3']-estimate)<1e-14
    selected=sum(p['counts']['valid'] for p in result['populations'])
    legacy=result['populations'][0]['cover_volume']*selected/512
    assert abs(estimate-legacy)>1e-5
    report=dict(passed=True,samples=512,exact_hard_volume=expected,estimate=estimate,observed_SE=se,
        default_prepare_only_unchanged=True,guide_model_sha256=sha(frozen),binary_sha256=sha(binary),
        all_jobs_use_frozen_model=True,single_cover_hybrid_uses_importance_weights=True,
        invalid_legacy_volume_formula=legacy,driver_sha256=sha(__file__),assessment_sha256=sha(campaign/'assessment-streaming.json'),
        scope='Synthetic zero-activity sphere with shifted capture center and rotated explicit anchor; no protein campaign is launched.')
    save(root/'smoke-result.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
