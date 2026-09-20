#!/usr/bin/env python3
"""Run the real mixture launcher/audit on a tiny independent sphere fixture.

The fixture has distant, noninteracting fixed spheres and uses test-only
128-draw, two-population constants. No protein inputs or production outputs
are modified. All emitted manifests state the actual test budget.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import run_native_cover_mixture_campaign as campaign


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def pose(position):
    return {'position':position,'orientation':[1.,0.,0.,0.]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=args.out.resolve();binary=args.binary.resolve()
    assert not root.exists()
    (root/'fixture/configs').mkdir(parents=True)
    shutil.copy2(__file__,root/'smoke-driver.py')
    shape=root/'fixture/sphere.json'
    save(shape,{'name':'smoke sphere','volume':4*math.pi/3,
                'atoms':[{'center':[0.,0.,0.],'radius':1.}]})
    a=pose([50.,0.,0.]);b=pose([0.,50.,0.])
    config={'shape':str(shape),'fixed_poses':[a],
        'capture_center':[0.,0.,0.],'capture_radius':18.,'depletant_radius':1.5,
        'reservoir_density':.035,'poisson_lambda_ratio':64.,
        'metadata':{'native_poses':[pose([0.,0.,0.])],'rigid_members':[pose([0.,0.,0.])],
            'member_error_scale':2.,'angle_error_scale_deg':35.,'source_periodic_box':[1000.]*3}}
    base=root/'fixture/base-config.json';save(base,config)
    baseline_population=root/'baseline-population'
    with (root/'baseline.log').open('x') as log:
        subprocess.run([str(binary),'--config',str(base),'--out',str(baseline_population),
            '--samples','4','--seed','19101','--lambda-ratio','64','--cloud-replicates','2'],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    baseline=root/'baseline-A';(baseline/'provenance').mkdir(parents=True)
    frozen=baseline/'provenance/native-region-normalizer';shutil.copy2(binary,frozen)
    save(baseline/'manifest.json',{'jobs':[{'output':str(baseline_population)}],
        'config_sha256':sha(base),'binary_sha256':sha(frozen)})
    variants={}
    for label,fixed in [('empty',[]),('A',[a]),('B',[b]),('AB',[a,b])]:
        item=copy.deepcopy(config);item['fixed_poses']=fixed
        item['metadata']['cooperative_plan']='synthetic launcher smoke only'
        path=root/'fixture/configs'/f'{label}.json';save(path,item)
        variants[label]={'path':str(path),'sha256':sha(path)}
    design=root/'fixture/manifest.json'
    save(design,{'source_config':str(base),'source_sha256':sha(base),'configs':variants})
    campaign.SAMPLES=128;campaign.REPLICATES=2
    sys.argv=[str(Path(campaign.__file__)),'--plan',str(design),'--A-root',str(baseline),
        '--factorial-root',str(root),'--out',str(root/'campaign'),'--binary',str(binary),'--execute']
    campaign.main()
    checks={}
    expected=json.loads((baseline_population/'cover.json').read_text())['volume']
    for label in variants:
        result=json.loads((root/'campaign'/label/'assessment-streaming.json').read_text())
        assert result['all_rows_and_hashes_validated'] and result['samples']==256
        assert result['raw_cloud_points']==0
        physical=result['regions']['native'];hard=result['hard_regions']['native']
        assert physical==hard
        estimate=math.exp(physical['log_normalizer']);se=estimate*physical['relative_SE']
        assert abs(estimate-expected)<6*se
        checks[label]={'samples':256,'estimate':estimate,'exact_volume':expected,
            'observed_SE':se,'raw_cloud_points':0,'assessment_sha256':sha(root/'campaign'/label/'assessment-streaming.json')}
    save(root/'smoke-result.json',{'passed':True,'test_only_samples_per_population':128,
        'test_only_populations_per_target':2,'total_fixture_draws':2048,
        'binary_sha256':sha(binary),'driver_sha256':sha(Path(__file__)),
        'checks':checks,'scope':'Synthetic sphere/far-neighbor fixture; production defaults remain four populations of 16384 draws.'})
    print(json.dumps({'smoke_passed':True,'out':str(root)},indent=2))


if __name__=='__main__':main()
