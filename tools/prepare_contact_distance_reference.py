#!/usr/bin/env python3
"""Freeze deterministic sphere-reference inputs; never launches a sampler."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import numpy as np
from contact_distance_reference import cartesian_from_radii,require,sha
from test_contact_distance_reference import fixture


def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Reference inputs must be new');out.mkdir(parents=True)
    region,guide,config,shape=fixture()
    shape.update(name='unit-sphere-distance-guide-reference',volume=4*math.pi/3)
    def write(name,data):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');return p
    shape_path=write('shape.json',shape);shape_hash=sha(shape_path)
    config.update(shape=str(shape_path),initial_pose=dict(position=[0.,3.,0.],orientation=[1.,0.,0.,0.]),
        depletant_radius=.1,reservoir_density=30.,poisson_lambda_ratio=64.,translation_steps=[.1],rotation_steps_deg=[1.],
        rotation_probability=.5,local_attempts_per_cycle=1,uniform_probability=.5,seed=1,
        metadata=dict(reference='two-distance-independent-correlated-unit-sphere-v1',no_physical_weights=True))
    config_path=write('config.json',config)
    region['fixed_neighbor']=copy.deepcopy(config['fixed_poses'][0]);region['gaussian_chart']['anchors'][0]['position']=[1.2,0.,0.]
    region['gaussian_chart'].update(schema='weighted-pose-mixture-v1',coordinate_convention='anchor-body-relative',shape_sha256=shape_hash,weights=[1.])
    region.update(shape_sha256=shape_hash,physical_fixed_neighbors=config['fixed_poses'],capture_center=config['capture_center'],capture_radius=config['capture_radius'],
        activity=config['reservoir_density'],depletant_radius=config['depletant_radius'],physical_metric=config['metadata'],
        minimum_original_q=0.,minimum_original_q_inclusive=True,maximum_original_q_inclusive=True,
        definition='Correlated raw chart around world origin; full R4 including hard-invalid zeros. Passive sphere reference; no Poisson clouds.')
    region_path=write('region.json',region);guide['region_sha256']=sha(region_path)
    arms=[]
    for label,b,seed in [('uniform_phi',0.,202610016501),('localized_phi',.9,202610016502)]:
        arm=copy.deepcopy(guide);arm['azimuth']['localized_probability']=b
        path=write('guides/'+label+'.json',arm)
        arms.append(dict(arm=label,guide=str(path),samples=128,seed=seed,physical_jobs=0))
    # Predetermined Cartesian contact/noncontact and angular/outside-domain probes.
    L=np.linalg.cholesky(np.asarray(region['gaussian_chart']['covariances'][0]));probes=[]
    raw=[('center_hard_invalid',[0.,0.,0.,0.,0.,0.]),
        ('axis_outside_contact',[3.,0.,0.,.3,-.2,.1]),('outside_R4',[9.,1.,2.,.1,.2,-.3]),
        ('outside_capture',[20.,0.,0.,.4,.1,-.2])]
    for index,phi in enumerate([.1,2.1,4.4]):
        p=cartesian_from_radii(np.array([-1.2,0,0]),np.array([1.2,0,0]),2.04,2.06,phi)
        raw.append(('two_contact_'+str(index),[*p,.2,-.1,.3]))
    for name,x in raw:probes.append(dict(id=name,latent=np.linalg.solve(L,np.asarray(x)).tolist()))
    (out/'probes.jsonl').write_text(''.join(json.dumps(p,allow_nan=False)+'\n' for p in probes))
    plan=dict(schema='contact-distance-passive-reference-allocation-v1',scope='Two fixed passive sphere-reference arms; no physical weights or depletants.',
        arms=arms,config=str(config_path),region=str(region_path),probes=str(out/'probes.jsonl'),total_fresh_draws=256,total_probe_queries=2*len(probes))
    write('allocation.json',plan)
    archive=out/'provenance';archive.mkdir()
    for name in ['prepare_contact_distance_reference.py','test_contact_distance_reference.py','contact_distance_reference.py']:
        p=Path(__file__).with_name(name);shutil.copy2(p,archive/name)
    fixture_path=Path(__file__).resolve().parents[1]/'tests/data/contact_distance_polygon_reference.json';shutil.copy2(fixture_path,archive/fixture_path.name)
    files=[p for p in out.rglob('*') if p.is_file()]
    write('manifest.json',dict(complete=True,files={str(p.relative_to(out)):sha(p) for p in files}))
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.out),indent=2))
