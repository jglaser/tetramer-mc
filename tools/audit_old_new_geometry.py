#!/usr/bin/env python3
"""Compile archived and current geometry together; compare identical endpoint poses."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
from audit_previous_smc_regions import recompute_q

ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def rotation(pose):return Rotation.from_quat(np.asarray(pose['orientation'])[[1,2,3,0]])
def aabb(shape,pose,rd):
    centers=rotation(pose).apply(np.array([a['center'] for a in shape['atoms']]))+pose['position']
    radii=np.array([a['radius']+rd for a in shape['atoms']])
    return np.min(centers-radii[:,None],axis=0),np.max(centers+radii[:,None],axis=0)

def verify_current_configs(out):
    """Supplement the pointwise audit with both actual docking configurations."""
    data=read(out/'input.json');records=[]
    for case in data['cases']:
        site=case['site']
        current_path=ROOT/('runs/involution-docking-conditional-5000/configs/site0-m1-native-r00-mixture.json'
            if site==0 else 'runs/involution-docking-smoke-200/configs/site1-m1-native-r00-mixture.json')
        current=read(current_path)
        source_summary=Path(case['source_summary'])
        old_path=OLD/'configs'/(source_summary.parent.name+'.json');old=read(old_path);oldenv=read(old['environment'])
        assert sha(old['shape'])==sha(current['shape'])
        assert old['depletant_radius']==current['depletant_radius']==data['rd']
        assert old['reservoir_density']==current['reservoir_density']==data['activity']
        assert oldenv['fixed_poses']==current['fixed_poses']
        assert oldenv['capture_center']==current['capture_center']
        assert oldenv['capture_radius']==current['capture_radius']
        old_q=recompute_q(case['pose'],oldenv,old)
        current_q=recompute_q(case['pose'],current['metadata'],current['metadata'])
        assert abs(old_q-current_q)<1e-12
        records.append(dict(id=case['id'],old_q=old_q,current_q=current_q,
            config_path=str(current_path),config_sha256=sha(current_path),
            old_config_path=str(old_path),old_config_sha256=sha(old_path),
            shape_sha256=sha(current['shape']),fixed_pose_match=True,capture_match=True,radius_activity_match=True))
    save(out/'current-config-comparison.json',dict(passed=True,cases=records,verifier_sha256=sha(__file__)))

def main(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Use a fresh output directory')
    archive=out/'provenance';archive.mkdir();crate=out/'probe';(crate/'src').mkdir(parents=True)
    source_geometry=OLD/'implementation/src/geometry.rs'
    shutil.copy2(source_geometry,crate/'src/geometry.rs')
    shutil.copy2(ROOT/'tools/old_new_geometry_probe.rs',crate/'src/main.rs')
    shutil.copy2(Path(__file__),archive/'audit_old_new_geometry.py')
    shutil.copy2(ROOT/'tools/audit_previous_smc_regions.py',archive/'audit_previous_smc_regions.py')
    cfg0=read(ROOT/'runs/involution-docking-conditional-5000/configs/site0-m1-native-r00-mixture.json')
    shape_path=Path(cfg0['shape']);shape=read(shape_path)
    shutil.copy2(shape_path,archive/'shape.json')
    files={shape_path,source_geometry,ROOT/'src/geometry.rs',ROOT/'src/math.rs',ROOT/'src/overlap_weight.rs',
           OLD/'implementation/src/single_body_depletion.rs',OLD/'implementation/src/bin/coordination_smc.rs'}
    cases=[]
    for site in [0,1]:
        for basin in ['native','other_adsorbed']:
            paths=sorted((OLD/'runs').glob(f'site{site}-m1-r1.5-z0.035-{basin}-*/summary.json'))
            paths=sorted(paths,key=lambda p:read(p)['logZ'],reverse=True)[:1 if basin=='native' else 2]
            for summary_path in paths:
                summary=read(summary_path);config_path=OLD/'configs'/(summary_path.parent.name+'.json');config=read(config_path)
                env_path=Path(config['environment']);env=read(env_path)
                files.update([summary_path,config_path,env_path,Path(config['shape'])])
                assert sha(config['shape'])==sha(shape_path)
                assert config['depletant_radius']==cfg0['depletant_radius']==1.5
                assert config['reservoir_density']==cfg0['reservoir_density']==.035
                assert env['capture_center']==cfg0['capture_center'] and env['capture_radius']==cfg0['capture_radius']
                if site==0:assert env['fixed_poses']==cfg0['fixed_poses']
                points=summary['final_particles'];q=np.array([p['q'] for p in points])
                # Deterministic diversity within the largest-normalizer populations.
                indices=[int(np.argmin(q)),int(np.argmax(q))]
                for index in indices:
                    p=points[index];q_new=recompute_q(p['pose'],env,config)
                    assert abs(q_new-p['q'])<1e-10
                    assert np.linalg.norm(np.array(p['pose']['position'])-env['capture_center'])<=env['capture_radius']
                    lower,upper=aabb(shape,p['pose'],1.5)
                    fixedbounds=[aabb(shape,fp,1.5) for fp in env['fixed_poses']]
                    lower=np.maximum(lower,np.min([b[0] for b in fixedbounds],axis=0))
                    upper=np.minimum(upper,np.max([b[1] for b in fixedbounds],axis=0))
                    # Remote periodic images cannot intersect the compact exclusion domains.
                    bound=max(np.linalg.norm(a['center'])+a['radius']+1.5 for a in shape['atoms'])
                    for fixed in env['fixed_poses']:
                        assert min(config['box_lengths'])-np.linalg.norm(np.array(fixed['position'])-p['pose']['position'])>2*bound
                    cases.append(dict(id=summary_path.parent.name+f'-p{index:03d}',pose=p['pose'],fixed=env['fixed_poses'],
                        box_lengths=config['box_lengths'],bounds_lo=lower.tolist(),bounds_hi=upper.tolist(),seed=3870811+len(cases)*101,
                        source_summary=str(summary_path),source_particle=index,population_logQ=summary['logZ'],old_q=p['q'],new_q=q_new,
                        radius_source='Actual run config; legacy environment bath metadata are ignored',site=site))
    save(out/'input.json',dict(shape=str(archive/'shape.json'),rd=1.5,activity=.035,common_points=args.points,
        cloud_repeats=16,lambda_ratio=64,gate_options=cfg0['endpoint_gate'],cases=cases))
    verify_current_configs(out)
    # Use one package identity for dependencies shared with tetramer-mc.
    # The archived implementation's vector/microstate/interaction sources are
    # byte-identical to this pinned snapshot; verify before compiling it.
    vendor=ROOT/'vendor/hoomd-rs'
    old_vendor=Path('/home/xvg/protein-nucleation/vendor/hoomd-rs')
    for dep in ['hoomd-vector','hoomd-interaction','hoomd-microstate']:
        for p in (vendor/dep).rglob('*'):
            if p.is_file():assert sha(p)==sha(old_vendor/dep/p.relative_to(vendor/dep))
    cargo='[package]\nname="old-new-geometry-probe"\nversion="0.1.0"\nedition="2024"\n[dependencies]\n'
    cargo+='anyhow="1"\nserde={version="1",features=["derive"]}\nserde_json="1"\nrand="0.10"\n'
    cargo+=f'tetramer-mc={{path="{ROOT}"}}\n'
    for dep in ['hoomd-vector','hoomd-interaction','hoomd-microstate']:
        cargo+=f'{dep}={{path="{vendor/dep}"}}\n'
    (crate/'Cargo.toml').write_text(cargo)
    save(out/'provenance.json',dict(input_sha256={str(p):sha(p) for p in sorted(files)},
        script_sha256=sha(__file__),rust_probe_sha256=sha(ROOT/'tools/old_new_geometry_probe.rs'),
        selection='For each site, largest-normalizer native population and two largest-normalizer other populations; minimum-q and maximum-q endpoints of each. Twelve deterministic cases.',
        precision='Common-point binomial C uncertainty plus independent current-envelope Poisson C uncertainty; full predicate agreement is a paired comparison, not limited by independent C sampling error.'))
    env=os.environ.copy();env['CARGO_TARGET_DIR']=str(ROOT/'target/geometry-audit')
    with (out/'build.log').open('w') as log:
        subprocess.run(['/home/xvg/.cargo/bin/cargo','build','--offline','--release','--manifest-path',str(crate/'Cargo.toml')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=Path(env['CARGO_TARGET_DIR'])/'release/old-new-geometry-probe'
    shutil.copy2(binary,archive/'old-new-geometry-probe')
    with (out/'probe.log').open('w') as log:
        subprocess.run([str(archive/'old-new-geometry-probe'),str(out/'input.json'),str(out/'results.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
    save(out/'artifacts.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifacts.json'})
    print(json.dumps(read(out/'results.json'),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs/old-new-geometry-audit')
    parser.add_argument('--points',type=int,default=1048576)
    parser.add_argument('--verify-existing',type=Path,help='Verify current-vs-old configs for an existing completed pointwise audit')
    args=parser.parse_args()
    if args.verify_existing:
        verify_current_configs(args.verify_existing.resolve())
    else:
        main(args)
