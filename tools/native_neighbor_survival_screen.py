#!/usr/bin/env python3
"""Hard-only screen of existing native poses against the second/third fixed neighbor."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def rotation(pose):
    return Rotation.from_quat(np.asarray(pose["orientation"])[[1,2,3,0]]).as_matrix()


def validate_inputs(native_root, environment_path, shape_path):
    campaign=json.loads((native_root/'manifest.json').read_text())
    config=json.loads((native_root/'provenance/input-config.json').read_text())
    first=json.loads((Path(campaign['jobs'][0]['output'])/'manifest.json').read_text())
    env=json.loads(environment_path.read_text())
    assert hashlib.sha256(shape_path.read_bytes()).hexdigest()==first['shape_sha256'], 'Different atomic shape'
    assert len(config['fixed_poses'])==1 and env['fixed_poses'][0]==config['fixed_poses'][0], 'Different original M1 neighbor'
    metric=config['metadata']
    for field in ['native_poses','rigid_members']:
        assert env[field]==metric[field], f'Different {field}'
    for field in ['capture_center','capture_radius']:
        assert env[field]==config[field], f'Different {field}'
    assert env['native_position_tolerance_A']==metric['member_error_scale']
    assert env['native_angle_tolerance_degrees']==metric['angle_error_scale_deg']
    extracted=native_root/'valid-native-poses.jsonl'
    extracted_hash=hashlib.sha256(extracted.read_bytes()).hexdigest()
    concentration=json.loads((native_root/'concentration.json').read_text())
    assert extracted_hash==concentration['valid_pose_file_sha256'], 'Changed extracted native poses'
    return {'shape_matches':True,'original_M1_neighbor_matches':True,'native_metric_and_capture_match':True,
            'valid_pose_file_sha256':extracted_hash,'bath_fields':'Deliberately ignored: hard-only geometry screen'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root",type=Path,required=True)
    parser.add_argument("--environment",type=Path,required=True)
    parser.add_argument("--shape",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists()
    guards=validate_inputs(args.native_root,args.environment,args.shape)
    args.out.mkdir(parents=True)
    start=time.monotonic();env=json.loads(args.environment.read_text());shape=json.loads(args.shape.read_text())
    analysis=json.loads((args.native_root/'assessment-streaming.json').read_text())
    top={(r['replicate'],r['draw']):r for r in analysis['top_weights']}
    N=analysis['regions']['native']['nonzero'];sampling_seed=98670920;rng=np.random.default_rng(sampling_seed)
    selected=set(rng.choice(N,size=min(2048,N),replace=False).tolist())
    atoms=np.asarray([a['center'] for a in shape['atoms']]);radii=np.asarray([a['radius'] for a in shape['atoms']])
    trees=[]
    for fixed in env['fixed_poses'][1:]:
        centers=atoms@rotation(fixed).T+fixed['position']
        groups=[(float(r),cKDTree(centers[radii==r])) for r in np.unique(radii)]
        groups.sort(key=lambda x:x[1].n,reverse=True);trees.append(groups)
    assert len(trees)==2
    rows=[]
    for index,line in enumerate((args.native_root/'valid-native-poses.jsonl').open()):
        row=json.loads(line);key=(row['replicate'],row['draw'])
        if index not in selected and key not in top:continue
        mobile=atoms@rotation(row['pose']).T+row['pose']['position'];valid=[];gaps=[]
        for groups in trees:
            gap=np.inf
            for radius,tree in groups:
                d=tree.query(mobile,k=1,workers=1)[0];gap=min(gap,float(np.min(d-radii-radius)))
                if gap<0:break
            valid.append(gap>=0);gaps.append(gap)
        rows.append({'replicate':key[0],'draw':key[1],'uniform_subset':index in selected,'leading_weight':key in top,
                     'q':row['q'],'new_neighbor_B_valid':bool(valid[0]),'new_neighbor_C_valid':bool(valid[1]),
                     'M2_valid':bool(valid[0]),'M3_valid':bool(all(valid)),
                     'new_neighbor_gap_A':gaps,'original_M1_log_importance':row['log_importance_weight']})
    uniform=[r for r in rows if r['uniform_subset']];leading=[r for r in rows if r['leading_weight']]
    result={'scope':'Geometry-only screen of existing M1-hard-native draws; no new poses or depletion weights. Leading subset is post-selected and is not a probability estimate.',
            'uniform_subset_count':len(uniform),'all_native_valid_draws':N,
            'uniform_M2_valid':sum(r['M2_valid'] for r in uniform),'uniform_M3_valid':sum(r['M3_valid'] for r in uniform),
            'leading_count':len(leading),'leading_M2_valid':sum(r['M2_valid'] for r in leading),'leading_M3_valid':sum(r['M3_valid'] for r in leading),
            'rows':rows,'wall_seconds':time.monotonic()-start,
            'environment':str(args.environment),'environment_sha256':hashlib.sha256(args.environment.read_bytes()).hexdigest(),
            'shape_sha256':hashlib.sha256(args.shape.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_guards':guards,'uniform_subset_sampling_seed':sampling_seed,
            'valid_pose_file_sha256':guards['valid_pose_file_sha256']}
    (args.out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
