"""Select two completed-pilot contact cages and freeze local-only training.

Selection is pilot-informed, not a new validation sample. This tool only reads
saved evidence and writes metadata; it never launches a physical trajectory.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

PYTHON='/home/xvg/protein-nucleation/.venv/bin/python'


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def tokens(row):
    result={tuple(t) for t in row['patch_tokens']}
    require(len(result)==len(row['patch_tokens']),'Repeated contact token')
    return result


def jaccard(left,right):
    a,b=tokens(left),tokens(right)
    return 1-len(a&b)/len(a|b) if a|b else 0.


def choose_seeds(rows):
    """Fixed deterministic rule; no native registry field is inspected."""
    candidates=[r for r in rows if r['physical_valid'] and r['region'] not in ('A_patch_complete','unbound')]
    require(candidates and all(math.isfinite(r['log_rb_importance']) for r in candidates),'No finite competing contact candidates')
    ordered=sorted(candidates,key=lambda r:(-r['log_rb_importance'],r['stratum_id'],r['ordinal']))
    first=ordered[0]
    second=next((r for r in ordered if jaccard(first,r)>=.5),None)
    require(second is not None,'No second contact fingerprint meets predeclared Jaccard distance')
    return [first,second],ordered


def angle_degrees(left,right):
    for q in (left,right):
        require(len(q)==4 and all(math.isfinite(x) for x in q) and abs(sum(x*x for x in q)-1)<2e-10,'Invalid saved quaternion')
    return math.degrees(2*math.acos(min(1.,abs(math.fsum(x*y for x,y in zip(left,right))))))


def seed_number(allocation_digest,seed_index,stream):
    data=b'competing-cage-local-training-v1\0'+bytes.fromhex(allocation_digest)+seed_index.to_bytes(8,'little')+stream.to_bytes(8,'little')
    return int.from_bytes(hashlib.sha256(data).digest()[:8],'little')


def validate_limits(limits):
    expected=dict(cpu_seconds=600,wall_seconds=1200,memory_bytes=4*1024**3,
        raw_per_leg=20_000_000,raw_per_outer=40_000_000,raw_campaign=2_000_000_000,
        retained_per_leg=20_000_000,retained_per_outer=40_000_000,retained_campaign=2_000_000_000)
    require(limits==expected,'Changed reviewed per-chain limits')


def make_config(base,selected,seed_index,stream,allocation_digest,limits):
    validate_limits(limits)
    cfg=copy.deepcopy(base)
    require(cfg['translation_steps']==[.2] and cfg['rotation_steps_deg']==[1.]
            and cfg['rotation_probability']==.5 and cfg['local_attempts_per_cycle']==4
            and cfg['expected_fixed_body_count']==263 and cfg['depletant_radius']==1.5
            and cfg['reservoir_density']==.035 and cfg['poisson_lambda_ratio']==64.,'Changed physical/local baseline')
    cfg['initial_pose']=copy.deepcopy(selected['proposed_pose'])
    cfg['seed']=seed_number(allocation_digest,seed_index,stream)
    cfg['maximum_attempts']=11520;cfg['warmup_cycles']=256
    cfg['wall_seconds']=float(limits['wall_seconds'])
    cfg['limits']={k:limits[k] for k in ('raw_per_leg','raw_per_outer','raw_campaign',
        'retained_per_leg','retained_per_outer','retained_campaign')}
    cfg['limits']['cpu_seconds']=float(limits['cpu_seconds'])
    cfg['identity']=dict(kind='competing-cage-local-training',seed_index=seed_index,stream=stream,
        split='train' if stream<2 else 'heldout',moving_label=77,anchor_label=16,
        source_candidate=dict(stratum_id=selected['stratum_id'],ordinal=selected['ordinal']),
        allocation_sha256=allocation_digest,
        source_reference='Original source_state remains immutable; only initial_pose changes the physical start.',
        scope='Short conditional local trajectories for proposal fitting; no equilibration or assembly conclusion.')
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--expected-spec-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    require(sha(args.spec)==args.expected_spec_sha256,'Preparation specification changed')
    spec=read(args.spec);out=args.out.resolve();require(not out.exists(),'Fresh metadata root required')
    require(spec['schema']=='competing-cage-training-preparation-v1'
        and spec['seed_count']==2 and spec['streams_per_seed']==4
        and spec['train_streams']==[0,1] and spec['heldout_streams']==[2,3]
        and spec['warmup_cycles']==256 and spec['production_cycles']==2048
        and spec['cycles']==2304 and spec['attempts_per_cycle']==5
        and spec['sample_every']==1 and spec['secondary_contact_jaccard_min']==.5,
        'Changed frozen training allocation/selection')
    validate_limits(spec['limits'])
    files={}
    def bind(path,digest=None):
        path=Path(path).resolve();actual=sha(path);require(digest is None or actual==digest,'Changed input: '+str(path))
        require(str(path) not in files or files[str(path)]==actual,'Conflicting input binding')
        files[str(path)]=actual;return dict(path=str(path),sha256=actual)
    def asset(name):
        a=spec['assets'][name];bind(a['path'],a['sha256']);return read(a['path'])
    bind(args.spec,args.expected_spec_sha256);bind(__file__);bind(Path(PYTHON).resolve())
    report=asset('physical_report');noise=asset('noise_report');inventory=asset('physical_inventory')
    require(report['complete'] and report['passed'] and report['all_attempts']==32768
        and noise['complete'] and noise['passed'] and noise['all_attempts']==32768
        and noise['new_clouds']==noise['new_geometry_queries']==noise['new_poses']==0,
        'Incomplete saved physical/Rao-Blackwell evidence')
    require(noise['input_sha256'][spec['assets']['physical_report']['path']]==spec['assets']['physical_report']['sha256'],
        'Secondary analysis references a different primary report')
    contribution=spec['assets']['physical_contributions'];bind(contribution['path'],contribution['sha256'])
    require(contribution['sha256']==report['contributions_sha256'],'Wrong physical contributions')
    bank={}
    with Path(contribution['path']).open() as stream:
        for line in stream:
            row=json.loads(line);key=(row['stratum_id'],row['ordinal']);require(key not in bank,'Duplicate bank draw');bank[key]=row
    require(len(bank)==32768,'Incomplete all-attempt bank')
    candidates=[];seen=set()
    for job in inventory:
        p=(Path(job['result'])/'rows.jsonl').resolve();a=bind(p,report['input_sha256'][str(p)])
        with p.open() as stream:
            for line in stream:
                row=json.loads(line);key=(job['stratum_id'],row['saved_row_identity']['ordinal'])
                require(key in bank and key not in seen,'Repeated/unknown physical row');seen.add(key);saved=bank[key]
                require(saved['physical_valid'] and saved['proposed_pose']==row['pose'] and len(row['clouds'])==2
                    and all(c['progress']['complete'] for c in row['clouds']),'Incomplete saved positive-weight pair')
                lower=row['envelope']['lower_volume'];total=0
                for cloud in row['clouds']:
                    weight=cloud['weight'];require(weight['lower_volume']==lower,'Clouds have different lower volumes')
                    require(type(weight['overlap_points']) is int and weight['overlap_points']>=0,'Invalid saved count')
                    expected=.035*lower+weight['overlap_points']*math.log1p(.035/2.24)
                    require(abs(expected-weight['log_weight'])<2e-8,'Saved positive-weight algebra differs');total+=weight['overlap_points']
                if saved['region'] not in ('A_patch_complete','unbound'):
                    candidates.append(dict(saved,log_rb_importance=.035*lower+total*math.log1p(.035/(2*2.24))-saved['log_q_arm'],
                        physical_rows=a,physical_pose_index=row['pose_index'],source_positive_counts=[c['weight']['overlap_points'] for c in row['clouds']]))
    require(seen=={key for key,r in bank.items() if r['physical_valid']},'Not every valid draw has physical evidence')
    chosen,ordered=choose_seeds(candidates)
    base=asset('base_config');source=asset('source_state');guide=asset('reference_guide')
    region_config=asset('reference_regions_config');asset('patch_map')
    require(region_config['inputs']['patch_map']==spec['assets']['patch_map'],
        'Changed original contact-patch reference')
    require(base['source_state']==spec['assets']['source_state']['path']
        and base['expected_sha256']['source_state']==spec['assets']['source_state']['sha256']
        and source['moving_label']==77 and source['anchor_label']==16,'Changed source/anchor reference')
    for name in ('shape','fixed_context','source_state'):bind(base[name],base['expected_sha256'][name])
    model=spec['assets']['model'];bind(model['path'],model['sha256']);require(base['expected_sha256']['model']==model['sha256'],'Changed model')
    build=asset('build_manifest');receipt=asset('build_receipt');binary=spec['assets']['binary']
    bind(binary['path'],binary['sha256']);require(receipt['complete'] and receipt['passed']
        and receipt['binaries_sha256'][binary['path']]==binary['sha256'],'Unvalidated existing sampler')
    build_root=Path(spec['assets']['build_manifest']['path']).parent
    for name,digest in build['sources_sha256'].items():bind(build_root/'source'/name,digest)
    for name in ('controller','driver'):bind(spec['assets'][name]['path'],spec['assets'][name]['sha256'])
    out.mkdir();(out/'code').mkdir();(out/'configs').mkdir();(out/'centers').mkdir()
    selections=[]
    for index,row in enumerate(chosen):
        pose=row['proposed_pose'];angle=angle_degrees(pose['orientation'],source['pose']['orientation'])
        fingerprint=hashlib.sha256(json.dumps(sorted(list(t) for t in tokens(row)),separators=(',',':')).encode()).hexdigest()
        selections.append(dict(seed_index=index,candidate_identity=dict(stratum_id=row['stratum_id'],ordinal=row['ordinal']),
            pose=pose,region=row['region'],neighbor_fingerprint=row['neighbor_labels'],patch_tokens=row['patch_tokens'],
            contact_fingerprint_sha256=fingerprint,source_patch_fraction=row['source_fraction'],
            log_rb_importance=row['log_rb_importance'],rank=ordered.index(row),physical_rows=row['physical_rows'],
            physical_pose_index=row['physical_pose_index'],source_geometry_rows=row['source_rows'],
            source_positive_counts=row['source_positive_counts'],mahalanobis_squared=row['mahalanobis_squared'],
            original_source_rotation_angle_deg=angle,original_chart_distance_to_seam_deg=180-angle,
            new_seed_chart_rotation_angle_deg=0.,anchor_relative_rotation_angle_deg=angle_degrees(pose['orientation'],source['anchor_pose']['orientation']),
            patch_jaccard_from_first=jaccard(chosen[0],row),
            position_distance_from_first=math.sqrt(sum((a-b)**2 for a,b in zip(pose['position'],chosen[0]['proposed_pose']['position']))),
            rotation_angle_from_first_deg=angle_degrees(pose['orientation'],chosen[0]['proposed_pose']['orientation'])))
    selection=dict(schema='competing-cage-seed-selection-v1',complete=True,passed=True,
        rule='Largest RB importance contribution among all saved valid contacts outside A_T and unbound; second is first in the same decreasing-RB order with full-token Jaccard distance>=0.5. Ties: ascending(stratum_id,ordinal).',
        eligible_contacts=len(ordered),selected=selections,source_state=spec['assets']['source_state'],
        source_reference_unchanged=True,chart_centers='Each selected saved pose; independent auxiliary chart center, not a new physical source reference.',
        angular_length=guide['source_chart']['angular_length'],new_geometry_queries=0,new_clouds=0,new_poses=0,
        scope='Pilot-informed training-start selection only; no new validation population, native-label selection, basin equilibrium assertion or assembly conclusion.')
    save(out/'selection.json',selection);selection_asset=bind(out/'selection.json')
    limits=spec['limits']
    allocation=dict(schema='competing-cage-local-allocation-v1',selection=selection_asset,
        chains=8,seeds=2,streams_per_seed=4,train_streams=[0,1],heldout_streams=[2,3],
        cycles=2304,warmup_cycles=256,production_cycles=2048,local_attempts_per_cycle=4,
        actual_local_slots_per_cycle=5,attempts_per_chain=11520,warmup_attempts_per_chain=1280,
        production_attempts_per_chain=10240,total_attempts=92160,total_production_attempts=81920,
        sample_every=1,maximum_workers=4,threads_per_job=1,limits=limits,
        steps=dict(translation=.2,rotation_degrees=1.,rotation_probability=.5),
        seed_rule='SHA256(domain NUL,allocation SHA bytes,seed_index u64LE,stream u64LE); first8 digest bytes as u64LE.',
        retained_states='Every attempt_complete retained_pose including rejection repeats; never keep accepted states only. Warmup remains archived and excluded from fit.',
        split_scope='Streams0,1 fit; streams2,3 held out for descriptive validation. Short local chains need not have equilibrated; no target-width or physical-weight claim.',
        physical_scope='One mobile tetramer77 with263 fixed spectators at historical500uM geometry; rd1.5,z.035,lambda_ratio64. This is not finite-system106.8uM assembly.')
    save(out/'allocation.json',allocation);allocation_asset=bind(out/'allocation.json')
    for name in ('driver','controller'):
        target=out/'code'/('driver.py' if name=='driver' else 'run_lanes.py')
        shutil.copyfile(spec['assets'][name]['path'],target);bind(target,spec['assets'][name]['sha256'])
    shutil.copyfile(__file__,out/'code/prepare_competing_cage_training.py');bind(out/'code/prepare_competing_cage_training.py')
    jobs=[[] for _ in range(4)];training=[];seeds=set()
    for index,row in enumerate(chosen):
        center=dict(schema='source-chart-center-v1',frame='saved-spherical-center',pose=row['proposed_pose'],
            provenance=f"Frozen competing seed{index}; selection {selection_asset['sha256']}; original physical source/reference retained.")
        center_path=out/'centers'/f'seed{index}.json';save(center_path,center);center_asset=bind(center_path)
        for stream in range(4):
            cfg=make_config(base,row,index,stream,allocation_asset['sha256'],limits)
            require(cfg['seed'] not in seeds,'Random seed collision');seeds.add(cfg['seed'])
            identity=f'seed{index}-stream{stream}-local';path=out/'configs'/f'{identity}.json';save(path,cfg);cfg_asset=bind(path)
            lane_root=out/f'lane-{stream}';result=lane_root/'results'/identity
            jobs[stream].append(dict(id=identity,population=identity,phase='producer',
                argv=[binary['path'],'--config',str(path),'--model',model['path'],'--out',str(result),
                    '--cycles','2304','--sample-every','1','--method','local','--correlation','0.0'],
                cpu_limit_seconds=600,wall_limit_seconds=1200,address_space_limit_bytes=limits['memory_bytes'],
                terminal=dict(path=str(result/'summary.json'),success_contract='complete_and_passed')))
            training.append(dict(id=identity,seed_index=index,stream=stream,split='train' if stream<2 else 'heldout',
                config=cfg_asset,result=str(result),seed=cfg['seed'],initial_pose=row['proposed_pose'],chart_center=center_asset,
                execution_receipt=str(lane_root/'execution/jobs'/f'{index:03d}-{identity}'/'success.json'),
                source_candidate_identity=dict(stratum_id=row['stratum_id'],ordinal=row['ordinal']),
                source_state=spec['assets']['source_state'],reference_guide=spec['assets']['reference_guide'],
                attempts=11520,warmup_attempts=1280,production_attempts=10240))
    save(out/'inventory.json',training);inventory_asset=bind(out/'inventory.json')
    protocol=dict(schema='competing-cage-local-training-protocol-v1',allocation=allocation_asset,
        selection=selection_asset,inventory=inventory_asset,source_state=spec['assets']['source_state'],
        reference_regions_config=spec['assets']['reference_regions_config'],patch_map=spec['assets']['patch_map'],
        regions=region_config['inputs']['regions'],
        reference_guide=spec['assets']['reference_guide'],binary=binary,model=model,
        input_sha256=dict(files),launched=False)
    save(out/'protocol.json',protocol);bind(out/'protocol.json')
    module_spec=importlib.util.spec_from_file_location('held_driver',out/'code/driver.py');driver=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(driver)
    lanes=[]
    for lane,queue in enumerate(jobs):
        directory=out/f'lane-{lane}';directory.mkdir();(directory/'results').mkdir()
        plan=dict(schema='native-class-physical-execution-v1',root=str(directory),maximum_workers=1,threads=1,
            files=dict(files),executable_resolutions={binary['path']:str(Path(binary['path']).resolve())},jobs=queue)
        path=directory/'execution-plan.json';save(path,plan);driver.verify_plan(path,plan,sha(path),fresh=True)
        lanes.append(dict(path=str(path),sha256=sha(path)))
    manifest=dict(schema='fixed-context-correlated095-four-lanes-v1',root=str(out),maximum_workers=4,
        threads_per_job=1,python=PYTHON,python_resolution=str(Path(PYTHON).resolve()),driver=str(out/'code/driver.py'),
        files=dict(files)|{p['path']:p['sha256'] for p in lanes},lanes=lanes,
        ordered_lane_job_ids=[job['id'] for queue in jobs for job in queue],controller_wall_seconds=2490)
    save(out/'controller-manifest.json',manifest)
    module_spec=importlib.util.spec_from_file_location('held_controller',out/'code/run_lanes.py');controller=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(controller)
    controller.verify(out/'controller-manifest.json',sha(out/'controller-manifest.json'),True)
    save(out/'preparation.json',dict(complete=True,passed=True,launched=False,controller_manifest_sha256=sha(out/'controller-manifest.json'),
        chains=8,total_attempts=92160,selection=selection_asset,allocation=allocation_asset,new_geometry_queries=0,new_clouds=0,new_poses=0,
        source_reference_unchanged=True,scope='Prepared only; root reviews worker admission before any launch.'))
    print(json.dumps(read(out/'preparation.json')))


if __name__=='__main__':main()
