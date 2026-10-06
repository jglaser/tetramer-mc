"""Prepare two clouds at every valid fresh draw; preserve all hard-zero attempts."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

from prepare_context_broadened_physical import panel_entries, caps, repository_root, executable_resolutions
from prepare_context_multicage_guides import ALLOCATIONS, CLOUD_RULE, PHYSICAL
from analyze_context_multicage_physical import validate_inventory

PYTHON='/home/xvg/protein-nucleation/.venv/bin/python'


ROOT=repository_root(__file__)
SCORER_BUILD=ROOT/'results/context-overlap-panel-build-20261005-v2'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def seed(digest,identity):
    return int.from_bytes(hashlib.sha256(b'context-multicage-all-valid-physical-v1\0'+
            digest.encode()+b'\0'+identity.encode()).digest()[:8],'little')


def validate_policy(policy):
    require(policy['schema']=='context-multicage-physical-policy-v1'
            and policy['all_attempts']==32768 and policy['strata']==32
            and policy['population_denominator']==4096 and policy['clouds_per_valid_pose']==2
            and policy['depletant_radius']==1.5 and policy['activity']==.035
            and policy['lambda_intensity']==2.24,'Changed fixed physical allocation')
    require(policy['estimator']==CLOUD_RULE and policy['cpu_seconds_per_job']==600
        and policy['wall_seconds_per_job']==1200 and policy['memory_bytes_per_job']==4*1024**3,
        'Changed physical runtime or estimator allocation')
    caps(policy,4096)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--expected-audit-sha256',required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--expected-policy-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();geometry=args.geometry.resolve();out=args.out.resolve();files={}
    require(not out.exists() and out.parent==ROOT/'results','Fresh repository result root required')
    def bind(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        require(expected is None or expected==digest,'Changed input: '+str(path))
        require(str(path) not in files or files[str(path)]==digest,'Conflicting input binding')
        files[str(path)]=digest;return dict(path=str(path),sha256=digest)
    audit_asset=bind(args.audit,args.expected_audit_sha256);audit=read(args.audit)
    require(audit['complete'] and audit['passed'] and audit['decoded_candidates']==32768
            and audit['independent_panel_size']==512
            and audit['schema']=='context-multicage-guide-independent-audit-v1','Incomplete independent geometry audit')
    for path,digest in audit['input_sha256'].items():bind(path,digest)
    contribution_path=args.audit.parent/'contributions.jsonl'
    contribution_asset=bind(contribution_path,audit['contributions_sha256'])
    contributions=[json.loads(l) for l in contribution_path.read_text().splitlines()]
    require(len(contributions)==32768,'Lost attempted geometry draws')
    policy_asset=bind(args.policy,args.expected_policy_sha256);policy=read(args.policy)
    validate_policy(policy)
    manifest_asset=bind(geometry/'controller-manifest.json');manifest=read(geometry/'controller-manifest.json')
    receipt_asset=bind(geometry/'controller/receipt.json');receipt=read(receipt_asset['path'])
    require(receipt['complete'] and receipt['passed'] and receipt['manifest_sha256']==manifest_asset['sha256'],
            'Geometry controller did not finish cleanly')
    inventory_asset=bind(geometry/'inventory.json');inventory=read(inventory_asset['path'])
    require(len(inventory)==len({e['id'] for e in inventory})==32
            and sum(e['draws'] for e in inventory)==32768,'Changed geometry inventory')
    mixture_asset=bind(audit['mixture_manifest']['path'],audit['mixture_manifest']['sha256'])
    mixture=read(mixture_asset['path'])
    require(mixture['schema']=='context-multicage-source-mixture-v1'
        and mixture['physical_cloud_allocation']==CLOUD_RULE and mixture['physical_conditions']==PHYSICAL,
        'Changed prospectively frozen two-cloud weight law')
    require(policy['estimator']==CLOUD_RULE,'Policy changes the predeclared weight rule')
    validate_inventory([dict(e, stratum_id=e['id'], attempts=e['draws']) for e in inventory])
    drain_asset=bind(geometry/'host-drained.json');drain=read(drain_asset['path'])
    require(drain['complete'] and drain['passed'] and drain['manifest_sha256']==manifest_asset['sha256']
        and drain['controller_receipt_sha256']==receipt_asset['sha256']
        and len(drain['owned_groups'])==37 and all(not g['same_process'] and not g['group_exists'] for g in drain['owned_groups']),
        'Geometry producers have not fully drained')
    require(all(audit['input_sha256'].get(str(geometry/p))==sha(geometry/p)
        for p in ('inventory.json','controller-manifest.json')), 'Geometry audit belongs to another campaign')
    build=read(SCORER_BUILD/'receipt.json');require(build['complete'] and build['passed'],'Scorer build failed')
    bind(SCORER_BUILD/'receipt.json');bind(SCORER_BUILD/'freeze.json')
    for name,digest in read(SCORER_BUILD/'freeze.json')['source_sha256'].items():bind(SCORER_BUILD/'source'/name,digest)
    binary=SCORER_BUILD/'target/release/examples/fixed_saved_pose_overlap'
    require(build['binaries_sha256'][str(binary)]=='10c1be09018226745506dfbdd9ef3894ebc83c4d11029a1d41da2801a8fcd8a1', 'Changed validated scorer')
    bind(binary,build['binaries_sha256'][str(binary)])
    bundle_paths=list((SCORER_BUILD/'target/release/build').glob('tetramer-mc-*/out/source-bundle.json'))
    require(len(bundle_paths)==1,'Ambiguous scorer source bundle')
    scorer_bundle=bind(bundle_paths[0])
    require(build['source_unchanged'] and build['protected_unchanged'],'Invalid build protection receipt')
    build_manifest=bind(SCORER_BUILD/'manifest.json',build['manifest_sha256'])
    bind(Path(PYTHON).resolve());bind(__file__)
    out.mkdir();(out/'code').mkdir();(out/'configs').mkdir();(out/'panels').mkdir()
    for name in ('driver.py','run_lanes.py'):
        path=out/'code'/name;shutil.copyfile(geometry/'code'/name,path);bind(path,sha(geometry/'code'/name))
    zero_worker=out/'code/empty_physical_stratum.py'
    shutil.copyfile(ROOT/'tools/empty_physical_stratum.py',zero_worker);bind(zero_worker)
    campaign=dict(schema='context-multicage-physical-campaign-v1',geometry_manifest=manifest_asset,
                  geometry_audit=audit_asset,geometry_contributions=contribution_asset,geometry_host_drained=drain_asset,policy=policy_asset,
                  scorer_source_bundle=scorer_bundle,scorer_build_manifest=build_manifest,scorer_build_receipt=bind(SCORER_BUILD/'receipt.json'),
                  scorer=dict(path=str(binary),sha256=sha(binary)),mixture_manifest=mixture_asset,estimator=CLOUD_RULE)
    save(out/'campaign.json',campaign);campaign_asset=bind(out/'campaign.json')
    jobs=[[] for _ in range(4)];physical_inventory=[];expected_raw=0.;valid_total=0
    by_stratum={entry['id']:[] for entry in inventory}
    for row in contributions:
        require(row['stratum_id'] in by_stratum,'Unknown contribution stratum')
        by_stratum[row['stratum_id']].append(row)
    for entry in inventory:
        config_asset=bind(entry['config'],entry['config_sha256']);source_config=read(config_asset['path'])
        source_rows=bind(Path(entry['result'])/'rows.jsonl')
        require(audit['input_sha256'].get(source_rows['path'])==source_rows['sha256'],'Rows lack audit binding')
        rows=[json.loads(l) for l in Path(source_rows['path']).read_text().splitlines()]
        selected=panel_entries(entry,rows,by_stratum[entry['id']],source_rows)
        panel=dict(schema='context-fixed-pose-overlap-panel-v1',entries=selected,source_rows=source_rows,
            stratum_id=entry['id'],attempts=entry['draws'],every_valid_pose_included=True,
            hard_invalid_count=entry['draws']-len(selected),geometry_audit=audit_asset,
            scope='All valid members of a complete independent importance stratum; omitted rows are authenticated hard zeros retained in the population denominator.')
        path=out/'panels'/(entry['id']+'.json');save(path,panel);panel_asset=bind(path)
        cfg=dict(schema='fixed-saved-pose-overlap-v1',inputs=source_config['inputs'],panel=panel_asset,
                 seed=seed(campaign_asset['sha256'],entry['id']),activity=policy['activity'],
                 envelope=policy['envelope'],envelope_tolerance=policy['envelope_tolerance'],
                 limits=dict(cpu_seconds=policy['cpu_seconds_per_job'],wall_seconds=policy['wall_seconds_per_job'],**policy['patch_limits']),
                 cloud_limits=caps(policy,entry['draws']))
        cfg['lambda']=policy['lambda_intensity']
        path=out/'configs'/(entry['id']+'.json');save(path,cfg);config_asset=bind(path)
        lane=entry['population_index'];result=out/f'lane-{lane}'/'results'/entry['id'];ordinal=len(jobs[lane])
        argv=([str(binary)] if selected else [PYTHON,'-B',str(zero_worker)])+['--config',str(path),'--out',str(result)]
        jobs[lane].append(dict(id=entry['id'],population=entry['id'],phase='producer',argv=argv,
            cpu_limit_seconds=policy['cpu_seconds_per_job'],wall_limit_seconds=policy['wall_seconds_per_job'],
            address_space_limit_bytes=policy['memory_bytes_per_job'],terminal=dict(path=str(result/'summary.json'),success_contract='complete_and_passed')))
        physical_inventory.append(dict(id=entry['id'],stratum_id=entry['id'],comparison_arm=entry['comparison_arm'],
            population_index=lane,component=entry['component'],attempts=entry['draws'],config=config_asset,panel=panel_asset,
            result=str(result),execution_receipt=str(out/f'lane-{lane}'/'execution/jobs'/f"{ordinal:03}-{entry['id']}"/'success.json'),
            empty=not selected,valid_poses=len(selected),source_rows=source_rows,geometry_config=entry['config']))
        valid_total+=len(selected)
        expected_raw+=sum(2*cfg['lambda']*e['metadata']['original_row']['envelope']['uncertain_volume'] for e in selected)
    require(all(len(lane)==8 for lane in jobs),'Physical lane inventory differs')
    validate_inventory(physical_inventory)
    save(out/'inventory.json',physical_inventory);bind(out/'inventory.json')
    protocol=dict(campaign,jobs=physical_inventory,
                  all_attempts=32768,population_denominator=4096,valid_poses=valid_total,clouds=2*valid_total,
                  expected_uncapped_raw_points=expected_raw,new_pose_draws=0,all_attempt_denominators=True)
    protocol['schema']='context-multicage-physical-protocol-v1'
    save(out/'protocol.json',protocol);bind(out/'protocol.json')
    spec=importlib.util.spec_from_file_location('physical_driver',out/'code/driver.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    lanes=[]
    for lane,queue in enumerate(jobs):
        directory=out/f'lane-{lane}';directory.mkdir();(directory/'results').mkdir()
        plan=dict(schema='native-class-physical-execution-v1',root=str(directory),maximum_workers=1,threads=1,files=dict(files),
                  executable_resolutions=executable_resolutions(queue),jobs=queue)
        path=directory/'execution-plan.json';save(path,plan);driver.verify_plan(path,plan,sha(path),fresh=True)
        lanes.append(dict(path=str(path),sha256=sha(path)))
    all_files=dict(files)|{r['path']:r['sha256'] for r in lanes}
    controller=dict(schema='fixed-context-correlated095-four-lanes-v1',root=str(out),maximum_workers=4,threads_per_job=1,
                    python=PYTHON,python_resolution=str(Path(PYTHON).resolve()),driver=str(out/'code/driver.py'),files=all_files,
                    lanes=lanes,ordered_lane_job_ids=[j['id'] for lane in jobs for j in lane],controller_wall_seconds=8*(policy['wall_seconds_per_job']+15)+60)
    save(out/'controller-manifest.json',controller)
    spec=importlib.util.spec_from_file_location('physical_controller',out/'code/run_lanes.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify(out/'controller-manifest.json',sha(out/'controller-manifest.json'),True)
    save(out/'preparation.json',dict(complete=True,passed=True,launched=False,controller_manifest_sha256=sha(out/'controller-manifest.json'),
        policy=policy_asset,valid_poses=valid_total,clouds=2*valid_total,expected_uncapped_raw_points=expected_raw,all_attempts=32768))
    print(json.dumps(read(out/'preparation.json')))


if __name__=='__main__':main()
