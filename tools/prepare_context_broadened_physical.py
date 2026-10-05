"""Prepare two clouds at every valid fresh draw; preserve all hard-zero attempts."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

PYTHON='/home/xvg/protein-nucleation/.venv/bin/python'


def repository_root(script_path):
    """Resolve both tools/ and frozen results/<campaign>/code/ locations.

    A snapshot's immediate parent is not the repository. Do not depend on cwd
    or silently redirect the scorer build and output into the snapshot tree.
    """
    for candidate in Path(script_path).resolve().parents:
        if all((candidate/name).is_file() for name in
               ('Cargo.toml','src/overlap_weight.rs','examples/fixed_saved_pose_overlap.rs')):
            return candidate
    raise ValueError('Cannot locate tetramer-mc repository from preparer snapshot')


ROOT=repository_root(__file__)
SCORER_BUILD=ROOT/'results/context-overlap-panel-build-20261005-v2'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def panel_entries(entry,rows,contributions,asset):
    require(len(rows)==len(contributions)==entry['draws'],'Incomplete stratum attempt inventory')
    entries=[]
    for ordinal,(row,contribution) in enumerate(zip(rows,contributions)):
        require(row['input']['ordinal']==contribution['ordinal']==ordinal and row['complete'],
                'Attempt order or completion changed')
        require(contribution['stratum_id']==entry['id'] and all(contribution[k]==entry[k]
                for k in ('comparison_arm','population_index','component')),'Changed stratum identity')
        valid=row['actual']['physical_valid']
        require(type(valid) is bool and contribution['physical_valid']==valid
                and row['physical_zero']==(not valid),'Hard-zero classification differs')
        require(contribution['region']==(row['region'] if valid else 'hard_invalid'),
                'Saved region differs from audited contribution')
        require(contribution['proposed_pose']==row['input']['proposed_pose'],
                'Saved pose differs from audited contribution')
        require(row['clouds']==[] and row['physical_weight_status']=='not_estimated'
                and row['log_physical_contribution'] is None,
                'Expected every attempted row to be unscored geometry')
        require(type(contribution['log_q_arm']) in (float,int)
                and math.isfinite(contribution['log_q_arm']),'Invalid full-arm density')
        if valid:
            require(row['actual']['wall_valid'] and row['actual']['core_valid']
                    and row['clouds']==[] and row['physical_weight_status']=='not_estimated',
                    'Expected unscored valid geometry pose')
            entries.append(dict(id=f"{entry['id']}-o{ordinal}",source_rows=asset,ordinal=ordinal,
                pose=row['input']['proposed_pose'],metadata=dict(original_row=row,
                geometry_contribution=contribution,log_q_balanced=contribution['log_q_arm'],
                density_scope='Alias for the complete comparison-arm mixture, not the generating stratum law.')))
    require(len(entries)==sum(r['actual']['physical_valid'] for r in rows),'Valid pose omitted')
    return entries


def caps(policy,attempts):
    require(type(attempts) is int and attempts>0,'Invalid original attempt count')
    c=policy['cloud_limits']
    require(c==dict(raw_per_cloud=1000000,processed_per_cloud=1000000,
        raw_per_pose=2000000,processed_per_pose=2000000,total_per_attempt=2000000,callback_interval=8192),
        'Changed predeclared full-envelope cloud budget')
    return {k:c[k] for k in ('raw_per_cloud','processed_per_cloud','raw_per_pose','processed_per_pose','callback_interval')}|{
        'raw_total':c['total_per_attempt']*attempts,'processed_total':c['total_per_attempt']*attempts}


def seed(digest,identity):
    return int.from_bytes(hashlib.sha256(b'context-broadened-all-valid-physical-v1\0'+
            digest.encode()+b'\0'+identity.encode()).digest()[:8],'little')


def executable_resolutions(jobs):
    """A lane need not contain both scoring and explicitly empty jobs."""
    return {job['argv'][0]:str(Path(job['argv'][0]).resolve()) for job in jobs}


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
            and audit['independent_panel_size']==512,'Incomplete independent geometry audit')
    for path,digest in audit['input_sha256'].items():bind(path,digest)
    contribution_path=args.audit.parent/'contributions.jsonl'
    contribution_asset=bind(contribution_path,audit['contributions_sha256'])
    contributions=[json.loads(l) for l in contribution_path.read_text().splitlines()]
    require(len(contributions)==32768,'Lost attempted geometry draws')
    policy_asset=bind(args.policy,args.expected_policy_sha256);policy=read(args.policy)
    require(policy['schema']=='context-broadened-physical-policy-v1'
            and policy['all_attempts']==32768 and policy['strata']==20
            and policy['population_denominator']==4096 and policy['clouds_per_valid_pose']==2
            and policy['depletant_radius']==1.5 and policy['activity']==.035
            and policy['lambda_intensity']==2.24,'Changed fixed physical allocation')
    manifest_asset=bind(geometry/'controller-manifest.json');manifest=read(geometry/'controller-manifest.json')
    receipt_asset=bind(geometry/'controller/receipt.json');receipt=read(receipt_asset['path'])
    require(receipt['complete'] and receipt['passed'] and receipt['manifest_sha256']==manifest_asset['sha256'],
            'Geometry controller did not finish cleanly')
    inventory_asset=bind(geometry/'inventory.json');inventory=read(inventory_asset['path'])
    require(len(inventory)==len({e['id'] for e in inventory})==20
            and sum(e['draws'] for e in inventory)==32768,'Changed geometry inventory')
    mixture_asset=bind(ROOT/'results/context-broadened-frozen-guides-20261005/result/mixture-manifest.json',
                       'd74a4d564fd71547a52292c5d7a9a84ab14cf967e8ce0072389f3958b7bee827')
    mixture=read(mixture_asset['path'])
    build=read(SCORER_BUILD/'receipt.json');require(build['complete'] and build['passed'],'Scorer build failed')
    bind(SCORER_BUILD/'receipt.json');bind(SCORER_BUILD/'freeze.json')
    for name,digest in read(SCORER_BUILD/'freeze.json')['source_sha256'].items():bind(SCORER_BUILD/'source'/name,digest)
    binary=SCORER_BUILD/'target/release/examples/fixed_saved_pose_overlap'
    bind(binary,build['binaries_sha256'][str(binary)])
    bind(Path(PYTHON).resolve());bind(__file__)
    out.mkdir();(out/'code').mkdir();(out/'configs').mkdir();(out/'panels').mkdir()
    for name in ('driver.py','run_lanes.py'):
        path=out/'code'/name;shutil.copyfile(geometry/'code'/name,path);bind(path,sha(geometry/'code'/name))
    zero_worker=out/'code/empty_physical_stratum.py'
    shutil.copyfile(ROOT/'tools/empty_physical_stratum.py',zero_worker);bind(zero_worker)
    campaign=dict(schema='context-broadened-physical-campaign-v1',geometry_manifest=manifest_asset,
                  geometry_audit=audit_asset,geometry_contributions=contribution_asset,policy=policy_asset,
                  scorer=dict(path=str(binary),sha256=sha(binary)),mixture_manifest=mixture_asset)
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
    require(all(len(lane)==5 for lane in jobs),'Physical lane inventory differs')
    save(out/'inventory.json',physical_inventory);bind(out/'inventory.json')
    protocol=dict(campaign,jobs=physical_inventory,
                  all_attempts=32768,population_denominator=4096,valid_poses=valid_total,clouds=2*valid_total,
                  expected_uncapped_raw_points=expected_raw,new_pose_draws=0,all_attempt_denominators=True)
    protocol['schema']='context-broadened-physical-protocol-v1'
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
                    lanes=lanes,ordered_lane_job_ids=[j['id'] for lane in jobs for j in lane],controller_wall_seconds=5*1215+60)
    save(out/'controller-manifest.json',controller)
    spec=importlib.util.spec_from_file_location('physical_controller',out/'code/run_lanes.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify(out/'controller-manifest.json',sha(out/'controller-manifest.json'),True)
    save(out/'preparation.json',dict(complete=True,passed=True,launched=False,controller_manifest_sha256=sha(out/'controller-manifest.json'),
        policy=policy_asset,valid_poses=valid_total,clouds=2*valid_total,expected_uncapped_raw_points=expected_raw,all_attempts=32768))
    print(json.dumps(read(out/'preparation.json')))


if __name__=='__main__':main()
