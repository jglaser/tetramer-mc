"""Freeze two completed competing-cage residence fits; metadata only.

This preparer never loads pose histories, evaluates coordinates, or launches a
job. The frozen plan first runs synthetic controls, then reduces all eight
completed local histories with the predeclared train/heldout split.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path('/home/xvg/tetramer-mc')
PYTHON='/home/xvg/protein-nucleation/.venv/bin/python'
REDUCER='analyze_competing_cage_covariance.py'
TEST='test_competing_cage_covariance.py'
FIT_RULE=dict(training_streams=[0,1],heldout_streams=[2,3],phase='production',covariance_denominator='N',
    bandwidth_multiplier=1.,ridge_absolute_scaled=1e-10,ridge_trace_factor=1e-6,
    translation_scale_angstrom=.1,rotation_scale_degrees=.5,
    gaussian_components_per_declared_cage=1,heldout_parameter_updates=0)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--expected-manifest-sha256',required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--expected-spec-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();training=args.training.resolve();out=args.out.resolve()
    require(training.parent==out.parent==ROOT/'results' and not out.exists(),'Expected results roots and fresh output')
    require(sha(args.spec)==args.expected_spec_sha256,'Changed frozen fit specification')
    spec=read(args.spec)
    require(spec['schema']=='competing-cage-fit-specification-v1' and spec['fit_rule']==FIT_RULE
        and spec['synthetic_tests']==10 and spec['training_chains']==8 and spec['total_retained_states']==92160
        and spec['limits']==dict(tests_cpu_seconds=10,tests_wall_seconds=30,reduction_cpu_seconds=60,
            reduction_wall_seconds=120,memory_bytes=2*1024**3,workers=1,threads=1),'Changed predeclared fit specification')
    files={}
    def bind(path,digest=None):
        path=Path(path).resolve();actual=sha(path)
        require(digest is None or digest==actual,'Changed input: '+str(path))
        require(str(path) not in files or files[str(path)]==actual,'Conflicting binding')
        files[str(path)]=actual;return dict(path=str(path),sha256=actual)
    def asset(a):bind(a['path'],a['sha256']);return read(a['path'])
    spec_asset=bind(args.spec,args.expected_spec_sha256)
    manifest_asset=bind(training/'controller-manifest.json',args.expected_manifest_sha256)
    manifest=read(manifest_asset['path'])
    require(manifest['maximum_workers']==4 and manifest['threads_per_job']==1 and len(manifest['lanes'])==4,'Changed training lanes')
    receipt_asset=bind(training/'controller/receipt.json');receipt=read(receipt_asset['path'])
    require(receipt['complete'] and receipt['passed'] and receipt['source_unchanged']
        and receipt['manifest_sha256']==manifest_asset['sha256'] and receipt['error'] is None
        and receipt['drain_error'] is None and len(receipt['completed_lanes'])==4
        and all(r['complete'] and r['returncode']==0 for r in receipt['completed_lanes']),'Training controller incomplete')
    host_asset=bind(training/'host-drained.json');host=read(host_asset['path'])
    require(host['complete'] and host['passed'] and host['manifest_sha256']==manifest_asset['sha256']
        and host['controller_receipt_sha256']==receipt_asset['sha256']
        and len(host['owned_groups'])==len({g['pid'] for g in host['owned_groups']})==13
        and all(not g['same_process'] and not g['group_exists'] for g in host['owned_groups']),'Training groups not drained')
    inv_asset=bind(training/'inventory.json',manifest['files'][str(training/'inventory.json')]);inventory=read(inv_asset['path'])
    protocol_asset=bind(training/'protocol.json',manifest['files'][str(training/'protocol.json')]);training_protocol=read(protocol_asset['path'])
    allocation=asset(training_protocol['allocation']);selection=asset(training_protocol['selection'])
    require(allocation['chains']==8 and allocation['total_attempts']==92160
        and allocation['train_streams']==[0,1] and allocation['heldout_streams']==[2,3]
        and allocation['production_attempts_per_chain']==10240 and selection['complete'] and selection['passed']
        and [s['seed_index'] for s in selection['selected']]==[0,1],'Changed allocation/selection')
    require(len(inventory)==8 and {(j['seed_index'],j['stream']) for j in inventory}=={(i,s) for i in range(2) for s in range(4)}
        and len({j['seed'] for j in inventory})==8,'Incomplete or duplicate training inventory')
    by_id={j['id']:j for j in inventory};require(len(by_id)==8,'Duplicate chain ID')
    completed={}
    for lane in manifest['lanes']:
        plan=asset(lane);lane_root=Path(lane['path']).parent
        bind(lane_root/'execution/summary.json');status=read(lane_root/'execution/summary.json')
        require(status['complete'] and status['passed'] and status['plan_sha256']==lane['sha256']
            and len(status['completed'])==2 and status['failure'] is None and not status['active'] and not status['unstarted'],'Incomplete training lane')
        for done in status['completed']:
            require(done['id'] in by_id and done['id'] not in completed,'Duplicate/unexpected completed chain')
            item=by_id[done['id']];result=Path(item['result']).resolve()
            actual_receipt=bind(item['execution_receipt'])
            require(read(actual_receipt['path'])==done and done['success'] and done['child_drained']
                and done['returncode']==0 and not done['timeout'] and done['retries']==done['replacements']==0
                and Path(done['terminal']['path'])==result/'summary.json','Failed or mismatched training chain')
            summary=asset(done['terminal']);cfg=asset(item['config'])
            require(summary['complete'] and summary['passed'] and summary['method']=='local'
                and summary['completed_cycles']==2304 and summary['completed_attempts']==11520
                and summary['identity']==cfg['identity'] and summary['bindings']['config']==item['config']['sha256']
                and cfg['initial_pose']==item['initial_pose'] and cfg['seed']==item['seed']
                and cfg['identity']['seed_index']==item['seed_index'] and cfg['identity']['stream']==item['stream']
                and cfg['identity']['source_candidate']==item['source_candidate_identity']
                and item['split']==cfg['identity']['split']==('train' if item['stream']<2 else 'heldout'),'Training identities differ')
            require(summary['bindings']['source_state']==training_protocol['source_state']['sha256']
                and summary['bindings']['model']==training_protocol['model']['sha256'],'Changed immutable physical reference')
            for name in ('shape','fixed_context','source_state'):
                p=Path(cfg[name]);p=p if p.is_absolute() else Path(item['config']['path']).parent/p
                bind(p,cfg['expected_sha256'][name]);require(summary['bindings'][name]==cfg['expected_sha256'][name],'Executed physical asset differs')
            center=asset(item['chart_center'])
            require(center['pose']==item['initial_pose']==selection['selected'][item['seed_index']]['pose'],'Chart center/start selection differs')
            completed[item['id']]=dict(stream=item['stream'],seed=item['seed'],config=item['config'],
                summary=done['terminal'],receipt=actual_receipt,journal=bind(result/'events.jsonl'),
                fixed_context=dict(path=str(Path(cfg['fixed_context']).resolve()),sha256=summary['bindings']['fixed_context']))
    require(set(completed)==set(by_id),'Missing retained training history')
    source=training_protocol['source_state'];model=training_protocol['model'];asset(source);m=asset(model)
    contexts=[v['fixed_context'] for v in completed.values()];require(all(c==contexts[0] for c in contexts),'Changed context between chains')
    region_reference=spec['patch_reference']
    for a in region_reference.values():asset(a)
    require(region_reference['patch_map']==training_protocol['patch_map']
        and region_reference['regions']==training_protocol['reference_regions_config']
        and read(region_reference['regions']['path'])['inputs']['regions']==training_protocol['regions']
        and read(region_reference['regions']['path'])['inputs']['patch_map']==training_protocol['patch_map'],'Original patch reference differs')
    runtime=asset(spec['runtime_plan']);driver_asset=spec['driver'];bind(driver_asset['path'],driver_asset['sha256'])
    require(runtime['executable_resolutions'].get(PYTHON)==str(Path(PYTHON).resolve()),'Changed Python runtime')
    numerical=0
    for path,digest in runtime['files'].items():
        if '/site-packages/' in path or Path(path).resolve()==Path(PYTHON).resolve():bind(path,digest);numerical+=1
    require(numerical>=300,'Incomplete inherited numerical runtime')
    # Snapshot only this reducer's static local Python import closure, never import it here.
    names={};pending=[REDUCER,TEST]
    while pending:
        name=pending.pop()
        if name in names:continue
        src=ROOT/'tools'/name;names[name]=sha(src)
        for node in ast.walk(ast.parse(src.read_text())):
            modules=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                target=module.split('.')[0]+'.py'
                if (ROOT/'tools'/target).exists():pending.append(target)
    require(names[REDUCER]==spec['reducer_sha256'] and names[TEST]==spec['tests_sha256'],'Unreviewed reducer/tests')
    out.mkdir();code=out/'code';code.mkdir()
    for name,digest in names.items():shutil.copyfile(ROOT/'tools'/name,code/name);bind(code/name,digest)
    shutil.copyfile(driver_asset['path'],code/'driver.py');bind(code/'driver.py',driver_asset['sha256'])
    shutil.copyfile(__file__,code/'prepare_competing_cage_covariance.py');bind(code/'prepare_competing_cage_covariance.py')
    cages=[]
    for cage_id in range(2):
        jobs=sorted([j for j in inventory if j['seed_index']==cage_id],key=lambda j:j['stream'])
        require(all(j['chart_center']==jobs[0]['chart_center'] for j in jobs),'Chart center differs between streams')
        cages.append(dict(cage_id=cage_id,chart_center=jobs[0]['chart_center'],initial_pose=jobs[0]['initial_pose'],
            source_candidate_identity=jobs[0]['source_candidate_identity'],
            streams=[{k:v for k,v in completed[j['id']].items() if k!='fixed_context'} for j in jobs]))
    protocol=dict(schema='competing-cage-covariance-v1',specification=spec_asset,fit_rule=FIT_RULE,
        quantiles=[0.,.01,.05,.25,.5,.75,.95,.99,1.],nominal_coverage=[.5,.9,.95,.99],
        cycles=2304,warmup_cycles=256,slots_per_cycle=5,
        physical_conditions=dict(depletant_radius=1.5,activity=.035,lambda_ratio=64.),
        angular_length=m.get('base_model',m)['angular_length'],original_source_state=source,context=contexts[0],model=model,
        patch_reference=dict(region_reference,definitions=training_protocol['regions'],unchanged_during_training=True),cages=cages,
        training_manifest=manifest_asset,training_inventory=inv_asset,training_controller=receipt_asset,training_host_drain=host_asset,
        input_sha256=dict(files),scope='Two pilot-selected conditional cages, one frozen Gaussian each. No equilibrium covariance or assembly claim.')
    save(out/'protocol.json',protocol);bind(out/'protocol.json')
    wrapper=code/'run_tests.py';wrapper.write_text('''import json,sys,time,unittest
from pathlib import Path
root=Path(__file__).resolve().parent.parent
suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern="test_competing_cage_covariance.py")
started=time.process_time()
with (root/"tests.log").open("x") as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
passed=result.wasSuccessful() and result.testsRun==10
with (root/"tests.json").open("x") as stream:
 json.dump(dict(complete=True,passed=passed,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),cpu_seconds=time.process_time()-started,new_poses=0,new_geometry_queries=0,new_clouds=0),stream,indent=2)
sys.exit(0 if passed else 1)
''');bind(wrapper)
    limits=spec['limits']
    plan=dict(schema='native-class-physical-execution-v1',root=str(out),maximum_workers=1,threads=1,files=files,
        executable_resolutions={PYTHON:str(Path(PYTHON).resolve())},jobs=[
            dict(id='ten-synthetic-controls',population='synthetic-only',phase='algebra',argv=[PYTHON,'-B',str(wrapper)],
                cpu_limit_seconds=limits['tests_cpu_seconds'],wall_limit_seconds=limits['tests_wall_seconds'],
                address_space_limit_bytes=limits['memory_bytes'],terminal=dict(path=str(out/'tests.json'),success_contract='complete_and_passed')),
            dict(id='two-cages-eight-streams',population='all-predeclared-retained-states',phase='statistics',
                argv=[PYTHON,'-B',str(code/REDUCER),'--protocol',str(out/'protocol.json'),'--out',str(out/'result')],
                cpu_limit_seconds=limits['reduction_cpu_seconds'],wall_limit_seconds=limits['reduction_wall_seconds'],
                address_space_limit_bytes=limits['memory_bytes'],terminal=dict(path=str(out/'result/report.json'),success_contract='complete_and_passed'))])
    save(out/'execution-plan.json',plan)
    module_spec=importlib.util.spec_from_file_location('held_fit_driver',code/'driver.py');driver=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(driver)
    driver.verify_plan(out/'execution-plan.json',plan,sha(out/'execution-plan.json'),fresh=True)
    save(out/'preparation.json',dict(complete=True,passed=True,launched=False,plan_sha256=sha(out/'execution-plan.json'),
        protocol_sha256=sha(out/'protocol.json'),source_snapshots=names,retained_states=92160,production_states=81920,
        declared_training_states=40960,heldout_fit_samples=0,new_poses=0,new_geometry_queries=0,new_clouds=0))
    print(json.dumps(read(out/'preparation.json')))


if __name__=='__main__':main()
