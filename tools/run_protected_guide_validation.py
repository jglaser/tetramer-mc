#!/usr/bin/env python3
"""Freeze and execute one independent, fixed protected-guide R4 validation.

Historical campaigns and their failed gates are immutable. This controller does
not fit, refill, retry, pool populations, change the physical target, or launch
assembly. The shared executor drains failures before admitting another job.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from analyze_latent_region import LatentImportanceGuide
from prepare_protected_guide_validation import validate_preparation
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_confirmation import (runtime, worker_environment, THREAD_ENV,
    CAMPAIGN_SCHEMA, POPULATION_SCHEMA, DENSITY_MEASURE)
from run_entry_shell_reference_campaign import read, write, sha, require, inside, file_hashes
from run_full_vessel_comparison import execute_group
from run_mobile_posterior_pilot import verify_bundle
from run_smc_guide_pilot import (validate_base_package, verify_output, verify_assessment,
    audit_step, OLD_GUIDE_SHA256, SPHERE_REFERENCE_SHA256, STRATA, CONVERGENCE)
from run_smc_importance_bridge import process_token

ROOT = Path(__file__).resolve().parents[1]
RUNNER_NAME = 'run_protected_guide_validation.py'
ANALYZER_NAME = 'analyze_protected_guide_validation.py'
WORKFLOW_NAME = 'run_protected_guide_workflow.py'
SCHEMA = 'protected-guide-validation-controller-v1'
STATUS_SCHEMA = 'protected-guide-validation-status-v1'
BINARY_SHA256 = 'b0e051638276de13336090c21b4f2c8def693d5b6037eac1e2591dba29225094'
BUNDLE_SHA256 = '3de89630a7a797a8e79f13b5c094390dcb6c6a5a1e1368e0f8788b6c83d68e4e'
ARMS = [dict(id=name, samples=draws, alpha=.5, lambda_ratio=intensity, component_count=count)
    for name,draws,intensity,count in [('bank',1048576,128.,80),
        ('protected',1048576,128.,84),('small',262144,128.,84),('intensity256',262144,256.,84)]]
SEEDS = [148101010+1009*i for i in range(16)]
TOTAL = 10485760
DECISION_ARMS = ['bank','protected','intensity256']
COMPARISONS = [['bank','protected'],['protected','small'],
               ['protected','intensity256'],['small','intensity256']]


def source_entries(root):
    return [root/RUNNER_NAME, root/ANALYZER_NAME, root/WORKFLOW_NAME, root/'analyze_latent_region.py']


def declared_seeds(value):
    """Conservatively inventory every explicitly named seed in a declaration."""
    result = set()
    if isinstance(value, dict):
        for key,item in value.items():
            if key == 'seed' and type(item) is int:
                result.add(item)
            elif key.endswith('seeds') and isinstance(item, list):
                result.update(s for s in item if type(s) is int)
            result.update(declared_seeds(item))
    elif isinstance(value, list):
        for item in value: result.update(declared_seeds(item))
    return result


def seed_inventory(repository, package, preparation):
    sources, previous = {}, set()
    for path in sorted((Path(repository)/'runs').rglob('protocol.json')):
        previous.update(declared_seeds(read(path)))
        sources[str(path.resolve())] = sha(path)
    training = set(read(Path(preparation)/'plan.json')['training_seeds'])
    training.update(a['seed'] for a in read(Path(package)/'plan.json')['anchor_metadata'])
    reference = set(read(Path(preparation)/'proposal-reference/declaration.json')['seeds'])
    all_seeds = previous | training | reference
    require(not all_seeds & set(SEEDS), 'Fresh validation seed collides with a prior declaration or training source')
    return dict(sources=sources, protocol_seeds=sorted(previous), training_seeds=sorted(training),
        proposal_reference_seeds=sorted(reference), seeds=sorted(all_seeds), fresh_seeds=SEEDS,
        scope='All recursive runs/**/protocol.json declarations, source training seeds, and proposal reference seeds; no raw replay')


def validate_inventory(common, package, preparation):
    inventory = read(common/'seed-inventory.json'); previous = set()
    for digest in inventory['sources'].values():
        path = common/'seed-protocols'/(digest+'.json')
        require(sha(path) == digest, 'Archived seed protocol changed')
        previous.update(declared_seeds(read(path)))
    training = set(read(preparation/'plan.json')['training_seeds'])
    training.update(a['seed'] for a in read(package/'plan.json')['anchor_metadata'])
    reference = set(read(preparation/'proposal-reference/declaration.json')['seeds'])
    require(inventory['protocol_seeds'] == sorted(previous)
        and inventory['training_seeds'] == sorted(training)
        and inventory['proposal_reference_seeds'] == sorted(reference)
        and inventory['seeds'] == sorted(previous | training | reference)
        and inventory['fresh_seeds'] == SEEDS and not set(inventory['seeds']) & set(SEEDS),
        'Seed inventory or independence changed')


def validate_protected(preparation, package, binary, bundle_path):
    preparation, package = Path(preparation), Path(package)
    plan = validate_preparation(preparation)
    for key,name in [('region_sha256','region.json'),('shape_sha256','shape.json'),
                     ('config_sha256','config.json'),('reference_region_sha256','old-r5-region.json')]:
        require(plan[key] == sha(package/name), 'Protected preparation target changed: '+key)
    require(plan['guide_sha256']['bank'] == sha(package/'guide-bank.json') == OLD_GUIDE_SHA256,
            'Retained bank changed')
    guide = LatentImportanceGuide(read(preparation/'guide-protected.json'),sha(package/'region.json'))
    require(guide.count == 84 and guide.alpha == .5, 'Protected guide must retain 84 components and uniform probability one half')
    require(plan['proposal_reference'] == 'proposal-reference' and plan['sphere_reference'] == 'sphere-reference.json',
            'Reference layout changed')
    sphere_path = preparation/'sphere-reference.json'
    require(sha(binary) == BINARY_SHA256 == plan['binary_sha256']
        and sha(bundle_path) == BUNDLE_SHA256 == plan['source_bundle_sha256']
        and sha(sphere_path) == SPHERE_REFERENCE_SHA256 == plan['sphere_reference_sha256'],
        'Executable or reused sphere reference changed')
    sphere = read(sphere_path)
    require(sphere['schema'] == 'conditional-ray-sphere-crosslanguage-v1' and sphere['complete'] is True
        and sphere['binary_sha256'] == BINARY_SHA256 and sphere['source_bundle_sha256'] == BUNDLE_SHA256
        and len(sphere['results']) == 2 and [r['activity'] for r in sphere['results']] == [0.,2.]
        and all(r['physical_returncode'] == r['audit_returncode'] == 0
            and all(c['passed'] is True for c in r['checks']) for r in sphere['results']),
        'Missing matching zero/finite-activity executable reference')
    return plan


def command(root, arm, job):
    root = Path(root); archive = root/arm['id']/'provenance'
    return [str(root/'common/latent-region-normalizer'),'--config',str(archive/'config.json'),
        '--region',str(archive/'region.json'),'--importance-guide',str(archive/'importance-guide.json'),
        '--out',job['directory'],'--samples',str(job['samples']),'--seed',str(job['seed']),
        '--cloud-replicates','2','--lambda-ratio',str(arm['lambda_ratio'])]


def copy_frozen(source, destination):
    for name in [*read(source/'freeze.json')['files'],'freeze.json']:
        target = inside(destination,name); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(inside(source,name),target)


def freeze(out, binary, bundle_path, package, protected_preparation, source_root):
    out,binary,bundle_path,package,protected_preparation,source_root = map(lambda p:Path(p).resolve(),
        (out,binary,bundle_path,package,protected_preparation,source_root))
    require(not out.exists(), 'Fresh validation directory required; no overwrite')
    base_plan,geometry = validate_base_package(package)
    protected_plan = validate_protected(protected_preparation,package,binary,bundle_path)
    inventory = seed_inventory(ROOT,package,protected_preparation)
    bundle,rust = verify_bundle(binary,bundle_path,source_root)
    python = local_dependencies(source_entries(ROOT/'tools'))
    out.mkdir(); common = out/'common'; common.mkdir()
    for name,path in [('latent-region-normalizer',binary),('source-bundle.json',bundle_path),('controller.py',Path(__file__))]:
        shutil.copy2(path,common/name)
    for name,path in python.items(): shutil.copy2(path,common/name)
    for name,entry in bundle['files'].items():
        destination=inside(common/'source',name);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(entry['text'].encode())
    copy_frozen(package,common/'reference-package')
    copy_frozen(protected_preparation,common/'protected-preparation')
    (common/'seed-protocols').mkdir()
    for path,digest in inventory['sources'].items():
        require(sha(path) == digest, 'Seed source changed during freeze')
        shutil.copy2(path,common/'seed-protocols'/(digest+'.json'))
    write(common/'geometry-preflight.json',geometry);write(common/'seed-inventory.json',inventory)
    jobs,arms = [],[]
    for index,spec in enumerate(ARMS):
        arm=dict(spec);base=out/arm['id'];archive=base/'provenance';archive.mkdir(parents=True)
        for folder in ('runs','logs'): (base/folder).mkdir()
        for name,path in python.items(): shutil.copy2(path,archive/name)
        guide = package/'guide-bank.json' if arm['id'] == 'bank' else protected_preparation/'guide-protected.json'
        for name,path in [('source-bundle.json',bundle_path),('latent-region-normalizer',binary),
            ('source-config.json',package/'config.json'),('shape.json',package/'shape.json'),
            ('region.json',package/'region.json'),('importance-guide.json',guide)]: shutil.copy2(path,archive/name)
        config=read(package/'config.json');config['shape']=str(archive/'shape.json');write(archive/'config.json',config)
        arm_jobs=[]
        for i in range(4):
            job=dict(id=f'r{i:02d}',arm=arm['id'],kind='physical',seed=SEEDS[4*index+i],samples=arm['samples'],
                directory=str(base/'runs'/f'r{i:02d}'),log=str(base/'logs'/f'r{i:02d}.log'))
            job['command']=command(out,arm,job);arm_jobs.append(job);jobs.append(job)
        manifest=dict(schema=CAMPAIGN_SCHEMA,jobs=arm_jobs,workers=4,physical_activity=.035,
            lambda_ratio=arm['lambda_ratio'],cloud_replicates=2,config_sha256=sha(archive/'config.json'),
            shape_sha256=sha(archive/'shape.json'),region_sha256=sha(archive/'region.json'),
            importance_guide_sha256=sha(archive/'importance-guide.json'),archive_sha256=file_hashes(archive),allocation=arm)
        write(base/'manifest.json',manifest);arm['manifest_sha256']=sha(base/'manifest.json');arms.append(arm)
    protocol=dict(schema=SCHEMA,arms=arms,jobs=jobs,total_unconditional_draws=TOTAL,maximum_physical_workers=8,
        maximum_total_workers=32,maximum_classification_workers=16,repository=str(ROOT.resolve()),controller_sha256=sha(__file__),
        binary_sha256=sha(binary),source_bundle_sha256=sha(bundle_path),rust_sources=rust,
        python_sources={k:sha(p) for k,p in python.items()},runtime=runtime(),thread_environment=THREAD_ENV,
        preparation_plan_sha256=sha(package/'plan.json'),preparation_freeze_sha256=sha(package/'freeze.json'),
        protected_preparation_plan_sha256=sha(protected_preparation/'plan.json'),
        protected_preparation_freeze_sha256=sha(protected_preparation/'freeze.json'),
        guide_sha256=protected_plan['guide_sha256'],seed_inventory_sha256=sha(common/'seed-inventory.json'),
        references=dict(proposal_validation_sha256=sha(protected_preparation/'proposal-reference/validation.json'),
            proposal_freeze_sha256=sha(protected_preparation/'proposal-reference/freeze.json'),
            sphere_validation_sha256=sha(protected_preparation/'sphere-reference.json')),
        native_definition='common/reference-package/native-region/definition.json',native_definition_sha256=base_plan['native_definition_sha256'],
        region_sha256=base_plan['region_sha256'],shape_sha256=base_plan['shape_sha256'],
        reference_region='common/reference-package/old-r5-region.json',reference_region_sha256=base_plan['reference_region_sha256'],
        supplemental_definition='common/reference-package/native-partition-definition.json',supplemental_definition_sha256=base_plan['supplemental_definition_sha256'],
        decision_arms=DECISION_ARMS,comparisons=COMPARISONS,
        primary_classes=['registered_native_entry','contact_no_native_entry','unbound_no_native_entry'],strata=STRATA,convergence=CONVERGENCE,
        estimator='Unconditional H_R4 H_capture H_hard J W/q; full untruncated mixture; two independent clouds per valid pose; invalid/exterior zeros remain in all original denominators.',
        scope='Fresh frozen validation, not replacement or pooling of earlier evidence. No fitting, retries, adaptive stopping or automatic full-vessel/assembly launch. Small-arm quality is diagnostic; all four predeclared pair comparisons remain mandatory.')
    write(out/'protocol.json',protocol);write(out/'freeze.json',dict(files=file_hashes(out)))
    validate(out);return protocol


def validate(out):
    out=Path(out).resolve();protocol=read(out/'protocol.json');common=out/'common'
    require(protocol['schema'] == SCHEMA and protocol['total_unconditional_draws'] == TOTAL
        and protocol['maximum_physical_workers'] == 8 and protocol['maximum_total_workers'] == 32
        and protocol['maximum_classification_workers'] == 16, 'Wrong validation allocation')
    frozen=read(out/'freeze.json')['files']
    require({'protocol.json','common/controller.py','common/seed-inventory.json'} <= set(frozen), 'Incomplete validation freeze')
    for name,digest in frozen.items(): require(sha(inside(out,name)) == digest,'Frozen file changed: '+name)
    require(sha(common/'controller.py') == protocol['controller_sha256'], 'Controller changed')
    require(sha(common/'latent-region-normalizer') == protocol['binary_sha256'] == BINARY_SHA256
        and sha(common/'source-bundle.json') == protocol['source_bundle_sha256'] == BUNDLE_SHA256, 'Executable/source changed')
    _,rust=verify_bundle(common/'latent-region-normalizer',common/'source-bundle.json',common/'source')
    require(rust == protocol['rust_sources'], 'Rust closure changed')
    closure=local_dependencies(source_entries(common))
    require(set(closure) == set(protocol['python_sources']), 'Python closure changed')
    for name,digest in protocol['python_sources'].items(): require(sha(common/name) == digest,'Python source changed')
    require(protocol['thread_environment'] == THREAD_ENV, 'Worker thread limits changed')
    package=common/'reference-package';preparation=common/'protected-preparation'
    plan,_=validate_base_package(package)
    protected=validate_protected(preparation,package,common/'latent-region-normalizer',common/'source-bundle.json')
    for prefix,source in [('',package),('protected_',preparation)]:
        require(sha(source/'plan.json') == protocol[prefix+'preparation_plan_sha256']
            and sha(source/'freeze.json') == protocol[prefix+'preparation_freeze_sha256'], 'Preparation binding changed')
    require(protocol['guide_sha256'] == protected['guide_sha256'], 'Proposal selection changed')
    require(protocol['references'] == dict(proposal_validation_sha256=sha(preparation/'proposal-reference/validation.json'),
        proposal_freeze_sha256=sha(preparation/'proposal-reference/freeze.json'),
        sphere_validation_sha256=sha(preparation/'sphere-reference.json')), 'Reference bindings changed')
    require(sha(common/'seed-inventory.json') == protocol['seed_inventory_sha256'], 'Seed inventory changed')
    validate_inventory(common,package,preparation)
    for key,name in [('native_definition','native-region/definition.json'),('reference_region','old-r5-region.json'),
                     ('supplemental_definition','native-partition-definition.json')]:
        require(protocol[key] == 'common/reference-package/'+name and sha(inside(out,protocol[key]))
            == protocol[key+'_sha256'] == plan[key+'_sha256'], 'Classifier/partition changed')
    require(protocol['region_sha256'] == plan['region_sha256'] and protocol['shape_sha256'] == plan['shape_sha256'], 'Physical target changed')
    require(protocol['strata'] == STRATA and protocol['convergence'] == CONVERGENCE
        and protocol['decision_arms'] == DECISION_ARMS and protocol['comparisons'] == COMPARISONS,
        'Analysis declaration changed')
    require(len(protocol['arms']) == len(ARMS), 'Wrong arm allocation')
    all_jobs=[]
    for index,(arm,expected) in enumerate(zip(protocol['arms'],ARMS)):
        require({k:arm[k] for k in expected} == expected, 'Arm allocation changed')
        base=out/arm['id'];archive=base/'provenance';manifest=read(base/'manifest.json')
        require(sha(base/'manifest.json') == arm['manifest_sha256'], 'Arm manifest changed')
        require(manifest['schema'] == CAMPAIGN_SCHEMA and len(manifest['jobs']) == 4 and manifest['workers'] == 4
            and manifest['physical_activity'] == .035 and manifest['lambda_ratio'] == arm['lambda_ratio']
            and manifest['cloud_replicates'] == 2 and manifest['allocation'] == expected, 'Arm law/schema changed')
        require(manifest['archive_sha256'] == file_hashes(archive), 'Arm source/input closure changed')
        for name in ('config','region','shape'): require(manifest[name+'_sha256'] == sha(archive/(name+'.json')), 'Arm input changed')
        guide_name='bank' if arm['id'] == 'bank' else 'protected'
        require(sha(archive/'importance-guide.json') == manifest['importance_guide_sha256'] == protected['guide_sha256'][guide_name], 'Guide changed')
        for name in ('region.json','shape.json','source-config.json'):
            require(sha(archive/name) == sha(package/('config.json' if name == 'source-config.json' else name)), 'Arm target changed')
        config=read(archive/'config.json');original=read(package/'config.json');restored=copy.deepcopy(config);restored['shape']=original['shape']
        require(restored == original and config['shape'] == str(archive/'shape.json'), 'Physical config changed')
        for i,job in enumerate(manifest['jobs']):
            require(job['seed'] == SEEDS[4*index+i] and job['samples'] == arm['samples'] and job['kind'] == 'physical'
                and job['arm'] == arm['id'] and job['id'] == f'r{i:02d}' and job['command'] == command(out,arm,job), 'Job law changed')
            require(job['directory'] == str(base/'runs'/job['id']) and job['log'] == str(base/'logs'/f"{job['id']}.log"), 'Job path escaped')
            all_jobs.append(job)
    require(all_jobs == protocol['jobs'] and len(all_jobs) == 16 and sum(j['samples'] for j in all_jobs) == TOTAL, 'Incomplete allocation')
    return protocol


def preflight(out):
    out=Path(out).resolve();protocol=validate(out)
    require(sha(__file__) == protocol['controller_sha256'], 'Use exact archived controller')
    require(runtime() == protocol['runtime'], 'Python audit runtime changed')
    require(not (out/'status.json').exists(), 'Validation already launched; no retries')
    for arm in protocol['arms']:
        base=out/arm['id']
        require(not any((base/'runs').iterdir()) and not any((base/'logs').iterdir())
            and not (base/'assessment').exists(), 'Existing outputs; no overwrite')
    return protocol


def run(out):
    out=Path(out).resolve();protocol=preflight(out)
    state=dict(schema=STATUS_SCHEMA,complete=False,phase='physical',started=time.time(),
        pid=os.getpid(),process_birth=process_token(os.getpid()),protocol_sha256=sha(out/'protocol.json'),
        jobs=[dict(j,status='pending') for j in protocol['jobs']],audits={})
    with (out/'status.json').open('x') as stream: stream.write('{}\n')
    def snapshot():
        write(out/'status.tmp',state);(out/'status.tmp').replace(out/'status.json')
    def interrupted(signum,frame): raise InterruptedError('Controller received signal '+str(signum))
    previous_term=signal.signal(signal.SIGTERM,interrupted);snapshot()
    try:
        execute_group(state['jobs'],snapshot,workers=8,repository=protocol['repository'],env=worker_environment())
        state['phase']='physical_validation';snapshot();validate(out)
        for job in state['jobs']: job['output']=verify_output(out,protocol,job)
        state['phase']='audit';state['audits']={a['id']:audit_step(out,a) for a in protocol['arms']};snapshot()
        execute_group(list(state['audits'].values()),snapshot,workers=1,repository=protocol['repository'],env=worker_environment())
        for arm in protocol['arms']:
            analysis=out/arm['id']/'assessment/analysis.json'
            verify_assessment(out,protocol,arm,read(analysis),state['jobs'])
            state['audits'][arm['id']]['analysis_sha256']=sha(analysis);snapshot()
        validate(out);state.update(complete=True,phase='complete',finished=time.time());snapshot()
    except BaseException as error:
        state.update(phase=state['phase']+'_failed',exception=repr(error),finished=time.time());snapshot();raise
    finally: signal.signal(signal.SIGTERM,previous_term)
    return state


def validate_completed(out):
    out=Path(out).resolve();protocol=validate(out);status=read(out/'status.json')
    require(status['schema'] == STATUS_SCHEMA and status['complete'] is True and status['phase'] == 'complete'
        and status['protocol_sha256'] == sha(out/'protocol.json'), 'Validation is not complete')
    require(len(status['jobs']) == len(protocol['jobs']), 'Terminal allocation changed')
    for frozen,job in zip(protocol['jobs'],status['jobs']):
        require({k:job[k] for k in frozen} == frozen and job['status'] == 'complete' and job['returncode'] == 0, 'Terminal physical job differs')
        require(job['output'] == verify_output(out,protocol,job), 'Terminal output binding changed')
    require(set(status['audits']) == {a['id'] for a in protocol['arms']}, 'Terminal audit allocation changed')
    for arm in protocol['arms']:
        audit=status['audits'][arm['id']];expected=audit_step(out,arm)
        require(all(audit[k] == v for k,v in expected.items() if k != 'status') and audit['status'] == 'complete'
            and audit['returncode'] == 0 and audit['analysis_sha256'] == sha(out/arm['id']/'assessment/analysis.json'), 'Terminal audit binding changed')
        verify_assessment(out,protocol,arm,read(out/arm['id']/'assessment/analysis.json'),status['jobs'])
    return protocol,status


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','validate','preflight','run','validate-completed'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--binary',type=Path)
    parser.add_argument('--source-bundle',type=Path);parser.add_argument('--source-root',type=Path)
    parser.add_argument('--package',type=Path);parser.add_argument('--protected-preparation',type=Path)
    args=parser.parse_args()
    if args.action == 'freeze':
        if any(p is None for p in (args.binary,args.source_bundle,args.source_root,args.package,args.protected_preparation)):
            parser.error('freeze requires binary, source-bundle, source-root, package and protected-preparation')
        result=freeze(args.out,args.binary,args.source_bundle,args.package,args.protected_preparation,args.source_root)
    elif args.action == 'validate-completed': _,result=validate_completed(args.out)
    else: result={'validate':validate,'preflight':preflight,'run':run}[args.action](args.out)
    print(dict(action=args.action,out=str(args.out.resolve()),complete=result.get('complete')))
