#!/usr/bin/env python3
"""Freeze a fresh, bounded two-arm R4 guide pilot; never replace earlier evidence.

The unchanged 80-component bank and the frozen 84-component SMC-geometry guide
receive four independent populations each. No fitting, adaptive allocation,
conditional denominators, classifier changes, retries or assembly launch occur.
"""
from __future__ import annotations
import argparse
import copy
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from analyze_latent_region import LatentImportanceGuide
from prepare_shoulder_docking_benchmark import local_dependencies
import run_contact_confirmation as confirmation
from run_contact_confirmation import (runtime, worker_environment, nonnegative_count,
    THREAD_ENV, CAMPAIGN_SCHEMA, POPULATION_SCHEMA, DENSITY_MEASURE)
from run_entry_shell_reference_campaign import read, write, sha, require, inside, file_hashes
from run_mobile_posterior_pilot import verify_bundle
from run_full_vessel_comparison import execute_group
from run_smc_importance_bridge import process_token

ROOT = Path(__file__).resolve().parents[1]
RUNNER_NAME = 'run_smc_guide_pilot.py'
ANALYZER_NAME = 'analyze_smc_guide_pilot.py'
SCHEMA = 'smc-geometry-guide-pilot-controller-v1'
STATUS_SCHEMA = 'smc-geometry-guide-pilot-status-v1'
ARMS = [dict(id=name, samples=65536, alpha=.5, lambda_ratio=128., component_count=count)
        for name,count in [('bank',80),('smc',84)]]
SEEDS = [137101010+1009*i for i in range(8)]
TOTAL = 524288
GUIDE_SHA256 = '20cbf3cc6081d5ce2f434d4189dc890fa2f3c1d1a4fcad320317bea39e8142e8'
OLD_GUIDE_SHA256 = 'd0e62d1b23d627582ca5f393af515a1ca10b8ecc01ce1a0e75c40e3e18885929'
SPHERE_REFERENCE_SHA256 = '89c75a30cabfb3e9db6965f6f44f5db52134caa99a495d2c7fbcda2b9da90d34'
STRATA = dict(radial_edges=[0.,2.,3.,4.], angular_projection_squared_edges=[0.,4.,9.,16.],
    latent_orthants='six signs, zero assigned positive; all 64 bins retained')
CONVERGENCE = dict(population_relative_SE_max=.1, importance_ESS_min=200., largest_draw_max=.02,
    log_agreement_absolute_max=.2, log_agreement_combined_SE_max=3., deltaF_95_halfwidth_max=.5,
    significant_stratum_mass_fraction=.01, stratum_log_agreement_absolute_max=.2)


def validate_base_package(package):
    # Portable classifier import must not create bytecode in immutable inputs.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        return confirmation.validate_package(package)
    finally:
        sys.dont_write_bytecode = previous


def validate_geometry_preparation(geometry, package):
    geometry, package = Path(geometry).resolve(), Path(package).resolve()
    frozen = read(geometry/'freeze.json')['files']
    require({'plan.json','preparation.json','declaration.json','guide-r5-cov1.json'} <= set(frozen),
            'Incomplete SMC geometry preparation freeze')
    for name,digest in frozen.items():
        require(sha(inside(geometry,name)) == digest, 'SMC geometry preparation changed: '+name)
    plan, preparation = read(geometry/'plan.json'), read(geometry/'preparation.json')
    require(plan['schema'] == preparation['schema'] == 'smc-terminal-geometry-guide-preparation-v1'
            and plan['complete'] is preparation['complete'] is True,
            'Incomplete SMC geometry preparation')
    require(plan['fit_population_ids'] == ['r00','r01'] and plan['heldout_population_ids'] == ['r02','r03'],
            'SMC fitting/held-out population split changed')
    for data in (plan,preparation):
        require(data['all_terminal_slots_retained'] is True and data['native_classifier_calls'] == 0
                and data['physical_draws_launched'] == data['audits_replayed'] == 0,
                'Geometry preparation filtering or physics changed')
    require(sha(geometry/'guide-r5-cov1.json') == GUIDE_SHA256 == plan['guides']['r5-cov1']['sha256']
            == preparation['guides']['r5-cov1']['sha256'], 'Selected SMC guide changed')
    require(sha(package/'guide-bank.json') == OLD_GUIDE_SHA256, 'Old 80-component guide changed')
    original, guide = read(package/'guide-bank.json'), read(geometry/'guide-r5-cov1.json')
    parsed = LatentImportanceGuide(guide, sha(package/'region.json'))
    require(parsed.count == 84 and parsed.alpha == .5, 'Wrong SMC guide mixture')
    require(len(original['gaussian_components']) == 80, 'Old bank component count changed')
    for old,new in zip(original['gaussian_components'],guide['gaussian_components'][:80]):
        require(old['mean'] == new['mean'] and old['covariance'] == new['covariance']
                and new['weight'] == .5*old['weight'], 'Retained old guide geometry/weight changed')
    require(all(c['weight'] == .125 for c in guide['gaussian_components'][80:]),
            'Four new SMC components must share half the Gaussian mass')
    require(all(p['seed'] not in SEEDS for p in plan['populations']), 'SMC training seed collision')
    require(plan['input_sha256'][plan['old_guide']] == OLD_GUIDE_SHA256
            and plan['input_sha256'][plan['current_region']] == sha(package/'region.json')
            and plan['input_sha256'][plan['historical_r5_region']] == sha(package/'old-r5-region.json'),
            'SMC geometry source target changed')
    return plan


def seed_inventory(repository, package):
    """Bind existing campaign declarations; raw rows and completed audits are not replayed."""
    sources, seeds = {}, set()
    for path in sorted((Path(repository)/'runs').glob('*/protocol.json')):
        value = read(path)
        for job in value.get('jobs',[]):
            if isinstance(job,dict) and type(job.get('seed')) is int: seeds.add(job['seed'])
        sources[str(path.resolve())] = sha(path)
    seeds.update(a['seed'] for a in read(Path(package)/'plan.json')['anchor_metadata'])
    require(not set(SEEDS)&seeds, 'Fresh pilot seed collides with an existing declaration')
    return dict(sources=sources, seeds=sorted(seeds), fresh_seeds=SEEDS,
                scope='Top-level runs/*/protocol.json declarations and old-bank training anchors; no raw-row replay')


def validate_references(reference, sphere_reference, binary, bundle_path, package, geometry_preparation):
    reference, sphere_reference = Path(reference), Path(sphere_reference)
    frozen = read(reference/'freeze.json')['files']
    require({'validation.json','declaration.json'} <= set(frozen), 'Missing proposal reference freeze')
    for name,digest in frozen.items():require(sha(inside(reference,name)) == digest, 'Proposal reference changed: '+name)
    report, declaration = read(reference/'validation.json'), read(reference/'declaration.json')
    require(report['schema'] == declaration['schema'] == 'smc-guide-analytic-volume-reference-v1'
        and report['complete'] is True and report['physical_draws'] == report['classifiers_rerun'] == report['audits_replayed'] == 0
        and report['total_proposal_only_draws'] == 131072 and report['declaration_sha256'] == sha(reference/'declaration.json'),
        'Incomplete proposal-only reference')
    require(declaration['samples_per_population'] == 16384 and declaration['populations_per_arm'] == 4
        and declaration['arms'] == ['bank','smc'] and declaration['alpha'] == .5
        and declaration['expected_mean'] == 1. and declaration['weight_bounds'] == [0.,2.]
        and declaration['family_failure_probability'] == 1e-6 and len(declaration['seeds']) == 8
        and len(set(declaration['seeds'])) == 8 and not set(SEEDS)&set(declaration['seeds']), 'Reference allocation changed')
    require(set(report['arms']) == {'bank','smc'} and all(a['passed'] is True for a in report['arms'].values()), 'Proposal reference failed')
    bound=set(declaration['source_sha256'].values())
    require({GUIDE_SHA256,OLD_GUIDE_SHA256,sha(Path(package)/'region.json'),
        sha(Path(geometry_preparation)/'plan.json'),sha(Path(geometry_preparation)/'freeze.json')} <= bound,
        'Proposal reference belongs to another guide or target')
    for path,digest in declaration['code_sha256'].items():
        require(sha(reference/'source'/Path(path).name) == digest, 'Reference source closure changed')
    require(sha(sphere_reference) == SPHERE_REFERENCE_SHA256, 'Analytic sphere reference changed')
    sphere=read(sphere_reference)
    require(sphere['schema'] == 'conditional-ray-sphere-crosslanguage-v1' and sphere['complete'] is True
        and sphere['binary_sha256'] == sha(binary) and sphere['source_bundle_sha256'] == sha(bundle_path)
        and len(sphere['results']) == 2 and [r['activity'] for r in sphere['results']] == [0.,2.]
        and all(r['physical_returncode'] == r['audit_returncode'] == 0 and all(c['passed'] is True for c in r['checks'])
                for r in sphere['results']), 'Reused executable lacks matching analytic sphere proof')
    return dict(proposal_validation_sha256=sha(reference/'validation.json'),proposal_freeze_sha256=sha(reference/'freeze.json'),
                sphere_validation_sha256=sha(sphere_reference))


def source_entries(root):
    return [root/RUNNER_NAME, root/'analyze_latent_region.py', root/ANALYZER_NAME, root/'run_smc_guide_workflow.py']


def command(root, arm, job):
    archive = root/arm['id']/'provenance'
    return [str(root/'common/latent-region-normalizer'), '--config', str(archive/'config.json'),
            '--region', str(archive/'region.json'), '--importance-guide', str(archive/'importance-guide.json'),
            '--out', job['directory'], '--samples', str(job['samples']), '--seed', str(job['seed']),
            '--cloud-replicates', '2', '--lambda-ratio', '128.0']


def freeze(out, binary, bundle_path, package, geometry_preparation, source_root, proposal_reference, sphere_reference):
    out,binary,bundle_path,package,geometry_preparation,source_root,proposal_reference,sphere_reference = map(lambda p:Path(p).resolve(),
        (out,binary,bundle_path,package,geometry_preparation,source_root,proposal_reference,sphere_reference))
    require(not out.exists(), 'Fresh pilot directory required; no overwrite')
    plan,geometry = validate_base_package(package)
    smc_plan = validate_geometry_preparation(geometry_preparation,package)
    references = validate_references(proposal_reference,sphere_reference,binary,bundle_path,package,geometry_preparation)
    inventory = seed_inventory(ROOT,package)
    bundle,rust = verify_bundle(binary,bundle_path,source_root)
    python = local_dependencies(source_entries(ROOT/'tools'))
    out.mkdir();common=out/'common';common.mkdir()
    shutil.copy2(binary,common/'latent-region-normalizer');shutil.copy2(bundle_path,common/'source-bundle.json')
    shutil.copy2(__file__,common/'controller.py')
    for name,path in python.items():shutil.copy2(path,common/name)
    for name,entry in bundle['files'].items():
        destination=inside(common/'source',name);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(entry['text'].encode())
    for source,folder in [(package,'reference-package'),(geometry_preparation,'geometry-preparation'),(proposal_reference,'proposal-reference')]:
        for name in [*read(source/'freeze.json')['files'],'freeze.json']:
            destination=inside(common/folder,name);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(inside(source,name),destination)
    shutil.copy2(sphere_reference,common/'sphere-reference.json')
    write(common/'geometry-preflight.json',geometry);write(common/'seed-inventory.json',inventory)
    jobs,arms=[],[]
    for arm_index,spec in enumerate(ARMS):
        arm=dict(spec);base=out/arm['id'];archive=base/'provenance';archive.mkdir(parents=True)
        for folder in ('runs','logs'):(base/folder).mkdir()
        for name,path in python.items():shutil.copy2(path,archive/name)
        guide=package/'guide-bank.json' if arm['id']=='bank' else geometry_preparation/'guide-r5-cov1.json'
        for name,path in [('source-bundle.json',bundle_path),('latent-region-normalizer',binary),
                ('source-config.json',package/'config.json'),('shape.json',package/'shape.json'),
                ('region.json',package/'region.json'),('importance-guide.json',guide)]:shutil.copy2(path,archive/name)
        config=read(package/'config.json');config['shape']=str(archive/'shape.json');write(archive/'config.json',config)
        arm_jobs=[]
        for i in range(4):
            job=dict(id=f'r{i:02d}',arm=arm['id'],kind='physical',seed=SEEDS[4*arm_index+i],samples=arm['samples'],
                directory=str(base/'runs'/f'r{i:02d}'),log=str(base/'logs'/f'r{i:02d}.log'))
            job['command']=command(out,arm,job);arm_jobs.append(job);jobs.append(job)
        manifest=dict(schema=CAMPAIGN_SCHEMA,jobs=arm_jobs,workers=4,physical_activity=.035,
            lambda_ratio=128.,cloud_replicates=2,config_sha256=sha(archive/'config.json'),
            shape_sha256=sha(archive/'shape.json'),region_sha256=sha(archive/'region.json'),
            importance_guide_sha256=sha(archive/'importance-guide.json'),archive_sha256=file_hashes(archive),allocation=arm)
        write(base/'manifest.json',manifest);arm['manifest_sha256']=sha(base/'manifest.json');arms.append(arm)
    protocol=dict(schema=SCHEMA,arms=arms,jobs=jobs,total_unconditional_draws=TOTAL,maximum_physical_workers=8,
        maximum_total_workers=32,repository=str(ROOT.resolve()),
        controller_sha256=sha(__file__),binary_sha256=sha(binary),source_bundle_sha256=sha(bundle_path),
        rust_sources=rust,python_sources={k:sha(p) for k,p in python.items()},runtime=runtime(),thread_environment=THREAD_ENV,
        preparation_plan_sha256=sha(package/'plan.json'),preparation_freeze_sha256=sha(package/'freeze.json'),
        geometry_preparation_plan_sha256=sha(geometry_preparation/'plan.json'),
        geometry_preparation_freeze_sha256=sha(geometry_preparation/'freeze.json'),
        selected_geometry_guide='r5-cov1',geometry_guide_sha256=GUIDE_SHA256,old_guide_sha256=OLD_GUIDE_SHA256,
        seed_inventory_sha256=sha(common/'seed-inventory.json'),references=references,
        native_definition='common/reference-package/native-region/definition.json',native_definition_sha256=plan['native_definition_sha256'],
        region_sha256=plan['region_sha256'],shape_sha256=plan['shape_sha256'],
        reference_region='common/reference-package/old-r5-region.json',reference_region_sha256=plan['reference_region_sha256'],
        supplemental_definition='common/reference-package/native-partition-definition.json',supplemental_definition_sha256=plan['supplemental_definition_sha256'],
        decision_arms=['bank','smc'],comparisons=[['bank','smc']],
        primary_classes=['registered_native_entry','contact_no_native_entry','unbound_no_native_entry'],
        strata=STRATA,convergence=CONVERGENCE,
        estimator='Unconditional H_R4 H_capture H_hard J W/q; full untruncated mixture and two independent clouds per valid pose. All exterior and invalid draws remain zero in the original denominator.',
        mandatory_native_partition='old R5 chart radius <=5, original q>1, D170, intersected with full native R4; complementary full-native subset is also primary',
        scope='Fresh bounded pilot only. No fitting, adaptive stopping, retries, old-population pooling, larger allocation, or automatic assembly launch. Prior failed gates remain unchanged. Matching fixed-region evidence alone cannot establish finite-system assembly.')
    write(out/'protocol.json',protocol);write(out/'freeze.json',dict(files=file_hashes(out)))
    validate(out);return protocol


def validate(out):
    out=Path(out).resolve();protocol=read(out/'protocol.json');common=out/'common'
    require(protocol['schema']==SCHEMA and protocol['total_unconditional_draws']==TOTAL
        and protocol['maximum_physical_workers']==8 and protocol['maximum_total_workers']==32,
        'Wrong pilot allocation')
    frozen=read(out/'freeze.json')['files']
    require({'protocol.json','common/controller.py','common/seed-inventory.json'}<=set(frozen), 'Incomplete pilot freeze')
    for name,digest in frozen.items():require(sha(inside(out,name))==digest,'Frozen file changed: '+name)
    require(sha(common/'controller.py')==protocol['controller_sha256'],'Controller changed')
    require(sha(common/'latent-region-normalizer')==protocol['binary_sha256']
        and sha(common/'source-bundle.json')==protocol['source_bundle_sha256'],'Executable/source changed')
    _,rust=verify_bundle(common/'latent-region-normalizer',common/'source-bundle.json',common/'source')
    require(rust==protocol['rust_sources'],'Rust closure changed')
    closure=local_dependencies(source_entries(common))
    require(set(closure)==set(protocol['python_sources']),'Python closure changed')
    for name,digest in protocol['python_sources'].items():require(sha(common/name)==digest,'Python source changed')
    require(protocol['thread_environment']==THREAD_ENV,'Worker limits changed')
    package=common/'reference-package';plan,_=validate_base_package(package)
    validate_geometry_preparation(common/'geometry-preparation',package)
    require(validate_references(common/'proposal-reference',common/'sphere-reference.json',common/'latent-region-normalizer',
        common/'source-bundle.json',package,common/'geometry-preparation') == protocol['references'], 'Reference bindings changed')
    for prefix,source in [('',package),('geometry_',common/'geometry-preparation')]:
        require(sha(source/'plan.json')==protocol[prefix+'preparation_plan_sha256']
            and sha(source/'freeze.json')==protocol[prefix+'preparation_freeze_sha256'],'Preparation binding changed')
    require(protocol['selected_geometry_guide']=='r5-cov1' and protocol['geometry_guide_sha256']==GUIDE_SHA256
        and protocol['old_guide_sha256']==OLD_GUIDE_SHA256,'Proposal selection changed')
    require(sha(common/'seed-inventory.json')==protocol['seed_inventory_sha256'],'Seed declaration changed')
    inventory=read(common/'seed-inventory.json')
    require(inventory['fresh_seeds']==SEEDS and not set(inventory['seeds'])&set(SEEDS),'Seed collision')
    for key,name in [('native_definition','native-region/definition.json'),('reference_region','old-r5-region.json'),
                     ('supplemental_definition','native-partition-definition.json')]:
        require(protocol[key]=='common/reference-package/'+name and sha(inside(out,protocol[key]))
            ==protocol[key+'_sha256']==plan[key+'_sha256'],'Classifier/partition changed')
    require(protocol['region_sha256']==plan['region_sha256'] and protocol['shape_sha256']==plan['shape_sha256'],'Physical target changed')
    require(protocol['strata']==STRATA and protocol['convergence']==CONVERGENCE
        and protocol['decision_arms']==['bank','smc'] and protocol['comparisons']==[['bank','smc']], 'Analysis declaration changed')
    require(len(protocol['arms'])==2,'Wrong arm allocation')
    all_jobs=[]
    for index,(arm,expected) in enumerate(zip(protocol['arms'],ARMS)):
        require({k:arm[k] for k in expected}==expected,'Arm allocation changed')
        base=out/arm['id'];archive=base/'provenance';manifest=read(base/'manifest.json')
        require(sha(base/'manifest.json')==arm['manifest_sha256'],'Arm manifest changed')
        require(manifest['schema']==CAMPAIGN_SCHEMA and len(manifest['jobs'])==4 and manifest['workers']==4
            and manifest['physical_activity']==.035 and manifest['lambda_ratio']==128. and manifest['cloud_replicates']==2,
            'Arm law/schema changed')
        require(manifest['archive_sha256']==file_hashes(archive),'Arm source/input closure changed')
        for name in ('config','region','shape'):require(manifest[name+'_sha256']==sha(archive/(name+'.json')),'Arm input changed')
        require(sha(archive/'importance-guide.json')==manifest['importance_guide_sha256']
            ==(OLD_GUIDE_SHA256 if index==0 else GUIDE_SHA256),'Guide changed')
        for name in ('region.json','shape.json','source-config.json'):
            require(sha(archive/name)==sha(package/('config.json' if name=='source-config.json' else name)), 'Arm target changed')
        config=read(archive/'config.json');original=read(package/'config.json');restored=copy.deepcopy(config);restored['shape']=original['shape']
        require(restored==original and config['shape']==str(archive/'shape.json'),'Physical config changed')
        for i,job in enumerate(manifest['jobs']):
            require(job['seed']==SEEDS[4*index+i] and job['samples']==65536 and job['kind']=='physical'
                and job['arm']==arm['id'] and job['id']==f'r{i:02d}' and job['command']==command(out,arm,job),'Job law changed')
            require(job['directory']==str(base/'runs'/job['id']) and job['log']==str(base/'logs'/f"{job['id']}.log"),'Job path escaped')
            all_jobs.append(job)
    require(all_jobs==protocol['jobs'] and len(all_jobs)==8 and sum(j['samples'] for j in all_jobs)==TOTAL,'Incomplete allocation')
    return protocol


def preflight(out):
    out=Path(out).resolve();protocol=validate(out)
    require(sha(__file__)==protocol['controller_sha256'],'Use exact archived controller')
    require(runtime()==protocol['runtime'],'Python audit runtime changed')
    require(not (out/'status.json').exists(),'Pilot already launched; no retries')
    for arm in protocol['arms']:
        base=out/arm['id']
        require(not any((base/'runs').iterdir()) and not any((base/'logs').iterdir())
            and not (base/'assessment').exists(),'Existing outputs; no overwrite')
    return protocol


def verify_output(out, protocol, job):
    root = Path(out)/job['arm']; manifest = read(root/'manifest.json')
    arm = next(a for a in protocol['arms'] if a['id'] == job['arm'])
    directory = Path(job['directory']); summary = read(directory/'summary.json'); pm = read(directory/'manifest.json')
    require(summary['complete'] is True and summary['manifest'] == pm and summary['samples'] == job['samples'], 'Incomplete physical population')
    expected = dict(schema=POPULATION_SCHEMA, samples=job['samples'], seed=job['seed'], cloud_replicates=2,
        activity=.035, lambda_ratio=arm['lambda_ratio'], importance_uniform_probability=arm['alpha'], importance_component_count=arm['component_count'],
        proposal_density_measure=DENSITY_MEASURE, executable_sha256=protocol['binary_sha256'],
        source_bundle_sha256=protocol['source_bundle_sha256'], minimum_latent_radius=0., latent_radius=4.,
        minimum_original_q=0., maximum_original_q=None, minimum_original_q_inclusive=True, maximum_original_q_inclusive=True,
        physical_fixed_neighbors=read(root/'provenance/config.json')['fixed_poses'], chart_anchor=read(root/'provenance/region.json')['fixed_neighbor'])
    expected['lambda'] = .035*arm['lambda_ratio']
    for key in ('region_sha256', 'shape_sha256', 'config_sha256', 'importance_guide_sha256'): expected[key] = manifest[key]
    require(all(pm.get(k) == v for k, v in expected.items()), 'Population law changed')
    require(summary['estimates']['region']['draws'] == summary['estimates']['hard_region']['draws'] == job['samples'], 'Unconditional denominator changed')
    require(nonnegative_count(summary['shell_rejected']) and summary['shell_rejected'] <= job['samples'], 'Invalid exterior draw count')
    require(summary['samples_sha256'] == sha(directory/'samples.jsonl'), 'Rows changed')
    require(type(summary['sampler_cpu_seconds']) in (int, float) and math.isfinite(summary['sampler_cpu_seconds'])
            and summary['sampler_cpu_seconds'] >= 0., 'Invalid sampler CPU time')
    for name, field in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'), ('shape.json', 'shape_sha256'),
                        ('importance-guide.json', 'importance_guide_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        require(sha(directory/'provenance'/name) == pm[field], 'Population provenance changed')
    return dict(samples_sha256=sha(directory/'samples.jsonl'), summary_sha256=sha(directory/'summary.json'),
                manifest_sha256=sha(directory/'manifest.json'), sampler_cpu_seconds=summary['sampler_cpu_seconds'],
                shell_rejected=summary['shell_rejected'])


def verify_assessment(out, protocol, arm, analysis, jobs):
    base = Path(out)/arm['id']; manifest = read(base/'manifest.json'); selected = [j for j in jobs if j['arm'] == arm['id']]
    require(len(selected) == 4 and len({j['id'] for j in selected}) == 4, 'Wrong audit allocation')
    total = 4*arm['samples']; guide = analysis['importance_sampling']
    require(analysis['region_sha256'] == protocol['region_sha256'] == manifest['region_sha256'] and
            analysis['physical_fixed_neighbors'] == read(base/'provenance/config.json')['fixed_poses'], 'Audit physical target changed')
    require(analysis['estimate']['draws'] == analysis['hard_region']['draws'] == total == analysis['independently_reconstructed_poses'],
            'Audit dropped unconditional draws')
    require(guide['guide_sha256'] == manifest['importance_guide_sha256'] and guide['gaussian_component_count'] == arm['component_count']
            and guide['uniform_shell_probability'] == arm['alpha'] and guide['density_measure'] == DENSITY_MEASURE
            and guide['draws'] == total, 'Audit guide identity differs')
    populations = analysis['populations']
    require(len(populations) == 4 and {p['id'] for p in populations} == {j['id'] for j in selected}, 'Audit population missing/repeated')
    aggregate = {'uniform-shell': 0, 'gaussian': 0}; exterior = 0
    for job in selected:
        population = next(p for p in populations if p['id'] == job['id']); audit = population['importance_sampling_audit']
        require(population['seed'] == job['seed'] and population['estimate']['draws'] == population['hard_region']['draws'] == job['samples']
                and population['samples_sha256'] == job['output']['samples_sha256'], 'Audit population binding changed')
        counts = audit['branch_counts']; components = audit['gaussian_component_counts']
        require(audit['draws'] == job['samples'] and set(counts) == set(aggregate) and all(nonnegative_count(v) for v in counts.values())
                and sum(counts.values()) == job['samples'] and len(components) == arm['component_count']
                and all(nonnegative_count(v) for v in components) and sum(components) == counts['gaussian'], 'Audit component denominator changed')
        require(nonnegative_count(audit['shell_rejected']) and audit['shell_rejected'] <= counts['gaussian']
                and audit['shell_rejected'] == job['output']['shell_rejected'], 'Audit exterior zeros changed')
        for key in ('maximum_log_proposal_density_error', 'maximum_latent_vector_reconstruction_error', 'maximum_log_jacobian_error'):
            require(type(audit[key]) in (int, float) and math.isfinite(audit[key]) and 0. <= audit[key] < 2e-8, 'Audit reconstruction failed')
        aggregate = {k: v+counts[k] for k, v in aggregate.items()}; exterior += audit['shell_rejected']
        for name in ('samples', 'manifest', 'summary'):
            extension = 'jsonl' if name == 'samples' else 'json'
            require(sha(Path(job['directory'])/f'{name}.{extension}') == job['output'][name+'_sha256'], 'Output changed during audit')
    require(set(guide['branch_counts']) == set(aggregate) and all(nonnegative_count(v) for v in guide['branch_counts'].values())
            and guide['branch_counts'] == aggregate, 'Aggregate branch counts differ')
    require(nonnegative_count(guide['shell_rejected']) and guide['shell_rejected'] == exterior, 'Aggregate exterior zeros differ')


def audit_step(out,arm):
    base=Path(out)/arm['id']
    return dict(id=arm['id'],arm=arm['id'],kind='audit',directory=str(base/'assessment'),
        log=str(base/'logs/audit.log'),status='pending',
        command=[sys.executable,'-B',str(base/'provenance/analyze_latent_region.py'),'--root',str(base)])


def run(out):
    out=Path(out).resolve();protocol=preflight(out)
    state=dict(schema=STATUS_SCHEMA,complete=False,phase='physical',started=time.time(),
        pid=os.getpid(),process_birth=process_token(os.getpid()),protocol_sha256=sha(out/'protocol.json'),
        jobs=[dict(j,status='pending') for j in protocol['jobs']],audits={})
    with (out/'status.json').open('x') as stream:stream.write('{}\n')
    def snapshot():
        write(out/'status.tmp',state);(out/'status.tmp').replace(out/'status.json')
    def interrupted(signum,frame):raise InterruptedError('Controller received signal '+str(signum))
    previous_term=signal.signal(signal.SIGTERM,interrupted)
    snapshot()
    try:
        execute_group(state['jobs'],snapshot,workers=8,repository=protocol['repository'],env=worker_environment())
        state['phase']='physical_validation';snapshot();validate(out)
        for job in state['jobs']:job['output']=verify_output(out,protocol,job)
        state['phase']='audit';state['audits']={a['id']:audit_step(out,a) for a in protocol['arms']};snapshot()
        execute_group(list(state['audits'].values()),snapshot,workers=1,repository=protocol['repository'],env=worker_environment())
        for arm in protocol['arms']:
            analysis=out/arm['id']/'assessment/analysis.json'
            verify_assessment(out,protocol,arm,read(analysis),state['jobs'])
            state['audits'][arm['id']]['analysis_sha256']=sha(analysis);snapshot()
        validate(out);state.update(complete=True,phase='complete',finished=time.time());snapshot()
    except BaseException as error:
        state.update(phase=state['phase']+'_failed',exception=repr(error),finished=time.time());snapshot();raise
    finally:signal.signal(signal.SIGTERM,previous_term)
    return state


def validate_completed(out):
    """Rebind raw output, archived audits and all attempted denominators."""
    out=Path(out).resolve();protocol=validate(out);status=read(out/'status.json')
    require(status['schema']==STATUS_SCHEMA and status['complete'] is True and status['phase']=='complete'
        and status['protocol_sha256']==sha(out/'protocol.json'),'Pilot is not complete')
    require(len(status['jobs'])==len(protocol['jobs']),'Terminal allocation changed')
    for frozen,job in zip(protocol['jobs'],status['jobs']):
        require({k:job[k] for k in frozen}==frozen and job['status']=='complete' and job['returncode']==0,'Terminal physical job differs')
        require(job['output']==verify_output(out,protocol,job),'Terminal output binding changed')
    require(set(status['audits'])=={a['id'] for a in protocol['arms']},'Terminal audit allocation changed')
    for arm in protocol['arms']:
        audit=status['audits'][arm['id']];expected=audit_step(out,arm)
        require(all(audit[k]==v for k,v in expected.items() if k!='status') and audit['status']=='complete'
            and audit['returncode']==0 and audit['analysis_sha256']==sha(out/arm['id']/'assessment/analysis.json'),
            'Terminal audit binding changed')
        verify_assessment(out,protocol,arm,read(out/arm['id']/'assessment/analysis.json'),status['jobs'])
    return protocol,status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','validate','preflight','run','validate-completed'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--binary',type=Path)
    parser.add_argument('--source-bundle',type=Path);parser.add_argument('--source-root',type=Path)
    parser.add_argument('--package',type=Path);parser.add_argument('--geometry-preparation',type=Path)
    parser.add_argument('--proposal-reference',type=Path);parser.add_argument('--sphere-reference',type=Path)
    args=parser.parse_args()
    if args.action=='freeze':
        if any(p is None for p in (args.binary,args.source_bundle,args.source_root,args.package,args.geometry_preparation,args.proposal_reference,args.sphere_reference)):
            parser.error('freeze requires binary, source-bundle, source-root, package, geometry-preparation, proposal-reference and sphere-reference')
        result=freeze(args.out,args.binary,args.source_bundle,args.package,args.geometry_preparation,args.source_root,args.proposal_reference,args.sphere_reference)
    elif args.action=='validate-completed':
        protocol,result=validate_completed(args.out)
    else:result={'validate':validate,'preflight':preflight,'run':run}[args.action](args.out)
    print(dict(action=args.action,out=str(args.out.resolve()),complete=result.get('complete')))
