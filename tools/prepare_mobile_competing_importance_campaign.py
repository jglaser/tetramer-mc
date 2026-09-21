#!/usr/bin/env python3
"""Freeze eight fresh defensive-importance populations, without executing them."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
from analyze_latent_region import LatentImportanceGuide
from prepare_shoulder_docking_benchmark import local_dependencies
from run_mobile_posterior_pilot import inside, read, require, safe_relative, sha, verify_bundle, write

ROOT=Path(__file__).resolve().parents[1]
PREPARATION=ROOT/'runs/mobile-competing-importance-preparation-20260921'
SOURCE_REVIEW=ROOT/'runs/mobile-competing-importance-source-review-20260921'
SOURCE_REVIEW_SHA='eb82773c7eb5c4f8a1ea01744a85537f88c8754d9e9295c2bc88901ee77b8e35'
PYTHON_REVIEW_SHA='c1543dd0fde8dd7a625e03b1d8dfff3895a804bfca4777b9b6f31bcee05c42c0'
PREPARATION_SHA='d07cea08ad2a6e609c425009e86ed296b2fbfc8027d6786b21515e28dd1801d0'
BINARY_SHA='55fd708b5c58477be516f710d78abbe371ede05e8b621e7e3ddd4ac86ebe8b55'
BUNDLE_SHA='033d3776f2b49fbb66694b52334c284ae8bdbd95c3d310ceb7ba98bbaa18b302'
CONFIG_SHA='0ccfc3a533a42721dcfc66ec126e6f621847c4456feccb467820656deef9a0b1'
SHAPE_SHA='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
SHELLS={
    'competitor-shell-5-8':dict(region_sha256='70ed03f1bd610a68efff6fd90d5f8d53873d83c7d452747066d233465c3536c0',
        guide_sha256='b561fe37d8a59042b7aa12f79b723fce23ffdf9defe707638dec920bba6d63f9',inner=5.,outer=8.),
    'competitor-shell-8-12':dict(region_sha256='f4925d133f4d8210c52c6ab4da087aa43555b44662be478d8cdc859776d1a266',
        guide_sha256='3e1e4f881cf5eb9835d45f96665d7a9b0feaef525661b7bb5295ea0847756c82',inner=8.,outer=12.),
}
SCHEMA='mobile-competing-guided-outer-controller-v1'
SAMPLES,REPLICATES,WORKERS,CLOUDS,LAMBDA_RATIO=8192,4,8,2,64.
SEED_BASE=120501010
ANALYZER_NAME='analyze_latent_region.py'
RUNNER_NAME='run_mobile_competing_importance_campaign.py'
SCOPE=('Fresh fixed-size estimates of the unchanged finite competitor shells 5<rho<=8 and 8<rho<=12, with original q>1, '
       'the original two fixed neighbors, hard protein shape, activity, depletant radius and capture domain. '
       'Frozen alpha=0.5 guides have 32 untruncated latent Gaussian components each. All unconditional draws, including '
       'outside-shell and invalid zeros, enter J/q-weighted estimates. Previous rows selected the guides and are training '
       'data only; none are reused in fresh estimates. No fitting or adaptation during production. These shells do not '
       'represent complete competing basins or bound outside mass; the mobile return at rho18.84 remains outside both targets.')


def config_for(original,archive):
    result=copy.deepcopy(original);result['shape']=str(archive/'shape.json')
    return result


def command_for(folder,job):
    archive=folder/'provenance'
    return [str(archive/'latent-region-normalizer'),'--config',str(archive/'config.json'),
        '--region',str(archive/'region.json'),'--out',job['directory'],'--samples',str(SAMPLES),
        '--seed',str(job['seed']),'--lambda-ratio',str(LAMBDA_RATIO),'--cloud-replicates',str(CLOUDS),
        '--importance-guide',str(archive/'importance-guide.json')]


def jobs_for(folder,shell_index):
    jobs=[]
    for replicate in range(REPLICATES):
        name=f'r{replicate:02d}'
        job=dict(id=name,replicate=replicate,index=REPLICATES*shell_index+replicate,
            seed=SEED_BASE+1009*(REPLICATES*shell_index+replicate),samples=SAMPLES,
            directory=str(folder/'runs'/name),log=str(folder/'logs'/f'{name}.log'))
        job['command']=command_for(folder,job);jobs.append(job)
    return jobs


def check_physics(original,region,guide,shell):
    expected=SHELLS[shell]
    require(region['minimum_mahalanobis_radius']==expected['inner'] and region['mahalanobis_radius']==expected['outer'],'Target shell changed')
    require(region['minimum_original_q']==1. and region['minimum_original_q_inclusive'] is False,'Require the original q>1 window')
    require(region.get('maximum_original_q') is None,'Unexpected new upper q bound')
    require(region['physical_fixed_neighbors']==original['fixed_poses'] and region['fixed_neighbor'] in original['fixed_poses'],'Fixed scaffold changed')
    for key,cfgkey in [('physical_metric','metadata'),('activity','reservoir_density'),('depletant_radius','depletant_radius'),
                       ('capture_center','capture_center'),('capture_radius','capture_radius')]:
        require(region[key]==original[cfgkey],'Physical target differs: '+key)
    require(region['shape_sha256']==SHAPE_SHA and region['gaussian_chart']['shape_sha256']==SHAPE_SHA,'Wrong hard shape')
    law=LatentImportanceGuide(guide,expected['region_sha256'])
    require(law.alpha==.5 and law.count==32,'Require frozen 32-component half-uniform guide')
    require(original.get('target_region') is None,'Unexpected docking-only target constraint')
    return law


def freeze(out,preparation=PREPARATION):
    out,preparation=Path(out).resolve(),Path(preparation).resolve()
    require(not out.exists(),'Fresh campaign directory required')
    require(not inside(out,preparation) and not inside(preparation,out),'Keep preparation and campaign separate')
    require(sha(preparation/'plan.json')==PREPARATION_SHA,'Guide preparation identity changed')
    for name,digest in read(preparation/'freeze.json').items():
        require(sha(preparation/safe_relative(name))==digest,'Frozen guide preparation changed: '+name)
    plan=read(preparation/'plan.json');source=preparation/'provenance'
    require(plan['schema']=='mobile-defensive-importance-guide-preparation-v1' and plan['production_launched'] is False
        and plan['new_physical_draws']==0,'Require the inert guide preparation')
    require(plan['validated_binary_sha256']==BINARY_SHA and plan['validated_source_bundle_sha256']==BUNDLE_SHA,'Validation executable differs')
    binary,bundle_path=source/'latent-region-normalizer',source/'source-bundle.json'
    require(sha(binary)==BINARY_SHA and sha(bundle_path)==BUNDLE_SHA,'Reviewed binary/bundle changed')
    require(sha(SOURCE_REVIEW/'validation.json')==SOURCE_REVIEW_SHA,'Reviewed source archive identity changed')
    require(sha(SOURCE_REVIEW/'python-closure-complete.json')==PYTHON_REVIEW_SHA,'Reviewed Python closure identity changed')
    bundle,rust_sources=verify_bundle(binary,bundle_path,SOURCE_REVIEW/'source')
    require(rust_sources==read(SOURCE_REVIEW/'validation.json')['source_sha256'],'Reviewed source map changed')
    python_review=read(SOURCE_REVIEW/'python-closure-complete.json')
    for name,digest in python_review['source_sha256'].items():
        require(sha(SOURCE_REVIEW/'python'/safe_relative(name))==digest,'Reviewed audit dependency changed: '+name)
    require(sha(source/'physical-config.json')==CONFIG_SHA,'Original physical configuration changed')
    original=read(source/'physical-config.json');shape=Path(original['shape'])
    require(shape.is_absolute() and sha(shape)==SHAPE_SHA,'Original hard shape changed')
    guide_records={g['region']:g for g in plan['guides']}
    require(len(plan['guides'])==2 and set(guide_records)==set(SHELLS),'Wrong frozen guide pair')
    for shell,expected in SHELLS.items():
        region_path=source/f'region-{shell}.json';guide_path=preparation/f'guide-{shell}.json'
        require(sha(region_path)==expected['region_sha256']==guide_records[shell]['region_sha256'],'Target region bytes changed')
        require(sha(guide_path)==expected['guide_sha256']==guide_records[shell]['guide_sha256'],'Guide bytes changed')
        check_physics(original,read(region_path),read(guide_path),shell)
    dependencies=local_dependencies([Path(__file__),Path(__file__).with_name(RUNNER_NAME),
        Path(__file__).with_name(ANALYZER_NAME),Path(__file__).with_name('run_latent_region_campaign.py')])
    # Subsequent unrelated normalizer/auditor development must not enter this
    # already reviewed campaign. Import and literal-filename dependencies are immutable.
    dependencies.update({name:SOURCE_REVIEW/'python'/name for name in python_review['source_sha256']})
    top=out/'provenance';top.mkdir(parents=True)
    for name,path in dependencies.items():shutil.copy2(path,top/name)
    shutil.copy2(preparation/'plan.json',top/'guide-preparation-plan.json')
    shutil.copy2(preparation/'freeze.json',top/'guide-preparation-freeze.json')
    campaigns=[]
    for index,(shell,expected) in enumerate(SHELLS.items()):
        folder=out/shell;archive=folder/'provenance';archive.mkdir(parents=True)
        sources=dict(dependencies)
        sources.update({'latent-region-normalizer':binary,'source-bundle.json':bundle_path,
            'shape.json':shape,'original-config.json':source/'physical-config.json',
            'region.json':source/f'region-{shell}.json','importance-guide.json':preparation/f'guide-{shell}.json',
            'guide-preparation-plan.json':preparation/'plan.json','physical-preflight.json':source/'physical-preflight.json',
            'physical-protocol.json':source/'physical-protocol.json','implementation-validation.json':source/'implementation-validation.json',
            'reviewed-source.json':SOURCE_REVIEW/'validation.json','reviewed-python.json':SOURCE_REVIEW/'python-closure-complete.json'})
        input_hashes={name:sha(path) for name,path in sources.items()}
        for name,path in sources.items():
            destination=archive/safe_relative(name);destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination);require(sha(destination)==input_hashes[name],'Input changed during freeze: '+name)
        for name,record in bundle['files'].items():
            target=archive/'source'/safe_relative(name);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(record['text']);require(sha(target)==rust_sources[name],'Embedded source copy differs')
        write(archive/'config.json',config_for(original,archive))
        # This is the unchanged chart already bound inside exact region bytes.
        write(archive/'chart-model.json',read(archive/'region.json')['gaussian_chart'])
        jobs=jobs_for(folder,index)
        for name in ('runs','logs'):(folder/name).mkdir()
        archive_hashes={p.relative_to(archive).as_posix():sha(p) for p in sorted(archive.rglob('*')) if p.is_file()}
        manifest=dict(schema='importance-latent-region-campaign-v1',region=shell,frozen=True,production_launched=False,
            jobs=jobs,workers=4,physical_activity=original['reservoir_density'],lambda_ratio=LAMBDA_RATIO,
            cloud_replicates=CLOUDS,region_sha256=expected['region_sha256'],importance_guide_sha256=expected['guide_sha256'],
            archive_sha256=archive_hashes,source_inputs={name:str(path) for name,path in sources.items()},
            config_sha256=sha(archive/'config.json'),original_config_sha256=CONFIG_SHA,shape_sha256=SHAPE_SHA,
            chart_model_sha256=sha(archive/'chart-model.json'),binary_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,
            rust_sources=rust_sources,observer_sha256=sha(archive/ANALYZER_NAME),
            guide_preparation_plan_sha256=PREPARATION_SHA,preparation_source=str(preparation),
            orchestration='Frozen driver launches the eight population commands directly; the original launcher is archived with its full dependency closure for provenance. No nested workers.',
            scope=SCOPE)
        write(folder/'manifest.json',manifest)
        campaigns.append(dict(region=shell,path=str(folder),manifest_sha256=sha(folder/'manifest.json'),
            region_sha256=expected['region_sha256'],guide_sha256=expected['guide_sha256'],
            samples_per_population=SAMPLES,expected_jobs=[{k:j[k] for k in ('id','replicate','seed','samples')} for j in jobs]))
    protocol=dict(schema=SCHEMA,production_launched=False,campaigns=campaigns,maximum_physical_workers=WORKERS,
        samples_per_population=SAMPLES,populations_per_region=REPLICATES,total_unconditional_draws=2*REPLICATES*SAMPLES,
        total_jobs=8,lambda_ratio=LAMBDA_RATIO,cloud_replicates=CLOUDS,
        controller_sha256=sha(Path(__file__).with_name(RUNNER_NAME)),freezer_sha256=sha(__file__),
        physical_executable_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,guide_preparation_plan_sha256=PREPARATION_SHA,
        seed_policy='Eight fresh streams 120501010 + 1009*j, j=0,...,7, shell-major then replicate-major.',
        stopping='Exactly 8192 draws in each of eight populations. Any launch/physical failure drains all started children and forbids retries and audits. Audit each shell once, only after all physical jobs and source bindings pass.',
        inference='Report separate fixed-N population estimates, all-zero populations, observed ESS and largest weights, paired-cloud variance, CPU and guide audit. No conditioning on validity, no training-row reuse, no optional stopping.',scope=SCOPE)
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    validate(out)
    return protocol


def validate(out):
    out=Path(out).resolve();protocol=read(out/'protocol.json')
    require(protocol['schema']==SCHEMA,'Unknown guided control schema')
    require((protocol['maximum_physical_workers'],protocol['samples_per_population'],protocol['populations_per_region'],
        protocol['total_unconditional_draws'],protocol['cloud_replicates'],protocol['lambda_ratio'],protocol['total_jobs'])
        ==(WORKERS,SAMPLES,REPLICATES,65536,CLOUDS,LAMBDA_RATIO,8),'Fixed experiment allocation changed')
    require(protocol['physical_executable_sha256']==BINARY_SHA and protocol['source_bundle_sha256']==BUNDLE_SHA,'Executable binding differs')
    require(protocol['guide_preparation_plan_sha256']==PREPARATION_SHA,'Preparation binding differs')
    for name,digest in read(out/'freeze.json')['files'].items():
        require(sha(out/safe_relative(name))==digest,'Frozen campaign changed: '+name)
    entries=protocol['campaigns'];require(len(entries)==2 and [e['region'] for e in entries]==list(SHELLS),'Wrong shell pair/order')
    seeds=[];originals=[]
    for index,entry in enumerate(entries):
        shell=entry['region'];folder=out/shell;archive=folder/'provenance';expected=SHELLS[shell]
        require(entry['path']==str(folder) and sha(folder/'manifest.json')==entry['manifest_sha256'],'Campaign manifest binding differs')
        manifest=read(folder/'manifest.json')
        require(manifest['schema']=='importance-latent-region-campaign-v1' and manifest['region']==shell,'Wrong shell campaign schema')
        require(manifest['region_sha256']==entry['region_sha256']==expected['region_sha256']==sha(archive/'region.json'),'Region binding differs')
        require(manifest['importance_guide_sha256']==entry['guide_sha256']==expected['guide_sha256']==sha(archive/'importance-guide.json'),'Guide binding differs')
        for name,digest in manifest['archive_sha256'].items():
            require(sha(archive/safe_relative(name))==digest,'Archived input changed: '+name)
        _,rust=verify_bundle(archive/'latent-region-normalizer',archive/'source-bundle.json',archive/'source')
        require(rust==manifest['rust_sources'] and sha(archive/'latent-region-normalizer')==manifest['binary_sha256']==BINARY_SHA,'Reviewed binary/source identity differs')
        require(sha(archive/'source-bundle.json')==manifest['source_bundle_sha256']==BUNDLE_SHA,'Bundle identity differs')
        require(sha(archive/'original-config.json')==manifest['original_config_sha256']==CONFIG_SHA,'Original physical config differs')
        require(sha(archive/'shape.json')==manifest['shape_sha256']==SHAPE_SHA,'Hard shape differs')
        original=read(archive/'original-config.json');config=read(archive/'config.json');originals.append(original)
        require(config==config_for(original,archive) and sha(archive/'config.json')==manifest['config_sha256'],'Physical config changed beyond shape path relocation')
        check_physics(original,read(archive/'region.json'),read(archive/'importance-guide.json'),shell)
        require(read(archive/'chart-model.json')==read(archive/'region.json')['gaussian_chart'] and sha(archive/'chart-model.json')==manifest['chart_model_sha256'],'Regional model differs')
        require(manifest['observer_sha256']==sha(archive/ANALYZER_NAME),'Observer identity differs')
        require(manifest['cloud_replicates']==CLOUDS and manifest['lambda_ratio']==LAMBDA_RATIO and manifest['workers']==4,'Campaign allocation differs')
        jobs=jobs_for(folder,index);require(manifest['jobs']==jobs,'Job allocation, command or fresh seed differs')
        require(entry['expected_jobs']==[{k:j[k] for k in ('id','replicate','seed','samples')} for j in jobs],'Protocol job binding differs')
        seeds.extend(j['seed'] for j in jobs)
    require(originals[0]==originals[1] and len(set(seeds))==8,'Physical scaffolds or independent streams differ')
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=freeze(args.out)
    print(__import__('json').dumps(dict(output=str(args.out.resolve()),protocol_sha256=sha(args.out/'protocol.json'),
        populations=result['total_jobs'],total_draws=result['total_unconditional_draws'],physical_launches=0),indent=2))


if __name__=='__main__':main()
