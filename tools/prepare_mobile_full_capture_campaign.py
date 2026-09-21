#!/usr/bin/env python3
"""Freeze a full D170 reciprocal importance campaign without launching draws."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
from prepare_shoulder_docking_benchmark import local_dependencies
from prepare_smc_normalizer_atlas import unwrap_proposal_model
from run_mobile_posterior_pilot import read,require,safe_relative,sha,verify_bundle,write

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'runs/mobile-full-capture-source-review-20260921'
REVIEW_SHA='31224e59f54632981e7d471c4eca154c0ef67f17e31e4942d04b5ac218915786'
BINARY_SHA='4f81061fdc9e32cf3a141e725dd00403ea536d5268085dab1422176b7037f5dc'
BUNDLE_SHA='8fea2ffb7826ce582507c82f23b8dbe6c7e8fa202e3b930ccac5e4204ddc9daa'
REFERENCE=ROOT/'runs/mobile-competing-reference-preparation-20260921'
CONFIG_SHA='2ed28d5b618209d51dd8a0381c95a8e58853a7e9780f66d53fb027ec07318cf1'
SHAPE_SHA='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
MODEL=ROOT/'examples/frozen-reciprocal-mixture.json'
MODEL_SHA='dc9218c9706e1691af9336ef1c0e87a75a86994933dfd41cc6e1d1cd3dc52aa3'
NATIVE_SOURCE=ROOT/'runs/mobile-reciprocal-atlas-benchmark-20260921/reciprocal'
NATIVE_MANIFEST_SHA='29fb93ee1c61611d734d963de87f363ae69c0aebd0a1bf4472d34db5c7141cb3'
NATIVE_DEFINITION=ROOT/'runs/mobile-native-region-definition-20260921/definition.json'
NATIVE_DEFINITION_SHA='5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'
PARTITION_FILES={
    'region-native-r4.json':'6dfb093d4f1e534c76e98725369490d40f440ca920c562defaa4117fd82998f9',
    'model-native-original.json':'86aedd9218381a47a4efef756bab82e58f03aee78f31d58873f10ab9cdc2b667',
    'model-competitor-geometric.json':'fe517e333b11ee0ceedb0f80f0241e06c7d15e461eb0a64e78959fb10e9b9460',
    'region-competitor-r3.json':'0c9116fdd8f26ac8610b15130961d8dfe162ad9676706edf951e4d09ff28f846',
}
ARMS={'epsilon-0p1':.1,'epsilon-0p5':.5}
SAMPLES,REPLICATES,WORKERS,CLOUDS,LAMBDA_RATIO=8192,4,8,2,64.
SEED_BASE=122101010
ANALYZER_NAME='analyze_basin_normalizers.py'
RUNNER_NAME='run_mobile_full_capture_campaign.py'
SCHEMA='mobile-full-capture-controller-v1'
SCOPE=('Independent fixed-N integration of the entire declared D170 center-capture sphere, every proper orientation, '
       'and hard validity against the exact two observed fixed tetramers, at rd=1.5A and z=.035A^-3. '
       'Both frozen proposal anchors are marginalized and the full reciprocal atlas+uniform-cube/Haar density enters the denominator. '
       'The two epsilon values change only sampling, not the physical target. Every invalid draw remains a zero. '
       'This is conditional on the fixed scaffold and within technical D170 capture, not the complete original-wall ensemble, '
       'an association cost for forming the scaffold, or an assembly simulation. No adaptation, optional stopping, old-row reuse or valid-only normalization. '
       'Observed ESS and population agreement cannot certify unobserved high-weight poses.')


def config_for(original,archive):
    value=copy.deepcopy(original);value['shape']=str(archive/'shape.json');return value


def validate_physics(config,model):
    require(config['capture_center']==[0.,0.,0.] and config['capture_radius']==170.,'Require unchanged D170 capture')
    require(config['depletant_radius']==1.5 and config['reservoir_density']==.035 and config['poisson_lambda_ratio']==64.,'Require original bath and lambda64')
    require(len(config['fixed_poses'])==2 and config.get('target_region') is None,'Require two fixed neighbors with no extra target mask')
    require(config['metadata']['fixed_body_ids']==[2,1] and config['metadata']['moving_body_id']==0,'Physical body identities differ')
    base,flags=unwrap_proposal_model(model)
    require(flags==[True]*150 and base['shape_sha256']==SHAPE_SHA,'Require unchanged whole150 reciprocal proposal')
    require(base['coordinate_convention']=='anchor-body-relative','Require proper body-relative charts')


def command_for(folder,job,epsilon):
    archive=folder/'provenance'
    return [str(archive/'basin-normalizer'),'--config',str(archive/'config.json'),'--model',str(archive/'model.json'),
        '--out',job['directory'],'--samples',str(SAMPLES),'--seed',str(job['seed']),
        '--covariance-scale','1','--cloud-replicates',str(CLOUDS),'--uniform-probability',str(epsilon)]


def jobs_for(folder,index,epsilon):
    jobs=[]
    for replicate in range(REPLICATES):
        name=f'r{replicate:02d}'
        j=dict(id=name,replicate=replicate,index=index*REPLICATES+replicate,seed=SEED_BASE+1009*(index*REPLICATES+replicate),
            samples=SAMPLES,cloud_replicates=CLOUDS,covariance_std_scale=1.,proposal_anchor_index=None,
            directory=str(folder/'runs'/name),log=str(folder/'logs'/f'{name}.log'))
        j['command']=command_for(folder,j,epsilon);jobs.append(j)
    return jobs


def partition_metadata(config):
    return dict(schema='mobile-full-capture-partition-v1',physical_target='hard_valid within capture D170; d3t times normalized SO(3) Haar',
        q_metric=config['metadata'],chart_anchor=config['fixed_poses'][0],physical_fixed_body_ids=[2,1],moving_body_id=0,
        chart_files=dict(native='model-native-original.json',competitor='model-competitor-geometric.json'),
        chart_measure='u=L^-1([t-anchor_t, ell*Cayley(R*anchor_R^T)]-mean); rho=Euclidean norm(u), after pose transformed into the frozen chart anchor frame.',
        primary_partition=[dict(name='original_native',criterion='q <= 1'),
            dict(name='original_other_competitor_ball12',criterion='q > 1 and finite competitor rho <= 12'),
            dict(name='original_other_competitor_remainder',criterion='q > 1 and (competitor rho > 12 or exact competitor Cayley seam)')],
        native_subpartition=[dict(name='original_native_r4',criterion='q <= 1 and finite original-native-chart rho <= 4'),
            dict(name='original_native_rest',criterion='q <= 1 and outside the finite original-native-chart radius4 ball')],
        boundary_rules='q=1 belongs to native; rho_comp=12 belongs to competitor ball; rho_native=4 belongs to nativeR4. Exact Cayley seam is outside any finite chart ball and included in its remainder. Non-seam numerical inverse failures are audit errors, never silently censored.',
        depletion_split='Every region is additionally split by the exact recorded depletion_contact Boolean (exclusion overlap with the fixed-neighbor union).',
        native_instantaneous_entry=dict(body_symmetry_order=1,maximum_body_member_center_error_A=2.,maximum_body_angle_degrees=15.,
            minimum_prescribed_external_bonds=1,maximum_monomer_center_error_A=3.,maximum_monomer_angle_degrees=20.,
            maximum_atomic_gap_A=2.,minimum_shared_native_residue_pairs=1,observed_residue_pair_gap_A=2.,
            reference_native_patch_gap_A=1.,hysteresis=False,
            prescription='At least one prescribed (member_i,member_j,directed_class) external bond must pass all monomer-entry conditions. This is separate from original single-reference q.'),
        row_accounting='All masks retain original unconditional population denominator, including zeros. No proposal-component labels define a physical region.',
        scope='Primary three-way partition and nested nativeR4/rest are exhaustive only on the hard-valid D170 target. Finite chart regions are descriptive subsets; no missing-tail bound is implied.')


def freeze(out):
    out=Path(out).resolve();require(not out.exists(),'Use a fresh campaign directory')
    require(sha(REVIEW/'validation.json')==REVIEW_SHA,'Reviewed source record changed')
    review=read(REVIEW/'validation.json');binary,bundle_path=REVIEW/'basin-normalizer',REVIEW/'source-bundle.json'
    require(sha(binary)==BINARY_SHA and sha(bundle_path)==BUNDLE_SHA,'Reviewed executable changed')
    bundle,rust=verify_bundle(binary,bundle_path,REVIEW/'source');require(rust==review['rust_sources'],'Reviewed Rust map differs')
    for name,digest in review['python_sha256'].items():require(sha(REVIEW/'python'/name)==digest,'Reviewed Python audit changed')
    original_path=REFERENCE/'config.json';require(sha(original_path)==CONFIG_SHA,'Observed scaffold config changed')
    original=read(original_path);shape=Path(original['shape'])
    require(shape.is_absolute() and sha(shape)==SHAPE_SHA and sha(MODEL)==MODEL_SHA,'Original shape or reciprocal model changed')
    validate_physics(original,read(MODEL))
    for name,digest in PARTITION_FILES.items():require(sha(REFERENCE/name)==digest,'Original regional chart changed: '+name)
    native=read(REFERENCE/'region-native-r4.json');competitor=read(REFERENCE/'region-competitor-r3.json')
    for region in (native,competitor):
        require(region['physical_fixed_neighbors']==original['fixed_poses'] and region['fixed_neighbor']==original['fixed_poses'][0]
            and region['physical_metric']==original['metadata'],'Partition chart frame/metric differs')
    require(sha(NATIVE_SOURCE/'manifest.json')==NATIVE_MANIFEST_SHA,'Native catalogue source binding changed')
    native_manifest=read(NATIVE_SOURCE/'manifest.json');native_archive=NATIVE_SOURCE/'provenance'
    reference_hashes={k:v for k,v in native_manifest['input_sha256'].items() if k.startswith('reference/')}
    require(len(reference_hashes)==8,'Incomplete native observer reference closure')
    observer_inputs={'native-reference/'+k[len('reference/'):]:native_archive/k for k in reference_hashes}
    for name in ('monomer-shape.json','native-pair-motifs.json','fixed-snapshot.json'):
        observer_inputs[name]=native_archive/name
    for name,path in observer_inputs.items():
        original_name=name.replace('native-reference/','reference/')
        require(sha(path)==native_manifest['input_sha256'][original_name],'Native observer source changed: '+name)
    require(sha(NATIVE_DEFINITION)==NATIVE_DEFINITION_SHA,'Instantaneous native definition changed')
    definition=read(NATIVE_DEFINITION)
    require(definition['physical_config_sha256']==CONFIG_SHA and definition['shape_sha256']==SHAPE_SHA
        and definition['fixed_poses']==original['fixed_poses'],'Instantaneous native definition target differs')
    definition_inputs={name:NATIVE_DEFINITION.parent/'inputs'/safe_relative(name) for name in definition['input_sha256']}
    for name,path in definition_inputs.items():
        require(sha(path)==definition['input_sha256'][name],'Native definition input changed: '+name)
    dependencies=local_dependencies([Path(__file__),Path(__file__).with_name(RUNNER_NAME),Path(__file__).with_name(ANALYZER_NAME)])
    dependencies.update({name:REVIEW/'python'/name for name in review['python_sha256']})
    top=out/'provenance';top.mkdir(parents=True)
    for name,path in dependencies.items():shutil.copy2(path,top/name)
    native_dir=top/'native-region';native_dir.mkdir()
    shutil.copy2(NATIVE_DEFINITION,native_dir/'definition.json')
    for name,path in definition_inputs.items():
        target=native_dir/'inputs'/safe_relative(name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);require(sha(target)==definition['input_sha256'][name],'Native definition changed while copying')
    (top/'partition').mkdir()
    for name in PARTITION_FILES:shutil.copy2(REFERENCE/name,top/'partition'/name)
    metadata=partition_metadata(original);write(top/'partition/partition-metadata.json',metadata)
    campaigns=[]
    for index,(arm,epsilon) in enumerate(ARMS.items()):
        folder=out/arm;archive=folder/'provenance';archive.mkdir(parents=True)
        sources=dict(dependencies);sources.update(observer_inputs)
        sources.update({'basin-normalizer':binary,'source-bundle.json':bundle_path,'reviewed-source.json':REVIEW/'validation.json',
            'input-config.json':original_path,'shape.json':shape,'model.json':MODEL,
            'physical-preflight.json':REFERENCE/'preflight.json','physical-preparation-plan.json':REFERENCE/'plan.json'})
        sources.update({'partition/'+name:REFERENCE/name for name in PARTITION_FILES})
        sources['partition/partition-metadata.json']=top/'partition/partition-metadata.json'
        for name,path in sources.items():
            target=archive/safe_relative(name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            require(sha(target)==sha(path),'Input changed while freezing: '+name)
        for name,record in bundle['files'].items():
            target=archive/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(record['text'])
            require(sha(target)==rust[name],'Embedded Rust copy differs')
        write(archive/'config.json',config_for(original,archive))
        for name in ('runs','logs'):(folder/name).mkdir()
        jobs=jobs_for(folder,index,epsilon)
        hashes={p.relative_to(archive).as_posix():sha(p) for p in sorted(archive.rglob('*')) if p.is_file()}
        manifest=dict(schema=1,experiment_schema='mobile-full-capture-arm-v1',arm=arm,frozen=True,production_launched=False,
            source_inputs={name:dict(path=str(path),sha256=sha(path)) for name,path in sources.items()},archive_sha256=hashes,
            jobs=jobs,workers=4,model_components=150,virtual_model_components=300,proposal_anchor_index=None,
            covariance_std_scale=1.,uniform_probability=epsilon,rust_sources=rust,observer_sha256=hashes[ANALYZER_NAME],
            binary_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,model_sha256=MODEL_SHA,shape_sha256=SHAPE_SHA,
            config_sha256=hashes['config.json'],original_config_sha256=CONFIG_SHA,
            partition_metadata_sha256=hashes['partition/partition-metadata.json'],
            physical=dict(depletant_radius=1.5,activity=.035,capture_center=original['capture_center'],capture_radius=170.,
                fixed_poses=original['fixed_poses'],fixed_neighbor_count=2,lambda_ratio=LAMBDA_RATIO,uniform_probability=epsilon),
            design='Frozen exact reciprocal model; covariance multiplier1; both anchors marginalized; epsilon changes via CLI only, config otherwise unchanged except shape archive path.',
            inference=SCOPE)
        write(folder/'manifest.json',manifest)
        campaigns.append(dict(arm=arm,path=str(folder),manifest_sha256=sha(folder/'manifest.json'),epsilon=epsilon,
            expected_jobs=[{k:j[k] for k in ('id','replicate','seed','samples','covariance_std_scale','proposal_anchor_index')} for j in jobs]))
    protocol=dict(schema=SCHEMA,production_launched=False,campaigns=campaigns,maximum_physical_workers=WORKERS,
        samples_per_population=SAMPLES,populations_per_arm=REPLICATES,total_unconditional_draws=65536,total_jobs=8,
        cloud_replicates=CLOUDS,lambda_ratio=LAMBDA_RATIO,covariance_std_scale=1.,proposal_anchor_index=None,
        physical_executable_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,model_sha256=MODEL_SHA,
        original_config_sha256=CONFIG_SHA,shape_sha256=SHAPE_SHA,reviewed_source_sha256=REVIEW_SHA,
        controller_sha256=sha(Path(__file__).with_name(RUNNER_NAME)),freezer_sha256=sha(__file__),
        partition=dict(metadata='provenance/partition/partition-metadata.json',metadata_sha256=sha(top/'partition/partition-metadata.json'),
            chart_files_sha256=PARTITION_FILES,native_observer_inputs_sha256={name:sha(path) for name,path in observer_inputs.items()},
            native_definition='provenance/native-region/definition.json',native_definition_sha256=NATIVE_DEFINITION_SHA,
            native_definition_inputs_sha256=definition['input_sha256']),
        seed_policy='Fresh independent streams 122101010+1009*j for j0..7, epsilon-arm-major then replicate-major.',
        stopping='Fixed8192 draws in eight populations. Drain all started children after any failure. No retries. Audit each arm once only after all physical jobs and input bindings succeed.',
        scope=SCOPE)
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    validate(out);return protocol


def validate(out):
    out=Path(out).resolve();protocol=read(out/'protocol.json')
    require(protocol['schema']==SCHEMA,'Unknown full-capture protocol')
    require((protocol['maximum_physical_workers'],protocol['samples_per_population'],protocol['populations_per_arm'],
        protocol['total_unconditional_draws'],protocol['cloud_replicates'],protocol['lambda_ratio'],protocol['total_jobs'],
        protocol['covariance_std_scale'],protocol['proposal_anchor_index'])==(8,8192,4,65536,2,64.,8,1.,None),'Fixed allocation or anchor law changed')
    require(protocol['physical_executable_sha256']==BINARY_SHA and protocol['source_bundle_sha256']==BUNDLE_SHA
        and protocol['model_sha256']==MODEL_SHA and protocol['original_config_sha256']==CONFIG_SHA,'Executable/model/config bindings differ')
    for name,digest in read(out/'freeze.json')['files'].items():require(sha(out/safe_relative(name))==digest,'Frozen input changed: '+name)
    native_definition=out/safe_relative(protocol['partition']['native_definition'])
    require(sha(native_definition)==protocol['partition']['native_definition_sha256']==NATIVE_DEFINITION_SHA,'Native definition binding differs')
    require(read(native_definition)['input_sha256']==protocol['partition']['native_definition_inputs_sha256'],'Native input map differs')
    for name,digest in protocol['partition']['native_definition_inputs_sha256'].items():
        require(sha(native_definition.parent/'inputs'/safe_relative(name))==digest,'Frozen native definition input changed: '+name)
    require(len(protocol['campaigns'])==2 and [e['arm'] for e in protocol['campaigns']]==list(ARMS),'Wrong arm pair/order')
    for index,entry in enumerate(protocol['campaigns']):
        arm=entry['arm'];epsilon=ARMS[arm];folder=out/arm;archive=folder/'provenance'
        require(entry['path']==str(folder) and entry['epsilon']==epsilon and sha(folder/'manifest.json')==entry['manifest_sha256'],'Arm binding differs')
        m=read(folder/'manifest.json');require(m['schema']==1 and m['experiment_schema']=='mobile-full-capture-arm-v1' and m['arm']==arm,'Wrong arm manifest')
        for name,digest in m['archive_sha256'].items():require(sha(archive/safe_relative(name))==digest,'Archived input changed: '+name)
        _,rust=verify_bundle(archive/'basin-normalizer',archive/'source-bundle.json',archive/'source')
        require(rust==m['rust_sources'] and sha(archive/'basin-normalizer')==BINARY_SHA and sha(archive/'source-bundle.json')==BUNDLE_SHA,'Reviewed executable differs')
        require(sha(archive/'model.json')==MODEL_SHA and sha(archive/'shape.json')==SHAPE_SHA and sha(archive/'input-config.json')==CONFIG_SHA,'Target/model bytes changed')
        original=read(archive/'input-config.json');cfg=read(archive/'config.json')
        require(cfg==config_for(original,archive) and sha(archive/'config.json')==m['config_sha256'],'Physical config changed beyond shape path')
        validate_physics(cfg,read(archive/'model.json'))
        require(m['physical']==dict(depletant_radius=1.5,activity=.035,capture_center=cfg['capture_center'],capture_radius=170.,
            fixed_poses=cfg['fixed_poses'],fixed_neighbor_count=2,lambda_ratio=64.,uniform_probability=epsilon),'Arm physical target or proposal epsilon differs')
        jobs=jobs_for(folder,index,epsilon);require(m['jobs']==jobs,'Frozen jobs/commands/fresh seeds differ')
        require(m['proposal_anchor_index'] is None and m['covariance_std_scale']==1. and m['uniform_probability']==epsilon,'Anchor marginalization or scale differs')
        require(entry['expected_jobs']==[{k:j[k] for k in ('id','replicate','seed','samples','covariance_std_scale','proposal_anchor_index')} for j in jobs],'Protocol job bindings differ')
        require(read(archive/'partition/partition-metadata.json')==partition_metadata(original),'Partition definition changed')
        for name,digest in PARTITION_FILES.items():require(sha(archive/'partition'/name)==digest,'Partition chart changed')
        for name,digest in protocol['partition']['native_observer_inputs_sha256'].items():require(sha(archive/name)==digest,'Native catalogue/input binding changed')
        require(m['observer_sha256']==sha(archive/ANALYZER_NAME),'Auditor changed')
    return protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();p=freeze(args.out)
    print(__import__('json').dumps(dict(output=str(args.out.resolve()),protocol_sha256=sha(args.out/'protocol.json'),
        populations=p['total_jobs'],total_draws=p['total_unconditional_draws'],physical_launches=0),indent=2))


if __name__=='__main__':main()
