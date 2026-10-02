#!/usr/bin/env python3
"""Authenticate one completed full-vessel stage, then reduce saved statistics.

No geometry, native classification, sampling or previous-stage replay. This is
an analysis entry point, not a dispatcher or an assembly stability certificate.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import math
from pathlib import Path
import shutil
import time

from analyze_r4_smc_control import Ledger, close, read, require, sha, write
from analyze_mobile_native_pocket import local_sources
from compare_full_vessel_stage import frozen_artifact, recorded, bind_source_closure
from prepare_streaming_vessel_comparison import validate as validate_preparation, STAGES, ARMS
from partition_vessel_streaming import AUDITS, SUPPORTS, PRIMARY
from compare_streaming_vessel_statistics import compare, validate_partition


def validate_manifest(plan,job,manifest):
    physical = plan['physical']; hashes = plan['input_sha256']
    expected = dict(samples=job['samples'],seed=job['seed'],cloud_replicates=2,covariance_scale=1.,
        uniform_probability=.1,proposal_anchor_index=None,executable_sha256=plan['binary_sha256'],
        source_bundle_sha256=plan['source_bundle_sha256'],config_sha256=hashes['config.json'],
        model_sha256=hashes['model.json'],shape_sha256=hashes['shape.json'],activity=physical['activity'],
        atomic_wall=dict(center=physical['wall_center'],radius=physical['wall_radius']),bath_wall_permeable=True,
        physical_fixed_neighbor_count=2,pose_proposal_schema=3,base_component_count=178,virtual_component_count=328,
        proposal_model_kind='reciprocal-pose-mixture-v1',
        attempt_journal='attempts.jsonl; begin before each attempt; no retries',resume_supported=False)
    for name,value in expected.items(): require(name in manifest and manifest[name] == value,'Executed manifest differs: '+name)
    close(manifest['lambda'],physical['activity']*physical['lambda_ratio'],'Cloud intensity differs')
    if job['arm'] == 'vessel':
        require(manifest['schema'] == 4 and 'outer_mixture_schema' not in manifest,'Baseline law changed')
    else:
        require(job['arm'] == 'half_mixture','Unknown proposal arm')
        expected = dict(schema=6,outer_mixture_schema='full-vessel-hard-free-line-half-mixture-v1',outer_vessel_probability=.5,
            latent_region_sha256=hashes['current_R4.json'],latent_guide_sha256=hashes['guide.json'],
            latent_guide_schema='defensive-hard-free-line-guide-v1',latent_gaussian_component_count=92,
            latent_defensive_uniform_probability=.5,latent_reference_ball_is_target_restriction=False,
            density_measure='Lebesgue center volume times normalized SO(3) Haar measure')
        for name,value in expected.items(): require(manifest.get(name) == value,'Executed guided law differs: '+name)
        require(manifest['latent_source_capture'] == dict(center=[0.,0.,0.],radius=170.,
                    restricts_target=False,conditions_guide=True),'Source conditioning/domain changed')


def load_population(preparation,plan,job,ledger):
    common,inputs = preparation/'common',preparation/'inputs'
    root,audit_root,part_root = [Path(job[k]).resolve() for k in ('directory','audit_directory','partition_directory')]
    af = frozen_artifact(ledger,audit_root,('analysis.json','status.json','geometry.jsonl'))
    pf = frozen_artifact(ledger,part_root,('analysis.json','status.json','labels.jsonl'))
    audit_path,part_path = audit_root/'analysis.json',part_root/'analysis.json'
    audit,part = read(ledger.bind(audit_path)),read(ledger.bind(part_path))
    expected_audit = AUDITS[0 if job['arm'] == 'vessel' else 1]
    require(audit['schema'] == expected_audit and audit['complete'] is True and part['complete'] is True,
            'Wrong or incomplete population audit')
    for directory,analysis in ((audit_root,audit_path),(part_root,part_path)):
        status = read(ledger.bind(directory/'status.json'))
        require(status['complete'] is True and status['phase'] == 'complete'
                and status['analysis_sha256'] == sha(analysis),'Analysis status/hash differs')
    require(Path(audit['population']).resolve() == root and Path(part['population']).resolve() == root
            and Path(part['audit']).resolve() == audit_path,'Population/audit/partition lineage differs')
    am,pm = audit['source_sha256'],part['input_sha256']
    recorded(ledger,audit_path,pm,sha(audit_path)); recorded(ledger,audit_root/'geometry.jsonl',pm,audit['geometry_sha256'])
    ledger.bind(audit_root/'geometry.jsonl',audit['geometry_sha256']); ledger.bind(part_root/'labels.jsonl',part['labels_sha256'])
    audit_entry = 'audit_vessel_baseline_streaming.py' if job['arm'] == 'vessel' else 'audit_hard_free_vessel_streaming.py'
    bind_source_closure(ledger,audit_root,am,common,plan,audit_entry,af)
    bind_source_closure(ledger,part_root,pm,common,plan,'partition_vessel_streaming.py',pf)
    manifest = read(recorded(ledger,root/'manifest.json',am))
    recorded(ledger,root/'manifest.json',pm,sha(root/'manifest.json'))
    require(manifest == audit['manifest'] == part['manifest'],'Manifest snapshots differ')
    validate_manifest(plan,job,manifest)
    summary = read(recorded(ledger,root/'summary.json',am))
    recorded(ledger,root/'summary.json',pm,sha(root/'summary.json'))
    require(summary['complete'] is True and summary['manifest'] == manifest and summary['numerical_nulls'] == 0
            and summary['samples'] == job['samples'] and not (root/'failure.json').exists(),'Incomplete physical population')
    for name,key in [('samples.jsonl','samples_sha256'),('attempts.jsonl','attempts_sha256')]:
        require(summary[key] == audit[key],'Physical completion/audit hash differs')
        recorded(ledger,root/name,am,audit[key]); recorded(ledger,root/name,pm,audit[key])
    for name,key in [('input-config.json','config_sha256'),('model.json','model_sha256'),
                     ('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
        recorded(ledger,root/'provenance'/name,am,manifest[key])
    if job['arm'] == 'half_mixture':
        for name,key in [('latent-region.json','latent_region_sha256'),('latent-guide.json','latent_guide_sha256')]:
            recorded(ledger,root/'provenance'/name,am,manifest[key])
    config = read(recorded(ledger,root/'config.json',am)); expected = read(inputs/'config.json')
    recorded(ledger,root/'config.json',pm,sha(root/'config.json'))
    for value in (config,expected):
        for key in ('target_region','proposal_anchor_index'):
            if value.get(key) is None: value.pop(key,None)
    require(config == expected,'Executed physical/proposal configuration changed')
    binary = common/'basin-normalizer'; binding = audit['executable_binding']
    require(Path(binding['path']).resolve() == binary and binding['sha256'] == plan['binary_sha256']
            and binding['artifact_verified'] is True,'Audited executable differs')
    recorded(ledger,binary,am,plan['binary_sha256'])
    definition_path = inputs/'native-region/definition.json'
    definition = read(recorded(ledger,definition_path,pm,plan['native_definition_sha256']))
    binding = part['native_binding']
    require(Path(binding['definition']).resolve() == definition_path
            and binding['definition_sha256'] == part['native_definition_sha256'] == plan['native_definition_sha256']
            and binding['input_sha256'] == definition['input_sha256']
            and binding['runtime_sha256'] == definition['input_sha256']['source/native_contact_regions.py'],
            'Complete native observer binding differs')
    for name,digest in definition['input_sha256'].items(): recorded(ledger,definition_path.parent/'inputs'/name,pm,digest)
    require(set(part['region_paths']) == set(SUPPORTS),'Missing historical pocket definitions')
    for name in SUPPORTS:
        path = inputs/(name+'.json'); require(Path(part['region_paths'][name]).resolve() == path,'Pocket path changed')
        recorded(ledger,path,pm,plan['input_sha256'][name+'.json'])
    if job['arm'] == 'vessel':
        binding = audit['reporting_region_binding']; expected = inputs/'current_R4.json'
        require(Path(binding['path']).resolve() == expected and binding['sha256'] == plan['input_sha256']['current_R4.json']
                and binding['affects_proposal'] is False and binding['restricts_target'] is False,'Baseline reporting changes target/proposal')
        recorded(ledger,expected,am,plan['input_sha256']['current_R4.json'])
        ledger.bind(audit_root/'provenance/reporting-region.json',plan['input_sha256']['current_R4.json'])
    validate_partition(part,job['samples'])
    for kind,label in [('Qz','total'),('Q0','hard_total')]:
        before,after = audit['estimates']['total'][kind],part['estimates']['total'][kind]
        require(before['draws'] == after['draws'] and before['nonzero'] == after['nonzero']
                and (before['logQ'] is None) == (after['logQ'] is None),'Changed audited total support')
        if after['logQ'] is not None:
            close(before['logQ'],after['logQ'],'Changed audited total mass')
            close(before['ess'],after['ess'],'Changed audited total ESS')
            close(before['max_fraction'],after['max_fraction'],'Changed audited largest contribution')
            close(summary['estimates'][label]['log_normalizer'],after['logQ'],'Changed physical summary')
        else: require(summary['estimates'][label]['log_normalizer'] is None,'Unobserved total acquired physical mass')
    cpu = summary['sampler_cpu_seconds']; require(math.isfinite(cpu) and cpu > 0,'Missing positive sampler CPU')
    return dict(id=job['id'],arm=job['arm'],stage=job['stage'],population=job['population'],seed=job['seed'],
        samples=job['samples'],directory=str(root),audit_sha256=sha(audit_path),partition_sha256=sha(part_path),
        raw_sha256=audit['samples_sha256'],sampler_CPU_seconds=cpu,audit_CPU_seconds=audit['analysis_CPU_seconds'],
        classification_CPU_seconds=part['analysis_CPU_seconds'],estimates=part['estimates'],partition_data=part)


def run(preparation,expected_plan_hash,stage,out):
    preparation,out = Path(preparation).resolve(),Path(out).resolve()
    require(not out.exists(),'Fresh stage analysis destination required'); require(stage in dict(STAGES),'Select one stage')
    started = time.process_time(); ledger = Ledger(); ledger.bind(preparation/'plan.json',expected_plan_hash)
    plan = validate_preparation(preparation); ledger.frozen(preparation)
    sources = local_sources(__file__)
    for name,path in sources.items():
        # The reader may be frozen later, but its shared statistical/audit code
        # must remain exactly the code already frozen in the preparation.
        ledger.bind(path,plan['sources'].get(name))
    populations = [load_population(preparation,plan,job,ledger) for job in plan['jobs'] if job['stage'] == stage]
    result = compare(populations,stage)
    costs = {}
    for arm in ARMS:
        selected = [p for p in populations if p['arm'] == arm]
        total = sum(p['sampler_CPU_seconds'] for p in selected)
        costs[arm] = dict(sampler_CPU_seconds=total,audit_CPU_seconds=sum(p['audit_CPU_seconds'] for p in selected),
            classification_CPU_seconds=sum(p['classification_CPU_seconds'] for p in selected),
            importance_ESS_per_sampler_CPU_second={name:result['arms'][arm]['Qz']['estimates'][name]['importance_ESS']/total
                for name in ('total',*PRIMARY)})
    result.update(schema='authenticated-streaming-vessel-stage-v1',input_authentication_performed=True,
        preparation=str(preparation),preparation_sha256=expected_plan_hash,physical_target=plan['physical'],
        reporting_support_sha256={name:plan['input_sha256'][name+'.json'] for name in SUPPORTS},
        native_definition_sha256=plan['native_definition_sha256'],input_sha256=ledger.files,costs=costs,
        efficiency_scope='IID importance-weight ESS per sampler CPU, not autocorrelation ESS or assembly mixing.',
        new_pose_draws=0,new_clouds=0,new_geometry_queries=0,new_native_classifier_calls=0,
        analysis_CPU_seconds=time.process_time()-started)
    ledger.recheck(); out.mkdir(parents=True); (out/'provenance').mkdir()
    for name,path in sources.items():
        shutil.copy2(path,out/'provenance'/name)
        require(sha(out/'provenance'/name) == ledger.files[str(Path(path).resolve())],'Analysis source changed while copying')
    ledger.recheck(); write(out/'analysis.json',result)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True); parser.add_argument('--preparation-sha256',required=True)
    parser.add_argument('--stage',choices=tuple(dict(STAGES)),required=True); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); result = run(args.preparation,args.preparation_sha256,args.stage,args.out)
    print(dict(complete=result['complete'],stage=result['stage'],diagnostics=result['diagnostics']))
