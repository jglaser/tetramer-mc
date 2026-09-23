#!/usr/bin/env python3
"""Freeze the matched full-vessel allocation, without dispatching physics.

Preparation is allowed while regional confirmation is running. A separate
evidence predicate is provided for a future authenticated dispatcher;
preparation and validation never launch a population or authorize dispatch.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import scipy
from analyze_r4_smc_control import Ledger, read, require, sha, write
from analyze_mobile_native_pocket import local_sources, load_classifier
from run_mobile_posterior_pilot import verify_bundle
from physical_latent_guide import PhysicalLatentGuide
from vessel_contact_partition import RegionSupport, SUPPORTS

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='full-vessel-matched-comparison-preparation-v1'
STAGES=(('standard',65536),('large',262144))
ARMS=('vessel','half_mixture')
TOTAL=2621440
WALL=223.32617672378387
SEED_BASE=136101010
SOURCES={
 'source-config.json':('runs/mobile-wall-contact-campaign-20260921/coverage/provenance/config.json','fe7a0c7d806e477aee38d4a085f974908e04e2fbf0d39ca0f9ce5bace38cb3e5'),
 'model.json':('runs/mobile-wall-contact-campaign-20260921/coverage/provenance/model.json','feb4011c622c3104bbe909a29685bd7f077e28f0c847f630c69d8fa87939c20e'),
 'shape.json':('runs/mobile-wall-contact-campaign-20260921/coverage/provenance/shape.json','c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'),
 'current_R4.json':('runs/refined-contact-bank-preparation-20260922/region.json','924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02'),
 'guide.json':('runs/refined-contact-bank-preparation-20260922/guide-bank.json','d0e62d1b23d627582ca5f393af515a1ca10b8ecc01ce1a0e75c40e3e18885929'),
 'old_native_R4.json':('runs/mobile-wall-contact-campaign-20260921/provenance/regions/region-native-r4.json','6dfb093d4f1e534c76e98725369490d40f440ca920c562defaa4117fd82998f9'),
 'old_alternative_R5.json':('runs/mobile-wall-contact-campaign-20260921/provenance/regions/region-alternative-r5.json','76ea65088e302d6b6478ac033af9b67b7cf21cb7cea1f854f70cdcd6db0473ae'),
 'old_alternative_R32.json':('runs/mobile-wall-contact-campaign-20260921/provenance/regions/region-alternative-r32.json','e5bf1e11cd3d455ffd29243214944236e7e5900d1cf35eb03f515e1e5d733796'),
}
NATIVE=('runs/mobile-wall-contact-campaign-20260921/provenance/native-region/definition.json','5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9')
REFERENCE=('runs/full-vessel-reference-dispatch-20260922/validation.json','7b08ba0d358924f485c0077b525309ccf88afe947673bef54a7713fa4163907b')
REFERENCE_INDEX=('runs/full-vessel-latent-reference-validation-20260922/fixture-index.json','10528a9678cf24f4335fb113e2cb64c1ca33c5e4b4cd87c446537c08a0829628')
THREADS={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS')}


def jobs_for(out):
    out=Path(out).resolve(); jobs=[]; common=out/'common'; inputs=out/'inputs'
    for stage,n in STAGES:
        for index in range(4):
            for arm in ARMS:
                job=dict(id=f'{stage}-{arm}-r{index:02d}',stage=stage,arm=arm,population=index,samples=n,
                    seed=SEED_BASE+1009*len(jobs))
                base=out/stage/arm; destination=base/'runs'/f'r{index:02d}'
                command=[str(common/'basin-normalizer'),'--config',str(inputs/'config.json'),'--model',str(inputs/'model.json'),
                    '--out',str(destination),'--samples',str(n),'--seed',str(job['seed']),'--cloud-replicates','2',
                    '--covariance-scale','1','--uniform-probability','0.1','--wall-radius',str(WALL),'--wall-center','0','0','0']
                if arm=='half_mixture':command+=['--latent-region',str(inputs/'current_R4.json'),'--latent-guide',str(inputs/'guide.json')]
                audit=base/'audits'/f'r{index:02d}'; partition=base/'partitions'/f'r{index:02d}'
                audit_command=[sys.executable,str(common/('audit_full_vessel_baseline.py' if arm=='vessel' else 'audit_full_vessel_latent.py')),
                    '--population',str(destination),'--out',str(audit),'--native-definition',str(inputs/'native-region/definition.json')]
                if arm=='vessel':audit_command+=['--region',str(inputs/'current_R4.json'),'--guide',str(inputs/'guide.json')]
                partition_command=[sys.executable,str(common/'vessel_contact_partition.py'),'--audit',str(audit/'analysis.json'),'--out',str(partition)]
                for name in SUPPORTS:partition_command+=['--'+name.replace('_','-'),str(inputs/(name+'.json'))]
                job.update(directory=str(destination),log=str(base/'logs'/f'r{index:02d}.log'),command=command,
                    audit_directory=str(audit),audit_command=audit_command,partition_directory=str(partition),partition_command=partition_command)
                jobs.append(job)
    require(len(jobs)==16 and sum(j['samples'] for j in jobs)==TOTAL and len({j['seed'] for j in jobs})==16,'Allocation changed')
    return jobs


def diagnostic_state(item):
    """An imprecise/missing independent estimate is not a contradiction."""
    if item.get('unresolved'):return 'unresolved'
    difference=item.get('log_mass_difference_SMC_minus_importance'); se=item.get('combined_population_SE')
    require(difference is not None and se is not None and np.isfinite([difference,se]).all() and se>=0,'Malformed independent diagnostic')
    absolute=abs(difference)<=.2; statistical=abs(difference)<=3*se+1e-12
    return 'corroborated' if absolute and statistical else 'material_contradiction' if not absolute and not statistical else 'unresolved'


def evidence_gate(confirmation, comparisons):
    """Numerical predicate only; authenticate_gate_inputs must bind identities."""
    require(confirmation.get('schema')=='contact-confirmation-comparison-v1' and confirmation.get('complete') is True,'Completed regional analysis required')
    convergence=confirmation['convergence']; checks=convergence['checks']
    expected={'main_regional_quality','main_paired_free_energy_precision','every_control_regional_agreement',
        'every_control_direct_free_energy_agreement','significant_original_strata_agreement','classifier_contact_consistency','native_partition_sum'}
    require(set(checks)==expected and all(type(v) is bool for v in checks.values()),'Regional gate contract changed')
    require(convergence['confirmation_passed'] is True and all(checks.values()),'Regional convergence checks have not passed')
    require(len(comparisons)==2,'Both independent SMC proposal controls must be audited and compared')
    states=[]
    for index,comparison in enumerate(comparisons):
        require(comparison.get('schema')=='r4-smc-importance-comparison-v1' and comparison.get('complete') is True,'Incomplete matching SMC comparison')
        primary=comparison['primary_regions']
        require(set(primary)=={'total','registered_native_entry','old_R5_intersection_native','remaining_R4_native'},'Independent primary regions changed')
        require(set(comparison['arms'])==set(confirmation['arms']),'Independent comparison dropped an importance arm')
        for arm,value in comparison['arms'].items():
            for kind in ('Qz','Q0'):
                for region in primary:states.append(dict(control=index,arm=arm,region=region,kind=kind,state=diagnostic_state(value['masses'][kind][region])))
            for family,entries in value['strata'].items():
                for bin_index,entry in enumerate(entries):
                    for region,item in entry.items():
                        if item['decision_relevant']:
                            states.append(dict(control=index,arm=arm,region=region,kind='Qz',stratum=[family,bin_index],state=diagnostic_state(item)))
    require(not any(s['state']=='material_contradiction' for s in states),'Material matching-target SMC disagreement requires explanation before dispatch')
    return dict(regional_checks_passed=True,independent_states=states,
        independent_status='unresolved' if any(s['state']=='unresolved' for s in states) else 'corroborated',
        scope='Numerical prerequisite for the next full-vessel measurement, subject to authenticated provenance. No descendant ESS, rare-terminal-hit or equal-occupancy gate. Missing full-vessel coverage is measured next.')


def authenticate_gate_inputs(confirmation_path, comparison_paths, expected_protocol_hashes):
    """Read-only authenticated gate; expected protocols come from the SMC freeze.

    No population is launched and no authorization file is emitted here.
    """
    require(set(expected_protocol_hashes)=={'broad','narrow'} and len(set(expected_protocol_hashes.values()))==2,
        'Distinct pre-frozen broad and narrow protocols required')
    require(len(comparison_paths)==2 and len({str(Path(p).resolve()) for p in comparison_paths})==2,'Two distinct comparison artifacts required')
    ledger=Ledger(); confirmation_path=Path(confirmation_path).resolve(); ledger.frozen(confirmation_path.parent)
    confirmation=read(ledger.bind(confirmation_path)); confirmation_hash=sha(confirmation_path)
    comparisons=[]; protocols=[]; seeds=set(); smc_paths=set()
    for path in comparison_paths:
        path=Path(path).resolve(); ledger.frozen(path.parent); comparison=read(ledger.bind(path))
        require(Path(comparison['inputs']['importance']).resolve()==confirmation_path
            and comparison['input_sha256'][str(confirmation_path)]==confirmation_hash,'Comparison targets a different confirmation analysis')
        smc_path=Path(comparison['inputs']['smc']).resolve(); require(smc_path not in smc_paths,'Same SMC control supplied twice'); smc_paths.add(smc_path)
        ledger.frozen(smc_path.parent); smc=read(ledger.bind(smc_path,comparison['input_sha256'][str(smc_path)]))
        require(smc['schema']=='smc-r4-control-analysis-v1' and smc['complete'] is True,'Incomplete independent SMC audit')
        protocols.append(smc['protocol_sha256'])
        current={p['seed'] for p in smc['populations']}
        require(len(smc['populations'])==len(current)==4 and seeds.isdisjoint(current),'SMC population streams overlap'); seeds.update(current)
        for name,digest in comparison['input_sha256'].items():ledger.bind(name,digest)
        comparisons.append(comparison)
    require(set(protocols)==set(expected_protocol_hashes.values()) and len(set(protocols))==2,'Independent control protocol identities differ')
    result=evidence_gate(confirmation,comparisons); ledger.recheck()
    return dict(**result,input_sha256=ledger.files,smc_protocol_sha256=protocols,smc_population_seeds=sorted(seeds),dispatch_performed=False)


def freeze(out,binary,bundle):
    out=Path(out).resolve(); require(not out.exists(),'Fresh inert preparation directory required'); ledger=Ledger()
    source,hashes=verify_bundle(binary,bundle,ROOT)
    for _,(path,digest) in SOURCES.items():ledger.bind(ROOT/path,digest)
    classifier,binding=load_classifier(ledger.bind(ROOT/NATIVE[0],NATIVE[1]))
    for name,digest in classifier.definition['input_sha256'].items():ledger.bind((ROOT/NATIVE[0]).parent/'inputs'/name,digest)
    reference=read(ledger.bind(ROOT/REFERENCE[0],REFERENCE[1])); require(reference['complete'] and reference['attempts']==768 and len(reference['audits'])==6,'Missing completed integrated references')
    # Bind the checked physical implementation, including exact embedded source
    # closure; this is not a claim that the release executable was itself tested.
    fixture_index=read(ledger.bind(ROOT/REFERENCE_INDEX[0],REFERENCE_INDEX[1]))
    record=fixture_index['fixtures'][0]; fixture=Path(record['output'])
    fixture_manifest=read(ledger.bind(fixture/'manifest.json',record['manifest_sha256']))
    checked_bundle=read(ledger.bind(fixture/'provenance/source-bundle.json',fixture_manifest['source_bundle_sha256']))
    require(source==checked_bundle,'Release source closure differs from the already validated integration')
    ledger.bind(fixture/'provenance/source-bundle.json')
    entries=[Path(__file__),ROOT/'tools/audit_full_vessel_baseline.py',ROOT/'tools/audit_full_vessel_latent.py',ROOT/'tools/vessel_contact_partition.py']
    closure={}
    for entry in entries:closure.update(local_sources(entry))
    for path in closure.values():ledger.bind(path)
    out.mkdir(parents=True); common=out/'common'; inputs=out/'inputs'; common.mkdir(); inputs.mkdir()
    for name,(path,_) in SOURCES.items():shutil.copy2(ROOT/path,inputs/name)
    shutil.copytree((ROOT/NATIVE[0]).parent,inputs/'native-region')
    config=read(inputs/'source-config.json'); config['shape']=str(inputs/'shape.json'); write(inputs/'config.json',config)
    require(config['capture_radius']==273 and config['capture_center']==[0.,0.,0.] and config['depletant_radius']==1.5
        and config['reservoir_density']==.035 and config['poisson_lambda_ratio']==64 and len(config['fixed_poses'])==2
        and config.get('target_region') is None,'Fixed physical law differs')
    guide=PhysicalLatentGuide.from_files(inputs/'current_R4.json',inputs/'guide.json',expected_shape_sha256=SOURCES['shape.json'][1])
    require(guide.guide.alpha==.5 and guide.guide.count==80 and guide.physical_fixed_neighbors==config['fixed_poses'],'Frozen regional guide differs')
    for name in SUPPORTS:
        r=read(inputs/(name+'.json')); RegionSupport(r)
        require(r['physical_fixed_neighbors']==config['fixed_poses'] and r['physical_metric']==config['metadata'],'Witness physical identity differs')
    for name,path in closure.items():shutil.copy2(path,common/name)
    shutil.copy2(binary,common/'basin-normalizer'); shutil.copy2(bundle,common/'source-bundle.json')
    for name,entry in source['files'].items():
        path=common/'source'/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(entry['text'].encode())
    jobs=jobs_for(out)
    for j in jobs:
        for name in ('directory','log','audit_directory','partition_directory'):Path(j[name]).parent.mkdir(parents=True,exist_ok=True)
    ledger.recheck()
    plan=dict(schema=SCHEMA,created=time.time(),preparation_only=True,physical_jobs_launched=0,repository=str(ROOT),
        python=sys.executable,python_version=sys.version,python_sha256=sha(sys.executable),numerical_runtime=dict(numpy=np.__version__,scipy=scipy.__version__),
        stages=[dict(name=s,draws_per_population=n,independent_populations_per_arm=4) for s,n in STAGES],arms=list(ARMS),jobs=jobs,
        total_unconditional_draws=TOTAL,maximum_physical_workers=8,analysis_workers=4,maximum_all_workers=32,thread_environment=THREADS,
        binary_sha256=sha(common/'basin-normalizer'),source_bundle_sha256=sha(common/'source-bundle.json'),rust_sources=hashes,
        sources={name:sha(common/name) for name in closure},input_sha256={str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file()},
        source_bindings=ledger.files,native_definition_sha256=NATIVE[1],
        proposal=dict(baseline='existing coverage law:178 base/328 virtual, all2 anchors, uniform.1, covariance stdscale1',
            mixed='0.5 p_vessel + 0.5 q_80bank/J; bank internal uniform.5, untruncated Gaussians'),
        physical=dict(depletant_radius=1.5,activity=.035,lambda_ratio=64,cloud_replicates=2,wall_center=[0.,0.,0.],wall_radius=WALL,
            capture_radius=273,measure='d3t in Angstrom cubed times normalized proper SO(3) Haar',bath_wall_permeable=True),
        gates=dict(regional='confirmation_passed AND all recorded checks',independent='completed broad+narrow matching-target comparison, no material contradiction; unresolved precision explicitly retained',
            stage_order='large starts only after all standard jobs and audits succeed; both allocations frozen now, no adaptation or pooling',
            failure='stop new launches and drain every started child; no retries, overwrite or discarded attempts',
            assembly='not authorized by this preparation or pair evidence'),
        validation=dict(integrated_reference_sha256=REFERENCE[1],source_identical_to_checked_integration=True,
            release_execution_checked=False,scope='Exact checked source closure embedded in new release build; release-mode integrated baseline/reference check remains a dispatch prerequisite.'),
        scope='Inert fixed allocation. A dispatcher and stage aggregator must be reviewed before execution; no protein process is started by this program.')
    write(out/'plan.json',plan); write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})); validate(out)
    return plan


def validate(out):
    out=Path(out).resolve(); ledger=Ledger(); ledger.frozen(out); p=read(out/'plan.json')
    require(p['schema']==SCHEMA and p['preparation_only'] is True and p['physical_jobs_launched']==0,'Not an inert preparation')
    require(p['jobs']==jobs_for(out) and p['total_unconditional_draws']==TOTAL,'Frozen allocation differs')
    require(p['thread_environment']==THREADS and p['maximum_physical_workers']==8 and p['maximum_all_workers']==32,'Worker limits differ')
    require(p['python_version']==sys.version and p['python_sha256']==sha(sys.executable)
        and p['numerical_runtime']==dict(numpy=np.__version__,scipy=scipy.__version__) and sys.flags.optimize==0,'Frozen runtime differs')
    verify_bundle(out/'common/basin-normalizer',out/'common/source-bundle.json',out/'common/source')
    for name,digest in p['sources'].items():ledger.bind(out/'common'/name,digest)
    require(sha(out/'common/basin-normalizer')==p['binary_sha256'],'Executable artifact differs')
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('freeze','validate')); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--binary',type=Path); parser.add_argument('--bundle',type=Path); args=parser.parse_args()
    if args.action=='freeze':
        require(args.binary is not None and args.bundle is not None,'Explicit binary and source bundle required'); p=freeze(args.out,args.binary,args.bundle)
    else:p=validate(args.out)
    print(json.dumps(dict(preparation_only=True,jobs=len(p['jobs']),total_draws=p['total_unconditional_draws'])),flush=True)


if __name__=='__main__':main()
