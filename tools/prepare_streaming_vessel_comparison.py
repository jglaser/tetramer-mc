#!/usr/bin/env python3
"""Freeze the new full-vessel measurement; never dispatch physical jobs.

This replaces no historical preparation. Regional controls, matching SMC
reconciliation and a reviewed authenticated dispatcher remain prerequisites.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from fractions import Fraction as F
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import scipy

from analyze_r4_smc_control import Ledger, read, require, sha, write
from analyze_mobile_native_pocket import local_sources
from run_mobile_posterior_pilot import verify_bundle
from prepare_full_vessel_comparison import ROOT, SOURCES, NATIVE, STAGES, ARMS, TOTAL, WALL, THREADS
from physical_hard_free_line_vessel import PhysicalHardFreeLineGuide
from partition_vessel_streaming import SUPPORTS, validate_supports, bind_native
from certify_vessel_region_containment import certify, read_exact

SCHEMA = 'hard-free-streaming-vessel-preparation-v1'
SEED_BASE = 610031001
LAMBDA_RATIO = 128.
GUIDE = (Path('/vast/xvg/tetramer-mc-runs/hard-free-line-population-size-20261001/conditioned/r00/provenance/importance-guide.json'),
         '085e5e9d348698802becc11c1a4b9a5ff5cbc9894e8942e73df8fb15ba48a439')
BINARY = ROOT/'results/normalizer-wall-journal-parity-20261002/common/basin-normalizer'
BINARY_SHA = '5ebc7599e2bfe457da0aa46ccf99ab0fc789eb21b604b5fcc31226036bf979d6'
BUNDLE = BINARY.parent/'source-bundle.json'
BUNDLE_SHA = '0d012425681497200a3d60967c5fd9a652e6e4e5fc0b79ba5d2d9b8053128859'
REFERENCES = {
    'integrated_references': ('results/hard-free-vessel-validation-20261001/validation.json',
        'fccb767afa120882d00f712080d6cd64024053f00fff24d3901d74dcdbf8ab9d'),
    'journal_parity': ('results/normalizer-wall-journal-parity-20261002/validation.json',
        'cbd2d7d5c415d85000b843cbb7ded3b54a2741c856d0cd57df66b0d4e7a97d89')}


def jobs_for(out, python=None):
    out = Path(out).resolve(); common, inputs = out/'common',out/'inputs'; jobs = []
    python = python or sys.executable
    for stage,n in STAGES:
        for i in range(4):
            for arm in ARMS:
                base = out/stage/arm; destination = base/'runs'/f'r{i:02}'
                audit = base/'audits'/f'r{i:02}'; partition = base/'partitions'/f'r{i:02}'
                seed = SEED_BASE+len(jobs)
                command = [str(common/'basin-normalizer'),'--config',str(inputs/'config.json'),'--model',str(inputs/'model.json'),
                    '--out',str(destination),'--samples',str(n),'--seed',str(seed),'--cloud-replicates','2',
                    '--covariance-scale','1','--uniform-probability','0.1','--wall-radius',str(WALL),'--wall-center','0','0','0']
                if arm == 'half_mixture':
                    command += ['--latent-region',str(inputs/'current_R4.json'),'--latent-guide',str(inputs/'guide.json')]
                audit_script = 'audit_vessel_baseline_streaming.py' if arm == 'vessel' else 'audit_hard_free_vessel_streaming.py'
                audit_command = [python,'-B',str(common/audit_script),'--directory',str(destination),
                    '--out',str(audit),'--binary',str(common/'basin-normalizer'),'--batch-size','64']
                if arm == 'vessel': audit_command += ['--region',str(inputs/'current_R4.json')]
                partition_command = [python,'-B',str(common/'partition_vessel_streaming.py'),'--audit',str(audit/'analysis.json'),
                    '--native-definition',str(inputs/'native-region/definition.json'),'--out',str(partition)]
                for name in SUPPORTS: partition_command += ['--'+name.replace('_','-'),str(inputs/(name+'.json'))]
                jobs.append(dict(id=f'{stage}-{arm}-r{i:02}',stage=stage,arm=arm,population=i,samples=n,seed=seed,
                    directory=str(destination),command=command,log=str(base/'logs'/f'r{i:02}.log'),
                    audit_directory=str(audit),audit_command=audit_command,
                    partition_directory=str(partition),partition_command=partition_command))
    require(len(jobs) == 16 and sum(j['samples'] for j in jobs) == TOTAL
            and len({j['seed'] for j in jobs}) == 16, 'Fixed allocation changed')
    return jobs


def runtime():
    return dict(python=sys.executable,version=sys.version,executable_sha256=sha(sys.executable),
                numpy=np.__version__,scipy=scipy.__version__,optimized=sys.flags.optimize)


def validate_contract(plan,out):
    require(plan['schema'] == SCHEMA and plan['preparation_only'] is True and plan['physical_jobs_launched'] == 0
            and plan['dispatch_ready'] is False,'Not an inert preparation')
    require(plan['jobs'] == jobs_for(out) and plan['total_unconditional_draws'] == TOTAL,'Fixed jobs changed')
    require(plan['stages'] == [dict(name=name,draws_per_population=n,independent_populations_per_arm=4) for name,n in STAGES]
            and plan['arms'] == list(ARMS),'Stage allocation changed')
    require(plan['maximum_physical_workers'] == 8 and plan['maximum_audit_workers'] == 4
            and plan['maximum_all_workers'] == 32 and plan['thread_environment'] == THREADS,'Worker limits changed')
    require(plan['physical'] == dict(depletant_radius=1.5,activity=.035,lambda_ratio=LAMBDA_RATIO,cloud_replicates=2,
            wall_center=[0.,0.,0.],wall_radius=WALL,capture_radius=273,bath_wall_permeable=True,
            measure='Lebesgue center volume times normalized SO(3) Haar measure'),'Physical target/cloud law changed')
    require(plan['input_sha256']['guide.json'] == GUIDE[1]
            and plan['input_sha256']['current_R4.json'] == SOURCES['current_R4.json'][1]
            and plan['input_sha256']['shape.json'] == SOURCES['shape.json'][1]
            and plan['native_definition_sha256'] == NATIVE[1], 'Frozen shape/guide/region/observer changed')


def freeze(out):
    out = Path(out).resolve(); require(not out.exists(),'Fresh inert preparation required')
    require(sys.flags.optimize == 0,'Unoptimized Python required')
    ledger = Ledger(); ledger.bind(BINARY,BINARY_SHA); ledger.bind(BUNDLE,BUNDLE_SHA)
    bundle, hashes = verify_bundle(BINARY,BUNDLE,ROOT)
    references = {name:read(ledger.bind(ROOT/path,digest)) for name,(path,digest) in REFERENCES.items()}
    exact = references['integrated_references']; parity = references['journal_parity']
    require(exact['complete'] is True and exact['independently_audited_new_law_attempts'] == 16640
            and exact['analytic_references']['complete'] is True, 'Incomplete integrated references')
    require(all(j[k]['passed'] is True for j in exact['analytic_references']['jobs'] for k in ('hard','depletion','haar_moment')),
            'Analytic reference failed')
    require(parity['complete'] is True and parity['exact_sample_parity'] is True and parity['new_attempts'] == 384
            and {j['id'] for j in parity['jobs']} == {'baseline','gaussian','hard-free'}
            and all(j['passed'] is True and j['returncode'] == 0 for j in parity['jobs']), 'Journal compatibility incomplete')
    parity_root = (ROOT/REFERENCES['journal_parity'][0]).parent
    parity_plan = read(ledger.bind(parity_root/'plan.json',parity['plan_sha256']))
    require(parity_plan['binary_sha256'] == BINARY_SHA and parity_plan['source_bundle_sha256'] == BUNDLE_SHA,
            'Journal control used another executable')
    for job in parity['jobs']:
        for filename,key in [('samples.jsonl','samples_sha256'),('attempts.jsonl','attempts_sha256'),('summary.json','summary_sha256')]:
            ledger.bind(parity_root/job['id']/filename,job[key])
    sources = {k:(ROOT/v[0],v[1]) for k,v in SOURCES.items() if k != 'guide.json'}
    sources['guide.json'] = GUIDE
    for path,digest in sources.values(): ledger.bind(path,digest)
    definition = ledger.bind(ROOT/NATIVE[0],NATIVE[1]); native = read(definition)
    native_inputs = {name:ledger.bind(definition.parent/'inputs'/name,digest) for name,digest in native['input_sha256'].items()}
    entries = [Path(__file__),*(ROOT/'tools'/name for name in ('audit_vessel_baseline_streaming.py',
        'audit_hard_free_vessel_streaming.py','partition_vessel_streaming.py','compare_streaming_vessel_statistics.py'))]
    closure = {}
    for entry in entries: closure.update(local_sources(entry))
    for path in closure.values(): ledger.bind(path)
    out.mkdir(parents=True); common,inputs = out/'common',out/'inputs'; common.mkdir(); inputs.mkdir()
    for name,(path,_) in sources.items(): shutil.copy2(path,inputs/name)
    (inputs/'native-region/inputs').mkdir(parents=True)
    shutil.copy2(definition,inputs/'native-region/definition.json')
    for name,path in native_inputs.items():
        target = inputs/'native-region/inputs'/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    config = read(inputs/'source-config.json'); config.update(shape=str(inputs/'shape.json'),poisson_lambda_ratio=LAMBDA_RATIO)
    require(config['capture_radius'] == 273 and config['capture_center'] == [0.,0.,0.]
            and config['depletant_radius'] == 1.5 and config['reservoir_density'] == .035
            and len(config['fixed_poses']) == 2 and config.get('target_region') is None,'Changed physical target')
    write(inputs/'config.json',config)
    shape_hash = SOURCES['shape.json'][1]
    guide = read(inputs/'guide.json')
    require(guide['defensive_uniform_shell_probability'] == .5 and guide['conditional_probability'] == 1.
            and guide['raw_translation_axes'] == [0,1,2] and len(guide['gaussian_components']) == 92,'Wrong frozen guide')
    PhysicalHardFreeLineGuide.from_files(inputs/'current_R4.json',inputs/'guide.json',
        vessel_config=config,shape=read(inputs/'shape.json'),expected_shape_sha256=shape_hash)
    validate_supports({k:read(inputs/(k+'.json')) for k in SUPPORTS},config,dict(shape_sha256=shape_hash))
    bind_native(ledger,inputs/'native-region/definition.json',config,dict(shape_sha256=shape_hash))
    certificate = certify(read_exact(inputs/'current_R4.json'),read_exact(inputs/'shape.json'),[0,0,0],F(str(WALL)),[0,0,0],F(273))
    require(certificate['wall_containment_proven'] and certificate['capture_containment_proven'],'R4 domain inclusion unresolved')
    write(inputs/'r4-vessel-containment.json',certificate)
    for name,path in closure.items(): shutil.copy2(path,common/name)
    shutil.copy2(BINARY,common/'basin-normalizer'); shutil.copy2(BUNDLE,common/'source-bundle.json')
    for name,entry in bundle['files'].items():
        path = common/'source'/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(entry['text'].encode())
    ledger.recheck()
    plan = dict(schema=SCHEMA,created=time.time(),preparation_only=True,physical_jobs_launched=0,dispatch_ready=False,
        stages=[dict(name=name,draws_per_population=n,independent_populations_per_arm=4) for name,n in STAGES],
        arms=list(ARMS),jobs=jobs_for(out),total_unconditional_draws=TOTAL,runtime=runtime(),thread_environment=THREADS,
        maximum_physical_workers=8,maximum_audit_workers=4,maximum_all_workers=32,
        binary_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,rust_sources=hashes,
        sources={name:sha(common/name) for name in closure},
        input_sha256={str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file()},
        source_bindings=ledger.files,native_definition_sha256=NATIVE[1],
        physical=dict(depletant_radius=1.5,activity=.035,lambda_ratio=LAMBDA_RATIO,cloud_replicates=2,
            wall_center=[0.,0.,0.],wall_radius=WALL,capture_radius=273,bath_wall_permeable=True,
            measure='Lebesgue center volume times normalized SO(3) Haar measure'),
        proposal=dict(baseline='Unchanged full vessel atlas with 0.1 cube/Haar component, all anchors, covariance scale 1',
            guided='0.5 p_vessel + 0.5 q_hard_free/J; frozen 92 components, alpha=.5, beta=1, xyz axes',
            source_capture_is_target_restriction=False),
        differences_from_old_unlaunched_preparation=dict(guide_components=[80,92],
            lambda_ratio=[64,LAMBDA_RATIO],logging='Same journal guarantees for both arms',
            analysis='Streaming complete geometry audit, once-only full native partition, all regional strata'),
        prerequisites=dict(regional='Complete fixed pilot, population-size and probability/intensity assessments; no waived failures',
            independent='Matching-target SMC discrepancies must be resolved or explicitly investigated before dispatch',
            implementation='Authenticated stage loader and reviewed bounded dispatcher still required',
            stage_order='Complete standard physics/audits/partitions before large; no adaptation or pooling',
            failure='No retry; preserve every attempt and drain started children',assembly='Not decided by these conditional scaffold data'),
        scope='Inert preparation only. No physical process is executed. Completed references are reused; '
              'the historical 80-component preparation and running simulations remain unchanged.')
    write(out/'plan.json',plan)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    return validate(out)


def validate(out):
    out = Path(out).resolve(); ledger = Ledger(); ledger.frozen(out); plan = read(ledger.bind(out/'plan.json'))
    validate_contract(plan,out)
    require(plan['runtime'] == runtime() and plan['thread_environment'] == THREADS,'Frozen runtime changed')
    require(plan['binary_sha256'] == BINARY_SHA and plan['source_bundle_sha256'] == BUNDLE_SHA,'Wrong checked executable')
    verify_bundle(out/'common/basin-normalizer',out/'common/source-bundle.json',out/'common/source')
    for prefix,key in [('common','sources'),('inputs','input_sha256')]:
        for name,digest in plan[key].items(): ledger.bind(out/prefix/name,digest)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=('freeze','validate'))
    parser.add_argument('--out',type=Path,required=True); args = parser.parse_args()
    result = freeze(args.out) if args.action == 'freeze' else validate(args.out)
    print(dict(preparation_only=result['preparation_only'],jobs=len(result['jobs']),draws=result['total_unconditional_draws'],dispatch_ready=False))
