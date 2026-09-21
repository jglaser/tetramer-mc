#!/usr/bin/env python3
"""Freeze and run the prespecified geometry-only four-mobile-tetramer control.

The proposal has no supplied inter-tetramer registry. The initial ABC scaffold
and the retained-D start remain native-informed physical preparations.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_mobile_four_body_growth as positive
from mobile_native_pocket_campaign import execute_batches
from prepare_shoulder_docking_benchmark import local_dependencies
from run_mobile_posterior_pilot import (COORDINATE_ARCHIVE, exclusive_write, read,
    require, safe_relative, sha, validate_residue_coordinates, verify_bundle, write)
from analyze_mobile_posterior_pilot import GEOMETRY_FOUR_BODY_SCHEMA, validate_campaign_jobs

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'runs/mobile-geometry-proposal-preparation-20260921'
REVIEW=ROOT/'runs/mobile-four-body-growth-campaign-20260921/original'
SCHEMA=GEOMETRY_FOUR_BODY_SCHEMA
CONTROLLER_SCHEMA='geometry-mobile-four-body-growth-controller-v1'
SEED_BASE=129101010
BASE_SHA='1a39c8cc0d2977016d5dcd2072328de99612495eab76f6aac8a4aca5086e1519'
GENERATION_SHA='0e18a3511b8e2c2e6ab9c79001e70d7d885263315353c2b5dd32ba5e1ee16161'
GENERATOR_SHA='b2e08464bb4ec5e2688906e46c775d38e7c1a117b186676fd923e2ed372b010d'
SCOPE=('Seeded four-mobile-tetramer growth with a geometry-only reciprocal proposal. The same native ABC scaffold and '
    'retained native D start are supplied physical initial conditions. Two fresh streams per start, 2000 sweeps, unchanged '
    'bath, wall, local/capture/posterior/GCA/shift laws. No native inter-tetramer proposal charts, adaptation, fitting, '
    'label-based proposal selection, physical-kinetics or equilibrium claim. Native contacts are downstream observations.')


def package_check(package):
    package=Path(package).resolve();frozen=read(package/'freeze.json')['files']
    require(frozen,'Empty proposal package')
    for name,digest in frozen.items():require(sha(package/safe_relative(name))==digest,'Proposal package changed: '+name)
    m=read(package/'manifest.json')
    require(m['schema']=='geometry-reciprocal-proposal-preparation-v1','Wrong proposal preparation schema')
    require(m['native_informed_proposal']is False and m['native_geometry_used_in_preparation']is False,'Native-informed proposal preparation')
    require(m['model']=='model.json'and m['base_model']=='provenance/source-model.json'
        and m['generation_provenance']=='provenance/generation.json','Unexpected proposal source paths')
    require(m['base_model_sha256']==sha(package/m['base_model'])==BASE_SHA,'Geometry source model differs')
    require(m['generation_provenance_sha256']==sha(package/m['generation_provenance'])==GENERATION_SHA,'Geometry construction differs')
    require(m['generator_source']=='provenance/generator.rs'and
        m['generator_source_sha256']==sha(package/m['generator_source'])==GENERATOR_SHA,'Geometry generator source differs')
    require(m['model_sha256']==sha(package/'model.json')and m['shape_sha256']==positive.SHAPE_SHA,'Model or shape identity differs')
    model=read(package/'model.json');base=read(package/m['base_model'])
    require(set(model)=={'schema','base_model','reciprocal_components'}and model['schema']=='reciprocal-pose-mixture-v1'
        and model['base_model']==base,'Reciprocal wrapper changes geometry atlas')
    flags=model['reciprocal_components']
    require(len(flags)==96 and all(x is True for x in flags)and m['reciprocal_components']==flags
        and all(x is True for x in m['reciprocal_components'])
        and m['base_components']==96 and m['virtual_components']==192,'Wrong reciprocal branch allocation')
    require(len(base['weights'])==96 and base['shape_sha256']==positive.SHAPE_SHA,'Wrong base atlas')
    return m


def configured(example,starts,folder,start,replicate,index,model_sha):
    # Reuse the established physical preparation; replace only descriptive
    # proposal provenance and the fresh physical stream seed.
    cfg=positive.configured(example,starts,folder,'original',start,replicate,index)
    cfg['seed']=SEED_BASE+1009*index
    cfg['metadata'].update(atlas_variant='geometry',model_sha256=model_sha,
        native_informed_proposal=False,native_initial_scaffold=True,
        native_geometry_used_in_proposal_preparation=False,scope=SCOPE)
    return cfg


def freeze(out,package=PACKAGE):
    out=Path(out).resolve();package=Path(package).resolve();require(not out.exists(),'Fresh campaign required')
    preparation=package_check(package);model_sha=preparation['model_sha256']
    require(sha(positive.DESIGN/'recommended-poses.json')==positive.DESIGN_POSES_SHA
        and sha(positive.EXAMPLE)==positive.EXAMPLE_SHA,'Initial preparation changed')
    for name,digest in read(positive.DESIGN/'freeze.json').items():
        require(sha(positive.DESIGN/safe_relative(name))==digest,'Initial geometry design changed')
    source=REVIEW/'provenance';old=read(REVIEW/'manifest.json')
    require(sha(source/'tetramer-mc')==positive.BINARY_SHA and sha(source/'source-bundle.json')==positive.BUNDLE_SHA,'Reviewed physical binary changed')
    bundle,rust_sources=verify_bundle(source/'tetramer-mc',source/'source-bundle.json',source/'source')
    names=[n for n in old['input_sha256']if n.startswith('reference/')]+['tetramer-shape.json','monomer-shape.json','native-pair-motifs.json']
    require(len(names)==11,'Incomplete observer/shape reference')
    inputs={n:source/n for n in names}
    for name,path in inputs.items():require(sha(path)==old['input_sha256'][name],'Physical/reference source changed: '+name)
    residue=validate_residue_coordinates(inputs['monomer-shape.json'],inputs[COORDINATE_ARCHIVE])
    analyzer=Path(__file__).with_name('analyze_mobile_posterior_pilot.py')
    sources=local_dependencies([Path(__file__),analyzer,Path(__file__).with_name('test_geometry_four_body_growth.py')])
    top=out/'provenance';top.mkdir(parents=True)
    for name,path in sources.items():shutil.copy2(path,top/name)
    shutil.copytree(package,top/'proposal-preparation')
    shutil.copytree(positive.DESIGN,top/'geometry-design')
    shutil.copy2(positive.EXAMPLE,top/'source-example.json')
    shutil.copy2(REVIEW/'manifest.json',top/'positive-reference-manifest.json')
    starts=read(positive.DESIGN/'recommended-poses.json');example=read(positive.EXAMPLE)
    folder=out/'geometry';archive=folder/'provenance';archive.mkdir(parents=True)
    for name,path in dict(inputs,**sources,**{'tetramer-mc':source/'tetramer-mc','source-bundle.json':source/'source-bundle.json','model.json':package/'model.json'}).items():
        target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for name,entry in bundle['files'].items():
        target=archive/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(entry['text'])
    for name in ('configs','runs','logs'):(folder/name).mkdir()
    jobs=[]
    for start in positive.STARTS:
        for replicate in (0,1):
            index=len(jobs);ident=f'{start}-r{replicate:02d}-c09';path=folder/'configs'/f'{ident}.json'
            cfg=configured(example,starts,folder,start,replicate,index,model_sha);write(path,cfg)
            job=dict(id=ident,index=index,start=start,replicate=replicate,mode='c09',seed=cfg['seed'],
                config=str(path),config_sha256=sha(path),directory=str(folder/'runs'/ident),log=str(folder/'logs'/f'{ident}.log'))
            job['command']=positive.command(folder,job);jobs.append(job)
    manifest=dict(schema=SCHEMA,atlas_variant='geometry',native_informed_proposal=False,native_initial_scaffold=True,
        frozen=True,binary_supplied=True,binary_sha256=positive.BINARY_SHA,source_bundle_sha256=positive.BUNDLE_SHA,rust_sources=rust_sources,
        sweeps=2000,burn_sweeps=400,sample_every=1,workers=4,bodies=4,body_count=4,tracked_body_index=3,scaffold_body_indices=[0,1,2],
        jobs=jobs,seeds=[j['seed']for j in jobs],total_attempts=4*6*2000,total_single_body_attempts=4*4*2000,total_collective_attempts=4*2*2000,
        model=str(archive/'model.json'),model_sha256=model_sha,shape_sha256=positive.SHAPE_SHA,observer_sha256=sha(analyzer),
        reference=str(archive/'reference'),residue_coordinate_validation=residue,
        proposal_preparation_manifest_sha256=sha(package/'manifest.json'),
        input_sha256={p.relative_to(archive).as_posix():sha(p)for p in archive.rglob('*')if p.is_file()},scope=SCOPE)
    write(folder/'manifest.json',manifest)
    protocol=dict(schema=CONTROLLER_SCHEMA,campaigns=[dict(arm='geometry',path=str(folder),manifest_sha256=sha(folder/'manifest.json'))],
        controller_sha256=sha(__file__),physical_executable_sha256=positive.BINARY_SHA,source_bundle_sha256=positive.BUNDLE_SHA,
        observer_sha256=sha(analyzer),model_sha256=model_sha,proposal_preparation_manifest_sha256=sha(package/'manifest.json'),
        proposal_preparation_freeze_sha256=sha(package/'freeze.json'),base_model_sha256=BASE_SHA,generation_provenance_sha256=GENERATION_SHA,
        generator_source_sha256=GENERATOR_SHA,
        total_jobs=4,maximum_physical_workers=4,audit_workers=4,sweeps=2000,burn_sweeps=400,seed_base=SEED_BASE,
        total_attempts=4*6*2000,body_count=4,tracked_body_index=3,scaffold_body_indices=[0,1,2],
        native_informed_proposal=False,native_initial_scaffold=True,
        first_endpoint='New D attachment and native-connected four; separate catalogue-cycle consistency remains downstream.',
        stopping='Exactly four fresh streams, 2000 sweeps. Drain all started children; no retry, overwrite, extension, or audit after physical failure.',scope=SCOPE)
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(files={p.relative_to(out).as_posix():sha(p)for p in out.rglob('*')if p.is_file()}))
    validate(out);return protocol


def validate(out):
    out=Path(out).resolve();p=read(out/'protocol.json')
    require(p['schema']==CONTROLLER_SCHEMA and p['controller_sha256']==sha(out/'provenance'/Path(__file__).name),'Controller identity differs')
    for name,digest in read(out/'freeze.json')['files'].items():require(sha(out/safe_relative(name))==digest,'Frozen input changed: '+name)
    require(p['total_jobs']==p['maximum_physical_workers']==p['audit_workers']==4,'Worker/job allocation changed')
    require(p['native_informed_proposal']is False and p['native_initial_scaffold']is True,'Incorrect proposal/initialization provenance')
    package=out/'provenance/proposal-preparation';prep=package_check(package)
    require(sha(package/'manifest.json')==p['proposal_preparation_manifest_sha256']and sha(package/'freeze.json')==p['proposal_preparation_freeze_sha256'],'Preparation binding differs')
    require(p['model_sha256']==prep['model_sha256'],'Model binding differs')
    folder=out/'geometry';archive=folder/'provenance'
    require(p['campaigns']==[dict(arm='geometry',path=str(folder),manifest_sha256=sha(folder/'manifest.json'))],'Campaign allocation differs')
    m=read(folder/'manifest.json');require(m['schema']==SCHEMA and m['body_count']==m['bodies']==4,'Observer domain differs')
    require((m['sweeps'],m['burn_sweeps'],m['sample_every'],m['workers'])==(2000,400,1,4),'Run allocation differs')
    require(m['model_sha256']==sha(archive/'model.json')==p['model_sha256'],'Proposal bytes differ')
    require(m['binary_sha256']==sha(archive/'tetramer-mc')==positive.BINARY_SHA and m['source_bundle_sha256']==sha(archive/'source-bundle.json')==positive.BUNDLE_SHA,'Physical executable differs')
    _,rust=verify_bundle(archive/'tetramer-mc',archive/'source-bundle.json',archive/'source');require(m['rust_sources']==rust,'Source closure changed')
    for name,digest in m['input_sha256'].items():require(sha(archive/safe_relative(name))==digest,'Archived dependency differs')
    require(m['observer_sha256']==p['observer_sha256']==sha(archive/'analyze_mobile_posterior_pilot.py'),'Observer identity differs')
    starts=read(out/'provenance/geometry-design/recommended-poses.json');example=read(out/'provenance/source-example.json')
    require(sha(out/'provenance/geometry-design/recommended-poses.json')==positive.DESIGN_POSES_SHA and sha(out/'provenance/source-example.json')==positive.EXAMPLE_SHA,'Initial state source changed')
    require(len(m['jobs'])==4 and m['seeds']==[SEED_BASE+1009*i for i in range(4)],'Fresh independent stream allocation differs')
    for i,(j,(start,replicate))in enumerate(zip(m['jobs'],[(s,r)for s in positive.STARTS for r in (0,1)])):
        ident=f'{start}-r{replicate:02d}-c09'
        require((j['id'],j['start'],j['replicate'],j['mode'],j['index'],j['seed'])==(ident,start,replicate,'c09',i,SEED_BASE+1009*i),'Job allocation differs')
        require(j['config']==str(folder/'configs'/f'{ident}.json')and j['directory']==str(folder/'runs'/ident)and j['log']==str(folder/'logs'/f'{ident}.log'),'Paths differ')
        require(read(j['config'])==configured(example,starts,folder,start,replicate,i,p['model_sha256'])and sha(j['config'])==j['config_sha256'],'Physical configuration differs')
        require(j['command']==positive.command(folder,j),'Physical command differs')
    validate_campaign_jobs(m,dict(complete=True,running=False,jobs=[dict(id=j['id'],status='complete',exit_code=0)for j in m['jobs']]))
    return p


def check(out):
    out=Path(out).resolve();p=validate(out);require(sha(__file__)==p['controller_sha256'],'Use exact archived controller')
    require(not(out/'status.json').exists()and not(out/'geometry-audit.log').exists(),'No retry/overwrite')
    folder=out/'geometry'
    require(not(folder/'status.json').exists()and not(folder/'assessment').exists()
        and not any((folder/'runs').iterdir())and not any((folder/'logs').iterdir()),'Existing physical/audit output')
    return p


def run(out):
    out=Path(out).resolve();p=check(out);folder=out/'geometry';manifest=read(folder/'manifest.json')
    jobs=[dict(j,arm='geometry',status='pending')for j in manifest['jobs']]
    state=dict(schema='geometry-mobile-four-body-growth-status-v1',complete=False,phase='physical',protocol_sha256=sha(out/'protocol.json'),jobs=jobs,audits={},started=time.time())
    exclusive_write(out/'status.json',state)
    def snapshot():
        write(out/'status.json',state)
        selected=[dict(j,exit_code=j.get('returncode'))for j in jobs]
        write(folder/'status.json',dict(running=any(j['status']=='running'for j in selected),
            complete=all(j['status']=='complete'and j['exit_code']==0 for j in selected),jobs=selected))
    snapshot()
    try:
        execute_batches(jobs,snapshot);state['phase']='physical_validation';snapshot();validate(out)
        for j in jobs:j['output']=positive.verify_output(manifest,j)
        state['phase']='audit';snapshot()
        argv=[sys.executable,'-B',str(folder/'provenance/analyze_mobile_posterior_pilot.py'),'--campaign',str(folder),'--out',str(folder/'assessment'),'--workers','4']
        a=dict(argv=argv,started=time.time(),returncode=None);state['audits']['geometry']=a;snapshot()
        with(out/'geometry-audit.log').open('xb')as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        a.update(returncode=result.returncode,finished=time.time());snapshot();result.check_returncode()
        assessment=read(folder/'assessment/analysis.json')
        require(assessment['complete']and len(assessment['runs'])==4 and all(r['passed']for r in assessment['runs']),'Incomplete observer')
        require(assessment['manifest_sha256']==sha(folder/'manifest.json')and assessment['terminal_status_sha256']==sha(folder/'status.json')
            and assessment['analyzer_sha256']==manifest['observer_sha256'],'Observer identity/terminal binding differs')
        a['analysis_sha256']=sha(folder/'assessment/analysis.json')
        state.update(complete=True,phase='complete',finished=time.time());snapshot();return state
    except BaseException as error:
        state.update(complete=False,phase=state['phase']+'_failed',exception=repr(error),finished=time.time());snapshot();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','validate','run'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--package',type=Path,default=PACKAGE);args=parser.parse_args()
    result=freeze(args.out,args.package)if args.action=='freeze'else check(args.out)if args.action=='validate'else run(args.out)
    print(json.dumps(dict(action=args.action,out=str(args.out.resolve()),complete=result.get('complete'))))
