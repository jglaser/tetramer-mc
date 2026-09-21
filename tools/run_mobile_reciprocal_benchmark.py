#!/usr/bin/env python3
"""Freeze or run the eight matched exact-reciprocal controls exactly once.

Freeze/preflight execute no binary. Run drains all started children before
reporting failure and audits only after all eight physical jobs succeed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from prepare_shoulder_docking_benchmark import local_dependencies
from prepare_mobile_reciprocal_benchmark import SOURCE_MODEL_SHA, SNAPSHOT_SHA
from prepare_smc_normalizer_atlas import unwrap_proposal_model
from run_mobile_posterior_pilot import (
    COORDINATE_ARCHIVE, exclusive_write, inside, read, relocated_config,
    require, safe_relative, sha, validate_residue_coordinates, verify_bundle, write,
)

ROOT=Path(__file__).resolve().parents[1]
PREPARATION=ROOT/'runs/mobile-reciprocal-atlas-preparation-20260921'
BINARY=ROOT/'target/reciprocal-review/release/tetramer-mc'
BINARY_SHA='b0ea3ebafc4d410abae3ac120b19e8349b48a2d99dd4f7157ac58557185e8776'
BUNDLE=ROOT/'target/reciprocal-review/release/build/tetramer-mc-356a8e0b913327fe/out/source-bundle.json'
REFERENCE=ROOT/'runs/mobile-posterior-reference-recovery-20260921/reference'
BUNDLE_SHA='b8c9505673d87b30589e776465427d0df4261d88b22045fe2dd8621ebbbf6c49'
ANALYZER=ROOT/'tools/analyze_mobile_posterior_pilot.py'
SCHEMA='mobile-reciprocal-atlas-benchmark-v1'
SWEEPS,BURN,WORKERS=2000,400,8


def command_for(campaign,job):
    return [str(campaign/'provenance/tetramer-mc'),'run','--config',job['config'],
        '--model',str(campaign/'provenance/model.json'),'--method','learned',
        '--out',job['directory'],'--sweeps',str(SWEEPS),'--sample-every','1','--no-gsd']


def configured(source,campaign,replicate):
    result=relocated_config(source,campaign)
    result['seed']=source['seed']+4036*replicate
    result['metadata'].update(start='competing',replicate=replicate)
    return result


def freeze(benchmark,preparation=PREPARATION,binary=BINARY,bundle_path=BUNDLE,*,preparation_sha256):
    benchmark,preparation,binary,bundle_path=[Path(p).resolve() for p in (benchmark,preparation,binary,bundle_path)]
    require(not benchmark.exists(),'Fresh benchmark directory required')
    require(not inside(benchmark,preparation),'Do not modify the atlas preparation')
    require(isinstance(preparation_sha256,str) and len(preparation_sha256)==64,'Supply the reviewed preparation SHA256')
    require(sha(preparation/'plan.json')==preparation_sha256,'Atlas preparation identity changed')
    plan=read(preparation/'plan.json')
    require(plan['schema']=='mobile-reciprocal-atlas-preparation-v1','Wrong preparation schema')
    require(plan['legacy_model_sha256']==SOURCE_MODEL_SHA and plan['source_snapshot_sha256']==SNAPSHOT_SHA,'Wrong atlas/snapshot source')
    require(plan['base_components']==150 and plan['reciprocal_components']==[True]*150,'Wrong reciprocal specification')
    for name,digest in read(preparation/'freeze.json').items():
        require(sha(preparation/safe_relative(name))==digest,'Frozen atlas input changed: '+name)
    require(not plan['production_launched'] and len(plan['controls'])==4,'Require the inert four-control preparation')
    require(sha(binary)==BINARY_SHA,'Wrong reviewed reciprocal executable')
    require(sha(bundle_path)==BUNDLE_SHA,'Wrong reviewed reciprocal source bundle')
    bundle,rust_sources=verify_bundle(binary,bundle_path,ROOT)
    controls={(c['atlas'],c['mode']):c for c in plan['controls']}
    require(set(controls)=={(a,m) for a in ('legacy','reciprocal') for m in ('c0','c09')},'Incomplete atlas/correlation grid')
    marker=read(REFERENCE/'reference-recovery.json')
    references=dict(marker['original_reference_sha256'])
    references[marker['added_coordinates']['path']]=marker['added_coordinates']['sha256']
    require(len(references)==8,'Require all five observer modules and three data files')
    for name,digest in references.items():
        require(sha(REFERENCE/name)==digest,'Source native reference changed: '+name)
    residue_validation=validate_residue_coordinates(preparation/'provenance/monomer-shape.json',
        REFERENCE/'results/native-geometry-repair/rebuilt-hydrogens/heavy-coordinates.json')
    dependencies=local_dependencies([Path(__file__).resolve(),ANALYZER])
    # Preserve the complete existing observable protocol; only allocation/scope change.
    analysis=read(ROOT/'runs/mobile-posterior-pilot-preparation-20260921/analysis-plan.json')
    analysis.update(schema='mobile-reciprocal-atlas-analysis-plan-v1',
        primary_comparison='Legacy versus exact reciprocal symmetrization of the unchanged 150-chart atlas at each correlation; c0 versus c09 within each atlas. Exact same competing initial snapshot and physical target.',
        contextual_control='None: all arms use identical branch fractions and local/GCA/shift moves.',
        initialization='One deliberately selected, non-equilibrated trapped configuration. No comparison across equilibrium initial populations is claimed.',
        primary_endpoint='Per-run loss/rearrangement of the initial unregistered body0-body2 contact and first native body0 attachment; retain scaffold/body identities and attempted-update attribution. Native connectivity of all three bodies is secondary.',
        rules=['All eight predetermined runs, every retained repeat and every attempted update; no optional stopping or best-run selection.',
               'No production fitting, adaptation, repair or fixed body; all three proteins remain mobile.',
               'Both atlases remain native-informed; no new contact centers or fitted parameters are added. Reciprocal branches exactly invert the unchanged Gaussian base poses.',
               'No equilibrium speedup, physical kinetics or global thermodynamic conclusion from this short preparation test.'])
    scope=plan['physical_scope']+' Eight fixed-duration controls of escape/re-registration from one shared competing snapshot; all source-defined atom/wall/bath constraints retained.'
    top=benchmark/'provenance';top.mkdir(parents=True)
    for name,path in dependencies.items():shutil.copy2(path,top/name)
    write(top/'analysis-plan.json',analysis)
    shutil.copy2(preparation/'plan.json',top/'atlas-preparation-plan.json')
    manifests=[]
    for variant in ('legacy','reciprocal'):
        campaign=benchmark/variant;archive=campaign/'provenance';archive.mkdir(parents=True)
        sources=dict(dependencies)
        sources.update({'tetramer-mc':binary,'source-bundle.json':bundle_path,
            'tetramer-shape.json':preparation/'provenance/shape.json',
            'monomer-shape.json':preparation/'provenance/monomer-shape.json',
            'native-pair-motifs.json':preparation/'provenance/native-pair-motifs.json',
            'model.json':preparation/f'model-{variant}.json',
            'fixed-snapshot.json':preparation/'provenance/fixed-snapshot.json',
            'atlas-preparation-plan.json':preparation/'plan.json','analysis-plan.json':top/'analysis-plan.json'})
        sources.update({'reference/'+name:REFERENCE/name for name in references})
        for mode in ('c0','c09'):
            control=controls[variant,mode]
            require(sha(control['config'])==control['config_sha256'],'Source control changed')
            sources[f'sourceconfigs/{mode}.json']=Path(control['config'])
        input_hashes={name:sha(path) for name,path in sources.items()}
        for name,path in sources.items():
            target=archive/safe_relative(name);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            require(sha(target)==input_hashes[name],'Input changed while freezing: '+name)
        for name,record in bundle['files'].items():
            target=archive/'source'/safe_relative(name);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(record['text'])
            require(sha(target)==rust_sources[name],'Embedded Rust copy differs')
            input_hashes['source/'+name]=rust_sources[name]
        for name in ('configs','runs','logs'):(campaign/name).mkdir()
        jobs=[]
        for replicate in (0,1):
            for mode in ('c0','c09'):
                name=f'competing-r{replicate:02}-{mode}'
                config=configured(read(archive/f'sourceconfigs/{mode}.json'),campaign,replicate)
                path=campaign/'configs'/f'{name}.json';write(path,config)
                job=dict(id=name,index=len(jobs),start='competing',mode=mode,replicate=replicate,
                    seed=config['seed'],config=str(path),config_sha256=sha(path),
                    directory=str(campaign/'runs'/name),log=str(campaign/'logs'/f'{name}.log'))
                job['command']=command_for(campaign,job);jobs.append(job)
        manifest=dict(schema=SCHEMA,atlas_variant=variant,frozen=True,launched=False,binary_supplied=True,
            binary_sha256=input_hashes['tetramer-mc'],source_bundle_sha256=input_hashes['source-bundle.json'],
            rust_sources=rust_sources,reviewed_source_root=str(ROOT),preparation_path=str(preparation),
            preparation_plan_sha256=preparation_sha256,sweeps=SWEEPS,burn_sweeps=BURN,sample_every=1,
            workers=4,bodies=3,jobs=jobs,seeds=[j['seed'] for j in jobs],
            total_attempts=4*5*SWEEPS,total_single_body_attempts=4*3*SWEEPS,total_collective_attempts=4*2*SWEEPS,
            model=str(archive/'model.json'),model_sha256=input_hashes['model.json'],
            shape_sha256=input_hashes['tetramer-shape.json'],analysis_plan_sha256=input_hashes['analysis-plan.json'],
            observer_sha256=input_hashes[ANALYZER.name],reference=str(archive/'reference'),
            residue_coordinate_validation=residue_validation,input_sha256=input_hashes,
            source_paths={name:str(path) for name,path in sources.items()},scope=scope)
        write(campaign/'manifest.json',manifest)
        write(campaign/'commands.json',dict(jobs=jobs,actual_kernel_executions=0))
        write(campaign/'freeze.json',dict(files={p.relative_to(campaign).as_posix():sha(p)
            for p in sorted(campaign.rglob('*')) if p.is_file()}))
        manifests.append(dict(atlas=variant,path=str(campaign),manifest_sha256=sha(campaign/'manifest.json')))
    protocol=dict(schema='matched-mobile-reciprocal-atlas-controller-v1',production_launched=False,
        sweeps=SWEEPS,burn_sweeps=BURN,sample_every=1,workers=WORKERS,total_jobs=8,total_attempts=80000,
        campaigns=manifests,preparation_plan_sha256=preparation_sha256,controller_sha256=sha(__file__),
        physical_executable_sha256=BINARY_SHA,scope=scope,
        retained='Every sweep and every move log, including null/rejected/repeated states. Assess all eight runs separately.',
        stopping='Exactly 2000 sweeps per run. Any physical failure drains started children and forbids all audits/retries.',
        observer='Same unchanged per-run native/atomic/move audit as completed pilot; explicit four-job schema. Reference data closure includes authoritative coordinate records.')
    write(benchmark/'protocol.json',protocol)
    write(benchmark/'freeze.json',dict(files={p.relative_to(benchmark).as_posix():sha(p)
        for p in sorted(benchmark.rglob('*')) if p.is_file()}))
    validate(benchmark)
    return protocol


def validate(benchmark):
    benchmark=Path(benchmark).resolve();protocol=read(benchmark/'protocol.json')
    require(protocol['schema']=='matched-mobile-reciprocal-atlas-controller-v1','Unknown benchmark schema')
    require((protocol['sweeps'],protocol['burn_sweeps'],protocol['sample_every'],protocol['workers'],protocol['total_jobs'])
            ==(SWEEPS,BURN,1,WORKERS,8),'Benchmark allocation changed')
    for name,digest in read(benchmark/'freeze.json')['files'].items():
        require(sha(benchmark/safe_relative(name))==digest,'Frozen benchmark changed: '+name)
    require(len(protocol['campaigns'])==2 and {c['atlas'] for c in protocol['campaigns']}=={'legacy','reciprocal'},'Wrong atlas pair')
    seeds=[];physical_starts=[];models={};physical_configs=[]
    for entry in protocol['campaigns']:
        campaign=benchmark/entry['atlas'];archive=campaign/'provenance'
        require(entry['path']==str(campaign) and sha(campaign/'manifest.json')==entry['manifest_sha256'],'Campaign binding changed')
        manifest=read(campaign/'manifest.json')
        require(manifest['schema']==SCHEMA and manifest['atlas_variant']==entry['atlas'],'Campaign schema/variant changed')
        require((manifest['sweeps'],manifest['burn_sweeps'],manifest['sample_every'],manifest['workers'],manifest['bodies'])==(SWEEPS,BURN,1,4,3),'Campaign allocation changed')
        for name,digest in manifest['input_sha256'].items():
            require(sha(archive/safe_relative(name))==digest,'Archived input changed: '+name)
        _,rust=verify_bundle(archive/'tetramer-mc',archive/'source-bundle.json',archive/'source')
        require(rust==manifest['rust_sources'],'Embedded source map changed')
        require(manifest['binary_sha256']==BINARY_SHA==protocol['physical_executable_sha256']==sha(archive/'tetramer-mc'),'Wrong executable')
        require(manifest['source_bundle_sha256']==BUNDLE_SHA==sha(archive/'source-bundle.json'),'Wrong source bundle')
        require(manifest['preparation_plan_sha256']==protocol['preparation_plan_sha256']==sha(archive/'atlas-preparation-plan.json'),'Preparation binding changed')
        require(manifest['model']==str(archive/'model.json') and sha(manifest['model'])==manifest['model_sha256'],'Model binding changed')
        models[entry['atlas']]=read(archive/'model.json')
        if entry['atlas']=='legacy':
            require(manifest['model_sha256']==SOURCE_MODEL_SHA,'Legacy model is not byte-identical')
        require(manifest['reference']==str(archive/'reference'),'Observer reference is not frozen')
        require(not (archive/'reference/reference-recovery.json').exists(),'Production reference must not contain recovery marker')
        require(len([k for k in manifest['input_sha256'] if k.startswith('reference/')])==8,'Incomplete reference closure')
        require(validate_residue_coordinates(archive/'monomer-shape.json',archive/COORDINATE_ARCHIVE)==manifest['residue_coordinate_validation'],'Residue data changed')
        jobs=manifest['jobs'];require(len(jobs)==len({j['id'] for j in jobs})==4,'Need four distinct jobs')
        require({(j['start'],j['mode'],j['replicate']) for j in jobs}=={('competing',m,r) for m in ('c0','c09') for r in (0,1)},'Incomplete control grid')
        for index,job in enumerate(jobs):
            original=read(archive/f"sourceconfigs/{job['mode']}.json")
            expected=configured(original,campaign,job['replicate']);config=read(job['config'])
            require(job['index']==index and job['seed']==expected['seed'],'Seed/index changed')
            require(job['config']==str(campaign/'configs'/f"{job['id']}.json") and
                job['directory']==str(campaign/'runs'/job['id']) and job['log']==str(campaign/'logs'/f"{job['id']}.log"),'Job paths changed')
            require(config==expected and sha(job['config'])==job['config_sha256'],'Frozen physical config changed')
            require(job['command']==command_for(campaign,job),'Frozen command changed')
            require(config['fixed_body_indices']==[] and config['seed_labels']==[],'A physical body was fixed')
            require(config['frozen_posterior']==dict(probability=.5,correlation=0. if job['mode']=='c0' else .9),'Correlation control changed')
            physical={k:v for k,v in config.items() if k not in ('seed','metadata','shape','monomer_shape','frozen_posterior')}
            physical_configs.append(physical)
            physical_starts.append(config['initial_poses']);seeds.append(job['seed'])
        require(manifest['seeds']==[j['seed'] for j in jobs],'Manifest seed list changed')
    require(len(set(seeds))==8 and all(p==physical_starts[0] for p in physical_starts),'Seeds or shared exact initial state differ')
    require(set(seeds)=={119201010+1009*j for j in range(8)},'Use the eight fresh predetermined seeds')
    require(all(c==physical_configs[0] for c in physical_configs),'Physical targets or move fractions differ between controls')
    base,flags=unwrap_proposal_model(models['reciprocal'])
    require(base==models['legacy'] and flags==[True]*150,'Reciprocal model must exactly wrap all unchanged 150 base charts')
    return protocol


def check(benchmark,journal,assessment):
    benchmark,journal,assessment=[Path(p).resolve() for p in (benchmark,journal,assessment)]
    require(all(not inside(a,b) and not inside(b,a) for a,b in ((benchmark,journal),(benchmark,assessment),(journal,assessment))),
        'Benchmark, journal and assessment paths must be separate')
    protocol=validate(benchmark)
    require(sha(__file__)==protocol['controller_sha256'],'Run the frozen controller')
    require(not journal.exists() and not assessment.exists() and not (benchmark/'status.json').exists(),'No restart or overwrite')
    for entry in protocol['campaigns']:
        campaign=Path(entry['path'])
        require(not (campaign/'status.json').exists(),'A child campaign was already reserved')
        require(not any((campaign/'runs').iterdir()) and not any((campaign/'logs').iterdir()),'Existing physical outputs/logs')
    return protocol


def run(benchmark,journal,assessment):
    benchmark,journal,assessment=[Path(p).resolve() for p in (benchmark,journal,assessment)]
    protocol=check(benchmark,journal,assessment)
    journal.mkdir(parents=True);shutil.copy2(__file__,journal/'runner.py')
    state=dict(complete=False,phase='physical',started=time.time(),protocol_sha256=sha(benchmark/'protocol.json'),jobs=[])
    exclusive_write(benchmark/'status.json',state)
    manifests={entry['atlas']:read(Path(entry['path'])/'manifest.json') for entry in protocol['campaigns']}
    statuses={variant:dict(running=True,complete=False,jobs=[dict(id=j['id'],status='pending',exit_code=None) for j in m['jobs']])
        for variant,m in manifests.items()}
    reserved=[]
    try:
        for variant,status in statuses.items():
            exclusive_write(benchmark/variant/'status.json',status);reserved.append(variant)
    except BaseException as error:
        for variant in reserved:
            statuses[variant].update(running=False,complete=False)
            for job in statuses[variant]['jobs']:job.update(status='not_started',reason='Reservation failed; no children launched.')
            write(benchmark/variant/'status.json',statuses[variant])
        state.update(phase='reservation_failed',exception=repr(error),finished=time.time())
        write(benchmark/'status.json',state);write(journal/'status.json',state);raise
    jobs=[(variant,index,job) for variant,m in manifests.items() for index,job in enumerate(m['jobs'])]
    active={};next_index=0;failed=False;exception=None
    def snapshot():
        state['jobs']=[dict(atlas=variant,**record) for variant,status in statuses.items() for record in status['jobs']]
        write(journal/'status.json',state)
        write(benchmark/'status.json',state)
        for variant,status in statuses.items():write(benchmark/variant/'status.json',status)
    def finish(key,child):
        variant,index=key;code=child.wait()
        statuses[variant]['jobs'][index].update(status='complete' if code==0 else 'failed',exit_code=code,finished=time.time())
        print('Finished',variant,statuses[variant]['jobs'][index]['id'],'exit',code,flush=True)
        return code
    try:
        while active or (next_index<len(jobs) and not failed):
            while not failed and next_index<len(jobs) and len(active)<WORKERS:
                variant,index,job=jobs[next_index];next_index+=1
                require(sha(job['command'][0])==manifests[variant]['binary_sha256'],'Executable changed at launch')
                require(sha(job['config'])==job['config_sha256'] and sha(manifests[variant]['model'])==manifests[variant]['model_sha256'],'Job input changed at launch')
                require(not Path(job['directory']).exists(),'Output appeared before launch')
                record=statuses[variant]['jobs'][index];record.update(status='launching',started=time.time(),argv=job['command'],log=job['log']);snapshot()
                try:
                    with Path(job['log']).open('xb') as log:child=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
                except BaseException as error:
                    record.update(status='launch_failed',exception=repr(error),finished=time.time());raise
                active[variant,index]=child;record.update(status='running',pid=child.pid);snapshot()
                print('Started',variant,job['id'],'PID',child.pid,flush=True)
            for key,child in list(active.items()):
                if child.poll() is None:continue
                failed=finish(key,child)!=0 or failed;del active[key];snapshot()
            if active:time.sleep(.5)
    except BaseException as error:
        failed,exception=True,error;state['exception']=repr(error)
    finally:
        for key,child in active.items():failed=finish(key,child)!=0 or failed
        for status in statuses.values():
            for record in status['jobs']:
                if record['status']=='pending':record.update(status='not_started',reason='Stopped after failure; no retry.')
            status.update(running=False,complete=all(j['status']=='complete' and j['exit_code']==0 for j in status['jobs']))
        snapshot()
    if failed or not all(status['complete'] for status in statuses.values()):
        state.update(phase='physical_failed',complete=False,finished=time.time());snapshot()
        if exception is not None:raise exception
        raise RuntimeError('A physical job failed; all started children drained, no retry or audit')
    try:
        validate(benchmark)
        require(not assessment.exists(),'Assessment appeared during physical run')
        assessment.mkdir(parents=True)
        state.update(phase='audit',audits={});snapshot()
        for variant,manifest in manifests.items():
            campaign=benchmark/variant;destination=assessment/variant
            argv=[sys.executable,'-B',str(campaign/'provenance/analyze_mobile_posterior_pilot.py'),
                '--campaign',str(campaign),'--out',str(destination),'--reference',manifest['reference'],'--workers','4']
            state['audits'][variant]=dict(argv=argv,started=time.time());snapshot()
            with (journal/f'{variant}-analysis.log').open('xb') as log:result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            state['audits'][variant].update(exit_code=result.returncode,finished=time.time());snapshot();result.check_returncode()
            assessed=read(destination/'analysis.json')
            require(assessed['complete'] and len(assessed['runs'])==4 and assessed['manifest_sha256']==sha(campaign/'manifest.json')
                and assessed['terminal_status_sha256']==sha(campaign/'status.json') and assessed['analyzer_sha256']==manifest['observer_sha256'],'Observer assessment binding failed')
            rows={r['id']:r for r in assessed['runs']}
            require(set(rows)=={j['id'] for j in manifest['jobs']} and all(rows[j['id']]['passed'] and all(rows[j['id']][k]==j[k]
                for k in ('mode','start','replicate','seed')) for j in manifest['jobs']),'Observer omitted a completed run')
            state['audits'][variant]['analysis_sha256']=sha(destination/'analysis.json');snapshot()
    except BaseException as error:
        state.update(phase='audit_failed',complete=False,exception=repr(error),finished=time.time());snapshot();raise
    state.update(phase='complete',complete=True,finished=time.time());snapshot()
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    frozen=sub.add_parser('freeze');frozen.add_argument('--benchmark',type=Path,required=True)
    frozen.add_argument('--preparation',type=Path,default=PREPARATION)
    frozen.add_argument('--preparation-sha256',required=True)
    frozen.add_argument('--binary',type=Path,default=BINARY)
    frozen.add_argument('--bundle',type=Path,default=BUNDLE)
    for action in ('preflight','run'):
        command=sub.add_parser(action)
        for field in ('benchmark','journal','assessment'):command.add_argument('--'+field,type=Path,required=True)
    args=parser.parse_args()
    if args.action=='freeze':result=freeze(args.benchmark,args.preparation,args.binary,args.bundle,preparation_sha256=args.preparation_sha256)
    elif args.action=='preflight':result=check(args.benchmark,args.journal,args.assessment)
    else:result=run(args.benchmark,args.journal,args.assessment)
    print(__import__('json').dumps(dict(action=args.action,jobs=result.get('total_jobs',len(result.get('jobs',[]))),
        complete=result.get('complete'),physical_launches=0 if args.action!='run' else None),indent=2))


if __name__=='__main__':main()
