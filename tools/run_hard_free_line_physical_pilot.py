#!/usr/bin/env python3
"""Frozen, independent 4x16384-per-arm physical line-guide pilot.

Only --run starts jobs. No automatic extension, assembly launch or retry.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from prepare_hard_free_line_score import read, sha, require, write, REGION_SHA, SHAPE_SHA
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_confirmation import runtime, worker_environment
from run_contact_tail_pilot import copy_frozen, verify_frozen
from run_full_vessel_comparison import execute_group
from run_mobile_posterior_pilot import verify_bundle
from prepare_hard_free_line_fresh import seeds_in, ROOTS, DECLARATIONS
from run_smc_guide_pilot import STRATA, CONVERGENCE
from analyze_hard_free_line_physical import analyze
from hard_free_line_physical_reference import audit

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hard-free-line-physical-pilot-v1'
ARMS = [dict(id=name, beta=beta, alpha=.5, component_count=92, samples=16384,
             lambda_ratio=128.) for name,beta in [('baseline',0.),('conditioned',1.)]]
SEEDS = [610012301+i for i in range(8)]


def validate_design(p):
    require(p['schema'] == SCHEMA and p['arms'] == ARMS, 'Changed physical allocation')
    require(p['strata'] == STRATA and p['convergence'] == CONVERGENCE, 'Changed original diagnostics')
    require(p['total_unconditional_draws'] == 131072 and p['cloud_replicates'] == 2
        and p['maximum_physical_workers'] == 2 and p['maximum_audit_workers'] == 4
        and p['maximum_total_workers'] == 32, 'Changed population/cloud/worker budget')
    require([j['seed'] for j in p['jobs']] == SEEDS and len(p['jobs']) == 8, 'Changed independent streams')
    for a in ARMS:
        selected = [j for j in p['jobs'] if j['arm'] == a['id']]
        require([j['id'] for j in selected] == [f'r{i:02}' for i in range(4)]
            and all(j['samples'] == 16384 for j in selected), 'Changed per-arm populations')
    require(p['physical_activity'] == .035 and p['depletant_radius'] == 1.5
        and p['region_sha256'] == REGION_SHA and p['shape_sha256'] == SHAPE_SHA, 'Changed physical target')
    require(not p['full_vessel_gate_open'] and not p['assembly_gate_open'], 'Pilot cannot open production gates')


def physical_command(root, arm, job):
    common = root/'common'
    return [str(common/'latent-region-normalizer'), '--config', str(common/'config.json'),
        '--region', str(common/'reference-package/region.json'), '--importance-guide', str(common/(arm['id']+'.json')),
        '--out', job['directory'], '--samples', str(job['samples']), '--seed', str(job['seed']),
        '--cloud-replicates', '2', '--lambda-ratio', '128']


def inventory(roots, proposed):
    files, seen = {}, set()
    command=['rg','--files','--hidden','--no-ignore',*map(str,roots)]
    for name in DECLARATIONS: command += ['-g',name]
    result=subprocess.run(command,text=True,capture_output=True)
    require(result.returncode in (0,1),'Seed inventory search failed: '+result.stderr)
    for path in sorted({Path(p).resolve() for p in result.stdout.splitlines()}):
        files[str(path)] = sha(path); seen.update(seeds_in(read(path)))
    require(not (set(proposed)&seen), 'Seed collision with an existing declaration')
    return dict(files=files, prior_seeds=sorted(seen), fresh_seeds=list(proposed))


def freeze(out, binary, bundle, rust_source, preparation, evidence, repository=ROOT):
    out,binary,bundle,rust_source,preparation,evidence,repository = map(lambda p:Path(p).resolve(),
        (out,binary,bundle,rust_source,preparation,evidence,repository))
    require(not out.exists(), 'Fresh immutable physical pilot required')
    verify_frozen(preparation)
    packet = read(evidence)
    require(packet['schema'] == 'hard-free-line-physical-prerequisites-v1' and packet['complete']
        and packet['all_checks_passed'] and packet['fresh_proposal_validated']
        and packet['binary_sha256'] == sha(binary) and packet['source_bundle_sha256'] == sha(bundle),
        'Exact-binary analytic/independent/fresh-proposal prerequisites required')
    for path,digest in packet['files'].items(): require(sha(path) == digest, 'Prerequisite source changed')
    source,rust_hashes = verify_bundle(binary,bundle,rust_source)
    old = Path('/vast/xvg/tetramer-mc-runs/contact-tail-pilot-20261001')
    package = old/'common/reference-package'; verify_frozen(package)
    require(sha(package/'region.json') == REGION_SHA and sha(package/'shape.json') == SHAPE_SHA,
        'Original physical package changed')
    seeds = inventory([repository/'runs',*map(Path,ROOTS)],SEEDS)
    sources = local_dependencies([Path(__file__)])
    out.mkdir(parents=True); common = out/'common'; common.mkdir()
    for name,path in sources.items(): shutil.copy2(path, common/name)
    shutil.copy2(binary,common/'latent-region-normalizer'); shutil.copy2(bundle,common/'source-bundle.json')
    for name,entry in source['files'].items():
        p=common/'rust-source'/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(entry['text'])
    copy_frozen(package,common/'reference-package')
    shutil.copy2(evidence,common/'prerequisites.json')
    write(common/'seed-inventory.json',seeds)
    config=read(package/'config.json'); config['shape']=str(common/'reference-package/shape.json')
    write(common/'config.json',config)
    candidate=read(preparation/'guides/xyz.json')
    require(candidate['region_sha256'] == REGION_SHA and candidate['raw_translation_axes'] == [0,1,2]
        and candidate['minimum_conditional_mass'] == 1e-12
        and len(candidate['gaussian_components']) == 92
        and candidate['defensive_uniform_shell_probability'] == .5
        and 'contact_widths_A' not in candidate and 'contact_neighbor_indices' not in candidate,
        'Frozen primary xyz law changed')
    jobs=[]
    for arm in ARMS:
        guide=copy.deepcopy(candidate); guide['conditional_probability']=arm['beta']; write(common/(arm['id']+'.json'),guide)
        (out/arm['id']).mkdir(); (out/'audits'/arm['id']).mkdir(parents=True)
        for index in range(4):
            job=dict(id=f'r{index:02}',arm=arm['id'],kind='physical',status='pending',samples=16384,
                seed=SEEDS[len(jobs)],directory=str(out/arm['id']/f'r{index:02}'),
                log=str(out/arm['id']/f'r{index:02}.log'))
            job['command']=physical_command(out,arm,job);jobs.append(job)
    p=dict(schema=SCHEMA,arms=ARMS,jobs=jobs,total_unconditional_draws=131072,cloud_replicates=2,
        maximum_physical_workers=2,maximum_audit_workers=4,maximum_total_workers=32,
        physical_activity=.035,depletant_radius=1.5,region_sha256=REGION_SHA,shape_sha256=SHAPE_SHA,
        binary_sha256=sha(binary),source_bundle_sha256=sha(bundle),rust_sources=rust_hashes,
        controller_sha256=sha(__file__),python_sources={n:sha(v) for n,v in sources.items()},
        repository=str(repository),runtime=runtime(),strata=STRATA,convergence=CONVERGENCE,
        supplemental_definition_sha256=sha(package/'native-partition-definition.json'),
        preparation=str(preparation),preparation_freeze_sha256=sha(preparation/'freeze.json'),
        prerequisites=str(evidence),prerequisites_sha256=sha(evidence),
        full_vessel_gate_open=False,assembly_gate_open=False,
        rationale='Fresh matched independent pilot because old92 populations informed guide selection. Completed old weights/classifications are retained as retrospective evidence, not replayed or pooled.',
        scope='Fixed R4 native/competing/unbound weights and importance ESS per sampler CPU only; no adaptive stopping, proposal refitting, continuation, retry, hidden invalid-draw removal or assembly conclusion.')
    validate_design(p);write(out/'protocol.json',p)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    validate(out,sha(out/'protocol.json'));return sha(out/'protocol.json')


def validate(out,expected):
    out=Path(out).resolve();verify_frozen(out)
    require(sha(out/'protocol.json') == expected, 'Protocol changed')
    p=read(out/'protocol.json');validate_design(p)
    require(runtime() == p['runtime'] and sys.flags.optimize == 0, 'Audit runtime changed')
    require(sha(__file__) == p['controller_sha256'] and
        {n:sha(v) for n,v in local_dependencies([Path(__file__)]).items()} == p['python_sources'], 'Frozen source closure changed')
    verify_bundle(out/'common/latent-region-normalizer',out/'common/source-bundle.json',out/'common/rust-source')
    packet=read(out/'common/prerequisites.json')
    for path,digest in packet['files'].items():require(sha(path)==digest,'Prerequisite evidence changed')
    require(sha(p['prerequisites'])==p['prerequisites_sha256'],'Prerequisite receipt changed')
    preparation=Path(p['preparation']);verify_frozen(preparation)
    require(sha(preparation/'freeze.json')==p['preparation_freeze_sha256'],'Prepared guide binding changed')
    for j in p['jobs']:
        arm=next(a for a in ARMS if a['id']==j['arm'])
        require(j['command']==physical_command(out,arm,j),'Physical command changed')
    return p


def verify_output(root, protocol, job):
    directory=Path(job['directory']);common=root/'common'
    manifest=read(directory/'manifest.json');summary=read(directory/'summary.json')
    expected=dict(schema='importance-latent-region-normalizer-v6',
        guide_schema='defensive-hard-free-line-guide-v1',samples=job['samples'],seed=job['seed'],
        activity=.035,lambda_ratio=128.,cloud_replicates=2,importance_uniform_probability=.5,
        importance_component_count=92,region_sha256=REGION_SHA,shape_sha256=SHAPE_SHA,
        executable_sha256=protocol['binary_sha256'],source_bundle_sha256=protocol['source_bundle_sha256'],
        config_sha256=sha(common/'config.json'),importance_guide_sha256=sha(common/(job['arm']+'.json')))
    require(all(manifest.get(k)==v for k,v in expected.items()),'Completed population identity/target/executable differs')
    require(summary['complete'] and summary['samples']==job['samples'] and summary['manifest']==manifest
        and not (directory/'failure.json').exists(),'Incomplete/failed physical population')
    for name in ('samples','attempts'):
        require(sha(directory/(name+'.jsonl'))==summary[name+'_sha256'],'Raw output changed: '+name)
    return {name+'_sha256':sha(directory/(name+suffix)) for name,suffix in
            [('samples','.jsonl'),('attempts','.jsonl'),('summary','.json'),('manifest','.json')]}


def run(out,expected):
    out=Path(out).resolve();p=validate(out,expected)
    require(Path(__file__).resolve()==out/'common/run_hard_free_line_physical_pilot.py','Use frozen controller')
    require(not (out/'status.json').exists(),'No retry/overwrite of a started physical pilot')
    state=dict(complete=False,phase='physical',started=time.time(),pid=os.getpid(),
        protocol_sha256=expected,jobs=copy.deepcopy(p['jobs']),audits=[])
    write(out/'status.json',state)
    def snapshot():
        temporary=out/'status.tmp';temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(out/'status.json')
    def interrupted(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);raise InterruptedError('Drain started children')
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        execute_group(state['jobs'],snapshot,2,p['repository'],worker_environment(),before_launch=lambda:validate(out,expected))
        for j in state['jobs']:j['output']=verify_output(out,p,j)
        state['phase']='independent_audit';snapshot()
        for j in p['jobs']:
            result=out/'audits'/j['arm']/(j['id']+'.json')
            state['audits'].append(dict(id=j['arm']+'/'+j['id'],kind='audit',status='pending',directory=str(result),
                log=str(result.with_suffix('.log')),command=[sys.executable,'-B',str(out/'common/hard_free_line_physical_reference.py'),
                '--directory',j['directory'],'--out',str(result)]))
        execute_group(state['audits'],snapshot,4,p['repository'],worker_environment())
        state['phase']='classification';snapshot()
        result=analyze(out)
        state['comparison_sha256']=sha(out/'comparison/analysis.json')
        validate(out,expected);state.update(complete=True,phase='complete',finished=time.time());snapshot()
    except BaseException as error:
        state.update(complete=False,phase=state['phase']+'_failed',error=repr(error),finished=time.time());snapshot();raise
    finally:signal.signal(signal.SIGTERM,previous)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze')
    for name in ('out','binary','bundle','rust-source','preparation','evidence'):f.add_argument('--'+name,type=Path,required=True)
    for action in ('validate','run'):
        p=sub.add_parser(action);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-protocol-sha256',required=True)
    a=parser.parse_args()
    if a.action=='freeze':print(freeze(a.out,a.binary,a.bundle,a.rust_source,a.preparation,a.evidence))
    elif a.action=='validate':validate(a.out,a.expected_protocol_sha256);print('Validated; no jobs launched')
    else:print(run(a.out,a.expected_protocol_sha256)['phase'])
