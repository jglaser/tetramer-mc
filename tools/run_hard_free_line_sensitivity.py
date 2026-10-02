#!/usr/bin/env python3
"""Fixed independent alpha/cloud controls against the completed original pilot.

Freeze and validate launch nothing. Only run starts the two fixed control arms;
there is no retry, resume, source-pilot replay or automatic scientific gate opening.
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
from analyze_hard_free_line_sensitivity import PLAN, CONTROLS, analyze, compare_sensitivity
from hard_free_line_physical_reference import audit  # include the unchanged executable audit closure

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hard-free-line-sensitivity-stage-v1'
PILOT_SCHEMA = 'hard-free-line-physical-pilot-v1'
PILOT_ARMS = [dict(id=name, beta=beta, alpha=.5, component_count=92, samples=16384,
                  lambda_ratio=128.) for name,beta in [('baseline',0.),('conditioned',1.)]]
PILOT_SEEDS = [610012301+i for i in range(8)]
ARMS = copy.deepcopy(CONTROLS)
SEEDS = [610014301+i for i in range(8)]
NEW_SOURCES = {'run_hard_free_line_sensitivity.py', 'analyze_hard_free_line_sensitivity.py',
               'analyze_hard_free_line_population_size.py'}
UNCHANGED = ('latent-region-normalizer','source-bundle.json','config.json')


def validate_allocation(p, schema, arms, seeds):
    require(p['schema'] == schema and p['arms'] == arms, 'Changed fixed sensitivity/source allocation')
    require(p['strata'] == STRATA and p['convergence'] == CONVERGENCE, 'Changed original diagnostics')
    require(p['total_unconditional_draws'] == 131072 and p['cloud_replicates'] == 2
        and p['maximum_physical_workers'] == 2 and p['maximum_audit_workers'] == 4
        and p['maximum_total_workers'] == 32, 'Changed population/cloud/worker budget')
    require([j['seed'] for j in p['jobs']] == seeds and len(p['jobs']) == 8, 'Changed independent streams')
    for arm in arms:
        selected = [j for j in p['jobs'] if j['arm'] == arm['id']]
        require([j['id'] for j in selected] == [f'r{i:02}' for i in range(4)]
            and all(j['samples'] == 16384 for j in selected), 'Changed per-arm populations')
    require(p['physical_activity'] == .035 and p['depletant_radius'] == 1.5
        and p['region_sha256'] == REGION_SHA and p['shape_sha256'] == SHAPE_SHA, 'Changed physical target')
    require(not p['full_vessel_gate_open'] and not p['assembly_gate_open'], 'Regional controls cannot open production gates')


def validate_design(p):
    validate_allocation(p, SCHEMA, ARMS, SEEDS)
    require(p['sensitivity_comparison'] == PLAN and p['pilot_evidence']['schema'] == PILOT_SCHEMA
        and p['pilot_evidence']['complete'] is True and p['pilot_evidence']['total_unconditional_draws'] == 131072
        and p['old_samples_pooled'] is False, 'Missing fixed independent sensitivity evidence/plan')


def physical_command(root, arm, job):
    common = Path(root)/'common'
    return [str(common/'latent-region-normalizer'), '--config', str(common/'config.json'),
        '--region', str(common/'reference-package/region.json'), '--importance-guide', str(common/(arm['id']+'.json')),
        '--out', job['directory'], '--samples', str(job['samples']), '--seed', str(job['seed']),
        '--cloud-replicates', '2', '--lambda-ratio', format(arm['lambda_ratio'], 'g')]


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

def completed_pilot(pilot):
    """Authenticate terminal aggregate evidence without replaying any old row."""
    pilot=Path(pilot).resolve();state=read(pilot/'status.json')
    require(state['complete'] is True and state['phase']=='complete', 'Source pilot must be complete')
    require(len(state['jobs'])==len(state['audits'])==8 and all(j['status']=='complete'
        and j['returncode']==0 for j in state['jobs']+state['audits']), 'Source pilot jobs/audits incomplete')
    frozen=verify_frozen(pilot);p=read(pilot/'protocol.json');validate_allocation(p, PILOT_SCHEMA, PILOT_ARMS, PILOT_SEEDS)
    require(p['schema']==PILOT_SCHEMA and sha(pilot/'protocol.json')==state['protocol_sha256'],
            'Sensitivity source must be the original fixed pilot')
    data=read(pilot/'comparison/analysis.json')
    require(sha(pilot/'comparison/analysis.json')==state['comparison_sha256']
        and data['complete'] is True and data['schema']=='hard-free-line-physical-comparison-v1'
        and data['protocol_sha256']==state['protocol_sha256']
        and data['total_unconditional_draws']==131072 and set(data['arms'])=={'baseline','conditioned'},
        'Source pilot aggregate binding changed')
    require(not data['diagnostics']['full_vessel_gate_open'] and not data['diagnostics']['assembly_gate_open'],
            'Source pilot cannot establish assembly or full-vessel stability')
    files={str(pilot/name):digest for name,digest in frozen.items()}
    for path in (pilot/'status.json',pilot/'freeze.json',pilot/'comparison/analysis.json'):
        files[str(path)]=sha(path)
    for arm in PILOT_ARMS:
        records=data['arms'][arm['id']]['populations']
        require(len(records)==4 and len({r['id'] for r in records})==4
            and len({r['seed']for r in records})==4,'Source pilot populations missing or repeated')
        for record in records:
            jobs=[j for j in p['jobs'] if j['arm']==arm['id'] and j['id']==record['id']]
            require(len(jobs)==1 and all(record[k]==jobs[0][k]for k in ('seed','samples','arm')),
                    'Source pilot population identity differs')
            terminal=[j for j in state['jobs'] if j['arm']==arm['id'] and j['id']==record['id']]
            require(len(terminal)==1 and terminal[0]['output']['samples_sha256']==record['samples_sha256'],
                    'Source classification differs from completed draws')
            path=pilot/'audits'/arm['id']/(record['id']+'.json');audit_value=read(path)
            require(sha(path)==record['independent_audit_sha256'] and audit_value['complete']
                and audit_value['geometry_mode']=='full' and audit_value['samples']==16384
                and audit_value['samples_sha256']==record['samples_sha256'], 'Source independent audit differs')
            files[str(path)]=sha(path)
        require(all(e['row_uncertainty']['draws']==65536 for e in data['arms'][arm['id']]['estimates'].values()),
                'Source pilot dropped attempted denominators')
    evidence=dict(schema=PILOT_SCHEMA,complete=True,root=str(pilot),total_unconditional_draws=131072,
        protocol_sha256=state['protocol_sha256'],comparison_sha256=state['comparison_sha256'],files=files,
        scope='Completed fixed pilot informs only the new fixed sensitivity design. Its rows, classifiers and audits '
              'are not replayed, and its estimates are not pooled with new populations.')
    return p,data,evidence

def validate_prerequisites(path, binary_sha256, bundle_sha256):
    packet = read(path)
    require(packet['schema'] == 'hard-free-line-sensitivity-prerequisites-v1'
        and packet['complete'] is True and packet['all_checks_passed'] is True
        and packet['binary_sha256'] == binary_sha256 and packet['source_bundle_sha256'] == bundle_sha256
        and packet['arms'] == ARMS and packet['independent_geometry'] == 'full',
        'Exact-binary sensitivity reference prerequisites required')
    require(isinstance(packet['files'], dict) and bool(packet['files']), 'Empty sensitivity reference evidence')
    for path,digest in packet['files'].items():
        require(Path(path).is_absolute() and sha(path) == digest, 'Sensitivity prerequisite evidence changed: '+path)
    return packet


def validate_guide(guide, alpha):
    require(guide['schema'] == 'defensive-hard-free-line-guide-v1' and guide['region_sha256'] == REGION_SHA
        and guide['raw_translation_axes'] == [0,1,2] and guide['minimum_conditional_mass'] == 1e-12
        and guide['conditional_probability'] == 1. and guide['defensive_uniform_shell_probability'] == alpha
        and len(guide['gaussian_components']) == 92
        and 'contact_widths_A' not in guide and 'contact_neighbor_indices' not in guide,
        'Conditioned guide law changed')


def freeze(out, pilot, prerequisites, repository=ROOT):
    out,pilot,prerequisites,repository = map(lambda p:Path(p).resolve(),(out,pilot,prerequisites,repository))
    require(not out.exists(), 'Fresh immutable sensitivity stage required')
    old,_,evidence = completed_pilot(pilot)
    previous = pilot/'common'
    source,rust_hashes = verify_bundle(previous/'latent-region-normalizer',previous/'source-bundle.json',previous/'rust-source')
    require(sha(previous/'latent-region-normalizer') == old['binary_sha256']
        and sha(previous/'source-bundle.json') == old['source_bundle_sha256'], 'Validated pilot binary/bundle changed')
    validate_prerequisites(prerequisites, old['binary_sha256'], old['source_bundle_sha256'])
    config = read(previous/'config.json')
    require(Path(config['shape']).resolve() == previous/'reference-package/shape.json', 'Pilot configuration has an unbound shape path')
    require(sha(previous/'reference-package/region.json') == REGION_SHA
        and sha(previous/'reference-package/shape.json') == SHAPE_SHA
        and config['depletant_radius'] == 1.5 and config['reservoir_density'] == .035, 'Pilot physical shape/region/bath changed')
    original_guide = read(previous/'conditioned.json')
    validate_guide(original_guide, .5)
    seeds = inventory([repository/'runs',*map(Path,ROOTS)], SEEDS)
    require(set(SEEDS).isdisjoint(j['seed'] for j in old['jobs']), 'Sensitivity seeds reuse pilot streams')
    sources = local_dependencies([Path(__file__)])
    require(NEW_SOURCES.issubset(sources), 'Missing new aggregate/controller source')
    for name,path in sources.items():
        if name not in NEW_SOURCES:
            require(name in old['python_sources'] and sha(path) == old['python_sources'][name]
                and sha(previous/name) == old['python_sources'][name], 'Reused Python audit/classifier changed: '+name)
    out.mkdir(parents=True)
    common = out/'common'; common.mkdir()
    for name,path in sources.items():
        shutil.copy2(path if name in NEW_SOURCES else previous/name, common/name)
    for name in UNCHANGED:
        shutil.copy2(previous/name, common/name)
    shutil.copy2(previous/'conditioned.json', common/'pilot-conditioned.json')
    shutil.copy2(previous/'conditioned.json', common/'lambda64.json')
    alpha02 = copy.deepcopy(original_guide); alpha02['defensive_uniform_shell_probability'] = .2
    write(common/'alpha02.json', alpha02)
    shutil.copy2(prerequisites, common/'prerequisites.json')
    for name,entry in source['files'].items():
        path = common/'rust-source'/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(entry['text'])
    copy_frozen(previous/'reference-package', common/'reference-package')
    write(common/'seed-inventory.json', seeds)
    write(common/'pilot-evidence.json', evidence)
    write(common/'sensitivity-comparison-plan.json', PLAN)
    jobs = []
    for arm in ARMS:
        (out/arm['id']).mkdir(); (out/'audits'/arm['id']).mkdir(parents=True)
        for i in range(4):
            job = dict(id=f'r{i:02}',arm=arm['id'],kind='physical',status='pending',samples=16384,
                seed=SEEDS[len(jobs)],directory=str(out/arm['id']/f'r{i:02}'),log=str(out/arm['id']/f'r{i:02}.log'))
            job['command'] = physical_command(out, arm, job); jobs.append(job)
    p = copy.deepcopy(old)
    p.update(schema=SCHEMA,arms=copy.deepcopy(ARMS),jobs=jobs,total_unconditional_draws=131072,
        controller_sha256=sha(__file__),python_sources={n:sha(v) for n,v in sources.items()},
        new_aggregate_sources=sorted(NEW_SOURCES),repository=str(repository),runtime=runtime(),rust_sources=rust_hashes,
        pilot_evidence=evidence,sensitivity_comparison=copy.deepcopy(PLAN),old_samples_pooled=False,
        prerequisites=str(prerequisites),prerequisites_sha256=sha(prerequisites),
        unchanged_pilot_common_sha256={name:sha(previous/name) for name in UNCHANGED},
        pilot_conditioned_sha256=sha(previous/'conditioned.json'),
        rationale='Independent one-factor guide/cloud controls against the completed conditioned pilot. '
                  'Only defensive probability or auxiliary intensity changes in each control; no old data are pooled.',
        scope='Four independent 16384-draw populations per control, two clouds per valid pose, '
              'all unconditional zeros retained. No retry, guide fit, pilot replay or gate relaxation.')
    validate_design(p); write(out/'protocol.json', p)
    write(out/'freeze.json', dict(files={str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*')) if path.is_file()}))
    validate(out, sha(out/'protocol.json'))
    return sha(out/'protocol.json')


def validate(out, expected):
    out = Path(out).resolve(); verify_frozen(out)
    require(sha(out/'protocol.json') == expected, 'Protocol changed')
    p = read(out/'protocol.json'); validate_design(p)
    require(runtime() == p['runtime'] and sys.flags.optimize == 0, 'Audit runtime changed')
    require(sha(__file__) == p['controller_sha256']
        and {n:sha(v) for n,v in local_dependencies([Path(__file__)]).items()} == p['python_sources']
        and p['new_aggregate_sources'] == sorted(NEW_SOURCES), 'Frozen source closure changed')
    common = out/'common'
    verify_bundle(common/'latent-region-normalizer', common/'source-bundle.json', common/'rust-source')
    require(sha(common/'prerequisites.json') == p['prerequisites_sha256']
        and sha(p['prerequisites']) == p['prerequisites_sha256'], 'Sensitivity prerequisite receipt changed')
    validate_prerequisites(common/'prerequisites.json', p['binary_sha256'], p['source_bundle_sha256'])
    require(read(common/'pilot-evidence.json') == p['pilot_evidence']
        and read(common/'sensitivity-comparison-plan.json') == PLAN, 'Sensitivity frozen evidence/analysis plan changed')
    for path,digest in p['pilot_evidence']['files'].items():
        require(sha(path) == digest, 'Completed pilot evidence changed: '+path)
    for name,digest in p['unchanged_pilot_common_sha256'].items():
        require(sha(common/name) == digest, 'Sensitivity physical input differs from pilot: '+name)
    require(sha(common/'pilot-conditioned.json') == p['pilot_conditioned_sha256']
        and sha(common/'lambda64.json') == p['pilot_conditioned_sha256'], 'Original guide bytes changed')
    original = read(common/'pilot-conditioned.json'); validate_guide(original, .5)
    for arm in ARMS:
        guide = read(common/(arm['id']+'.json')); validate_guide(guide, arm['alpha'])
        expected_guide = dict(original, defensive_uniform_shell_probability=arm['alpha'])
        require(guide == expected_guide, 'Sensitivity guide differs beyond defensive probability')
    for job in p['jobs']:
        arm = next(a for a in p['arms'] if a['id'] == job['arm'])
        require(job['directory'] == str(out/arm['id']/job['id'])
            and job['log'] == str(out/arm['id']/(job['id']+'.log'))
            and job['command'] == physical_command(out,arm,job), 'Physical command/destination changed')
    return p


def verify_output(root, protocol, job):
    root = Path(root); directory = Path(job['directory']); common = root/'common'
    arm = next(a for a in ARMS if a['id'] == job['arm'])
    manifest = read(directory/'manifest.json'); summary = read(directory/'summary.json')
    expected = dict(schema='importance-latent-region-normalizer-v6',
        guide_schema='defensive-hard-free-line-guide-v1',samples=job['samples'],seed=job['seed'],
        activity=.035,lambda_ratio=arm['lambda_ratio'],cloud_replicates=2,
        importance_uniform_probability=arm['alpha'],importance_component_count=92,
        region_sha256=REGION_SHA,shape_sha256=SHAPE_SHA,
        executable_sha256=protocol['binary_sha256'],source_bundle_sha256=protocol['source_bundle_sha256'],
        config_sha256=sha(common/'config.json'),importance_guide_sha256=sha(common/(job['arm']+'.json')))
    require(all(manifest.get(k) == v for k,v in expected.items()), 'Completed population identity/target/executable differs')
    require(summary['complete'] and summary['samples'] == job['samples'] and summary['manifest'] == manifest
        and not (directory/'failure.json').exists(), 'Incomplete/failed physical population')
    for name in ('samples','attempts'):
        require(sha(directory/(name+'.jsonl')) == summary[name+'_sha256'], 'Raw output changed: '+name)
    return {name+'_sha256':sha(directory/(name+suffix)) for name,suffix in
            [('samples','.jsonl'),('attempts','.jsonl'),('summary','.json'),('manifest','.json')]}


def claim_status(out,state):
    """Atomic one-run claim; a concurrent invocation cannot replace accounting."""
    with (Path(out)/'status.json').open('x') as stream:
        stream.write(json.dumps(state,indent=2,allow_nan=False)+'\n')

def require_host_namespace():
    """The shared capacity scanner must see host processes, not sandbox PID 1."""
    require("codex-linux-san" not in Path("/proc/1/comm").read_text(),
        "Run only in the host PID namespace; sandbox cannot observe global capacity")


def run(out,expected):
    out=Path(out).resolve();p=validate(out,expected)
    require(Path(__file__).resolve()==out/'common/run_hard_free_line_sensitivity.py','Use frozen controller')
    require_host_namespace()
    require(not (out/'status.json').exists(),'No retry/overwrite of a started sensitivity stage')
    state=dict(complete=False,phase='physical',started=time.time(),pid=os.getpid(),
        protocol_sha256=expected,jobs=copy.deepcopy(p['jobs']),audits=[])
    claim_status(out,state)
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
        state['phase']='sensitivity_comparison';snapshot()
        pilot_path=Path(p['pilot_evidence']['root'])/'comparison/analysis.json'
        require(sha(pilot_path)==p['pilot_evidence']['comparison_sha256'],'Completed pilot summary changed')
        comparison=compare_sensitivity(read(pilot_path),result,p['sensitivity_comparison'])
        comparison['input_sha256']={str(pilot_path):sha(pilot_path),
            str(out/'comparison/analysis.json'):state['comparison_sha256'],str(out/'protocol.json'):expected}
        comparison_path=out/'sensitivity-comparison.json'
        write(comparison_path,comparison);state['sensitivity_comparison_sha256']=sha(comparison_path)
        validate(out,expected);state.update(complete=True,phase='complete',finished=time.time());snapshot()
    except BaseException as error:
        state.update(complete=False,phase=state['phase']+'_failed',error=repr(error),finished=time.time());snapshot();raise
    finally:signal.signal(signal.SIGTERM,previous)
    return state

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze')
    for name in ('out','pilot','prerequisites'):f.add_argument('--'+name,type=Path,required=True)
    for action in ('validate','run'):
        p=sub.add_parser(action);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-protocol-sha256',required=True)
    args=parser.parse_args()
    if args.action == 'freeze':print(freeze(args.out,args.pilot,args.prerequisites))
    elif args.action == 'validate':validate(args.out,args.expected_protocol_sha256);print('Validated; no jobs launched')
    else:print(run(args.out,args.expected_protocol_sha256)['phase'])
