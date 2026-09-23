#!/usr/bin/env python3
"""Gated, one-shot execution of the unchanged frozen full-vessel allocation.

Freeze and preflight launch nothing. Failed scientific gates cannot be bypassed
by this controller. Started children drain on failure; no draw is retried.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from analyze_r4_smc_control import Ledger, read, require, sha, write
from prepare_full_vessel_comparison import validate as validate_preparation, authenticate_gate_inputs, THREADS
from prepare_shoulder_docking_benchmark import local_dependencies
from run_smc_importance_bridge import (SCHEMA as BRIDGE_SCHEMA, runtime, process_token,
    dependency_contract, comparison_steps, bind_terminal, validate_result)

SCHEMA = 'gated-full-vessel-workflow-v1'
PHYSICAL_NAMES = {'basin-normalizer', 'latent-region-normalizer', 'latent-region-smc',
    'native-region-normalizer', 'tetramer-mc', 'docking-mc', 'gate-benchmark', 'gate_benchmark', 'contact_atlas'}


def stages_for(out, preparation, plan):
    out, preparation = Path(out).resolve(), Path(preparation).resolve()
    stages = []
    for stage in ('standard', 'large'):
        groups = []
        jobs = [j for j in plan['jobs'] if j['stage'] == stage]
        require(len(jobs) == 8, 'Each stage needs both four-population arms')
        for kind in ('physical', 'audit', 'partition'):
            group = []
            for job in jobs:
                directory = job['directory' if kind == 'physical' else kind+'_directory']
                command = job['command' if kind == 'physical' else kind+'_command']
                group.append(dict(id=job['id']+'-'+kind, job_id=job['id'], kind=kind,
                    directory=directory, command=command,
                    log=job['log'] if kind == 'physical' else str(out/'logs'/(job['id']+'-'+kind+'.log'))))
            groups.append(dict(kind=kind, workers=8 if kind == 'physical' else 4, steps=group))
        groups.append(dict(kind='aggregate', workers=1, steps=[dict(id=stage+'-aggregate',
            kind='aggregate', directory=str(out/(stage+'-comparison')),
            log=str(out/'logs'/(stage+'-aggregate.log')),
            command=[sys.executable, '-B', '-E', str(out/'common/compare_full_vessel_stage.py'),
                '--preparation', str(preparation), '--stage', stage, '--out', str(out/(stage+'-comparison'))])]))
        stages.append(dict(name=stage, groups=groups))
    return stages


def bind_preparation(ledger, directory, expected):
    directory = Path(directory).resolve(); ledger.frozen(directory)
    ledger.bind(directory/'plan.json', expected); plan = validate_preparation(directory)
    for path, digest in plan['source_bindings'].items(): ledger.bind(path, digest)
    ledger.bind(directory/'common/basin-normalizer', plan['binary_sha256'])
    ledger.bind(directory/'common/source-bundle.json', plan['source_bundle_sha256'])
    return plan


def bind_release(ledger, path, expected, preparation):
    path = Path(path).resolve(); report = read(ledger.bind(path, expected)); root = path.parent
    require(report['schema'] == 'full-vessel-release-reference-validation-v1'
        and report['complete'] is True and report['failure'] is None
        and report['optimized_release_executable'] is True and report['same_binary_both_arms'] is True
        and report['retries'] == 0 and report['not_launched'] == []
        and report['attempts'] == 256, 'Incomplete optimized release validation')
    for key in ('binary_sha256', 'source_bundle_sha256'):
        require(report[key] == preparation[key], 'Release reference artifact differs: '+key)
    require(report['rust_source_sha256'] == preparation['rust_sources'], 'Release source closure differs')
    require(all(digest==preparation['sources'][name] for name,digest in report['audit_source_sha256'].items() if name in preparation['sources']), 'Checked release auditor source differs')
    require(all(name in report['audit_source_sha256'] for name in ('audit_full_vessel_baseline.py','audit_full_vessel_latent.py')), 'Release auditor references missing')
    ledger.bind(root/'plan.json', report['plan_sha256']); ledger.bind(root/'freeze.json', report['freeze_sha256'])
    for name, digest in report['all_files_sha256'].items():
        file = (root/name).resolve(); require(file.is_relative_to(root), 'Release artifact escaped its root')
        ledger.bind(file, digest)
    jobs = report['jobs']; expected_names = {'legacy-all', 'reciprocal-all', 'reciprocal-selected', 'half-mixture-reciprocal-all'}
    require(len(jobs) == 4 and {j['name'] for j in jobs} == expected_names, 'Release reference controls changed')
    for job in jobs:
        require(job['complete'] is True and job['physical_returncode'] == job['audit_returncode'] == 0
            and job['executable_verified'] is True and job['attempts'] == 64, 'Release reference did not finish')
        ledger.bind(root/job['name']/'audit/analysis.json', job['audit_analysis_sha256'])
    return report


def bind_bridge(ledger, directory, expected):
    directory = Path(directory).resolve(); ledger.frozen(directory)
    bridge = read(ledger.bind(directory/'plan.json', expected))
    require(bridge['schema'] == BRIDGE_SCHEMA and bridge['physical_workers'] == 0
        and bridge['comparison_workers'] == 1 and bridge['runtime'] == runtime(), 'Unexpected bridge runtime or law')
    for name, digest in bridge['sources'].items(): ledger.bind(directory/'common'/name, digest)
    for name, digest in bridge['input_bindings'].items(): ledger.bind(name, digest)
    dependency = bridge['dependency']; smc_root = Path(dependency['directory'])
    source = read(ledger.bind(smc_root/'plan.json', dependency['plan_sha256']))
    require(dependency['contract'] == dependency_contract(smc_root, source), 'SMC dependency allocation changed')
    require(bridge['steps'] == comparison_steps(directory, smc_root, bridge['importance']['path'], sys.executable),
        'Bridge comparison destinations or commands changed')
    return bridge, source



def match_target(confirmation, preparation, regional_config, vessel_config):
    pairs = [('shape_sha256','shape.json'), ('region_sha256','current_R4.json'),
             ('reference_region_sha256','old_alternative_R5.json')]
    require(all(confirmation[k]==preparation['input_sha256'][name] for k,name in pairs)
        and confirmation['native_definition']['definition_sha256']==preparation['native_definition_sha256'],
        'Regional gate belongs to a different shape, region or native classifier')
    require(all(regional_config[k]==vessel_config[k] for k in ('depletant_radius','reservoir_density','fixed_poses','metadata')),
        'Regional gate bath, scaffold or physical registration metric differs')


def bind_matching_target(ledger, bridge, preparation, preparation_root):
    confirmation=read(ledger.bind(bridge['importance']['path'],bridge['importance']['sha256']))
    regional=read(ledger.bind(Path(confirmation['campaign'])/'bank/provenance/config.json'))
    vessel=read(ledger.bind(Path(preparation_root)/'inputs/config.json',preparation['input_sha256']['config.json']))
    match_target(confirmation,preparation,regional,vessel)


def authenticated_gate(ledger, bridge_root, bridge, smc):
    confirmation=read(ledger.bind(bridge['importance']['path'],bridge['importance']['sha256']))
    if confirmation['convergence']['confirmation_passed'] is not True:
        failed=[name for name,passed in confirmation['convergence']['checks'].items() if passed is not True]
        raise ValueError('Frozen regional convergence gate failed: '+', '.join(failed))
    bridge_root = Path(bridge_root); state = read(ledger.bind(bridge_root/'status.json'))
    require(state['schema'] == BRIDGE_SCHEMA and state['plan_sha256'] == sha(bridge_root/'plan.json')
        and state.get('error') is None and state['complete'] is True and state['phase'] == 'complete',
        'Independent comparison workflow is not successfully complete')
    require(len(state['steps']) == 2, 'Missing bridge comparison steps')
    bind_terminal(bridge['dependency'], smc, ledger)
    for expected, actual in zip(bridge['steps'], state['steps']):
        require(all(actual.get(k) == v for k, v in expected.items()) and actual['returncode'] == 0,
            'Bridge terminal command differs or failed')
        checked = validate_result(expected, bridge, ledger)
        require(all(actual.get(k) == v for k, v in checked.items()), 'Bridge output ledger differs')
    protocols = dict(zip(('broad', 'narrow'), [c['protocol_sha256'] for c in smc['controls']]))
    result = authenticate_gate_inputs(bridge['importance']['path'],
        [Path(s['output'])/'analysis.json' for s in bridge['steps']], protocols)
    for path, digest in result['input_sha256'].items(): ledger.bind(path, digest)
    return result


def freeze(out, preparation, preparation_sha256, release, release_sha256, bridge_root, bridge_sha256):
    out, preparation, release, bridge_root = map(lambda p: Path(p).resolve(), (out, preparation, release, bridge_root))
    require(not out.exists(), 'Fresh workflow directory required'); ledger = Ledger()
    plan = bind_preparation(ledger, preparation, preparation_sha256)
    bind_release(ledger, release, release_sha256, plan)
    bridge, _ = bind_bridge(ledger, bridge_root, bridge_sha256)
    bind_matching_target(ledger,bridge,plan,preparation)
    # The exact confirmation may fail. Freeze records it; only run applies the gate.
    ledger.bind(bridge['importance']['path'], bridge['importance']['sha256'])
    source = Path(__file__).resolve(); aggregator = source.with_name('compare_full_vessel_stage.py')
    closure = local_dependencies([source, aggregator]); original = {str(p):sha(p) for p in closure.values()}
    out.mkdir(); (out/'common').mkdir(); (out/'logs').mkdir()
    for name, path in closure.items(): shutil.copy2(path, out/'common'/name)
    sources = {name:sha(out/'common'/name) for name in closure}
    require(all(sources[Path(p).name] == digest for p,digest in original.items()), 'Source changed while archiving')
    value = dict(schema=SCHEMA, created=time.time(), repository=plan['repository'], runtime=runtime(),
        preparation=dict(path=str(preparation), sha256=preparation_sha256),
        release=dict(path=str(release), sha256=release_sha256), bridge=dict(path=str(bridge_root), sha256=bridge_sha256),
        input_bindings=ledger.files, sources=sources,
        aggregator_sources={name:sources[name] for name in local_dependencies([aggregator])}, thread_environment=THREADS,
        stages=stages_for(out, preparation, plan), physical_cap=8, worker_cap=32,
        physics_launched=False, scope='Frozen next calculation only. No assembly authorization. Fixed failing evidence cannot silently be replaced.',
        continuation='Read-only preflight may repeat. Run is one-shot. No normalizer checkpoints exist; partial attempts remain and are never retried.')
    ledger.recheck(); write(out/'plan.json', value)
    write(out/'freeze.json', dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    return value


def validate(out, expected, gate=True):
    out = Path(out).resolve(); ledger = Ledger(); ledger.frozen(out)
    plan = read(ledger.bind(out/'plan.json', expected))
    require(plan['schema'] == SCHEMA and plan['runtime'] == runtime() and sys.flags.optimize == 0
        and plan['thread_environment'] == THREADS and plan['physical_cap'] == 8 and plan['worker_cap'] == 32,
        'Workflow runtime, worker budget or source law changed')
    require(Path(__file__).resolve() == out/'common/run_full_vessel_comparison.py', 'Use the frozen controller in common/')
    for name,digest in plan['sources'].items(): ledger.bind(out/'common'/name,digest)
    for path,digest in plan['input_bindings'].items(): ledger.bind(path,digest)
    prep = bind_preparation(ledger, **dict(directory=plan['preparation']['path'], expected=plan['preparation']['sha256']))
    bind_release(ledger, plan['release']['path'], plan['release']['sha256'], prep)
    bridge, smc = bind_bridge(ledger, plan['bridge']['path'], plan['bridge']['sha256'])
    bind_matching_target(ledger,bridge,prep,plan['preparation']['path'])
    require(plan['stages'] == stages_for(out, plan['preparation']['path'], prep), 'Workflow commands or allocation changed')
    require(plan['aggregator_sources']=={name:sha(path) for name,path in local_dependencies([out/'common/compare_full_vessel_stage.py']).items()}, 'Aggregator source closure changed')
    evidence = authenticated_gate(ledger, plan['bridge']['path'], bridge, smc) if gate else None
    ledger.recheck(); return plan, prep, ledger, evidence


def worker_counts(repository):
    """Same-user scientific jobs in this workspace; OS/browser threads are excluded."""
    repository = Path(repository).resolve(); physical, workers = [], 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            if proc.stat().st_uid != os.getuid(): continue
            fields = (proc/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0] == 'Z': continue
            executable = Path(os.readlink(proc/'exe').removesuffix(' (deleted)'))
            argv = [os.fsdecode(a) for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
            is_physical = executable.name.removesuffix(' (deleted)') in PHYSICAL_NAMES
            scientific = is_physical or ('python' in executable.name and any(str(repository)+'/' in a or 'protein-nucleation' in a for a in argv[1:]))
            if scientific: workers += int(fields[17])  # Linux num_threads.
            if is_physical: physical.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError): continue
    return dict(physical_pids=physical, workers=workers)


def execute_group(steps, snapshot, workers, repository, env, popen=subprocess.Popen,
                  pause=time.sleep, capacity=worker_counts, birth=process_token, before_launch=lambda:None):
    """Poll before refilling, stop launches on failure, and drain every started child."""
    require(1 <= workers <= 8, 'Invalid child budget')
    active, cursor, error = [], 0, None
    try:
        while cursor < len(steps) or active:
            for item, child in list(active):
                code = child.poll()
                if code is None: continue
                child.wait(); active.remove((item,child))
                item.update(status='complete' if code == 0 else 'failed',returncode=code,finished=time.time()); snapshot()
                if code != 0: error = RuntimeError('Child failed; no further launches or retries')
            if error is not None: break
            if cursor < len(steps) and len(active) < workers:
                count = capacity(repository); item = steps[cursor]
                available = count['workers'] < 32 and (item['kind'] != 'physical' or len(count['physical_pids']) < 8)
                if available:
                    before_launch()
                    count=capacity(repository)
                    if count['workers']>=32 or (item['kind']=='physical' and len(count['physical_pids'])>=8):
                        pause(.1); continue
                    require(not Path(item['directory']).exists(), 'Existing child output; no overwrite')
                    with Path(item['log']).open('xb') as stream:
                        child = popen(item['command'], cwd=repository, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    active.append((item,child)); cursor += 1
                    item.update(status='running',pid=child.pid,process_birth=birth(child.pid),started=time.time()); snapshot()
                    continue  # Poll all children before the next launch.
            if cursor < len(steps) or active: pause(.1)
    except BaseException as exc: error = exc
    finally:
        for item,child in active:
            while True:
                try: code=child.wait(); break
                except (KeyboardInterrupt, InterruptedError) as exc:
                    if error is None: error=exc
            item.update(status='complete' if code == 0 else 'failed',returncode=code,finished=time.time())
        for item in steps:
            if item['status'] == 'pending': item['status']='not_started'
        snapshot()
    if error is not None: raise error


def verify_step(step, prep, ledger, stage, workflow):
    root = Path(step['directory']); result = {}
    if step['kind'] == 'physical':
        manifest=read(ledger.bind(root/'manifest.json')); summary=read(ledger.bind(root/'summary.json'))
        job=next(j for j in prep['jobs'] if j['id']==step['job_id'])
        require(summary['complete'] is True and summary['manifest']==manifest and summary['samples']==job['samples'], 'Incomplete physical population')
        required=dict(samples=job['samples'],seed=job['seed'],cloud_replicates=2,activity=.035)
        required.update(executable_sha256=prep['binary_sha256'],source_bundle_sha256=prep['source_bundle_sha256'],
            config_sha256=prep['input_sha256']['config.json'],model_sha256=prep['input_sha256']['model.json'],shape_sha256=prep['input_sha256']['shape.json'])
        require(all(manifest.get(k)==v for k,v in required.items()), 'Physical executable, target, seed or attempted allocation differs')
        ledger.bind(root/'samples.jsonl')  # The independent audit checks every attempt and full proposal law.
        for name in ('input-config.json','model.json','shape.json','source-bundle.json'):
            key={'input-config.json':'config_sha256','model.json':'model_sha256','shape.json':'shape_sha256','source-bundle.json':'source_bundle_sha256'}[name]
            ledger.bind(root/'provenance'/name,manifest[key])
    else:
        ledger.frozen(root); value=read(ledger.bind(root/'analysis.json'))
        require(value['complete'] is True, 'Incomplete audit, partition or stage aggregation')
        expected={'audit':('full-vessel-baseline-audit-v1','full-vessel-latent-audit-v1'),
            'partition':('full-vessel-contact-partition-v1',),'aggregate':('full-vessel-stage-comparison-v1',)}[step['kind']]
        require(value['schema'] in expected, 'Unexpected analysis result schema')
        if step['kind']=='aggregate':
            expected_sources={str(Path(workflow['stages'][0]['groups'][-1]['steps'][0]['command'][3]).parent/name):digest
                for name,digest in workflow['aggregator_sources'].items()}
            require(value['stage']==stage and value['preparation']==workflow['preparation']['path']
                and value['preparation_sha256']==workflow['preparation']['sha256']
                and value['source_sha256']==expected_sources, 'Aggregation stage, preparation or source closure differs')
            for path,digest in value['input_sha256'].items(): ledger.bind(path,digest)
            for path,digest in value['source_sha256'].items(): ledger.bind(path,digest)
        else:
            job=next(j for j in prep['jobs'] if j['id']==step['job_id'])
            require(value['population']==job['directory'] and value['manifest']==read(Path(job['directory'])/'manifest.json'), 'Audit or partition belongs to another population')
            native=(value['native_binding']['definition_sha256'] if step['kind']=='audit' else value['native_definition_sha256'])
            require(native==prep['native_definition_sha256'], 'Native classifier changed')
            if step['kind']=='audit':
                require(value['schema']==('full-vessel-baseline-audit-v1' if job['arm']=='vessel' else 'full-vessel-latent-audit-v1'), 'Audit arm changed')
            else: require(value['audit']==str(Path(job['audit_directory'])/'analysis.json'), 'Partition uses a different audit')
        result['analysis_sha256']=sha(root/'analysis.json')
    result['output_sha256']=ledger.files.copy(); return result


def run(out, expected):
    out=Path(out).resolve()
    with (out/'launch-claim.json').open('x') as stream:
        import json
        json.dump(dict(pid=os.getpid(),process_birth=process_token(os.getpid()),plan_sha256=expected,started=time.time()),stream)
    state=dict(schema=SCHEMA,complete=False,phase='validating',pid=os.getpid(),process_birth=process_token(os.getpid()),
        plan_sha256=expected,started=time.time(),groups=[],physics_launched=0)
    def snapshot():
        state['physics_launched'] = sum(item['kind']=='physical' and 'pid' in item
            for group in state['groups'] for item in group['steps'])
        write(out/'status.tmp',state); (out/'status.tmp').replace(out/'status.json')
    with (out/'status.json').open('x') as stream: stream.write('{}\n')
    snapshot()
    def interrupted(signum, frame): raise InterruptedError('Controller received signal '+str(signum))
    previous_term=signal.signal(signal.SIGTERM,interrupted)
    try:
        plan,prep,ledger,evidence=validate(out,expected)
        state['gate']=evidence
        for stage in plan['stages']:
            for group in stage['groups']:
                for item in group['steps']:
                    require(not Path(item['directory']).exists() and not Path(item['log']).exists(), 'Existing output or log; no retries')
        # Different workflow directories cannot race to consume the same allocation.
        with (Path(plan['preparation']['path'])/'execution-claim.json').open('x') as stream:
            import json
            json.dump(dict(workflow=str(out),plan_sha256=expected,pid=os.getpid(),process_birth=process_token(os.getpid())),stream)
        env=dict(os.environ,**THREADS,PYTHONOPTIMIZE='0',PYTHONDONTWRITEBYTECODE='1')
        executable_inputs=Ledger(); completed_outputs=Ledger()
        # Recheck executable/input bytes per launch, but large completed raw records
        # at stage boundaries. Every audit also verifies its actual input records.
        prep_root=Path(plan['preparation']['path'])
        for path,digest in ledger.files.items():
            candidate=Path(path)
            if candidate.is_relative_to(out/'common') or candidate.is_relative_to(prep_root/'common') or candidate.is_relative_to(prep_root/'inputs'):
                executable_inputs.bind(path,digest)
        for stage in plan['stages']:
            ledger.recheck(); completed_outputs.recheck()
            for group in stage['groups']:
                items=[dict(copy.deepcopy(s),status='pending',returncode=None) for s in group['steps']]
                state['groups'].append(dict(stage=stage['name'],kind=group['kind'],steps=items))
                state['phase']=stage['name']+'-'+group['kind']; snapshot()
                execute_group(items,snapshot,group['workers'],plan['repository'],env,before_launch=executable_inputs.recheck)
                for item in items:
                    outputs=Ledger(); item.update(verify_step(item,prep,outputs,stage['name'],plan))
                    for path,digest in outputs.files.items(): completed_outputs.bind(path,digest)
                    item['log_sha256']=sha(item['log'])
                snapshot()
            # Stage diagnostics may fail; fixed larger allocation remains required.
            # Successful execution/complete accounting, not statistical pass flags, gates it.
        ledger.recheck(); completed_outputs.recheck(); state.update(complete=True,phase='complete',finished=time.time()); snapshot()
    except BaseException as error:
        state.update(complete=False,phase='failed',error=repr(error),finished=time.time()); snapshot(); raise
    finally: signal.signal(signal.SIGTERM,previous_term)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze'); f.add_argument('--out',type=Path,required=True)
    for name in ('preparation','release','bridge'):
        f.add_argument('--'+name,type=Path,required=True); f.add_argument('--expected-'+name+'-sha256',required=True)
    for name in ('preflight','run'):
        p=sub.add_parser(name); p.add_argument('--out',type=Path,required=True); p.add_argument('--expected-plan-sha256',required=True)
    a=parser.parse_args()
    if a.action=='freeze':
        freeze(a.out,a.preparation,a.expected_preparation_sha256,a.release,a.expected_release_sha256,a.bridge,a.expected_bridge_sha256)
        print(dict(plan_sha256=sha(a.out/'plan.json'),physics_launched=False),flush=True)
    elif a.action=='preflight':
        _,_,_,gate=validate(a.out,a.expected_plan_sha256); print(dict(eligible=True,gate=gate,physics_launched=False),flush=True)
    else: print(run(a.out,a.expected_plan_sha256)['phase'],flush=True)


if __name__=='__main__':main()
