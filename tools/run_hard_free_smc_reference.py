#!/usr/bin/env python3
"""Freeze or explicitly execute the eleven one-shot sphere/Haar SMC references.

One child at a time; no capacity waiting, retries, refill, classifier or extension.
These are software references, not protein, mixing or assembly evidence.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'RAYON_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
os.environ.update(THREAD_ENV)
from prepare_hard_free_line_score import read, require, sha, write
from prepare_hard_free_line_sensitivity_reference import PYTHON, capacity, runtime
from prepare_hard_free_smc_reference import jobs
from prepare_shoulder_docking_benchmark import local_dependencies
from run_smc_importance_bridge import process_token

PREPARATION_SHA = '2e63ba15b3b97ca42d9d1acd802a1341817d26f7b6625346aa5123bddad1b65d'
SCOPE = 'Sphere/Haar software references only; no protein, mixing-speedup or finite-assembly evidence.'


def verify(directory, expected):
    require(read(directory/'freeze.json')['files'] == expected, 'Frozen file catalog differs')
    for name, digest in expected.items():
        path = (directory/name).resolve()
        require(path.is_relative_to(directory) and sha(path) == digest, 'Frozen input changed: '+name)


def preparation(directory):
    require(sha(directory/'plan.json') == PREPARATION_SHA, 'Different fixed reference allocation')
    plan = read(directory/'plan.json')
    verify(directory, dict(plan['files_sha256'], **{'plan.json': PREPARATION_SHA}))
    require(plan['jobs'] == jobs(directory), 'Job arguments differ from allocation')
    require(runtime() == plan['runtime'], 'Pinned reference runtime changed')
    return plan


def prepare(directory, out):
    directory, out = Path(directory).resolve(), Path(out).resolve()
    plan = preparation(directory)
    require(not out.exists(), 'Fresh execution freeze required')
    sources = local_dependencies([Path(__file__), Path(__file__).with_name('test_run_hard_free_smc_reference.py')])
    (out/'source').mkdir(parents=True)
    for name, source in sources.items(): shutil.copy2(source, out/'source'/name)
    frozen = {str(p.relative_to(out)): sha(p) for p in sorted((out/'source').iterdir())}
    execution = dict(schema='hard-free-smc-reference-execution-v1', preparation=str(directory),
        preparation_plan_sha256=PREPARATION_SHA, preparation_freeze_sha256=sha(directory/'freeze.json'),
        runtime=plan['runtime'], files_sha256=frozen, physical_workers=1, audit_workers=1,
        no_retries=True, no_capacity_wait=True, physical_gate_open=False, scope=SCOPE)
    write(out/'execution-plan.json', execution)
    write(out/'freeze.json', dict(files=dict(frozen, **{'execution-plan.json': sha(out/'execution-plan.json')})))
    return dict(prepared=True, launched=False, execution=str(out), plan_sha256=sha(out/'execution-plan.json'))


def validate(execution, expected):
    require(sha(execution/'execution-plan.json') == expected, 'Execution plan hash differs')
    ep = read(execution/'execution-plan.json'); directory = Path(ep['preparation'])
    verify(execution, dict(ep['files_sha256'], **{'execution-plan.json': expected}))
    require(Path(__file__).resolve() == execution/'source'/Path(__file__).name, 'Run the frozen executor source')
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and 'source/'+Path(filename).name in ep['files_sha256']:
            require(Path(filename).resolve() == execution/'source'/Path(filename).name, 'Unfrozen local import')
    require(ep['preparation_plan_sha256'] == PREPARATION_SHA
        and sha(directory/'freeze.json') == ep['preparation_freeze_sha256'], 'Preparation binding differs')
    plan = preparation(directory)
    require(runtime() == ep['runtime'] == plan['runtime'], 'Execution runtime changed')
    return ep, plan


def population_masses(summary):
    if summary['zero_estimate']:
        require(summary['log_Z'] is None and not summary['terminal_particles'], 'Invalid completed zero')
        return dict(total=0., contact=0., unbound=0.)
    total = math.exp(summary['log_Z']); particles = summary['terminal_particles']
    require(math.isfinite(total) and total > 0 and particles, 'Invalid terminal mass')
    contact = sum(math.dist(p['pose']['position'], [0., 0., 0.]) < 1.4 for p in particles)
    return dict(total=total, contact=total*contact/len(particles), unbound=total*(len(particles)-contact)/len(particles))


def check_mean(values, exact):
    require(len(values) == 4 and all(math.isfinite(v) and v >= 0 for v in values), 'Four independent masses required')
    mean = math.fsum(values)/4; se = math.sqrt(math.fsum((v-mean)**2 for v in values)/12)
    return dict(population_masses=values, mean=mean, standard_error=se, reference=exact,
        absolute_difference=abs(mean-exact), tolerance=6*se+1e-6, passed=abs(mean-exact) <= 6*se+1e-6)


def check_output(directory, plan, job, audit_path=None):
    root = Path(job['directory']); inputs = directory/'inputs'/job['fixture']; files = {}
    def bind(path, digest=None):
        value = sha(path); require(digest is None or value == digest, 'Output binding differs: '+str(path))
        files[str(path)] = value
        return read(path) if path.suffix == '.json' else None
    state = bind(root/'status.json'); summary = bind(root/'summary.json', state['summary_sha256'])
    manifest = bind(root/'manifest.json', summary['manifest_sha256'])
    suffix = '-hard-free-initial-guide' if job['guide'] else ''
    require(state['complete'] is True and state['phase'] == 'complete' and summary['complete'] is True
        and summary['schema'] == 'latent-region-smc'+suffix+'-summary-v1'
        and manifest['schema'] == 'latent-region-smc'+suffix+'-v1', 'Incomplete/wrong SMC output')
    expected = dict(config=str(inputs/'config.json'), region=str(inputs/'region.json'), out=str(root),
        initial_reference_region=None, initial_current_probability=1., initial_draws=job['initial_draws'],
        population=job['population'], seed=job['seed'], bridge='proposal_density',
        schedule=[i/job['stages'] for i in range(job['stages']+1)], sweeps_per_stage=job['sweeps'],
        cloud_replicates=plan['cloud_replicates'], lambda_ratio=plan['lambda_ratio'])
    require(manifest['options'] == expected, 'Output options differ from fixed job argv')
    require(manifest['executable_sha256'] == plan['binary_sha256']
        and manifest['source_bundle_sha256'] == plan['source_bundle_sha256'], 'Output build differs')
    for name in ['config', 'region', 'shape', 'source-bundle']:
        original = directory/'common/source-bundle.json' if name == 'source-bundle' else inputs/(name+'.json')
        bind(root/'provenance'/(name+'.json'), sha(original))
        require(manifest[name.replace('-', '_')+'_sha256'] == sha(original), 'Manifest input hash differs')
    if job['guide']:
        guide = inputs/job['guide']; bind(root/'provenance/initial-guide.json', sha(guide))
        require(manifest['initial_guide']['path'] == str(guide)
            and manifest['initial_guide']['sha256'] == sha(guide), 'Output guide differs')
    else: require('initial_guide' not in manifest, 'Unexpected guided legacy control')
    for name in ['initialization', 'stages']+(['attempts'] if job['guide'] else []):
        bind(root/(name+'.jsonl'), summary[name+'_sha256'])
    require(summary['initial_draws'] == job['initial_draws']
        and type(summary['initial_hits']) is int and 0 <= summary['initial_hits'] <= job['initial_draws']
        and state['completed_stage'] == summary['completed_stage'] == (0 if summary['zero_estimate'] else job['stages'])
        and len(summary['terminal_particles']) == (0 if summary['zero_estimate'] else job['population']), 'Output allocation incomplete')
    if job['purpose'] == 'zero-hit':
        require(summary['zero_estimate'] is True and summary['initial_hits'] == 0, 'Empty hard target was refilled')
    if audit_path is not None:
        audit = bind(audit_path); ledger = audit['input_sha256']
        require(audit['schema'] == 'hard-free-initial-guide-smc-independent-audit-v1'
            and audit['complete'] is True and audit['directory'] == str(root)
            and audit['initial_draws'] == summary['initial_draws'] and audit['initial_hits'] == summary['initial_hits']
            and audit['completed_stage'] == summary['completed_stage']
            and (audit['log_Z'] == summary['log_Z'] or (audit['log_Z'] is not None and summary['log_Z'] is not None
                and abs(audit['log_Z']-summary['log_Z']) <= 2e-10))
            and audit['terminal_particles'] == len(summary['terminal_particles']), 'Audit receipt differs')
        required = dict(files); required.pop(str(audit_path))
        required[str(directory/'common/latent-region-smc')] = plan['binary_sha256']
        for source in local_dependencies([directory/'common/audit_hard_free_smc.py']).values():
            required[str(source)] = plan['files_sha256']['common/'+source.name]
        require(all(ledger.get(p) == digest for p, digest in required.items()), 'Audit omits required bound input/source')
        for path, digest in ledger.items(): bind(Path(path), digest)
    return summary, dict(job=job['id'], input_sha256=files, masses=population_masses(summary),
        independently_audited=audit_path is not None)


def parity_record(left, right, path=''):
    ignored = {'guide_density', 'guide_density_cache', 'initial_guide_draw', 'hard_free_line_draw',
        'proposal_component', 'selected_initial_chart', 'proposed_guide_density'}
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left)-ignored == set(right)-ignored, 'Parity keys differ: '+path)
        for key in set(left)-ignored:
            if key in ('pose', 'old_pose', 'proposed_pose'):
                require(left[key] == right[key], 'Parity pose differs: '+path+'/'+key)
            else: parity_record(left[key], right[key], path+'/'+key)
    elif isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), 'Parity length differs: '+path)
        for a, b in zip(left, right): parity_record(a, b, path)
    elif type(left) is float and type(right) is float:
        tolerance = 2e-8 if 'latent' in path and 'log_' not in path else (
            2e-10 if 'log_' in path or 'jacobian' in path or 'ESS' in path else 0.)
        require(math.isfinite(left) and math.isfinite(right) and abs(left-right) <= tolerance, 'Parity scalar differs: '+path)
    else: require(type(left) is type(right) and left == right, 'Parity exact field differs: '+path)


def uniform_parity(directory):
    roots = [directory/'populations'/name for name in ['uniform-guided', 'uniform-legacy']]
    for name in ['initialization.jsonl', 'stages.jsonl']:
        rows = [[json.loads(line) for line in (root/name).read_text().splitlines()] for root in roots]
        parity_record(*rows, path=name)
    summaries = [read(root/'summary.json') for root in roots]
    keys = ['zero_estimate', 'log_Z', 'initial_draws', 'initial_hits', 'completed_stage',
        'terminal_particles', 'ancestry', 'initial_weight_ESS']
    parity_record(*[{k: s.get(k) for k in keys} for s in summaries], path='summary')
    return dict(passed=True, scalar_tolerance=2e-10, latent_tolerance=2e-8,
        exact='Generated poses, parents, gate/cloud counts, accepted decisions and retained endpoints.',
        scope='Same-stream implementation control; not independent evidence.')


def summarize(execution, directory, plan, state):
    receipts = {}
    for job in plan['jobs']:
        _, receipt = check_output(directory, plan, job, execution/(job['id']+'-audit.json') if job['guide'] else None)
        receipts[job['id']] = receipt
    means = {str(z): {region: check_mean([receipts[j['id']]['masses'][region]
        for j in plan['jobs'] if j['purpose'] == 'analytic' and j['activity'] == z], exact)
        for region, exact in plan['exact'][str(z)].items()} for z in [0, 4]}
    result = dict(schema='hard-free-smc-reference-results-v1', complete=True, means=means,
        analytic_passed=all(c['passed'] for group in means.values() for c in group.values()),
        uniform_parity=uniform_parity(directory), populations=receipts, execution_plan_sha256=sha(execution/'execution-plan.json'),
        preparation_plan_sha256=PREPARATION_SHA, runtime=runtime(), physical_gate_open=False, scope=SCOPE,
        uncertainty='SE across four independent population masses per activity, including unconditional completed zeros; '
            'terminal descendants are not IID replicates. All six fixed analytic comparisons retained.')
    write(execution/'analysis.json', result)
    return result


def run(execution, expected):
    execution = Path(execution).resolve(); ep, plan = validate(execution, expected); directory = Path(ep['preparation'])
    def ready():
        counts = capacity()
        require(counts['private_pid_namespace'] is False, 'Host capacity unavailable in private PID namespace')
        require(len(counts['physical_pids']) < 8 and counts['workers'] < 32, 'Capacity unavailable; no wait or launch')
        return counts
    for job in plan['jobs']:
        require(not Path(job['directory']).exists(), 'Existing physical output; no retry')
        for suffix in ['-audit.json', '-audit.log', '-physical.log']:
            require(not (execution/(job['id']+suffix)).exists(), 'Existing child receipt/log; no retry')
    require(not (execution/'status.json').exists(), 'Existing execution status; no retry')
    require(not (execution/'analysis.json').exists(), 'Existing reference analysis; no retry')
    counts = ready()
    write(directory/'reference-launch-claim.json', dict(execution=str(execution), plan_sha256=expected,
        pid=os.getpid(), time=time.time(), capacity=counts))
    state = dict(complete=False, phase='claimed', execution_plan_sha256=expected, jobs=[])
    def save():
        temporary = execution/'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n'); temporary.replace(execution/'status.json')
    stopping = [False]
    def interrupted(signum, frame): stopping[0] = True
    handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    child = None; record = None
    env = dict(os.environ, PYTHONOPTIMIZE='0', **THREAD_ENV)
    save()
    try:
        for job in plan['jobs']:
            record = dict(id=job['id'], steps=[]); state['jobs'].append(record)
            audit_path = execution/(job['id']+'-audit.json')
            steps = [('physical', job['argv'])]+([('audit', [PYTHON, str(directory/'common/audit_hard_free_smc.py'),
                '--directory', job['directory'], '--binary', str(directory/'common/latent-region-smc'),
                '--out', str(audit_path)])] if job['guide'] else [])
            for kind, argv in steps:
                validate(execution, expected); require(runtime() == ep['runtime'], 'Runtime changed before launch')
                observed = ready(); step = dict(kind=kind, argv=argv, phase='starting', capacity=observed, start=time.time())
                record['steps'].append(step); state['phase'] = kind; save()
                require(not stopping[0], 'Interrupted; stop subsequent launches')
                with (execution/(job['id']+'-'+kind+'.log')).open('xb') as log:
                    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                    step.update(pid=child.pid, start_ticks=process_token(child.pid), phase='running'); save()
                    code = child.wait(); child = None
                step.update(returncode=code, phase='complete' if code == 0 else 'failed', end=time.time()); save()
                require(code == 0, 'Child failed; no retries or subsequent launches')
                require(not stopping[0], 'Interrupted; active child finished, stop subsequent launches')
                _, receipt = check_output(directory, plan, job, audit_path if kind == 'audit' else None)
                record['receipt'] = receipt; save()
            if job['id'] == 'uniform-legacy': state['uniform_parity'] = uniform_parity(directory)
        validate(execution, expected)
        result = summarize(execution, directory, plan, state)
        state.update(complete=True, phase='complete', analysis_sha256=sha(execution/'analysis.json'),
            analytic_passed=result['analytic_passed']); save()
        return result
    except BaseException as error:
        for sig in handlers: signal.signal(sig, signal.SIG_IGN)
        if child is not None:
            code = child.wait()
            record['steps'][-1].update(returncode=code, phase='drained', end=time.time())
        state.update(phase='failed', error=str(error)); save()
        raise
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--preparation', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--execution', type=Path, required=True); p.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    result = prepare(args.preparation, args.out) if args.command == 'prepare' else run(args.execution, args.expected_plan_sha256)
    print(json.dumps({k: result[k] for k in ('prepared', 'launched', 'execution', 'plan_sha256', 'complete', 'analytic_passed') if k in result}))
