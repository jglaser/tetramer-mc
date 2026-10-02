#!/usr/bin/env python3
"""Fixed four-population protein SMC bridge control with independent audits.

Only the explicit run command launches work. Completed IID prerequisites may
fail scientific convergence; such failures are retained, never waived here.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from analyze_r4_smc_control import Ledger, read, require, sha, write, validate_control
from prepare_hard_free_line_sensitivity_reference import PYTHON, runtime, capacity
from prepare_shoulder_docking_benchmark import local_dependencies
from run_hard_free_line_physical_pilot import inventory
from prepare_hard_free_line_fresh import ROOTS
from run_full_vessel_comparison import execute_group
from run_contact_confirmation import worker_environment

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(610032001+i for i in range(4))
REFERENCE_RESULT_SHA = 'e0f56afa31bb81bf8524bbd8d43121a96e6694938523eb1be0118d738a7a640f'
BINARY_SHA = '92553e2ed79801811008460c4e725cf73a6a5f21fb785c3e09c42312507103dd'
BROAD_SHA = 'c702ebded0b1e33471e429fcedc6c8cc7b3c5985fb8f1fdb245e801e95ed5729'
ALLOCATION = dict(populations=4, initial_draws=262144, population=2048, stages=128,
    sweeps=4, cloud_replicates=2, lambda_ratio=128.)
ANALYSIS = dict(primary_regions=['registered_native_entry','old_R5_intersection_native','remaining_R4_native'],
    reporting_regions=['total','registered_native_entry','old_R5_intersection_native','remaining_R4_native',
        'contact_no_native_entry','unbound_no_native_entry'],
    profile_stages=list(range(0,129,16)), terminal='All retained particles; normalizer times indicator fraction.',
    initialization='Classify stage-zero retained ancestors only; all unconditional attempts remain independently audited.',
    uncertainty='Four independent population mass means; descendants are not independent replicates.',
    comparisons=['completed historical broad SMC','completed larger IID baseline','completed larger IID conditioned'],
    absolute_log_mass_difference_max=.2, independent_linear_difference_SE_multiplier=3.,
    zero_terminal_count='Unresolved mass, never a physical zero, upper bound or epsilon substitute.',
    pooling=False, classifier_replay=False)
SCOPE = ('Single-factor initialization/bridge control against the completed broad local-mutation '
    'SMC control. Fixed R4 physical endpoint, unchanged local steps and SMC allocation. '
    'Implementation audits precede once-only contact classification and population comparison. '
    'No full-vessel or assembly gate is opened by completion; failed IID diagnostics remain failed.')


def expected_options(out, seed):
    return dict(config=str(out/'common/config.json'), region=str(out/'common/region.json'),
        out=str(out/'populations'/f'r{SEEDS.index(seed):02}'), initial_reference_region=None,
        initial_current_probability=1., initial_draws=ALLOCATION['initial_draws'],
        population=ALLOCATION['population'], seed=seed, bridge='proposal_density',
        schedule=[i/ALLOCATION['stages'] for i in range(ALLOCATION['stages']+1)],
        sweeps_per_stage=ALLOCATION['sweeps'], cloud_replicates=2, lambda_ratio=128.)


def jobs(out):
    values = []
    for i, seed in enumerate(SEEDS):
        directory = out/'populations'/f'r{i:02}'
        command = [str(out/'common/latent-region-smc'), '--config', str(out/'common/config.json'),
            '--region', str(out/'common/region.json'), '--initial-guide', str(out/'common/guide.json'),
            '--out', str(directory), '--seed', str(seed), '--bridge', 'proposal-density',
            '--initial-draws', '262144', '--population', '2048', '--stages', '128',
            '--sweeps-per-stage', '4', '--cloud-replicates', '2', '--lambda-ratio', '128']
        values.append(dict(id=f'r{i:02}', seed=seed, kind='physical', directory=str(directory),
            log=str(out/f'r{i:02}-physical.log'), command=command, status='pending'))
    return values


def completed_iid(root, comparison_name, expected_protocol):
    """Read only complete evidence; no convergence flag is promoted to a pass."""
    ledger = Ledger(); state = read(ledger.bind(root/'status.json'))
    require(state['complete'] is True and state['phase'] == 'complete', 'Wait for completed IID controls')
    require(state['protocol_sha256'] == expected_protocol, 'Different IID allocation')
    protocol = read(ledger.bind(root/'protocol.json', expected_protocol))
    require(len(state['jobs']) == len(state['audits']) == 8 and all(j['status'] == 'complete'
        and j['returncode'] == 0 for j in state['jobs']+state['audits']), 'Incomplete IID jobs/audits')
    summary = read(ledger.bind(root/'comparison/analysis.json', state['comparison_sha256']))
    require(summary['complete'] is True and summary['protocol_sha256'] == expected_protocol, 'IID summary binding differs')
    comparison = read(ledger.bind(root/(comparison_name+'.json'), state[comparison_name.replace('-', '_')+'_sha256']))
    require(comparison['complete'] and not comparison['full_vessel_gate_open'] and not comparison['assembly_gate_open'],
        'IID comparison inference scope differs')
    return protocol, ledger.files


def prepare(out, larger, sensitivity):
    out, larger, sensitivity = [Path(p).resolve() for p in (out, larger, sensitivity)]
    require(not out.exists(), 'Fresh preparation required; no overwrite')
    lp, left = completed_iid(larger, 'population-size-comparison',
        'cbcf1400a38365f2fa24585110da389b64174fc7e3a563e26cf1618a161c588e')
    _, right = completed_iid(sensitivity, 'sensitivity-comparison',
        '100fdb3b68b2a1bfdf9de6fdce61a7759894d0818f7affa6a9b9e7d0ba977cc6')
    ledger = Ledger()
    for path, digest in dict(left, **right).items(): ledger.bind(path, digest)
    refs = ROOT/'results/hard-free-smc-reference-execution-20261001/analysis.json'
    ref = read(ledger.bind(refs, REFERENCE_RESULT_SHA))
    require(ref['complete'] and ref['analytic_passed'] and ref['uniform_parity']['passed'], 'SMC references failed')
    broad_path = ROOT/'runs/smc-r4-density-bridge-control-20260922/protocol.json'
    broad = validate_control(broad_path, BROAD_SHA)
    for path, digest in broad['bindings'].items(): ledger.bind(path, digest)
    historical = ROOT/'runs/smc-r4-controls-workflow-20260922/broad-analysis/analysis.json'
    evidence = read(ledger.bind(historical))
    require(evidence['complete'] is True, 'Historical matching control incomplete')
    binary = ROOT/'results/hard-free-smc-reference-preparation-20261001/common/latent-region-smc'
    bundle = binary.with_name('source-bundle.json')
    ledger.bind(binary, BINARY_SHA); ledger.bind(bundle)
    require(bundle.read_bytes() in binary.read_bytes(), 'Source bundle not embedded in validated binary')
    guide_path = larger/'common/conditioned.json'
    guide = read(ledger.bind(guide_path))
    region_path = Path(broad['protocol']['physical_target']['region'])
    config_path = Path(broad['protocol']['physical_target']['config'])
    shape_path = Path(broad['config']['shape'])
    require(guide['schema'] == 'defensive-hard-free-line-guide-v1'
        and guide['defensive_uniform_shell_probability'] == .5 and guide['conditional_probability'] == 1.
        and guide['raw_translation_axes'] == [0,1,2] and len(guide['gaussian_components']) == 92
        and guide['region_sha256'] == sha(region_path) == lp['region_sha256']
        and sha(shape_path) == lp['shape_sha256'], 'Guide/target identity differs')
    cfg = broad['config']
    require(cfg['reservoir_density'] == .035 and cfg['depletant_radius'] == 1.5
        and cfg['translation_steps'] == [.2,2.] and cfg['rotation_steps_deg'] == [1.5,15.], 'Matching physical/mutation settings differ')
    sources = local_dependencies([Path(__file__), ROOT/'tools/audit_hard_free_smc.py',
        ROOT/'tools/test_run_hard_free_protein_smc.py'])
    for path in sources.values(): ledger.bind(path)
    roots = list(dict.fromkeys([str(ROOT/'runs'), str(ROOT/'results'), *ROOTS]))
    seed_inventory = inventory(roots, SEEDS)
    (out/'common').mkdir(parents=True)
    for src, name in [(binary,'latent-region-smc'),(bundle,'source-bundle.json'),(config_path,'config.json'),
        (region_path,'region.json'),(shape_path,'shape.json'),(guide_path,'guide.json')]: shutil.copy2(src, out/'common'/name)
    for name, src in sources.items(): shutil.copy2(src, out/'common'/name)
    write(out/'seed-inventory.json', seed_inventory)
    ledger.recheck()
    plan = dict(schema='hard-free-protein-smc-control-v1', allocation=ALLOCATION, analysis=ANALYSIS, seeds=list(SEEDS),
        repository=str(ROOT), python=PYTHON, runtime=runtime(), preparation_input_sha256=ledger.files,
        # The byte-identical config retains this absolute shape path. Everything
        # else executed is copied into common; later repository development
        # must not invalidate already frozen worker source files.
        external_required_sha256={str(shape_path):sha(shape_path)},
        binary_sha256=BINARY_SHA, source_bundle_sha256=sha(bundle), jobs=jobs(out),
        reference_results_sha256=REFERENCE_RESULT_SHA, historical_protocol=str(broad_path),
        historical_protocol_sha256=BROAD_SHA, historical_analysis=str(historical),
        guide_sha256=sha(guide_path), physical_workers=2, audit_workers=4, all_physical_cap=8,
        all_worker_cap=32, no_retries=True, scope=SCOPE, full_vessel_gate_open=False, assembly_gate_open=False,
        expected_options=[expected_options(out,s) for s in SEEDS],
        files_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    write(out/'plan.json', plan)
    write(out/'freeze.json', dict(files=dict(plan['files_sha256'], **{'plan.json':sha(out/'plan.json')})))
    return dict(prepared=True, launched=False, plan_sha256=sha(out/'plan.json'))


def validate(out, expected):
    require(sha(out/'plan.json') == expected, 'Preparation plan changed')
    plan = read(out/'plan.json'); ledger = Ledger(); ledger.frozen(out)
    require(plan['allocation'] == ALLOCATION and plan['analysis'] == ANALYSIS
        and plan['seeds'] == list(SEEDS) and plan['jobs'] == jobs(out)
        and plan['expected_options'] == [expected_options(out,s) for s in SEEDS], 'Allocation/options changed')
    require(plan['binary_sha256'] == BINARY_SHA and plan['physical_workers'] == 2 and plan['audit_workers'] == 4
        and not plan['full_vessel_gate_open'] and not plan['assembly_gate_open'], 'Execution scope differs')
    for path, digest in plan['external_required_sha256'].items(): ledger.bind(path, digest)
    require(runtime() == plan['runtime'], 'Pinned Python runtime changed')
    return plan


def check_output(out, plan, job, with_audit=False):
    root = Path(job['directory']); ledger = Ledger()
    state = read(ledger.bind(root/'status.json'))
    require(state['complete'] is True and state['phase'] == 'complete', 'SMC output incomplete')
    summary = read(ledger.bind(root/'summary.json', state['summary_sha256']))
    manifest = read(ledger.bind(root/'manifest.json', summary['manifest_sha256']))
    require(summary['complete'] is True and manifest['schema'] == 'latent-region-smc-hard-free-initial-guide-v1'
        and summary['schema'] == 'latent-region-smc-hard-free-initial-guide-summary-v1'
        and manifest['options'] == expected_options(out,job['seed']), 'Wrong SMC law/allocation')
    require(manifest['executable_sha256'] == BINARY_SHA and manifest['source_bundle_sha256'] == plan['source_bundle_sha256']
        and manifest['initial_guide']['sha256'] == plan['guide_sha256']
        and manifest['initial_guide']['path'] == str(out/'common/guide.json'), 'SMC binary/guide differs')
    for name in ('config','region','shape','source-bundle','initial-guide'):
        source = out/'common'/('guide.json' if name == 'initial-guide' else name+'.json')
        ledger.bind(root/'provenance'/(name+'.json'), sha(source))
    for name in ('initialization','stages','attempts'):
        ledger.bind(root/(name+'.jsonl'),summary[name+'_sha256'])
    require(summary['initial_draws'] == ALLOCATION['initial_draws'] and
        state['completed_stage'] == summary['completed_stage'] == (0 if summary['zero_estimate'] else 128)
        and len(summary['terminal_particles']) == (0 if summary['zero_estimate'] else 2048), 'Incomplete attempted allocation')
    if with_audit:
        audit = read(out/(job['id']+'-audit.json'))
        require(audit['complete'] and audit['schema'] == 'hard-free-initial-guide-smc-independent-audit-v1'
            and audit['directory'] == str(root) and audit['initial_draws'] == summary['initial_draws']
            and audit['initial_hits'] == summary['initial_hits']
            and (audit['log_Z'] == summary['log_Z'] or (audit['log_Z'] is not None and summary['log_Z'] is not None
                and abs(audit['log_Z']-summary['log_Z']) <= 2e-10)), 'Independent audit differs')
        required = dict(ledger.files)
        required[str(out/'common/latent-region-smc')] = BINARY_SHA
        for source in local_dependencies([out/'common/audit_hard_free_smc.py']).values(): required[str(source)] = sha(source)
        require(all(audit['input_sha256'].get(p) == h for p,h in required.items()), 'Audit omits required bindings')
        for path,digest in audit['input_sha256'].items(): ledger.bind(path,digest)
        ledger.bind(out/(job['id']+'-audit.json'))
    return dict(input_sha256=ledger.files, initial_draws=summary['initial_draws'], initial_hits=summary['initial_hits'])


def run(out, expected):
    out = Path(out).resolve(); plan = validate(out,expected)
    require(Path(__file__).resolve() == out/'common'/Path(__file__).name, 'Use frozen controller')
    for module in tuple(sys.modules.values()):
        filename = getattr(module,'__file__',None)
        if filename and 'common/'+Path(filename).name in plan['files_sha256']:
            require(Path(filename).resolve() == out/'common'/Path(filename).name, 'Unfrozen local import')
    require(not (out/'status.json').exists() and all(not Path(j['directory']).exists() for j in plan['jobs']), 'No retry/overwrite')
    def available(repository=None):
        counts = capacity()
        require(not counts['private_pid_namespace'], 'Host process namespace required')
        return counts
    counts = available()
    require(len(counts['physical_pids']) <= 6 and counts['workers'] <= 30, 'Insufficient launch capacity')
    state = dict(complete=False, phase='physical', pid=os.getpid(), started=time.time(),
        plan_sha256=expected, jobs=copy.deepcopy(plan['jobs']), audits=[])
    with (out/'status.json').open('x') as stream: json.dump(state,stream)
    def snapshot():
        write(out/'status.tmp',state); (out/'status.tmp').replace(out/'status.json')
    def interrupted(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN); raise InterruptedError('Drain started children')
    previous = signal.signal(signal.SIGTERM,interrupted)
    try:
        execute_group(state['jobs'],snapshot,2,plan['repository'],worker_environment(),capacity=available,
            before_launch=lambda:validate(out,expected))
        for job in state['jobs']: job['output'] = check_output(out,plan,job)
        state['phase'] = 'independent_audit'; snapshot()
        state['audits'] = [dict(id=j['id'],kind='audit',status='pending',directory=str(out/(j['id']+'-audit.json')),
            log=str(out/(j['id']+'-audit.log')),command=[PYTHON,'-B',str(out/'common/audit_hard_free_smc.py'),
            '--directory',j['directory'],'--binary',str(out/'common/latent-region-smc'),
            '--out',str(out/(j['id']+'-audit.json'))]) for j in plan['jobs']]
        execute_group(state['audits'],snapshot,4,plan['repository'],worker_environment(),capacity=available)
        for job in state['jobs']: job['audited_output'] = check_output(out,plan,job,True)
        validate(out,expected)
        state.update(complete=True,phase='audited_awaiting_classification',finished=time.time(),scope=SCOPE)
        snapshot()
    except BaseException as error:
        state.update(complete=False,phase=state['phase']+'_failed',error=repr(error),finished=time.time()); snapshot(); raise
    finally: signal.signal(signal.SIGTERM,previous)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action',required=True)
    f = sub.add_parser('freeze'); f.add_argument('--out',type=Path,required=True)
    f.add_argument('--larger',type=Path,required=True); f.add_argument('--sensitivity',type=Path,required=True)
    for action in ('validate','run'):
        p = sub.add_parser(action); p.add_argument('--out',type=Path,required=True); p.add_argument('--expected-plan-sha256',required=True)
    args = parser.parse_args()
    value = prepare(args.out,args.larger,args.sensitivity) if args.action == 'freeze' else (
        validate(args.out.resolve(),args.expected_plan_sha256) if args.action == 'validate' else run(args.out,args.expected_plan_sha256))
    print(json.dumps({k:value[k] for k in ('prepared','launched','plan_sha256','phase') if k in value}))
