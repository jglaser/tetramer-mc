#!/usr/bin/env python3
"""Fixed independent R4-minus-native SMC control; never authorize assembly.

All attempted initialization draws and zero populations remain in each estimate.
The native predicate changes this control's target explicitly; original full-R4
importance estimates are compared only through their matching nonnative class.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS')}
os.environ.update(THREAD_ENV)

from analyze_r4_smc_control import Ledger, read, require, sha, validate_config_identity
from compile_native_entry import compile_definition
from prepare_shoulder_docking_benchmark import local_dependencies
from run_full_vessel_comparison import execute_group, process_token
from validate_native_excluded_smc_reference import seed_values

SCHEMA = 'native-excluded-smc-fixed-control-v1'
CLASSES = ('total', 'contact_no_native_entry', 'unbound_no_native_entry')
STRATA = {'radial': 3, 'angular': 3, 'orthant': 64}
ARMS = {'narrow': dict(population=2048, translation_steps=[.05], rotation_steps_deg=[.1]),
        'large': dict(population=4096, translation_steps=[.05], rotation_steps_deg=[.1]),
        'broad': dict(population=2048, translation_steps=[.2, 2.], rotation_steps_deg=[1.5, 15.])}
SEEDS = tuple(149101010 + 1009*i for i in range(12))
INITIAL_DRAWS = 262144
STAGES = 128
SWEEPS = 4
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
REGION_SHA = '924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02'
REFERENCE_SHA = '76ea65088e302d6b6478ac033af9b67b7cf21cb7cea1f854f70cdcd6db0473ae'
DEFINITION_SHA = '5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'
COMPILED_SHA = 'dbb3c3e32259f506b9c979ebb9d1fd773cb78507c1f808fd8dd0ed319e2eade4'
MATCHING_SHA = '9aa835f9bc6bbe807e29d560655dea896bffce5899125e56be8751c7313930d4'
SCOPE = ('Independent fixed-scaffold R4-minus-complete-native control only. '
    'No full-vessel, finite-assembly or bulk-stability authorization. '
    'SMC terminal descendants are correlated; no descendant IID ESS or standard error. '
    'Native zero is structural, not evidence of physical rarity. All-zero estimates '
    'are retained, not retried. Agreement does not bound unseen modes.')


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def jobs_for(out):
    out = Path(out).resolve(); jobs = []
    for i in range(4):
        for arm, spec in ARMS.items():
            ident = f'{arm}-r{i:02d}'; seed = SEEDS[len(jobs)]
            output = out / 'populations' / ident
            argv = [str(out / 'common/latent-region-smc'),
                '--config', str(out / 'inputs' / (arm + '-config.json')),
                '--region', str(out / 'inputs/region.json'),
                '--initial-reference-region', str(out / 'inputs/reference-region.json'),
                '--initial-current-probability', '0.5',
                '--exclude-native-entry', str(out / 'inputs/native-compiled.json'),
                '--out', str(output), '--seed', str(seed), '--bridge', 'proposal-density',
                '--initial-draws', str(INITIAL_DRAWS), '--population', str(spec['population']),
                '--stages', str(STAGES), '--sweeps-per-stage', str(SWEEPS),
                '--cloud-replicates', '2', '--lambda-ratio', '128']
            jobs.append(dict(id=ident, arm=arm, seed=seed, output=str(output), argv=argv))
    return jobs


def allocation():
    particle_stages = 4*sum(a['population'] for a in ARMS.values())*STAGES
    return dict(arms=ARMS, independent_populations_per_arm=4, initial_draws_per_population=INITIAL_DRAWS,
        stages=STAGES, moves_per_particle_per_stage=SWEEPS, cloud_replicates=2, lambda_ratio=128.,
        total_initial_draws=12*INITIAL_DRAWS, potential_particle_evaluations=particle_stages,
        total_mutation_attempts=particle_stages*SWEEPS, initial_current_probability=.5,
        stage_profiles=list(range(0, STAGES+1, 16)))


def validate_reference(receipt_path, expected, ledger):
    path = ledger.bind(receipt_path, expected); value = read(path)
    require(value['schema'] == 'native-excluded-smc-audited-reference-v1'
        and value['complete'] is True and value['reference_tolerance_passed'] is True,
        'Independent reference audit has not passed')
    pops = value['populations']
    wanted = {f'z{z}-r{i:02d}' for z in (0, 4) for i in range(8)} | {'zero'}
    require(len(pops) == 17 and {p['id'] for p in pops} == wanted,
        'All sixteen fresh references and the zero fixture must be audited')
    for entry in pops:
        ledger.bind(entry['audit'], entry['sha256'])
    for source, digest in value['files'].items():
        ledger.bind(source, digest)
    require(value.get('auditor_sources'), 'Missing independently validated auditor source closure')
    return value


def target_check(root, baseline):
    root = Path(root); inputs = root / 'inputs'
    require(sha(inputs / 'shape.json') == SHAPE_SHA and sha(inputs / 'region.json') == REGION_SHA
        and sha(inputs / 'reference-region.json') == REFERENCE_SHA
        and sha(inputs / 'native/definition.json') == DEFINITION_SHA
        and sha(inputs / 'native-compiled.json') == COMPILED_SHA, 'Frozen physical definitions differ')
    current = read(inputs / 'region.json'); reference = read(inputs / 'reference-region.json')
    require(current['mahalanobis_radius'] == 4 and current['minimum_original_q'] == 0
        and current.get('minimum_mahalanobis_radius', 0) == 0 and 'maximum_original_q' not in current,
        'The target must be the complete frozen R4 ball')
    for key in ('physical_fixed_neighbors','capture_center','capture_radius','activity',
                'depletant_radius','physical_metric','shape_sha256'):
        require(current[key] == reference[key], 'R4/reference physical target mismatch: ' + key)
    for arm, spec in ARMS.items():
        config = read(inputs / (arm + '-config.json'))
        validate_config_identity(config, baseline, allowed=('translation_steps', 'rotation_steps_deg'))
        require(config['shape'] == str((inputs / 'shape.json').resolve()), 'Shape path escapes frozen inputs')
        require(config['translation_steps'] == spec['translation_steps']
            and config['rotation_steps_deg'] == spec['rotation_steps_deg'], 'Proposal scale changed')
        require(config['depletant_radius'] == 1.5 and config['reservoir_density'] == .035
            and config['fixed_poses'] == current['physical_fixed_neighbors']
            and config['capture_radius'] == current['capture_radius']
            and config['capture_center'] == current['capture_center']
            and config['metadata'] == current['physical_metric'], 'Physical target changed')
    compiled, model = compile_definition(inputs / 'native/definition.json')
    require(compiled == read(inputs / 'native-compiled.json'), 'Compiled predicate differs from original Python geometry')
    require(model.fixed_poses == baseline['fixed_poses'], 'Native predicate scaffold differs')


def freeze(out, repository, receipt_path, receipt_sha):
    out = Path(out).resolve(); repository = Path(repository).resolve()
    require(not out.exists(), 'Fresh control directory required; no overwrite or retry')
    require(sys.flags.optimize == 0, 'Python optimization invalidates checks')
    ledger = Ledger(); evidence = validate_reference(receipt_path, receipt_sha, ledger)
    reference_root = repository / 'runs/native-excluded-smc-reference-20260924'
    reference_plan = read(ledger.bind(reference_root / 'plan.json', evidence['reference_plan_sha256']))
    binary = ledger.bind(reference_root / 'common/latent-region-smc', evidence['binary_sha256'])
    bundle = ledger.bind(reference_root / 'common/source-bundle.json', evidence['source_bundle_sha256'])
    require(bundle.read_bytes() in binary.read_bytes(), 'Source bundle is not embedded in executable')
    require(reference_plan['binary_sha256'] == evidence['binary_sha256'], 'Reference binary differs')
    provenance = repository / 'runs/contact-confirmation-campaign-20260922/bank/provenance'
    shape = ledger.bind(provenance / 'shape.json', SHAPE_SHA)
    region = ledger.bind(provenance / 'region.json', REGION_SHA)
    reference = ledger.bind(repository / 'runs/refined-contact-bank-preparation-20260922/old-r5-region.json', REFERENCE_SHA)
    baseline = ledger.bind(repository / 'runs/smc-r4-density-bridge-narrow-control-20260922/common/config.json')
    native_dir = repository / 'runs/protected-guide-validation-20260923/common/reference-package/native-region'
    ledger.bind(native_dir / 'definition.json', DEFINITION_SHA)
    compiled = ledger.bind(repository / 'runs/native-entry-compiled-20260924/compiled.json', COMPILED_SHA)
    matching = ledger.bind(repository / 'runs/protected-guide-matching-review-20260924/review.json', MATCHING_SHA)
    # Archive every dynamically loaded native-definition input, not just its manifest.
    for path in native_dir.rglob('*'):
        if path.is_file(): ledger.bind(path)
    inventory = []; previous = set()
    for path in sorted({p for name in ('protocol.json','plan.json','manifest.json')
                        for p in (repository / 'runs').rglob(name)}):
        seeds = seed_values(read(path)); previous.update(seeds)
        inventory.append(dict(path=str(path), sha256=sha(path), seeds=sorted(seeds)))
    require(not previous.intersection(SEEDS), 'A fixed independent seed was used previously')
    from analyze_native_excluded_smc import audit_population  # source-closure dependency
    closure = local_dependencies([Path(__file__), Path(__file__).with_name('test_native_excluded_smc_campaign.py')])
    auditor_closure = local_dependencies([Path(__file__).with_name('analyze_native_excluded_smc.py')])
    require({name: sha(path) for name, path in auditor_closure.items()} == evidence['auditor_sources'],
        'The protein auditor differs from the independently validated reference auditor')
    out.mkdir(parents=True); (out / 'common').mkdir(); (out / 'inputs').mkdir()
    for name, path in closure.items(): shutil.copy2(path, out / 'common' / name)
    shutil.copy2(binary, out / 'common/latent-region-smc'); shutil.copy2(bundle, out / 'common/source-bundle.json')
    for source, name in ((shape,'shape.json'), (region,'region.json'), (reference,'reference-region.json'),
                         (baseline,'baseline-config.json'), (compiled,'native-compiled.json'), (matching,'matching-review.json')):
        shutil.copy2(source, out / 'inputs' / name)
    shutil.copytree(native_dir, out / 'inputs/native')
    base = read(baseline)
    for arm, spec in ARMS.items():
        config = copy.deepcopy(base); config['shape'] = str(out / 'inputs/shape.json')
        for name in ('translation_steps','rotation_steps_deg'): config[name] = spec[name]
        write(out / 'inputs' / (arm + '-config.json'), config)
    target_check(out, base)
    write(out / 'seed-inventory.json', dict(files=inventory, all_seeds=sorted(previous)))
    shutil.copy2(receipt_path, out / 'inputs/reference-audit.json')
    plan = dict(schema=SCHEMA, created=time.time(), repository=str(repository), python=sys.executable,
        python_version=sys.version, python_sha256=sha(sys.executable), thread_environment=THREAD_ENV,
        allocation=allocation(), maximum_physical_workers=8, maximum_total_workers=32, audit_workers=4,
        sources={name: sha(out / 'common' / name) for name in closure},
        source_bindings=ledger.files, reference_audit_sha256=receipt_sha,
        binary_sha256=evidence['binary_sha256'], source_bundle_sha256=evidence['source_bundle_sha256'],
        physical_target=dict(shape_sha256=SHAPE_SHA, region_sha256=REGION_SHA,
            definition_sha256=DEFINITION_SHA, compiled_sha256=COMPILED_SHA,
            target='Hcapture Hhard IR4 (1-Inative) exp(zC) d3t dHaar',
            measure='Cartesian Angstrom cubed times normalized proper SO(3) Haar',
            full_vessel=False), jobs=jobs_for(out), scope=SCOPE,
        analysis=dict(class_order=CLASSES, strata=STRATA,
            comparisons=[['narrow','large'], ['narrow','broad'], ['large','broad']],
            importance_arms=['bank','protected','small','intensity256'],
            relative_SE_max=.1, absolute_log_difference_max=.2, combined_SE_max=3.,
            significant_stratum_fraction=.01,
            no_descendant_importance_ESS=True,
            fixed_budget_failure='Report unresolved, with every zero and unstable stratum retained; no extension.'),
        failure='Stop new launches and drain all started children. No replacement or overwrite.',
        production_authorized=False)
    write(out / 'plan.json', plan)
    write(out / 'freeze.json', dict(files={str(p.relative_to(out)): sha(p)
        for p in out.rglob('*') if p.is_file()}))
    ledger.recheck()
    return plan


def validate(out, expected, check_upstream=True):
    out = Path(out).resolve(); ledger = Ledger(); ledger.bind(out / 'plan.json', expected); ledger.frozen(out)
    plan = read(out / 'plan.json')
    require(plan['schema'] == SCHEMA and plan['allocation'] == allocation()
        and plan['jobs'] == jobs_for(out) and plan['thread_environment'] == THREAD_ENV
        and plan['maximum_physical_workers'] == 8 and plan['maximum_total_workers'] == 32
        and plan['audit_workers'] == 4 and plan['production_authorized'] is False,
        'Fixed allocation or scope changed')
    require(sys.flags.optimize == 0 and sys.executable == plan['python']
        and sys.version == plan['python_version'] and sha(sys.executable) == plan['python_sha256'],
        'Frozen Python runtime differs')
    require(sha(__file__) == plan['sources'][Path(__file__).name], 'Controller source differs')
    require(sha(out / 'common/latent-region-smc') == plan['binary_sha256']
        and sha(out / 'common/source-bundle.json') == plan['source_bundle_sha256'], 'Binary source binding differs')
    if check_upstream:
        for path, digest in plan['source_bindings'].items(): ledger.bind(path, digest)
        target_check(out, read(out / 'inputs/baseline-config.json'))
    return plan, ledger


def context_for(out, plan, job):
    inputs = Path(out) / 'inputs'
    return dict(protocol=plan, config=read(inputs / (job['arm'] + '-config.json')),
        current=read(inputs / 'region.json'), reference=read(inputs / 'reference-region.json'),
        definition=inputs / 'native/definition.json', compiled_definition=inputs / 'native-compiled.json',
        definition_sha256=DEFINITION_SHA, stages_to_classify=plan['allocation']['stage_profiles'])


def audit_one(out, expected, ident):
    from analyze_native_excluded_smc import audit_population
    out = Path(out).resolve(); plan, ledger = validate(out, expected)
    matching = [j for j in plan['jobs'] if j['id'] == ident]
    require(len(matching) == 1, 'Unknown fixed population')
    target = out / 'audits' / ident
    require(not target.exists(), 'Existing population audit; no overwrite')
    result = audit_population(context_for(out, plan, matching[0]), matching[0], target)
    ledger.recheck()
    return result


def mass_statistics(logs):
    """Whole independent populations, including zero estimates; never log-average."""
    require(len(logs) >= 2 and all(x is None or math.isfinite(x) for x in logs), 'Invalid population masses')
    n = len(logs); finite = [x for x in logs if x is not None]
    if not finite:
        return dict(log_Q=None, population_relative_SE=None, population_log_masses=logs,
            populations=n, nonzero_populations=0, unresolved='No positive population estimate')
    offset = max(finite); values = [0. if x is None else math.exp(x-offset) for x in logs]
    mean = math.fsum(values)/n
    se = math.sqrt(math.fsum((x-mean)**2 for x in values)/(n*(n-1)))
    return dict(log_Q=offset+math.log(mean), population_relative_SE=se/mean,
        population_log_masses=logs, populations=n, nonzero_populations=len(finite), unresolved=None)


def mass_comparison(left, right):
    if left['log_Q'] is None or right['log_Q'] is None:
        return dict(passed=False, unresolved='Unobserved contribution; not evidence of zero mass')
    difference = left['log_Q']-right['log_Q']
    error = math.hypot(left['population_relative_SE'], right['population_relative_SE'])
    offset = max(left['log_Q'], right['log_Q'])
    a, b = math.exp(left['log_Q']-offset), math.exp(right['log_Q']-offset)
    linear_se = math.hypot(a*left['population_relative_SE'], b*right['population_relative_SE'])
    within_se = abs(a-b) <= 3*linear_se
    return dict(log_difference=difference, delta_method_log_SE=error,
        linear_comparison_log_scale=offset, scaled_linear_difference=a-b, scaled_linear_SE=linear_se,
        within_absolute_limit=abs(difference)<=.2, within_three_SE=within_se,
        passed=abs(difference)<=.2 and within_se)


def aggregate(populations):
    require(len(populations) == 4, 'Exactly four independent populations per arm required')
    result = dict(estimates={key: mass_statistics([p['terminal']['log_masses'][key] for p in populations])
        for key in CLASSES}, Q0={key: mass_statistics([p['initial_physical_hard_log_masses'][key] for p in populations])
        for key in CLASSES}, strata={})
    for family, size in STRATA.items():
        result['strata'][family] = [{key: mass_statistics([
            p['terminal']['log_strata'][family][key][i] for p in populations]) for key in CLASSES}
            for i in range(size)]
    return result


def authenticated_report(out, plan, job, ledger):
    path = Path(out) / 'audits' / job['id'] / 'population.json'
    ledger.frozen(path.parent)
    status = read(path.parent / 'status.json')
    require(status['complete'] is True and status['phase'] == 'complete', 'Independent audit incomplete')
    ledger.bind(path, status['population_sha256'])
    report = read(path)
    config = read(Path(out) / 'inputs' / (job['arm'] + '-config.json'))
    expected_native = dict(definition_sha256=DEFINITION_SHA, compiled_sha256=COMPILED_SHA,
        shape_sha256=SHAPE_SHA, fixed_poses=config['fixed_poses'])
    require(report['schema'] == 'native-excluded-smc-population-audit-v1' and report['complete'] is True
        and report['id'] == job['id'] and report['seed'] == job['seed'] and report['initial_draws'] == INITIAL_DRAWS
        and report['native_definition'] == expected_native
        and report['independent_analytic_test_adapter'] is False, 'Unrecognized/mismatched independent protein audit')
    for source, digest in report['source_sha256'].items(): ledger.bind(source, digest)
    for name, digest in report['label_sha256'].items(): ledger.bind(path.parent / name, digest)
    require(report['terminal']['particles'] == (0 if report['zero_estimate'] else ARMS[job['arm']]['population']),
        'Terminal allocation differs')
    return report


def finish(out, plan):
    out = Path(out); reports = {}; ledger = Ledger()
    for job in plan['jobs']:
        reports[job['id']] = authenticated_report(out, plan, job, ledger)
    arms = {arm: aggregate([reports[j['id']] for j in plan['jobs'] if j['arm'] == arm]) for arm in ARMS}
    comparisons = {}; strata = {}
    for a, b in plan['analysis']['comparisons']:
        name = a + '-vs-' + b
        comparisons[name] = {key: mass_comparison(arms[a]['estimates'][key], arms[b]['estimates'][key]) for key in CLASSES}
        strata[name] = {}
        for family, size in STRATA.items():
            strata[name][family] = []
            for i in range(size):
                row = {}
                for key in CLASSES:
                    left, right = arms[a]['strata'][family][i][key], arms[b]['strata'][family][i][key]
                    fractions = [0. if item['log_Q'] is None or arms[arm]['estimates'][key]['log_Q'] is None
                        else math.exp(item['log_Q']-arms[arm]['estimates'][key]['log_Q'])
                        for item, arm in ((left,a),(right,b))]
                    row[key] = dict(fractions=fractions, significant=max(fractions)>=.01,
                        comparison=mass_comparison(left,right))
                strata[name][family].append(row)
    prior = read(out / 'inputs/matching-review.json')
    importance = {arm: {old: {quantity: mass_comparison(arms[arm][key]['contact_no_native_entry'],
        prior['new_population_statistics'][old][quantity]['contact_no_native_entry'])
        for quantity, key in [('Qz','estimates'),('Q0','Q0')]}
        for old in plan['analysis']['importance_arms']} for arm in ARMS}
    quality = {arm: {key: dict(population_relative_SE=value['population_relative_SE'],
        passed=value['population_relative_SE'] is not None and value['population_relative_SE'] <= .1)
        for key,value in estimates['estimates'].items() if key != 'unbound_no_native_entry'}
        for arm,estimates in arms.items()}
    report = dict(schema=SCHEMA, complete=True, scope=SCOPE, arms=arms, quality=quality,
        comparisons=comparisons, strata=strata, matching_importance_contact_comparisons=importance,
        audits={j['id']: dict(path=str(out/'audits'/j['id']/'population.json'),
            sha256=sha(out/'audits'/j['id']/'population.json')) for j in plan['jobs']},
        finite_R4_unbound_bound=prior['finite_R4_unbound_bound'],
        full_vessel_authorized=False, assembly_authorized=False, physical_conclusion='unresolved',
        note='A successful restricted denominator control alone cannot repair all full-R4 convergence failures.')
    ledger.recheck()
    report['audited_source_sha256'] = ledger.files
    write(out / 'analysis.json', report)
    return report


def run(out, expected):
    out = Path(out).resolve(); plan, ledger = validate(out, expected)
    require(Path(__file__).resolve() == out / 'common' / Path(__file__).name, 'Run the frozen controller')
    require(not any(Path(j['output']).exists() for j in plan['jobs']), 'Existing population; no restart')
    with (out / 'claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), process_birth=process_token(os.getpid()), plan_sha256=expected), stream)
    for name in ('logs','populations','audits'): (out / name).mkdir()
    state = dict(schema=SCHEMA, complete=False, phase='physical', pid=os.getpid(),
        process_birth=process_token(os.getpid()), plan_sha256=expected, started=time.time(), jobs=[], audits=[])
    for job in plan['jobs']:
        state['jobs'].append(dict(id=job['id'], kind='physical', command=job['argv'],
            directory=job['output'], log=str(out/'logs'/(job['id']+'.log')), status='pending'))
        command = [plan['python'], '-B', str(out/'common'/Path(__file__).name), 'audit',
            '--out', str(out), '--expected-plan-sha256', expected, '--id', job['id']]
        state['audits'].append(dict(id=job['id'], kind='audit', command=command,
            directory=str(out/'audits'/job['id']), log=str(out/'logs'/(job['id']+'-audit.log')), status='pending'))
    def snapshot(): write(out / 'status.json', state)
    def interrupted(signum, frame): raise InterruptedError(f'Signal {signum}; drain started children')
    old = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        snapshot(); env = dict(os.environ, **THREAD_ENV, PYTHONOPTIMIZE='0')
        execute_group(state['jobs'], snapshot, 8, plan['repository'], env,
            before_launch=lambda: validate(out, expected, check_upstream=False))
        state['phase'] = 'independent_audits'; snapshot(); ledger.recheck()
        execute_group(state['audits'], snapshot, 4, plan['repository'], env,
            before_launch=lambda: validate(out, expected, check_upstream=False))
        state['phase'] = 'analysis'; snapshot(); finish(out, plan); ledger.recheck()
        state.update(phase='complete', complete=True, finished=time.time(), analysis_sha256=sha(out/'analysis.json'))
    except BaseException as error:
        state.update(phase='failed', error=repr(error), finished=time.time()); raise
    finally:
        snapshot()
        for sig, handler in old.items(): signal.signal(sig, handler)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--out',type=Path,required=True)
    f.add_argument('--repository',type=Path,required=True); f.add_argument('--reference-audit',type=Path,required=True)
    f.add_argument('--expected-reference-audit-sha256',required=True)
    for name in ('run','audit','validate'):
        p = sub.add_parser(name); p.add_argument('--out',type=Path,required=True)
        p.add_argument('--expected-plan-sha256',required=True)
        if name == 'audit': p.add_argument('--id',required=True)
    args = parser.parse_args()
    if args.action == 'freeze': freeze(args.out,args.repository,args.reference_audit,args.expected_reference_audit_sha256)
    elif args.action == 'audit': audit_one(args.out,args.expected_plan_sha256,args.id)
    elif args.action == 'run': run(args.out,args.expected_plan_sha256)
    else: validate(args.out,args.expected_plan_sha256)


if __name__ == '__main__': main()
