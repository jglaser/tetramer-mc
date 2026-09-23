#!/usr/bin/env python3
"""One-pass analysis of the frozen protected-guide validation and controls.

This does not replace the failed confirmation campaign or open its assembly gate.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import signal
import time

from analyze_contact_confirmation import (classify_population, summarize_arm,
    CLASSES, DECISION_CLASSES, PARTS, PRIMARY, SUPPLEMENTAL_SCHEMA,
    validate_regions, load_classifier, validate_classifier_target,
    quality_gate, free_energy_interval, compare_mass,
    compare_free_energy_intervals, unbound_volume_bound)
from analyze_mobile_native_pocket import require, read, write, sha, inside, local_sources
from run_full_vessel_comparison import worker_counts

ARM_NAMES = ('bank', 'protected', 'small', 'intensity256')
ALLOCATION = ((1_048_576, 80, 128.), (1_048_576, 84, 128.),
              (262_144, 84, 128.), (262_144, 84, 256.))
DECISION_ARMS = ('bank', 'protected', 'intensity256')
COMPARISONS = [['bank', 'protected'], ['protected', 'small'],
               ['protected', 'intensity256'], ['small', 'intensity256']]
TOTAL_DRAWS = 10_485_760
CONVERGENCE = dict(population_relative_SE_max=.1, importance_ESS_min=200., largest_draw_max=.02,
    log_agreement_absolute_max=.2, log_agreement_combined_SE_max=3., deltaF_95_halfwidth_max=.5,
    significant_stratum_mass_fraction=.01, stratum_log_agreement_absolute_max=.2)
STRATA = dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.],
    latent_orthants='six signs, zero assigned positive; all 64 bins retained')
SCOPE = ('Fresh frozen validation of a reweighted 84-component guide against the unchanged bank, '
    'with separately sampled population-size and cloud-intensity controls. Four independent populations '
    'per arm retain all 10,485,760 attempted denominators, exterior and hard-invalid zeros, two clouds '
    'per valid pose, and the original R4/scaffold/shape/Haar measure/native classifier. No training '
    'or earlier physical rows are pooled or reclassified. Population-based convergence diagnostics '
    'remain necessary but do not bound unseen modes. Earlier failed evidence is not superseded; '
    'full-vessel and assembly gates remain closed pending physical evidence review. No adaptive '
    'stopping, extra allocation or automatic assembly launch occurs.')


def validate_design(protocol):
    require(protocol['convergence'] == CONVERGENCE, 'Original convergence thresholds changed')
    require(protocol['strata'] == STRATA, 'Original radial/angular/orthant strata changed')
    require(protocol['comparisons'] == COMPARISONS, 'Frozen proposal/size/intensity comparisons changed')
    require(protocol['decision_arms'] == list(DECISION_ARMS), 'Decision-arm quality policy changed')
    require([a['id'] for a in protocol['arms']] == list(ARM_NAMES), 'Exactly four frozen arms required')
    for arm, (n, components, intensity) in zip(protocol['arms'], ALLOCATION):
        require(arm['samples'] == n and arm['component_count'] == components
            and arm['alpha'] == .5 and arm['lambda_ratio'] == intensity, 'Frozen proposal allocation changed')
    jobs = protocol['jobs']
    require(len(jobs) == 16 and len({j['seed'] for j in jobs}) == 16,
        'Sixteen independent populations required')
    require(len({(j['arm'], j['id']) for j in jobs}) == 16, 'Duplicate population identity')
    for name, (n, _, _) in zip(ARM_NAMES, ALLOCATION):
        selected = [j for j in jobs if j['arm'] == name]
        require(len(selected) == 4 and {j['id'] for j in selected} == {f'r{i:02d}' for i in range(4)}
            and all(j['samples'] == n for j in selected), 'Four complete frozen populations per arm required')
    require(protocol['total_unconditional_draws'] == sum(j['samples'] for j in jobs) == TOTAL_DRAWS,
        'The bounded validation allocation changed')


def evaluate(arms, protocol):
    validate_design(protocol)
    require(set(arms) == set(ARM_NAMES), 'Four analysis arms required')
    gates = protocol['convergence']
    quality = {a: {name: quality_gate(arms[a]['estimates'][name], gates)
        for name in DECISION_CLASSES} for a in ARM_NAMES}
    intervals = {a: free_energy_interval(arms[a], gates) for a in ARM_NAMES}
    comparisons, significant = {}, []
    for left_name, right_name in COMPARISONS:
        left, right = arms[left_name], arms[right_name]
        masses = {name: {kind: compare_mass(left['estimates'][name], right['estimates'][name], kind, gates)
            for kind in ('Qz', 'Q0')} for name in CLASSES}
        contrast = compare_free_energy_intervals(intervals[left_name], intervals[right_name], gates)
        pair_strata = []
        for family in ('radial', 'angular', 'orthant'):
            for name in DECISION_CLASSES:
                a_bins, b_bins = left['strata'][family][name], right['strata'][family][name]
                require(len(a_bins) == len(b_bins), 'Comparison stratum counts differ')
                for a, b in zip(a_bins, b_bins):
                    require(a['bin'] == b['bin'], 'Comparison stratum ordering differs')
                    fa, fb = a['observed_class_fraction']['Qz'], b['observed_class_fraction']['Qz']
                    if max(fa or 0., fb or 0.) < gates['significant_stratum_mass_fraction']:
                        continue
                    limit = dict(gates, log_agreement_absolute_max=gates['stratum_log_agreement_absolute_max'])
                    pair_strata.append(dict(left_arm=left_name, right_arm=right_name, family=family,
                        region=name, bin=a['bin'], left_fraction=fa, right_fraction=fb,
                        **compare_mass(a, b, 'Qz', limit)))
        significant.extend(pair_strata)
        comparisons[left_name + '/' + right_name] = dict(left_arm=left_name, right_arm=right_name,
            masses=masses, free_energy_contrast=contrast, significant_strata=pair_strata,
            matched_population_size=(next(a for a in protocol['arms'] if a['id'] == left_name)['samples']
                                    == next(a for a in protocol['arms'] if a['id'] == right_name)['samples']))
    checks = dict(decision_arm_regional_quality=all(quality[a][n]['passed']
                      for a in DECISION_ARMS for n in DECISION_CLASSES),
        decision_arm_paired_free_energy_precision=all(intervals[a]['passed'] for a in DECISION_ARMS),
        regional_agreement=all(c['masses'][n][k]['passed'] for c in comparisons.values()
            for n in DECISION_CLASSES for k in ('Qz', 'Q0')),
        total_hard_mass_agreement=all(c['masses']['total']['Q0']['passed'] for c in comparisons.values()),
        direct_free_energy_agreement=all(c['free_energy_contrast']['passed'] for c in comparisons.values()),
        significant_original_strata_agreement=all(v['passed'] for v in significant),
        classifier_contact_consistency=all(a['native_entry_unbound_anomaly_count'] == 0 for a in arms.values()),
        native_partition_sum=all(a['native_partition_sum_verified'] for a in arms.values()))
    efficiency = {}
    for left_name, right_name in COMPARISONS:
        efficiency[left_name + '/' + right_name] = {}
        for name in CLASSES:
            a = arms[left_name]['observed_importance_ESS_per_cpu_second'][name]
            b = arms[right_name]['observed_importance_ESS_per_cpu_second'][name]
            efficiency[left_name + '/' + right_name][name] = dict(left=a, right=b,
                right_over_left=None if a is None or b is None or a <= 0 else b / a)
    passed = all(checks.values())
    return dict(regional_diagnostics_passed=passed, passed=False, checks=checks, quality=quality,
        small_arm_standalone_quality_is_diagnostic=True, decision_arms=list(DECISION_ARMS),
        free_energy_intervals=intervals, comparisons=comparisons, significant_strata=significant,
        significant_stratum_disagreements=[v for v in significant if not v['passed']],
        observed_importance_efficiency=efficiency,
        efficiency_scope='Importance ESS/CPU is descriptive, not trajectory mixing or a certified speedup. '
            'No improvement or per-stratum Pareto requirement enters a physical convergence gate.',
        verdict='fresh regional diagnostics passed; independent coverage/evidence review and assembly remain outstanding'
            if passed else 'unresolved finite-R4 validation',
        original_confirmation_superseded=False, full_vessel_gate_open=False,
        full_wall_coverage_established=False, assembly_stability_established=False,
        missing_confirmatory_checks=['independent important-region and remaining-mass coverage',
            'matching-target reconciliation with completed physical evidence',
            'remaining full-vessel contribution', 'finite-system assembly and boundary controls'])


def validate_terminal(root):
    """Authenticate completed new physical/audit results without replaying them."""
    from run_protected_guide_validation import validate_completed
    protocol, status = validate_completed(root)
    validate_design(protocol)
    for name, path in local_sources(__file__).items():
        require(protocol['python_sources'].get(name) == sha(path), 'Use exactly frozen analyzer closure: ' + name)
    definition = inside(root, protocol['native_definition'])
    reference_path = inside(root, protocol['reference_region'])
    supplemental_path = inside(root, protocol['supplemental_definition'])
    for path, key in ((definition, 'native_definition_sha256'), (reference_path, 'reference_region_sha256'),
            (supplemental_path, 'supplemental_definition_sha256')):
        require(sha(path) == protocol[key], 'Frozen classification input changed: ' + key)
    supplemental, reference = read(supplemental_path), read(reference_path)
    require(supplemental['schema'] == SUPPLEMENTAL_SCHEMA, 'Unexpected supplemental definition')
    for key in ('reference_region_sha256', 'region_sha256', 'native_definition_sha256'):
        require(supplemental[key] == protocol[key], 'Supplemental target binding differs: ' + key)
    classifier, binding = load_classifier(definition)
    assessments = {}
    for arm in protocol['arms']:
        base = root / arm['id']; current = read(base / 'provenance/region.json')
        validate_regions(current, reference)
        config = read(base / 'provenance/config.json')
        require(config['depletant_radius'] == 1.5 and config['reservoir_density'] == .035
            and len(config['fixed_poses']) == 2, 'Fixed bath or scaffold changed')
        validate_classifier_target(config, protocol['shape_sha256'], classifier.definition, definition)
        assessment = read(base / 'assessment/analysis.json')
        n = sum(j['samples'] for j in protocol['jobs'] if j['arm'] == arm['id'])
        require(assessment['estimate']['draws'] == assessment['hard_region']['draws']
            == assessment['independently_reconstructed_poses'] == n, 'Completed audit discarded attempts')
        assessments[arm['id']] = assessment
    return protocol, status, assessments, definition, binding, reference_path, supplemental


def report(result):
    lines = ['# Frozen protected-guide validation', '', result['convergence']['verdict'] + '.', '', SCOPE, '',
        '| Arm | Region | log Qz | Population RSE | Importance ESS | Largest draw |',
        '|---|---|---:|---:|---:|---:|']
    for arm in ARM_NAMES:
        for name in PRIMARY + PARTS:
            e = result['arms'][arm]['estimates'][name]
            row, pop = e['row_uncertainty'], e['population_uncertainty']
            values = ['unobserved', '—', '—', '—'] if row['log_Qz'] is None else [
                f'{row["log_Qz"]:.6f}', f'{pop["Qz_relative_SE"]:.2%}',
                f'{row["Qz_ESS"]:.1f}', f'{row["largest_Qz_fraction"]:.2%}']
            lines.append('| ' + arm + ' | ' + name + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'All original radial/angular strata and 64 orthants are retained. '
        'Agreement requires both |Δlog Q| ≤0.2 and ≤3 combined population standard errors. '
        'The paired native/no-entry free-energy intervals use population covariance and Student-t3; '
        'they are not rigorous unseen-tail bounds. An unobserved region is not assigned zero physical mass.', '',
        'Observed importance ESS/CPU is descriptive. Its improvement for contact-without-entry is not required. '
        'Even passing regional diagnostics require evidence review before full-vessel or assembly production.', '',
        'Every attempted draw remains in the compressed records archive; each contributing pose has '
        'one full native and one exclusion-contact classification. All earlier audits and classifications are reused untouched.']
    return '\n'.join(lines) + '\n'


def write_status(path, state):
    """Replace the status atomically so interruption cannot expose partial JSON."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        write(temporary, state)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def interrupted_by_sigterm(signum, frame):
    # A second TERM must not interrupt the executor's orderly shutdown wait.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise InterruptedError('SIGTERM requested validation analysis shutdown; draining submitted workers')


def initialize_classification_worker():
    # Group signals are handled by the parent; preserve submitted one-pass work.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def wait_for_classification_capacity(repository, requested, state, snapshot,
                                     capacity=worker_counts, pause=time.sleep):
    """Reserve room in the observed shared budget before creating the pool.

    The current parent is already counted. Reserve two extra parent management
    threads as well as the single-threaded children. Like the physical executor,
    this checks same-user scientific work immediately before launching; unrelated
    external processes are not under this controller's authority.
    """
    require(type(requested) is int and 1 <= requested <= 16, 'Invalid classification worker request')
    while True:
        observed = capacity(repository)
        state['phase'] = 'waiting_for_classification_capacity'
        state['classification_capacity'] = dict(observed_workers=observed['workers'],
            physical_pids=observed['physical_pids'], requested_children=requested,
            reserved_management_threads=2, maximum_total_workers=32)
        snapshot()
        if observed['workers'] + requested + 2 <= 32:
            state['phase'] = 'first_classification_pass'
            snapshot()
            return
        pause(.5)


def analyze(campaign, out, workers=16):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    require(type(workers) is int and 1 <= workers <= 16, 'At most sixteen classification workers')
    require(not out.exists(), 'Fresh analysis output required; no reclassification retries')
    protocol, status, assessments, definition, binding, reference_path, supplemental = validate_terminal(root)
    out.mkdir(parents=True); start = time.time()
    state = dict(complete=False, phase='first_classification_pass', completed_populations=[], started=start)
    previous_sigterm = signal.signal(signal.SIGTERM, interrupted_by_sigterm)
    try:
        write_status(out / 'status.json', state)
        tasks = []
        for job in protocol['jobs']:
            arm = next(a for a in protocol['arms'] if a['id'] == job['arm'])
            terminal = next(t for t in status['jobs'] if (t['arm'], t['id']) == (job['arm'], job['id']))
            audit = next(a for a in assessments[job['arm']]['populations'] if a['id'] == job['id'])
            tasks.append(dict(root=str(root), out=str(out), job=job, arm=arm, output=terminal['output'], audit=audit,
                definition=str(definition), classifier_binding=binding, strata=protocol['strata'],
                reference_region=str(reference_path), reference_region_sha256=protocol['reference_region_sha256'],
                supplemental_definition_sha256=protocol['supplemental_definition_sha256']))
        wait_for_classification_capacity(protocol['repository'], workers, state,
            lambda: write_status(out / 'status.json', state))
        records = []
        # The executor drains all submitted work, including queued tasks, on failure. No reclassification.
        with ProcessPoolExecutor(max_workers=workers, initializer=initialize_classification_worker) as pool:
            futures = [pool.submit(classify_population, task) for task in tasks]
            for future in as_completed(futures):
                record = future.result(); records.append(record)
                state['completed_populations'].append(record['arm'] + '/' + record['id'])
                write_status(out / 'status.json', state)
                print('First classification completed: ' + record['arm'] + '/' + record['id'], flush=True)
        state['phase'] = 'statistics'; write_status(out / 'status.json', state)
        arms = {a['id']: summarize_arm(out, a, [r for r in records if r['arm'] == a['id']],
            protocol['strata'], assessments[a['id']]) for a in protocol['arms']}
        for record in records:
            require(sha(inside(out, record['labels'])) == record['labels_sha256'], 'Labels changed during analysis')
        checked, checked_status, _, _, checked_binding, _, _ = validate_terminal(root)
        require(checked == protocol and checked_status == status and checked_binding == binding,
            'Frozen sources changed during analysis')
        result = dict(schema='protected-guide-validation-comparison-v1', complete=True, campaign=str(root),
            protocol_sha256=sha(root / 'protocol.json'), status_sha256=sha(root / 'status.json'),
            freeze_sha256=sha(root / 'freeze.json'), region_sha256=protocol['region_sha256'],
            shape_sha256=protocol['shape_sha256'], native_definition=binding,
            supplemental_definition=supplemental, supplemental_definition_sha256=protocol['supplemental_definition_sha256'],
            reference_region_sha256=protocol['reference_region_sha256'], arms=arms,
            convergence=evaluate(arms, protocol), protocol_convergence=protocol['convergence'],
            strata_definition=protocol['strata'], primary_partition=list(PRIMARY), decision_regions=list(DECISION_CLASSES),
            total_unconditional_draws=protocol['total_unconditional_draws'],
            training_scope='Frozen weights and component geometry; fresh seeds only. Previously inspected fit holdouts are not production validation or pooled evidence.',
            unbound_finite_region_bound=unbound_volume_bound(read(root / 'bank/provenance/region.json')),
            classifier_passes_per_population=1, old_classifications_rerun=0, old_audits_replayed=0,
            original_confirmation_superseded=False, scope=SCOPE,
            analysis_wall_seconds=time.time() - start, analysis_workers=workers)
        sources = out / 'provenance'; sources.mkdir()
        for name, path in local_sources(__file__).items():
            shutil.copy2(path, sources / name)
        for path, name in ((root / 'protocol.json', 'campaign-protocol.json'),
                (reference_path, 'old-r5-region.json'),
                (inside(root, protocol['supplemental_definition']), 'native-partition-definition.json')):
            shutil.copy2(path, sources / name)
        result['analyzer_source_sha256'] = {p.name: sha(p) for p in sources.iterdir()}
        write(out / 'analysis.json', result); (out / 'report.md').write_text(report(result))
        state.update(complete=True, phase='complete', finished=time.time(), analysis_sha256=sha(out / 'analysis.json'))
        write_status(out / 'status.json', state)
        write(out / 'freeze.json', {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        return result
    except BaseException as error:
        state.update(complete=False, phase=state['phase'] + '_failed', exception=repr(error), finished=time.time())
        write_status(out / 'status.json', state)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    analyze(args.campaign, args.out, args.workers)
    print(args.out.resolve(), flush=True)
