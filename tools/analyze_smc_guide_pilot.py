#!/usr/bin/env python3
"""One-pass analysis of the bounded fresh bank-versus-SMC geometry guide pilot.

This does not replace the failed confirmation campaign or open its assembly gate.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
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

ARM_NAMES = ('bank', 'smc')
SAMPLES = 65_536
TOTAL_DRAWS = 2 * 4 * SAMPLES
CONVERGENCE = dict(population_relative_SE_max=.1, importance_ESS_min=200., largest_draw_max=.02,
    log_agreement_absolute_max=.2, log_agreement_combined_SE_max=3., deltaF_95_halfwidth_max=.5,
    significant_stratum_mass_fraction=.01, stratum_log_agreement_absolute_max=.2)
STRATA = dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.],
    latent_orthants='six signs, zero assigned positive; all 64 bins retained')
SCOPE = ('Fresh independent two-arm proposal validation on the unchanged R4 region, repaired shape, '
    'two-anchor scaffold, physical measure and complete native classifier. Four populations of 65,536 '
    'attempts per arm use two independent Poisson clouds at intensity/activity ratio 128. All exterior '
    'and hard-invalid draws remain zero contributions in their original denominators. The old-R5 '
    'native intersection and its native complement are classified during the same first pass. '
    'Training and earlier confirmation rows are not pooled or reclassified. This bounded pilot '
    'cannot replace the failed fixed confirmation, open the full-vessel gate, certify unseen-mode '
    'coverage, establish mixing or decide assembly. No automatic follow-up allocation is authorized.')


def validate_design(protocol):
    """Fail closed on a changed physical allocation or relaxed diagnostic rule."""
    require(protocol['convergence'] == CONVERGENCE, 'Original convergence thresholds changed')
    require(protocol['strata'] == STRATA, 'Original radial/angular/orthant strata changed')
    require(protocol['comparisons'] == [['bank', 'smc']], 'Frozen two-arm comparison changed')
    require(protocol['decision_arms'] == list(ARM_NAMES), 'Both proposal arms require diagnostics')
    arms = protocol['arms']
    require([a['id'] for a in arms] == list(ARM_NAMES), 'Exactly bank and smc arms required')
    for arm, components in zip(arms, (80, 84)):
        require(arm['samples'] == SAMPLES and arm['component_count'] == components
            and arm['alpha'] == .5 and arm['lambda_ratio'] == 128., 'Frozen proposal allocation changed')
    jobs = protocol['jobs']
    require(len(jobs) == 8 and len({j['seed'] for j in jobs}) == 8, 'Eight independent populations required')
    require(len({(j['arm'], j['id']) for j in jobs}) == 8, 'Duplicate population identity')
    for name in ARM_NAMES:
        selected = [j for j in jobs if j['arm'] == name]
        require(len(selected) == 4 and all(j['samples'] == SAMPLES for j in selected),
            'Four complete populations per arm required')
    require(protocol['total_unconditional_draws'] == sum(j['samples'] for j in jobs) == TOTAL_DRAWS,
        'The bounded pilot allocation changed')


def evaluate(arms, protocol):
    validate_design(protocol)
    require(set(arms) == set(ARM_NAMES), 'Exactly bank and smc analysis arms required')
    gates = protocol['convergence']
    quality = {a: {name: quality_gate(arms[a]['estimates'][name], gates)
        for name in DECISION_CLASSES} for a in ARM_NAMES}
    intervals = {a: free_energy_interval(arms[a], gates) for a in ARM_NAMES}
    left, right = (arms[a] for a in ARM_NAMES)
    masses = {name: {kind: compare_mass(left['estimates'][name], right['estimates'][name], kind, gates)
        for kind in ('Qz', 'Q0')} for name in CLASSES}
    contrast = compare_free_energy_intervals(intervals['bank'], intervals['smc'], gates)
    significant = []
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
                significant.append(dict(left_arm='bank', right_arm='smc', family=family,
                    region=name, bin=a['bin'], left_fraction=fa, right_fraction=fb,
                    **compare_mass(a, b, 'Qz', limit)))
    checks = dict(both_arms_regional_quality=all(v['passed'] for a in quality.values() for v in a.values()),
        both_paired_free_energy_precision=all(v['passed'] for v in intervals.values()),
        regional_agreement=all(masses[n][k]['passed'] for n in DECISION_CLASSES for k in ('Qz', 'Q0')),
        total_hard_mass_agreement=masses['total']['Q0']['passed'],
        direct_free_energy_agreement=contrast['passed'],
        significant_original_strata_agreement=all(v['passed'] for v in significant),
        classifier_contact_consistency=all(a['native_entry_unbound_anomaly_count'] == 0 for a in arms.values()),
        native_partition_sum=all(a['native_partition_sum_verified'] for a in arms.values()))
    efficiency = {}
    for name in CLASSES:
        a = left['observed_importance_ESS_per_cpu_second'][name]
        b = right['observed_importance_ESS_per_cpu_second'][name]
        efficiency[name] = dict(bank=a, smc=b,
            smc_over_bank=None if a is None or b is None or a <= 0 else b / a)
    passed = all(checks.values())
    return dict(pilot_diagnostics_passed=passed, passed=False, checks=checks, quality=quality,
        free_energy_intervals=intervals, comparisons={'smc': dict(left_arm='bank', right_arm='smc',
            masses=masses, free_energy_contrast=contrast)}, significant_strata=significant,
        significant_stratum_disagreements=[v for v in significant if not v['passed']],
        observed_importance_efficiency=efficiency,
        efficiency_scope='Observed importance ESS per sampler CPU second, not mixing ESS or a certified speedup. '
            'Efficiency improvements in contact-without-entry are not a pass requirement.',
        verdict='bounded proposal diagnostics passed; original confirmation and assembly gates remain closed'
            if passed else 'unresolved finite-R4 proposal validation',
        original_confirmation_superseded=False, full_vessel_gate_open=False,
        full_wall_coverage_established=False, assembly_stability_established=False,
        missing_confirmatory_checks=['new proposal population-size and cloud-intensity sensitivity',
            'independent important-region and remaining-mass coverage', 'remaining full-vessel contribution',
            'finite-system assembly and boundary controls'])


def validate_terminal(root):
    """Authenticate completed new physical/audit results without replaying them."""
    from run_smc_guide_pilot import validate_completed
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
    lines = ['# Fresh SMC geometry guide pilot', '', result['convergence']['verdict'] + '.', '', SCOPE, '',
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
        'Even a passing pilot cannot replace the failed confirmation or authorize full-vessel/assembly production.', '',
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
    raise InterruptedError('SIGTERM requested pilot analysis shutdown; draining submitted workers')


def initialize_classification_worker():
    # Group signals are handled by the parent; preserve submitted one-pass work.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def analyze(campaign, out, workers=4):
    root, out = Path(campaign).resolve(), Path(out).resolve()
    require(type(workers) is int and 1 <= workers <= 4, 'At most four classification workers')
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
        result = dict(schema='smc-guide-pilot-comparison-v1', complete=True, campaign=str(root),
            protocol_sha256=sha(root / 'protocol.json'), status_sha256=sha(root / 'status.json'),
            freeze_sha256=sha(root / 'freeze.json'), region_sha256=protocol['region_sha256'],
            shape_sha256=protocol['shape_sha256'], native_definition=binding,
            supplemental_definition=supplemental, supplemental_definition_sha256=protocol['supplemental_definition_sha256'],
            reference_region_sha256=protocol['reference_region_sha256'], arms=arms,
            convergence=evaluate(arms, protocol), protocol_convergence=protocol['convergence'],
            strata_definition=protocol['strata'], primary_partition=list(PRIMARY), decision_regions=list(DECISION_CLASSES),
            total_unconditional_draws=protocol['total_unconditional_draws'],
            training_scope='Frozen SMC-derived proposal; independent seeds and populations only. No training estimates pooled.',
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
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    analyze(args.campaign, args.out, args.workers)
    print(args.out.resolve(), flush=True)
