#!/usr/bin/env python3
"""Authenticate and compare one frozen, independent full-vessel stage.

Consumes existing audit/partition sufficient statistics. No sampling, contact
search, native classification, or combination of the standard and large stages.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from analyze_mobile_native_pocket import local_sources
from analyze_r4_smc_control import Ledger, close, nullable_log, read, require, sha, write
from prepare_full_vessel_comparison import ARMS, STAGES, jobs_for, validate as validate_preparation
from vessel_contact_partition import PRIMARY, SUPPORTS, PARTITIONS, independent_mass

SCHEMA = 'full-vessel-stage-comparison-v1'
KINDS = ('Qz', 'Q0')
CLASSES = ('total', *PRIMARY, *PARTITIONS, *('witness:' + s for s in SUPPORTS),
           *(name + ':' + part for name in PRIMARY for part in PARTITIONS))
THRESHOLDS = dict(population_RSE_max=.1, importance_ESS_min=200., largest_draw_fraction_max=.02,
                  delta_F_halfwidth_95_max=.5, between_arm_SE_multiple=3., between_arm_absolute_max=.2)


def frozen_artifact(ledger, root, required):
    """A true flag alone is never a frozen input or a source binding."""
    root = Path(root).resolve()
    entries = ledger.frozen(root)
    require(set(required) <= set(entries), 'Required artifact omitted from freeze: ' + str(root))
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p != root / 'freeze.json'}
    require(actual == set(entries), 'Artifact contains unfrozen or missing files: ' + str(root))
    return entries


def recorded(ledger, path, mapping, expected=None):
    path = Path(path).resolve()
    require(str(path) in mapping, 'Required input omitted from recorded bindings: ' + str(path))
    digest = mapping[str(path)]
    require(expected is None or digest == expected, 'Recorded identity differs: ' + str(path))
    return ledger.bind(path, digest)


def bind_source_closure(ledger, root, mapping, common, plan, entry, frozen):
    closure = local_sources(common / entry)
    for name, path in closure.items():
        require(name in plan['sources'], 'Unprepared analysis source: ' + name)
        digest = plan['sources'][name]
        recorded(ledger, path, mapping, digest)
        archive = 'provenance/' + name
        require(archive in frozen, 'Unfrozen archived analysis source: ' + archive)
        ledger.bind(root / archive, digest)


def validate_manifest(plan, job, manifest):
    expected = dict(samples=job['samples'], seed=job['seed'], cloud_replicates=2,
                    covariance_scale=1., uniform_probability=.1, proposal_anchor_index=None,
                    executable_sha256=plan['binary_sha256'], source_bundle_sha256=plan['source_bundle_sha256'],
                    config_sha256=plan['input_sha256']['config.json'],
                    model_sha256=plan['input_sha256']['model.json'], shape_sha256=plan['input_sha256']['shape.json'],
                    activity=plan['physical']['activity'],
                    atomic_wall=dict(center=plan['physical']['wall_center'], radius=plan['physical']['wall_radius']),
                    bath_wall_permeable=True, physical_fixed_neighbor_count=2,
                    pose_proposal_schema=3, base_component_count=178, virtual_component_count=328,
                    proposal_model_kind='reciprocal-pose-mixture-v1')
    for name, value in expected.items():
        require(name in manifest and manifest[name] == value, 'Executed population identity differs: ' + name)
    close(manifest['lambda'], plan['physical']['activity'] * plan['physical']['lambda_ratio'], 'Cloud intensity differs')
    if job['arm'] == 'vessel':
        require(manifest['schema'] == 4 and 'outer_mixture_schema' not in manifest,
                'Baseline must retain the original vessel proposal')
    else:
        require(manifest['schema'] == 5 and manifest['outer_mixture_schema'] == 'full-vessel-latent-half-mixture-v1'
                and manifest['outer_vessel_probability'] == .5, 'Half-mixture proposal differs')
        require(manifest['latent_region_sha256'] == plan['input_sha256']['current_R4.json']
                and manifest['latent_guide_sha256'] == plan['input_sha256']['guide.json'], 'Executed latent inputs differ')
        require(manifest['latent_reference_ball_is_target_restriction'] is False
                and manifest['latent_source_capture']['restricts_target'] is False,
                'Reporting support became a target restriction')
        require(manifest['latent_defensive_uniform_probability'] == .5
                and manifest['latent_gaussian_component_count'] == 80
                and manifest['density_measure'] == 'Lebesgue center volume times normalized SO(3) Haar measure',
                'Latent law or physical measure differs')


def check_mass(value, n):
    require(type(value['draws']) is int and value['draws'] == n, 'Changed unconditional attempted denominator')
    count = value['nonzero']
    require(type(count) is int and 0 <= count <= n, 'Invalid nonzero count')
    for name in ('logQ', 'log_sum_squared_weights', 'log_max_weight'):
        nullable_log(value[name])
        require((value[name] is None) == (count == 0), 'Zero/nonzero mass metadata differs')
    if count == 0:
        require(value['ess'] == 0 and value['max_fraction'] is None, 'Unobserved mass has fabricated diagnostics')
        return
    log_sum = value['logQ'] + math.log(n)
    ess = math.exp(2 * log_sum - value['log_sum_squared_weights'])
    largest = math.exp(value['log_max_weight'] - log_sum)
    close(value['ess'], ess, 'ESS sufficient statistics differ')
    close(value['max_fraction'], largest, 'Largest draw sufficient statistics differ')
    log_square_fraction = value['log_sum_squared_weights'] - 2 * log_sum
    log_max_fraction = value['log_max_weight'] - log_sum
    tolerance = 2e-8 + 2e-11 * max(abs(log_square_fraction), abs(log_max_fraction))
    require(2 * log_max_fraction - tolerance <= log_square_fraction <= log_max_fraction + tolerance,
            'Impossible joint squared-weight/maximum moments')
    require(1 - 1e-7 <= ess <= count * (1 + 1e-7) and 1 / count - 1e-9 <= largest <= 1 + 1e-9,
            'Impossible nonnegative-weight moments')


def partition_sum(estimates, parent, children):
    for kind in KINDS:
        total = estimates[parent][kind]
        parts = [estimates[c][kind] for c in children]
        require(total['nonzero'] == sum(p['nonzero'] for p in parts), 'Exhaustive partition counts differ')
        for field in ('logQ', 'log_sum_squared_weights', 'log_max_weight'):
            values = [p[field] for p in parts if p[field] is not None]
            if not values:
                require(total[field] is None, 'Empty partition acquired mass')
            else:
                expected = max(values) if field == 'log_max_weight' else float(logsumexp(values))
                close(total[field], expected, 'Exhaustive partition moments differ: ' + parent)


def validate_partition(partition, n):
    require(partition['schema'] == 'full-vessel-contact-partition-v1' and partition['complete'] is True,
            'Completed full-vessel partition required')
    require(partition['samples'] == n and type(partition['invalid_draws']) is int
            and 0 <= partition['invalid_draws'] <= n, 'Partition attempted/invalid counts differ')
    estimates = partition['estimates']
    require(set(estimates) == set(CLASSES), 'Missing or changed exhaustive reporting classes')
    for value in estimates.values():
        require(set(value) == set(KINDS), 'Missing physical mass kind')
        for kind in KINDS:
            check_mass(value[kind], n)
        require(value['Qz']['nonzero'] == value['Q0']['nonzero'], 'Hard and depletion support differs')
    require(estimates['total']['Qz']['nonzero'] == n - partition['invalid_draws'], 'Invalid zeros dropped')
    partition_sum(estimates, 'total', PRIMARY)
    for inside, outside in (PARTITIONS[:2], PARTITIONS[2:]):
        partition_sum(estimates, 'total', (inside, outside))
        for name in PRIMARY:
            partition_sum(estimates, name, (name + ':' + inside, name + ':' + outside))
        for part in (inside, outside):
            partition_sum(estimates, part, tuple(name + ':' + part for name in PRIMARY))
    for kind in KINDS:
        require(estimates['witness:current_R4'][kind] == estimates['inside_current_R4'][kind],
                'Current reporting witness differs from its partition')
        union = estimates['inside_measured_pockets'][kind]
        for name in SUPPORTS:
            witness = estimates['witness:' + name][kind]
            require(witness['nonzero'] <= union['nonzero'], 'Witness exceeds the Boolean pocket union')
            if witness['logQ'] is not None:
                require(union['logQ'] is not None and witness['logQ'] <= union['logQ'] + 1e-8,
                        'Witness mass exceeds the Boolean pocket union')


def load_population(preparation, plan, job, ledger):
    common, inputs = preparation / 'common', preparation / 'inputs'
    root, audit_root, part_root = (Path(job[k]).resolve() for k in ('directory', 'audit_directory', 'partition_directory'))
    audit_frozen = frozen_artifact(ledger, audit_root, ('analysis.json', 'labels.jsonl'))
    part_frozen = frozen_artifact(ledger, part_root, ('analysis.json', 'labels.jsonl'))
    audit = read(ledger.bind(audit_root / 'analysis.json'))
    part = read(ledger.bind(part_root / 'analysis.json'))
    require(audit['schema'] == ('full-vessel-baseline-audit-v1' if job['arm'] == 'vessel' else 'full-vessel-latent-audit-v1')
            and audit['complete'] is True and audit.get('native_binding'), 'Matching completed full-native audit required')
    require(Path(audit['population']).resolve() == root and Path(part['population']).resolve() == root
            and Path(part['audit']).resolve() == audit_root / 'analysis.json', 'Population/audit lineage differs')
    am, pm = audit['source_sha256'], part['input_sha256']
    for mapping in (am, pm):
        for path, digest in mapping.items():
            ledger.bind(path, digest)
    recorded(ledger, audit_root / 'analysis.json', pm, sha(audit_root / 'analysis.json'))
    recorded(ledger, audit_root / 'labels.jsonl', pm, audit_frozen['labels.jsonl'])
    ledger.bind(part_root / 'labels.jsonl', part['labels_sha256'])
    bind_source_closure(ledger, audit_root, am, common, plan,
                        'audit_full_vessel_baseline.py' if job['arm'] == 'vessel' else 'audit_full_vessel_latent.py', audit_frozen)
    bind_source_closure(ledger, part_root, pm, common, plan, 'vessel_contact_partition.py', part_frozen)
    manifest = read(recorded(ledger, root / 'manifest.json', am))
    recorded(ledger, root / 'manifest.json', pm, sha(root / 'manifest.json'))
    require(manifest == audit['manifest'] == part['manifest'], 'Manifest snapshots differ')
    validate_manifest(plan, job, manifest)
    summary = read(recorded(ledger, root / 'summary.json', am))
    require(summary['complete'] is True and summary['manifest'] == manifest and summary['numerical_nulls'] == 0
            and summary['samples'] == job['samples'], 'Population incomplete or attempted draws missing')
    raw = recorded(ledger, root / 'samples.jsonl', am, audit['raw_sample_binding']['sha256'])
    recorded(ledger, raw, pm, sha(raw))
    pairs = [('input-config.json', 'config_sha256'), ('model.json', 'model_sha256'),
             ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256')]
    if job['arm'] == 'half_mixture':
        pairs += [('latent-region.json', 'latent_region_sha256'), ('latent-guide.json', 'latent_guide_sha256')]
    for name, key in pairs:
        recorded(ledger, root / 'provenance' / name, am, manifest[key])
    config = read(recorded(ledger, root / 'config.json', am))
    recorded(ledger, root / 'config.json', pm, sha(root / 'config.json'))
    expected_config = read(inputs / 'config.json')
    for cfg in (config, expected_config):
        for key in ('target_region', 'proposal_anchor_index'):
            if cfg.get(key) is None:
                cfg.pop(key, None)
    require(config == expected_config and config.get('target_region') is None, 'Actual physical/proposal config differs')
    definition_path = inputs / 'native-region/definition.json'
    definition = read(recorded(ledger, definition_path, am, plan['native_definition_sha256']))
    binding = audit['native_binding']
    require(Path(binding['definition']).resolve() == definition_path
            and binding['definition_sha256'] == part['native_definition_sha256'] == plan['native_definition_sha256']
            and binding['input_sha256'] == definition['input_sha256'], 'Native observer identity differs')
    for name, digest in definition['input_sha256'].items():
        recorded(ledger, definition_path.parent / 'inputs' / name, am, digest)
    require(binding['runtime_sha256'] == definition['input_sha256']['source/native_contact_regions.py'],
            'Native runtime identity differs')
    require(set(part['region_paths']) == set(SUPPORTS), 'Reporting support identities missing')
    for name in SUPPORTS:
        path = inputs / (name + '.json')
        require(Path(part['region_paths'][name]).resolve() == path, 'Reporting support path differs')
        recorded(ledger, path, pm, plan['input_sha256'][name + '.json'])
    if job['arm'] == 'vessel':
        b = audit['reporting_guide_binding']
        for field, filename in (('region', 'current_R4.json'), ('guide', 'guide.json')):
            require(Path(b[field]).resolve() == inputs / filename
                    and b[field + '_sha256'] == plan['input_sha256'][filename], 'Baseline reporting binding differs')
            recorded(ledger, inputs / filename, am, plan['input_sha256'][filename])
            ledger.bind(audit_root / 'provenance' / ('reporting-' + field + '.json'), plan['input_sha256'][filename])
        require(b['restricts_target'] is False and b['affects_proposal'] is False, 'Baseline reporting changes proposal/target')
    validate_partition(part, job['samples'])
    for name in ('total', *PRIMARY):
        for kind in KINDS:
            before, after = audit['estimates'][name][kind], part['estimates'][name][kind]
            require(before['draws'] == after['draws'] and before['nonzero'] == after['nonzero'],
                    'Audited primary attempted/nonzero counts changed')
            require((before['logQ'] is None) == (after['logQ'] is None), 'Audited zero/nonzero mass changed')
            require((before['max_fraction'] is None) == (after['max_fraction'] is None), 'Audited maximum support changed')
            if before['logQ'] is not None:
                close(before['logQ'], after['logQ'], 'Audited primary mass changed')
                require(before['ess'] > 0 and before['max_fraction'] > 0, 'Invalid audited primary weight moments')
                close(math.log(before['ess']), math.log(after['ess']), 'Audited primary squared-weight moments changed')
                close(math.log(before['max_fraction']), math.log(after['max_fraction']), 'Audited primary maximum changed')
            else:
                require(before['ess'] == after['ess'] == 0, 'Audited unobserved ESS changed')
    for label, kind in (('total', 'Qz'), ('hard_total', 'Q0')):
        expected = summary['estimates'][label]['log_normalizer']
        actual = part['estimates']['total'][kind]['logQ']
        require((expected is None) == (actual is None), 'Summary zero/nonzero total differs')
        if expected is not None:
            close(expected, actual, 'Summary total differs')
    return dict(id=job['id'], arm=job['arm'], stage=job['stage'], population=job['population'], seed=job['seed'],
                samples=job['samples'], directory=str(root), audit=str(audit_root / 'analysis.json'),
                partition=str(part_root / 'analysis.json'), manifest_sha256=sha(root / 'manifest.json'),
                audit_sha256=sha(audit_root / 'analysis.json'), partition_sha256=sha(part_root / 'analysis.json'),
                raw_sha256=sha(raw), invalid_draws=part['invalid_draws'], estimates=part['estimates'])


def contrast_from_columns(log_means, normalized, names):
    ni, ci = (CLASSES.index(name) for name in names)
    if log_means[ni] is None or log_means[ci] is None:
        return dict(passed=False, unresolved='Native or competing mass unobserved; no finite contrast or upper bound.')
    difference = normalized[:, ci] - normalized[:, ni]
    se = float(difference.std(ddof=1) / 2)
    value = log_means[ci] - log_means[ni]
    half = float(student_t.ppf(.975, 3)) * se
    return dict(beta_F_native_minus_noentry=value, population_SE=se, halfwidth_95=half,
                interval_95=[value - half, value + half], degrees_of_freedom=3,
                native_class=names[0], competing_class=names[1], passed=half <= THRESHOLDS['delta_F_halfwidth_95_max'])


def arm_statistics(populations, kind):
    require(len(populations) == 4, 'Four independent whole populations required')
    estimates, columns, normalized, offsets, log_means = {}, [], [], [], []
    for name in CLASSES:
        values = [p['estimates'][name][kind] for p in populations]
        item = independent_mass(values)
        logs = np.asarray([nullable_log(v['logQ']) for v in values])
        offset = float(logs[np.isfinite(logs)].max()) if np.isfinite(logs).any() else 0.
        x = np.exp(logs - offset)
        mean = float(x.mean())
        columns.append(x); offsets.append(offset); normalized.append(x / mean if mean else np.zeros(4))
        log_means.append(item['log_Q'])
        checks = dict(population_RSE=item['population_relative_SE'] is not None and item['population_relative_SE'] <= .1,
                      importance_ESS=item['importance_ESS'] >= 200,
                      largest_draw=item['largest_contribution'] is not None and item['largest_contribution'] <= .02)
        item.update(quality_checks=checks, passed=all(checks.values()),
                    populations=[dict(id=p['id'], seed=p['seed'], **v) for p, v in zip(populations, values)])
        if mean:
            half = float(student_t.ppf(.975, 3)) * item['population_relative_SE']
            item.update(log_mass_population_SE=item['population_relative_SE'], log_mass_halfwidth_95=half,
                        log_mass_interval_95=[item['log_Q'] - half, item['log_Q'] + half])
        estimates[name] = item
    x, z = np.asarray(columns).T, np.asarray(normalized).T
    covariance = np.cov(x, rowvar=False, ddof=1) / 4
    relative = np.cov(z, rowvar=False, ddof=1) / 4
    matrix = [[float(relative[i, j]) if log_means[i] is not None and log_means[j] is not None else None
               for j in range(len(CLASSES))] for i in range(len(CLASSES))]
    contrasts = dict(native_vs_contact_noentry=contrast_from_columns(log_means, z, PRIMARY[:2]))
    # The requested competing basin is contact/no-entry; also retain the full
    # no-entry union as a separate observable, without renaming the basin.
    ci, ui, ni = (CLASSES.index(name) for name in (PRIMARY[1], PRIMARY[2], PRIMARY[0]))
    finite = [log_means[i] for i in (ci, ui) if log_means[i] is not None]
    if finite and log_means[ni] is not None:
        union_log = float(logsumexp(finite))
        union = sum(z[:, i] * math.exp(log_means[i] - union_log) for i in (ci, ui) if log_means[i] is not None)
        temp_logs, temp_z = list(log_means), z.copy()
        temp_logs[ci], temp_z[:, ci] = union_log, union
        item = contrast_from_columns(temp_logs, temp_z, PRIMARY[:2])
        item['competing_class'] = 'contact_no_native_entry + unbound_no_native_entry'
        contrasts['native_vs_all_noentry'] = item
    else:
        contrasts['native_vs_all_noentry'] = dict(passed=False, unresolved='Native or full no-entry union unobserved.')
    return dict(estimates=estimates, contrasts=contrasts, covariance_of_mean=dict(class_order=list(CLASSES),
                column_log_scales=offsets, scaled_matrix=covariance.tolist(), log_mass_delta_method_matrix=matrix),
                uncertainty_scope='Four independent linear population masses including zero estimates. Cov(mean)=sample Cov/4; '
                'per-column log scales multiply covariance entry (i,j) by exp(scale_i+scale_j). Log/contrast intervals use t(3). '
                'Combined draw ESS and maximum fractions are weight diagnostics, not the uncertainty sample size.')


def agreement(left, right, value='log_Q', error='population_relative_SE'):
    if left.get(value) is None or right.get(value) is None:
        return dict(passed=False, state='unresolved', unresolved='At least one contribution unobserved; no epsilon or missing-mass bound.')
    difference = left[value] - right[value]
    se = math.hypot(left[error], right[error])
    absolute, statistical = abs(difference) <= .2, abs(difference) <= 3 * se + 1e-12
    return dict(difference_half_mixture_minus_vessel=difference, combined_population_SE=se,
                absolute_passed=absolute, three_SE_passed=statistical, passed=absolute and statistical,
                state='corroborated' if absolute and statistical else 'material_disagreement' if not absolute and not statistical else 'unresolved')


def compare(populations, stage):
    require(stage in dict(STAGES), 'Select standard or large independently')
    expected_n = dict(STAGES)[stage]
    require(len(populations) == 8 and len({p['id'] for p in populations}) == 8
            and len({p['seed'] for p in populations}) == 8, 'Missing or dependent stage populations')
    require(all(p['stage'] == stage and p['samples'] == expected_n for p in populations), 'Mixed stage or denominator')
    require(set(p['arm'] for p in populations) == set(ARMS), 'Missing comparison arm')
    arms = {}
    for arm in ARMS:
        selected = sorted((p for p in populations if p['arm'] == arm), key=lambda p: p['population'])
        require([p['population'] for p in selected] == list(range(4)), 'Four ordered independent populations per arm required')
        arms[arm] = {kind: arm_statistics(selected, kind) for kind in KINDS}
    comparisons = {kind: {name: agreement(arms['half_mixture'][kind]['estimates'][name], arms['vessel'][kind]['estimates'][name])
                         for name in CLASSES} for kind in KINDS}
    contrasts = {kind: {name: agreement(arms['half_mixture'][kind]['contrasts'][name], arms['vessel'][kind]['contrasts'][name],
                                      'beta_F_native_minus_noentry', 'population_SE')
                        for name in arms['vessel'][kind]['contrasts']} for kind in KINDS}
    checks = dict(primary_Qz_quality=all(arms[a]['Qz']['estimates'][c]['passed'] for a in ARMS for c in ('total', *PRIMARY)),
                  native_contact_DeltaF_precision=all(arms[a]['Qz']['contrasts']['native_vs_contact_noentry']['passed'] for a in ARMS),
                  primary_Qz_and_Q0_agreement=all(comparisons[k][c]['passed'] for k in KINDS for c in ('total', *PRIMARY)),
                  native_contact_DeltaF_agreement=contrasts['Qz']['native_vs_contact_noentry']['passed'])
    return dict(schema=SCHEMA, complete=True, stage=stage, draws_per_population=expected_n,
                independent_populations_per_arm=4, arms=arms, between_arms=comparisons, between_arm_contrasts=contrasts,
                thresholds=THRESHOLDS, diagnostics=dict(checks=checks, declared_primary_diagnostics_passed=all(checks.values())),
                populations_pooled=False, stages_pooled=False, exhaustive_partition_verified=True,
                populations=[{k: v for k, v in p.items() if k != 'estimates'} for p in populations],
                full_vessel_unseen_modes_certified=False, assembly_stability_established=False,
                coverage_conclusion='unresolved: observed proposal agreement and precision do not bound unseen mass',
                scope='Separate fixed stage. All unconditional attempts, including invalid zeros, retain original N. '
                'Native/contact-noentry/unbound and the complements of current R4 and the Boolean measured-pocket union are exhaustive. '
                'Every reporting contribution is retained, including tiny or unobserved contributions. Diagnostic failures '
                'are unresolved evidence, not physical instability; passing diagnostics is neither an unseen-mode certificate nor an assembly gate.')


def run(preparation, stage, out):
    preparation, out = Path(preparation).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh stage comparison directory required')
    require(stage in dict(STAGES), 'Select exactly one frozen stage')
    plan = validate_preparation(preparation)
    ledger = Ledger()
    frozen = ledger.frozen(preparation)
    require('plan.json' in frozen, 'Preparation plan omitted from freeze')
    require(plan['jobs'] == jobs_for(preparation), 'Frozen allocation changed')
    for group, prefix in (('sources', 'common'), ('input_sha256', 'inputs')):
        for name, digest in plan[group].items():
            key = prefix + '/' + name
            require(key in frozen and frozen[key] == digest, 'Preparation input/source omitted or changed: ' + key)
            ledger.bind(preparation / key, digest)
    for filename, field in (('basin-normalizer', 'binary_sha256'), ('source-bundle.json', 'source_bundle_sha256')):
        key = 'common/' + filename
        require(key in frozen and frozen[key] == plan[field], 'Unfrozen executable/source artifact')
        ledger.bind(preparation / key, plan[field])
    sources = local_sources(__file__)
    source_hashes = {str(path): sha(path) for path in sources.values()}
    populations = [load_population(preparation, plan, job, ledger) for job in plan['jobs'] if job['stage'] == stage]
    result = compare(populations, stage)
    ledger.recheck()
    require(all(sha(path) == digest for path, digest in source_hashes.items()), 'Stage comparison source changed')
    result.update(preparation=str(preparation), preparation_sha256=sha(preparation / 'plan.json'),
                  protocol_sha256=sha(preparation / 'plan.json'), input_sha256=ledger.files, source_sha256=source_hashes,
                  native_definition_sha256=plan['native_definition_sha256'], physical_target=plan['physical'],
                  reporting_support_sha256={name: plan['input_sha256'][name + '.json'] for name in SUPPORTS})
    out.mkdir(parents=True)
    (out / 'provenance').mkdir()
    for name, path in sources.items():
        shutil.copy2(path, out / 'provenance' / name)
        require(sha(out / 'provenance' / name) == source_hashes[str(path)], 'Source changed while archiving')
    ledger.recheck()
    write(out / 'analysis.json', result)
    write(out / 'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True, help='Frozen preparation directory containing plan.json')
    parser.add_argument('--stage', choices=tuple(dict(STAGES)), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.preparation, args.stage, args.out)
    print(dict(complete=result['complete'], stage=result['stage'], diagnostics=result['diagnostics']), flush=True)


if __name__ == '__main__':
    main()
