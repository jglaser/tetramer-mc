#!/usr/bin/env python3
"""Assess regional prerequisites for the next vessel measurement, without dispatch.

The bound plan names the completed sensitivity comparison and matching SMC
comparison. Implementation admissions remain distinct from statistical evidence.
Only JSON metadata is read; no poses, clouds, classifiers or row audits are rerun.
Passing this assessment cannot certify the unmeasured vessel remainder or assembly.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import native_class_physical_followup as followup
import compare_native_class_physical_smc as smc
from analyze_conditional_ray_campaign import unbound_volume_bound

stats = followup.statistics
require = stats.require
PLAN_SCHEMA = 'native-class-vessel-prerequisites-plan-v1'
SCHEMA = 'native-class-vessel-prerequisites-v1'
HISTORICAL = ('unrestricted_broad', 'unrestricted_narrow', 'restricted_narrow',
              'restricted_large', 'restricted_broad')
CONTACT_REGIONS = ('native', 'competing', 'native_old_r5', 'native_remainder')
UNBOUND_SOURCE = dict(path='/home/xvg/tetramer-mc/runs/mobile-conditional-ray-comparison-20260921/analysis.json',
    sha256='f2cebea9ba45fabe9683c531130d0ab5a678a233d635524415b176421d295b1e')
SMC_PINS = dict(smc_comparison='1ae01629d998e188f0112470dcb961a70efe2677182c240b0352ee5d6756ab4c',
                smc_plan='24c7c6b70af211dbc6188b277cb49269ddb621a04991dfaf080d6a29a8bf798d')


def comparison_state(value):
    """Use the archived independent *linear* SE test, never a log-SE substitute."""
    require(type(value['observed']) is bool and type(value['passed']) is bool,
            'Malformed independent comparison')
    require(value['SE_multiplier'] == 3. and value['absolute_limit'] == .2,
            'Changed comparison thresholds')
    if not value['observed']:
        require(value['passed'] is False and value['SE_passed'] is None
                and value['absolute_passed'] is None, 'Unobserved comparison claimed resolved')
        return 'unresolved'
    require(type(value['SE_passed']) is bool and type(value['absolute_passed']) is bool
            and value['passed'] == (value['SE_passed'] and value['absolute_passed']),
            'Inconsistent comparison decision')
    if value['passed']:
        return 'corroborated'
    return ('material_contradiction' if not value['SE_passed'] and not value['absolute_passed']
            else 'unresolved')


def validate_arm(arm_id, arm, target):
    """Recheck saved sufficient statistics, partitions and attempted denominators."""
    records = arm['populations']
    require(len(records) == 8 and len({r['seed'] for r in records}) == 8,
            'Eight independent populations required')
    require(arm['classifier_contact_consistency_passed'] is True
            and all(r['classifier_contact_consistency_passed'] is True
                    and r['native_entry_unbound_anomalies'] == 0 for r in records),
            'Native classifier/contact consistency failed')
    for kind in ('Qz', 'Q0'):
        values = arm['estimates'][kind]
        for family, size in [(None, 1), *stats.BINS.items()]:
            blocks = [values['primary']] if family is None else values['strata'][family]
            require(len(blocks) == size, 'Missing stratum')
            for index, block in enumerate(blocks):
                declaration, rows = block['declaration'], block['population_records']
                require(declaration['arm_id'] == arm_id
                        and declaration['target_and_regions_sha256'] == smc.scope_id(target, family, index)
                        and tuple(declaration['regions']) == stats.REGIONS,
                        'Changed population or stratum target')
                require([(r['id'], r['seed'], r['draws'], r['unconditional_denominator']) for r in rows]
                        == [(r['id'], r['seed'], r['draws'], r['unconditional_denominator']) for r in records],
                        'Population denominator or identity differs')
                for row in rows:
                    smc.partition({smc.MAP[k]: v for k, v in row['log_masses'].items()})
                recomputed = stats.population.summarize_populations(declaration, rows)
                require(block['population_statistics'] == recomputed,
                        'Population summary does not match linear population masses')
                for region in stats.REGIONS:
                    diagnostic = block['row_diagnostics'][region]
                    require(diagnostic['observed'] is recomputed['estimates'][region]['observed_positive'],
                            'Row/population observation flag differs')
                    smc.equal_log(diagnostic['logQ'], recomputed['estimates'][region]['log_linear_mean'],
                                  'Row/population mean mismatch')
                for region in stats.DECISIONS:
                    require(block['quality'][region] == stats.quality(block['row_diagnostics'][region],
                                recomputed['estimates'][region]), 'Changed regional quality decision')
                if family is not None:
                    for region in stats.REGIONS:
                        whole = values['primary']['row_diagnostics'][region]['logQ']
                        part = block['row_diagnostics'][region]['logQ']
                        expected = None if whole is None else 0. if part is None else math.exp(part-whole)
                        require(block['observed_parent_mass_fraction'][region] == expected,
                                'Stratum mass fraction differs from linear masses')
            if family is not None:
                for population_index in range(8):
                    for region in stats.REGIONS:
                        whole = values['primary']['population_records'][population_index]['log_masses'][region]
                        summed = smc.logsum(b['population_records'][population_index]['log_masses'][region] for b in blocks)
                        smc.equal_log(whole, summed, 'Strata do not partition the primary population mass')


def regional_diagnostics(comparison):
    """Retain primary failures as well as fresh sensitivity comparisons."""
    require(comparison['schema'] == followup.COMPARISON_SCHEMA and comparison['complete'] is True
            and comparison['gates'] == stats.GATES and comparison['missing_studies'] == [],
            'Completed unchanged sensitivity comparison required')
    expected_arms = {'hard_free', 'class', *(a['id'] for a in followup.ARMS)}
    require(set(comparison['arms']) == expected_arms, 'Missing proposal or sensitivity arm')
    seeds = [r['seed'] for a in comparison['arms'].values() for r in a['populations']]
    require(len(seeds) == 40 and len(set(seeds)) == 40, 'Overlapping independent populations')
    for arm_id, arm in comparison['arms'].items():
        validate_arm(arm_id, arm, comparison['target_and_regions_sha256'])
    all_pairs = comparison['retained_primary_comparisons'] + comparison['comparisons']
    expected_pairs = {('hard_free', 'class'), *(('class', a['id']) for a in followup.ARMS)}
    require(len(all_pairs) == len(expected_pairs)
            and {(p['left'], p['right']) for p in all_pairs} == expected_pairs,
            'Missing primary or sensitivity comparison')
    for pair in all_pairs:
        history = [{k: entry[k] for k in ('family', 'bin', 'region')}
                   for entry in pair['kinds']['Qz']['strata'] if entry['historical_failure']]
        require(pair == followup.comparison_pair(pair['left'], pair['right'], comparison['arms'], history),
                'Comparison differs from saved independent populations or stratum policy')
    diagnostics = followup.admission.summarize_diagnostics(dict(arms=comparison['arms'], comparisons=all_pairs))
    # Unobserved unbound mass is not evidence of zero. Keep it separate so a
    # reviewed deterministic finite-R4 bound can discharge that obligation.
    contact_issues = [x for x in diagnostics['fixed_comparison_issues'] if x['region'] != 'unbound']
    return dict(diagnostics=diagnostics,
        contact_checks_passed=not any((diagnostics['regional_quality_issues'],
            diagnostics['free_energy_precision_issues'], contact_issues,
            diagnostics['unstable_stratum_comparisons'])),
        unbound_comparison_issues=[x for x in diagnostics['fixed_comparison_issues'] if x['region'] == 'unbound'],
        contact_comparison_issues=contact_issues)


def independent_diagnostics(report, target):
    require(report['schema'] == smc.SCHEMA and report['complete'] is True
            and report['target_and_regions_sha256'] == target and report['region_mapping'] == smc.MAP,
            'Matching-target completed SMC comparison required')
    expected = {(a, h) for a in ('hard_free', 'class') for h in HISTORICAL}
    pairs = report['comparisons']
    require(len(pairs) == len(expected)
            and {(p['current_arm'], p['historical_arm']) for p in pairs} == expected,
            'Missing independent SMC control')
    require(set(report['historical_population_batches']) == set(HISTORICAL),
            'Historical population inventory changed')
    seeds = []
    for name, batches in report['historical_population_batches'].items():
        require(set(batches) == {'Qz', 'Q0'}, 'Missing historical hard or physical mass')
        for kind, batch in batches.items():
            valid = stats.population._validate(batch['declaration'], batch['population_records'])
            expected_regions = smc.RESTRICTED if name.startswith('restricted_') else stats.REGIONS
            require(valid.count == 4 and valid.target_and_regions_sha256 == target
                    and valid.arm_id == name and valid.draws == 262144
                    and valid.regions == tuple(expected_regions),
                    'Changed SMC population target or count')
            if kind == 'Qz': seeds.extend(valid.seeds)
        require(batches['Qz']['declaration'] == batches['Q0']['declaration'],
                'Hard and physical SMC population identities differ')
    require(len(seeds) == len(set(seeds)) == 20, 'SMC population seeds overlap')
    states = []
    for pair in pairs:
        restricted = pair['historical_arm'].startswith('restricted_')
        require(pair['restricted'] is restricted, 'Restricted/unrestricted target conflated')
        regions = set(smc.RESTRICTED if restricted else stats.REGIONS)
        for kind in ('Qz', 'Q0'):
            block = pair['kinds'][kind]
            require(set(block['regions']) == regions, 'Changed independent region inventory')
            if kind == 'Qz':
                wanted = {(f, i, r) for f, n in stats.BINS.items() for i in range(n) for r in regions}
                got = [(s['family'], s['bin'], s['region']) for s in block['strata']]
                require(len(got) == len(wanted) and set(got) == wanted and block['strata_available'] is True,
                        'Independent physical stratum inventory differs')
            else:
                require(block['strata'] == [] and block['strata_available'] is False
                        and block['unavailable_reason'], 'Do not manufacture missing historical hard strata')
            for region, value in block['regions'].items():
                states.append(dict(current_arm=pair['current_arm'], historical_arm=pair['historical_arm'],
                    kind=kind, region=region, stratum=None, state=comparison_state(value)))
            for entry in block['strata']:
                if entry['material']:
                    states.append(dict(current_arm=pair['current_arm'], historical_arm=pair['historical_arm'],
                        kind=kind, region=entry['region'], stratum=[entry['family'], entry['bin']],
                        state=comparison_state(entry['comparison'])))
    return dict(states=states,
        material_contradictions=[x for x in states if x['state'] == 'material_contradiction'],
        unresolved=[x for x in states if x['state'] == 'unresolved'],
        scope='Unobserved SMC terminal regions are unresolved, not zero; restricted totals are not full totals.')


def unbound_evidence(report, comparison, region, source_ref):
    """Authenticate the finite *absolute* bound; do not certify relative smallness."""
    target = comparison['target_descriptor']
    require(report['schema'] == 'conditional-ray-reference-comparison-v1' and report['complete'] is True
            and report['region_sha256'] == target['region_sha256']
            and report['shape_sha256'] == target['shape_sha256']
            and report['native_definition']['definition_sha256'] == target['native_definition_sha256'],
            'Unbound certificate has another region, shape or classifier')
    computed = unbound_volume_bound(region)
    require(report['unbound_finite_region_bound'] == computed, 'Geometric unbound bound changed')
    return dict(source=source_ref, absolute_bound=computed,
        measure=target['physical_measure'], region_sha256=target['region_sha256'],
        Qz_unbound_estimate_resolved=False, Q0_unbound_estimate_resolved=False,
        relative_negligibility_proven=False,
        scope='Finite absolute bound only. Ratios to estimated contact masses are not rigorous '
              'fraction bounds. In particular this bound does not establish negligible Q0 unbound mass.')


def completed_followup(protocol, protocol_ref, admitted, fresh, metadata):
    """Check terminal lifecycle links, inheriting already executed row audits."""
    root = Path(protocol['root']).resolve()
    admission_ref = dict(path=str(root/'analysis/admission-plan.json'), sha256=admitted['plan_sha256'])
    plan = metadata.read(admission_ref)
    require(plan['schema'] == followup.admission.PLAN_SCHEMA and plan['protocol'] == protocol_ref
            and plan['statistics_result'] == admitted['statistics_result'], 'Follow-up admission plan differs')
    for name in ('protocol','execution_plan','execution_status','statistics_plan','statistics_result'):
        smc.bound(plan[name], admitted['input_sha256'])
    execution, done, statistical_plan = [metadata.read(plan[k]) for k in
                                        ('execution_plan','execution_status','statistics_plan')]
    require(Path(plan['execution_status']['path']).resolve() == root/'execution/summary.json'
            and Path(execution['root']).resolve() == root
            and done['schema'] == execution['schema'] == followup.admission.EXECUTION_SCHEMA
            and done['complete'] is True and done['passed'] is True and done['failure'] is None
            and done['active'] is None and done['unstarted'] == []
            and done['plan_sha256'] == plan['execution_plan']['sha256']
            and done['retries'] == done['replacements'] == 0
            and done['maximum_workers'] == done['threads'] == execution['maximum_workers'] == execution['threads'] == 1,
            'Follow-up execution has not completed and drained')
    require(not (root/'execution/failure.json').exists(), 'Preserved execution failure')
    smc.bound(protocol_ref, execution['files'])
    expected = {(f'{a["id"]}-{p["id"]}', phase) for a in protocol['arms'] for p in a['populations']
                for phase in ('producer','algebra','labels','selection','geometry')} | {('all','statistics')}
    require(len(execution['jobs']) == len(done['completed']) == len(expected) == 121
            and {(j['population'],j['phase']) for j in execution['jobs']} == expected,
            'Missing follow-up execution phase')
    for job, terminal in zip(execution['jobs'], done['completed']):
        require(all(terminal[k] == job[k] for k in ('id','population','phase','argv'))
                and terminal['success'] is True and terminal['returncode'] == 0
                and terminal['child_drained'] is True and terminal['error'] is None
                and terminal['timeout'] is False and terminal['terminal_hash_error'] is None
                and terminal['retries'] == terminal['replacements'] == 0,
                'Follow-up terminal phase is incomplete or differs')
        smc.bound(terminal['terminal'], admitted['input_sha256'])
    require(done['completed'][-1]['terminal'] == admitted['statistics_result'], 'Statistics terminal differs')
    require(statistical_plan['schema'] == stats.PLAN_SCHEMA and fresh['schema'] == stats.SCHEMA
            and fresh['complete'] is True and fresh['comparisons'] == []
            and fresh['plan_sha256'] == plan['statistics_plan']['sha256'], 'Wrong completed fresh statistics')
    smc.bound(plan['statistics_plan'], fresh['input_sha256'])
    require(all(statistical_plan[k] == protocol[k] for k in
                ('target','target_and_regions_sha256','strata','gates','comparisons')),
            'Fresh statistics changed the frozen target or diagnostics')
    require(admitted['target_and_regions_sha256'] == protocol['target_and_regions_sha256']
            and admitted['total_unconditional_attempts'] == followup.TOTAL_ATTEMPTS
            and admitted['external_primary'] == protocol['external_primary']
            and admitted['primary_is_external_control'] is True,
            'Follow-up admission target, allocation or external control differs')


def load(plan_path, digest):
    """Pin reports to the already completed implementation and execution evidence."""
    metadata = smc.Metadata()
    plan = metadata.read(dict(path=str(Path(plan_path).resolve()), sha256=digest))
    require(plan['schema'] == PLAN_SCHEMA, 'Wrong prerequisite plan')
    require(all(plan[key]['sha256'] == value for key,value in SMC_PINS.items()),
            'Independent SMC artifact needs a new reviewed binding')
    comparison = metadata.read(plan['followup_comparison'])
    protocol = metadata.read(comparison['protocol']); followup.protocol_inventory(protocol)
    require(comparison['protocol'] == plan['followup_protocol'], 'Unexpected follow-up protocol')
    # These five fixed primary pins protect the completed source control.
    primary_bindings = stats.Bindings()
    primary = followup.primary_evidence(protocol['external_primary'], primary_bindings)
    for path,digest in primary_bindings.files.items():
        require(path not in metadata.files or metadata.files[path] == digest, 'Conflicting primary binding')
        metadata.files[path] = digest
    admitted = metadata.read(comparison['followup_admission'])
    fresh = metadata.read(comparison['statistics_result'])
    require(admitted['schema'] == followup.ADMISSION_SCHEMA and admitted['complete'] is True
            and all(admitted[k] is True for k in ('implementation_admission_passed',
                'execution_lifecycle_passed', 'selected_full_geometry_gate_satisfied'))
            and admitted['protocol'] == comparison['protocol']
            and admitted['statistics_result'] == comparison['statistics_result'],
            'Completed follow-up implementation admission required')
    completed_followup(protocol, comparison['protocol'], admitted, fresh, metadata)
    for ref in (comparison['protocol'], comparison['followup_admission'], comparison['statistics_result']):
        smc.bound(ref, comparison['input_sha256'])
    require(comparison['external_primary'] == protocol['external_primary']
            and comparison['target_and_regions_sha256'] == protocol['target_and_regions_sha256']
            == fresh['target_and_regions_sha256'] == primary['statistics']['target_and_regions_sha256']
            and comparison['target_descriptor'] == fresh['target_descriptor'] == primary['statistics']['target_descriptor']
            and fresh['gates'] == stats.GATES, 'Unmatched target, scaffold, measure or physical classifier')
    followup.check_report_inventory(fresh, protocol['arms'], admitted['populations'])
    require(comparison['arms'] == dict(primary['statistics']['arms'], **fresh['arms'])
            and comparison['retained_primary_comparisons'] == primary['statistics']['comparisons'],
            'Changed inherited populations or primary failures')
    required = [(p['left'], p['right']) for p in protocol['external_comparisons']]
    require([(p['left'], p['right']) for p in comparison['comparisons']] == required,
            'Changed sensitivity comparison order')
    for actual, declared in zip(comparison['comparisons'], protocol['external_comparisons']):
        expected = followup.comparison_pair(declared['left'], declared['right'], comparison['arms'],
                                           declared['historical_failed_strata'])
        require(actual == expected, 'Changed frozen historical stratum policy')
    independent = metadata.read(plan['smc_comparison'])
    smc_plan = metadata.read(plan['smc_plan'])
    require(independent['plan_sha256'] == plan['smc_plan']['sha256']
            and smc_plan['schema'] == smc.PLAN_SCHEMA
            and smc_plan['current']['admission'] == protocol['external_primary']['admission'],
            'SMC comparison uses another primary control')
    # Authenticate the existing report's JSON evidence without replaying the SMC
    # analyses or any scientific row streams. Non-JSON raw inputs stay upstream.
    for path, sha256 in independent['input_sha256'].items():
        metadata.read(dict(path=path, sha256=sha256))
    smc.bound(plan['smc_plan'], independent['input_sha256'])
    require(plan['unbound_reference'] == UNBOUND_SOURCE, 'Unreviewed unbound-bound source')
    bound_report = metadata.read(plan['unbound_reference'])
    region = metadata.read(protocol['target']['region'])
    unbound = unbound_evidence(bound_report, comparison, region, plan['unbound_reference'])
    require(independent['historical_verdicts_unchanged']['restricted']['finite_R4_unbound_bound']
            == unbound['absolute_bound'], 'Independent unbound-bound copy differs')
    metadata.finish()
    return plan, comparison, independent, unbound, metadata


def assess(comparison, independent, unbound):
    regional = regional_diagnostics(comparison)
    external = independent_diagnostics(independent, comparison['target_and_regions_sha256'])
    current_seeds = {r['seed'] for arm in comparison['arms'].values() for r in arm['populations']}
    old_seeds = {r['seed'] for group in independent['historical_population_batches'].values()
                 for r in group['Qz']['population_records']}
    require(current_seeds.isdisjoint(old_seeds), 'Current and historical streams overlap')
    require(unbound['region_sha256'] == comparison['target_descriptor']['region_sha256']
            and unbound['relative_negligibility_proven'] is False,
            'Explicit absolute-bound scope required')
    checks = dict(regional_contact_diagnostics=regional['contact_checks_passed'],
        no_material_matching_SMC_contradiction=not external['material_contradictions'],
        finite_R4_unbound_absolute_bound_retained=True)
    return dict(schema=SCHEMA, complete=True, target_and_regions_sha256=comparison['target_and_regions_sha256'],
        checks=checks, full_vessel_measurement_prerequisites_passed=all(checks.values()),
        regional=regional, independent=external, finite_R4_unbound_evidence=unbound,
        admission_scope='Contact-region prerequisites for measuring the full vessel; '
                        'not complete Q0 or Qz normalization of R4.',
        independent_precision_status='unresolved' if external['unresolved'] else 'corroborated',
        full_vessel_convergence_established=False, assembly_gate_open=False,
        finite_system_verdict='unresolved', unseen_vessel_mass='not measured by regional evidence',
        physical_jobs_launched=0, new_pose_draws=0, new_Poisson_clouds=0, parsed_scientific_rows=0,
        scope='Prerequisites only. No dispatch authority, unseen-mass certificate or finite-assembly conclusion. '
              'SMC precision limitations remain explicit even if no material contradiction is observed.')


def run(plan_path, digest):
    plan, comparison, independent, unbound, metadata = load(plan_path, digest)
    result = assess(comparison, independent, unbound)
    result.update(plan_sha256=digest, input_sha256=metadata.files,
        source_sha256={str(p): stats.streaming.sha(p) for p in followup.local_sources(__file__).values()})
    metadata.finish()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Fresh prerequisites report required')
    result = run(args.plan, args.plan_sha256)
    with args.out.open('x') as handle:
        import json
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
