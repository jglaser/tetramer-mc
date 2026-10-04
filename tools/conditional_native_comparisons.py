"""Pure descriptive comparisons of all 128 completed conditional native chains.

Only saved per-chain metrics enter this reducer. It does not read trajectories,
repeat ESS estimation, classify poses, concatenate chains or pool contexts.
"""
from __future__ import annotations

import copy
import itertools
import math

from analyze_evolving_dimer_benchmark import key, occupancy_difference, require, total_variation

SCHEMA = 'conditional-native-comparisons-v1'
ARMS = ('local', 'm4', 'singleton_two_neighbor', 'singleton_two_neighbor_unfused')
STARTS = ('source', 'proposal_prepared')
DESCRIPTORS = ('external_partners', 'external_motifs', 'internal_motifs')
FRACTIONS = ('any_native_attachment_fraction', 'external_native_attachment_fraction',
             'internal_native_attachment_fraction', 'resolved_native_attachment_fraction',
             'empty_native_registry_fraction', 'mobile_registry_resolved_fraction',
             'mobile_cycle_frustration_fraction')
COUNTS = ('enter_nonempty', 'leave_nonempty', 'direct_nonempty_resolved_changes',
          'direct_nonempty_unresolved_changes', 'completed_returns',
          'nonempty_resolved_returns', 'completed_passages')
SCOPE = ('Descriptive conditional-target comparisons at saved block cadence. '
         'Presence-vector ESS is not categorical environment ESS or an equilibrium guarantee. '
         'All streams and undefined ESS are retained. Occupancy uses 4096 production endpoints; '
         'event counts include the block-512 boundary and rates use full sampler CPU, including warmup. '
         'Equal initialization means, empty traces or equal event rates do not establish mixing. '
         'Pairwise initialization distances share trajectories and are dependent descriptors; '
         'the four streams within a start, not the 16 cross-start distances, are the sampling units. '
         'No context pooling, trajectory concatenation, physical kinetics, unseen-support bound, '
         'assembly claim or requirement of equal forward/reverse event rates.')


def number(value, name, *, lower=0., upper=None):
    require(type(value) in (int, float) and math.isfinite(value) and value >= lower
            and (upper is None or value <= upper), 'Invalid '+name)
    return value


def identity(chain):
    value = chain['identity']; job = chain['job']
    require(type(value) is dict and set(value) == {'context_index', 'arm', 'initialization', 'stream'}
            and type(value['context_index']) is int and value['context_index'] in range(4)
            and value['arm'] in ARMS and value['initialization'] in STARTS
            and type(value['stream']) is int and value['stream'] in range(4)
            and all(job[k] == v for k, v in value.items()), 'Invalid native chain identity')
    require(type(chain['chain_id']) is str and bool(chain['chain_id']), 'Missing native chain ID')
    return tuple(value[k] for k in ('context_index', 'arm', 'initialization', 'stream'))


def descriptor_summary(value, cpu):
    ess = copy.deepcopy(value['presence_ess'])
    require(ess['samples'] == 4096 and ess['sampling_CPU_seconds'] == cpu, 'Presence ESS denominator differs')
    estimate, rate = ess['apparent_ess'], ess['apparent_ess_per_sampling_CPU_second']
    if estimate is None:
        require(rate is None and type(ess.get('reason')) is str and bool(ess['reason']), 'Undefined ESS needs its reason')
    else:
        number(estimate, 'presence ESS', upper=4096)
        require(estimate > 0 and math.isclose(number(rate, 'presence ESS rate'), estimate/cpu,
                    rel_tol=1e-12, abs_tol=0.), 'Presence ESS rate uses a different CPU denominator')
    occupancy = dict(value['environments']['occupancy'])
    require(occupancy and all(type(label) is str for label in occupancy), 'Missing production environment occupancy')
    for label, fraction in occupancy.items():
        number(fraction, 'environment occupancy', upper=1.)
        require(label in value['environment_values'] and key(value['environment_values'][label]) == label,
                'Environment label/value mismatch')
    require(math.isclose(math.fsum(occupancy.values()), 1., rel_tol=0., abs_tol=1e-12), 'Environment occupancy mass differs')
    positive = [label for label, fraction in occupancy.items() if fraction > 0]
    require(len(positive) != 1 or estimate is None, 'Constant descriptor must retain undefined ESS')
    marginal = copy.deepcopy(value['marginal_occupancy']); seen = set()
    for row in marginal:
        token = tuple(row['key'])
        require(token not in seen, 'Repeated marginal key'); seen.add(token)
        number(row['fraction'], 'marginal occupancy', upper=1.)
    counts = {name: value['counts'][name] for name in COUNTS}
    rates = {name: value['rates_per_full_sampler_cpu_second'][name] for name in COUNTS}
    for name, count in counts.items():
        require(type(count) is int and 0 <= count <= 4096, 'Invalid native event count')
        require(math.isclose(number(rates[name], 'native event rate'), count/cpu, rel_tol=1e-12, abs_tol=0.),
                'Native event rate uses a different CPU denominator')
    return dict(presence_ess=ess, environment_occupancy=occupancy, marginal_occupancy=marginal,
                counts=counts, rates_per_full_sampler_cpu_second=rates,
                constant_production_environment=len(positive) == 1,
                nonempty_production_fraction=math.fsum(occupancy[label] for label in positive
                    if value['environment_values'][label]),
                nonempty_production_environments=sum(bool(value['environment_values'][label]) for label in positive))


def stream_summary(chain):
    metrics = chain['metrics']; cpu = number(metrics['full_sampler_cpu_seconds'], 'full sampler CPU')
    require(cpu > 0 and metrics['schema'] == 'conditional-native-metrics-v1'
            and metrics['production_samples'] == 4096 and metrics['retained_endpoints'] == 4609,
            'Native chain allocation/schema differs')
    fractions = {name: number(metrics[name], name, upper=1.) for name in FRACTIONS}
    isolate = dict(metrics['mobile_native_isolate_fraction_by_member'])
    require(len(isolate) == 2, 'Two mobile isolate fractions required')
    for value in isolate.values(): number(value, 'mobile isolate fraction', upper=1.)
    return dict(chain_id=chain['chain_id'], identity=copy.deepcopy(chain['identity']),
                full_sampler_cpu_seconds=cpu, fractions=fractions,
                mobile_native_isolate_fraction_by_member=isolate,
                descriptors={name: descriptor_summary(metrics[name], cpu) for name in DESCRIPTORS},
                no_production_native_attachment=fractions['any_native_attachment_fraction'] == 0.)


def difference(a, b):
    return None if a is None or b is None else b-a


def compare_descriptors(a, b):
    return dict(environment_total_variation=total_variation(a['environment_occupancy'], b['environment_occupancy']),
                marginal_occupancy_max_difference=occupancy_difference(a['marginal_occupancy'], b['marginal_occupancy'], 'key'),
                presence_ess_per_full_sampler_cpu_second_difference=difference(
                    a['presence_ess']['apparent_ess_per_sampling_CPU_second'],
                    b['presence_ess']['apparent_ess_per_sampling_CPU_second']),
                rates_per_full_sampler_cpu_second_difference={name:
                    b['rates_per_full_sampler_cpu_second'][name]-a['rates_per_full_sampler_cpu_second'][name]
                    for name in COUNTS})


def mean_occupancy(rows, field):
    """Equal stream weights; never concatenate trajectories or average ESS."""
    maps = [row[field] if field == 'environment_occupancy' else
            {tuple(item['key']): item['fraction'] for item in row[field]} for row in rows]
    mean = {label: math.fsum(row.get(label, 0.) for row in maps)/len(maps)
            for label in sorted(set().union(*(row.keys() for row in maps)))}
    return mean if field == 'environment_occupancy' else [dict(key=k, fraction=v) for k, v in mean.items()]


def initialization_summary(context, arm, left, right):
    result = dict(context_index=context, arm=arm,
                  source_chain_ids=[row['chain_id'] for row in left],
                  proposal_prepared_chain_ids=[row['chain_id'] for row in right],
                  rng_paired=False, descriptors={},
                  fraction_mean_comparisons={name: dict(source_mean=math.fsum(r['fractions'][name] for r in left)/4,
                    proposal_prepared_mean=math.fsum(r['fractions'][name] for r in right)/4,
                    difference=math.fsum(r['fractions'][name] for r in right)/4-math.fsum(r['fractions'][name] for r in left)/4)
                    for name in FRACTIONS},
                  both_starts_all_empty=all(r['no_production_native_attachment'] for r in left+right),
                  pairwise_distances_are_independent=False,
                  initialization_agreement_establishes_mixing=False)
    for name in DESCRIPTORS:
        a = [r['descriptors'][name] for r in left]; b = [r['descriptors'][name] for r in right]
        mean_a, mean_b = (mean_occupancy(rows, 'environment_occupancy') for rows in (a, b))
        marginal_a, marginal_b = (mean_occupancy(rows, 'marginal_occupancy') for rows in (a, b))
        def pair(i, j, first, second):
            x, y = first[i], second[j]
            return dict(left_stream=i, right_stream=j,
                environment_total_variation=total_variation(x['environment_occupancy'], y['environment_occupancy']),
                marginal_occupancy_max_difference=occupancy_difference(x['marginal_occupancy'], y['marginal_occupancy'], 'key'))
        result['descriptors'][name] = dict(source_mean_occupancy=mean_a, proposal_prepared_mean_occupancy=mean_b,
            mean_environment_total_variation=total_variation(mean_a, mean_b),
            mean_marginal_occupancy_max_difference=occupancy_difference(marginal_a, marginal_b, 'key'),
            cross_start_comparisons=[pair(i, j, a, b) for i, j in itertools.product(range(4), repeat=2)],
            within_source_comparisons=[pair(i, j, a, a) for i, j in itertools.combinations(range(4), 2)],
            within_proposal_prepared_comparisons=[pair(i, j, b, b) for i, j in itertools.combinations(range(4), 2)],
            both_starts_all_constant=all(r['constant_production_environment'] for r in a+b),
            both_starts_all_empty=all(r['nonempty_production_fraction'] == 0. for r in a+b),
            no_nonempty_resolved_returns=all(r['counts']['nonempty_resolved_returns'] == 0 for r in a+b),
            streams_without_defined_presence_ess={start: [i for i, r in enumerate(rows) if r['presence_ess']['apparent_ess'] is None]
                for start, rows in zip(STARTS, (a, b))})
    return result


def reduce_native_comparisons(chains):
    """Validate the fixed inventory and compare existing per-chain summaries."""
    require(type(chains) is list and len(chains) == 128, 'Exactly 128 native chains required')
    indexed = {identity(chain): chain for chain in chains}
    expected = set(itertools.product(range(4), ARMS, STARTS, range(4)))
    require(set(indexed) == expected and len({c['chain_id'] for c in chains}) == 128,
            'Native comparison inventory is missing or duplicated')
    rows = {ident: stream_summary(chain) for ident, chain in indexed.items()}
    groups = []
    for context, arm, start in itertools.product(range(4), ARMS, STARTS):
        streams = [rows[context, arm, start, stream] for stream in range(4)]
        groups.append(dict(context_index=context, arm=arm, initialization=start,
            independent_stream_metrics=streams,
            presence_ess_defined_streams={name: sum(row['descriptors'][name]['presence_ess']['apparent_ess'] is not None
                for row in streams) for name in DESCRIPTORS},
            presence_ess_undefined_streams={name: sum(row['descriptors'][name]['presence_ess']['apparent_ess'] is None
                for row in streams) for name in DESCRIPTORS}))
    contrasts = []
    for context, start, stream in itertools.product(range(4), STARTS, range(4)):
        for left_arm, right_arm in itertools.combinations(ARMS, 2):
            a, b = (rows[context, arm, start, stream] for arm in (left_arm, right_arm))
            require(set(a['mobile_native_isolate_fraction_by_member']) == set(b['mobile_native_isolate_fraction_by_member']),
                    'Conditional mobile labels differ')
            contrasts.append(dict(context_index=context, initialization=start, stream=stream,
                left_chain_id=a['chain_id'], right_chain_id=b['chain_id'], left_arm=left_arm, right_arm=right_arm,
                difference_direction='right minus left', local_rng_roles_paired=True,
                collective_rng_roles_paired=left_arm.startswith('singleton_') and right_arm.startswith('singleton_'),
                fraction_differences={name: b['fractions'][name]-a['fractions'][name] for name in FRACTIONS},
                mobile_isolate_fraction_differences={member: b['mobile_native_isolate_fraction_by_member'][member]-value
                    for member, value in a['mobile_native_isolate_fraction_by_member'].items()},
                descriptors={name: compare_descriptors(a['descriptors'][name], b['descriptors'][name]) for name in DESCRIPTORS}))
    initializations = [initialization_summary(context, arm,
        [rows[context, arm, 'source', stream] for stream in range(4)],
        [rows[context, arm, 'proposal_prepared', stream] for stream in range(4)])
        for context, arm in itertools.product(range(4), ARMS)]
    return dict(schema=SCHEMA, complete=True, groups=groups, arm_contrasts=contrasts,
                initialization_agreement=initializations, input_chains=128,
                production_samples_per_chain=4096, context_pooling=False, ess_reestimated=False,
                scope=SCOPE, equilibrium_established=False, assembly_gate_open=False)
