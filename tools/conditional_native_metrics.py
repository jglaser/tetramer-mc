"""Pure retained-endpoint extraction and conditional native-history reduction.

The future caller MUST authenticate a completed prior arithmetic-replay receipt
and its exact journal bytes before using the reader. This module neither checks
that receipt nor replays elementary proposals, hard validity, or bath arithmetic.
There is no file IO, classification, population pooling, or equilibrium claim.
"""
from __future__ import annotations

import copy
import math

from analyze_evolving_dimer_benchmark import (categorical_summary, key,
                                             presence_matrix, require)
from mobile_posterior_metrics import apparent_effective_count
from native_contact_regions import pose_arrays


ELEMENTARY_KINDS = frozenset(('local', 'factorized_dimer', 'two_neighbor_singleton'))
SCOPE = ('One conditional chain at saved block cadence, retaining all production '
         'residence. Native labels are instantaneous entry descriptors. Empty '
         'registry is retained; catalogue consistency alone is not attachment. '
         'No context pooling, trajectory concatenation, physical kinetics, '
         'unseen-state bound, or equilibrium inference. Events within a block '
         'are unresolved; all rates use full sampler CPU including warmup.')


def _allocation(blocks, warmup):
    require(type(blocks) is int and type(warmup) is int
            and 0 <= warmup < blocks, 'Require integer 0 <= warmup < blocks')


def _cpu(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Sampler CPU must be finite and nonnegative')
    return value


def _selected(value):
    require(type(value) is list and len(value) == 2, 'Exactly two retained poses required')
    for pose in value:
        pose_arrays(pose)


def iter_authenticated_endpoints(rows, *, job, initial, blocks=4608, warmup=512):
    """Extract all endpoints from an already authenticated decoded-row iterator.

    Exhaust this iterator before declaring successful completion. A truncated
    tail raises after earlier endpoints have been yielded. Known elementary
    records are skipped without examining proposal arithmetic; their validation
    belongs to the completed, byte-bound prior receipt. Initial records in the
    existing format omit ``production``; if present it must be false.
    """
    _allocation(blocks, warmup)
    require(type(job) is dict and job, 'Nonempty expected job identity required')
    _selected(initial)
    expected = 0
    previous_cpu = None
    for row in rows:
        require(type(row) is dict, 'Journal record must be an object')
        kind = row.get('kind')
        if kind in ELEMENTARY_KINDS:
            require(0 < expected <= blocks, 'Elementary record outside retained block sequence')
            continue
        require(expected <= blocks, 'Unexpected journal tail')
        require(kind == ('initial' if expected == 0 else 'retained_block'),
                'Expected initial followed by every retained block')
        require(type(row.get('block')) is int and row['block'] == expected,
                'Missing, repeated or out-of-order retained block')
        if expected == 0:
            require(row.get('job') == job and row.get('selected') == initial
                    and row.get('conditional_target') is True, 'Initial job/poses/target differ')
            require('production' not in row or row['production'] is False,
                    'Initial endpoint cannot be production')
        else:
            require(row.get('production') is (expected > warmup), 'Production flag differs')
            require('job' not in row or row['job'] == job, 'Retained job identity differs')
        _selected(row.get('selected'))
        cpu = _cpu(row.get('sampler_cpu_seconds'))
        require(previous_cpu is None or cpu >= previous_cpu, 'Nonmonotone retained CPU')
        previous_cpu = cpu
        yield dict(kind=kind, block=expected, production=expected > warmup,
                   selected=copy.deepcopy(row['selected']), sampler_cpu_seconds=cpu)
        expected += 1
    require(expected == blocks+1, 'Incomplete initial/retained endpoint inventory')


def _keys(value):
    require(type(value) in (list, tuple), 'Native keys must be a sequence')
    result = []
    for item in value:
        require(type(item) in (list, tuple) and len(item) == 3
                and all(type(i) is int for i in item)
                and 0 <= item[0] < item[1], 'Invalid canonical native key')
        result.append(tuple(item))
    require(len(set(result)) == len(result), 'Repeated native key')
    return tuple(sorted(result))


def _descriptor(row, members):
    external = _keys(row['external_native_keys'])
    internal = _keys(row['internal_native_keys'])
    mobile = _keys(row['mobile_related_native_keys'])
    fixed = _keys(row['fixed_native_keys'])
    all_keys = _keys(row['instantaneous_native_keys'])
    require(all(len(set(k[:2]) & members) == 1 for k in external), 'Invalid external native key')
    require(all(set(k[:2]) == members for k in internal), 'Invalid internal native key')
    require(set(mobile) == set(external) | set(internal), 'Mobile native partition differs')
    require(all(not (set(k[:2]) & members) for k in fixed), 'Fixed key involves a mobile body')
    require(set(all_keys) == set(fixed) | set(mobile), 'Full native key union differs')
    components = row['mobile_native_components']
    require(type(components) is list and components, 'Mobile components required, including isolates')
    covered = set()
    for component in components:
        bodies = component['bodies']
        require(type(bodies) is list and bodies and all(type(i) is int and i >= 0 for i in bodies)
                and len(set(bodies)) == len(bodies) and not (set(bodies) & covered)
                and bool(set(bodies) & members), 'Invalid or overlapping mobile components')
        covered.update(bodies)
        require(type(component['catalogue_consistent']) is bool
                and type(component['independent_cycles']) is int
                and component['independent_cycles'] >= 0, 'Invalid component consistency/cycle count')
    require(members <= covered, 'Missing mobile component or isolate')
    resolved = row['mobile_registry_resolved']
    require(type(resolved) is bool and resolved is all(c['catalogue_consistent'] for c in components),
            'Mobile consistency aggregate differs')
    return dict(external=external, partners=tuple(sorted({k[:2] for k in external})),
                internal=internal, mobile=mobile, fixed=fixed, resolved=resolved,
                components=copy.deepcopy(components))


def _environment_metrics(values, resolved, blocks, cpu):
    """O(N) observed transitions/returns; no enumeration of environment pairs.

    values[0] is the last warmup endpoint, or initial when warmup=0. It supplies
    an event boundary only: occupancy and ESS exclude it. A return is re-entry
    after an observed departure and may traverse empty/unresolved registry.
    """
    labels = [key(value) for value in values]
    events = categorical_summary(labels, blocks)
    occupancy = categorical_summary(labels[1:], blocks[1:])['occupancy']
    events['occupancy'] = occupancy
    index = {block: i for i, block in enumerate(blocks)}
    empty_prefix, unresolved_prefix = [0], [0]
    for value, status in zip(values, resolved):
        empty_prefix.append(empty_prefix[-1]+int(not value))
        unresolved_prefix.append(unresolved_prefix[-1]+int(not status))
    counts = dict(enter_nonempty=0, leave_nonempty=0,
                  direct_nonempty_resolved_changes=0, direct_nonempty_unresolved_changes=0)
    for i in range(1, len(values)):
        if values[i] == values[i-1]:
            continue
        if not values[i-1]:
            counts['enter_nonempty'] += 1
        elif not values[i]:
            counts['leave_nonempty'] += 1
        elif resolved[i-1] and resolved[i]:
            counts['direct_nonempty_resolved_changes'] += 1
        else:
            counts['direct_nonempty_unresolved_changes'] += 1
    for event in events['completed_returns']:
        a, b = index[event['departure_block']], index[event['return_block']]
        event.update(returned_environment_nonempty=bool(values[b]),
                     return_endpoints_resolved=resolved[a-1] and resolved[b],
                     passed_through_empty=empty_prefix[b] > empty_prefix[a],
                     passed_through_unresolved=unresolved_prefix[b] > unresolved_prefix[a])
    returns = events['completed_returns']
    counts['completed_returns'] = len(returns)
    counts['nonempty_resolved_returns'] = sum(e['returned_environment_nonempty']
        and e['return_endpoints_resolved'] for e in returns)
    counts['completed_passages'] = events['completed_passages']
    matrix, columns = presence_matrix(values[1:])
    return dict(environments=events,
                environment_values={label: value for label, value in zip(labels, values)},
                presence_ess=apparent_effective_count(matrix, cpu),
                marginal_occupancy=[dict(key=column, fraction=float(matrix[:, i].mean()))
                                    for i, column in enumerate(columns)],
                counts=counts, rates_per_full_sampler_cpu_second={name: count/cpu for name, count in counts.items()},
                event_scope='Events span last warmup (or initial) baseline through production. '
                    'Occupancy/ESS use production only. Direct resolved changes require two nonempty '
                    'distinct adjacent sets and both mobile graphs resolved. Returns may traverse empty '
                    'or unresolved sets, flagged individually. Initial residence is left-censored; '
                    'uncompleted departures at the end are censored. No dwell or metastability claim.')


def native_metrics(trace, *, members, full_sampler_cpu_seconds, blocks=4608, warmup=512):
    """Reduce one complete chain of endpoint/native dictionaries, without geometry.

    Each row combines the endpoint's block/production/CPU fields with the
    ConditionalNativeObserver output. Inputs are not mutated. No comparison or
    concatenation across chains/contexts is performed here.
    """
    _allocation(blocks, warmup)
    require(type(members) in (list, tuple) and len(members) == 2
            and all(type(i) is int and i >= 0 for i in members) and len(set(members)) == 2,
            'Two distinct integer mobile labels required')
    members = set(members)
    cpu = _cpu(full_sampler_cpu_seconds)
    require(cpu > 0, 'Full sampler CPU must be positive')
    rows = list(trace)
    require(len(rows) == blocks+1, 'Complete single-chain endpoint inventory required')
    descriptors = []
    previous_cpu = None
    for block, row in enumerate(rows):
        require(type(row.get('block')) is int and row['block'] == block
                and row.get('production') is (block > warmup), 'Native endpoint sequence/production differs')
        current_cpu = _cpu(row.get('sampler_cpu_seconds'))
        require((previous_cpu is None or current_cpu >= previous_cpu) and current_cpu <= cpu,
                'Native endpoint CPU differs from full sampler CPU')
        previous_cpu = current_cpu
        descriptor = _descriptor(row, members)
        require(not descriptors or descriptor['fixed'] == descriptors[0]['fixed'], 'Fixed native baseline changed')
        descriptors.append(descriptor)
    production = descriptors[warmup+1:]
    boundary = descriptors[warmup:]
    axis = list(range(warmup, blocks+1))
    resolved = [value['resolved'] for value in boundary]
    summary = {name: _environment_metrics([value[field] for value in boundary], resolved, axis, cpu)
               for name, field in [('external_partners', 'partners'), ('external_motifs', 'external'),
                                   ('internal_motifs', 'internal')]}
    fraction = lambda predicate: sum(bool(predicate(value)) for value in production)/len(production)
    return dict(schema='conditional-native-metrics-v1', production_samples=len(production),
                retained_endpoints=len(rows), full_sampler_cpu_seconds=cpu,
                production_window_cpu_seconds=rows[-1]['sampler_cpu_seconds']-rows[warmup]['sampler_cpu_seconds'],
                initial_baseline=descriptors[0], production_boundary_baseline=descriptors[warmup],
                fixed_native_keys=descriptors[0]['fixed'],
                any_native_attachment_fraction=fraction(lambda d: d['mobile']),
                external_native_attachment_fraction=fraction(lambda d: d['external']),
                internal_native_attachment_fraction=fraction(lambda d: d['internal']),
                resolved_native_attachment_fraction=fraction(lambda d: d['mobile'] and d['resolved']),
                empty_native_registry_fraction=fraction(lambda d: not d['mobile']),
                mobile_registry_resolved_fraction=fraction(lambda d: d['resolved']),
                mobile_cycle_frustration_fraction=fraction(lambda d: not d['resolved']),
                mobile_native_isolate_fraction_by_member={str(member): fraction(
                    lambda d, member=member: not any(member in k[:2] for k in d['mobile']))
                    for member in sorted(members)},
                **summary, scope=SCOPE)
