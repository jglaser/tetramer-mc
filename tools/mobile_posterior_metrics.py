"""Pure graph-history diagnostics for the frozen mobile-posterior pilot.

Every attempted physical update, including self-loops, is retained. Occupancy
and apparent ESS use the final observation of each postburn sweep. The native
graph carries entry/retention hysteresis: these path-dependent descriptors are
not instantaneous q<=1 region probabilities, equilibrium or physical rates.
"""
from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np


KINDS = ('native', 'nonspecific')
EXCHANGE_RULE = (
    'For each body and lost partner, retain one pending loss. Complete one '
    'exchange at the first strictly later attempted update that gains a '
    'different partner while the lost partner remains absent; choose the '
    'smallest gained partner ID if several arrive together. Reattachment of '
    'the lost partner first cancels that pending loss. One gain may complete '
    'several pending losses. Losses accompanied by any same-update gain are '
    'consumed by the separately reported replacements and never enter pending '
    'sequential losses. Registry changes without an edge loss never count.'
)


def apparent_effective_count(values, cpu):
    """FFT/Geyer trace diagnostic, copied from analyze_posterior_docking_pilot.

    The estimator matches that module's effective_count/positive_monotone_iact
    without importing its physical-audit and archived-data dependencies.
    """
    if not math.isfinite(cpu) or cpu <= 0:
        raise ValueError('Sampler CPU must be finite and positive')
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Apparent ESS requires a finite scalar/vector record')
    n = len(x)
    common = dict(samples=n, sampling_CPU_seconds=cpu,
                  scope='Sample-centered finite-record descriptor only; stationarity, coverage and equilibrium ESS are not established.')
    if n < 2:
        return dict(common, iact_samples=None, apparent_ess=None, apparent_ess_per_sampling_CPU_second=None,
                    reason='Fewer than two observed sweep endpoints; mixing unresolved.')
    x = x-x.mean(axis=0)
    size = 1 << (2*n-1).bit_length()
    power = np.abs(np.fft.rfft(x, n=size, axis=0))**2
    covariance = np.fft.irfft(power.sum(axis=1), n=size)[:n]/n
    if covariance[0] < 1e-24:
        return dict(common, iact_samples=None, apparent_ess=None, apparent_ess_per_sampling_CPU_second=None,
                    reason='Constant observed descriptor; unresolved mixing, not infinite ESS.')
    rho = covariance/covariance[0]
    pairs, last = [], float('inf')
    for lag in range(0, min(len(rho)-1, max(1, n//2)), 2):
        pair = float(rho[lag]+rho[lag+1])
        if pair <= 0:
            break
        pair = min(pair, last)
        pairs.append(pair)
        last = pair
    raw = -1+2*sum(pairs)
    tau = max(1., raw)
    ess = n/tau
    return dict(common, iact_samples=tau, unfloored_iact_samples=raw,
                apparent_ess=ess, apparent_ess_per_sampling_CPU_second=ess/cpu,
                last_included_lag=2*len(pairs)-1,
                estimator='Biased sample-centered FFT autocovariance; sum vector component covariances; Geyer positive monotone lag pairs (0,1),(2,3),... capped at half the record, IACT floor 1.')


def _canonical_edges(edges, body_count):
    result = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError('Each edge must contain two body IDs')
        a, b = edge
        if any(not isinstance(i, int) or isinstance(i, bool) for i in (a, b)):
            raise ValueError('Body IDs must be integers')
        if not (0 <= a < body_count and 0 <= b < body_count) or a == b:
            raise ValueError('Invalid body edge')
        key = tuple(sorted((a, b)))
        if key in result:
            raise ValueError('Duplicate unordered body edge')
        result.add(key)
    return result


def _largest_component(body_count, edges):
    neighbors = [set() for _ in range(body_count)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    unseen, largest = set(range(body_count)), 0
    while unseen:
        stack = [unseen.pop()]
        size = 0
        while stack:
            body = stack.pop()
            size += 1
            new = neighbors[body] & unseen
            unseen.difference_update(new)
            stack.extend(new)
        largest = max(largest, size)
    return largest


def _occupancy(indices, axis, bits, edge_labels, body_count):
    n = len(indices)
    selected = bits[indices]
    degree = np.zeros((n, body_count), dtype=int)
    largest = []
    for row, flags in enumerate(selected):
        present = [edge for edge, active in zip(edge_labels, flags) if active]
        for a, b in present:
            degree[row, a] += 1
            degree[row, b] += 1
        largest.append(_largest_component(body_count, present))
    fraction = lambda flags: float(np.mean(flags)) if n else None
    return dict(samples=n,
                first_sweep=axis[indices[0]]['sweep'] if n else None,
                last_sweep=axis[indices[-1]]['sweep'] if n else None,
                edge_occupancy=[dict(bodies=list(edge), fraction=fraction(selected[:, k]))
                                for k, edge in enumerate(edge_labels)],
                degree_by_body=[dict(body=body, probabilities={str(k): fraction(degree[:, body] == k)
                                                               for k in range(body_count)})
                                for body in range(body_count)],
                edge_count_probabilities={str(k): fraction(selected.sum(axis=1) == k)
                                          for k in range(len(edge_labels)+1)},
                largest_component_size_probabilities={str(k): fraction(np.asarray(largest) == k)
                                                       for k in range(1, body_count+1)})


def _episodes(axis, neighbors, start_index):
    result = []
    first = start_index

    def append(last, exit_index):
        start, final = axis[first], axis[last]
        exit_row = axis[exit_index] if exit_index is not None else None
        end = final if exit_row is None else exit_row
        result.append(dict(
            neighbors=list(neighbors[first]),
            first_observation_index=first, last_observation_index=last,
            exit_observation_index=exit_index,
            first_observed_serial=start['serial'], first_observed_sweep=start['sweep'],
            last_observed_serial=final['serial'], last_observed_sweep=final['sweep'],
            exit_serial=None if exit_row is None else exit_row['serial'],
            exit_sweep=None if exit_row is None else exit_row['sweep'],
            entry_source=start['source'], exit_source=None if exit_row is None else exit_row['source'],
            attempted_update_dwell=end['serial']-start['serial'],
            sweep_dwell=end['sweep']-start['sweep'],
            observed_attempt_span=final['serial']-start['serial'],
            observations=last-first+1,
            left_censored=first == start_index, right_censored=exit_row is None))

    for index in range(start_index+1, len(axis)):
        if neighbors[index] != neighbors[first]:
            append(index-1, index)
            first = index
    append(len(axis)-1, None)
    seen, returns = set(), 0
    for episode in result:
        value = tuple(episode['neighbors'])
        returns += int(value in seen)
        seen.add(value)
    return dict(episodes=result, observed_environment_returns=returns)


def _exchanges(axis, body_neighbors, body):
    pending, events = {}, []
    for index in range(1, len(axis)):
        before, after = set(body_neighbors[index-1]), set(body_neighbors[index])
        gained, lost = sorted(after-before), sorted(before-after)
        # Resolve only losses from an earlier update. Adding current losses
        # afterward makes the strictly-later condition structural.
        for old_partner, loss_index in list(pending.items()):
            if old_partner in after:
                del pending[old_partner]
            elif gained:
                new_partner = gained[0]
                loss, gain = axis[loss_index], axis[index]
                events.append(dict(
                    body=body, lost_partner=old_partner, gained_partner=new_partner,
                    serial=gain['serial'], sweep=gain['sweep'], source=gain['source'],
                    loss_serial=loss['serial'], loss_sweep=loss['sweep'], loss_source=loss['source'],
                    gain_serial=gain['serial'], gain_sweep=gain['sweep'], gain_source=gain['source'],
                    attempted_updates_between_loss_and_gain=gain['serial']-loss['serial'],
                    sweeps_between_loss_and_gain=gain['sweep']-loss['sweep'],
                    neighbors_before_loss=list(body_neighbors[loss_index-1]),
                    neighbors_after_loss=list(body_neighbors[loss_index]),
                    neighbors_before_gain=list(body_neighbors[index-1]),
                    neighbors_after_gain=list(body_neighbors[index]),
                    intervening_neighbor_states=[dict(axis[k], neighbors=list(body_neighbors[k]))
                                                 for k in range(loss_index, index+1)]))
                del pending[old_partner]
        # Every current loss is already paired with every current gain in the
        # separate replacement list. Never reuse those detachments later.
        if not gained:
            for partner in lost:
                pending[partner] = index
    unresolved = [dict(body=body, lost_partner=partner,
                       loss_serial=axis[index]['serial'], loss_sweep=axis[index]['sweep'],
                       loss_source=axis[index]['source'], right_censored=True)
                  for partner, index in sorted(pending.items())]
    return events, unresolved


def _same_attempt_replacements(axis, body_neighbors, body):
    events = []
    for index in range(1, len(axis)):
        before, after = set(body_neighbors[index-1]), set(body_neighbors[index])
        for lost in sorted(before-after):
            for gained in sorted(after-before):
                events.append(dict(axis[index], body=body, lost_partner=lost, gained_partner=gained,
                                   neighbors_before=list(body_neighbors[index-1]),
                                   neighbors_after=list(body_neighbors[index]),
                                   category='same_attempt_partner_replacement'))
    return events


def _rates(axis, indices, events, keys, cpu=None):
    sources = Counter(axis[index]['source'] for index in indices)
    n = len(indices)
    counts = {key: sum(len(event[key]) for event in events) for key in keys}
    by_source = {}
    for source, denominator in sorted(sources.items()):
        observed = {key: sum(len(event[key]) for event in events if event['source'] == source)
                    for key in keys}
        by_source[source] = dict(attempted_updates=denominator, counts=observed,
                                 per_source_attempted_update={key: value/denominator for key, value in observed.items()},
                                 per_all_attempted_updates={key: value/n for key, value in observed.items()})
    return dict(attempted_updates=n, counts=counts,
                per_attempted_update={key: value/n if n else None for key, value in counts.items()},
                sampling_CPU_seconds=cpu,
                per_sampling_CPU_second=None if cpu is None else {key: value/cpu for key, value in counts.items()},
                by_source=by_source)


def summarize_graph_history(history, body_count, burn_sweeps, total_sweeps, postburn_cpu):
    """Summarize an initial state followed by every physical attempted update.

    Initial row: serial=-1, sweep=0, source='initial'. Later serials are exactly
    0,1,..., and sweeps are consecutive, with one or more rows per sweep. Each
    row supplies native_edges and nonspecific_edges. Extra fields are ignored.
    The input is never modified; initial bonds are occupancy, not formations.
    """
    if not isinstance(body_count, int) or isinstance(body_count, bool) or body_count < 2:
        raise ValueError('At least two bodies are required')
    if not (isinstance(burn_sweeps, int) and isinstance(total_sweeps, int)
            and 0 <= burn_sweeps < total_sweeps):
        raise ValueError('Require 0 <= burn_sweeps < total_sweeps')
    if not math.isfinite(postburn_cpu) or postburn_cpu <= 0:
        raise ValueError('Postburn sampler CPU must be finite and positive')
    if len(history) < 2:
        raise ValueError('Initial state and attempted updates are required')
    axis, graphs, endpoints = [], {kind: [] for kind in KINDS}, {}
    previous_sweep = 0
    for index, row in enumerate(history):
        serial, sweep, source = row['serial'], row['sweep'], row['source']
        if serial != index-1 or not isinstance(serial, int) or isinstance(serial, bool):
            raise ValueError('Every attempted update must have a contiguous serial')
        if not isinstance(sweep, int) or isinstance(sweep, bool):
            raise ValueError('Sweep indices must be integers')
        if not isinstance(source, str) or not source:
            raise ValueError('Each observation needs a source label')
        if index == 0:
            if sweep != 0 or source != 'initial':
                raise ValueError('First observation must be the initial state')
        elif not (1 <= sweep <= total_sweeps and previous_sweep <= sweep <= previous_sweep+1):
            raise ValueError('Attempted updates must cover consecutive ordered sweeps')
        axis.append(dict(serial=serial, sweep=sweep, source=source))
        endpoints[sweep] = index
        previous_sweep = sweep
        for kind in KINDS:
            graphs[kind].append(_canonical_edges(row[kind+'_edges'], body_count))
            if index and row.get('accepted') is False and graphs[kind][-1] != graphs[kind][-2]:
                raise ValueError('Rejected physical update changed a graph')
    if previous_sweep != total_sweeps or set(endpoints) != set(range(total_sweeps+1)):
        raise ValueError('History must include every declared sweep')

    labels = list(itertools.combinations(range(body_count), 2))
    retained = [endpoints[sweep] for sweep in range(burn_sweeps+1, total_sweeps+1)]
    all_attempts = list(range(1, len(axis)))
    post_attempts = [index for index in all_attempts if axis[index]['sweep'] > burn_sweeps]
    baseline = endpoints[burn_sweeps]
    split = len(retained)//2
    summaries, individual, joint = {}, [], []
    for kind in KINDS:
        graph = graphs[kind]
        bits = np.asarray([[edge in edges for edge in labels] for edges in graph], dtype=bool)
        events = []
        for index in range(1, len(axis)):
            formed, detached = graph[index]-graph[index-1], graph[index-1]-graph[index]
            if formed or detached:
                events.append(dict(axis[index],
                                   formed_edges=[list(edge) for edge in sorted(formed)],
                                   detached_edges=[list(edge) for edge in sorted(detached)],
                                   before_edges=[list(edge) for edge in sorted(graph[index-1])],
                                   after_edges=[list(edge) for edge in sorted(graph[index])]))
        post_events = [event for event in events if event['sweep'] > burn_sweeps]
        bodies, exchanges, unresolved, replacements = [], [], [], []
        for body in range(body_count):
            neighbors = [tuple(sorted(b if a == body else a for a, b in edges if body in (a, b)))
                         for edges in graph]
            body_exchanges, body_unresolved = _exchanges(axis, neighbors, body)
            exchanges.extend(body_exchanges)
            unresolved.extend(body_unresolved)
            replacements.extend(_same_attempt_replacements(axis, neighbors, body))
            bodies.append(dict(body=body, neighbors_by_observation=[list(value) for value in neighbors],
                               degree_by_observation=[len(value) for value in neighbors],
                               episodes=dict(full=_episodes(axis, neighbors, 0),
                                             postburn=_episodes(axis, neighbors, baseline))))
        exchanges.sort(key=lambda event: (event['gain_serial'], event['body'], event['lost_partner'], event['gained_partner']))
        replacements.sort(key=lambda event: (event['serial'], event['body'], event['lost_partner'], event['gained_partner']))
        for event in exchanges:
            event['loss_precedes_postburn_window'] = event['loss_sweep'] <= burn_sweeps
        # Reuse the event-rate machinery with one count per completed exchange.
        exchange_counts = [dict(event, completed_exchanges=[(event['body'], event['lost_partner'], event['gained_partner'])])
                           for event in exchanges]
        post_exchanges = [event for event in exchange_counts if event['gain_sweep'] > burn_sweeps]
        replacement_counts = [dict(event, same_attempt_replacements=[(event['body'], event['lost_partner'], event['gained_partner'])])
                              for event in replacements]
        post_replacements = [event for event in replacement_counts if event['sweep'] > burn_sweeps]
        retained_bits = bits[retained]
        largest = np.asarray([_largest_component(body_count, graph[index]) for index in retained])
        indicators = np.column_stack([largest == size for size in range(1, body_count+1)])
        indicator_ess = dict(
            any_edge=apparent_effective_count(retained_bits.any(axis=1), postburn_cpu),
            connected=apparent_effective_count(largest == body_count, postburn_cpu),
            largest_component_size={str(size): apparent_effective_count(largest == size, postburn_cpu)
                                    for size in range(1, body_count+1)},
            largest_component_indicators_joint=apparent_effective_count(indicators, postburn_cpu))
        for column, edge in enumerate(labels):
            individual.append(dict(kind=kind, bodies=list(edge),
                                   **apparent_effective_count(retained_bits[:, column], postburn_cpu)))
        joint.append(retained_bits)
        summaries[kind] = dict(
            initial_edges=[list(edge) for edge in sorted(graph[0])],
            edge_labels=[list(edge) for edge in labels],
            edge_presence_by_observation=bits.astype(int).tolist(),
            events=events,
            event_rates=dict(full=_rates(axis, all_attempts, events, ('formed_edges', 'detached_edges')),
                             postburn=_rates(axis, post_attempts, post_events, ('formed_edges', 'detached_edges'), postburn_cpu)),
            partner_exchanges=exchanges,
            same_attempt_partner_replacements=replacements,
            pending_partner_losses_at_end=unresolved,
            exchange_rates=dict(full=_rates(axis, all_attempts, exchange_counts, ('completed_exchanges',)),
                                postburn=_rates(axis, post_attempts, post_exchanges, ('completed_exchanges',), postburn_cpu),
                                completed_postburn_with_preburn_loss=sum(event['loss_precedes_postburn_window'] for event in post_exchanges)),
            replacement_rates=dict(full=_rates(axis, all_attempts, replacement_counts, ('same_attempt_replacements',)),
                                   postburn=_rates(axis, post_attempts, post_replacements, ('same_attempt_replacements',), postburn_cpu)),
            bodies=bodies,
            occupancy=dict(postburn=_occupancy(retained, axis, bits, labels, body_count),
                           first_half=_occupancy(retained[:split], axis, bits, labels, body_count),
                           second_half=_occupancy(retained[split:], axis, bits, labels, body_count)),
            graph_indicator_apparent_ess=indicator_ess)
    return dict(
        schema='mobile-posterior-graph-history-v1', body_count=body_count,
        observation_axis=axis, graphs=summaries,
        individual_edge_apparent_ess=individual,
        joint_edge_apparent_ess=apparent_effective_count(np.column_stack(joint), postburn_cpu),
        postburn_sampling=dict(burn_sweeps=burn_sweeps, total_sweeps=total_sweeps,
                               retained_sweeps=list(range(burn_sweeps+1, total_sweeps+1)),
                               endpoint_observation_indices=retained,
                               baseline_observation_index=baseline,
                               physical_attempts=len(post_attempts), sampling_CPU_seconds=postburn_cpu,
                               half_split='First floor(N/2) retained sweep endpoints, then the remaining endpoints.'),
        rules=dict(partner_exchange=EXCHANGE_RULE,
                   same_attempt_replacement='Separately label each (body, lost partner, gained partner) combination when one physical update removes and adds incident edges. These events retain before/after neighbor identities and source. Current losses consumed by these replacements never enter pending sequential losses; older pending losses can still resolve on the current gain. If accepted=false is supplied, changing either graph is rejected as inconsistent.',
                   native_graph='The input native graph carries entry/retention hysteresis. Its occupancy and ESS describe a path-dependent graph process, not an instantaneous q<=1 thermodynamic region probability. Instantaneous entry-only observables must be reported separately.',
                   events='Graph edges only; initial bonds are not formations. All local/global/collective attempted updates and self-loops enter denominators. Multiple edge changes in one update count separately.',
                   residence='Per-body neighbor-set episodes. First episode in each full/postburn window is left censored; unfinished last episode is right censored. Dwell ends at the exit update, or final observation if censored. Sweep differences may be zero for within-sweep episodes; serial differences count physical attempted updates.',
                   postburn_exchange='Count completions after burn; separately report those whose observed loss was at/before burn. Every loss/gain identity and all intervening attempted-update neighbor states are retained.',
                   sampling='Occupancies and apparent ESS use exactly the last state of each retained sweep, without thinning or removal of repeats. Event and residence histories retain every physical attempted update.',
                   cpu='Every postburn event rate and apparent ESS uses the supplied full postburn sampler CPU, including all kernels, rejections and logging. Full-history rates use attempts only because no full-history CPU was supplied.'),
        scope='Descriptive native-informed preparation pilot. Native entry/retention hysteresis makes its graph observables path dependent; they are not instantaneous q<=1 thermodynamic region probabilities. Apparent ESS does not establish stationarity, equilibrium, coverage, independent preparations, physical time, or sampling speedup.')
