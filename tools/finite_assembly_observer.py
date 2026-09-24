"""Instantaneous finite-assembly observables; no proposal or equilibrium claim.

The legacy observer remains byte-identical. This module composes its atom-level
contact predicate with component-wise catalogue and periodic image checks.
Unresolved native registry is retained in the exhaustive partition.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from analyze_contact_efficiency import (ContactObserver, apparent_effective_count,
    pose_arrays, reference_assessment, require, sampled_exchanges)
from native_graph_consistency import NativeGraphConsistency


REGIONS = ('dispersed', 'contact_no_entry', 'registered_small',
           'registered_eight', 'registered_growth', 'remaining')
REFERENCE_SIZE = 8
SCOPE = ('Instantaneous contact and catalogue-registry diagnostics. Eight is the '
         'supplied seed size, not an assumed critical nucleus. Apparent ESS is '
         'not a stationarity or coverage certificate. Saved passages are not '
         'physical kinetic rates. Unresolved registry is retained, never dropped.')


def _components(n, edges):
    adjacency = [set() for _ in range(n)]
    for i, j in edges:
        adjacency[i].add(j); adjacency[j].add(i)
    result, unseen = [], set(range(n))
    while unseen:
        stack, reached = [min(unseen)], set()
        while stack:
            i = stack.pop()
            if i in reached:
                continue
            reached.add(i); stack.extend(adjacency[i]-reached)
        result.append(sorted(reached)); unseen -= reached
    return result


def _edges(n, values):
    edges = set()
    for pair in values:
        require(len(pair) == 2, 'An edge requires two body IDs')
        i, j = pair
        require(type(i) is type(j) is int and 0 <= i < j < n,
                'Ordered distinct body IDs required')
        edges.add((i, j))
    return sorted(edges)


def _image_lift(component, edges, positions, lengths):
    """A lift exists iff integer image offsets close on every edge."""
    if lengths is None:
        return True
    images = {component[0]: np.zeros(3, dtype=np.int64)}
    stack = [component[0]]
    while stack:
        i = stack.pop()
        for a, b in edges:
            if i not in (a, b):
                continue
            j = b if i == a else a
            winding = -np.floor((positions[j]-positions[i])/lengths+.5).astype(np.int64)
            predicted = images[i]+winding
            if j in images:
                if not np.array_equal(images[j], predicted):
                    return False
            else:
                images[j] = predicted; stack.append(j)
    return True


def graph_summary(n, exclusion_edges, native_keys, motifs, positions,
                  box_lengths=None, reference_size=REFERENCE_SIZE):
    """Classify the entire graph, without deleting frustrated edges or bodies.

    The largest certified component is a lower bound when another component is
    unresolved. No maximal compatible subgraph is selected. Trees require no
    independent cycle test but must still satisfy every complete pair label.
    """
    require(type(n) is int and n > 0, 'Positive body count required')
    require(reference_size == REFERENCE_SIZE and type(reference_size) is int,
            'Version-one reference size is exactly eight')
    positions = np.asarray(positions, float)
    require(positions.shape == (n, 3) and np.isfinite(positions).all(), 'Invalid positions')
    lengths = None if box_lengths is None else np.asarray(box_lengths, float)
    if lengths is not None:
        require(lengths.shape == (3,) and np.isfinite(lengths).all() and np.all(lengths > 0),
                'Invalid periodic lengths')
        require(np.max(np.abs(positions/lengths)) < 2**48,
                'Periodic positions too large for a reliable image lift')
    exclusion_edges = _edges(n, exclusion_edges)
    keys = set()
    labels = {m['id'] for m in motifs}
    for key in native_keys:
        require(len(key) == 3, 'Native key requires two bodies and one motif')
        i, j, label = key
        _edges(n, [(i, j)])
        require(type(label) is int and label in labels, 'Unknown native motif')
        keys.add((i, j, label))
    keys = sorted(keys)
    native_edges = sorted({(i, j) for i, j, _ in keys})
    exclusion_components = _components(n, exclusion_edges)
    checker = NativeGraphConsistency(motifs, n)
    native_components = []
    for component in _components(n, native_edges):
        members = set(component)
        local_keys = [k for k in keys if k[0] in members]
        local_edges = [e for e in native_edges if e[0] in members]
        lift = _image_lift(component, local_edges, positions, lengths)
        # Ordinary-space catalogue closure is inapplicable to a winding graph.
        # Do not turn missing quotient-space semantics into apparent frustration.
        cycle = checker.check(local_keys) if lift else None
        consistent = cycle['consistent'] if cycle is not None else None
        native_components.append(dict(bodies=component,
            catalogue_consistent=consistent, ordinary_space_lift=lift,
            certified=consistent is True and lift,
            independent_cycles=len(local_edges)-len(component)+1,
            edges=[list(e) for e in local_edges],
            chosen_edge_labels=cycle['chosen_edge_labels'] if cycle is not None else {}))
    largest_exclusion = max(map(len, exclusion_components))
    largest_native_raw = max(len(c['bodies']) for c in native_components)
    largest_native_certified = max([1]+[len(c['bodies']) for c in native_components if c['certified']])
    resolved = all(c['certified'] for c in native_components)
    if not resolved:
        environment = 'remaining'
    elif not keys:
        environment = 'dispersed' if largest_exclusion == 1 else 'contact_no_entry'
    elif largest_native_raw < reference_size:
        environment = 'registered_small'
    elif largest_native_raw == reference_size:
        environment = 'registered_eight'
    else:
        environment = 'registered_growth'
    return dict(exclusion_edges=[list(e) for e in exclusion_edges],
        exclusion_components=exclusion_components, native_components=native_components,
        native_edges=[list(e) for e in native_edges], instantaneous_native_keys=[list(k) for k in keys],
        largest_exclusion=largest_exclusion, largest_native_raw=largest_native_raw,
        largest_native_certified=largest_native_certified, native_registry_resolved=resolved,
        native_cycle_frustrated=any(c['catalogue_consistent'] is False for c in native_components),
        native_periodic_winding=any(not c['ordinary_space_lift'] for c in native_components),
        environment=environment)


class FiniteAssemblyObserver:
    """Passive pose observer, including mobile bodies outside a crystallite."""
    def __init__(self, shape, patch_map, config, native):
        require(native is not None, 'Frozen complete native definition required')
        self.contact = ContactObserver(shape, patch_map, config, native=None)
        self.native = native
        self.n = len(config['initial_poses'])

    def classify(self, poses):
        require(len(poses) == self.n, 'Body inventory changed')
        observation = self.contact.classify(poses)
        positions, _ = pose_arrays(poses)
        periodic = self.contact.boundary['kind'] == 'periodic'
        keys = []
        for i, j in itertools.combinations(range(self.n), 2):
            position = positions[j].copy()
            if periodic:
                position -= self.contact.lengths*np.floor((position-positions[i])/self.contact.lengths+.5)
            partner = dict(poses[j], position=position.tolist())
            keys.extend((i, j, int(m['motif_id']))
                        for m in self.native.classify_pair(poses[i], partner))
        edges = sorted({(i, j) for i, j, _, _ in observation['tokens']})
        graph = graph_summary(self.n, edges, keys, self.native.motifs, positions,
                              self.contact.lengths if periodic else None)
        return dict(tokens=observation['tokens'], **graph)


def summarize_series(observations, frames, window, reference=None, threshold=.01):
    """Fixed-cadence retained-state statistics after the legacy run audit.

    This function is also usable for synthetic validation. The production caller
    must first authenticate the trajectory and obtain window from validate_frames.
    A reference must be independently authenticated against this observer law.
    """
    require(len(observations) == len(frames) and len(frames) > 1, 'Incomplete observation inventory')
    chosen, selected = window['indices'], window['sample_indices']
    require(len(chosen) >= 2 and chosen == list(range(chosen[0], chosen[-1]+1))
            and selected == chosen[1:] and 0 <= chosen[0] < chosen[-1] < len(frames),
            'Window must include one baseline followed by every retained endpoint')
    cpu = window['cpu_seconds']
    require(math.isfinite(cpu) and cpu > 0, 'Positive sampler CPU required')
    require(math.isfinite(threshold) and 0 < threshold < .5, 'Invalid appreciability threshold')
    sweeps = [frames[i]['sweep'] for i in chosen]
    stride = window['cadence_sweeps']
    require(type(stride) is int and stride > 0
            and all(b-a == stride for a, b in zip(sweeps, sweeps[1:])), 'Irregular saved cadence')
    labels = [obs['environment'] for obs in observations]
    require(set(labels) <= set(REGIONS), 'Unrecognized environment')
    def indicator_fingerprint(field):
        tokens = sorted({tuple(t) for i in chosen for t in observations[i][field]})
        token_sets = [set(map(tuple, observations[i][field])) for i in selected]
        bits = np.asarray([[t in present for t in tokens] for present in token_sets], float).reshape(len(selected), len(tokens))
        result = apparent_effective_count(bits, cpu)
        result.update(observed_token_dictionary=[list(t) for t in tokens],
            constant_observed_token_count=int(np.sum(np.ptp(bits, axis=0) == 0)),
            unseen_token_coverage='Unseen contacts have no estimated ESS. No token filters the proposal.')
        return result
    fingerprint = indicator_fingerprint('tokens')
    native_fingerprint = indicator_fingerprint('instantaneous_native_keys')
    occupancies = {}
    for label in REGIONS:
        values = np.asarray([labels[i] == label for i in selected], float)
        split = len(values)//2
        occupancies[label] = dict(fraction=float(values.mean()), observations=int(values.sum()),
            first_half_fraction=float(values[:split].mean()) if split else None,
            second_half_fraction=float(values[split:].mean()), apparent_ess=apparent_effective_count(values, cpu))
    events = sampled_exchanges([labels[i] for i in chosen], sweeps, set(REGIONS)-{'remaining'})
    for pair in events['pairs']:
        pair['equilibrium_interpretation'] = reference_assessment(pair, reference, threshold)
        pair['completed_passages_per_sampler_cpu_second'] = (pair['forward']+pair['reverse'])/cpu
    size_statistics = {}
    for name in ('largest_exclusion', 'largest_native_raw', 'largest_native_certified'):
        values = np.asarray([observations[i][name] for i in selected], float)
        size_statistics[name] = dict(mean=float(values.mean()), apparent_ess=apparent_effective_count(values, cpu))
    counts = {}
    for i in selected:
        obs = observations[i]
        key = (obs['largest_exclusion'], obs['largest_native_raw'],
               obs['largest_native_certified'], obs['native_registry_resolved'])
        counts[key] = counts.get(key, 0)+1
    joint = [dict(largest_exclusion=k[0], largest_native_raw=k[1], largest_native_certified=k[2],
                  registry_resolved=k[3], observations=v, fraction=v/len(selected))
             for k, v in sorted(counts.items())]
    n = sum(map(len, observations[chosen[0]]['exclusion_components']))
    histograms = {}
    for name in ('exclusion', 'native_raw', 'native_certified'):
        matrix = np.zeros((len(selected), n))
        for row, i in enumerate(selected):
            obs = observations[i]
            require(sum(map(len, obs['exclusion_components'])) == n, 'Body inventory changed')
            groups = obs['exclusion_components'] if name == 'exclusion' else [
                c['bodies'] for c in obs['native_components']
                if name == 'native_raw' or c['certified']]
            for group in groups:
                matrix[row, len(group)-1] += 1
        histograms[name] = dict(sizes=list(range(1, n+1)),
            mean_component_counts=matrix.mean(axis=0).tolist(),
            apparent_ess=apparent_effective_count(matrix, cpu))
    return dict(fingerprint=fingerprint, native_fingerprint=native_fingerprint,
        component_number_statistics=histograms, environment_occupancies=occupancies,
        environment_exchanges=events, labels_by_frame=labels, size_statistics=size_statistics,
        size_pair_occupancies=joint, window=window, scope=SCOPE,
        size_coverage='Unobserved size bins have zero empirical counts, not established zero equilibrium probability.')
