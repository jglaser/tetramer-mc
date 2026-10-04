"""Passive native registry for two mobile bodies against frozen spectators.

The caller supplies hard/wall-valid retained poses and the frozen, independently
bound native classifier. This adapter changes neither the sampler nor the native
predicate. The centroid filter is only a necessary-condition candidate search;
every reported label comes from the unchanged full pair classifier.
"""
from __future__ import annotations

import copy
import json

from native_contact_regions import pose_arrays, require
from native_graph_consistency import NativeGraphConsistency
from native_pair_candidates import NativePairCandidates


SCOPE = ('Instantaneous conditional native registry; not all-mobile assembly, '
         'a hard/wall check, thermodynamic stability, or a sampling-efficiency claim. '
         'No hysteresis, persistence, contact-edge filter or acceptance filter. '
         'Catalogue cycle inconsistency is retained; no compatible subgraph is selected.')


def canonical(poses):
    return json.dumps(poses, sort_keys=True, separators=(',', ':'), allow_nan=False)


class ConditionalNativeObserver:
    """Keep spectator labels fixed; invalidate mobile pair caches by exact poses.

    `classify` must be called for every saved endpoint, including rejections.
    Caching affects computation only. It never collapses repeated observations.
    Native definition/shape equivalence and retained-state provenance belong to
    the enclosing analysis plan, not this geometry adapter.

    `mobile_registry_resolved` includes isolated bodies and says only that the
    reported labels are consistent. Attachment requires actual native edges.
    """
    def __init__(self, native, source, members, *, boundary='spherical', graph_cache_entries=1024):
        require(len(source) >= 2, 'At least two labelled bodies required')
        require(len(members) == 2 and len(set(members)) == 2
                and all(type(i) is int and 0 <= i < len(source) for i in members),
                'Exactly two distinct mobile labels required')
        require(boundary in ('open', 'spherical'), 'Periodic native images are not implemented here')
        require(type(graph_cache_entries) is int and graph_cache_entries > 0,
                'Positive graph cache limit required')
        for pose in source:
            pose_arrays(pose)
        self.native = native
        self.source = copy.deepcopy(source)
        self.members = tuple(members)
        self.member_set = set(members)
        self.labels = {m['id'] for m in native.motifs}
        require(len(self.labels) == len(native.motifs)
                and all(type(label) is int for label in self.labels), 'Unique integer motif IDs required')
        motifs = [dict(position=m['relative_position'], orientation=m['relative_orientation'])
                  for m in native.motifs]
        self.filter = NativePairCandidates(native.member_positions, motifs,
            native.definition['criteria']['body_member_position_entry_A'], boundary=boundary)
        self.checker = NativeGraphConsistency(native.motifs, len(source))
        self.graph_cache_entries = graph_cache_entries
        self.fixed_keys = None
        self.cache = {}
        self.previous_endpoint = None
        self.counts = dict(observations=0, candidate_searches=0, fixed_pair_calls=0,
                           mobile_pair_calls=0, mobile_pair_cache_hits=0,
                           whole_endpoint_cache_hits=0)

    def _candidates(self, poses, mobile_labels=None):
        self.counts['candidate_searches'] += 1
        return self.filter.candidate_pairs(poses, mobile_labels=mobile_labels)

    def _classify_pair(self, state, pair, role):
        # Count a begun query even if the frozen classifier raises; errors are
        # never converted into an empty native result.
        self.counts[role+'_pair_calls'] += 1
        a, b = pair
        matches = self.native.classify_pair(state[a], state[b])
        require(type(matches) is list, 'Native pair classifier must return every matching motif')
        keys = set()
        for match in matches:
            label = match['motif_id']
            require(type(label) is int and label in self.labels, 'Unknown native motif label')
            keys.add((a, b, label))
        return tuple(sorted(keys))

    def _graph(self, keys):
        key = tuple(sorted(set(map(tuple, keys))))
        if key not in self.checker.cache and len(self.checker.cache) >= self.graph_cache_entries:
            # A bounded computational cache, not a filter on retained graphs.
            self.checker.cache.pop(next(iter(self.checker.cache)))
        return self.checker.check(key)

    def classify(self, selected):
        require(len(selected) == 2, 'Exactly two retained mobile poses required')
        for pose in selected:
            pose_arrays(pose)
        endpoint_identity = canonical(selected)
        if self.previous_endpoint is not None and self.previous_endpoint[0] == endpoint_identity:
            self.counts['observations'] += 1
            self.counts['whole_endpoint_cache_hits'] += 1
            result = copy.deepcopy(self.previous_endpoint[1])
            result['observer_counts'] = dict(self.counts)
            return result
        state = copy.deepcopy(self.source)
        for body, pose in zip(self.members, selected):
            state[body] = copy.deepcopy(pose)
        if self.fixed_keys is None:
            fixed = set()
            for pair in self._candidates(self.source):
                if not self.member_set.intersection(pair):
                    fixed.update(self._classify_pair(self.source, pair, 'fixed'))
            self.fixed_keys = tuple(sorted(fixed))
        mobile = set()
        for pair in self._candidates(state, self.members):
            identity = canonical([state[i] for i in pair])
            cached = self.cache.get(pair)
            if cached is not None and cached[0] == identity:
                keys = cached[1]
                self.counts['mobile_pair_cache_hits'] += 1
            else:
                keys = self._classify_pair(state, pair, 'mobile')
                self.cache[pair] = (identity, keys)
            mobile.update(keys)
        keys = sorted(set(self.fixed_keys) | mobile)
        graph = self._graph(keys)
        mobile_components = []
        for component in graph['components']:
            if not self.member_set.intersection(component):
                continue
            bodies = set(component)
            local_keys = [key for key in keys if key[0] in bodies]
            # A frustrated, disconnected spectator component must not erase a
            # separate mobile component's result. All global keys remain above.
            local = self._graph(local_keys)
            mobile_components.append(dict(bodies=component,
                catalogue_consistent=local['consistent'],
                independent_cycles=local['independent_cycles'],
                native_keys=local_keys, chosen_edge_labels=local['chosen_edge_labels']))
        internal = tuple(sorted(self.members))
        self.counts['observations'] += 1
        result = dict(instantaneous_native_keys=keys,
            fixed_native_keys=list(self.fixed_keys),
            mobile_related_native_keys=sorted(mobile),
            external_native_keys=sorted(key for key in mobile if key[:2] != internal),
            internal_native_keys=sorted(key for key in mobile if key[:2] == internal),
            full_native_graph=graph, mobile_native_components=mobile_components,
            mobile_registry_resolved=all(c['catalogue_consistent'] for c in mobile_components),
            observer_counts=dict(self.counts), scope=SCOPE)
        self.previous_endpoint = (endpoint_identity, copy.deepcopy(result))
        return result
