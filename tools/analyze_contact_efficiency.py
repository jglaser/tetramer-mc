#!/usr/bin/env python3
"""Passive, fixed-cadence contact diagnostics for frozen all-mobile Rust runs.

Repeated retained states are observations. Neither accepted-only trajectories
nor irregularly spaced endpoints define the chain analyzed here. Apparent ESS
is an observable-specific finite-record diagnostic, not proof of equilibrium.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mobile_posterior_metrics import apparent_effective_count


SCHEMA = 'passive-contact-efficiency-v1'
PLAN_SCHEMA = 'frozen-contact-efficiency-plan-v1'
SCOPE = ('Fixed-cadence retained-state diagnostics, including all rejection residence at the saved cadence. '
         'Events between saved endpoints are unresolved. Apparent ESS and CPU rates do not certify stationarity, '
         'coverage, independent preparations, physical kinetics or equilibrium assembly. No equal forward/reverse '
         'transition-rate requirement is imposed on unequally occupied environments.')
FROZEN_MODES = ('auxiliary_transport', 'reversible_jump', 'contact_memory', 'conditional_closure',
                'atlas_transport', 'atlas_mask', 'assembly_bias')
MATCHED_SCHEDULE = ('local_translation_std_A', 'local_small_angle_std_degrees',
                    'global_probability', 'gca_probability', 'center_shift_probability',
                    'poisson_lambda_ratio', 'endpoint_gate')
# Config and GateOptions serde defaults in src/simulation.rs and src/depletion.rs.
GATE_DEFAULTS = dict(max_cells=2047, max_depth=14, min_width=.5)
CONFIG_DEFAULTS = dict(monomer_shape=None, boundary=dict(kind='periodic'),
    gca_probability=0., center_shift_probability=0., poisson_lambda_ratio=16.,
    global_probability=.5, local_translation_std_A=.2, local_small_angle_std_degrees=1.,
    learned_uniform_weight=.1, endpoint_gate=GATE_DEFAULTS, seed_labels=[],
    fixed_body_indices=[], metadata=None, frozen_posterior=None, **{k: None for k in FROZEN_MODES})
CONFIG_REQUIRED = ('shape', 'box_lengths', 'initial_poses', 'seed', 'depletant_radius', 'reservoir_density')
# The only additions/overrides emitted by simulation::run. Paths and
# metadata.native_pair_motifs are canonicalized; CLI options and resumed/wrapped
# initial poses are reconciled separately below. Rust ignores unknown input
# fields, but they cannot become unexplained effective configuration fields.
EFFECTIVE_OVERRIDES = ('method', 'sweeps', 'sample_every', 'initial_poses', 'shape', 'monomer_shape',
    'metadata', 'uniform_proposal_cube_lengths', 'coordinate_origin_sweep',
    'resolved_contact_memory', 'coordinate_frame_convention')
FRAME_CONVENTION = ('Stored poses use sphere-centered coordinates; coordinate positions = stored positions + '
    'coordinate_wall_center; center shifts subtract their common displacement from the wall center. '
    'Origin established at coordinate_origin_sweep.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, 'Invalid '+label)
    return value


def pose_arrays(poses):
    p = np.asarray([x['position'] for x in poses], float)
    q = np.asarray([x['orientation'] for x in poses], float)
    require(p.shape == (len(poses), 3) and q.shape == (len(poses), 4)
            and np.isfinite(p).all() and np.isfinite(q).all(), 'Invalid retained pose')
    require(np.max(abs(np.linalg.norm(q, axis=1)-1)) <= 1e-8, 'Unnormalized quaternion')
    return p, Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def validate_effective_config(config, declared, manifest, summary, shape):
    """Reconcile archived Rust input, serde defaults and the explicit run overrides."""
    require(all(k in declared for k in CONFIG_REQUIRED), 'Incomplete archived Rust input config')
    expected = copy.deepcopy(CONFIG_DEFAULTS)
    expected.update({k: declared[k] for k in (*CONFIG_REQUIRED, *CONFIG_DEFAULTS) if k in declared})
    require(isinstance(expected['endpoint_gate'], dict), 'Invalid archived endpoint gate')
    require(set(expected['endpoint_gate']) <= set(GATE_DEFAULTS), 'Unknown archived endpoint gate field')
    expected['endpoint_gate'] = {**GATE_DEFAULTS, **expected['endpoint_gate']}
    require(set(config) <= set(expected) | set(EFFECTIVE_OVERRIDES), 'Unknown effective config field')
    for key, value in expected.items():
        if key not in EFFECTIVE_OVERRIDES:
            # Only None-valued optional modes may be skipped by Rust serialization.
            require(config.get(key) == value and (key in config or value is None),
                    'Effective config differs from archived input/default: '+key)
    for key in ('shape', 'monomer_shape'):
        value = expected[key]
        require((value is None and config.get(key) is None) or
                (isinstance(value, str) and value and isinstance(config.get(key), str)
                 and Path(config[key]).is_absolute()), 'Invalid canonicalized config path: '+key)
    metadata = copy.deepcopy(expected['metadata'])
    if isinstance(metadata, dict) and isinstance(metadata.get('native_pair_motifs'), str):
        actual = config.get('metadata')
        require(isinstance(actual, dict) and isinstance(actual.get('native_pair_motifs'), str)
                and Path(actual['native_pair_motifs']).is_absolute(), 'Invalid canonical native catalogue path')
        metadata['native_pair_motifs'] = actual['native_pair_motifs']
    require(config.get('metadata') == metadata == summary['initial_metadata'], 'Effective metadata differs from archived input')
    require(manifest['boundary'] == expected['boundary'], 'Manifest boundary differs from archived input')
    if declared.get('spherical_radius') is not None:
        require(expected['boundary'].get('radius') == declared['spherical_radius'], 'Legacy spherical radius differs')
    for key in (*FROZEN_MODES, 'frozen_posterior'):
        require(config.get(key) == manifest.get(key) == summary.get(key), 'Manifest proposal mode differs: '+key)
    require(config['method'] == summary['method'] and config['sweeps'] == summary['requested_sweeps'],
            'Effective CLI overrides disagree with summary')
    integer(config['sample_every'], 'effective sample cadence', 1)
    if manifest.get('resume') is None:
        require(manifest['initial_sweep'] == 0, 'Initial sweep requires declared resume')
        poses = copy.deepcopy(expected['initial_poses'])
        if expected['boundary']['kind'] == 'periodic':
            for p in poses:
                p['position'] = [x % length for x, length in zip(p['position'], expected['box_lengths'])]
        require(config['initial_poses'] == poses, 'Initial poses differ without a resumed checkpoint')
    else:
        require(manifest['initial_sweep'] > 0 and len(config['initial_poses']) == len(expected['initial_poses']),
                'Invalid resumed initial poses or sweep')
    bound = max(np.linalg.norm(a['center'])+a['radius'] for a in shape['atoms'])
    lengths = (expected['box_lengths'] if expected['boundary']['kind'] == 'periodic'
               else [2*(expected['boundary']['radius']+bound)]*3)
    require(np.shape(config['uniform_proposal_cube_lengths']) == (3,)
            and np.allclose(config['uniform_proposal_cube_lengths'], lengths, rtol=1e-14, atol=0.),
            'Effective uniform proposal cell differs')
    origin = integer(config['coordinate_origin_sweep'], 'coordinate origin sweep')
    require(origin <= manifest['initial_sweep'] and (manifest.get('resume') is not None or origin == 0),
            'Invalid effective coordinate origin')
    require(config['resolved_contact_memory'] == manifest.get('contact_memory'), 'Resolved contact memory differs')
    require(config['coordinate_frame_convention'] == FRAME_CONVENTION, 'Unknown effective coordinate frame')


def physical_identity(config, shape_sha):
    """Target identity deliberately excludes proposal, RNG and initial positions."""
    return dict(shape_sha256=shape_sha, bodies=len(config['initial_poses']),
                boundary=config['boundary'], box_lengths=config['box_lengths'],
                depletant_radius=config['depletant_radius'], reservoir_density=config['reservoir_density'],
                fixed_body_indices=config.get('fixed_body_indices', []),
                measure='d^3t times normalized Haar for each labelled rigid body; ideal bath wall-permeable')


def validate_counts(counts, bodies):
    for kind in ('local', 'global'):
        c = counts[kind]
        for key in ('attempted', 'hard_valid', 'accepted', 'hard_rejected', 'proposal_nulls'):
            integer(c[key], kind+' '+key)
        require(c['attempted'] == c['hard_valid']+c['hard_rejected']+c['proposal_nulls'],
                'Incomplete attempted-move dispositions')
        require(c['accepted'] <= c['hard_valid'], 'Accepted count exceeds valid count')
    require(counts['selected_body_updates'] == counts['local']['attempted']+counts['global']['attempted'],
            'Selected updates do not match attempted dispositions')
    each = counts['selected_body_updates_by_body']
    require(len(each) == bodies, 'Missing per-body attempted dispositions')
    for c in each:
        integer(c, 'selected body count')
    require(sum(each) == counts['selected_body_updates'], 'Per-body attempt sum differs')
    for kind in ('gca', 'center_shift'):
        c = counts[kind]
        for key in ('attempted', 'completed', 'transformed_bodies'):
            integer(c[key], kind+' '+key)
        require(c['completed'] <= c['attempted'], 'Collective dispositions invalid')


def numeric_leaves(value, prefix=()):
    if isinstance(value, dict):
        return {p: v for key, child in value.items() for p, v in numeric_leaves(child, prefix+(key,)).items()}
    if isinstance(value, list):
        return {p: v for key, child in enumerate(value) for p, v in numeric_leaves(child, prefix+(key,)).items()}
    return {prefix: value} if type(value) is int else {}


def validate_frames(frames, config, summary, manifest, checkpoint, burn_sweep, end_sweep):
    """Audit the entire saved axis, then select an explicitly declared regular window."""
    require(summary['complete'] is True and summary['all_bodies_mobile'] is True, 'Completed all-mobile run required')
    require(not config.get('fixed_body_indices'), 'Fixed bodies are outside this analyzer')
    require(all(config.get(key) is None and manifest.get(key) is None for key in FROZEN_MODES),
            'Ordinary physical ESS requires a frozen, unbiased proposal; adaptive/bias modes unsupported here')
    n = len(config['initial_poses']); integer(n, 'body count', 2)
    require(summary['bodies'] == n, 'Summary body count differs')
    start, total = summary['initial_sweep'], summary['completed_sweeps']
    stride = integer(config['sample_every'], 'sample cadence', 1)
    require(start == manifest['initial_sweep'] and total == config['sweeps'] == summary['requested_sweeps'],
            'Segment sweep identity differs')
    expected = [start]+[s for s in range(start+1, total+1) if s % stride == 0 or s == total]
    require([f['sweep'] for f in frames] == expected, 'Missing, duplicate or accepted-only saved endpoints')
    require(frames[0]['poses'] == config['initial_poses'], 'Initial retained poses differ from effective config')
    require(frames[-1]['poses'] == checkpoint['poses'] and frames[-1]['counts'] == checkpoint['counts'] == summary['counts'],
            'Terminal trajectory/checkpoint/summary closure failed')
    require(frames[0]['counts'] == summary['initial_counts'] and checkpoint['completed_sweeps'] == total,
            'Initial counts or checkpoint sweep mismatch')
    require(checkpoint['master_seed'] == config['seed'] and checkpoint['method'] == summary['method'] == config['method'],
            'Checkpoint seed or method mismatch')
    prior_cpu, prior_counts = -1., None
    for frame in frames:
        require(len(frame['poses']) == n and frame['boundary'] == config['boundary']['kind'], 'Frame identity differs')
        pose_arrays(frame['poses'])
        require(frame.get('assembly_bias') is None, 'Biased frame lacks supported physical ESS correction')
        cpu = frame['sampler_cpu_seconds']
        require(math.isfinite(cpu) and cpu >= 0 and cpu > prior_cpu, 'Missing or nonmonotone sampler CPU')
        prior_cpu = cpu
        validate_counts(frame['counts'], n)
        now = numeric_leaves(frame['counts'])
        if prior_counts is not None:
            require(now.keys() == prior_counts.keys() and all(now[k] >= prior_counts[k] for k in now), 'Counts decreased or changed schema')
        prior_counts = now
        increments = [v-a for v, a in zip(frame['counts']['selected_body_updates_by_body'], frames[0]['counts']['selected_body_updates_by_body'])]
        require(increments == [frame['sweep']-start]*n, 'Every body must retain one attempted update per sweep')
    cpu_total = summary['sampler_cpu_seconds']
    require(math.isfinite(cpu_total) and cpu_total >= prior_cpu, 'Invalid terminal sampler CPU')
    for key, value in numeric_leaves(summary['segment_counts']).items():
        require(value == numeric_leaves(summary['counts'])[key]-numeric_leaves(summary['initial_counts'])[key],
                'Segment disposition totals disagree')
    require(numeric_leaves(summary['segment_counts']).keys() == numeric_leaves(summary['counts']).keys(),
            'Missing segment dispositions')
    by_sweep = {f['sweep']: i for i, f in enumerate(frames)}
    require(burn_sweep in by_sweep and end_sweep in by_sweep and burn_sweep < end_sweep, 'Window endpoints must be saved sweeps')
    indices = list(range(by_sweep[burn_sweep], by_sweep[end_sweep]+1))
    require(all(frames[b]['sweep']-frames[a]['sweep'] == stride for a, b in zip(indices, indices[1:])),
            'Irregular terminal/resume interval: declare a regular saved window')
    cpu = frames[indices[-1]]['sampler_cpu_seconds']-frames[indices[0]]['sampler_cpu_seconds']
    require(cpu > 0, 'Nonpositive window CPU')
    # The baseline establishes the first interval; it is not an extra endpoint sample.
    return dict(indices=indices, sample_indices=indices[1:], cpu_seconds=cpu, cadence_sweeps=stride,
                burn_sweep=burn_sweep, end_sweep=end_sweep, body_count=n)


class ContactObserver:
    """Stateless atom-level exclusion contacts, with a frozen body-frame patch map."""
    def __init__(self, shape, patch_map, config, native=None):
        self.atoms = np.asarray([a['center'] for a in shape['atoms']], float)
        self.radii = np.asarray([a['radius'] for a in shape['atoms']], float)
        require(self.atoms.shape == (len(self.radii), 3) and len(self.radii) > 0
                and np.isfinite(self.atoms).all() and np.isfinite(self.radii).all()
                and np.all(self.radii > 0), 'Invalid shape geometry')
        require(len(patch_map) == len(self.radii) and all(isinstance(s, str) and s for s in patch_map),
                'Frozen nonempty patch ID required for every atom; no filtering')
        self.patches = patch_map
        self.bound = float(np.max(np.linalg.norm(self.atoms, axis=1)+self.radii))
        self.rd = config['depletant_radius']; require(math.isfinite(self.rd) and self.rd >= 0, 'Invalid depletant radius')
        self.boundary = config['boundary']; self.lengths = np.asarray(config['box_lengths'], float)
        require(self.boundary['kind'] in ('spherical', 'periodic'), 'Unsupported boundary')
        require(self.lengths.shape == (3,) and np.isfinite(self.lengths).all() and np.all(self.lengths > 0), 'Invalid cell')
        if self.boundary['kind'] == 'periodic':
            require(np.all(self.lengths > 4*(self.bound+self.rd)), 'Unique minimum-image contact bound not satisfied')
        self.native = native
        self.checker = None
        if native is not None:
            from native_graph_consistency import NativeGraphConsistency
            self.checker = NativeGraphConsistency(native.motifs, len(config['initial_poses']))

    def classify(self, poses):
        p, rotation = pose_arrays(poses)
        rotated = np.einsum('bij,aj->bai', rotation, self.atoms)
        tokens, native_keys = set(), []
        if self.boundary['kind'] == 'spherical':
            clearance = self.boundary['radius']-np.linalg.norm(rotated+p[:, None, :], axis=2)-self.radii
            require(np.min(clearance) >= -2e-8, 'Retained pose violates atomic wall')
        for i, j in itertools.combinations(range(len(poses)), 2):
            delta = p[j]-p[i]
            if self.boundary['kind'] == 'periodic':
                delta -= self.lengths*np.floor(delta/self.lengths+.5)
            if np.linalg.norm(delta) <= 2*(self.bound+self.rd):
                near = cKDTree(rotated[i]).sparse_distance_matrix(cKDTree(rotated[j]+delta),
                    2*(float(self.radii.max())+self.rd), output_type='ndarray')
                if len(near):
                    gap = near['v']-self.radii[near['i']]-self.radii[near['j']]
                    require(float(gap.min()) >= -1e-8, 'Retained inter-body hard overlap')
                    for a, b in zip(near['i'][gap < 2*self.rd], near['j'][gap < 2*self.rd]):
                        tokens.add((i, j, self.patches[a], self.patches[b]))
            if self.native is not None:
                # Reimage only the pair. Open-space classifier performs its own exact predicates.
                partner = dict(poses[j], position=(p[i]+delta).tolist())
                native_keys.extend((i, j, m['motif_id']) for m in self.native.classify_pair(poses[i], partner))
        cycle = self.checker.check(native_keys) if self.checker else None
        if cycle is not None and self.boundary['kind'] == 'periodic':
            # The catalogue checker uses a lift to ordinary space. A graph
            # winding around the periodic cell needs a different closure law.
            edges = {(i, j) for i, j, _ in native_keys}
            images = {}
            for root in range(len(poses)):
                if root in images:
                    continue
                images[root] = np.zeros(3, dtype=int); stack = [root]
                while stack:
                    i = stack.pop()
                    for a, b in edges:
                        if i not in (a, b):
                            continue
                        j = b if a == i else a
                        winding = -np.floor((p[j]-p[i])/self.lengths+.5).astype(int)
                        predicted = images[i]+winding
                        if j in images:
                            require(np.array_equal(images[j], predicted),
                                    'Periodic native graph has nonzero winding; ordinary-space cycle observer is inapplicable')
                        else:
                            images[j] = predicted; stack.append(j)
        return dict(tokens=sorted(tokens), instantaneous_native_keys=sorted(native_keys), native_cycle=cycle)


def environment_label(observation, definitions):
    tokens = set(map(tuple, observation['tokens']))
    matches = []
    for definition in definitions:
        required, forbidden = map(lambda name: set(map(tuple, definition.get(name, []))), ('required_tokens', 'forbidden_tokens'))
        match = required <= tokens and not (forbidden & tokens)
        if 'native_consistent_connected' in definition:
            require(observation['native_cycle'] is not None, 'Native environment requires a frozen native observer')
            match &= observation['native_cycle']['consistent_connected'] is definition['native_consistent_connected']
        if match:
            matches.append(definition['id'])
    require(len(matches) <= 1, 'Declared environments overlap on an observed pose')
    return matches[0] if matches else 'remaining'


def sampled_exchanges(labels, sweeps, target_ids):
    """Observed A→B passages, allowing remaining states; returns/roundtrips separate."""
    events, episodes = [], []
    previous_target = None; previous_index = None; path = []
    first = 0
    for index, label in enumerate(labels):
        if index and label != labels[index-1]:
            episodes.append(dict(environment=labels[first], start_sweep=sweeps[first], last_observed_sweep=sweeps[index-1],
                                 exit_sweep=sweeps[index], observations=index-first,
                                 observed_sweep_span=sweeps[index]-sweeps[first], left_censored=first == 0, right_censored=False))
            first = index
        if label not in target_ids:
            continue
        if previous_target is not None and previous_target != label:
            events.append(dict(source=previous_target, target=label, source_last_observed_sweep=sweeps[previous_index],
                               arrival_sweep=sweeps[index], intervening_saved_labels=labels[previous_index+1:index]))
            path.append(label)
        elif previous_target is None:
            path.append(label)
        previous_target, previous_index = label, index
    episodes.append(dict(environment=labels[first], start_sweep=sweeps[first], last_observed_sweep=sweeps[-1],
                         exit_sweep=None, observations=len(labels)-first, observed_sweep_span=sweeps[-1]-sweeps[first],
                         left_censored=first == 0, right_censored=True))
    directional = Counter((e['source'], e['target']) for e in events)
    pairs = []
    for a, b in itertools.combinations(sorted(target_ids), 2):
        # Another declared environment interrupts a pair roundtrip.
        lengths, sequence = [], []
        for x in path:
            if x not in (a, b):
                lengths.append(len(sequence)); sequence = []
            elif not sequence or sequence[-1] != x:
                sequence.append(x)
        lengths.append(len(sequence))
        pairs.append(dict(regions=[a, b], forward=directional[a, b], reverse=directional[b, a],
                          completed_nonoverlapping_roundtrips=sum(max(0, (length-1)//2) for length in lengths)))
    return dict(events=events, pairs=pairs, sampled_episodes=episodes,
                scope='Saved-frame passages and censored episode spans only; transient crossings and within-interval residence are unknown.')


def reference_assessment(pair, reference, threshold):
    """A missing transition is never converted into a thermodynamic zero."""
    labels = pair['regions']
    if reference is None:
        return dict(status='unassessed_equilibrium_weights', missing_transition_is_failure=False)
    intervals = [reference['probability_bounds'][name] for name in labels]
    negligible = [name for name, interval in zip(labels, intervals) if interval[1] < threshold]
    appreciable = all(interval[0] >= threshold for interval in intervals)
    if negligible:
        return dict(status='independently_bounded_negligible_region', regions=negligible,
                    frequent_returns_required=False, missing_transition_is_failure=False)
    if appreciable:
        completed = pair['forward'] > 0 and pair['reverse'] > 0
        return dict(status='both_regions_independently_appreciable', completed_both_directions=completed,
                    mixing_limitation_flag=not completed, scope='Flag is a finite-run sampling limitation, not thermodynamic instability.')
    return dict(status='reference_bounds_inconclusive', missing_transition_is_failure=False)


def summarize_observations(observations, frames, window, definitions, reference=None, threshold=.01):
    all_ids = [d['id'] for d in definitions]+['remaining']
    chosen = window['indices']; selected = window['sample_indices']
    tokens = sorted({tuple(t) for k in chosen for t in observations[k]['tokens']})
    token_sets = [set(map(tuple, observations[k]['tokens'])) for k in selected]
    bits = np.asarray([[token in s for token in tokens] for s in token_sets], dtype=float).reshape(len(selected), len(tokens))
    cpu = window['cpu_seconds']
    fingerprint = apparent_effective_count(bits, cpu)
    fingerprint['observed_token_dictionary'] = [list(t) for t in tokens]
    fingerprint['unseen_token_coverage'] = 'Unseen contacts have no estimated ESS; this dictionary is an observed diagnostic, not a proposal filter.'
    fingerprint['constant_observed_token_count'] = int(np.sum(np.ptp(bits, axis=0) == 0)) if len(bits) else 0
    labels = [environment_label(obs, definitions) for obs in observations]
    occupancies = {}
    for name in all_ids:
        values = np.asarray([labels[k] == name for k in selected], float)
        split = len(values)//2
        occupancies[name] = dict(fraction=float(values.mean()), observations=int(values.sum()),
                                first_half_fraction=float(values[:split].mean()) if split else None,
                                second_half_fraction=float(values[split:].mean()),
                                apparent_ess=apparent_effective_count(values, cpu))
    events = sampled_exchanges([labels[k] for k in chosen], [frames[k]['sweep'] for k in chosen], set(all_ids)-{'remaining'})
    for pair in events['pairs']:
        pair['equilibrium_interpretation'] = reference_assessment(pair, reference, threshold)
        pair['completed_passages_per_sampler_cpu_second'] = (pair['forward']+pair['reverse'])/cpu
    return dict(fingerprint=fingerprint, environment_occupancies=occupancies, environment_exchanges=events,
                labels_by_frame=labels, window=window, scope=SCOPE)


def load_bound_file(binding, root):
    path = Path(binding['path'])
    if not path.is_absolute():
        path = root/path
    path = path.resolve()
    require(sha(path) == binding['sha256'], 'Frozen analysis input changed: '+str(path))
    return path


def native_input_bindings(definition_path):
    """Bind the entire declared closure, including inputs not read by classification."""
    definition_path = Path(definition_path).resolve()
    root = (definition_path.parent/'inputs').resolve()
    bindings = {str(definition_path): sha(definition_path)}
    for name, expected in read(definition_path)['input_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe native input path')
        path = root/relative
        require(path.resolve().is_relative_to(root), 'Native input escapes archived closure')
        # Retain the archived pathname so retargeted links are also rechecked.
        require(sha(path) == expected, 'Frozen native definition input changed: '+name)
        bindings[str(path)] = expected
    return bindings


def recheck_bindings(bindings, message):
    for path, expected in bindings.items():
        require(sha(path) == expected, message+': '+str(path))


def validate_probability_bounds(bounds, ids):
    require(set(bounds) == set(ids+['remaining']), 'Reference must cover every environment and remaining')
    for interval in bounds.values():
        require(len(interval) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in interval)
                and 0 <= interval[0] <= interval[1] <= 1, 'Invalid physical probability bounds')
    require(math.fsum(x[0] for x in bounds.values()) <= 1 <= math.fsum(x[1] for x in bounds.values()),
            'Physical probability bounds cannot cover an exhaustive partition')


def analyze(run, plan_path, out):
    run, plan_path, out = map(lambda p: Path(p).resolve(), (run, plan_path, out))
    require(__debug__, 'Run with PYTHONOPTIMIZE=0 for checked native catalogue predicates')
    require(not out.exists(), 'Use a fresh output directory')
    started = time.process_time()
    plan = read(plan_path); require(plan['schema'] == PLAN_SCHEMA, 'Unsupported observer plan')
    definitions = plan['environments']; ids = [d['id'] for d in definitions]
    require(ids and len(set(ids)) == len(ids) and 'remaining' not in ids and all(isinstance(s, str) and s for s in ids),
            'Unique declared environments required; remaining is reserved')
    for definition in definitions:
        require(set(definition) <= {'id', 'required_tokens', 'forbidden_tokens', 'native_consistent_connected'}, 'Unknown environment criterion')
        if 'native_consistent_connected' in definition:
            require(type(definition['native_consistent_connected']) is bool, 'Native criterion must be boolean')
        for name in ('required_tokens', 'forbidden_tokens'):
            for token in definition.get(name, []):
                require(len(token) == 4 and type(token[0]) is type(token[1]) is int
                        and 0 <= token[0] < token[1] and all(isinstance(x, str) and x for x in token[2:]),
                        'Environment tokens require ordered body IDs and two patch IDs')
    for key in ('preparation_id', 'proposal_arm'):
        require(isinstance(plan[key], str) and plan[key], 'Missing declared '+key)
    names = ('config.json', 'manifest.json', 'summary.json', 'checkpoint.json', 'trajectory.jsonl',
             'provenance/input-config.json', 'provenance/shape.json', 'provenance/source-bundle.json')
    bindings = {name: sha(run/name) for name in names}
    config, manifest, summary, checkpoint = [read(run/name) for name in names[:4]]
    for field, filename in [('config_sha256', 'provenance/input-config.json'), ('shape_sha256', 'provenance/shape.json')]:
        require(manifest[field] == summary[field] == checkpoint[field] == bindings[filename], 'Run provenance mismatch: '+field)
    require(manifest['source_bundle_sha256'] == bindings['provenance/source-bundle.json'], 'Source bundle changed')
    require(manifest.get('model_sha256') == summary.get('model_sha256') == checkpoint.get('model_sha256'), 'Model identity differs')
    if manifest.get('model_sha256') is not None:
        model = run/'provenance/frozen-relative-model.json'
        require(sha(model) == manifest['model_sha256'], 'Frozen proposal changed'); bindings[str(model.relative_to(run))] = sha(model)
    shape = read(run/'provenance/shape.json')
    validate_effective_config(config, read(run/'provenance/input-config.json'), manifest, summary, shape)
    frames = [json.loads(line) for line in (run/'trajectory.jsonl').read_text().splitlines()]
    window = validate_frames(frames, config, summary, manifest, checkpoint, plan['burn_sweep'], plan['end_sweep'])
    identity = physical_identity(config, manifest['shape_sha256'])
    require(digest(identity) == plan['physical_identity_sha256'], 'Analysis plan targets different physical measure')
    patch_path = load_bound_file(plan['patch_map'], plan_path.parent); patch = read(patch_path)
    require(patch['schema'] == 'body-frame-atom-patch-map-v1' and patch['shape_sha256'] == manifest['shape_sha256'], 'Patch/shape identity differs')
    implementation = {name: sha(Path(__file__).with_name(name))
                      for name in ('analyze_contact_efficiency.py', 'mobile_posterior_metrics.py')}
    native = None; dependencies = {str(plan_path): sha(plan_path), str(patch_path): sha(patch_path)}
    dependencies.update({str(Path(__file__).with_name(name).resolve()): value for name, value in implementation.items()})
    if plan.get('native_definition') is not None:
        from native_contact_regions import NativeContactRegions
        native_path = load_bound_file(plan['native_definition'], plan_path.parent)
        dependencies.update(native_input_bindings(native_path))
        native = NativeContactRegions(native_path)
        require(native.shape_sha256 == manifest['shape_sha256'], 'Native observer uses another shape')
        for source in ('native_contact_regions.py', 'native_graph_consistency.py'):
            path = Path(__file__).with_name(source).resolve(); dependencies[str(path)] = sha(path)
            implementation[source] = dependencies[str(path)]
    observer = ContactObserver(shape, patch['atom_patch_ids'], config, native)
    for definition in definitions:
        for name in ('required_tokens', 'forbidden_tokens'):
            for i, j, a, b in definition.get(name, []):
                require(j < window['body_count'] and a in observer.patches and b in observer.patches,
                        'Environment token is outside the fixed body/patch dictionary')
    observations = [observer.classify(frame['poses']) for frame in frames]
    observer_definition_sha = digest(dict(environments=definitions, patch_map_sha256=sha(patch_path),
        native_definition_sha256=None if native is None else native.definition_sha256))
    reference = None; threshold = plan.get('appreciable_probability', .01)
    require(math.isfinite(threshold) and 0 < threshold < .5, 'Invalid appreciability threshold')
    if plan.get('physical_reference') is not None:
        path = load_bound_file(plan['physical_reference'], plan_path.parent); reference = read(path)
        require(reference['schema'] == 'independent-contact-region-probability-bounds-v1'
                and reference['physical_identity_sha256'] == digest(identity)
                and reference['region_definition_sha256'] == digest(definitions)
                and reference['observer_definition_sha256'] == observer_definition_sha
                and reference['coverage_resolved'] is True and reference['population_diagnostics_passed'] is True,
                'Independent reference lacks matching target, regions or coverage')
        require(reference['independent_of_run_sha256'] == bindings['trajectory.jsonl'], 'Reference independence attestation does not identify this trajectory')
        evidence = load_bound_file(reference['evidence'], path.parent)
        dependencies.update({str(path): sha(path), str(evidence): sha(evidence)})
        validate_probability_bounds(reference['probability_bounds'], ids)
    result = summarize_observations(observations, frames, window, definitions, reference, threshold)
    result.update(schema=SCHEMA, complete=True, run=str(run), plan_sha256=sha(plan_path), physical_identity=identity,
                  region_definition_sha256=digest(definitions), observer_definition_sha256=observer_definition_sha,
                  initialization=dict(master_seed=config['seed'], preparation=config.get('metadata', {}),
                      preparation_id=plan['preparation_id'], proposal_arm=plan['proposal_arm'],
                      initial_poses_sha256=digest(frames[0]['poses']), invocation_initial_sweep=summary['initial_sweep'],
                      resume=manifest.get('resume'), initial_environment=result['labels_by_frame'][0]),
                  measurement_identity=dict(patch_map_sha256=sha(patch_path),
                      native_definition_sha256=None if native is None else native.definition_sha256,
                      cadence_sweeps=window['cadence_sweeps']),
                  schedule={key: config[key] for key in MATCHED_SCHEDULE}, proposal=dict(method=config['method'],
                      model_sha256=manifest.get('model_sha256'), frozen_posterior=config.get('frozen_posterior'),
                      learned_uniform_weight=config.get('learned_uniform_weight')),
                  source_sha256=bindings, dependency_sha256=dependencies, implementation_sha256=implementation,
                  observer_cpu_seconds=time.process_time()-started,
                  equilibrium_reference_scope='Reference coverage/independence are explicit evidence attestations; this observer verifies bindings, not the external free-energy derivation.')
    recheck_bindings({str(run/name): expected for name, expected in bindings.items()}, 'Input changed during observation')
    recheck_bindings(dependencies, 'Observer dependency changed during observation')
    out.mkdir(parents=True)
    save(out/'analysis.json', result)
    with (out/'observations.jsonl').open('w') as stream:
        for frame, obs, label in zip(frames, observations, result['labels_by_frame']):
            stream.write(json.dumps(dict(sweep=frame['sweep'], sampler_cpu_seconds=frame['sampler_cpu_seconds'],
                                        environment=label, **obs), allow_nan=False)+'\n')
    save(out/'freeze.json', dict(files={name: sha(out/name) for name in ('analysis.json', 'observations.jsonl')},
         observer_sha256=implementation['analyze_contact_efficiency.py'],
         ess_implementation_sha256=implementation['mobile_posterior_metrics.py'], implementation_sha256=implementation))
    return result


def population_mean(values):
    x = np.asarray(values, float)
    require(x.ndim == 1 and np.isfinite(x).all(), 'Invalid independent population measurements')
    return dict(populations=len(x), mean=float(x.mean()),
                population_standard_error=float(x.std(ddof=1)/math.sqrt(len(x))) if len(x) > 1 else None)


def compare_reports(reports):
    """Independent stream means, never concatenated-chain or accepted-event ESS."""
    require(len(reports) >= 2, 'At least two report populations are required')
    first = reports[0]
    for r in reports:
        require(r['schema'] == SCHEMA and r['complete'] is True, 'Incomplete contact report')
        for key in ('physical_identity', 'region_definition_sha256', 'observer_definition_sha256',
                    'measurement_identity', 'schedule', 'implementation_sha256'):
            require(r[key] == first[key], 'Comparison must match '+key)
        require((r['window']['burn_sweep'], r['window']['end_sweep']) ==
                (first['window']['burn_sweep'], first['window']['end_sweep']), 'Matched observation windows required')
    identities = [r['initialization']['master_seed'] for r in reports]
    require(len(set(identities)) == len(identities), 'Independent-stream comparison rejects reused/paired seeds or continued segments')
    groups, arm_models = {}, {}
    for r in reports:
        key = (r['initialization']['proposal_arm'], r['initialization']['preparation_id'])
        arm_models.setdefault(key[0], r['proposal'])
        require(arm_models[key[0]] == r['proposal'], 'One proposal arm must retain one frozen model across starts/streams')
        groups.setdefault(key, []).append(r)
    output = []
    for (arm, start), members in sorted(groups.items()):
        rates = [r['fingerprint']['apparent_ess_per_sampling_CPU_second'] for r in members]
        output.append(dict(proposal_arm=arm, preparation_id=start, populations=len(members),
            all_four_streams_present=len(members) >= 4,
            initialization_pose_hashes=[r['initialization']['initial_poses_sha256'] for r in members],
            region_occupancies={name: population_mean([r['environment_occupancies'][name]['fraction'] for r in members])
                                for name in first['environment_occupancies']},
            fingerprint_ess_per_cpu=None if any(x is None for x in rates) else population_mean(rates),
            constant_fingerprint_populations=sum(x is None for x in rates)))
    contrasts = []
    for left, right in itertools.combinations(output, 2):
        kind = ('initialization' if left['proposal_arm'] == right['proposal_arm'] else
                'proposal' if left['preparation_id'] == right['preparation_id'] else None)
        if kind is None:
            continue
        for region, a in left['region_occupancies'].items():
            b = right['region_occupancies'][region]
            se = None if a['population_standard_error'] is None or b['population_standard_error'] is None else math.hypot(a['population_standard_error'], b['population_standard_error'])
            difference = a['mean']-b['mean']
            contrasts.append(dict(kind=kind, region=region,
                left=[left['proposal_arm'], left['preparation_id']], right=[right['proposal_arm'], right['preparation_id']],
                difference=difference, combined_population_standard_error=se,
                diagnostic_agreement_within_three_observed_se=None if se is None else abs(difference) <= 3*se,
                zero_observed_variance=se == 0,
                scope='Observed population means only; zero variance, common trapping or unseen modes can invalidate a convergence interpretation.'))
    return dict(schema='contact-efficiency-independent-stream-comparison-v1', groups=output, contrasts=contrasts,
                scope=SCOPE+' Independent stream means supply between-run uncertainty. Null ESS values remain null; no pooling chains or artificial iid ESS.')


def compare_report_files(paths):
    """Verify each frozen report and its observations before population comparison."""
    reports, bindings = [], {}
    for value in paths:
        path = Path(value).resolve()
        require(path.name == 'analysis.json', 'Comparison requires frozen analysis.json reports')
        freeze_path = path.with_name('freeze.json')
        bindings[str(freeze_path)] = sha(freeze_path)
        freeze = read(freeze_path)
        require(set(freeze['files']) == {'analysis.json', 'observations.jsonl'}, 'Incomplete report freeze')
        for name, expected in freeze['files'].items():
            item = path.with_name(name)
            require(sha(item) == expected, 'Frozen contact report changed: '+str(item))
            bindings[str(item)] = expected
        report = read(path)
        implementation = report['implementation_sha256']
        require(freeze['observer_sha256'] == implementation['analyze_contact_efficiency.py']
                and freeze['ess_implementation_sha256'] == implementation['mobile_posterior_metrics.py']
                and freeze['implementation_sha256'] == implementation, 'Report implementation freeze differs')
        for name, expected in implementation.items():
            require(any(Path(source).name == name and value == expected
                        for source, value in report['dependency_sha256'].items()),
                    'Report implementation is missing from dependency bindings: '+name)
        reports.append(report)
    result = compare_reports(reports)
    recheck_bindings(bindings, 'Comparison input changed')
    result['source_sha256'] = bindings
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--compare', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.compare:
        require(args.run is None and args.plan is None, 'Compare does not accept a run/plan')
        require(not args.out.exists(), 'Use a fresh comparison output')
        result = compare_report_files(args.compare)
        args.out.mkdir(parents=True); save(args.out/'comparison.json', result)
        print(json.dumps(dict(complete=True, out=str(args.out.resolve()), physical_simulations=0))); return
    require(args.run is not None and args.plan is not None, 'Run and frozen analysis plan are required')
    result = analyze(args.run, args.plan, args.out)
    print(json.dumps(dict(complete=True, out=str(args.out.resolve()), physical_simulations=0,
                         observer_cpu_seconds=result['observer_cpu_seconds'])))


if __name__ == '__main__':
    main()
