#!/usr/bin/env python3
"""Independent reference for a frozen rho=0 saved-candidate preflight.

This module never samples poses or Poisson clouds. A separately frozen run may
evaluate the declared saved-pose panel, using the existing Python contact
observer independently of the Rust preflight. Numerical density reconstruction
uses the actual map factors, not the direct atlas factors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from analyze_contact_efficiency import pose_arrays
from analyze_context_prior_pilot import SingleMovingPatchObserver
from dimer_destination_density import exported_covariance, map_cholesky, ordered_sum
from normalizer_proposal_density import scalar_cholesky
from prepare_smc_normalizer_atlas import Density, relative_poses, unwrap_proposal_model

PANEL_ORDINALS = tuple(range(0, 2304, 72))
LOG_ATOL = 2e-8
LOG_RTOL = 2e-10
REGIONS = ('hard_invalid', 'A_patch_0_0.25', 'A_patch_0.25_0.5',
           'A_patch_0.5_0.75', 'A_patch_0.75_1', 'A_patch_complete',
           'B', 'other_contact', 'unbound')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def finite_log(value, status):
    if status == 'negative_infinity':
        require(value is None, 'Negative infinity must use an explicit null scalar')
        return -math.inf
    require(status == 'finite' and type(value) in (int, float) and math.isfinite(value),
            'Invalid finite log-density representation')
    return float(value)


def close_log(actual, expected, name):
    if actual == expected == -math.inf:
        return dict(error=0., tolerance=0., ratio=0.)
    require(math.isfinite(actual) and math.isfinite(expected), 'Nonfinite disagreement: ' + name)
    error = abs(actual-expected)
    tolerance = LOG_ATOL+LOG_RTOL*(1+abs(actual)+abs(expected))
    require(error <= tolerance, 'Independent density mismatch: ' + name)
    return dict(error=error, tolerance=tolerance, ratio=error/tolerance)


def full_log_q(log_g, pose, halfwidth, uniform_probability=.5):
    require(math.isfinite(halfwidth) and halfwidth > 0 and 0 < uniform_probability < 1,
            'Invalid immutable uniform mixture')
    require(math.isfinite(log_g) or log_g == -math.inf, 'Invalid learned density')
    p = pose['position']
    require(len(p) == 3 and all(math.isfinite(v) for v in p), 'Invalid position')
    inside = all(abs(v) <= halfwidth for v in p)
    log_u = -3*math.log(2*halfwidth) if inside else -math.inf
    return float(np.logaddexp(math.log(uniform_probability)+log_u,
                             math.log1p(-uniform_probability)+log_g)), log_u, inside


class MapDensityReference:
    """Normalized physical law of the frozen DockingProposal map charts."""
    def __init__(self, model):
        self.model = model
        base, flags = unwrap_proposal_model(model)
        self.density = Density(model)
        factors = np.asarray([map_cholesky(exported_covariance(scalar_cholesky(c)))
                              for c in base['covariances']])
        self.density.lower = factors[self.density.base_indices]
        self.density.logdet = np.asarray([
            ordered_sum(math.log(float(lower[i, i])) for i in range(6))
            for lower in self.density.lower])
        total = ordered_sum(base['weights'])
        self.base_log_prior = np.asarray([
            math.log((float(base['weights'][k])/total)/(2 if flags[k] else 1))
            for k in self.density.base_indices])
        self.density.weights = np.exp(self.base_log_prior)
        self.branches = [dict(component_index=int(k), inverted=bool(flag))
                         for k, flag in zip(self.density.base_indices, self.density.inverted)]

    def evaluate(self, poses, anchor, log_prior=None):
        prior = self.base_log_prior if log_prior is None else np.asarray(log_prior, float)
        require(prior.shape == self.base_log_prior.shape and np.isfinite(prior).all()
                and abs(float(logsumexp(prior))) <= 1e-12, 'Unnormalized or incomplete virtual prior')
        relative = relative_poses(poses, anchor)
        _, norms, weighted = self.density.evaluate(relative)
        component_logs = weighted-self.base_log_prior[None, :]
        return logsumexp(component_logs+prior[None, :], axis=1), component_logs, norms


def canonical_tokens(tokens):
    result = []
    for token in tokens:
        require(isinstance(token, (list, tuple)) and len(token) == 4,
                'Malformed patch token')
        a, b, pa, pb = token
        require(type(a) is int and type(b) is int and a < b and 77 in (a, b)
                and isinstance(pa, str) and pa and isinstance(pb, str) and pb,
                'Patch token must use canonical global body IDs and nonempty patch labels')
        result.append((a, b, pa, pb))
    require(len(set(result)) == len(result), 'Repeated patch token')
    return set(result)


def region_from_tokens(tokens, source217, hard_valid=True):
    reference = canonical_tokens(source217)
    require(reference and all(t[:2] == (77, 217) for t in reference), 'Invalid source217 reference')
    if not hard_valid:
        require(tokens is None, 'Invalid pose must not be classified as physical contact support')
        return dict(region='hard_invalid', neighbors=None, source217_intersection=None,
                    source217_union=None, source217_inclusion=None, source217_jaccard=None,
                    source217_bin=None)
    present = canonical_tokens(tokens)
    neighbors = sorted({b if a == 77 else a for a, b, _, _ in present})
    secondary = {t for t in present if t[:2] == (77, 217)}
    common, union = len(secondary & reference), len(secondary | reference)
    # Integer comparisons make the declared quarter-bin boundaries explicit.
    index = 4 if common == len(reference) else (4*common)//len(reference)
    if neighbors == [16, 217]:
        name = 'A'
    elif neighbors == [16, 56]:
        name = 'B'
    else:
        name = 'other_contact' if neighbors else 'unbound'
    return dict(region=name, neighbors=neighbors, source217_intersection=common,
                source217_union=union, source217_inclusion=common/len(reference),
                source217_jaccard=common/union, source217_bin=index if name == 'A' else None)


def validate_bank_row(row, ordinal, identity, expected_cycles=2304):
    require(type(ordinal) is int and 0 <= ordinal < expected_cycles, 'Bad bank ordinal')
    require(row['ordinal'] == ordinal and row['cycle'] == ordinal+1 and row['slot'] == 4
            and row['attempt_index'] == 5*ordinal+4, 'Missing, duplicated, or reordered candidate')
    require(row['identity'] == identity and type(row['event_index']) is int
            and row['event_index'] == 17+16*ordinal, 'Wrong source identity/event')
    require(row['production'] is (row['cycle'] > 256), 'Wrong warmup/production label')
    require(type(row['accepted']) is bool and row['branch'] in ('uniform', 'involution'),
            'Wrong proposal provenance')
    require(type(row['wall_valid']) is bool and
            ((row['wall_valid'] and type(row['core_valid']) is bool)
             or (not row['wall_valid'] and row['core_valid'] is None)), 'Wrong delayed core status')
    valid = row['wall_valid'] and row['core_valid']
    require(row['status'] == ('accepted' if row['accepted'] else
            'wall_rejected' if not row['wall_valid'] else
            'core_rejected' if not row['core_valid'] else 'bath_rejected'), 'Wrong complete disposition')
    require(not row['accepted'] or valid, 'Hard-invalid candidate cannot have been accepted')
    saved = row['saved_log_g']
    require((row['branch'] == 'uniform' and saved is None)
            or (row['branch'] == 'involution' and type(saved) in (int, float) and math.isfinite(saved)),
            'Wrong saved learned density schema')
    # Validates finite normalized orientation and positions, without atom geometry.
    pose_arrays([row['proposed_pose']])
    return bool(valid)


def expected_region(details):
    if details['region'] == 'A':
        return REGIONS[1+details['source217_bin']]
    return None if details['region'] == 'hard_invalid' else details['region']


def audit_scalar_row(row, bank, identity, halfwidth, source217, expected_cycles=2304):
    """Validate every saved row, without evaluating atom geometry or G again."""
    valid = validate_bank_row(bank, bank['ordinal'], identity, expected_cycles)
    require(row['kind'] == 'candidate' and row['complete'] is True and row['input'] == bank,
            'Preflight candidate differs from its exact immutable bank input')
    actual = row['actual']
    require(actual['wall_valid'] is bank['wall_valid']
            and actual['core_valid'] is bank['core_valid']
            and actual['physical_valid'] is valid and actual['verdict_matches'] is True,
            'New geometry disposition differs from audited physical candidate')
    require(row['physical_zero'] is (not valid), 'Wrong explicit hard-invalid zero')
    density = row['density']
    log_g = finite_log(density['log_g'], density['log_g_status'])
    require(type(density['log_q']) in (int, float) and math.isfinite(density['log_q']),
            'A generated candidate must have finite full proposal density')
    log_q, log_u, contains = full_log_q(log_g, bank['proposed_pose'], halfwidth)
    checks = [close_log(density['log_q'], log_q, 'complete50/50mixture')]
    require(density['uniform_contains'] is contains, 'Wrong uniform cube membership')
    if contains:
        checks.append(close_log(density['log_u'], log_u, 'normalized Haar uniform'))
    else:
        require(density['log_u'] is None, 'Outside-cube density must be null/-infinity')
    if bank['branch'] == 'involution':
        checks.append(close_log(log_g, bank['saved_log_g'], 'all saved learned endpoint densities'))
        checks.append(close_log(density['saved_log_g_delta'], log_g-bank['saved_log_g'],
                                'recorded saved-density residual'))
    else:
        require(density['saved_log_g_delta'] is None, 'Uniform branch lacks a saved learned score')
    if not valid:
        require(row['patches'] is None and row['region'] is None and row['envelope'] is None,
                'Hard-invalid candidate must remain an explicit zero with no physical classification')
        return checks
    patches = row['patches']
    tokens = patches['tokens']
    canonical = canonical_tokens(tokens)
    require(tokens == [list(t) for t in sorted(canonical)], 'Patch output is not sorted/canonical')
    reference = region_from_tokens(tokens, source217)
    require(row['region'] == expected_region(reference)
            and patches['neighbor_labels'] == reference['neighbors'], 'Wrong exhaustive contact region')
    for output_name, reference_name in [('source_intersection', 'source217_intersection'),
            ('source_union', 'source217_union'), ('source_fraction', 'source217_inclusion'),
            ('jaccard', 'source217_jaccard')]:
        require(patches[output_name] == reference[reference_name], 'Wrong frozen-source patch statistic')
    require(patches['secondary_tokens'] == [list(t) for t in sorted(canonical) if t[:2] == (77, 217)],
            'Secondary patch inventory was filtered')
    # Same immutable canonical serialization as the held physical observer.
    from context_prior_metrics import fingerprint
    require(patches['fingerprint'] == fingerprint(canonical), 'Patch fingerprint differs')
    envelope = row['envelope']
    require(envelope is not None, 'Valid candidate lacks allocated envelope')
    lo, hi, uncertain = (envelope[k] for k in ('lower_volume', 'upper_volume', 'uncertain_volume'))
    require(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (lo, hi, uncertain))
            and lo <= hi and math.isclose(hi-lo, uncertain, rel_tol=2e-12, abs_tol=1e-10),
            'Invalid geometric envelope interval')
    labels = envelope['fixed_labels']
    require(labels == sorted(set(labels)) and all(type(x) is int and 0 <= x < 264 and x != 77 for x in labels)
            and set(reference['neighbors']) <= set(labels), 'Envelope omits an observed contact or mislabels the context')
    return checks


def audit_preflight_events(lines, rows, source217):
    """All attempted stages and completion rows remain paired, with no skips."""
    iterator = iter(lines)
    index = 0
    def event(kind):
        nonlocal index
        try:value = json.loads(next(iterator))
        except StopIteration as error:raise ValueError('Incomplete preflight event prefix') from error
        require(value['kind'] == kind and value['event_index'] == index, 'Reordered/incomplete preflight event journal')
        index += 1
        return value
    event('source_begun')
    source = event('source_complete')
    source_tokens = canonical_tokens(source['tokens'])
    require({t for t in source_tokens if t[:2] == (77, 217)} == canonical_tokens(source217)
            and source['neighbor_labels'] == [16, 217], 'Reclassified source reference differs')
    for ordinal, row in enumerate(rows):
        begun = event('candidate_begun')
        require(begun['ordinal'] == ordinal and begun['input'] == row['input'], 'Candidate begin/output mismatch')
        density = event('density_complete')
        require(density['ordinal'] == ordinal and density['density'] == row['density'], 'Density stage/output mismatch')
        geometry = event('geometry_complete')
        require(geometry['ordinal'] == ordinal and geometry['wall_valid'] is row['actual']['wall_valid']
                and geometry['core_valid'] is row['actual']['core_valid'], 'Geometry stage/output mismatch')
        if row['actual']['physical_valid']:
            patches = event('patches_complete')
            require(patches['ordinal'] == ordinal and patches['patches'] == row['patches']
                    and patches['region'] == row['region'], 'Patch stage/output mismatch')
            envelope = event('envelope_complete')
            require(envelope['ordinal'] == ordinal and envelope['envelope'] == row['envelope'], 'Envelope stage/output mismatch')
        require(event('candidate_complete')['ordinal'] == ordinal, 'Missing complete candidate marker')
    require(next(iterator, None) is None, 'Additional/unclassified preflight events')
    return index


def coverage_summary(rows, density=.035, intensity_ratio=64.):
    """Unconditional hard-volume and budget diagnostics, never depletion weights."""
    require(rows, 'Empty population is not an integration estimate')
    n = len(rows)
    groups = {region: [] for region in REGIONS}
    for row in rows:
        region = row['region'] if row['actual']['physical_valid'] else 'hard_invalid'
        require(region in groups, 'Unclassified candidate')
        groups[region].append(row)
    partitions = []
    for region in REGIONS:
        selected = groups[region]
        log_terms = [-r['density']['log_q'] for r in selected if r['actual']['physical_valid']]
        log_mass = float(logsumexp(log_terms))-math.log(n) if log_terms else -math.inf
        mass = math.exp(log_mass) if math.isfinite(log_mass) else 0.
        partitions.append(dict(region=region, attempted_count=len(selected),
            unconditional_fraction=len(selected)/n,
            hard_only_importance_mass=mass,
            log_hard_only_importance_mass=log_mass if math.isfinite(log_mass) else None))
    valid = [r for r in rows if r['actual']['physical_valid']]
    lower = [float(r['envelope']['lower_volume']) for r in valid]
    upper = [float(r['envelope']['upper_volume']) for r in valid]
    def describe(values):
        return dict(n=len(values), minimum=min(values) if values else None,
                    maximum=max(values) if values else None,
                    mean=math.fsum(values)/len(values) if values else None)
    uncertain = [float(r['envelope']['uncertain_volume']) for r in valid]
    expected_points = 2*density*intensity_ratio*math.fsum(uncertain)
    return dict(attempted=n, hard_valid=len(valid), hard_invalid=n-len(valid),
        denominator=n, partitions=partitions,
        full_source217_inclusion_hits=sum(r['patches']['source_fraction'] == 1 for r in valid),
        full_source217_inclusion_A_hits=len(groups['A_patch_complete']),
        log_physical_weight_lower=describe([density*v for v in lower]),
        log_physical_weight_upper=describe([density*v for v in upper]),
        log_interval_width=describe([density*(hi-lo) for lo, hi in zip(lower, upper)]),
        total_envelope_lower_volume=math.fsum(lower), total_envelope_upper_volume=math.fsum(upper),
        total_envelope_uncertain_volume=math.fsum(uncertain),
        expected_complete_allocation_raw_points=expected_points,
        cloud_coefficient=2*density*intensity_ratio,
        cloud_scope='Expected raw draws for the declared complete two-cloud allocation: 2*lambda*sum(uncertain_volume). This is not a bound on realized Poisson counts. Runtime/cap stopping can lower processed work; no clouds are sampled here.',
        mass_scope='Hard-only importance sum H_wall*H_core/Q divided by every attempted draw. No Poisson/depletion weights and no equilibrium-contact conclusion.',
        envelope_scope='Saved geometric bounds and z-times-volume diagnostics only; ordinary floating-point geometry obligations remain explicit.')


class GeometryReference:
    """Independent saved-pose wall/core predicates, then held patch observer.

    The atom-level hard test is separate because the retained-state observer
    rejects hard-invalid poses. Pruning guards do not change strict predicates.
    """
    def __init__(self, shape, patch_map, context, wall_radius, rd):
        self.patch = SingleMovingPatchObserver(shape, patch_map, context, wall_radius, rd)
        self.atoms = self.patch.observer.atoms
        self.radii = self.patch.observer.radii
        self.bound = self.patch.observer.bound
        self.radius = wall_radius
        self.fixed_world = {}
        self.geometry_calls = 0

    def world(self, pose):
        position, rotation = pose_arrays([pose])
        rotated = ((rotation[0, :, 0]*self.atoms[:, 0, None]
                    + rotation[0, :, 1]*self.atoms[:, 1, None])
                    + rotation[0, :, 2]*self.atoms[:, 2, None])
        return rotated+position[0]

    def classify(self, pose):
        self.geometry_calls += 1
        moving = self.world(pose)
        available = self.radius-self.radii
        squared = (moving[:, 0]**2+moving[:, 1]**2)+moving[:, 2]**2
        wall_gap = available-np.sqrt(squared)
        wall_valid = bool(np.all(available >= 0) and np.all(squared <= available**2))
        result = dict(wall_valid=wall_valid, core_valid=None, patch_tokens=None,
                      minimum_atomic_wall_gap=float(wall_gap.min()), minimum_checked_core_gap=None)
        if not wall_valid:
            return result
        pos = np.asarray(pose['position'])
        guard = 512*np.finfo(float).eps*(1+np.linalg.norm(pos)
                + np.linalg.norm(self.patch.positions, axis=1)+self.bound)
        candidates = np.flatnonzero(np.linalg.norm(self.patch.positions-pos, axis=1)
                                    <= 2*self.bound+guard)
        left_tree = cKDTree(moving)
        minimum = math.inf
        for index in candidates:
            label = self.patch.labels[int(index)]
            if label not in self.fixed_world:
                self.fixed_world[label] = self.world(self.patch.fixed[label])
            fixed = self.fixed_world[label]
            near = left_tree.sparse_distance_matrix(cKDTree(fixed),
                2*float(self.radii.max())+float(guard[index]), output_type='ndarray')
            if len(near):
                aa, bb = near['i'], near['j']
                delta = moving[aa]-fixed[bb]
                d2 = (delta[:, 0]**2+delta[:, 1]**2)+delta[:, 2]**2
                radii = self.radii[aa]+self.radii[bb]
                minimum = min(minimum, float(np.min(np.sqrt(d2)-radii)))
                if np.any(d2 < radii**2):
                    return dict(result, core_valid=False, minimum_checked_core_gap=minimum)
        tokens = self.patch.classify(pose)['patch_tokens']
        return dict(result, core_valid=True, patch_tokens=tokens,
                    minimum_checked_core_gap=minimum if math.isfinite(minimum) else None)


def bind_map_source(path):
    """Bind reviewed factor, coordinate and export arithmetic without executing Rust."""
    from analyze_context_transport_stationarity import verify_compiled_chart_source
    from normalizer_proposal_density import bind_source_bundle
    from audit_proposal_density_probe import SOURCE_EXCERPTS
    bundle = read(path)
    for name in ('src/basin_involution.rs', 'src/proposal.rs', 'src/docking.rs'):
        entry = bundle['files'][name]
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'],
                'Corrupt compiled source entry')
    verify_compiled_chart_source(bundle['files']['src/basin_involution.rs']['text'])
    bind_source_bundle(path)
    for name, start, end, expected in SOURCE_EXCERPTS[:2]:
        text = bundle['files'][name]['text']
        first = text.index(start)
        require(hashlib.sha256(text[first:text.index(end, first)].encode()).hexdigest() == expected,
                'Unreviewed map constructor/export arithmetic')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    require(protocol['schema'] == 'context-candidate-bank-audit-v1'
            and protocol['panel_ordinals'] == list(PANEL_ORDINALS), 'Changed fixed audit allocation')
    require(not args.out.exists(), 'Fresh independent audit output required')
    bindings = protocol['input_sha256']
    for path, digest in bindings.items():
        require(sha(path) == digest, 'Changed audit input: '+path)

    def bound(path):
        key = str(Path(path).resolve())
        require(key in bindings, 'Unbound independent audit input: '+key)
        return read(key)

    inventory = bound(protocol['bank_inventory'])
    extracted = bound(protocol['bank_summary'])
    require(extracted['complete'] and extracted['passed'] and extracted['records'] == 36864
            and extracted['inventory_sha256'] == sha(protocol['bank_inventory']),
            'Incomplete bank extraction')
    regions = bound(protocol['regions'])
    require(extracted['regions_sha256'] == sha(protocol['regions'])
            and regions['a_neighbors'] == [16, 217] and regions['b_neighbors'] == [16, 56]
            and regions['secondary_label'] == 217
            and regions['inclusion_boundaries'] == [0., .25, .5, .75, 1.], 'Changed frozen regions')
    source217 = regions['source_secondary_tokens']
    require(len(canonical_tokens(source217)) == 16, 'Wrong full source reference')
    expected = {(start, stream, arm)
                for start in ('saved_body77', 'highest_original_prior_valid_neighbor_distinct_center')
                for stream in range(4) for arm in ('original', 'context')}
    require(len(inventory) == 16 and {(r['start'], r['stream'], r['arm']) for r in inventory} == expected,
            'Missing candidate populations')
    ids = {r['id'] for r in inventory}
    preflights = protocol['preflights']
    require(len(preflights) == 16 and {r['id'] for r in preflights} == ids,
            'All16 complete preflights required before any independent geometry')
    completed = {}
    for item in preflights:
        directory = Path(item['result'])
        summary = bound(directory/'summary.json')
        receipt = bound(item['execution_receipt'])
        require(receipt['success'] and receipt['child_drained'] and receipt['returncode'] == 0
                and Path(receipt['terminal']['path']).resolve() == (directory/'summary.json').resolve()
                and receipt['terminal']['sha256'] == sha(directory/'summary.json'),
                'Preflight not completed/drained')
        require(summary['schema'] == 'context-candidate-bank-preflight-v1'
                and summary['complete'] and summary['passed']
                and summary['attempted_records'] == summary['completed_records'] == 2304
                and summary['clouds_generated'] == summary['new_poses_generated'] == 0
                and summary['source_reference_checked'] is True, 'Incomplete preflight')
        completed[item['id']] = (item, summary)
    require(str(Path(protocol['source_bundle']).resolve()) in bindings, 'Unbound compiled source')
    bind_map_source(Path(protocol['source_bundle']))
    args.out.mkdir()
    started = time.process_time()
    maps, geometries = {}, {}
    populations, panel_results, scalar_checks = [], [], []
    with (args.out/'attempts.jsonl').open('x') as journal:
        def emit(row):
            journal.write(json.dumps(row, allow_nan=False)+'\n');journal.flush()
        try:
            for bank_item in inventory:
                emit(dict(kind='population_begun', id=bank_item['id']))
                item, summary = completed[bank_item['id']]
                directory = Path(item['result'])
                cfg = bound(item['config'])
                original = bound(cfg['invocation_config']['path'])
                require(cfg['invocation_config']['sha256'] == sha(cfg['invocation_config']['path'])
                        == bank_item['config_sha256'] and original['identity'] == cfg['identity'],
                        'Wrong original invocation binding')
                require(cfg['bank_sha256'] == bank_item['bank_sha256'] == sha(bank_item['bank'])
                        and cfg['expected_records'] == 2304 and cfg['expected_virtual_branches'] == 2048,
                        'Bank/branch allocation differs')
                require(cfg['regions'] == {k: regions[k] for k in (
                    'a_neighbors', 'b_neighbors', 'secondary_label', 'source_secondary_tokens', 'inclusion_boundaries')},
                    'Preflight region definitions changed')
                require(original['uniform_probability'] == .5 and original['depletant_radius'] == 1.5
                        and original['reservoir_density'] == .035 and original['poisson_lambda_ratio'] == 64.,
                        'Wrong physical proposal law')
                producer_protocol = bound(directory/'protocol.json')
                require(producer_protocol['config'] == cfg
                        and producer_protocol['config_sha256'] == sha(item['config'])
                        and producer_protocol['source_bundle_sha256'] == sha(protocol['source_bundle'])
                        and producer_protocol['no_poisson_clouds'] is True, 'Preflight provenance mismatch')
                model = bound(cfg['model']['path'])
                require(cfg['model']['sha256'] == sha(cfg['model']['path']) == bank_item['model_sha256'],
                        'Model digest differs')
                key = cfg['model']['sha256']
                if key not in maps:
                    maps[key] = MapDensityReference(model)
                reference = maps[key]
                prior = None
                if cfg['prior'] is not None:
                    require(bank_item['arm'] == 'context'
                            and cfg['prior']['sha256'] == bank_item['prior_sha256'] == sha(cfg['prior']['path']),
                            'Wrong context prior')
                    prior = bound(cfg['prior']['path'])['log_prior']
                else:
                    require(bank_item['arm'] == 'original' and bank_item['prior'] is None, 'Missing context prior')
                manifest = bound(directory/'chart-manifest.json')
                require(len(manifest['charts']) == len(reference.branches) == 2048
                        and manifest['angular_length'] == reference.density.ell
                        and manifest['uniform_probability'] == .5, 'Chart manifest inventory differs')
                effective_prior = reference.base_log_prior if prior is None else prior
                for label, (chart, branch) in enumerate(zip(manifest['charts'], reference.branches)):
                    require(chart['virtual_label'] == label and chart['component_index'] == branch['component_index']
                            and chart['inverted'] is branch['inverted'], 'Changed reciprocal virtual ordering')
                    scalar_checks.append(close_log(chart['effective_log_prior'], float(effective_prior[label]), 'prior'))
                    require(np.allclose(chart['reconstructed_map_lower'], reference.density.lower[label],
                                        rtol=2e-14, atol=2e-14), 'Exported manifest map factor differs')
                def physical_asset(name):
                    path = Path(original[name])
                    if not path.is_absolute():path = Path(cfg['invocation_config']['path']).parent/path
                    value = bound(path)
                    require(sha(path) == original['expected_sha256'][name], 'Changed physical asset')
                    return value
                shape, context, source = (physical_asset(k) for k in ('shape', 'fixed_context', 'source_state'))
                require(context['excluded_moving_labels'] == [77] and context['anchor_label'] == 16
                        and [b['label'] for b in context['bodies']] == [i for i in range(264) if i != 77],
                        'Moving body or frame contamination')
                anchor = next(b['pose'] for b in context['bodies'] if b['label'] == 16)
                require(anchor == source['anchor_pose'] == manifest['anchor_pose'], 'Anchor pose differs')
                patch_map = bound(cfg['patch_map']['path'])
                require(cfg['patch_map']['sha256'] == sha(cfg['patch_map']['path'])
                        and patch_map['shape_sha256'] == original['expected_sha256']['shape'], 'Changed patch map')
                geometry_key = tuple(original['expected_sha256'][k] for k in ('shape', 'fixed_context', 'source_state'))
                if geometry_key not in geometries:
                    geometries[geometry_key] = GeometryReference(shape, patch_map['atom_patch_ids'], context,
                        source['spherical_wall_radius'], original['depletant_radius'])
                geometry = geometries[geometry_key]
                half = manifest['uniform_half_width']
                close_log(half, source['spherical_wall_radius']+geometry.bound, 'immutable uniform half width')
                require(str(Path(bank_item['bank']).resolve()) in bindings
                        and str((directory/'rows.jsonl').resolve()) in bindings,
                        'Missing row-file pin')
                banks = [json.loads(line) for line in Path(bank_item['bank']).read_text().splitlines()]
                rows = [json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
                require(len(banks) == len(rows) == 2304, 'Missing or additional preflight rows')
                for ordinal, (bank, row) in enumerate(zip(banks, rows)):
                    require(bank['ordinal'] == ordinal, 'Candidate output order changed')
                    scalar_checks.extend(audit_scalar_row(row, bank, cfg['identity'], half, source217))
                require(sum(r['actual']['physical_valid'] for r in rows) == summary['physical_valid_records']
                        and sum(not r['actual']['physical_valid'] for r in rows) == summary['physical_zero_records'],
                        'Preflight disposition counts differ')
                actual_counts = {name: 0 for name in REGIONS if name != 'hard_invalid'}
                actual_counts.update(Counter(r['region'] for r in rows if r['actual']['physical_valid']))
                require(actual_counts == summary['regions'], 'Preflight region count mismatch')
                require(str((directory/'events.jsonl').resolve()) in bindings, 'Unbound attempt journal')
                with (directory/'events.jsonl').open() as events:
                    count = audit_preflight_events(events, rows, source217)
                require(count == summary['journal_events'], 'Preflight event denominator differs')
                aggregate = coverage_summary(rows)
                close_log(summary['uncertain_volume_sum'], aggregate['total_envelope_uncertain_volume'],
                          'all candidate envelope volumes')
                close_log(summary['prospective_two_cloud_expected_raw_points'],
                          aggregate['expected_complete_allocation_raw_points'], 'declared cloud work')
                for phase, selected in [('all', rows), ('warmup', rows[:256]), ('production', rows[256:])]:
                    populations.append(dict(id=bank_item['id'], arm=bank_item['arm'], start=bank_item['start'],
                        stream=bank_item['stream'], seed=bank_item['seed'], phase=phase, **coverage_summary(selected)))
                emit(dict(kind='scalar_population_complete', id=bank_item['id'], rows=2304))
                panel = [rows[i] for i in PANEL_ORDINALS]
                emit(dict(kind='density_panel_begun', id=bank_item['id'], ordinals=list(PANEL_ORDINALS)))
                values, _, _ = reference.evaluate([r['input']['proposed_pose'] for r in panel], anchor, prior)
                for row, expected_g in zip(panel, values):
                    ordinal = row['input']['ordinal']
                    emit(dict(kind='geometry_panel_pose_begun', id=bank_item['id'], ordinal=ordinal))
                    numeric = close_log(finite_log(row['density']['log_g'], row['density']['log_g_status']),
                                        float(expected_g), 'independent panel G')
                    expected_q, _, _ = full_log_q(float(expected_g), row['input']['proposed_pose'], half)
                    q_numeric = close_log(row['density']['log_q'], expected_q, 'independent panel Q')
                    physical = geometry.classify(row['input']['proposed_pose'])
                    require(physical['wall_valid'] is row['actual']['wall_valid']
                            and physical['core_valid'] is row['actual']['core_valid'], 'Independent panel hard verdict differs')
                    if physical['core_valid'] is True:
                        require([list(t) for t in physical['patch_tokens']] == row['patches']['tokens'],
                                'Independent panel patch inventory differs')
                    result = dict(id=bank_item['id'], ordinal=ordinal, branch=row['input']['branch'],
                        numeric_logG=numeric, numeric_logQ=q_numeric, geometry=physical,
                        region=row['region'], no_pose_replacement=True)
                    panel_results.append(result);emit(dict(kind='panel_pose_complete', **result))
                emit(dict(kind='population_complete', id=bank_item['id']))
            require(len(panel_results) == 512 and len(populations) == 48, 'Incomplete declared audit')
            full = [r for r in populations if r['phase'] == 'all']
            zero_hits = [r['id'] for r in full if r['full_source217_inclusion_A_hits'] == 0]
            zero_any = [r['id'] for r in full if r['full_source217_inclusion_hits'] == 0]
            report = dict(schema='context-candidate-bank-independent-audit-v1', complete=True, passed=True,
                physical_weight_status='not_estimated', input_sha256=bindings,
                protocol_sha256=sha(args.protocol), source_sha256=sha(__file__),
                audited_candidates=36864, independent_panel_size=512, panel_ordinals=list(PANEL_ORDINALS),
                scalar_checks=len(scalar_checks), maximum_scalar_check_ratio=max(r['ratio'] for r in scalar_checks),
                populations=populations, panel=panel_results,
                coverage_gate=dict(status='limited' if zero_hits else 'observed_hits_only',
                    populations_without_A_sourceT_pocket_hits=zero_hits,
                    populations_without_any_sourceT_complete_hits=zero_any,
                    interpretation='Defined sourceT patch-pocket coverage only; no hits is not a zero physical mass or exclusion of every native/equivalent pocket. Observed hits do not certify adequate importance ESS.'),
                new_pose_draws=0, clouds_generated=0, geometry_panel_classifications=sum(g.geometry_calls for g in geometries.values()),
                cpu_seconds=time.process_time()-started,
                interpretation='Complete/passed means provenance, arithmetic and fixed-panel agreement, not thermodynamic convergence. Hard-only masses retain every attempted denominator; no depletion estimator or assembly conclusion.')
            for path, digest in bindings.items():require(sha(path) == digest, 'Input changed during audit')
            with (args.out/'report.json').open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False);stream.write('\n')
        except BaseException as error:
            emit(dict(kind='fatal', error=repr(error), completed_populations=len(populations)//3,
                      completed_panel_poses=len(panel_results)))
            with (args.out/'failure.json').open('x') as stream:
                json.dump(dict(complete=False, passed=False, error=repr(error),
                    physical_weight_status='not_estimated', prefix_preserved=True), stream, indent=2)
            raise


if __name__ == '__main__':
    main()
