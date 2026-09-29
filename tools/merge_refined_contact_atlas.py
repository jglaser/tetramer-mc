#!/usr/bin/env python3
"""Replace sampled slot charts while preserving every original slot's mass.

Mapping is geometric/provenance information, never a native classifier. Every
old slot survives; untouched components are copied exactly. Independent fits
for one old slot receive equal shares of that slot's original proposal mass.
All source freezes, quality gates, component partitions and refinement stream
identities are checked before any output is written.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from prepare_mobile_reciprocal_benchmark import reciprocal_envelope, density_preflight
from prepare_mobile_posterior_pilot import validate_model


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def append(model, source, index, weight):
    for key in ['anchors', 'means', 'covariances']:
        model[key].append(copy.deepcopy(source[key][index]))
    model['weights'].append(float(weight))


def frozen_input(directory):
    """Verify the complete freeze and read model/metrics from verified bytes."""
    freeze_bytes = (directory/'freeze.json').read_bytes()
    freeze = json.loads(freeze_bytes)
    entries = freeze.get('files')
    require(isinstance(entries, dict), 'Frozen input must list file hashes')
    required = {'model.json', 'fit-metrics.json', 'manifest.json'}
    require(required <= entries.keys(), 'Freeze must cover model, fit metrics and manifest')
    payload = {'freeze.json': freeze_bytes}
    for name, digest in entries.items():
        relative = Path(name)
        require(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
                'Unsafe path in frozen input: '+name)
        path = directory/relative
        require(path.resolve().is_relative_to(directory), 'Frozen input path escapes its directory: '+name)
        raw = path.read_bytes()
        require(sha_bytes(raw) == digest, 'Frozen input changed: '+str(path))
        if name in required:
            payload[name] = raw
    manifest = json.loads(payload['manifest.json'])
    require(manifest.get('complete') is True, 'Input fit is not complete')
    require(manifest.get('model_sha256') == sha_bytes(payload['model.json']),
            'Fit manifest model hash differs from verified model')
    wrapped, metrics = json.loads(payload['model.json']), json.loads(payload['fit-metrics.json'])
    base = wrapped.get('base_model')
    require(isinstance(base, dict), 'Require an exact reciprocal model wrapper')
    count = validate_model(base, base.get('shape_sha256'))
    require(wrapped == reciprocal_envelope(base), 'Require the complete exact reciprocal envelope')
    rows = metrics.get('slots')
    require(isinstance(rows, list) and rows, 'Fit metrics must preserve all slots')
    used = []
    for index, row in enumerate(rows):
        require(row.get('global_slot') == index, 'Fit slot indices must be complete and ordered')
        indices = [row.get(key) for key in ['initial_component', 'refined_component']
                   if row.get(key) is not None]
        require(indices and all(type(i) is int and 0 <= i < count for i in indices),
                'Invalid slot component index')
        used.extend(indices)
    require(len(used) == count and sorted(used) == list(range(count)),
            'Slot components must partition the entire source model exactly once')
    return wrapped, metrics, payload


def check_refined_row(row):
    require(row.get('initial_component') is None, 'Use refined-only fit for replacements')
    require(row.get('exploration_gate_passed') is True, 'Refinement quality gate failed')
    unique = row.get('unique_physical_chart_poses')
    rank = row.get('empirical_rank')
    require(type(unique) is int and unique >= 2 and type(rank) is int and rank >= 1,
            'Replacement must have observed exploration; regularizer-only covariance is not learned')
    requested = row.get('requested_gate')
    require(isinstance(requested, dict), 'Missing explicit refinement gate thresholds')
    require(unique >= requested.get('minimum_unique_poses', math.inf)
            and rank >= requested.get('minimum_empirical_rank', math.inf)
            and row.get('minimum_retained_pca_ess_estimate', -1) >= requested.get('minimum_ess', math.inf),
            'Recorded refinement diagnostics do not satisfy the requested quality gate')
    seed = row.get('seeds', {}).get('refinement')
    require(type(seed) is int and 0 <= seed < 2**64,
            'Replacement requires its recorded independent refinement RNG seed')
    return seed


def merge(args):
    old, new, out = [Path(value).resolve() for value in [args.base_fit, args.refined_fit, args.out]]
    require(not out.exists(), 'Fresh output required')
    a, old_metrics, old_payload = frozen_input(old)
    b, new_metrics, new_payload = frozen_input(new)
    am, bm = a['base_model'], b['base_model']
    require(am['shape_sha256'] == bm['shape_sha256'] and am['angular_length'] == bm['angular_length'],
            'Replacement shape or angular units differ')
    ar, br = old_metrics['slots'], new_metrics['slots']
    mapping_bytes = Path(args.mapping).read_bytes()
    mapping = json.loads(mapping_bytes)
    require(isinstance(mapping, list) and len(mapping) == len(br) and mapping,
            'Mapping must contain exactly one original slot for every refined row')
    require(new_metrics.get('exploration_gate_passed') is True, 'Refined fit did not pass its quality gate')
    groups, streams = {}, set()
    for index, original_slot in enumerate(mapping):
        require(type(original_slot) is int and 0 <= original_slot < len(ar), 'Invalid original slot mapping')
        row = br[index]
        stream = check_refined_row(row)
        require(stream not in streams, 'Repeated refinement RNG stream cannot be counted as independent')
        streams.add(stream)
        x, y = row['reference_pose'], ar[original_slot]['reference_pose']
        require(np.allclose(x['position'], y['position'], rtol=0, atol=1e-9),
                'Mapped reference translations differ')
        require(abs(abs(np.dot(x['orientation'], y['orientation']))-1) < 1e-9,
                'Mapped reference orientations differ')
        groups.setdefault(original_slot, []).append(index)
    merged, old_subset, new_subset = [copy.deepcopy(am) for _ in range(3)]
    for model in [merged, old_subset, new_subset]:
        for key in ['anchors', 'means', 'covariances', 'weights']:
            model[key] = []
    audit = []
    for original_slot, row in enumerate(ar):
        indices = [row[key] for key in ['initial_component', 'refined_component'] if row[key] is not None]
        mass = math.fsum(am['weights'][index] for index in indices)
        before = len(merged['weights'])
        replacement_rows = groups.get(original_slot, [])
        if replacement_rows:
            for row_index in replacement_rows:
                component = br[row_index]['refined_component']
                append(merged, bm, component, mass/len(replacement_rows))
                append(new_subset, bm, component, mass/len(replacement_rows))
            for index in indices:
                append(old_subset, am, index, am['weights'][index])
        else:
            for index in indices:
                append(merged, am, index, am['weights'][index])
        actual = math.fsum(merged['weights'][before:])
        require(math.isclose(actual, mass, rel_tol=5e-15, abs_tol=0.), 'Original slot mass was changed')
        audit.append(dict(original_slot=original_slot, mass=mass, updated=bool(replacement_rows),
            old_components=indices, refined_rows=replacement_rows,
            refinement_streams=[br[i]['seeds']['refinement'] for i in replacement_rows],
            new_components=list(range(before, len(merged['weights'])))))
    validate_model(merged, am['shape_sha256'])
    checks = density_preflight(merged, reciprocal_envelope(merged), physical_poses=[], seed=20260928901)
    for subset in [old_subset, new_subset]:
        total = math.fsum(subset['weights'])
        require(total > 0, 'Replacement subset is empty')
        subset['weights'] = [weight/total for weight in subset['weights']]
        validate_model(subset, am['shape_sha256'])
    out.mkdir(parents=True)
    provenance = out/'provenance'
    provenance.mkdir()
    for label, payload in [('base', old_payload), ('refined', new_payload)]:
        directory = provenance/label
        directory.mkdir()
        for name, raw in payload.items():
            (directory/name).write_bytes(raw)
    (provenance/'mapping.json').write_bytes(mapping_bytes)
    (provenance/'merge_refined_contact_atlas.py').write_bytes(Path(__file__).read_bytes())
    for name, base in [('model', merged), ('old-updated-slots-only', old_subset), ('new-updated-slots-only', new_subset)]:
        write(out/(name+'.json'), reciprocal_envelope(base))
    write(out/'preflight.json', checks)
    write(out/'slot-map.json', audit)
    manifest = dict(complete=True, base_fit=str(old), refined_fit=str(new), mapping=mapping,
        base_model_sha256=sha_bytes(old_payload['model.json']), refined_model_sha256=sha_bytes(new_payload['model.json']),
        model_sha256=sha(out/'model.json'), mapping_sha256=sha_bytes(mapping_bytes), source_sha256=sha(__file__),
        updated_slots=len(groups), preserved_slots=len(ar)-len(groups),
        input_sha256={p.relative_to(provenance).as_posix():sha(p) for p in sorted(provenance.rglob('*')) if p.is_file()},
        scope='Frozen proposal replacement only. Every old slot and exact slot mass retained; no equilibrium basin weights or native-label selection. Existing production uniform branch is unchanged.',
        provenance_scope='Verified source freeze receipts and their model/metrics/manifests are archived. Transitive discovery artifacts remain in the original source directories under their verified hashes.')
    write(out/'manifest.json', manifest)
    write(out/'freeze.json', dict(files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(out=str(out), updated_slots=len(groups), preserved_slots=len(ar)-len(groups))))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['base-fit', 'refined-fit', 'mapping', 'out']:
        parser.add_argument('--'+name, required=True)
    merge(parser.parse_args())
