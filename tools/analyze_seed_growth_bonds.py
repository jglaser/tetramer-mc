#!/usr/bin/env python3
"""Compare initial seed bond types with new contacts in a frozen trajectory prefix.

Reuses authenticated viewer entry labels, classifies only uncached saved frames,
and reports monomer families separately from registered tetramer motif families.
No hysteresis, intrinsic rigid-body bonds, physical rates or equilibrium claims.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_cache(path):
    raw = path.read_bytes()
    if path.suffix == '.html':
        match = re.search(rb'<script id="trajectory-data" type="application/json">(.*?)</script>', raw, re.S)
        require(match is not None, 'Missing viewer data')
        return json.loads(match.group(1)), digest(raw)
    return json.loads(raw), digest(raw)


def inverse_families(motifs):
    positions = np.array([m['relative_position'] for m in motifs])
    rotations = Rotation.from_quat(np.array([m['relative_orientation'] for m in motifs])[:, [1, 2, 3, 0]]).as_matrix()
    result = {}
    for k, motif in enumerate(motifs):
        t, r = -rotations[k].T @ positions[k], rotations[k].T
        candidates = [j for j in range(len(motifs))
                      if np.linalg.norm(positions[j]-t) < 1e-6
                      and np.linalg.norm(rotations[j]-r) < 1e-6]
        require(len(candidates) == 1, 'Ambiguous inverse motif')
        other = motifs[candidates[0]]['id']
        result[motif['id']] = '/'.join(map(str, sorted({motif['id'], other})))
    return result


def seed_connected(n, pairs, seeds):
    adjacency = [set() for _ in range(n)]
    for i, j in pairs:
        adjacency[i].add(j); adjacency[j].add(i)
    found = set(seeds); stack = list(seeds)
    while stack:
        for j in adjacency[stack.pop()] - found:
            found.add(j); stack.append(j)
    return found


def summarize(frames, field, seeds):
    """Rows are {family,bodies,members?}; unique bond identities, not iid events."""
    def key(row):
        return tuple(row['bodies']) + tuple(row.get('members', []))
    initial = defaultdict(set)
    baseline_all = defaultdict(set)
    for row in frames[0][field]:
        baseline_all[row['family']].add(key(row))
        if set(row['bodies']) <= seeds:
            initial[row['family']].add(key(row))
    ever = defaultdict(lambda: defaultdict(set))
    first = {}
    onsets = Counter()
    previous = {(r['family'], key(r)) for r in frames[0][field]}
    families = set(initial)
    for frame in frames[1:]:
        rows = frame[field]
        connected = seed_connected(frame['bodies'], [r['bodies'] for r in rows], seeds)
        present = {(r['family'], key(r)) for r in rows}
        for r in rows:
            family, bond = r['family'], key(r)
            families.add(family)
            if set(r['bodies']) <= seeds or bond in baseline_all[family]:
                continue
            group = 'seed_newcomer' if set(r['bodies']) & seeds else 'newcomer_newcomer'
            ever[family][group].add(bond)
            ever[family]['all_newcomer'].add(bond)
            if set(r['bodies']) <= connected:
                ever[family]['seed_connected'].add(bond)
            first.setdefault(family, frame['sweep'])
            if (family, bond) not in previous:
                onsets[family] += 1
        previous = present
    output = []
    last = frames[-1][field]
    connected = seed_connected(frames[-1]['bodies'], [r['bodies'] for r in last], seeds)
    for family in sorted(families):
        current = defaultdict(set)
        for row in last:
            if row['family'] != family:
                continue
            group = 'seed_seed' if set(row['bodies']) <= seeds else ('seed_newcomer' if set(row['bodies']) & seeds else 'newcomer_newcomer')
            current[group].add(key(row))
            if not set(row['bodies']) <= seeds and set(row['bodies']) <= connected:
                current['seed_connected_newcomer'].add(key(row))
        output.append(dict(family=family, initial_seed_bonds=len(initial[family]),
            initial_seed_bond_ids=sorted(initial[family]),
            distinct_newcomer_bonds={k:len(ever[family][k]) for k in
                ('seed_newcomer','newcomer_newcomer','seed_connected','all_newcomer')},
            distinct_newcomer_bond_ids={k:sorted(ever[family][k]) for k in
                ('seed_newcomer','newcomer_newcomer','seed_connected','all_newcomer')},
            first_saved_newcomer_sweep=first.get(family), sampled_onsets=onsets[family],
            latest_bonds={k:len(current[k]) for k in
                ('seed_seed','seed_newcomer','newcomer_newcomer','seed_connected_newcomer')}))
    return dict(families=output,
        absent_during_growth=[r['family'] for r in output if r['initial_seed_bonds'] and not r['distinct_newcomer_bonds']['all_newcomer']],
        absent_directly_at_seed=[r['family'] for r in output if r['initial_seed_bonds'] and not r['distinct_newcomer_bonds']['seed_newcomer']],
        absent_from_seed_connected_growth=[r['family'] for r in output if r['initial_seed_bonds'] and not r['distinct_newcomer_bonds']['seed_connected']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('/home/xvg/protein-nucleation'))
    args = parser.parse_args()
    run, reference = args.run.resolve(), args.reference.resolve()
    require(not args.out.exists(), 'Fresh analysis output directory required')
    config_raw = (run/'config.json').read_bytes(); config = json.loads(config_raw)
    shape_raw = (run/'provenance/shape.json').read_bytes(); shape = json.loads(shape_raw)
    require(config['boundary']['kind'] == 'spherical', 'This comparison currently uses spherical open geometry')
    cache, cache_sha = load_cache(args.cache)
    require(cache['native_bonds']['available'], 'No cached native labels')
    require(cache['source_sha256']['config'] == digest(config_raw), 'Cached config changed')
    require(cache['source_sha256']['shape'] == digest(shape_raw), 'Cached shape changed')
    for path, expected in cache['native_bonds']['source_sha256'].items():
        require(sha(path) == expected, 'Classifier input changed: '+path)
    with (run/'trajectory.jsonl').open('rb') as handle:
        stat = os.fstat(handle.fileno())
        raw = handle.read(stat.st_size)
    raw = raw[:raw.rfind(b'\n')+1]
    lines = raw.splitlines(keepends=True)
    records = [json.loads(line) for line in lines]
    require(records and records[0]['sweep'] == 0, 'Need the original seed frame')
    require(all(a['sweep'] < b['sweep'] for a,b in zip(records,records[1:])), 'Frames must be ordered')
    require(len(cache['frames']) <= len(records), 'Cache extends past current trajectory')
    running_hash = hashlib.sha256(); hash_prefix = None
    for count, line in enumerate(lines, 1):
        running_hash.update(line)
        if running_hash.hexdigest() == cache['source_sha256']['trajectory']:
            hash_prefix = count
    require(hash_prefix is not None, 'Cached trajectory source is not an authenticated prefix')
    for saved, observed in zip(cache['frames'], records):
        require(saved['sweep'] == observed['sweep'] and saved['poses'] == observed['poses']
                and saved['seed_labels'] == observed['seed_labels'], 'Cached physical frame changed')
    seeds = set(config['seed_labels']); n = len(records[0]['poses'])
    require(seeds and all(set(f['seed_labels']) == seeds and len(f['poses']) == n for f in records), 'Seed/body inventory changed')
    sys.path.insert(0, str(reference/'scripts'))
    from tetramer_order import TetramerOrder
    from analyze_precursor_exchange import prepare_templates
    analysis_config = dict(config, rigid_members=shape['rigid_members'])
    analysis_config['box_lengths'] = [8*(config['boundary']['radius']+cache['body_bound'])]*3
    motif_path = Path(config.get('native_pair_motifs', config['metadata']['native_pair_motifs']))
    if not motif_path.is_absolute(): motif_path = run/motif_path
    analysis_config['native_pair_motifs'] = str(motif_path)
    templates = prepare_templates(Path(config['monomer_shape']), reference/'results/c1c3-scaffold/motifs.json', reference/'results/native-neighbor-classes/classification.json')
    classifier = TetramerOrder(analysis_config, run, templates=templates)
    families = inverse_families(classifier.body_motifs)
    class_families = {r['label']:r['family'] for r in classifier.references}
    cutoff = 2*float(np.max(np.linalg.norm(classifier.member_positions,axis=1)))+classifier.cutoff
    cutoff = np.nextafter(cutoff, math.inf)
    args.out.mkdir(parents=True)
    (args.out/'trajectory-prefix.jsonl').write_bytes(raw)
    frozen = dict(schema='seed-growth-native-bonds-v1', run=str(run),
        source_trajectory_sha256=digest(raw), source_snapshot_bytes=len(raw),
        source_file_bytes_at_snapshot=stat.st_size, cached_frames=len(cache['frames']),
        cache_sha256=cache_sha, cache_source_trajectory_matches_prefix_frames=hash_prefix,
        cache_provenance_note='The viewer hashes its live source after classification; its authenticated hash may include more frames than its cached labels. Every cached pose is also compared exactly.',
        config_sha256=digest(config_raw), shape_sha256=digest(shape_raw),
        classifier_sources=cache['native_bonds']['source_sha256'],
        script_sha256=sha(__file__), frames=len(records), first_sweep=records[0]['sweep'],
        last_sweep=records[-1]['sweep'], bodies=n, seed_labels=sorted(seeds),
        depletant_radius=config['depletant_radius'], activity=config['reservoir_density'],
        concentration_uM=n/(4*math.pi*config['boundary']['radius']**3/3)/(6.02214076e-10),
        monomer_protocol=classifier.protocol, new_frame_body_pair_cutoff_A=cutoff,
        scope='Every saved frame in one frozen trajectory prefix, no thinning. Unique intertetramer member bonds and registered body-pair motif families; intrinsic contacts excluded. Newcomer means original nonseed label. Samples can miss short-lived contacts between frames. No physical rates, independent-event counts, equilibrium or nucleation claim.',
        registered_reuse='Reuse strict-entry monomer witnesses in unchanged registered_body_graphs; omit stay-only edges, which cannot change entry matches. Do not report stay/support counts from this reuse.',
        reciprocal_motif_families=families, completed=False)
    (args.out/'manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
    classified = []
    with (args.out/'bonds.jsonl').open('w') as output:
        for index, frame in enumerate(records):
            if index < len(cache['frames']):
                edges = [dict(bodies=b['bodies'], members=b['members'], class_label=label,
                              class_family=class_families[label],entry=True)
                         for b in cache['frames'][index]['native_bonds'] for label in b['classes']]
            else:
                positions = np.array([p['position'] for p in frame['poses']])
                pairs = cKDTree(positions).query_pairs(cutoff, output_type='set')
                result = classifier.classify(frame, pair_filter=pairs)
                edges = result['native_entry_monomer_edges']
            # Registered entry has an entry witness: stay-only edges are immaterial.
            registered = dict(native_stay_monomer_edges=edges)
            classifier.registered_body_graphs(frame, registered, sorted(seeds))
            monomer = {(e['class_family'],tuple(e['bodies']),tuple(e['members'])) for e in edges}
            motifs = {(families[m['motif_id']],tuple(m['bodies'])) for m in registered['registered_tetramer_motifs'] if m['entry']}
            row = dict(sweep=frame['sweep'], bodies=n,
                monomer_bonds=[dict(family=c,bodies=list(b),members=list(m))for c,b,m in sorted(monomer)],
                registered_bonds=[dict(family=c,bodies=list(b))for c,b in sorted(motifs)],
                directed_registered_motifs=[m for m in registered['registered_tetramer_motifs'] if m['entry']],
                registered_graph=registered['registered_tetramer_entry'])
            classified.append(row);output.write(json.dumps(row,separators=(',',':'))+'\n')
    frozen.update(completed=True, monomer=summarize(classified,'monomer_bonds',seeds),
        registered=summarize(classified,'registered_bonds',seeds),
        latest_registered_graph=classified[-1]['registered_graph'],
        new_frames_classified=len(records)-len(cache['frames']),
        bonds_sha256=sha(args.out/'bonds.jsonl'))
    (args.out/'summary.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({k:frozen[k] for k in ('completed','frames','last_sweep','new_frames_classified')}))
    for level in ('monomer','registered'):
        print(level)
        for row in frozen[level]['families']:
            print(row['family'],row['initial_seed_bonds'],row['distinct_newcomer_bonds'],row['latest_bonds'])


if __name__ == '__main__':
    main()
