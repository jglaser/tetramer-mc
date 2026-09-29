#!/usr/bin/env python3
"""Post-hoc move and complete-registry audit of a matched atlas pilot.

Every population and saved frame is retained. Native labels enter only this
evaluation. Chart-center labels are not labels of individual proposed poses.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from finite_assembly_observer import graph_summary
from native_contact_regions import NativeContactRegions, CRITERIA, pose_arrays
from prepare_shoulder_docking_benchmark import local_dependencies


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def inverse_pose(pose):
    t, r = pose_arrays(pose)
    q = Rotation.from_matrix(r.T).as_quat()
    return dict(position=(-r.T@t).tolist(), orientation=q[[3, 0, 1, 2]].tolist())


def native_frame(frame, classifier, seeds):
    poses = frame['poses']
    positions = np.array([p['position'] for p in poses])
    # Complete entry bounds every member displacement. Averaging these errors
    # gives |t| <= max|T_motif| + 2|mean(member centers)| + entry_tolerance.
    # This broad phase cannot remove a complete-entry pair (roundoff guard).
    cutoff = (np.linalg.norm(classifier.motif_positions, axis=1).max()
              + 2*np.linalg.norm(classifier.member_positions.mean(axis=0))
              + CRITERIA['body_member_position_entry_A'] + 1e-7)
    keys = [(int(i), int(j), int(m['motif_id']))
            for i, j in sorted(cKDTree(positions).query_pairs(cutoff))
            for m in classifier.classify_pair(poses[i], poses[j])]
    graph = graph_summary(len(poses), [], keys, classifier.motifs, positions)
    seed_components = [c for c in graph['native_components'] if seeds.intersection(c['bodies'])]
    edges = graph['native_edges']
    return dict(sweep=frame['sweep'], sampler_cpu_seconds=frame['sampler_cpu_seconds'],
        registered_edges=len(edges), native_keys=keys,
        registered_edges_outside_original_seed=sum(not (i in seeds and j in seeds) for i, j in edges),
        registered_seed_free_edges=sum((i in seeds) != (j in seeds) for i, j in edges),
        largest_certified=graph['largest_native_certified'],
        native_cycle_frustrated=graph['native_cycle_frustrated'],
        original_seed_components=[dict(bodies=c['bodies'], certified=c['certified']) for c in seed_components],
        largest_seed_component=max(map(lambda c:len(c['bodies']), seed_components), default=0),
        largest_certified_seed_component=max([0]+[len(c['bodies']) for c in seed_components if c['certified']]),
        nonseed_registered_component_sizes=sorted([len(c['bodies']) for c in graph['native_components']
            if len(c['bodies']) > 1 and not seeds.intersection(c['bodies'])], reverse=True))


def analyze(campaign, definition, out):
    campaign, out = Path(campaign).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Fresh audit output required')
    plan, status = read(campaign/'plan.json'), read(campaign/'status.json')
    if not status['complete'] or any(j['status'] != 'complete' for j in status['jobs']):
        raise ValueError('Require every completed population')
    for name, digest in plan['input_sha256'].items():
        if sha(campaign/name) != digest:
            raise ValueError('Frozen benchmark input changed: '+name)
    classifier = NativeContactRegions(definition)
    continuation = campaign.parent
    slot_map = read(continuation/'merged-atlas/slot-map.json')
    component_maps = {arm:{index:row['original_slot'] for row in slot_map
                          for index in row[key]} for arm, key in
                      [('initial', 'old_components'), ('discovered', 'new_components')]}
    metrics_path = continuation/'inputs/base-fit-receipt/fit-metrics.json'
    metrics = read(metrics_path)
    identity = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
    groups = {row['global_slot']:sorted({m['motif_id'] for pose in
              (row['reference_pose'], inverse_pose(row['reference_pose']))
              for m in classifier.classify_pair(identity, pose)})
              for row in metrics['slots']}
    groups = {slot:labels for slot, labels in groups.items() if labels}
    updated = {row['original_slot'] for row in slot_map if row['updated']}
    sources = {str(p):sha(p) for p in [campaign/'plan.json', campaign/'status.json',
        continuation/'merged-atlas/slot-map.json', metrics_path, Path(definition)]}
    results = []
    for job in plan['jobs']:
        root = Path(job['directory'])
        cfg, summary = read(root/'config.json'), read(root/'summary.json')
        if summary['shape_sha256'] != classifier.shape_sha256 or cfg['boundary']['kind'] != 'spherical':
            raise ValueError('Require matching hard shape and open spherical geometry')
        if not summary['complete'] or summary['completed_sweeps'] != plan['sweeps']:
            raise ValueError('Incomplete pilot')
        seeds = set(cfg['seed_labels'])
        counts = collections.defaultdict(collections.Counter)
        def record(label, move):
            counts[label].update(attempted=1, hard_valid=int(move.get('hard_valid', False)),
                                 accepted=int(move['accepted']))
        digest = hashlib.sha256()
        with (root/'moves.jsonl').open('rb') as stream:
            for raw in stream:
                digest.update(raw)
                m = json.loads(raw)
                q = m.get('proposal') or {}
                if m['kind'] == 'local':
                    record('local', m)
                elif m['kind'] == 'cluster_event':
                    record('cluster_'+str(q.get('branch')), m)
                elif m['kind'] == 'global':
                    record('all_global', m)
                    if q.get('kernel') == 'full-mixture-capture' and q.get('branch') == 'learned':
                        slot = component_maps[job['arm']][q['component_index']]
                        labels = ['learned_redraw', 'updated_slot_redraw' if slot in updated else 'unchanged_slot_redraw']
                        if slot in groups:
                            group = 'native_center_'+','.join(map(str, groups[slot]))
                            labels += ['native_center_redraw', group]
                            if q['anchor_index'] in seeds and m['moving_index'] not in seeds:
                                labels += ['original_seed_anchored_native_center_redraw', 'seed_anchored_'+group]
                        for label in labels:
                            record(label, m)
                    elif q.get('kernel') == 'frozen-posterior' and q.get('branch') == 'involution':
                        record('correlated_transport', m)
                    elif q.get('branch') == 'uniform':
                        record('uniform_relocation', m)
                    else:
                        raise ValueError('Unclassified global proposal')
        sources[str(root/'moves.jsonl')] = digest.hexdigest()
        for key in ('local', 'all_global'):
            expected = summary['counts']['local' if key == 'local' else 'global']
            for field in ('attempted', 'hard_valid', 'accepted'):
                if counts[key][field] != expected[field]:
                    raise ValueError('Move/summary count mismatch')
        frames = [json.loads(line) for line in (root/'trajectory.jsonl').read_text().splitlines()]
        if [f['sweep'] for f in frames] != [0, 100, 200]:
            raise ValueError('Unexpected fixed pilot frame cadence')
        observations = [native_frame(frame, classifier, seeds) for frame in frames]
        results.append(dict(id=job['id'], arm=job['arm'], seed=job['seed'], counts=dict(counts),
            sampler_cpu_seconds=summary['sampler_cpu_seconds'], wall_seconds=summary['wall_seconds'],
            learned_redraw_accepts_per_cpu_second=counts['learned_redraw']['accepted']/summary['sampler_cpu_seconds'],
            cluster_phase=summary['counts']['cluster_phase'], frames=observations))
        for name in ('config.json', 'summary.json', 'trajectory.jsonl', 'manifest.json'):
            sources[str(root/name)] = sha(root/name)
    aggregated = {}
    for arm in sorted({r['arm'] for r in results}):
        populations = [r for r in results if r['arm'] == arm]
        counts = collections.defaultdict(collections.Counter)
        for pop in populations:
            for label, row in pop['counts'].items():
                counts[label].update(row)
        cpu = sum(p['sampler_cpu_seconds'] for p in populations)
        aggregated[arm] = dict(populations=len(populations), sampler_cpu_seconds=cpu, counts=dict(counts),
            learned_redraw_accepts_per_cpu_second=counts['learned_redraw']['accepted']/cpu,
            final_registered_edges=[p['frames'][-1]['registered_edges'] for p in populations],
            final_largest_certified=[p['frames'][-1]['largest_certified'] for p in populations],
            final_largest_certified_seed_component=[p['frames'][-1]['largest_certified_seed_component'] for p in populations])
    report = dict(complete=True, arms=aggregated, populations=results,
        native_center_labels_posthoc=groups, source_sha256=sources,
        physical=dict(radius_A=cfg['depletant_radius'], activity_A_minus3=cfg['reservoir_density'],
                      bodies=len(cfg['initial_poses']), seed_bodies=len(seeds)),
        definitions=dict(native='Complete instantaneous member/angle/residue-patch entry, followed by catalogue cycle consistency.',
            chart_center='Union of posthoc labels of the original reference center and its physical inverse; '
                         'does not classify either the fitted mean or the individual proposal.',
            cpu='Whole production sampler CPU; excludes offline training and posthoc analysis.'),
        scope='Two 200-sweep streams per arm from identical supplied starts. Three saved frames cannot estimate '
              'contact ESS, independent exchanges or equilibrium weights. Accepted captures per CPU are a proposal '
              'accessibility diagnostic, not a mixing speedup. No native labels entered fitting or reweighting.')
    out.mkdir(parents=True)
    (out/'source').mkdir()
    for name, path in local_dependencies([Path(__file__)]).items():
        shutil.copy2(path, out/'source'/name)
    (out/'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(complete=True, files={str(p.relative_to(out)):sha(p)
        for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    return aggregated


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--native-definition', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.campaign, args.native_definition, args.out), indent=2))
