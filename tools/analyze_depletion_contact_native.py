#!/usr/bin/env python3
"""Post-run complete native-entry and registered-tetramer frame diagnostics.

Native data enter only this observer. No discovery, fit, move choice, acceptance,
or winner selection is performed. Every completed benchmark arm is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from export_viewer import native_bonds, poses


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def graph_counts(graph, seeds):
    seeds = set(seeds)
    edges = graph['edges']
    return dict(
        bonds=graph['bonds'],
        seed_seed_bonds=sum(i in seeds and j in seeds for i, j in edges),
        seed_nonseed_bonds=sum((i in seeds) != (j in seeds) for i, j in edges),
        nonseed_nonseed_bonds=sum(i not in seeds and j not in seeds for i, j in edges),
        seed_component=graph['seed_component'],
        seed_component_size=graph['seed_component_size'],
        retained_original_seed_bodies=graph['retained_original_seed_bodies'],
        seed_component_newcomers=graph['seed_component_newcomers'],
        seed_component_newcomer_ids=graph['seed_component_newcomer_ids'],
        largest_component_size=graph['largest_component_size'],
    )


def analyze(benchmark, out, reference):
    benchmark, out, reference = (Path(p).resolve() for p in (benchmark, out, reference))
    if out.exists():
        raise ValueError('Use a fresh native-diagnostic output directory')
    plan, status = read(benchmark/'plan.json'), read(benchmark/'status.json')
    if not status['complete'] or any(job['status'] != 'complete' for job in status['jobs']):
        raise ValueError('All benchmark populations must complete before native evaluation')
    sys.path.insert(0, str(reference/'scripts'))
    from tetramer_order import TetramerOrder
    original_classify = TetramerOrder.classify
    records = []

    # Keep export_viewer's exact geometry/observer setup while retaining the
    # complete classifier output it normally reduces to display monomer bonds.
    def recorded_classify(observer, frame, pair_filter=None):
        if pair_filter is not None:
            raise ValueError('Complete pair graphs are required')
        result = original_classify(observer, frame)
        if not result['complete_pair_graph']:
            raise ValueError('Incomplete native graph')
        records.append(result)
        return result

    out.mkdir(parents=True)
    summaries, sources, protocols = [], {}, {}
    try:
        TetramerOrder.classify = recorded_classify
        for job in plan['jobs']:
            run = Path(job['directory'])
            summary = read(run/'summary.json')
            if not summary['complete'] or summary['completed_sweeps'] != plan['sweeps']:
                raise ValueError('Incomplete run: '+job['id'])
            config = read(run/'config.json')
            shape_path = run/'provenance/shape.json'
            if not shape_path.is_file():
                shape_path = Path(config['shape'])
                if not shape_path.is_absolute():
                    shape_path = run/shape_path
            shape = read(shape_path)
            frames = []
            for line in (run/'trajectory.jsonl').read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                frames.append(dict(sweep=row['sweep'], poses=poses(row),
                    seed_labels=row.get('seed_labels', config.get('seed_labels', []))))
            expected = list(range(0, plan['sweeps']+1, plan.get('sample_every', 100)))
            # The requested pilot has exactly the saved frames 0, 100, 200.
            if [frame['sweep'] for frame in frames] != expected:
                raise ValueError('Unexpected saved frame allocation: '+job['id'])
            boundary = config['boundary']
            if boundary['kind'] != 'spherical':
                raise ValueError('This diagnostic expects the spherical comparison')
            records.clear()
            observer = native_bonds(config, shape, frames, run, 'spherical',
                                    boundary['radius'], reference, 'on')
            if not observer['available'] or len(records) != len(frames):
                raise ValueError('Complete observer did not classify every saved frame')
            local = []
            for frame, result in zip(frames, records):
                if 'registered_tetramer_entry' not in result:
                    raise ValueError('Certified tetramer motifs required for registry graph')
                if result['classified_body_pairs'] != len(frame['poses'])*(len(frame['poses'])-1)//2:
                    raise ValueError('Observer skipped body pairs')
                local.append(dict(sweep=frame['sweep'], bodies=len(frame['poses']),
                    seed_labels=frame['seed_labels'],
                    native_entry=graph_counts(result['native_entry'], frame['seed_labels']),
                    registered_tetramer_entry=graph_counts(result['registered_tetramer_entry'], frame['seed_labels']),
                    native_entry_monomer_edges=len(result['native_entry_monomer_edges']),
                    complete_pair_graph=True, classified_body_pairs=result['classified_body_pairs']))
            files = {'config':run/'config.json', 'trajectory':run/'trajectory.jsonl',
                     'summary':run/'summary.json', 'shape':shape_path}
            input_hashes = {name:dict(path=str(path), sha256=sha(path)) for name, path in files.items()}
            value = dict(id=job['id'], arm=job['arm'], stream=job['stream'], observer=observer,
                         source_sha256=input_hashes, frames=local, complete_frame_classifications=records.copy())
            write_new(out/(job['id']+'.json'), value)
            summaries.append({key:value[key] for key in ['id', 'arm', 'stream', 'frames']})
            sources[job['id']] = input_hashes
            protocols[job['id']] = observer
    finally:
        TetramerOrder.classify = original_classify
    report = dict(schema='native-blind-contact-proposal-posthoc-native-diagnostics-v1', complete=True,
        benchmark=str(benchmark), benchmark_plan_sha256=sha(benchmark/'plan.json'),
        benchmark_status_sha256=sha(benchmark/'status.json'),
        observer_script_sha256=sha(Path(__file__)),
        viewer_observer_setup_sha256=sha(Path(__file__).with_name('export_viewer.py')),
        populations=summaries, source_sha256=sources, observer_protocols=protocols,
        outputs_sha256={p.name:sha(p) for p in sorted(out.glob('*.json'))},
        seed_component_definition='Component with the most original seed bodies; ties use total size then deterministic IDs, following TetramerOrder.graph_summary.',
        scope='Every saved frame of every completed arm; complete external body-pair graph; strict native entry and full certified tetramer registry entry. No retention hysteresis, inferred transitions, or selection feedback.',
        limitation='Native-like association graphs do not establish lattice perfection, thermodynamic stability, equilibrium, crystal growth, or physical kinetics. The supplied seed is evaluation context.')
    write_new(out/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path,
                        default=Path(__file__).resolve().parents[2]/'protein-nucleation')
    args = parser.parse_args()
    report = analyze(args.benchmark, args.out, args.reference)
    print(json.dumps(dict(complete=report['complete'], populations=report['populations']), indent=2))


if __name__ == '__main__':
    main()
