#!/usr/bin/env python3
"""Prepare/run a bounded, native-blind exact-repair audit of cached FFT peaks.

Prepare freezes uniform reference slots and source hashes without running any
repair. Run preserves all slots, including failed repairs, in its denominator.
This tests raw-peak repair and held-out overlap, not equilibrium basin recovery.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import resource
import subprocess
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def uniform_slots(total, count):
    if not 1 <= count <= total:
        raise ValueError('slot count must be between one and the total')
    return [i*total//count for i in range(count)]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def matching_pose(a, b):
    return (np.allclose(a['position'], b['position'], atol=1e-10, rtol=0)
            and abs(float(np.dot(a['orientation'], b['orientation']))) >= 1.-1e-10)


def prepare(a):
    if a.steps < 0 or a.points <= 0 or not np.isfinite(a.rd) or a.rd < 0:
        raise ValueError('invalid repair budget or bath radius')
    export = json.loads(a.export.read_text())
    metrics = json.loads(a.fit_metrics.read_text())
    if export.get('native_selection') is not False:
        raise ValueError('export must be native-blind')
    references = json.loads(a.references.read_text())
    if sha(a.references) != export['reference_sha256']:
        raise ValueError('reference hash mismatch')
    if len(references) != len(export['slots']) or len(metrics['slots']) != len(references):
        raise ValueError('slot count mismatch')
    selected = uniform_slots(len(references), a.slots)
    a.out.mkdir(parents=True, exist_ok=False)
    inputs = a.out/'inputs'; inputs.mkdir()
    snapshots = dict(shape=a.shape, repair=ROOT/'tools/select_fft_contact_starts.py',
                     scorer=a.scorer, controller=Path(__file__), export=a.export,
                     fit_metrics=a.fit_metrics, references=a.references)
    frozen = {}
    for key, src in snapshots.items():
        dst = inputs/(key+('.json' if key in ('shape', 'export', 'fit_metrics', 'references')
                           else '.py' if key in ('repair', 'controller') else ''))
        shutil.copy2(src, dst)
        frozen[key] = dict(path=str(dst.resolve()), sha256=sha(dst), original=str(src.resolve()))
    rows = []
    for i, slot in enumerate(selected):
        f = metrics['slots'][slot]
        if f['global_slot'] != slot:
            raise ValueError('fit metric slot order mismatch')
        source = metrics['sources'][f['source_index']]
        path = Path(source['directory'])/'discovery.json'
        if sha(path) != source['discovery_sha256']:
            raise ValueError('source discovery hash mismatch: '+str(path))
        d = json.loads(path.read_text())
        if d['config']['rd'] != a.rd:
            raise ValueError('depletant radius mismatch with archived discovery')
        original = d['slots'][f['local_slot']]
        if not matching_pose(original['initial_pose'], references[slot]):
            raise ValueError('source initial pose differs from audited reference')
        if d['shape_sha256'] != frozen['shape']['sha256']:
            raise ValueError('shape mismatch with archived discovery')
        valid = original['optimized_validation']
        if len(valid) != 2 or valid[0]['seed'] == valid[1]['seed']:
            raise ValueError('expected two independent archived validation clouds')
        rows.append(dict(slot=slot, raw=export['slots'][slot], reference_pose=references[slot],
                         archived_optimized_pose=original['optimized_pose'],
                         archived_validation=valid, source=dict(path=str(path), sha256=source['discovery_sha256'],
                           source_index=f['source_index'], local_slot=f['local_slot']),
                         repair_seed=a.seed+1009*i, score_seed=a.seed+1000003+1009*i))
    plan = dict(schema='fft-subset-exact-recovery-v1', complete=False, prepared=True,
                selection_rule='floor(i*total/count), i=0..count-1; fixed before labels',
                selection_indices=selected, total_reference_slots=len(references), native_information=False,
                all_slot_denominator=len(rows), budget=export['budget'],
                rd=a.rd, repair_steps=a.steps, clearance=a.clearance, score_points=a.points,
                frozen=frozen, slots=rows, worker_limit=4,
                scope='Exact existing geometric repair then fresh union-overlap validation. No local optimization, dynamics, FFT or basin-mass inference.')
    write(a.out/'plan.json', plan)
    print(json.dumps(dict(prepared=True, slots=len(rows), indices=selected,
                         command=f'{os.sys.executable} {inputs / "controller.py"} run --prepared {a.out.resolve()} --workers 4')))


def worker(job):
    root, row, plan = job
    root = Path(root)
    out = root/'slots'/f"s{row['slot']:04d}"
    out.mkdir()
    start_cpu, start_wall = time.process_time(), time.monotonic()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_start_cpu = child_start.ru_utime+child_start.ru_stime
    base = dict(slot=row['slot'], raw_present=row['raw']['present'], repaired_pose=None,
                repair_succeeded=False, score=None, execution_error=None)
    try:
        if row['raw']['present']:
            spec = importlib.util.spec_from_file_location('frozen_fft_repair', plan['frozen']['repair']['path'])
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            module._init(plan['frozen']['shape']['path'])
            p = row['raw']['pose']
            repaired = module.repair((p['position'], p['orientation'], row['repair_seed'],
                                      plan['repair_steps'], plan['clearance']))
            base['repaired_pose'] = repaired
            base['repair_succeeded'] = repaired is not None
            if repaired is not None:
                pose_file = out/'pose.json'; write(pose_file, [repaired])
                command = [plan['frozen']['scorer']['path'], '--shape', plan['frozen']['shape']['path'],
                           '--poses', str(pose_file), '--rd', str(plan['rd']),
                           '--points', str(plan['score_points']), '--seed', str(row['score_seed'])]
                base['score_command'] = command
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                (out/'score.stdout.json').write_text(result.stdout)
                (out/'score.stderr.txt').write_text(result.stderr)
                scored = json.loads(result.stdout)
                if len(scored) != 1:
                    raise ValueError('invalid score output')
                base['score'] = scored[0]
                base['repair_succeeded'] = bool(scored[0]['hard_valid'])
    except Exception:
        base['execution_error'] = traceback.format_exc()
    base['worker_cpu_seconds'] = time.process_time()-start_cpu
    child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    base['scorer_child_cpu_seconds'] = child_end.ru_utime+child_end.ru_stime-child_start_cpu
    base['wall_seconds'] = time.monotonic()-start_wall
    write(out/'result.json', base)
    return base


def summarize(rows, plan):
    ref_by_slot = {r['slot']:r for r in plan['slots']}
    detailed = []
    for result in sorted(rows, key=lambda x:x['slot']):
        row = ref_by_slot[result['slot']]
        v = row['archived_validation']
        mean = float(np.mean([x['volume'] for x in v]))
        se = float(np.sqrt(sum(x['standard_error']**2 for x in v))/len(v))
        score = result['score']
        recovered = (result['execution_error'] is None and result['repair_succeeded']
                     and score is not None and score['hard_valid'])
        volume = float(score['score']['volume']) if recovered else 0.
        detailed.append(dict(**result, archived_overlap_A3=mean, archived_overlap_se_A3=se,
                             recovered_overlap_A3=volume, overlap_ratio_zero_on_failure=volume/mean if mean > 0 else None,
                             overlap_at_least_90pct_archived=bool(recovered and volume >= .9*mean)))
    n = plan['all_slot_denominator']
    return dict(complete=len(rows)==n and all(r['execution_error'] is None for r in rows),
                attempted_slots=len(rows), denominator=n,
                repaired_hard_valid=sum(r['repair_succeeded'] for r in rows),
                overlap_at_least_90pct_archived=sum(r['overlap_at_least_90pct_archived'] for r in detailed),
                mean_overlap_ratio_unconditional=float(np.mean([r['overlap_ratio_zero_on_failure'] for r in detailed
                                                               if r['overlap_ratio_zero_on_failure'] is not None])),
                python_worker_cpu_seconds=sum(r['worker_cpu_seconds'] for r in rows),
                scorer_child_cpu_seconds=sum(r['scorer_child_cpu_seconds'] for r in rows),
                total_cpu_seconds=sum(r['worker_cpu_seconds']+r['scorer_child_cpu_seconds'] for r in rows),
                caveat='Failed repairs remain zero in unconditional ratios; execution errors invalidate completion. Pair overlap is not basin mass.',
                native_information=False, slots=detailed)


def run(a):
    root = a.prepared.resolve()
    plan = json.loads((root/'plan.json').read_text())
    if not 1 <= a.workers <= plan['worker_limit']:
        raise ValueError('worker count exceeds fixed limit')
    for item in plan['frozen'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('frozen input changed: '+item['path'])
    if sha(__file__) != plan['frozen']['controller']['sha256']:
        raise ValueError('run the archived controller or exact same version')
    (root/'slots').mkdir(exist_ok=False)
    started = time.monotonic()
    results = []
    # Every allocated slot is submitted once. A worker records an individual
    # error and the executor drains all submitted work rather than retrying.
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(worker, (str(root), row, plan)) for row in plan['slots']]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write(root/'progress.json', dict(completed=len(results), allocated=len(plan['slots'])))
    summary = summarize(results, plan)
    summary['wall_seconds'] = time.monotonic()-started
    write(root/'report.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='slots'}, indent=2))
    if not summary['complete']:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--export', type=Path, required=True)
    q.add_argument('--fit-metrics', type=Path, required=True)
    q.add_argument('--references', type=Path, required=True)
    q.add_argument('--shape', type=Path, required=True)
    q.add_argument('--scorer', type=Path, default=ROOT/'target/release/pair-overlap-score')
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--slots', type=int, default=32)
    q.add_argument('--steps', type=int, default=400)
    q.add_argument('--points', type=int, default=8192)
    q.add_argument('--rd', type=float, default=1.4)
    q.add_argument('--clearance', type=float, default=.02)
    q.add_argument('--seed', type=int, default=202609289001)
    q.set_defaults(func=prepare)
    q = sub.add_parser('run')
    q.add_argument('--prepared', type=Path, required=True)
    q.add_argument('--workers', type=int, default=4)
    q.set_defaults(func=run)
    a = p.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
