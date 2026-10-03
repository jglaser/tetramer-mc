#!/usr/bin/env python3
"""Count every archived atlas center once on three preselected saved clouds.

Freeze first, then execute the copied method. Reuses completed center geometry
and guided-draw counts: no new points, poses, hard geometry, baths, or proposals.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

import numpy as np
import scipy
from analyze_auxiliary_overlap_probe import CountOracle, Checks, audit_cloud

NAMES = ('blind_memory_allslot64', 'blind_fft512slots', 'native_informed178')
SIZES = (128, 2048, 328)
LIMITATIONS = [
    'One existing root-thinned finite cloud per atlas, selected by case_index=0 and attempt=0 before counts. No new random data.',
    'Counts and count/source ratios are finite-cloud descriptions, not exact overlap volumes or physical weights.',
    'Center rows uniformly enumerate labels; branch-weight sums describe normalized center labels, not Gaussian success mass or equilibrium weights.',
    'Distinct atlas clouds carry sampling variation. Shared-cloud center counts are dependent; reciprocal poses need not have equal finite-cloud counts.',
    'Raw guided counts exist only after internal geometric success. Missing counts remain null, never zero. Saved prefixes have stopping/cap and root-success conditioning.',
    'Selected m4 prefixes contain m1 prefixes. The raw join deduplicates their common draws and retains every rejected, learned, and uniform draw.',
    'Paired learned-draw comparisons condition on geometric success and a stopped prefix; they cannot identify causal covariance directions or estimate full Gaussian overlap mass.',
    'No center redecoding, native filtering, geometry rerun, physical bath, state update, fitting, or production change.',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def local_closure(path):
    pending, found = [Path(path).resolve()], set()
    while pending:
        item = pending.pop()
        if item in found:
            continue
        found.add(item)
        for node in ast.walk(ast.parse(item.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) else
                     [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                candidate = item.parent / ((name or '').split('.')[0] + '.py')
                if candidate.is_file():
                    pending.append(candidate)
    return sorted(found)


def validate_centers(queries, rows, config):
    require(len(queries) == len(rows) == config['queries'] == sum(SIZES), 'Incomplete center allocation')
    require([(a['name'], a['virtual_branches']) for a in config['atlases']] == list(zip(NAMES, SIZES)), 'Changed atlases')
    expected = [(name, branch) for name, n in zip(NAMES, SIZES) for branch in range(n)]
    require([(r['atlas'], r['branch']) for r in rows] == expected, 'Filtered/reordered centers')
    for i, (query, row) in enumerate(zip(queries, rows)):
        require(query['query'] == row['query'] == i and all(row[k] == v for k, v in query.items()), 'Center pose/query changed')
        require(row['status'] == 'complete' and row['latent'] == [0.] * 6, 'Incomplete/noncenter row')
        require(row['core_overlap'] == (row['clearance_A'] < 0), 'Cached core predicate differs')
        require(row['exclusion_contact'] == (row['clearance_A'] < 2 * config['depletant_radius_A']), 'Cached contact predicate differs')
    for name in NAMES:
        require(abs(math.fsum(r['branch_weight'] for r in rows if r['atlas'] == name) - 1) < 1e-12, 'Unnormalized center labels')


def freeze(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    require(not out.exists(), 'Freeze directory already exists; do not replace or extend a run')
    center = root / 'results/dimer-mean-clearance-20261002'
    guide = root / 'results/auxiliary-overlap-probe-20261003'
    cc, gp = read(center / 'config.json'), read(guide / 'protocol.json')
    ref = Path(gp['reference_config']['path'])
    files = {
        'center_config': center / 'config.json', 'center_freeze': center / 'freeze.json',
        'center_queries': center / 'queries.jsonl', 'center_ledger': center / 'execution/queries.jsonl',
        'center_analysis': center / 'execution/analysis.json', 'center_terminal': center / 'execution/terminal.json',
        'center_postprocess': center / 'postprocess.json', 'guide_protocol': guide / 'protocol.json',
        'guide_review': guide / 'completed-review.json', 'guide_analysis': guide / 'analysis.json',
        'guide_terminal': guide / 'execution/terminal.json', 'guide_attempts': guide / 'execution/attempts.jsonl',
        'guide_clouds': guide / 'execution/clouds.jsonl', 'guide_uniforms': guide / 'execution/cloud-uniforms.bin',
        'reference_config': ref,
    }
    files.update({'center_input/' + name: Path(rec['path']) for name, rec in cc['files'].items()})
    selected = [row for row in jsonl(files['guide_clouds']) if row['case_index'] == 0 and row['attempt'] == 0]
    require([(r['atlas_index'], r['cloud_id']) for r in selected] == [(0, 0), (1, 256), (2, 512)], 'Wrong fixed clouds')
    out.mkdir(parents=True)
    copied = out / 'method'
    copied.mkdir()
    methods = {}
    for source in local_closure(__file__):
        dest = copied / source.name
        shutil.copyfile(source, dest)
        methods[source.name] = dict(**binding(dest), source_path=str(source))
    config = dict(schema='saved-dimer-center-overlap-v1', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        files={name: binding(path) for name, path in files.items()}, methods=methods,
        atlas_names=list(NAMES), virtual_branches=list(SIZES), selected_clouds=selected,
        allocation=dict(center_count_queries=2504, root_cloud_reconstructions=3, new_draw_count_queries=0,
            new_hard_geometry_queries=0, new_points=0, new_poses=0, new_baths=0, repeats=0),
        selection='All archived center rows, including every invalid and reciprocal branch; atlas cloud case_index=0, attempt=0.',
        metrics='Unweighted min/p05/median/p95/max; center-label weighted mean; counts and unrenormalized label weights >= 0.5 source and >= source; matched stopped raw-prefix joins.',
        maximum_cpu_seconds=180, maximum_wall_seconds=240, scientific_threads=1,
        depletant_radius_A=1.4, output=str(out / 'execution'), limitations=LIMITATIONS,
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__))
    save(out / 'config.json', config)
    save(out / 'freeze.json', dict(config_sha256=sha(out / 'config.json'), allocation=config['allocation'],
        method_sha256=methods[Path(__file__).name]['sha256'], frozen_before_evaluative_loop=True))
    print(json.dumps(dict(frozen=str(out), config_sha256=sha(out / 'config.json'), allocation=config['allocation'])))


def quantiles(values):
    return dict(zip(('min', 'p05', 'median', 'p95', 'max'), np.quantile(values, [0, .05, .5, .95, 1]).tolist())) if values else None


def center_summary(rows, old):
    weight = math.fsum(r['branch_weight'] for r in rows)
    return dict(branches=len(rows), center_label_weight=weight,
        counts=quantiles([r['guidance_count'] for r in rows]),
        count_retention=quantiles([r['guidance_count'] / old for r in rows]),
        weighted_mean_retention=math.fsum(r['branch_weight'] * r['guidance_count'] / old for r in rows) / weight if weight else None,
        at_least_half_source=dict(branches=sum(2 * r['guidance_count'] >= old for r in rows),
            center_label_weight=math.fsum(r['branch_weight'] for r in rows if 2 * r['guidance_count'] >= old)),
        at_least_source=dict(branches=sum(r['guidance_count'] >= old for r in rows),
            center_label_weight=math.fsum(r['branch_weight'] for r in rows if r['guidance_count'] >= old)))


def stopped_draw_join(arms, centers, old):
    records = {m: arms[m]['outcome']['attempts'][0]['internal_draws'] for m in ('m1', 'm4')}
    project = lambda rs: [{k: v for k, v in r.items() if k != 'guidance_count'} for r in rs]
    require(project(records['m1']) == project(records['m4'][:len(records['m1'])]), 'Guided prefixes differ')
    rows = []
    for i, record in enumerate(records['m4']):
        draw, feasible = record['draw'], record['feasibility']
        geometric = not feasible['internal_core_overlap'] and feasible['internal_exclusion_contact']
        require(('guidance_count' in record) == geometric, 'Censored-count contract differs')
        label = draw.get('trace', {}).get('target_label')
        center = centers[label['branch']] if draw['branch'] == 'learned' else None
        require((center is not None) == (draw['branch'] == 'learned'), 'Wrong center join')
        count = record.get('guidance_count')
        rows.append(dict(raw_index=i + 1, methods=['m1', 'm4'] if i < len(records['m1']) else ['m4'],
            record=record, selected_center_query=center['query'] if center else None,
            selected_center_core_overlap=center['core_overlap'] if center else None,
            selected_center_guidance_count=center['guidance_count'] if center else None,
            draw_guidance_count=count, draw_count_retention=count / old if count is not None else None,
            draw_minus_center_count=count - center['guidance_count'] if center and count is not None else None))
    return rows


def execute(config_path):
    config_path = Path(config_path).resolve()
    config, frozen = read(config_path), read(config_path.parent / 'freeze.json')
    require(sha(config_path) == frozen['config_sha256'], 'Changed frozen config')
    require(config['scientific_threads'] == 1 and config['allocation'] == frozen['allocation'], 'Changed allocation')
    for record in [*config['files'].values(), *config['methods'].values()]:
        require(sha(record['path']) == record['sha256'], 'Changed frozen input: ' + record['path'])
    require(Path(config['methods'][Path(__file__).name]['path']).resolve() == Path(__file__).resolve(), 'Execute the frozen method copy')
    get = lambda name: config['files'][name]['path']
    cc, cf, ca, ct = [read(get('center_' + key)) for key in ('config', 'freeze', 'analysis', 'terminal')]
    require(ct['complete'] and ct['passed'] and ca['complete'] and ca['passed'], 'Incomplete prior centers')
    require(cf['config_sha256'] == sha(get('center_config')) == ca['config_sha256'], 'Center config binding')
    require(ct['ledger_sha256'] == sha(get('center_ledger')) == ca['ledger_sha256'], 'Center ledger binding')
    require(ct['analysis_sha256'] == sha(get('center_analysis')), 'Center analysis binding')
    require(cf['queries_sha256'] == sha(get('center_queries')) == ca['queries_sha256'], 'Center query binding')
    for name, rec in cc['files'].items():
        require(rec['sha256'] == config['files']['center_input/' + name]['sha256'], 'Changed archived center input')
    gp, gr, ga, gt = [read(get('guide_' + key)) for key in ('protocol', 'review', 'analysis', 'terminal')]
    require(gr['passed'] and gr['complete'] and ga['passed'] and ga['complete'] and gt['summary']['complete'], 'Incomplete prior guide')
    for key, name in [('analysis.json', 'analysis'), ('protocol.json', 'protocol'), ('execution/attempts.jsonl', 'attempts'),
                      ('execution/clouds.jsonl', 'clouds'), ('execution/cloud-uniforms.bin', 'uniforms'), ('execution/terminal.json', 'terminal')]:
        require(gr['output_hashes'][key] == sha(get('guide_' + name)), 'Guide review binding')
    ref = read(get('reference_config'))
    require(gp['reference_config']['sha256'] == sha(get('reference_config')), 'Reference binding')
    require(ref['shape']['sha256'] == cc['files']['shape']['sha256'], 'Different shape')
    require([(a['name'], a['model']['sha256']) for a in ref['atlases']] == [(a['name'], a['model_sha256']) for a in cc['atlases']], 'Different atlas models')
    require(ref['depletant_radius'] == cc['depletant_radius_A'] == config['depletant_radius_A'], 'Different inflation')
    queries, centers = jsonl(get('center_queries')), jsonl(get('center_ledger'))
    validate_centers(queries, centers, cc)
    attempts = [r for r in jsonl(get('guide_attempts')) if r['case_index'] == 0 and r['attempt'] == 0]
    require(len(attempts) == 6, 'Incomplete selected guided arms')
    out = Path(config['output'])
    out.mkdir()
    resource.setrlimit(resource.RLIMIT_CPU, (config['maximum_cpu_seconds'], config['maximum_cpu_seconds']))
    def timeout(signum, frame):
        raise TimeoutError('Frozen wall-time allocation exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(config['maximum_wall_seconds'])
    cpu, wall = time.process_time(), time.monotonic()
    save(out / 'started.json', dict(config_sha256=sha(config_path), pid=os.getpid(),
        recorded_utc=datetime.now(timezone.utc).isoformat(), scientific_threads=1))
    shape = read(get('center_input/shape'))
    oracle = CountOracle([a['center'] for a in shape['atoms']], [a['radius'] + config['depletant_radius_A'] for a in shape['atoms']])
    summary, drawrows, countrows, checks = [], [], [], Checks()
    with Path(get('guide_uniforms')).open('rb') as binary, (out / 'centers.jsonl').open('x') as ledger:
        for ai, (name, meta) in enumerate(zip(NAMES, config['selected_clouds'])):
            require(meta['atlas_index'] == ai and meta['case_index'] == meta['attempt'] == 0, 'Wrong frozen cloud')
            binary.seek(meta['raw_byte_offset'])
            points = audit_cloud(meta, binary.read(meta['raw_byte_length']), oracle, checks, name)
            require(not checks.failures, 'Cloud reconstruction failed')
            arms = {r['method']: r for r in attempts if r['atlas'] == name}
            require(set(arms) == {'m1', 'm4'} and all(r['cloud_id'] == meta['cloud_id'] for r in arms.values()), 'Wrong selected arms')
            old = arms['m1']['outcome']['guidance']['old_count']
            require(old > 0 and old == arms['m4']['outcome']['guidance']['old_count'], 'Cached source counts differ')
            for m, arm in arms.items():
                cached = [r for r in ga['rows'] if r['atlas'] == name and r['case'] == arm['case']['name'] and r['attempt'] == 0 and r['method'] == m]
                require(len(cached) == 1 and cached[0]['old_count'] == old, 'Analysis source count differs')
            atlasrows = []
            for center in (r for r in centers if r['atlas'] == name):
                count = oracle.relative(points, center['relative_pose'])
                row = dict(**center, cloud_id=meta['cloud_id'], source_count=old, guidance_count=count, count_retention=count / old)
                ledger.write(json.dumps(row, allow_nan=False) + '\n')
                atlasrows.append(row)
            ledger.flush()
            countrows.extend(atlasrows)
            raw = stopped_draw_join(arms, atlasrows, old)
            drawrows.extend(dict(atlas=name, cloud_id=meta['cloud_id'], **r) for r in raw)
            paired = [r for r in raw if r['draw_minus_center_count'] is not None]
            groups = {key: [r for r in atlasrows if predicate(r)] for key, predicate in {
                'all': lambda r: True, 'core_valid': lambda r: not r['core_overlap'],
                'core_valid_contact': lambda r: not r['core_overlap'] and r['exclusion_contact'],
                'core_collision': lambda r: r['core_overlap']}.items()}
            summary.append(dict(atlas=name, cloud_id=meta['cloud_id'], retained_points=len(points), source_count=old,
                centers={key: center_summary(rows, old) for key, rows in groups.items()},
                selected_arms={m: dict(status=a['outcome']['status'], stage_status=a['outcome']['attempts'][0]['status'],
                    internal_draws=len(a['outcome']['attempts'][0]['internal_draws']), guidance=a['outcome']['guidance']) for m, a in arms.items()},
                unique_saved_raw_prefix=dict(draws=len(raw), learned=sum(r['record']['draw']['branch'] == 'learned' for r in raw),
                    core_collisions=sum(r['record']['feasibility']['internal_core_overlap'] for r in raw),
                    exclusion_noncontacts=sum(not r['record']['feasibility']['internal_exclusion_contact'] for r in raw),
                    cached_count_draws=sum(r['draw_guidance_count'] is not None for r in raw),
                    cached_count_retention=quantiles([r['draw_count_retention'] for r in raw if r['draw_guidance_count'] is not None]),
                    paired_learned_geometrically_valid_draws=len(paired),
                    paired_center_counts=quantiles([r['selected_center_guidance_count'] for r in paired]),
                    paired_draw_counts=quantiles([r['draw_guidance_count'] for r in paired]),
                    paired_draw_minus_center=quantiles([r['draw_minus_center_count'] for r in paired]),
                    lower_equal_higher=dict(Counter('lower' if r['draw_minus_center_count'] < 0 else 'higher' if r['draw_minus_center_count'] > 0 else 'equal' for r in paired)))))
            print(json.dumps(dict(atlas=name, completed_centers=len(atlasrows), source_count=old, core_valid=summary[-1]['centers']['core_valid'])), flush=True)
    require(len(countrows) == config['allocation']['center_count_queries'], 'Incomplete count loop')
    with (out / 'saved-raw-draw-join.jsonl').open('x') as stream:
        for row in drawrows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    analysis = dict(schema=config['schema'], complete=True, passed=True, config_sha256=sha(config_path),
        allocation=config['allocation'], summary=summary, cached_probe_comparisons=ga['comparisons'],
        cloud_checks=checks.count, failures=checks.failures, limitations=LIMITATIONS,
        cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - wall)
    save(out / 'analysis.json', analysis)
    save(out / 'terminal.json', dict(complete=True, passed=True, center_count_queries=len(countrows),
        root_cloud_reconstructions=3, new_hard_geometry_queries=0, new_points=0, new_poses=0, new_baths=0,
        analysis_sha256=sha(out / 'analysis.json'), centers_sha256=sha(out / 'centers.jsonl'),
        raw_draw_join_sha256=sha(out / 'saved-raw-draw-join.jsonl'), cpu_seconds=analysis['cpu_seconds'], wall_seconds=analysis['wall_seconds']))
    signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', type=Path)
    mode.add_argument('--execute', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.freeze:
        freeze(args.root, args.freeze)
    else:
        execute(args.execute)
