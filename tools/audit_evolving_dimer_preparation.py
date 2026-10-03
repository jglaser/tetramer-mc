#!/usr/bin/env python3
"""Freeze, then independently audit saved clouds and geometric dimer starts.

This tool never draws poses, points, depletants or native labels. The separate
freeze operation authenticates the complete input and Python source closure
before any protein geometry is evaluated. The audit uses NumPy/SciPy sphere
predicates and the independently implemented complete reciprocal-mixture density.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import scipy

from analyze_auxiliary_overlap_probe import CountOracle, point_hash
from analyze_capped_dimer_probe import (
    Checks, DimerDestinationDensity, DimerGeometry, feasibility, is_feasible,
    read, require, serial, sha,
)
from analyze_factorized_dimer_probe import audit_factorized, strict_equal

SCHEMA = 'evolving-dimer-preparation-independent-audit-v1'
MASTER = 6100300601
PAIRS = [[27, 132], [32, 110], [9, 24], [11, 246]]
INITIALIZATIONS = ['source', 'proposal_prepared']
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
FRAME_SHA = 'e69e6c4696eb7d8fece5c2ef025364620dc21d7db7a5d92ff5ff6a896ed59b2e'
ATLAS_SHA = 'f6be7889418ae39c078719c64af989aaa7c7bbbe004989badaa56d34121b8e49'


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(serial(value), handle, indent=2, allow_nan=False)
        handle.write('\n')


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked_file(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'Invalid BoundFile')
    path = Path(value['path']).resolve()
    require(path.is_file() and sha(path) == value['sha256'], 'Changed bound input: '+str(path))
    return path


def checked_json(value):
    return read(checked_file(value))


def dependencies(paths):
    """Archive the flat local import closure; no imported source is omitted."""
    result, pending = {}, list(paths)
    while pending:
        path = Path(pending.pop()).resolve()
        if path.name in result:
            require(result[path.name] == path, 'Ambiguous local import '+path.name)
            continue
        result[path.name] = path
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = ([v.name for v in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                local = path.parent/(name.split('.')[0]+'.py')
                if local.is_file():
                    pending.append(local)
    return result


def validate_contract(config, protocol):
    require(config['schema'] == 'evolving-dimer-benchmark-v1'
            and protocol['schema'] == 'evolving-dimer-protocol-v1', 'Unexpected configuration schema')
    require(config['master_seed'] == MASTER, 'Changed master seed')
    require(config['allocation'] == protocol['allocation'], 'Allocation/protocol differ')
    a = config['allocation']
    for key, expected in [('contexts', 4), ('streams', 4), ('cloud_banks', 32),
                          ('raw_points_per_cloud', 16384), ('preparation_streams', 16),
                          ('prep_outer_cap', 256), ('maximum_preparation_outers', 4096),
                          ('extension', False), ('replacement', False), ('native_classification', False),
                          ('preparation_energy_evaluations', 0)]:
        require(a[key] == expected, 'Changed preparation allocation: '+key)
    require(config['contexts'] == protocol['contexts']
            and [[c['root'], c['child']] for c in config['contexts']] == PAIRS, 'Held-out panel differs')
    require(config['physical'] == dict(depletant_radius=1.4, activity=.0275, lambda_ratio=64.,
                                      wall_radius=593.742500239952), 'Physical conditions differ')
    f = config['factorized']
    require([f[k] for k in ['root_cap', 'internal_cap', 'joint_cap', 'order',
                            'uniform_probability', 'uniform_half_width', 'tau']]
            == [32, 32, 1, 'root_first', .5, 160., .25], 'Factorized proposal differs')
    p = config['preparation']
    require([p[k] for k in ['kind', 'outer_cap', 'minimum_max_center_displacement_A',
                          'minimum_max_body_orientation_degrees', 'criterion', 'energy_filter', 'native_filter']]
            == ['first_geometrically_feasible_unguided', 256, 5., 10., 'OR', False, False],
            'Geometric preparation rule differs')
    c = config['cloud']
    require([c[k] for k in ['raw_count', 'banks', 'scope', 'generate_all_before_preparation',
                          'shared_across_arms', 'refresh_threshold_each_dimer_attempt']]
            == [16384, 32, 'fixed_per_chain_family', True, True, True], 'Cloud law differs')
    for name, digest in [('shape', SHAPE_SHA), ('source_frame', FRAME_SHA), ('atlas', ATLAS_SHA)]:
        require(config[name]['sha256'] == digest, 'Changed physical/proposal identity: '+name)


def bind_inputs(base):
    """Hash/ledger structure checks only. Never evaluate protein geometry here."""
    base = Path(base).resolve()
    config, binding, manifest = [read(base/p) for p in ['config.json', 'binding.json', 'prepared/manifest.json']]
    protocol = checked_json(config['protocol'])
    validate_contract(config, protocol)
    require(binding['config_sha256'] == sha(base/'config.json')
            and binding['protocol_sha256'] == config['protocol']['sha256'], 'Execution binding differs')
    for key, path in [('freeze_sha256', 'freeze.json'),
                      ('example_source_sha256', 'common/source/examples/evolving_dimer_benchmark.rs'),
                      ('compiled_source_bundle_sha256', 'common/source-bundle.json'),
                      ('executable_sha256', 'common/evolving_dimer_benchmark')]:
        require(binding[key] == sha(base/path), 'Bound execution closure differs: '+key)
    require(manifest['schema'] == 'evolving-dimer-prepared-starts-v1'
            and manifest['complete'] is True and manifest['passed'] is True
            and manifest['all_attempts_retained'] is True, 'Preparation incomplete or failed')
    require(manifest['config_sha256'] == sha(base/'config.json')
            and manifest['binding_sha256'] == sha(base/'binding.json'), 'Prepared provenance differs')
    inputs = {}
    def add(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha(path)
    def visit(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256'}:
                add(checked_file(value))
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(config)
    for name in ['config.json', 'binding.json', 'freeze.json', 'prepared/manifest.json',
                 'common/source-bundle.json', 'common/evolving_dimer_benchmark',
                 'common/source/examples/evolving_dimer_benchmark.rs']:
        add(base/name)
    files = {}
    for item in manifest['files']:
        path = checked_file(item)
        require(path.is_relative_to(base/'prepared') and path.name != 'manifest.json'
                and str(path) not in files, 'Duplicated or invalid prepared inventory')
        files[str(path)] = item['sha256']
    actual = {str(p.resolve()) for p in (base/'prepared').rglob('*') if p.is_file()
              and p.resolve() != base/'prepared/manifest.json'}
    require(set(files) == actual, 'Omitted or extra prepared file')
    require(len(files) == 96, 'Prepared inventory allocation differs')
    clouds, starts = manifest['cloud_banks'], manifest['alternative_starts']
    cloud_keys = [(b['context_index'], b['initialization'], b['stream']) for b in clouds]
    expected_clouds = [(c, init, s) for c in range(4) for init in INITIALIZATIONS for s in range(4)]
    require(cloud_keys == expected_clouds, 'Missing, duplicated, or reordered cloud family')
    start_keys = [(s['context_index'], s['stream']) for s in starts]
    require(start_keys == [(c, s) for c in range(4) for s in range(4)], 'Alternative-start families differ')
    referenced = []
    for items, names in [(clouds, ['raw', 'metadata']), (starts, ['record', 'ledger'])]:
        for item in items:
            for name in names:
                path = checked_file(item[name])
                require(files.get(str(path)) == item[name]['sha256'], 'Uninventoried prepared reference')
                referenced.append(str(path))
    require(len(referenced) == len(set(referenced)) == len(files), 'Prepared files aliased or unreferenced')
    inputs.update(files)
    return config, manifest, inputs


def freeze(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    require(not output.exists(), 'Audit destination already exists')
    _, _, inputs = bind_inputs(base)
    closure = dependencies([Path(__file__), Path(__file__).with_name('test_audit_evolving_dimer_preparation.py')])
    (output/'source').mkdir(parents=True)
    sources = {}
    for name, path in sorted(closure.items()):
        digest = sha(path)
        shutil.copyfile(path, output/'source'/name)
        require(sha(output/'source'/name) == digest, 'Source changed during archive')
        sources[name] = digest
    protocol = dict(schema='evolving-dimer-preparation-audit-protocol-v1', base=str(base),
        complete=True, scientific_execution_started=False, input_sha256=inputs, source_sha256=sources,
        python_version=sys.version, python_executable=sys.executable,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        maximum_workers=1, raw_cloud_points=32*16384, alternative_starts=16,
        density_tolerance=dict(absolute=2e-7, relative=2e-10),
        cloud_bounds='The frozen SphereTree::bounds(0) pads each atomic AABB by 512*EPS*(1+max_i(hypot(hypot(cx,cy),cz)+r_i)). This padding changes only the saved uniform-cloud box, never sphere membership.',
        operations=['Reconstruct all raw AABB points; closed root exclusion membership and every retained index.',
            'Validate every source, raw root/internal draw, assembled endpoint and recovered frame independently.',
            'Validate all spectators, atomic spherical wall, core disjointness and internal exclusion contact.',
            'Reconstruct complete reciprocal-mixture densities including uniform defense and coordinate Jacobians.',
            'Validate every preparation null and first geometry-qualified candidate; retain unconditional ledger denominators.'],
        geometry='cKDTree only prunes candidates; NumPy atomic distance predicates determine membership.',
        new_pose_draws=0, new_point_draws=0, new_Poisson_clouds=0, native_classifier_calls=0,
        limitations=['No equilibrium or initialization independence inference from geometry-conditioned starts.',
            'No proof of pseudorandomness, floating-point geometry, or mathematically exact thinning.',
            'Saved raw variates are reconstructed, not regenerated with the Rust RNG.',
            'Physical diagnostic conditions 1.4 A/.0275 A^-3 are separate from 1.5 A/.035 A^-3.'])
    write(output/'protocol.json', protocol)
    return dict(status='frozen_without_geometry_queries', protocol=record(output/'protocol.json'),
                sources=len(sources), bound_inputs=len(inputs))


def cloud_bounds(counter):
    """Independent scalar reconstruction of the declared conservative AABB.

    SphereTree::bounds pads the atomic box. The older FFT screen instead used
    the unpadded box, so blindly reusing CountOracle.low/high is incorrect here.
    This explicit formula is checked exactly; no geometry tolerance is enlarged.
    """
    bound = max(math.hypot(math.hypot(float(c[0]), float(c[1])), float(c[2]))+float(r)
                for c, r in zip(counter.centers, counter.radii))
    guard = 512.*np.finfo(float).eps*(1.+bound)
    low = np.asarray([min(float(c[k])-float(r)-guard for c, r in zip(counter.centers, counter.radii))
                      for k in range(3)])
    high = np.asarray([max(float(c[k])+float(r)+guard for c, r in zip(counter.centers, counter.radii))
                       for k in range(3)])
    return low, high, guard


def audit_cloud(meta, raw, counter, checks, tag, expected_count=16384):
    require(set(meta) == {'raw_count', 'low', 'high', 'kept_indices', 'cpu_seconds'}, 'Cloud metadata differs')
    require(len(raw) == expected_count*24, 'Truncated or extended cloud')
    require(type(meta['raw_count']) is int and meta['raw_count'] == expected_count, 'Cloud size differs')
    require(math.isfinite(meta['cpu_seconds']) and meta['cpu_seconds'] >= 0, 'Invalid cloud CPU')
    low, high, guard = cloud_bounds(counter)
    strict_equal(meta['low'], low.tolist(), checks, 'guarded root AABB low', tag)
    strict_equal(meta['high'], high.tolist(), checks, 'guarded root AABB high', tag)
    u = np.frombuffer(raw, dtype='<f8').reshape((expected_count, 3))
    require(np.isfinite(u).all() and np.all((u >= 0) & (u < 1)), 'Invalid saved uniform')
    points = low+(high-low)*u
    expected = np.flatnonzero(counter.members(points)).tolist()
    indices = meta['kept_indices']
    require(all(type(i) is int for i in indices), 'Noninteger retained point index')
    strict_equal(indices, expected, checks, 'independent root membership', tag)
    return dict(raw_points=expected_count, retained_points=len(expected),
                reconstructed_raw_sha256=point_hash(points),
                reconstructed_retained_sha256=point_hash(points[expected]),
                setup_cpu_seconds=meta['cpu_seconds'], conservative_AABB_padding_A=guard)


def separation(old, new):
    translations, angles = [], []
    for a, b in zip(old, new):
        p, q = np.asarray(a['orientation'], float), np.asarray(b['orientation'], float)
        require(p.shape == q.shape == (4,) and np.isfinite(p).all() and np.isfinite(q).all()
                and np.linalg.norm(p) > 0 and np.linalg.norm(q) > 0, 'Invalid preparation quaternion')
        translations.append(float(np.linalg.norm(np.asarray(a['position'])-b['position'])))
        angles.append(math.degrees(2*math.acos(min(1., abs(float(p@q))/(np.linalg.norm(p)*np.linalg.norm(q))))))
    return dict(max_center_displacement_A=max(translations), max_body_orientation_degrees=max(angles))


def qualifies(old, new, preparation):
    value = separation(old, new)
    return (value['max_center_displacement_A'] >= preparation['minimum_max_center_displacement_A']
            or value['max_body_orientation_degrees'] >= preparation['minimum_max_body_orientation_degrees'])


def audit_start(rows, saved, helper, oracle, case, old, anchor, preparation, checks, tag, progress=None):
    require(saved['source'] == old and saved['status'] == 'prepared'
            and saved['is_equilibrium_sample'] is False, 'Invalid saved preparation source/status')
    require(type(saved['attempts']) is int and 1 <= saved['attempts'] <= preparation['outer_cap']
            and len(rows) == saved['attempts'], 'Missing or excessive preparation attempts')
    members = [case['root'], case['child']]
    geometry = oracle.fingerprint(members, old)
    source = feasibility(oracle, case, old, geometry)
    require(geometry['hard_valid'] and is_feasible(source), 'Invalid initial pair')
    results = []
    for j, row in enumerate(rows, 1):
        where = f'{tag}/attempt{j}'
        require(type(row['attempt']) is int and row['attempt'] == j, 'Preparation attempt gap')
        require('failure' not in row, 'Retained fatal preparation; audit cannot pass')
        require(set(row) == {'attempt', 'outcome', 'selected'} and type(row['selected']) is bool,
                'Preparation row schema differs')
        outcome = row['outcome']
        for key, expected in [('members', members), ('anchor_label', case['anchor']), ('old', old)]:
            strict_equal(outcome[key], expected, checks, 'fixed preparation '+key, where)
        require(outcome.get('guidance') is None, 'Guided preparation not allocated')
        frames = [outcome['source_frame']]+[a['frame'] for a in outcome['attempts'] if a['frame'] is not None]
        require(all(frame.get('guidance') is None for frame in frames), 'Guided preparation frame')
        require(all('guidance_count' not in d for a in outcome['attempts'] for d in a['internal_draws']),
                'Unguided preparation measured counts')
        # This runner does not save passive contact fingerprints. Independently
        # construct adapter values; full saved predicates are still cross-checked.
        contacts = [None if a['proposed'] is None else oracle.fingerprint(members, a['proposed'])['contacts']
                    for a in outcome['attempts']]
        result = audit_factorized(helper, oracle, case, old, anchor, outcome, contacts, source, checks, where)
        candidate = outcome['candidate']
        proposed = None if candidate is None else [candidate['root'], candidate['child']]
        chosen = proposed is not None and qualifies(old, proposed, preparation)
        require(row['selected'] == chosen, 'Selection differs from geometric OR rule')
        require(chosen == (j == len(rows)), 'Not the first qualifying candidate or hidden continuation')
        if chosen:
            strict_equal(saved['selected'], proposed, checks, 'saved selected candidate', where)
        result.update(attempt=j, status=outcome['status'], selected=chosen,
                      separation=None if proposed is None else separation(old, proposed))
        results.append(result)
        if progress is not None:
            progress(dict(kind='preparation_attempt', tag=where, status=outcome['status'], selected=chosen,
                          checks=checks.count, failures=len(checks.failures)))
    return dict(source_feasibility=source, source_geometry=geometry, attempts=results,
                selected_separation=results[-1]['separation'])


def audit(base, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    protocol = read(output/'protocol.json')
    require(protocol['schema'] == 'evolving-dimer-preparation-audit-protocol-v1'
            and protocol['base'] == str(base) and protocol['scientific_execution_started'] is False,
            'Audit protocol differs')
    require(Path(__file__).resolve() == output/'source'/Path(__file__).name, 'Execute archived auditor')
    for name, digest in protocol['source_sha256'].items():
        require(sha(output/'source'/name) == digest, 'Audit source changed: '+name)
    require(sys.version == protocol['python_version'] and np.__version__ == protocol['numpy_version']
            and scipy.__version__ == protocol['scipy_version'], 'Numerical runtime changed')
    config, manifest, inputs = bind_inputs(base)
    require(inputs == protocol['input_sha256'], 'Inputs changed after audit freeze')
    started = time.process_time()
    checks, clouds, starts = Checks(), [], []
    result = dict(schema=SCHEMA, complete=False, passed=False, protocol=record(output/'protocol.json'),
                  physical=config['physical'], source_contexts=config['contexts'], input_sha256=inputs,
                  new_pose_draws=0, new_point_draws=0, new_Poisson_clouds=0, native_classifier_calls=0,
                  limitations=protocol['limitations'])
    with (output/'progress.jsonl').open('x') as ledger:
        def progress(value):
            ledger.write(json.dumps(serial(value), allow_nan=False)+'\n')
            ledger.flush()
        try:
            source, frame = checked_json(config['source_config']), checked_json(config['source_frame'])
            require(source['initial_poses'] == frame['poses'] and frame['sweep'] == 7400
                    and len(frame['poses']) == 264, 'Source configuration differs')
            require(source['depletant_radius'] == 1.4 and source['reservoir_density'] == .0275
                    and source['boundary'] == dict(kind='spherical', radius=config['physical']['wall_radius']),
                    'Source physical measure differs')
            state = frame['poses']
            shape = checked_json(config['shape'])
            oracle = DimerGeometry(shape, state, 1.4, config['physical']['wall_radius'])
            counter = CountOracle(oracle.centers, oracle.radii+oracle.rd)
            helper = DimerDestinationDensity(checked_json(config['atlas']))
            require(len(helper.map_density.weights) == 2048, 'Incomplete reciprocal mixture')
            for bank in manifest['cloud_banks']:
                key = [bank[k] for k in ['context_index', 'initialization', 'stream']]
                item = audit_cloud(checked_json(bank['metadata']), checked_file(bank['raw']).read_bytes(),
                                   counter, checks, '/'.join(map(str, key)))
                clouds.append(dict(family=key, **item))
                progress(dict(kind='cloud', family=key, retained_points=item['retained_points'], checks=checks.count))
            for start in manifest['alternative_starts']:
                ci, stream = start['context_index'], start['stream']
                saved = checked_json(start['record'])
                require(saved['context_index'] == ci and saved['stream'] == stream
                        and saved['attempts'] == start['attempts'] and start['status'] == 'prepared',
                        'Prepared record family differs')
                with checked_file(start['ledger']).open() as handle:
                    rows = [json.loads(line) for line in handle]
                case = config['contexts'][ci]
                old, anchor = [state[case[k]] for k in ['root', 'child']], state[case['anchor']]
                item = audit_start(rows, saved, helper, oracle, case, old, anchor, config['preparation'],
                                   checks, f'{ci}/{stream}', progress)
                starts.append(dict(context_index=ci, stream=stream, **item))
            result.update(complete=True, passed=not checks.failures)
        except Exception as error:
            result['fatal_error'] = f'{type(error).__name__}: {error}'
            progress(dict(kind='fatal', error=result['fatal_error']))
    attempts = [row for start in starts for row in start['attempts']]
    result.update(checks=checks.count, failures=checks.failures,
        maximum_absolute_errors=dict(checks.maximum_absolute_errors), clouds=clouds, starts=starts,
        summary=dict(cloud_banks=len(clouds), raw_cloud_points=sum(c['raw_points'] for c in clouds),
            retained_cloud_points=sum(c['retained_points'] for c in clouds), alternative_starts=len(starts),
            preparation_attempts=len(attempts), candidates=sum(a['candidate'] is not None for a in attempts),
            nulls=sum(a['candidate'] is None for a in attempts),
            stage_status_counts=dict(Counter(a['stage_status'] for a in attempts)),
            raw_root_draws=sum(a['root_draws'] for a in attempts),
            raw_internal_draws=sum(a['internal_draws'] for a in attempts)),
        analyzer_cpu_seconds=time.process_time()-started, progress=record(output/'progress.jsonl'))
    write(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--audit', action='store_true')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    answer = freeze(args.run, args.output) if args.freeze else audit(args.run, args.output)
    print(json.dumps({k: serial(answer[k]) for k in ['status', 'protocol', 'complete', 'passed', 'summary',
        'fatal_error', 'checks', 'maximum_absolute_errors', 'analyzer_cpu_seconds'] if k in answer}, indent=2))
    raise SystemExit(0 if args.freeze or answer.get('passed') else 1)
