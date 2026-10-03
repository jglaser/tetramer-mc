#!/usr/bin/env python3
"""Independently audit the fixed FFT covariance-width proposal comparison.

Every saved draw, failed stage, null, source density and complete mixture
correction is retained. Unguided overlap counts remain unmeasured. This auditor
reconstructs recorded coordinates; it never samples proposals or physical baths.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from analyze_auxiliary_overlap_probe import CountOracle, audit_cloud, audit_guided, audit_rng
from analyze_factorized_dimer_probe import audit_factorized, strict_equal, internal_ok
from analyze_capped_dimer_probe import (
    read, sha, bound, require, Checks, DimerDestinationDensity, DimerGeometry,
    bind_current_source, feasibility, relative_pose, fingerprints, quantiles, serial,
    REFERENCE_CONFIG_SHA256, PANEL_SHA256, log_value,
)

MASTER = 6100300401
SCALES = [.125, .25, .5]
METHODS = ['unguided', 'm4']
NAMES = ['blind_fft512slots_tau0p125', 'blind_fft512slots_tau0p25', 'blind_fft512slots_tau0p5']
FFT_SHA = 'c460dc61fb7bf9e76f1d4dca77987b5886ea82cdda5d703ea5de6a25733fcb08'
CAPS = dict(root=32, internal=32, joint=1)
TIMINGS = ['proposal_cpu_seconds', 'cloud_construction_cpu_seconds',
           'guidance_setup_cpu_seconds', 'standalone_proposal_cpu_seconds',
           'contact_diagnostic_cpu_seconds', 'old_density_evaluation_cpu_seconds']


def seed(master, case, attempt, role):
    return int(hashlib.sha256(f'fft-width-probe-v1/{master}/{case}/{attempt}/{role}'.encode()).hexdigest()[:16], 16)


def method_order(case, attempt, master=MASTER):
    rows = [[i, method, seed(master, case, attempt, f'execution_order/{i}/{method}')]
            for i in range(3) for method in METHODS]
    return sorted(rows, key=lambda r: (r[2], r[0], r[1]))


def validate_scaled(original, scaled, tau):
    """Authenticate the whole JSON, including off-diagonal covariances and means."""
    require(tau in SCALES, 'Unallocated covariance width')
    require(original['schema'] == 'reciprocal-pose-mixture-v1' and
            original['base_model']['schema'] == 'weighted-pose-mixture-v1', 'Unknown original model')
    before, after = original['base_model']['covariances'], scaled['base_model']['covariances']
    require(len(before) > 0 and len(before) == len(after), 'Changed component count')
    for a, b in zip(before, after):
        require(len(a) == len(b) == 6, 'Covariance dimensions')
        for xrow, yrow in zip(a, b):
            require(len(xrow) == len(yrow) == 6, 'Covariance row dimensions')
            for x, y in zip(xrow, yrow):
                require(type(x) in (int, float) and type(y) in (int, float) and
                        math.isfinite(x) and math.isfinite(y) and y == x * (tau * tau),
                        'Covariance entry is not scaled by tau squared')
    restored = copy.deepcopy(scaled)
    restored['base_model']['covariances'] = copy.deepcopy(before)
    require(restored == original, 'Changed non-covariance atlas data')


def expected_allocation():
    return dict(schema='fft-width-allocation-v1', contexts=8, slots_per_context=32,
        covariance_scales=SCALES, methods=METHODS, new_outer_attempts=1536,
        clouds=256, raw_points_per_cloud=16384, raw_cloud_points=4194304,
        raw_cloud_bytes=100663296, maximum_raw_edge_draws=98304,
        historical_tau1_attempts=512, historical_new_draws=0, extension=False,
        master_seed=MASTER, caps=CAPS, order='root_first', uniform_probability=.5,
        uniform_half_width_A=160., physical_bath_draws=0, state_updates=0,
        native_classification=False)


def validate_contract(config, protocol):
    require(config['schema'] == 'fft-width-screen-v1' and
            config['density_law'] == 'map-factor-full-mixture-v1', 'Unknown width law')
    require(config['master_seed'] == protocol['master_seed'] == MASTER, 'Wrong master seed')
    require([config[k] for k in ('root_cap', 'internal_cap', 'factorized_joint_cap',
                                'attempts_per_context', 'cloud_raw_count')] == [32, 32, 1, 32, 16384],
            'Changed fixed allocation')
    require(config['factorized_order'] == 'root_first', 'Changed stage order')
    require([(a['name'], a['tau']) for a in config['scaled_atlases']] == list(zip(NAMES, SCALES)),
            'Changed width allocation')
    require(config['scaled_atlases'] == protocol['scaled_atlases'], 'Protocol width artifacts differ')
    require(config['scientific_allocation'] == protocol['scientific_allocation'], 'Allocation binding differs')
    allocation = bound(config['scientific_allocation'])
    require(allocation == protocol['allocation'] == expected_allocation(), 'Changed scientific allocation')
    require(protocol['source_atlas_index'] == 1 and protocol['source_atlas_sha256'] == FFT_SHA and
            protocol['covariance_scales'] == SCALES and protocol['methods'] == METHODS, 'Changed source/arms')
    require(protocol['uniform_probability'] == .5 and protocol['uniform_half_width_A'] == 160. and
            protocol['density_law'] == config['density_law'], 'Changed defensive/full-F law')
    require(protocol['caps'] == CAPS and protocol['order'] == 'root_first', 'Changed caps/order')
    require(protocol['density_tolerance'] == dict(absolute=2e-7, relative=2e-10), 'Changed tolerance')
    require(protocol['physical_bath_draws'] == protocol['state_updates'] == 0 and
            protocol['native_classification'] is False, 'Not a passive allocation')
    require(protocol['cloud_law']['raw_count'] == 16384 and protocol['cloud_law']['is_poisson'] is False,
            'Changed auxiliary cloud law')
    roles = ['cloud', 'proposal', 'threshold'] + [f'execution_order/{i}/{m}' for i in range(3) for m in METHODS]
    seeds = [seed(MASTER, c, j, role) for c in range(8) for j in range(32) for role in roles]
    require(len(set(seeds)) == 2304, 'RNG domain collision')
    return allocation


def source_densities(helper, old, anchor):
    coordinates = [relative_pose(anchor, old[0]), relative_pose(old[0], old[1])]
    densities = [helper.edge_density(p, .5, 160., kind='map') for p in coordinates]
    return coordinates, densities


def audit_source_density(row, coordinates, densities, checks, tag):
    """Always called, including nulls; expected densities belong to this width."""
    require(len(row['old_coordinates']) == len(row['old_edges']) == 2, 'Incomplete source edge densities')
    for i, (coordinate, expected) in enumerate(zip(coordinates, densities)):
        checks.pose(row['old_coordinates'][i], coordinate, 'independent source coordinate', tag)
        require(set(row['old_edges'][i]) == {'log_uniform', 'log_learned', 'log_full'}, 'Incomplete source F')
        for key in ('log_uniform', 'log_learned', 'log_full'):
            checks.close(log_value(row['old_edges'][i][key]), expected[key], 'source map-factor ' + key, tag)
    if row['outcome']['candidate'] is not None:
        actual = row['outcome']['candidate']['diagnostics']['old_edges']
        require(actual == row['old_edges'], 'Candidate source F differs from explicit width source F')


def audit_outcome(helper, oracle, counter, points, case, old, anchor, row, source, checks, tag):
    method, outcome = row['method'], row['outcome']
    if method == 'm4':
        return audit_guided(helper, oracle, counter, points, case, old, anchor, outcome,
                            row['raw_contacts'], source, 4, checks, tag)
    require(method == 'unguided', 'Unallocated arm')
    require(outcome.get('guidance') is None, 'Unguided outcome has guidance')
    frames = [outcome['source_frame']] + [a['frame'] for a in outcome['attempts'] if a['frame'] is not None]
    require(all(f.get('guidance') is None for f in frames), 'Unguided frame has count data')
    require(all('guidance_count' not in draw for a in outcome['attempts'] for draw in a['internal_draws']),
            'Unexpected unguided count query')
    result = audit_factorized(helper, oracle, case, old, anchor, outcome, row['raw_contacts'], source, checks, tag)
    result.update(old_count=None, threshold=None, threshold_failures=0,
                  internal_geometric_passes=sum(internal_ok(d['feasibility']) for a in outcome['attempts'] for d in a['internal_draws']),
                  count_queries=0, point_tests=0)
    if result['candidate'] is not None:
        candidate = result['candidate']
        candidate.update(old_count=None, new_count=None, count_retention=None, aux_log_correction=0.,
                         complete_log_correction=candidate['log_reverse_forward'])
    return result


def same_width_prefixes(pair, checks, tag):
    require(set(pair) == set(METHODS), 'Missing same-width arm')
    unguided, guided = pair['unguided'], pair['m4']
    for field in ('old_coordinates', 'old_edges'):
        strict_equal(unguided[field], guided[field], checks, 'same-width ' + field, tag)
    u, g = [r['outcome']['attempts'] for r in (unguided, guided)]
    require(len(u) == len(g) == 1, 'Missing fixed joint attempt')
    strict_equal(u[0]['root_draws'], g[0]['root_draws'], checks, 'same-width root prefix', tag)
    a, b = u[0]['internal_draws'], g[0]['internal_draws']
    require(len(a) <= len(b), 'Guidance stopped before unconditioned first success')
    project = lambda records: [{k: v for k, v in r.items() if k != 'guidance_count'} for r in records]
    strict_equal(project(a), project(b[:len(a)]), checks, 'same-width internal prefix', tag)
    if len(a) == len(b):
        strict_equal(unguided['rng_after_fingerprint'], guided['rng_after_fingerprint'], checks,
                     'same-stop proposal RNG', tag)


def shared_cloud_thresholds(rows, checks, tag):
    """Cross-width pairing stops at the common cloud/threshold, not raw poses."""
    require(set(rows) == {(i, m) for i in range(3) for m in METHODS}, 'Missing width/arm')
    guides = [rows[i, 'm4'] for i in range(3)]
    for row in guides[1:]:
        for key in ('point_count', 'old_count', 'integer_draws', 'threshold', 'm'):
            strict_equal(row['outcome']['guidance'][key], guides[0]['outcome']['guidance'][key],
                         checks, 'shared-cloud guide ' + key, tag)
        strict_equal(row['auxiliary_rng_after_fingerprint'], guides[0]['auxiliary_rng_after_fingerprint'],
                     checks, 'shared-threshold RNG', tag)
    uniforms = [rows[i, 'unguided'] for i in range(3)]
    for row in uniforms[1:]:
        strict_equal(row['auxiliary_rng_after_fingerprint'], uniforms[0]['auxiliary_rng_after_fingerprint'],
                     checks, 'unused auxiliary RNG', tag)
    for i in range(3):
        same_width_prefixes({m: rows[i, m] for m in METHODS}, checks, f'{tag}/width{i}')


def summarize(rows):
    candidates = [r['candidate'] for r in rows if r['candidate'] is not None]
    cpu = sum(r['standalone_proposal_cpu_seconds'] for r in rows)
    q = lambda key: quantiles([r[key] for r in rows if r[key] is not None])
    cq = lambda key: quantiles([c[key] for c in candidates if c[key] is not None])
    return dict(outer_attempts=len(rows), candidate_count=len(candidates), proposal_nulls=len(rows) - len(candidates),
        candidate_fraction=len(candidates) / len(rows) if rows else None,
        raw_edge_draws=sum(r['raw_edge_draws'] for r in rows), root_draws=sum(r['root_draws'] for r in rows),
        internal_draws=sum(r['internal_draws'] for r in rows), assembled_endpoints=sum(r['assembled'] for r in rows),
        threshold_failures=sum(r['threshold_failures'] for r in rows),
        internal_geometric_passes=sum(r['internal_geometric_passes'] for r in rows),
        count_queries=sum(r['count_queries'] for r in rows), point_tests=sum(r['point_tests'] for r in rows),
        **{key: sum(r[key] for r in rows) for key in TIMINGS},
        candidates_per_standalone_cpu_second=len(candidates) / cpu if cpu else None,
        status_counts=dict(Counter(r['status'] for r in rows)), stage_status_counts=dict(Counter(r['stage_status'] for r in rows)),
        candidates_with_external_contacts=sum(c['external_contacts'] > 0 for c in candidates),
        both_intended_contacts=sum(c['both_intended_contacts'] for c in candidates),
        old_count=q('old_count'), threshold=q('threshold'), new_count=cq('new_count'),
        count_retention=cq('count_retention'), log_reverse_forward=cq('log_reverse_forward'),
        aux_log_correction=cq('aux_log_correction'), complete_log_correction=cq('complete_log_correction'))


def analyze(run):
    clock = time.process_time()
    run = Path(run)
    config, binding, terminal, protocol = [read(run / p) for p in ('config.json', 'binding.json', 'terminal.json', 'protocol.json')]
    require(binding['schema'] == 'fft-width-probe-binding-v1', 'Unknown binding')
    for name, key in [('config.json', 'config_sha256'), ('protocol.json', 'protocol_sha256'), ('example.rs', 'example_source_sha256')]:
        require(sha(run / name) == binding[key], 'Changed binding ' + name)
    for name, key in [('attempts.jsonl', 'attempts_sha256'), ('clouds.jsonl', 'clouds_sha256'), ('cloud-uniforms.bin', 'cloud_uniforms_sha256')]:
        require(sha(run / name) == terminal[key], 'Changed ledger ' + name)
    require(terminal['summary']['complete'] is True, 'Retained fatal run; no replacement allocation')
    allocation = validate_contract(config, protocol)
    require(config['reference_config']['sha256'] == REFERENCE_CONFIG_SHA256, 'Changed reference config')
    ref = bound(config['reference_config'])
    panel = bound(ref['panel'])
    require(ref['panel']['sha256'] == PANEL_SHA256 and ref['cases'] == panel['cases'] and len(ref['cases']) == 8,
            'Changed fixed contexts')
    require(ref['uniform_probability'] == .5 and ref['uniform_half_width'] == 160. and
            ref['depletant_radius'] == 1.4 and ref['activity'] == .0275, 'Changed physical/defensive model')
    source, frame, archive, shape = [bound(ref[key]) for key in ('source_config', 'source_frame', 'source_freeze_manifest', 'shape')]
    require(source['initial_poses'] == frame['poses'] and archive['frame_sha256'] == ref['source_frame']['sha256'] and
            archive['shape_sha256'] == ref['shape']['sha256'], 'Changed frozen source')
    require(source['depletant_radius'] == ref['depletant_radius'] and source['reservoir_density'] == ref['activity'] and
            source['boundary'] == dict(kind='spherical', radius=ref['wall_radius']), 'Source physical conditions differ')
    source_binding = bind_current_source(run, config, binding, protocol)
    for name, digest in protocol['guidance_source_sha256'].items():
        require(config['compiled_source_sha256'][name] == digest, 'Changed guidance source')
        source_binding[name] = digest
    require(set(protocol['guidance_source_sha256']) == {'src/factorized_dimer.rs', 'src/auxiliary_overlap_threshold.rs'},
            'Incomplete guidance source binding')
    require(sha(__file__) == protocol['audit_files'][Path(__file__).name]['sha256'], 'Unbound executing auditor')
    require(len(ref['atlases']) == 3 and ref['atlases'][1]['name'] == 'blind_fft512slots' and
            ref['atlases'][1]['model']['sha256'] == FFT_SHA, 'Wrong original atlas')
    original = bound(ref['atlases'][1]['model'])
    helpers = []
    for atlas in config['scaled_atlases']:
        model = bound(atlas['model'])
        validate_scaled(original, model, atlas['tau'])
        helpers.append(DimerDestinationDensity(model))
    require(len(original['base_model']['covariances']) == 1024 and
            all(len(helper.map_density.weights) == 2048 for helper in helpers), 'Changed complete FFT support')
    state = source['initial_poses']
    oracle = DimerGeometry(shape, state, ref['depletant_radius'], ref['wall_radius'])
    counter = CountOracle(oracle.centers, oracle.radii + oracle.rd)
    for key, expected in [('low', counter.low.tolist()), ('high', counter.high.tolist())]:
        require(protocol['cloud_law'][key] == expected, 'Cloud bounds differ')
    volume = float(np.prod(counter.width))
    require(protocol['cloud_law']['volume_A3'] == volume and
            protocol['cloud_law']['effective_raw_intensity_Aminus3'] == 16384 / volume, 'Cloud intensity differs')
    checks, results, sources = Checks(), [], []
    retained_total = 0
    with (run / 'attempts.jsonl').open() as stream, (run / 'clouds.jsonl').open() as metadata, (run / 'cloud-uniforms.bin').open('rb') as blob:
        for ci, case in enumerate(ref['cases']):
            old, anchor = [state[case['root']], state[case['child']]], state[case['anchor']]
            geometry = oracle.fingerprint([case['root'], case['child']], old)
            sourcef = feasibility(oracle, case, old, geometry)
            require(geometry['hard_valid'] and sourcef['internal_exclusion_contact'], 'Ineligible fixed source')
            expected_sources = [source_densities(helper, old, anchor) for helper in helpers]
            sources.append(dict(case=case['name'], feasibility=sourcef,
                widths=[dict(atlas=a['name'], tau=a['tau'], old_coordinates=coords, old_edges=densities)
                        for a, (coords, densities) in zip(config['scaled_atlases'], expected_sources)]))
            for attempt in range(32):
                cloud_id, tag = ci * 32 + attempt, f'{ci}/{attempt}'
                line = metadata.readline()
                require(bool(line), 'Missing cloud')
                meta = json.loads(line)
                for key, expected in [('cloud_id', cloud_id), ('case_index', ci), ('attempt', attempt),
                    ('seed', seed(MASTER, ci, attempt, 'cloud')), ('threshold_seed', seed(MASTER, ci, attempt, 'threshold')),
                    ('raw_byte_offset', cloud_id * 16384 * 24), ('raw_byte_length', 16384 * 24)]:
                    strict_equal(meta[key], expected, checks, 'cloud ' + key, tag)
                audit_rng(meta, 'rng_after_fingerprint')
                points = audit_cloud(meta, blob.read(16384 * 24), counter, checks, tag)
                retained_total += len(points)
                require(math.isfinite(meta['cloud_construction_cpu_seconds']) and meta['cloud_construction_cpu_seconds'] >= 0, 'Invalid cloud timing')
                order, rows = method_order(ci, attempt), {}
                for position, (ai, method, order_seed) in enumerate(order):
                    line = stream.readline()
                    require(bool(line), 'Missing outer row')
                    row = json.loads(line)
                    atlas, where = config['scaled_atlases'][ai], f'{tag}/{ai}/{method}'
                    rows[ai, method] = row
                    require(row['status'] == 'completed', 'Retained fatal outer')
                    for key, expected in [('atlas', atlas['name']), ('atlas_index', ai), ('scale_index', ai), ('source_atlas_index', 1),
                        ('tau', atlas['tau']), ('case', case), ('case_index', ci), ('attempt', attempt), ('method', method),
                        ('execution_order', order), ('execution_position', position), ('order_seed', order_seed),
                        ('seed', seed(MASTER, ci, attempt, 'proposal')), ('old', old), ('anchor_pose', anchor),
                        ('old_contacts', geometry['contacts']), ('cloud_id', cloud_id), ('cloud_seed', meta['seed']),
                        ('auxiliary_seed', meta['threshold_seed']), ('m', 4 if method == 'm4' else 0),
                        ('cloud_construction_cpu_seconds', meta['cloud_construction_cpu_seconds'])]:
                        strict_equal(row[key], expected, checks, 'row ' + key, where)
                    for key in ('rng_after_fingerprint', 'auxiliary_rng_after_fingerprint'):
                        audit_rng(row, key)
                    for key in TIMINGS:
                        require(math.isfinite(row[key]) and row[key] >= 0, 'Invalid timing ' + key)
                    charged = row['proposal_cpu_seconds'] + (row['cloud_construction_cpu_seconds'] + row['guidance_setup_cpu_seconds'] if method == 'm4' else 0.)
                    checks.close(row['standalone_proposal_cpu_seconds'], charged, 'standalone charged CPU', where)
                    outcome = row['outcome']
                    for key, expected in [('members', [case['root'], case['child']]), ('anchor_label', case['anchor']), ('old', old)]:
                        strict_equal(outcome[key], expected, checks, 'outcome ' + key, where)
                    audit_source_density(row, *expected_sources[ai], checks, where)
                    result = audit_outcome(helpers[ai], oracle, counter, points, case, old, anchor, row, sourcef, checks, where)
                    count = result['root_draws'] + result['internal_draws']
                    strict_equal(row['raw_edge_draws'], count, checks, 'raw edge count', where)
                    require(count <= 64, 'Raw edge cap exceeded')
                    if result['candidate'] is None:
                        require(row['complete_log_correction'] is None, 'Complete correction without candidate')
                    else:
                        checks.close(row['complete_log_correction'], result['candidate']['complete_log_correction'], 'complete correction', where)
                    item = dict(atlas=atlas['name'], atlas_index=ai, tau=atlas['tau'], case=case['name'], case_index=ci,
                                attempt=attempt, cloud_id=cloud_id, method=method, execution_position=position,
                                status=outcome['status'], raw_edge_draws=count, old_edges=row['old_edges'],
                                **{key: row[key] for key in TIMINGS}, **result)
                    if result['candidate'] is not None:
                        item['contact_change'] = fingerprints(geometry['contacts'], result['candidate']['contacts'])
                    results.append(item)
                shared_cloud_thresholds(rows, checks, tag)
        require(not stream.readline() and not metadata.readline() and not blob.read(1), 'Unallocated extra row/cloud/point')
    total = summarize(results)
    require(len(results) == 1536 and total['raw_edge_draws'] <= 98304, 'Wrong completed allocation')
    for key, expected in [('outer_attempts', 1536), ('raw_edge_draws', total['raw_edge_draws']),
        ('candidates', total['candidate_count']), ('clouds', 256), ('raw_cloud_points', 4194304),
        ('retained_cloud_points', retained_total), ('physical_bath_draws', 0), ('state_updates', 0)]:
        strict_equal(terminal['summary']['result'][key], expected, checks, 'terminal ' + key, 'terminal')
    comparisons = [dict(atlas=a['name'], tau=a['tau'], method=m,
        summary=summarize([r for r in results if r['atlas'] == a['name'] and r['method'] == m]),
        contexts=[dict(case=c['name'], summary=summarize([r for r in results if r['atlas'] == a['name'] and r['method'] == m and r['case'] == c['name']])) for c in ref['cases']])
        for a in config['scaled_atlases'] for m in METHODS]
    return dict(schema='fft-width-independent-analysis-v1', complete=True, passed=not checks.failures,
        checks=checks.count, failures=checks.failures, maximum_absolute_errors=dict(checks.maximum_absolute_errors),
        source_binding=source_binding, input_hashes={p: sha(run / p) for p in
            ('config.json', 'binding.json', 'protocol.json', 'source-bundle.json', 'example.rs', 'attempts.jsonl', 'clouds.jsonl', 'cloud-uniforms.bin', 'terminal.json')},
        scientific_allocation_sha256=config['scientific_allocation']['sha256'], allocation=allocation,
        summary=total, comparisons=comparisons, source_contexts=sources, rows=results,
        analyzer_cpu_seconds=time.process_time() - clock, probe_cpu_seconds=terminal['cpu_seconds'],
        limitations=protocol['limitations'], density_tolerance=protocol['density_tolerance'],
        pairing=dict(independent_context_slots=256, new_outers=1536, clouds=256,
            same_width_raw_prefixes_shared=True, threshold_shared_across_widths=True,
            internal_raw_draws_matched_across_widths=False, historical_tau1_reused=False,
            unguided_overlap_counts='unmeasured; None retained', statistical_claim='dependent passive fixed-context comparison; no physical acceptance or ESS claim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.run)
    except Exception as error:
        result = dict(schema='fft-width-independent-analysis-v1', complete=False, passed=False, fatal_error=f'{type(error).__name__}: {error}')
    with args.output.open('x') as out:
        json.dump(serial(result), out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({key: serial(result.get(key)) for key in ('complete', 'passed', 'checks', 'fatal_error', 'summary', 'maximum_absolute_errors', 'analyzer_cpu_seconds')}, indent=2))
    raise SystemExit(0 if result.get('passed') else 1)
