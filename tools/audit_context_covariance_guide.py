"""Independent audit of the frozen full/diagonal source-covariance control.

Reuses the completed guide auditor's scalar, journal and generation checks.
Admits only the separately frozen eight-population prospective allocation.
No physical weight, cloud draw, proposal resampling, or fitted parameter update.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from audit_context_candidate_bank import (GeometryReference, MapDensityReference,
    bind_map_source, canonical_tokens, close_log, coverage_summary, finite_log, require)
from audit_context_source_guide import (scalar_row, audit_generation, audit_events,
                                      read, sha)
from source_guide_reference import SourceDensity

PANEL_ORDINALS = tuple(range(0, 2048, 32))
ARMS = ('full', 'diagonal')


def validate_contract(protocol, expected_sha):
    """Authenticate allocation/assets before interpreting any candidate results."""
    require(type(expected_sha) is str and len(expected_sha) == 64
            and all(c in '0123456789abcdef' for c in expected_sha), 'Invalid expected allocation digest')
    require(protocol['schema'] == 'context-covariance-guide-audit-v1'
            and protocol['panel_ordinals'] == list(PANEL_ORDINALS)
            and protocol['expected_populations'] == 8
            and protocol['draws_per_population'] == 2048, 'Changed audit allocation')
    bindings = protocol['input_sha256']
    def asset(record):
        path = Path(record['path']).resolve()
        require(record['sha256'] == bindings.get(str(path)) == sha(path), 'Unbound/changed frozen asset')
        return read(path)
    require(protocol['allocation']['sha256'] == expected_sha, 'Wrong admitted allocation digest')
    allocation = asset(protocol['allocation'])
    require(allocation['schema'] == 'context-source-covariance-geometry-allocation-v1'
            and allocation['arms'] == list(ARMS) and allocation['streams_per_arm'] == 4
            and allocation['draws_per_population'] == 2048 and allocation['populations'] == 8
            and allocation['total_draws'] == 16384 and allocation['clouds'] == 0
            and allocation['mixture'] == [.5, .25, .25]
            and allocation['training_streams'] == [0, 1]
            and allocation['heldout_streams'] == [2, 3]
            and allocation['panel_ordinals'] == list(PANEL_ORDINALS), 'Changed frozen allocation contents')
    require(allocation['guides'] == protocol['guides'] and allocation['reduction'] == protocol['reduction'],
            'Guide/reduction differs from admitted allocation')
    reduction = asset(protocol['reduction'])
    require(reduction['complete'] and reduction['passed'], 'Incomplete guide reduction')
    require(set(protocol['guides']) == set(ARMS), 'Changed guide arm inventory')
    guides = {}
    for arm in ARMS:
        guide = asset(protocol['guides'][arm])
        require(guide['schema'] == 'context-covariance-frozen-guide-v1' and guide['arm'] == arm
                and guide['training_streams'] == [0, 1] and guide['heldout_streams'] == [2, 3]
                and guide['reduction_sha256'] == protocol['reduction']['sha256'], 'Guide reduction provenance differs')
        chart = guide['source_chart']
        require(set(chart) == {'angular_length', 'covariance', 'explicit_gaussian'}
                and chart['explicit_gaussian']['schema'] == 'source-gaussian-v1'
                and type(chart['explicit_gaussian']['provenance']) is str
                and bool(chart['explicit_gaussian']['provenance'].strip()), 'Invalid explicit guide chart')
        guides[arm] = dict(guide, asset_sha256=protocol['guides'][arm]['sha256'])
    require(guides['full']['source_chart']['angular_length'] == guides['diagonal']['source_chart']['angular_length']
            and guides['full']['source_chart']['explicit_gaussian']['mean']
                == guides['diagonal']['source_chart']['explicit_gaussian']['mean'], 'Arms must share center/mean')
    full = np.asarray(guides['full']['source_chart']['covariance'], float)
    diagonal = np.asarray(guides['diagonal']['source_chart']['covariance'], float)
    require(full.shape == diagonal.shape == (6, 6) and np.isfinite(full).all()
            and np.isfinite(diagonal).all() and np.array_equal(np.diag(np.diag(full)), diagonal),
            'Diagonal arm must retain exactly the full covariance diagonal')
    return allocation, guides


def validate_admitted_config(entry, cfg, digest, guides, identities):
    """No result-dependent config/guide fallback; actual executed bytes are bound."""
    arm, stream = entry['arm'], entry['stream']
    require(arm in ARMS and type(stream) is int and stream in range(4)
            and (arm, stream) not in identities, 'Missing/repeated population identity')
    population = 4*ARMS.index(arm)+stream
    require(entry['population'] == cfg['population'] == population
            and entry['config_sha256'] == digest
            and cfg['identity']['arm'] == arm and cfg['identity']['stream'] == stream
            and cfg['identity']['guide_sha256'] == guides[arm]['asset_sha256']
            and cfg['source_chart'] == guides[arm]['source_chart'], 'Actual config differs from frozen guide/admission')
    require(type(cfg['seed']) is int and 0 <= cfg['seed'] < 2**64, 'Invalid population seed')
    identities.add((arm, stream))
    return arm, stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-allocation-sha256', required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    allocation, guides = validate_contract(protocol, args.expected_allocation_sha256)
    bindings = protocol['input_sha256']
    for path, digest in bindings.items(): require(sha(path) == digest, 'Changed input: '+path)
    def bound(path):
        key = str(Path(path).resolve())
        require(key in bindings, 'Unbound audit input: '+key)
        return read(key)
    entries = protocol['populations']
    require(len(entries) == 8 and len({r['id'] for r in entries}) == 8,
            'Incomplete population allocation')
    bound(protocol['source_bundle']); bind_map_source(Path(protocol['source_bundle']))
    for entry in entries:
        directory = Path(entry['result'])
        s, receipt = bound(directory/'summary.json'), bound(entry['execution_receipt'])
        require(s['complete'] and s['passed'] and s['mode'] == 'geometry'
                and s['attempted_records'] == s['completed_records'] == s['denominator'] == s['new_poses_generated'] == 2048
                and s['clouds_begun'] == s['clouds_completed'] == s['raw_points'] == s['processed_points'] == 0,
                'Incomplete/nongeometry source population')
        require(receipt['success'] and receipt['child_drained'] and receipt['returncode'] == 0
                and Path(receipt['terminal']['path']).resolve() == (directory/'summary.json').resolve()
                and receipt['terminal']['sha256'] == sha(directory/'summary.json'), 'Population not drained')
    require(not args.out.exists(), 'Fresh audit output required')
    args.out.mkdir(); started = time.process_time()
    populations, panel, checks, decodes = [], [], [], []
    atlas_cache, geometry_cache, identities = {}, {}, set()
    with (args.out/'events.jsonl').open('x') as journal:
        def emit(row):
            journal.write(json.dumps(row, allow_nan=False)+'\n'); journal.flush()
        try:
            for entry in entries:
                emit(dict(kind='population_begun', id=entry['id']))
                cfg = bound(entry['config']); inputs = cfg['inputs']; directory = Path(entry['result'])
                producer, manifest = bound(directory/'protocol.json'), bound(directory/'chart-manifest.json')
                digest = sha(entry['config'])
                require(cfg['schema'] == 'context-source-guide-v1' and cfg['mode'] == 'geometry'
                        and cfg['draws'] == 2048 and cfg['mixture'] == [.5, .25, .25]
                        and producer['config'] == cfg and producer['config_sha256'] == digest
                        and producer['source_bundle_sha256'] == sha(protocol['source_bundle']), 'Changed producer law')
                for p, h in producer['input_sha256'].items():
                    require(bindings.get(str(Path(p).resolve())) == h, 'Missing direct physical/proposal pin')
                for name in ('invocation_config', 'model', 'prior', 'patch_map'):
                    asset = inputs[name]
                    require(bindings.get(str(Path(asset['path']).resolve())) == asset['sha256'], 'Changed input asset binding')
                arm, stream = validate_admitted_config(entry, cfg, digest, guides, identities)
                invocation_path = Path(inputs['invocation_config']['path'])
                original = bound(invocation_path)
                require(sha(invocation_path) == inputs['invocation_config']['sha256'], 'Invocation pin differs')
                def physical_asset(name):
                    path = Path(original[name]); path = path if path.is_absolute() else invocation_path.parent/path
                    require(bindings.get(str(path.resolve())) == original['expected_sha256'][name], 'Physical asset pin differs')
                    return bound(path)
                shape, context, source_state = (physical_asset(k) for k in ('shape', 'fixed_context', 'source_state'))
                require(original['depletant_radius'] == 1.5 and original['reservoir_density'] == .035
                        and original['poisson_lambda_ratio'] == 64., 'Changed physical conditions')
                anchor = source_state['anchor_pose']
                require(context['excluded_moving_labels'] == [77]
                        and [b['label'] for b in context['bodies']] == [i for i in range(264) if i != 77]
                        and anchor == next(b['pose'] for b in context['bodies'] if b['label'] == 16), 'Changed anchor/context')
                model = bound(inputs['model']['path']); prior = bound(inputs['prior']['path'])['log_prior']
                key = sha(inputs['model']['path'])
                if key not in atlas_cache: atlas_cache[key] = MapDensityReference(model)
                atlas = atlas_cache[key]
                source = SourceDensity(cfg['source_chart'], source_state['pose'], anchor)
                source_manifest = manifest['source_chart']; params = source_manifest['parameters']
                require(manifest['anchor_pose'] == anchor and manifest['mixture'] == cfg['mixture']
                        and manifest['angular_length'] == source.ell == atlas.density.ell
                        and source_manifest['angular_length'] == source.ell
                        and source_manifest['explicit_gaussian'] == cfg['source_chart']['explicit_gaussian']
                        and np.allclose(params['anchor_position'], source.center['position'], atol=2e-12, rtol=2e-14)
                        and np.allclose(params['anchor_rotation'], source.center_rotation, atol=2e-13, rtol=2e-14)
                        and params['mean'] == cfg['source_chart']['explicit_gaussian']['mean'] and params['weight'] == 1.
                        and np.allclose(params['covariance'], source.covariance, atol=0, rtol=2e-14)
                        and np.allclose(source_manifest['reconstructed_map_lower'], source.lower, atol=0, rtol=2e-14),
                        'Source chart differs from declared guide')
                require(len(manifest['charts']) == len(atlas.branches) == 2048, 'Atlas inventory differs')
                for index, (chart, branch) in enumerate(zip(manifest['charts'], atlas.branches)):
                    require(chart['virtual_label'] == index and chart['component_index'] == branch['component_index']
                            and chart['inverted'] is branch['inverted']
                            and np.allclose(chart['reconstructed_map_lower'], atlas.density.lower[index], atol=2e-14, rtol=2e-14),
                            'Atlas map parameters differ')
                    checks.append(close_log(chart['effective_log_prior'], prior[index], 'context prior'))
                patch_map = bound(inputs['patch_map']['path'])
                key = tuple(original['expected_sha256'][k] for k in ('shape', 'fixed_context', 'source_state'))
                if key not in geometry_cache:
                    geometry_cache[key] = GeometryReference(shape, patch_map['atom_patch_ids'], context,
                        source_state['spherical_wall_radius'], 1.5)
                geometry = geometry_cache[key]; half = manifest['uniform_half_width']
                checks.append(close_log(half, source_state['spherical_wall_radius']+geometry.bound, 'uniform halfwidth'))
                regions = inputs['regions']; reference = regions['source_secondary_tokens']
                require(regions['a_neighbors'] == [16, 217] and regions['b_neighbors'] == [16, 56]
                        and regions['secondary_label'] == 217 and len(canonical_tokens(reference)) == 16
                        and regions['inclusion_boundaries'] == [0., .25, .5, .75, 1.], 'Changed regions')
                require(str((directory/'rows.jsonl').resolve()) in bindings, 'Unbound candidate rows')
                rows = [json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
                require(len(rows) == 2048, 'Incomplete candidate denominator')
                for ordinal, row in enumerate(rows):
                    checks.extend(scalar_row(row, ordinal, half, reference, source, cfg['mixture']))
                    decodes.append(audit_generation(row['input'], cfg, digest, ordinal, half, source, atlas, anchor))
                    require(row['clouds'] == [] and row['log_physical_contribution'] is None
                            and row['physical_weight_status'] == 'not_estimated', 'Unallocated physical weight')
                    hard = row['log_hard_only_contribution']
                    if row['actual']['physical_valid']: checks.append(close_log(hard, -row['density']['log_q'], 'hard contribution'))
                    else: require(hard is None, 'Hard-invalid contribution differs')
                require(str((directory/'events.jsonl').resolve()) in bindings, 'Unbound journal')
                count = audit_events(directory/'events.jsonl', rows, reference)
                summary = bound(directory/'summary.json'); aggregate = coverage_summary(rows)
                require(summary['identity'] == cfg['identity'] and summary['unconditional_denominator'] is True,
                        'Wrong summary identity/denominator convention')
                require(count == summary['journal_events'] and aggregate['hard_valid'] == summary['physical_valid_records'], 'Summary count differs')
                require(len(summary['regions']) == len(aggregate['partitions']) == 9, 'Incomplete mass partition')
                checks.append(close_log(summary['expected_complete_allocation_raw_points'], aggregate['expected_complete_allocation_raw_points'], 'prospective cost'))
                for group, reported in zip(sorted(aggregate['partitions'], key=lambda r:r['region']), summary['regions']):
                    require(group['region'] == reported['region'] and group['attempted_count'] == reported['count']
                            and reported['log_physical_mass'] is None, 'Wrong unconditional mass partition')
                    if group['attempted_count'] and group['region'] != 'hard_invalid':
                        checks.append(close_log(reported['log_hard_only_mass'], group['log_hard_only_importance_mass'], 'hard-only region mass'))
                    else: require(reported['log_hard_only_mass'] is None, 'Spurious region mass')
                populations.append(dict(id=entry['id'], arm=arm, stream=stream, **aggregate))
                emit(dict(kind='scalar_population_complete', id=entry['id']))
                selected = [rows[i] for i in PANEL_ORDINALS]
                logs = atlas.evaluate([r['input']['proposed_pose'] for r in selected], anchor, prior)[0]
                for row, g in zip(selected, logs):
                    ordinal = row['input']['ordinal']
                    emit(dict(kind='panel_pose_begun', id=entry['id'], ordinal=ordinal))
                    numeric = close_log(finite_log(row['density']['log_g'], row['density']['log_g_status']), float(g), 'panel atlas density')
                    physical = geometry.classify(row['input']['proposed_pose'])
                    require(physical['wall_valid'] is row['actual']['wall_valid']
                            and physical['core_valid'] is row['actual']['core_valid'], 'Independent hard verdict differs')
                    if physical['core_valid'] is True:
                        require([list(t) for t in physical['patch_tokens']] == row['patches']['tokens'], 'Independent patch inventory differs')
                    result = dict(id=entry['id'], ordinal=ordinal, numeric=numeric, geometry=physical)
                    panel.append(result); emit(dict(kind='panel_pose_complete', **result))
            require(len(populations) == 8 and len(panel) == 512 and len(decodes) == 16384, 'Incomplete audit')
            for path, digest in bindings.items(): require(sha(path) == digest, 'Input changed during audit')
            report = dict(schema='context-covariance-guide-independent-audit-v1', complete=True, passed=True,
                protocol_sha256=sha(args.protocol), source_sha256=sha(__file__), input_sha256=bindings,
                allocation_sha256=args.expected_allocation_sha256, guides=protocol['guides'],
                reduction=protocol['reduction'], unconditional_denominator=True,
                scalar_checks=len(checks), maximum_scalar_check_ratio=max(r['ratio'] for r in checks),
                decoded_candidates=len(decodes), maximum_position_error=max(r['position_error'] for r in decodes),
                maximum_rotation_error=max(r['rotation_matrix_error'] for r in decodes),
                independent_panel_size=len(panel), panel=panel, populations=populations,
                physical_weight_status='not_estimated', clouds_generated=0, new_poses_generated=0,
                cpu_seconds=time.process_time()-started,
                scope='Prospective frozen full-versus-diagonal source-informed proposal coverage only; training streams0,1 and descriptive heldout2,3 are not new validation populations. No physical-weight or assembly conclusion.')
            (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        except BaseException as error:
            emit(dict(kind='fatal', error=repr(error), completed_populations=len(populations), completed_panel_poses=len(panel)))
            (args.out/'failure.json').write_text(json.dumps(dict(complete=False, passed=False,
                error=repr(error), prefix_preserved=True), indent=2)+'\n')
            raise


if __name__ == '__main__':
    main()
