"""Independent audit of complete source-guide geometry populations.

The panel is fixed before generation; no pose, failed draw or population can be
substituted. This module does not estimate depletion weights in geometry mode.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from audit_context_candidate_bank import (
    GeometryReference, MapDensityReference, bind_map_source, canonical_tokens,
    close_log, coverage_summary, expected_region, finite_log, region_from_tokens,
    require,
)
from context_prior_metrics import fingerprint
from source_guide_reference import SourceDensity, atlas_decode, mixture_log_density, pose_error

PANEL_ORDINALS = tuple(range(0, 512, 16))


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scalar_row(row, ordinal, half, source217, source, mixture):
    require(row['complete'] is True and row['input']['ordinal'] == ordinal,
            'Missing/reordered generated candidate')
    item, d, actual = row['input'], row['density'], row['actual']
    require(item['branch'] in ('uniform', 'context', 'source'), 'Unknown generated branch')
    value = item['proposed_pose']
    g = finite_log(d['log_g'], d['log_g_status'])
    s = finite_log(d['log_source'], d['log_source_status'])
    inside = all(abs(float(x)) <= half for x in value['position'])
    u = -3*math.log(2*half) if inside else -math.inf
    require((d['log_u'] is None) is (not inside), 'Wrong nullable uniform density')
    checks = [close_log(float(d['log_u']) if inside else -math.inf, u, 'uniform'),
              close_log(s, source.evaluate(value), 'all-row source density'),
              close_log(d['log_q'], mixture_log_density(u, g, s, mixture), 'all-row complete Q')]
    require(d['uniform_contains'] is inside, 'Uniform-cube indicator differs')
    require(type(actual['wall_valid']) is bool and
            ((actual['wall_valid'] and type(actual['core_valid']) is bool)
             or (not actual['wall_valid'] and actual['core_valid'] is None)), 'Wrong hard-test status')
    valid = bool(actual['wall_valid'] and actual['core_valid'])
    require(actual['physical_valid'] is valid and row['physical_zero'] is (not valid),
            'Wrong physical zero')
    if not valid:
        require(row['patches'] is None and row['region'] is None and row['envelope'] is None,
                'Invalid pose has a physical region/envelope')
        return checks
    patches = row['patches']
    details = region_from_tokens(patches['tokens'], source217)
    require(row['region'] == expected_region(details), 'Wrong exhaustive region')
    require(patches['neighbor_labels'] == details['neighbors']
            and patches['source_intersection'] == details['source217_intersection']
            and patches['source_union'] == details['source217_union']
            and patches['source_fraction'] == details['source217_inclusion']
            and patches['jaccard'] == details['source217_jaccard']
            and patches['fingerprint'] == fingerprint(patches['tokens']), 'Wrong patch summary')
    canonical_tokens(patches['tokens'])
    e = row['envelope']
    lo, uncertain, hi = (float(e[k]) for k in ('lower_volume', 'uncertain_volume', 'upper_volume'))
    require(all(math.isfinite(v) and v >= 0 for v in (lo, uncertain, hi)), 'Invalid overlap bounds')
    checks.append(close_log(hi, lo+uncertain, 'overlap-bound sum'))
    require(set(details['neighbors']).issubset(e['fixed_labels']), 'Contact omitted by pruning')
    return checks


def role_seed(config, digest, ordinal, role):
    h = hashlib.sha256(b'context-source-guide-independent-v1\0')
    h.update(digest.encode())
    for number in (config['seed'], config['population'], ordinal):
        h.update(number.to_bytes(8, 'little'))
    h.update(role.encode())
    return h.hexdigest()


def audit_generation(item, config, digest, ordinal, half, source, atlas, anchor):
    require(item['identity'] == config['identity'] and item['population'] == config['population'],
            'Generated candidate identity differs')
    require(item['role_seeds'] == {role: role_seed(config, digest, ordinal, role)
            for role in ('component', 'label', 'latent', 'uniform', 'cloud0', 'cloud1')},
            'RNG role identity differs')
    u = item['component_uniform']
    require(type(u) in (float, int) and 0 <= u < 1, 'Invalid component coin')
    branch = 'uniform' if u < .5 else 'context' if u < .75 else 'source'
    require(item['branch'] == branch, 'Component law differs from fixed mixture')
    if branch == 'uniform':
        require(item['latent'] is None and item['selected_virtual_label'] is None,
                'Uniform branch has learned latent')
        values = np.asarray(item['cube_uniforms'], float)
        normals = np.asarray(item['quaternion_normals'], float)
        require(values.shape == (3,) and ((0 <= values) & (values < 1)).all()
                and normals.shape == (4,) and np.isfinite(normals).all()
                and np.linalg.norm(normals) > 0, 'Invalid recorded uniform draw')
        expected = dict(position=((2*values-1)*half).tolist(),
                        orientation=(normals/np.linalg.norm(normals)).tolist())
    else:
        require(item['cube_uniforms'] is None and item['quaternion_normals'] is None,
                'Learned branch has uniform draw')
        if branch == 'source':
            require(item['selected_virtual_label'] is None, 'Source branch chose an atlas label')
            expected = source.decode(item['latent'])
        else:
            expected = atlas_decode(atlas, anchor, item['selected_virtual_label'], item['latent'])
    return pose_error(item['proposed_pose'], expected)


def audit_events(path, rows, reference):
    events = [json.loads(line) for line in Path(path).read_text().splitlines()]
    require([r['event_index'] for r in events] == list(range(len(events))), 'Missing/reordered events')
    cursor = 0
    def take(kind):
        nonlocal cursor
        require(cursor < len(events) and events[cursor]['kind'] == kind, 'Wrong event stage '+kind)
        row = events[cursor]; cursor += 1
        return row
    take('setup_begun'); take('source_begun')
    s = take('source_complete')['patches']
    require(canonical_tokens(s['secondary_tokens']) == canonical_tokens(reference)
            and s['neighbor_labels'] == [16, 217] and s['source_fraction'] == 1,
            'Source reference changed')
    for ordinal, row in enumerate(rows):
        require(take('candidate_begun')['ordinal'] == ordinal, 'Wrong begun ordinal')
        e = take('candidate_generated')
        require(e['ordinal'] == ordinal and e['input'] == row['input'], 'Generated pose modified')
        e = take('density_complete')
        require(e['ordinal'] == ordinal and e['density'] == row['density'], 'Density modified')
        e = take('geometry_complete')
        require(e['ordinal'] == ordinal and all(e[k] == row['actual'][k]
                for k in ('wall_valid', 'core_valid')), 'Geometry modified')
        if row['actual']['physical_valid']:
            e = take('patches_complete')
            require(e['ordinal'] == ordinal and e['patches'] == row['patches']
                    and e['region'] == row['region'], 'Patches modified')
            e = take('envelope_complete')
            require(e['ordinal'] == ordinal and e['envelope'] == row['envelope'], 'Envelope modified')
        require(take('candidate_complete')['ordinal'] == ordinal, 'Wrong complete ordinal')
    require(cursor == len(events), 'Additional events or clouds in geometry allocation')
    return cursor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    require(protocol['schema'] == 'context-source-guide-audit-v1'
            and protocol['panel_ordinals'] == list(PANEL_ORDINALS), 'Changed audit allocation')
    bindings = protocol['input_sha256']
    for path, digest in bindings.items(): require(sha(path) == digest, 'Changed input: '+path)
    def bound(path):
        key = str(Path(path).resolve())
        require(key in bindings, 'Unbound audit input: '+key)
        return read(key)
    entries = protocol['populations']
    require(len(entries) == 16 and len({r['id'] for r in entries}) == 16,
            'Incomplete population allocation')
    require(protocol['draws_per_population'] == 512 and protocol['expected_populations'] == 16,
            'Changed draw allocation')
    bound(protocol['source_bundle']); bind_map_source(Path(protocol['source_bundle']))
    for entry in entries:
        directory = Path(entry['result'])
        s, receipt = bound(directory/'summary.json'), bound(entry['execution_receipt'])
        require(s['complete'] and s['passed'] and s['mode'] == 'geometry'
                and s['attempted_records'] == s['completed_records'] == s['denominator'] == s['new_poses_generated'] == 512
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
                        and cfg['draws'] == 512 and cfg['mixture'] == [.5, .25, .25]
                        and producer['config'] == cfg and producer['config_sha256'] == digest
                        and producer['source_bundle_sha256'] == sha(protocol['source_bundle']), 'Changed producer law')
                for p, h in producer['input_sha256'].items():
                    require(bindings.get(str(Path(p).resolve())) == h, 'Missing direct physical/proposal pin')
                for name in ('invocation_config', 'model', 'prior', 'patch_map'):
                    asset = inputs[name]
                    require(bindings.get(str(Path(asset['path']).resolve())) == asset['sha256'], 'Changed input asset binding')
                width, stream = (cfg['identity'][k] for k in ('width_index', 'stream'))
                require((width, stream) not in identities and width in range(4) and stream in range(4)
                        and cfg['population'] == 4*width+stream, 'Wrong population identity')
                identities.add((width, stream))
                st, degrees = ((.1, .5), (.2, 1.), (.4, 2.), (.8, 4.))[width]
                require(cfg['source_chart']['translation_sigma'] == st
                        and cfg['source_chart']['rotation_scale_deg'] == degrees, 'Changed source widths')
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
                        and np.allclose(params['anchor_position'], source.center['position'], atol=2e-12, rtol=2e-14)
                        and np.allclose(params['anchor_rotation'], source.center_rotation, atol=2e-13, rtol=2e-14)
                        and params['mean'] == [0.]*6 and params['weight'] == 1.
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
                require(len(rows) == 512, 'Incomplete candidate denominator')
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
                populations.append(dict(id=entry['id'], width_index=width, stream=stream, **aggregate))
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
            require(len(populations) == 16 and len(panel) == 512 and len(decodes) == 8192, 'Incomplete audit')
            for path, digest in bindings.items(): require(sha(path) == digest, 'Input changed during audit')
            report = dict(schema='context-source-guide-independent-audit-v1', complete=True, passed=True,
                protocol_sha256=sha(args.protocol), source_sha256=sha(__file__), input_sha256=bindings,
                scalar_checks=len(checks), maximum_scalar_check_ratio=max(r['ratio'] for r in checks),
                decoded_candidates=len(decodes), maximum_position_error=max(r['position_error'] for r in decodes),
                maximum_rotation_error=max(r['rotation_matrix_error'] for r in decodes),
                independent_panel_size=len(panel), panel=panel, populations=populations,
                physical_weight_status='not_estimated', clouds_generated=0, new_poses_generated=0,
                cpu_seconds=time.process_time()-started,
                scope='Source-informed proposal coverage only, no physical-weight or assembly conclusion.')
            (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        except BaseException as error:
            emit(dict(kind='fatal', error=repr(error), completed_populations=len(populations), completed_panel_poses=len(panel)))
            (args.out/'failure.json').write_text(json.dumps(dict(complete=False, passed=False,
                error=repr(error), prefix_preserved=True), indent=2)+'\n')
            raise


if __name__ == '__main__':
    main()
