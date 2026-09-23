#!/usr/bin/env python3
"""Independent schema-4 vessel audit with a separately bound reporting chart.

The frozen regional guide contributes labels only. The denominator remains the
complete original vessel law, with no outer mixture and no extra Jacobian.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.special import logsumexp

from audit_full_vessel_latent import (VesselDensity, PrunedExclusionContact,
    check_generation_metadata, check_cloud_envelopes_and_counts, log_close)
from analyze_basin_normalizers import audit_pose_proposal, audit_wall_domain, moments, paired_noise
from analyze_mobile_native_pocket import load_classifier, local_sources
from analyze_r4_smc_control import Ledger, close, nullable_log, read, require, sha, write
from physical_latent_guide import PhysicalLatentGuide
from prepare_deep_far_normalizer_atlas import registration


def check_rows(config, manifest, rows, vessel, reporting):
    """Reconstruct original q and cloud arithmetic for every unconditional row."""
    require(manifest['schema'] == 4 and 'outer_mixture_schema' not in manifest,
            'Unmodified schema-4 baseline required')
    require(rows and [r['draw'] for r in rows] == list(range(manifest['samples'])),
            'Missing attempted draw')
    require(all(r.get('pose') is not None for r in rows), 'Censored numerical null')
    require(all('outer_branch' not in r and 'latent_proposal' not in r for r in rows),
            'Baseline has outer-mixture generation metadata')
    poses = [r['pose'] for r in rows]
    logs, geometry = vessel.evaluate(poses)
    require(np.isfinite(logs).all(), 'Generated pose lacks vessel support')
    coordinates = reporting.evaluate_many(poses)
    metric = registration(poses, config['metadata'])
    errors = []
    for i, row in enumerate(rows):
        require(row.get('proposal') is not None, 'Missing vessel generation metadata')
        errors.append(log_close(row['log_proposal_density'], logs[i], 'Full original vessel density differs'))
        require(type(row['capture_valid']) is bool and row['capture_valid'] == bool(geometry['capture'][i]),
                'Capture predicate differs')
        require(type(row['hard_valid']) is bool and type(row['wall_valid']) is bool,
                'Non-Boolean hard/wall flag')
        if row['hard_valid']:
            require(row['capture_valid'] and row['wall_valid'], 'Hard-valid row leaves physical domain')
            close(row['log_hard_weight'], -float(logs[i]),
                  'Baseline hard weight must use original vessel density, no mixture or extra J')
            close(row['q'], float(metric[i]), 'Original physical registration metric differs')
            require(len(row['clouds']) == manifest['cloud_replicates'] == 2, 'Two independent clouds required')
            for cloud in row['clouds']:
                require(type(cloud['overlap_points']) is int and cloud['overlap_points'] >= 0
                    and math.isfinite(cloud['lower_volume']) and cloud['lower_volume'] >= 0
                    and math.isfinite(cloud['uncertain_volume']) and cloud['uncertain_volume'] >= 0,
                    'Invalid cloud count/volume')
                z, lam = manifest['activity'], manifest['lambda']
                factor = z * cloud['lower_volume'] + cloud['overlap_points'] * math.log1p(z / lam) if z else 0.
                close(cloud['log_weight'], factor, 'Poisson estimator count identity differs')
            expected = float(logsumexp([c['log_weight'] for c in row['clouds']]) - math.log(2) - logs[i])
            close(row['log_importance_weight'], expected, 'Arithmetic cloud-mean weight differs')
        else:
            require(all(row[k] is None for k in ('log_hard_weight', 'log_importance_weight', 'q', 'region', 'depletion_contact'))
                    and not row['clouds'], 'Invalid attempted draw lost its explicit zero')
    return dict(checked_attempts=len(rows), maximum_log_density_error=max(errors),
        valid_outside_R4=sum(r['hard_valid'] and not c.in_reference_ball for r, c in zip(rows, coordinates)),
        scope='Original vessel density only on every attempted world pose; regional coordinates are reporting labels.'), coordinates


def estimate_labels(rows, labels):
    """All attempted rows enter each class denominator, including invalid zeros."""
    require(len(rows) == len(labels) and rows, 'Missing classification rows')
    estimates = {}
    for name in labels[0]['classes']:
        selected = [label['classes'][name] for label in labels]
        logs = np.asarray([r['log_importance_weight'] if s else -np.inf for r, s in zip(rows, selected)])
        hard = np.asarray([r['log_hard_weight'] if s else -np.inf for r, s in zip(rows, selected)])
        pairs = np.asarray([[c['log_weight'] - r['log_proposal_density'] for c in r['clouds']]
                            if s else [-np.inf, -np.inf] for r, s in zip(rows, selected)])
        estimates[name] = dict(Qz=moments(logs), Q0=moments(hard), paired_noise=paired_noise(logs, pairs))
    return estimates


def audit(root, out, region_path, guide_path, native_definition=None):
    require(sys.flags.optimize == 0, 'Independent legacy checks require Python assertions')
    root, out = Path(root).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh audit output required')
    ledger = Ledger(); sources = local_sources(__file__)
    for path in sources.values(): ledger.bind(path)
    manifest = read(ledger.bind(root / 'manifest.json'))
    summary = read(ledger.bind(root / 'summary.json'))
    require(manifest['schema'] == 4 and 'outer_mixture_schema' not in manifest, 'Schema-4 baseline required')
    require(manifest['bath_wall_permeable'] is True and 'atomic_wall' in manifest, 'Full atomic-wall baseline required')
    require(summary['complete'] and summary['manifest'] == manifest and summary['numerical_nulls'] == 0,
            'Incomplete population')
    raw = ledger.bind(root / 'samples.jsonl')
    for name, key in [('input-config.json', 'config_sha256'), ('model.json', 'model_sha256'),
                      ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        ledger.bind(root / 'provenance' / name, manifest[key])
    config = read(ledger.bind(root / 'config.json'))
    original = read(root / 'provenance/input-config.json')
    for key in ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'metadata'):
        require(config[key] == original[key], 'Physical config differs: ' + key)
    require(config['reservoir_density'] == manifest['activity'], 'Activity differs')
    require(math.isfinite(manifest['activity']) and manifest['activity'] >= 0
            and math.isfinite(manifest['lambda']) and manifest['lambda'] > 0, 'Invalid bath intensity')
    close(manifest['lambda'], config['poisson_lambda_ratio'] * manifest['activity'] if manifest['activity'] > 0 else 1.,
          'Bath cloud intensity differs')
    require(config.get('target_region') is None, 'Unsupported hidden target restriction')
    region_path, guide_path = ledger.bind(region_path), ledger.bind(guide_path)
    reporting = PhysicalLatentGuide.from_files(region_path, guide_path, expected_shape_sha256=manifest['shape_sha256'])
    require(reporting.physical_fixed_neighbors == config['fixed_poses'], 'Reporting/vessel scaffold differs')
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    vessel = VesselDensity(config, manifest, read(root / 'provenance/model.json'), root / 'provenance/source-bundle.json')
    density_check, coordinates = check_rows(config, manifest, rows, vessel, reporting)
    # This view labels every baseline row as vessel-generated for the shared
    # metadata helper. It changes no density, weight, manifest, or source row.
    generation_view = [dict(row, outer_branch='vessel') for row in rows]
    generation_metadata = check_generation_metadata(config, manifest, generation_view, vessel)
    counts = check_cloud_envelopes_and_counts(manifest, summary, rows)
    generation = audit_pose_proposal(root, dict(proposal_anchor_index=manifest['proposal_anchor_index']), manifest, rows)
    wall = audit_wall_domain(root, manifest, rows, summary)
    contact = PrunedExclusionContact(read(root / 'provenance/shape.json'), config['fixed_poses'], config['depletant_radius'])
    classifier = None; native_binding = None
    if native_definition:
        classifier, native_binding = load_classifier(native_definition)
        definition = classifier.definition
        require(definition['shape_sha256'] == manifest['shape_sha256'] and definition['fixed_poses'] == config['fixed_poses'],
                'Native geometry identity differs')
        native_path = ledger.bind(native_definition)
        for name, digest in definition['input_sha256'].items(): ledger.bind(native_path.parent / 'inputs' / name, digest)
        native_config = read(native_path.parent / 'inputs/physical-config.json')
        for key in ('fixed_poses', 'depletant_radius', 'reservoir_density', 'metadata'):
            require(config[key] == native_config[key], 'Native observer physical law differs: ' + key)
    labels = []; near_boundary = 0
    for row, coordinate in zip(rows, coordinates):
        geometry = contact.classify(row['pose'], capture_valid=row['capture_valid'], wall_valid=row['wall_valid'])
        near = geometry['near_core_boundary']; near_boundary += int(near)
        hard = row['capture_valid'] and row['wall_valid'] and geometry['core_disjoint']
        require(row['hard_valid'] == hard or near, 'Independent atom hard predicate differs away from roundoff boundary')
        native_label = classifier.classify(row['pose']) if classifier and row['hard_valid'] else None
        if row['hard_valid']:
            require(row['depletion_contact'] == geometry['exclusion_contact'], 'Independent exclusion contact differs')
        native = bool(native_label['native_any']) if native_label else False
        valid, inside = row['hard_valid'], coordinate.in_reference_ball
        classes = dict(total=valid, inside_R4=valid and inside, outside_R4=valid and not inside,
                       exclusion_contact=valid and geometry['exclusion_contact'], unbound=valid and not geometry['exclusion_contact'])
        if classifier:
            classes.update(registered_native_entry=valid and native,
                contact_no_native_entry=valid and geometry['exclusion_contact'] and not native,
                unbound_no_native_entry=valid and not geometry['exclusion_contact'] and not native)
            require(sum(classes[k] for k in ('registered_native_entry', 'contact_no_native_entry', 'unbound_no_native_entry')) == int(valid),
                    'Complete class partition fails')
        labels.append(dict(draw=row['draw'], classes=classes, geometry=geometry, native=native_label, near_core_boundary=near))
    estimates = estimate_labels(rows, labels)
    log_close(summary['estimates']['total']['log_normalizer'], nullable_log(estimates['total']['Qz']['logQ']), 'Total estimator differs')
    log_close(summary['estimates']['hard_total']['log_normalizer'], nullable_log(estimates['total']['Q0']['logQ']), 'Hard estimator differs')
    ledger.recheck(); out.mkdir(parents=True); provenance = out / 'provenance'; provenance.mkdir()
    for name, path in sources.items():
        shutil.copy2(path, provenance / name)
        require(sha(provenance / name) == ledger.files[str(Path(path).resolve())], 'Audit source changed while archiving')
    for name, path in [('reporting-region.json', region_path), ('reporting-guide.json', guide_path)]:
        shutil.copy2(path, provenance / name)
        require(sha(provenance / name) == ledger.files[str(path)], 'Reporting input changed while archiving')
    ledger.recheck()
    with (out / 'labels.jsonl').open('w') as stream:
        for row in labels: stream.write(json.dumps(row, allow_nan=False) + '\n')
    result = dict(schema='full-vessel-baseline-audit-v1', complete=True, population=str(root), manifest=manifest,
        density_audit=density_check, vessel_generation_audit=generation,
        all_vessel_generation_metadata_audit=generation_metadata, primitive_count_audit=counts, wall_audit=wall,
        reporting_guide_binding=dict(region_sha256=sha(region_path), guide_sha256=sha(guide_path),
            region=str(region_path), guide=str(guide_path), restricts_target=False, affects_proposal=False,
            scope='Separately bound auditor inputs; the schema-4 sampler manifest contains neither reporting chart nor guide.',
            R4_label_scope='inside_R4/outside_R4 use the chart ball only, for schema-5 audit compatibility. Source capture is not part of these labels; exact measured-region intersections require the separately bound reporting partition.'),
        raw_sample_binding=dict(sha256=ledger.files[str(raw)], external_expected_sha256=None,
            scope='Actual bytes bound at audit start and rechecked afterward; campaign separately freezes execution inputs.'),
        executable_binding=dict(manifest_sha256=manifest['executable_sha256'], artifact_verified=False,
            scope='Campaign must verify the executable artifact; this auditor binds its archived source bundle.'),
        independently_checked_atom_poses=contact.counts['exact_core_contact_pose_checks'], geometry_audit=contact.report(),
        near_core_boundary_poses=near_boundary, native_binding=native_binding, estimates=estimates, source_sha256=ledger.files,
        scope='One population audit. Original vessel q is unchanged; R4 is reporting only. Unconditional zeros remain. '
              'Count algebra does not prove exact thinning, geometry arithmetic, independent RNG streams, or convergence.')
    write(out / 'analysis.json', result)
    write(out / 'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--region', type=Path, required=True)
    parser.add_argument('--guide', type=Path, required=True)
    parser.add_argument('--native-definition', type=Path)
    args = parser.parse_args()
    result = audit(args.population, args.out, args.region, args.guide, args.native_definition)
    print(json.dumps(dict(complete=result['complete'], density_audit=result['density_audit'])), flush=True)


if __name__ == '__main__': main()
