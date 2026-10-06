#!/usr/bin/env python3
"""Bounded, independent schema-7/8 native-class full-vessel audit.

Reuses the validated complete class/axis/component density reconstruction.
Every attempted row and journal entry is visited in order, including invalid
zeros. Saves atomic geometry for a separate native partition; this is not an
assembly inference or dispatch/admission authority.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import time

from analyze_mobile_native_pocket import local_sources
from analyze_r4_smc_control import Ledger, close, read, require, sha, write
from audit_hard_free_vessel_streaming import (
    CLASSES, WallOracle, attempt_batches, memberships, row_counts,
)
import physical_native_class_line_vessel as reference
from streaming_weight_moments import RegionMoments

SCHEMA = 'full-vessel-native-class-line-streaming-audit-v1'


def validate_manifest(manifest, summary):
    require(type(manifest['schema']) is int and manifest['schema'] in (7, 8)
            and manifest['outer_mixture_schema'] == reference.SCHEMA
            and manifest['latent_guide_schema'] == reference.line.SCHEMA
            and manifest['outer_vessel_probability'] == .5, 'Wrong native-class vessel law')
    if manifest['schema'] == 8:
        require(manifest.get('pre_envelope_schema') == 7
                and manifest.get('vessel_uniform_schema') == 'one-atom-wall-envelope-v1'
                and isinstance(manifest.get('vessel_uniform_envelope'), dict),
                'Schema 8 needs the class wall-envelope contract')
    else:
        require(not any(k in manifest for k in ('pre_envelope_schema', 'vessel_uniform_schema',
                                               'vessel_uniform_envelope')),
                'Schema 7 cannot conceal a wall-envelope proposal')
    require(summary['complete'] is True and summary['manifest'] == manifest
            and summary['numerical_nulls'] == 0 and summary['samples'] == manifest['samples'],
            'Incomplete or changed population')
    require(type(manifest['samples']) is int and manifest['samples'] > 0
            and manifest['cloud_replicates'] == 2, 'Positive all-attempt allocation and two clouds required')
    require(manifest['attempt_journal'] == 'attempts.jsonl; begin before each attempt; no retries'
            and manifest['resume_supported'] is False, 'Missing once-only attempt contract')
    require(manifest['density_measure'] == 'Lebesgue center volume times normalized SO(3) Haar measure'
            and manifest['latent_reference_ball_is_target_restriction'] is False
            and manifest['latent_source_capture']['restricts_target'] is False
            and manifest['latent_source_capture']['conditions_guide'] is True
            and manifest['bath_wall_permeable'] is True, 'Changed physical measure/domain')


class BatchChecks:
    """Only temporary row indices/N change; complete mixture checks are reused."""
    sums = ('checked_attempts', 'valid_outside_R4', 'structural_zero_queries',
            'exact_chart_seams', 'valid_outside_source_capture')
    maxima = ('maximum_log_density_error', 'maximum_interval_endpoint_error', 'maximum_inverse_CDF_error')
    fixed = ('source_capture', 'vessel_capture', 'chart_factor_validation', 'scope')

    def __init__(self):
        self.density = None
        self.generation_count = self.batches = self.peak_rows = 0
        self.primitive = Counter()

    def check(self, config, manifest, rows, vessel, guide):
        require(rows, 'Empty audit batch')
        adapted = [dict(row, draw=i) for i, row in enumerate(rows)]
        local = dict(manifest, samples=len(rows))
        density = reference.check_rows(config, local, adapted, vessel, guide)
        require(set(density) == set(self.sums+self.maxima+self.fixed+('outer_branches',)),
                'Unhandled native-class density audit field')
        if self.density is None:
            self.density = copy.deepcopy(density)
        else:
            require(all(self.density[k] == density[k] for k in self.fixed), 'Batch context changed')
            for key in self.sums:
                self.density[key] += density[key]
            for key in self.maxima:
                self.density[key] = max(self.density[key], density[key])
            self.density['outer_branches'] = dict(Counter(self.density['outer_branches'])
                                                  + Counter(density['outer_branches']))
        generation = reference.vessel_reference.check_generation_metadata(config, local, adapted, vessel)
        require(set(generation) == ({'checked_vessel_generation_rows', 'scope'}
                if generation['checked_vessel_generation_rows'] else {'checked_vessel_generation_rows'}),
                'Unhandled generation audit field')
        if generation['checked_vessel_generation_rows']:
            require(generation['scope'] == 'Generation metadata and selected-anchor densities; RNG replay is a separate implementation obligation.',
                    'Changed vessel-generation contract')
        self.generation_count += generation['checked_vessel_generation_rows']
        counts = reference.vessel_reference.check_cloud_envelopes_and_counts(
            local, dict(samples=len(rows), **row_counts(rows)), adapted)
        require(set(counts) == set(row_counts(rows)), 'Unhandled primitive counter')
        self.primitive.update(counts)
        self.batches += 1
        self.peak_rows = max(self.peak_rows, len(rows))

    def finish(self, manifest, summary):
        require(self.density is not None and self.density['checked_attempts']
                == manifest['samples'] == summary['samples'], 'Incomplete unconditional denominator')
        require(all(summary[key] == value for key, value in self.primitive.items()),
                'Global rejection/point counters differ')
        require(sum(self.density['outer_branches'].values()) == manifest['samples'],
                'Outer branch accounting lost attempts')
        require(self.generation_count == self.density['outer_branches'].get('vessel', 0),
                'Lost vessel generation checks')


def context(root, manifest, ledger, definition_path, synthetic):
    """Authenticate both compiled native bytes and the physical-shape witness."""
    config = read(ledger.bind(root/'config.json'))
    original = read(root/'provenance/input-config.json')
    for key in ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'metadata'):
        require(config[key] == original[key], 'Physical config changed: '+key)
    require(config['reservoir_density'] == manifest['activity'] and config.get('target_region') is None,
            'Changed bath or hidden target')
    require(math.isfinite(manifest['activity']) and manifest['activity'] >= 0
            and math.isfinite(manifest['lambda']) and manifest['lambda'] > 0, 'Invalid bath/cloud intensity')
    close(manifest['lambda'], config['poisson_lambda_ratio']*manifest['activity']
          if manifest['activity'] > 0 else 1., 'Auxiliary intensity differs')
    shape = read(root/'provenance/shape.json')
    provenance = manifest['compiled_native']
    compiled_path = ledger.bind(root/'provenance/compiled-native.json', provenance['compiled_sha256'])
    compiled = read(compiled_path)
    require(compiled['source_definition_sha256'] == provenance['source_definition_sha256']
            and compiled['source_input_sha256'] == provenance['source_input_sha256'],
            'Native provenance differs')
    witness = reference.regional.validate_shape_witness(compiled, shape, provenance['shape_compatibility'],
        provenance['compiled_sha256'], manifest['shape_sha256'])
    if synthetic:
        require(definition_path is None and len(shape['atoms']) <= 64 and len(compiled['monomer_atoms']) <= 16,
                'Synthetic compiled adapter cannot audit proteins')
        observer = reference.line.observer_from_compiled_for_synthetic(compiled)
    else:
        require(definition_path is not None, 'Original frozen native definition required')
        observer = reference.NativeContactRegions(ledger.bind(definition_path, provenance['source_definition_sha256']))
        reference.line.compare_compiled_definition(compiled, observer)
        for name, digest in observer.definition['input_sha256'].items():
            ledger.bind(observer.root/name, digest)
    guide = reference.PhysicalNativeClassLineGuide.from_files(root/'provenance/latent-region.json',
        root/'provenance/latent-guide.json', vessel_config=config, shape=shape,
        expected_shape_sha256=manifest['shape_sha256'], compiled_path=compiled_path, observer=observer)
    require(manifest['latent_defensive_uniform_probability'] == guide.guide.alpha
            and manifest['latent_gaussian_component_count'] == guide.guide.count, 'Changed guide mixture')
    source = manifest['latent_source_capture']
    require(source['center'] == guide.region['capture_center'] and source['radius'] == guide.region['capture_radius'],
            'Changed source conditioning capture')
    bundle = root/'provenance/source-bundle.json'
    vessel = reference.vessel_reference.VesselDensity(config, manifest, read(root/'provenance/model.json'), bundle, shape=shape)
    require((manifest['schema'] == 8) == (vessel.envelope is not None), 'Unvalidated wall-envelope law')
    return config, shape, guide, vessel, witness


def audit(directory, out, binary, batch_size=64, *, definition_path=None, synthetic=False):
    require(sys.flags.optimize == 0, 'Independent checks require unoptimized Python')
    require(type(batch_size) is int and 1 <= batch_size <= 1024, 'Batch size must be 1..1024')
    root, out, binary = (Path(p).resolve() for p in (directory, out, binary))
    require(not out.exists(), 'Fresh audit destination required')
    started = time.process_time()
    ledger = Ledger()
    sources = local_sources(__file__)
    for path in sources.values():
        ledger.bind(path)
    manifest = read(ledger.bind(root/'manifest.json'))
    summary = read(ledger.bind(root/'summary.json'))
    validate_manifest(manifest, summary)
    require(not (root/'failure.json').exists(), 'Failed physical population')
    for name, key in [('input-config.json', 'config_sha256'), ('model.json', 'model_sha256'),
                      ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256'),
                      ('latent-region.json', 'latent_region_sha256'), ('latent-guide.json', 'latent_guide_sha256')]:
        ledger.bind(root/'provenance'/name, manifest[key])
    ledger.bind(binary, manifest['executable_sha256'])
    bundle = root/'provenance/source-bundle.json'
    require(bundle.read_bytes() in binary.read_bytes(), 'Pinned source bundle absent from executable')
    samples = ledger.bind(root/'samples.jsonl', summary['samples_sha256'])
    attempts = ledger.bind(root/'attempts.jsonl', summary['attempts_sha256'])
    config, shape, guide, vessel, witness = context(root, manifest, ledger, definition_path, synthetic)
    wall = WallOracle(config, manifest, shape, read(bundle))
    contact = reference.vessel_reference.PrunedExclusionContact(shape, config['fixed_poses'], config['depletant_radius'])
    checks = BatchChecks()
    reducers = {name: RegionMoments(manifest['samples']) for name in CLASSES}
    processed = near = 0
    out.mkdir(parents=True)

    def status(phase, **extra):
        temporary = out/'status.json.tmp'
        write(temporary, dict(complete=phase == 'complete', phase=phase, processed_attempts=processed,
                             batch_size=batch_size, **extra))
        temporary.replace(out/'status.json')

    status('audit')
    try:
        with (out/'geometry.jsonl').open('x') as output:
            for rows in attempt_batches(samples, attempts, manifest['samples'], batch_size):
                checks.check(config, manifest, rows, vessel, guide)
                for row in rows:
                    wall.check(row)
                    geometry = contact.classify(row['pose'], capture_valid=row['capture_valid'], wall_valid=row['wall_valid'])
                    classes = memberships(row, geometry)
                    require(set(classes) == set(reducers), 'Changed region inventory')
                    near += int(geometry['near_core_boundary'])
                    for name, selected in classes.items():
                        reducers[name].add(row, selected)
                    output.write(json.dumps(dict(draw=row['draw'], classes=classes, geometry=geometry), allow_nan=False)+'\n')
                    processed += 1
                output.flush()
                status('audit')
        checks.finish(manifest, summary)
        require(processed == wall.checked == manifest['samples'] and wall.rejected == summary['wall_rejected'],
                'Atomic wall denominator/count differs')
        estimates = {name: reducer.result() for name, reducer in reducers.items()}
        for key, kind in [('total', 'Qz'), ('hard_total', 'Q0')]:
            reference.vessel_reference.log_close(summary['estimates'][key]['log_normalizer'],
                reference.vessel_reference.nullable_log(estimates['total'][kind]['logQ']), 'Total unconditional estimate differs')
        ledger.recheck()
        (out/'provenance').mkdir()
        for name, path in sources.items():
            shutil.copy2(path, out/'provenance'/name)
            require(sha(out/'provenance'/name) == ledger.files[str(Path(path).resolve())], 'Audit source changed while copying')
        ledger.recheck()
        result = dict(schema=SCHEMA, complete=True, population=str(root), manifest=manifest,
            density_audit=checks.density, vessel_generation_audit=dict(checked_vessel_generation_rows=checks.generation_count),
            primitive_count_audit=dict(checks.primitive), wall_audit=wall.result(), geometry_audit=contact.report(),
            near_core_boundary_poses=near, estimates=estimates, shape_witness=witness,
            source_sha256=ledger.files, samples_sha256=sha(samples), attempts_sha256=sha(attempts),
            geometry_sha256=sha(out/'geometry.jsonl'),
            executable_binding=dict(path=str(binary), sha256=manifest['executable_sha256'], artifact_verified=True),
            batching=dict(requested_rows=batch_size, peak_rows=checks.peak_rows, batches=checks.batches,
                scalar_top_weights_per_region=max(1, (manifest['samples']+99)//100),
                scope='O(batch_size) full traces plus scalar top-one-percent heaps; fixed geometry/source buffers are population-independent.'),
            analysis_CPU_seconds=time.process_time()-started, new_pose_draws=0, new_Poisson_clouds=0,
            new_native_classifier_calls=0,
            geometry_reconstruction=dict(physical_pose_checks=processed, density_pose_checks=processed,
                scope='Saved poses only. Density may reconstruct several axes/channels and conditional draw traces per pose; these are not counted as single primitive atom queries.'),
            scope='Independent density/generation, hard/wall/exclusion predicates and logged Poisson accounting. '
                  'All attempted zeros and complete physical marginal retained. Physical native labels need a separate partition. '
                  'No convergence, dispatch or assembly authority; RNG independence, exact thinning, predicates and floating point remain obligations.')
        if vessel.envelope is not None:
            result['vessel_uniform_envelope'] = vessel.envelope.witness
        write(out/'analysis.json', result)
        status('complete', analysis_sha256=sha(out/'analysis.json'))
        write(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
        return result
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'out', 'binary'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--definition', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    result = audit(args.directory, args.out, args.binary, args.batch_size,
                   definition_path=args.definition, synthetic=args.synthetic)
    print(json.dumps(dict(complete=result['complete'], samples=result['manifest']['samples'], batching=result['batching'])))
