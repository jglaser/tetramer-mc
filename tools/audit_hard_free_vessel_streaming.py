#!/usr/bin/env python3
"""Batch the existing full-vessel hard-free audit without changing its checks.

No sampling or native classification. All attempted rows and journal entries are
visited once in order. Geometry labels are saved for a later, once-only native
partition pass. Memory is one batch of traces plus scalar top-one-percent heaps.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from analyze_mobile_native_pocket import local_sources
from analyze_r4_smc_control import Chart, Ledger, close, read, require, sha, write
import physical_hard_free_line_vessel as reference
from streaming_weight_moments import RegionMoments

SCHEMA = 'full-vessel-hard-free-line-streaming-audit-v1'
CLASSES = ('total', 'inside_R4', 'outside_R4', 'latent_zero', 'exclusion_contact', 'unbound')


def attempt_batches(samples, attempts, draws, batch_size):
    require(type(draws) is int and draws > 0 and type(batch_size) is int and batch_size > 0,
            'Positive allocation and batch size required')
    batch = []; seen = 0
    with Path(samples).open() as rows, Path(attempts).open() as journal:
        for text, attempted in zip_longest(rows, journal):
            require(text is not None and attempted is not None, 'Missing attempted row or journal entry')
            row, entry = json.loads(text), json.loads(attempted)
            require(seen < draws and type(row['draw']) is int and row['draw'] == seen,
                    'Missing, repeated, reordered or excess draw')
            require(type(entry.get('draw')) is int and entry == dict(draw=seen, state='begin'),
                    'Attempt journal identity differs')
            batch.append(row); seen += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
    require(seen == draws, 'Truncated unconditional allocation')
    if batch: yield batch


def row_counts(rows):
    return dict(hard_valid=sum(r['hard_valid'] for r in rows),
                capture_rejected=sum(not r['capture_valid'] for r in rows),
                wall_rejected=sum(r['capture_valid'] and not r['wall_valid'] for r in rows),
                hard_rejected=sum(r['capture_valid'] and r['wall_valid'] and not r['hard_valid'] for r in rows),
                raw_points=sum(c['raw_points'] for r in rows for c in r['clouds']))


class BatchChecks:
    """Adapt only local row indices/N; compare summed counters with global N later."""
    sums = ('checked_attempts', 'valid_outside_R4', 'structural_zero_queries',
            'exact_chart_seams', 'valid_outside_source_capture')
    maxima = ('maximum_log_density_error', 'maximum_interval_endpoint_error', 'maximum_inverse_CDF_error')
    fixed = ('source_capture', 'vessel_capture', 'scope')

    def __init__(self, *, conditioned_density_audit=False):
        self.density = None
        self.conditioned_density_audit = conditioned_density_audit
        self.generation_count = 0
        self.primitive = Counter()
        self.batches = self.peak_rows = 0

    def check(self, config, manifest, rows, vessel, guide):
        # The original helpers require draws 0..N-1. Original stream identities
        # were checked above; temporary shallow copies leave all poses/weights
        # and nested metadata untouched, and never modify the saved records.
        adapted = [dict(row, draw=i) for i, row in enumerate(rows)]
        local = dict(manifest, samples=len(rows))
        options={'conditioned_density_audit':True} if self.conditioned_density_audit else {}
        density = reference.check_rows(config, local, adapted, vessel, guide, **options)
        extra=('conditioned_density_audit',) if self.conditioned_density_audit else ()
        require(set(density) == set(self.sums+self.maxima+self.fixed+('outer_branches',)+extra),
                'Unhandled density audit field')
        if self.conditioned_density_audit:
            require(set(density['conditioned_density_audit'])==
                    {'schema','complete','checked_attempts','maxima','scope','domain'},
                    'Unhandled supplemental density field')
        if self.density is None:
            self.density = dict(density, outer_branches=dict(density['outer_branches']))
        else:
            require(all(self.density[k] == density[k] for k in self.fixed), 'Batch physical context changed')
            for key in self.sums: self.density[key] += density[key]
            for key in self.maxima: self.density[key] = max(self.density[key], density[key])
            self.density['outer_branches'] = dict(Counter(self.density['outer_branches'])+Counter(density['outer_branches']))
            if self.conditioned_density_audit:
                old,new=self.density['conditioned_density_audit'],density['conditioned_density_audit']
                require(set(old)==set(new) and set(old['maxima'])==set(new['maxima']),
                        'Unhandled supplemental density field')
                require(all(old[k]==new[k] for k in ('schema','complete','scope','domain')),
                        'Supplemental audit context changed')
                old['checked_attempts']+=new['checked_attempts']
                for key,value in new['maxima'].items():old['maxima'][key]=max(old['maxima'][key],value)
        generation = reference.vessel_reference.check_generation_metadata(config, local, adapted, vessel)
        self.generation_count += generation['checked_vessel_generation_rows']
        summary = dict(samples=len(rows), **row_counts(rows))
        counts = reference.vessel_reference.check_cloud_envelopes_and_counts(local, summary, adapted)
        self.primitive.update(counts)
        self.batches += 1; self.peak_rows = max(self.peak_rows, len(rows))

    def finish(self, manifest, summary):
        require(self.density is not None and self.density['checked_attempts'] == manifest['samples']
                == summary['samples'], 'Incomplete total audit denominator')
        require(all(summary[key] == value for key, value in self.primitive.items()),
                'Global rejection/point counters differ from all attempted rows')
        require(sum(self.density['outer_branches'].values()) == manifest['samples'],
                'Outer branch accounting lost attempts')
        require(self.generation_count == self.density['outer_branches'].get('vessel', 0),
                'Lost vessel generation checks')
        if self.conditioned_density_audit:
            require(self.density['conditioned_density_audit']['checked_attempts']==manifest['samples'],
                    'Incomplete supplemental audit denominator')


class WallOracle:
    """Same direct transformed-atom predicate as audit_wall_domain, loaded once."""
    def __init__(self, config, manifest, shape, source_bundle):
        require(manifest['pose_proposal_schema'] in (1, 2, 3) and manifest['bath_wall_permeable'] is True,
                'Unsupported proposal or bath-wall law')
        for entry in source_bundle['files'].values():
            require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'],
                    'Corrupt embedded source file')
        self.centers = np.asarray([a['center'] for a in shape['atoms']], float)
        self.radii = np.asarray([a['radius'] for a in shape['atoms']], float)
        require(len(self.radii) > 0 and self.centers.shape == (len(self.radii), 3)
                and np.isfinite(self.centers).all() and np.isfinite(self.radii).all()
                and (self.radii > 0).all(), 'Invalid atomic shape')
        bound = float(np.max(np.linalg.norm(self.centers, axis=1)+self.radii))
        require(abs(manifest['shape_bound']-bound) < 2e-10, 'Shape bound differs')
        self.wall = manifest['atomic_wall']; self.center = np.asarray(self.wall['center'], float)
        self.radius = self.wall['radius']
        require(self.center.shape == (3,) and np.isfinite(self.center).all() and math.isfinite(self.radius)
                and self.radius > 0 and (self.radii <= self.radius).all(), 'Invalid atomic wall')
        required = np.linalg.norm(self.center-config['capture_center'])+self.radius+bound
        require(config['capture_radius'] >= required+256*np.finfo(float).eps*(1+required),
                'Capture fails complete wall enclosure')
        self.allowed2 = (self.radius-self.radii)**2
        require(all(self.contains(p) for p in config['fixed_poses']), 'Fixed scaffold leaves atomic wall')
        self.checked = self.rejected = 0

    def contains(self, pose):
        rotation = Chart.rotation(pose['orientation'])
        translation = np.asarray(pose['position'], float)
        require(translation.shape == (3,) and np.isfinite(translation).all(), 'Invalid pose translation')
        positions = self.centers@rotation.T+translation-self.center
        return bool(np.all(np.sum(positions**2, axis=1) <= self.allowed2))

    def check(self, row):
        valid = self.contains(row['pose'])
        require(type(row['wall_valid']) is bool and row['wall_valid'] == valid, 'Atomic wall predicate differs')
        if valid:
            require(row['capture_valid'] is True, 'Capture truncates a wall-valid pose')
        else:
            require(row['hard_valid'] is False and not row['clouds']
                    and all(row[k] is None for k in ('log_importance_weight', 'log_hard_weight', 'q', 'region', 'depletion_contact')),
                    'Wall-invalid attempt lost its zero weight')
            self.rejected += int(row['capture_valid'])
        self.checked += 1

    def result(self):
        return dict(checked_poses=self.checked, wall_rejected_inside_capture=self.rejected,
                    atomic_wall=self.wall, bath_wall_permeable=True,
                    oracle='Direct transformed atomic sphere squared distances; complete capture enclosure checked '
                           'independently once; every attempted pose checked, no bath-wall clipping')


def memberships(row, geometry):
    expected = row['capture_valid'] and row['wall_valid'] and geometry['core_disjoint']
    require(type(row['hard_valid']) is bool and row['hard_valid'] == expected,
            'Independent atomic hard predicate differs')
    valid = row['hard_valid']
    if valid:
        require(row['depletion_contact'] == geometry['exclusion_contact'], 'Independent exclusion contact differs')
    inside = row['latent_density']['in_reference_ball']
    return dict(total=valid, inside_R4=valid and inside, outside_R4=valid and not inside,
                latent_zero=valid and row['latent_density']['structural_zero'],
                exclusion_contact=valid and geometry['exclusion_contact'],
                unbound=valid and not geometry['exclusion_contact'])


def audit(directory, out, binary, batch_size=64, *, conditioned_density_audit=False):
    require(sys.flags.optimize == 0, 'Independent checks require unoptimized Python')
    require(type(batch_size) is int and 1 <= batch_size <= 1024, 'Batch size must be between 1 and 1024')
    root, out, binary = (Path(p).resolve() for p in (directory, out, binary))
    require(not out.exists(), 'Fresh audit destination required')
    started = time.process_time(); ledger = Ledger(); sources = local_sources(__file__)
    for path in sources.values(): ledger.bind(path)
    manifest = read(ledger.bind(root/'manifest.json')); summary = read(ledger.bind(root/'summary.json'))
    require(manifest['schema'] == 6 and manifest['outer_mixture_schema'] == reference.SCHEMA
            and manifest['outer_vessel_probability'] == .5, 'Wrong full-vessel hard-free law')
    require(summary['complete'] is True and summary['manifest'] == manifest and summary['numerical_nulls'] == 0
            and not (root/'failure.json').exists(), 'Incomplete or failed population')
    require(type(manifest['samples']) is int and manifest['samples'] > 0 and manifest['cloud_replicates'] == 2,
            'Positive complete allocation and two clouds required')
    require(manifest['density_measure'] == 'Lebesgue center volume times normalized SO(3) Haar measure'
            and manifest['latent_reference_ball_is_target_restriction'] is False
            and manifest['latent_source_capture']['restricts_target'] is False
            and manifest['bath_wall_permeable'] is True, 'Changed physical measure/domain')
    for name, key in [('input-config.json', 'config_sha256'), ('model.json', 'model_sha256'),
                      ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256'),
                      ('latent-region.json', 'latent_region_sha256'), ('latent-guide.json', 'latent_guide_sha256')]:
        ledger.bind(root/'provenance'/name, manifest[key])
    ledger.bind(binary, manifest['executable_sha256'])
    bundle = root/'provenance/source-bundle.json'
    require(bundle.read_bytes() in binary.read_bytes(), 'Pinned source bundle is absent from executable')
    samples = ledger.bind(root/'samples.jsonl', summary['samples_sha256'])
    attempts = ledger.bind(root/'attempts.jsonl', summary['attempts_sha256'])
    config = read(ledger.bind(root/'config.json')); original = read(root/'provenance/input-config.json')
    for key in ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'metadata'):
        require(config[key] == original[key], 'Physical config changed: '+key)
    require(config['reservoir_density'] == manifest['activity'] and config.get('target_region') is None,
            'Changed bath or hidden target')
    require(math.isfinite(manifest['activity']) and manifest['activity'] >= 0
            and math.isfinite(manifest['lambda']) and manifest['lambda'] > 0, 'Invalid bath/cloud intensity')
    close(manifest['lambda'], config['poisson_lambda_ratio']*manifest['activity']
          if manifest['activity'] > 0 else 1., 'Auxiliary intensity differs')
    shape = read(root/'provenance/shape.json')
    guide = reference.PhysicalHardFreeLineGuide.from_files(root/'provenance/latent-region.json',
        root/'provenance/latent-guide.json', vessel_config=config, shape=shape,
        expected_shape_sha256=manifest['shape_sha256'])
    require(manifest['latent_defensive_uniform_probability'] == guide.guide.alpha
            and manifest['latent_gaussian_component_count'] == guide.guide.count, 'Guide mixture changed')
    source = manifest['latent_source_capture']
    require(source['center'] == guide.region['capture_center'] and source['radius'] == guide.region['capture_radius'],
            'Source capture metadata changed')
    vessel = reference.vessel_reference.VesselDensity(config, manifest, read(root/'provenance/model.json'), bundle)
    wall = WallOracle(config, manifest, shape, read(bundle))
    contact = reference.vessel_reference.PrunedExclusionContact(shape, config['fixed_poses'], config['depletant_radius'])
    checks = BatchChecks(conditioned_density_audit=conditioned_density_audit)
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
                    for name, selected in classes.items(): reducers[name].add(row, selected)
                    output.write(json.dumps(dict(draw=row['draw'], classes=classes, geometry=geometry), allow_nan=False)+'\n')
                    processed += 1
                output.flush(); status('audit')
        checks.finish(manifest, summary)
        require(wall.checked == manifest['samples'] and wall.rejected == summary['wall_rejected'],
                'Atomic wall denominator/count differs')
        estimates = {name: reducer.result() for name, reducer in reducers.items()}
        for key, kind in [('total', 'Qz'), ('hard_total', 'Q0')]:
            reference.vessel_reference.log_close(summary['estimates'][key]['log_normalizer'],
                reference.vessel_reference.nullable_log(estimates['total'][kind]['logQ']), 'Total unconditional estimate differs')
        ledger.recheck(); (out/'provenance').mkdir()
        for name, path in sources.items():
            shutil.copy2(path, out/'provenance'/name)
            require(sha(out/'provenance'/name) == ledger.files[str(Path(path).resolve())], 'Audit source changed while copying')
        ledger.recheck()
        result = dict(schema=SCHEMA, complete=True, population=str(root), manifest=manifest,
            density_audit=checks.density,
            vessel_generation_audit=dict(checked_vessel_generation_rows=checks.generation_count,
                scope='Original generation-coordinate, label and selected-anchor density checks on every vessel draw.'),
            primitive_count_audit=dict(checks.primitive), wall_audit=wall.result(), geometry_audit=contact.report(),
            near_core_boundary_poses=near, estimates=estimates, source_sha256=ledger.files,
            samples_sha256=sha(samples), attempts_sha256=sha(attempts), geometry_sha256=sha(out/'geometry.jsonl'),
            executable_binding=dict(path=str(binary), sha256=manifest['executable_sha256'], artifact_verified=True),
            batching=dict(requested_rows=batch_size, peak_rows=checks.peak_rows, batches=checks.batches,
                scalar_top_weights_per_region=max(1, (manifest['samples']+99)//100),
                scope='O(batch_size) full geometry traces plus two scalar top-one-percent heaps per region; '
                      'shape/model/source and executable buffers are independent of population size.'),
            analysis_CPU_seconds=time.process_time()-started, new_pose_draws=0, new_Poisson_clouds=0,
            new_native_classifier_calls=0,
            scope='Complete independent density, generation, atomic wall/core/contact and logged Poisson count checks. '
                  'Original unfiltered target and every attempted zero retained. No convergence or native-assembly claim. '
                  'RNG independence, exact thinning/envelope certification and floating-point execution remain obligations.')
        write(out/'analysis.json', result)
        status('complete', analysis_sha256=sha(out/'analysis.json'))
        write(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
        return result
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'out', 'binary'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--conditioned-density-audit', action='store_true')
    args = parser.parse_args()
    result = audit(args.directory, args.out, args.binary, args.batch_size,
                   conditioned_density_audit=args.conditioned_density_audit)
    print(json.dumps(dict(complete=result['complete'], samples=result['manifest']['samples'], batching=result['batching'])))
