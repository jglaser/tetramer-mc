#!/usr/bin/env python3
"""Independent physical v7 class-line estimator audit; no sampling on import.

The established hard-free reference supplies unchanged attempted-row/zero,
Poisson-count, linear two-cloud mean, target-window and moment checks. Only the
proposal label adapter and independently reconstructed class-conditioned q are
new. Compact traces preserve every interval; missing component records are
recomputed, never inferred from the selected channel. Counts cannot by themselves
prove independent RNG streams, exact thinning, or envelope geometry correctness.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

import hard_free_line_physical_reference as physical
import native_class_line_reference as line
from native_contact_regions import NativeContactRegions

require, read, sha, close = physical.require, physical.read, physical.sha, physical.close
SCHEMA = 'importance-latent-region-normalizer-v7'
GUIDE_SCHEMA = line.SCHEMA
PROPOSAL_KIND = 'raw-translation-class-conditioned-Gaussian-mixture'
MEASURE = physical.MEASURE


def stream_key(seed, draw, cloud, role):
    """Independent key derivation check, not a replay or RNG independence proof."""
    import hashlib
    require(all(type(v) is int and 0 <= v < 2**64 for v in (seed, draw, cloud)), 'Invalid stream key')
    return hashlib.sha256(b'tetramer-uniform-latent-region-v1'+seed.to_bytes(8, 'little')
        +draw.to_bytes(8, 'little')+cloud.to_bytes(8, 'little')+role.encode()).hexdigest()


def validate_stream_contract(manifest):
    require(manifest['density_trace_contract']['format'] == 'class-line-compact-v1'
            and manifest['density_trace_contract']['diagnostic_full_trace_unchanged'] is True,
            'Wrong physical trace contract')
    expected = dict(hash_domain='tetramer-uniform-latent-region-v1',
        key_fields=['master seed u64 little endian', 'draw u64 little endian', 'cloud index u64 little endian', 'role UTF-8 bytes'],
        proposal_role='latent', proposal_cloud_index=0, physical_cloud_role='cloud',
        physical_cloud_indices='0..cloud_replicates, separately derived; no proposal RNG consumption')
    require(manifest['random_stream_contract'] == expected, 'Changed independent cloud role contract')
    return expected


def poisson_weight_moments(activity, intensity, lower_volume, overlap_uncertain_volume):
    """PGF moments for exp(z L)(1+z/lambda)^K, K~Poisson(lambda B)."""
    require(activity >= 0 and intensity > 0 and lower_volume >= 0 and overlap_uncertain_volume >= 0,
            'Invalid PGF parameters')
    mean = math.exp(activity*(lower_volume+overlap_uncertain_volume))
    variance = mean*mean*math.expm1(activity*activity*overlap_uncertain_volume/intensity)
    return mean, variance


def read_weights(rows, samples, region, arm):
    """Use the validated J W/q and unconditional-denominator checks unchanged.

    Renaming the lineage for the older parser is explicit and private; all other
    row fields, zeros, denominators, selected components, and cloud data persist.
    """
    adapted = []
    for row in rows:
        require(row['proposal_branch'] in ('uniform-shell', 'native-class-line'), 'Unexpected class-guide proposal branch')
        adapted.append(dict(row, proposal_branch='hard-free-line' if row['proposal_branch'] == 'native-class-line' else 'uniform-shell'))
    return physical.read_weights(adapted, samples, region, arm)


def compact_density(recon, u, details):
    """Reconstruct all geometry, class masses, fallbacks and complete q."""
    expected = recon.density(u)
    if expected.get('conditioning_disabled'):
        require(details.get('conditioning_disabled') is True, 'Missing disabled class law')
        return dict(expected, interval_error=0.)
    require(details.get('trace_format') == 'class-line-compact-v1', 'Wrong compact class trace contract')
    close(details['raw_coordinates'], recon.raw(u), 'Compact raw chart coordinates differ')
    line.log_close(details['baseline_log_density'], expected['baseline_log_density'], 'Compact baseline q differs')
    require([a['axis'] for a in details['axes']] == recon.axes, 'Missing/reordered compact class axes')
    endpoint_error = 0.; factors = np.zeros(len(recon.weights))
    for actual, axis in zip(details['axes'], expected['axes']):
        require('components' not in actual, 'Compact trace unexpectedly contains per-component records')
        endpoint_error = max(endpoint_error, line.compare_geometry(actual, axis, recon, u))
        for k, component in enumerate(axis['components']):
            factors[k] += sum(c['probability']*b['multiplier'] for c, b in zip(recon.channels, component['channels']))/len(recon.axes)
    close(details['component_mixture_multipliers'], factors, 'Compact full class mixture differs', atol=1e-12, rtol=3e-7)
    return dict(expected, interval_error=endpoint_error)


def equal_sphere_overlap(radius, distance):
    """Analytic intersection volume, independent of atom/BVH sampling."""
    require(math.isfinite(radius) and radius > 0 and math.isfinite(distance) and distance >= 0, 'Invalid sphere geometry')
    return math.pi*(4*radius+distance)*(2*radius-distance)**2/12 if distance < 2*radius else 0.


def cayley_haar_ball_mass(radius):
    """Normalized SO(3) Haar mass for ||Cayley||<=radius."""
    require(math.isfinite(radius) and radius >= 0, 'Invalid Cayley ball')
    if radius < .01:
        # Independent convergent power series prevents atan(c)-c/(1+c²)
        # cancellation at the R4 translation endpoint.
        return 4/math.pi*sum((-1.)**k*(k+1)/(2*k+3)*radius**(2*k+3) for k in range(12))
    return 2/math.pi*(math.atan(radius)-radius/(1+radius*radius))


def analytic_sphere_mass(core_radius, depletant_radius, activity, latent_radius=4.,
                         translation_scale=1., angular_scale=1., angular_length=8., capture_radius=10.):
    """Exact 1D reduction of the 6D sphere-union R4 integral, then quadrature.

    The chart is centered on the fixed sphere, has zero mean/anchor and diagonal
    blocks sigma_t I3 and sigma_a I3. Its ball couples translation radius r to
    angular Cayley cutoff (sigma_a/ell)*sqrt(R²-(r/sigma_t)²). Orientation is
    physically irrelevant to identical spheres, but its Haar/J measure remains.
    """
    require(core_radius > 0 and depletant_radius >= 0 and activity >= 0
            and translation_scale > 0 and angular_scale > 0 and angular_length > 0
            and latent_radius > 0 and capture_radius > 0, 'Invalid analytic fixture')
    lower, upper = 2*core_radius, min(capture_radius, latent_radius*translation_scale)
    if upper <= lower:
        return dict(mass=0., quadrature_absolute_error=0.)
    def integrand(r):
        cutoff = angular_scale/angular_length*math.sqrt(max(0., latent_radius**2-(r/translation_scale)**2))
        volume = equal_sphere_overlap(core_radius+depletant_radius, r)
        return 4*math.pi*r*r*cayley_haar_ball_mass(cutoff)*math.exp(activity*volume)
    points = [x for x in [2*(core_radius+depletant_radius)] if lower < x < upper]
    mass, error = quad(integrand, lower, upper, points=points, epsabs=1e-11, epsrel=2e-11, limit=200)
    require(math.isfinite(mass) and mass > 0 and error < 1e-8*mass, 'Analytic quadrature unresolved')
    return dict(mass=mass, quadrature_absolute_error=error, radial_limits=[lower, upper],
                measure='4π r² dr times normalized SO(3) Haar Cayley-ball mass; 6D R4 coupling retained')


def validate_shape_witness(compiled, shape, report, compiled_hash, shape_hash):
    """Independently verify the reported complete atom bijection and gap bound."""
    require(compiled['source_input_sha256']['tetramer-shape.json'] == shape_hash,
            'Compiled native source shape hash differs from physical bytes')
    require(report['compiled_sha256'] == compiled_hash and report['expected_shape_sha256'] == shape_hash,
            'Static shape witness provenance differs')
    require(report['center_tolerance_a'] == 1e-10 and report['radius_tolerance_a'] == 1e-12
            and report['observer_hard_overlap_tolerance_a'] == 1e-8, 'Changed static shape tolerances')
    atoms = np.array([a['center'] for a in compiled['monomer_atoms']], float)
    radii = np.array([a['radius'] for a in compiled['monomer_atoms']], float)
    expanded = np.concatenate([atoms @ np.asarray(m['rotation']).T + m['position'] for m in compiled['members']])
    expanded_radii = np.tile(radii, len(compiled['members']))
    physical_atoms = np.asarray([a['center'] for a in shape['atoms']], float)
    physical_radii = np.asarray([a['radius'] for a in shape['atoms']], float)
    count = len(expanded); mapping = report['physical_index_by_native_atom']
    require(count == len(physical_atoms) == report['native_atoms'] == report['physical_atoms'] == report['matched_atoms'],
            'Static shape witness atom counts differ')
    require(len(mapping) == count and all(type(i) is int for i in mapping) and sorted(mapping) == list(range(count)),
            'Static shape witness is not a full bijection')
    require(report['compatible'] is True and report['unmatched_native_atoms'] == [] and report['unmatched_physical_atoms'] == [],
            'Static shape witness leaves unmatched atoms')
    errors = np.linalg.norm(expanded-physical_atoms[mapping], axis=1)
    radius_errors = abs(expanded_radii-physical_radii[mapping])
    require(np.isfinite(errors).all() and np.all(errors <= 1e-10) and np.all(radius_errors <= 1e-12),
            'Native/member geometry differs from physical union')
    bound = 2*float(np.max(errors+np.maximum(expanded_radii-physical_radii[mapping], 0)))
    require(bound <= 1e-8 and report['hard_valid_implication_within_tolerance'] is True,
            'Physical hard validity does not imply observer hard tolerance')
    for key, expected in [('matched_max_center_error_a', float(errors.max())),
                          ('matched_max_radius_error_a', float(radius_errors.max())),
                          ('pair_overlap_slack_bound_a', bound)]:
        close(report[key], expected, 'Static shape witness '+key+' differs', atol=1e-13, rtol=1e-10)
    return dict(atoms=count, complete_bijection=True, maximum_center_error_a=float(errors.max()),
                maximum_radius_error_a=float(radius_errors.max()), pair_overlap_slack_bound_a=bound)


def audit(directory, *, definition_path=None, synthetic=False, journal=None):
    root = Path(directory).resolve(); started = time.process_time()
    manifest, summary = read(root/'manifest.json'), read(root/'summary.json')
    require(manifest['schema'] == SCHEMA and manifest['guide_schema'] == GUIDE_SCHEMA
            and manifest['proposal_kind'] == PROPOSAL_KIND, 'Wrong physical class-guide schema')
    require(summary['complete'] is True and summary['manifest'] == manifest and not (root/'failure.json').exists(),
            'Incomplete or failed physical population')
    require(manifest['cloud_replicates'] == 2 and type(manifest['samples']) is int and manifest['samples'] > 0,
            'Expected fixed positive draw count and two clouds')
    require(type(manifest['seed']) is int and 0 <= manifest['seed'] < 2**64, 'Invalid population seed')
    stream_contract = validate_stream_contract(manifest)
    bindings = {}
    def bind(path, digest=None):
        path = Path(path).resolve(); actual = sha(path)
        if digest is not None: require(actual == digest, 'Changed bound input '+str(path))
        bindings[str(path)] = actual
        return path
    for source in [root/'manifest.json', root/'summary.json', Path(__file__), Path(physical.__file__),
                   Path(physical.native.__file__), Path(physical.statistics.__file__), Path(line.__file__),
                   Path(line.hard.__file__), Path(line.hard.normal.__file__), Path(__import__('native_contact_regions').__file__)]:
        bind(source)
    for name, key in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'), ('shape.json', 'shape_sha256'),
                      ('importance-guide.json', 'importance_guide_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        bind(root/'provenance'/name, manifest[key])
    region, guide, config, shape = [read(root/'provenance'/n) for n in ('region.json', 'importance-guide.json', 'input-config.json', 'shape.json')]
    provenance = manifest['compiled_native']
    compiled = read(bind(root/'provenance/compiled-native.json', provenance['compiled_sha256']))
    shape_witness = validate_shape_witness(compiled, shape, provenance['shape_compatibility'], provenance['compiled_sha256'], manifest['shape_sha256'])
    require(guide['compiled_native']['sha256'] == provenance['compiled_sha256']
            and compiled['source_definition_sha256'] == provenance['source_definition_sha256']
            and compiled['source_input_sha256'] == provenance['source_input_sha256'], 'Native provenance differs')
    require(guide['region_sha256'] == manifest['region_sha256'] and region['shape_sha256'] == manifest['shape_sha256']
            and region['gaussian_chart']['shape_sha256'] == manifest['shape_sha256']
            and guide['shape_sha256'] == manifest['shape_sha256'], 'Guide/chart/shape binding differs')
    fixed = region.get('physical_fixed_neighbors', [region['fixed_neighbor']])
    require(fixed == config['fixed_poses'] == manifest['physical_fixed_neighbors'] == guide['fixed_poses'] == compiled['fixed_poses']
            and region['fixed_neighbor'] == manifest['chart_anchor'], 'Changed fixed scaffold or chart anchor')
    for rk, ck in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'), ('depletant_radius', 'depletant_radius'),
                   ('activity', 'reservoir_density'), ('physical_metric', 'metadata')]:
        require(region[rk] == config[ck], 'Changed physical '+rk)
        if rk in ('capture_center', 'capture_radius', 'depletant_radius'):
            require(guide[rk] == config[ck], 'Changed guide geometry '+rk)
    if synthetic:
        require(definition_path is None and len(shape['atoms']) <= 64 and len(compiled['monomer_atoms']) <= 16,
                'Synthetic compiled adapter cannot audit proteins')
        observer = line.observer_from_compiled_for_synthetic(compiled)
    else:
        require(definition_path is not None, 'Original frozen native definition required')
        observer = NativeContactRegions(bind(definition_path, provenance['source_definition_sha256']))
        line.compare_compiled_definition(compiled, observer)
        for name, digest in observer.definition['input_sha256'].items(): bind(observer.root/name, digest)
    recon = line.Reconstructor(region, guide, config, shape, observer)
    require(manifest['activity'] == region['activity'], 'Activity differs')
    require(math.isfinite(manifest['lambda_ratio']) and manifest['lambda_ratio'] > 0, 'Invalid intensity ratio')
    close(manifest['lambda'], manifest['activity']*manifest['lambda_ratio'] if manifest['activity'] > 0 else 1., 'Auxiliary intensity differs')
    require(manifest['importance_uniform_probability'] == recon.alpha and manifest['importance_component_count'] == len(recon.weights)
            and manifest['proposal_density_measure'] == MEASURE, 'Proposal mixture/measure differs')
    require(manifest['latent_radius'] == recon.radius and manifest['minimum_latent_radius'] == 0., 'Changed target region')
    close(manifest['log_latent_ball_volume'], recon.logvolume, 'Latent ball volume differs')
    close(manifest['log_latent_shell_volume'], recon.logvolume, 'Latent shell volume differs')
    for key, default in [('minimum_original_q', None), ('maximum_original_q', None), ('minimum_original_q_inclusive', True), ('maximum_original_q_inclusive', True)]:
        require(manifest.get(key, default) == region.get(key, default), 'Changed original q-window')
    for key in ('minimum_original_q_inclusive', 'maximum_original_q_inclusive'):
        require(type(manifest.get(key, True)) is bool and type(region.get(key, True)) is bool, 'q endpoint flag must be boolean')
    require(manifest.get('resume_supported') is False, 'Unsupported continuation contract')
    bind(root/'samples.jsonl', summary['samples_sha256']); bind(root/'attempts.jsonl', summary['attempts_sha256'])
    rows = [json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    attempts = [json.loads(s) for s in (root/'attempts.jsonl').read_text().splitlines()]
    require(attempts == [dict(draw=i, state='begin') for i in range(manifest['samples'])], 'Missing/repeated/reordered attempted journal')
    require(summary['samples'] == manifest['samples'], 'Changed draw denominator')
    arrays = read_weights(rows, manifest['samples'], region, dict(alpha=recon.alpha, component_count=len(recon.weights)))
    counts, maxima = Counter(), Counter(); raw_points = 0
    journal_path = None if journal is None else Path(journal).resolve()
    stream = None if journal_path is None else journal_path.open('x')
    def emit(event):
        if stream is not None: stream.write(json.dumps(event, allow_nan=False)+'\n'); stream.flush()
    emit(dict(state='started', input_sha256=bindings, samples=len(rows), synthetic=synthetic))
    for row in rows:
        keys = [stream_key(manifest['seed'], row['draw'], index, role) for index, role in [(0, 'latent'), (0, 'cloud'), (1, 'cloud')]]
        require(len(set(keys)) == 3, 'Proposal/cloud replica key collision')
        emit(dict(state='begin', draw=row['draw'])); tick = time.process_time()
        try:
            u = np.asarray(row['latent']); raw, position, rotation, jac = recon.decode(u)
            close(row['pose']['position'], position, 'Position backmap differs')
            close(line.hard.normal.rotation(row['pose']['orientation']), rotation, 'Orientation backmap differs')
            relative = recon.Rf.T @ line.hard.normal.rotation(row['pose']['orientation']) @ np.asarray(recon.anchor['rotation']).T
            quaternion = Rotation.from_matrix(relative).as_quat(); require(quaternion[3] != 0, 'Chart inverse seam')
            recovered = np.r_[recon.Rf.T@(np.asarray(row['pose']['position'])-recon.fixed['position'])-recon.anchor['position'],
                              recon.ell*quaternion[:3]/quaternion[3]]
            backmap = line.scalar_chart_solve(recon.L0, recovered-recon.m0)
            close(backmap, u, 'Independent inverse chart differs'); close(row['backmapped_latent'], backmap, 'Saved inverse differs')
            close(row['backmapped_radius'], np.linalg.norm(backmap), 'Inverse radius differs')
            close(row['log_physical_jacobian'], jac, 'Physical Jacobian differs')
            close(row['physical_jacobian'], math.exp(jac), 'Exponentiated Jacobian differs')
            result = compact_density(recon, u, row['native_class_line_density'])
            close(row['log_proposal_density'], result['log_density'], 'Complete physical proposal q differs', atol=2e-7, rtol=1e-11)
            draw = row['native_class_line_draw']
            require(type(draw['conditional']) is bool, 'Missing Boolean conditioner flag')
            if draw['conditional']:
                require(row['proposal_branch'] == 'native-class-line' and draw['component'] == row['proposal_component']
                        and recon.beta > 0, 'Selected physical component differs')
            generated = line.audit_draw(draw, recon, u, result)
            maxima['inverse_cdf'] = max(maxima['inverse_cdf'], generated['inverse_error'])
            maxima['interval_endpoints'] = max(maxima['interval_endpoints'], result['interval_error'], generated['endpoint_error'])
            maxima['log_proposal_density'] = max(maxima['log_proposal_density'], abs(row['log_proposal_density']-result['log_density']))
            maxima['log_physical_jacobian'] = max(maxima['log_physical_jacobian'], abs(row['log_physical_jacobian']-jac))
            maxima['latent_backmap'] = max(maxima['latent_backmap'], float(abs(backmap-u).max()))
            capture = bool(np.linalg.norm(position-config['capture_center']) <= config['capture_radius'])
            hard = bool(capture and recon.hard_valid(position, rotation))
            require(row['capture_valid'] == capture and row['hard_valid'] == hard, 'Physical hard/capture indicator differs')
            target_q = physical.native.native_q(region['physical_metric'], row['pose'])
            close(row['q'], target_q, 'Original physical q differs')
            require(row['region_valid'] == physical.q_contains(target_q, region), 'Target q-window differs')
            counts.update(attempted=1, capture_rejected=int(not capture), hard_rejected=int(capture and not hard),
                          region_rejected=int(hard and not row['region_valid']), shell_rejected=int(not row['shell_valid']),
                          contributing=int(math.isfinite(arrays['z'][row['draw']])), conditioned_draws=int(draw['conditional']),
                          fallback_draws=int(draw.get('fallback', False)))
            if row['clouds']:
                for cloud in row['clouds']:
                    physical.cloud_log_weight(cloud, manifest['activity'], manifest['lambda'])
                    raw_points += cloud['raw_points']
                for key in ('lower_volume', 'upper_volume', 'uncertain_volume', 'retained_cells', 'created_cells', 'certified_cells'):
                    require(row['clouds'][0][key] == row['clouds'][1][key], 'Clouds use different geometric envelopes')
        except Exception as exc:
            emit(dict(state='failed', draw=row['draw'], error_type=type(exc).__name__, error=str(exc), cpu_seconds=time.process_time()-tick))
            if stream is not None: stream.close()
            raise
        emit(dict(state='complete', draw=row['draw'], log_density=result['log_density'], maximum_errors=dict(maxima), cpu_seconds=time.process_time()-tick))
    for key in ('capture_rejected', 'hard_rejected', 'region_rejected', 'shell_rejected'):
        require(summary[key] == counts[key], 'Changed rejection summary '+key)
    require(summary['raw_points'] == raw_points, 'Raw cloud total differs')
    estimate, hard_estimate = physical.statistics.moments(arrays['z']), physical.statistics.moments(arrays['h'])
    physical.check_estimate(summary['estimates']['region'], estimate)
    physical.check_estimate(summary['estimates']['hard_region'], hard_estimate)
    require(math.isfinite(summary['sampler_cpu_seconds']) and summary['sampler_cpu_seconds'] >= 0, 'Invalid sampler CPU')
    for path, digest in bindings.items(): require(sha(path) == digest, 'Input changed during audit')
    emit(dict(state='finished', counts=dict(counts), maximum_errors=dict(maxima)))
    if stream is not None: stream.close()
    return dict(schema='independent-native-class-line-physical-audit-v1', complete=True, passed=True, synthetic=synthetic,
        root=str(root), seed=manifest['seed'], samples=manifest['samples'], finite_count=estimate['nonzero'],
        estimate=estimate, hard_region=hard_estimate, paired_noise=physical.statistics.paired_noise(arrays['z'], arrays['pairs']),
        counts=dict(counts), maximum_errors=dict(maxima), input_sha256=bindings, files=bindings,
        samples_sha256=summary['samples_sha256'], attempts_sha256=summary['attempts_sha256'],
        independently_reconstructed_geometry_rows=len(rows), sampler_cpu_seconds=summary['sampler_cpu_seconds'],
        analysis_cpu_seconds=time.process_time()-started, chart_factor_validation=recon.chart_factor_validation,
        independent_shape_witness=shape_witness,
        executable_sha256=manifest['executable_sha256'], source_bundle_sha256=manifest['source_bundle_sha256'],
        journal=None if journal_path is None else dict(path=str(journal_path), sha256=sha(journal_path)),
        stream_contract=stream_contract, independently_checked_distinct_role_keys=3*len(rows),
        new_pose_draws=0, new_Poisson_clouds=0, branch_labels={'0': 'uniform-shell', '1': 'native-class-line'},
        scope='Every attempted draw remains in N, including hard/region/exterior zeros. Independent complete class q,J,geometry,drawtrace and unchanged Poisson/count linear-two-cloud accounting. No equilibrium native-weight, full-domain or assembly conclusion.',
        implementation_obligations='Distinct replica RNG streams, exact thinning and envelope certification remain pinned implementation obligations; cloud counts alone do not establish independence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', '--root', dest='root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--definition', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--journal', type=Path)
    args = parser.parse_args(); require(not args.out.exists(), 'Output must be new')
    value = audit(args.root, definition_path=args.definition, synthetic=args.synthetic, journal=args.journal)
    with args.out.open('x') as output: output.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True, passed=True, samples=value['samples'], counts=value['counts'], maximum_errors=value['maximum_errors'])))
