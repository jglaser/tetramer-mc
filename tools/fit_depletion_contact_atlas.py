#!/usr/bin/env python3
"""Freeze an equally allocated contact proposal from native-blind discovery.

Only the supplied rigid shape and completed discovery outputs supply data.
Every independent slot contributes an initial shape-contact rolling Gaussian
and a Gaussian fit to ALL retained exact pair-MC poses, including rejected
repeats. These correlated, finite traces do not estimate equilibrium basin
weights, converged local masses, IID errors or effective sample sizes.

In the reference chart x=(t-t_ref, ell*Cayley(R R_ref.T)), the fit uses the
population scatter S (denominator n). For the shape-contact rolling baseline
B=L L.T, W=L^-1 S L^-T, and eigendecomposition W=U diag(d) U.T, the frozen
covariance is L U diag(max((1-shrinkage)*d+shrinkage, covariance_floor)) U.T L.T.
The floor is a dimensionless variance in baseline-whitened coordinates.
All Gaussian tails remain. Hard cores and the separate uniform proposal branch
belong to the production runner, not to this normalized proposal file.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import numpy as np
import scipy
from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from analyze_involution_docking_campaign import ChartAudit
from prepare_mobile_posterior_pilot import validate_model
from prepare_mobile_reciprocal_benchmark import density_preflight, reciprocal_envelope
from prepare_shoulder_docking_benchmark import local_dependencies
from prepare_smc_normalizer_atlas import Density

DISCOVERY_SCHEMA = 'native-blind-depletion-contact-discovery-v1'
FIT_SCHEMA = 'native-blind-depletion-contact-atlas-fit-v1'
LIMITATION = ('Finite correlated refinement traces, including rejected repeated states, are '
              'proposal-design data only. No IID, effective sample size, convergence, '
              'local-mass or equilibrium basin-weight claim is made. Every slot has equal '
              'proposal mass irrespective of scores, acceptance, or retained sample count.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def serialized(value):
    """Stable, strict JSON; outputs carry no timestamps or output-directory paths."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write(serialized(value))


@dataclass(frozen=True)
class FitOptions:
    translation_width: float = 1.0
    angle_width_degrees: float = 3.0
    shrinkage: float = 0.25
    covariance_floor: float = 0.01
    initial_weight: float = 0.5
    weight_activity: float = 0.0
    audit_seed: int = 130200011

    def validate(self):
        for name in ('translation_width', 'angle_width_degrees', 'covariance_floor'):
            value = getattr(self, name)
            require(math.isfinite(value) and value > 0, f'{name} must be positive and finite')
        require(math.isfinite(self.shrinkage) and 0 <= self.shrinkage <= 1,
                'shrinkage must lie in [0,1]')
        require(math.isfinite(self.initial_weight) and 0 < self.initial_weight < 1,
                'initial_weight must lie strictly between zero and one; both branches are required')
        require(math.isfinite(self.weight_activity) and self.weight_activity >= 0,
                'weight_activity must be finite and nonnegative')
        require(type(self.audit_seed) is int and 0 <= self.audit_seed < 2**64,
                'audit_seed must fit in u64')


def shape_scale(shape):
    atoms = shape['atoms']
    require(isinstance(atoms, list) and len(atoms) > 0, 'Shape must contain atoms')
    xyz = np.asarray([a['center'] for a in atoms], dtype=float)
    radii = np.asarray([a['radius'] for a in atoms], dtype=float)
    require(xyz.shape == (len(atoms), 3) and np.isfinite(xyz).all(), 'Invalid shape centers')
    require(radii.shape == (len(atoms),) and np.isfinite(radii).all() and (radii > 0).all(),
            'Invalid shape radii')
    rms = float(np.sqrt(np.mean(np.sum((xyz-xyz.mean(axis=0))**2, axis=1))))
    return 2*max(rms, float(radii.max())), dict(center_rms=rms, maximum_atom_radius=float(radii.max()))


def pose_arrays(pose):
    position = np.asarray(pose['position'], dtype=float)
    quaternion = np.asarray(pose['orientation'], dtype=float)
    require(position.shape == (3,) and np.isfinite(position).all(), 'Invalid pose position')
    require(quaternion.shape == (4,) and np.isfinite(quaternion).all()
            and abs(float(quaternion@quaternion)-1) < 1e-8, 'Invalid unit pose quaternion')
    return position, Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def chart_anchor(pose):
    position, rotation = pose_arrays(pose)
    return dict(position=position.tolist(), rotation=rotation.tolist())


def chart_coordinates(reference, poses, ell):
    anchor_t, anchor_r = pose_arrays(reference)
    coordinates, seam_distances = [], []
    for pose in poses:
        position, rotation = pose_arrays(pose)
        quaternion = Rotation.from_matrix(rotation@anchor_r.T).as_quat()
        seam_distances.append(abs(float(quaternion[3])))
        # Fail the whole fit, never trim a sample or silently remove its slot.
        require(abs(quaternion[3]) > 1e-12,
                'Retained pose is at the reference Cayley seam; cannot freeze this chart without dropping data')
        coordinates.append(np.r_[position-anchor_t, ell*quaternion[:3]/quaternion[3]])
    result = np.asarray(coordinates)
    require(result.ndim == 2 and result.shape[1] == 6 and np.isfinite(result).all(),
            'Invalid retained chart coordinates')
    return result, min(seam_distances)


def contact_lever(contact, pose, shape):
    """Validate the recorded atom witness using only the rigid shape.

    The lever is measured from the zero-gap radial contact origin; the chart
    anchor has the separately recorded outward gap. This is the same rolling
    construction as contact_atlas.rs, not a fitted physical spring constant.
    """
    ray = contact['radial_contact']
    if ray.get('kind') == 'nearest_pair':
        return nearest_pair_lever(contact, pose, shape)
    direction = np.asarray(ray['direction'], dtype=float)
    distance = float(ray['distance'])
    require(direction.shape == (3,) and np.isfinite(direction).all()
            and abs(np.linalg.norm(direction)-1) < 1e-9 and math.isfinite(distance) and distance >= 0,
            'Invalid radial-contact ray')
    position, rotation = pose_arrays(pose)
    _, witness_r = pose_arrays(dict(position=[0., 0., 0.], orientation=ray['orientation']))
    require(np.allclose(rotation, witness_r, rtol=0, atol=1e-9), 'Contact orientation differs from pose')
    require(np.linalg.norm(position-direction*(position@direction)) < 1e-7
            and position@direction >= distance-1e-7, 'Pose is not outward of its radial contact')
    atoms = shape['atoms']
    moving, fixed = ray['moving_atom'], ray['fixed_atom']
    require(type(moving) is int and type(fixed) is int and 0 <= moving < len(atoms)
            and 0 <= fixed < len(atoms), 'Invalid contact witness atom indices')
    contact_position = direction*distance
    delta = contact_position+rotation@np.asarray(atoms[moving]['center'])-np.asarray(atoms[fixed]['center'])
    length = float(np.linalg.norm(delta))
    radius = atoms[moving]['radius']+atoms[fixed]['radius']
    require(length > 0 and abs(length-radius) < 1e-7*max(1., radius), 'Witness atoms are not tangent')
    surface = np.asarray(atoms[fixed]['center'])+delta/length*atoms[fixed]['radius']
    lever = surface-contact_position
    require(np.allclose(surface, contact['surface_point'], rtol=1e-10, atol=1e-8)
            and np.allclose(lever, contact['mobile_surface_lever'], rtol=1e-10, atol=1e-8),
            'Recorded surface point or rolling lever differs from shape witness')
    return lever


def nearest_pair_lever(contact, pose, shape):
    """Validate a supplied-start witness: the globally closest atom pair.

    Such poses may lie inside their outermost radial contact (interlocked
    interfaces), so the lever is measured from the pose origin instead.
    """
    witness = contact['radial_contact']
    position, rotation = pose_arrays(pose)
    _, witness_r = pose_arrays(dict(position=[0., 0., 0.], orientation=witness['orientation']))
    require(np.allclose(rotation, witness_r, rtol=0, atol=1e-9)
            and np.allclose(position, witness['position'], rtol=0, atol=1e-9),
            'Nearest-pair witness differs from pose')
    atoms = shape['atoms']
    centers = np.asarray([a['center'] for a in atoms], dtype=float)
    radii = np.asarray([a['radius'] for a in atoms], dtype=float)
    moving, fixed = witness['moving_atom'], witness['fixed_atom']
    require(type(moving) is int and type(fixed) is int and 0 <= moving < len(atoms)
            and 0 <= fixed < len(atoms), 'Invalid contact witness atom indices')
    placed = centers@rotation.T+position
    recorded = float(np.linalg.norm(placed[moving]-centers[fixed])-radii[moving]-radii[fixed])
    require(abs(recorded-witness['gap']) < 1e-7, 'Witness gap differs from its atom pair')
    # Every pair with a gap below the recorded one lies within this center distance.
    reach = recorded+2*float(radii.max())+1e-9
    pairs = cKDTree(placed).sparse_distance_matrix(cKDTree(centers), reach, output_type='ndarray')
    gaps = pairs['v']-radii[pairs['i']]-radii[pairs['j']]
    require(len(gaps) > 0 and gaps.min() >= 0, 'Nearest-pair pose is not hard-valid')
    require(abs(gaps.min()-recorded) < 1e-9, 'Witness is not the closest atom pair')
    delta = placed[moving]-centers[fixed]
    surface = centers[fixed]+delta/np.linalg.norm(delta)*radii[fixed]
    lever = surface-position
    require(np.allclose(surface, contact['surface_point'], rtol=1e-10, atol=1e-8)
            and np.allclose(lever, contact['mobile_surface_lever'], rtol=1e-10, atol=1e-8),
            'Recorded surface point or rolling lever differs from shape witness')
    return lever


def rolling_covariance(lever, ell, options):
    x, y, z = lever
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = np.deg2rad(options.angle_width_degrees)
    factor = np.zeros((6, 6))
    factor[:3, :3] = options.translation_width*np.eye(3)
    factor[:3, 3:] = angle*cross
    factor[3:, 3:] = ell*angle/2*np.eye(3)
    return factor@factor.T


def regularized_moments(coordinates, baseline, options):
    """MLE scatter of the retained time series; no IID or uncertainty estimate."""
    coordinates = np.asarray(coordinates, dtype=float)
    require(coordinates.ndim == 2 and coordinates.shape[1] == 6 and len(coordinates) > 0
            and np.isfinite(coordinates).all(), 'Require nonempty finite six-dimensional coordinates')
    mean = coordinates.mean(axis=0)
    centered = coordinates-mean
    scatter = centered.T@centered/len(coordinates)
    lower = np.linalg.cholesky(baseline)
    whitened = solve_triangular(lower, centered.T, lower=True).T
    whitened_scatter = whitened.T@whitened/len(coordinates)
    raw_values, vectors = np.linalg.eigh((whitened_scatter+whitened_scatter.T)*.5)
    shrunk = (1-options.shrinkage)*np.maximum(raw_values, 0.)+options.shrinkage
    values = np.maximum(shrunk, options.covariance_floor)
    factor = lower@vectors
    covariance = (factor*values)@factor.T
    covariance = (covariance+covariance.T)*.5
    np.linalg.cholesky(covariance)
    return mean, covariance, dict(empirical_mean=mean.tolist(), empirical_covariance=scatter.tolist(),
        empirical_covariance_eigenvalues=np.linalg.eigvalsh(scatter).tolist(),
        whitened_empirical_eigenvalues=raw_values.tolist(), whitened_final_eigenvalues=values.tolist(),
        floored_eigenvalues=int(np.count_nonzero(shrunk < options.covariance_floor)),
        retained_count=len(coordinates), scatter_denominator=len(coordinates))


def width_metrics(covariance, ell):
    diagonal = np.diag(covariance)
    return dict(covariance_eigenvalues=np.linalg.eigvalsh(covariance).tolist(),
        translation_marginal_std=np.sqrt(diagonal[:3]).tolist(),
        angular_coordinate_marginal_std=np.sqrt(diagonal[3:]).tolist(),
        linearized_angle_marginal_std_degrees=np.rad2deg(2*np.sqrt(diagonal[3:])/ell).tolist(),
        covariance_log_determinant=float(np.linalg.slogdet(covariance)[1]),
        covariance_condition_number=float(np.linalg.cond(covariance)))


def empty_model(shape_hash, ell):
    return dict(schema='weighted-pose-mixture-v1', coordinate_convention='anchor-body-relative',
        shape_sha256=shape_hash, angular_length=ell, anchors=[], means=[], covariances=[], weights=[])


def append_component(model, anchor, mean, covariance, weight):
    model['anchors'].append(copy.deepcopy(anchor))
    model['means'].append(np.asarray(mean).tolist())
    model['covariances'].append(np.asarray(covariance).tolist())
    model['weights'].append(float(weight))


def fit_model(shape, discoveries, shape_hash, options=FitOptions()):
    """Pure deterministic fit; no paths, native references, scores or RNG enter it."""
    options.validate()
    if isinstance(discoveries, dict):
        discoveries = [discoveries]
    require(len(discoveries) > 0, 'Require at least one discovery population')
    ell, scale = shape_scale(shape)
    indexed, seed_slots = [], set()
    for source_index, discovery in enumerate(discoveries):
        require(discovery['schema'] == DISCOVERY_SCHEMA, 'Unsupported discovery schema')
        require(discovery['shape_sha256'] == shape_hash, 'Discovery physical shape differs')
        slots = discovery['slots']
        require(isinstance(slots, list) and slots, 'Discovery has no independent slots')
        ids = [slot['slot'] for slot in slots]
        require(all(type(index) is int and index >= 0 for index in ids) and len(set(ids)) == len(ids),
                'Local slot IDs must be distinct nonnegative integers')
        if 'starts' in discovery.get('config', {}):
            require(discovery['config']['starts'] == len(slots), 'Discovery lost an independent slot')
        seed = discovery.get('config', {}).get('seed')
        require(type(seed) is int and 0 <= seed < 2**64, 'Discovery must record its u64 source seed')
        for local_slot in ids:
            key = (seed, local_slot)
            require(key not in seed_slots,
                    'Repeated source seed and local slot cannot be counted as independent discovery')
            seed_slots.add(key)
        indexed.extend((source_index, slot) for slot in slots)
    count = len(indexed)
    # Slot proposal masses: equal (weight_activity = 0), or tempered pair Boltzmann
    # factors exp(z_w C) of each slot's own search-side score. Held-out validation
    # clouds never enter. Any positive frozen weights leave the target unchanged
    # because production uses the complete proposal density.
    if options.weight_activity > 0:
        log_raw = np.array([options.weight_activity*float(slot['search_optimized_score']['volume'])
                            for _, slot in indexed])
        masses = np.exp(log_raw-log_raw.max()); masses /= masses.sum()
    else:
        masses = np.full(count, 1./count)
    base, initial = empty_model(shape_hash, ell), empty_model(shape_hash, ell)
    reports = []
    for index, (source_index, slot) in enumerate(indexed):
        initial_anchor, reference = chart_anchor(slot['initial_pose']), slot['optimized_pose']
        initial_lever = contact_lever(slot['initial_contact'], slot['initial_pose'], shape)
        refined_lever = contact_lever(slot['optimized_contact'], reference, shape)
        initial_cov = rolling_covariance(initial_lever, ell, options)
        fit_baseline = rolling_covariance(refined_lever, ell, options)
        samples = slot['refinement_samples']
        require(isinstance(samples, list) and samples, f'Slot {source_index}:{slot["slot"]} has no retained refinement poses')
        steps = [sample['step'] for sample in samples]
        require(all(type(step) is int and step >= 0 for step in steps)
                and all(a < b for a, b in zip(steps, steps[1:])), 'Retained steps must strictly increase')
        require(all(type(sample['accepted']) is bool for sample in samples), 'Retained acceptance flags must be Boolean')
        # Do not condition on accepted, score, validation, motif or any label.
        poses = [sample['pose'] for sample in samples]
        coordinates, seam_distance = chart_coordinates(reference, poses, ell)
        mean, covariance, report = regularized_moments(coordinates, fit_baseline, options)
        mass = float(masses[index])
        append_component(base, initial_anchor, np.zeros(6), initial_cov, options.initial_weight*mass)
        append_component(base, chart_anchor(reference), mean, covariance, (1-options.initial_weight)*mass)
        append_component(initial, initial_anchor, np.zeros(6), initial_cov, mass)
        report.update(global_slot=index, source_index=source_index, local_slot=slot['slot'],
            source_seed=discoveries[source_index].get('config', {}).get('seed'),
            seeds=copy.deepcopy(slot.get('seeds', {})), raw_slot_weight=1., slot_proposal_mass=mass,
            initial_component=2*index, refined_component=2*index+1,
            reference_pose=copy.deepcopy(reference), reference_rule='recorded optimized contact pose (radial or nearest-pair witness)',
            retained_steps=steps, retained_rejected_count=sum(not sample['accepted'] for sample in samples),
            retained_accepted_count=sum(sample['accepted'] for sample in samples),
            unique_retained_poses=len({serialized(pose) for pose in poses}),
            minimum_absolute_reference_quaternion_scalar=seam_distance,
            initial_width=width_metrics(initial_cov, ell), refined_width=width_metrics(covariance, ell),
            refinement_regularizer_width=width_metrics(fit_baseline, ell),
            initial_contact_lever=initial_lever.tolist(), refinement_contact_lever=refined_lever.tolist())
        reports.append(report)
    validate_model(base, shape_hash)
    validate_model(initial, shape_hash)
    metrics = dict(schema=FIT_SCHEMA, source_populations=len(discoveries), independent_slots=count,
        base_components=len(base['weights']), virtual_components=2*len(base['weights']),
        angular_length=ell, angular_length_rule='2*max(atom-center RMS about their mean, maximum atom radius)',
        shape_scale=scale, options=asdict(options), retained_count=sum(r['retained_count'] for r in reports),
        retained_rejected_count=sum(r['retained_rejected_count'] for r in reports), slots=reports,
        slot_allocation=('All independent slots equally weighted; no score or population-size weights.'
                         if options.weight_activity == 0 else
                         'Tempered search-score weights exp(weight_activity*C_search); held-out clouds unused.'),
        effective_slot_count=float(1/np.sum(masses**2)),
        covariance_rule=__doc__.split('\n\n')[2], limitation=LIMITATION,
        native_geometry_inputs=0, production_pose_inputs=0, physical_updates=0, bath_queries=0,
        full_gaussian_support=True, uniform_branch_in_model=False)
    return reciprocal_envelope(base), reciprocal_envelope(initial), metrics


def proposal_preflight(model, seed):
    """Independent normalized measure, exact inverse and physical Jacobian probes."""
    base = model['base_model']
    check = density_preflight(base, model, [], seed=seed)
    audit, density = ChartAudit(model), Density(model)
    latent = np.array([.2, -.3, .4, .1, -.2, .3])
    normal_log = float(-3*np.log(2*np.pi)-.5*latent@latent)
    roundtrip, measure, independent = [], [], []
    probes = []
    for label in range(len(density.weights)):
        pose, log_jacobian = audit.decode(label, latent)
        roundtrip.append(float(np.max(np.abs(audit.encode(label, pose)-latent))))
        component_log = density.evaluate([pose])[2][0, label]-np.log(density.weights[label])
        measure.append(float(abs(component_log+log_jacobian-normal_log)))
        independent.append(float(abs(audit.gaussian(label, pose)-component_log)))
        if label < 4:
            probes.append(dict(label=label, latent=latent.tolist(), pose=pose,
                physical_log_jacobian=log_jacobian, full_log_density=float(density.evaluate([pose])[0][0]),
                unweighted_component_log_density=float(component_log)))
    # Check ordinary and inverse physical charts for the first and last slots.
    jacobian_checks = []
    labels = sorted(label for label in {0, 1, 2, 3, len(density.weights)-4, len(density.weights)-3,
                                       len(density.weights)-2, len(density.weights)-1}
                    if 0 <= label < len(density.weights))
    for label in labels:
        pose, log_jacobian = audit.decode(label, latent)
        _, rotation = audit.arrays(pose)
        numeric = np.zeros((6, 6))
        epsilon = 2e-5
        for column in range(6):
            delta = np.eye(6)[column]*epsilon
            hi, _ = audit.decode(label, latent+delta)
            lo, _ = audit.decode(label, latent-delta)
            hp, hr = audit.arrays(hi)
            lp, lr = audit.arrays(lo)
            numeric[:3, column] = (hp-lp)/(2*epsilon)
            numeric[3:, column] = (Rotation.from_matrix(hr@rotation.T).as_rotvec()
                                  -Rotation.from_matrix(lr@rotation.T).as_rotvec())/(2*epsilon)
        measured = abs(float(np.linalg.det(numeric)))/(8*np.pi**2)
        relative_error = abs(measured/math.exp(log_jacobian)-1)
        require(relative_error < 2e-5, 'Physical chart Jacobian finite difference failed')
        jacobian_checks.append(dict(label=label, relative_error=relative_error))
    require(max(roundtrip) < 2e-7 and max(measure) < 2e-7 and max(independent) < 2e-7,
            'Independent Gaussian measure or chart inverse check failed')
    return dict(passed=True, reciprocal_density=check, all_virtual_branches=len(density.weights),
        maximum_encode_decode_error=max(roundtrip), maximum_normalized_measure_error=max(measure),
        maximum_independent_component_density_error=max(independent),
        physical_jacobian_checks=jacobian_checks, density_probes=probes,
        normalization='Every full Gaussian integrates to one under its Cayley/Haar Jacobian; '
                      'positive base weights sum to one, and exact physical inversion splits each weight equally.',
        physical_updates=0, bath_queries=0, native_geometry_inputs=0, production_pose_inputs=0)


def load_discovery(directory, shape_hash):
    directory = Path(directory).resolve()
    if directory.is_file():
        require(directory.name == 'discovery.json', 'Expected discovery directory or discovery.json')
        directory = directory.parent
    manifest_bytes = (directory/'manifest.json').read_bytes()
    discovery_bytes = (directory/'discovery.json').read_bytes()
    manifest, discovery = json.loads(manifest_bytes), json.loads(discovery_bytes)
    require(manifest.get('schema') == DISCOVERY_SCHEMA and manifest.get('complete') is True,
            'Require a completed native-blind discovery manifest')
    require(manifest.get('shape_sha256') == shape_hash and discovery.get('shape_sha256') == shape_hash,
            'Discovery shape hash differs from supplied shape')
    output_hashes = manifest.get('outputs_sha256', {})
    expected = output_hashes.get('discovery.json', manifest.get('discovery_sha256'))
    require(expected == sha_bytes(discovery_bytes), 'Discovery payload SHA256 differs from complete manifest')
    archived = {'manifest.json': manifest_bytes, 'discovery.json': discovery_bytes}
    for name, digest in output_hashes.items():
        require(name in ('discovery.json', 'shape.json', 'source-bundle.json', 'provenance.json'),
                'Unexpected discovery artifact; no arbitrary manifest paths are read')
        raw = discovery_bytes if name == 'discovery.json' else (directory/name).read_bytes()
        require(sha_bytes(raw) == digest, 'Discovery artifact SHA256 differs: '+name)
        if name == 'shape.json':
            require(digest == shape_hash, 'Archived discovery shape differs')
        archived[name] = raw
    return discovery, dict(directory=str(directory), manifest=manifest_bytes,
                           discovery=discovery_bytes, files=archived)


def prepare(shape_path, discovery_paths, out, options=FitOptions()):
    options.validate()
    out, shape_path = Path(out).resolve(), Path(shape_path).resolve()
    require(not out.exists(), 'Require a fresh output directory; frozen proposals are never overwritten')
    shape_bytes = shape_path.read_bytes()
    shape, shape_hash = json.loads(shape_bytes), sha_bytes(shape_bytes)
    discoveries, archives = [], []
    for path in discovery_paths:
        discovery, archive = load_discovery(path, shape_hash)
        discoveries.append(discovery)
        archives.append(archive)
    require(len({archive['directory'] for archive in archives}) == len(archives),
            'Repeated discovery directories are not independent populations')
    require(len({sha_bytes(archive['discovery']) for archive in archives}) == len(archives),
            'Identical discovery populations cannot be counted as independent slots twice')
    model, initial, metrics = fit_model(shape, discoveries, shape_hash, options)
    metrics['sources'] = [dict(source_index=i, directory=a['directory'],
        discovery_sha256=sha_bytes(a['discovery']), manifest_sha256=sha_bytes(a['manifest']))
        for i, a in enumerate(archives)]
    checks = dict(refined=proposal_preflight(model, options.audit_seed),
                  initial_only=proposal_preflight(initial, options.audit_seed))
    # Source closure archives code only. No constants naming historical native
    # models, motifs or configurations are dereferenced by this preparation.
    sources = local_dependencies([Path(__file__).resolve()])
    provenance = out/'provenance'
    provenance.mkdir(parents=True)
    (provenance/'shape.json').write_bytes(shape_bytes)
    for index, archive in enumerate(archives):
        target = provenance/f'discovery-{index:03d}'
        target.mkdir()
        for name, raw in sorted(archive['files'].items()):
            (target/name).write_bytes(raw)
    code = provenance/'tools'
    code.mkdir()
    for name, path in sorted(sources.items()):
        shutil.copyfile(path, code/name)
    write_new(out/'model.json', model)
    write_new(out/'model-base.json', model['base_model'])
    write_new(out/'model-initial-only.json', initial)
    write_new(out/'fit-metrics.json', metrics)
    write_new(out/'preflight.json', checks)
    manifest = dict(schema=FIT_SCHEMA, complete=True, model='model.json',
        model_sha256=sha_bytes((out/'model.json').read_bytes()), shape_sha256=shape_hash,
        initial_only_model='model-initial-only.json',
        independent_slots=metrics['independent_slots'], source_populations=len(discoveries),
        base_components=metrics['base_components'], virtual_components=metrics['virtual_components'],
        input_sha256={p.relative_to(provenance).as_posix(): sha_bytes(p.read_bytes())
                      for p in sorted(provenance.rglob('*')) if p.is_file()},
        outputs_sha256={p.name: sha_bytes(p.read_bytes()) for p in sorted(out.glob('*.json'))},
        options=asdict(options), runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        native_informed_proposal=False, native_geometry_used_in_preparation=False,
        production_launched=False, physical_updates=0, bath_queries=0,
        support='Full immutable Gaussian support; no rejection, truncation, native gate, or uniform branch in the file.',
        limitation=LIMITATION)
    write_new(out/'manifest.json', manifest)
    write_new(out/'freeze.json', dict(files={p.relative_to(out).as_posix(): sha_bytes(p.read_bytes())
        for p in sorted(out.rglob('*')) if p.is_file()}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', type=Path, required=True)
    parser.add_argument('--discovery', type=Path, nargs='+', action='append', required=True,
                        help='Completed discovery directories, in fixed order; may be repeated')
    parser.add_argument('--out', type=Path, required=True)
    defaults = FitOptions()
    for name in ('translation_width', 'angle_width_degrees', 'shrinkage', 'covariance_floor', 'initial_weight',
                 'weight_activity'):
        parser.add_argument('--'+name.replace('_', '-'), type=float, default=getattr(defaults, name))
    parser.add_argument('--audit-seed', type=int, default=defaults.audit_seed)
    args = parser.parse_args()
    options = FitOptions(**{name: getattr(args, name) for name in asdict(defaults)})
    manifest = prepare(args.shape, [p for group in args.discovery for p in group], args.out, options)
    print(json.dumps(dict(complete=True, output=str(args.out.resolve()),
        model_sha256=manifest['model_sha256'], independent_slots=manifest['independent_slots'],
        base_components=manifest['base_components'], virtual_components=manifest['virtual_components'],
        preflight_passed=True, physical_updates=0, bath_queries=0), sort_keys=True))


if __name__ == '__main__':
    main()
