#!/usr/bin/env python3
"""Freeze alternative importance guides from completed, independent SMC populations.

No SMC normalizer or ancestry weight enters the proposal weights. These models
are retrospective proposal controls, not estimates of physical basin weights.
"""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity")
BASE = ROOT / "runs/involution-docking-conditional-5000/provenance/model.json"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def arrays(poses):
    t = np.asarray([p["position"] for p in poses], dtype=float)
    q = np.asarray([p["orientation"] for p in poses], dtype=float)
    assert np.isfinite(t).all() and np.isfinite(q).all()
    assert np.max(np.abs(np.sum(q * q, axis=1) - 1.)) < 1e-8
    q /= np.linalg.norm(q, axis=1)[:, None]
    return t, q, Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()


def relative_poses(poses, fixed):
    t, _, r = arrays(poses)
    ft, _, fr = arrays([fixed])
    rt = (t - ft[0]) @ fr[0]
    rr = fr[0].T @ r
    rq = Rotation.from_matrix(rr).as_quat()[:, [3, 0, 1, 2]]
    return [dict(position=p.tolist(), orientation=q.tolist()) for p, q in zip(rt, rq)]


def convert_model(model, fixed):
    ft, _, fr = arrays([fixed])
    block = np.zeros((6, 6))
    block[:3, :3] = block[3:, 3:] = fr[0].T
    result = copy.deepcopy(model)
    result["anchors"] = [dict(position=(fr[0].T @ (np.asarray(a["position"]) - ft[0])).tolist(),
                               rotation=(fr[0].T @ np.asarray(a["rotation"])).tolist())
                         for a in model["anchors"]]
    result["means"] = (np.asarray(model["means"]) @ block.T).tolist()
    result["covariances"] = (block @ np.asarray(model["covariances"]) @ block.T).tolist()
    result["coordinate_convention"] = "anchor-body-relative"
    return result


def unwrap_proposal_model(model):
    """Return the legacy Gaussian model and explicit exact-reciprocal flags.

    The envelope deliberately lacks legacy Gaussian fields, so readers without
    reciprocal support fail rather than silently sampling a different density.
    Legacy dictionaries, including schema-less analysis fixtures, are returned
    unchanged. Reciprocal models require body-relative pose coordinates.
    """
    if not isinstance(model, dict):
        raise ValueError('A proposal model must be a dictionary')
    if model.get('schema') != 'reciprocal-pose-mixture-v1':
        if 'base_model' in model or 'reciprocal_components' in model:
            raise ValueError('Reciprocal fields require the reciprocal-pose-mixture-v1 envelope')
        return model, [False]*len(model['weights'])
    if set(model) != {'schema', 'base_model', 'reciprocal_components'}:
        raise ValueError('Reciprocal envelope requires exactly schema, base_model, reciprocal_components')
    base, flags = model['base_model'], model['reciprocal_components']
    if not isinstance(base, dict) or 'base_model' in base or 'reciprocal_components' in base or base.get('schema') == 'reciprocal-pose-mixture-v1':
        raise ValueError('Nested or ambiguous reciprocal base model')
    required = {'angular_length', 'anchors', 'means', 'covariances', 'weights'}
    if not required <= set(base):
        raise ValueError('Reciprocal base model lacks required Gaussian fields')
    if base.get('coordinate_convention') != 'anchor-body-relative':
        raise ValueError('Reciprocal proposal requires anchor-body-relative base coordinates')
    weights = base['weights']
    if not isinstance(weights, list) or not weights:
        raise ValueError('Reciprocal base model needs nonempty Gaussian weights')
    if not isinstance(flags, list) or len(flags) != len(weights) or any(type(flag) is not bool for flag in flags):
        raise ValueError('reciprocal_components must contain one strict Boolean per base Gaussian')
    if any(not isinstance(base[key], list) or len(base[key]) != len(weights) for key in ('anchors', 'means', 'covariances')):
        raise ValueError('Reciprocal Gaussian arrays have inconsistent component counts')
    return base, list(flags)


def reciprocal_virtual_branches(base, flags):
    """Component k, followed by its inverse if selected; selected mass splits equally."""
    indices, inverted, weights = [], [], []
    for index, (weight, reciprocal) in enumerate(zip(base['weights'], flags)):
        indices.append(index)
        inverted.append(False)
        weights.append(weight*.5 if reciprocal else weight)
        if reciprocal:
            indices.append(index)
            inverted.append(True)
            weights.append(weight*.5)
    return np.asarray(indices, dtype=int), np.asarray(inverted, dtype=bool), np.asarray(weights, dtype=float)


def reciprocal_arrays(t, r):
    """SE(3) inverse: (-R.T t,R.T), preserving d³t times Haar exactly."""
    inverse = r.swapaxes(-1, -2)
    return -np.einsum('nij,nj->ni', inverse, t), inverse


class Density:
    """Physical density with virtual exact-reciprocal branches, if requested."""
    def __init__(self, model):
        self.proposal_model = model
        self.model, self.reciprocal_components = unwrap_proposal_model(model)
        self.base_indices, self.inverted, self.weights = reciprocal_virtual_branches(self.model, self.reciprocal_components)
        self.ell = float(self.model["angular_length"])
        self.mean = np.asarray(self.model["means"])[self.base_indices]
        self.covariance = np.asarray(self.model["covariances"])[self.base_indices]
        self.anchors = [self.model['anchors'][i] for i in self.base_indices]
        self.lower = np.linalg.cholesky(self.covariance)
        self.logdet = np.log(np.diagonal(self.lower, axis1=1, axis2=2)).sum(axis=1)
        assert np.all(self.weights > 0) and abs(self.weights.sum() - 1) < 1e-12

    def evaluate(self, poses):
        t, _, r = arrays(poses)
        inverse_t, inverse_r = reciprocal_arrays(t, r) if self.inverted.any() else (None, None)
        logs, norms = [], []
        for k, anchor in enumerate(self.anchors):
            translation, rotation = (inverse_t, inverse_r) if self.inverted[k] else (t, r)
            delta = rotation @ np.asarray(anchor["rotation"]).T
            q = Rotation.from_matrix(delta).as_quat()
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                c = q[:, :3] / q[:, 3, None]
                x = np.column_stack((translation - anchor["position"], self.ell * c))
                finite = np.isfinite(x).all(axis=1)
                z = solve_triangular(self.lower[k], (x[finite] - self.mean[k]).T, lower=True).T
                quadratic = np.sum(z * z, axis=1)
                log_jacobian = self.logdet[k] - 3 * np.log(self.ell) - 2 * np.log(np.pi) - 2 * np.log1p(np.sum(c[finite]**2, axis=1))
                values = -3 * np.log(2 * np.pi) - .5 * quadratic - log_jacobian
            log = np.full(len(t), -np.inf)
            norm = np.full(len(t), np.inf)
            log[finite], norm[finite] = values + np.log(self.weights[k]), np.sqrt(quadratic)
            logs.append(log)
            norms.append(norm)
        logs, norms = np.asarray(logs).T, np.asarray(norms).T
        return logsumexp(logs, axis=1), norms, logs

    def draw_component(self, rng, component, count):
        latent = rng.normal(size=(count, 6)) @ self.lower[component].T + self.mean[component]
        anchor = self.anchors[component]
        t = latent[:, :3] + anchor["position"]
        q = np.column_stack((latent[:, 3:] / self.ell, np.ones(count)))
        r = Rotation.from_quat(q).as_matrix() @ np.asarray(anchor["rotation"])
        if self.inverted[component]:
            t, r = reciprocal_arrays(t, r)
        q = Rotation.from_matrix(r).as_quat()[:, [3, 0, 1, 2]]
        return [dict(position=p.tolist(), orientation=v.tolist()) for p, v in zip(t, q)]


def fit_population(poses, replica, ell, absolute_floor, relative_floor, std_scale):
    t, q, r = arrays(poses)
    # Among actually observed orientations choose the one with the greatest
    # minimum absolute quaternion dot: a maximin sampled reference. This
    # minimizes the largest Cayley radius, without trimming any endpoint.
    worst_dot = np.min(np.abs(q @ q.T), axis=1)
    anchor_index = int(np.argmax(worst_dot))
    anchor_r = r[anchor_index]
    delta = Rotation.from_matrix(r @ anchor_r.T).as_quat()
    if np.any(delta[:, 3] == 0):
        raise ValueError(f"Population {replica} crosses an exact Cayley seam; one chart is insufficient")
    c = delta[:, :3] / delta[:, 3, None]
    x = np.column_stack((t - t[anchor_index], ell * c))
    mean = x.mean(axis=0)
    centered = x - mean
    covariance = centered.T @ centered / len(x)  # MLE scatter, not effective independent data.
    raw_eigenvalues, eigenvectors = np.linalg.eigh(.5 * (covariance + covariance.T))
    floor = max(absolute_floor**2, relative_floor * max(float(raw_eigenvalues[-1]), 0.))
    eigenvalues = np.maximum(raw_eigenvalues, floor) * std_scale**2
    covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
    covariance = .5 * (covariance + covariance.T)
    np.linalg.cholesky(covariance)
    component = dict(anchor=dict(position=t[anchor_index].tolist(), rotation=anchor_r.tolist()),
                     mean=mean.tolist(), covariance=covariance.tolist())
    report = dict(replica=replica, endpoint_count=len(x), anchor_endpoint_index=anchor_index,
                  anchor_rule="sampled orientation maximizing minimum absolute quaternion dot to all endpoints",
                  minimum_absolute_quaternion_dot=float(np.min(np.abs(delta[:, 3]))),
                  maximum_chart_angle_deg=float(np.degrees(2 * np.arccos(np.min(np.abs(delta[:, 3]))))),
                  maximum_cayley_radius=float(np.max(np.linalg.norm(c, axis=1))),
                  raw_eigenvalues_A2=raw_eigenvalues.tolist(), eigenvalue_floor_before_scale_A2=floor,
                  floored_eigenvalue_count=int(np.count_nonzero(raw_eigenvalues < floor)),
                  final_eigenvalues_A2=eigenvalues.tolist(), final_condition_number=float(eigenvalues[-1] / eigenvalues[0]),
                  endpoint_standard_deviation_scale=std_scale,
                  inference="Covariance describes a correlated/clonal empirical cloud, not an independent equilibrium basin sample")
    return component, report


def model_from_components(components, ell, shape_hash):
    return dict(schema="weighted-pose-mixture-v1", angular_length=ell,
                shape_sha256=shape_hash, coordinate_convention="laboratory",
                anchors=[c["anchor"] for c in components], means=[c["mean"] for c in components],
                covariances=[c["covariance"] for c in components], weights=[1 / len(components)] * len(components))


def quantiles(values):
    values = np.asarray(values)
    assert np.isfinite(values).all()
    return dict(zip(["min", "p05", "median", "p95", "max"], np.quantile(values, [0, .05, .5, .95, 1]).tolist()))


def localized_guides(populations, families, replicas, environment, converted, args, ell, shape_hash):
    """Geometry chooses groups; ancestry enters diagnostics only."""
    poses = [p for replica in replicas for p in populations[replica]]
    source_replicas = np.concatenate([np.full(len(populations[i]), i) for i in replicas])
    source_families = [(replica, family) for replica in replicas for family in families[replica]]
    point_weights = np.concatenate([np.full(len(populations[i]), 1 / (len(replicas) * len(populations[i]))) for i in replicas])
    t, q, _ = arrays(poses)
    angle = 2 * np.arccos(np.clip(np.abs(q @ q.T), 0., 1.))
    distance2 = cdist(t, t, "sqeuclidean") / args.translation_cover_radius**2
    distance2 += (angle / np.radians(args.rotation_cover_radius_deg))**2
    np.fill_diagonal(distance2, 0.)
    medoids = [0]
    nearest = distance2[:, 0].copy()
    labels = np.zeros(len(poses), dtype=int)
    while nearest.max() > 1.:
        index = int(np.argmax(nearest))
        medoids.append(index)
        improved = distance2[:, index] < nearest
        labels[improved] = len(medoids) - 1
        nearest[improved] = distance2[improved, index]
    sizes = np.bincount(labels, minlength=len(medoids))
    retained = np.flatnonzero(sizes >= args.minimum_cluster_size)
    if len(retained) > args.maximum_local_components:
        raise ValueError(f"Geometric cover has {len(retained)} qualifying local components, exceeding cap {args.maximum_local_components}; no clusters were truncated")
    if not len(retained):
        raise ValueError("Geometric cover produced no qualifying local components")
    components, details, weights = [], [], []
    for group in retained:
        ids = np.flatnonzero(labels == group)
        points = [poses[i] for i in ids]
        component, fit = fit_population(points, f"local-{group}", ell, args.coordinate_std_floor,
                                        args.relative_eigenvalue_floor, args.endpoint_std_scale)
        ancestry = Counter(source_families[i] for i in ids)
        details.append(dict(cover_group=int(group), endpoint_count=len(ids),
                            cover_medoid_endpoint_index=int(medoids[group]),
                            maximum_normalized_radius=float(np.sqrt(nearest[ids]).max()),
                            proposal_mass=float(point_weights[ids].sum()),
                            source_population_counts=dict(Counter(str(source_replicas[i]) for i in ids)),
                            distinct_ancestry_families=len(ancestry), largest_ancestry_fraction=max(ancestry.values()) / len(ids),
                            chart_fit=fit))
        components.append(component)
        weights.append(float(point_weights[ids].sum()))
    laboratory = model_from_components(components, ell, shape_hash)
    model = convert_model(laboratory, environment["fixed_poses"][0])
    model["weights"] = weights
    fallback = ~np.isin(labels, retained)
    fallback_details = []
    for replica in replicas:
        ids = fallback & (source_replicas == replica)
        if not ids.any():
            continue
        for field in ("anchors", "means", "covariances"):
            model[field].append(copy.deepcopy(converted[field][replica]))
        mass = float(point_weights[ids].sum())
        model["weights"].append(mass)
        fallback_details.append(dict(source_population=replica, endpoint_count=int(ids.sum()),
                                     proposal_mass=mass, meaning="Full-population Gaussian supplies coverage for small geometric groups"))
    assert abs(sum(model["weights"]) - 1.) < 1e-12
    # Check full normalized local sub-mixture in its laboratory and relative
    # charts before adding fallback components from already validated charts.
    laboratory["weights"] = (np.asarray(weights) / sum(weights)).tolist()
    relative_local = convert_model(laboratory, environment["fixed_poses"][0])
    lab_density, rel_density = Density(laboratory), Density(relative_local)
    rng = np.random.default_rng(731837 + len(poses) + replicas[0])
    check_poses = [p for k in range(len(components)) for p in lab_density.draw_component(rng, k, 2)]
    a = lab_density.evaluate(check_poses)[0]
    b = rel_density.evaluate(relative_poses(check_poses, environment["fixed_poses"][0]))[0]
    conversion_error = float(np.max(np.abs(a - b)))
    assert conversion_error < 2e-6, conversion_error
    report = dict(metric="sqrt((center_distance/translation_radius)^2+(SO3_geodesic_angle/rotation_radius)^2)",
                  translation_radius_A=args.translation_cover_radius, rotation_radius_deg=args.rotation_cover_radius_deg,
                  cover_rule="Deterministic farthest-first observed medoids, starting at endpoint zero; nearest-medoid assignment",
                  all_endpoint_count=len(poses), covering_medoids=len(medoids),
                  maximum_normalized_cover_radius=float(np.sqrt(nearest).max()),
                  minimum_cluster_size=args.minimum_cluster_size, qualifying_local_components=len(retained),
                  maximum_local_components=args.maximum_local_components,
                  fallback_endpoint_count=int(fallback.sum()), fallback_endpoint_fraction=float(np.mean(fallback)),
                  fallback_proposal_fraction=float(point_weights[fallback].sum()),
                  local_charts=details, fallback_components=fallback_details,
                  local_chart_conversion_maximum_log_density_error=conversion_error,
                  ancestry_role="Diagnostic only; never used to define geometric groups or physical weights")
    return model, report


def candidate_audit(model, environment, shape, count, seed):
    """Direct atom-sphere distances, using a KD tree only to reject far pairs."""
    rng = np.random.default_rng(seed)
    density = Density(model)
    centers = np.asarray([a["center"] for a in shape["atoms"]])
    radii = np.asarray([a["radius"] for a in shape["atoms"]])
    fixed_t, _, fixed_r = arrays(environment["fixed_poses"])
    fixed_centers = centers @ fixed_r[0].T + fixed_t[0]
    fixed_tree = cKDTree(fixed_centers)
    result = []
    for k in range(len(model["weights"])):
        relative = density.draw_component(rng, k, count)
        t, _, r = arrays(relative)
        t = t @ fixed_r[0].T + fixed_t[0]
        r = fixed_r[0] @ r
        capture, hard = 0, 0
        for p, rotation in zip(t, r):
            if np.linalg.norm(p - environment["capture_center"]) > environment["capture_radius"]:
                continue
            capture += 1
            moved = centers @ rotation.T + p
            neighbors = fixed_tree.query_ball_point(moved, radii + radii.max())
            clash = False
            for i, js in enumerate(neighbors):
                if js and np.any(np.sum((moved[i] - fixed_centers[js])**2, axis=1) < (radii[i] + radii[js])**2):
                    clash = True
                    break
            hard += int(not clash)
        result.append(dict(component=k, draws=count, capture_valid=capture, hard_and_capture_valid=hard,
                           physical_probability_estimate=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=int, choices=[0, 1], default=0)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--base-model", type=Path, default=BASE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--base-weight", type=float, default=.5)
    parser.add_argument("--coordinate-std-floor", type=float, default=.02,
                        help="Absolute eigenvalue floor is this squared in [Angstrom, ell*Cayley] coordinates")
    parser.add_argument("--relative-eigenvalue-floor", type=float, default=1e-8)
    parser.add_argument("--endpoint-std-scale", type=float, default=1.,
                        help="Multiply newly fitted Gaussian standard deviations after flooring; base atlas unchanged")
    parser.add_argument("--candidate-probes-per-component", type=int, default=32)
    parser.add_argument("--candidate-probes-total", type=int,
                        help="Approximate total candidate budget per model; local mode defaults to512")
    parser.add_argument("--diagnostic-uniform-probability", type=float, default=.1,
                        help="Uniform cube/Haar fraction used only in reported full-q coverage diagnostics")
    parser.add_argument("--local", action="store_true", help="Fit bounded geometric groups; retain full-population fallbacks for small groups")
    parser.add_argument("--translation-cover-radius", type=float, default=4.)
    parser.add_argument("--rotation-cover-radius-deg", type=float, default=15.)
    parser.add_argument("--minimum-cluster-size", type=int, default=8)
    parser.add_argument("--maximum-local-components", type=int, default=128)
    args = parser.parse_args()
    if not 0 < args.base_weight < 1:
        parser.error("base-weight must lie strictly between zero and one")
    if min(args.coordinate_std_floor, args.endpoint_std_scale) <= 0 or not 0 <= args.relative_eigenvalue_floor < 1:
        parser.error("invalid covariance floors or scale")
    if args.candidate_probes_per_component < 1:
        parser.error("positive candidate-probes-per-component required")
    if not 0 < args.diagnostic_uniform_probability < 1:
        parser.error("diagnostic-uniform-probability must lie in (0,1)")
    if args.translation_cover_radius <= 0 or not 0 < args.rotation_cover_radius_deg <= 180 or args.minimum_cluster_size < 2 or args.maximum_local_components < 1:
        parser.error("invalid local cover controls")
    if args.candidate_probes_total is not None and args.candidate_probes_total < 1:
        parser.error("positive candidate-probes-total required")
    default_output = "smc-normalizer-local-atlases" if args.local else "smc-normalizer-atlases"
    out = (args.out or ROOT / f"runs/{default_output}/site{args.site}").resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use a fresh empty output directory")
    source = args.source.resolve()
    manifest = read(source / "manifest.json")
    selected = sorted([j for j in manifest["jobs"] if j["environment_id"] == f"site{args.site}-m1"
                       and j["basin"] == "other_adsorbed" and j["depletant_radius_A"] == 1.5
                       and j["activity_A_minus3"] == .035], key=lambda j: j["replica"])
    assert [j["replica"] for j in selected] == list(range(8)), "Need the eight specified independent SMC populations"
    base = read(args.base_model)
    assert base["coordinate_convention"] == "anchor-body-relative"
    assert len(base["weights"]) == 5, "This control explicitly retains the previously frozen five-component atlas"
    assert abs(sum(base["weights"]) - 1) < 1e-12
    ell = float(base["angular_length"])
    out.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "models"):
        (out / name).mkdir()
    shutil.copy2(args.base_model, out / "inputs/base-model.json")
    shutil.copy2(source / "manifest.json", out / "inputs/source-manifest.json")
    shutil.copy2(__file__, out / "inputs/prepare_smc_normalizer_atlas.py")
    populations, families, components, fit_reports, provenance = {}, {}, [], [], []
    environment, shape_hash = None, None
    for job in selected:
        config_path = source / job["config"]
        assert sha(config_path) == job["config_sha256"]
        config = read(config_path)
        assert config["depletant_radius"] == 1.5 and config["reservoir_density"] == .035
        env_path = Path(config["environment"])
        shape_path = Path(config["shape"])
        env_path = env_path if env_path.is_absolute() else source / env_path
        shape_path = shape_path if shape_path.is_absolute() else source / shape_path
        if environment is None:
            environment = read(env_path)
            assert len(environment["fixed_poses"]) == 1
            shape_hash = sha(shape_path)
            assert base["shape_sha256"] == shape_hash
            shutil.copy2(env_path, out / "inputs/environment.json")
            shutil.copy2(shape_path, out / "inputs/shape.json")
        assert sha(env_path) == sha(out / "inputs/environment.json") and sha(shape_path) == shape_hash
        summary_path = source / job["directory"] / "summary.json"
        summary = read(summary_path)
        assert summary["complete"] and not summary["zero_estimate"]
        poses = [p["pose"] for p in summary["final_particles"]]
        assert len(poses) == summary["population"]
        replica = job["replica"]
        populations[replica] = poses
        families[replica] = [p["family_id"] for p in summary["final_particles"]]
        component, report = fit_population(poses, replica, ell, args.coordinate_std_floor,
                                           args.relative_eigenvalue_floor, args.endpoint_std_scale)
        family_counts = Counter(families[replica])
        report.update(distinct_families=len(family_counts), largest_family_fraction=max(family_counts.values()) / len(poses),
                      endpoint_q=quantiles([p["q"] for p in summary["final_particles"]]))
        components.append(component)
        fit_reports.append(report)
        shutil.copy2(config_path, out / f"inputs/population-{replica}-config.json")
        shutil.copy2(summary_path, out / f"inputs/population-{replica}-summary.json")
        provenance.append(dict(replica=replica, seed=config["seed"], job=job["id"],
                               config=str(config_path), config_sha256=sha(config_path),
                               summary=str(summary_path), summary_sha256=sha(summary_path),
                               use="poses only; logZ and family weights never enter the fit or guide weights"))
    lab = model_from_components(components, ell, shape_hash)
    converted = convert_model(lab, environment["fixed_poses"][0])
    lab_density, relative_density = Density(lab), Density(converted)
    rng = np.random.default_rng(926710 + args.site)
    conversion = []
    for k in range(8):
        poses = lab_density.draw_component(rng, k, 4)
        a = lab_density.evaluate(poses)[0]
        b = relative_density.evaluate(relative_poses(poses, environment["fixed_poses"][0]))[0]
        error = float(np.max(np.abs(a - b)))
        assert error < 2e-6, (k, error)
        conversion.append(dict(replica=k, independent_gaussian_draws=4, maximum_full_log_density_error=error))
    write(out / "inputs/new-laboratory-charts.json", lab)
    write(out / "inputs/new-relative-charts.json", converted)
    all_poses = [p for replica in range(8) for p in populations[replica]]
    relative_all = relative_poses(all_poses, environment["fixed_poses"][0])
    base_log = Density(base).evaluate(relative_all)[0]
    epsilon = args.diagnostic_uniform_probability
    log_uniform = np.log(epsilon) - 3 * np.log(2 * environment["capture_radius"])
    base_full_log = np.logaddexp(log_uniform, np.log1p(-epsilon) + base_log)
    rows, coverage, local_reports = [], {}, {}
    shape = read(out / "inputs/shape.json")
    for name, replicas in [("fold-a", list(range(4))), ("fold-b", list(range(4, 8))), ("all", list(range(8)))]:
        if args.local:
            guides, local_reports[name] = localized_guides(populations, families, replicas, environment,
                converted, args, ell, shape_hash)
        else:
            guides = {field: [copy.deepcopy(converted[field][k]) for k in replicas]
                      for field in ("anchors", "means", "covariances")}
            guides["weights"] = [1 / len(replicas)] * len(replicas)
        model = dict(schema="weighted-pose-mixture-v1", angular_length=ell,
                     coordinate_convention="anchor-body-relative", shape_sha256=shape_hash)
        for field in ("anchors", "means", "covariances"):
            model[field] = copy.deepcopy(base[field]) + copy.deepcopy(guides[field])
        model["weights"] = (args.base_weight * np.asarray(base["weights"])).tolist() + ((1 - args.base_weight) * np.asarray(guides["weights"])).tolist()
        model["proposal_provenance"] = dict(kind="frozen SMC-cloud importance guide; no physical mixture interpretation",
            base_model_sha256=sha(args.base_model), fitted_replicas=replicas,
            base_total_weight=args.base_weight, population_weight_rule="equal selected-population input weights, unrelated to SMC normalizers",
            local_geometry_cover=args.local, local_fit=local_reports.get(name),
            source_inputs="../inputs", native_informed_base=True,
            configured_site=args.site, actual_depletant_radius_A=1.5, actual_activity_A_minus3=.035)
        write(out / f"models/{name}.json", model)
        density = Density(model)
        log_q, norms, logs = density.evaluate(relative_all)
        full_log_q = np.logaddexp(log_uniform, np.log1p(-epsilon) + log_q)
        own = np.repeat([replica in replicas for replica in range(8)], [len(populations[i]) for i in range(8)])
        result = {}
        for label, mask in [("fit_clouds", own), ("other_fold_clouds", ~own)]:
            if not np.any(mask):
                continue
            posterior_new = np.exp(logsumexp(logs[mask, 5:], axis=1) - log_q[mask])
            result[label] = dict(endpoint_count=int(mask.sum()), gaussian_log_density=quantiles(log_q[mask]),
                gain_over_original_base_log_density=quantiles(log_q[mask] - base_log[mask]),
                full_proposal_log_density=quantiles(full_log_q[mask]),
                full_proposal_gain_over_base=quantiles(full_log_q[mask] - base_full_log[mask]),
                full_proposal_over_uniform_threshold_fractions={str(factor): float(np.mean(full_log_q[mask] - log_uniform >= np.log(factor))) for factor in [10, 100, 1000]},
                minimum_new_chart_whitened_radius=quantiles(np.min(norms[mask, 5:], axis=1)),
                new_component_posterior_fraction_gt_0_9=float(np.mean(posterior_new > .9)),
                interpretation="Retrospective cloud coverage only; no unobserved-space or equilibrium guarantee")
        result["per_source_population"] = []
        start = 0
        for replica in range(8):
            end = start + len(populations[replica])
            result["per_source_population"].append(dict(replica=replica, fitted=replica in replicas,
                gaussian_log_density=quantiles(log_q[start:end]), minimum_new_chart_whitened_radius=quantiles(np.min(norms[start:end, 5:], axis=1))))
            start = end
        budget = args.candidate_probes_total or (512 if args.local else None)
        probes = int(np.ceil(budget / len(model["weights"]))) if budget else args.candidate_probes_per_component
        result["independent_candidate_geometry"] = candidate_audit(model, environment, shape, probes, 713462 + args.site * 10 + len(replicas) + replicas[0])
        coverage[name] = result
        rows.append(dict(name=name, path=f"models/{name}.json", sha256=sha(out / f"models/{name}.json"),
                         component_count=len(model["weights"]), fitted_replicas=replicas))
    report = dict(schema=1, site=args.site, target=dict(depletant_radius_A=1.5, activity_A_minus3=.035,
        capture_center=environment["capture_center"], capture_radius=environment["capture_radius"]),
        note="Environment-file bath fields are historical metadata; source configuration bath values are authoritative",
        source=str(source), base_model=str(args.base_model.resolve()), shape_sha256=shape_hash,
        settings=dict(base_weight=args.base_weight, coordinate_std_floor_A=args.coordinate_std_floor,
                      relative_eigenvalue_floor=args.relative_eigenvalue_floor, endpoint_std_scale=args.endpoint_std_scale,
                      angular_length=ell, candidate_probes_per_component=args.candidate_probes_per_component,
                      diagnostic_uniform_probability=epsilon, local_geometry_cover=args.local,
                      translation_cover_radius_A=args.translation_cover_radius, rotation_cover_radius_deg=args.rotation_cover_radius_deg,
                      minimum_cluster_size=args.minimum_cluster_size, maximum_local_components=args.maximum_local_components),
        populations=provenance, fits=fit_reports, localized_fits=local_reports, frame_conversion=conversion, models=rows, coverage=coverage,
        guarantees=["Every Gaussian is normalized in translation times proper normalized Haar measure",
                    "The runtime normalizer must retain its positive uniform cube/Haar branch and evaluate the complete mixture density",
                    "No production samples or weight estimates enter fitting"],
        limitations=["SMC endpoints share ancestry and are not independent equilibrium fit data",
                     "One Gaussian per population is a proposal choice, not a discovered physical basin",
                     "Both folds are retrospective guides; held-out here only means excluded from that fold's new components",
                     "Guide coverage at known snapshots does not establish complete physical region coverage"])
    write(out / "report.json", report)
    write(out / "input-hashes.json", {str(p.relative_to(out)): sha(p) for p in sorted((out / "inputs").iterdir())})
    print(json.dumps(dict(out=str(out), models=rows, max_conversion_error=max(c["maximum_full_log_density_error"] for c in conversion),
                          maximum_condition_number=max(f["final_condition_number"] for f in fit_reports)), indent=2))


if __name__ == "__main__":
    main()
