#!/usr/bin/env python3
"""Independent physical density of the untruncated Gaussian regional proposal.

WORLD poses are evaluated relative to the frozen chart anchor. R4 membership
selects only the uniform-ball term and is a reporting flag, never a physical
acceptance rule. No wall, hard-core, capture, native, or Poisson evaluator is
called here. Existing frozen density and chart/Jacobian helpers are unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation

from analyze_latent_region import LatentImportanceGuide
from review_conditional_ray_extremes import jacobian


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _poses(values):
    """Validate scalar-first unit quaternions before SciPy's independent map."""
    positions = np.asarray([p['position'] for p in values], dtype=float)
    quaternions = np.asarray([p['orientation'] for p in values], dtype=float)
    require(positions.shape == (len(values), 3) and quaternions.shape == (len(values), 4), 'Invalid pose array shape')
    require(np.isfinite(positions).all() and np.isfinite(quaternions).all(), 'Nonfinite world pose')
    norms = np.sum(quaternions**2, axis=1)
    require(np.all(abs(norms-1.) <= 1e-8), 'Pose quaternion is not normalized')
    return positions, Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()


@dataclass(frozen=True)
class PhysicalGuideDensity:
    latent: list[float] | None
    in_reference_ball: bool
    log_latent_density: float | None
    log_physical_jacobian: float | None
    log_physical_density: float


class PhysicalLatentGuide:
    """Independent evaluator matching the new Rust physical-guide record fields.

    `evaluate` returns one record; `evaluate_many` returns records for a batch;
    `log_densities` efficiently returns only the batch physical log densities.
    The exact Cayley seam has zero component density and undefined coordinates
    represented by None. A finite nonzero scalar quaternion is never removed by
    an epsilon band. Density zeros use -inf mathematically, not a replacement
    floor. No pose generation or filtering is provided by this evaluator.
    """
    def __init__(self, region: Mapping, guide: Mapping, *, region_sha256: str,
                 expected_shape_sha256: str):
        self.region = json.loads(json.dumps(region))
        require(self.region['shape_sha256'] == expected_shape_sha256, 'Regional shape hash differs')
        require(self.region['mahalanobis_radius'] == 4.
                and self.region.get('minimum_mahalanobis_radius', 0.) == 0., 'Require the full reference R4 ball')
        require(self.region['minimum_original_q'] == 0.
                and self.region.get('minimum_original_q_inclusive', True) is True
                and 'maximum_original_q' not in self.region, 'Require a full R4 source without a q filter')
        self.guide = LatentImportanceGuide(guide, region_sha256)
        self.region_sha256 = region_sha256
        chart = self.region['gaussian_chart']
        require(chart['shape_sha256'] == expected_shape_sha256, 'Chart shape hash differs')
        require(chart.get('coordinate_convention') == 'anchor-body-relative', 'Require anchor-body-relative chart')
        require(len(chart['anchors']) == len(chart['means']) == len(chart['covariances']) == len(chart['weights']) == 1,
                'Exactly one chart component is required')
        require(math.isfinite(chart['weights'][0]) and chart['weights'][0] > 0., 'Invalid chart weight')
        self.ell = float(chart['angular_length'])
        require(math.isfinite(self.ell) and self.ell > 0., 'Invalid angular length')
        self.mean = np.asarray(chart['means'][0], float)
        self.covariance = np.asarray(chart['covariances'][0], float)
        require(self.mean.shape == (6,) and self.covariance.shape == (6, 6)
                and np.isfinite(self.mean).all() and np.isfinite(self.covariance).all(), 'Invalid chart mean/covariance')
        require(np.all(abs(self.covariance-self.covariance.T) <= 1e-12*(1+np.maximum(abs(self.covariance),abs(self.covariance.T)))),
                'Chart covariance is not symmetric')
        # Independent LAPACK factor, matching the existing Python chart audit.
        self.covariance = .5*(self.covariance+self.covariance.T)
        self.lower = np.linalg.cholesky(self.covariance)
        # Chart uses the symmetric average after its tolerance check. Give the
        # unchanged Jacobian helper that same matrix, including near-symmetric
        # inputs; otherwise its raw NumPy lower triangle would define another L.
        chart['covariances'][0] = self.covariance.tolist()
        self.log_det = float(np.log(np.diag(self.lower)).sum())
        fixed_t, fixed_r = _poses([self.region['fixed_neighbor']])
        self.fixed_t, self.fixed_r = fixed_t[0], fixed_r[0]
        self.physical_fixed_neighbors = self.region.get('physical_fixed_neighbors', [self.region['fixed_neighbor']])
        require(self.physical_fixed_neighbors and self.region['fixed_neighbor'] in self.physical_fixed_neighbors,
                'Missing physical chart anchor')
        _poses(self.physical_fixed_neighbors)
        capture_center = np.asarray(self.region['capture_center'], float)
        capture_radius = float(self.region['capture_radius'])
        require(capture_center.shape == (3,) and np.isfinite(capture_center).all()
                and math.isfinite(capture_radius) and capture_radius > 0., 'Invalid source capture metadata')
        anchor = chart['anchors'][0]
        self.anchor_t, self.anchor_r = np.asarray(anchor['position'], float), np.asarray(anchor['rotation'], float)
        require(self.anchor_t.shape == (3,) and self.anchor_r.shape == (3, 3)
                and np.isfinite(self.anchor_t).all() and np.isfinite(self.anchor_r).all(), 'Invalid chart anchor')
        require(np.max(abs(self.anchor_r.T@self.anchor_r-np.eye(3))) < 1e-8
                and abs(np.linalg.det(self.anchor_r)-1.) < 1e-8, 'Chart anchor rotation is not proper')
        self.log_volume = 3*math.log(math.pi)+6*math.log(4.)-math.log(6.)
        self.sources = {}

    @classmethod
    def from_files(cls, region_path, guide_path, *, expected_shape_sha256):
        region_path, guide_path = Path(region_path).resolve(), Path(guide_path).resolve()
        region_raw, guide_raw = region_path.read_bytes(), guide_path.read_bytes()
        digest = hashlib.sha256(region_raw).hexdigest()
        result = cls(json.loads(region_raw), json.loads(guide_raw), region_sha256=digest,
                     expected_shape_sha256=expected_shape_sha256)
        result.sources = {str(region_path): digest, str(guide_path): hashlib.sha256(guide_raw).hexdigest()}
        return result

    def _arrays(self, poses: Sequence[Mapping]):
        n = len(poses)
        if not n:
            return np.empty((0, 6)), np.empty(0, bool), np.empty(0, bool), np.empty(0), np.empty(0), np.empty(0)
        positions, rotations = _poses(poses)
        relative_t = (positions-self.fixed_t)@self.fixed_r-self.anchor_t
        relative_r = self.fixed_r.T@rotations@self.anchor_r.T
        quaternion = Rotation.from_matrix(relative_r).as_quat()
        seam = quaternion[:, 3] == 0.
        active = ~seam
        latent = np.full((n, 6), np.nan)
        in_ball = np.zeros(n, bool)
        logj = np.full(n, np.nan)
        logq = np.full(n, -np.inf)
        physical = np.full(n, -np.inf)
        if active.any():
            # Same exact chart algebra as chart_coordinates, without that
            # diagnostic's >1e-12 seam guard: every nonzero scalar is retained.
            coordinates = np.column_stack((relative_t[active], self.ell*quaternion[active, :3]/quaternion[active, 3, None]))
            u = solve_triangular(self.lower, (coordinates-self.mean).T, lower=True).T
            require(np.isfinite(u).all(), 'Nonfinite inverse chart coordinates')
            latent[active] = u
            bounded = np.all(abs(u) <= 4., axis=1)
            inside = np.zeros(len(u), bool)
            inside[bounded] = np.einsum('ij,ij->i', u[bounded], u[bounded]) <= 16.
            in_ball[active] = inside
            # Reuse the independent, previously audited physical Jacobian.
            with np.errstate(over='ignore', invalid='ignore'):
                ordinary_logj = jacobian(u, self.region)
            # Beyond the squared-Cayley FP64 range, the same Jacobian uses
            # 1+|c|²=1/w². Pure-uniform exterior queries remain legitimate zeros.
            stable_logj = (self.log_det-3*math.log(self.ell)-2*math.log(math.pi)
                           +4*np.log(abs(quaternion[active, 3])))
            logj[active] = np.where(np.isfinite(ordinary_logj), ordinary_logj, stable_logj)
            require(np.isfinite(logj[active]).all(), 'Nonfinite physical Jacobian')
            with np.errstate(over='ignore', invalid='ignore'):
                logq[active] = self.guide.log_density(u, in_ball[active], self.log_volume)
            allowed_zero = (self.guide.alpha == 1.) & ~in_ball[active]
            require(np.all(np.isfinite(logq[active]) | (allowed_zero & np.isneginf(logq[active]))),
                    'Unrepresentable positive Gaussian density')
            physical[active] = logq[active]-logj[active]
            require(np.all(np.isfinite(physical[active]) | np.isneginf(physical[active])), 'Nonfinite physical density')
        return latent, seam, in_ball, logq, logj, physical

    def evaluate_many(self, poses: Sequence[Mapping]) -> list[PhysicalGuideDensity]:
        latent, seam, in_ball, logq, logj, physical = self._arrays(poses)
        return [PhysicalGuideDensity(None if seam[i] else latent[i].tolist(), bool(in_ball[i]),
                    None if seam[i] else float(logq[i]), None if seam[i] else float(logj[i]), float(physical[i]))
                for i in range(len(poses))]

    def evaluate(self, world_pose: Mapping) -> PhysicalGuideDensity:
        return self.evaluate_many([world_pose])[0]

    def log_densities(self, world_poses: Sequence[Mapping]) -> np.ndarray:
        return self._arrays(world_poses)[-1]


def half_mixture_log_density(log_vessel, log_regional_physical):
    """Complete .5*p_vessel+.5*q_D/J_D at the SAME world pose.

    Inputs already include every component/anchor within their branch. The
    generation branch is deliberately absent. Either component may have zero
    density; two zeros return -inf. No target/support indicator is applied.
    """
    vessel, regional = np.broadcast_arrays(np.asarray(log_vessel, float), np.asarray(log_regional_physical, float))
    for value in (vessel, regional):
        require(np.all(np.isfinite(value) | np.isneginf(value)), 'Density log must be finite or negative infinity')
    result = np.logaddexp(vessel, regional)-math.log(2.)
    return float(result) if result.ndim == 0 else result
