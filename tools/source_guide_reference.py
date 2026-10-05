"""Independent physical densities for the source-centered conditional guide.

No sampling, geometry queries, or production-kernel imports. Source covariance
enters the map directly, unlike the atlas loader's covariance round trip.
"""
import math

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

from audit_context_candidate_bank import require


def rotation(pose):
    q = np.asarray(pose['orientation'], float)
    require(q.shape == (4,) and np.isfinite(q).all()
            and abs(float(q@q)-1) < 2e-10, 'Invalid source-guide quaternion')
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def pose(position, matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return dict(position=np.asarray(position).tolist(), orientation=q[[3, 0, 1, 2]].tolist())


def relative(value, anchor):
    r = rotation(anchor)
    return pose(r.T@(np.asarray(value['position'])-anchor['position']), r.T@rotation(value))


def compose(anchor, value):
    r = rotation(anchor)
    return pose(np.asarray(anchor['position'])+r@value['position'], r@rotation(value))


def cayley(u):
    # Quaternion (1,u)/sqrt(1+u.u) gives the active Cayley rotation.
    u = np.asarray(u, float)
    q = np.r_[u, 1.]
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q).as_matrix()


def expected_covariance(spec):
    if 'explicit_gaussian' in spec:
        require('translation_sigma' not in spec and 'rotation_scale_deg' not in spec,
                'Explicit source Gaussian cannot declare legacy widths')
        explicit = spec['explicit_gaussian']
        require(isinstance(explicit, dict)
                and set(explicit) == {'schema', 'mean', 'provenance'}
                and explicit['schema'] == 'source-gaussian-v1'
                and isinstance(explicit['provenance'], str)
                and bool(explicit['provenance'].strip()), 'Invalid explicit source Gaussian')
        ell = float(spec['angular_length'])
        mean = np.asarray(explicit['mean'], float)
        require(math.isfinite(ell) and ell > 0
                and mean.shape == (6,) and np.isfinite(mean).all(), 'Invalid explicit source mean or scale')
        actual = np.asarray(spec['covariance'], float)
        require(actual.shape == (6, 6) and np.isfinite(actual).all(), 'Invalid explicit source covariance')
        magnitude = float(np.max(np.abs(actual)))
        require(magnitude > 0
                and np.all(np.abs(actual-actual.T) <= 1e-12*(magnitude+np.abs(actual))),
                'Asymmetric explicit source covariance')
        try:
            np.linalg.cholesky(.5*(actual+actual.T))
        except np.linalg.LinAlgError as error:
            raise ValueError('Explicit source covariance is not positive definite') from error
        return actual
    ell, st, degrees = (float(spec[k]) for k in
                        ('angular_length', 'translation_sigma', 'rotation_scale_deg'))
    require(math.isfinite(ell) and ell > 0 and math.isfinite(st) and st > 0
            and math.isfinite(degrees) and 0 < degrees < 180, 'Invalid declared source width')
    sc = ell*math.tan(degrees*math.pi/360)
    expected = np.diag([st*st]*3+[sc*sc]*3)
    actual = np.asarray(spec['covariance'], float)
    require(actual.shape == (6, 6) and np.isfinite(actual).all()
            and np.allclose(actual, expected, rtol=2e-14, atol=0),
            'Source covariance differs from predeclared diagonal width')
    return actual


def chart_world_pose(spec, original_source):
    """Resolve proposal origin without changing source/contact reference assets."""
    if spec.get('chart_center') is None:
        return original_source
    center = spec['chart_center']
    require(isinstance(center, dict)
            and set(center) == {'schema', 'frame', 'pose', 'provenance'}
            and center['schema'] == 'source-chart-center-v1'
            and center['frame'] == 'saved-spherical-center'
            and isinstance(center['provenance'], str) and bool(center['provenance'].strip()),
            'Invalid chart center schema, physical frame, or provenance')
    value = center['pose']
    require(isinstance(value, dict) and {'position', 'orientation'} <= set(value),
            'Invalid chart center pose')
    position = np.asarray(value['position'], float)
    require(position.shape == (3,) and np.isfinite(position).all(), 'Invalid chart center position')
    rotation(value)  # The same strict finite/unit-quaternion contract as Rust.
    return value


class SourceDensity:
    def __init__(self, spec, source_pose, anchor_pose):
        self.ell = float(spec['angular_length'])
        self.covariance = expected_covariance(spec)
        self.mean = np.asarray(spec.get('explicit_gaussian', {}).get('mean', [0.]*6), float)
        # The implemented strict Cholesky accepts small symmetry roundoff, then
        # uses the symmetric average; no covariance ridge is introduced here.
        self.lower = np.linalg.cholesky(.5*(self.covariance+self.covariance.T))
        self.logdet = float(np.log(np.diag(self.lower)).sum())
        self.anchor = anchor_pose
        self.chart_center_pose = chart_world_pose(spec, source_pose)
        self.center = relative(self.chart_center_pose, anchor_pose)
        self.center_rotation = rotation(self.center)

    def evaluate(self, value):
        value = relative(value, self.anchor)
        q = Rotation.from_matrix(rotation(value)@self.center_rotation.T).as_quat()
        if q[3] == 0:
            return -math.inf
        u = q[:3]/q[3]
        x = np.r_[np.asarray(value['position'])-self.center['position'], self.ell*u]
        standardized = solve_triangular(self.lower, x-self.mean, lower=True)
        log_normal = -.5*float(standardized@standardized)-3*math.log(2*math.pi)-self.logdet
        log_volume = -3*math.log(self.ell)-2*math.log(math.pi)-2*math.log1p(float(u@u))
        result = log_normal-log_volume
        require(math.isfinite(result), 'Unrepresentable independent source density')
        return result

    def decode(self, latent):
        z = np.asarray(latent, float)
        require(z.shape == (6,) and np.isfinite(z).all(), 'Invalid source latent')
        x = self.mean+self.lower@z
        local = pose(np.asarray(self.center['position'])+x[:3],
                     cayley(x[3:]/self.ell)@self.center_rotation)
        return compose(self.anchor, local)


def mixture_log_density(log_u, log_g, log_source, probabilities=(.5, .25, .25)):
    probabilities = np.asarray(probabilities, float)
    require(probabilities.shape == (3,) and np.isfinite(probabilities).all()
            and (probabilities >= 0).all() and abs(float(probabilities.sum())-1) < 1e-14,
            'Unnormalized source-guide mixture')
    values = np.asarray([log_u, log_g, log_source], float)
    require(not np.isnan(values).any() and not np.isposinf(values).any(), 'Invalid component log density')
    logs = np.full(3, -np.inf)
    positive = probabilities > 0
    logs[positive] = np.log(probabilities[positive])+values[positive]
    return float(logsumexp(logs))


def atlas_decode(reference, anchor, branch, latent):
    density = reference.density
    require(type(branch) is int and 0 <= branch < len(density.lower), 'Wrong atlas label')
    z = np.asarray(latent, float)
    require(z.shape == (6,) and np.isfinite(z).all(), 'Invalid atlas latent')
    x = density.mean[branch]+density.lower[branch]@z
    center = density.anchors[branch]
    p = np.asarray(center['position'])+x[:3]
    r = cayley(x[3:]/density.ell)@np.asarray(center['rotation'])
    if density.inverted[branch]:
        p, r = -r.T@p, r.T
    return compose(anchor, pose(p, r))


def pose_error(actual, expected):
    p = float(np.linalg.norm(np.asarray(actual['position'])-expected['position']))
    r = float(np.max(np.abs(rotation(actual)-rotation(expected))))
    require(p < 2e-8 and r < 2e-10, 'Independent latent decoding differs')
    return dict(position_error=p, rotation_matrix_error=r)
