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


class SourceDensity:
    def __init__(self, spec, source_pose, anchor_pose):
        self.ell = float(spec['angular_length'])
        self.covariance = expected_covariance(spec)
        self.lower = np.linalg.cholesky(self.covariance)
        self.logdet = float(np.log(np.diag(self.lower)).sum())
        self.anchor = anchor_pose
        self.center = relative(source_pose, anchor_pose)
        self.center_rotation = rotation(self.center)

    def evaluate(self, value):
        value = relative(value, self.anchor)
        q = Rotation.from_matrix(rotation(value)@self.center_rotation.T).as_quat()
        if q[3] == 0:
            return -math.inf
        u = q[:3]/q[3]
        x = np.r_[np.asarray(value['position'])-self.center['position'], self.ell*u]
        standardized = solve_triangular(self.lower, x, lower=True)
        log_normal = -.5*float(standardized@standardized)-3*math.log(2*math.pi)-self.logdet
        log_volume = -3*math.log(self.ell)-2*math.log(math.pi)-2*math.log1p(float(u@u))
        result = log_normal-log_volume
        require(math.isfinite(result), 'Unrepresentable independent source density')
        return result

    def decode(self, latent):
        z = np.asarray(latent, float)
        require(z.shape == (6,) and np.isfinite(z).all(), 'Invalid source latent')
        x = self.lower@z
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
