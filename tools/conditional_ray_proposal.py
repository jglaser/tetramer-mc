"""Independent density for exact radial guides inside a conditional ellipsoid.

Only the radial conditional law changes. Orientations and ray directions retain
their original uniform-six-ball marginals. Empty guide intervals fall back on
the original ray, without another orientation or direction draw.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation
from entry_shell_proposal import require, finite_number

SCHEMA = 'defensive-conditional-ray-guide-v1'
POPULATION_SCHEMA = 'importance-latent-region-normalizer-v3'
PROPOSAL_KIND = 'conditional-ray-interval-mixture'


def union(intervals):
    result = []
    for lo, hi in sorted(intervals):
        if hi <= lo:
            continue
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((lo, hi))
    return result


def difference(outer, inner):
    result = []
    for lo, hi in union(outer):
        cursor = lo
        for a, b in union(inner):
            if b <= cursor or a >= hi:
                continue
            if a > cursor:
                result.append((cursor, min(a, hi)))
            cursor = max(cursor, b)
            if cursor >= hi:
                break
        if cursor < hi:
            result.append((cursor, hi))
    return result


def cubic_mass(intervals):
    return sum((b-a)*(b*b+a*b+a*a) for a, b in intervals)


class ConditionalRayGuide:
    schema = SCHEMA
    population_schema = POPULATION_SCHEMA
    proposal_kind = PROPOSAL_KIND
    branch = 'conditional-ray'

    def __init__(self, guide, region, region_sha256):
        require(set(guide) == {'schema', 'region_sha256', 'defensive_uniform_shell_probability',
                            'inner_radius', 'widths', 'interfaces'}, 'Unknown ray guide fields')
        require(guide['schema'] == SCHEMA and guide['region_sha256'] == region_sha256, 'Wrong ray guide or region')
        require(isinstance(region_sha256, str) and len(region_sha256) == 64 and
                all(c in '0123456789abcdefABCDEF' for c in region_sha256), 'Invalid region hash')
        self.alpha = guide['defensive_uniform_shell_probability']
        self.inner = guide['inner_radius']
        self.widths = guide['widths']
        require(finite_number(self.alpha) and 0 < self.alpha <= 1, 'Positive uniform floor required')
        require(finite_number(self.inner) and self.inner >= 0, 'Invalid guide inner radius')
        require(isinstance(self.widths, list) and self.widths and
                all(finite_number(w) and w > 0 and self.inner+w > self.inner and self.inner+w <= math.sqrt(np.finfo(float).max) for w in self.widths), 'Invalid widths')
        self.count = len(self.widths)
        require(region.get('minimum_mahalanobis_radius', 0.) == 0., 'Ray guide requires a full latent ball')
        self.radius = region['mahalanobis_radius']
        require(finite_number(self.radius) and self.radius > 0, 'Invalid latent radius')
        self.log_volume = 3*math.log(math.pi)+6*math.log(self.radius)-math.log(6)
        chart = region['gaussian_chart']
        require(chart['weights'] == [1.] and chart['coordinate_convention'] == 'anchor-body-relative'
                and 'base_model' not in chart, 'One ordinary chart required')
        self.mean = np.asarray(chart['means'][0], float)
        cov = np.asarray(chart['covariances'][0], float)
        require(self.mean.shape == (6,) and cov.shape == (6, 6) and np.isfinite(cov).all()
                and np.isfinite(self.mean).all() and np.allclose(cov, cov.T, rtol=1e-12, atol=1e-14), 'Invalid covariance')
        self.lower = np.linalg.cholesky(cov)
        self.log_det = np.log(np.diag(self.lower)).sum()
        self.angular_lower = np.linalg.cholesky(cov[3:, 3:])
        self.gain = np.linalg.solve(cov[3:, 3:], cov[3:, :3]).T
        conditional = cov[:3, :3]-self.gain@cov[3:, :3]
        self.conditional_lower = np.linalg.cholesky((conditional+conditional.T)/2)
        self.ell = chart['angular_length']
        require(finite_number(self.ell) and self.ell > 0, 'Invalid angular scale')
        self.anchor_t = np.asarray(chart['anchors'][0]['position'], float)
        self.anchor_r = np.asarray(chart['anchors'][0]['rotation'], float)
        q = np.asarray(region['fixed_neighbor']['orientation'], float)
        self.fixed_t = np.asarray(region['fixed_neighbor']['position'], float)
        require(q.shape == (4,) and np.isfinite(q).all() and abs(np.linalg.norm(q)-1) < 1e-9, 'Invalid fixed rotation')
        self.fixed_r = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        require(self.anchor_t.shape == self.fixed_t.shape == (3,) and self.anchor_r.shape == (3, 3)
                and np.isfinite(self.anchor_t).all() and np.isfinite(self.fixed_t).all()
                and np.isfinite(self.anchor_r).all() and np.allclose(self.anchor_r.T@self.anchor_r, np.eye(3), atol=1e-10)
                and abs(np.linalg.det(self.anchor_r)-1) < 1e-10, 'Invalid chart anchor')
        interfaces = guide['interfaces']
        require(isinstance(interfaces, list) and interfaces, 'Missing interfaces')
        self.interfaces = []
        for interface in interfaces:
            require(set(interface) == {'moving_members', 'target_world_members'}, 'Unknown interface fields')
            moving, target = (np.asarray(interface[k], float) for k in ('moving_members', 'target_world_members'))
            require(moving.ndim == 2 and moving.shape[1:] == (3,) and len(moving) > 0
                    and moving.shape == target.shape and np.isfinite(moving).all() and np.isfinite(target).all(), 'Invalid member arrays')
            self.interfaces.append((moving, target))

    def coordinates(self, latents):
        u = np.asarray(latents, float)
        require(u.ndim == 2 and u.shape[1] == 6 and np.isfinite(u).all(), 'Invalid latent coordinates')
        return u@self.lower.T+self.mean

    def world(self, latents):
        x = self.coordinates(latents)
        c = x[:, 3:]/self.ell
        rotation = self.fixed_r@Rotation.from_quat(np.column_stack((c, np.ones(len(c))))).as_matrix()@self.anchor_r
        translation = (x[:, :3]+self.anchor_t)@self.fixed_r.T+self.fixed_t
        return translation, rotation, x

    def rays(self, latents):
        _, rotations, x = self.world(latents)
        a = solve_triangular(self.angular_lower, (x[:, 3:]-self.mean[3:]).T, lower=True).T
        s2 = self.radius**2-np.sum(a*a, axis=1)
        require(np.all(s2 >= -1e-10*(1+self.radius**2)), 'Angular point outside marginal support')
        s = np.sqrt(np.maximum(0., s2))
        mu = self.mean[:3]+(x[:, 3:]-self.mean[3:])@self.gain.T
        v = solve_triangular(self.conditional_lower, (x[:, :3]-mu).T, lower=True).T
        radii = np.linalg.norm(v, axis=1)
        directions = np.zeros_like(v); directions[:, 0] = 1.
        nonzero = radii > 0
        directions[nonzero] = v[nonzero]/radii[nonzero, None]
        origins = (mu+self.anchor_t)@self.fixed_r.T+self.fixed_t
        slopes = directions@self.conditional_lower.T@self.fixed_r.T
        return dict(s=s, radius=radii, direction=directions, mu=mu, origins=origins, slopes=slopes,
                    rotations=rotations, coordinates=x)

    def intervals(self, ray):
        """Vectorized member inequalities; set algebra is independent per ray."""
        n = len(ray['s']); a = np.sum(ray['slopes']**2, axis=1)
        require(np.all(a > 0) and np.isfinite(a).all(), 'Degenerate ray slope')
        ranges = []
        for cutoff in [self.inner]+[self.inner+w for w in self.widths]:
            interfaces = []
            for moving, target in self.interfaces:
                delta = ray['origins'][:, None, :]+np.einsum('nij,kj->nki', ray['rotations'], moving)-target
                b = np.einsum('nkj,nj->nk', delta, ray['slopes'])
                c = np.sum(delta*delta, axis=2)-cutoff**2
                disc = b*b-a[:, None]*c
                root = np.sqrt(np.maximum(0., disc))
                # Product-of-roots form avoids cancellation for small roots.
                large = -b-np.copysign(root, b)
                first = large/a[:, None]
                second = np.divide(c, large, out=(-b/a[:, None]).copy(), where=large != 0)
                lo = np.maximum(0., np.minimum(first, second).max(axis=1))
                hi = np.minimum(ray['s'], np.maximum(first, second).min(axis=1))
                good = (disc > 0).all(axis=1) & (hi > lo)
                interfaces.append((lo, hi, good))
            ranges.append([union([(lo[i], hi[i]) for lo, hi, good in interfaces if good[i]]) for i in range(n)])
        return [[difference(ranges[k+1][i], ranges[0][i]) for k in range(self.count)] for i in range(n)]

    def diagnostics(self, latents):
        ray = self.rays(latents); intervals = self.intervals(ray)
        factors = np.ones((len(intervals), self.count)); masses = np.zeros_like(factors)
        support = np.ones_like(factors, dtype=bool)
        for i, components in enumerate(intervals):
            for k, pieces in enumerate(components):
                mass = cubic_mass(pieces); masses[i, k] = mass
                if mass > 0:
                    support[i, k] = any(lo <= ray['radius'][i] <= hi for lo, hi in pieces)
                    factors[i, k] = ray['s'][i]**3/mass if support[i, k] else 0.
        require(np.isfinite(factors).all(), 'Unrepresentable ray density')
        return dict(ray=ray, intervals=intervals, factors=factors, empty=masses == 0., support=support)

    def log_density(self, latents, shell_valid, log_volume):
        u = np.asarray(latents, float); inside = np.asarray(shell_valid, bool)
        require(inside.shape == (len(u),) and abs(log_volume-self.log_volume) < 1e-10, 'Wrong ball measure')
        result = np.full(len(u), -np.inf)
        if self.alpha == 1:
            result[inside] = -log_volume
        elif inside.any():
            d = self.diagnostics(u[inside])
            result[inside] = -log_volume+np.log(self.alpha+(1-self.alpha)*d['factors'].mean(axis=1))
        return result

    def draw_for_validation(self, rng, count):
        normal = rng.normal(size=(count, 6))
        u = normal/np.linalg.norm(normal, axis=1)[:, None]*(self.radius*rng.random(count)**(1/6))[:, None]
        selected = np.full(count, -1); fallback = [None]*count
        if self.alpha == 1:
            return u, selected, fallback
        active = np.flatnonzero(rng.random(count) >= self.alpha)
        selected[active] = rng.integers(self.count, size=len(active))
        info = self.diagnostics(u); ray = info['ray']
        x = ray['coordinates'].copy()
        for i in active:
            k = selected[i]; pieces = info['intervals'][i][k]
            mass = cubic_mass(pieces); fallback[i] = bool(mass == 0.)
            if mass == 0:
                continue  # Conditional baseline r^3 is already uniform.
            value = rng.random()*mass; radius = None
            for lo, hi in pieces:
                piece = (hi-lo)*(hi*hi+hi*lo+lo*lo)
                if value <= piece:
                    radius = np.cbrt(lo**3+value); break
                value -= piece
            require(radius is not None, 'Invalid interval draw; never retry')
            x[i, :3] = ray['mu'][i]+self.conditional_lower@ray['direction'][i]*radius
        u[active] = solve_triangular(self.lower, (x[active]-self.mean).T, lower=True).T
        return u, selected, fallback
