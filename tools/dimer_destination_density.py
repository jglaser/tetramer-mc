"""Read-only reconstruction of the frozen DockingProposal constructor and maps.

The constructor exports the initial scalar factor's L L^T, then factors that
covariance separately for scoring and for the map. Those binary64 factors need
not agree. Keep both laws visible; a successful finite witness is not an exact
closure certificate or a bound on unseen poses. No random-number API is used.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation

from normalizer_proposal_density import scalar_cholesky, symmetric_covariance
from prepare_smc_normalizer_atlas import Density, unwrap_proposal_model


SOURCE_EXCERPTS = (
    ('src/docking.rs', 'pub fn new(\n        model: FrozenRelativePoseProposal,',
     '    pub fn with_anchor_index',
     'b8fb3aa8546f38902cd707b7fb96a487ecfce55cfbdf859f8dfd8aba4947d63a'),
    ('src/proposal.rs', '    pub fn component_parameters(',
     '    /// Replace Gaussian parameters',
     '48c6c1cb0146e16f3a4f859cfa9d3334072ebe6ad49f8d7aa883887824d5a948'),
    ('src/proposal.rs', 'fn prepare_cholesky(', '/// The finite Cayley coordinate',
     '09ddf40beec7203b00ef5e837bfcb359bd8bd3a265034644ba9d03875d8887ea'),
    ('src/basin_involution.rs', 'fn cholesky(', 'fn chart_parameters_valid(',
     'e06c2385578d028b4a68cb912f64011c63b62ea8f441910dc13945731bb6c238'),
    ('src/basin_involution.rs', '    pub fn encode(',
     '\n}\n\n/// The correlated latent map',
     '27ddbe575ec34808dbe4867dd20e4f55aaa8545e33d1ea8a454b9112593b5d3d'),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordered_sum(values):
    """Binary64 left-to-right sum, without NumPy/BLAS or compensated summation."""
    total = 0.
    for value in values:
        total += float(value)
    return total


def exported_covariance(lower):
    lower = np.asarray(lower, float)
    require(lower.shape == (6, 6) and np.isfinite(lower).all(), 'Invalid factor')
    return np.asarray([[ordered_sum(float(lower[i, k])*float(lower[j, k])
                                   for k in range(min(i, j)+1))
                        for j in range(6)] for i in range(6)])


def map_cholesky(covariance):
    """FixedBasinInvolution: subtract the ordered sum once at each pivot."""
    _, symmetric = symmetric_covariance(covariance)
    lower = np.zeros((6, 6))
    for i in range(6):
        for j in range(i+1):
            residual = float(symmetric[i, j])-ordered_sum(
                float(lower[i, k])*float(lower[j, k]) for k in range(j))
            if i == j:
                require(math.isfinite(residual) and residual > 0,
                        'Map covariance is not positive definite')
                lower[i, j] = math.sqrt(residual)
            else:
                lower[i, j] = residual/float(lower[j, j])
            require(math.isfinite(lower[i, j]), 'Unrepresentable map factor')
    return lower


def bind_source_bundle(path):
    raw = Path(path).read_bytes()
    bundle = json.loads(raw)
    checked = []
    for name, start, end, expected in SOURCE_EXCERPTS:
        entry = bundle['files'][name]
        source = entry['text']
        require(hashlib.sha256(source.encode()).hexdigest() == entry['sha256'],
                'Corrupt embedded source: '+name)
        first = source.index(start)
        digest = hashlib.sha256(source[first:source.index(end, first)].encode()).hexdigest()
        require(digest == expected, 'Unknown constructor/map arithmetic: '+name)
        checked.append(dict(file=name, starts_with=start, sha256=digest))
    return dict(source_bundle_sha256=hashlib.sha256(raw).hexdigest(), excerpts=checked)


def pose_arrays(pose):
    position = np.asarray(pose['position'], float)
    quaternion = np.asarray(pose['orientation'], float)
    require(position.shape == (3,) and quaternion.shape == (4,)
            and np.isfinite(position).all() and np.isfinite(quaternion).all(),
            'Invalid pose coordinates')
    require(abs(float(quaternion@quaternion)-1.) <= 1e-8, 'Invalid pose quaternion')
    return position, Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def _pose(position, rotation):
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    answer = dict(position=np.asarray(position).tolist(), orientation=quaternion.tolist())
    pose_arrays(answer)
    return answer


def relative_pose(anchor, pose):
    at, ar = pose_arrays(anchor)
    pt, pr = pose_arrays(pose)
    return _pose(ar.T@(pt-at), ar.T@pr)


def compose_pose(anchor, pose):
    at, ar = pose_arrays(anchor)
    pt, pr = pose_arrays(pose)
    return _pose(at+ar@pt, ar@pr)


def inverse_pose(pose):
    position, rotation = pose_arrays(pose)
    return _pose(-rotation.T@position, rotation.T)


def pose_errors(actual, expected):
    at, ar = pose_arrays(actual)
    et, er = pose_arrays(expected)
    return dict(position_max_abs=float(np.max(abs(at-et))),
                rotation_matrix_max_abs=float(np.max(abs(ar-er))))


def _maximum(values):
    values = np.asarray(values)
    index = np.unravel_index(np.argmax(values), values.shape)
    return dict(value=float(values[index]), index=[int(i) for i in index])


def factor_comparison(score, mapped, covariances):
    """Factor differences and exact-real covariance differences of stored floats."""
    delta = abs(score-mapped)
    scale = np.maximum(abs(score), abs(mapped))
    relative = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
    covariance_delta = np.zeros_like(delta)
    as_fraction = lambda x: Fraction.from_float(float(x))
    for b in range(len(score)):
        for i in range(6):
            for j in range(i+1):
                difference = sum((as_fraction(score[b, i, k])*as_fraction(score[b, j, k])
                                  - as_fraction(mapped[b, i, k])*as_fraction(mapped[b, j, k])
                                  for k in range(j+1)), Fraction(0))
                covariance_delta[b, i, j] = covariance_delta[b, j, i] = float(abs(difference))
    cs = np.asarray([exported_covariance(lower) for lower in score])
    cm = np.asarray([exported_covariance(lower) for lower in mapped])
    conditions = np.asarray([np.linalg.cond(c) for c in covariances])
    require(np.isfinite(conditions).all(), 'Unrepresentable covariance condition estimate')
    return dict(different_factor_components=int(np.any(score != mapped, axis=(1, 2)).sum()),
                factor_absolute=_maximum(delta), factor_entry_relative=_maximum(relative),
                factor_per_component_maxnorm_relative=float(np.max(
                    delta.max(axis=(1, 2))/scale.max(axis=(1, 2)))),
                covariance_ordered_reconstruction_absolute=_maximum(abs(cs-cm)),
                covariance_exact_real_absolute=_maximum(covariance_delta),
                covariance_exact_per_component_maxnorm_relative=float(np.max(
                    covariance_delta.max(axis=(1, 2))/np.maximum(abs(cs), abs(cm)).max(axis=(1, 2)))),
                condition_estimates=conditions.tolist(),
                condition_max=float(conditions.max()), condition_max_component=int(conditions.argmax()),
                scope='Differences of constructor coefficients, not a global density or closure bound.')


class DimerDestinationDensity:
    """Both scored and map-generated atlas densities plus deterministic trace maps."""
    def __init__(self, model, *, source_bundle=None):
        base, flags = unwrap_proposal_model(model)
        require(base.get('coordinate_convention') == 'anchor-body-relative',
                'Dimer charts must be body-relative')
        require(base.get('dfs') is None or all(v is None for v in base['dfs']),
                'Only Gaussian dimer charts are supported')
        self.source_binding = bind_source_bundle(source_bundle) if source_bundle is not None else None
        self.base_factors = np.asarray([scalar_cholesky(c) for c in base['covariances']])
        self.exported_covariances = np.asarray([exported_covariance(l) for l in self.base_factors])
        self.score_factors = np.asarray([scalar_cholesky(c) for c in self.exported_covariances])
        self.map_factors = np.asarray([map_cholesky(c) for c in self.exported_covariances])
        self.score, self.map_density = Density(model), Density(model)
        total = ordered_sum(base['weights'])
        require(math.isfinite(total) and total > 0 and abs(total-1.) <= 2e-12,
                'Invalid base probabilities')
        for density, factors in [(self.score, self.score_factors), (self.map_density, self.map_factors)]:
            density.weights = np.asarray([(float(base['weights'][k])/total)/(2 if flags[k] else 1)
                                          for k in density.base_indices])
            density.lower = factors[density.base_indices]
            density.logdet = np.asarray([ordered_sum(math.log(float(l[i, i])) for i in range(6))
                                        for l in density.lower])
        self.factor_report = factor_comparison(self.score_factors, self.map_factors,
                                               self.exported_covariances)

    def _branch(self, branch):
        require(type(branch) is int and 0 <= branch < len(self.map_density.weights),
                'Invalid virtual branch')
        density = self.map_density
        return density, density.anchors[branch], density.lower[branch]

    def encode(self, branch, pose):
        density, anchor, lower = self._branch(branch)
        if density.inverted[branch]:
            pose = inverse_pose(pose)
        position, rotation = pose_arrays(pose)
        q = Rotation.from_matrix(rotation@np.asarray(anchor['rotation']).T).as_quat()
        require(q[3] != 0., 'Exact Cayley seam')
        coordinates = np.r_[position-anchor['position'], density.ell*q[:3]/q[3]]
        require(np.isfinite(coordinates).all(), 'Unrepresentable Cayley coordinates')
        latent = solve_triangular(lower, coordinates-density.mean[branch], lower=True)
        require(np.isfinite(latent).all(), 'Unrepresentable latent coordinates')
        return latent

    def coordinates(self, branch, latent):
        density, _, lower = self._branch(branch)
        latent = np.asarray(latent, float)
        require(latent.shape == (6,) and np.isfinite(latent).all(), 'Invalid map latent')
        coordinates = np.asarray([float(density.mean[branch, i])+ordered_sum(
            float(lower[i, j])*float(latent[j]) for j in range(i+1)) for i in range(6)])
        require(np.isfinite(coordinates).all(), 'Unrepresentable decoded coordinates')
        return coordinates

    def decode(self, branch, latent):
        density, anchor, _ = self._branch(branch)
        coordinates = self.coordinates(branch, latent)
        c = coordinates[3:]/density.ell
        require(np.isfinite(c).all(), 'Unrepresentable decoded Cayley coordinates')
        length = math.hypot(1., *c)
        rotation = Rotation.from_quat(np.r_[c/length, 1./length]).as_matrix()
        pose = _pose(coordinates[:3]+anchor['position'], rotation@np.asarray(anchor['rotation']))
        return inverse_pose(pose) if density.inverted[branch] else pose

    def log_volume(self, branch, latent):
        density, _, _ = self._branch(branch)
        c = self.coordinates(branch, latent)[3:]/density.ell
        value = float(density.logdet[branch])-3*math.log(density.ell)-2*math.log(math.pi)
        value -= 4*math.log(math.hypot(1., *c))
        require(math.isfinite(value), 'Unrepresentable chart volume')
        return value

    def edge_density(self, pose, alpha, half_width, *, kind='score'):
        require(math.isfinite(alpha) and 0 <= alpha <= 1 and math.isfinite(half_width)
                and half_width > 0, 'Invalid defensive parameters')
        require(kind in ('score', 'map'), 'Unknown density factor law')
        density = self.score if kind == 'score' else self.map_density
        position, _ = pose_arrays(pose)
        uniform = -3*(math.log(2.)+math.log(half_width)) if np.all(abs(position) <= half_width) else -math.inf
        learned = float(density.evaluate([pose])[0][0])
        a = math.log(alpha)+uniform if alpha > 0 else -math.inf
        b = math.log1p(-alpha)+learned if alpha < 1 else -math.inf
        return dict(log_uniform=uniform, log_learned=learned, log_full=float(np.logaddexp(a, b)))

    def reconstruct_tree_edge(self, old_pose, source, target, noise, rho=.7, density_values=None):
        """Reconstruct supplied labels/noise; never generate a new proposal trace.

        Density diagnostics use the reconstructed relative endpoint. Callers
        should also score the saved step.handle separately to distinguish pose
        reconstruction roundoff from density arithmetic.
        """
        require(math.isfinite(rho) and -1 <= rho <= 1, 'Invalid correlation')
        self._branch(source)
        self._branch(target)
        noise = np.asarray(noise, float)
        require(noise.shape == (6,) and np.isfinite(noise).all(), 'Invalid trace noise')
        z = self.encode(source, old_pose)
        sine = math.sqrt(1.-rho*rho)
        new_z = rho*z+sine*noise
        inverse_noise = sine*z-rho*noise
        endpoint = self.decode(target, new_z)
        auxiliary = .5*(ordered_sum(float(v)*float(v) for v in noise)
                        -ordered_sum(float(v)*float(v) for v in inverse_noise))
        jacobian = self.log_volume(target, new_z)-self.log_volume(source, z)
        if density_values is None:
            full, _, components = self.score.evaluate([old_pose, endpoint])
            values = (full[0], full[1], components[0, source], components[1, target])
        else:
            values = density_values
        require(len(values) == 4 and all(math.isfinite(v) for v in values),
                'Four finite cached density values required')
        old_full, new_full, old_component, new_component = map(float, values)
        labels = (new_component-new_full)-(old_component-old_full)
        labels += math.log(float(self.score.weights[source]))-math.log(float(self.score.weights[target]))
        return dict(pose=endpoint, source_latent=z.tolist(), target_latent=new_z.tolist(),
                    inverse_trace=dict(source=target, target=source, noise=inverse_noise.tolist()),
                    log_extended_jacobian=jacobian, log_auxiliary_ratio=auxiliary,
                    log_correction=jacobian+auxiliary, label_log_reverse_forward=float(labels),
                    expanded_log_reverse_forward=float(jacobian+auxiliary+labels),
                    full_old_member_log_density=old_full, full_new_member_log_density=new_full,
                    log_reverse_forward=old_full-new_full,
                    density_coordinates='reconstructed endpoint' if density_values is None else 'caller supplied endpoints')

    def audit_tree_edge(self, edge, rho=.7, density_values=None):
        """Return observed errors; thresholds belong to caller.

        Optional density_values is (old logG, saved-handle logG,
        old-source weighted log component, saved-handle target weighted log
        component), independently computed by the caller's batched scorer.
        It avoids all mixture evaluations here. The caller binds it to this
        exact old pose, saved handle and branch pair.
        """
        trace = edge['trace']
        require(trace is not None and edge['step'] is not None, 'A null edge has no finite map witness')
        saved = edge['step']
        if density_values is None:
            full, _, components = self.score.evaluate([edge['old_relative_pose'], saved['handle']])
            density_values = (full[0], full[1], components[0, trace['source']], components[1, trace['target']])
        rebuilt = self.reconstruct_tree_edge(edge['old_relative_pose'], trace['source'], trace['target'],
                                             trace['noise'], rho, density_values)
        maps = saved['step']
        errors = pose_errors(saved['handle'], rebuilt['pose'])
        for key in ('source_latent', 'target_latent'):
            errors[key+'_max_abs'] = float(np.max(abs(np.asarray(maps[key])-rebuilt[key])))
        errors['inverse_noise_max_abs'] = float(np.max(abs(
            np.asarray(maps['inverse_trace']['noise'])-rebuilt['inverse_trace']['noise'])))
        require(maps['inverse_trace']['source'] == trace['target']
                and maps['inverse_trace']['target'] == trace['source'], 'Inverse labels differ')
        for key in ('log_extended_jacobian', 'log_auxiliary_ratio', 'log_correction'):
            errors[key+'_abs'] = abs(float(maps[key])-rebuilt[key])
        old_full, new_full, old_component, new_component = map(float, density_values)
        labels = (new_component-new_full)-(old_component-old_full)
        labels += math.log(float(self.score.weights[trace['source']]))-math.log(float(self.score.weights[trace['target']]))
        saved_endpoint = dict(full_old_member_log_density=old_full,
                              full_new_member_log_density=new_full,
                              log_reverse_forward=old_full-new_full,
                              label_log_reverse_forward=float(labels))
        for key, value in saved_endpoint.items():
            errors[key+'_abs'] = abs(float(saved[key])-value)
        errors['expanded_log_reverse_forward_abs'] = abs(
            float(saved['expanded_log_reverse_forward'])-rebuilt['expanded_log_reverse_forward'])
        return dict(reconstruction=rebuilt, saved_endpoint_density=saved_endpoint, errors=errors,
                    scope='Observed reconstruction errors only; caller applies declared tolerances. No closure certificate.')
