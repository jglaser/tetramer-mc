#!/usr/bin/env python3
"""Independent native-class translation-line geometry and normalized density.

No protein data are opened and no geometry/random queries occur on import. Native
geometry is reconstructed from the original Python observer's monomer atoms,
residue IDs, member frames, and complete motif catalogue. The implementation uses
unpruned leaf atom pairs and NumPy long-double quadratic roots; Rust BVH intervals
are never inputs. This is an audit/reference implementation, not a production MC
kernel. Closed native thresholds and strict core/exclusion thresholds are distinct.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import copy
import math
import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.special import logsumexp

import hard_free_line_reference as hard
from native_contact_regions import CRITERIA, NativeContactRegions, angle_degrees, pose_arrays

require, interval, union, contains = hard.require, hard.interval, hard.union, hard.contains
interval_masses = hard.interval_masses
SCHEMA = 'defensive-native-class-line-guide-v1'


def intersection(left, right):
    """Canonical endpoint-aware intersection, retaining closed singletons."""
    output = []
    for a in union(left):
        for b in union(right):
            lo, hi = max(a['lower'], b['lower']), min(a['upper'], b['upper'])
            lc = (lo > a['lower'] or a['lower_closed']) and (lo > b['lower'] or b['lower_closed'])
            uc = (hi < a['upper'] or a['upper_closed']) and (hi < b['upper'] or b['upper_closed'])
            candidate = interval(lo, hi, lc, uc)
            if hard.nonempty(candidate):
                output.append(candidate)
    return union(output)


def difference(left, right):
    """A\\B for finite canonical sets, including isolated tangent witnesses."""
    answer = union(left)
    for b in union(right):
        pieces = []
        for a in answer:
            overlap = intersection([a], [b])
            if not overlap:
                pieces.append(a)
                continue
            c = overlap[0]
            for v in [interval(a['lower'], c['lower'], a['lower_closed'], not c['lower_closed']),
                      interval(c['upper'], a['upper'], not c['upper_closed'], a['upper_closed'])]:
                if hard.nonempty(v):
                    pieces.append(v)
        answer = union(pieces)
    return answer


def distance_intervals(delta, direction, radius, segment, *, inclusive=True):
    """Solve ||delta+s*direction|| <= radius (or < radius), clipped to segment.

    Zero direction, exact tangencies, and zero-radius closed sets are explicit.
    Arithmetic failure is an error and can never become an empty-ray fallback.
    """
    p, d = np.asarray(delta, np.longdouble), np.asarray(direction, np.longdouble)
    require(p.shape == d.shape and p.ndim == 1 and np.isfinite(p).all()
            and np.isfinite(d).all(), 'Invalid distance line')
    require(math.isfinite(radius) and radius >= 0, 'Invalid distance radius')
    require(len(segment) == 2 and np.isfinite(segment).all() and segment[0] <= segment[1], 'Invalid finite line segment')
    whole = [interval(*segment)]
    a = d @ d
    if a == 0:
        value = p @ p
        allowed = value <= np.longdouble(radius)**2 if inclusive else value < np.longdouble(radius)**2
        return whole if allowed else []
    center = -(p @ d) / a
    nearest = p + center*d
    residual = np.longdouble(radius)**2 - nearest @ nearest
    require(np.isfinite(center) and np.isfinite(residual), 'Nonfinite quadratic geometry')
    if residual < 0 or (residual == 0 and not inclusive):
        return []
    half = np.sqrt(residual/a)
    lo, hi = float(center-half), float(center+half)
    require(math.isfinite(lo) and math.isfinite(hi), 'Unrepresentable quadratic endpoints')
    return intersection(whole, [interval(lo, hi, inclusive, inclusive)])


def closed_ball_chord(origin, direction, radius):
    """Finite closed chord, including isolated tangencies; failure stays fatal."""
    p, d = np.asarray(origin, np.longdouble), np.asarray(direction, np.longdouble)
    require(p.shape == d.shape and np.isfinite(p).all() and np.isfinite(d).all()
            and math.isfinite(radius) and radius > 0, 'Invalid closed chord')
    a = d @ d
    require(a > 0 and np.isfinite(a), 'Unrepresentable chord speed')
    center = -(p @ d)/a
    nearest = p+center*d
    residual = np.longdouble(radius)**2-nearest @ nearest
    require(np.isfinite(center) and np.isfinite(residual), 'Unrepresentable closed chord')
    if residual < 0:
        return None
    half = np.sqrt(residual/a)
    result = [float(center-half), float(center+half)]
    require(np.isfinite(result).all() and (result[0] < result[1] if residual > 0 else result[0] <= result[1]),
            'Unrepresentable positive chord width')
    return result


def orthant_intervals(origin, direction, orthant, segment):
    """Six ORIGINAL whitened-coordinate signs; zero belongs to positive bit."""
    u, d = np.asarray(origin, float), np.asarray(direction, float)
    require(u.shape == d.shape == (6,) and np.isfinite(u).all() and np.isfinite(d).all(), 'Invalid whitened line')
    require(type(orthant) is int and 0 <= orthant < 64, 'Invalid orthant')
    current = [interval(*segment)]
    for i, (a, b) in enumerate(zip(u, d)):
        positive = bool(orthant & (1 << i))
        if b == 0:
            if (a >= 0) != positive:
                return []
            continue
        root = -a/b
        if (b > 0) == positive:
            allowed = [interval(max(segment[0], root), segment[1], positive if root >= segment[0] else True, True)]
        else:
            allowed = [interval(segment[0], min(segment[1], root), True, positive if root <= segment[1] else True)]
        current = intersection(current, allowed)
    return current


def leaf_contact_intervals(fixed, moving, direction, fixed_radii, moving_radii,
                           gap, segment, *, inclusive, pairs=None):
    """Unpruned atom-pair union. Ordered (fixed atom, moving atom) pairs."""
    fixed, moving = np.asarray(fixed, float), np.asarray(moving, float)
    fr, mr = np.asarray(fixed_radii, float), np.asarray(moving_radii, float)
    require(fixed.shape == (len(fr), 3) and moving.shape == (len(mr), 3), 'Atom geometry mismatch')
    require(np.isfinite(fixed).all() and np.isfinite(moving).all()
            and np.isfinite(fr).all() and np.isfinite(mr).all()
            and (fr >= 0).all() and (mr >= 0).all() and math.isfinite(gap) and gap >= 0,
            'Invalid contact geometry')
    velocity = np.asarray(direction, np.longdouble)
    require(velocity.shape == (3,) and np.isfinite(velocity).all(), 'Invalid contact velocity')
    speed2 = velocity @ velocity
    fixed_ld, moving_ld = np.asarray(fixed, np.longdouble), np.asarray(moving, np.longdouble)
    fr_ld, mr_ld = np.asarray(fr, np.longdouble), np.asarray(mr, np.longdouble)
    if pairs is not None:
        pairs = np.asarray(list(pairs), int).reshape((-1, 2))
        require(np.all((pairs[:, 0] >= 0) & (pairs[:, 0] < len(fixed)))
                and np.all((pairs[:, 1] >= 0) & (pairs[:, 1] < len(moving))), 'Invalid ordered residue atom pair')
    def chunks():
        if pairs is None:
            # Bounded arrays contain EVERY leaf pair, without a BVH/KD-tree or
            # a geometric candidate pruning predicate.
            for start in range(0, len(moving), 32):
                js = np.arange(start, min(start+32, len(moving)))
                yield np.repeat(np.arange(len(fixed)), len(js)), np.tile(js, len(fixed))
        else:
            for start in range(0, len(pairs), 65536):
                rows = pairs[start:start+65536]
                yield rows[:, 0], rows[:, 1]
    answer, count = [], 0
    for ii, jj in chunks():
        count += len(ii)
        delta = moving_ld[jj]-fixed_ld[ii]
        radius = fr_ld[ii]+mr_ld[jj]+np.longdouble(gap)
        if speed2 == 0:
            squared = np.sum(delta*delta, axis=1)
            allowed = squared <= radius*radius if inclusive else squared < radius*radius
            if allowed.any():
                answer = [interval(*segment)]
            continue
        centers = -(delta @ velocity)/speed2
        nearest = delta+centers[:, None]*velocity
        residual = radius*radius-np.sum(nearest*nearest, axis=1)
        require(np.isfinite(centers).all() and np.isfinite(residual).all(), 'Nonfinite leaf quadratic')
        positive = residual >= 0 if inclusive else residual > 0
        half = np.sqrt(residual[positive]/speed2)
        lower = np.asarray(centers[positive]-half, float)
        upper = np.asarray(centers[positive]+half, float)
        require(np.isfinite(lower).all() and np.isfinite(upper).all(), 'Unrepresentable leaf roots')
        lo, hi = np.maximum(lower, segment[0]), np.minimum(upper, segment[1])
        lc, uc = (lo > lower) | inclusive, (hi < upper) | inclusive
        keep = (lo < hi) | ((lo == hi) & lc & uc)
        answer = union(answer+[interval(l, h, a, b) for l, h, a, b in zip(lo[keep], hi[keep], lc[keep], uc[keep])])
    return dict(intervals=answer, leaf_atom_pairs=count)



class NativeLineReference:
    """Entire union of the original classifier's motifs and ANY supporting bond."""
    def __init__(self, observer):
        require(isinstance(observer, NativeContactRegions), 'Original native observer required')
        self.observer = observer
        require(len(observer.member_positions) == 4, 'Complete four-member body required')
        self.residue_atoms = {int(r): np.flatnonzero(observer.residues == r).tolist()
                              for r in np.unique(observer.residues)}

    def pair(self, anchor, origin, rotation, direction, segment):
        model = self.observer
        ta, ra = pose_arrays(anchor)
        d = ra.T @ (np.asarray(origin)-ta)
        r = ra.T @ np.asarray(rotation)
        velocity = ra.T @ np.asarray(direction)
        motifs, native, cache = [], [], {}
        for motif, target_position, target_rotation in zip(model.motifs, model.motif_positions, model.motif_rotations):
            angle = angle_degrees(r, target_rotation)
            if angle > CRITERIA['body_orientation_entry_deg']:
                motifs.append(dict(motif_id=motif['id'], intervals=[], reason='body_angle'))
                continue
            allowed = [interval(*segment)]
            for member in model.member_positions:
                delta = r @ member + d - target_rotation @ member - target_position
                allowed = intersection(allowed, distance_intervals(delta, velocity,
                    CRITERIA['body_member_position_entry_A'], segment))
            if not allowed:
                motifs.append(dict(motif_id=motif['id'], intervals=[], reason='member_positions'))
                continue
            bonds = []
            for contact in motif['member_contacts']:
                i, j, label = contact['member_i'], contact['member_j'], contact['directed_class']
                key = (i, j, label)
                if key not in cache:
                    reference = model.references[label]
                    mr = model.member_rotations[i].T @ r @ model.member_rotations[j]
                    md = model.member_rotations[i].T @ (r @ model.member_positions[j] + d - model.member_positions[i])
                    vd = model.member_rotations[i].T @ velocity
                    bond = []
                    atom_count = 0
                    if angle_degrees(mr, reference['rotation']) <= CRITERIA['monomer_orientation_entry_deg']:
                        displacement = distance_intervals(md-reference['position'], vd,
                            CRITERIA['monomer_position_entry_A'], segment)
                        if displacement:
                            pairs = []
                            for residue_pair in sorted(reference['native_residue_pairs']):
                                a, b = divmod(int(residue_pair), model.residue_count)
                                pairs.extend((ia, ib) for ia in self.residue_atoms.get(a, [])
                                             for ib in self.residue_atoms.get(b, []))
                            # Every pair on the WHOLE segment, not merely a motif's
                            # first surviving subinterval: cache keys remain exact.
                            contacts = leaf_contact_intervals(model.atoms, model.atoms @ mr.T + md,
                                vd, model.radii, model.radii, CRITERIA['contact_entry_A'],
                                segment, inclusive=True, pairs=pairs)
                            bond = intersection(displacement, contacts['intervals'])
                            atom_count = contacts['leaf_atom_pairs']
                    cache[key] = dict(intervals=bond, leaf_atom_pairs=atom_count)
                bonds.extend(cache[key]['intervals'])
            found = intersection(allowed, union(bonds))
            motifs.append(dict(motif_id=motif['id'], intervals=found))
            native.extend(found)
        return dict(intervals=union(native), motifs=motifs,
                    unique_bond_queries=len(cache), leaf_atom_pairs=sum(v['leaf_atom_pairs'] for v in cache.values()))

    def all_anchors(self, anchors, origin, rotation, direction, segment):
        rows = [self.pair(a, origin, rotation, direction, segment) for a in anchors]
        return dict(intervals=union([v for row in rows for v in row['intervals']]), anchors=rows,
                    leaf_atom_pairs=sum(r['leaf_atom_pairs'] for r in rows))


def chart_fp64_factor(covariance):
    """Reconstruct the *defined computational chart*, without epsilon rounding.

    Chart::new passes through proposal validation: scalar Cholesky, explicit
    L L^T export, then Chart's scalar Cholesky. Orthants refer to that actual
    coordinate map, whose exact zero bits can differ from a BLAS factorization.
    This narrowly mirrors coordinate-convention arithmetic; Schur conditionals,
    Normal masses, all native predicates and atom roots remain independent.
    """
    covariance = np.asarray(covariance, float)
    require(covariance.shape == (6, 6) and np.isfinite(covariance).all(), 'Invalid chart covariance')
    require(np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12), 'Asymmetric chart covariance')
    first = np.zeros((6, 6))
    for i in range(6):
        for j in range(i+1):
            remainder = .5*(float(covariance[i, j])+float(covariance[j, i]))
            for k in range(j):
                remainder -= float(first[i, k])*float(first[j, k])
            require(math.isfinite(remainder) and (i != j or remainder > 0), 'Nonpositive chart pivot')
            first[i, j] = math.sqrt(remainder) if i == j else remainder/first[j, j]
    reconstructed = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            value = 0.
            for k in range(min(i, j)+1):
                value += float(first[i, k])*float(first[j, k])
            reconstructed[i, j] = value
    second = np.zeros((6, 6))
    for i in range(6):
        for j in range(i+1):
            products = 0.
            for k in range(j):
                products += float(second[i, k])*float(second[j, k])
            remainder = .5*(float(reconstructed[i, j])+float(reconstructed[j, i]))-products
            require(math.isfinite(remainder) and (i != j or remainder > 0), 'Nonpositive exported chart pivot')
            second[i, j] = math.sqrt(remainder) if i == j else remainder/second[j, j]
    # Independent high-precision residual, computed from the exact input and
    # output binary64 values; no topology or sign decision is changed by it.
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 80
        matrix = [[Decimal.from_float(float(v)) for v in row] for row in covariance]
        lower = [[Decimal.from_float(float(v)) for v in row] for row in second]
        residual = max(abs(sum(lower[i][k]*lower[j][k] for k in range(6))
                           -(matrix[i][j]+matrix[j][i])/2) for i in range(6) for j in range(6))
        scale = max(abs(v) for row in matrix for v in row)
        relative = float(residual/scale)
    require(relative <= 1e-12, 'Computational chart fails independent covariance residual check')
    return second, dict(convention='Rust Chart::new proposal-validation/export/scalar-FP64 factor order; no coefficient clipping',
                       decimal_precision=80, relative_covariance_residual=relative,
                       scipy_factor_maximum_absolute_difference=float(abs(second-np.linalg.cholesky(covariance)).max()))


def scalar_chart_solve(lower, right):
    """Exact operation order of the frozen raw→whitened coordinate convention."""
    result = np.zeros(6)
    for i in range(6):
        total = 0.
        for j in range(i):
            total += float(lower[i, j])*float(result[j])
        result[i] = (float(right[i])-total)/float(lower[i, i])
    require(np.isfinite(result).all(), 'Unrepresentable inverse chart')
    return result


def conditional_branch(hard_free, selected_class, mean, sigma, floor, query):
    """Normalized class→hard-free→unconditional law; one floor in all branches."""
    require(math.isfinite(mean) and math.isfinite(sigma) and sigma > 0 and 0 < floor < 1, 'Invalid Normal conditioning')
    require(not difference(selected_class, hard_free), 'Class not a subset of hard-free support')
    zh = float(interval_masses(hard_free, mean, sigma))
    zc = float(interval_masses(selected_class, mean, sigma))
    if zc > floor:
        effective, mass, target = selected_class, zc, 'class'
    elif zh > floor:
        effective, mass, target = hard_free, zh, 'hard_free'
    else:
        effective, mass, target = [], 1., 'unconditional'
    allowed = target == 'unconditional' or contains(effective, query)
    return dict(class_mass=zc, hard_free_mass=zh, effective_mass=mass, fallback=target,
                query_coordinate_allowed=allowed, multiplier=1/mass if allowed else 0.)


class Reconstructor(hard.Reconstructor):
    """Full independent component/axis/channel mixture and physical Jacobian."""
    def __init__(self, region, guide, config, shape, observer):
        require(guide['schema'] == SCHEMA, 'Wrong native-class line schema')
        adapted = copy.deepcopy(guide)
        adapted['schema'] = 'defensive-hard-free-line-guide-v1'
        super().__init__(region, adapted, config, shape)
        self.guide = copy.deepcopy(guide)
        self.L0, self.chart_factor_validation = chart_fp64_factor(region['gaussian_chart']['covariances'][0])
        self.rawmeans = self.m0+self.means @ self.L0.T
        covariance = np.asarray([c['covariance'] for c in guide['gaussian_components']]).reshape((-1, 6, 6))
        rawcov = np.einsum('ij,kjl,ml->kim', self.L0, covariance, self.L0)
        for axis in self.axes:
            others = [i for i in range(6) if i != axis]
            coefficients, sigmas = [], []
            for matrix in rawcov:
                coefficient = solve(matrix[np.ix_(others, others)], matrix[others, axis], assume_a='pos')
                variance = matrix[axis, axis]-matrix[axis, others] @ coefficient
                require(math.isfinite(variance) and variance > 0, 'Nonpositive independent Schur variance')
                coefficients.append(coefficient); sigmas.append(math.sqrt(variance))
            self.conditionals[axis] = (others, np.asarray(coefficients), np.asarray(sigmas))
        self.channels = copy.deepcopy(guide['class_channels'])
        require(bool(self.channels), 'Missing class channels')
        for c in self.channels:
            require(c['class'] in ('hard_free', 'native', 'contact_without_native'), 'Unknown line class')
            require(math.isfinite(c['probability']) and c['probability'] > 0, 'Invalid channel probability')
            if c.get('orthant') is not None:
                require(type(c['orthant']) is int and 0 <= c['orthant'] < 64, 'Invalid channel orthant')
        mass = 0.
        for channel in self.channels: mass += channel['probability']
        require(abs(mass-1.) <= 1e-12, 'Channel probabilities must sum to one')
        for channel in self.channels: channel['probability'] /= mass
        self.native = NativeLineReference(observer)

    def raw(self, u):
        answer = np.zeros(6)
        for i in range(6):
            total = 0.
            for j in range(i+1): total += float(self.L0[i, j])*float(u[j])
            answer[i] = float(self.m0[i])+total
        require(np.isfinite(answer).all(), 'Unrepresentable raw chart coordinates')
        return answer

    def reconstruct_axis(self, u, axis, use_tree=False):
        require(axis in self.axes, 'Unknown raw translation axis')
        raw = self.raw(u); raw[axis] = 0.
        u0 = scalar_chart_solve(self.L0, raw-self.m0)
        du = scalar_chart_solve(self.L0, np.eye(6)[axis])
        _, origin, R, _ = self.decode(u0)
        direction = self.Rf[:, axis]
        from native_contact_regions import make_pose
        geometry = dict(axis=axis, origin=make_pose(origin, R), direction=direction.tolist(), intervals=[])
        r4 = closed_ball_chord(u0, du, self.radius)
        if r4 is None:
            geometry['empty_reason'] = 'no_R4_chord'
        else:
            capture = closed_ball_chord(origin-np.asarray(self.config['capture_center']), direction, self.config['capture_radius'])
            if capture is None:
                geometry['empty_reason'] = 'no_capture_chord'
            else:
                segment = [max(r4[0], capture[0]), min(r4[1], capture[1])]
                if segment[0] > segment[1]:
                    geometry['empty_reason'] = 'disjoint_chords'
                else:
                    geometry.update(segment=segment, **hard.hard_free_intervals(self.shape, self.config['fixed_poses'],
                        origin, R, direction, segment, use_tree=use_tree))
        H = geometry['intervals']
        native, contacts = [], []
        details = None
        if 'segment' in geometry:
            position, rotation = pose_arrays(geometry['origin'])
            direction, segment = geometry['direction'], geometry['segment']
            details = self.native.all_anchors(self.config['fixed_poses'], position, rotation, direction, segment)
            native = details['intervals']
            moving = self.atoms @ rotation.T + position
            for fixed in self.fixed_world:
                contacts.extend(leaf_contact_intervals(fixed, moving, direction, self.radii, self.radii,
                    2*self.config['depletant_radius'], segment, inclusive=False)['intervals'])
            contacts = union(contacts)
        classes = dict(hard_free=H, native=intersection(H, native),
                       contact_without_native=difference(intersection(H, contacts), native))
        x = self.raw(u); raw = x.copy(); raw[axis] = 0
        u0 = scalar_chart_solve(self.L0, raw-self.m0)
        du = scalar_chart_solve(self.L0, np.eye(6)[axis])
        rows = []
        for i, c in enumerate(self.channels):
            allowed = classes[c['class']]
            if c.get('orthant') is not None and 'segment' in geometry:
                allowed = intersection(allowed, orthant_intervals(u0, du, c['orthant'], geometry['segment']))
            rows.append(dict(c, channel=i, intervals=allowed))
        return dict(geometry, hard_free_intervals=H, native_intervals=native,
                    exclusion_contact_intervals=contacts, channels=rows, native_geometry=details)

    def density(self, u):
        u = np.asarray(u); x = self.raw(u); logs = self.gaussian_logs(u)
        uniform = math.log(self.alpha)-self.logvolume if np.linalg.norm(u) <= self.radius else -math.inf
        baseline = float(np.logaddexp(uniform, math.log1p(-self.alpha)+logsumexp(logs))) if self.alpha < 1 else uniform
        if self.beta == 0 or self.alpha == 1:
            return dict(log_density=baseline, baseline_log_density=baseline, conditioning_disabled=True,
                        axes=[], fallback_component_branches=0)
        factors = np.zeros(len(logs)); axes = []; fallback_count = 0
        for axis in self.axes:
            geometry = self.reconstruct_axis(u, axis)
            means, sigmas = self.conditional(x, axis)
            components = []
            for k, (mean, sigma) in enumerate(zip(means, sigmas)):
                branches = []
                for c in geometry['channels']:
                    branch = conditional_branch(geometry['intervals'], c['intervals'], mean, sigma, self.floor, x[axis])
                    factors[k] += c['probability']*branch['multiplier']/len(self.axes)
                    fallback_count += int(branch['fallback'] != 'class')
                    branches.append(dict(branch, channel=c['channel']))
                components.append(dict(component=k, conditional_mean=float(mean), conditional_sigma=float(sigma),
                    hard_free_mass=float(interval_masses(geometry['intervals'], mean, sigma)), channels=branches))
            axes.append(dict(geometry, components=components))
        correction = 1-self.beta+self.beta*factors
        positive = correction > 0
        result = float(np.logaddexp(uniform, math.log1p(-self.alpha)+logsumexp(logs[positive]+np.log(correction[positive]))))
        return dict(log_density=result, baseline_log_density=baseline, axes=axes,
                    component_branches=len(logs)*len(self.axes)*len(self.channels),
                    fallback_component_branches=fallback_count, component_multipliers=correction.tolist())


def compare_intervals(actual, expected, label, *, atol=1e-9):
    """Whole-line topology and endpoint checks, not only sampled membership."""
    hard.normal.validate_intervals(actual)
    hard.normal.validate_intervals(expected)
    require(len(actual) == len(expected), label+' topology differs')
    maximum = 0.
    for a, b in zip(actual, expected):
        require(a['lower_closed'] == b['lower_closed'] and a['upper_closed'] == b['upper_closed'], label+' endpoint inclusion differs')
        hard.close([a['lower'], a['upper']], [b['lower'], b['upper']], label+' endpoints differ', atol=atol, rtol=0)
        maximum = max(maximum, abs(a['lower']-b['lower']), abs(a['upper']-b['upper']))
    return maximum


def observer_from_compiled_for_synthetic(data):
    """Toy-only adapter using unchanged Python classifier predicates.

    Protein audits must pass an original frozen NativeContactRegions definition;
    this adapter alone cannot validate the Rust definition's export provenance.
    """
    from scipy.spatial import cKDTree
    require(data['schema'] == 'native-entry-compiled-v1' and data['criteria'] == CRITERIA,
            'Unsupported compiled observer')
    model = NativeContactRegions.__new__(NativeContactRegions)
    model.member_positions = np.asarray([m['position'] for m in data['members']], float)
    model.member_rotations = np.asarray([m['rotation'] for m in data['members']], float)
    model.atoms = np.asarray([a['center'] for a in data['monomer_atoms']], float)
    model.radii = np.asarray([a['radius'] for a in data['monomer_atoms']], float)
    model.residues = np.asarray([a['residue'] for a in data['monomer_atoms']], int)
    model.residue_count = data['residue_count']
    model.tree = cKDTree(model.atoms)
    model.references = {r['label']: dict(r, position=np.asarray(r['position']), rotation=np.asarray(r['rotation']),
                                        native_residue_pairs=set(r['native_residue_pairs'])) for r in data['references']}
    model.motifs = copy.deepcopy(data['motifs'])
    model.motif_positions = np.asarray([m['position'] for m in data['motifs']], float)
    model.motif_rotations = np.asarray([m['rotation'] for m in data['motifs']], float)
    model.fixed_poses = copy.deepcopy(data['fixed_poses'])
    return model


def compare_compiled_definition(data, observer):
    """Exported numerical inputs are checked against original frozen inputs."""
    other = observer_from_compiled_for_synthetic(data)
    require(data['source_definition_sha256'] == observer.definition_sha256, 'Wrong source native definition')
    require(data['source_input_sha256'] == observer.definition['input_sha256'], 'Changed source native input map')
    require(data['fixed_poses'] == observer.fixed_poses, 'Compiled native scaffold changed')
    for key in ('member_positions', 'member_rotations', 'atoms', 'radii', 'residues', 'motif_positions', 'motif_rotations'):
        hard.close(getattr(other, key), getattr(observer, key), 'Compiled native '+key+' changed', atol=1e-12, rtol=0)
    require(other.residue_count == observer.residue_count and list(other.references) == list(observer.references),
            'Compiled native residue/reference indexing changed')
    for key, ref in observer.references.items():
        actual = other.references[key]
        for field in ('label', 'family', 'native_residue_pairs'):
            require(ref[field] == actual[field], 'Compiled native reference '+field+' changed')
        for field in ('position', 'rotation'):
            hard.close(actual[field], ref[field], 'Compiled reference '+field+' changed', atol=1e-12, rtol=0)
    require([m['id'] for m in other.motifs] == [m['id'] for m in observer.motifs], 'Compiled motif order changed')
    require([m['member_contacts'] for m in other.motifs] == [[{key: c[key] for key in ('member_i', 'member_j', 'directed_class')}
            for c in m['member_contacts']] for m in observer.motifs], 'Compiled supporting bonds changed')


def log_close(actual, expected, label):
    if expected == -math.inf:
        require(actual is None, label+' should be a structural zero')
    else:
        hard.close(actual, expected, label, atol=2e-7, rtol=1e-11)


def compare_geometry(actual, expected, recon, u):
    require(actual['axis'] == expected['axis'], 'Line axis changed')
    require(actual.get('empty_reason') == expected.get('empty_reason'), 'Different empty-line reason')
    axis = expected['axis']; raw = recon.raw(u); raw[axis] = 0.
    u0 = scalar_chart_solve(recon.L0, raw-recon.m0)
    du = scalar_chart_solve(recon.L0, np.eye(6)[axis])
    _, position, rotation, _ = recon.decode(u0)
    hard.close(actual['latent_line_origin'], u0, 'Whitened line origin differs')
    hard.close(actual['latent_line_direction'], du, 'Whitened line direction differs')
    hard.close(actual['origin']['position'], position, 'Line world origin differs')
    hard.close(pose_arrays(actual['origin'])[1], rotation, 'Line world orientation differs')
    hard.close(actual['direction'], recon.Rf[:, axis], 'Line world direction differs')
    if 'segment' in expected:
        hard.close(actual['segment'], expected['segment'], 'Line clip segment differs')
    maximum = 0.
    for key in ('hard_free_intervals', 'native_intervals', 'exclusion_contact_intervals'):
        maximum = max(maximum, compare_intervals(actual[key], expected[key], key))
    require(len(actual['channels']) == len(expected['channels']), 'Missing class channel geometry')
    for a, b in zip(actual['channels'], expected['channels']):
        for key in ('channel', 'class', 'probability', 'orthant'):
            require(a.get(key) == b.get(key), 'Changed geometry channel '+key)
        maximum = max(maximum, compare_intervals(a['intervals'], b['intervals'], 'class intervals'))
        if b.get('orthant') is not None and 'segment' in expected:
            support = orthant_intervals(u0, du, b['orthant'], expected['segment'])
            maximum = max(maximum, compare_intervals(a['orthant_intervals'], support, 'orthant intervals'))
        elif a.get('orthant_intervals') is not None:
            raise ValueError('Unexpected orthant restriction')
    return maximum


def compare_density_trace(actual, expected, recon, u):
    if expected.get('conditioning_disabled'):
        require(actual.get('conditioning_disabled') is True, 'Missing disabled-conditioning flag')
        return 0.
    hard.close(actual['raw_coordinates'], recon.raw(u), 'Density raw coordinate changed')
    log_close(actual['baseline_log_density'], expected['baseline_log_density'], 'Detailed baseline density differs')
    require([v['axis'] for v in actual['axes']] == recon.axes, 'Missing/reordered density axes')
    maximum = 0.; factors = np.zeros(len(recon.weights))
    for a, b in zip(actual['axes'], expected['axes']):
        maximum = max(maximum, compare_geometry(a, b, recon, u))
        require(len(a['components']) == len(recon.weights), 'Missing component branches')
        for ca, cb in zip(a['components'], b['components']):
            require(ca['component'] == cb['component'], 'Changed component index')
            for key in ('conditional_mean', 'conditional_sigma'):
                hard.close(ca[key], cb[key], 'Different Schur '+key)
            require(len(ca['channels']) == len(recon.channels), 'Missing conditional class branches')
            mixture = 0.
            for actual_branch, branch, channel in zip(ca['channels'], cb['channels'], recon.channels):
                require(actual_branch['channel'] == branch['channel'], 'Conditional channel index differs')
                for key in ('class_mass', 'hard_free_mass', 'effective_mass'):
                    hard.close(actual_branch[key], branch[key], 'Different '+key, atol=2e-15, rtol=3e-7)
                hard.close(actual_branch['multiplier'], branch['multiplier'], 'Different class multiplier', atol=1e-12, rtol=3e-7)
                require(actual_branch['fallback_target'] == branch['fallback'], 'Wrong fallback target')
                require(actual_branch['query_coordinate_allowed'] == branch['query_coordinate_allowed'], 'Wrong class membership')
                mixture += channel['probability']*branch['multiplier']
            hard.close(ca['channel_mixture_multiplier'], mixture, 'Different channel mixture multiplier', atol=1e-12, rtol=3e-7)
            factors[ca['component']] += mixture/len(recon.axes)
    hard.close(actual['component_mixture_multipliers'], factors, 'Different axis/component mixture', atol=1e-12, rtol=3e-7)
    return maximum


def audit_draw(draw, recon, u, density):
    if not draw['conditional']:
        hard.close(draw['original_latent'], u, 'Unconditioned draw changed')
        return dict(inverse_error=0., endpoint_error=0.)
    axis, component, channel = draw['axis'], draw['component'], draw['channel']
    require(axis in recon.axes and type(component) is int and 0 <= component < len(recon.weights)
            and type(channel) is int and 0 <= channel < len(recon.channels), 'Bad selected labels')
    expected_channel = recon.channels[channel]
    require(all(draw['class'].get(k) == expected_channel.get(k) for k in ('class', 'probability', 'orthant')),
            'Changed selected channel')
    choice = draw['uniform_channel_selection']; lo = sum(c['probability'] for c in recon.channels[:channel])
    require(0 < choice < 1 and lo <= choice <= lo+expected_channel['probability'], 'Wrong channel categorical selection')
    old, x = recon.raw(draw['original_latent']), recon.raw(u)
    other = [i for i in range(6) if i != axis]
    hard.close(old[other], x[other], 'Conditioner changed another raw coordinate')
    geometry = next(g for g in density['axes'] if g['axis'] == axis)
    maximum = compare_geometry(draw['geometry'], geometry, recon, u)
    means, sigmas = recon.conditional(x, axis); mean, sigma = means[component], sigmas[component]
    branch = geometry['components'][component]['channels'][channel]
    for key, value in [('conditional_mean', mean), ('conditional_sigma', sigma),
                       ('class_mass', branch['class_mass']), ('hard_free_mass', branch['hard_free_mass']),
                       ('effective_mass', branch['effective_mass'])]:
        hard.close(draw[key], value, 'Draw '+key+' differs', atol=2e-15 if 'mass' in key else 2e-8, rtol=3e-7)
    require(draw['fallback_target'] == branch['fallback'] and draw['fallback'] == (branch['fallback'] != 'class'),
            'Wrong selected fallback')
    if branch['fallback'] == 'unconditional':
        hard.close(draw['original_latent'], u, 'Unconditional fallback redrew original coordinates')
        require('selected_interval' not in draw, 'Unconditional fallback has a conditional interval')
        return dict(inverse_error=0., endpoint_error=maximum)
    intervals = geometry['channels'][channel]['intervals'] if branch['fallback'] == 'class' else geometry['hard_free_intervals']
    require(contains(intervals, x[axis]), 'Draw outside effective support')
    selected = draw['selected_interval']; found = None
    for i, v in enumerate(intervals):
        if abs(v['lower']-selected['lower']) < 1e-9 and abs(v['upper']-selected['upper']) < 1e-9:
            compare_intervals([selected], [v], 'Selected draw interval'); found = i; break
    require(found is not None, 'Selected interval absent from complete union')
    mass = float(interval_masses([selected], mean, sigma))
    hard.close(draw['selected_interval_mass'], mass, 'Selected interval mass differs', atol=2e-15, rtol=3e-7)
    require(mass > 0, 'Selected zero-mass interval')
    lower_mass = float(interval_masses(intervals[:found], mean, sigma))
    selected_position = draw['uniform_interval_selection']*branch['effective_mass']
    require(0 < draw['uniform_interval_selection'] < 1 and lower_mass-2e-14 <= selected_position <= lower_mass+mass+2e-14,
            'Wrong Normal interval categorical selection')
    probability = float(hard.normal.normal_masses((selected['lower']-mean)/sigma, (x[axis]-mean)/sigma))/mass
    error = abs(probability-draw['uniform_within_interval'])
    require(0 < draw['uniform_within_interval'] < 1 and error < 2e-6, 'Inverse conditional CDF differs')
    hard.close(draw['returned_raw_coordinate'], x[axis], 'Returned raw coordinate differs')
    return dict(inverse_error=error, endpoint_error=maximum)


def audit(directory, *, definition_path=None, synthetic=False, journal=None):
    """Audit every saved attempt/query. No new poses or Poisson clouds."""
    import json
    from collections import Counter
    from pathlib import Path
    import time
    from scipy.spatial import cKDTree
    started = time.process_time(); root = Path(directory).resolve()
    manifest, summary = hard.read(root/'manifest.json'), hard.read(root/'summary.json')
    require(manifest['schema'] == 'native-class-line-guide-audit-v1' and manifest['physical_jobs'] == 0, 'Wrong audit schema/scope')
    require(summary['complete'] is True and summary['manifest'] == manifest and not (root/'failure.json').exists(), 'Incomplete/failed query ledger')
    bindings = {}
    def bind(path, expected=None):
        path = Path(path).resolve(); digest = hard.sha(path)
        if expected is not None:
            require(digest == expected, 'Changed input '+str(path))
        bindings[str(path)] = digest
        return path
    bind(root/'manifest.json'); bind(root/'summary.json')
    for source in [Path(__file__), Path(hard.__file__), Path(hard.normal.__file__)]: bind(source)
    data = {}
    for name, key in [('config', 'config_sha256'), ('region', 'region_sha256'), ('importance-guide', 'guide_sha256'),
                      ('shape', 'shape_sha256'), ('source-bundle', 'source_bundle_sha256'), ('compiled-native', 'compiled_native_sha256')]:
        data[name] = hard.read(bind(root/'provenance'/f'{name}.json', manifest[key]))
    config, region, guide, shape, compiled = [data[k] for k in ('config', 'region', 'importance-guide', 'shape', 'compiled-native')]
    require(guide['region_sha256'] == manifest['region_sha256'] and region['shape_sha256'] == manifest['shape_sha256'], 'Shape/region guide binding differs')
    require(guide['compiled_native']['sha256'] == manifest['compiled_native_sha256']
            and compiled['source_definition_sha256'] == manifest['native_definition_sha256'], 'Native definition binding differs')
    for key, cfg_key in [('fixed_poses', 'fixed_poses'), ('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'), ('depletant_radius', 'depletant_radius')]:
        require(guide[key] == config[cfg_key], 'Guide/physical '+key+' changed')
    require(guide['shape_sha256'] == manifest['shape_sha256'] and compiled['fixed_poses'] == config['fixed_poses'], 'Compiled shape/scaffold changed')
    require(config['fixed_poses'] == region.get('physical_fixed_neighbors', [region['fixed_neighbor']]), 'Changed physical scaffold')
    for rk, ck in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'), ('activity', 'reservoir_density'), ('depletant_radius', 'depletant_radius'), ('physical_metric', 'metadata')]:
        require(region[rk] == config[ck], 'Changed physical '+rk)
    if synthetic:
        require(definition_path is None, 'Synthetic audit must not mix an original definition')
        require(len(shape['atoms']) <= 64 and len(compiled['monomer_atoms']) <= 16, 'Synthetic adapter cannot audit protein data')
        observer = observer_from_compiled_for_synthetic(compiled)
    else:
        require(definition_path is not None, 'Original frozen native definition required for protein audit')
        observer = NativeContactRegions(bind(definition_path, manifest['native_definition_sha256']))
        compare_compiled_definition(compiled, observer)
        for name, digest in observer.definition['input_sha256'].items():
            bind(observer.root/name, digest)
    bind(Path(__import__('native_contact_regions').__file__))
    recon = Reconstructor(region, guide, config, shape, observer)
    hard.close(manifest['log_latent_ball_volume'], recon.logvolume, 'Wrong latent ball volume')
    hard.close(manifest['latent_radius'], recon.radius, 'Wrong latent radius')
    rows = {name: [json.loads(s) for s in bind(root/(name+'.jsonl'), summary[name+'_sha256']).read_text().splitlines()]
            for name in ('samples', 'probes')}
    probes = []
    if manifest['probes_sha256'] is not None:
        probes = [json.loads(s) for s in bind(root/'provenance/probes.jsonl', manifest['probes_sha256']).read_text().splitlines()]
    require(len(rows['samples']) == manifest['samples'] == summary['samples']
            and [r['id'] for r in rows['samples']] == list(range(manifest['samples'])), 'Missing/reordered unconditional attempts')
    require(len(rows['probes']) == len(probes) == summary['probes'], 'Missing saved probes')
    for a, b in zip(rows['probes'], probes):
        require(a['id'] == b['id'] and a['latent'] == b['latent'], 'Changed saved query')
    attempts = [json.loads(s) for s in bind(root/'attempts.jsonl', summary['attempts_sha256']).read_text().splitlines()]
    require(attempts == [dict(ordinal=i, kind=r['kind'], id=r['id'], state='begin')
                        for i, r in enumerate(rows['samples']+rows['probes'])], 'Missing/repeated/reordered attempt journal')
    counts, maxima, audited = Counter(), Counter(), []
    journal_path = None if journal is None else Path(journal).resolve()
    journal_stream = None if journal_path is None else journal_path.open('x')
    def emit_journal(event):
        if journal_stream is not None:
            journal_stream.write(json.dumps(event, allow_nan=False)+'\n')
            journal_stream.flush()
    emit_journal(dict(state='started', schema='native-class-line-independent-audit-journal-v1',
                synthetic=synthetic, input_sha256=bindings, samples=len(rows['samples']), probes=len(rows['probes'])))
    ordinal = 0
    for kind, records in rows.items():
        for row in records:
            row_started = time.process_time()
            emit_journal(dict(state='begin', ordinal=ordinal, kind=row['kind'], id=row['id']))
            try:
                require(row['kind'] == ('fresh' if kind == 'samples' else 'probe'), 'Changed row type')
                u = np.asarray(row['latent']); require(u.shape == (6,) and np.isfinite(u).all(), 'Invalid latent')
                x, position, rotation, jac = recon.decode(u); density = recon.density(u)
                for key, value in [('latent_radius', np.linalg.norm(u)), ('raw_coordinates', x), ('log_physical_jacobian', jac)]:
                    hard.close(row[key], value, 'Changed '+key)
                hard.close(row['pose']['position'], position, 'Pose translation differs')
                hard.close(pose_arrays(row['pose'])[1], rotation, 'Pose orientation differs')
                log_close(row['log_proposal_density'], density['log_density'], 'Full q differs')
                log_close(row['baseline_log_density'], density['baseline_log_density'], 'Baseline q differs')
                maxima['interval_endpoint_error'] = max(maxima['interval_endpoint_error'], compare_density_trace(row['density_details'], density, recon, u))
                maxima['jacobian_error'] = max(maxima['jacobian_error'], abs(row['log_physical_jacobian']-jac))
                if math.isfinite(density['log_density']):
                    maxima['log_density_error'] = max(maxima['log_density_error'], abs(row['log_proposal_density']-density['log_density']))
                hard_valid = recon.hard_valid(position, rotation)
                capture = bool(np.linalg.norm(position-config['capture_center']) <= config['capture_radius'])
                shell = bool(np.linalg.norm(u) <= recon.radius)
                require(row['hard_valid'] == hard_valid and row['capture_valid'] == capture and row['shell_valid'] == shell, 'Physical domain predicate differs')
                valid = hard_valid and capture and shell
                if valid:
                    matches = [observer.classify_pair(a, row['pose']) for a in config['fixed_poses']]
                    decision = row['native_decision']; anchors = [i for i, m in enumerate(matches) if m]
                    require(decision['native_any'] == bool(anchors) and decision['matched_anchor_indices'] == anchors, 'Pointwise complete native classification differs')
                    require(len(decision['per_anchor']) == len(matches), 'Missing native anchor')
                    for i, (record, found) in enumerate(zip(decision['per_anchor'], matches)):
                        require(record['anchor_index'] == i and record['matched_motif_ids'] == [m['motif_id'] for m in found], 'Pointwise motif union differs')
                        require(len(record['matches']) == len(found), 'Missing detailed motif matches')
                        for actual, expected in zip(record['matches'], found):
                            require(actual['motif_id'] == expected['motif_id'], 'Changed motif order')
                            for key in ('maximum_member_position_error_A', 'proper_orientation_error_deg'):
                                hard.close(actual[key], expected[key], 'Changed motif '+key, atol=3e-6)
                            require(len(actual['supporting_member_bonds']) == len(expected['supporting_member_bonds']), 'Changed ANY-bond support')
                            for ab, eb in zip(actual['supporting_member_bonds'], expected['supporting_member_bonds']):
                                for key in ('members', 'class_label', 'class_family', 'shared_reference_residue_pairs_entry'):
                                    require(ab[key] == eb[key], 'Changed supporting bond '+key)
                                for key in ('position_error_A', 'orientation_error_deg', 'minimum_gap_A'):
                                    if eb[key] is None: require(ab[key] is None, 'Missing bond gap')
                                    else: hard.close(ab[key], eb[key], 'Changed supporting bond '+key, atol=3e-6)
                    mobile = recon.atoms @ rotation.T+position
                    contact = []
                    # Independent point predicate (not line interval membership).
                    moving_tree = cKDTree(mobile)
                    reach = 2*recon.radii.max()+2*config['depletant_radius']
                    for fixed, tree in zip(recon.fixed_world, recon.fixed_trees):
                        neighbors = moving_tree.query_ball_tree(tree, reach+1e-9)
                        contact.append(any(np.any(np.linalg.norm(fixed[js]-mobile[i], axis=1)
                            < recon.radii[i]+recon.radii[js]+2*config['depletant_radius']) for i, js in enumerate(neighbors) if js))
                    require(row['exclusion_contact_by_anchor'] == contact, 'Pointwise exclusion contacts differ')
                else:
                    require(row['native_decision'] is None and row['exclusion_contact_by_anchor'] is None, 'Invalid attempt classified')
                require(math.isfinite(row['observer_cpu_seconds']) and row['observer_cpu_seconds'] >= 0, 'Invalid observer CPU')
                if kind == 'samples':
                    require(row['draw'] is not None, 'Missing attempted-draw trace')
                    result = audit_draw(row['draw'], recon, u, density)
                    maxima['inverse_CDF_error'] = max(maxima['inverse_CDF_error'], result['inverse_error'])
                    maxima['draw_interval_endpoint_error'] = max(maxima['draw_interval_endpoint_error'], result['endpoint_error'])
                    counts.update(attempted=1, conditioned_draws=int(row['draw']['conditional']),
                        fallback_draws=int(row['draw'].get('fallback', False)), hard_capture_shell_valid=int(valid))
                else:
                    require(row['draw'] is None, 'Saved probe drew a pose')
                audited.append(dict(kind=row['kind'], id=row['id'], hard_capture_shell_valid=valid,
                                    log_density=None if density['log_density'] == -math.inf else density['log_density']))
            except Exception as exc:
                emit_journal(dict(state='failed', ordinal=ordinal, kind=row.get('kind'), id=row.get('id'),
                            error_type=type(exc).__name__, error=str(exc), cpu_seconds=time.process_time()-row_started))
                if journal_stream is not None: journal_stream.close()
                raise
            emit_journal(dict(state='complete', ordinal=ordinal, row=audited[-1], maxima=dict(maxima),
                        cpu_seconds=time.process_time()-row_started))
            ordinal += 1
    for key in ('conditioned_draws', 'fallback_draws', 'hard_capture_shell_valid'):
        require(counts[key] == summary[key], 'Summary '+key+' differs')
    for key in ('draw_cpu_seconds', 'density_cpu_seconds'):
        hard.close(summary[key], math.fsum(r[key] for r in rows['samples']), 'Summary '+key+' differs', atol=1e-9)
    for path, digest in bindings.items(): require(hard.sha(path) == digest, 'Input changed during audit')
    emit_journal(dict(state='finished', audited_rows=ordinal, counts=dict(counts), maximum_errors=dict(maxima)))
    if journal_stream is not None: journal_stream.close()
    return dict(schema='independent-native-class-line-audit-v1', complete=True, passed=True, synthetic=synthetic,
                root=str(root), rows=audited, counts=dict(counts), maximum_errors=dict(maxima),
                journal=None if journal_path is None else dict(path=str(journal_path), sha256=hard.sha(journal_path)),
                input_sha256=bindings, chart_factor_validation=recon.chart_factor_validation, analysis_cpu_seconds=time.process_time()-started,
                executable_sha256=manifest['executable_sha256'], source_bundle_sha256=manifest['source_bundle_sha256'],
                new_pose_draws=0, new_Poisson_clouds=0,
                scope='Every attempted pose/query; unpruned complete native leaf intervals, full normalized component/axis/channel density, independent J, direct classifier, inverse CDF and failure/zero accounting. No new poses or physical weight estimates.')


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--definition', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--journal', type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Audit output must be new')
    result = audit(args.directory, definition_path=args.definition, synthetic=args.synthetic, journal=args.journal)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True, counts=result['counts'], maximum_errors=result['maximum_errors'],
                         analysis_cpu_seconds=result['analysis_cpu_seconds'])))
