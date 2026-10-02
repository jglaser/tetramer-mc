#!/usr/bin/env python3
"""Sufficient exact-rational bound for a complete chart ball inside a vessel.

JSON decimals denote exact real inputs here. This proves a geometric domain
inclusion for that model, not IEEE floating-point execution or MC convergence.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path


def require(ok, message):
    if not ok: raise ValueError(message)


def number(value):
    require(type(value) in (int, F), 'Use exact integer or JSON-decimal rational inputs')
    return F(value)


def sqrt_upper(value, digits=40):
    value = number(value); require(value >= 0 and type(digits) is int and digits >= 0, 'Invalid square root bound')
    scale = 10**digits
    numerator = value.numerator*scale*scale
    ceiling = (numerator+value.denominator-1)//value.denominator
    root = math.isqrt(ceiling)
    if root*root < ceiling: root += 1
    answer = F(root, scale)
    require(answer*answer >= value, 'Square-root enclosure failed')
    return answer


def norm_upper(vector):
    return sqrt_upper(sum(number(v)**2 for v in vector))


def rotation(quaternion):
    require(len(quaternion) == 4, 'Quaternion requires four coordinates')
    w, x, y, z = map(number, quaternion); n = w*w+x*x+y*y+z*z
    require(n > 0, 'Zero quaternion')
    return [[v/n for v in row] for row in [
        [w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z]]]


def positive_covariance(covariance):
    require(len(covariance) == 6 and all(len(r) == 6 for r in covariance), 'Expected 6D covariance')
    a = [list(map(number, row)) for row in covariance]
    require(all(a[i][j] == a[j][i] for i in range(6) for j in range(6)), 'Exact symmetric covariance required')
    # Exact rational LDL elimination: positive pivots certify positive definiteness.
    for k in range(6):
        require(a[k][k] > 0, 'Covariance is not positive definite')
        for i in range(k+1, 6):
            for j in range(k+1, 6): a[i][j] -= a[i][k]*a[k][j]/a[k][k]


def certify(region, shape, wall_center, wall_radius, capture_center, capture_radius):
    chart = region['gaussian_chart']; fixed = region['fixed_neighbor']
    require(len(chart['means']) == len(chart['covariances']) == len(chart['anchors']) == 1,
            'A single frozen chart defines the region')
    covariance = chart['covariances'][0]; positive_covariance(covariance)
    mean = list(map(number, chart['means'][0])); anchor = list(map(number, chart['anchors'][0]['position']))
    require(len(mean) == 6 and len(anchor) == len(fixed['position']) == 3, 'Coordinate dimensions differ')
    rotate = rotation(fixed['orientation']); raw_mean = [a+b for a, b in zip(anchor, mean[:3])]
    center = [number(fixed['position'][i])+sum(rotate[i][j]*raw_mean[j] for j in range(3)) for i in range(3)]
    radius = number(region['mahalanobis_radius']); require(radius > 0, 'Positive chart radius required')
    # ||L_translation||_op <= ||L_translation||_F = sqrt(trace(Cov_tt)).
    extent = radius*sqrt_upper(sum(number(covariance[i][i]) for i in range(3)))
    require(shape['atoms'], 'Nonempty sphere union required')
    body = F(0)
    for atom in shape['atoms']:
        atom_radius = number(atom['radius']); require(atom_radius > 0 and len(atom['center']) == 3, 'Invalid atom sphere')
        body = max(body, norm_upper(atom['center'])+atom_radius)
    require(len(wall_center) == len(capture_center) == 3, 'Three-dimensional vessel centers required')
    wall_bound = norm_upper([x-number(y) for x, y in zip(center, wall_center)])+extent+body
    capture_bound = norm_upper([x-number(y) for x, y in zip(center, capture_center)])+extent
    wall_margin = number(wall_radius)-wall_bound; capture_margin = number(capture_radius)-capture_bound
    return dict(schema='chart-vessel-containment-certificate-v1',
        wall_containment_proven=wall_margin >= 0, capture_containment_proven=capture_margin >= 0,
        bounds={name: dict(exact_upper=str(value), display=float(value)) for name, value in
                [('translation_extent', extent), ('body_radius', body), ('outer_atomic_radius', wall_bound),
                 ('outer_center_capture_radius', capture_bound)]},
        clearance_lower_bounds={name: dict(exact_lower=str(value), display=float(value)) for name, value in
                [('atomic_wall', wall_margin), ('center_capture', capture_margin)]},
        method='Triangle inequality; exact quaternion rotation; Frobenius bound from translational covariance trace; '
               'rational outward square-root enclosures; exact-rational positive-definiteness check.',
        scope='Entire chart ball, irrespective of hard overlap, native labels or historical q/capture cuts. '
              'A failed sufficient bound is unresolved containment, not evidence of clipping. '
              'Exact-real JSON-decimal geometry only; floating-point implementation remains an obligation.')


def read_exact(path):
    return json.loads(Path(path).read_text(), parse_float=F,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON number')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('region', 'shape', 'out'): parser.add_argument('--'+name, type=Path, required=True)
    for name in ('wall-center', 'capture-center'): parser.add_argument('--'+name, type=F, nargs=3, required=True)
    for name in ('wall-radius', 'capture-radius'): parser.add_argument('--'+name, type=F, required=True)
    args = parser.parse_args(); require(not args.out.exists(), 'Fresh certificate destination required')
    region = read_exact(args.region); shape_hash = hashlib.sha256(args.shape.read_bytes()).hexdigest()
    require(region['shape_sha256'] == region['gaussian_chart']['shape_sha256'] == shape_hash,
            'Certificate shape differs from the frozen chart')
    result = certify(region, read_exact(args.shape), args.wall_center, args.wall_radius,
                     args.capture_center, args.capture_radius)
    result['input_sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (args.region, args.shape, Path(__file__))}
    result['vessel'] = dict(wall_center=list(map(str, args.wall_center)), wall_radius=str(args.wall_radius),
                           capture_center=list(map(str, args.capture_center)), capture_radius=str(args.capture_radius))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('wall_containment_proven', 'capture_containment_proven', 'clearance_lower_bounds')}))
