#!/usr/bin/env python3
"""Exact-real geometric upper bound on the unbound full-vessel integral.

Normalized proper-rotation Haar measure has volume one. The relative depletion
weight is one whenever moving and fixed exclusion unions are disjoint. No poses,
Poisson points or native labels are sampled by this program.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
from certify_vessel_region_containment import number,norm_upper,read_exact,require


def bound(shape,wall_radius,capture_radius):
    wall_radius,capture_radius = number(wall_radius),number(capture_radius)
    require(wall_radius > 0 and capture_radius > 0 and shape['atoms'],'Positive domains and nonempty shape required')
    atoms = shape['atoms']; n = len(atoms)
    require(all(len(a['center']) == 3 and number(a['radius']) > 0 for a in atoms),'Invalid body spheres')
    mean = [sum(number(a['center'][i]) for a in atoms)/n for i in range(3)]
    mean_radius = sum(number(a['radius']) for a in atoms)/n
    # For every rotation R and allowed center t, average the inequalities
    # ||t + R c_a - wall_center|| <= wall_radius-r_a, then use convexity
    # and the triangle inequality. No convexity of the particle is assumed.
    atom_radius = max(F(0),wall_radius-mean_radius+norm_upper(mean))
    radius = min(atom_radius,capture_radius)
    # Archimedes' strict bound pi < 22/7 yields an exact rational upper bound.
    volume = F(4,3)*F(22,7)*radius**3
    coarse = F(4,3)*F(22,7)*capture_radius**3
    return dict(schema='ideal-depletion-unbound-vessel-mass-bound-v1',
        upper_bound_A3=str(volume),display_upper_bound_A3=float(volume),
        display_log_upper_bound=math.log(volume.numerator)-math.log(volume.denominator) if volume else None,
        exact_zero_bound=volume == 0,
        body_centroid=[str(x) for x in mean],mean_atomic_radius=str(mean_radius),
        translation_radius_upper=str(atom_radius),effective_radius_upper=str(radius),
        coarse_capture_upper_A3=str(coarse),display_bound_over_coarse=float(volume/coarse),
        pi_upper_bound='22/7',normalized_SO3_Haar_volume=1,
        physical_weight='exp[z |E(moving) intersect union E(fixed)|] = 1 on unbound poses',
        scope='Exact-real JSON-decimal geometry. Bounds all hard-valid unbound poses and every subset '
              '(including unbound outside measured pockets). Wall and capture centers may differ: '
              'intersection volume is at most the smaller enclosing-ball volume. '
              'No bound on adsorbed competing contacts, no statistical lower bound on native mass, '
              'and no assembly conclusion. Floating-point wall execution remains an obligation.')


def run(preparation,expected_plan_hash,out):
    root,out = Path(preparation).resolve(),Path(out).resolve()
    require(not out.exists(),'Fresh bound artifact required')
    plan_path = root/'plan.json'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    require(digest(plan_path) == expected_plan_hash,'Preparation identity changed')
    plan = json.loads(plan_path.read_text()); physical = plan['physical']
    require(physical['bath_wall_permeable'] is True and physical['measure'] == 'Lebesgue center volume times normalized SO(3) Haar measure',
            'Wall bath or physical measure differs')
    shape_path,config_path = root/'inputs/shape.json',root/'inputs/config.json'
    for path in (shape_path,config_path): require(digest(path) == plan['input_sha256'][path.name],'Physical input changed')
    config = read_exact(config_path)
    require(number(config['capture_radius']) == F(str(physical['capture_radius'])),'Capture definition differs')
    result = bound(read_exact(shape_path),F(str(physical['wall_radius'])),number(config['capture_radius']))
    result.update(preparation=str(root),preparation_sha256=expected_plan_hash,physical_target=physical,
        input_sha256={str(p):digest(p) for p in (plan_path,shape_path,config_path,Path(__file__).resolve(),
            Path(__file__).resolve().parent/'certify_vessel_region_containment.py')},
        new_pose_draws=0,new_clouds=0,new_geometry_queries=0,new_native_classifier_calls=0)
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,required=True); parser.add_argument('--preparation-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True); args = parser.parse_args()
    value = run(args.preparation,args.preparation_sha256,args.out)
    print({k:value[k] for k in ('display_upper_bound_A3','display_log_upper_bound','display_bound_over_coarse')})
