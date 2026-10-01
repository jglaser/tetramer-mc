#!/usr/bin/env python3
"""Freeze component-mean contact labels and a zero-Poisson three-arm pilot.

The only geometric inputs used to construct labels are the frozen old92
Gaussian means, original chart, hard shape and fixed scaffold. Archived probe
poses are copied only after all labels and guides have been written.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

GUIDE_SHA = 'a00a4470d898e06f66b78dcb7f1c1b42d7c4dc76f9a111dff9ce6268e09a96c0'
REGION_SHA = '924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02'
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
WIDTHS = [.02, .1, .5]
ARMS = [('baseline92', 0., 0.), ('uniform_phi92', .5, 0.), ('localized_phi92', .5, .9)]
SEEDS = [610061001 + 1009 * i for i in range(12)]
COVERAGE_SEEDS = [610078001 + 1009 * i for i in range(3)]
TIE_TOLERANCE = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def declared_seeds(value):
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'seed' and type(item) is int:
                result.add(item)
            elif key.endswith('seeds') and isinstance(item, list):
                result.update(s for s in item if type(s) is int)
            result.update(declared_seeds(item))
    elif isinstance(value, list):
        for item in value:
            result.update(declared_seeds(item))
    return result


class NearestSurface:
    """Exact nearest surface gap using one KD tree per fixed atomic radius.

    Within a group, minimizing center distance minimizes surface gap. Querying
    every moving atom against every group therefore finds the global minimum.
    A second radius query includes all near ties, which are recomputed directly.
    The selected pair is strict floating-point argmin (gap, moving, fixed), not
    an approximate nearest pair chosen within the tolerance.
    """
    def __init__(self, fixed, fixed_radii):
        self.fixed = np.asarray(fixed)
        self.radii = np.asarray(fixed_radii)
        self.groups = []
        for radius in sorted(set(self.radii.tolist())):
            indices = np.flatnonzero(self.radii == radius)
            self.groups.append((radius, indices, cKDTree(self.fixed[indices])))

    def query(self, moving, moving_radii):
        moving = np.asarray(moving)
        moving_radii = np.asarray(moving_radii)
        cached = []
        best = math.inf
        for radius, indices, tree in self.groups:
            distance, _ = tree.query(moving, k=1, eps=0, workers=1)
            gaps = distance - moving_radii - radius
            best = min(best, float(gaps.min()))
            cached.append((radius, indices, tree, gaps))
        candidates = {}
        for radius, indices, tree, gaps in cached:
            for i in np.flatnonzero(gaps <= best + TIE_TOLERANCE):
                reach = max(0., best + TIE_TOLERANCE + moving_radii[i] + radius)
                for local_j in tree.query_ball_point(moving[i], np.nextafter(reach, math.inf), eps=0):
                    j = int(indices[local_j])
                    distance = float(np.linalg.norm(moving[i] - self.fixed[j]))
                    gap = distance - float(moving_radii[i]) - float(radius)
                    candidates[(int(i), j)] = (gap, distance)
        require(candidates, 'Nearest-gap candidate set unexpectedly empty')
        ranked = sorted((gap, i, j, distance) for (i, j), (gap, distance) in candidates.items())
        gap, i, j, distance = ranked[0]
        ties = [dict(moving_atom=a, fixed_atom=b, gap_A=g, gap_above_selected_A=g-gap)
                for g, a, b, _ in ranked if g <= gap + TIE_TOLERANCE]
        return dict(moving_atom=i, fixed_atom=j, minimum_surface_gap_A=gap,
                    center_distance_A=distance, core_radius_sum_A=float(moving_radii[i]+self.radii[j]),
                    exact_float_minimum_ties=sum(g == gap for g, _, _, _ in ranked),
                    near_tie_tolerance_A=TIE_TOLERANCE, near_ties=ties,
                    nearest_center_query_roundoff_A=abs(gap-best))


def polygon_area(core, distance, width):
    """Diagnostic convex clipping only; does not sample or define production q."""
    a, b = core
    polygon = [np.array(p) for p in [(a,b), (a+width,b), (a+width,b+width), (a,b+width)]]
    for normal, offset in [(np.array([-1.,-1.]),-distance),
                           (np.array([1.,-1.]),distance), (np.array([-1.,1.]),distance)]:
        old, polygon = polygon, []
        if not old:
            break
        for start, end in zip(old, old[1:]+old[:1]):
            f0, f1 = float(normal@start-offset), float(normal@end-offset)
            inside0, inside1 = f0 <= 0, f1 <= 0
            if inside0:
                polygon.append(start)
            if inside0 != inside1:
                polygon.append(start + (end-start)*(f0/(f0-f1)))
    if len(polygon) < 3:
        return 0.
    p = np.array(polygon) - polygon[0]
    return abs(float(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1))))/2


def prepare(out, repository):
    started = time.process_time()
    require(not out.exists(), 'Fresh preparation directory required')
    old = repository/'runs/contact-line-passive-preparation-v3-20261001'
    frozen = read(old/'freeze.json')
    for name, digest in frozen['files'].items():
        require(sha(old/name) == digest, 'Original immutable preparation changed: '+name)
    sources = {}
    def bind(path, expected=None):
        digest = sha(path)
        require(expected is None or digest == expected, 'Source hash mismatch: '+str(path))
        sources[str(path.resolve())] = digest
        return path
    bind(old/'freeze.json')
    guide_path = bind(old/'common/original92.json', GUIDE_SHA)
    region_path = bind(old/'common/region.json', REGION_SHA)
    shape_path = bind(old/'common/shape.json', SHAPE_SHA)
    config_path = bind(old/'common/config.json')
    guide, region, shape, config = map(read, (guide_path, region_path, shape_path, config_path))
    require(len(guide['gaussian_components']) == 92 and guide['defensive_uniform_shell_probability'] == .5,
            'Original guide structure changed')
    require(region['physical_fixed_neighbors'] == config['fixed_poses'], 'Scaffold mismatch')
    require(region['activity'] == config['reservoir_density'] == .035 and
            region['depletant_radius'] == config['depletant_radius'] == 1.5, 'Physical conditions changed')
    out.mkdir(parents=True)
    (out/'common').mkdir()
    (out/'guides').mkdir()
    for name, source in [('original92.json',guide_path),('region.json',region_path),('shape.json',shape_path)]:
        shutil.copy2(source,out/'common'/name)
    config['shape'] = str(out/'common/shape.json')
    write_new(out/'common/config.json',config)

    chart = region['gaussian_chart']
    L = np.linalg.cholesky(chart['covariances'][0])
    base_mean = np.asarray(chart['means'][0])
    anchor = chart['anchors'][0]
    rotation = lambda pose: Rotation.from_quat(np.roll(pose['orientation'],-1)).as_matrix()
    body = rotation(region['fixed_neighbor'])
    body_position = np.asarray(region['fixed_neighbor']['position'])
    atom_centers = np.array([a['center'] for a in shape['atoms']])
    atom_radii = np.array([a['radius'] for a in shape['atoms']])
    fixed_world = [atom_centers@rotation(p).T+np.asarray(p['position']) for p in config['fixed_poses']]
    trees = [NearestSurface(p,atom_radii) for p in fixed_world]
    component_pairs, diagnostics = [], []
    for k, component in enumerate(guide['gaussian_components']):
        u = np.asarray(component['mean'])
        x = base_mean+L@u
        c = x[3:]/chart['angular_length']
        quaternion = np.r_[c,1.]
        quaternion /= np.linalg.norm(quaternion)
        R = body@Rotation.from_quat(quaternion).as_matrix()@np.asarray(anchor['rotation'])
        position = body_position+body@(np.asarray(anchor['position'])+x[:3])
        rotated_atoms = atom_centers@R.T
        mobile = rotated_atoms+position
        nearest = [tree.query(mobile,atom_radii) for tree in trees]
        pairs = [dict(neighbor_index=j,moving_atom=nearest[j]['moving_atom'],fixed_atom=nearest[j]['fixed_atom'])
                 for j in (0,1)]
        component_pairs.append(pairs)
        effective = [fixed_world[p['neighbor_index']][p['fixed_atom']]-rotated_atoms[p['moving_atom']] for p in pairs]
        D = float(np.linalg.norm(effective[1]-effective[0]))
        raw_covariance = L@np.asarray(component['covariance'])@L.T
        conditional = raw_covariance[:3,:3]-raw_covariance[:3,3:]@np.linalg.solve(raw_covariance[3:,3:],raw_covariance[3:,:3])
        areas = [polygon_area([p['core_radius_sum_A'] for p in nearest[:2]],D,w) for w in WIDTHS]
        diagnostics.append(dict(component=k,weight=component['weight'],latent_mean=u.tolist(),raw_mean=x.tolist(),
            world_pose=dict(position=position.tolist(),orientation=np.roll(Rotation.from_matrix(R).as_quat(),1).tolist()),
            contact_pairs=pairs,nearest_by_scaffold=nearest,mean_hard_valid=all(p['minimum_surface_gap_A']>=0 for p in nearest),
            mean_R4_valid=bool(region['minimum_mahalanobis_radius']<=np.linalg.norm(u)<=region['mahalanobis_radius']),
            mean_capture_valid=bool(np.linalg.norm(position-np.asarray(region['capture_center']))<=region['capture_radius']),
            simultaneous_contact_by_width=[all(p['minimum_surface_gap_A']<=w for p in nearest[:2]) for w in WIDTHS],
            effective_center_distance_A=D,polygon_area_by_width_A2=areas,
            conditional_translation_principal_SD_A=np.sqrt(np.linalg.eigvalsh(conditional)).tolist()))

    unique = Counter(tuple((p['neighbor_index'],p['moving_atom'],p['fixed_atom']) for p in pair) for pair in component_pairs)
    summary = dict(components_retained=len(diagnostics),unique_pair_pair_labels=len(unique),
        hard_valid_component_means=sum(r['mean_hard_valid'] for r in diagnostics),
        hard_and_R4_and_capture_valid_means=sum(r['mean_hard_valid'] and r['mean_R4_valid'] and r['mean_capture_valid'] for r in diagnostics),
        both_contact_by_width_means=[sum(r['simultaneous_contact_by_width'][j] for r in diagnostics) for j in range(3)],
        hard_valid_both_contact_by_width_means=[sum(r['mean_hard_valid'] and r['simultaneous_contact_by_width'][j] for r in diagnostics) for j in range(3)],
        positive_polygon_by_width_means=[sum(r['polygon_area_by_width_A2'][j]>1e-16 and r['effective_center_distance_A']>1e-8 for r in diagnostics) for j in range(3)],
        near_tied_scaffold_queries=sum(len(p['near_ties'])>1 for r in diagnostics for p in r['nearest_by_scaffold']),
        exact_tied_scaffold_queries=sum(p['exact_float_minimum_ties']>1 for r in diagnostics for p in r['nearest_by_scaffold']),
        minimum_effective_center_distance_A=min(r['effective_center_distance_A'] for r in diagnostics),
        maximum_KD_direct_gap_difference_A=max(p['nearest_center_query_roundoff_A'] for r in diagnostics for p in r['nearest_by_scaffold']))
    write_new(out/'component-labels.json',dict(schema='frozen-component-mean-two-contact-labels-v1',
        source_sha256=dict(sources),summary=summary,components=diagnostics,
        rule='For every unchanged old92 latent Gaussian mean, decode raw pose and minimize distance-ri-rj over all atom pairs separately for each scaffold. KD trees grouped by fixed radius; strict minimum then lexicographic tie break; no native/contact/hard-valid filtering.',
        probe_data_used=False,fit_data_used=False,labels_selected_at_runtime=False,
        scope='Deterministic geometry of the previously frozen integration model; not physical weights or native-blind assembly discovery'))
    for name,beta,b in ARMS:
        candidate = copy.deepcopy(guide)
        candidate.update(schema='defensive-contact-distance-guide-v1',component_contact_pairs=component_pairs,
            contact_neighbor_indices=[0,1],contact_widths_A=WIDTHS,conditional_probability=beta,
            minimum_center_distance=1e-8,minimum_polygon_area=1e-16,
            azimuth=dict(localized_probability=b,radius_floor=1e-8,projection_floor=1e-10,gamma_min=.01,gamma_max=math.pi))
        write_new(out/'guides'/f'{name}.json',candidate)
    # Probe contents cannot influence label construction; copy only after guides exist.
    for name in ('probes.jsonl','coverage-probes.jsonl','coverage-selection.json','axis-diagnostics.json'):
        shutil.copy2(bind(old/name),out/name)
    require(len((out/'probes.jsonl').read_text().splitlines())==78 and
            len((out/'coverage-probes.jsonl').read_text().splitlines())==128, 'Frozen probe allocation changed')

    previous, inventory = set(), {}
    declaration_paths = set()
    for pattern in ('protocol.json','plan.json','prospective-plan.json','prospective-pilot-plan.json'):
        declaration_paths.update((repository/'runs').rglob(pattern))
    for path in sorted(declaration_paths):
        previous.update(declared_seeds(read(path)))
        inventory[str(path.resolve())] = sha(path)
    require(len(set(SEEDS+COVERAGE_SEEDS))==15 and not previous.intersection(SEEDS+COVERAGE_SEEDS),
            'Passive stream seed collision')
    executable = repository/'target-validation-distance-guide/release/contact-distance-guide-audit'
    def job(arm, identity, seed, samples, probe_file):
        cmd=[str(executable),'--config',str(out/'common/config.json'),'--region',str(out/'common/region.json'),
            '--importance-guide',str(out/'guides'/f'{arm}.json'),'--out',str(out/'audit'/arm/identity),
            '--samples',str(samples),'--seed',str(seed)]
        if probe_file:
            cmd += ['--probes',str(out/probe_file)]
        return dict(arm=arm,id=identity,seed=seed,fresh_proposal_draws=samples,
            archived_probe_queries=78 if probe_file=='probes.jsonl' else 128 if probe_file else 0,
            probe_file=probe_file,command=cmd)
    jobs=[job(arm,f'r{i:02}',SEEDS[4*j+i],128,'probes.jsonl' if i==0 else None)
          for j,(arm,_,_) in enumerate(ARMS) for i in range(4)]
    coverage=[job(arm,'coverage',COVERAGE_SEEDS[j],0,'coverage-probes.jsonl') for j,(arm,_,_) in enumerate(ARMS)]
    write_new(out/'plan.json',dict(schema='contact-distance-proposal-only-preparation-v1',source_sha256=sources,
        jobs=jobs,coverage_jobs=coverage,fresh_draws=1536,archived_probe_queries=234,additional_coverage_queries=384,
        total_archived_queries=618,total_output_pose_rows=2154,maximum_CPU_workers=1,
        physical_jobs_launched=0,proposal_jobs_launched=0,new_Poisson_clouds=0,
        arms=[dict(name=name,conditional_probability=beta,localized_probability=b) for name,beta,b in ARMS],
        widths_A=WIDTHS,contact_neighbor_indices=[0,1],minimum_center_distance=1e-8,minimum_polygon_area=1e-16,
        azimuth_floors=dict(radius_floor=1e-8,projection_floor=1e-10,gamma_min=.01,gamma_max=math.pi),
        label_source='Only frozen old92 component means, original chart, shape and scaffold; all92 labels retained without filtering; copied probes never used to choose labels',
        support='50% uniform R4 retained, original untruncated Gaussian half mixed with normalized full-translation conditioning; complete q >=0.5 old92q',
        fallback='Small effective-center distance or polygon area: original conditional 3D translation Normal at the same angular coordinates. Azimuth projection degeneracy: uniform azimuth. No retry or angular redraw.',
        density='Full component/width/fallback/azimuth mixture, angular Gaussian marginal and |det L0| raw-to-latent density factor; unchanged physical J and unconditional hard/domain zeros',
        seed_inventory=inventory,keep_all_draws=True,optional_stopping=False,retries=False,larger_autoextension=False,
        physical_campaign_authorized=False,physical_gates_unchanged=True,
        production_validation=['reference geometry, density, normalization, inverse-CDF and fallback tests',
            'bind reviewed executable/source bundle and independent Python auditor before any execution',
            'freeze draining single-worker dispatcher preserving every attempted draw'],
        diagnostics=['hard-valid/R4/capture and whole-union joint-contact retention per total CPU, independent populations',
            'selected-label contacts separately from whole-union contacts, angular/radial geometry failures and fallback',
            'full-mixture density/Jacobian reconstruction, >=0.5oldq support, all labels and width branches',
            'critical archived moments separately by source and retrospective breadth coverage; no physical masses'],
        scope='Bounded passive proposal diagnostic only, no depletant clouds, normalizers, equilibrium conclusion or assembly launch',
        label_summary=summary,preparation_CPU_seconds=time.process_time()-started))
    shutil.copy2(__file__,out/'source.py')
    write_new(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(out=str(out),prepared=True,summary=summary,fresh_draws=1536,archived_queries=618,
                         jobs_launched=0,plan_sha256=sha(out/'plan.json')),indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    prepare(args.out.resolve(),args.repository.resolve())
