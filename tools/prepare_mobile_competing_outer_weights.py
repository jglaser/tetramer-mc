#!/usr/bin/env python3
"""Freeze two outer finite-region integrations using only the reviewed old engine."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runs/mobile-competing-reference-pilot-20260921'
PREP = ROOT / 'runs/mobile-competing-reference-preparation-20260921'
BINARY_SHA = 'db4dfbbf23e594ccc49e829cd47436e8e39e33dd9d13e8f0eac22e103d2b45df'
BUNDLE_SHA = '163ec16e4b888648ebc1ffbc0bc3a69de312f0e1f8a0f4b7fbcd672179a2e51b'
SHELLS = ((5., 8.), (8., 12.))
SCOPE = ('Only disjoint frozen finite competitor shells 5<rho<=8 and 8<rho<=12, '
         'with original q>1 on the unchanged observed scaffold. These shells do not '
         'bound the mass outside rho<=12, other contact pockets, the rest of the '
         'native q<=1 domain, or the global physical configuration space. '
         'Unconditional invalid draws remain zeros; no proposal mixture enters the integral.')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def require(value, message):
    if not value:
        raise ValueError(message)


def shell_region(old, inner, outer):
    require(math.isfinite(inner) and math.isfinite(outer) and 0 <= inner < outer,
            'Invalid shell boundaries')
    require(old['minimum_original_q'] == 1. and old['minimum_original_q_inclusive'] is False,
            'Expected unchanged open q>1 restriction')
    region = copy.deepcopy(old)
    region.update(minimum_mahalanobis_radius=inner, mahalanobis_radius=outer,
                  definition=f'Frozen finite competitor shell {inner:g}<rho<={outer:g}; '
                  'original q>1; unchanged geometric chart, exact final scaffold, hard shape, '
                  'technical capture170 and physical sphere. No contact mask or basin completion claim.')
    return region


def containment(region, config, shape):
    model = region['gaussian_chart']
    require(len(model['anchors']) == 1 and model['weights'] == [1.], 'Single chart required')
    fixed = region['fixed_neighbor']
    rotation = Rotation.from_quat(np.asarray(fixed['orientation'])[[1, 2, 3, 0]]).as_matrix()
    center = (np.asarray(fixed['position']) + rotation @
              (np.asarray(model['anchors'][0]['position']) + np.asarray(model['means'][0][:3])))
    lower = np.linalg.cholesky(np.asarray(model['covariances'][0]))
    excursion = region['mahalanobis_radius'] * float(np.linalg.norm(lower[:3], ord=2))
    shape_bound = max(float(np.linalg.norm(a['center'])) + a['radius'] for a in shape['atoms'])
    require(config['capture_center'] == [0., 0., 0.], 'Certificate requires the unchanged origin capture')
    capture = config['capture_radius']
    wall = config['metadata']['physical_sphere_radius_A']
    center_upper = float(np.linalg.norm(center)) + excursion
    require(center_upper < capture, 'Finite shell not certified inside capture')
    require(capture + shape_bound < wall, 'Capture not certified inside physical wall')
    return dict(chart_center_norm_A=float(np.linalg.norm(center)),
                maximum_translation_from_chart_center_A=excursion,
                entire_region_center_norm_upper_bound_A=center_upper,
                capture_radius_A=capture, shape_bound_A=shape_bound,
                physical_sphere_radius_A=wall,
                guaranteed_atomic_wall_clearance_A=wall-capture-shape_bound,
                maximum_chart_rotation_degrees=math.degrees(2*math.atan(
                    region['mahalanobis_radius']*float(np.linalg.norm(lower[3:], ord=2))/model['angular_length'])),
                argument='Triangle inequality bounds every translated atom for every orientation. '
                'The technical capture ball is strictly inside the original physical wall; '
                'each shell lies wholly inside capture.')


def validate_partition(regions):
    bounds = [(r['minimum_mahalanobis_radius'], r['mahalanobis_radius']) for r in regions]
    require(bounds == list(SHELLS), 'Unexpected frozen partition')
    stripped = []
    for r in regions:
        stripped.append({k: v for k, v in r.items() if k not in
                         ('minimum_mahalanobis_radius', 'mahalanobis_radius', 'definition')})
    require(stripped[0] == stripped[1], 'Shells changed more than radial bounds')
    probes = [0., 3., 5., np.nextafter(5., math.inf), 8., np.nextafter(8., math.inf),
              12., np.nextafter(12., math.inf), 20.]
    for rho in probes:
        memberships = [inner < rho <= outer for inner, outer in bounds]
        require(sum(memberships) == int(5. < rho <= 12.), 'Shell overlap or gap')
    return dict(disjoint=True, union='5<rho<=12 intersect unchanged q>1/hard/capture predicates',
                completed_inner_partition='[0,3] union (3,5]; new shells extend this to [0,12]',
                boundary_probes=len(probes), outside_mass_bounded=False,
                note='Exact endpoints have zero Lebesgue/Haar measure, and the declared convention assigns each to one stratum.')


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh immutable output directory')
    protocol_old = read(OLD/'protocol.json')
    require(read(OLD/'status.json')['complete'], 'Old campaign not complete')
    require(sha(PREP/'plan.json') == protocol_old['preparation_sha256'], 'Old preparation changed')
    sources = protocol_old['source_sha256']
    require(sources['latent-region-normalizer'] == BINARY_SHA and
            sources['source-bundle.json'] == BUNDLE_SHA, 'Wrong reviewed engine')
    for name, digest in sources.items():
        require(sha(OLD/'provenance'/name) == digest, f'Old archived dependency changed: {name}')
    bundle = read(OLD/'provenance/source-bundle.json')
    require(len(bundle['files']) == 31, 'Expected 31 embedded source files')
    for name, info in bundle['files'].items():
        require(hashlib.sha256(info['text'].encode()).hexdigest() == info['sha256'],
                f'Embedded source digest mismatch: {name}')

    archive = out/'provenance'
    archive.mkdir(parents=True)
    for name in sources:
        shutil.copy2(OLD/'provenance'/name, archive/name)
    shutil.copy2(Path(__file__), archive/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('run_mobile_competing_outer_weights.py'), out/'driver.py')
    for source, name in [(PREP/'config.json', 'original-config.json'),
                         (PREP/'region-competitor-shell-3-5.json', 'original-shell-3-5.json'),
                         (PREP/'preflight.json', 'original-preflight.json')]:
        shutil.copy2(source, archive/name)
    config = read(archive/'original-config.json')
    shutil.copy2(config['shape'], archive/'shape.json')
    require(sha(archive/'shape.json') == read(archive/'preparation-plan.json')['shape_sha256'],
            'Hard shape changed')
    config['shape'] = str(archive/'shape.json')
    write(out/'config.json', config)
    old_region = read(archive/'original-shell-3-5.json')
    require(old_region['physical_fixed_neighbors'] == config['fixed_poses'] and
            old_region['physical_metric'] == config['metadata'], 'Scaffold/metric mismatch')
    regions = [shell_region(old_region, *bounds) for bounds in SHELLS]
    preflight = dict(passed=True, partition=validate_partition(regions), regions={},
                     source_file_count=31, unchanged_engine=True,
                     Poisson_clouds=0, physical_MC_updates=0)
    commands = []
    for i, region in enumerate(regions):
        lo, hi = SHELLS[i]
        name = f'competitor-shell-{lo:g}-{hi:g}'
        path = out/f'region-{name}.json'
        write(path, region)
        preflight['regions'][name] = containment(region, config, read(archive/'shape.json'))
        seed = 118601010+1009*4*i
        argv = [sys.executable, '-B', str(archive/'run_latent_region_campaign.py'),
                '--out', str(out/name), '--config', str(out/'config.json'), '--region', str(path),
                '--binary', str(archive/'latent-region-normalizer'), '--samples', '8192',
                '--replicates', '4', '--workers', '4', '--seed', str(seed),
                '--lambda-ratio', '64', '--cloud-replicates', '2']
        commands.append(dict(region=name, argv=argv, expected_jobs=[dict(
            id=f'r{j:02d}', seed=seed+1009*j, samples=8192) for j in range(4)],
            region_sha256=sha(path)))
    write(out/'preflight.json', preflight)
    protocol = dict(schema='mobile-competing-outer-regions-campaign-v1', production_launched=False,
                    source_campaign=str(OLD), source_protocol_sha256=sha(OLD/'protocol.json'),
                    source_completed_status_sha256=sha(OLD/'status.json'),
                    fixed_preparation=str(PREP), preparation_sha256=sha(PREP/'plan.json'),
                    commands=commands, maximum_physical_workers=8, samples_per_population=8192,
                    populations_per_region=4, total_unconditional_draws=65536,
                    lambda_ratio=64., cloud_replicates=2, scope=SCOPE,
                    source_sha256={p.name: sha(p) for p in sorted(archive.iterdir())},
                    input_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    write(out/'protocol.json', protocol)
    print(json.dumps(dict(output=str(out), protocol_sha256=sha(out/'protocol.json'),
                         run_command=[sys.executable, '-B', str(out/'driver.py')],
                         production_launched=False, preflight=preflight), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    prepare(parser.parse_args().out)
