#!/usr/bin/env python3
"""Freeze geometry-scaled local references at a recorded AB far-contact peak."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import numpy as np

from analyze_native_region_reference import native_q
from prepare_cayley_rms_cover import read, write, sha, require, uniform_draws, check_coordinates
from prepare_peak_neighborhood import geometry_model, verify_member_geometry
from prepare_intermediate_local_region import BINARY, BINARY_SHA
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_smc_normalizer_atlas import Density, relative_poses
from run_shoulder_mis_campaign import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'runs/ab-competing-frozen-atlas-20260920'
CANDIDATES = ROOT/'runs/ab-far-contact-candidates-20260920/analysis.json'
RADII = (.5, 1., 2.)
SEEDS = (106001010, 106101010, 106201010)
PROBE_SEED = 106301010
POPULATIONS, SAMPLES, PROBES = 4, 16384, 256
PHYSICAL = ('metadata', 'fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density')


def original_peak(candidate):
    """Use the exact archived point and weight; no selected-pose correction."""
    sample = Path(candidate['source_samples_path'])
    manifest = Path(candidate['source_manifest_path'])
    require(sha(sample) == candidate['source_samples_sha256'], 'Historical sample changed')
    require(sha(manifest) == candidate['source_manifest_sha256'], 'Historical manifest changed')
    source = read(manifest)
    require(source['seed'] == candidate['seed'], 'Selected stream changed')
    selected = None
    with sample.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['draw'] == candidate['draw']:
                selected = row; break
    require(selected is not None, 'Selected historical draw missing')
    require(selected['pose'] == candidate['pose'] and selected['q'] == candidate['q'], 'Selected pose changed')
    require(selected['log_importance_weight'] == candidate['original_log_importance_weight'], 'Selected weight changed')
    require(selected['hard_valid'] and selected['capture_valid'] and selected['q'] >= 5., 'Wrong physical selection')
    return selected, source


def prepare(out):
    require(not out.exists(), 'Use a fresh preparation directory')
    candidates = read(CANDIDATES); require(candidates['complete'], 'Candidate audit incomplete')
    require(sha(CANDIDATES.parent/'protocol.json') == candidates['protocol_sha256'], 'Candidate protocol changed')
    for name, digest in candidates['archived_sha256'].items():
        require(sha(CANDIDATES.parent/'provenance'/name) == digest, 'Candidate archive changed')
    index = candidates['selected_peak']; require(type(index) is int, 'Explicit frozen selection index required')
    chosen = candidates['candidates'][index]
    original, manifest = original_peak(chosen)
    require(chosen['width'] == 4., 'Keep the declared broad-arm maximum')
    cfg = read(SOURCE/'config.json'); old = read(SOURCE/'model-competitor.json')
    source_cfg_path = Path(chosen['source_config_path'])
    require(sha(source_cfg_path) == chosen['source_config_sha256'], 'Selected physical config changed')
    require(all(read(source_cfg_path)[k] == cfg[k] for k in PHYSICAL), 'Selected point has different physical neighbors or metric')
    require(original == chosen['original_row'], 'Selected original-row snapshot changed')
    seal = read(SOURCE/'report.json')
    for name in ('config.json', 'model-competitor.json', 'region-competitor-r3.json'):
        require(sha(SOURCE/name) == seal['outputs'][name], 'Old frozen reference changed')
    shape = Path(cfg['shape']); shape_sha = sha(shape)
    require(shape_sha == old['shape_sha256'] == manifest['shape_sha256'], 'Physical shape changed')
    previous = read(ROOT/'runs/ab-intermediate-expanded-atlas-repeat-preparation-20260920/config.json')
    require(all(cfg[k] == previous[k] for k in PHYSICAL), 'Different AB target from intermediate reference')
    require(sha(BINARY) == BINARY_SHA, 'Reviewed latent executable changed')
    require(cfg['capture_radius'] == 18. and cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035,
            'Physical bath/capture changed')
    atom = AtomUnionAudit(read(shape), cfg['fixed_poses'])
    peak = chosen['pose']; gaps = atom.gaps(peak)
    require(min(gaps) >= 0 and 5 <= native_q(cfg['metadata'], peak) < 37, 'Selected peak fails geometry')
    fixed = cfg['fixed_poses'][0]
    model, geometry = geometry_model(cfg['metadata'], fixed, peak, shape_sha, old['angular_length'])
    archive = out/'provenance'; archive.mkdir(parents=True)
    sources = {'input-config.json': SOURCE/'config.json', 'shape.json': shape,
        'candidates.json': CANDIDATES, 'old-competitor-model.json': SOURCE/'model-competitor.json',
        'old-competitor-region.json': SOURCE/'region-competitor-r3.json',
        'old-freeze.json': SOURCE/'report.json', **local_dependencies([Path(__file__)])}
    for name, path in sources.items(): shutil.copy2(path, archive/name)
    frozen_cfg = copy.deepcopy(cfg); frozen_cfg['shape'] = str(archive/'shape.json')
    write(out/'config.json', frozen_cfg); write(out/'model.json', model)
    write(out/'selected-pose.json', chosen); write(out/'selected-original-row.json', original)
    write(out/'geometry.json', geometry)
    campaigns = []
    for radius, base in zip(RADII, SEEDS):
        label = f'{radius:g}'.replace('.', 'p')
        region = dict(fixed_neighbor=fixed, physical_fixed_neighbors=cfg['fixed_poses'],
            capture_center=cfg['capture_center'], capture_radius=cfg['capture_radius'],
            activity=cfg['reservoir_density'], depletant_radius=cfg['depletant_radius'],
            physical_metric=cfg['metadata'], shape_sha256=shape_sha, gaussian_chart=model,
            minimum_original_q=5., maximum_original_q=37., minimum_original_q_inclusive=True,
            maximum_original_q_inclusive=False, minimum_mahalanobis_radius=0., mahalanobis_radius=radius,
            definition='Frozen geometry-scaled ball around one historical AB far maximum, intersected with original q>=5, capture and both hard neighbors. A local region only.')
        path = out/f'region-r{label}.json'; write(path, region)
        output = ROOT/f'runs/ab-far-peak-reference-20260920/r{label}'
        require(not output.exists(), 'Reference output already exists')
        campaigns.append(dict(radius_A=radius, region=str(path), region_sha256=sha(path), output=str(output),
            populations=POPULATIONS, samples_per_population=SAMPLES, seed_base=base,
            seeds=[base+1009*i for i in range(POPULATIONS)]))
    protocol = dict(schema='AB-far-geometry-peak-reference-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        physical={k: cfg[k] for k in PHYSICAL}, original_q_window=dict(minimum=5., maximum=37., lower_inclusive=True, upper_inclusive=False),
        selection_source=dict(path=str(CANDIDATES), sha256=sha(CANDIDATES), selected_peak=index),
        selection='Highest original width-four far contribution. Both high-weight discoveries are retained as evidence, not reused as independent reference draws.',
        sampling='Uniform latent six-balls with exact center/Haar Jacobian. All unconditional draws and zeros retained; nested estimates are not added.',
        analysis='Independent original-density/Poisson audit; radial shells and intersection with unchanged old competitor R3, preserving row covariance and independent population errors.',
        executable=str(BINARY), executable_sha256=BINARY_SHA, lambda_ratio=64., cloud_replicates=2,
        campaigns=campaigns, probe_count_per_radius=PROBES, probe_seeds=[PROBE_SEED+1009*i for i in range(len(RADII))],
        old_region_sha256=sha(SOURCE/'region-competitor-r3.json'), selected_atomic_gaps_A=gaps,
        input_sha256={str(p): sha(p) for p in sources.values()}, archived_sha256={n: sha(archive/n) for n in sources},
        scope='Finite-neighborhood calibration. Full q>=5 is measured separately; selected-point weight is never substituted for an integral.')
    write(out/'protocol.json', protocol)
    write(out/'freeze.json', {p.name: sha(p) for p in out.glob('*.json')})
    reports = []
    for i, radius in enumerate(RADII):
        poses, latent, x, logj = uniform_draws(model, fixed, radius, PROBES, PROBE_SEED+1009*i)
        checks = check_coordinates(model, fixed, poses, latent, x, logj)
        checks.update(verify_member_geometry(cfg['metadata'], fixed, peak, poses, latent, x, model, geometry, logj))
        rows = []
        for j, pose in enumerate(poses):
            q = native_q(cfg['metadata'], pose); local_gaps = atom.gaps(pose)
            capture = float(np.linalg.norm(np.asarray(pose['position'])-cfg['capture_center'])) <= cfg['capture_radius']
            rows.append(dict(draw=j, pose=pose, q=q, atomic_gaps_A=local_gaps,
                valid=bool(capture and min(local_gaps) >= 0 and 5 <= q < 37), log_jacobian=float(logj[j])))
        label = f'{radius:g}'.replace('.', 'p'); path = out/f'geometry-probes-r{label}.jsonl'
        with path.open('x') as handle:
            for row in rows: handle.write(json.dumps(row, allow_nan=False)+'\n')
        reports.append(dict(radius_A=radius, valid=sum(r['valid'] for r in rows), unconditional_draws=PROBES,
            checks=checks, probe_sha256=sha(path)))
    observed_radii = Density(model).evaluate(relative_poses([p['pose'] for p in candidates['candidates']], fixed))[1][:, 0]
    write(out/'report.json', dict(complete=True, probes=reports,
        candidate_geometric_radii_A=observed_radii.tolist(),
        freeze_sha256=sha(out/'freeze.json'), no_physical_weight_draws=True,
        scope='Geometry probes only. No model refit, physical-weight tuning or allocation change.'))
    print(json.dumps(dict(preparation=str(out), probes=reports), allow_nan=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); prepare(args.out.resolve())
