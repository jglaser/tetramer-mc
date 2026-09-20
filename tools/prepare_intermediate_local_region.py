#!/usr/bin/env python3
"""Freeze a local weighted-chart control without drawing or refitting poses."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / 'runs/ab-intermediate-guide-preparation-20260920'
BINARY = ROOT / 'runs/ab-shoulder-cayley-reference-4x262144-l64-20260920/provenance/latent-region-normalizer'
BINARY_SHA = 'd5c112c8c9a93fb3914d935d087acb2a3b0cf222d89f3bd2614a1d63826bab0d'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(out, radius, samples, replicates, seed):
    assert math.isfinite(radius) and radius > 0
    assert min(samples, replicates) > 0 and seed >= 0
    assert not out.exists(), 'Use a fresh directory'
    freeze = read(GUIDES / 'freeze.json')
    for name, digest in freeze['archived_sha256'].items():
        assert sha(GUIDES / 'provenance' / name) == digest
    assert sha(GUIDES / 'protocol.json') == freeze['protocol_sha256']
    assert sha(GUIDES / 'config.json') == freeze['config_sha256']
    assert sha(GUIDES / 'model-weighted.json') == freeze['model_sha256']['weighted']
    assert sha(BINARY) == BINARY_SHA
    model = read(GUIDES / 'model-weighted.json')
    config = read(GUIDES / 'config.json')
    source = read(GUIDES / 'provenance/source-region.json')
    assert model['weights'] == [1.0] and len(model['means']) == len(model['covariances']) == 1
    assert source['physical_fixed_neighbors'] == config['fixed_poses']
    assert source['physical_metric'] == config['metadata']
    assert model['shape_sha256'] == source['shape_sha256'] == sha(Path(config['shape']))
    assert config['reservoir_density'] == source['activity'] == .035
    assert config['depletant_radius'] == source['depletant_radius'] == 1.5
    assert config['capture_radius'] == source['capture_radius'] == 18.
    assert source['minimum_original_q'] == 2 and source['maximum_original_q'] == 5
    assert source['minimum_original_q_inclusive'] and not source['maximum_original_q_inclusive']
    region = {key: source[key] for key in (
        'fixed_neighbor', 'physical_fixed_neighbors', 'capture_center', 'capture_radius',
        'shape_sha256', 'activity', 'depletant_radius', 'physical_metric',
        'minimum_original_q', 'maximum_original_q',
        'minimum_original_q_inclusive', 'maximum_original_q_inclusive')}
    region.update(gaussian_chart=model, mahalanobis_radius=radius,
                  minimum_mahalanobis_radius=0.,
                  definition='Frozen weighted-chart ball intersected with original 2<=q<5, capture, and full AB hard support. Local region only; no complete physical coverage claim.',
                  chart_model_sha256=freeze['model_sha256']['weighted'])
    provenance = out / 'provenance'
    provenance.mkdir(parents=True)
    sources = {'prepare_intermediate_local_region.py': Path(__file__),
               'guide-freeze.json': GUIDES / 'freeze.json',
               'guide-protocol.json': GUIDES / 'protocol.json',
               'model-weighted.json': GUIDES / 'model-weighted.json',
               'source-region.json': GUIDES / 'provenance/source-region.json'}
    for name, path in sources.items():
        shutil.copy2(path, provenance / name)
    shutil.copy2(GUIDES / 'config.json', out / 'config.json')
    write(out / 'region.json', region)
    t = radius * radius / 2
    probability = 1 - math.exp(-t) * (1 + t + t*t/2)
    protocol = dict(
        schema='intermediate-local-reference-plan-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        region_sha256=sha(out / 'region.json'), config_sha256=sha(out / 'config.json'),
        chart_model_sha256=freeze['model_sha256']['weighted'],
        executable=str(BINARY), executable_sha256=BINARY_SHA,
        samples_per_population=samples, replicates=replicates,
        seeds=[seed + 1009*i for i in range(replicates)],
        lambda_ratio=64., cloud_replicates=2,
        radial_partition_edges=[0., radius/4, radius/2, 3*radius/4, radius],
        radial_endpoint_convention='First bin [0,R/4]; subsequent bins lower open, upper closed.',
        unconditional_Gaussian_ball_probability=probability,
        selection='Chart learned from earlier direct reference; radius chosen after guide diagnosis, before fresh local draws. Existing guide restrictions are retrospective diagnostics. New uniform draws independently integrate this now-fixed region; no pooled normalizer or held-out region discovery claim.',
        sampling='Uniform latent six-ball, exact pose Jacobian, original target masks; invalid draws remain zeros in fixed N.',
        scope='Local physical mass and hard volume only. Gaussian probability is not physical coverage. Complement remains explicit and unresolved.',
        analysis='Separate full-N ball, radial shells and guide complement; reconstruct densities and Jacobians; report row and population uncertainty, weight concentration and paired-cloud noise. No optional stopping or refitting within this campaign.',
        archive_sha256={name: sha(provenance / name) for name in sources})
    write(out / 'protocol.json', protocol)
    write(out / 'freeze.json', {name: sha(out / name) for name in ('region.json', 'config.json', 'protocol.json')})
    print(json.dumps(dict(out=str(out), region_sha256=protocol['region_sha256'],
                         protocol_sha256=sha(out / 'protocol.json'),
                         Gaussian_ball_probability=probability)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--radius', type=float, default=4.)
    parser.add_argument('--samples', type=int, default=16384)
    parser.add_argument('--replicates', type=int, default=4)
    parser.add_argument('--seed', type=int, default=99971010)
    args = parser.parse_args()
    prepare(args.out.resolve(), args.radius, args.samples, args.replicates, args.seed)
