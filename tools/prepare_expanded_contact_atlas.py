#!/usr/bin/env python3
"""Freeze a two-contact atlas and a fixed complete-window comparison budget."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import numpy as np

from analyze_native_region_reference import GaussianGuide
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_contact_atlas import BINARY, BINARY_SHA, CENTERS as OLD_CENTERS, FIT as OLD_FIT
from prepare_intermediate_guides import geometry_probe, PHYSICAL
from prepare_intermediate_local_region import ROOT, GUIDES
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_peak_neighborhood import geometry_model
from prepare_smc_normalizer_atlas import Density, arrays, relative_poses
from run_shoulder_mis_campaign import local_dependencies

CENTERS = ROOT/'runs/ab-intermediate-expanded-contact-centers-20260920/selection.json'
NEW_FIT = ROOT/'runs/ab-intermediate-atlas-peak-local-fit-20260920'
ARMS = (('narrow', .2, 102101010), ('broad', .4, 102201010))
SAMPLES, POPULATIONS = 32768, 8
PREFIXES = (8192, 16384, 32768)


def assemble(cfg, centers, fits, width):
    require(len(fits) == 2 and len(centers) > 0 and math.isfinite(width) and width > 0,
            'Need two local fits, geometric centers and a positive width')
    keys = ('schema', 'angular_length', 'coordinate_convention', 'shape_sha256')
    model = {key:copy.deepcopy(fits[0][key]) for key in keys}
    model.update(anchors=[], means=[], covariances=[], weights=[])
    for point in centers:
        geometric, _ = geometry_model(cfg['metadata'], cfg['fixed_poses'][0], point['pose'],
                                      model['shape_sha256'], model['angular_length'])
        model['anchors'].extend(geometric['anchors']); model['means'].extend(geometric['means'])
        model['covariances'].append((width**2*np.asarray(geometric['covariances'][0])).tolist())
        model['weights'].append(.6/len(centers))
    for fit in fits:
        require(all(fit[key] == model[key] for key in keys), 'Fit coordinate convention or shape changed')
        require(fit['weights'] == [1.] and len(fit['anchors']) == len(fit['means']) == len(fit['covariances']) == 1,
                'Each local fit must have one component')
        for covariance_scale, weight in ((1., .16), (4., .04)):
            model['anchors'].extend(copy.deepcopy(fit['anchors']))
            model['means'].extend(copy.deepcopy(fit['means']))
            model['covariances'].append((covariance_scale*np.asarray(fit['covariances'][0])).tolist())
            model['weights'].append(weight)
    require(abs(sum(model['weights'])-1.) < 2e-12, 'Unnormalized mixture')
    return model


def check_archive(directory, mapping):
    for name, digest in mapping.items():
        require(sha(directory/name) == digest, 'Changed archived artifact: '+name)


def prepare(out):
    require(not out.exists(), 'Use a fresh preparation directory')
    centers = read(CENTERS); old_centers = read(OLD_CENTERS)
    require(centers['complete'] and centers['no_physical_draws'], 'Incomplete center extension')
    check_archive(CENTERS.parent/'provenance', centers['archived_sha256'])
    for name, digest in centers.get('input_sha256', {}).items():
        require(sha(Path(name)) == digest, 'Center source changed')
    fits = [read(directory/'model-weighted.json') for directory in (OLD_FIT, NEW_FIT)]
    reports = [read(directory/'report.json') for directory in (OLD_FIT, NEW_FIT)]
    require(all(r['complete'] and r['no_new_geometry_or_physical_draws'] for r in reports), 'Incomplete fit')
    check_archive(OLD_FIT, read(OLD_FIT/'artifact-hashes.json'))
    for name, entry in read(OLD_FIT/'provenance-sha256.json').items():
        require(sha(OLD_FIT/'provenance'/name) == entry['sha256'], 'Old fit source changed')
    check_archive(NEW_FIT/'provenance', reports[1]['archived_sha256'])
    for name, digest in reports[1]['input_sha256'].items():
        require(sha(Path(name)) == digest, 'New fit source changed')
    cfg = read(GUIDES/'config.json'); shape = Path(cfg['shape']); source = read(GUIDES/'provenance/source-region.json')
    require(sha(GUIDES/'config.json') == read(GUIDES/'freeze.json')['config_sha256'], 'Original config changed')
    require(sha(shape) == fits[0]['shape_sha256'] == fits[1]['shape_sha256'], 'Shape changed')
    require(sha(BINARY) == BINARY_SHA, 'Physical kernel changed')
    require(source['physical_metric'] == cfg['metadata'] and source['physical_fixed_neighbors'] == cfg['fixed_poses'],
            'Original region target differs')
    require((source['minimum_original_q'], source['maximum_original_q'], source['minimum_original_q_inclusive'],
             source['maximum_original_q_inclusive']) == (2., 5., True, False), 'Original window changed')
    require(cfg['capture_radius'] == 18. and cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035,
            'Physical baseline changed')
    for report in reports:
        require(all(report['physical'][key] == cfg[key] for key in PHYSICAL), 'Different fit environment')
    selected = centers['selected_centers']; indexes = centers['selected_indices']; candidates = centers['candidates']
    require([p['pose'] for p in selected[:36]] == [p['pose'] for p in old_centers['selected_centers']],
            'Original geometric centers changed')
    peak = read(ROOT/'runs/ab-intermediate-atlas-peak-geometry-20260920/selected-pose.json')['pose']
    require(selected[36]['pose'] == peak, 'New maximum must be an explicit geometric center')
    members, _, _ = arrays(cfg['metadata']['rigid_members']); t, _, r = arrays([p['pose'] for p in candidates])
    world = np.einsum('nij,mj->nmi', r, members)+t[:, None, :]
    distances = np.sqrt(np.mean(np.sum((world[:, None]-world[np.asarray(indexes)][None, :])**2, axis=3), axis=2))
    require(float(distances.min(axis=1).max()) <= .5, 'Observed candidate cover failed')
    for point, index in zip(selected, indexes):
        require(point['pose'] == candidates[index]['pose'], 'Center index/pose differs')
    models = {name:assemble(cfg, selected, fits, width) for name, width, _ in ARMS}
    archive = out/'provenance'; archive.mkdir(parents=True)
    sources = {'input-config.json':GUIDES/'config.json', 'source-region.json':GUIDES/'provenance/source-region.json',
               'shape.json':shape, 'center-selection.json':CENTERS, 'original-centers.json':OLD_CENTERS,
               'old-fit-model.json':OLD_FIT/'model-weighted.json', 'old-fit-report.json':OLD_FIT/'report.json',
               'new-fit-model.json':NEW_FIT/'model-weighted.json', 'new-fit-report.json':NEW_FIT/'report.json',
               **local_dependencies([Path(__file__)])}
    for name, path in sources.items(): shutil.copy2(path, archive/name)
    frozen_cfg = copy.deepcopy(cfg); frozen_cfg['shape'] = str(archive/'shape.json'); write(out/'config.json', frozen_cfg)
    def records(entries):
        return {name:dict(path=str(ROOT/path), sha256=sha(ROOT/path)) for name, path in entries}
    charts = records((('old_peak', 'runs/ab-intermediate-new-peak-geometry-20260920/model.json'),
                      ('new_peak', 'runs/ab-intermediate-atlas-peak-geometry-20260920/model.json'),
                      ('old_weighted', 'runs/ab-intermediate-guide-preparation-20260920/model-weighted.json')))
    references = records((('old_peak', 'runs/ab-intermediate-new-peak-audit-20260920/analysis.json'),
                          ('new_peak', 'runs/ab-intermediate-atlas-peak-reference-audit-20260920/analysis.json')))
    totals = records((('old_peak', 'runs/ab-intermediate-new-peak-multiscale-20260920/analysis.json'),
                      ('new_peak', 'runs/ab-intermediate-atlas-peak-multiscale-20260920/analysis.json')))
    protocol = dict(schema='intermediate-expanded-contact-atlas-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        physical={key:copy.deepcopy(cfg[key]) for key in PHYSICAL},
        q_window=dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False),
        center_selection=dict(path=str(CENTERS), sha256=sha(CENTERS), count=len(selected),
            observed_candidate_count=len(candidates), observed_maximum_nearest_member_RMS_A=float(distances.min(axis=1).max())),
        local_fits=[dict(path=str(directory/'model-weighted.json'), sha256=sha(directory/'model-weighted.json')) for directory in (OLD_FIT, NEW_FIT)],
        assembly='60% equal geometric centers +16% old local fit +4% old fit at fourfold covariance +16% new local fit +4% new fit at fourfold covariance.',
        hybrid='25% complete q<=5 product cover +75%*(95% frozen atlas +5% capture cube with normalized Haar); full mixture denominator on every unconditional draw.',
        analysis_charts=charts, reference_audits=references, reference_totals=totals,
        arms=[dict(name=name, geometric_latent_SD_A=width, components=len(models[name]['weights']),
            samples_per_population=SAMPLES, populations=POPULATIONS, seeds=[seed+1009*i for i in range(POPULATIONS)],
            output=str(ROOT/f'runs/ab-intermediate-expanded-atlas-{name}-8x32768-l64-20260920')) for name, width, seed in ARMS],
        prefix_counts=list(PREFIXES),
        prefix_rule='First 8192, 16384 and all 32768 unconditional draws within each of eight populations. Prefixes are correlated sensitivity diagnostics; no stopping, discarded populations or independent-significance claim.',
        analysis='Independent original-density all-row audit. Old/new balls at radii .5,1,2; direct disjoint old R2, new R2, and outsideboth partition, with same-row covariance. Width arms remain separate.',
        precision_screen='Report row and population relative SE, largest contribution, prefix sensitivity and width differences. Small observed errors do not bound unseen tails or certify full-window convergence.',
        probe_count_per_arm=256, probe_seeds=[102001010, 102002019],
        executable=str(BINARY), executable_sha256=BINARY_SHA, lambda_ratio=64., cloud_replicates=2,
        selection_scope='Centers and fits use historical references; local comparisons are calibration. All new production is independent after freezing both models. No refit or adaptive stop.',
        coverage_scope='Observed-point cover is not a basin cover. Product and cube components preserve complete target support. The entire outsideboth region is retained.',
        input_sha256={str(path):sha(path) for path in sources.values()},
        archived_sha256={name:sha(archive/name) for name in sources})
    write(out/'protocol.json', protocol)
    for name, model in models.items():
        model['proposal_provenance'] = dict(kind='Frozen expanded contact atlas', arm=name, protocol_sha256=sha(out/'protocol.json'),
            physical=protocol['physical'], proposal_anchor_index=0, inference=protocol['selection_scope'])
        write(out/f'model-{name}.json', model)
    seal = dict(protocol_sha256=sha(out/'protocol.json'), config_sha256=sha(out/'config.json'),
        model_sha256={name:sha(out/f'model-{name}.json') for name in models}, archived_sha256=protocol['archived_sha256'])
    write(out/'freeze.json', seal)
    atom = AtomUnionAudit(read(shape), cfg['fixed_poses']); probes = {}
    for i, (name, _, _) in enumerate(ARMS):
        report, rows = geometry_probe(models[name], cfg, atom, 256, protocol['probe_seeds'][i])
        poses = [row['pose'] for row in rows]; pure = Density(models[name]).evaluate(relative_poses(poses, cfg['fixed_poses'][0]))[0]
        description = dict(weight=.75, uniform_probability=.05, anchor_index=0, anchor_pose=cfg['fixed_poses'][0],
                           capture_center=cfg['capture_center'], cube_lengths=[36.]*3)
        scalar = GaussianGuide(description, models[name], cfg, fits[0]['shape_sha256'])
        inside = np.array([all(-18 <= pose['position'][k]-cfg['capture_center'][k] < 18 for k in range(3)) for pose in poses])
        expected = np.logaddexp(math.log(.95)+pure, np.where(inside, math.log(.05)-3*math.log(36.), -math.inf))
        error = float(np.max(np.abs([scalar.log_density(p)[0]-e for p, e in zip(poses, expected)])))
        require(error < 2e-9, 'Independent scalar/vectorized mixture density disagreement')
        report['scalar_vectorized_density_max_error'] = error
        with (out/f'geometry-probes-{name}.jsonl').open('w') as handle:
            for row in rows: handle.write(json.dumps(row, allow_nan=False)+'\n')
        probes[name] = report
        print(json.dumps(dict(arm=name, centers=len(selected), components=len(models[name]['weights']),
                             valid=report['valid_original_window'], draws=256, density_error=error)), flush=True)
    require(sha(out/'protocol.json') == seal['protocol_sha256'], 'Protocol changed after freezing')
    for name, digest in seal['model_sha256'].items(): require(sha(out/f'model-{name}.json') == digest, 'Model changed after freezing')
    write(out/'report.json', dict(complete=True, probes=probes, freeze_sha256=sha(out/'freeze.json'),
        probe_sha256={name:sha(out/f'geometry-probes-{name}.jsonl') for name in models},
        scope='Cloud-free geometry and normalized-density validation only; physical draws use the frozen production protocol.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); prepare(args.out.resolve())
