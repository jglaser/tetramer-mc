#!/usr/bin/env python3
"""Audit expanded contact atlases and predeclared, correlated sample prefixes.

Every source keeps its original full proposal density and unconditional sample
count. The original two geometric neighborhoods and remaining space form a
direct disjoint partition, with same-row covariance. Prefixes share draws and
must not be treated as independent validation or used for optional stopping.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import heapq
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_contact_atlas import AtlasMoments, KEYS as ATLAS_KEYS, PHYSICAL_KEYS, WINDOW
from analyze_peak_neighborhood import geometry_model_audit, partition_check
from audit_shoulder_mis_independently import chart_values, near
from compare_intermediate_local_reference import PartitionMoments, KEYS as LOCAL_KEYS, read_batches
from compare_intermediate_reference import load_guided, population_rse
from diagnose_atlas_peak_regions import center_separation, threeway_add, THREEWAY_KEYS
from prepare_cayley_rms_cover import read, require, sha, write
from prepare_native_confirmation_atlas import AtomUnionAudit
from run_shoulder_mis_campaign import local_dependencies


def validate_prefix_counts(samples, prefix_counts):
    """Validate frozen counts without prescribing one particular run length."""
    require(type(samples) is int and samples >= 2, 'Require an integer unconditional sample budget')
    require(isinstance(prefix_counts, (list, tuple)) and bool(prefix_counts) and
            all(type(count) is int and count >= 2 for count in prefix_counts),
            'Require nonempty integer prefix counts')
    require(all(a < b for a, b in zip(prefix_counts, prefix_counts[1:])) and prefix_counts[-1] == samples,
            'Require increasing fixed prefix counts ending at full N')


def validate_prefix_schedule(protocol):
    require(protocol['arms'], 'Require at least one frozen arm')
    for arm in protocol['arms']:
        validate_prefix_counts(arm['samples_per_population'], protocol['prefix_counts'])


def validate_repeat_source(protocol, freeze):
    """A larger repeat may change allocations, never the frozen model bytes."""
    declared = protocol.get('repeat_source')
    if declared is None:
        return None, {}
    path = Path(declared['path']).resolve(); require(sha(path) == declared['sha256'], 'Original preparation protocol changed')
    source, source_freeze = read(path), read(path.parent/'freeze.json')
    require(source['schema'] == protocol['schema'] and source_freeze['protocol_sha256'] == sha(path),
            'Original source preparation is not frozen consistently')
    require(declared['model_sha256'] == source_freeze['model_sha256'] == freeze['model_sha256'],
            'Repeat proposal models were changed or refitted')
    for key in ('physical', 'q_window', 'analysis_charts', 'reference_audits', 'reference_totals',
                'executable_sha256', 'lambda_ratio', 'cloud_replicates'):
        require(protocol[key] == source[key], f'Repeat changed a physical/proposal control: {key}')
    source_seeds = {seed for arm in source['arms'] for seed in arm['seeds']}
    repeat_seeds = {seed for arm in protocol['arms'] for seed in arm['seeds']}
    require(not source_seeds.intersection(repeat_seeds), 'Repeat reuses original population streams')
    for arm, digest in declared['model_sha256'].items():
        require(sha(path.parent/f'model-{arm}.json') == digest, 'Original model bytes changed')
    return dict(path=str(path), sha256=sha(path), model_sha256=declared['model_sha256'],
        disjoint_population_seeds=True, models_unchanged=True,
        qualification='Independent fixed-model repetition, conditional on the already selected physical domain and analysis regions. Original and repeated estimates remain separate.'), {
            'repeat-source-protocol.json': path, 'repeat-source-freeze.json': path.parent/'freeze.json'}


class ExpandedMoments:
    """Reuse audited masks for the old/new geometry and old weighted charts."""
    def __init__(self):
        self.old = AtlasMoments()
        self.new = PartitionMoments(2.)
        self.threeway = PartitionMoments(2.)

    def add(self, old_radius=None, new_radius=None, weighted_radius=None,
            physical=None, hard=None, cloud_pair=None):
        self.old.add(old_radius, weighted_radius, physical, hard, cloud_pair)
        self.new.add(new_radius, physical, hard, cloud_pair)
        threeway_add(self.threeway, new_radius, old_radius, physical, hard, cloud_pair)

    def merge(self, other):
        self.old.merge(other.old); self.new.merge(other.new); self.threeway.merge(other.threeway)

    def report(self):
        result = self.old.report()
        result['new_neighborhood'] = self.new.report()
        result['new_neighborhood']['partition_checks'] = {
            kind: partition_check(values, ('ball', 'complement')) for kind, values in self.new.values.items()}
        raw = self.threeway.report()
        result['disjoint_threeway'] = {kind: {name: raw[kind][key] for name, key in THREEWAY_KEYS.items()}
            for kind in ('physical', 'hard')}
        checks = {kind: partition_check(values, ('inner_half', 'outer_half', 'complement'))
            for kind, values in self.threeway.values.items()}
        inverse = {value: key for key, value in THREEWAY_KEYS.items()}
        for check in checks.values():
            for covariance in check['covariances']:
                covariance['left'] = inverse[covariance['left']]
                covariance['right'] = inverse[covariance['right']]
        result['disjoint_threeway']['partition_checks'] = checks
        return result


class PrefixMoments:
    """Fixed first-m-draw estimates; invalid trials still advance every prefix."""
    def __init__(self, samples, prefix_counts):
        validate_prefix_counts(samples, prefix_counts)
        self.samples = samples; self.count = 0
        self.prefixes = {count: ExpandedMoments() for count in prefix_counts}

    def add(self, draw, *args):
        require(draw == self.count and draw < self.samples, 'Missing, duplicate or extra prefix row')
        for maximum, moments in self.prefixes.items():
            if draw < maximum: moments.add(*args)
        self.count += 1

    def complete(self):
        require(self.count == self.samples, 'Incomplete unconditional population')


def finish(aggregate, populations):
    require(populations and len({p['samples'] for p in populations}) == 1, 'Population errors require equal fixed N')
    result = aggregate.report()
    for kind in ('physical', 'hard'):
        for key in ATLAS_KEYS:
            result[kind][key]['independent_population_RSE'] = population_rse([p[kind][key]['logQ'] for p in populations])
        for section, keys in (('new_neighborhood', LOCAL_KEYS), ('disjoint_threeway', THREEWAY_KEYS)):
            for key in keys:
                result[section][kind][key]['independent_population_RSE'] = population_rse(
                    [p[section][kind][key]['logQ'] for p in populations])
    result['populations'] = populations
    return result


def nested_prefix_comparison(prefix, full):
    """IID finite-variance relation; estimate variance with the full stream.

    Cov(mean_m,mean_N)=Var(W)/N, so Var(mean_m-mean_N)=(N/m-1)*Var(mean_N).
    This accounts for shared rows; neither observed variance nor this relation
    protects against unseen tails. Physical and cloud noise both enter W.
    """
    m, n = prefix['draws'], full['draws']
    require(2 <= m < n, 'Need a proper nested prefix')
    ratio = n/m
    if full['logQ'] is None:
        require(prefix['logQ'] is None, 'Nonzero prefix inside zero full total')
        delta = uncertainty = logvar = None
    else:
        delta = -1. if prefix['logQ'] is None else math.expm1(prefix['logQ']-full['logQ'])
        uncertainty = math.sqrt(ratio-1.)*full['row_RSE']
        logvar = (full['log_variance_of_mean']+math.log(ratio-1.)
                  if full['log_variance_of_mean'] is not None else None)
    return dict(prefix_draws=m, full_draws=n, relative_prefix_minus_full=delta,
        observed_SE_difference_relative_to_full_mean=uncertainty,
        estimated_log_variance_of_difference=logvar,
        estimated_log_covariance_of_means=full['log_variance_of_mean'],
        theoretical_IID_correlation=math.sqrt(m/n),
        qualification='Correlated same-stream estimates. Covariance and difference variance use the full-stream row variance under the fixed IID law, conditional on finite variance. Observed errors do not bound unseen tails.')


def audit_campaign(root, arm, preparation, protocol, models, selected):
    root = Path(root).resolve(); cfg = read(root/'provenance/config.json'); master = read(root/'manifest.json')
    declared = next(item for item in protocol['arms'] if item['name'] == arm)
    prepared_cfg = read(preparation/'config.json')
    require(Path(declared['output']).resolve() == root, 'Campaign differs from frozen arm')
    require(all(cfg[key] == prepared_cfg[key] for key in PHYSICAL_KEYS), 'Physical target changed')
    require(master['q_window'] == protocol['q_window'] == WINDOW, 'Original q window changed')
    require(master['total_unconditional_draws'] == declared['populations']*declared['samples_per_population'], 'Unconditional budget changed')
    require([j['seed'] for j in master['jobs']] == declared['seeds'] and len(master['jobs']) == declared['populations'],
            'Fixed independent populations changed')
    require(sha(root/'provenance/guide-model.json') == sha(preparation/f'model-{arm}.json'), 'Frozen proposal changed')
    for job in master['jobs']:
        manifest = read(Path(job['output'])/'manifest.json'); summary = read(Path(job['output'])/'summary.json')
        require(summary['complete'] and manifest['samples'] == summary['samples'] == declared['samples_per_population'],
                'Incomplete or changed population budget')
        require(manifest['cloud_replicates'] == protocol['cloud_replicates'] == 2 and
                manifest['lambda_ratio'] == protocol['lambda_ratio'] == 64., 'Poisson law changed')
        require(manifest['executable_sha256'] == protocol['executable_sha256'], 'Physical executable changed')
        require(manifest['guide']['model_sha256'] == sha(preparation/f'model-{arm}.json') and
                manifest['guide']['weight'] == .75 and manifest['guide']['uniform_probability'] == .05,
                'Complete frozen hybrid proposal changed')
    print(f'{arm}: auditing every original hybrid density, q, cloud factor and unconditional denominator', flush=True)
    original = load_guided(root)
    require(all(model['shape_sha256'] == original['shape_sha256'] for model in models.values()), 'Analysis shape differs')
    prefix_counts = protocol['prefix_counts']
    aggregate = {count: ExpandedMoments() for count in prefix_counts}
    populations = {count: [] for count in prefix_counts}; input_hashes = {}; heap = []; ordinal = 0
    for job in master['jobs']:
        directory = Path(job['output']); local = PrefixMoments(declared['samples_per_population'], prefix_counts)
        digest = hashlib.sha256()
        for lines, rows in read_batches(directory/'samples.jsonl'):
            for line in lines: digest.update(line)
            positions = np.asarray([row['pose']['position'] for row in rows])
            quats = np.asarray([row['pose']['orientation'] for row in rows])
            rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()
            radii = {name: chart_values(model, positions, rotations, cfg['fixed_poses'][0])[1][:, 0]
                     for name, model in models.items()}
            for i, row in enumerate(rows):
                if 'zero' in row:
                    local.add(row['draw']); continue
                physical, hard = row['log_importance_weight'], row['log_hard_weight']
                pair = [value+hard for value in row['cloud_log_weights']]
                near(float(np.logaddexp(*pair))-math.log(2), physical)
                old_radius, new_radius, weighted_radius = [float(radii[name][i]) for name in ('old_peak', 'new_peak', 'old_weighted')]
                local.add(row['draw'], old_radius, new_radius, weighted_radius, physical, hard, pair)
                point = dict(population=directory.name, seed=job['seed'], draw=row['draw'], pose=row['pose'], q=row['q'],
                    original_log_importance_weight=physical, original_log_proposal_density=row['log_proposal_density'],
                    log_boltzmann_mean=row['log_boltzmann_mean'], cloud_log_weights=row['cloud_log_weights'],
                    cloud_overlap_counts=row['cloud_overlap_counts'], lower_volume=row['lower_volume'],
                    old_peak_radius_A=old_radius, new_peak_radius_A=new_radius, old_weighted_radius=weighted_radius,
                    proposal_family=row['proposal_family'], guide_branch=row['guide_branch'], guide_component=row['guide_component'])
                entry = (physical, ordinal, point); ordinal += 1
                if len(heap) < 8: heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]: heapq.heapreplace(heap, entry)
        local.complete(); sample_path = str(directory/'samples.jsonl')
        require(digest.hexdigest() == original['sample_sha256'][sample_path], 'Rows changed after independent full-density audit')
        input_hashes[sample_path] = digest.hexdigest()
        for name in ('manifest.json', 'summary.json'): input_hashes[str(directory/name)] = sha(directory/name)
        for count in prefix_counts:
            populations[count].append(dict(id=directory.name, seed=job['seed'], samples=count, **local.prefixes[count].report()))
            aggregate[count].merge(local.prefixes[count])
    prefix_reports = {count: finish(aggregate[count], populations[count]) for count in prefix_counts}
    result = prefix_reports[prefix_counts[-1]]
    for kind in ('physical', 'hard'):
        for field in ('logQ', 'row_RSE'):
            actual, expected = result[kind]['full'][field], original[kind]['all'][field]
            if actual is None: require(expected is None, 'Full zero estimate changed')
            else: near(actual, expected)
    comparisons = []
    for count in prefix_counts[:-1]:
        current = prefix_reports[count]
        comparisons.append(dict(samples_per_population=count,
            physical={key: nested_prefix_comparison(current['disjoint_threeway']['physical'][key],
                                                    result['disjoint_threeway']['physical'][key])
                      for key in THREEWAY_KEYS},
            hard={key: nested_prefix_comparison(current['disjoint_threeway']['hard'][key],
                                                result['disjoint_threeway']['hard'][key])
                  for key in THREEWAY_KEYS}))
    atom = AtomUnionAudit(read(root/'provenance/shape.json'), cfg['fixed_poses']); top = []
    full_sum = aggregate[prefix_counts[-1]].old.values['physical']['full'].total
    for _, _, point in sorted(heap, reverse=True):
        gaps = atom.gaps(point['pose']); require(min(gaps) >= 0, 'Top-weight pose fails independent atom-union check')
        point['minimum_atomic_gap_by_neighbor_A'] = gaps
        point['fraction_of_full_weight'] = math.exp(point['original_log_importance_weight']-full_sum)
        top.append(point)
    print(f'{arm}: full logQ={result["physical"]["full"]["logQ"]}, RSE={result["physical"]["full"]["row_RSE"]}', flush=True)
    return dict(root=str(root), arm=arm, **result,
        prefixes=[dict(samples_per_population=count, **prefix_reports[count]) for count in prefix_counts],
        prefix_comparisons=comparisons, top_poses=top, independent_atom_check_count=len(top),
        CPU_seconds=original['CPU_seconds'], original_full_window_audit=original, input_sha256=input_hashes,
        prefix_qualification='Predeclared first-m draws in each population, including all zeros; prefixes share rows with later estimates and are correlated. No optional stopping, fresh seeds, pooling or independence claim.')


def reference_comparisons(campaigns, protocol):
    rows = []
    for label in ('old_peak', 'new_peak'):
        declared = protocol['reference_audits'][label]; path = Path(declared['path'])
        require(sha(path) == declared['sha256'], 'Historical local reference changed')
        audit = read(path); require(audit['complete'], 'Incomplete local reference')
        total_declared = protocol['reference_totals'][label]; total_path = Path(total_declared['path'])
        require(sha(total_path) == total_declared['sha256'], 'Historical multiscale reference changed')
        totals = read(total_path); require(totals['complete'] and totals['source_sha256'] == sha(path), 'Reference totals use another audit')
        for radius in (.5, 1., 2.):
            reference = next(c for c in audit['campaigns'] if c['radius_A'] == radius)
            region_path = Path(reference['root'])/'provenance/region.json'; region = read(region_path)
            original = reference['original_reference_audit']
            require(sha(region_path) == original['region_sha256'], 'Historical reference region bytes changed')
            require(region['gaussian_chart'] == read(Path(protocol['analysis_charts'][label]['path'])) and
                    region['mahalanobis_radius'] == radius and region['minimum_mahalanobis_radius'] == 0.,
                    'Reference and fresh chart masks differ')
            require((region['minimum_original_q'], region['maximum_original_q'],
                     region['minimum_original_q_inclusive'], region['maximum_original_q_inclusive']) == (2., 5., True, False),
                    'Reference original q window differs')
            require(all(original['physical_signature'][key] == protocol['physical'][key] for key in PHYSICAL_KEYS),
                    'Historical reference physical target differs')
            local_key = {.5: 'peak_r_le_0p5', 1.: 'peak_r_le_1', 2.: 'peak_r_le_2'}[radius]
            new_key = {.5: 'shell0', 1.: 'inner_half', 2.: 'ball'}[radius]
            estimates = {c['arm']: c['physical'][local_key] if label == 'old_peak'
                         else c['new_neighborhood']['physical'][new_key] for c in campaigns}
            summed = next((t for t in totals['totals'] if t['radius_A'] == radius), None)
            rows.append(dict(neighborhood=label, radius_A=radius, fresh_atlases=estimates,
                historical_uniform_reference=reference['physical']['full'],
                historical_multiscale_reference=summed['physical'] if summed else reference['physical']['full'],
                qualification='Fixed matched physical masks. Historical local samples informed atlas construction, so this is calibration against independently sampled fresh production, not untouched holdout validation. No pooling or retrospective density substitution.'))
    return rows


def analyze(preparation, campaign_paths, out, workers=1):
    preparation, out = Path(preparation).resolve(), Path(out).resolve()
    require(not out.exists(), 'Use a fresh audit directory'); started = time.monotonic()
    require(type(workers) is int and workers >= 1, 'Require a positive independent-audit worker count')
    protocol, freeze = read(preparation/'protocol.json'), read(preparation/'freeze.json')
    require(protocol['schema'] == 'intermediate-expanded-contact-atlas-v1', 'Unexpected frozen protocol')
    require(sha(preparation/'protocol.json') == freeze['protocol_sha256'] and
            sha(preparation/'config.json') == freeze['config_sha256'], 'Frozen preparation changed')
    for name, digest in freeze['archived_sha256'].items():
        require(sha(preparation/'provenance'/name) == digest, 'Frozen preparation archive changed')
    cfg = read(preparation/'config.json')
    require(cfg['capture_radius'] == 18. and cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035,
            'Physical baseline changed')
    require(all(cfg[key] == protocol['physical'][key] for key in PHYSICAL_KEYS), 'Protocol target changed')
    validate_prefix_schedule(protocol)
    repeat_source, repeat_inputs = validate_repeat_source(protocol, freeze)
    models, selected, geometries = {}, {}, {}
    sources = {'protocol.json': preparation/'protocol.json', 'freeze.json': preparation/'freeze.json',
               'config.json': preparation/'config.json', **repeat_inputs}
    for name, declaration in protocol['analysis_charts'].items():
        path = Path(declaration['path']); require(sha(path) == declaration['sha256'], 'Frozen chart changed')
        models[name] = read(path); sources[f'chart-{name}.json'] = path
        if name != 'old_weighted':
            peak_path = path.parent/'selected-pose.json'; selected[name] = read(peak_path)['pose']
            geometries[name], _ = geometry_model_audit(models[name], dict(fixed_neighbor=cfg['fixed_poses'][0]), cfg, selected[name])
            sources[f'pose-{name}.json'] = peak_path
    require(set(models) == {'old_peak', 'new_peak', 'old_weighted'}, 'Analysis charts changed')
    separation = center_separation(cfg, selected['new_peak'], selected['old_peak'])
    require(separation['two_radius_2_balls_disjoint'], 'Reference balls not disjoint')
    for label, declarations in (('reference', protocol['reference_audits']), ('totals', protocol['reference_totals'])):
        for name, declaration in declarations.items():
            path = Path(declaration['path']); require(sha(path) == declaration['sha256'], 'Frozen reference changed')
            sources[f'{label}-{name}.json'] = path
    for arm in protocol['arms']:
        path = preparation/f'model-{arm["name"]}.json'
        require(sha(path) == freeze['model_sha256'][arm['name']], 'Frozen arm model changed')
        sources[f'model-{arm["name"]}.json'] = path
    archive = out/'provenance'; archive.mkdir(parents=True)
    for name, path in {**local_dependencies([Path(__file__)]), **sources}.items():
        (archive/name).write_bytes(path.read_bytes())
    input_hashes = {str(path): sha(path) for path in sources.values()}
    records = []; arm_names = set(); seeds = set(); requests = []
    for path in campaign_paths:
        path = Path(path).resolve()
        arm = next((a for a in protocol['arms'] if Path(a['output']).resolve() == path), None)
        require(arm is not None and arm['name'] not in arm_names, 'Unknown or repeated arm')
        arm_names.add(arm['name']); require(not seeds.intersection(arm['seeds']), 'Repeated population stream')
        require(len(set(arm['seeds'])) == len(arm['seeds']), 'Repeated seed within arm')
        seeds.update(arm['seeds'])
        requests.append((path, arm['name'], preparation, protocol, models, selected))
    require(arm_names == {a['name'] for a in protocol['arms']}, 'Missing frozen arm')
    if workers == 1:
        for request in requests:
            record = audit_campaign(*request)
            write(out/f'{record["arm"]}.json', record); records.append(record)
    else:
        # Independent arms read disjoint population files. Parent alone writes
        # derived outputs, and no source row is audited by both workers.
        with ProcessPoolExecutor(max_workers=min(workers, len(requests))) as executor:
            futures = [executor.submit(audit_campaign, *request) for request in requests]
            for future in as_completed(futures):
                record = future.result()
                write(out/f'{record["arm"]}.json', record); records.append(record)
    order = {arm['name']: i for i, arm in enumerate(protocol['arms'])}
    records.sort(key=lambda record: order[record['arm']])
    for path, digest in input_hashes.items(): require(sha(Path(path)) == digest, 'Input changed during audit')
    result = dict(complete=True, campaigns=records, original_q_window=WINDOW, prefix_counts=protocol['prefix_counts'],
        repeat_source_verification=repeat_source,
        independent_arm_audit_workers=min(workers, len(requests)),
        reference_comparisons=reference_comparisons(records, protocol), geometry_reconstructions=geometries,
        center_separation=separation, input_sha256=input_hashes,
        archived_sha256={p.name: sha(p) for p in archive.iterdir()}, elapsed_seconds=time.monotonic()-started,
        estimator='Every fixed-mask estimate divides by the frozen complete proposal density and full unconditional N. Invalid and off-mask draws remain zero. Independent arms are not pooled.',
        coverage='Old radius-2 ball, new radius-2 ball and outside-both directly partition the complete original intermediate region. Observed errors and zero hits do not bound unseen weight.',
        calibration='Historical references and contact centers informed proposal preparation. New production uses fixed models and independent streams; reference comparisons are calibration, not untouched holdout validation.',
        variance='Same-row disjoint-mask covariance is included; nested sample prefixes are correlated and are never counted as independent confirmations.',
        scope='Conditional fixed-AB contact statistics. Does not include native/shoulder/distant domains, neighbor assembly costs, MCMC mixing or a crystallization verdict.')
    write(out/'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1, help='Independent arm audit processes; does not change physical sampling')
    args = parser.parse_args(); analyze(args.preparation, args.campaign, args.out, args.workers)
