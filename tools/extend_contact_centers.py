#!/usr/bin/env python3
"""Preserve the old contact atlas and cover additional audited poses deterministically."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pose_sha(pose):
    return hashlib.sha256(json.dumps(pose, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def extend_indices(matrix, initial, mandatory, radius=.5):
    """Old order is immutable; append mandatory, then first-index farthest ties."""
    n = len(matrix)
    require(matrix.shape == (n, n) and np.isfinite(matrix).all() and np.min(matrix) >= 0,
            'Invalid distance matrix')
    require(np.array_equal(matrix, matrix.T) and np.array_equal(np.diag(matrix), np.zeros(n)),
            'Distance matrix must be symmetric with zero diagonal')
    selected = list(initial)
    require(selected and len(set(selected)) == len(selected)
            and all(0 <= i < n for i in selected) and 0 <= mandatory < n,
            'Invalid initial or mandatory indices')
    require(np.isfinite(radius) and radius > 0, 'Positive cover radius required')
    progression = []
    if mandatory not in selected:
        selected.append(mandatory)
    nearest = np.min(matrix[:, selected], axis=1)
    progression.append(dict(reason='preserve old centers and append mandatory maximum',
                            center_index=mandatory, selected_count=len(selected),
                            maximum_uncovered_RMS_A=float(nearest.max())))
    while nearest.max() > radius:
        index = int(np.argmax(nearest))
        require(index not in selected, 'Farthest-first made no progress')
        selected.append(index); nearest = np.minimum(nearest, matrix[:, index])
        progression.append(dict(reason='farthest-first', center_index=index, selected_count=len(selected),
                                maximum_uncovered_RMS_A=float(nearest.max())))
    return selected, progression


def extend(root, out):
    require(not out.exists(), 'Use a fresh output directory')
    paths = dict(old=root/'runs/ab-intermediate-contact-centers-20260920/selection.json',
                 atlas=root/'runs/ab-intermediate-contact-atlas-audit-20260920/analysis.json',
                 peak=root/'runs/ab-intermediate-atlas-peak-reference-extremes-20260920/analysis.json')
    sources = {name:read(path) for name, path in paths.items()}
    verified = {}

    def verify(path, expected):
        key = str(path)
        if key not in verified:
            verified[key] = sha(path)
        require(verified[key] == expected, 'Changed source: '+key)

    for name, source in sources.items():
        require(source['complete'], 'Incomplete source: '+name)
        for filename, digest in source['archived_sha256'].items():
            verify(paths[name].parent/'provenance'/filename, digest)
    old, atlas, peak = (sources[key] for key in ('old', 'atlas', 'peak'))
    require(old['no_physical_draws'] and len(old['selected_centers']) == 36 and len(old['candidates']) == 64,
            'Unexpected old atlas')
    for name, digest in old['source_sha256'].items():
        verify(Path(old['source_paths'][name]), digest)
    for path, digest in atlas['input_sha256'].items():
        verify(Path(path), digest)
    local_path = paths['peak'].parent/'provenance/local-analysis.json'
    verify(local_path, peak['source_analysis_sha256'])
    local = read(local_path)
    require(local['complete'], 'Incomplete local reference analysis')
    for path, digest in local['input_sha256'].items():
        verify(Path(path), digest)
    cfg_path = paths['old'].parent/'provenance/config.json'; cfg = read(cfg_path)
    require(cfg['reservoir_density'] == .035 and cfg['depletant_radius'] == 1.5
            and cfg['capture_radius'] == 18. and len(cfg['fixed_poses']) == 2, 'Changed baseline target')
    shape_hash = sha(Path(cfg['shape']))
    physical = ('metadata', 'fixed_poses', 'capture_center', 'capture_radius', 'reservoir_density', 'depletant_radius')
    candidates = copy.deepcopy(old['candidates'])
    for candidate in candidates:
        candidate['pose_sha256'] = pose_sha(candidate['pose'])
        candidate['source_audit_sha256'] = sha(paths['old'])

    def add_campaign(campaign, group, sample_hashes, manifest_hashes, latent):
        campaign_root = Path(campaign['root'])
        root_manifest = None
        if latent:
            verify(campaign_root/'manifest.json', manifest_hashes)
            root_manifest = read(campaign_root/'manifest.json')
            for name, digest in root_manifest['archive_sha256'].items():
                verify(campaign_root/'provenance'/name, digest)
        campaign_cfg = read(campaign_root/'provenance/config.json')
        for key in physical:
            require(campaign_cfg[key] == cfg[key], 'Different physical target: '+key)
        verify(campaign_root/'provenance/shape.json', shape_hash)
        require(len(campaign['top_poses']) == 8, 'Expected eight extrema per campaign')
        grouped = {}
        for point in campaign['top_poses']:
            require(min(point['minimum_atomic_gap_by_neighbor_A']) >= 0, 'Unaudited hard overlap')
            grouped.setdefault(point['population'], []).append(point)
        for population, points in grouped.items():
            samples = campaign_root/'runs'/population/'samples.jsonl'
            manifest_path = samples.parent/'manifest.json'
            verify(samples, sample_hashes[str(samples)])
            if not latent:
                verify(manifest_path, manifest_hashes[str(manifest_path)])
            else:
                verified[str(manifest_path)] = sha(manifest_path)
            manifest = read(manifest_path)
            if latent:
                job = next(job for job in root_manifest['jobs'] if job['id'] == population)
                require(manifest['seed'] == job['seed'] and manifest['samples'] == job['samples'],
                        'Changed local population identity')
                require(manifest['region_sha256'] == root_manifest['region_sha256']
                        and manifest['config_sha256'] == root_manifest['archive_sha256']['config.json'],
                        'Changed local population physical inputs')
            require(manifest['shape_sha256'] == shape_hash, 'Different manifest shape')
            window = manifest.get('q_window')
            if window is not None:
                require(window == dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False),
                        'Changed q window')
            if latent:
                region = read(campaign_root/'provenance/region.json')
                require((region['minimum_original_q'], region['maximum_original_q'],
                         region['minimum_original_q_inclusive'], region['maximum_original_q_inclusive']) == (2., 5., True, False),
                        'Changed latent q window')
                require(region['physical_metric'] == cfg['metadata'] and region['physical_fixed_neighbors'] == cfg['fixed_poses'],
                        'Changed latent physical geometry')
            wanted = {p['draw']:p for p in points}; found = set()
            with samples.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['draw'] not in wanted:
                        continue
                    point = wanted[row['draw']]
                    require(row['pose'] == point['pose'] and manifest['seed'] == point['seed'], 'Extremum pose/seed mismatch')
                    require(row['log_importance_weight'] == point['original_log_importance_weight'], 'Extremum weight mismatch')
                    found.add(row['draw'])
                    if len(found) == len(wanted):
                        break
            require(found == set(wanted), 'Missing source extremum')
        for point in campaign['top_poses']:
            candidates.append(dict(source_index=len(candidates), id=f"{group}:{point['population']}:{point['draw']}",
                                   source_group=group, source_root=str(campaign_root), population=point['population'],
                                   seed=point['seed'], draw=point['draw'], pose=point['pose'],
                                   pose_sha256=pose_sha(point['pose']), source_audit_sha256=sha(paths['peak' if latent else 'atlas']),
                                   source_log_boltzmann_mean=point['log_boltzmann_mean']))

    for campaign in atlas['campaigns']:
        add_campaign(campaign, 'atlas-'+campaign['arm'], campaign['input_sha256'], campaign['input_sha256'], False)
    for campaign in peak['campaigns']:
        original = next(p for p in local['campaigns'] if p['root'] == campaign['root'])['original_reference_audit']
        require(original['sample_sha256'] == campaign['sample_sha256'], 'Changed local source hashes')
        add_campaign(campaign, f"atlas-peak-{campaign['radius_A']:g}", campaign['sample_sha256'], original['manifest_sha256'], True)
    require(len(candidates) == 104 and len({p['id'] for p in candidates}) == 104, 'Unexpected candidate identities')
    broad = next(c for c in atlas['campaigns'] if c['arm'] == 'broad')['top_poses'][0]
    mandatory_id = f"atlas-broad:{broad['population']}:{broad['draw']}"
    mandatory = next(i for i, p in enumerate(candidates) if p['id'] == mandatory_id)
    require(broad['draw'] == 14669 and broad['seed'] == 101502019, 'Unexpected mandatory maximum')
    # Reuse the archived selector's normalized-quaternion/member transformation helper.
    selector_path = paths['old'].parent/'provenance/select_centers.py'
    spec = importlib.util.spec_from_file_location('archived_contact_selector', selector_path)
    selector = importlib.util.module_from_spec(spec); spec.loader.exec_module(selector)
    members, _ = selector.pose_arrays(cfg['metadata']['rigid_members'])
    require(np.array_equal(members.mean(axis=0), np.zeros(3)), 'Members must remain centered')
    translations, rotations = selector.pose_arrays([p['pose'] for p in candidates])
    world = np.einsum('nij,mj->nmi', rotations, members)+translations[:, None, :]
    matrix = np.sqrt(np.mean(np.sum((world[:, None]-world[None, :])**2, axis=3), axis=2))
    require(np.allclose(matrix[:64, :64], old['RMS_distance_matrix_A'], rtol=0., atol=1e-12), 'Old member geometry changed')
    initial = old['selected_indices']
    for center, index in zip(old['selected_centers'], initial):
        require(center['pose'] == candidates[index]['pose'] and center['id'] == candidates[index]['id'], 'Old center changed')
    selected, progression = extend_indices(matrix, initial, mandatory)
    require(selected[:36] == initial and selected[36] == mandatory, 'Old order or mandatory first append changed')
    moment = members.T @ members/len(members); a = np.trace(moment)*np.eye(3)-moment
    chart = np.zeros((len(candidates), len(selected)))
    for slot, center in enumerate(selected):
        relative = rotations @ rotations[center].T
        quaternion = Rotation.from_matrix(relative).as_quat()
        require(np.min(np.abs(quaternion[:, 3])) > 1e-12, 'Observed Cayley singularity')
        c = quaternion[:, :3]/quaternion[:, 3, None]
        quadratic = 4*np.einsum('ni,ij,nj->n', c, rotations[center] @ a @ rotations[center].T, c)
        translation2 = np.sum((translations-translations[center])**2, axis=1)
        chart[:, slot] = np.sqrt(translation2+quadratic)
        require(np.max(np.abs(translation2+quadratic/(1+np.sum(c*c, axis=1))-matrix[:, center]**2)) < 1e-10,
                'Cayley/member RMS identity failed')
    nearest_slots = np.argmin(matrix[:, selected], axis=1)
    mappings = [dict(candidate_index=i, candidate_id=p['id'], nearest_center_slot=int(nearest_slots[i]),
                     nearest_RMS_A=float(matrix[i, selected[nearest_slots[i]]]),
                     nearest_chart_radius_A=float(chart[i].min())) for i, p in enumerate(candidates)]
    sources_to_archive = {'old-selection.json':paths['old'], 'atlas-audit.json':paths['atlas'],
                          'peak-extremes.json':paths['peak'], 'local-analysis.json':local_path,
                          'config.json':cfg_path, 'old-selector.py':selector_path,
                          Path(__file__).name:Path(__file__)}
    archive = out/'provenance'; archive.mkdir(parents=True)
    for name, path in sources_to_archive.items():
        shutil.copy2(path, archive/name)
    result = dict(schema='extended-finite-contact-center-cover-v1', complete=True, no_physical_draws=True,
                  candidates=candidates, selected_indices=selected,
                  selected_centers=[dict(candidates[index], component_index=slot, proposed_weight=1/len(selected))
                                    for slot, index in enumerate(selected)],
                  preserved_center_count=36, mandatory_center_id=mandatory_id, mandatory_center_index=mandatory,
                  candidate_order='Old 64 in original order; atlas narrow then broad audit order (8 each); new peak radius .5,1,2 audit order (8 each).',
                  algorithm='Preserve all old 36 center indices/order. Append selected broad maximum explicitly, even if already within the cover radius. Then append first candidate attaining largest nearest-center RMS until <= .5 A.',
                  coverage_radius_A=.5, RMS_distance_matrix_A=matrix.tolist(), chart_radius_matrix_A=chart.tolist(),
                  nearest_center_mappings=mappings, selection_progression=progression,
                  coverage=dict(candidate_count=len(candidates), center_count=len(selected),
                                added_center_count=len(selected)-36,
                                maximum_nearest_RMS_A=max(p['nearest_RMS_A'] for p in mappings),
                                maximum_nearest_chart_radius_A=float(chart.min(axis=1).max())),
                  physical={key:cfg[key] for key in physical}, shape_sha256=shape_hash,
                  original_q_window=dict(minimum=2., maximum=5., lower_inclusive=True, upper_inclusive=False),
                  geometry=dict(member_moment_A2=moment.tolist(), rotational_metric_A2=a.tolist()),
                  source_paths={key:str(path) for key, path in paths.items()},
                  source_sha256={key:sha(path) for key, path in paths.items()}, verified_source_sha256=verified,
                  input_sha256={**verified, **{str(path):sha(path) for path in sources_to_archive.values()}},
                  archived_sha256={name:sha(archive/name) for name in sources_to_archive},
                  inference='Only a cover of 104 observed historical poses. These are training data, not independent validation or a cover of physical basins. Equal center weights are proposal allocation, not equilibrium occupancy. Preserve a complete-support defensive component and use fresh independent physical production.')
    (out/'selection.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); result = extend(args.root.resolve(), args.out.resolve())
    print(json.dumps(result['coverage'], indent=2))
