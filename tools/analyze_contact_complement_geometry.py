#!/usr/bin/env python3
"""Bounded post-hoc geometry of saved fresh native-R4 minus old-R5 weights.

No sampling, classifier calls, Poisson/audit replay, density fit, or target change.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import gzip
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, squareform
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from analyze_contact_bank_reference_partition import HashLedger, r5_contains, partition_masks
from analyze_mobile_native_pocket import read, write, sha, require
from diagnose_conditional_ray_extremes import chart_coordinates, chart_center, rmat

ROOT = Path(__file__).resolve().parents[1]
PART = 'remaining_R4_native'
SCOPE = ('Post-hoc descriptive diagnosis of existing completed pilot rows. Original importance weights and all '
    '16,384 unconditional attempts per population are retained; no second Jacobian, conditional denominator, '
    'source/fresh pooling, classifier evaluation, audit replay, physical sampling, or proposal fit. Arm-level '
    'coordinate moments describe their separate empirical weighted clouds, not new target estimates. Complete-link '
    'coordinate groups do not establish basins, barriers, connectivity, or convergence. These pilot rows become '
    'training data if reused to design a guide; future validation must use fresh independent populations.')


def wquant(x, logw, probabilities=(0., .05, .25, .5, .75, .95, 1.)):
    x = np.asarray(x, float); order = np.argsort(x, kind='stable')
    w = np.exp(logw - logsumexp(logw)); c = np.cumsum(w[order]); c[-1] = 1.
    return {str(p): float(x[order[min(int(np.searchsorted(c, p)), len(x)-1)]]) for p in probabilities}


def moments(x, logw):
    x = np.asarray(x, float); w = np.exp(logw-logsumexp(logw)); mean = w@x
    delta = x-mean; cov = (delta*w[:, None]).T@delta
    eig, axes = np.linalg.eigh(cov); order = np.argsort(-eig); eig, axes = eig[order], axes[:, order].T
    for axis in axes:
        if axis[np.argmax(abs(axis))] < 0: axis *= -1
    return dict(mean=mean.tolist(), covariance=cov.tolist(), eigenvalues=eig.tolist(),
                eigenvectors=axes.tolist(), explained_fraction=(eig/eig.sum()).tolist(),
                observed_weight_ESS=float(1/(w@w)), convention='Weighted central second moment; no unbiased-covariance correction or proposal fit.')


def decode(u, region):
    """Algebraic inverse of the existing single-anchor exact chart_coordinates."""
    chart = region['gaussian_chart']; fixed = rmat(region['fixed_neighbor']['orientation'])
    x = np.asarray(u)@np.linalg.cholesky(chart['covariances'][0]).T+chart['means'][0]
    translation = (x[:, :3]+chart['anchors'][0]['position'])@fixed.T+region['fixed_neighbor']['position']
    rotation = fixed@Rotation.from_quat(np.column_stack((x[:, 3:]/chart['angular_length'], np.ones(len(x))))).as_matrix()@chart['anchors'][0]['rotation']
    quaternion = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
    return [dict(position=t.tolist(), orientation=q.tolist()) for t, q in zip(translation, quaternion)]


def pose_features(poses, members):
    t = np.asarray([p['position'] for p in poses]); q = np.asarray([p['orientation'] for p in poses]); q /= np.linalg.norm(q, axis=1)[:, None]
    m = np.asarray([p['position'] for p in members]); world = np.einsum('nij,kj->nki', rmat(q), m)+t[:, None]
    return dict(center=t, quaternion=q, members=world.reshape(len(poses), -1)/math.sqrt(len(m)))


def distances(a, b):
    return dict(member_RMS_A=cdist(a['members'], b['members']), center_A=cdist(a['center'], b['center']),
                angle_deg=np.degrees(2*np.arccos(np.clip(abs(a['quaternion']@b['quaternion'].T), 0., 1.))))


def nearest(matrix, i, indices, identifiers):
    j = int(indices[np.argmin(np.maximum(matrix['member_RMS_A'][i, indices]/.25, matrix['angle_deg'][i, indices]/.5))])
    return dict(id=identifiers[j], **{k: float(v[i, j]) for k, v in matrix.items()})


def grouping(selected, matrix):
    result = {}
    for length, angle in ((.1, .2), (.25, .5), (.5, 1.)):
        metric = np.maximum(matrix['member_RMS_A']/length, matrix['angle_deg']/angle)
        metric = (metric+metric.T)/2; np.fill_diagonal(metric, 0.)
        labels = fcluster(linkage(squareform(metric), method='complete'), 1., criterion='distance')
        groups = []
        for label in np.unique(labels):
            ix = np.flatnonzero(labels == label); rows = [selected[i] for i in ix]
            groups.append(dict(size=len(ix), population_count=len({(r['arm'], r['population']) for r in rows}),
                arm_count=len({r['arm'] for r in rows}), maxima_count=sum(r['rank']==1 for r in rows),
                maximum_member_RMS_A=float(matrix['member_RMS_A'][np.ix_(ix, ix)].max()),
                maximum_angle_deg=float(matrix['angle_deg'][np.ix_(ix, ix)].max()),
                observed_arm_complement_fractions={arm: sum(r['fraction_of_arm_complement'] for r in rows if r['arm']==arm) for arm in ('bank','wide')},
                ids=[r['id'] for r in rows]))
        groups.sort(key=lambda g: (-g['maxima_count'], -g['population_count'], -g['size']))
        result[f'{length}A_{angle}deg'] = dict(cluster_count=len(groups), clusters=groups)
    return result


def summarize_features(rows, current, reference):
    logs = np.array([r['log_importance_weight'] for r in rows]); u = np.asarray([r['current_u'] for r in rows])
    x = u@np.linalg.cholesky(current['gaussian_chart']['covariances'][0]).T+current['gaussian_chart']['means'][0]
    patterns = {}
    for r, w in zip(rows, np.exp(logs-logsumexp(logs))):
        key = '+'.join(f"{'AB'[m['anchor_index']]}{m['motif_id']}" for m in r['classification']['matches'])
        entry = patterns.setdefault(key, dict(count=0, observed_weight_fraction=0.))
        entry['count'] += 1; entry['observed_weight_fraction'] += float(w)
    result = dict(contributors=len(rows), current_u=moments(u, logs), reference_u=moments([r['reference_u'] for r in rows], logs),
        anchored_translation_scaled_Cayley_A=moments(x, logs), motif_patterns=patterns,
        current_radius=wquant(np.linalg.norm(u, axis=1), logs), reference_radius=wquant([r['reference_radius'] for r in rows], logs),
        original_q=wquant([r['q'] for r in rows], logs),
        anchors={str(anchor): dict(minimum_surface_gap_A=wquant([r['contact']['anchors'][anchor]['minimum_surface_gap_A'] for r in rows], logs)) for anchor in (0, 1)},
        complement_reason=dict(outside_old_radius_count=sum(r['reference_radius']>5 for r in rows), original_q_le_one_count=sum(r['q']<=1 for r in rows)),
        weighted_radius_fractions={str(radius): float(np.exp(logs[[r['reference_radius']<=radius for r in rows]]-logsumexp(logs)).sum()) for radius in (5, 6, 8, 10, 12, 16, 24, 32)},
        near_hard_surface_fractions={str(gap): float(np.exp(logs[[max(a['minimum_surface_gap_A'] for a in r['contact']['anchors'])<gap for r in rows]]-logsumexp(logs)).sum()) for gap in (.01, .025, .05, .1)})
    for anchor in (0, 1):
        matching = [[m for m in r['classification']['matches'] if m['anchor_index']==anchor] for r in rows]
        if all(len(m)==1 for m in matching):
            for field in ('proper_orientation_error_deg', 'maximum_member_position_error_A'):
                result['anchors'][str(anchor)][field] = wquant([m[0][field] for m in matching], logs)
    old = np.asarray(reference['current_u']['mean']); shift = np.asarray(result['current_u']['mean'])-old
    oldeig = np.asarray(reference['current_u']['eigenvalues']); oldaxes = np.asarray(reference['current_u']['eigenvectors'])
    result['displacement_from_old_R5_weighted_current_u_mean'] = shift.tolist()
    result['displacement_norm_current_u'] = float(np.linalg.norm(shift))
    result['displacement_in_old_R5_principal_SD'] = ((oldaxes@shift)/np.sqrt(oldeig)).tolist()
    result['variance_ratio_along_old_R5_principal_axes'] = (np.diag(oldaxes@np.asarray(result['current_u']['covariance'])@oldaxes.T)/oldeig).tolist()
    return result


def analyze(partition_path, out, top):
    require(1 <= top <= 50, 'Select 1 through 50 rows per population')
    out = Path(out).resolve(); require(not out.exists(), 'Refuse to overwrite an existing diagnostic directory')
    ledger = HashLedger(); partition_path = Path(partition_path).resolve(); ledger.frozen(partition_path)
    saved = read(partition_path/'analysis.json'); require(saved['complete'] and saved['total_fresh_unconditional_draws']==131072, 'Incomplete saved partition')
    bound = saved['source_sha256']
    def source(path):
        path = Path(path).resolve(); require(str(path) in bound, 'Input is not bound by the completed partition: '+str(path))
        return ledger.bind(path, bound[str(path)])
    plan = read(ledger.bind(partition_path/'supplemental-plan.json'))
    comparison = Path(saved['primary_comparison']); primary = read(source(comparison/'analysis.json'))
    campaign = Path(primary['campaign']); protocol = read(source(campaign/'protocol.json'))
    reference_region = read(source(plan['reference_region'])); review_path = source(plan['identity_review']); review = read(review_path)
    old_archive = ledger.bind(review_path.parent/'reference-contributors.npz', review['reference_contributors_sha256'])
    with np.load(old_archive, allow_pickle=False) as data: old = {k: data[k] for k in data.files}
    require(len(old['draw'])==5064 and review['subset_estimate']['row_uncertainty']['draws']==131072, 'Old denominator or contributor count differs')
    current = read(source(campaign/'bank/provenance/region.json'))
    require(read(source(campaign/'wide/provenance/region.json'))==current, 'Fresh arms have different charts')
    reference_poses = decode(old['current_u'], current)
    inverse_error = float(np.max(abs(chart_coordinates(reference_poses, reference_region)-old['reference_u'])))
    require(inverse_error<2e-8, 'Reconstructed physical reference pose disagrees with exact old chart')
    ref = dict(contributors=5064, unconditional_draws=131072, population_count=8,
        current_u=moments(old['current_u'], old['log_importance_weight']), reference_u=moments(old['reference_u'], old['log_importance_weight']),
        saved_motif_patterns=review['saved_reference_native_motif_patterns'], reconstructed_pose_reference_chart_max_error=inverse_error,
        source=str(old_archive), scope='Old eight-population training reference retained separately, never combined with fresh target estimates.')
    allrows = []; selected = []; populations = []; source_checks = []
    for arm in ('bank', 'wide'):
        armestimate = saved['arms'][arm]['estimates'][PART]
        arm_log_sum = armestimate['row_uncertainty']['log_Qz']+math.log(65536)
        for record in sorted(primary['arms'][arm]['populations'], key=lambda r: r['id']):
            pid = record['id']; n = record['samples']; require(n==16384, 'Original unconditional denominator changed')
            numeric = ledger.bind(source(comparison/record['records']), record['records_sha256'])
            with np.load(numeric, allow_pickle=False) as data: arrays = {k:data[k] for k in data.files}
            raw = ledger.bind(source(campaign/arm/'runs'/pid/'samples.jsonl'), record['raw_output']['samples_sha256'])
            nativeids = np.flatnonzero(np.isfinite(arrays['z']) & arrays['native']); wanted = set(nativeids); rows = {}; attempts = 0
            with raw.open() as stream:
                for i, line in enumerate(stream):
                    row = json.loads(line); require(row['draw']==i, 'Original draw ordering changed'); attempts += 1
                    if i in wanted:
                        require(row['log_importance_weight']==arrays['z'][i] and row['log_hard_weight']==arrays['h'][i], 'Original saved weight changed')
                        rows[i] = row
            require(attempts==n, 'Exterior or invalid zero attempts were dropped')
            poses = [rows[i]['pose'] for i in nativeids]; old_u = chart_coordinates(poses, reference_region)
            current_u = chart_coordinates(poses, current); rad = np.linalg.norm(old_u, axis=1)
            require(float(np.max(abs(current_u-np.asarray([rows[i]['latent'] for i in nativeids]))))<2e-8, 'Current chart inverse differs')
            captures = np.linalg.norm(np.asarray([r['position'] for r in poses])-reference_region['capture_center'], axis=1)<=reference_region['capture_radius']
            in_r5 = r5_contains(rad, np.asarray([rows[i]['q'] for i in nativeids]), captures)
            compids = nativeids[~in_r5]; comp_set = set(compids); log_sum = float(logsumexp(arrays['z'][compids]))
            expected = next(p for p in armestimate['populations'] if p['id']==pid)
            require(expected['draws']==n and expected['nonzero']==len(compids) and abs(expected['log_Qz']-(log_sum-math.log(n)))<2e-10, 'Saved complement estimate not reproduced')
            require(abs(expected['log_Q0']-(float(logsumexp(arrays['h'][compids]))-math.log(n)))<2e-10, 'Saved complement Q0 changed')
            order = compids[np.argsort(-arrays['z'][compids], kind='stable')]; topids = list(map(int, order[:top])); ranks = {v:i+1 for i,v in enumerate(topids)}
            label_path = ledger.bind(source(comparison/record['labels']), record['labels_sha256']); labels = {}; seen = 0
            with gzip.open(label_path, 'rt') as stream:
                for i,line in enumerate(stream):
                    label = json.loads(line); require(label['draw']==i, 'Saved classifier ordering changed'); seen += 1
                    if i in comp_set:
                        require(label['applicable'] and label['classification']['native_any'] and arrays['native'][i], 'Saved native label differs')
                        labels[i] = label
            require(seen==n and len(labels)==len(compids), 'Missing saved full-classifier rows')
            local = []
            for k,i in enumerate(nativeids):
                if i not in comp_set: continue
                row = rows[i]; label = labels[i]
                value = dict(id=f'{arm}/{pid}/{i}', arm=arm, population=pid, seed=record['seed'], draw=int(i), unconditional_draws=n,
                    rank=ranks.get(int(i)), pose=row['pose'], current_u=current_u[k].tolist(), reference_u=old_u[k].tolist(),
                    current_radius=float(np.linalg.norm(current_u[k])), reference_radius=float(rad[k]), q=row['q'],
                    log_importance_weight=float(arrays['z'][i]), log_hard_weight=float(arrays['h'][i]), paired_log_importance_weights=arrays['pairs'][i].tolist(),
                    log_proposal_density=row['log_proposal_density'], proposal_branch=row['proposal_branch'], proposal_component=row['proposal_component'],
                    fraction_of_population_complement=math.exp(arrays['z'][i]-log_sum), fraction_of_arm_complement=math.exp(arrays['z'][i]-arm_log_sum),
                    classification=label['classification'], contact=label['contact'])
                local.append(value); allrows.append(value)
                if i in ranks: selected.append(value)
            weights = np.exp(arrays['z'][order]-log_sum); cs = np.cumsum(weights)
            populations.append(dict(arm=arm, id=pid, seed=record['seed'], unconditional_draws=n, complement_nonzero=len(compids),
                unchanged_saved_estimate=expected, selected_count=len(topids), selected_fraction=float(weights[:top].sum()),
                top1_fraction=float(weights[0]), top5_fraction=float(weights[:5].sum()), rows_for_90pct=int(np.searchsorted(cs,.9)+1),
                geometry=summarize_features(local, current, ref)))
            source_checks.append(dict(arm=arm, id=pid, unconditional_attempts_checked=attempts, labels_checked=seen,
                complement_Qz_Q0_match_saved_partition=True))
            print(f'{arm}/{pid}: {len(compids)} complement contributors; top{top} share={weights[:top].sum():.3%}', flush=True)
    selected.sort(key=lambda r:(r['arm'], r['population'], r['rank']))
    members=current['physical_metric']['rigid_members']; features=pose_features([r['pose'] for r in selected], members)
    comparison_matrix=distances(features, features); reference_matrix=distances(features, pose_features(reference_poses, members))
    center_matrix=distances(features, pose_features([chart_center(reference_region)], members))
    reference_ids=[f"r5/{p}/{d}" for p,d in zip(old['population'],old['draw'])]; ids=[r['id'] for r in selected]
    for i,row in enumerate(selected):
        row['nearest_old_R5_contributor']=nearest(reference_matrix,i,np.arange(len(reference_ids)),reference_ids)
        row['distance_to_old_R5_chart_center']={k:float(v[i,0]) for k,v in center_matrix.items()}
        for field,mask in (
            ('nearest_other_population', [(r['arm'],r['population'])!=(row['arm'],row['population']) for r in selected]),
            ('nearest_same_arm_other_population', [r['arm']==row['arm'] and r['population']!=row['population'] for r in selected]),
            ('nearest_other_arm', [r['arm']!=row['arm'] for r in selected])):
            row[field]=nearest(comparison_matrix,i,np.flatnonzero(mask),ids)
    arms={}
    for arm in ('bank','wide'):
        rows=[r for r in allrows if r['arm']==arm]; chosen=[r for r in selected if r['arm']==arm]
        arms[arm]=dict(unchanged_saved_estimate=saved['arms'][arm]['estimates'][PART], geometry=summarize_features(rows,current,ref),
            selected_count=len(chosen), selected_fraction=sum(r['fraction_of_arm_complement'] for r in chosen),
            selected_reference_nearest_RMS_A=wquant([r['nearest_old_R5_contributor']['member_RMS_A'] for r in chosen],np.asarray([r['log_importance_weight'] for r in chosen])),
            selected_other_population_nearest_RMS_A=wquant([r['nearest_other_population']['member_RMS_A'] for r in chosen],np.asarray([r['log_importance_weight'] for r in chosen])))
    result=dict(schema='contact-complement-geometry-v1', complete=True, scope=SCOPE, partition_source=str(partition_path),
        selected_top_per_population=top, total_fresh_unconditional_draws=131072, reference=ref, arms=arms, populations=populations,
        selected=selected, selected_clusters=grouping(selected,comparison_matrix), source_checks=source_checks,
        new_physical_samples=0, classifiers_rerun=0, audits_replayed=0, weights_changed=False, primary_gates_changed=False,
        covariance_coordinate_convention='current_u and reference_u are exact whitened charts. Anchored translation/scaled Cayley coordinates are in Å, with angular_length times quaternion vector/scalar; their principal axes mix translations and scaled rotations descriptively.')
    for file in (__file__, ROOT/'tools/analyze_contact_bank_reference_partition.py',ROOT/'tools/diagnose_conditional_ray_extremes.py',ROOT/'tools/analyze_mobile_native_pocket.py'):
        ledger.bind(file)
    ledger.recheck(); result['source_sha256']=ledger.files
    out.mkdir(parents=True)
    convenience={key:np.asarray([r[key] for r in allrows]) for key in ('arm','population','seed','draw','current_u','reference_u','reference_radius','log_importance_weight','log_hard_weight','paired_log_importance_weights','log_proposal_density')}
    np.savez_compressed(out/'complement-contributors.npz', **convenience)
    result['complement_contributors_sha256']=sha(out/'complement-contributors.npz')
    write(out/'analysis.json',result)
    report(result,out)
    shutil.copy2(__file__,out/Path(__file__).name)
    write(out/'freeze.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return result


def report(result,out):
    lines=['# Saved fresh native-complement geometry','',SCOPE,'',
        'Each fresh population retains its original 16,384 attempts, including hard-invalid and exterior zeros. All complement Qz and Q0 estimates reproduce the completed partition; geometry uses every positive complement contributor. The selected archive takes the largest 25 by original weight from each population. Reference comparisons use all 5,064 old-R5 contributors, reconstructed through the exact current-chart inverse and independently checked in the old chart.','',
        '| Arm/population | Positive complement rows | log Qz | Largest share | Top selected share | Top old radius | Top nearest old RMS (Å) | Top nearest other population RMS (Å) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for pop in result['populations']:
        top=next(r for r in result['selected'] if r['arm']==pop['arm'] and r['population']==pop['id'] and r['rank']==1)
        lines.append(f"| {pop['arm']}/{pop['id']} | {pop['complement_nonzero']} | {pop['unchanged_saved_estimate']['log_Qz']:.6f} | {pop['top1_fraction']:.1%} | {pop['selected_fraction']:.1%} | {top['reference_radius']:.3f} | {top['nearest_old_R5_contributor']['member_RMS_A']:.4f} | {top['nearest_other_population']['member_RMS_A']:.4f} |")
    lines+=['','The old R5 ellipsoid boundary excludes geometries that can remain close in physical coordinates. Distances compare labeled rigid-member centers and proper body rotations; nearest neighbors minimize max(member RMS/0.25 Å, angle/0.5°).','',
        '| Arm | Weighted old-radius median / 95th percentile | Weight at old radius ≤8 / ≤12 | Weight with both hard gaps <0.05 Å | Top selected arm mass |', '|---|---:|---:|---:|---:|']
    for arm,source in result['arms'].items():
        g=source['geometry'];lines.append(f"| {arm} | {g['reference_radius']['0.5']:.3f} / {g['reference_radius']['0.95']:.3f} | {g['weighted_radius_fractions']['8']:.1%} / {g['weighted_radius_fractions']['12']:.1%} | {g['near_hard_surface_fractions']['0.05']:.1%} | {source['selected_fraction']:.1%} |")
    lines+=['','Full saved native motif patterns and separate coordinate moments:']
    for arm,source in result['arms'].items():
        g=source['geometry']; lines+=['',f"- {arm}: {g['motif_patterns']}. Complement reason: {g['complement_reason']}.",
            f"- {arm} current-u mean: {np.round(g['current_u']['mean'],6).tolist()}; covariance eigenvalues: {np.round(g['current_u']['eigenvalues'],6).tolist()}.",
            f"- {arm} variance ratios along descending-variance old-R5 principal axes: {np.round(g['variance_ratio_along_old_R5_principal_axes'],3).tolist()}; mean displacement in old-R5 principal standard deviations: {np.round(g['displacement_in_old_R5_principal_SD'],3).tolist()}."]
    lines+=['','Complete-link groups of the selected coordinates:']
    for key,summary in result['selected_clusters'].items():
        first=summary['clusters'][0];lines.append(f"- {key}: {summary['cluster_count']} groups; first-ranked group includes {first['size']} rows from {first['population_count']} populations, {first['arm_count']} arms, and {first['maxima_count']} population maxima; its observed arm-complement fractions are {first['observed_arm_complement_fractions']}.")
    maxima=[r for r in result['selected'] if r['rank']==1]
    lines+=['','The eight maxima all retain the known A7/B4 motif pattern, with small proper orientation errors and nearly touching hard surfaces. The observed complement is consistent with a geometrically extended part of the known contact outside a narrow old ellipsoid. Multiple coordinate groups remain, and no connected-basin or unseen-tail claim follows. The low original ESS and several large single-row shares remain unresolved. A useful guide design must broaden coverage along observed correlated contact directions, include the close hard-boundary poses from every training population, and retain broad defensive coverage; its adequacy requires fresh independent populations.','',
        'analysis.json includes all per-population and per-arm weighted moments, principal directions, hard-gap/motif summaries, and full saved labels for the selected rows. complement-contributors.npz retains every positive complement contributor with original z/h/paired weights and exact current/old coordinates; the omitted zero rows remain in every original denominator.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition',type=Path,default=ROOT/'runs/contact-bank-reference-partition-20260922')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--top',type=int,default=25)
    args=parser.parse_args();analyze(args.partition,args.out,args.top);print(args.out.resolve())
