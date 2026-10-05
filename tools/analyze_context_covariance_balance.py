"""Saved-pose deterministic-mixture hard-region weights; no geometry or sampling."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from source_guide_reference import SourceDensity, rotation

ARMS = ('full', 'diagonal')
BINS = (0., 6., 12., 24., 48., math.inf)
REGIONS = ('A_patch_0_0.25', 'A_patch_0.25_0.5', 'A_patch_0.5_0.75', 'A_patch_0.75_1',
           'A_patch_complete', 'B', 'other_contact', 'unbound', 'hard_invalid')


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def source_batch(density, poses):
    """Same normalized-Haar Cayley chart as the frozen independent reference."""
    positions = np.asarray([p['position'] for p in poses], float)
    quaternions = np.asarray([p['orientation'] for p in poses], float)
    require(positions.shape == (len(poses), 3) and quaternions.shape == (len(poses), 4)
            and np.isfinite(positions).all() and np.isfinite(quaternions).all()
            and np.max(np.abs(np.sum(quaternions**2, axis=1)-1)) < 2e-10, 'Invalid saved poses')
    anchor = rotation(density.anchor)
    relative_rotations = anchor.T @ Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
    delta = relative_rotations @ density.center_rotation.T
    q = Rotation.from_matrix(delta).as_quat()
    seam = q[:, 3] == 0
    logs = np.full(len(poses), -np.inf); squared = np.full(len(poses), np.inf)
    good = ~seam
    u = q[good, :3]/q[good, 3, None]
    t = (positions[good]-density.anchor['position'])@anchor-density.center['position']
    x = np.column_stack((t, density.ell*u))
    z = solve_triangular(density.lower, (x-density.mean).T, lower=True).T
    squared[good] = np.einsum('ij,ij->i', z, z)
    log_j = -3*math.log(density.ell)-2*math.log(math.pi)-2*np.log1p(np.einsum('ij,ij->i',u,u))
    logs[good] = -.5*squared[good]-3*math.log(2*math.pi)-density.logdet-log_j
    require(np.isfinite(logs[good]).all() and np.isfinite(squared[good]).all(), 'Unrepresentable source chart')
    return logs, squared, seam


def log_component(value, status=None):
    if status == 'negative_infinity' or (status is None and value is None):
        require(value is None, 'Inconsistent zero-density representation'); return -math.inf
    require(status in (None, 'finite') and type(value) in (int, float) and math.isfinite(value), 'Invalid component density')
    return float(value)


def balanced_log_q(log_u, log_g, log_full, log_diagonal):
    values = np.stack((log_u, log_g, log_full, log_diagonal))
    require(not np.isnan(values).any() and not np.isposinf(values).any(), 'Invalid mixture density')
    return logsumexp(values+np.log([.5, .25, .125, .125])[:, None], axis=0)


def close_logs(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(np.array_equal(np.isneginf(actual), np.isneginf(expected)), 'Density support differs')
    good = np.isfinite(actual)&np.isfinite(expected)
    errors = np.abs(actual[good]-expected[good])
    tolerance = 2e-8+2e-10*(1+np.abs(actual[good])+np.abs(expected[good]))
    require(np.all(errors <= tolerance), 'Saved density differs from independently reconstructed source law')
    return float(np.max(errors/tolerance)) if len(errors) else 0.


def importance_summary(log_weights, denominator):
    require(type(denominator) is int and denominator > 0, 'Every attempt must remain in denominator')
    values = np.asarray(log_weights, float)
    require(values.ndim == 1 and len(values) <= denominator and np.isfinite(values).all(), 'Invalid regional contributions')
    if not len(values):
        return dict(hits=0, denominator=denominator, mass=0., log_mass=None,
                    importance_ess=0., largest_fraction=None, log_weight_sum=None)
    total = float(logsumexp(values)); log_mass = total-math.log(denominator)
    mass = math.exp(log_mass)
    require(math.isfinite(mass), 'Unrepresentable hard-region mass')
    return dict(hits=len(values), denominator=denominator, mass=mass, log_mass=log_mass,
                importance_ess=float(math.exp(2*total-float(logsumexp(2*values)))),
                largest_fraction=float(math.exp(float(values.max())-total)), log_weight_sum=total)


def across_populations(rows):
    values = np.asarray([r['mass'] for r in rows], float)
    require(len(values) == 4 and np.isfinite(values).all(), 'Exactly four populations required')
    mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd/2
    return dict(populations=4, mean_mass=mean, population_standard_deviation=sd, standard_error=se,
                population_relative_standard_error=se/mean if mean else None,
                minimum_mass=float(values.min()), maximum_mass=float(values.max()),
                uncertainty='Sample standard deviation of four independent population masses divided by sqrt(4); not an IID-pooled-weight error estimate.')


def radial_bin(value):
    require(not math.isnan(value) and value >= 0, 'Invalid squared Mahalanobis radius')
    return min(int(np.searchsorted(BINS, value, side='right'))-1, 4)


def decomposition(records, total):
    """Two separately normalized radial partitions, never summed together."""
    output = []
    for chart in ARMS:
        rows = []
        for b in range(5):
            for arm in ARMS:
                for branch in ('uniform', 'context', 'source'):
                    chosen = [r['log_weight'] for r in records if r['arm'] == arm and r['branch'] == branch
                              and radial_bin(r['mahalanobis_squared'][chart]) == b]
                    s = importance_summary(chosen, total['denominator'])
                    rows.append(dict(bin_index=b, lower=BINS[b], upper=BINS[b+1] if b < 4 else None,
                        generating_arm=arm, generating_branch=branch, **s,
                        fraction_of_region_weight=math.exp(s['log_weight_sum']-total['log_weight_sum'])
                            if s['hits'] and total['hits'] else 0.))
        require(sum(r['hits'] for r in rows) == total['hits'], 'Radial partition lost a hit')
        if total['hits']:
            require(abs(sum(r['fraction_of_region_weight'] for r in rows)-1) < 2e-12, 'Radial masses do not close')
        output.append(dict(chart=chart, entries=rows, total_fraction=sum(r['fraction_of_region_weight'] for r in rows)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); protocol = read(args.protocol)
    require(protocol['schema'] == 'context-covariance-balance-reduction-v1'
            and protocol['arms'] == list(ARMS) and protocol['streams'] == list(range(4))
            and protocol['attempts_per_arm_population'] == 2048 and protocol['total_attempts'] == 16384
            and protocol['mahalanobis_squared_bins'] == [0, 6, 12, 24, 48, None], 'Changed fixed allocation')
    bindings = protocol['input_sha256']
    for path, digest in bindings.items(): require(sha(path) == digest, 'Changed input: '+path)
    def bound(path):
        require(str(Path(path).resolve()) in bindings, 'Unbound input: '+str(path)); return read(path)
    audit = bound(protocol['audit_report']); audited = audit['input_sha256']
    require(audit['complete'] and audit['passed'] and audit['decoded_candidates'] == 16384
            and audit['independent_panel_size'] == 512, 'Incomplete source evidence')
    entries = protocol['populations']
    require(len(entries) == 8 and {(r['arm'],r['stream']) for r in entries} == {(a,s) for a in ARMS for s in range(4)},
            'Changed population inventory')
    guide_specs = {arm: bound(protocol['guides'][arm]['path'])['source_chart'] for arm in ARMS}
    source_state = bound(protocol['source_state']['path'])
    densities = {a: SourceDensity(spec, source_state['pose'], source_state['anchor_pose']) for a,spec in guide_specs.items()}
    require(not args.out.exists(), 'Fresh output required'); args.out.mkdir()
    started = time.process_time(); paired = {s:[] for s in range(4)}; arm_stats=[]; checks=[]; seams={a:0 for a in ARMS}
    with (args.out/'events.jsonl').open('x') as events, (args.out/'contributions.jsonl').open('x') as contribution_file:
        def emit(row):events.write(json.dumps(row,allow_nan=False)+'\n');events.flush()
        try:
            for entry in entries:
                emit(dict(kind='population_begun', id=entry['id']))
                arm, stream = entry['arm'], entry['stream']; cfg = bound(entry['config'])
                require(cfg['source_chart'] == guide_specs[arm] and cfg['identity']['guide_sha256'] == protocol['guides'][arm]['sha256']
                        and cfg['draws'] == 2048 and cfg['mode'] == 'geometry' and cfg['mixture'] == [.5,.25,.25], 'Changed generating law')
                path = Path(entry['result'])/'rows.jsonl'; key=str(path.resolve())
                require(bindings.get(key) == audited.get(key), 'Rows lack the passed independent audit hash')
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                require(len(rows) == 2048, 'Incomplete all-attempt denominator')
                poses = [r['input']['proposed_pose'] for r in rows]
                source_logs={}; radius={}
                for a, density in densities.items():
                    source_logs[a], radius[a], seam = source_batch(density, poses); seams[a]+=int(seam.sum())
                log_u = np.asarray([log_component(r['density']['log_u']) for r in rows])
                log_g = np.asarray([log_component(r['density']['log_g'],r['density']['log_g_status']) for r in rows])
                old_q = np.asarray([r['density']['log_q'] for r in rows])
                own = np.asarray([log_component(r['density']['log_source'],r['density']['log_source_status']) for r in rows])
                checks.append(close_logs(own,source_logs[arm]))
                checks.append(close_logs(old_q,logsumexp(np.stack((log_u,log_g,source_logs[arm]))+np.log([.5,.25,.25])[:,None],axis=0)))
                qbar = balanced_log_q(log_u,log_g,source_logs['full'],source_logs['diagonal'])
                require(np.isfinite(qbar).all(), 'Zero mixture density at a saved generated pose')
                own_t=[]
                for i,row in enumerate(rows):
                    require(row['input']['ordinal'] == i and row['input']['identity'] == cfg['identity']
                            and row['complete'] and row['physical_weight_status']=='not_estimated' and row['clouds']==[], 'Changed candidate identity/scope')
                    valid = row['actual']['physical_valid']; region = row['region'] if valid else 'hard_invalid'
                    require(region in REGIONS and (valid or row['region'] is None), 'Invalid region partition')
                    is_t=region=='A_patch_complete'
                    if is_t: require(row['patches']['source_fraction']==1 and row['patches']['neighbor_labels']==[16,217], 'Wrong A_T indicator')
                    record=dict(arm=arm,stream=stream,ordinal=i,branch=row['input']['branch'],region=region,physical_valid=valid,
                        log_g=float(log_g[i]) if math.isfinite(log_g[i]) else None,
                        log_u=float(log_u[i]) if math.isfinite(log_u[i]) else None,
                        log_source={a:float(source_logs[a][i]) if math.isfinite(source_logs[a][i]) else None for a in ARMS},
                        mahalanobis_squared={a:float(radius[a][i]) if math.isfinite(radius[a][i]) else None for a in ARMS},
                        log_q_original=float(old_q[i]),log_q_balanced=float(qbar[i]),
                        log_weight=-float(qbar[i]) if valid else None,is_A_T=is_t)
                    require(not is_t or all(record['mahalanobis_squared'][a] is not None for a in ARMS), 'A_T Cayley seam cannot be classified')
                    contribution_file.write(json.dumps(record,allow_nan=False)+'\n')
                    paired[stream].append(record)
                    if is_t:own_t.append(-float(old_q[i]))
                contribution_file.flush()
                arm_stats.append(dict(arm=arm,stream=stream,**importance_summary(own_t,2048)))
                emit(dict(kind='population_complete',id=entry['id'],attempts=2048,A_T_hits=len(own_t)))
            populations=[]
            for stream,records in paired.items():
                require(len(records)==4096,'Missing paired attempts')
                target=[r for r in records if r['is_A_T']]
                stats=importance_summary([r['log_weight'] for r in target],4096)
                partitions=[dict(region=g,**importance_summary([r['log_weight'] for r in records
                    if r['region']==g and r['physical_valid']],4096),attempted_count=sum(r['region']==g for r in records)) for g in REGIONS]
                populations.append(dict(stream=stream,**stats,regions=partitions,radial_decomposition=decomposition(target,stats)))
            all_target=[r for records in paired.values() for r in records if r['is_A_T']]
            pooled=importance_summary([r['log_weight'] for r in all_target],16384)
            for path,digest in bindings.items():require(sha(path)==digest,'Input changed during reduction')
            report=dict(schema='context-covariance-balance-report-v1',complete=True,passed=True,
                protocol_sha256=sha(args.protocol),source_sha256=sha(__file__),input_sha256=bindings,
                all_attempts=16384,populations=populations,balanced_population_summary=across_populations(populations),
                original_arm_populations=arm_stats,
                original_arm_summaries={a:across_populations([r for r in arm_stats if r['arm']==a]) for a in ARMS},
                pooled=pooled,pooled_radial_decomposition=decomposition(all_target,pooled),
                maximum_saved_density_check_ratio=max(checks),source_chart_seam_counts=seams,
                contributions_sha256=sha(args.out/'contributions.jsonl'),geometry_queries=0,cloud_draws=0,new_poses=0,
                physical_weight_status='not_estimated',cpu_seconds=time.process_time()-started,
                interpretation='Hard-only defined A_T contact-region volume. Four paired-population masses estimate sampling uncertainty; ESS/largest are weight-concentration diagnostics, not independent contact samples. Combined populations have twice each individual arm draw count. No equilibrium/depletion or assembly inference.')
            (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        except BaseException as error:
            emit(dict(kind='fatal',error=repr(error)))
            (args.out/'failure.json').write_text(json.dumps(dict(complete=False,passed=False,error=repr(error),prefix_preserved=True),indent=2)+'\n')
            raise

if __name__=='__main__':main()
