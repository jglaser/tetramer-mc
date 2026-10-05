#!/usr/bin/env python3
"""Independent complete-journal audit of the correlation-0.95 cached sphere singleton reference.

Reconstructs map-factor densities, reciprocal charts, Haar/noise/label factors,
sphere endpoint predicates and retained decisions. Does not sample a bath,
generate sources, or replay random numbers. Exact thinning remains covered by
the separate bath implementation/reference, not by saved-count arithmetic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest

from analyze_context_transport_stationarity import Chart, verify_compiled_chart_source
from dimer_destination_density import exported_covariance, inverse_pose, pose_arrays
from normalizer_proposal_density import scalar_cholesky

DESIGN_SHA = 'd61dd358a0b59a50648c9dc286ec71b79ffb43639d6be48b7f19c9d7e55c3657'
RHO = .95
NOISE_SCALE = math.sqrt((1-RHO)*(1+RHO))
ARMS = ('original_prior', 'context_prior')
ATOL, RTOL = 2e-8, 2e-10


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integer(x, name):
    require(type(x) is int and x >= 0, 'Invalid integer ' + name)
    return x


def same_observations(actual, expected):
    require(type(actual) is list and len(actual) == 12 and all(type(v) is bool for v in actual)
            and actual == expected, 'Malformed or changed binary observables')


def observations(pair):
    x, a = pose_arrays(pair[0]); y, b = pose_arrays(pair[1])
    close = float((x-y) @ (x-y)) < 1.
    return [bool(x[0] < 0), bool(x[1] < 0), bool(x[2] < 0), bool(x @ x < 1), close,
            bool(x @ y > 0), bool(a[0, 0] > 0), bool(a[1, 1] > 0), bool(a[2, 2] > 0),
            bool((a.T @ b)[0, 0] > 0), bool(x[0] < 0 and a[0, 0] > 0),
            bool(close and a[1, 1] > 0)]


def sphere_status(pose, fixed):
    t, _ = pose_arrays(pose)
    wall = float(t @ t) <= 1.5**2
    clear = all(float((t-pose_arrays(q)[0]) @ (t-pose_arrays(q)[0])) >= .2**2 for q in fixed)
    return wall, clear


class Audit:
    def __init__(self):
        self.checked = 0
        self.maximum_scaled_error = 0.
        self.maximum_absolute_error = 0.

    def near(self, actual, expected, name, scale=None):
        a, e = np.asarray(actual, float), np.asarray(expected, float)
        require(a.shape == e.shape and np.isfinite(a).all() and np.isfinite(e).all(), 'Invalid reference field '+name)
        error = float(np.max(np.abs(a-e), initial=0.))
        bound = 1+float(np.max(np.abs(e), initial=0.)) if scale is None else float(scale)
        require(math.isfinite(bound) and bound >= 0, 'Invalid error scale')
        tolerance = ATOL+RTOL*bound
        require(error <= tolerance, f'{name}: difference {error} > {tolerance}')
        self.checked += 1
        self.maximum_scaled_error = max(self.maximum_scaled_error, error/tolerance)
        self.maximum_absolute_error = max(self.maximum_absolute_error, error)

    def pose(self, actual, expected, name):
        at, ar = pose_arrays(actual); et, er = pose_arrays(expected)
        self.near(at, et, name+'/translation')
        self.near(ar, er, name+'/rotation')


class AtlasReference:
    """Original normalized map-generated charts; inversion has unit measure Jacobian."""
    def __init__(self, model):
        require(model['schema'] == 'reciprocal-pose-mixture-v1' and
                model['reciprocal_components'] == [True, False], 'Changed toy flags')
        base = model['base_model']
        require(base['weights'] == [.35, .65] and base['angular_length'] == 1., 'Changed toy weights/scale')
        self.charts = []
        for anchor, mean, covariance in zip(base['anchors'], base['means'], base['covariances']):
            # Exactly the direct-loader factor -> exported LL^T -> map factor path.
            effective = exported_covariance(scalar_cholesky(covariance))
            self.charts.append(Chart(dict(anchor_position=anchor['position'], anchor_rotation=anchor['rotation'],
                mean=mean, covariance=effective), 1.))
        require(len(self.charts) == 2, 'Changed toy component count')
        self.base_indices = [0, 0, 1]
        self.flags = [False, True, False]
        self.priors = [[.175, .175, .65], [.9175, .0175, .065], [.0175, .9175, .065]]

    def component_logs(self, pose):
        opposite = inverse_pose(pose)
        return np.asarray([self.charts[b].log_density(opposite if flag else pose)
                           for b, flag in zip(self.base_indices, self.flags)])

    def proposal(self, old, candidate, info, prior_index, audit):
        require(info['anchor_index'] == 0, 'Changed fixed-origin anchor')
        if info['branch'] == 'uniform':
            t, _ = pose_arrays(candidate)
            require(np.all(t >= -1.5) and np.all(t < 1.5), 'Uniform candidate outside fixed cube')
            audit.near(info['log_reverse_forward'], 0., 'uniform logq')
            return 0., 1.
        require(info['branch'] == 'involution' and info['source_law'] == 'posterior' and
                info['identity'] is False, 'Wrong learned proposal kind')
        trace = info['trace']; i, j = trace['source'], trace['target']
        require(type(i) is int and type(j) is int and 0 <= i < 3 and 0 <= j < 3, 'Invalid selected labels')
        require(info['source_component_index'] == self.base_indices[i] and
                info['target_component_index'] == self.base_indices[j] and
                info['source_inverted'] is self.flags[i] and info['target_inverted'] is self.flags[j],
                'Changed reciprocal branch representation')
        old_input = inverse_pose(old) if self.flags[i] else old
        a, b = self.charts[self.base_indices[i]], self.charts[self.base_indices[j]]
        z, noise = a.encode(old_input), np.asarray(trace['noise'], float)
        require(noise.shape == (6,) and np.isfinite(noise).all(), 'Invalid map noise')
        target_z=RHO*z+NOISE_SCALE*noise
        reverse_noise=NOISE_SCALE*z-RHO*noise
        proposed = b.decode(target_z)
        proposed = inverse_pose(proposed) if self.flags[j] else proposed
        audit.pose(candidate, proposed, 'candidate')
        step = info['step']
        audit.pose(step['pose'], proposed, 'map pose')
        audit.near(step['source_latent'], z, 'source latent')
        audit.near(step['target_latent'], target_z, 'correlated target latent')
        require(step['inverse_trace']['source'] == j and step['inverse_trace']['target'] == i, 'Wrong inverse labels')
        audit.near(step['inverse_trace']['noise'], reverse_noise, 'correlated inverse noise')
        norm_before=float(z@z+noise@noise);norm_after=float(target_z@target_z+reverse_noise@reverse_noise)
        audit.near(norm_after,norm_before,'extended Gaussian norm',1+abs(norm_before)+abs(norm_after))
        ja, jb = a.log_volume(z), b.log_volume(target_z)
        logj = jb-ja; old_noise, next_noise = -.5*float(noise @ noise), -.5*float(reverse_noise @ reverse_noise)
        logn = next_noise-old_noise
        map_scale = 1+abs(ja)+abs(jb)+abs(old_noise)+abs(next_noise)
        audit.near(step['log_extended_jacobian'], logj, 'Haar Jacobian', 1+abs(ja)+abs(jb))
        audit.near(step['log_auxiliary_ratio'], logn, 'Gaussian noise ratio', 1+abs(old_noise)+abs(next_noise))
        audit.near(step['log_correction'], logj+logn, 'map correction', map_scale)
        # Reverse from the SAVED target, rather than decode(encode(old)).
        # Both exact reciprocal wrappers participate in the round trip.
        saved_target=inverse_pose(candidate) if self.flags[j] else candidate
        encoded_target=b.encode(saved_target)
        saved_inverse_noise=np.asarray(step['inverse_trace']['noise'],float)
        recovered_z=RHO*encoded_target+NOISE_SCALE*saved_inverse_noise
        recovered_noise=NOISE_SCALE*encoded_target-RHO*saved_inverse_noise
        audit.near(encoded_target,target_z,'saved target encoding')
        audit.near(recovered_z,z,'reverse source latent')
        audit.near(recovered_noise,noise,'reverse recovered forward noise')
        back = a.decode(recovered_z); back = inverse_pose(back) if self.flags[i] else back
        audit.pose(back, old, 'inverse physical recovery')
        lp = np.log(self.priors[prior_index]); lx = self.component_logs(old); ly = self.component_logs(candidate)
        gx, gy = float(logsumexp(lp+lx)), float(logsumexp(lp+ly))
        audit.near(info['selected_source_log_density'], lx[i], 'source component density')
        audit.near(info['selected_target_log_density'], ly[j], 'target component density')
        audit.near(info['full_old_gaussian_log_density'], gx, 'full old density')
        audit.near(info['full_new_gaussian_log_density'], gy, 'full new density')
        source = lp[i]+lx[i]-gx; reverse = lp[j]+ly[j]-gy
        label = reverse+lp[i]-source-lp[j]
        label_scale = 1+abs(lp[i])+abs(lp[j])+abs(lx[i])+abs(ly[j])+abs(gx)+abs(gy)
        audit.near(info['source_log_probability'], source, 'source responsibility', 1+abs(lp[i])+abs(lx[i])+abs(gx))
        audit.near(info['inverse_source_log_probability'], reverse, 'inverse responsibility', 1+abs(lp[j])+abs(ly[j])+abs(gy))
        audit.near(info['label_log_reverse_forward'], label, 'label ratio', 2*label_scale)
        audit.near(info['expanded_log_reverse_forward'], logj+logn+label, 'expanded correction', map_scale+2*label_scale)
        audit.near(logj+logn+label, gx-gy, 'cancellation identity', map_scale+2*label_scale)
        audit.near(info['log_reverse_forward'], gx-gy, 'full proposal ratio', 1+abs(gx)+abs(gy))
        return gx-gy, 1+abs(gx)+abs(gy)


def paired_test(counts, cutoff):
    zero, one, up, down = map(int, counts)
    require(zero+one == 8192 and 0 <= up <= zero and 0 <= down <= one, 'Invalid paired counts')
    n = up+down
    p = float(binomtest(up, n, .5, alternative='two-sided').pvalue) if n else 1.
    return dict(source_zero=zero, source_one=one, up=up, down=down, changed=n,
                paired_mean_difference=(up-down)/8192, p_value=p, bonferroni_cutoff=cutoff,
                reject=p < cutoff, passed=p >= cutoff)


def audit_reference(directory, source_bundle, producer_source, expected_producer_sha, expected_step_sha):
    protocol, summary = read(directory/'protocol.json'), read(directory/'summary.json')
    require(protocol['schema']=='context-prior-correlated095-physical-reference-v1'
        and summary['schema']=='context-prior-correlated095-physical-summary-v1'
        and protocol['correlation']==summary['correlation']==RHO,'Wrong correlation/reference contract')
    require(summary['complete'] and summary['execution_passed'] and summary['passed'], 'Incomplete physical reference')
    require(summary['events_sha256'] == sha(directory/'events.jsonl') and summary['protocol_sha256'] == sha(directory/'protocol.json'), 'Changed completed records')
    require(protocol['design_sha256'] == DESIGN_SHA and protocol['physical_calls'] == 16384 and protocol['source_points'] == 8192,
            'Changed fixed reference allocation')
    bundle = read(source_bundle)
    require(protocol['source_bundle_sha256'] == sha(source_bundle), 'Wrong compiled source bundle')
    require(sha(producer_source) == expected_producer_sha == protocol['source_example_sha256'], 'Wrong compiled example authority')
    for name in ('src/basin_involution.rs', 'src/context_docking.rs', 'src/proposal.rs'):
        entry = bundle['files'][name]
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'Corrupt compiled source')
    require(bundle['files']['src/context_docking.rs']['sha256'] == expected_step_sha, 'Wrong shared physical step')
    require(bundle['files']['src/proposal.rs']['sha256'] == '810333e16291d56d3bae19e02504fb53cb529a1c4626c3ea2fe75c7b21eeb217', 'Unreviewed loader arithmetic')
    verify_compiled_chart_source(bundle['files']['src/basin_involution.rs']['text'])
    bindings = (protocol['source_cache'], protocol['source_receipt'])
    for binding in bindings:
        require(sha(binding['path']) == binding['sha256'], 'Changed cached source input')
    require(read(bindings[1]['path']) == protocol['source_authority'], 'Source authority differs')
    sources = [json.loads(line) for line in Path(bindings[0]['path']).read_text().splitlines()]
    require(len(sources) == 8192, 'Missing cached sources')
    reference, audit = AtlasReference(protocol['atlas_model']), Audit()
    audit.near(protocol['effective_log_priors'], np.log(reference.priors), 'complete normalized prior vectors')
    design = protocol['design']; names = [r['name'] for r in design['binary_diagnostics']]
    require(design['proposal']['correlation']==RHO and
        'context-prior-cached-iid-correlated095-reference-v1' in design['proposal']['role_seeds'],
        'Wrong correlation or fresh role-seed namespace')
    require(names == protocol['observable_names'] and len(names) == 12, 'Changed observable panel')
    paired = np.zeros((3, 4, 12, 4), dtype=np.int64)
    count_keys = ('attempted','learned','uniform','hard_rejected','physical_decisions','accepted',
        'learned_nonzero_ratio','accepted_learned_nonzero_ratio','gained','lost','raw_points','retained_points')
    counts = [{key:0 for key in count_keys} for _ in ARMS]
    events_read = 0; total_raw = 0; total_retained = 0
    origin = dict(position=[0.,0.,0.], orientation=[1.,0.,0.,0.])
    with (directory/'events.jsonl').open() as stream:
        def event(kind):
            nonlocal events_read
            line = next(stream, None)
            require(line is not None, 'Truncated event inventory')
            row = json.loads(line); events_read += 1
            require(row['kind'] == kind, 'Changed event order')
            return row
        for ordinal, source in enumerate(sources):
            s, i = divmod(ordinal, 2048)
            require(source['stream'] == s and source['source_index'] == i and source['kind'] == 'cached_iid_source', 'Changed source order')
            begun, entry = event('source_begin'), event('source')
            require((begun['stream'],begun['source_index'],begun['source_attempt']) == (s,i,source['source_attempt']) and
                    entry['source'] == source, 'Filtered/replaced source')
            old_pair = source['old']; old = old_pair[0]; fixed = [origin, old_pair[1]]
            before = observations(old_pair)
            same_observations(entry['observations'],before)
            require(all(sphere_status(old,fixed)), 'Wrong source predicate')
            context_index = 1 if old_pair[1]['position'][0] >= 0 else 2
            require(entry['context_prior_index'] == context_index, 'Prior depends on wrong state')
            for arm_index, arm in enumerate(ARMS):
                begin, row = event('arm_begin'), event('arm_result')
                for e in (begin,row): require((e['stream'],e['source_index'],e['arm']) == (s,i,arm), 'Wrong arm inventory')
                step = row['step']; c = counts[arm_index]; c['attempted'] += 1
                require(step['old_pose'] == old and type(step['accepted']) is bool, 'Wrong source or acceptance type')
                candidate = step['proposed_pose']; require(candidate is not None, 'Uncertified numerical null')
                prior_index = 0 if arm_index == 0 else context_index
                logq, qscale = reference.proposal(old,candidate,step['proposal'],prior_index,audit)
                learned = step['proposal']['branch'] == 'involution'
                c['learned' if learned else 'uniform'] += 1
                wall, core = sphere_status(candidate,fixed)
                require(step['wall_valid'] is wall, 'Wrong atomic sphere-wall predicate')
                if wall: require(step['core_valid'] is core, 'Wrong sphere core predicate')
                else: require(step['core_valid'] is None, 'Unexpected core test after wall rejection')
                audit.near(step['log_proposal_reverse_forward'], logq, 'saved complete proposal ratio', qscale)
                physical = wall and core
                if not physical:
                    require(step['status'] == ('wall_rejected' if not wall else 'core_rejected') and
                            step['gate'] is None and step['accepted'] is False and
                            all(step[k] is None for k in ('raw_log_acceptance','log_acceptance','acceptance_uniform','log_uniform')),
                            'Invalid hard-rejection accounting')
                    c['hard_rejected'] += 1
                else:
                    g = step['gate']; require(isinstance(g,dict), 'Missing complete bath cloud')
                    for key in ('gained','lost','raw_points','retained_points','retained_cells','created_cells'): integer(g[key],key)
                    require(g['retained_points'] == g['gained']+g['lost'] <= g['raw_points'] <= 100000, 'Invalid complete cloud counts')
                    weight = math.log1p(.5/2.)*(g['gained']-g['lost'])
                    audit.near(g['log_weight'],weight,'bath count ratio',1+abs(weight))
                    raw = weight+logq; raw_scale = 1+abs(weight)+qscale
                    audit.near(step['raw_log_acceptance'],raw,'raw physical ratio',raw_scale)
                    audit.near(step['log_acceptance'],min(0.,raw),'clipped physical ratio',raw_scale)
                    u = step['acceptance_uniform']; require(type(u) in (int,float) and 0 <= u < 1, 'Invalid acceptance U')
                    logu = math.log(u) if u else -math.inf
                    if u: audit.near(step['log_uniform'],logu,'log U')
                    else: require(step['log_uniform'] is None,'Zero U must retain explicit null-log convention')
                    take = logu < min(0.,raw)
                    require(step['accepted'] is take and step['status'] == ('accepted' if take else 'bath_rejected'), 'Wrong physical decision')
                    c['physical_decisions'] += 1; c['accepted'] += int(take)
                    for key in ('gained','lost','raw_points','retained_points'): c[key] += g[key]
                    total_raw += g['raw_points']; total_retained += g['retained_points']
                    if learned and abs(logq) > 1e-9:
                        c['learned_nonzero_ratio'] += 1; c['accepted_learned_nonzero_ratio'] += int(take)
                retained = candidate if step['accepted'] else old
                require(step['retained_pose'] == retained and all(sphere_status(retained,fixed)), 'Lost rejected residence')
                after = observations([retained,old_pair[1]])
                same_observations(row['observations_retained'],after)
                require(row['completed_raw_points'] == total_raw and row['completed_retained_points'] == total_retained, 'Bath budget counted twice or dropped')
                rows = [(arm_index,after)]
                if arm_index == 0:
                    require(row['negative'] is None, 'Wrong negative-control allocation')
                else:
                    neg = row['negative']; take = physical and (step['acceptance_uniform'] == 0 or math.log(step['acceptance_uniform']) < min(0.,step['gate']['log_weight']))
                    np_pose = candidate if take else old
                    require(neg['id'] == 'context_omitted_proposal_ratio' and neg['accepted'] is take and neg['retained_pose'] == np_pose,
                            'Wrong paired negative control')
                    if physical: audit.near(neg['log_acceptance'],min(0.,step['gate']['log_weight']),'negative ratio')
                    else: require(neg['log_acceptance'] is None,'Negative hard rejection drew a new gate')
                    no = observations([np_pose,old_pair[1]])
                    same_observations(neg['observations'],no)
                    rows.append((2,no))
                for ai,values in rows:
                    for k,(b,a) in enumerate(zip(before,values)):
                        paired[ai,s,k,int(b)] += 1
                        paired[ai,s,k,2] += int(not b and a)
                        paired[ai,s,k,3] += int(b and not a)
        require(next(stream,None) is None,'Unexpected extra events')
    require(events_read == summary['events'] == 49152 and summary['physical_calls'] == 16384 and
            summary['negative_algebraic_decisions'] == 8192 and summary['source_points'] == 8192, 'Incomplete allocation')
    require(counts == summary['counts'] and paired.tolist() == summary['paired_counts'], 'Saved counts disagree with independent reduction')
    require(summary['raw_points'] == total_raw <= 10000000 and summary['retained_points'] == total_retained <= 10000000,
            'Changed total bath allocation')
    require(all(summary[k] == 0 for k in ('new_source_draws','retries','replacements','extensions')), 'Unplanned source allocation')
    exercises = [c['attempted']==8192 and c['learned']>100 and c['uniform']>100 and c['hard_rejected']>0 and
        c['physical_decisions']>100 and c['accepted']>50 and c['learned_nonzero_ratio']>20 and
        c['accepted_learned_nonzero_ratio']>0 and c['gained']>0 and c['lost']>0 for c in counts]
    require(summary['exercise_passed'] is all(exercises), 'Exercise flag disagreement')
    totals = paired.sum(axis=1)
    valid = [dict(arm=arm,tests=[dict(observable=name,**paired_test(c,.05/24))
        for name,c in zip(names,totals[index])]) for index,arm in enumerate(ARMS)]
    negative = [dict(observable=name,**paired_test(c,.05/12)) for name,c in zip(names,totals[2])]
    primary_pass = all(t['passed'] for arm in valid for t in arm['tests'])
    return dict(schema='context-prior-correlated095-reference-independent-audit-v1',correlation=RHO,complete=True,
        passed=primary_pass and all(exercises),arithmetic_passed=True,primary_tests_passed=primary_pass,
        exercise_passed=all(exercises),negative_control_detected=any(t['reject'] for t in negative),
        negative_rejected=sum(t['reject'] for t in negative),negative_weak_limitation=not any(t['reject'] for t in negative),
        valid_arms=valid,negative_tests=negative,per_population_paired_counts=paired.tolist(),counts=counts,
        retained_sources=8192,retained_calls=16384,retained_events=events_read,
        raw_points=total_raw,retained_points=total_retained,
        numerical_checks=dict(fields=audit.checked,absolute_tolerance=ATOL,relative_tolerance=RTOL,
            maximum_absolute_difference=audit.maximum_absolute_error,maximum_scaled_difference=audit.maximum_scaled_error),
        source_hashes=dict(producer=expected_producer_sha,shared_step=expected_step_sha,source_bundle=sha(source_bundle)),
        scope='Complete saved-record analytic sphere/map arithmetic and IID paired stationarity diagnostic. No source/bath/RNG generation or protein geometry. Does not independently establish thinning, full finite-precision balance, trajectory mixing or protein assembly.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('directory','source-bundle','producer-source','out'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-producer-sha',required=True); p.add_argument('--expected-step-sha',required=True)
    args = p.parse_args()
    require(not args.out.exists(),'Fresh audit output required')
    result = audit_reference(args.directory,args.source_bundle,args.producer_source,args.expected_producer_sha,args.expected_step_sha)
    result['analyzer_sha256'] = sha(__file__)
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('passed','primary_tests_passed','exercise_passed','negative_control_detected','negative_rejected')}))


if __name__ == '__main__':
    main()
