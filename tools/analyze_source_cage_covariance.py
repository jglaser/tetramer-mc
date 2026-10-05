"""Residence-weighted descriptive pose covariance from four completed local chains.
No geometry, point clouds, proposal fitting, or proposal-density evaluations.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

QUANTILES = [0., .01, .05, .25, .5, .75, .95, .99, 1.]
SCHEMA = 'source-cage-saved-covariance-v1'


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path): return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def qnorm(values):
    q = np.asarray(values, float)
    require(q.shape[-1] == 4 and np.isfinite(q).all(), 'Invalid quaternion coordinates')
    lengths = np.linalg.norm(q, axis=-1, keepdims=True)
    require((np.abs(lengths*lengths-1) <= 1e-8).all(), 'Quaternion outside production normalization tolerance')
    return q/lengths


def qconj(q): return np.asarray(q)*np.array([1., -1., -1., -1.])


def qmul(a, b):
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    w = a[..., :1]*b[..., :1]-np.sum(a[..., 1:]*b[..., 1:], axis=-1, keepdims=True)
    v = a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]+np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((w, v), axis=-1)


def rotate(q, v):
    q, v = np.asarray(q), np.asarray(v)
    return v+2*np.cross(q[..., 1:], np.cross(q[..., 1:], v)+q[..., :1]*v)


def coordinates(poses, source, anchor, ell):
    p = np.asarray([r['position'] for r in poses], float)
    require(p.shape == (len(poses), 3) and np.isfinite(p).all(), 'Invalid saved position')
    q = qnorm([r['orientation'] for r in poses])
    qa, qs = qnorm(anchor['orientation']), qnorm(source['orientation'])
    t = rotate(qconj(qa), p-np.asarray(anchor['position']))
    ts = rotate(qconj(qa), np.asarray(source['position'])-anchor['position'])
    qr = qmul(qconj(qa), q)
    qsr = qmul(qconj(qa), qs)
    delta = qnorm(qmul(qr, qconj(qsr)))
    w, v = delta[:, 0], delta[:, 1:]
    bad = w == 0
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        x = np.c_[t-ts, ell*v/w[:, None]]
    bad |= ~np.isfinite(x).all(axis=1)
    # The chart ratio is unchanged by quaternion sign. Rotation magnitudes use
    # the shortest rotation and are only descriptive, not Gaussian coordinates.
    angles = np.degrees(2*np.arctan2(np.linalg.norm(v, axis=1), np.abs(w)))
    return x, dict(failure_indices=np.flatnonzero(bad).tolist(),
        minimum_absolute_quaternion_scalar=float(np.min(np.abs(w))),
        near_seam_count=int(np.sum(np.abs(w) <= 1e-12)),
        translation_norm=np.linalg.norm(t-ts, axis=1), rotation_angle_degrees=angles)


def covariance(x):
    x = np.asarray(x, float)
    require(x.ndim == 2 and x.shape[1] == 6 and len(x) > 0 and np.isfinite(x).all(), 'Invalid descriptive covariance input')
    mean = x.mean(axis=0)
    centered = x-mean
    return mean, centered.T@centered/len(x)


def matrix_description(x):
    mean, cov = covariance(x)
    standard = np.sqrt(np.maximum(np.diag(cov), 0))
    denominator = standard[:, None]*standard[None, :]
    correlations = np.divide(cov, denominator, out=np.zeros_like(cov), where=denominator > 0)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return dict(mean=mean.tolist(), covariance=cov.tolist(), standard_deviation=standard.tolist(),
        correlation=[[float(correlations[i, j]) if denominator[i, j] > 0 else None for j in range(6)] for i in range(6)],
        eigenvalues=eigenvalues.tolist(), eigenvectors_columns=eigenvectors.tolist(),
        numerical_rank=int(np.linalg.matrix_rank(cov)),
        condition_number=float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0] > 0 else None,
        quantiles=np.quantile(x, QUANTILES, axis=0, method='linear').tolist())


def describe(x, movements, scales, ell):
    return dict(samples=len(x), covariance_denominator='N; descriptive residence distribution, not independent-sample uncertainty',
        map_coordinates_angstrom=matrix_description(x),
        physical_translation_angstrom_cayley_dimensionless=matrix_description(x/np.array([1., 1., 1., ell, ell, ell])),
        source_width_scaled=matrix_description(x/scales),
        translation_norm_angstrom_quantiles=np.quantile(movements['translation_norm'], QUANTILES, method='linear').tolist(),
        rotation_angle_degrees_quantiles=np.quantile(movements['rotation_angle_degrees'], QUANTILES, method='linear').tolist())


def decompose(groups, scales):
    all_values = np.concatenate(groups)
    mean, total = covariance(all_values)
    within = np.zeros((6, 6)); between = np.zeros((6, 6))
    for x in groups:
        m, c = covariance(x); weight = len(x)/len(all_values)
        within += weight*c
        between += weight*np.outer(m-mean, m-mean)
    residual = np.max(np.abs(total-within-between))
    require(residual <= 1e-12+1e-10*np.max(np.abs(total)), 'Within/between covariance identity failed')
    norm = np.outer(scales, scales)
    return dict(total=total.tolist(), within_stream=within.tolist(), between_stream_means=between.tolist(),
        scaled_total=(total/norm).tolist(), scaled_within_stream=(within/norm).tolist(),
        scaled_between_stream_means=(between/norm).tolist(), maximum_identity_residual=float(residual),
        weights=[len(x)/len(all_values) for x in groups], interpretation='Descriptive trajectory spread decomposition; between-stream differences need not be equilibrated.')


def retained_states(path, identity, source_pose, cycles=2304, warmup=256):
    rows = []; previous = source_pose; events = 0; completed_cycles = 0; initial = 0
    with Path(path).open() as stream:
        for line in stream:
            event = json.loads(line)
            require(event['event_index'] == events, 'Missing/reordered physical event'); events += 1
            if event['kind'] == 'initial_state':
                require(event['pose'] == source_pose and event['identity'] == identity and initial == 0, 'Wrong source preparation')
                initial += 1
            elif event['kind'] == 'attempt_complete':
                index = len(rows); cycle, slot = index//5+1, index%5
                require(event['attempt_index'] == index and event['cycle'] == cycle and event['slot'] == slot
                        and event['identity'] == identity and event['global'] is False
                        and event['production'] is (cycle > warmup), 'Wrong local retained-state inventory')
                require(event['old_pose'] == previous
                        and event['retained_pose'] == (event['proposed_pose'] if event['accepted'] else previous),
                        'Lost rejection residence or pose recurrence')
                previous = event['retained_pose']
                rows.append(dict(attempt_index=index, cycle=cycle, slot=slot, production=cycle > warmup,
                    pose=previous, accepted=event['accepted'], neighbors=event['exclusion_contact_labels']))
            elif event['kind'] == 'cycle_complete':
                completed_cycles += 1
                require(event['cycle'] == completed_cycles and len(rows) == 5*completed_cycles
                        and event['pose'] == previous, 'Missing cycle closure')
            elif event['kind'] in ('fatal', 'attempt_failed'):
                raise ValueError('Failed physical trajectory is inadmissible')
    require(initial == 1 and len(rows) == cycles*5 and completed_cycles == cycles
            and events == 4+cycles*16, 'Incomplete physical trajectory')
    return rows, dict(events=events, attempted_states=len(rows), production_states=sum(r['production'] for r in rows),
        rejected_states=sum(not r['accepted'] for r in rows), accepted_states=sum(r['accepted'] for r in rows))


def source_initial_matches(initial, source):
    return initial is None or initial == source


def success(receipt, terminal):
    r = read(receipt)
    require(r['success'] and r['child_drained'] and r['returncode'] == 0
            and Path(r['terminal']['path']).resolve() == Path(terminal).resolve()
            and r['terminal']['sha256'] == sha(terminal), 'Unauthenticated completed/drained result')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); protocol = read(args.protocol)
    require(protocol['schema'] == SCHEMA and protocol['streams'] == [0, 1, 2, 3]
            and protocol['train_streams'] == [0, 1] and protocol['heldout_streams'] == [2, 3]
            and protocol['quantiles'] == QUANTILES and protocol['covariance_denominator'] == 'N', 'Changed descriptive protocol')
    for p, h in protocol['input_sha256'].items(): require(sha(p) == h, 'Changed frozen input: '+p)
    def bound(p):
        require(str(Path(p).resolve()) in protocol['input_sha256'], 'Unbound reduction input: '+str(p))
        return read(p)
    source = bound(protocol['source']); context = bound(protocol['context']); model = bound(protocol['model'])
    ell = float(model.get('base_model', model)['angular_length'])
    require(ell == protocol['angular_length'] and context['excluded_moving_labels'] == [77]
            and source['moving_label'] == 77 and source['anchor_label'] == context['anchor_label'] == 16,
            'Changed source/chart frame')
    anchor = next(b['pose'] for b in context['bodies'] if b['label'] == 16)
    require(source['anchor_pose'] == anchor, 'Source/anchor mismatch')
    scales = np.array([.1]*3+[ell*math.tan(math.pi/720)]*3)
    require(np.allclose(scales, protocol['scale_vector'], rtol=1e-15, atol=0), 'Changed reporting scale')
    require(len(protocol['inventory']) == 4 and [x['stream'] for x in protocol['inventory']] == [0, 1, 2, 3], 'Wrong stream inventory')
    args.out.mkdir(); started = time.process_time(); populations = []; data = {}; audits = []
    with (args.out/'attempts.jsonl').open('x') as journal:
        def emit(v): journal.write(json.dumps(v, allow_nan=False)+'\n'); journal.flush()
        try:
            for item in protocol['inventory']:
                emit(dict(kind='stream_begun', stream=item['stream']))
                report = bound(item['observer_report']); summary = bound(item['physical_summary']); cfg = bound(item['config'])
                success(item['physical_receipt'], item['physical_summary']); success(item['observer_receipt'], item['observer_report'])
                require(report['complete'] and report['passed'] and summary['method'] == 'local'
                        and report['identity'] == summary['identity'] == cfg['identity']
                        and cfg['identity']['arm'] == 'local' and cfg['identity']['start'] == 'saved_body77'
                        and cfg['identity']['stream'] == item['stream'] and cfg['seed'] == item['seed']
                        and report['input_sha256'][item['journal']] == sha(item['journal'])
                        and source_initial_matches(cfg['initial_pose'], source['pose']), 'Changed local control authority')
                for name, p in [('source_state', protocol['source']), ('fixed_context', protocol['context']), ('model', protocol['model'])]:
                    require(summary['bindings'][name] == sha(p), 'Changed local physical/proposal input')
                rows, audit = retained_states(item['journal'], cfg['identity'], source['pose'])
                for phase in ('warmup', 'production'):
                    selected = [r for r in rows if r['production'] is (phase == 'production')]
                    x, movement = coordinates([r['pose'] for r in selected], source['pose'], anchor, ell)
                    failures = [selected[i]['attempt_index'] for i in movement['failure_indices']]
                    one = dict(stream=item['stream'], phase=phase, states=len(selected),
                        accepted_states=sum(r['accepted'] for r in selected), rejected_states=sum(not r['accepted'] for r in selected),
                        neighbor_residence={','.join(map(str,k)):v for k,v in Counter(tuple(r['neighbors']) for r in selected).items()},
                        chart_failures=failures, near_seam_count=movement['near_seam_count'],
                        minimum_absolute_quaternion_scalar=movement['minimum_absolute_quaternion_scalar'])
                    if failures: one['coordinate_status'] = 'undefined; no state dropped or replaced'
                    else:
                        one.update(coordinate_status='complete', **describe(x, movement, scales, ell))
                        data[(item['stream'], phase)] = (x, movement)
                    populations.append(one)
                audits.append(dict(stream=item['stream'], **audit)); emit(dict(kind='stream_complete', stream=item['stream'], **audit))
            grouped = []
            for phase in ('warmup', 'production'):
                for label, streams in [('all', [0,1,2,3]), ('train', [0,1]), ('heldout', [2,3])]:
                    if not all((s, phase) in data for s in streams):
                        grouped.append(dict(phase=phase, group=label, streams=streams, coordinate_status='undefined due to preserved chart failure')); continue
                    groups = [data[(s,phase)][0] for s in streams]
                    movement = {k:np.concatenate([data[(s,phase)][1][k] for s in streams]) for k in ('translation_norm','rotation_angle_degrees')}
                    grouped.append(dict(phase=phase, group=label, streams=streams, coordinate_status='complete',
                        **describe(np.concatenate(groups), movement, scales, ell), covariance_decomposition=decompose(groups, scales)))
            for p, h in protocol['input_sha256'].items(): require(sha(p) == h, 'Input changed during reduction')
            report = dict(schema=SCHEMA, complete=True, passed=True, source_sha256=sha(__file__), protocol_sha256=sha(args.protocol),
                input_sha256=protocol['input_sha256'], chart=protocol['chart'], scale_vector=scales.tolist(), angular_length=ell,
                quantile_probabilities=QUANTILES, inventory_audits=audits, per_stream=populations, pooled=grouped,
                production_states=sum(r['production_states'] for r in audits), total_retained_states=sum(r['attempted_states'] for r in audits),
                chart_failures=sum(len(r['chart_failures']) for r in populations),
                regularization_candidates=protocol['regularization_candidates'],
                cpu_seconds=time.process_time()-started, geometry_queries=0, cloud_draws=0, proposal_draws=0, fitted_proposal_densities=0,
                interpretation='Descriptive residence-weighted motion of four source-start local trajectories. Covariance is not an equilibrium width or uncertainty estimate; training/heldout split is reserved, not used to fit or score proposals.')
            with (args.out/'report.json').open('x') as f: json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
        except BaseException as e:
            emit(dict(kind='fatal', error=repr(e), completed_streams=len(audits)))
            with (args.out/'failure.json').open('x') as f: json.dump(dict(complete=False,passed=False,error=repr(e),prefix_preserved=True),f)
            raise


if __name__ == '__main__': main()
