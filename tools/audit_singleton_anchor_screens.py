#!/usr/bin/env python3
"""Bounded atlas-mean arithmetic on the five frozen context-0 states.

This is a screen diagnostic, not a catalogue builder or a sampler. It never
factorizes covariance, fits a Gaussian, or queries atomic/contact geometry.
Preparation authenticates metadata only; execution requires the existing
single-worker controller. Original screens are checked before alternatives.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import scipy
from scipy.spatial import cKDTree
import scipy.spatial._ckdtree as _ckdtree
import run_native_class_physical_campaign as driver

read, require, sha, write = driver.read, driver.require, driver.sha, driver.write
SCHEMA = 'singleton-anchor-screens-v1'
LIMITS = dict(cpu_limit_seconds=300, wall_limit_seconds=600, address_space_limit_bytes=2*2**30)
POLICY = dict(max_screens=90, branches=2048, nearest_anchors=8, distance_A=8.,
              angle_radians=math.pi/3, distance_pair_cap=500000, query_chunk=64,
              distance_ambiguity_A=1e-8, angle_ambiguity_radians=1e-10)
COUNT_KEYS = ('single_labels', 'decoded_means', 'mean_decode_failures', 'possible_interface_pairs',
              'decoded_interface_pairs', 'distance_rejected_pairs', 'angle_rejected_pairs',
              'candidate_pairs_before_cap')
SCOPE = ('Fixed context 0, source and four original starts, both moving labels. '
         'Mean-distance/orientation necessary screens only: no Gaussian support, feasible-center, '
         'native, sampling-efficiency or equilibrium conclusion. Alternatives are never ranked by outcomes.')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def source_paths():
    return [Path(__file__).resolve(), Path(driver.__file__).resolve()]


def runtime():
    import numpy._core._multiarray_umath as native_numpy
    paths = [sys.executable, np.__file__, scipy.__file__, native_numpy.__file__, _ckdtree.__file__]
    return dict(python=sys.version, optimization=sys.flags.optimize,
                executable_argv=str(Path(sys.executable).absolute()),
                numpy=np.__version__, scipy=scipy.__version__,
                files={str(Path(p).resolve()): sha(p) for p in paths})


def recheck(files):
    for path, digest in files.items(): require(sha(path) == digest, 'Changed frozen input: '+path)


def finite_array(value, shape):
    a = np.asarray(value, dtype=float)
    require(a.shape == shape and np.isfinite(a).all(), 'Invalid finite array')
    return a


def rotation(q):
    q = finite_array(q, (4,)); n = math.sqrt(sum(float(v)*float(v) for v in q))
    require(abs(n*n-1.) <= 1e-8, 'Unnormalized quaternion')
    w, x, y, z = q/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def quaternion(r):
    r = finite_array(r, (3, 3)); tr = r[0, 0]+r[1, 1]+r[2, 2]
    if tr > 0:
        s = 2*math.sqrt(tr+1)
        q = [s/4, (r[2,1]-r[1,2])/s, (r[0,2]-r[2,0])/s, (r[1,0]-r[0,1])/s]
    elif r[0,0] > r[1,1] and r[0,0] > r[2,2]:
        s = 2*math.sqrt(1+r[0,0]-r[1,1]-r[2,2])
        q = [(r[2,1]-r[1,2])/s, s/4, (r[0,1]+r[1,0])/s, (r[0,2]+r[2,0])/s]
    elif r[1,1] > r[2,2]:
        s = 2*math.sqrt(1+r[1,1]-r[0,0]-r[2,2])
        q = [(r[0,2]-r[2,0])/s, (r[0,1]+r[1,0])/s, s/4, (r[1,2]+r[2,1])/s]
    else:
        s = 2*math.sqrt(1+r[2,2]-r[0,0]-r[1,1])
        q = [(r[1,0]-r[0,1])/s, (r[0,2]+r[2,0])/s, (r[1,2]+r[2,1])/s, s/4]
    q = np.array(q); q *= (-1 if q[0] < 0 else 1)/math.sqrt(sum(float(v)*float(v) for v in q))
    return q


def compose(a, b):
    ra = rotation(a[1])
    return a[0]+ra@b[0], quaternion(ra@rotation(b[1]))


def invert(pose):
    p, q = pose
    return -(rotation(q).T@p), q*np.array([1., -1., -1., -1.])


def decode_virtual_means(model, expected_branches=None):
    """Decode z=0 directly. Stored covariance entries are never evaluated."""
    require(model['schema'] == 'reciprocal-pose-mixture-v1', 'Wrong reciprocal atlas schema')
    base = model['base_model']; flags = model['reciprocal_components']
    require(base['schema'] == 'weighted-pose-mixture-v1'
            and base['coordinate_convention'] == 'anchor-body-relative', 'Wrong base chart convention')
    ell = base['angular_length']; require(type(ell) in (int,float) and math.isfinite(ell) and ell > 0, 'Invalid angular scale')
    n = len(base['anchors'])
    require(n > 0 and len(base['means']) == len(flags) == n and all(type(f) is bool for f in flags), 'Wrong chart inventory')
    result = []
    for index, (anchor, mean, reciprocal) in enumerate(zip(base['anchors'], base['means'], flags)):
        mean = finite_array(mean, (6,)); position = finite_array(anchor['position'], (3,))+mean[:3]
        r = finite_array(anchor['rotation'], (3,3))
        require(np.max(np.abs(r.T@r-np.eye(3))) < 1e-8 and abs(np.linalg.det(r)-1) < 1e-8,
                'Invalid atlas rotation')
        u = mean[3:]/ell; d = math.hypot(1., *u)
        pose = (position, quaternion(rotation(np.r_[1., u]/d)@r))
        require(np.isfinite(pose[0]).all(), 'Unrepresentable chart mean')
        for reciprocal_branch in ([False, True] if reciprocal else [False]):
            p, q = invert(pose) if reciprocal_branch else pose
            result.append(dict(component=index, inverted=reciprocal_branch, position=p, orientation=q))
    require(expected_branches is None or len(result) == expected_branches, 'Wrong virtual branch count')
    return result


def placed_means(branches, anchor):
    a = (finite_array(anchor['position'], (3,)), finite_array(anchor['orientation'], (4,)))
    identity = (np.zeros(3), np.array([1.,0.,0.,0.]))
    values = [compose(compose(a, (b['position'], b['orientation'])), identity) for b in branches]
    positions, quaternions = np.array([v[0] for v in values]), np.array([v[1] for v in values])
    require(np.isfinite(positions).all() and np.isfinite(quaternions).all(), 'Invalid placed mean')
    return positions, quaternions


def anchor_inventory(poses, moving, other, control, rmean, nearest=8):
    """Reads no moving pose. The original control is always first, exactly once."""
    require(len({moving,other,control}) == 3 and all(0 <= i < len(poses) for i in (moving,other,control)), 'Invalid labels')
    require(math.isfinite(rmean) and rmean >= 0, 'Invalid mean-radius bound')
    origin = finite_array(poses[other]['position'], (3,)); bound = 2*rmean+POLICY['distance_A']
    ranked = []
    for label, pose in enumerate(poses):
        if label in (moving,other): continue
        distance = math.hypot(*(finite_array(pose['position'], (3,))-origin))
        ranked.append((distance, label))
    ranked.sort()
    selected = [(d, i) for d,i in ranked[:nearest] if d <= bound]
    control_d = next(d for d,i in ranked if i == control)
    return [dict(anchor=control, control=True, distance_A=control_d, within_mean_bound=control_d <= bound,
                 selection_ambiguous=abs(control_d-bound) <= POLICY['distance_ambiguity_A'])]+[
        dict(anchor=i, control=False, distance_A=d, within_mean_bound=True,
             selection_ambiguous=abs(d-bound) <= POLICY['distance_ambiguity_A']) for d,i in selected if i != control]


def screen(first, second, counts, *, cap=500000, chunk=64, progress=lambda: None):
    """KD search is conservative; strict distance/angle checks define counts."""
    require(type(cap) is int and 0 <= cap <= POLICY['distance_pair_cap'] and type(chunk) is int and 1 <= chunk <= 64,
            'Invalid screen budget')
    p, q = first; pp, qq = second; n, m = len(p), len(pp)
    require(p.shape == (n,3) and pp.shape == (m,3) and q.shape == (n,4) and qq.shape == (m,4), 'Wrong screen arrays')
    require(all(np.isfinite(a).all() for a in (p,q,pp,qq)), 'Nonfinite screen arrays')
    counts.update(single_labels=n+m, decoded_means=n+m, mean_decode_failures=0,
        possible_interface_pairs=n*m, decoded_interface_pairs=n*m, distance_pairs=0,
        angle_rejected_pairs=0, candidate_pairs_before_cap=0, distance_ambiguous_pairs=0,
        angle_ambiguous_pairs=0, left_means_completed=0, query_chunks=0)
    tree = cKDTree(pp); cutoff = POLICY['distance_A']; angle = POLICY['angle_radians']
    for start in range(0,n,chunk):
        progress()
        neighbors = tree.query_ball_point(p[start:start+chunk], cutoff+POLICY['distance_ambiguity_A'], workers=1, return_sorted=True)
        counts['query_chunks'] += 1
        for offset, indices in enumerate(neighbors):
            k = start+offset; indices = np.asarray(indices, dtype=int)
            delta = pp[indices]-p[k]
            distance = np.hypot(np.hypot(delta[:,0],delta[:,1]),delta[:,2])
            counts['distance_ambiguous_pairs'] += int(np.count_nonzero(np.abs(distance-cutoff) <= POLICY['distance_ambiguity_A']))
            indices = indices[distance < cutoff]
            require(counts['distance_pairs']+len(indices) <= cap, 'Per-screen distance-pair cap exceeded')
            counts['distance_pairs'] += len(indices)
            # Same sign-independent SO(3) angle as the Rust screen.
            dots = np.sum(qq[indices]*q[k],axis=1)
            angles = 2*np.arccos(np.minimum(1.,np.abs(dots)))
            counts['angle_ambiguous_pairs'] += int(np.count_nonzero(np.abs(angles-angle) <= POLICY['angle_ambiguity_radians']))
            passing = int(np.count_nonzero(angles < angle))
            counts['candidate_pairs_before_cap'] += passing
            counts['angle_rejected_pairs'] += len(indices)-passing
            counts['left_means_completed'] += 1
        progress()
    counts['distance_rejected_pairs'] = n*m-counts['distance_pairs']
    counts['threshold_ambiguous'] = bool(counts['distance_ambiguous_pairs'] or counts['angle_ambiguous_pairs'])
    return counts


def validate_baseline(counts, saved):
    require(all(counts[key] == saved[key] for key in COUNT_KEYS), 'Independent original screen disagrees with saved Rust counts')


def make_plan(manifest_path, validation_path, *, root):
    """No mean decoding or candidate search occurs during preparation."""
    root = Path(root).resolve(); manifest_ref = reference(manifest_path); manifest = read(manifest_ref['path'])
    oldroot = Path(manifest_ref['path']).parent; oldplan_path = oldroot/'execution-plan.json'; oldplan = read(oldplan_path)
    require(len(oldplan['jobs']) == 1, 'Wrong prior diagnostic lifecycle')
    terminal = driver.completed_terminal(oldroot, oldplan, oldplan['jobs'][0]['id'])
    oldstatus = read(oldroot/'execution/summary.json')
    require(oldstatus['complete'] is True and oldstatus['passed'] is True, 'Prior controller incomplete')
    summary = read(terminal['path']); journal = Path(manifest['output'])/'attempts.jsonl'
    require(terminal['path'] == str(Path(manifest['output'])/'summary.json')
            and summary['schema'] == 'singleton-fusion-diagnostic-v1'
            and summary['manifest_sha256'] == manifest_ref['sha256'] and sha(journal) == summary['attempts_sha256'], 'Prior diagnostic identity differs')
    require(manifest['schema'] == 'singleton-fusion-diagnostic-manifest-v1' and manifest['context'] == 'whole_27_132'
            and manifest['members'] == [27,132] and manifest['anchor'] == 228
            and manifest['coordinate_frame'] == 'sphere_centered' and manifest['wall_center'] == [0.,0.,0.], 'Wrong fixed context/frame')
    require(manifest['oligomer']['pair_distance_A'] == 8. and manifest['oligomer']['pair_angle_degrees'] == 60., 'Screen thresholds changed')
    state_ids = ['source']+[f'prepared-{i}' for i in range(4)]
    require([s['id'] for s in manifest['states']] == state_ids and len(manifest['constructions']) == 10, 'Wrong original state inventory')
    expected = [dict(id=f'{s}-{i}', state=s, moving=i, neighbors=[j,228]) for s in state_ids for i,j in [(27,132),(132,27)]]
    require(manifest['constructions'] == expected, 'Wrong ten original constructions')
    files = dict(oldplan['files']); files[str(oldplan_path)] = sha(oldplan_path)
    for path in (oldroot/'preparation.json', oldroot/'execution/claim.json', oldroot/'execution/summary.json',
                 oldroot/'execution/status.json', journal, Path(terminal['path']), Path(manifest_ref['path'])):
        files[str(path.resolve())] = sha(path)
    jobdir = driver.job_directory(oldroot,0,oldplan['jobs'][0])
    for name in ('attempt.json','process.json','exit.json','success.json'):
        path = jobdir/name; files[str(path)] = sha(path)
    for key in ('config','shape','model'):
        ref = manifest[key]; require(summary['input_sha256'].get(ref['path']) == ref['sha256'], 'Missing prior bound input'); files[ref['path']] = ref['sha256']
    source = None
    for item in manifest['states']:
        ref = item['file']; require(item['poses_pointer'] == '/poses' and summary['input_sha256'].get(ref['path']) == ref['sha256'], 'Wrong bound state')
        files[ref['path']] = ref['sha256']; value = read(ref['path'])
        require(value['coordinate_frame'] == 'sphere_centered' and len(value['poses']) == 264, 'Wrong state/frame')
        poses = value['poses']
        for pose in poses: finite_array(pose['position'],(3,)); rotation(pose['orientation'])
        if source is None: source = poses
        else: require(all(pose == source[i] for i,pose in enumerate(poses) if i not in (27,132)), 'A fixed spectator changed')
    require(read(manifest['config']['path'])['initial_poses'] == source, 'Source state does not match Config')
    model = read(manifest['model']['path'])
    require(model['base_model']['shape_sha256'] == manifest['shape']['sha256']
            and len(model['base_model']['anchors']) == 1024 and model['reciprocal_components'] == [True]*1024, 'Atlas identity/branch inventory differs')
    baseline = {}
    with journal.open() as stream:
        for line in stream:
            require(len(line) <= 2**21, 'Oversized diagnostic metadata row')
            row = json.loads(line)
            if row['state'] == 'complete':
                ordinal = len(baseline); require(row['ordinal'] == ordinal and row['construction'] == expected[ordinal]
                    and row['error'] is None and row['diagnostics']['complete'] is True, 'Prior construction failed/reordered')
                baseline[row['construction']['id']] = {k:row['diagnostics'][k] for k in COUNT_KEYS}
    require(list(baseline) == [x['id'] for x in expected], 'Missing prior original screens')
    validation_ref = reference(validation_path); validation = read(validation_ref['path'])
    require(validation['complete'] is True and validation['passed'] is True
            and validation['source_before'] == validation['source_after'], 'Validation is not passed/source-stable')
    sources = {str(path):sha(path) for path in source_paths()}
    test = Path(__file__).with_name('test_audit_singleton_anchor_screens.py').resolve()
    for path,digest in dict(sources, **{str(test):sha(test)}).items():
        require(validation['source_before'].get(path) == digest, 'Stale/omitted validated source: '+path)
    files.update(sources); files[validation_ref['path']] = validation_ref['sha256']; files[str(test)] = sha(test)
    runtime_id = runtime(); require(runtime_id['optimization'] == 0, 'Unoptimized Python required'); files.update(runtime_id['files'])
    recheck(files)
    return dict(schema=SCHEMA, root=str(root), output=str(root/'analysis'), manifest=manifest_ref,
        prior_terminal=terminal, prior_journal=reference(journal), validation=validation_ref,
        input_sha256=files, source_sha256=sources, runtime=runtime_id, baseline=baseline,
        limits=LIMITS, policy=POLICY, scope=SCOPE)


def worker_argv(plan_path, digest, output):
    return [str(Path(sys.executable).absolute()), '-B', str(Path(__file__).resolve()),
            '--plan', str(plan_path), '--plan-sha256', digest, '--out', str(output)]


def prepare(manifest_path, validation_path, *, out):
    root = Path(out).resolve(); require(not root.exists(), 'Fresh output root required')
    plan = make_plan(manifest_path, validation_path, root=root)
    root.mkdir(); write(root/'preparation-claim.json', dict(schema=SCHEMA, pid=os.getpid()))
    try:
        path = root/'protocol.json'; write(path, plan); ref = reference(path)
        job = dict(id='context0-anchor-screens', population='whole_27_132', phase='geometry',
                   argv=worker_argv(path,ref['sha256'],plan['output']), **LIMITS,
                   terminal=dict(path=str(root/'analysis/summary.json'), success_contract='complete_and_passed'))
        files = dict(plan['input_sha256']); files[ref['path']] = ref['sha256']
        python = plan['runtime']['executable_argv']
        execution = dict(schema=driver.SCHEMA, root=str(root), maximum_workers=1, threads=1,
            files=files, jobs=[job], executable_resolutions={python:str(Path(python).resolve())},
            preparation_receipt=str(root/'preparation.json'), scope=SCOPE)
        write(root/'execution-plan.json',execution); recheck(files)
        receipt = dict(schema=SCHEMA, complete=True, launched=False, protocol=ref,
                       execution_plan=reference(root/'execution-plan.json'))
        write(root/'preparation.json',receipt); return receipt
    except BaseException as error:
        write(root/'preparation-failure.json',dict(schema=SCHEMA, complete=False, error=repr(error))); raise


def live_authority(plan_path, digest, plan):
    root = Path(plan['root']); require(plan_path == root/'protocol.json', 'Wrong plan location')
    path = root/'execution-plan.json'; execution = read(path); driver.verify_plan(path,execution,sha(path))
    require(len(execution['jobs']) == 1, 'One worker required'); job = execution['jobs'][0]
    require(job['argv'] == worker_argv(plan_path,digest,plan['output']) and all(job[k] == v for k,v in LIMITS.items())
            and job['terminal'] == dict(path=str(Path(plan['output'])/'summary.json'),success_contract='complete_and_passed'), 'Worker contract changed')
    expected = dict(plan['input_sha256']); expected[str(plan_path)] = digest
    require(execution['files'] == expected, 'Worker input closure differs')
    claim = read(root/'execution/claim.json'); status = read(root/'execution/status.json')
    require(claim['pid'] == os.getppid() and claim['birth_ticks'] == driver._birth(os.getppid())
            and claim['plan_sha256'] == sha(path) and claim['maximum_workers'] == claim['threads'] == 1
            and claim['retries'] == claim['replacements'] == 0, 'Missing actual controller authority')
    require(status['plan_sha256'] == claim['plan_sha256'] and status['failure'] is None
            and status['active'] == dict(ordinal=0,id=job['id'],population=job['population'],phase='geometry'), 'Worker is not active')


def execute(plan, emit, check, active):
    """Pure arithmetic worker core; tests supply tiny synthetic metadata."""
    manifest = read(plan['manifest']['path']); branches = decode_virtual_means(read(manifest['model']['path']), plan['policy']['branches'])
    rmean = max(math.hypot(*b['position']) for b in branches)
    atoms = read(manifest['shape']['path'])['atoms']
    core_bound = max(math.hypot(*finite_array(a['center'],(3,)))+float(a['radius']) for a in atoms)
    require(math.isfinite(core_bound) and core_bound > 0, 'Invalid core-bound metadata')
    states = {item['id']:read(item['file']['path'])['poses'] for item in manifest['states']}
    inventories = {}
    for item in manifest['constructions']:
        inventories[item['id']] = anchor_inventory(states[item['state']],item['moving'],item['neighbors'][0],manifest['anchor'],rmean)
    emit(dict(state='inventory', Rmean_A=rmean, anchor_bound_A=2*rmean+8, core_bound_A=core_bound, inventories=inventories))
    results = []
    # Controls first, then alternatives in frozen construction / center-distance order.
    jobs = [(item,entry) for control in (True,False) for item in manifest['constructions']
            for entry in inventories[item['id']] if entry['control'] == control]
    require(len(jobs) <= plan['policy']['max_screens'], 'Screen inventory exceeds budget')
    for item,entry in jobs:
        check(); active.clear(); active.update(construction=item, selection=entry, counts={})
        emit(dict(state='screen_begin',ordinal=len(results),**active))
        poses = states[item['state']]
        first = placed_means(branches,poses[item['neighbors'][0]])
        second = placed_means(branches,poses[entry['anchor']])
        def progress():
            check(); emit(dict(state='screen_progress',ordinal=len(results),**active))
        counts = screen(first,second,active['counts'],cap=plan['policy']['distance_pair_cap'],chunk=plan['policy']['query_chunk'],progress=progress)
        if entry['control']: validate_baseline(counts,plan['baseline'][item['id']])
        result = dict(construction=item,selection=entry,counts=dict(counts),baseline_agrees=True if entry['control'] else None)
        results.append(result); emit(dict(state='screen_complete',ordinal=len(results)-1,**result)); active.clear()
    return dict(screens=results, Rmean_A=rmean, anchor_bound_A=2*rmean+8, core_bound_A=core_bound,
                inventories=inventories, original_screens_checked=sum(r['selection']['control'] for r in results))


def run(plan_path, digest, out):
    path, out = Path(plan_path).resolve(), Path(out).resolve(); require(sha(path) == digest,'Plan changed')
    plan = read(path); require(plan['schema'] == SCHEMA and plan['limits'] == LIMITS and plan['policy'] == POLICY, 'Frozen policy differs')
    require(out == Path(plan['output']) and not out.exists(), 'Fresh declared output required')
    require(plan['source_sha256'] == {str(p):sha(p) for p in source_paths()} and plan['runtime'] == runtime(), 'Source/runtime differs')
    recheck(plan['input_sha256']); live_authority(path,digest,plan)
    out.mkdir(); started, wall = time.process_time(),time.monotonic(); active = {}; handlers = {}
    def check():
        require(time.process_time()-started <= LIMITS['cpu_limit_seconds'], 'CPU budget exhausted')
        require(time.monotonic()-wall <= LIMITS['wall_limit_seconds'], 'Wall budget exhausted')
    def stop(sig,_): raise RuntimeError('Worker received signal '+str(sig))
    with (out/'attempts.jsonl').open('x') as journal:
        def emit(row):
            journal.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n'); journal.flush(); os.fsync(journal.fileno())
        try:
            for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGXCPU): handlers[sig]=signal.signal(sig,stop)
            emit(dict(state='started',plan_sha256=digest,input_sha256=plan['input_sha256']))
            result = execute(plan,emit,check,active); check(); recheck(plan['input_sha256'])
            require(sha(path) == digest and plan['runtime'] == runtime(), 'Plan/runtime changed')
            summary = dict(schema=SCHEMA,complete=True,passed=True,plan_sha256=digest,scope=SCOPE,
                cpu_seconds=time.process_time()-started,wall_seconds=time.monotonic()-wall,
                input_sha256=plan['input_sha256'],source_sha256=plan['source_sha256'],runtime=plan['runtime'],
                attempts_sha256=sha(out/'attempts.jsonl'),fits=0,core_queries=0,native_queries=0,
                depletion_queries=0,pose_draws=0,**result)
            write(out/'summary.json',summary); return summary
        except BaseException as error:
            failure=dict(schema=SCHEMA,complete=False,passed=False,plan_sha256=digest,error=repr(error),active=active,
                         cpu_seconds=time.process_time()-started,wall_seconds=time.monotonic()-wall)
            emit(dict(state='failed',**failure)); failure['attempts_sha256']=sha(out/'attempts.jsonl')
            write(out/'failure.json',failure); raise
        finally:
            for sig,handler in handlers.items(): signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true'); parser.add_argument('--manifest'); parser.add_argument('--validation')
    parser.add_argument('--plan'); parser.add_argument('--plan-sha256'); parser.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.prepare:
        require(args.manifest and args.validation and not args.plan and not args.plan_sha256,'Preparation needs manifest and validation only')
        print(json.dumps(prepare(args.manifest,args.validation,out=args.out),indent=2))
    else:
        require(args.plan and args.plan_sha256 and not args.manifest and not args.validation,'Execution needs frozen plan only')
        run(args.plan,args.plan_sha256,args.out)


if __name__ == '__main__': main()
