#!/usr/bin/env python3
"""Audit a frozen, complete map-center query list without drawing proposals.

Use --config PATH with the schema illustrated by
results/dimer-mean-clearance-20261002/config.json. Every input and this method
must be hashed before invocation; output must name a new directory. Preparation
must enumerate all active virtual branches at latent zero using an authenticated
map decoder. This tool only measures the supplied centers: no native labels,
spectators, random queries, fitting, proposal success or physical acceptance.

The original dated method remains immutable. A new invocation must bind this
tool's own hash and a new output directory, never overwrite a historical config.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False);stream.write('\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def timeout(signum, frame):
    raise TimeoutError('Frozen wall-time limit reached')


def minimum_gap(points,centers,radii,groups):
    best=None
    for radius,indices,tree in groups:
        distances,nearest=tree.query(points,k=1,eps=0,workers=1)
        gaps=distances-radii-radius;i=int(np.argmin(gaps));j=int(indices[nearest[i]])
        distance=float(np.linalg.norm(points[i]-centers[j]));gap=distance-float(radii[i])-radius
        if best is None or gap < best['clearance_A']:
            best=dict(clearance_A=gap,moving_atom=i,fixed_atom=j,distance_A=distance,
                      moving_radius_A=float(radii[i]),fixed_radius_A=radius)
    return best


def validate_queries(queries, config):
    require(config['scientific_threads'] == 1, 'Only one scientific worker is supported')
    require(isinstance(config['queries'], int) and config['queries'] > 0, 'Invalid allocation')
    require(len(queries) == config['queries'], 'Wrong fixed allocation')
    expected = {a['name']: a['virtual_branches'] for a in config['atlases']}
    require(len(expected) == len(config['atlases']), 'Duplicate atlas')
    require(sum(expected.values()) == len(queries), 'Atlas allocation mismatch')
    branches = {name: [] for name in expected}
    weights = {name: [] for name in expected}
    for index, query in enumerate(queries):
        require(query['query'] == index, 'Lost/reordered query')
        require(query['atlas'] in expected, 'Undeclared atlas')
        require(query['latent'] == [0.] * 6, 'Query is not a map center')
        branches[query['atlas']].append(query['branch'])
        weight = query['branch_weight']
        require(math.isfinite(weight) and weight > 0, 'Invalid branch weight')
        weights[query['atlas']].append(weight)
        pose = query['relative_pose']
        position = np.asarray(pose['position'], dtype=float)
        orientation = np.asarray(pose['orientation'], dtype=float)
        require(position.shape == (3,) and orientation.shape == (4,), 'Invalid pose dimensions')
        require(np.isfinite(position).all() and np.isfinite(orientation).all(), 'Nonfinite pose')
        require(abs(float(np.dot(orientation, orientation)) - 1.) <= 1e-10, 'Nonunit orientation')
    for name, count in expected.items():
        require(branches[name] == list(range(count)), 'Incomplete/reordered atlas branches')
        require(abs(math.fsum(weights[name]) - 1.) <= 1e-12, 'Unnormalized branch weights')


def main(config_path):
    config=read(config_path)
    for record in config['files'].values():
        require(sha(record['path']) == record['sha256'], 'Changed frozen input: '+record['path'])
    require(sha(__file__) == config['method_sha256'], 'Changed audit method')
    out=Path(config['output']);out.mkdir()
    stat=Path('/proc/self/stat').read_text()
    save(out/'handle.json',dict(pid=os.getpid(),start_ticks=int(stat[stat.rindex(')')+2:].split()[19]),
        pid_namespace=os.readlink('/proc/self/ns/pid'),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        pid_scope='namespace PID only; not a host PID claim',config_sha256=sha(config_path),scientific_threads=1))
    resource.setrlimit(resource.RLIMIT_CPU,(config['maximum_cpu_seconds'],config['maximum_cpu_seconds']))
    signal.signal(signal.SIGALRM,timeout);signal.alarm(config['maximum_wall_seconds'])
    cpu,clock=time.process_time(),time.monotonic();rows=[]
    try:
        shape=read(config['files']['shape']['path'])
        centers=np.asarray([a['center'] for a in shape['atoms']],float)
        radii=np.asarray([a['radius'] for a in shape['atoms']],float)
        require(centers.ndim == 2 and centers.shape == (len(radii), 3) and len(radii) > 0,
                'Invalid atom dimensions')
        require(np.isfinite(centers).all() and np.isfinite(radii).all() and (radii>0).all(),'Invalid atoms')
        groups=[(float(r),np.flatnonzero(radii==r),cKDTree(centers[radii==r])) for r in np.unique(radii)]
        queries=[json.loads(line) for line in Path(config['files']['queries']['path']).read_text().splitlines()]
        validate_queries(queries, config)
        with (out/'queries.jsonl').open('x') as ledger:
            for index,q in enumerate(queries):
                require(q['query']==index,'Lost/reordered query')
                start=time.process_time();abort=False
                try:
                    p=q['relative_pose'];rotation=Rotation.from_quat(np.asarray(p['orientation'])[[1,2,3,0]]).as_matrix()
                    points=centers@rotation.T+p['position'];best=minimum_gap(points,centers,radii,groups)
                    row=dict(**q,status='complete',**best,core_overlap=best['clearance_A']<0,
                             exclusion_contact=best['clearance_A']<2*config['depletant_radius_A'],
                             threshold_near=abs(best['clearance_A'])<=config['threshold_near_A'],
                             cpu_seconds=time.process_time()-start)
                except Exception as error:
                    row=dict(**q,status='error',error=f'{type(error).__name__}: {error}',cpu_seconds=time.process_time()-start)
                    abort=isinstance(error,TimeoutError)
                ledger.write(json.dumps(row,allow_nan=False)+'\n');ledger.flush();rows.append(row)
                if abort:raise TimeoutError('Frozen wall-time limit reached; failed query retained')
        complete=[r for r in rows if r['status']=='complete'];errors=[r for r in rows if r['status']=='error']
        summary=[];pairs=[]
        for atlas in config['atlases']:
            selected=[r for r in complete if r['atlas']==atlas['name']]
            values=np.asarray([r['clearance_A'] for r in selected]);weights=[r['branch_weight'] for r in selected]
            summary.append(dict(atlas=atlas['name'],attempted=sum(r['atlas']==atlas['name'] for r in rows),
                completed=len(selected),colliding_centers=sum(r['core_overlap'] for r in selected),
                hard_valid_centers=sum(not r['core_overlap'] for r in selected),
                threshold_near_centers=sum(r['threshold_near'] for r in selected),
                normalized_weight_of_colliding_center_branches=math.fsum(r['branch_weight'] for r in selected if r['core_overlap']),
                normalized_weight_of_hard_valid_center_branches=math.fsum(r['branch_weight'] for r in selected if not r['core_overlap']),
                normalized_weight_of_threshold_near_center_branches=math.fsum(r['branch_weight'] for r in selected if r['threshold_near']),
                complete_branch_weight=math.fsum(weights),
                clearance_quantiles_A=dict(zip(['min','p05','median','p95','max'],np.quantile(values,[0,.05,.5,.95,1]).tolist())) if len(values) else None))
            by_component=collections.defaultdict(dict)
            for r in selected:by_component[r['component']][r['inverted']]=r
            for component,branches in by_component.items():
                if len(branches)==2:
                    a,b=branches[False],branches[True]
                    pairs.append(dict(atlas=atlas['name'],component=component,queries=[a['query'],b['query']],
                        absolute_clearance_difference_A=abs(a['clearance_A']-b['clearance_A']),
                        same_core_predicate=a['core_overlap']==b['core_overlap'],
                        both_within_threshold_band=a['threshold_near'] and b['threshold_near']))
        bad_pairs=[p for p in pairs if p['absolute_clearance_difference_A']>config['reciprocal_tolerance_A']
                   or (not p['same_core_predicate'] and not p['both_within_threshold_band'])]
        report=dict(complete=len(rows)==len(queries),passed=not errors and not bad_pairs,
            config_sha256=sha(config_path),queries_sha256=config['files']['queries']['sha256'],
            ledger_sha256=sha(out/'queries.jsonl'),method_sha256=sha(__file__),summary=summary,
            errors=errors,reciprocal_pairs=pairs,reciprocal_failures=bad_pairs,
            scope='Every virtual branch at standardized latent zero. Center labels and their normalized weights are descriptive, NOT Gaussian collision probabilities, proposal success, physical acceptance or assembly.',
            geometry='For each moving atom and each fixed-atom radius class, exact KD nearest-center search minimizes surface gap within that class; then minimize over all atoms/classes. No fixed spectators, native labels or stochastic queries.',
            cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-clock)
        save(out/'analysis.json',report)
        terminal=dict(complete=report['complete'],passed=report['passed'],analysis_sha256=sha(out/'analysis.json'),
                      ledger_sha256=report['ledger_sha256'],queries=len(rows),cpu_seconds=report['cpu_seconds'],wall_seconds=report['wall_seconds'])
    except Exception as error:
        terminal=dict(complete=False,passed=False,error=f'{type(error).__name__}: {error}',queries=len(rows),
                      cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-clock)
    finally:
        signal.alarm(0)
    save(out/'terminal.json',terminal);print(json.dumps(terminal),flush=True)
    return 0 if terminal['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    raise SystemExit(main(parser.parse_args().config))
