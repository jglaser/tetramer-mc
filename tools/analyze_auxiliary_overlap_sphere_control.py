#!/usr/bin/env python3
"""Independent saved-latent hard-sphere threshold-control audit; no RNG draws.

Reconstructs all source rejection trials and staged uniform-edge draws, strict
hard/contact predicates, closed guidance counts, threshold support, both MH
rules and retained states. Eight pooled paired normal tests use Bonferroni .01;
the separately specified omitted-ratio control must increase close contact.
No physical bath, trajectory ESS, PRNG replay or exact floating-point measure
certificate is supplied by this audit.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

CORE = .05
EXCLUSION = .45
HALF_WIDTH = 1.
CAPS = dict(root=8, internal=8, joint=1)
IDENTITY = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
OBSERVABLES = ('separation', 'close_contact', 'overlap_count', 'root_w2')
ABS_TOLERANCE = 3e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ordered_sum(values):
    total = 0.
    for value in values:
        total += value
    return total


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name)
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, name)
    return value


def unit(values):
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'Nonfinite normal variate')
    length = math.sqrt(ordered_sum(v*v for v in values))
    require(length > 0 and math.isfinite(length), 'Invalid normal-vector norm')
    return [v/length for v in values]


def rotation(q):
    w, x, y, z = unit(q)
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matvec(matrix, vector):
    return [ordered_sum(a*b for a, b in zip(row, vector)) for row in matrix]


def quaternion_product(a, b):
    w, x, y, z = a
    s, u, v, t = b
    return unit([w*s-x*u-y*v-z*t, w*u+x*s+y*t-z*v,
                 w*v-x*t+y*s+z*u, w*t+x*v-y*u+z*s])


def sub(a, b):
    return [x-y for x, y in zip(a, b)]


def norm_squared(vector):
    return ordered_sum(x*x for x in vector)


def compose(a, b):
    delta = matvec(rotation(a['orientation']), b['position'])
    return dict(position=[x+y for x, y in zip(a['position'], delta)],
                orientation=quaternion_product(a['orientation'], b['orientation']))


def relative(a, b):
    q = unit(a['orientation'])
    return dict(position=matvec(transpose(rotation(q)), sub(b['position'], a['position'])),
                orientation=quaternion_product([q[0], -q[1], -q[2], -q[3]], b['orientation']))


def validate_pose(pose):
    require(set(pose) == {'position', 'orientation'}, 'Malformed pose')
    require(len(pose['position']) == 3 and len(pose['orientation']) == 4, 'Pose dimensions')
    require(all(type(x) in (int, float) and math.isfinite(x) for x in pose['position']+pose['orientation']), 'Nonfinite pose')
    require(abs(norm_squared(pose['orientation'])-1) <= 1e-8, 'Nonnormalized quaternion')


class Checks:
    def __init__(self):
        self.count = 0
        self.max_pose_error = 0.
        self.max_scalar_error = 0.

    def exact(self, actual, expected, name):
        self.count += 1
        require(actual == expected, name)

    def close(self, actual, expected, name):
        finite(actual, name); finite(expected, name)
        error = abs(actual-expected)
        self.max_scalar_error = max(self.max_scalar_error, error)
        self.count += 1
        require(error <= ABS_TOLERANCE, name + f': {actual} != {expected}')

    def pose(self, actual, expected, name):
        validate_pose(actual); validate_pose(expected)
        error = max(abs(x-y) for x, y in zip(actual['position'], expected['position']))
        qa, qb = actual['orientation'], expected['orientation']
        error = max(error, min(max(abs(x-y) for x, y in zip(qa, qb)),
                               max(abs(x+y) for x, y in zip(qa, qb))))
        self.count += 1
        self.max_pose_error = max(self.max_pose_error, error)
        require(error <= ABS_TOLERANCE, name + f': error {error}')


def grid_points():
    return [[.1*i, .1*j, .1*k] for i in range(-4, 5) for j in range(-4, 5) for k in range(-4, 5)
            if norm_squared([.1*i, .1*j, .1*k]) <= EXCLUSION**2]


class SphereOracle:
    def __init__(self, points):
        self.points = points
        require(all(len(p) == 3 and all(math.isfinite(v) for v in p) and norm_squared(p) <= EXCLUSION**2 for p in points), 'Invalid fixed guidance cloud')

    def internal(self, edge):
        distance2 = norm_squared(edge['position'])
        return dict(internal_core_overlap=distance2 < (2*CORE)**2,
                    internal_exclusion_contact=distance2 < (2*EXCLUSION)**2)

    def root(self, pose):
        return dict(spectator_core_collisions=[2] if norm_squared(pose['position']) < (2*CORE)**2 else [], wall_valid=True)

    def full(self, members):
        root, child = members
        pair = self.internal(dict(position=sub(child['position'], root['position'])))
        return dict(**pair, spectator_core_collisions=[self.root(root)['spectator_core_collisions'],
                    self.root(child)['spectator_core_collisions']], wall_valid=[True, True])

    def count_relative(self, edge):
        # Independently exploit the isotropic sphere: orientation cannot change
        # its geometric point membership. No tolerance changes the <= predicate.
        x, y, z = edge['position']; r2 = EXCLUSION**2
        return sum((a-x)**2+(b-y)**2+(c-z)**2 <= r2 for a, b, c in self.points)

    def count_world(self, members):
        root, child = members
        matrix = rotation(root['orientation'])
        offset = sub(root['position'], child['position'])
        r2 = EXCLUSION**2
        count = 0
        for point in self.points:
            vector = matvec(matrix, point)
            count += norm_squared([x+y for x, y in zip(vector, offset)]) <= r2
        return count


def internally_feasible(value):
    return not value['internal_core_overlap'] and value['internal_exclusion_contact']


def root_feasible(value):
    return not value['spectator_core_collisions'] and value['wall_valid']


def fully_feasible(value):
    return internally_feasible(value) and not any(value['spectator_core_collisions']) and all(value['wall_valid'])


def audit_frame(checks, oracle, frame, members, raw_relative, expected_raw_count):
    edges = [relative(IDENTITY, members[0]), relative(members[0], members[1])]
    for actual, expected in zip(frame['recovered_edges'], edges):
        checks.pose(actual, expected, 'Recovered edge')
    reconstructed = [compose(IDENTITY, frame['recovered_edges'][0]), None]
    reconstructed[1] = compose(reconstructed[0], frame['recovered_edges'][1])
    for actual, expected in zip(frame['reconstructed_members'], reconstructed):
        checks.pose(actual, expected, 'Reconstructed member')
    checks.exact(frame['reconstructed_feasibility'], oracle.full(frame['reconstructed_members']), 'Reconstructed full geometry')
    checks.exact(frame['reconstructed_root'], oracle.root(frame['reconstructed_members'][0]), 'Reconstructed root geometry')
    checks.exact(frame['internal_relative'], oracle.internal(frame['recovered_edges'][1]), 'Recovered internal geometry')
    checks.exact(frame['reconstructed_feasibility'], oracle.full(members), 'Full frame predicate disagreement')
    world_geometry = oracle.full(members)
    checks.exact(frame['internal_relative'], {key: world_geometry[key] for key in ('internal_core_overlap', 'internal_exclusion_contact')}, 'Internal frame predicate disagreement')
    checks.exact(frame['reconstructed_root'], oracle.root(members[0]), 'Root frame predicate disagreement')
    counts = dict(relative_count=oracle.count_relative(raw_relative), recovered_count=oracle.count_relative(frame['recovered_edges'][1]),
                  world_count=oracle.count_world(members), reconstructed_world_count=oracle.count_world(frame['reconstructed_members']))
    checks.exact(frame['guidance'], counts, 'Recorded frame counts')
    require(len(set(counts.values())) == 1, 'Guidance frame disagreement')
    checks.exact(counts['relative_count'], expected_raw_count, 'Raw/frame count disagreement')
    return counts['relative_count']


def uniform_trace(checks, record, old):
    checks.pose(record['old_relative_pose'], old, 'Raw old edge')
    checks.exact(record['branch'], 'uniform', 'Nonuniform branch in pure uniform control')
    checks.exact(record['null_reason'], None, 'Fatal raw draw cannot be rejected')
    trace = record['trace']
    require(set(trace) == {'branch_uniform', 'translation_uniforms', 'quaternion_normals'}, 'Unexpected uniform trace')
    require(0 <= finite(trace['branch_uniform'], 'Branch uniform') < 1, 'Invalid branch uniform')
    uniforms = trace['translation_uniforms']
    require(len(uniforms) == 3 and all(0 <= finite(x, 'Translation uniform') < 1 for x in uniforms), 'Invalid translation variates')
    require(len(trace['quaternion_normals']) == 4, 'Quaternion variate length')
    expected = dict(position=[(2*x-1)*HALF_WIDTH for x in uniforms], orientation=unit(trace['quaternion_normals']))
    checks.pose(record['proposed_relative_pose'], expected, 'Uniform latent reconstruction')
    return record['proposed_relative_pose']


def audit_outcome(checks, oracle, outcome, old, m):
    checks.exact(outcome['old'], old, 'Changed source endpoint')
    checks.exact(outcome['caps'], CAPS, 'Changed stage caps')
    checks.exact(outcome['order'], 'root_first', 'Changed stage order')
    checks.exact(outcome['members'], [0, 1], 'Changed selected labels')
    checks.exact(outcome['anchor_label'], 2, 'Changed fixed anchor')
    source = oracle.full(old)
    require(fully_feasible(source), 'Source outside channel domain')
    checks.exact(outcome['source_feasibility'], source, 'Wrong source predicates')
    old_edges = [relative(IDENTITY, old[0]), relative(old[0], old[1])]
    d = outcome['guidance']; old_count = oracle.count_relative(old_edges[1])
    checks.exact(d['point_count'], len(oracle.points), 'Wrong cloud size')
    checks.exact(d['m'], m, 'Wrong multiplicity')
    checks.exact(d['old_count'], old_count, 'Wrong source K')
    audit_frame(checks, oracle, outcome['source_frame'], old, old_edges[1], old_count)
    draws = d['integer_draws']; threshold = d['threshold']
    require(len(draws) == m and all(type(v) is int and 0 <= v <= old_count for v in draws), 'Threshold support or draw count')
    checks.exact(threshold, max(draws), 'Threshold is not the retained maximum')
    attempts = outcome['attempts']; require(len(attempts) == 1, 'J=1 must retain exactly one joint attempt')
    attempt = attempts[0]; checks.exact(attempt['index'], 1, 'Joint attempt index')
    roots, internals = attempt['root_draws'], attempt['internal_draws']
    require(1 <= len(roots) <= CAPS['root'], 'Root cap/stopping')
    root_pass = False
    for i, record in enumerate(roots):
        checks.exact(record['index'], i+1, 'Root draw index')
        edge = uniform_trace(checks, record['draw'], old_edges[0])
        expected_root = compose(IDENTITY, edge)
        checks.pose(record['world_pose'], expected_root, 'Root composition')
        predicate = oracle.root(record['world_pose']); checks.exact(record['feasibility'], predicate, 'Root predicate')
        root_pass = root_feasible(predicate)
        require(not root_pass or i == len(roots)-1, 'Continued after root success')
    measured = 4
    internal_pass = False
    if root_pass:
        require(1 <= len(internals) <= CAPS['internal'], 'Internal cap/stopping')
        for i, record in enumerate(internals):
            checks.exact(record['index'], i+1, 'Internal draw index')
            edge = uniform_trace(checks, record['draw'], old_edges[1])
            predicate = oracle.internal(edge); checks.exact(record['feasibility'], predicate, 'Internal predicate')
            internal_pass = internally_feasible(predicate)
            if internal_pass:
                count = oracle.count_relative(edge); measured += 1
                checks.exact(record.get('guidance_count'), count, 'Internal K')
                internal_pass = count >= threshold
            else:
                checks.exact(record.get('guidance_count'), None, 'Geometric failure acquired measured K')
            require(not internal_pass or i == len(internals)-1, 'Continued after internal success')
    else:
        require(len(roots) == CAPS['root'] and not internals, 'Root exhaustion must skip internal stage')
    candidate = outcome['candidate']
    if not root_pass or not internal_pass:
        if root_pass: require(len(internals) == CAPS['internal'], 'Unexhausted failed internal cap')
        checks.exact(attempt['status'], 'internal_cap_exhausted' if root_pass else 'root_cap_exhausted', 'Stage null status')
        for name in ('proposed', 'final_feasibility', 'frame'):
            checks.exact(attempt[name], None, 'Exhausted stage acquired endpoint')
        checks.exact(candidate, None, 'Exhausted cap acquired candidate')
    else:
        expected = [compose(IDENTITY, roots[-1]['draw']['proposed_relative_pose']), None]
        expected[1] = compose(expected[0], internals[-1]['draw']['proposed_relative_pose'])
        members = attempt['proposed']
        for actual, want in zip(members, expected): checks.pose(actual, want, 'Assembled endpoint')
        checks.exact(members[0], roots[-1]['world_pose'], 'Root must be copied identically')
        final = oracle.full(members); checks.exact(attempt['final_feasibility'], final, 'Final hard geometry')
        checks.exact(internals[-1]['feasibility'], {key: final[key] for key in ('internal_core_overlap', 'internal_exclusion_contact')}, 'Raw internal/world predicate disagreement')
        checks.exact(roots[-1]['feasibility'], oracle.root(members[0]), 'Raw root/world predicate disagreement')
        count = audit_frame(checks, oracle, attempt['frame'], members,
                            internals[-1]['draw']['proposed_relative_pose'], internals[-1]['guidance_count'])
        measured += 4
        require(count >= threshold, 'Endpoint outside threshold support')
        if fully_feasible(final):
            require(candidate is not None, 'Valid endpoint lost candidate')
            checks.exact(attempt['status'], 'candidate', 'Final candidate status')
            checks.exact([candidate['root'], candidate['child']], members, 'Changed candidate')
            checks.exact(d['new_count'], count, 'Wrong new K')
            auxiliary = m*(math.log1p(old_count)-math.log1p(count))
            checks.close(d['aux_log_correction'], auxiliary, 'Auxiliary ratio')
            log_uniform = -3*math.log(2)
            diagnostics = candidate['diagnostics']
            for side in ('old_edges', 'new_edges'):
                require(len(diagnostics[side]) == 2, 'Missing edge density')
                for density in diagnostics[side]:
                    checks.close(density['log_uniform'], log_uniform, 'Uniform density')
                    checks.close(density['log_full'], log_uniform, 'Full pure-uniform density')
                    require(density['log_learned'] == '-inf' or type(density['log_learned']) in (int, float) and math.isfinite(density['log_learned']), 'Invalid inactive learned density')
            for name in ('full_old_log_density', 'full_new_log_density'):
                checks.close(diagnostics[name], 2*log_uniform, 'Joint full density')
            for name in ('log_reverse_forward', 'selection_log_reverse_forward', 'log_tree_coordinate_jacobian'):
                checks.close(diagnostics[name], 0., 'Unexpected density/selection/Jacobian correction')
        else:
            checks.exact(attempt['status'], 'final_rejected', 'Final failure status')
            checks.exact(candidate, None, 'Hard-invalid candidate')
    checks.exact(outcome['status'], 'candidate' if candidate is not None else 'cap_exhausted', 'Outer status')
    checks.exact(d['count_queries'], measured, 'Count query bookkeeping')
    checks.exact(d['point_tests'], measured*len(oracle.points), 'Point query bookkeeping')
    if candidate is None:
        checks.exact(d['new_count'], None, 'Null acquired final K')
        checks.exact(d['aux_log_correction'], None, 'Null acquired correction')
    return dict(candidate=candidate, old_count=old_count, new_count=d['new_count'],
                auxiliary=d['aux_log_correction'], root_draws=len(roots), internal_draws=len(internals))


class Moments:
    def __init__(self): self.n=0; self.mean=0.; self.m2=0.
    def add(self, value):
        self.n += 1
        delta=value-self.mean; self.mean += delta/self.n; self.m2 += delta*(value-self.mean)
    def report(self, one_sided=False):
        require(self.n >= 2, 'Too few paired observations')
        se=math.sqrt(max(0.,self.m2)/(self.n-1)/self.n)
        z=self.mean/se if se else (0. if self.mean == 0 else math.copysign(math.inf,self.mean))
        p=.5*math.erfc(z/math.sqrt(2)) if one_sided else math.erfc(abs(z)/math.sqrt(2))
        return dict(n=self.n,mean=self.mean,se=se,z=z if math.isfinite(z) else ('+inf' if z>0 else '-inf'),p_normal=p)


def observables(oracle, members):
    edge=relative(members[0],members[1]);distance=math.sqrt(norm_squared(edge['position']))
    return dict(separation=distance,close_contact=float(distance<.6),overlap_count=float(oracle.count_relative(edge)),
                root_w2=unit(members[0]['orientation'])[0]**2)


def seed(population, slot, role):
    text=f'auxiliary-overlap-sphere-v1/6100300201/{population}/{slot}/{role}'
    return int(hashlib.sha256(text.encode()).hexdigest()[:16],16)


def audit_source(checks, oracle, trace, saved_source):
    require(1 <= len(trace) <= 1000, 'Missing or excessive source generation attempts')
    for index, record in enumerate(trace):
        checks.exact(record['status'],'complete','Incomplete source attempt')
        u=record['root_uniform']; radial=record['radial_uniform']
        require(len(u)==3 and all(0 <= finite(v,'Source root uniform') < 1 for v in u), 'Invalid source root uniforms')
        require(0 <= finite(radial,'Source radial uniform') < 1, 'Invalid source radial uniform')
        require(len(record['root_normals'])==4 and len(record['relative_normals'])==4 and len(record['direction_normals'])==3, 'Invalid source normal dimensions')
        root=dict(position=[2*v-1 for v in u],orientation=unit(record['root_normals']))
        radius=(.1**3+radial*(.9**3-.1**3))**(1/3)
        edge=dict(position=[v*radius for v in unit(record['direction_normals'])],orientation=unit(record['relative_normals']))
        expected=[compose(IDENTITY,root),None];expected[1]=compose(expected[0],edge)
        require(len(record['poses'])==2,'Missing source members')
        for pose,want in zip(record['poses'],expected): checks.pose(pose,want,'Source latent reconstruction')
        geometry=oracle.full(record['poses'])
        checks.exact(record['feasibility'],geometry,'Source generation hard predicates')
        checks.exact(fully_feasible(geometry),index==len(trace)-1,'Source rejected/accepted stopping')
    checks.exact(saved_source,trace[-1]['poses'],'Source differs from final generative trial')
    return len(trace)


def audit_row(checks, oracle, row, population, slot):
    checks.exact(row['population'],population,'Missing/reordered population')
    checks.exact(row['slot'],slot,'Missing/reordered slot')
    checks.exact(row['status'],'complete','Fatal outer cannot be treated as a null')
    checks.exact(row['source_seed'],seed(population,slot,'source'),'Source seed domain')
    source_tries=audit_source(checks,oracle,row['source_trace'],row['source'])
    require(len(row['arms'])==2 and [a['m'] for a in row['arms']]==[1,4],'Missing/reordered multiplicity arm')
    source_values=observables(oracle,row['source'])
    result=dict(population=population,slot=slot,source_tries=source_tries,source=source_values,arms=[])
    for arm in row['arms']:
        m=arm['m']
        for role in ('proposal','auxiliary','mh'):
            checks.exact(arm[role+'_seed'],seed(population,slot,role),'Seed domain '+role)
        for name in ('proposal_rng_after','auxiliary_rng_after'):
            require(type(arm[name]) is int and 0 <= arm[name] < 2**64,'Invalid RNG fingerprint')
        require(finite(arm['proposal_cpu_seconds'],'Proposal CPU')>=0,'Negative proposal CPU')
        audited=audit_outcome(checks,oracle,arm['outcome'],row['source'],m)
        logu=finite(arm['log_uniform'],'MH log uniform');require(logu<0,'MH uniform outside Open01')
        require(type(arm['accepted']) is bool,'Nonboolean acceptance')
        candidate=audited['candidate']; new=[candidate['root'],candidate['child']] if candidate else None
        correction=audited['auxiliary'] if candidate else None
        if correction is None:
            checks.exact(arm['complete_log_correction'],None,'Null has correction')
            expected_accept=False
        else:
            checks.close(arm['complete_log_correction'],correction,'Checked complete correction must include auxiliary')
            expected_accept=logu < min(0.,correction)
        checks.exact(arm['accepted'],expected_accept,'Corrected MH decision')
        expected_retained=new if expected_accept else row['source']
        checks.exact(arm['retained'],expected_retained,'Corrected retained endpoint')
        values=observables(oracle,arm['retained'])
        item=dict(m=m,candidate=candidate is not None,accepted=expected_accept,
                  old_count=audited['old_count'],new_count=audited['new_count'],auxiliary=correction,
                  root_draws=audited['root_draws'],internal_draws=audited['internal_draws'],
                  proposal_cpu_seconds=arm['proposal_cpu_seconds'],retained=values,
                  paired_delta={name:values[name]-source_values[name] for name in OBSERVABLES})
        if m==1:
            checks.exact(arm['wrong_without_auxiliary'],None,'Unexpected m1 negative control')
        else:
            wrong=arm['wrong_without_auxiliary'];wrong_accept=candidate is not None
            checks.exact(wrong['accepted'],wrong_accept,'Omitted-ratio MH on same candidate and U')
            checks.exact(wrong['retained'],new if wrong_accept else row['source'],'Wrong-control retained endpoint')
            wrong_values=observables(oracle,wrong['retained'])
            item['wrong_without_auxiliary']=dict(accepted=wrong_accept,retained=wrong_values,
                paired_delta={name:wrong_values[name]-source_values[name] for name in OBSERVABLES})
        result['arms'].append(item)
    first,second=row['arms']
    checks.exact(first['log_uniform'],second['log_uniform'],'Shared MH variate')
    a,b=first['outcome'],second['outcome']
    checks.exact(a['guidance']['integer_draws'][0],b['guidance']['integer_draws'][0],'Shared auxiliary prefix')
    checks.exact(a['attempts'][0]['root_draws'],b['attempts'][0]['root_draws'],'Shared root proposal prefix')
    left,right=a['attempts'][0]['internal_draws'],b['attempts'][0]['internal_draws']
    checks.exact(left[:min(len(left),len(right))],right[:min(len(left),len(right))],'Shared internal proposal prefix')
    return result


def validate_config(config):
    expected=dict(schema='auxiliary-overlap-sphere-control-v1',master_seed=6100300201,populations=4,
        draws_per_population=4096,m=[1,4],core_radius=CORE,exclusion_radius=EXCLUSION,
        uniform_half_width=HALF_WIDTH,activity=0.,caps=CAPS,order='root_first')
    for key,value in expected.items():require(config.get(key)==value,'Changed frozen allocation/model: '+key)


def summarize(results):
    pooled={(m,key):Moments() for m in (1,4) for key in OBSERVABLES}
    wrong={key:Moments() for key in OBSERVABLES}
    populations=defaultdict(lambda:{(m,key):Moments() for m in (1,4) for key in OBSERVABLES})
    source={key:Moments() for key in OBSERVABLES}
    candidates=[0,0];accepted=[0,0];wrong_accepted=0
    source_tries=0;raw_edges=0;cpu=0.
    for row in results:
        source_tries+=row['source_tries']
        for key in OBSERVABLES:source[key].add(row['source'][key])
        for index,arm in enumerate(row['arms']):
            m=arm['m'];candidates[index]+=arm['candidate'];accepted[index]+=arm['accepted']
            raw_edges+=arm['root_draws']+arm['internal_draws'];cpu+=arm['proposal_cpu_seconds']
            for key in OBSERVABLES:
                delta=arm['paired_delta'][key];pooled[m,key].add(delta);populations[row['population']][m,key].add(delta)
            if m==4:
                wrong_accepted+=arm['wrong_without_auxiliary']['accepted']
                for key in OBSERVABLES:wrong[key].add(arm['wrong_without_auxiliary']['paired_delta'][key])
    primary=[]
    for (m,key),stat in pooled.items():
        item=dict(m=m,observable=key,**stat.report());item['bonferroni_p']=min(1.,8*item['p_normal'])
        item['flag']=item['bonferroni_p']<.01;primary.append(item)
    negative=dict(observable='close_contact',**wrong['close_contact'].report(one_sided=True))
    negative['passed']=negative['mean']>0 and negative['p_normal']<.01
    return dict(independent_sources=len(results),corrected_decisions=2*len(results),candidate_counts=candidates,
        accepted_counts=accepted,negative_control_accepted=wrong_accepted,source_generation_attempts=source_tries,
        raw_edge_draws=raw_edges,proposal_cpu_seconds=cpu,
        primary_tests=primary,primary_family_size=8,primary_family_alpha=.01,
        primary_family_passed=not any(r['flag'] for r in primary),negative_control=negative,
        negative_control_secondary={key:stat.report() for key,stat in wrong.items()},
        source_observables={key:{name:value for name,value in stat.report().items() if name in ('n','mean','se')} for key,stat in source.items()},
        populations=[dict(population=p,descriptive_only=True,paired_tests=[dict(m=m,observable=key,**stat.report()) for (m,key),stat in entries.items()]) for p,entries in sorted(populations.items())])


def audit(run, prelaunch):
    started=time.process_time();checks=Checks()
    result=dict(schema='auxiliary-overlap-sphere-audit-v1',complete=False,passed=False,failures=[],rows=[],
        tolerance=dict(pose_absolute=ABS_TOLERANCE,scalar_absolute=ABS_TOLERANCE,strict_integer_and_geometry=True),
        limitations=['Paired normal tests are finite-sample diagnostics, not a proof of stationarity.',
          'Four populations are independent source panels, not sequential trajectories; no ESS or protein claim.',
          'Source and uniform-edge latent values are reconstructed; the PRNG and normal-generator implementation are not replayed.',
          'Inactive learned-component scores at uniform_probability=1 are checked for valid log values but do not enter full F.',
          'Hard/closed-point sphere predicates are independently recomputed without changing thresholds; no general FP measure theorem.'])
    try:
        frozen=read(prelaunch);frozen_files=frozen['files']
        require(frozen_files,'Empty prelaunch closure')
        for path,digest in frozen_files.items():require(sha(path)==digest,'Changed prelaunch file: '+path)
        require(sha(Path(__file__).resolve()) in frozen_files.values(),'Current analyzer missing from frozen closure')
        names=('config.json','binding.json','example.rs','source-bundle.json','cloud.json','attempts.jsonl','terminal.json')
        hashes={name:sha(run/name) for name in names};result['input_hashes']=hashes
        result['prelaunch']=dict(path=str(Path(prelaunch).resolve()),sha256=sha(prelaunch))
        config=read(run/'config.json');validate_config(config)
        binding=read(run/'binding.json')
        for name,key in [('config.json','config_sha256'),('example.rs','source_sha256'),('source-bundle.json','source_bundle_sha256')]:
            require(hashes[name]==binding[key],'Execution binding mismatch: '+name)
            require(hashes[name] in frozen_files.values(),'Execution source/config missing from prelaunch: '+name)
        require(binding['executable_sha256'] in frozen_files.values(),'Execution binary absent from prelaunch')
        points=read(run/'cloud.json');checks.exact(points,grid_points(),'Changed fixed point-mass cloud')
        oracle=SphereOracle(points)
        terminal=read(run/'terminal.json')
        require(terminal['summary']['complete'],'Incomplete/fatal campaign')
        require(terminal['attempts_sha256']==hashes['attempts.jsonl'],'Terminal ledger hash mismatch')
        with (run/'attempts.jsonl').open() as source:
            for index,line in enumerate(source):
                require(index<4*4096,'Excess source rows')
                row=json.loads(line);result['rows'].append(audit_row(checks,oracle,row,index//4096,index%4096))
        require(len(result['rows'])==4*4096,'Lost independent sources or nulls')
        summary=summarize(result['rows'])
        expected=terminal['summary']['result']
        for key in ('independent_sources','corrected_decisions','candidate_counts','accepted_counts','negative_control_accepted'):
            checks.exact(expected[key],summary[key],'Terminal count '+key)
        checks.exact(expected['new_physical_clouds'],0,'Unexpected physical bath')
        summary['process_cpu_seconds']=finite(terminal['cpu_seconds'],'Process CPU')
        require(summary['process_cpu_seconds']>=summary['proposal_cpu_seconds']-1e-6,'Proposal CPU exceeds process CPU')
        result['summary']=summary;result['complete']=True;result['arithmetic_and_trace_passed']=True
        result['passed']=summary['primary_family_passed'] and summary['negative_control']['passed']
        if not summary['primary_family_passed']:result['failures'].append('Prespecified corrected stationarity family flagged')
        if not summary['negative_control']['passed']:result['failures'].append('Prespecified wrong-ratio positive-shift control not detected')
    except Exception as error:
        result['failures'].append(f'{type(error).__name__}: {error}')
        result['failed_after_complete_rows']=len(result['rows'])
    result['checks']=checks.count;result['max_pose_error']=checks.max_pose_error;result['max_scalar_error']=checks.max_scalar_error
    result['analyzer_cpu_seconds']=time.process_time()-started
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prelaunch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.run,args.prelaunch)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('complete','passed','checks','failures','analyzer_cpu_seconds')}))
    raise SystemExit(0 if result['complete'] and result['passed'] else 1)
