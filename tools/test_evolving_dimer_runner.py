#!/usr/bin/env python3
"""Authenticated sphere-only CLI checks for the evolving conditional runner.

This executes the actual release example, never the production executable and
never protein geometry. It checks restart/rejected-state behavior and independently
reconstructs densities, sphere predicates and auxiliary counts. It is not an
equilibration, stationarity, protein-geometry, or assembly test.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from analyze_capped_dimer_probe import Checks, decode_edge
from dimer_destination_density import DimerDestinationDensity, relative_pose, compose_pose, pose_arrays
import prepare_evolving_dimer_benchmark as prep

ROOT = Path(__file__).resolve().parents[1]
RADIUS, RD, ACTIVITY = .45, .3, .5


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write('\n')


def replace(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def dispatch_control(base, scenario='nonzero'):
    """Execute the real dispatcher with short fake jobs, never scientific jobs."""
    require(scenario in ('nonzero', 'launch_log_error', 'malformed_terminal'), 'Unknown dispatch scenario')
    base.mkdir()
    (base/'common').mkdir()
    output = base/'execution'
    program = base/'common/evolving_dimer_benchmark'
    program.write_text('#!'+sys.executable+'\n'+'''import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--job', type=int, required=True)
p.add_argument('--config', required=True)
args, _=p.parse_known_args()
config=json.loads(Path(args.config).read_text())
root=Path(config['output'])
root.mkdir(exist_ok=True)
out=root/f'job-{args.job:03}'
out.mkdir()
(out/'started.json').write_text(json.dumps({'pid':os.getpid(),'time':time.time()}))
scenario=config['synthetic_scenario']
if args.job == 0:
    time.sleep(.2)
    if scenario == 'nonzero':
        raise SystemExit(17)
    if scenario == 'malformed_terminal':
        (out/'terminal.json').write_text('Deliberately invalid synthetic JSON.')
        raise SystemExit(0)
    if scenario == 'launch_log_error':
        (Path(args.config).parent/'dispatch/job-002.log').write_text('Preserve synthetic preexisting log.\\n')
else:
    time.sleep(8. if scenario == 'launch_log_error' else 4.)
(out/'terminal.json').write_text(json.dumps({'complete':True,'synthetic':True,'finished':time.time()}))
''')
    program.chmod(0o755)
    count = 4 if scenario == 'launch_log_error' else 3
    write(base/'config.json', dict(jobs=[dict(id=i) for i in range(count)], output=str(output),
                                  synthetic_scenario=scenario))
    write(base/'prepared.json', dict(complete=True, synthetic=True))
    write(base/'binding.json', dict(synthetic=True))
    write(base/'run-binding.json', dict(config_sha256=sha(base/'config.json'),
        prepared_manifest=record(base/'prepared.json'), prelaunch_binding=record(base/'binding.json')))
    script = ROOT/'tools/run_evolving_dimer_benchmark.py'
    inputs = {str(p): sha(p) for p in [program, base/'config.json', base/'run-binding.json',
                                      base/'prepared.json', base/'binding.json', script]}
    write(base/'review.json', dict(complete=True, passed=True, input_sha256=inputs))
    command = [sys.executable, str(script), '--base', str(base), '--review', str(base/'review.json')]
    closure_results = []
    for index, omitted in enumerate([program, base/'config.json', base/'run-binding.json', base/'binding.json', script]):
        incomplete = {key: value for key, value in inputs.items() if key != str(omitted)}
        path = base/f'incomplete-review-{index}.json'
        write(path, dict(complete=True, passed=True, input_sha256=incomplete))
        cmd = [sys.executable, str(script), '--base', str(base), '--review', str(path), '--workers', '2']
        refused = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        receipt = dict(command=cmd, omitted=str(omitted), returncode=refused.returncode,
                       stdout=refused.stdout, stderr=refused.stderr)
        write(base/f'refused-review-{index}.json', receipt)
        require(refused.returncode != 0 and 'omits an execution closure input' in refused.stderr
                and not (base/'dispatch').exists(), 'Dispatcher accepted incomplete review closure')
        closure_results.append(str(omitted))
    refused = subprocess.run(command+['--workers', '4'], capture_output=True, text=True, timeout=20)
    write(base/'refused-four-workers.json', dict(command=command+['--workers', '4'],
        returncode=refused.returncode, stdout=refused.stdout, stderr=refused.stderr))
    require(refused.returncode != 0 and 'At most three new physical workers' in refused.stderr
            and not (base/'dispatch').exists(), 'Dispatcher accepted excess workers')
    launched = subprocess.run(command+['--workers', '2'], capture_output=True, text=True, timeout=20)
    write(base/'launch-call.json', dict(command=command+['--workers', '2'], returncode=launched.returncode,
                                     stdout=launched.stdout, stderr=launched.stderr))
    require(launched.returncode == 0, 'Synthetic dispatcher failed to launch')
    deadline = time.monotonic()+25
    observations = []
    while time.monotonic() < deadline:
        status = base/'dispatch/status.json'
        if status.exists():
            value = read(status)
            if not observations or value != observations[-1]: observations.append(value)
            if value['complete']: break
        time.sleep(.05)
    write(base/'observed-statuses.json', observations)
    require(observations and observations[-1]['complete'], 'Synthetic dispatcher did not drain')
    final = observations[-1]
    require(not final['passed'] and not final['active'], 'Failed job wrongly counted as successful')
    unstarted = 3 if scenario == 'launch_log_error' else 2
    require([x['id'] for x in final['unstarted']] == [unstarted]
            and not (output/f'job-{unstarted:03}').exists(), 'Queued job launched after failure or replaced it')
    done = {x['job']['id']: x for x in final['completed']}
    require(done[1]['returncode'] == 0 and done[1]['success'], 'In-flight success was not drained')
    if scenario == 'launch_log_error':
        require(set(done) == {0, 1, 2} and done[0]['success'] and not done[2]['success']
                and done[2]['pid'] is None and done[2]['returncode'] is None
                and 'FileExistsError' in done[2]['launch_failure'] and not (output/'job-002').exists(),
                'Launch exception failed to drain or entered simulation')
        require((base/'dispatch/job-002.log').read_text() == 'Preserve synthetic preexisting log.\n',
                'Log-open failure overwrote the previous receipt')
    else:
        require(set(done) == {0, 1} and not done[0]['success'], 'Failed job missing from completion ledger')
        if scenario == 'nonzero': require(done[0]['returncode'] == 17, 'Nonzero return code lost')
        else:
            require(done[0]['returncode'] == 0 and 'JSONDecodeError' in done[0]['terminal_error'],
                    'Malformed terminal not reported as failure')
    require(any(s['failure_draining'] and len(s['active']) == 1 for s in observations),
            'Failure drain state not observed')
    require(max(len(s['active']) for s in observations) <= 2,
            'Dispatcher exceeded allocated active worker count')
    return dict(passed=True, scenario=scenario, synthetic_jobs_started=2, unstarted_jobs=[unstarted],
                maximum_observed_active=max(len(s['active']) for s in observations),
                failure_drained=True, excess_workers_refused=True,
                incomplete_review_closures_refused=closure_results, source=record(script))


def scientific(value):
    """Remove timing and job identity only; retain every physical variate/count."""
    if isinstance(value, list):
        return [scientific(v) for v in value]
    if isinstance(value, dict):
        return {k: scientific(v) for k, v in value.items()
                if 'cpu_seconds' not in k and k not in ('job',)}
    return value


def fixture(base, bundle):
    base.mkdir()
    (base/'common').mkdir()
    pose = lambda x: dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.])
    state = [pose(0.), pose(1.2), pose(-1.2)]
    files = dict(shape=dict(name='CLI validation sphere', volume=4*math.pi*RADIUS**3/3,
                           atoms=[dict(center=[0., 0., 0.], radius=RADIUS)]),
                 source_config=dict(initial_poses=state), source_frame=dict(poses=state),
                 panel=dict(synthetic=True, native_labels=False),
                 patch_map=dict(synthetic=True, not_used_by_runner=True))
    inputs = {}
    for name, contents in files.items():
        path = base/'common'/f'{name}.json'
        write(path, contents)
        inputs[name] = record(path)
    model = dict(schema='weighted-pose-mixture-v1', coordinate_convention='anchor-body-relative',
        shape_sha256=inputs['shape']['sha256'], angular_length=1., weights=[1.],
        anchors=[dict(position=[1.2, 0., 0.], rotation=np.eye(3).tolist())],
        means=[[0.]*6], covariances=[np.diag([.04]*3+[.3]*3).tolist()])
    write(base/'common/atlas.json', model)
    inputs['atlas'] = record(base/'common/atlas.json')
    allocation = dict(contexts=1, streams=1, warmup_blocks=0, production_blocks=16,
        local_member_order=[0, 1, 0, 1], prep_outer_cap=128,
        synthetic_validation=True, no_protein_sampling=True)
    write(base/'common/scientific-allocation.json', allocation)
    write(base/'protocol.json', dict(schema='evolving-cli-sphere-control-v1',
        allocation=allocation, sources=inputs,
        controls=['local', 'unguided', 'm4', 'same-family resume', 'prepared start',
                  'checkpoint state/counters/row count/tail tampering', 'receipt preservation']))
    context = dict(name='sphere_chain', root=0, child=1, anchor=2)
    config = prep.configuration(base, inputs, [context],
        {k: v['sha256'] for k, v in bundle['files'].items()})
    config['allocation'] = allocation
    config['jobs'] = [dict(id=i, context_index=0, initialization='source', stream=0,
        arm=arm) for i, arm in enumerate(['local', 'unguided']+['m4']*6)]
    config['jobs'].append(dict(id=8, context_index=0, initialization='proposal_prepared',
                               stream=0, arm='m4'))
    config['physical'] = dict(depletant_radius=RD, activity=ACTIVITY,
                              lambda_ratio=64., wall_radius=8.)
    config['local'].update(translation_std_A=.12, rotation_std_degrees=10.)
    config['factorized'].update(uniform_half_width=2., root_cap=16, internal_cap=16)
    config['preparation'].update(minimum_max_center_displacement_A=.15,
                                 minimum_max_body_orientation_degrees=10., outer_cap=128)
    config['cloud'].update(raw_count=512, banks=2)
    config['envelope'] = dict(max_cells=63, max_depth=6, min_width=0.)
    config['limits'] = dict(raw_per_leg=1_000_000, raw_per_outer=2_000_000,
        raw_campaign=20_000_000, retained_per_leg=1_000_000, retained_per_outer=2_000_000,
        retained_campaign=20_000_000, cpu_seconds=120.)
    write(base/'config.json', config)
    return config


class Validator:
    def __init__(self, base, binary, source_bundle):
        self.base, self.binary = base, binary
        self.commands = []
        self.checks = Checks()
        bundle = read(source_bundle)
        for name, entry in bundle['files'].items():
            require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'],
                    'Corrupt embedded source: '+name)
            require(sha(ROOT/name) == entry['sha256'], 'Stale compiled source: '+name)
        self.plan = fixture(base/'toy', bundle)
        self.toy = base/'toy'
        self.config = self.toy/'config.json'
        self.pre = self.toy/'prelaunch.json'
        write(self.pre, dict(config_sha256=sha(self.config),
            protocol_sha256=sha(self.toy/'protocol.json'),
            example_source_sha256=sha(ROOT/'examples/evolving_dimer_benchmark.rs'),
            compiled_source_bundle_sha256=sha(source_bundle), executable_sha256=sha(binary)))
        self.binding = self.pre
        self.model = DimerDestinationDensity(read(self.plan['atlas']['path']))
        self.anchor = read(self.plan['source_frame']['path'])['poses'][2]
        self.counters = Counter()

    def call(self, mode, job=None, stop=None, resume=None, succeeds=True):
        command = [str(self.binary), '--config', str(self.config), '--binding', str(self.binding),
                   '--mode', mode]
        if job is not None: command += ['--job', str(job)]
        if stop is not None: command += ['--stop-after-block', str(stop)]
        if resume is not None: command += ['--resume', str(resume)]
        before = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        log = dict(command=command, returncode=result.returncode, stdout=result.stdout,
                   stderr=result.stderr, wall_seconds=time.monotonic()-before,
                   expected_success=succeeds)
        self.commands.append(log)
        write(self.base/f'command-{len(self.commands):02}.json', log)
        require((result.returncode == 0) == succeeds,
                'Unexpected CLI exit: '+json.dumps(log))
        return log

    def prepare(self):
        self.call('prepare-starts')
        manifest = self.toy/'prepared/manifest.json'
        m = read(manifest)
        require(m['complete'] and m['passed'] and m['all_attempts_retained'], 'Incomplete preparation')
        for f in m['files']:
            require(sha(f['path']) == f['sha256'], 'Unbound preparation output')
        require(len(m['cloud_banks']) == 2 and len(m['alternative_starts']) == 1,
                'Wrong synthetic preparation allocation')
        start = m['alternative_starts'][0]
        rows = [json.loads(v) for v in Path(start['ledger']['path']).read_text().splitlines()]
        require(len(rows) == start['attempts'] and rows[-1]['selected']
                and not any(r['selected'] for r in rows[:-1]), 'Unretained start attempts')
        self.binding = self.toy/'run-binding.json'
        write(self.binding, dict(config_sha256=sha(self.config), prelaunch_binding=record(self.pre),
                                 prepared_manifest=record(manifest)))
        self.manifest = m

    def path(self, job):
        return self.toy/'execution'/f'job-{job:03}'

    def close(self, actual, expected, what):
        if isinstance(actual, str) and actual == '-inf': actual = -math.inf
        self.checks.close(actual, expected, what, 'CLI synthetic sphere')
        require(actual == expected or (math.isfinite(actual) and math.isfinite(expected)
                and abs(actual-expected) <= 2e-7+2e-10*max(abs(actual), abs(expected))), what)

    def hard_valid(self, poses):
        points = [np.array(p['position']) for p in poses]+[np.array(self.anchor['position'])]
        return (all(np.linalg.norm(p) <= 8.-RADIUS for p in points[:2])
                and all(np.linalg.norm(points[i]-points[j]) >= 2*RADIUS
                        for i in range(3) for j in range(i)))

    def contact(self, poses):
        return np.linalg.norm(np.array(poses[0]['position'])-poses[1]['position']) < 2*(RADIUS+RD)

    def cloud(self, initialization):
        bank = next(b for b in self.manifest['cloud_banks'] if b['initialization'] == initialization)
        meta = read(bank['metadata']['path'])
        raw = np.fromfile(bank['raw']['path'], dtype='<f8').reshape(-1, 3)
        require(raw.shape == (512, 3) and ((0 <= raw) & (raw < 1)).all(), 'Invalid raw guidance points')
        all_points = np.array(meta['low'])+(np.array(meta['high'])-meta['low'])*raw
        kept = np.where(np.sum(all_points**2, axis=1) <= (RADIUS+RD)**2)[0].tolist()
        require(kept == meta['kept_indices'], 'Independent root-sphere thinning differs')
        return all_points[kept]

    def count(self, points, relative):
        return int(np.sum(np.sum((points-np.array(relative['position']))**2, axis=1)
                             <= (RADIUS+RD)**2))

    def bath(self, gate):
        for field in ('gained', 'lost', 'raw_points', 'retained_points'):
            require(type(gate[field]) is int and gate[field] >= 0, 'Invalid bath counter')
        require(gate['retained_points'] == gate['gained']+gate['lost']
                <= gate['raw_points'], 'Bath count partition differs')
        self.close(gate['log_weight'], math.log1p(1/64)*(gate['gained']-gate['lost']),
                   'Independent gained/lost bath identity')

    def outcome(self, row, old, points):
        outcome = row['proposal']
        require(outcome['old'] == old, 'Dimer source differs from retained state')
        if not self.contact(old):
            require(outcome['status'] == 'source_outside_domain' and not outcome['attempts']
                    and outcome['candidate'] is None, 'Disconnected pair is not an identity move')
            self.counters['outside_contact_self_loops'] += 1
            return
        old_edges = [relative_pose(self.anchor, old[0]), relative_pose(old[0], old[1])]
        guidance = outcome.get('guidance')
        threshold = None
        if guidance is not None:
            old_count = self.count(points, old_edges[1])
            require(guidance['old_count'] == old_count and guidance['point_count'] == len(points)
                    and guidance['m'] == 4 and len(guidance['integer_draws']) == 4,
                    'Guidance source counts differ')
            threshold = max(guidance['integer_draws'])
            require(0 <= threshold <= old_count and threshold == guidance['threshold'],
                    'Threshold law trace invalid')
        for attempt in outcome['attempts']:
            for stage in ('root', 'internal'):
                draws = attempt[stage+'_draws']
                passed = False
                for index, draw in enumerate(draws):
                    require(not passed and draw['index'] == index+1, 'Hidden edge retry/index')
                    relative = decode_edge(self.model, draw['draw'], .5, 2., self.checks, stage)
                    self.checks.pose(draw['draw']['old_relative_pose'], old_edges[stage == 'internal'],
                                     'evolving source tree edge', stage)
                    p = np.array(relative['position'])
                    if stage == 'root':
                        world = compose_pose(self.anchor, relative)
                        self.checks.pose(draw['world_pose'], world, 'root composition', stage)
                        p = np.array(draw['world_pose']['position'])
                        geometry = dict(spectator_core_collisions=([2] if np.linalg.norm(p-self.anchor['position']) < 2*RADIUS else []),
                                        wall_valid=bool(np.linalg.norm(p) <= 8.-RADIUS))
                        passed = not geometry['spectator_core_collisions'] and geometry['wall_valid']
                    else:
                        geometry = dict(internal_core_overlap=bool(np.linalg.norm(p) < 2*RADIUS),
                                        internal_exclusion_contact=bool(np.linalg.norm(p) < 2*(RADIUS+RD)))
                        passed = not geometry['internal_core_overlap'] and geometry['internal_exclusion_contact']
                        if passed and threshold is not None:
                            count = self.count(points, relative)
                            require(draw['guidance_count'] == count, 'Rejected or successful guidance count differs')
                            passed = count >= threshold
                    require(draw['feasibility'] == geometry, 'Independent raw sphere predicate differs')
                    self.counters['audited_raw_edges'] += 1
                if draws: require(passed or len(draws) == 16, 'Premature cap exhaustion')
        candidate = outcome['candidate']
        if candidate is None:
            require(row['status'] == 'proposal_self_loop', 'Candidate null not retained as self loop')
            return
        new = [candidate['root'], candidate['child']]
        require(new == row['proposed'] and self.hard_valid(new) and self.contact(new),
                'Candidate final geometry differs')
        new_edges = [relative_pose(self.anchor, new[0]), relative_pose(new[0], new[1])]
        diag = candidate['diagnostics']
        sums = []
        for prefix, edges in [('old', old_edges), ('new', new_edges)]:
            expected = [self.model.edge_density(p, .5, 2., kind='map') for p in edges]
            for a, b in zip(diag[prefix+'_edges'], expected):
                for key in b: self.close(a[key], b[key], 'Complete defensive edge '+key)
            sums.append(sum(v['log_full'] for v in expected))
            self.close(diag['full_'+prefix+'_log_density'], sums[-1], 'Complete tree F')
        correction = sums[0]-sums[1]
        self.close(diag['log_reverse_forward'], correction, 'Complete F ratio')
        require(diag['log_tree_coordinate_jacobian'] == diag['selection_log_reverse_forward'] == 0.,
                'Unexpected fixed-label selection/Jacobian')
        if guidance is not None:
            new_count = self.count(points, new_edges[1])
            require(guidance['new_count'] == new_count and new_count >= threshold,
                    'Accepted-domain guidance support differs')
            auxiliary = 4*(math.log1p(guidance['old_count'])-math.log1p(new_count))
            self.close(guidance['aux_log_correction'], auxiliary, 'Auxiliary count correction')
            correction += auxiliary
        self.close(row['complete_log_correction'], correction, 'Exactly one F and auxiliary correction')
        legs, aggregate = row['bath']['legs'], row['bath']['aggregate']
        require(len(legs) == 2, 'Dimer bath is not two singleton legs')
        for leg in legs: self.bath(leg)
        for key in ('gained', 'lost', 'raw_points', 'retained_points'):
            require(aggregate[key] == sum(l[key] for l in legs), 'Path aggregation differs')
        self.close(aggregate['log_weight'], sum(l['log_weight'] for l in legs), 'Path log weight sum')
        order = row['bath']['ordered_members']
        require(sorted(order) == [0, 1], 'Path order invalid')
        intermediate = copy.deepcopy(old)
        intermediate[order[0]] = new[order[0]]
        require(row['bath']['intermediate_selected'] == intermediate, 'Path copied intermediate differs')
        self.close(row['log_acceptance_ratio'], correction+aggregate['log_weight'], 'Full physical MH identity')

    def audit(self, job):
        directory = self.path(job)
        rows = [json.loads(line) for line in (directory/'trajectory.jsonl').read_text().splitlines()]
        terminal = read(directory/'terminal.json')
        arm = self.plan['jobs'][job]['arm']
        initial = self.plan['jobs'][job]['initialization']
        require(rows[0]['kind'] == 'initial' and rows[0]['block'] == 0, 'Initial state missing')
        points = self.cloud(initial)
        old = rows[0]['selected']
        require(self.hard_valid(old), 'Initial sphere state invalid')
        counts = dict(local_attempted=0, local_accepted=0, dimer_attempted=0, dimer_accepted=0,
                      dimer_self_loop=0)
        raw = retained = block = local_index = 0
        for row in rows[1:]:
            if row['kind'] == 'retained_block':
                block += 1
                require(row['block'] == block and local_index == 4, 'Incomplete or reordered block')
                require(row['selected'] == old and row['counts'] == counts
                        and row['raw'] == raw and row['retained'] == retained, 'Retained block lost attempts/counts')
                local_index = 0
                continue
            require(row['block'] == block+1, 'Attempt block differs')
            proposed = copy.deepcopy(old)
            if row['kind'] == 'local':
                member = [0, 1, 0, 1][local_index]
                require(row['attempt'] == local_index and row['member'] == member and row['old'] == old[member],
                        'Local source or member schedule differs')
                local_index += 1
                proposed[member] = row['proposed']
                require((row['status'] == 'hard_rejected') == (not self.hard_valid(proposed)),
                        'Local hard rejection differs')
                counts['local_attempted'] += 1
                counts['local_accepted'] += row['accepted']
                if 'bath' in row:
                    self.bath(row['bath'])
                    self.close(row['log_acceptance_ratio'], row['bath']['log_weight'], 'Local MH contains bath only')
                    raw += row['bath']['raw_points']
                    retained += row['bath']['retained_points']
            else:
                require(arm != 'local' and local_index == 4 and row['kind'] == 'factorized_dimer',
                        'Unexpected dimer scheduling')
                self.outcome(row, old, points)
                counts['dimer_attempted'] += 1
                counts['dimer_accepted'] += row['accepted']
                counts['dimer_self_loop'] += row['status'] == 'proposal_self_loop'
                if 'proposed' in row: proposed = row['proposed']
                if 'bath' in row:
                    raw += row['bath']['aggregate']['raw_points']
                    retained += row['bath']['aggregate']['retained_points']
            if row['status'] == 'completed':
                ratio = float(row['log_acceptance_ratio'])
                require(row['log_u'] < 0 and row['accepted'] == (row['log_u'] < min(0., ratio)),
                        'Acceptance variate/decision differs')
                self.counters['completed_physical_decisions'] += 1
                self.counters['mh_rejections'] += not row['accepted']
            else:
                require(not row['accepted'] and 'bath' not in row and 'log_u' not in row,
                        'Self loop spent physical draws')
                self.counters[row['status']] += 1
            if row['accepted']: old = proposed
            require(row['retained'] == old and self.hard_valid(old), 'Rejected state dropped or accepted pose differs')
            self.counters['attempt_rows'] += 1
        require(block == 16 and counts['local_attempted'] == 64
                and counts['dimer_attempted'] == (0 if arm == 'local' else 16), 'Wrong full allocation')
        require(terminal['complete'] and terminal['counts'] == counts and terminal['raw'] == raw
                and terminal['retained'] == retained and terminal['blocks'] == 16, 'Terminal counts differ')
        cp = read(directory/'checkpoint.json')
        require(cp['selected'] == old and cp['journal_rows'] == len(rows)
                and cp['journal_bytes'] == (directory/'trajectory.jsonl').stat().st_size
                and cp['journal_sha256'] == sha(directory/'trajectory.jsonl'), 'Final checkpoint differs')
        require(not self.checks.failures, 'Independent pose/density checks failed: '+str(self.checks.failures))
        return rows

    def run(self):
        self.prepare()
        for job in (0, 1, 2, 8): self.call('run', job=job)
        self.call('run', job=3, stop=7)
        paused = self.path(3)/'paused.json'
        paused_sha = sha(paused)
        self.call('run', job=3, resume=self.path(3)/'checkpoint.json')
        require(sha(paused) == paused_sha, 'Graceful continuation overwrote pause receipt')
        trajectories = {job: self.audit(job) for job in (0, 1, 2, 3, 8)}
        require(scientific(trajectories[2]) == scientific(trajectories[3]),
                'Uninterrupted and resumed scientific traces differ')
        for key in ('counts', 'raw', 'retained', 'blocks'):
            require(read(self.path(2)/'terminal.json')[key] == read(self.path(3)/'terminal.json')[key],
                    'Uninterrupted/resumed terminal '+key)
        for job, defect in [(4, 'state'), (5, 'counts'), (6, 'tail'), (7, 'rows')]:
            self.call('run', job=job, stop=7)
            path = self.path(job)/'checkpoint.json'
            cp = read(path)
            if defect == 'state': cp['selected'][0]['position'][0] += .01
            elif defect == 'counts': cp['counts']['local_accepted'] += 1
            elif defect == 'rows': cp['journal_rows'] += 1
            else:
                with (self.path(job)/'trajectory.jsonl').open('a') as out: out.write('{}\n')
            if defect != 'tail': replace(path, cp)
            originals = {p.name: sha(p) for p in self.path(job).iterdir() if p.is_file()}
            self.call('run', job=job, resume=path, succeeds=False)
            for name, digest in originals.items():
                require(sha(self.path(job)/name) == digest, 'Refused continuation altered '+name)
            failure = self.path(job)/'failure.json'
            require(failure.exists(), 'Refused corruption lacked fatal receipt')
            receipt = sha(failure)
            self.call('run', job=job, resume=path, succeeds=False)
            require(sha(failure) == receipt, 'Repeated fatal continuation replaced failure receipt')
        terminal_files = {p.name: sha(p) for p in self.path(2).iterdir() if p.is_file()}
        self.call('run', job=2, resume=self.path(2)/'checkpoint.json', succeeds=False)
        self.call('run', job=2, succeeds=False)
        require({p.name: sha(p) for p in self.path(2).iterdir() if p.is_file()} == terminal_files,
                'Completed run refusal changed existing receipts')
        require(self.counters['mh_rejections'] > 0 and self.counters['hard_rejected'] > 0
                and self.counters['audited_raw_edges'] > 0, 'Synthetic controls did not exercise rejected paths')
        dispatcher = dispatch_control(self.base/'synthetic-dispatch')
        return dict(schema='evolving-dimer-cli-validation-v1', complete=True, passed=True,
            counters=dict(self.counters), independent_checks=self.checks.count,
            maximum_absolute_errors=dict(self.checks.maximum_absolute_errors),
            compared_blocks=16, stop_after_block=7, same_family_job_ids=[2, 3],
            commands=len(self.commands), dispatcher=dispatcher, no_protein_sampling=True,
            limitations=['Synthetic sphere paths; not protein geometry or equilibrium convergence.',
                'Poisson arithmetic and recorded counters reconstructed; bath point coordinates not regenerated.',
                'Finite observed traces and corruption checks do not certify floating-point execution.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--source-bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dispatch-only', action='store_true',
                        help='Only synthetic controller failure-draining controls; no physical sampler calls.')
    args = parser.parse_args()
    if args.dispatch_only:
        output = args.output.resolve()
        output.mkdir(parents=True)
        sources = [Path(__file__).resolve(), ROOT/'tools/run_evolving_dimer_benchmark.py']
        for source in sources:
            (output/source.name).write_bytes(source.read_bytes())
        write(output/'frozen-validation-inputs.json', dict(sources=[record(p) for p in sources],
            dispatch_only=True, physical_attempts=0))
        try:
            controls = [dispatch_control(output/scenario, scenario) for scenario in
                        ('nonzero', 'launch_log_error', 'malformed_terminal')]
            result = dict(schema='evolving-dispatch-validation-v1', complete=True, passed=True,
                          controls=controls, physical_attempts=0)
        except Exception as error:
            result = dict(schema='evolving-dispatch-validation-v1', complete=False, passed=False,
                          error=f'{type(error).__name__}: {error}', physical_attempts=0)
        write(output/'validation.json', result)
        print(json.dumps(result, indent=2))
        return 0 if result['passed'] else 1
    if args.binary is None or args.source_bundle is None:
        parser.error('--binary and --source-bundle are required unless --dispatch-only is selected')
    binary, bundle, output = args.binary.resolve(), args.source_bundle.resolve(), args.output.resolve()
    output.mkdir(parents=True)
    source_files = {Path(m.__file__).resolve() for m in sys.modules.values()
                    if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'
                    and Path(m.__file__).resolve().is_relative_to(ROOT/'tools')}
    source_files.update([Path(__file__).resolve(), ROOT/'tools/run_evolving_dimer_benchmark.py',
                         ROOT/'examples/evolving_dimer_benchmark.rs'])
    archived = []
    for source in sorted(source_files):
        destination = output/'validation-source'/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        archived.append(record(destination))
    (output/'source-bundle.json').write_bytes(bundle.read_bytes())
    write(output/'frozen-validation-inputs.json', dict(binary=record(binary), source_bundle=record(bundle),
        runner_source=record(ROOT/'examples/evolving_dimer_benchmark.rs'),
        validator=record(__file__), preparer=record(prep.__file__),
        dispatcher=record(ROOT/'tools/run_evolving_dimer_benchmark.py'),
        archived_source=archived,
        objective='Actual sphere CLI, independent reconstruction and graceful restart/corruption controls; no protein sampling.'))
    try:
        result = Validator(output, binary, bundle).run()
    except Exception as error:
        result = dict(schema='evolving-dimer-cli-validation-v1', complete=False, passed=False,
                      error=f'{type(error).__name__}: {error}')
    write(output/'validation.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
