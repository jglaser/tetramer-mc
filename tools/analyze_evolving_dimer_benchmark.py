#!/usr/bin/env python3
"""Frozen native-free diagnostics for conditional two-mobile-body trajectories.

This adapter deliberately does not call the all-mobile trajectory validator:
262 fixed spectators define a different conditional target. Elementary journals
are replayed before their retained endpoints are classified. No native geometry
is read. Descriptors are threshold observables, not identified thermodynamic
basins; finite-record ESS does not establish equilibrium or unseen coverage.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from analyze_contact_efficiency import ContactObserver, pose_arrays
from mobile_posterior_metrics import apparent_effective_count

SOURCE_FILES = ['tools/analyze_evolving_dimer_benchmark.py',
    'tools/test_analyze_evolving_dimer_benchmark.py', 'tools/analyze_contact_efficiency.py',
    'tools/mobile_posterior_metrics.py']


def require(condition, message):
    if not condition: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def key(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def write(path, value):
    with Path(path).open('x') as f: f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def analysis_plan():
    return dict(schema='conditional-dimer-analysis-plan-v1', native_observer=False,
        source_files=SOURCE_FILES,
        observations='Initial endpoint plus every retained_block, including warmup and rejected residence. All elementary attempts are replayed; no accepted-only filtering.',
        target='Only the two fixed labelled members move against every other body held at the source pose. Distinct contexts are distinct conditional targets and are never pooled.',
        patch_map='Frozen complete 32-patch atom map from config.patch_map; no patch or native-label filtering.',
        pair_rule='Classify the two jointly updated poses against frozen spectators, canonical global IDs, and classify the internal pair exactly once. Conservative center bound with the existing floating-point guard. Reuse only exact pair-pose self-loops.',
        metrics=['production patch-token presence ESS', 'production partner-edge presence ESS',
                 'production whole-fingerprint categorical ESS', 'instantaneous internal contact occupancy',
                 'whole fingerprint passages, completed returns, and transition counts',
                 'patch/partner occupancy and Jaccard changes at every block'],
        ess='mobile_posterior_metrics.apparent_effective_count on all production endpoints; constant observables yield null ESS, not infinite efficiency. Categorical fingerprints use exact one-hot coordinates. No trajectories are concatenated.',
        comparisons='Four independent streams within each context/arm/initialization. Summarize per-stream metrics; paired arm differences match the same stream and initialization because local RNG roles are shared. Compare occupancy with total variation and patch marginals across initializations; this is descriptive agreement, not an equilibrium test.',
        costs='Report actual full sampler CPU including nulls/rejections, production-window CPU, separately measured preparation/cloud/setup/observer CPU when available. Missing setup/preparation timing is explicit; never invent zero cost.',
        scope='Conditional relaxation at 1.4 A, activity .0275 A^-3 and inherited 500 uM source; not the original assembly decision conditions, finite-system assembly evidence, native registry, physical kinetics, or proof of equilibrium.')


class ConditionalObserver:
    """Cache pair observations only when both exact poses are unchanged."""
    def __init__(self, shape, patch_map, config, source, members):
        require(len(members) == 2 and len(set(members)) == 2 and all(type(i) is int and 0 <= i < len(source) for i in members), 'Invalid mobile labels')
        require(config['boundary']['kind'] == 'spherical', 'This adapter is nonperiodic')
        self.observer = ContactObserver(shape, patch_map, config, native=None)
        self.source = copy.deepcopy(source); self.members = tuple(members)
        self.cache = {}; self.calls = 0; self.hits = 0

    def classify(self, selected):
        require(len(selected) == 2, 'Expected two retained poses')
        pose_arrays(selected)
        state = list(self.source)
        for member, pose in zip(self.members, selected): state[member] = pose
        positions = np.asarray([p['position'] for p in state], float)
        candidates = set(); obs = self.observer
        for i in self.members:
            guard = 512*np.finfo(float).eps*(1+np.linalg.norm(positions[i])+np.linalg.norm(positions, axis=1)+obs.bound+obs.rd)
            candidates.update(tuple(sorted((i, int(j)))) for j in np.flatnonzero(
                np.linalg.norm(positions-positions[i], axis=1) <= 2*(obs.bound+obs.rd)+guard) if i != j)
        tokens = set()
        # Wall validation even for an isolated body outside every pair bound.
        pos, rot = pose_arrays(selected)
        atoms = np.einsum('ijk,ak->iaj', rot, obs.atoms)+pos[:, None, :]
        require(np.min(obs.boundary['radius']-np.linalg.norm(atoms, axis=2)-obs.radii) >= -2e-8, 'Retained mobile pose violates atomic wall')
        for pair in sorted(candidates):
            a, b = pair; poses = [state[a], state[b]]; identity = canonical(poses)
            cached = self.cache.get(pair)
            if cached is not None and cached[0] == identity:
                found = cached[1]; self.hits += 1
            else:
                classified = obs.classify(poses)
                require(not classified['instantaneous_native_keys'] and classified['native_cycle'] is None, 'Native observer unexpectedly active')
                found = tuple((a, b, pa, pb) for _, _, pa, pb in classified['tokens'])
                self.cache[pair] = (identity, found); self.calls += 1
            tokens.update(found)
        edges = sorted({(a, b) for a, b, _, _ in tokens})
        internal = tuple(sorted(self.members))
        return dict(patch_tokens=sorted(tokens), partner_edges=edges,
            internal_contact=internal in edges,
            external_edges=[e for e in edges if e != internal],
            partners_by_member={str(m): sorted(b if a == m else a for a, b in edges if m in (a, b)) for m in self.members},
            fingerprint=key(sorted(tokens)))


def validate_journal(rows, config, job, initial, terminal=None):
    """Replay every elementary update. Yield only verified retained endpoints."""
    rows=iter(rows)
    first=next(rows,None)
    require(first is not None and first['kind'] == 'initial' and first['block'] == 0
            and first['job'] == job and first['selected'] == initial
            and first['conditional_target'] is True, 'Initial journal identity differs')
    members = [config['contexts'][job['context_index']][k] for k in ('root', 'child')]
    selected = copy.deepcopy(initial); points = [first]; prior_cpu = first['sampler_cpu_seconds']
    require(math.isfinite(prior_cpu) and prior_cpu >= 0, 'Invalid initial CPU')
    counts = dict(local_attempted=0, local_accepted=0, dimer_attempted=0, dimer_accepted=0, dimer_self_loop=0)
    raw = retained = 0
    total = config['allocation']['warmup_blocks']+config['allocation']['production_blocks']
    for block in range(1, total+1):
        expected = [('local', attempt, slot) for attempt, slot in enumerate([0, 1, 0, 1])]
        if job['arm'] != 'local': expected.append(('factorized_dimer', None, None))
        for kind, attempt, slot in expected:
            row=next(rows,None)
            require(row is not None, 'Missing elementary attempt')
            require(row['kind'] == kind and row['block'] == block and 'fatal_error' not in row,
                    'Attempt sequence incomplete or fatal')
            require(type(row['accepted']) is bool, 'Invalid acceptance flag')
            cpu = row['sampler_cpu_seconds']; require(math.isfinite(cpu) and cpu >= prior_cpu, 'Nonmonotone sampler CPU'); prior_cpu = cpu
            if kind == 'local':
                require(row['attempt'] == attempt and row['member'] == members[slot] and row['old'] == selected[slot], 'Local replay mismatch')
                require(row['status'] in ('completed', 'hard_rejected'), 'Unknown local disposition')
                counts['local_attempted'] += 1; counts['local_accepted'] += row['accepted']
                if row['accepted']: selected[slot] = row['proposed']
                if row['status'] == 'hard_rejected': require(not row['accepted'], 'Hard rejection accepted')
                gate = row.get('bath')
            else:
                require(row['members'] == members and row['old'] == selected and row['anchor'] == config['contexts'][job['context_index']]['anchor'], 'Dimer replay mismatch')
                require(row['status'] in ('completed', 'proposal_self_loop'), 'Unknown dimer disposition')
                counts['dimer_attempted'] += 1; counts['dimer_accepted'] += row['accepted']
                if row['status'] == 'proposal_self_loop':
                    require(not row['accepted'] and row['proposal']['candidate'] is None, 'False proposal self-loop')
                    counts['dimer_self_loop'] += 1
                if row['accepted']: selected = copy.deepcopy(row['proposed'])
                gate = row.get('bath', {}).get('aggregate')
            if row['status'] == 'completed':
                ratio = -math.inf if row['log_acceptance_ratio'] == '-inf' else row['log_acceptance_ratio']
                require(not math.isnan(ratio) and ratio != math.inf and math.isfinite(row['log_u']) and row['log_u'] < 0
                        and row['accepted'] == (row['log_u'] < min(0., ratio)), 'Acceptance decision mismatch')
                require(gate is not None, 'Completed physical gate missing')
            if gate:
                raw += gate['raw_points']; retained += gate['retained_points']
            require(row['retained'] == selected, 'Rejected/accepted retained state mismatch')
        row=next(rows,None)
        require(row is not None, 'Missing retained block')
        require(row['kind'] == 'retained_block' and row['block'] == block
                and row['production'] is (block > config['allocation']['warmup_blocks'])
                and row['selected'] == selected and row['counts'] == counts
                and row['raw'] == raw and row['retained'] == retained, 'Retained block/count closure mismatch')
        require(math.isfinite(row['sampler_cpu_seconds']) and row['sampler_cpu_seconds'] >= prior_cpu, 'Nonmonotone retained CPU')
        prior_cpu = row['sampler_cpu_seconds']; points.append(row)
    require(next(rows,None) is None, 'Unexpected journal tail')
    if terminal is not None:
        require(terminal['complete'] is True and terminal['conditional_target'] is True and terminal['job'] == job
                and terminal['blocks'] == total and terminal['counts'] == counts
                and terminal['raw'] == raw and terminal['retained'] == retained
                and terminal['cpu_seconds'] >= prior_cpu, 'Terminal journal closure mismatch')
    return points


def categorical_summary(labels, blocks):
    frequencies = Counter(labels); transitions = Counter(); returns = []; last_exit = {}
    for index in range(1, len(labels)):
        old, new = labels[index-1:index+1]
        if old != new:
            transitions[(old, new)] += 1
            last_exit[old] = blocks[index]
            if new in last_exit:
                returns.append(dict(environment=new, departure_block=last_exit.pop(new), return_block=blocks[index]))
    return dict(occupancy={label: count/len(labels) for label, count in sorted(frequencies.items())},
        transitions=[dict(source=a, target=b, count=count) for (a,b),count in sorted(transitions.items())],
        completed_passages=sum(transitions.values()), completed_returns=returns,
        exchange_definition='Each changed whole fingerprint is a completed observed passage; return means re-entering an earlier exited fingerprint at a strictly later block. No dwell/native/basin filtering. Events within a block unresolved.')


def presence_matrix(values):
    columns = sorted(set().union(*(set(map(tuple, value)) for value in values)))
    indices = {value:i for i,value in enumerate(columns)}
    result = np.zeros((len(values), max(1,len(columns))), float)
    for row, value in enumerate(values):
        for token in value: result[row, indices[tuple(token)]] = 1
    return result, columns


def summarize_trace(trace, warmup, sampler_cpu):
    production = [row for row in trace if row['block'] > warmup]
    require(production, 'No production observations')
    labels = [row['fingerprint'] for row in production]; dictionary = sorted(set(labels))
    # One-hot categorical coordinates avoid imposing an arbitrary distance/order
    # on fingerprint IDs; rejected residence remains repeated rows.
    fingerprints, _ = presence_matrix([[(label,)] for label in labels])
    patches, patch_ids = presence_matrix([row['patch_tokens'] for row in production])
    partners, partner_ids = presence_matrix([row['partner_edges'] for row in production])
    return dict(production_samples=len(production), full_sampler_cpu_seconds=sampler_cpu,
        fingerprint=categorical_summary(labels, [r['block'] for r in production]),
        fingerprint_ess=apparent_effective_count(fingerprints, sampler_cpu),
        patch_ess=apparent_effective_count(patches, sampler_cpu),
        partner_ess=apparent_effective_count(partners, sampler_cpu),
        patch_occupancy=[dict(token=token, fraction=float(patches[:,i].mean())) for i,token in enumerate(patch_ids)],
        partner_occupancy=[dict(edge=edge, fraction=float(partners[:,i].mean())) for i,edge in enumerate(partner_ids)],
        internal_contact_fraction=float(np.mean([r['internal_contact'] for r in production])),
        unique_fingerprints=len(dictionary), constant_fingerprint=len(dictionary)==1,
        singleton_fingerprint_fraction=sum(v == 1 for v in Counter(labels).values())/len(labels),
        partner_environments=categorical_summary([key(r['partner_edges']) for r in production],[r['block'] for r in production]),
        scope=analysis_plan()['scope'])


def total_variation(a, b): return .5*math.fsum(abs(a.get(k,0.)-b.get(k,0.)) for k in a.keys()|b.keys())


def occupancy_difference(left,right,label):
    a={tuple(r[label]):r['fraction'] for r in left};b={tuple(r[label]):r['fraction'] for r in right}
    return max((abs(a.get(k,0.)-b.get(k,0.)) for k in a.keys()|b.keys()),default=0.)


def comparison_summaries(chains):
    grouped = defaultdict(list)
    for row in chains: grouped[(row['job']['context_index'], row['job']['arm'], row['job']['initialization'])].append(row)
    result = []
    for (context, arm, initialization), group in sorted(grouped.items()):
        result.append(dict(context_index=context, arm=arm, initialization=initialization,
            streams=[r['job']['stream'] for r in group],
            independent_stream_metrics=[dict(stream=r['job']['stream'],
                fingerprint_ess=r['metrics']['fingerprint_ess']['apparent_ess'],
                fingerprint_ess_per_cpu=r['metrics']['fingerprint_ess']['apparent_ess_per_sampling_CPU_second'],
                internal_contact_fraction=r['metrics']['internal_contact_fraction'],
                completed_passages=r['metrics']['fingerprint']['completed_passages'],
                completed_returns=len(r['metrics']['fingerprint']['completed_returns'])) for r in group]))
    comparisons=[]
    # Compare same-target streams across arms and initializations; never claim
    # four contexts are replicate draws from one physical distribution.
    for context in sorted({r['job']['context_index'] for r in chains}):
        population=[r for r in chains if r['job']['context_index']==context]
        for a,b in itertools.combinations(population,2):
            ja,jb=a['job'],b['job']
            if ja['stream']!=jb['stream']:continue
            if ja['arm']!=jb['arm'] and ja['initialization']!=jb['initialization']:continue
            comparisons.append(dict(context_index=context,left=ja,right=jb,
                local_rng_paired=ja['initialization']==jb['initialization'],
                fingerprint_total_variation=total_variation(a['metrics']['fingerprint']['occupancy'],b['metrics']['fingerprint']['occupancy']),
                internal_contact_difference=b['metrics']['internal_contact_fraction']-a['metrics']['internal_contact_fraction'],
                patch_occupancy_max_difference=occupancy_difference(a['metrics'].get('patch_occupancy',[]),b['metrics'].get('patch_occupancy',[]),'token'),
                partner_occupancy_max_difference=occupancy_difference(a['metrics'].get('partner_occupancy',[]),b['metrics'].get('partner_occupancy',[]),'edge')))
    return dict(groups=result, descriptive_paired_comparisons=comparisons,
        warning='Streams are independent within an arm; cross-arm local random numbers are paired. Initial conditions are deliberately different, not independent physical preparations. No context or trajectory pooling; no equilibrium claim.')


def analyze(base, output):
    from prepare_evolving_dimer_benchmark import verify_freeze, checked_file, validate_prepared_manifest
    base=Path(base).resolve(); output=Path(output).resolve(); require(not output.exists(), 'Analysis output exists')
    config=verify_freeze(base); frozen_plan=read(base/'analysis-plan.json')
    require(frozen_plan == analysis_plan(), 'Frozen analysis plan differs from current observer')
    protocol=read(base/'protocol.json')
    for name in SOURCE_FILES: require(sha(Path(__file__).parents[1]/name)==protocol['source_files'][name], 'Observer source changed after freeze: '+name)
    binding=read(base/'run-binding.json'); prepared, _=validate_prepared_manifest(base, checked_file(binding['prepared_manifest']))
    require(binding['config_sha256']==sha(base/'config.json') and binding['protocol_sha256']==sha(base/'protocol.json'), 'Run provenance changed')
    shape=read(checked_file(config['shape'])); patch=read(checked_file(config['patch_map'])); source=read(checked_file(config['source_frame']))['poses']
    require(patch['shape_sha256']==config['shape']['sha256'] and len(patch['atom_patch_ids'])==len(shape['atoms'])
            and set(patch['atom_patch_ids'])==set(patch['patch_dictionary']), 'Patch identity differs')
    observer_config=dict(boundary=dict(kind='spherical',radius=config['physical']['wall_radius']),
        box_lengths=[2*config['physical']['wall_radius']]*3, depletant_radius=config['physical']['depletant_radius'])
    output.mkdir(); chains=[]; input_files={}; started=time.process_time()
    write(output/'input-binding.json',dict(config_sha256=sha(base/'config.json'), run_binding_sha256=sha(base/'run-binding.json'),
        analysis_plan_sha256=sha(base/'analysis-plan.json'), source_files={n:protocol['source_files'][n] for n in SOURCE_FILES}))
    for job in config['jobs']:
        job_start=time.process_time(); directory=Path(config['output'])/f"job-{job['id']:03}"
        terminal=read(directory/'terminal.json'); trajectory=checked_file(terminal['trajectory'])
        require(terminal['config_sha256']==sha(base/'config.json') and terminal['binding_sha256']==sha(base/'run-binding.json'), 'Job provenance differs')
        input_files[str(trajectory)]=sha(trajectory); input_files[str(directory/'terminal.json')]=sha(directory/'terminal.json')
        context=config['contexts'][job['context_index']]; members=[context[k] for k in ('root','child')]
        initial=[source[i] for i in members]
        if job['initialization']=='proposal_prepared':
            saved=next(s for s in prepared['alternative_starts'] if s['context_index']==job['context_index'] and s['stream']==job['stream'])
            initial=read(checked_file(saved['record']))['selected']
        with trajectory.open() as handle:
            points=validate_journal((json.loads(line) for line in handle),config,job,initial,terminal)
        observer=ConditionalObserver(shape,patch['atom_patch_ids'],observer_config,source,members)
        trace=[]; prior=None
        with (output/f"job-{job['id']:03}-observations.jsonl").open('x') as handle:
            for point in points:
                observation=observer.classify(point['selected']); tokens=set(observation['patch_tokens'])
                old=set(prior['patch_tokens']) if prior else tokens; union=old|tokens
                observation.update(block=point['block'],production=point['block']>config['allocation']['warmup_blocks'],
                    sampler_cpu_seconds=point['sampler_cpu_seconds'],
                    jaccard_from_previous=1-len(old&tokens)/len(union) if union else 0.)
                handle.write(canonical(observation)+'\n'); trace.append(observation); prior=observation
        metrics=summarize_trace(trace,config['allocation']['warmup_blocks'],terminal['cpu_seconds'])
        metrics['production_cpu_seconds']=terminal['cpu_seconds']-points[config['allocation']['warmup_blocks']]['sampler_cpu_seconds']
        chains.append(dict(job=job,metrics=metrics,observer_cpu_seconds=time.process_time()-job_start,
            pair_classifications=observer.calls,exact_pair_cache_hits=observer.hits,
            trajectory=terminal['trajectory'],counts=terminal['counts'],geometry_load_cpu_seconds=terminal.get('geometry_load_cpu_seconds')))
    costs=dict(sampler_cpu_seconds=math.fsum(c['metrics']['full_sampler_cpu_seconds'] for c in chains),
        cloud_cpu_seconds=math.fsum(read(checked_file(b['metadata']))['cpu_seconds'] for b in prepared['cloud_banks']),
        preparation_cpu_seconds=prepared.get('preparation_cpu_seconds'),
        preparation_geometry_load_cpu_seconds=prepared.get('geometry_load_cpu_seconds'),
        per_chain_geometry_load_cpu_seconds=math.fsum(c['geometry_load_cpu_seconds'] for c in chains)
            if all(c['geometry_load_cpu_seconds'] is not None for c in chains) else None,
        observer_cpu_seconds=time.process_time()-started)
    cost_parts=[costs['sampler_cpu_seconds'],costs['preparation_cpu_seconds'],
        costs['preparation_geometry_load_cpu_seconds'],costs['per_chain_geometry_load_cpu_seconds']]
    costs['setup_inclusive_sampling_cpu_seconds']=math.fsum(cost_parts) if all(v is not None for v in cost_parts) else None
    costs['cloud_cost_note']='Cloud CPU is a subset of preparation CPU; not added a second time. Observer CPU is reported separately.'
    result=dict(schema='conditional-dimer-analysis-v1',complete=True,chains=chains,
        comparisons=comparison_summaries(chains),costs=costs,input_files=input_files,analysis_plan=frozen_plan)
    write(output/'analysis.json',result)
    write(output/'manifest.json',dict(complete=True,files={str(p.relative_to(output)):sha(p) for p in output.iterdir() if p.is_file()}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.base,args.output);print(json.dumps(result['costs'],indent=2))
