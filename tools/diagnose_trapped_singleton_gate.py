#!/usr/bin/env python3
"""Bounded extraction of saved singleton gate factors; no physics or replay."""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import signal
import sys
import time

from analyze_mobile_native_pocket import local_sources
import bind_conditional_native_audit as admission
from run_evolving_dimer_analysis import owned_child
from run_native_class_physical_campaign import read, require, sha, write

SCHEMA = 'trapped-singleton-gate-diagnostic-v1'
ANALYSIS_SHA = '2d956ca548b1c3f2abd563d5d6473f6f766f5e05117002603ac9c899e29c98ff'
ARMS = ('singleton_two_neighbor', 'singleton_two_neighbor_unfused')
STARTS = ('source', 'proposal_prepared')
LIMITS = dict(cpu_seconds=600, wall_seconds=1200, address_space_bytes=4*1024**3,
              threads=1, maximum_record_bytes=64*1024**2)
ALLOCATION = dict(chains=16, blocks=4608, warmup=512, attempts=73728,
                  production_attempts=65536, maximum_inner_trials=2359296,
                  geometry_queries=0, physical_draws=0, arithmetic_replays=0)
TERMS = ('proposal_log_ratio', 'bath_log_factor', 'combined_log_ratio', 'log_u',
         'old_log_full', 'old_log_uniform', 'old_log_learned',
         'new_log_full', 'new_log_uniform', 'new_log_learned')
INTERPRETATION = dict(
    saved_identity='T = C + B; C = log G(old) - log G(new). This identity and MH decision were already replayed in the bound contact audit.',
    proposal_sign='Positive C favors the candidate; negative C penalizes it under the complete defensive proposal.',
    bath_sign='Positive B favors the candidate in this realized auxiliary Poisson draw; B is not a physical free-energy difference.',
    other_terms='No separate anchor-selection, source-label or Jacobian term in this independent conditional singleton redraw.',
    densities='G is the complete 50/50 uniform/learned mixture. Saved unweighted component logs are retained without rescoring or normalization.',
    labels='A learned target label is catalogue-local and rebuilt each attempt; it is not a native contact label. No source label is sampled.',
    retries='All inner trials are retained. hard_rejections and wall_rejections can overlap; geometric_rejections is their union.',
    statistics='Descriptive finite-record factor distributions, retaining missing and negative-infinity terms separately. No factor exponentiation, physical expectation, causation claim, gate replay or counterfactual acceptance.',
    low_acceptance='Persistent binding can be favorable native registry. Low departure rates alone do not establish a sampling failure or unwanted misregistration.',
    selection='Context 2 was chosen post hoc because the completed benchmark reported no accepted source-start singleton moves. This is a development diagnostic, not a prospective context-level hypothesis test.',
    strata='Within that selected context, all 16 chains and every attempt are fixed before journal inspection, separated by arm/start/stream/member and warmup versus production; no further event or chain filtering.')


def reference(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))


def closure(): return {str(p): sha(p) for p in local_sources(__file__).values()}


def identity(job):
    return tuple(job[k] for k in ('context_index', 'arm', 'initialization', 'stream'))


def make_plan(contact_root, root):
    """Authenticate metadata and hash journals; never decode a trajectory row."""
    b = admission.Bindings(); origin = admission._contact_receipt(b, contact_root, singleton=True)
    require(origin['analysis']['sha256'] == ANALYSIS_SHA, 'Different completed contact analysis')
    config = origin['config']; context = config['contexts'][2]
    require([context[k] for k in ('root', 'child', 'anchor')] == [9, 24, 135], 'Different conditional labels')
    require(config['physical']['depletant_radius'] == 1.4 and config['physical']['activity'] == .0275,
            'Different conditional physical target')
    chosen = [c for c in origin['result']['chains'] if c['job']['context_index'] == 2 and c['job']['arm'] in ARMS]
    expected = set(itertools.product((2,), ARMS, STARTS, range(4)))
    require(len(chosen) == 16 and {identity(c['job']) for c in chosen} == expected, 'Missing/duplicate selected chains')
    chains = []
    for chain in sorted(chosen, key=lambda c: identity(c['job'])):
        job = chain['job']; require(chain['reused_control'] is False, 'Singleton marked as cached control')
        directory = Path(config['output']).resolve()/f"job-{job['id']:03}"
        trajectory = chain['trajectory']; require(trajectory['path'] == str(directory/'trajectory.jsonl'), 'Changed journal path')
        require(origin['result']['input_sha256'].get(trajectory['path']) == trajectory['sha256'], 'Journal omitted by replay receipt')
        b.file(trajectory['path'], origin['plan']['files']); b.load(origin['config_binding'])
        terminal = b.file(directory/'terminal.json', origin['result']['input_sha256']); value = b.load(terminal)
        require(origin['plan']['files'].get(terminal['path']) == terminal['sha256'] and value['complete'] is True
                and value['conditional_target'] is True and value['job'] == job and value['blocks'] == 4608
                and value['trajectory'] == trajectory and value['counts'] == chain['counts']
                and value['config_sha256'] == origin['config_binding']['sha256']
                and value['binding_sha256'] == origin['run_binding_ref']['sha256']
                and value['counts']['singleton_attempted'] == 4608
                and not (directory/'failure.json').exists(), 'Incomplete/unbound selected terminal')
        chains.append(dict(id=f"{job['arm']}-{job['initialization']}-s{job['stream']}", job=job,
                           trajectory=trajectory, terminal=terminal, counts=chain['counts']))
    frozen_sources = {}
    for name in ('src/evolving_dimer.rs', 'src/two_neighbor_singleton.rs', 'src/defensive_dimer_proposal.rs',
                 'src/oligomer_proposal.rs', 'src/depletion.rs', 'examples/evolving_dimer_benchmark.rs'):
        frozen_sources[name] = b.file(Path(origin['plan']['base'])/'common/source'/name, origin['plan']['files'])
    return dict(schema=SCHEMA+'-plan', root=str(Path(root).resolve()), contact_root=str(Path(contact_root).resolve()),
        contact_receipts={k: origin[k] for k in ('summary', 'manifest', 'analysis', 'plan_binding')},
        config=origin['config_binding'], physical=config['physical'], members=[9, 24], anchor=135,
        chains=chains, allocation=ALLOCATION, limits=LIMITS, interpretation=INTERPRETATION,
        frozen_producer_sources=frozen_sources, input_sha256=b.files, source_sha256=closure(),
        runtime=admission.runtime(), launched=False)


def log_value(value):
    if value is None: return None
    if value == '-inf': return -math.inf
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid saved log term')
    return float(value)


def extract_attempt(row, *, chain_id, members):
    """Copy recorded terms/labels only. Deliberately do not replay their identities."""
    p = row['proposal']; slot = (row['block']-1) % 2
    require(row['member_slot'] == slot and row['member'] == members[slot]
            and row['neighbors'] == [members[1-slot], 135], 'Wrong member cadence')
    require(type(row['accepted']) is bool and row['status'] in ('completed', 'proposal_self_loop'), 'Unexpected attempt status')
    trials = p['trials']; require(type(trials) is list and len(trials) <= 32, 'Too many inner trials')
    inner = []
    for index, trial in enumerate(trials, 1):
        require(trial['index'] == index and trial['branch'] in ('uniform', 'learned')
                and trial['disposition'] in ('candidate', 'geometric_rejection'), 'Unexpected inner-trial record')
        inner.append({k: trial[k] for k in ('index', 'branch', 'disposition', 'trace', 'proposed_pose', 'density', 'feasibility')})
    terms = dict(proposal_log_ratio=row.get('complete_log_correction'), bath_log_factor=row.get('bath', {}).get('log_weight'),
                 combined_log_ratio=row.get('log_acceptance_ratio'), log_u=row.get('log_u'))
    for side in ('old', 'new'):
        for part in ('full', 'uniform', 'learned'):
            terms[f'{side}_log_{part}'] = (p.get(side+'_density') or {}).get('log_'+part)
    for term in terms.values(): log_value(term)
    if row['status'] == 'completed':
        require(p['status'] == 'candidate' and all(terms[name] is not None for name in TERMS[:4]), 'Candidate has undefined gate terms')
    else:
        require(not row['accepted'] and p['status'] in ('cap_zero', 'source_zero_reverse_flow', 'cap_exhausted')
                and all(terms[name] is None for name in TERMS[:4]), 'Self-loop has gate terms')
    return dict(chain_id=chain_id, block=row['block'], phase='production' if row['block'] > 512 else 'warmup',
        member_slot=slot, member=row['member'], neighbors=row['neighbors'], accepted=row['accepted'],
        status=row['status'], proposal_status=p['status'], terms=terms,
        saved_proposal_full_log_reverse_forward=p.get('full_log_reverse_forward'),
        source_pose=row['old'], candidate_pose=row.get('proposed'), retained_poses=row['retained'],
        source_label=None, source_label_reason='Independent redraw samples only a destination label.',
        inner_trials=inner, proposal_counts=p['counts'], saved_bath=row.get('bath'))


def iter_attempts(rows, chain, *, blocks=4608):
    """Check exact record/block/member inventory, without replaying MC state or arithmetic."""
    rows = iter(rows); first = next(rows, None)
    require(first is not None and first['kind'] == 'initial' and first['block'] == 0
            and first['job'] == chain['job'] and first['conditional_target'] is True, 'Initial record differs')
    for block in range(1, blocks+1):
        for index in range(4):
            row = next(rows, None)
            require(row is not None and row['kind'] == 'local' and row['block'] == block and row['attempt'] == index,
                    'Missing or reordered local record')
        row = next(rows, None)
        require(row is not None and row['kind'] == 'two_neighbor_singleton' and row['block'] == block,
                'Missing or reordered singleton attempt')
        yield extract_attempt(row, chain_id=chain['id'], members=[9, 24])
        retained = next(rows, None)
        require(retained is not None and retained['kind'] == 'retained_block' and retained['block'] == block
                and retained['production'] is (block > 512), 'Missing retained block')
    require(next(rows, None) is None, 'Unexpected journal tail')


def decoded_rows(path, maximum):
    with Path(path).open('rb') as stream:
        while raw := stream.readline(maximum+1):
            require(len(raw) <= maximum and raw.endswith(b'\n'), 'Oversized/truncated record')
            yield json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Invalid constant '+x)))


class Stratum:
    def __init__(self):
        self.counts = Counter(); self.proposal_counts = Counter(); self.statuses = Counter(); self.branches = Counter()
        self.target_kinds = Counter(); self.terms = {name: [] for name in TERMS}; self.signs = Counter()

    def add(self, event):
        self.counts['attempts'] += 1; self.counts['accepted'] += event['accepted']
        self.counts['proposal_self_loops'] += event['status'] == 'proposal_self_loop'
        self.counts['candidates'] += event['status'] == 'completed'
        self.statuses[event['proposal_status']] += 1
        for name, value in event['proposal_counts'].items():
            require(type(value) is int and value >= 0, 'Invalid saved proposal count'); self.proposal_counts[name] += value
        for trial in event['inner_trials']:
            self.branches[trial['branch']+'/'+trial['disposition']] += 1
            label = trial['trace'].get('target_label')
            self.target_kinds['uniform' if trial['branch'] == 'uniform' else label['kind']] += 1
        for name, value in event['terms'].items(): self.terms[name].append(log_value(value))
        def sign(value): return 'missing' if value is None else 'negative' if value < 0 else 'positive' if value > 0 else 'zero'
        self.signs[sign(log_value(event['terms']['proposal_log_ratio']))+'/'+sign(log_value(event['terms']['bath_log_factor']))] += 1

    def result(self):
        return dict(counts=dict(self.counts), proposal_counts=dict(self.proposal_counts), proposal_statuses=dict(self.statuses),
                    inner_branch_dispositions=dict(self.branches), inner_target_kinds=dict(self.target_kinds),
                    proposal_bath_signs=dict(self.signs), terms={name: summarize(values) for name, values in self.terms.items()})


def summarize(values):
    finite = sorted(v for v in values if v is not None and math.isfinite(v))
    def quantile(p):
        index = (len(finite)-1)*p; i = int(index); j = min(i+1, len(finite)-1)
        return (1-(index-i))*finite[i]+(index-i)*finite[j]
    return dict(attempts=len(values), missing=sum(v is None for v in values),
        negative_infinity=sum(v == -math.inf for v in values), finite=len(finite),
        finite_negative=sum(v < 0 for v in finite), finite_zero=sum(v == 0 for v in finite),
        finite_positive=sum(v > 0 for v in finite),
        finite_summary=None if not finite else dict(mean=math.fsum(v/len(finite) for v in finite),
            minimum=finite[0], maximum=finite[-1], quantiles={str(p): quantile(p) for p in (.05, .25, .5, .75, .95)}))


def verify_plan(root, digest):
    root = Path(root).resolve(); path = root/'plan.json'
    require(sha(path) == digest, 'Changed diagnostic plan'); plan = read(path)
    require(plan['schema'] == SCHEMA+'-plan' and plan['root'] == str(root)
            and plan['allocation'] == ALLOCATION and plan['limits'] == LIMITS
            and plan['interpretation'] == INTERPRETATION, 'Changed diagnostic declaration')
    require(plan['source_sha256'] == closure() and plan['runtime'] == admission.runtime(), 'Changed source/runtime')
    for path, expected in plan['input_sha256'].items(): require(sha(path) == expected, 'Changed authenticated diagnostic input')
    return plan


def prepare(contact_root, root):
    root = Path(root).resolve(); require(not root.exists(), 'Fresh diagnostic root required')
    plan = make_plan(contact_root, root); root.mkdir(parents=True)
    write(root/'plan.json', plan); (root/'source').mkdir()
    for path, digest in plan['source_sha256'].items():
        saved = root/'source'/Path(path).name
        with saved.open('xb') as stream: stream.write(Path(path).read_bytes())
        require(sha(saved) == digest, 'Source changed during archive')
    write(root/'preparation.json', dict(complete=True, launched=False, plan=reference(root/'plan.json'),
        allocation=ALLOCATION, limits=LIMITS, raw_rows_decoded=0, geometry_queries=0, arithmetic_replays=0,
        requires_root_launch_approval=True))
    return reference(root/'plan.json')


def append(stream, value):
    stream.write(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n'); stream.flush(); os.fsync(stream.fileno())


def worker_argv(root, digest):
    return [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(root), '--plan-sha256', digest]


def worker(root, digest):
    root = Path(root).resolve(); plan = verify_plan(root, digest); claim = read(root/'claim.json')
    require(claim['plan_sha256'] == digest and claim['pid'] == os.getppid()
            and claim['argv'] == worker_argv(root, digest), 'Unowned diagnostic worker')
    parent = Path(f'/proc/{claim["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
    require(int(parent[19]) == claim['birth_ticks'] and parent[0] not in ('Z', 'X'), 'Parent identity changed')
    output = root/'analysis'; output.mkdir(); started = time.process_time(); wall = time.monotonic()
    counts = Counter(); completed = []; groups = []; files = {}
    def stop(number, _): raise RuntimeError('Stopped diagnostic signal '+str(number))
    signals = (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU); old = {s: signal.signal(s, stop) for s in signals}
    with (output/'progress.jsonl').open('x') as progress:
        try:
            for chain in plan['chains']:
                append(progress, dict(state='chain_begin', chain_id=chain['id']))
                strata = defaultdict(Stratum); path = output/(chain['id']+'.jsonl'); chain_counts = Counter()
                with path.open('x') as stream:
                    for event in iter_attempts(decoded_rows(chain['trajectory']['path'], LIMITS['maximum_record_bytes']), chain):
                        require(time.process_time()-started <= LIMITS['cpu_seconds'] and time.monotonic()-wall <= LIMITS['wall_seconds'], 'Diagnostic budget exceeded')
                        counts['attempts_begun'] += 1; require(counts['attempts_begun'] <= ALLOCATION['attempts'], 'Attempt cap exceeded')
                        strata[event['phase'], event['member_slot']].add(event)
                        counts['inner_trials'] += len(event['inner_trials'])
                        require(counts['inner_trials'] <= ALLOCATION['maximum_inner_trials'], 'Inner trial cap exceeded')
                        append(stream, event); counts['attempts_completed'] += 1
                        counts['production_attempts'] += event['phase'] == 'production'
                        chain_counts['accepted'] += event['accepted']; chain_counts['self_loops'] += event['status'] == 'proposal_self_loop'
                require(chain_counts['accepted'] == chain['counts']['singleton_accepted']
                        and chain_counts['self_loops'] == chain['counts']['singleton_self_loop'], 'Saved attempt counts differ from completed receipt')
                require(sha(chain['trajectory']['path']) == chain['trajectory']['sha256'], 'Journal changed while reading')
                for (phase, slot), value in sorted(strata.items()):
                    result = value.result(); require(result['counts']['attempts'] == (2048 if phase == 'production' else 256), 'Incomplete stratum')
                    groups.append(dict(chain_id=chain['id'], job=chain['job'], phase=phase, member_slot=slot,
                                       member=plan['members'][slot], **result))
                files[str(path)] = sha(path); completed.append(chain['id'])
                append(progress, dict(state='chain_complete', chain_id=chain['id'], counts=dict(counts)))
            require(len(completed) == 16 and len(groups) == 64 and counts['attempts_begun'] == counts['attempts_completed'] == 73728
                    and counts['production_attempts'] == 65536, 'Incomplete diagnostic allocation')
            verify_plan(root, digest)
            matched = []
            index = {(g['job']['arm'], g['job']['initialization'], g['job']['stream'], g['phase'], g['member_slot']): g for g in groups}
            for start, stream, phase, slot in itertools.product(STARTS, range(4), ('warmup', 'production'), range(2)):
                a, b = (index[arm, start, stream, phase, slot] for arm in ARMS)
                matched.append(dict(initialization=start, stream=stream, phase=phase, member_slot=slot, member=plan['members'][slot],
                    left_chain=a['chain_id'], right_chain=b['chain_id'], difference_direction='unfused minus fused',
                    accepted_difference=b['counts']['accepted']-a['counts']['accepted'],
                    finite_mean_term_differences={name: None if any(g['terms'][name]['finite_summary'] is None for g in (a, b)) else
                        b['terms'][name]['finite_summary']['mean']-a['terms'][name]['finite_summary']['mean'] for name in TERMS},
                    conditional_finite_means=True, paired_rng_roles=True))
            files[str(output/'progress.jsonl')] = sha(output/'progress.jsonl')
            result = dict(schema=SCHEMA, complete=True, passed=True, plan_sha256=digest, counts=dict(counts), strata=groups,
                matched_arm_comparisons=matched, files=files, interpretation=INTERPRETATION,
                cpu_seconds=time.process_time()-started, wall_seconds=time.monotonic()-wall,
                geometry_queries=0, physical_draws=0, arithmetic_replays=0, free_energy_estimate=False)
            write(output/'summary.json', result)
        except BaseException as error:
            for s in signals: signal.signal(s, signal.SIG_IGN)
            append(progress, dict(state='failure', error=repr(error), counts=dict(counts)))
            write(output/'failure.json', dict(complete=False, passed=False, error=repr(error), counts=dict(counts),
                completed_chains=completed, partial_outputs_retained=True, retries=0, replacements=0))
            raise
        finally:
            for s, handler in old.items(): signal.signal(s, handler)


def run(root, digest):
    root = Path(root).resolve(); plan = verify_plan(root, digest)
    require(not (root/'claim.json').exists() and not (root/'analysis').exists(), 'Diagnostic already attempted; no retry')
    stat = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    argv = worker_argv(root, digest)
    write(root/'claim.json', dict(plan_sha256=digest, pid=os.getpid(), birth_ticks=int(stat[19]), argv=argv, retries=0, replacements=0))
    def stop(number, _): raise RuntimeError('Stopped diagnostic controller signal '+str(number))
    old = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        owned_child(argv, root, root, LIMITS['cpu_seconds'], LIMITS['wall_seconds'], LIMITS['address_space_bytes'])
        verify_plan(root, digest); result = read(root/'analysis/summary.json')
        require(result['complete'] is True and result['passed'] is True and result['plan_sha256'] == digest
                and result['counts']['attempts_completed'] == 73728, 'Incomplete diagnostic output')
        write(root/'summary.json', dict(complete=True, passed=True, plan_sha256=digest, analysis=reference(root/'analysis/summary.json'),
              exit=reference(root/'exit.json'), new_physical_draws=0, geometry_queries=0))
    except BaseException as error:
        write(root/'failure.json', dict(complete=False, passed=False, plan_sha256=digest, error=repr(error), partial_outputs_retained=True))
        raise
    finally:
        for s, handler in old.items(): signal.signal(s, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare', type=Path); choice.add_argument('--run', type=Path); choice.add_argument('--worker', type=Path)
    parser.add_argument('--contact-root', type=Path); parser.add_argument('--plan-sha256'); args = parser.parse_args()
    if args.prepare: print(json.dumps(prepare(args.contact_root, args.prepare)))
    elif args.run: run(args.run, args.plan_sha256)
    else: worker(args.worker, args.plan_sha256)
