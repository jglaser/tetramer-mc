"""Metadata-only binding for one complete 128-chain conditional native audit.

No classifier, journal decoder, proposal replay, or output writer is invoked.
Completed contact receipts are the authority for the exact retained journals.
Only the future worker reads their rows. The caller freezes this returned plan
and supplies a single job to the existing generic lifecycle driver.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

from analyze_mobile_native_pocket import local_sources
import audit_native_pair_candidates as initial_audit
import run_native_class_physical_campaign as driver

read, require, sha = driver.read, driver.require, driver.sha
PLAN_SCHEMA = 'conditional-native-audit-plan-v1'
ARMS = ('local', 'm4', 'singleton_two_neighbor', 'singleton_two_neighbor_unfused')
STARTS = ('source', 'proposal_prepared')
CHECKPOINT_SEED = 'conditional-native-omitted-pairs-20261004-v1'
CHECKPOINT_RULE = ('SHA256(seed + NUL + canonical chain identity + NUL + half index); '
                   'unsigned big-endian digest modulo 2048, plus 513 or 2561. '
                   'One checkpoint in each production half; independent of trajectory contents.')
ALLOCATION = dict(chains=128, blocks=4608, warmup=512, retained_endpoints=589952,
    production_endpoints=524288, main_pair_query_cap=309724800,
    checkpoint_pair_query_cap=134400, checkpoint_endpoints=256,
    fixed_geometry_queries=0, new_physical_draws=0, arithmetic_replays=0)
LIMITS = dict(cpu_seconds=36000, wall_seconds=72000, address_space_bytes=16*1024**3,
              threads=1, max_record_bytes=64*1024**2)
SCOPE = ('All 128 predeclared conditional chains, including rejected residence; '
         'no favorable-outcome selection, new sampling, bath queries, or proposal replay. '
         'Initial fixed native keys are inherited from the completed exhaustive audit. '
         'No context pooling, equilibrium claim, or physical-kinetics interpretation.')


class Bindings:
    def __init__(self): self.files = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve(); digest = sha(path)
        require(expected is None or digest == expected, 'Changed bound input: '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == digest, 'Input changed during binding')
        self.files[str(path)] = digest
        return dict(path=str(path), sha256=digest)

    def load(self, reference):
        require(type(reference) is dict and set(reference) == {'path', 'sha256'}, 'Exact BoundFile required')
        return read(self.bind(reference['path'], reference['sha256'])['path'])

    def file(self, path, files):
        path = str(Path(path).resolve())
        require(path in files, 'Prior receipt omits input: '+path)
        return self.bind(path, files[path])


def runtime():
    return initial_audit.runtime()


def source_paths():
    """Bind both new entrypoints without importing/executing the worker."""
    found = {}
    for entry in (Path(__file__), Path(__file__).with_name('run_conditional_native_audit.py')):
        require(entry.is_file(), 'Missing native audit worker source')
        for name, path in local_sources(entry).items():
            require(name not in found or found[name] == path, 'Ambiguous local source name')
            found[name] = path
    return dict(sorted(found.items()))


def source_closure():
    return {name: sha(path) for name, path in source_paths().items()}


def identity(job):
    require(type(job) is dict and type(job.get('context_index')) is int
            and 0 <= job['context_index'] < 4 and type(job.get('stream')) is int
            and 0 <= job['stream'] < 4 and job.get('arm') in (*ARMS, 'unguided')
            and job.get('initialization') in STARTS, 'Invalid chain identity')
    return {key: job[key] for key in ('context_index', 'arm', 'initialization', 'stream')}


def identity_tuple(job):
    value = identity(job)
    return tuple(value[key] for key in ('context_index', 'arm', 'initialization', 'stream'))


def checkpoint_blocks(chain_identity):
    import json
    value = identity(chain_identity)
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return [513+2048*half+int.from_bytes(hashlib.sha256(
        (CHECKPOINT_SEED+'\0'+text+'\0'+str(half)).encode()).digest(), 'big') % 2048
        for half in range(2)]


def _successful_exit(b, root):
    value = read(b.bind(root/'exit.json')['path'])
    require(value['child_started'] is True and value['child_drained'] is True
            and type(value['returncode']) is int and value['returncode'] == 0
            and value['error'] is None, 'Contact observer child did not drain successfully')
    require(not (root/'failure.json').exists() and not (root/'preparation-failure.json').exists(),
            'Failed contact observer cannot authorize native analysis')


def _contact_receipt(b, root, *, singleton):
    root = Path(root).resolve()
    summary_ref = b.bind(root/'summary.json'); summary = b.load(summary_ref)
    require(summary['complete'] is True and summary['passed'] is True
            and summary['native_observer'] is False, 'Completed contact observer required')
    _successful_exit(b, root)
    plan_ref = b.bind(root/'execution-plan.json', summary['plan_sha256']); plan = b.load(plan_ref)
    require(plan['root'] == str(root) and plan['maximum_workers'] == 1, 'Contact execution root/worker scope differs')
    claim = read(b.bind(root/'claim.json')['path'])
    require(claim['plan_sha256'] == plan_ref['sha256'], 'Contact claim belongs to another plan')
    if singleton:
        require(plan['schema'] == 'two-neighbor-singleton-observer-execution-v1'
                and summary['schema'] == 'two-neighbor-singleton-observer-completion-v1'
                and plan['new_chains'] == plan['reused_control_chains'] == 64
                and plan['retries'] == plan['replacements'] == 0
                and plan['partial_analysis_allowed'] is False, 'Changed singleton observer allocation')
        manifest_ref = summary['manifest']
    else:
        require(plan['schema'] == 'evolving-dimer-analysis-execution-v1' and summary['chains'] == 96
                and plan['expected_chains'] == 96 and plan['restart'] is False
                and plan['partial_analysis_allowed'] is False, 'Changed original observer allocation')
        manifest_ref = dict(path=str(root/'analysis/manifest.json'), sha256=summary['manifest_sha256'])
    require(manifest_ref['path'] == str(root/'analysis/manifest.json')
            and summary['analysis']['path'] == str(root/'analysis/analysis.json'), 'Contact output path differs')
    manifest = b.load(manifest_ref); result = b.load(summary['analysis'])
    require(manifest['complete'] is True and result['complete'] is True
            and manifest['files']['analysis.json'] == summary['analysis']['sha256'], 'Contact result/manifest differs')
    require(result['schema'] == ('two-neighbor-singleton-analysis-v1' if singleton else 'conditional-dimer-analysis-v1'),
            'Wrong contact result schema')
    chains = result['chains']; expected_count = 128 if singleton else 96
    require(len(chains) == expected_count and len({identity_tuple(c['job']) for c in chains}) == expected_count,
            'Missing/repeated contact chain identity')
    if singleton:
        require(result['new_chains'] == result['reused_control_chains'] == 64
                and result['new_geometry_endpoints'] == 294976
                and result['new_physical_draws'] == result['old_geometry_queries'] == 0
                and result['native_observer'] is False, 'Unexpected singleton observer work')
        new = [c for c in chains if c.get('reused_control') is False]
        old = [c for c in chains if c.get('reused_control') is True]
        require(len(new) == len(old) == 64 and [c['job'] for c in new] == plan['new_jobs']
                and [c['job'] for c in old] == plan['control_jobs'], 'Contact jobs differ from frozen plan')
        outputs = new
    else:
        outputs = chains
    expected_files = {'analysis.json', 'input-binding.json'} | {
        f"job-{c['job']['id']:03}-observations.jsonl" for c in outputs}
    require(set(manifest['files']) == expected_files, 'Contact output inventory differs')
    # Hash the completed cache files without reading/interpreting any rows.
    for name, digest in manifest['files'].items(): b.bind(root/'analysis'/name, digest)
    input_binding = read(root/'analysis/input-binding.json')
    if singleton:
        require(input_binding['input_sha256'] == result['input_sha256']
                and input_binding['plan'] == result['analysis_plan'], 'Contact input binding differs')
        require(all(plan['files'].get(path) == digest for path, digest in result['input_sha256'].items()),
                'Singleton result has unbound inputs')
    else:
        require(all(plan['files'].get(path) == digest for path, digest in result['input_files'].items()),
                'Original result has unbound inputs')
    config_ref = b.file(Path(plan['base'])/'config.json', plan['files']); config = b.load(config_ref)
    run_ref = b.file(Path(plan['base'])/'run-binding.json', plan['files']); run_binding = b.load(run_ref)
    require(run_binding['config_sha256'] == config_ref['sha256'], 'Prior run config binding differs')
    return dict(root=root, summary=summary_ref, plan_binding=plan_ref, manifest=manifest_ref,
                analysis=summary['analysis'], plan=plan, result=result,
                config_binding=config_ref, config=config, run_binding=run_binding, run_binding_ref=run_ref)


def _initial_receipt(b, root, config_ref):
    root = Path(root).resolve()
    execution_ref = b.bind(root/'execution-plan.json'); execution = b.load(execution_ref)
    summary_ref = b.bind(root/'execution/summary.json'); completion = b.load(summary_ref)
    require(completion['complete'] is True and completion['passed'] is True
            and completion['plan_sha256'] == execution_ref['sha256']
            and completion['active'] is None and completion['unstarted'] == []
            and completion['failure'] is None and len(execution['jobs']) == len(completion['completed']) == 1,
            'Initial reference controller is incomplete')
    job = execution['jobs'][0]
    terminal_ref = driver.completed_terminal(root, execution, job['id'])
    require(completion['completed'][0]['terminal'] == terminal_ref
            and terminal_ref['path'] == str(root/'audit/summary.json'), 'Initial completion terminal differs')
    terminal = b.load(terminal_ref)
    plan_ref = b.bind(root/'audit-plan.json', terminal['plan_sha256']); plan = b.load(plan_ref)
    require(plan['schema'] == initial_audit.PLAN_SCHEMA and plan['config'] == config_ref
            and plan['allocation'] == initial_audit.ALLOCATION
            and terminal['schema'] == initial_audit.SCHEMA and terminal['complete'] is True
            and terminal['passed'] is True and terminal['allocation'] == initial_audit.ALLOCATION
            and terminal['reference_queries_begun'] == terminal['reference_queries_completed'] == 43116,
            'Wrong/incomplete exhaustive initial reference')
    require(terminal['input_sha256'] == plan['input_sha256']
            and terminal['source_sha256'] == plan['source_sha256']
            and terminal['runtime'] == plan['runtime']
            and terminal['setup_inventory'] == plan['setup_inventory'], 'Initial audit provenance differs')
    for path, digest in plan['input_sha256'].items(): b.bind(path, digest)
    b.load(plan['definition']); b.load(plan['compiled_native']); b.load(plan['witness'])
    require(plan['definition']['sha256'] == initial_audit.DEFINITION_SHA
            and plan['shape']['sha256'] == initial_audit.SHAPE_SHA
            and plan['witness']['sha256'] == initial_audit.WITNESS_SHA, 'Initial native target differs')
    states = terminal['states']
    require([s['state_id'] for s in states] == ['source']+[
        f'context-{c}-stream-{s}' for c, s in itertools.product(range(4), range(4))]
        and all(s['passed'] is True and s['second_classifier_pass'] is False for s in states)
        and states[0]['reference_pair_calls'] == 34716
        and all(s['reference_pair_calls'] == 525 for s in states[1:]), 'Initial state/reference inventory differs')
    b.bind(terminal['ledger']['path'], terminal['ledger']['sha256'])
    # Preserve all lifecycle receipts which completed_terminal authenticated.
    for path in [root/'execution/claim.json', *driver.job_directory(root, 0, job).glob('*.json')]: b.bind(path)
    return dict(plan=plan, result=terminal, bindings=dict(execution_plan=execution_ref,
                controller_summary=summary_ref, plan=plan_ref, summary=terminal_ref))


def _chain(b, chain, origin, config, starts):
    job = chain['job']; ident = identity(job); tag = identity_tuple(job)
    require(type(job.get('id')) is int and job['id'] >= 0, 'Invalid campaign-local job ID')
    directory = Path(origin['config']['output']).resolve()/f"job-{job['id']:03}"
    trajectory = chain['trajectory']
    require(trajectory['path'] == str(directory/'trajectory.jsonl'), 'Journal escaped original job directory')
    input_map = origin['result']['input_sha256' if origin['result']['schema'] == 'two-neighbor-singleton-analysis-v1' else 'input_files']
    require(input_map.get(trajectory['path']) == trajectory['sha256'], 'Prior arithmetic receipt omits journal')
    b.file(trajectory['path'], origin['plan']['files']); b.load(origin['config_binding'])
    b.bind(trajectory['path'], trajectory['sha256'])
    terminal_ref = b.file(directory/'terminal.json', input_map)
    require(origin['plan']['files'].get(terminal_ref['path']) == terminal_ref['sha256'], 'Unbound original terminal')
    terminal = b.load(terminal_ref)
    require(terminal['complete'] is True and terminal['conditional_target'] is True
            and terminal['job'] == job and terminal['blocks'] == 4608
            and terminal['trajectory'] == trajectory and terminal['counts'] == chain['counts']
            and terminal['config_sha256'] == origin['config_binding']['sha256']
            and terminal['binding_sha256'] == origin['run_binding_ref']['sha256']
            and not (directory/'failure.json').exists(), 'Original terminal/chain identity differs')
    cpu = terminal['cpu_seconds']
    require(type(cpu) in (float, int) and math.isfinite(cpu) and cpu > 0
            and cpu == chain['metrics']['full_sampler_cpu_seconds']
            and chain['metrics']['production_samples'] == 4096, 'Full CPU or retained allocation differs')
    context = config['contexts'][ident['context_index']]
    members = [context['root'], context['child']]
    initial = (dict(kind='source', frame=config['source_frame']) if ident['initialization'] == 'source'
               else dict(kind='proposal_prepared', record=starts[ident['context_index'], ident['stream']]['record']))
    return dict(id=f'c{tag[0]}-{tag[1]}-{tag[2]}-s{tag[3]}', identity=ident, job=job,
                origin_config=origin['config_binding'], trajectory=trajectory, terminal=terminal_ref,
                prior_replay_analysis=origin['analysis'], prior_replay_plan=origin['plan_binding'],
                initial=initial, members=members, full_sampler_cpu_seconds=cpu,
                checkpoint_blocks=checkpoint_blocks(ident),
                geometry_load_cpu_seconds=terminal.get('geometry_load_cpu_seconds'),
                contact_observer_cpu_seconds=chain.get('observer_cpu_seconds'))


def make_plan(contact_root, initial_audit_root, *, root):
    """Return a deterministic plan only after all contact outcomes are complete."""
    b = Bindings()
    new = _contact_receipt(b, contact_root, singleton=True)
    config = new['config']; controls = config['control_analysis']
    old = _contact_receipt(b, Path(controls['summary']['path']).parent, singleton=False)
    require(all(old[key] == controls[key] for key in ('summary', 'manifest', 'analysis')),
            'Contact controls differ from frozen benchmark')
    require(config['inherited_campaign']['config'] == old['config_binding']
            and all(old['config'][key] == config[key] for key in ('source_frame', 'shape', 'contexts', 'physical')),
            'Conditional targets differ across old/new arms')
    require(len(config['contexts']) == 4, 'Exactly four fixed conditional contexts required')
    expected = set(itertools.product(range(4), ARMS, STARTS, range(4)))
    require({identity_tuple(c['job']) for c in new['result']['chains']} == expected,
            'Require exactly the frozen 128-chain product')
    old_by_key = {identity_tuple(c['job']): c for c in old['result']['chains']}
    require(set(old_by_key) == set(itertools.product(range(4), ('local', 'unguided', 'm4'), STARTS, range(4))),
            'Original contact inventory differs')
    prepared_ref = new['run_binding']['prepared_manifest']
    require(prepared_ref == old['run_binding']['prepared_manifest']
            == config['inherited_campaign']['prepared_manifest'], 'Prepared starts differ across arms')
    prepared = b.load(prepared_ref)
    require(prepared['complete'] is True and prepared['passed'] is True
            and prepared['config_sha256'] == old['config_binding']['sha256'], 'Prepared manifest is incomplete')
    starts = {(s['context_index'], s['stream']): s for s in prepared['alternative_starts']}
    require(len(starts) == len(prepared['alternative_starts']) == 16
            and set(starts) == set(itertools.product(range(4), range(4))), 'Prepared-start inventory differs')
    frame = b.load(config['source_frame']); b.load(config['shape'])
    require(len(frame['poses']) == 264, 'Exactly 264 labelled source bodies required')
    for (context, stream), start in starts.items():
        record = start['record']; value = b.load(record)
        require(start['status'] == value['status'] == 'prepared' and value['context_index'] == context
                and value['stream'] == stream and value['is_equilibrium_sample'] is False
                and len(value['selected']) == 2, 'Invalid inherited prepared start')
        for origin in (new, old):
            require(origin['run_binding']['prepared_files'].get(record['path']) == record['sha256'], 'Start is not bound by original run')
    initial = _initial_receipt(b, initial_audit_root, new['config_binding'])
    native = initial['plan']; initial_result = initial['result']
    require(native['contexts'] == config['contexts'] and native['source_frame'] == config['source_frame']
            and native['shape'] == config['shape'] and native['starts'] == [
                dict(context_index=c, stream=s, members=[config['contexts'][c][k] for k in ('root', 'child')],
                     record=starts[c, s]['record']) for c, s in itertools.product(range(4), range(4))],
            'Initial differential audit covered different states')
    source_keys = initial_result['states'][0]['reference_native_keys']
    require(source_keys == initial_result['states'][0]['instantaneous_native_keys'], 'Initial full source key union differs')
    fixed = {str(c): [k for k in source_keys if not set(k[:2]) & {context['root'], context['child']}]
             for c, context in enumerate(config['contexts'])}
    require(fixed == initial_result['fixed_native_keys_by_context'], 'Initial fixed source keys differ')
    chains = []
    for chain in sorted(new['result']['chains'], key=lambda c: identity_tuple(c['job'])):
        control = chain['job']['arm'] in ('local', 'm4')
        require(chain['reused_control'] is control, 'Wrong control/new classification')
        if control:
            prior = old_by_key[identity_tuple(chain['job'])]
            require(all(chain[k] == prior[k] for k in ('job', 'trajectory', 'counts'))
                    and chain['metrics']['full_sampler_cpu_seconds'] == prior['metrics']['full_sampler_cpu_seconds'],
                    'Copied control no longer matches its original arithmetic receipt')
        chains.append(_chain(b, chain, old if control else new, config, starts))
    root = Path(root).resolve()
    return dict(schema=PLAN_SCHEMA, root=str(root), output=str(root/'analysis'),
        contact_root=str(new['root']), initial_audit_root=str(Path(initial_audit_root).resolve()),
        contact_receipts={label: {k: value[k] for k in ('summary', 'manifest', 'analysis', 'plan_binding')}
                          for label, value in [('singleton', new), ('controls', old)]},
        initial_audit_receipts=initial['bindings'], benchmark_config=new['config_binding'],
        native_inputs={key: native[key] for key in ('definition', 'compiled_native', 'witness', 'shape', 'source_frame')},
        setup_inventory=native['setup_inventory'], contexts=config['contexts'],
        source_native_keys=source_keys, fixed_native_keys_by_context=fixed, chains=chains,
        checkpoint_seed=CHECKPOINT_SEED, checkpoint_rule=CHECKPOINT_RULE,
        allocation=ALLOCATION, limits=LIMITS, input_sha256=b.files,
        source_sha256=source_closure(), runtime=runtime(), scope=SCOPE, launched=False,
        inherited_preparation_costs={k: old['result']['costs'].get(k) for k in
            ('cloud_cpu_seconds', 'preparation_cpu_seconds', 'preparation_geometry_load_cpu_seconds', 'cloud_cost_note')})
