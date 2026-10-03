#!/usr/bin/env python3
"""Matched anchor-guided extension; reuse completed controls without geometry.

Only new retained endpoints may reach the frozen ConditionalObserver. Control
metrics are reused; new external-only descriptors are computed from its cached
observations. None of these finite-record descriptors certifies equilibrium.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import analyze_evolving_dimer_benchmark as previous

require, read, sha, canonical, write = (previous.require, previous.read, previous.sha,
                                      previous.canonical, previous.write)


def analysis_plan():
    return dict(schema='root-guided-dimer-analysis-plan-v1',
        source_files=['tools/analyze_root_guided_dimer_benchmark.py',
                      'tools/test_analyze_root_guided_dimer_benchmark.py']+previous.SOURCE_FILES,
        new_chains=32, reused_control_chains=32,
        new_retained_initial_observations=32*4609, new_production_observations=32*4096,
        maximum_mobile_related_pairs_per_endpoint=525,
        maximum_new_pair_classifications=32*4609*525,
        original_plan=previous.analysis_plan(),
        controls='Reuse completed m4 metrics and exact hash-bound cached observations; never rerun controls or their atom geometry.',
        pairing='Same context, initialization and stream. Local, proposal, internal-threshold, bath and acceptance roles shared; root threshold alone uses an extra independent role. Within-arm streams remain independent.',
        external_metrics='Exclude the mobile-mobile edge. Compute external edge-presence ESS, whole external-edge-set passages/returns, unique external neighbor labels, and any external-contact occupancy from every production endpoint including rejected residence.',
        external_interpretation='An external set change can be contact gain/loss; it is not necessarily a new binding partner, metastable basin transition, or native registry. Empty external sets are retained. Same mobile-pair attachment/detachment contributes no external passage.',
        target='Same two-mobile conditional target, fixed spectators, radius1.4Å/activity0.0275Å^-3. No original-condition, full-system or native stability inference.',
        new_geometry_only=True, new_physical_draws=0, native_observer=False,
        complete_inventory_required=True, trajectories_concatenated=False,
        comparison='Per-context, per-initialization, per-stream rates and paired differences; retain undefined ESS for constants, full sampler CPU including warmup, initial-condition agreement, and all unvisited environments.')


def external_metrics(trace, warmup, cpu, members):
    """Pure reduction of already classified rows; no geometry or pose access."""
    selected = set(members)
    require(len(selected) == 2, 'Two distinct mobile labels required')
    production = [row for row in trace if row['block'] > warmup]
    require(production, 'No production external observations')
    values = []
    for row in production:
        edges = [tuple(e) for e in row['external_edges']]
        require(len(set(edges)) == len(edges), 'Repeated external edge')
        require(all(len(e)==2 and e[0]<e[1] and len(set(e)&selected)==1 for e in edges),
                'External edge must join exactly one mobile label')
        expected = {tuple(e) for e in row['partner_edges'] if set(e) != selected}
        require(set(edges)==expected, 'External edge partition differs')
        values.append(sorted(edges))
    matrix, columns = previous.presence_matrix(values)
    labels = [previous.key(edges) for edges in values]
    neighbors = sorted({i for edges in values for e in edges for i in e if i not in selected})
    return dict(ess=previous.apparent_effective_count(matrix, cpu),
        environments=previous.categorical_summary(labels, [r['block'] for r in production]),
        occupancy=[dict(edge=edge, fraction=float(matrix[:,i].mean())) for i,edge in enumerate(columns)],
        any_contact_fraction=sum(bool(e) for e in values)/len(values),
        unique_external_neighbor_labels=neighbors,
        interpretation=analysis_plan()['external_interpretation'])


def comparison_summaries(chains):
    """Keep target/stream pairing and add external-only finite-record contrasts.

    Rates use the same full sampler CPU as ESS, including warmup and rejected
    work. Differences are right minus left. A constant descriptor's undefined
    ESS remains undefined; absence of observed exchanges is an observed zero,
    not a bound on equilibrium exchange or a claim of convergence.
    """
    result = previous.comparison_summaries(chains)
    def identity(job):
        return tuple(job[k] for k in ('context_index', 'arm', 'initialization', 'stream'))
    indexed = {identity(c['job']): c for c in chains}
    require(len(indexed) == len(chains), 'Duplicate comparison stream')
    def summary(chain):
        m = chain['metrics']; external = m['external_only']
        cpu = m['full_sampler_cpu_seconds']
        require(math.isfinite(cpu) and cpu > 0, 'Invalid external rate CPU')
        env = external['environments']
        passages, returns = env['completed_passages'], len(env['completed_returns'])
        return dict(apparent_ess=external['ess']['apparent_ess'],
            apparent_ess_per_sampling_CPU_second=external['ess']['apparent_ess_per_sampling_CPU_second'],
            full_sampler_cpu_seconds=cpu,
            any_contact_fraction=external['any_contact_fraction'],
            completed_passages=passages, completed_returns=returns,
            passages_per_sampling_CPU_second=passages/cpu,
            returns_per_sampling_CPU_second=returns/cpu,
            unique_external_neighbor_labels=external['unique_external_neighbor_labels'])
    def difference(left, right):
        if left is None or right is None:
            return None
        require(math.isfinite(left) and math.isfinite(right), 'Nonfinite external comparison')
        return right-left
    for group in result['groups']:
        for entry in group['independent_stream_metrics']:
            job = dict(group, stream=entry['stream'])
            entry['external_only'] = summary(indexed[identity(job)])
    for pair in result['descriptive_paired_comparisons']:
        left, right = (indexed[identity(pair[side])] for side in ('left', 'right'))
        a, b = (c['metrics']['external_only'] for c in (left, right))
        sa, sb = summary(left), summary(right)
        pair['control_global_rng_roles_paired'] = pair['left']['initialization'] == pair['right']['initialization']
        pair['external_only'] = dict(left=sa, right=sb, difference_direction='right minus left',
            environment_total_variation=previous.total_variation(
                a['environments']['occupancy'], b['environments']['occupancy']),
            edge_occupancy_max_difference=previous.occupancy_difference(a['occupancy'], b['occupancy'], 'edge'),
            any_contact_fraction_difference=sb['any_contact_fraction']-sa['any_contact_fraction'],
            apparent_ess_per_sampling_CPU_second_difference=difference(
                sa['apparent_ess_per_sampling_CPU_second'], sb['apparent_ess_per_sampling_CPU_second']),
            passages_per_sampling_CPU_second_difference=(
                sb['passages_per_sampling_CPU_second']-sa['passages_per_sampling_CPU_second']),
            returns_per_sampling_CPU_second_difference=(
                sb['returns_per_sampling_CPU_second']-sa['returns_per_sampling_CPU_second']),
            interpretation=analysis_plan()['external_interpretation'])
    return result


def checked(value):
    path=Path(value['path']).resolve()
    require(sha(path)==value['sha256'], 'Changed input '+str(path))
    return path


def read_cached_trace(path, expected_sha, warmup=512, blocks=4608):
    require(sha(path)==expected_sha, 'Changed cached observations')
    rows=[]
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            require(line.endswith('\n'), 'Truncated cached observation')
            row=json.loads(line)
            require(row['block']==index and row['production'] is (index>warmup),
                    'Cached observation sequence differs')
            rows.append(row)
    require(len(rows)==blocks+1, 'Missing cached retained endpoints')
    return rows


def analyze(base, output):
    from prepare_evolving_dimer_root_extension import verify
    base, output = Path(base).resolve(), Path(output).resolve()
    config = verify(base)
    require(not output.exists(), 'Fresh analysis output required')
    require(read(base/'analysis-plan.json') == analysis_plan(), 'Changed frozen analysis plan')
    protocol = read(base/'protocol.json')
    for name in analysis_plan()['source_files']:
        require(sha(Path(__file__).parents[1]/name)==protocol['source_files'][name],
                'Observer source changed '+name)
    binding = read(base/'run-binding.json')
    require(binding['config_sha256']==sha(base/'config.json'), 'Run binding differs')
    prepared = read(checked(binding['prepared_manifest']))
    controls = config['control_analysis']
    control_summary = read(checked(controls['summary']))
    control_manifest = read(checked(controls['manifest']))
    old = read(checked(controls['analysis']))
    require(control_summary['complete'] and control_summary['passed'] and control_manifest['complete']
            and old['complete'] and len(old['chains'])==96, 'Controls are incomplete')
    require(control_summary['analysis']['sha256']==controls['analysis']['sha256']
            and control_summary['manifest_sha256']==controls['manifest']['sha256'],
            'Control receipt chain differs')
    old_m4 = {c['job']['id']:c for c in old['chains'] if c['job']['arm']=='m4'}
    require(len(old_m4)==32, 'Exactly32 completed m4 controls required')
    mapping = {m['job_id']:m for m in config['control_mapping']}
    require(len(mapping)==len(config['jobs'])==32
            and {m['control_job_id'] for m in mapping.values()}==set(old_m4), 'Changed pair inventory')
    inputs = {str(base/name):sha(base/name) for name in
              ['config.json','run-binding.json','analysis-plan.json','protocol.json']}
    items = []
    # Bind all32 complete trajectories and cached controls before geometry.
    for job in config['jobs']:
        require(job['arm']=='root_m4', 'Unexpected new arm')
        link = mapping[job['id']]; control = old_m4[link['control_job_id']]
        require(all(job[k]==control['job'][k]==link[k] for k in
                    ['context_index','initialization','stream']), 'Unmatched control preparation')
        terminal_path=Path(config['output'])/f"job-{job['id']:03}"/'terminal.json'
        terminal=read(terminal_path); trajectory=checked(terminal['trajectory'])
        require(terminal['complete'] is True and terminal['conditional_target'] is True
                and terminal['job']==job and terminal['blocks']==4608
                and terminal['config_sha256']==inputs[str(base/'config.json')]
                and terminal['binding_sha256']==inputs[str(base/'run-binding.json')], 'New chain is incomplete or mismatched')
        require(not (terminal_path.parent/'failure.json').exists(), 'Failed chain cannot be analyzed as complete')
        cached = controls['observations'][str(control['job']['id'])]
        cache_path=checked(cached)
        require(control_manifest['files'][cache_path.name]==cached['sha256'], 'Cached control lacks original observation binding')
        for path in [terminal_path,trajectory,cache_path]: inputs[str(path)]=sha(path)
        items.append((job,terminal,trajectory,control,cache_path))
    for label in ('summary','manifest','analysis'):
        path=checked(controls[label]); inputs[str(path)]=controls[label]['sha256']
    shape=read(checked(config['shape'])); patches=read(checked(config['patch_map']))
    source=read(checked(config['source_frame']))['poses']
    require(patches['shape_sha256']==config['shape']['sha256']
            and len(patches['atom_patch_ids'])==len(shape['atoms']), 'Patch map differs')
    observer_config=dict(boundary=dict(kind='spherical',radius=config['physical']['wall_radius']),
                         box_lengths=[2*config['physical']['wall_radius']]*3,
                         depletant_radius=config['physical']['depletant_radius'])
    warmup=config['allocation']['warmup_blocks']; output.mkdir()
    write(output/'input-binding.json',dict(input_sha256=inputs, plan=analysis_plan(),
        source_sha256={n:protocol['source_files'][n] for n in analysis_plan()['source_files']}))
    started=time.process_time(); chains=[]; endpoint_count=0
    for job,terminal,trajectory,control,cache_path in items:
        tick=time.process_time(); context=config['contexts'][job['context_index']]
        members=[context[k] for k in ('root','child')]
        initial=[source[i] for i in members]
        if job['initialization']=='proposal_prepared':
            start=next(s for s in prepared['alternative_starts'] if
                       s['context_index']==job['context_index'] and s['stream']==job['stream'])
            initial=read(checked(start['record']))['selected']
        with trajectory.open() as stream:
            points=previous.validate_journal((json.loads(line) for line in stream),config,job,initial,terminal)
        observer=previous.ConditionalObserver(shape,patches['atom_patch_ids'],observer_config,source,members)
        trace=[]; prior=None
        with (output/f"job-{job['id']:03}-observations.jsonl").open('x') as handle:
            for point in points:
                observation=observer.classify(point['selected']); tokens=set(observation['patch_tokens'])
                previous_tokens=set(prior['patch_tokens']) if prior else tokens; union=previous_tokens|tokens
                observation.update(block=point['block'],production=point['block']>warmup,
                    sampler_cpu_seconds=point['sampler_cpu_seconds'],
                    jaccard_from_previous=1-len(previous_tokens&tokens)/len(union) if union else 0.)
                handle.write(canonical(observation)+'\n'); trace.append(observation); prior=observation
        endpoint_count+=len(trace)
        metrics=previous.summarize_trace(trace,warmup,terminal['cpu_seconds'])
        metrics['production_cpu_seconds']=terminal['cpu_seconds']-points[warmup]['sampler_cpu_seconds']
        metrics['external_only']=external_metrics(trace,warmup,terminal['cpu_seconds'],members)
        chains.append(dict(job=job,metrics=metrics,reused_control=False,observer_cpu_seconds=time.process_time()-tick,
            pair_classifications=observer.calls,exact_pair_cache_hits=observer.hits,
            trajectory=terminal['trajectory'],counts=terminal['counts'],
            geometry_load_cpu_seconds=terminal.get('geometry_load_cpu_seconds')))
        old_trace=read_cached_trace(cache_path,inputs[str(cache_path)],warmup)
        reused=copy.deepcopy(control); reused['reused_control']=True
        reused['metrics']['external_only']=external_metrics(old_trace,warmup,control['metrics']['full_sampler_cpu_seconds'],members)
        reused['new_geometry_queries']=0
        chains.append(reused)
    require(endpoint_count==analysis_plan()['new_retained_initial_observations'], 'Incomplete endpoint allocation')
    for path,digest in inputs.items(): require(sha(path)==digest, 'Input changed during analysis '+path)
    new=[c for c in chains if not c['reused_control']]
    comparisons=comparison_summaries(chains)
    costs=dict(new_sampler_cpu_seconds=math.fsum(c['metrics']['full_sampler_cpu_seconds'] for c in new),
        reused_control_sampler_cpu_seconds=math.fsum(c['metrics']['full_sampler_cpu_seconds'] for c in chains if c['reused_control']),
        new_geometry_load_cpu_seconds=math.fsum(c['geometry_load_cpu_seconds'] for c in new)
            if all(c['geometry_load_cpu_seconds'] is not None for c in new) else None,
        new_analysis_cpu_seconds=time.process_time()-started,
        note='Rates retain each actual full sampler CPU. Shared inherited cloud/start preparation was not repeated; original campaign setup/observer costs remain in its bound analysis, not silently set to zero or charged twice.')
    result=dict(schema='root-guided-dimer-analysis-v1',complete=True,new_chains=32,reused_control_chains=32,
        chains=chains,comparisons=comparisons,costs=costs,input_sha256=inputs,
        new_geometry_endpoints=endpoint_count,new_physical_draws=0,native_observer=False,
        analysis_plan=analysis_plan())
    write(output/'analysis.json',result)
    write(output/'manifest.json',dict(complete=True,files={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=analyze(args.base,args.output)
    print(json.dumps(dict(complete=True,new_chains=result['new_chains'],reused_control_chains=result['reused_control_chains'])))
