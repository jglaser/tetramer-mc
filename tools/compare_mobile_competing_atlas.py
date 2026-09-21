#!/usr/bin/env python3
"""Compare eight completed mobile audits without rerunning physics or geometry."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

EDGES=((0,1),(0,2),(1,2))
SCOPE=('Descriptive preparation benchmark from one shared, deliberately selected, non-equilibrium competing snapshot. '
       'Both full atlases are native-informed; no production adaptation occurs. '
       'Native graphs use entry/retention hysteresis, distinct from instantaneous thermodynamic regions. '
       'Association or one-way escape is not proof of stationary efficiency, equilibrium populations, crystallization, or physical kinetics. '
       'An arbitrary three-body native graph is not the specific native R4 pose region integrated on the earlier fixed snapshot.')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def comparison_design(protocol):
    schema=protocol['schema']
    if schema=='matched-mobile-competing-atlas-controller-v1':
        variant,stem='augmented','mobile-competing-atlas'
        label='legacy vs augmented atlas'
        title='Does restoring the reciprocal contact proposal release the trapped configuration?'
    elif schema=='matched-mobile-reciprocal-atlas-controller-v1':
        variant,stem='reciprocal','mobile-reciprocal-atlas'
        label='legacy vs exact reciprocal atlas'
        title='Does exact reciprocal pose sampling release the trapped configuration?'
    else:
        raise ValueError('Unknown matched atlas comparison schema')
    variants=('legacy',variant)
    entries=protocol['campaigns']
    if len(entries)!=2 or {e['atlas'] for e in entries}!=set(variants):
        raise ValueError('Require exactly the declared legacy and comparison atlas')
    return dict(variants=variants,campaign_schema=stem+'-benchmark-v1',
        comparison_schema=stem+'-comparison-v1',figure_basename=stem+'-comparison',
        label=label,title=title)


def validate_grid(manifest,status,design):
    if manifest['schema']!=design['campaign_schema'] or manifest['atlas_variant'] not in design['variants']:
        raise ValueError('Wrong campaign schema/variant')
    jobs=manifest['jobs'];terminal=status['jobs']
    expected={('competing',mode,replicate) for mode in ('c0','c09') for replicate in (0,1)}
    if len(jobs)!=4 or len({j['id'] for j in jobs})!=4 or {(j['start'],j['mode'],j['replicate']) for j in jobs}!=expected:
        raise ValueError('Wrong factorial allocation')
    if len(terminal)!=4 or {j['id'] for j in terminal}!={j['id'] for j in jobs}:
        raise ValueError('Wrong terminal job identities')
    if not status['complete'] or status['running'] or not all(j['status']=='complete' and j['exit_code']==0 for j in terminal):
        raise ValueError('Incomplete physical campaign')
    return {j['id']:j for j in jobs}


def sort_runs(runs,design):
    order={name:index for index,name in enumerate(design['variants'])}
    if any(r['atlas'] not in order for r in runs):raise ValueError('Run atlas not in declared comparison')
    return sorted(runs,key=lambda r:(order[r['atlas']],r['mode'],r['replicate']))


def edge_set(row,kind):
    edges={tuple(sorted(edge)) for edge in row[kind+'_edges']}
    if not edges<=set(EDGES):raise ValueError('Invalid three-body graph')
    return edges


def event_record(row):
    return {key:row[key] for key in ('serial','sweep','source')}


def transitions(history,flags):
    if len(history)!=len(flags) or not flags:raise ValueError('Complete boolean history required')
    entries=[];exits=[]
    for i in range(1,len(flags)):
        if flags[i] and not flags[i-1]:entries.append(event_record(history[i]))
        elif flags[i-1] and not flags[i]:exits.append(event_record(history[i]))
    return dict(initial=bool(flags[0]),final=bool(flags[-1]),entries=entries,exits=exits,
        first_observed_true=event_record(history[next(i for i,v in enumerate(flags) if v)]) if any(flags) else None,
        right_censored_no_entry=not any(flags),
        completed_returns_to_initial_state=len(exits) if not flags[0] else len(entries))


def candidate_statistics(records,burn,kernels):
    result={}
    for name,subset in (('full',records),('postburn',[r for r in records if r['sweep']>burn])):
        def statistics(selected):
            if any(not math.isfinite(r['gate_log_weight']) or not math.isfinite(r['proposal_correction'])
                   or not math.isfinite(r['log_acceptance']) or r['log_acceptance']>1e-12 for r in selected):
                raise ValueError('Nonfinite or positive candidate log acceptance')
            alphas=[math.exp(min(0.,r['log_acceptance'])) for r in selected]
            return dict(hard_valid_new_native_registry_candidates=len(selected),accepted=sum(r['accepted'] for r in selected),
                positive_recorded_bath_log_factors=sum(r['gate_log_weight']>0 for r in selected),
                accepted_with_positive_recorded_bath_log_factor=sum(r['accepted'] and r['gate_log_weight']>0 for r in selected),
                maximum_recorded_conditional_alpha=max(alphas) if alphas else None,
                sum_recorded_conditional_alphas=math.fsum(alphas),
                maximum_recorded_log_acceptance=max((r['log_acceptance'] for r in selected),default=None),
                minimum_proposal_correction=min((r['proposal_correction'] for r in selected),default=None),
                maximum_proposal_correction=max((r['proposal_correction'] for r in selected),default=None))
        result[name]=dict(total=statistics(subset),by_kernel={k:statistics([r for r in subset if r['source']==k]) for k in sorted(kernels)})
    result['scope']='Only hard-valid global proposals that create at least one native registry key absent from the current hysteretic registry are recorded by the frozen observer. Poisson bath log factors are random acceptance factors, not energies/free energies. Conditional-alpha sums are descriptive sums for the realized proposals/clouds, not stationary transition rates or predicted fresh-chain rates.'
    return result


def summarize(run,variant,burn,sweeps):
    history=run['graph_history'];rows=run['rows']
    if history[0]['serial']!=-1 or history[0]['sweep']!=0:raise ValueError('Missing initial graph')
    if [r['serial'] for r in history]!=list(range(-1,len(history)-1)):raise ValueError('Missing attempted updates')
    if [r['sweep'] for r in rows]!=list(range(sweeps+1)):raise ValueError('Missing stored endpoints')
    native=[edge_set(row,'native') for row in history]
    near=[edge_set(row,'nonspecific') for row in history]
    if native[0]!={(1,2)} or near[0]!={(0,2),(1,2)}:raise ValueError('Unexpected competing starting graph')
    if any(not n<=c for n,c in zip(native,near)):raise ValueError('Native graph outside exclusion contacts')
    endpoint={row['sweep']:i for i,row in enumerate(history)}
    if set(endpoint)!=set(range(sweeps+1)):raise ValueError('Attempted updates do not cover declared sweeps')
    for row in rows:
        i=endpoint[row['sweep']]
        if {tuple(v) for v in row['native']['edges']}!=native[i] or {tuple(v) for v in row['nonspecific']['edges']}!=near[i]:
            raise ValueError('Stored endpoint graph differs from attempted-update history')
    descriptors={
        'body0_native':[any(0 in edge for edge in edges) for edges in native],
        'all_three_native_connected':[len(edges)>=2 for edges in native],
        'all_three_native_edges':[len(edges)==3 for edges in native],
        'initial_0_2_exclusion_contact':[(0,2) in edges for edges in near],
        'initial_0_2_native_registration':[(0,2) in edges for edges in native],
        'initial_scaffold_1_2_native':[(1,2) in edges for edges in native],
    }
    events={key:transitions(history,flags) for key,flags in descriptors.items()}
    # Registration of the existing contacting pair differs from its physical detachment.
    retained_registration=[dict(event,initial_contact_still_present=(0,2) in near[event['serial']+1])
        for event in events['initial_0_2_native_registration']['entries']]
    full_cpu=float(run['sampler_cpu_seconds']);post_cpu=float(run['postburn_sampler_cpu_seconds'])
    if not(math.isfinite(full_cpu) and full_cpu>0 and math.isfinite(post_cpu) and post_cpu>0):raise ValueError('Invalid sampler CPU')
    def attachment_time(event):
        if event is None:return None
        k=event['sweep'];lo=max(k-1,0)
        candidate=next((r for r in run['native_candidates'] if r['serial']==event['serial']),None)
        registry_change=next((r for r in run['changes'] if r['serial']==event['serial']),None)
        return dict(event,moving_body_from_global_candidate=None if candidate is None else candidate['moving'],
            new_registered_keys_from_global_candidate=None if candidate is None else candidate['new_registered_keys'],
            native_registry_change=registry_change,
            cumulative_sampler_CPU_bracket_seconds=[rows[lo]['sampler_cpu_seconds'],rows[k]['sampler_cpu_seconds']],
            note='Event occurs within this stored-sweep CPU bracket; no sub-sweep timing inferred.')
    partners={}
    for kind in ('native','nonspecific'):
        metric=run['graph_metrics']['graphs'][kind]
        body=next(v for v in metric['bodies'] if v['body']==0)
        partners[kind]=dict(sequential=[v for v in metric['partner_exchanges'] if v['body']==0],
            same_attempt=[v for v in metric['same_attempt_partner_replacements'] if v['body']==0],
            pending_losses=[v for v in metric['pending_partner_losses_at_end'] if v['body']==0],
            observed_environment_returns=body['episodes']['full']['observed_environment_returns'],
            postburn_environment_returns=body['episodes']['postburn']['observed_environment_returns'],
            episodes=body['episodes'],all_body_exchange_counts=dict(sequential=len(metric['partner_exchanges']),same_attempt=len(metric['same_attempt_partner_replacements'])))
    retained=[endpoint[k] for k in range(burn+1,sweeps+1)];middle=len(retained)//2
    occupancy={key:dict(full_postburn=float(np.mean([flags[i] for i in retained])),
        first_half=float(np.mean([flags[i] for i in retained[:middle]])),
        second_half=float(np.mean([flags[i] for i in retained[middle:]]))) for key,flags in descriptors.items()}
    graph_masks={kind:[sum((1<<k) for k,edge in enumerate(EDGES) if edge in graph[endpoint[s]]) for s in range(sweeps+1)]
        for kind,graph in (('native',native),('nonspecific',near))}
    rates={key:dict(full_entries_per_sampler_CPU_second=len(value['entries'])/full_cpu,
        full_exits_per_sampler_CPU_second=len(value['exits'])/full_cpu,
        postburn_entries_per_sampler_CPU_second=sum(v['sweep']>burn for v in value['entries'])/post_cpu,
        postburn_exits_per_sampler_CPU_second=sum(v['sweep']>burn for v in value['exits'])/post_cpu)
        for key,value in events.items()}
    body0_candidates=[v for v in run['native_candidates'] if v['moving']==0]
    incident_candidates=[v for v in run['native_candidates'] if any(0 in key[:2] for key in v['new_registered_keys'])]
    motif_episodes=[]
    for row in rows:
        keys=row['registered_keys']
        if motif_episodes and motif_episodes[-1]['registered_keys']==keys:motif_episodes[-1]['last_sweep']=row['sweep']
        else:motif_episodes.append(dict(first_sweep=row['sweep'],last_sweep=row['sweep'],registered_keys=keys))
    return dict(id=run['id'],atlas=variant,mode=run['mode'],replicate=run['replicate'],seed=run['seed'],
        sampler_CPU_seconds=full_cpu,postburn_sampler_CPU_seconds=post_cpu,
        attempted_updates=len(history)-1,saved_endpoints=len(rows),
        first_body0_native=attachment_time(events['body0_native']['first_observed_true']),
        first_all_three_native=attachment_time(events['all_three_native_connected']['first_observed_true']),
        first_native_triangle=attachment_time(events['all_three_native_edges']['first_observed_true']),
        first_initial_0_2_exclusion_contact_loss=attachment_time(events['initial_0_2_exclusion_contact']['exits'][0])
            if events['initial_0_2_exclusion_contact']['exits'] else None,
        events=events,registration_of_initial_contact=retained_registration,body0_partner_metrics=partners,
        postburn_occupancy=occupancy,event_rates=rates,graph_masks_at_saved_sweeps=graph_masks,
        initial_registered_keys=run['initial_registered_keys'],final_registered_keys=rows[-1]['registered_keys'],
        native_registry_changes=run['changes'],registered_motif_episodes_at_saved_sweeps=motif_episodes,
        body0_moving_native_candidate_statistics=candidate_statistics(body0_candidates,burn,run['branch_counts']),
        body0_incident_native_candidate_statistics=candidate_statistics(incident_candidates,burn,run['branch_counts']),
        body0_moving_native_candidates=body0_candidates,body0_incident_native_candidates=incident_candidates,
        branch_counts=run['branch_counts'],joint_edge_apparent_ess=run['graph_metrics']['joint_edge_apparent_ess'],
        scope=SCOPE)


def load_completed(benchmark,assessment):
    protocol=read(benchmark/'protocol.json');state=read(benchmark/'status.json')
    design=comparison_design(protocol)
    if not state['complete'] or state['phase']!='complete':
        raise ValueError('Both physical campaigns and their saved audits must be complete')
    if state['protocol_sha256']!=sha(benchmark/'protocol.json'):raise ValueError('Controller protocol binding changed')
    for name,digest in read(benchmark/'freeze.json')['files'].items():
        if sha(benchmark/name)!=digest:raise ValueError('Frozen benchmark input changed: '+name)
    summaries=[];bindings=[];configs=[];all_seeds=[]
    for entry in protocol['campaigns']:
        variant=entry['atlas'];campaign=benchmark/variant;destination=assessment/variant
        manifest,status=read(campaign/'manifest.json'),read(campaign/'status.json')
        jobs=validate_grid(manifest,status,design)
        if manifest['atlas_variant']!=variant:raise ValueError('Wrong campaign binding')
        if (manifest['sweeps'],manifest['burn_sweeps'],manifest['sample_every'])!=(protocol['sweeps'],protocol['burn_sweeps'],protocol['sample_every']):
            raise ValueError('Campaign duration differs from protocol')
        if sha(campaign/'manifest.json')!=entry['manifest_sha256']:raise ValueError('Campaign binding changed')
        result=read(destination/'analysis.json')
        if not result['complete'] or sha(destination/'analysis.json')!=state['audits'][variant]['analysis_sha256']:
            raise ValueError('Completed assessment identity changed')
        if result['manifest_sha256']!=sha(campaign/'manifest.json') or result['terminal_status_sha256']!=sha(campaign/'status.json'):
            raise ValueError('Assessment does not match physical terminal state')
        if result['analyzer_sha256']!=manifest['observer_sha256'] or result['observer_execution']!='frozen-reference-assessment':
            raise ValueError('Wrong archived observer')
        if len(jobs)!=4 or len(result['runs'])!=4 or {r['id'] for r in result['runs']}!=set(jobs):raise ValueError('Missing or extra runs')
        for run in result['runs']:
            job=jobs[run['id']]
            if not run['passed'] or any(run[k]!=job[k] for k in ('mode','start','replicate','seed')):raise ValueError('Failed/incorrect run assessment')
            for name,digest in run['source_sha256'].items():
                if sha(Path(job['directory'])/name)!=digest:raise ValueError('Audited physical source changed')
            if read(destination/'runs'/job['id']/'analysis.json')!=run:raise ValueError('Per-run assessment differs')
            cfg=read(job['config'])
            if cfg['fixed_body_indices'] or cfg['seed_labels']:raise ValueError('A physical body was fixed')
            configs.append(cfg);all_seeds.append(job['seed'])
            summaries.append(summarize(run,variant,manifest['burn_sweeps'],manifest['sweeps']))
        bindings.append(dict(atlas=variant,manifest_sha256=sha(campaign/'manifest.json'),
            physical_status_sha256=sha(campaign/'status.json'),assessment_sha256=sha(destination/'analysis.json'),
            model_sha256=manifest['model_sha256'],observer_sha256=manifest['observer_sha256']))
    if len(summaries)!=8 or len(set(all_seeds))!=8:raise ValueError('Require all eight distinct streams')
    keys=('initial_poses','fixed_body_indices','seed_labels','boundary','depletant_radius','reservoir_density',
        'poisson_lambda_ratio','global_probability','learned_uniform_weight','local_translation_std_A','local_small_angle_std_degrees',
        'gca_probability','center_shift_probability','endpoint_gate')
    if any(any(c[k]!=configs[0][k] for k in keys) for c in configs):raise ValueError('Unmatched physical/proposal controls')
    return protocol,sort_runs(summaries,design),bindings


def render(out,runs,burn,sweeps,design):
    palette=['#f1f1f4','#bdc5eb','#e1bad7','#9166ac','#75b6c1','#8cbc85','#dab15e','#2c704c']
    descriptions=['none','0–1','0–2','0–1 + 0–2','1–2','0–1 + 1–2','0–2 + 1–2','all three edges']
    fig=plt.figure(figsize=(14,10.5));grid=fig.add_gridspec(3,1,height_ratios=[2.4,2.4,2.1],hspace=.33,
        left=.20,right=.96,bottom=.09,top=.82)
    axes=[]
    labels=[f"{r['atlas'].capitalize()} · {r['mode']} · r{r['replicate']}" for r in runs]
    for slot,(kind,title) in enumerate((('native','a  Native registry graph (entry/retention hysteresis)'),
                                        ('nonspecific','b  Exclusion-contact graph (native contacts included)'))):
        ax=fig.add_subplot(grid[slot]);axes.append(ax)
        values=np.asarray([r['graph_masks_at_saved_sweeps'][kind] for r in runs])
        ax.imshow(values,aspect='auto',interpolation='nearest',cmap=ListedColormap(palette),
            norm=BoundaryNorm(np.arange(-.5,8.5),8),extent=(-.5,sweeps+.5,7.5,-.5))
        ax.set_yticks(range(8),labels,fontsize=9.5);ax.axvline(burn,color='#333333',ls='--',lw=1)
        for y in (1.5,3.5,5.5):ax.axhline(y,color='white',lw=3 if y==3.5 else 1.5)
        ax.set_title(title,loc='left',fontsize=12)
        if slot==1:ax.set_xlabel(f'MC sweeps · each stored endpoint and repeat retained · dashed line: burn {burn}')
        else:ax.tick_params(labelbottom=False)
    table_ax=fig.add_subplot(grid[2]);table_ax.axis('off')
    def when(event):return '—' if event is None else str(event['sweep'])
    values=[]
    for r in runs:
        physical=r['events']['initial_0_2_exclusion_contact']
        switches=r['body0_partner_metrics']['nonspecific']
        values.append([f"{r['atlas']} {r['mode']} r{r['replicate']}",when(r['first_body0_native']),when(r['first_all_three_native']),when(r['first_native_triangle']),
            when(r['first_initial_0_2_exclusion_contact_loss']),str(len(physical['entries'])),
            f"{len(switches['sequential'])}/{len(switches['same_attempt'])}",f"{r['postburn_occupancy']['all_three_native_connected']['full_postburn']:.3f}",
            f"{r['sampler_CPU_seconds']:.1f}"])
    table=table_ax.table(cellText=values,colLabels=['Run','0 first\nnative','First native\nconnected 3','First native\ntriangle','0–2 first\ncontact loss','0–2\nreturns','Partner\nswitches*','Connected 3\npostburn','Full CPU\nseconds'],
        cellLoc='center',colLoc='center',bbox=[-.17,.0,1.17,1.0],colWidths=[.19,.08,.105,.095,.105,.065,.085,.105,.085])
    table.auto_set_font_size(False);table.set_fontsize(8.5)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#d4d6dd');cell.set_linewidth(.4)
        if row==0:cell.set_facecolor('#e9edf2');cell.set_text_props(weight='bold')
        elif row>=5:cell.set_facecolor('#f3f7f4')
    fig.suptitle(design['title'],
        x=.06,y=.975,ha='left',fontsize=16,weight='bold')
    fig.text(.06,.94,'Eight fixed-duration runs · same three mobile tetramers · rᵈ = 1.5 Å, z = 0.035 Å⁻³ · '+design['label'],fontsize=11)
    fig.text(.06,.905,'Colors list the complete graph edges. Body 0 is initially unregistered; bodies 1–2 form the native scaffold.',fontsize=10,color='#444444')
    fig.legend(handles=[Patch(fc=color,label=label) for color,label in zip(palette,descriptions)],ncol=8,
        loc='upper left',bbox_to_anchor=(.055,.886),frameon=False,fontsize=9,columnspacing=1.25,handlelength=1.2)
    fig.text(.06,.05,'Event columns use all attempted updates; numbers are sweeps, “—” means not observed. Returns require an observed loss first. '
        '*Body 0 exclusion-partner switches: sequential / same update.',fontsize=9)
    fig.text(.06,.022,'One selected non-equilibrium start; both atlases are native-informed. Native connectivity and apparent persistence do not establish equilibrium or crystallization.',fontsize=9,color='#444444')
    for ext in ('png','svg','pdf'):fig.savefig(out/f"{design['figure_basename']}.{ext}",dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('benchmark','assessment','out'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();benchmark,assessment,out=[p.resolve() for p in (args.benchmark,args.assessment,args.out)]
    if out.exists():raise ValueError('Use a fresh comparison artifact directory')
    protocol,runs,bindings=load_completed(benchmark,assessment)
    design=comparison_design(protocol)
    result=dict(schema=design['comparison_schema'],complete=True,
        protocol_sha256=sha(benchmark/'protocol.json'),controller_status_sha256=sha(benchmark/'status.json'),
        inputs=bindings,runs=runs,scope=SCOPE,
        definitions=dict(graph_mask='Bits 0,1,2 encode unordered edges 0–1,0–2,1–2 respectively.',
            initial_contact_loss='The initially present 0–2 edge disappears from the exact exclusion-contact graph; loss of native registration alone is not physical contact loss.',
            registration='A native 0–2 graph edge can form while the original exclusion contact persists; report this separately.',
            event_time='Attempted-update serial, sweep and source retained. First-event CPU is bracketed by saved cumulative sweep CPU endpoints.',
            returns='Completed return to a descriptor’s initial Boolean state requires leaving it first. Physical 0–2 returns are reattachments after an observed contact loss.',
            efficiency='Event/CPU ratios are finite preparation-event diagnostics. No pooled stationary ESS or equilibrium speedup is inferred.'))
    out.mkdir(parents=True);write(out/'comparison.json',result)
    render(out,runs,protocol['burn_sweeps'],protocol['sweeps'],design)
    archive=out/'provenance';archive.mkdir();shutil.copy2(__file__,archive/'compare_mobile_competing_atlas.py')
    shutil.copy2(benchmark/'protocol.json',archive/'protocol.json');shutil.copy2(benchmark/'status.json',archive/'controller-status.json')
    write(out/'provenance.json',dict(complete=True,tool_sha256=sha(__file__),source_bindings=bindings,
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        archived_sha256={p.name:sha(p) for p in archive.iterdir()},scope=SCOPE))
    print(out/(design['figure_basename']+'.png'))


if __name__=='__main__':main()
