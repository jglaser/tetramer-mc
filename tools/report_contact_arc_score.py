#!/usr/bin/env python3
"""Summarize archived-probe arc scores, never infer new physical weights."""
from collections import Counter
import math
from pathlib import Path
import statistics
from prepare_contact_arc_score import ARMS, OLD_ARMS, read, rows, require


def logsumexp(values):
    largest=max(values)
    return largest+math.log(sum(math.exp(v-largest) for v in values))


def moment(logterms):
    total=logsumexp(logterms)
    return dict(log_second_moment=total,contribution_ESS=math.exp(2*total-logsumexp([2*t for t in logterms])),
                largest_contribution=math.exp(max(logterms)-total),rows=len(logterms))


def report(preparation, execution):
    preparation,execution=Path(preparation),Path(execution)
    old=read(preparation/'saved-scores.json')['rows'];new=rows(execution/'rust/scores.jsonl')
    require(len(old)==len(new)==206 and [r['id'] for r in new]==[r['id'] for r in old],
            'Score report allocation differs')
    maxima=dict(old_distance_logq=0.,baseline_logq=0.,physical_logJ=0.)
    for a,b in zip(old,new):
        require(a['latent']==b['latent'],'Score report pose changed')
        for key in ('hard_valid','shell_valid','capture_valid'):
            require(a['saved_geometry'][key]==b[key],'Saved pose flags changed')
        maxima['physical_logJ']=max(maxima['physical_logJ'],abs(a['saved_geometry']['log_physical_jacobian']-b['log_physical_jacobian']))
        require(len(b['arms'])==2 and [r['arm_index'] for r in b['arms']]==[0,1],'Score arm order changed')
        for i,arm in enumerate(ARMS):
            result=b['arms'][i]
            maxima['old_distance_logq']=max(maxima['old_distance_logq'],abs(result['old_distance_log_density']-a['old_log_q'][arm]))
            maxima['baseline_logq']=max(maxima['baseline_logq'],abs(result['baseline_log_density']-a['old_log_q']['baseline92']))
            require(result['log_proposal_density']>=a['old_log_q']['baseline92']-math.log(2)-1e-10,
                    'Old92 defensive support bound violated')
    require(max(maxima.values())<1e-9,'Saved q/J reconstruction mismatch')
    diagnostics={}
    for source_arm,count in [('baseline',4),('expanded',74)]:
        selected=[i for i,r in enumerate(old) if r['group']=='critical' and r['metadata']['source']['arm']==source_arm]
        require(len(selected)==count,'Critical source partition changed')
        values={name:[] for name in [*OLD_ARMS,*['arc_'+a for a in ARMS]]}
        for i in selected:
            r=old[i];source=r['metadata']['source']
            numerator=sum(source['paired_log_weights'])+source['log_q']-math.log(source['source_attempted_draws'])
            for arm in OLD_ARMS:values[arm].append(numerator-r['old_log_q'][arm])
            for j,arm in enumerate(ARMS):values['arc_'+arm].append(numerator-new[i]['arms'][j]['log_proposal_density'])
        m={key:moment(v) for key,v in values.items()}
        for arm in ARMS:
            candidate=m['arc_'+arm]
            candidate['second_moment_ratio_vs_old92']=math.exp(candidate['log_second_moment']-m['baseline92']['log_second_moment'])
            candidate['second_moment_ratio_vs_old_distance']=math.exp(candidate['log_second_moment']-m[arm]['log_second_moment'])
        diagnostics[source_arm]=dict(original_source_components=84 if source_arm=='baseline' else 92,
            rows=count,laws=m,scope='Previously selected critical competing-orthant55 subset only; low contribution ESS prevents a reliable scale-up prediction.')
    breadth={}
    for kind in ('native_R5','native_complement','competing','invalid'):
        selected=[i for i,r in enumerate(old) if r['group']=='breadth' and r['metadata']['coverage_class']==kind]
        require(len(selected)==32,'Breadth class count changed')
        breadth[kind]={}
        for j,arm in enumerate(ARMS):
            ratios={name:[new[i]['arms'][j]['log_proposal_density']-old[i]['old_log_q'][name] for i in selected]
                    for name in ('baseline92',arm)}
            breadth[kind][arm]={name:dict(minimum=min(v),median=statistics.median(v),maximum=max(v)) for name,v in ratios.items()}
    arm_stats={}
    for j,arm in enumerate(ARMS):
        branches=[w for row in new for component in row['arms'][j]['components'] for w in component['widths']]
        active=[b for b in branches if 'arc_mass' in b]
        incremental_geometry=0.;required_geometry=0.
        for row in new:
            cached={c['id']:c['cpu_seconds'] for c in row['geometry_cache']['circles']}
            ids={w['geometry_id'] for c in row['arms'][j]['components'] for w in c['widths'] if 'geometry_id' in w}
            previous={w['geometry_id'] for a in row['arms'][:j] for c in a['components'] for w in c['widths'] if 'geometry_id' in w}
            require(ids.issubset(cached),'Required geometry absent from saved cache')
            required_geometry+=sum(cached[k] for k in ids)
            incremental_geometry+=sum(cached[k] for k in ids-previous)
        score_cpu=sum(row['arms'][j]['score_cpu_seconds'] for row in new)
        nongeometry=score_cpu-incremental_geometry
        require(nongeometry>=-1e-9,'Incremental geometry CPU exceeds containing score timer')
        arm_stats[arm]=dict(complete_densities=206,component_width_branches=len(branches),
            geometrically_supported_branches=len(active),arc_floor_fallback_branches=sum(b['arc_fallback'] for b in active),
            query_phi_excluded_branches=sum(not b['query_phi_allowed'] for b in active),
            outer_fallback_reasons=dict(Counter(b['fallback_reason'] for b in branches if b.get('fallback'))),
            minimum_positive_arc_mass=min((b['arc_mass'] for b in active if b['arc_mass']>0),default=None),
            score_CPU_seconds=score_cpu,incremental_geometry_CPU_seconds=incremental_geometry,
            required_geometry_CPU_seconds=required_geometry,nongeometry_CPU_seconds=nongeometry,
            required_geometry_plus_nongeometry_CPU_accounting_proxy=required_geometry+nongeometry,
            note='CLI-first arm includes shared geometry construction; second arm mostly reuses it. Required geometry plus nongeometry is accounting only, not an independently measured arm run. Joint scoring CPU is the primary measured cost.')
    summary=read(execution/'rust/summary.json')
    return dict(schema='contact-arc-score-review-v1',complete=True,queries=206,candidate_densities=412,
        critical_source_diagnostics=diagnostics,breadth=breadth,arms=arm_stats,
        saved_density_reconstruction_maxima=maxima,
        distinct_circles=summary['distinct_circles'],geometry_CPU_seconds=summary['geometry_cpu_seconds'],
        combined_score_CPU_seconds=summary['score_cpu_seconds'],total_Rust_CPU_seconds=summary['cpu_seconds'],
        full_score_ms_per_unique_query=1000*summary['score_cpu_seconds']/206,
        new_pose_draws=0,new_Poisson_clouds=0,new_physical_mass_estimates=0,
        scope='Retrospective complete-density scores on all frozen probes; no new draw, throughput test, physical normalizer or equilibrium conclusion.',
        physical_gates_unchanged=True,physical_campaign_ready=False)
