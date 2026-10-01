#!/usr/bin/env python3
"""Unconditional proposal diagnostics; no physical weights or new geometry."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from prepare_hard_free_line_fresh import ARMS,jobs,read,sha,write,require
from observe_hard_free_line_fresh import PARTS,BRANCHES,branch


def read_rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def fraction(counts,denominators):
    require(len(counts)==len(denominators)==4 and all(d>0 for d in denominators),'Four unconditional populations required')
    values=[c/n for c,n in zip(counts,denominators)]
    return dict(count=sum(counts),denominator=sum(denominators),fraction=sum(counts)/sum(denominators),
                population_fractions=values,population_SE=statistics.stdev(values)/2)


def fallback_reason(draw):
    if not draw.get('fallback'):return None
    geometry=draw.get('geometry',{})
    if geometry.get('empty_reason'):return geometry['empty_reason']
    if not geometry.get('hard_free_intervals'):return 'empty_hard_free_set'
    return 'positive_geometry_mass_at_or_below_floor'


def population(directory,identity):
    directory=Path(directory);summary=read(directory/'summary.json');audit=read(directory/'independent-audit.json')
    labels=read_rows(directory/'observer/labels.jsonl');observer=read(directory/'observer/summary.json')
    records=read_rows(directory/'samples.jsonl')
    require(summary['complete'] and audit['complete'] and observer['complete'],'Incomplete population')
    require(len(records)==len(labels)==summary['samples']==observer['attempted']==256,'Missing fresh attempts')
    require(summary['probes']==audit['probes']==0 and observer['width_contacts_used'] is False,'Unexpected probes/contact sentinel')
    require(sha(directory/'observer/labels.jsonl')==observer['labels_sha256'],'Observer labels changed')
    require(observer['input_samples_sha256']==sha(directory/'samples.jsonl'),'Observer population mismatch')
    require(observer['native_classifier_calls']==observer['contact_classifier_calls']==observer['valid'],'Classifier calls mismatch')
    fallbacks=Counter();branch_counts=Counter();branch_valid=Counter();axis=Counter();valid=hard=inside=capture=0
    for i,(r,label) in enumerate(zip(records,labels)):
        require(r['id']==label['attempt_index']==i and r['kind']=='fresh','Changed attempt identity')
        require(all(label[k]==identity[k] for k in ('arm','population','seed')),'Observer stream changed')
        require(label['physical_flags']=={k:r[k] for k in ('hard_valid','shell_valid','capture_valid')},'Flags changed')
        b=branch(r);okay=r['hard_valid'] and r['shell_valid'] and r['capture_valid']
        require(label['proposal_branch']==b and label['applicable']==okay,'Branch/applicability changed')
        branch_counts[b]+=1;branch_valid[b]+=int(okay);valid+=okay
        hard+=r['hard_valid'];inside+=r['shell_valid'];capture+=r['capture_valid']
        if r['draw'].get('conditional'):
            axis[str(r['draw']['axis'])]+=1
            if r['draw'].get('fallback'):fallbacks[fallback_reason(r['draw'])]+=1
        require((label['classification'] is not None)==okay and (label['contact'] is not None)==okay,'Invalid pose was classified')
    require(valid==observer['valid']==summary['hard_capture_shell_valid'],'Valid counts disagree')
    require(sum(observer['partition'].values())==256 and sum(branch_counts.values())==256,'Lost unconditional denominator')
    return dict(**identity,attempts=256,hard_valid=hard,inside_R4=inside,capture_valid=capture,valid=valid,
        partition=observer['partition'],events=observer['events'],branches=observer['branches'],
        branch_counts={b:branch_counts[b] for b in BRANCHES},branch_valid={b:branch_valid[b] for b in BRANCHES},
        selected_axes=dict(axis),fallback_reasons=dict(fallbacks),
        conditioned_draws=summary['conditioned_draws'],fallback_draws=summary['fallback_draws'],
        proposal_CPU_seconds={k:summary[k] for k in ['draw_cpu_seconds','density_cpu_seconds','fresh_total_cpu_seconds']},
        independent_audit_CPU_seconds=audit['analysis_cpu_seconds'],
        classifier_CPU_seconds=observer['classifier_CPU_seconds'],classifier_startup_CPU_seconds=observer['classifier_startup_CPU_seconds'],
        independent_maximum_errors=audit['maximum_errors'],
        native_without_exclusion_contact_attempts=observer['native_without_exclusion_contact_attempts'],
        input_sha256={str(directory/name):sha(directory/name) for name in
            ['samples.jsonl','summary.json','attempts.jsonl','independent-audit.json','observer/labels.jsonl','observer/summary.json']})


def report(root):
    root=Path(root).resolve();protocol=read(root/'protocol.json')
    populations=[population(root/j['arm']/j['id'],dict(arm=j['arm'],population=j['id'],seed=j['seed'])) for j in jobs()]
    arms={}
    for arm in ARMS:
        selected=[p for p in populations if p['arm']==arm];denominators=[p['attempts'] for p in selected]
        metrics={k:fraction([p[k] for p in selected],denominators) for k in ['hard_valid','inside_R4','capture_valid','valid']}
        for section in ('partition','events'):
            metrics.update({k:fraction([p[section][k] for p in selected],denominators) for k in selected[0][section]})
        cpu={k:sum(p['proposal_CPU_seconds'][k] for p in selected) for k in selected[0]['proposal_CPU_seconds']}
        branch_totals={b:dict(attempts=sum(p['branch_counts'][b] for p in selected),valid=sum(p['branch_valid'][b] for p in selected)) for b in BRANCHES}
        for v in branch_totals.values():v['retention']=v['valid']/v['attempts'] if v['attempts'] else None
        reasons=Counter()
        for p in selected:reasons.update(p['fallback_reasons'])
        arms[arm]=dict(attempts=sum(denominators),metrics=metrics,branches=branch_totals,fallback_reasons=dict(reasons),
            proposal_CPU_seconds=cpu,proposal_attempts_per_CPU_second=sum(denominators)/cpu['fresh_total_cpu_seconds'],
            events_per_proposal_CPU_second={k:v['count']/cpu['fresh_total_cpu_seconds'] for k,v in metrics.items()},
            independent_audit_CPU_seconds=sum(p['independent_audit_CPU_seconds'] for p in selected),
            classifier_CPU_seconds=sum(p['classifier_CPU_seconds'] for p in selected),
            classifier_startup_CPU_seconds=sum(p['classifier_startup_CPU_seconds'] for p in selected))
    comparisons={}
    for key,b in arms['baseline']['metrics'].items():
        n=arms['xyz']['metrics'][key]
        comparisons[key]=dict(fraction_difference=n['fraction']-b['fraction'],
            combined_population_SE=math.hypot(n['population_SE'],b['population_SE']),
            proposal_event_CPU_ratio=(arms['xyz']['events_per_proposal_CPU_second'][key]/arms['baseline']['events_per_proposal_CPU_second'][key])
                if b['count'] else None)
    return dict(schema='hard-free-line-fresh-report-v1',complete=True,root=str(root),total_attempted_draws=2048,
        arms=arms,populations=populations,comparisons=comparisons,protocol_sha256=sha(root/'protocol.json'),
        native_definition_sha256=protocol['native_definition_sha256'],new_Poisson_clouds=0,new_physical_mass_estimates=0,
        physical_convergence_gates_passed=False,width_contacts_used=False,
        scope='Four independent proposal populations per arm, unconditional attempted-draw denominators. CPU ratios concern proposal generation/evaluation only, not physical weights, MH acceptance, equilibrium occupancy, effective independent contact samples or assembly.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();write(a.out,report(a.root))
