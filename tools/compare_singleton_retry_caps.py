#!/usr/bin/env python3
"""Audit completed independent-redraw caps against retained-pool transport.

Passive analysis only. Every event and raw candidate is counted; native observer
results are imported from the frozen controller's single completed scan.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil


def read(path): return json.loads(Path(path).read_text())
def require(ok, message):
    if not ok: raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def quantiles(values):
    a=sorted(values)
    if not a:return {'count':0}
    def at(f):
        t=f*(len(a)-1); i=int(t)
        return a[i]+(t-i)*(a[min(i+1,len(a)-1)]-a[i])
    return dict(count=len(a),minimum=a[0],q10=at(.1),median=at(.5),q90=at(.9),maximum=a[-1],mean=math.fsum(a)/len(a))
def finite_number(value):return isinstance(value,(int,float)) and math.isfinite(value)
def close(a,b):return abs(a-b)<=1e-9*(1+abs(a)+abs(b))


def audit_event(row):
    """Raise on lost trials, retry-after-success, inconsistent rejection or ratio."""
    require(row['kind']=='cluster_event' and len(row['members'])==1,'Expected singleton event')
    p=row['proposal']; branch=p['branch']; independent=branch=='independent_conditioned'
    if not row['accepted']:
        require(row['old_poses']==row['retained_poses'],'Rejected state was not retained')
    else:
        require(row['hard_valid'] and row['proposed_poses']==row['retained_poses'],'Invalid accepted endpoint')
    raw=[]
    if independent:
        require('source_latent' not in p and 'source' not in p,'Independent branch unexpectedly sampled a source')
        raw=p.get('raw_trials'); cap=p['max_trials']; n=p['raw_trial_count']
        require(isinstance(raw,list) and len(raw)==n and 0<=n<=cap and cap>0,'Raw allocation/count mismatch')
        for k,r in enumerate(raw):
            require(r['trial']==k+1,'Missing or unordered raw trial')
            require('target_label' in r and len(r['target_latent'])==6,'Raw label/latent missing')
            require(not r['hard_valid'] or r['candidate'] is not None,'Valid raw trial has no pose')
            if k+1<len(raw):require(not r['hard_valid'],'Retried after first hard-valid candidate')
        if p['cap_exhausted']:
            require(n==cap and all(not r['hard_valid'] for r in raw),'Incorrect exhausted cap')
            require(row['proposed_poses'] is None and not row['accepted'],'Exhausted cap produced a candidate')
        elif row['proposed_poses'] is not None:
            require(raw and raw[-1]['hard_valid'] and row['hard_valid'],'Candidate is not first hard-valid')
            require(row['proposed_poses']==[raw[-1]['candidate']],'Candidate differs from retained raw trial')
        else:
            # Density/representation failures after a valid raw draw, or before
            # any draw, are explicit null events, never implicit retries.
            require(p.get('null_reason') and (not raw or raw[-1]['hard_valid']), 'Unexplained non-cap null')
    if branch in ('uniform','local_rigid'):
        require(p.get('anchor_pool') is None and p['log_reverse_forward']==0,'Unused pool correction on uniform branch')
    if row.get('gate') is not None and finite_number(p.get('log_reverse_forward')):
        old=next((p[k] for k in ('full_old_log_density','full_old_member_log_density') if finite_number(p.get(k))),None)
        new=next((p[k] for k in ('full_new_log_density','full_new_member_log_density') if finite_number(p.get(k))),None)
        if old is not None and new is not None:
            require(close(p['map_log_reverse_forward'],old-new),'Full guide-density correction mismatch')
        if p.get('pool_forward_log_probability') is not None:
            pool=p['pool_reverse_log_probability']-p['pool_forward_log_probability']
            require(close(pool,p['anchor_log_reverse_forward']),'Ordered-pool correction mismatch')
            require(close(p['map_log_reverse_forward']+pool,p['log_reverse_forward']),'Pool correction missing or counted twice')
        expected=min(0.,row['gate']['log_weight']+p['log_reverse_forward'])
        require(close(expected,row['log_acceptance']),'Physical acceptance log does not reconcile')
    return raw


def extract(row, field):
    p=row['proposal']
    if field=='map':return 0. if p['branch']=='local_rigid' else p.get('map_log_reverse_forward')
    if field=='pool':return 0. if p['branch']=='local_rigid' else p.get('anchor_log_reverse_forward')
    if field=='total_proposal':return p.get('log_reverse_forward')
    if field=='gate':return (row.get('gate') or {}).get('log_weight')
    if field=='log_acceptance':return row.get('log_acceptance')
    if field=='old_log_G':return p.get('full_old_log_density',p.get('full_old_member_log_density'))
    if field=='new_log_G':return p.get('full_new_log_density',p.get('full_new_member_log_density'))
    raise ValueError(field)


def distributions(rows):
    return {k:quantiles([x for r in rows if finite_number(x:=extract(r,k))])
            for k in ('old_log_G','new_log_G','map','pool','total_proposal','gate','log_acceptance')}


def summarize(rows):
    c=Counter({k:0 for k in ('attempted','learned','independent','uniform','hard_valid','learned_valid_candidates',
        'local','local_valid_candidates','measured_channel_events','measured_valid_candidates','accepted','accepted_learned','contact_changes','exchanges','attachments','detachments','proposal_nulls',
        'raw_trials','raw_single_trials','raw_fused_trials','raw_single_valid','raw_fused_valid','raw_hard_failures','raw_decode_nulls','raw_first_valid','cap_exhaustions','post_valid_nulls','pre_draw_nulls',
        'fused_available','fused_target','both_actual_neighbors','eligible_multi_neighbor')})
    catalogue=0.;raw_counts=[]
    learned_rows=[]; measured_rows=[]
    for row in rows:
        raw=audit_event(row);p=row['proposal'];ind=p['branch']=='independent_conditioned'
        learned=p['branch'] in ('involution','independent_conditioned'); local=p['branch']=='local_rigid'
        c.update(attempted=1,learned=int(learned),independent=int(ind),uniform=int(p['branch']=='uniform'),
            hard_valid=int(row['hard_valid']),learned_valid_candidates=int(learned and row['hard_valid']),
            local=int(local),local_valid_candidates=int(local and row['hard_valid']),
            measured_channel_events=int(learned or local),measured_valid_candidates=int((learned or local) and row['hard_valid']),
            accepted=int(row['accepted']),accepted_learned=int(learned and row['accepted']),proposal_nulls=int(row['proposed_poses'] is None))
        if learned or local:measured_rows.append(row)
        if learned:
            learned_rows.append(row)
            degree=len(row['diagnostic']['external_partners']);both=len(row['diagnostic']['pool_direct_partners'])>=2
            c.update(eligible_multi_neighbor=int(degree>=2),both_actual_neighbors=int(both))
        if ind:
            c.update(raw_trials=len(raw),
                raw_single_trials=sum(r['target_label'].get('kind')=='single' for r in raw),
                raw_fused_trials=sum(r['target_label'].get('kind')=='fused' for r in raw),
                raw_single_valid=sum(r['hard_valid'] and r['target_label'].get('kind')=='single' for r in raw),
                raw_fused_valid=sum(r['hard_valid'] and r['target_label'].get('kind')=='fused' for r in raw),
                raw_hard_failures=sum(not r['hard_valid'] for r in raw),
                raw_decode_nulls=sum(r['candidate'] is None for r in raw),raw_first_valid=sum(r['hard_valid'] for r in raw),
                cap_exhaustions=int(p['cap_exhausted']),post_valid_nulls=int(bool(raw) and raw[-1]['hard_valid'] and row['proposed_poses'] is None),
                pre_draw_nulls=int(not raw))
            raw_counts.append(len(raw))
        if p.get('oligomer'):
            o=p['oligomer'];c['fused_available']+=o['fused_components']>0
            c['fused_target']+=o.get('target_label',p.get('target_label',{})).get('kind')=='fused'
            catalogue+=sum(o['build_seconds'])
        if row['accepted']:
            gains=len(row['proposed_gained_contacts']);losses=len(row['proposed_lost_contacts'])
            c.update(contact_changes=int(gains+losses>0),exchanges=int(gains>0 and losses>0),
                attachments=int(gains>0 and losses==0),detachments=int(losses>0 and gains==0))
    strata={
        'measured_valid':[r for r in measured_rows if r['hard_valid']],
        'measured_one_neighbor_valid':[r for r in measured_rows if r['hard_valid'] and len(r['diagnostic']['external_partners'])==1],
        'measured_multi_neighbor_valid':[r for r in measured_rows if r['hard_valid'] and len(r['diagnostic']['external_partners'])>=2],
        'learned':learned_rows,
        'learned_valid':[r for r in learned_rows if r['hard_valid']],
        'learned_accepted':[r for r in learned_rows if r['accepted']],
        'learned_one_neighbor':[r for r in learned_rows if len(r['diagnostic']['external_partners'])==1],
        'learned_multi_neighbor':[r for r in learned_rows if len(r['diagnostic']['external_partners'])>=2],
        'learned_multi_neighbor_valid':[r for r in learned_rows if len(r['diagnostic']['external_partners'])>=2 and r['hard_valid']],
        'independent_valid_after_retry':[r for r in learned_rows if r['proposal']['branch']=='independent_conditioned' and r['hard_valid'] and r['proposal']['raw_trial_count']>1],
    }
    ratio=lambda a,b:a/b if b else None
    return dict(counts=dict(c),catalogue_elapsed_seconds=catalogue,raw_trials_per_event=quantiles(raw_counts),
        rates=dict(measured_valid_per_channel_event=ratio(c['measured_valid_candidates'],c['measured_channel_events']),
                   first_valid_per_independent_event=ratio(c['raw_first_valid'],c['independent']),
                   exhaustion_per_independent_event=ratio(c['cap_exhaustions'],c['independent']),
                   first_valid_per_raw=ratio(c['raw_first_valid'],c['raw_trials']),
                   raw_per_independent_event=ratio(c['raw_trials'],c['independent']),
                   learned_valid_per_learned_event=ratio(c['learned_valid_candidates'],c['learned']),
                   hard_valid_per_all_event=ratio(c['hard_valid'],c['attempted']),
                   accepted_per_all_event=ratio(c['accepted'],c['attempted']),
                   contact_changes_per_all_event=ratio(c['contact_changes'],c['attempted'])),
        strata={k:dict(events=len(rs),hard_valid=sum(r['hard_valid'] for r in rs),accepted=sum(r['accepted'] for r in rs),distributions=distributions(rs)) for k,rs in strata.items()})


def load(path,label,cap,local=False):
    path=Path(path).resolve();protocol=read(path/'protocol.json');status=read(path/'status.json');analysis=read(path/'analysis.json')
    require(status['complete'] and not status['running'] and analysis['complete'],'Completed physical jobs and frozen analysis required')
    require(analysis['protocol_sha256']==sha(path/'protocol.json'),'Protocol changed after analysis')
    for p,h in read(path/'freeze.json').items():require(sha(path/p)==h,'Frozen input changed: '+p)
    patch_path=path/'patch-analysis/patch-changes.json'
    require(patch_path.exists(),'Run passive analyze_singleton_patches.py first')
    patches=read(patch_path)
    require(patches['protocol_sha256']==sha(path/'protocol.json') and patches['shape_sha256']==protocol['shape_sha256'],'Patch observer/protocol identity mismatch')
    patch_pops={p['id']:p for p in patches['populations']}
    known={p['id']:p for p in analysis['populations']};pops=[];by_arm={};hashes={};accepted_details={}
    require(protocol.get('independent_max_trials')==cap,'Unexpected independent cap')
    if local:require(protocol['learned_kernel']=='local' and protocol['arms']==['local'],'Expected local-only control')
    for job in protocol['jobs']:
        config=read(job['config']);directory=Path(job['directory']);summary=read(directory/'summary.json')
        require(config['cluster_phase'].get('singleton_independent_max_trials')==cap,'Config cap mismatch')
        require(config['cluster_phase']['transport_probability']==(0. if local else 1.),'Wrong transport/local channel probability')
        require(summary['complete'] and summary['all_attempted_events_retained'] and summary['all_phase_endpoints_replayed'],'Incomplete event replay')
        for name in ('events','phases'):
            p=directory/(name+'.jsonl');h=sha(p);require(h==summary[name+'_sha256'],'Changed physical output');hashes[str(p)]=h
        rows=[r for r in map(json.loads,(directory/'events.jsonl').read_text().splitlines()) if r['kind']=='cluster_event']
        s=summarize(rows);c=s['counts'];counts=summary['counts']
        for row in rows:
            if not row['accepted']:continue
            proposal=row['proposal'];oligomer=proposal.get('oligomer') or {}
            accepted_details[job['id'],row['phase'],row['event']]=dict(
                branch=proposal['branch'],raw_trials=proposal.get('raw_trial_count'),
                target_label=proposal.get('target_label',oligomer.get('target_label')),
                fused_components=oligomer.get('fused_components'),anchor_pool=proposal.get('anchor_pool'),
                map=extract(row,'map'),pool=extract(row,'pool'),gate=extract(row,'gate'),log_acceptance=row['log_acceptance'])
        mappings={'attempted':'events','accepted':'accepted','hard_valid':'hard_valid','exchanges':'completed_exchanges',
                  'attachments':'accepted_attachments','detachments':'accepted_detachments','proposal_nulls':'proposal_nulls'}
        if cap is not None:mappings.update(raw_trials='independent_raw_trials',cap_exhaustions='independent_cap_exhaustions')
        for a,b in mappings.items():require(c[a]==counts.get(b,0),'Summary counter mismatch: '+a)
        native=known[job['id']].get('native');require(native is not None,'Completed passive native observer missing')
        patch=patch_pops[job['id']]
        require(patch['counts']['attempted']==c['attempted'] and patch['counts']['accepted']==c['accepted'],'Patch observer denominator mismatch')
        s.update(id=job['id'],population=job['population'],arm=job['arm'],seed=job['seed'],native=native,patch=patch,
            kernel_cpu_seconds=summary['kernel_cpu_seconds'],total_cpu_seconds=summary['total_cpu_seconds'])
        pops.append(s);by_arm.setdefault(job['arm'],[]).extend(rows)
    arms={}
    for arm,rows in by_arm.items():
        s=summarize(rows);records=[r for r in pops if r['arm']==arm];cpu=sum(r['kernel_cpu_seconds'] for r in records)
        s.update(patch=patches['arms'][arm],kernel_cpu_seconds=cpu,total_cpu_seconds=sum(r['total_cpu_seconds'] for r in records),
            kernel_ms_per_event=1000*cpu/s['counts']['attempted'],
            contact_changes_per_cpu_second=s['counts']['contact_changes']/cpu,
            learned_valid_per_cpu_second=s['counts']['learned_valid_candidates']/cpu,
            measured_valid_per_cpu_second=s['counts']['measured_valid_candidates']/cpu,
            native_gains=sum(r['native']['gained'] for r in records),native_losses=sum(r['native']['lost'] for r in records))
        arms[arm]=s
    for event in patches['accepted_events']:
        event['proposal_summary']=accepted_details[event['job'],event['phase'],event['event']]
    return dict(label=label,cap=cap,local=local,directory=str(path),protocol=protocol,populations=pops,arms=arms,
        accepted_patch_events=patches['accepted_events'],input_sha256=hashes,analysis_sha256=sha(path/'analysis.json'),patch_analysis_sha256=sha(patch_path),patch_map_sha256=patches['patch_map_sha256'])


def f(x):return '—' if x is None else f'{x:.2f}'
def median(r,stratum,key):return r['strata'][stratum]['distributions'][key].get('median')


def report(data,out):
    p=data['pilots'][0]['protocol'];rows=[(z['label'],arm,r) for z in data['pilots'] for arm,r in z['arms'].items()]
    lines=['# Independent hard-conditioning cap comparison','',
      'This is a fixed-state reset experiment, not an equilibrium, trajectory ESS, nucleation-rate or assembly-stability estimate. '
      'Each event, raw rejection, exhausted cap and retained endpoint enters the audit. Native labels are passive observations '
      'imported from each frozen controller; this comparison does not rerun that scan.','',
      f'All arms use the same sweep-{p["checkpoint_sweep"]:,} configuration, {p["body_count"]} mobile tetramers, '
      f'depletant radius {p["depletant_radius"]} Å and activity {p["depletant_activity"]} Å⁻³. '
      f'The independent cap arms have {p["populations"]} populations of {p["phases"]} reset phases per chart family. '
      'Completed retained-contact correlated controls are reused without rerunning them. The optional local-only control uses the same original translation and rotation scales.','',
      '![Retry-cap comparison](retry-cap-comparison.png)','',
      '| Kernel | Charts | Events | Learned | Raw trials | First hard-valid | Exhausted caps | Accepted | Contact changes | Native gains/losses | Kernel CPU s |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,arm,r in rows:
        c=r['counts'];raw=str(c['raw_trials']) if c['independent'] else '—';valid=str(c['raw_first_valid']) if c['independent'] else str(c['measured_valid_candidates'])
        lines.append(f'| {label} | {arm} | {c["attempted"]} | {c["learned"]} | {raw} | {valid} | {c["cap_exhaustions"] if c["independent"] else "—"} | {c["accepted"]} | {c["contact_changes"]} | {r["native_gains"]}/{r["native_losses"]} | {r["kernel_cpu_seconds"]:.2f} |')
    lines+=['','“First hard-valid” is a raw-success count for independent draws and a learned candidate count for correlated transport. '
      'For the local control it counts hard-valid local candidates. Raw trials are not interchangeable with channel events. The correlated branch has no independent-retry denominator. '
      'A represented-source norm stratum is deliberately absent: independent redraw never samples or encodes a source Gaussian.','',
      '## Amortization and physical exchange','',
      '| Kernel | Charts | Valid / measured channel | Mean raw / independent event | First-valid / raw | Catalogue elapsed s | CPU ms / all events | Valid / CPU s | Contact changes / CPU s |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,arm,r in rows:
        q=r['rates'];lines.append(f'| {label} | {arm} | {f(q["measured_valid_per_channel_event"])} | {f(q["raw_per_independent_event"])} | {f(q["first_valid_per_raw"])} | {r["catalogue_elapsed_seconds"]:.2f} | {r["kernel_ms_per_event"]:.2f} | {r["measured_valid_per_cpu_second"]:.2f} | {r["contact_changes_per_cpu_second"]:.4f} |')
    lines+=['','Catalogue timers measure elapsed wall time (`Instant`), whereas kernel CPU uses process CPU time. '
      'They must not be added or interpreted as the same resource. Zero catalogue time for unfused charts means no fused '
      'catalogue was built, not zero proposal overhead. Populations and events encounter different retained contexts; '
      'we do not extrapolate cap-8 success as `1-(1-p)^8` from a pooled cap-1 probability.','',
      '## Complete acceptance corrections','',
      'The following are separate medians among hard-valid learned or local candidates, with all source environments included. '
      'They are not summands for one representative move. A realized Poisson log factor is not a physical free energy.','',
      '| Kernel | Charts | Valid candidates | Full G log ratio | Ordered-pool log ratio | Total proposal | Poisson log factor | Final log acceptance |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for label,arm,r in rows:
        lines.append(f'| {label} | {arm} | {r["strata"]["measured_valid"]["events"]} | '+' | '.join(f(median(r,'measured_valid',k)) for k in ['map','pool','total_proposal','gate','log_acceptance'])+' |')
    lines+=['','## Fixed surface patches','',
      'Partner identity alone misses surface reorganization. The frozen 32-patch atom map is applied passively to '
      'each accepted moving-body pair; conservative old/new center bounds limit the pairs checked. All rejected '
      'events contribute zero patch changes. No native classifier is invoked by this patch audit.', '',
      '| Kernel | Charts | Patch-changing events | Same-partner patch changes | Mean patch Jaccard / all events | Patch changes / CPU s | Median accepted displacement Å | Median accepted rotation ° |',
      '|---|---|---:|---:|---:|---:|---:|---:|']
    for label,arm,r in rows:
        q=r['patch'];c=q['counts']
        lines.append(f'| {label} | {arm} | {c["patch_changes"]} | {c["same_partner_patch_changes"]} | {q["mean_jaccard_per_all_event"]:.6f} | {q["patch_changes_per_kernel_cpu_second"]:.4f} | {f(q["accepted_displacement_A"].get("median"))} | {f(q["accepted_rotation_degrees"].get("median"))} |')
    lines+=['','Patch changes include coarse boundary crossings and within-basin fluctuations. Their frequency is '
      'not an ESS or evidence of native incorporation; displacement, rotation and native observations remain distinct.', '',
      '## Independent populations','',
      '| Kernel | Population/arm | Events | Raw | Valid channel | Cap exhausted | Accepted | Contact changes | Exchanges | Patch changes | Native gains/losses | CPU s |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for z in data['pilots']:
        for r in z['populations']:
            c=r['counts'];lines.append(f'| {z["label"]} | {r["id"]} | {c["attempted"]} | {c["raw_trials"] if c["independent"] else "—"} | {c["measured_valid_candidates"]} | {c["cap_exhaustions"] if c["independent"] else "—"} | {c["accepted"]} | {c["contact_changes"]} | {c["exchanges"]} | {r["patch"]["counts"]["patch_changes"]} | {r["native"]["gained"]}/{r["native"]["lost"]} | {r["kernel_cpu_seconds"]:.2f} |')
    lines+=['','## Largest observed surface changes','',
      'The two largest accepted patch distances per arm are shown descriptively, with displacement as a tie-breaker. '
      'These post hoc examples are not a significance test or a predefined basin classification.', '',
      '| Kernel/charts | Population / phase / event | Old partners → new partners | Patch Jaccard | Displacement Å | Rotation ° | Native gains/losses | Raw trials / target kind / fused available |',
      '|---|---|---|---:|---:|---:|---:|---|']
    for z in data['pilots']:
        pops={r['id']:r for r in z['populations']}
        for arm in z['arms']:
            examples=sorted([e for e in z['accepted_patch_events'] if e['arm']==arm and e['patch_changed']],
                key=lambda e:(e['jaccard_distance'],e['displacement_A']),reverse=True)[:2]
            for e in examples:
                native=[n for n in pops[e['job']]['native']['events'] if n['phase']==e['phase'] and n['member']==e['member'] and n['event_time']==e['event_time']]
                gains=sum(len(n['gained']) for n in native);losses=sum(len(n['lost']) for n in native)
                lines.append(f'| {z["label"]}/{arm} | {e["job"]} / {e["phase"]} / {e["event"]} | {e["old_partners"]} → {e["new_partners"]} | {e["jaccard_distance"]:.3f} | {e["displacement_A"]:.3f} | {e["rotation_degrees"]:.3f} | {gains}/{losses} | {e["proposal_summary"]["raw_trials"]} / {(e["proposal_summary"]["target_label"] or {}).get("kind", "—")} / {e["proposal_summary"]["fused_components"]} |')
    lines+=['','[Full distributions and raw-count reconciliation](retry-cap-comparison.json) include one-neighbor and multi-neighbor strata, '
      'post-valid numerical nulls, decode nulls and acceptance after retries. All raw records are retained in the hashed original event streams. '
      'Events within reset phases are correlated; no binomial error bars or ESS estimates are inferred. '
      'Higher throughput without patch reorganization or completed competing-environment exchanges does not resolve the sampling objective.','']
    (out/'report.md').write_text('\n'.join(lines))


def plot(data,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows=[(p,arm,r) for p in data['pilots'] for arm,r in p['arms'].items()]
    colors=['#7194b5','#265e91','#d59b70','#b96a2d','#7bb69b','#277c5b','#8e739f']
    labels=[p['label']+'\n'+arm for p,arm,r in rows]
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    def bars(ax,key,title,ylabel,scale=1):
        for i,(p,arm,r) in enumerate(rows):
            value=key(r)
            if value is None:continue
            ax.bar(i,scale*value,color=colors[i%len(colors)],width=.7)
            pop=[z for z in p['populations'] if z['arm']==arm];dots=[]
            for z in pop:
                z=dict(z);z['kernel_ms_per_event']=1000*z['kernel_cpu_seconds']/z['counts']['attempted']
                z['measured_valid_per_cpu_second']=z['counts']['measured_valid_candidates']/z['kernel_cpu_seconds']
                dots.append(key(z))
            ax.scatter(i+np.linspace(-.17,.17,len(dots)),[v*scale for v in dots],s=22,facecolors='white',edgecolors='#333',zorder=3)
        ax.set_xticks(range(len(rows)),labels,fontsize=8);ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    bars(axes[0,0],lambda r:r['rates']['measured_valid_per_channel_event'],'A  Hard-valid channel candidates','% of learned or local channel events',100)
    bars(axes[0,1],lambda r:r['rates']['raw_per_independent_event'],'B  Independent raw-trial cost','Raw trials / independent event')
    bars(axes[0,2],lambda r:r['patch']['patch_changes_per_all_event'],'C  Accepted fixed-patch changes','Per 1,000 total events',1000)
    bars(axes[1,0],lambda r:r['kernel_ms_per_event'],'D  Kernel CPU cost','CPU ms / total event')
    bars(axes[1,1],lambda r:r['measured_valid_per_cpu_second'],'E  Candidate throughput','Hard-valid channel candidates / CPU s')
    ax=axes[1,2]
    for i,(p,arm,r) in enumerate(rows):
        q=r['strata']['measured_valid']['distributions']['gate']
        if q['count']:ax.errorbar(i,q['median'],yerr=[[q['median']-q['q10']],[q['q90']-q['median']]],fmt='o',color=colors[i%len(colors)],capsize=4)
    ax.set_xticks(range(len(rows)),labels,fontsize=8);ax.axhline(0,color='#444',linewidth=.8)
    ax.set_title('F  Physical auxiliary acceptance factor',loc='left',fontweight='bold');ax.set_ylabel('Log Poisson gate; median and 10–90%');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Independent hard-conditioning: finite caps and correlated control',fontsize=17,fontweight='bold')
    fig.supxlabel('Dots: independent populations. All events retained. No source-chart norm conditioning. Fixed-state resets do not measure ESS or assembly stability.',fontsize=10)
    for ext in ('png','svg','pdf'):fig.savefig(out/('retry-cap-comparison.'+ext),dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correlated',type=Path,required=True);parser.add_argument('--cap1',type=Path,required=True)
    parser.add_argument('--local',type=Path,help='Optional matched local-only singleton control')
    parser.add_argument('--cap8',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();pilots=[load(a.correlated,'Correlated',None),load(a.cap1,'Cap 1',1),load(a.cap8,'Cap 8',8)]
    if a.local:pilots.append(load(a.local,'Local',None,local=True))
    fields=['checkpoint_sweep','body_count','populations','phases','singleton_rate','duration','primary_uniform_probability',
            'anchor_pool_selection','depletant_radius','depletant_activity','model_sha256','shape_sha256','native_observer']
    for key in fields:require(all(p['protocol'][key]==pilots[0]['protocol'][key] for p in pilots),'Unmatched control parameter: '+key)
    require(all(p['protocol']['arms']==['unfused','fused'] for p in pilots if not p['local']),'Learned arm mismatch')
    require(len({p['patch_map_sha256'] for p in pilots})==1,'Different fixed patch maps')
    initial=None;seeds=[];local_scales=None
    for p in pilots:
        for job in p['protocol']['jobs']:
            config=read(job['config']);poses=config['initial_poses']
            if initial is None:initial=poses
            require(poses==initial,'Reset state mismatch');require(config['cluster_phase']['anchor_count']==2,'Unexpected anchor count')
            scales=[config['cluster_phase'][k] for k in ('local_translation_std_A','local_small_angle_std_degrees')]
            if local_scales is None:local_scales=scales
            require(scales==local_scales,'Local proposal scales mismatch')
            seeds.append(job['seed'])
    require(len(seeds)==len(set(seeds)),'Independent population seeds overlap')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    data=dict(schema='singleton-retry-cap-comparison-v1',scope='Fixed-state resets; not equilibrium, contact ESS or assembly stability',
      pilots=pilots,script_sha256=sha(__file__),matched_parameters=fields,
      source_norm='Not defined for target-only independent draws; no source-norm conditioning used.',
      count_semantics='fused_target counts an available final target-label field (correlated candidates or independent hard-valid endpoint). For independent label allocation including failed candidates, use raw_single_trials/raw_fused_trials and raw_single_valid/raw_fused_valid. Measured channel means learned or local, excluding the uniform defensive branch.')
    (out/'retry-cap-comparison.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');report(data,out);plot(data,out)
    archive=out/'analysis-source';archive.mkdir(exist_ok=True);shutil.copy2(__file__,archive/Path(__file__).name)
    outputs=['retry-cap-comparison.json','report.md','retry-cap-comparison.png','retry-cap-comparison.svg','retry-cap-comparison.pdf','analysis-source/'+Path(__file__).name]
    (out/'manifest.json').write_text(json.dumps({'complete':True,'files':{p:sha(out/p) for p in outputs}},indent=2)+'\n')
    print(json.dumps({p['label']:{a:r['counts'] for a,r in p['arms'].items()} for p in pilots},indent=2))

if __name__=='__main__':main()
