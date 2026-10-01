#!/usr/bin/env python3
"""Compare completed frozen singleton reset pilots without launching simulation.

All attempted moves, including invalid and null trials, enter denominators.
Per-population observations are retained; events within reset phases are not
independent equilibrium samples. Native labels are passive observer outputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def quantiles(values):
    a = sorted(values)
    if not a:
        return {'count': 0}
    def at(f):
        t = f * (len(a) - 1)
        i = int(t)
        return a[i] + (t-i) * (a[min(i+1, len(a)-1)] - a[i])
    return dict(count=len(a), minimum=a[0], q10=at(.1), median=at(.5), q90=at(.9),
                maximum=a[-1], mean=math.fsum(a)/len(a))


def norm(row, key='source_latent'):
    v = row['proposal'].get('step', {}).get(key)
    return None if v is None else math.sqrt(math.fsum(x*x for x in v))


def metrics(rows):
    out = {}
    for name, extract in (
        ('source_norm', norm), ('target_norm', lambda r: norm(r, 'target_latent')),
        ('map_log_correction', lambda r: r['proposal'].get('map_log_reverse_forward')),
        ('pool_log_correction', lambda r: r['proposal'].get('anchor_log_reverse_forward')),
        ('total_log_correction', lambda r: r['proposal'].get('log_reverse_forward')),
        ('poisson_log_gate', lambda r: (r.get('gate') or {}).get('log_weight')),
        ('log_acceptance', lambda r: r.get('log_acceptance')),
    ):
        values = [v for r in rows if (v := extract(r)) is not None]
        require(all(math.isfinite(v) for v in values), 'Nonfinite logged metric: '+name)
        out[name] = quantiles(values)
    return out


def summarize(rows):
    c = Counter(attempted=0, learned=0, uniform=0, hard_valid=0, accepted=0,
                contact_changes=0, exchanges=0, attachments=0, detachments=0,
                source_two_neighbors=0, learned_source_two_neighbors=0,
                pool_recorded=0, pool_two_contacts=0, pool_two_contacts_eligible=0,
                fused_available=0, fused_source=0, fused_target=0, null=0)
    for r in rows:
        p, d = r['proposal'], r['diagnostic']
        learned = p['branch'] == 'involution'
        degree = len(d['external_partners'])
        actual = len(d['pool_direct_partners'])
        gained, lost = len(r['proposed_gained_contacts']), len(r['proposed_lost_contacts'])
        c.update(attempted=1, learned=int(learned), uniform=int(p['branch']=='uniform'),
                 hard_valid=int(r['hard_valid']), accepted=int(r['accepted']),
                 source_two_neighbors=int(degree >= 2), learned_source_two_neighbors=int(learned and degree >= 2),
                 pool_recorded=int(p.get('anchor_pool') is not None), pool_two_contacts=int(actual >= 2),
                 pool_two_contacts_eligible=int(learned and degree >= 2 and actual >= 2),
                 null=int(r['proposed_poses'] is None))
        if p.get('oligomer'):
            o = p['oligomer']
            c.update(fused_available=int(o['fused_components'] > 0),
                     fused_source=int(o['source_label']['kind']=='fused'),
                     fused_target=int(o['target_label']['kind']=='fused'))
        if r['accepted']:
            c.update(contact_changes=int(gained+lost > 0), exchanges=int(gained>0 and lost>0),
                     attachments=int(gained>0 and lost==0), detachments=int(lost>0 and gained==0))
    n = c['attempted']
    out = dict(counts=dict(c), fractions_all_attempts={k: c[k]/n if n else None for k in c if k!='attempted'},
               fractions_conditional={
                   'two_contacts_given_eligible_learned': c['pool_two_contacts_eligible']/c['learned_source_two_neighbors'] if c['learned_source_two_neighbors'] else None,
                   'fused_available_given_learned': c['fused_available']/c['learned'] if c['learned'] else None,
               }, metrics=metrics(rows))
    groups = {
        'learned': lambda r: r['proposal']['branch']=='involution',
        'learned_hard_valid': lambda r: r['proposal']['branch']=='involution' and r['hard_valid'],
        'covered_source': lambda r: norm(r) is not None and norm(r)<=6,
        'covered_source_hard_valid': lambda r: norm(r) is not None and norm(r)<=6 and r['hard_valid'],
        'covered_one_neighbor': lambda r: norm(r) is not None and norm(r)<=6 and len(r['diagnostic']['external_partners'])==1,
        'covered_multi_neighbor': lambda r: norm(r) is not None and norm(r)<=6 and len(r['diagnostic']['external_partners'])>=2,
        'covered_multi_neighbor_hard_valid': lambda r: norm(r) is not None and norm(r)<=6 and len(r['diagnostic']['external_partners'])>=2 and r['hard_valid'],
        'covered_pool_two_contacts': lambda r: norm(r) is not None and norm(r)<=6 and len(r['diagnostic']['pool_direct_partners'])>=2,
        'accepted': lambda r: r['accepted'],
    }
    out['strata'] = {}
    for name, predicate in groups.items():
        selected = [r for r in rows if predicate(r)]
        out['strata'][name] = dict(attempted=len(selected), hard_valid=sum(r['hard_valid'] for r in selected),
            accepted=sum(r['accepted'] for r in selected), metrics=metrics(selected))
    return out


def load_pilot(path, label):
    path = Path(path).resolve()
    protocol, status, analysis = read(path/'protocol.json'), read(path/'status.json'), read(path/'analysis.json')
    require(status['complete'] and not status['running'] and analysis['complete'], 'Pilot incomplete: '+str(path))
    require(analysis['protocol_sha256']==sha(path/'protocol.json'), 'Analyzed protocol changed')
    for name, digest in read(path/'freeze.json').items():
        require(sha(path/name)==digest, 'Frozen input changed: '+name)
    known = {p['id']: p for p in analysis['populations']}
    populations, records_by_arm, hashes = [], {}, {}
    for job in protocol['jobs']:
        directory = Path(job['directory'])
        summary = read(directory/'summary.json')
        require(summary['complete'] and summary['all_attempted_events_retained'] and summary['all_phase_endpoints_replayed'], 'Incomplete replay receipt')
        event_path = directory/'events.jsonl'
        for name in ('events', 'phases'):
            p = directory/(name+'.jsonl')
            digest = sha(p)
            require(digest==summary[name+'_sha256'], 'Output changed: '+str(p))
            hashes[str(p)] = digest
        rows = [r for r in map(json.loads,event_path.read_text().splitlines()) if r['kind']=='cluster_event']
        result = summarize(rows)
        require(result['counts']['attempted']==summary['counts']['events'], 'All-attempt count mismatch')
        native = known[job['id']].get('native', {})
        result.update(id=job['id'], arm=job['arm'], population=job['population'], seed=job['seed'],
                      native=native, kernel_cpu_seconds=summary['kernel_cpu_seconds'], total_cpu_seconds=summary['total_cpu_seconds'],
                      accepted_events=[dict(phase=r['phase'], event=r['event'], members=r['members'],
                        gained=r['proposed_gained_contacts'], lost=r['proposed_lost_contacts'],
                        source_norm=norm(r), target_norm=norm(r,'target_latent'),
                        anchor_pool=r['proposal'].get('anchor_pool'),
                        map_correction=r['proposal'].get('map_log_reverse_forward'),
                        pool_correction=r['proposal'].get('anchor_log_reverse_forward'),
                        gate=(r.get('gate') or {}).get('log_weight'),
                        log_acceptance=r.get('log_acceptance')) for r in rows if r['accepted']])
        populations.append(result)
        records_by_arm.setdefault(job['arm'], []).extend(rows)
    arms = {}
    for arm, rows in records_by_arm.items():
        records = [p for p in populations if p['arm']==arm]
        result = summarize(rows)
        cpu = sum(p['kernel_cpu_seconds'] for p in records)
        result.update(kernel_cpu_seconds=cpu, total_cpu_seconds=sum(p['total_cpu_seconds'] for p in records),
                      cpu_ms_per_attempt=1000*cpu/result['counts']['attempted'],
                      contact_changes_per_kernel_cpu_second=result['counts']['contact_changes']/cpu,
                      native_gained=sum(p['native'].get('gained',0) for p in records),
                      native_lost=sum(p['native'].get('lost',0) for p in records))
        arms[arm] = result
    return dict(label=label, directory=str(path), pool_selection=protocol.get('anchor_pool_selection','nearest'),
                protocol=protocol, populations=populations, arms=arms, input_sha256=hashes,
                protocol_sha256=sha(path/'protocol.json'), analysis_sha256=sha(path/'analysis.json'))


def med(summary, stratum, metric):
    return summary['strata'][stratum]['metrics'][metric].get('median')


def fmt(x, digits=2):
    return '—' if x is None else f'{x:.{digits}f}'


def report(data, out):
    rows = [(p['label'], arm, r) for p in data['pilots'] for arm,r in p['arms'].items()]
    protocol = data['pilots'][0]['protocol']
    text = ['# Retained contact-pool singleton control', '',
        f'This comparison uses completed independent reset-phase pilots from the same sweep-{protocol["checkpoint_sweep"]:,} configuration. '
        'It measures conditional proposal opportunity and retained contact changes, not equilibrium, contact ESS, '
        'assembly stability or trajectory mixing speedup. All attempted events, including hard-invalid and null moves, '
        'are retained. Events within one reset phase are correlated.', '',
        f'Each pool law has {protocol["populations"]} independent populations of {protocol["phases"]} phases per fusion arm. The preserved shape, atlas, '
        f'{protocol["body_count"]} mobile tetramers, depletant radius {protocol["depletant_radius"]} Å, activity {protocol["depletant_activity"]} Å⁻³, '
        f'correlation {protocol["correlation"]} and defensive primary probability {protocol["primary_uniform_probability"]} match. Nearest-pool populations are reused; '
        'the retained contact-pool populations use fresh streams. Fusion differs within each matched pool-law comparison.', '',
        '![Pool comparison](comparison.png)', '',
        '| Pool | Charts | Attempts | Valid | Accepted | Contact changes | Exchanges | Native gains/losses | Kernel CPU (s) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,arm,r in rows:
        c=r['counts']
        text.append(f'| {label} | {arm} | {c["attempted"]} | {c["hard_valid"]} | {c["accepted"]} | {c["contact_changes"]} | {c["exchanges"]} | {r["native_gained"]}/{r["native_lost"]} | {r["kernel_cpu_seconds"]:.2f} |')
    candidate = data['pilots'][1]
    fused = candidate['arms'].get('fused')
    if fused is not None:
        text += ['', f'The retained-contact fused arm accepted {fused["counts"]["accepted"]} moves, '
            f'with {fused["counts"]["contact_changes"]} contact-changing events and '
            f'{fused["native_gained"]}/{fused["native_lost"]} native pair-entry gains/losses. '
            'Improved coverage or more accepted moves is not itself evidence of structural reorganization.', '']
    exchanges = [(r, e) for r in candidate['populations'] for e in r['accepted_events'] if e['gained'] and e['lost']]
    for r, e in exchanges[:5]:
        text += [f'Accepted exchange in {r["id"]}, reset phase {e["phase"]}, event {e["event"]}: '
                 f'moving bodies {e["members"]}, lost edges {e["lost"]}, gained edges {e["gained"]}.', '']
    text += ['These are conditional reset observations; small counts do not establish a rate advantage. '
        'The gate diagnostics below test retention of surrounding overlap, not thermodynamic preference '
        'over native alternatives.', '', '## Actual neighboring constraints', '',
        '“Both neighbors” means two distinct spectators in the retained pool actually touch the moving body at the '
        'source, measured using the exclusion-contact graph. The conditional denominator includes every learned '
        'attempt whose source has at least two neighbors; it does not condition on successful fusion or hard validity.', '',
        '| Pool | Charts | Both contacts / all attempts | Both contacts / eligible learned | Fusion available / learned | Source norm median | CPU ms / attempt |',
        '|---|---|---:|---:|---:|---:|---:|']
    for label,arm,r in rows:
        c=r['counts']
        text.append(f'| {label} | {arm} | {c["pool_two_contacts"]}/{c["attempted"]} | {c["pool_two_contacts_eligible"]}/{c["learned_source_two_neighbors"]} | {c["fused_available"]}/{c["learned"]} | {fmt(r["strata"]["learned"]["metrics"]["source_norm"].get("median"))} | {r["cpu_ms_per_attempt"]:.2f} |')
    text += ['', '## Acceptance contributions', '',
        'The following posthoc stratum contains hard-valid proposals with source Gaussian norm ≤6. These '
        'are separate medians, not summands for a representative move. The Poisson gate is an auxiliary '
        'realization, not a physical free-energy estimate. Missing correction entries for invalid proposals remain missing.', '',
        '| Pool | Charts | Stratum count | Map log correction | Full pool log correction | Full proposal log correction | Poisson gate | Final log acceptance |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for label,arm,r in rows:
        st='covered_source_hard_valid'; names=['map_log_correction','pool_log_correction','total_log_correction','poisson_log_gate','log_acceptance']
        text.append(f'| {label} | {arm} | {r["strata"][st]["attempted"]} | '+' | '.join(fmt(med(r,st,n)) for n in names)+' |')
    text += ['', 'For the nearest law the pool term is the primary-anchor ratio; the remaining anchors '
        'are deterministic functions of the retained primary and fixed spectators. For contact selection '
        'the term is the joint ordered-pool ratio, including every selected spectator. It is applied once. '
        'The uniform pose branch receives no anchor factor.', '', '## Independent populations', '',
        '| Pool | Arm/population | Attempts | Both / eligible learned | Fused / learned | Valid | Accepted | Contact changes | Native gains/losses | CPU s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in data['pilots']:
        for r in p['populations']:
            c=r['counts']; n=r['native']
            text.append(f'| {p["label"]} | {r["id"]} | {c["attempted"]} | {c["pool_two_contacts_eligible"]}/{c["learned_source_two_neighbors"]} | {c["fused_available"]}/{c["learned"]} | {c["hard_valid"]} | {c["accepted"]} | {c["contact_changes"]} | {n.get("gained",0)}/{n.get("lost",0)} | {r["kernel_cpu_seconds"]:.2f} |')
    text += ['', 'Full source-norm, target-norm, correction and gate distributions, including one-neighbor '
        'and multi-neighbor strata, appear in [comparison.json](comparison.json). Native observations '
        'classify accepted pair-entry changes; they do not infer whole-crystal stability or independent basin samples.', '',
        'All frozen inputs, event streams and phase streams are hash-checked. Each original controller '
        'validated deterministic endpoint replay. This script does not launch physical jobs or edit '
        'the frozen controllers, configurations or trajectory files. Population tables preserve the '
        'independent allocation; no within-phase Bernoulli independence is assumed.', '',
        'The new kernel passed five focused correctness checks, including independent four-sphere '
        'equilibrium and full runner replay/restart. Archived receipts are in '
        '[validation/contact-pool-validation-20261001.log](validation/contact-pool-validation-20261001.log) '
        'with source hashes and an archive checksum manifest.', '']
    (out/'report.md').write_text('\n'.join(text))


def plot(data, out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [(p, arm, r) for p in data['pilots'] for arm,r in p['arms'].items()]
    colors=(['#7189a2','#245f96','#d69565','#b95b1f']*math.ceil(len(rows)/4))[:len(rows)]
    labels=[p['label']+'\n'+arm for p,arm,r in rows]
    fig,axes=plt.subplots(2,3,figsize=(15,8.6),layout='constrained')
    def bars(ax, extract, title, ylabel, percent=False):
        vals=[extract(r) for p,arm,r in rows]
        scale=100 if percent else 1
        ax.bar(range(len(rows)),[v*scale for v in vals],color=colors,alpha=.85,width=.65)
        for i,(p,arm,r) in enumerate(rows):
            pops=[z for z in p['populations'] if z['arm']==arm]
            yy=[]
            for z in pops:
                z=dict(z)
                z['cpu_ms_per_attempt']=1000*z['kernel_cpu_seconds']/z['counts']['attempted']
                yy.append(extract(z)*scale)
            xx=i+np.linspace(-.15,.15,len(yy))
            ax.scatter(xx,yy,s=23,facecolors='white',edgecolors='#222',zorder=3)
            ax.text(i,vals[i]*scale,f'{vals[i]*scale:.1f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(range(len(rows)),labels,fontsize=9);ax.set_title(title,loc='left',fontweight='bold')
        ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    bars(axes[0,0],lambda r:r['fractions_conditional']['two_contacts_given_eligible_learned'] or 0,
         'A  Pool contains two actual neighbors','% of eligible learned attempts',True)
    bars(axes[0,1],lambda r:r['fractions_conditional']['fused_available_given_learned'] or 0,
         'B  Fused catalogue available','% of all learned attempts',True)
    bars(axes[0,2],lambda r:r['fractions_all_attempts']['hard_valid'],
         'C  Hard-valid proposals','% of all attempts',True)
    bars(axes[1,0],lambda r:r['fractions_all_attempts']['contact_changes'],
         'D  Accepted contact changes','% of all attempts',True)
    bars(axes[1,1],lambda r:r['cpu_ms_per_attempt'],'E  Kernel cost','CPU ms / all attempts')
    ax=axes[1,2]
    for i,(p,arm,r) in enumerate(rows):
        q=r['strata']['covered_multi_neighbor_hard_valid']['metrics']['poisson_log_gate']
        if q['count']:
            ax.errorbar(i,q['median'],yerr=[[q['median']-q['q10']],[q['q90']-q['median']]],fmt='o',color=colors[i],capsize=5)
            ax.annotate(f'n={q["count"]}',(i,q['median']),xytext=((-8 if i==len(rows)-1 else 6),7),textcoords='offset points',fontsize=8,ha=('right' if i==len(rows)-1 else 'left'))
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(len(rows)),labels,fontsize=9)
    ax.set_title('F  Gate for covered multi-neighbor sources',loc='left',fontweight='bold')
    ax.set_ylabel('Log Poisson factor; median and 10–90%');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Retained contact-pool control — fixed-state reset phases',fontsize=17,fontweight='bold')
    protocol=data['pilots'][0]['protocol']
    fig.supxlabel(f'{protocol["populations"]} populations × {protocol["phases"]} phases per arm; dots show populations. No contact ESS or equilibrium claim.\n'
                  f'Covered: source norm ≤6; gate panel additionally requires hard validity. r = {protocol["depletant_radius"]} Å, z = {protocol["depletant_activity"]} Å⁻³.',fontsize=10)
    for suffix in ('png','svg','pdf'):
        fig.savefig(out/('comparison.'+suffix),dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    pilots=[load_pilot(args.baseline,'Nearest'),load_pilot(args.candidate,'Contact')]
    keys=['checkpoint_sweep','body_count','populations','phases','singleton_rate','duration','correlation',
          'primary_uniform_probability','depletant_radius','depletant_activity','model_sha256','shape_sha256','native_observer']
    for k in keys:
        require(pilots[0]['protocol'][k]==pilots[1]['protocol'][k], 'Unmatched pilot parameter: '+k)
    require(pilots[0]['protocol']['arms']==pilots[1]['protocol']['arms'], 'Different chart arms')
    configs=[[read(Path(j['config'])) for j in z['protocol']['jobs']] for z in pilots]
    initial=[c[0]['initial_poses'] for c in configs]
    expected_count=configs[0][0]['cluster_phase']['anchor_count']
    for configs_in_pilot, poses in zip(configs,initial):
        require(all(c['initial_poses']==poses for c in configs_in_pilot), 'Reset configurations differ within pilot')
        require(all(c['cluster_phase']['anchor_count']==expected_count for c in configs_in_pilot), 'Different anchor counts')
    require(initial[0]==initial[1], 'Different reset configurations')
    seeds=[[p['seed'] for p in z['populations']] for z in pilots]
    require(not set(seeds[0]).intersection(seeds[1]), 'Pilot streams are not independent')
    data=dict(schema='singleton-pool-comparison-v1',scope='Fixed-state reset proposal diagnostic; not equilibrium, ESS, or mixing speedup.',
              pilots=pilots,script_sha256=sha(__file__),matched_parameters=keys,
              stratum_note='Source norm <=6 and neighbor-degree strata are posthoc descriptive diagnostics.')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    report(data,out);plot(data,out)
    (out/'analysis-source').mkdir(exist_ok=True)
    shutil.copy2(__file__,out/'analysis-source'/Path(__file__).name)
    print(json.dumps({p['label']:{a:{k:r[k] for k in ['counts','kernel_cpu_seconds','native_gained','native_lost']} for a,r in p['arms'].items()} for p in pilots},indent=2))

if __name__=='__main__':
    main()
