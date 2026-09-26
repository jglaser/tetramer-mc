#!/usr/bin/env python3
"""Fresh-cloud overlap validation and completed matched-proposal diagnostics."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_depletion_contact_atlas import read, write, sha


def analyze_moves(run):
    summary=read(run/'summary.json');config=read(run/'config.json')
    assert summary['complete']
    counts=defaultdict(Counter);gates=defaultdict(list);single=defaultdict(set)
    digest=hashlib.sha256();records=0
    with (run/'moves.jsonl').open('rb') as stream:
        for raw in stream:
            digest.update(raw);records+=1;m=json.loads(raw);p=m.get('proposal') or {}
            key=':'.join([m['kind'],p.get('kernel',''),p.get('branch','')]);c=counts[key]
            c['records']+=1
            if 'accepted' in m:c.update(attempted=1,accepted=int(m['accepted']))
            if 'hard_valid' in m:c['hard_valid']+=int(m['hard_valid'])
            if m['kind'] in ('global','local'):
                assert m['moving_index'] not in single[m['sweep']]
                single[m['sweep']].add(m['moving_index'])
            if m.get('hard_valid') and m.get('gate'):
                q=p.get('log_reverse_forward') or 0.;w=m['gate']['log_weight'];a=m['log_acceptance']
                if m['kind'] in ('global','local'):assert abs(a-min(0.,q+w))<1e-8
                gates[key].append([q,w,a,int(m['accepted'])])
    n=len(config['initial_poses']);sweeps=summary['completed_sweeps']
    assert set(single)==set(range(1,sweeps+1)) and all(s==set(range(n)) for s in single.values())
    decomposition={}
    for key,rows in gates.items():
        a=np.array(rows)
        decomposition[key]=dict(hard_valid=len(a),median_log_q_ratio=float(np.median(a[:,0])),
            median_log_gate=float(np.median(a[:,1])),median_log_alpha=float(np.median(a[:,2])),
            maximum_log_alpha=float(a[:,2].max()),sum_recorded_alpha=float(np.exp(a[:,2]).sum()))
    return dict(run=str(run),records=records,moves_sha256=digest.hexdigest(),counts=dict(counts),
        decomposition=decomposition,summary=summary,summary_sha256=sha(run/'summary.json'),
        one_single_update_per_body_per_sweep_verified=True)


def main(args):
    root=args.campaign.resolve();out=args.out.resolve()
    if out.exists():raise ValueError('Fresh analysis output required')
    plan=read(root/'plan.json');assert read(root/'status.json')['complete']
    rows=[];source_hashes={};process_seconds=0.
    for job in plan['jobs']:
        pop=Path(job['directory']);manifest=read(pop/'manifest.json');d=read(pop/'discovery.json')
        assert manifest['complete'];process_seconds+=manifest['wall_seconds']
        for name,digest in manifest['outputs_sha256'].items():assert sha(pop/name)==digest
        source_hashes[str(pop/'manifest.json')]=sha(pop/'manifest.json')
        for s in d['slots']:
            r=dict(population=job['id'],slot=s['slot'])
            for label in ['initial','optimized']:
                cloud=s[label+'_validation'];r[label]=sum(v['volume'] for v in cloud)/2
                r[label+'_se']=math.sqrt(sum(v['standard_error']**2 for v in cloud))/2
            r.update(gain=r['optimized']-r['initial'],gain_se=math.hypot(r['initial_se'],r['optimized_se']),
                refinement_accepted=s['refinement_counts']['accepted'],refinement_attempts=d['config']['refine_steps'])
            r['z_gain']=d['config']['activity']*r['gain'];rows.append(r)
    result=dict(rows=rows,starts=len(rows),positive_gain=sum(r['gain']>0 for r in rows),
        positive_gain_above_2se=sum(r['gain']>2*r['gain_se'] for r in rows),
        mean_initial=float(np.mean([r['initial'] for r in rows])),mean_optimized=float(np.mean([r['optimized'] for r in rows])),
        mean_z_gain=float(np.mean([r['z_gain'] for r in rows])),discovery_process_wall_seconds=process_seconds,
        refinement_accepted=sum(r['refinement_accepted'] for r in rows),refinement_attempts=sum(r['refinement_attempts'] for r in rows),
        source_hashes=source_hashes,
        scope='Pointwise integration errors only. z delta overlap is a fixed-pose pair log-weight gain, not an integrated association free energy. Finite refinement is correlated and not established stationary. Complete benchmark attempts include rejections; accepted count is not ESS or speedup.')
    if args.benchmark:
        bench=args.benchmark.resolve();assert read(bench/'status.json')['complete']
        result['benchmark']=[dict(arm=j['arm'],stream=j['stream'],**analyze_moves(Path(j['directory']))) for j in read(bench/'plan.json')['jobs']]
    out.mkdir();write(out/'summary.json',result)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');ax=axes[0]
    for job in plan['jobs']:
        rs=[r for r in rows if r['population']==job['id']]
        ax.errorbar([r['initial'] for r in rs],[r['optimized'] for r in rs],xerr=[2*r['initial_se'] for r in rs],
            yerr=[2*r['optimized_se'] for r in rs],fmt='o',ms=5,label=job['id'],alpha=.85)
    top=1.08*max(max(r['initial'],r['optimized']) for r in rows)
    ax.plot([0,top],[0,top],'--',color='.6');ax.set(xlabel='Initial overlap (Å³)',ylabel='Optimized overlap (Å³)',
        title=f"Fresh-cloud check: {result['positive_gain']}/{len(rows)} improved",xlim=(0,top),ylim=(0,top));ax.legend(fontsize=8)
    ax=axes[1]
    if 'benchmark' in result:
        key='global:full-mixture-capture:learned'
        acceptance_labels=[]
        for k,arm in enumerate(['initial','discovered']):
            rs=[r for r in result['benchmark'] if r['arm']==arm]
            accepted=sum(r['counts'][key]['accepted'] for r in rs)
            attempted=sum(r['counts'][key]['attempted'] for r in rs)
            acceptance_labels.append(f'{arm} {accepted}/{attempted:,}')
            for i,(field,sign) in enumerate([('median_log_q_ratio',-1),('median_log_gate',1)]):
                ys=[sign*r['decomposition'][key][field] for r in rs];y=float(np.mean(ys));x=i+(k-.5)*.32
                ax.bar(x,y,width=.3,color=f'C{k}',label=arm if i==0 else None)
                ax.plot([x,x],ys,color='black',lw=1)
                ax.text(x,y+.4,f'{y:.2f}',ha='center',fontsize=9)
        ax.set(xticks=range(2),xticklabels=['Proposal cost: −log(q_old/q_new)','Depletion reward: log W'],ylim=(0,31),
            ylabel='Mean of run medians, hard-valid draws',title='Learned redraws: cost still dominates');ax.legend()
        ax.text(.97,.72,'Accepted learned redraws\n'+'\n'.join(acceptance_labels),
            transform=ax.transAxes,ha='right',fontsize=9,
            bbox=dict(facecolor='white',edgecolor='.85',boxstyle='round,pad=.5'))
    else:
        ax.hist([r['z_gain'] for r in rows],bins=10);ax.set(xlabel='Pair log-weight gain z ΔC',ylabel='Starts')
    fig.suptitle('Native-blind contact discovery · r = 1.4 Å, z = 0.0275 Å⁻³')
    fig.savefig(out/'comparison.png',dpi=170);fig.savefig(out/'comparison.svg');plt.close(fig)
    write(out/'provenance.json',dict(script_sha256=sha(__file__),plan_sha256=sha(root/'plan.json'),summary_sha256=sha(out/'summary.json')))
    print(json.dumps({k:result[k] for k in ['starts','positive_gain','mean_initial','mean_optimized','mean_z_gain']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True,type=Path)
    p.add_argument('--benchmark',type=Path);p.add_argument('--out',required=True,type=Path);main(p.parse_args())
