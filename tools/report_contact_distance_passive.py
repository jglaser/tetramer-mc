#!/usr/bin/env python3
"""Bound review/figure from completed passive and native-observer outputs only."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
def write(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def report(passive,observer,out):
    passive,observer,out=map(lambda p:Path(p).resolve(),(passive,observer,out))
    require(not out.exists(),'Fresh review required')
    bindings={}
    def bind(p,digest=None):
        actual=sha(p);require(digest is None or actual==digest,'Completed source changed: '+str(p))
        bindings[str(p)]=actual
        return read(p)
    state=bind(passive/'status.json');obs_state=bind(observer/'status.json')
    require(state['complete'] and obs_state['complete'],'Both passive and native observer must complete')
    data=bind(passive/'analysis.json',state['analysis_sha256']);native=bind(observer/'analysis.json',obs_state['analysis_sha256'])
    bind(passive/'protocol.json',state['protocol_sha256']);bind(observer/'protocol.json',obs_state['protocol_sha256'])
    for job in [*state['jobs'],*state['coverage_jobs']]:
        for name,digest in job['output_sha256'].items():
            p=Path(job['directory'])/name;require(sha(p)==digest,'Passive audited output changed');bindings[str(p)]=digest
    for p,digest in native['input_sha256'].items():require(sha(p)==digest,'Observer source changed');bindings[p]=digest
    names=['baseline92','uniform_phi92','localized_phi92'];stats={}
    for name in names:
        a=data['arms'][name];n=native['arms'][name];pops=a['populations'];npops=n['populations']
        for p in npops:
            directory=observer/name/p['population'];bind(directory/'summary.json')
            file=directory/'labels.jsonl';require(sha(file)==p['labels_sha256'],'Native labels changed');bindings[str(file)]=sha(file)
        partition={key:sum(p['partition'][key] for p in npops) for key in npops[0]['partition']}
        conditioned={key:sum(p['branches']['conditioned_success'][key] for p in npops)
                     for key in npops[0]['branches']['conditioned_success']}
        counts=dict(attempts=sum(p['draws'] for p in pops),hard_valid_in_domain=sum(p['hard_shell_capture'] for p in pops),
                    shell=sum(p['shell'] for p in pops),native=partition['valid_native_entry'],
                    competing=partition['valid_contact_without_native_entry'],unbound=partition['valid_unbound_without_native_entry'],
                    both_native_anchors=sum(p['native_both_anchors'] for p in npops),
                    cycle_consistent_triangle=sum(p['registry_triangle'] for p in npops),
                    joint_by_width=[sum(p['joint_contact_by_width'][i] for p in pops) for i in range(3)],
                    conditional=sum(p['conditional'] for p in pops),fallback=sum(p['fallback'] for p in pops),
                    conditioned_valid=conditioned['valid_native_entry']+conditioned['valid_contact_without_native_entry']+conditioned['valid_unbound_without_native_entry'])
        require(counts['attempts']==512 and sum(partition.values())==512 and
                counts['hard_valid_in_domain']==counts['native']+counts['competing']+counts['unbound'],
                'Attempt/class partitions differ')
        cpu=a['total_fresh_cpu_seconds']
        stats[name]=dict(counts=counts,partition=partition,conditioned=conditioned,
            fresh_CPU_seconds=cpu,fresh_ms_per_attempt=1000*cpu/512,
            classifier_CPU_seconds=n['classifier_CPU_seconds'],
            child_CPU_seconds=a['total_child_cpu_seconds'],
            probe_CPU_seconds=a['total_probe_cpu_seconds'],
            independent_audit_CPU_seconds=sum(j['audit_cpu_seconds'] for j in [*state['jobs'],*state['coverage_jobs']] if j['arm']==name),
            rates_per_fresh_CPU=dict(native=counts['native']/cpu,competing=counts['competing']/cpu,
                hard_valid=counts['hard_valid_in_domain']/cpu,joint_by_width=a['joint_by_width_per_fresh_cpu']),
            proposal_fractions=a['fractions'],native_fractions=n['fractions'])
    baseline=stats['baseline92']
    for name,s in stats.items():
        s['fresh_CPU_ratio_vs_baseline']=s['fresh_CPU_seconds']/baseline['fresh_CPU_seconds']
        s['rate_ratios_vs_baseline']={key:s['rates_per_fresh_CPU'][key]/baseline['rates_per_fresh_CPU'][key]
                                    for key in ('native','competing','hard_valid')}
    summary=dict(schema='contact-distance-passive-review-v1',complete=True,source_sha256=bindings,stats=stats,
        passive_wall_seconds=state['finished']-state['started'],observer_CPU_seconds=native['total_observer_CPU_seconds'],
        archived_moment_diagnostics=data['archived_moment_diagnostics'],breadth=data['coverage'],
        maximum_independent_audit_errors=data['maximum_independent_audit_errors'],
        fresh_attempts=1536,archived_queries=618,new_Poisson_clouds=0,new_physical_mass_estimates=0,
        decision='No physical weight campaign with these laws: localized azimuth improves on uniform azimuth, but not overall native/contact efficiency over baseline; archived weight moments worsen and remain poorly resolved.',
        scope='Proposal-only geometric event frequencies and passive CPU. No thermodynamic or assembly-stability inference.')
    out.mkdir(parents=True);write(out/'summary.json',summary)
    colors=['#495e78','#c98d55','#168a7c'];labels=['Old92','Uniform azimuth','Localized azimuth']
    fig,axes=plt.subplots(2,2,figsize=(12.5,7.7))
    ax=axes[0,0];x=np.arange(3);nv=[stats[n]['counts']['native'] for n in names];cv=[stats[n]['counts']['competing'] for n in names]
    ax.bar(x,nv,color='#168a7c',label='Native entry');ax.bar(x,cv,bottom=nv,color='#5b77a0',label='Contact without native entry')
    for i,(a,b) in enumerate(zip(nv,cv)):
        ax.text(i,a/2,str(a),ha='center',va='center',color='white');ax.text(i,a+b/2,str(b),ha='center',va='center',color='white')
    ax.set_xticks(x,labels);ax.set_ylim(0,125);ax.set_ylabel('Count among all 512 attempted poses')
    ax.set_title('No increase in overall native yield');ax.legend(frameon=False,fontsize=9)
    ax=axes[0,1];bottom=np.zeros(2)
    for key,color,label in [('outside_R4_or_capture','#ddd5ce','Outside domain'),('inside_domain_hard_invalid','#9a9390','Hard overlap'),
                            ('valid_native_entry','#168a7c','Native'),('valid_contact_without_native_entry','#5b77a0','Competing contact')]:
        values=np.array([stats[n]['conditioned'][key] for n in names[1:]])
        ax.bar(np.arange(2),values,bottom=bottom,color=color,label=label)
        for i,(v,b) in enumerate(zip(values,bottom)):
            if v>=8:ax.text(i,b+v/2,str(v),ha='center',va='center',color='white' if key.startswith('valid') else '#262626')
        bottom+=values
    ax.set_xticks([0,1],['Uniform φ\n4/118 survive','Localized φ\n21/122 survive']);ax.set_ylim(0,145)
    ax.set_ylabel('Selected conditional branches');ax.set_title('Localized azimuth repairs much of the domain loss')
    ax.legend(frameon=False,fontsize=8,ncol=2,loc='upper left')
    ax=axes[1,0]
    for i,name in enumerate(names):
        value=stats[name]['proposal_fractions']['joint_width_0.02']
        ax.bar(i,100*value['mean'],yerr=100*value['population_SE'],capsize=4,color=colors[i])
        ax.text(i,100*(value['mean']+value['population_SE'])+.1,f"{stats[name]['counts']['joint_by_width'][0]}/512",ha='center')
    ax.set_xticks(x,labels);ax.set_ylim(0,3.2);ax.set_ylabel('Both surface gaps ≤0.02Å (% of attempts)')
    ax.set_title('9 versus 2 tight contacts is exploratory\nError bars: four-population SE')
    ax=axes[1,1]
    for offset,key,color,label in [(-.19,'native','#168a7c','Native entry'),(.19,'competing','#5b77a0','Competing contact')]:
        rates=[stats[n]['rates_per_fresh_CPU'][key] for n in names]
        ax.bar(x+offset,rates,width=.36,color=color,label=label)
        for i,v in enumerate(rates):ax.text(i+offset,v+2,f'{v:.1f}',ha='center',fontsize=9)
    ax.set_xticks(x,labels);ax.set_ylim(0,160);ax.set_ylabel('Events / proposal-only CPU second')
    ax.set_title('Baseline remains more efficient overall');ax.legend(frameon=False,fontsize=9)
    for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Exact two-distance guide: useful geometric control, no physical campaign justified',fontsize=14)
    fig.text(.5,.013,'1,536 fresh poses + 618 archived density queries; zero Poisson clouds. Geometric proposal events are not equilibrium samples.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95))
    for suffix in ('png','svg'):fig.savefig(out/f'contact-distance-passive.{suffix}',dpi=180)
    plt.close(fig);shutil.copy2(__file__,out/'source.py')
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(out=str(out),source_files_bound=len(bindings),summary_sha256=sha(out/'summary.json'),
                         new_physical_jobs=0,counts={name:s['counts'] for name,s in stats.items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('passive','observer','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report(a.passive,a.observer,a.out)
