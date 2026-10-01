#!/usr/bin/env python3
"""Plot completed frozen proposal diagnostics without rereading or classifying poses."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def plot(root,out):
    root,out=Path(root).resolve(),Path(out).resolve()
    if out.exists():raise ValueError('New report directory required')
    result,status,protocol=[read(root/name) for name in ['analysis.json','status.json','protocol.json']]
    if not result['complete'] or not status['complete'] or status['analysis_sha256']!=sha(root/'analysis.json'):
        raise ValueError('Completed audited pilot required')
    if result['protocol_sha256']!=sha(root/'protocol.json') or result['total_attempted_draws']!=2048:
        raise ValueError('Fixed allocation/protocol mismatch')
    out.mkdir(parents=True)
    keys=['valid','exclusion_contact','valid_native_entry','both_scaffold_contacts','native_both_anchors','registry_triangle']
    names=['Hard + domain valid','Any exclusion contact','Any native entry','Both scaffold contacts','Both native anchors','Native registry triangle']
    arms=['baseline','xyz'];colors=['#718096','#167c80'];labels=['Original Gaussian92','Feasibility-conditioned xyz']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.3),gridspec_kw={'width_ratios':[1.35,1.1,.9]})
    y=np.arange(len(keys));height=.32
    for i,(arm,color,label) in enumerate(zip(arms,colors,labels)):
        values=result['arms'][arm];metrics=values['metrics'];offset=(i-.5)*height
        frac=np.array([metrics[k]['fraction'] for k in keys])*100
        err=np.array([metrics[k]['population_SE'] for k in keys])*100
        axes[0].barh(y+offset,frac,height,xerr=err,color=color,label=label,error_kw=dict(elinewidth=1,capsize=2))
        for j,k in enumerate(keys):axes[0].text(frac[j]+err[j]+.8,y[j]+offset,str(metrics[k]['count']),va='center',fontsize=8,color=color)
        rates=[values['events_per_proposal_CPU_second'][k] for k in keys]
        axes[1].barh(y+offset,rates,height,color=color)
        cpu=values['proposal_CPU_seconds'];total=cpu['fresh_total_cpu_seconds']
        pieces=[cpu['draw_cpu_seconds'],cpu['density_cpu_seconds'],max(0,total-cpu['draw_cpu_seconds']-cpu['density_cpu_seconds'])]
        bottom=0
        for value,c,name in zip(pieces,['#90c8aa','#efb34f','#b5bfd0'],['Draw','Complete density','Other loop cost']):
            axes[2].bar(i,value/values['attempts']*1000,bottom=bottom,color=c,label=name if i==0 else None)
            bottom+=value/values['attempts']*1000
        axes[2].text(i,bottom+.06,f'{1000*total/values["attempts"]:.2f} ms',ha='center',fontsize=9)
    axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].set_xlabel('Events / all attempted proposals (%)')
    axes[0].set_title('Fresh retention and geometry');axes[0].set_xlim(0,max(result['arms'][a]['metrics']['valid']['fraction']*100 for a in arms)*1.3)
    axes[0].legend(loc='lower right',frameon=False,fontsize=9)
    axes[1].set_yticks(y,['']*len(y));axes[1].invert_yaxis();axes[1].set_xlabel('Events / proposal-loop CPU second')
    axes[1].set_title('Proposal-only throughput')
    axes[2].set_xticks([0,1],['Gaussian92','Conditioned']);axes[2].set_ylabel('CPU ms / attempted proposal')
    axes[2].set_title('Measured proposal cost');axes[2].legend(frameon=False,fontsize=9)
    for ax in axes[:2]:ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    conditioned=result['arms']['xyz']['branches'];success=conditioned['conditioned_success'];fallback=conditioned['conditioned_fallback']
    fig.suptitle('Fresh hard-free line pilot: 4 independent × 256 proposals per arm',fontsize=14,y=.995)
    fig.text(.035,.018,f'Counts label the bars; error bars are four-population SE. Conditioned draws: {success["attempts"]} successful, {fallback["attempts"]} fallback.\nZero Poisson clouds. Contact/native labels are observers; these are not accepted moves, equilibrium occupancies or a thermodynamic test.',fontsize=9)
    fig.tight_layout(rect=(.01,.105,1,.95));fig.savefig(out/'hard-free-line-fresh.png',dpi=180);fig.savefig(out/'hard-free-line-fresh.svg');plt.close(fig)
    shutil.copy2(__file__,out/'source.py')
    write(out/'summary.json',dict(schema='hard-free-line-fresh-figure-v1',complete=True,
        analysis_sha256=sha(root/'analysis.json'),protocol_sha256=sha(root/'protocol.json'),arms=result['arms'],
        comparisons=result['comparisons'],source_sha256=sha(__file__),scope=result['scope']))
    write(out/'manifest.json',dict(input_sha256={str(root/n):sha(root/n) for n in ['analysis.json','status.json','protocol.json']},
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},new_pose_draws=0,new_Poisson_clouds=0,new_classifier_calls=0))
    return str(out/'hard-free-line-fresh.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(plot(a.root,a.out))
