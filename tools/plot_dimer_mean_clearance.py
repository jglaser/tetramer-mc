#!/usr/bin/env python3
"""Plot deterministic center clearances and an authenticated saved-draw join."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args(); root=args.directory
    summary=json.loads((root/'execution/analysis.json').read_text())
    post=json.loads((root/'postprocess.json').read_text())
    ledger=root/'execution/queries.jsonl'
    if not summary['passed'] or not summary['complete'] or not post['passed']:
        raise ValueError('Complete passing center and join audits required')
    if sha(ledger)!=post['center_ledger_sha256'] or sha(root/'execution/analysis.json')!=post['analysis_sha256']:
        raise ValueError('Changed center evidence')
    if sha(root/'saved-draw-center-join.jsonl')!=post['join_ledger_sha256']:
        raise ValueError('Changed saved-draw join')
    rows=[json.loads(s) for s in ledger.read_text().splitlines()]
    joins=[json.loads(s) for s in (root/'saved-draw-center-join.jsonl').read_text().splitlines()]
    if len(rows)!=2504 or len(joins)!=post['joined_rows'] or len(joins)!=512:
        raise ValueError('Incomplete evidence')
    names=[('blind_memory_allslot64','Blind memory (128 branches)','#64748b'),
           ('blind_fft512slots','Blind FFT (2,048 branches)','#2563eb'),
           ('native_informed178','Native-informed (328 branches)','#7c3aed')]
    fig,(left,right)=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.3,1]})
    for name,label,color in names:
        values=np.sort([r['clearance_A'] for r in rows if r['atlas']==name])
        left.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=label,color=color,lw=1.8)
    left.set_xscale('symlog',linthresh=.01)
    left.axvspan(-5,0,color='#fee2e2',alpha=.7,zorder=-1)
    left.axvline(0,color='#9f1239',ls=':',lw=1)
    left.set_xlim(-5,60);left.set_ylim(0,1.02)
    left.set_xticks([-3,-.1,0,.01,.1,1,10,50],['−3','−0.1','0','0.01','0.1','1','10','50'])
    left.set_xlabel('Minimum atomic surface gap at chart center (Å)')
    left.set_ylabel('Cumulative fraction of virtual branch centers')
    left.set_title('Blind FFT centers lie close to the hard boundary',fontsize=11)
    left.legend(fontsize=8,loc='upper left',frameon=False)
    left.grid(axis='y',alpha=.2)
    counts=post['join_table']; n=post['joined_rows']
    center_clash=sum(r['count'] for r in counts if r['center_core_overlap'])
    draw_clash=sum(r['count'] for r in counts if r['sampled_internal_core_overlap'])
    valid=np.array([n-center_clash,n-draw_clash]); clash=np.array([center_clash,draw_clash])
    y=np.array([1,0]); right.barh(y,valid/n*100,color='#0f766e',height=.48,label='No core clash')
    right.barh(y,clash/n*100,left=valid/n*100,color='#d97706',height=.48,label='Core clash')
    right.set_yticks(y,['Same drawn\ncomponents: centers','Gaussian draws\naround those centers'])
    right.text(50,1,f'{valid[0]}/{n} feasible',ha='center',va='center',color='white',fontsize=10)
    right.text(53,0,f'{clash[1]}/{n} collide',ha='center',va='center',color='white',fontsize=10)
    right.annotate(f'{valid[1]} feasible',xy=(valid[1]/n*50,0),xytext=(20,-.53),
                   ha='center',fontsize=9,arrowprops={'arrowstyle':'-','color':'#0f766e'})
    right.set_xlim(0,100);right.set_ylim(-.8,1.55)
    right.set_xlabel('Fraction of 512 saved blind FFT draws (%)')
    right.set_title('Every sampled component center was feasible',fontsize=11)
    right.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,fontsize=8)
    for ax in (left,right):ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Blind FFT draws cross the hard boundary around feasible centers',fontsize=13,y=.985)
    fig.text(.5,.014,'Left: all virtual branches, equally counted. Right: paired saved component labels and their draws. Pair geometry only; neither panel is an equilibrium occupancy.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.095,1,.94))
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    outputs=[args.output_prefix.with_suffix(ext) for ext in ('.png','.svg','.json')]
    if any(p.exists() for p in outputs):raise FileExistsError('Refusing to replace plot output')
    for p in outputs[:2]:
        fig.savefig(p,dpi=170,bbox_inches='tight')
        if p.suffix=='.svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    record={'inputs':{str(p):sha(p) for p in [ledger,root/'execution/analysis.json',root/'postprocess.json',root/'saved-draw-center-join.jsonl']},
            'joined_draws':n,'center_clashes':center_clash,'sampled_internal_clashes':draw_clash,
            'left_measure':'Unweighted virtual branch centers, not mixture sample mass.',
            'right_measure':'Same512 saved branch labels at center and after Gaussian draw; no physical acceptance.'}
    outputs[2].write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
