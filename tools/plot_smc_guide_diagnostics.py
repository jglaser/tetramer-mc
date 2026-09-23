#!/usr/bin/env python3
"""Plot saved offline design scores; no fitting, physics or new uncertainty estimate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source, out):
    if out.exists():
        raise ValueError('Refuse to overwrite a report')
    freeze=json.loads((source.parent/'freeze.json').read_text())
    if digest(source)!=freeze['files'][source.name]:
        raise ValueError('Completed diagnostic source changed')
    j=json.loads(source.read_text())
    if not j['complete'] or j['new_physical_draws']!=0:
        raise ValueError('Expected completed offline diagnostic')
    out.mkdir(parents=True)
    candidates=['r5-cov1','r5-cov4','r5-sign-cov1','r5-sign-cov4']
    classes=['old_R5_intersection_native','remaining_R4_native','contact_no_native_entry']
    titles=['Native inside historical R5','Native outside historical R5','Competing contact: no native entry']
    arms=['bank','wide','small','alpha02','intensity256']
    colors=['#1766ad','#de7b17','#2c8c62','#aa3e89','#665dba']
    labels=['Bank source, 128','Wide source, 128','Small source, 128','20% uniform source, 128','Bank source, 256']
    fig,axes=plt.subplots(1,3,figsize=(13,5.2),layout='constrained')
    for ax,cl,title in zip(axes,classes,titles):
        for a,(arm,color,label) in enumerate(zip(arms,colors,labels)):
            values=[j['aggregates'][arm]['groups'][cl]['guides'][name]['two_cloud_noisy']['ratio_to_bank'] for name in candidates]
            ax.scatter(np.arange(4)+(a-2)*.13,values,color=color,s=40,label=label,zorder=3)
        ax.axhline(1,color='#444444',ls='--',lw=1)
        ax.set_yscale('log');ax.set_ylim(.04,2.3)
        ax.set_yticks([.05,.1,.2,.5,1,2],labels=['0.05','0.1','0.2','0.5','1','2'])
        ax.set_xticks(range(4),labels=['R5 split\ncov ×1','R5 split\ncov ×4','R5 + sign\ncov ×1','R5 + sign\ncov ×4'])
        ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Two-cloud second moment / unchanged bank\nLower is favorable; not a speedup estimate')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9,
        title='Saved source populations; number is auxiliary intensity ratio λ/z',title_fontsize=9)
    fig.suptitle('SMC geometry helps the native proposal, but diverts coverage from competing contacts\n'
        'Retrospective ratios from four populations per source arm; no confidence intervals or unseen-mode guarantee',fontsize=12)
    fig.savefig(out/'second-moment-tradeoff.png',dpi=180)
    fig.savefig(out/'second-moment-tradeoff.svg')
    plt.close(fig)
    metadata={'schema':'smc-guide-diagnostic-figure-v1','complete':True,
        'source':str(source),'source_sha256':digest(source),'code_sha256':digest(Path(__file__)),
        'new_physical_draws':0,'uncertainty_intervals':False,
        'scope':'Descriptive second-moment ratios within each independent source arm/cloud law. No convergence or speedup claim.',
        'files':{p.name:digest(p) for p in sorted(out.iterdir())}}
    (out/'figure.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.source.resolve(),args.out.resolve())
