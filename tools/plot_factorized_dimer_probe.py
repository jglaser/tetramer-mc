#!/usr/bin/env python3
"""Plot the completed passive comparison, never physical acceptance or ESS."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(path,output):
    report=json.loads(path.read_text())
    assert report['complete'] and report['passed'] and not report['failures']
    assert report['summary']['outer_attempts']==1536
    names=list(dict.fromkeys(row['atlas'] for row in report['comparisons']))
    assert len(names)==3
    groups={(row['atlas'],row['method']):row['summary'] for row in report['comparisons']}
    for summary in groups.values():assert summary['outer_attempts']==256
    methods=['whole_joint','factorized'];labels=['Whole dimer retries','Separate edge retries']
    colors=['#537e9e','#bd5b32']
    metrics=[('candidate_count','Feasible candidates / 256 attempts'),
             ('candidates_with_external_contacts','External-contact candidates / 256 attempts'),
             ('external_per_cpu','External-contact candidates / proposal CPU s')]
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.5),layout='constrained')
    values={}
    for ax,(metric,title) in zip(axes,metrics):
        for j,method in enumerate(methods):
            y=[]
            for name in names:
                summary=groups[name,method]
                value=(summary['candidates_with_external_contacts']/summary['proposal_cpu_seconds']
                       if metric=='external_per_cpu' else summary[metric])
                y.append(value)
            values[f'{metric}/{method}']=y
            bars=ax.bar(np.arange(3)+(j-.5)*.35,y,width=.33,color=colors[j],label=labels[j])
            ax.bar_label(bars,labels=[f'{v:.1f}' if metric=='external_per_cpu' else str(v) for v in y],fontsize=8,padding=3)
        ax.set_xticks(np.arange(3),['Blind memory','Blind FFT','Native-informed'])
        ax.set_title(title,fontsize=10,pad=12)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.yaxis.grid(True,alpha=.18)
        lo,hi=ax.get_ylim();ax.set_ylim(0,hi*1.14 if hi else 1)
    axes[0].legend(frameon=False,fontsize=8,loc='upper left')
    fig.suptitle('Staged protein-dimer proposals: geometry-only comparison',fontsize=14)
    fig.supxlabel('Fixed sweep-7400 contexts; maximum 64 edge draws per attempt. No physical acceptance or native-registry measurement.\nCPU comparison includes deferred density scoring; excludes contact diagnostics and setup. Timings are point measurements.',fontsize=9)
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.svg'):fig.savefig(output.with_suffix(suffix),dpi=180)
    svg=output.with_suffix('.svg');svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    output.with_suffix('.json').write_text(json.dumps(dict(source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),atlases=names,values=values,scope='Passive candidate throughput, not physical sampling efficiency.'),indent=2)+'\n')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();plot(args.analysis,args.output)
