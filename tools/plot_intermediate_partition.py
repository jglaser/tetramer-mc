#!/usr/bin/env python3
"""Plot partitioned and historical intermediate estimates with observed errors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def point(ax,y,logq,rse,color,marker='o'):
    if logq is None:return
    lower=-math.log1p(-rse) if rse<1-1e-12 else 5.
    upper=math.log1p(rse)
    ax.errorbar(logq,y,xerr=np.array([[lower],[upper]]),fmt=marker,color=color,capsize=4,markersize=7)
    if rse>=1-1e-12:ax.annotate('',xy=(logq-lower-.5,y),xytext=(logq-lower,y),arrowprops=dict(arrowstyle='->',color=color))


def plot(total_path,historical_path,out):
    assert not out.exists()
    total=json.loads(total_path.read_text());history=json.loads(historical_path.read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),gridspec_kw={'width_ratios':[1.2,1]})
    labels=[]
    for i,piece in enumerate(total['pieces']):
        lo,hi=piece['region']['minimum'],piece['region']['maximum'];row=piece['physical']
        label=f'r ≤ {hi:g}' if lo==0 else (f'{lo:g} < r ≤ {hi:g}' if hi is not None else f'r > {lo:g}')
        labels.append(label);point(axes[0],i,row['logQ'],row['row_RSE'],'#20765b')
        fraction=format(row['maximum_fraction'],'.2%' if row['maximum_fraction']>.99 else '.1%')
        axes[0].annotate(f"largest draw {fraction}",(row['logQ'],i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set_yticks(range(len(labels)),labels);axes[0].invert_yaxis()
    axes[0].set_title('Disjoint reference contributions')
    controls=[('New partition',total['physical']['logQ'],total['physical']['row_RSE'],'#20765b')]
    guides=history['guided_campaigns']
    for kind,label,color in [('mixture','Historical mixture','#c57722'),('geometry','Historical geometry','#4770aa')]:
        row=next(v for k,v in guides.items() if kind in k)['physical']['all']
        controls.append((label,row['logQ'],row['row_RSE'],color))
    # The historical direct reference supplied the proposal-training data.
    reference=history['reference']['physical']['all']
    controls.append(('Training cover',reference['logQ'],reference['row_RSE'],'#777777'))
    for i,(label,q,se,color) in enumerate(controls):point(axes[1],i,q,se,color)
    axes[1].set_yticks(range(len(controls)),[c[0] for c in controls]);axes[1].invert_yaxis()
    axes[1].set_title('Complete original 2 ≤ q < 5')
    for ax in axes:
        ax.set_xlabel(r'$\log Q$');ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.margins(y=.25)
    fig.suptitle('Outer contacts expose unresolved weight in the intermediate integral',fontsize=13)
    fig.text(.5,.08,'Fixed AB neighbors; depletant radius 1.5 Å, activity 0.035 Å⁻³. r is dimensionless weighted-chart radius.',ha='center',fontsize=9)
    fig.text(.5,.045,f"One new draw supplies {total['physical']['largest_single_draw_fraction']:.1%} of the partitioned estimate.",ha='center',fontsize=10)
    fig.text(.5,.01,'Bars transform Q ± observed SE to log scale. They are not confidence bounds on unseen weight; training and historical controls are labeled.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.14,1,.93))
    out.mkdir(parents=True)
    for extension in ('png','svg'):fig.savefig(out/f'intermediate-partition.{extension}',dpi=180,bbox_inches='tight')
    shutil.copy2(__file__,out/'plotter.py')
    (out/'provenance.json').write_text(json.dumps(dict(inputs={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (total_path,historical_path)},
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    plt.close(fig);print(out/'intermediate-partition.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--total',type=Path,required=True)
    p.add_argument('--historical',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.total,a.historical,a.out)
