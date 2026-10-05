"""Visualize a fixed core/tail overlap diagnostic; no region-mass inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    assert report['complete'] and report['passed'] and not report['region_mass_estimated']
    rows=report['records']; colors=['#2866a3','#419d91','#df9d34','#c54854']
    labels=['core: r² < 6','6 ≤ r² < 12','12 ≤ r² < 24','tail: 24 ≤ r² < 48']
    fig,axes=plt.subplots(1,2,figsize=(12.6,5.6))
    fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.84,wspace=.24)
    for b in range(4):
        group=[r for r in rows if r['metadata']['full_chart_bin_index']==b]
        x=np.asarray([r['metadata']['mahalanobis_squared']['full'] for r in group])
        for ax,key,ci in zip(axes,('z_overlap','log_physical_importance_score'),
                            ('simultaneous_score_interval','simultaneous_importance_interval')):
            y=np.asarray([r[key] for r in group]); interval=np.asarray([r[ci] for r in group])
            ax.errorbar(x,y,yerr=np.vstack((y-interval[:,0],interval[:,1]-y)),fmt='o',ms=5,
                        color=colors[b],capsize=2,lw=1,label=labels[b],alpha=.9)
    for ax in axes:
        ax.set_xlabel('Squared Mahalanobis radius under the full-covariance guide',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
        for x in (6,12,24):ax.axvline(x,color='#999999',lw=.7,ls=':',zorder=-2)
    axes[0].set_ylabel('Overlap benefit z O  [kBT]')
    axes[0].set_title('Does depletion suppress geometric tails?')
    axes[1].set_ylabel('Log physical importance score: z O − log q̄')
    axes[1].set_title('After correcting for proposal density')
    axes[0].legend(frameon=False,fontsize=8,loc='best')
    fig.suptitle('Fixed 24-pose diagnostic · 1.5 Å depletants · activity 0.035 Å⁻³',fontsize=13)
    fig.text(.5,.045,'All six far-tail poses + six hash-selected poses per inner stratum. Bars: simultaneous 95% point-cloud intervals.\n'
             'Conditional fixed environment at historical 500 μM. No regional free energy or assembly conclusion.',
             ha='center',fontsize=9,color='#444444')
    args.out.mkdir(parents=True,exist_ok=False)
    for ext in ('png','svg'):fig.savefig(args.out/f'core-tail-overlap.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
