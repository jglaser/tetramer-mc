"""Render completed conditional importance evidence; never query geometry or fit."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS=('baseline','broadened')
COLORS={'baseline':'#4477aa','broadened':'#cc6677'}
LABELS={'baseline':'F/D baseline','broadened':'F/D + broad F'}
GROUPS=('A_T','contact_without_A_T','unbound','full_domain')
TICKS=('A with full T','Other contacts','Unbound','Full domain')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def plot(report,out,source_path):
    if not(report['schema']=='context-broadened-physical-report-v1' and report['complete']
           and report['passed'] and report['all_attempts']==32768):
        raise ValueError('Completed fixed-allocation physical report required')
    if out.exists():raise ValueError('Fresh render output required')
    out.mkdir(parents=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8.7),constrained_layout=True)
    mass,eff=axes[0]
    zero_count={'mass':0,'efficiency':0}
    for ai,arm in enumerate(ARMS):
        populations=[p for p in report['populations'] if p['comparison_arm']==arm]
        if len(populations)!=4:raise ValueError('Exactly four populations per arm required')
        offset=(-.14,.14)[ai]
        for gi,g in enumerate(GROUPS):
            values=[p['groups'][g]['physical']['mass'] for p in populations]
            efficiency=[p['importance_ess_per_cpu'][g] for p in populations]
            jitter=np.linspace(-.045,.045,4)
            for i,(v,e) in enumerate(zip(values,efficiency)):
                if v>0:mass.scatter(gi+offset+jitter[i],v,color=COLORS[arm],s=27,alpha=.75)
                else:zero_count['mass']+=1
                if e>0:eff.scatter(gi+offset+jitter[i],e,color=COLORS[arm],s=27,alpha=.75)
                else:zero_count['efficiency']+=1
            summary=report['arm_summaries'][arm][g]
            if summary['mean_mass']>0:
                mass.plot([gi+offset-.085,gi+offset+.085],[summary['mean_mass']]*2,color=COLORS[arm],lw=2.5)
            positive=[e for e in efficiency if e>0]
            if positive:eff.plot([gi+offset-.085,gi+offset+.085],[sum(efficiency)/4]*2,color=COLORS[arm],lw=2.5)
        for chart,ax in zip(('full','diagonal'),axes[1]):
            bins=[p['radial_partitions'][chart]['A_T'] for p in populations]
            weights=np.asarray([[r['mass'] for r in b] for b in bins])
            totals=np.asarray([p['groups']['A_T']['physical']['mass'] for p in populations])
            if not np.allclose(weights.sum(axis=1),totals,rtol=2e-10,atol=0):
                raise ValueError('A_T radial physical masses do not sum to population target')
            for values,total in zip(weights,totals):
                if total>0:ax.plot(range(6),values/total,color=COLORS[arm],alpha=.20,lw=.9)
            pooled=weights.sum(axis=0);denominator=pooled.sum()
            if denominator>0:
                ax.plot(range(6),pooled/denominator,'o-',color=COLORS[arm],lw=2,label=LABELS[arm])
    for ax,title,ylabel in ((mass,'Physical regional masses','Importance mass [Å³ × normalized Haar]'),
                             (eff,'Importance concentration per CPU','Importance ESS / (geometry + scoring CPU s)')):
        ax.set_yscale('log');ax.set_xticks(range(4),TICKS,rotation=12)
        ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
    mass.text(.02,.02,'Dots: four populations; bars: arithmetic means',transform=mass.transAxes,fontsize=8)
    eff.text(.02,.02,'Weight ESS, not trajectory mixing ESS',transform=eff.transAxes,fontsize=8)
    for chart,ax in zip(('full','diagonal'),axes[1]):
        ax.set_title(f'A-with-full-T weight by {chart} chart radius')
        ax.set_xticks(range(6),['[0,6)','[6,12)','[12,24)','[24,48)','[48,96)','[96,∞)'])
        ax.set_xlabel('Squared Mahalanobis radius');ax.set_ylabel('Fraction of estimated A-with-full-T mass')
        ax.set_ylim(0,1.03);ax.grid(axis='y',alpha=.2)
    axes[1,0].legend(frameon=False,fontsize=9)
    axes[1,1].text(.02,.98,'Bold: pooled mass fractions\nFaint: each independent population',
        transform=axes[1,1].transAxes,va='top',fontsize=8)
    fig.suptitle('Frozen protein environment: conditional importance comparison\n'
                 'rd = 1.5 Å, z = 0.035 Å⁻³ · 4 × 4,096 attempts per arm · two positive weights per valid pose',fontsize=13)
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    receipt=dict(complete=True,passed=True,source_report=str(source_path.resolve()),source_report_sha256=sha(source_path),
        source_sha256=sha(__file__),zeros_omitted_from_log_axes=zero_count,
        outputs={name:sha(out/name) for name in ('comparison.png','comparison.pdf')},
        scope='Conditional frozen-context evidence only. Region labels describe coarse patches, not independently determined native registry. No finite-system assembly or stability conclusion.',
        radial_scope='Each chart separately partitions A_T. Pooled fractions combine equal-size independent populations; faint lines expose population disagreement.')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    plot(json.loads(args.report.read_bytes()),args.out,args.report)

if __name__=='__main__':main()
