#!/usr/bin/env python3
"""Plot an audited reset-state diagnostic, with no trajectory/ESS inference."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.analysis.read_text())
    if not (data['complete'] and data['passed'] and not data['failures'] and data['summary']['outer_attempts']==1536):
        raise ValueError('A complete passed fixed allocation is required')
    available={r['atlas'] for r in data['comparisons']}
    # Match the frozen names without assuming a later campaign's spelling.
    names=[(next(x for x in available if prefix in x),label) for prefix,label in
           [('memory','Blind memory'),('fft','Blind FFT'),('native','Native-informed')]]
    groups={(r['atlas'],r['method']):r['summary'] for r in data['comparisons']}
    fig,axes=plt.subplots(1,3,figsize=(16,5.3),constrained_layout=True)
    methods=[('whole_joint','Whole','#527e9e','o'),('factorized','Staged','#ba5a30','^')]
    colors=['#578f62','#5c7ab5','#a96299']
    for (name,label),color in zip(names,colors):
        for method,short,_,marker in methods:
            rows=[r for r in data['rows'] if r['candidate'] and r['atlas']==name and r['method']==method]
            axes[0].scatter([r['log_proposal_correction'] for r in rows],[r['log_depletion_factor'] for r in rows],
                            s=11,alpha=.5,color=color,marker=marker,label=label+' / '+short)
    axes[0].plot([-80,70],[80,-70],color='#555555',linestyle='--',linewidth=1,label='log R = 0')
    axes[0].set_xlabel('Log reverse/forward proposal density')
    axes[0].set_ylabel('Realized log depletion factor')
    axes[0].set_title('971 candidates: two corrections',fontsize=12,pad=15)
    axes[0].legend(frameon=False,fontsize=7,loc='lower left')
    for j,(method,label,color,_) in enumerate(methods):
        quantiles=[groups[(n,method)]['log_ratio']['quantiles'] for n,_ in names]
        xs=[i+(j-.5)*.2 for i in range(3)]
        med=[q['median'] for q in quantiles]
        axes[1].errorbar(xs,med,yerr=[[q['median']-q['p05'] for q in quantiles],[q['p95']-q['median'] for q in quantiles]],
                         fmt='o',capsize=4,color=color,label=label)
        values=[groups[(name,method)]['replay_plus_saved_proposal_cpu_seconds'] for name,_ in names]
        bars=axes[2].bar([i+(j-.5)*.34 for i in range(3)],values,width=.34,label=label,color=color)
        axes[2].bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=4,fontsize=10)
    axes[1].axhline(0,color='#555555',linestyle='--',linewidth=1)
    axes[1].set_title('Candidate log R: median, 5–95% range',fontsize=12,pad=15)
    axes[1].set_ylabel('Log MH ratio (not a free-energy estimate)')
    axes[1].legend(frameon=False,fontsize=9)
    axes[2].set_title('Proposal + replay CPU s / 256 trials',fontsize=12,pad=15)
    axes[2].set_ylim(0,max(groups[(n,m)]['replay_plus_saved_proposal_cpu_seconds'] for n,_ in names for m,_,_,_ in methods)*1.18)
    axes[2].legend(frameon=False,fontsize=9)
    for ax in axes[1:]:
        ax.set_xticks(range(3),[label for _,label in names])
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"Full depletion correction: {data['summary']['accepted']} accepts in 1,536 fixed reset trials",fontsize=17)
    fig.supxlabel('Fixed reset contexts at 1.4 Å, z = 0.0275 Å⁻³, 500 μM. All geometric and physical rejections retained.\n'
                  'Candidate quantile ranges are not confidence intervals. CPU excludes shared setup/output overhead; these are not mixing or ESS estimates.',fontsize=10)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):
        fig.savefig(args.output.with_suffix('.'+suffix),dpi=160)
    svg=args.output.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    args.output.with_suffix('.json').write_text(json.dumps(dict(analysis=str(args.analysis.resolve()),
        analysis_sha256=hashlib.sha256(args.analysis.read_bytes()).hexdigest(),matplotlib=matplotlib.__version__,
        scope='Frozen reset-state physical decisions; point measurements, no native registry or equilibrium/ESS inference.'),indent=2)+'\n')


if __name__=='__main__':main()
