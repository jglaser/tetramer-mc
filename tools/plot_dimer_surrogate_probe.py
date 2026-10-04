"""Plot completed conditional surrogate diagnostics, with all eight streams."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.root/'analysis.json').read_text())
    assert report['complete']
    for path,digest in report['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    refs={r['id']:r for r in json.loads((args.root/'references.json').read_text())}
    rows=[json.loads(s) for s in (args.root/'execution/scores.jsonl').read_text().splitlines()]
    args.output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,3,figsize=(13,4.6))
    fig.subplots_adjust(left=.065,right=.99,bottom=.25,top=.82,wspace=.33)
    colors=['#c66a22','#146ca0']
    for index,n in enumerate([4096,16384]):
        selected=[r for r in rows if r['raw_prefix']==n and r['new'] is not None]
        physical=[refs[r['id']]['noisy_delta_physical_log_weight'] for r in selected]
        error=[r['delta_log_surrogate']-x for r,x in zip(selected,physical)]
        axes[0].scatter(physical,error,color=colors[index],s=22,alpha=.8,label=f'{n:,} raw points')
        stats=[g for g in report['groups'] if g['raw_prefix']==n]
        axes[1].plot(range(8),[g['rmse'] for g in stats],marker='o',color=colors[index])
        axes[2].plot(range(8),[1000*g['mean_score_cpu'] for g in stats],marker='o',color=colors[index])
    axes[0].axhline(0,color='black',lw=.7)
    axes[0].axhspan(-1,1,color='gray',alpha=.12)
    axes[0].set(xlabel='Noisy physical log-weight change',ylabel='Guide − physical estimate (kBT)',title='All 36 saved candidate endpoints')
    axes[0].legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_xticks(range(8),[f'{k:03}' for k in range(8)])
        ax.set_xlabel('Saved chain ID')
    axes[1].set(ylabel='RMS discrepancy (kBT)',title='Each chain retained separately')
    axes[2].set(ylabel='Mean CPU time per score (ms)',title='Includes all 128 slots and failures',ylim=(0,None))
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('Fixed-cloud many-body overlap guide: consistency and cost',fontsize=14)
    fig.text(.01,.025,'Nested prefixes reuse the clouds that guided these proposals. Descriptive comparison, not an independent error certificate.\n'
             'Physical reference uses saved Poisson counts (mean estimated SE 1.14 kBT). No physical sampling speedup measured.',fontsize=9)
    for ext in ['png','svg']:fig.savefig(args.output/f'surrogate-score.{ext}',dpi=170,bbox_inches='tight')
    inputs=[args.root/'analysis.json',args.root/'execution/scores.jsonl',args.root/'references.json']
    with (args.output/'receipt.json').open('x') as f:
        json.dump(dict(complete=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            new_geometry_queries=0,new_statistics_estimated=0),f,indent=2)


if __name__=='__main__':main()
