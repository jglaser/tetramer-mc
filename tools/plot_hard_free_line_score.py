#!/usr/bin/env python3
"""Plot completed feasibility-line score diagnostics without rescoring poses."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def plot(execution,out):
    execution,out=Path(execution).resolve(),Path(out).resolve();require(not out.exists(),'Fresh plot directory required')
    bindings={}
    def bind(path,digest=None):
        actual=sha(path);require(digest is None or actual==digest,'Saved report changed: '+str(path))
        bindings[str(path)]=actual;return path
    state=read(bind(execution/'status.json'));require(state['complete'],'Wait for the completed independent audit')
    bind(execution/'protocol.json',state['protocol_sha256'])
    for name,digest in state['output_sha256'].items():bind(execution/name,digest)
    data=read(execution/'analysis.json');audit=read(execution/'independent-audit.json')
    require(data['complete'] and audit['complete'] and data['queries']==206 and data['reported_densities']==824,
            'Wrong fixed score allocation')
    for path,digest in data['source_sha256'].items():bind(Path(path),digest)
    arms=['x','y','z','xyz'];colors=['#637da0','#ba8752','#8a73a5','#168a7c']
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.2),gridspec_kw={'width_ratios':[1.15,1.4,1.]})
    ax=axes[0]
    for source,color,marker,offset in [('baseline','#625590','o',-.04),('expanded','#168a7c','s',.04)]:
        entry=data['critical_source_diagnostics'][source];m=entry['laws']
        values=[m[a]['second_moment_ratio_vs_old92'] for a in arms]
        ax.plot(np.arange(4)+offset,values,color=color,marker=marker,label=f"{entry['rows']} critical rows",lw=1.5)
        ess=m['xyz']['contribution_ESS']
        ax.annotate(f'ESS {ess:.2f}',(3+offset,values[-1]),xytext=(-3,10 if source=='baseline' else -16),
                    textcoords='offset points',ha='right',fontsize=8,color=color)
    ax.axhline(1,color='#999',linestyle='--',lw=1);ax.set_yscale('log');ax.set_xticks(range(4),['x','y','z','xyz\n(primary)'])
    ax.set_ylabel('Paired second moment / old92');ax.set_title('Critical tail: source groups stay separate\nRetrospective; lower is favorable',fontsize=10)
    ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.12)
    ax=axes[1];classes=['native_R5','native_complement','competing']
    for j,(arm,color) in enumerate(zip(arms,colors)):
        values=[]
        for kind in classes:
            d=data['breadth'][kind][arm];require(d['rows']==32 and d['zero_density_rows']==0,'Target support lost')
            values.append(d['finite_log_ratio_values'])
        b=ax.boxplot(values,positions=np.arange(3)+(j-1.5)*.18,widths=.15,manage_ticks=False,
            patch_artist=True,showfliers=False,whis=(0,100),medianprops=dict(color='white'))
        for box in b['boxes']:box.set_facecolor(color)
        ax.plot([],[],color=color,lw=5,label=arm)
    ax.axhline(0,color='#999',linestyle='--',lw=1);ax.set_xticks(range(3),['Native R5','Native\ncomplement','Competing'])
    ax.set_ylabel('log(q / old92 q)');ax.set_title('Valid breadth: all 32 poses per class\nQuartiles and full saved range',fontsize=10)
    ax.legend(frameon=False,fontsize=8,ncol=4)
    ax=axes[2];invalid=[data['breadth']['invalid'][a] for a in arms]
    require(all(d['rows']==32 and d['zero_density_rows']==0 for d in invalid),'This fixed in-R4 control must retain uniform support')
    b=ax.boxplot([d['finite_log_ratio_values'] for d in invalid],patch_artist=True,showfliers=False,
        whis=(0,100),medianprops=dict(color='white'))
    for box,color in zip(b['boxes'],colors):box.set_facecolor(color)
    ax.axhline(0,color='#999',linestyle='--',lw=1);ax.set_xticks(range(1,5),['x','y','z','xyz'])
    ax.set_ylabel('log(q / old92 q)');ax.set_title('Invalid control: all 32 poses retained\nLower density is permissible here',fontsize=10)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    cpu=data['full_xyz_density_CPU_seconds'];geo=sum(g['geometry_CPU_seconds'] for g in data['geometry'].values())
    fig.suptitle('Keep the Gaussian measure; condition only on feasible line intervals',fontsize=14)
    fig.text(.5,.05,f"206 fixed queries; xyz evaluated once, x/y/z derived from saved branches. "
        f"Full xyz q: {cpu:.3f} CPU s ({data['full_xyz_ms_per_query']:.2f} ms/query); geometry {geo:.3f} s.",ha='center',fontsize=9)
    fig.text(.5,.015,'No fresh pose draws, Poisson clouds, physical masses or mixing measurements. The favorable second-moment sign is guaranteed; its magnitude and computational cost are diagnostic.',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.93));out.mkdir(parents=True)
    for suffix in ('png','svg'):fig.savefig(out/f'hard-free-line-score.{suffix}',dpi=180)
    plt.close(fig);shutil.copy2(__file__,out/'source.py')
    write(out/'summary.json',dict(complete=True,schema='hard-free-line-score-figure-v1',source_sha256=bindings,
        score_analysis=data,independent_maximum_errors=audit['maximum_errors'],physical_campaign_ready=False,
        scope='Completed saved-score display only; no new geometry or physical inference.'))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(out=str(out),summary_sha256=sha(out/'summary.json'),source_files_bound=len(bindings)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('execution','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();plot(a.execution,a.out)
