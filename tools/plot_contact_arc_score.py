#!/usr/bin/env python3
"""Archive a scientific figure from completed score-only outputs; no rescoring."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def plot(execution,out):
    execution,out=Path(execution).resolve(),Path(out).resolve()
    require(not out.exists(),'Fresh plot output directory required')
    bindings={}
    def bind(path,expected=None):
        digest=sha(path);require(expected is None or digest==expected,'Completed source changed: '+str(path))
        bindings[str(path)]=digest;return path
    state=read(bind(execution/'status.json'))
    require(state['complete'],'Score execution and independent audit must finish before plotting')
    protocol=read(bind(execution/'protocol.json',state['protocol_sha256']))
    for name,digest in state['output_sha256'].items():bind(execution/name,digest)
    data=read(execution/'analysis.json');audit=read(execution/'independent-audit.json')
    require(data['complete'] and audit['complete'] and data['queries']==206 and data['candidate_densities']==412,
            'Fixed allocation or audit differs')
    preparation=Path(protocol['preparation'])
    manifest=read(bind(preparation/'freeze.json',protocol['preparation_freeze_sha256']))
    for name,digest in manifest['files'].items():bind(preparation/name,digest)
    old=read(preparation/'saved-scores.json')['rows']
    new=[json.loads(line) for line in (execution/'rust/scores.jsonl').read_text().splitlines()]
    require([r['id'] for r in old]==[r['id'] for r in new],'Saved query order differs')
    colors=['#495e78','#c98d55','#168a7c','#dfab75','#08786d']
    laws=['baseline92','uniform_phi92','localized_phi92','arc_uniform_phi92','arc_localized_phi92']
    labels=['Old92','Distance\nuniform φ','Distance\nlocalized φ','Arc\nuniform φ','Arc\nlocalized φ']
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.5),gridspec_kw={'width_ratios':[1.08,1.]})
    ax=axes[0];x=np.arange(len(laws));moments=data['critical_source_diagnostics']
    for source,marker,offset,color in [('baseline','o',-.055,'#625590'),('expanded','s',.055,'#168a7c')]:
        m=moments[source];baseline=m['laws']['baseline92']['log_second_moment']
        values=[math.exp(m['laws'][law]['log_second_moment']-baseline) for law in laws]
        ax.plot(x+offset,values,marker=marker,color=color,lw=1.6,label=f"Original source: {m['rows']} critical rows")
        for j in (3,4):
            ess=m['laws'][laws[j]]['contribution_ESS']
            ax.annotate(f'ESS {ess:.2f}',(j+offset,values[j]),xytext=(0,10 if source=='baseline' else -17),
                textcoords='offset points',ha='center',fontsize=8,color=color)
    ax.axhline(1,color='#aaa',linestyle='--',lw=1);ax.set_yscale('log')
    ax.set_xticks(x,labels);ax.set_ylabel('Paired second moment / old92\nSmaller is favorable on these saved rows')
    ax.set_title('Critical competing tail: retrospective subsets\n4-row and 74-row sources stay separate',fontsize=11)
    ax.legend(frameon=False,fontsize=9,loc='best');ax.grid(axis='y',alpha=.12)
    ax=axes[1];classes=['native_R5','native_complement','competing','invalid']
    for j,(arm,color,offset,label) in enumerate([('uniform_phi92','#c98d55',-.17,'Arc: uniform φ'),
                                                ('localized_phi92','#168a7c',.17,'Arc: localized φ')]):
        distributions=[]
        for kind in classes:
            ids=[i for i,r in enumerate(old) if r['group']=='breadth' and r['metadata']['coverage_class']==kind]
            require(len(ids)==32,'Missing breadth rows')
            distributions.append([new[i]['arms'][j]['log_proposal_density']-old[i]['old_log_q']['baseline92'] for i in ids])
        b=ax.boxplot(distributions,positions=np.arange(4)+offset,widths=.27,patch_artist=True,
            manage_ticks=False,whis=(0,100),showfliers=False,
            medianprops=dict(color='white',linewidth=1.5))
        for patch in b['boxes']:patch.set_facecolor(color);patch.set_alpha(.8)
        ax.plot([],[],color=color,lw=7,label=label)
    ax.axhline(0,color='#aaa',linestyle='--',lw=1)
    ax.axhline(-math.log(2),color='#8f7777',linestyle=':',lw=1,label='Defensive lower bound')
    ax.set_xticks(range(4),['Native R5','Native\ncomplement','Competing','Invalid'])
    ax.set_ylabel('log(candidate q / old92 q)')
    ax.set_title('Breadth: all 32 saved poses per class\nBoxes: quartiles; whiskers: full saved range',fontsize=11)
    ax.legend(frameon=False,fontsize=8,loc='best')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    cpu=data['combined_score_CPU_seconds'];geo=data['geometry_CPU_seconds']
    fig.suptitle('Hard-free arcs: full proposal-density evaluation, no new physical sampling',fontsize=14)
    fig.text(.5,.04,f"206 fixed poses × 2 laws; {data['distinct_circles']:,} distinct circles. "
        f"Joint full-q CPU {cpu:.3f} s, including shared geometry {geo:.3f} s ({data['full_score_ms_per_unique_query']:.2f} ms/pose).",
        ha='center',fontsize=9)
    fig.text(.5,.01,'Low contribution ESS and retrospective selection limit interpretation. No new pose draws, Poisson clouds, physical masses or sampling-speed measurements.',
        ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.93))
    out.mkdir(parents=True)
    for suffix in ('png','svg'):fig.savefig(out/f'contact-arc-score.{suffix}',dpi=180)
    plt.close(fig)
    shutil.copy2(__file__,out/'source.py')
    write(out/'summary.json',dict(schema='contact-arc-score-plot-v1',complete=True,source_sha256=bindings,
        score_analysis=data,independent_audit_summary={k:v for k,v in audit.items() if k not in ('input_sha256',)},
        physical_campaign_ready=False,scope='Read-only report of completed fixed scores; no rescoring or new inference.'))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(out=str(out),summary_sha256=sha(out/'summary.json'),source_files_bound=len(bindings)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.execution,a.out)
