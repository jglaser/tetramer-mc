#!/usr/bin/env python3
"""Plot audited conditional proposal feasibility; no physical acceptance claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = {'blind_memory_allslot64': 'Blind memory (64 slots)',
         'blind_fft512slots': 'Blind discovery (512 slots)',
         'native_informed178': 'Native-informed control'}
LAWS = [('tree','Correlated'),('independent_learned','Independent G'),('defensive','Independent ½U + ½G')]
COLORS = ['#64748b','#2563eb','#c45b25']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.analysis.read_text());config=json.loads(args.config.read_text())
    if not (report.get('complete') and report.get('passed') and len(report['rows'])==4608):
        raise ValueError('Complete successful independent audit required')
    if hashlib.sha256(args.config.read_bytes()).hexdigest()!=report['input_hashes']['config.json']:
        raise ValueError('Plot config differs from audited config')
    cases={c['name']:c for c in config['cases']}
    def both(r):
        if not r.get('hard_valid'):return False
        c=cases[r['case']];edges=r['geometry']['contacts']
        return sorted([c['root'],c['child']]) in edges and sorted([c['root'],c['anchor']]) in edges
    fig,axes=plt.subplots(1,3,figsize=(13.4,4.7),sharey=True)
    summary=[]
    width=.23
    for ax,atlas in zip(axes,config['atlases']):
        name=atlas['name']
        for k,(law,label) in enumerate(LAWS):
            rows=[r for r in report['rows'] if r['atlas']==name and r['law']==law]
            if len(rows)!=512:raise ValueError('Plot arm allocation differs')
            values=[sum(r.get('hard_valid',False) for r in rows)/len(rows),sum(both(r) for r in rows)/len(rows)]
            x=np.arange(2)+(k-1)*width
            ax.bar(x,np.array(values)*100,width,color=COLORS[k],label=label,alpha=.86,zorder=2)
            for ci,c in enumerate(config['cases']):
                cell=[r for r in rows if r['case']==c['name']]
                if len(cell)!=64:raise ValueError('Plot context allocation differs')
                local=[sum(r.get('hard_valid',False) for r in cell)/64,sum(both(r) for r in cell)/64]
                jitter=(ci-3.5)*width/11
                ax.scatter(x+jitter,np.array(local)*100,s=13,facecolors='white',edgecolors=COLORS[k],linewidths=.65,zorder=3)
            summary.append({'atlas':name,'law':law,'attempts':len(rows),'hard_valid':int(round(values[0]*len(rows))),
                            'hard_valid_both_tree_contacts':int(round(values[1]*len(rows)))})
        ax.set_title(NAMES[name],fontsize=11,pad=12)
        ax.set_xticks([0,1],['Hard-valid','Hard-valid + both\ntree contacts'])
        ax.grid(axis='y',color='#e2e8f0',zorder=0)
        ax.spines[['top','right']].set_visible(False)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Fraction of every attempted proposal (%)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False)
    fig.suptitle('Protein dimer destinations: feasible exclusion contacts',fontsize=14,y=.99)
    fig.text(.5,.014,'Bars: 512 attempts per arm. Dots: eight fixed contexts, 64 attempts each. Contacts mean exclusion overlap; no bath acceptance or native classification.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.17,1,.96))
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['png','svg']:
        path=args.output_prefix.with_suffix('.'+extension)
        if path.exists():raise ValueError('Plot output already exists')
        fig.savefig(path,dpi=170,bbox_inches='tight')
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    record={'analysis_sha256':hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
            'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'definition':'hard-valid endpoint and exclusion edges root-child plus root-anchor; no native classifier or physical acceptance',
            'arms':summary}
    with args.output_prefix.with_suffix('.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
if __name__=='__main__':main()
