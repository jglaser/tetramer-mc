#!/usr/bin/env python3
"""Plot matched A-neighbor masses and genealogy for the frozen q<=2 SMC pilot."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(root, out):
    assert not out.exists(), 'Preserve existing figure'
    assessment_path=root/'assessment-protein-n512.json'
    refpath=root/'matched-independent-references.json'
    assessment=read(assessment_path); assert assessment['passed']
    group=assessment['groups']['protein-n512']
    assert group['matched_reference_sha256']==sha(refpath)
    refs=read(refpath)
    hashes={str(p):sha(p) for p in [assessment_path,refpath,root/'protocol.json',Path(__file__)]}
    fig,axes=plt.subplots(1,3,figsize=(16.4,4.8),layout='constrained',gridspec_kw={'width_ratios':[1.15,1.,1.]})
    methods=[('SMC: four N=512 populations',group,'#d95f02',-.2,False),
             ('Direct importance guide',refs['sources']['basin-normalizer-importance-guided-16384-l64'],'#087f8c',0.,True),
             ('Direct MIS refined',refs['sources']['basin-normalizer-mis-refined-16384-l64'],'#746bb0',.2,True)]
    open_lower=[]
    for label,record,color,shift,row_errors in methods:
        for i,region in enumerate(['native','shoulder','total']):
            value=record['regions'][region];mean=value['mean_Q']
            se=value['row_relative_se']*mean if row_errors else value['Q_se']
            y=math.log(mean)
            upper=math.log(mean+se)
            if mean>se:
                lower=math.log(mean-se)
            else:
                lower=y
                open_lower.append((i+shift,y,color))
            axes[0].errorbar(i+shift,y,yerr=[[y-lower],[upper-y]],fmt='o',color=color,
                             capsize=4,label=label if not i else None)
        if not row_errors:
            for i,region in enumerate(['native','shoulder','total']):
                values=record['regions'][region]['population_Q']
                for j,value in enumerate(values):
                    if value:axes[0].scatter(i+shift+(j-1.5)*.018,math.log(value),s=15,color=color,alpha=.35)
    axes[0].set(xticks=[0,1,2],xticklabels=['Native q≤1','Shoulder 1<q≤2','Total q≤2'],
                ylabel='log Q  [Q in Å³ × normalized Haar]',title='Same A-only target; independent estimators')
    axes[0].legend(fontsize=8,loc='best')
    for x,y,color in open_lower:
        axes[0].annotate('',xy=(x,.02),xycoords=('data','axes fraction'),
                         xytext=(x,y),textcoords='data',arrowprops={'arrowstyle':'->','color':color})
    if open_lower:
        axes[0].text(.03,.02,'Down arrow: lower one-SE limit reaches Q=0',transform=axes[0].transAxes,fontsize=7)
    colors=['#087f8c','#d95f02','#746bb0','#558b2f']
    initial_notes=[]
    for index,pop in enumerate(group['populations']):
        path=root/'results'/pop['id'];stagepath=path/'stages.jsonl';poppath=path/'populations.jsonl'
        assert sha(stagepath)==pop['stage_audit']['stage_sha256']
        assert sha(poppath)==pop['stage_audit']['population_sha256']
        hashes[str(stagepath)]=sha(stagepath);hashes[str(poppath)]=sha(poppath)
        stages=[json.loads(line) for line in stagepath.open()]
        with poppath.open() as f:first=json.loads(f.readline())['particles']
        initial_native=sum(p['q']<=.5 for p in first)
        initial_notes.append(str(initial_native))
        axes[1].plot([s['t'] for s in stages],[s['family_ess'] for s in stages],color=colors[index],label=f'Population {index}')
        native_path=pop['stage_audit']['native_path']
        assert len(native_path)==len(stages)
        axes[2].plot([s['t'] for s in native_path],[s['native_fraction'] for s in native_path],color=colors[index],label=f'Population {index}')
    axes[1].set(xlabel='Annealing coordinate t (activity changes)',ylabel='Initial-family ESS (population 512)',
                title='Genealogy diagnoses ancestry loss',yscale='log')
    axes[1].legend(fontsize=8,loc='best')
    axes[1].text(.03,.06,'Initial native endpoints / 512: '+', '.join(initial_notes),transform=axes[1].transAxes,fontsize=8)
    axes[2].set(xlabel='Annealing coordinate t (activity changes)',ylabel='Native endpoint fraction',
                title='Native-region coverage during annealing',ylim=(0,None))
    axes[2].text(.03,.94,'Correlated descendants, changing targets',transform=axes[2].transAxes,fontsize=8,va='top')
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('A-neighbor q≤2 control · r_d=1.5 Å · z=0.035 Å⁻³\nBars: one observed SE (whole SMC populations; independent direct rows). Genealogy is not physical mixing.',fontsize=10)
    out.mkdir(parents=True)
    for extension in ['png','svg']:fig.savefig(out/f'smc-shoulder-control.{extension}',dpi=180)
    plt.close(fig)
    hashes.update({str(p):sha(p) for p in out.iterdir() if p.is_file()})
    (out/'provenance.json').write_text(json.dumps({'input_and_output_sha256':hashes,
        'qualifications':'Same A-only fixed q<=2 target. Observed SE excludes unseen-mode uncertainty. Stage genealogy measures ancestry, not equilibrium contact autocorrelation.'},indent=2)+'\n')
    print(out/'smc-shoulder-control.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.out.resolve())
