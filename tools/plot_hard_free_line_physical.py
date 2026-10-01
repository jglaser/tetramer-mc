#!/usr/bin/env python3
"""Display only a completed, controller-bound regional physical comparison.

Reads archived aggregate analysis, never raw rows, labels, poses or clouds.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REGIONS=('registered_native_entry','contact_no_native_entry','old_R5_intersection_native','remaining_R4_native')
LABELS=('Native total','Competing','Native in old R5','Native remainder')
ARMS=('baseline','conditioned')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)


def load_completed(execution):
    root=Path(execution).resolve();state=read(root/'status.json')
    require(state['complete'] is True and state['phase']=='complete','Wait for complete physical sampling, audits and classification')
    require(len(state['jobs'])==8 and len(state['audits'])==8
        and all(j['status']=='complete' and j['returncode']==0 for j in state['jobs']+state['audits']),
        'Every physical population and independent audit must complete')
    protocol=root/'protocol.json';analysis=root/'comparison/analysis.json'
    require(sha(protocol)==state['protocol_sha256'] and sha(analysis)==state['comparison_sha256'],'Controller-bound analysis/protocol changed')
    p=read(protocol);a=read(analysis)
    require(a['complete'] is True and a['schema']=='hard-free-line-physical-comparison-v1'
        and a['protocol_sha256']==sha(protocol),'Unexpected or unbound physical analysis')
    require(a['total_unconditional_draws']==p['total_unconditional_draws'],'Attempted denominator changed')
    require(not a['diagnostics']['full_vessel_gate_open'] and not a['diagnostics']['assembly_gate_open'],
            'This regional summary cannot establish full-vessel or assembly stability')
    require(set(a['arms'])==set(ARMS),'Unexpected comparison arms')
    for name,arm in a['arms'].items():
        require(len(arm['populations'])==4 and len({v['seed'] for v in arm['populations']})==4,'Expected four independent populations')
        n=sum(v['samples'] for v in arm['populations'])
        for region in REGIONS:
            e=arm['estimates'][region]
            require(len(e['populations'])==4 and e['row_uncertainty']['draws']==n,'Region dropped an attempted denominator')
    bindings={str(p):sha(p) for p in (root/'status.json',protocol,analysis,Path(__file__).resolve())}
    return a,bindings


def summarize(data):
    ratios={}
    for region in REGIONS:
        values={name:data['arms'][name]['observed_importance_ESS_per_cpu_second'][region] for name in ARMS}
        observed=all(data['arms'][name]['estimates'][region]['row_uncertainty']['log_Qz'] is not None for name in ARMS)
        ratio=values['conditioned']/values['baseline'] if observed and all(v is not None and v>0 for v in values.values()) else None
        ratios[region]=dict(ESS_per_CPU=values,conditioned_over_baseline=ratio,
            status='observed importance diagnostic' if ratio is not None else 'unresolved or unobserved; not a zero efficiency claim')
    failed=[x for x in data['diagnostics']['all_stratum_comparisons'] if x['significant'] and not x['comparison']['passed']]
    unobserved=[]
    for name in ARMS:
        for region in REGIONS:
            for i,pop in enumerate(data['arms'][name]['estimates'][region]['populations']):
                if pop['log_Qz'] is None:unobserved.append(dict(arm=name,region=region,population=pop.get('id',f'r{i:02}')))
    return dict(efficiency_ratios=ratios,failed_material_strata=failed,failed_material_strata_count=len(failed),
        unobserved_population_masses=unobserved,free_energy_intervals=data['diagnostics']['free_energy_intervals'],
        quality=data['diagnostics']['quality'],regional_comparisons=data['diagnostics']['regional_comparisons'],
        unconditional_draws=data['total_unconditional_draws'],full_vessel_resolved=False,finite_assembly_resolved=False,
        scope='Four independent linear population mass estimates displayed as log mass. Importance ESS per sampler CPU is descriptive, not contact-mixing ESS or a converged physical speedup. Regional checks do not establish full-vessel or finite-assembly stability; unobserved mass is unresolved, not zero.')


def plot(execution,out):
    out=Path(out).resolve();require(not out.exists(),'Fresh plot destination required')
    data,bindings=load_completed(execution);summary=summarize(data)
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.5),gridspec_kw={'width_ratios':[1.05,1.5,1.]})
    colors={'baseline':'#627fa5','conditioned':'#008674'}
    ax=axes[0]
    for i,region in enumerate(REGIONS):
        ratio=summary['efficiency_ratios'][region]['conditioned_over_baseline']
        if ratio is None:ax.text(i,.1,'unresolved',rotation=90,ha='center',va='bottom',transform=ax.get_xaxis_transform(),fontsize=9)
        else:ax.bar(i,ratio,color=colors['conditioned']);ax.annotate(f'{ratio:.3g}x',(i,ratio),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(1,color='#777',ls='--',lw=1);ax.set_yscale('log');ax.set_ylabel('Conditioned / baseline importance ESS per CPU')
    ax.set_xticks(range(4),['Native\ntotal','Competing','Old R5\nnative','Remaining\nnative']);ax.set_title('Observed importance efficiency',fontsize=11)
    ax=axes[1]
    for ai,name in enumerate(ARMS):
        for ri,region in enumerate(REGIONS):
            pops=data['arms'][name]['estimates'][region]['populations']
            for pi,pop in enumerate(pops):
                if pop['log_Qz'] is not None:
                    x=ri+(ai-.5)*.25+(pi-1.5)*.035
                    ax.scatter(x,pop['log_Qz'],color=colors[name],marker='o' if ai==0 else 's',s=30,
                        label=name if ri==0 and pi==0 else None)
            mean=data['arms'][name]['estimates'][region]['population_uncertainty']['log_Qz']
            if mean is not None:ax.plot([ri+(ai-.5)*.25-.075,ri+(ai-.5)*.25+.075],[mean,mean],color=colors[name],lw=2)
    ax.set_xticks(range(4),['Native\ntotal','Competing','Old R5\nnative','Remaining\nnative']);ax.set_ylabel('log of independent linear population mass Qz')
    ax.set_title('Four populations per arm and region\nShort bars: log of mean linear mass',fontsize=11)
    ax.legend(frameon=False,fontsize=9)
    if summary['unobserved_population_masses']:ax.text(.02,.02,f"{len(summary['unobserved_population_masses'])} unobserved population masses\nlisted explicitly in report; no zero plotted",transform=ax.transAxes,fontsize=8,va='bottom')
    ax=axes[2]
    for i,name in enumerate(ARMS):
        interval=summary['free_energy_intervals'][name]
        if interval.get('observed'):
            mid=interval['beta_F_native_minus_noentry'];half=interval['halfwidth_95']
            ax.errorbar(mid,i,xerr=half,fmt='o',capsize=4,color=colors[name])
        else:ax.text(.05,i,'unresolved',transform=ax.get_yaxis_transform(),va='center',color=colors[name])
    ax.set_yticks([0,1],ARMS);ax.set_ylim(-.5,1.5);ax.axvline(0,color='#aaa',ls='--',lw=1)
    ax.set_xlabel('β(F_native − F_competing)');ax.set_title('Paired-population 95% intervals\nStudent-t, 3 degrees of freedom',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)
    fig.suptitle('Completed fixed-R4 physical comparison — full-vessel and assembly questions remain open',fontsize=13)
    fig.text(.5,.03,f"All {summary['unconditional_draws']:,} attempts retained; {summary['failed_material_strata_count']} failed material strata listed without truncation in report.md. No raw rows replayed or reclassified.",ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.92));out.mkdir(parents=True)
    for suffix in ('png','svg'):fig.savefig(out/f'hard-free-line-physical.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# Completed hard-free-line physical comparison','',summary['scope'],'',
        '| Region | Conditioned / baseline importance ESS per CPU |','|---|---:|']
    for region,label in zip(REGIONS,LABELS):
        ratio=summary['efficiency_ratios'][region]['conditioned_over_baseline'];lines.append(f"| {label} | {'unresolved' if ratio is None else f'{ratio:.6g}'} |")
    lines+=['','## Native versus competing free energy','']
    for name,v in summary['free_energy_intervals'].items():
        lines.append(f"- {name}: "+(f"{v['beta_F_native_minus_noentry']:.6g} ± {v['halfwidth_95']:.6g} kBT (population-based 95% interval)." if v.get('observed') else 'unresolved; one or both masses were unobserved.'))
    lines+=['','## Regional quality gates','',
        'These archived gates remain separate from the free-energy interval; a narrow interval does not override a failed mass-quality check.','',
        '| Arm | Region | Population RSE | Importance ESS | Largest draw fraction | Passed |',
        '|---|---|---:|---:|---:|---|']
    def number(v):return 'unobserved' if v is None else f'{v:.6g}'
    for name,regions in summary['quality'].items():
        for region,gate in regions.items():
            values=gate.get('values',{})
            lines.append(f"| {name} | {region} | {number(values.get('population_RSE'))} | {number(values.get('importance_ESS'))} | {number(values.get('largest_draw'))} | {gate['passed']} |")
    lines+=['','## Between-arm regional comparisons','',
        '| Region | Log baseline / conditioned | Combined population SE | Absolute gate | SE gate | Passed |',
        '|---|---:|---:|---|---|---|']
    for region,comparison in summary['regional_comparisons'].items():
        lines.append(f"| {region} | {number(comparison.get('log_left_minus_right'))} | {number(comparison.get('combined_population_log_delta_SE'))} | {comparison.get('absolute_passed','unobserved')} | {comparison.get('SE_passed','unobserved')} | {comparison['passed']} |")
    lines+=['','## Every failed material stratum','',f"Count: {len(summary['failed_material_strata'])}.",'',
        '| Family | Region | Bin | Result |','|---|---|---:|---|']
    for item in summary['failed_material_strata']:
        c=item['comparison'];text=(f"log ratio {c['log_left_minus_right']:.6g}; combined SE {c['combined_population_log_delta_SE']:.6g}; absolute pass {c['absolute_passed']}; SE pass {c['SE_passed']}" if c.get('observed') else c.get('reason','unobserved mass'))
        lines.append(f"| {item['family']} | {item['region']} | {item['bin']} | {text} |")
    lines+=['','## Unobserved population masses','']
    if not summary['unobserved_population_masses']:lines.append('None in these four reported regions.')
    for item in summary['unobserved_population_masses']:lines.append(f"- {item['arm']} / {item['region']} / {item['population']}: unobserved, not a physical zero or upper bound.")
    lines+=['','![Regional physical comparison](hard-free-line-physical.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    for p,digest in bindings.items():require(sha(p)==digest,'Archived analysis changed during display')
    summary.update(schema='hard-free-line-physical-display-v1',complete=True,input_sha256=bindings,new_pose_draws=0,new_Poisson_clouds=0,new_native_classifier_calls=0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');shutil.copyfile(__file__,out/'source.py')
    (out/'manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(dict(complete=plot(a.execution,a.out)['complete'],out=str(a.out))))
