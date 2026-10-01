#!/usr/bin/env python3
"""Review completed passive outputs; no geometry, classifier or physical replay."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnose_fitted_kernel_shear import read,sha,require,write_new


def review(root,out):
    require(not out.exists(),'Fresh review directory required')
    analysis=read(root/'analysis.json');status=read(root/'status.json')
    require(status['complete'] and analysis['complete'] and sha(root/'analysis.json')==status['analysis_sha256'],
            'Completed analysis binding changed')
    bindings={str(root/name):sha(root/name) for name in ('analysis.json','status.json','protocol.json')}
    fallback=Counter();by_width=defaultdict(Counter);maximum=defaultdict(float)
    for job in [*status['jobs'],status['coverage_job']]:
        directory=Path(job['directory'])
        for name,digest in job['output_sha256'].items():
            require(sha(directory/name)==digest,'Audited output changed')
            bindings[str(directory/name)]=digest
        audit=read(directory/'independent-audit.json')
        for key,value in audit['maximum_errors'].items():maximum[key]=max(maximum[key],value)
        if job['arm']!='line92' or job['id']=='coverage':continue
        for line in (directory/'samples.jsonl').read_text().splitlines():
            row=json.loads(line);draw=row['draw']
            if not draw['conditional']:continue
            geometry=draw['geometry'];width=str(draw['width_A'])
            if not draw['fallback']:category='successful_conditioning'
            elif geometry.get('empty_reason')=='no_positive_hard_free_length':category='no_hard_free_line'
            elif 'empty_reason' in geometry:category='no_domain_chord'
            elif not geometry['widths'][draw['width_index']]['intervals']:category='no_joint_contact_interval'
            else:category='conditional_mass_floor'
            fallback[category]+=1;by_width[width][category]+=1
    require(sum(fallback.values())==65 and fallback['successful_conditioning']==10,'Selected-branch counts differ')
    stats={}
    for arm,values in analysis['arms'].items():
        populations=values['populations'];n=sum(p['draws'] for p in populations)
        counts={key:sum(p[key] for p in populations) for key in ('hard_shell_capture','shell','conditional','fallback')}
        counts['joint_contact_by_width']=[sum(p['joint_contact_by_width'][i] for p in populations) for i in range(3)]
        errors={}
        for key,i in [('hard_shell_capture',None),('joint_width_0.02',0),('joint_width_0.1',1),('joint_width_0.5',2)]:
            fractions=np.array([p['hard_shell_capture']/p['draws'] if i is None else p['joint_contact_by_width'][i]/p['draws'] for p in populations])
            errors[key]=dict(mean=float(fractions.mean()),population_SE=float(fractions.std(ddof=1)/math.sqrt(len(fractions))),
                populations=fractions.tolist(),scope='Four-population descriptive mean and SE; small passive allocation, no equilibrium inference')
        stats[arm]=dict(draws=n,counts=counts,fractions=errors,
            fresh_ms_per_draw=1000*values['total_fresh_cpu_seconds']/n,
            proposal_draw_and_density_ms_per_draw=1000*sum(p['draw_cpu_seconds']+p['density_cpu_seconds'] for p in populations)/n,
            full_child_cpu_seconds=values['total_child_cpu_seconds'],fresh_cpu_seconds=values['total_fresh_cpu_seconds'],
            probe_cpu_seconds=values['total_probe_cpu_seconds'],
            startup_and_unattributed_cpu_seconds=sum(p['setup_and_unattributed_cpu_seconds'] for p in populations),
            audit_cpu_seconds=sum(p['independent_audit_cpu_seconds'] for p in populations),
            hard_shell_capture_per_fresh_cpu=values['hard_shell_capture_per_fresh_cpu'],
            joint_by_width_per_fresh_cpu=values['joint_by_width_per_fresh_cpu'])
    previous=root.parent/'contact-tail-pilot-20261001/comparison/analysis.json'
    previous_data=read(previous);bindings[str(previous)]=sha(previous)
    old_cpu=previous_data['arms']['expanded']['sampler_cpu_seconds'];old_n=sum(p['samples'] for p in previous_data['arms']['expanded']['populations'])
    result=dict(schema='completed-contact-line-passive-review-v1',complete=True,source_sha256=bindings,
        wall_seconds=status['finished']-status['started'],stats=stats,selected_branch_counts=dict(fallback),
        selected_branch_counts_by_width={k:dict(v) for k,v in by_width.items()},maximum_independent_audit_errors=dict(maximum),
        critical_retrospective_moments=analysis['archived_moment_diagnostics'],coverage=analysis['coverage'],
        fresh_CPU_ratio=stats['line92']['fresh_ms_per_draw']/stats['baseline92']['fresh_ms_per_draw'],
        previous_physical92_ms_per_draw=1000*old_cpu/old_n,
        decision='Do not launch a physical weight pilot for this fixed-x variant. Correctness passed, useful-contact gain unproven, fallback common and passive CPU higher.',
        no_further_jobs=True,new_Poisson_clouds=0,physical_mass_estimates_generated=0)
    out.mkdir(parents=True);write_new(out/'summary.json',result)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),gridspec_kw={'width_ratios':[1.25,1.1,1.]})
    labels=['Hard-valid\ninside R4','Both gaps\n≤0.1 Å','Both gaps\n≤0.5 Å']
    for shift,arm,color in [(-.18,'baseline92','#485e75'),(.18,'line92','#0a8f86')]:
        entries=[stats[arm]['fractions'][key] for key in ['hard_shell_capture','joint_width_0.1','joint_width_0.5']]
        axes[0].bar(np.arange(3)+shift,[e['mean'] for e in entries],width=.34,color=color,label=arm,
                    yerr=[e['population_SE'] for e in entries],capsize=3)
    axes[0].set_xticks(range(3),labels);axes[0].set_ylabel('Fraction of all 256 attempted draws')
    axes[0].set_ylim(0,.34);axes[0].legend(frameon=False,fontsize=9)
    axes[0].set_title('No observed broad-contact gain\nError bars: population SE',fontsize=11)
    categories=['No hard-free\nline','No joint\ncontact interval','Successful\nconditioning']
    counts=[fallback['no_hard_free_line'],fallback['no_joint_contact_interval'],fallback['successful_conditioning']]
    axes[1].bar(range(3),counts,color=['#b47b4d','#d3a580','#0a8f86'])
    for i,c in enumerate(counts):axes[1].text(i,c+.6,str(c),ha='center')
    axes[1].set_xticks(range(3),categories);axes[1].set_ylim(0,40);axes[1].set_ylabel('Of 65 selected conditional branches')
    axes[1].set_title('55 / 65 branches fall back\nZero failures from the mass floor',fontsize=11)
    costs=[stats[a]['fresh_ms_per_draw'] for a in ('baseline92','line92')]
    axes[2].bar(range(2),costs,color=['#485e75','#0a8f86'])
    for i,c in enumerate(costs):axes[2].text(i,c+.08,f'{c:.2f} ms',ha='center')
    axes[2].set_xticks(range(2),['Baseline92','Line92']);axes[2].set_ylim(0,5.4)
    axes[2].set_ylabel('Fresh proposal-only CPU per draw')
    axes[2].set_title(f'{result["fresh_CPU_ratio"]:.2f}× passive cost\nNot a physical-simulation slowdown',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Exact one-coordinate conditioning: validated, but not enough in this pilot',fontsize=13)
    fig.text(.5,.015,'512 fresh proposal draws + 284 archived queries; zero Poisson clouds. Finite-system assembly remains unresolved.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.91))
    for suffix in ('png','svg'):fig.savefig(out/f'contact-line-passive.{suffix}',dpi=180)
    plt.close(fig)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    write_new(out/'manifest.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k:result[k] for k in ('wall_seconds','fresh_CPU_ratio','previous_physical92_ms_per_draw','selected_branch_counts','maximum_independent_audit_errors')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();review(a.root.resolve(),a.out.resolve())
