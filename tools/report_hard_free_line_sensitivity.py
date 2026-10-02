#!/usr/bin/env python3
"""Display completed sensitivity controls; never read raw poses or replay MC."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[_key] = '1'
import argparse
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_r4_smc_control import Ledger,read,require,sha,write
from analyze_hard_free_line_sensitivity import PLAN,SCHEMA,REFERENCE,CONTROLS,validate_summary
from plot_hard_free_line_physical import load_completed,REGIONS
from analyze_mobile_native_pocket import local_sources

LABEL = dict(reference='Reference: α=.5, λ/z=128',alpha02='α=.2, λ/z=128',lambda64='α=.5, λ/z=64')
COLORS = dict(reference='#637c9b',alpha02='#b27620',lambda64='#008573')


def load(root):
    root = Path(root).resolve(); ledger = Ledger()
    state = read(ledger.bind(root/'status.json'))
    require(state['complete'] is True and state['phase'] == 'complete','Wait for all sampling, auditing and classification')
    require(len(state['jobs']) == len(state['audits']) == 8 and all(j['status'] == 'complete' and j['returncode'] == 0
        for j in state['jobs']+state['audits']),'All eight jobs and audits must complete')
    protocol = read(ledger.bind(root/'protocol.json',state['protocol_sha256']))
    current = read(ledger.bind(root/'comparison/analysis.json',state['comparison_sha256']))
    comparison = read(ledger.bind(root/'sensitivity-comparison.json',state['sensitivity_comparison_sha256']))
    require(current['protocol_sha256'] == state['protocol_sha256'] and current['schema'] == SCHEMA,
            'Current analysis belongs to another protocol')
    require(comparison['schema'] == 'hard-free-line-sensitivity-comparison-v1' and comparison['complete'] is True
            and comparison['plan'] == protocol['sensitivity_comparison'] == PLAN,'Comparison plan changed')
    validate_summary(current,SCHEMA,CONTROLS)
    pilot_root = Path(protocol['pilot_evidence']['root'])
    pilot,bindings = load_completed(pilot_root)
    for path,digest in bindings.items(): ledger.bind(path,digest)
    pilot_protocol = read(pilot_root/'protocol.json')
    require(protocol['physical_activity'] == .035 and protocol['depletant_radius'] == 1.5
            and all(protocol[k] == pilot_protocol[k] for k in
                    ('physical_activity','depletant_radius','region_sha256','shape_sha256')),'Physical target identity changed')
    require(sha(pilot_root/'comparison/analysis.json') == protocol['pilot_evidence']['comparison_sha256'],
            'Comparison uses another completed pilot')
    for path,digest in comparison['input_sha256'].items(): ledger.bind(path,digest)
    require(comparison['input_sha256'][str((root/'comparison/analysis.json').resolve())] == state['comparison_sha256']
            and comparison['input_sha256'][str((pilot_root/'comparison/analysis.json').resolve())]
                == protocol['pilot_evidence']['comparison_sha256'],'Comparison inputs differ')
    validate_summary(pilot,'hard-free-line-physical-comparison-v1',[dict(REFERENCE,id='baseline',beta=0.),REFERENCE])
    require(not comparison['stages_pooled'] and not comparison['full_vessel_gate_open']
            and not comparison['assembly_gate_open'],'Changed inference scope')
    for path in local_sources(__file__).values(): ledger.bind(path)
    return dict(reference=pilot['arms']['conditioned'],**current['arms']),comparison,ledger


def summarize(arms,comparison):
    efficiencies = {}
    for control in ('alpha02','lambda64'):
        efficiencies[control] = {}
        for region in REGIONS:
            a = arms[control]['observed_importance_ESS_per_cpu_second'][region]
            b = arms['reference']['observed_importance_ESS_per_cpu_second'][region]
            observed = all(arms[k]['estimates'][region]['row_uncertainty']['log_Qz'] is not None for k in ('reference',control))
            ratio = a/b if observed and a is not None and b is not None and a > 0 and b > 0 else None
            efficiencies[control][region] = dict(control=a,reference=b,ratio=ratio)
    return dict(checks=comparison['checks'],sensitivity_checks_passed=comparison['sensitivity_checks_passed'],
        free_energy_intervals=comparison['stage_free_energy_intervals'],quality=comparison['stage_quality'],
        regional_comparisons=comparison['regional_comparisons'],contrast_comparisons=comparison['free_energy_contrast_comparisons'],
        failed_material_strata=comparison['failed_material_strata'],all_stratum_comparisons=comparison['all_stratum_comparisons'],
        importance_efficiency=efficiencies,physical_target=dict(radius_A=1.5,activity_A_minus_3=.035),
        scope='Fixed two-neighbor R4 integral. Independent four-population controls; all attempted zeros retained. '
              'Importance ESS per sampler CPU is descriptive, not contact-mixing ESS. '
              'No full-vessel, finite-system or unseen-mode conclusion; unobserved mass remains unresolved.')


def report(root,out):
    out = Path(out).resolve(); require(not out.exists(),'Fresh report destination required')
    arms,comparison,ledger = load(root); result = summarize(arms,comparison)
    fig,axes = plt.subplots(1,3,figsize=(15.5,5),gridspec_kw=dict(width_ratios=[1.1,1.5,1.]))
    ax = axes[0]
    for i,name in enumerate(LABEL):
        value = result['free_energy_intervals'][name]
        if value.get('observed'):
            ax.errorbar(value['beta_F_native_minus_noentry'],i,xerr=value['halfwidth_95'],fmt='o',capsize=4,color=COLORS[name])
        else: ax.text(.05,i,'unobserved',transform=ax.get_yaxis_transform())
    ax.set_yticks(range(3),list(LABEL.values())); ax.set_ylim(-.5,2.5)
    ax.set_xlabel('β(F_native − F_competing)'); ax.set_title('Paired-population 95% intervals\nStudent-t, 3 degrees of freedom',fontsize=11)
    ax = axes[1]
    for ci,name in enumerate(('alpha02','lambda64')):
        for ri,region in enumerate(REGIONS):
            ratio = result['importance_efficiency'][name][region]['ratio']; x = ri+(ci-.5)*.34
            if ratio is not None:
                ax.bar(x,ratio,width=.3,color=COLORS[name],label=LABEL[name] if ri == 0 else None)
                ax.annotate(f'{ratio:.2f}×',(x,ratio),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
            else: ax.text(x,.05,'unresolved',rotation=90,ha='center',transform=ax.get_xaxis_transform(),fontsize=8)
    ax.axhline(1.,color='#888',ls='--',lw=1); ax.set_xticks(range(4),['Native\ntotal','Competing','Old R5\nnative','Remaining\nnative'])
    ax.set_ylabel('Control / reference importance ESS per sampler CPU'); ax.set_title('Observed integration efficiency',fontsize=11); ax.legend(frameon=False,fontsize=8)
    ax = axes[2]; families = ('radial','angular','orthant')
    for ci,name in enumerate(('alpha02','lambda64')):
        counts = [sum(s['control'] == name and s['family'] == family for s in result['failed_material_strata']) for family in families]
        ax.bar([i+(ci-.5)*.34 for i in range(3)],counts,width=.3,color=COLORS[name])
    ax.set_xticks(range(3),families); ax.set_ylabel('Failed material Qz strata'); ax.set_title('All failures retained in report\nAbsolute 0.2 and linear 3-SE tests',fontsize=11)
    for ax in axes: ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.12)
    fig.suptitle('Completed probability / cloud-intensity controls — fixed R4, not assembly stability',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.92)); out.mkdir(parents=True)
    for suffix in ('png','svg'): fig.savefig(out/f'sensitivity.{suffix}',dpi=180)
    plt.close(fig)
    def number(value): return 'unobserved' if value is None else f'{value:.6g}'
    lines=['# Fixed probability and cloud-intensity controls','',result['scope'],'',
        f"Sensitivity diagnostics passed: **{result['sensitivity_checks_passed']}**.",'',
        '| Check | Passed |','|---|---|']
    lines += [f'| {k} | {v} |' for k,v in result['checks'].items()]
    lines += ['','## Native versus competing free energy','', '| Arm | ΔF/kBT | Population 95% half-width |','|---|---:|---:|']
    for name,v in result['free_energy_intervals'].items():
        lines.append(f"| {LABEL[name]} | {number(v.get('beta_F_native_minus_noentry'))} | {number(v.get('halfwidth_95'))} |")
    lines += ['','## Regional quality','', '| Arm | Region | Population RSE | Importance ESS | Largest draw | Passed |','|---|---|---:|---:|---:|---|']
    for name,regions in result['quality'].items():
        for region,v in regions.items():
            values = v.get('values',{})
            lines.append(f"| {name} | {region} | {number(values.get('population_RSE'))} | {number(values.get('importance_ESS'))} | {number(values.get('largest_draw'))} | {v['passed']} |")
    lines += ['','## Every regional mass comparison','', '| Control | Region | Kind | Log control/reference | Combined log delta SE | Absolute pass | Linear SE pass | Passed |','|---|---|---|---:|---:|---|---|---|']
    for name,regions in result['regional_comparisons'].items():
        for region,kinds in regions.items():
            for kind,c in kinds.items():
                lines.append(f"| {name} | {region} | {kind} | {number(c.get('log_control_minus_reference'))} | {number(c.get('combined_population_log_delta_SE'))} | {c.get('absolute_passed',False)} | {c.get('SE_passed',False)} | {c['passed']} |")
    lines += ['','## Every failed material stratum','',f"Count: {len(result['failed_material_strata'])}.",'',
        '| Control | Family | Region | Bin | Log ratio | Absolute pass | Linear SE pass |','|---|---|---|---:|---:|---|---|']
    for s in result['failed_material_strata']:
        c = s['comparisons']['Qz']; lines.append(f"| {s['control']} | {s['family']} | {s['region']} | {s['bin']} | {number(c.get('log_control_minus_reference'))} | {c.get('absolute_passed',False)} | {c.get('SE_passed',False)} |")
    lines += ['','## Observed importance ESS per sampler CPU','', '| Control | Region | Control/reference |','|---|---|---:|']
    for name,regions in result['importance_efficiency'].items():
        for region,value in regions.items(): lines.append(f"| {name} | {region} | {number(value['ratio'])} |")
    lines += ['','![Completed sensitivity comparison](sensitivity.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    ledger.recheck(); result.update(schema='hard-free-line-sensitivity-display-v1',complete=True,input_sha256=ledger.files,
        new_pose_draws=0,new_clouds=0,new_geometry_queries=0,new_native_classifier_calls=0)
    write(out/'analysis.json',result); shutil.copy2(__file__,out/'source.py')
    write(out/'freeze.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); value = report(args.root,args.out)
    print(dict(complete=value['complete'],checks=value['checks'],failed_material_strata=len(value['failed_material_strata'])))
