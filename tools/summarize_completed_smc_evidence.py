#!/usr/bin/env python3
"""Summarize authenticated completed calculations without replaying any audit.

Population means, errors and predeclared comparisons are copied from completed
reports. Genealogy is descriptive; neither descendants nor ancestry counts are
converted into physical independent samples.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from analyze_r4_smc_control import Ledger, read, require, sha, write

REGIONS=('registered_native_entry','old_R5_intersection_native','remaining_R4_native')
TITLES=('All native entry in R4','Native entry inside old R5','Native entry outside old R5')


def summarize(bridge, smc, authentication, out):
    bridge,smc,authentication,out=map(lambda p:Path(p).resolve(),(bridge,smc,authentication,out))
    require(not out.exists(),'Fresh completed-evidence report directory required')
    ledger=Ledger(); auth=read(ledger.bind(authentication))
    require(auth['schema']=='completed-smc-bridge-evidence-authentication-v1' and auth['complete'] is True,
        'Successful terminal artifact authentication required')
    for path,digest in auth['input_sha256'].items():ledger.bind(path,digest)
    reports={name:read(ledger.bind(bridge/(name+'-comparison/analysis.json'))) for name in ('broad','narrow')}
    analyses={name:read(ledger.bind(smc/(name+'-analysis/analysis.json'))) for name in reports}
    require(all(str(bridge/(name+'-comparison/analysis.json')) in auth['input_sha256'] for name in reports),
        'Authentication does not cover these comparisons')
    source=Path(__file__).resolve();ledger.bind(source)
    statistics={name:reports['broad']['arms'][name]['importance_population_statistics'] for name in reports['broad']['arms']}
    require(all(statistics[name]==reports['narrow']['arms'][name]['importance_population_statistics'] for name in statistics),
        'The independent comparisons use different importance populations')
    statistics.update({name+' SMC':report['smc_population_statistics'] for name,report in reports.items()})
    means={name:{region:value['Qz']['estimates'][region] for region in REGIONS} for name,value in statistics.items()}
    diagnostics={}; genealogies={}; quality=[]
    for method,report in reports.items():
        diagnostics[method]={}
        for arm,item in report['arms'].items():
            records=[]
            for family,entries in item['strata'].items():
                for index,entry in enumerate(entries):
                    for region,value in entry.items():
                        if value['decision_relevant'] and region in report['primary_regions']:
                            material=(value.get('absolute_passed') is False and value.get('three_SE_passed') is False)
                            records.append(dict(family=family,index=index,region=region,material_disagreement=material,**value))
            diagnostics[method][arm]=dict(decision_relevant_comparisons=len(records),
                failed=sum(not v['passed'] for v in records),material_disagreements=sum(v['material_disagreement'] for v in records),
                records=records)
            for kind in ('Qz','Q0'):
                quality.extend(dict(method=method,arm=arm,kind=kind,region=region,**v)
                    for region,v in item['masses'][kind].items())
        genealogies[method]=[]
        for population in analyses[method]['populations']:
            terminal,last=population['terminal'],population['stages'][-1]
            require(last['stage']==128 and last['beta']==1.,'Incomplete terminal SMC population')
            genealogies[method].append(dict(id=population['id'],seed=population['seed'],
                sampler_CPU_hours=population['sampler_cpu_seconds']/3600,log_Z=last['log_Z'],
                counts=terminal['counts'],class_distinct_initial_families=terminal['class_distinct_initial_families'],
                descendants_from_initially_other_class=terminal['descendants_from_initially_other_class'],
                ancestry=last['ancestry']))
    bank=statistics['bank']['Qz']['estimates']; noentry_fraction=math.exp(bank['contact_no_native_entry']['log_Q']-bank['total']['log_Q'])
    mass_fractions={name:math.exp(values['remaining_R4_native']['log_Q']-values['registered_native_entry']['log_Q']) for name,values in means.items()}
    hard=[v for v in quality if v['kind']=='Q0' and 'unresolved' not in v]
    result=dict(schema='completed-smc-evidence-summary-v1',complete=True,physical_conclusion='unresolved sampling limitation',
        region_log_masses=means,matched_mass_comparisons=quality,strata_diagnostics=diagnostics,genealogy=genealogies,
        native_complement_fraction_point_estimates=mass_fractions,
        observed_hard_volume_comparisons=dict(count=len(hard),all_passed=all(v['passed'] for v in hard),
            maximum_absolute_log_difference=max(abs(v['log_mass_difference_SMC_minus_importance']) for v in hard)),
        rare_noentry_diagnostic=dict(importance_bank_conditional_fraction_point_estimate=noentry_fraction,
            SMC_terminal_records=sum(p['counts']['total'] for records in genealogies.values() for p in records),
            observed_SMC_noentry=sum(p['counts']['contact_no_native_entry'] for records in genealogies.values() for p in records),
            scope='The importance point estimate is unconverged. This calculation explains why missing terminal hits need not be a sampling defect; it supplies no missing-mass bound.'),
        original_estimator='SMC region mass is terminal normalizer times terminal indicator fraction; arithmetic means across four independent populations, including zero region estimates.',
        scope='Read-only synthesis. No endpoint fraction substituted for a mass, no new draws, no audit replay, no pooling, no new convergence thresholds. '
            'Genealogy is concentration, not physical ESS. Fixed-scaffold R4 evidence cannot decide mobile assembly or bulk stability.',
        input_sha256=ledger.files)
    ledger.recheck();out.mkdir()
    write(out/'summary.json',result)
    colors=['#173f5f']*5+['#b34a35','#268672']; names=list(means); y=np.arange(len(names))
    fig,axes=plt.subplots(1,3,figsize=(13,5.2),sharey=True)
    multiplier=float(t.ppf(.975,3))
    for ax,region,title in zip(axes,REGIONS,TITLES):
        reference=means['bank'][region]['log_Q']
        for i,name in enumerate(names):
            item=means[name][region]
            ax.errorbar(item['log_Q']-reference,i,xerr=multiplier*item['population_relative_SE'],
                fmt='o',color=colors[i],capsize=3,markersize=5)
        ax.axvline(0,color='0.5',ls='--',lw=1);ax.set_title(title,fontsize=11)
        ax.set_xlabel('log mass relative to bank estimate');ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(y,names);axes[0].invert_yaxis()
    fig.suptitle('Matching shape, scaffold, R4 domain and physical measure',fontsize=14)
    fig.text(.5,.025,'Bars: diagnostic 95% t(3) delta intervals from four independent population masses.\n'
        'The bank reference is uncertain; this is not a simultaneous comparison interval or a coverage certificate.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.11,1,.94]);fig.savefig(out/'matching-region-masses.png',dpi=170);fig.savefig(out/'matching-region-masses.svg');plt.close(fig)
    ledger.recheck();write(out/'freeze.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bridge','smc','authentication','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=summarize(a.bridge,a.smc,a.authentication,a.out)
    print(dict(complete=r['complete'],physical_conclusion=r['physical_conclusion'],hard_volume_checks=r['observed_hard_volume_comparisons']))


if __name__=='__main__':main()
