#!/usr/bin/env python3
"""Compare frozen-guide validation campaigns without pooling them with training.

Every original row is streamed again against its audited checksum. Observed
concentration and variance-per-CPU are finite-budget diagnostics, not MCMC
mixing times or evidence that the native mass has converged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_native_factorial_screen import physical_signature


ROOT=Path(__file__).resolve().parents[1]
QBINS=[0.,.1,.2,.4,.6,.8,1.]
COLORS=['#777a80','#207d8e','#b46031','#875193']
NAMES=['Source confirmation\n(training only)','Selected pilot\nGaussian + cover',
       'Wider pilot\nGaussian + cover','Wider confirmation\nGaussian + cover']


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block:=handle.read(1<<20):value.update(block)
    return value.hexdigest()


def label(row,manifest):
    """Null geometric component denotes a guide family, never 'uniform'."""
    if manifest.get('guide') is not None:
        if row['proposal_family']=='guide':
            assert row['proposal_component'] is None
            if row['guide_branch']=='uniform':
                assert row['guide_component'] is None
                return 'guide: uniform cube + Haar'
            assert row['guide_branch']=='learned' and row['guide_component'] is not None
            return f"guide: Gaussian {row['guide_component']}"
        assert row['proposal_family']=='cover' and row['guide_branch'] is None
    mixture=manifest.get('cover_mixture')
    if mixture is not None:
        component=row.get('proposal_component',0)
        assert component is not None
        return f"cover: scale {mixture['scales'][component]:g}"
    return 'cover: uniform outer'


def independent_population_se(logs,combined):
    if combined is None or len(logs)<2:return None
    ratios=[math.exp(v-combined) if v is not None else 0. for v in logs]
    assert abs(sum(ratios)/len(ratios)-1)<1e-9
    return math.sqrt(sum((x-1)**2 for x in ratios)/(len(ratios)*(len(ratios)-1)))


def concentration(path,data):
    """Streaming analogue of compare_native_cover_campaigns.ab_concentration.

    It adds exact guide branch labels, hashes, hard-mass q bins and explicit
    all-zero handling. The legacy paired-sum ratio is retained for comparison.
    """
    campaign=read(path/'manifest.json');audited={p['replicate']:p for p in data['populations']}
    logq=data['regions']['native']['log_normalizer']
    logh=data['hard_regions']['native']['log_normalizer']
    total=square=cloud_sum=hard_total=0.
    qmass=np.zeros(6);qhard=np.zeros(6);qvalid=np.zeros(6,dtype=int)
    components=defaultdict(lambda:{'draws':0,'valid':0,'mass':0.,'hard_mass':0.})
    sample_hashes={};count=0
    for job in campaign['jobs']:
        pop=Path(job['output']);manifest=read(pop/'manifest.json');summary=read(pop/'summary.json')
        expected=audited[pop.name];assert manifest['cloud_replicates']==2 and summary['complete']
        assert manifest['seed']==job['seed'] and manifest['samples']==expected['samples']
        if manifest.get('guide') is not None:
            assert sha(pop/'provenance/guide-model.json')==manifest['guide']['model_sha256']
        digest=hashlib.sha256();n=0
        with (pop/'samples.jsonl').open('rb') as handle:
            for raw in handle:
                digest.update(raw);row=json.loads(raw)
                assert row['draw']==n;n+=1;count+=1
                selected=components[label(row,manifest)];selected['draws']+=1
                if 'zero' in row:continue
                assert logq is not None and logh is not None and 0<=row['q']<=1
                logs=row['cloud_log_weights'];assert len(logs)==2
                invg=row.get('log_hard_weight',math.log(manifest['cover']['volume']))
                maximum=max(logs)
                expected_weight=invg+maximum+math.log(sum(math.exp(x-maximum) for x in logs)/2)
                assert abs(expected_weight-row['log_importance_weight'])<2e-10
                y=math.exp(row['log_importance_weight']-logq)
                h=math.exp(invg-logh)
                a,b=[math.exp(w+invg-logq) for w in logs]
                assert abs((a+b)/2/y-1)<2e-10
                total+=y;square+=y*y;hard_total+=h;cloud_sum+=(a-b)**2/4
                index=min(5,int(np.searchsorted(QBINS,row['q'],side='right'))-1)
                assert index>=0
                qmass[index]+=y;qhard[index]+=h;qvalid[index]+=1
                selected['valid']+=1;selected['mass']+=y;selected['hard_mass']+=h
        assert n==manifest['samples']==expected['samples']
        assert digest.hexdigest()==summary['samples_sha256']==expected['sample_sha256']
        sample_hashes[pop.name]=digest.hexdigest()
    assert count==data['samples']==campaign['total_unconditional_draws']
    if logq is not None:
        assert abs(total/count-1)<1e-9 and abs(hard_total/count-1)<1e-9
        assert abs(total*total/square/data['regions']['native']['ess']-1)<1e-8
    else:assert total==hard_total==0
    centered=square-total*total/count
    fraction=cloud_sum/centered if centered>0 else None
    for component in components.values():
        component['fraction_of_estimated_mass']=component.pop('mass')/total if total else None
        component['fraction_of_estimated_hard_mass']=component.pop('hard_mass')/hard_total if hard_total else None
    return {'q_bin_edges':QBINS,'q_bin_mass_fractions':(qmass/total).tolist() if total else None,
        'q_bin_hard_mass_fractions':(qhard/hard_total).tolist() if hard_total else None,
        'q_bin_valid_counts':qvalid.tolist(),'components':dict(components),
        'paired_cloud_fraction_of_observed_variance':fraction,
        'paired_cloud_fraction_with_unbiased_total_sample_variance':fraction*(count-1)/count if fraction is not None else None,
        'paired_cloud_formula':'sum ((W1-W2)/(2g))^2 / [sum Y^2-(sum Y)^2/N], Y=(W1+W2)/(2g); original zeros retained',
        'paired_cloud_caveat':'No clipping to [0,1]. Two-cloud noise and unseen tails can make this decomposition unreliable; it is descriptive.',
        'sample_sha256_verified_again':sample_hashes}


def summarize(path):
    data=read(path/'assessment-streaming.json')
    assert data['complete'] and data['all_rows_and_hashes_validated']
    assert len({p['samples'] for p in data['populations']})==1
    physical=data['regions']['native'];hard=data['hard_regions']['native']
    logs=[p['regions']['native']['log_normalizer'] for p in data['populations']]
    hard_logs=[p['hard_regions']['native']['log_normalizer'] for p in data['populations']]
    pop_error=independent_population_se(logs,physical['log_normalizer'])
    hard_error=independent_population_se(hard_logs,hard['log_normalizer'])
    assert pop_error==data['independent_population_relative_SE'] or math.isclose(pop_error,data['independent_population_relative_SE'],rel_tol=1e-12)
    cpu=data['cpu_seconds'];point=physical['relative_SE']
    efficiency={'observed_ESS_per_CPU_second':physical['ess']/cpu if cpu>0 else None,
        'observed_point_RSE_squared_times_CPU':point*point*cpu if point is not None else None,
        'observed_population_RSE_squared_times_CPU':pop_error*pop_error*cpu if pop_error is not None else None,
        'scope':'Fixed-budget importance-integration diagnostics; neither a converged efficiency ratio nor an MCMC mixing speedup.'}
    return {'root':str(path),'assessment_sha256':sha(path/'assessment-streaming.json'),
        'campaign_manifest_sha256':sha(path/'manifest.json'),'physical':physical,'hard':hard,
        'hard_volume_A3':data['zero_activity_volume_estimate_A3'],
        'hard_volume_SE_A3':data['zero_activity_volume_SE_A3'],
        'independent_population_relative_SE':pop_error,'hard_independent_population_relative_SE':hard_error,
        'population_logs':logs,'population_hard_logs':hard_logs,
        'population_sizes':[p['samples'] for p in data['populations']],
        'population_native_point_RSE':[p['regions']['native']['relative_SE'] for p in data['populations']],
        'cpu_seconds':cpu,'raw_cloud_points':data['raw_cloud_points'],
        'max_population_wall_seconds':data['max_population_wall_seconds'],
        'guide':data.get('guide'),'concentration':concentration(path,data),'efficiency':efficiency,
        'leading_original_rows':data['top_weights'][:5]}


def linear_mean_difference(first,second):
    """Difference of independent mean estimates divided by observed linear SE.

    Scale out their largest log mass to avoid enormous physical weights.
    No normal approximation, confidence level, or hypothesis test is asserted.
    """
    la=first['physical']['log_normalizer'];lb=second['physical']['log_normalizer']
    if la is None or lb is None:return {'resolved_as_diagnostic':False}
    offset=max(la,lb);a=math.exp(la-offset);b=math.exp(lb-offset)
    result={'resolved_as_diagnostic':True,'log_mass_difference_second_minus_first':lb-la,
        'mass_ratio_second_over_first':math.exp(lb-la) if abs(lb-la)<700 else None,
        'scaled_linear_difference_second_minus_first':b-a,'log_scale':offset,
        'scope':'Independent validation means; observed standard errors may miss tails. These standardized differences are diagnostics, not calibrated significance tests.'}
    for name,key in [('population','independent_population_relative_SE'),('point','relative_SE')]:
        ra=first[key] if name=='population' else first['physical'][key]
        rb=second[key] if name=='population' else second['physical'][key]
        se=math.hypot(a*ra,b*rb) if ra is not None and rb is not None else None
        result[name+'_scaled_linear_standard_error']=se
        result[name+'_standardized_linear_difference']=(b-a)/se if se is not None and se>0 else None
    return result


def make_figure(reports,out):
    fig,axes=plt.subplots(2,3,figsize=(14.8,8.5),layout='constrained')
    rows=list(reports.values());count=len(rows);width=.8/count
    for x,(row,color) in enumerate(zip(rows,COLORS)):
        logs=[v for v in row['population_logs'] if v is not None]
        axes[0,0].scatter(x+np.linspace(-.09,.09,len(logs)),logs,s=24,alpha=.55,color=color)
        if row['physical']['log_normalizer'] is not None:
            axes[0,0].errorbar(x,row['physical']['log_normalizer'],yerr=row['independent_population_relative_SE'],color=color,marker='D',capsize=4,markersize=7)
        maximum=row['physical']['maximum_point_fraction']
        axes[0,1].bar(x,100*maximum if maximum is not None else 0,color=color,width=.65)
        axes[0,1].text(x,(100*maximum if maximum is not None else 0)+2,f"ESS {row['physical']['ess']:.1f}",ha='center',fontsize=9)
        axes[0,2].bar(x,row['cpu_seconds'],color=color,width=.65)
        axes[0,2].text(x,row['cpu_seconds']*1.03,f"N={row['physical']['samples']:,}",ha='center',fontsize=8)
        hs=[math.exp(v)*1e6 for v in row['population_hard_logs'] if v is not None]
        axes[1,0].scatter(x+np.linspace(-.09,.09,len(hs)),hs,s=24,alpha=.55,color=color)
        error=row['hard_independent_population_relative_SE']
        axes[1,0].errorbar(x,row['hard_volume_A3']*1e6,yerr=None if error is None else row['hard_volume_A3']*1e6*error,color=color,marker='D',capsize=4,markersize=7)
        fractions=row['concentration']['q_bin_mass_fractions']
        if fractions is not None:axes[1,1].bar(np.arange(6)+(x-(count-1)/2)*width,np.array(fractions)*100,width=.96*width,color=color,label=NAMES[x].replace('\n',' '))
        fraction=row['concentration']['paired_cloud_fraction_of_observed_variance']
        if fraction is not None:axes[1,2].bar(x,100*fraction,color=color,width=.65)
    for ax in [axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,2]]:ax.set_xticks(range(count),NAMES[:count],fontsize=8)
    axes[0,0].set(title='Native mass: independent populations',ylabel='log Q');axes[0,0].text(.02,.02,'Diamonds: mean; bars: observed delta SE',transform=axes[0,0].transAxes,fontsize=8)
    axes[0,1].set(title='Weight concentration',ylabel='Largest original contribution / %',ylim=(0,110))
    axes[0,2].set(title='Work at the declared budgets',ylabel='Summed sampling CPU seconds');axes[0,2].margins(y=.15)
    axes[1,0].set(title='Hard-valid native volume',ylabel='Q at zero activity / 10⁻⁶ Å³')
    axes[1,1].set_xticks(range(6),['0–.1','.1–.2','.2–.4','.4–.6','.6–.8','.8–1'],fontsize=8)
    axes[1,1].set(title='Where original physical weights lie',xlabel='Original registration error q',ylabel='Estimated native mass / %')
    axes[1,1].legend(fontsize=7)
    axes[1,2].set(title='Paired-cloud noise diagnostic',ylabel='Fraction of observed variance / %')
    for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Independent native AB integration controls — proposals and estimates remain separate',fontsize=14)
    fig.savefig(out/'native-guided-comparison.png',dpi=170);plt.close(fig)


def write_report(result,out):
    lines=['# Native AB guided integration controls','',
        'The confirmation trained the frozen proposals. Each fresh campaign estimates the same original native AB region independently; no training and validation weights are pooled.','',
        '| Campaign | Draws | log Q | Point / population RSE | ESS | Largest weight | CPU s | Hard volume Å³ | Paired-cloud fraction |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def f(value,spec='.3f'):return 'unresolved' if value is None else format(value,spec)
    for name,row in result['campaigns'].items():
        p=row['physical'];cloud=row['concentration']['paired_cloud_fraction_of_observed_variance']
        lines.append(f"| {name} | {p['samples']:,} | {f(p['log_normalizer'])} | {f(p['relative_SE'])} / {f(row['independent_population_relative_SE'])} | {p['ess']:.2f} | {f(p['maximum_point_fraction'],'.2%')} | {row['cpu_seconds']:.1f} | {row['hard_volume_A3']:.4g} | {f(cloud)} |")
    lines+=['','| Campaign | Hard-volume ESS | Largest hard weight | Hard population RSE |',
            '| --- | ---: | ---: | ---: |']
    for name,row in result['campaigns'].items():
        h=row['hard']
        lines.append(f"| {name} | {h['ess']:.2f} | {f(h['maximum_point_fraction'],'.2%')} | {f(row['hard_independent_population_relative_SE'])} |")
    lines+=['','![Comparison](native-guided-comparison.png)']
    if 'selected_pilot_vs_wider_confirmation' in result:
        difference=result['selected_pilot_vs_wider_confirmation']
        if difference['resolved_as_diagnostic']:
            lines+=['',f"The independent wider confirmation / selected-pilot mass ratio is **{f(difference['mass_ratio_second_over_first'])}**. Their linear-scale difference, divided by its observed combined standard error, is **{f(difference['population_standardized_linear_difference'])}** using independent-population errors and **{f(difference['point_standardized_linear_difference'])}** using pointwise errors. These are finite-sample diagnostics, not calibrated significance tests."]
    lines+=['',
        'Population dots and observed delta-method error bars describe the realized independent populations; they cannot bound unseen high-weight regions. Q estimates are unbiased under the frozen proposal law, but their logs, ESS, ratios and these reported error estimates are not convergence certificates.','',
        'The paired-cloud diagnostic retains each original pair and all zero draws. It may exceed one and is not clipped. A large value suggests conditional Poisson noise contributes substantially to the observed variance; it cannot exclude geometric tail problems.','',
        'The JSON includes every population log mass, hard mass, original-q mass histogram, explicit Gaussian/cube/cover component labels, five leading original rows, and observed ESS/CPU and variance-times-CPU diagnostics. These are fixed-budget importance-integration measurements, not MCMC mixing times or established speedups.','',
        'Hard-volume concentration must be assessed separately: the narrow learned proposals can yield many physically weighted native poses while only a few outer-cover hits determine most of the zero-activity volume. These sparse hard estimates are **not substituted into the factorial hard-support contrast C0**.','',
        'All source sample hashes were rechecked against the independent streaming assessments. Physical shape, bath, original native metric, capture region, both neighbors and base cover agree between campaigns. No selected high-weight pose has been re-estimated or replaced.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def self_test():
    cover={'cover_mixture':{'scales':[.1,.2,.4,1.]}}
    assert label({'proposal_component':2},cover)=='cover: scale 0.4'
    guide={'guide':{},'cover_mixture':{'scales':[1.]}}
    assert label({'proposal_family':'guide','proposal_component':None,'guide_branch':'uniform','guide_component':None},guide)=='guide: uniform cube + Haar'
    assert label({'proposal_family':'guide','proposal_component':None,'guide_branch':'learned','guide_component':0},guide)=='guide: Gaussian 0'
    assert label({'proposal_family':'cover','proposal_component':0,'guide_branch':None},guide)=='cover: scale 1'
    assert independent_population_se([None,None],None) is None
    assert independent_population_se([0.],0.) is None
    # Includes a zero-mass population: population means [0,2] have mean1 and SE1.
    assert abs(independent_population_se([None,math.log(2)],0.)-1)<1e-14
    # [1,3,5,7] mean4; sample variance20/3, variance of mean5/3.
    assert abs(independent_population_se([math.log(x) for x in [1,3,5,7]],math.log(4))-math.sqrt(5/3)/4)<1e-14
    a={'physical':{'log_normalizer':math.log(2),'relative_SE':.5},'independent_population_relative_SE':.25}
    b={'physical':{'log_normalizer':math.log(3),'relative_SE':1/3},'independent_population_relative_SE':1/3}
    difference=linear_mean_difference(a,b)
    assert abs(difference['point_standardized_linear_difference']-1/math.sqrt(2))<1e-14
    assert abs(difference['population_standardized_linear_difference']-1/math.sqrt(1.25))<1e-14
    # Full-N cloud diagnostic includes zeros and nonuniform 1/g hard weights.
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);pop=root/'runs/r00';pop.mkdir(parents=True)
        rows=[{'draw':0,'q':2.,'zero':'q','proposal_component':0},
              {'draw':1,'q':.2,'zero':'hard','proposal_component':0}]
        for draw,q,invg,clouds in [(2,.2,2.,[1.,1.]),(3,.5,4.,[2.,4.])]:
            rows.append({'draw':draw,'q':q,'proposal_component':0,
                'log_hard_weight':math.log(invg),'cloud_log_weights':[math.log(x) for x in clouds],
                'log_importance_weight':math.log(invg*sum(clouds)/2)})
        (pop/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        digest=sha(pop/'samples.jsonl')
        manifest={'cloud_replicates':2,'samples':4,'seed':19,'cover':{'volume':10.},'cover_mixture':{'scales':[1.]}}
        (pop/'manifest.json').write_text(json.dumps(manifest))
        (pop/'summary.json').write_text(json.dumps({'complete':True,'samples_sha256':digest}))
        (root/'manifest.json').write_text(json.dumps({'total_unconditional_draws':4,'jobs':[{'output':str(pop),'seed':19}]}))
        data={'samples':4,'regions':{'native':{'log_normalizer':math.log(3.5),'ess':196/148}},
              'hard_regions':{'native':{'log_normalizer':math.log(1.5)}},
              'populations':[{'replicate':'r00','samples':4,'sample_sha256':digest}]}
        result=concentration(root,data)
        assert abs(result['paired_cloud_fraction_of_observed_variance']-16/99)<1e-14
        assert abs(result['paired_cloud_fraction_with_unbiased_total_sample_variance']-12/99)<1e-14
        assert abs(result['q_bin_mass_fractions'][2]-1/7)<1e-14
        assert abs(result['q_bin_hard_mass_fractions'][2]-1/3)<1e-14
        assert result['components']['cover: scale 1']['draws']==4
    print('Component labels, fixed-population uncertainty and full-N paired-cloud/hard-weight controls pass.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/native-cover-mixture-ab-confirmation-8x131072-l64-20260920')
    parser.add_argument('--selected',type=Path,default=ROOT/'runs/native-ab-guided-selected-4x8192-l64-20260920')
    parser.add_argument('--wide',type=Path,default=ROOT/'runs/native-ab-guided-wide-4x8192-l64-20260920')
    parser.add_argument('--confirmation',type=Path,help='Optional independent larger campaign with the same frozen wide guide')
    parser.add_argument('--out',type=Path,default=ROOT/'runs/native-ab-guided-comparison-20260920')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    paths={name:path.resolve() for name,path in [('source_confirmation',args.source),('fresh_selected',args.selected),('fresh_wide',args.wide)]}
    if args.confirmation:paths['wide_confirmation']=args.confirmation.resolve()
    signatures=[physical_signature(path) for path in paths.values()]
    assert all(s==signatures[0] for s in signatures),'Different physical native AB targets/base covers'
    seeds=[job['seed'] for path in paths.values() for job in read(path/'manifest.json')['jobs']]
    assert len(seeds)==len(set(seeds)),'Source and fresh campaign seeds must all be distinct'
    reports={name:summarize(path) for name,path in paths.items()}
    result={'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),
        'same_physical_native_AB_target_verified':True,'campaigns':reports,
        'training_validation_not_pooled':True,'source_code_sha256':sha(Path(__file__)),
        'analysis_environment':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'source_role':'The source confirmation trained both guides; fresh draws validate the frozen guides conditionally. The source remains a separate noisy comparator.',
        'limitation':'Observed concentration and efficiency are pilot diagnostics. No global native probability, converged mass, or MCMC speedup is claimed.'}
    if args.confirmation:
        assert reports['wide_confirmation']['guide']==reports['fresh_wide']['guide'],'Confirmation guide changed'
        result['selected_pilot_vs_wider_confirmation']=linear_mean_difference(reports['fresh_selected'],reports['wide_confirmation'])
        result['wider_pilot_vs_wider_confirmation']=linear_mean_difference(reports['fresh_wide'],reports['wide_confirmation'])
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'provenance').mkdir()
    for source in [Path(__file__),Path(__file__).with_name('analyze_native_factorial_screen.py')]:shutil.copy2(source,out/'provenance'/source.name)
    for name,path in paths.items():shutil.copy2(path/'assessment-streaming.json',out/'provenance'/f'{name}-assessment.json')
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    make_figure(reports,out);write_report(result,out)
    print(json.dumps({name:{k:row[k] for k in ['physical','hard_volume_A3','cpu_seconds','independent_population_relative_SE','efficiency']} for name,row in reports.items()},indent=2))


if __name__=='__main__':main()
