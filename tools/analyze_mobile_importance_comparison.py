#!/usr/bin/env python3
"""Compare hash-bound saved uniform and defensive-importance shell estimates.

This reads existing audits and their saved rows; it does not rerun the density,
geometry or Poisson audits, fit a guide, or launch physical calculations.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
SHELLS=('competitor-shell-5-8','competitor-shell-8-12')
UNIFORM=ROOT/'runs/mobile-competing-outer-weights-20260921'
PREPARATION=ROOT/'runs/mobile-competing-importance-preparation-20260921'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(value,message):
    if not value:raise ValueError(message)


def paired_moments(log_z,log_h):
    """Fixed-N paired ratio covariance, retaining all attempted zero rows."""
    z,h=np.asarray(log_z,float),np.asarray(log_h,float)
    require(z.ndim==h.ndim==1 and z.shape==h.shape and len(z)>=2,'Require paired fixed-N vectors')
    require(not np.isnan(z).any() and not np.isnan(h).any() and not np.isposinf(z).any()
            and not np.isposinf(h).any(),'Invalid log weights')
    require(np.array_equal(np.isfinite(z),np.isfinite(h)),'Different invalid-zero supports')
    n=len(z);nonzero=int(np.isfinite(z).sum())
    if not nonzero:
        return dict(draws=n,nonzero=0,log_Qz=None,log_Q0=None,log_enhancement=None,
                    Qz_relative_SE=None,Q0_relative_SE=None,log_enhancement_SE=None,
                    Qz_ESS=0.,largest_Qz_fraction=None,covariance_relative=None,
                    scope='No nonzero observations: unresolved mass, not a physical zero or upper bound.')
    total_z,total_h=logsumexp(z),logsumexp(h)
    wz,wh=np.exp(z-total_z),np.exp(h-total_h)
    rz,rh=wz-1/n,wh-1/n
    cov=n/(n-1)*np.array([[rz@rz,rz@rh],[rh@rz,rh@rh]])
    lz,lh=float(total_z-math.log(n)),float(total_h-math.log(n))
    return dict(draws=n,nonzero=nonzero,log_Qz=lz,log_Q0=lh,log_enhancement=lz-lh,
                Qz_relative_SE=math.sqrt(float(cov[0,0])),Q0_relative_SE=math.sqrt(float(cov[1,1])),
                log_enhancement_SE=math.sqrt(float(n/(n-1)*np.sum((wz-wh)**2))),
                Qz_ESS=float(1/(wz@wz)),largest_Qz_fraction=float(wz.max()),
                covariance_relative=cov.tolist(),
                scope='Observed delta-method row SE, with paired physical/hard weights and the unconditional denominator. Unseen tails are not bounded.')


def check_estimate(actual,recorded_z,recorded_h):
    require(actual['draws']==recorded_z['draws']==recorded_h['draws'],'Unconditional count changed')
    for got,wanted in ((actual['log_Qz'],recorded_z['logQ']),(actual['log_Q0'],recorded_h['logQ'])):
        require(got is None and wanted is None or got is not None and wanted is not None and abs(got-wanted)<2e-10,
                'Hash-bound rows disagree with the completed audit')


def physical_identity(config_a,config_b,region_a,region_b):
    require({k:v for k,v in config_a.items()if k!='shape'}=={k:v for k,v in config_b.items()if k!='shape'},
            'Physical configuration differs beyond archived shape relocation')
    require(region_a==region_b,'Frozen physical shell differs')


def work_precision(cpu,row,population):
    require(math.isfinite(cpu) and cpu>=0,'Invalid CPU cost')
    return dict(CPU_seconds=cpu,
                CPU_times_Qz_row_relative_variance=None if row['Qz_relative_SE'] is None else cpu*row['Qz_relative_SE']**2,
                CPU_times_Qz_population_relative_variance=None if population['Qz_relative_SE'] is None else cpu*population['Qz_relative_SE']**2,
                scope='Observed planning metric sampler CPU*RSE^2, smaller is better under comparable variance scaling. Guide preparation and audits are excluded. It is not a converged efficiency estimate or a speedup guarantee.')


def validate_uniform_parent(root):
    p,s=read(root/'protocol.json'),read(root/'status.json')
    require(p['schema']=='mobile-competing-outer-regions-campaign-v1','Unexpected uniform controller')
    require(s['complete'] and not s['running'],'Uniform campaign incomplete')
    require(s['protocol_sha256']==sha(root/'protocol.json') and s['driver_sha256']==sha(root/'driver.py'),'Uniform controller changed')
    for name,digest in p['source_sha256'].items():require(sha(root/'provenance'/name)==digest,'Uniform source changed')
    for name,digest in p['input_sha256'].items():require(sha(root/name)==digest,'Uniform frozen input changed')
    bindings={}
    for name in SHELLS:
        spec=next(c for c in p['commands']if c['region']==name)
        terminal=next(c for c in s['jobs']if c['region']==name)
        require(terminal['exit_code']==terminal['audit_exit_code']==0,'Uniform calculation or audit failed')
        bindings[name]=dict(root=str(root/name),assessment_sha256=terminal['assessment_sha256'],
            expected_jobs=spec['expected_jobs'],region_sha256=spec['region_sha256'],
            binary_sha256=p['source_sha256']['latent-region-normalizer'],
            bundle_sha256=p['source_sha256']['source-bundle.json'])
    return bindings,dict(protocol_sha256=sha(root/'protocol.json'),status_sha256=sha(root/'status.json'))


def validate_preparation(root):
    for name,digest in read(root/'freeze.json').items():require(sha(root/name)==digest,'Frozen guide preparation changed')
    plan=read(root/'plan.json')
    require(plan['schema']=='mobile-defensive-importance-guide-preparation-v1','Unexpected guide preparation')
    for name,digest in plan['source_sha256'].items():require(sha(root/'provenance'/name)==digest,'Guide source changed')
    designs=read(root/'provenance/design-analysis.json')
    for name,digest in designs['input_sha256'].items():require(sha(name)==digest,'Guide construction rows changed')
    for g in plan['guides']:
        require(sha(g['path'])==g['guide_sha256'],'Strict guide changed')
        guide=read(g['path']);source=read(root/'provenance'/f"design-{g['region']}.json")
        require(guide['region_sha256']==source['region_sha256']==g['region_sha256'],'Guide target changed')
        require(guide['defensive_uniform_shell_probability']==source['defensive_uniform_shell_probability']==.5,'Guide floor changed')
        require(len(guide['gaussian_components'])==len(source['gaussian_components'])==32,'Guide component count changed')
        require(all(all(a[k]==b[k]for k in ('weight','mean','covariance'))for a,b in zip(guide['gaussian_components'],source['gaussian_components'])),
                'Converted guide was refitted')
    return plan


def validate_guided_parent(root,preparation,preparation_sha256):
    """The preparation controller is frozen separately from the physical runs."""
    p,s=read(root/'protocol.json'),read(root/'status.json')
    require(p['schema']=='mobile-competing-guided-outer-controller-v1','Unexpected guided controller')
    require(s['schema']=='mobile-competing-guided-outer-status-v1' and s['phase']=='complete'
            and s['complete'] and not s.get('running',False),'Guided campaign incomplete')
    require(s['protocol_sha256']==sha(root/'protocol.json'),'Guided protocol changed')
    require(p['guide_preparation_plan_sha256']==preparation_sha256,'Guided preparation binding differs')
    require(p['physical_executable_sha256']==preparation['validated_binary_sha256'] and
            p['source_bundle_sha256']==preparation['validated_source_bundle_sha256'],'Guided executable binding differs')
    require(p['controller_sha256']==sha(root/'provenance/run_mobile_competing_importance_campaign.py') and
            p['freezer_sha256']==sha(root/'provenance/prepare_mobile_competing_importance_campaign.py'),'Guided controller source changed')
    for name,digest in read(root/'freeze.json')['files'].items():
        require(sha(root/name)==digest,'Guided frozen artifact changed')
    require(len(p['campaigns'])==2 and {c['region']for c in p['campaigns']}==set(SHELLS),'Unexpected guided shell pair')
    require(len(s['jobs'])==8 and len({(j['region'],j['id'])for j in s['jobs']})==8,'Guided terminal job set differs')
    bindings={}
    for name in SHELLS:
        entry=next(c for c in p['campaigns']if c['region']==name)
        path=Path(entry['path']).resolve()
        require(path==root/name,'Guided campaign path differs')
        require(sha(path/'manifest.json')==entry['manifest_sha256'],'Guided campaign manifest changed')
        audit=s['audits'][name]
        require(audit.get('returncode',audit.get('exit_code'))==0,'Guided audit failed')
        require(audit['manifest_sha256']==entry['manifest_sha256'],'Audit belongs to another guided manifest')
        manifest=read(path/'manifest.json')
        prepared=next(g for g in preparation['guides']if g['region']==name)
        require(entry['guide_sha256']==prepared['guide_sha256']==manifest['importance_guide_sha256'],'Guided input differs from frozen design')
        require(entry['region_sha256']==prepared['region_sha256']==manifest['region_sha256'],'Guided region differs')
        terminal_jobs={j['id']:j for j in s['jobs']if j['region']==name}
        require(set(terminal_jobs)=={j['id']for j in entry['expected_jobs']},'Guided terminal populations differ')
        for job in entry['expected_jobs']:
            terminal=terminal_jobs[job['id']]
            require(terminal['status']=='complete' and terminal['exit_code']==0 and
                    all(terminal[k]==job[k]for k in ('seed','samples')),'Guided population terminal record differs')
            for filename,key in [('summary.json','summary_sha256'),('manifest.json','manifest_sha256'),('samples.jsonl','samples_sha256')]:
                require(sha(path/'runs'/job['id']/filename)==terminal['output'][key],'Terminally bound population output changed')
        bindings[name]=dict(root=str(path),assessment_sha256=audit['analysis_sha256'],
            expected_jobs=entry['expected_jobs'],region_sha256=entry['region_sha256'],
            guide_sha256=entry['guide_sha256'],binary_sha256=preparation['validated_binary_sha256'],
            bundle_sha256=preparation['validated_source_bundle_sha256'])
    return bindings,dict(protocol_sha256=sha(root/'protocol.json'),status_sha256=sha(root/'status.json'))


def load_arm(binding,arm):
    root=Path(binding['root']);manifest=read(root/'manifest.json')
    require(sha(root/'assessment/analysis.json')==binding['assessment_sha256'],'Saved completed assessment changed')
    assessed=read(root/'assessment/analysis.json')
    for name,digest in manifest['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,'Campaign archive changed')
    require(manifest['region_sha256']==assessed['region_sha256']==binding['region_sha256']==sha(root/'provenance/region.json'),'Physical region hash differs')
    require(manifest['lambda_ratio']==64. and manifest['cloud_replicates']==2,'Cloud law changed')
    require(len(manifest['jobs'])==len(binding['expected_jobs'])==4,'Missing population')
    config=read(root/'provenance/config.json');region=read(root/'provenance/region.json')
    require(sha(root/'provenance/shape.json')==region['shape_sha256'],'Hard shape changed')
    importance=arm=='guided'
    require(manifest['schema']==('importance-latent-region-campaign-v1'if importance else'uniform-latent-region-campaign-v1'),'Arm sampling law differs')
    if importance:
        require(sha(root/'provenance/importance-guide.json')==binding['guide_sha256']==assessed['importance_sampling']['guide_sha256'],'Guide archive or audit changed')
    all_z,all_h,populations,source_hashes=[],[],[],{}
    counts=dict(draws=0,nonzero=0,outside_shell=0,inside_capture_invalid=0,inside_hard_invalid=0,inside_q_invalid=0,
                branch_counts={'uniform-shell':0,'gaussian':0},gaussian_component_counts=[0]*32 if importance else [])
    cpu=0.
    for job,expected in zip(manifest['jobs'],binding['expected_jobs']):
        require(all(job[k]==expected[k]for k in ('id','seed','samples')) and job['samples']==8192,'Population seed or budget differs')
        directory=Path(job['directory']).resolve();require(directory==root/'runs'/job['id'],'Population directory escaped campaign')
        terminal=read(root/f"{job['id']}-status.json")
        require(terminal['returncode']==0,'Population failed')
        summary=read(directory/'summary.json');pop_manifest=read(directory/'manifest.json')
        require(summary['complete'] and summary['manifest']==pop_manifest,'Population summary differs')
        require(pop_manifest['seed']==job['seed'] and pop_manifest['samples']==job['samples'],'Population denominator or seed changed')
        require(pop_manifest['executable_sha256']==binding['binary_sha256'] and pop_manifest['source_bundle_sha256']==binding['bundle_sha256'],'Compiled source or binary changed')
        require(pop_manifest['config_sha256']==sha(root/'provenance/config.json') and pop_manifest['region_sha256']==binding['region_sha256'],'Population physical target changed')
        for name,key in [('input-config.json','config_sha256'),('region.json','region_sha256'),('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
            require(sha(directory/'provenance'/name)==pop_manifest[key],'Population provenance changed')
        if importance:
            require(pop_manifest['importance_guide_sha256']==sha(directory/'provenance/importance-guide.json')==binding['guide_sha256'],'Population guide changed')
        audited=next(p for p in assessed['populations']if p['id']==job['id'])
        require(audited['seed']==job['seed'],'Audited population seed changed')
        sample_path=directory/'samples.jsonl';digest=sha(sample_path)
        require(digest==audited['samples_sha256']==summary['samples_sha256'],'Saved audited rows changed')
        z,h=[],[]
        with sample_path.open()as stream:
            for i,line in enumerate(stream):
                row=json.loads(line);require(row['draw']==i,'Unconditional draw sequence changed')
                z.append(-math.inf if row['log_importance_weight']is None else row['log_importance_weight'])
                h.append(-math.inf if row['log_hard_weight']is None else row['log_hard_weight'])
                counts['draws']+=1
                branch=row['proposal_branch']if importance else'uniform-shell'
                counts['branch_counts'][branch]+=1
                if importance and branch=='gaussian':counts['gaussian_component_counts'][row['proposal_component']]+=1
                shell=row['shell_valid']if importance else True
                if not shell:counts['outside_shell']+=1
                elif not row['capture_valid']:counts['inside_capture_invalid']+=1
                elif not row['hard_valid']:counts['inside_hard_invalid']+=1
                elif not row['region_valid']:counts['inside_q_invalid']+=1
                else:counts['nonzero']+=1
        require(len(z)==job['samples'],'Changed attempted denominator')
        values=paired_moments(z,h);check_estimate(values,audited['estimate'],audited['hard_region'])
        populations.append(dict(id=job['id'],seed=job['seed'],CPU_seconds=summary['sampler_cpu_seconds'],**values))
        cpu+=summary['sampler_cpu_seconds'];all_z.extend(z);all_h.extend(h)
        for filename in ('samples.jsonl','summary.json','manifest.json'):
            source_hashes[str(directory/filename)]=sha(directory/filename)
    row=paired_moments(all_z,all_h);check_estimate(row,assessed['estimate'],assessed['hard_region'])
    pop=paired_moments([-math.inf if p['log_Qz']is None else p['log_Qz']for p in populations],
                       [-math.inf if p['log_Q0']is None else p['log_Q0']for p in populations])
    require(sum(counts[k]for k in ('nonzero','outside_shell','inside_capture_invalid','inside_hard_invalid','inside_q_invalid'))==counts['draws']==row['draws'],
            'Disjoint failure accounting differs from denominator')
    require(counts['nonzero']==row['nonzero'],'Nonzero accounting differs')
    if importance:
        require(counts['outside_shell']==assessed['importance_sampling']['shell_rejected'] and counts['branch_counts']==assessed['importance_sampling']['branch_counts'],
                'Saved proposal accounting differs from audit')
    return dict(arm=arm,campaign=str(root),row_uncertainty=row,population_uncertainty=pop,populations=populations,
                work_precision=work_precision(cpu,row,pop),proposal_accounting=counts,
                paired_cloud_variance=assessed['estimate'].get('paired_noise'),
                audited_weight_concentration=assessed['estimate'],source_sha256=source_hashes,
                manifest_sha256=sha(root/'manifest.json'),assessment_sha256=binding['assessment_sha256'],
                physical_config=config,physical_region=region)


def make_plot(regions,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'uniform':'#3179a8','guided':'#b75c37'}
    fig,axes=plt.subplots(2,3,figsize=(12.4,7.2))
    columns=[('log_Qz','Qz_relative_SE','log physical regional mass Qz'),
             ('log_Q0','Q0_relative_SE','log accessible pose volume Q0'),
             ('log_enhancement','log_enhancement_SE','log depletion enhancement Qz/Q0')]
    for i,region in enumerate(regions):
        for j,(value,se,title)in enumerate(columns):
            ax=axes[i,j]
            for x,arm in enumerate(region['arms']):
                color=colors[arm['arm']]
                for offset,population in zip([-.16,-.055,.055,.16],arm['populations']):
                    if population[value]is not None:
                        ax.errorbar(x+offset,population[value],yerr=population[se],fmt='o',color=color,
                                    markersize=4,alpha=.8,linewidth=.9,capsize=2)
                pooled=arm['row_uncertainty']
                if pooled[value]is not None:
                    ax.errorbar(x,pooled[value],yerr=pooled[se],fmt='D',color='black',markersize=5,capsize=4,linewidth=1.3,zorder=5)
                else:ax.text(x,.5,'unresolved',transform=ax.get_xaxis_transform(),ha='center')
            ax.set_xticks([0,1],['Uniform\n(training data)','Guided\n(frozen proposal)'])
            ax.set_xlim(-.4,1.4);ax.grid(axis='y',alpha=.18)
            if i==0:ax.set_title(title,fontsize=10)
            if j==0:ax.set_ylabel(region['region'].replace('competitor-shell-','Shell ').replace('-','–')+'\nlog weight')
    fig.suptitle('Same finite contact regions; four independent populations per proposal',fontsize=13)
    fig.text(.5,.035,'Colored points: all four population means ± observed row SE. Black diamonds: pooled estimate ± observed row SE.\n'
             'Old uniform rows trained the frozen guide: arms are not pooled or assigned an independence-based difference test.\n'
             'Each error is descriptive; unseen tails and all mass outside these fixed shells remain unresolved.',ha='center',fontsize=8.2)
    fig.tight_layout(rect=[0,.13,1,.95])
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'importance-comparison.{suffix}',dpi=180)
    plt.close(fig)


def analyze(guided,uniform,preparation,out):
    guided,uniform,preparation,out=map(lambda p:Path(p).resolve(),(guided,uniform,preparation,out))
    require(not out.exists(),'Use a fresh immutable comparison output')
    plan=validate_preparation(preparation)
    ub,ujournal=validate_uniform_parent(uniform);gb,gjournal=validate_guided_parent(guided,plan,sha(preparation/'plan.json'))
    regions=[];seeds=[]
    for name in SHELLS:
        old,new=load_arm(ub[name],'uniform'),load_arm(gb[name],'guided')
        physical_identity(old['physical_config'],new['physical_config'],old['physical_region'],new['physical_region'])
        seeds.extend(p['seed']for arm in (old,new)for p in arm['populations'])
        old_metric=old['work_precision']['CPU_times_Qz_row_relative_variance'];new_metric=new['work_precision']['CPU_times_Qz_row_relative_variance']
        regions.append(dict(region=name,arms=[old,new],
            guided_minus_uniform_log_Qz=None if old['row_uncertainty']['log_Qz']is None or new['row_uncertainty']['log_Qz']is None else new['row_uncertainty']['log_Qz']-old['row_uncertainty']['log_Qz'],
            uniform_over_guided_observed_work_precision=None if old_metric is None or new_metric is None or new_metric==0 else old_metric/new_metric,
            comparison_scope='Old uniform draws helped construct the frozen proposal. No independent-arm covariance model or pooled estimate is assumed. New populations are independent conditional on the frozen guide. The work-precision ratio uses unresolved observed moments and is not a validated speedup.'))
    require(len(seeds)==len(set(seeds)),'A seed was reused across populations or arms')
    result=dict(schema='mobile-importance-comparison-v1',complete=True,regions=regions,
        uniform_parent=ujournal,guided_parent=gjournal,preparation_sha256=sha(preparation/'plan.json'),
        audits_rerun=0,physical_jobs_launched=0,models_fitted=0,
        scope='Only unchanged shells 5<rho<=8 and 8<rho<=12 at rd1.5Å,z0.035Å^-3 on the exact fixed scaffold. '
              'No full contact-basin coverage, equilibrium assembly preference, global normalization, or outside-mass bound is established.')
    out.mkdir();shutil.copy2(Path(__file__),out/'analyzer.py');write(out/'analysis.json',result)
    make_plot(regions,out)
    lines=['Same finite-region integrals under uniform and frozen guided proposals.\n',
           '| Shell | Proposal | log Qz | row RSE | population RSE | ESS | maximum row | CPU s | CPU × row RSE² |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in regions:
        for a in r['arms']:
            s,p,w=a['row_uncertainty'],a['population_uncertainty'],a['work_precision']
            if s['log_Qz']is None:lines.append(f"| {r['region']} | {a['arm']} | unresolved | — | — | 0 | — | {w['CPU_seconds']:.2f} | — |")
            else:lines.append(f"| {r['region']} | {a['arm']} | {s['log_Qz']:.6f} | {s['Qz_relative_SE']:.2%} | {p['Qz_relative_SE']:.2%} | {s['Qz_ESS']:.1f} | {s['largest_Qz_fraction']:.2%} | {w['CPU_seconds']:.2f} | {w['CPU_times_Qz_row_relative_variance']:.3f} |")
    lines+=['','| Shell | Proposal | log Q0 | log(Qz/Q0) | paired row SE | nonzero / attempted | outside shell |','|---|---|---:|---:|---:|---:|---:|']
    for r in regions:
        for a in r['arms']:
            s,c=a['row_uncertainty'],a['proposal_accounting']
            numbers='unresolved | unresolved | —'if s['log_Q0']is None else f"{s['log_Q0']:.6f} | {s['log_enhancement']:.6f} | {s['log_enhancement_SE']:.6f}"
            lines.append(f"| {r['region']} | {a['arm']} | {numbers} | {c['nonzero']} / {c['draws']} | {c['outside_shell']} |")
    lines+=['','| Shell | Proposal | observed cloud variance fraction | uniform / Gaussian draws | all-zero populations |',
            '|---|---|---:|---:|---:|']
    for r in regions:
        for a in r['arms']:
            noise=a['paired_cloud_variance'];fraction=None if noise is None else noise.get('paired_cloud_variance_fraction')
            value='unresolved'if fraction is None else f'{fraction:.2%}'
            branches=a['proposal_accounting']['branch_counts'];zeros=sum(p['nonzero']==0 for p in a['populations'])
            lines.append(f"| {r['region']} | {a['arm']} | {value} | {branches['uniform-shell']} / {branches['gaussian']} | {zeros} / 4 |")
    lines+=['',f'![Population comparison]({out}/importance-comparison.png)','',
        'The uniform rows trained the guide; no combined estimate or independent-arm significance test is reported. '
        'CPU × RSE² and its ratio are observed planning diagnostics, not guarantees of efficiency. '
        'Population dispersion is estimated from only four populations. All attempted invalid draws are retained.',
        '',result['scope']]
    (out/'report.md').write_text('\n'.join(lines)+'\n');write(out/'freeze.json',{p.name:sha(p)for p in out.iterdir()if p.is_file()})
    print('\n'.join(lines[:18]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guided',type=Path,required=True)
    parser.add_argument('--uniform',type=Path,default=UNIFORM)
    parser.add_argument('--preparation',type=Path,default=PREPARATION)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();analyze(args.guided,args.uniform,args.preparation,args.out)
