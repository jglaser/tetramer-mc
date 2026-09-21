#!/usr/bin/env python3
"""Compare a frozen native repeat, its pilot, and identical targeted shells."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
import argparse
import math
from pathlib import Path

from analyze_expanded_contact_atlas import nested_prefix_comparison
from analyze_latent_region import original_q_window, validate_manifest_q_window
from analyze_native_tail_reference import KEYS, WINDOW, validate_terminal_status
from analyze_shoulder_contact_atlas import labeled_comparison
from audit_shoulder_mis_independently import near
from native_repeat_moments import pooled_populations
from prepare_cayley_rms_cover import read, write, sha, require
from run_shoulder_mis_campaign import local_dependencies

PHYSICAL = ('metadata','fixed_poses','capture_center','capture_radius','depletant_radius','reservoir_density')
SCOPE = ('Independent original-weight estimates for the identical original native mask and AB environment. '
         'Observed errors are diagnostics, not nominal significance tests or unseen-mass bounds. No pooling, '
         'tail substitution, renormalization by success count, or MCMC mixing claim.')


def check_native(path):
    result=read(path);require(result['complete'] and result['original_q_window']==WINDOW,'Wrong/incomplete native audit')
    for name,digest in result['archived_sha256'].items():require(sha(path.parent/'provenance'/name)==digest,'Audited archive changed')
    for name,digest in result['input_sha256'].items():require(sha(name)==digest,'Audited input changed')
    return result


def as_standard(old):
    q=old['log_normalizer']; rse=old['relative_SE']
    return dict(draws=old['samples'],nonzero=old['nonzero'],logQ=q,row_RSE=rse,
        independent_population_RSE=old['independent_population_relative_SE'],weight_ESS=old['ess'],
        maximum_fraction=old['maximum_point_fraction'],
        log_variance_of_mean=2*(q+math.log(rse)) if q is not None and rse else None)


def checked_uniform(campaign, edges, cfg, old_model, sources):
    """Verify prior audited bytes/runtime identity without repeating its row audit."""
    root=Path(campaign['root']); master=read(root/'manifest.json'); region=read(root/'provenance/region.json')
    actual=read(root/'provenance/config.json'); assessment=root/'assessment/analysis.json'
    require(all(actual[k]==cfg[k] for k in PHYSICAL),'Targeted shell physical target differs')
    require(region['gaussian_chart']==old_model and original_q_window(region)==WINDOW,'Targeted shell chart or q differs')
    require((region['minimum_mahalanobis_radius'],region['mahalanobis_radius'])==edges,'Different old-chart shell')
    require(sha(root/'provenance/shape.json')==old_model['shape_sha256'],'Targeted shape differs')
    require(sha(root/'provenance/region.json')==campaign['region_sha256']==master['region_sha256'],'Shell changed')
    for name,digest in master['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,'Targeted archive changed')
    require(sha(assessment)==campaign['assessment_sha256'],'Targeted audit changed')
    audited=read(assessment);n=0;seeds=[]
    for job in master['jobs']:
        path=Path(job['directory']);summary=read(path/'summary.json');run=summary['manifest']
        recorded=next(p for p in campaign['input_hashes'] if p['id']==job['id'])
        require(summary['complete'] and summary['samples']==job['samples']==run['samples'],'Targeted original N differs')
        require(run['seed']==job['seed'] and run['region_sha256']==campaign['region_sha256'],'Targeted runtime differs')
        require(run['activity']==.035 and run['lambda_ratio']==64 and run['cloud_replicates']==2,'Targeted bath differs')
        validate_manifest_q_window(run,region,WINDOW)
        status=root/f"{job['id']}-status.json";validate_terminal_status(read(status),job)
        for file,key in [('samples.jsonl','samples_sha256'),('summary.json','summary_sha256')]:
            require(sha(path/file)==recorded[key],'Previously audited targeted data changed')
            sources[str(path/file)]=recorded[key]
        require(summary['samples_sha256']==recorded['samples_sha256'],'Targeted sample seal differs')
        sources[str(status)]=sha(status);n+=job['samples'];seeds.append(job['seed'])
    require(n==campaign['estimate']['samples']==audited['independently_reconstructed_poses'],'Targeted audit incomplete')
    for p in [root/'manifest.json',assessment,root/'provenance/config.json',root/'provenance/region.json']:
        sources[str(p)]=sha(p)
    return dict(root=str(root),edges=list(edges),physical=as_standard(campaign['estimate']),seeds=seeds,
                CPU_seconds=campaign['cpu_seconds'])


def compare(plan_path,out):
    require(not out.exists(),'Fresh comparison output required')
    executing=local_dependencies([Path(__file__)])
    execution_hashes={str(p):sha(p) for p in executing.values()}
    plan=read(plan_path);require(plan['schema']=='native-complete-cover-repeat-comparison-v1','Wrong comparison plan')
    require(plan['original_q_window']==WINDOW and plan['masks']==list(KEYS),'Comparison target changed')
    inputs={name:Path(row['path']) for name,row in plan['inputs'].items()}
    for name,path in inputs.items():require(sha(path)==plan['inputs'][name]['sha256'],'Frozen comparison input changed')
    repeat_path=Path(plan['repeat_analysis']);pilot=check_native(inputs['pilot']);repeat=check_native(repeat_path)
    prep=inputs['repeat_protocol'].parent;protocol=read(inputs['repeat_protocol'])
    require(Path(protocol['proposed_campaign']['output'])==Path(repeat['campaign']),'Repeat campaign differs')
    require(plan['prefix_population_counts']==[p['populations'] for p in protocol['analysis_prefixes']],'Unfrozen prefixes')
    cfg=read(prep/'config.json');old_model=read(prep/'old-model.json')
    prior_cfg=read(inputs['pilot'].parent/'provenance/config.json')
    require(all(cfg[k]==prior_cfg[k] for k in PHYSICAL),'Pilot/repeat physical target differs')
    for filename in ('model.json','old-model.json','region.json'):
        require(read(prep/filename)==read(inputs['pilot'].parent/'provenance'/filename),'Pilot/repeat law changed')
    n=protocol['proposed_campaign']['samples_per_population'];pops=repeat['populations']
    require([p['seed'] for p in pops]==protocol['proposed_campaign']['seeds'],'Repeat order/streams changed')
    require(all(p['samples']==n for p in pops),'Repeat fixed population sizes differ')
    sources={str(p):sha(p) for p in [plan_path,repeat_path,*inputs.values()]}
    original=read(inputs['original_shell_comparison']);confirmation=read(inputs['five_eight_confirmation'])
    require(original['original_model_sha256']==sha(prep/'old-model.json'),'Targeted old chart differs')
    require(confirmation['original_comparison_sha256']==sha(inputs['original_shell_comparison']),'Confirmation lineage differs')
    for path,key,file in [(inputs['original_shell_comparison'],'script_sha256','compare_tail_references.py'),
                         (inputs['original_shell_comparison'],'classifier_sha256','original_classifier.py'),
                         (inputs['five_eight_confirmation'],'script_sha256','compare_five_eight_confirmation.py'),
                         (inputs['five_eight_confirmation'],'comparison_helper_sha256','compare_tail_references.py')]:
        require(sha(path.parent/'provenance'/file)==read(path)[key],'Targeted comparison source changed')
    targeted={}
    for key,edges,campaign in [
        ('old_4_lt_r_le_5',(4.,5.),original['uniform_shell_campaigns']['4-5']),
        ('old_5_lt_r_le_8',(5.,8.),confirmation['confirmation']),
        ('old_8_lt_r_le_12',(8.,12.),original['uniform_shell_campaigns']['8-12'])]:
        targeted[key]=checked_uniform(campaign,edges,cfg,old_model,sources)
    seed_groups=[[p['seed'] for p in run['populations']] for run in [pilot,repeat]]+[x['seeds'] for x in targeted.values()]
    seeds=[s for group in seed_groups for s in group];require(len(seeds)==len(set(seeds)),'Comparison streams overlap')
    prefixes=[]
    for count in plan['prefix_population_counts']:
        r=pooled_populations(pops[:count]);prefixes.append(dict(populations=count,**r))
    reconstructed=prefixes[-1]
    for kind in ('physical','hard'):
        for key in KEYS:
            a,b=reconstructed[kind][key],repeat[kind][key]
            require(a['draws']==b['draws'] and a['nonzero']==b['nonzero'],'Moment reconstruction changed full N')
            for field in ('logQ','row_RSE','weight_ESS','maximum_fraction','log_variance_of_mean','paired_cloud_variance_fraction'):
                if b[field] is None:require(a[field] is None,'Moment reconstruction changed empty statistic')
                else:near(a[field],b[field],2e-8)
    contrasts={kind:{key:labeled_comparison(repeat[kind][key],pilot[kind][key],'repeat','pilot',SCOPE)
                     for key in KEYS} for kind in ('physical','hard')}
    targeted_contrasts={label:{key:labeled_comparison(run['physical'][key],target['physical'],label,'targeted_uniform',SCOPE)
        for key,target in targeted.items()} for label,run in [('pilot',pilot),('repeat',repeat)]}
    prefix_contrasts=[dict(earlier_populations=a['populations'],later_populations=b['populations'],
        **{kind:{key:nested_prefix_comparison(a[kind][key],b[kind][key]) for key in KEYS} for kind in ('physical','hard')})
        for i,a in enumerate(prefixes) for b in prefixes[i+1:]]
    # No completed row audit is replayed; this layer uses checked raw moments.
    archive=out/'provenance';archive.mkdir(parents=True)
    files={'plan.json':plan_path,'pilot-analysis.json':inputs['pilot'],'repeat-analysis.json':repeat_path,
        'repeat-protocol.json':inputs['repeat_protocol'],'original-shell-comparison.json':inputs['original_shell_comparison'],
        'five-eight-confirmation.json':inputs['five_eight_confirmation'],**executing}
    for name,path in files.items():(archive/name).write_bytes(path.read_bytes())
    sources.update({str(p):sha(p) for p in files.values()})
    for path,digest in {**sources,**execution_hashes}.items():require(sha(path)==digest,'Input/source changed during comparison')
    result=dict(complete=True,original_q_window=WINDOW,masks=list(KEYS),pilot=pilot,repeat=repeat,
        independent_repeat_comparisons=contrasts,targeted_uniform=targeted,targeted_comparisons=targeted_contrasts,
        repeat_prefixes=prefixes,repeat_prefix_comparisons=prefix_contrasts,
        full_original_moments_reconstructed=True,no_new_draws_or_data_audit=True,no_pooling=True,
        source_sha256=sources,executing_source_sha256=execution_hashes,
        archived_sha256={name:sha(archive/name) for name in files},scope=SCOPE)
    write(out/'analysis.json',result)
    print(dict(complete=True,output=str(out/'analysis.json'),tail=repeat['physical']['old_r_gt_12']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    compare(args.plan.resolve(),args.out.resolve())
