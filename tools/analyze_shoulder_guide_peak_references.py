#!/usr/bin/env python3
"""Audit frozen three-neighborhood shoulder calibration without stitching tails.

The four new proposal laws are uniform local balls. Their original densities
and full unconditional denominators survive every geometric exclusion.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
import hashlib
import heapq
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from analyze_far_peak_reference import add_population_errors,sum_independent_estimates
from analyze_latent_region import original_q_window
from analyze_peak_neighborhood import geometry_model_audit
from analyze_shoulder_peak_reference import (WINDOW,SHOULDER_WINDOW,compare_physical,
    geometry_batch,original_weights,check_equal_mass,verify_reference_manifest)
from audit_shoulder_mis_independently import near
from compare_far_local_history import independent_comparison,validate_runtime_metric
from compare_intermediate_local_reference import read_batches
from prepare_cayley_rms_cover import read,write,sha,require
from prepare_native_confirmation_atlas import AtomUnionAudit
from run_shoulder_mis_campaign import local_dependencies
from shoulder_union_moments import CENTERS,MASKS,UnionMoments

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_PREPARATION=ROOT/'runs/ab-shoulder-guide-peak-reference-preparation-20260921'
RADII=(.25,.5)


def same_row_exclusion(before,after,covariance):
    """Before/after comparison retaining covariance of the same weighted rows."""
    require(before['draws']==after['draws'],'Compared masks changed original N')
    b,a=before['logQ'],after['logQ']
    result=dict(before=before,assigned=after,same_row_covariance=covariance,
        assigned_fraction=None,removed_logQ=None,removed_log_variance_of_mean=None,
        removed_row_SE_scaled=None,scale_logQ=None,assigned_fraction_delta_SE=None,
        scope='Same-row comparison within this source support, with covariance. Removed mass is a geometric exclusion, not a rejected physical transition or sampling-rate claim.')
    if b is None:return result
    require(a is None or a<=b+2e-9,'Assigned region exceeds original ball')
    fraction=0. if a is None else min(1.,math.exp(a-b));result['assigned_fraction']=fraction
    result['removed_logQ']=None if fraction==1. else b+math.log1p(-fraction)
    scale=b; vb=0. if before['log_variance_of_mean'] is None else math.exp(before['log_variance_of_mean']-2*scale)
    va=0. if after['log_variance_of_mean'] is None else math.exp(after['log_variance_of_mean']-2*scale)
    cov=0. if covariance['sign']==0 else covariance['sign']*math.exp(covariance['log_absolute_covariance']-2*scale)
    removed_variance=vb+va-2*cov
    fraction_variance=va+fraction*fraction*vb-2*fraction*cov
    tolerance=2e-10*max(vb,va,abs(cov),1e-300)
    require(removed_variance>=-tolerance and fraction_variance>=-tolerance,'Same-row covariance gives negative variance')
    removed_variance=max(0.,removed_variance);fraction_variance=max(0.,fraction_variance)
    result.update(removed_log_variance_of_mean=2*scale+math.log(removed_variance) if removed_variance else None,
        removed_row_SE_scaled=math.sqrt(removed_variance),scale_logQ=scale,assigned_fraction_delta_SE=math.sqrt(fraction_variance))
    return result


def finish(aggregate,populations):
    result=add_population_errors(aggregate.report(),populations,aggregate.keys)
    for kind in ('physical','hard'):
        full=result[kind]['full']['logQ']
        for row in result[kind].values():row['fraction_of_observed_supported_inner_mass']=math.exp(row['logQ']-full) if row['logQ'] is not None else None
    return result


def assert_terminal(root,master):
    """No new density auditor starts before all original populations finish."""
    for job in master['jobs']:
        status=read(root/f"{job['id']}-status.json")
        summary=read(Path(job['directory'])/'summary.json')
        require(status['returncode']==0 and status['id']==job['id'],'Physical runner not terminal success')
        require(summary['complete'] and summary['samples']==job['samples'],'Physical population incomplete')


def ensure_original_audits(requests,out,allow_run):
    state=dict(phase='original_latent_audits',processes=[]);live=[];path=out/'runner-state.json'
    # Preflight all requests before launching any child.
    for declared,root,master in requests:
        assert_terminal(root,master)
        require((root/'assessment/analysis.json').exists() or allow_run,'Original audit missing; await root authorization')
    try:
        for declared,root,_ in requests:
            target=root/'assessment/analysis.json';label=f"{declared['name']}-r{str(declared['radius_A']).replace('.','p')}"
            if target.exists():state['processes'].append(dict(label=label,reused=True,path=str(target),sha256=sha(target)));continue
            command=[sys.executable,str(root/'provenance/analyze_latent_region.py'),'--root',str(root)]
            logpath=out/f'original-audit-{label}.log';log=logpath.open('x')
            try:process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            except BaseException:log.close();raise
            record=dict(label=label,pid=process.pid,command=command,log=str(logpath),output=str(target),terminal=False)
            live.append((process,log,record));state['processes'].append(record);write(path,state)
            print(f"Original latent auditor {label}: PID{process.pid}",flush=True)
    finally:
        # A launch failure also drains already-created children; no valid audit
        # is abandoned, killed, or restarted by another child's failure.
        for process,log,record in live:
            record.update(exit_code=process.wait(),terminal=True);log.close();write(path,state)
    failures=[r for r in state['processes'] if r.get('exit_code',0)]
    state['phase']='original_audits_failed' if failures else 'original_audits_complete';write(path,state)
    require(not failures,'Original auditor failure; diagnose without replaying successful audits')
    return state


def read_history(declared,cfg,shape_sha):
    """Load frozen prior audits only; data will be remasked once in this driver."""
    root=Path(declared['root']);path=Path(declared['comparison_path']);master=read(root/'manifest.json')
    require(sha(path)==declared['comparison_sha256'] and sha(root/'manifest.json')==declared['manifest_sha256'],'Historical comparison/master changed')
    old_cfg=read(root/'provenance/config.json');compare_physical(old_cfg,cfg)
    require(sha(root/'provenance/shape.json')==shape_sha,'Historical hard shape changed')
    for name,digest in master['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,'Historical archive changed')
    comparison=read(path);kind='latent' if declared['name']=='direct-inner' else 'guided'
    if kind=='latent':
        require(comparison['complete'],'Historical direct comparison incomplete');record=comparison['focused_band']
        auditpath=root/'assessment/analysis.json';region=read(root/'provenance/region.json')
        require(original_q_window(region)==declared['original_q_window']==record['integration_window']==WINDOW,'Historical strict inner window changed')
        for name,digest in record['input_sha256'].items():require(sha(name)==digest,'Historical direct audit changed')
        require(region['physical_fixed_neighbors']==cfg['fixed_poses'] and region['fixed_neighbor']==cfg['fixed_poses'][0],'Historical direct AB/frame changed')
        original=read(auditpath);require(original['independently_reconstructed_poses']==declared['original_N'],'Historical direct all-row audit incomplete')
        expected={kind:record[kind]['logQ'] for kind in ('physical','hard')}
    else:
        record=comparison['campaigns'][declared['comparison_key']];auditpath=root/'assessment-streaming.json';original=read(auditpath)
        require(comparison['q_window']==declared['original_q_window']==original['q_window']==SHOULDER_WINDOW,'Historical broader shoulder law changed')
        require(sha(auditpath)==record['assessment_sha256'] and sha(root/'provenance/config.json')==record['config_sha256'],'Historical guided audit/config changed')
        require(original['complete'] and original['all_rows_and_hashes_validated'] and original['independent_complete_cover_validated'],'Historical guide density audit incomplete')
        require(original['analysis_script_sha256']==master['archive_sha256']['analyze_native_region_reference.py'],'Historical original auditor changed')
        require(original['samples']==original['independent_q_and_density_rows']==declared['original_N'],'Historical guide full N changed')
        expected=dict(physical=record['bands']['0']['logQ'],hard=record['hard_bands']['0']['logQ'])
    require(record['root']==str(root),'Historical root identity differs')
    sources={str(p):sha(p) for p in (path,root/'manifest.json',root/'provenance/config.json',auditpath)}
    return dict(root=root,master=master,kind=kind,original=original,record=record,expected=expected,
        sources=sources,audit_path=auditpath,declared=declared)


def validate_population_header(summary,run,n_expected,seed,kind):
    require(summary['complete'] and summary['samples']==n_expected and run['seed']==seed,'Original population summary/count/seed changed')
    # The older guided summary has no embedded runtime manifest. It is bound
    # separately by its prior audit's summary SHA and the run manifest below.
    if kind=='latent':require(summary['manifest']==run,'Original latent embedded manifest changed')


def remask(context,cfg,models,geometries,owner=None,radius=None):
    root=context['root'];master=context['master'];kind=context['kind'];original=context['original']
    aggregate=UnionMoments();populations=[];sources=dict(context['sources']);top=[];errors={name:{} for name in CENTERS};seen=set()
    atom=AtomUnionAudit(read(root/'provenance/shape.json'),cfg['fixed_poses'])
    for pi,job in enumerate(master['jobs']):
        directory=Path(job['directory'] if kind=='latent' else job['output']);sample=directory/'samples.jsonl'
        summary=read(directory/'summary.json');run=read(directory/'manifest.json');n_expected=job['samples'] if kind=='latent' else run['samples']
        validate_population_header(summary,run,n_expected,job['seed'],kind)
        require(run['activity']==cfg['reservoir_density'],'Original runtime activity changed')
        require(run['lambda_ratio']==64 and run['lambda']==64*cfg['reservoir_density'] and run['cloud_replicates']==2,'Original runtime Poisson law changed')
        require(job['seed'] not in seen,'Duplicate source population seed');seen.add(job['seed'])
        if kind=='latent':
            prior=next(p for p in original['populations'] if p['id']==job['id']);digest_expected=prior['samples_sha256']
            require(run['lambda_ratio']==master['lambda_ratio']==64 and run['cloud_replicates']==master['cloud_replicates']==2,'Original latent cloud law changed')
            require(run['config_sha256']==sha(root/'provenance/config.json') and run['shape_sha256']==models['direct']['shape_sha256'],'Original latent configuration/shape differs')
            require(run['executable_sha256']==master['archive_sha256']['latent-region-normalizer'],'Original latent executable differs')
        else:
            require(run['depletant_radius']==cfg['depletant_radius'],'Original guided runtime depletant radius changed')
            prior=next(p for p in original['populations'] if p['replicate']==directory.name);digest_expected=prior['sample_sha256']
            require(sha(directory/'summary.json')==prior['summary_sha256'],'Guided summary changed after full audit')
            require(run['q_window']==original['q_window']==SHOULDER_WINDOW,'Guided source target changed')
            validate_runtime_metric(run['metric'],cfg['metadata'])
            require(run['guide']==original['guide'] and run['cover_mixture']==original['cover_mixture'],'Original full hybrid density changed')
            require(run['shape_sha256']==models['direct']['shape_sha256'] and run['config_sha256']==master['config_sha256'],'Original guided physical input differs')
            require(run['executable_sha256']==master['binary_sha256'],'Original guided executable differs')
        if context.get('record'):require(context['record']['sample_sha256'][str(sample)]==digest_expected,'Previously audited source rows changed')
        local=UnionMoments();digest=hashlib.sha256();count=0
        for lines,rows in read_batches(sample):
            for line in lines:digest.update(line)
            radii=[]
            for name in CENTERS:
                current_r,current_j,current=geometry_batch(rows,geometries[name],models[name],cfg)
                if name==owner:
                    require(np.all(current_r<=radius*(1+1e-10)),'New draw outside frozen local ball')
                    current['latent_radius']=near(current_r,[r['latent_radius'] for r in rows]);current['jacobian']=near(current_j,[r['log_physical_jacobian'] for r in rows])
                for key,value in current.items():errors[name][key]=max(errors[name].get(key,0.),value)
                radii.append(current_r)
            for i,row in enumerate(rows):
                require(row['draw']==count,'Original row ordering changed');count+=1
                weights=original_weights(row,kind);triple=tuple(float(r[i]) for r in radii)
                local.add(triple,*weights)
                if weights[0] is not None:
                    record=dict(population=directory.name,seed=job['seed'],draw=row['draw'],pose=row['pose'],q=row['q'],
                        geometric_radii_A=dict(zip(CENTERS,triple)),original_log_importance_weight=weights[0],original_row=row,source_samples_path=str(sample))
                    entry=(weights[0],-pi,-row['draw'],record)
                    if len(top)<8:heapq.heappush(top,entry)
                    elif entry[:3]>top[0][:3]:heapq.heapreplace(top,entry)
        require(count==n_expected,'Original unconditional N changed')
        require(digest.hexdigest()==digest_expected==summary['samples_sha256'],'Original rows changed since full audit')
        sources[str(sample)]=digest.hexdigest()
        for file in ('manifest.json','summary.json'):sources[str(directory/file)]=sha(directory/file)
        aggregate.merge(local);populations.append(dict(id=directory.name,seed=job['seed'],samples=count,**local.report()))
    result=finish(aggregate,populations)
    for kind in ('physical','hard'):check_equal_mass(result[kind]['full']['logQ'],context['expected'][kind])
    extrema=[]
    for _,_,_,row in sorted(top,key=lambda t:t[:3],reverse=True):
        gaps=atom.gaps(row['pose']);require(min(gaps)>=0,'Positive extreme clashes independent atomic AB')
        row['minimum_AB_atomic_gaps_A']=gaps;row['source_samples_sha256']=sources[row['source_samples_path']];extrema.append(row)
    return dict(root=str(root),name=context['name'],owner=owner,radius_A=radius,**result,populations=populations,
        original_unconditional_draws=sum(p['samples'] for p in populations),top8_atomic_checks=extrema,input_sha256=sources,
        geometry_reconstruction_errors=errors,original_audit_path=str(context['audit_path']),original_audit_sha256=sha(context['audit_path']),
        CPU_seconds=context['CPU_seconds'],support_scope='Original local ball intersected with strict inner window; unsupported regions are not estimable' if owner else 'Entire original strict inner window with original full source N, including broader shoulder zeros')


def combine_assigned(campaigns,direct_reference):
    require({(c['owner'],c['radius_A']) for c in campaigns}=={(n,r) for n in ('mixture','geometry') for r in RADII} and len(campaigns)==4,'Need all four frozen fresh campaigns')
    seeds=[p['seed'] for c in campaigns for p in c['populations']]
    direct_seeds=[p['seed'] for c in direct_reference['campaigns'] for p in c['populations']]
    require(len(seeds)==len(set(seeds)) and not set(seeds)&set(direct_seeds),'Assigned-region source streams overlap')
    regions={};union={};pieces={}
    for kind in ('physical','hard'):
        regions[kind]={'direct':copy.deepcopy(direct_reference['independent_shell_sum'][kind]['ball0p5'])}
        for name in ('mixture','geometry'):
            rows=[]
            for radius,suffix in ((.25,'assigned_ball0p25'),(.5,'assigned_radial_1')):
                campaign=next(c for c in campaigns if c['owner']==name and c['radius_A']==radius)
                rows.append(campaign[kind][f'{name}_{suffix}'])
            regions[kind][name]=sum_independent_estimates(rows)
        # These three assigned pieces use independent streams, including the
        # already-frozen two-stream direct estimate; no historical term enters.
        union[kind]=sum_independent_estimates([dict(row,draws=row['total_unconditional_draws']) for row in regions[kind].values()])
        union[kind]['unresolved_source_shells']=sum(row['unresolved_zero_pieces'] for row in regions[kind].values())
    for name in CENTERS:
        owner_campaigns=direct_reference['campaigns'] if name=='direct' else [c for c in campaigns if c['owner']==name]
        pieces[name]=[dict(root=c['root'],radius_A=c['radius_A'],selected_mask=('radial_0' if c['radius_A']==.25 else 'radial_1') if name=='direct' else
            f"{name}_assigned_{'ball0p25' if c['radius_A']==.25 else 'radial_1'}",original_population_samples=[p['samples'] for p in c['populations']],
            source_seeds=[p['seed'] for p in c['populations']]) for c in owner_campaigns]
    return dict(regions=regions,union=union,source_pieces=pieces,
        allocation='Direct first. Mixture excludes direct. Geometry excludes both. Each center uses independent [0,.25] and(.25,.5] source shells; original N and densities remain intact.',
        scope='Finite three-ball union only. Historical outside-union is excluded from this sum. No overlapping whole-ball addition, historical/fresh stitching, or full-window normalizer claim.')


def matching_comparisons(histories,campaigns,combined):
    result={};exclusions={}
    for campaign in campaigns:
        name=campaign['owner'];radius=campaign['radius_A'];label=f'{name}-R{radius}'
        suffixes=('ball0p25',) if radius==.25 else ('ball0p25','radial_1','ball0p5')
        exclusions[label]={kind:{} for kind in ('physical','hard')}
        for kind in ('physical','hard'):
            for suffix in suffixes:
                before=f'{name}_{suffix}';assigned=f'{name}_assigned_{suffix}'
                exclusions[label][kind][suffix]=same_row_exclusion(campaign[kind][before],campaign[kind][assigned],campaign['same_row_covariances'][kind][before][assigned])
        result[label]={}
        for history in histories:
            result[label][history['name']]={kind:{mask:independent_comparison(history[kind][mask],campaign[kind][mask])
                for suffix in suffixes for mask in (f'{name}_{suffix}',f'{name}_assigned_{suffix}')} for kind in ('physical','hard')}
    assigned={}
    for history in histories:
        assigned[history['name']]={kind:{name:independent_comparison(history[kind][f'{name}_assigned_ball0p5'],combined['regions'][kind][name]) for name in CENTERS} for kind in ('physical','hard')}
        assigned[history['name']]['union']={kind:independent_comparison(history[kind]['union'],combined['union'][kind]) for kind in ('physical','hard')}
    # The two local proposals provide independent observations of the same
    # radius.25 piece. Whole-radius.5 and its two-source shell sum share rows;
    # that dependent contrast is explicitly not constructed here.
    nested={}
    for name in ('mixture','geometry'):
        small=next(c for c in campaigns if c['owner']==name and c['radius_A']==.25)
        large=next(c for c in campaigns if c['owner']==name and c['radius_A']==.5)
        nested[name]={kind:{mask:independent_comparison(small[kind][mask],large[kind][mask]) for mask in (f'{name}_ball0p25',f'{name}_assigned_ball0p25')} for kind in ('physical','hard')}
        for kind in nested[name]:
            for entry in nested[name][kind].values():entry['scope']='Two independent radius proposals measuring the identical frozen radius.25 region. Observed errors do not bound unseen tails.'
    return dict(matching_region_comparisons=result,assigned_region_comparisons=assigned,
        priority_exclusion_comparisons=exclusions,fresh_nested_region_controls=nested,
        dependent_comparison_note='Each center assigned two-shell sum shares its outer-shell rows with that center whole-radius.5 campaign; no independent-error contrast between these is made. Source IDs and seeds remain explicit.')


def validate_plan(protocol):
    require(protocol['original_q_window']==WINDOW,'Strict inner target changed')
    require(len(protocol['campaigns'])==4 and {(c['name'],c['radius_A']) for c in protocol['campaigns']}=={(n,r) for n in ('mixture','geometry') for r in RADII},'Four frozen centers/radii changed')
    require(all(c['populations']==16 and c['samples_per_population']==16384 and c['workers']==8 for c in protocol['campaigns']),'Frozen draw or worker allocation changed')
    plan=protocol['analysis_plan']
    require(plan['center_order']==list(CENTERS) and plan['closed_ball_radius_A']==.5,'Priority order or boundary changed')
    require(plan['memberships']==['rho_direct<=.5','rho_mixture<=.5 AND rho_direct>.5','rho_geometry<=.5 AND rho_direct>.5 AND rho_mixture>.5'],'Priority exclusions changed')
    expected=[dict(name=name,assigned_region_index=i+1,shell_sources=[
        dict(campaign_radius_A=.25,own_shell=[0.,.25],lower_inclusive=True,upper_inclusive=True),
        dict(campaign_radius_A=.5,own_shell=[.25,.5],lower_inclusive=False,upper_inclusive=True)]) for i,name in enumerate(('mixture','geometry'))]
    require(plan['new_center_sources']==expected,'Fixed independent source-shell allocation changed')
    require(plan['outside_union']=='rho_direct>.5 AND rho_mixture>.5 AND rho_geometry>.5 within original strict1<q<1.1, capture and AB hard target','Outside-union complement changed')
    require(plan['direct_source']['field']=='independent_shell_sum' and plan['direct_source']['mask']=='ball0p5','Direct source estimator changed')


def analyze(preparation,out,run_audits=False):
    prep=Path(preparation).resolve();out=Path(out).resolve();require(not out.exists(),'Use a fresh audit output directory')
    protocol=read(prep/'protocol.json');freeze=read(prep/'freeze.json');prepared=read(prep/'report.json')
    require(prepared['complete'] and prepared['prepared_only'] and prepared['protocol_sha256']==sha(prep/'protocol.json') and prepared['freeze_sha256']==sha(prep/'freeze.json'),'Preparation incomplete or changed')
    for name,digest in freeze.items():require(sha(prep/name)==digest,'Frozen preparation changed')
    for name,digest in protocol['archived_sha256'].items():require(sha(prep/'provenance'/name)==digest,'Preparation archive changed')
    for name,digest in protocol['input_sha256'].items():require(sha(name)==digest,'Original frozen input changed')
    validate_plan(protocol)
    cfg=read(prep/'config.json');compare_physical(cfg,protocol['physical'])
    require(sha(prep/'config.json')==protocol['config_sha256'] and sha(prep/'provenance/shape.json')==protocol['shape_sha256'],'Physical config/shape changed')
    require(sha(prep/'commands.json')==protocol['commands_sha256'],'Frozen commands changed')
    direct_source=protocol['analysis_plan']['direct_source'];direct_path=Path(direct_source['analysis_path']);direct=read(direct_path)
    require(sha(direct_path)==direct_source['analysis_sha256']==sha(prep/'provenance/direct-analysis.json'),'Prior direct calibration changed')
    require(direct['complete'] and direct['original_q_window']==WINDOW and direct['independent_shell_sum']['pieces']==direct_source['pieces'],'Prior direct target/shell allocation changed')
    for name,digest in direct['archived_sha256'].items():require(sha(direct_path.parent/'provenance'/name)==digest,'Prior direct executed source changed')
    for name,digest in direct['source_sha256'].items():require(sha(name)==digest,'Prior direct physical source changed')
    direct_prep=Path(protocol['direct_reference_preparation']);direct_cfg=read(direct_prep/'config.json');compare_physical(direct_cfg,cfg)
    require(sha(direct_prep/'model.json')==sha(prep/'provenance/direct-model.json'),'Prior direct geometry chart changed')
    require(sha(direct_prep/'selected-pose.json')==sha(prep/'provenance/direct-selected-pose.json'),'Prior direct selected pose changed')
    models={'direct':read(prep/'provenance/direct-model.json')};peaks={'direct':read(prep/'provenance/direct-selected-pose.json')['pose']};selection={}
    discovery=read(prep/'provenance/discovery.json');guided=read(prep/'provenance/guided-comparison.json')
    require(sha(prep/'provenance/discovery.json')==protocol['selection_source']['sha256']==sha(protocol['selection_source']['path']),'Discovery selection changed')
    for center,expected in zip(protocol['centers'],(('mixture','r03',99604037,30939),('geometry','r01',99612019,33280))):
        name,population,seed,draw=expected;require(center['name']==name,'Frozen center order changed')
        modelpath=Path(center['model']);selectedpath=Path(center['selected_pose']);selected=read(selectedpath)
        require(sha(modelpath)==center['model_sha256'] and sha(selectedpath)==center['selected_pose_sha256'],'Frozen selected chart/row changed')
        require((selected['population'],selected['seed'],selected['draw'])==(population,seed,draw),'Selected original identity changed')
        source=f'{name}-confirmation';record=guided['campaigns'][source]['top_16'][0]
        require(dict(selected['original_row'],population=population)==record,'Selected original row/weight differs from audited maximum')
        require(selected['pose']==record['pose'] and selected['original_log_importance_weight']==record['log_importance_weight'],'Selected pose or original weight changed')
        for key in ('samples','manifest','campaign_manifest','config','comparison','full_density_audit'):
            require(sha(selected[f'source_{key}_path'])==selected[f'source_{key}_sha256'],'Selected original source hash changed')
        models[name]=read(modelpath);peaks[name]=selected['pose'];selection[name]=selected
    require(set(models)==set(CENTERS) and all(m['shape_sha256']==protocol['shape_sha256'] for m in models.values()),'Three chart shapes differ')
    geometry_reports={};geometries={}
    for name in CENTERS:
        geometry_reports[name],geometries[name]=geometry_model_audit(models[name],dict(fixed_neighbor=cfg['fixed_poses'][0]),cfg,peaks[name])
    requests=[]
    for declared in protocol['campaigns']:
        root=Path(declared['output']);master=verify_reference_manifest(root,declared,protocol,cfg,models[declared['name']]);assert_terminal(root,master)
        requests.append((declared,root,master))
    histories_context=[read_history(h,cfg,protocol['shape_sha256']) for h in protocol['historical_campaigns']]
    newseeds=[j['seed'] for _,_,master in requests for j in master['jobs']]
    oldseeds=[j['seed'] for context in histories_context for j in context['master']['jobs']]
    oldseeds += [p['seed'] for c in direct['campaigns'] for p in c['populations']]
    require(len(newseeds)==len(set(newseeds)) and not set(newseeds)&set(oldseeds),'Fresh/historical/direct streams overlap')
    out.mkdir(parents=True);archive=out/'provenance';archive.mkdir()
    executing=local_dependencies([Path(__file__)])
    static={'protocol.json':prep/'protocol.json','freeze.json':prep/'freeze.json','config.json':prep/'config.json',
        'commands.json':prep/'commands.json','direct-analysis.json':direct_path,
        **{f'model-{n}.json':(prep/'provenance/direct-model.json' if n=='direct' else Path(next(c for c in protocol['centers'] if c['name']==n)['model'])) for n in CENTERS}}
    for name,path in {**executing,**static}.items():(archive/name).write_bytes(Path(path).read_bytes())
    execution_hashes={str(path):sha(path) for path in executing.values()}
    write(out/'execution-source-freeze.json',execution_hashes)
    ensure_original_audits(requests,out,run_audits)
    campaigns=[]
    for declared,root,master in requests:
        auditpath=root/'assessment/analysis.json';original=read(auditpath)
        require(original['original_q_window']==WINDOW and original['independently_reconstructed_poses']==declared['populations']*declared['samples_per_population'],'New original all-row audit incomplete')
        context=dict(root=root,master=master,kind='latent',original=original,record=None,name=f"{declared['name']}-R{declared['radius_A']}",audit_path=auditpath,
            expected=dict(physical=original['estimate']['logQ'],hard=original['hard_region']['logQ']),CPU_seconds=original['sampler_cpu_seconds'],
            sources={str(p):sha(p) for p in (root/'manifest.json',root/'provenance/config.json',root/'provenance/region.json',auditpath)})
        result=remask(context,cfg,models,geometries,declared['name'],declared['radius_A'])
        require(result['original_unconditional_draws']==declared['populations']*declared['samples_per_population'],'New source full N changed')
        campaigns.append(result);print(f"{result['name']}: logQ={result['physical']['full']['logQ']}",flush=True)
    histories=[]
    for context in histories_context:
        context.update(name=context['declared']['name'],CPU_seconds=context['record']['CPU_seconds'])
        result=remask(context,cfg,models,geometries);require(result['original_unconditional_draws']==context['declared']['original_N'],'Historical full N changed')
        histories.append(result)
    combined=combine_assigned(campaigns,direct);comparisons=matching_comparisons(histories,campaigns,combined)
    sources={str(prep/name):sha(prep/name) for name in freeze};sources[str(direct_path)]=sha(direct_path)
    for campaign in campaigns+histories:sources.update(campaign['input_sha256'])
    for name,digest in {**sources,**execution_hashes}.items():require(sha(name)==digest,'Input or executing source changed during audit')
    outside={h['name']:{kind:dict(estimate=h[kind]['outside_union'],full_inner_estimate=h[kind]['full'],
        same_row_union_covariance=h['same_row_covariances'][kind]['union']['outside_union']) for kind in ('physical','hard')} for h in histories}
    report=dict(complete=True,original_q_window=WINDOW,center_order=list(CENTERS),closed_ball_radius_A=.5,
        campaigns=campaigns,histories=histories,direct_reference=dict(path=str(direct_path),sha256=sha(direct_path),independent_shell_sum=direct['independent_shell_sum']),
        independent_assigned_region_sums=combined['regions'],independent_union_sum=combined['union'],independent_source_pieces=combined['source_pieces'],
        historical_outside_union=outside,**comparisons,selection=selection,geometric_model_audits=geometry_reports,
        source_sha256=sources,executing_source_sha256=execution_hashes,archived_sha256={p.name:sha(p) for p in archive.iterdir()},
        scope='Finite priority-disjoint three-ball inner-shoulder calibration. Every source keeps its original density/Jacobian and full unconditional N, including all invalid and outer-shoulder zeros. Historical outside-union estimates remain separate; no whole-inner estimate is formed by stitching them to fresh finite-region means. Fresh assigned shells sum only independent streams and independent variances. Within-source before/after comparisons retain covariance. Data-selected centers and finite agreement do not establish absence of unseen mass, convergence, equilibrium mixing or assembly.')
    write(out/'analysis.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preparation',type=Path,default=DEFAULT_PREPARATION)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run-original-audits',action='store_true')
    args=parser.parse_args();result=analyze(args.preparation,args.out,args.run_original_audits)
    print(dict(complete=result['complete'],output=str(args.out/'analysis.json')))
