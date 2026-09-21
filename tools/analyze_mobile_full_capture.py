#!/usr/bin/env python3
"""Partition already audited full-capture importance rows without rerunning audits.

The primary partition is exhaustive on valid poses in the fixed D170 target.
No unobserved region is called zero, and different proposal laws stay separate.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
PRIMARY=('native_q_le_1','competitor_r12','remaining_q_gt_1')
NAMES=('total',)+PRIMARY+('native_r4','native_outside_r4','bound','unbound',
    'competitor_r3','competitor_shell_3_5','competitor_shell_5_8','competitor_shell_8_12')
PAIRINGS=(('native_q_le_1','total'),('competitor_r12','total'),('remaining_q_gt_1','total'),
          ('native_q_le_1','competitor_r12'),('native_q_le_1','remaining_q_gt_1'),
          ('native_r4','native_q_le_1'),('bound','total'))


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(value,message):
    if not value:raise ValueError(message)


def moments(logs):
    values=np.asarray(logs,float);n=len(values)
    require(values.ndim==1 and n>=2 and not np.isnan(values).any() and not np.isposinf(values).any(),'Invalid fixed-N log weights')
    nonzero=int(np.isfinite(values).sum())
    if not nonzero:
        return dict(draws=n,nonzero=0,logQ=None,ESS=0.,relative_SE=None,maximum_row_fraction=None,
                    interpretation='Unobserved regional mass, not physical zero or an upper bound.')
    total=logsumexp(values);weights=np.exp(values-total)
    variance=float(n/(n-1)*np.sum((weights-1/n)**2))
    return dict(draws=n,nonzero=nonzero,logQ=float(total-math.log(n)),ESS=float(1/(weights@weights)),
                relative_SE=math.sqrt(variance),maximum_row_fraction=float(weights.max()),
                top_one_percent_fraction=float(np.sort(weights)[-max(1,math.ceil(.01*n)):].sum()),
                interpretation='Observed contributions only; unseen high-weight poses remain possible.')


def paired_ratio(log_a,log_b):
    """Shared-draw delta covariance; supports may differ or be disjoint."""
    a,b=np.asarray(log_a,float),np.asarray(log_b,float)
    require(a.shape==b.shape and a.ndim==1 and len(a)>=2,'Ratio vectors differ')
    ma,mb=moments(a),moments(b)
    if ma['logQ']is None or mb['logQ']is None:
        return dict(log_ratio=None,log_ratio_SE=None,relative_covariance=None,
                    reason='One or both regional integrals have no nonzero observations.')
    n=len(a);wa=np.exp(a-logsumexp(a));wb=np.exp(b-logsumexp(b))
    covariance=float(n/(n-1)*((wa-1/n)@(wb-1/n)))
    variance=float(n/(n-1)*np.sum((wa-wb)**2))
    return dict(log_ratio=ma['logQ']-mb['logQ'],log_ratio_SE=math.sqrt(variance),
                relative_covariance=covariance,
                scope='Paired shared-draw delta SE; never an independent-numerator/denominator error formula.')


def paired_noise(logs,pairs):
    values=np.asarray(logs,float);pairs=np.asarray(pairs,float)
    require(pairs.shape==(len(values),2),'Expected two clouds per unconditional draw')
    if not np.isfinite(values).any():return None
    offset=float(max(values.max(),pairs.max()));a,b=np.exp(pairs-offset).T;y=(a+b)/2
    total=float(np.var(y,ddof=1));cloud=float(np.mean((a-b)**2)/4);pose=total-cloud
    mean=float(np.mean(y));n=len(y)
    return dict(log_weight_offset=offset,scaled_total_variance=total,scaled_cloud_variance=cloud,
                scaled_residual_pose_variance=pose,cloud_fraction=cloud/total if total>0 else None,
                relative_variance_mean_cloud=cloud/n/mean**2,relative_variance_mean_pose=pose/n/mean**2,
                scope='Observed two-cloud decomposition; residual pose variance is not clipped. It does not bound unseen tails.')


def chart_radii(poses,region):
    """Original frozen left-Cayley chart radii; exact seams map to infinity."""
    if not poses:return np.empty(0),np.empty(0,dtype=bool)
    model=region['gaussian_chart'];require(len(model['weights'])==1,'Partition requires one original chart')
    fixed=region['fixed_neighbor'];anchor=model['anchors'][0]
    fixed_r=Rotation.from_quat(np.asarray(fixed['orientation'])[[1,2,3,0]]).as_matrix()
    r=Rotation.from_quat(np.array([p['orientation']for p in poses])[:,[1,2,3,0]]).as_matrix()
    relative_r=np.einsum('ij,njk->nik',fixed_r.T,r)
    delta=relative_r@np.asarray(anchor['rotation']).T
    quaternion=Rotation.from_matrix(delta).as_quat()
    seam=quaternion[:,3]==0.
    displacement=np.array([p['position']for p in poses])-fixed['position']
    relative_t=displacement@fixed_r-anchor['position']
    radius=np.full(len(poses),np.inf)
    selected=np.flatnonzero(~seam)
    if len(selected):
        with np.errstate(over='raise',divide='raise',invalid='raise'):
            coordinates=np.column_stack((relative_t[selected],model['angular_length']*quaternion[selected,:3]/quaternion[selected,3,None]))
            lower=np.linalg.cholesky(np.asarray(model['covariances'][0]))
            require(np.isfinite(coordinates).all(),'Non-seam chart inverse overflow')
            u=solve_triangular(lower,(coordinates-model['means'][0]).T,lower=True).T
            # hypot avoids overflow from squaring ordinary large coordinates.
            # A remaining non-seam failure is an error, never a censored row.
            radius[selected]=np.hypot.reduce(u,axis=1)
            require(np.isfinite(radius[selected]).all(),'Non-seam chart norm overflow')
    require(not np.isnan(radius).any(),'Undefined partition radius')
    return radius,seam


def partition_masks(valid,q,rho_comp,rho_native,bound):
    valid=np.asarray(valid,bool);q=np.asarray(q,float);rc=np.asarray(rho_comp,float);rn=np.asarray(rho_native,float);bound=np.asarray(bound,bool)
    require(valid.ndim==1 and all(x.shape==valid.shape for x in (q,rc,rn,bound)),'Partition vectors differ')
    require(np.isfinite(q[valid]).all() and not np.isnan(rc[valid]).any() and not np.isnan(rn[valid]).any(),'Invalid region coordinate')
    native=valid&(q<=1.);other=valid&(q>1.)
    masks=dict(total=valid,native_q_le_1=native,competitor_r12=other&(rc<=12.),remaining_q_gt_1=other&(rc>12.),
               native_r4=native&(rn<=4.),native_outside_r4=native&(rn>4.),bound=valid&bound,unbound=valid&~bound,
               competitor_r3=other&(rc<=3.),competitor_shell_3_5=other&(rc>3.)&(rc<=5.),
               competitor_shell_5_8=other&(rc>5.)&(rc<=8.),competitor_shell_8_12=other&(rc>8.)&(rc<=12.))
    require(np.array_equal(sum(masks[n].astype(int)for n in PRIMARY),valid.astype(int)),'Primary partition has an overlap or gap')
    require(np.array_equal(masks['native_r4']|masks['native_outside_r4'],native),'Native split fails')
    compnames=('competitor_r3','competitor_shell_3_5','competitor_shell_5_8','competitor_shell_8_12')
    require(np.array_equal(sum(masks[n].astype(int)for n in compnames),masks['competitor_r12'].astype(int)),'Competitor strata overlap or leave gaps')
    for name in tuple(k for k in masks if k not in ('total','bound','unbound')):
        masks[name+'_bound']=masks[name]&bound
        masks[name+'_unbound']=masks[name]&~bound
    return masks


def region_stats(log_z,log_h,pairs,mask):
    z=np.where(mask,log_z,-np.inf);h=np.where(mask,log_h,-np.inf);p=np.where(mask[:,None],pairs,-np.inf)
    return dict(physical=moments(z),hard=moments(h),depletion_enhancement=paired_ratio(z,h),cloud_variance=paired_noise(z,p))


def relative_covariance(log_columns):
    names=list(log_columns);arrays=[np.asarray(log_columns[n],float)for n in names]
    n=len(arrays[0]);weights=[]
    for a in arrays:
        require(a.shape==(n,),'Covariance denominators differ')
        weights.append(np.exp(a-logsumexp(a))-1/n if np.isfinite(a).any() else None)
    matrix=[[None if a is None or b is None else float(n/(n-1)*(a@b))for b in weights]for a in weights]
    return dict(regions=names,covariance_relative=matrix,
                scope='Shared attempted draws, normalized by the observed regional means. Entries involving unobserved regions are unresolved.')


def verify_frozen_parent(root,audit_recovery=None):
    protocol,status=read(root/'protocol.json'),read(root/'status.json')
    require(protocol['schema']=='mobile-full-capture-controller-v1','Unexpected full-capture controller')
    if audit_recovery is None:
        require(status['complete'] and status['phase']=='complete','Full-capture campaign incomplete')
    else:
        from recover_mobile_full_capture_audit import validate_completed
        _, recovered=validate_completed(root,audit_recovery)
        status=dict(status,audits=recovered['audits'],audit_recovery=str(Path(audit_recovery).resolve()))
    require(status['protocol_sha256']==sha(root/'protocol.json'),'Controller protocol changed')
    frozen=read(root/'freeze.json');files=frozen.get('files',frozen)
    for name,digest in files.items():require(sha(root/name)==digest,'Frozen campaign input changed: '+name)
    require(len(protocol['campaigns'])==2 and len(status['jobs'])==8,'Expected two laws and eight populations')
    return protocol,status


def load_native_classifier(definition):
    if definition is None:return None,None
    definition=Path(definition).resolve()
    data=read(definition)
    runtime=definition.parent/'inputs/source/native_contact_regions.py'
    require(runtime.exists(),'Frozen stateless native classifier runtime missing')
    require(sha(runtime)==data['input_sha256']['source/native_contact_regions.py'],'Frozen native runtime hash differs')
    spec=importlib.util.spec_from_file_location('frozen_native_contact_regions',runtime)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.NativeContactRegions(definition),dict(definition=str(definition),definition_sha256=sha(definition),runtime_sha256=sha(runtime))


def target_equal(config,region,shape_sha):
    require(region['shape_sha256']==shape_sha,'Reference hard shape differs')
    for region_key,config_key in [('physical_fixed_neighbors','fixed_poses'),('capture_center','capture_center'),
        ('capture_radius','capture_radius'),('activity','reservoir_density'),('depletant_radius','depletant_radius'),('physical_metric','metadata')]:
        require(region[region_key]==config[config_key],'Reference physical target differs: '+region_key)


def analyze_arm(root,entry,status,classifier,out):
    arm=entry['arm'];folder=Path(entry['path']).resolve();require(folder==root/arm,'Arm path escaped campaign')
    require(sha(folder/'manifest.json')==entry['manifest_sha256'],'Arm manifest changed')
    manifest=read(folder/'manifest.json');require(manifest['experiment_schema']=='mobile-full-capture-arm-v1','Unexpected arm schema')
    audit_record=status['audits'][arm]
    require(audit_record['returncode']==0 and audit_record['manifest_sha256']==entry['manifest_sha256'],'Arm audit failed or differs')
    assessment_path=Path(audit_record['assessment_path'])if 'audit_recovery'in status else folder/'assessment/analysis.json'
    require(sha(assessment_path)==audit_record['analysis_sha256'],'Completed audit changed')
    assessment=read(assessment_path);require(not assessment['pending'],'Audit omitted pending populations')
    for name,digest in manifest['archive_sha256'].items():require(sha(folder/'provenance'/name)==digest,'Archived input changed: '+name)
    cfg=read(folder/'provenance/config.json');shape_sha=sha(folder/'provenance/shape.json')
    if classifier is not None:
        require(classifier.fixed_poses==cfg['fixed_poses'] and classifier.shape_sha256==shape_sha,'Native classifier scaffold or hard shape differs')
    partition=folder/'provenance/partition'
    native_region=read(partition/'region-native-r4.json');comp_region=read(partition/'region-competitor-r3.json')
    target_equal(cfg,native_region,shape_sha);target_equal(cfg,comp_region,shape_sha)
    require(cfg['capture_radius']==170. and cfg['reservoir_density']==.035 and cfg['depletant_radius']==1.5,'Wrong fixed D170 physical target')
    jobs=manifest['jobs'];require(len(jobs)==4 and len(entry['expected_jobs'])==4,'Expected four independent populations')
    terminal={j['id']:j for j in status['jobs']if j['arm']==arm}
    require(len(terminal)==4,'Terminal population set differs')
    arrays=[];populations=[];hashes={};cpu=0.;label_path=out/f'{arm}-native-labels.jsonl'
    label_stream=label_path.open('x')if classifier is not None else None
    try:
        for job,expected in zip(jobs,entry['expected_jobs']):
            require(all(job[k]==expected[k]for k in ('id','seed','samples')) and job['samples']==8192,'Population budget or seed differs')
            term=terminal[job['id']];require(term['status']=='complete' and term['exit_code']==0,'Physical population failed')
            directory=Path(job['directory']).resolve();require(directory==folder/'runs'/job['id'],'Population path differs')
            for filename,key in [('summary.json','summary_sha256'),('manifest.json','manifest_sha256'),('samples.jsonl','samples_sha256')]:
                digest=sha(directory/filename);require(digest==term['output'][key],'Terminally bound output changed');hashes[str(directory/filename)]=digest
            summary=read(directory/'summary.json');pm=read(directory/'manifest.json')
            require(summary['complete'] and summary['manifest']==pm and pm['schema']==3,'Schema3 population incomplete or inconsistent')
            require(pm['seed']==job['seed'] and pm['samples']==8192 and pm['cloud_replicates']==2,'Population allocation changed')
            require(pm['uniform_probability']==entry['epsilon'] and pm['covariance_scale']==1. and pm['proposal_anchor_index']is None,
                    'Full-support law or anchor marginalization changed')
            require(pm['physical_fixed_neighbor_count']==2 and pm['base_component_count']==150 and pm['virtual_component_count']==300
                    and len(pm['reciprocal_components'])==150 and all(pm['reciprocal_components']),'Reciprocal law differs')
            require(pm['executable_sha256']==manifest['archive_sha256']['basin-normalizer'],'Executable differs')
            for filename,key in [('input-config.json','config_sha256'),('model.json','model_sha256'),('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
                require(sha(directory/'provenance'/filename)==pm[key],'Population provenance changed')
            require(pm['config_sha256']==sha(folder/'provenance/config.json') and pm['shape_sha256']==shape_sha,'Population physical input differs')
            require(pm['model_sha256']==manifest['model_sha256'] and pm['source_bundle_sha256']==manifest['source_bundle_sha256'],
                    'Population model or reviewed source differs')
            require(pm['activity']==.035 and pm['lambda']==.035*64.,'Population bath law differs')
            audited=next(p for p in assessment['populations']if p['job']['id']==job['id'])
            require(audited['job']['seed']==job['seed'],'Audited population seed differs')
            for filename in ('samples.jsonl','summary.json','manifest.json'):
                require(assessment['provenance'][str(directory/filename)]==hashes[str(directory/filename)],'Audit row/source binding differs')
            rows=[json.loads(line)for line in(directory/'samples.jsonl').read_text().splitlines()]
            require(len(rows)==8192 and [r['draw']for r in rows]==list(range(8192)),'Unconditional row sequence changed')
            valid=np.array([r['hard_valid']for r in rows],bool);indices=np.flatnonzero(valid)
            poses=[rows[i]['pose']for i in indices]
            rc=np.full(len(rows),np.inf);rn=rc.copy();seams=np.zeros(len(rows),bool)
            if len(poses):rc[indices],seams[indices]=chart_radii(poses,comp_region);rn[indices],_=chart_radii(poses,native_region)
            q=np.array([r['q']if r['q']is not None else np.inf for r in rows]);bound=np.array([r['depletion_contact']is True for r in rows])
            masks=partition_masks(valid,q,rc,rn,bound)
            if classifier is not None:
                any_entry=np.zeros(len(rows),bool);both_entry=any_entry.copy();triangle_entry=any_entry.copy()
                for index in indices:
                    label=classifier.classify(rows[index]['pose'])
                    any_entry[index]=label['native_any'];both_entry[index]=label['cooperative_entry']
                    triangle_entry[index]=label['registry_consistent_triangle']
                    label_stream.write(json.dumps(dict(population=job['id'],draw=int(index),classification=label),allow_nan=False)+'\n')
                masks.update(instantaneous_native_entry_any=valid&any_entry,instantaneous_native_entry_both=valid&both_entry,
                             instantaneous_registry_triangle=valid&triangle_entry,instantaneous_no_native_entry=valid&~any_entry)
                for name in PRIMARY:
                    masks[name+'_native_entry']=masks[name]&any_entry
                    masks[name+'_no_native_entry']=masks[name]&~any_entry
            z=np.array([r['log_importance_weight']if r['log_importance_weight']is not None else -np.inf for r in rows])
            h=np.array([r['log_hard_weight']if r['log_hard_weight']is not None else -np.inf for r in rows])
            pairs=np.array([[c['log_weight']-r['log_proposal_density']for c in r['clouds']]if r['hard_valid']else[-np.inf,-np.inf]for r in rows])
            stats={name:region_stats(z,h,pairs,mask)for name,mask in masks.items()}
            for ours,theirs in [('total','total'),('native_q_le_1','native'),('bound','bound'),('unbound','unbound')]:
                got,want=stats[ours]['physical']['logQ'],audited['estimates'][theirs]['logQ']
                require(got is None and want is None or got is not None and want is not None and abs(got-want)<2e-10,'Saved audit aggregation differs')
            arrays.append(dict(z=z,h=h,pairs=pairs,masks=masks))
            populations.append(dict(id=job['id'],seed=job['seed'],CPU_seconds=summary['sampler_cpu_seconds'],
                estimates=stats,hard_valid=int(valid.sum()),capture_rejected=sum(not r['capture_valid']for r in rows),
                exact_competitor_chart_seams=int(seams.sum()),source_sha256=hashes[str(directory/'samples.jsonl')]))
            cpu+=summary['sampler_cpu_seconds']
    finally:
        if label_stream is not None:label_stream.close()
    z=np.concatenate([a['z']for a in arrays]);h=np.concatenate([a['h']for a in arrays]);pairs=np.concatenate([a['pairs']for a in arrays])
    masks={name:np.concatenate([a['masks'][name]for a in arrays])for name in arrays[0]['masks']}
    estimates={}
    for name,mask in masks.items():
        stats=region_stats(z,h,pairs,mask)
        popz=np.array([-np.inf if p['estimates'][name]['physical']['logQ']is None else p['estimates'][name]['physical']['logQ']for p in populations])
        poph=np.array([-np.inf if p['estimates'][name]['hard']['logQ']is None else p['estimates'][name]['hard']['logQ']for p in populations])
        stats['population_uncertainty']=dict(physical=moments(popz),hard=moments(poph),depletion_enhancement=paired_ratio(popz,poph))
        stats['population_logQ']=[None if not np.isfinite(v)else float(v)for v in popz]
        stats['allzero_populations']=int(np.isneginf(popz).sum())
        stats['CPU_times_row_relative_variance']=None if stats['physical']['relative_SE']is None else cpu*stats['physical']['relative_SE']**2
        stats['CPU_times_population_relative_variance']=None if stats['population_uncertainty']['physical']['relative_SE']is None else cpu*stats['population_uncertainty']['physical']['relative_SE']**2
        stats['cost_precision_scope']='Observed planning metrics, not a demonstrated speedup or unseen-tail bound.'
        estimates[name]=stats
    contrasts={}
    for a,b in PAIRINGS:
        row=paired_ratio(np.where(masks[a],z,-np.inf),np.where(masks[b],z,-np.inf))
        pa=[-np.inf if p['estimates'][a]['physical']['logQ']is None else p['estimates'][a]['physical']['logQ']for p in populations]
        pb=[-np.inf if p['estimates'][b]['physical']['logQ']is None else p['estimates'][b]['physical']['logQ']for p in populations]
        contrasts[a+'/'+b]=dict(row=row,population=paired_ratio(pa,pb))
    return dict(arm=arm,epsilon=entry['epsilon'],samples=len(z),CPU_seconds=cpu,populations=populations,estimates=estimates,
        physical_ratio_contrasts=contrasts,primary_relative_covariance=relative_covariance({n:np.where(masks[n],z,-np.inf)for n in PRIMARY}),
        source_sha256=hashes,manifest_sha256=entry['manifest_sha256'],assessment_sha256=audit_record['analysis_sha256'],
        native_labels_sha256=sha(label_path)if classifier is not None else None,
        physical_config=cfg,shape_sha256=shape_sha,
        scope='Conditional D170 mass only. All invalid proposals retained as zeros; independent proposal laws are not pooled. '
              'No-entry native labels are instantaneous threshold failures, not proof of a nonnative equilibrium basin.')


def compare_references(arms):
    paths=[ROOT/'runs/mobile-competing-reference-comparison-20260921/analysis.json',
           ROOT/'runs/mobile-competing-outer-comparison-20260921/analysis.json']
    output=[]
    for path in paths:
        require(path.exists(),'Independent regional reference missing')
        result=read(path);require(result['complete'],'Independent reference incomplete')
        for name,digest in read(path.parent/'freeze.json').items():require(sha(path.parent/name)==digest,'Saved reference changed')
        if 'reference-comparison' in path.parent.name:
            records=[r for r in result['regions']if r['name']=='native-r4']
            mapping={'native-r4':'native_r4'}
        else:
            records=[r for r in result['derived_disjoint_sums']if r['name']=='competitor-r12']
            mapping={'competitor-r12':'competitor_r12'}
        # Every named reference stratum has an archived exact region/config.
        for reference in result['regions']:
            cfg=read(Path(reference['campaign'])/'provenance/config.json')
            region=read(Path(reference['campaign'])/'provenance/region.json')
            for arm in arms:target_equal(arm['physical_config'],region,arm['shape_sha256'])
            for source,digest in reference['source_sha256'].items():require(sha(source)==digest,'Independent regional rows changed')
        for record in records:
            for arm in arms:
                fresh=arm['estimates'][mapping[record['name']]]['physical'];old=record['row_uncertainty']
                output.append(dict(arm=arm['arm'],region=record['name'],reference=str(path),reference_sha256=sha(path),
                    full_capture_logQ=fresh['logQ'],regional_logQ=old['log_Qz'],
                    log_difference=None if fresh['logQ']is None else fresh['logQ']-old['log_Qz'],
                    full_capture_row_SE=fresh['relative_SE'],regional_row_SE=old['Qz_relative_SE'],
                    scope='Identical fixed-region physical target; distinct sampling laws reported separately, never pooled. Both error estimates can miss unseen tails.'))
    return output


def plot_populations(arms,out):
    """Display every independent population and its observed delta SE."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('total',)+PRIMARY
    labels=('Total','Original q≤1','Competitor R12','q>1 remainder')
    fig,axes=plt.subplots(2,3,figsize=(14,8),squeeze=False,layout='constrained')
    for row,arm in enumerate(arms):
        for col,(kind,title) in enumerate((('physical','log Qz'),('hard','log Q0'),('depletion_enhancement','log(Qz / Q0)'))):
            ax=axes[row,col]
            for index,name in enumerate(names):
                for j,population in enumerate(arm['populations']):
                    item=population['estimates'][name][kind]
                    value=item['log_ratio']if col==2 else item['logQ']
                    error=item['log_ratio_SE']if col==2 else item['relative_SE']
                    if value is not None:
                        ax.errorbar(index+(j-1.5)*.12,value,yerr=error,fmt='o',ms=4,c=f'C{j}',alpha=.8,
                                    label=f'Population {j+1}'if index==0 and row==0 and col==0 else None)
                pooled=arm['estimates'][name][kind]
                value=pooled['log_ratio']if col==2 else pooled['logQ']
                error=pooled['log_ratio_SE']if col==2 else pooled['relative_SE']
                if value is not None:ax.errorbar(index+.27,value,yerr=error,fmt='D',ms=4,c='black')
            ax.set_xticks(range(len(names)),labels,rotation=20,ha='right')
            ax.set_ylabel(f'ε = {arm["epsilon"]}: {title}');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Full D170 integration — separate full-support proposal laws\nFour independent populations; black diamonds pool only within each law. Unobserved regions omitted, not zero.')
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'population-comparison.{suffix}',dpi=160)
    plt.close(fig)


def analyze(root,out,native_definition=None,audit_recovery=None):
    root,out=Path(root).resolve(),Path(out).resolve();require(not out.exists(),'Use fresh immutable analysis output')
    protocol,status=verify_frozen_parent(root,audit_recovery)
    frozen_definition=root/protocol['partition']['native_definition']
    if native_definition is not None:
        require(sha(native_definition)==protocol['partition']['native_definition_sha256'],'Requested native definition differs from prelaunch freeze')
    require(sha(frozen_definition)==protocol['partition']['native_definition_sha256'],'Frozen native definition changed')
    classifier,classifier_binding=load_native_classifier(frozen_definition)
    out.mkdir();arms=[analyze_arm(root,e,status,classifier,out)for e in protocol['campaigns']]
    seeds=[p['seed']for a in arms for p in a['populations']];require(len(seeds)==len(set(seeds))==8,'Population streams repeated')
    references=compare_references(arms)
    result=dict(schema='mobile-full-capture-partition-analysis-v1',complete=True,arms=arms,independent_regional_comparisons=references,
        protocol_sha256=sha(root/'protocol.json'),status_sha256=sha(root/'status.json'),native_classifier=classifier_binding,
        audit_recovery=None if audit_recovery is None else dict(path=str(Path(audit_recovery).resolve()),
            manifest_sha256=sha(Path(audit_recovery)/'manifest.json'),status_sha256=sha(Path(audit_recovery)/'status.json')),
        audits_rerun=0,physical_jobs_launched=0,
        primary_partition='q<=1; q>1 and rho_comp<=12; q>1 and rho_comp>12, including exact Cayley seams in the final region',
        scope='Exhaustive partition of the specified D170 conditional target only. Exhaustive labels and positive proposal support do not certify statistical coverage. '
              'No result bounds unseen high-weight contributions, mass outside D170, or the full spherical/mobile assembly ensemble.')
    write(out/'analysis.json',result);shutil.copy2(Path(__file__),out/'analyzer.py')
    plot_populations(arms,out)
    lines=['Independent full-support D170 contact integration; separate proposal laws.\n',
           '| epsilon | Region | log Qz | row RSE | population RSE | ESS | largest row | nonzero / attempted |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        for name in ('total',)+PRIMARY+('native_r4','native_outside_r4','bound','unbound',
            'instantaneous_native_entry_any','instantaneous_native_entry_both','instantaneous_registry_triangle','instantaneous_no_native_entry'):
            item=arm['estimates'][name];a=item['physical'];p=item['population_uncertainty']['physical']
            values='unresolved | — | — | 0 | —'if a['logQ']is None else f"{a['logQ']:.6f} | {a['relative_SE']:.2%} | {p['relative_SE']:.2%} | {a['ESS']:.1f} | {a['maximum_row_fraction']:.2%}"
            lines.append(f"| {arm['epsilon']} | {name} | {values} | {a['nonzero']} / {a['draws']} |")
        lines.append(f"\nSampler CPU at epsilon {arm['epsilon']}: {arm['CPU_seconds']:.2f} s.\n")
    lines+=['',result['scope'],'','Qz/Q0 enhancement, shared-draw ratio covariance, paired-cloud variance, every population, nested strata and reference comparisons are retained in analysis.json. '
        'A missing region is unresolved, not zero. Instantaneous native-entry labels, when available, are separate diagnostics and never replace the original q/rho partition.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');write(out/'freeze.json',{p.name:sha(p)for p in out.iterdir()if p.is_file()})
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--native-definition',type=Path)
    parser.add_argument('--audit-recovery',type=Path,help='Separately validated corrected observer; original failed experiment remains unchanged')
    args=parser.parse_args();analyze(args.root,args.out,args.native_definition,args.audit_recovery)
