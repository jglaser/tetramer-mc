#!/usr/bin/env python3
"""Post-hoc geometry/noise diagnostics from existing, immutable reference rows.

No physical sampling, raw-audit replay, classifier change, or fitted proposal.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='1'
import argparse,gzip,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist,squareform
from scipy.spatial.transform import Rotation
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.special import logsumexp
from analyze_mobile_native_pocket import require,read,write,sha,summarize
from analyze_mobile_competing_reference import paired_moments

ROOT=Path(__file__).resolve().parents[1]
CLASSES=('registered_native_entry','contact_no_native_entry')

def rmat(q):return Rotation.from_quat(np.asarray(q)[..., [1,2,3,0]]).as_matrix()
def chart_coordinates(poses,region):
    model=region['gaussian_chart'];R=rmat(region['fixed_neighbor']['orientation'])
    t=np.asarray([p['position']for p in poses]);r=rmat([p['orientation']for p in poses])
    relative_t=(t-region['fixed_neighbor']['position'])@R-model['anchors'][0]['position']
    relative_r=R.T@r@np.asarray(model['anchors'][0]['rotation']).T
    q=Rotation.from_matrix(relative_r).as_quat();require(np.all(np.abs(q[:,3])>1e-12),'Unexpected chart seam')
    x=np.column_stack((relative_t,model['angular_length']*q[:,:3]/q[:,3,None]))
    return solve_triangular(np.linalg.cholesky(model['covariances'][0]),(x-model['means'][0]).T,lower=True).T

def chart_center(region):
    chart=region['gaussian_chart'];mean=np.asarray(chart['means'][0]);rf=rmat(region['fixed_neighbor']['orientation'])
    pose_r=rf@Rotation.from_quat(np.r_[mean[3:]/chart['angular_length'],1.]).as_matrix()@chart['anchors'][0]['rotation']
    q=Rotation.from_matrix(pose_r).as_quat()[[3,0,1,2]]
    return dict(position=(region['fixed_neighbor']['position']+rf@(np.asarray(chart['anchors'][0]['position'])+mean[:3])).tolist(),orientation=q.tolist())

def selected_lines(path,wanted,compressed=False):
    result={};opener=gzip.open if compressed else open
    with opener(path,'rt')as stream:
        for i,line in enumerate(stream):
            if i in wanted:
                value=json.loads(line);require(value['draw']==i,'Saved draw identity changed');result[i]=value
            if len(result)==len(wanted):break
    require(set(result)==set(wanted),'Missing selected rows')
    return result

def cloud_diagnostic(row,z,ratio):
    c=row['clouds'];require(len(c)==2,'Expected paired clouds');a,b=c
    for field in ('lower_volume','uncertain_volume','upper_volume'):
        require(a[field]==b[field],'Paired geometric envelopes differ')
    n=np.array([a['overlap_points'],b['overlap_points']],float);lw=np.array([a['log_weight'],b['log_weight']])
    lam=ratio*z;factor=math.log1p(1/ratio)
    require(np.max(abs(lw-(z*a['lower_volume']+n*factor)))<1e-10,'Saved cloud/count identity changed')
    count_sum=float(n.sum());c_est=a['lower_volume']+count_sum/(2*lam)
    log_proxy=z*c_est;noise_proxy=count_sum/(2*ratio*ratio)
    return dict(raw_points=[a['raw_points'],b['raw_points']],overlap_points=n.astype(int).tolist(),
                lower_volume=a['lower_volume'],uncertain_volume=a['uncertain_volume'],log_cloud_weights=lw.tolist(),
                absolute_log_cloud_difference=float(abs(lw[0]-lw[1])),
                count_difference_standardized=None if count_sum==0 else float((n[0]-n[1])/math.sqrt(count_sum)),
                z_times_count_estimated_overlap=log_proxy,z_times_count_overlap_SE=math.sqrt(count_sum)/(2*ratio),
                single_cloud_log_variance_plugin=count_sum/2*factor*factor,
                single_cloud_relative_variance_plugin=math.expm1(noise_proxy),
                two_cloud_relative_SE_plugin=math.sqrt(math.expm1(noise_proxy)/2),
                scope='Counts diagnose geometry and cloud noise only. The count-derived zC proxy is not substituted for the unbiased physical weight; top-row selection biases these diagnostics.')

def summary_values(values):
    a=np.asarray(values,float)
    return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)))if len(a)else None

def select_current(comparison,out,k):
    data=read(comparison/'analysis.json');campaign=Path(data['campaign']);frozen=read(comparison/'freeze.json')
    require(data['complete'] and read(comparison/'status.json')['complete'] and sha(comparison/'analysis.json')==frozen['analysis.json'],'Comparison incomplete or changed')
    require(sha(campaign/'protocol.json')==data['protocol_sha256'] and sha(campaign/'status.json')==data['status_sha256'],'Physical campaign binding changed')
    protocol=read(campaign/'protocol.json');require(read(campaign/'status.json')['complete'],'Physical campaign incomplete')
    region=read(campaign/'pilot_uniform/provenance/region.json');guide=read(campaign/'pilot_ray/provenance/importance-guide.json')
    bindings={str(comparison/'analysis.json'):sha(comparison/'analysis.json'),str(campaign/'protocol.json'):sha(campaign/'protocol.json')}
    chosen=[];coverage=[]
    for arm,source in data['arms'].items():
        total_n=source['estimates']['total']['row_uncertainty']['draws']
        for pop in sorted(source['populations'],key=lambda p:p['id']):
            archive=comparison/pop['records'];require(sha(archive)==pop['records_sha256'],'Numeric record changed');bindings[str(archive)]=sha(archive)
            with np.load(archive,allow_pickle=False)as arrays:
                valid=np.isfinite(arrays['z']);masks={CLASSES[0]:valid&arrays['native'].astype(bool),CLASSES[1]:valid&arrays['contact'].astype(bool)&~arrays['native'].astype(bool)}
                selections={};details={}
                for cls,mask in masks.items():
                    ids=np.flatnonzero(mask);ids=ids[np.argsort(-arrays['z'][ids],kind='stable')];require(len(ids)>0,'Missing class in population')
                    logs=arrays['z'][ids];total=float(logsumexp(logs));fraction=np.exp(logs-total);cs=np.cumsum(fraction)
                    selections[cls]=ids[:k].astype(int).tolist();details[cls]=(total,dict(zip(ids.astype(int).tolist(),logs.tolist())))
                    coverage.append(dict(arm=arm,population=pop['id'],region=cls,nonzero=len(ids),top1_fraction=float(fraction[0]),
                                         top5_fraction=float(fraction[:5].sum()),top10_fraction=float(fraction[:10].sum()),
                                         selected_fraction=float(fraction[:k].sum()),rows_for_50pct=int(np.searchsorted(cs,.5)+1),
                                         rows_for_90pct=int(np.searchsorted(cs,.9)+1),rows_for_99pct=int(np.searchsorted(cs,.99)+1)))
            raw_path=campaign/arm/'runs'/pop['id']/'samples.jsonl';label_path=comparison/pop['labels']
            require(sha(raw_path)==pop['raw_output']['samples_sha256'] and sha(label_path)==pop['labels_sha256'],'Selected raw/label source changed')
            bindings[str(raw_path)]=pop['raw_output']['samples_sha256'];bindings[str(label_path)]=pop['labels_sha256']
            wanted=set(sum(selections.values(),[]));rows=selected_lines(raw_path,wanted);labels=selected_lines(label_path,wanted,True)
            for cls,ids in selections.items():
                total,weights=details[cls];class_total=source['estimates'][cls]['row_uncertainty']['log_Qz']+math.log(total_n)
                for rank,i in enumerate(ids,1):
                    row,label=rows[i],labels[i];require(abs(row['log_importance_weight']-weights[i])<1e-10,'Archived weight changed')
                    native=label['classification']['native_any'];require(native==(cls==CLASSES[0]) and label['contact']['exclusion_contact'],'Saved class changed')
                    matrix=rmat(row['pose']['orientation']);body_errors=[]
                    for interface in guide['interfaces']:
                        placed=np.asarray(interface['moving_members'])@matrix.T+row['pose']['position']
                        distances=np.linalg.norm(placed-interface['target_world_members'],axis=1)
                        body_errors.append(dict(maximum_member_error_A=float(distances.max()),rms_member_error_A=float(np.sqrt(np.mean(distances**2)))))
                    cloud=cloud_diagnostic(row,.035,source['allocation']['lambda_ratio'])
                    chosen.append(dict(id=f'{arm}/{pop["id"]}/{i}/{cls}',arm=arm,population=pop['id'],seed=pop['seed'],draw=i,region=cls,rank_in_population=rank,
                                       fraction_of_population_class=math.exp(weights[i]-total),fraction_of_arm_class=math.exp(weights[i]-class_total),
                                       pose=row['pose'],current_latent=row['latent'],current_radius=row['latent_radius'],
                                       log_hard_weight=row['log_hard_weight'],log_importance_weight=row['log_importance_weight'],
                                       log_proposal_density=row['log_proposal_density'],proposal_branch=row['proposal_branch'],selected_ray_fallback=row['selected_ray_fallback'],
                                       body_errors=body_errors,classification=label['classification'],contact=label['contact'],cloud=cloud))
            print('Read saved extremes '+arm+'/'+pop['id'],flush=True)
    return data,protocol,region,chosen,coverage,bindings

def reference_intersection(reference,current,bindings):
    data=read(reference/'analysis.json');freeze=read(reference/'freeze.json');freeze=freeze.get('files',freeze)
    require(data['complete'] and sha(reference/'analysis.json')==freeze['analysis.json'],'Native reference changed')
    core=next(s for s in data['strata']if s['name']=='r5repeat');root=Path(core['campaign']);region=read(root/'provenance/region.json')
    require(sha(root/'provenance/region.json')==core['region_sha256'],'Native chart changed')
    for field in ('physical_fixed_neighbors','capture_center','capture_radius','shape_sha256','activity','depletant_radius','physical_metric'):
        require(region[field]==current[field],'Native reference physical target differs: '+field)
    labels_path=reference/core['native_labels']['path'];require(sha(labels_path)==core['native_labels']['sha256'],'Reference labels changed')
    bindings[str(reference/'analysis.json')]=sha(reference/'analysis.json');bindings[str(root/'provenance/region.json')]=sha(root/'provenance/region.json');bindings[str(labels_path)]=sha(labels_path)
    labels={}
    with labels_path.open()as stream:
        for line in stream:
            row=json.loads(line)
            if row['applicable']:labels[(row['population'],row['draw'])]=row['label']['native_any']
    populations=[];representatives=[];members=current['physical_metric']['rigid_members']
    for pop in core['populations']:
        file=root/'runs'/pop['id']/'samples.jsonl';require(sha(file)==core['source_sha256'][str(file)],'Reference rows changed');bindings[str(file)]=sha(file)
        with file.open()as stream:rows=[json.loads(line)for line in stream]
        n=pop['draws'];require(len(rows)==n and [r['draw']for r in rows]==list(range(n)),'Reference denominator changed')
        z=np.array([-np.inf if r['log_importance_weight']is None else r['log_importance_weight']for r in rows]);h=np.array([-np.inf if r['log_hard_weight']is None else r['log_hard_weight']for r in rows]);valid=np.isfinite(z)
        require(abs(float(logsumexp(z)-math.log(n))-pop['log_Qz'])<1e-10,'Reference weight check differs')
        ids=np.flatnonzero(valid);u=chart_coordinates([rows[i]['pose']for i in ids],current);radius=np.linalg.norm(u,axis=1)
        include=np.zeros(n,bool);include[ids]=radius<=current['mahalanobis_radius'];native=np.zeros(n,bool)
        native[ids]=[labels[(pop['id'],int(i))]for i in ids]
        pairs=np.full((n,2),-np.inf)
        for i in ids:pairs[i]=[h[i]+c['log_weight']for c in rows[i]['clouds']]
        populations.append(dict(id=pop['id'],seed=pop['seed'],z=z,h=h,pairs=pairs,masks=dict(intersection=include,intersection_native=include&native)))
        ranked=ids[np.argsort(-z[ids],kind='stable')[:5]]
        for i in ranked:
            row=rows[i];representatives.append(dict(id=f'reference/{pop["id"]}/{i}',arm='independent_R5',population=pop['id'],draw=int(i),pose=row['pose'],
                                                  current_latent=chart_coordinates([row['pose']],current)[0].tolist(),reference_latent=row['latent'],
                                                  log_importance_weight=row['log_importance_weight'],cloud=cloud_diagnostic(row,.035,64.),
                                                  native=bool(native[i])))
    estimates={name:summarize(populations,name,[p['masks'][name]for p in populations])for name in ('intersection','intersection_native')}
    return region,dict(source=str(reference),source_sha256=sha(reference/'analysis.json'),original_R5=core['row_uncertainty'],
                       original_R5_population_uncertainty=core['population_uncertainty'],current_R4_intersection=estimates,
                       representatives=representatives,scope='Post-hoc reporting intersection of two already-frozen regions, using the original independent R5 unconditional weights and native labels. No new samples, no raw audit replay, and no whole-basin estimate.' )

def geometries(records,members):
    poses=[r['pose']for r in records];t=np.asarray([p['position']for p in poses]);q=np.asarray([p['orientation']for p in poses]);q/=np.linalg.norm(q,axis=1)[:,None]
    r=rmat(q);m=np.asarray([p['position']for p in members]);world=np.einsum('nij,kj->nki',r,m)+t[:,None,:]
    rms=cdist(world.reshape(len(records),-1)/math.sqrt(len(m)),world.reshape(len(records),-1)/math.sqrt(len(m)))
    angle=np.degrees(2*np.arccos(np.clip(np.abs(q@q.T),0.,1.)));np.fill_diagonal(angle,0.)
    center=cdist(t,t);return rms,angle,center

def cluster_diagnostics(chosen,rms,angle):
    result={};ids=np.arange(len(chosen))
    for cls in CLASSES:
        selected=ids[[r['region']==cls for r in chosen]];group={}
        for distance,rotation in [(.1,.2),(.25,.5),(.5,1.)]:
            metric=np.maximum(rms[np.ix_(selected,selected)]/distance,angle[np.ix_(selected,selected)]/rotation);np.fill_diagonal(metric,0.)
            labels=fcluster(linkage(squareform(metric,checks=True),method='complete'),1.,criterion='distance')
            components=[]
            for label in np.unique(labels):
                ii=selected[labels==label];rr=[chosen[i]for i in ii]
                by_arm={a:float(sum(r['fraction_of_arm_class']for r in rr if r['arm']==a))for a in sorted({r['arm']for r in chosen})}
                components.append(dict(size=len(ii),population_count=len({(r['arm'],r['population'])for r in rr}),arm_count=len({r['arm']for r in rr}),
                                       top_one_population_count=sum(r['rank_in_population']==1 for r in rr),selected_ids=[r['id']for r in rr],
                                       observed_arm_class_fractions=by_arm,max_pair_member_RMS_A=float(rms[np.ix_(ii,ii)].max()),max_pair_angle_deg=float(angle[np.ix_(ii,ii)].max())))
            components.sort(key=lambda c:(-c['top_one_population_count'],-c['population_count'],-c['size']))
            group[f'{distance}A_{rotation}deg']=dict(cluster_count=len(components),clusters=components,
                scope='Complete-link grouping of selected coordinates at descriptive resolution, not a free-energy basin or a connectivity/kinetic proof.')
        result[cls]=group
    return result

def annotate_geometry(chosen,reference,region,current):
    allrows=chosen+reference['representatives'];members=current['physical_metric']['rigid_members'];rms,angle,center=geometries(allrows,members);n=len(chosen)
    alternative=chart_coordinates([r['pose']for r in chosen],region);current_u=np.asarray([r['current_latent']for r in chosen]);latent_dist=cdist(current_u,current_u)
    alt_center=chart_center(region);r0,a0,c0=geometries(allrows+[dict(pose=alt_center)],members)
    for i,row in enumerate(chosen):
        row['reference_latent']=alternative[i].tolist();row['reference_radius']=float(np.linalg.norm(alternative[i]))
        row['distance_to_reference_ideal']=dict(member_RMS_A=float(r0[i,-1]),angle_deg=float(a0[i,-1]),center_A=float(c0[i,-1]))
        for name,mask in [
            ('other_population_same_class',[(r['arm'],r['population'])!=(row['arm'],row['population'])and r['region']==row['region']for r in chosen]),
            ('same_arm_other_population_same_class',[r['arm']==row['arm']and r['population']!=row['population']and r['region']==row['region']for r in chosen]),
            ('other_class',[r['region']!=row['region']for r in chosen])]:
            indices=np.flatnonzero(mask);score=np.maximum(rms[i,indices]/.25,angle[i,indices]/.5);j=int(indices[np.argmin(score)])
            row['nearest_'+name]=dict(id=chosen[j]['id'],member_RMS_A=float(rms[i,j]),angle_deg=float(angle[i,j]),center_A=float(center[i,j]),latent_distance=float(latent_dist[i,j]))
        indices=np.arange(n,len(allrows));score=np.maximum(rms[i,indices]/.25,angle[i,indices]/.5);j=int(indices[np.argmin(score)])
        row['nearest_reference_top5']=dict(id=allrows[j]['id'],member_RMS_A=float(rms[i,j]),angle_deg=float(angle[i,j]),center_A=float(center[i,j]))
    return rms[:n,:n],angle[:n,:n],center[:n,:n],cluster_diagnostics(chosen,rms[:n,:n],angle[:n,:n])

def aggregate(chosen,data):
    result={}
    for arm in data['arms']:
        result[arm]={}
        for cls in CLASSES:
            rows=[r for r in chosen if r['arm']==arm and r['region']==cls];top=[r for r in rows if r['rank_in_population']==1]
            result[arm][cls]=dict(selected_rows=len(rows),captured_observed_arm_class_fraction=sum(r['fraction_of_arm_class']for r in rows),
                selected_weight_inside_reference={str(rad):sum(r['fraction_of_arm_class']for r in rows if r['reference_radius']<=rad)for rad in [5,8,12,16,24,32]},
                top1_reference_radius=summary_values([r['reference_radius']for r in top]),
                top1_reference_member_RMS_A=summary_values([r['distance_to_reference_ideal']['member_RMS_A']for r in top]),
                top1_nearest_other_population_RMS_A=summary_values([r['nearest_other_population_same_class']['member_RMS_A']for r in top]),
                top1_nearest_same_arm_other_population_RMS_A=summary_values([r['nearest_same_arm_other_population_same_class']['member_RMS_A']for r in top]),
                top1_nearest_other_population_latent_distance=summary_values([r['nearest_other_population_same_class']['latent_distance']for r in top]),
                top1_log_cloud_difference=summary_values([r['cloud']['absolute_log_cloud_difference']for r in top]),
                top1_count_difference_standardized=summary_values([abs(r['cloud']['count_difference_standardized'])for r in top]),
                top1_z_overlap_proxy=summary_values([r['cloud']['z_times_count_estimated_overlap']for r in top]),
                top1_two_cloud_relative_SE_plugin=summary_values([r['cloud']['two_cloud_relative_SE_plugin']for r in top]),
                motif_patterns=sorted(set(tuple((m['anchor_index'],m['motif_id'])for m in r['classification']['matches'])for r in top)),
                top1_fallback_counts=sum(r['selected_ray_fallback']is True for r in top),top1_uniform_counts=sum(r['proposal_branch']=='uniform-shell'for r in top))
    return result

def plot(result,rms,out):
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    chosen=result['selected'];fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    for ax,cls in zip(axes,CLASSES):
        indices=[i for i,r in enumerate(chosen)if r['region']==cls and r['rank_in_population']==1]
        matrix=rms[np.ix_(indices,indices)];im=ax.imshow(matrix,origin='lower',cmap='viridis',vmin=0,vmax=matrix.max());fig.colorbar(im,ax=ax,label='Labeled member-center RMS distance (Å)')
        ticks=[0,4,8,12,16,20];labels=[chosen[indices[i]]['arm']for i in ticks];ax.set_xticks(ticks,labels,rotation=35,ha='right');ax.set_yticks(ticks,labels)
        ax.set_title('Top native row per population'if cls==CLASSES[0]else'Top no-entry row per population')
    fig.suptitle('Saved dominant geometries; coordinate distances are not basin barriers')
    fig.savefig(out/'dominant-pose-distances.png',dpi=160);fig.savefig(out/'dominant-pose-distances.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),constrained_layout=True)
    for cls,color in zip(CLASSES,['#2865a4','#cf6928']):
        rows=[r for r in chosen if r['region']==cls];size=[15+160*r['fraction_of_arm_class']for r in rows]
        axes[0].scatter([r['distance_to_reference_ideal']['member_RMS_A']for r in rows],[r['cloud']['z_times_count_estimated_overlap']for r in rows],s=size,alpha=.5,color=color,label=cls.replace('_',' '))
        axes[1].scatter([r['cloud']['z_times_count_estimated_overlap']for r in rows],[r['cloud']['absolute_log_cloud_difference']for r in rows],s=size,alpha=.5,color=color)
    axes[0].set_xlabel('Member RMS distance to independent native-pocket ideal (Å)');axes[0].set_ylabel('z × count-estimated overlap (geometry proxy)');axes[0].legend(fontsize=8)
    axes[1].set_xlabel('z × count-estimated overlap');axes[1].set_ylabel('|log W1 − log W2|');fig.suptitle('Existing selected poses only; size is observed within-arm weight share')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'dominant-pose-geometry-noise.png',dpi=160);fig.savefig(out/'dominant-pose-geometry-noise.svg');plt.close(fig)

def report(result):
    intersection=result['reference']['current_R4_intersection']['intersection_native'];r=intersection['row_uncertainty'];p=intersection['population_uncertainty']
    lines=['# Dominant saved poses: geometry and Poisson noise','','No new physical samples, raw-audit replay, region change, or classifier change. Existing fixed experiments remain untouched.','',
           f'The saved independent R5 population restricted to current R4 and native entry gives log Qz={r["log_Qz"]:.6f}, row RSE={r["Qz_relative_SE"]:.2%}, population RSE={p["Qz_relative_SE"]:.2%}, ESS={r["Qz_ESS"]:.1f}. This is a post-hoc intersection diagnostic using every original denominator; it is not the whole R4 weight.','',
           '![Distances between dominant saved poses](dominant-pose-distances.png)','',
           '![Geometry versus cloud noise](dominant-pose-geometry-noise.png)','',
           '| Arm | Class | Captured observed mass | Top1 nearest other-population RMS: median / max Å | Top1 native-reference radius: min / max | Top1 log-cloud discrepancy: median / max |',
           '|---|---|---:|---:|---:|---:|']
    for arm,groups in result['aggregate'].items():
        for cls,g in groups.items():
            d=g['top1_nearest_other_population_RMS_A'];a=g['top1_reference_radius'];c=g['top1_log_cloud_difference']
            lines.append(f'| {arm} | {cls} | {g["captured_observed_arm_class_fraction"]:.2%} | {d["median"]:.3f} / {d["max"]:.3f} | {a["min"]:.2f} / {a["max"]:.2f} | {c["median"]:.3f} / {c["max"]:.3f} |')
    lines+=['','Nearest neighbors are selected by max(member RMS / .25 Å, proper angular difference / .5°). The candidate bank contains the top10 rows of each class in each population; it is not an equilibrium sample. Complete-link groups at three resolutions are descriptive geometry, not demonstrated basins.','',
            '| Arm/population | Class | Top draw | Arm mass share | Raw / overlap point pairs | max member errors A7 / B4 Å | Native-reference radius |',
            '|---|---|---:|---:|---|---:|---:|']
    for row in result['selected']:
        if row['arm']not in ('large_uniform','large_ray')or row['rank_in_population']!=1:continue
        c=row['cloud'];e=row['body_errors']
        lines.append(f'| {row["arm"]}/{row["population"]} | {row["region"]} | {row["draw"]} | {row["fraction_of_arm_class"]:.2%} | {c["raw_points"]} / {c["overlap_points"]} | {e[0]["maximum_member_error_A"]:.4f} / {e[1]["maximum_member_error_A"]:.4f} | {row["reference_radius"]:.2f} |')
    lines+=['','All480 selected rows, pairwise distances, count/noise diagnostics, per-population concentration of weight, and reference intersection estimates are archived. The count-based overlap proxy does not replace exp(zC) or its unbiased estimator. Extreme-row selection biases cloud-noise summaries. Failure to sample a neighborhood remains a proposal/coverage result, not a negative physical verdict.']
    return '\n'.join(lines)+'\n'

def main(comparison,reference,out,k=10):
    comparison,reference,out=[Path(p).resolve()for p in (comparison,reference,out)];require(not out.exists(),'Fresh diagnostic output required');require(k==10,'Fixed diagnostic top10 allocation')
    data,protocol,region,chosen,coverage,bindings=select_current(comparison,out,k)
    alt,ref=reference_intersection(reference,region,bindings)
    require(data['native_definition']['definition_sha256']==read(reference/'analysis.json')['native_definition']['definition_sha256'],'Native classifier differs')
    rms,angle,center,clusters=annotate_geometry(chosen,ref,alt,region)
    out.mkdir(parents=True);np.savez_compressed(out/'pairwise-distances.npz',member_RMS_A=rms,angle_deg=angle,center_distance_A=center,ids=np.asarray([r['id']for r in chosen]))
    result=dict(schema='conditional-ray-extreme-diagnostic-v1',complete=True,new_physical_samples=0,raw_audits_rerun=0,
                comparison=str(comparison),comparison_sha256=sha(comparison/'analysis.json'),region_sha256=data['region_sha256'],native_definition_sha256=data['native_definition']['definition_sha256'],
                selected_per_population_class=k,selected=chosen,coverage=coverage,aggregate=aggregate(chosen,data),clusters=clusters,reference=ref,
                pairwise_distances_sha256=sha(out/'pairwise-distances.npz'),source_sha256=bindings,
                scope='Post-hoc saved-row geometry/noise diagnostics. No pooling across proposal arms, no equilibrium basin claims, and no new physical samples.')
    plot(result,rms,out);write(out/'analysis.json',result);(out/'report.md').write_text(report(result));shutil.copy2(__file__,out/Path(__file__).name)
    write(out/'freeze.json',{p.name:sha(p)for p in out.iterdir()if p.is_file()})
    print(json.dumps(dict(out=str(out),complete=True,analysis_sha256=sha(out/'analysis.json'),reference_intersection=ref['current_R4_intersection']['intersection_native']['row_uncertainty'])),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.comparison,a.reference,a.out)
