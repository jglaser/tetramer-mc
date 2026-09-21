#!/usr/bin/env python3
"""Classify completed finite shoulder-reference rows, without sampling or raw audits.

The unchanged base146-at-B R4 is a physical region, not a Gaussian target.
Native entry and exclusion contact are stateless reporting masks. Every invalid
or nonmember draw remains zero in its original unconditional denominator.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from analyze_mobile_native_pocket import (
    require,read,sha,write,inside,load_classifier,local_sources,summarize,label_keys,target_equal)
from analyze_mobile_competing_reference import paired_moments,check_estimate
from analyze_mobile_full_capture import chart_radii
from analyze_mobile_wall_contacts import (
    PRIMARY,summarize_regions,reference_target,read_weights as wall_weights,
    validate_parent as validate_wall_parent)

ROOT=Path(__file__).resolve().parents[1]
REGION_SHA='924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02'
DEFINITION_SHA='5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'
BINARY_SHA='55fd708b5c58477be516f710d78abbe371ede05e8b621e7e3ddd4ac86ebe8b55'
BUNDLE_SHA='033d3776f2b49fbb66694b52334c284ae8bdbd95c3d310ceb7ba98bbaa18b302'
WALL_COMPARISON_SHA='b357d024e7f395dc0ec4538f82078293a88a9c7f4e271b10a1b1bf82c5f7d326'
SAMPLES=16384
SEEDS=[128101010+1009*j for j in range(4)]
SCOPE=('Independent finite base146-at-B R4 reference, conditional on the exact observed two-body scaffold and bath. '
       'Registered native entry, exclusion contact without entry, and unbound without entry partition all hard-valid poses. '
       'No-entry is a threshold/catalogue label, not a distinct competing basin. Original q and alternative R32 are reporting masks only. '
       'Uniform latent-volume weights retain all attempted draws, including hard-invalid zeros. '
       'Fresh reference estimates and exploratory global discovery estimates are never pooled. '
       'Observed row errors, independent-population dispersion and paired-cloud noise do not establish tail coverage or equilibrium.')


class ExclusionContact:
    """Exact variable-radius atom-union gap via independent Euclidean KD trees.

    A nearest-center pair supplies an upper bound g on the minimum surface
    gap. A better pair must have distance <= moving_radius+max_fixed_radius+g;
    all such candidates are enumerated before taking direct Euclidean gaps.
    Depletion exclusion spheres overlap iff the minimum gap is < 2*rd.
    """
    def __init__(self,shape,fixed,depletant_radius):
        self.centers=np.asarray([a['center']for a in shape['atoms']],float)
        self.radii=np.asarray([a['radius']for a in shape['atoms']],float)
        require(self.centers.shape==(len(self.radii),3) and len(self.radii)>0
                and np.isfinite(self.centers).all() and np.isfinite(self.radii).all()
                and (self.radii>0).all(),'Invalid atomic union')
        require(math.isfinite(depletant_radius) and depletant_radius>0,'Invalid depletion radius')
        self.rd=depletant_radius
        self.fixed=[self.placed(p)for p in fixed]
        self.trees=[cKDTree(p)for p in self.fixed]
    def placed(self,pose):
        rotation=Rotation.from_quat(np.asarray(pose['orientation'])[[1,2,3,0]]).as_matrix()
        return self.centers@rotation.T+pose['position']
    def minimum_gap(self,points,index):
        fixed,tree,r=self.fixed[index],self.trees[index],self.radii
        d,j=tree.query(points,k=1)
        initial=d-r-r[j];i=int(np.argmin(initial));best=float(initial[i]);witness=[i,int(j[i])]
        # Cushion enlarges only the candidate list, not the final predicate.
        candidates=tree.query_ball_point(points,np.maximum(0.,r+r.max()+best+1e-10))
        for i,indices in enumerate(candidates):
            if not indices:continue
            indices=np.asarray(indices,int)
            gaps=np.linalg.norm(fixed[indices]-points[i],axis=1)-r[i]-r[indices]
            k=int(np.argmin(gaps))
            if gaps[k]<best:best=float(gaps[k]);witness=[i,int(indices[k])]
        return dict(anchor_index=index,minimum_surface_gap_A=best,moving_fixed_atom_indices=witness,
                    exclusion_contact=bool(best<2*self.rd))
    def classify(self,pose,require_hard_valid=True):
        points=self.placed(pose)
        gaps=[self.minimum_gap(points,i)for i in range(len(self.fixed))]
        if require_hard_valid:
            require(all(g['minimum_surface_gap_A']>=-1e-8 for g in gaps),
                    'Saved contributing hard-valid flag disagrees with independent contact geometry')
        return dict(exclusion_contact=any(g['exclusion_contact']for g in gaps),anchors=gaps,
                    contact_surface_gap_threshold_A=2*self.rd,
                    near_zero_negative_gap=any(-1e-8<=g['minimum_surface_gap_A']<0 for g in gaps))


def partition_masks(valid,contact,native,anchors,triangle,q,alternative_radius):
    valid=np.asarray(valid,bool)
    arrays=[np.asarray(v)for v in (contact,native,anchors,triangle,q,alternative_radius)]
    require(valid.ndim==1 and all(v.shape==valid.shape for v in arrays),'Partition shape differs')
    contact,native,anchors,triangle,q,rho=arrays
    contact,native,triangle=[v.astype(bool)for v in (contact,native,triangle)]
    require(np.isfinite(q[valid]).all() and not np.isnan(rho[valid]).any(),'Invalid diagnostic coordinates')
    require(np.all(np.isin(anchors,[0,1,2])) and np.array_equal(native[valid],anchors[valid]>0),'Native labels disagree')
    require(not np.any(valid & triangle & (anchors!=2)),'Registry triangle needs both anchors')
    masks=dict(total=valid,registered_native_entry=valid & native,
        contact_no_native_entry=valid & contact & ~native,unbound_no_native_entry=valid & ~contact & ~native,
        exclusion_contact=valid & contact,unbound=valid & ~contact,native_entry_unbound=valid & native & ~contact,
        native_one_anchor=valid & (anchors==1),native_both_anchors=valid & (anchors==2),registry_triangle=valid & triangle,
        original_q_le_1=valid & (q<=1.),original_q_gt_1=valid & (q>1.),
        alternative_native_r32=valid & (q>1.) & (rho<=32.),outside_alternative_native_r32=valid & ~((q>1.) & (rho<=32.)))
    require(np.array_equal(sum(masks[k].astype(int)for k in PRIMARY),valid.astype(int)),'Primary partition has gap/overlap')
    return masks


def validate_reference(config,region,shape_sha,source_config,source_model):
    """Check both physical target and unchanged component chart, including B anchor."""
    normalized=copy.deepcopy(config);normalized['shape']=source_config['shape']
    require(normalized==source_config,'Config differs beyond shape relocation')
    target_equal(config,region,shape_sha)
    require(region['fixed_neighbor']==config['fixed_poses'][1],'Reference chart must use exact B anchor')
    require(region.get('minimum_mahalanobis_radius',0.)==0. and region['mahalanobis_radius']==4.,'Expected complete R4 ball')
    require(region['minimum_original_q']==0. and region.get('minimum_original_q_inclusive') is True
            and 'maximum_original_q' not in region,'Physical reference must not apply q/native entry filter')
    require(config['capture_center']==[0.,0.,0.] and config['capture_radius']==170.
            and config['reservoir_density']==.035 and config['depletant_radius']==1.5,'Physical bath/domain differs')
    base=source_model['base_model'];chart=region['gaussian_chart']
    expected={k:copy.deepcopy(base[k])for k in ('schema','coordinate_convention','angular_length','shape_sha256')}
    expected.update(weights=[1.],**{k:[copy.deepcopy(base[k][146])]for k in ('anchors','means','covariances')})
    require(chart==expected,'Reference chart differs from unchanged ordinary base146')
    return reference_target(config,region,shape_sha,4.)


def validate_classifier_target(config,shape_sha,definition,definition_path):
    require(definition['shape_sha256']==shape_sha and definition['fixed_poses']==config['fixed_poses'],
            'Native classifier scaffold/shape differs')
    source=read(Path(definition_path).parent/'inputs/physical-config.json')
    normalized=copy.deepcopy(config);normalized['shape']=source['shape']
    require(normalized==source,'Native classifier physical config/bath/capture differs')
    require(sha(Path(definition_path).parent/'inputs/physical-config.json')==definition['physical_config_sha256'],
            'Native physical source binding differs')


def read_weights(rows,samples,region):
    require(len(rows)==samples and [r['draw']for r in rows]==list(range(samples)),'Unconditional draws missing/repeated')
    z,h,pairs=[],[],[]
    for row in rows:
        require(row['pose'] is not None and all(type(row[k]) is bool for k in ('hard_valid','region_valid','capture_valid')),'Missing pose/support flag')
        require(math.isfinite(row['latent_radius']) and 0.<=row['latent_radius']<=region['mahalanobis_radius']*(1+1e-12),'Draw outside R4')
        require(math.isfinite(row['q']) and row['q']>=0 and row['region_valid'] is True,'Unexpected q filter in physical reference')
        require(row['capture_valid'],'R4 enclosure contradicted by saved capture flag')
        if not row['hard_valid']:
            require(row['log_hard_weight'] is None and row['log_importance_weight'] is None and not row['clouds'],'Invalid draw must retain zero')
            z.append(-np.inf);h.append(-np.inf);pairs.append([-np.inf,-np.inf]);continue
        require(len(row['clouds'])==2 and math.isfinite(row['log_hard_weight']) and math.isfinite(row['log_importance_weight']),'Invalid two-cloud weight')
        pair=[row['log_hard_weight']+c['log_weight']for c in row['clouds']]
        require(all(math.isfinite(x)for x in pair) and abs(float(logsumexp(pair)-math.log(2))-row['log_importance_weight'])<2e-10,'Saved cloud mean differs')
        z.append(row['log_importance_weight']);h.append(row['log_hard_weight']);pairs.append(pair)
    return dict(z=np.asarray(z),h=np.asarray(h),pairs=np.asarray(pairs))


def validate_jobs(jobs,terminal):
    require(len(jobs)==len(terminal)==4 and {j['id']for j in jobs}=={f'r{j:02d}'for j in range(4)}
            and {j['id']for j in terminal}=={j['id']for j in jobs},'Missing/repeated population')
    for job in jobs:
        index=int(job['id'][1:]);term=next(t for t in terminal if t['id']==job['id'])
        require(job['seed']==SEEDS[index] and job['samples']==SAMPLES,'Fresh seed or fixed N changed')
        require(term['status']=='complete' and term['returncode']==0
                and all(term[k]==job[k]for k in ('id','seed','samples')),'Population not complete')


def validate_campaign(root):
    protocol,status=read(root/'protocol.json'),read(root/'status.json')
    require(protocol['schema']=='latent-reference-controller-v1' and status['schema']=='latent-reference-status-v1','Unexpected reference controller/status')
    require(status['complete'] and status['phase']=='complete' and not status.get('running',False),'Reference campaign incomplete')
    require(status['protocol_sha256']==sha(root/'protocol.json'),'Terminal protocol binding changed')
    for name,digest in read(root/'freeze.json')['files'].items():
        require(sha(inside(root,name))==digest,'Frozen reference input changed: '+name)
    require(protocol['manifest_sha256']==sha(root/'manifest.json'),'Campaign manifest changed')
    require(protocol['samples_per_population']==SAMPLES and protocol['populations']==4
            and protocol['seeds']==SEEDS and protocol['seed_base']==SEEDS[0]
            and protocol['total_unconditional_draws']==4*SAMPLES,'Independent fixed-N design changed')
    require(protocol['physical_executable_sha256']==BINARY_SHA and protocol['source_bundle_sha256']==BUNDLE_SHA
            and protocol['cloud_replicates']==2 and protocol['lambda_ratio']==64. and protocol['chart_anchor_index']==1,'Reviewed law/source changed')
    manifest=read(root/'manifest.json')
    require(manifest['schema']=='uniform-latent-region-campaign-v1' and 'importance_guide_sha256' not in manifest
            and 'importance-guide.json' not in manifest['archive_sha256'],'Expected uniform latent law without guide')
    for name,digest in manifest['archive_sha256'].items():
        require(sha(inside(root/'provenance',name))==digest,'Archived reference input changed: '+name)
    require(manifest['region_sha256']==protocol['source_region_sha256']==sha(root/'provenance/region.json')==REGION_SHA,'Frozen R4 identity changed')
    require(manifest['config_sha256']==sha(root/'provenance/config.json') and manifest['shape_sha256']==sha(root/'provenance/shape.json'),'Config/shape binding changed')
    require(manifest['archive_sha256']['latent-region-normalizer']==BINARY_SHA
            and manifest['archive_sha256']['source-bundle.json']==BUNDLE_SHA,'Executable/source differs')
    require(manifest['cloud_replicates']==2 and manifest['lambda_ratio']==64. and manifest['physical_activity']==.035,'Bath law changed')
    validate_jobs(manifest['jobs'],status['jobs'])
    package=root/'provenance/reference-package';p=read(package/'plan.json');record=protocol['reference_package']
    require(sha(package/'plan.json')==record['plan_sha256'] and sha(package/'freeze.json')==record['freeze_sha256'],'Reference preparation changed')
    for name,digest in read(package/'freeze.json')['files'].items():require(sha(inside(package,name))==digest,'Prepared input changed')
    require(sha(root/'provenance/source-config.json')==protocol['source_config_sha256']==p['config_sha256'],'Supplied prepared config changed')
    require(p['region_sha256']==manifest['region_sha256'] and p['shape_sha256']==manifest['shape_sha256'],'Prepared target differs')
    definition=inside(package,p['native_definition'])
    require(sha(definition)==record['native_definition_sha256']==p['native_definition_sha256']==DEFINITION_SHA,'Frozen native definition changed')
    config=read(root/'provenance/config.json');region=read(root/'provenance/region.json')
    upper=validate_reference(config,region,manifest['shape_sha256'],read(package/'provenance/source-config.json'),read(package/'provenance/source-model.json'))
    bound=max(float(np.linalg.norm(a['center']))+a['radius']for a in read(root/'provenance/shape.json')['atoms'])
    require(upper+bound<config['metadata']['physical_sphere_radius_A'],'Finite target not enclosed by original atomic wall')
    require(Path(config['shape']).resolve()==root/'provenance/shape.json','Wrong runtime shape path')
    validate_classifier_target(config,manifest['shape_sha256'],read(definition),definition)
    require(status['audit']['returncode']==0 and sha(root/'assessment/analysis.json')==status['audit']['analysis_sha256'],'Completed raw audit failed/changed')
    assessment=read(root/'assessment/analysis.json')
    require(assessment['region_sha256']==REGION_SHA and assessment['independently_reconstructed_poses']==4*SAMPLES
            and len(assessment['populations'])==4 and {p['id']for p in assessment['populations']}=={j['id']for j in manifest['jobs']},'Raw audit omitted rows/populations')
    return protocol,status,manifest,config,region,definition,assessment


def add_native_masks(populations):
    keys={key for p in populations for rowkeys in p['label_keys']for key in rowkeys}
    keys.update(('native_any','both_anchor_entry','registry_triangle','motif7_4_triangle'))
    for p in populations:
        for key in keys:p['masks']['label:'+key]=np.asarray([key in k for k in p['label_keys']],bool) & p['masks']['total']


def load_populations(root,manifest,status,assessment,config,region,classifier,alternative,out):
    contact=ExclusionContact(read(root/'provenance/shape.json'),config['fixed_poses'],config['depletant_radius'])
    populations,bindings,anomalies=[],{},[];cpu=0.;negative=0
    labels_path=out/'reference-labels.jsonl'
    with labels_path.open('x')as stream:
        for job in manifest['jobs']:
            print('Classifying saved finite-reference poses: '+job['id'],flush=True)
            directory=Path(job['directory']).resolve();require(directory==root/'runs'/job['id'],'Population path differs')
            term=next(t for t in status['jobs']if t['id']==job['id'])
            for name,key in [('samples.jsonl','samples_sha256'),('manifest.json','manifest_sha256'),('summary.json','summary_sha256')]:
                digest=sha(directory/name);require(digest==term['output'][key],'Terminal population changed: '+name)
                bindings[str(directory/name)]=digest
            pm,summary=read(directory/'manifest.json'),read(directory/'summary.json')
            require(summary['complete'] and summary['manifest']==pm and summary['samples']==SAMPLES
                    and pm['schema']=='uniform-latent-region-normalizer-v2','Population incomplete or wrong law')
            expected=dict(seed=job['seed'],samples=SAMPLES,cloud_replicates=2,activity=.035,lambda_ratio=64.,
                region_sha256=REGION_SHA,config_sha256=manifest['config_sha256'],shape_sha256=manifest['shape_sha256'],
                executable_sha256=BINARY_SHA,source_bundle_sha256=BUNDLE_SHA,physical_fixed_neighbors=config['fixed_poses'],
                chart_anchor=region['fixed_neighbor'],minimum_latent_radius=0.,minimum_original_q=0.,minimum_original_q_inclusive=True)
            require(all(pm.get(k)==v for k,v in expected.items()) and pm.get('maximum_original_q') is None and pm['lambda']==.035*64.,'Population physical law/region differs')
            for name,key in [('input-config.json','config_sha256'),('region.json','region_sha256'),('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
                require(sha(directory/'provenance'/name)==pm[key],'Population frozen input changed: '+name)
            audit=next(p for p in assessment['populations']if p['id']==job['id'])
            require(audit['seed']==job['seed'] and audit['samples_sha256']==summary['samples_sha256']==bindings[str(directory/'samples.jsonl')],'Audited rows/seed changed')
            rows=[json.loads(line)for line in (directory/'samples.jsonl').open()]
            arrays=read_weights(rows,SAMPLES,region);valid=np.isfinite(arrays['z']);indices=np.flatnonzero(valid)
            check_estimate(paired_moments(arrays['z'],arrays['h']),audit['estimate'],audit['hard_region'])
            require(int(valid.sum())==summary['estimates']['region']['nonzero']==summary['estimates']['hard_region']['nonzero'],'Saved contributing flag counts disagree')
            radii=np.full(SAMPLES,np.inf)
            if len(indices):radii[indices],_=chart_radii([rows[i]['pose']for i in indices],alternative)
            native=np.zeros(SAMPLES,bool);anchors=np.zeros(SAMPLES,int);triangles=np.zeros(SAMPLES,bool);contacts=np.zeros(SAMPLES,bool);keys=[]
            for i,row in enumerate(rows):
                label=classifier.classify(row['pose'])if valid[i]else None
                measured=contact.classify(row['pose'])if valid[i]else None
                keys.append(label_keys(label)if label is not None else set())
                if label is not None:
                    native[i]=label['native_any'];anchors[i]=label['native_anchor_count'];triangles[i]=label['registry_consistent_triangle']
                    contacts[i]=measured['exclusion_contact'];negative+=int(measured['near_zero_negative_gap'])
                    if native[i] and not contacts[i]:anomalies.append(dict(population=job['id'],draw=i,pose=row['pose'],classification=label,contact=measured))
                stream.write(json.dumps(dict(population=job['id'],seed=job['seed'],draw=i,applicable=bool(valid[i]),
                    classification=label,contact=measured,alternative_native_radius=float(radii[i])if valid[i]else None),separators=(',',':'),allow_nan=False)+'\n')
            masks=partition_masks(valid,contacts,native,anchors,triangles,np.array([r['q']for r in rows]),radii)
            populations.append(dict(id=job['id'],seed=job['seed'],masks=masks,label_keys=keys,**arrays))
            require(math.isfinite(summary['sampler_cpu_seconds']) and summary['sampler_cpu_seconds']>=0,'Invalid physical CPU')
            cpu+=summary['sampler_cpu_seconds']
    add_native_masks(populations)
    estimates,covariance,ratios=summarize_regions(populations)
    check_estimate(estimates['total']['row_uncertainty'],assessment['estimate'],assessment['hard_region'])
    return dict(estimates=estimates,primary_covariance=covariance,primary_ratios=ratios,
        sampler_cpu_seconds=cpu,source_sha256=bindings,native_labels=dict(path=labels_path.name,sha256=sha(labels_path)),
        native_entry_unbound_anomalies=anomalies,near_zero_negative_core_gap_count=negative,
        contact_definition='Exact minimum atom-surface gap to either fixed neighbor < 2*rd = 3 Å. Positive overlap of exclusion-sphere unions; no q/native criterion enters contact.',
        hard_flag_validation='Saved hard flags and all-row hash identities come from the completed frozen raw audit. Contributing poses additionally pass independent atom-union gap >= -1e-8 Å during contact classification; negative gaps within this floating-point audit tolerance are counted.'),populations


def load_wall_comparison(path,config,shape_sha,region,definition_sha):
    path=Path(path).resolve();require(sha(path/'analysis.json')==WALL_COMPARISON_SHA,'Completed wall comparison identity differs')
    for name,digest in read(path/'freeze.json').items():require(sha(inside(path,name))==digest,'Completed wall comparison changed: '+name)
    data=read(path/'analysis.json');require(data['complete'] and data['native_definition']['definition_sha256']==definition_sha,'Wall/native comparison differs')
    root=Path(data['campaign']).resolve();protocol,status=validate_wall_parent(root)
    require(sha(root/'protocol.json')==data['protocol_sha256'] and sha(root/'status.json')==data['status_sha256'],'Wall completed campaign binding differs')
    alternative_path=inside(root,protocol['references']['alternative_r32']);alternative=read(alternative_path)
    target_equal(config,alternative,shape_sha)
    require(alternative['mahalanobis_radius']==32. and alternative.get('minimum_mahalanobis_radius',0.)==0.
            and alternative['minimum_original_q']==1. and alternative.get('minimum_original_q_inclusive') is False,'Alternative R32 reporting region changed')
    reference_target(config,alternative,shape_sha,32.)
    for arm in data['arms']:
        source=read(root/arm['arm']/'provenance/config.json');changed=copy.deepcopy(source)
        changed['shape']=config['shape'];changed['capture_radius']=config['capture_radius']
        require(changed==config and source['capture_radius']==273. and arm['shape_sha256']==shape_sha,'Global/finite scaffold or bath differs')
    return path,data,root,alternative,dict(comparison=str(path),comparison_sha256=sha(path/'analysis.json'),
        protocol_sha256=sha(root/'protocol.json'),status_sha256=sha(root/'status.json'),
        alternative_region=str(alternative_path),alternative_region_sha256=sha(alternative_path),
        scope='Same finite R4 geometric target; old full-wall proposal laws and unconditional budgets stay separate. These discovery rows selected the reference and are exploratory, not independent confirmation.')


def exploratory_wall_subsets(path,data,root,region,alternative):
    """Reuse saved class labels and weights; no old atomic/classifier audit replay."""
    results=[]
    for arm in data['arms']:
        print('Restricting saved exploratory wall arm to the same finite R4: '+arm['arm'],flush=True)
        folder=root/arm['arm'];manifest=read(folder/'manifest.json')
        labels_path=inside(path,arm['native_labels']['path'])
        require(sha(labels_path)==arm['native_labels']['sha256'],'Saved wall labels changed')
        labels={}
        for line in labels_path.open():
            item=json.loads(line)
            if item['classification'] is not None:labels[item['population'],item['draw']]=item['classification']
        pops=[];bindings={str(labels_path):sha(labels_path)}
        for job in manifest['jobs']:
            require(job['seed'] not in SEEDS,'New reference reuses exploratory seed')
            directory=Path(job['directory']).resolve();sample=directory/'samples.jsonl'
            for name in ('samples.jsonl','manifest.json','summary.json'):
                digest=sha(directory/name);require(digest==arm['source_sha256'][str(directory/name)],'Saved global input changed')
                bindings[str(directory/name)]=digest
            rows=[json.loads(line)for line in sample.open()];arrays=wall_weights(rows,job['samples'])
            valid=np.isfinite(arrays['z']);indices=np.flatnonzero(valid)
            rho=np.full(len(rows),np.inf);ra=np.full(len(rows),np.inf)
            if len(indices):rho[indices],_=chart_radii([rows[i]['pose']for i in indices],region)
            subset=valid & (rho<=4.);selected=np.flatnonzero(subset)
            if len(selected):ra[selected],_=chart_radii([rows[i]['pose']for i in selected],alternative)
            native=np.zeros(len(rows),bool);anchors=np.zeros(len(rows),int);triangles=np.zeros(len(rows),bool);keys=[]
            for i,row in enumerate(rows):
                label=labels.get((job['id'],i))if subset[i]else None
                require(not subset[i] or label is not None,'Missing selected saved wall label')
                keys.append(label_keys(label)if label is not None else set())
                if label is not None:native[i]=label['native_any'];anchors[i]=label['native_anchor_count'];triangles[i]=label['registry_consistent_triangle']
            masks=partition_masks(subset,[r['depletion_contact'] is True for r in rows],native,anchors,triangles,
                [0. if r['q'] is None else r['q']for r in rows],ra)
            # Zero outside R4 before summarizing, retaining the original full-wall N.
            arrays={key:np.where(subset[:,None] if key=='pairs' else subset,value,-np.inf)for key,value in arrays.items()}
            pops.append(dict(id=job['id'],seed=job['seed'],masks=masks,label_keys=keys,**arrays))
        add_native_masks(pops);estimates,covariance,ratios=summarize_regions(pops)
        results.append(dict(arm=arm['arm'],estimates=estimates,primary_covariance=covariance,primary_ratios=ratios,source_sha256=bindings,
            original_unrestricted_primary_estimates={name:arm['estimates'][name]for name in PRIMARY},
            scope='Post-selection exploratory restriction of saved full-wall rows to unchanged base146-at-B R4. Original weights, native labels, exclusion-contact flags and all invalid/out-of-region zeros retained. No pooling with the fresh reference or other proposal arm.'))
    return results


def report(result):
    lines=['# Independent finite native-threshold shoulder reference','',SCOPE,'',
        '| Finite-region class | log Qz | row RSE | population RSE | log Q0 | log(Qz/Q0) ± paired row SE | ESS | max fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('total',)+PRIMARY+('native_both_anchors','registry_triangle','original_q_le_1','alternative_native_r32','outside_alternative_native_r32'):
        item=result['reference']['estimates'][name];row,pop=item['row_uncertainty'],item['population_uncertainty']
        if row['log_Qz'] is None:lines.append(f'| {name} | unobserved | — | — | — | — | — | — |');continue
        lines.append(f"| {name} | {row['log_Qz']:.6f} | {row['Qz_relative_SE']:.2%} | {pop['Qz_relative_SE']:.2%} | {row['log_Q0']:.6f} | {row['log_enhancement']:.6f} ± {row['log_enhancement_SE']:.6f} | {row['Qz_ESS']:.1f} | {row['largest_Qz_fraction']:.2%} |")
    lines+=['',f"Four independent populations × {SAMPLES:,} unconditional attempts; two clouds per contributing pose. Physical CPU: {result['reference']['sampler_cpu_seconds']:.1f} s. Invalid poses remain zero. Per-population estimates, shared-class covariance and paired-cloud variance are retained in analysis.json.",'',
        'The physical ellipsoid has no q or native/contact filter. Its center enters motifs7/4 and its surrounding shoulder can fail the 2 Å body-entry threshold. Registered entry has precedence in the exhaustive partition; any registered but unbound pose is retained and separately flagged.', '',
        f"Native-entry/unbound anomalies: {len(result['reference']['native_entry_unbound_anomalies'])}. Contributing poses with independent core gaps between −1e−8 and 0 Å: {result['reference']['near_zero_negative_core_gap_count']}.", '',
        '## Same R4 in exploratory global rows','',
        'These older rows selected the finite reference. Their original proposal denominators and full attempted N remain intact; neither global proposal law is pooled with the fresh reference. Their restricted estimates are exploratory and do not provide independent confirmation.', '',
        '| Source | log total Qz | log entry Qz | log contact/no-entry Qz | log unbound/no-entry Qz |',
        '|---|---:|---:|---:|---:|']
    for name,estimates in [('fresh uniform reference',result['reference']['estimates'])]+[(a['arm'],a['estimates'])for a in result['exploratory_wall_subsets']]:
        values=[estimates[key]['row_uncertainty']['log_Qz']for key in ('total',)+PRIMARY]
        lines.append('| '+name+' | '+' | '.join('unobserved'if x is None else f'{x:.6f}'for x in values)+' |')
    lines+=['','No observed contribution is not a physical zero or an upper bound. A finite R4 result does not determine all mass outside the native-entry catalogue or establish equilibrium contact frequencies.']
    return '\n'.join(lines)+'\n'


def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('total',)+PRIMARY
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for axis,kind in zip(axes,('Qz','Q0')):
        for offset,(title,estimates)in enumerate([('fresh uniform R4',result['reference']['estimates'])]+[(r['arm']+' exploratory',r['estimates'])for r in result['exploratory_wall_subsets']]):
            xs,ys,errors=[],[],[]
            for index,name in enumerate(names):
                r=estimates[name]['row_uncertainty']
                if r['log_'+kind] is not None:xs.append(index+.18*(offset-1));ys.append(r['log_'+kind]);errors.append(r[kind+'_relative_SE'])
            axis.errorbar(xs,ys,yerr=errors,fmt='o',capsize=3,label=title,markersize=4)
        axis.set_xticks(range(len(names)),['whole R4','native entry','contact, no entry','unbound, no entry'],rotation=20,ha='right')
        axis.set_ylabel('log '+kind+' (row delta SE)');axis.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('One fixed finite region; proposal estimates remain separate')
    fig.savefig(out/'finite-reference.png',dpi=170);plt.close(fig)


def analyze(campaign,out,wall_comparison):
    root,out=Path(campaign).resolve(),Path(out).resolve()
    require(not out.exists(),'Use fresh downstream output')
    protocol,status,manifest,config,region,definition,assessment=validate_campaign(root)
    classifier,binding=load_classifier(definition)
    wall_path,wall_data,wall_root,alternative,wall_binding=load_wall_comparison(wall_comparison,config,manifest['shape_sha256'],region,binding['definition_sha256'])
    out.mkdir(parents=True)
    try:
        reference,_=load_populations(root,manifest,status,assessment,config,region,classifier,alternative,out)
        exploratory=exploratory_wall_subsets(wall_path,wall_data,wall_root,region,alternative)
        result=dict(schema='mobile-threshold-reference-comparison-v1',complete=True,campaign=str(root),
            protocol_sha256=sha(root/'protocol.json'),status_sha256=sha(root/'status.json'),freeze_sha256=sha(root/'freeze.json'),
            manifest_sha256=sha(root/'manifest.json'),assessment_sha256=sha(root/'assessment/analysis.json'),
            region_sha256=REGION_SHA,shape_sha256=manifest['shape_sha256'],native_definition=binding,
            primary_partition=list(PRIMARY),reference=reference,exploratory_wall_subsets=exploratory,
            exploratory_binding=wall_binding,scope=SCOPE)
        sources=out/'provenance';sources.mkdir()
        for name,path in local_sources(__file__).items():shutil.copy2(path,sources/name)
        for name,path in [('finite-region.json',root/'provenance/region.json'),('alternative-r32.json',Path(wall_binding['alternative_region'])),
                          ('campaign-protocol.json',root/'protocol.json'),('campaign-manifest.json',root/'manifest.json')]:shutil.copy2(path,sources/name)
        result['analyzer_source_sha256']={p.name:sha(p)for p in sources.iterdir()}
        write(out/'analysis.json',result);(out/'report.md').write_text(report(result));plot(result,out)
        write(out/'freeze.json',{p.relative_to(out).as_posix():sha(p)for p in sorted(out.rglob('*'))if p.is_file()})
        print(json.dumps(dict(complete=True,out=str(out),analysis_sha256=sha(out/'analysis.json'))),flush=True)
        return result
    except Exception as error:
        write(out/'failure.json',dict(complete=False,error=str(error),scope='Saved-row downstream analysis only; no sampler or raw audit invoked. Partial outputs preserved.'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--wall-comparison',type=Path,default=ROOT/'runs/mobile-wall-contact-comparison-20260921')
    args=parser.parse_args();analyze(args.campaign,args.out,args.wall_comparison)
