#!/usr/bin/env python3
"""Read saved audited importance rows, identify native sites, check atomic geometry.

No sampling, bath-weight evaluation, density re-evaluation or audit is performed.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=ROOT/'runs/mobile-full-capture-campaign-20260921'
COMPARISON=ROOT/'runs/mobile-full-capture-comparison-20260921'
SCOPE=('Post hoc geometry and saved-row contribution diagnostics, conditional on the observed two-tetramer scaffold. '
       'All density, importance-weight and two-cloud values are copied unchanged. A realized cloud factor is not a pose free energy. '
       'Neither largest-weight pose selection nor observed contribution fractions estimate basin free energies or establish convergence. '
       'Native registry is the frozen instantaneous entry catalogue, not global crystal certification. '
       'Candidate references are for new independent finite-region tests only; these discovery rows must not be reused as independent validation.')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(ok,msg):
    if not ok:raise ValueError(msg)
def rt(p):return np.asarray(p['position']),Rotation.from_quat(np.asarray(p['orientation'])[[1,2,3,0]]).as_matrix()
def pose(t,r):return dict(position=np.asarray(t).tolist(),orientation=Rotation.from_matrix(r).as_quat()[[3,0,1,2]].tolist())
def compose(a,b):
    t,r=rt(a);u,s=rt(b);return pose(t+r@u,r@s)
def inverse(p):
    t,r=rt(p);return pose(-r.T@t,r.T)
def relative(p,a):return compose(inverse(a),p)
def differences(a,b,members):
    t,r=rt(a);u,s=rt(b);d=members@r.T+t-(members@s.T+u);length=np.linalg.norm(d,axis=1)
    return dict(center_distance_A=float(np.linalg.norm(t-u)),member_rms_A=float(np.sqrt(np.mean(length**2))),
        maximum_member_distance_A=float(length.max()),orientation_degrees=float(np.degrees(Rotation.from_matrix(r.T@s).magnitude())))
def native_q(p,metric,members):
    return min(max(differences(p,ref,members)['maximum_member_distance_A']/metric['member_error_scale'],
        differences(p,ref,members)['orientation_degrees']/metric['angle_error_scale_deg'])for ref in metric['native_poses'])
def radius(p,region):
    model=region['gaussian_chart'];a=model['anchors'][0];t,r=rt(relative(p,region['fixed_neighbor']))
    q=Rotation.from_matrix(r@np.array(a['rotation']).T).as_quat()
    if q[3]==0:return dict(radius=None,exact_seam=True)
    x=np.r_[t-a['position'],model['angular_length']*q[:3]/q[3]]-model['means'][0]
    z=solve_triangular(np.linalg.cholesky(model['covariances'][0]),x,lower=True)
    v=float(np.hypot.reduce(z));require(np.isfinite(v),'Non-seam radius inverse failed')
    return dict(radius=v,exact_seam=False)
def site_key(classification):
    matches=classification.get('matches',[])
    return '|'.join(f"body{body}:"+','.join(map(str,sorted({m['motif_id']for m in matches if m['anchor_body_id']==body})))
        for body in (2,1)) if matches else 'no_native_entry'

class AtomicGeometry:
    """Independent exact minimum of variable-radius sphere surface distances.

    A nearest-center witness supplies a surface-gap upper bound g. Any better
    pair must have distance <= r_moving+r_fixed_max+g. Exhaustive radius queries
    enumerate every such pair, then direct Euclidean distances find the minimum.
    No rigid-body BVH or Rust overlap predicate is reused.
    """
    def __init__(self,shape,fixed,wall):
        self.centers=np.array([a['center']for a in shape['atoms']]);self.radii=np.array([a['radius']for a in shape['atoms']])
        self.fixed=[self.placed(p)for p in fixed];self.trees=[cKDTree(p)for p in self.fixed];self.wall=wall
    def placed(self,p):
        t,r=rt(p);return self.centers@r.T+t
    def gap(self,points,index):
        fixed=self.fixed[index];tree=self.trees[index];r=self.radii
        d,j=tree.query(points,k=1);g=d-r-r[j];i=int(np.argmin(g));best=float(g[i]);witness=[i,int(j[i])]
        # The numerical cushion expands the search, never changes the measured gap.
        bounds=np.maximum(0.,r+r.max()+best+1e-10);near=tree.query_ball_point(points,bounds);pairs=0
        for i,js in enumerate(near):
            if not js:continue
            js=np.asarray(js,dtype=int);gaps=np.linalg.norm(fixed[js]-points[i],axis=1)-r[i]-r[js];pairs+=len(js)
            k=int(np.argmin(gaps))
            if gaps[k]<best:best=float(gaps[k]);witness=[i,int(js[k])]
        return dict(minimum_surface_gap_A=best,moving_fixed_atom_indices=witness,exact_candidate_pairs=int(pairs))
    def check(self,p):
        points=self.placed(p);clearance=self.wall['radius']-np.linalg.norm(points-self.wall['center'],axis=1)-self.radii
        gaps=[dict(anchor_index=i,fixed_body_id=(2,1)[i],**self.gap(points,i))for i in range(2)]
        return dict(atom_count=len(points),fixed_neighbor_gaps=gaps,
            hard_valid_at_zero_tolerance=all(g['minimum_surface_gap_A']>=0 for g in gaps),
            original_atomic_wall_clearance_A=float(clearance.min()),limiting_wall_atom=int(np.argmin(clearance)),
            original_atomic_wall_valid=bool(np.all(clearance>=0)),wall=self.wall)


def run(out,campaign=CAMPAIGN,comparison=COMPARISON):
    out=Path(out).resolve();campaign=Path(campaign).resolve();comparison=Path(comparison).resolve()
    require(not out.exists(),'Use a fresh diagnostic output directory')
    comparison_freeze=read(comparison/'freeze.json')
    for name,digest in comparison_freeze.items():require(sha(comparison/name)==digest,'Saved comparison changed: '+name)
    analysis=read(comparison/'analysis.json');require(analysis['complete'],'Completed saved comparison required')
    protocol=read(campaign/'protocol.json');require(sha(campaign/'protocol.json')==analysis['protocol_sha256'],'Protocol differs')
    definition_path=campaign/protocol['partition']['native_definition'];definition=read(definition_path)
    require(sha(definition_path)==analysis['native_classifier']['definition_sha256'],'Native definition differs')
    inputs=definition_path.parent/'inputs'
    for name,digest in definition['input_sha256'].items():require(sha(inputs/name)==digest,'Native input changed: '+name)
    cfg=read(inputs/'physical-config.json');shape=read(inputs/'tetramer-shape.json');snapshot=read(inputs/'fixed-snapshot.json')
    members=np.array([p['position']for p in cfg['metadata']['rigid_members']]);fixed=cfg['fixed_poses']
    wall=dict(center=snapshot['original_global']['wall_center'],radius=snapshot['original_global']['wall_radius'])
    geometry=AtomicGeometry(shape,fixed,wall)
    regions={name:read(campaign/'provenance/partition'/filename)for name,filename in [('native','region-native-r4.json'),('competitor','region-competitor-r3.json')]}
    motifs={m['id']:m for m in read(inputs/'native-pair-motifs.json')['motifs']}
    motifpose={k:dict(position=m['relative_position'],orientation=m['relative_orientation'])for k,m in motifs.items()}
    original=cfg['metadata']['native_poses'][0];ab=snapshot['ab_gauge'];map_pose=dict(position=ab['common_translation'],orientation=Rotation.from_matrix(ab['common_rotation']).as_quat()[[3,0,1,2]].tolist())
    out.mkdir();shutil.copy2(__file__,out/'diagnostic.py')
    hashes={str(comparison/name):sha(comparison/name)for name in ('analysis.json','freeze.json')}
    hashes.update({str(definition_path):sha(definition_path),str(campaign/'protocol.json'):sha(campaign/'protocol.json')})
    top_records=[];populations=[];site_examples={};global_selected=[];site_groups={};raw_file=out/'top16-per-population-saved-rows.jsonl'
    with raw_file.open('x')as stream:
        for arm in analysis['arms']:
            label_path=comparison/f"{arm['arm']}-native-labels.jsonl";require(sha(label_path)==arm['native_labels_sha256'],'Labels differ')
            hashes[str(label_path)]=sha(label_path)
            labels={(v['population'],v['draw']):v['classification']for v in map(json.loads,label_path.read_text().splitlines())}
            armgroups=defaultdict(list);armtotal=[]
            for population in arm['populations']:
                pid=population['id'];directory=campaign/arm['arm']/'runs'/pid;path=directory/'samples.jsonl'
                require(sha(path)==arm['source_sha256'][str(path)],'Audited saved sample bytes differ');hashes[str(path)]=sha(path)
                rows=list(map(json.loads,path.read_text().splitlines()));require(len(rows)==8192,'Fixed draw allocation differs')
                valid=[r for r in rows if r['log_importance_weight']is not None]
                require(all(r['hard_valid']for r in valid),'Nonzero invalid row')
                total=float(logsumexp([r['log_importance_weight']for r in valid]));selected=sorted(valid,key=lambda r:r['log_importance_weight'],reverse=True)[:16]
                popgroups=defaultdict(list)
                for row in valid:
                    classification=labels.get((pid,row['draw']),{});key=site_key(classification)
                    popgroups[key].append(row['log_importance_weight']);armgroups[key].append(row['log_importance_weight']);armtotal.append(row['log_importance_weight'])
                summaries=[]
                for rank,row in enumerate(selected,1):
                    label=labels.get((pid,row['draw']),{});key=site_key(label);site_examples.setdefault(key,label)
                    identity=dict(arm=arm['arm'],population=pid,seed=population['seed'],draw=row['draw'])
                    stream.write(json.dumps(dict(**identity,rank_in_population=rank,sample=row),allow_nan=False)+'\n')
                    q=native_q(row['pose'],cfg['metadata'],members);require(abs(q-row['q'])<2e-8,'Independent native metric differs')
                    clouds=[dict(c,conditional_log_second_moment_upper=cfg['reservoir_density']**2*c['uncertain_volume']/(cfg['reservoir_density']*cfg['poisson_lambda_ratio']))for c in row['clouds']]
                    record=dict(**identity,rank_in_population=rank,site_key=key,pose=row['pose'],pose_ab_gauge=compose(map_pose,row['pose']),
                        recorded_log_importance_weight=row['log_importance_weight'],recorded_log_proposal_density=row['log_proposal_density'],
                        recorded_log_hard_weight=row['log_hard_weight'],recorded_proposal=row['proposal'],recorded_clouds=clouds,
                        cloud_log_factor_difference=row['clouds'][0]['log_weight']-row['clouds'][1]['log_weight'],
                        observed_fraction_of_population_total=float(np.exp(row['log_importance_weight']-total)),
                        original_q=row['q'],independent_q=q,original_native_chart=radius(row['pose'],regions['native']),
                        competitor_chart=radius(row['pose'],regions['competitor']),saved_native_classification=label,
                        difference_from_original_native=differences(row['pose'],original,members))
                    if rank<=2:
                        record['independent_atomic_geometry']=geometry.check(row['pose'])
                        require(record['independent_atomic_geometry']['hard_valid_at_zero_tolerance'],'Saved top contributor has atomic clash')
                        require(record['independent_atomic_geometry']['original_atomic_wall_valid'],'Saved top contributor leaves original wall')
                    top_records.append(record);summaries.append(dict(draw=row['draw'],site_key=key,recorded_log_importance_weight=row['log_importance_weight']))
                    global_selected.append(record)
                groups=[dict(site_key=key,nonzero_rows=len(values),observed_fraction_of_population_total=float(np.exp(logsumexp(values)-total)))for key,values in popgroups.items()]
                populations.append(dict(arm=arm['arm'],id=pid,seed=population['seed'],nonzero_rows=len(valid),top16=summaries,
                    groups=sorted(groups,key=lambda g:g['observed_fraction_of_population_total'],reverse=True)))
            arm_sum=logsumexp(armtotal)
            site_groups[arm['arm']]=sorted([dict(site_key=k,nonzero_rows=len(v),observed_fraction_of_arm_total=float(np.exp(logsumexp(v)-arm_sum)))for k,v in armgroups.items()],key=lambda x:x['observed_fraction_of_arm_total'],reverse=True)
    sites=[]
    for key,label in site_examples.items():
        if key=='no_native_entry':continue
        ideal=[]
        for match in label['matches']:
            i=match['anchor_index'];k=match['motif_id'];p=compose(fixed[i],motifpose[k])
            ideal.append(dict(anchor_index=i,anchor_body_id=(2,1)[i],motif_id=k,pose=p,pose_ab_gauge=compose(map_pose,p),
                difference_from_original_native=differences(p,original,members),original_q=native_q(p,cfg['metadata'],members),
                independent_atomic_geometry=geometry.check(p),prescribed_bonds=[{f:c[f]for f in ('member_i','member_j','family','directed_class')}for c in motifs[k]['member_contacts']]))
        comparisons=[dict(first=[a['anchor_body_id'],a['motif_id']],second=[b['anchor_body_id'],b['motif_id']],
            **differences(a['pose'],b['pose'],members))for i,a in enumerate(ideal)for b in ideal[i+1:]]
        cycles=[]
        for a in ideal:
            for b in ideal:
                if a['anchor_index']!=0 or b['anchor_index']!=1:continue
                implied=compose(motifpose[a['motif_id']],inverse(motifpose[b['motif_id']]))
                errors=[(differences(implied,p,members)['maximum_member_distance_A'],k,differences(implied,p,members))for k,p in motifpose.items()]
                _,k,error=min(errors,key=lambda x:x[0]);cycles.append(dict(anchor0_to_moving_motif=a['motif_id'],anchor1_to_moving_motif=b['motif_id'],
                    implied_anchor0_to_anchor1_motif=k,catalogue_closure=error,
                    observed_fixed_scaffold_error=differences(relative(fixed[1],fixed[0]),motifpose[k],members)))
        sites.append(dict(site_key=key,ideal_catalogue_poses=ideal,observed_scaffold_composition_differences=comparisons,independent_catalogue_cycles=cycles))
    originalmatches=[]
    for i,f in enumerate(fixed):
        d=relative(original,f);errors=[(differences(d,p,members)['maximum_member_distance_A'],k,differences(d,p,members))for k,p in motifpose.items()]
        _,k,e=min(errors,key=lambda x:x[0]);originalmatches.append(dict(anchor_body_id=(2,1)[i],closest_motif_id=k,**e))
    main='body2:7|body1:4';require(main in site_examples,'Expected dominant discovered site absent; inspect before choosing reference')
    model=read(campaign/'epsilon-0p1/provenance/model.json')['base_model'];k=19;candidate_model={
        'schema':'weighted-pose-mixture-v1','angular_length':model['angular_length'],'coordinate_convention':'anchor-body-relative',
        'shape_sha256':model['shape_sha256'],'anchors':[model['anchors'][k]],'means':[model['means'][k]],
        'covariances':[model['covariances'][k]],'weights':[1.]}
    candidate=dict(status='PROPOSED finite-reference chart, not an integration result or launched campaign',site_key=main,
        fixed_neighbor=fixed[0],physical_fixed_neighbors=fixed,capture_center=cfg['capture_center'],capture_radius=cfg['capture_radius'],
        shape_sha256=definition['shape_sha256'],activity=cfg['reservoir_density'],depletant_radius=cfg['depletant_radius'],
        physical_metric=cfg['metadata'],gaussian_chart=candidate_model,
        provenance=dict(source_model_sha256=sha(campaign/'epsilon-0p1/provenance/model.json'),base_component_index=19,inverted=False,
            source='Preexisting frozen atlas component19, selected after discovery; no refitting of its covariance or mean.',
            note='Choose and freeze finite latent support separately; independently sample it and retain all hard-invalid zeros. Do not extrapolate it to the whole site.'))
    write(out/'candidate-existing-component19.json',candidate)
    globaltop=sorted(global_selected,key=lambda r:r['recorded_log_importance_weight'],reverse=True)[:16]
    result=dict(schema='mobile-native-remainder-diagnostic-v1',complete=True,physical_jobs_launched=0,bath_weight_queries=0,audits_rerun=0,
        scope=SCOPE,source_sha256=hashes,selected_rows=top_records,populations=populations,observed_site_groups=site_groups,sites=sites,
        original_native_pose=original,original_native_ideal_motif_matches=originalmatches,
        global_top16=[{k:r[k]for k in ('arm','population','seed','draw','site_key','recorded_log_importance_weight')}for r in globaltop],
        atomic_geometry_checks=sum('independent_atomic_geometry'in r for r in top_records),
        atomic_check_method=AtomicGeometry.__doc__,candidate_reference='candidate-existing-component19.json',
        unchanged_saved_rows_sha256=sha(raw_file),diagnostic_sha256=sha(__file__))
    write(out/'analysis.json',result)
    lines=['# Saved native remainder geometry','',SCOPE,'',
        '| Arm | Population | Top draw | Native motif site | Saved log importance weight | Minimum atom gap / Å | Wall clearance / Å |',
        '|---|---|---:|---|---:|---:|---:|']
    for r in top_records:
        if r['rank_in_population']!=1:continue
        g=r['independent_atomic_geometry'];gap=min(v['minimum_surface_gap_A']for v in g['fixed_neighbor_gaps'])
        lines.append(f"| {r['arm']} | {r['population']} | {r['draw']} | {r['site_key']} | {r['recorded_log_importance_weight']:.6f} | {gap:.6g} | {g['original_atomic_wall_clearance_A']:.5f} |")
    lines+=['','Top16 per population are preserved exactly in `top16-per-population-saved-rows.jsonl`. All detailed saved cloud realizations, motif contacts and chart radii are in `analysis.json`.',
        '', 'The candidate file isolates preexisting ordinary atlas component19 against body2. Its finite support and fresh sampling allocation require a separate frozen design.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    write(out/'freeze.json',{p.name:sha(p)for p in out.iterdir()if p.is_file()})
    print(json.dumps(dict(output=str(out),selected_rows=len(top_records),atomic_checks=result['atomic_geometry_checks'],site_groups=site_groups),indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.out)
