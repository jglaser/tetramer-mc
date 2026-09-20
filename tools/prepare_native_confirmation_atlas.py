#!/usr/bin/env python3
"""Freeze a regularized, cross-validated native AB pose guide; no physical production.

All original positive importance weights are retained for proposal fitting.
The final model requires fresh independent integration with a normalized defense.
"""
from __future__ import annotations

import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from prepare_smc_normalizer_atlas import Density,arrays,relative_poses,read,sha,write


ROOT=Path(__file__).resolve().parents[1]
ELL=55.02283113084892
FAMILIES={'single':([1.],[1.]),'broad_tail':([1.,2.],[.8,.2])}
SELECTION={'minimum_positive_folds':6,'minimum_drop_best_mean_gain_nat':.1,
           'require_positive_pooled_mass_gain':True,
           'scope':'Predeclared deterministic proposal-selection heuristic, not a significance test.'}


def geometric_floor(ell=ELL,translation_std=.05,angle_axis_scale_deg=.1):
    assert np.isfinite([ell,translation_std,angle_axis_scale_deg]).all()
    assert ell>0 and translation_std>0 and 0<angle_axis_scale_deg<180
    angular=ell*np.tan(np.radians(angle_axis_scale_deg)/2)
    return np.diag([translation_std**2]*3+[angular**2]*3)


def coordinates(poses,anchor,ell=ELL):
    position,_,rotation=arrays(poses)
    delta=Rotation.from_matrix(rotation@np.asarray(anchor['rotation']).T).as_quat()
    assert np.all(delta[:,3]!=0),'Exact Cayley seam in fitting chart'
    c=delta[:,:3]/delta[:,3,None]
    x=np.column_stack((position-np.asarray(anchor['position']),ell*c))
    assert np.isfinite(x).all()
    return x


def weighted_fit(x,log_weights,floor):
    assert len(x)==len(log_weights)>0 and np.isfinite(log_weights).all() and np.isfinite(x).all()
    assert floor.shape==(6,6) and np.isfinite(floor).all() and np.allclose(floor,floor.T)
    np.linalg.cholesky(floor)
    weights=np.exp(log_weights-logsumexp(log_weights))
    mean=weights@x
    centered=x-mean
    raw=centered.T@(weights[:,None]*centered)
    raw=.5*(raw+raw.T)
    covariance=raw+floor
    covariance=.5*(covariance+covariance.T)
    np.linalg.cholesky(covariance)
    eigenvalues=np.linalg.eigvalsh(covariance)
    return {'mean':mean,'raw_covariance':raw,'covariance':covariance,
            'normalized_weight_ESS':float(1/(weights@weights)),
            'largest_normalized_weight':float(weights.max()),
            'raw_eigenvalues':np.linalg.eigvalsh(raw),'eigenvalues':eigenvalues,
            'condition_number':float(eigenvalues[-1]/eigenvalues[0])}


def chart_log_density(x,fit,family,ell=ELL):
    scales,weights=FAMILIES[family]
    terms=[math.log(w)+multivariate_normal.logpdf(x,mean=fit['mean'],cov=fit['covariance']*s*s)
           for s,w in zip(scales,weights)]
    c2=np.sum((x[:,3:]/ell)**2,axis=1)
    return logsumexp(terms,axis=0)+3*np.log(ell)+2*np.log(np.pi)+2*np.log1p(c2)


def choose_family(gains,population_mass_weights):
    gains=np.asarray(gains);mass=np.asarray(population_mass_weights)
    assert gains.shape==mass.shape and len(gains)==8 and np.isfinite(gains).all()
    assert np.all(gains>=math.log(.8)-2e-9), 'Broad-tail law lost its retained single-Gaussian bound'
    positive=int(np.sum(gains>0));drop_mean=float((gains.sum()-gains.max())/(len(gains)-1))
    pooled=float(mass@gains)
    selected='broad_tail' if positive>=SELECTION['minimum_positive_folds'] and drop_mean>=SELECTION['minimum_drop_best_mean_gain_nat'] and pooled>0 else 'single'
    return {'selected_family':selected,'positive_folds':positive,
            'equal_population_mean_gain_nat':float(gains.mean()),
            'drop_largest_gain_mean_nat':drop_mean,'pooled_mass_gain_nat':pooled,
            'rule':SELECTION}


def cross_validate(x,log_weights,groups,floor,ell=ELL):
    labels=np.unique(groups);assert len(labels)==8
    masses=np.array([logsumexp(log_weights[groups==label]) for label in labels])
    mass_weights=np.exp(masses-logsumexp(masses))
    reports=[]
    for label,population_mass in zip(labels,mass_weights):
        held=groups==label;fit=weighted_fit(x[~held],log_weights[~held],floor)
        weights=np.exp(log_weights[held]-logsumexp(log_weights[held]))
        scores={name:float(weights@chart_log_density(x[held],fit,name,ell)) for name in FAMILIES}
        reports.append({'heldout_population':int(label),'training_populations':[int(i) for i in labels if i!=label],
            'training_nonzero':int((~held).sum()),'heldout_nonzero':int(held.sum()),
            'heldout_weight_ESS':float(1/(weights@weights)),
            'original_population_mass_fraction':float(population_mass),
            'weighted_predictive_log_densities':scores,
            'broad_tail_minus_single_nat':scores['broad_tail']-scores['single'],
            'training_mean':fit['mean'].tolist(),'training_condition_number':fit['condition_number']})
    selection=choose_family([r['broad_tail_minus_single_nat'] for r in reports],mass_weights)
    return {'folds':reports,'selection':selection,
            'warning':'Overlapping training sets make folds dependent. Original-weight concentration limits predictive diagnostics; they do not estimate integration variance or physical convergence.'}


def model_from_fit(fit,anchor,shape_hash,family,ell=ELL):
    scales,weights=FAMILIES[family]
    return {'schema':'weighted-pose-mixture-v1','angular_length':ell,
        'coordinate_convention':'anchor-body-relative','shape_sha256':shape_hash,
        'anchors':[copy.deepcopy(anchor) for _ in scales],
        'means':[fit['mean'].tolist() for _ in scales],
        'covariances':[(fit['covariance']*s*s).tolist() for s in scales],
        'weights':weights.copy()}


def laboratory_model(model,fixed):
    t,_,r=arrays([fixed]);r=r[0];block=np.zeros((6,6));block[:3,:3]=block[3:,3:]=r
    result=copy.deepcopy(model);result['coordinate_convention']='laboratory'
    result['anchors']=[{'position':(t[0]+r@np.asarray(c['position'])).tolist(),
                        'rotation':(r@np.asarray(c['rotation'])).tolist()} for c in model['anchors']]
    result['means']=(np.asarray(model['means'])@block.T).tolist()
    result['covariances']=(block@np.asarray(model['covariances'])@block.T).tolist()
    return result


def native_q(poses,metadata):
    t,_,r=arrays(poses);ref_t,_,ref_r=arrays(metadata['native_poses'])
    members=np.asarray([p['position'] for p in metadata['rigid_members']])
    candidates=[]
    for tr,rr in zip(ref_t,ref_r):
        moved=np.einsum('nij,kj->nki',r,members)+t[:,None,:]
        fixed=members@rr.T+tr
        error=np.linalg.norm(moved-fixed,axis=2).max(axis=1)/metadata['member_error_scale']
        angle=Rotation.from_matrix(r@rr.T).magnitude()/np.radians(metadata['angle_error_scale_deg'])
        candidates.append(np.maximum(error,angle))
    return np.minimum.reduce(candidates)


class AtomUnionAudit:
    """Exact nearest-center check separately for every fixed radius and neighbor."""
    def __init__(self,shape,fixed):
        self.centers=np.array([a['center'] for a in shape['atoms']]);self.radii=np.array([a['radius'] for a in shape['atoms']])
        t,_,r=arrays(fixed);self.fixed=[]
        for translation,rotation in zip(t,r):
            centers=self.centers@rotation.T+translation
            self.fixed.append([(float(radius),cKDTree(centers[self.radii==radius])) for radius in np.unique(self.radii)])

    def gaps(self,pose):
        t,_,r=arrays([pose]);moved=self.centers@r[0].T+t[0]
        return [float(min(np.min(tree.query(moved,k=1,workers=1)[0]-self.radii-radius)
                          for radius,tree in groups)) for groups in self.fixed]


def to_world(poses,fixed):
    t,_,r=arrays(poses);ft,_,fr=arrays([fixed]);q=Rotation.from_matrix(fr[0]@r).as_quat()[:,[3,0,1,2]]
    return [{'position':p.tolist(),'orientation':v.tolist()} for p,v in zip(t@fr[0].T+ft[0],q)]


def geometry_probes(model,config,audit,count,seed,anchor_index):
    rng=np.random.default_rng(seed);density=Density(model);reports=[];rows=[]
    for component,weight in enumerate(model['weights']):
        poses=to_world(density.draw_component(rng,component,count),config['fixed_poses'][anchor_index])
        q=native_q(poses,config['metadata']);passed=[];hard_flags=[]
        for i,pose in enumerate(poses):
            gaps=audit.gaps(pose);hard=all(g>=0 for g in gaps)
            capture=np.linalg.norm(np.asarray(pose['position'])-config['capture_center'])<=config['capture_radius']
            passed.append(hard and capture and q[i]<=1);hard_flags.append(hard)
            rows.append({'component':component,'draw':i,'pose':pose,'q':float(q[i]),
                'capture_valid':bool(capture),'hard_valid_all_neighbors':hard,
                'minimum_gap_by_neighbor_A':gaps,'native_hard_capture':bool(passed[-1])})
        reports.append({'component':component,'proposal_weight':weight,'draws':count,
            'hard_valid_all_neighbors':int(sum(hard_flags)),
            'native_hard_capture_valid':int(sum(passed)),
            'native_hard_capture_fraction':float(np.mean(passed)),
            'capture_fraction':float(np.mean([np.linalg.norm(np.asarray(p['position'])-config['capture_center'])<=config['capture_radius'] for p in poses])),
            'native_fraction':float(np.mean(q<=1))})
    fraction=sum(r['proposal_weight']*r['native_hard_capture_fraction'] for r in reports)
    se=math.sqrt(sum(r['proposal_weight']**2*r['native_hard_capture_fraction']*(1-r['native_hard_capture_fraction'])/r['draws'] for r in reports))
    return {'components':reports,'mixture_native_hard_capture_fraction':fraction,
            'observed_binomial_SE_fraction':se,'scope':'Independent proposal geometry only: no depletion clouds, equilibrium weights, or physical simulation.'},rows


def load_training(root):
    campaign=read(root/'manifest.json');assessment=read(root/'assessment-streaming.json')
    assert assessment['all_rows_and_hashes_validated'] and len(campaign['jobs'])==8
    audited={r['replicate']:r for r in assessment['populations']};rows=[];groups=[];sources=[];seen=set();reference=None
    for group,job in enumerate(campaign['jobs']):
        path=Path(job['output']);manifest=read(path/'manifest.json');summary=read(path/'summary.json')
        config=read(path/'provenance/config.json');shape_hash=sha(path/'provenance/shape.json')
        signature={'shape_sha256':shape_hash,'metric':manifest['metric'],
                   **{k:config[k] for k in ['fixed_poses','capture_center','capture_radius','depletant_radius','reservoir_density']}}
        assert manifest['config_sha256']==sha(path/'provenance/config.json') and manifest['shape_sha256']==shape_hash
        if reference is None:reference=signature;physical=config;shape=read(path/'provenance/shape.json')
        assert signature==reference and manifest['seed']==job['seed'] and job['seed'] not in seen
        assert summary['complete'];seen.add(job['seed']);digest=hashlib.sha256();draws=0;valid=0
        with (path/'samples.jsonl').open('rb') as handle:
            for line in handle:
                digest.update(line);row=json.loads(line);assert row['draw']==draws;draws+=1
                if 'zero' in row:continue
                assert 0<=row['q']<=1 and math.isfinite(row['log_importance_weight'])
                rows.append(dict(row,source_population=group,source_replicate=path.name,
                                 source_seed=job['seed'],source_file=str(path/'samples.jsonl')))
                groups.append(group);valid+=1
        assert draws==summary['samples']==manifest['samples']==audited[path.name]['samples']
        assert valid==audited[path.name]['counts']['valid']
        assert digest.hexdigest()==summary['samples_sha256']==audited[path.name]['sample_sha256']
        sources.append({'population':group,'replicate':path.name,'seed':job['seed'],
            'sample_file':str(path/'samples.jsonl'),'sample_sha256':digest.hexdigest(),
            'unconditional_draws':draws,'nonzero':valid,'manifest_sha256':sha(path/'manifest.json')})
    assert len({s['unconditional_draws'] for s in sources})==1,'Declare unequal-population pooling before use'
    assert sum(s['unconditional_draws'] for s in sources)==assessment['samples']
    assert len(rows)==assessment['regions']['native']['nonzero']
    assert all(s['nonzero']>0 for s in sources),'Each heldout population needs observed fitting mass'
    return rows,np.asarray(groups),physical,shape,reference,sources,assessment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=ROOT/'runs/native-cover-mixture-ab-confirmation-8x131072-l64-20260920')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--anchor-index',type=int,default=0)
    parser.add_argument('--translation-floor-A',type=float,default=.05)
    parser.add_argument('--rotation-axis-floor-deg',type=float,default=.1)
    parser.add_argument('--probes-per-component',type=int,default=512)
    args=parser.parse_args();started=time.monotonic();root=args.campaign.resolve();out=args.out.resolve()
    assert not out.exists(),'Use a fresh frozen-fit output directory'
    assert args.probes_per_component>0 and args.translation_floor_A>0 and args.rotation_axis_floor_deg>0
    (out/'provenance').mkdir(parents=True)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'campaign':str(root),
        'campaign_manifest_sha256':sha(root/'manifest.json'),'assessment_sha256':sha(root/'assessment-streaming.json'),
        'angular_length_A':ELL,'proposal_anchor_index':args.anchor_index,
        'translation_floor_A':args.translation_floor_A,'rotation_axis_floor_deg':args.rotation_axis_floor_deg,
        'floor_rule':'Add diag(sigma_t^2 I3, [ell*tan(theta_axis/2)]^2 I3) in deployment anchor coordinates; angular value is a nominal axis-angle scale, not an exact transformed variance.',
        'candidate_families':FAMILIES,'selection_rule':SELECTION,'wide_model_std_multiplier':2.,
        'probes_per_component':args.probes_per_component,'no_physical_production':True,'script_sha256':sha(__file__)}
    write(out/'predeclared-fit-protocol.json',protocol)
    for path,name in [(Path(__file__),'prepare_native_confirmation_atlas.py'),
                      (Path(__file__).with_name('prepare_smc_normalizer_atlas.py'),'prepare_smc_normalizer_atlas.py'),
                      (root/'manifest.json','training-campaign-manifest.json'),
                      (root/'assessment-streaming.json','training-assessment.json')]:shutil.copy2(path,out/'provenance'/name)
    rows,groups,config,shape,signature,sources,assessment=load_training(root)
    assert len(config['fixed_poses'])==2 and 0<=args.anchor_index<2
    assert len(config['metadata']['native_poses'])==1
    fixed=config['fixed_poses'][args.anchor_index]
    poses=[r['pose'] for r in rows];relative=relative_poses(poses,fixed)
    native_relative=relative_poses(config['metadata']['native_poses'],fixed)
    t0,_,r0=arrays(native_relative);anchor={'position':t0[0].tolist(),'rotation':r0[0].tolist()}
    x=coordinates(relative,anchor);logw=np.asarray([r['log_importance_weight'] for r in rows])
    assert np.max(np.abs(native_q(poses,config['metadata'])-np.array([r['q'] for r in rows])))<2e-8
    floor=geometric_floor(ELL,args.translation_floor_A,args.rotation_axis_floor_deg)
    fit=weighted_fit(x,logw,floor)
    assert abs(fit['normalized_weight_ESS']/assessment['regions']['native']['ess']-1)<1e-9
    cv=cross_validate(x,logw,groups,floor);family=cv['selection']['selected_family']
    model=model_from_fit(fit,anchor,signature['shape_sha256'],family)
    provenance={'kind':'Frozen all-population original-weight native AB covariance guide',
        'physical_target':signature,'training_campaign':str(root),'training_nonzero':len(rows),
        'training_unconditional_draws':assessment['samples'],'proposal_anchor_index':args.anchor_index,
        'selected_family':family,'floor':floor.tolist(),'fit_protocol_sha256':sha(out/'predeclared-fit-protocol.json'),
        'requirements':'Native target unchanged; full AB environment; normalized outer geometric defense plus existing atlas/cube law; fresh independent draws only.'}
    model['proposal_provenance']=provenance
    wide=copy.deepcopy(model);wide['covariances']=(4*np.asarray(model['covariances'])).tolist()
    wide['proposal_provenance']=dict(provenance,model_wide_std_multiplier=2.,width_choice='Prespecified sensitivity; not selected using new physical data')
    write(out/'model.json',model);write(out/'model-wide.json',wide);write(out/'cross-validation.json',cv)
    write(out/'provenance/physical-config.json',config)
    original_shape=Path(sources[0]['sample_file']).parent/'provenance/shape.json'
    shutil.copy2(original_shape,out/'provenance/shape.json')
    assert sha(out/'provenance/shape.json')==signature['shape_sha256']
    with (out/'provenance/training-poses.jsonl').open('w') as handle:
        for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
    lab=laboratory_model(model,fixed)
    body_values=Density(model).evaluate(relative)[0];lab_values=Density(lab).evaluate(poses)[0]
    manual=chart_log_density(x,fit,family)
    coordinate_audit={'max_lab_body_log_density_difference':float(np.max(np.abs(body_values-lab_values))),
        'max_independent_scipy_chart_log_density_difference':float(np.max(np.abs(body_values-manual))),
        'maximum_native_chart_angle_deg':float(np.degrees(2*np.arctan(np.linalg.norm(x[:,3:]/ELL,axis=1))).max())}
    assert max(coordinate_audit[k] for k in coordinate_audit if 'difference' in k)<2e-7
    audit=AtomUnionAudit(shape,config['fixed_poses'])
    rng=np.random.default_rng(98971010)
    checked=np.unique(np.concatenate((rng.choice(len(rows),min(128,len(rows)),replace=False),np.argsort(logw)[-16:])))
    training_checks=[]
    for i in checked:
        gaps=audit.gaps(poses[i]);assert all(g>=0 for g in gaps),(int(i),gaps)
        training_checks.append({'training_row':int(i),'source_population':int(groups[i]),'source_draw':rows[i]['draw'],
                                'minimum_gap_by_neighbor_A':gaps})
    probes={}
    for index,(name,item) in enumerate([('model',model),('model-wide',wide)]):
        report,probe_rows=geometry_probes(item,config,audit,args.probes_per_component,98972010+1009*index,args.anchor_index)
        probes[name]=report
        with (out/f'{name}-geometry-probes.jsonl').open('w') as handle:
            for row in probe_rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
    # Conditional cost forecast based only on geometric success and the previous
    # valid-pose cloud cost; no new depletant points are generated here.
    prior_cpu_per_valid=assessment['cpu_seconds']/len(rows)
    base_cover=assessment['populations'][0]['cover_volume']
    cover_fraction=assessment['zero_activity_volume_estimate_A3']/base_cover
    cube_fraction=assessment['zero_activity_volume_estimate_A3']/(2*config['capture_radius'])**3
    for report in probes.values():
        report['hybrid_valid_fraction_forecast_beta075_epsilon005']=.25*cover_fraction+.75*(.95*report['mixture_native_hard_capture_fraction']+.05*cube_fraction)
        report['forecast_CPU_seconds_for_4x8192']=32768*report['hybrid_valid_fraction_forecast_beta075_epsilon005']*prior_cpu_per_valid
    report={'complete':True,'model_sha256':sha(out/'model.json'),'wide_model_sha256':sha(out/'model-wide.json'),
        'fit':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit.items()},
        'floor':floor.tolist(),'selection':cv['selection'],'coordinate_density_audit':coordinate_audit,
        'training_geometry_checks':training_checks,'geometry_probes':probes,'source_populations':sources,
        'training_nonzero':len(rows),'training_unconditional_draws':assessment['samples'],
        'wall_seconds':time.monotonic()-started,'cost_forecast_caveat':'Uses prior valid-pose cost and noisy hard-volume estimate. Candidate overlap cloud costs may change; no runtime guarantee.',
        'scope':'Frozen proposal fitting and independent static geometry only. No physical production, fresh mass estimate, selected-weight replacement, or equilibrium-covariance claim.',
        'archived_sha256':{p.name:sha(p) for p in (out/'provenance').iterdir()}}
    write(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ['complete','model_sha256','wide_model_sha256','selection','coordinate_density_audit','geometry_probes','training_nonzero','wall_seconds']},indent=2))


if __name__=='__main__':main()
