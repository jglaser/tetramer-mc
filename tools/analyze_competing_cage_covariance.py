"""One frozen Gaussian per predeclared competing cage from complete local residence.

Initial preparation, proposal-chart center and immutable region reference are
separate inputs. No geometry or physical samples are generated. Heldout streams
are evaluated once and never alter the fitted parameters or ridge rule.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2

from analyze_source_cage_covariance import (coordinates,covariance,decompose,describe,
    retained_states,success,require,read,sha,QUANTILES)
from prepare_context_covariance_guides import freeze_moments
from mobile_posterior_metrics import apparent_effective_count

SCHEMA='competing-cage-covariance-v1'
COVERAGE=[.5,.9,.95,.99]
FIT_RULE=dict(training_streams=[0,1],heldout_streams=[2,3],phase='production',covariance_denominator='N',
    bandwidth_multiplier=1.,ridge_absolute_scaled=1e-10,ridge_trace_factor=1e-6,
    translation_scale_angstrom=.1,rotation_scale_degrees=.5,
    gaussian_components_per_declared_cage=1,heldout_parameter_updates=0)


def validate_center(center):
    require(center['schema']=='source-chart-center-v1' and center['frame']=='saved-spherical-center'
            and isinstance(center['provenance'],str) and bool(center['provenance'].strip()),'Invalid frozen chart center')
    # Reuse chart quaternion/position validation without any atom transformation.
    x,m=coordinates([center['pose']],center['pose'],center['pose'],1.)
    require(not m['failure_indices'] and np.max(np.abs(x))<1e-12,'Invalid center pose')


def validate_cage_inventory(cages):
    require([c['cage_id'] for c in cages]==[0,1],'Exactly the two predeclared cage identities are required')
    require(cages[0]['initial_pose']!=cages[1]['initial_pose'],'Competing seeds must remain distinct')
    require(all([s['stream'] for s in c['streams']]==[0,1,2,3] for c in cages),'Missing/changed stream inventory')
    seeds=[s['seed'] for c in cages for s in c['streams']]
    require(len(set(seeds))==8 and all(type(s) is int and 0<=s<2**64 for s in seeds),'Invalid or repeated independent stream seed')


def validate_patch_reference(reference,patch_reference):
    """The immutable reference is a complete source-guide configuration."""
    require(reference['inputs']['regions']==patch_reference['definitions']
            and reference['inputs']['patch_map']==patch_reference['patch_map']
            and patch_reference['unchanged_during_training'] is True,'Region reference is immutable')


def fit_gaussian(train,scales):
    """Fixed scaled ridge, including an explicit degeneracy report, never tuning."""
    mean,cov=covariance(train);mean,full,_,epsilon=freeze_moments(mean,cov,scales)
    scaled=cov/np.outer(scales,scales);raw_eigen=np.linalg.eigvalsh(.5*(scaled+scaled.T))
    unique=len(np.unique(train,axis=0));rank=int(np.linalg.matrix_rank(scaled))
    ridge_trace=6*epsilon;trace=float(np.trace(scaled))
    status='no_observed_motion_ridge_only' if unique==1 else 'rank_deficient_regularized' if rank<6 else 'full_empirical_rank'
    return dict(mean=mean.tolist(),covariance=full.tolist(),unregularized_covariance=cov.tolist(),
        normalized_gaussian=True,training_samples=len(train),unique_training_coordinate_states=unique,
        empirical_rank=rank,fit_status=status,raw_scaled_eigenvalues=raw_eigen.tolist(),
        scaled_variance_trace=trace,scaled_ridge=epsilon,
        ridge_fraction_of_regularized_trace=ridge_trace/(trace+ridge_trace),
        relative_covariance_change=float(np.linalg.norm(full-cov)/np.linalg.norm(cov)) if np.linalg.norm(cov)>0 else None,
        caveat='A ridge makes a normalized proposal, not an identified physical basin. Unique stored states are not independent samples; rank/low motion and heldout mismatch remain visible.')


def likelihood_diagnostics(x,fit,ell):
    """Normalized-Haar density once; chi-square coverage is a Gaussian fit diagnostic."""
    x=np.asarray(x,float);require(x.ndim==2 and x.shape[1]==6 and len(x)>0 and np.isfinite(x).all(),'Invalid evaluation coordinates')
    lower=np.linalg.cholesky(np.asarray(fit['covariance']));mean=np.asarray(fit['mean'])
    z=solve_triangular(lower,(x-mean).T,lower=True).T;r2=np.einsum('ij,ij->i',z,z)
    c=x[:,3:]/ell
    logj=-3*math.log(ell)-2*math.log(math.pi)-2*np.log1p(np.einsum('ij,ij->i',c,c))
    logp=-.5*r2-3*math.log(2*math.pi)-np.log(np.diag(lower)).sum()-logj
    require(np.isfinite(logp).all(),'Unrepresentable heldout density')
    return dict(samples=len(x),mean_log_density=float(logp.mean()),negative_mean_log_density=float(-logp.mean()),
        log_density_quantiles=np.quantile(logp,QUANTILES).tolist(),mean_squared_mahalanobis=float(r2.mean()),
        squared_mahalanobis_quantiles=np.quantile(r2,QUANTILES).tolist(),
        nominal_gaussian_coverage=[dict(probability=p,squared_radius=float(chi2.ppf(p,6)),
            count=int(np.sum(r2<=chi2.ppf(p,6))),fraction=float(np.mean(r2<=chi2.ppf(p,6)))) for p in COVERAGE],
        measure='Translation volume times normalized rotational Haar; Cayley Jacobian divided exactly once.',
        scope='Unweighted residence likelihood and nominal Gaussian coordinate coverage, not physical equilibrium probabilities or independent-validation uncertainty.')


def movement_diagnostics(x,rows,cpu,scales):
    accepted=np.array([r['accepted'] for r in rows],bool)
    return dict(unique_retained_coordinate_states=len(np.unique(x,axis=0)),
        unique_accepted_coordinate_states=len(np.unique(x[accepted],axis=0)) if accepted.any() else 0,
        accepted_states=int(accepted.sum()),rejected_states=int((~accepted).sum()),
        maximum_coordinate_range=np.ptp(x,axis=0).tolist(),
        coordinate_autocorrelation=[apparent_effective_count(x[:,j]/scales[j],cpu) for j in range(6)],
        scaled_vector_autocorrelation=apparent_effective_count(x/scales,cpu),
        CPU_denominator='Full training-chain invocation CPU, reused for each phase; no phase-specific CPU attribution.',
        scope='Every retained elementary state, including all rejection residence. Constant descriptors have undefined ESS; these finite-record diagnostics do not certify equilibrium or independent basin coverage.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();protocol=read(args.protocol);bindings=protocol['input_sha256']
    require(protocol['schema']==SCHEMA and protocol['fit_rule']==FIT_RULE and protocol['quantiles']==QUANTILES
            and protocol['nominal_coverage']==COVERAGE and protocol['cycles']==2304 and protocol['warmup_cycles']==256
            and protocol['slots_per_cycle']==5 and protocol['physical_conditions']==dict(depletant_radius=1.5,activity=.035,lambda_ratio=64.),'Changed fixed fit protocol')
    for path,digest in bindings.items():require(sha(path)==digest,'Changed bound input: '+path)
    def asset(a):
        path=str(Path(a['path']).resolve());require(bindings.get(path)==a['sha256'],'Unbound asset')
        return read(path)
    original=asset(protocol['original_source_state']);context=asset(protocol['context']);model=asset(protocol['model'])
    asset(protocol['patch_reference']['patch_map']);reference=asset(protocol['patch_reference']['regions'])
    validate_patch_reference(reference,protocol['patch_reference'])
    ell=float(model.get('base_model',model)['angular_length']);require(ell==protocol['angular_length'],'Changed angular length')
    anchor=next(b['pose'] for b in context['bodies'] if b['label']==16)
    require(original['anchor_pose']==anchor and original['moving_label']==77
            and context['excluded_moving_labels']==[77] and [b['label'] for b in context['bodies']]==[i for i in range(264) if i!=77], 'Changed fixed frame/context')
    scales=np.array([.1]*3+[ell*math.tan(math.pi/720)]*3)
    cages=protocol['cages'];validate_cage_inventory(cages)
    require(not args.out.exists(),'Fresh result required');args.out.mkdir();started=time.process_time();output=[]
    with (args.out/'events.jsonl').open('x') as journal:
        def emit(value):journal.write(json.dumps(value,allow_nan=False)+'\n');journal.flush()
        try:
            for cage in cages:
                emit(dict(kind='cage_begun',cage_id=cage['cage_id']))
                center=asset(cage['chart_center']);validate_center(center)
                require(center['pose']==cage['initial_pose'],'Predeclared seed-centered chart changed')
                require([s['stream'] for s in cage['streams']]==[0,1,2,3], 'Missing/changed stream inventory')
                data={};stream_reports=[];audits=[]
                for item in cage['streams']:
                    emit(dict(kind='stream_begun',cage_id=cage['cage_id'],stream=item['stream']))
                    cfg=asset(item['config']);summary=asset(item['summary']);asset(item['receipt'])
                    success(item['receipt']['path'],item['summary']['path'])
                    require(cfg['initial_pose']==cage['initial_pose'] and cfg['identity']==summary['identity']
                            and cfg['identity']['stream']==item['stream'] and cfg['seed']==item['seed']
                            and cfg['identity']['seed_index']==cage['cage_id']
                            and cfg['identity']['split']==('train' if item['stream']<2 else 'heldout')
                            and cfg['identity']['source_candidate']==cage['source_candidate_identity']
                            and summary['complete'] and summary['passed'] and summary['method']=='local'
                            and summary['completed_cycles']==2304 and summary['completed_attempts']==11520
                            and cfg['warmup_cycles']==256 and cfg['local_attempts_per_cycle']==4
                            and cfg['translation_steps']==[.2] and cfg['rotation_steps_deg']==[1.]
                            and cfg['rotation_probability']==.5 and cfg['depletant_radius']==1.5
                            and cfg['reservoir_density']==.035 and cfg['poisson_lambda_ratio']==64.,'Changed local training preparation/schedule')
                    for name,a in [('source_state',protocol['original_source_state']),('fixed_context',protocol['context']),('model',protocol['model'])]:
                        require(summary['bindings'][name]==a['sha256'],'Original physical/proposal reference changed')
                    require(summary['bindings']['config']==item['config']['sha256'],'Config not bound to executed summary')
                    path=str(Path(item['journal']['path']).resolve());require(bindings.get(path)==item['journal']['sha256'],'Journal lacks binding')
                    rows,audit=retained_states(path,cfg['identity'],cage['initial_pose'])
                    audits.append(dict(stream=item['stream'],**audit))
                    for phase in ('warmup','production'):
                        chosen=[r for r in rows if r['production'] is (phase=='production')]
                        x,m=coordinates([r['pose'] for r in chosen],center['pose'],anchor,ell)
                        failures=[chosen[i]['attempt_index'] for i in m['failure_indices']]
                        result=dict(stream=item['stream'],phase=phase,states=len(chosen),chart_failures=failures,
                            near_seam_count=m['near_seam_count'],minimum_absolute_quaternion_scalar=m['minimum_absolute_quaternion_scalar'],
                            neighbor_residence={','.join(map(str,k)):v for k,v in Counter(tuple(r['neighbors']) for r in chosen).items()})
                        if failures:result.update(coordinate_status='undefined; every seam state retained, none dropped')
                        else:
                            result.update(coordinate_status='complete',**describe(x,m,scales,ell),
                                movement=movement_diagnostics(x,chosen,summary['invocation_cpu_seconds'],scales))
                            data[item['stream'],phase]=(x,m)
                        stream_reports.append(result)
                    emit(dict(kind='stream_complete',cage_id=cage['cage_id'],stream=item['stream'],**audit))
                pooled=[]
                for phase in ('warmup','production'):
                    for name,streams in [('train',[0,1]),('heldout',[2,3]),('all',[0,1,2,3])]:
                        if not all((s,phase) in data for s in streams):
                            pooled.append(dict(phase=phase,group=name,coordinate_status='undefined due to retained seam'));continue
                        groups=[data[s,phase][0] for s in streams]
                        movements={key:np.concatenate([data[s,phase][1][key] for s in streams]) for key in ('translation_norm','rotation_angle_degrees')}
                        pooled.append(dict(phase=phase,group=name,coordinate_status='complete',streams=streams,
                            **describe(np.concatenate(groups),movements,scales,ell),covariance_decomposition=decompose(groups,scales)))
                fitted=None;diagnostics={};guide_asset=None
                if all((s,'production') in data for s in (0,1)):
                    train=np.concatenate([data[s,'production'][0] for s in (0,1)])
                    require(len(train)==20480,'Wrong training denominator')
                    fitted=fit_gaussian(train,scales)
                    for stream in range(4):
                        key=(stream,'production')
                        diagnostics[str(stream)]=likelihood_diagnostics(data[key][0],fitted,ell) if key in data else dict(status='undefined; preserved chart seam')
                    for label,streams in [('train',[0,1]),('heldout',[2,3])]:
                        diagnostics[label]=likelihood_diagnostics(np.concatenate([data[s,'production'][0] for s in streams]),fitted,ell) if all((s,'production') in data for s in streams) else dict(status='undefined; preserved chart seam')
                    guide=dict(schema='competing-cage-frozen-guide-v1',cage_id=cage['cage_id'],fit_status=fitted['fit_status'],
                        source_chart=dict(angular_length=ell,covariance=fitted['covariance'],chart_center=center,
                            explicit_gaussian=dict(schema='source-gaussian-v1',mean=fitted['mean'],provenance=f"Predeclared cage {cage['cage_id']}, trainstreams0,1; protocol {sha(args.protocol)}")),
                        protocol_sha256=sha(args.protocol),fit_rule=FIT_RULE,original_source_state=protocol['original_source_state'],
                        patch_reference=protocol['patch_reference'],
                        training_samples=20480,heldout_fit_samples=0,normalized_gaussian=True,
                        scope='One frozen kernel for one predeclared starting cage. Includes every production residence state, even if the trajectory leaves the initial cage; no native-label filtering or equilibrium claim.')
                    path=args.out/(str(cage['cage_id'])+'-guide.json');path.write_text(json.dumps(guide,indent=2,allow_nan=False)+'\n')
                    guide_asset=dict(path=str(path.resolve()),sha256=sha(path))
                output.append(dict(cage_id=cage['cage_id'],initial_pose=cage['initial_pose'],chart_center=center,
                    original_source_pose=original['pose'],anchor_pose=anchor,inventory_audits=audits,per_stream=stream_reports,
                    pooled=pooled,fit=fitted,guide=guide_asset,likelihood_diagnostics=diagnostics,
                    fit_status='training_chart_failure_no_fit' if fitted is None else fitted['fit_status']))
                emit(dict(kind='cage_complete',cage_id=cage['cage_id'],fit_status=output[-1]['fit_status']))
            for path,digest in bindings.items():require(sha(path)==digest,'Bound input changed')
            result=dict(schema=SCHEMA,complete=True,passed=True,source_sha256=sha(__file__),protocol_sha256=sha(args.protocol),
                input_sha256=bindings,fit_rule=FIT_RULE,cages=output,scale_vector=scales.tolist(),angular_length=ell,
                total_retained_states=92160,production_states=81920,declared_training_states=40960,
                fitted_training_states=sum(c['fit']['training_samples'] for c in output if c['fit'] is not None),
                heldout_fit_samples=0,patch_reference=protocol['patch_reference'],
                new_poses=0,new_geometry_queries=0,new_clouds=0,cpu_seconds=time.process_time()-started,
                scope='Two predeclared starting cages, one Gaussian each, fixed context at historical500uM. Low motion/rank deficiency and serial dependence remain explicit. Heldout likelihood/nominal coverage diagnose a proposal approximation, not equilibrium basin probabilities or assembly.')
            (args.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        except BaseException as e:
            emit(dict(kind='fatal',error=repr(e),completed_cages=len(output)))
            (args.out/'failure.json').write_text(json.dumps(dict(complete=False,passed=False,error=repr(e),prefix_preserved=True),indent=2)+'\n')
            raise

if __name__=='__main__':main()
