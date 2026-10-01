#!/usr/bin/env python3
"""Independent full-vessel adapter for the normalized hard-free-line proposal.

The source capture belongs to the proposal's conditional intervals. The vessel
capture and atomic wall belong to the physical target. Neither conditions away
an attempted proposal. Complete physical density is q_latent/J, once.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
import hard_free_line_reference as line
import hard_free_line_physical_reference as regional
import physical_latent_guide as original
import audit_full_vessel_latent as vessel_reference
from analyze_basin_normalizers import audit_wall_domain, moments, paired_noise
from analyze_r4_smc_control import Ledger
from analyze_mobile_native_pocket import local_sources

require,read,sha,close=line.require,line.read,line.sha,line.close
SCHEMA='full-vessel-hard-free-line-half-mixture-v1'


@dataclass(frozen=True)
class PhysicalGuideDensity:
    latent:list|None
    in_reference_ball:bool
    log_latent_density:float|None
    log_physical_jacobian:float|None
    log_physical_density:float
    structural_zero:bool
    reconstruction:dict|None


class PhysicalHardFreeLineGuide(original.PhysicalLatentGuide):
    """Reuse checked chart validation; keep the actual guide and source geometry."""
    def __init__(self,region,guide,vessel_config,shape,*,region_sha256,expected_shape_sha256):
        require(guide['schema']=='defensive-hard-free-line-guide-v1','Wrong hard-free proposal law')
        require(guide['region_sha256']==region_sha256,'Wrong frozen source region')
        gaussian=dict(schema='defensive-latent-shell-guide-v1',region_sha256=region_sha256,
            defensive_uniform_shell_probability=guide['defensive_uniform_shell_probability'],gaussian_components=copy.deepcopy(guide['gaussian_components']))
        # Private chart-validation adapter only: no Gaussian law is substituted
        # for the actual conditional density evaluated below.
        super().__init__(region,gaussian,region_sha256=region_sha256,expected_shape_sha256=expected_shape_sha256)
        require(self.physical_fixed_neighbors==vessel_config['fixed_poses'],'Source/vessel physical scaffold differs')
        self.original_guide=copy.deepcopy(guide);source=copy.deepcopy(vessel_config)
        source['capture_center']=copy.deepcopy(region['capture_center']);source['capture_radius']=region['capture_radius']
        self.recon=line.Reconstructor(self.region,guide,source,shape)
        self.source_config=source;self.vessel_config=copy.deepcopy(vessel_config)

    @classmethod
    def from_files(cls,region_path,guide_path,*,vessel_config,shape,expected_shape_sha256):
        region_path,guide_path=Path(region_path).resolve(),Path(guide_path).resolve()
        value=cls(read(region_path),read(guide_path),vessel_config,shape,region_sha256=sha(region_path),expected_shape_sha256=expected_shape_sha256)
        value.sources={str(region_path):sha(region_path),str(guide_path):sha(guide_path)};return value

    def evaluate(self,pose):
        positions,rotations=original._poses([pose])
        relative=self.fixed_r.T@rotations[0]@self.anchor_r.T
        q=Rotation.from_matrix(relative).as_quat()
        if q[3]==0.:
            return PhysicalGuideDensity(None,False,None,None,-math.inf,True,None)
        raw=np.r_[self.fixed_r.T@(positions[0]-self.fixed_t)-self.anchor_t,self.ell*q[:3]/q[3]]
        u=solve_triangular(self.lower,raw-self.mean,lower=True)
        require(np.isfinite(u).all(),'Unrepresentable inverse chart coordinates')
        inside=bool(np.all(abs(u)<=4.) and u@u<=16.)
        logj=self.log_det-3*math.log(self.ell)-2*math.log(math.pi)+4*math.log(abs(q[3]))
        require(math.isfinite(logj),'Unrepresentable physical Jacobian')
        if self.recon.alpha==1.:
            # A legitimate exterior zero does not require enormous Cayley
            # coordinates to pass through irrelevant Gaussian/geometry code.
            logq=-self.recon.logvolume if inside else -math.inf
            result=dict(log_density=logq,baseline_log_density=logq,conditioning_disabled=True,axes=[],fallback_component_branches=0)
            structural=not inside
        else:
            with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
                result=self.recon.density(u)
                gaussian=self.recon.gaussian_logs(u)
            if self.recon.beta<1.:
                active=np.ones(len(gaussian),bool)
            else:
                active=np.zeros(len(gaussian),bool)
                scored_raw=self.recon.raw(u)
                for axis in result['axes']:
                    active|=np.asarray(axis['component_fallbacks'],bool)|line.contains(axis['intervals'],scored_raw[axis['axis']])
            structural=bool(not inside and not active.any())
            require(np.isfinite(gaussian[active]).all(),'Unrepresentable positive Gaussian component; not a structural zero')
            logq=result['log_density']
            require(math.isfinite(logq) or (structural and logq==-math.inf),'Unrepresentable positive guide density')
            require((logq==-math.inf)==structural,'Structural support certificate differs from complete density')
        physical=logq-logj
        require(math.isfinite(physical) or(structural and physical==-math.inf),'Unrepresentable physical proposal density')
        return PhysicalGuideDensity(u.tolist(),inside,logq,logj,physical,structural,result)

    def evaluate_many(self,poses):return [self.evaluate(p) for p in poses]
    def log_densities(self,poses):return np.array([self.evaluate(p).log_physical_density for p in poses])


def audit_trace(actual,density,guide):
    """Compare compact Rust trace against already reconstructed full geometry."""
    if density.latent is None:
        require(actual is None,'Exact seam has an interval trace');return 0.
    expected=density.reconstruction;require(actual is not None,'Missing complete conditional trace')
    if expected.get('conditioning_disabled'):
        require(actual.get('conditioning_disabled') is True,'Missing disabled-conditioning flag');return 0.
    r=guide.recon;u=np.asarray(density.latent);raw=r.raw(u);close(actual['raw_coordinates'],raw,'Density raw coordinates differ')
    vessel_reference.log_close(actual['baseline_log_density'],expected['baseline_log_density'],'Baseline density differs')
    require(actual['component_branches']==expected['component_branches']
        and actual['fallback_component_branches']==expected['fallback_component_branches'],'Conditional branch counts differ')
    require([a['axis'] for a in actual['axes']]==r.axes,'Density axis sequence differs')
    g=r.gaussian_logs(u);uniform=math.log(r.alpha)-r.logvolume if density.in_reference_ball else -math.inf
    maximum=0.
    for a,b in zip(actual['axes'],expected['axes']):
        maximum=max(maximum,line.compare_axis(a,b));m=np.asarray(b['conditional_masses']);fallback=np.asarray(b['component_fallbacks'])
        factors=np.ones(len(g));factors[~fallback]=1/m[~fallback] if line.contains(b['intervals'],raw[b['axis']]) else 0.
        correction=1-r.beta+r.beta*factors;positive=correction>0
        logq=float(np.logaddexp(uniform,math.log1p(-r.alpha)+logsumexp(g[positive]+np.log(correction[positive]))))
        vessel_reference.log_close(a['axis_log_proposal_density'],logq,'Single-axis full density differs')
    return maximum


def check_rows(config,manifest,rows,vessel,guide):
    require(manifest['schema']==6 and manifest['outer_mixture_schema']==SCHEMA
        and manifest['latent_guide_schema']=='defensive-hard-free-line-guide-v1','Wrong vessel hard-free schema')
    # The frozen outer-law/weight auditor accepts a density evaluator object;
    # its qphysical, mixture, no-extra-J and unconditional-zero checks apply as is.
    evaluated=guide.evaluate_many([r['pose'] for r in rows])
    class Cached:
        def evaluate_many(self,poses):
            require(poses==[r['pose'] for r in rows],'Unexpected cached-density request');return evaluated
    result=vessel_reference.check_rows(config,manifest,rows,vessel,Cached())
    intervals=0.;inverse=0.;zeros=0;seams=0;outside_source=0
    for row,density in zip(rows,evaluated):
        record=row['latent_density']
        require(type(record['structural_zero']) is bool and record['structural_zero']==density.structural_zero,'Structural-zero certificate differs')
        zeros+=density.structural_zero;seams+=density.latent is None
        intervals=max(intervals,audit_trace(record['hard_free_line_density'],density,guide))
        source_inside=math.dist(row['pose']['position'],guide.region['capture_center'])<=guide.region['capture_radius']
        outside_source+=bool(row['hard_valid'] and not source_inside)
        if row['outer_branch']=='latent':
            require(not density.structural_zero,'Generated latent pose has zero component support')
            generated=row['latent_proposal'];component=generated['gaussian_component']
            adapter=dict(latent=generated['latent'],hard_free_line_draw=generated['hard_free_line_draw'],
                proposal_branch='uniform-shell' if component is None else 'hard-free-line',proposal_component=component)
            inverse=max(inverse,regional.audit_draw(adapter,guide.recon,density.reconstruction))
        elif row['latent_proposal'] is not None:
            raise ValueError('Vessel branch contains latent generation metadata')
    result.update(structural_zero_queries=zeros,exact_chart_seams=seams,valid_outside_source_capture=outside_source,
        maximum_interval_endpoint_error=intervals,maximum_inverse_CDF_error=inverse,
        source_capture=copy.deepcopy(guide.region['capture_radius']),vessel_capture=config['capture_radius'],
        scope='Every world pose scored under the complete normalized full-R6 guide and original vessel law. Source capture controls conditional intervals only. qphysical=qlatent/J exactly once; outer .5 mixture keeps full vessel support. Structural zeros are certified by support, never exp(logq) underflow.')
    return result


def audit(directory):
    require(sys.flags.optimize==0,'Frozen reference checks require assertions enabled')
    root=Path(directory).resolve();started=time.process_time();ledger=Ledger()
    for p in local_sources(__file__).values():ledger.bind(p)
    manifest=read(ledger.bind(root/'manifest.json'));summary=read(ledger.bind(root/'summary.json'))
    require(manifest['schema']==6 and manifest['outer_mixture_schema']==SCHEMA and manifest['outer_vessel_probability']==.5,'Wrong outer mixture')
    require(summary['complete'] is True and summary['manifest']==manifest and summary['numerical_nulls']==0
        and not(root/'failure.json').exists(),'Incomplete or failed population')
    require(manifest['density_measure']=='Lebesgue center volume times normalized SO(3) Haar measure'
        and manifest['latent_reference_ball_is_target_restriction'] is False
        and manifest['latent_source_capture']['restricts_target'] is False and manifest['bath_wall_permeable'] is True,'Changed physical measure/domain')
    for name,key in [('input-config.json','config_sha256'),('model.json','model_sha256'),('shape.json','shape_sha256'),
        ('source-bundle.json','source_bundle_sha256'),('latent-region.json','latent_region_sha256'),('latent-guide.json','latent_guide_sha256')]:
        ledger.bind(root/'provenance'/name,manifest[key])
    samples=ledger.bind(root/'samples.jsonl',summary['samples_sha256']);attempts=ledger.bind(root/'attempts.jsonl',summary['attempts_sha256'])
    config=read(ledger.bind(root/'config.json'));original_config=read(root/'provenance/input-config.json')
    for key in ('fixed_poses','capture_center','capture_radius','depletant_radius','metadata'):
        require(config[key]==original_config[key],'Physical config changed: '+key)
    require(config['reservoir_density']==manifest['activity'] and config.get('target_region') is None,'Changed bath/hidden target')
    close(manifest['lambda'],config['poisson_lambda_ratio']*manifest['activity'] if manifest['activity']>0 else 1.,'Auxiliary intensity differs')
    rows=[json.loads(s) for s in samples.read_text().splitlines()]
    require([json.loads(s) for s in attempts.read_text().splitlines()]==[dict(draw=i,state='begin') for i in range(manifest['samples'])],
            'Missing/repeated/reordered attempted draw')
    shape=read(root/'provenance/shape.json')
    guide=PhysicalHardFreeLineGuide.from_files(root/'provenance/latent-region.json',root/'provenance/latent-guide.json',
        vessel_config=config,shape=shape,expected_shape_sha256=manifest['shape_sha256'])
    require(manifest['latent_defensive_uniform_probability']==guide.guide.alpha
        and manifest['latent_gaussian_component_count']==guide.guide.count,'Changed guide mixture')
    source=manifest['latent_source_capture']
    require(source['center']==guide.region['capture_center'] and source['radius']==guide.region['capture_radius'],'Changed source capture')
    vessel=vessel_reference.VesselDensity(config,manifest,read(root/'provenance/model.json'),root/'provenance/source-bundle.json')
    density_check=check_rows(config,manifest,rows,vessel,guide)
    generation=vessel_reference.check_generation_metadata(config,manifest,rows,vessel)
    primitive=vessel_reference.check_cloud_envelopes_and_counts(manifest,summary,rows)
    wall=audit_wall_domain(root,dict(manifest,schema=4),rows,summary)
    contact=vessel_reference.PrunedExclusionContact(shape,config['fixed_poses'],config['depletant_radius'])
    near=0;classes=[]
    for row in rows:
        measured=contact.classify(row['pose'],capture_valid=row['capture_valid'],wall_valid=row['wall_valid'])
        expected=row['capture_valid'] and row['wall_valid'] and measured['core_disjoint']
        require(row['hard_valid']==expected,'Independent atomic hard predicate differs')
        near+=measured['near_core_boundary']
        if row['hard_valid']:require(row['depletion_contact']==measured['exclusion_contact'],'Independent exclusion contact differs')
        valid=row['hard_valid'];inside=row['latent_density']['in_reference_ball']
        classes.append(dict(total=valid,inside_R4=valid and inside,outside_R4=valid and not inside,
            latent_zero=valid and row['latent_density']['structural_zero'],
            exclusion_contact=valid and measured['exclusion_contact'],unbound=valid and not measured['exclusion_contact']))
    estimates={}
    for name in classes[0]:
        selected=[c[name] for c in classes]
        logs=np.array([r['log_importance_weight'] if yes else -np.inf for r,yes in zip(rows,selected)])
        hard=np.array([r['log_hard_weight'] if yes else -np.inf for r,yes in zip(rows,selected)])
        pairs=np.array([[c['log_weight']-r['log_proposal_density'] for c in r['clouds']] if yes else [-np.inf,-np.inf] for r,yes in zip(rows,selected)])
        estimates[name]=dict(Qz=moments(logs),Q0=moments(hard),paired_noise=paired_noise(logs,pairs))
    for key,kind in [('total','Qz'),('hard_total','Q0')]:
        vessel_reference.log_close(summary['estimates'][key]['log_normalizer'],vessel_reference.nullable_log(estimates['total'][kind]['logQ']), 'Total all-attempt estimate differs')
    ledger.recheck()
    return dict(schema='full-vessel-hard-free-line-independent-audit-v1',complete=True,population=str(root),manifest=manifest,
        density_audit=density_check,vessel_generation_audit=generation,primitive_count_audit=primitive,wall_audit=wall,
        geometry_audit=contact.report(),near_core_boundary_poses=int(near),estimates=estimates,
        source_sha256=ledger.files,samples_sha256=sha(samples),attempts_sha256=sha(attempts),analysis_CPU_seconds=time.process_time()-started,
        new_pose_draws=0,new_Poisson_clouds=0,new_native_classifier_calls=0,
        scope='One full-vessel population audit, not convergence. All attempts and physical wall/hard zeros retained. Native regions are not classified here. RNG independence, exact thinning/envelope certification and floating-point execution remain source-bound obligations.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    require(not args.out.exists(),'Fresh output required');result=audit(args.directory);args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(args.out)
