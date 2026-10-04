#!/usr/bin/env python3
"""Independent full-vessel adapter for the normalized native-class-line proposal.

The source capture belongs to the proposal's conditional intervals. The vessel
capture and atomic wall belong to the physical target. Neither conditions away
an attempted proposal. Complete physical density is q_latent/J, once.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
import copy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
import native_class_line_reference as line
import native_class_line_physical_reference as regional
from native_contact_regions import NativeContactRegions
import physical_latent_guide as original
import audit_full_vessel_latent as vessel_reference
from analyze_basin_normalizers import audit_wall_domain, moments, paired_noise
from analyze_r4_smc_control import Ledger
from analyze_mobile_native_pocket import local_sources

require,read,sha,close=regional.require,regional.read,regional.sha,regional.close
SCHEMA='full-vessel-native-class-line-half-mixture-v1'


@dataclass(frozen=True)
class PhysicalGuideDensity:
    latent:list|None
    in_reference_ball:bool
    log_latent_density:float|None
    log_physical_jacobian:float|None
    log_physical_density:float
    structural_zero:bool
    reconstruction:dict|None


class PhysicalNativeClassLineGuide(original.PhysicalLatentGuide):
    """World density of the full class/axis/component law, not a class target.

    The computational chart factor is independently checked against its frozen
    covariance at 80-digit precision by Reconstructor. Tiny represented entries
    are not clipped: they may determine an orthant boundary.
    """
    def __init__(self,region,guide,vessel_config,shape,observer,*,region_sha256,expected_shape_sha256):
        require(guide['schema']==line.SCHEMA,'Wrong native-class proposal law')
        require(guide['region_sha256']==region_sha256,'Wrong frozen source region')
        require(guide['shape_sha256']==expected_shape_sha256,'Wrong guide shape')
        gaussian=dict(schema='defensive-latent-shell-guide-v1',region_sha256=region_sha256,
            defensive_uniform_shell_probability=guide['defensive_uniform_shell_probability'],
            gaussian_components=copy.deepcopy(guide['gaussian_components']))
        super().__init__(region,gaussian,region_sha256=region_sha256,expected_shape_sha256=expected_shape_sha256)
        require(self.physical_fixed_neighbors==vessel_config['fixed_poses']==guide['fixed_poses']==observer.fixed_poses,
                'Source/vessel/native physical scaffold differs')
        for key in ('capture_center','capture_radius'):
            require(guide[key]==region[key],'Changed source guide '+key)
        require(guide['depletant_radius']==region['depletant_radius']==vessel_config['depletant_radius'],
                'Changed depletant radius')
        self.original_guide=copy.deepcopy(guide);source=copy.deepcopy(vessel_config)
        source['capture_center']=copy.deepcopy(region['capture_center']);source['capture_radius']=region['capture_radius']
        self.recon=line.Reconstructor(region,guide,source,shape,observer)
        cumulative=0.
        for channel in self.recon.channels:
            following=cumulative+channel['probability']
            require(math.isfinite(following) and following-cumulative>=np.finfo(float).eps and following<=1.,
                    'Unrepresentable class channel categorical increment')
            cumulative=following
        require(cumulative==1.,'Class categorical cumulative probability must end at one')
        self.lower=self.recon.L0
        self.log_det=float(np.log(np.diag(self.lower)).sum())
        self.source_config=source;self.vessel_config=copy.deepcopy(vessel_config)

    @classmethod
    def from_files(cls,region_path,guide_path,*,compiled_path,observer,vessel_config,shape,expected_shape_sha256):
        region_path,guide_path,compiled_path=map(lambda p:Path(p).resolve(),(region_path,guide_path,compiled_path))
        guide,compiled=read(guide_path),read(compiled_path)
        require(guide['compiled_native']['sha256']==sha(compiled_path),'Changed compiled native bytes')
        require(compiled['source_input_sha256']['tetramer-shape.json']==expected_shape_sha256,
                'Native source shape differs')
        require(compiled['fixed_poses']==guide['fixed_poses'],'Compiled native scaffold differs')
        value=cls(read(region_path),guide,vessel_config,shape,observer,
                  region_sha256=sha(region_path),expected_shape_sha256=expected_shape_sha256)
        value.sources={str(p):sha(p) for p in (region_path,guide_path,compiled_path)}
        return value

    def evaluate(self,pose):
        positions,rotations=original._poses([pose])
        relative=self.fixed_r.T@rotations[0]@self.anchor_r.T
        q=Rotation.from_matrix(relative).as_quat()
        if q[3]==0.:
            return PhysicalGuideDensity(None,False,None,None,-math.inf,True,None)
        raw=np.r_[self.fixed_r.T@(positions[0]-self.fixed_t)-self.anchor_t,self.ell*q[:3]/q[3]]
        u=line.scalar_chart_solve(self.lower,raw-self.mean)
        require(np.isfinite(u).all(),'Unrepresentable inverse chart coordinates')
        inside=bool(np.all(abs(u)<=4.) and u@u<=16.)
        logj=self.log_det-3*math.log(self.ell)-2*math.log(math.pi)+4*math.log(abs(q[3]))
        require(math.isfinite(logj),'Unrepresentable physical Jacobian')
        if self.recon.alpha==1.:
            # A pure uniform exterior zero needs no Gaussian or geometry query.
            logq=-self.recon.logvolume if inside else -math.inf
            result=dict(log_density=logq,baseline_log_density=logq,conditioning_disabled=True,axes=[])
            structural=not inside
        else:
            with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
                result=self.recon.density(u)
                gaussian=self.recon.gaussian_logs(u)
            active=positive_components(self.recon,result)
            if self.recon.beta>0.:
                factors=np.asarray(result['component_multipliers'])
                require(np.isfinite(factors).all() and np.array_equal(factors>0,active),
                        'Class multiplier disagrees with analytic support; no underflow hole is allowed')
            structural=bool(not inside and not active.any())
            require(np.isfinite(gaussian[active]).all(),
                    'Unrepresentable positive Gaussian component; not a structural zero')
            logq=result['log_density']
            require(math.isfinite(logq) or(structural and logq==-math.inf),'Unrepresentable positive guide density')
            require((logq==-math.inf)==structural,'Structural support certificate differs from complete density')
        physical=logq-logj
        require(math.isfinite(physical) or(structural and physical==-math.inf),'Unrepresentable physical proposal density')
        return PhysicalGuideDensity(u.tolist(),inside,logq,logj,physical,structural,result)

    def evaluate_many(self,poses):return [self.evaluate(p) for p in poses]
    def log_densities(self,poses):return np.array([self.evaluate(p).log_physical_density for p in poses])


def positive_components(recon,result):
    """Support uses logical membership, never floating point mixture weights.

    Every axis/channel has strictly positive prior mass. The fallback already
    selected from its conditional Normal mass is retained; no second geometry
    traversal, no underflow-based component removal, and no q floor is used.
    """
    if recon.beta<1.:return np.ones(len(recon.weights),bool)
    active=np.zeros(len(recon.weights),bool)
    for axis in result['axes']:
        for k,component in enumerate(axis['components']):
            for branch in component['channels']:
                active[k] |= branch['fallback']=='unconditional' or branch['query_coordinate_allowed']
    return active


def audit_trace(actual,density,guide):
    """Check the compact trace against independently rebuilt full geometry."""
    if density.latent is None:
        require(actual is None,'Exact seam has an interval trace');return 0.
    expected=density.reconstruction;require(actual is not None,'Missing complete conditional trace')
    if expected.get('conditioning_disabled'):
        require(actual.get('conditioning_disabled') is True,'Missing disabled-conditioning flag');return 0.
    r=guide.recon;u=np.asarray(density.latent)
    require(actual.get('trace_format')=='class-line-compact-v1','Wrong compact class trace contract')
    close(actual['raw_coordinates'],r.raw(u),'Density raw coordinates differ')
    vessel_reference.log_close(actual['baseline_log_density'],expected['baseline_log_density'],'Baseline density differs')
    require([a['axis'] for a in actual['axes']]==r.axes,'Density axis sequence differs')
    maximum=0.;factors=np.zeros(len(r.weights))
    for a,b in zip(actual['axes'],expected['axes']):
        require('components' not in a,'Expected compact class trace')
        maximum=max(maximum,line.compare_geometry(a,b,r,u))
        for k,component in enumerate(b['components']):
            factors[k]+=sum(c['probability']*branch['multiplier'] for c,branch in zip(r.channels,component['channels']))/len(r.axes)
    close(actual['component_mixture_multipliers'],factors,'Complete class mixture differs',atol=1e-12,rtol=3e-7)
    return maximum


def check_rows(config,manifest,rows,vessel,guide):
    require(manifest['schema']==7 and manifest['outer_mixture_schema']==SCHEMA
        and manifest['latent_guide_schema']==line.SCHEMA,'Wrong vessel native-class schema')
    evaluated=guide.evaluate_many([r['pose'] for r in rows])
    class Cached:
        def evaluate_many(self,poses):
            require(poses==[r['pose'] for r in rows],'Unexpected cached-density request');return evaluated
    result=vessel_reference.check_rows(config,manifest,rows,vessel,Cached())
    intervals=0.;inverse=0.;zeros=0;seams=0;outside_source=0
    for row,density in zip(rows,evaluated):
        record=row['latent_density']
        require(type(record['structural_zero']) is bool and record['structural_zero']==density.structural_zero,
                'Structural-zero certificate differs')
        zeros+=density.structural_zero;seams+=density.latent is None
        intervals=max(intervals,audit_trace(record['native_class_line_density'],density,guide))
        source_inside=math.dist(row['pose']['position'],guide.region['capture_center'])<=guide.region['capture_radius']
        outside_source+=bool(row['hard_valid'] and not source_inside)
        if row['outer_branch']=='latent':
            require(not density.structural_zero,'Generated latent pose has zero component support')
            generated=row['latent_proposal'];component=generated['gaussian_component']
            require(component is None or(type(component)is int and 0<=component<len(guide.recon.weights)),
                    'Invalid latent selected component')
            close(generated['latent'],density.latent,'Generated latent inverse differs')
            draw=generated['native_class_line_draw']
            if draw['conditional']:
                require(component==draw['component'],'Conditional component differs from selected Gaussian')
            # Generation and world-density scoring are separate FP operations.
            # A roundtrip can cross an exact orthant boundary. Audit the saved
            # generative trace at its own latent point; q still uses world pose.
            generated_u=np.asarray(generated['latent'])
            generated_density=guide.recon.density(generated_u) if draw['conditional'] else {}
            audit=line.audit_draw(draw,guide.recon,generated_u,generated_density)
            inverse=max(inverse,audit['inverse_error']);intervals=max(intervals,audit['endpoint_error'])
        else:
            require(row['latent_proposal'] is None,'Vessel branch contains latent generation metadata')
    result.update(structural_zero_queries=zeros,exact_chart_seams=seams,valid_outside_source_capture=outside_source,
        maximum_interval_endpoint_error=intervals,maximum_inverse_CDF_error=inverse,
        source_capture=guide.region['capture_radius'],vessel_capture=config['capture_radius'],
        chart_factor_validation=guide.recon.chart_factor_validation,
        scope='Complete component/axis/channel density over full R6; source capture restricts conditioning only. qphysical=qlatent/J once; .5 vessel mixture preserves full target. Class fallback and exterior support certified independently of floating-point weight underflow.')
    return result


def audit(directory, *, definition_path=None, synthetic=False):
    require(sys.flags.optimize==0,'Frozen reference checks require assertions enabled')
    root=Path(directory).resolve();started=time.process_time();ledger=Ledger()
    for p in local_sources(__file__).values():ledger.bind(p)
    manifest=read(ledger.bind(root/'manifest.json'));summary=read(ledger.bind(root/'summary.json'))
    require(manifest['schema']==7 and manifest['outer_mixture_schema']==SCHEMA and manifest['outer_vessel_probability']==.5,'Wrong outer mixture')
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
    provenance=manifest['compiled_native']
    compiled_path=ledger.bind(root/'provenance/compiled-native.json',provenance['compiled_sha256'])
    compiled=read(compiled_path)
    require(compiled['source_definition_sha256']==provenance['source_definition_sha256']
        and compiled['source_input_sha256']==provenance['source_input_sha256'],'Native provenance differs')
    shape_witness=regional.validate_shape_witness(compiled,shape,provenance['shape_compatibility'],
        provenance['compiled_sha256'],manifest['shape_sha256'])
    if synthetic:
        require(definition_path is None and len(shape['atoms'])<=64 and len(compiled['monomer_atoms'])<=16,
                'Synthetic compiled adapter cannot audit proteins')
        observer=line.observer_from_compiled_for_synthetic(compiled)
    else:
        require(definition_path is not None,'Original frozen native definition required')
        observer=NativeContactRegions(ledger.bind(definition_path,provenance['source_definition_sha256']))
        line.compare_compiled_definition(compiled,observer)
        for name,digest in observer.definition['input_sha256'].items():ledger.bind(observer.root/name,digest)
    guide=PhysicalNativeClassLineGuide.from_files(root/'provenance/latent-region.json',root/'provenance/latent-guide.json',
        vessel_config=config,shape=shape,expected_shape_sha256=manifest['shape_sha256'],
        compiled_path=compiled_path,observer=observer)
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
    return dict(schema='full-vessel-native-class-line-independent-audit-v1',complete=True,population=str(root),manifest=manifest,
        density_audit=density_check,vessel_generation_audit=generation,primitive_count_audit=primitive,wall_audit=wall,
        shape_witness=shape_witness,geometry_audit=contact.report(),near_core_boundary_poses=int(near),estimates=estimates,
        source_sha256=ledger.files,samples_sha256=sha(samples),attempts_sha256=sha(attempts),analysis_CPU_seconds=time.process_time()-started,
        new_pose_draws=0,new_Poisson_clouds=0,native_interval_geometry_reconstructed=True,
        scope='One full-vessel population audit, not convergence. All attempts and physical wall/hard zeros retained. Native predicates define proposal intervals only; physical native partitioning remains a separate audit. RNG independence, exact thinning/envelope certification and floating-point execution remain source-bound obligations.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--definition',type=Path);p.add_argument('--synthetic',action='store_true');args=p.parse_args()
    require(not args.out.exists(),'Fresh output required');result=audit(args.directory,definition_path=args.definition,synthetic=args.synthetic);args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(args.out)
