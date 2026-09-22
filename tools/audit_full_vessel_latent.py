#!/usr/bin/env python3
"""Independent schema-5 full-vessel mixture audit, retaining every draw.

The legacy auditor remains unchanged. Its generation check is used only for
vessel-generated rows; full vessel and latent densities are evaluated on ALL
world poses here. No proposal or weight is changed in the recorded estimator.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.special import logsumexp

from analyze_basin_normalizers import audit_pose_proposal, audit_wall_domain, moments, paired_noise
from analyze_mobile_native_pocket import load_classifier, local_sources
from analyze_mobile_threshold_reference import ExclusionContact
from analyze_r4_smc_control import Ledger, close, nullable_log, read, require, sha, write
from normalizer_proposal_density import NormalizerProposalDensity
from physical_latent_guide import PhysicalLatentGuide, half_mixture_log_density
from prepare_deep_far_normalizer_atlas import registration
from prepare_smc_normalizer_atlas import Density, relative_poses, unwrap_proposal_model

SCHEMA='full-vessel-latent-half-mixture-v1'


def log_close(recorded, expected, message):
    value=nullable_log(recorded)
    if expected == -math.inf:
        require(value == -math.inf,message)
        return 0.
    close(value,float(expected),message)
    return abs(value-expected)



class PrunedExclusionContact:
    """One-sided geometry certificates; no contact is inferred from bounds.

    The caller has already independently checked capture and every atomic wall
    predicate. Outside that domain, core/contact geometry is not applicable to
    the zero target weight. Inside it, centers farther than twice the complete
    body bound plus 2 rd certify BOTH hard and exclusion disjointness. All other
    anchor pairs use the existing exact variable-radius atom-union gap search.
    """
    def __init__(self,shape,fixed,depletant_radius):
        require(bool(fixed),'At least one fixed neighbor required')
        self.exact=ExclusionContact(shape,fixed,depletant_radius)
        self.bound=max(math.hypot(*a['center'])+a['radius'] for a in shape['atoms'])
        require(math.isfinite(self.bound) and self.bound>0,'Unrepresentable complete body bound')
        self.fixed_centers=[tuple(p['position']) for p in fixed]
        self.rd=depletant_radius
        self.counts=dict(attempts=0,domain_invalid_pose_skips=0,capture_invalid_pose_skips=0,
            wall_invalid_pose_skips=0,certified_disjoint_pose_checks=0,
            exact_core_contact_pose_checks=0,certified_disjoint_anchor_checks=0,
            exact_core_contact_anchor_checks=0)

    def classify(self,pose,*,capture_valid,wall_valid):
        require(type(capture_valid) is bool and type(wall_valid) is bool,'Domain flags must be checked Booleans')
        self.counts['attempts']+=1
        if not capture_valid or not wall_valid:
            self.counts['domain_invalid_pose_skips']+=1
            self.counts['capture_invalid_pose_skips' if not capture_valid else 'wall_invalid_pose_skips']+=1
            return dict(applicable=False,method='not-applicable-domain-invalid',
                reason='outside-capture' if not capture_valid else 'outside-atomic-wall',
                core_disjoint=None,exclusion_contact=None,anchors=[],near_core_boundary=False)
        anchors=[];points=None;exact_gaps=[];near_count=0
        position=tuple(pose['position'])
        for i,fixed in enumerate(self.fixed_centers):
            distance=math.hypot(*(a-b for a,b in zip(position,fixed)))
            # Guard subtraction, norms/body-bound arithmetic, and rigid-frame
            # roundoff. Strict inequality deliberately leaves tangencies and a
            # finite roundoff neighborhood for the exact atom calculation.
            scale=1.+sum(abs(v) for v in (*position,*fixed))+distance+2*self.bound+2*self.rd
            guard=1024*np.finfo(float).eps*scale
            lower=distance-2*self.bound-guard
            if math.isfinite(lower) and lower>2*self.rd:
                anchors.append(dict(anchor_index=i,method='certified-disjoint',
                    minimum_surface_gap_A=None,surface_gap_lower_bound_A=lower,
                    moving_fixed_atom_indices=None,core_disjoint=True,exclusion_contact=False,
                    center_distance_A=distance,body_bound_A=self.bound,roundoff_guard_A=guard))
                self.counts['certified_disjoint_anchor_checks']+=1
            else:
                if points is None:points=self.exact.placed(pose)
                result=self.exact.minimum_gap(points,i);gap=result['minimum_surface_gap_A']
                result.update(method='exact-atom-gap',core_disjoint=gap>=0)
                anchors.append(result);exact_gaps.append(gap);near_count+=1
                self.counts['exact_core_contact_anchor_checks']+=1
        self.counts['exact_core_contact_pose_checks' if near_count else 'certified_disjoint_pose_checks']+=1
        near=bool(exact_gaps and abs(min(exact_gaps))<1e-8)
        return dict(applicable=True,method='exact-and-certified' if 0<near_count<len(anchors)
            else 'exact-atom-gap' if near_count else 'certified-disjoint',
            core_disjoint=all(a['core_disjoint'] for a in anchors),
            exclusion_contact=any(a['exclusion_contact'] for a in anchors),anchors=anchors,
            contact_surface_gap_threshold_A=2*self.rd,near_core_boundary=near,
            near_zero_negative_gap=any(-1e-8<=g<0 for g in exact_gaps))

    def report(self):
        require(self.counts['attempts']==self.counts['domain_invalid_pose_skips']
                +self.counts['certified_disjoint_pose_checks']+self.counts['exact_core_contact_pose_checks'],
                'Geometry pruning lost an attempted pose')
        return dict(**self.counts,body_bound_A=self.bound,
            disjoint_certificate='center distance minus twice complete body bound minus conservative roundoff guard > 2 rd',
            scope='All attempted densities and atomic-wall predicates remain audited. Only domain-invalid zero-target core/contact checks and strictly certified disjoint anchor checks are pruned. Bounds never certify a contact; minimum gap is null on certificate-only anchors.')


class VesselDensity:
    """Full original anchor-averaged law, independent of generation branch."""
    def __init__(self,config,manifest,raw_model,source_bundle=None):
        self.config=config;self.manifest=manifest
        base,flags=unwrap_proposal_model(raw_model)
        require(base.get('dfs') is None or all(v is None for v in base['dfs']), 'Gaussian vessel model required')
        require(base['shape_sha256']==manifest['shape_sha256'],'Vessel shape identity differs')
        schema=manifest['pose_proposal_schema'];require(schema in (1,2,3),'Unknown vessel proposal law')
        if schema==3:
            require(any(flags) and manifest['covariance_scale']==1.,'Reciprocal law/scaling differs')
            require(manifest['reciprocal_components']==flags and manifest['base_component_count']==len(flags)
                and manifest['virtual_component_count']==len(flags)+sum(flags),'Reciprocal component metadata differs')
            self.density=NormalizerProposalDensity(raw_model,source_bundle=source_bundle)
        else:
            require(not any(flags),'Active reciprocal law uses schema 3')
            model=dict(base,covariances=(np.asarray(base['covariances'])*manifest['covariance_scale']**2).tolist())
            self.density=Density(model)
        index=manifest['proposal_anchor_index']
        self.indices=[index] if index is not None else list(range(len(config['fixed_poses'])))
        require(self.indices and all(type(i) is int and 0<=i<len(config['fixed_poses']) for i in self.indices),'Invalid physical proposal anchors')
        require(manifest['physical_fixed_neighbor_count']==len(config['fixed_poses']),'Scaffold size differs')
        self.epsilon=manifest['uniform_probability'];self.radius=config['capture_radius']
        require(0<self.epsilon<=1 and math.isfinite(self.radius) and self.radius>0,'Invalid vessel defensive law')

    def evaluate(self,poses):
        require(len(poses)>0,'Nonempty audit batch required')
        displacement=np.asarray([p['position'] for p in poses])-self.config['capture_center']
        cube=np.all((displacement>=-self.radius)&(displacement<self.radius),axis=1)
        uniform=np.where(cube,math.log(self.epsilon)-3*math.log(2*self.radius),-np.inf)
        gaussians=np.asarray([self.density.evaluate(relative_poses(poses,self.config['fixed_poses'][i]))[0] for i in self.indices])
        anchors=np.logaddexp(uniform[None,:],math.log1p(-self.epsilon)+gaussians) if self.epsilon<1 else np.broadcast_to(uniform,gaussians.shape)
        full=logsumexp(anchors,axis=0)-math.log(len(self.indices))
        return full,dict(anchor_log_densities=anchors,cube=cube,capture=np.linalg.norm(displacement,axis=1)<=self.radius)


def check_rows(config,manifest,rows,vessel,latent):
    """Density/Jacobian/generation/weight algebra; geometry is checked separately."""
    require(rows and [r['draw'] for r in rows]==list(range(manifest['samples'])),'Missing attempted draw')
    poses=[r['pose'] for r in rows];require(all(p is not None for p in poses),'Censored numerical null')
    vessel_logs,vessel_geometry=vessel.evaluate(poses)
    latent_rows=latent.evaluate_many(poses)
    mixture=half_mixture_log_density(vessel_logs,[r.log_physical_density for r in latent_rows])
    require(np.isfinite(mixture).all(),'Generated pose lacks mixture support')
    errors=[];branches=Counter();outside_valid=0
    expected_q=registration(poses,config['metadata'])
    for i,(row,density,mixed) in enumerate(zip(rows,latent_rows,mixture)):
        branches[row['outer_branch']]+=1
        errors.extend([log_close(row['log_vessel_proposal_density'],vessel_logs[i],'Full vessel density differs'),
                       log_close(row['log_latent_physical_density'],density.log_physical_density,'Full latent physical density differs'),
                       log_close(row['log_proposal_density'],mixed,'Complete outer mixture differs')])
        require(mixed>=vessel_logs[i]-math.log(2)-1e-12,'Lost vessel defensive support')
        record=row['latent_density'];seam=density.latent is None
        require(record['coordinate_chart_seam']==seam and record['in_reference_ball']==density.in_reference_ball,'Latent reporting flags differ')
        if seam:
            require(record['latent'] is None and record['log_latent_density'] is None
                and record['log_physical_jacobian'] is None,'Exact chart seam must have undefined coordinates')
        else:
            require(np.allclose(record['latent'],density.latent,rtol=2e-11,atol=2e-8),'Inverse world-pose chart differs')
            log_close(record['log_latent_density'],density.log_latent_density,'Latent mixture differs')
            log_close(record['log_physical_jacobian'],density.log_physical_jacobian,'Physical Jacobian differs')
        if row['outer_branch']=='latent':
            require(row['proposal'] is None and row['latent_proposal'] is not None and not seam,'Invalid latent generation metadata')
            generated=row['latent_proposal'];component=generated['gaussian_component']
            require(np.allclose(generated['latent'],density.latent,rtol=2e-11,atol=2e-8),'Generated/inverse latent coordinates differ')
            close(generated['latent_radius'],float(np.linalg.norm(density.latent)),'Generated latent radius differs')
            if component is None: require(density.in_reference_ball,'Uniform latent draw left R4')
            else: require(type(component) is int and 0<=component<manifest['latent_gaussian_component_count']
                          and manifest['latent_defensive_uniform_probability']<1,'Invalid Gaussian generation component')
        else:
            require(row['outer_branch']=='vessel' and row['proposal'] is not None and row['latent_proposal'] is None,'Unknown or contradictory outer branch')
        require(type(row['capture_valid']) is bool and row['capture_valid']==bool(vessel_geometry['capture'][i]),'Capture predicate differs')
        require(type(row['hard_valid']) is bool and type(row['wall_valid']) is bool,'Non-Boolean hard/wall flag')
        if row['hard_valid']:
            require(row['capture_valid'] and row['wall_valid'],'Hard-valid row leaves physical domain')
            close(row['log_hard_weight'],-mixed,'Physical hard weight must use inverse FULL mixture, no extra J')
            close(row['q'],float(expected_q[i]),'Original physical registration metric differs')
            require(len(row['clouds'])==manifest['cloud_replicates']==2,'Two independent clouds required')
            for cloud in row['clouds']:
                require(type(cloud['overlap_points']) is int and cloud['overlap_points']>=0
                    and math.isfinite(cloud['lower_volume']) and cloud['lower_volume']>=0
                    and math.isfinite(cloud['uncertain_volume']) and cloud['uncertain_volume']>=0,'Invalid cloud count/volume')
                z=manifest['activity'];lam=manifest['lambda']
                factor=z*cloud['lower_volume']+cloud['overlap_points']*math.log1p(z/lam) if z else 0.
                close(cloud['log_weight'],factor,'Poisson estimator count identity differs')
            close(row['log_importance_weight'],float(logsumexp([c['log_weight'] for c in row['clouds']])-math.log(2)-mixed),'Arithmetic cloud-mean weight differs')
            outside_valid+=int(not density.in_reference_ball)
        else:
            require(all(row[k] is None for k in ('log_hard_weight','log_importance_weight','q','region','depletion_contact'))
                    and not row['clouds'],'Invalid attempted draw lost its explicit zero')
    return dict(checked_attempts=len(rows),maximum_log_density_error=max(errors),outer_branches=dict(branches),
        valid_outside_R4=outside_valid,scope='All densities evaluated in physical measure on every attempted world pose; no validity conditioning.')



def check_generation_metadata(config, manifest, rows, vessel):
    """Every vessel law needs this check: the old auditor labels only schema 3.

    This verifies coordinates, branch/component support and selected-anchor
    forward/reverse densities. It does not reconstruct the random stream.
    """
    selected=[r for r in rows if r['outer_branch']=='vessel']
    if not selected:return dict(checked_vessel_generation_rows=0)
    _,geometry=vessel.evaluate([r['pose'] for r in selected])
    _,initial=vessel.evaluate([config['initial_pose']])
    flags=getattr(vessel.density,'reciprocal_components',[])
    count=len(vessel.density.model['weights'])
    reciprocal=manifest['pose_proposal_schema']==3
    for i,row in enumerate(selected):
        proposal=row['proposal'];anchor=proposal['anchor_index']
        require(type(proposal['moving_index']) is int and proposal['moving_index']==0
                and type(anchor) is int and 1<=anchor<=len(vessel.indices),'Invalid vessel generation anchor label')
        require(proposal['null_reason'] is None and proposal['candidate'] is not None,'Censored vessel generation')
        candidate=proposal['candidate'];world=row['pose']
        require(np.allclose(candidate['position'],np.asarray(world['position'])-config['capture_center'],rtol=0,atol=2e-8),
                'Vessel generation/world translation differs')
        a=np.asarray(candidate['orientation']);b=np.asarray(world['orientation'])
        require(a.shape==b.shape==(4,) and np.isfinite(a).all()
                and min(np.linalg.norm(a-b),np.linalg.norm(a+b))<2e-8,'Vessel generation/world rotation differs')
        component=proposal['component_index']
        if proposal['branch']=='uniform':
            require(geometry['cube'][i] and component is None and 'component_inverted' not in proposal,
                    'Uniform vessel generation metadata differs')
        else:
            require(proposal['branch']=='learned' and vessel.epsilon<1
                    and type(component) is int and 0<=component<count,'Unsupported learned vessel component')
            if reciprocal:
                inverted=proposal['component_inverted']
                require(type(inverted) is bool and (not inverted or flags[component]),'Unsupported reciprocal generation label')
            else:require('component_inverted' not in proposal,'Legacy generator has reciprocal label')
        new=float(geometry['anchor_log_densities'][anchor-1,i]);old=float(initial['anchor_log_densities'][anchor-1,0])
        log_close(proposal['new_log_density'],new,'Selected-anchor generation density differs')
        log_close(proposal['old_log_density'],old,'Initial selected-anchor density differs')
        close(proposal['log_reverse_forward'],old-new,'Selected-anchor forward/reverse ratio differs')
    return dict(checked_vessel_generation_rows=len(selected),scope='Generation metadata and selected-anchor densities; RNG replay is a separate implementation obligation.')


def check_cloud_envelopes_and_counts(manifest, summary, rows):
    """Check logged envelope/count consistency without claiming exact thinning."""
    require(summary['samples']==manifest['samples']==len(rows),'Summary attempted-draw denominator differs')
    counts=dict(hard_valid=sum(r['hard_valid'] for r in rows),
        capture_rejected=sum(not r['capture_valid'] for r in rows),
        wall_rejected=sum(r['capture_valid'] and not r['wall_valid'] for r in rows),
        hard_rejected=sum(r['capture_valid'] and r['wall_valid'] and not r['hard_valid'] for r in rows),
        raw_points=0)
    for row in rows:
        if not row['hard_valid']:continue
        expected=('native_core' if row['q']<=.8 else 'native_shell' if row['q']<=1
                  else 'shoulder' if row['q']<2 else 'intermediate' if row['q']<5 else 'distant')
        require(type(row['depletion_contact']) is bool and row['region']==expected+('_bound' if row['depletion_contact'] else '_unbound'),
                'Original exhaustive region label differs')
        first=row['clouds'][0]
        for cloud in row['clouds']:
            for field in ('raw_points','overlap_points','retained_cells','created_cells','certified_cells'):
                require(type(cloud[field]) is int and cloud[field]>=0,'Invalid primitive Poisson/envelope count')
            require(cloud['overlap_points']<=cloud['raw_points'],'Thinned count exceeds raw Poisson count')
            require(cloud['retained_cells']<=cloud['created_cells'] and cloud['certified_cells']<=cloud['created_cells'],
                    'Envelope cell count exceeds constructed cells')
            close(cloud['upper_volume'],cloud['lower_volume']+cloud['uncertain_volume'],'Envelope volume decomposition differs')
            for field in ('lower_volume','uncertain_volume','upper_volume','retained_cells','created_cells','certified_cells'):
                require(cloud[field]==first[field],'Independent clouds used different pose envelopes')
            if manifest['activity']==0 or cloud['uncertain_volume']==0:
                require(cloud['raw_points']==cloud['overlap_points']==0,'Zero-intensity/volume cloud sampled nonzero count')
            counts['raw_points']+=cloud['raw_points']
    require(all(summary[k]==v for k,v in counts.items()),'Summary rejection/point counters differ from all rows')
    return counts


def audit(root,out,native_definition=None):
    require(sys.flags.optimize==0,'Independent legacy checks require Python assertions')
    root=Path(root).resolve();out=Path(out).resolve();require(not out.exists(),'Fresh audit output required')
    ledger=Ledger();audit_sources=local_sources(__file__)
    for path in audit_sources.values():ledger.bind(path)
    manifest=read(ledger.bind(root/'manifest.json'));summary=read(ledger.bind(root/'summary.json'))
    require(manifest['schema']==5 and manifest['outer_mixture_schema']==SCHEMA and manifest['outer_vessel_probability']==.5,'Unknown outer mixture')
    require(manifest['density_measure']=='Lebesgue center volume times normalized SO(3) Haar measure','Physical density measure differs')
    require(manifest['latent_reference_ball_is_target_restriction'] is False
        and manifest['latent_source_capture']['restricts_target'] is False and manifest['bath_wall_permeable'] is True,'Latent/wall target law differs')
    require(summary['complete'] and summary['manifest']==manifest and summary['numerical_nulls']==0,'Incomplete population')
    # This normalizer does not emit a raw-sample hash in summary.json. Bind
    # the actual bytes now; an external pre-frozen digest is not asserted.
    raw=ledger.bind(root/'samples.jsonl')
    for name,key in [('input-config.json','config_sha256'),('model.json','model_sha256'),('shape.json','shape_sha256'),
        ('source-bundle.json','source_bundle_sha256'),('latent-region.json','latent_region_sha256'),('latent-guide.json','latent_guide_sha256')]:
        ledger.bind(root/'provenance'/name,manifest[key])
    config=read(ledger.bind(root/'config.json'));original=read(root/'provenance/input-config.json')
    for key in ('fixed_poses','capture_center','capture_radius','depletant_radius','metadata'):
        require(config[key]==original[key],'Physical config differs: '+key)
    require(config['reservoir_density']==manifest['activity'],'Activity differs')
    require(math.isfinite(manifest['activity']) and manifest['activity']>=0
        and math.isfinite(manifest['lambda']) and manifest['lambda']>0,'Invalid bath intensity')
    close(manifest['lambda'],config['poisson_lambda_ratio']*manifest['activity'] if manifest['activity']>0 else 1.,'Bath cloud intensity differs')
    require(config.get('target_region') is None,'Unsupported hidden target restriction')
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    vessel=VesselDensity(config,manifest,read(root/'provenance/model.json'),root/'provenance/source-bundle.json')
    latent=PhysicalLatentGuide.from_files(root/'provenance/latent-region.json',root/'provenance/latent-guide.json',expected_shape_sha256=manifest['shape_sha256'])
    require(latent.physical_fixed_neighbors==config['fixed_poses'],'Latent/vessel scaffold differs')
    require(manifest['latent_defensive_uniform_probability']==latent.guide.alpha
        and manifest['latent_gaussian_component_count']==latent.guide.count,'Latent mixture manifest differs')
    source=manifest['latent_source_capture']
    require(source['center']==latent.region['capture_center'] and source['radius']==latent.region['capture_radius'],'Source capture metadata differs')
    density_check=check_rows(config,manifest,rows,vessel,latent)
    generation_metadata_check=check_generation_metadata(config,manifest,rows,vessel)
    count_check=check_cloud_envelopes_and_counts(manifest,summary,rows)
    # Explicit compatibility adapters: the engine sees only its original schema
    # and vessel-generated rows, with component-law weights on temporary COPIES.
    # The full-mixture original weights were checked above and remain untouched.
    legacy_manifest=dict(manifest,schema=4)
    vessel_rows=[]
    for row in rows:
        if row['outer_branch']!='vessel':continue
        adapted=copy.deepcopy(row);adapted['log_proposal_density']=row['log_vessel_proposal_density']
        if row['hard_valid']:adapted['log_hard_weight']=-row['log_vessel_proposal_density']
        vessel_rows.append(adapted)
    generation_check=(audit_pose_proposal(root,dict(proposal_anchor_index=manifest['proposal_anchor_index']),legacy_manifest,vessel_rows)
                      if vessel_rows else dict(checked_actual_poses=0))
    wall_check=audit_wall_domain(root,legacy_manifest,rows,summary)
    contact=PrunedExclusionContact(read(root/'provenance/shape.json'),config['fixed_poses'],config['depletant_radius'])
    classifier=None;native_binding=None
    if native_definition:
        classifier,native_binding=load_classifier(native_definition);definition=classifier.definition
        require(definition['shape_sha256']==manifest['shape_sha256'] and definition['fixed_poses']==config['fixed_poses'],'Native geometry identity differs')
        native_path=Path(native_definition).resolve();ledger.bind(native_path)
        for name,digest in definition['input_sha256'].items():ledger.bind(native_path.parent/'inputs'/name,digest)
        native_config=read(native_path.parent/'inputs/physical-config.json')
        for key in ('fixed_poses','depletant_radius','reservoir_density','metadata'):
            require(config[key]==native_config[key],'Native observer physical law differs: '+key)
    labels=[];near_boundary=0
    for row in rows:
        geometry=contact.classify(row['pose'],capture_valid=row['capture_valid'],wall_valid=row['wall_valid'])
        near=geometry['near_core_boundary'];near_boundary+=int(near)
        expected_hard=row['capture_valid'] and row['wall_valid'] and geometry['core_disjoint']
        require(row['hard_valid']==expected_hard or near,'Independent atom hard predicate differs away from roundoff boundary')
        label=classifier.classify(row['pose']) if classifier and row['hard_valid'] else None
        if row['hard_valid']:
            require(row['depletion_contact']==geometry['exclusion_contact'],'Independent exclusion contact differs')
        native=bool(label['native_any']) if label else False
        valid=row['hard_valid'];inside=row['latent_density']['in_reference_ball']
        classes=dict(total=valid,inside_R4=valid and inside,outside_R4=valid and not inside,
                     exclusion_contact=valid and geometry['exclusion_contact'],unbound=valid and not geometry['exclusion_contact'])
        if classifier:
            classes.update(registered_native_entry=valid and native,
                contact_no_native_entry=valid and geometry['exclusion_contact'] and not native,
                unbound_no_native_entry=valid and not geometry['exclusion_contact'] and not native)
            require(sum(classes[k] for k in ('registered_native_entry','contact_no_native_entry','unbound_no_native_entry'))==int(valid),'Complete class partition fails')
        labels.append(dict(draw=row['draw'],classes=classes,geometry=geometry,native=label,near_core_boundary=near))
    estimates={}
    for name in labels[0]['classes']:
        selected=[label['classes'][name] for label in labels]
        logs=np.asarray([r['log_importance_weight'] if s else -np.inf for r,s in zip(rows,selected)])
        hard=np.asarray([r['log_hard_weight'] if s else -np.inf for r,s in zip(rows,selected)])
        pairs=np.asarray([[c['log_weight']-r['log_proposal_density'] for c in r['clouds']] if s else [-np.inf,-np.inf] for r,s in zip(rows,selected)])
        estimates[name]=dict(Qz=moments(logs),Q0=moments(hard),paired_noise=paired_noise(logs,pairs))
    log_close(summary['estimates']['total']['log_normalizer'],nullable_log(estimates['total']['Qz']['logQ']),'Total estimator differs')
    log_close(summary['estimates']['hard_total']['log_normalizer'],nullable_log(estimates['total']['Q0']['logQ']),'Hard estimator differs')
    ledger.recheck();out.mkdir(parents=True);provenance=out/'provenance';provenance.mkdir()
    for name,path in audit_sources.items():
        shutil.copy2(path,provenance/name)
        require(sha(provenance/name)==ledger.files[str(Path(path).resolve())],'Audit source changed while archiving')
    ledger.recheck()
    with (out/'labels.jsonl').open('w') as stream:
        for row in labels:stream.write(json.dumps(row,allow_nan=False)+'\n')
    result=dict(schema='full-vessel-latent-audit-v1',complete=True,population=str(root),manifest=manifest,
        density_audit=density_check,vessel_generation_audit=generation_check,
        all_vessel_generation_metadata_audit=generation_metadata_check,primitive_count_audit=count_check,wall_audit=wall_check,
        raw_sample_binding=dict(sha256=ledger.files[str(raw)],external_expected_sha256=None,
            scope='Actual raw bytes bound at audit start and rechecked afterward; no external pre-frozen raw hash supplied.'),
        executable_binding=dict(manifest_sha256=manifest['executable_sha256'],artifact_verified=False,
            scope='Campaign execution provenance must bind the executable artifact; this auditor verifies its archived source bundle, not a supplied binary.'),
        independently_checked_atom_poses=contact.counts['exact_core_contact_pose_checks'],
        geometry_audit=contact.report(),near_core_boundary_poses=near_boundary,
        native_binding=native_binding,estimates=estimates,source_sha256=ledger.files,
        scope='One population audit, not convergence or finite-system evidence. All unconditional attempts retained. Poisson count algebra is checked; exact thinning geometry and FP execution remain implementation obligations.')
    write(out/'analysis.json',result)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--population',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--native-definition',type=Path)
    args=parser.parse_args();result=audit(args.population,args.out,args.native_definition)
    print(json.dumps(dict(complete=result['complete'],density_audit=result['density_audit'])),flush=True)


if __name__=='__main__':main()
