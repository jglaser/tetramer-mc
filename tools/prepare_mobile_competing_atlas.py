#!/usr/bin/env python3
"""Freeze contact-centered proposal controls without fitting or physical sampling.

The reciprocal chart covariance is a first-order design choice. A Gaussian
under reciprocal-pose inversion is not claimed to remain Gaussian. Each
resulting Gaussian is independently normalized, and the production kernels
must continue to use the full mixture density in their Hastings correction.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import copy
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_mobile_competing_reference import competitor_model
from prepare_mobile_posterior_pilot import combine_models, validate_model, verify_density_identity
from prepare_shoulder_docking_benchmark import local_dependencies
from prepare_smc_normalizer_atlas import Density, arrays, read, relative_poses, sha, write

ROOT=Path(__file__).resolve().parents[1]
GEOMETRY=ROOT/'runs/mobile-competing-geometry-20260921'
REFERENCE=ROOT/'runs/mobile-competing-reference-preparation-20260921'
CAMPAIGN=ROOT/'runs/mobile-posterior-pilot-12x2000-20260921'
JOB='mobile3-dispersed-r01-c09'
SOURCE_MODEL_SHA='d0f3f82960c218ecbb0617b78412bb8072a3f33a726e128465d9418e0690af68'
FINAL_SNAPSHOT_SHA='4f74942180010463fbefb51c6b116fad675428c27533d6558e8bb0b87e1029b9'
NEW_SEED_BASE=117901010


def require(condition,message):
    if not condition:
        raise ValueError(message)


def pose(t,r):
    return dict(position=np.asarray(t).tolist(),orientation=Rotation.from_matrix(r).as_quat()[[3,0,1,2]].tolist())


def invert_pose(value):
    t,_,r=arrays([value])
    return pose(-r[0].T@t[0],r[0].T)


def cross_matrix(t):
    x,y,z=t
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def reciprocal_jacobian(t,r,ell):
    """Derivative at a chart center for left-Cayley rotations.

    Forward: t=t0+dt, R=C(a/ell) R0. Inverse coordinates use center
    (-R0.T t0,R0.T). Then a'=-R0.T a exactly, while
    dt'=-R0.T dt -(2/ell) R0.T [t0]_cross a + O(||(dt,a)||²).
    """
    require(math.isfinite(ell) and ell>0,'Invalid angular length')
    result=np.zeros((6,6))
    result[:3,:3]=result[3:,3:]=-r.T
    result[:3,3:]=-(2./ell)*r.T@cross_matrix(t)
    return result


def chart_pose(model,coordinates):
    anchor=model['anchors'][0]
    value=np.asarray(coordinates)
    q=np.r_[value[3:]/model['angular_length'],1.]
    r=Rotation.from_quat(q).as_matrix()@np.asarray(anchor['rotation'])
    return pose(np.asarray(anchor['position'])+value[:3],r)


def chart_coordinates(model,value):
    t,_,r=arrays([value])
    anchor=model['anchors'][0]
    q=Rotation.from_matrix(r[0]@np.asarray(anchor['rotation']).T).as_quat()
    require(abs(q[3])>1e-10,'Probe crosses the reciprocal chart seam')
    return np.r_[t[0]-anchor['position'],model['angular_length']*q[:3]/q[3]]


def reciprocal_model(forward):
    require(validate_model(forward,forward['shape_sha256'])==1,'Expected a single forward chart')
    require(forward['means']==[[0.]*6] and forward['weights']==[1.],'Forward chart must be centered')
    anchor=forward['anchors'][0]
    t,r=np.asarray(anchor['position']),np.asarray(anchor['rotation'])
    j=reciprocal_jacobian(t,r,forward['angular_length'])
    covariance=j@np.asarray(forward['covariances'][0])@j.T
    # Symmetrize only floating-point multiplication roundoff, without clipping
    # eigenvalues, adding a floor, or changing the linearized construction.
    symmetry_error=float(np.max(np.abs(covariance-covariance.T)))
    covariance=.5*(covariance+covariance.T)
    result=copy.deepcopy(forward)
    result['anchors']=[dict(position=(-r.T@t).tolist(),rotation=r.T.tolist())]
    result['covariances']=[covariance.tolist()]
    validate_model(result,result['shape_sha256'])
    return result,j,symmetry_error


def validate_reciprocal(forward,inverse,j):
    derivative_errors={}
    for epsilon in (1e-4,2e-5):
        numeric=np.zeros((6,6))
        for column in range(6):
            step=np.eye(6)[column]*epsilon
            hi=chart_coordinates(inverse,invert_pose(chart_pose(forward,step)))
            lo=chart_coordinates(inverse,invert_pose(chart_pose(forward,-step)))
            numeric[:,column]=(hi-lo)/(2*epsilon)
        error=float(np.max(np.abs(numeric-j)))
        require(error<2e-8,'Reciprocal coordinate Jacobian finite difference failed')
        derivative_errors[str(epsilon)]=error
    reciprocal_anchor=inverse['anchors'][0]
    ji=reciprocal_jacobian(np.asarray(reciprocal_anchor['position']),np.asarray(reciprocal_anchor['rotation']),inverse['angular_length'])
    require(np.max(np.abs(ji@j-np.eye(6)))<1e-12,'Reciprocal derivatives do not invert')
    center=chart_pose(forward,np.zeros(6))
    inverse_center=chart_pose(inverse,np.zeros(6))
    at_inverse=chart_coordinates(inverse,invert_pose(center))
    at_forward=chart_coordinates(forward,invert_pose(inverse_center))
    require(max(np.max(np.abs(at_inverse)),np.max(np.abs(at_forward)))<1e-12,'Reciprocal chart centers differ')
    require(abs(np.linalg.det(j)-1)<1e-12,'Reciprocal center Jacobian changed physical volume')
    probe=np.array([.11,-.07,.09,.02,-.04,.06])
    nonlinear_errors=[]
    for scale in (1.,.5,.25):
        actual=chart_coordinates(inverse,invert_pose(chart_pose(forward,scale*probe)))
        residual=actual-j@(scale*probe)
        nonlinear_errors.append(dict(scale=scale,translation_error_A=float(np.linalg.norm(residual[:3])),
            angular_coordinate_error=float(np.linalg.norm(residual[3:]))))
    require(nonlinear_errors[-1]['translation_error_A']>1e-9,'Probe must expose nonzero nonlinear translation correction')
    require(all(row['angular_coordinate_error']<1e-12 for row in nonlinear_errors),'Rotational reciprocal coordinates differ')
    require(nonlinear_errors[0]['translation_error_A']/nonlinear_errors[1]['translation_error_A']>3.9,
        'Reciprocal first-order residual did not scale quadratically')
    return dict(passed=True,finite_difference_maximum_absolute_errors=derivative_errors,
        center_inverse_coordinate_error=float(max(np.max(np.abs(at_inverse)),np.max(np.abs(at_forward)))),
        reciprocal_derivative_product_error=float(np.max(np.abs(ji@j-np.eye(6)))),
        determinant=float(np.linalg.det(j)),forward_to_reciprocal_jacobian=j.tolist(),
        symmetry_roundoff_note='Covariance symmetrization only removes matrix-product roundoff; no eigenvalue floor.',
        nonlinear_probe=probe.tolist(),nonlinear_residuals=nonlinear_errors,
        limitation='The inverse mean is the exact reciprocal pose; inverse covariance is linearized. It is not the exact pushforward of the forward Gaussian. Independent normalized Gaussian densities are used in the full proposal mixture.')


def prepare(out):
    out=Path(out).resolve()
    require(not out.exists(),'Use a fresh preparation directory')
    require(sha(GEOMETRY/'fixed-snapshot.json')==FINAL_SNAPSHOT_SHA,'Exact source snapshot changed')
    geometry,snapshot=read(GEOMETRY/'analysis.json'),read(GEOMETRY/'fixed-snapshot.json')
    for record in geometry['source'].values():
        require(sha(record['path'])==record['sha256'],'Frozen geometry source changed: '+record['path'])
    directory=CAMPAIGN/'runs'/JOB
    original_path=directory/'provenance/frozen-relative-model.json'
    require(sha(original_path)==SOURCE_MODEL_SHA,'Original 150-component proposal changed')
    original=read(original_path)
    shape=read(directory/'provenance/shape.json')
    shape_hash=sha(directory/'provenance/shape.json')
    require(validate_model(original,shape_hash)==150,'Require original 150 charts')
    final=snapshot['original_global']['poses']
    require(final==read(directory/'checkpoint.json')['poses'],'Checkpoint no longer matches exact snapshot')
    require(snapshot['body_assignment']==dict(A=2,B=1,mobile=0),'Wrong scaffold assignment')
    forward=competitor_model(final[0],final[2],original,translation_std=.25,angular_std_degrees=.25)
    require(forward==read(REFERENCE/'model-competitor-geometric.json'),'Geometric chart differs from finite-region reference chart')
    inverse,j,symmetry_error=reciprocal_model(forward)
    reciprocal_check=validate_reciprocal(forward,inverse,j)
    augmented,groups=combine_models([original,forward,inverse],[.8,.1,.1],shape_hash)
    require(augmented['anchors'][:150]==original['anchors'],'Old anchors changed')
    require(augmented['means'][:150]==original['means'],'Old means changed')
    require(augmented['covariances'][:150]==original['covariances'],'Old covariances changed')
    require(np.array_equal(np.asarray(augmented['weights'][:150]),.8*np.asarray(original['weights'])),'Old relative weights changed')
    density_identity=verify_density_identity(augmented,[original,forward,inverse],[.8,.1,.1],seed=117902017)
    centers=[chart_pose(model,np.zeros(6)) for model in (forward,inverse)]
    component_checks=[]
    for model in (forward,inverse):
        center=chart_pose(model,np.zeros(6)); lower=np.linalg.cholesky(model['covariances'][0])
        ell=model['angular_length']
        logdet=np.log(np.diag(lower)).sum()
        expected=-3*math.log(2*math.pi)-logdet+3*math.log(ell)+2*math.log(math.pi)
        observed=Density(model).evaluate([center])[0][0]
        require(abs(observed-expected)<2e-10,'Centered Gaussian/Haar density identity failed')
        component_checks.append(dict(minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(model['covariances'][0]).min()),
            covariance_condition_number=float(np.linalg.cond(model['covariances'][0])),
            centered_log_density=float(observed),centered_log_density_error=float(abs(observed-expected))))
    old_log=Density(original).evaluate(centers)[0]
    new_log=Density(augmented).evaluate(centers)[0]
    density_gain=[dict(direction=name,old_log_G=float(old),new_log_G=float(new),log_density_gain=float(new-old))
        for name,old,new in zip(('body0-relative-body2','body2-relative-body0'),old_log,new_log)]
    provenance=out/'provenance';provenance.mkdir(parents=True)
    sources={'shape.json':directory/'provenance/shape.json','monomer-shape.json':CAMPAIGN/'provenance/monomer-shape.json',
        'native-pair-motifs.json':CAMPAIGN/'provenance/native-pair-motifs.json',
        'source-model.json':original_path,'source-effective-config.json':directory/'config.json',
        'source-submitted-config.json':CAMPAIGN/'configs'/f'{JOB}.json','source-checkpoint.json':directory/'checkpoint.json',
        'geometry-analysis.json':GEOMETRY/'analysis.json','fixed-snapshot.json':GEOMETRY/'fixed-snapshot.json',
        'reference-competitor-model.json':REFERENCE/'model-competitor-geometric.json'}
    sources.update(local_dependencies([Path(__file__)]))
    for name,source in sources.items():shutil.copy2(source,provenance/name)
    shutil.copy2(original_path,out/'model-legacy.json')
    write(out/'model-contact-forward.json',forward)
    write(out/'model-contact-reciprocal.json',inverse)
    write(out/'model-augmented.json',augmented)
    original_config=read(provenance/'source-submitted-config.json')
    require(original_config['fixed_body_indices']==[] and original_config['seed_labels']==[],'Source was not all mobile')
    configs=out/'configs';configs.mkdir()
    controls=[]
    for atlas in ('legacy','augmented'):
        for mode,correlation in (('c0',0.),('c09',.9)):
            cfg=copy.deepcopy(original_config)
            cfg.update(shape=str(provenance/'shape.json'),monomer_shape=str(provenance/'monomer-shape.json'),
                initial_poses=copy.deepcopy(final),seed=NEW_SEED_BASE+1009*len(controls),
                frozen_posterior=dict(probability=.5,correlation=correlation))
            cfg['metadata']=dict(start='exact-observed-final-snapshot',mode=mode,atlas=atlas,
                source_job=JOB,source_sweep=2000,source_snapshot_sha256=FINAL_SNAPSHOT_SHA,
                native_pair_motifs=str(provenance/'native-pair-motifs.json'),
                native_informed_proposal=True,preparation_equilibrated=False,training_feedback=False,
                added_contact_charts=(2 if atlas=='augmented' else 0),
                scope='All three tetramers mobile at the exact observed terminal configuration. The legacy atlas is native-informed; added chart centers use observed pair geometry only. Every proposal density remains frozen, with full-mixture Hastings correction. No capture region or fixed-scaffold target is introduced.')
            require(cfg['boundary']==original_config['boundary'],'Physical spherical wall changed')
            for key in ('global_probability','learned_uniform_weight','local_translation_std_A','local_small_angle_std_degrees',
                        'gca_probability','center_shift_probability','depletant_radius','reservoir_density','poisson_lambda_ratio','endpoint_gate'):
                require(cfg[key]==original_config[key],'Physical/sampling control changed: '+key)
            require(cfg['global_probability']==.5 and cfg['frozen_posterior']['probability']==.5,'Unmatched move fractions')
            require(cfg['initial_poses']==snapshot['original_global']['poses'],'A source pose was changed')
            require(not cfg['fixed_body_indices'] and not cfg['seed_labels'],'A body was fixed')
            name=f'{atlas}-{mode}'
            cfgpath=configs/f'{name}.json';write(cfgpath,cfg)
            model_path=out/f'model-{atlas}.json'
            controls.append(dict(id=name,atlas=atlas,mode=mode,correlation=correlation,seed=cfg['seed'],
                config=str(cfgpath),config_sha256=sha(cfgpath),model=str(model_path),model_sha256=sha(model_path)))
    checks=dict(passed=True,reciprocal=reciprocal_check,covariance_symmetry_error_before_symmetrization=symmetry_error,
        component_checks=component_checks,full_density_identity=density_identity,center_density_gain=density_gain,
        original_component_preservation=True,all_mobile_same_exact_snapshot=True,
        physical_updates=0,Poisson_clouds=0,production_launched=False,
        scope='Proposal-map and density checks only. Higher proposal density is not an acceptance or sampling gain. No physical weights or fitted covariance enter these charts.')
    write(out/'preflight.json',checks)
    plan=dict(schema='mobile-competing-atlas-preparation-v1',production_launched=False,
        source_job=JOB,source_snapshot_sha256=FINAL_SNAPSHOT_SHA,legacy_model_sha256=SOURCE_MODEL_SHA,
        controls=controls,groups=groups,shapes=dict(tetramer_sha256=shape_hash,monomer_sha256=sha(provenance/'monomer-shape.json')),
        chart_design=dict(translation_std_A=.25,angular_coordinate_scale_degrees=.25,
            forward_component=150,reciprocal_component=151,geometric_covariance_fitting=False,
            forward_center='Exact body0 relative to body2 at source final snapshot.',
            reciprocal_center='Exact reciprocal body2 relative to body0; covariance J Sigma J^T uses the first-order reciprocal-pose derivative.',
            covariance_limitation='Reciprocal Gaussian is a new independently normalized proposal component, not the exact nonlinear pushforward of the forward Gaussian.',
            chart_weights=[.1,.1],legacy_retained_mass=.8),
        balance='All atlas parameters are frozen before physical sampling. Existing capture uses its full uniform-plus-G density; existing posterior involution uses the full physical Gaussian mixture G in its correction. Reciprocal component construction affects efficiency only and supplies no exemption from the exact forward/reverse density calculation.',
        matched_control='Four inert configs share the same exact final physical poses, wall, bath, local steps, branch fractions, GCA and shift. Correlation c0 vs c09 and legacy vs augmented atlas are the only algorithm differences; seeds are distinct. Parent must specify and freeze run lengths, repetitions, observables, executable and output paths before production.',
        physical_scope='Native-informed, all-mobile preparation control. Added contact charts are based on observed pair geometry; they do not make the full atlas geometry-only. This is not independent validation of global equilibrium, a crystal-assembly result, or a completed exchange benchmark.',
        input_sha256={name:sha(provenance/name) for name in sources},
        outputs_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*.json')) if not p.is_relative_to(provenance)})
    write(out/'plan.json',plan)
    write(out/'freeze.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(__import__('json').dumps(dict(output=str(out),plan_sha256=sha(out/'plan.json'),
        components=152,preflight_passed=True,physical_updates=0,controls=len(controls),density_gain=density_gain),indent=2))
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs/mobile-competing-atlas-preparation-20260921')
    prepare(parser.parse_args().out)
