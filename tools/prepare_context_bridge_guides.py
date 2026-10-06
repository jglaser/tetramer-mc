"""Freeze a source-to-competing-cage geometry probe without querying geometry.

The bridge is a declared Gaussian proposal heuristic, not an exact Gaussian
pushforward through a nonlinear chart change or an assumed hard-free path.
Every resulting chart retains the existing exact density/Jacobian convention.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from source_guide_reference import SourceDensity, compose, pose, relative, rotation

OLD_COMPONENTS = ('full', 'diagonal', 'broad_full', 'cage0', 'cage1')
ALPHAS = (0., 1/3, 2/3, 1.)
WIDTHS = (1., 4.)
BRIDGES = tuple(f'bridge_a{i}_b{int(b)}' for i in range(4) for b in WIDTHS)
COMPONENTS = OLD_COMPONENTS+BRIDGES
DENSITY_COMPONENTS = ('uniform', 'context', *COMPONENTS)
RUNNER_MIXTURE = (.5, .25, .25)
ALONG_SIGMA = 1/6
POPULATIONS = 4
ATTEMPTS = 2048
ALLOCATIONS = {
    'baseline': dict(zip(COMPONENTS, (384,384,256,512,512, *([0]*8)))),
    'bridge': dict(zip(COMPONENTS, (192,192,128,256,256, *([128]*8)))),
}
PHYSICAL = dict(depletant_radius=1.5, activity=.035, **{'lambda':2.24})
DIAGNOSTICS = dict(primary_source_intersection_counts=list(range(8,16)),
    secondary_source_intersection_counts=list(range(5,16)),
    source_token_denominator=16, original_source_chart_tail_bins=[24.,48.,96.,None],
    all_neighbor_sets_recorded=True, exact_A_neighbors=[16,217],
    minimum_populations_with_primary_hits=3, minimum_total_primary_hits=20,
    population_scope='bridge arm; any neighbor set and exact-A subset reported separately',
    admission_scope='Geometry coverage only; neither physical-weight convergence nor native registry or assembly.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def skew(u):
    x,y,z = np.asarray(u,float)
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def recenter_jacobian(mean, ell):
    """Derivative at the old mean of Cayley(u) Cayley(u_mean)^-1."""
    mean = np.asarray(mean,float)
    require(mean.shape==(6,) and np.isfinite(mean).all()
            and math.isfinite(ell) and ell>0,'Invalid chart mean or angular length')
    u = mean[3:]/ell
    result = np.eye(6)
    result[3:,3:] = (np.eye(3)+skew(u))/(1+float(u@u))
    return result


def recenter_endpoint(spec, original_source, anchor):
    density = SourceDensity(spec, original_source, anchor)
    center = density.decode(np.zeros(6))
    jacobian = recenter_jacobian(density.mean,density.ell)
    covariance = jacobian@density.covariance@jacobian.T
    covariance = .5*(covariance+covariance.T)
    np.linalg.cholesky(covariance)
    return dict(center=center, covariance=covariance, jacobian=jacobian,
                angular_length=density.ell, original_mean=density.mean)


def bridge_specs(full_spec, cage_spec, original_source, anchor, provenance):
    require(isinstance(provenance,str) and bool(provenance.strip()),'Missing bridge provenance')
    endpoints = [recenter_endpoint(spec,original_source,anchor) for spec in (full_spec,cage_spec)]
    ell = endpoints[0]['angular_length']
    require(endpoints[1]['angular_length']==ell,'Different endpoint angular lengths')
    left,right = [relative(e['center'],anchor) for e in endpoints]
    p0,p1 = np.asarray(left['position']),np.asarray(right['position'])
    r0,r1 = rotation(left),rotation(right)
    omega = Rotation.from_matrix(r1@r0.T).as_rotvec()
    # An exactly half-turn bridge has no unique shortest arc; reject ambiguity.
    require(float(np.linalg.norm(omega))<math.pi-1e-10,'Ambiguous half-turn bridge')
    tangent = np.r_[p1-p0,.5*ell*omega]
    along = ALONG_SIGMA**2*np.outer(tangent,tangent)
    result = {}
    for i,alpha in enumerate(ALPHAS):
        center_local = pose((1-alpha)*p0+alpha*p1,
                            Rotation.from_rotvec(alpha*omega).as_matrix()@r0)
        center_world = compose(anchor,center_local)
        interpolation = (1-alpha)*endpoints[0]['covariance']+alpha*endpoints[1]['covariance']
        for width in WIDTHS:
            name = f'bridge_a{i}_b{int(width)}'
            covariance = width**2*interpolation+along
            covariance = .5*(covariance+covariance.T)
            np.linalg.cholesky(covariance)
            spec = dict(angular_length=ell,covariance=covariance.tolist(),
                explicit_gaussian=dict(schema='source-gaussian-v1',mean=[0.]*6,
                    provenance=f'{provenance}; alpha_index={i}; std_width={width:g}'),
                chart_center=dict(schema='source-chart-center-v1',frame='saved-spherical-center',
                    pose=center_world,provenance=provenance))
            result[name] = dict(schema='context-bridge-frozen-guide-v1',
                alpha_index=i,alpha=alpha,standard_deviation_width=width,
                along_parameter_sigma=ALONG_SIGMA,source_chart=spec,
                covariance_construction='b^2*((1-alpha)*Sigma_F_recentered+alpha*Sigma_C0_recentered)+(1/6)^2*v*v^T',
                tangent_anchor_coordinates=tangent.tolist(),
                scope='Frozen normalized proposal heuristic only; center feasibility and intermediate contacts are not assumed.')
    metadata = dict(endpoint_names=['full','cage0'],
        endpoint_centers=[e['center'] for e in endpoints],
        endpoint_recenter_jacobians=[e['jacobian'].tolist() for e in endpoints],
        endpoint_recentered_covariances=[e['covariance'].tolist() for e in endpoints],
        original_endpoint_means=[e['original_mean'].tolist() for e in endpoints],
        anchor_frame_rotation_vector=omega.tolist(),tangent_anchor_coordinates=tangent.tolist(),
        translation_separation=float(np.linalg.norm(p1-p0)),
        rotation_separation_degrees=float(np.linalg.norm(omega))*180/math.pi,
        covariance_transport_scope='First-order tangent covariance at each decoded mean; not exact nonlinear Gaussian transport. New normalized Gaussian proposals are defined by these matrices.')
    return result,metadata


def density_coefficients(arm):
    require(arm in ALLOCATIONS,'Unknown arm')
    counts = ALLOCATIONS[arm]
    require(sum(counts.values())==ATTEMPTS,'Changed unconditional allocation')
    return np.array([.5,.25]+[.25*counts[c]/ATTEMPTS for c in COMPONENTS])


def log_mixture(arm, component_logs):
    values = np.asarray(component_logs,float)
    require(values.shape==(len(DENSITY_COMPONENTS),) and not np.isnan(values).any()
            and not np.isposinf(values).any(),'Invalid component log density')
    terms = [math.log(p)+v for p,v in zip(density_coefficients(arm),values) if p>0 and v!=-math.inf]
    if not terms:
        return -math.inf
    high = max(terms)
    return high+math.log(math.fsum(math.exp(v-high) for v in terms))


def future_populations():
    return [dict(comparison_arm=arm,population_index=i,denominator=ATTEMPTS,total_attempts=ATTEMPTS,
        strata=[dict(component=c,draws=n,deterministic_fraction=n/ATTEMPTS) for c,n in counts.items() if n])
        for arm,counts in ALLOCATIONS.items() for i in range(POPULATIONS)]


def validate_recipe(recipe):
    require(recipe['schema']=='context-bridge-source-recipe-v1'
        and recipe['components']==list(COMPONENTS) and recipe['component_draws']==ALLOCATIONS
        and recipe['arms']==list(ALLOCATIONS) and recipe['populations_per_arm']==4
        and recipe['attempts_per_population']==2048 and recipe['runner_mixture']==list(RUNNER_MIXTURE)
        and recipe['alphas']==list(ALPHAS) and recipe['standard_deviation_widths']==list(WIDTHS)
        and recipe['along_parameter_sigma']==ALONG_SIGMA and recipe['physical_conditions']==PHYSICAL
        and recipe['mode']=='geometry' and recipe['cloud_draws']==0
        and recipe['diagnostics']==DIAGNOSTICS and recipe['reuse_old_draws'] is False
        and recipe['seeds_materialized'] is False and recipe['heldout_tuning_allowed'] is False,
        'Changed prospective bridge recipe')


def validate_baseline(old):
    require(old['schema']=='context-multicage-source-mixture-v1' and old['complete'] and old['passed']
        and old['runner_mixture']==list(RUNNER_MIXTURE) and old['physical_conditions']==PHYSICAL
        and tuple(old['guides'])==OLD_COMPONENTS,'Changed or incomplete baseline')
    expected = np.r_[old['effective_density_coefficients']['multicage'],np.zeros(8)]
    require(np.array_equal(expected,density_coefficients('baseline')),'Baseline density differs')
    regions = old['source_inputs']['regions']
    require(regions['a_neighbors']==[16,217] and regions['b_neighbors']==[16,56]
        and regions['secondary_label']==217 and len(regions['source_secondary_tokens'])==16
        and regions['inclusion_boundaries']==[0.,.25,.5,.75,1.],'Changed source contact classifier')


def local_sources(start):
    """Read-only static local-import closure; no source modules are imported."""
    directory = Path(start).resolve().parent
    pending = [Path(start).name];found = {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        source = directory/name
        found[name] = source
        for node in ast.walk(ast.parse(source.read_text())):
            modules = ([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                       [x.name for x in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                child = module.split('.')[0]+'.py'
                if (directory/child).is_file():
                    pending.append(child)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--expected-recipe-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args();bindings = {}
    def bind(asset,decode=True):
        path = Path(asset['path']).resolve();digest = sha(path)
        require(digest==asset['sha256'],'Changed metadata/source input: '+str(path))
        require(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting source binding')
        bindings[str(path)] = digest
        return read(path) if decode else None
    recipe = bind(dict(path=str(args.recipe.resolve()),sha256=args.expected_recipe_sha256))
    validate_recipe(recipe)
    old = bind(recipe['old_mixture_manifest']);validate_baseline(old)
    for path,digest in old['input_sha256'].items():
        bind(dict(path=path,sha256=digest),False)
    guides = {name:bind(old['guides'][name]) for name in OLD_COMPONENTS}
    physical = {name:bind(asset) for name,asset in old['source_frame_assets'].items()}
    source = physical['source_state'];context = physical['fixed_context']
    require(source['moving_label']==77 and source['anchor_label']==context['anchor_label']==16
            and context['excluded_moving_labels']==[77] and len(context['bodies'])==263
            and source['anchor_pose']==next(b['pose'] for b in context['bodies'] if b['label']==16),
            'Changed physical source frame or neighborhood')
    validation = bind(recipe['validation_report']);validation_plan = bind(recipe['validation_plan'])
    summary = bind(recipe['validation_summary']);drain = bind(recipe['validation_host_drained'])
    require(validation['complete'] and validation['passed'] and validation['tests']==recipe['validation_tests']
        and validation['new_poses']==validation['new_clouds']==validation['protein_queries']==0
        and summary['complete'] and summary['passed'] and summary['failure'] is None
        and summary['active'] is None and not summary['unstarted'] and len(summary['completed'])==1
        and summary['plan_sha256']==recipe['validation_plan']['sha256']
        and summary['completed'][0]['terminal']==recipe['validation_report']
        and summary['completed'][0]['success'] and summary['completed'][0]['child_drained']
        and summary['completed'][0]['returncode']==0
        and drain['complete'] and drain['passed']
        and drain['execution_plan_sha256']==recipe['validation_plan']['sha256']
        and drain['execution_receipt_sha256']==recipe['validation_summary']['sha256']
        and len(drain['owned_groups'])==2
        and all(not g['same_process'] and not g['group_exists'] for g in drain['owned_groups']),
        'Incomplete or unrelated bridge validation')
    source_assets = local_sources(__file__)
    for path in source_assets.values():
        require(validation_plan['files'].get(str(path))==sha(path),'Source absent from passed validation: '+str(path))
        bind(dict(path=str(path),sha256=sha(path)),False)
    bridges,construction = bridge_specs(guides['full']['source_chart'],guides['cage0']['source_chart'],
        source['pose'],source['anchor_pose'],f'Frozen bridge recipe {args.expected_recipe_sha256}')
    residual = density_coefficients('bridge')-.5*density_coefficients('baseline')
    require(np.all(residual>=0),'Lost pointwise defensive bound')
    require(not args.out.exists(),'Fresh metadata-only output required')
    args.out.mkdir();(args.out/'guides').mkdir();(args.out/'code').mkdir()
    frozen = {}
    for name in COMPONENTS:
        path = args.out/'guides'/f'{name}.json'
        if name in OLD_COMPONENTS:
            shutil.copyfile(old['guides'][name]['path'],path)
            require(sha(path)==old['guides'][name]['sha256'],'Old guide snapshot changed')
        else:
            save(path,bridges[name])
        frozen[name] = dict(path=str(path.resolve()),sha256=sha(path))
    frozen_sources = {}
    for name,source_path in source_assets.items():
        target = args.out/'code'/name;shutil.copyfile(source_path,target)
        require(sha(target)==sha(source_path),'Source snapshot changed')
        frozen_sources[str(target.resolve())] = sha(target)
    manifest = dict(schema='context-bridge-source-mixture-v1',complete=True,passed=True,
        source_sha256=sha(__file__),recipe=dict(path=str(args.recipe.resolve()),sha256=args.expected_recipe_sha256),
        input_sha256=bindings,source_snapshot_sha256=frozen_sources,
        guides=frozen,original_guides=old['guides'],old_mixture_manifest=recipe['old_mixture_manifest'],
        density_component_order=list(DENSITY_COMPONENTS),runner_schema='context-source-guide-v1',
        runner_mixture=list(RUNNER_MIXTURE),
        effective_density_coefficients={arm:density_coefficients(arm).tolist() for arm in ALLOCATIONS},
        mixture_identity='Q_bridge=.5*Q_current_multicage+.5*mean_8(r_bridge); r_j=.5U+.25G+.25g_j.',
        mixture_lower_bound=dict(fraction=.5,baseline_arm='baseline',new_arm='bridge',
            nonnegative_remainder_coefficients=residual.tolist(),maximum_existing_importance_weight_inflation=2.,
            scope='Pointwise density bound only; not a promise of realized variance, ESS or CPU improvement.'),
        density_measure='Translation volume times normalized rotational Haar; chart Jacobian enters each g once; no additional mixture Jacobian.',
        source_inputs=old['source_inputs'],source_frame_assets=old['source_frame_assets'],envelope=old['envelope'],
        existing_runner=old['existing_runner'],source_bundle=old['source_bundle'],
        physical_conditions=PHYSICAL,mode='geometry',physical_weight_status='not_estimated',
        construction=construction,alphas=list(ALPHAS),standard_deviation_widths=list(WIDTHS),
        along_parameter_sigma=ALONG_SIGMA,diagnostics=DIAGNOSTICS,
        future_populations=future_populations(),total_future_attempts=16384,future_stratum_jobs=72,
        independent_geometry_panel=dict(size=512,stride=32,scope='Predeclared all-attempt deterministic panel; no result-dependent redraw.'),
        no_outcome_based_refit=True,heldout_tuning_allowed=False,old_guides_unchanged=True,
        new_pose_draws=0,geometry_queries=0,cloud_draws=0,new_fit_samples=0,
        seeds_materialized=False,execution_materialized=False,assembly_kernel_changed=False,
        scope='Historical500uM frozen263-body geometry probe at unchanged rd1.5/z.035 inputs; source-contact signatures are not native registry. This cannot decide finite-system assembly near106.8uM.')
    save(args.out/'mixture-manifest.json',manifest)
    for path,digest in bindings.items():
        require(sha(path)==digest,'Input changed during metadata preparation')
    report = dict(complete=True,passed=True,manifest=dict(path=str((args.out/'mixture-manifest.json').resolve()),
        sha256=sha(args.out/'mixture-manifest.json')),guides=frozen,new_poses=0,new_geometry_queries=0,
        new_clouds=0,fitted_samples=0,seeds_materialized=False,execution_materialized=False)
    save(args.out/'report.json',report);print(json.dumps(report))


if __name__=='__main__':
    main()
