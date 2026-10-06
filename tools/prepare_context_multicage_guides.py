"""Freeze a five-chart importance comparison without generating samples.

Every stratum uses r_j=.5 U+.25 G+.25 g_j. Baseline is the existing broadened
mixture; multicage retains half that law and adds the two frozen cage laws.
The complete arm density, not the generating stratum density, divides weights.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from prepare_context_broadened_guides import validate_chart, validate_controls

COMPONENTS=('full','diagonal','broad_full','cage0','cage1')
DENSITY_COMPONENTS=('uniform','context',*COMPONENTS)
RUNNER_MIXTURE=(.5,.25,.25)
ALLOCATIONS={
    'baseline':dict(full=1536,diagonal=1536,broad_full=1024,cage0=0,cage1=0),
    'multicage':dict(full=768,diagonal=768,broad_full=512,cage0=1024,cage1=1024),
}
POPULATIONS=4
ATTEMPTS=4096
PHYSICAL={'depletant_radius':1.5,'activity':.035,'lambda':2.24}
CLOUD_RULE=dict(clouds_per_valid_pose=2,intensity=2.24,
    primary_estimator='pooled-count-rao-blackwell',secondary_estimator='arithmetic-pair-positive-weights',
    primary_log_formula='z*L+(K0+K1)*log1p(z/(2*lambda))',
    common_certified_lower_volume=True,independent_equal_intensity_clouds=True,
    score_every_valid_pose=True,zero_volume_clouds_audited=True,
    retain_hard_invalid_zeros=True,population_denominator=4096,
    maximum_cloud_count=65536,new_clouds_only=True)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def save(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def density_coefficients(arm):
    require(arm in ALLOCATIONS,'Unknown arm')
    counts=ALLOCATIONS[arm]
    require(sum(counts.values())==ATTEMPTS,'Changed unconditional allocation')
    return np.array([.5,.25]+[.25*counts[c]/ATTEMPTS for c in COMPONENTS])


def log_mixture(arm,component_logs):
    """Component densities use the same physical measure; no extra Jacobian."""
    logs=np.asarray(component_logs,float)
    require(logs.shape==(7,) and not np.isnan(logs).any() and not np.isposinf(logs).any(),
        'Invalid component log density')
    terms=[math.log(p)+x for p,x in zip(density_coefficients(arm),logs) if p>0 and x!=-math.inf]
    if not terms:return -math.inf
    high=max(terms)
    return high+math.log(math.fsum(math.exp(x-high) for x in terms))


def future_populations():
    return [dict(comparison_arm=arm,population_index=population,denominator=ATTEMPTS,
        total_attempts=ATTEMPTS,strata=[dict(component=c,draws=n,deterministic_fraction=n/ATTEMPTS)
            for c,n in counts.items() if n])
        for arm,counts in ALLOCATIONS.items() for population in range(POPULATIONS)]


def validate_center(center):
    require(set(center)=={'schema','frame','pose','provenance'}
        and center['schema']=='source-chart-center-v1' and center['frame']=='saved-spherical-center'
        and isinstance(center['provenance'],str) and center['provenance'].strip(),
        'Invalid independent chart-center metadata')
    pose=center['pose'];q=np.asarray(pose['orientation'],float);t=np.asarray(pose['position'],float)
    require(set(pose)=={'position','orientation'} and q.shape==(4,) and t.shape==(3,)
        and np.isfinite(q).all() and np.isfinite(t).all() and abs(float(q@q)-1)<2e-10,
        'Invalid frozen chart-center pose')


def validate_cage(guide,cage_id,old_manifest):
    require(guide['schema']=='competing-cage-frozen-guide-v1' and guide['cage_id']==cage_id
        and guide['heldout_fit_samples']==0 and guide['normalized_gaussian'] is True,
        'Changed frozen cage fit identity')
    fit=guide['fit_rule']
    require(fit['training_streams']==[0,1] and fit['heldout_streams']==[2,3]
        and fit['phase']=='production' and fit['bandwidth_multiplier']==1.
        and fit['ridge_absolute_scaled']==1e-10 and fit['ridge_trace_factor']==1e-6
        and fit['gaussian_components_per_declared_cage']==1 and fit['heldout_parameter_updates']==0,
        'Changed predeclared fit rule')
    require(guide['original_source_state']==old_manifest['source_frame_assets']['source_state'],
        'Chart center must not replace the physical source reference')
    ref=guide['patch_reference']
    require(ref['unchanged_during_training'] is True and ref['patch_map']==old_manifest['source_inputs']['patch_map']
        and ref['definitions']==old_manifest['source_inputs']['regions'],'Changed original contact regions')
    chart=copy.deepcopy(guide['source_chart']);center=chart.pop('chart_center')
    validate_center(center);validate_chart(chart)
    return center


def validate_reference_state(old_manifest,source,context,invocation,model):
    require(source['moving_label']==77 and source['anchor_label']==context['anchor_label']==16
        and source['boundary']=='spherical' and context['excluded_moving_labels']==[77]
        and len(context['bodies'])==263
        and source['anchor_pose']==next(b['pose'] for b in context['bodies'] if b['label']==16),
        'Changed physical frame or fixed neighborhood')
    require(invocation['depletant_radius']==1.5 and invocation['reservoir_density']==.035
        and invocation['poisson_lambda_ratio']==64. and invocation['expected_fixed_body_count']==263,
        'Changed physical model')
    for name,record in old_manifest['source_frame_assets'].items():
        require(invocation[name]==record['path'] and invocation['expected_sha256'][name]==record['sha256'],
            'Changed original physical input binding')
    require(invocation['expected_sha256']['model']==old_manifest['source_inputs']['model']['sha256'],
        'Changed common context proposal/model')
    return float(model.get('base_model',model)['angular_length'])


def validate_recipe(recipe):
    require(recipe['schema']=='context-multicage-source-recipe-v1'
        and recipe['arms']==list(ALLOCATIONS) and recipe['components']==list(COMPONENTS)
        and recipe['component_draws']==ALLOCATIONS and recipe['populations_per_arm']==POPULATIONS
        and recipe['attempts_per_population']==ATTEMPTS and recipe['runner_mixture']==list(RUNNER_MIXTURE)
        and recipe['physical_conditions']==PHYSICAL and recipe['physical_cloud_allocation']==CLOUD_RULE
        and recipe['reuse_old_draws'] is False and recipe['seeds_materialized'] is False
        and recipe['heldout_tuning_allowed'] is False,'Changed prospective recipe')


def validate_fit_completion(recipe,summary,drain):
    require(summary['complete'] and summary['passed'] and not summary['unstarted']
        and summary['active'] is None and summary['failure'] is None
        and summary['plan_sha256']==recipe['fit_execution_plan']['sha256']
        and len(summary['completed'])==2
        and all(j['success'] and j['child_drained'] and j['returncode']==0 for j in summary['completed'])
        and summary['completed'][-1]['terminal']==recipe['fit_report'],
        'Incomplete or unrelated frozen fit execution')
    require(drain['complete'] and drain['passed']
        and drain['report_sha256']==recipe['fit_report']['sha256']
        and drain['protocol_sha256']==recipe['fit_protocol']['sha256']
        and drain['execution_plan_sha256']==recipe['fit_execution_plan']['sha256']
        and drain['execution_summary_sha256']==recipe['fit_execution_summary']['sha256'],
        'Fit drain belongs to another execution')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True);parser.add_argument('--expected-recipe-sha256',required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    bindings={}
    def bind(record,decode=True):
        path=Path(record['path']).resolve();digest=sha(path)
        require(digest==record['sha256'],'Changed input: '+str(path));bindings[str(path)]=digest
        return read(path) if decode else None
    recipe=bind(dict(path=str(args.recipe),sha256=args.expected_recipe_sha256));validate_recipe(recipe)
    old=bind(recipe['old_mixture_manifest'])
    require(old['schema']=='context-broadened-source-mixture-v1' and old['complete'] and old['passed']
        and old['runner_mixture']==list(RUNNER_MIXTURE) and old['physical_conditions']==PHYSICAL,
        'Incomplete or changed old law')
    old_coeff=old['effective_density_coefficients']['broadened']
    require(np.array_equal(np.array(old_coeff+[0.,0.]),density_coefficients('baseline')),
        'Baseline must reproduce the existing broadened density exactly')
    require({c:recipe['guides'][c] for c in COMPONENTS[:3]}==old['guides'],'Old guide assets changed')
    guides={c:bind(recipe['guides'][c]) for c in COMPONENTS}
    validate_controls(guides['full'],guides['diagonal']);validate_chart(guides['broad_full']['source_chart'])
    broad=guides['broad_full'];full=guides['full']['source_chart']
    require(broad['derived_from']==old['guides']['full'] and broad['covariance_scale']==4.
        and broad['source_chart']['explicit_gaussian']['mean']==full['explicit_gaussian']['mean']
        and np.array_equal(np.asarray(broad['source_chart']['covariance']),4*np.asarray(full['covariance'])),
        'Changed old broadening transform')
    for a in old['source_inputs'].values():
        if isinstance(a,dict) and set(a)=={'path','sha256'}:bind(a)
    physical={n:bind(a) for n,a in old['source_frame_assets'].items()}
    invocation=bind(old['source_inputs']['invocation_config']);model=bind(old['source_inputs']['model'])
    ell=validate_reference_state(old,physical['source_state'],physical['fixed_context'],invocation,model)
    fit=bind(recipe['fit_report']);fit_protocol=bind(recipe['fit_protocol']);fit_drain=bind(recipe['fit_host_drained'])
    bind(recipe['fit_execution_plan']);fit_summary=bind(recipe['fit_execution_summary'])
    validate_fit_completion(recipe,fit_summary,fit_drain)
    require(fit['complete'] and fit['passed'] and fit_drain['complete'] and fit_drain['passed']
        and fit['protocol_sha256']==recipe['fit_protocol']['sha256']
        and fit['heldout_fit_samples']==0 and fit['new_poses']==fit['new_geometry_queries']==fit['new_clouds']==0,
        'Incomplete frozen fit provenance')
    require(fit_protocol['original_source_state']==old['source_frame_assets']['source_state']
        and fit_protocol['context']==old['source_frame_assets']['fixed_context'],
        'Cage fit used different physical references')
    for cage_id in range(2):
        name=f'cage{cage_id}';guide=guides[name];center=validate_cage(guide,cage_id,old)
        cage=fit['cages'][cage_id]
        require(cage['cage_id']==cage_id and cage['guide']==recipe['guides'][name]
            and center==cage['chart_center'] and guide['protocol_sha256']==recipe['fit_protocol']['sha256']
            and guide['source_chart']['explicit_gaussian']['mean']==cage['fit']['mean']
            and guide['source_chart']['covariance']==cage['fit']['covariance'],
            'Guide is not the untouched frozen fit')
        reference=bind(guide['patch_reference']['regions'])
        require(reference['inputs']['regions']==old['source_inputs']['regions']
            and reference['inputs']['patch_map']==old['source_inputs']['patch_map'],'Changed region reference')
    require(all(g['source_chart']['angular_length']==ell for g in guides.values()),'Changed chart angular length')
    runner=recipe['runner'];build=bind(runner['build_manifest']);receipt=bind(runner['build_receipt']);drain=bind(runner['build_host_drain'])
    bind(runner['binary'],False);bind(runner['source_bundle'])
    require(receipt['complete'] and receipt['passed'] and receipt['source_unchanged'] and receipt['protected_unchanged']
        and receipt['manifest_sha256']==runner['build_manifest']['sha256']
        and receipt['binaries_sha256'][runner['binary']['path']]==runner['binary']['sha256']
        and drain['complete'] and drain['passed']
        and drain['receipt_sha256']==runner['build_receipt']['sha256']
        and drain['manifest_sha256']==runner['build_manifest']['sha256']
        and drain['source_bundle_sha256']==runner['source_bundle']['sha256'],
        'Unvalidated or unrelated chart-center runner/drain')
    build_root=Path(runner['build_manifest']['path']).parent
    for name,digest in build['sources_sha256'].items():bind(dict(path=str(build_root/'source'/name),sha256=digest),False)
    residual=density_coefficients('multicage')-.5*density_coefficients('baseline')
    require(np.array_equal(residual,np.array([.25,.125,0.,0.,0.,.0625,.0625])),
        'Pointwise half-baseline proof changed')
    require(not args.out.exists(),'Fresh metadata-only output directory required');args.out.mkdir();(args.out/'guides').mkdir()
    frozen={}
    for name,a in recipe['guides'].items():
        path=args.out/'guides'/f'{name}.json';shutil.copyfile(a['path'],path)
        require(sha(path)==a['sha256'],'Guide snapshot differs');frozen[name]=dict(path=str(path.resolve()),sha256=a['sha256'])
    manifest=dict(schema='context-multicage-source-mixture-v1',complete=True,passed=True,
        source_sha256=sha(__file__),recipe=dict(path=str(args.recipe.resolve()),sha256=args.expected_recipe_sha256),
        input_sha256=bindings,guides=frozen,original_guides=recipe['guides'],
        density_component_order=list(DENSITY_COMPONENTS),runner_schema='context-source-guide-v1',
        runner_mixture=list(RUNNER_MIXTURE),effective_density_coefficients={a:density_coefficients(a).tolist() for a in ALLOCATIONS},
        mixture_identity='Q_new=.5*Q_old_broadened+.25*r_cage0+.25*r_cage1; r_j=.5U+.25G+.25g_j.',
        mixture_lower_bound=dict(fraction=.5,baseline_arm='baseline',new_arm='multicage',
            nonnegative_remainder_coefficients=residual.tolist(),maximum_existing_importance_weight_inflation=2.,
            interpretation='Pointwise density/raw-second-moment bound only; no guaranteed realized ESS, stratified variance or CPU speedup.'),
        density_measure='Translation volume times normalized rotational Haar; each g_j divides by its own chart Jacobian exactly once. No further Jacobian at mixture level.',
        source_inputs=old['source_inputs'],source_frame_assets=old['source_frame_assets'],envelope=old['envelope'],
        existing_runner=runner['binary'],source_bundle=runner['source_bundle'],cage_fit_report=recipe['fit_report'],
        heldout_policy='Descriptive validation only; no component rejection, width adjustment, mixture tuning or selection based on heldout loss.',
        physical_conditions=PHYSICAL,physical_cloud_allocation=CLOUD_RULE,
        physical_weight_rule='W_RB/Q_arm; two fresh independent equal-intensity clouds and a common certified L for every valid pose, including U=L. Keep all hard-invalid zeros and denominator4096. Arithmetic-pair estimator remains secondary.',
        future_populations=future_populations(),total_future_attempts=32768,future_stratum_jobs=32,
        stratum_interpretation='Counts refer to full runner laws, not chart-branch counts; each cage receives256 expected direct Gaussian-branch draws per population.',
        fresh_control_rule='Four fresh independent populations per arm; old pilot/training draws are never pooled into evaluation.',
        chart_center_scope='The two selected seeds center auxiliary proposal coordinates only; all physical source/frame/region definitions remain unchanged.',
        new_pose_draws=0,geometry_queries=0,cloud_draws=0,new_fit_samples=0,seeds_materialized=False,
        execution_materialized=False,assembly_kernel_changed=False,
        scope='Prospective fixed-context proposal-efficiency comparison only; no new coverage, equilibrium or assembly claim.')
    save(args.out/'mixture-manifest.json',manifest)
    for path,digest in bindings.items():require(sha(path)==digest,'Input changed during metadata preparation')
    save(args.out/'report.json',dict(complete=True,passed=True,manifest=dict(path=str((args.out/'mixture-manifest.json').resolve()),sha256=sha(args.out/'mixture-manifest.json')),
        guides=frozen,poses_generated=0,geometry_queries=0,cloud_draws=0,fitted_samples=0,seeds_materialized=False,execution_materialized=False))
    print(json.dumps(read(args.out/'report.json')))


if __name__=='__main__':main()
