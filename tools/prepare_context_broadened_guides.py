"""Freeze a Gaussian scale-mixture recipe without drawing poses or choosing seeds.

The existing one-source-chart runner supplies stratified draws from
q_j = .5 U + .25 G + .25 g_j. Pool every stratum with its declared mixture
density and the complete 4096-attempt denominator. No assembly kernel changes.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

COMPONENTS = ('full', 'diagonal', 'broad_full')
DENSITY_COMPONENTS = ('uniform', 'context', *COMPONENTS)
RUNNER_MIXTURE = (.5, .25, .25)
ALLOCATIONS = {
    'baseline': {'full': 2048, 'diagonal': 2048, 'broad_full': 0},
    'broadened': {'full': 1536, 'diagonal': 1536, 'broad_full': 1024},
}
POPULATIONS = 4
ATTEMPTS = 4096
COVARIANCE_SCALE = 4.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate_chart(chart):
    require(set(chart) == {'angular_length', 'covariance', 'explicit_gaussian'},
            'Expected explicit frozen Gaussian source chart')
    explicit = chart['explicit_gaussian']
    require(set(explicit) == {'schema', 'mean', 'provenance'}
            and explicit['schema'] == 'source-gaussian-v1'
            and isinstance(explicit['provenance'], str)
            and bool(explicit['provenance'].strip()), 'Invalid explicit Gaussian metadata')
    ell = float(chart['angular_length'])
    mean, covariance = np.asarray(explicit['mean'], float), np.asarray(chart['covariance'], float)
    require(math.isfinite(ell) and ell > 0 and mean.shape == (6,)
            and covariance.shape == (6, 6) and np.isfinite(mean).all()
            and np.isfinite(covariance).all(), 'Invalid Gaussian chart parameters')
    magnitude = float(np.max(np.abs(covariance)))
    require(magnitude > 0 and np.all(np.abs(covariance-covariance.T)
            <= 1e-12*(magnitude+np.abs(covariance))), 'Asymmetric chart covariance')
    try:
        lower = np.linalg.cholesky(.5*(covariance+covariance.T))
    except np.linalg.LinAlgError as error:
        raise ValueError('Chart covariance must already be positive definite') from error
    require(np.isfinite(lower).all(), 'Unrepresentable frozen Cholesky factor')
    return mean, covariance


def validate_controls(full, diagonal):
    for name, guide in [('full', full), ('diagonal', diagonal)]:
        require(guide['schema'] == 'context-covariance-frozen-guide-v1'
                and guide['arm'] == name and guide['training_streams'] == [0, 1]
                and guide['heldout_streams'] == [2, 3], 'Changed existing guide identity')
        validate_chart(guide['source_chart'])
    f, d = full['source_chart'], diagonal['source_chart']
    require(full['reduction_sha256'] == diagonal['reduction_sha256']
            and full['fit_plan_sha256'] == diagonal['fit_plan_sha256']
            and f['angular_length'] == d['angular_length']
            and f['explicit_gaussian']['mean'] == d['explicit_gaussian']['mean'],
            'Controls must retain the same frozen training mean and frame')
    require(np.array_equal(np.diag(np.diag(f['covariance'])), np.asarray(d['covariance'])),
            'Diagonal control differs from full covariance diagonal')


def broaden(full, full_asset):
    """The only numerical transformation is multiplying the complete C by four."""
    _, covariance = validate_chart(full['source_chart'])
    chart = copy.deepcopy(full['source_chart'])
    with np.errstate(over='ignore', invalid='ignore'):
        chart['covariance'] = (COVARIANCE_SCALE*covariance).tolist()
    chart['explicit_gaussian']['provenance'] = (
        f'Deterministic covariance scale 4 of frozen full guide {full_asset["sha256"]}; '
        'unchanged fitted mean, angular length and saved source/anchor frame; no additional fit or ridge.')
    validate_chart(chart)
    return dict(schema='context-scaled-covariance-frozen-guide-v1', component='broad_full',
        source_chart=chart, derived_from=full_asset,
        covariance_scale=COVARIANCE_SCALE, standard_deviation_scale=2.,
        training_streams=full['training_streams'], heldout_streams=full['heldout_streams'],
        reduction_sha256=full['reduction_sha256'], fit_plan_sha256=full['fit_plan_sha256'],
        inherited_regularization=copy.deepcopy(full['regularization']), additional_regularization=False,
        new_fit_samples=0, new_pose_draws=0,
        interpretation='Fixed scale broadening of a source-informed proposal; not an equilibrium covariance or native-blind discovery.')


def density_coefficients(arm):
    require(arm in ALLOCATIONS, 'Unknown comparison arm')
    allocation = ALLOCATIONS[arm]
    require(sum(allocation.values()) == ATTEMPTS, 'Changed all-attempt denominator')
    return np.array([.5, .25]+[.25*allocation[c]/ATTEMPTS for c in COMPONENTS])


def log_mixture(arm, component_logs):
    """Exact deterministic-mixture law; the component logs already contain J."""
    values = np.asarray(component_logs, float)
    require(values.shape == (5,) and not np.isnan(values).any()
            and not np.isposinf(values).any(), 'Invalid normalized component log densities')
    probabilities = density_coefficients(arm)
    terms = [math.log(p)+v for p, v in zip(probabilities, values) if p > 0 and v != -math.inf]
    if not terms:
        return -math.inf
    high = max(terms)
    return high + math.log(sum(math.exp(v-high) for v in terms))


def stratified_allocation():
    result = []
    for arm, counts in ALLOCATIONS.items():
        for population in range(POPULATIONS):
            result.append(dict(arm=arm, population=population, total_attempts=ATTEMPTS,
                strata=[dict(component=c, draws=counts[c], deterministic_fraction=counts[c]/ATTEMPTS)
                        for c in COMPONENTS if counts[c]],
                denominator=ATTEMPTS, independent_seed_assignment='pending separate frozen execution plan'))
    return result


def validate_recipe(recipe):
    require(recipe['schema'] == 'context-broadened-source-recipe-v1'
            and recipe['arms'] == ['baseline', 'broadened']
            and recipe['components'] == list(COMPONENTS)
            and recipe['populations_per_arm'] == POPULATIONS
            and recipe['attempts_per_population'] == ATTEMPTS
            and recipe['component_draws'] == ALLOCATIONS
            and recipe['covariance_scale'] == COVARIANCE_SCALE
            and recipe['runner_mixture'] == list(RUNNER_MIXTURE)
            and recipe['physical_conditions'] == {'depletant_radius': 1.5, 'activity': .035, 'lambda': 2.24}
            and recipe['seeds_materialized'] is False
            and recipe['physical_cloud_allocation'] is None
            and recipe['reuse_old_draws'] is False, 'Changed prospective recipe')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--expected-recipe-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bindings = {}
    def bind(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, 'Changed input: '+str(path))
        require(str(path) not in bindings or bindings[str(path)] == expected, 'Conflicting input pin')
        bindings[str(path)] = expected
        return read(path)
    def asset(record):
        return bind(record['path'], record['sha256'])
    recipe = bind(args.recipe, args.expected_recipe_sha256)
    validate_recipe(recipe)
    guides = {c: asset(recipe['guides'][c]) for c in ('full', 'diagonal')}
    validate_controls(guides['full'], guides['diagonal'])
    allocation = asset(recipe['source_allocation'])
    require(allocation['schema'] == 'context-source-covariance-geometry-allocation-v1'
            and allocation['guides'] == recipe['guides']
            and allocation['mixture'] == list(RUNNER_MIXTURE), 'Changed original source law')
    require(allocation['reduction']['sha256'] == guides['full']['reduction_sha256']
            and allocation['fit_plan']['sha256'] == guides['full']['fit_plan_sha256'],
            'Source frame/fit provenance differs')
    for record in allocation['inputs'].values():
        if isinstance(record, dict) and set(record) == {'path', 'sha256'}:
            asset(record)
    physical = {name: asset(record) for name, record in allocation['direct_physical_assets'].items()}
    source, context = physical['source_state'], physical['fixed_context']
    require(source['moving_label'] == 77 and source['anchor_label'] == context['anchor_label'] == 16
            and source['boundary'] == 'spherical'
            and context['excluded_moving_labels'] == [77]
            and source['anchor_pose'] == next(b['pose'] for b in context['bodies'] if b['label'] == 16),
            'Changed saved pose frame or fixed neighborhood')
    invocation = asset(allocation['inputs']['invocation_config'])
    require(invocation['depletant_radius'] == 1.5 and invocation['reservoir_density'] == .035
            and invocation['poisson_lambda_ratio'] == 64., 'Changed physical model')
    asset(allocation['reduction']); asset(allocation['fit_plan'])
    # Executable bytes are authenticated without loading or executing the runner.
    binary = allocation['binary']
    require(sha(binary['path']) == binary['sha256'], 'Validated existing runner changed')
    bindings[str(Path(binary['path']).resolve())] = binary['sha256']
    asset(allocation['source_bundle'])
    broad = broaden(guides['full'], recipe['guides']['full'])
    baseline = density_coefficients('baseline'); expanded = density_coefficients('broadened')
    residual = expanded-.75*baseline
    require(np.array_equal(residual, np.array([.125, .0625, 0., 0., .0625])),
            'Mixture lower-bound proof differs')
    require(not args.out.exists(), 'Fresh metadata-only output directory required')
    args.out.mkdir()
    path = args.out/'broad-full.json'
    save(path, broad)
    frozen_guides = dict(recipe['guides'], broad_full=dict(path=str(path.resolve()), sha256=sha(path)))
    manifest = dict(schema='context-broadened-source-mixture-v1', complete=True, passed=True,
        source_sha256=sha(__file__), recipe=dict(path=str(args.recipe.resolve()), sha256=args.expected_recipe_sha256),
        input_sha256=bindings, guides=frozen_guides,
        density_measure='Translation volume times normalized rotational Haar; each chart density already divides by the existing Cayley Jacobian. No additional Jacobian at mixture level.',
        density_component_order=list(DENSITY_COMPONENTS),
        effective_density_coefficients={arm: density_coefficients(arm).tolist() for arm in ALLOCATIONS},
        mixture_lower_bound=dict(fraction=.75, baseline_arm='baseline', new_arm='broadened',
            nonnegative_remainder_coefficients=residual.tolist(),
            maximum_existing_importance_weight_inflation=4/3,
            interpretation='Pointwise density bound only; no guarantee of improved acceptance, ESS, mixing or assembly.'),
        mixture_identity='q_broadened=.75*q_baseline+.25*q_B; q_j=.5*U+.25*G+.25*g_j.',
        source_inputs=allocation['inputs'], source_frame_assets=allocation['direct_physical_assets'],
        physical_conditions=recipe['physical_conditions'], envelope=allocation['envelope'],
        existing_runner=binary, source_bundle=allocation['source_bundle'], runner_schema='context-source-guide-v1',
        runner_mixture=list(RUNNER_MIXTURE),
        future_populations=stratified_allocation(), total_future_attempts=2*POPULATIONS*ATTEMPTS,
        future_stratum_jobs=sum(len(p['strata']) for p in stratified_allocation()),
        stratum_interpretation='Counts are total draws of each existing runner law (.5 U+.25 G+.25 g_j), not guaranteed source-component draws. Combine all strata with the arm mixture q and denominator4096; retain every hard-invalid zero.',
        covariance_transform='C_B=4*C_F, mean_B=mean_F; source/anchor poses and angular_length unchanged; no new fit or ridge.',
        source_covariance_status='Source-informed saved-local-control proposal. The original fit used training streams0,1; heldout2,3 remain unfitted. No equilibrium covariance claim.',
        physical_weight_rule='Any future physical campaign must freeze its own two-cloud allocation and evaluate W/q using the arm density in translation times normalized Haar. The chart Jacobian is already included in q; do not multiply by it again. This metadata preparation estimates no physical weights.',
        fresh_control_rule='Both arms require fresh independent populations and independently assigned subjob seeds. No previous draw is pooled into these future controls.',
        new_pose_draws=0, geometry_queries=0, cloud_draws=0, new_fit_samples=0,
        seeds_materialized=False, physical_cloud_allocation=None, assembly_kernel_changed=False,
        execution_materialized=False)
    save(args.out/'mixture-manifest.json', manifest)
    for path, expected in bindings.items():
        require(sha(path) == expected, 'Input changed during preparation: '+path)
    save(args.out/'report.json', dict(complete=True, passed=True, source_sha256=sha(__file__),
        manifest=dict(path=str((args.out/'mixture-manifest.json').resolve()), sha256=sha(args.out/'mixture-manifest.json')),
        guides=frozen_guides, geometry_queries=0, cloud_draws=0, poses_generated=0, fitted_samples=0,
        seeds_materialized=False, execution_materialized=False))
    print(json.dumps(read(args.out/'report.json')))


if __name__ == '__main__':
    main()
