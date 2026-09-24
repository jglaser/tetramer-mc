#!/usr/bin/env python3
"""One frozen, retrospective kernel-shear fit on already classified pilot rows.

No physical sampling, native classification, geometry predicates, or old audit
replays. The four fitting populations and four previously inspected holdouts
are disjoint. This diagnostic never reads the later protected validation rows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from scipy.special import logsumexp

from prepare_shoulder_docking_benchmark import local_dependencies
from prepare_contact_bank_guides import log_proposal

SCHEMA = 'fitted-kernel-shear-diagnostic-v1'
FIT = dict(centers=16, bandwidth=1., ridge=.01, minimum_ess=64., seed=48020260924)
ORDER = [3, 4, 5, 0, 1, 2]
CLASSES = {'native_inside_R5': 0, 'native_complement': 1, 'contact_no_native_entry': 2}
STRATA = {'radial': 3, 'angular': 3, 'orthant': 64}
PINS = dict(analysis='41264f95c98e3dce7d6748ee32bb80aee666b82e8e0fd8b1e351d875e0c846be',
    guide='867a3be62ec834ee96cf5cd3149dfa82f06158986b892f2dfccd83bbe91d1f48',
    region='924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02',
    shape='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9')
SCOPE = ('Retrospective held-out proposal-density diagnostic on archived fixed-R4 '
    'contact rows. Equal-class weighting uses cached native-informed labels and '
    'is proposal design, not a change of physical target or template-free discovery. '
    'The holdouts were previously inspected; they are not pristine prospective '
    'validation. No new physical draws, geometry/native predicates, old audit '
    'replays, operational guide changes, convergence-gate changes, or sampling '
    'speedup/assembly claims. Later protected-validation records remain unused.')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    import hashlib
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def status(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def bind(path, bindings, expected=None):
    path = Path(path).resolve(); digest = sha(path)
    require(expected is None or digest == expected, 'Archived input hash differs: '+str(path))
    bindings[str(path)] = digest
    return path


def recheck(bindings):
    for path, digest in bindings.items():
        require(sha(path) == digest, 'Frozen input/source changed: '+str(path))


def transform_for(region):
    covariance = np.asarray(region['gaussian_chart']['covariances'][0], float)
    require(covariance.shape == (6, 6) and np.isfinite(covariance).all(), 'Invalid R4 covariance')
    return np.linalg.cholesky(covariance)[ORDER, :]


def freeze(out, repository):
    """Bind byte-identical archived inputs and code before fitting or holdout reads."""
    out = Path(out).resolve(); repository = Path(repository).resolve()
    require(not out.exists(), 'Fresh diagnostic output required; no overwrite/retry')
    require(sys.flags.optimize == 0, 'Run without Python optimization')
    bindings = {}
    comparison = repository/'runs/smc-geometry-guide-pilot-20260923/comparison'
    analysis_path = bind(comparison/'analysis.json', bindings, PINS['analysis'])
    analysis = read(analysis_path)
    require(analysis['complete'] is True and analysis['region_sha256'] == PINS['region'] and
            analysis['shape_sha256'] == PINS['shape'], 'Unexpected archived pilot target')
    preparation = repository/'runs/protected-guide-preparation-20260923'
    guide_path = bind(preparation/'guide-protected.json', bindings, PINS['guide'])
    region_path = bind(preparation/'region.json', bindings, PINS['region'])
    for name in ('plan.json', 'preparation.json', 'freeze.json'):
        bind(preparation/name, bindings)
    documentation = bind(repository/'docs/kernel-shear-fitting.md', bindings)
    guide = read(guide_path); region = read(region_path)
    require(len(guide['gaussian_components']) == 84 and
            guide['defensive_uniform_shell_probability'] == .5 and
            guide['region_sha256'] == PINS['region'], 'Protected84 baseline differs')
    datasets = []
    for arm in ('bank', 'smc'):
        populations = analysis['arms'][arm]['populations']
        require(len(populations) == 4 and {p['id'] for p in populations} == {'r00', 'r01', 'r02', 'r03'},
                'Four original independent populations per source arm required')
        for record in sorted(populations, key=lambda p: p['id']):
            path = bind(comparison/record['records'], bindings, record['records_sha256'])
            require(record['samples'] == 65536, 'Fixed pilot allocation differs')
            datasets.append(dict(arm=arm, id=record['id'], seed=record['seed'], samples=record['samples'],
                role='training' if record['id'] in ('r00', 'r01') else 'heldout',
                records=str(path), records_sha256=record['records_sha256']))
    require(len({p['seed'] for p in datasets}) == 8, 'Training/holdout seeds overlap')
    # Resolve the implementation only after it exists; never freeze a partial closure.
    sources = local_dependencies([Path(__file__), Path(__file__).with_name('fit_kernel_shear.py'),
        Path(__file__).with_name('test_fit_kernel_shear.py'),
        Path(__file__).with_name('test_diagnose_fitted_kernel_shear.py')])
    require('fit_kernel_shear.py' in sources and 'kernel_shear.py' in sources, 'Missing frozen fitting implementation')
    for path in sources.values(): bind(path, bindings)
    out.mkdir(parents=True); (out/'source').mkdir()
    for name, path in sources.items(): shutil.copy2(path, out/'source'/name)
    shutil.copy2(guide_path, out/'guide.json'); shutil.copy2(region_path, out/'region.json')
    shutil.copy2(documentation, out/'documentation.md')
    plan = dict(schema=SCHEMA, scope=SCOPE, repository=str(repository),
        python=sys.executable, python_sha256=sha(sys.executable), python_version=sys.version,
        fit=FIT, transform=transform_for(region).tolist(), transform_convention='y=T u; T=Cholesky(R4 covariance)[3,4,5,0,1,2; :]',
        physical_coordinate_note='Angular block is the existing ell-scaled Cayley coordinate; constant chart offsets are immaterial.',
        guide_sha256=PINS['guide'], region_sha256=PINS['region'], datasets=datasets,
        documentation=dict(source=str(documentation), sha256=sha(documentation), archive='documentation.md'),
        training_attempts=262144, heldout_attempts=262144, fit_calls=1, tuning=False,
        class_balance=dict(class_ids=[0, 1, 2], probabilities=[1/3]*3,
            formula='log_weight_i=z_i-logsumexp(z in training class)-log(3); no heldout quantities'),
        regression=dict(responsibilities='Fixed baseline COMPLETE-mixture responsibilities including the defensive component',
            component_weights='w_i*r_ik then normalized conditionally within k',
            penalty='0.5*ridge times squared coefficients in each conditional-component regression; global surrogate penalty 0.5*tau_k*ridge times squared coefficients',
            insufficient_ESS='Keep zero warp below responsibility-weighted ESS64',
            geometry='All84 original mixture weights, affine means/covariances, and alpha=.5 retained'),
        evaluation=dict(classes=CLASSES, strata=STRATA, include_residual=True, include_all_valid=True,
            full_mixture=True, keep_all_unconditional_draws=True, uncertainty_unit='whole population',
            two_cloud_noisy_M2='N^-1 sum exp(2*z+log_q_source-log_q_candidate)',
            paired_physical_M2='N^-1 sum exp(pairs[:,0]+pairs[:,1]+log_q_source-log_q_candidate)',
            log_density='original u coordinates; physical Jacobian cancels in baseline/warp difference'),
        input_and_source_sha256=bindings, sources={name: sha(out/'source'/name) for name in sources},
        no_new_physical_validation=True)
    write_new(out/'plan.json', plan)
    write_new(out/'declaration-freeze.json', dict(files={str(p.relative_to(out)): sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()}))
    recheck(bindings)
    return plan


def validate(out, expected):
    out = Path(out).resolve()
    require(sha(out/'plan.json') == expected, 'Predeclared plan hash differs')
    plan = read(out/'plan.json')
    require(plan['schema'] == SCHEMA and plan['fit'] == FIT and plan['fit_calls'] == 1 and
            plan['tuning'] is False and plan['training_attempts'] == plan['heldout_attempts'] == 262144,
            'Fixed one-fit design differs')
    require(sys.flags.optimize == 0 and sys.executable == plan['python'] and
            sha(sys.executable) == plan['python_sha256'] and sys.version == plan['python_version'],
            'Frozen Python runtime differs')
    recheck(plan['input_and_source_sha256'])
    frozen = read(out/'declaration-freeze.json')['files']
    for name, digest in frozen.items():
        path = (out/name).resolve()
        require(path.is_relative_to(out) and sha(path) == digest, 'Declaration/source archive changed')
    require(sha(__file__) == plan['sources'][Path(__file__).name], 'Driver differs from frozen source')
    return plan


def load_rows(record):
    """Read the existing cache only; no classifier or bath evaluation."""
    require(sha(record['records']) == record['records_sha256'], 'Archived records changed')
    with np.load(record['records'], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    n = record['samples']; required = {'u', 'z', 'h', 'pairs', 'log_q', 'log_physical_jacobian',
        'draw', 'source_n', 'class_id', 'support', 'native', 'contact', 'bin_radial', 'bin_angular', 'bin_orthant'}
    require(required <= set(arrays), 'Incomplete archived scoring fields')
    require(arrays['u'].shape == (n, 6) and np.isfinite(arrays['u']).all() and
            arrays['pairs'].shape == (n, 2), 'Wrong coordinate/cloud allocation')
    require(all(len(value) == n for value in arrays.values()) and
            np.array_equal(arrays['draw'], np.arange(n)) and np.all(arrays['source_n'] == n),
            'Original unconditional attempt identities/denominators changed')
    for name in ('z', 'h', 'pairs'):
        require(not np.isnan(arrays[name]).any() and not np.isposinf(arrays[name]).any(), 'Invalid saved log weights')
    valid = np.isfinite(arrays['z'])
    require(np.array_equal(valid, np.isfinite(arrays['h'])) and
            np.array_equal(np.isfinite(arrays['pairs']), np.repeat(valid[:, None], 2, axis=1)),
            'Hard/depletion/cloud zero support differs')
    require(np.isfinite(arrays['log_q']).all() and np.isfinite(arrays['log_physical_jacobian']).all(),
            'Nonfinite saved proposal/Jacobian')
    require(np.all(arrays['support'][valid]) and np.all(np.isin(arrays['class_id'][valid], [-1, 0, 1, 2])) and
            np.all(arrays['class_id'][~valid] == -2), 'Saved class partition differs')
    if valid.any():
        require(np.max(abs(arrays['h'][valid]+arrays['log_q'][valid]-arrays['log_physical_jacobian'][valid])) < 2e-8,
                'Saved J/q accounting differs')
        require(np.max(abs(logsumexp(arrays['pairs'][valid], axis=1)-math.log(2)-arrays['z'][valid])) < 2e-8,
                'Saved two-cloud mean differs')
    for family, size in STRATA.items():
        require(np.all((arrays['bin_'+family][valid] >= 0) & (arrays['bin_'+family][valid] < size)),
                'Invalid valid-row stratum')
    return arrays


def training_data(pieces):
    """Externally normalized equal-class physical weights; invalid zeros retained in metadata."""
    require(pieces, 'No fitting populations')
    u = np.concatenate([p['u'] for p in pieces]); z = np.concatenate([p['z'] for p in pieces])
    classes = np.concatenate([p['class_id'] for p in pieces]); valid = np.isfinite(z)
    logw = np.full(len(z), -np.inf); normalization = {}
    for name, identity in CLASSES.items():
        mask = valid & (classes == identity)
        require(mask.any(), 'Training class is unobserved: '+name)
        total = float(logsumexp(z[mask])); logw[mask] = z[mask]-total-math.log(3)
        normalization[name] = dict(rows=int(mask.sum()), log_physical_weight_sum=total,
            normalized_mass=float(np.exp(logsumexp(logw[mask]))),
            physical_weight_ESS=float(np.exp(2*total-logsumexp(2*z[mask]))))
    keep = np.isfinite(logw)
    require(abs(float(logsumexp(logw[keep]))) < 1e-12, 'Training likelihood weights are not normalized')
    return u[keep], logw[keep], dict(attempts=len(z), selected_rows=int(keep.sum()),
        invalid_zero_rows=int((~valid).sum()), residual_valid_rows=int((valid & (classes == -1)).sum()),
        class_normalization=normalization)


def logsum_or_none(values):
    return float(logsumexp(values)) if len(values) else None


def moments(terms, n):
    terms = np.asarray(terms, float)
    if not len(terms):
        return dict(log_sum=None, log_sum_squares=None, max_log_term=None, log_M2=None,
                    contribution_ESS=0., largest_fraction=None)
    total = float(logsumexp(terms)); squared = float(logsumexp(2*terms)); maximum = float(terms.max())
    return dict(log_sum=total, log_sum_squares=squared, max_log_term=maximum, log_M2=total-math.log(n),
        contribution_ESS=float(math.exp(2*total-squared)), largest_fraction=float(math.exp(maximum-total)))


def group_masks(arrays):
    valid = np.isfinite(arrays['z'])
    base = {name: valid & (arrays['class_id'] == cid) for name, cid in CLASSES.items()}
    base.update(residual_valid=valid & (arrays['class_id'] == -1), all_valid=valid)
    for name, mask in base.items():
        yield name, mask
        for family, size in STRATA.items():
            for index in range(size): yield f'{name}:{family}:{index}', mask & (arrays['bin_'+family] == index)


def evaluate_group(arrays, baseline, warped, mask):
    n = len(arrays['draw']); count = int(mask.sum()); z = arrays['z'][mask]
    if not count:
        likelihood = dict(log_weight_sum=None, log_weight_square_sum=None, maximum_log_weight=None,
            log_physical_mass=None, weight_ESS=0., largest_weight_fraction=None,
            baseline_mean_log_density=None, warped_mean_log_density=None, mean_log_density_gain=None)
    else:
        logsum = float(logsumexp(z)); weights = np.exp(z-logsum)
        likelihood = dict(log_weight_sum=logsum, log_weight_square_sum=float(logsumexp(2*z)),
            maximum_log_weight=float(max(z)), log_physical_mass=logsum-math.log(n),
            weight_ESS=float(np.exp(2*logsum-logsumexp(2*z))), largest_weight_fraction=float(weights.max()),
            baseline_mean_log_density=float(weights@baseline[mask]), warped_mean_log_density=float(weights@warped[mask]),
            mean_log_density_gain=float(weights@(warped[mask]-baseline[mask])))
    second = {}
    for label, terms in [('two_cloud_noisy', 2*z), ('paired_physical', arrays['pairs'][mask].sum(axis=1))]:
        values = {}
        for name, density in [('baseline', baseline), ('warped', warped)]:
            values[name] = moments(terms+arrays['log_q'][mask]-density[mask], n)
        values['log_warp_to_baseline_ratio'] = (values['warped']['log_M2']-values['baseline']['log_M2']) if count else None
        second[label] = values
    return dict(attempts=n, contributing_rows=count, likelihood=likelihood, second_moments=second)


def combine_group(reports):
    """Combine diagnostic sums with original N; keep independent-population rows elsewhere."""
    n = sum(p['attempts'] for p in reports); count = sum(p['contributing_rows'] for p in reports)
    if not count: return dict(attempts=n, contributing_rows=0, likelihood=copy_empty_likelihood(),
        second_moments={kind: dict(baseline=moments([], n), warped=moments([], n), log_warp_to_baseline_ratio=None)
                        for kind in ('two_cloud_noisy', 'paired_physical')})
    present = [p for p in reports if p['contributing_rows']]
    logs = [p['likelihood']['log_weight_sum'] for p in present]; total = float(logsumexp(logs))
    fractions = np.exp(np.asarray(logs)-total)
    likelihood = dict(log_weight_sum=total,
        log_weight_square_sum=float(logsumexp([p['likelihood']['log_weight_square_sum'] for p in present])),
        maximum_log_weight=max(p['likelihood']['maximum_log_weight'] for p in present),
        log_physical_mass=total-math.log(n))
    likelihood['weight_ESS'] = math.exp(2*total-likelihood['log_weight_square_sum'])
    likelihood['largest_weight_fraction'] = math.exp(likelihood['maximum_log_weight']-total)
    for name in ('baseline_mean_log_density', 'warped_mean_log_density', 'mean_log_density_gain'):
        likelihood[name] = math.fsum(float(f)*p['likelihood'][name] for f, p in zip(fractions, present))
    second = {}
    for kind in ('two_cloud_noisy', 'paired_physical'):
        values = {}
        for name in ('baseline', 'warped'):
            parts = [p['second_moments'][kind][name] for p in present]
            s = float(logsumexp([p['log_sum'] for p in parts])); ss = float(logsumexp([p['log_sum_squares'] for p in parts]))
            mx = max(p['max_log_term'] for p in parts)
            values[name] = dict(log_sum=s, log_sum_squares=ss, max_log_term=mx, log_M2=s-math.log(n),
                contribution_ESS=math.exp(2*s-ss), largest_fraction=math.exp(mx-s))
        values['log_warp_to_baseline_ratio'] = values['warped']['log_M2']-values['baseline']['log_M2']; second[kind] = values
    return dict(attempts=n, contributing_rows=count, likelihood=likelihood, second_moments=second)


def copy_empty_likelihood():
    return dict(log_weight_sum=None, log_weight_square_sum=None, maximum_log_weight=None,
        log_physical_mass=None, weight_ESS=0., largest_weight_fraction=None,
        baseline_mean_log_density=None, warped_mean_log_density=None, mean_log_density_gain=None)


def draw_plot(path, populations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    classes = list(CLASSES); figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {'native_inside_R5': '#3465a4', 'native_complement': '#75507b', 'contact_no_native_entry': '#c17d11'}
    for index, name in enumerate(classes):
        for j, pop in enumerate(populations):
            if pop['role'] != 'heldout': continue
            group = pop['groups'][name]; x = index+(j%4-1.5)*.045
            gain = group['likelihood']['mean_log_density_gain']
            ratio = group['second_moments']['paired_physical']['log_warp_to_baseline_ratio']
            marker = 'o' if pop['arm'] == 'bank' else '^'
            if gain is not None: axes[0].scatter(x, gain, marker=marker, color=colors[name], alpha=.85)
            if ratio is not None: axes[1].scatter(x, ratio, marker=marker, color=colors[name], alpha=.85)
    for ax in axes:
        ax.axhline(0., color='.5', lw=1); ax.set_xticks(range(3), ['Native\ninside R5', 'Native\nremainder', 'Contact\nno entry'])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Physical-weighted log-density gain\npositive favors shear')
    axes[1].set_ylabel('log(paired M₂ shear / affine)\nnegative favors shear')
    figure.suptitle('Previously inspected holdouts · circles: bank source; triangles: SMC-guide source', fontsize=11)
    figure.tight_layout(); figure.savefig(path, dpi=170); plt.close(figure)


def run(out, expected_plan_sha256):
    from fit_kernel_shear import KernelMixture, fit_shears
    out = Path(out).resolve(); plan = validate(out, expected_plan_sha256)
    require(Path(__file__).resolve() == out/'source'/Path(__file__).name, 'Run the archived driver')
    with (out/'claim.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), plan_sha256=expected_plan_sha256), stream)
    state = dict(complete=False, phase='training', fit_calls=0, holdout_rows_read_before_model_freeze=0)
    status(out/'status.json', state); start = time.monotonic()
    try:
        baseline = KernelMixture.from_guide(read(out/'guide.json'), np.asarray(plan['transform']), radius=4.)
        training = [record for record in plan['datasets'] if record['role'] == 'training']
        pieces = [load_rows(record) for record in training]
        u, weights, training_summary = training_data(pieces)
        require(training_summary['attempts'] == plan['training_attempts'], 'Fitting allocation differs')
        baseline_before = baseline.to_dict()
        state['fit_calls'] = 1; status(out/'status.json', state)
        fitted, fit_report = fit_shears(baseline, u, weights, **plan['fit'])
        require(baseline.to_dict() == baseline_before, 'Fitting mutated the affine control')
        write_new(out/'baseline-model.json', baseline.to_dict())
        write_new(out/'fitted-model.json', fitted.to_dict())
        write_new(out/'fit.json', dict(training=training_summary, diagnostics=fit_report,
            fit=plan['fit'], fit_calls=1, tuning=False, holdout_rows_read=0))
        write_new(out/'model-freeze.json', dict(files={name: sha(out/name) for name in
            ('plan.json', 'baseline-model.json', 'fitted-model.json', 'fit.json')}))
        # No holdout array has been opened before the model freeze above.
        baseline = KernelMixture.from_dict(read(out/'baseline-model.json'))
        fitted = KernelMixture.from_dict(read(out/'fitted-model.json'))
        del pieces, u, weights
        state['phase'] = 'frozen_evaluation'; status(out/'status.json', state)
        (out/'evaluations').mkdir(); populations = []
        for record in plan['datasets']:
            arrays = load_rows(record); q0 = baseline.log_density(arrays['u']); q1 = fitted.log_density(arrays['u'])
            require(q0.shape == q1.shape == (record['samples'],) and np.isfinite(q0).all() and np.isfinite(q1).all(),
                    'Complete candidate density is not finite on archived rows')
            direct = log_proposal(arrays['u'], read(out/'guide.json'))
            density_error = float(np.max(abs(q0-direct)))
            require(density_error < 2e-8, 'Affine transform changed the original complete mixture density')
            groups = {name: evaluate_group(arrays, q0, q1, mask) for name, mask in group_masks(arrays)}
            saved = out/'evaluations'/f"{record['arm']}-{record['id']}.npz"
            np.savez_compressed(saved, **arrays, log_baseline_density=q0, log_warped_density=q1)
            populations.append(dict(**record, groups=groups, attempted_rows_preserved=len(arrays['draw']),
                baseline_density_max_error=density_error,
                evaluated_rows=str(saved.relative_to(out)), evaluated_rows_sha256=sha(saved)))
        aggregates = {}
        for role in ('training', 'heldout'):
            for arm in ('bank', 'smc'):
                chosen = [p for p in populations if p['role'] == role and p['arm'] == arm]
                require(len(chosen) == 2, 'Population split changed during evaluation')
                aggregates[role+'_'+arm] = {name: combine_group([p['groups'][name] for p in chosen])
                    for name in chosen[0]['groups']}
        # Retain paired population outcomes; two holdouts per source arm do not
        # warrant a precise asymptotic confidence interval.
        recheck(plan['input_and_source_sha256'])
        for name, digest in read(out/'model-freeze.json')['files'].items():
            require(sha(out/name) == digest, 'Frozen model changed during evaluation')
        draw_plot(out/'heldout-comparison.png', populations)
        result = dict(schema=SCHEMA, complete=True, scope=SCOPE, plan_sha256=expected_plan_sha256,
            fit=read(out/'fit.json'), populations=populations, aggregates=aggregates,
            all_unconditional_draws_preserved=sum(p['attempted_rows_preserved'] for p in populations),
            training_attempts=plan['training_attempts'], heldout_attempts=plan['heldout_attempts'],
            affine_parameters_and_mixture_weights_unchanged=True, new_physical_draws=0, classifier_calls=0,
            geometry_predicate_calls=0, old_audits_replayed=0, protected_validation_rows_read=0,
            operational_guide_written=False, gate_promotion=False,
            population_error_scope='Report each paired heldout outcome; two holdouts per source arm are not a precise confidence interval.',
            artifact_sha256={name: sha(out/name) for name in ('baseline-model.json', 'fitted-model.json', 'model-freeze.json', 'heldout-comparison.png')},
            source_and_input_sha256=plan['input_and_source_sha256'], documentation=plan.get('documentation'),
            wall_seconds=time.monotonic()-start)
        write_new(out/'analysis.json', result)
        state.update(complete=True, phase='complete', analysis_sha256=sha(out/'analysis.json')); status(out/'status.json', state)
        write_new(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob('*')) if p.is_file()}))
        return result
    except BaseException as error:
        state.update(complete=False, phase='failed', error=repr(error)); status(out/'status.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('freeze'); p.add_argument('--out', type=Path, required=True); p.add_argument('--repository', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--out', type=Path, required=True); p.add_argument('--expected-plan-sha256', required=True)
    args = parser.parse_args()
    if args.action == 'freeze': freeze(args.out, args.repository)
    else: run(args.out, args.expected_plan_sha256)


if __name__ == '__main__': main()
