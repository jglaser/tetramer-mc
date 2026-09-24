"""No-fit independent-population evaluator tests; no archived rows are scored."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import evaluate_frozen_kernel_shear as evaluator
from test_diagnose_fitted_kernel_shear import arrays


def group_fixture(total=10., observed_index=0):
    groups = {}
    for name in evaluator.CLASSES:
        groups[name] = dict(likelihood=dict(log_weight_sum=math.log(total), mean_log_density_gain=.1),
            second_moments=dict(paired_physical=dict(log_warp_to_baseline_ratio=-.2)))
        for family, size in evaluator.STRATA.items():
            for index in range(size):
                mass = total if index == observed_index else 0.
                groups[f'{name}:{family}:{index}'] = dict(likelihood=dict(
                    log_weight_sum=math.log(mass) if mass else None,
                    mean_log_density_gain=.1 if mass else None), second_moments=dict(
                    paired_physical=dict(log_warp_to_baseline_ratio=-.2 if mass else None)))
    return groups


def pilot_fixture():
    return dict(populations=[dict(arm=arm, id=f'r{i:02d}', seed=j*4+i+1,
        role='training' if i < 2 else 'heldout', groups=group_fixture(total=10. if i < 2 else 1e100))
        for j, arm in enumerate(('bank', 'smc')) for i in range(4)])


def comparison_fixture():
    return dict(complete=True, region_sha256=evaluator.PINS['region'], shape_sha256=evaluator.PINS['shape'],
        arms={arm: dict(allocation=dict(samples=evaluator.N, alpha=.5, lambda_ratio=128.),
            populations=[dict(id=f'r{i:02d}', seed=100+4*j+i, samples=evaluator.N,
                records=f'{arm}/r{i:02d}/records.npz', records_sha256='a'*64) for i in range(4)])
            for j, arm in enumerate(evaluator.ARMS)})


def model_fixture():
    chart = dict(mean=[0.]*6, lower=np.eye(6).tolist(), shear=dict(dimension=6,
        conditioning=[0, 1, 2], shifted=[3, 4, 5], centers=[], coefficients=[], bandwidth=1.))
    value = dict(schema='frozen-kernel-shear-mixture-v1', alpha=.5, radius=4.,
        transform=np.eye(6).tolist(), weights=[1/84]*84, charts=[copy.deepcopy(chart) for _ in range(84)])
    return value


class FrozenEvaluationTests(unittest.TestCase):
    def test_fixed_eight_populations_and_disjoint_seeds(self):
        pilot = pilot_fixture(); comparison = comparison_fixture()
        rows = evaluator.declared_datasets(comparison, '/tmp/frozen-comparison', pilot)
        self.assertEqual(len(rows), 8); self.assertEqual(sum(p['samples'] for p in rows), 8388608)
        self.assertEqual([(p['arm'], p['id']) for p in rows],
            [(a, f'r{i:02d}') for a in ('bank', 'protected') for i in range(4)])
        changes = [lambda p: p['arms']['bank']['populations'].pop(),
                   lambda p: p['arms']['bank']['populations'][0].update(seed=1),
                   lambda p: p['arms']['protected']['populations'][0].update(seed=100),
                   lambda p: p['arms']['bank']['populations'][0].update(samples=12),
                   lambda p: p['arms']['bank']['allocation'].update(lambda_ratio=256.)]
        for change in changes:
            bad = copy.deepcopy(comparison); change(bad)
            with self.assertRaises(ValueError): evaluator.declared_datasets(bad, '/tmp/example', pilot)

    def test_important_groups_use_only_training_and_keep_orthant63(self):
        pilot = pilot_fixture(); selected = evaluator.training_importance(pilot)
        self.assertEqual(len(selected), 3*(1+3+3+64))
        for name in evaluator.CLASSES:
            self.assertTrue(selected[name]['important'])
            self.assertTrue(selected[name+':orthant:0']['important'])
            self.assertIn(name+':orthant:63', selected)
            self.assertFalse(selected[name+':orthant:63']['important'])
        for p in pilot['populations']:
            if p['role'] == 'heldout': p['groups'] = {}  # Not even accessed.
        self.assertEqual(selected, evaluator.training_importance(pilot))

    def test_training_threshold_uses_combined_linear_mass(self):
        pilot = pilot_fixture()
        for p in pilot['populations']:
            if p['role'] != 'training': continue
            p['groups']['contact_no_native_entry:orthant:62']['likelihood']['log_weight_sum'] = math.log(.11)
            p['groups']['contact_no_native_entry:orthant:63']['likelihood']['log_weight_sum'] = math.log(.09)
        result = evaluator.training_importance(pilot)
        self.assertAlmostEqual(result['contact_no_native_entry:orthant:62']['training_class_fraction'], .011)
        self.assertTrue(result['contact_no_native_entry:orthant:62']['important'])
        self.assertFalse(result['contact_no_native_entry:orthant:63']['important'])

    def test_fixed_candidate_allows_only_warp_parameters(self):
        baseline = model_fixture(); fitted = copy.deepcopy(baseline)
        fitted['charts'][0]['shear'].update(centers=[[0., 0., 0.]], coefficients=[[.1, .2, .3]])
        evaluator.same_frozen_affine(baseline, fitted)
        changes = [lambda p: p.update(alpha=.2), lambda p: p.update(radius=5.),
                   lambda p: p['weights'].__setitem__(0, .5),
                   lambda p: p['charts'][0]['mean'].__setitem__(0, .1),
                   lambda p: p['charts'][0]['lower'][0].__setitem__(0, 2.),
                   lambda p: p['charts'][0]['shear'].update(conditioning=[0, 2, 3])]
        for change in changes:
            bad = copy.deepcopy(fitted); change(bad)
            with self.assertRaises(ValueError): evaluator.same_frozen_affine(baseline, bad)

    def test_unobserved_training_stratum_reported_without_changing_declaration(self):
        declaration = evaluator.training_importance(pilot_fixture()); unchanged = copy.deepcopy(declaration)
        groups = group_fixture()
        key = 'contact_no_native_entry:orthant:63'
        groups[key] = dict(likelihood=dict(log_weight_sum=math.log(2.), mean_log_density_gain=-.1),
                          second_moments=dict(paired_physical=dict(log_warp_to_baseline_ratio=.4)))
        diagnostic = evaluator.group_diagnostics(groups, declaration)
        self.assertIn(key, diagnostic['training_unobserved_but_evaluation_fraction_ge_1percent'])
        self.assertFalse(diagnostic['groups'][key]['important'])
        self.assertNotIn(key, diagnostic['important_regressions'])
        self.assertEqual(declaration, unchanged)
        groups['native_inside_R5']['second_moments']['paired_physical']['log_warp_to_baseline_ratio'] = .2
        diagnostic = evaluator.group_diagnostics(groups, declaration)
        self.assertIn('native_inside_R5', diagnostic['important_regressions'])
        self.assertNotIn('passed', diagnostic)

    def test_run_never_fits_preserves_every_row_and_source_arm(self):
        import fit_kernel_shear
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'pilot').mkdir(); (root/'source').mkdir()
            evaluator.write_new(root/'pilot/baseline-model.json', {'kind': 'baseline'})
            evaluator.write_new(root/'pilot/fitted-model.json', {'kind': 'fitted'})
            pins = dict(evaluator.PINS, baseline=evaluator.sha(root/'pilot/baseline-model.json'),
                        fitted=evaluator.sha(root/'pilot/fitted-model.json'))
            datasets = [dict(arm=arm, id=f'r{i:02d}', seed=100+4*j+i, samples=8,
                role='independent_retrospective', records='unused', records_sha256='unused')
                for j, arm in enumerate(evaluator.ARMS) for i in range(4)]
            plan = dict(datasets=datasets, all_groups_per_population=355, total_attempts=64,
                input_and_source_sha256={}, density_timing='synthetic-only',
                training_only_group_declaration=evaluator.training_importance(pilot_fixture()))
            class Model:
                def __init__(self, kind): self.kind = kind
                @classmethod
                def from_dict(cls, value): return cls(value['kind'])
                def log_density(self, u): return np.full(len(u), .1 if self.kind == 'fitted' else 0.)
            def plot(path, populations): Path(path).write_text('synthetic plot')
            with mock.patch.object(evaluator, '__file__', str(root/'source'/Path(evaluator.__file__).name)), \
                 mock.patch.object(evaluator, 'validate', return_value=plan), \
                 mock.patch.object(evaluator, 'load_rows', side_effect=lambda record: arrays()), \
                 mock.patch.object(evaluator, 'KernelMixture', Model), \
                 mock.patch.object(evaluator, 'comparison_plot', side_effect=plot), \
                 mock.patch.object(evaluator, 'PINS', pins), \
                 mock.patch.object(fit_kernel_shear, 'fit_shears', side_effect=AssertionError('Must not fit')) as fit:
                result = evaluator.run(root, 'synthetic')
            fit.assert_not_called()
            self.assertEqual(result['fit_calls'], 0); self.assertEqual(result['all_unconditional_draws_preserved'], 64)
            self.assertEqual(set(result['aggregates']), {'bank', 'protected'})
            self.assertEqual(len(result['populations']), 8)
            for p in result['populations']:
                self.assertEqual(len(p['groups']), 355)
                self.assertIn('contact_no_native_entry:orthant:63', p['groups'])
                with np.load(root/p['evaluated_rows'], allow_pickle=False) as saved:
                    self.assertEqual(len(saved['draw']), 8); self.assertTrue(np.isneginf(saved['z'][-1]))
                    self.assertEqual(saved['class_id'][-2], -1)

    def test_freeze_rejects_existing_directory_before_accessing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'keep').write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'Fresh'): evaluator.freeze(root, '/nonexistent')
            self.assertEqual((root/'keep').read_text(), 'preserve')


if __name__ == '__main__': unittest.main()
