"""Synthetic saved-stream tests; no protein files, geometry, poses or clouds drawn.

Receipt fixtures are explicitly fabricated attestations, not scientific audits.
Original-target hash constants are patched only to bind these temporary synthetic
bytes. The production entry point has no target-override or synthetic bypass.
"""
from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.stats import t

import analyze_native_class_physical_populations as adapter
import native_class_line_physical_algebra_audit as streaming
import native_class_line_physical_reference as full
from test_contact_population_statistics import fixture as population_fixture
from test_native_class_line_weight_row import synthetic_rows


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def lines(path, rows):
    Path(path).write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=streaming.sha(path))


def fixture(base, counts=(4,), *, scales=None):
    base = Path(base).resolve(); common = base/'common'; common.mkdir()
    shape = dict(synthetic=True, atoms=[]); save(common/'shape.json', shape)
    shape_sha = streaming.sha(common/'shape.json')
    region = dict(activity=.035, depletant_radius=1.5, mahalanobis_radius=4., minimum_mahalanobis_radius=0.,
        minimum_original_q=0., minimum_original_q_inclusive=True, maximum_original_q_inclusive=True,
        shape_sha256=shape_sha, physical_fixed_neighbors=[{'synthetic': 1}, {'synthetic': 2}],
        capture_center=[0., 0., 0.], capture_radius=170., physical_metric={'synthetic': True},
        gaussian_chart=dict(covariances=[np.eye(6).tolist()]))
    save(common/'region.json', region)
    old = dict(region, mahalanobis_radius=5., minimum_original_q=1., minimum_original_q_inclusive=False)
    save(common/'old.json', old); save(common/'native.json', dict(synthetic=True))
    target = {key: ref(common/name) for key, name in [('region', 'region.json'),
              ('old_r5_region', 'old.json'), ('native_definition', 'native.json')]}
    with (mock.patch.object(adapter, 'ORIGINAL_REGION_SHA', target['region']['sha256']),
          mock.patch.object(adapter, 'ORIGINAL_SHAPE_SHA', shape_sha)):
        _, descriptor, target_id = adapter.target_identity(target, adapter.Bindings())
    arms = []; all_rows = {}
    for arm_index, count in enumerate(counts):
        arm = dict(id=f'arm{arm_index}', samples=8, populations=[])
        for p in range(count):
            root = base/f'a{arm_index}-p{p}'; root.mkdir(); prov = root/'provenance'; prov.mkdir()
            seed = 1000+100*arm_index+p
            manifest, _, source = synthetic_rows(activity=.035)
            manifest.update(samples=8, seed=seed, resume_supported=False,
                            compiled_native=dict(source_definition_sha256=target['native_definition']['sha256']))
            manifest.pop('maximum_original_q')
            for name, value, key in [('region.json', region, 'region_sha256'),
                        ('shape.json', shape, 'shape_sha256'), ('input-config.json', {}, 'config_sha256'),
                        ('importance-guide.json', {}, 'importance_guide_sha256'),
                        ('source-bundle.json', {'synthetic': True}, 'source_bundle_sha256')]:
                save(prov/name, value); manifest[key] = streaming.sha(prov/name)
            rows = []
            scale = scales[p] if scales is not None else p+1.
            for i in range(8):
                row = copy.deepcopy(source[0]); row['draw'] = i
                u = [[0.]*6, [2., 0., 0., 0., 0., 0.], [3., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 4., 0., 0.], [0.]*6, [0.]*6, [5., 0., 0., 0., 0., 0.], [0.]*6][i]
                row.update(latent=u, latent_radius=math.hypot(*u), q=1.5 if i == 0 else 1.,
                    log_proposal_density=-4.-math.log(scale),
                    proposal_branch='native-class-line', proposal_component=0,
                    native_class_line_draw=dict(conditional=False, original_latent=u.copy()),
                    region_valid=True, capture_valid=i != 5, hard_valid=i not in (4, 5, 7), shell_valid=i != 6)
                if i < 4:
                    row['log_hard_weight'] = 2.+math.log(scale)
                    row['log_importance_weight'] += math.log(scale)
                else:
                    row.update(log_hard_weight=None, log_importance_weight=None, clouds=[])
                rows.append(row)
            save(root/'manifest.json', manifest); lines(root/'samples.jsonl', rows)
            lines(root/'attempts.jsonl', [dict(draw=i, state='begin') for i in range(8)])
            checked = [adapter.weights.validate_row(row, expected_draw=i, manifest=manifest, region=region)
                       for i, row in enumerate(rows)]
            counters = adapter.Counter()
            for value in checked: counters.update(value['counters'])
            estimate = full.physical.statistics.moments(np.array([v['z'] for v in checked]))
            hard = full.physical.statistics.moments(np.array([v['h'] for v in checked]))
            summary = dict(complete=True, manifest=manifest, samples=8,
                samples_sha256=streaming.sha(root/'samples.jsonl'), attempts_sha256=streaming.sha(root/'attempts.jsonl'),
                sampler_cpu_seconds=.25, estimates=dict(region=estimate, hard_region=hard))
            save(root/'summary.json', summary)
            input_paths = [root/'manifest.json', root/'summary.json', root/'samples.jsonl',
                           root/'attempts.jsonl', prov/'region.json']
            inputs = {str(path): streaming.sha(path) for path in input_paths}
            sources = {str(Path(__file__).resolve()): streaming.sha(__file__)}
            algebra = dict(schema=streaming.SCHEMA, complete=True, passed=True, seed=seed, samples=8,
                root=str(root), all_rows_algebra=8, geometry_certified=False, physical_contact_labels_certified=False,
                input_sha256=inputs, source_sha256=sources, runtime=dict(synthetic_receipt=True),
                counts=dict(counters), estimate=estimate, hard_region=hard,
                samples_sha256=summary['samples_sha256'], attempts_sha256=summary['attempts_sha256'])
            save(root/'algebra.json', algebra)
            sample_lines = (root/'samples.jsonl').read_bytes().splitlines(keepends=True)
            labels = []
            for i, row in enumerate(rows):
                native, contact = i < 2, i < 3
                labels.append(dict(draw=i, sample_record_sha256=adapter.hashlib.sha256(sample_lines[i]).hexdigest(),
                    hard_valid=row['hard_valid'], applicable=i < 4,
                    native=native if i < 4 else None, exclusion_contact=contact if i < 4 else None,
                    old_r5_radius=5. if i < 4 else None, old_capture_valid=True if i < 4 else None,
                    classification=dict(native_any=native, native_anchor_count=int(native),
                        registry_consistent_triangle=False, matches=[]) if i < 4 else None,
                    contact=dict(exclusion_contact=contact, synthetic=True) if i < 4 else None))
            lines(root/'labels.jsonl', labels)
            label_receipt = dict(schema=adapter.LABEL_SCHEMA, complete=True, passed=True, seed=seed, samples=8,
                manifest_sha256=streaming.sha(root/'manifest.json'), samples_sha256=summary['samples_sha256'],
                attempts_sha256=summary['attempts_sha256'], labels=ref(root/'labels.jsonl'),
                target_and_regions_sha256=target_id, input_sha256=inputs, source_sha256=sources,
                runtime=dict(synthetic_receipt=True),
                physical_hard_validity_scope='every_attempt_capture_and_atomic',
                physical_native_contact_scope='every_contributing_pose',
                region_sha256=descriptor['region_sha256'], reference_region_sha256=descriptor['old_r5_region_sha256'],
                shape_sha256=descriptor['shape_sha256'], definition_sha256=descriptor['native_definition_sha256'],
                strata_sha256=descriptor['strata_sha256'])
            save(root/'label-receipt.json', label_receipt)
            arm['populations'].append(dict(id=f'p{p:02}', seed=seed, directory=str(root),
                manifest=ref(root/'manifest.json'), summary=ref(root/'summary.json'),
                algebra=ref(root/'algebra.json'), labels=ref(root/'label-receipt.json')))
            all_rows[(arm_index, p)] = rows
        arms.append(arm)
    plan = dict(schema=adapter.PLAN_SCHEMA, target=target, target_and_regions_sha256=target_id,
                strata=adapter.STRATA, gates=adapter.GATES, arms=arms,
                comparisons=[] if len(arms) == 1 else [dict(left='arm0', right='arm1',
                    historical_failed_strata=[dict(family='orthant', bin=22, region='competing')])])
    path = base/'plan.json'; save(path, plan)
    return path, plan, all_rows


def run(plan_path, plan):
    with (mock.patch.object(adapter, 'ORIGINAL_REGION_SHA', plan['target']['region']['sha256']),
          mock.patch.object(adapter, 'ORIGINAL_SHAPE_SHA', streaming.read(plan['target']['region']['path'])['shape_sha256'])):
        return adapter.analyze(plan_path)


def change_labels(plan_path, plan, mutate):
    slot = plan['arms'][0]['populations'][0]
    receipt_path = Path(slot['labels']['path']); receipt = streaming.read(receipt_path)
    labels_path = Path(receipt['labels']['path'])
    rows = [json.loads(line) for line in labels_path.read_text().splitlines()]
    mutate(rows); lines(labels_path, rows)
    receipt['labels'] = ref(labels_path); save(receipt_path, receipt)
    slot['labels'] = ref(receipt_path); save(plan_path, plan)


class AdapterTests(unittest.TestCase):
    def test_four_and_eight_denominators_covariance_partitions_and_free_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, rows = fixture(tmp, (4, 8))
            result = run(path, plan)
            for arm_index, p in enumerate((4, 8)):
                arm = result['arms'][f'arm{arm_index}']; primary = arm['estimates']['Qz']['primary']
                stats = primary['population_statistics']; row = primary['row_diagnostics']
                self.assertEqual(stats['total_unconditional_draws'], 8*p)
                native = [2*math.exp(rows[(arm_index, j)][0]['log_importance_weight'])/8 for j in range(p)]
                comp = np.asarray(native)/2
                self.assertAlmostEqual(stats['estimates']['native']['linear_mean']['value'], np.mean(native))
                self.assertAlmostEqual(stats['estimates']['native']['linear_population_SE']['value'], np.std(native, ddof=1)/math.sqrt(p))
                contrast = stats['free_energy_contrast']
                self.assertEqual(contrast['degrees_of_freedom'], p-1)
                self.assertAlmostEqual(contrast['Student_t_quantile'], t.ppf(.975, p-1))
                self.assertAlmostEqual(contrast['beta_F_native_minus_competing'], math.log(.5))
                self.assertLess(contrast['halfwidth_95'], 1e-14)
                covariance = stats['covariance_of_population_means']; scales = covariance['per_region_log_scales']
                recovered = np.asarray(covariance['scaled_matrix'])*np.exp(np.add.outer(scales, scales))
                self.assertAlmostEqual(recovered[1, 2], np.cov(native, comp, ddof=1)[0, 1]/p)
                for parent, parts in [('total', ['native', 'competing', 'unbound']), ('native', ['native_old_r5', 'native_remainder'])]:
                    self.assertAlmostEqual(math.exp(row[parent]['logQ']), sum(math.exp(row[x]['logQ']) for x in parts))
                self.assertEqual(row['total']['nonzero'], 4*p)
                self.assertTrue(all(r['draws'] == 8*p for r in row.values()))
            self.assertFalse(result['selected_full_geometry_gate_satisfied'])
            self.assertFalse(result['physical_campaign_gate_open'])
            self.assertEqual(result['new_atom_geometry_queries'], 0)
            json.dumps(result, allow_nan=False)

    def test_row_ess_maximum_and_linear_two_cloud_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, rows = fixture(tmp)
            result = run(path, plan); primary = result['arms']['arm0']['estimates']['Qz']['primary']
            vector = np.array([math.exp(rows[(0, p)][0]['log_importance_weight']) for p in range(4)])
            self.assertAlmostEqual(primary['row_diagnostics']['native_old_r5']['ess'], vector.sum()**2/(vector@vector))
            self.assertAlmostEqual(primary['row_diagnostics']['native_old_r5']['max_fraction'], max(vector)/sum(vector))
            self.assertEqual(primary['row_diagnostics']['native_old_r5']['maximum'], dict(population='p03', draw=0))
            self.assertFalse(primary['quality']['native']['passed'])
            r = rows[(0, 0)][0]
            expected = math.exp(r['log_hard_weight'])*sum(math.exp(c['log_weight']) for c in r['clouds'])/2
            self.assertAlmostEqual(math.exp(r['log_importance_weight']), expected)
            self.assertNotAlmostEqual(r['log_importance_weight'], r['log_hard_weight']+sum(c['log_weight'] for c in r['clouds'])/2)

    def test_original_strata_zero_signs_exteriors_and_historical_empty_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, _ = fixture(tmp, (4, 4))
            result = run(path, plan)
            strata = result['arms']['arm0']['estimates']['Qz']['strata']
            self.assertEqual(len(strata['orthant']), 64)
            self.assertEqual(strata['orthant'][63]['row_diagnostics']['total']['nonzero'], 16)
            self.assertEqual(strata['radial'][0]['row_diagnostics']['total']['nonzero'], 4)
            self.assertEqual(strata['radial'][1]['row_diagnostics']['total']['nonzero'], 4)
            self.assertEqual(strata['radial'][2]['row_diagnostics']['total']['nonzero'], 8)
            empty = strata['orthant'][22]
            self.assertIsNone(empty['population_statistics']['estimates']['competing']['population_relative_SE'])
            self.assertFalse(empty['quality']['competing']['passed'])
            compared = result['comparisons'][0]['kinds']['Qz']['strata']
            old = next(x for x in compared if x['family'] == 'orthant' and x['bin'] == 22 and x['region'] == 'competing')
            self.assertTrue(old['historical_failure']); self.assertFalse(old['material'])
            self.assertFalse(old['comparison']['observed']); self.assertFalse(old['comparison']['passed'])
            self.assertEqual(result['arms']['arm0']['populations'][0]['stratum_attempts']['radial:-1'], 1)

    def test_comparisons_use_population_specific_linear_se_and_log_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, _ = fixture(tmp, (4, 8))
            result = run(path, plan)
            comparison = result['comparisons'][0]['kinds']['Qz']['regions']['native']
            self.assertEqual(comparison['left_population_count'], 4)
            self.assertEqual(comparison['right_population_count'], 8)
            self.assertEqual(comparison['SE_passed'], abs(comparison['scaled_linear_difference']) <=
                3*comparison['scaled_independent_difference_SE']+comparison['scaled_comparison_roundoff_tolerance'])
            self.assertEqual(comparison['absolute_passed'], abs(comparison['log_left_minus_right']) <= .2)
            contrast = result['comparisons'][0]['kinds']['Qz']['free_energy_contrast']
            self.assertTrue(contrast['passed'])

    def test_direct_free_energy_detects_opposite_shifts_when_marginal_masses_pass(self):
        for change in (.05, .1):
            with self.subTest(change=change):
                x = np.array([1., 2., 3., 4.])
                left, lrows = population_fixture(x, x, arm='left')
                right, rrows = population_fixture((1+change)*x, (1-change)*x, arm='right', seed_start=200)
                for region in ('native', 'competing'):
                    self.assertTrue(adapter.population.compare_population_masses(left, lrows, right, rrows, region)['passed'])
                a = dict(declaration=left, population_statistics=adapter.population.summarize_populations(left, lrows))
                b = dict(declaration=right, population_statistics=adapter.population.summarize_populations(right, rrows))
                result = adapter.compare_free_energy(a, b)
                self.assertFalse(result['passed']); self.assertFalse(result['SE_passed'])
                self.assertEqual(result['absolute_passed'], change == .05)
                self.assertLess(result['combined_population_SE'], 1e-14)
        zero, zrows = population_fixture([0.]*4, [1.]*4, arm='zero', seed_start=300)
        result = adapter.compare_free_energy(a, dict(declaration=zero,
            population_statistics=adapter.population.summarize_populations(zero, zrows)))
        self.assertFalse(result['observed']); self.assertFalse(result['passed'])
        self.assertIsNone(result['combined_population_SE'])

    def test_regional_and_stratum_pair_noise_keeps_unconditional_zeros(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, rows = fixture(tmp)
            result = run(path, plan)
            primary = result['arms']['arm0']['estimates']['Qz']['primary']
            for name, selected in [('total', (0, 1, 2, 3)), ('native', (0, 1)),
                                   ('competing', (2,)), ('native_old_r5', (0,))]:
                noise = primary['paired_noise'][name]; pairs = []
                for p in range(4):
                    for i, row in enumerate(rows[(0, p)]):
                        pairs.append([math.exp(row['log_hard_weight']+c['log_weight']-noise['log_weight_offset'])
                                      for c in row['clouds']] if i in selected else [0., 0.])
                a, b = np.asarray(pairs).T; y = (a+b)/2
                self.assertEqual(noise['unconditional_denominator'], 32)
                self.assertAlmostEqual(noise['scaled_total_variance'], np.var(y, ddof=1))
                self.assertAlmostEqual(noise['scaled_paired_cloud_variance'], np.mean((a-b)**2)/4)
                self.assertAlmostEqual(noise['scaled_residual_pose_variance'], np.var(y, ddof=1)-np.mean((a-b)**2)/4)
                self.assertAlmostEqual(noise['scaled_mean_W1W2'], np.mean(a*b))
            strata = result['arms']['arm0']['estimates']['Qz']['strata']
            self.assertIsNone(strata['orthant'][22]['paired_noise']['native'])
            self.assertEqual(strata['orthant'][63]['paired_noise']['native'], primary['paired_noise']['native'])
            # Exactly constant row means with unequal clouds have a negative
            # finite-sample pose residual. The diagnostic must retain it.
            moment = adapter.PairMoment()
            for _ in range(4): moment.add([math.log(1.), math.log(3.)])
            self.assertLess(moment.result(4)['scaled_residual_pose_variance'], 0.)

    def test_mismatched_strata_cannot_be_compared_under_parent_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, _ = fixture(tmp, (4, 4))
            result = run(path, plan)
            a = result['arms']['arm0']['estimates']['Qz']['strata']['radial']
            b = result['arms']['arm1']['estimates']['Qz']['strata']['radial']
            self.assertEqual(a[0]['parent_target_and_regions_sha256'], a[1]['parent_target_and_regions_sha256'])
            self.assertNotEqual(a[0]['declaration']['target_and_regions_sha256'], a[1]['declaration']['target_and_regions_sha256'])
            with self.assertRaisesRegex(ValueError, 'Physical target or complete region'):
                adapter.compare_blocks(a[0], b[1])
            self.assertTrue(adapter.compare_blocks(a[0], b[0])['native']['passed'])

    def test_native_without_contact_preserves_native_mass_and_flags_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, rows = fixture(tmp)
            def anomaly(labels):
                labels[0]['exclusion_contact'] = False
                labels[0]['contact']['exclusion_contact'] = False
            change_labels(path, plan, anomaly)
            result = run(path, plan); arm = result['arms']['arm0']
            self.assertFalse(arm['classifier_contact_consistency_passed'])
            self.assertEqual(arm['populations'][0]['native_entry_unbound_anomalies'], 1)
            primary = arm['estimates']['Qz']['primary']['row_diagnostics']
            expected = sum(2*math.exp(rows[(0, p)][0]['log_importance_weight']) for p in range(4))/32
            self.assertAlmostEqual(math.exp(primary['native']['logQ']), expected)
            self.assertEqual(primary['native']['nonzero'], 8)
            self.assertEqual(primary['unbound']['nonzero'], 4)

    def test_original_target_identity_is_enforced_without_synthetic_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = fixture(tmp)
            with self.assertRaisesRegex(ValueError, 'Original R4 target'):
                adapter.analyze(path)

    def test_bound_inputs_and_fingerprint_tampering_rejected(self):
        for what in ('file', 'fingerprint', 'strata', 'seed'):
            with self.subTest(what=what), tempfile.TemporaryDirectory() as tmp:
                path, plan, _ = fixture(tmp)
                if what == 'file':
                    p = Path(plan['arms'][0]['populations'][0]['directory'])/'samples.jsonl'
                    p.write_text(p.read_text()+'\n')
                elif what == 'fingerprint': plan['target_and_regions_sha256'] = '0'*64
                elif what == 'strata': plan['strata'] = dict(plan['strata'], radial_edges=[0, 4])
                else: plan['arms'][0]['populations'][1]['seed'] = plan['arms'][0]['populations'][0]['seed']
                save(path, plan)
                with self.assertRaises(ValueError): run(path, plan)

    def test_independent_label_receipt_required_not_algebra_geometry_flags(self):
        for field, value in [('physical_hard_validity_scope', 'contributing_only'),
                             ('physical_native_contact_scope', 'guide_interval_membership'),
                             ('reference_region_sha256', '0'*64), ('complete', False)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                path, plan, _ = fixture(tmp); slot = plan['arms'][0]['populations'][0]
                receipt_path = Path(slot['labels']['path']); receipt = streaming.read(receipt_path)
                receipt[field] = value; save(receipt_path, receipt); slot['labels'] = ref(receipt_path); save(path, plan)
                with self.assertRaises(ValueError): run(path, plan)

    def test_label_stream_identity_order_and_invalid_zeros(self):
        mutations = [lambda x: x.pop(), lambda x: x.append(copy.deepcopy(x[-1])),
            lambda x: x[0].update(draw=1), lambda x: x[0].update(sample_record_sha256='0'*64),
            lambda x: x[4].update(hard_valid=True), lambda x: x[4].update(native=True),
            lambda x: x[0].update(native=False)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                path, plan, _ = fixture(tmp); change_labels(path, plan, mutation)
                with self.assertRaises(ValueError): run(path, plan)

    def test_old_r5_strict_q_and_inclusive_radius(self):
        row = dict(draw=0, hard_valid=True, q=1.)
        label = dict(draw=0, sample_record_sha256='s', hard_valid=True, applicable=True, native=True,
            exclusion_contact=True, old_r5_radius=5., old_capture_valid=True,
            classification=dict(native_any=True, native_anchor_count=1, registry_consistent_triangle=False),
            contact=dict(exclusion_contact=True))
        self.assertIn('native_remainder', adapter.label_regions(label, row, dict(z=0.), 's'))
        row['q'] = 1.000001
        self.assertIn('native_old_r5', adapter.label_regions(label, row, dict(z=0.), 's'))
        label['old_r5_radius'] = 5.000001
        self.assertIn('native_remainder', adapter.label_regions(label, row, dict(z=0.), 's'))
        label['old_r5_radius'] = 5.; label['old_capture_valid'] = False
        self.assertIn('native_remainder', adapter.label_regions(label, row, dict(z=0.), 's'))

    def test_maximum_ties_and_extreme_log_moments(self):
        for shift in (-1000., 1000.):
            moment = adapter.Moment()
            for i, value in enumerate([1., 3., 3.]): moment.add(shift+math.log(value), dict(draw=i))
            result = moment.result(8)
            self.assertAlmostEqual(result['logQ'], shift+math.log(7/8))
            self.assertAlmostEqual(result['ess'], 49/19)
            self.assertAlmostEqual(result['max_fraction'], 3/7)
            self.assertEqual(result['maximum'], dict(draw=1))

    def test_no_geometry_observer_or_full_audit_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, plan, _ = fixture(tmp)
            error = AssertionError('No geometry or full density replay permitted')
            with ExitStack() as stack:
                for module, name in [(full, 'NativeContactRegions'), (full.line, 'Reconstructor'),
                    (full.line, 'leaf_contact_intervals'), (full.line.hard, 'hard_free_intervals'),
                    (full, 'audit'), (streaming, 'audit'), (streaming.algebra, 'AlgebraLaw')]:
                    stack.enter_context(mock.patch.object(module, name, side_effect=error))
                result = run(path, plan)
                self.assertTrue(result['complete'])
                self.assertIn(str(Path(adapter.stratify.__code__.co_filename).resolve()), result['source_sha256'])
                self.assertIn(str(Path(adapter.validate_regions.__code__.co_filename).resolve()), result['source_sha256'])


if __name__ == '__main__': unittest.main()
