"""Synthetic row contracts only: no geometry, scientific inputs or random draws."""
import copy
import math
import unittest
from unittest import mock

import numpy as np
from scipy.special import logsumexp

import native_class_line_physical_reference as previous
import native_class_line_weight_row as validator


def synthetic_rows(*, activity=.3, uncertain=.4):
    """Pure fabricated arithmetic fixtures, reusable by streaming-wrapper tests."""
    region = dict(activity=activity, mahalanobis_radius=4., minimum_mahalanobis_radius=0.,
                  minimum_original_q=0., maximum_original_q=2.,
                  minimum_original_q_inclusive=True, maximum_original_q_inclusive=True)
    manifest = dict(schema=previous.SCHEMA, guide_schema=previous.GUIDE_SCHEMA,
        proposal_kind=previous.PROPOSAL_KIND, proposal_density_measure=previous.MEASURE,
        samples=6, seed=1234567, cloud_replicates=2, importance_uniform_probability=.5,
        importance_component_count=2, activity=activity, lambda_ratio=2.,
        latent_radius=4., minimum_latent_radius=0.,
        minimum_original_q=0., maximum_original_q=2.,
        minimum_original_q_inclusive=True, maximum_original_q_inclusive=True,
        density_trace_contract=dict(format='class-line-compact-v1', diagnostic_full_trace_unchanged=True),
        random_stream_contract=dict(hash_domain='tetramer-uniform-latent-region-v1',
            key_fields=['master seed u64 little endian', 'draw u64 little endian',
                        'cloud index u64 little endian', 'role UTF-8 bytes'],
            proposal_role='latent', proposal_cloud_index=0, physical_cloud_role='cloud',
            physical_cloud_indices='0..cloud_replicates, separately derived; no proposal RNG consumption'))
    manifest['lambda'] = activity*manifest['lambda_ratio'] if activity > 0 else 1.
    rows = []
    for i in range(6):
        u = [5. if i == 4 else .1*i, 0., 0., 0., 0., 0.]
        row = dict(draw=i, latent=u, latent_radius=math.hypot(*u), q=3. if i == 5 else 1.,
                   log_physical_jacobian=-2., log_proposal_density=-4.,
                   capture_valid=i != 3, hard_valid=i not in (2, 3),
                   region_valid=i != 5, shell_valid=i != 4,
                   proposal_branch='uniform-shell' if i == 0 else 'native-class-line',
                   proposal_component=None if i == 0 else i % 2,
                   native_class_line_draw=dict(conditional=False, original_latent=u.copy()),
                   log_hard_weight=None, log_importance_weight=None, clouds=[])
        if i < 2:
            clouds = []
            for raw, overlap in ((3, 1), (12, 9)):
                if activity == 0 or uncertain == 0:
                    raw = overlap = 0
                clouds.append(dict(lower_volume=.2, uncertain_volume=uncertain,
                    upper_volume=.2+uncertain, raw_points=raw, overlap_points=overlap,
                    retained_cells=2 if uncertain else 0, created_cells=5, certified_cells=1,
                    log_weight=activity*.2+overlap*math.log1p(activity/manifest['lambda'])))
            row.update(clouds=clouds, log_hard_weight=2.,
                       log_importance_weight=2.+float(logsumexp([c['log_weight'] for c in clouds]))-math.log(2))
        rows.append(row)
    return manifest, region, rows


def validate_all(manifest, region, rows):
    return [validator.validate_row(r, expected_draw=i, manifest=manifest, region=region)
            for i, r in enumerate(rows)]


class WeightRowTests(unittest.TestCase):
    def test_equivalent_to_existing_arrays_and_noise_with_all_invalid_zeros(self):
        manifest, region, rows = synthetic_rows()
        before = copy.deepcopy(rows)
        old = previous.read_weights(rows, 6, region, dict(alpha=.5, component_count=2))
        new = validate_all(manifest, region, rows)
        self.assertEqual(rows, before)
        for key in ('z', 'h', 'pairs', 'branch', 'component', 'support'):
            np.testing.assert_array_equal(np.asarray([r[key] for r in new]), old[key])
        for key in ('z', 'h'):
            self.assertEqual(previous.physical.statistics.moments([r[key] for r in new]),
                             previous.physical.statistics.moments(old[key]))
        self.assertEqual(previous.physical.statistics.paired_noise(
            np.array([r['z'] for r in new]), np.array([r['pairs'] for r in new])),
            previous.physical.statistics.paired_noise(old['z'], old['pairs']))
        self.assertEqual(sum(r['counters']['attempted'] for r in new), 6)
        self.assertEqual(sum(r['counters']['contributing'] for r in new), 2)
        self.assertEqual([r['counters']['contributing'] for r in new], [1, 1, 0, 0, 0, 0])
        self.assertEqual(sum(r['raw_points'] for r in new), 30)
        self.assertGreater(previous.physical.statistics.paired_noise(old['z'], old['pairs'])['scaled_paired_cloud_variance'], 0)
        for i in range(2, 6):
            self.assertEqual(new[i]['z'], -math.inf)
            self.assertEqual(new[i]['h'], -math.inf)
            self.assertEqual(new[i]['pairs'], [-math.inf, -math.inf])

    def test_zero_activity_and_zero_uncertain_volume(self):
        for activity, uncertain in ((0., .4), (.3, 0.), (0., 0.)):
            with self.subTest(activity=activity, uncertain=uncertain):
                manifest, region, rows = synthetic_rows(activity=activity, uncertain=uncertain)
                result = validate_all(manifest, region, rows)
                for r in result[:2]:
                    self.assertEqual(r['raw_points'], 0)
                    self.assertAlmostEqual(r['z'], r['h']+activity*.2)
                    self.assertEqual(r['pairs'][0], r['pairs'][1])
                bad = copy.deepcopy(rows[0]); bad['clouds'][0]['raw_points'] = 1
                with self.assertRaisesRegex(ValueError, 'Zero-intensity/volume'):
                    validator.validate_row(bad, expected_draw=0, manifest=manifest, region=region)

    def test_two_cloud_arithmetic_mean_not_geometric_mean(self):
        manifest, region, rows = synthetic_rows()
        row = rows[0]
        row['log_importance_weight'] = row['log_hard_weight'] + sum(c['log_weight'] for c in row['clouds'])/2
        with self.assertRaisesRegex(ValueError, 'arithmetic mean'):
            validator.validate_row(row, expected_draw=0, manifest=manifest, region=region)
        with self.assertRaisesRegex(ValueError, 'arithmetic mean'):
            previous.read_weights(rows, 6, region, dict(alpha=.5, component_count=2))

    def test_original_draw_ids_never_renumbered(self):
        manifest, region, rows = synthetic_rows()
        self.assertEqual(validator.validate_row(rows[5], expected_draw=5, manifest=manifest, region=region)['draw'], 5)
        for actual, expected in ((5, 0), (0, 1), (6, 6), (-1, -1), (True, 1), (1, True)):
            with self.subTest(actual=actual, expected=expected):
                row = copy.deepcopy(rows[1]); row['draw'] = actual
                with self.assertRaises(ValueError):
                    validator.validate_row(row, expected_draw=expected, manifest=manifest, region=region)
        with self.assertRaisesRegex(ValueError, 'Missing/repeated'):
            previous.read_weights([rows[1], rows[0], *rows[2:]], 6, region, dict(alpha=.5, component_count=2))

    def test_role_keys_are_separate_and_seed_draw_indexed(self):
        manifest, region, rows = synthetic_rows()
        results = validate_all(manifest, region, rows)
        keys = [key for r in results for key in r['role_keys']]
        self.assertEqual(len(set(keys)), 18)
        expected = [previous.stream_key(manifest['seed'], 5, index, role)
                    for index, role in ((0, 'latent'), (0, 'cloud'), (1, 'cloud'))]
        self.assertEqual(results[5]['role_keys'], expected)
        manifest['seed'] += 1
        changed = validator.validate_row(rows[0], expected_draw=0, manifest=manifest, region=region)
        self.assertTrue(set(changed['role_keys']).isdisjoint(results[0]['role_keys']))
        manifest['random_stream_contract']['physical_cloud_role'] = 'latent'
        with self.assertRaisesRegex(ValueError, 'role contract'):
            validator.validate_row(rows[0], expected_draw=0, manifest=manifest, region=region)

    def test_invalid_support_always_zero_no_clouds(self):
        for i in range(2, 6):
            for field in ('log_hard_weight', 'log_importance_weight', 'clouds'):
                with self.subTest(row=i, field=field):
                    manifest, region, rows = synthetic_rows()
                    rows[i][field] = copy.deepcopy(rows[0][field])
                    with self.assertRaisesRegex(ValueError, 'Invalid/exterior'):
                        validator.validate_row(rows[i], expected_draw=i, manifest=manifest, region=region)
                    with self.assertRaisesRegex(ValueError, 'invalid/exterior'):
                        previous.read_weights(rows, 6, region, dict(alpha=.5, component_count=2))

    def test_scalar_flags_and_branches_reject_tampering(self):
        for mutation in ('hard_boolean', 'capture', 'shell', 'q', 'radius', 'unknown_branch',
                         'component', 'component_boolean', 'uniform_component', 'ray', 'conditional_uniform'):
            with self.subTest(mutation=mutation):
                manifest, region, rows = synthetic_rows(); row = rows[0]
                if mutation == 'hard_boolean': row['hard_valid'] = 1
                if mutation == 'capture': row['capture_valid'] = False
                if mutation == 'shell': row['shell_valid'] = False
                if mutation == 'q': row['q'] = 3.
                if mutation == 'radius': row['latent_radius'] = 1.
                if mutation == 'unknown_branch': row['proposal_branch'] = 'hard-free-line'
                if mutation == 'component': row['proposal_branch'] = 'native-class-line'; row['proposal_component'] = 9
                if mutation == 'component_boolean': row['proposal_branch'] = 'native-class-line'; row['proposal_component'] = True
                if mutation == 'uniform_component': row['proposal_component'] = 0
                if mutation == 'ray': row['selected_ray_fallback'] = False
                if mutation == 'conditional_uniform': row['native_class_line_draw'].update(conditional=True, component=0)
                with self.assertRaises(ValueError):
                    validator.validate_row(row, expected_draw=0, manifest=manifest, region=region)

    def test_conditioned_lineage_and_fallback_counters(self):
        manifest, region, rows = synthetic_rows(); row = rows[1]
        row['native_class_line_draw'].update(conditional=True, component=1, fallback=True)
        result = validator.validate_row(row, expected_draw=1, manifest=manifest, region=region)
        self.assertEqual(result['counters']['conditioned_draws'], 1)
        self.assertEqual(result['counters']['fallback_draws'], 1)
        row['native_class_line_draw']['component'] = 0
        with self.assertRaisesRegex(ValueError, 'lineage'):
            validator.validate_row(row, expected_draw=1, manifest=manifest, region=region)

    def test_poisson_counts_bounds_intensity_and_replica_envelopes(self):
        for mutation in ('missing_cloud', 'extra_cloud', 'wrong_weight', 'overlap_gt_raw', 'bool_count',
                         'upper', 'negative_volume', 'no_retained', 'replica_envelope', 'lambda', 'lambda_zero'):
            with self.subTest(mutation=mutation):
                manifest, region, rows = synthetic_rows(); row = rows[0]; cloud = row['clouds'][0]
                if mutation == 'missing_cloud': row['clouds'].pop()
                if mutation == 'extra_cloud': row['clouds'].append(copy.deepcopy(cloud))
                if mutation == 'wrong_weight': cloud['log_weight'] += .1
                if mutation == 'overlap_gt_raw': cloud['overlap_points'] = 99
                if mutation == 'bool_count': cloud['raw_points'] = True
                if mutation == 'upper': cloud['upper_volume'] += 1.
                if mutation == 'negative_volume': cloud['lower_volume'] = -.1
                if mutation == 'no_retained': cloud['retained_cells'] = 0
                if mutation == 'replica_envelope': row['clouds'][1]['created_cells'] += 1
                if mutation == 'lambda': manifest['lambda'] *= 2
                if mutation == 'lambda_zero': manifest['lambda'] = 0.
                with self.assertRaises(ValueError):
                    validator.validate_row(row, expected_draw=0, manifest=manifest, region=region)

    def test_hard_weight_and_nonfinite_or_boolean_arithmetic(self):
        for field, value in [('log_hard_weight', 3.), ('log_hard_weight', None), ('log_importance_weight', math.nan),
                             ('log_physical_jacobian', True), ('log_proposal_density', math.inf),
                             ('latent_radius', -1.), ('q', math.nan)]:
            with self.subTest(field=field, value=value):
                manifest, region, rows = synthetic_rows(); rows[0][field] = value
                with self.assertRaises(ValueError):
                    validator.validate_row(rows[0], expected_draw=0, manifest=manifest, region=region)

    def test_manifest_and_endpoint_contracts(self):
        for field, value in [('samples', 0), ('samples', True), ('seed', True), ('seed', 2**64),
                             ('cloud_replicates', 1), ('importance_uniform_probability', 0.),
                             ('importance_component_count', False), ('activity', True), ('lambda_ratio', -1.),
                             ('minimum_original_q_inclusive', 1)]:
            with self.subTest(field=field):
                manifest, region, rows = synthetic_rows(); manifest[field] = value
                with self.assertRaises(ValueError):
                    validator.validate_row(rows[0], expected_draw=0, manifest=manifest, region=region)
        manifest, region, rows = synthetic_rows()
        for data in (manifest, region): data['maximum_original_q_inclusive'] = False
        rows[5]['q'] = 2.
        self.assertEqual(validator.validate_row(rows[5], expected_draw=5, manifest=manifest, region=region)['z'], -math.inf)
        rows[5]['q'] = 1.999
        with self.assertRaisesRegex(ValueError, 'original-q predicate'):
            validator.validate_row(rows[5], expected_draw=5, manifest=manifest, region=region)
        # Null/omitted unbounded maxima are equivalent metadata, with no ID remap.
        for data in (manifest, region): data['maximum_original_q'] = None
        rows[0]['q'] = 1.e6
        self.assertTrue(math.isfinite(validator.validate_row(rows[0], expected_draw=0, manifest=manifest, region=region)['z']))

    def test_no_geometry_constructor_or_full_audit_called(self):
        manifest, region, rows = synthetic_rows()
        error = AssertionError('Geometry must not run in scalar row audit')
        with (mock.patch.object(previous, 'NativeContactRegions', side_effect=error),
              mock.patch.object(previous.line, 'Reconstructor', side_effect=error),
              mock.patch.object(previous.line.hard, 'hard_free_intervals', side_effect=error),
              mock.patch.object(previous.line, 'leaf_contact_intervals', side_effect=error),
              mock.patch.object(previous, 'audit', side_effect=error)):
            self.assertEqual(len(validate_all(manifest, region, rows)), 6)


if __name__ == '__main__': unittest.main()
