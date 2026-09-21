"""Fixed-N strata, paired errors, and fail-closed saved-result bindings."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_mobile_native_pocket import (
    ARMS, SEEDS, independent_shell_fraction, label_keys, nested_masks,
    paired_noise, ratio, read_weights, sha, sum_regions, summarize,
    target_equal, validate_jobs, validate_parent, validate_regions)


def log(values):
    with np.errstate(divide='ignore'):
        return np.log(np.asarray(values, float))


def population(index, hard, cloud1, cloud2, radii=None):
    hard = np.asarray(hard, float)
    pairs = np.column_stack((hard*np.asarray(cloud1), hard*np.asarray(cloud2)))
    return dict(id=f'r{index:02d}', seed=SEEDS[index], z=log(pairs.mean(axis=1)),
        h=log(hard), pairs=log(pairs), radius=np.asarray(radii if radii is not None else [1.]*len(hard)))


class PocketAnalysisTests(unittest.TestCase):
    def test_nested_subsets_keep_full_attempted_denominator_and_boundaries(self):
        pops = [population(i, [1., 0., 1., 1.], [2.]*4, [2.]*4, [3., 4., 4., 5.]) for i in range(4)]
        masks = [nested_masks(p['radius']) for p in pops]
        np.testing.assert_array_equal(masks[0]['r4'], [True, True, True, False])
        np.testing.assert_array_equal(masks[0]['shell4to5'], [False, False, False, True])
        a = summarize(pops, 'R4', [m['r4'] for m in masks])
        b = summarize(pops, '4–5', [m['shell4to5'] for m in masks])
        total = summarize(pops, 'R5')
        self.assertEqual(a['row_uncertainty']['draws'], 16)
        self.assertAlmostEqual(a['row_uncertainty']['log_Qz'], 0.)
        self.assertAlmostEqual(math.exp(a['row_uncertainty']['log_Qz'])+math.exp(b['row_uncertainty']['log_Qz']),
                               math.exp(total['row_uncertainty']['log_Qz']))
        self.assertAlmostEqual(a['fraction_of_parent']['Qz']['row_uncertainty']['observed_fraction'], 2/3)
        self.assertGreater(a['fraction_of_parent']['Qz']['row_uncertainty']['SE'], 0.)

    def test_correlated_enhancement_cancels_pose_noise_for_constant_bath(self):
        pops = [population(i, [1., 3., 0., 2.], [7.]*4, [7.]*4) for i in range(4)]
        value = summarize(pops, 'constant bath')
        self.assertAlmostEqual(value['row_uncertainty']['log_enhancement'], math.log(7))
        self.assertAlmostEqual(value['row_uncertainty']['log_enhancement_SE'], 0., places=14)
        self.assertGreater(value['row_uncertainty']['Qz_relative_SE'], 0.)
        self.assertAlmostEqual(value['population_uncertainty']['Qz_relative_SE'], 0.)
        self.assertAlmostEqual(value['paired_cloud_noise']['relative_variance_mean_cloud'], 0.)

    def test_cloud_noise_keeps_invalid_zeros_and_exact_variance_formula(self):
        p = population(0, [1., 0., 1., 0.], [2., 0., 4., 0.], [4., 0., 8., 0.])
        a = paired_noise(p['z'], p['pairs'])
        y = np.array([3., 0., 6., 0.])
        cloud = np.mean(np.array([2., 0., 4., 0.])**2)/4
        self.assertAlmostEqual(a['relative_variance_mean_cloud'], cloud/4/y.mean()**2)
        self.assertAlmostEqual(a['cloud_fraction'], cloud/np.var(y, ddof=1))
        bad = p['pairs'].copy();bad[1] = 0.
        with self.assertRaisesRegex(ValueError, 'support'):
            paired_noise(p['z'], bad)

    def test_independent_strata_sum_and_shell_shared_denominator(self):
        core = summarize([population(i, [1., 0., 1., 0.], [2.+i]*4, [2.+i]*4) for i in range(4)], 'core')
        shell = summarize([population(i, [2., 1., 0., 1.], [4.+i]*4, [4.+i]*4) for i in range(4)], 'shell')
        combined = sum_regions([core, shell], 'R8')
        a, b = core['row_uncertainty'], shell['row_uncertainty']
        ca, cb = math.exp(a['log_Qz']), math.exp(b['log_Qz'])
        self.assertAlmostEqual(math.exp(combined['row_uncertainty']['log_Qz']), ca+cb)
        f = cb/(ca+cb)
        fraction = independent_shell_fraction(core, shell)['row_uncertainty']['Qz']
        self.assertAlmostEqual(fraction['observed_fraction'], f)
        self.assertAlmostEqual(fraction['SE'], f*(1-f)*math.hypot(a['Qz_relative_SE'], b['Qz_relative_SE']))
        wanted = (1-f)**2*a['Qz_relative_SE']**2+f*f*b['Qz_relative_SE']**2
        self.assertAlmostEqual(combined['row_uncertainty']['Qz_relative_SE']**2, wanted)
        self.assertNotAlmostEqual(ca+cb, (ca+cb)/2)  # strata are integrals, not concatenated draws

    def test_unobserved_subregion_is_unresolved_not_a_bound(self):
        pops = [population(i, [1., 0.], [2., 0.], [2., 0.]) for i in range(4)]
        none = summarize(pops, 'unobserved', [[False, False]]*4)
        self.assertIsNone(none['row_uncertainty']['log_Qz'])
        self.assertEqual(none['fraction_of_parent']['Qz']['row_uncertainty']['observed_fraction'], 0.)
        self.assertIsNone(none['fraction_of_parent']['Qz']['row_uncertainty']['SE'])
        combined = sum_regions([summarize(pops, 'seen'), none], 'sum')
        self.assertIn('unresolved', combined)

    def test_population_errors_retain_independent_mean_scatter(self):
        pops = [population(i, [1.]*4, [v]*4, [v]*4) for i, v in enumerate([1., 1., 1., 20.])]
        result = summarize(pops, 'unequal means')
        self.assertGreater(result['population_uncertainty']['Qz_relative_SE'], result['row_uncertainty']['Qz_relative_SE'])
        self.assertGreater(result['population_uncertainty']['log_enhancement_SE'], 0.)

    def test_missing_duplicate_seed_or_failed_jobs_fail_closed(self):
        jobs = [dict(id=f'r{i:02d}', seed=SEEDS[i], samples=8192) for i in range(4)]
        terminal = [dict(j, arm='r5', returncode=0) for j in jobs]
        validate_jobs(jobs, terminal, 'r5')
        cases = [(jobs[:-1], terminal), (jobs, terminal[:-1])]
        bad = copy.deepcopy(jobs);bad[1]['seed'] = bad[0]['seed'];cases.append((bad, terminal))
        bad = copy.deepcopy(terminal);bad[1]['returncode'] = 1;cases.append((jobs, bad))
        for a, b in cases:
            with self.assertRaises(ValueError):validate_jobs(a, b, 'r5')

    def test_target_and_chart_changes_fail_closed(self):
        root = Path(__file__).resolve().parents[1]/'runs/mobile-native-pocket-campaign-20260921'
        cfg = json.loads((root/'r5/provenance/config.json').read_text())
        shape = json.loads((root/'r5/provenance/shape.json').read_text())
        regions = {name:json.loads((root/name/'provenance/region.json').read_text()) for name in ARMS}
        shape_sha = sha(root/'r5/provenance/shape.json')
        validate_regions(regions, cfg, shape, shape_sha)
        for field, value in [('activity', .04), ('physical_fixed_neighbors', list(reversed(cfg['fixed_poses']))), ('shape_sha256', 'wrong')]:
            changed = copy.deepcopy(regions['r5']);changed[field] = value
            with self.assertRaisesRegex(ValueError, 'differs'):target_equal(cfg, changed, shape_sha)
        changed = copy.deepcopy(regions);changed['shell5to8']['gaussian_chart']['covariances'][0][0][0] *= 2
        with self.assertRaisesRegex(ValueError, 'charts differ'):validate_regions(changed, cfg, shape, shape_sha)
        changed = copy.deepcopy(regions);changed['r5']['minimum_original_q_inclusive'] = True
        with self.assertRaisesRegex(ValueError, 'q>1'):validate_regions(changed, cfg, shape, shape_sha)

    def test_raw_zeros_draw_sequence_and_cloud_mean_fail_closed(self):
        valid = dict(draw=0, latent_radius=1., q=2., hard_valid=True, capture_valid=True, region_valid=True,
            log_hard_weight=0., log_importance_weight=math.log(2), clouds=[dict(log_weight=math.log(2))]*2)
        invalid = dict(valid, draw=1, hard_valid=False, log_hard_weight=None, log_importance_weight=None, clouds=[])
        rows = [valid, invalid]
        result = read_weights(rows, 2, dict(mahalanobis_radius=5.))
        self.assertTrue(np.isneginf(result['z'][1]))
        for key, value in [('draw', 0), ('log_importance_weight', 0.), ('clouds', [dict(log_weight=1.)])]:
            bad = copy.deepcopy(rows);bad[1][key] = value
            with self.assertRaises(ValueError):read_weights(bad, 2, dict(mahalanobis_radius=5.))
        bad = copy.deepcopy(rows);bad[0]['log_importance_weight'] += .01
        with self.assertRaisesRegex(ValueError, 'Cloud mean'):read_weights(bad, 2, dict(mahalanobis_radius=5.))

    def test_all_matches_and_target_triangle_remain_separate_labels(self):
        label = dict(native_any=True, cooperative_entry=True, registry_consistent_triangle=True,
            matches=[dict(anchor_index=0, motif_id=7), dict(anchor_index=0, motif_id=9), dict(anchor_index=1, motif_id=4)],
            registry_consistent_triangles=[dict(anchor0_to_moving_motif=7, anchor1_to_moving_motif=4)])
        keys = label_keys(label)
        self.assertTrue({'anchor0_motif7', 'anchor0_motif9', 'anchor1_motif4', 'motif7_4_triangle'} <= keys)
        label['registry_consistent_triangles'][0]['anchor1_to_moving_motif'] = 8
        self.assertNotIn('motif7_4_triangle', label_keys(label))
        self.assertIn('registry_triangle', label_keys(label))

    def test_terminal_and_frozen_input_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);(root/'input').write_text('fixed')
            protocol = dict(schema='mobile-native-pocket-controller-v1', campaigns=[dict(arm=a) for a in ARMS],
                samples_per_population=8192, populations_per_region=4, cloud_replicates=2, lambda_ratio=64., total_jobs=8)
            (root/'protocol.json').write_text(json.dumps(protocol))
            status = dict(complete=True, phase='complete', protocol_sha256=sha(root/'protocol.json'),
                jobs=[dict(arm=a, id=f'r{i:02d}') for a in ARMS for i in range(4)], audits={a:{} for a in ARMS})
            (root/'status.json').write_text(json.dumps(status))
            (root/'freeze.json').write_text(json.dumps(dict(files={'input':sha(root/'input')})))
            validate_parent(root)
            (root/'input').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Frozen campaign'):validate_parent(root)
            (root/'input').write_text('fixed');status['complete'] = False
            (root/'status.json').write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError, 'incomplete'):validate_parent(root)


if __name__ == '__main__':
    unittest.main()
