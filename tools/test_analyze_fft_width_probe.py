"""Deterministic width transform, source-score and paired-ledger controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import analyze_fft_width_probe as a
import test_analyze_factorized_dimer_probe as factorized
import test_analyze_auxiliary_overlap_probe as guided
from test_dimer_destination_density import fixture as density_fixture
import prepare_fft_width_probe as prep


def scaled(original, tau):
    result = copy.deepcopy(original)
    result['base_model']['covariances'] = [[[x * (tau * tau) for x in row] for row in c]
                                          for c in original['base_model']['covariances']]
    return result


def paired_rows():
    result = {}
    for ai in range(3):
        # Widths intentionally have unrelated saved poses/raw traces. Only the
        # within-width pair is identical; cross-width stopping can diverge.
        out = dict(attempts=[dict(root_draws=[{'trace': ai}], internal_draws=[{'index': 1, 'trace': ai}])])
        common = dict(old_coordinates=[{'width': ai}], old_edges=[{'density': ai}],
                      rng_after_fingerprint=[ai, 2, 3, 4], outcome=out)
        u = copy.deepcopy(common)
        u['auxiliary_rng_after_fingerprint'] = [1, 2, 3, 4]
        g = copy.deepcopy(common)
        g['auxiliary_rng_after_fingerprint'] = [5, 6, 7, 8]
        g['outcome']['guidance'] = dict(point_count=20, old_count=5, integer_draws=[1, 2, 4, 3], threshold=4, m=4)
        g['outcome']['attempts'][0]['internal_draws'][0]['guidance_count'] = 4
        result[ai, 'unguided'], result[ai, 'm4'] = u, g
    return result


class WidthAuditTests(unittest.TestCase):
    def test_full_covariance_transform_preserves_means_and_reciprocal_centers(self):
        original = density_fixture(correlated=True)
        base = a.DimerDestinationDensity(original)
        latent = [.1, -.2, .3, -.4, .5, -.6]
        for tau in a.SCALES:
            model = scaled(original, tau)
            a.validate_scaled(original, model, tau)
            helper = a.DimerDestinationDensity(model)
            for branch in (0, 1):
                checks = a.Checks()
                checks.pose(helper.decode(branch, [0.] * 6), base.decode(branch, [0.] * 6), 'fixed center')
                checks.pose(helper.decode(branch, latent), base.decode(branch, [tau * x for x in latent]), 'correlated width')
                self.assertEqual(checks.failures, [])
            self.assertEqual(helper.map_density.weights.tolist(), base.map_density.weights.tolist())
        for defect in ('mean', 'cross_covariance', 'reciprocal', 'weight', 'angular_length', 'tau_not_squared'):
            value = scaled(original, .25)
            if defect == 'mean': value['base_model']['means'][0][0] += .1
            elif defect == 'cross_covariance': value['base_model']['covariances'][0][3][0] = 0.
            elif defect == 'reciprocal': value['reciprocal_components'][0] = False
            elif defect == 'weight': value['base_model']['weights'][0] += .1
            elif defect == 'angular_length': value['base_model']['angular_length'] += 1.
            else: value['base_model']['covariances'] = (np.asarray(original['base_model']['covariances']) * .25).tolist()
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                a.validate_scaled(original, value, .25)
        with self.assertRaises(ValueError): a.validate_scaled(original, original, 1.)

    def test_every_null_source_density_is_scored_at_its_own_width(self):
        original = density_fixture(correlated=True)
        base = a.DimerDestinationDensity(original)
        coords = [base.decode(0, [.6] * 6), base.decode(1, [.2] * 6)]
        helpers = [a.DimerDestinationDensity(scaled(original, tau)) for tau in a.SCALES]
        values = []
        for helper in helpers:
            with patch.object(helper.score, 'evaluate', side_effect=AssertionError('obsolete factor')):
                values.append([helper.edge_density(p, .5, 160., kind='map') for p in coords])
        self.assertEqual(len({v[1]['log_learned'] for v in values}), 3)
        for i in range(3):
            row = dict(old_coordinates=coords, old_edges=values[i], outcome=dict(candidate=None))
            checks = a.Checks()
            a.audit_source_density(row, coords, values[i], checks, 'null')
            self.assertEqual(checks.failures, [])
            bad = copy.deepcopy(row)
            bad['old_edges'] = values[(i + 1) % 3]
            checks = a.Checks()
            a.audit_source_density(bad, coords, values[i], checks, 'wrong width')
            self.assertTrue(checks.failures)
            bad = copy.deepcopy(row)
            del bad['old_edges'][1]['log_uniform']
            with self.assertRaises(ValueError): a.audit_source_density(bad, coords, values[i], a.Checks(), 'missing')

    def test_candidate_source_density_cannot_override_explicit_width_source(self):
        h, o, case, old, anchor, out, contacts, source = factorized.fixture()
        coords, expected = a.source_densities(h, old, anchor)
        row = dict(old_coordinates=coords, old_edges=expected, outcome=out)
        a.audit_source_density(row, coords, expected, a.Checks(), 'candidate')
        out['candidate']['diagnostics']['old_edges'][1]['log_full'] += .1
        with self.assertRaisesRegex(ValueError, 'explicit width source'):
            a.audit_source_density(row, coords, expected, a.Checks(), 'bad')

    def test_unguided_never_evaluates_overlap_and_keeps_counts_missing(self):
        h, o, case, old, anchor, out, contacts, source = factorized.fixture()
        row = dict(method='unguided', outcome=out, raw_contacts=contacts)
        counter = a.CountOracle(o.centers, o.radii + o.rd)
        with patch.object(counter, 'relative', side_effect=AssertionError('unguided count')), \
             patch.object(counter, 'world', side_effect=AssertionError('unguided count')):
            checks = a.Checks()
            result = a.audit_outcome(h, o, counter, None, case, old, anchor, row, source, checks, 'unguided')
        self.assertEqual(checks.failures, [])
        self.assertIsNone(result['old_count'])
        self.assertIsNone(result['candidate']['count_retention'])
        self.assertIsNone(result['candidate']['new_count'])
        self.assertEqual(result['count_queries'], 0)
        self.assertEqual(result['candidate']['complete_log_correction'], result['candidate']['log_reverse_forward'])
        out['attempts'][0]['internal_draws'][0]['guidance_count'] = 0
        with self.assertRaisesRegex(ValueError, 'Unexpected unguided'):
            a.audit_outcome(h, o, counter, None, case, old, anchor, row, source, a.Checks(), 'bad')

    def test_guided_reuses_exact_geometry_count_and_full_density_audit(self):
        h, o, counter, points, case, old, anchor, out, contacts, source, m = guided.fixture(m=4)
        row = dict(method='m4', outcome=out, raw_contacts=contacts)
        checks = a.Checks()
        result = a.audit_outcome(h, o, counter, points, case, old, anchor, row, source, checks, 'm4')
        self.assertEqual(checks.failures, [])
        self.assertIsNotNone(result['candidate']['count_retention'])
        out['attempts'][0]['internal_draws'][0]['guidance_count'] += 1
        with self.assertRaisesRegex(ValueError, 'guidance count'):
            a.audit_outcome(h, o, counter, points, case, old, anchor, row, source, a.Checks(), 'bad')

    def test_pairing_does_not_assume_cross_width_raw_draw_alignment(self):
        rows = paired_rows()
        checks = a.Checks()
        a.shared_cloud_thresholds(rows, checks, 'all widths')
        self.assertEqual(checks.failures, [])
        for defect in ('root', 'internal', 'threshold', 'rng', 'source_score', 'missing'):
            bad = copy.deepcopy(rows)
            if defect == 'root': bad[1, 'm4']['outcome']['attempts'][0]['root_draws'][0]['trace'] += 1
            elif defect == 'internal': bad[1, 'm4']['outcome']['attempts'][0]['internal_draws'][0]['trace'] += 1
            elif defect == 'threshold': bad[1, 'm4']['outcome']['guidance']['integer_draws'][1] += 1
            elif defect == 'rng': bad[1, 'm4']['rng_after_fingerprint'][0] += 1
            elif defect == 'source_score': bad[1, 'm4']['old_edges'][0]['density'] += 1
            else: del bad[1, 'unguided']
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                a.shared_cloud_thresholds(bad, a.Checks(), 'bad')

    def test_domain_seeds_order_and_exact_allocation(self):
        roles = ['cloud', 'proposal', 'threshold'] + [f'execution_order/{i}/{m}' for i in range(3) for m in a.METHODS]
        seeds = [a.seed(a.MASTER, ci, j, role) for ci in range(8) for j in range(32) for role in roles]
        self.assertEqual(len(set(seeds)), 2304)
        for ci in range(8):
            for j in range(32):
                order = a.method_order(ci, j)
                self.assertEqual({(i, m) for i, m, _ in order}, {(i, m) for i in range(3) for m in a.METHODS})
                self.assertEqual(order, sorted(order, key=lambda r: (r[2], r[0], r[1])))
        self.assertEqual(a.expected_allocation(), prep.allocation())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'allocation.json'
            path.write_text(json.dumps(prep.allocation()))
            record = dict(path=str(path), sha256=a.sha(path))
            atlases = [dict(name=name, tau=tau, model={}) for name, tau in zip(a.NAMES, a.SCALES)]
            config = dict(schema='fft-width-screen-v1', density_law='map-factor-full-mixture-v1', master_seed=a.MASTER,
                root_cap=32, internal_cap=32, factorized_joint_cap=1, attempts_per_context=32, cloud_raw_count=16384,
                factorized_order='root_first', scaled_atlases=atlases, scientific_allocation=record)
            protocol = dict(master_seed=a.MASTER, scaled_atlases=atlases, scientific_allocation=record, allocation=prep.allocation(),
                caps=a.CAPS, order='root_first', density_tolerance=dict(absolute=2e-7, relative=2e-10),
                physical_bath_draws=0, state_updates=0, native_classification=False, cloud_law=dict(raw_count=16384, is_poisson=False),
                source_atlas_index=1, source_atlas_sha256=a.FFT_SHA, covariance_scales=a.SCALES, methods=a.METHODS,
                uniform_probability=.5, uniform_half_width_A=160., density_law=config['density_law'])
            a.validate_contract(config, protocol)
            for defect in ('slots', 'uniform', 'scale', 'extension', 'physical', 'tolerance'):
                c, p = copy.deepcopy(config), copy.deepcopy(protocol)
                if defect == 'slots': c['attempts_per_context'] += 1
                elif defect == 'uniform': p['uniform_probability'] = .25
                elif defect == 'scale': c['scaled_atlases'][0]['tau'] = 1.
                elif defect == 'extension': p['allocation']['extension'] = True
                elif defect == 'physical': p['physical_bath_draws'] = 1
                else: p['density_tolerance']['absolute'] = 1.
                with self.subTest(defect=defect), self.assertRaises(ValueError): a.validate_contract(c, p)

    def test_summary_retains_null_denominators_and_unmeasured_counts(self):
        common = dict(raw_edge_draws=32, root_draws=32, internal_draws=0, assembled=0, threshold_failures=0,
                      internal_geometric_passes=0, count_queries=0, point_tests=0, old_count=None, threshold=None,
                      **{key: .01 for key in a.TIMINGS})
        null = dict(common, status='cap_exhausted', stage_status='root_cap_exhausted', candidate=None)
        candidate = dict(common, status='candidate', stage_status='candidate', candidate=dict(
            old_count=None, new_count=None, count_retention=None, log_reverse_forward=-3.,
            aux_log_correction=0., complete_log_correction=-3., external_contacts=0, both_intended_contacts=False))
        result = a.summarize([null, candidate])
        self.assertEqual((result['outer_attempts'], result['candidate_count'], result['proposal_nulls']), (2, 1, 1))
        self.assertEqual(result['candidate_fraction'], .5)
        self.assertEqual(result['count_retention']['count'], 0)
        self.assertIsNone(result['count_retention']['quantiles'])


if __name__ == '__main__':
    unittest.main()
