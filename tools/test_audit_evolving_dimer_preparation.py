"""Synthetic saved-data corruption checks; no protein draws or geometry queries."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

import audit_evolving_dimer_preparation as a
import test_analyze_factorized_dimer_probe as f


def fixture():
    helper, oracle, case, old, anchor, outcome, _, _ = f.fixture()
    candidate = outcome['candidate']
    selected = [candidate['root'], candidate['child']]
    saved = dict(source=old, selected=selected, status='prepared', is_equilibrium_sample=False, attempts=1)
    rows = [dict(attempt=1, outcome=outcome, selected=True)]
    preparation = dict(outer_cap=256, minimum_max_center_displacement_A=0., minimum_max_body_orientation_degrees=10.)
    return rows, saved, helper, oracle, case, old, anchor, preparation


class PreparationAuditTests(unittest.TestCase):
    def test_cloud_reconstruction_closed_boundary_and_empty_retention(self):
        oracle = a.CountOracle([[0., 0., 0.]], [1.])
        uniforms = np.asarray([[.5, .5, .5], [.9, .9, .9], [0., .5, .5]], dtype='<f8')
        meta = dict(raw_count=3, low=[-1.]*3, high=[1.]*3, kept_indices=[0, 2], cpu_seconds=.01)
        checks = a.Checks()
        result = a.audit_cloud(meta, uniforms.tobytes(), oracle, checks, 'toy', expected_count=3)
        self.assertEqual(result['retained_points'], 2)
        self.assertFalse(checks.failures)
        meta.update(raw_count=1, kept_indices=[])
        self.assertEqual(a.audit_cloud(meta, uniforms[1:2].tobytes(), oracle, a.Checks(), 'empty', 1)['retained_points'], 0)

    def test_cloud_indices_bounds_and_variates_are_independently_checked(self):
        oracle = a.CountOracle([[0., 0., 0.]], [1.])
        original = dict(raw_count=2, low=[-1.]*3, high=[1.]*3, kept_indices=[0], cpu_seconds=0.)
        raw = np.asarray([[.5, .5, .5], [.9, .9, .9]], dtype='<f8').tobytes()
        for kind in ['missing', 'extra', 'duplicate', 'boolean', 'bounds', 'count', 'truncated', 'extended', 'one', 'nan']:
            meta, blob = copy.deepcopy(original), raw
            if kind == 'missing': meta['kept_indices'] = []
            elif kind == 'extra': meta['kept_indices'] = [0, 1]
            elif kind == 'duplicate': meta['kept_indices'] = [0, 0]
            elif kind == 'boolean': meta['kept_indices'] = [False]
            elif kind == 'bounds': meta['low'][0] += .1
            elif kind == 'count': meta['raw_count'] = 3
            elif kind == 'truncated': blob = blob[:-8]
            elif kind == 'extended': blob += blob[:8]
            else:
                value = np.frombuffer(blob, dtype='<f8').copy()
                value[0] = 1. if kind == 'one' else math.nan
                blob = value.tobytes()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                a.audit_cloud(meta, blob, oracle, a.Checks(), kind, expected_count=2)

    def test_preparation_uses_or_rule_and_quaternion_sign_invariance(self):
        identity = dict(position=[0.]*3, orientation=[1., 0., 0., 0.])
        old = [copy.deepcopy(identity), copy.deepcopy(identity)]
        p = dict(minimum_max_center_displacement_A=5., minimum_max_body_orientation_degrees=10.)
        for defect in ['translation', 'rotation', 'opposite_sign', 'neither']:
            new = copy.deepcopy(old)
            if defect == 'translation': new[0]['position'][0] = 5.
            elif defect == 'rotation': new[1]['orientation'] = [math.cos(math.pi/8), math.sin(math.pi/8), 0., 0.]
            elif defect == 'opposite_sign': new[0]['orientation'][0] = -1.
            else: new[0]['position'][0] = 4.9
            self.assertEqual(a.qualifies(old, new, p), defect in ['translation', 'rotation'])

    def test_full_density_and_raw_generation_checked_without_sampling(self):
        inputs = fixture()
        checks = a.Checks()
        with patch.object(np.random, 'default_rng', side_effect=AssertionError('no sampling')):
            result = a.audit_start(*inputs, checks, 'synthetic')
        self.assertFalse(checks.failures)
        self.assertEqual(len(result['attempts']), 1)
        self.assertTrue(result['attempts'][0]['candidate']['feasible'])
        for name in ['log_reverse_forward', 'log_tree_coordinate_jacobian', 'selection_log_reverse_forward']:
            values = fixture()
            values[0][0]['outcome']['candidate']['diagnostics'][name] += .2
            checks = a.Checks()
            a.audit_start(*values, checks, 'mutated')
            self.assertTrue(checks.failures, name)

    def test_every_source_and_candidate_geometry_is_independently_checked(self):
        for defect in ['source', 'root', 'frame', 'internal', 'final', 'selected']:
            values = fixture()
            row = values[0][0]
            outcome = row['outcome']
            if defect == 'source': outcome['source_feasibility']['wall_valid'][0] = False
            elif defect == 'root': outcome['attempts'][0]['root_draws'][0]['feasibility']['wall_valid'] = False
            elif defect == 'frame': outcome['source_frame']['reconstructed_root']['wall_valid'] = False
            elif defect == 'internal': outcome['attempts'][0]['internal_draws'][0]['feasibility']['internal_core_overlap'] = True
            elif defect == 'final': outcome['attempts'][0]['final_feasibility']['wall_valid'][1] = False
            else: values[1]['selected'][0]['position'][0] += 1.
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                a.audit_start(*values, a.Checks(), defect)

    def test_ledger_gaps_early_success_hidden_retry_and_fatal_rejected(self):
        for defect in ['gap', 'missing', 'selected_false', 'retry', 'fatal', 'equilibrium', 'guided']:
            values = fixture()
            rows, saved = values[:2]
            if defect == 'gap': rows[0]['attempt'] = 2
            elif defect == 'missing': saved['attempts'] = 2
            elif defect == 'selected_false': rows[0]['selected'] = False
            elif defect == 'retry':
                rows.append(copy.deepcopy(rows[0])); rows[-1]['attempt'] = 2; saved['attempts'] = 2
            elif defect == 'fatal': rows[0]['failure'] = {'reason': 'fatal'}
            elif defect == 'equilibrium': saved['is_equilibrium_sample'] = True
            else: rows[0]['outcome']['guidance'] = {'m': 4}
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                a.audit_start(*values, a.Checks(), defect)

    def test_nulls_are_retained_before_first_qualified_success(self):
        values = list(fixture())
        rows, saved, helper, oracle, case, old, anchor, preparation = values
        null = copy.deepcopy(rows[0])
        outcome = null['outcome']
        trial = outcome['attempts'][0]
        edge = copy.deepcopy(trial['root_draws'][0])
        f.change_uniform(edge, dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]))
        edge.update(world_pose=anchor, feasibility=f.a.root_geometry(oracle, case, anchor))
        trial.update(status='root_cap_exhausted', root_draws=[dict(copy.deepcopy(edge), index=i+1) for i in range(32)],
                     internal_draws=[], proposed=None, final_feasibility=None, frame=None)
        outcome.update(status='cap_exhausted', candidate=None)
        null['selected'] = False
        rows[0]['attempt'] = 2
        rows.insert(0, null)
        saved['attempts'] = 2
        checks = a.Checks()
        result = a.audit_start(*values, checks, 'null then candidate')
        self.assertFalse(checks.failures)
        self.assertEqual(len(result['attempts']), 2)
        self.assertIsNone(result['attempts'][0]['candidate'])
        self.assertEqual(result['attempts'][0]['root_draws'], 32)

    def test_binding_requires_exact_hash_and_copies_transitive_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'first.py').write_text('from second import value\n')
            (root/'second.py').write_text('import third\nvalue=1\n')
            (root/'third.py').write_text('pass\n')
            closure = a.dependencies([root/'first.py'])
            self.assertEqual(set(closure), {'first.py', 'second.py', 'third.py'})
            bound = a.record(root/'first.py')
            self.assertEqual(a.checked_file(bound), root/'first.py')
            (root/'first.py').write_text('pass\n')
            with self.assertRaisesRegex(ValueError, 'Changed bound'):
                a.checked_file(bound)


if __name__ == '__main__':
    unittest.main()
