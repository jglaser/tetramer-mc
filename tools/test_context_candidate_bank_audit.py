"""Deterministic bank-reference controls; no protein poses or random draws."""
import copy
import json
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

import audit_context_candidate_bank as audit
from context_prior_metrics import fingerprint


def pose(position, rotation=None):
    matrix = np.eye(3) if rotation is None else np.asarray(rotation)
    q = Rotation.from_matrix(matrix).as_quat()
    return dict(position=list(position), orientation=q[[3, 0, 1, 2]].tolist())


def model():
    lower = np.diag([.7, .8, 1.1, .4, .6, .9])
    lower[3, 0] = .18
    lower[4, 1] = -.11
    lower[5, 2] = .23
    return dict(schema='reciprocal-pose-mixture-v1', reciprocal_components=[True, False],
        base_model=dict(coordinate_convention='anchor-body-relative', angular_length=1.7,
            anchors=[dict(position=[.3, -.2, .1], rotation=Rotation.from_rotvec([.2, -.3, .1]).as_matrix().tolist()),
                     dict(position=[-1., .4, .2], rotation=Rotation.from_rotvec([-.4, .1, .2]).as_matrix().tolist())],
            means=[[.1, -.2, .05, .15, -.1, .2], [-.3, .1, .2, -.1, .3, .05]],
            covariances=[(lower@lower.T).tolist(), ((1.2*lower)@(1.2*lower).T).tolist()],
            weights=[.4, .6]))


def compose(anchor, relative):
    a = Rotation.from_quat(np.asarray(anchor['orientation'])[[1, 2, 3, 0]]).as_matrix()
    r = Rotation.from_quat(np.asarray(relative['orientation'])[[1, 2, 3, 0]]).as_matrix()
    return pose(a@relative['position']+anchor['position'], a@r)


def independent_component(reference, relative, branch):
    t = np.asarray(relative['position'])
    r = Rotation.from_quat(np.asarray(relative['orientation'])[[1, 2, 3, 0]]).as_matrix()
    if reference.density.inverted[branch]:
        t, r = -r.T@t, r.T
    anchor = reference.density.anchors[branch]
    quaternion = Rotation.from_matrix(r@np.asarray(anchor['rotation']).T).as_quat()
    c = quaternion[:3]/quaternion[3]
    x = np.r_[t-anchor['position'], reference.density.ell*c]
    lower = reference.density.lower[branch]
    gaussian = multivariate_normal.logpdf(x, mean=reference.density.mean[branch], cov=lower@lower.T)
    log_volume_x = -3*math.log(reference.density.ell)-2*math.log(math.pi)-2*math.log1p(c@c)
    return gaussian-log_volume_x


def reference_tokens():
    return [[77, 217, f'moving{i}', f'fixed{i}'] for i in range(16)]


def preflight_row(ordinal, tokens, valid=True, branch='uniform'):
    identity = dict(start='fixed', stream=0, arm='context')
    log_g = math.log(.5)
    bank = dict(ordinal=ordinal, cycle=ordinal+1, slot=4, attempt_index=5*ordinal+4,
        event_index=17+16*ordinal, identity=identity, proposed_pose=pose([0., 0., 0.]),
        branch=branch, wall_valid=valid, core_valid=True if valid else None,
        status='bath_rejected' if valid else 'wall_rejected',
        saved_log_g=log_g if branch == 'involution' else None, production=False, accepted=False)
    q, u, inside = audit.full_log_q(log_g, bank['proposed_pose'], 2.)
    details = audit.region_from_tokens(tokens if valid else None, reference_tokens(), valid)
    patches = None if not valid else dict(tokens=sorted(tokens), neighbor_labels=details['neighbors'],
        fingerprint=fingerprint(tokens), secondary_tokens=sorted(t for t in tokens if t[:2] == [77, 217]),
        source_intersection=details['source217_intersection'], source_union=details['source217_union'],
        source_fraction=details['source217_inclusion'], jaccard=details['source217_jaccard'])
    return dict(kind='candidate', complete=True, input=bank,
        density=dict(log_g_status='finite', log_g=log_g, log_u=u, uniform_contains=inside, log_q=q,
                     saved_log_g_delta=0. if branch == 'involution' else None),
        actual=dict(wall_valid=valid, core_valid=True if valid else None, physical_valid=valid, verdict_matches=True),
        patches=patches, region=audit.expected_region(details),
        physical_zero=not valid,
        envelope=dict(lower_volume=2., uncertain_volume=3., upper_volume=5.,
                      fixed_labels=details['neighbors']) if valid else None)


class CandidateBankAuditTests(unittest.TestCase):
    def test_full_normalized_density_haar_reciprocal_and_both_priors(self):
        reference = audit.MapDensityReference(model())
        anchor = pose([2., -1., .7], Rotation.from_rotvec([.5, .2, -.3]).as_matrix())
        relative = [pose([.2, .3, -.5], Rotation.from_rotvec([.4, -.7, .1]).as_matrix()),
                    pose([-1.1, .2, .8], Rotation.from_rotvec([-.2, .1, .9]).as_matrix())]
        world = [compose(anchor, r) for r in relative]
        self.assertEqual(reference.branches, [dict(component_index=0, inverted=False),
            dict(component_index=0, inverted=True), dict(component_index=1, inverted=False)])
        for prior in [reference.base_log_prior, np.log([.8, .15, .05])]:
            actual, components, _ = reference.evaluate(world, anchor, prior)
            expected = np.asarray([[independent_component(reference, p, b) for b in range(3)] for p in relative])
            np.testing.assert_allclose(components, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual, logsumexp(expected+prior, axis=1), rtol=1e-12, atol=1e-12)
            # Missing reciprocal inversion or normalized-Haar factor must be visible.
            self.assertGreater(abs(expected[0, 0]-expected[0, 1]), .01)
            with self.assertRaises(ValueError):
                audit.close_log(float(actual[0])-math.log(8*math.pi**2), float(actual[0]), 'wrong Haar')

    def test_common_rigid_frame_preserves_full_density(self):
        reference = audit.MapDensityReference(model())
        anchor = pose([.3, .4, -.2], Rotation.from_rotvec([.1, .2, -.3]).as_matrix())
        p = pose([.7, -.2, .9], Rotation.from_rotvec([-.2, .4, .5]).as_matrix())
        shift = pose([4., -3., 1.], Rotation.from_rotvec([.7, -.6, .3]).as_matrix())
        a = reference.evaluate([p], anchor)[0]
        b = reference.evaluate([compose(shift, p)], compose(shift, anchor))[0]
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)

    def test_full_Q_includes_uniform_and_gaussian_at_every_candidate(self):
        p = pose([0., 0., 0.]);g = math.log(.7)
        q, u, inside = audit.full_log_q(g, p, 2.)
        self.assertTrue(inside)
        self.assertAlmostEqual(math.exp(q), .5/64+.5*.7)
        self.assertAlmostEqual(math.exp(u), 1/64)
        for x in [-2., 2.]:
            self.assertTrue(audit.full_log_q(g, pose([x, 0., 0.]), 2.)[2])
        q, u, inside = audit.full_log_q(g, pose([2.01, 0., 0.]), 2.)
        self.assertFalse(inside);self.assertEqual(u, -math.inf);self.assertAlmostEqual(q, g-math.log(2))
        self.assertAlmostEqual(audit.full_log_q(-math.inf, p, 2.)[0], -math.log(128))
        self.assertEqual(audit.finite_log(None, 'negative_infinity'), -math.inf)
        with self.assertRaises(ValueError):audit.finite_log(None, 'finite')

    def test_region_partition_all_quarter_boundaries_and_extra_tokens(self):
        reference = reference_tokens()
        anchor = [[16, 77, 'anchor', 'moving']]
        for count, expected_bin in [(1, 0), (3, 0), (4, 1), (7, 1), (8, 2), (11, 2), (12, 3), (15, 3), (16, 4)]:
            result = audit.region_from_tokens(anchor+reference[:count], reference)
            self.assertEqual(result['region'], 'A');self.assertEqual(result['source217_bin'], expected_bin)
            self.assertEqual(result['source217_inclusion'], count/16)
        # A may touch217 through a token entirely outside source reference.
        extra = [77, 217, 'other-moving', 'other-fixed']
        zero = audit.region_from_tokens(anchor+[extra], reference)
        self.assertEqual((zero['region'], zero['source217_bin']), ('A', 0))
        full = audit.region_from_tokens(anchor+reference+[extra], reference)
        self.assertEqual(full['source217_bin'], 4);self.assertEqual(full['source217_jaccard'], 16/17)
        self.assertEqual(audit.region_from_tokens(anchor+[[56, 77, 'fixed', 'moving']], reference)['region'], 'B')
        self.assertEqual(audit.region_from_tokens([], reference)['region'], 'unbound')
        self.assertEqual(audit.region_from_tokens(anchor, reference)['region'], 'other_contact')
        self.assertEqual(audit.region_from_tokens(None, reference, False)['region'], 'hard_invalid')
        with self.assertRaises(ValueError):audit.region_from_tokens([], reference, False)
        with self.assertRaises(ValueError):audit.region_from_tokens(anchor+anchor, reference)

    def test_complete_inventory_statuses_and_no_rejected_filter(self):
        identity = dict(start='fixed', stream=0, arm='context')
        row = dict(ordinal=0, cycle=1, slot=4, attempt_index=4, event_index=17, identity=identity,
            proposed_pose=pose([0., 0., 0.]), branch='uniform', wall_valid=False, core_valid=None,
            status='wall_rejected', saved_log_g=None, production=False, accepted=False)
        self.assertFalse(audit.validate_bank_row(row, 0, identity))
        for field, bad in [('ordinal', 1), ('attempt_index', 3), ('event_index', 18), ('identity', {}),
                           ('production', True), ('core_valid', False), ('saved_log_g', 1.), ('accepted', True)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.validate_bank_row(dict(row, **{field: bad}), 0, identity)
        for status, core, accepted in [('core_rejected', False, False), ('bath_rejected', True, False), ('accepted', True, True)]:
            test = dict(row, wall_valid=True, core_valid=core, accepted=accepted, status=status,
                        branch='involution', saved_log_g=-2.)
            self.assertEqual(audit.validate_bank_row(test, 0, identity), core)
        self.assertEqual(len(audit.PANEL_ORDINALS), 32)
        self.assertEqual((audit.PANEL_ORDINALS[0], audit.PANEL_ORDINALS[-1]), (0, 2232))

    def test_synthetic_strict_geometry_global_patch_ids_and_wall_zero(self):
        shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.5)])
        context = dict(bodies=[dict(label=16, pose=pose([-1.5, 0., 0.])),
                              dict(label=217, pose=pose([1.5, 0., 0.]))])
        reference = audit.GeometryReference(shape, ['bodypatch'], context, 10., .3)
        free = reference.classify(pose([0., 0., 0.]))
        self.assertTrue(free['wall_valid'] and free['core_valid'])
        self.assertEqual(free['patch_tokens'], [(16, 77, 'bodypatch', 'bodypatch'),
                                               (77, 217, 'bodypatch', 'bodypatch')])
        # Exact core tangency is permitted; strict penetration is not.
        self.assertTrue(reference.classify(pose([-.5, 0., 0.]))['core_valid'])
        overlap = reference.classify(pose([-.5001, 0., 0.]))
        self.assertFalse(overlap['core_valid']);self.assertIsNone(overlap['patch_tokens'])
        wall = reference.classify(pose([9.5001, 0., 0.]))
        self.assertFalse(wall['wall_valid']);self.assertIsNone(wall['core_valid'])

    def test_exclusion_tangency_noncommuting_frame_and_asymmetric_patch_sides(self):
        shape = dict(atoms=[dict(center=[-.5, 0., 0.], radius=.2),
                            dict(center=[.5, 0., 0.], radius=.2)])
        q = Rotation.from_rotvec([.3, -.2, .5]).as_matrix()
        rigid = pose([2., -1., .7], q)
        center = compose(rigid, pose([0., 0., 0.]))
        fixed = compose(rigid, pose([1.5, 0., 0.], Rotation.from_rotvec([math.pi/2, 0., 0.]).as_matrix()))
        context = dict(bodies=[dict(label=217, pose=fixed)])
        reference = audit.GeometryReference(shape, ['minus', 'plus'], context, 20., .3)
        result = reference.classify(center)
        self.assertTrue(result['wall_valid'] and result['core_valid'])
        self.assertEqual(result['patch_tokens'], [(77, 217, 'plus', 'minus')])
        # The one-sphere exact exclusion tangency must remain non-contact.
        sphere = dict(atoms=[dict(center=[0., 0., 0.], radius=.5)])
        ref = audit.GeometryReference(sphere, ['only'],
            dict(bodies=[dict(label=16, pose=pose([1.5, 0., 0.]))]), 10., .25)
        self.assertEqual(ref.classify(pose([0., 0., 0.]))['patch_tokens'], [])

    def test_all_scalar_rows_require_full_Q_and_complete_regions(self):
        row = preflight_row(0, sorted([[16, 77, 'a', 'b']]+reference_tokens()), branch='involution')
        audit.audit_scalar_row(row, row['input'], row['input']['identity'], 2., reference_tokens(), 3)
        for key, changed in [('density', dict(row['density'], log_q=row['density']['log_g'])),
                             ('region', 'B'), ('envelope', dict(lower_volume=2., uncertain_volume=3., upper_volume=6.))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.audit_scalar_row(dict(row, **{key: changed}), row['input'], row['input']['identity'], 2., reference_tokens(), 3)
        invalid = preflight_row(1, [], False)
        audit.audit_scalar_row(invalid, invalid['input'], invalid['input']['identity'], 2., reference_tokens(), 3)
        with self.assertRaises(ValueError):
            audit.audit_scalar_row(dict(invalid, region='unbound'), invalid['input'], invalid['input']['identity'], 2., reference_tokens(), 3)

    def test_population_volume_denominators_invalid_zero_and_cloud_work(self):
        rows = [preflight_row(0, [], False), preflight_row(1, []),
                preflight_row(2, sorted([[16, 77, 'a', 'b']]+reference_tokens()))]
        result = audit.coverage_summary(rows)
        self.assertEqual(result['attempted'], 3);self.assertEqual(result['hard_valid'], 2)
        self.assertEqual(sum(r['attempted_count'] for r in result['partitions']), 3)
        groups = {r['region']: r for r in result['partitions']}
        self.assertEqual(groups['hard_invalid']['hard_only_importance_mass'], 0.)
        expected = math.exp(-rows[1]['density']['log_q'])/3
        self.assertAlmostEqual(groups['unbound']['hard_only_importance_mass'], expected)
        self.assertAlmostEqual(groups['A_patch_complete']['hard_only_importance_mass'], expected)
        self.assertEqual(result['full_source217_inclusion_hits'], 1)
        self.assertAlmostEqual(result['expected_complete_allocation_raw_points'], 4.48*6)
        self.assertAlmostEqual(result['total_envelope_upper_volume'], 10.)

    def test_complete_preflight_event_history_and_all_source_reference(self):
        tokens = sorted([[16, 77, 'a', 'b']]+reference_tokens())
        rows = [preflight_row(0, [], False), preflight_row(1, tokens)]
        events = [dict(kind='source_begun'), dict(kind='source_complete', tokens=tokens, neighbor_labels=[16, 217])]
        for i, row in enumerate(rows):
            events.extend([dict(kind='candidate_begun', ordinal=i, input=row['input']),
                dict(kind='density_complete', ordinal=i, density=row['density']),
                dict(kind='geometry_complete', ordinal=i, wall_valid=row['actual']['wall_valid'], core_valid=row['actual']['core_valid'])])
            if row['actual']['physical_valid']:
                events.extend([dict(kind='patches_complete', ordinal=i, patches=row['patches'], region=row['region']),
                               dict(kind='envelope_complete', ordinal=i, envelope=row['envelope'])])
            events.append(dict(kind='candidate_complete', ordinal=i))
        for i, event in enumerate(events):event['event_index'] = i
        encoded = lambda data: [json.dumps(r) for r in data]
        self.assertEqual(audit.audit_preflight_events(encoded(events), rows, reference_tokens()), 12)
        for bad in [events[:-1], events+[events[-1]], events[:4]+events[5:]]:
            with self.assertRaises(ValueError):audit.audit_preflight_events(encoded(bad), rows, reference_tokens())


if __name__ == '__main__':
    unittest.main()
