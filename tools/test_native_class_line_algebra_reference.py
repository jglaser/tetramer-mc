"""Fixed synthetic proposal-algebra tests; no protein data or physical draws."""
import copy
import json
import math
import time
import unittest
from unittest import mock

import numpy as np

import native_class_line_algebra_reference as algebra
import native_class_line_reference as line
import test_native_class_line_reference as fixtures


def compact(recon, u):
    """Serialize a toy full-reference result into the Rust compact contract."""
    result = recon.density(u)
    if result.get('conditioning_disabled'): return {'conditioning_disabled': True}, result
    axes = []
    for original in result['axes']:
        axis = copy.deepcopy(original); axis.pop('components'); axis.pop('native_geometry', None)
        raw = recon.raw(u); raw[axis['axis']] = 0.
        u0 = line.scalar_chart_solve(recon.L0, raw-recon.m0)
        du = line.scalar_chart_solve(recon.L0, np.eye(6)[axis['axis']])
        axis.update(latent_line_origin=u0.tolist(), latent_line_direction=du.tolist())
        for channel in axis['channels']:
            channel['orthant_intervals'] = line.orthant_intervals(u0, du, channel['orthant'], axis['segment']) if channel.get('orthant') is not None and 'segment' in axis else None
        axes.append(axis)
    trace = dict(trace_format='class-line-compact-v1', class_scope=algebra.CLASS_SCOPE,
        raw_coordinates=recon.raw(u).tolist(), baseline_log_density=result['baseline_log_density'], axes=axes,
        component_mixture_multipliers=((np.asarray(result['component_multipliers'])-1+recon.beta)/recon.beta).tolist())
    return trace, result


def case(*, axes=(0, 1, 2), beta=.4, components=None):
    args = fixtures.ConditionalLawTests().setup(axes=axes)
    args[1]['conditional_probability'] = beta
    if components is not None:
        original = args[1]['gaussian_components']
        args[1]['gaussian_components'] = []
        for k in range(components):
            c = copy.deepcopy(original[k % len(original)])
            c['weight'] = 1.+k/100.
            c['mean'][k % 6] += (k % 7)/30.
            args[1]['gaussian_components'].append(c)
        args[1]['class_channels'] = [dict(c) for c in args[1]['class_channels']]+[
            {'class': 'contact_without_native', 'probability': .15, 'orthant': 22},
            {'class': 'native', 'probability': .15, 'orthant': 55}]
        for c in args[1]['class_channels'][:3]: c['probability'] *= .7
    return args, line.Reconstructor(*args), algebra.AlgebraLaw(*args[:3])


def rebuild_saved_channels(law, u, trace):
    """Independent scalar branch reference for deliberately fabricated unions."""
    factors = np.zeros(len(law.weights)); x = law.raw(u)
    for axis in trace['axes']:
        h, n, e = [axis[key] for key in ('hard_free_intervals', 'native_intervals', 'exclusion_contact_intervals')]
        base = {'hard_free': h, 'native': line.intersection(h, n),
                'contact_without_native': line.difference(line.intersection(h, e), n)}
        means, sigmas = law.conditional(x, axis['axis']); channels = []
        for i, channel in enumerate(law.channels):
            support = None
            if channel.get('orthant') is not None and 'segment' in axis:
                support = line.orthant_intervals(axis['latent_line_origin'], axis['latent_line_direction'], channel['orthant'], axis['segment'])
            intervals = base[channel['class']] if support is None else line.intersection(base[channel['class']], support)
            channels.append(dict(channel, channel=i, intervals=intervals, orthant_intervals=support))
            for k, (mean, sigma) in enumerate(zip(means, sigmas)):
                branch = line.conditional_branch(h, intervals, mean, sigma, law.floor, x[axis['axis']])
                factors[k] += channel['probability']*branch['multiplier']/len(law.axes)
        axis['channels'] = channels
    trace['component_mixture_multipliers'] = factors.tolist()
    return trace


def selected_draw(law, u, trace, evaluated, channel=0, component=0):
    axis = evaluated['axes'][0]; a = axis['axis']; branch = axis['channels'][channel]
    old = law.raw(u); old[a] += .03
    draw = dict(conditional=True, original_latent=law.inverse_raw(old).tolist(), axis=a,
        component=component, channel=channel, **{'class': copy.deepcopy(law.channels[channel])},
        uniform_channel_selection=sum(c['probability'] for c in law.channels[:channel])+law.channels[channel]['probability']/2,
        conditional_mean=float(axis['conditional_means'][component]), conditional_sigma=float(axis['conditional_sigmas'][component]),
        class_mass=float(branch['class_masses'][component]), hard_free_mass=float(axis['hard_free_masses'][component]),
        effective_mass=float(branch['effective_masses'][component]),
        fallback_target=algebra.FALLBACKS[branch['fallback_indices'][component]],
        fallback=bool(branch['fallback_indices'][component]), geometry=copy.deepcopy(trace['axes'][0]))
    if draw['fallback_target'] == 'unconditional':
        draw['original_latent'] = np.asarray(u).tolist(); return draw
    intervals = branch['intervals'] if draw['fallback_target'] == 'class' else axis['hard_free_intervals']
    mean, sigma = draw['conditional_mean'], draw['conditional_sigma']; x = law.raw(u)[a]
    k = next(i for i, v in enumerate(intervals) if line.contains([v], x))
    mass = float(line.interval_masses([intervals[k]], mean, sigma))
    before = float(line.interval_masses(intervals[:k], mean, sigma))
    within = float(line.hard.normal.normal_masses((intervals[k]['lower']-mean)/sigma, (x-mean)/sigma))/mass
    draw.update(selected_interval=copy.deepcopy(intervals[k]), selected_interval_mass=mass,
        uniform_interval_selection=(before+.5*mass)/draw['effective_mass'],
        uniform_within_interval=within, returned_raw_coordinate=float(x))
    return draw


class AlgebraTests(unittest.TestCase):
    def test_correlated_chart_raw_pose_inverse_jacobian_and_full_law(self):
        _, full, law = case()
        for u in (np.zeros(6), np.array([.1, -.2, .3, .4, -.5, .6]), np.array([5., 0., 0., 0., 0., 0.])):
            trace, expected = compact(full, u); actual = law.evaluate_saved(u, trace)
            np.testing.assert_allclose(law.raw(u), full.raw(u), atol=0, rtol=0)
            np.testing.assert_allclose(law.inverse_raw(law.raw(u)), u, atol=2e-15)
            for a, b in zip(law.decode(u), full.decode(u)): np.testing.assert_allclose(a, b, atol=2e-15)
            np.testing.assert_allclose(law.inverse_pose(actual['position'], actual['rotation']), u, atol=2e-14)
            self.assertAlmostEqual(actual['log_density'], expected['log_density'], places=12)
            np.testing.assert_allclose(actual['component_multipliers'], expected['component_multipliers'], atol=1e-12)
            self.assertFalse(actual['geometry_certified'])
            for axis, old in zip(actual['axes'], expected['axes']):
                for k, component in enumerate(old['components']):
                    for c, branch in enumerate(component['channels']):
                        self.assertEqual(algebra.FALLBACKS[axis['channels'][c]['fallback_indices'][k]], branch['fallback'])
                        self.assertAlmostEqual(axis['channels'][c]['effective_masses'][k], branch['effective_mass'], places=14)

    def test_all_92_components_three_axes_five_channels_and_mass_cache(self):
        _, full, law = case(components=92); u = np.zeros(6); trace, expected = compact(full, u)
        with mock.patch.object(line, 'interval_masses', wraps=line.interval_masses) as masses:
            actual = law.evaluate_saved(u, trace)
        self.assertEqual(actual['component_branches'], 92*3*5)
        self.assertEqual(masses.call_count, sum(a['distinct_mass_unions'] for a in actual['axes']))
        self.assertTrue(all(np.shape(call.args[1]) == (92,) for call in masses.call_args_list))
        self.assertAlmostEqual(actual['log_density'], expected['log_density'], places=11)
        np.testing.assert_allclose(actual['component_multipliers'], expected['component_multipliers'], rtol=1e-13)
        changed = copy.deepcopy(trace); changed['component_mixture_multipliers'][-1] += .1
        with self.assertRaisesRegex(ValueError, 'mixture factors'): law.evaluate_saved(u, changed)

    def test_no_observer_tree_or_leaf_geometry_is_called(self):
        args, full, _ = case(); u = np.zeros(6); trace, _ = compact(full, u)
        forbidden = AssertionError('Geometry was called by algebra-only code')
        with (mock.patch.object(line.NativeContactRegions, '__init__', side_effect=forbidden),
              mock.patch.object(line, 'leaf_contact_intervals', side_effect=forbidden),
              mock.patch.object(line.hard, 'hard_free_intervals', side_effect=forbidden),
              mock.patch.object(line, 'cKDTree', side_effect=forbidden),
              mock.patch.object(line.hard, 'cKDTree', side_effect=forbidden),
              mock.patch.object(line.NativeLineReference, 'all_anchors', side_effect=forbidden)):
            law = algebra.AlgebraLaw(*args[:3]); evaluated = law.evaluate_saved(u, trace)
            draw = selected_draw(law, u, trace, evaluated)
            law.verify_draw(draw, u, evaluated)
            self.assertFalse(any(hasattr(law, k) for k in ('atoms', 'radii', 'native', 'fixed_trees', 'fixed_world')))

    def test_poisoned_channels_orthants_factors_frames_and_contract_fail(self):
        _, full, law = case(); u = np.zeros(6); trace, _ = compact(full, u)
        changes = [lambda t: t['axes'][0]['channels'][0].update(intervals=[]),
            lambda t: t['axes'][0]['channels'][2].update(orthant_intervals=[]),
            lambda t: t['axes'][0]['origin']['position'].__setitem__(0, 1.),
            lambda t: t['axes'][0]['direction'].__setitem__(0, 2.),
            lambda t: t['axes'][0]['latent_line_direction'].__setitem__(0, 2.),
            lambda t: t['axes'][0]['segment'].__setitem__(0, -50.),
            lambda t: t['axes'].pop(), lambda t: t.update(class_scope='selected native motif'),
            lambda t: t['component_mixture_multipliers'].__setitem__(0, .25)]
        for i, change in enumerate(changes):
            altered = copy.deepcopy(trace); change(altered)
            with self.subTest(change=i), self.assertRaises(ValueError): law.evaluate_saved(u, altered)

    def test_invalid_base_unions_are_rejected_without_assuming_native_subset_hard(self):
        _, full, law = case(axes=(0,)); u = np.zeros(6); trace, _ = compact(full, u)
        for values in ([line.interval(1., 0.)], [line.interval(-1., 0.), line.interval(0., 1.)],
                       [line.interval(-100., 100.)], [dict(lower=0., upper=1., lower_closed=1, upper_closed=True)]):
            bad = copy.deepcopy(trace); bad['axes'][0]['native_intervals'] = values
            with self.subTest(values=values), self.assertRaises(ValueError): law.evaluate_saved(u, bad)
        # N and E are whole geometric unions; only the derived channels lie in H.
        altered = copy.deepcopy(trace); a = altered['axes'][0]
        a['native_intervals'] = [line.interval(-1., 1.)]; a['exclusion_contact_intervals'] = [line.interval(-2., 2.)]
        rebuild_saved_channels(law, u, altered)
        law.evaluate_saved(u, altered)

    def test_self_consistent_wrong_base_geometry_can_pass_only_algebra(self):
        _, full, law = case(axes=(0,)); u = np.zeros(6); trace, expected = compact(full, u)
        wrong = copy.deepcopy(trace); a = wrong['axes'][0]
        a['hard_free_intervals'] = [line.interval(*a['segment'])]
        a['native_intervals'], a['exclusion_contact_intervals'] = [], []
        rebuild_saved_channels(law, u, wrong)
        checked = law.evaluate_saved(u, wrong)
        self.assertFalse(checked['geometry_certified'])
        self.assertIn('conditional on saved', checked['audit_scope'])
        with self.assertRaises(ValueError): line.compare_geometry(a, expected['axes'][0], full, u)

    def test_disabled_uniform_no_components_beta_zero_and_exterior(self):
        for alpha, beta in ((1., .4), (.5, 0.)):
            args, _, _ = case(beta=beta); args[1]['defensive_uniform_shell_probability'] = alpha
            if alpha == 1.: args[1]['gaussian_components'] = []
            full = line.Reconstructor(*args); law = algebra.AlgebraLaw(*args[:3])
            for u in (np.zeros(6), np.array([5., 0., 0., 0., 0., 0.])):
                trace, expected = compact(full, u); actual = law.evaluate_saved(u, trace)
                self.assertEqual(actual['log_density'], expected['log_density'])
                self.assertEqual(actual['axes'], [])
                with self.assertRaises(ValueError): law.evaluate_saved(u, dict(trace, axes=[]))

    def test_empty_chords_singleton_and_zero_positive_orthants(self):
        args, _, _ = case(axes=(0,)); args[0]['gaussian_chart']['covariances'] = [np.eye(6).tolist()]
        args[1]['class_channels'] = [{'class': 'hard_free', 'probability': .5, 'orthant': 63},
                                      {'class': 'hard_free', 'probability': .5, 'orthant': 62}]
        for capture_center, capture_radius, u, reason in [([0., 0., 0.], 100., [0., 5., 0., 0., 0., 0.], 'no_R4_chord'),
                ([0., 2., 0.], .1, [0.]*6, 'no_capture_chord'),
                ([20., 0., 0.], .1, [0.]*6, 'disjoint_chords'),
                ([0., 1., 0.], 1., [0.]*6, None)]:
            cfg = copy.deepcopy(args[2]); cfg.update(capture_center=capture_center, capture_radius=capture_radius)
            full = line.Reconstructor(args[0], args[1], cfg, args[3], args[4]); law = algebra.AlgebraLaw(args[0], args[1], cfg)
            u = np.asarray(u); trace, _ = compact(full, u); actual = law.evaluate_saved(u, trace)
            self.assertEqual(actual['axes'][0].get('empty_reason'), reason)
            if reason is None:
                self.assertEqual(actual['axes'][0]['segment'], [0., 0.])
                self.assertEqual(actual['axes'][0]['channels'][0]['orthant_intervals'], [line.interval(0., 0.)])
                self.assertEqual(actual['axes'][0]['channels'][1]['orthant_intervals'], [])
                self.assertTrue(np.all(actual['axes'][0]['hard_free_masses'] == 0))
            bad = copy.deepcopy(trace); bad['axes'][0]['empty_reason'] = 'no_R4_chord' if reason is None else 'disjoint_chords'
            if bad['axes'][0]['empty_reason'] != reason:
                with self.assertRaises(ValueError): law.evaluate_saved(u, bad)

    def test_class_hard_free_unconditional_floor_and_selected_cdf(self):
        args, full, law = case(axes=(0,)); u = np.zeros(6); trace, _ = compact(full, u)
        trace['axes'][0].update(hard_free_intervals=[line.interval(-1., 1.)], native_intervals=[line.interval(-.1, .1)],
                               exclusion_contact_intervals=[line.interval(-2., 2.)])
        for floor, target in ((1e-12, 'class'), (.2, 'hard_free'), (.99, 'unconditional')):
            spec = copy.deepcopy(args[1]); spec['minimum_conditional_mass'] = floor
            law = algebra.AlgebraLaw(args[0], spec, args[2]); t = rebuild_saved_channels(law, u, copy.deepcopy(trace))
            actual = law.evaluate_saved(u, t); draw = selected_draw(law, u, t, actual, channel=1)
            self.assertEqual(draw['fallback_target'], target)
            self.assertLess(law.verify_draw(draw, u, actual)['inverse_error'], 1e-14)
            wrong = copy.deepcopy(draw); wrong['fallback_target'] = 'bad'
            with self.assertRaisesRegex(ValueError, 'fallback'): law.verify_draw(wrong, u, actual)
            if target != 'unconditional':
                wrong = copy.deepcopy(draw); wrong['uniform_within_interval'] = .9
                with self.assertRaisesRegex(ValueError, 'CDF'): law.verify_draw(wrong, u, actual)
        means, sigmas = law.conditional(law.raw(u), 0)
        exact = float(line.interval_masses([line.interval(-.1, .1)], means, sigmas)[0])
        spec = copy.deepcopy(args[1]); spec['minimum_conditional_mass'] = exact
        law = algebra.AlgebraLaw(args[0], spec, args[2]); t = rebuild_saved_channels(law, u, copy.deepcopy(trace))
        self.assertEqual(law.evaluate_saved(u, t)['axes'][0]['channels'][1]['fallback_indices'][0], 1)

    def test_channel_boundary_belongs_to_next_category_and_retained_coordinates(self):
        _, full, law = case(axes=(0,)); u = np.zeros(6); trace, _ = compact(full, u)
        actual = law.evaluate_saved(u, trace); draw = selected_draw(law, u, trace, actual)
        draw['uniform_channel_selection'] = law.channels[0]['probability']
        with self.assertRaisesRegex(ValueError, 'categorical'): law.verify_draw(draw, u, actual)
        draw = selected_draw(law, u, trace, actual)
        raw = law.raw(draw['original_latent']); raw[1] += .1; draw['original_latent'] = law.inverse_raw(raw).tolist()
        with self.assertRaisesRegex(ValueError, 'retained'): law.verify_draw(draw, u, actual)

    def test_interval_cdf_roundoff_ambiguity_is_reported_and_tiny_identity_is_retained(self):
        _, full, law = case(axes=(0,)); raw = np.zeros(6); raw[0] = -.3
        u = law.inverse_raw(raw); trace, _ = compact(full, u)
        trace['axes'][0].update(hard_free_intervals=[line.interval(-1., 1.)],
            native_intervals=[line.interval(-.5, -.1), line.interval(.1, .5)],
            exclusion_contact_intervals=[])
        rebuild_saved_channels(law, u, trace)
        actual = law.evaluate_saved(u, trace); draw = selected_draw(law, u, trace, actual, channel=1)
        first_mass = draw['selected_interval_mass']
        # This recorded first-bin choice differs from the independently rounded
        # CDF boundary by 1e-15 mass, within the existing 2e-14 audit tolerance.
        draw['uniform_interval_selection'] = (first_mass+1e-15)/draw['effective_mass']
        result = law.verify_draw(draw, u, actual)
        self.assertTrue(result['interval_selection_ambiguous'])
        self.assertEqual(result['interval_selection_cdf_tolerance'], 2e-14)
        draw['uniform_interval_selection'] = (first_mass+1e-10)/draw['effective_mass']
        with self.assertRaisesRegex(ValueError, 'interval categorical'): law.verify_draw(draw, u, actual)
        # Both tiny interval endpoints lie within the generic 1e-9 matching
        # tolerance. Exact identity in the saved draw geometry resolves them.
        raw[0] = 3e-12; u = law.inverse_raw(raw); trace, _ = compact(full, u)
        trace['axes'][0].update(hard_free_intervals=[line.interval(-1., 1.)],
            native_intervals=[line.interval(-4e-12, -2e-12), line.interval(2e-12, 4e-12)],
            exclusion_contact_intervals=[])
        rebuild_saved_channels(law, u, trace)
        actual = law.evaluate_saved(u, trace); draw = selected_draw(law, u, trace, actual, channel=1)
        self.assertEqual(draw['fallback_target'], 'class')
        self.assertLess(law.verify_draw(draw, u, actual)['inverse_error'], 1e-12)

    def test_fixed_synthetic_timing_only(self):
        # Fixed before timing: one 92-component/3-axis/5-channel toy, u=0, 16
        # evaluations. The full toy reference is evaluated once for its trace.
        iterations = 16
        _, full, law = case(components=92); u = np.zeros(6); trace, _ = compact(full, u)
        started = time.process_time()
        for _ in range(iterations): law.evaluate_saved(u, trace)
        elapsed = time.process_time()-started
        print('SYNTHETIC_ALGEBRA_TIMING '+json.dumps(dict(iterations=iterations, components=92,
            axes=3, channels=5, cpu_seconds=elapsed, seconds_per_evaluation=elapsed/iterations,
            protein_geometry_queries=0, scope='Algebra only on one fixed toy trace; not a protein runtime estimate.')), flush=True)


if __name__ == '__main__': unittest.main()
