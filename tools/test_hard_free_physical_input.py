"""Synthetic-only tests for the v6 physical-input bridge."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

import hard_free_physical_input as backend
from test_hard_free_line_physical_reference import fixture as old_fixture, write


def fixture(root):
    values = old_fixture(root)
    manifest, summary = values[-2:]
    manifest.update(resume_supported=False, attempt_journal=backend.ATTEMPT_JOURNAL)
    summary['manifest'] = copy.deepcopy(manifest)
    write(root/'manifest.json', manifest); write(root/'summary.json', summary)
    return values


def binder(bindings):
    def bind(path, expected=None):
        path = Path(path).resolve(); digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and digest != expected: raise ValueError('Changed bound input '+str(path))
        bindings[str(path)] = digest
        return path
    return bind


def compact(reference, u):
    """Tiny-sphere full geometry creates an independent test input only."""
    full = reference.density(u)
    details = dict(raw_coordinates=reference.raw(u).tolist(), baseline_log_density=full['baseline_log_density'],
        component_branches=full['component_branches'], fallback_component_branches=full['fallback_component_branches'], axes=[])
    for axis in full['axes']:
        guide = copy.deepcopy(reference.guide); guide['raw_translation_axes'] = [axis['axis']]
        one = backend.line.Reconstructor(reference.region, guide, reference.config, reference.shape)
        saved = {k: copy.deepcopy(v) for k, v in axis.items() if not k.startswith('conditional_') and k != 'component_fallbacks'}
        saved['hard_free_intervals'] = saved.pop('intervals')
        q = one.density(u)['log_density']; saved['axis_log_proposal_density'] = q if math.isfinite(q) else None
        details['axes'].append(saved)
    return full, details


def active_fixture(root, beta=1.):
    region, guide, config, shape, rows, manifest, summary = fixture(root)
    guide['conditional_probability'] = beta
    reference = backend.line.Reconstructor(region, guide, config, shape)
    row = copy.deepcopy(rows[0]); full, details = compact(reference, row['latent'])
    row['hard_free_line_density'] = details; row['log_proposal_density'] = full['log_density']
    row['log_hard_weight'] = row['log_physical_jacobian']-full['log_density']
    row['log_importance_weight'] = float(logsumexp([row['log_hard_weight']+c['log_weight'] for c in row['clouds']])-math.log(2))
    axis = full['axes'][0]; mean, sigma, mass = [axis[k][0] for k in ('conditional_means', 'conditional_sigmas', 'conditional_masses')]
    selected = axis['intervals'][0]
    imass = float(backend.line.interval_masses([selected], mean, sigma))
    within = float(backend.normal.normal_masses((selected['lower']-mean)/sigma, -mean/sigma))/imass
    row['hard_free_line_draw'] = dict(conditional=True, original_latent=row['latent'], axis=axis['axis'], component=0,
        conditional_mean=mean, conditional_sigma=sigma, conditional_mass=mass, fallback=False,
        geometry=copy.deepcopy(details['axes'][0]), selected_interval=copy.deepcopy(selected), selected_interval_mass=imass,
        uniform_interval_selection=.5, uniform_within_interval=within, returned_raw_coordinate=0., inverse_probability_error=0.)
    return region, guide, config, shape, rows, manifest, row, reference, full


class HardFreePhysicalInputTests(unittest.TestCase):
    def test_all_attempt_accounting_and_unchanged_v6_identity(self):
        with tempfile.TemporaryDirectory() as d:
            region, _, _, _, rows, manifest, _ = fixture(Path(d))
            before = json.dumps([manifest, rows], sort_keys=True)
            got = [backend.validate_row(row, expected_draw=i, manifest=manifest, region=region) for i, row in enumerate(rows)]
            self.assertEqual([v['counters']['attempted'] for v in got], [1, 1, 1])
            self.assertEqual([v['counters']['contributing'] for v in got], [1, 0, 0])
            self.assertEqual([v['z'] for v in got[1:]], [-math.inf, -math.inf])
            self.assertEqual(got[1]['pairs'], [-math.inf, -math.inf]); self.assertEqual(got[2]['counters']['shell_rejected'], 1)
            self.assertEqual(len({key for v in got for key in v['role_keys']}), 9)
            self.assertEqual([v['branch'] for v in got], [1, 1, 1]); self.assertEqual(sum(v['raw_points'] for v in got), 6)
            self.assertEqual(before, json.dumps([manifest, rows], sort_keys=True))

    def test_wrong_schema_class_contracts_and_typed_lineage_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            region, _, _, _, rows, manifest, _ = fixture(Path(d))
            for field, value in [('schema', 'importance-latent-region-normalizer-v7'), ('compiled_native', {}),
                                  ('random_stream_contract', {}), ('samples', True), ('seed', 2**64), ('cloud_replicates', 1)]:
                with self.subTest(field=field):
                    bad = dict(manifest, **{field: value})
                    with self.assertRaises(ValueError): backend.validate_manifest(bad, region)
            for field, value in [('proposal_branch', 'native-class-line'), ('native_class_line_density', {}),
                                  ('hard_valid', 1), ('draw', True), ('proposal_component', True), ('log_proposal_density', math.inf)]:
                with self.subTest(field=field):
                    bad = dict(rows[0], **{field: value})
                    with self.assertRaises(ValueError): backend.validate_row(bad, expected_draw=0, manifest=manifest, region=region)

    def test_two_cloud_average_and_invalid_zeros_are_strict(self):
        with tempfile.TemporaryDirectory() as d:
            region, _, _, _, rows, manifest, _ = fixture(Path(d))
            good = backend.validate_row(rows[0], expected_draw=0, manifest=manifest, region=region)
            self.assertAlmostEqual(good['z'], float(logsumexp(good['pairs'])-math.log(2)))
            bad = copy.deepcopy(rows[0]); bad['log_importance_weight'] = bad['log_hard_weight']+sum(c['log_weight'] for c in bad['clouds'])/2
            with self.assertRaisesRegex(ValueError, 'arithmetic mean'): backend.validate_row(bad, expected_draw=0, manifest=manifest, region=region)
            for i in (1, 2):
                bad = dict(rows[i], log_importance_weight=0.)
                with self.assertRaisesRegex(ValueError, 'zero weight'): backend.validate_row(bad, expected_draw=i, manifest=manifest, region=region)
            bad = copy.deepcopy(rows[0]); bad['clouds'][1]['retained_cells'] += 1
            with self.assertRaisesRegex(ValueError, 'envelope differs'): backend.validate_row(bad, expected_draw=0, manifest=manifest, region=region)

    def test_authenticated_provenance_and_rows_without_geometry_initialization(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, _, _, _, rows, _, _ = fixture(root); bindings = {}
            with (patch.object(backend.line.Reconstructor, '__init__', side_effect=AssertionError('hard geometry init')),
                 patch.object(backend.normal.Reconstructor, '__init__', side_effect=AssertionError('normal geometry init')),
                 patch.object(backend.numeric.AlgebraLaw, '__init__', side_effect=AssertionError('class-law init')),
                 patch.object(backend.line, 'cKDTree', side_effect=AssertionError('tree init')),
                 patch.object(backend.normal, 'cKDTree', side_effect=AssertionError('tree init'))):
                manifest, summary, region, config, law, witness = backend.provenance(root, binder(bindings))
                self.assertIsNone(witness); self.assertTrue(law.chart_factor_validation)
                got = [backend.check_row(row, i, manifest, region, config, law) for i, row in enumerate(rows)]
                self.assertEqual(sum(v[0]['counters']['attempted'] for v in got), 3)
                self.assertLess(max(max(v[1].values()) for v in got), 1e-12)
            self.assertEqual(len(bindings), 9)
            self.assertEqual(summary['manifest']['schema'], backend.SCHEMA)
            self.assertFalse(hasattr(law, 'atoms')); self.assertFalse(hasattr(law, 'channels'))

    def test_changed_bound_files_and_incomplete_metadata_fail(self):
        for file in ('shape.json', 'input-config.json', 'region.json', 'importance-guide.json', 'source-bundle.json', 'samples.jsonl', 'attempts.jsonl'):
            with self.subTest(file=file), tempfile.TemporaryDirectory() as d:
                root = Path(d); fixture(root)
                target = root/file if file.endswith('.jsonl') else root/'provenance'/file
                target.write_text('{}\n')
                with self.assertRaisesRegex(ValueError, 'Changed bound input'): backend.provenance(root, binder({}))
        for mode in ('resume', 'journal', 'failure', 'denominator'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root = Path(d); *_, manifest, summary = fixture(root)
                if mode == 'resume': manifest['resume_supported'] = True
                if mode == 'journal': manifest['attempt_journal'] = 'retry'
                if mode == 'failure': write(root/'failure.json', {})
                if mode == 'denominator': summary['samples'] = 2
                summary['manifest'] = manifest; write(root/'manifest.json', manifest); write(root/'summary.json', summary)
                with self.assertRaises(ValueError): backend.provenance(root, binder({}))

    def test_every_row_pose_inverse_jacobian_q_capture_and_full_density(self):
        with tempfile.TemporaryDirectory() as d:
            region, guide, config, _, rows, manifest, _ = fixture(Path(d)); law = backend.AlgebraLaw(region, guide, config)
            changes = [('q', 100.), ('log_proposal_density', -20.), ('physical_jacobian', 3.),
                ('backmapped_latent', [1.]*6), ('backmapped_radius', 1.), ('capture_valid', False),
                ('pose', dict(position=[8., 8., 8.], orientation=[1., 0., 0., 0.]))]
            for field, value in changes:
                with self.subTest(field=field):
                    # This invalid zero row still must pass all pose/scalar checks.
                    bad = copy.deepcopy(rows[1]); bad[field] = value
                    with self.assertRaises(ValueError): backend.check_row(bad, 1, manifest, region, config, law)

    def test_active_all_axis_mixture_and_successful_inverse_cdf(self):
        with tempfile.TemporaryDirectory() as d:
            region, guide, config, _, _, manifest, row, reference, expected = active_fixture(Path(d))
            law = backend.AlgebraLaw(region, guide, config)
            with patch.object(backend.line, 'hard_free_intervals', side_effect=AssertionError('geometry query')):
                got, errors, diagnostics = backend.check_row(row, 0, manifest, region, config, law)
            self.assertTrue(got['counters']['conditioned_draws']); self.assertEqual(diagnostics['component_branches'], 6)
            self.assertEqual(diagnostics['fallback_component_branches'], expected['fallback_component_branches'])
            self.assertEqual(diagnostics['hard_membership_axes'], 3); self.assertLess(errors['inverse_cdf'], 1e-12)
            evaluated = law.evaluate_saved(row['latent'], row['hard_free_line_density'])
            self.assertAlmostEqual(evaluated['log_density'], expected['log_density'], places=12)
            self.assertFalse(evaluated['geometry_certified'])
            bad = copy.deepcopy(row); bad['hard_free_line_draw']['uniform_within_interval'] = .8
            with self.assertRaisesRegex(ValueError, 'Inverse conditional CDF'): backend.check_row(bad, 0, manifest, region, config, law)
            bad = copy.deepcopy(row); bad['hard_free_line_draw']['selected_interval']['lower_closed'] = False
            with self.assertRaisesRegex(ValueError, 'endpoint inclusion'): backend.check_row(bad, 0, manifest, region, config, law)

    def test_component_floor_fallback_preserves_original_draw(self):
        with tempfile.TemporaryDirectory() as d:
            region, guide, config, _, _, manifest, row, _, expected = active_fixture(Path(d))
            law = backend.AlgebraLaw(region, guide, config); axis = expected['axes'][0]
            row['proposal_component'] = 1
            row['hard_free_line_draw'] = dict(conditional=True, original_latent=row['latent'], axis=0, component=1,
                conditional_mean=axis['conditional_means'][1], conditional_sigma=axis['conditional_sigmas'][1],
                conditional_mass=axis['conditional_masses'][1], fallback=True, geometry=copy.deepcopy(row['hard_free_line_density']['axes'][0]))
            accounting, _, _ = backend.check_row(row, 0, manifest, region, config, law)
            self.assertEqual(accounting['counters']['fallback_draws'], 1)
            bad = copy.deepcopy(row); bad['hard_free_line_draw']['original_latent'][0] += .01
            with self.assertRaises(ValueError): backend.check_row(bad, 0, manifest, region, config, law)
            bad = copy.deepcopy(row); bad['hard_free_line_draw'] = dict(conditional=False, original_latent=bad['latent'])
            with self.assertRaisesRegex(ValueError, 'beta=1'): backend.check_row(bad, 0, manifest, region, config, law)

    def test_saved_frames_counts_and_topology_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            region, guide, config, _, _, _, row, _, _ = active_fixture(Path(d)); law = backend.AlgebraLaw(region, guide, config)
            def altered_segment(value): value['axes'][0]['segment'][0] -= .1
            def absent_origin(value): value['axes'][0].pop('origin')
            def changed_direction(value): value['axes'][0]['direction'] = [0., 1., 0.]
            def empty_reason(value): value['axes'][0]['empty_reason'] = 'no_R4_chord'
            def double_union(value): value['axes'][0]['hard_free_intervals'] *= 2
            def missing_axis(value): value['axes'].pop()
            def wrong_branches(value): value['component_branches'] = True
            for mutate in (altered_segment, absent_origin, changed_direction, empty_reason, double_union, missing_axis, wrong_branches):
                with self.subTest(mutate=mutate.__name__):
                    bad = copy.deepcopy(row['hard_free_line_density']); mutate(bad)
                    with self.assertRaises((ValueError, KeyError)): law.evaluate_saved(row['latent'], bad)

    def test_empty_chords_and_uniform_only_disable_conditioning(self):
        with tempfile.TemporaryDirectory() as d:
            region, guide, config, _, _, _, _ = fixture(Path(d)); guide['conditional_probability'] = 1.
            law = backend.AlgebraLaw(region, guide, config)
            self.assertIsNone(law.chord(np.array([0., 4., 0.]), np.array([1., 0., 0.]), 4.))
            u = np.array([0., 0., 8., 0., 0., 0.]); frame = law.line_frame(u, 0)
            self.assertEqual(frame['empty_reason'], 'no_R4_chord')
            saved = dict(frame, hard_free_intervals=[]); law._saved_axis(u, saved)
            saved['hard_free_intervals'] = [backend.line.interval(-1., 1.)]
            with self.assertRaises(ValueError): law._saved_axis(u, saved)
            guide['defensive_uniform_shell_probability'] = 1.; guide['gaussian_components'] = []
            law = backend.AlgebraLaw(region, guide, config)
            result = law.evaluate_saved(np.zeros(6), dict(conditioning_disabled=True))
            self.assertAlmostEqual(result['log_density'], -law.logvolume); self.assertEqual(result['axes'], [])

    def test_zero_measure_saved_union_does_not_infer_physical_invalidity(self):
        with tempfile.TemporaryDirectory() as d:
            region, guide, config, _, _, manifest, row, _, expected = active_fixture(Path(d))
            law = backend.AlgebraLaw(region, guide, config)
            details = row['hard_free_line_density']; baseline = details['baseline_log_density']
            for axis in details['axes']:
                axis.update(hard_free_intervals=[], empty_reason='no_positive_hard_free_length', axis_log_proposal_density=baseline)
            details['fallback_component_branches'] = details['component_branches']
            row['log_proposal_density'] = baseline
            row['log_hard_weight'] = row['log_physical_jacobian']-baseline
            row['log_importance_weight'] = float(logsumexp([row['log_hard_weight']+c['log_weight'] for c in row['clouds']])-math.log(2))
            trace = row['hard_free_line_draw']
            row['hard_free_line_draw'] = dict(conditional=True, original_latent=row['latent'], axis=0, component=0,
                conditional_mean=trace['conditional_mean'], conditional_sigma=trace['conditional_sigma'],
                conditional_mass=0., fallback=True, geometry=copy.deepcopy(details['axes'][0]))
            accounting, _, diagnostics = backend.check_row(row, 0, manifest, region, config, law)
            self.assertEqual(diagnostics['hard_membership_axes'], 0)
            self.assertEqual(diagnostics['indeterminate_zero_measure_axes'], 3)
            self.assertEqual(accounting['counters']['contributing'], 1)
            # Algebra has preserved the saved label, not independently
            # certified that any isolated point is geometrically feasible.
            self.assertTrue(row['hard_valid'])


if __name__ == '__main__': unittest.main()
