#!/usr/bin/env python3
"""Synthetic preparation-contract tests; not physical protein evidence."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import prepare_protected_guide_validation as prep


def guide_fixture():
    components = [dict(weight=1/80, mean=[i*.005, 0., 0., 0., 0., 0.],
                       covariance=(np.eye(6)*.25).tolist()) for i in range(80)]
    bank = dict(schema='defensive-latent-shell-guide-v1', region_sha256='0'*64,
                defensive_uniform_shell_probability=.5, gaussian_components=components)
    source = copy.deepcopy(bank)
    for c in source['gaussian_components']:
        c['weight'] *= .5
    source['gaussian_components'] += [dict(weight=.125, mean=[2.+i, 0., 0., 0., 0., 0.],
        covariance=np.eye(6).tolist()) for i in range(4)]
    fit = dict(complete=True, optimizer=dict(success=True), missing_training_mandatory_groups=[],
               candidate_weights=dict(protected_minimax=[1/84]*84))
    return bank, source, fit


def refreeze(root):
    old = root/'freeze.json'
    if old.exists():
        old.unlink()
    prep.write(old, dict(files=prep.file_hashes(root)))


class GuideTests(unittest.TestCase):
    def test_exact_weights_geometry_and_input_preservation(self):
        bank, source, fit = guide_fixture(); original = copy.deepcopy(source)
        guide = prep.build_protected(bank, source, fit)
        self.assertEqual([c['weight'] for c in guide['gaussian_components']], fit['candidate_weights']['protected_minimax'])
        self.assertEqual(source, original)
        self.assertEqual(guide['defensive_uniform_shell_probability'], .5)
        for a, b in zip(guide['gaussian_components'], source['gaussian_components']):
            self.assertEqual(a['mean'], b['mean']); self.assertEqual(a['covariance'], b['covariance'])

    def test_invalid_weight_vectors_and_failed_fit_are_rejected(self):
        bank, source, fit = guide_fixture()
        for bad in ([1/84]*83, [float('nan')]+[1/84]*83, [0.]+[1/84]*83, [.02]*84):
            value = copy.deepcopy(fit); value['candidate_weights']['protected_minimax'] = bad
            with self.assertRaises(ValueError):
                prep.build_protected(bank, source, value)
        for key, value in [('complete', False), ('missing_training_mandatory_groups', ['noentry:62'])]:
            bad = copy.deepcopy(fit); bad[key] = value
            with self.assertRaises(ValueError):
                prep.build_protected(bank, source, bad)
        fit['optimizer']['success'] = False
        with self.assertRaises(ValueError):
            prep.build_protected(bank, source, fit)

    def test_region_geometry_and_defensive_probability_cannot_change(self):
        bank, source, fit = guide_fixture()
        for mutation in ('mean', 'covariance', 'region', 'alpha'):
            changed = copy.deepcopy(source)
            if mutation == 'mean': changed['gaussian_components'][0]['mean'][0] += .1
            if mutation == 'covariance': changed['gaussian_components'][0]['covariance'][0][0] += .1
            if mutation == 'region': changed['region_sha256'] = '1'*64
            if mutation == 'alpha': changed['defensive_uniform_shell_probability'] = .2
            with self.assertRaises(ValueError): prep.build_protected(bank, changed, fit)

    def test_untruncated_proposals_preserve_every_exterior_zero(self):
        _, source, _ = guide_fixture()
        for component in source['gaussian_components']:
            component['mean'][0] += 10
        first = prep.draw_proposal(np.random.default_rng(17), source, 1000)
        second = prep.draw_proposal(np.random.default_rng(17), source, 1000)
        for a, b in zip(first, second): self.assertTrue(np.array_equal(a, b))
        u, branch, component = first; support = np.sum(u*u, axis=1) <= 16
        lv = 3*math.log(math.pi)-math.lgamma(4)+6*math.log(4)
        q = prep.LatentImportanceGuide(source, '0'*64).log_density(u, support, lv)
        self.assertLess(np.max(abs(q-prep.log_proposal(u, source))), 1e-10)
        arrays = dict(u=u, branch=branch, component=component, log_q=q, support=support,
                      volume_weight=np.where(support, np.exp(-lv-q), 0.), draw=np.arange(1000))
        report = prep.reference_population(arrays, 1000, 17, lv)
        self.assertGreater(report['exterior_zeros'], 300)
        self.assertEqual(report['samples'], 1000)
        self.assertEqual(report['mean'], arrays['volume_weight'].sum()/1000)
        for field in ('draw', 'support', 'volume_weight'):
            bad = {k:v.copy() for k, v in arrays.items()}
            if field == 'draw': bad[field][0] = 1
            elif field == 'support': bad[field][0] = ~bad[field][0]
            else: bad[field][np.flatnonzero(~support)[0]] = 1.
            with self.assertRaises(ValueError): prep.reference_population(bad, 1000, 17, lv)

    def test_hoeffding_controls_ten_events_with_unconditional_n(self):
        expected = math.sqrt(2*math.log(20/1e-6)/65536)
        self.assertEqual(prep.hoeffding_halfwidth(65536), expected)
        self.assertEqual(prep.hoeffding_halfwidth(16384), 2*expected)
        with self.assertRaises(ValueError): prep.hoeffding_halfwidth(0)

    def test_seed_inventory_catches_reference_and_reserved_collisions(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); campaign = root/'runs/prior'; campaign.mkdir(parents=True)
            path = campaign/'protocol.json'
            for seed in (prep.REFERENCE_SEEDS[0], prep.RESERVED_PHYSICAL_SEEDS[0]):
                if path.exists(): path.unlink()
                prep.write(path, dict(jobs=[dict(seed=seed)]))
                with self.assertRaises(ValueError): prep.seed_inventory(root)
            path.unlink(); prep.write(path, dict(jobs=[dict(seed=12)], nested=dict(seeds=[13, 14])))
            self.assertEqual(prep.seed_inventory(root)['seeds'], [12, 13, 14])

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError): prep.inside(Path(t), '../escape')


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name); runs = self.repo/'runs'
        self.package = runs/'refined-contact-bank-preparation-20260922'
        self.fit = runs/'contact-guide-stratum-reweighting-20260923'
        self.pilot = runs/'smc-geometry-guide-pilot-20260923'
        self.geometry = runs/'smc-geometry-guide-preparation-20260923'
        for p in (self.package, self.fit, self.geometry, self.pilot/'smc/provenance', self.pilot/'common'):
            p.mkdir(parents=True)
        prep.write(self.package/'region.json', dict(minimum_mahalanobis_radius=0., mahalanobis_radius=4.))
        bank, source, fit = guide_fixture()
        bank['region_sha256'] = source['region_sha256'] = prep.sha(self.package/'region.json')
        prep.write(self.package/'guide-bank.json', bank); prep.write(self.pilot/'smc/provenance/importance-guide.json', source)
        for name in ('shape.json', 'config.json', 'old-r5-region.json'):
            prep.write(self.package/name, dict(synthetic_fixture=True))
        for path in (self.package/'plan.json', self.geometry/'plan.json', self.fit/'plan.json', self.pilot/'protocol.json'):
            prep.write(path, dict(populations=[dict(seed=234)]))
        prep.write(self.fit/'analysis.json', fit); refreeze(self.fit)
        sphere = dict(schema='conditional-ray-sphere-crosslanguage-v1', complete=True, binary_sha256='a'*64,
            source_bundle_sha256='b'*64, results=[dict(activity=z, physical_returncode=0, audit_returncode=0,
                checks=[dict(passed=True)]) for z in (0., 2.)])
        prep.write(self.pilot/'common/sphere-reference.json', sphere)
        pins = dict(fit_analysis=prep.sha(self.fit/'analysis.json'), fit_freeze=prep.sha(self.fit/'freeze.json'),
            bank=prep.sha(self.package/'guide-bank.json'), source_guide=prep.sha(self.pilot/'smc/provenance/importance-guide.json'),
            sphere=prep.sha(self.pilot/'common/sphere-reference.json'), binary='a'*64, bundle='b'*64)
        pins.update({key:prep.sha(self.package/name) for key, name in
            [('region','region.json'),('shape','shape.json'),('config','config.json'),('reference_region','old-r5-region.json')]})
        for context in (patch.object(prep, 'PINS', pins), patch.object(prep, 'N', 64)):
            context.start(); self.addCleanup(context.stop)
        self.out = runs/'prepared'

    def test_preparation_roundtrip_is_portable_and_does_not_mutate_sources(self):
        before = prep.file_hashes(self.fit); result = prep.prepare(self.out, self.repo)
        self.assertTrue(result['complete']); self.assertEqual(before, prep.file_hashes(self.fit))
        self.assertEqual(result['physical_draws_launched'], 0)
        self.assertEqual(result['guide_sha256']['bank'], prep.PINS['bank'])
        self.assertEqual(prep.validate_preparation(self.out), result)
        renamed = self.repo/'moved-preparation'; self.out.rename(renamed)
        self.assertEqual(prep.validate_preparation(renamed), result)

    def test_no_overwrite_and_changed_pinned_input_fail_closed(self):
        self.out.mkdir(); (self.out/'user-file').write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'Fresh preparation'): prep.prepare(self.out, self.repo)
        self.assertEqual((self.out/'user-file').read_text(), 'preserve')
        (self.out/'user-file').unlink(); self.out.rmdir()
        (self.fit/'analysis.json').write_text('{}')
        with self.assertRaises(ValueError): prep.prepare(self.out, self.repo)
        self.assertFalse(self.out.exists())

    def test_refrozen_changed_weights_still_fail_exact_fit_binding(self):
        prep.prepare(self.out, self.repo); path = self.out/'guide-protected.json'; value = prep.read(path)
        value['gaussian_components'][0]['weight'] += .001
        value['gaussian_components'][1]['weight'] -= .001
        path.unlink(); prep.write(path, value); refreeze(self.out)
        with self.assertRaisesRegex(ValueError, 'exact fixed fitted weights'): prep.validate_preparation(self.out)

    def test_reference_archive_and_source_mutations_are_rejected(self):
        prep.prepare(self.out, self.repo)
        path = self.out/'proposal-reference/bank-r00.npz'; path.write_bytes(b'changed')
        with self.assertRaises(ValueError): prep.validate_preparation(self.out)

    def test_failed_population_not_hidden_by_passing_arm(self):
        prep.prepare(self.out, self.repo); ref = self.out/'proposal-reference'; path = ref/'validation.json'
        value = prep.read(path); value['arms']['protected']['populations'][0]['passed'] = False
        path.unlink(); prep.write(path, value); refreeze(ref); refreeze(self.out)
        with self.assertRaises(ValueError): prep.validate_preparation(self.out)


if __name__ == '__main__': unittest.main()
