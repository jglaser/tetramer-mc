"""Repeat allocation, unchanged-law and fresh-seed guards without sampling."""
import copy
import unittest

from prepare_native_tail_repeat import LAW_FIELDS, POPULATIONS, PREFIX_POPULATIONS, SAMPLES, SEED, WORKERS, validate_repeat


class NativeTailRepeatTests(unittest.TestCase):
    def fixture(self):
        base = {name:name for name in LAW_FIELDS}
        base.update(geometry_probe_seed=113601010, proposed_campaign=dict(samples_per_population=32768,
            populations=4, workers=4, seed_base=113501010, seeds=[113501010+1009*i for i in range(4)],
            lambda_ratio=64., cloud_replicates=2))
        repeat = copy.deepcopy(base)
        repeat.update(geometry_probe_count=0, geometry_probe_seed=None)
        repeat['analysis_prefixes'] = [dict(populations=n, samples=n*SAMPLES,
            population_ids=[f'r{i:02d}' for i in range(n)]) for n in PREFIX_POPULATIONS]
        repeat['proposed_campaign'].update(samples_per_population=SAMPLES, populations=POPULATIONS,
            workers=WORKERS, seed_base=SEED, seeds=[SEED+1009*i for i in range(POPULATIONS)])
        return base, repeat

    def test_budget_and_fresh_seed_schedule(self):
        base, repeat = self.fixture(); validate_repeat(base, repeat)
        self.assertEqual(SAMPLES*POPULATIONS, 4194304)
        for key, bad in (('samples_per_population', 32768), ('populations', 4), ('workers', 4),
                         ('seed_base', 113501010), ('seeds', base['proposed_campaign']['seeds'])):
            changed = copy.deepcopy(repeat); changed['proposed_campaign'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError): validate_repeat(base, changed)

    def test_law_masks_and_clouds_cannot_change(self):
        base, repeat = self.fixture()
        for key in LAW_FIELDS:
            changed = copy.deepcopy(repeat); changed[key] = None
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'law or reporting'):
                validate_repeat(base, changed)
        for key, bad in (('lambda_ratio', 32.), ('cloud_replicates', 1)):
            changed = copy.deepcopy(repeat); changed['proposed_campaign'][key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'cloud law'):
                validate_repeat(base, changed)

    def test_existing_geometry_evidence_is_reused_without_new_draws(self):
        base, repeat = self.fixture()
        for key, bad in (('geometry_probe_count', 2048), ('geometry_probe_seed', 113601010)):
            changed = copy.deepcopy(repeat); changed[key] = bad
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'new geometry probes'):
                validate_repeat(base, changed)

    def test_pilot_and_probe_seeds_must_be_disjoint(self):
        base, repeat = self.fixture()
        for key in ('pilot', 'probe'):
            changed = copy.deepcopy(base)
            if key == 'pilot': changed['proposed_campaign']['seeds'][0] = SEED
            else: changed['geometry_probe_seed'] = SEED
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Reused seed'):
                validate_repeat(changed, repeat)

    def test_prefixes_are_frozen_in_population_order(self):
        base, repeat = self.fixture()
        self.assertEqual([p['samples'] for p in repeat['analysis_prefixes']], [1048576, 2097152, 4194304])
        repeat['analysis_prefixes'][0]['population_ids'][0] = 'r15'
        with self.assertRaisesRegex(ValueError, 'prefixes'): validate_repeat(base, repeat)


if __name__ == '__main__': unittest.main()
