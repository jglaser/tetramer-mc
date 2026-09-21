"""Allocation changes must not alter the frozen target or proposal."""
import copy
import unittest

from prepare_far_atlas_repeat import PREFIXES, SEEDS, UNCHANGED, validate_repeat, validate_command_options


class FarAtlasRepeatTests(unittest.TestCase):
    def test_unsupported_kernel_only_options_rejected(self):
        help_text = '--root ROOT --config CONFIG --samples SAMPLES --workers WORKERS'
        validate_command_options(['python', 'runner.py', '--root', 'new', '--samples', '65536'], help_text)
        with self.assertRaises(ValueError): validate_command_options(['runner.py', '--lambda-ratio', '64'], help_text)

    def setUp(self):
        self.original = {k: {k: 'frozen'} for k in UNCHANGED}
        self.original['arms'] = [dict(name=name, geometric_latent_SD_A=width, components=180,
            populations=8, samples_per_population=32768, seeds=[base+1009*i for i in range(8)], output='old-'+name)
            for name, width, base in [('narrow', .2, 107101010), ('broad', .4, 107201010)]]
        self.repeated = copy.deepcopy(self.original); self.repeated['prefix_counts'] = PREFIXES.copy()
        for arm in self.repeated['arms']:
            arm.update(populations=16, samples_per_population=65536, seeds=[SEEDS[arm['name']]+1009*i for i in range(16)], output='new-'+arm['name'])
        self.hashes = dict(narrow='old-narrow-bytes', broad='old-broad-bytes')
        self.old_seeds = {s for a in self.original['arms'] for s in a['seeds']}

    def check(self, repeated=None, hashes=None, prior=None):
        validate_repeat(self.original, repeated or self.repeated, self.hashes, hashes or self.hashes, self.old_seeds if prior is None else prior)

    def test_allocation_only_repeat_passes(self): self.check()

    def test_changed_target_prior_fit_or_region_rejected(self):
        for key in ('physical', 'q_window', 'hybrid', 'center_selection', 'local_fit', 'analysis', 'executable_sha256'):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.repeated); changed[key] = 'altered'
                with self.assertRaises(ValueError): self.check(changed)

    def test_model_byte_change_rejected(self):
        with self.assertRaises(ValueError): self.check(hashes=dict(self.hashes, broad='refitted'))

    def test_known_stream_reuse_rejected(self):
        prior = self.old_seeds | {self.repeated['arms'][0]['seeds'][0]}
        with self.assertRaises(ValueError): self.check(prior=prior)
        changed = copy.deepcopy(self.repeated); changed['arms'][0]['seeds'][0] = next(iter(self.old_seeds))
        with self.assertRaises(ValueError): self.check(changed)

    def test_population_width_and_prefix_changes_rejected(self):
        for key, value in [('samples_per_population', 32768), ('populations', 8), ('geometric_latent_SD_A', .3), ('components', 179)]:
            changed = copy.deepcopy(self.repeated); changed['arms'][0][key] = value
            with self.assertRaises(ValueError): self.check(changed)
        changed = copy.deepcopy(self.repeated); changed['prefix_counts'] = [8192, 32768, 65536]
        with self.assertRaises(ValueError): self.check(changed)


if __name__ == '__main__': unittest.main()
