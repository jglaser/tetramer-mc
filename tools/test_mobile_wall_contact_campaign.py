"""Full-wall target and mixed reciprocal-law regressions; no physical draws."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from mobile_wall_contact_campaign import (
    CAPTURE_RADIUS, MODEL, REFERENCE, WALL, command, model_counts, physical_config, read,
)


class Campaign(unittest.TestCase):
    def test_scaffold_and_bath_unchanged(self):
        source = read(REFERENCE/'config.json'); before = copy.deepcopy(source)
        result = physical_config(source, Path('/tmp/archived-shape.json'))
        self.assertEqual(source, before)
        result['shape'] = source['shape']; result['capture_radius'] = source['capture_radius']
        self.assertEqual(result, source)

    def test_support_contains_every_possible_wall_valid_center(self):
        source = read(REFERENCE/'config.json'); atoms = read(source['shape'])['atoms']
        bound = max(math.dist(atom['center'], [0, 0, 0])+atom['radius'] for atom in atoms)
        required = WALL['radius']+bound
        self.assertGreater(CAPTURE_RADIUS, required+256*math.ulp(1.)*(1+required))
        self.assertLess(source['capture_radius'], required)

    def test_rejects_unexpected_target_or_wall(self):
        source = read(REFERENCE/'config.json')
        for key, value in [('target_region', {}), ('reservoir_density', .04)]:
            changed = copy.deepcopy(source); changed[key] = value
            with self.assertRaises(ValueError): physical_config(changed, '/tmp/shape.json')
        source['metadata']['physical_sphere_radius_A'] += 1
        with self.assertRaises(ValueError): physical_config(source, '/tmp/shape.json')

    def test_partial_reciprocal_counts_are_dynamic(self):
        original = model_counts(MODEL)
        self.assertEqual((original['base_component_count'], original['virtual_component_count']), (150, 300))
        model = read(MODEL)
        base = model['base_model']
        for key in ('anchors', 'means', 'covariances', 'weights'):
            base[key].append(copy.deepcopy(base[key][0]))
        model['reciprocal_components'].append(False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'model.json'; path.write_text(json.dumps(model))
            counts = model_counts(path)
        self.assertEqual((counts['base_component_count'], counts['virtual_component_count']), (151, 301))
        self.assertFalse(counts['reciprocal_components'][-1])

    def test_cli_selects_full_wall_and_all_anchors(self):
        argv = command(Path('/tmp/campaign'), dict(directory='/tmp/pop', samples=16384, seed=17))
        self.assertEqual(float(argv[argv.index('--wall-radius')+1]), WALL['radius'])
        self.assertEqual(argv[argv.index('--wall-center')+1:], ['0', '0', '0'])
        self.assertNotIn('--proposal-anchor-index', argv)
        self.assertNotIn('--region', argv)


if __name__ == '__main__': unittest.main()
