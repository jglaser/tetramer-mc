import copy
import unittest
from prepare_native_class_line_probe import channels, jobs, select_probes


class PreparationTest(unittest.TestCase):
    def test_fixed_allocation_channels_and_independent_seeds(self):
        allocation = jobs()
        self.assertEqual(len(allocation), 8)
        self.assertEqual(sum(j['samples'] for j in allocation), 1024)
        self.assertEqual(len({j['seed'] for j in allocation}), 8)
        self.assertEqual(allocation, jobs())
        self.assertEqual(channels('hard_free'), [{'class': 'hard_free', 'probability': 1.}])
        self.assertAlmostEqual(sum(x['probability'] for x in channels('class')), 1.)
        self.assertEqual(channels('class')[-1], {'class': 'native', 'probability': .2, 'orthant': 55})
        with self.assertRaises(ValueError): channels('winning-arm')

    def test_selection_uses_preserved_order_and_includes_invalid(self):
        selected = [dict(id=str(i), group=g, latent=[i, 0, 0, 0, 0, 0])
                    for i, g in enumerate(['competing22']*2+['competing62']*2+
                                          ['old-native55']*2+['remaining-native55']*2)]
        saved = dict(rows=[dict(id=str(i), group='breadth', latent=[i+10, 0, 0, 0, 0, 0],
                               metadata=dict(coverage_class=g))
                          for i, g in enumerate(['native_R5']*32+['native_complement']*32+
                                                ['competing']*32+['invalid']*32)])
        result = select_probes(selected, saved)
        self.assertEqual(len(result), 40)
        self.assertEqual([v['id'] for v in result[-8:]], ['breadth:'+str(i) for i in range(96, 104)])
        self.assertEqual(result[:8], [dict(id='critical:'+v['id'], latent=v['latent'], group=v['group'], source=v) for v in selected])
        bad = copy.deepcopy(selected); bad[1]['latent'] = bad[0]['latent']
        with self.assertRaises(ValueError): select_probes(bad, saved)
        with self.assertRaises(ValueError): select_probes(selected[:-1], saved)


if __name__ == '__main__': unittest.main()
