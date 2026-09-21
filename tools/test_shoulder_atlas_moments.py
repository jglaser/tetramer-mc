"""Independent dense-array checks for full/inner/outer shoulder covariance."""
import itertools
import math
import unittest

import numpy as np

from shoulder_atlas_moments import KEYS, ShoulderAtlasMoments, inner_key
from test_shoulder_union_moments import oracle_masks


def add_rows(acc, qs, radii, physical, hard):
    for q, r, w, h in zip(qs, radii, physical, hard):
        if w:
            # Unequal clouds preserve the original linear mean, and exercise
            # the variance diagnostic while leaving the density untouched.
            acc.add(q, r, math.log(w), math.log(h), (math.log(.5*w), math.log(1.5*w)))
        else:
            acc.add(q, r)


class FullShoulderMomentsTests(unittest.TestCase):
    def test_full_covariance_against_dense_oracle(self):
        radii = np.array(list(itertools.product((.1, .4, .8), repeat=3))*3)
        qs = np.array([1.05]*27+[1.1]*27+[1.8]*27)
        physical = np.array([1.+i*7%17 for i in range(len(qs))])
        hard = np.array([1.+i*11%13 for i in range(len(qs))])
        physical[::7] = hard[::7] = 0
        acc = ShoulderAtlasMoments(); add_rows(acc, qs, radii, physical, hard)
        result = acc.report()
        masks = dict(full=np.ones(len(qs), dtype=bool), outer=qs >= 1.1)
        masks.update({inner_key(k): v & (qs < 1.1) for k,v in oracle_masks(radii).items()})
        self.assertEqual(tuple(masks), KEYS)
        for kind, weights in [('physical', physical), ('hard', hard)]:
            dense = np.column_stack([masks[k]*weights for k in KEYS])
            expected = np.cov(dense, rowvar=False, ddof=1)/len(qs)
            got = np.array([[v['sign']*math.exp(v['log_absolute_covariance']) if v['sign'] else 0
                             for v in row.values()] for row in result['same_row_covariances'][kind].values()])
            np.testing.assert_allclose(got, expected, rtol=5e-12, atol=2e-13)
            for j,k in enumerate(KEYS):
                row=result[kind][k]
                self.assertEqual(row['draws'], len(qs))
                self.assertAlmostEqual(math.exp(row['logQ']), dense[:,j].mean())
            self.assertTrue(all(v['complete_row_partition'] for v in result['partition_checks'][kind].values()))
        full = KEYS.index('full'); parts = [KEYS.index(k) for k in ('inner_union','inner_outside_union','outer')]
        self.assertAlmostEqual(got[np.ix_(parts,parts)].sum(), got[full,full])

    def test_strict_boundaries_and_full_n(self):
        qs = [1., math.nextafter(1., math.inf), math.nextafter(1.1, -math.inf), 1.1, math.nextafter(2., -math.inf), 2.]
        acc = ShoulderAtlasMoments()
        for q in qs:
            if 1 < q < 2: acc.add(q, (0.,0.,0.), 0., 0., (0.,0.))
            else: acc.add(q)
        out = acc.report()
        self.assertEqual(out['physical']['full']['nonzero'],4)
        self.assertEqual(out['physical']['inner']['nonzero'],2)
        self.assertEqual(out['physical']['outer']['nonzero'],2)
        self.assertTrue(all(row['draws']==6 for row in out['physical'].values()))
        self.assertAlmostEqual(math.exp(out['physical']['full']['logQ']), 4/6)
        for q in (1.,2.,math.nan,math.inf):
            with self.assertRaises(ValueError): ShoulderAtlasMoments().add(q,(0.,0.,0.),0.,0.,(0.,0.))

    def test_merge_and_zero_support(self):
        qs=[1.05,1.1,1.5,1.02,1.9,1.01]; radii=[(.2,.4,.8)]*6
        w=[0.,2.,4.,3.,0.,5.];h=[0.,1.,2.,1.,0.,2.]
        whole=ShoulderAtlasMoments();add_rows(whole,qs,radii,w,h)
        left=ShoulderAtlasMoments();right=ShoulderAtlasMoments()
        add_rows(left,qs[:3],radii[:3],w[:3],h[:3]);add_rows(right,qs[3:],radii[3:],w[3:],h[3:])
        merged=left.merge(right).report(); expected=whole.report()
        for kind in ('physical','hard'):
            for key in KEYS:
                self.assertEqual(merged[kind][key]['draws'],6)
                a,b=merged[kind][key]['logQ'],expected[kind][key]['logQ']
                if a is None:self.assertIsNone(b)
                else:self.assertAlmostEqual(a,b)
        zero=ShoulderAtlasMoments();zero.add();zero.add()
        result=zero.report()
        self.assertTrue(all(row['logQ'] is None for row in result['physical'].values()))
        self.assertTrue(all(row['sign']==0 for rows in result['same_row_covariances']['physical'].values() for row in rows.values()))

    def test_cloud_pair_validation_and_noise_retained(self):
        acc=ShoulderAtlasMoments()
        with self.assertRaises(ValueError):acc.add(1.2,None,0.,0.,(0.,1.))
        with self.assertRaises(ValueError):acc.add(cloud_pair=(0.,0.))
        with self.assertRaises(ValueError):acc.add(1.05,None,0.,0.,(0.,0.))
        acc=ShoulderAtlasMoments()
        add_rows(acc,[1.05,1.2],[(.1,.1,.1)]*2,[2.,4.],[1.,1.])
        result=acc.report()
        self.assertGreater(result['physical']['full']['paired_cloud_variance_fraction'],0.)


if __name__ == '__main__':unittest.main()
