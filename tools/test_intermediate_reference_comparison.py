"""Independent linear-moment controls for original-density intermediate bands."""
import math
import unittest

import numpy as np

from compare_intermediate_reference import BandMoments,band_index,finish_group,population_rse


class IntermediateReferenceControls(unittest.TestCase):
    def test_half_open_band_boundaries_partition_exact_original_interval(self):
        for q,label in [(1.999,None),(2.,'0'),(2.5,'1'),(3.,'2'),(4.,'3'),(5.,None)]:
            self.assertEqual(band_index(q),label)
        for i,boundary in enumerate((2.5,3.,4.)):
            self.assertEqual(band_index(math.nextafter(boundary,0.)),str(i))
            self.assertEqual(band_index(math.nextafter(boundary,math.inf)),str(i+1))

    def test_full_n_row_moments_and_paired_cloud_noise_match_linear_formula(self):
        rows=[(2.,True,2.,6.,2.),(2.25,True,1.,3.,.5),(2.5,True,4.,4.,1.),
            (3.5,True,0.5,1.5,.25),(4.5,True,3.,9.,3.),(5.,True,10.,12.,4.),
            (2.2,False,0.,0.,0.),(None,False,0.,0.,0.)]
        value=BandMoments()
        for q,valid,a,b,h in rows:
            if valid:value.add(q,True,math.log((a+b)/2),math.log(h),[math.log(a),math.log(b)])
            else:value.add()
        report=value.report();n=len(rows)
        for key in ('all','0','1','2','3'):
            def selected(row):return row[1] and band_index(row[0]) is not None and (key=='all' or band_index(row[0])==key)
            pair=np.asarray([[a,b] if selected(row) else [0.,0.] for row in rows for q,valid,a,b,h in [row]])
            linear=pair.mean(axis=1);hard=np.asarray([row[4] if selected(row) else 0. for row in rows])
            physical=report['physical'][key];h=report['hard'][key]
            self.assertEqual(physical['draws'],n);self.assertEqual(h['draws'],n)
            self.assertAlmostEqual(math.exp(physical['logQ']),float(linear.mean()))
            self.assertAlmostEqual(math.exp(h['logQ']),float(hard.mean()))
            self.assertAlmostEqual(physical['row_RSE'],float(linear.std(ddof=1)/math.sqrt(n)/linear.mean()))
            self.assertAlmostEqual(physical['maximum_fraction'],float(linear.max()/linear.sum()))
            noise=float(np.mean((pair[:,0]-pair[:,1])**2)/4/linear.var(ddof=1))
            if noise:self.assertAlmostEqual(physical['paired_cloud_variance_fraction'],noise)
            else:self.assertIsNone(physical['paired_cloud_variance_fraction'])
        self.assertEqual(report['q_equals_2']['nonzero'],1)
        self.assertAlmostEqual(math.exp(report['q_equals_2']['logQ']),4/n)
        self.assertAlmostEqual(math.exp(report['physical']['all']['logQ']),sum(math.exp(report['physical'][str(i)]['logQ']) for i in range(4)))

    def test_empty_bands_remain_unresolved(self):
        values=BandMoments()
        for _ in range(8):values.add()
        for kind in ('physical','hard'):
            for result in values.report()[kind].values():
                self.assertEqual(result['draws'],8);self.assertEqual(result['nonzero'],0)
                self.assertIsNone(result['logQ']);self.assertIsNone(result['row_RSE'])
                self.assertIsNone(result['maximum_fraction'])
                self.assertIn('not a mass upper bound',result['coverage'])

    def test_population_errors_use_equal_fixed_population_means(self):
        local=[];pooled=BandMoments();means=[]
        for i,weight in enumerate((2.,4.,8.)):
            m=BandMoments();m.add(2.2,True,math.log(weight),0.,[math.log(weight),math.log(weight)])
            for _ in range(3):m.add()
            report=m.report();local.append(dict(id=str(i),seed=i,samples=4,**report));pooled.merge(m);means.append(weight/4)
        result=finish_group('fixture',pooled,local,1.,{},'shape')
        expected=np.std(means,ddof=1)/math.sqrt(3)/np.mean(means)
        self.assertAlmostEqual(result['physical']['all']['independent_population_RSE'],expected)
        self.assertAlmostEqual(math.exp(result['physical']['all']['logQ']),np.mean(means))
        self.assertIsNone(population_rse([None,None]))
        bad=[dict(p) for p in local];bad[0]['samples']=5
        with self.assertRaises(ValueError):finish_group('fixture',pooled,bad,1.,{},'shape')


if __name__=='__main__':unittest.main()
