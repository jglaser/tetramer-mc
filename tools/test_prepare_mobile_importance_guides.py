"""Check that implementation-schema conversion does not silently refit the guide."""
import copy
import unittest
from prepare_mobile_importance_guides import strict_guide


class GuideConversionTests(unittest.TestCase):
    def test_preserves_all_density_parameters_and_source(self):
        source=dict(schema='defensive-latent-shell-guide-design-v1',implemented=False,
                    region_sha256='fixed',defensive_uniform_shell_probability=.5,
                    gaussian_components=[dict(weight=.3,mean=[1.]*6,covariance=[[2.]*6]*6,
                                              source_population='r00',source_draw=10),
                                         dict(weight=.7,mean=[-1.]*6,covariance=[[3.]*6]*6,
                                              source_population='r01',source_draw=22)])
        old=copy.deepcopy(source);result=strict_guide(source)
        self.assertEqual(source,old)
        self.assertEqual(result['schema'],'defensive-latent-shell-guide-v1')
        self.assertEqual(result['region_sha256'],source['region_sha256'])
        self.assertEqual(result['defensive_uniform_shell_probability'],.5)
        for new,prior in zip(result['gaussian_components'],source['gaussian_components']):
            self.assertEqual(set(new),{'weight','mean','covariance'})
            self.assertTrue(all(new[k]==prior[k]for k in new))

    def test_refuses_unexpected_source_schema(self):
        with self.assertRaisesRegex(ValueError,'frozen guide design'):
            strict_guide(dict(schema='some-new-fit',implemented=False))


if __name__=='__main__':unittest.main()
