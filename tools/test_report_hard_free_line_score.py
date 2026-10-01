import copy
import math
import unittest
from report_hard_free_line_score import derive,distribution,lse


class DerivedAxisDensities(unittest.TestCase):
    def fixture(self,inside=True,allowed=True,masses=(.2,.5,1.)):
        guide=dict(schema='defensive-hard-free-line-guide-v1',raw_translation_axes=[0,1,2],
            conditional_probability=1.,minimum_conditional_mass=1e-12,defensive_uniform_shell_probability=.5,
            gaussian_components=[dict(weight=1.)])
        volume=2.;uniform=math.log(.5)-volume if inside else -math.inf
        baseline=lse([uniform,math.log(.5)])
        axes=[];qs=[]
        for axis,mass in enumerate(masses):
            fallback=mass<=1e-12
            factor=1 if fallback else 1/mass if allowed else 0
            q=lse([uniform,math.log(.5*factor) if factor else -math.inf]);qs.append(q)
            axes.append(dict(axis=axis,components=[dict(component=0,gaussian_log_density=0.,
                conditional_mean=0.,conditional_sigma=1.,conditional_mass=mass,fallback=fallback,
                query_coordinate_allowed=allowed)],axis_log_proposal_density=q if math.isfinite(q) else None,
                geometry_cpu_seconds=.1,hard_free_intervals=[]))
        xyz=lse(qs)-math.log(3)
        row=dict(shell_valid=inside,hard_valid=allowed,capture_valid=True,baseline_log_density=baseline,
            log_proposal_density=xyz if math.isfinite(xyz) else None,
            density_details=dict(axes=axes,component_branches=3,
                fallback_component_branches=sum(m<=1e-12 for m in masses)))
        return row,guide,volume

    def test_complete_axis_average_and_valid_density_increase(self):
        args=self.fixture();result=derive(*args)
        self.assertAlmostEqual(result['log_q']['xyz'],lse(result['log_q'][a] for a in ('x','y','z'))-math.log(3))
        self.assertTrue(all(q>=result['baseline_log_q'] for q in result['log_q'].values()))

    def test_floor_fallback_retains_original_conditional(self):
        result=derive(*self.fixture(inside=False,allowed=False,masses=(0.,1e-13,1e-12)))
        self.assertTrue(all(q==result['baseline_log_q'] for q in result['log_q'].values()))

    def test_genuine_zero_density_outside_uniform_domain_retained(self):
        result=derive(*self.fixture(inside=False,allowed=False))
        self.assertEqual(result['log_q'],dict(x=None,y=None,z=None,xyz=None))
        summary=distribution([-math.inf,0.,1.])
        self.assertEqual(summary['rows'],3);self.assertEqual(summary['zero_density_rows'],1)
        self.assertTrue(summary['minimum_is_negative_infinity'])

    def test_missing_axis_or_wrong_fallback_rejected(self):
        args=self.fixture();args[0]['density_details']['axes'].pop()
        with self.assertRaises(ValueError):derive(*args)
        args=self.fixture();args[0]['density_details']['axes'][0]['components'][0]['fallback']=True
        with self.assertRaises(ValueError):derive(*args)


if __name__=='__main__':unittest.main()
