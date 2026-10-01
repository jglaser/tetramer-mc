import copy
import math
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal,norm
from hard_free_line_reference import Reconstructor,contains,hard_free_intervals,interval,interval_masses,subtract_closed_segment,union
from test_analyze_contact_line_audit import setup as old_setup


def setup(axes=(0,),beta=1.):
    region,guide,config,shape=old_setup(axes,beta)
    guide['schema']='defensive-hard-free-line-guide-v1';guide.pop('contact_widths_A');guide.pop('contact_neighbor_indices')
    return region,guide,config,shape


class WholeLineGeometryTests(unittest.TestCase):
    def test_union_and_difference_retain_isolated_tangent(self):
        blocked=[interval(-2.,0.,True,False),interval(0.,2.,False,True)]
        self.assertEqual(union(blocked),blocked)
        self.assertEqual(subtract_closed_segment([-2.,2.],blocked),[interval(0.,0.)])
        self.assertEqual(subtract_closed_segment([-2.,2.],[interval(-2.,2.)]),[])
        self.assertEqual(subtract_closed_segment([-2.,2.],[]),[interval(-2.,2.)])

    def test_two_sphere_clearance_and_tangent_no_overlap(self):
        shape=dict(atoms=[dict(center=[0.,0.,0.],radius=1.)])
        pose=lambda p:dict(position=p,orientation=[1.,0.,0.,0.])
        got=hard_free_intervals(shape,[pose([-2.05,0,0]),pose([2.05,0,0])],np.zeros(3),np.eye(3),[1.,0,0],[-3.,3.])
        self.assertEqual(len(got['intervals']),1)
        np.testing.assert_allclose([got['intervals'][0]['lower'],got['intervals'][0]['upper']],[-.05,.05],atol=3e-16)
        self.assertTrue(got['intervals'][0]['lower_closed'] and got['intervals'][0]['upper_closed'])
        tangent=hard_free_intervals(shape,[pose([0,2.,0])],np.zeros(3),np.eye(3),[1.,0,0],[-3.,3.])
        self.assertEqual(tangent['intervals'],[interval(-3.,3.)])

    def test_projected_candidates_match_all_pairs_and_direct_witnesses(self):
        shape=dict(atoms=[dict(center=p,radius=r) for p,r in [([-.5,0.,0.],.5),([.4,.2,0.],.7),([0.,-.3,.4],.4)]])
        fixed=[dict(position=[-2.,.4,.1],orientation=[1.,0,0,0]),dict(position=[2.,-.4,.2],orientation=[1.,0,0,0])]
        direction=np.array([1.,.2,-.1]);direction/=np.linalg.norm(direction);origin=np.array([0.,.1,.05]);segment=[-4.,4.]
        fast=hard_free_intervals(shape,fixed,origin,np.eye(3),direction,segment)
        slow=hard_free_intervals(shape,fixed,origin,np.eye(3),direction,segment,use_tree=False)
        self.assertEqual(fast['intervals'],slow['intervals'])
        atoms=np.array([a['center'] for a in shape['atoms']]);r=np.array([a['radius'] for a in shape['atoms']])
        for s in np.linspace(*segment,257):
            moving=atoms+origin+s*direction
            valid=all(np.all(np.linalg.norm(moving[:,None,:]-(atoms+f['position'])[None,:,:],axis=-1)>=r[:,None]+r[None,:]) for f in fixed)
            self.assertEqual(contains(fast['intervals'],s),valid)


class GaussianLawTests(unittest.TestCase):
    def test_complete_density_and_component_specific_floor(self):
        r=Reconstructor(*setup());u=np.zeros(6);result=r.density(u)
        self.assertEqual(result['fallback_component_branches'],1)
        axes=result['axes'];m=np.array(axes[0]['conditional_masses']);g=np.exp(r.gaussian_logs(u))
        expected=.5*math.exp(-r.logvolume)+.5*(g[0]/m[0]+g[1])
        self.assertAlmostEqual(math.exp(result['log_density'])/expected,1.,places=13)
        self.assertGreater(result['log_density'],result['baseline_log_density'])

    def test_conditional_normalization_and_retained_five_coordinates(self):
        r=Reconstructor(*setup());u=np.zeros(6);x=r.raw(u);geometry=r.reconstruct_axis(u,0);means,sigmas=r.conditional(x,0)
        for m,s in zip(means,sigmas):
            mass=float(interval_masses(geometry['intervals'],m,s))
            if mass<=r.floor:value=quad(lambda v:norm.pdf(v,m,s),m-12*s,m+12*s)[0]
            else:value=sum(quad(lambda v:norm.pdf(v,m,s)/mass,i['lower'],i['upper'])[0] for i in geometry['intervals'])
            self.assertAlmostEqual(value,1.,places=11)
        changed=x.copy();changed[0]=.02;u2=np.linalg.solve(r.L0,changed-r.m0)
        np.testing.assert_allclose(r.decode(u)[2],r.decode(u2)[2],atol=1e-14)
        np.testing.assert_allclose(r.conditional(x,0)[0],r.conditional(changed,0)[0],atol=1e-14)

    def test_axis_mixture_and_physical_pointwise_dominance(self):
        region,guide,cfg,shape=setup((0,1,2));r=Reconstructor(region,guide,cfg,shape);u=np.zeros(6)
        whole=r.density(u);parts=[]
        for axis in r.axes:
            g=copy.deepcopy(guide);g['raw_translation_axes']=[axis];part=Reconstructor(region,g,cfg,shape).density(u)
            self.assertGreaterEqual(part['log_density'],part['baseline_log_density']-1e-14);parts.append(math.exp(part['log_density']))
        self.assertAlmostEqual(math.exp(whole['log_density']),sum(parts)/3,places=13)

    def test_uniform_disabled_empty_line_and_zero_density_outside_support(self):
        for mode in ['uniform','disabled','empty']:
            reg,g,cfg,shape=setup()
            if mode=='uniform':g['defensive_uniform_shell_probability']=1.;g['gaussian_components']=[]
            if mode=='disabled':g['conditional_probability']=0.
            if mode=='empty':cfg['fixed_poses']=[dict(position=[0.,0.,0.],orientation=[1.,0,0,0])];cfg['capture_radius']=.1
            r=Reconstructor(reg,g,cfg,shape);result=r.density(np.zeros(6))
            self.assertEqual(result['log_density'],result['baseline_log_density'])
        reg,g,cfg,shape=setup();g['gaussian_components']=g['gaussian_components'][:1];r=Reconstructor(reg,g,cfg,shape)
        # Fixed outer coordinates pass through the valid central gap, but the
        # queried translation is outside R4. beta=1 removes its Gaussian tail.
        raw=np.zeros(6);raw[0]=8.;u=np.linalg.solve(r.L0,raw-r.m0)
        result=r.density(u);self.assertEqual(result['log_density'],-math.inf)
        self.assertTrue(math.isfinite(result['baseline_log_density']))

    def test_new_schema_rejects_contact_constraints(self):
        reg,g,cfg,shape=setup();g['contact_widths_A']=[.1]
        with self.assertRaisesRegex(ValueError,'must not carry contact'):Reconstructor(reg,g,cfg,shape)


if __name__=='__main__':unittest.main()
