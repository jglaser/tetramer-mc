#!/usr/bin/env python3
"""Deterministic arc, probability and tiny-union tests; no protein diagnosis."""
import math
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from contact_circle_reference import (TAU,azimuth_mass,circle_from_case,complement,contains,direct_hard_valid,
    hard_free_arcs,interval,open_arc,pair_forbidden_arcs,union,wrapped_cauchy_interval_mass)


BASIS=[[1.,0.,0.],[0.,1.,0.]]
IDENTITY=[1.,0.,0.,0.]


class ArcTests(unittest.TestCase):
    def test_decimal_near_tangent_references(self):
        fixture=Path(__file__).resolve().parents[1]/'tests/data/contact_circle_tangent_reference.json'
        for case in json.loads(fixture.read_text())['cases']:
            with self.subTest(case=case['name']):
                arcs=pair_forbidden_arcs([0.,0.,0.],case['fixed_offset'],case['radius_sum'],case['circle_radius'],BASIS)
                allowed=complement(arcs);length=math.fsum(i['upper']-i['lower'] for i in allowed)
                self.assertAlmostEqual(length,case['allowed_length'],delta=3e-12)
                if case['analytic_topology']=='none':self.assertEqual(arcs,[])
                if case['analytic_topology']=='all':self.assertEqual(allowed,[])

    def test_wrapping_and_exact_tangencies(self):
        arcs=pair_forbidden_arcs([0.,0.,0.],[1.,0.,0.],1.,1.,BASIS)
        self.assertEqual(len(arcs),2);self.assertTrue(contains(arcs,0));self.assertFalse(contains(arcs,math.pi))
        self.assertAlmostEqual(arcs[0]['upper'],math.pi/3,places=14)
        self.assertEqual(pair_forbidden_arcs([0.,0.,0.],[2.,0.,0.],1.,1.,BASIS),[])
        all_except=pair_forbidden_arcs([0.,0.,0.],[1.,0.,0.],2.,1.,BASIS)
        self.assertEqual(complement(all_except),[interval(math.pi,math.pi,True,True)])
        self.assertEqual(azimuth_mass(complement(all_except),dict(mode=0.,gamma=.1,localized_probability=.9)),0.)
        self.assertTrue(contains(complement(all_except),math.pi));self.assertFalse(contains(all_except,math.pi))

    def test_constant_circle_and_axis_cases(self):
        for rho,position,s,full in [(0.,[.5,0,0],1.,True),(0.,[1.,0,0],1.,False),
            (1.,[0.,0,0],1.,False),(1.,[0.,0,0],2.,True),(1.,[0.,0,2.],3.,True)]:
            got=pair_forbidden_arcs([0.,0.,0.],position,s,rho,BASIS)
            self.assertEqual(got,[interval(0,TAU,True,False)] if full else [])

    def test_endpoint_union_complement_involution(self):
        pieces=[interval(0,.5,True,False),interval(.5,1.,False,True),interval(1.,2.,False,False),interval(3.,4.,True,True)]
        joined=union(pieces);allowed=complement(joined)
        self.assertEqual(complement(allowed),joined)
        self.assertTrue(contains(allowed,.5));self.assertFalse(contains(allowed,1.));self.assertTrue(contains(allowed,2.))
        self.assertFalse(contains(allowed,3.));self.assertFalse(contains(allowed,4.))
        self.assertEqual(union([*open_arc(0.,1.),*open_arc(math.pi,math.pi-.5)]),[interval(0,TAU,True,False)])

    def test_near_tangent_nonzero_sector_not_clamped(self):
        # A one-ulp growth beyond external tangency yields a small positive arc.
        s=math.nextafter(1.,math.inf);arcs=pair_forbidden_arcs([0.,0.,0.],[2.,0.,0.],s,1.,BASIS)
        self.assertGreater(sum(i['upper']-i['lower'] for i in arcs),0)
        self.assertLess(sum(i['upper']-i['lower'] for i in arcs),1e-6)

    def test_wrapped_cauchy_analytic_integrals_and_partition(self):
        for gamma in [.01,.2,1.,math.pi]:
            for mode in [0.,.7,4.8]:
                def density(phi):return math.sinh(gamma)/(2*math.pi*(2*math.sinh(gamma/2)**2+2*math.sin((phi-mode)/2)**2))
                for lo,hi in [(0.,TAU),(.1,.100000000001),(0.,.3),(2.,5.7),(5.,TAU)]:
                    numerical=quad(density,lo,hi,epsabs=1e-14,epsrel=1e-12,points=[mode] if lo<mode<hi else None)[0]
                    analytic=wrapped_cauchy_interval_mass(lo,hi,mode,gamma)
                    self.assertAlmostEqual(analytic/numerical,1.,delta=3e-11)
                cuts=[0.,.05,.9,2.6,4.1,6.1,TAU]
                self.assertAlmostEqual(sum(wrapped_cauchy_interval_mass(a,b,mode,gamma) for a,b in zip(cuts[:-1],cuts[1:])),1.,places=13)
        arcs=union([interval(.2,.5),interval(5.9,TAU)])
        law=dict(mode=.3,gamma=.02,localized_probability=.9)
        self.assertAlmostEqual(azimuth_mass(arcs,law)+azimuth_mass(complement(arcs),law),1.,places=13)
        self.assertAlmostEqual(azimuth_mass(arcs,dict(law,localized_probability=0)),(.3+TAU-5.9)/TAU,places=14)

    def test_random_pair_witnesses(self):
        generator=np.random.default_rng(842)
        for _ in range(80):
            fixed=generator.uniform(-2.,2.,3);rho=float(generator.uniform(.01,2.));s=float(generator.uniform(.1,2.))
            arcs=pair_forbidden_arcs([0.,0.,0.],fixed,s,rho,BASIS)
            for phi in np.linspace(.031,TAU-.017,97):
                point=rho*np.array([math.cos(phi),math.sin(phi),0.])
                self.assertEqual(contains(arcs,phi),np.linalg.norm(point-fixed)<s)


class TinyUnionTests(unittest.TestCase):
    def test_frozen_pair_label_basis_and_original_circle(self):
        shape=dict(atoms=[dict(center=[0.,0.,0.],radius=1.)]);fixed=[dict(position=[-1.2,0.,0.],orientation=IDENTITY),dict(position=[1.2,0.,0.],orientation=IDENTITY)]
        rho=math.sqrt(2.2**2-1.2**2);phi=.7
        case=dict(pose=dict(position=[0.,-rho*math.sin(phi),rho*math.cos(phi)],orientation=IDENTITY),
            contact_pairs=[dict(neighbor_index=k,moving_atom=0,fixed_atom=0) for k in range(2)],radii=[2.2,2.2],
            phi=phi,circle_center=[0.,0.,0.],circle_radius=rho)
        got=circle_from_case(shape,dict(fixed_poses=fixed),case)
        self.assertTrue(np.allclose(got['basis'],[[0,0,1],[0,-1,0]]))
        self.assertLess(max(got['reconstruction_errors'].values()),1e-14)

    def test_tree_candidates_match_all_pairs_and_direct_witnesses(self):
        generator=np.random.default_rng(4121)
        for case in range(18):
            centers=generator.uniform(-1.2,1.2,(5,3));radii=generator.uniform(.12,.36,5)
            shape=dict(atoms=[dict(center=c.tolist(),radius=float(r)) for c,r in zip(centers,radii)])
            rotation=Rotation.from_rotvec(generator.normal(size=3)*.3).as_quat();orientation=rotation[[3,0,1,2]].tolist()
            fixed=[dict(position=generator.uniform(-2.,2.,3).tolist(),orientation=IDENTITY) for _ in range(2)]
            center=generator.uniform(-.5,.5,3);rho=float(generator.uniform(.2,2.))
            tree=hard_free_arcs(shape,fixed,orientation,center,rho,BASIS)
            brute=hard_free_arcs(shape,fixed,orientation,center,rho,BASIS,use_tree=False)
            self.assertEqual(tree['forbidden'],brute['forbidden'],case)
            for phi in np.linspace(.017,TAU-.029,127):
                self.assertEqual(contains(tree['allowed'],phi),direct_hard_valid(shape,fixed,orientation,center,rho,BASIS,phi),(case,phi))

    def test_full_coverage_early_exit_is_exact(self):
        shape=dict(atoms=[dict(center=[0.,0.,0.],radius=1.),dict(center=[8.,0.,0.],radius=.1)])
        fixed=[dict(position=[0.,0.,0.],orientation=IDENTITY)]
        got=hard_free_arcs(shape,fixed,IDENTITY,[0.,0.,0.],.2,BASIS)
        self.assertTrue(got['full_coverage_early_exit']);self.assertFalse(got['has_any_allowed_point'])
        self.assertEqual(got['uniform_allowed_probability'],0.)

    def test_rigid_frame_covariance(self):
        shape=dict(atoms=[dict(center=[0.,0.,0.],radius=.5),dict(center=[.4,.2,0.],radius=.2)])
        fixed=[dict(position=[1.5,.2,.1],orientation=IDENTITY)];center=np.array([0.,0.,0.]);rho=.8
        before=hard_free_arcs(shape,fixed,IDENTITY,center,rho,BASIS)
        Q=Rotation.from_rotvec([.4,-.2,.3]);R=Q.as_matrix();q=Q.as_quat()[[3,0,1,2]].tolist();offset=np.array([20.,-14.,3.])
        transformed=[dict(position=(R@p['position']+offset).tolist(),orientation=q) for p in fixed]
        after=hard_free_arcs(shape,transformed,q,offset,rho,np.asarray(BASIS)@R.T)
        self.assertEqual(len(before['allowed']),len(after['allowed']))
        for a,b in zip(before['allowed'],after['allowed']):
            self.assertAlmostEqual(a['lower'],b['lower'],places=12);self.assertAlmostEqual(a['upper'],b['upper'],places=12)


if __name__=='__main__':unittest.main()
