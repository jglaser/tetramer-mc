#!/usr/bin/env python3
"""Saved interval-record checks only; no atomic geometry or random draws."""
import copy
import math
import unittest
from compare_contact_circle_feasibility import compare_case,contains,partition_errors,interval_errors


def interval(lo,hi,lc=True,uc=False):
    return dict(lower=lo,upper=hi,lower_closed=lc,upper_closed=uc)


def records():
    allowed=[interval(0.,math.pi,True,True)];forbidden=[interval(math.pi,2*math.pi,False,False)]
    case=dict(id='case',arm='uniform_phi92',population='r00',width_index=0,
        azimuth_law=dict(mode=0.,gamma=1.,localized_probability=0.),circle_center=[0.,0.,0.],circle_radius=1.,
        pose=dict(orientation=[1.,0.,0.,0.]),phi=math.pi/2,hard_valid=True,shell_valid=False,capture_valid=True)
    rust=dict(id=case['id'],arm=case['arm'],population='r00',width_index=0,azimuth_law=case['azimuth_law'],
        circle=dict(center=case['circle_center'],radius=1.,basis=[[1.,0.,0.],[0.,1.,0.]]),
        moving_orientation=case['pose']['orientation'],geometry=dict(allowed=dict(intervals=allowed),
            forbidden=dict(intervals=forbidden),counts=dict(node_pairs_visited=1)),
        uniform_allowed_mass=.5,original_law_allowed_mass=.5,original_hard_valid=True,
        witnesses=[dict(phi=case['phi'],hard_free=True),dict(phi=1.5*math.pi,hard_free=False)],geometry_cpu_seconds=.001)
    python=dict(id=case['id'],arm=case['arm'],population='r00',basis=rust['circle']['basis'],
        original_phi=case['phi'],allowed=copy.deepcopy(allowed),forbidden=copy.deepcopy(forbidden),
        allowed_length=math.pi,uniform_allowed_probability=.5,azimuth_allowed_probability=.5,
        has_positive_allowed_length=True,has_any_allowed_point=True,original_hard_valid=True,
        analysis_cpu_seconds=.002,counts=dict(leaf_pairs=1))
    return case,rust,python


class ComparisonTests(unittest.TestCase):
    def test_agreement_and_old_domain_flag_does_not_remove_case(self):
        result=compare_case(*records())
        self.assertTrue(result['passed']);self.assertTrue(result['has_positive_allowed_length'])
        self.assertFalse(result['original_hard_and_domain_valid'])

    def test_mass_endpoint_topology_and_witness_tampering_detected(self):
        for change,expected in [
            (lambda p:p.update(azimuth_allowed_probability=.51),'original_law_mass'),
            (lambda p:p['allowed'][0].update(upper=math.pi+1e-8),'allowed_endpoints'),
            (lambda p:p['allowed'][0].update(upper_closed=False),'allowed_endpoint_topology'),
            (lambda p:p.update(original_hard_valid=False),'original_hard_predicate'),
            (lambda p:p.update(allowed=[]),'allowed_interval_count')]:
            c,r,p=records();change(p);result=compare_case(c,r,p)
            self.assertFalse(result['passed']);self.assertFalse(result['checks'][expected])
        c,r,p=records();r['witnesses'][1]['hard_free']=True
        self.assertFalse(compare_case(c,r,p)['checks']['all_saved_witness_memberships'])

    def test_absolute_tolerance_reported_without_relative_scaling(self):
        c,r,p=records();p['azimuth_allowed_probability']+=1e-11
        result=compare_case(c,r,p);self.assertTrue(result['passed'])
        self.assertGreater(result['absolute_errors']['original_law_mass'],0.)

    def test_tangent_singleton_and_periodic_cut(self):
        allowed=[interval(math.pi,math.pi,True,True)]
        forbidden=[interval(0.,math.pi,True,False),interval(math.pi,2*math.pi,False,False)]
        self.assertFalse(interval_errors(allowed));self.assertFalse(partition_errors(allowed,forbidden))
        self.assertTrue(contains(allowed,math.pi));self.assertFalse(contains(allowed,0.))
        self.assertTrue(contains(forbidden,2*math.pi))

    def test_invalid_canonical_intervals_rejected(self):
        self.assertTrue(interval_errors([interval(0.,2*math.pi,True,True)]))
        self.assertTrue(interval_errors([interval(0.,1.),interval(.9,1.2)]))
        self.assertTrue(partition_errors([],[]))


if __name__=='__main__':unittest.main()
