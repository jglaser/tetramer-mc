#!/usr/bin/env python3
"""Synthetic, adversarial tests of the split audit; no simulation or campaign I/O."""
import copy
import math
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.special import logsumexp
from physical_hard_free_line_vessel import PhysicalGuideDensity
import hard_free_line_reference as line
import conditioned_density_audit as audit


class SyntheticReference:
    def __init__(self, means=(0., .5), sigmas=(1., 1.3), axes=(0,)):
        self.axes = list(axes)
        self.weights = np.array([.7, .3]) if len(means) == 2 else np.ones(len(means))/len(means)
        self.means, self.sigmas = np.array(means), np.array(sigmas)
        self.floor = 1e-12
        self.alpha, self.beta, self.radius, self.logvolume = .5, 1., 4., math.log(math.pi**3*4**6/6)

    def raw(self, u): return np.asarray(u)
    def conditional(self, x, axis): return self.means.copy(), self.sigmas.copy()
    def gaussian_logs(self, u):
        return np.log(self.weights)-3*math.log(2*math.pi)-np.dot(u,u)/2


def compose(r, u, factors):
    correction = 1-r.beta+r.beta*np.asarray(factors)
    active = correction > 0
    uniform = math.log(r.alpha)-r.logvolume if np.dot(u,u) <= r.radius**2 else -math.inf
    return float(np.logaddexp(uniform,math.log1p(-r.alpha)+logsumexp(r.gaussian_logs(u)[active]+np.log(correction[active]))))


def fixture(independent=None, saved=None, r=None, u=None, full=False):
    r = r or SyntheticReference()
    u = np.array([0.]*6) if u is None else np.array(u,float)
    if independent is None:
        independent = [line.interval(-1.3e-6, 1.4e-6)]
    if saved is None:
        saved = copy.deepcopy(independent)
    if isinstance(independent,list): independent={axis:independent for axis in r.axes}
    if isinstance(saved,list): saved={axis:saved for axis in r.axes}
    actual_axes, reference_axes, all_saved, all_ref = [], [], [], []
    ref_fallbacks = 0; actual_fallbacks = 0
    for axis in r.axes:
        means,sigmas=r.conditional(u,axis)
        mr=line.interval_masses(independent[axis],means,sigmas); fr=mr<=r.floor
        ms=line.interval_masses(saved[axis],means,sigmas); fs=ms<=r.floor
        def factors(m,f,iv):
            v=np.ones(len(m));v[~f]=1/m[~f] if line.contains(iv,u[axis]) else 0.;return v
        vr,vs=factors(mr,fr,independent[axis]),factors(ms,fs,saved[axis])
        actual=dict(axis=axis,hard_free_intervals=saved[axis],axis_log_proposal_density=compose(r,u,vs))
        reference=dict(axis=axis,intervals=independent[axis],conditional_means=means.tolist(),conditional_sigmas=sigmas.tolist(),conditional_masses=mr.tolist(),component_fallbacks=fr.tolist())
        if full:
            actual['components']=[dict(component=k,gaussian_log_density=r.gaussian_logs(u)[k]-math.log(r.weights[k]),conditional_mean=means[k],conditional_sigma=sigmas[k],conditional_mass=ms[k],fallback=bool(fs[k]),query_coordinate_allowed=line.contains(saved[axis],u[axis])) for k in range(len(means))]
        actual_axes.append(actual);reference_axes.append(reference);all_saved.append(vs);all_ref.append(vr)
        ref_fallbacks+=int(fr.sum());actual_fallbacks+=int(fs.sum())
    qs=compose(r,u,np.mean(all_saved,axis=0));qr=compose(r,u,np.mean(all_ref,axis=0))
    baseline=compose(r,u,np.ones(len(r.weights)))
    reconstruction=dict(log_density=qr,baseline_log_density=baseline,axes=reference_axes,component_branches=len(r.axes)*len(r.weights),fallback_component_branches=ref_fallbacks)
    trace=dict(raw_coordinates=u.tolist(),baseline_log_density=baseline,axes=actual_axes,component_branches=len(r.axes)*len(r.weights),fallback_component_branches=actual_fallbacks)
    expected=PhysicalGuideDensity(u.tolist(),bool(u@u<=16),qr,1.2,qr-1.2,qr==-math.inf,reconstruction)
    record=dict(latent=u.tolist(),in_reference_ball=expected.in_reference_ball,log_latent_density=qs,log_physical_jacobian=1.2,log_physical_density=qs-1.2,structural_zero=qs==-math.inf,hard_free_line_density=trace)
    return record,expected,SimpleNamespace(recon=r)


class SplitDensityAuditTests(unittest.TestCase):
    def test_positive_narrow_gap_preserves_strict_arithmetic(self):
        ref=[line.interval(-1.3e-6,1.4e-6)]
        saved=[line.interval(-1.3e-6,1.4e-6+8e-14)]
        record,expected,guide=fixture(ref,saved,full=True)
        self.assertGreater(abs(record['log_latent_density']-expected.log_latent_density),audit.LOG_ATOL)
        result=audit.audit_conditioned_density(record,expected,guide)
        self.assertTrue(result['complete'])
        self.assertLess(result['maxima']['same_input_log_error'],1e-12)
        self.assertGreater(result['maxima']['log_envelope_width'],audit.LOG_ATOL)
        self.assertLess(result['maxima']['log_envelope_width'],audit.LOG_ENVELOPE_CAP)
        self.assertAlmostEqual(result['log_latent_density_saved_intervals'],record['log_latent_density'])
        self.assertAlmostEqual(result['log_latent_density_independent_geometry'],expected.log_latent_density)

    def test_complete_multi_axis_density_is_not_endpoint_average(self):
        r=SyntheticReference(axes=(0,1,2))
        ref={0:[line.interval(-1.3e-6,1.4e-6)],1:[line.interval(-.7,.8)],2:[]}
        saved=copy.deepcopy(ref);saved[0][0]['upper']+=8e-14
        record,expected,guide=fixture(ref,saved,r)
        result=audit.audit_conditioned_density(record,expected,guide)
        self.assertEqual(len(result['axes']),3)
        self.assertAlmostEqual(result['log_latent_density_saved_intervals'],record['log_latent_density'])
        self.assertEqual(result['axes'][2]['component_fallbacks'],[True,True])

    def test_union_envelope_does_not_double_count_overlapping_hulls(self):
        a=[line.interval(0.,1.,False,False),line.interval(1.+1e-10,2.,False,False)]
        b=[line.interval(0.,1.+2e-10,False,False),line.interval(1.+3e-10,2.,False,False)]
        inner,outer=audit.interval_envelope(a,b)
        self.assertEqual(len(inner),2);self.assertEqual(len(outer),1)
        self.assertEqual(outer[0]['lower'],0.);self.assertEqual(outer[0]['upper'],2.)
        masses=line.interval_masses(outer,np.array([0.]),np.array([1.]))
        expected=line.interval_masses([line.interval(0.,2.)],np.array([0.]),np.array([1.]))
        np.testing.assert_array_equal(masses,expected)

    def test_saved_density_tampering_rejected_even_inside_geometry_cap(self):
        record,expected,guide=fixture()
        record['log_latent_density']+=1e-7;record['log_physical_density']+=1e-7
        with self.assertRaisesRegex(ValueError,'saved-input density'):
            audit.audit_conditioned_density(record,expected,guide)

    def test_axis_density_tampering_rejected(self):
        record,expected,guide=fixture()
        record['hard_free_line_density']['axes'][0]['axis_log_proposal_density']+=1e-7
        with self.assertRaisesRegex(ValueError,'Single-axis saved-input density'):
            audit.audit_conditioned_density(record,expected,guide)

    def test_jacobian_and_physical_density_tampering_rejected(self):
        for key in ['log_physical_jacobian','log_physical_density']:
            with self.subTest(key=key):
                record,expected,guide=fixture();record[key]+=1e-4
                with self.assertRaises(ValueError):audit.audit_conditioned_density(record,expected,guide)

    def test_coordinate_raw_baseline_and_support_tampering_rejected(self):
        for key in ['latent','raw','baseline','support','axis_order','count']:
            with self.subTest(key=key):
                record,expected,guide=fixture();trace=record['hard_free_line_density']
                if key=='latent':record['latent'][0]+=1e-3
                elif key=='raw':trace['raw_coordinates'][0]+=1e-3
                elif key=='baseline':trace['baseline_log_density']+=1e-4
                elif key=='support':record['structural_zero']=True
                elif key=='axis_order':trace['axes'][0]['axis']=1
                elif key=='count':trace['component_branches']+=1
                with self.assertRaises(ValueError):audit.audit_conditioned_density(record,expected,guide)

    def test_endpoint_tolerance_is_unchanged(self):
        ref=[line.interval(-.7,.8)];saved=[line.interval(-.7,.8+2e-9)]
        with self.assertRaisesRegex(ValueError,'hard-free interval'):
            audit.audit_conditioned_density(*fixture(ref,saved))

    def test_small_absolute_geometry_error_can_exceed_log_envelope_cap(self):
        ref=[line.interval(-1.3e-6,1.4e-6)];saved=[line.interval(-1.3e-6,1.4e-6+1e-11)]
        with self.assertRaisesRegex(ValueError,'log-envelope cap'):
            audit.audit_conditioned_density(*fixture(ref,saved))

    def test_topology_and_endpoint_flags_rejected(self):
        for saved in [[line.interval(-1.3e-6,0.),line.interval(1e-8,1.4e-6)],
                      [line.interval(-1.3e-6,1.4e-6,False,True)]]:
            with self.subTest(saved=saved):
                with self.assertRaises(ValueError):audit.audit_conditioned_density(*fixture(saved=saved))

    def test_support_switch_below_endpoint_tolerance_is_rejected(self):
        ref=[line.interval(-1e-3,2e-14)];saved=[line.interval(-1e-3,-2e-14)]
        with self.assertRaisesRegex(ValueError,'Query support'):
            audit.audit_conditioned_density(*fixture(ref,saved))

    def test_per_component_floor_crossing_rejected(self):
        r=SyntheticReference(means=(0.,),sigmas=(1.,))
        half=math.sqrt(2*math.pi)*r.floor/2
        ref=[line.interval(-half*.99,half*.99)];saved=[line.interval(-half*1.01,half*1.01)]
        with self.assertRaisesRegex(ValueError,'fallback'):
            audit.audit_conditioned_density(*fixture(ref,saved,r))

    def test_equal_aggregate_fallback_counts_do_not_hide_component_swap(self):
        r=SyntheticReference(means=(-1.,1.),sigmas=(.01,.01))
        half=math.sqrt(2*math.pi)*r.floor*.01/2
        # At 1.0 these intervals are near machine precision; a slightly larger
        # floor keeps the adversarial exact count swap robust on every platform.
        r.floor=1e-7;half=math.sqrt(2*math.pi)*r.floor*.01/2
        ref=[line.interval(-1-half*.98,-1+half*.98),line.interval(1-half*1.02,1+half*1.02)]
        saved=[line.interval(-1-half*1.02,-1+half*1.02),line.interval(1-half*.98,1+half*.98)]
        record,expected,guide=fixture(ref,saved,r)
        self.assertEqual(record['hard_free_line_density']['fallback_component_branches'],expected.reconstruction['fallback_component_branches'])
        with self.assertRaisesRegex(ValueError,'Per-component'):
            audit.audit_conditioned_density(record,expected,guide)

    def test_full_component_trace_flags_checked_directly(self):
        record,expected,guide=fixture(full=True)
        record['hard_free_line_density']['axes'][0]['components'][0]['fallback']=True
        with self.assertRaisesRegex(ValueError,'Recorded component'):
            audit.audit_conditioned_density(record,expected,guide)

    def test_empty_interval_fallback_is_valid(self):
        result=audit.audit_conditioned_density(*fixture([],[]))
        self.assertEqual(result['axes'][0]['component_fallbacks'],[True,True])
        self.assertEqual(result['maxima']['log_envelope_width'],0.)

    def test_invalid_input_nan_and_missing_jacobian_fail(self):
        for key,value in [('log_latent_density',float('nan')),('log_physical_jacobian',None)]:
            record,expected,guide=fixture();record[key]=value
            with self.assertRaises(ValueError):audit.audit_conditioned_density(record,expected,guide)
        record,expected,guide=fixture();del record['log_physical_jacobian']
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            audit.audit_conditioned_density(record,expected,guide)

    def test_disabled_conditioner_keeps_original_strict_check(self):
        record,expected,guide=fixture();r=guide.recon;r.beta=0.
        q=compose(r,np.zeros(6),np.ones(len(r.weights)))
        expected=PhysicalGuideDensity([0.]*6,True,q,1.2,q-1.2,False,dict(conditioning_disabled=True,log_density=q))
        record.update(log_latent_density=q,log_physical_density=q-1.2,hard_free_line_density=dict(conditioning_disabled=True))
        self.assertTrue(audit.audit_conditioned_density(record,expected,guide)['complete'])
        record['log_latent_density']+=1e-7;record['log_physical_density']+=1e-7
        with self.assertRaisesRegex(ValueError,'Disabled'):
            audit.audit_conditioned_density(record,expected,guide)

if __name__=='__main__': unittest.main()
