"""Exact reciprocal SE(3) density, chart and virtual-trace controls.

All data are synthetic; no protein geometry, bath, production or audit replay.
"""
import copy
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from prepare_smc_normalizer_atlas import (
    Density, arrays, reciprocal_arrays, unwrap_proposal_model,
)
from analyze_involution_docking_campaign import ChartAudit
from analyze_mobile_posterior_pilot import audit_densities, full_capture_log_density


def fixture(count=3):
    result=dict(schema='weighted-pose-mixture-v1',coordinate_convention='anchor-body-relative',
        shape_sha256='synthetic',angular_length=2.3,anchors=[],means=[],covariances=[],weights=[])
    for i in range(count):
        result['anchors'].append(dict(position=[2.+3*i,-1.+.7*i,.3-i],
            rotation=Rotation.from_rotvec([.2+.3*i,-.1+.2*i,.4-.1*i]).as_matrix().tolist()))
        result['means'].append([.1*i,-.05*i,.03*i,.07*i,-.02*i,.04*i])
        lower=np.diag([.8,.5,.7,.4,.6,.3]);lower[2,0]=.13;lower[4,1]=-.08
        result['covariances'].append(((1+.2*i)*lower@lower.T).tolist())
        result['weights'].append((i+1)/(count*(count+1)/2))
    return result


def envelope(base,flags=None):
    return dict(schema='reciprocal-pose-mixture-v1',base_model=base,
        reciprocal_components=[True]*len(base['weights']) if flags is None else flags)


def inversion(values):
    t,_,r=arrays(values);t,r=reciprocal_arrays(t,r)
    q=Rotation.from_matrix(r).as_quat()[:,[3,0,1,2]]
    return [dict(position=p.tolist(),orientation=v.tolist()) for p,v in zip(t,q)]


def decode_base(base,index,z):
    lower=np.linalg.cholesky(base['covariances'][index])
    value=np.asarray(base['means'][index])+lower@z
    q=np.r_[value[3:]/base['angular_length'],1.]
    r=Rotation.from_quat(q).as_matrix()@np.asarray(base['anchors'][index]['rotation'])
    p=value[:3]+base['anchors'][index]['position']
    jac=np.log(np.diag(lower)).sum()-3*np.log(base['angular_length'])-2*np.log(np.pi)-2*np.log1p(np.sum((value[3:]/base['angular_length'])**2))
    return dict(position=p.tolist(),orientation=Rotation.from_matrix(r).as_quat()[[3,0,1,2]].tolist()),float(jac)


def posterior_record(model,source,target,correlation):
    density=Density(model);base,_=unwrap_proposal_model(model)
    z=np.array([.4,-.3,.7,.5,-.2,.6]);noise=np.array([-.5,.1,.3,-.4,.6,.2]);sine=np.sqrt(1-correlation**2)
    target_z=correlation*z+sine*noise;inverse_noise=sine*z-correlation*noise
    old,jo=decode_base(base,int(density.base_indices[source]),z)
    new,jn=decode_base(base,int(density.base_indices[target]),target_z)
    if density.inverted[source]:old=inversion([old])[0]
    if density.inverted[target]:new=inversion([new])[0]
    totals,_,logs=density.evaluate([old,new]);weights=np.log(density.weights)
    old_selected=logs[0,source]-weights[source];new_selected=logs[1,target]-weights[target]
    source_log=logs[0,source]-totals[0];reverse_log=logs[1,target]-totals[1]
    label=reverse_log+weights[source]-source_log-weights[target]
    auxiliary=.5*(noise@noise-inverse_noise@inverse_noise)
    proposal=dict(branch='involution',kernel='frozen-posterior',source_law='posterior',
        moving_index=0,anchor_index=1,correlation=correlation,
        trace=dict(source=source,target=target,noise=noise.tolist()),
        step=dict(pose=new,source_latent=z.tolist(),target_latent=target_z.tolist(),
            inverse_trace=dict(source=target,target=source,noise=inverse_noise.tolist()),
            log_extended_jacobian=jn-jo,log_auxiliary_ratio=auxiliary,log_correction=jn-jo+auxiliary),
        source_component_index=int(density.base_indices[source]),target_component_index=int(density.base_indices[target]),
        source_inverted=bool(density.inverted[source]),target_inverted=bool(density.inverted[target]),
        selected_source_log_density=old_selected,selected_target_log_density=new_selected,
        full_old_gaussian_log_density=totals[0],full_new_gaussian_log_density=totals[1],
        source_log_probability=source_log,inverse_source_log_probability=reverse_log,
        label_log_reverse_forward=label,expanded_log_reverse_forward=jn-jo+auxiliary+label,
        log_reverse_forward=totals[0]-totals[1])
    identity=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])
    return dict(kernel='frozen-posterior',accepted=True,proposal=proposal,old=old,new=new,
        anchor=identity,old_relative=old,new_relative=new)


class ReciprocalPoseDensityTests(unittest.TestCase):
    def test_legacy_and_all_false_preserve_arrays_densities_and_draws(self):
        base=fixture();legacy,flags=unwrap_proposal_model(base)
        self.assertIs(legacy,base);self.assertEqual(flags,[False]*3)
        plain,wrapped=Density(base),Density(envelope(base,[False]*3))
        a=plain.draw_component(np.random.default_rng(3),1,50)
        b=wrapped.draw_component(np.random.default_rng(3),1,50)
        self.assertEqual(a,b)
        for x,y in zip(plain.evaluate(a),wrapped.evaluate(a)):np.testing.assert_array_equal(x,y)
        del base['schema'];self.assertIs(unwrap_proposal_model(base)[0],base)

    def test_invalid_envelopes_fail_closed(self):
        good=envelope(fixture())
        invalid=[]
        for key in ('base_model','reciprocal_components'):
            value=copy.deepcopy(good);value.pop(key);invalid.append(value)
        for flags in (None,[True], [True,1,False], [True,None,False], 'yes'):
            value=copy.deepcopy(good);value['reciprocal_components']=flags;invalid.append(value)
        value=copy.deepcopy(good);value['weights']=[1.];invalid.append(value)
        value=copy.deepcopy(good);value['base_model']['coordinate_convention']='laboratory';invalid.append(value)
        value=copy.deepcopy(good);value['base_model']['reciprocal_components']=[False]*3;invalid.append(value)
        value=copy.deepcopy(good);value['base_model']=copy.deepcopy(good);invalid.append(value)
        value=copy.deepcopy(good);value['base_model'].pop('means');invalid.append(value)
        value=copy.deepcopy(good);value['schema']='wrong';invalid.append(value)
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):unwrap_proposal_model(value)

    def test_virtual_order_and_component_mass_are_explicit(self):
        density=Density(envelope(fixture(),[True,False,True]))
        np.testing.assert_array_equal(density.base_indices,[0,0,1,2,2])
        np.testing.assert_array_equal(density.inverted,[False,True,False,False,True])
        np.testing.assert_allclose(density.weights,[1/12,1/12,1/3,1/4,1/4],atol=0,rtol=1e-15)

    def test_exact_full_density_is_reciprocal_symmetric_when_all_selected(self):
        base=fixture();raw=Density(base);density=Density(envelope(base))
        values=raw.draw_component(np.random.default_rng(12),2,100)
        inverses=inversion(values)
        expected=np.logaddexp(raw.evaluate(values)[0],raw.evaluate(inverses)[0])-np.log(2)
        np.testing.assert_allclose(density.evaluate(values)[0],expected,atol=2e-12)
        np.testing.assert_allclose(density.evaluate(inverses)[0],expected,atol=2e-12)

    def test_partial_density_uses_only_selected_reciprocals(self):
        base=fixture();raw=Density(base);wrapped=Density(envelope(base,[True,False,True]))
        values=raw.draw_component(np.random.default_rng(8),1,50)
        _,_,forward=raw.evaluate(values);_,_,reverse=raw.evaluate(inversion(values))
        expected=np.column_stack([forward[:,0]-np.log(2),reverse[:,0]-np.log(2),forward[:,1],
            forward[:,2]-np.log(2),reverse[:,2]-np.log(2)])
        np.testing.assert_allclose(wrapped.evaluate(values)[2],expected,atol=5e-11)
        np.testing.assert_allclose(wrapped.evaluate(values)[0],logsumexp(expected,axis=1),atol=5e-11)

    def test_decode_encode_and_normalized_gaussian_measure_identity(self):
        model=envelope(fixture());audit=ChartAudit(model);density=Density(model);rng=np.random.default_rng(44)
        for label in range(len(density.weights)):
            for z in rng.normal(size=(5,6)):
                value,jac=audit.decode(label,z)
                np.testing.assert_allclose(audit.encode(label,value),z,atol=2e-13)
                _,_,logs=density.evaluate([value])
                normalized=logs[0,label]-np.log(density.weights[label])+jac
                self.assertAlmostEqual(normalized,-3*np.log(2*np.pi)-.5*z@z,places=11)
                self.assertAlmostEqual(audit.gaussian(label,value),logs[0,label]-np.log(density.weights[label]),places=11)

    def test_physical_jacobian_includes_nonlinear_inverse_without_extra_factor(self):
        audit=ChartAudit(envelope(fixture()));z=np.array([.4,-.3,.6,.2,.8,-.5]);eps=2e-5
        for label in (0,1,4,5):
            value,logj=audit.decode(label,z);_,base_r=audit.arrays(value);numeric=np.zeros((6,6))
            for column in range(6):
                shift=np.eye(6)[column]*eps
                hi,_=audit.decode(label,z+shift);lo,_=audit.decode(label,z-shift)
                hp,hr=audit.arrays(hi);lp,lr=audit.arrays(lo)
                numeric[:3,column]=(hp-lp)/(2*eps)
                numeric[3:,column]=(Rotation.from_matrix(hr@base_r.T).as_rotvec()-Rotation.from_matrix(lr@base_r.T).as_rotvec())/(2*eps)
            observed=abs(np.linalg.det(numeric))/(8*np.pi**2)
            self.assertAlmostEqual(observed/math.exp(logj),1.,places=8)

    def test_inverse_branch_draw_is_exact_pointwise_image_of_base_draw(self):
        base=fixture();wrapped=Density(envelope(base));plain=Density(base)
        for k in range(3):
            wanted=inversion(plain.draw_component(np.random.default_rng(16+k),k,30))
            actual=wrapped.draw_component(np.random.default_rng(16+k),2*k+1,30)
            wt,_,wr=arrays(wanted);at,_,ar=arrays(actual)
            np.testing.assert_allclose(at,wt,atol=2e-14);np.testing.assert_allclose(ar,wr,atol=2e-14)

    def test_virtual_draw_frequencies_and_latent_moments(self):
        model=envelope(fixture(),[True,False,True]);density=Density(model);audit=ChartAudit(model);rng=np.random.default_rng(108)
        labels=rng.choice(len(density.weights),size=6000,p=density.weights)
        counts=np.bincount(labels,minlength=len(density.weights))
        standardized=(counts-6000*density.weights)/np.sqrt(6000*density.weights*(1-density.weights))
        self.assertLess(np.max(np.abs(standardized)),4.)
        for k,count in enumerate(counts):
            samples=density.draw_component(rng,k,int(count))
            latent=np.asarray([audit.encode(k,value) for value in samples])
            self.assertLess(np.max(np.abs(latent.mean(axis=0))),.16)
            self.assertLess(np.max(np.abs(latent.var(axis=0)-1)),.22)

    def test_linearized_inverse_gaussian_is_not_the_exact_reciprocal_density(self):
        base=fixture(1);base['means']=[[0.]*6]
        t=np.asarray(base['anchors'][0]['position']);r=np.asarray(base['anchors'][0]['rotation']);ell=base['angular_length']
        x,y,z=t;cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        j=np.zeros((6,6));j[:3,:3]=j[3:,3:]=-r.T;j[:3,3:]=-(2/ell)*r.T@cross
        approximate=copy.deepcopy(base);approximate['anchors']=[dict(position=(-r.T@t).tolist(),rotation=r.T.tolist())]
        approximate['covariances']=[(j@np.asarray(base['covariances'][0])@j.T).tolist()]
        old,_=decode_base(base,0,np.array([1.,-.7,.5,2.,-1.5,1.8]));value=inversion([old])[0]
        exact=Density(envelope(base)).evaluate([value])[2][0,1]+np.log(2)
        approximate_log=Density(approximate).evaluate([value])[0][0]
        self.assertGreater(abs(exact-approximate_log),.1)

    def test_virtual_posterior_maps_and_full_responsibility_audits(self):
        model=envelope(fixture(),[True,False,True])
        for source,target in ((1,2),(2,4),(4,1)):
            for correlation in (0.,.9,-.4):
                record=posterior_record(model,source,target,correlation)
                result=audit_densities(model,[record],dict(frozen_posterior=dict(correlation=correlation)),ChartAudit(model))
                self.assertEqual(result['posterior_checks'],1);self.assertEqual(result['map_checks'],1)
                wrong=copy.deepcopy(record);wrong['proposal']['source_inverted']=not wrong['proposal']['source_inverted']
                with self.assertRaises(AssertionError):audit_densities(model,[wrong],dict(frozen_posterior=dict(correlation=correlation)),ChartAudit(model))

    def test_reciprocal_capture_validates_base_index_and_branch(self):
        model=envelope(fixture(),[True,False,True]);audit=ChartAudit(model)
        old,_=audit.decode(0,np.zeros(6));new,_=audit.decode(4,np.ones(6)*.1)
        cfg=dict(uniform_proposal_cube_lengths=[100.]*3,learned_uniform_weight=.1)
        oq=full_capture_log_density(audit.full_gaussian(old),old,[100.]*3,.1)
        nq=full_capture_log_density(audit.full_gaussian(new),new,[100.]*3,.1)
        record=dict(kernel='full-mixture-capture',old=old,new=new,old_relative=old,new_relative=new,accepted=False,
            proposal=dict(branch='learned',component_index=2,component_inverted=True,old_log_density=oq,new_log_density=nq,log_reverse_forward=oq-nq))
        self.assertEqual(audit_densities(model,[record],cfg,audit)['full_mixture_checks'],1)
        wrong=copy.deepcopy(record);wrong['proposal']['component_index']=1
        with self.assertRaises(AssertionError):audit_densities(model,[wrong],cfg,ChartAudit(model))


if __name__=='__main__':unittest.main()
