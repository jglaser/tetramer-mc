#!/usr/bin/env python3
"""Independent controls for frozen native AB covariance proposal fitting."""
import copy
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_native_confirmation_atlas import (
    AtomUnionAudit, Density, ELL, chart_log_density, choose_family, coordinates,
    cross_validate, geometric_floor, laboratory_model, model_from_fit, native_q,
    relative_poses, to_world, weighted_fit,
)


def pose(t,rotation=None):
    rotation=Rotation.identity() if rotation is None else rotation
    return {'position':list(t),'orientation':rotation.as_quat()[[3,0,1,2]].tolist()}


class NativeConfirmationFitTests(unittest.TestCase):
    def test_original_weights_are_pooled_and_translation_rotation_coupled(self):
        x=np.array([[0.,0.,0.,0.,0.,0.],[1.,0.,0.,2.,0.,0.]])
        floor=geometric_floor()
        fit=weighted_fit(x,np.log([1.,9.]),floor)
        np.testing.assert_allclose(fit['mean'],[.9,0,0,1.8,0,0],atol=1e-14)
        self.assertAlmostEqual(fit['raw_covariance'][0,3],.18)
        np.testing.assert_allclose(fit['covariance']-fit['raw_covariance'],floor,atol=1e-15)
        self.assertAlmostEqual(fit['normalized_weight_ESS'],100/82)
        # Equal-population normalization would instead return .5 and 1.
        self.assertNotAlmostEqual(fit['mean'][0],.5)

    def test_floor_uses_declared_physical_scales_even_for_rank_zero_data(self):
        floor=geometric_floor()
        np.testing.assert_allclose(np.diag(floor)[:3],.05**2)
        np.testing.assert_allclose(np.diag(floor)[3:],(ELL*math.tan(math.radians(.1)/2))**2)
        fit=weighted_fit(np.ones((3,6)),np.log([.1,.2,.7]),floor)
        np.testing.assert_allclose(fit['covariance'],floor,atol=1e-30)
        self.assertTrue(np.all(fit['eigenvalues']>0))
        for bad in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(AssertionError):geometric_floor(translation_std=bad)

    def test_left_cayley_roundtrip_and_haar_density_under_rotated_world_frame(self):
        rng=np.random.default_rng(7621)
        anchor={'position':[.3,-.2,1.1],'rotation':Rotation.from_rotvec([.2,.7,-.4]).as_matrix().tolist()}
        b=rng.normal(size=(6,6));cov=.003*(b@b.T)+geometric_floor()
        fit={'mean':np.array([.1,-.2,.05,.3,-.1,.2]),'covariance':cov}
        model=model_from_fit(fit,anchor,'fixture','broad_tail')
        poses=Density(model).draw_component(rng,0,64)
        x=coordinates(poses,anchor)
        c=x[:,3:]/ELL
        delta=Rotation.from_quat(np.column_stack((c,np.ones(len(c))))).as_matrix()
        reconstructed=delta@np.asarray(anchor['rotation'])
        actual=Rotation.from_quat(np.array([p['orientation'] for p in poses])[:,[1,2,3,0]]).as_matrix()
        np.testing.assert_allclose(reconstructed,actual,atol=8e-16)
        wrong=np.asarray(anchor['rotation'])@delta
        self.assertGreater(float(np.max(np.abs(wrong-actual))),1e-4)
        manual=chart_log_density(x,fit,'broad_tail')
        np.testing.assert_allclose(Density(model).evaluate(poses)[0],manual,atol=2e-11)
        fixed=pose([2.,-4.,1.],Rotation.from_rotvec([-.6,.1,.9]))
        world=to_world(poses,fixed)
        back=relative_poses(world,fixed)
        np.testing.assert_allclose(coordinates(back,anchor),x,atol=2e-14)
        np.testing.assert_allclose(Density(laboratory_model(model,fixed)).evaluate(world)[0],manual,atol=2e-11)
        wide=copy.deepcopy(model);wide['covariances']=(4*np.asarray(model['covariances'])).tolist()
        np.testing.assert_allclose(np.array(wide['covariances']),4*np.array(model['covariances']))
        np.testing.assert_array_equal(model['covariances'][0],cov)

    def test_lopo_excludes_held_population_without_equalizing_training_masses(self):
        rng=np.random.default_rng(765)
        x=rng.normal(size=(32,6));groups=np.repeat(np.arange(8),4)
        logw=np.arange(32)/3
        floor=geometric_floor()
        result=cross_validate(x,logw,groups,floor)
        for fold in result['folds']:
            k=fold['heldout_population'];wanted=weighted_fit(x[groups!=k],logw[groups!=k],floor)
            self.assertNotIn(k,fold['training_populations'])
            np.testing.assert_allclose(fold['training_mean'],wanted['mean'])
            self.assertEqual(fold['training_nonzero'],28)
            self.assertGreaterEqual(fold['broad_tail_minus_single_nat'],math.log(.8)-1e-9)

    def test_selection_cannot_be_driven_by_one_outlier_fold(self):
        self.assertEqual(choose_family([10]+[-.1]*7,np.ones(8)/8)['selected_family'],'single')
        self.assertEqual(choose_family([20]+[.01]*7,np.ones(8)/8)['selected_family'],'single')
        self.assertEqual(choose_family([.2]*8,np.ones(8)/8)['selected_family'],'broad_tail')
        with self.assertRaises(AssertionError):choose_family([-1.]+[.2]*7,np.ones(8)/8)

    def test_both_neighbor_atom_unions_match_bruteforce_with_unequal_radii(self):
        shape={'atoms':[{'center':[0.,0.,0.],'radius':.6},
                        {'center':[.2,.3,.4],'radius':.35}]}
        fixed=[pose([10.,0.,0.]),pose([.8,0.,0.],Rotation.from_rotvec([.5,-.3,.2]))]
        moving=pose([0.,0.,0.],Rotation.from_rotvec([-.2,.1,.3]))
        result=AtomUnionAudit(shape,fixed).gaps(moving)
        def centers(p):
            r=Rotation.from_quat(np.array(p['orientation'])[[1,2,3,0]]).as_matrix()
            return np.array([a['center'] for a in shape['atoms']])@r.T+p['position']
        radii=np.array([a['radius'] for a in shape['atoms']])
        brute=[np.min(np.linalg.norm(centers(moving)[:,None,:]-centers(f)[None,:,:],axis=2)-radii[:,None]-radii[None,:]) for f in fixed]
        np.testing.assert_allclose(result,brute,atol=1e-14)
        self.assertGreater(result[0],0)
        self.assertLess(result[1],0)

    def test_native_metric_is_original_max_member_in_rotated_offcenter_frame(self):
        ref=pose([3.,-2.,4.],Rotation.from_rotvec([.3,-.6,.2]))
        test=pose(np.array(ref['position'])+[.3,0,0],Rotation.from_rotvec([.3,-.6,.2]))
        metadata={'native_poses':[ref],'rigid_members':[pose([2,1,0]),pose([2,-1,3])],
                  'member_error_scale':2.,'angle_error_scale_deg':15.}
        np.testing.assert_allclose(native_q([ref,test],metadata),[0.,.15],atol=2e-15)


if __name__=='__main__':unittest.main()
