"""Finite target identity, atom-union contact, and unconditional partition controls."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_mobile_threshold_reference import (
    ExclusionContact,PRIMARY,SEEDS,SAMPLES,partition_masks,read_weights,summarize_regions,
    validate_reference,validate_classifier_target,validate_jobs,sha)


def pose(t=(0.,0.,0.),q=(1.,0.,0.,0.)):return dict(position=list(t),orientation=list(q))
def logs(values):
    with np.errstate(divide='ignore'):return np.log(np.asarray(values,float))
def masks(valid,contact,native,anchors,q=None,rho=None,triangles=None):
    n=len(valid)
    return partition_masks(valid,contact,native,anchors,[False]*n if triangles is None else triangles,
        [2.]*n if q is None else q,[40.]*n if rho is None else rho)


def fixture():
    cfg=dict(shape='source-shape',fixed_poses=[pose(),pose([0.,1.,0.])],capture_center=[0.,0.,0.],capture_radius=170.,
        reservoir_density=.035,depletant_radius=1.5,metadata=dict(original='metric'))
    base=dict(schema='weighted-pose-mixture-v1',coordinate_convention='anchor-body-relative',angular_length=10.,shape_sha256='shape',
        weights=[1/147]*147,anchors=[dict(position=[3.,0.,0.],rotation=np.eye(3).tolist())for _ in range(147)],
        means=[[0.]*6 for _ in range(147)],covariances=[(np.eye(6)*.01).tolist()for _ in range(147)])
    chart={k:copy.deepcopy(base[k])for k in ('schema','coordinate_convention','angular_length','shape_sha256')}
    chart.update(weights=[1.],**{k:[copy.deepcopy(base[k][146])]for k in ('anchors','means','covariances')})
    region=dict(fixed_neighbor=cfg['fixed_poses'][1],physical_fixed_neighbors=cfg['fixed_poses'],capture_center=cfg['capture_center'],capture_radius=170.,
        shape_sha256='shape',activity=.035,depletant_radius=1.5,physical_metric=cfg['metadata'],gaussian_chart=chart,
        minimum_mahalanobis_radius=0.,mahalanobis_radius=4.,minimum_original_q=0.,minimum_original_q_inclusive=True)
    model=dict(schema='reciprocal-pose-mixture-v1',base_model=base,reciprocal_components=[True]*147)
    return cfg,region,model


class ThresholdReferenceTests(unittest.TestCase):
    def test_exhaustive_partition_native_precedence_and_invalid_zeros(self):
        m=masks([True]*4+[False],[True,True,False,False,True],[True,False,False,True,True],[2,0,0,1,1])
        np.testing.assert_array_equal(sum(m[k].astype(int)for k in PRIMARY),[1,1,1,1,0])
        self.assertEqual(np.flatnonzero(m['registered_native_entry']).tolist(),[0,3])
        self.assertEqual(np.flatnonzero(m['contact_no_native_entry']).tolist(),[1])
        self.assertEqual(np.flatnonzero(m['unbound_no_native_entry']).tolist(),[2])
        self.assertTrue(m['native_entry_unbound'][3])

    def test_historical_q_and_r32_are_not_native_labels(self):
        m=masks([True]*4,[True]*4,[False,True,True,False],[0,1,2,0],q=[0.,1.,2.,2.],rho=[1.,1.,32.,np.nextafter(32.,np.inf)])
        self.assertTrue(m['original_q_le_1'][0]);self.assertFalse(m['registered_native_entry'][0])
        self.assertEqual(np.flatnonzero(m['alternative_native_r32']).tolist(),[2])
        self.assertTrue(m['registered_native_entry'][1]);self.assertFalse(m['alternative_native_r32'][1])

    def test_exclusion_union_boundary_is_strict_and_either_anchor_counts(self):
        shape=dict(atoms=[dict(center=[0.,0.,0.],radius=1.)])
        contact=ExclusionContact(shape,[pose(),pose([20.,0.,0.])],1.5)
        self.assertFalse(contact.classify(pose([5.,0.,0.]))['exclusion_contact'])
        self.assertTrue(contact.classify(pose([np.nextafter(5.,0.),0.,0.]))['exclusion_contact'])
        r=contact.classify(pose([16.,0.,0.]));self.assertFalse(r['anchors'][0]['exclusion_contact']);self.assertTrue(r['anchors'][1]['exclusion_contact'])
        with self.assertRaisesRegex(ValueError,'hard-valid'):contact.classify(pose([1.,0.,0.]))

    def test_variable_radius_geometry_matches_exhaustive_pair_distances(self):
        # A farther center with larger radius can determine surface contact.
        shape=dict(atoms=[dict(center=[0.,0.,0.],radius=.1),dict(center=[0.,4.,0.],radius=2.),dict(center=[2.,1.,0.],radius=.4)])
        fixed=[pose(),pose([12.,-3.,1.])];g=ExclusionContact(shape,fixed,.7)
        moving=pose([7.,1.,.5],Rotation.from_euler('z',37,degrees=True).as_quat()[[3,0,1,2]])
        result=g.classify(moving,require_hard_valid=False)
        for i,item in enumerate(result['anchors']):
            gaps=np.linalg.norm(g.placed(moving)[:,None,:]-g.fixed[i][None,:,:],axis=2)-g.radii[:,None]-g.radii[None,:]
            self.assertAlmostEqual(item['minimum_surface_gap_A'],float(gaps.min()),places=12)
            self.assertEqual(item['exclusion_contact'],bool(np.any(gaps<1.4)))
        turn=Rotation.from_euler('xyz',[22,-41,13],degrees=True);shift=np.array([5.,-7.,11.])
        def transform(p):
            r=turn*Rotation.from_quat(np.asarray(p['orientation'])[[1,2,3,0]])
            return pose(turn.apply(p['position'])+shift,r.as_quat()[[3,0,1,2]])
        other=ExclusionContact(shape,[transform(p)for p in fixed],.7).classify(transform(moving),require_hard_valid=False)
        np.testing.assert_allclose([v['minimum_surface_gap_A']for v in result['anchors']],[v['minimum_surface_gap_A']for v in other['anchors']],atol=1e-12)

    def test_no_q_filter_and_saved_cloud_mean_with_unconditional_zero(self):
        row=dict(draw=0,pose=pose(),hard_valid=True,region_valid=True,capture_valid=True,latent_radius=0.,q=0.,
            log_hard_weight=2.,log_importance_weight=2.+math.log(3),clouds=[dict(log_weight=math.log(2)),dict(log_weight=math.log(4))])
        badcore=dict(row,draw=1,hard_valid=False,log_hard_weight=None,log_importance_weight=None,clouds=[])
        got=read_weights([row,badcore],2,dict(mahalanobis_radius=4.))
        self.assertTrue(np.isneginf(got['z'][1]));self.assertAlmostEqual(got['z'][0],2.+math.log(3))
        changed=copy.deepcopy(row);changed['region_valid']=False
        with self.assertRaisesRegex(ValueError,'q filter'):read_weights([changed,badcore],2,dict(mahalanobis_radius=4.))
        changed=copy.deepcopy(badcore);changed['log_hard_weight']=0.
        with self.assertRaisesRegex(ValueError,'zero'):read_weights([row,changed],2,dict(mahalanobis_radius=4.))
        with self.assertRaisesRegex(ValueError,'missing/repeated'):read_weights([row],2,dict(mahalanobis_radius=4.))

    def test_partition_statistics_keep_fixed_N_paired_cancellation_and_cloud_noise(self):
        m=masks([True,True,True,False],[True,True,False,False],[True,False,False,False],[1,0,0,0])
        h=np.array([1.,2.,3.,0.]);z=h*3
        pops=[dict(id=f'r{i:02d}',seed=SEEDS[i],z=logs(z),h=logs(h),pairs=logs(np.column_stack([h*2,h*4])),masks=m)for i in range(4)]
        estimates,cov,_=summarize_regions(pops);total=estimates['total']
        self.assertEqual(total['row_uncertainty']['draws'],16)
        self.assertAlmostEqual(math.exp(total['row_uncertainty']['log_Qz']),18/4)
        self.assertAlmostEqual(total['row_uncertainty']['log_enhancement_SE'],0.,places=14)
        self.assertAlmostEqual(sum(math.exp(estimates[k]['row_uncertainty']['log_Qz'])for k in PRIMARY),18/4)
        self.assertGreater(total['paired_cloud_noise']['relative_variance_mean_cloud'],0)
        names=cov['row']['regions'];self.assertLess(cov['row']['covariance_relative'][names.index(PRIMARY[0]+':Qz')][names.index(PRIMARY[1]+':Qz')],0)
        self.assertEqual(total['population_count'],4)

    def test_exact_B_chart_and_target_binding_reject_covariance_q_and_physics_changes(self):
        cfg,region,model=fixture();relocated=copy.deepcopy(cfg);relocated['shape']='archived-shape'
        validate_reference(relocated,region,'shape',cfg,model)
        for key,value in [('fixed_neighbor',cfg['fixed_poses'][0]),('minimum_original_q',1.),('minimum_original_q_inclusive',False),('capture_radius',169.),('activity',.04)]:
            changed=copy.deepcopy(region);changed[key]=value
            with self.assertRaises(ValueError):validate_reference(relocated,changed,'shape',cfg,model)
        changed=copy.deepcopy(region);changed['maximum_original_q']=1.
        with self.assertRaisesRegex(ValueError,'filter'):validate_reference(relocated,changed,'shape',cfg,model)
        changed=copy.deepcopy(region);changed['gaussian_chart']['covariances'][0][0][0]*=2
        with self.assertRaisesRegex(ValueError,'unchanged ordinary'):validate_reference(relocated,changed,'shape',cfg,model)
        changed=copy.deepcopy(relocated);changed['depletant_radius']=1.6
        with self.assertRaisesRegex(ValueError,'Config differs'):validate_reference(changed,region,'shape',cfg,model)

    def test_classifier_bound_to_exact_config_and_shape(self):
        cfg,_,_=fixture()
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);(root/'inputs').mkdir();path=root/'inputs/physical-config.json';path.write_text(json.dumps(cfg))
            definition=dict(shape_sha256='shape',fixed_poses=cfg['fixed_poses'],physical_config_sha256=sha(path))
            validate_classifier_target(cfg,'shape',definition,root/'definition.json')
            changed=copy.deepcopy(cfg);changed['reservoir_density']=.04
            with self.assertRaisesRegex(ValueError,'config/bath'):validate_classifier_target(changed,'shape',definition,root/'definition.json')
            with self.assertRaisesRegex(ValueError,'scaffold/shape'):validate_classifier_target(cfg,'other',definition,root/'definition.json')
            path.write_text(json.dumps(dict(cfg,shape='different source bytes')))
            with self.assertRaisesRegex(ValueError,'source binding'):validate_classifier_target(cfg,'shape',definition,root/'definition.json')

    def test_missing_failed_or_reused_seed_population_rejected(self):
        jobs=[dict(id=f'r{i:02d}',seed=SEEDS[i],samples=SAMPLES)for i in range(4)]
        terminal=[dict(j,status='complete',returncode=0)for j in jobs];validate_jobs(jobs,terminal)
        with self.assertRaises(ValueError):validate_jobs(jobs[:-1],terminal)
        changed=copy.deepcopy(jobs);changed[1]['seed']=changed[0]['seed']
        with self.assertRaisesRegex(ValueError,'seed'):validate_jobs(changed,terminal)
        changed=copy.deepcopy(terminal);changed[2]['returncode']=1
        with self.assertRaisesRegex(ValueError,'not complete'):validate_jobs(jobs,changed)

if __name__=='__main__':unittest.main()
