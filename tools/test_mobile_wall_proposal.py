import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_mobile_wall_proposal import (ALLOCATION, SCALES, build_model, chart_center,
    contact_candidates, geometric_maximin, member_embedding, recenter)
from diagnose_mobile_native_remainder import compose, pose
from prepare_smc_normalizer_atlas import Density


def fixture():
    single=dict(schema='weighted-pose-mixture-v1',coordinate_convention='anchor-body-relative',
        shape_sha256='test',angular_length=4.,anchors=[dict(position=[1.,2.,3.],rotation=np.eye(3).tolist())],
        means=[[.1,.2,.3,.01,.02,.03]],covariances=[np.diag([.2,.3,.4,.1,.12,.14]).tolist()],weights=[1.])
    base=copy.deepcopy(single)
    for key in ('anchors','means','covariances'):base[key]=base[key]*150
    base['weights']=[1/150]*150
    original=dict(schema='reciprocal-pose-mixture-v1',base_model=base,reciprocal_components=[True]*150)
    sites=[]
    for block,count in [('known_competitor',1),('diverse_unregistered_contacts',8),
        ('confirmed_native_core',1),('original_native',1),('outside_d170_native',3)]:
        sites.extend(dict(block=block,site_id=f'{block}{i}',model=copy.deepcopy(single),anchor_index=i%2)for i in range(count))
    return original,sites


class WallProposalTests(unittest.TestCase):
    def test_allocation_retains_original_and_expands_exact_virtual_branches(self):
        original,sites=fixture();model,blocks=build_model(original,sites)
        base=model['base_model'];self.assertEqual(len(base['weights']),178)
        self.assertEqual(model['reciprocal_components'],[True]*150+[False]*28)
        for key in ('anchors','means','covariances'):self.assertEqual(base[key][:150],original['base_model'][key])
        density=Density(model);self.assertEqual(len(density.weights),328)
        self.assertAlmostEqual(sum(density.weights),1.)
        for name,mass in ALLOCATION.items():
            actual=sum(b['learned_mass']for b in blocks if b['name']==name)
            self.assertAlmostEqual(actual,mass)
        for b in blocks[1:]:
            self.assertEqual(b['intended_anchor_learned_mass'],b['learned_mass']/2)
            self.assertEqual(b['incidental_anchor_learned_mass'],b['learned_mass']/2)

    def test_widths_copy_covariance_and_do_not_push_forward(self):
        original,sites=fixture();model,_=build_model(original,sites)
        for i,scale in enumerate(SCALES):
            np.testing.assert_array_equal(model['base_model']['covariances'][150+i],np.array(sites[0]['model']['covariances'][0])*scale**2)
        center=pose([8.,-2.,1.],Rotation.from_rotvec([.3,-.4,.7]).as_matrix())
        changed=recenter(sites[0]['model'],center)
        self.assertEqual(changed['covariances'],sites[0]['model']['covariances'])
        np.testing.assert_allclose(chart_center(changed)['position'],center['position'],atol=1e-14)
        self.assertEqual(changed['means'],[[0.]*6])

    def test_contact_filter_excludes_unbound_and_native_rows(self):
        rows=[];labels={}
        for n,(hard,contact,native)in enumerate([(True,True,False),(True,False,False),(True,True,True),(False,None,False)]):
            rows.append(dict(arm='a',population='r0',sample=dict(draw=n,hard_valid=hard,depletion_contact=contact)))
            labels[('a','r0',n)]=dict(native_any=native)
        self.assertEqual([v['sample']['draw']for v in contact_candidates(rows,labels)],[0])
        del labels[('a','r0',0)]
        with self.assertRaisesRegex(ValueError,'Missing'):contact_candidates(rows,labels)

    def test_selection_ignores_weights_and_is_common_isometry_invariant(self):
        members=np.array([[0,0,0],[1,0,0],[0,2,0],[0,0,3.]])
        initial=pose([0,0,0],np.eye(3))
        candidates=[dict(arm='a',population='r0',sample=dict(draw=i,pose=pose([i,0,0],Rotation.from_rotvec([0,.1*i,0]).as_matrix()),
            log_importance_weight=999 if i==2 else-999))for i in range(1,12)]
        selected=geometric_maximin(candidates,members,initial,4)
        moved=copy.deepcopy(candidates);common=pose([-7,2,8],Rotation.from_rotvec([.5,.1,-.7]).as_matrix())
        for value in moved:
            value['sample']['pose']=compose(common,value['sample']['pose']);value['sample']['log_importance_weight']*=100
        again=geometric_maximin(moved,members,compose(common,initial),4)
        self.assertEqual([v['sample']['draw']for v in selected],[v['sample']['draw']for v in again])
        np.testing.assert_allclose([v['selection_distance_A']for v in selected],[v['selection_distance_A']for v in again],atol=2e-14)

    def test_member_embedding_distance_is_member_rms(self):
        members=np.arange(12).reshape(4,3);a=pose([1,2,3],np.eye(3));b=pose([4,6,3],np.eye(3))
        self.assertAlmostEqual(np.linalg.norm(member_embedding(a,members)-member_embedding(b,members)),5.)

    def test_invalid_partial_model_design_rejected(self):
        original,sites=fixture();broken=copy.deepcopy(original);broken['reciprocal_components'][1]=False
        with self.assertRaisesRegex(ValueError,'150 atlas'):build_model(broken,sites)
        with self.assertRaisesRegex(ValueError,'Empty'):build_model(original,[s for s in sites if s['block']!='original_native'])
        sites[0]['anchor_index']=2
        with self.assertRaisesRegex(ValueError,'anchor'):build_model(original,sites)

    def test_geometric_selection_needs_distinct_centers(self):
        p=pose([0,0,0],np.eye(3));candidate=dict(arm='a',population='r0',sample=dict(draw=0,pose=p))
        with self.assertRaisesRegex(ValueError,'distinct'):geometric_maximin([candidate],np.eye(3),p,1)


if __name__=='__main__':unittest.main()
