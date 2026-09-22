"""Every bank slot remains, with geometric widths and exact reciprocal density."""
import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import prepare_native_blind_memory_proposal as preparation

class BankPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoints=[preparation.read(preparation.SOURCE/f'runs/free-r{i:02d}-rj-memory-on/checkpoint.json') for i in range(4)]
        cls.shape=preparation.read(preparation.SHAPE)

    def test_all_sixty_four_slots_keep_order_and_equal_weight(self):
        model,order=preparation.construct(self.checkpoints,self.shape)
        self.assertEqual(model['weights'],[1/64.]*64)
        self.assertEqual(order,[dict(replicate=i,slot=j) for i in range(4) for j in range(16)])
        for i,anchor in enumerate(model['anchors']):
            p=self.checkpoints[i//16]['contact_memory_state']['poses'][i%16]
            self.assertEqual(anchor['position'],p['position'])
            q=np.array(p['orientation'])[[1,2,3,0]]
            np.testing.assert_allclose(anchor['rotation'],Rotation.from_quat(q).as_matrix(),atol=1e-14)
        self.assertTrue(np.all(np.linalg.eigvalsh(model['covariances'])>0))
        np.testing.assert_allclose(np.asarray(model['covariances'])[:,:3,:3],np.broadcast_to(np.eye(3),(64,3,3)))

    def test_production_poses_and_duplicate_bank_centers_do_not_select_components(self):
        original,_=preparation.construct(self.checkpoints,self.shape)
        changed=copy.deepcopy(self.checkpoints)
        for c in changed:c['poses']=[];c['metadata']={'native':True,'density_score':1e300}
        same,_=preparation.construct(changed,self.shape);self.assertEqual(original,same)
        changed[0]['contact_memory_state']['poses'][0]=copy.deepcopy(changed[0]['contact_memory_state']['poses'][1])
        duplicate,_=preparation.construct(changed,self.shape)
        self.assertEqual(len(duplicate['weights']),64);self.assertEqual(duplicate['anchors'][0],duplicate['anchors'][1])
        self.assertEqual(duplicate['weights'],original['weights'])

    def test_missing_bank_slot_or_invalid_pose_fails_instead_of_filtering(self):
        for mode in ('bank','slot','pose','sweep','shape'):
            changed=copy.deepcopy(self.checkpoints)
            if mode=='bank':changed.pop()
            elif mode=='slot':changed[1]['contact_memory_state']['poses'].pop()
            elif mode=='pose':changed[2]['contact_memory_state']['poses'][0]['orientation']=[0.]*4
            elif mode=='sweep':changed[3]['completed_sweeps']=999
            else:changed[0]['shape_sha256']='wrong'
            with self.subTest(mode=mode),self.assertRaises(ValueError):preparation.construct(changed,self.shape)

    def test_reciprocal_envelope_does_not_change_or_refit_bank(self):
        model,_=preparation.construct(self.checkpoints,self.shape)
        wrapped=preparation.reciprocal_envelope(model)
        self.assertEqual(wrapped['base_model'],model);self.assertEqual(wrapped['reciprocal_components'],[True]*64)
        value=preparation.density_preflight(model,wrapped,[],seed=131200011)
        self.assertLess(value['maximum_log_density_identity_error'],1e-10)
        self.assertLess(value['maximum_reciprocal_symmetry_error'],1e-10)

if __name__=='__main__':unittest.main()
