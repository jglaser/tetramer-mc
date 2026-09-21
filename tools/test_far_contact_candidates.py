import copy
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_far_contact_candidates import select_rows,member_distances


def row(draw,q,weight,hard=True):
    return {'draw':draw,'q':q,'hard_valid':hard,'capture_valid':hard,'log_importance_weight':weight}

class FarCandidateTests(unittest.TestCase):
    def test_far_mask_excludes_huge_shoulder_weights(self):
        rows=[row(0,1.5,100.),row(1,4.999,99.),row(2,5.,1.),row(3,10.,2.),row(4,None,None,False)]
        self.assertEqual([r['draw'] for r in select_rows(rows,8)],[3,2])
    def test_deterministic_tie_break_and_original_rows_preserved(self):
        rows=[row(9,6.,2.),row(2,7.,2.),row(4,8.,1.)];saved=copy.deepcopy(rows)
        selected=select_rows(rows,1)
        self.assertIs(selected[0],rows[1]);self.assertEqual(rows,saved)
    def test_nonfinite_positive_weight_rejected(self):
        with self.assertRaisesRegex(ValueError,'finite'):select_rows([row(0,8.,math.inf)],1)
    def test_member_rms_is_symmetric_and_invariant_to_common_frame(self):
        members=np.array([[-1.,0.,0.],[1.,0.,0.],[0.,2.,0.]])
        poses=[{'position':[0.,0.,0.],'orientation':[1.,0.,0.,0.]},
               {'position':[3.,0.,0.],'orientation':[1.,0.,0.,0.]}]
        d=member_distances(poses,members);np.testing.assert_allclose(d,[[0.,3.],[3.,0.]])
        rot=Rotation.from_rotvec([.3,.1,-.2]);q=rot.as_quat()[[3,0,1,2]].tolist()
        shifted=[{'position':(rot.apply(p['position'])+[4.,-2.,1.]).tolist(),'orientation':q} for p in poses]
        np.testing.assert_allclose(member_distances(shifted,members),d,atol=1e-14)

if __name__=='__main__':unittest.main()
