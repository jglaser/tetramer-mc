import copy
import hashlib
import json
import math
import unittest
from prepare_context_overlap_panel import SALT,canonical_id,rank_digest,select_panel,bound_cost


def records():
    out=[]
    for b,n in enumerate((9,8,7,6)):
        for j in range(n):
            out.append(dict(arm='full' if j%2 else 'diagonal',stream=b,ordinal=j,
                is_A_T=True,physical_valid=True,region='A_patch_complete',mahalanobis_squared={'full':[1,8,16,32][b],'diagonal':2.}))
    out.append(dict(arm='full',stream=0,ordinal=100,is_A_T=False,physical_valid=False,region='hard_invalid'))
    return out

class PanelTests(unittest.TestCase):
    def test_exact_hash_and_input_order_invariance(self):
        rows=records();a,p=select_panel(rows,(9,8,7,6,0));b,q=select_panel(list(reversed(rows)),(9,8,7,6,0))
        self.assertEqual(a,b);self.assertEqual(p,q);self.assertEqual(len(a),24)
        r=rows[0];expected=hashlib.sha256((SALT+json.dumps({k:r[k] for k in ('arm','stream','ordinal')},sort_keys=True,separators=(',',':'))).encode()).hexdigest()
        self.assertEqual(rank_digest(r),expected)
    def test_all_tail_and_finite_bank_probabilities(self):
        selected,proof=select_panel(records(),(9,8,7,6,0))
        self.assertEqual([r['inclusion_probability'] for r in proof],[6/9,6/8,6/7,1.])
        self.assertEqual({canonical_id(r['record']) for r in selected if r['bin_index']==3},
                         {canonical_id(r) for r in records() if r.get('mahalanobis_squared',{}).get('full')==32})
    def test_no_weight_based_selection_or_geometry(self):
        before,_=select_panel(records(),(9,8,7,6,0));changed=records()
        for j,r in enumerate(changed):r['log_q_balanced']=1000*(-1)**j;r['arbitrary_future_overlap']=j*j
        after,_=select_panel(changed,(9,8,7,6,0))
        self.assertEqual([canonical_id(r['record']) for r in before],[canonical_id(r['record']) for r in after])
    def test_missing_duplicate_and_indicator_corruption_rejected(self):
        with self.assertRaises(ValueError):select_panel(records()[:-2],(9,8,7,6,0))
        with self.assertRaises(ValueError):select_panel(records()+[records()[0]],(9,8,7,6,0))
        bad=records();bad[0]['physical_valid']=False
        with self.assertRaises(ValueError):select_panel(bad,(9,8,7,6,0))
    def test_poisson_cost_and_variance_bound_zero_and_positive(self):
        zero=bound_cost(2.,0.);self.assertEqual(zero['expected_complete_allocation_raw_points'],0.)
        self.assertEqual(zero['two_cloud_relative_variance_upper_bound'],0.)
        value=bound_cost(1.,3.,z=.5,lam=2.)
        self.assertEqual(value['expected_complete_allocation_raw_points'],12.)
        self.assertAlmostEqual(value['two_cloud_relative_variance_upper_bound'],math.expm1(.375)/2)
        self.assertEqual(value['log_physical_weight_bounds'],[.5,2.])
        self.assertEqual(value['two_cloud_score_variance_upper'],.1875)
    def test_invalid_identity_and_negative_volume_rejected(self):
        with self.assertRaises(ValueError):canonical_id(dict(arm='full',stream=0,ordinal=-1))
        with self.assertRaises(ValueError):bound_cost(0.,-1.)

if __name__=='__main__':unittest.main()
