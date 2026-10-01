import copy
import unittest
from compare_singleton_retry_caps import audit_event, summarize

POSE=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])
NEW=dict(position=[1.,0.,0.],orientation=[1.,0.,0.,0.])

def event(valid=True, accepted=False, cap=3, raw_count=2):
    raw=[dict(trial=k+1,candidate=NEW,hard_valid=valid and k==raw_count-1,
              target_label=dict(kind='single',branch=k),target_latent=[float(k)]*6) for k in range(raw_count)]
    p=dict(branch='independent_conditioned',max_trials=cap,raw_trial_count=raw_count,raw_trials=raw,
           cap_exhausted=not valid,full_old_log_density=3.,full_new_log_density=1.,map_log_reverse_forward=2.,
           pool_forward_log_probability=-2.,pool_reverse_log_probability=-3.,anchor_log_reverse_forward=-1.,log_reverse_forward=1.)
    return dict(kind='cluster_event',members=[0],proposal=p,old_poses=[POSE],retained_poses=[NEW if accepted else POSE],
      proposed_poses=[NEW] if valid else None,accepted=accepted,hard_valid=valid,
      gate={'log_weight':-2.} if valid else None,log_acceptance=-1. if valid else None,
      proposed_gained_contacts=[[0,2]] if valid else [],proposed_lost_contacts=[[0,1]] if valid else [],
      diagnostic=dict(external_partners=[1,3],pool_direct_partners=[1,3]))

class Tests(unittest.TestCase):
    def test_valid_and_exhausted_keep_every_raw(self):
        valid=event(accepted=True);rejected=event();failed=event(False,cap=3,raw_count=3)
        s=summarize([valid,rejected,failed]);c=s['counts']
        self.assertEqual(c['attempted'],3);self.assertEqual(c['raw_trials'],7)
        self.assertEqual(c['raw_hard_failures'],5);self.assertEqual(c['raw_first_valid'],2)
        self.assertEqual(c['cap_exhaustions'],1);self.assertEqual(c['contact_changes'],1)
        self.assertEqual(s['rates']['first_valid_per_raw'],2/7)
        self.assertNotIn('source_norm',s)
    def test_no_retry_after_hard_valid(self):
        r=event();r['proposal']['raw_trials'][0]['hard_valid']=True
        with self.assertRaisesRegex(ValueError,'Retried'):audit_event(r)
    def test_missing_raw_and_wrong_endpoint_rejected(self):
        r=event();r['proposal']['raw_trial_count']=1
        with self.assertRaisesRegex(ValueError,'allocation'):audit_event(r)
        r=event();r['proposed_poses']=[POSE]
        with self.assertRaisesRegex(ValueError,'differs'):audit_event(r)
    def test_rejected_pose_and_double_pool_are_detected(self):
        r=event();r['retained_poses']=[NEW]
        with self.assertRaisesRegex(ValueError,'Rejected'):audit_event(r)
        r=event();r['proposal']['log_reverse_forward']=0.
        with self.assertRaisesRegex(ValueError,'twice'):audit_event(r)
    def test_post_valid_numerical_null_is_not_another_raw_retry(self):
        r=event();r['proposed_poses']=None;r['hard_valid']=False;r['gate']=None;r['log_acceptance']=None
        r['proposal']['null_reason']='nonfinite density';s=summarize([r])
        self.assertEqual(s['counts']['raw_first_valid'],1)
        self.assertEqual(s['counts']['post_valid_nulls'],1)
        self.assertEqual(s['counts']['learned_valid_candidates'],0)
    def test_uniform_branch_has_no_anchor_factor(self):
        r=event();r['proposal']=dict(branch='uniform',log_reverse_forward=0.)
        r['gate']['log_weight']=-1.;audit_event(r)
        r['proposal']['anchor_pool']=[1]
        with self.assertRaisesRegex(ValueError,'Unused'):audit_event(r)
    def test_local_channel_has_own_denominator_and_no_unused_pool(self):
        r=event(accepted=True);r['proposal']=dict(branch='local_rigid',log_reverse_forward=0.)
        r['gate']['log_weight']=-1.;s=summarize([r]);c=s['counts']
        self.assertEqual(c['local'],1);self.assertEqual(c['learned'],0)
        self.assertEqual(c['measured_valid_candidates'],1)
        self.assertEqual(s['rates']['measured_valid_per_channel_event'],1.)
        self.assertIsNone(s['rates']['raw_per_independent_event'])
        self.assertEqual(s['strata']['measured_valid']['distributions']['map']['median'],0.)
        r['proposal']['anchor_pool']=[1]
        with self.assertRaisesRegex(ValueError,'Unused'):audit_event(r)
if __name__=='__main__':unittest.main()
