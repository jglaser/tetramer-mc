import unittest
from analyze_contact_efficiency import ContactObserver
from analyze_singleton_patches import changed_patches, aggregate

def pose(x):return dict(position=[x,0.,0.],orientation=[1.,0.,0.,0.])
class Tests(unittest.TestCase):
    def setUp(self):
        self.shape=dict(atoms=[dict(center=[-.3,0.,0.],radius=.1),dict(center=[.3,0.,0.],radius=.1)])
        self.config=dict(depletant_radius=.1,boundary=dict(kind='spherical',radius=20.),box_lengths=[50.]*3)
        self.observer=ContactObserver(self.shape,['left','right'],self.config,native=None)
    def test_patch_change_at_same_partner_matches_full_observer(self):
        old=[pose(0.),pose(.85),pose(9.)];new=pose(1.7)
        result=changed_patches(self.observer,old,0,new,[1],[1])
        full_old=set(map(tuple,self.observer.classify(old)['tokens']))
        full_new=set(map(tuple,self.observer.classify([new,*old[1:]])['tokens']))
        self.assertEqual(set(map(tuple,result['old_patch_tokens'])),full_old)
        self.assertEqual(set(map(tuple,result['new_patch_tokens'])),full_new)
        self.assertEqual(result['candidate_neighbors'],[1])
        self.assertTrue(result['same_partners']);self.assertTrue(result['patch_changed'])
        self.assertEqual(result['jaccard_distance'],1.)
        self.assertEqual(result['displacement_A'],1.7)
    def test_rejections_are_zero_in_all_event_denominator(self):
        r=changed_patches(self.observer,[pose(0.),pose(.85)],0,pose(1.7),[1],[1])
        s=aggregate([r],10,2.)
        self.assertEqual(s['counts']['rejected'],9)
        self.assertEqual(s['mean_jaccard_per_all_event'],.1)
        self.assertEqual(s['same_partner_patch_changes_per_all_event'],.1)
    def test_wrong_recorded_contact_and_missing_bound_partner_fail(self):
        with self.assertRaisesRegex(ValueError,'disagree'):
            changed_patches(self.observer,[pose(0.),pose(.85)],0,pose(1.7),[],[1])
        with self.assertRaisesRegex(ValueError,'omitted'):
            changed_patches(self.observer,[pose(0.),pose(.85),pose(9.)],0,pose(1.7),[1,2],[1])
if __name__=='__main__':unittest.main()
