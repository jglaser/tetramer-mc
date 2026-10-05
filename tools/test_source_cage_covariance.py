import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_source_cage_covariance import coordinates,covariance,decompose,retained_states,source_initial_matches


def pose(p, r):
    q=Rotation.from_matrix(r).as_quat()
    return dict(position=list(p),orientation=q[[3,0,1,2]].tolist())


class SavedCageTests(unittest.TestCase):
    def test_exact_source_and_noncommuting_common_frame(self):
        a=pose([2.,-1.,3.],Rotation.from_rotvec([.2,-.3,.1]).as_matrix())
        s=pose([4.,1.,2.],Rotation.from_rotvec([-.1,.4,.3]).as_matrix())
        x,m=coordinates([s],s,a,2.);np.testing.assert_allclose(x,0,atol=1e-14)
        f=Rotation.from_rotvec([.4,.1,-.2]).as_matrix();shift=np.array([-2.,1.,5.])
        def change(p):
            q=np.array(p['orientation']);return pose(f@p['position']+shift,f@Rotation.from_quat(q[[1,2,3,0]]).as_matrix())
        moved=pose([4.1,.9,2.2],Rotation.from_rotvec([-.11,.42,.29]).as_matrix())
        y,_=coordinates([moved],s,a,2.);z,_=coordinates([change(moved)],change(s),change(a),2.)
        np.testing.assert_allclose(y,z,rtol=0,atol=4e-14)

    def test_chart_matches_explicit_fixed_anchor_cayley(self):
        ra=Rotation.from_rotvec([.2,.3,-.1]).as_matrix();rs=Rotation.from_rotvec([-.3,.1,.2]).as_matrix()
        a=pose([1.,2.,3.],ra);s=pose([3.,1.,4.],ra@rs);u=np.array([.01,-.02,.03]);dc=Rotation.from_quat(np.r_[u,1.]/np.linalg.norm(np.r_[u,1.])).as_matrix()
        delta=np.array([.1,-.2,.3]);new=pose(np.array(s['position'])+ra@delta,ra@dc@rs)
        x,m=coordinates([new],s,a,5.);np.testing.assert_allclose(x[0],np.r_[delta,5*u],atol=2e-14)
        self.assertAlmostEqual(m['rotation_angle_degrees'][0],np.degrees(2*np.arctan(np.linalg.norm(u))))

    def test_seam_is_retained_as_undefined_not_dropped(self):
        origin=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]);half=dict(position=[1.,0.,0.],orientation=[0.,1.,0.,0.])
        x,m=coordinates([origin,half,origin],origin,origin,2.)
        self.assertEqual(len(x),3);self.assertEqual(m['failure_indices'],[1]);self.assertEqual(m['near_seam_count'],1)

    def test_residence_repeats_change_descriptive_covariance(self):
        x=np.zeros((4,6));x[-1,0]=4.;mean,c=covariance(x)
        self.assertEqual(mean[0],1.);self.assertEqual(c[0,0],3.)
        self.assertNotEqual(covariance(np.unique(x,axis=0))[1][0,0],c[0,0])

    def test_within_between_decomposition_preserves_state_weights(self):
        x=np.arange(18,dtype=float).reshape(3,6);y=np.arange(30,dtype=float).reshape(5,6)+12
        d=decompose([x,y],np.arange(1,7));np.testing.assert_allclose(np.array(d['within_stream'])+d['between_stream_means'],d['total'],atol=1e-12)
        self.assertEqual(d['weights'],[3/8,5/8]);self.assertLess(d['maximum_identity_residual'],1e-12)

    def test_source_initial_may_be_explicit_but_must_equal_certified_pose(self):
        source=dict(position=[1.,2.,3.],orientation=[1.,0.,0.,0.])
        self.assertTrue(source_initial_matches(None,source))
        self.assertTrue(source_initial_matches(json.loads(json.dumps(source)),source))
        altered=dict(source,position=[1.,2.,3.001])
        self.assertFalse(source_initial_matches(altered,source))

    def test_stream_inventory_keeps_rejections_and_detects_missing_or_changed_pose(self):
        source=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]);identity={'stream':0};events=[]
        def add(kind,**kw):events.append(dict(event_index=len(events),kind=kind,**kw))
        for kind in ('invocation_begun','prepared','fixed_context_audit'):add(kind)
        add('initial_state',pose=source,identity=identity)
        previous=source
        for cycle in (1,2):
            for slot in range(5):
                idx=(cycle-1)*5+slot;candidate=dict(position=[float(idx+1),0.,0.],orientation=source['orientation']);accepted=slot==0
                add('attempt_begun');add('attempt_decision');retained=candidate if accepted else previous
                add('attempt_complete',attempt_index=idx,cycle=cycle,slot=slot,identity=identity,global_=False,production=cycle>1,old_pose=previous,proposed_pose=candidate,retained_pose=retained,accepted=accepted,exclusion_contact_labels=[16,217])
                events[-1]['global']=events[-1].pop('global_');previous=retained
            add('cycle_complete',cycle=cycle,pose=previous)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'events.jsonl'
            def write():p.write_text(''.join(json.dumps(e)+'\n' for e in events))
            write();rows,audit=retained_states(p,identity,source,cycles=2,warmup=1)
            self.assertEqual((len(rows),audit['production_states'],audit['rejected_states']),(10,5,8));self.assertEqual(rows[0]['pose'],rows[4]['pose'])
            events[12]['retained_pose']=source;write()
            with self.assertRaises(ValueError):retained_states(p,identity,source,cycles=2,warmup=1)


if __name__=='__main__':unittest.main()
