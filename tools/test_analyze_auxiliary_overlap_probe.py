"""Deterministic guided-ledger controls; no protein clouds or simulation draws."""
import copy
import hashlib
import math
import unittest
from unittest.mock import patch
import numpy as np
import analyze_auxiliary_overlap_probe as a
import test_analyze_factorized_dimer_probe as old
from test_analyze_capped_dimer_probe import pose


def fixture(m=1,points=None,threshold=1):
    h,o,c,source,anchor,out,contacts,s=old.fixture()
    points=np.asarray([[0.,0.,0.],[.5,0.,0.],[.75,0.,0.]] if points is None else points,dtype=float).reshape((-1,3))
    counter=a.CountOracle(o.centers,o.radii+o.rd)
    def frame_counts(frame,members,raw):
        values=dict(relative_count=counter.relative(points,raw),recovered_count=counter.relative(points,frame['recovered_edges'][1]),world_count=counter.world(points,members),reconstructed_world_count=counter.world(points,frame['reconstructed_members']))
        assert len(set(values.values()))==1
        frame['guidance']=values;return values['relative_count']
    kold=frame_counts(out['source_frame'],source,out['source_frame']['recovered_edges'][1])
    attempt=out['attempts'][0];knew=frame_counts(attempt['frame'],attempt['proposed'],attempt['internal_draws'][0]['draw']['proposed_relative_pose'])
    attempt['internal_draws'][0]['guidance_count']=knew
    out['guidance']=dict(point_count=len(points),m=m,old_count=kold,threshold=threshold,integer_draws=[threshold]*m,new_count=knew,
        aux_log_correction=m*(math.log1p(kold)-math.log1p(knew)),count_queries=9,point_tests=9*len(points))
    return [h,o,counter,points,c,source,anchor,out,contacts,s,m]


def run(f):
    checks=a.Checks();result=a.audit_guided(*f,checks,'toy');return result,checks


class GuidedAuditTests(unittest.TestCase):
    def test_complete_frame_density_and_zero_count_case(self):
        for f in [fixture(),fixture(m=4),fixture(points=[[-.75,0,0]],threshold=0),fixture(points=[],threshold=0)]:
            result,checks=run(f);self.assertEqual(checks.failures,[]);self.assertEqual(result['candidate']['complete_log_correction'],result['candidate']['log_reverse_forward'])
            self.assertEqual(result['count_queries'],9)

    def test_closed_membership_is_not_strict_core_contact(self):
        counter=a.CountOracle([[0,0,0]],[1.])
        self.assertEqual(counter.members([[1,0,0],[np.nextafter(1.,2.),0,0],[0,0,0]]).tolist(),[True,False,True])
        self.assertEqual(counter.relative(np.array([[0.,0.,0.]]),pose(1.)),1)

    def test_raw_cloud_transform_root_thinning_and_tamper(self):
        counter=a.CountOracle([[0.,0.,0.]],[1.]);u=np.array([[.5,.5,.5],[0.,.5,.5],[.99,.99,.99]],dtype='<f8');points=counter.low+counter.width*u
        keep=np.flatnonzero(counter.members(points));meta=dict(raw_count=3,low=counter.low.tolist(),high=counter.high.tolist(),width=counter.width.tolist(),kept_indices=keep.tolist(),retained_count=len(keep),raw_uniform_sha256=hashlib.sha256(u.tobytes()).hexdigest(),transformed_raw_sha256=a.point_hash(points),retained_points_sha256=a.point_hash(points[keep]))
        checks=a.Checks();got=a.audit_cloud(meta,u.tobytes(),counter,checks,'raw',3);self.assertEqual(checks.failures,[]);np.testing.assert_array_equal(got,points[keep])
        for kind in ['omit','bounds','hash','retained','uniform','truncated']:
            m=copy.deepcopy(meta);raw=u.tobytes()
            if kind=='omit':m['kept_indices']=[]
            elif kind=='bounds':m['low'][0]-=1e-12
            elif kind=='hash':m['transformed_raw_sha256']='bad'
            elif kind=='retained':m['retained_points_sha256']='bad'
            elif kind=='uniform':v=u.copy();v[0,0]=1.;raw=v.tobytes();m['raw_uniform_sha256']=hashlib.sha256(raw).hexdigest()
            else:raw=raw[:-1]
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.audit_cloud(m,raw,counter,a.Checks(),'bad',3)

    def test_threshold_rejection_is_retained_and_changes_first_success(self):
        f=fixture(threshold=2);attempt=f[7]['attempts'][0]
        rejected=copy.deepcopy(attempt['internal_draws'][0]);old.change_uniform(rejected,pose(1.8));rejected['feasibility']=a.internal_geometry(f[1],pose(1.8));rejected['guidance_count']=0
        self.assertTrue(a.internal_ok(rejected['feasibility']))
        attempt['internal_draws'].insert(0,rejected);attempt['internal_draws'][1]['index']=2
        f[7]['guidance'].update(count_queries=10,point_tests=10*len(f[3]))
        result,checks=run(f);self.assertEqual(checks.failures,[]);self.assertEqual(result['threshold_failures'],1);self.assertEqual(result['internal_draws'],2)
        bad=copy.deepcopy(f);bad[7]['attempts'][0]['internal_draws'][0]['guidance_count']=2
        with self.assertRaisesRegex(ValueError,'guidance count'):run(bad)
        bad=copy.deepcopy(f);bad[7]['guidance']['threshold']=0;bad[7]['guidance']['integer_draws']=[0]
        with self.assertRaisesRegex(ValueError,'Hidden retry'):run(bad)

    def test_unmeasured_count_is_not_zero_and_no_hidden_retry(self):
        f=fixture();attempt=f[7]['attempts'][0];rejected=copy.deepcopy(attempt['internal_draws'][0]);old.change_uniform(rejected,pose(8.));rejected['feasibility']=a.internal_geometry(f[1],pose(8.));rejected.pop('guidance_count')
        attempt['internal_draws'].insert(0,rejected);attempt['internal_draws'][1]['index']=2
        result,checks=run(f);self.assertEqual(checks.failures,[]);self.assertEqual(result['count_queries'],9)
        rejected['guidance_count']=0
        with self.assertRaisesRegex(ValueError,'Measured guidance'):run(f)
        f=fixture();f[7]['attempts'][0]['internal_draws']*=2
        with self.assertRaisesRegex(ValueError,'Hidden retry'):run(f)

    def test_internal_exhaustion_and_full_denominator(self):
        f=fixture(threshold=2);attempt=f[7]['attempts'][0];record=copy.deepcopy(attempt['internal_draws'][0]);old.change_uniform(record,pose(1.8));record['feasibility']=a.internal_geometry(f[1],pose(1.8));record['guidance_count']=0
        attempt.update(status='internal_cap_exhausted',internal_draws=[dict(copy.deepcopy(record),index=i+1) for i in range(32)],proposed=None,final_feasibility=None,frame=None)
        f[7].update(status='cap_exhausted',candidate=None);f[8][:]=[None];f[7]['guidance'].update(new_count=None,aux_log_correction=None,count_queries=36,point_tests=36*len(f[3]))
        result,checks=run(f);self.assertEqual(checks.failures,[]);self.assertEqual(result['threshold_failures'],32);self.assertIsNone(result['candidate'])
        attempt['internal_draws'].pop()
        with self.assertRaisesRegex(ValueError,'Premature'):run(f)

    def test_count_frame_auxiliary_full_density_and_counter_corruption(self):
        for kind in ['frame','source','threshold','integer','count','point','aux','full']:
            f=fixture();out=f[7];d=out['guidance']
            if kind=='frame':out['attempts'][0]['frame']['guidance']['world_count']+=1
            elif kind=='source':d['old_count']+=1
            elif kind=='threshold':d['threshold']+=1
            elif kind=='integer':d['integer_draws'][0]=d['old_count']+1
            elif kind=='count':d['count_queries']+=1
            elif kind=='point':d['point_tests']+=1
            elif kind=='aux':d['aux_log_correction']+=.1
            else:out['candidate']['diagnostics']['log_reverse_forward']+=.1
            with self.subTest(kind=kind):
                if kind in ['aux','full']:
                    _,checks=run(f);self.assertTrue(checks.failures)
                else:
                    with self.assertRaises(ValueError):run(f)

    def test_nonzero_auxiliary_ratio_is_not_folded_into_full_density(self):
        f=fixture(m=4);h,o,counter,points,case,source,anchor,out,contacts,s,m=f;a0=out['attempts'][0]
        root=a0['proposed'][0];relative=pose(1.5);child=a.compose_pose(root,relative)
        old.change_uniform(a0['internal_draws'][0],relative);a0['internal_draws'][0]['feasibility']=a.internal_geometry(o,relative)
        proposed=[root,child];geom=o.fingerprint([0,1],proposed);full=a.feasibility(o,case,proposed,geom)
        a0.update(proposed=proposed,final_feasibility=full,frame=old.frame(o,case,anchor,proposed));contacts[:]=[geom['contacts']]
        knew=counter.relative(points,relative);a0['internal_draws'][0]['guidance_count']=knew;a0['frame']['guidance']={k:knew for k in ['relative_count','recovered_count','world_count','reconstructed_world_count']}
        old_edges=[a.relative_pose(anchor,source[0]),a.relative_pose(source[0],source[1])];new_edges=[a.relative_pose(anchor,root),relative]
        od=[h.edge_density(p,.5,160.,kind='map') for p in old_edges];nd=[h.edge_density(p,.5,160.,kind='map') for p in new_edges]
        of=sum(v['log_full'] for v in od);nf=sum(v['log_full'] for v in nd)
        out['candidate'].update(root=root,child=child,diagnostics=dict(old_edges=od,new_edges=nd,full_old_log_density=of,full_new_log_density=nf,log_reverse_forward=of-nf,log_tree_coordinate_jacobian=0.,selection_log_reverse_forward=0.))
        aux=4*math.log(4/3);out['guidance'].update(new_count=knew,aux_log_correction=aux)
        result,checks=run(f);self.assertEqual(checks.failures,[]);self.assertAlmostEqual(result['candidate']['complete_log_correction'],of-nf+aux)
        self.assertGreater(aux,0.)

    def test_common_prefix_and_nested_threshold_checks(self):
        f1=fixture();f4=fixture(m=4);base=old.fixture()[5]
        row=lambda out:dict(outcome=out,rng_after_fingerprint=[1,2,3,4])
        b=row(base);rows=dict(m1=row(f1[7]),m4=row(f4[7]));checks=a.Checks();a.common_prefixes(b,rows,checks,'paired');self.assertEqual(checks.failures,[])
        for kind in ['raw','threshold','rng']:
            r=copy.deepcopy(rows)
            if kind=='raw':r['m4']['outcome']['attempts'][0]['internal_draws'][0]['draw']['trace']['branch_uniform']=.4
            elif kind=='threshold':r['m4']['outcome']['guidance']['integer_draws'][0]=0
            else:r['m4']['rng_after_fingerprint'][0]=0
            with self.subTest(kind=kind),self.assertRaises(ValueError):a.common_prefixes(b,r,a.Checks(),'bad')

    def test_seed_domains_are_unique_and_old_seed_reused(self):
        seeds=[a.seed(6100300101,x,y,z,k) for x in range(3) for y in range(8) for z in range(32) for k in ['cloud','threshold','execution_order']]
        seeds += [a.old_seed(6100203101,x,y,z,'factorized') for x in range(3) for y in range(8) for z in range(32)]
        self.assertEqual(len(set(seeds)),3072);self.assertEqual(len({tuple(a.method_order(x,y,z)) for x in range(3) for y in range(8) for z in range(32)}),2)


if __name__=='__main__':unittest.main()
