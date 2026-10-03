"""Synthetic latent/trace fixtures only; no random proposal or cloud generation."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import analyze_auxiliary_overlap_sphere_control as a


def pose(position):
    return dict(position=list(position),orientation=[1.,0.,0.,0.])


def source_try(oracle,root_uniform):
    radial=(.7**3-.1**3)/(.9**3-.1**3)
    radius=(.1**3+radial*(.9**3-.1**3))**(1/3)
    root=pose([2*v-1 for v in root_uniform]);edge=pose([-radius,0.,0.])
    members=[a.compose(a.IDENTITY,root),a.compose(root,edge)]
    return dict(root_uniform=root_uniform,root_normals=[1.,0.,0.,0.],direction_normals=[-1.,0.,0.],
        radial_uniform=radial,relative_normals=[1.,0.,0.,0.],poses=members,feasibility=oracle.full(members),status='complete')


def frame(oracle,members,raw):
    edges=[a.relative(a.IDENTITY,members[0]),a.relative(*members)]
    reconstructed=[a.compose(a.IDENTITY,edges[0]),None];reconstructed[1]=a.compose(reconstructed[0],edges[1])
    return dict(recovered_edges=edges,reconstructed_members=reconstructed,
        reconstructed_feasibility=oracle.full(reconstructed),reconstructed_root=oracle.root(reconstructed[0]),
        internal_relative=oracle.internal(edges[1]),guidance=dict(relative_count=oracle.count_relative(raw),
            recovered_count=oracle.count_relative(edges[1]),world_count=oracle.count_world(members),
            reconstructed_world_count=oracle.count_world(reconstructed)))


def edge_record(old,new):
    return dict(old_relative_pose=old,branch='uniform',null_reason=None,proposed_relative_pose=new,
        trace=dict(branch_uniform=.2,translation_uniforms=[(v+1)/2 for v in new['position']],quaternion_normals=[1.,0.,0.,0.]))


def fixture(population=0,slot=0):
    oracle=a.SphereOracle(a.grid_points())
    trace=[source_try(oracle,[.5,.5,.5]),source_try(oracle,[.7,.5,.5])]
    old=trace[-1]['poses'];old_edges=[a.relative(a.IDENTITY,old[0]),a.relative(*old)]
    old_count=oracle.count_relative(old_edges[1]);root=pose([.6,.6,0.]);internal=pose([.2,0.,0.])
    new=[root,a.compose(root,internal)];new_count=oracle.count_relative(internal)
    density=dict(log_uniform=-3*math.log(2),log_full=-3*math.log(2),log_learned=-2.)
    diagnostics=dict(old_edges=[density,density],new_edges=[density,density],full_old_log_density=-6*math.log(2),
        full_new_log_density=-6*math.log(2),log_reverse_forward=0.,selection_log_reverse_forward=0.,log_tree_coordinate_jacobian=0.)
    candidate=dict(root=new[0],child=new[1],diagnostics=diagnostics)
    arms=[]
    for m in [1,4]:
        integers=[old_count//2] if m==1 else [old_count//2,old_count,0,old_count//3]
        auxiliary=m*(math.log1p(old_count)-math.log1p(new_count))
        root_record=dict(index=1,draw=edge_record(old_edges[0],root),world_pose=root,feasibility=oracle.root(root))
        internal_record=dict(index=1,draw=edge_record(old_edges[1],internal),feasibility=oracle.internal(internal),guidance_count=new_count)
        outcome=dict(status='candidate',caps=copy.deepcopy(a.CAPS),order='root_first',members=[0,1],anchor_label=2,
            old=old,source_feasibility=oracle.full(old),source_frame=frame(oracle,old,old_edges[1]),
            attempts=[dict(index=1,status='candidate',root_draws=[root_record],internal_draws=[internal_record],
                           proposed=new,final_feasibility=oracle.full(new),frame=frame(oracle,new,internal))],candidate=candidate,
            guidance=dict(point_count=len(oracle.points),m=m,old_count=old_count,threshold=max(integers),integer_draws=integers,
                          new_count=new_count,aux_log_correction=auxiliary,count_queries=9,point_tests=9*len(oracle.points)))
        arms.append(dict(m=m,outcome=outcome,log_uniform=-.001,complete_log_correction=auxiliary,accepted=False,retained=old,
            wrong_without_auxiliary=dict(accepted=True,retained=new) if m==4 else None,
            proposal_seed=a.seed(population,slot,'proposal'),auxiliary_seed=a.seed(population,slot,'auxiliary'),
            mh_seed=a.seed(population,slot,'mh'),proposal_cpu_seconds=.001,proposal_rng_after=18,auxiliary_rng_after=19))
    return oracle,dict(population=population,slot=slot,source_seed=a.seed(population,slot,'source'),source=old,
                      source_trace=trace,status='complete',arms=arms)


class SphereAudit(unittest.TestCase):
    def test_complete_latent_guided_and_wrong_endpoint_pipeline(self):
        oracle,row=fixture()
        result=a.audit_row(a.Checks(),oracle,row,0,0)
        self.assertEqual(result['source_tries'],2)
        self.assertEqual(len(result['arms']),2)
        for arm in result['arms']:
            self.assertTrue(arm['candidate']);self.assertFalse(arm['accepted'])
            self.assertLess(arm['auxiliary'],0.)
        self.assertEqual(result['arms'][1]['wrong_without_auxiliary']['paired_delta']['close_contact'],1.)
        # Change only the common MH variate: both corrected endpoints now move.
        for arm in row['arms']:
            arm['log_uniform']=-100.;arm['accepted']=True
            c=arm['outcome']['candidate'];arm['retained']=[c['root'],c['child']]
        a.audit_row(a.Checks(),oracle,row,0,0)

    def test_all_nulls_preserved_and_no_phantom_decision(self):
        oracle,row=fixture()
        for arm in row['arms']:
            o=arm['outcome'];old=o['old'];old_edge=a.relative(a.IDENTITY,old[0]);root=pose([0.,0.,0.])
            roots=[dict(index=i+1,draw=edge_record(old_edge,root),world_pose=root,feasibility=oracle.root(root)) for i in range(8)]
            o['status']='cap_exhausted';o['candidate']=None
            o['attempts']=[dict(index=1,status='root_cap_exhausted',root_draws=roots,internal_draws=[],proposed=None,final_feasibility=None,frame=None)]
            o['guidance'].update(new_count=None,aux_log_correction=None,count_queries=4,point_tests=4*len(oracle.points))
            arm.update(complete_log_correction=None,accepted=False,retained=old)
            if arm['m']==4:arm['wrong_without_auxiliary']=dict(accepted=False,retained=old)
        result=a.audit_row(a.Checks(),oracle,row,0,0)
        self.assertTrue(all(not item['candidate'] for item in result['arms']))
        self.assertTrue(all(item['root_draws']==8 for item in result['arms']))
        row['arms'][0]['complete_log_correction']=0.
        with self.assertRaises(ValueError):a.audit_row(a.Checks(),oracle,row,0,0)

    def test_tampered_source_threshold_correction_stopping_state_and_prefix(self):
        mutations=[
            lambda r:r['source_trace'][0]['root_uniform'].__setitem__(0,.8),
            lambda r:r['source_trace'].__setitem__(0,copy.deepcopy(r['source_trace'][-1])),
            lambda r:r['arms'][0]['outcome']['guidance'].update(threshold=0),
            lambda r:r['arms'][0]['outcome']['guidance'].update(aux_log_correction=0.),
            lambda r:r['arms'][0].update(complete_log_correction=0.),
            lambda r:r['arms'][0].update(accepted=True),
            lambda r:r['arms'][1]['wrong_without_auxiliary'].update(accepted=False),
            lambda r:r['arms'][1]['wrong_without_auxiliary'].update(retained=r['source']),
            lambda r:r['arms'][0]['outcome']['caps'].update(joint=2),
            lambda r:r['arms'][0]['outcome']['attempts'][0]['internal_draws'].append(copy.deepcopy(r['arms'][0]['outcome']['attempts'][0]['internal_draws'][0])),
            lambda r:r['arms'][0]['outcome']['source_frame']['guidance'].update(world_count=999),
            lambda r:r['arms'][0]['outcome']['guidance'].update(count_queries=10),
            lambda r:r['arms'][0].update(proposal_seed=0),
            lambda r:r.update(status='fatal'),
            lambda r:r['arms'].pop(),
        ]
        for mutate in mutations:
            oracle,row=fixture();mutate(row)
            with self.assertRaises(ValueError):a.audit_row(a.Checks(),oracle,row,0,0)

    def test_strict_body_predicates_and_closed_cloud_membership(self):
        oracle=a.SphereOracle([[.45,0.,0.],[0.,0.,0.]])
        self.assertEqual(oracle.count_relative(pose([.9,0.,0.])),1)
        self.assertEqual(oracle.count_relative(pose([.9+1e-8,0.,0.])),0)
        self.assertFalse(oracle.internal(pose([.1,0.,0.]))['internal_core_overlap'])
        self.assertFalse(oracle.internal(pose([.9,0.,0.]))['internal_exclusion_contact'])
        self.assertTrue(oracle.internal(pose([.1-1e-8,0.,0.]))['internal_core_overlap'])
        # Child may leave the root proposal cube: no artificial child wall.
        self.assertTrue(a.fully_feasible(oracle.full([pose([.9,.9,.9]),pose([1.1,.9,.9])])))
        with self.assertRaises(ValueError):a.SphereOracle([[.450001,0.,0.]])

    def test_eight_pooled_tests_and_separate_wrong_control(self):
        rows=[]
        for i in range(4):
            for slot in range(2):
                oracle,row=fixture(i,slot);rows.append(a.audit_row(a.Checks(),oracle,row,i,slot))
        summary=a.summarize(rows)
        self.assertEqual(len(summary['primary_tests']),8)
        self.assertTrue(summary['primary_family_passed'])
        self.assertTrue(summary['negative_control']['passed'])
        self.assertEqual(summary['negative_control']['mean'],1.)
        self.assertEqual(summary['primary_tests'][0]['p_normal'],1.)
        json.dumps(summary,allow_nan=False)
        for row in rows:
            row['arms'][0]['paired_delta']['close_contact']=1.
            row['arms'][1]['wrong_without_auxiliary']['paired_delta']['close_contact']=0.
        failed=a.summarize(rows)
        self.assertFalse(failed['primary_family_passed'])
        self.assertFalse(failed['negative_control']['passed'])

    def test_internal_geometry_exhaustion_skips_all_cloud_queries(self):
        oracle,row=fixture()
        for arm in row['arms']:
            o=arm['outcome'];attempt=o['attempts'][0];old_edge=a.relative(*o['old']);edge=pose([0.,0.,0.])
            attempt['internal_draws']=[dict(index=i+1,draw=edge_record(old_edge,edge),feasibility=oracle.internal(edge)) for i in range(8)]
            attempt.update(status='internal_cap_exhausted',proposed=None,final_feasibility=None,frame=None)
            o.update(status='cap_exhausted',candidate=None)
            o['guidance'].update(new_count=None,aux_log_correction=None,count_queries=4,point_tests=4*len(oracle.points))
            arm.update(complete_log_correction=None,accepted=False,retained=o['old'])
            if arm['m']==4:arm['wrong_without_auxiliary']=dict(accepted=False,retained=o['old'])
        result=a.audit_row(a.Checks(),oracle,row,0,0)
        self.assertTrue(all(item['internal_draws']==8 for item in result['arms']))
        row['arms'][0]['outcome']['attempts'][0]['internal_draws'][0]['guidance_count']=0
        with self.assertRaises(ValueError):a.audit_row(a.Checks(),oracle,row,0,0)

    def test_partial_campaign_and_changed_prelaunch_preserve_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'execution';run.mkdir()
            config=dict(schema='auxiliary-overlap-sphere-control-v1',master_seed=6100300201,populations=4,
                draws_per_population=4096,m=[1,4],core_radius=.05,exclusion_radius=.45,uniform_half_width=1.,
                activity=0.,caps=a.CAPS,order='root_first')
            def save(name,value): (run/name).write_text(json.dumps(value))
            save('config.json',config);save('source-bundle.json',{'files':{}});save('cloud.json',a.grid_points())
            (run/'example.rs').write_text('synthetic wiring fixture only')
            binary=root/'example';binary.write_bytes(b'not executable: test hash only')
            binding=dict(config_sha256=a.sha(run/'config.json'),source_sha256=a.sha(run/'example.rs'),
                source_bundle_sha256=a.sha(run/'source-bundle.json'),executable_sha256=a.sha(binary))
            save('binding.json',binding)
            lines=[fixture(0,slot)[1] for slot in range(2)]
            (run/'attempts.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in lines))
            save('terminal.json',dict(summary=dict(complete=True,result={}),attempts_sha256=a.sha(run/'attempts.jsonl')))
            files=[run/name for name in ('config.json','source-bundle.json','example.rs')]+[binary,Path(a.__file__).resolve()]
            prelaunch=root/'prelaunch.json';prelaunch.write_text(json.dumps({'files':{str(p):a.sha(p) for p in files}}))
            result=a.audit(run,prelaunch)
            self.assertFalse(result['passed']);self.assertFalse(result['complete'])
            self.assertEqual(len(result['rows']),2)
            self.assertIn('Lost independent sources',result['failures'][0])
            (run/'example.rs').write_text('changed')
            changed=a.audit(run,prelaunch)
            self.assertIn('Changed prelaunch file',changed['failures'][0])
            self.assertEqual(len(changed['rows']),0)

    def test_paired_moments_use_differences_and_sample_se(self):
        moments=a.Moments()
        for value in [-1.,0.,1.,1.]:moments.add(value)
        stat=moments.report()
        self.assertEqual(stat['mean'],.25)
        self.assertAlmostEqual(stat['se'],math.sqrt(.9166666666666666/4))
        require_fail=a.Moments()
        require_fail.add(1.)
        with self.assertRaises(ValueError):require_fail.report()


if __name__=='__main__':unittest.main()
