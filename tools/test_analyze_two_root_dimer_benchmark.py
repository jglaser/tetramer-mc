"""Synthetic saved-record fixtures only; no geometry, sampling or live inputs."""
import copy
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import analyze_two_root_dimer_benchmark as a


def pose(x):return dict(position=[float(x),0.,0.],orientation=[1.,0.,0.,0.])


def gate(gained=0,lost=0):
    return dict(gained=gained,lost=lost,raw_points=gained+lost,retained_points=gained+lost,
                created_cells=0,retained_cells=0,envelope_volume=0.,log_weight=(gained-lost)*math.log1p(1/64))


def feasibility(contact=True):
    return dict(internal_core_overlap=False,spectator_core_collisions=[[],[]],wall_valid=[True,True],internal_exclusion_contact=contact)


def frame(f,count):
    return dict(reconstructed_feasibility=copy.deepcopy(f),
                reconstructed_root=dict(spectator_core_collisions=f['spectator_core_collisions'][0],wall_valid=f['wall_valid'][0]),
                internal_relative={k:f[k] for k in ('internal_core_overlap','internal_exclusion_contact')},
                guidance={k:count for k in ('relative_count','recovered_count','world_count','reconstructed_world_count')})


def fixture(members=(9,24),initialization='source'):
    c=dict(contexts=[dict(root=members[0],child=members[1],anchor=71)],two_root_policy=copy.deepcopy(a.POLICY),
           allocation=dict(warmup_blocks=1,production_blocks=3),source_frame={'synthetic':'source'},
           physical=dict(lambda_ratio=64.),factorized=dict(order='root_first',uniform_half_width=4.,uniform_probability=.5,
                                                          root_cap=2,internal_cap=2,joint_cap=1))
    job=dict(id=0,context_index=0,arm=a.ARM,initialization=initialization,stream=0)
    initial=[pose(1),pose(2)];selected=copy.deepcopy(initial);bank=dict(synthetic='bound cloud')
    rows=[dict(kind='initial',block=0,job=job,selected=copy.deepcopy(initial),conditional_target=True,
               sampler_cpu_seconds=0.,fixed_source=c['source_frame'],cloud=bank,
               root_order_contract=a.root_contract(list(members),71,bank))]
    counts=dict(local_attempted=0,local_accepted=0,dimer_attempted=0,dimer_accepted=0,dimer_self_loop=0)
    for block,root in enumerate((1,0,1,0),1):
        for attempt,slot in enumerate((0,1,0,1)):
            rows.append(dict(kind='local',block=block,attempt=attempt,member=members[slot],old=copy.deepcopy(selected[slot]),
                proposed=pose(999),accepted=False,status='hard_rejected',retained=copy.deepcopy(selected),sampler_cpu_seconds=float(len(rows))))
            counts['local_attempted']+=1
        order=[members[root],members[1-root]];old=[selected[root],selected[1-root]]
        g=dict(m=4,point_count=1,old_count=0,new_count=None,threshold=0,integer_draws=[0]*4,aux_log_correction=None)
        draw=dict(index=1,feasibility=dict(spectator_core_collisions=[],wall_valid=False),draw=dict(proposed_relative_pose=pose(1),null_reason=None))
        attempt=dict(index=1,status='root_cap_exhausted',root_draws=[copy.deepcopy(draw)]*2,internal_draws=[],proposed=None)
        for i,x in enumerate(attempt['root_draws'],1):attempt['root_draws'][i-1]=dict(x,index=i)
        p=dict(members=order,anchor_label=71,old=copy.deepcopy(old),caps=dict(root=2,internal=2,joint=1),order='root_first',
               guidance=g,candidate=None,status='cap_exhausted',attempts=[attempt],source_feasibility=feasibility(),source_frame=frame(feasibility(),0))
        row=dict(kind='factorized_dimer',block=block,members=order,anchor=71,old=copy.deepcopy(old),
                 canonical_members=list(members),selected_members=order,root_slot=root,root_order_probability=.5,
                 root_order_log_reverse_forward=0.,root_order_rng_role='two_root_m4/root_order',canonical_old=copy.deepcopy(selected),
                 proposal=p,status='proposal_self_loop',accepted=False,sampler_cpu_seconds=float(len(rows)))
        if block<=2:
            uniform=-3*math.log(8.);density=dict(log_uniform=uniform,log_learned=uniform,log_full=uniform)
            diagnostics=dict(old_edges=[density,density],new_edges=[density,density],full_old_log_density=2*uniform,
                             full_new_log_density=2*uniform,log_reverse_forward=0.,selection_log_reverse_forward=0.,log_tree_coordinate_jacobian=0.)
            new=[pose(10+block),pose(20+block)]
            g.update(new_count=1,aux_log_correction=-4*math.log(2.))
            p.update(status='candidate',candidate=dict(root=new[0],child=new[1],diagnostics=diagnostics))
            root_draw=copy.deepcopy(draw);root_draw['feasibility']['wall_valid']=True
            internal_draw=copy.deepcopy(draw);internal_draw.update(feasibility=dict(internal_core_overlap=False,internal_exclusion_contact=True),guidance_count=1)
            attempt.update(status='candidate',root_draws=[root_draw],internal_draws=[internal_draw],proposed=new,
                           final_feasibility=feasibility(),frame=frame(feasibility(),1))
            row.update(status='completed',proposed=new,canonical_proposed=[new[root],new[1-root]],
                       complete_log_correction=g['aux_log_correction'],bath=dict(legs=[gate(),gate()],aggregate=gate(),ordered_members=order),
                       log_acceptance_ratio=g['aux_log_correction'],log_u=-10. if block==1 else -.1,accepted=block==1)
            if row['accepted']:selected=copy.deepcopy(row['canonical_proposed'])
        elif block==4:
            p.update(status='source_outside_domain',attempts=[],source_feasibility=feasibility(False),source_frame=frame(feasibility(False),0))
            g.update(threshold=None,integer_draws=[])
        row['retained']=copy.deepcopy(selected);rows.append(row)
        counts['dimer_attempted']+=1;counts['dimer_accepted']+=row['accepted'];counts['dimer_self_loop']+=row['status']=='proposal_self_loop'
        rows.append(dict(kind='retained_block',block=block,production=block>1,selected=copy.deepcopy(selected),raw=0,retained=0,
                         counts=copy.deepcopy(counts),sampler_cpu_seconds=float(len(rows))))
    terminal=dict(complete=True,conditional_target=True,job=job,blocks=4,counts=counts,raw=0,retained=0,cpu_seconds=float(len(rows)))
    return c,job,initial,rows,terminal,bank


class TwoRootReplayTests(unittest.TestCase):
    def validate(self,fixture):
        c,j,s,rows,t,cloud=fixture
        with mock.patch.object(a.previous,'ConditionalObserver',side_effect=AssertionError('No geometry')):
            return a.validate_journal(rows,c,j,s,t,cloud)

    def test_all_residence_general_labels_and_unchanged_nested_records(self):
        for members in ((9,24),(24,9),(0,2)):
            f=fixture(members);before=copy.deepcopy(f);points=self.validate(f)
            self.assertEqual(f,before);self.assertEqual([r['block'] for r in points],list(range(5)))
            self.assertEqual(points[-1]['selected'],[pose(21),pose(11)])
            self.assertTrue(all(p['selected']==points[1]['selected'] for p in points[1:]))
            c,j,s,rows,_,bank=f
            mapped=list(a.canonical_rows(rows,c,j,s,bank))
            for raw,projected in zip(rows,mapped):
                if raw['kind']=='factorized_dimer':
                    self.assertEqual(projected['proposal'],raw['proposal'])
                    self.assertEqual(projected.get('bath'),raw.get('bath'))
                    self.assertEqual(projected['old'],raw['canonical_old'])

    def test_root_contract_labels_and_canonical_projection_tampering_rejected(self):
        for field,value in [('root_slot',True),('root_slot',2),('root_order_probability',.4),
                            ('root_order_log_reverse_forward',.1),('root_order_rng_role','m4/root_order'),
                            ('canonical_members',[24,9]),('selected_members',[9,24]),('members',[9,24]),
                            ('canonical_old',[pose(2),pose(1)]),('canonical_proposed',[pose(11),pose(21)])]:
            f=fixture();f[3][5][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.validate(f)
        for mode in ('policy','contract','cloud','source','selected_order','retained'):
            f=fixture()
            if mode=='policy':f[0]['two_root_policy']['root_probabilities']=[.4,.6]
            elif mode=='contract':f[3][0]['root_order_contract']['internal']['point_frame']='canonical_root_body'
            elif mode=='cloud':f[3][0]['cloud']={'wrong':1}
            elif mode=='source':f[3][0]['fixed_source']={'wrong':1}
            elif mode=='selected_order':f[3][5]['proposal']['members']=[9,24]
            else:f[3][5]['retained']=list(reversed(f[3][5]['retained']))
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.validate(f)

    def test_complete_density_auxiliary_bath_and_null_factors_checked(self):
        for mode in ('density','old_sum','ratio','aux','threshold','complete','bath','null_bath','null_candidate','incomplete_draw'):
            f=fixture();r=f[3][5];q=r['proposal']['candidate']['diagnostics']
            if mode=='density':q['old_edges'][0]['log_full']+=1
            elif mode=='old_sum':q['full_old_log_density']+=1
            elif mode=='ratio':q['log_reverse_forward']+=1
            elif mode=='aux':r['proposal']['guidance']['aux_log_correction']+=1
            elif mode=='threshold':r['proposal']['guidance']['integer_draws']=[1]*4
            elif mode=='complete':r['complete_log_correction']+=1
            elif mode=='bath':r['bath']['aggregate']['gained']=1
            elif mode=='null_bath':f[3][17]['bath']={}
            elif mode=='null_candidate':f[3][17]['proposal']['candidate']=q
            else:r['proposal']['attempts'][0]['root_draws'][0]['draw']['null_reason']='decode failure'
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.validate(f)

    def test_saved_stage_exhaustion_first_success_and_source_predicates(self):
        for mode in ('short_root','root_then_internal','successful_root_retried','source_contact','hard_source',
                     'short_internal','successful_internal_retried','candidate_final_invalid','source_frame','candidate_frame'):
            f=fixture();p=f[3][17]['proposal'];attempt=p['attempts'][0]
            if mode=='short_root':attempt['root_draws'].pop()
            elif mode=='root_then_internal':attempt['internal_draws']=[copy.deepcopy(f[3][5]['proposal']['attempts'][0]['internal_draws'][0])]
            elif mode=='successful_root_retried':attempt['root_draws'][0]['feasibility']['wall_valid']=True
            elif mode=='source_contact':
                f[3][23]['proposal'].update(source_feasibility=feasibility(),source_frame=frame(feasibility(),0))
            elif mode=='hard_source':p['source_feasibility']['internal_core_overlap']=True
            elif mode in ('short_internal','successful_internal_retried'):
                success=copy.deepcopy(f[3][5]['proposal']['attempts'][0])
                attempt.update(status='internal_cap_exhausted',root_draws=success['root_draws'],internal_draws=success['internal_draws'])
                if mode=='short_internal':
                    attempt['internal_draws'][0].update(feasibility=dict(internal_core_overlap=True,internal_exclusion_contact=True),guidance_count=None)
                else:attempt['internal_draws'].append(dict(copy.deepcopy(attempt['internal_draws'][0]),index=2))
            elif mode=='candidate_final_invalid':f[3][5]['proposal']['attempts'][0]['final_feasibility']['wall_valid'][1]=False
            elif mode=='source_frame':p['source_frame']['guidance']['world_count']=1
            else:f[3][5]['proposal']['attempts'][0]['frame']['guidance']['recovered_count']=0
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.validate(f)

    def test_valid_internal_exhaustion_and_final_rejection_keep_residence(self):
        for mode in ('internal_cap_exhausted','final_rejected'):
            f=fixture();attempt=f[3][17]['proposal']['attempts'][0]
            success=copy.deepcopy(f[3][5]['proposal']['attempts'][0]);attempt['root_draws']=success['root_draws'];attempt['status']=mode
            if mode=='internal_cap_exhausted':
                draw=success['internal_draws'][0];draw.update(feasibility=dict(internal_core_overlap=True,internal_exclusion_contact=True),guidance_count=None)
                attempt['internal_draws']=[copy.deepcopy(draw),dict(copy.deepcopy(draw),index=2)]
            else:
                world=feasibility();world['wall_valid'][1]=False
                attempt.update(internal_draws=success['internal_draws'],proposed=success['proposed'],
                               final_feasibility=world,frame=frame(world,1))
            points=self.validate(f);self.assertEqual(points[3]['selected'],points[2]['selected'])

    def test_per_root_factors_include_nulls_rejects_and_support_zeros(self):
        c,j,s,rows,t,cloud=fixture();report=a.new_root_diagnostics(c,j)
        # A valid always-reject candidate with zero reverse density is distinct
        # from a null proposal and from an omitted/nonfinite numeric score.
        row=rows[11];q=row['proposal']['candidate']['diagnostics']
        q['old_edges']=[dict(log_uniform='-inf',log_learned='-inf',log_full='-inf') for _ in range(2)]
        q['full_old_log_density']=q['log_reverse_forward']='-inf'
        row['complete_log_correction']=row['log_acceptance_ratio']='-inf'
        a.validate_journal(rows,c,j,s,t,cloud,report)
        self.assertEqual([report[str(i)]['attempts'] for i in (0,1)],[2,2])
        self.assertEqual([report[str(i)]['candidates'] for i in (0,1)],[1,1])
        self.assertEqual([report[str(i)]['accepted'] for i in (0,1)],[0,1])
        self.assertEqual(report['0']['null_status_counts'],{'source_outside_domain':1})
        self.assertEqual(report['1']['null_stage_counts'],{'root_cap_exhausted':1})
        self.assertEqual(report['0']['candidate_factors']['source_full_log_density']['negative_infinity_count'],1)
        self.assertEqual(report['0']['candidate_factors']['source_full_log_density']['finite_count'],0)
        self.assertIsNone(report['0']['candidate_factors']['source_full_log_density']['finite_mean'])
        self.assertEqual(report['0']['edge_support_zeros']['old_edges']['log_full'],2)
        self.assertEqual(report['0']['edge_support_zeros']['new_edges']['log_full'],0)
        self.assertAlmostEqual(report['1']['candidate_factors']['auxiliary_log_correction']['finite_mean'],-4*math.log(2.))

    def test_every_fatal_missing_extra_or_nonmonotone_attempt_is_rejected(self):
        for field in ('fatal_error','proposal_failure','bath_failure'):
            for index in (1,5,17):
                f=fixture();f[3][index][field]={'error':'synthetic'}
                with self.subTest(field=field,index=index),self.assertRaises(ValueError):self.validate(f)
        for mode in ('missing','extra','cpu','counts','terminal','status'):
            f=fixture()
            if mode=='missing':f[3].pop(8)
            elif mode=='extra':f[3].append(copy.deepcopy(f[3][-1]))
            elif mode=='cpu':f[3][8]['sampler_cpu_seconds']=-1
            elif mode=='counts':f[3][-1]['counts']['dimer_accepted']+=1
            elif mode=='terminal':f[4]['complete']=False
            else:f[3][5]['status']='in_progress'
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.validate(f)

    def test_continuation_chunks_and_both_initializations_have_identical_replay(self):
        for init in ('source','proposal_prepared'):
            f=fixture(initialization=init);expected=self.validate(f);split=list(f)
            split[3]=itertools.chain(iter(f[3][:13]),iter(f[3][13:]))
            self.assertEqual(self.validate(split),expected)

    def test_external_nonempty_exchanges_returns_through_empty_and_full_cpu(self):
        A=[[9,50]];B=[[24,70]];values=[A,A,B,[],A,A]
        trace=[dict(block=i,external_edges=x) for i,x in enumerate(values)]
        m=a.external_activity(trace,1,20.)
        self.assertEqual(m['production_samples'],4);self.assertEqual(m['nonempty_fraction'],.75)
        self.assertEqual((m['direct_nonempty_changes'],m['entering_nonempty'],m['leaving_nonempty']),(1,1,1))
        self.assertEqual(len(m['completed_nonempty_returns']),1)
        self.assertTrue(m['completed_nonempty_returns'][0]['passed_through_empty'])
        self.assertEqual(m['direct_nonempty_changes_per_full_CPU_second'],.05)
        self.assertEqual(m['completed_nonempty_returns_per_full_CPU_second'],.05)
        empty=a.external_activity([dict(block=i,external_edges=[]) for i in range(5)],1,20.)
        self.assertEqual(empty['direct_nonempty_changes'],0);self.assertEqual(empty['completed_nonempty_returns'],[])

    def test_incomplete_controller_refused_before_any_journal_or_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'execution').mkdir()
            for name in ('config.json','run-binding.json','analysis-plan.json','protocol.json','execution-plan.json',
                         'execution/claim.json','execution/status.json','execution/summary.json'):
                (root/name).write_text(json.dumps(dict(complete=False,passed=False,failure=None,active={'pid':1},
                    unstarted=[],completed=[],plan_sha256='0'*64)))
            with mock.patch.object(a.previous,'ConditionalObserver',side_effect=AssertionError('No geometry')), \
                 self.assertRaisesRegex(ValueError,'All eight sampler jobs'):
                a.bind_complete_inputs(root,{})


if __name__=='__main__':unittest.main()
