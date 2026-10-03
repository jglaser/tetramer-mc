import copy
import math
import unittest

from analyze_factorized_dimer_physical import audit_decision, audit_gate, seed


def pose(x):
    return dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.])


def fixture():
    plan = dict(master_seed=71,candidate_ledger={'sha256':'0'*64},activity=.0275,**{'lambda':1.76},
        limits=dict(raw_per_leg=100,retained_per_leg=100,raw_per_outer=200,retained_per_outer=200))
    state = [pose(0.),pose(4.),pose(8.)]
    old = state[:2]
    proposed = [pose(1.),pose(5.)]
    candidate = dict(root=proposed[0],child=proposed[1],diagnostics=dict(full_old_log_density=-10.,
        full_new_log_density=-9.,log_reverse_forward=-1.,selection_log_reverse_forward=0.,log_tree_coordinate_jacobian=0.))
    cached = dict(index=0,atlas_index=0,case_index=0,attempt=0,method='factorized',
        case=dict(root=0,child=1,anchor=2),old=old,anchor_pose=state[2],proposal_status='candidate',
        candidate=candidate,proposal_cpu_seconds=.2,contact_diagnostic_cpu_seconds=.01)
    first = dict(gained=3,lost=1,raw_points=10,retained_points=4,retained_cells=3,created_cells=5,
                 envelope_volume=2.,log_weight=2*math.log1p(plan['activity']/plan['lambda']))
    second = dict(gained=0,lost=1,raw_points=7,retained_points=1,retained_cells=2,created_cells=3,
                  envelope_volume=1.,log_weight=-math.log1p(plan['activity']/plan['lambda']))
    aggregate = {k:first[k]+second[k] for k in first}
    gate = dict(order='first_then_second',ordered_members=[0,1],intermediate_selected=[proposed[0],old[1]],
                legs=[first,second],aggregate=aggregate)
    feasible = dict(internal_core_overlap=False,spectator_core_collisions=[[],[]],wall_valid=[True,True],internal_exclusion_contact=True)
    row = dict(index=0,cached=cached,source_state_sha256='source',source_selected=old,proposed_selected=proposed,
        gate_seed=seed(plan,cached,'gate'),mh_seed=seed(plan,cached,'mh'),log_uniform=-2.,accepted=True,
        gate_failure=None,physical_replay_cpu_seconds=.7,status='physical_decision',source_feasibility=feasible,
        endpoint_feasibility=copy.deepcopy(feasible),q_correction=-1.,gate=gate,log_ratio=-1.+aggregate['log_weight'],
        gate_rng_after_fingerprint=[1,2,3,4],gate_cpu_seconds=.5,retained_selected=proposed,
        retained_state=proposed+[state[2]])
    return row,cached,state,'source',plan


class DecisionAudit(unittest.TestCase):
    def test_accept_reject_and_second_order(self):
        args=fixture()
        result=audit_decision(*args)
        self.assertEqual(result['raw_points'],17)
        self.assertGreater(result['acceptance_probability'],0)
        row,cached,state,_,_=args
        row['log_uniform']=-.1
        row['accepted']=False
        row['retained_selected']=cached['old']
        row['retained_state']=state
        row['gate']['order']='second_then_first'
        row['gate']['ordered_members']=[1,0]
        row['gate']['intermediate_selected']=[state[0],row['proposed_selected'][1]]
        audit_decision(*args)

    def test_negative_infinity_and_null(self):
        args=fixture()
        row,cached,state,_,_=args
        d=cached['candidate']['diagnostics']
        d['full_old_log_density']=d['log_reverse_forward']='-inf'
        row['q_correction']=row['log_ratio']='-inf'
        row['accepted']=False
        row['retained_state']=state
        row['retained_selected']=cached['old']
        self.assertEqual(audit_decision(*args)['acceptance_probability'],0)
        cached['candidate']=None
        cached['proposal_status']='cap_exhausted'
        row['status']='proposal_null'
        row['gate']=row['log_ratio']=row['proposed_selected']=None
        del row['q_correction'],row['gate_cpu_seconds']
        self.assertEqual(audit_decision(*args)['raw_points'],0)

    def test_tampered_counts_intermediate_mh_and_state(self):
        mutations=[
            lambda r:r['gate']['legs'][0].update(gained=8),
            lambda r:r['gate']['aggregate'].update(raw_points=18),
            lambda r:r['gate']['aggregate'].update(log_weight=1.),
            lambda r:r['gate'].update(intermediate_selected=r['proposed_selected']),
            lambda r:r['gate'].update(ordered_members=[1,0]),
            lambda r:r.update(accepted=False),
            lambda r:r.update(log_ratio=0.),
            lambda r:r['retained_state'].__setitem__(2,pose(12.)),
            lambda r:r.update(source_state_sha256='wrong'),
            lambda r:r.update(mh_seed=r['gate_seed']),
            lambda r:r.update(log_uniform=0.),
            lambda r:r.update(status='fatal'),
            lambda r:r.update(gate_failure={'reason':'budget'}),
            lambda r:r.update(gate_cpu_seconds=2.),
        ]
        for mutate in mutations:
            args=fixture()
            mutate(args[0])
            with self.assertRaises(ValueError):audit_decision(*args)

    def test_zero_volume_and_budget(self):
        *_,plan=fixture()
        gate=dict(gained=0,lost=0,raw_points=0,retained_points=0,retained_cells=0,created_cells=0,envelope_volume=0.,log_weight=0.)
        audit_gate(gate,plan)
        for key,value in [('raw_points',1),('gained',-1),('log_weight',math.nan)]:
            broken=dict(gate,**{key:value})
            with self.assertRaises(ValueError):audit_gate(broken,plan)
        gate.update(envelope_volume=1.,raw_points=101)
        with self.assertRaises(ValueError):audit_gate(gate,plan)

    def test_full_density_and_geometry_guard(self):
        for where,key,value in [('diagnostics','full_new_log_density',-8.),
                                 ('diagnostics','selection_log_reverse_forward',1.),
                                 ('endpoint_feasibility','wall_valid',[False,True])]:
            args=fixture()
            target=args[1]['candidate']['diagnostics'] if where=='diagnostics' else args[0][where]
            target[key]=value
            with self.assertRaises(ValueError):audit_decision(*args)


if __name__=='__main__':unittest.main()
