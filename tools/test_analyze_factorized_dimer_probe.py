"""Deterministic ledger/geometry/density corruption checks, no physical draws."""
import copy
import unittest
import numpy as np
import analyze_factorized_dimer_probe as a
import test_analyze_capped_dimer_probe as prior
from test_analyze_capped_dimer_probe import pose


def frame(oracle,case,anchor,members):
    edges=[a.relative_pose(anchor,members[0]),a.relative_pose(members[0],members[1])]
    root=a.compose_pose(anchor,edges[0]);new=[root,a.compose_pose(root,edges[1])]
    return dict(recovered_edges=edges,reconstructed_members=new,
        reconstructed_feasibility=a.feasibility(oracle,case,new,oracle.fingerprint([case['root'],case['child']],new)),
        reconstructed_root=a.root_geometry(oracle,case,new[0]),internal_relative=a.internal_geometry(oracle,edges[1]))


def fixture():
    h,o,c,old,anchor,raw,g=prior.CappedAuditTests().fixture()
    new=[raw['draw']['candidate']['root'],raw['draw']['candidate']['child']]
    r=dict(index=1,draw=raw['draw']['edges'][0],world_pose=new[0],feasibility=a.root_geometry(o,c,new[0]))
    i=dict(index=1,draw=raw['draw']['edges'][1],feasibility=a.internal_geometry(o,raw['draw']['edges'][1]['proposed_relative_pose']))
    source=a.feasibility(o,c,old,o.fingerprint([0,1],old))
    attempt=dict(index=1,status='candidate',root_draws=[r],internal_draws=[i],proposed=new,final_feasibility=raw['feasibility'],frame=frame(o,c,anchor,new))
    out=dict(status='candidate',caps=a.CAPS.copy(),order='root_first',members=[0,1],anchor_label=2,old=old,
        source_feasibility=source,source_frame=frame(o,c,anchor,old),attempts=[attempt],candidate=raw['draw']['candidate'])
    return h,o,c,old,anchor,out,[g['contacts']],source


def run(f):
    checks=a.Checks();result=a.audit_factorized(*f,checks,'test');return result,checks


def change_uniform(record,relative):
    edge=record['draw'];edge['proposed_relative_pose']=relative
    edge['trace']['translation_uniforms']=((np.asarray(relative['position'])/160+1)/2).tolist()


class FactorizedAuditTests(unittest.TestCase):
    def test_frozen_rng_domains_and_execution_order(self):
        seeds=[a.stream_seed(6100203101,ai,ci,j,role) for ai in range(3) for ci in range(8) for j in range(32) for role in a.METHODS+['execution_order']]
        self.assertEqual(len(set(seeds)),2304)
        orders={tuple(a.method_order(6100203101,ai,ci,j)) for ai in range(3) for ci in range(8) for j in range(32)}
        self.assertEqual(orders,{tuple(a.METHODS),tuple(a.METHODS[::-1])})

    def test_full_generation_frame_and_density(self):
        result,checks=run(fixture());self.assertEqual(checks.failures,[])
        self.assertEqual((result['root_draws'],result['internal_draws'],result['assembled']),(1,1,1))
        self.assertTrue(result['candidate']['feasible'])
        f=list(fixture());f[5]['candidate']['diagnostics']['log_reverse_forward']+=.1
        _,checks=run(f);self.assertTrue(any(v['check']=='joint log_reverse_forward' for v in checks.failures))

    def test_geometry_and_frame_corruptions_are_fatal(self):
        for key in ['source_frame','frame','raw']:
            f=list(fixture());out=f[5]
            if key=='source_frame':out['source_frame']['internal_relative']['internal_core_overlap']=True
            elif key=='frame':out['attempts'][0]['frame']['reconstructed_root']['wall_valid']=False
            else:out['attempts'][0]['root_draws'][0]['feasibility']['wall_valid']=False
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Fatal geometry/frame mismatch'):run(f)

    def test_first_success_hidden_retry_and_missing_draw(self):
        f=list(fixture());r=f[5]['attempts'][0]['root_draws'];r.append(copy.deepcopy(r[0]));r[-1]['index']=2
        with self.assertRaisesRegex(ValueError,'Hidden retries'):run(f)
        f=list(fixture());f[5]['attempts'][0]['root_draws']=[]
        with self.assertRaisesRegex(ValueError,'Premature edge'):run(f)
        f=list(fixture());f[5]['attempts']*=2
        with self.assertRaisesRegex(ValueError,'Unallocated joint retry'):run(f)

    def test_root_exhaustion_short_circuits_internal(self):
        f=list(fixture());h,o,c,old,anchor,out,contacts,source=f;attempt=out['attempts'][0]
        record=copy.deepcopy(attempt['root_draws'][0]);change_uniform(record,pose(0))
        record['world_pose']=anchor;record['feasibility']=a.root_geometry(o,c,anchor)
        self.assertFalse(a.root_ok(record['feasibility']))
        attempt.update(status='root_cap_exhausted',root_draws=[dict(copy.deepcopy(record),index=j+1) for j in range(32)],internal_draws=[],proposed=None,final_feasibility=None,frame=None)
        out.update(status='cap_exhausted',candidate=None);contacts[:]=[None]
        result,checks=run(f);self.assertEqual(checks.failures,[]);self.assertEqual(result['root_draws'],32)
        attempt['internal_draws']=[{}]
        with self.assertRaisesRegex(ValueError,'Internal stage ran'):run(f)

    def test_internal_exhaustion_and_source_outside_retained(self):
        f=list(fixture());h,o,c,old,anchor,out,contacts,source=f;attempt=out['attempts'][0]
        record=copy.deepcopy(attempt['internal_draws'][0]);change_uniform(record,pose(8));record['feasibility']=a.internal_geometry(o,pose(8))
        attempt.update(status='internal_cap_exhausted',internal_draws=[dict(copy.deepcopy(record),index=j+1) for j in range(32)],proposed=None,final_feasibility=None,frame=None)
        out.update(status='cap_exhausted',candidate=None);contacts[:]=[None]
        result,checks=run(f);self.assertEqual(checks.failures,[]);self.assertEqual(result['internal_draws'],32)
        # A no-contact source is an explicit zero-draw identity, not redrawn.
        old[1]=pose(8);source=a.feasibility(o,c,old,o.fingerprint([0,1],old));f[-1]=source
        out.update(status='source_outside_domain',source_feasibility=source,source_frame=frame(o,c,anchor,old),attempts=[]);contacts[:]=[]
        result,checks=run(f);self.assertEqual(checks.failures,[]);self.assertEqual(result['stage_status'],'source_outside_domain')

    def test_final_child_wall_failure_not_accepted(self):
        f=list(fixture());h,o,c,old,anchor,out,contacts,source=f
        # Existing sphere toy: move accepted root to just inside wall, child outside.
        attempt=out['attempts'][0];root=pose(99.);relative=pose(1.);child=pose(100.)
        change_uniform(attempt['root_draws'][0],a.relative_pose(anchor,root));attempt['root_draws'][0].update(world_pose=root,feasibility=a.root_geometry(o,c,root))
        change_uniform(attempt['internal_draws'][0],relative);attempt['internal_draws'][0]['feasibility']=a.internal_geometry(o,relative)
        new=[root,child];geom=o.fingerprint([0,1],new);full=a.feasibility(o,c,new,geom)
        attempt.update(status='final_rejected',proposed=new,final_feasibility=full,frame=frame(o,c,anchor,new));out.update(status='cap_exhausted',candidate=None);contacts[:]=[geom['contacts']]
        result,checks=run(f);self.assertEqual(checks.failures,[]);self.assertEqual(result['stage_status'],'final_rejected')

    def test_allocation_contract_rejects_extra_draws_or_changed_tolerance(self):
        config=dict(master_seed=6100203101,joint_cap=32,root_cap=32,internal_cap=32,factorized_joint_cap=1,attempts_per_context=32,factorized_order='root_first')
        protocol=dict(master_seed=6100203101,caps=dict(whole_joint=32,factorized_root=32,factorized_internal=32,factorized_joint=1),factorized_order='root_first',
            allocation=dict(atlases=3,contexts=8,outer_slots_per_context=32,methods=a.METHODS,total_method_trials=1536,independent_context_slots=768,maximum_raw_edge_draws_per_method_trial=64,maximum_executed_raw_edge_draws=98304,extension=False),
            density_tolerance=dict(absolute=2e-7,relative=2e-10),physical_draws=0,state_updates=0,native_classification=False,native_label_filtering=False)
        a.validate_contract(config,protocol)
        for mutation in ['attempts','seed','tolerance','extension','order']:
            c,p=copy.deepcopy(config),copy.deepcopy(protocol)
            if mutation=='attempts':c['attempts_per_context']=33
            elif mutation=='seed':c['master_seed']+=1
            elif mutation=='tolerance':p['density_tolerance']['absolute']=1e-5
            elif mutation=='extension':p['allocation']['extension']=True
            else:p['factorized_order']='internal_first'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):a.validate_contract(c,p)

    def test_summary_keeps_failed_denominators(self):
        rows=[dict(status='cap_exhausted',stage_status='root_cap_exhausted',root_draws=32,internal_draws=0,raw_edge_draws=32,assembled=0,candidate=None,proposal_cpu_seconds=.2,contact_diagnostic_cpu_seconds=.01),
              dict(status='candidate',stage_status='candidate',root_draws=1,internal_draws=1,raw_edge_draws=2,assembled=1,candidate=dict(external_contacts=2,both_intended_contacts=False,log_reverse_forward=-3.),proposal_cpu_seconds=.1,contact_diagnostic_cpu_seconds=.01)]
        summary=a.summarize(rows);self.assertEqual(summary['outer_attempts'],2);self.assertEqual(summary['raw_edge_draws'],34);self.assertEqual(summary['candidate_fraction'],.5)


if __name__=='__main__':unittest.main()
