"""Pure saved-graph/candidate summaries, without physical or atomic queries."""
import copy
import math
import unittest
from compare_mobile_competing_atlas import candidate_statistics,summarize,transitions
from mobile_posterior_metrics import summarize_graph_history


class MobileCompetingComparisonTests(unittest.TestCase):
    def fixture(self):
        descriptions=[(0,[[1,2]],[[0,2],[1,2]]),
            (1,[[1,2]],[[0,2],[1,2]]),(1,[[0,1],[1,2]],[[0,1],[1,2]]),
            (2,[[0,1],[1,2]],[[0,1],[1,2]]),(2,[[1,2]],[[0,2],[1,2]]),
            (3,[[0,2],[1,2]],[[0,2],[1,2]]),(3,[[0,2],[1,2]],[[0,2],[1,2]])]
        history=[dict(serial=i-1,sweep=s,native_edges=n,nonspecific_edges=p,
            source='initial' if i==0 else 'frozen-posterior:involution') for i,(s,n,p) in enumerate(descriptions)]
        rows=[]
        for s in range(4):
            h=[r for r in history if r['sweep']==s][-1]
            keys=[[a,b,8 if a==1 else (5 if b==1 else 6)] for a,b in h['native_edges']]
            rows.append(dict(sweep=s,sampler_cpu_seconds=s*2.,native=dict(edges=h['native_edges']),
                nonspecific=dict(edges=h['nonspecific_edges']),registered_keys=keys))
        candidates=[dict(serial=0,sweep=1,source='frozen-posterior:involution',moving=0,accepted=False,
            new_registered_keys=[[0,1,5]],proposal_correction=-7.,gate_log_weight=3.,log_acceptance=-4.),
            dict(serial=1,sweep=1,source='frozen-posterior:involution',moving=1,accepted=True,
            new_registered_keys=[[0,1,5]],proposal_correction=0.,gate_log_weight=2.,log_acceptance=0.),
            dict(serial=4,sweep=3,source='frozen-posterior:involution',moving=0,accepted=True,
            new_registered_keys=[[0,2,6]],proposal_correction=-2.,gate_log_weight=10.,log_acceptance=0.)]
        return dict(id='synthetic',mode='c09',replicate=0,seed=1,graph_history=history,rows=rows,
            sampler_cpu_seconds=6.,postburn_sampler_cpu_seconds=4.,native_candidates=candidates,
            changes=[dict(serial=1,sweep=1,formed=[[0,1,5]],broken=[]),dict(serial=3,sweep=2,formed=[],broken=[[0,1,5]]),
                     dict(serial=4,sweep=3,formed=[[0,2,6]],broken=[])],initial_registered_keys=[[1,2,8]],
            graph_metrics=summarize_graph_history(history,3,1,3,4.),branch_counts={'frozen-posterior:involution':{'attempted':6}})

    def test_first_events_use_all_updates_and_preserve_moving_body_motifs(self):
        result=summarize(self.fixture(),'legacy',1,3)
        self.assertEqual(result['first_body0_native']['serial'],1)
        self.assertEqual(result['first_body0_native']['moving_body_from_global_candidate'],1)
        self.assertEqual(result['first_body0_native']['new_registered_keys_from_global_candidate'],[[0,1,5]])
        self.assertEqual(result['first_initial_0_2_exclusion_contact_loss']['serial'],1)
        self.assertEqual(len(result['events']['initial_0_2_exclusion_contact']['entries']),1)
        self.assertEqual(result['events']['body0_native']['completed_returns_to_initial_state'],1)
        self.assertEqual(result['final_registered_keys'],[[0,2,6],[1,2,8]])

    def test_registration_while_contact_persists_is_not_a_contact_loss(self):
        result=summarize(self.fixture(),'augmented',1,3)
        self.assertEqual(result['registration_of_initial_contact'],[dict(serial=4,sweep=3,source='frozen-posterior:involution',initial_contact_still_present=True)])
        self.assertEqual(len(result['body0_partner_metrics']['nonspecific']['same_attempt']),2)
        self.assertEqual(len(result['body0_partner_metrics']['nonspecific']['sequential']),0)
        self.assertEqual(result['postburn_occupancy']['all_three_native_connected']['full_postburn'],.5)

    def test_connected_native_three_does_not_require_the_third_edge(self):
        result=summarize(self.fixture(),'legacy',1,3)
        self.assertEqual(result['first_all_three_native']['sweep'],1)
        self.assertIsNone(result['first_native_triangle'])
        self.assertTrue(result['events']['all_three_native_edges']['right_censored_no_entry'])

    def test_moving_and_incident_candidates_are_distinct_and_burn_is_strict(self):
        result=summarize(self.fixture(),'legacy',1,3)
        moving=result['body0_moving_native_candidate_statistics'];incident=result['body0_incident_native_candidate_statistics']
        self.assertEqual(moving['full']['total']['hard_valid_new_native_registry_candidates'],2)
        self.assertEqual(incident['full']['total']['hard_valid_new_native_registry_candidates'],3)
        self.assertEqual(moving['full']['total']['positive_recorded_bath_log_factors'],2)
        self.assertAlmostEqual(moving['full']['total']['sum_recorded_conditional_alphas'],1+math.exp(-4))
        self.assertEqual(moving['postburn']['total']['hard_valid_new_native_registry_candidates'],1)
        self.assertEqual(moving['postburn']['total']['accepted'],1)

    def test_returns_require_an_observed_departure_and_initial_true_is_censored(self):
        history=[dict(serial=i-1,sweep=max(0,i),source='synthetic') for i in range(4)]
        result=transitions(history,[True,True,False,True])
        self.assertEqual(result['completed_returns_to_initial_state'],1)
        self.assertEqual(len(result['entries']),1)
        self.assertEqual(result['first_observed_true']['serial'],-1)
        self.assertEqual(transitions(history,[False]*4)['right_censored_no_entry'],True)

    def test_corrupt_endpoint_and_invalid_candidate_alpha_fail(self):
        run=self.fixture();run['rows'][1]['native']={'edges':[[1,2]]}
        with self.assertRaisesRegex(ValueError,'endpoint graph'):summarize(run,'legacy',1,3)
        record=copy.deepcopy(self.fixture()['native_candidates'][0]);record['log_acceptance']=.1
        with self.assertRaisesRegex(ValueError,'acceptance'):candidate_statistics([record],1,[record['source']])


if __name__=='__main__':unittest.main()
