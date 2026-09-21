"""Pure saved-graph/candidate summaries, without physical or atomic queries."""
import copy
import math
from pathlib import Path
import unittest
from unittest.mock import patch
from compare_mobile_competing_atlas import (
    candidate_statistics,comparison_design,render,sort_runs,summarize,transitions,validate_grid,
)
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

    def test_schema_defines_atlas_pair_and_preserves_legacy_first_order(self):
        for middle,variant in (('competing','augmented'),('reciprocal','reciprocal')):
            protocol=dict(schema=f'matched-mobile-{middle}-atlas-controller-v1',
                campaigns=[dict(atlas=variant),dict(atlas='legacy')])
            design=comparison_design(protocol)
            self.assertEqual(design['variants'],('legacy',variant))
            self.assertEqual(design['comparison_schema'],f'mobile-{middle}-atlas-comparison-v1')
            runs=[dict(atlas=a,mode=m,replicate=r) for a in (variant,'legacy') for m in ('c09','c0') for r in (1,0)]
            ordered=sort_runs(runs,design)
            self.assertEqual([(r['atlas'],r['mode'],r['replicate']) for r in ordered],
                [(a,m,r) for a in ('legacy',variant) for m in ('c0','c09') for r in (0,1)])
            bad=copy.deepcopy(protocol);bad['campaigns'][0]['atlas']='legacy'
            with self.assertRaises(ValueError):comparison_design(bad)
            bad=copy.deepcopy(protocol);bad['schema']='unknown'
            with self.assertRaises(ValueError):comparison_design(bad)
        reciprocal=comparison_design(dict(schema='matched-mobile-reciprocal-atlas-controller-v1',
            campaigns=[dict(atlas='legacy'),dict(atlas='reciprocal')]))
        self.assertNotIn('augmented',str(reciprocal))

    def test_comparison_grid_rejects_mixed_schema_and_wrong_terminal_ids(self):
        design=comparison_design(dict(schema='matched-mobile-reciprocal-atlas-controller-v1',
            campaigns=[dict(atlas='legacy'),dict(atlas='reciprocal')]))
        jobs=[dict(id=f'{m}-{r}',start='competing',mode=m,replicate=r) for m in ('c0','c09') for r in (0,1)]
        manifest=dict(schema=design['campaign_schema'],atlas_variant='reciprocal',jobs=jobs)
        status=dict(complete=True,running=False,jobs=[dict(id=j['id'],status='complete',exit_code=0) for j in jobs])
        self.assertEqual(set(validate_grid(manifest,status,design)),{j['id'] for j in jobs})
        for case in ('schema','variant','allocation','duplicate','terminal','failure'):
            m,s=copy.deepcopy(manifest),copy.deepcopy(status)
            if case=='schema':m['schema']='mobile-competing-atlas-benchmark-v1'
            elif case=='variant':m['atlas_variant']='augmented'
            elif case=='allocation':m['jobs'][0]['mode']='capture_only'
            elif case=='duplicate':m['jobs'][0]['id']=m['jobs'][1]['id']
            elif case=='terminal':s['jobs'][0]['id']='other'
            else:s['jobs'][0]['exit_code']=1
            with self.subTest(case=case),self.assertRaises(ValueError):validate_grid(m,s,design)

    def test_reciprocal_render_labels_and_filenames_do_not_claim_augmented_atlas(self):
        design=comparison_design(dict(schema='matched-mobile-reciprocal-atlas-controller-v1',
            campaigns=[dict(atlas='legacy'),dict(atlas='reciprocal')]))
        runs=[]
        for variant in design['variants']:
            for mode in ('c0','c09'):
                for repeat in (0,1):
                    run=self.fixture();run.update(mode=mode,replicate=repeat)
                    runs.append(summarize(run,variant,1,3))
        saved=[]
        def capture(fig,path,**kwargs):
            from matplotlib.text import Text
            saved.append((Path(path).name,[text.get_text() for text in fig.findobj(Text)]))
        with patch('matplotlib.figure.Figure.savefig',autospec=True,side_effect=capture):
            render(Path('/not-created'),runs,1,3,design)
        self.assertEqual([name for name,_ in saved],[f'mobile-reciprocal-atlas-comparison.{ext}' for ext in ('png','svg','pdf')])
        labels=' '.join(saved[0][1]);self.assertIn('exact reciprocal',labels)
        self.assertIn('Reciprocal · c0 · r0',labels);self.assertNotIn('augmented',labels.lower())

    def test_reciprocal_variant_preserves_every_existing_metric(self):
        baseline=summarize(self.fixture(),'legacy',1,3)
        reciprocal=summarize(self.fixture(),'reciprocal',1,3)
        reciprocal['atlas']='legacy'
        self.assertEqual(baseline,reciprocal)


if __name__=='__main__':unittest.main()
