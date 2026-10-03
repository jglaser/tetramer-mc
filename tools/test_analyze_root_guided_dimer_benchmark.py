"""Cached-label controls: no atom geometry, protein poses, or physical samples."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze_root_guided_dimer_benchmark as module


def row(block, external=(), internal=False):
    return dict(block=block,production=block>0,
        external_edges=[list(e) for e in external],
        partner_edges=[list(e) for e in external]+([[1,2]] if internal else []))


def chain(arm, initialization, external, cpu, stream=0, context=0):
    trace=[row(0)]+[row(i+1, edges, internal=bool(i%2)) for i,edges in enumerate(external)]
    for observation in trace:
        observation['patch_tokens']=[[*e,0,0] for e in observation['partner_edges']]
        observation['fingerprint']=module.previous.key(observation['patch_tokens'])
        observation['internal_contact']=[1,2] in observation['partner_edges']
    metrics=module.previous.summarize_trace(trace,0,cpu)
    metrics['external_only']=module.external_metrics(trace,0,cpu,[1,2])
    return dict(job=dict(arm=arm,initialization=initialization,stream=stream,context_index=context),metrics=metrics)


class ExternalControlMetrics(unittest.TestCase):
    def test_internal_attachment_changes_are_not_external_exchanges(self):
        trace=[row(i,internal=bool(i%2)) for i in range(7)]
        with patch.object(module.previous,'ConditionalObserver',side_effect=AssertionError('No geometry')):
            result=module.external_metrics(trace,0,12.,[1,2])
        self.assertIsNone(result['ess']['apparent_ess'])
        self.assertEqual(result['environments']['completed_passages'],0)
        self.assertEqual(result['environments']['completed_returns'],[])
        self.assertEqual(result['any_contact_fraction'],0)
        self.assertEqual(result['unique_external_neighbor_labels'],[])

    def test_empty_external_sets_and_rejected_residence_are_retained(self):
        ext=[[(1,8)],[(1,8)],[],[(1,8)],[(2,9)],[(1,8)]]
        trace=[row(0)]+[row(i+1,e,internal=bool(i%2)) for i,e in enumerate(ext)]
        result=module.external_metrics(trace,0,10.,[1,2])
        self.assertEqual(result['ess']['samples'],6)
        self.assertEqual(result['environments']['completed_passages'],4)
        self.assertEqual(len(result['environments']['completed_returns']),2)
        self.assertEqual(result['unique_external_neighbor_labels'],[8,9])
        self.assertEqual(result['any_contact_fraction'],5/6)
        self.assertEqual({tuple(x['edge']):x['fraction'] for x in result['occupancy']},
                         {(1,8):4/6,(2,9):1/6})

    def test_changed_partition_or_foreign_edges_rejected(self):
        for malformed in [row(1,[(1,2)]),row(1,[(8,9)]),row(1,[(1,8),(1,8)])]:
            with self.subTest(row=malformed), self.assertRaises(ValueError):
                module.external_metrics([malformed],0,1.,[1,2])
        bad=row(1,[(1,8)]);bad['partner_edges']=[]
        with self.assertRaises(ValueError): module.external_metrics([bad],0,1.,[1,2])

    def test_cached_hash_sequence_and_attempt_count_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache.jsonl'
            original=[dict(block=i,production=i>1) for i in range(4)]
            def save(rows):
                path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            save(original); digest=module.sha(path)
            self.assertEqual(module.read_cached_trace(path,digest,warmup=1,blocks=3),original)
            save(original[:-1])
            with self.assertRaises(ValueError): module.read_cached_trace(path,digest,warmup=1,blocks=3)
            with self.assertRaises(ValueError): module.read_cached_trace(path,module.sha(path),warmup=1,blocks=3)
            changed=[dict(r) for r in original]; changed[2]['block']=1;save(changed)
            with self.assertRaises(ValueError): module.read_cached_trace(path,module.sha(path),warmup=1,blocks=3)

    def test_frozen_allocation_and_scope(self):
        plan=module.analysis_plan()
        self.assertEqual(plan['new_retained_initial_observations'],147488)
        self.assertEqual(plan['new_production_observations'],131072)
        self.assertEqual(plan['maximum_new_pair_classifications'],77431200)
        self.assertEqual(plan['new_chains'],plan['reused_control_chains'])
        self.assertFalse(plan['native_observer'])
        self.assertFalse(plan['trajectories_concatenated'])

    def test_external_paired_rates_keep_cpu_and_null_ess(self):
        left=chain('m4','source',[[],[],[],[]],10.)
        right=chain('root_m4','source',[[(1,8)],[],[(1,8)],[]],2.)
        with patch.object(module.previous,'ConditionalObserver',side_effect=AssertionError('No geometry')):
            result=module.comparison_summaries([left,right])
        comparison,=result['descriptive_paired_comparisons']
        self.assertTrue(comparison['control_global_rng_roles_paired'])
        value=comparison['external_only']
        self.assertEqual(value['difference_direction'],'right minus left')
        self.assertEqual(value['environment_total_variation'],.5)
        self.assertEqual(value['edge_occupancy_max_difference'],.5)
        self.assertEqual(value['any_contact_fraction_difference'],.5)
        self.assertIsNone(value['left']['apparent_ess'])
        self.assertIsNotNone(value['right']['apparent_ess'])
        self.assertIsNone(value['apparent_ess_per_sampling_CPU_second_difference'])
        self.assertEqual(value['left']['completed_passages'],0)
        self.assertEqual(value['right']['completed_passages'],3)
        self.assertEqual(value['right']['completed_returns'],2)
        self.assertEqual(value['passages_per_sampling_CPU_second_difference'],1.5)
        self.assertEqual(value['returns_per_sampling_CPU_second_difference'],1.)
        self.assertEqual(result['groups'][0]['independent_stream_metrics'][0]['external_only'],value['left'])

    def test_external_initialization_agreement_and_pairing_do_not_pool(self):
        chains=[chain('m4','source',[[(1,8)],[],[(1,8)],[]],4.),
                chain('root_m4','source',[[],[(2,9)],[],[(2,9)]],8.),
                chain('root_m4','proposal_prepared',[[],[(2,9)],[],[(2,9)]],16.),
                chain('root_m4','source',[[],[],[],[]],1.,stream=1),
                chain('m4','source',[[],[],[],[]],1.,context=1)]
        result=module.comparison_summaries(chains)
        pairs=result['descriptive_paired_comparisons']
        self.assertEqual(len(pairs),2)
        cross_arm, initial=pairs
        self.assertTrue(cross_arm['control_global_rng_roles_paired'])
        value=cross_arm['external_only']
        self.assertEqual(value['environment_total_variation'],.5)
        self.assertEqual(value['edge_occupancy_max_difference'],.5)
        self.assertAlmostEqual(value['apparent_ess_per_sampling_CPU_second_difference'],
            value['right']['apparent_ess_per_sampling_CPU_second']-value['left']['apparent_ess_per_sampling_CPU_second'])
        self.assertFalse(initial['control_global_rng_roles_paired'])
        self.assertEqual(initial['external_only']['environment_total_variation'],0.)
        self.assertEqual(initial['external_only']['edge_occupancy_max_difference'],0.)
        self.assertEqual(initial['external_only']['passages_per_sampling_CPU_second_difference'],-3/16)
        self.assertEqual(len(result['groups']),4)
        with self.assertRaisesRegex(ValueError,'Duplicate comparison stream'):
            module.comparison_summaries([chains[0],chains[0]])


if __name__=='__main__': unittest.main()
