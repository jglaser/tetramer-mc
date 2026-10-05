"""Deterministic saved-summary reductions; no geometry or sampler invocations."""
import copy
import itertools
import unittest
from compare_context_prior_pilot import ARMS,STARTS,compare,efficiency_pair,probability_comparison,patch_marginal_comparison


def fixture():
    reports=[]
    for start,stream,arm in itertools.product(STARTS,range(4),ARMS):
        p=.8 if start==STARTS[0] else .2
        ess=dict(apparent_ess_per_sampling_CPU_second=None if stream==0 else 1.+ARMS.index(arm))
        sample=dict(patch_ess=ess,fingerprint_ess=ess,neighbor_ess=ess,
            singleton_fingerprint_fraction=0.,unique_fingerprints=2,empty_contact_fraction=0.,
            fingerprint=dict(occupancy={'x':p,'y':1-p}),neighbor_sets=dict(occupancy={'a':p,'b':1-p}),
            patch_occupancy=[dict(token=(16,77,'a','b'),fraction=p)])
        ab=dict(instantaneous_unconditional_occupancy={'A':dict(count=8192,fraction=.8),'B':dict(count=2048,fraction=.2),'Other':dict(count=0,fraction=0.)},
            directional_counts={'A->B':1,'B->A':0},completed_returns=[],nonoverlapping_roundtrips=[])
        reports.append(dict(complete=True,passed=True,identity=dict(start=start,stream=stream,arm=arm),
            journal_audit=dict(attempts=11520),prior_active=arm=='context',native_classifier_calls=0,new_poses=0,physical_draws=0,
            metrics=dict(all_attempts=dict(sample,samples=10240),cycle_endpoints=dict(sample,samples=2048),
                certified_environment_exchanges=ab,persistent_neighbors=dict(completed_partner_exchanges=1),proposal_and_bath_factors=[]),
            costs=dict(invocation_cpu_seconds=10.),offline_observer_cpu_seconds=1.))
    return reports


class Compare(unittest.TestCase):
    def test_complete_support_partition_and_tv(self):
        result=probability_comparison({'a':.7,'b':.3},{'b':.4,'c':.6})
        self.assertAlmostEqual(result['total_variation'],.7)
        self.assertEqual(result['intersection'],1)
        self.assertEqual(result['complete_support_partition'],dict(shared=['b'],left_only=['a'],right_only=['c']))
        self.assertAlmostEqual(result['left_probability_in_shared_support'],.3)

    def test_patch_marginals_do_not_impose_normalization(self):
        a=[dict(token=(16,77,'x','y'),fraction=.8),dict(token=(16,77,'x','z'),fraction=.8)]
        result=patch_marginal_comparison(a,[])
        self.assertEqual(result['maximum_absolute_difference'],.8)

    def test_all24_independent_identities_and_null_ratios(self):
        reports=fixture();result=compare(reports)
        self.assertEqual(len(result['streams']),24)
        self.assertEqual(len(result['paired_efficiency']),144)
        self.assertEqual(len(result['between_start_comparisons']),96)
        self.assertEqual(len(result['within_start_comparisons']),72)
        self.assertEqual(len(result['equal_stream_average_start_comparisons']),6)
        self.assertEqual(sum(r['ratio'] is None for r in result['paired_efficiency']),36)
        self.assertTrue(all(r['defined_streams']==3 and r['undefined_streams']==1 for r in result['efficiency_ratio_summaries']))
        self.assertEqual(result['costs']['total_full_sampler_cpu_seconds'],240.)
        self.assertAlmostEqual(result['equal_stream_average_start_comparisons'][0]['fingerprint']['total_variation'],.6)
        self.assertIsNone(result['costs']['total_end_to_end_cpu_seconds'])
        with self.assertRaises(ValueError):compare(reports[:-1])
        broken=copy.deepcopy(reports);broken[-1]=broken[0]
        with self.assertRaises(ValueError):compare(broken)


if __name__=='__main__':unittest.main()
