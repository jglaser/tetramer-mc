"""Synthetic endpoint -> full classifier -> registry-history integration."""
import copy
import unittest

from conditional_native_metrics import iter_authenticated_endpoints, native_metrics
from conditional_native_observer import ConditionalNativeObserver
from test_conditional_native_observer import ToyNative, pose


class NativePipelineTests(unittest.TestCase):
    def test_residence_detachment_and_return_have_distinct_counting_and_cpu(self):
        source = [pose(x) for x in (0, 4, 8, 30)]
        initial = [pose(0), pose(12)]
        selected = [initial, initial, [pose(0), pose(40)],
                    [pose(-10), pose(40)], initial]
        job = dict(context_index=0, arm='synthetic', initialization='source', stream=0)
        rows = [dict(kind='initial', block=0, conditional_target=True, job=job,
                     selected=initial, sampler_cpu_seconds=0.)]
        for block in range(1, 5):
            # The reader deliberately does not revalidate elementary proposal
            # arithmetic. A production caller must authenticate its prior receipt.
            rows.append(dict(kind='local', accepted=block > 1))
            rows.append(dict(kind='retained_block', block=block, production=block > 1,
                             selected=selected[block], sampler_cpu_seconds=float(block)))
        observer = ConditionalNativeObserver(ToyNative(), source, [0, 3])
        trace = []
        for endpoint in iter_authenticated_endpoints(rows, job=job, initial=initial, blocks=4, warmup=1):
            trace.append(dict(endpoint, **observer.classify(endpoint['selected'])))
        before = copy.deepcopy(trace)
        report = native_metrics(trace, members=[0, 3], full_sampler_cpu_seconds=5., blocks=4, warmup=1)
        self.assertEqual(trace, before)
        self.assertEqual(report['retained_endpoints'], 5)
        self.assertEqual(report['production_samples'], 3)
        self.assertAlmostEqual(report['external_native_attachment_fraction'], 2/3)
        self.assertAlmostEqual(report['empty_native_registry_fraction'], 1/3)
        self.assertEqual(report['mobile_registry_resolved_fraction'], 1.)
        self.assertEqual(observer.counts['whole_endpoint_cache_hits'], 1)
        counts = report['external_partners']['counts']
        self.assertEqual(counts['direct_nonempty_resolved_changes'], 1)
        self.assertEqual(counts['leave_nonempty'], 1)
        self.assertEqual(counts['enter_nonempty'], 1)
        returns = report['external_partners']['environments']['completed_returns']
        self.assertEqual(len(returns), 1)
        self.assertTrue(returns[0]['returned_environment_nonempty'])
        self.assertTrue(returns[0]['passed_through_empty'])
        self.assertEqual(report['external_partners']['rates_per_full_sampler_cpu_second']['enter_nonempty'], .2)
        self.assertIsNone(report['internal_motifs']['presence_ess']['apparent_ess'])


if __name__ == '__main__':
    unittest.main()
