"""Synthetic, no-sampling integration of audit -> classification -> summaries."""
import contextlib
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import analyze_hard_free_line_physical as analysis
import hard_free_line_physical_reference as physical
from test_hard_free_line_physical_reference import fixture, write
from run_smc_guide_pilot import STRATA, CONVERGENCE


class Native:
    definition = {'synthetic': True}
    calls = 0

    def classify(self, pose):
        self.calls += 1
        native = self.calls % 2 == 1
        return dict(native_any=native, native_anchor_count=1 if native else 0,
                    registry_consistent_triangle=False)


class Contact:
    calls = 0

    def classify(self, pose):
        self.calls += 1
        return dict(exclusion_contact=True, near_zero_negative_gap=False)


def campaign(root):
    arms=[dict(id=name, alpha=.5, beta=0., component_count=2, samples=3) for name in ('baseline','conditioned')]
    jobs=[];package=root/'common/reference-package';package.mkdir(parents=True)
    (package/'native-region').mkdir();write(package/'native-region/definition.json',{'synthetic':True})
    for ai,arm in enumerate(arms):
        (root/'audits'/arm['id']).mkdir(parents=True)
        for i in range(4):
            directory=root/'jobs'/arm['id']/f'r{i:02}';directory.mkdir(parents=True)
            region,_,_,_,_,manifest,summary=fixture(directory)
            seed=100+4*ai+i;manifest['seed']=seed;summary['manifest']=manifest
            write(directory/'manifest.json',manifest);write(directory/'summary.json',summary)
            write(root/'audits'/arm['id']/(f'r{i:02}.json'),physical.audit(directory))
            jobs.append(dict(id=f'r{i:02}',arm=arm['id'],directory=str(directory),seed=seed,samples=3))
    write(package/'region.json',region);write(package/'old-r5-region.json',region)
    protocol=dict(arms=arms,jobs=jobs,strata=STRATA,convergence=CONVERGENCE,total_unconditional_draws=24,
                  supplemental_definition_sha256='synthetic-partition-definition')
    write(root/'protocol.json',protocol)
    return protocol


def substitutes(native,contact):
    stack=contextlib.ExitStack()
    # Only classifiers and their external definition compatibility are mocked.
    # Pose/weight audit, chart backmap, strata, stored masks and summaries are real.
    stack.enter_context(patch.object(analysis,'load_classifier',return_value=(native,dict(definition_sha256='synthetic-native'))))
    stack.enter_context(patch.object(analysis,'ExclusionContact',return_value=contact))
    stack.enter_context(patch.object(analysis,'validate_regions',return_value=None))
    stack.enter_context(patch.object(analysis,'validate_classifier_target',return_value=None))
    return stack


class ClassificationPipelineTests(unittest.TestCase):
    def test_all_attempts_single_classification_and_complete_summaries(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);campaign(root);native,contact=Native(),Contact()
            with substitutes(native,contact):result=analysis.analyze(root)
            self.assertEqual(native.calls,8);self.assertEqual(contact.calls,8)
            self.assertEqual(result['total_unconditional_draws'],24)
            self.assertFalse(result['diagnostics']['assembly_gate_open'])
            self.assertFalse(result['diagnostics']['full_vessel_gate_open'])
            for arm in result['arms'].values():
                self.assertEqual(arm['estimates']['total']['row_uncertainty']['draws'],12)
                self.assertEqual(len(arm['populations']),4)
                self.assertTrue(arm['native_partition_sum_verified'])
                self.assertNotIn('gaussian',arm['branch_weight_contributions'])
                self.assertEqual(len(arm['strata']['orthant']['total']),64)
                for population in arm['populations']:
                    self.assertEqual(population['full_native_classifier_calls'],1)
                    path=root/'comparison'/population['records']
                    with np.load(path,allow_pickle=False) as rows:
                        self.assertEqual(len(rows['z']),3);self.assertEqual(int(np.isfinite(rows['z']).sum()),1)
                        self.assertEqual(rows['bin_radial'].tolist(),[0,0,-1])
                        self.assertEqual(rows['branch'].tolist(),[1,1,1])
                        self.assertEqual(rows['source_n'].tolist(),[3,3,3])
                    with gzip.open(root/'comparison'/population['labels'],'rt') as stream:
                        labels=[json.loads(s) for s in stream]
                    self.assertEqual([r['applicable'] for r in labels],[True,False,False])

    def test_stale_audit_rejected_before_any_classifier_call(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);protocol=campaign(root);directory=Path(protocol['jobs'][0]['directory'])
            summary=physical.read(directory/'summary.json');summary['sampler_cpu_seconds']=.2
            write(directory/'summary.json',summary);native,contact=Native(),Contact()
            with substitutes(native,contact), self.assertRaisesRegex((AssertionError,ValueError),'audit refers'):
                analysis.analyze(root)
            self.assertEqual(native.calls,0);self.assertEqual(contact.calls,0)


if __name__=='__main__':unittest.main()
