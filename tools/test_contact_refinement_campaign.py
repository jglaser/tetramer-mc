"""Controller allocation/provenance and independent-chain diagnostic controls."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_contact_refinement_campaign as campaign
import analyze_contact_refinement_campaign as analysis


class CampaignControl(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.shape = self.root/'shape.json'
        self.shape.write_text('{}\n')
        self.binary = self.root/'fake-discovery'
        self.binary.write_text('#!/bin/sh\necho --refine-adapt --refine-contact-pivot\n')
        self.binary.chmod(0o755)
        rows = [dict(global_slot=i, source_index=0, local_slot=i, source_seed=123,
                     reference_pose=dict(position=[2., 0., 0.], orientation=[1., 0., 0., 0.]))
                for i in range(3)]
        campaign.write(self.source/'fit-metrics.json', dict(independent_slots=3, slots=rows))
        campaign.write(self.source/'manifest.json', dict(complete=True, native_informed_proposal=False,
                       native_geometry_used_in_preparation=False, shape_sha256=campaign.sha(self.shape)))
        campaign.write(self.source/'freeze.json', dict(files={name: campaign.sha(self.source/name)
                       for name in ('fit-metrics.json', 'manifest.json')}))
        self.out = self.root/'campaign'
        self.args = campaign.parser().parse_args(['prepare', '--source-fit', str(self.source),
            '--shape', str(self.shape), '--binary', str(self.binary), '--out', str(self.out),
            '--slots', '3', '--slots-per-job', '2', '--refine-steps', '12', '--burn', '4', '--save-every', '2'])

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_slots_replicates_and_equal_arm_allocations_are_frozen_without_launch(self):
        report = campaign.prepare(self.args)
        plan = campaign.validate(self.out)
        self.assertFalse(report['launched'])
        self.assertEqual(report['jobs'], 8)
        self.assertEqual(plan['total_attempts'], 144)
        self.assertEqual(sorted(row['new_global_slot'] for row in plan['source_slot_map']), list(range(6)))
        self.assertFalse((self.out/'status.json').exists())
        for first, second in zip(plan['jobs'][::2], plan['jobs'][1::2]):
            self.assertEqual(first['seed'], second['seed'])
            self.assertEqual(first['initial_poses'], second['initial_poses'])
            self.assertEqual(first['slot_map'], second['slot_map'])
        self.assertEqual(len({job['seed'] for job in plan['jobs']}), 4)

    def test_no_overwrite_and_changed_frozen_plan_rejected(self):
        campaign.prepare(self.args)
        with self.assertRaisesRegex(ValueError, 'Fresh campaign'):
            campaign.prepare(self.args)
        plan = campaign.read(self.out/'plan.json')
        plan['workers'] = 5
        campaign.write(self.out/'plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'Frozen plan changed'):
            campaign.validate(self.out)

    def test_native_informed_source_and_shape_change_rejected(self):
        manifest = campaign.read(self.source/'manifest.json')
        manifest['native_informed_proposal'] = True
        campaign.write(self.source/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'native-blind'):
            campaign.prepare(self.args)
        manifest['native_informed_proposal'] = False
        campaign.write(self.source/'manifest.json', manifest)
        self.shape.write_text('{"changed":true}\n')
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            campaign.prepare(self.args)


class ReplicateDiagnostics(unittest.TestCase):
    def test_constants_do_not_become_iid_success(self):
        x = np.zeros((32, 6))
        result = analysis.replicate_diagnostics([x, x])
        self.assertEqual(result['undefined_split_rhat_axes'], 6)
        self.assertIsNone(result['maximum_pooled_pca_split_rhat'])
        self.assertIsNone(analysis.drift(x, x)['covariance_relative_frobenius'])

    def test_matching_iid_and_shifted_chain_controls(self):
        rng = np.random.default_rng(8173)
        first, second = rng.normal(size=(2, 1024, 6))
        self.assertLess(analysis.replicate_diagnostics([first, second])['maximum_pooled_pca_split_rhat'], 1.05)
        self.assertGreater(analysis.replicate_diagnostics([first, second+5])['maximum_pooled_pca_split_rhat'], 2)
        self.assertEqual(analysis.drift(first, first)['covariance_relative_frobenius'], 0.)


if __name__ == '__main__':
    unittest.main()
