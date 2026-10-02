"""Fixed allocation/CLI controls; preparation never executes a sampler."""
import copy
from pathlib import Path
import tempfile
import unittest
import prepare_streaming_vessel_comparison as prep


class PreparationTests(unittest.TestCase):
    def test_both_stages_keep_independent_streams_and_matching_physics(self):
        jobs = prep.jobs_for('/tmp/inert-vessel-preparation')
        self.assertEqual(len(jobs),16); self.assertEqual(sum(j['samples'] for j in jobs),2621440)
        self.assertEqual(len({j['seed'] for j in jobs}),16)
        for stage,n in prep.STAGES:
            for arm in prep.ARMS:
                selected = [j for j in jobs if j['stage'] == stage and j['arm'] == arm]
                self.assertEqual([j['population'] for j in selected],list(range(4)))
                self.assertTrue(all(j['samples'] == n for j in selected))
        for job in jobs:
            command = job['command']; audit = job['audit_command']; partition = job['partition_command']
            self.assertEqual(command[command.index('--cloud-replicates')+1],'2')
            self.assertEqual('--latent-guide' in command,job['arm'] == 'half_mixture')
            self.assertEqual('--region' in audit,job['arm'] == 'vessel')
            self.assertNotIn('--native-definition',audit)
            self.assertIn('--native-definition',partition)
            self.assertTrue(all('--'+name.replace('_','-') in partition for name in prep.SUPPORTS))

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'Fresh inert'): prep.freeze(Path(temporary))

    def test_contract_rejects_budget_target_or_proposal_drift(self):
        root = Path('/tmp/inert-vessel-preparation')
        plan = dict(schema=prep.SCHEMA,preparation_only=True,physical_jobs_launched=0,dispatch_ready=False,
            jobs=prep.jobs_for(root),total_unconditional_draws=prep.TOTAL,
            stages=[dict(name=s,draws_per_population=n,independent_populations_per_arm=4) for s,n in prep.STAGES],
            arms=list(prep.ARMS),maximum_physical_workers=8,maximum_audit_workers=4,maximum_all_workers=32,
            thread_environment=prep.THREADS,
            physical=dict(depletant_radius=1.5,activity=.035,lambda_ratio=128.,cloud_replicates=2,
                wall_center=[0.,0.,0.],wall_radius=prep.WALL,capture_radius=273,bath_wall_permeable=True,
                measure='Lebesgue center volume times normalized SO(3) Haar measure'),
            input_sha256={'guide.json':prep.GUIDE[1],'current_R4.json':prep.SOURCES['current_R4.json'][1],
                          'shape.json':prep.SOURCES['shape.json'][1]},native_definition_sha256=prep.NATIVE[1])
        prep.validate_contract(plan,root)
        for mutation in ('draws','seed','threads','bath','guide','dispatch'):
            bad = copy.deepcopy(plan)
            if mutation == 'draws': bad['jobs'][0]['samples'] -= 1
            elif mutation == 'seed': bad['jobs'][1]['seed'] = bad['jobs'][0]['seed']
            elif mutation == 'threads': bad['maximum_all_workers'] = 33
            elif mutation == 'bath': bad['physical']['activity'] = .04
            elif mutation == 'guide': bad['input_sha256']['guide.json'] = '0'*64
            else: bad['dispatch_ready'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): prep.validate_contract(bad,root)


if __name__ == '__main__': unittest.main()
