import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_hard_free_protein_smc as runner


class ProteinControlTests(unittest.TestCase):
    def test_fixed_allocation_matches_cli_without_reference_initializer(self):
        root = Path('/synthetic/control')
        jobs = runner.jobs(root)
        self.assertEqual(len(jobs),4)
        self.assertEqual(len({j['seed'] for j in jobs}),4)
        for job in jobs:
            command = job['command']; args = dict(zip(command[1::2],command[2::2]))
            expected = runner.expected_options(root,job['seed'])
            for key in ('population','initial_draws','seed','sweeps_per_stage','cloud_replicates'):
                self.assertEqual(int(args['--'+key.replace('_','-')]),expected[key])
            self.assertEqual(args['--out'],expected['out'])
            self.assertEqual(args['--initial-guide'],str(root/'common/guide.json'))
            self.assertNotIn('--initial-reference-region',args)
            self.assertEqual(expected['schedule'],[i/128 for i in range(129)])
            self.assertIsNone(expected['initial_reference_region'])
            self.assertEqual(expected['initial_current_probability'],1.)

    def fixture(self,root):
        (root/'comparison').mkdir()
        runner.write(root/'protocol.json',dict(schema='synthetic'))
        digest = runner.sha(root/'protocol.json')
        runner.write(root/'comparison/analysis.json',dict(complete=True,protocol_sha256=digest))
        # Failed convergence is a retained prerequisite, not a failed execution.
        runner.write(root/'sensitivity-comparison.json',dict(complete=True,full_vessel_gate_open=False,
            assembly_gate_open=False,sensitivity_checks_passed=False))
        state = dict(complete=True,phase='complete',protocol_sha256=digest,
            comparison_sha256=runner.sha(root/'comparison/analysis.json'),
            sensitivity_comparison_sha256=runner.sha(root/'sensitivity-comparison.json'),
            jobs=[dict(status='complete',returncode=0) for _ in range(8)],
            audits=[dict(status='complete',returncode=0) for _ in range(8)])
        runner.write(root/'status.json',state)
        return state,digest

    def test_completed_failed_diagnostics_remain_admissible_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); _,digest=self.fixture(root)
            _,files=runner.completed_iid(root,'sensitivity-comparison',digest)
            self.assertEqual(len(files),4)
            self.assertFalse(runner.read(root/'sensitivity-comparison.json')['sensitivity_checks_passed'])

    def test_incomplete_population_or_audit_refuses_prerequisite(self):
        for change in ('phase','job','audit'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as name:
                root=Path(name);state,digest=self.fixture(root)
                if change=='phase':state.update(complete=False,phase='classification')
                elif change=='job':state['jobs'][3]['returncode']=1
                else:state['audits'].pop()
                runner.write(root/'status.json',state)
                with self.assertRaises(ValueError):runner.completed_iid(root,'sensitivity-comparison',digest)

    def test_tampered_summary_and_promoted_gate_refused(self):
        for change in ('summary','gate'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as name:
                root=Path(name);state,digest=self.fixture(root)
                if change=='summary':
                    runner.write(root/'comparison/analysis.json',dict(complete=True,protocol_sha256='other'))
                else:
                    p=root/'sensitivity-comparison.json';data=runner.read(p);data['assembly_gate_open']=True;runner.write(p,data)
                    state['sensitivity_comparison_sha256']=runner.sha(p);runner.write(root/'status.json',state)
                with self.assertRaises(ValueError):runner.completed_iid(root,'sensitivity-comparison',digest)

    def test_missing_terminal_output_cannot_be_reported_complete(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);job=runner.jobs(root)[0]
            Path(job['directory']).mkdir(parents=True)
            runner.write(Path(job['directory'])/'status.json',dict(complete=False,phase='initialization'))
            with self.assertRaisesRegex(ValueError,'incomplete'):runner.check_output(root,{},job)

    def test_host_capacity_is_checked_before_claim_or_any_dispatch(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            plan=dict(jobs=runner.jobs(root),files_sha256={})
            for counts in [dict(private_pid_namespace=True,physical_pids=[],workers=0),
                    dict(private_pid_namespace=False,physical_pids=list(range(7)),workers=7),
                    dict(private_pid_namespace=False,physical_pids=[],workers=31)]:
                with self.subTest(counts=counts),patch.object(runner,'validate',return_value=plan), \
                        patch.object(runner,'__file__',str(root/'common/run_hard_free_protein_smc.py')), \
                        patch.object(runner,'capacity',return_value=counts),patch.object(runner,'execute_group') as dispatch:
                    with self.assertRaises(ValueError):runner.run(root,'synthetic')
                    dispatch.assert_not_called();self.assertFalse((root/'status.json').exists())

    def test_physical_failure_preserves_claim_and_never_starts_audits(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            plan=dict(jobs=runner.jobs(root),files_sha256={},repository=str(root))
            with patch.object(runner,'validate',return_value=plan), \
                    patch.object(runner,'__file__',str(root/'common/run_hard_free_protein_smc.py')), \
                    patch.object(runner,'capacity',return_value=dict(private_pid_namespace=False,physical_pids=[],workers=0)), \
                    patch.object(runner,'execute_group',side_effect=RuntimeError('drained child failure')) as dispatch:
                with self.assertRaisesRegex(RuntimeError,'drained'):runner.run(root,'synthetic')
                state=runner.read(root/'status.json');self.assertEqual(state['phase'],'physical_failed')
                self.assertFalse(state['complete']);self.assertEqual(state['audits'],[])
                with self.assertRaisesRegex(ValueError,'No retry'):runner.run(root,'synthetic')
                dispatch.assert_called_once()


if __name__=='__main__':unittest.main()
