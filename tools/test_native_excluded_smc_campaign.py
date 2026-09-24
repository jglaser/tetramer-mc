"""Control allocation/provenance/statistics tests; no protein sampling."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import run_native_excluded_smc_campaign as runner


class ControlTests(unittest.TestCase):
    def test_independent_allocation_and_only_scale_population_differences(self):
        jobs = runner.jobs_for('/tmp/example-control')
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({j['seed'] for j in jobs}), 12)
        self.assertEqual(len({j['output'] for j in jobs}), 12)
        self.assertEqual([j['arm'] for j in jobs], ['narrow','large','broad']*4)
        for job in jobs:
            options = dict(zip(job['argv'][1::2],job['argv'][2::2]))
            self.assertEqual(options['--initial-draws'], '262144')
            self.assertEqual(options['--population'], '4096' if job['arm']=='large' else '2048')
            self.assertEqual(options['--cloud-replicates'], '2')
            self.assertEqual(options['--lambda-ratio'], '128')
            self.assertEqual(options['--initial-current-probability'], '0.5')
            self.assertEqual(options['--bridge'], 'proposal-density')
            self.assertEqual(options['--stages'], '128')
            self.assertEqual(options['--sweeps-per-stage'], '4')
            self.assertTrue(options['--exclude-native-entry'].endswith('/inputs/native-compiled.json'))
        budget = runner.allocation()
        self.assertEqual(budget['total_initial_draws'], 3145728)
        self.assertEqual(budget['potential_particle_evaluations'], 4194304)
        self.assertEqual(budget['total_mutation_attempts'], 16777216)

    def test_population_statistics_keep_zeros_and_average_linear_weights(self):
        logs = [math.log(1), math.log(2), math.log(4), None]
        result = runner.mass_statistics(logs)
        self.assertAlmostEqual(math.exp(result['log_Q']), 7/4)
        values = [1,2,4,0]; mean = 7/4
        expected = math.sqrt(sum((x-mean)**2 for x in values)/12)/mean
        self.assertAlmostEqual(result['population_relative_SE'], expected)
        self.assertEqual(result['populations'],4)
        self.assertEqual(result['nonzero_populations'],3)
        self.assertEqual(result['population_log_masses'], logs)
        zero = runner.mass_statistics([None]*4)
        self.assertIsNone(zero['log_Q']); self.assertIsNone(zero['population_relative_SE'])
        self.assertFalse(runner.mass_comparison(zero, zero)['passed'])
        shifted = runner.mass_statistics([None if x is None else x+1000 for x in logs])
        self.assertAlmostEqual(shifted['log_Q'], result['log_Q']+1000)
        self.assertAlmostEqual(shifted['population_relative_SE'], expected)
        for malformed in ([0], [0,float('nan')], [0,float('inf')]):
            with self.assertRaises(ValueError): runner.mass_statistics(malformed)

    def test_both_comparison_limits_are_required(self):
        def item(log,se): return dict(log_Q=log,population_relative_SE=se)
        significant = runner.mass_comparison(item(.15,.01),item(0,.01))
        absolute = runner.mass_comparison(item(.3,.2),item(0,.2))
        passed = runner.mass_comparison(item(.1,.03),item(0,.03))
        self.assertFalse(significant['passed']); self.assertTrue(significant['within_absolute_limit'])
        self.assertFalse(absolute['passed']); self.assertTrue(absolute['within_three_SE'])
        self.assertTrue(passed['passed'])
        # A delta-method log-SE test would pass here; the linear mass test fails.
        nonlinear = runner.mass_comparison(item(.18,0),item(0,.065))
        self.assertLess(.18,3*.065)
        self.assertFalse(nonlinear['within_three_SE'])

    def test_aggregate_preserves_all_strata_and_hard_only_estimates(self):
        populations=[]
        for i in range(4):
            classes = dict(total=math.log(i+1),contact_no_native_entry=math.log(i+1),unbound_no_native_entry=None)
            bins = {f: {k: [value]+[None]*(size-1) for k,value in classes.items()}
                    for f,size in runner.STRATA.items()}
            populations.append(dict(terminal=dict(log_masses=classes,log_strata=bins),
                initial_physical_hard_log_masses={k: None if v is None else v-10 for k,v in classes.items()}))
        result = runner.aggregate(populations)
        self.assertEqual(set(result['estimates']),set(runner.CLASSES))
        for f,size in runner.STRATA.items():
            self.assertEqual(len(result['strata'][f]),size)
            self.assertIsNone(result['strata'][f][-1]['total']['log_Q'])
        self.assertAlmostEqual(result['Q0']['total']['log_Q'],result['estimates']['total']['log_Q']-10)
        with self.assertRaises(ValueError): runner.aggregate(populations[:-1])

    def reference(self, root):
        populations=[]
        for ident in [f'z{z}-r{i:02d}' for z in (0,4) for i in range(8)]+['zero']:
            path=root/(ident+'.json'); runner.write(path,dict(id=ident))
            populations.append(dict(id=ident,audit=str(path),sha256=runner.sha(path)))
        receipt=dict(schema='native-excluded-smc-audited-reference-v1', complete=True,
            reference_tolerance_passed=True,populations=populations,files={},auditor_sources={'audit.py':'f'*64})
        path=root/'receipt.json'; runner.write(path,receipt)
        return path,receipt

    def test_reference_gate_binds_all_populations_and_refuses_missing_failed_or_mutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,receipt=self.reference(root)
            runner.validate_reference(path,runner.sha(path),runner.Ledger())
            for field,value in [('complete',False),('reference_tolerance_passed',False),('populations',receipt['populations'][:-1]),('auditor_sources',{})]:
                changed=copy.deepcopy(receipt); changed[field]=value; runner.write(path,changed)
                with self.assertRaises(ValueError): runner.validate_reference(path,runner.sha(path),runner.Ledger())
            runner.write(path,receipt)
            Path(receipt['populations'][0]['audit']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):
                runner.validate_reference(path,runner.sha(path),runner.Ledger())

    def validated_fixture(self, root):
        (root/'common').mkdir()
        (root/'common/latent-region-smc').write_text('dummy executable')
        (root/'common/source-bundle.json').write_text('dummy source')
        plan=dict(schema=runner.SCHEMA,allocation=runner.allocation(),jobs=runner.jobs_for(root),
            thread_environment=runner.THREAD_ENV,maximum_physical_workers=8,maximum_total_workers=32,
            audit_workers=4,production_authorized=False,python=sys.executable,python_version=sys.version,
            python_sha256=runner.sha(sys.executable),sources={Path(runner.__file__).name:runner.sha(runner.__file__)},
            binary_sha256=runner.sha(root/'common/latent-region-smc'),
            source_bundle_sha256=runner.sha(root/'common/source-bundle.json'))
        self.reseal(root,plan)
        return plan

    def reseal(self, root, plan):
        runner.write(root/'plan.json',plan)
        runner.write(root/'freeze.json',dict(files={str(p.relative_to(root)):runner.sha(p)
            for p in root.rglob('*') if p.is_file() and p.name!='freeze.json'}))

    def test_validate_rejects_tampered_or_semantically_resealed_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan=self.validated_fixture(root)
            runner.validate(root,runner.sha(root/'plan.json'),check_upstream=False)
            for field,value in [('maximum_physical_workers',9),('maximum_total_workers',33),
                    ('audit_workers',5),('production_authorized',True),('python_sha256','f'*64)]:
                altered=copy.deepcopy(plan); altered[field]=value; self.reseal(root,altered)
                with self.assertRaises(ValueError): runner.validate(root,runner.sha(root/'plan.json'),check_upstream=False)
            altered=copy.deepcopy(plan); altered['jobs'][0]['seed']+=1; self.reseal(root,altered)
            with self.assertRaises(ValueError): runner.validate(root,runner.sha(root/'plan.json'),check_upstream=False)
            self.reseal(root,plan); (root/'common/latent-region-smc').write_text('tampered')
            with self.assertRaises(ValueError): runner.validate(root,runner.sha(root/'plan.json'),check_upstream=False)

    def test_authenticate_report_rejects_resealed_wrong_target_and_partial_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); job=runner.jobs_for(root)[0]; target=root/'audits'/job['id']
            config=dict(fixed_poses=[dict(position=[0,0,0],orientation=[1,0,0,0])])
            runner.write(root/'inputs'/'narrow-config.json',config)
            report=dict(schema='native-excluded-smc-population-audit-v1',complete=True,
                id=job['id'],seed=job['seed'],initial_draws=runner.INITIAL_DRAWS,
                native_definition=dict(definition_sha256=runner.DEFINITION_SHA,
                    compiled_sha256=runner.COMPILED_SHA,shape_sha256=runner.SHAPE_SHA,
                    fixed_poses=config['fixed_poses']),independent_analytic_test_adapter=False,
                source_sha256={},label_sha256={},zero_estimate=False,terminal=dict(particles=2048))
            def save(value,complete=True):
                runner.write(target/'population.json',value)
                runner.write(target/'status.json',dict(complete=complete,phase='complete',
                    population_sha256=runner.sha(target/'population.json')))
                runner.write(target/'freeze.json',dict(files={name:runner.sha(target/name)
                    for name in ('population.json','status.json')}))
            save(report); runner.authenticated_report(root,{},job,runner.Ledger())
            for field,value in [('seed',1),('initial_draws',1),('independent_analytic_test_adapter',True),
                                ('native_definition',{}),('terminal',dict(particles=1024))]:
                altered=copy.deepcopy(report); altered[field]=value; save(altered)
                with self.assertRaises(ValueError): runner.authenticated_report(root,{},job,runner.Ledger())
            save(report,complete=False)
            with self.assertRaises(ValueError): runner.authenticated_report(root,{},job,runner.Ledger())
            save(report); (target/'population.json').write_text('changed')
            with self.assertRaises(ValueError): runner.authenticated_report(root,{},job,runner.Ledger())

    def test_freeze_refuses_existing_output_before_touching_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'keep').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Fresh control'):
                runner.freeze(root,root,'absent','f'*64)
            self.assertEqual((root/'keep').read_text(),'preserve')

    def test_failed_physical_group_prevents_all_audits_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'common').mkdir(); script=root/'common'/Path(runner.__file__).name
            plan=dict(jobs=runner.jobs_for(root),python=sys.executable,repository=str(root))
            ledger=mock.Mock()
            def fail(steps,snapshot,workers,repository,env,**kw):
                self.assertEqual(workers,8)
                self.assertEqual(len(steps),12)
                self.assertTrue(all(s['kind']=='physical' for s in steps))
                self.assertEqual(env['OPENBLAS_NUM_THREADS'],'1')
                steps[0]['status']='failed'
                for step in steps[1:]:step['status']='not_started'
                snapshot(); raise RuntimeError('injected physical failure')
            with mock.patch.object(runner,'__file__',str(script)), \
                    mock.patch.object(runner,'validate',return_value=(plan,ledger)), \
                    mock.patch.object(runner,'execute_group',side_effect=fail) as execute:
                with self.assertRaisesRegex(RuntimeError,'injected physical failure'): runner.run(root,'f'*64)
            self.assertEqual(execute.call_count,1)
            state=runner.read(root/'status.json')
            self.assertFalse(state['complete']);self.assertEqual(state['phase'],'failed')
            self.assertTrue(all(x['status']=='pending' for x in state['audits']))
            self.assertTrue((root/'claim.json').exists())
            self.assertFalse((root/'analysis.json').exists())


if __name__=='__main__': unittest.main()
