"""Synthetic benchmark contracts and retained records; never launch sampling."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from contact_benchmark_contract import (ATTEMPTS, COMMON, PROPOSAL, SCHEMA,
    digest, validate_comparison, validate_contract, validate_report)
from analyze_contact_efficiency import (SCHEMA as REPORT_SCHEMA, analyze,
    compare_reports, physical_identity, save, sha, validate_frames)
from test_analyze_contact_efficiency import DEFINITIONS, file_fixture, fixture


def contract_fixture():
    physical = {'bodies': 3, 'target': 'synthetic'}
    common = dict(local_translation_std_A=.2, local_small_angle_std_degrees=1.,
        gca_probability=.1, center_shift_probability=.1, poisson_lambda_ratio=64.,
        endpoint_gate=dict(max_cells=2047, max_depth=14, min_width=.5))
    def arm(role, probability, model, posterior=None):
        return dict(role=role, global_probability=probability,
            method='local-uniform' if role == 'local' else 'learned',
            model_sha256=model, frozen_posterior=posterior, learned_uniform_weight=.1)
    contract = dict(schema=SCHEMA, physical_identity_sha256=digest(physical),
        observer_definition_sha256='b'*64, region_definition_sha256='c'*64,
        common_schedule=common, window=dict(burn_sweep=0, end_sweep=8, cadence_sweeps=1),
        preparations=['dispersed'], streams_per_preparation=4,
        arms=dict(local=arm('local', 0., None), redraw=arm('independent-redraw', .5, 'a'*64),
            transport=arm('correlated-transport', .5, 'a'*64, dict(probability=.9, correlation=.9))),
        single_body_attempt_budget=ATTEMPTS)
    reports = []
    for name, spec in contract['arms'].items():
        for stream in range(4):
            reports.append(dict(schema=REPORT_SCHEMA, complete=True, physical_identity=physical,
                observer_definition_sha256=contract['observer_definition_sha256'],
                region_definition_sha256=contract['region_definition_sha256'],
                measurement_identity={}, implementation_sha256={'synthetic': 'd'*64},
                schedule=common | {'global_probability': spec['global_probability']},
                proposal={k: spec[k] for k in PROPOSAL}, window=contract['window'],
                window_attempts=dict(local=24 if name == 'local' else 12,
                    **{'global': 0 if name == 'local' else 12}),
                initialization=dict(proposal_arm=name, preparation_id='dispersed',
                    master_seed=1000+len(reports), invocation_initial_sweep=0, resume=None,
                    initial_poses_sha256='e'*64),
                fingerprint=dict(apparent_ess_per_sampling_CPU_second=2.+stream),
                environment_occupancies=dict(A=dict(fraction=.1+.01*stream),
                    remaining=dict(fraction=.9-.01*stream))))
    bind(reports, contract)
    return contract, reports


def bind(reports, contract):
    for report in reports:
        report['benchmark_contract'] = dict(path='/synthetic/contract.json', sha256=digest(contract),
            content=copy.deepcopy(contract), content_sha256=digest(contract))
        report['dependency_sha256'] = {'/synthetic/contract.json': digest(contract)}


class BenchmarkContractTests(unittest.TestCase):
    def test_three_arms_have_four_independent_streams_and_common_controls(self):
        contract, reports = contract_fixture()
        result = compare_reports(reports)
        self.assertEqual(len(result['groups']), 3)
        self.assertTrue(all(g['populations'] == 4 for g in result['groups']))
        self.assertEqual(result['benchmark_contract']['common_schedule'], contract['common_schedule'])
        self.assertAlmostEqual(result['groups'][0]['region_occupancies']['A']['mean'], .115)

    def test_legacy_uncontracted_comparison_still_requires_equal_schedule(self):
        _, reports = contract_fixture()
        for report in reports: report.pop('benchmark_contract')
        with self.assertRaisesRegex(ValueError, 'match schedule'): compare_reports(reports)

    def test_changed_local_collective_noise_or_cost_control_rejected(self):
        for key in COMMON:
            contract, reports = contract_fixture()
            reports[0]['schedule'] = copy.deepcopy(reports[0]['schedule'])
            value = reports[0]['schedule'][key]
            reports[0]['schedule'][key] = dict(value, max_cells=10) if isinstance(value, dict) else value+.01
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'common local'):
                validate_comparison(reports)

    def test_redraw_and_transport_share_model_floor_and_frequency(self):
        for key, value in [('model_sha256', 'd'*64), ('learned_uniform_weight', .2), ('global_probability', .4)]:
            contract, _ = contract_fixture(); contract['arms']['transport'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'share global frequency'):
                validate_contract(contract)

    def test_zero_correlation_posterior_is_not_independent_capture_redraw(self):
        contract, _ = contract_fixture()
        contract['arms']['redraw']['frozen_posterior'] = dict(probability=.9, correlation=0.)
        with self.assertRaisesRegex(ValueError, 'Independent redraw'): validate_contract(contract)
        contract, _ = contract_fixture(); contract['arms']['transport']['frozen_posterior']['correlation'] = 0.
        with self.assertRaisesRegex(ValueError, 'nonzero correlation'): validate_contract(contract)

    def test_involutive_transport_and_minimal_tree_budget_allowed(self):
        for rho in (-1., 1.):
            contract, _ = contract_fixture()
            contract['arms']['transport']['frozen_posterior']['correlation'] = rho
            contract['common_schedule']['endpoint_gate'].update(max_depth=0, min_width=0.)
            validate_contract(contract)

    def test_missing_extra_paired_or_undeclared_populations_fail(self):
        for mode in ('missing', 'extra', 'paired', 'undeclared'):
            _, reports = contract_fixture()
            if mode == 'missing': reports.pop()
            elif mode == 'extra': reports.append(copy.deepcopy(reports[0]))
            elif mode == 'paired': reports[-1]['initialization']['master_seed'] = reports[0]['initialization']['master_seed']
            else: reports[-1]['initialization']['preparation_id'] = 'unplanned'
            with self.subTest(mode=mode), self.assertRaises(ValueError): validate_comparison(reports)

    def test_attempt_budget_and_pure_local_control_are_observed(self):
        contract, reports = contract_fixture(); report = reports[0]
        report['window_attempts'] = dict(local=23, **{'global': 1})
        with self.assertRaisesRegex(ValueError, 'Pure-local'): validate_report(report, contract)
        report['window_attempts'] = dict(local=23, **{'global': 0})
        with self.assertRaisesRegex(ValueError, 'attempt budget'): validate_report(report, contract)

    def test_resume_changed_target_and_unbound_contract_fail(self):
        for mode in ('resume', 'target', 'binding', 'mixed'):
            _, reports = contract_fixture()
            if mode == 'resume': reports[0]['initialization']['resume'] = 'checkpoint.json'
            elif mode == 'target': reports[0]['physical_identity'] = {'bodies': 4}
            elif mode == 'binding': reports[0]['dependency_sha256'] = {}
            else: reports[0]['benchmark_contract'] = None
            with self.subTest(mode=mode), self.assertRaises(ValueError): validate_comparison(reports)

    def test_file_observer_binds_contract_and_records_all_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run, plan_path = file_fixture(root)
            config = json.loads((run/'config.json').read_text()); config['global_probability'] = 0.
            save(run/'config.json', config)
            declared = json.loads((run/'provenance/input-config.json').read_text()); declared['global_probability'] = 0.
            save(run/'provenance/input-config.json', declared)
            for name in ('manifest', 'summary', 'checkpoint'):
                value = json.loads((run/(name+'.json')).read_text())
                value['config_sha256'] = sha(run/'provenance/input-config.json')
                save(run/(name+'.json'), value)
            contract, _ = contract_fixture()
            manifest = json.loads((run/'manifest.json').read_text())
            contract['physical_identity_sha256'] = digest(physical_identity(config, manifest['shape_sha256']))
            contract['region_definition_sha256'] = digest(DEFINITIONS)
            contract['observer_definition_sha256'] = digest(dict(environments=DEFINITIONS,
                patch_map_sha256=sha(root/'patch.json'), native_definition_sha256=None))
            contract['common_schedule'] = {k: config[k] for k in COMMON}
            contract['preparations'] = ['synthetic']
            contract['arms']['local']['learned_uniform_weight'] = config['learned_uniform_weight']
            save(root/'contract.json', contract)
            plan = json.loads(plan_path.read_text()); plan.update(proposal_arm='local',
                benchmark_contract=dict(path='contract.json', sha256=sha(root/'contract.json')))
            save(plan_path, plan)
            report = analyze(run, plan_path, root/'out')
            self.assertEqual(report['window_attempts'], dict(local=24, **{'global': 0}))
            self.assertEqual(report['benchmark_contract']['content'], contract)
            self.assertEqual(report['dependency_sha256'][str(root/'contract.json')], sha(root/'contract.json'))

    def test_bias_window_opt_in_does_not_relax_other_auxiliary_restrictions(self):
        def biased():
            c, frames, s, m, cp = fixture()
            c['assembly_bias'] = m['assembly_bias'] = dict(values=[0., 1., 2.])
            for frame in frames: frame['assembly_bias'] = dict(largest_component_size=2, bias=1., log_reweight=1.)
            return c, frames, s, m, cp
        c, f, s, m, cp = biased()
        with self.assertRaises(ValueError): validate_frames(f, c, s, m, cp, 0, 8)
        validate_frames(f, c, s, m, cp, 0, 8, allow_frozen_assembly_bias=True)
        for mode in ('missing', 'table', 'adaptive'):
            c, f, s, m, cp = biased()
            if mode == 'missing': f[2].pop('assembly_bias')
            elif mode == 'table': m['assembly_bias'] = dict(values=[0., 0., 0.])
            else: c['atlas_transport'] = dict(enabled=True)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_frames(f, c, s, m, cp, 0, 8, allow_frozen_assembly_bias=True)


if __name__ == '__main__': unittest.main()
