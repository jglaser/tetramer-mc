"""Synthetic completed receipts only; journal/cache bytes are inert and unread."""
from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bind_conditional_native_audit as binder


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
    return dict(path=str(path.resolve()), sha256=binder.sha(path))


def opaque(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'These synthetic bytes are deliberately not JSON rows.\n')
    return dict(path=str(path.resolve()), sha256=binder.sha(path))


def pose(x):
    return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


class Fixture:
    def __init__(self, root):
        self.root = root
        self.shape = save(root/'shape.json', dict(atoms=[]))
        self.frame = save(root/'frame.json', dict(poses=[pose(i) for i in range(264)]))
        self.contexts = [dict(root=2*c, child=2*c+1, anchor=100+c) for c in range(4)]
        physical = dict(depletant_radius=1.4, activity=.0275, lambda_ratio=64., wall_radius=593.742500239952)
        old_config = dict(output=str(root/'old-campaign/execution'), contexts=self.contexts,
                          physical=physical, source_frame=self.frame, shape=self.shape)
        self.old_config = save(root/'old-campaign/config.json', old_config)
        self.starts = []
        for c, s in itertools.product(range(4), range(4)):
            ref = save(root/f'starts/{c}-{s}.json', dict(status='prepared', context_index=c, stream=s,
                is_equilibrium_sample=False, selected=[pose(2*c), pose(2*c+1)]))
            self.starts.append(dict(context_index=c, stream=s, record=ref, status='prepared'))
        self.prepared = save(root/'prepared.json', dict(complete=True, passed=True,
            config_sha256=self.old_config['sha256'], alternative_starts=self.starts))
        self.prepared_files = {s['record']['path']: s['record']['sha256'] for s in self.starts}
        self.old_run = save(root/'old-campaign/run-binding.json', dict(config_sha256=self.old_config['sha256'],
            prepared_manifest=self.prepared, prepared_files=self.prepared_files))
        old_chains, old_inputs = self.chains(root/'old-campaign', self.old_config, self.old_run,
                                            ('local', 'unguided', 'm4'))
        self.old_contact = root/'old-contact'
        self.old_refs = self.contact(self.old_contact, root/'old-campaign', old_chains, old_inputs, singleton=False)
        new_config = dict(old_config, output=str(root/'new-campaign/execution'),
            control_analysis={k: self.old_refs[k] for k in ('summary', 'manifest', 'analysis')},
            inherited_campaign=dict(config=self.old_config, prepared_manifest=self.prepared))
        self.new_config = save(root/'new-campaign/config.json', new_config)
        self.new_run = save(root/'new-campaign/run-binding.json', dict(config_sha256=self.new_config['sha256'],
            prepared_manifest=self.prepared, prepared_files=self.prepared_files))
        new_chains, new_inputs = self.chains(root/'new-campaign', self.new_config, self.new_run,
                         ('singleton_two_neighbor', 'singleton_two_neighbor_unfused'))
        for chain in new_chains: chain['reused_control'] = False
        controls = [dict(copy.deepcopy(chain), reused_control=True, new_geometry_queries=0)
                    for chain in old_chains if chain['job']['arm'] in ('local', 'm4')]
        self.new_contact = root/'new-contact'
        self.new_refs = self.contact(self.new_contact, root/'new-campaign', new_chains+controls,
                                     new_inputs, singleton=True)
        self.initial_root = root/'initial-audit'
        self.definition = save(root/'native/definition.json', {})
        self.compiled = save(root/'native/compiled.json', {})
        self.witness = save(root/'native/witness.json', {})
        native_inputs = {ref['path']: ref['sha256'] for ref in
            (self.definition, self.compiled, self.witness, self.frame, self.shape)}
        starts = [dict(context_index=s['context_index'], stream=s['stream'],
                       members=[2*s['context_index'], 2*s['context_index']+1], record=s['record'])
                  for s in self.starts]
        self.initial_plan = dict(schema=binder.initial_audit.PLAN_SCHEMA, config=self.new_config,
            definition=self.definition, compiled_native=self.compiled, witness=self.witness,
            shape=self.shape, source_frame=self.frame, contexts=self.contexts, starts=starts,
            allocation=binder.initial_audit.ALLOCATION, setup_inventory=dict(observer_instances=1),
            input_sha256=native_inputs, source_sha256={'synthetic.py': '1'*64}, runtime={'synthetic': True})
        plan_ref = save(self.initial_root/'audit-plan.json', self.initial_plan)
        keys = [[0, 2, 7]]
        states = [dict(state_id='source', reference_pair_calls=34716, reference_native_keys=keys,
                       instantaneous_native_keys=keys, passed=True, second_classifier_pass=False)]
        states += [dict(state_id=f'context-{c}-stream-{s}', reference_pair_calls=525,
                        passed=True, second_classifier_pass=False) for c, s in itertools.product(range(4), range(4))]
        fixed = {str(c): [k for k in keys if not set(k[:2]) & {2*c, 2*c+1}] for c in range(4)}
        self.initial_result = dict(schema=binder.initial_audit.SCHEMA, complete=True, passed=True,
            plan_sha256=plan_ref['sha256'], allocation=binder.initial_audit.ALLOCATION,
            reference_queries_begun=43116, reference_queries_completed=43116,
            input_sha256=native_inputs, source_sha256=self.initial_plan['source_sha256'],
            runtime=self.initial_plan['runtime'], setup_inventory=self.initial_plan['setup_inventory'],
            states=states, fixed_native_keys_by_context=fixed,
            ledger=opaque(self.initial_root/'audit/attempts.jsonl'))
        execution = save(self.initial_root/'execution-plan.json', dict(jobs=[dict(id='initial-reference')]))
        self.initial_completion = dict(complete=True, passed=True, plan_sha256=execution['sha256'],
            active=None, unstarted=[], failure=None, completed=[])
        save(self.initial_root/'execution/claim.json', {})
        self.seal_initial()
        self.output = root/'future-audit'

    def chains(self, base, config, run, arms):
        chains = []; files = {r['path']: r['sha256'] for r in (config, run)}
        for identifier, (c, arm, initialization, s) in enumerate(itertools.product(range(4), arms, binder.STARTS, range(4))):
            job = dict(id=identifier, context_index=c, arm=arm, initialization=initialization, stream=s)
            directory = base/'execution'/f'job-{identifier:03}'
            trajectory = opaque(directory/'trajectory.jsonl')
            cpu = float(100+identifier)
            counts = dict(local_attempted=18432)
            terminal = save(directory/'terminal.json', dict(complete=True, conditional_target=True, job=job,
                blocks=4608, trajectory=trajectory, counts=counts, config_sha256=config['sha256'],
                binding_sha256=run['sha256'], cpu_seconds=cpu, geometry_load_cpu_seconds=2.))
            for ref in (trajectory, terminal): files[ref['path']] = ref['sha256']
            chains.append(dict(job=job, trajectory=trajectory, counts=counts, observer_cpu_seconds=3.,
                metrics=dict(full_sampler_cpu_seconds=cpu, production_samples=4096)))
        return chains, files

    def contact(self, root, base, chains, files, *, singleton):
        result = dict(schema='two-neighbor-singleton-analysis-v1' if singleton else 'conditional-dimer-analysis-v1',
                      complete=True, chains=chains, analysis_plan={}, costs={})
        result['input_sha256' if singleton else 'input_files'] = files
        result.update(dict(new_chains=64, reused_control_chains=64, new_geometry_endpoints=294976,
            new_physical_draws=0, old_geometry_queries=0, native_observer=False) if singleton else {})
        outputs = [c for c in chains if c['reused_control'] is False] if singleton else chains
        cache_files = {}
        for chain in outputs:
            ref = opaque(root/'analysis'/f"job-{chain['job']['id']:03}-observations.jsonl")
            cache_files[Path(ref['path']).name] = ref['sha256']
        input_binding = dict(input_sha256=files, plan={}) if singleton else {}
        input_ref = save(root/'analysis/input-binding.json', input_binding)
        cache_files['input-binding.json'] = input_ref['sha256']
        plan = dict(schema='two-neighbor-singleton-observer-execution-v1' if singleton else 'evolving-dimer-analysis-execution-v1',
                    root=str(root), base=str(base), files=files, maximum_workers=1, partial_analysis_allowed=False)
        if singleton:
            plan.update(new_chains=64, reused_control_chains=64, retries=0, replacements=0,
                new_jobs=[c['job'] for c in outputs], control_jobs=[c['job'] for c in chains if c['reused_control'] is True])
        else: plan.update(expected_chains=96, restart=False)
        plan_ref = save(root/'execution-plan.json', plan)
        save(root/'claim.json', dict(plan_sha256=plan_ref['sha256']))
        save(root/'exit.json', dict(child_started=True, child_drained=True, returncode=0, error=None))
        save(root/'analysis/manifest.json', dict(complete=True, files=cache_files))
        save(root/'summary.json', dict(complete=True, passed=True, native_observer=False,
             plan_sha256=plan_ref['sha256'], **({'schema':'two-neighbor-singleton-observer-completion-v1'} if singleton else {'chains':96})))
        return self.seal_contact(root, result, singleton=singleton)

    def seal_contact(self, root, result, *, singleton=True):
        analysis_ref = save(root/'analysis/analysis.json', result)
        manifest = binder.read(root/'analysis/manifest.json'); manifest['files']['analysis.json'] = analysis_ref['sha256']
        manifest_ref = save(root/'analysis/manifest.json', manifest)
        summary = binder.read(root/'summary.json'); summary['analysis'] = analysis_ref
        if singleton: summary['manifest'] = manifest_ref
        else: summary['manifest_sha256'] = manifest_ref['sha256']
        summary_ref = save(root/'summary.json', summary)
        return dict(summary=summary_ref, manifest=manifest_ref, analysis=analysis_ref)

    def seal_initial(self):
        self.initial_terminal = save(self.initial_root/'audit/summary.json', self.initial_result)
        self.initial_completion['completed'] = [dict(terminal=self.initial_terminal)]
        save(self.initial_root/'execution/summary.json', self.initial_completion)

    def patches(self):
        return mock.patch.multiple(binder.initial_audit, DEFINITION_SHA=self.definition['sha256'],
                                   SHAPE_SHA=self.shape['sha256'], WITNESS_SHA=self.witness['sha256'])


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Fixture(Path(self.temporary.name))
        for patcher in (self.fixture.patches(),
                        mock.patch.object(binder, 'source_closure', return_value={'synthetic.py': '2'*64}),
                        mock.patch.object(binder, 'runtime', return_value={'synthetic': True}),
                        mock.patch.object(binder.driver, 'completed_terminal', side_effect=lambda *_: self.fixture.initial_terminal),
                        mock.patch.object(binder.initial_audit, 'load_bounded_classifier', side_effect=AssertionError('No geometry'))):
            patcher.start(); self.addCleanup(patcher.stop)

    def make(self):
        f = self.fixture
        return binder.make_plan(f.new_contact, f.initial_root, root=f.output)

    def test_complete_128_inventory_all_original_journals_and_cpu_are_bound(self):
        plan = self.make()
        self.assertEqual(len(plan['chains']), 128)
        self.assertEqual(len({c['id'] for c in plan['chains']}), 128)
        self.assertEqual(plan['allocation']['retained_endpoints'], 589952)
        self.assertEqual(plan['allocation']['checkpoint_pair_query_cap'], 134400)
        self.assertEqual(plan['limits']['cpu_seconds'], 36000)
        self.assertEqual(plan['fixed_native_keys_by_context'], self.fixture.initial_result['fixed_native_keys_by_context'])
        self.assertFalse(self.fixture.output.exists())
        self.assertFalse(plan['launched'])
        self.assertEqual(plan, self.make())
        for chain in plan['chains']:
            self.assertEqual(plan['input_sha256'][chain['trajectory']['path']], chain['trajectory']['sha256'])
            self.assertEqual(plan['input_sha256'][chain['terminal']['path']], chain['terminal']['sha256'])
            self.assertGreater(chain['full_sampler_cpu_seconds'], 0)
            self.assertEqual(chain['geometry_load_cpu_seconds'], 2.)
            self.assertIn(chain['initial']['kind'], binder.STARTS)

    def test_checkpoint_selection_uses_only_identity_and_both_production_halves(self):
        plan = self.make()
        for chain in plan['chains']:
            a, b = chain['checkpoint_blocks']
            self.assertTrue(513 <= a <= 2560 < b <= 4608)
            self.assertEqual([a, b], binder.checkpoint_blocks(dict(chain['identity'], arbitrary_outcome='ignored')))
        self.assertEqual(sum(len(c['checkpoint_blocks']) for c in plan['chains']), 256)

    def test_contact_completion_is_required_even_with_completed_native_reference(self):
        path = self.fixture.new_contact/'summary.json'
        summary = binder.read(path); summary['complete'] = False; save(path, summary)
        with self.assertRaisesRegex(ValueError, 'Completed contact'):
            self.make()
        path.unlink()
        with self.assertRaises(FileNotFoundError): self.make()

    def test_failed_or_undrained_contact_observer_cannot_authorize(self):
        path = self.fixture.new_contact/'exit.json'; value = binder.read(path)
        value['child_drained'] = False; save(path, value)
        with self.assertRaisesRegex(ValueError, 'drain'): self.make()

    def test_repeated_chain_identity_is_rejected(self):
        root = self.fixture.new_contact; value = binder.read(root/'analysis/analysis.json')
        value['chains'][1] = copy.deepcopy(value['chains'][0]); self.fixture.seal_contact(root, value)
        with self.assertRaisesRegex(ValueError, 'Missing/repeated'): self.make()

    def test_old_journal_is_hashed_through_original_arithmetic_receipt(self):
        result = binder.read(self.fixture.old_contact/'analysis/analysis.json')
        chain = next(c for c in result['chains'] if c['job']['arm'] == 'local')
        Path(chain['trajectory']['path']).write_bytes(b'changed old journal; never decoded')
        with self.assertRaisesRegex(ValueError, 'Changed bound input'): self.make()

    def test_old_terminal_is_bound_and_old_job_ids_do_not_alias_new_ids(self):
        result = binder.read(self.fixture.old_contact/'analysis/analysis.json')
        chain = next(c for c in result['chains'] if c['job']['arm'] == 'm4')
        path = Path(chain['trajectory']['path']).parent/'terminal.json'
        value = binder.read(path); value['cpu_seconds'] += 1; save(path, value)
        with self.assertRaisesRegex(ValueError, 'Changed bound input'): self.make()

    def test_full_cpu_must_match_terminal_and_production_allocation(self):
        root = self.fixture.new_contact; value = binder.read(root/'analysis/analysis.json')
        value['chains'][0]['metrics']['full_sampler_cpu_seconds'] += 1
        self.fixture.seal_contact(root, value)
        with self.assertRaisesRegex(ValueError, 'Full CPU'): self.make()

    def test_copied_control_cannot_change_its_original_job_or_cost(self):
        root = self.fixture.new_contact; value = binder.read(root/'analysis/analysis.json')
        chain = next(c for c in value['chains'] if c['reused_control'])
        chain['metrics']['full_sampler_cpu_seconds'] += 1
        self.fixture.seal_contact(root, value)
        with self.assertRaisesRegex(ValueError, 'Copied control'): self.make()

    def test_source_preparation_records_are_still_bound(self):
        path = Path(self.fixture.starts[0]['record']['path'])
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'Changed bound input'): self.make()

    def test_initial_reference_must_be_complete_and_exactly_43116_calls(self):
        self.fixture.initial_result['reference_queries_completed'] -= 1
        self.fixture.seal_initial()
        with self.assertRaisesRegex(ValueError, 'exhaustive'): self.make()

    def test_fixed_source_keys_cannot_be_selected_or_replaced(self):
        self.fixture.initial_result['fixed_native_keys_by_context']['3'] = []
        self.fixture.seal_initial()
        with self.assertRaisesRegex(ValueError, 'fixed source keys'): self.make()

    def test_false_contact_outcomes_do_not_select_away_any_chain(self):
        root = self.fixture.new_contact; value = binder.read(root/'analysis/analysis.json')
        for chain in value['chains']:
            chain['metrics']['descriptive_failure'] = dict(exchanges=0, no_mixing=True)
        self.fixture.seal_contact(root, value)
        self.assertEqual(len(self.make()['chains']), 128)


if __name__ == '__main__': unittest.main()
