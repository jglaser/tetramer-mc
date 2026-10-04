"""Synthetic journals/contact caches only; never query protein geometry."""
import copy
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_flexible_surrogate_benchmark as a
import test_analyze_rigid_surrogate_benchmark as old_fixture

pose, gate = old_fixture.pose, old_fixture.gate


def fixture(arm=a.ARMS[1], initialization='source'):
    config, job, initial, _, _, bank, metadata, prepared, start = old_fixture.fixture(
        a.rigid.ARMS[a.ARMS.index(arm)], initialization)
    config['flexible_surrogate_policy'] = dict(config.pop('surrogate_policy'), schema='flexible-surrogate-policy-v1')
    config['limits'] = dict(raw_per_leg=1000, retained_per_leg=1000, raw_per_outer=1000,
        retained_per_outer=1000, raw_campaign=10000, retained_campaign=10000, cpu_seconds=1000.)
    job['arm'] = arm; contract = a.surrogate_contract(config, job, bank, metadata)
    settings = contract['effective_config']; selected = copy.deepcopy(initial)
    rows = [dict(kind='initial', block=0, job=job, selected=copy.deepcopy(initial), conditional_target=True,
        sampler_cpu_seconds=0., fixed_source=config['source_frame'], cloud=bank, cloud_cpu_seconds=.01,
        prepared_manifest=prepared, prepared_start=start, flexible_surrogate_contract=contract)]
    counts = dict(local_attempted=0, local_accepted=0, dimer_attempted=0, dimer_accepted=0, dimer_self_loop=0)
    raw = retained = 0
    for block in range(1, 5):
        for attempt, slot in enumerate((0, 1, 0, 1)):
            local = dict(kind='local', block=block, attempt=attempt, member=(9, 24)[slot],
                old=copy.deepcopy(selected[slot]), proposed=pose(999), accepted=False, status='hard_rejected',
                sampler_cpu_seconds=float(len(rows)))
            # Exercise the source after an accepted local, not just a chain of
            # local rejections followed by the initial outer source again.
            if block == 1 and attempt == 0:
                local.update(proposed=pose(1.1), accepted=True, status='completed', bath=gate(),
                             log_u=-1., log_acceptance_ratio=0.)
                selected[slot] = local['proposed']; counts['local_accepted'] += 1
            local['retained'] = copy.deepcopy(selected); rows.append(local)
            counts['local_attempted'] += 1
        old = copy.deepcopy(selected); current = copy.deepcopy(old)
        old_score = old_fixture.score(old, contract, config); current_score = old_score
        before = dict(raw=raw, retained=retained)
        rows.append(dict(kind='flexible_surrogate_attempt_begun', status='begun', block=block,
            members=[9, 24], config=settings, old=copy.deepcopy(old), raw=raw,
            bath_retained=retained, sampler_cpu_seconds=float(len(rows))))
        steps = []; inner_counts = dict(attempted=0, accepted=0, hard_rejected=0, mh_rejected=0)
        for index in range(settings['inner_steps']):
            slot = index % 2; target = copy.deepcopy(current)
            target[slot] = pose(float(block+1) if index == 0 else current[slot]['position'][0]+1.)
            step = dict(index=index, old=copy.deepcopy(current), selected_slot=slot, selected_label=(9, 24)[slot],
                old_score=current_score['log_surrogate'], proposed=target, accepted=False, status='hard_rejected')
            inner_counts['attempted'] += 1
            if index == 0 and block != 3:
                candidate_score = old_fixture.score(target, contract, config)
                delta = candidate_score['log_surrogate']-current_score['log_surrogate']
                log_u = -.0001 if block == 4 else -10.; accepted = log_u < min(0., delta)
                step.update(status='completed', proposed_score=candidate_score, log_acceptance_ratio=delta,
                            log_u=log_u, accepted=accepted)
                inner_counts['accepted' if accepted else 'mh_rejected'] += 1
                if accepted: current, current_score = copy.deepcopy(target), candidate_score
            else:
                inner_counts['hard_rejected'] += 1
            step.update(retained=copy.deepcopy(current), retained_score=current_score['log_surrogate']); steps.append(step)
        correction = old_score['log_surrogate']-current_score['log_surrogate']
        outcome = dict(kind='flexible_surrogate_chain', block=block, members=[9, 24], selection_probabilities=[.5, .5],
            config=settings, old=old, old_score=old_score, proposed=current, proposed_score=current_score,
            steps=steps, inner_counts=inner_counts, budget_before=before, budget_after=copy.deepcopy(before),
            complete_log_correction=correction, accepted=False, physical_decisions=0,
            status='identity_self_loop', sampler_cpu_seconds=float(len(rows)))
        if current != old:
            leg = gate(64 if block == 2 else 0); legs = [leg, gate()]
            slots = [0, 1] if block % 2 else [1, 0]
            if slots[0] == 1: legs.reverse()
            aggregate = {k: legs[0][k]+legs[1][k] for k in leg}
            middle = copy.deepcopy(old); middle[slots[0]] = current[slots[0]]
            bath = dict(order='first_then_second' if slots[0] == 0 else 'second_then_first',
                        ordered_members=[(9, 24)[i] for i in slots], intermediate_selected=middle,
                        legs=legs, aggregate=aggregate)
            ratio = aggregate['log_weight']+correction; log_u = -.0001 if block == 2 else -10.
            accepted = log_u < min(0., ratio)
            raw += aggregate['raw_points']; retained += aggregate['retained_points']
            outcome.update(status='completed', physical_decisions=1, bath=bath, log_u=log_u,
                log_acceptance_ratio=ratio, accepted=accepted, budget_after=dict(raw=raw, retained=retained))
            if accepted: selected = copy.deepcopy(current)
        outcome['retained'] = copy.deepcopy(selected); rows.append(outcome)
        counts['dimer_attempted'] += 1; counts['dimer_accepted'] += outcome['accepted']
        counts['dimer_self_loop'] += outcome['status'] == 'identity_self_loop'
        rows.append(dict(kind='retained_block', block=block, production=block > 1, selected=copy.deepcopy(selected),
            counts=copy.deepcopy(counts), raw=raw, retained=retained, sampler_cpu_seconds=float(len(rows))))
    terminal = dict(complete=True, conditional_target=True, job=job, blocks=4, counts=counts,
        raw=raw, retained=retained, cpu_seconds=float(len(rows)))
    return config, job, initial, rows, terminal, bank, metadata, prepared, start


def check(data):
    c, j, initial, rows, terminal, bank, metadata, prepared, start = data
    with mock.patch.object(a.previous, 'ConditionalObserver', side_effect=AssertionError('No geometry')):
        return a.validate_journal(iter(rows), c, j, initial, terminal, bank, metadata, prepared, start)


def observation(block, tokens, external=True):
    tokens = list(tokens)+([(9, 71, 'p', 'q')] if external else [])
    edges = sorted({t[:2] for t in tokens})
    return dict(block=block, production=block > 1, patch_tokens=tokens, partner_edges=edges,
        internal_contact=(9, 24) in edges, external_edges=[e for e in edges if e != (9, 24)],
        fingerprint=a.previous.key(sorted(tokens)))


class JournalTests(unittest.TestCase):
    def test_all_arms_starts_local_source_and_all_residences(self):
        for arm, start in itertools.product(a.ARMS, a.STARTS):
            data = fixture(arm, start); original = copy.deepcopy(data)
            points = check(data)
            self.assertEqual(data, original); self.assertEqual(len(points), 5)
            self.assertEqual(data[3][5]['old'][0], pose(1.1))
            self.assertEqual(data[3][6]['steps'][0]['proposed'][1], data[3][6]['old'][1])
            self.assertEqual(points[1]['selected'], points[2]['selected'])
            self.assertEqual(points[2]['selected'], points[3]['selected'])
            self.assertEqual(points[-1]['counts']['local_accepted'], 1)

    def test_inner_contract_mutations(self):
        mutations = [
            lambda r: r['steps'].pop(),
            lambda r: r.update(selection_probabilities=[.6, .4]),
            lambda r: r['steps'][0].update(selected_label=0),
            lambda r: r['steps'][0].update(selected_slot=9),
            lambda r: r['steps'][0]['proposed'][1]['position'].__setitem__(0, 88.),
            lambda r: r['steps'][0]['old'][0]['position'].__setitem__(0, 88.),
            lambda r: r['steps'][0].update(accepted=False),
            lambda r: r['steps'][1].update(log_u=-1.),
            lambda r: r['steps'][0].update(retained_score=999.),
            lambda r: r['inner_counts'].update(attempted=9),
            lambda r: r['old_score'].update(twice_overlap_units=99),
            lambda r: r['old_score'].update(nearby_spectators=[263, 1]),
            lambda r: r['old_score'].update(log_surrogate=math.nan),
        ]
        for mutate in mutations:
            data = fixture(); mutate(data[3][6])
            with self.assertRaises(ValueError): check(data)

    def test_path_outer_budget_fatal_and_begun_mutations(self):
        mutations = [
            lambda r: r[6].update(complete_log_correction=-r[6]['complete_log_correction']),
            lambda r: r[6].update(physical_decisions=2),
            lambda r: r[6].update(accepted=False),
            lambda r: r[6]['bath'].update(ordered_members=[24, 9]),
            lambda r: r[6]['bath']['intermediate_selected'][1]['position'].__setitem__(0, 88.),
            lambda r: r[13]['bath']['aggregate'].update(raw_points=88),
            lambda r: r[13]['bath']['legs'][1].update(log_weight=0.),
            lambda r: r[13]['budget_after'].update(raw=88),
            lambda r: r[20].update(bath=gate()),
            lambda r: r[6].update(bath_failure={'failed_leg':1}),
            lambda r: r[6].update(fatal_error='budget exhausted'),
            lambda r: r[5].update(raw=1),
            lambda r: r[5].update(sampler_cpu_seconds=99.),
            lambda r: r.pop(5),
            lambda r: r.append(copy.deepcopy(r[5])),
        ]
        for mutate in mutations:
            data = fixture(); mutate(data[3])
            with self.assertRaises(ValueError): check(data)
        data = fixture(); data[0]['limits']['raw_per_leg'] = 0
        with self.assertRaises(ValueError): check(data)
        data = fixture(); data[0]['limits']['raw_per_leg'] = 0
        data[3][1]['bath'] = dict(gate(1), gained=1, raw_points=2, retained_points=2, log_weight=0.)
        with self.assertRaisesRegex(ValueError, 'Local completed bath exceeds cap'): check(data)
        data = fixture(); data[3][:] = data[3][:6]
        with self.assertRaises(ValueError): check(data)

    def test_identity_midpoint_and_score_contract(self):
        data = fixture(); data[3][0]['flexible_surrogate_contract']['point_volume'] *= 2
        with self.assertRaises(ValueError): check(data)
        data = fixture(); data[3][0]['flexible_surrogate_contract']['handle'] = 9
        with self.assertRaises(ValueError): check(data)
        # A copied midpoint can coincide exactly; the scalar path auditor must
        # validate the saved path without introducing a new hard-core filter.
        old = [pose(0), pose(1)]; proposed = old[::-1]
        bath = dict(order='first_then_second', ordered_members=[9, 24],
                    intermediate_selected=[old[1], old[1]], legs=[gate(2), gate(3)],
                    aggregate={k: gate(2)[k]+gate(3)[k] for k in gate()})
        a.validate_path(bath, old, proposed, [9, 24], data[0])

    def test_checkpoint_bytes_hash_selected_and_counter_closure(self):
        data = fixture(); points = check(data); terminal = data[4]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trajectory.jsonl'; path.write_text('synthetic bound bytes\n')
            terminal.update(config_sha256='config', binding_sha256='binding',
                            trajectory=dict(path=str(path), sha256=a.sha(path)))
            last = points[-1]
            cp = dict(schema='evolving-dimer-checkpoint-v1', job=data[1], block=4, selected=last['selected'],
                counts=last['counts'], raw=last['raw'], retained=last['retained'], config_sha256='config',
                binding_sha256='binding', journal_rows=29, journal_bytes=path.stat().st_size,
                journal_sha256=terminal['trajectory']['sha256'], cpu_seconds=terminal['cpu_seconds'])
            a.validate_checkpoint(cp, terminal, last, path, data[1])
            for field, value in (('journal_rows', 28), ('journal_bytes', 1), ('journal_sha256', 'other'),
                                  ('selected', [pose(99), pose(100)]), ('raw', 999)):
                changed = dict(cp, **{field:value})
                with self.assertRaises(ValueError): a.validate_checkpoint(changed, terminal, last, path, data[1])

    def test_inventory_and_lifecycle_gate_before_large_inputs(self):
        config = fixture()[0]; config['physical'] = copy.deepcopy(a.scalar.PHYSICAL)
        config['local'] = dict(translation_std_A=.2, rotation_std_degrees=1., member_order=[0, 1, 0, 1],
                               pair_contact_required=False)
        config['allocation'] = dict(warmup_blocks=512, production_blocks=4096)
        config['jobs'] = [dict(id=n, context_index=0, arm=arm, initialization=start, stream=stream)
            for n, (arm, start, stream) in enumerate(itertools.product(a.ARMS, a.STARTS, range(4)))]
        a.validate_inventory(config); config['jobs'].pop()
        with self.assertRaises(ValueError): a.validate_inventory(config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'execution').mkdir()
            (root/'execution-plan.json').write_text(json.dumps(dict(jobs=[{}]*24)))
            status = dict(complete=False, passed=False, failure=None, active='running', unstarted=[], completed=[], plan_sha256='x')
            for name in ('status', 'summary'): (root/'execution'/f'{name}.json').write_text(json.dumps(status))
            with mock.patch.object(a, 'sha', side_effect=AssertionError('No hash before completion')):
                with self.assertRaises(ValueError): a.bind_complete_inputs(root, config)


class InternalMetricsTests(unittest.TestCase):
    def test_baseline_residence_nonempty_returns_and_external_separation(self):
        x, y = [(9, 24, 'a', 'b')], [(9, 24, 'c', 'd')]
        values = [[], x, y, y, x, [], x]
        trace = [observation(i, v) for i, v in enumerate(values)]
        before = copy.deepcopy(trace)
        with mock.patch.object(a.previous, 'ConditionalObserver', side_effect=AssertionError('No geometry')):
            result = a.internal_metrics(trace, 1, 10., [9, 24])
        self.assertEqual(trace, before); self.assertEqual(result['production_samples'], 5)
        self.assertEqual(result['internal_contact_fraction'], .8)
        self.assertEqual((result['attachments'], result['detachments']), (1, 1))
        activity = result['patch_set_activity']
        self.assertEqual(activity['direct_nonempty_changes'], 2)
        self.assertEqual([r['passed_through_empty'] for r in activity['completed_nonempty_returns']], [False, True])
        self.assertEqual(activity['direct_nonempty_changes_per_full_CPU_second'], .2)
        self.assertEqual(sorted(r['fraction'] for r in result['patch_set_occupancy']), [.2, .4, .4])
        self.assertEqual(a.shared.external_activity(trace, 1, 10.)['direct_nonempty_changes'], 0)

    def test_constant_patch_set_ess_null_and_partition_mutations(self):
        trace = [observation(i, [(9, 24, 'a', 'b')]) for i in range(6)]
        result = a.internal_metrics(trace, 1, 10., [9, 24])
        self.assertIsNone(result['patch_set_ess']['apparent_ess'])
        self.assertEqual(result['patch_set_activity']['direct_nonempty_changes'], 0)
        for mutate in (lambda r:r.update(internal_contact=False), lambda r:r['patch_tokens'].append(r['patch_tokens'][0]),
                       lambda r:r.update(external_edges=[]), lambda r:r.update(block=99)):
            changed = copy.deepcopy(trace); mutate(changed[2])
            with self.assertRaises(ValueError): a.internal_metrics(changed, 1, 10., [9, 24])

    def test_comparisons_do_not_duplicate_histories(self):
        traces = [[observation(i, [(9, 24, 'a', 'b')] if i % 2 else []) for i in range(6)]]*2
        chains = [dict(job=dict(context_index=0, arm=arm, initialization='source', stream=0),
                       metrics=dict(internal_organization=a.internal_metrics(t, 1, 10., [9, 24])))
                  for arm, t in zip(a.ARMS[:2], traces)]
        fake = dict(groups=[dict(context_index=0, arm=c['job']['arm'], initialization='source',
                    independent_stream_metrics=[dict(stream=0)]) for c in chains],
                    descriptive_paired_comparisons=[dict(left=chains[0]['job'], right=chains[1]['job'])])
        with mock.patch.object(a.external, 'comparison_summaries', return_value=fake):
            result = a.comparison_summaries(chains)
        rendered = json.dumps(result)
        for forbidden in ('"patch_set_occupancy":', '"patch_tokens":', '"departure_block":', '"return_block":'):
            self.assertNotIn(forbidden, rendered)
        self.assertIn('completed_nonempty_returns', rendered)


def cache_fixture():
    root = Path('/synthetic/rigid-analysis')
    umbrella_path = root/'analysis/analysis.json'; umbrella_manifest = root/'analysis/manifest.json'
    origin_path = Path('/synthetic/old-analysis/analysis.json'); origin_manifest = origin_path.with_name('manifest.json')
    ref = lambda path: dict(path=str(path), sha256='hash:'+str(path))
    documents = {}; chains = []; old_chains = []; entries = []; umbrella_inputs = {}; old_inputs = {}
    new_files = {'analysis.json':ref(umbrella_path)['sha256'], 'input-binding.json':ref(root/'analysis/input-binding.json')['sha256']}
    old_files = {'analysis.json':ref(origin_path)['sha256']}
    for arms, inherited in ((a.rigid.ARMS, False), (('local', 'm4'), True)):
        analysis_path = origin_path if inherited else umbrella_path
        manifest_path = origin_manifest if inherited else umbrella_manifest
        for job_id, (arm, start, stream) in enumerate(itertools.product(arms, a.STARTS, range(4))):
            job = dict(id=job_id, context_index=0, arm=arm, initialization=start, stream=stream)
            observation = analysis_path.parent/f'job-{job_id:03}-observations.jsonl'
            terminal_path = analysis_path.parent/f'job-{job_id:03}'/'terminal.json'
            trajectory = ref(terminal_path.with_name('trajectory.jsonl'))
            chain = dict(job=job, counts={'synthetic':0}, trajectory=trajectory,
                         metrics={'full_sampler_cpu_seconds':10.})
            chains.append(chain)
            if inherited: old_chains.append(chain)
            documents[str(terminal_path)] = dict(complete=True, conditional_target=True, job=job,
                counts=chain['counts'], trajectory=trajectory, blocks=4608, cpu_seconds=10.)
            entry = dict(job=job, observation=ref(observation), manifest=ref(manifest_path),
                         terminal=ref(terminal_path), analysis=ref(analysis_path))
            entries.append(entry)
            (old_inputs if inherited else umbrella_inputs)[str(terminal_path)] = ref(terminal_path)['sha256']
            (old_files if inherited else new_files)[observation.name] = ref(observation)['sha256']
            if inherited:
                for value in (entry['observation'], entry['analysis'], entry['manifest']):
                    umbrella_inputs[value['path']] = value['sha256']
    umbrella = dict(schema='rigid-surrogate-analysis-v1', complete=True, chains=chains, input_sha256=umbrella_inputs)
    documents[str(umbrella_path)] = umbrella
    documents[str(umbrella_manifest)] = dict(complete=True, files=new_files)
    documents[str(root/'analysis/input-binding.json')] = dict(input_sha256=copy.deepcopy(umbrella_inputs))
    documents[str(origin_path)] = dict(schema='conditional-dimer-analysis-v1', complete=True,
                                      chains=old_chains, input_files=old_inputs)
    documents[str(origin_manifest)] = dict(complete=True, files=old_files)
    config = dict(control_analysis=dict(analysis=ref(umbrella_path), summary=ref(root/'execution/summary.json'),
        manifest=ref(umbrella_manifest), input_binding=ref(root/'analysis/input-binding.json'), authority={}, observations=entries))
    def bind(value):
        path = Path(value['path']).resolve()
        a.require(value['sha256'] == 'hash:'+str(path), 'Synthetic binding mismatch')
        return path
    return config, documents, umbrella, bind


class CachedControlTests(unittest.TestCase):
    def run_cache(self, data):
        config, documents, umbrella, bind = data
        with mock.patch.object(a.authority, 'authenticate', return_value=(umbrella, {})), \
             mock.patch.object(a, 'read', side_effect=lambda path: documents[str(Path(path).resolve())]), \
             mock.patch.object(a.previous, 'ConditionalObserver', side_effect=AssertionError('No old geometry')):
            return a.bind_cached_controls(config, bind)

    def test_full40_unique_identities_despite_overlapping_job_ids(self):
        data = cache_fixture(); cached = self.run_cache(data)
        self.assertEqual(len(cached), 40)
        self.assertEqual(len({a.identity(c['job']) for c, _ in cached}), 40)
        self.assertLess(len({c['job']['id'] for c, _ in cached}), 40)

    def test_duplicate_wrong_chain_swap_and_missing_origin_bindings_fail(self):
        for defect in ('duplicate', 'swapped observation', 'missing inherited closure', 'wrong terminal', 'wrong manifest'):
            data = cache_fixture(); config, docs, umbrella, _ = data
            entries = config['control_analysis']['observations']
            if defect == 'duplicate': entries[1] = copy.deepcopy(entries[0])
            if defect == 'swapped observation': entries[0]['observation'], entries[1]['observation'] = entries[1]['observation'], entries[0]['observation']
            if defect == 'missing inherited closure':
                path = entries[-1]['observation']['path']
                del umbrella['input_sha256'][path]
                del docs[config['control_analysis']['input_binding']['path']]['input_sha256'][path]
            if defect == 'wrong terminal': docs[entries[-1]['terminal']['path']]['cpu_seconds'] = 11.
            if defect == 'wrong manifest':
                del docs[entries[-1]['manifest']['path']]['files'][Path(entries[-1]['observation']['path']).name]
            with self.subTest(defect=defect), self.assertRaises((ValueError, KeyError)):
                self.run_cache(data)


if __name__ == '__main__':
    unittest.main()
