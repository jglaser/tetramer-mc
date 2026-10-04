"""Five synthetic saved-record/binding tests; no real journal reads or geometry."""
import copy
import itertools
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import diagnose_trapped_singleton_gate as diagnostic


def pose(x): return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


def attempt(block=1, *, self_loop=False):
    slot = (block-1) % 2; moving = [9, 24][slot]; neighbors = [[24, 135], [9, 135]][slot]
    density = dict(log_full=-8., log_uniform=-17., log_learned='-inf')
    trial = dict(index=1, branch='learned', disposition='candidate',
        trace=dict(target=3, target_label=dict(kind='single', member=0, anchor=1, branch=3), target_latent=[0.]*6),
        proposed_pose=pose(1), density=density, feasibility=dict(wall_valid=True, spectator_core_collisions=[]))
    p = dict(status='candidate', trials=[trial], old_density=density, new_density=density,
        full_log_reverse_forward=-4., counts=dict(uniform=0, learned=1, geometric_rejections=0,
            hard_rejections=0, wall_rejections=0, candidates=1, fatal_trials=0))
    row = dict(kind='two_neighbor_singleton', block=block, member_slot=slot, member=moving, neighbors=neighbors,
               old=pose(0), proposed=pose(1), retained=[pose(0), pose(2)], proposal=p,
               status='completed', accepted=False, complete_log_correction=-4.,
               bath=dict(log_weight=-9., gained=2, lost=99), log_acceptance_ratio=-13., log_u=-.5)
    if self_loop:
        row['status'] = 'proposal_self_loop'; p['status'] = 'cap_exhausted'; p['new_density'] = None
        p['full_log_reverse_forward'] = None; trial['disposition'] = 'geometric_rejection'
        trial['feasibility'] = dict(wall_valid=False, spectator_core_collisions=[5])
        p['counts'].update(geometric_rejections=1, hard_rejections=1, wall_rejections=1, candidates=0)
        for name in ('proposed', 'bath', 'complete_log_correction', 'log_acceptance_ratio', 'log_u'): row.pop(name)
    return row


def rows(chain, blocks=2):
    value = [dict(kind='initial', block=0, job=chain['job'], conditional_target=True)]
    for block in range(1, blocks+1):
        value.extend(dict(kind='local', block=block, attempt=i) for i in range(4))
        value.extend([attempt(block), dict(kind='retained_block', block=block, production=block > 512)])
    return value


class GateDiagnosticTests(unittest.TestCase):
    def test_saved_factors_are_copied_without_replaying_density_bath_or_mh(self):
        row = attempt(); row['accepted'] = True
        # Deliberately inconsistent toy numbers establish that this extractor
        # depends on prior replay authority instead of silently repeating it.
        row['log_acceptance_ratio'] = -777.
        value = diagnostic.extract_attempt(row, chain_id='toy', members=[9, 24])
        self.assertEqual(value['terms']['proposal_log_ratio'], -4.)
        self.assertEqual(value['terms']['bath_log_factor'], -9.)
        self.assertEqual(value['terms']['combined_log_ratio'], -777.)
        self.assertTrue(value['accepted']); self.assertIsNone(value['source_label'])
        self.assertEqual(value['inner_trials'][0]['trace']['target_label']['anchor'], 1)
        self.assertEqual(row['proposal']['old_density']['log_learned'], '-inf')

    def test_self_loops_retries_and_missing_vs_negative_infinity_are_preserved(self):
        value = diagnostic.extract_attempt(attempt(self_loop=True), chain_id='toy', members=[9, 24])
        stats = diagnostic.Stratum(); stats.add(value); result = stats.result()
        self.assertEqual(result['counts']['proposal_self_loops'], 1)
        self.assertEqual(result['proposal_counts']['geometric_rejections'], 1)
        self.assertEqual(result['proposal_counts']['hard_rejections'], 1)
        self.assertEqual(result['proposal_counts']['wall_rejections'], 1)
        self.assertEqual(result['terms']['bath_log_factor']['missing'], 1)
        self.assertEqual(result['terms']['old_log_learned']['negative_infinity'], 1)
        self.assertIsNone(result['terms']['bath_log_factor']['finite_summary'])
        self.assertEqual(value['inner_trials'][0]['disposition'], 'geometric_rejection')

    def test_exact_record_and_member_cadence_including_missing_tail(self):
        chain = dict(id='toy', job=dict(id=2)); fixture = rows(chain)
        actual = list(diagnostic.iter_attempts(fixture, chain, blocks=2))
        self.assertEqual([(v['block'], v['member']) for v in actual], [(1, 9), (2, 24)])
        self.assertEqual(diagnostic.extract_attempt(attempt(512), chain_id='toy', members=[9, 24])['phase'], 'warmup')
        self.assertEqual(diagnostic.extract_attempt(attempt(513), chain_id='toy', members=[9, 24])['phase'], 'production')
        for mode in ('tail', 'extra', 'local', 'member', 'repeat'):
            with self.subTest(mode=mode):
                bad = copy.deepcopy(fixture)
                if mode == 'tail': bad.pop()
                elif mode == 'extra': bad.append(dict(kind='retained_block', block=3))
                elif mode == 'local': bad[1]['attempt'] = 2
                elif mode == 'member': bad[5]['member'] = 24
                else: bad[11]['block'] = 1
                with self.assertRaises(ValueError): list(diagnostic.iter_attempts(bad, chain, blocks=2))

    def test_factor_statistics_and_bounded_decoder_reject_malformed_values(self):
        value = diagnostic.summarize([None, -math.inf, -4., 0., 8.])
        self.assertEqual((value['attempts'], value['missing'], value['negative_infinity'], value['finite']), (5, 1, 1, 3))
        self.assertEqual(value['finite_summary']['quantiles']['0.5'], 0.)
        for bad in (float('inf'), float('nan'), True, 'NaN'):
            with self.assertRaises(ValueError): diagnostic.log_value(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'toy.jsonl'
            for data, maximum in (('{}', 100), ('{"x":NaN}\n', 100), ('{"x":10}\n', 3)):
                path.write_text(data)
                with self.assertRaises(ValueError): list(diagnostic.decoded_rows(path, maximum))

    def test_metadata_plan_binds_all_sixteen_journals_before_any_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base = root/'campaign'; base.mkdir()
            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True); diagnostic.write(path, value)
                return diagnostic.reference(path)
            config = dict(output=str(base/'run'), contexts=[{}, {}, dict(root=9, child=24, anchor=135)],
                          physical=dict(depletant_radius=1.4, activity=.0275))
            config_ref = save(base/'config.json', config); run_ref = save(base/'run-binding.json', {})
            files = {config_ref['path']: config_ref['sha256'], run_ref['path']: run_ref['sha256']}; chains = []
            for i, (arm, start, stream) in enumerate(itertools.product(diagnostic.ARMS, diagnostic.STARTS, range(4))):
                job = dict(id=i, context_index=2, arm=arm, initialization=start, stream=stream)
                folder = base/'run'/f'job-{i:03}'; folder.mkdir(parents=True)
                trajectory = folder/'trajectory.jsonl'; trajectory.write_text('not JSON; hashing only\n')
                ref = diagnostic.reference(trajectory); counts = dict(singleton_attempted=4608, singleton_accepted=0, singleton_self_loop=0)
                terminal = save(folder/'terminal.json', dict(complete=True, conditional_target=True, job=job,
                    blocks=4608, trajectory=ref, counts=counts, config_sha256=config_ref['sha256'], binding_sha256=run_ref['sha256']))
                files.update({ref['path']: ref['sha256'], terminal['path']: terminal['sha256']})
                chains.append(dict(job=job, trajectory=ref, counts=counts, reused_control=False))
            for name in ('src/evolving_dimer.rs', 'src/two_neighbor_singleton.rs', 'src/defensive_dimer_proposal.rs',
                         'src/oligomer_proposal.rs', 'src/depletion.rs', 'examples/evolving_dimer_benchmark.rs'):
                source = save(base/'common/source'/name, {}); files[source['path']] = source['sha256']
            origin = dict(config=config, config_binding=config_ref, run_binding_ref=run_ref,
                analysis=dict(path=str(root/'analysis.json'), sha256=diagnostic.ANALYSIS_SHA),
                plan=dict(base=str(base), files=files), result=dict(chains=chains, input_sha256=files),
                summary={}, manifest={}, plan_binding={})
            with mock.patch.object(diagnostic.admission, '_contact_receipt', return_value=origin), \
                 mock.patch.object(diagnostic, 'decoded_rows', side_effect=AssertionError('No decode in preparation')), \
                 mock.patch.object(diagnostic, 'closure', return_value={}), \
                 mock.patch.object(diagnostic.admission, 'runtime', return_value={'synthetic': True}):
                plan = diagnostic.make_plan(root/'contact', root/'new')
                self.assertEqual(len(plan['chains']), 16)
                self.assertEqual(plan['allocation']['attempts'], 73728)
                self.assertTrue(all(c['trajectory']['path'] in plan['input_sha256'] for c in plan['chains']))
                Path(chains[0]['trajectory']['path']).write_text('changed\n')
                with self.assertRaisesRegex(ValueError, 'Changed bound input'):
                    diagnostic.make_plan(root/'contact', root/'new')


if __name__ == '__main__': unittest.main()
