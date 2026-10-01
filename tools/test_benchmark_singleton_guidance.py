"""Synthetic controls: no physical executable is launched by these tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_singleton_guidance as b


class Child:
    def __init__(self, index, code=0):
        self.pid, self.code, self.waits = index + 1000, code, 0

    def poll(self):
        return self.code

    def wait(self):
        self.waits += 1
        return self.code


def event(accepted=False, hard=False, norm=100., phase=1):
    pose = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
    return dict(kind='cluster_event', members=[0], accepted=accepted, hard_valid=hard,
        proposed_gained_contacts=[[0,1]], proposed_lost_contacts=[[0,2]],
        old_poses=[pose], retained_poses=[pose], proposed_poses=[pose], phase=phase,
        diagnostic=dict(topology='embedded', pool_contains_direct_partner=True),
        proposal=dict(branch='involution', full_old_log_density=-9., full_new_log_density=-1.,
            step=dict(source_latent=[norm,0.,0.,0.,0.,0.], log_auxiliary_ratio=-1.),
            oligomer=dict(fused_components=2, source_label=dict(kind='single'), target_label=dict(kind='fused'),
                build_seconds=[.1,.2,.3])), log_acceptance=-2. if hard else None)


class SingletonBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self):
        pose = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
        return dict(initial_poses=[pose, copy.deepcopy(pose)], seed=17,
            shape='old-shape', boundary=dict(kind='spherical', radius=10.),
            fixed_body_indices=[], seed_labels=[0], metadata={},
            local_translation_std_A=.2, local_small_angle_std_degrees=1.,
            depletant_radius=1.4, reservoir_density=.0275,
            cluster_phase=dict(dimer_rate=1., trimer_rate=.25, oligomer={'max_components':32}))

    def frozen(self, populations=2):
        snapshot = self.root/'snapshot'
        (snapshot/'provenance').mkdir(parents=True)
        b.write(snapshot/'config.json', self.source())
        b.write(snapshot/'provenance/shape.json', {'synthetic':'shape'})
        b.write(snapshot/'provenance/frozen-relative-model.json', {'synthetic':'model'})
        checkpoint = dict(poses=self.source()['initial_poses'], completed_sweeps=60900,
            shape_sha256=b.sha(snapshot/'provenance/shape.json'),
            model_sha256=b.sha(snapshot/'provenance/frozen-relative-model.json'))
        checkpoint['poses'][0]['position'] = [2.,3.,4.]
        b.write(snapshot/'checkpoint.json', checkpoint)
        text = 'synthetic source includes singleton_rate'
        bundle = self.root/'source-bundle.json'
        b.write(bundle, dict(files={'src/cluster_phase.rs': dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())}))
        binary = self.root/'synthetic-binary'
        binary.write_bytes(b'NEVER EXECUTE\n'+bundle.read_bytes())
        binary.chmod(0o700)
        out = self.root/'control'
        b.freeze(out, snapshot, binary, bundle, populations=populations, phases=5)
        return out

    def test_frozen_configs_share_checkpoint_and_only_change_fusion(self):
        out = self.frozen()
        protocol = b.validate(out)
        self.assertEqual(protocol['expected_events_per_population'], .1)
        self.assertEqual(len({j['seed'] for j in protocol['jobs']}), 4)
        left, right = [b.read(j['config']) for j in protocol['jobs'][:2]]
        self.assertEqual(left['initial_poses'][0]['position'], [2.,3.,4.])
        self.assertNotIn('oligomer', left['cluster_phase'])
        self.assertEqual(right['cluster_phase'].pop('oligomer'), {})
        self.assertEqual(left['cluster_phase'], right['cluster_phase'])
        self.assertEqual(left['cluster_phase']['singleton_rate'], 1.)
        self.assertEqual(left['cluster_phase']['dimer_rate'], 0.)
        self.assertEqual(left['depletant_radius'], 1.4)
        self.assertEqual(left['reservoir_density'], .0275)
        self.assertFalse((out/'status.json').exists())

    def test_changed_frozen_inputs_rejected(self):
        out = self.frozen()
        (out/'provenance/model.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            b.validate(out)

    def test_contact_pool_control_preserves_physics_and_fusion_comparison(self):
        source = self.source()
        checkpoint = dict(poses=source['initial_poses'], completed_sweeps=60900)
        old = b.configured(source, checkpoint, self.root, 'unfused', 123, 1., .01, .1)
        new = b.configured(source, checkpoint, self.root, 'unfused', 123, 1., .01, .1,
                           'contact_without_replacement')
        self.assertEqual(new['cluster_phase'].pop('anchor_pool_selection'),
                         'contact_without_replacement')
        self.assertEqual(old, new)
        fused = b.configured(source, checkpoint, self.root, 'fused', 123, 1., .01, .1,
                             'contact_without_replacement')
        self.assertEqual(fused['cluster_phase'].pop('oligomer'), {})
        fused['cluster_phase'].pop('anchor_pool_selection')
        self.assertEqual(fused['cluster_phase'], old['cluster_phase'])

    def test_denominators_include_hard_rejection_and_uniform(self):
        rows = [event(), event(accepted=True, hard=True, norm=1.), event(hard=True, norm=3.)]
        uniform = event()
        uniform['proposal'] = dict(branch='uniform')
        rows.append(uniform)
        stats = b.event_statistics(rows)
        self.assertEqual(stats['counts']['attempted'], 4)
        self.assertEqual(stats['fractions']['accepted'], .25)
        self.assertEqual(stats['fractions']['hard_valid'], .5)
        self.assertEqual(stats['counts']['accepted_exchanges'], 1)
        self.assertEqual(stats['metrics']['source_latent_norm_all']['count'], 3)
        self.assertEqual(stats['metrics']['source_latent_norm_valid']['median'], 2.)
        self.assertEqual(stats['metrics']['source_latent_norm_accepted']['median'], 1.)
        self.assertEqual(stats['metrics']['source_latent_norm_all']['maximum'], 100.)

    def test_independent_cap_config_and_exhausted_attempt_accounting(self):
        source = self.source()
        checkpoint = dict(poses=source['initial_poses'], completed_sweeps=60900)
        config = b.configured(source, checkpoint, self.root, 'fused', 123, 1., .01, .1,
                              'contact_without_replacement', 8)
        self.assertEqual(config['cluster_phase']['singleton_independent_max_trials'], 8)
        self.assertEqual(config['cluster_phase']['dimer_rate'], 0.)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            b.configured(source, checkpoint, self.root, 'fused', 123, 1., .01, .1,
                         'contact_without_replacement', 0)
        exhausted = event()
        exhausted['proposed_poses'] = None
        exhausted['proposal'] = dict(branch='independent_conditioned', raw_trial_count=8,
                                     max_trials=8, cap_exhausted=True,
                                     raw_trials=[dict(hard_valid=False)]*8)
        stats = b.event_statistics([exhausted, event(accepted=True, hard=True)])
        self.assertEqual(stats['counts']['learned'], 2)
        self.assertEqual(stats['counts']['independent_raw_trials'], 8)
        self.assertEqual(stats['counts']['independent_cap_exhaustions'], 1)
        self.assertEqual(stats['counts']['proposal_nulls'], 1)
        self.assertEqual(stats['counts']['attempted'], 2)
        self.assertEqual(stats['fractions']['accepted'], .5)
        invalid = copy.deepcopy(exhausted)
        invalid['proposal']['raw_trials'][0] = dict(hard_valid=True)
        with self.assertRaisesRegex(ValueError, 'retried after'):
            b.event_statistics([invalid])

    def test_local_control_retains_clock_physics_and_move_sizes(self):
        source = self.source()
        checkpoint = dict(poses=source['initial_poses'], completed_sweeps=60900)
        config = b.configured(source, checkpoint, self.root, 'local', 123, 1., .01, .1,
                              local_only=True)
        phase = config['cluster_phase']
        self.assertEqual(phase['transport_probability'], 0.)
        self.assertEqual(phase['singleton_rate'], 1.)
        self.assertEqual(phase['duration'], .01)
        self.assertEqual(phase['local_translation_std_A'], source['local_translation_std_A'])
        self.assertEqual(phase['local_small_angle_std_degrees'], source['local_small_angle_std_degrees'])
        self.assertEqual(config['reservoir_density'], source['reservoir_density'])
        self.assertNotIn('oligomer', phase)
        self.assertNotIn('singleton_independent_max_trials', phase)

    def test_success_waits_for_all_children(self):
        out = self.frozen()
        children = []
        def start(*args, **kwargs):
            child = Child(len(children))
            children.append(child)
            return child
        with patch.object(b.subprocess, 'Popen', side_effect=start):
            result = b.run(out)
        self.assertTrue(result['complete'])
        self.assertEqual(len(children), 4)
        self.assertTrue(all(c.waits == 1 for c in children))
        with self.assertRaises(ValueError):
            b.run(out)

    def test_failure_drains_only_started_children_and_retains_all_statuses(self):
        out = self.frozen()
        children = []
        def start(*args, **kwargs):
            child = Child(len(children), code=7 if not children else 0)
            children.append(child)
            return child
        with patch.object(b.subprocess, 'Popen', side_effect=start):
            with self.assertRaisesRegex(RuntimeError, 'Physical benchmark failed'):
                b.run(out)
        self.assertEqual(len(children), 2)
        self.assertTrue(all(c.waits == 1 for c in children))
        status = b.read(out/'status.json')
        self.assertFalse(status['complete'])
        self.assertFalse(status['running'])
        self.assertEqual([j['status'] for j in status['jobs']], ['failed','complete','not_started','not_started'])

    def test_launch_exception_drains_other_children(self):
        out = self.frozen()
        child = Child(0)
        with patch.object(b.subprocess, 'Popen', side_effect=[child, OSError('launch test')]):
            with self.assertRaisesRegex(OSError, 'launch test'):
                b.run(out)
        self.assertEqual(child.waits, 1)
        self.assertEqual([j['status'] for j in b.read(out/'status.json')['jobs']],
            ['complete', 'launch_failed', 'not_started', 'not_started'])

    def test_worker_cap_rejects_before_writing(self):
        with self.assertRaisesRegex(ValueError, 'maximum two'):
            b.freeze(self.root/'never-created', self.root, self.root/'binary', self.root/'bundle', workers=3)
        self.assertFalse((self.root/'never-created').exists())

    def test_native_observer_restarts_every_phase_and_retains_rejections(self):
        initial = self.source()['initial_poses']
        class Classifier:
            def classify_pair(self, left, right):
                return [dict(motif_id=7)] if left['position'][0] == 1. else []
        rows = [event(accepted=True, hard=True, phase=1), event(phase=2)]
        rows[0]['retained_poses'][0] = dict(position=[1.,0.,0.], orientation=[1.,0.,0.,0.])
        rows[0]['event_time'] = .001
        result = b.native_changes(rows, initial, Classifier())
        self.assertEqual(result['gained'], 1)
        self.assertEqual(result['lost'], 0)


if __name__ == '__main__':
    unittest.main()
