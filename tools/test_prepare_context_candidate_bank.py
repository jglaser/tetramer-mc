"""Small deterministic extraction/authentication fixtures; no real journals."""
import copy
import io
import json
import unittest

import prepare_context_candidate_bank as bank


IDENTITY = dict(start='saved_body77', stream=0, arm='context')
POSE = dict(position=[1., 2., 3.], orientation=[1., 0., 0., 0.])


def journal(cycles=3):
    rows = [dict(kind='invocation_begun', method='posterior_involution', resume=None, certify=None),
            dict(kind='prepared'), dict(kind='fixed_context_audit'), dict(kind='initial_state')]
    for cycle in range(1, cycles+1):
        for slot in range(5):
            index = 5*(cycle-1)+slot
            rows.append(dict(kind='attempt_begun', cycle=cycle, slot=slot, attempt_index=index))
            branch = 'uniform' if cycle % 2 else 'involution'
            status = ('wall_rejected', 'core_rejected', 'bath_rejected', 'accepted')[(cycle-1) % 4]
            proposal = dict(branch=branch)
            if branch == 'involution':
                proposal.update(source_law='posterior', identity=False, full_new_gaussian_log_density=-3.)
            step = dict(proposed_pose=copy.deepcopy(POSE), accepted=status == 'accepted', status=status,
                        wall_valid=status != 'wall_rejected',
                        core_valid=None if status == 'wall_rejected' else status != 'core_rejected',
                        proposal=proposal)
            rows.append(dict(kind='attempt_decision', cycle=cycle, slot=slot, attempt_index=index, step=step))
            rows.append(dict(kind='attempt_complete', cycle=cycle, slot=slot, attempt_index=index,
                identity=IDENTITY, global_=slot == 4, proposed_pose=copy.deepcopy(POSE),
                wall_valid=step['wall_valid'], core_valid=step['core_valid'], accepted=step['accepted'],
                production=cycle > 256, proposal=proposal))
            rows[-1]['global'] = rows[-1].pop('global_')
        rows.append(dict(kind='cycle_complete', cycle=cycle))
    for index, row in enumerate(rows):
        row['event_index'] = index
    return rows


def encoded(rows):
    return io.StringIO(''.join(json.dumps(row)+'\n' for row in rows))


def reports():
    tokens = sorted([[77, 217, f'm{i}', f'p{i}'] for i in range(16)])
    initial = dict(neighbor_labels=[16, 217], patch_tokens=[[16, 77, 'anchor', 'mobile']]+tokens,
                   fingerprint='immutable-source-fingerprint')
    return [dict(initial_observation=copy.deepcopy(initial)) for _ in range(12)]


class CandidateBankExtractionTests(unittest.TestCase):
    def test_all_candidates_preserve_rejections_ordinals_and_original_poses(self):
        events = journal(4)
        rows = list(bank.extract_rows(encoded(events), IDENTITY, expected_cycles=4))
        self.assertEqual(len(rows), 4)
        self.assertEqual([r['status'] for r in rows],
                         ['wall_rejected', 'core_rejected', 'bath_rejected', 'accepted'])
        self.assertEqual([r['ordinal'] for r in rows], [0, 1, 2, 3])
        self.assertEqual([r['event_index'] for r in rows], [17, 33, 49, 65])
        self.assertEqual([r['attempt_index'] for r in rows], [4, 9, 14, 19])
        self.assertEqual([r['saved_log_g'] for r in rows], [None, -3., None, -3.])
        self.assertTrue(all(r['proposed_pose'] == POSE and r['identity'] == IDENTITY for r in rows))
        self.assertEqual(sum(r['accepted'] for r in rows), 1)

    def test_missing_reordered_duplicate_and_wrong_complete_events_fail(self):
        base = journal(2)
        mutations = []
        missing = copy.deepcopy(base);del missing[18];mutations.append(missing)
        reorder = copy.deepcopy(base);reorder[17], reorder[18] = reorder[18], reorder[17];mutations.append(reorder)
        duplicate = copy.deepcopy(base);duplicate.insert(18, copy.deepcopy(duplicate[17]));mutations.append(duplicate)
        for field, value in [('identity', {}), ('proposed_pose', dict(POSE, position=[9., 2., 3.])),
                             ('accepted', True), ('production', True), ('global', False)]:
            changed = copy.deepcopy(base);changed[18][field] = value;mutations.append(changed)
        for changed in mutations:
            with self.subTest(change=changed[17:19]), self.assertRaises(ValueError):
                list(bank.extract_rows(encoded(changed), IDENTITY, expected_cycles=2))

    def test_fatal_null_or_changed_learned_law_never_becomes_zero(self):
        base = journal(2)
        for field, value in [('status', 'fatal'), ('proposed_pose', None),
                             ('proposed_pose', dict(POSE, orientation=[2., 0., 0., 0.])),
                             ('accepted', True), ('core_valid', True)]:
            changed = copy.deepcopy(base);changed[17]['step'][field] = value
            with self.subTest(field=field), self.assertRaises((ValueError, TypeError)):
                list(bank.extract_rows(encoded(changed), IDENTITY, expected_cycles=2))
        for field, value in [('branch', 'local'), ('source_law', 'prior'),
                             ('identity', True), ('full_new_gaussian_log_density', None)]:
            changed = copy.deepcopy(base);changed[33]['step']['proposal'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                list(bank.extract_rows(encoded(changed), IDENTITY, expected_cycles=2))

    def test_incomplete_tail_keeps_only_complete_prefix_and_reports_failure(self):
        rows = journal(2)[:34]
        source = bank.extract_rows(encoded(rows), IDENTITY, expected_cycles=2)
        self.assertEqual(next(source)['ordinal'], 0)
        with self.assertRaises(ValueError):
            next(source)
        with self.assertRaises(ValueError):
            list(bank.extract_rows(encoded(journal(2)), IDENTITY, expected_cycles=3))

    def test_source_reference_uses_all_twelve_initial_patterns(self):
        source = reports();result = bank.source_reference(source)
        self.assertEqual(len(result['source_secondary_tokens']), 16)
        self.assertEqual(result['inclusion_boundaries'], [0., .25, .5, .75, 1.])
        for changed in [source[:11], source+[source[0]]]:
            with self.assertRaises(ValueError):bank.source_reference(changed)
        changed = copy.deepcopy(source);changed[-1]['initial_observation']['patch_tokens'].pop()
        with self.assertRaises(ValueError):bank.source_reference(changed)
        for mutation in ['wrong_neighbor', 'duplicate', 'unsorted']:
            changed = reports()
            for row in changed:
                initial = row['initial_observation']
                if mutation == 'wrong_neighbor':initial['neighbor_labels'] = [16, 56]
                elif mutation == 'duplicate':initial['patch_tokens'][-1] = initial['patch_tokens'][-2]
                else:initial['patch_tokens'][-1], initial['patch_tokens'][-2] = initial['patch_tokens'][-2], initial['patch_tokens'][-1]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):bank.source_reference(changed)

    def test_completed_drained_receipt_binds_exact_terminal(self):
        receipt = dict(success=True, child_drained=True, returncode=0,
                       terminal=dict(path='/tmp/example-terminal.json', sha256='a'*64))
        bank.success(receipt, '/tmp/example-terminal.json', 'a'*64)
        for field, value in [('success', False), ('child_drained', False), ('returncode', 1),
                             ('terminal', dict(path='/tmp/other.json', sha256='a'*64)),
                             ('terminal', dict(path='/tmp/example-terminal.json', sha256='b'*64))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                bank.success(dict(receipt, **{field: value}), '/tmp/example-terminal.json', 'a'*64)


if __name__ == '__main__':
    unittest.main()
