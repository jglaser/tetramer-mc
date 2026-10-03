"""Synthetic freeze/selection/manifest tests: no scientific poses or draws."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import struct
import unittest
from unittest.mock import patch

import prepare_evolving_dimer_benchmark as prep


def row(pair, parent=None):
    parent = pair if parent is None else parent
    return dict(members=pair, parent_component_members=parent,
        whole_component=pair == parent, witnesses=[dict(stream='saved', line=1, phase=1, event=1)])


def config_fixture(base):
    (base/'common/source/examples').mkdir(parents=True)
    prep.write(base/'common/scientific-allocation.json', prep.allocation())
    prep.write(base/'protocol.json', dict(schema='synthetic-only'))
    config = prep.configuration(base, {}, [], {'src/synthetic.rs': hashlib.sha256(b'synthetic').hexdigest()})
    prep.write(base/'config.json', config)
    example = base/'common/source'/prep.EXAMPLE
    example.write_text('example source')
    prep.write(base/'freeze.json', dict(schema='evolving-dimer-preparation-freeze-v1', complete=True,
        scientific_execution_started=False,
        files={str(p.relative_to(base)): prep.sha(p) for p in base.rglob('*') if p.is_file()},
        original_input_bindings={}))
    return config


def prepared_fixture(base):
    base.mkdir(exist_ok=True)
    poses = [dict(position=[i, 0, 0], orientation=[1, 0, 0, 0]) for i in range(2)]
    prep.write(base/'source.json', dict(poses=poses))
    prep.write(base/'config.json', dict(source_frame=prep.record(base/'source.json'),
        contexts=[dict(root=0, child=1) for _ in range(4)],cloud=dict(raw_count=2),
        preparation=dict(minimum_max_center_displacement_A=5.,minimum_max_body_orientation_degrees=10.)))
    (base/'binding.json').write_text('{}\n')
    directory = base/'prepared'
    directory.mkdir()
    files, clouds, starts = [], [], []
    def file(name):
        path = directory/name
        path.write_text('Synthetic inventory fixture only.\n')
        bound = prep.record(path)
        files.append(bound)
        return bound
    for c in range(4):
        for initialization in prep.INITIALIZATIONS:
            for s in range(4):
                prefix = f'{c}-{initialization}-{s}'
                raw=file(prefix+'-raw.bin');metadata=file(prefix+'-metadata.json')
                Path(raw['path']).write_bytes(struct.pack('<6d', .1,.2,.3,.4,.5,.6))
                Path(metadata['path']).write_text(json.dumps(dict(raw_count=2,low=[-1.]*3,high=[1.]*3,kept_indices=[0],cpu_seconds=.01)))
                raw['sha256']=prep.sha(raw['path']);metadata['sha256']=prep.sha(metadata['path'])
                clouds.append(dict(context_index=c, initialization=initialization, stream=s,raw=raw,metadata=metadata))
        for s in range(4):
            start = file(f'{c}-{s}-start.json')
            ledger = file(f'{c}-{s}-attempts.jsonl')
            selected=copy.deepcopy(poses);selected[0]['position'][0]=6.
            Path(start['path']).write_text(json.dumps(dict(context_index=c, stream=s,
                status='prepared', attempts=1+s, source=poses, selected=selected)))
            Path(ledger['path']).write_text(''.join(json.dumps(dict(attempt=i, selected=i == 1+s,
                outcome=dict(candidate=dict(root=selected[0] if i == 1+s else poses[0], child=poses[1]))))+'\n' for i in range(1, 2+s)))
            start['sha256'] = prep.sha(start['path'])
            ledger['sha256'] = prep.sha(ledger['path'])
            starts.append(dict(context_index=c, stream=s, status='prepared', attempts=1+s,
                record=start, ledger=ledger))
    manifest = dict(schema='evolving-dimer-prepared-starts-v1', complete=True, passed=True,
        all_attempts_retained=True, config_sha256=prep.sha(base/'config.json'),
        binding_sha256=prep.sha(base/'binding.json'), files=files,
        cloud_banks=clouds, alternative_starts=starts)
    prep.write(directory/'manifest.json', manifest)
    return directory/'manifest.json', manifest


class EvolvingPreparationTests(unittest.TestCase):
    def test_exclude_entire_previous_parents_before_lexicographic_selection(self):
        table = [row([0, 1], [0, 1, 2]), row([0, 2], [0, 1, 2]), row([3, 4]),
                 row([5, 6]), row([7, 8], [7, 8, 9]), row([7, 9], [7, 8, 9]),
                 row([10, 11], [10, 11, 12])]
        selected, detail = prep.select_held_out(list(reversed(table)), previous=[[0, 1], [3, 4]], per_stratum=1)
        self.assertEqual([r['members'] for r in selected], [[5, 6], [7, 8]])
        self.assertEqual(detail['eligible_counts'], {'true': 1, 'false': 3})
        self.assertNotIn([0, 2], [r['members'] for r in selected])

    def test_selection_ignores_outcomes_native_and_energy_fields(self):
        table = [row([0, 1]), row([2, 3]), row([4, 5], [4, 5, 6])]
        want = prep.select_held_out(table, previous=[[0, 1]], per_stratum=1)[0]
        for i, r in enumerate(table):
            r.update(accepted=i % 2 == 0, native=True, energy=-1e100, candidate=None, hard_valid=False)
        actual = prep.select_held_out(table, previous=[[0, 1]], per_stratum=1)[0]
        self.assertEqual([r['members'] for r in want], [r['members'] for r in actual])
        table[1]['members'][0] = 100
        self.assertEqual(actual[0]['members'], [2, 3])

    def test_invalid_or_insufficient_archived_panel_rejected(self):
        valid = [row([0, 1]), row([2, 3]), row([4, 5], [4, 5, 6])]
        for defect in ['duplicate', 'missing', 'whole', 'no_witness', 'no_embedded']:
            table = copy.deepcopy(valid)
            if defect == 'duplicate': table.append(copy.deepcopy(table[1]))
            elif defect == 'missing': table.pop(0)
            elif defect == 'whole': table[2]['whole_component'] = True
            elif defect == 'no_witness': table[1]['witnesses'] = []
            else: table.pop()
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                prep.select_held_out(table, previous=[[0, 1]], per_stratum=1)

    def test_anchor_nearest_root_nonmember_tie_break_and_no_state_mutation(self):
        state = [dict(position=p, orientation=[1., 0., 0., 0.])
                 for p in [[0., 0., 0.], [.1, 0., 0.], [1., 0., 0.], [-1., 0., 0.]]]
        old = copy.deepcopy(state)
        context = prep.make_contexts(state, [row([0, 1])])[0]
        self.assertEqual(context['anchor'], 2)
        self.assertEqual(state, old)
        state[2]['position'][0] = float('nan')
        with self.assertRaises(ValueError): prep.make_contexts(state, [row([0, 1])])

    def test_complete_covariance_transform_including_cross_terms(self):
        covariance = [[float(10*i+j+1) for j in range(6)] for i in range(6)]
        original = dict(schema='reciprocal-pose-mixture-v1', reciprocal_components=[True],
            base_model=dict(schema='weighted-pose-mixture-v1', covariances=[covariance],
                            means=[[1, 2, 3, 4, 5, 6]], other='preserve'))
        scaled = copy.deepcopy(original)
        scaled['base_model']['covariances'] = [[[x*.0625 for x in r] for r in covariance]]
        prep.verify_covariance_transform(original, scaled)
        for defect in ['off_diagonal', 'mean', 'reciprocal']:
            broken = copy.deepcopy(scaled)
            if defect == 'off_diagonal': broken['base_model']['covariances'][0][1][4] += .1
            elif defect == 'mean': broken['base_model']['means'][0][0] += .1
            else: broken['reciprocal_components'][0] = False
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                prep.verify_covariance_transform(original, broken)

    def test_jobs_exact_fixed_allocation_and_family_pairing(self):
        jobs = prep.jobs()
        self.assertEqual(len(jobs), 96)
        self.assertEqual([j['id'] for j in jobs], list(range(96)))
        self.assertEqual([j['queue_key'] for j in jobs], sorted(j['queue_key'] for j in jobs))
        families = {}
        for j in jobs:
            key = (j['context_index'], j['initialization'], j['stream'])
            families.setdefault(key, []).append(j['arm'])
            self.assertEqual(j['seed_family'], dict(context_index=key[0], initialization=key[1], stream=key[2]))
        self.assertEqual(len(families), 32)
        self.assertTrue(all(sorted(arms) == sorted(prep.ARMS) for arms in families.values()))
        self.assertEqual(prep.jobs(), jobs)
        expected = hashlib.sha256(b'evolving-dimer-v1/6100300601/0/source/0/-1/queue/local').hexdigest()[:16]
        self.assertEqual(prep.seed(prep.MASTER, 0, 'source', 0, -1, 'queue/local'), int(expected, 16))
        full=prep.runtime_seed(prep.MASTER,0,'source',0,1,'local/0/proposal')
        self.assertEqual(len(full),32)
        self.assertEqual(full,hashlib.sha256(b'evolving-dimer-v1/6100300601/0/source/0/1/local/0/proposal').digest())

    def test_config_parameters_and_allocation_arithmetic(self):
        with tempfile.TemporaryDirectory() as temp:
            c = config_fixture(Path(temp))
            a = c['allocation']
            self.assertEqual((a['chains'], a['local_attempts'], a['dimer_attempts']), (96, 1769472, 294912))
            self.assertEqual(a['raw_cloud_bytes'], 12582912)
            self.assertEqual(a['maximum_preparation_outers'], 4096)
            self.assertEqual((c['factorized']['tau'], c['factorized']['uniform_probability']), (.25, .5))
            self.assertFalse(c['local']['pair_contact_required'])
            self.assertEqual(c['factorized']['outside_pair_contact'], 'self_loop')
            self.assertEqual(c['local']['member_order'], [0, 1, 0, 1])
            self.assertEqual(c['limits']['raw_campaign'], 20_000_000_000)
            self.assertEqual(c['cloud']['banks'], 32)
            self.assertEqual(set(c['protocol']), {'path', 'sha256'})
            self.assertFalse(c['preparation']['energy_filter'])
            self.assertFalse(c['preparation']['native_filter'])

    def test_successful_manifest_binds_every_file_and_all_family_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            path, manifest = prepared_fixture(base)
            actual, files = prep.validate_prepared_manifest(base, path)
            self.assertEqual(actual, manifest)
            self.assertEqual(len(files), 96)
            self.assertEqual(set(files), {v['path'] for v in manifest['files']})

    def test_failed_extended_duplicate_or_tampered_preparations_rejected(self):
        for defect in ['failed', 'omitted_attempts', 'extended', 'duplicate_start', 'duplicate_cloud',
                       'hash', 'omitted_file', 'extra_file', 'source_provenance', 'missing_raw', 'bool_id']:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                path, m = prepared_fixture(base)
                if defect == 'failed': m['alternative_starts'][0]['status'] = 'fatal'
                elif defect == 'omitted_attempts': m['all_attempts_retained'] = False
                elif defect == 'extended': m['alternative_starts'][0]['attempts'] = 257
                elif defect == 'duplicate_start': m['alternative_starts'][1] = m['alternative_starts'][0]
                elif defect == 'duplicate_cloud': m['cloud_banks'][1] = m['cloud_banks'][0]
                elif defect == 'hash': Path(m['files'][0]['path']).write_text('tampered')
                elif defect == 'omitted_file': m['files'].pop()
                elif defect == 'extra_file': (base/'prepared/unbound.json').write_text('{}')
                elif defect == 'source_provenance': m['config_sha256'] = '0'*64
                elif defect == 'missing_raw': del m['cloud_banks'][0]['raw']
                else: m['cloud_banks'][0]['context_index'] = False
                path.write_text(json.dumps(m))
                with self.assertRaises((ValueError, KeyError)):
                    prep.validate_prepared_manifest(base, path)

    def test_preparation_inventory_cannot_escape_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            path, m = prepared_fixture(base)
            outside = base/'outside.json'
            outside.write_text('not a preparation file')
            m['files'].append(prep.record(outside))
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'Invalid or duplicate'):
                prep.validate_prepared_manifest(base, path)

    def test_ledger_first_success_completeness_and_pose_correspondence(self):
        for defect in ['gap', 'early_selection', 'fatal', 'source_pose', 'selected_pose', 'attempt_count']:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                path, manifest = prepared_fixture(base)
                start = manifest['alternative_starts'][1]
                if defect in ['source_pose', 'selected_pose', 'attempt_count']:
                    target = start['record']
                    value = prep.read(target['path'])
                    if defect == 'source_pose': value['source'][0]['position'][0] += 1
                    elif defect == 'selected_pose': value['selected'][0]['position'][0] += 1
                    else: value['attempts'] += 1
                    Path(target['path']).write_text(json.dumps(value))
                else:
                    target = start['ledger']
                    rows = [json.loads(line) for line in Path(target['path']).read_text().splitlines()]
                    if defect == 'gap': rows[0]['attempt'] = 3
                    elif defect == 'early_selection': rows[0]['selected'] = True
                    else: rows[0]['failure'] = 'fatal'
                    Path(target['path']).write_text(''.join(json.dumps(r)+'\n' for r in rows))
                target['sha256'] = prep.sha(target['path'])
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError): prep.validate_prepared_manifest(base, path)

    def test_cloud_arithmetic_and_nonfinite_raw_rejected(self):
        for defect in ['raw_count','index','bounds','cpu','variate']:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as temp:
                base=Path(temp);path,m=prepared_fixture(base);bank=m['cloud_banks'][0]
                target=bank['raw'] if defect=='variate' else bank['metadata']
                if defect=='variate':Path(target['path']).write_bytes(struct.pack('<6d',float('nan'),.2,.3,.4,.5,.6))
                else:
                    v=prep.read(target['path'])
                    if defect=='raw_count':v['raw_count']=3
                    elif defect=='index':v['kept_indices']=[2]
                    elif defect=='bounds':v['low'][0]=v['high'][0]
                    else:v['cpu_seconds']=-1
                    Path(target['path']).write_text(json.dumps(v))
                target['sha256']=prep.sha(target['path']);path.write_text(json.dumps(m))
                with self.assertRaises(ValueError):prep.validate_prepared_manifest(base,path)

    def test_first_displacement_eligible_candidate_cannot_be_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);path,m=prepared_fixture(base);start=m['alternative_starts'][1];target=start['ledger']
            rows=[json.loads(v) for v in Path(target['path']).read_text().splitlines()]
            rows[0]['outcome']['candidate']=copy.deepcopy(rows[-1]['outcome']['candidate'])
            Path(target['path']).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            target['sha256']=prep.sha(target['path']);path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'first-feasible'):prep.validate_prepared_manifest(base,path)

    def test_source_closure_recurses_imports_without_executing_them(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            (base/'a.py').write_text('import b\nraise RuntimeError("must not execute")\n')
            (base/'b.py').write_text('from c import value\n')
            (base/'c.py').write_text('value=1\n')
            self.assertEqual([p.name for p in prep.source_dependencies([base/'a.py'])],['a.py','b.py','c.py'])

    def test_freeze_tampering_and_allocation_change_fail_before_binding(self):
        for defect in ['bytes', 'allocation', 'jobs']:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                c = config_fixture(base)
                prep.verify_freeze(base)
                if defect == 'bytes':
                    (base/'protocol.json').write_text('changed')
                else:
                    if defect == 'allocation': c['allocation']['production_blocks'] += 1
                    else: c['jobs'].pop()
                    (base/'config.json').write_text(json.dumps(c))
                    f = prep.read(base/'freeze.json')
                    f['files']['config.json'] = prep.sha(base/'config.json')
                    (base/'freeze.json').write_text(json.dumps(f))
                with self.assertRaises(ValueError): prep.verify_freeze(base)

    def test_binary_binding_authenticates_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root/'run'
            base.mkdir()
            c = config_fixture(base)
            (root/'examples').mkdir()
            (root/prep.EXAMPLE).write_text('example source')
            executable = root/'synthetic-executable'
            executable.write_bytes(b'not an executable; never launched')
            bundle = root/'source-bundle.json'
            prep.write(bundle, dict(files={'src/synthetic.rs': dict(text='synthetic',
                sha256=c['compiled_source_sha256']['src/synthetic.rs'])}))
            with patch.object(prep, 'ROOT', root):
                result = prep.bind_executable(base, executable, bundle)
                self.assertEqual(result['status'], 'bound_without_launch')
                self.assertEqual(prep.read(base/'binding.json')['executable_sha256'], prep.sha(executable))
                with self.assertRaises(ValueError): prep.bind_executable(base, executable, bundle)


if __name__ == '__main__':
    unittest.main()
