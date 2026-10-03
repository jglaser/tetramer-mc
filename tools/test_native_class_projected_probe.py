"""Metadata-only 56-row freeze and mocked worker lifecycle checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import prepare_native_class_projected_probe as prepare
import run_native_class_projected_probe as worker


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, allow_nan=False)+'\n' for v in rows))


def fixture(tmp):
    """Complete metadata graph with opaque toy geometry, never reconstructed."""
    base = tmp/'source'; repository = Path(prepare.__file__).resolve().parents[1]
    definition = base/'execution-code/native-definition/definition.json'
    observer_source = repository/'tools/native_contact_regions.py'
    atom_file = definition.parent/'inputs/monomer-shape.json'; save(atom_file, dict(synthetic=True))
    observer_copy = definition.parent/'inputs/source/native_contact_regions.py'
    observer_copy.parent.mkdir(parents=True); observer_copy.write_bytes(observer_source.read_bytes())
    classification = definition.parent/'inputs/reference/results/native-neighbor-classes/classification.json'
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    save(classification, dict(classes=[dict(label='toy', body_delta=[3., 0., 0.], body_relative_rotation=identity)]))
    catalogue = definition.parent/'inputs/native-pair-motifs.json'
    save(catalogue, dict(motifs=[dict(member_contacts=[dict(member_i=0, member_j=1, directed_class='toy')])]))
    poses = [dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]),
             dict(position=[3., 0., 0.], orientation=[1., 0., 0., 0.])]
    save(definition, dict(fixed_poses=poses, input_sha256={'monomer-shape.json': prepare.sha(atom_file),
        'source/native_contact_regions.py': prepare.sha(observer_source),
        'reference/results/native-neighbor-classes/classification.json': prepare.sha(classification),
        'native-pair-motifs.json': prepare.sha(catalogue)}))
    executable = base/'execution-code/contact-line-guide-audit'; executable.write_text('Synthetic placeholder; never executed.\n')
    queries = {}
    for label in ('hard_free-r00', 'saved'):
        directory = base/'queries'/label; (directory/'provenance').mkdir(parents=True)
        probe_count = 40 if label == 'saved' else 0
        manifest = dict(schema='native-class-line-guide-audit-v1', physical_jobs=0,
            samples=128 if label == 'hard_free-r00' else 0, seed=17 if label == 'hard_free-r00' else 0,
            executable_sha256=prepare.sha(executable), native_definition_sha256=prepare.sha(definition))
        for name, key in prepare.PROVENANCE.items():
            path = directory/'provenance'/(name+'.json'); save(path, dict(synthetic=True, file=name))
            manifest[key] = prepare.sha(path)
        samples = [dict(kind='fresh', id=i, latent=[i/100., 0., 0., 0., 0., 0.],
                        log_proposal_density=float(i), draw=dict(saved=True)) for i in range(manifest['samples'])]
        probes = [dict(kind='probe', id='saved-'+str(i), latent=[i/100., 0., 0., 0., 0., 0.],
                       log_proposal_density=float(i), draw=None) for i in range(probe_count)]
        jsonl(directory/'samples.jsonl', samples); jsonl(directory/'probes.jsonl', probes)
        jsonl(directory/'attempts.jsonl', [dict(ordinal=i, kind=r['kind'], id=r['id'], state='begin')
              for i, r in enumerate(samples+probes)])
        if probes:
            p = directory/'provenance/probes.jsonl'; jsonl(p, [dict(id=r['id'], latent=r['latent']) for r in probes])
            manifest['probes_sha256'] = prepare.sha(p)
        else: manifest['probes_sha256'] = None
        save(directory/'manifest.json', manifest)
        save(directory/'summary.json', dict(complete=True, manifest=manifest,
            samples=len(samples), probes=len(probes), samples_sha256=prepare.sha(directory/'samples.jsonl'),
            probes_sha256=prepare.sha(directory/'probes.jsonl'), attempts_sha256=prepare.sha(directory/'attempts.jsonl')))
        queries[label] = prepare.read_query(base, label)
    inputs = {str(p): h for p, h in queries['hard_free-r00']['files'].items()}
    inputs[str(definition)] = prepare.sha(definition)
    for p in (atom_file, observer_copy, classification, catalogue): inputs[str(p)] = prepare.sha(p)
    # A prior source module belongs in the archive but is never rerun.
    for name in ('native_class_line_reference.py', 'hard_free_line_reference.py',
                 'analyze_contact_line_audit.py', 'native_contact_regions.py'):
        p = base/'execution-code/tools'/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# old reference '+name+'\n')
        inputs[str(p)] = prepare.sha(p)
    files = {name: h for name, h in inputs.items() if '/queries/' not in name}
    files[str(executable)] = prepare.sha(executable)
    files[str(Path(prepare.sys.executable).absolute())] = prepare.sha(prepare.sys.executable)
    jobs = []
    for label in ('hard_free-r00', 'saved'):
        manifest = queries[label]['manifest']; arm = 'hard_free' if label == 'hard_free-r00' else 'class'
        for filename, source in [('common/config.json', 'config'), ('common/region.json', 'region'),
                                 ('guides/'+arm+'.json', 'importance-guide')]:
            p = base/filename; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((base/'queries'/label/'provenance'/(source+'.json')).read_bytes()); files[str(p)] = prepare.sha(p)
        argv = [str(executable), '--config', str(base/'common/config.json'), '--region', str(base/'common/region.json'),
            '--importance-guide', str(base/'guides'/(arm+'.json')), '--out', str(base/'queries'/label),
            '--samples', str(manifest['samples']), '--seed', str(manifest['seed'])]
        if label == 'saved':
            p = base/'probes.jsonl'; p.write_bytes((base/'queries/saved/provenance/probes.jsonl').read_bytes())
            files[str(p)] = prepare.sha(p); argv += ['--probes', str(p)]
        jobs.append(dict(id=label, phase='query', arm=arm, samples=manifest['samples'],
            probes=40 if label == 'saved' else 0, seed=manifest['seed'], argv=argv,
            terminal=str(base/'queries'/label/'summary.json')))
    jobs.append(dict(id='hard_free-r00-audit', query_id='hard_free-r00', phase='audit',
        argv=[str(Path(prepare.sys.executable).absolute()), str(base/'execution-code/tools/native_class_line_reference.py'),
            str(base/'queries/hard_free-r00'), '--definition', str(definition), '--output', str(base/'audits/hard_free-r00.json'),
            '--journal', str(base/'audits/hard_free-r00.journal.jsonl')]))
    save(base/'execution-plan.json', dict(schema='native-class-line-reviewed-execution-v1', ready=True,
        maximum_workers=1, physical_clouds=0, executable_sha256=prepare.sha(executable), jobs=jobs, files=files))
    events = [dict(state='started', schema='native-class-line-independent-audit-journal-v1',
        synthetic=False, samples=128, probes=0, input_sha256=inputs)]
    for i in range(16):
        events.extend([dict(state='begin', ordinal=i, kind='fresh', id=i),
            dict(state='complete', ordinal=i, row=dict(kind='fresh', id=i, log_density=float(i)),
                 maxima=dict(log_density_error=0.), cpu_seconds=1.)])
    jsonl(base/'audits/hard_free-r00.journal.jsonl', events)
    return base, repository


def reviewed(root):
    path = root/'review.json'; save(path, dict(complete=True, passed=True,
        allocation_sha256=prepare.sha(root/'allocation.json'), freeze_sha256=prepare.sha(root/'freeze.json')))
    return path


def claim(root, review):
    save(root/'execution-claim.json', dict(freeze_sha256=prepare.sha(root/'freeze.json'),
        allocation_sha256=prepare.sha(root/'allocation.json'), review=dict(path=str(review), sha256=prepare.sha(review))))


def mocked_models(root, plan, emit, progress):
    n = plan['observer_setup']['reference_contact_calls']
    progress['observer_setup_counts'] = dict(reference_contacts_started=n, reference_contacts_completed=n,
                                            scaffold_classifier_started=1, scaffold_classifier_completed=1)
    return {'hard_free-r00': 'fresh', 'saved': 'saved'}


class ProjectedProbeTests(unittest.TestCase):
    def test_observer_setup_inventory_wrappers_and_cap_preserve_partial_progress(self):
        import numpy as np
        cls = worker.reference.NativeContactRegions
        with tempfile.TemporaryDirectory() as tmp:
            base, _ = fixture(Path(tmp))
            inventory = prepare.setup_inventory(base/'execution-code/native-definition/definition.json')
            self.assertEqual(inventory['reference_contact_calls'], 1)
            self.assertEqual(inventory['maximum_total_contact_calls'], 2)
            query = inventory['reference_contact_queries'][0]
            calls = []
            def contact(model, position, rotation, cutoff):
                calls.append(cutoff); return dict(synthetic=True)
            def classifier(model, anchor, moving):
                return model._contacts(np.asarray(query['position']), np.asarray(query['rotation']), 2.)
            for fail in (False, True):
                counts = []; progress = {}; limits = copy.deepcopy(inventory)
                if fail: limits['maximum_scaffold_contact_calls'] = 0
                with (mock.patch.object(cls, '_contacts', contact), mock.patch.object(cls, 'classify_pair', classifier)):
                    model = cls.__new__(cls)
                    def initialize():
                        with worker.count_observer_setup(limits, counts.append, progress):
                            model._contacts(np.asarray(query['position']), np.asarray(query['rotation']), 1.)
                            model.classify_pair(*inventory['fixed_scaffold_poses'])
                    if fail:
                        with self.assertRaisesRegex(ValueError, 'query cap'): initialize()
                    else:
                        initialize(); prepare.check_setup_counts(inventory, progress['observer_setup_counts'])
                    self.assertIs(cls._contacts, contact); self.assertIs(cls.classify_pair, classifier)
                self.assertEqual(progress['observer_setup_counts']['reference_contacts_completed'], 1)
                self.assertEqual(progress['observer_setup_counts']['scaffold_classifier_started'], 1)
                self.assertEqual(progress['observer_setup_counts']['scaffold_classifier_completed'], 0 if fail else 1)
                self.assertEqual(sum(e['state'] == 'setup_query_begin' for e in counts), 2 if fail else 3)
            self.assertEqual(calls, [1., 2., 1.])

    def test_selected_execution_authority_edges_are_required(self):
        for mode in ('not_ready', 'wrong_seed', 'wrong_executable', 'wrong_guide', 'missing_old_source', 'duplicate_job'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                base, repo = fixture(Path(tmp)); root = Path(tmp)/'frozen'; p = base/'execution-plan.json'; value = prepare.read(p)
                if mode == 'not_ready': value['ready'] = False
                elif mode == 'wrong_seed': value['jobs'][0]['seed'] += 1
                elif mode == 'wrong_executable': value['executable_sha256'] = '0'*64
                elif mode == 'wrong_guide': value['jobs'][0]['argv'][6] = str(base/'guides/class.json')
                elif mode == 'missing_old_source': del value['files'][str(base/'execution-code/tools/native_class_line_reference.py')]
                else: value['jobs'].append(copy.deepcopy(value['jobs'][0]))
                save(p, value)
                with self.assertRaises(ValueError): prepare.prepare(root, base, repo)
                self.assertFalse(root.exists())

    def test_direct_toy_row_uses_one_optional_density_and_preserves_trace_checks(self):
        import numpy as np
        import test_native_class_line_reference as fixtures
        reference = worker.reference
        args = fixtures.ConditionalLawTests().setup()
        old, fast = reference.Reconstructor(*args), reference.Reconstructor(*args, use_tree=True)
        u = np.zeros(6); density = old.density(u); actual = copy.deepcopy(density)
        actual['raw_coordinates'] = old.raw(u).tolist()
        actual['component_mixture_multipliers'] = ((np.asarray(density['component_multipliers'])-1+old.beta)/old.beta).tolist()
        for axis in actual['axes']:
            raw = old.raw(u); raw[axis['axis']] = 0.
            u0 = reference.scalar_chart_solve(old.L0, raw-old.m0)
            du = reference.scalar_chart_solve(old.L0, np.eye(6)[axis['axis']])
            axis.update(latent_line_origin=u0.tolist(), latent_line_direction=du.tolist())
            for channel in axis['channels']:
                channel['orthant_intervals'] = reference.orthant_intervals(u0, du, channel['orthant'], axis['segment']) if channel.get('orthant') is not None and 'segment' in axis else None
            for component in axis['components']:
                for branch in component['channels']: branch['fallback_target'] = branch.pop('fallback')
                component['channel_mixture_multiplier'] = sum(c['probability']*b['multiplier'] for c, b in zip(old.channels, component['channels']))
        raw, position, rotation, jacobian = old.decode(u)
        from native_contact_regions import make_pose
        row = dict(kind='fresh', id=0, latent=u.tolist(), latent_radius=0., raw_coordinates=raw.tolist(),
            log_physical_jacobian=jacobian, pose=make_pose(position, rotation),
            log_proposal_density=density['log_density'], baseline_log_density=density['baseline_log_density'],
            density_details=actual, draw=dict(conditional=False, original_latent=u.tolist()))
        with mock.patch.object(fast, 'density', wraps=fast.density) as evaluate:
            result = worker.compare_row(row, fast)
            self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(result['counts']['density_queries'], 1)
        self.assertEqual(result['counts']['axis_geometries'], 3)
        self.assertEqual(result['maxima']['interval_endpoint_error'], 0.)
        self.assertLessEqual(result['counts']['exclusion_evaluated_pairs'], result['counts']['exclusion_possible_pairs'])
        changed = copy.deepcopy(row); changed['draw']['original_latent'][0] = 1.
        with self.assertRaisesRegex(ValueError, 'Unconditioned draw changed'): worker.compare_row(changed, fast)
        with self.assertRaisesRegex(ValueError, 'Unpruned reconstruction'): worker.compare_row(row, old)

    def test_worker_refuses_imports_outside_archive_before_geometry(self):
        import run_evolving_dimer_analysis
        with tempfile.TemporaryDirectory() as tmp:
            base, repo = fixture(Path(tmp)); root = Path(tmp)/'frozen'; prepare.prepare(root, base, repo)
            with self.assertRaisesRegex(ValueError, 'outside its frozen namespace'):
                worker.verify_loaded_namespace(root)

    def test_freeze_exact_inventory_and_append_only_prefix_without_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, repo = fixture(Path(tmp)); root = Path(tmp)/'frozen'
            with mock.patch.object(worker.reference, 'Reconstructor', side_effect=AssertionError('No geometry')):
                receipt = prepare.prepare(root, base, repo); plan = prepare.verify(root)
            self.assertEqual(receipt['rows'], 56); self.assertEqual(receipt['geometry_queries'], 0)
            self.assertEqual([r['id'] for r in plan['rows'][:16]], list(range(16)))
            self.assertEqual([r['id'] for r in plan['rows'][16:]], ['saved-'+str(i) for i in range(40)])
            self.assertFalse(plan['full_independent_audit_complete'])
            with (base/'audits/hard_free-r00.journal.jsonl').open('a') as stream:
                stream.write('{"state":"begin","ordinal":16,"kind":"fresh","id":16}\n')
            prepare.verify(root)
            self.assertEqual(len((root/'prior-unpruned-prefix.jsonl').read_bytes().splitlines()), 33)
            with self.assertRaisesRegex(ValueError, 'Fresh'): prepare.prepare(root, base, repo)

    def test_incomplete_or_changed_prior_prefix_refuses_freeze(self):
        for mode in ('incomplete', 'wrong_id', 'wrong_q', 'wrong_input'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                base, repo = fixture(Path(tmp)); root = Path(tmp)/'frozen'; p = base/'audits/hard_free-r00.journal.jsonl'
                events = prepare.read_jsonl(p)
                if mode == 'incomplete': events.pop()
                elif mode == 'wrong_id': events[-1]['row']['id'] = 16
                elif mode == 'wrong_q': events[-1]['row']['log_density'] = 100.
                else: events[0]['input_sha256'][str(base/'queries/hard_free-r00/summary.json')] = '0'*64
                jsonl(p, events)
                with self.assertRaises(ValueError): prepare.prepare(root, base, repo)
                self.assertFalse(root.exists())

    def test_archive_scope_closure_rows_and_live_prefix_tampering_are_fatal(self):
        for mode in ('file', 'omit', 'cap', 'row', 'prefix'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                base, repo = fixture(Path(tmp)); root = Path(tmp)/'frozen'; prepare.prepare(root, base, repo)
                if mode == 'file': (root/'code/native_class_line_reference.py').write_text('# changed\n')
                elif mode == 'omit':
                    value = prepare.read(root/'freeze.json'); del value['files']['code/native_class_line_reference.py']; save(root/'freeze.json', value)
                elif mode == 'cap':
                    value = prepare.read(root/'allocation.json'); value['row_count'] = 57; save(root/'allocation.json', value)
                elif mode == 'row':
                    rows = prepare.read_jsonl(root/'selected-rows.jsonl'); rows[0]['id'] = 50; jsonl(root/'selected-rows.jsonl', rows)
                else:
                    p = base/'audits/hard_free-r00.journal.jsonl'; value = prepare.read_jsonl(p); value[-1]['cpu_seconds'] = 2.; jsonl(p, value)
                with self.assertRaises(ValueError): prepare.verify(root)

    def test_worker_fixed_56_journal_and_pending_correspondence_with_mocked_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, repo = fixture(Path(tmp)); root = Path(tmp)/'frozen'; prepare.prepare(root, base, repo)
            review = reviewed(root); claim(root, review)
            def row_result(row, recon, progress):
                return dict(log_density=row['log_proposal_density'], log_physical_jacobian=0.,
                    maxima=dict(interval_endpoint_error=0.), counts=dict(density_queries=1, axis_geometries=3))
            with (mock.patch.object(worker, '__file__', str(root/'code/run_native_class_projected_probe.py')),
                 mock.patch.object(worker, 'verify_loaded_namespace'),
                 mock.patch.object(worker, 'reconstructors', side_effect=mocked_models),
                 mock.patch.object(worker, 'compare_row', side_effect=row_result) as compare):
                worker.worker(root)
            self.assertEqual(compare.call_count, 56)
            result = prepare.read(root/'result.json'); events = prepare.read_jsonl(root/'journal.jsonl')
            self.assertEqual(result['completed_rows'], 56)
            self.assertEqual(result['counts'], dict(density_queries=56, axis_geometries=168))
            self.assertEqual([e['ordinal'] for e in events if e['state'] == 'begin'], list(range(56)))
            self.assertEqual([e['ordinal'] for e in events if e['state'] == 'complete'], list(range(56)))
            self.assertEqual(result['saved_unpruned_correspondence_pending_rows'], 40)
            self.assertFalse(result['full_independent_audit_complete'])

    def test_worker_failure_retains_exact_begin_and_stops_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, repo = fixture(Path(tmp)); root = Path(tmp)/'frozen'; prepare.prepare(root, base, repo)
            review = reviewed(root); claim(root, review)
            with (mock.patch.object(worker, '__file__', str(root/'code/run_native_class_projected_probe.py')),
                 mock.patch.object(worker, 'verify_loaded_namespace'),
                 mock.patch.object(worker, 'reconstructors', side_effect=mocked_models),
                 mock.patch.object(worker, 'compare_row', side_effect=ValueError('fixed row failed')) as compare):
                with self.assertRaisesRegex(ValueError, 'fixed row'): worker.worker(root)
            self.assertEqual(compare.call_count, 1)
            events = prepare.read_jsonl(root/'journal.jsonl')
            self.assertEqual([r['state'] for r in events], ['started', 'setup_begin', 'setup_complete', 'begin', 'failed'])
            self.assertEqual(events[-1]['current']['ordinal'], 0); self.assertEqual(events[-1]['retries'], 0)
            self.assertFalse((root/'result.json').exists())

    def test_review_required_and_controller_failure_is_nonretryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, repo = fixture(Path(tmp)); root = Path(tmp)/'frozen'; prepare.prepare(root, base, repo)
            review = reviewed(root); value = prepare.read(review); value['passed'] = False; save(review, value)
            with self.assertRaisesRegex(ValueError, 'review'): prepare.run(root, review)
            self.assertFalse((root/'execution-claim.json').exists())
            value['passed'] = True; save(review, value)
            with mock.patch('run_evolving_dimer_analysis.owned_child', side_effect=ValueError('synthetic child failure')) as child:
                with self.assertRaisesRegex(ValueError, 'synthetic child'): prepare.run(root, review)
                self.assertEqual(child.call_count, 1)
                self.assertEqual(child.call_args.args[3:5], (3600, 7200))
                with self.assertRaises(FileExistsError): prepare.run(root, review)
                self.assertEqual(child.call_count, 1)
            self.assertTrue(prepare.read(root/'failure.json')['partial_outputs_retained'])


if __name__ == '__main__': unittest.main()
