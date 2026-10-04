"""Synthetic full worker lifecycle; prior metadata admission is mocked separately."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_conditional_native_audit as worker


def pose(x):
    return dict(position=[float(x), 0., 0.], orientation=[1., 0., 0., 0.])


class ToyNative:
    member_positions = [[0., 0., 0.]]
    motifs = [dict(id=0, relative_position=[1., 0., 0.], relative_orientation=[1., 0., 0., 0.])]
    definition = dict(criteria=dict(body_member_position_entry_A=.1))

    def __init__(self, fail=False):
        self.fail = fail

    def classify_pair(self, a, b):
        if self.fail:
            raise RuntimeError('synthetic classification failure')
        return [dict(motif_id=0)] if abs(b['position'][0]-a['position'][0]-1.) <= .1 else []


def fixture(root):
    def save(name, value):
        path = root/name
        worker.write(path, value)
        return worker.reference(path)

    source = save('source.json', dict(poses=[pose(0), pose(1), pose(2)]))
    definition = save('definition.json', {'synthetic': True})
    source_code = root/'frozen-source.py'
    source_code.write_text('# Synthetic source binding for lifecycle test only.\n')
    source_hashes = {'frozen-source.py': worker.sha(source_code)}
    chains = []
    for arm in worker.binding.ARMS:
        for initialization in worker.binding.STARTS:
            for stream in range(4):
                ident = dict(context_index=0, arm=arm, initialization=initialization, stream=stream)
                name = f'{arm}-{initialization}-{stream}'
                job = dict(id=len(chains), **ident)
                path = root/(name+'.jsonl')
                rows = [dict(kind='initial', block=0, conditional_target=True, job=job,
                             selected=[pose(0), pose(1)], sampler_cpu_seconds=0.)]
                for block in range(1, 5):
                    # Elementary details already belong to the prior authenticated
                    # replay. This worker must not try to re-evaluate that move.
                    rows.extend([dict(kind='local', opaque='prior validated'),
                        dict(kind='retained_block', block=block, production=block > 1,
                             selected=[pose(0), pose(1)], sampler_cpu_seconds=float(block))])
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                chains.append(dict(id=name, identity=ident, job=job, members=[0, 1],
                    initial=dict(kind='source', frame=source), trajectory=worker.reference(path),
                    terminal=definition, full_sampler_cpu_seconds=5., checkpoint_blocks=[2, 4],
                    geometry_load_cpu_seconds=.2, contact_observer_cpu_seconds=.3))
    inputs = {source['path']: source['sha256'], definition['path']: definition['sha256']}
    inputs.update({c['trajectory']['path']: c['trajectory']['sha256'] for c in chains})
    plan = dict(root=str(root), output=str(root/'analysis'), contact_root=str(root/'toy-contact'),
        initial_audit_root=str(root/'toy-initial'), native_inputs=dict(source_frame=source, definition=definition),
        setup_inventory={}, contexts=[dict(root=0, child=1)], fixed_native_keys_by_context={'0': []},
        chains=chains, allocation=dict(chains=32, blocks=4, warmup=1, retained_endpoints=160,
            production_endpoints=96, main_pair_query_cap=480, checkpoint_pair_query_cap=192,
            checkpoint_endpoints=64),
        limits=dict(cpu_seconds=60, wall_seconds=60, max_record_bytes=65536),
        input_sha256=inputs, source_sha256=source_hashes, runtime={'synthetic': True},
        inherited_preparation_costs={'preparation_cpu_seconds': 2.})
    save('audit-plan.json', plan)
    return plan, {'frozen-source.py': source_code}


class NativeWorkerPipelineTests(unittest.TestCase):
    def run_fixture(self, root, *, fail=False):
        plan, sources = fixture(root)
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(worker.binding, 'make_plan', return_value=copy.deepcopy(plan)))
            stack.enter_context(mock.patch.object(worker.binding, 'runtime', return_value=plan['runtime']))
            stack.enter_context(mock.patch.object(worker.binding, 'source_paths', return_value=sources, create=True))
            stack.enter_context(mock.patch.object(worker.binding, 'source_closure', return_value=plan['source_sha256']))
            stack.enter_context(mock.patch.object(worker.binding, 'checkpoint_blocks', return_value=[2, 4]))
            stack.enter_context(mock.patch.object(worker, 'live_authority'))
            stack.enter_context(mock.patch.object(worker, 'load_bounded_classifier',
                return_value=(ToyNative(fail=fail), {'synthetic': True})))
            return worker.run(root/'audit-plan.json', worker.sha(root/'audit-plan.json'), root/'analysis')

    def test_complete_streaming_pipeline_preserves_residence_and_cost_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_fixture(root)
            self.assertTrue(result['complete'] and result['passed'])
            self.assertEqual(len(result['chains']), 32)
            counts = result['query_counts']
            self.assertEqual((counts['endpoints_begun'], counts['endpoints_completed']), (160, 160))
            self.assertEqual(counts['production_endpoints'], 96)
            self.assertEqual(counts['checkpoint_endpoints'], 64)
            self.assertEqual(counts['main_begun'], 2)
            self.assertEqual(counts['checkpoint_begun'], 64)
            self.assertEqual(result['contexts'][0]['observer_counts']['whole_endpoint_cache_hits'], 159)
            self.assertEqual(result['contexts'][0]['observer_counts']['fixed_pair_calls'], 0)
            self.assertEqual(result['contexts'][0]['graph_records'], 1)
            for chain in result['chains']:
                rows = [json.loads(line) for line in Path(chain['endpoints']['path']).read_text().splitlines()]
                self.assertEqual([r['block'] for r in rows], list(range(5)))
                self.assertEqual(sum(r['production'] for r in rows), 3)
                self.assertEqual(chain['metrics']['full_sampler_cpu_seconds'], 5.)
                self.assertEqual([r['block'] for r in chain['checkpoints']], [2, 4])
            for path, digest in result['files'].items():
                self.assertEqual(worker.sha(path), digest)
            self.assertFalse((root/'analysis/failure.json').exists())

    def test_classifier_failure_keeps_attempt_and_partial_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'synthetic classification failure'):
                self.run_fixture(root, fail=True)
            failure = worker.read(root/'analysis/failure.json')
            self.assertFalse(failure['complete'])
            self.assertEqual(failure['completed_chains'], [])
            self.assertEqual(failure['query_counts']['main_begun'], 1)
            self.assertEqual(failure['query_counts'].get('main_completed', 0), 0)
            ledger = [json.loads(line) for line in Path(failure['ledger']['path']).read_text().splitlines()]
            self.assertEqual(ledger[-1]['state'], 'failure')
            self.assertTrue(any(r['state'] == 'query_begin' for r in ledger))
            self.assertEqual(len(list((root/'analysis/chains').glob('*.jsonl'))), 1)
            self.assertFalse((root/'analysis/summary.json').exists())
            self.assertEqual(worker.sha(failure['ledger']['path']), failure['ledger']['sha256'])


if __name__ == '__main__':
    unittest.main()
