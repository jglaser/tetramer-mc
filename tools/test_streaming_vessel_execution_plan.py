"""Deterministic scheduling controls; no frozen inputs are changed or launched."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_streaming_vessel_comparison as prep
import streaming_vessel_execution_plan as scheduling


def preparation(root):
    return dict(schema=prep.SCHEMA,preparation_only=True,physical_jobs_launched=0,dispatch_ready=False,
        jobs=prep.jobs_for(root),total_unconditional_draws=prep.TOTAL,
        stages=[dict(name=s,draws_per_population=n,independent_populations_per_arm=4) for s,n in prep.STAGES],
        arms=list(prep.ARMS),maximum_physical_workers=8,maximum_audit_workers=4,maximum_all_workers=32,
        thread_environment=copy.deepcopy(prep.THREADS),runtime=dict(python=prep.sys.executable),
        physical=dict(depletant_radius=1.5,activity=.035,lambda_ratio=128.,cloud_replicates=2,
            wall_center=[0.,0.,0.],wall_radius=prep.WALL,capture_radius=273,bath_wall_permeable=True,
            measure='Lebesgue center volume times normalized SO(3) Haar measure'),
        input_sha256={'guide.json':prep.GUIDE[1],'current_R4.json':prep.SOURCES['current_R4.json'][1],
                      'shape.json':prep.SOURCES['shape.json'][1]},native_definition_sha256=prep.NATIVE[1])


class StreamingExecutionPlanTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('/tmp/inert-streaming-execution-fixture')
        self.out,self.preparation = self.root/'workflow',self.root/'preparation'
        self.plan = preparation(self.preparation)
        self.preparation_hash,self.reader_hash = 'a'*64,'b'*64

    def build(self, plan=None):
        return scheduling.execution_plan(self.out,self.preparation,self.plan if plan is None else plan,
                                         self.preparation_hash,self.reader_hash)

    def validate(self, value):
        return scheduling.validate(value,self.out,self.preparation,self.plan,
                                    self.preparation_hash,self.reader_hash)

    def test_exact_frozen_jobs_stage_order_and_independent_streams(self):
        result = self.build()
        self.validate(result)
        self.assertEqual([s['name'] for s in result['stages']],['standard','large'])
        jobs = {j['id']:j for j in self.plan['jobs']}
        physical_ids = []
        for stage in result['stages']:
            self.assertEqual([g['kind'] for g in stage['groups']],['physical','audit','partition','aggregate'])
            self.assertEqual([g['workers'] for g in stage['groups']],[8,4,4,1])
            self.assertEqual([len(g['steps']) for g in stage['groups']],[8,8,8,1])
            for group in stage['groups'][:3]:
                kind = group['kind']
                for step in group['steps']:
                    job = jobs[step['job_id']]
                    self.assertEqual(job['stage'],stage['name'])
                    self.assertEqual(step['command'],job['command' if kind=='physical' else kind+'_command'])
                    self.assertEqual(step['directory'],job['directory' if kind=='physical' else kind+'_directory'])
                    if kind=='physical':
                        self.assertEqual(step['log'],job['log']); physical_ids.append(step['job_id'])
            for arm in prep.ARMS:
                arm_jobs = [jobs[j] for j in physical_ids if jobs[j]['stage']==stage['name'] and jobs[j]['arm']==arm]
                self.assertEqual([j['population'] for j in arm_jobs],list(range(4)))
        self.assertEqual(set(physical_ids),set(jobs))
        self.assertEqual(len({jobs[j]['seed'] for j in physical_ids}),16)
        self.assertEqual(sum(jobs[j]['samples'] for j in physical_ids),2621440)
        self.assertEqual(result['budgets'],dict(maximum_physical_workers=8,maximum_audit_workers=4,maximum_all_workers=32))

    def test_aggregate_command_pins_frozen_reader_and_preparation_hash(self):
        for stage in self.build()['stages']:
            step = stage['groups'][-1]['steps'][0]
            self.assertEqual(step['reader_sha256'],self.reader_hash)
            self.assertEqual(step['command'],[prep.sys.executable,'-B',str(self.out/'common'/scheduling.READER),
                '--preparation',str(self.preparation),'--preparation-sha256',self.preparation_hash,
                '--stage',stage['name'],'--out',str(self.out/(stage['name']+'-comparison'))])

    def test_pure_construction_never_opens_files_launches_or_mutates_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.out,self.preparation = root/'workflow',root/'preparation'
            self.plan = preparation(self.preparation); original = copy.deepcopy(self.plan)
            with patch.object(Path,'open',side_effect=AssertionError('Unexpected file access')), \
                 patch('subprocess.Popen',side_effect=AssertionError('Unexpected process launch')):
                result = self.build(); self.validate(result)
            self.assertEqual(list(root.iterdir()),[])
            self.assertEqual(self.plan,original)
            for name in ('dispatch_ready','gate_authority','launch_entry_point'):
                self.assertIs(result[name],False)
            # Returned mutable data must not alias commands or environment in preparation.
            result['stages'][0]['groups'][0]['steps'][0]['command'].append('--invalid')
            result['thread_environment']['OMP_NUM_THREADS'] = '999'
            self.assertEqual(self.plan,original)

    def test_plan_mutations_rejected_including_stage_completeness_and_budgets(self):
        mutations = [lambda x:x['stages'].reverse(),
            lambda x:x['stages'][0]['groups'].reverse(),
            lambda x:x['stages'][0]['groups'][0]['steps'].pop(),
            lambda x:x['stages'][0]['groups'][0].update(workers=9),
            lambda x:x['stages'][0]['groups'][1].update(workers=8),
            lambda x:x['budgets'].update(maximum_all_workers=33),
            lambda x:x['stages'][0]['groups'][0]['steps'][0]['command'].append('--different'),
            lambda x:x['stages'][0]['groups'][0]['steps'][0].update(directory='/tmp/other'),
            lambda x:x['stages'][0]['groups'][-1]['steps'][0]['command'].remove('--preparation-sha256'),
            lambda x:x['stages'][0]['groups'][-1]['steps'][0].update(reader_sha256='c'*64),
            lambda x:x['reader'].update(path='/mutable/repository/reader.py'),
            lambda x:x.update(dispatch_ready=True),lambda x:x.update(gate_authority=True)]
        for i,mutate in enumerate(mutations):
            value = self.build(); mutate(value)
            with self.subTest(mutation=i),self.assertRaises(ValueError): self.validate(value)

    def test_preparation_drift_and_missing_independent_populations_rejected(self):
        mutations = [lambda x:x['jobs'].pop(),
            lambda x:x['jobs'].append(copy.deepcopy(x['jobs'][0])),
            lambda x:x['jobs'][1].update(seed=x['jobs'][0]['seed']),
            lambda x:x['jobs'][1].update(population=3),
            lambda x:x['jobs'][0].update(samples=32),
            lambda x:x['jobs'][0]['audit_command'].append('--relaxed'),
            lambda x:x.update(maximum_audit_workers=8),
            lambda x:x.update(dispatch_ready=True),
            lambda x:x['physical'].update(lambda_ratio=64.),
            lambda x:x['runtime'].update(python='/different/python')]
        for i,mutate in enumerate(mutations):
            value = copy.deepcopy(self.plan); mutate(value)
            with self.subTest(mutation=i),self.assertRaises(ValueError): self.build(value)

    def test_digest_and_separate_directory_requirements(self):
        for digest in ('',None,'short','A'*64,'g'*64):
            with self.subTest(digest=digest),self.assertRaises(ValueError):
                scheduling.stages_for(self.out,self.preparation,self.plan,digest,self.reader_hash)
            with self.subTest(reader_digest=digest),self.assertRaises(ValueError):
                scheduling.stages_for(self.out,self.preparation,self.plan,self.preparation_hash,digest)
        for out in (self.preparation,self.preparation/'nested',self.preparation.parent):
            with self.subTest(out=out),self.assertRaises(ValueError):
                scheduling.stages_for(out,self.preparation,self.plan,self.preparation_hash,self.reader_hash)


if __name__ == '__main__':
    unittest.main()
