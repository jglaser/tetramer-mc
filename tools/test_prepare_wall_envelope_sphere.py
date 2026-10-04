"""Small deterministic reference controls: no executable, sampling or geometry."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analyze_native_class_vessel_sphere as shared
import analyze_wall_envelope_sphere as analysis
import prepare_native_class_vessel_sphere as original
import prepare_wall_envelope_sphere as preparation


def synthetic_population(directory, draws=4, schema=8, activity=.4):
    """Handwritten admissible journal for reduction tests, not a proposal audit."""
    directory.mkdir()
    job = dict(id='synthetic',directory=str(directory),samples=draws,seed=17,activity=activity,population=0)
    plan = dict(executable=dict(sha256='a'*64),source_bundle=dict(sha256='b'*64))
    manifest = dict(schema=schema,samples=draws,seed=17,activity=activity,cloud_replicates=2,
        executable_sha256='a'*64,source_bundle_sha256='b'*64,pre_envelope_schema=7,
        vessel_uniform_schema='one-atom-wall-envelope-v1',
        vessel_uniform_envelope=dict(atom_index=0,atom_center=[0.,0.,0.],atom_radius=.3,
            wall_center=preparation.CENTER,wall_radius=3.,envelope_radius=2.7,log_volume=math.log(4*math.pi*2.7**3/3)),
        outer_mixture_schema='full-vessel-native-class-line-half-mixture-v1',outer_vessel_probability=.5,
        uniform_probability=.4,latent_defensive_uniform_probability=.5,
        latent_guide_schema='defensive-native-class-line-guide-v1',
        atomic_wall=dict(center=preparation.CENTER,radius=3.),
        density_measure='Lebesgue center volume times normalized SO(3) Haar measure',
        latent_reference_ball_is_target_restriction=False,bath_wall_permeable=True,
        latent_source_capture=dict(center=preparation.CENTER,radius=1.4,conditions_guide=True,restricts_target=False),
        **{'lambda':16*activity if activity else 1.})
    volume = original.overlap(1.); lam = manifest['lambda']; logq = math.log(.1)
    clouds = [dict(lower_volume=0.,upper_volume=volume,uncertain_volume=volume,
        created_cells=3,certified_cells=1,retained_cells=2,
        raw_points=n,overlap_points=n,log_weight=n*math.log1p(activity/lam))
        for n in (0,1 if activity else 0)]
    noisy = sum(math.exp(c['log_weight']-logq) for c in clouds)/2
    rows = []
    for i in range(draws):
        valid = i == 0
        position = list(preparation.CENTER); position[0] += 1. if valid else 0.
        rows.append(dict(draw=i,pose=dict(position=position,orientation=[1.,0.,0.,0.]),hard_valid=valid,
            log_proposal_density=logq,log_hard_weight=-logq if valid else None,
            log_importance_weight=math.log(noisy) if valid else None,clouds=clouds if valid else [],
            latent_density=dict(in_reference_ball=False,log_physical_jacobian=-3.)))
    (directory/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (directory/'attempts.jsonl').write_text(''.join(json.dumps(dict(draw=i,state='begin'))+'\n' for i in range(draws)))
    summary = dict(complete=True,manifest=manifest,numerical_nulls=0,samples=draws,hard_valid=1,
        samples_sha256=preparation.sha(directory/'samples.jsonl'),attempts_sha256=preparation.sha(directory/'attempts.jsonl'),
        estimates=dict(total=dict(log_normalizer=math.log(noisy/draws)),
            hard_total=dict(log_normalizer=math.log(10./draws))),sampler_cpu_seconds=.25)
    preparation.write(directory/'summary.json',summary); preparation.write(directory/'manifest.json',manifest)
    audit = dict(schema='full-vessel-native-class-line-independent-audit-v1',complete=True,population=str(directory),
        manifest=manifest,new_pose_draws=0,new_Poisson_clouds=0,source_sha256={},
        samples_sha256=summary['samples_sha256'],attempts_sha256=summary['attempts_sha256'])
    return job,plan,audit


class WallEnvelopeReferenceTests(unittest.TestCase):
    def test_fixed_allocation_commands_seeds_and_old_law_untouched(self):
        root = Path('/tmp/unlaunched-wall-envelope-reference'); python = '/synthetic/python'
        jobs = preparation.jobs_for(root,python)
        self.assertEqual(len(jobs),8); self.assertEqual(sum(j['samples'] for j in jobs),131072)
        self.assertEqual([j['samples'] for j in jobs],[12288]*4+[20480]*4)
        self.assertEqual([sum(j['samples'] for j in jobs if j['activity'] == z) for z in (0.,.4)],[49152,81920])
        self.assertEqual([j['seed'] for j in jobs],list(range(6900410001,6900410005))+list(range(6900411001,6900411005)))
        old = original.jobs_for(root,python)
        self.assertTrue(set(j['seed'] for j in jobs).isdisjoint(j['seed'] for j in old))
        self.assertEqual(original.DRAWS,2048); self.assertEqual(original.TOTAL,16384)
        for job in jobs:
            self.assertEqual(job['argv'].count('--wall-uniform-envelope'),1)
            self.assertEqual(job['argv'][job['argv'].index('--samples')+1],str(job['samples']))
            self.assertEqual(job['producer_terminal']['success_contract'],'complete')
            self.assertEqual(job['audit_terminal']['success_contract'],'complete')
        self.assertTrue(all('--wall-uniform-envelope' not in j['argv'] for j in old))

    def test_prospective_allocation_hash_and_law_mutations(self):
        self.assertEqual(preparation.sha(preparation.ALLOCATION),preparation.ALLOCATION_SHA)
        data = preparation.read(preparation.ALLOCATION); preparation.validate_allocation(data)
        for key,value in [('total_attempts',131073),('extensions',1),('rd',.6),('inner_uniform_probability',.5)]:
            damaged = copy.deepcopy(data); damaged[key] = value
            with self.subTest(key=key),self.assertRaises(ValueError): preparation.validate_allocation(damaged)
        damaged = copy.deepcopy(data); damaged['draws_per_population']['0.0'] = 20480
        with self.assertRaises(ValueError): preparation.validate_allocation(damaged)
        self.assertEqual(preparation.analytic_reference()['references'],original.references())
        self.assertEqual(preparation.CRITERIA,original.CRITERIA)

    def test_population_variable_attempts_keep_zeros_and_variance_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); results = []
            for n in (2,4):
                job,plan,audit = synthetic_population(root/f'n{n}',draws=n)
                result = shared.population(job,plan,audit,shared.Ledger(),root/f'companion{n}.jsonl',expected_manifest_schema=8)
                results.append(result)
                self.assertEqual(result['samples'],n)
                self.assertEqual(result['counts']['invalid_zeros'],n-1)
                self.assertEqual(result['regions']['total']['exact']['count'],n)
                self.assertAlmostEqual(result['regions']['total']['hard']['mean'],10/n)
                rows = [json.loads(line) for line in (root/f'companion{n}.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),n)
                self.assertEqual(sum(r['exact_importance'] > 0 for r in rows),1)
                self.assertTrue(all(r['exact_importance'] == r['noisy_importance'] == r['hard_importance'] == 0 for r in rows[1:]))
                known = math.sqrt(sum(r['conditional_residual_variance'] for r in rows))/n
                self.assertGreater(known,0)
                self.assertAlmostEqual(result['regions']['total']['known_conditional_residual_SE'],known)
            self.assertAlmostEqual(results[0]['regions']['total']['exact']['mean'],2*results[1]['regions']['total']['exact']['mean'])
            self.assertAlmostEqual(results[0]['regions']['total']['known_conditional_residual_SE'],2*results[1]['regions']['total']['known_conditional_residual_SE'])

    def test_default_schema7_hard_identity_and_schema8_explicit_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job,plan,audit = synthetic_population(root/'old',schema=7,activity=0.)
            result = shared.population(job,plan,audit,shared.Ledger(),root/'old-companion.jsonl')
            total = result['regions']['total']
            self.assertEqual(total['exact']['mean'],total['noisy']['mean'])
            self.assertAlmostEqual(total['hard']['mean'],2.5)
            self.assertEqual(total['known_conditional_residual_SE'],0.)
            job,plan,audit = synthetic_population(root/'new',schema=8)
            with self.assertRaisesRegex(ValueError,'population identity'):
                shared.population(job,plan,audit,shared.Ledger(),root/'wrong-schema.jsonl')
            self.assertFalse((root/'wrong-schema.jsonl').exists())
            manifest = audit['manifest']; analysis.check_manifest(manifest,job)
            for key,value in [('pre_envelope_schema',6),('uniform_probability',1.),('lambda',3.2)]:
                damaged = copy.deepcopy(manifest); damaged[key] = value
                with self.subTest(key=key),self.assertRaises(ValueError): analysis.check_manifest(damaged,job)
            damaged = copy.deepcopy(manifest); damaged['vessel_uniform_envelope']['atom_index'] = 1
            with self.assertRaisesRegex(ValueError,'witness'): analysis.check_manifest(damaged,job)

    def test_attempt_loss_fails_before_companion_reduction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); job,plan,audit = synthetic_population(root/'population')
            path = root/'population/attempts.jsonl'; path.write_text(path.read_text().splitlines()[0]+'\n')
            summary_path = root/'population/summary.json'; summary = preparation.read(summary_path)
            summary['attempts_sha256'] = preparation.sha(path); audit['attempts_sha256'] = summary['attempts_sha256']
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'denominator lost'):
                shared.population(job,plan,audit,shared.Ledger(),root/'companion.jsonl',expected_manifest_schema=8)
            self.assertFalse((root/'companion.jsonl').exists())

    def test_validation_receipt_drain_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'receipt.json'
            data = dict(complete=True,passed=True,error=None,jobs=[dict(returncode=0,child_drained=True)])
            preparation.write(path,data); digest = preparation.sha(path)
            self.assertEqual(preparation.passed_receipt(path,digest),data)
            data['jobs'][0]['child_drained'] = False; path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'hash'): preparation.passed_receipt(path,digest)
            with self.assertRaisesRegex(ValueError,'drained'): preparation.passed_receipt(path)

    def test_changed_plan_commands_are_rejected_before_population_reads(self):
        root = Path('/tmp/unlaunched-wall-envelope-reference'); python = '/synthetic/python'
        plan = dict(schema=preparation.SCHEMA,root=str(root),launched=False,preparation_only=True,
            criteria=preparation.CRITERIA,resource_limits=preparation.RESOURCE_LIMITS,
            physical=preparation.PHYSICAL,proposal=preparation.PROPOSAL,
            total_attempts=preparation.TOTAL,maximum_clouds=preparation.MAXIMUM_CLOUDS,
            populations_per_activity=4,samples_per_population_by_activity=preparation.DRAWS_BY_ACTIVITY,
            activities=list(preparation.ACTIVITIES),maximum_workers=1,threads=1,retries=0,replacements=0,extensions=0,
            jobs=preparation.jobs_for(root,python),python=python)
        plan['jobs'][0]['argv'].remove('--wall-uniform-envelope'); ledger = mock.Mock()
        with self.assertRaisesRegex(ValueError,'commands/seeds'): analysis.check_preparation(root,plan,{},ledger)
        ledger.frozen.assert_called_once_with(root)
        self.assertFalse(ledger.bind.called)


if __name__ == '__main__': unittest.main()
