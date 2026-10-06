"""Deterministic baseline-audit controls; no random physical jobs."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from scipy.special import logsumexp

import audit_vessel_baseline_streaming as baseline
from test_audit_hard_free_vessel_streaming import complete_fixture
from test_wall_envelope_vessel import witness


def fixture(root):
    binary = complete_fixture(root); config = baseline.read(root/'config.json')
    manifest = baseline.read(root/'manifest.json'); manifest['schema'] = 4
    manifest.pop('outer_mixture_schema'); manifest.pop('outer_vessel_probability')
    for key in list(manifest):
        if key.startswith('latent_'): manifest.pop(key)
    manifest.update(attempt_journal='attempts.jsonl; begin before each attempt; no retries',resume_supported=False)
    vessel = baseline.baseline.VesselDensity(config,manifest,baseline.read(root/'provenance/model.json'))
    old_log = float(vessel.evaluate([config['initial_pose']])[1]['anchor_log_densities'][0,0])
    rows = [json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    for row in rows:
        logq = row['log_vessel_proposal_density']
        for key in ['outer_branch','latent_proposal','latent_density','log_vessel_proposal_density','log_latent_physical_density']:
            row.pop(key)
        row['log_proposal_density'] = logq
        if row['hard_valid']: row['log_hard_weight'] = row['log_importance_weight'] = -logq
        if row['proposal'] is None:
            row['proposal'] = dict(moving_index=0,anchor_index=1,null_reason=None,candidate=copy.deepcopy(row['pose']),
                branch='uniform',component_index=None,new_log_density=logq,old_log_density=old_log,
                log_reverse_forward=old_log-logq)
    (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    baseline.write(root/'manifest.json',manifest)
    summary = baseline.read(root/'summary.json'); summary['manifest'] = manifest
    summary['samples_sha256'] = baseline.sha(root/'samples.jsonl')
    total = float(logsumexp([r['log_hard_weight'] for r in rows if r['hard_valid']])-math.log(len(rows)))
    summary['estimates'] = {k:dict(log_normalizer=total) for k in ['total','hard_total']}
    baseline.write(root/'summary.json',summary)
    return binary,root/'provenance/latent-region.json'


def save_rows(root,rows):
    (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary = baseline.read(root/'summary.json')
    summary['samples_sha256'] = baseline.sha(root/'samples.jsonl')
    baseline.write(root/'summary.json',summary)


def envelope_fixture(root, *, activity=.4, pure_uniform=False):
    """Hand-authored poses and unequal counts; no target or Poisson sampling."""
    binary,region = fixture(root)
    manifest = baseline.read(root/'manifest.json'); shape = baseline.read(root/'provenance/shape.json')
    config = baseline.read(root/'config.json'); config['reservoir_density'] = activity
    baseline.write(root/'config.json',config); baseline.write(root/'provenance/input-config.json',config)
    manifest.update(schema=8,pre_envelope_schema=4,vessel_uniform_schema='one-atom-wall-envelope-v1',
        vessel_uniform_envelope=witness(shape,manifest['atomic_wall']),activity=activity,
        config_sha256=baseline.sha(root/'provenance/input-config.json'),
        **{'lambda':config['poisson_lambda_ratio']*activity if activity else 1.})
    if pure_uniform: manifest['uniform_probability'] = 1.
    vessel = baseline.baseline.VesselDensity(config,manifest,baseline.read(root/'provenance/model.json'),shape=shape)
    rows = [json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
    if pure_uniform: rows = rows[:5]  # Four valid plus one core-invalid zero, all inside the envelope.
    manifest['samples'] = len(rows)
    logs,geometry = vessel.evaluate([r['pose'] for r in rows])
    old = float(vessel.evaluate([config['initial_pose']])[1]['anchor_log_densities'][0,0])
    for i,row in enumerate(rows):
        new = float(logs[i]); uniform = bool(geometry['uniform_support'][i])
        row['log_proposal_density'] = new
        row['proposal'].update(branch='uniform' if uniform else 'learned',
            component_index=None if uniform else 0,new_log_density=new,old_log_density=old,
            log_reverse_forward=old-new)
        if row['hard_valid']:
            for j,cloud in enumerate(row['clouds']):
                if activity and row['depletion_contact']:
                    cloud.update(uncertain_volume=.75,upper_volume=cloud['lower_volume']+.75,
                                 raw_points=5+j,overlap_points=2+3*j,created_cells=2,retained_cells=1)
                cloud['log_weight'] = activity*cloud['lower_volume']+cloud['overlap_points']*math.log1p(activity/manifest['lambda'])
            row['log_hard_weight'] = -new
            row['log_importance_weight'] = float(logsumexp([c['log_weight'] for c in row['clouds']])-math.log(2)-new)
    (root/'attempts.jsonl').write_text(''.join(json.dumps(dict(draw=i,state='begin'))+'\n' for i in range(len(rows))))
    baseline.write(root/'manifest.json',manifest); save_rows(root,rows)
    summary = baseline.read(root/'summary.json')
    summary.update(manifest=manifest,samples=len(rows),attempts_sha256=baseline.sha(root/'attempts.jsonl'),
                   **baseline.row_counts(rows))
    summary['estimates'] = {name:dict(log_normalizer=float(logsumexp([r[field] for r in rows if r['hard_valid']])-math.log(len(rows))))
                           for name,field in [('total','log_importance_weight'),('hard_total','log_hard_weight')]}
    baseline.write(root/'summary.json',summary)
    return binary,region


class BaselineAuditTests(unittest.TestCase):
    def test_schema4_row_algebra_matches_existing_reference_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'population'; _,region = fixture(root)
            manifest = baseline.read(root/'manifest.json'); config = baseline.read(root/'config.json')
            rows = [json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines()]
            before = copy.deepcopy((manifest,rows))
            vessel = baseline.baseline.VesselDensity(config,manifest,baseline.read(root/'provenance/model.json'))
            reporting = baseline.ReportingChart(baseline.read(region))
            expected = baseline.baseline.check_rows(config,manifest,rows,vessel,reporting)
            actual = baseline.check_rows(config,manifest,rows,vessel,reporting)
            self.assertEqual(actual,expected); self.assertEqual((manifest,rows),before)

    def test_reporting_chart_changes_labels_but_not_target_or_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary,region = fixture(population)
            first = baseline.audit(population,root/'first',binary,region,2)
            alternate = baseline.read(region)
            alternate['gaussian_chart']['covariances'] = [[[100. if i == j else 0. for j in range(6)] for i in range(6)]]
            alternate['capture_radius'] = .01
            baseline.write(root/'alternate.json',alternate)
            second = baseline.audit(population,root/'second',binary,root/'alternate.json',64)
            self.assertEqual(first['estimates']['total'],second['estimates']['total'])
            self.assertNotEqual(first['estimates']['inside_R4']['Q0']['nonzero'],second['estimates']['inside_R4']['Q0']['nonzero'])
            self.assertEqual(first['estimates']['total']['Q0']['nonzero'],4)
            self.assertEqual(first['primitive_count_audit'],second['primitive_count_audit'])
            self.assertEqual(first['density_audit']['checked_attempts'],7)
            self.assertEqual(first['batching']['peak_rows'],2)
            self.assertEqual(second['batching']['peak_rows'],7)
            self.assertFalse(first['reporting_region_binding']['affects_proposal'])
            self.assertFalse(first['reporting_region_binding']['restricts_target'])

    def test_missing_journal_and_wrong_reporting_shape_fail_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary,region = fixture(population)
            bad = baseline.read(region); bad['shape_sha256'] = '0'*64
            baseline.write(root/'bad.json',bad)
            with self.assertRaisesRegex(ValueError,'Reporting shape/scaffold'):
                baseline.audit(population,root/'wrong-shape',binary,root/'bad.json')
            self.assertFalse((root/'wrong-shape').exists())
            manifest = baseline.read(population/'manifest.json'); manifest.pop('attempt_journal')
            baseline.write(population/'manifest.json',manifest)
            summary = baseline.read(population/'summary.json'); summary['manifest'] = manifest
            baseline.write(population/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'journal/domain absent'):
                baseline.audit(population,root/'no-journal',binary,region)
            self.assertFalse((root/'no-journal').exists())

    def test_an_extra_density_factor_fails_and_preserves_earlier_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary,region = fixture(population)
            rows = [json.loads(s) for s in (population/'samples.jsonl').read_text().splitlines()]
            rows[4]['log_hard_weight'] += .2
            (population/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            summary = baseline.read(population/'summary.json'); summary['samples_sha256'] = baseline.sha(population/'samples.jsonl')
            baseline.write(population/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'no mixture or extra J'):
                baseline.audit(population,root/'failed',binary,region,2)
            self.assertEqual(len((root/'failed/geometry.jsonl').read_text().splitlines()),4)
            self.assertEqual(baseline.read(root/'failed/status.json')['phase'],'failed')
            self.assertFalse((root/'failed/analysis.json').exists())


class WallEnvelopeBaselineTests(unittest.TestCase):
    def test_exact_contract_admits_only_schema4_or_schema8_predecessor4(self):
        base = dict(schema=4)
        good = dict(schema=8,pre_envelope_schema=4,vessel_uniform_schema='one-atom-wall-envelope-v1',
                    vessel_uniform_envelope={})
        baseline.validate_proposal_contract(base); baseline.validate_proposal_contract(good)
        bads = [dict(base,schema=value) for value in (4.,8.,True,5,6,7)]
        bads += [dict(good,pre_envelope_schema=value) for value in (4.,True,7)]
        bads += [dict(good,vessel_uniform_schema='cube'),dict(good,vessel_uniform_envelope=None)]
        for key in ('outer_mixture_schema','outer_vessel_probability','latent_region_sha256'):
            bads += [dict(base,**{key:None}),dict(good,**{key:None})]
        for key in ('pre_envelope_schema','vessel_uniform_schema','vessel_uniform_envelope'):
            bads.append(dict(base,**{key:good[key]}))
            missing = copy.deepcopy(good); missing.pop(key); bads.append(missing)
        for bad in bads:
            with self.subTest(manifest=bad),self.assertRaises(ValueError): baseline.validate_proposal_contract(bad)

    def test_complete_depletion_audit_batches_and_all_invalid_zero_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary,region = envelope_fixture(population)
            outputs = [baseline.audit(population,root/f'batch-{b}',binary,region,b) for b in (1,3,64)]
            for result in outputs:
                self.assertEqual(result['schema'],baseline.SCHEMA)
                self.assertEqual(result['manifest']['schema'],8)
                self.assertEqual(result['manifest']['pre_envelope_schema'],4)
                self.assertEqual(result['vessel_uniform_envelope'],result['manifest']['vessel_uniform_envelope'])
                self.assertEqual(result['estimates'],outputs[0]['estimates'])
                self.assertEqual(result['geometry_sha256'],outputs[0]['geometry_sha256'])
                self.assertEqual(result['density_audit']['checked_attempts'],7)
                self.assertEqual(result['estimates']['total']['Qz']['nonzero'],4)
                self.assertEqual(result['estimates']['total']['Qz']['draws'],7)
                self.assertGreater(result['estimates']['total']['Qz']['logQ'],result['estimates']['total']['Q0']['logQ'])
                for key in ('wall_rejected','capture_rejected','hard_rejected'):
                    self.assertEqual(result['primitive_count_audit'][key],1)
                self.assertEqual(result['primitive_count_audit']['raw_points'],11)
                self.assertEqual(result['new_pose_draws']+result['new_clouds']+result['new_native_classifier_calls'],0)
            self.assertEqual([r['batching']['peak_rows'] for r in outputs],[1,3,7])

    def test_pure_uniform_analytic_sphere_measure_keeps_core_invalid_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'
            binary,region = envelope_fixture(population,activity=0.,pure_uniform=True)
            result = baseline.audit(population,root/'audit',binary,region,2)
            expected = math.log(4*math.pi*9**3/3)+math.log(4/5)
            self.assertAlmostEqual(result['estimates']['total']['Q0']['logQ'],expected,places=12)
            self.assertEqual(result['estimates']['total']['Q0'],result['estimates']['total']['Qz'])
            self.assertEqual(result['estimates']['total']['Q0']['draws'],5)
            self.assertEqual(result['primitive_count_audit']['hard_rejected'],1)

    def test_changed_envelope_witness_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary,region = envelope_fixture(population)
            manifest = baseline.read(population/'manifest.json'); manifest['vessel_uniform_envelope']['log_volume'] += .1
            baseline.write(population/'manifest.json',manifest)
            summary = baseline.read(population/'summary.json'); summary['manifest'] = manifest
            baseline.write(population/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'Wall-envelope normalization'):
                baseline.audit(population,root/'failed',binary,region)
            self.assertFalse((root/'failed').exists())

    def test_full_density_jacobian_arithmetic_clouds_and_invalid_zero_are_checked(self):
        cases = [('cube','Full original vessel density'),('jacobian','no mixture or extra J'),
                 ('cloud','Arithmetic cloud-mean'),('zero','explicit zero'),('latent','outer-mixture generation')]
        for case,message in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); population = root/'population'; binary,region = envelope_fixture(population)
                rows = [json.loads(s) for s in (population/'samples.jsonl').read_text().splitlines()]
                if case == 'cube': rows[4]['log_proposal_density'] -= math.log(2.)
                if case == 'jacobian': rows[4]['log_hard_weight'] += .2
                if case == 'cloud': rows[4]['log_importance_weight'] = sum(c['log_weight'] for c in rows[4]['clouds'])/2-rows[4]['log_proposal_density']
                if case == 'zero': rows[5]['log_hard_weight'] = 0.
                if case == 'latent': rows[4]['latent_density'] = None
                save_rows(population,rows)
                with self.assertRaisesRegex(ValueError,message): baseline.audit(population,root/'failed',binary,region,2)
                self.assertEqual(len((root/'failed/geometry.jsonl').read_text().splitlines()),4)
                self.assertEqual(baseline.read(root/'failed/status.json')['phase'],'failed')
                self.assertFalse((root/'failed/analysis.json').exists())

    def test_uniform_generation_outside_envelope_is_rejected_even_with_learned_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); population = root/'population'; binary,region = envelope_fixture(population)
            rows = [json.loads(s) for s in (population/'samples.jsonl').read_text().splitlines()]
            rows[5]['proposal'].update(branch='uniform',component_index=None)
            save_rows(population,rows)
            with self.assertRaisesRegex(ValueError,'Uniform vessel generation metadata'):
                baseline.audit(population,root/'failed',binary,region,2)
            self.assertEqual(len((root/'failed/geometry.jsonl').read_text().splitlines()),4)


def saved_fixture_control(repository, out):
    """Opt-in: revisit 16 archived sphere poses twice; never execute a sampler."""
    repository,out = Path(repository),Path(out)
    population = repository/'results/wall-envelope-cli-validation-20261004/attempt01/pure-uniform'
    binary = Path('/tmp/tetramer-wall-envelope-20261004/debug/basin-normalizer')
    region = repository/'results/native-class-vessel-cli-validation-20261004/attempt01/populations/region.json'
    earlier = population.parent/'pure-uniform-audit.json'
    inputs = [population/name for name in ('manifest.json','summary.json','config.json','samples.jsonl','attempts.jsonl')]
    inputs += list((population/'provenance').glob('*.json'))+[binary,region,earlier]
    hashes = {str(p.resolve()):baseline.sha(p) for p in inputs}
    original = baseline.read(earlier); manifest = baseline.read(population/'manifest.json')
    baseline.require(original['complete'] and original['rows'] == manifest['samples'] == 16,
                     'Wrong archived pure-uniform sphere allocation')
    baseline.require(manifest['activity'] == 0. and manifest['uniform_probability'] == 1.,
                     'Expected hard-only pure-uniform archived limit')
    results = [baseline.audit(population,out/f'batch-{b}',binary,region,b) for b in (3,64)]
    baseline.require(results[0]['estimates'] == results[1]['estimates']
                     and results[0]['geometry_sha256'] == results[1]['geometry_sha256'],
                     'Archived baseline batch sizes changed result')
    for result in results:
        total = result['estimates']['total']
        baseline.require(total['Q0']['draws'] == 16 and total['Q0']['nonzero'] == 15
                         and total['Q0'] == total['Qz'],'Archived invalid-zero denominator differs')
        baseline.close(total['Q0']['logQ'],original['log_volume']+math.log(15/16),
                       'Archived analytic sphere measure differs')
        baseline.require(result['vessel_uniform_envelope'] == manifest['vessel_uniform_envelope'],
                         'Archived envelope witness differs')
    baseline.require(all(baseline.sha(p) == digest for p,digest in hashes.items()),'Archived fixture mutated')
    return dict(complete=True,passed=True,unique_saved_sphere_poses=16,saved_pose_checks=32,
                new_pose_draws=0,new_clouds=0,new_native_classifier_calls=0,
                maximum_log_density_error=max(r['density_audit']['maximum_log_density_error'] for r in results),
                input_sha256=hashes,analysis_sha256={str(out/f'batch-{b}'/'analysis.json'):
                    baseline.sha(out/f'batch-{b}'/'analysis.json') for b in (3,64)})


if __name__ == '__main__': unittest.main()
