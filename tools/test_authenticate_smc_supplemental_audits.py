"""Synthetic provenance/recovery controls; no simulations or audit reruns."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import authenticate_smc_supplemental_audits as auth
from analyze_r4_smc_control import Ledger, read, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


class Fixture:
    def __init__(self, root):
        self.campaign = root / 'physical'
        self.root = root / 'supplement'
        binary = self.campaign / 'common/latent-region-smc'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'frozen executable\n')
        allocation = dict(populations=4, initial_draws=4, population=2, stages=2,
                          sweeps=1, cloud_replicates=2, lambda_ratio=128.)
        jobs = [dict(id=i, kind='physical', seed=100+k, status='pending',
                     directory=str(self.campaign/'populations'/i),
                     log=str(self.campaign/(i+'.log')), command=[str(binary), i])
                for k, i in enumerate(auth.IDS)]
        self.physical = dict(repository='/frozen/repository', python='/frozen/python',
                             allocation=allocation, jobs=jobs, binary_sha256=sha(binary))
        write(self.campaign/'plan.json', self.physical)
        physical_bindings = {}
        for identity in auth.IDS:
            directory = self.campaign/'populations'/identity
            summary = dict(initial_draws=4, initial_hits=2, completed_stage=2, log_Z=1.,
                terminal_particles=[dict(pose=dict(position=[k, 0, 0], orientation=[1, 0, 0, 0]))
                                    for k in range(2)])
            write(directory/'summary.json', summary)
            write(directory/'status.json', dict(complete=True, phase='complete'))
            write(directory/'manifest.json', dict(schema='synthetic'))
            for name in ['config', 'region', 'shape', 'source-bundle', 'initial-guide']:
                write(directory/'provenance'/(name+'.json'), dict(name=name))
            for name in ['initialization', 'stages', 'attempts']:
                (directory/(name+'.jsonl')).write_text('{}\n')
            files = [p for p in directory.rglob('*') if p.is_file()]
            required = {str(p):sha(p) for p in files}
            required[str(binary)] = sha(binary)
            physical_bindings[identity] = dict(directory=str(directory), required_audit_inputs=required)
        original_audits = []
        for identity in auth.IDS:
            destination = str(self.campaign/(identity+'-audit.json'))
            command = [self.physical['python'], '-B', str(self.campaign/'common/audit_hard_free_smc.py'),
                       '--directory', str(self.campaign/'populations'/identity), '--binary', str(binary),
                       '--out', destination]
            original_audits.append(dict(id=identity, kind='audit', directory=destination,
                log=str(self.campaign/(identity+'-audit.log')), command=command,
                status='complete' if identity in auth.REUSED else 'failed',
                returncode=0 if identity in auth.REUSED else 1))
            if identity in auth.REUSED:
                write(Path(destination), dict(complete=True, id=identity))
            else:
                (self.campaign/(identity+'-audit.log')).write_text('Frozen numerical failure\n')
        original_state = dict(complete=False, phase='independent_audit_failed',
            plan_sha256=sha(self.campaign/'plan.json'), audits=original_audits,
            jobs=[dict(j, status='complete', returncode=0) for j in jobs])
        write(self.campaign/'status.json', original_state)
        common = self.root/'common'
        common.mkdir(parents=True)
        for name in auth.AUDIT_SOURCE_NAMES:
            (common/name).write_text('# Synthetic immutable source\n')
        (common/'audit_hard_free_smc.py').write_text('\n'.join(
            'import '+Path(name).stem for name in sorted(auth.AUDIT_SOURCE_NAMES)
            if name != 'audit_hard_free_smc.py')+'\n')
        (common/'conditioned_density_audit.py').write_text('\n'.join(
            f'{name} = {value!r}' for name, value in auth.CONSTANTS.items())+'\n')
        for name in ['run.py', 'prepare.py', 'validation.json', 'frozen-tests.log']:
            (self.root/name).write_text('# Frozen validation fixture\n')
        inputs = {str(p):sha(p) for p in self.campaign.rglob('*')
                  if p.is_file() and p.suffix != '.jsonl'}
        supplemental_jobs = []
        for identity in auth.SUPPLEMENTED:
            destination = str(self.root/(identity+'-audit.json'))
            command = [self.physical['python'], '-B', str(common/'audit_hard_free_smc.py'),
                       '--directory', str(self.campaign/'populations'/identity), '--binary', str(binary),
                       '--out', destination, '--conditioned-density-audit']
            supplemental_jobs.append(dict(id=identity, kind='audit', status='pending',
                directory=destination, log=str(self.root/(identity+'-audit.log')), command=command))
        self.plan = dict(schema=auth.PLAN_SCHEMA, repository=self.physical['repository'],
            original_campaign=str(self.campaign), workers=2, physical_jobs=0, new_pose_draws=0, new_clouds=0,
            numerical_contract=auth.NUMERICAL_CONTRACT, input_sha256=inputs,
            physical=physical_bindings, jobs=supplemental_jobs,
            source_sha256={str(p.relative_to(self.root)):sha(p) for p in self.root.rglob('*') if p.is_file()},
            audit_sources={name:sha(common/name) for name in auth.AUDIT_SOURCE_NAMES},
            reused_audits=[dict(id=i, path=str(self.campaign/(i+'-audit.json')),
                               sha256=sha(self.campaign/(i+'-audit.json'))) for i in auth.REUSED])
        self.save_plan()
        complete_jobs = []
        for job in supplemental_jobs:
            identity = job['id']
            required = copy.deepcopy(physical_bindings[identity]['required_audit_inputs'])
            required.update({str(common/name):digest for name, digest in self.plan['audit_sources'].items()})
            receipt = dict(schema=auth.AUDIT_SCHEMA, complete=True, density_audit_mode='split_conditioned',
                directory=str(self.campaign/'populations'/identity), input_sha256=required,
                initial_draws=4, initial_hits=2, completed_stage=2, terminal_particles=2,
                attempted_events=12, initial_branches=dict(uniform=2, gaussian=2),
                new_pose_draws=0, new_Poisson_clouds=0, new_classifier_calls=0,
                guide_evaluations=4, analytic_sphere_cloud_bounds_checked=0, audit_cpu_seconds=.01,
                maximum_interval_error=0., maximum_inverse_CDF_error=0., log_Z=1.,
                conditioned_density_audit=dict(schema=auth.CONDITIONING_SCHEMA, queries=4,
                    certificates_retained=2, maxima=dict.fromkeys(auth.MAXIMUM_FIELDS, 0.)))
            write(Path(job['directory']), receipt)
            complete_jobs.append(dict(job, status='complete', returncode=0,
                                      receipt_sha256=sha(job['directory'])))
        self.state = dict(complete=True, phase='supplemental_audits_complete', plan_sha256=self.plan_sha,
                          physical_jobs=0, jobs=complete_jobs)
        self.save_state()
        self.calls = []

    def save_plan(self):
        write(self.root/'plan.json', self.plan)
        self.plan_sha = sha(self.root/'plan.json')

    def save_state(self):
        write(self.root/'status.json', self.state)

    def repin_plan(self):
        self.save_plan()
        self.state['plan_sha256'] = self.plan_sha
        self.save_state()

    def edit_receipt(self, mutate, identity='r02'):
        path = self.root/(identity+'-audit.json')
        receipt = read(path)
        mutate(receipt)
        write(path, receipt)
        next(j for j in self.state['jobs'] if j['id']==identity)['receipt_sha256'] = sha(path)
        self.save_state()

    def checked(self, out, plan, job, with_audit=False):
        """Stand in for the existing heavy checker, retaining real byte hashing."""
        self.calls.append((job['id'], with_audit))
        directory = Path(job['directory'])
        files = {str(p):sha(p) for p in directory.rglob('*') if p.is_file()}
        if with_audit:
            for p in [self.campaign/'common/latent-region-smc', self.campaign/(job['id']+'-audit.json')]:
                files[str(p)] = sha(p)
        return dict(input_sha256=files, initial_draws=4, initial_hits=2)

    def validate(self, ledger=None):
        return auth.validate_plan(self.root, self.plan_sha, self.campaign, self.physical, ledger or Ledger())

    def authenticate(self, ledger=None):
        with patch.object(auth.control, 'check_output', side_effect=self.checked):
            return auth.authenticate(self.root, self.plan_sha, self.campaign, self.physical, ledger or Ledger())


class SupplementalAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(Path(self.tmp.name))

    def test_terminal_authentication_preserves_original_failure(self):
        original = (self.f.campaign/'status.json').read_bytes()
        ledger = Ledger()
        outputs = self.f.authenticate(ledger)
        self.assertEqual(set(outputs), set(auth.IDS))
        self.assertEqual(self.f.calls, [('r00',True),('r01',True),('r02',False),('r03',False)])
        self.assertEqual((self.f.campaign/'status.json').read_bytes(), original)
        self.assertFalse((self.f.campaign/'r02-audit.json').exists())
        for identity in auth.SUPPLEMENTED:
            self.assertEqual(outputs[identity]['audit_mode'], 'split_conditioned')
            self.assertEqual(len(outputs[identity]['input_sha256']), 39)
            self.assertIn(str(self.f.campaign/(identity+'-audit.log')), ledger.files)
        self.assertIn(str(self.f.root/'status.json'), ledger.files)
        ledger.recheck()

    def test_preparation_allows_live_status_and_never_reads_estimates(self):
        self.f.state.update(complete=False, phase='supplemental_audits')
        self.f.save_state()
        original_read = auth.read
        def metadata_only(path):
            self.assertNotEqual(Path(path).name, 'summary.json')
            self.assertNotEqual(Path(path), self.f.root/'status.json')
            return original_read(path)
        ledger = Ledger()
        with patch.object(auth, 'read', side_effect=metadata_only), \
             patch.object(auth.control, 'check_output', side_effect=AssertionError('Not preparation-safe')):
            result = self.f.validate(ledger)
        self.assertEqual(result, self.f.plan)
        self.assertNotIn(str(self.f.root/'status.json'), ledger.files)
        self.assertFalse(any(p.endswith('.jsonl') for p in ledger.files))

    def test_pending_status_cannot_authenticate(self):
        self.f.state.update(complete=False, phase='supplemental_audits')
        self.f.save_state()
        with self.assertRaisesRegex(ValueError, 'not successfully complete'):
            self.f.authenticate()
        self.assertEqual(self.f.calls, [])

    def test_input_tampering_and_missing_failure_log(self):
        for target in ['populations/r02/provenance/shape.json', 'r03-audit.log']:
            with self.subTest(target=target):
                path = self.f.campaign/target
                original = path.read_bytes()
                path.write_bytes(original+b' changed')
                with self.assertRaises(ValueError): self.f.validate()
                path.write_bytes(original)
        path = self.f.campaign/'r03-audit.log'
        path.unlink()
        with self.assertRaises(FileNotFoundError): self.f.validate()

    def test_frozen_source_tampering_is_rejected(self):
        path = self.f.root/'common/conditioned_density_audit.py'
        path.write_text(path.read_text()+'# Changed\n')
        with self.assertRaises(ValueError): self.f.validate()

    def test_no_requirement_on_mutable_repository_sources(self):
        self.assertFalse(Path(self.f.physical['repository']).exists())
        self.f.validate()

    def test_missing_or_excess_supplemental_jobs_are_rejected(self):
        original = copy.deepcopy(self.f.plan['jobs'])
        for jobs in [original[:1], original+[dict(original[0], id='r00')], [original[0], original[0]]]:
            self.f.plan['jobs'] = jobs
            self.f.repin_plan()
            with self.subTest(ids=[j['id'] for j in jobs]), self.assertRaises(ValueError):
                self.f.validate()

    def test_failed_receipt_cannot_be_backfilled(self):
        write(self.f.campaign/'r02-audit.json', dict(complete=True))
        with self.assertRaisesRegex(ValueError, 'overwritten'): self.f.validate()

    def test_original_failed_status_cannot_be_relabelled_even_if_repinned(self):
        path = self.f.campaign/'status.json'
        state = read(path)
        state['audits'][2].update(status='complete', returncode=0)
        write(path, state)
        self.f.plan['input_sha256'][str(path)] = sha(path)
        self.f.repin_plan()
        with self.assertRaisesRegex(ValueError, 'relabeled'): self.f.validate()

    def test_relabelled_source_or_import_closure_is_rejected(self):
        original = copy.deepcopy(self.f.plan)
        self.f.plan['audit_sources']['counterfeit.py'] = self.f.plan['audit_sources'].pop('entry_shell_proposal.py')
        self.f.repin_plan()
        with self.assertRaisesRegex(ValueError, 'source names'): self.f.validate()
        self.f.plan = original
        path = self.f.root/'common/audit_hard_free_smc.py'
        path.write_text(path.read_text().replace('import entry_shell_proposal\n', ''))
        self.f.plan['source_sha256']['common/audit_hard_free_smc.py'] = sha(path)
        self.f.plan['audit_sources']['audit_hard_free_smc.py'] = sha(path)
        self.f.repin_plan()
        with self.assertRaisesRegex(ValueError, 'import closure'): self.f.validate()

    def test_changed_tolerances_rejected_even_when_new_source_hashes_agree(self):
        path = self.f.root/'common/conditioned_density_audit.py'
        path.write_text(path.read_text().replace('LOG_ENVELOPE_CAP = 2e-07', 'LOG_ENVELOPE_CAP = 1e-06'))
        self.f.plan['source_sha256']['common/conditioned_density_audit.py'] = sha(path)
        self.f.plan['audit_sources']['conditioned_density_audit.py'] = sha(path)
        self.f.repin_plan()
        with self.assertRaisesRegex(ValueError, 'tolerances'): self.f.validate()

    def test_wrong_numerical_contract_and_new_physical_allocation_fail(self):
        original = copy.deepcopy(self.f.plan)
        for key, value in [('numerical_contract','relaxed'), ('new_pose_draws',1), ('physical_jobs',1), ('new_clouds',1)]:
            self.f.plan = copy.deepcopy(original)
            self.f.plan[key] = value
            self.f.repin_plan()
            with self.subTest(key=key), self.assertRaises(ValueError): self.f.validate()

    def test_raw_byte_changes_are_detected_at_runtime(self):
        path = self.f.campaign/'populations/r02/attempts.jsonl'
        path.write_text('{}\n{}\n')
        self.f.validate()  # Metadata preparation does not scan these large streams.
        with self.assertRaisesRegex(ValueError, 'Raw physical byte'): self.f.authenticate()

    def test_receipt_digest_and_source_bindings_are_mandatory(self):
        path = self.f.root/'r02-audit.json'
        original = path.read_bytes()
        path.write_bytes(original+b' ')
        with self.assertRaises(ValueError): self.f.authenticate()
        path.write_bytes(original)
        def relabel(receipt):
            old = str(self.f.root/'common/entry_shell_proposal.py')
            receipt['input_sha256'][str(self.f.root/'common/fake.py')] = receipt['input_sha256'].pop(old)
        self.f.edit_receipt(relabel)
        with self.assertRaisesRegex(ValueError, 'omits, adds or relabels'): self.f.authenticate()

    def test_receipt_counters_mode_and_normalizer_are_checked(self):
        original = read(self.f.root/'r02-audit.json')
        changes = [('initial_draws',3), ('initial_hits',1), ('completed_stage',1), ('terminal_particles',1),
            ('attempted_events',11), ('initial_branches',dict(uniform=4, unknown=0)),
            ('new_pose_draws',1), ('new_Poisson_clouds',1), ('new_classifier_calls',1),
            ('density_audit_mode','strict'), ('complete',False), ('log_Z',1.1),
            ('maximum_interval_error',2e-9), ('maximum_inverse_CDF_error',2e-6),
            ('audit_cpu_seconds',float('nan')), ('guide_evaluations',True)]
        for key, value in changes:
            def mutate(receipt):
                receipt.clear(); receipt.update(copy.deepcopy(original)); receipt[key] = value
            self.f.edit_receipt(mutate)
            with self.subTest(key=key), self.assertRaises(ValueError): self.f.authenticate()

    def test_conditioning_query_certificate_and_caps_are_checked(self):
        original = read(self.f.root/'r02-audit.json')
        changes = [('queries',3), ('queries',9), ('certificates_retained',1), ('schema','wrong'),
                   ('log_envelope_width',2.01e-7), ('interval_endpoint_error',1.01e-9),
                   ('latent_coordinate_error',2e-8), ('independent_geometry_log_difference',1e-8),
                   ('jacobian_error',float('nan'))]
        for key, value in changes:
            def mutate(receipt):
                receipt.clear(); receipt.update(copy.deepcopy(original))
                conditioned = receipt['conditioned_density_audit']
                (conditioned['maxima'] if key in auth.MAXIMUM_FIELDS else conditioned)[key] = value
            self.f.edit_receipt(mutate)
            with self.subTest(key=key), self.assertRaises(ValueError): self.f.authenticate()

    def test_terminal_job_set_and_frozen_command_are_checked(self):
        original = copy.deepcopy(self.f.state)
        for mode in ['extra', 'failed', 'command', 'plan', 'physical']:
            self.f.state = copy.deepcopy(original)
            if mode=='extra': self.f.state['jobs'].append(dict(self.f.state['jobs'][0], id='r00'))
            elif mode=='failed': self.f.state['jobs'][0]['returncode'] = 1
            elif mode=='command': self.f.state['jobs'][0]['command'].pop()
            elif mode=='plan': self.f.state['plan_sha256'] = '0'*64
            else: self.f.state['physical_jobs'] = 1
            self.f.save_state()
            with self.subTest(mode=mode), self.assertRaises(ValueError): self.f.authenticate()


if __name__ == '__main__':
    unittest.main()
