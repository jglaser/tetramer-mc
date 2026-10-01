import copy
import json
from pathlib import Path
import tempfile
import unittest

import run_contact_tail_pilot as target


class TailPilotTests(unittest.TestCase):
    def protocol(self):
        return dict(schema=target.SCHEMA, arms=copy.deepcopy(target.ARMS),
            total_unconditional_draws=131072, maximum_physical_workers=2,
            maximum_total_workers=32, strata=copy.deepcopy(target.STRATA),
            convergence=copy.deepcopy(target.CONVERGENCE), full_vessel_gate_open=False,
            assembly_gate_open=False, jobs=[dict(arm=arm['id'], id=f'r{i:02}',
                samples=16384, seed=target.PILOT_SEEDS[4*a+i])
                for a, arm in enumerate(target.ARMS) for i in range(4)])

    def test_exact_declared_allocation(self):
        target.validate_design(self.protocol())

    def test_no_missing_zero_population_or_duplicate_seed(self):
        for mutation in ('remove', 'duplicate', 'size'):
            value = self.protocol()
            if mutation == 'remove': value['jobs'].pop()
            elif mutation == 'duplicate': value['jobs'][1]['seed'] = value['jobs'][0]['seed']
            else: value['jobs'][0]['samples'] = 16385
            with self.assertRaises(ValueError): target.validate_design(value)

    def test_no_threshold_or_thermodynamic_gate_changes(self):
        for mutation in ('threshold', 'vessel', 'assembly', 'workers'):
            value = self.protocol()
            if mutation == 'threshold': value['convergence']['importance_ESS_min'] = 20
            elif mutation == 'vessel': value['full_vessel_gate_open'] = True
            elif mutation == 'assembly': value['assembly_gate_open'] = True
            else: value['maximum_physical_workers'] = 8
            with self.assertRaises(ValueError): target.validate_design(value)

    def test_commands_bind_guide_two_clouds_and_original_counts(self):
        job = self.protocol()['jobs'][-1]; job['directory'] = '/tmp/reference-expanded'
        command = target.command('/tmp/frozen', target.ARMS[1], job)
        for flag, expected in [('--importance-guide', '/tmp/frozen/expanded/provenance/importance-guide.json'),
                               ('--samples', '16384'), ('--cloud-replicates', '2'),
                               ('--lambda-ratio', '128.0')]:
            self.assertEqual(command[command.index(flag)+1], expected)

    def test_frozen_evidence_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'input.json').write_text('{}')
            (root/'freeze.json').write_text(json.dumps({'files': {'input.json': target.sha(root/'input.json')}}))
            target.verify_frozen(root)
            (root/'input.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError): target.verify_frozen(root)

    def test_archive_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'freeze.json').write_text(json.dumps({'files': {'../other': 'x'}}))
            with self.assertRaises(ValueError): target.verify_frozen(root)


if __name__ == '__main__': unittest.main()
