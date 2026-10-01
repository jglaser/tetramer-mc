import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analyze_hard_free_line_physical import relabel_nonuniform
from run_hard_free_line_physical_pilot import (ARMS, SEEDS, SCHEMA, STRATA, CONVERGENCE,
    REGION_SHA, SHAPE_SHA, inventory, physical_command, validate_design, verify_output)


def plan():
    return dict(schema=SCHEMA, arms=copy.deepcopy(ARMS), strata=copy.deepcopy(STRATA),
        convergence=copy.deepcopy(CONVERGENCE), total_unconditional_draws=131072,
        cloud_replicates=2, maximum_physical_workers=2, maximum_audit_workers=4,
        maximum_total_workers=32, physical_activity=.035, depletant_radius=1.5,
        region_sha256=REGION_SHA, shape_sha256=SHAPE_SHA,
        full_vessel_gate_open=False, assembly_gate_open=False,
        jobs=[dict(id=f'r{i%4:02}', arm=ARMS[i//4]['id'], seed=seed, samples=16384)
              for i, seed in enumerate(SEEDS)])


class PhysicalPilotContract(unittest.TestCase):
    def test_fixed_allocation_and_no_gate_relaxation(self):
        original=plan();validate_design(original)
        for key, value in [('total_unconditional_draws',65536),('cloud_replicates',1),
                ('maximum_audit_workers',32),('physical_activity',.0275),('depletant_radius',1.4),
                ('full_vessel_gate_open',True),('assembly_gate_open',True)]:
            bad=copy.deepcopy(original);bad[key]=value
            with self.assertRaises(ValueError,msg=key):validate_design(bad)
        for mutate in [lambda p:p['jobs'][0].update(samples=8192),
                       lambda p:p['jobs'][0].update(seed=SEEDS[1]),
                       lambda p:p['arms'][0].update(beta=.5),
                       lambda p:p['convergence'].update(extra_gate=False)]:
            bad=copy.deepcopy(original);mutate(bad)
            with self.assertRaises(ValueError):validate_design(bad)

    def test_commands_keep_physical_cloud_and_attempted_allocation(self):
        root=Path('/tmp/new-line-pilot')
        for arm in ARMS:
            j=dict(directory=str(root/arm['id']/'r00'),samples=16384,seed=SEEDS[0])
            command=physical_command(root,arm,j)
            self.assertEqual(command[command.index('--samples')+1],'16384')
            self.assertEqual(command[command.index('--cloud-replicates')+1],'2')
            self.assertEqual(command[command.index('--lambda-ratio')+1],'128')
            self.assertEqual(command[command.index('--importance-guide')+1],str(root/'common'/(arm['id']+'.json')))

    def test_seed_inventory_rejects_collision_across_independent_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a',Path(directory)/'b';a.mkdir();b.mkdir()
            (a/'protocol.json').write_text(json.dumps(dict(jobs=[dict(seed=17)])))
            (b/'declaration.json').write_text(json.dumps(dict(seeds=[23,24])))
            value=inventory([a,b],SEEDS)
            self.assertEqual(value['prior_seeds'],[17,23,24]);self.assertEqual(len(value['files']),2)
            with self.assertRaises(ValueError):inventory([a,b],[23])

    def test_branch_relabel_changes_no_estimate_or_raw_reference(self):
        original=dict(gaussian={'log_Qz':10.,'counts':[0,4]},raw_row='hard-free-line',
                      sub=[dict(gaussian={'draws':128})])
        snapshot=copy.deepcopy(original);converted=relabel_nonuniform(original)
        self.assertEqual(original,snapshot)
        self.assertEqual(converted['nonuniform_guide'],snapshot['gaussian'])
        self.assertEqual(converted['sub'][0]['nonuniform_guide']['draws'],128)
        self.assertEqual(converted['raw_row'],'hard-free-line')

    def test_physical_output_bound_to_declared_stream_and_guide(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);job=dict(directory=str(root/'run'),samples=16384,seed=42,arm='conditioned')
            protocol=dict(binary_sha256='binary',source_bundle_sha256='bundle')
            manifest=dict(schema='importance-latent-region-normalizer-v6',
                guide_schema='defensive-hard-free-line-guide-v1',samples=16384,seed=42,
                activity=.035,lambda_ratio=128.,cloud_replicates=2,importance_uniform_probability=.5,
                importance_component_count=92,region_sha256=REGION_SHA,shape_sha256=SHAPE_SHA,
                executable_sha256='binary',source_bundle_sha256='bundle',
                config_sha256='same',importance_guide_sha256='same')
            summary=dict(complete=True,samples=16384,manifest=manifest,
                samples_sha256='same',attempts_sha256='same')
            def read(path):return manifest if Path(path).name=='manifest.json' else summary
            with patch('run_hard_free_line_physical_pilot.read',side_effect=read),\
                 patch('run_hard_free_line_physical_pilot.sha',return_value='same'):
                self.assertEqual(verify_output(root,protocol,job)['samples_sha256'],'same')
                manifest['seed']=43
                with self.assertRaisesRegex(ValueError,'identity/target'):verify_output(root,protocol,job)
                manifest['seed']=42;manifest['importance_guide_sha256']='different-guide'
                with self.assertRaisesRegex(ValueError,'identity/target'):verify_output(root,protocol,job)


if __name__=='__main__':unittest.main()
