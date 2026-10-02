"""Manifest checks and artifact-lineage controls with deterministic toy rows."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import analyze_streaming_vessel_stage as analysis
import partition_vessel_streaming as partition
from test_partition_vessel_streaming import prepared
from test_prepare_streaming_vessel_comparison import prep


def fixture(root,guided=False):
    common,inputs = root/'common',root/'inputs'; common.mkdir(); inputs.mkdir()
    audit,paths,definition = prepared(inputs,guided_arm=guided,binary_path=common/'basin-normalizer',native_directory='native-region')
    output = root/'partition'; partition.analyze(audit,paths,definition,output)
    population = inputs/'population'; manifest = analysis.read(population/'manifest.json')
    for saved,name in [('shape.json','shape.json'),('model.json','model.json')]:
        shutil.copy2(population/'provenance'/saved,inputs/name)
    shutil.copy2(population/'config.json',inputs/'config.json')
    shutil.copy2(population/'provenance/source-bundle.json',common/'source-bundle.json')
    if guided: shutil.copy2(population/'provenance/latent-guide.json',inputs/'guide.json')
    sources = {}
    for entry in ('audit_hard_free_vessel_streaming.py','audit_vessel_baseline_streaming.py','partition_vessel_streaming.py'):
        sources.update(analysis.local_sources(Path(__file__).parent/entry))
    for name,path in sources.items(): (common/name).symlink_to(path)
    plan = dict(sources={name:analysis.sha(path) for name,path in sources.items()},
        binary_sha256=manifest['executable_sha256'],source_bundle_sha256=manifest['source_bundle_sha256'],
        input_sha256={str(p.relative_to(inputs)):analysis.sha(p) for p in inputs.rglob('*') if p.is_file()},
        native_definition_sha256=analysis.sha(definition))
    job = dict(id='synthetic',arm='half_mixture' if guided else 'vessel',stage='standard',population=0,seed=123,
        samples=7,directory=str(population),audit_directory=str(audit.parent),partition_directory=str(output))
    return plan,job


class AuthenticationTests(unittest.TestCase):
    def test_complete_lineage_both_arms_without_replaying_geometry_or_native_search(self):
        for guided in (False,True):
            with self.subTest(guided=guided), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); plan,job = fixture(root,guided)
                # Real schema-4/6 geometry audits and complete synthetic native
                # partitions are used. Only protein-specific manifest constants
                # are separately tested below rather than applied to toy spheres.
                with patch.object(analysis,'validate_manifest') as manifest_check:
                    result = analysis.load_population(root,plan,job,analysis.Ledger())
                manifest_check.assert_called_once()
                self.assertEqual(result['samples'],7); self.assertEqual(result['sampler_CPU_seconds'],1.)
                self.assertEqual(result['partition_data']['new_native_classifier_calls'],4)
                self.assertEqual(result['partition_data']['new_geometry_queries'],0)
                self.assertEqual(result['estimates']['total']['Qz']['draws'],7)

    def test_changed_labels_incomplete_status_and_wrong_raw_binding_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); plan,job = fixture(root)
            labels = root/'partition/labels.jsonl'; labels.write_text(labels.read_text()+'{}\n')
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):
                analysis.load_population(root,plan,job,analysis.Ledger())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); plan,job = fixture(root)
            status_path = root/'partition/status.json'; status = analysis.read(status_path); status['complete'] = False
            analysis.write(status_path,status)
            freeze = analysis.read(root/'partition/freeze.json'); freeze['files']['status.json'] = analysis.sha(status_path)
            analysis.write(root/'partition/freeze.json',freeze)
            with self.assertRaisesRegex(ValueError,'status/hash'):
                analysis.load_population(root,plan,job,analysis.Ledger())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); plan,job = fixture(root)
            with (root/'inputs/population/samples.jsonl').open('a') as stream: stream.write('{}\n')
            with patch.object(analysis,'validate_manifest'), self.assertRaisesRegex(ValueError,'Hash mismatch'):
                analysis.load_population(root,plan,job,analysis.Ledger())

    def test_protein_manifest_contract_rejects_geometry_cloud_guide_and_journal_drift(self):
        plan = dict(physical=dict(activity=.035,lambda_ratio=128.,wall_center=[0.,0.,0.],wall_radius=prep.WALL),
            binary_sha256='binary',source_bundle_sha256='bundle',
            input_sha256={name:name for name in ('config.json','model.json','shape.json','current_R4.json','guide.json')})
        job = dict(samples=65536,seed=610031001,arm='half_mixture')
        manifest = dict(samples=65536,seed=610031001,cloud_replicates=2,covariance_scale=1.,uniform_probability=.1,
            proposal_anchor_index=None,executable_sha256='binary',source_bundle_sha256='bundle',config_sha256='config.json',
            model_sha256='model.json',shape_sha256='shape.json',activity=.035,**{'lambda':4.48},
            atomic_wall=dict(center=[0.,0.,0.],radius=prep.WALL),bath_wall_permeable=True,physical_fixed_neighbor_count=2,
            pose_proposal_schema=3,base_component_count=178,virtual_component_count=328,proposal_model_kind='reciprocal-pose-mixture-v1',
            attempt_journal='attempts.jsonl; begin before each attempt; no retries',resume_supported=False,
            schema=6,outer_mixture_schema='full-vessel-hard-free-line-half-mixture-v1',outer_vessel_probability=.5,
            latent_region_sha256='current_R4.json',latent_guide_sha256='guide.json',latent_guide_schema='defensive-hard-free-line-guide-v1',
            latent_gaussian_component_count=92,latent_defensive_uniform_probability=.5,latent_reference_ball_is_target_restriction=False,
            density_measure='Lebesgue center volume times normalized SO(3) Haar measure',
            latent_source_capture=dict(center=[0.,0.,0.],radius=170.,restricts_target=False,conditions_guide=True))
        analysis.validate_manifest(plan,job,manifest)
        for key,value in [('lambda',2.24),('activity',.04),('schema',5),('latent_gaussian_component_count',80),
                          ('latent_reference_ball_is_target_restriction',True),('attempt_journal',None),('cloud_replicates',1)]:
            with self.subTest(key=key), self.assertRaises(ValueError): analysis.validate_manifest(plan,job,dict(manifest,**{key:value}))
        baseline = {k:v for k,v in manifest.items() if not k.startswith(('latent_','outer_'))}
        baseline['schema'] = 4
        analysis.validate_manifest(plan,dict(job,arm='vessel'),baseline)
        with self.assertRaises(ValueError): analysis.validate_manifest(plan,dict(job,arm='vessel'),manifest)


if __name__ == '__main__': unittest.main()
