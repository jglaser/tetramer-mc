#!/usr/bin/env python3
"""Preparation-only launcher controls; never starts a physical sampler."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_native_cover_mixture_analysis import metric,pose


class QWindowRunnerControls(unittest.TestCase):
    def invoke(self,path,extra):
        config=path/'config.json';shape=path/'shape.json';binary=path/'binary'
        shape.write_text('{}\n')
        config.write_text(json.dumps(dict(shape=str(shape),metadata=metric(),fixed_poses=[pose([10.,0.,0.])],
            capture_center=[0.,0.,0.],capture_radius=3.,depletant_radius=.5,reservoir_density=0.)))
        binary.write_text('#!/bin/sh\nprintf "%s\\n" "--q-min --q-max --q-lower-open --q-upper-open"\n');binary.chmod(0o755)
        command=[sys.executable,str(Path(__file__).with_name('run_native_region_reference.py')),
            '--root',str(path/'campaign'),'--config',str(config),'--binary',str(binary),'--samples','16',
            '--replicates','2','--seed-base','72101','--prepare-only']+extra
        return subprocess.run(command,capture_output=True,text=True)

    def test_default_has_no_new_window_flags(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);result=self.invoke(path,[]);self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((path/'campaign/manifest.json').read_text())
            self.assertNotIn('q_window',manifest)
            self.assertTrue(all('--q-min' not in job['command'] for job in manifest['jobs']))

    def test_window_frozen_in_protocol_commands_and_analyzer(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);result=self.invoke(path,['--q-min','1','--q-max','2','--q-lower-open','--q-upper-open'])
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((path/'campaign/manifest.json').read_text())
            self.assertEqual(manifest['q_window'],dict(minimum=1.,maximum=2.,lower_inclusive=False,upper_inclusive=False))
            self.assertEqual(manifest['cover_metric']['member_error_scale'],2.)
            self.assertEqual(manifest['cover_metric']['angle_error_scale_deg'],180.)
            self.assertEqual([j['seed'] for j in manifest['jobs']],[72101,73110])
            for job in manifest['jobs']:
                self.assertIn('--q-lower-open',job['command']);self.assertIn('--q-upper-open',job['command'])
                self.assertEqual(job['command'][job['command'].index('--q-max')+1],'2.0')
                self.assertEqual(job['command'][job['command'].index('--config')+1],str(path/'campaign/provenance/config.json'))
            frozen_config=json.loads((path/'campaign/provenance/config.json').read_text())
            self.assertEqual(frozen_config['shape'],str(path/'campaign/provenance/shape.json'))
            original=json.loads((path/'config.json').read_text())
            self.assertEqual(frozen_config['metadata'],original['metadata'])
            self.assertEqual(frozen_config['fixed_poses'],original['fixed_poses'])
            self.assertNotEqual(manifest['source_config_sha256'],manifest['config_sha256'])
            for name,digest in manifest['archive_sha256'].items():
                self.assertEqual(hashlib.sha256((path/'campaign/provenance'/name).read_bytes()).hexdigest(),digest)
            archived=path/'campaign/provenance/analyze_native_region_reference.py'
            self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(),manifest['analyzer_sha256'])
            self.assertEqual(list((path/'campaign/runs').iterdir()),[])

    def test_invalid_window_never_creates_campaign(self):
        for extra in [['--q-min','-1'],['--q-max','nan'],['--q-min','1','--q-max','1']]:
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as d:
                path=Path(d);result=self.invoke(path,extra)
                self.assertNotEqual(result.returncode,0);self.assertFalse((path/'campaign').exists())


if __name__=='__main__':unittest.main()
