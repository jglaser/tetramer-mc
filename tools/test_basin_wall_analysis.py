"""Full-wall schema and independently reconstructed atomic wall controls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analyze_basin_normalizers import audit_wall_domain, audit_campaign_domain, population
import test_basin_reciprocal_analysis as reciprocal_tests


class WallAudit(unittest.TestCase):
    def test_campaign_wall_cannot_be_omitted_at_population_launch(self):
        wall=dict(center=[7.,-4.,3.],radius=3.5)
        physical=dict(atomic_wall=wall,bath_wall_permeable=True)
        manifest=dict(schema=4,atomic_wall=wall,bath_wall_permeable=True)
        audit_campaign_domain(physical,manifest)
        audit_campaign_domain({},dict(schema=3))
        for p,m in [(physical,dict(schema=3)),({},manifest),
                    (physical,dict(manifest,bath_wall_permeable=False)),
                    (dict(physical,atomic_wall=dict(wall,radius=3.6)),manifest)]:
            with self.subTest(physical=p,manifest=m), self.assertRaises(AssertionError):audit_campaign_domain(p,m)

    def fixture(self, root):
        manifest, rows, job = reciprocal_tests.ReciprocalBasinAudit().fixture(root)
        manifest.update(schema=4,pose_proposal_schema=3,atomic_wall=dict(center=[7.,-4.,3.],radius=3.5),
                        shape_bound=.1,bath_wall_permeable=True)
        rejected=0
        for row in rows:
            row['wall_valid']=bool(np.linalg.norm(np.asarray(row['pose']['position'])-[7.,-4.,3.]) <= 3.4)
            if not row['wall_valid']:
                rejected+=int(row['capture_valid']);row.update(hard_valid=False,q=None,region=None,
                    depletion_contact=None,log_hard_weight=None,log_importance_weight=None,clouds=[])
        summary=dict(complete=True,manifest=manifest,numerical_nulls=0,wall_rejected=rejected,estimates={},sampler_cpu_seconds=1.)
        return manifest, rows, job, summary

    def write(self, root, manifest, rows, summary):
        (root/'manifest.json').write_text(json.dumps(manifest))
        (root/'summary.json').write_text(json.dumps(summary))
        (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_full_wall_population_retains_invalid_zero_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job,summary=self.fixture(root);self.write(root,m,rows,summary)
            result=population(job)
            self.assertEqual(result['estimates']['total']['draws'],7)
            self.assertEqual(result['estimates']['total']['nonzero'],3)
            self.assertEqual(result['proposal_audit']['wall_domain']['wall_rejected_inside_capture'],2)

    def test_refuses_wrong_wall_flags_nonzero_invalids_and_clipped_bath(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job,summary=self.fixture(root)
            for index,key,value in [(2,'wall_valid',True),(2,'log_hard_weight',1.),(1,'wall_valid',False)]:
                with self.subTest(index=index,key=key):
                    bad=copy.deepcopy(rows);bad[index][key]=value
                    with self.assertRaises(AssertionError):audit_wall_domain(root,m,bad,summary)
            with self.assertRaises(AssertionError):audit_wall_domain(root,dict(m,bath_wall_permeable=False),rows,summary)

    def test_refuses_truncated_support_and_outside_fixed_scaffold(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job,summary=self.fixture(root)
            path=root/'config.json';config=json.loads(path.read_text());original=copy.deepcopy(config)
            config['capture_radius']=3.;path.write_text(json.dumps(config))
            with self.assertRaises(AssertionError):audit_wall_domain(root,m,rows,summary)
            config=original;config['fixed_poses'][0]['position']=[11.,-4.,3.];path.write_text(json.dumps(config))
            with self.assertRaisesRegex(AssertionError,'Fixed scaffold'):audit_wall_domain(root,m,rows,summary)

    def test_wrong_domain_schema_and_source_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows,job,summary=self.fixture(root)
            for key,value in [('pose_proposal_schema',1),('source_bundle_sha256','0'*64),('schema',5),('schema',3)]:
                with self.subTest(key=key):
                    bad=dict(m);bad[key]=value;record=dict(summary,manifest=bad);self.write(root,bad,rows,record)
                    with self.assertRaises(AssertionError):population(job)


if __name__=='__main__':unittest.main()
