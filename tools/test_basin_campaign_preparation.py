"""Preparation controls: physical-neighbor preservation and frozen options."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from run_basin_normalizer_campaign import prepare, validate_inputs


class BasinCampaignPreparation(unittest.TestCase):
    def fixture(self, root):
        def pose(x): return {'position':[x,0,0], 'orientation':[1,0,0,0]}
        shape=root/'shape.json';shape.write_text('{"atoms":[{"center":[0,0,0],"radius":1}]}')
        binary=root/'binary';binary.write_text('unused during preparation')
        cfg={'shape':str(shape),'fixed_poses':[pose(0),pose(10)],'initial_pose':pose(4),
            'capture_center':[0,0,0],'capture_radius':18,'depletant_radius':1.5,'reservoir_density':.035,
            'poisson_lambda_ratio':64,'uniform_probability':.1,
            'metadata':{'native_poses':[pose(0)],'rigid_members':[pose(0)],'member_error_scale':2,'angle_error_scale_deg':15}}
        model={'shape_sha256':hashlib.sha256(shape.read_bytes()).hexdigest(),'coordinate_convention':'anchor-body-relative',
               'weights':[1.], 'anchors':[{'position':[0,0,0],'rotation':[[1,0,0],[0,1,0],[0,0,1]]}],
               'means':[[0]*6],'covariances':[[[float(i==j) for j in range(6)] for i in range(6)]]}
        config=root/'input.json';config.write_text(json.dumps(cfg))
        model_path=root/'model.json';model_path.write_text(json.dumps(model))
        args=argparse.Namespace(out=root/'campaign',binary=binary,config=config,model=model_path,
            lambda_ratio=None,proposal_anchor_index=0,scales=[1.],replicates=2,samples=8,seed=11,cloud_replicates=2,workers=2)
        return cfg,model,shape,args

    def test_selected_anchor_is_archived_but_all_neighbors_remain_physical(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,model,shape,args=self.fixture(root)
            out=prepare(args)
            manifest=json.loads((out/'manifest.json').read_text())
            frozen=json.loads((out/'provenance/config.json').read_text())
            self.assertEqual(frozen['fixed_poses'],cfg['fixed_poses'])
            self.assertEqual(manifest['physical']['fixed_neighbor_count'],2)
            self.assertEqual(manifest['proposal_anchor_index'],0)
            for job in manifest['jobs']:
                self.assertEqual(job['command'][-2:],['--proposal-anchor-index','0'])
                self.assertIn(str(out/'provenance/model.json'),job['command'])
            for name,digest in manifest['archive_sha256'].items():
                self.assertEqual(hashlib.sha256((out/'provenance'/name).read_bytes()).hexdigest(),digest)

    def test_default_keeps_all_anchor_command_and_validates_unselected_neighbor(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg,model,shape,args=self.fixture(root)
            args.proposal_anchor_index=None
            out=prepare(args)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertNotIn('--proposal-anchor-index',manifest['jobs'][0]['command'])
            bad=copy.deepcopy(cfg);bad['fixed_poses'][1]['orientation']=[0,0,0,0]
            with self.assertRaises(ValueError):validate_inputs(bad,model,shape,0)
            with self.assertRaises(ValueError):validate_inputs(cfg,model,shape,2)
            bad_model=copy.deepcopy(model);bad_model['shape_sha256']='wrong'
            with self.assertRaises(ValueError):validate_inputs(cfg,bad_model,shape,0)


if __name__=='__main__':unittest.main()
