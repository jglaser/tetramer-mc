"""Analytic Gaussian/Haar controls for the schema2 production audit."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from analyze_basin_normalizers import audit_selected_anchor, sha


class SelectedAnchorAudit(unittest.TestCase):
    def fixture(self, root):
        def save(path,value):path.write_text(json.dumps(value))
        def pose(x):return {'position':[7+x,-4,3],'orientation':[1,0,0,0]}
        (root/'provenance').mkdir()
        shape=root/'provenance/shape.json';save(shape,{'atoms':[{'center':[0,0,0],'radius':.1}]})
        cfg={'fixed_poses':[pose(0),pose(15)],'capture_center':[7,-4,3],'capture_radius':2.,'reservoir_density':.4,
             'depletant_radius':.7,'metadata':{'native_poses':[pose(0)],'rigid_members':[{'position':[0,0,0],'orientation':[1,0,0,0]}],
                 'member_error_scale':2.,'angle_error_scale_deg':30.}}
        save(root/'config.json',cfg);save(root/'provenance/input-config.json',cfg)
        model={'shape_sha256':sha(shape),'coordinate_convention':'anchor-body-relative','angular_length':1.,'weights':[1.],
               'anchors':[{'position':[0,0,0],'rotation':[[1,0,0],[0,1,0],[0,0,1]]}],
               'means':[[0.]*6],'covariances':[[[float(i==j) for j in range(6)] for i in range(6)]]}
        save(root/'provenance/model.json',model)
        manifest={'schema':2,'proposal_anchor_index':0,'physical_fixed_neighbor_count':2,'activity':.4,
                  'config_sha256':sha(root/'provenance/input-config.json'),'model_sha256':sha(root/'provenance/model.json'),
                  'shape_sha256':sha(shape),'covariance_scale':2.,'uniform_probability':.2}
        rows=[]
        for x in [0.,1.,2.,-2.,2.5]:
            # Six independent N(0,4) variables and Cayley/Haar Jacobian pi²
            # at identity: density exp(-x²/8)/(8*pi*2^6).
            gaussian=math.exp(-x*x/8)/(8*math.pi*2**6)
            density=.8*gaussian+(.2/4**3 if -2<=x<2 else 0)
            valid=0<x<=2 or x==-2
            rows.append({'pose':pose(x),'log_proposal_density':math.log(density),'capture_valid':abs(x)<=2,
                         'hard_valid':valid,'q':abs(x)/2 if valid else None,
                         'region':'native_core_bound' if valid and abs(x)<2 else 'native_shell_bound' if valid else None})
        return manifest,rows

    def test_full_density_capture_boundaries_and_original_q(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows=self.fixture(root)
            result=audit_selected_anchor(root,{'proposal_anchor_index':0},m,rows)
            self.assertEqual(result['checked_actual_poses'],5)
            self.assertEqual(result['checked_q'],3)
            self.assertLess(result['maximum_log_density_error'],1e-12)

    def test_density_q_anchor_and_input_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows=self.fixture(root);job={'proposal_anchor_index':0}
            bad=copy.deepcopy(rows);bad[0]['log_proposal_density']+=.01
            with self.assertRaises(AssertionError):audit_selected_anchor(root,job,m,bad)
            bad=copy.deepcopy(rows);bad[1]['q']+=.01
            with self.assertRaises(AssertionError):audit_selected_anchor(root,job,m,bad)
            with self.assertRaises(AssertionError):audit_selected_anchor(root,{'proposal_anchor_index':1},m,rows)
            bad=copy.deepcopy(rows);bad[0]['pose']=None
            with self.assertRaisesRegex(AssertionError,'null'):audit_selected_anchor(root,job,m,bad)
            cfg=json.loads((root/'config.json').read_text());cfg['fixed_poses'][1]['position'][0]+=.1
            (root/'config.json').write_text(json.dumps(cfg))
            with self.assertRaises(AssertionError):audit_selected_anchor(root,job,m,rows)

    def test_gaussian_only_and_model_hash_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);m,rows=self.fixture(root);path=root/'provenance/model.json'
            model=json.loads(path.read_text());model['dfs']=[3.]
            path.write_text(json.dumps(model))
            with self.assertRaises(AssertionError):audit_selected_anchor(root,{'proposal_anchor_index':0},m,rows)
            m['model_sha256']=sha(path)
            with self.assertRaisesRegex(AssertionError,'Gaussian'):audit_selected_anchor(root,{'proposal_anchor_index':0},m,rows)


if __name__=='__main__':unittest.main()
