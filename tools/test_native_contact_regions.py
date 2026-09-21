"""Independent comparisons with the archived native-entry observer; no MC."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from native_contact_regions import NativeContactRegions, compose, freeze_definition, make_pose, read


class NativeRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temp.name)/'definition'
        freeze_definition(cls.folder)
        cls.classifier = NativeContactRegions(cls.folder/'definition.json')
        cls.inputs = cls.folder/'inputs'
        scripts = cls.inputs/'reference/scripts'
        sys.path.insert(0,str(scripts))
        cls.observer_module = importlib.import_module('tetramer_order')
        cls.precursor = importlib.import_module('analyze_precursor_exchange')
        assert Path(cls.observer_module.__file__).resolve() == scripts/'tetramer_order.py'
        assert Path(cls.precursor.__file__).resolve() == scripts/'analyze_precursor_exchange.py'
        cls.cfg = dict(box_lengths=[4096.]*3, rigid_members=read(cls.inputs/'tetramer-shape.json')['rigid_members'],
            monomer_shape=str(cls.inputs/'monomer-shape.json'), native_pair_motifs=str(cls.inputs/'native-pair-motifs.json'))
        templates = cls.precursor.prepare_templates(cls.cfg['monomer_shape'],cls.inputs/'reference/results/c1c3-scaffold/motifs.json',
                                                   cls.inputs/'reference/results/native-neighbor-classes/classification.json')
        cls.order = cls.observer_module.TetramerOrder(cls.cfg,cls.inputs,templates=templates)
        cls.origin = make_pose([0.,0.,0.],np.eye(3))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.inputs/'reference/scripts'))
        # Avoid leaving cached modules referring to deleted temporary files.
        for name,module in list(sys.modules.items()):
            if str(getattr(module,'__file__','')).startswith(str(cls.inputs/'reference/scripts')):
                del sys.modules[name]
        cls.temp.cleanup()

    def oracle(self,anchor,moving):
        result=self.order.classify(dict(poses=[anchor,moving]),pair_filter=[(0,1)])
        return sorted(r['motif_id'] for r in result['registered_tetramer_motifs'] if r['entry'])

    def ours(self,anchor,moving):
        return sorted(r['motif_id'] for r in self.classifier.classify_pair(anchor,moving))

    def test_all_exact_motifs_and_reversed_body_order_match_archived_observer(self):
        for motif in self.classifier.motifs:
            relative=self.classifier.motif_poses[motif['id']]
            with self.subTest(motif=motif['id']):
                forward=self.ours(self.origin,relative)
                self.assertIn(motif['id'],forward)
                self.assertEqual(forward,self.oracle(self.origin,relative))
                backward=self.ours(relative,self.origin)
                self.assertTrue(backward)
                self.assertEqual(backward,self.oracle(relative,self.origin))

    def test_global_proper_isometry_and_quaternion_sign(self):
        transform=make_pose([103.,-47.,29.],Rotation.from_rotvec([.8,-.5,.2]).as_matrix())
        for index in (0,3,6,13):
            relative=self.classifier.motif_poses[index]
            anchor,moving=compose(transform,self.origin),compose(transform,relative)
            self.assertEqual(self.ours(self.origin,relative),self.ours(anchor,moving))
            self.assertEqual(self.ours(anchor,moving),self.oracle(anchor,moving))
            moving['orientation']=(-np.asarray(moving['orientation'])).tolist()
            self.assertEqual(self.ours(anchor,moving),self.oracle(anchor,moving))

    def test_nonmatching_and_stateless_repeated_queries(self):
        far=make_pose([500.,300.,-200.],Rotation.from_rotvec([.1,.7,-.2]).as_matrix())
        self.assertEqual(self.ours(self.origin,far),[])
        self.assertEqual(self.oracle(self.origin,far),[])
        native=read(self.inputs/'physical-config.json')['metadata']['native_poses'][0]
        before=self.classifier.classify(native)
        self.classifier.classify(far)
        self.assertEqual(before,self.classifier.classify(native))
        self.assertFalse(self.classifier.classify(far)['native_any'])

    def test_fixed_scaffold_cooperative_entry_preserves_both_anchor_matches(self):
        cfg=read(self.inputs/'physical-config.json')
        native=cfg['metadata']['native_poses'][0]
        result=self.classifier.classify(native)
        self.assertTrue(result['native_any'])
        self.assertTrue(result['cooperative_entry'])
        self.assertEqual(result['matched_anchor_indices'],[0,1])
        for i,anchor in enumerate(cfg['fixed_poses']):
            self.assertEqual(sorted(r['motif_id'] for r in result['matches'] if r['anchor_index']==i),self.oracle(anchor,native))
        self.assertTrue(result['registry_consistent_triangle'])
        self.assertTrue(all(r['supporting_member_bonds'] for r in result['matches']))

    def test_multiple_matching_motifs_are_not_collapsed(self):
        # A duplicate geometry with a distinct synthetic label exposes any
        # winner-take-all selection. It does not alter the frozen catalogue.
        model=copy.copy(self.classifier)
        duplicate=copy.deepcopy(model.motifs[0]);duplicate['id']=1000
        model.motifs=[*model.motifs,duplicate]
        model.motif_positions=np.vstack([model.motif_positions,model.motif_positions[0]])
        model.motif_rotations=np.concatenate([model.motif_rotations,model.motif_rotations[:1]])
        found=model.classify_pair(self.origin,model.motif_poses[0])
        self.assertIn(0,[r['motif_id'] for r in found]);self.assertIn(1000,[r['motif_id'] for r in found])

    def test_residue_patch_is_required_even_at_exact_body_motif(self):
        model=copy.copy(self.classifier);model.references=copy.deepcopy(model.references)
        for ref in model.references.values():ref['native_residue_pairs']={10**12}
        self.assertEqual(model.classify_pair(self.origin,model.motif_poses[0]),[])

    def test_entry_member_tolerance_is_not_retention_tolerance(self):
        # Isolate the declared pose gate from patch geometry using a positive
        # synthetic patch response. The real atomic patch has its own tests.
        model=copy.copy(self.classifier)
        all_pairs=set().union(*(r['native_residue_pairs'] for r in model.references.values()))
        model._contacts=lambda d,r,cutoff:dict(minimum_gap_A=.1,residue_pairs=sorted(all_pairs))
        for offset,expected in ((1.999,True),(2.001,False),(2.9,False)):
            moving=copy.deepcopy(model.motif_poses[0]);moving['position'][0]+=offset
            found=[m['motif_id'] for m in model.classify_pair(self.origin,moving)]
            self.assertEqual(0 in found,expected)

    def test_preserved_outside_capture_witness_is_native_without_q_filter(self):
        witness=dict(position=[-110.48995177833392,136.0113205273436,-2.7406713685702364],
            orientation=[.6882360353297444,-.1701385672833947,.5760388146466761,-.4068947180989217])
        result=self.classifier.classify(witness)
        self.assertGreater(np.linalg.norm(witness['position']),170.)
        self.assertTrue(result['native_any'])
        self.assertIn(0,[r['motif_id'] for r in result['matches'] if r['anchor_index']==0])
        self.assertEqual(self.ours(self.classifier.fixed_poses[0],witness),self.oracle(self.classifier.fixed_poses[0],witness))
        # This test classifies the preserved pose; it does not recompute its
        # previously established full tetramer hard or wall margins.

    def test_input_hash_and_criterion_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            import shutil
            target=Path(directory)/'copy';shutil.copytree(self.folder,target)
            p=target/'definition.json';definition=read(p);definition['criteria']['body_member_position_entry_A']=3.
            p.write_text(json.dumps(definition))
            with self.assertRaisesRegex(ValueError,'definition'):NativeContactRegions(p)
            shutil.copy2(self.folder/'definition.json',p)
            shape=target/'inputs/monomer-shape.json';shape.write_text(shape.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'input changed'):NativeContactRegions(p)


if __name__=='__main__':unittest.main()
