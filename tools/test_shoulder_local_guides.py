"""Focused controls for frozen-law selection and original-weight preservation."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import logsumexp

from fit_shoulder_local_guides import WINDOW, check_region, collect_rows, select_campaigns
from prepare_native_confirmation_atlas import geometric_floor, weighted_fit


class ShoulderLocalGuideTests(unittest.TestCase):
    def source_fixture(self, root):
        jobs, hashes = [], {}
        # Population masses differ by nine to one. Each has one zero that must
        # remain in its original denominator, even though it cannot affect a fit.
        for index, weight in enumerate((9., 1.)):
            folder = root/f'r{index:02}'
            folder.mkdir()
            positive = dict(draw=0, log_importance_weight=math.log(weight), q=1.05,
                hard_valid=True, capture_valid=True, region_valid=True, latent_radius=.4,
                assigned_to_other_center=True, pose={'position':[float(index),0.,0.], 'orientation':[1.,0.,0.,0.]})
            zero = dict(positive, draw=1, log_importance_weight=None, hard_valid=False)
            path = folder/'samples.jsonl'
            path.write_text(json.dumps(positive)+'\n'+json.dumps(zero)+'\n')
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            jobs.append(dict(id=f'r{index:02}', seed=index+101, samples=2, directory=str(folder)))
        expected = dict(draws=4, nonzero=2, logQ=math.log(10/4), weight_ESS=100/82,
                        maximum_fraction=.9, population_logQ_values=[math.log(9/2), math.log(1/2)])
        return dict(jobs=jobs), hashes, expected

    def test_whole_radius_laws_selected_without_shell_or_owner_pooling(self):
        direct = dict(complete=True, original_q_window=WINDOW,
            campaigns=[dict(root='direct-small',radius_A=.25),dict(root='direct-large',radius_A=.5)])
        assessment = dict(complete=True,original_q_window=WINDOW,campaigns=[
            dict(root=f'{name}-{radius}', owner=name, radius_A=radius)
            for name in ('mixture','geometry') for radius in (.25,.5)])
        chosen = select_campaigns(assessment,direct)
        self.assertEqual(chosen['direct']['root'],'direct-large')
        self.assertEqual(chosen['mixture']['root'],'mixture-0.5')
        self.assertEqual(chosen['geometry']['root'],'geometry-0.5')
        changed=copy.deepcopy(assessment)
        changed['campaigns'].append(dict(root='duplicate',owner='geometry',radius_A=.5))
        with self.assertRaises(ValueError): select_campaigns(changed,direct)

    def test_unconditional_zeros_original_weights_and_nonassigned_points_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            master, hashes, expected = self.source_fixture(Path(folder))
            rows, labels, logw, populations, _, reconstruction = collect_rows(master,hashes,expected,2,2)
        np.testing.assert_allclose(logw,[math.log(9),0])
        self.assertEqual(labels.tolist(),['r00','r01'])
        self.assertTrue(all(r['assigned_to_other_center'] for r in rows))
        self.assertEqual(reconstruction['original_full_N'],4)
        self.assertEqual(reconstruction['zero_rows'],2)
        self.assertAlmostEqual(reconstruction['original_logQ'],math.log(2.5))
        self.assertEqual([p['unconditional_draws'] for p in populations],[2,2])
        x=np.zeros((2,6));x[1,0]=1
        floor=geometric_floor(55.)
        fit=weighted_fit(x,logw,floor)
        self.assertAlmostEqual(fit['mean'][0],.1)
        self.assertNotAlmostEqual(fit['mean'][0],.5)  # Equal-population reweighting is wrong.
        np.testing.assert_allclose(fit['covariance']-fit['raw_covariance'],floor,atol=1e-15)

    def test_changed_row_hash_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            master, hashes, expected = self.source_fixture(Path(folder))
            path=Path(next(iter(hashes)));path.write_text(path.read_text().replace('"q": 1.05','"q":  1.05'))
            with self.assertRaises(ValueError): collect_rows(master,hashes,expected,2,2)

    def test_wrong_original_N_and_repeated_population_seed_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            master, hashes, expected = self.source_fixture(Path(folder))
            with self.assertRaises(ValueError): collect_rows(master,hashes,dict(expected,draws=2),2,2)
            master['jobs'][1]['seed']=master['jobs'][0]['seed']
            with self.assertRaises(ValueError): collect_rows(master,hashes,expected,2,2)

    def test_positive_weight_outside_strict_window_fails_even_with_new_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            master, hashes, expected = self.source_fixture(Path(folder))
            path=Path(next(iter(hashes)))
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['q']=1.
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError): collect_rows(master,hashes,expected,2,2)

    def test_other_radius_or_inner_hole_cannot_define_fit_support(self):
        cfg=dict(fixed_poses=[{'position':[0,0,0]}],capture_center=[0,0,0],capture_radius=18.,
                 reservoir_density=.035,depletant_radius=1.5,metadata={})
        region=dict(mahalanobis_radius=.5,minimum_mahalanobis_radius=0.,minimum_original_q=1.,
            maximum_original_q=1.1,minimum_original_q_inclusive=False,maximum_original_q_inclusive=False,
            fixed_neighbor=cfg['fixed_poses'][0],physical_fixed_neighbors=cfg['fixed_poses'],
            capture_center=cfg['capture_center'],capture_radius=18.,activity=.035,depletant_radius=1.5,
            physical_metric={},shape_sha256='shape',gaussian_chart={'shape_sha256':'shape'})
        check_region(region,cfg,'shape')
        with self.assertRaises(ValueError): check_region(dict(region,mahalanobis_radius=.25),cfg,'shape')
        with self.assertRaises(ValueError): check_region(dict(region,minimum_mahalanobis_radius=.25),cfg,'shape')


if __name__ == '__main__': unittest.main()
