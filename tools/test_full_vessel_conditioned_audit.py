"""Exterior/seam and saved-law audit controls; no sampled poses or clouds."""
import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np

import conditioned_density_audit as split
import physical_hard_free_line_vessel as vessel
import audit_hard_free_vessel_streaming as stream
from test_conditioned_density_audit import fixture, SyntheticReference
from test_physical_hard_free_line_vessel import setup, trace, rows_fixture
from test_audit_hard_free_vessel_streaming import complete_fixture


def density_record(guide, density):
    return dict(latent=copy.deepcopy(density.latent),in_reference_ball=density.in_reference_ball,
        log_latent_density=density.log_latent_density,
        log_physical_jacobian=density.log_physical_jacobian,
        log_physical_density=density.log_physical_density,
        structural_zero=density.structural_zero,hard_free_line_density=trace(guide,density))


class FullVesselConditionedAuditTests(unittest.TestCase):
    def test_actual_exterior_zero_fallback_and_invalid_rows_preserved(self):
        cfg,manifest,rows,base,guide=rows_fixture()
        original=copy.deepcopy(rows)
        old=vessel.check_rows(cfg,manifest,rows,base,guide)
        new=vessel.check_rows(cfg,manifest,rows,base,guide,conditioned_density_audit=True)
        extra=new.pop('conditioned_density_audit')
        self.assertEqual(old,new)
        self.assertEqual(extra['checked_attempts'],4)
        self.assertEqual(new['structural_zero_queries'],1)
        self.assertEqual(new['valid_outside_R4'],2)
        self.assertEqual(original,rows)
        for field in ('log_proposal_density','log_hard_weight','log_importance_weight'):
            bad=copy.deepcopy(rows);bad[0][field]+=1e-3
            with self.assertRaises(ValueError):
                vessel.check_rows(cfg,manifest,bad,base,guide,conditioned_density_audit=True)
        bad=copy.deepcopy(rows);bad[-1]['log_importance_weight']=0.
        with self.assertRaisesRegex(ValueError,'explicit zero'):
            vessel.check_rows(cfg,manifest,bad,base,guide,conditioned_density_audit=True)

    def test_exact_seam_and_large_finite_near_seam_keep_support(self):
        guide,*_=setup(alpha=1.)
        seam=guide.evaluate(dict(position=[3.,0.,0.],orientation=[0.,1.,0.,0.]))
        self.assertTrue(split.audit_conditioned_density(density_record(guide,seam),seam,guide,
                                                       full_vessel=True)['complete'])
        near=guide.evaluate(dict(position=[3.,0.,0.],orientation=[1e-12,1.,0.,0.]))
        self.assertIsNotNone(near.latent)
        self.assertGreater(max(abs(x) for x in near.latent),1e10)
        record=density_record(guide,near)
        # The full-vessel check already permits scale-relative coordinate
        # roundoff. Neither q nor J is granted a larger density tolerance.
        k=int(np.argmax(np.abs(near.latent)));record['latent'][k]+=1.
        result=split.audit_conditioned_density(record,near,guide,full_vessel=True)
        self.assertTrue(result['complete']);self.assertTrue(near.structural_zero)
        with self.assertRaisesRegex(ValueError,'latent coordinates'):
            split.audit_conditioned_density(record,near,guide)
        record['log_physical_jacobian']+=1e-4
        with self.assertRaisesRegex(ValueError,'Jacobian'):
            split.audit_conditioned_density(record,near,guide,full_vessel=True)

    def test_negative_infinity_only_when_component_is_inactive(self):
        class Exterior(SyntheticReference):
            def gaussian_logs(self,u):return np.array([-math.inf,-math.inf])
        r=Exterior()
        record,expected,guide=fixture([split.line.interval(-1.,1.)],r=r,u=[5.,0.,0.,0.,0.,0.])
        self.assertTrue(expected.structural_zero)
        self.assertTrue(split.audit_conditioned_density(record,expected,guide,full_vessel=True)['complete'])
        with self.assertRaisesRegex(ValueError,'Invalid Gaussian'):
            split.audit_conditioned_density(record,expected,guide)
        # Empty intervals imply fallback, activating the Gaussian tail. A log
        # underflow there cannot be relabelled as genuine geometric zero.
        record,expected,guide=fixture([],r=r,u=[5.,0.,0.,0.,0.,0.])
        with self.assertRaisesRegex(ValueError,'positive Gaussian'):
            split.audit_conditioned_density(record,expected,guide,full_vessel=True)
        for invalid in (math.nan, math.inf):
            class Bad(Exterior):
                def gaussian_logs(self,u):return np.array([invalid,-math.inf])
            record,expected,guide=fixture([split.line.interval(-1.,1.)],r=Bad(),u=[5.,0.,0.,0.,0.,0.])
            with self.assertRaises(ValueError):
                split.audit_conditioned_density(record,expected,guide,full_vessel=True)

    def test_saved_interval_q_is_used_in_unchanged_outer_mixture(self):
        # Synthetic chart used to isolate saved-law arithmetic, not certify
        # the world/chart mapping. Actual mappings are tested above.
        ref=[split.line.interval(-1.3e-6,1.4e-6)]
        saved=[split.line.interval(-1.3e-6,1.4e-6+8e-14)]
        record,density,guide=fixture(ref,saved)
        cfg,manifest,rows,base,_=rows_fixture()
        row=copy.deepcopy(rows[0]);row.update(outer_branch='vessel',proposal={},latent_proposal=None)
        record['coordinate_chart_seam']=False
        row['latent_density']={k:v for k,v in record.items() if k!='log_physical_density'}
        row['log_latent_physical_density']=record['log_physical_density']
        q=float(np.logaddexp(row['log_vessel_proposal_density'],record['log_physical_density'])-math.log(2))
        row.update(log_proposal_density=q,log_hard_weight=-q,log_importance_weight=-q)
        guide=SimpleNamespace(recon=guide.recon,region={'capture_center':[0.,0.,0.],'capture_radius':170.},
                              evaluate_many=lambda poses:[density])
        local=dict(manifest,samples=1)
        result=vessel.check_rows(cfg,local,[row],base,guide,conditioned_density_audit=True)
        self.assertGreater(result['conditioned_density_audit']['maxima']['independent_geometry_log_difference'],split.LOG_ATOL)
        self.assertLess(result['maximum_log_density_error'],1e-12)
        self.assertEqual(row['log_importance_weight'],-q)
        # The old strict independent-geometry comparator must still reject;
        # the optional path did not silently relax the default.
        with self.assertRaises(ValueError):vessel.check_rows(cfg,local,[row],base,guide)

    def test_streaming_sums_and_maxima_cover_every_batch(self):
        with TemporaryDirectory() as name:
            root=Path(name);population=root/'population';binary=complete_fixture(population)
            first=stream.audit(population,root/'first',binary,2,conditioned_density_audit=True)
            second=stream.audit(population,root/'second',binary,4,conditioned_density_audit=True)
            self.assertEqual(first['density_audit'],second['density_audit'])
            self.assertEqual(first['estimates'],second['estimates'])
            self.assertEqual(first['density_audit']['conditioned_density_audit']['checked_attempts'],
                             first['manifest']['samples'])
            self.assertGreater(first['batching']['batches'],second['batching']['batches'])
            self.assertEqual(first['new_pose_draws'],0)
            self.assertEqual(first['new_Poisson_clouds'],0)


if __name__=='__main__':unittest.main()
