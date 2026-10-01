#!/usr/bin/env python3
"""Algebraic index-guide controls plus passive saved toy branch checks."""
import json
import math
from pathlib import Path
import unittest
import numpy as np


def example():
    # Rows are normalized component laws; columns are two physical and one
    # hard-invalid state. Conditioning transfers mass away from invalidity.
    rho=np.array([.3,.7])
    old=np.array([[.15,.25,.6],[.3,.1,.6]])
    new=np.array([[.25,.65,.1],[.75,.15,.1]])
    f=np.array([2.,.7,0.])
    return rho,old,new,f


class ExplicitIndexIdentities(unittest.TestCase):
    def test_unbiased_joint_weight_and_second_moment_sandwich(self):
        rho,old,new,f=example();Qo=rho@old;Qn=rho@new
        psi=rho[:,None]*old/Qo
        weight=f*psi/(rho[:,None]*new)
        self.assertTrue(np.allclose(psi.sum(axis=0),1))
        self.assertAlmostEqual(float(np.sum(rho[:,None]*new*weight)),float(f.sum()),places=14)
        S=np.sum(psi**2/(rho[:,None]*new),axis=0)
        support=f>0
        self.assertTrue(np.all(S[support]>=1/Qn[support]-1e-14))
        self.assertTrue(np.all(S[support]<=1/Qo[support]+1e-14))
        self.assertLessEqual(float(np.sum(f*f*S)),float(np.sum(f*f/Qo)))

    def test_chi_squared_identity_and_optimal_conditional(self):
        rho,old,new,_=example();Qo=rho@old;Qn=rho@new
        psi=rho[:,None]*old/Qo;optimal=rho[:,None]*new/Qn
        S=np.sum(psi*psi/(rho[:,None]*new),axis=0)
        chi=np.sum((psi-optimal)**2/optimal,axis=0)
        np.testing.assert_allclose(Qn*S,1+chi,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(np.sum(optimal**2/(rho[:,None]*new),axis=0),1/Qn,rtol=1e-14)

    def test_hard_invalid_support_is_essential_and_floor_identity(self):
        rho,old,new,_=example();Qo=rho@old
        psi=rho[:,None]*old/Qo;S=np.sum(psi**2/(rho[:,None]*new),axis=0)
        self.assertGreater(S[2],1/Qo[2])
        unchanged=np.sum(psi**2/(rho[:,None]*old),axis=0)
        np.testing.assert_allclose(unchanged,1/Qo,rtol=1e-14)

    def test_same_conditional_poisson_noise_preserves_second_moment_bound(self):
        rho,old,new,f=example();Qo=rho@old
        psi=rho[:,None]*old/Qo;S=np.sum(psi**2/(rho[:,None]*new),axis=0)
        # F=f*N/lambda, with N|x~Poisson(lambda(x)); the cloud law is
        # independent of the retained guide index conditional on x.
        intensity=np.array([.5,3.,1.]);moment=f*f*(1+1/intensity)
        self.assertLessEqual(float(moment@S),float(np.sum(moment/Qo)))


class ArchivedToyBranchControls(unittest.TestCase):
    def test_all_saved_toy_branches_without_geometry_reexecution(self):
        root=Path(__file__).resolve().parents[1]/'results/contact-arc-reference-validation-20261001'
        jobs=['two_spheres','partial_spectator','partial_spectator_floor1','blocked_spectator']
        if not all((root/job/'python.json').is_file() for job in jobs):
            self.skipTest('Archived toy scores are optional ignored validation artifacts, absent in this checkout')
        valid_branches=0;strict_gains=0;invalid_losses=0;fallbacks=0;queries=0
        for job in jobs:
            data=json.loads((root/job/'python.json').read_text())
            for row in data['rows']:
                queries+=1
                for arm in row['arms']:
                    for branch in arm['branches']:
                        old=branch['old_weighted_latent_log_density'];new=branch['new_weighted_latent_log_density']
                        old=-math.inf if old is None else old;new=-math.inf if new is None else new
                        if branch['outer_fallback'] or branch['arc_fallback']:
                            self.assertEqual(old,new);fallbacks+=1
                        if row['hard_valid']:
                            self.assertGreaterEqual(new,old-1e-13);valid_branches+=1
                            strict_gains+=int(new>old+1e-10)
                        elif new<old:invalid_losses+=1
                    if job.endswith('floor1'):
                        self.assertEqual(arm['log_density'],arm['old_distance_log_density'])
        self.assertEqual(queries,60);self.assertGreater(valid_branches,0)
        self.assertGreater(strict_gains,0);self.assertGreater(invalid_losses,0);self.assertGreater(fallbacks,0)


if __name__=='__main__':unittest.main()
