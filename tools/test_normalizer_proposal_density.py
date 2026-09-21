"""Effective-factor arithmetic controls and a bounded saved-row regression."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import logsumexp

from normalizer_proposal_density import (NormalizerProposalDensity, bind_source_bundle,
    certify_covariance, scalar_cholesky)
from prepare_smc_normalizer_atlas import Density, relative_poses

ROOT=Path(__file__).resolve().parents[1]
POP=ROOT/'runs/mobile-full-capture-campaign-20260921/epsilon-0p1/runs/r00'


def model(covariance):
    base=dict(coordinate_convention='anchor-body-relative',angular_length=1.7,weights=[1.],
        anchors=[dict(position=[.2,-.3,.4],rotation=np.eye(3).tolist())],
        means=[[.1,-.1,.2,.05,-.1,.2]],covariances=[np.asarray(covariance).tolist()])
    return dict(schema='reciprocal-pose-mixture-v1',base_model=base,reciprocal_components=[True])


class EffectiveFactorTests(unittest.TestCase):
    def test_well_conditioned_density_unchanged_and_virtual_branches_normalized(self):
        m=model(np.diag([.4,.8,1.2,.7,.9,1.3])**2)
        d=NormalizerProposalDensity(m)
        poses=[dict(position=[.6,-.2,.1],orientation=[1.,0.,0.,0.]),
               dict(position=[-.4,.3,.8],orientation=[.5,.5,.5,.5])]
        np.testing.assert_allclose(d.evaluate(poses)[0],Density(m).evaluate(poses)[0],rtol=0,atol=1e-13)
        np.testing.assert_array_equal(d.lower,Density(m).lower)
        self.assertEqual(list(d.weights),[.5,.5])
        self.assertLessEqual(d.factor_diagnostics[0]['maximum_componentwise_bound_ratio'],1.)

    def test_ill_conditioned_covariance_has_independent_roundoff_certificate(self):
        q,_=np.linalg.qr(np.random.default_rng(473).normal(size=(6,6)))
        covariance=q@np.diag([1e-7,3e-6,.002,.3,8.,100.])@q.T
        d=NormalizerProposalDensity(model(covariance));r=d.factor_diagnostics[0]
        self.assertGreater(r['covariance_condition'],1e8)
        self.assertEqual(r['componentwise_entries_checked'],21)
        self.assertLessEqual(r['maximum_componentwise_bound_ratio'],1.)
        self.assertGreater(r['lapack_whitened_covariance_maximum_eigenvalue_deviation'],1e-10)
        self.assertGreater(r['maximum_derived_roundoff_bound'],r['maximum_absolute_covariance_residual'])

    def test_perturbed_factor_fails_componentwise_not_norm_tolerance(self):
        covariance=np.diag([1.,2.,3.,.1,.2,.3])
        lower=scalar_cholesky(covariance);lower[5,0]=1e-10
        with self.assertRaisesRegex(ValueError,'certificate failed'):certify_covariance(covariance,lower)
        lower=scalar_cholesky(covariance);lower[0,0]*=1+1e-10
        with self.assertRaisesRegex(ValueError,'certificate failed'):certify_covariance(covariance,lower)

    def test_symmetric_rounding_and_invalid_covariance(self):
        covariance=np.eye(6);covariance[1,0]=.2;covariance[0,1]=.2+1e-14
        certificate=certify_covariance(covariance,scalar_cholesky(covariance))
        self.assertGreater(certificate['maximum_input_covariance_asymmetry'],0.)
        for bad in (np.zeros((6,6)),np.diag([1,1,1,1,1,-1]),np.full((6,6),np.nan)):
            with self.assertRaises(ValueError):scalar_cholesky(bad)

    def test_legacy_and_all_false_are_explicitly_rejected(self):
        m=model(np.eye(6))
        with self.assertRaisesRegex(ValueError,'active reciprocal'):NormalizerProposalDensity(m['base_model'])
        m['reciprocal_components']=[False]
        with self.assertRaisesRegex(ValueError,'active reciprocal'):NormalizerProposalDensity(m)

    def test_source_binding_rejects_changed_algorithm_even_with_updated_file_hash(self):
        source=POP/'provenance/source-bundle.json'
        self.assertIn('source_bundle_sha256',bind_source_bundle(source))
        bundle=json.loads(source.read_text())
        entry=bundle['files']['src/proposal.rs'];entry['text']=entry['text'].replace('let mut lower = [[0.0; 6]; 6];','let mut lower = [[1.0; 6]; 6];',1)
        entry['sha256']=hashlib.sha256(entry['text'].encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bundle.json';p.write_text(json.dumps(bundle))
            with self.assertRaisesRegex(ValueError,'Unknown sampler'):bind_source_bundle(p)

    def test_six_saved_rows_reconstruct_recorded_density_without_tolerance_change(self):
        manifest=json.loads((POP/'manifest.json').read_text());cfg=json.loads((POP/'config.json').read_text())
        raw=(POP/'provenance/model.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),manifest['model_sha256'])
        m=json.loads(raw);wanted={6078,8167,4922,590,1686,6223}
        rows=[json.loads(line) for line in (POP/'samples.jsonl').read_text().splitlines()]
        rows=[r for r in rows if r['draw'] in wanted];self.assertEqual(len(rows),6)
        d=NormalizerProposalDensity(m,source_bundle=POP/'provenance/source-bundle.json')
        poses=[r['pose'] for r in rows]
        def full(density):
            logs=np.asarray([density.evaluate(relative_poses(poses,fixed))[0] for fixed in cfg['fixed_poses']])
            uniform=np.log(manifest['uniform_probability'])-3*np.log(2*cfg['capture_radius'])
            return logsumexp(np.logaddexp(uniform,np.log1p(-manifest['uniform_probability'])+logs),axis=0)-np.log(2)
        recorded=np.asarray([r['log_proposal_density'] for r in rows])
        self.assertGreater(np.max(abs(full(Density(m))-recorded)),2e-8)
        self.assertLess(np.max(abs(full(d)-recorded)),2e-8)
        self.assertLess(np.max(abs(full(d)-recorded)),1e-9)
        self.assertTrue(all(r['maximum_componentwise_bound_ratio']<=1 for r in d.factor_diagnostics))


if __name__=='__main__':unittest.main()
