"""Synthetic saved-row controls for defensive latent importance sampling.

No physical executable, hard-shape replay or old campaign is run here.
"""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from analyze_latent_region import (
    DENSITY_MEASURE, GUIDE_FILE, IMPORTANCE_SCHEMA, LatentImportanceGuide,
    analyze, latent_shell_contains, shell_log_volume,
)
from prepare_smc_normalizer_atlas import read, sha, write
from test_latent_region_qwindow_analysis import fixture as uniform_fixture, pose


def guide_json(region_sha,alpha=.5):
    lower=np.diag([.7,.9,.8,.6,.75,.85])
    lower[3,0]=.2;lower[5,1]=-.25
    covariance=(lower@lower.T).tolist()
    return dict(schema='defensive-latent-shell-guide-v1',region_sha256=region_sha,
        defensive_uniform_shell_probability=alpha,gaussian_components=[
            dict(weight=2.,mean=[2.2,0.,0.,0.,0.,0.],covariance=covariance),
            dict(weight=3.,mean=[-4.,0.,0.,0.,0.,0.],covariance=covariance)])


def reference_logq(guide,latents,inside,logv):
    """Independent SciPy covariance-density path, without the auditor's class."""
    alpha=guide['defensive_uniform_shell_probability']
    total=np.where(inside,math.log(alpha)-logv,-np.inf)
    if alpha<1.:
        mass=sum(c['weight'] for c in guide['gaussian_components'])
        components=[math.log(c['weight']/mass)+multivariate_normal.logpdf(latents,mean=c['mean'],cov=c['covariance'])
            for c in guide['gaussian_components']]
        total=np.logaddexp(total,math.log1p(-alpha)+logsumexp(components,axis=0))
    return total


def save_rows(directory,summary,rows):
    (directory/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary['samples_sha256']=sha(directory/'samples.jsonl')
    write(directory/'summary.json',summary)


def importance_fixture(root,alpha=.5):
    region,population,summary,_,directory=uniform_fixture(root)
    archive=root/'provenance'
    config=read(archive/'config.json');config['capture_radius']=10.
    region.update(capture_radius=10.,minimum_mahalanobis_radius=1.,minimum_original_q=0.,maximum_original_q=10.)
    write(archive/'config.json',config);write(archive/'region.json',region)
    region_sha=sha(archive/'region.json')
    guide=guide_json(region_sha,alpha)
    if alpha==1.:guide['gaussian_components']=[]
    write(archive/GUIDE_FILE,guide)
    for source,target in [('config.json','input-config.json'),('region.json','region.json'),(GUIDE_FILE,GUIDE_FILE)]:
        (directory/'provenance'/target).write_bytes((archive/source).read_bytes())
    points=[.5,1.,math.nextafter(1.,2.),1.5,2.,3.,math.nextafter(3.,4.),4.]
    if alpha==1.:points=[x for x in points if latent_shell_contains(x,region)]
    latents=np.asarray([[x,0.,0.,0.,0.,0.] for x in points])
    flags=np.asarray([latent_shell_contains(x,region) for x in points])
    logv=shell_log_volume(region);logq=reference_logq(guide,latents,flags,logv)
    rows=[];logs=[]
    for i,(x,u,inside,q) in enumerate(zip(points,latents,flags,logq)):
        hard=x!=1.5;valid=bool(inside and hard)
        logj=-2*math.log(math.pi);weight=logj-q
        uniform=alpha==1. or (inside and i%2==0)
        row=dict(draw=i,latent=u.tolist(),latent_radius=x,pose=pose(x),q=x,
            backmapped_latent=u.tolist(),backmapped_radius=x,log_physical_jacobian=logj,
            capture_valid=True,hard_valid=hard,region_valid=True,shell_valid=bool(inside),
            log_proposal_density=float(q),proposal_branch='uniform-shell' if uniform else 'gaussian',
            proposal_component=None if uniform else i%2,
            log_importance_weight=float(weight) if valid else None,log_hard_weight=float(weight) if valid else None,
            clouds=[dict(log_weight=0.,lower_volume=0.,overlap_points=0) for _ in range(2)] if valid else [])
        rows.append(row);logs.append(weight if valid else -np.inf)
    population.update(schema=IMPORTANCE_SCHEMA,samples=len(rows),config_sha256=sha(archive/'config.json'),
        region_sha256=region_sha,importance_guide_sha256=sha(archive/GUIDE_FILE),
        importance_uniform_probability=alpha,importance_component_count=len(guide['gaussian_components']),
        proposal_density_measure=DENSITY_MEASURE,minimum_latent_radius=1.,minimum_original_q=0.,maximum_original_q=10.,
        log_latent_ball_volume=logv,log_latent_shell_volume=logv)
    value=float(logsumexp(logs)-math.log(len(rows)))
    summary.update(manifest=population,samples=len(rows),shell_rejected=int((~flags).sum()),
        estimates=dict(region=dict(logQ=value),hard_region=dict(logQ=value)))
    write(directory/'manifest.json',population);save_rows(directory,summary,rows)
    campaign=read(root/'manifest.json')
    campaign.update(schema='importance-latent-region-campaign-v1',region_sha256=region_sha,
        importance_guide_sha256=sha(archive/GUIDE_FILE),
        archive_sha256={p.name:sha(p) for p in archive.iterdir()})
    campaign['jobs'][0]['samples']=len(rows);write(root/'manifest.json',campaign)
    return region,guide,population,summary,rows,directory


def quiet_analyze(root):
    with contextlib.redirect_stdout(io.StringIO()):return analyze(root)


class DefensiveLatentDensityTests(unittest.TestCase):
    def test_full_density_matches_independent_covariance_formula_inside_and_outside(self):
        region=dict(mahalanobis_radius=3.,minimum_mahalanobis_radius=1.)
        guide=guide_json('0'*64);density=LatentImportanceGuide(guide,'0'*64)
        u=np.array([[0.,0.,0.,0.,0.,0.],[2.,.1,-.2,.2,-.3,.1],[5.,1.,-2.,.4,.3,-.2]])
        flags=[latent_shell_contains(np.linalg.norm(v),region) for v in u]
        expected=reference_logq(guide,u,flags,shell_log_volume(region))
        np.testing.assert_allclose(density.log_density(u,flags,shell_log_volume(region)),expected,rtol=0,atol=2e-12)
        self.assertTrue(np.isfinite(expected).all())
        self.assertGreater(expected[1],math.log(.5)-shell_log_volume(region))

    def test_uniform_recovery_and_exact_shell_boundaries(self):
        region=dict(mahalanobis_radius=3.,minimum_mahalanobis_radius=1.)
        radii=[math.nextafter(1.,0.),1.,math.nextafter(1.,2.),3.,math.nextafter(3.,4.)]
        self.assertEqual([latent_shell_contains(x,region) for x in radii],[False,False,True,True,False])
        self.assertTrue(latent_shell_contains(0.,dict(mahalanobis_radius=3.)))
        guide=guide_json('0'*64,1.);guide['gaussian_components']=[]
        logv=shell_log_volume(region);density=LatentImportanceGuide(guide,'0'*64)
        values=density.log_density(np.array([[x,0.,0.,0.,0.,0.] for x in radii]),[False,False,True,True,False],logv)
        np.testing.assert_array_equal(values,np.array([-np.inf,-np.inf,-logv,-logv,-np.inf]))

    def test_constant_shell_integral_retains_frequent_exterior_gaussian_zeros(self):
        region=dict(mahalanobis_radius=2.,minimum_mahalanobis_radius=.5)
        guide=guide_json('0'*64)
        guide['gaussian_components']=[dict(weight=1.,mean=[6.,0.,0.,0.,0.,0.],covariance=(np.eye(6)*.8**2).tolist())]
        density=LatentImportanceGuide(guide,'0'*64);rng=np.random.default_rng(121001010);n=60_000
        uniform=rng.random(n)<.5
        u=rng.normal(size=(n,6))
        outer,inner=2.,.5
        radial=(inner**6+rng.random(n)*(outer**6-inner**6))**(1/6)
        u[uniform]*=(radial[uniform]/np.linalg.norm(u[uniform],axis=1))[:,None]
        u[~uniform]=rng.normal(size=((~uniform).sum(),6))*.8+np.array([6.,0.,0.,0.,0.,0.])
        flags=np.array([latent_shell_contains(r,region) for r in np.linalg.norm(u,axis=1)])
        self.assertGreater((~flags).sum(),.45*n)
        logv=shell_log_volume(region);q=density.log_density(u,flags,logv)
        weights=np.where(flags,np.exp(-q-logv),0.)
        # Integral of one over latent shell divided by its exact volume.
        estimate=float(weights.mean());se=float(weights.std(ddof=1)/math.sqrt(n))
        self.assertLess(abs(estimate-1.),5*se)
        # A valid-only denominator produces approximately twice the integral.
        self.assertGreater(float(weights[flags].mean()),1.9)
        np.testing.assert_allclose(q,reference_logq(guide,u,flags,logv),rtol=0,atol=1e-10)

    def test_strict_guide_validation_shape_spd_mass_and_region(self):
        for mutation in ('region','schema','extra','alpha0','alpha2','alpha_bool','empty','mean_shape','cov_shape','asymmetric','indefinite','weight0','nan'):
            guide=guide_json('0'*64)
            if mutation=='region':guide['region_sha256']='1'*64
            elif mutation=='schema':guide['schema']='unknown'
            elif mutation=='extra':guide['implemented']=False
            elif mutation=='alpha0':guide['defensive_uniform_shell_probability']=0.
            elif mutation=='alpha2':guide['defensive_uniform_shell_probability']=2.
            elif mutation=='alpha_bool':guide['defensive_uniform_shell_probability']=True
            elif mutation=='empty':guide['gaussian_components']=[]
            elif mutation=='mean_shape':guide['gaussian_components'][0]['mean']=[0.]*5
            elif mutation=='cov_shape':guide['gaussian_components'][0]['covariance']=[[1.]]
            elif mutation=='asymmetric':guide['gaussian_components'][0]['covariance'][0][1]=.2
            elif mutation=='indefinite':guide['gaussian_components'][0]['covariance'][0][0]=-1.
            elif mutation=='weight0':guide['gaussian_components'][0]['weight']=0.
            else:guide['gaussian_components'][0]['mean'][0]=math.nan
            with self.subTest(mutation=mutation),self.assertRaises((AssertionError,ValueError,np.linalg.LinAlgError)):
                LatentImportanceGuide(guide,'0'*64)


class DefensiveLatentSavedRowsTests(unittest.TestCase):
    def test_outside_rows_are_zero_even_when_hard_capture_and_q_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);region,guide,_,_,rows,_=importance_fixture(root)
            result=quiet_analyze(root)
            self.assertEqual(result['estimate']['draws'],8)
            self.assertEqual(result['importance_sampling']['shell_rejected'],4)
            self.assertEqual(result['estimate']['nonzero'],3)
            self.assertEqual(result['independently_reconstructed_poses'],8)
            self.assertEqual(result['estimate']['logQ'],result['hard_region']['logQ'])
            self.assertEqual(result['log_regional_depletion_enhancement'],0.)
            self.assertTrue(rows[-1]['hard_valid'] and rows[-1]['capture_valid'] and rows[-1]['region_valid'])
            self.assertFalse(rows[-1]['shell_valid']);self.assertIsNone(rows[-1]['log_importance_weight'])
            expected=logsumexp([r['log_importance_weight'] for r in rows if r['log_importance_weight'] is not None])-math.log(len(rows))
            self.assertAlmostEqual(result['estimate']['logQ'],expected)

    def test_alpha_one_matches_uniform_auditor_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,manifest,summary,rows,directory=importance_fixture(root,1.)
            importance=quiet_analyze(root)
            manifest['schema']='uniform-latent-region-normalizer-v2'
            for key in ['importance_guide_sha256','importance_uniform_probability','importance_component_count','proposal_density_measure']:
                manifest.pop(key)
            summary['manifest']=manifest;write(directory/'manifest.json',manifest);write(directory/'summary.json',summary)
            campaign=read(root/'manifest.json');campaign['schema']='uniform-latent-region-campaign-v1'
            campaign.pop('importance_guide_sha256');campaign['archive_sha256'].pop(GUIDE_FILE);write(root/'manifest.json',campaign)
            uniform=quiet_analyze(root)
            self.assertEqual(importance['estimate'],uniform['estimate'])
            self.assertEqual(importance['hard_region'],uniform['hard_region'])
            self.assertNotIn('importance_sampling',uniform)

    def test_rejects_density_branch_shell_vector_and_denominator_tampering(self):
        for mutation in ('shell','selected_density','wrong_component','uniform_outside','nonzero_outside','latent_direction','missing_row','conditioned_mean'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);region,_,_,summary,rows,directory=importance_fixture(root)
                if mutation=='shell':rows[-1]['shell_valid']=True
                elif mutation=='selected_density':rows[4]['log_proposal_density']=math.log(.5)-shell_log_volume(region)
                elif mutation=='wrong_component':rows[-1]['proposal_component']=2
                elif mutation=='uniform_outside':rows[-1].update(proposal_branch='uniform-shell',proposal_component=None)
                elif mutation=='nonzero_outside':rows[-1]['log_hard_weight']=0.
                elif mutation=='latent_direction':rows[4]['latent']=[0.,2.,0.,0.,0.,0.]
                elif mutation=='missing_row':rows.pop()
                else:summary['estimates']['region']['logQ']+=math.log(8/3)
                save_rows(directory,summary,rows)
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(AssertionError):analyze(root)

    def test_rejects_missing_changed_or_unbound_saved_guide(self):
        for mutation in ('population_missing','population_bytes','population_hash','alpha','count','measure','campaign_hash_missing','campaign_schema','guide_region'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);_,guide,manifest,summary,_,directory=importance_fixture(root)
                campaign=read(root/'manifest.json')
                if mutation=='population_missing':(directory/'provenance'/GUIDE_FILE).unlink()
                elif mutation=='population_bytes':(directory/'provenance'/GUIDE_FILE).write_text('{}')
                elif mutation=='population_hash':manifest['importance_guide_sha256']='0'*64
                elif mutation=='alpha':manifest['importance_uniform_probability']=.3
                elif mutation=='count':manifest['importance_component_count']=99
                elif mutation=='measure':manifest['proposal_density_measure']='physical Haar density'
                elif mutation=='campaign_hash_missing':campaign.pop('importance_guide_sha256')
                elif mutation=='campaign_schema':campaign['schema']='uniform-latent-region-campaign-v1'
                else:
                    guide['region_sha256']='0'*64;write(root/'provenance'/GUIDE_FILE,guide)
                    digest=sha(root/'provenance'/GUIDE_FILE)
                    campaign['archive_sha256'][GUIDE_FILE]=digest;campaign['importance_guide_sha256']=digest
                summary['manifest']=manifest;write(directory/'manifest.json',manifest);write(directory/'summary.json',summary)
                write(root/'manifest.json',campaign)
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaises((AssertionError,KeyError,FileNotFoundError)):analyze(root)


if __name__=='__main__':unittest.main()
