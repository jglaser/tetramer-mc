"""Deterministic audit/reduction controls; no protein data, geometry or randomness."""
import copy
import math
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

from audit_context_multicage_guide import (ARMS, COMPONENTS, COEFFICIENTS, STRATA,
    SourceDensity, arm_log_q, close_density_arrays, expected_identities,
    importance_summary, make_contribution, radial_bin, radial_summary,
    source_batch, summarize_records, validate_admitted_config, validate_contract, sha, scalar_row, validate_center_echo, validate_guide_set)
from source_guide_reference import pose
from test_context_source_guide_audit import row as legacy_row


def chart():
    lower=np.diag([.2,.3,.4,.12,.15,.18]);lower[3,0]=.04;lower[5,1]=-.03
    return dict(angular_length=2.,covariance=(lower@lower.T).tolist(),
        explicit_gaussian=dict(schema='source-gaussian-v1',mean=[.03,-.04,.02,.01,-.02,.03],provenance='synthetic'))


def metadata(arm='multicage',p=2,component='cage1'):
    guides={name:dict(source_chart=chart(),asset_sha256=str(i+1)*64) for i,name in enumerate(COMPONENTS)}
    index=[c for c,_ in STRATA[arm]].index(component);n=dict(STRATA[arm])[component]
    runner=8*p+(0 if arm=='baseline' else 3)+index
    ident=dict(kind='multicage-source-geometry',comparison_arm=arm,population_index=p,
        component=component,component_index=index,guide_sha256=guides[component]['asset_sha256'],
        campaign_sha256='a'*64,moving_label=77,anchor_label=16)
    cfg=dict(population=runner,seed=501,draws=n,identity=ident,source_chart=guides[component]['source_chart'])
    entry=dict(id=f'{arm}-{p}-{component}',comparison_arm=arm,population_index=p,
        component=component,component_index=index,population=runner,draws=n,
        panel_ordinals=list(range(0,n,64)),seed=501,config_sha256='b'*64)
    return entry,cfg,guides


def record(weight,region='A_patch_complete',component='full',branch='source',r_full=0,r_diag=0):
    valid=region!='hard_invalid'
    return dict(region=region,physical_valid=valid,log_hard_contribution=math.log(weight) if valid else None,
        is_A_T=region=='A_patch_complete',source_T_complete_all_regions=region in ('A_patch_complete','other_contact'),
        component=component,branch=branch,mahalanobis_squared=dict(full=r_full,diagonal=r_diag,broad_full=r_full/4,cage0=r_full+1,cage1=r_diag+2))


class MulticageGuideAuditTests(unittest.TestCase):
    def test_complete_five_chart_contract_and_frozen_fit_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bindings={}
            def save(name,value):
                path=root/name;path.write_text(json.dumps(value,allow_nan=False));bindings[str(path)]=sha(path)
                return dict(path=str(path),sha256=sha(path))
            patch=save('patch.json',{});source=save('source.json',{})
            regions=dict(a_neighbors=[16,217],b_neighbors=[16,56]);reference=dict(definitions=regions,patch_map=patch)
            guides={};cages=[]
            for name in COMPONENTS:
                c=chart()
                if name=='diagonal':c['covariance']=np.diag(np.diag(c['covariance'])).tolist()
                if name=='broad_full':c['covariance']=(4*np.asarray(c['covariance'])).tolist()
                g=dict(source_chart=c)
                if name.startswith('cage'):
                    i=int(name[-1]);center=dict(schema='source-chart-center-v1',frame='saved-spherical-center',
                        pose=pose([1.+i,-.2,.4],Rotation.from_rotvec([.1,.2,.3+i]).as_matrix()),provenance='frozen synthetic seed')
                    c['chart_center']=center;c['explicit_gaussian']['mean'][0]+=.1*(i+1)
                    g.update(schema='competing-cage-frozen-guide-v1',cage_id=i,training_samples=20480,heldout_fit_samples=0,
                        fit_rule={'frozen':True},patch_reference=reference,original_source_state=source)
                    a=save(name+'.json',g)
                    cages.append(dict(cage_id=i,guide=a,chart_center=center,initial_pose=center['pose'],
                        fit=dict(covariance=c['covariance'],mean=c['explicit_gaussian']['mean'])))
                else:a=save(name+'.json',g)
                guides[name]=a
            fit=save('fit.json',dict(schema='competing-cage-covariance-v1',complete=True,passed=True,
                total_retained_states=92160,production_states=81920,declared_training_states=40960,
                fitted_training_states=40960,heldout_fit_samples=0,fit_rule={'frozen':True},patch_reference=reference,
                angular_length=2.,cages=cages))
            bundle=save('bundle.json',{});inputs=dict(regions=regions,patch_map=patch)
            mixture=save('mixture.json',dict(schema='context-multicage-source-mixture-v1',complete=True,passed=True,
                total_future_attempts=32768,future_stratum_jobs=32,runner_mixture=[.5,.25,.25],guides=guides,
                source_inputs=inputs,source_bundle=bundle,effective_density_coefficients=COEFFICIENTS,
                density_component_order=['uniform','context',*COMPONENTS],cage_fit_report=fit))
            allocation=save('allocation.json',dict(schema='context-multicage-guide-geometry-allocation-v1',
                comparison_arms=list(ARMS),populations_per_arm=4,draws_per_population=4096,denominator=4096,
                populations=8,stratum_jobs=32,total_draws=32768,clouds=0,mode='geometry',mixture=[.5,.25,.25],
                panel_stride=64,panel_queries=512,squared_mahalanobis_bin_edges=[0,6,12,24,48,96,None],
                mixture_manifest=mixture,guides=guides,inputs=inputs,source_bundle=bundle,
                effective_density_coefficients=COEFFICIENTS,density_component_order=['uniform','context',*COMPONENTS],
                strata={arm:[dict(component=c,draws=n,deterministic_fraction=n/4096) for c,n in STRATA[arm]] for arm in ARMS}))
            campaign=save('campaign.json',dict(schema='context-multicage-guide-geometry-campaign-v1',allocation=allocation,mixture_manifest=mixture))
            inventory=[dict(id=str(i),seed=i) for i in range(32)];inv=save('inventory.json',inventory)
            protocol=dict(schema='context-multicage-guide-audit-v1',expected_strata=32,total_draws=32768,
                panel_stride=64,expected_panel_size=512,mahalanobis_squared_bins=[0,6,12,24,48,96,None],
                input_sha256=bindings,allocation=allocation,mixture_manifest=mixture,campaign=campaign,
                inventory=inv,strata=inventory,guides=guides,source_bundle=bundle['path'])
            self.assertEqual(validate_contract(protocol,allocation['sha256'])[0]['denominator'],4096)
            bad=copy.deepcopy(protocol);bad['expected_panel_size']=511
            with self.assertRaises(ValueError):validate_contract(bad,allocation['sha256'])
            bad=copy.deepcopy(protocol);bad['strata']=inventory[:-1]
            with self.assertRaises(ValueError):validate_contract(bad,allocation['sha256'])
            with self.assertRaises(ValueError):validate_contract(protocol,'0'*64)
            cage=Path(guides['cage0']['path']);changed=json.loads(cage.read_text())
            changed['source_chart']['chart_center']['pose']['position'][0]+=.01;cage.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):validate_contract(protocol,allocation['sha256'])

    def test_explicit_center_echo_and_unchanged_original_reference(self):
        s=chart();validate_center_echo(s,{})
        original=pose([0.,0.,0.],np.eye(3));s['chart_center']=dict(schema='source-chart-center-v1',frame='saved-spherical-center',
            pose=pose([2.,3.,4.],Rotation.from_rotvec([.1,.3,-.2]).as_matrix()),provenance='fixed synthetic cage')
        before=copy.deepcopy(original);manifest=dict(chart_center=copy.deepcopy(s['chart_center']))
        validate_center_echo(s,manifest)
        density=SourceDensity(s,original,pose([1.,2.,3.],np.eye(3)));self.assertEqual(original,before)
        sample=density.decode(np.zeros(6));wrong=copy.deepcopy(s);del wrong['chart_center']
        self.assertGreater(abs(density.evaluate(sample)-SourceDensity(wrong,original,pose([1.,2.,3.],np.eye(3))).evaluate(sample)),1.)
        for bad in ({},dict(chart_center=None),dict(chart_center=dict(s['chart_center'],frame='display-origin'))):
            with self.assertRaises(ValueError):validate_center_echo(s,bad)

    def test_stratified_balance_expectation_and_disjoint_support(self):
        laws=np.array([[.75,.25,0],[0,.5,.5],[.2,0,.8],[.5,.4,.1],[.2,.6,.2]])
        h=np.array([1.,0.,2.])
        for fractions in (np.array([.375,.375,.25,0,0]),np.array([.1875,.1875,.125,.25,.25])):
            q=fractions@laws
            self.assertAlmostEqual(q.sum(),1.)
            self.assertAlmostEqual(sum(f*np.sum(law*h/q) for f,law in zip(fractions,laws)),h.sum())
            self.assertNotAlmostEqual(sum(np.sum(law*h/q) for f,law in zip(fractions,laws) if f),h.sum())

    def test_full_q_matches_every_stratum_and_defensive_baseline(self):
        densities=np.array([[.2,.1],[.3,.4],[.9,.7],[.8,.2],[.1,1.3],[1.2,.3],[.4,1.1]])
        own=[.5*densities[0]+.25*densities[1]+.25*densities[i+2] for i in range(5)]
        base=np.exp(arm_log_q(np.log(densities),COEFFICIENTS['baseline']))
        multi=np.exp(arm_log_q(np.log(densities),COEFFICIENTS['multicage']))
        np.testing.assert_allclose(base,.375*own[0]+.375*own[1]+.25*own[2],rtol=2e-15)
        np.testing.assert_allclose(multi,.5*base+.25*own[3]+.25*own[4],rtol=2e-15)
        self.assertTrue(np.all(multi>=.5*base))
        self.assertTrue(np.all(multi>=.25*own[4]))
        with self.assertRaises(ValueError):arm_log_q(np.log(densities),[.5,.25,.125,.125,.1,0,0])

    def test_hard_invalid_zero_and_all_attempt_denominator(self):
        rows=[record(2.),record(3.,'other_contact'),record(0.,'hard_invalid'),record(0.,'hard_invalid')]
        summary=summarize_records(rows,4)
        self.assertAlmostEqual(summary['A_T']['mass'],.5)
        self.assertAlmostEqual(summary['T_any']['mass'],1.25)
        self.assertEqual(sum(r['attempted_count'] for r in summary['regions']),4)
        invalid=next(r for r in summary['regions'] if r['region']=='hard_invalid')
        self.assertEqual((invalid['attempted_count'],invalid['hits'],invalid['mass']),(2,0,0.))
        with self.assertRaises(ValueError):summarize_records(rows,2)
        self.assertEqual(importance_summary([],4)['importance_ess'],0.)

    def test_actual_identity_global_index_and_exact_guide_are_bound(self):
        entry,cfg,guides=metadata();seen=set()
        self.assertEqual(validate_admitted_config(entry,cfg,'b'*64,guides,seen,'a'*64),('multicage',2,'cage1'))
        with self.assertRaises(ValueError):validate_admitted_config(entry,cfg,'b'*64,guides,seen,'a'*64)
        for key in ('population','draws','seed'):
            bad=copy.deepcopy(cfg);bad[key]+=1
            with self.assertRaises(ValueError):validate_admitted_config(entry,bad,'b'*64,guides,set(),'a'*64)
        bad=copy.deepcopy(cfg);bad['source_chart']['explicit_gaussian']['mean'][0]+=.01
        with self.assertRaises(ValueError):validate_admitted_config(entry,bad,'b'*64,guides,set(),'a'*64)
        self.assertEqual(len(expected_identities()),32)
        self.assertEqual(sum(len(range(0,n,64))*4 for arm in ARMS for _,n in STRATA[arm]),512)

    def test_normalized_haar_jacobian_and_all_five_source_densities(self):
        anchor=pose([1.,2.,-.5],Rotation.from_rotvec([.2,-.3,.4]).as_matrix())
        center=pose([-1.,.3,2.],Rotation.from_rotvec([-.3,.1,.5]).as_matrix())
        original=chart();specs=[]
        for name in COMPONENTS:
            spec=copy.deepcopy(original)
            if name=='diagonal':spec['covariance']=np.diag(np.diag(spec['covariance'])).tolist()
            if name=='broad_full':spec['covariance']=(4*np.asarray(spec['covariance'])).tolist()
            if name.startswith('cage'):
                index=int(name[-1]);spec['chart_center']=dict(schema='source-chart-center-v1',frame='saved-spherical-center',
                    pose=pose([.3+index,-.2,1.],Rotation.from_rotvec([.1,-.2,.4+index]).as_matrix()),provenance='fixed synthetic cage')
                spec['explicit_gaussian']['mean'][0]+=.2*(index+1)
            specs.append(spec)
        densities=[SourceDensity(s,center,anchor) for s in specs]
        z=np.array([.3,-.7,.2,1.1,-.4,.8]);value=densities[0].decode(z)
        for density in densities:
            log,r,seam=source_batch(density,[value])
            self.assertAlmostEqual(log[0],density.evaluate(value),places=11)
            self.assertFalse(seam[0]);self.assertGreaterEqual(r[0],0.)
        x=np.asarray(original['explicit_gaussian']['mean'])+densities[0].lower@z
        u=x[3:]/original['angular_length']
        logJ=-3*math.log(original['angular_length'])-2*math.log(math.pi)-2*math.log1p(u@u)
        normal=multivariate_normal.logpdf(x,mean=original['explicit_gaussian']['mean'],cov=original['covariance'])
        self.assertAlmostEqual(densities[0].evaluate(value),normal-logJ,places=11)
        with self.assertRaises(ValueError):close_density_arrays([normal],[densities[0].evaluate(value)])
        with self.assertRaises(ValueError):close_density_arrays([normal-2*logJ],[densities[0].evaluate(value)])

    def test_full_q_guard_rejects_second_jacobian_on_saved_row(self):
        row,source=legacy_row(valid=False)
        scalar_row(row,0,2.,[],source,[.5,.25,.25])
        bad=copy.deepcopy(row);bad['density']['log_q']+=.2
        with self.assertRaises(ValueError):scalar_row(bad,0,2.,[],source,[.5,.25,.25])

    def test_radial_boundaries_and_separate_partition_closure(self):
        self.assertEqual([radial_bin(v) for v in [0,5.9,6,12,24,48,96,None]],[0,0,1,2,3,4,5,5])
        rows=[record(2.,r_full=6,r_diag=96),record(3.,component='diagonal',r_full=24,r_diag=0),
              record(1.,component='cage1',r_full=48,r_diag=12)]
        for family in radial_summary(rows,8):
            self.assertEqual(sum(r['hits'] for r in family['entries']),3)
            self.assertAlmostEqual(sum(r['fraction_of_A_T_weight'] for r in family['entries']),1.)
        with self.assertRaises(ValueError):radial_bin(-1.)

    def test_density_seam_and_nonfinite_guard(self):
        self.assertEqual(close_density_arrays([-math.inf,0.],[-math.inf,0.]),0.)
        for actual,expected in [([float('nan')],[0.]),([math.inf],[math.inf]),([-math.inf],[0.])]:
            with self.assertRaises(ValueError):close_density_arrays(actual,expected)
        logs=np.full((7,1),-math.inf);logs[0,0]=0.
        self.assertAlmostEqual(arm_log_q(logs,COEFFICIENTS['baseline'])[0],math.log(.5))

    def test_overlay_snapshot_preserves_invalid_and_T_outside_A(self):
        entry,_,_=metadata();entry['id']='test'
        row,_=legacy_row(valid=False)
        row['input']['identity']={'synthetic':True};row['input']['branch']='uniform'
        source_rows=dict(path='/synthetic/rows.jsonl',sha256='c'*64)
        got=make_contribution(entry,row,source_rows,1.,{c:-2. for c in COMPONENTS},{c:3. for c in COMPONENTS})
        self.assertIsNone(got['log_hard_contribution']);self.assertFalse(got['source_T_complete_all_regions'])
        self.assertEqual(got['proposed_pose'],row['input']['proposed_pose']);self.assertEqual(got['source_rows'],source_rows)
        row['actual']['physical_valid']=True;row['region']='other_contact'
        row['patches']=dict(source_fraction=1.,neighbor_labels=[16,56,217],tokens=[])
        got=make_contribution(entry,row,source_rows,1.,{c:-2. for c in COMPONENTS},{c:3. for c in COMPONENTS})
        self.assertTrue(got['source_T_complete_all_regions']);self.assertFalse(got['is_A_T'])
        self.assertEqual(got['log_hard_contribution'],-1.)


if __name__=='__main__':unittest.main()
