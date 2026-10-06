"""Synthetic bridge density, provenance, counters and allocation controls only."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal

import audit_context_bridge_guide as m
from prepare_context_bridge_guides import bridge_specs
from source_guide_reference import pose
from test_context_multicage_guide_audit import chart
from test_context_source_guide_audit import row as legacy_row


def source_tokens():
    return [[77,217,f'p{i}','q'] for i in range(16)]


def metadata(arm='bridge',population=2,component='bridge_a3_b4'):
    guides={name:dict(source_chart=chart(),asset_sha256=f'{i+1:064x}') for i,name in enumerate(m.COMPONENTS)}
    index=[c for c,_ in m.STRATA[arm]].index(component);n=dict(m.STRATA[arm])[component]
    runner=18*population+(0 if arm=='baseline' else 5)+index
    identity=dict(kind='bridge-source-geometry',comparison_arm=arm,population_index=population,
        component=component,component_index=index,guide_sha256=guides[component]['asset_sha256'],
        campaign_sha256='a'*64,moving_label=77,anchor_label=16)
    cfg=dict(population=runner,seed=501,draws=n,identity=identity,source_chart=guides[component]['source_chart'],
        limits=dict(cpu_seconds=200.,wall_seconds=400.,patch_node_visits_per_candidate=20000000,
            patch_leaf_tests_per_candidate=2000000,patch_node_visits_total=20000000*(n+1),patch_leaf_tests_total=2000000*(n+1)))
    entry=dict(id=f'{arm}-pop{population}-{component}',comparison_arm=arm,population_index=population,
        component=component,component_index=index,population=runner,draws=n,panel_ordinals=list(range(0,n,32)),
        seed=501,config_sha256='b'*64)
    return entry,cfg,guides


def record(weight=1.,count=16,exact_a=True,valid=True,component='full',branch='source',radius=0.,wall=None,core=None):
    region=('A_patch_complete' if count==16 else ('A_patch_0_0.25','A_patch_0.25_0.5','A_patch_0.5_0.75','A_patch_0.75_1')[count//4]) if exact_a else 'other_contact'
    return dict(region=region if valid else 'hard_invalid',physical_valid=valid,
        wall_valid=valid if wall is None else wall,core_valid=(True if valid else None) if core is None else core,
        log_hard_contribution=math.log(weight) if valid else None,is_A_T=valid and exact_a and count==16,
        source_T_complete_all_regions=valid and count==16,exact_A=valid and exact_a,
        exact_source_intersection_count=count if valid else None,component=component,branch=branch,
        mahalanobis_squared={name:radius for name in m.COMPONENTS})


def contract_fixture(root):
    bindings={}
    def save(name,value):
        path=root/name;path.write_text(json.dumps(value,allow_nan=False));bindings[str(path)]=m.sha(path)
        return dict(path=str(path),sha256=m.sha(path))
    anchor=pose([1.,2.,-.5],Rotation.from_rotvec([.2,-.3,.4]).as_matrix())
    original=pose([-.2,.3,.4],Rotation.from_rotvec([.1,.2,-.1]).as_matrix())
    source=save('source.json',dict(pose=original,anchor_pose=anchor))
    patch=save('patch.json',{})
    regions=dict(a_neighbors=[16,217],b_neighbors=[16,56],secondary_label=217,
        source_secondary_tokens=source_tokens(),inclusion_boundaries=[0.,.25,.5,.75,1.])
    reference=dict(definitions=regions,patch_map=patch);guides={};objects={};cages=[]
    for name in m.OLD_COMPONENTS:
        c=chart()
        if name=='diagonal':c['covariance']=np.diag(np.diag(c['covariance'])).tolist()
        if name=='broad_full':c['covariance']=(4*np.asarray(c['covariance'])).tolist()
        obj=dict(source_chart=c)
        if name.startswith('cage'):
            i=int(name[-1]);center=dict(schema='source-chart-center-v1',frame='saved-spherical-center',
                pose=pose([1.+i,-.2,.4],Rotation.from_rotvec([.1,.2,.3+i]).as_matrix()),provenance='fixed synthetic cage')
            c['chart_center']=center;c['explicit_gaussian']['mean'][0]+=.1*(i+1)
            obj.update(schema='competing-cage-frozen-guide-v1',cage_id=i,training_samples=20480,
                heldout_fit_samples=0,fit_rule={'frozen':True},patch_reference=reference,original_source_state=source)
        asset=save(name+'.json',obj);guides[name]=asset;objects[name]=obj
        if name.startswith('cage'):
            cages.append(dict(cage_id=i,guide=asset,chart_center=center,initial_pose=center['pose'],
                fit=dict(covariance=c['covariance'],mean=c['explicit_gaussian']['mean'])))
    fit=save('fit.json',dict(schema='competing-cage-covariance-v1',complete=True,passed=True,
        total_retained_states=92160,production_states=81920,declared_training_states=40960,fitted_training_states=40960,
        heldout_fit_samples=0,fit_rule={'frozen':True},patch_reference=reference,angular_length=2.,cages=cages))
    old=save('old.json',dict(schema='context-multicage-source-mixture-v1',complete=True,passed=True,
        guides=copy.deepcopy(guides),cage_fit_report=fit))
    bridges,construction=bridge_specs(objects['full']['source_chart'],objects['cage0']['source_chart'],original,anchor,'synthetic fixed recipe')
    for name,value in bridges.items():guides[name]=save(name+'.json',value)
    bundle=save('bundle.json',{});inputs=dict(regions=regions,patch_map=patch)
    mixture=save('mixture.json',dict(schema='context-bridge-source-mixture-v1',complete=True,passed=True,
        total_future_attempts=16384,future_stratum_jobs=72,runner_mixture=[.5,.25,.25],guides=guides,
        source_inputs=inputs,source_bundle=bundle,effective_density_coefficients=m.COEFFICIENTS,
        density_component_order=['uniform','context',*m.COMPONENTS],old_mixture_manifest=old,
        source_frame_assets=dict(source_state=source),construction=construction,alphas=[0.,1/3,2/3,1.],
        standard_deviation_widths=[1.,4.],along_parameter_sigma=1/6,diagnostics=m.DIAGNOSTICS,
        physical_conditions=dict(depletant_radius=1.5,activity=.035,**{'lambda':2.24})))
    allocation=save('allocation.json',dict(schema='context-bridge-guide-geometry-allocation-v1',comparison_arms=list(m.ARMS),
        populations_per_arm=4,draws_per_population=2048,denominator=2048,populations=8,stratum_jobs=72,
        total_draws=16384,clouds=0,mode='geometry',mixture=[.5,.25,.25],panel_stride=32,panel_queries=512,
        squared_mahalanobis_bin_edges=[0,6,12,24,48,96,None],mixture_manifest=mixture,guides=guides,inputs=inputs,
        source_bundle=bundle,effective_density_coefficients=m.COEFFICIENTS,density_component_order=['uniform','context',*m.COMPONENTS],
        diagnostics=m.DIAGNOSTICS,cpu_seconds_per_job=200,wall_seconds_per_job=400,memory_bytes_per_job=4*1024**3,
        strata={arm:[dict(component=c,draws=n,deterministic_fraction=n/2048) for c,n in m.STRATA[arm]] for arm in m.ARMS}))
    campaign=save('campaign.json',dict(schema='context-bridge-guide-geometry-campaign-v1',allocation=allocation,mixture_manifest=mixture))
    inventory=[dict(id=str(i),seed=i) for i in range(72)];inv=save('inventory.json',inventory)
    protocol=dict(schema='context-bridge-guide-audit-v1',expected_strata=72,total_draws=16384,panel_stride=32,
        expected_panel_size=512,mahalanobis_squared_bins=[0,6,12,24,48,96,None],input_sha256=bindings,allocation=allocation,
        mixture_manifest=mixture,campaign=campaign,inventory=inv,strata=inventory,guides=guides,source_bundle=bundle['path'])
    return protocol,save


class BridgeGuideAuditTests(unittest.TestCase):
    def test_complete_contract_independent_bridge_reconstruction_and_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol,_=contract_fixture(Path(directory))
            allocation,guides,mixture=m.validate_contract(protocol,protocol['allocation']['sha256'])
            self.assertEqual(allocation['denominator'],2048);self.assertEqual(len(guides),13)
            for key,value in [('expected_panel_size',511),('panel_stride',64),('expected_strata',71)]:
                bad=copy.deepcopy(protocol);bad[key]=value
                with self.assertRaises(ValueError):m.validate_contract(bad,protocol['allocation']['sha256'])
            with self.assertRaises(ValueError):m.validate_contract(protocol,'0'*64)
            path=Path(protocol['guides']['bridge_a1_b4']['path']);changed=json.loads(path.read_text())
            changed['source_chart']['covariance'][0][0]+=.01;path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):m.validate_contract(protocol,protocol['allocation']['sha256'])

    def test_bridge_recipe_mutations_rejected_even_with_updated_guide_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol,_=contract_fixture(Path(directory));_,guides,mixture=m.validate_contract(protocol,protocol['allocation']['sha256'])
            asset=lambda a:json.loads(Path(a['path']).read_text())
            for mutation in ('mean','width','center','covariance','alpha'):
                bad=copy.deepcopy(guides);g=bad['bridge_a2_b4'];spec=g['source_chart']
                if mutation=='mean':spec['explicit_gaussian']['mean'][0]+=.1
                elif mutation=='width':g['standard_deviation_width']=1.
                elif mutation=='center':spec['chart_center']['pose']['position'][0]+=.1
                elif mutation=='covariance':spec['covariance'][0][0]*=1.1
                else:g['alpha']=.5
                with self.assertRaises(ValueError):m.validate_guide_set(bad,mixture,asset)

    def test_all72_identities_panel_and_resource_caps(self):
        entry,cfg,guides=metadata();seen=set()
        self.assertEqual(m.validate_admitted_config(entry,cfg,'b'*64,guides,seen,'a'*64),('bridge',2,'bridge_a3_b4'))
        with self.assertRaises(ValueError):m.validate_admitted_config(entry,cfg,'b'*64,guides,seen,'a'*64)
        for key in ('population','draws','seed'):
            bad=copy.deepcopy(cfg);bad[key]+=1
            with self.assertRaises(ValueError):m.validate_admitted_config(entry,bad,'b'*64,guides,set(),'a'*64)
        bad=copy.deepcopy(cfg);bad['limits']['patch_node_visits_total']-=1
        with self.assertRaises(ValueError):m.validate_admitted_config(entry,bad,'b'*64,guides,set(),'a'*64)
        self.assertEqual(len(m.expected_identities()),72)
        self.assertEqual(sum(n*4 for arm in m.ARMS for _,n in m.STRATA[arm]),16384)
        self.assertEqual(sum(len(range(0,n,32))*4 for arm in m.ARMS for _,n in m.STRATA[arm]),512)

    def test_complete15_density_terms_and_pointwise_defensive_bound(self):
        densities=np.arange(1,31,dtype=float).reshape(15,2)/7
        own=[.5*densities[0]+.25*densities[1]+.25*densities[i+2] for i in range(13)]
        q={arm:np.exp(m.arm_log_q(np.log(densities),m.COEFFICIENTS[arm])) for arm in m.ARMS}
        expected=sum(n/2048*own[m.COMPONENTS.index(c)] for c,n in m.STRATA['baseline'])
        np.testing.assert_allclose(q['baseline'],expected,rtol=2e-15)
        np.testing.assert_allclose(q['bridge'],.5*q['baseline']+.5*np.mean(own[5:],axis=0),rtol=2e-15)
        self.assertTrue(np.all(q['bridge']>=.5*q['baseline']))
        with self.assertRaises(ValueError):m.arm_log_q(np.log(densities[:-1]),m.COEFFICIENTS['baseline'])

    def test_deterministic_strata_expectation_in_disjoint_discrete_limit(self):
        laws=np.array([[.7,.3,0.],[0.,.5,.5],[.1,.2,.7]])
        weights=np.array([.25,.25,.5]);q=weights@laws;target=np.array([2.,0.,5.])
        self.assertAlmostEqual(sum(w*np.sum(p*target/q) for w,p in zip(weights,laws)),target.sum())
        self.assertNotAlmostEqual(sum(np.sum(p*target/q) for p in laws),target.sum())

    def test_exact_token_counts_all_neighbors_and_exact_A_are_separate(self):
        rows=[record(count=k,exact_a=k%2==0) for k in range(17)]+[record(valid=False)]
        hist=m.token_histogram(rows,2048)
        self.assertEqual([r['hits'] for r in hist['all_neighbors']],[1]*17)
        self.assertEqual([r['hits'] for r in hist['exact_A']],[int(k%2==0) for k in range(17)])
        self.assertEqual(hist['hard_invalid_count'],1)
        self.assertTrue(all(r['denominator']==2048 for r in hist['all_neighbors']))

    def test_overlay_reconstructs_tokens_and_preserves_invalid_zeros(self):
        entry,_,_=metadata();row,_=legacy_row(valid=False);row['input']['identity']={};row['input']['branch']='source'
        assets=dict(path='/synthetic/rows.jsonl',sha256='c'*64);logs={c:-2. for c in m.COMPONENTS};radii={c:3. for c in m.COMPONENTS}
        got=m.make_contribution(entry,row,assets,1.,logs,radii,source_tokens())
        self.assertIsNone(got['exact_source_intersection_count']);self.assertIsNone(got['log_hard_contribution'])
        row['actual'].update(physical_valid=True,wall_valid=True,core_valid=True);row['region']='other_contact'
        row['patches']=dict(source_fraction=1.,neighbor_labels=[16,56,217],tokens=source_tokens())
        got=m.make_contribution(entry,row,assets,1.,logs,radii,source_tokens())
        self.assertEqual(got['exact_source_intersection_count'],16);self.assertFalse(got['exact_A'])
        self.assertTrue(got['source_T_complete_all_regions']);self.assertFalse(got['is_A_T'])
        row['patches']['source_fraction']=.5
        with self.assertRaises(ValueError):m.make_contribution(entry,row,assets,1.,logs,radii,source_tokens())

    def test_wall_core_failure_partition_never_infers_an_untested_clash(self):
        rows=[record(),record(valid=False),record(valid=False,wall=True,core=False),
              record(valid=False,wall=False,core=False)]
        hist=m.token_histogram(rows,2048)
        self.assertEqual(hist['hard_valid_count'],1);self.assertEqual(hist['hard_invalid_count'],3)
        self.assertEqual(hist['wall_invalid_count'],2)
        self.assertEqual(hist['core_invalid_with_valid_wall_count'],1)
        self.assertEqual(hist['core_not_evaluated_count'],1)
        bad=record(valid=False,wall=True)
        with self.assertRaises(ValueError):m.token_histogram([bad],2048)

    def test_invalid_denominator_and_per_component_branch_diagnostics(self):
        rows=[record(2.),record(3.,exact_a=False),record(valid=False),record(count=8,component='bridge_a2_b4')]
        summary=m.summarize_records(rows,4)
        self.assertAlmostEqual(summary['A_T']['mass'],.5)
        self.assertEqual(sum(r['attempted_count'] for r in summary['regions']),4)
        bridge=next(x for x in summary['component_diagnostics'] if x['component']=='bridge_a2_b4')
        self.assertEqual((bridge['alpha'],bridge['standard_deviation_width']),(2/3,4))
        self.assertEqual(bridge['branches']['source']['attempted_count'],1)
        self.assertEqual(bridge['branches']['source']['all_neighbors'][8]['hits'],1)
        self.assertEqual(bridge['branches']['source']['denominator'],4)
        with self.assertRaises(ValueError):m.summarize_records(rows,3)

    def test_predeclared_coverage_threshold_is_geometry_only_and_requires_three_populations(self):
        def population(n):return dict(source_token_counts=m.token_histogram([record(count=8)]*n,2048))
        gate=m.coverage_gate([population(n) for n in (7,7,6,0)])
        self.assertTrue(gate['all_neighbors']['passes_declared_geometry_threshold'])
        self.assertFalse(m.coverage_gate([population(n) for n in (10,10,0,0)])['all_neighbors']['passes_declared_geometry_threshold'])
        self.assertFalse(m.coverage_gate([population(n) for n in (6,6,6,0)])['all_neighbors']['passes_declared_geometry_threshold'])

    def test_full_chart_tail_bins_close_without_summing_other_chart_partitions(self):
        rows=[record(radius=v) for v in (0.,6.,12.,24.,48.,96.)]
        families=m.radial_summary(rows,2048)
        self.assertEqual(len(families),1);self.assertEqual(families[0]['chart'],'full')
        self.assertEqual(sum(r['hits'] for r in families[0]['entries']),6)
        self.assertAlmostEqual(sum(r['fraction_of_A_T_weight'] for r in families[0]['entries']),1.)
        self.assertEqual(m.radial_bin(None),5)

    def test_all13_source_densities_use_one_normalized_Haar_jacobian(self):
        anchor=pose([1.,2.,-.5],Rotation.from_rotvec([.2,-.3,.4]).as_matrix());center=pose([-1.,.3,2.],np.eye(3))
        base=chart();densities=[]
        for i,name in enumerate(m.COMPONENTS):
            spec=copy.deepcopy(base);spec['chart_center']=dict(schema='source-chart-center-v1',frame='saved-spherical-center',
                pose=pose([.2*i,-.1,.3],Rotation.from_rotvec([.01*i,.2,-.1]).as_matrix()),provenance='synthetic frozen center')
            densities.append(m.SourceDensity(spec,center,anchor))
        latent=np.array([.3,-.7,.2,1.1,-.4,.8]);sample=densities[0].decode(latent)
        for density in densities:
            logs,_,seam=m.source_batch(density,[sample]);self.assertFalse(seam[0])
            self.assertAlmostEqual(logs[0],density.evaluate(sample),places=10)
        x=densities[0].mean+densities[0].lower@latent;u=x[3:]/base['angular_length']
        logJ=-3*math.log(base['angular_length'])-2*math.log(math.pi)-2*math.log1p(u@u)
        normal=multivariate_normal.logpdf(x,mean=base['explicit_gaussian']['mean'],cov=base['covariance'])
        self.assertAlmostEqual(densities[0].evaluate(sample),normal-logJ,places=11)
        with self.assertRaises(ValueError):m.close_density_arrays([normal-2*logJ],[densities[0].evaluate(sample)])

    def test_cholesky_roundoff_near_zero_does_not_hide_meaningful_map_changes(self):
        expected=np.eye(6);expected[5,2]=0.002006524159211579
        rust=expected.copy();rust[5,2]=0.0020065241592116274
        self.assertFalse(np.allclose(rust,expected,atol=0,rtol=2e-14))
        self.assertTrue(m.map_lower_matches(rust,expected,expected@expected.T))
        self.assertFalse(m.map_lower_matches(rust,rust,2*(expected@expected.T)))
        rust[5,2]+=1e-9
        self.assertFalse(m.map_lower_matches(rust,expected,expected@expected.T))
        self.assertFalse(m.map_lower_matches(np.full((6,6),np.nan),expected,expected@expected.T))

    def test_explicit_seams_and_nonfinite_density_guard(self):
        self.assertEqual(m.close_density_arrays([-math.inf,0.],[-math.inf,0.]),0.)
        for actual,expected in [([math.nan],[0.]),([math.inf],[math.inf]),([-math.inf],[0.])]:
            with self.assertRaises(ValueError):m.close_density_arrays(actual,expected)
        values=np.full((15,1),-math.inf);values[0,0]=0.
        self.assertAlmostEqual(m.arm_log_q(values,m.COEFFICIENTS['bridge'])[0],math.log(.5))


if __name__=='__main__':unittest.main()
