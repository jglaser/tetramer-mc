"""Nonphysical synthetic-record checks of the new SMC audit and statistics."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.special import logsumexp
from analyze_r4_smc_control import (
    CLASSES,Chart,Proposal,Ledger,Observer,audit_population,audit_stage,aggregate,compare_analyses,
    population_statistics,parse_job_options,systematic_parents,validate_config_identity,
    validate_control,write,sha)


def pose(x):return dict(position=[float(x),0.,0.],orientation=[1.,0.,0.,0.])
def specs(z=0.):
    fixed=pose(0);metric=dict(native_poses=[pose(1.)],rigid_members=[fixed],member_error_scale=1.,angle_error_scale_deg=15.)
    config=dict(shape='shape.json',fixed_poses=[fixed],capture_center=[0.,0.,0.],capture_radius=170.,
        reservoir_density=z,depletant_radius=1.5,metadata=metric,translation_steps=[.2,2.],rotation_steps_deg=[1.5,15.],
        rotation_probability=.5,poisson_lambda_ratio=64.,endpoint_gate={})
    current=dict(fixed_neighbor=fixed,physical_fixed_neighbors=[fixed],capture_center=config['capture_center'],capture_radius=170.,activity=z,
        depletant_radius=1.5,physical_metric=metric,shape_sha256='shape',mahalanobis_radius=4.,minimum_original_q=0.,
        gaussian_chart=dict(angular_length=1.,means=[[0.]*6],covariances=[np.eye(6).tolist()],
            anchors=[dict(position=[0.,0.,0.],rotation=np.eye(3).tolist())]))
    reference=copy.deepcopy(current);reference.update(mahalanobis_radius=5.,minimum_original_q=1.,minimum_original_q_inclusive=False)
    reference['gaussian_chart']['means']=[[-3.,0.,0.,0.,0.,0.]]
    return config,current,reference


def ancestry(particles):
    counts={}
    for p in particles:counts[str(p['initial_ancestor'])]=counts.get(str(p['initial_ancestor']),0)+1
    n=len(particles)
    return dict(initial_ancestor_counts=counts,distinct_initial_ancestors=len(counts),
        initial_family_ESS=n*n/sum(x*x for x in counts.values()) if n else 0.,
        largest_initial_family_fraction=max(counts.values())/n if n else None)


class ToyObserver:
    def __init__(self,proposal):self.proposal=proposal;self.classifier_calls=0;self.contact_calls=0
    def classify(self,p,valid):
        self.classifier_calls+=int(valid);self.contact_calls+=int(valid)
        c,r,g=self.proposal.evaluate(p);native=valid and p['position'][0]>0;core=native and r['inside']
        classes=dict(total=valid,registered_native_entry=native,old_R5_intersection_native=core,
            remaining_R4_native=native and not core,contact_no_native_entry=valid and not native,unbound_no_native_entry=False)
        return dict(classes=classes,classification=dict(native_any=native) if valid else None,
                    contact=dict(exclusion_contact=valid) if valid else None,strata=self.proposal.current.bins(c['latent']) if valid else None)


def fixture(directory,zero=False,z=0.,mutate=False):
    root=Path(directory);base=root/'raw';(base/'provenance').mkdir(parents=True)
    config,current,reference=specs(z)
    config['shape']=str(root/'shape.json');write(config['shape'],{'atoms':[]});shape_hash=sha(config['shape'])
    current['shape_sha256']=reference['shape_sha256']=shape_hash
    for name,value in [('config.json',config),('region.json',current),('reference.json',reference)]:write(root/name,value)
    argv=['inert-binary','--config',str(root/'config.json'),'--region',str(root/'region.json'),
        '--initial-reference-region',str(root/'reference.json'),'--initial-current-probability','.5',
        '--bridge','proposal-density','--out',str(base),'--seed','5','--initial-draws','3','--population','2',
        '--stages','1','--sweeps-per-stage','1' if mutate else '0','--cloud-replicates','2','--lambda-ratio','8']
    job=dict(id='r00',seed=5,output=str(base),argv=argv)
    options=parse_job_options(job);proposal=Proposal(current,reference,.5)
    initial=[];valid=[];logs=[]
    for i,x in enumerate([-7.5]*3 if zero else [.5,2.5,-7.5]):
        p=pose(x);c,r,g=proposal.evaluate(p);ok=c['inside'];selected='current' if ok else 'reference';selected_data=c if ok else r
        row=dict(draw=i,pose=p,latent=c['latent'].tolist(),latent_radius=c['radius'],reference_latent=r['latent'].tolist(),
            log_physical_jacobian=c['log_jacobian'],reference_log_physical_jacobian=r['log_jacobian'],
            log_physical_proposal_density=g,current_ball_valid=ok,reference_ball_valid=r['inside'],
            selected_initial_chart=selected,selected_latent=selected_data['latent'].tolist(),
            selected_log_physical_jacobian=selected_data['log_jacobian'],capture_valid=True,hard_valid=ok,
            log_hard_weight=-g if ok else None,log_initial_weight=0. if ok else None)
        initial.append(row)
        if ok:valid.append(i);logs.append(0.)
    if zero:
        stages=[dict(stage=0,activity=0.,log_Z=None,Z=0.,zero_estimate=True,initial_draws=3,initial_hits=0,parents=[],particles=[])]
        terminal=[];logz=None;completed=0
    else:
        particles=[dict(pose=initial[i]['pose'],latent=initial[i]['latent'],initial_ancestor=i) for i in valid]
        logz=math.log(len(valid)/3)
        first=dict(stage=0,activity=0.,initial_draws=3,initial_hits=2,log_Z=logz,log_Z_increment=logz,
            initial_weight_ESS=2.,resampling_offset=0.,parent_initial_draws=valid,particles=copy.deepcopy(particles),ancestry=ancestry(particles))
        potentials=[];stage_logs=[]
        lam=8*z if z else 1.
        for i,p in enumerate(particles):
            g=proposal.evaluate(p['pose'])[2]
            cloud=dict(log_weight=z*.2+2*math.log1p(z/lam) if z else 0.,lower_volume=.2 if z else 0.,
                uncertain_volume=.3 if z else 0.,upper_volume=.5 if z else 0.,raw_points=4 if z else 0,
                overlap_points=2 if z else 0,retained_cells=1,created_cells=1,certified_cells=0)
            w=cloud['log_weight']-g;stage_logs.append(w)
            potentials.append(dict(particle=i,**copy.deepcopy(p),log_g=g,clouds=[copy.deepcopy(cloud),copy.deepcopy(cloud)],
                deterministic_log_correction=-g,log_incremental_weight=w))
        parents=systematic_parents(stage_logs,2,0.);terminal=[copy.deepcopy(particles[i]) for i in parents]
        events=[]
        if mutate:
            for i,p in enumerate(terminal):
                new=pose(p['pose']['position'][0]+.1);old_g=proposal.evaluate(p['pose'])[2];new_g=proposal.evaluate(new)[2]
                gate=dict(gained=3,lost=1,raw_points=4,retained_points=4,log_weight=2*math.log1p(1/64))
                events.append(dict(particle=i,sweep=0,old_pose=copy.deepcopy(p['pose']),proposed_pose=new,log_g_old=old_g,log_g_new=new_g,
                    beta=1.,deterministic_log_correction=0.,gate=gate,log_acceptance=0.,log_uniform=-1.,accepted=True))
                p['pose']=new;p['latent']=proposal.evaluate(new)[0]['latent'].tolist()
        inc=float(logsumexp(stage_logs)-math.log(2));logz+=inc;w=np.exp(np.asarray(stage_logs)-max(stage_logs))
        counts=dict(attempted=2 if mutate else 0,accepted=2 if mutate else 0,capture_rejected=0,region_rejected=0,
            hard_rejected=0,gate_rejected=0,raw_points=8 if mutate else 0,accepted_translation_changes=2 if mutate else 0,accepted_rotation_changes=0)
        stage=dict(stage=1,beta=1.,delta_beta=1.,activity=z,previous_activity=0.,delta_activity=z,incremental_lambda=lam,
            log_Z=logz,log_Z_increment=inc,pre_resampling_weight_ESS=float(w.sum()**2/(w@w)),
            potentials=potentials,parents=parents,resampling_offset=0.,particles=terminal,
            mutation_counts=counts,mutation_density_records=events,ancestry=ancestry(terminal))
        stages=[first,stage];completed=1
    (base/'initialization.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in initial))
    (base/'stages.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in stages))
    for name,source in [('config.json',root/'config.json'),('region.json',root/'region.json'),('shape.json',root/'shape.json'),('initial-reference-region.json',root/'reference.json')]:
        (base/'provenance'/name).write_bytes(source.read_bytes())
    write(base/'provenance/source-bundle.json',{})
    protocol=dict(binary_sha256='inert-binary-hash',source_bundle_sha256=sha(base/'provenance/source-bundle.json'),
        physical_target=dict(region_sha256=sha(root/'region.json'),shape_sha256=shape_hash))
    manifest=dict(schema='latent-region-smc-v1',options=options,executable_sha256=protocol['binary_sha256'],source_bundle_sha256=protocol['source_bundle_sha256'],
        config_sha256=sha(root/'config.json'),region_sha256=sha(root/'region.json'),shape_sha256=shape_hash,
        initial_reference_sha256=sha(root/'reference.json'))
    write(base/'manifest.json',manifest)
    summary=dict(schema='latent-region-smc-summary-v1',complete=True,zero_estimate=zero,log_Z=logz,Z=0. if zero else math.exp(logz),
        initial_draws=3,initial_hits=len(valid),completed_stage=completed,terminal_particles=terminal,ancestry=ancestry(terminal),
        manifest_sha256=sha(base/'manifest.json'),initialization_sha256=sha(base/'initialization.jsonl'),stages_sha256=sha(base/'stages.jsonl'))
    write(base/'summary.json',summary);write(base/'status.json',dict(complete=True,phase='complete',completed_stage=completed,summary_sha256=sha(base/'summary.json')))
    return dict(protocol=protocol,config=config,current=current,reference=reference),job,proposal,initial,stages


class SmcAuditTests(unittest.TestCase):
    def test_initial_all_M_and_unfiltered_reference_exterior_zero(self):
        with tempfile.TemporaryDirectory() as d:
            context,job,proposal,initial,stages=fixture(d)
            result=audit_population(context,job,Path(d)/'analysis',[0,1],ToyObserver(proposal))
            self.assertEqual(result['initial_draws'],3);self.assertEqual(result['initial_hits'],2)
            self.assertEqual(result['exterior_initial_attempts'],1)
            self.assertEqual(result['initial_class_counts']['total'],2)
            self.assertAlmostEqual(result['stages'][0]['log_Z'],math.log(2/3))
            self.assertAlmostEqual(result['initial_bridge_log_masses']['total'],math.log(2/3))
            expected_hard=float(logsumexp([-proposal.evaluate(r['pose'])[2] for r in initial if r['hard_valid']])-math.log(3))
            self.assertAlmostEqual(result['initial_physical_hard_log_masses']['total'],expected_hard)
            aggregated=aggregate([result]*4)
            self.assertEqual(aggregated['population_count'],4)
            self.assertAlmostEqual(aggregated['unresampled_initial_bridge']['estimates']['total']['log_Q'],math.log(2/3))
            self.assertEqual(result['terminal']['counts']['total'],2)
            self.assertAlmostEqual(result['terminal']['log_masses']['total'],stages[-1]['log_Z'])
            with self.assertRaisesRegex(FileExistsError,''):
                audit_population(context,job,Path(d)/'analysis',[0,1],ToyObserver(proposal))

    def test_zero_hit_population_preserved_at_all_declared_profiles(self):
        with tempfile.TemporaryDirectory() as d:
            context,job,proposal,_,_=fixture(d,zero=True)
            result=audit_population(context,job,Path(d)/'analysis',[0,1],ToyObserver(proposal))
            self.assertTrue(result['zero_estimate']);self.assertEqual(result['initial_draws'],3)
            self.assertEqual(set(result['profiles']),{'0','1'})
            self.assertTrue(all(x is None for x in result['terminal']['log_masses'].values()))
            stats=population_statistics([result['terminal']['log_masses']]*4)
            self.assertEqual(stats['population_count'],4);self.assertIsNone(stats['estimates']['total']['log_Q'])
            self.assertIn('neither zero physical',stats['estimates']['total']['unresolved'])

    def test_positive_cloud_gate_history_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            context,job,proposal,_,stages=fixture(d,z=4.,mutate=True)
            result=audit_population(context,job,Path(d)/'analysis',[0,1],ToyObserver(proposal))
            self.assertEqual(len(result['stages']),2)
            args=(stages[0]['particles'],proposal,context['config'],parse_job_options(job),1,stages[0]['log_Z'])
            edits=[('density',lambda s:s['potentials'][0].__setitem__('log_g',s['potentials'][0]['log_g']+.1)),
                   ('cloud',lambda s:s['potentials'][0]['clouds'][0].__setitem__('overlap_points',3)),
                   ('normalizer',lambda s:s.__setitem__('log_Z',s['log_Z']+.2)),
                   ('parent',lambda s:s['parents'].__setitem__(0,1)),
                   ('gate',lambda s:s['mutation_density_records'][0]['gate'].__setitem__('lost',0)),
                   ('move_type',lambda s:s['mutation_counts'].__setitem__('accepted_translation_changes',0)),
                   ('endpoint',lambda s:s['particles'][0]['pose']['position'].__setitem__(0,1.9)),
                   ('ancestor',lambda s:s['particles'][0].__setitem__('initial_ancestor',99))]
            for name,change in edits:
                with self.subTest(name=name):
                    stage=copy.deepcopy(stages[1]);change(stage)
                    with self.assertRaises(ValueError):audit_stage(stage,*args)
            raw=Path(job['output'])/'initialization.jsonl';raw.write_text(raw.read_text()+'{}\n')
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):
                audit_population(context,job,Path(d)/'corrupt',[0,1],ToyObserver(proposal))

    def test_chart_measure_and_old_filter_independence(self):
        config,current,reference=specs();reference['capture_radius']=.01;reference['minimum_original_q']=999.;reference['minimum_mahalanobis_radius']=4.99
        proposal=Proposal(current,reference,.5)
        for x in [.5,2.5,-7.5]:
            c,r,g=proposal.evaluate(pose(x));j=1/math.pi**2
            expected=(.5/(math.pi**3*4**6/6*j) if abs(x)<=4 else 0)+(.5/(math.pi**3*5**6/6*j) if abs(x+3)<=5 else 0)
            self.assertAlmostEqual(g,math.log(expected));self.assertEqual(c['inside'],abs(x)<=4)
        q=np.array([1.,.4,-.2,.3]);q=q/np.linalg.norm(q);p=dict(position=[.5,0.,0.],orientation=q.tolist())
        c=Chart(current).evaluate(p);expected_j=1/(math.pi**2*(1+(.4**2+.2**2+.3**2))**2)
        self.assertAlmostEqual(c['log_jacobian'],math.log(expected_j))

    def test_whole_population_covariance_and_zero_are_not_descendant_errors(self):
        rows=[]
        for core,comp,other in [(1,3,2),(3,5,4),(0,0,0),(2,2,2)]:
            linear=dict(total=core+comp+other,registered_native_entry=core+comp,old_R5_intersection_native=core,
                remaining_R4_native=comp,contact_no_native_entry=other,unbound_no_native_entry=0)
            rows.append({k:math.log(v) if v else None for k,v in linear.items()})
        result=population_statistics(rows);self.assertAlmostEqual(result['estimates']['total']['log_Q'],math.log(6))
        self.assertEqual(result['estimates']['total']['nonzero_populations'],3)
        self.assertAlmostEqual(result['free_energy_contrast']['beta_F_native_minus_noentry'],-math.log(2))
        self.assertAlmostEqual(result['free_energy_contrast']['SE'],0.)
        expected=np.cov(np.array([[6,4,1,3,2,0],[12,8,3,5,4,0],[0,0,0,0,0,0],[6,4,2,2,2,0]]),rowvar=False,ddof=1)/4
        covariance=result['covariance_of_mean'];actual=np.asarray(covariance['scaled_matrix'])*math.exp(covariance['log_scale'])
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        self.assertNotIn('importance_ESS',result)

    def test_physical_identity_excludes_only_documented_proposal_fields(self):
        a,_,_=specs();b=copy.deepcopy(a);b['translation_steps']=[.05];b['rotation_steps_deg']=[.1]
        validate_config_identity(a,b,('translation_steps','rotation_steps_deg'))
        with self.assertRaises(ValueError):validate_config_identity(a,b)
        b['rotation_probability']=.6
        with self.assertRaises(ValueError):validate_config_identity(a,b,('translation_steps','rotation_steps_deg'))
        with self.assertRaises(ValueError):validate_config_identity(a,a,('reservoir_density',))

    def test_freeze_corruption_and_resampling_offset(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'protocol.json',{'zero':'retained'});write(root/'freeze.json',{'files':{'protocol.json':sha(root/'protocol.json')}})
            ledger=Ledger();ledger.frozen(root);write(root/'protocol.json',{'zero':'removed'})
            with self.assertRaises(ValueError):ledger.recheck()
        self.assertEqual(systematic_parents([0.,0.],2,0.),[0,1])
        with self.assertRaises(ValueError):systematic_parents([0.,0.],2,.5)

    def test_direct_contrast_catches_opposite_mass_shifts(self):
        cfg,current,reference=specs()
        target=dict(shape_sha256='shape',region_sha256='R4',fixed_poses=cfg['fixed_poses'],capture_center=cfg['capture_center'],
            capture_radius=170.,activity=0.,depletant_radius=1.5,measure='physical')
        p=dict(physical_target=target,proposal={'reference_region_sha256':'R5'},jobs=[{'seed':i} for i in range(4)],
            allocation=dict(independent_populations=4,unconditional_initialization_draws_each=3))
        estimates={k:dict(log_Q=10.,population_relative_SE=.1) for k in CLASSES}
        a=dict(complete=True,native_definition_sha256='same',protocol_snapshot=p,physical_config=cfg,
            summary=dict(estimates=estimates,population_count=4,free_energy_contrast=dict(SE=.1,beta_F_native_minus_noentry=-2.),
                strata={f:[copy.deepcopy(estimates) for _ in range(size)] for f,size in [('radial',3),('angular',3),('orthant',64)]}),
            populations=[dict(id='r'+str(i),seed=i,zero_estimate=False,initial_draws=3,initial_hits=2) for i in range(4)])
        b=copy.deepcopy(a);b['protocol_snapshot']['jobs']=[{'seed':i+10} for i in range(4)]
        b['summary']['estimates']['registered_native_entry']['log_Q']+=.14
        b['summary']['estimates']['contact_no_native_entry']['log_Q']-=.14
        b['summary']['free_energy_contrast']['beta_F_native_minus_noentry']-=.28
        result=compare_analyses(a,b)
        self.assertTrue(result['classes']['registered_native_entry']['passed'])
        self.assertTrue(result['classes']['contact_no_native_entry']['passed'])
        self.assertFalse(result['direct_free_energy_contrast']['passed'])
        self.assertEqual(result['controls'][0]['populations'][0]['initial_draws'],3)
        self.assertEqual(len(result['strata']['orthant']),64)
        dropped=copy.deepcopy(b);dropped['populations'].pop()
        with self.assertRaises(ValueError):compare_analyses(a,dropped)
        b['physical_config']['reservoir_density']=.02
        with self.assertRaises(ValueError):compare_analyses(a,b)

if __name__=='__main__':unittest.main()
