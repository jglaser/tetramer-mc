"""Synthetic physical-row validation; no protein or physical random draws."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from scipy.stats import poisson

import native_class_line_physical_reference as audit
import native_class_line_reference as line
from prepare_native_class_line_physical_toy import prepare


def save(path, data): path.write_text(json.dumps(data, allow_nan=False)+'\n')


def contract():
    return dict(density_trace_contract=dict(format='class-line-compact-v1',diagnostic_full_trace_unchanged=True),
        random_stream_contract=dict(hash_domain='tetramer-uniform-latent-region-v1',
          key_fields=['master seed u64 little endian','draw u64 little endian','cloud index u64 little endian','role UTF-8 bytes'],
          proposal_role='latent',proposal_cloud_index=0,physical_cloud_role='cloud',
          physical_cloud_indices='0..cloud_replicates, separately derived; no proposal RNG consumption'))


def compact(recon, u):
    result = recon.density(u)
    if result.get('conditioning_disabled'): return dict(conditioning_disabled=True)
    axes = []
    for actual in result['axes']:
        a=copy.deepcopy(actual);a.pop('components'); axis=a['axis'];raw=recon.raw(u);raw[axis]=0
        u0=line.scalar_chart_solve(recon.L0,raw-recon.m0);du=line.scalar_chart_solve(recon.L0,np.eye(6)[axis])
        a.update(latent_line_origin=u0.tolist(),latent_line_direction=du.tolist())
        for c in a['channels']:
            c['orthant_intervals'] = line.orthant_intervals(u0,du,c['orthant'],a['segment']) if c.get('orthant') is not None and 'segment'in a else None
        axes.append(a)
    return dict(trace_format='class-line-compact-v1',raw_coordinates=recon.raw(u).tolist(),
        baseline_log_density=result['baseline_log_density'],axes=axes,
        component_mixture_multipliers=((np.asarray(result['component_multipliers'])-1+recon.beta)/recon.beta).tolist())


def fixture(base, beta=0.):
    inputs=base/'input';plan=prepare(inputs);job=next(j for j in plan['jobs'] if j['activity']==.3 and j['method']=='class')
    config=audit.read(job['config']);region=audit.read(job['region']);guide=audit.read(job['guide']);guide['conditional_probability']=beta
    shape=audit.read(inputs/'common/shape.json');compiled=audit.read(inputs/'common/compiled-native.json')
    out=base/'result';out.mkdir();prov=out/'provenance';prov.mkdir()
    for name,value in [('input-config.json',config),('region.json',region),('importance-guide.json',guide),('shape.json',shape),('source-bundle.json',dict(toy=True)),('compiled-native.json',compiled)]:save(prov/name,value)
    region['shape_sha256']=audit.sha(prov/'shape.json');region['gaussian_chart']['shape_sha256']=region['shape_sha256'];save(prov/'region.json',region)
    compiled['source_input_sha256']['tetramer-shape.json']=region['shape_sha256'];save(prov/'compiled-native.json',compiled)
    guide['region_sha256']=audit.sha(prov/'region.json');guide['shape_sha256']=region['shape_sha256'];guide['compiled_native']['sha256']=audit.sha(prov/'compiled-native.json');save(prov/'importance-guide.json',guide)
    recon=line.Reconstructor(region,guide,config,shape,line.observer_from_compiled_for_synthetic(compiled))
    rows=[]
    for index,u in enumerate([np.array([2.5,0,0,0,0,0.]),np.array([.1,0,0,0,0,0.]),np.array([8.,0,0,0,0,0.])]):
        raw,position,rotation,jac=recon.decode(u);density=recon.density(u);shell=bool(np.linalg.norm(u)<=4);hard=recon.hard_valid(position,rotation);valid=hard and shell
        clouds=[]
        if valid:
            for k in [1,3]:
                clouds.append(dict(lower_volume=.1,upper_volume=.5,uncertain_volume=.4,raw_points=4,overlap_points=k,
                    retained_cells=2,created_cells=5,certified_cells=1,log_weight=.3*.1+k*math.log1p(.3/19.2)))
        pose=dict(position=position.tolist(),orientation=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]].tolist())
        h=jac-density['log_density'] if valid else None
        row=dict(draw=index,latent=u.tolist(),latent_radius=float(np.linalg.norm(u)),pose=pose,
            backmapped_latent=u.tolist(),backmapped_radius=float(np.linalg.norm(u)),q=audit.physical.native.native_q(region['physical_metric'],pose),
            log_physical_jacobian=jac,physical_jacobian=math.exp(jac),shell_valid=shell,capture_valid=True,hard_valid=bool(hard),region_valid=True,
            log_proposal_density=density['log_density'],log_hard_weight=h,
            log_importance_weight=h+float(logsumexp([c['log_weight'] for c in clouds]))-math.log(2) if valid else None,
            clouds=clouds,proposal_branch='native-class-line',proposal_component=0,
            native_class_line_draw=dict(conditional=False,original_latent=u.tolist()),native_class_line_density=compact(recon,u))
        rows.append(row)
    (out/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n'for row in rows));(out/'attempts.jsonl').write_text(''.join(json.dumps(dict(draw=i,state='begin'))+'\n'for i in range(3)))
    manifest=dict(schema=audit.SCHEMA,guide_schema=audit.GUIDE_SCHEMA,proposal_kind=audit.PROPOSAL_KIND,
        samples=3,seed=123,cloud_replicates=2,activity=.3,lambda_ratio=64.,**{'lambda':19.2},
        compiled_native=dict(compiled_sha256=audit.sha(prov/'compiled-native.json'),source_definition_sha256=compiled['source_definition_sha256'],source_input_sha256=compiled['source_input_sha256'],shape_compatibility=dict(
          compiled_sha256=audit.sha(prov/'compiled-native.json'),expected_shape_sha256=region['shape_sha256'],
          native_atoms=4,physical_atoms=4,matched_atoms=4,center_tolerance_a=1e-10,radius_tolerance_a=1e-12,
          matched_max_center_error_a=0.,matched_max_radius_error_a=0.,physical_index_by_native_atom=list(range(4)),
          unmatched_native_atoms=[],unmatched_physical_atoms=[],compatible=True,pair_overlap_slack_bound_a=0.,
          observer_hard_overlap_tolerance_a=1e-8,hard_valid_implication_within_tolerance=True)),
        physical_fixed_neighbors=config['fixed_poses'],chart_anchor=region['fixed_neighbor'],importance_uniform_probability=.5,
        importance_component_count=2,proposal_density_measure=audit.MEASURE,latent_radius=4.,minimum_latent_radius=0.,
        log_latent_ball_volume=recon.logvolume,log_latent_shell_volume=recon.logvolume,minimum_original_q=0.,maximum_original_q=None,
        minimum_original_q_inclusive=True,maximum_original_q_inclusive=True,resume_supported=False,executable_sha256='0'*64,**contract())
    for name,key in [('input-config.json','config_sha256'),('region.json','region_sha256'),('shape.json','shape_sha256'),('importance-guide.json','importance_guide_sha256'),('source-bundle.json','source_bundle_sha256')]:manifest[key]=audit.sha(prov/name)
    save(out/'manifest.json',manifest)
    arrays=audit.read_weights(rows,3,region,dict(alpha=.5,component_count=2))
    summary=dict(complete=True,manifest=manifest,samples=3,samples_sha256=audit.sha(out/'samples.jsonl'),attempts_sha256=audit.sha(out/'attempts.jsonl'),capture_rejected=0,hard_rejected=1,region_rejected=0,shell_rejected=1,raw_points=8,sampler_cpu_seconds=.1,
        estimates=dict(region=audit.physical.statistics.moments(arrays['z']),hard_region=audit.physical.statistics.moments(arrays['h'])))
    save(out/'summary.json',summary)
    return out,recon,rows,manifest,summary


class PhysicalClassReferenceTests(unittest.TestCase):
    def test_pgf_mean_and_variance_exact_poisson_sum(self):
        for z,lam,L,B in [(0.,1.,.4,.7),(.3,.8,.2,1.3),(.2,3.,0.,0.)]:
            expected_mean,expected_var=audit.poisson_weight_moments(z,lam,L,B)
            k=np.arange(128);p=poisson.pmf(k,lam*B);w=np.exp(z*L)*(1+z/lam)**k
            self.assertAlmostEqual(float(p@w),expected_mean,places=13)
            self.assertAlmostEqual(float(p@((w-expected_mean)**2)),expected_var,places=13)

    def test_haar_cap_matches_jacobian_radial_integral(self):
        for upper in [0.,1e-4,.1,.5,2.,100.]:
            numerical=quad(lambda c:4/math.pi*c*c/(1+c*c)**2,0,upper,epsabs=1e-14)[0]
            self.assertAlmostEqual(audit.cayley_haar_ball_mass(upper),numerical,places=12)
        self.assertAlmostEqual(audit.equal_sphere_overlap(1.5,0.),4*math.pi*1.5**3/3,places=12)
        self.assertEqual(audit.equal_sphere_overlap(1.5,3.),0.)

    def test_one_dimensional_analytic_integral_keeps_six_ball_coupling(self):
        expected=audit.analytic_sphere_mass(1.,.5,.3)['mass']
        direct=quad(lambda r:4*math.pi*r*r*math.exp(.3*audit.equal_sphere_overlap(1.5,r))
            *quad(lambda a:4/math.pi*a*a/(1+a*a)**2,0,math.sqrt(16-r*r)/8,epsabs=1e-13)[0],2,4,points=[3],epsabs=1e-10)[0]
        self.assertAlmostEqual(expected,direct,places=10)
        hard=audit.analytic_sphere_mass(1.,.5,0.)['mass'];self.assertGreater(expected,hard)
        self.assertEqual(audit.analytic_sphere_mass(3.,.5,0.)['mass'],0.)

    def test_stream_contract_rejects_shared_roles(self):
        manifest=dict(seed=123,**contract());audit.validate_stream_contract(manifest)
        keys=[audit.stream_key(123,0,k,role)for k,role in[(0,'latent'),(0,'cloud'),(1,'cloud')]]
        self.assertEqual(len(set(keys)),3)
        bad=copy.deepcopy(manifest);bad['random_stream_contract']['physical_cloud_role']='latent'
        with self.assertRaisesRegex(ValueError,'role contract'):audit.validate_stream_contract(bad)

    def test_all_attempt_zeros_and_two_cloud_linear_weight(self):
        with tempfile.TemporaryDirectory()as d:
            out,r,rows,m,s=fixture(Path(d));result=audit.audit(out,synthetic=True,journal=Path(d)/'journal.jsonl')
            self.assertTrue(result['complete']);self.assertEqual(result['samples'],3);self.assertEqual(result['finite_count'],1)
            self.assertEqual(result['estimate']['draws'],3);self.assertEqual(result['counts']['shell_rejected'],1)
            self.assertEqual(result['independently_checked_distinct_role_keys'],9)
            bad=copy.deepcopy(rows);bad[1]['log_hard_weight']=0.
            with self.assertRaisesRegex(ValueError,'zero weight'):audit.read_weights(bad,3,r.region,dict(alpha=.5,component_count=2))
            bad=copy.deepcopy(rows);bad[0]['log_importance_weight']=bad[0]['log_hard_weight']+sum(c['log_weight']for c in bad[0]['clouds'])/2
            with self.assertRaisesRegex(ValueError,'arithmetic mean'):audit.read_weights(bad,3,r.region,dict(alpha=.5,component_count=2))

    def test_full_compact_class_density_and_mixture_tampering(self):
        with tempfile.TemporaryDirectory()as d:
            out,r,rows,m,s=fixture(Path(d),beta=1.)
            for row in rows:
                result=audit.compact_density(r,np.array(row['latent']),row['native_class_line_density'])
                self.assertAlmostEqual(result['log_density'],row['log_proposal_density'],places=12)
            self.assertTrue(audit.audit(out,synthetic=True)['passed'])
            bad=copy.deepcopy(rows[0]['native_class_line_density']);bad['component_mixture_multipliers'][0]+=1
            with self.assertRaisesRegex(ValueError,'mixture'):audit.compact_density(r,np.array(rows[0]['latent']),bad)
            bad=copy.deepcopy(rows[0]['native_class_line_density']);bad['axes']=bad['axes'][:-1]
            with self.assertRaisesRegex(ValueError,'axes'):audit.compact_density(r,np.array(rows[0]['latent']),bad)

    def test_missing_draw_replica_and_native_lineage_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            out,r,rows,m,s=fixture(Path(d));arm=dict(alpha=.5,component_count=2)
            with self.assertRaisesRegex(ValueError,'unconditional'):audit.read_weights(rows[:1],3,r.region,arm)
            bad=copy.deepcopy(rows);bad[0]['clouds']=bad[0]['clouds'][:1]
            with self.assertRaisesRegex(ValueError,'two independent'):audit.read_weights(bad,3,r.region,arm)
            bad=copy.deepcopy(rows);bad[0]['proposal_branch']='gaussian'
            with self.assertRaisesRegex(ValueError,'branch'):audit.read_weights(bad,3,r.region,arm)

    def test_static_native_shape_bijection_and_shape_hash(self):
        with tempfile.TemporaryDirectory()as d:
            out,r,rows,m,s=fixture(Path(d));compiled=audit.read(out/'provenance/compiled-native.json');shape=audit.read(out/'provenance/shape.json');report=m['compiled_native']['shape_compatibility'];compiled_hash=m['compiled_native']['compiled_sha256']
            self.assertTrue(audit.validate_shape_witness(compiled,shape,report,compiled_hash,m['shape_sha256'])['complete_bijection'])
            bad=copy.deepcopy(report);bad['physical_index_by_native_atom']=[0,0,2,3]
            with self.assertRaisesRegex(ValueError,'bijection'):audit.validate_shape_witness(compiled,shape,bad,compiled_hash,m['shape_sha256'])
            bad=copy.deepcopy(compiled);bad['source_input_sha256']['tetramer-shape.json']='f'*64
            with self.assertRaisesRegex(ValueError,'source shape hash'):audit.validate_shape_witness(bad,shape,report,compiled_hash,m['shape_sha256'])
            bad=copy.deepcopy(shape);bad['atoms'][0]['radius']+=1e-6
            with self.assertRaisesRegex(ValueError,'geometry differs'):audit.validate_shape_witness(compiled,bad,report,compiled_hash,m['shape_sha256'])

    def test_failure_and_changed_inputs_remain_fatal(self):
        for mode in ['failure','journal','shape','native_hash']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory()as d:
                out,r,rows,m,s=fixture(Path(d))
                if mode=='failure':save(out/'failure.json',dict(complete=False))
                if mode=='journal':
                    (out/'attempts.jsonl').write_text('');s['attempts_sha256']=audit.sha(out/'attempts.jsonl');save(out/'summary.json',s)
                if mode=='shape':(out/'provenance/shape.json').write_text('{}')
                if mode=='native_hash':(out/'provenance/compiled-native.json').write_text('{}')
                with self.assertRaises(ValueError):audit.audit(out,synthetic=True)


if __name__=='__main__':unittest.main()
