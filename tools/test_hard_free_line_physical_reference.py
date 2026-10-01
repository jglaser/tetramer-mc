import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
import hard_free_line_physical_reference as audit
from test_hard_free_line_reference import setup


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False)+'\n')


def cloud(k=1, activity=.2):
    return dict(lower_volume=.1, upper_volume=.5, uncertain_volume=.4, raw_points=3, overlap_points=k,
                retained_cells=2, created_cells=5, certified_cells=1,
                log_weight=activity*.1+k*math.log1p(activity/2.))


def make_rows(region, guide, config, shape):
    recon = audit.line.Reconstructor(region, guide, config, shape); rows = []
    # Explicit valid central gap, hard collision, and unbounded Gaussian tail.
    for i, u in enumerate([np.zeros(6), np.array([.1,0.,0.,0.,0.,0.]), np.array([8.,0.,0.,0.,0.,0.])]):
        raw, p, R, j = recon.decode(u); q = recon.density(u)['log_density']
        pose = dict(position=p.tolist(), orientation=Rotation.from_matrix(R).as_quat()[[3,0,1,2]].tolist())
        hard = recon.hard_valid(p, R); shell = np.linalg.norm(u) <= recon.radius; valid = hard and shell
        clouds = [cloud(1), cloud(3)] if valid else []
        h = j-q if valid else None; w = float(h+logsumexp([c['log_weight'] for c in clouds])-math.log(2)) if valid else None
        rows.append(dict(draw=i, latent=u.tolist(), latent_radius=float(np.linalg.norm(u)), pose=pose,
            backmapped_latent=u.tolist(), backmapped_radius=float(np.linalg.norm(u)), log_physical_jacobian=j,
            physical_jacobian=math.exp(j), q=audit.native.native_q(region['physical_metric'], pose),
            shell_valid=bool(shell), capture_valid=True, hard_valid=bool(hard), region_valid=True,
            log_importance_weight=w, log_hard_weight=h, clouds=clouds, log_proposal_density=q,
            proposal_branch='hard-free-line', proposal_component=0,
            hard_free_line_draw=dict(conditional=False, original_latent=u.tolist()),
            hard_free_line_density=dict(conditioning_disabled=True)))
    return rows


def fixture(root):
    reg, guide, cfg, shape = setup((0,1,2), beta=0.)
    metric = dict(native_poses=[dict(position=[0.,0.,0.], orientation=[1.,0.,0.,0.])],
                  rigid_members=[dict(position=[0.,0.,0.])], member_error_scale=1., angle_error_scale_deg=30.)
    cfg.update(depletant_radius=.1, reservoir_density=.2, metadata=metric)
    reg.update(physical_fixed_neighbors=cfg['fixed_poses'], capture_center=cfg['capture_center'], capture_radius=cfg['capture_radius'],
               activity=.2, depletant_radius=.1, physical_metric=metric, minimum_original_q=0.)
    prov=root/'provenance'; prov.mkdir(); write(prov/'shape.json', shape)
    reg['shape_sha256']=audit.sha(prov/'shape.json');reg['gaussian_chart']['shape_sha256']=reg['shape_sha256']
    write(prov/'region.json', reg); guide['region_sha256']=audit.sha(prov/'region.json')
    write(prov/'importance-guide.json', guide); write(prov/'input-config.json', cfg); write(prov/'source-bundle.json', dict(toy=True))
    recon=audit.line.Reconstructor(reg,guide,cfg,shape);rows=make_rows(reg,guide,cfg,shape)
    (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'attempts.jsonl').write_text(''.join(json.dumps(dict(draw=i,state='begin'))+'\n' for i in range(len(rows))))
    manifest=dict(schema=audit.SCHEMA,guide_schema=audit.GUIDE_SCHEMA,proposal_kind=audit.PROPOSAL_KIND,
        samples=len(rows),seed=100,cloud_replicates=2,activity=.2,lambda_ratio=10.,**{'lambda':2.},
        executable_sha256='0'*64,physical_fixed_neighbors=cfg['fixed_poses'],chart_anchor=reg['fixed_neighbor'],
        importance_uniform_probability=.5,importance_component_count=2,proposal_density_measure=audit.MEASURE,
        latent_radius=4.,minimum_latent_radius=0.,log_latent_ball_volume=recon.logvolume,log_latent_shell_volume=recon.logvolume,
        minimum_original_q=0.,maximum_original_q=None,minimum_original_q_inclusive=True,maximum_original_q_inclusive=True)
    for name,key in [('input-config.json','config_sha256'),('shape.json','shape_sha256'),('region.json','region_sha256'),
                     ('importance-guide.json','importance_guide_sha256'),('source-bundle.json','source_bundle_sha256')]:manifest[key]=audit.sha(prov/name)
    write(root/'manifest.json',manifest)
    arrays=audit.read_weights(rows,len(rows),reg,dict(alpha=.5,component_count=2))
    summary=dict(complete=True,manifest=manifest,samples=len(rows),samples_sha256=audit.sha(root/'samples.jsonl'),
        attempts_sha256=audit.sha(root/'attempts.jsonl'),capture_rejected=0,hard_rejected=sum(not r['hard_valid'] for r in rows),
        region_rejected=0,shell_rejected=1,raw_points=6,sampler_cpu_seconds=.1,
        estimates=dict(region=audit.statistics.moments(arrays['z']),hard_region=audit.statistics.moments(arrays['h'])))
    write(root/'summary.json',summary)
    return reg,guide,cfg,shape,rows,manifest,summary


class PhysicalReferenceTests(unittest.TestCase):
    def test_count_weight_and_zero_limits(self):
        self.assertAlmostEqual(audit.cloud_log_weight(cloud(),.2,2.),.02+math.log1p(.1))
        c=cloud();c.update(raw_points=0,overlap_points=0,log_weight=0.)
        self.assertEqual(audit.cloud_log_weight(c,0.,1.),0.)
        c=cloud();c.update(uncertain_volume=0.,upper_volume=.1,retained_cells=0,raw_points=0,overlap_points=0,log_weight=.02)
        self.assertAlmostEqual(audit.cloud_log_weight(c,.2,2.),.02)
        c['raw_points']=1
        with self.assertRaisesRegex(ValueError,'has points'):audit.cloud_log_weight(c,.2,2.)

    def test_cloud_tampering_rejected(self):
        for key,value,message in [('overlap_points',4,'exceeds'),('log_weight',1.,'weight differs'),('upper_volume',2.,'upper volume'),('raw_points',3.5,'Invalid cloud count')]:
            c=cloud();c[key]=value
            with self.assertRaisesRegex(ValueError,message):audit.cloud_log_weight(c,.2,2.)

    def test_full_population_keeps_invalid_and_exterior_zeros(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root);result=audit.audit(root)
            self.assertTrue(result['complete']);self.assertEqual(result['samples'],3);self.assertEqual(result['finite_count'],1)
            self.assertEqual(result['estimate']['draws'],3);self.assertEqual(result['estimate']['ess'],1.)
            self.assertEqual(result['counts']['shell_rejected'],1)
            self.assertEqual(result['branch_labels']['1'],'hard-free-line')

    def test_immutable_provenance_and_attempt_journal(self):
        for mode in ('provenance','journal','failure'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root=Path(d);*_,summary=fixture(root)
                if mode=='provenance':(root/'provenance/shape.json').write_text('{}\n')
                if mode=='journal':
                    (root/'attempts.jsonl').write_text(json.dumps(dict(draw=0,state='begin'))+'\n')
                    summary['attempts_sha256']=audit.sha(root/'attempts.jsonl');write(root/'summary.json',summary)
                if mode=='failure':write(root/'failure.json',dict(complete=False))
                with self.assertRaises(ValueError):audit.audit(root)

    def test_all_zero_population_is_unresolved_with_full_denominator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reg,_,_,_,rows,_,summary=fixture(root)
            rows=[dict(copy.deepcopy(rows[1]),draw=i) for i in range(3)]
            (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            arrays=audit.read_weights(rows,3,reg,dict(alpha=.5,component_count=2))
            summary.update(samples_sha256=audit.sha(root/'samples.jsonl'),hard_rejected=3,shell_rejected=0,raw_points=0,
                estimates=dict(region=audit.statistics.moments(arrays['z']),hard_region=audit.statistics.moments(arrays['h'])))
            write(root/'summary.json',summary);result=audit.audit(root)
            self.assertEqual(result['finite_count'],0);self.assertEqual(result['estimate']['draws'],3)
            self.assertIsNone(result['estimate']['logQ']);self.assertIsNone(result['paired_noise'])

    def test_nonzero_invalid_and_missing_denominator_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            reg,_,_,_,rows,_,_=fixture(Path(d));arm=dict(alpha=.5,component_count=2)
            bad=copy.deepcopy(rows);bad[1]['log_importance_weight']=0.
            with self.assertRaisesRegex(ValueError,'zero weight'):audit.read_weights(bad,3,reg,arm)
            with self.assertRaisesRegex(ValueError,'unconditional'):audit.read_weights(rows[:1],3,reg,arm)
            bad=copy.deepcopy(rows);bad[0]['proposal_branch']='gaussian'
            with self.assertRaisesRegex(ValueError,'hard-free-line branch'):audit.read_weights(bad,3,reg,arm)

    def test_two_cloud_mean_is_linear_not_geometric(self):
        with tempfile.TemporaryDirectory() as d:
            reg,_,_,_,rows,_,_=fixture(Path(d));row=rows[0]
            row['log_importance_weight']=row['log_hard_weight']+sum(c['log_weight'] for c in row['clouds'])/2
            with self.assertRaisesRegex(ValueError,'arithmetic mean'):audit.read_weights(rows,3,reg,dict(alpha=.5,component_count=2))

    def test_compact_all_axes_density_and_geometry_tamper(self):
        reg,g,cfg,shape=setup((0,1,2),beta=1.);r=audit.line.Reconstructor(reg,g,cfg,shape);u=np.zeros(6)
        full=r.density(u);details=dict(raw_coordinates=r.raw(u).tolist(),baseline_log_density=full['baseline_log_density'],
            component_branches=full['component_branches'],fallback_component_branches=full['fallback_component_branches'],axes=[])
        for a in full['axes']:
            one=copy.deepcopy(g);one['raw_translation_axes']=[a['axis']]
            q=audit.line.Reconstructor(reg,one,cfg,shape).density(u)['log_density']
            detail={k:copy.deepcopy(v) for k,v in a.items() if not k.startswith('conditional_') and k!='component_fallbacks'}
            detail['hard_free_intervals']=detail.pop('intervals');detail['axis_log_proposal_density']=q;details['axes'].append(detail)
        got=audit.compact_density(r,u,details)
        self.assertAlmostEqual(got['log_density'],full['log_density'],places=12)
        bad=copy.deepcopy(details);bad['axes'][0]['hard_free_intervals'][0]['lower']-=.01
        with self.assertRaisesRegex(ValueError,'hard-free interval'):audit.compact_density(r,u,bad)
        bad=copy.deepcopy(details);bad['axes']=bad['axes'][:-1]
        with self.assertRaisesRegex(ValueError,'density axes'):audit.compact_density(r,u,bad)


if __name__=='__main__':unittest.main()
