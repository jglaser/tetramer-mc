"""Regression checks for physical region and unnormalized terminal estimators."""
import copy
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_smc_shoulder_control import (ORIGINAL_METRIC, ball_volume, flat_density, haar_cap, population_statistics, recompute_q, rescale_config, sphere_exact)
POSE={'position':[0.,0.,0.], 'orientation':[1.,0.,0.,0.]}
ENV={'native_pose':POSE,'rigid_members':[dict(POSE,position=[-2.,1.,0.]),dict(POSE,position=[2.,-1.,0.])], 'capture_radius':18.}
SOURCE={**ORIGINAL_METRIC,'proposal_components':[dict(POSE,position_radius=2.,angle_radius_deg=15.,weight=1.)]}
class ShoulderControl(unittest.TestCase):
    def test_explicit_override_and_flat_density(self):
        cfg=rescale_config(SOURCE,ENV,512,1)
        self.assertGreater(cfg['proposal_components'][0]['position_radius'],4.)
        self.assertEqual(cfg['proposal_components'][0]['angle_radius_deg'],30.)
        self.assertEqual(cfg['metadata']['original_metric'],ORIGINAL_METRIC)
        self.assertAlmostEqual(flat_density(cfg,ENV),.491617546504177,places=13)
        self.assertEqual(SOURCE['proposal_components'][0]['position_radius'],2.)
    def test_rescaled_metric_for_arbitrary_proper_poses(self):
        cfg=rescale_config(SOURCE,ENV,512,1);rng=np.random.default_rng(918)
        for quat,t in zip(Rotation.random(100,random_state=rng).as_quat(),rng.normal(size=(100,3))):
            pose={'orientation':quat[[3,0,1,2]].tolist(),'position':t.tolist()}
            self.assertAlmostEqual(recompute_q(pose,ENV,cfg)*2,recompute_q(pose,ENV,ORIGINAL_METRIC),places=13)
    def test_noncentered_or_multiple_references_rejected(self):
        env=copy.deepcopy(ENV);env['rigid_members'][0]['position'][0]+=.1
        with self.assertRaisesRegex(ValueError,'centered'):rescale_config(SOURCE,env,512,1)
        env=copy.deepcopy(ENV);env['native_poses']=[POSE,POSE]
        with self.assertRaisesRegex(ValueError,'multiple'):rescale_config(SOURCE,env,512,1)
    def test_original_native_and_shoulder_masks_are_disjoint(self):
        for q in [0.,1.,math.nextafter(1.,math.inf),1.9,2.,2.1]:
            self.assertEqual(int(q<=1.)+int(1.<q<=2.),int(q<=2))
    def test_covariance_cannot_be_dropped(self):
        s=population_statistics([[1.,3.],[2.,2.],[3.,1.],[4.,0.]])
        self.assertAlmostEqual(s['regions']['total']['Q_se'],0.)
        self.assertLess(s['covariance_of_mean_native_shoulder'][0][1],0.)
        self.assertGreater(s['regions']['native']['Q_se'],0.)
    def test_zero_populations_are_not_discarded(self):
        s=population_statistics([[0.,0.],[2.,4.],[0.,0.],[2.,4.]])
        self.assertEqual(s['regions']['total']['mean_Q'],3.)
        self.assertEqual(s['population_count'],4)
    def test_terminal_fraction_needs_population_normalizer(self):
        p=np.array([.9,.1]);v=np.array([2.,8.]);w=v/p
        estimators=np.column_stack((w*np.array([1.,0.]),w*np.array([0.,1.])))
        np.testing.assert_allclose(p@estimators,v)
        self.assertNotAlmostEqual(float(p@np.array([1.,0.])),v[0])
    def test_sphere_zero_reference_and_positive_weight(self):
        zero=sphere_exact(0.);positive=sphere_exact(.4)
        self.assertAlmostEqual(zero['total'],haar_cap(30.)*(ball_volume(3.)-ball_volume(1.)),places=13)
        self.assertAlmostEqual(zero['native'],haar_cap(15.)*(ball_volume(2.)-ball_volume(1.)),places=13)
        self.assertAlmostEqual(positive['native']+positive['shoulder'],positive['total'])
        self.assertGreater(positive['native'],zero['native'])
        self.assertGreater(positive['total'],zero['total'])

class SavedStageAudit(unittest.TestCase):
    def fixture(self,root):
        import json
        from pathlib import Path
        g=math.log(2.)
        p=[{'family_id':j,'log_g':g,'q':.5} for j in range(2)]
        initialization={'hits':2,'draws':4}
        summary={'initialization':initialization,'logZ':-2*g,'final_particles':p}
        cfg={'smc_lambda_ratio':1.,'reference_activity':.4,'reservoir_density':0.,'population':2,'schedule':[0.,1.]}
        common={'family_ess':2.,'distinct_families':2}
        a=dict(common,stage=0,t=0.,logZ_increment=-g,logZ_total=-g)
        b=dict(common,stage=1,t=1.,K=[0,0],log_g=[g,g],log_weights=[-g,-g],logZ_increment=-g,logZ_total=-2*g,conditional_count_intensity=.4,z_eff=0.,smc_lambda=.4,log_c=0.,parents=[0,1],weight_ess=2.)
        stages=[a,b]
        (root/'stages.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in stages))
        (root/'populations.jsonl').write_text(''.join(json.dumps({'stage':j,'t':float(j),'particles':p})+'\n' for j in range(2)))
        return cfg,summary,stages
    def test_stage_audit_and_corrupted_factor_rejected(self):
        import tempfile,json
        from pathlib import Path
        from prepare_smc_shoulder_control import audit_stages
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,summary,stages=self.fixture(root)
            self.assertEqual(audit_stages(root,cfg,summary,2.)['stages_checked'],2)
            stages[1]['log_weights'][0]+=.2
            (root/'stages.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in stages))
            with self.assertRaisesRegex(ValueError,'weight mismatch'):audit_stages(root,cfg,summary,2.)
    def test_corrupted_lineage_rejected(self):
        import tempfile,json
        from pathlib import Path
        from prepare_smc_shoulder_control import audit_stages
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,summary,stages=self.fixture(root)
            stages[1]['parents']=[1,0]
            (root/'stages.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in stages))
            with self.assertRaisesRegex(ValueError,'lineage'):audit_stages(root,cfg,summary,2.)
    def test_native_reappearance_is_reported_without_independence_claim(self):
        import tempfile,json
        from pathlib import Path
        from prepare_smc_shoulder_control import audit_stages
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,summary,stages=self.fixture(root)
            populations=[json.loads(line) for line in (root/'populations.jsonl').read_text().splitlines()]
            for p in populations[0]['particles']:p['q']=.7
            populations[1]['particles'][0]['q']=.4
            populations[1]['particles'][1]['q']=.7
            summary['final_particles']=populations[1]['particles']
            (root/'populations.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in populations))
            a=audit_stages(root,cfg,summary,2.)
            self.assertEqual(a['first_native_stage'],1)
            self.assertEqual(a['native_path'][0]['native_endpoints'],0)
            self.assertEqual(a['native_path'][1]['native_families'],1)

if __name__=='__main__': unittest.main()
