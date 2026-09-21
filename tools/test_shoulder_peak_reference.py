"""Source-law preservation and fixed independent inner-shoulder references."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analyze_shoulder_peak_reference import (WINDOW, RADII, original_weights,
    combine_shells, finish, validate_plan, verify_reference_manifest, ensure_audits)
from prepare_cayley_rms_cover import write,sha
from shoulder_peak_moments import ShoulderMoments


class ShoulderReferenceTests(unittest.TestCase):
    def test_guided_original_density_and_outer_q_zeros(self):
        # Deliberately nonuniform proposal: inverse density must survive the
        # later local mask, including a very large outer-shoulder contribution.
        n=6; original=[]; accumulator=ShoulderMoments()
        for q,logg,clouds in [(1.05,math.log(.1),[0.,math.log(3.)]),
                              (1.2,math.log(.8),[10.,11.]),
                              (1.1,math.log(.3),[1.,2.]),
                              (1.08,math.log(.2),[1.,1.])]:
            weight=float(np.logaddexp(*clouds))-math.log(2)-logg
            row=dict(q=q,log_importance_weight=weight,log_hard_weight=-logg,
                log_proposal_density=logg,cloud_log_weights=clouds)
            before=copy.deepcopy(row);w,h,p=original_weights(row,'guided')
            self.assertEqual(row,before);accumulator.add(.1,w,h,p)
            original.append(0. if w is None else math.exp(w))
        for _ in range(2):
            self.assertEqual(original_weights({'zero':'hard','q':1.06},'guided'),(None,None,None));accumulator.add();original.append(0.)
        result=accumulator.report()['physical']['full']
        self.assertEqual(result['draws'],n);self.assertEqual(result['nonzero'],2)
        self.assertAlmostEqual(math.exp(result['logQ']),np.mean(original))
        self.assertAlmostEqual(math.exp(result['log_variance_of_mean']),np.var(original,ddof=1)/n)

    def test_latent_keeps_original_jacobian_and_strict_window(self):
        row=dict(q=1.07,log_hard_weight=-3.,log_importance_weight=-1.,
                 capture_valid=True,hard_valid=True,region_valid=True,clouds=[{'log_weight':2.},{'log_weight':2.}])
        self.assertEqual(original_weights(row,'latent'),(-1.,-3.,[-1.,-1.]))
        for q in (1.,1.1):
            candidate=dict(row,q=q)
            self.assertEqual(original_weights(candidate,'latent'),(None,None,None))
        with self.assertRaises(ValueError):original_weights(dict(row,hard_valid=False),'latent')
        with self.assertRaises(AssertionError):original_weights(dict(row,log_importance_weight=2.),'latent')

    def test_disjoint_sum_excludes_nested_interior(self):
        campaigns=[];selected_arrays=[]
        for index,radius in enumerate(RADII):
            populations=[];aggregate=ShoulderMoments()
            selected=np.array([2.,0.,3.,5.])*(index+1);selected_arrays.append(selected)
            for pop in range(2):
                local=ShoulderMoments()
                for w in selected[2*pop:2*pop+2]:
                    value=math.log(w) if w else None
                    local.add(.1 if index==0 else .4,value,0. if w else None,[value]*2 if w else None)
                # A second independent ball contains earlier-region mass;
                # its two huge inner observations MUST NOT enter shell sum.
                if index:local.add(.1,math.log(1000.),0.,[math.log(1000.)]*2)
                aggregate.merge(local);populations.append(dict(seed=100+10*index+pop,samples=local.values['physical']['full'].count,**local.report()))
            campaigns.append(dict(root=f'run{index}',radius_A=radius,populations=populations,**finish(aggregate,populations)))
        result=combine_shells(campaigns)['physical']['ball0p5']
        arrays=[selected_arrays[0],np.r_[selected_arrays[1],0.,0.]]
        expected=sum(a.mean() for a in arrays);variance=sum(a.var(ddof=1)/len(a) for a in arrays)
        self.assertAlmostEqual(math.exp(result['logQ']),expected)
        self.assertAlmostEqual(math.exp(result['log_variance_of_mean']),variance)
        self.assertNotAlmostEqual(expected,sum(math.exp(c['physical']['full']['logQ']) for c in campaigns))
        campaigns[1]['populations'][0]['seed']=campaigns[0]['populations'][0]['seed']
        with self.assertRaises(ValueError):combine_shells(campaigns)

    def test_frozen_runtime_law_binds_executable_clouds_seeds_and_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'provenance';archive.mkdir()
            cfg=dict(metadata=dict(native_poses=[],rigid_members=[],member_error_scale=2.,angle_error_scale_deg=15.),
                fixed_poses=[{'A':0},{'B':1}],capture_center=[0.,0.,0.],capture_radius=18.,depletant_radius=1.5,reservoir_density=.035)
            model={'test':'frozen'};region=dict(minimum_original_q=1.,maximum_original_q=1.1,minimum_original_q_inclusive=False,maximum_original_q_inclusive=False,
                gaussian_chart=model,mahalanobis_radius=.25,minimum_mahalanobis_radius=0.,physical_fixed_neighbors=cfg['fixed_poses'],fixed_neighbor=cfg['fixed_poses'][0],physical_metric=cfg['metadata'])
            write(archive/'config.json',cfg);write(archive/'region.json',region)
            (archive/'latent-region-normalizer').write_bytes(b'frozen executable');(archive/'analyze_latent_region.py').write_bytes(b'frozen auditor')
            hashes={p.name:sha(p) for p in archive.iterdir()}
            protocol=dict(executable_sha256=hashes['latent-region-normalizer'],lambda_ratio=64,cloud_replicates=2,archived_sha256=hashes)
            master=dict(region_sha256=hashes['region.json'],archive_sha256=hashes,lambda_ratio=64,cloud_replicates=2,jobs=[dict(seed=7,samples=100)])
            declared=dict(region_sha256=hashes['region.json'],radius_A=.25,populations=1,seeds=[7],samples_per_population=100)
            write(root/'manifest.json',master)
            verify_reference_manifest(root,declared,protocol,cfg,model)
            for field,value in [('executable_sha256','changed'),('lambda_ratio',32),('cloud_replicates',1)]:
                changed=dict(protocol,**{field:value})
                with self.assertRaises(ValueError):verify_reference_manifest(root,declared,changed,cfg,model)
            with self.assertRaises(ValueError):verify_reference_manifest(root,dict(declared,seeds=[8]),protocol,cfg,model)
            with self.assertRaises(ValueError):verify_reference_manifest(root,dict(declared,radius_A=.5),protocol,cfg,model)

    def test_audit_failure_drains_other_child_and_records_both(self):
        class Child:
            def __init__(self,pid,code):self.pid=pid;self.code=code;self.waited=False
            def wait(self):self.waited=True;return self.code
        children=[Child(11,1),Child(12,0)]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out';out.mkdir();requests=[({'radius_A':r},Path(tmp)/f'root{i}',{}) for i,r in enumerate(RADII)]
            with patch('analyze_shoulder_peak_reference.subprocess.Popen',side_effect=children):
                with self.assertRaises(ValueError):ensure_audits(requests,out,True)
            import json
            state=json.loads((out/'runner-state.json').read_text())
            self.assertTrue(all(c.waited for c in children));self.assertEqual(state['phase'],'original_audits_failed')
            self.assertEqual([p['exit_code'] for p in state['processes']],[1,0]);self.assertTrue(all(p['terminal'] for p in state['processes']))


if __name__=='__main__':unittest.main()
