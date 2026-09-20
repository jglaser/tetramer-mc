#!/usr/bin/env python3
"""Original-q window support, complete-cover, and fixed-N audit controls."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import analyze_native_region_reference as analysis
from test_native_cover_mixture_analysis import metric,pose
from test_native_hybrid_analysis import raw_model


WINDOW=dict(minimum=1.,maximum=2.,lower_inclusive=False,upper_inclusive=False)


def save(path,value):
    raw=(json.dumps(value)+'\n').encode();path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def complete_cover(original,window=WINDOW):
    # Analytic single centered member: translation ball, complete Haar rotations.
    cm=analysis.window_cover_metric(original,window)
    radius=cm['member_error_scale'];angle=math.radians(cm['angle_error_scale_deg'])
    return dict(reference=pose([0.,0.,0.]),centroid=[0.]*3,
        member_covariance=[[0.]*3 for _ in range(3)],moment_trace=0.,
        lambda_max_upper=4096*math.ulp(1.),l_lower=0.,ball_radius=radius,
        nominal_angle_cap=angle,angle_cap=angle,
        volume=4*radius**3/3*analysis.theta_minus_sin(angle))


def write_fixture(path,guided=False,mixture=False,all_zero=False):
    (path/'provenance').mkdir()
    original=metric();cfg=dict(metadata=original,capture_center=[0.,0.,0.],capture_radius=1.4,
        fixed_poses=[pose([0.,20.,0.]),pose([0.,-20.,0.])],depletant_radius=.5,reservoir_density=.5)
    shape_sha=save(path/'provenance/shape.json',dict(atoms=[dict(center=[0.]*3,radius=.01)]))
    cfg_sha=save(path/'provenance/config.json',cfg)
    cover=complete_cover(original);scales=[.5,1.] if mixture else [1.]
    covers=[]
    for scale in scales:
        c=copy.deepcopy(cover);c['ball_radius']*=scale;c['angle_cap']*=scale
        c['volume']=4*c['ball_radius']**3/3*analysis.theta_minus_sin(c['angle_cap']);covers.append(c)
    model=dict(scales=scales,weights=[1/len(scales)]*len(scales),covers=covers)
    manifest=dict(schema=4,samples=6,seed=729137,activity=.5,lambda_=2.,cloud_replicates=2,
        depletant_radius=.5,config_sha256=cfg_sha,shape_sha256=shape_sha,metric=original,
        q_window=WINDOW,cover_metric=analysis.window_cover_metric(original,WINDOW),cover_mixture=model)
    manifest['lambda']=manifest.pop('lambda_')
    guide=None
    if guided:
        raw=raw_model();raw['shape_sha256']=shape_sha
        raw['anchors'][0]['position']=[0.,-20.,0.]
        model_sha=save(path/'provenance/guide-model.json',raw)
        desc=dict(model_sha256=model_sha,weight=.75,uniform_probability=.05,anchor_index=0,
            anchor_pose=cfg['fixed_poses'][0],cube_lengths=[2.8]*3,capture_center=[0.]*3)
        manifest['guide']=desc;guide=analysis.GaussianGuide(desc,raw,cfg,shape_sha)
    rows=[]
    for i,(q,zero) in enumerate([(1.,'q'),(2.,'q'),(1.25,None),(1.5,'capture'),(1.3,'hard'),(.5,'q')]):
        p=pose([q,0.,0.]);component=0 if q<=1 else len(scales)-1
        g=(analysis.hybrid_log_density(model,guide,p)[0] if guide else analysis.proposal_log_density(model,p)[0])
        row=dict(draw=i,pose=p,q=q,proposal_family='cover',proposal_component=component,
            log_proposal_density=g,guide_component=None,guide_branch=None)
        if guided and i==2:row.update(proposal_family='guide',proposal_component=None,guide_component=0,guide_branch='learned')
        if all_zero and zero is None:zero='hard'
        if zero is not None:row['zero']=zero
        else:
            logs=[.5*.1+k*math.log1p(.5/2) for k in (1,2)]
            mean=analysis.logadd(*logs)-math.log(2)
            row.update(log_hard_weight=-g,log_boltzmann_mean=mean,log_importance_weight=mean-g,
                cloud_log_weights=logs,cloud_overlap_counts=[1,2],cloud_raw_points=[3,4],lower_volume=.1,upper_volume=1.)
        rows.append(row)
    save_rows(path,rows)
    def moments(field):
        m=analysis.fresh()
        for row in rows:
            if 'zero' not in row:analysis.add(m,row[field])
        return dict(nonzero=m['nonzero'],log_sum_weights=m['sum'] if m['nonzero'] else None,
            log_sum_squared_weights=m['square'] if m['nonzero'] else None)
    full=moments('log_importance_weight');hard=moments('log_hard_weight')
    summary=dict(samples=len(rows),samples_sha256=hashlib.sha256((path/'samples.jsonl').read_bytes()).hexdigest(),
        q_rejected=3,capture_rejected=1,hard_rejected=2 if all_zero else 1,cpu_seconds=0.,wall_seconds=0.,
        raw_cloud_points=0 if all_zero else 7,region=full,region_core=dict(nonzero=0),region_shell=full,
        hard_region=hard,hard_region_core=dict(nonzero=0),hard_region_shell=hard,
        physical_hard_log_cross_sum=None if all_zero else next(r['log_importance_weight']+r['log_hard_weight'] for r in rows if 'zero' not in r),
        component_draws=[sum(r['proposal_component']==i for r in rows) for i in range(len(scales))])
    if guide:summary.update(family_draws=dict(cover=5,guide=1),guide_component_draws=[1],guide_uniform_draws=0)
    for name,value in [('manifest.json',manifest),('summary.json',summary),('cover.json',cover)]:save(path/name,value)
    return rows


def save_rows(path,rows):
    (path/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary_path=path/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text());summary['samples_sha256']=hashlib.sha256((path/'samples.jsonl').read_bytes()).hexdigest()
        save(summary_path,summary)


class QWindowControls(unittest.TestCase):
    def test_exact_open_and_closed_boundaries(self):
        self.assertFalse(analysis.q_in_window(1.,WINDOW));self.assertFalse(analysis.q_in_window(2.,WINDOW))
        self.assertTrue(analysis.q_in_window(math.nextafter(1.,2.),WINDOW))
        self.assertTrue(analysis.q_in_window(math.nextafter(2.,1.),WINDOW))
        closed=dict(WINDOW,lower_inclusive=True,upper_inclusive=True)
        self.assertTrue(analysis.q_in_window(1.,closed));self.assertTrue(analysis.q_in_window(2.,closed))
        for update in [dict(minimum=-1),dict(maximum=1),dict(maximum=math.inf),dict(lower_inclusive=1)]:
            with self.subTest(update=update),self.assertRaises(AssertionError):analysis.validate_q_window(dict(WINDOW,**update))

    def test_scaled_metric_clamps_nominal_and_preserves_original(self):
        original=metric();before=copy.deepcopy(original)
        scaled=analysis.window_cover_metric(original,WINDOW)
        self.assertEqual(scaled['member_error_scale'],2.);self.assertEqual(scaled['angle_error_scale_deg'],180.)
        self.assertEqual(original,before)

    def test_complete_cover_detects_linearly_scaled_geometric_cap(self):
        original=metric();original['rigid_members']=[pose([3*x,3*y,3*z]) for x,y,z in
            [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]]
        cover=complete_cover(metric());cov=[[9.*float(i==j) for j in range(3)] for i in range(3)]
        slack=4096*math.ulp(1.)*28*4;upper=9+slack;lower=27-slack-upper
        cap=2*math.asin(2/(2*math.sqrt(lower)))
        cover.update(member_covariance=cov,moment_trace=27.,lambda_max_upper=upper,l_lower=lower,
            angle_cap=cap,volume=4*8/3*analysis.theta_minus_sin(cap))
        manifest=dict(metric=original,q_window=WINDOW,cover_metric=analysis.window_cover_metric(original,WINDOW))
        analysis.validate_window_cover(manifest,cover,dict(metadata=original))
        oldcap=2*math.asin(1/(2*math.sqrt(lower)))
        self.assertGreater(cap,2*oldcap)
        cover['angle_cap']=2*oldcap
        cover['volume']=4*8/3*analysis.theta_minus_sin(cover['angle_cap'])
        with self.assertRaises(AssertionError):analysis.validate_window_cover(manifest,cover,dict(metadata=original))

    def test_stream_full_density_zeros_and_target_labels(self):
        for guided,mixture in [(False,False),(False,True),(True,False),(True,True)]:
            with self.subTest(guided=guided,mixture=mixture),tempfile.TemporaryDirectory() as d:
                path=Path(d);rows=write_fixture(path,guided,mixture);result=analysis.analyze(path)
                self.assertEqual(result['samples'],6);self.assertEqual(result['counts']['valid'],1)
                self.assertEqual(result['independent_q_and_density_rows'],6)
                self.assertEqual(result['regions']['region_core']['nonzero'],0)
                self.assertNotIn('native',result['regions'])
                expected=next(r['log_importance_weight'] for r in rows if 'zero' not in r)-math.log(6)
                self.assertAlmostEqual(result['regions']['region']['log_normalizer'],expected)
                self.assertEqual(result['regions']['region'],result['regions']['region_shell'])
                analysis.require_matching_targets([result,copy.deepcopy(result)])
                changed=copy.deepcopy(result);changed['physical_signature']['q_window']=dict(WINDOW,upper_inclusive=True)
                with self.assertRaises(AssertionError):analysis.require_matching_targets([result,changed])

    def test_stream_rejects_density_q_geometry_hash_and_zero_corruption(self):
        for mutation in ('density','q','window','cover_metric','capture','pose','config','shape','zero_weight','family'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as d:
                path=Path(d);rows=write_fixture(path,True,True)
                if mutation=='density':rows[2]['log_proposal_density']+=.1
                elif mutation=='q':rows[2]['q']+=.1
                elif mutation=='capture':rows[3]['zero']='hard'
                elif mutation=='pose':rows[2]['pose']['position'][0]+=.1
                elif mutation=='zero_weight':rows[0]['log_importance_weight']=123.
                elif mutation=='family':rows[2]['proposal_family']='cover'
                elif mutation in ('config','shape'):(path/f'provenance/{mutation}.json').write_text('{}')
                else:
                    manifest=json.loads((path/'manifest.json').read_text())
                    if mutation=='window':manifest['q_window']['upper_inclusive']=True
                    else:manifest['cover_metric']['member_error_scale']*=1.1
                    save(path/'manifest.json',manifest)
                save_rows(path,rows)
                with self.assertRaises(AssertionError):analysis.analyze(path)

    def test_all_zero_region_preserves_every_draw(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);write_fixture(path,all_zero=True);result=analysis.analyze(path)
            self.assertEqual(result['regions']['region']['samples'],6)
            self.assertIsNone(result['regions']['region']['log_normalizer'])


if __name__=='__main__':unittest.main()
