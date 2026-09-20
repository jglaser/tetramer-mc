#!/usr/bin/env python3
"""Independent hybrid-density, frame, and fixed-N streaming controls."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import analyze_native_region_reference as analysis
from test_native_cover_mixture_analysis import cover,metric,model,pose,qmul


def raw_model():
    return dict(coordinate_convention='anchor-body-relative',angular_length=1.,shape_sha256='a'*64,
        anchors=[dict(position=[0.,0.,0.],rotation=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])],
        means=[[0.]*6],covariances=[[[float(i==j) for j in range(6)] for i in range(6)]],weights=[1.])


def config():
    return dict(capture_center=[0.,0.,0.],capture_radius=.3,fixed_poses=[pose([0.,0.,0.])])


def description(cfg=None):
    cfg=cfg or config()
    return dict(model_sha256='pending',weight=.75,uniform_probability=.05,anchor_index=0,
        anchor_pose=copy.deepcopy(cfg['fixed_poses'][0]),cube_lengths=[2*cfg['capture_radius']]*3,
        capture_center=list(cfg['capture_center']))


def analytic_identity_density(position):
    # Unit-covariance Gaussian at identity orientation, ell=1:
    # phi_6(t,0)*pi^2 = exp(-|t|²/2)/(8*pi).
    g=math.exp(-sum(x*x for x in position)/2)/(8*math.pi)
    u=.05/.6**3 if all(-.3<=x<.3 for x in position) else 0.
    radius=math.hypot(*position)
    covers=sum(.5/c['volume'] for c in model()['covers'] if radius<=c['ball_radius'])
    return .25*covers+.75*(u+.95*g)


def write_fixture(path,all_zero=False):
    cfg=config();desc=description(cfg);raw=raw_model()
    (path/'provenance').mkdir()
    encoded=json.dumps(raw).encode();(path/'provenance/guide-model.json').write_bytes(encoded)
    desc['model_sha256']=hashlib.sha256(encoded).hexdigest()
    samples=[([.1,0.,0.],'cover',None,0,None,None),
             ([.2,0.,0.],'guide','learned',None,0,None),
             ([.25,.25,.25],'guide','uniform',None,None,'capture'),
             ([.4,0.,0.],'guide','learned',None,0,'capture'),
             ([2.,0.,0.],'guide','learned',None,0,'q'),
             ([.6,0.,0.],'cover',None,1,None,'capture'),
             ([.15,0.,0.],'guide','uniform',None,None,'hard')]
    rows=[]
    for i,(position,family,branch,component,guide_component,zero) in enumerate(samples):
        g=math.log(analytic_identity_density(position))
        row=dict(draw=i,pose=pose(position),q=math.hypot(*position),proposal_family=family,
            guide_branch=branch,proposal_component=component,guide_component=guide_component,log_proposal_density=g)
        if all_zero and zero is None:zero='hard'
        if zero is not None:row['zero']=zero
        else:row.update(log_hard_weight=-g,log_boltzmann_mean=0.,log_importance_weight=-g,
            cloud_log_weights=[0.,0.],cloud_overlap_counts=[0,0],cloud_raw_points=[0,0],lower_volume=0.,upper_volume=1.)
        rows.append(row)
    text=''.join(json.dumps(row)+'\n' for row in rows);(path/'samples.jsonl').write_text(text)
    full=[math.exp(r['log_importance_weight']) for r in rows if 'zero' not in r]
    def moments(weights):
        if not weights:return {'nonzero':0,'log_sum_weights':None,'log_sum_squared_weights':None}
        return dict(nonzero=len(weights),log_sum_weights=math.log(sum(weights)),log_sum_squared_weights=math.log(sum(x*x for x in weights)))
    summary=dict(samples=len(rows),samples_sha256=hashlib.sha256(text.encode()).hexdigest(),
        q_rejected=1,capture_rejected=3,hard_rejected=3 if all_zero else 1,cpu_seconds=0.,wall_seconds=0.,raw_cloud_points=0,
        native=moments(full),native_core=moments(full),native_shell=moments([]),
        hard_native=moments(full),hard_native_core=moments(full),hard_native_shell=moments([]),
        physical_hard_log_cross_sum=math.log(sum(x*x for x in full)) if full else None,
        component_draws=[1,1],family_draws={'cover':2,'guide':5},guide_component_draws=[3],guide_uniform_draws=2)
    manifest=dict(schema=3,samples=len(rows),activity=0.,lambda_=1.,cloud_replicates=2,depletant_radius=.5,
        config_sha256='test',shape_sha256='a'*64,metric=metric(),cover_mixture=model(),guide=desc)
    manifest['lambda']=manifest.pop('lambda_')
    for name,value in [('summary.json',summary),('manifest.json',manifest),('cover.json',cover()),('provenance/config.json',cfg)]:
        (path/name).write_text(json.dumps(value))
    return rows,full


class HybridAnalysisControls(unittest.TestCase):
    def test_exact_both_families_and_cube_boundaries(self):
        guide=analysis.GaussianGuide(description(),raw_model(),config(),'a'*64)
        for point in ([0.,0.,0.],[.25,0.,0.],[.75,0.,0.],[2.,0.,0.]):
            actual=math.exp(analysis.hybrid_log_density(model(),guide,pose(point))[0])
            self.assertAlmostEqual(actual,analytic_identity_density(point),places=14)
        for x,inside in [(-.3,True),(.3,False),(.3-1e-14,True),(-.3-1e-14,False)]:
            self.assertEqual(analysis.hybrid_log_density(model(),guide,pose([x,0.,0.]))[2],inside)
        self.assertTrue(math.isfinite(analysis.hybrid_log_density(model(),guide,pose([2.,0.,0.]))[0]))

    def test_exact_uniform_guide_normalization_and_z0_integral(self):
        cfg=config();cfg['capture_radius']=2.
        desc=description(cfg);desc['uniform_probability']=1.
        guide=analysis.GaussianGuide(desc,raw_model(),cfg,'a'*64)
        single={'scales':[1.],'weights':[1.],'covers':[cover()]}
        volume=4*math.pi/3;cube=4**3
        inside=math.exp(analysis.hybrid_log_density(single,guide,pose([0.,0.,0.]))[0])
        outside=math.exp(analysis.hybrid_log_density(single,guide,pose([1.5,0.,0.]))[0])
        self.assertAlmostEqual(inside,.25/volume+.75/cube)
        self.assertAlmostEqual(outside,.75/cube)
        self.assertAlmostEqual(inside*volume+outside*(cube-volume),1.)
        self.assertAlmostEqual(inside*volume/inside,volume)

    def test_cayley_haar_scale_and_rotated_anchor(self):
        cfg=config();cfg['capture_center']=[4.,-3.,2.];cfg['capture_radius']=2.
        anchor=pose([3.,-2.,1.]);anchor['orientation']=[math.cos(.35),0.,math.sin(.35),0.]
        cfg['fixed_poses']=[anchor];desc=description(cfg);raw=raw_model();raw['angular_length']=2.
        raw['anchors'][0]['position']=[.3,.1,-.4]
        raw['anchors'][0]['rotation']=[[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]]
        # The rotational covariance is anisotropic and there is a t/c cross term.
        covariance=raw['covariances'][0];covariance[3][3]=2.;covariance[0][3]=covariance[3][0]=.25
        guide=analysis.GaussianGuide(desc,raw,cfg,'a'*64)
        c=[.4,-.2,.1];t=[.1,-.3,.2];rotation=analysis.normalized_quaternion([1.]+c)
        moved=analysis.rotate(anchor['orientation'],[t[i]+raw['anchors'][0]['position'][i] for i in range(3)])
        p=dict(position=[anchor['position'][i]+moved[i] for i in range(3)],
            orientation=qmul(anchor['orientation'],qmul(rotation,[0.,1.,0.,0.])))
        # Independent analytic inverse/determinant of the only coupled 2x2 block.
        x=t+[2*v for v in c];det=2-.25**2
        quadratic=(2*x[0]**2-.5*x[0]*x[3]+x[3]**2)/det+sum(x[i]**2 for i in (1,2,4,5))
        gaussian=math.exp(-quadratic/2)/(2*math.pi)**3/math.sqrt(det)*8*math.pi**2*(1+sum(v*v for v in c))**2
        expected=.05/64+.95*gaussian
        self.assertAlmostEqual(math.exp(guide.log_density(p)[0]),expected,places=13)
        shifted=copy.deepcopy(cfg);offset=[17.,-8.,5.]
        shifted['capture_center']=[a+b for a,b in zip(cfg['capture_center'],offset)]
        shifted['fixed_poses'][0]['position']=[a+b for a,b in zip(anchor['position'],offset)]
        newpose=copy.deepcopy(p);newpose['position']=[a+b for a,b in zip(p['position'],offset)]
        shifted_guide=analysis.GaussianGuide(description(shifted),raw,shifted,'a'*64)
        self.assertAlmostEqual(guide.log_density(p)[0],shifted_guide.log_density(newpose)[0],places=12)
        negative=copy.deepcopy(p);negative['orientation']=[-x for x in p['orientation']]
        self.assertEqual(guide.log_density(p)[0],guide.log_density(negative)[0])

    def test_duplicate_components_and_single_anchor_not_neighbor_average(self):
        cfg=config();cfg['fixed_poses'].append(pose([9.,0.,0.]))
        raw=raw_model();duplicate=copy.deepcopy(raw)
        for key in ('anchors','means','covariances'):duplicate[key]*=2
        duplicate['weights']=[.2,.8]
        a=analysis.GaussianGuide(description(cfg),raw,cfg,'a'*64)
        b=analysis.GaussianGuide(description(cfg),duplicate,cfg,'a'*64)
        self.assertAlmostEqual(a.log_density(pose([.1,0.,0.]))[0],b.log_density(pose([.1,0.,0.]))[0])
        # Additional physical neighbors do not enter the chosen-anchor guide density.
        one=analysis.GaussianGuide(description(),raw,config(),'a'*64)
        self.assertAlmostEqual(a.log_density(pose([.1,0.,0.]))[0],one.log_density(pose([.1,0.,0.]))[0])
        wrong=description(cfg);wrong['anchor_index']=1
        with self.assertRaises(AssertionError):analysis.GaussianGuide(wrong,raw,cfg,'a'*64)

    def test_invalid_guide_metadata_and_covariance_are_rejected(self):
        for field,value in [('weight',0.),('weight',1.),('uniform_probability',0.),
                            ('uniform_probability',1.1),('cube_lengths',[1.,1.,1.]),
                            ('capture_center',[1.,0.,0.])]:
            with self.subTest(field=field,value=value):
                desc=description();desc[field]=value
                with self.assertRaises(AssertionError):analysis.GaussianGuide(desc,raw_model(),config(),'a'*64)
        raw=raw_model();raw['covariances'][0][0][0]=0.
        with self.assertRaises(AssertionError):analysis.GaussianGuide(description(),raw,config(),'a'*64)

    def test_stream_keeps_zero_draws_and_checks_hard_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);rows,weights=write_fixture(path)
            result=analysis.analyze(path)
            self.assertEqual(result['samples'],7);self.assertEqual(result['counts']['valid'],2)
            self.assertAlmostEqual(math.exp(result['hard_regions']['native']['log_normalizer']),sum(weights)/7)
            self.assertEqual(result['proposal_family_counts'],{'cover':2,'guide':5})
            self.assertEqual(result['guide_component_counts'],[3]);self.assertEqual(result['guide_uniform_count'],2)
            self.assertAlmostEqual(result['paired_statistics']['observed_SE_log_depletion_enhancement'],0.,places=7)
            rows[0]['log_hard_weight']+=.1
            (path/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(AssertionError):analysis.analyze(path)

    def test_stream_rejects_family_only_density_and_invalid_branch(self):
        for mutation in ('density','branch','hash'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                path=Path(directory);rows,_=write_fixture(path)
                if mutation=='density':rows[0]['log_proposal_density']=analysis.proposal_log_density(model(),rows[0]['pose'])[0]
                elif mutation=='branch':rows[4]['guide_branch']='uniform';rows[4]['guide_component']=None
                else:(path/'provenance/guide-model.json').write_text('{}')
                (path/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(AssertionError):analysis.analyze(path)

    def test_all_zero_hybrid_and_pooling_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);write_fixture(path,all_zero=True);result=analysis.analyze(path)
            self.assertEqual(result['samples'],7);self.assertIsNone(result['regions']['native']['log_normalizer'])
            analysis.require_matching_targets([result,copy.deepcopy(result)])
            changed=copy.deepcopy(result);changed['guide']['weight']=.5
            with self.assertRaises(AssertionError):analysis.require_matching_targets([result,changed])


if __name__=='__main__':unittest.main()
