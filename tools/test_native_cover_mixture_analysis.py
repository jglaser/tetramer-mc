#!/usr/bin/env python3
"""Independent support, normalization, and streaming controls for nested covers."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import analyze_native_region_reference as analysis


def pose(x):
    return {'position':x,'orientation':[1.,0.,0.,0.]}


def cover(scale=1.):
    angle=scale*math.pi
    return {'reference':pose([0.,0.,0.]),'centroid':[0.,0.,0.],
            'ball_radius':scale,'angle_cap':angle,
            'volume':4*scale**3/3*(angle-math.sin(angle))}


def model():
    return {'scales':[.5,1.],'weights':[.5,.5],'covers':[cover(.5),cover()]}


def metric():
    return {'native_poses':[pose([0.,0.,0.])],
            'rigid_members':[pose([0.,0.,0.])],
            'member_error_scale':1.,'angle_error_scale_deg':180.}


def qmul(a,b):
    v=analysis.cross(a[1:],b[1:])
    return [a[0]*b[0]-sum(x*y for x,y in zip(a[1:],b[1:]))]+[
        a[0]*b[i+1]+b[0]*a[i+1]+v[i] for i in range(3)]


class MixtureAnalysisControls(unittest.TestCase):
    def test_annular_normalization_and_known_z0_mass(self):
        mixture=model();volumes=[c['volume'] for c in mixture['covers']]
        previous=0.;proposal_mass=0.;target_mass=0.;second=0.
        for j,volume in enumerate(volumes):
            probe=pose([.25 if j==0 else .75,0.,0.])
            density=math.exp(analysis.proposal_log_density(mixture,probe)[0])
            exact=sum(.5/v for v in volumes[j:])
            self.assertAlmostEqual(density,exact)
            shell=volume-previous
            proposal_mass+=shell*density
            target_mass+=shell*density/density
            second+=shell/density
            previous=volume
        self.assertAlmostEqual(proposal_mass,1.)
        self.assertAlmostEqual(target_mass,4*math.pi/3)
        self.assertGreater(second,target_mass**2)
        analysis.proposal_model({'cover_mixture':mixture},cover())
        broken=copy.deepcopy(mixture);broken['covers'][0]['volume']*=1.001
        with self.assertRaises(AssertionError):
            analysis.proposal_model({'cover_mixture':broken},cover())

    def test_no_unnormalized_support_guard_and_tiny_cap(self):
        mixture=model()
        self.assertEqual(analysis.proposal_log_density(mixture,pose([1.+1e-13,0.,0.]))[0],-math.inf)
        inside=analysis.proposal_log_density(mixture,pose([.5-1e-13,0.,0.]))[1]
        outside=analysis.proposal_log_density(mixture,pose([.5+1e-13,0.,0.]))[1]
        self.assertEqual(inside,[True,True]);self.assertEqual(outside,[False,True])
        tiny=copy.deepcopy(mixture)
        for c in tiny['covers']:c['angle_cap']*=1e-9
        theta=tiny['covers'][0]['angle_cap']*1.001
        p=pose([0.,0.,0.]);p['orientation']=[math.cos(theta/2),math.sin(theta/2),0.,0.]
        self.assertEqual(analysis.proposal_log_density(tiny,p)[1],[False,True])

    def test_off_center_and_global_frame_invariance(self):
        mixture=model();reference={'position':[2.,-3.,7.],
            'orientation':[math.cos(.3),math.sin(.3),0.,0.]};centroid=[3.,-2.,4.]
        for c in mixture['covers']:c.update(reference=reference,centroid=centroid)
        moving={'orientation':qmul(reference['orientation'],[math.cos(.1),0.,math.sin(.1),0.])}
        rc=analysis.rotate(moving['orientation'],centroid);r0c=analysis.rotate(reference['orientation'],centroid)
        moving['position']=[reference['position'][i]-rc[i]+r0c[i]+[.1,.2,.05][i] for i in range(3)]
        before=analysis.proposal_log_density(mixture,moving)
        world={'position':[-7.,6.,3.],'orientation':[math.cos(.4),0.,0.,math.sin(.4)]}
        def transform(p):
            rotated=analysis.rotate(world['orientation'],p['position'])
            return {'position':[rotated[i]+world['position'][i] for i in range(3)],
                    'orientation':qmul(world['orientation'],p['orientation'])}
        moved=copy.deepcopy(mixture)
        for c in moved['covers']:c['reference']=transform(reference)
        after=analysis.proposal_log_density(moved,transform(moving))
        self.assertAlmostEqual(before[0],after[0]);self.assertEqual(before[1],after[1])
        reversed_pose=copy.deepcopy(moving);reversed_pose['orientation']=[-x for x in moving['orientation']]
        self.assertEqual(before,analysis.proposal_log_density(mixture,reversed_pose))

    def write_fixture(self,path):
        mixture=model();rows=[]
        for i,(radius,component) in enumerate([(.25,0),(.75,1)]):
            p=pose([radius,0.,0.]);g=analysis.proposal_log_density(mixture,p)[0]
            rows.append({'draw':i,'q':radius,'pose':p,'proposal_component':component,
                'log_proposal_density':g,'log_hard_weight':-g,'log_boltzmann_mean':0.,
                'log_importance_weight':-g,'cloud_log_weights':[0.,0.],
                'cloud_overlap_counts':[0,0],'cloud_raw_points':[0,0],
                'lower_volume':0.,'upper_volume':1.})
        text=''.join(json.dumps(row)+'\n' for row in rows)
        (path/'samples.jsonl').write_text(text)
        full=[math.exp(row['log_importance_weight']) for row in rows]
        def moment(values):
            return {'nonzero':len(values),'log_sum_weights':math.log(sum(values)),
                    'log_sum_squared_weights':math.log(sum(x*x for x in values))}
        summary={'samples':2,'samples_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'q_rejected':0,'capture_rejected':0,'hard_rejected':0,'cpu_seconds':0,
            'wall_seconds':0,'raw_cloud_points':0,'native':moment(full),
            'native_core':moment(full),'native_shell':{'nonzero':0},
            'hard_native':moment(full),'hard_native_core':moment(full),'hard_native_shell':{'nonzero':0},
            'physical_hard_log_cross_sum':math.log(sum(x*x for x in full))}
        manifest={'schema':2,'samples':2,'activity':0.,'lambda':1.,'cloud_replicates':2,
            'depletant_radius':.5,'config_sha256':'test','shape_sha256':'test',
            'metric':metric(),'cover_mixture':mixture}
        (path/'summary.json').write_text(json.dumps(summary));(path/'manifest.json').write_text(json.dumps(manifest))
        (path/'cover.json').write_text(json.dumps(cover()));(path/'provenance').mkdir()
        (path/'provenance/config.json').write_text(json.dumps({'capture_center':[0.,0.,0.],'capture_radius':10.}))
        return rows,full

    def test_stream_reconstructs_full_density_and_hard_covariance(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);rows,weights=self.write_fixture(path)
            result=analysis.analyze(str(path))
            self.assertAlmostEqual(math.exp(result['hard_regions']['native']['log_normalizer']),sum(weights)/2)
            self.assertNotAlmostEqual(sum(weights)/2,cover()['volume'])
            self.assertAlmostEqual(result['paired_statistics']['observed_SE_log_depletion_enhancement'],0.,places=7)
            rows[0]['q']=.3  # Proposal support must not redefine the original metric.
            (path/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaises(AssertionError):analysis.analyze(str(path))


if __name__=='__main__':unittest.main()
