"""Frozen chart-origin controls; no protein geometry, sampling or fit."""
import copy
import json
import math
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal
from source_guide_reference import SourceDensity, cayley, compose, pose, pose_error, relative


def specification():
    lower=np.diag([.2,.3,.4,.1,.2,.3]);lower[3,0]=.07;lower[5,1]=-.08
    mean=np.array([.01,-.03,.02,.04,-.02,.01])
    return dict(angular_length=2.,covariance=(lower@lower.T).tolist(),
        explicit_gaussian=dict(schema='source-gaussian-v1',mean=mean.tolist(),provenance='synthetic')),lower,mean


def center(value):
    return dict(schema='source-chart-center-v1',frame='saved-spherical-center',pose=value,
                provenance='synthetic independent world pose; no native reference')


class SourceChartCenterTests(unittest.TestCase):
    def test_own_world_center_full_covariance_decode_and_haar_jacobian(self):
        spec,lower,mean=specification()
        source=pose([0.,1.,2.],Rotation.from_rotvec([.1,.3,-.4]).as_matrix())
        anchor=pose([1.,-2.,.3],Rotation.from_rotvec([.3,-.2,.4]).as_matrix())
        chosen=pose([-.4,.7,1.2],Rotation.from_rotvec([-.3,.4,.2]).as_matrix())
        spec['chart_center']=center(chosen)
        reference=SourceDensity(spec,source,anchor);local=relative(chosen,anchor)
        for z in (np.zeros(6),np.array([.2,-.3,.4,-.5,.6,-.7])):
            x=mean+lower@z;u=x[3:]/2.
            wanted=compose(anchor,pose(np.asarray(local['position'])+x[:3],
                cayley(u)@Rotation.from_quat(np.asarray(local['orientation'])[[1,2,3,0]]).as_matrix()))
            pose_error(reference.decode(z),wanted)
            logJ=np.log(np.diag(lower)).sum()-3*math.log(2.)-2*math.log(math.pi)-2*math.log1p(u@u)
            self.assertAlmostEqual(reference.evaluate(wanted),multivariate_normal.logpdf(z,mean=np.zeros(6),cov=np.eye(6))-logJ,places=10)
        shift=pose([9.,-3.,4.],Rotation.from_rotvec([.5,-.1,.3]).as_matrix())
        moved=copy.deepcopy(spec);moved['chart_center']['pose']=compose(shift,chosen)
        transformed=SourceDensity(moved,compose(shift,source),compose(shift,anchor))
        value=reference.decode([.2,-.3,.4,-.5,.6,-.7])
        self.assertAlmostEqual(reference.evaluate(value),transformed.evaluate(compose(shift,value)),places=10)

    def test_default_and_explicit_original_center_keep_reference_and_density(self):
        spec,_,_=specification();source=pose([.2,-.3,.4],Rotation.from_rotvec([.1,.2,-.3]).as_matrix())
        anchor=pose([1.,2.,3.],Rotation.from_rotvec([-.2,.5,.3]).as_matrix())
        frozen=json.dumps(dict(spec=spec,source=source,anchor=anchor),sort_keys=True)
        old=SourceDensity(spec,source,anchor);same=copy.deepcopy(spec);same['chart_center']=center(source)
        new=SourceDensity(same,source,anchor)
        for z in (np.zeros(6),np.arange(6)*.2):
            self.assertEqual(old.decode(z),new.decode(z))
            self.assertEqual(old.evaluate(old.decode(z)),new.evaluate(new.decode(z)))
        self.assertEqual(json.dumps(dict(spec=spec,source=source,anchor=anchor),sort_keys=True),frozen)
        same['chart_center']=None
        self.assertEqual(SourceDensity(same,source,anchor).decode(np.zeros(6)),old.decode(np.zeros(6)))

    def test_new_center_resolves_original_pi_seam(self):
        spec,_,_=specification();spec['explicit_gaussian']['mean']=[0.]*6
        source=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]);anchor=copy.deepcopy(source)
        chosen=dict(position=[.1,-.2,.3],orientation=[0.,1.,0.,0.])
        self.assertEqual(SourceDensity(spec,source,anchor).evaluate(chosen),-math.inf)
        spec['chart_center']=center(chosen);reference=SourceDensity(spec,source,anchor)
        pose_error(reference.decode(np.zeros(6)),chosen)
        self.assertTrue(math.isfinite(reference.evaluate(chosen)))
        self.assertEqual(source,dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.]))

    def test_invalid_center_schema_frame_provenance_pose_rejected(self):
        spec,_,_=specification();origin=dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])
        spec['chart_center']=center(origin);cases=[]
        for field in ('schema','frame','pose','provenance'):
            bad=copy.deepcopy(spec);del bad['chart_center'][field];cases.append(bad)
        for field,value in [('schema','unversioned'),('frame','viewer-origin'),('provenance',' '),('extra',1)]:
            bad=copy.deepcopy(spec);bad['chart_center'][field]=value;cases.append(bad)
        for field,value in [('position',[math.nan,0.,0.]),('position',[0.,0.]),
                            ('orientation',[0.]*4),('orientation',[1.+2e-9,0.,0.,0.])]:
            bad=copy.deepcopy(spec);bad['chart_center']['pose'][field]=value;cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):SourceDensity(bad,origin,origin)


if __name__=='__main__':unittest.main()
