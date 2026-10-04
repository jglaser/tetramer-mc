"""Deterministic wall-envelope density and trace controls; no physical draws."""
import copy
import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

import audit_full_vessel_latent as audit
from physical_latent_guide import half_mixture_log_density
from prepare_smc_normalizer_atlas import relative_poses
from test_audit_full_vessel_latent import data
from test_physical_latent_guide import fixture as chart_fixture


def pose(position, rotation=None):
    rotation=np.eye(3) if rotation is None else rotation
    return dict(position=list(position),orientation=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]].tolist())


def witness(shape,wall):
    # Independent fixture declaration: first equal maximum wins.
    index=max(range(len(shape['atoms'])),key=lambda i:shape['atoms'][i]['radius'])
    atom=shape['atoms'][index];radius=wall['radius']-atom['radius']
    return dict(atom_index=index,atom_center=copy.deepcopy(atom['center']),atom_radius=atom['radius'],
        wall_center=copy.deepcopy(wall['center']),wall_radius=wall['radius'],envelope_radius=radius,
        log_volume=math.log(4*math.pi/3)+3*math.log(radius))


def fixture(reciprocal=False,selected=None):
    cfg,manifest,rows,old,latent=data(reciprocal=reciprocal,selected=selected)
    shape=dict(atoms=[dict(center=[0.,0.,0.],radius=.2),dict(center=[1.5,-.5,.25],radius=.5),
                      dict(center=[-.5,.25,0.],radius=.5)])
    wall=dict(center=[4.,-2.,1.],radius=30.)
    manifest.update(schema=8,pre_envelope_schema=7,atomic_wall=wall,
                    vessel_uniform_schema=audit.WALL_ENVELOPE_SCHEMA,vessel_uniform_envelope=witness(shape,wall))
    base=chart_fixture(.5,coupled=True)[0]['gaussian_chart']
    raw=dict(schema='reciprocal-pose-mixture-v1',base_model=base,reciprocal_components=[True]) if reciprocal else base
    vessel=audit.VesselDensity(cfg,manifest,raw,shape=shape)
    return cfg,manifest,rows,vessel,latent,shape,raw,old


def reweight(cfg,manifest,rows,vessel,latent):
    rows=copy.deepcopy(rows);poses=[r['pose'] for r in rows]
    logs,geometry=vessel.evaluate(poses)
    densities=latent.evaluate_many(poses)
    mixed=half_mixture_log_density(logs,[d.log_physical_density for d in densities])
    _,initial=vessel.evaluate([cfg['initial_pose']])
    for i,row in enumerate(rows):
        row['log_vessel_proposal_density']=float(logs[i]);row['log_proposal_density']=float(mixed[i])
        if row['hard_valid']:
            row['log_hard_weight']=-float(mixed[i])
            row['log_importance_weight']=float(logsumexp([c['log_weight'] for c in row['clouds']])-math.log(2)-mixed[i])
        if row['outer_branch']=='vessel':
            p=row['proposal'];anchor=p['anchor_index']-1
            old=float(initial['anchor_log_densities'][anchor,0]);new=float(geometry['anchor_log_densities'][anchor,i])
            p.update(old_log_density=old,new_log_density=new,log_reverse_forward=old-new)
    return rows


class WallEnvelopeVesselTests(unittest.TestCase):
    def test_first_largest_witness_is_reconstructed_and_tampering_rejected(self):
        cfg,manifest,_,_,_,shape,raw,_=fixture()
        self.assertEqual(manifest['vessel_uniform_envelope']['atom_index'],1)
        for key in manifest['vessel_uniform_envelope']:
            bad=copy.deepcopy(manifest);value=bad['vessel_uniform_envelope'][key]
            if isinstance(value,list):value[0]+=.1
            elif key=='atom_index':bad['vessel_uniform_envelope'][key]=2  # same radius, wrong first tie
            else:bad['vessel_uniform_envelope'][key]+=.1
            with self.subTest(key=key),self.assertRaises(ValueError):
                audit.VesselDensity(cfg,bad,raw,shape=shape)
        for key in ('atom_index','wall_radius'):
            bad=copy.deepcopy(manifest);bad['vessel_uniform_envelope'][key]=True
            with self.subTest(boolean=key),self.assertRaises(ValueError):
                audit.VesselDensity(cfg,bad,raw,shape=shape)
        bad=copy.deepcopy(shape);bad['atoms'][2]['radius']=.6
        with self.assertRaises(ValueError):audit.VesselDensity(cfg,manifest,raw,shape=bad)

    def test_explicit_schema_shape_and_positive_radius_required(self):
        cfg,manifest,_,_,_,shape,raw,_=fixture()
        with self.assertRaisesRegex(ValueError,'Archived shape'):
            audit.VesselDensity(cfg,manifest,raw)
        for patch in (dict(schema=7),dict(pre_envelope_schema=6),dict(vessel_uniform_schema='cube')):
            with self.subTest(patch=patch),self.assertRaises(ValueError):
                audit.VesselDensity(cfg,dict(manifest,**patch),raw,shape=shape)
        for radius in (0.,.4,.5,-1.,math.inf,math.nan):
            bad=copy.deepcopy(manifest);bad['atomic_wall']['radius']=radius
            with self.subTest(radius=radius),self.assertRaises(ValueError):
                audit.VesselDensity(cfg,bad,raw,shape=shape)
        bad=copy.deepcopy(shape);bad['atoms'][0]['center'][0]=math.inf
        with self.assertRaises(ValueError):audit.VesselDensity(cfg,manifest,raw,shape=bad)

    def test_offset_rotations_boundary_and_ball_normalization(self):
        shape=dict(atoms=[dict(center=[1.5,-.5,.25],radius=.5)])
        wall=dict(center=[3.,-2.,1.],radius=4.)
        envelope=audit.AtomWallEnvelope(shape,wall,witness(shape,wall));radius=3.5
        for angles in ([0.,0.,0.],[.3,-.4,.2],[2.,-1.,.5]):
            rotation=Rotation.from_rotvec(angles).as_matrix()
            offset=np.asarray(wall['center'])-rotation@np.asarray(shape['atoms'][0]['center'])
            values,inside=envelope.evaluate([pose(offset+[.2,-.3,.4],rotation),
                pose(offset+[radius-1e-9,0.,0.],rotation),pose(offset+[radius+1e-9,0.,0.],rotation)])
            self.assertEqual(inside.tolist(),[True,True,False])
            self.assertAlmostEqual(values[0],-math.log(4*math.pi*radius**3/3),places=13)
            self.assertEqual(values[2],-math.inf)
            def radial(r):
                logq=envelope.evaluate([pose(offset+[r,0.,0.],rotation)])[0][0]
                return 4*math.pi*r*r*math.exp(logq)
            self.assertAlmostEqual(quad(radial,0.,radius,epsabs=1e-12)[0],1.,places=12)
            # Translation y -> t changes only a constant at fixed rotation.
            edges=np.asarray([(offset+np.eye(3)[i])-offset for i in range(3)])
            self.assertAlmostEqual(np.linalg.det(edges),1.,places=13)
        sphere=dict(atoms=[dict(center=[0.,0.,0.],radius=.5)])
        origin_wall=dict(center=[0.,0.,0.],radius=3.)
        exact=audit.AtomWallEnvelope(sphere,origin_wall,witness(sphere,origin_wall))
        self.assertTrue(exact.evaluate([pose([2.5,0.,0.])])[1][0])

    def test_envelope_is_not_full_wall_filter_and_invalid_queries_fail(self):
        shape=dict(atoms=[dict(center=[1.,0.,0.],radius=.5),dict(center=[-1.,0.,0.],radius=.2)])
        wall=dict(center=[0.,0.,0.],radius=3.)
        envelope=audit.AtomWallEnvelope(shape,wall,witness(shape,wall))
        query=pose([-3.,0.,0.])
        self.assertTrue(envelope.evaluate([query])[1][0])
        self.assertGreater(np.linalg.norm(np.asarray(query['position'])+shape['atoms'][1]['center'])+.2,3.)
        self.assertTrue(envelope.evaluate([dict(position=[0.,0.,0.],orientation=[1.+4e-9,0.,0.,0.])])[1][0])
        for bad in (dict(position=[math.nan,0.,0.],orientation=[1.,0.,0.,0.]),
                    dict(position=[0.,0.,0.],orientation=[0.,0.,0.,0.]),
                    dict(position=[0.,0.,0.],orientation=[1.+7e-9,0.,0.,0.])):
            with self.subTest(pose=bad),self.assertRaises(ValueError):envelope.evaluate([bad])

    def test_extreme_finite_distances_preserve_support_without_squaring(self):
        tiny=dict(atoms=[dict(center=[0.,0.,0.],radius=1e-200)])
        wall=dict(center=[0.,0.,0.],radius=3e-200)
        envelope=audit.AtomWallEnvelope(tiny,wall,witness(tiny,wall))
        logs,inside=envelope.evaluate([pose([0.,0.,0.]),pose([1e-190,0.,0.]),pose([1e200,0.,0.])])
        self.assertEqual(inside.tolist(),[True,False,False])
        self.assertTrue(math.isfinite(logs[0]))
        self.assertEqual(logs[1],-math.inf);self.assertEqual(logs[2],-math.inf)
        huge=dict(atoms=[dict(center=[1e308,0.,0.],radius=.5)])
        wall=dict(center=[-1e308,0.,0.],radius=3.)
        envelope=audit.AtomWallEnvelope(huge,wall,witness(huge,wall))
        with self.assertRaisesRegex(ValueError,'support query'):
            envelope.evaluate([pose([1e308,0.,0.])])

    def test_complete_learned_density_and_all_anchor_average_remain_present(self):
        for reciprocal in (False,True):
            cfg,manifest,rows,vessel,_,shape,raw,_=fixture(reciprocal=reciprocal)
            poses=[r['pose'] for r in rows]+[pose([80.,0.,0.])]
            full,geometry=vessel.evaluate(poses)
            learned=np.asarray([vessel.density.evaluate(relative_poses(poses,p))[0] for p in cfg['fixed_poses']])
            envelope,inside=vessel.envelope.evaluate(poses)
            expected=np.logaddexp(math.log(vessel.epsilon)+envelope,
                math.log1p(-vessel.epsilon)+logsumexp(learned,axis=0)-math.log(2))
            np.testing.assert_allclose(full,expected,atol=1e-13,rtol=0)
            self.assertFalse(inside[-1]);self.assertTrue(math.isfinite(full[-1]))
            selected=[]
            for index in (0,1):
                one=audit.VesselDensity(cfg,dict(manifest,proposal_anchor_index=index),raw,shape=shape)
                selected.append(one.evaluate(poses)[0])
            np.testing.assert_allclose(full,np.logaddexp(*selected)-math.log(2),atol=1e-13,rtol=0)
            self.assertGreater(float(np.max(abs(full-selected[0]))),.001)
            np.testing.assert_array_equal(geometry['uniform_support'],inside)

    def test_uniform_generation_support_replaces_cube_only_and_checks_old_new(self):
        cfg,manifest,rows,vessel,latent,shape,raw,_=fixture(reciprocal=True)
        # An offset atom can produce a center outside the old capture cube.
        cfg['capture_radius']=1.
        shape['atoms'][1]['center']=[50.,0.,0.]
        manifest['vessel_uniform_envelope']=witness(shape,manifest['atomic_wall'])
        vessel=audit.VesselDensity(cfg,manifest,raw,shape=shape)
        rotation=Rotation.from_rotvec([.2,.3,-.4]).as_matrix()
        position=np.asarray(manifest['atomic_wall']['center'])-rotation@np.asarray(shape['atoms'][1]['center'])
        world=pose(position,rotation)
        _,new=vessel.evaluate([world]);_,old=vessel.evaluate([cfg['initial_pose']])
        self.assertFalse(new['cube'][0]);self.assertTrue(new['uniform_support'][0])
        proposal=dict(moving_index=0,anchor_index=2,branch='uniform',component_index=None,null_reason=None,
            candidate=pose(position-np.asarray(cfg['capture_center']),rotation),
            old_log_density=float(old['anchor_log_densities'][1,0]),new_log_density=float(new['anchor_log_densities'][1,0]))
        proposal['log_reverse_forward']=proposal['old_log_density']-proposal['new_log_density']
        row=dict(outer_branch='vessel',pose=world,proposal=proposal)
        self.assertEqual(audit.check_generation_metadata(cfg,manifest,[row],vessel)['checked_vessel_generation_rows'],1)
        for field in ('old_log_density','new_log_density','log_reverse_forward'):
            bad=copy.deepcopy(row);bad['proposal'][field]+=.1
            with self.subTest(field=field),self.assertRaises(ValueError):
                audit.check_generation_metadata(cfg,manifest,[bad],vessel)
        bad=copy.deepcopy(row);bad['pose']['position'][0]+=100.
        bad['proposal']['candidate']['position'][0]+=100.
        with self.assertRaisesRegex(ValueError,'Uniform vessel generation'):
            audit.check_generation_metadata(cfg,manifest,[bad],vessel)

    def test_learned_generation_outside_envelope_and_reciprocal_label(self):
        cfg,manifest,_,vessel,_,_,_,_=fixture(reciprocal=True)
        world=pose([80.,0.,0.]);_,new=vessel.evaluate([world]);_,old=vessel.evaluate([cfg['initial_pose']])
        self.assertFalse(new['uniform_support'][0])
        oldlog=float(old['anchor_log_densities'][0,0]);newlog=float(new['anchor_log_densities'][0,0])
        proposal=dict(moving_index=0,anchor_index=1,branch='learned',component_index=0,
            component_inverted=True,null_reason=None,candidate=pose(np.asarray(world['position'])-cfg['capture_center']),
            old_log_density=oldlog,new_log_density=newlog,log_reverse_forward=oldlog-newlog)
        row=dict(outer_branch='vessel',pose=world,proposal=proposal)
        self.assertEqual(audit.check_generation_metadata(cfg,manifest,[row],vessel)['checked_vessel_generation_rows'],1)
        bad=copy.deepcopy(row);bad['proposal']['component_inverted']='true'
        with self.assertRaises(ValueError):audit.check_generation_metadata(cfg,manifest,[bad],vessel)

    def test_uniform_encoded_candidate_cannot_cross_boundary_within_coordinate_tolerance(self):
        cfg,manifest,_,vessel,_,_,_,_=fixture()
        envelope=vessel.envelope
        position=envelope.wall_center-envelope.atom_center+np.asarray([envelope.radius-5e-10,0.,0.])
        world=pose(position)
        _,new=vessel.evaluate([world]);_,old=vessel.evaluate([cfg['initial_pose']])
        oldlog=float(old['anchor_log_densities'][0,0]);newlog=float(new['anchor_log_densities'][0,0])
        proposal=dict(moving_index=0,anchor_index=1,branch='uniform',component_index=None,null_reason=None,
            candidate=pose(position-np.asarray(cfg['capture_center'])),
            old_log_density=oldlog,new_log_density=newlog,log_reverse_forward=oldlog-newlog)
        row=dict(outer_branch='vessel',pose=world,proposal=proposal)
        self.assertEqual(audit.check_generation_metadata(cfg,manifest,[row],vessel)['checked_vessel_generation_rows'],1)
        bad=copy.deepcopy(row);bad['proposal']['candidate']['position'][0]+=1e-9
        self.assertTrue(np.allclose(bad['proposal']['candidate']['position'],
            position-np.asarray(cfg['capture_center']),rtol=0,atol=2e-8))
        with self.assertRaisesRegex(ValueError,'reconstructed wall-envelope support'):
            audit.check_generation_metadata(cfg,manifest,[bad],vessel)

    def test_outer_half_mixture_weights_and_invalid_zeros_use_envelope(self):
        cfg,manifest,rows,vessel,latent,_,_,_=fixture()
        rows=reweight(cfg,manifest,rows,vessel,latent)
        self.assertEqual(audit.check_rows(cfg,manifest,rows,vessel,latent)['checked_attempts'],3)
        self.assertEqual(audit.check_generation_metadata(cfg,manifest,rows,vessel)['checked_vessel_generation_rows'],1)
        bad=copy.deepcopy(rows);bad[2]['log_importance_weight']=0.
        with self.assertRaisesRegex(ValueError,'explicit zero'):audit.check_rows(cfg,manifest,bad,vessel,latent)
        bad=copy.deepcopy(rows);bad[0]['log_hard_weight']+=bad[0]['latent_density']['log_physical_jacobian']
        with self.assertRaisesRegex(ValueError,'no extra J'):audit.check_rows(cfg,manifest,bad,vessel,latent)

    def test_native_class_schema8_requires_predecessor7_and_checked_envelope(self):
        import physical_native_class_line_vessel as native
        from test_physical_native_class_line_vessel import PhysicalNativeClassVesselTests
        toy=PhysicalNativeClassVesselTests();toy.setUp();self.addCleanup(toy.doCleanups)
        guide,manifest,rows,_=toy.rows()
        manifest.update(schema=8,pre_envelope_schema=7,atomic_wall=dict(center=[0.,0.,0.],radius=100.),
                        vessel_uniform_schema=audit.WALL_ENVELOPE_SCHEMA)
        manifest['vessel_uniform_envelope']=witness(toy.shape,manifest['atomic_wall'])
        vessel=audit.VesselDensity(toy.config,manifest,toy.region['gaussian_chart'],shape=toy.shape)
        logs,_=vessel.evaluate([r['pose'] for r in rows])
        mixed=half_mixture_log_density(logs,[d.log_physical_density for d in guide.evaluate_many([r['pose'] for r in rows])])
        for i,row in enumerate(rows):
            row['log_vessel_proposal_density']=float(logs[i]);row['log_proposal_density']=float(mixed[i])
            if row['hard_valid']:
                row['log_hard_weight']=row['log_importance_weight']=-float(mixed[i])
        self.assertEqual(native.check_rows(toy.config,manifest,rows,vessel,guide)['checked_attempts'],5)
        with self.assertRaisesRegex(ValueError,'native-class schema'):
            native.check_rows(toy.config,dict(manifest,pre_envelope_schema=4),rows,vessel,guide)
        legacy=dict(manifest,schema=7)
        for field in ('pre_envelope_schema','vessel_uniform_schema','vessel_uniform_envelope'):
            legacy.pop(field)
        unchecked=audit.VesselDensity(toy.config,legacy,toy.region['gaussian_chart'])
        with self.assertRaisesRegex(ValueError,'Unaudited vessel uniform envelope'):
            native.check_rows(toy.config,manifest,rows,unchecked,guide)

    def test_legacy_cube_density_and_generation_are_unchanged(self):
        for reciprocal in (False,True):
            cfg,manifest,rows,vessel,latent=data(reciprocal=reciprocal)
            poses=[r['pose'] for r in rows]
            logs,geometry=vessel.evaluate(poses)
            learned=np.asarray([vessel.density.evaluate(relative_poses(poses,p))[0] for p in cfg['fixed_poses']])
            cube=np.all(abs(np.asarray([p['position'] for p in poses])-cfg['capture_center'])<cfg['capture_radius'],axis=1)
            expected=np.logaddexp(np.where(cube,math.log(vessel.epsilon)-3*math.log(2*vessel.radius),-np.inf),
                math.log1p(-vessel.epsilon)+logsumexp(learned,axis=0)-math.log(2))
            np.testing.assert_allclose(logs,expected,rtol=0,atol=1e-13)
            self.assertIsNone(vessel.envelope)
            np.testing.assert_array_equal(geometry['uniform_support'],geometry['cube'])
            self.assertEqual(audit.check_rows(cfg,manifest,rows,vessel,latent)['checked_attempts'],3)
            self.assertEqual(audit.check_generation_metadata(cfg,manifest,rows,vessel)['checked_vessel_generation_rows'],1)


if __name__=='__main__':unittest.main()
