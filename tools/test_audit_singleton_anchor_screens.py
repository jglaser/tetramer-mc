"""Small synthetic arithmetic/metadata fixtures; no protein inputs or queries."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import audit_singleton_anchor_screens as audit


def pose(x=0., y=0., z=0., angle=0.):
    return dict(position=[x,y,z], orientation=[math.cos(angle/2),0.,0.,math.sin(angle/2)])


def atlas():
    return dict(schema='reciprocal-pose-mixture-v1', reciprocal_components=[True],
        base_model=dict(schema='weighted-pose-mixture-v1',coordinate_convention='anchor-body-relative',
            angular_length=2.,anchors=[dict(position=[1.,2.,3.],rotation=np.eye(3).tolist())],
            means=[[2.,0.,0.,0.,0.,2.]],covariances=['deliberately not factorizable']))


def save(path,value):
    path.write_text(json.dumps(value)); return audit.reference(path)


def toy_plan(root):
    model=atlas(); model['base_model']['means']=[[0.]*6]; model['base_model']['anchors'][0]['position']=[1.,0.,0.]
    model_ref=save(root/'atlas.json',model)
    state=save(root/'state.json',dict(poses=[pose(100),pose(),pose(30),pose(3)]))
    shape=save(root/'shape.json',dict(atoms=[dict(center=[0.,0.,0.],radius=.1)]))
    construction=dict(id='source-0',state='source',moving=0,neighbors=[1,2])
    manifest=save(root/'manifest.json',dict(model=model_ref,shape=shape,anchor=2,
        states=[dict(id='source',file=state)],constructions=[construction]))
    # Both means at +/-1: anchor separation30 means all four distances fail.
    baseline=dict(single_labels=4,decoded_means=4,mean_decode_failures=0,
        possible_interface_pairs=4,decoded_interface_pairs=4,distance_rejected_pairs=4,
        angle_rejected_pairs=0,candidate_pairs_before_cap=0)
    return dict(manifest=manifest,policy=dict(audit.POLICY,branches=2),baseline={'source-0':baseline})


def metadata_fixture(root):
    old=root/'old'; old.mkdir(); diagnostic=old/'diagnostic'; diagnostic.mkdir()
    execution=old/'execution'; execution.mkdir()
    job=dict(id='toy-passive',population='whole_27_132',phase='geometry')
    jobdir=audit.driver.job_directory(old,0,job); jobdir.mkdir(parents=True)
    for path in [old/'preparation.json',execution/'claim.json',execution/'status.json',
                 *(jobdir/name for name in ('attempt.json','process.json','exit.json','success.json'))]: save(path,{})
    save(execution/'summary.json',dict(complete=True,passed=True))
    poses=[pose(float(i)*10) for i in range(264)]
    states=[]
    for name in ['source']+[f'prepared-{i}' for i in range(4)]:
        ref=save(old/(name+'.json'),dict(coordinate_frame='sphere_centered',poses=poses))
        states.append(dict(id=name,file=ref,poses_pointer='/poses'))
    config=save(old/'config.json',dict(initial_poses=poses))
    shape=save(old/'shape.json',dict(atoms=[dict(center=[0.,0.,0.],radius=.1)]))
    # Metadata-only shape/inventory; intentionally not a decodable atlas.
    model=save(old/'atlas.json',dict(base_model=dict(shape_sha256=shape['sha256'],anchors=[None]*1024),
                                    reciprocal_components=[True]*1024))
    constructions=[dict(id=f'{s["id"]}-{i}',state=s['id'],moving=i,neighbors=[j,228])
                   for s in states for i,j in ((27,132),(132,27))]
    manifest=save(old/'manifest.json',dict(schema='singleton-fusion-diagnostic-manifest-v1',
        context='whole_27_132',members=[27,132],anchor=228,coordinate_frame='sphere_centered',wall_center=[0.,0.,0.],
        oligomer=dict(pair_distance_A=8.,pair_angle_degrees=60.),states=states,constructions=constructions,
        config=config,shape=shape,model=model,output=str(diagnostic)))
    rows=[dict(state='complete',ordinal=i,construction=c,error=None,
               diagnostics=dict(complete=True,**{k:0 for k in audit.COUNT_KEYS})) for i,c in enumerate(constructions)]
    journal=diagnostic/'attempts.jsonl'; journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    refs=[config,shape,model,manifest]+[s['file'] for s in states]
    files={r['path']:r['sha256'] for r in refs}
    terminal=save(diagnostic/'summary.json',dict(schema='singleton-fusion-diagnostic-v1',
        manifest_sha256=manifest['sha256'],attempts_sha256=audit.sha(journal),input_sha256=files))
    save(old/'execution-plan.json',dict(files=files,jobs=[job]))
    sources={str(p):audit.sha(p) for p in [*audit.source_paths(),Path(__file__).resolve()]}
    validation=save(root/'validation.json',dict(complete=True,passed=True,source_before=sources,source_after=sources))
    return manifest,terminal,validation


class AnchorScreenTests(unittest.TestCase):
    def test_zero_latent_decode_reciprocal_and_anchor_composition(self):
        branches=audit.decode_virtual_means(atlas(),2)
        self.assertEqual([(b['component'],b['inverted']) for b in branches],[(0,False),(0,True)])
        np.testing.assert_allclose(branches[0]['position'],[3.,2.,3.])
        # Cayley [0,0,1] is 90 degrees; inversion couples translation to rotation.
        np.testing.assert_allclose(branches[1]['position'],[-2.,3.,-3.],atol=1e-14)
        np.testing.assert_allclose(audit.rotation(branches[0]['orientation']),
                                   [[0,-1,0],[1,0,0],[0,0,1]],atol=1e-14)
        placed,_=audit.placed_means(branches,pose(10.,angle=math.pi/2))
        np.testing.assert_allclose(placed[0],[8.,3.,3.],atol=1e-14)
        np.testing.assert_allclose(placed[1],[7.,-2.,-3.],atol=1e-14)
        with self.assertRaisesRegex(ValueError,'branch count'): audit.decode_virtual_means(atlas(),3)

    def test_inventory_excludes_moving_and_ties_by_label_keeps_outside_control(self):
        poses=[pose(999),pose(),pose(100),pose(2),pose(-2),pose(4),pose(40)]
        original=audit.anchor_inventory(poses,0,1,2,1.,nearest=3)
        self.assertEqual([r['anchor'] for r in original],[2,3,4,5])
        self.assertFalse(original[0]['within_mean_bound'])
        altered=copy.deepcopy(poses); altered[0]={'not_even_a_pose':True}
        self.assertEqual(original,audit.anchor_inventory(altered,0,1,2,1.,nearest=3))
        altered[1]=pose(99)
        self.assertNotEqual(original,audit.anchor_inventory(altered,0,1,2,1.,nearest=3))

    def test_strict_distance_and_angle_thresholds_are_flagged(self):
        q=np.array([[1.,0.,0.,0.]])
        first=(np.zeros((1,3)),q)
        second=(np.array([[8.,0,0],[8.-1e-9,0,0],[8.+1e-9,0,0],[0,0,0],[0,0,0]]),
                np.array([q[0],q[0],q[0],pose(angle=math.pi/3-1e-8)['orientation'],
                          pose(angle=math.pi/3+1e-8)['orientation']]))
        result=audit.screen(first,second,{})
        self.assertEqual(result['distance_pairs'],3)
        self.assertEqual(result['candidate_pairs_before_cap'],2)
        self.assertEqual(result['angle_rejected_pairs'],1)
        self.assertEqual(result['distance_ambiguous_pairs'],3)
        self.assertTrue(result['threshold_ambiguous'])
        near=(np.zeros((1,3)),np.array([pose(angle=math.pi/3)['orientation']]))
        self.assertEqual(audit.screen(first,near,{})['angle_ambiguous_pairs'],1)

    def test_chunked_screen_matches_independent_nested_enumeration(self):
        p=np.array([[float(i)*3,0.,0.] for i in range(67)])
        pp=np.array([[float(i)*4,1.,0.] for i in range(53)])
        q=np.array([pose(angle=i*.07)['orientation'] for i in range(67)])
        qq=np.array([pose(angle=i*.11)['orientation'] for i in range(53)])
        distances=passing=0
        for i,a in enumerate(p):
            for j,b in enumerate(pp):
                d=math.dist(a,b)
                if d < 8:
                    distances+=1
                    if 2*math.acos(min(1.,abs(sum(x*y for x,y in zip(q[i],qq[j]))))) < math.pi/3: passing+=1
        result=audit.screen((p,q),(pp,qq),{},chunk=64)
        self.assertEqual((result['distance_pairs'],result['candidate_pairs_before_cap']),(distances,passing))
        self.assertEqual(result['query_chunks'],2)
        self.assertEqual(result['left_means_completed'],67)

    def test_pair_cap_retains_prefix_and_time_check_interrupts(self):
        values=(np.zeros((3,3)),np.array([[1.,0.,0.,0.]]*3)); counts={}
        with self.assertRaisesRegex(ValueError,'distance-pair cap'):
            audit.screen(values,values,counts,cap=4,chunk=1)
        self.assertEqual(counts['distance_pairs'],3)
        self.assertEqual(counts['candidate_pairs_before_cap'],3)
        self.assertEqual(counts['left_means_completed'],1)
        with self.assertRaisesRegex(ValueError,'budget exhausted'):
            audit.screen(values,values,{},progress=lambda:audit.require(False,'CPU budget exhausted'))

    def test_baseline_gate_precedes_alternatives_and_failed_prefix_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=toy_plan(Path(tmp)); events=[]; active={}
            result=audit.execute(plan,lambda r:events.append(copy.deepcopy(r)),lambda:None,active)
            self.assertEqual(result['original_screens_checked'],1)
            self.assertEqual([r['selection']['anchor'] for r in result['screens']],[2,3])
            self.assertTrue(result['screens'][0]['baseline_agrees'])
            self.assertIsNone(result['screens'][1]['baseline_agrees'])
            plan['baseline']['source-0']['candidate_pairs_before_cap']=1; events=[]
            with self.assertRaisesRegex(ValueError,'disagrees'):
                audit.execute(plan,lambda r:events.append(copy.deepcopy(r)),lambda:None,active)
            self.assertEqual(sum(r['state']=='screen_begin' for r in events),1)
            self.assertEqual(active['selection']['anchor'],2)

    def test_owned_run_failure_writes_partial_journal_and_no_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'analysis'; inputs=root/'input'; inputs.write_text('toy')
            plan=dict(schema=audit.SCHEMA,root=str(root),output=str(out),limits=audit.LIMITS,
                policy=audit.POLICY,source_sha256={str(p):audit.sha(p) for p in audit.source_paths()},
                runtime=audit.runtime(),input_sha256={str(inputs):audit.sha(inputs)})
            plan_path=root/'protocol.json'; save(plan_path,plan)
            def fail(_plan,emit,_check,active):
                active.update(construction={'id':'toy'},counts={'distance_pairs':4})
                emit(dict(state='screen_begin',**active)); raise ValueError('synthetic bounded failure')
            with mock.patch.object(audit,'live_authority'), mock.patch.object(audit,'execute',side_effect=fail):
                with self.assertRaisesRegex(ValueError,'synthetic bounded failure'):
                    audit.run(plan_path,audit.sha(plan_path),out)
            failure=audit.read(out/'failure.json')
            self.assertFalse(failure['complete']); self.assertEqual(failure['active']['counts']['distance_pairs'],4)
            self.assertEqual(failure['attempts_sha256'],audit.sha(out/'attempts.jsonl'))
            self.assertFalse((out/'summary.json').exists())
            self.assertEqual([json.loads(x)['state'] for x in (out/'attempts.jsonl').read_text().splitlines()],
                             ['started','screen_begin','failed'])

    def test_preparation_never_decodes_means_and_binds_validated_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); manifest,terminal,validation=metadata_fixture(root); output=root/'new'
            with mock.patch.object(audit.driver,'completed_terminal',return_value=terminal), \
                 mock.patch.object(audit,'decode_virtual_means',side_effect=AssertionError('Preparation must not decode')):
                receipt=audit.prepare(manifest['path'],validation['path'],out=output)
                execution=audit.read(receipt['execution_plan']['path'])
                audit.driver.verify_plan(receipt['execution_plan']['path'],execution,receipt['execution_plan']['sha256'],fresh=True)
                self.assertFalse(receipt['launched']); self.assertFalse((output/'analysis').exists())
                self.assertEqual(len(execution['jobs']),1)
                self.assertEqual(execution['jobs'][0]['cpu_limit_seconds'],300)
                self.assertEqual(execution['jobs'][0]['address_space_limit_bytes'],2*2**30)
                stale=audit.read(validation['path']); stale['source_before'][str(Path(audit.__file__).resolve())]='0'*64
                stale['source_after']=copy.deepcopy(stale['source_before']); Path(validation['path']).write_text(json.dumps(stale))
                with self.assertRaisesRegex(ValueError,'Stale/omitted'):
                    audit.make_plan(manifest['path'],validation['path'],root=root/'rejected')
                self.assertFalse((root/'rejected').exists())


if __name__ == '__main__': unittest.main()
