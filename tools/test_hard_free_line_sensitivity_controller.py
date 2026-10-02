"""Fixed sensitivity controller contracts; synthetic files and children only."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_hard_free_line_sensitivity as controller
from test_hard_free_line_physical_pilot import plan as original_plan


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def sensitivity_plan():
    p = original_plan()
    p.update(schema=controller.SCHEMA, arms=copy.deepcopy(controller.ARMS),
        sensitivity_comparison=copy.deepcopy(controller.PLAN), old_samples_pooled=False,
        pilot_evidence=dict(schema=controller.PILOT_SCHEMA, complete=True, total_unconditional_draws=131072))
    p['jobs'] = [dict(id=f'r{i%4:02}', arm=controller.ARMS[i//4]['id'], seed=seed, samples=16384)
                 for i,seed in enumerate(controller.SEEDS)]
    return p


def source_pilot(root):
    """Authenticate a tiny predecessor without fabricating or classifying poses."""
    common = root/'common'; package = common/'reference-package'; package.mkdir(parents=True)
    write(package/'shape.json', dict(test_shape=True)); write(package/'region.json', dict(test_region=True))
    write(package/'freeze.json', dict(files={p.name:controller.sha(p) for p in package.iterdir() if p.is_file()}))
    p = original_plan(); p['shape_sha256'] = controller.sha(package/'shape.json'); p['region_sha256'] = controller.sha(package/'region.json')
    write(common/'config.json', dict(shape=str(package/'shape.json'), depletant_radius=1.5, reservoir_density=.035))
    for arm in controller.PILOT_ARMS:
        write(common/(arm['id']+'.json'), dict(schema='defensive-hard-free-line-guide-v1',region_sha256=p['region_sha256'],
            raw_translation_axes=[0,1,2],minimum_conditional_mass=1e-12,conditional_probability=arm['beta'],
            defensive_uniform_shell_probability=.5,gaussian_components=[dict(index=i) for i in range(92)]))
    binary = common/'latent-region-normalizer'; binary.write_text('synthetic; never executed\n')
    bundle = dict(files={'src/lib.rs':dict(text='// synthetic source\n')})
    write(common/'source-bundle.json', bundle)
    (common/'rust-source/src').mkdir(parents=True); (common/'rust-source/src/lib.rs').write_text(bundle['files']['src/lib.rs']['text'])
    sources = controller.local_dependencies([Path(controller.__file__)])
    old_sources = {}
    for name,path in sources.items():
        if name in controller.NEW_SOURCES: continue
        (common/name).write_bytes(path.read_bytes()); old_sources[name] = controller.sha(common/name)
    p.update(binary_sha256=controller.sha(binary),source_bundle_sha256=controller.sha(common/'source-bundle.json'),
        rust_sources={},controller_sha256='old-controller',python_sources=old_sources,repository=str(root.parent),runtime=controller.runtime())
    records_by_arm = {}; terminal_jobs = []
    for arm in controller.PILOT_ARMS:
        records = []
        for job in (j for j in p['jobs'] if j['arm']==arm['id']):
            sample_hash = 'saved-'+arm['id']+'-'+job['id']
            audit = root/'audits'/arm['id']/(job['id']+'.json')
            write(audit, dict(complete=True,geometry_mode='full',samples=16384,samples_sha256=sample_hash))
            records.append(dict(job,samples_sha256=sample_hash,independent_audit_sha256=controller.sha(audit)))
            terminal_jobs.append(dict(job,status='complete',returncode=0,output=dict(samples_sha256=sample_hash)))
        records_by_arm[arm['id']] = dict(populations=records,estimates={'total':dict(row_uncertainty=dict(draws=65536))})
    write(root/'protocol.json', p)
    write(root/'freeze.json', dict(files={str(f.relative_to(root)):controller.sha(f) for f in root.rglob('*') if f.is_file()}))
    data = dict(schema='hard-free-line-physical-comparison-v1',complete=True,protocol_sha256=controller.sha(root/'protocol.json'),
        total_unconditional_draws=131072,arms=records_by_arm,diagnostics=dict(full_vessel_gate_open=False,assembly_gate_open=False))
    write(root/'comparison/analysis.json', data)
    write(root/'status.json', dict(complete=True,phase='complete',jobs=terminal_jobs,
        audits=[dict(status='complete',returncode=0) for _ in range(8)],protocol_sha256=controller.sha(root/'protocol.json'),
        comparison_sha256=controller.sha(root/'comparison/analysis.json')))
    reference = root.parent/'toy-reference.json'; write(reference, dict(complete=True, analytic_checks_passed=True))
    prerequisites = root.parent/'prerequisites.json'
    write(prerequisites, dict(schema='hard-free-line-sensitivity-prerequisites-v1',complete=True,all_checks_passed=True,
        binary_sha256=p['binary_sha256'],source_bundle_sha256=p['source_bundle_sha256'],arms=copy.deepcopy(controller.ARMS),
        independent_geometry='full',files={str(reference):controller.sha(reference)}))
    return p,bundle,prerequisites


def substitutes(p, bundle):
    stack = ExitStack()
    stack.enter_context(patch.object(controller, 'REGION_SHA', p['region_sha256']))
    stack.enter_context(patch.object(controller, 'SHAPE_SHA', p['shape_sha256']))
    stack.enter_context(patch.object(controller, 'verify_bundle', return_value=(bundle,{})))
    stack.enter_context(patch.object(controller, 'inventory', return_value=dict(files={},prior_seeds=controller.PILOT_SEEDS,fresh_seeds=controller.SEEDS)))
    return stack


class SensitivityControllerTests(unittest.TestCase):
    def test_only_fixed_one_factor_allocation_and_closed_gates(self):
        controller.validate_design(sensitivity_plan())
        for mutate in [lambda p:p.update(schema='arbitrary-control'),lambda p:p['arms'][0].update(alpha=.5),
            lambda p:p['arms'][1].update(lambda_ratio=128.),lambda p:p['arms'][0].update(beta=0.),
            lambda p:p['arms'][0].update(component_count=84),lambda p:p['jobs'][0].update(samples=65536),
            lambda p:p['jobs'][0].update(seed=controller.PILOT_SEEDS[0]),lambda p:p.update(old_samples_pooled=True),
            lambda p:p.update(cloud_replicates=1),lambda p:p.update(maximum_physical_workers=3),
            lambda p:p.update(maximum_audit_workers=8),lambda p:p.update(assembly_gate_open=True),
            lambda p:p['sensitivity_comparison'].update(stages_pooled=True)]:
            p=sensitivity_plan(); mutate(p)
            with self.assertRaises(ValueError): controller.validate_design(p)

    def test_source_pilot_must_be_complete_before_any_other_access(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); write(root/'status.json',dict(complete=False,phase='independent_audit'))
            with self.assertRaisesRegex(ValueError,'Source pilot must be complete'): controller.completed_pilot(root)

    def test_freeze_preserves_all_old_bytes_changes_only_alpha_and_commands(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); pilot=base/'pilot'; p,bundle,prerequisites=source_pilot(pilot); out=base/'sensitivity'
            before={str(f):controller.sha(f) for f in pilot.rglob('*') if f.is_file()}
            with substitutes(p,bundle):
                digest=controller.freeze(out,pilot,prerequisites,repository=base)
                frozen=controller.validate(out,digest)
                self.assertEqual([j['seed'] for j in frozen['jobs']],controller.SEEDS)
                self.assertEqual(frozen['total_unconditional_draws'],131072)
                for name in controller.UNCHANGED:
                    self.assertEqual((pilot/'common'/name).read_bytes(),(out/'common'/name).read_bytes())
                old=controller.read(pilot/'common/conditioned.json')
                self.assertEqual((out/'common/lambda64.json').read_bytes(),(pilot/'common/conditioned.json').read_bytes())
                alpha02=controller.read(out/'common/alpha02.json')
                self.assertEqual([k for k in old if old[k]!=alpha02[k]],['defensive_uniform_shell_probability'])
                for name in frozen['python_sources']:
                    if name not in controller.NEW_SOURCES:
                        self.assertEqual((out/'common'/name).read_bytes(),(pilot/'common'/name).read_bytes())
                for job in frozen['jobs']:
                    command=job['command']; self.assertEqual(command[command.index('--samples')+1],'16384')
                    self.assertEqual(command[command.index('--cloud-replicates')+1],'2')
                    self.assertEqual(command[command.index('--lambda-ratio')+1],'128' if job['arm']=='alpha02' else '64')
                self.assertFalse((out/'status.json').exists())
                self.assertEqual(before,{str(f):controller.sha(f) for f in pilot.rglob('*') if f.is_file()})
                with self.assertRaisesRegex(ValueError,'Fresh immutable'): controller.freeze(out,pilot,prerequisites,repository=base)
                (pilot/'comparison/analysis.json').write_text('{}\n')
                with self.assertRaisesRegex(ValueError,'pilot evidence changed'): controller.validate(out,digest)

    def test_prerequisites_bind_exact_binary_arms_geometry_and_reference_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p,bundle,path=source_pilot(root/'pilot'); original=controller.read(path)
            for change in [dict(complete=False),dict(all_checks_passed=False),dict(binary_sha256='different'),
                dict(source_bundle_sha256='different'),dict(arms=controller.PILOT_ARMS),dict(independent_geometry='trace'),dict(files={})]:
                write(path,dict(original,**change))
                with self.assertRaises(ValueError): controller.validate_prerequisites(path,p['binary_sha256'],p['source_bundle_sha256'])
            write(path,original)
            controller.validate_prerequisites(path,p['binary_sha256'],p['source_bundle_sha256'])
            reference=next(iter(original['files'])); Path(reference).write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'evidence changed'): controller.validate_prerequisites(path,p['binary_sha256'],p['source_bundle_sha256'])

    def test_inventory_rejects_current_or_previous_stream_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); write(root/'protocol.json',dict(jobs=[dict(seed=controller.SEEDS[0])]))
            with self.assertRaisesRegex(ValueError,'Seed collision'): controller.inventory([root],controller.SEEDS)
            result=controller.inventory([root],[999]); self.assertEqual(result['prior_seeds'],[controller.SEEDS[0]])

    def test_arm_specific_output_rejects_swapped_alpha_or_cloud_intensity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); protocol=dict(binary_sha256='binary',source_bundle_sha256='bundle')
            for arm in controller.ARMS:
                job=dict(directory=str(root/'run'),samples=16384,seed=controller.SEEDS[0],arm=arm['id'])
                manifest=dict(schema='importance-latent-region-normalizer-v6',guide_schema='defensive-hard-free-line-guide-v1',
                    samples=16384,seed=job['seed'],activity=.035,lambda_ratio=arm['lambda_ratio'],cloud_replicates=2,
                    importance_uniform_probability=arm['alpha'],importance_component_count=92,region_sha256=controller.REGION_SHA,
                    shape_sha256=controller.SHAPE_SHA,executable_sha256='binary',source_bundle_sha256='bundle',
                    config_sha256='same',importance_guide_sha256='same')
                summary=dict(complete=True,samples=16384,manifest=manifest,samples_sha256='same',attempts_sha256='same')
                with patch.object(controller,'read',side_effect=lambda p:manifest if Path(p).name=='manifest.json' else summary),patch.object(controller,'sha',return_value='same'):
                    controller.verify_output(root,protocol,job)
                    changes=[('importance_uniform_probability',.5 if arm['alpha']==.2 else .2),
                        ('lambda_ratio',64. if arm['lambda_ratio']==128 else 128.),('seed',controller.PILOT_SEEDS[0]),
                        ('importance_guide_sha256','other-guide'),('cloud_replicates',1)]
                    for key,bad in changes:
                        old=manifest[key]; manifest[key]=bad
                        with self.assertRaisesRegex(ValueError,'identity/target'): controller.verify_output(root,protocol,job)
                        manifest[key]=old
                    summary['attempts_sha256']='lost-attempt'
                    with self.assertRaisesRegex(ValueError,'Raw output changed: attempts'): controller.verify_output(root,protocol,job)

    def test_exclusive_launch_claim_preserves_first_accounting(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); controller.claim_status(root,dict(complete=False,jobs=[1])); before=(root/'status.json').read_bytes()
            with self.assertRaises(FileExistsError): controller.claim_status(root,dict(complete=False,jobs=[2]))
            self.assertEqual(before,(root/'status.json').read_bytes())

    def test_private_namespace_refuses_before_launch_claim(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with (patch.object(controller,'validate',return_value=sensitivity_plan()),
                  patch.object(controller,'__file__',str(root/'common/run_hard_free_line_sensitivity.py')),
                  patch.object(Path,'read_text',return_value='codex-linux-san\n'),
                  patch.object(controller,'claim_status') as claim,
                  patch.object(controller,'execute_group') as execute):
                with self.assertRaisesRegex(ValueError,'host PID namespace'): controller.run(root,'hash')
                claim.assert_not_called(); execute.assert_not_called()
            self.assertFalse((root/'status.json').exists())

    def test_failed_child_drains_started_peer_and_never_classifies_or_retries(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=sensitivity_plan(); p['repository']=str(root)
            for job in p['jobs']:
                job.update(kind='physical',status='pending',directory=str(root/job['arm']/job['id']),
                    log=str(root/job['arm']/(job['id']+'.log')),command=['synthetic-never-executed'])
                Path(job['log']).parent.mkdir(exist_ok=True)
            children=[]
            class Child:
                def __init__(self,index): self.pid=index+100; self.index=index; self.polls=0; self.waited=False
                def poll(self): self.polls+=1; return 1 if self.index==0 and self.polls>=2 else None
                def wait(self): self.waited=True; return 1 if self.index==0 else 0
            def popen(*args,**kwargs):
                child=Child(len(children)); children.append(child); return child
            real=controller.execute_group
            def execute(*args,**kwargs):
                return real(*args,**kwargs,popen=popen,pause=lambda x:None,
                    capacity=lambda r:dict(workers=0,physical_pids=[]),birth=lambda pid:'synthetic')
            with (patch.object(controller,'validate',return_value=p),
                  patch.object(controller,'__file__',str(root/'common/run_hard_free_line_sensitivity.py')),
                  patch.object(controller,'require_host_namespace'),
                  patch.object(controller,'execute_group',side_effect=execute),
                  patch.object(controller,'analyze') as analyze):
                with self.assertRaisesRegex(RuntimeError,'Child failed'): controller.run(root,'protocol-hash')
                self.assertEqual(len(children),2); self.assertTrue(all(c.waited for c in children)); analyze.assert_not_called()
                status=controller.read(root/'status.json'); self.assertFalse(status['complete']); self.assertEqual(status['phase'],'physical_failed')
                self.assertEqual([j['status'] for j in status['jobs']],['failed','complete']+['not_started']*6)
                self.assertEqual(status['audits'],[])
                with self.assertRaisesRegex(ValueError,'No retry'): controller.run(root,'protocol-hash')
                self.assertEqual(len(children),2)


if __name__=='__main__': unittest.main()
