"""Fixed-size controller and aggregate comparison tests; no physical jobs."""
import copy
from contextlib import ExitStack
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_hard_free_line_physical_pilot as controller
import analyze_hard_free_line_population_size as comparison
from test_hard_free_line_physical_pilot import plan as pilot_plan
from analyze_mobile_competing_reference import paired_moments


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def size_plan():
    p=pilot_plan();p.update(schema=controller.SIZE_SCHEMA,arms=copy.deepcopy(controller.SIZE_ARMS),
        total_unconditional_draws=524288,pilot_evidence=dict(schema=controller.SCHEMA,complete=True,total_unconditional_draws=131072),
        population_size_comparison=copy.deepcopy(comparison.PLAN),old_samples_pooled=False)
    for j,seed in zip(p['jobs'],controller.SIZE_SEEDS):j.update(seed=seed,samples=65536)
    return p


def source_pilot(root):
    """Small authenticated predecessor archive; physical audit payloads mocked."""
    common=root/'common';package=common/'reference-package';package.mkdir(parents=True)
    write(package/'shape.json',{'test_shape':True});write(package/'region.json',{'test_region':True})
    write(package/'freeze.json',dict(files={p.name:controller.sha(p)for p in package.iterdir()if p.is_file()}))
    p=pilot_plan();p['shape_sha256']=controller.sha(package/'shape.json');p['region_sha256']=controller.sha(package/'region.json')
    write(common/'config.json',dict(shape=str(package/'shape.json'),depletant_radius=1.5,reservoir_density=.035))
    for arm in controller.ARMS:
        write(common/(arm['id']+'.json'),dict(schema='defensive-hard-free-line-guide-v1',region_sha256=p['region_sha256'],
            raw_translation_axes=[0,1,2],minimum_conditional_mass=1e-12,conditional_probability=arm['beta'],
            defensive_uniform_shell_probability=.5,gaussian_components=[{}]*92))
    binary=common/'latent-region-normalizer';binary.write_text('synthetic, never executed\n')
    bundle=dict(files={'src/lib.rs':dict(text='// synthetic immutable source\n')})
    write(common/'source-bundle.json',bundle)
    (common/'rust-source/src').mkdir(parents=True);(common/'rust-source/src/lib.rs').write_text(bundle['files']['src/lib.rs']['text'])
    sources=controller.local_dependencies([Path(controller.__file__)])
    old_sources={}
    for name,path in sources.items():
        if name=='analyze_hard_free_line_population_size.py':continue
        (common/name).write_bytes(b'# frozen predecessor controller\n' if name==Path(controller.__file__).name else path.read_bytes())
        old_sources[name]=controller.sha(common/name)
    preparation=root.parent/'preparation';preparation.mkdir();write(preparation/'guide.json',dict(frozen=True))
    write(preparation/'freeze.json',dict(files={'guide.json':controller.sha(preparation/'guide.json')}))
    prerequisites=root.parent/'prerequisites.json';write(prerequisites,dict(files={}))
    (common/'prerequisites.json').write_bytes(prerequisites.read_bytes())
    p.update(binary_sha256=controller.sha(binary),source_bundle_sha256=controller.sha(common/'source-bundle.json'),
        rust_sources={},controller_sha256=old_sources[Path(controller.__file__).name],python_sources=old_sources,
        repository=str(root.parent),runtime=controller.runtime(),preparation=str(preparation),
        preparation_freeze_sha256=controller.sha(preparation/'freeze.json'),
        prerequisites=str(prerequisites),prerequisites_sha256=controller.sha(prerequisites))
    status_jobs=[];arms={}
    for arm in controller.ARMS:
        records=[]
        for j in (j for j in p['jobs']if j['arm']==arm['id']):
            sample_hash='saved-'+arm['id']+'-'+j['id']
            audit=root/'audits'/arm['id']/(j['id']+'.json')
            write(audit,dict(complete=True,geometry_mode='full',samples=16384,samples_sha256=sample_hash))
            records.append(dict(j,samples_sha256=sample_hash,independent_audit_sha256=controller.sha(audit)))
            status_jobs.append(dict(j,status='complete',returncode=0,output=dict(samples_sha256=sample_hash)))
        arms[arm['id']]=dict(populations=records,estimates={'total':dict(row_uncertainty=dict(draws=65536))})
    write(root/'protocol.json',p)
    write(root/'freeze.json',dict(files={str(f.relative_to(root)):controller.sha(f)for f in root.rglob('*')if f.is_file()}))
    data=dict(schema='hard-free-line-physical-comparison-v1',complete=True,protocol_sha256=controller.sha(root/'protocol.json'),
        total_unconditional_draws=131072,arms=arms,diagnostics=dict(full_vessel_gate_open=False,assembly_gate_open=False))
    write(root/'comparison/analysis.json',data)
    status=dict(complete=True,phase='complete',jobs=status_jobs,audits=[dict(status='complete',returncode=0)for _ in range(8)],
        protocol_sha256=controller.sha(root/'protocol.json'),comparison_sha256=controller.sha(root/'comparison/analysis.json'))
    write(root/'status.json',status)
    return p,bundle


def substitutes(p,bundle):
    stack=ExitStack()
    stack.enter_context(patch.object(controller,'REGION_SHA',p['region_sha256']))
    stack.enter_context(patch.object(controller,'SHAPE_SHA',p['shape_sha256']))
    stack.enter_context(patch.object(controller,'verify_bundle',return_value=(bundle,{})))
    stack.enter_context(patch.object(controller,'inventory',return_value=dict(files={},prior_seeds=controller.SEEDS,fresh_seeds=controller.SIZE_SEEDS)))
    return stack


def estimate(values,n,seedbase):
    logs=np.array([-math.inf if x==0 else math.log(x)for x in values]);stats=paired_moments(logs,logs-.5)
    populations=[dict(id=f'r{i:02}',seed=seedbase+i,draws=n,log_Qz=None if not math.isfinite(x)else float(x),
        log_Q0=None if not math.isfinite(x)else float(x-.5))for i,x in enumerate(logs)]
    return dict(populations=populations,population_uncertainty=stats,
        row_uncertainty=dict(stats,draws=4*n))


def stage(n,seedbase):
    arms={}
    for ai,name in enumerate(comparison.PLAN['arms']):
        e=estimate([1.,1.05,.95,1.],n,seedbase+4*ai)
        strata={}
        for family in comparison.PLAN['stratum_families']:
            count=64 if family=='orthant' else 3
            strata[family]={r:[dict(copy.deepcopy(e),bin=i,observed_class_fraction={'Qz':1./count})for i in range(count)]for r in comparison.PLAN['regions']}
        arms[name]=dict(estimates={r:copy.deepcopy(e)for r in comparison.PLAN['regions']},strata=strata)
    return dict(schema='hard-free-line-physical-comparison-v1',complete=True,total_unconditional_draws=8*n,arms=arms,
        diagnostics=dict(full_vessel_gate_open=False,assembly_gate_open=False,quality={},free_energy_intervals={}))


class PopulationSizeControllerTests(unittest.TestCase):
    def test_v1_and_only_fixed_new_stage_are_accepted(self):
        controller.validate_design(pilot_plan());controller.validate_design(size_plan())
        for mutate in [lambda p:p.update(schema='arbitrary-size'),lambda p:p['arms'][0].update(samples=32768),
            lambda p:p['jobs'][0].update(samples=16384),lambda p:p['jobs'][0].update(seed=controller.SEEDS[0]),
            lambda p:p.update(total_unconditional_draws=131072),lambda p:p.update(old_samples_pooled=True),
            lambda p:p.update(full_vessel_gate_open=True),lambda p:p['population_size_comparison'].update(stages_pooled=True)]:
            p=size_plan();mutate(p)
            with self.assertRaises(ValueError):controller.validate_design(p)

    def test_complete_pilot_required_before_further_input_access(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);write(root/'status.json',dict(complete=False,phase='independent_audit'))
            with self.assertRaisesRegex(ValueError,'Source pilot must be complete'):controller.completed_pilot(root)

    def test_freeze_clones_exact_inputs_and_preserves_predecessor(self):
        with tempfile.TemporaryDirectory()as d:
            base=Path(d);root=base/'pilot';root.mkdir();p,bundle=source_pilot(root);out=base/'large'
            before={str(f):controller.sha(f)for f in root.rglob('*')if f.is_file()}
            with substitutes(p,bundle):
                # A frozen old controller hash need not match the new source.
                self.assertNotEqual(p['controller_sha256'],controller.sha(controller.__file__))
                controller.completed_pilot(root)
                digest=controller.freeze_population_size(out,root,repository=base)
                larger=controller.validate(out,digest)
                self.assertEqual(larger['total_unconditional_draws'],524288)
                self.assertEqual([j['seed']for j in larger['jobs']],controller.SIZE_SEEDS)
                for name in larger['unchanged_pilot_common_sha256']:
                    self.assertEqual((root/'common'/name).read_bytes(),(out/'common'/name).read_bytes())
                self.assertEqual((out/'common/hard_free_line_physical_reference.py').read_bytes(),(root/'common/hard_free_line_physical_reference.py').read_bytes())
                self.assertFalse((out/'status.json').exists())
                for j in larger['jobs']:
                    self.assertEqual(j['command'][j['command'].index('--samples')+1],'65536')
                    self.assertEqual(j['command'][j['command'].index('--lambda-ratio')+1],'128')
                self.assertEqual(before,{str(f):controller.sha(f)for f in root.rglob('*')if f.is_file()})
                with self.assertRaisesRegex(ValueError,'Fresh immutable'):controller.freeze_population_size(out,root,repository=base)
                (root/'comparison/analysis.json').write_text('{}\n')
                with self.assertRaisesRegex(ValueError,'pilot evidence changed'):controller.validate(out,digest)

    def test_atomic_launch_claim_never_replaces_existing_accounting(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);controller.claim_status(root,dict(complete=False,jobs=[1]))
            saved=(root/'status.json').read_bytes()
            with self.assertRaises(FileExistsError):controller.claim_status(root,dict(complete=False,jobs=[2]))
            self.assertEqual(saved,(root/'status.json').read_bytes())

    def test_new_output_is_bound_to_large_size_seed_and_guide(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);job=dict(directory=str(root/'run'),samples=65536,seed=controller.SIZE_SEEDS[0],arm='baseline')
            manifest=dict(schema='importance-latent-region-normalizer-v6',guide_schema='defensive-hard-free-line-guide-v1',
                samples=65536,seed=job['seed'],activity=.035,lambda_ratio=128.,cloud_replicates=2,
                importance_uniform_probability=.5,importance_component_count=92,region_sha256=controller.REGION_SHA,
                shape_sha256=controller.SHAPE_SHA,executable_sha256='binary',source_bundle_sha256='bundle',
                config_sha256='same',importance_guide_sha256='same')
            summary=dict(complete=True,samples=65536,manifest=manifest,samples_sha256='same',attempts_sha256='same')
            with patch.object(controller,'read',side_effect=lambda p:manifest if Path(p).name=='manifest.json'else summary),patch.object(controller,'sha',return_value='same'):
                controller.verify_output(root,dict(binary_sha256='binary',source_bundle_sha256='bundle'),job)
                for key,bad in [('samples',16384),('seed',controller.SEEDS[0]),('importance_guide_sha256','different')]:
                    old=manifest[key];manifest[key]=bad
                    with self.assertRaisesRegex(ValueError,'identity/target'):controller.verify_output(root,dict(binary_sha256='binary',source_bundle_sha256='bundle'),job)
                    manifest[key]=old


class PopulationSizeComparisonTests(unittest.TestCase):
    def test_linear_se_gate_differs_from_log_delta_gate(self):
        a=estimate([1.,1.,1.,1.],16384,10);b=estimate([1.5,1.5,4.5,4.5],65536,20)
        result=comparison.compare_linear(a,b)
        self.assertTrue(result['SE_passed']);self.assertFalse(result['absolute_passed'])
        self.assertGreater(abs(result['log_larger_minus_pilot']),3*result['combined_population_log_delta_SE'])
        self.assertAlmostEqual(result['scaled_independent_difference_SE']*math.exp(result['log_scale']),math.sqrt(.75))

    def test_stage_comparison_does_not_pool_or_hide_material_strata(self):
        a,b=stage(16384,10),stage(65536,30)
        # A matched-bin discrepancy remains visible even when all other bins agree.
        entry=b['arms']['conditioned']['strata']['orthant'][comparison.PLAN['regions'][1]][39]
        changed=estimate([2.,2.1,1.9,2.],65536,34);entry.update(changed)
        result=comparison.compare_population_size(a,b,copy.deepcopy(comparison.PLAN))
        self.assertFalse(result['stages_pooled']);self.assertEqual(result['old_native_classifier_calls'],0)
        self.assertEqual(len(result['all_stratum_comparisons']),2*4*70)
        self.assertEqual(len(result['failed_material_strata']),1)
        self.assertEqual(result['failed_material_strata'][0]['bin'],39)
        self.assertFalse(result['full_vessel_gate_open']);self.assertFalse(result['assembly_gate_open'])

    def test_missing_denominator_shared_seed_and_unobserved_mass(self):
        a=estimate([1.,1.,1.,1.],16384,10);b=estimate([1.,1.,1.,1.],65536,20)
        b['populations'][0]['draws']=16384
        with self.assertRaisesRegex(ValueError,'denominator'):comparison.compare_linear(a,b)
        b=estimate([1.,1.,1.,1.],65536,10)
        with self.assertRaisesRegex(ValueError,'not independent'):comparison.compare_linear(a,b)
        result=comparison.compare_linear(a,estimate([0.,0.,0.,0.],65536,20))
        self.assertFalse(result['observed']);self.assertFalse(result['passed'])

    def test_extreme_finite_log_mass_is_not_relabelled_unobserved(self):
        a=estimate([1.,1.,1.,1.],16384,10);b=estimate([1.,1.,1.,1.],65536,20)
        for p in a['populations']:p['log_Qz']-=1000.;p['log_Q0']-=1000.
        for key in ('log_Qz','log_Q0'):a['population_uncertainty'][key]-=1000.
        result=comparison.compare_linear(a,b)
        self.assertTrue(result['observed']);self.assertFalse(result['passed'])
        self.assertEqual(result['log_larger_minus_pilot'],1000.)
        self.assertTrue(result['shared_display_scale_underflow']['pilot'])
        self.assertEqual(result['log_pilot_linear_mean'],-1000.)


if __name__=='__main__':unittest.main()
