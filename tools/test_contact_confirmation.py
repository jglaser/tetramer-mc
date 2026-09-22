"""Fail-closed allocation/proposal tests for the separate confirmation campaign."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import run_contact_confirmation as runner
from test_contact_bank_pilot import fixture as base_fixture
from prepare_refined_contact_bank import guide_variants
from prepare_contact_bank_guides import log_proposal


def fixture(root):
    repo, binary, bundle, package, pins = base_fixture(root)
    old = runner.read(package/'guide-bank.json')
    shutil.copy2(package/'guide-bank.json',package/'retained-guide.json')
    candidates = [dict(weight=1/24, mean=[.03*i,0.,0.,0.,0.,0.],
                       covariance=(np.eye(6)*.02).tolist()) for i in range(24)]
    guides = guide_variants(old, candidates)
    for name, guide in guides.items(): runner.write(package/f'guide-{name}.json', guide)
    plan = runner.read(package/'plan.json'); plan['schema'] = runner.PACKAGE_SCHEMA
    plan['guide_sha256'] = {name: runner.sha(package/f'guide-{name}.json') for name in guides}
    for i in range(24):
        plan['anchor_metadata'].append(dict(component_index=56+i,
            source_group='fresh-pilot', **{'class': ('old_R5_intersection_native','remaining_R4_native','contact_no_native_entry')[i%3]},
            arm='pilot', population_id=f'p{i//3:02d}', seed=3000+i//3, draw=i, source_samples=16384))
    runner.write(package/'selection.json', dict(specification='synthetic; no physical sampling'))
    plan['selection_sha256'] = runner.sha(package/'selection.json')
    shutil.copy2(package/'region.json', package/'old-r5-region.json')
    plan['reference_region_sha256'] = runner.sha(package/'old-r5-region.json')
    runner.write(package/'native-partition-definition.json', dict(schema='contact-confirmation-native-partition-v1',
        reference_region_sha256=plan['reference_region_sha256'], region_sha256=plan['region_sha256'],
        native_definition_sha256=plan['native_definition_sha256']))
    plan['supplemental_definition_sha256'] = runner.sha(package/'native-partition-definition.json')
    runner.write(package/'plan.json', plan); refreeze(package)
    return repo,binary,bundle,package,pins


def refreeze(package):
    (package/'freeze.json').unlink(missing_ok=True)
    runner.write(package/'freeze.json', dict(files=runner.file_hashes(package)))


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo,self.binary,self.bundle,self.package,pins = fixture(self.root)
        patches = patch.multiple(runner, ROOT=self.repo, PINS=pins, RETAINED_GUIDE_SHA256=runner.sha(self.package/'retained-guide.json'))
        patches.start(); self.addCleanup(patches.stop)

    def freeze(self):
        out=self.root/'confirmation'
        runner.freeze(out,self.binary,self.bundle,self.package)
        return out

    def test_twenty_fresh_populations_exact_control_laws_and_partition(self):
        out=self.freeze(); p=runner.validate(out)
        self.assertEqual(len(p['jobs']),20)
        self.assertEqual(p['total_unconditional_draws'],2228224)
        self.assertEqual(sum(j['samples'] for j in p['jobs']),2228224)
        self.assertEqual(len(set(j['seed'] for j in p['jobs'])),20)
        self.assertEqual(p['maximum_physical_workers'],8)
        self.assertEqual(p['decision_arms'],['bank','wide','alpha02','intensity256'])
        for arm in p['arms']:
            manifest=runner.read(out/arm['id']/'manifest.json')
            self.assertEqual(manifest['lambda_ratio'],arm['lambda_ratio'])
            self.assertEqual(manifest['jobs'][0]['command'][-1],str(arm['lambda_ratio']))
            self.assertEqual(manifest['jobs'][0]['samples'],32768 if arm['id']=='small' else 131072)
            guide=runner.read(out/arm['id']/'provenance/importance-guide.json')
            self.assertEqual(guide['defensive_uniform_shell_probability'],arm['alpha'])
            self.assertEqual(len(guide['gaussian_components']),80)
        self.assertEqual(runner.sha(out/p['reference_region']),p['reference_region_sha256'])
        self.assertEqual(runner.sha(out/p['supplemental_definition']),p['supplemental_definition_sha256'])

    def test_full_density_lower_bound_and_defensive_normalization(self):
        old=runner.read(self.package/'guide-bank.json')
        # Retained 56 components can be recovered without changing their relative weights.
        baseline=copy.deepcopy(old);baseline['gaussian_components']=baseline['gaussian_components'][:56]
        for c in baseline['gaussian_components']:c['weight']*=2
        rng=np.random.default_rng(197);u=rng.normal(size=(200,6))*4
        qnew=log_proposal(u,old);qold=log_proposal(u,baseline)
        self.assertTrue(np.all(qnew >= qold-np.log(2)-1e-12))
        for name in ('bank','wide','alpha02'):
            guide=runner.read(self.package/f'guide-{name}.json')
            self.assertAlmostEqual(sum(c['weight'] for c in guide['gaussian_components']),1.)
        self.assertEqual(runner.read(self.package/'guide-alpha02.json')['gaussian_components'],old['gaussian_components'])

    def test_control_density_and_training_slots_cannot_change(self):
        planpath=self.package/'plan.json';initial=runner.read(planpath)
        mutations=[]
        p=copy.deepcopy(initial);p['anchor_metadata'][56]['class']='native';mutations.append(p)
        p=copy.deepcopy(initial);p['anchor_metadata'][56]['seed']=runner.SEEDS[0];mutations.append(p)
        p=copy.deepcopy(initial);p['reference_region_sha256']='0'*64;mutations.append(p)
        for changed in mutations:
            runner.write(planpath,changed);refreeze(self.package)
            with self.assertRaises(ValueError):self.freeze()
            self.assertFalse((self.root/'confirmation').exists())
        runner.write(planpath,initial)
        guide=runner.read(self.package/'guide-alpha02.json');guide['gaussian_components'][0]['mean'][0]+=.1
        runner.write(self.package/'guide-alpha02.json',guide)
        initial['guide_sha256']['alpha02']=runner.sha(self.package/'guide-alpha02.json')
        runner.write(planpath,initial);refreeze(self.package)
        with self.assertRaisesRegex(ValueError,'Gaussian law'):self.freeze()

    def test_frozen_allocation_and_no_retry(self):
        out=self.freeze();runner.preflight(out)
        protocol=runner.read(out/'protocol.json');protocol['jobs'][0]['samples']+=1
        runner.write(out/'protocol.json',protocol)
        with self.assertRaisesRegex(ValueError,'Frozen file changed'):runner.validate(out)
        with self.assertRaisesRegex(ValueError,'Fresh campaign'):self.freeze()

    def test_existing_outputs_and_runtime_changes_fail_closed(self):
        out=self.freeze()
        with patch.object(runner,'runtime',return_value={'changed':True}):
            with self.assertRaisesRegex(ValueError,'runtime'):runner.preflight(out)
        (out/'bank/logs/preserve.log').write_text('preserve')
        with self.assertRaisesRegex(ValueError,'Existing outputs'):runner.preflight(out)
        self.assertEqual((out/'bank/logs/preserve.log').read_text(),'preserve')

    def test_population_law_checks_control_specific_alpha_and_intensity(self):
        out=self.freeze();p=runner.validate(out)
        for index in (0,12,16):
            job=p['jobs'][index];arm=p['arms'][index//4];d=Path(job['directory']);a=out/arm['id']/'provenance'
            (d/'provenance').mkdir(parents=True)
            m=runner.read(out/arm['id']/'manifest.json')
            pm=dict(schema=runner.POPULATION_SCHEMA,samples=job['samples'],seed=job['seed'],cloud_replicates=2,
                activity=.035,lambda_ratio=arm['lambda_ratio'],importance_uniform_probability=arm['alpha'],
                importance_component_count=80,proposal_density_measure=runner.DENSITY_MEASURE,
                executable_sha256=p['binary_sha256'],source_bundle_sha256=p['source_bundle_sha256'],
                minimum_latent_radius=0.,latent_radius=4.,minimum_original_q=0.,maximum_original_q=None,
                minimum_original_q_inclusive=True,maximum_original_q_inclusive=True,
                physical_fixed_neighbors=runner.read(a/'config.json')['fixed_poses'],chart_anchor=runner.read(a/'region.json')['fixed_neighbor'])
            pm['lambda']=.035*arm['lambda_ratio']
            for key in ('region_sha256','shape_sha256','config_sha256','importance_guide_sha256'):pm[key]=m[key]
            for target,source in [('input-config.json','config.json'),('region.json','region.json'),('shape.json','shape.json'),
                ('importance-guide.json','importance-guide.json'),('source-bundle.json','source-bundle.json')]:
                shutil.copy2(a/source,d/'provenance'/target)
            (d/'samples.jsonl').write_text('synthetic fixture; no samples\n')
            summary=dict(complete=True,manifest=pm,samples=job['samples'],shell_rejected=12,
                samples_sha256=runner.sha(d/'samples.jsonl'),sampler_cpu_seconds=1.,
                estimates={name:dict(draws=job['samples']) for name in ('region','hard_region')})
            runner.write(d/'manifest.json',pm);runner.write(d/'summary.json',summary)
            self.assertEqual(runner.verify_output(out,p,job)['shell_rejected'],12)
            wrong=copy.deepcopy(pm);wrong['importance_uniform_probability']=.7
            summary['manifest']=wrong;runner.write(d/'manifest.json',wrong);runner.write(d/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'Population law'):runner.verify_output(out,p,job)
            summary['manifest']=pm;summary['estimates']['region']['draws']-=12
            runner.write(d/'manifest.json',pm);runner.write(d/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'denominator'):runner.verify_output(out,p,job)

    def test_reused_reviewed_source_is_independent_of_new_working_code(self):
        reviewed=self.root/'reviewed';shutil.copytree(self.repo,reviewed)
        (self.repo/'src/synthetic.rs').write_text('new independent SMC module')
        with self.assertRaisesRegex(ValueError,'Reviewed source differs'):self.freeze()
        out=self.root/'confirmation'
        runner.freeze(out,self.binary,self.bundle,self.package,reviewed)
        runner.validate(out)


if __name__ == '__main__':unittest.main()
