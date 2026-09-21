"""Synthetic source/package/output bindings; never execute a physical sampler."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_entry_shell_reference_campaign as runner


def fixture(root):
    repo = root/'repo'; (repo/'src').mkdir(parents=True); (repo/'vendor').mkdir()
    (repo/'tools').symlink_to(runner.ROOT/'tools', target_is_directory=True)
    for name in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md', 'src/synthetic.rs'):
        (repo/name).write_text('// synthetic source '+name+'\n')
    source = {name: dict(text=(repo/name).read_text(), sha256=runner.sha(repo/name))
              for name in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md', 'src/synthetic.rs')}
    bundle = root/'source-bundle.json'; runner.write(bundle, dict(schema=1, files=source))
    binary = root/'synthetic-binary'; binary.write_bytes(b'not a physical executable\n'+bundle.read_bytes()); binary.chmod(0o755)
    package = root/'package'; (package/'inputs/native-region').mkdir(parents=True)
    pose = lambda x: dict(position=[x, 0., 0.], orientation=[1., 0., 0., 0.])
    shape = dict(atoms=[dict(center=[0., 0., 0.], radius=.5)])
    runner.write(package/'inputs/tetramer-shape.json', shape)
    shape_sha = runner.sha(package/'inputs/tetramer-shape.json')
    cfg = dict(shape=str(package/'inputs/tetramer-shape.json'), fixed_poses=[pose(0.), pose(10.)],
               capture_center=[0., 0., 0.], capture_radius=170., reservoir_density=.035,
               depletant_radius=1.5, poisson_lambda_ratio=64., metadata=dict(physical_sphere_radius_A=223.32617672378387))
    runner.write(package/'config.json', cfg); runner.write(package/'inputs/source-config.json', cfg)
    region = dict(fixed_neighbor=pose(10.), physical_fixed_neighbors=cfg['fixed_poses'],
        capture_center=cfg['capture_center'], capture_radius=170., activity=.035, depletant_radius=1.5,
        physical_metric=cfg['metadata'], shape_sha256=shape_sha, minimum_mahalanobis_radius=0.,
        mahalanobis_radius=4., minimum_original_q=0., minimum_original_q_inclusive=True,
        gaussian_chart=dict(schema='weighted-pose-mixture-v1', coordinate_convention='anchor-body-relative',
            angular_length=1., shape_sha256=shape_sha, weights=[1.],
            anchors=[dict(position=[3., 0., 0.], rotation=np.eye(3).tolist())], means=[[0.]*6],
            covariances=[(.04*np.eye(6)).tolist()]))
    runner.write(package/'region.json', region)
    guide = dict(schema=runner.GUIDE_SCHEMA, region_sha256=runner.sha(package/'region.json'),
        defensive_uniform_shell_probability=.2,
        entries=[dict(moving_member=[0., 0., 0.], target_world_member=[13., 0., 0.],
                      inner_radius=2., outer_radius=2.+width, weight=1/24)
                 for _ in range(8) for width in (.02, .1, .5)])
    runner.write(package/'guide.json', guide)
    native = 'inputs/native-region/definition.json'; runner.write(package/native, dict(diagnostic_only=True))
    plan = dict(schema='mobile-threshold-entry-shell-preparation-v1', complete=True, launched=False,
        config='config.json', region='region.json', guide='guide.json', shape_sha256=shape_sha,
        source_config_sha256=runner.sha(package/'inputs/source-config.json'), native_definition=native,
        native_definition_sha256=runner.sha(package/native), allocation=copy.deepcopy(runner.ALLOCATION))
    plan.update({name+'_sha256': runner.sha(package/(name+'.json')) for name in ('config', 'region', 'guide')})
    runner.write(package/'plan.json', plan)
    runner.write(package/'freeze.json', dict(files=runner.file_hashes(package)))
    pins = {name: runner.sha(package/name) for name in runner.PINS}
    return repo, binary, bundle, package, pins


class EntryShellBindings(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo, self.binary, self.bundle, self.package, pins = fixture(self.root)
        self.patch = patch.multiple(runner, ROOT=self.repo, PINS=pins)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def freeze(self):
        out = self.root/'campaign'
        runner.freeze(out, self.binary, self.bundle, self.package)
        return out

    def test_fixed_budget_exact_guide_target_and_current_closure(self):
        out = self.freeze(); p = runner.check(out); m = runner.read(out/'manifest.json')
        self.assertEqual(p['total_unconditional_draws'], 65536)
        self.assertEqual([j['seed'] for j in m['jobs']], [130101010, 130102019, 130103028, 130104037])
        self.assertEqual([j['samples'] for j in m['jobs']], [16384]*4)
        self.assertEqual(m['workers'], 4); self.assertEqual(m['schema'], runner.CAMPAIGN_SCHEMA)
        for name, source in [('region.json', 'region.json'), ('importance-guide.json', 'guide.json')]:
            self.assertEqual((out/'provenance'/name).read_bytes(), (self.package/source).read_bytes())
        cfg = runner.read(out/'provenance/config.json'); original = runner.read(self.package/'config.json')
        self.assertEqual(cfg.pop('shape'), str(out/'provenance/shape.json')); original.pop('shape')
        self.assertEqual(cfg, original)
        self.assertIn('entry_shell_proposal.py', p['python_sources'])
        self.assertIn('analyze_latent_region.py', p['python_sources'])
        self.assertIn('normalizer_proposal_density.py', p['python_sources'])
        self.assertEqual(set(p['rust_sources']), set(runner.read(self.bundle)['files']))
        for job in m['jobs']:
            self.assertEqual(job['command'], runner.command(out/'provenance', job))
            self.assertEqual(job['command'].count('--importance-guide'), 1)
            self.assertNotIn('--target-region', job['command'])
        # Renaming the entry point to controller.py must not alter closure identity.
        with patch.object(runner, '__file__', str(out/'provenance/controller.py')):
            runner.check(out)

    def test_embedding_current_source_and_bundle_membership_are_required(self):
        self.binary.write_bytes(b'wrong executable'); self.binary.chmod(0o755)
        with self.assertRaisesRegex(ValueError, 'embedded'): self.freeze()
        self.binary.write_bytes(self.bundle.read_bytes()); self.binary.chmod(0o755)
        (self.repo/'src/synthetic.rs').write_text('changed after build')
        with self.assertRaisesRegex(ValueError, 'Reviewed source differs'): self.freeze()
        bundle = runner.read(self.bundle); bundle['files']['src/synthetic.rs']['text'] = 'changed after build'
        runner.write(self.bundle, bundle); self.binary.write_bytes(self.bundle.read_bytes())
        with self.assertRaisesRegex(ValueError, 'text/hash'): self.freeze()

    def test_stale_package_and_physical_target_mismatch_fail_before_freeze(self):
        guide = self.package/'guide.json'; guide.write_text(guide.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'preparation changed'): self.freeze()
        self.assertFalse((self.root/'campaign').exists())

    def test_source_config_shape_only_rule_and_no_q_filter(self):
        # Rehash synthetic metadata to reach semantic checks independently of pins.
        for kind in ('bath', 'q', 'anchor'):
            with self.subTest(kind=kind):
                with tempfile.TemporaryDirectory() as tmp:
                    repo, binary, bundle, package, _ = fixture(Path(tmp))
                    cfg = runner.read(package/'config.json'); region = runner.read(package/'region.json')
                    if kind == 'bath': cfg['reservoir_density'] = .04
                    if kind == 'q': region['minimum_original_q'] = 1.
                    if kind == 'anchor': region['fixed_neighbor'] = cfg['fixed_poses'][0]
                    runner.write(package/'config.json', cfg); runner.write(package/'region.json', region)
                    plan = runner.read(package/'plan.json')
                    for name in ('config', 'region'): plan[name+'_sha256'] = runner.sha(package/(name+'.json'))
                    guide = runner.read(package/'guide.json'); guide['region_sha256'] = plan['region_sha256']
                    runner.write(package/'guide.json', guide); plan['guide_sha256'] = runner.sha(package/'guide.json')
                    runner.write(package/'plan.json', plan)
                    (package/'freeze.json').unlink(); runner.write(package/'freeze.json', dict(files=runner.file_hashes(package)))
                    with patch.object(runner, 'PINS', {n: runner.sha(package/n) for n in runner.PINS}):
                        with self.assertRaises(ValueError): runner.validate_package(package)

    def test_future_source_mutation_is_irrelevant_but_archived_mutation_blocks(self):
        out = self.freeze(); (self.repo/'src/synthetic.rs').write_text('later development')
        runner.validate(out)  # Standalone archived provenance, never mutable current Rust.
        (out/'provenance/entry_shell_proposal.py').write_text('tampered observer')
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'): runner.validate(out)

    def population_fixture(self, out):
        m = runner.read(out/'manifest.json'); job = m['jobs'][0]; d = Path(job['directory']); (d/'provenance').mkdir(parents=True)
        archive = out/'provenance'; region = runner.read(archive/'region.json')
        pm = dict(schema=runner.POPULATION_SCHEMA, samples=job['samples'], seed=job['seed'], cloud_replicates=2,
            activity=.035, **{'lambda': .035*64.}, lambda_ratio=64., guide_schema=runner.GUIDE_SCHEMA,
            proposal_kind=runner.PROPOSAL_KIND, importance_uniform_probability=.2, importance_component_count=24,
            proposal_density_measure=runner.DENSITY_MEASURE, physical_fixed_neighbors=runner.read(archive/'config.json')['fixed_poses'],
            chart_anchor=region['fixed_neighbor'], minimum_latent_radius=0., latent_radius=4., minimum_original_q=0.,
            maximum_original_q=None, minimum_original_q_inclusive=True, maximum_original_q_inclusive=True)
        pm.update({k: m[k] for k in ('config_sha256', 'region_sha256', 'shape_sha256', 'importance_guide_sha256')})
        pm.update(source_bundle_sha256=m['archive_sha256']['source-bundle.json'], executable_sha256=m['archive_sha256']['latent-region-normalizer'])
        for name, source in [('input-config.json', 'config.json'), ('region.json', 'region.json'), ('shape.json', 'shape.json'),
                             ('source-bundle.json', 'source-bundle.json'), ('importance-guide.json', 'importance-guide.json')]:
            shutil.copy2(archive/source, d/'provenance'/name)
        (d/'samples.jsonl').write_text('synthetic rows: raw auditor mocked in these tests\n')
        runner.write(d/'manifest.json', pm)
        summary = dict(complete=True, samples=job['samples'], manifest=pm, samples_sha256=runner.sha(d/'samples.jsonl'),
                       estimates={k: dict(draws=job['samples']) for k in ('region', 'hard_region')}, sampler_cpu_seconds=1.)
        runner.write(d/'summary.json', summary)
        return m, job, d, pm, summary

    def test_population_requires_v2_exact_guide_and_unconditional_denominator(self):
        out = self.freeze(); m, job, d, pm, summary = self.population_fixture(out)
        result = runner.verify_output(out, m, job)
        self.assertEqual(result['samples_sha256'], runner.sha(d/'samples.jsonl'))
        for field, value in [('schema', 'importance-latent-region-normalizer-v1'), ('importance_component_count', 8),
                             ('proposal_kind', 'gaussian'), ('seed', job['seed']+1), ('minimum_original_q', 1.)]:
            changed = copy.deepcopy(pm); changed[field] = value
            runner.write(d/'manifest.json', changed); summary['manifest'] = changed; runner.write(d/'summary.json', summary)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'law/input'): runner.verify_output(out, m, job)
        runner.write(d/'manifest.json', pm); summary['manifest'] = pm; summary['estimates']['region']['draws'] -= 1
        runner.write(d/'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'denominator'): runner.verify_output(out, m, job)

    def test_audit_binds_all_rows_and_selected_entry_support_counts(self):
        out = self.freeze(); m, job, d, _, _ = self.population_fixture(out)
        job['output'] = runner.verify_output(out, m, job); n = job['samples']
        counts = {'uniform-shell': 100, 'entry-shell': n-100}
        audit = dict(draws=n, branch_counts=counts, entry_shell_component_counts=[n-100]+[0]*23)
        p = dict(id=job['id'], seed=job['seed'], estimate=dict(draws=n), hard_region=dict(draws=n),
                 samples_sha256=job['output']['samples_sha256'], importance_sampling_audit=audit)
        a = dict(region_sha256=m['region_sha256'], estimate=dict(draws=n), hard_region=dict(draws=n),
            independently_reconstructed_poses=n, physical_fixed_neighbors=runner.read(out/'provenance/config.json')['fixed_poses'],
            populations=[p], importance_sampling=dict(guide_sha256=m['importance_guide_sha256'], uniform_shell_probability=.2,
                entry_shell_component_count=24, guide_schema=runner.GUIDE_SCHEMA, proposal_kind=runner.PROPOSAL_KIND,
                density_measure=runner.DENSITY_MEASURE, draws=n, branch_counts=counts))
        runner.verify_assessment(out, m, a, [job])
        for kind in ('branch', 'component', 'rows', 'population'):
            changed = copy.deepcopy(a)
            if kind == 'branch': changed['importance_sampling']['branch_counts']['entry-shell'] -= 1
            if kind == 'component': changed['populations'][0]['importance_sampling_audit']['entry_shell_component_counts'][0] -= 1
            if kind == 'rows': changed['independently_reconstructed_poses'] -= 1
            if kind == 'population': changed['populations'] = []
            with self.subTest(kind=kind), self.assertRaises(ValueError): runner.verify_assessment(out, m, changed, [job])
        (d/'samples.jsonl').write_text('different saved rows')
        with self.assertRaisesRegex(ValueError, 'row binding'): runner.verify_assessment(out, m, a, [job])


if __name__ == '__main__': unittest.main()
