"""Nonphysical stage uncertainty and authenticated-artifact failure controls."""
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import t as student_t

from analyze_r4_smc_control import Ledger, read, sha, write
from compare_full_vessel_stage import (CLASSES, KINDS, PRIMARY, SUPPORTS, agreement,
    arm_statistics, check_mass, compare, frozen_artifact, run, validate_manifest, validate_partition)
from prepare_full_vessel_comparison import jobs_for, STAGES
from vessel_contact_partition import Mass, partition


def estimate_fixture(scale=1., zero_native=False, n=65536, repeats=1):
    accumulators = {name: {kind: Mass() for kind in KINDS} for name in CLASSES}
    # Include overlapping pocket witnesses and an explicit invalid attempt;
    # the remaining n-13 zero attempts are also retained in the denominator.
    for i in range(12 * repeats):
        native = i % 3 == 0
        valid = not (native and zero_native)
        labels = partition(valid, {name: valid and j == i % 3 for j, name in enumerate(PRIMARY)},
                           {name: bool((i // 3) & (1 << (j % 2))) for j, name in enumerate(SUPPORTS)})
        for name, selected in labels.items():
            if selected:
                for kind, factor in (('Qz', scale), ('Q0', 1.)):
                    accumulators[name][kind].add(math.log(factor * (1 + i % 3)))
    estimates = {name: {kind: m.result(n) for kind, m in kinds.items()} for name, kinds in accumulators.items()}
    return dict(schema='full-vessel-contact-partition-v1', complete=True, samples=n,
                invalid_draws=n - repeats * (8 if zero_native else 12), estimates=estimates)


def population_fixture(stage='standard', scales=(1., 2., 3., 4.)):
    n = dict(STAGES)[stage]
    result = []
    for ai, arm in enumerate(('vessel', 'half_mixture')):
        for i, scale in enumerate(scales):
            p = estimate_fixture(scale, n=n)
            result.append(dict(id=f'{stage}-{arm}-r{i:02}', arm=arm, stage=stage, population=i,
                               seed=10 + ai * 4 + i, samples=n, estimates=p['estimates']))
    return result


def freeze(root):
    write(root / 'freeze.json', dict(files={str(p.relative_to(root)): sha(p) for p in root.rglob('*')
                                          if p.is_file() and p != root / 'freeze.json'}))


def artifact_fixture(prep):
    """Frozen synthetic records only; never executes the protein binary/audits."""
    common, inputs = prep / 'common', prep / 'inputs'
    common.mkdir(parents=True); (inputs / 'native-region/inputs/source').mkdir(parents=True)
    for name in ('audit_full_vessel_baseline.py', 'audit_full_vessel_latent.py', 'vessel_contact_partition.py'):
        (common / name).write_text('# Synthetic archived analysis source\n')
    (common / 'basin-normalizer').write_bytes(b'synthetic inert binary')
    write(common / 'source-bundle.json', dict(synthetic=True))
    write(inputs / 'config.json', dict(shape=str(inputs / 'shape.json'), target_region=None))
    for name in ('model.json', 'shape.json', 'guide.json', *(name + '.json' for name in SUPPORTS)):
        write(inputs / name, dict(synthetic=name))
    runtime = inputs / 'native-region/inputs/source/native_contact_regions.py'
    runtime.write_text('# Synthetic native observer\n')
    native = inputs / 'native-region/definition.json'
    definition = dict(input_sha256={'source/native_contact_regions.py': sha(runtime)})
    write(native, definition)
    jobs = jobs_for(prep)
    plan = dict(jobs=jobs, sources={p.name: sha(p) for p in common.glob('*.py')},
                input_sha256={str(p.relative_to(inputs)): sha(p) for p in inputs.rglob('*') if p.is_file()},
                native_definition_sha256=sha(native), binary_sha256=sha(common / 'basin-normalizer'),
                source_bundle_sha256=sha(common / 'source-bundle.json'),
                physical=dict(activity=.035, lambda_ratio=64, wall_center=[0., 0., 0.], wall_radius=223.32617672378387))
    write(prep / 'plan.json', plan); freeze(prep)
    for job in jobs:
        if job['stage'] != 'standard':
            continue
        root, audit_root, part_root = (Path(job[k]) for k in ('directory', 'audit_directory', 'partition_directory'))
        for directory in (root, audit_root, part_root):
            (directory / 'provenance').mkdir(parents=True)
        manifest = dict(samples=job['samples'], seed=job['seed'], cloud_replicates=2, covariance_scale=1.,
                        uniform_probability=.1, proposal_anchor_index=None, executable_sha256=plan['binary_sha256'],
                        source_bundle_sha256=plan['source_bundle_sha256'], activity=.035,
                        atomic_wall=dict(center=[0., 0., 0.], radius=plan['physical']['wall_radius']), bath_wall_permeable=True,
                        physical_fixed_neighbor_count=2, pose_proposal_schema=3, base_component_count=178,
                        virtual_component_count=328, proposal_model_kind='reciprocal-pose-mixture-v1', schema=4)
        manifest['lambda'] = .035 * 64
        pairs = [('input-config.json', 'config.json', 'config_sha256'), ('model.json', 'model.json', 'model_sha256'),
                 ('shape.json', 'shape.json', 'shape_sha256')]
        if job['arm'] == 'half_mixture':
            manifest.update(schema=5, outer_mixture_schema='full-vessel-latent-half-mixture-v1', outer_vessel_probability=.5,
                            latent_reference_ball_is_target_restriction=False, latent_source_capture={'restricts_target': False},
                            latent_defensive_uniform_probability=.5, latent_gaussian_component_count=80,
                            density_measure='Lebesgue center volume times normalized SO(3) Haar measure')
            pairs += [('latent-region.json', 'current_R4.json', 'latent_region_sha256'), ('latent-guide.json', 'guide.json', 'latent_guide_sha256')]
        for saved, filename, key in pairs:
            (root / 'provenance' / saved).write_bytes((inputs / filename).read_bytes())
            manifest[key] = sha(inputs / filename)
        (root / 'provenance/source-bundle.json').write_bytes((common / 'source-bundle.json').read_bytes())
        (root / 'config.json').write_bytes((inputs / 'config.json').read_bytes())
        (root / 'samples.jsonl').write_text('Synthetic raw bytes; no physics or classification replay.\n')
        p = estimate_fixture()
        write(root / 'manifest.json', manifest)
        write(root / 'summary.json', dict(complete=True, manifest=manifest, numerical_nulls=0, samples=job['samples'],
              estimates={name: {'log_normalizer': p['estimates']['total'][kind]['logQ']}
                         for name, kind in (('total', 'Qz'), ('hard_total', 'Q0'))}))
        audit_entry = 'audit_full_vessel_baseline.py' if job['arm'] == 'vessel' else 'audit_full_vessel_latent.py'
        (audit_root / 'provenance' / audit_entry).write_bytes((common / audit_entry).read_bytes())
        (audit_root / 'labels.jsonl').write_text('Synthetic bound native/contact labels\n')
        paths = [p for p in root.rglob('*') if p.is_file()] + [common / audit_entry, native, runtime]
        audit = dict(schema='full-vessel-baseline-audit-v1' if job['arm'] == 'vessel' else 'full-vessel-latent-audit-v1',
                     complete=True, population=str(root), manifest=manifest, estimates=p['estimates'],
                     raw_sample_binding={'sha256': sha(root / 'samples.jsonl')},
                     native_binding=dict(definition=str(native), definition_sha256=sha(native),
                                         input_sha256=definition['input_sha256'], runtime_sha256=sha(runtime)))
        if job['arm'] == 'vessel':
            audit['reporting_guide_binding'] = dict(region=str(inputs / 'current_R4.json'), guide=str(inputs / 'guide.json'),
                  region_sha256=sha(inputs / 'current_R4.json'), guide_sha256=sha(inputs / 'guide.json'), restricts_target=False, affects_proposal=False)
            for label, name in (('region', 'current_R4.json'), ('guide', 'guide.json')):
                paths.append(inputs / name)
                (audit_root / 'provenance' / ('reporting-' + label + '.json')).write_bytes((inputs / name).read_bytes())
        audit['source_sha256'] = {str(path): sha(path) for path in paths}
        write(audit_root / 'analysis.json', audit); freeze(audit_root)
        (part_root / 'labels.jsonl').write_text('Synthetic exhaustive pocket labels\n')
        (part_root / 'provenance/vessel_contact_partition.py').write_bytes((common / 'vessel_contact_partition.py').read_bytes())
        paths = [audit_root / 'analysis.json', audit_root / 'labels.jsonl', root / 'manifest.json', root / 'config.json',
                 root / 'samples.jsonl', common / 'vessel_contact_partition.py', *(inputs / (s + '.json') for s in SUPPORTS)]
        p.update(population=str(root), audit=str(audit_root / 'analysis.json'), manifest=manifest,
                 labels_sha256=sha(part_root / 'labels.jsonl'), native_definition_sha256=sha(native),
                 region_paths={name: str(inputs / (name + '.json')) for name in SUPPORTS},
                 input_sha256={str(path): sha(path) for path in paths})
        write(part_root / 'analysis.json', p); freeze(part_root)
    return plan


class StageStatisticsTests(unittest.TestCase):
    def test_complete_exhaustive_moments_and_attempted_denominator(self):
        p = estimate_fixture(); validate_partition(p, 65536)
        self.assertAlmostEqual(p['estimates']['total']['Qz']['logQ'], math.log(24 / 65536))
        for mutation in ('denominator', 'square_sum', 'invalid', 'missing'):
            bad = copy.deepcopy(p)
            if mutation == 'denominator': bad['estimates']['total']['Qz']['draws'] = 12
            elif mutation == 'square_sum': bad['estimates']['outside_measured_pockets']['Qz']['log_sum_squared_weights'] += 1
            elif mutation == 'invalid': bad['invalid_draws'] -= 1
            else: del bad['estimates']['unbound_no_native_entry:outside_measured_pockets']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validate_partition(bad, 65536)

    def test_joint_square_and_maximum_constraints_reject_impossible_weights(self):
        impossible = dict(draws=4, nonzero=4, logQ=0., log_sum_squared_weights=math.log(4),
                          log_max_weight=math.log(4), ess=4., max_fraction=1.)
        with self.assertRaisesRegex(ValueError, 'joint squared-weight'):
            check_mass(impossible, 4)
        # The opposite violation has a maximum too small for the square sum.
        impossible.update(log_sum_squared_weights=math.log(16), log_max_weight=0., ess=1., max_fraction=.25)
        with self.assertRaisesRegex(ValueError, 'joint squared-weight'):
            check_mass(impossible, 4)

    def test_linear_mass_covariance_and_t3_contrast(self):
        pops = population_fixture()[:4]
        summary = arm_statistics(pops, 'Qz')
        x = np.arange(1., 5.) * 4 / 65536
        native = summary['estimates'][PRIMARY[0]]
        self.assertAlmostEqual(native['log_Q'], math.log(x.mean()))
        self.assertAlmostEqual(native['population_relative_SE'], x.std(ddof=1) / 2 / x.mean())
        self.assertAlmostEqual(native['log_mass_halfwidth_95'], student_t.ppf(.975, 3) * x.std(ddof=1) / 2 / x.mean())
        covariance = summary['covariance_of_mean']; i = CLASSES.index(PRIMARY[0]); j = CLASSES.index(PRIMARY[1])
        expected = np.var(x, ddof=1) / 4 / x.mean() ** 2
        self.assertAlmostEqual(covariance['log_mass_delta_method_matrix'][i][j], expected)
        contrast = summary['contrasts']['native_vs_contact_noentry']
        self.assertAlmostEqual(contrast['beta_F_native_minus_noentry'], math.log(2))
        self.assertAlmostEqual(contrast['population_SE'], 0., places=14)
        self.assertAlmostEqual(summary['contrasts']['native_vs_all_noentry']['beta_F_native_minus_noentry'], math.log(5))

    def test_zero_population_and_unobserved_class_remain_in_inference(self):
        pops = population_fixture()[:4]
        pops[0]['estimates'] = estimate_fixture(zero_native=True)['estimates']
        summary = arm_statistics(pops, 'Qz')
        native = summary['estimates'][PRIMARY[0]]
        self.assertEqual(native['nonzero_populations'], 3)
        self.assertAlmostEqual(native['log_Q'], math.log((0 + 8 + 12 + 16) / 4 / 65536))
        for p in pops: p['estimates'] = estimate_fixture(zero_native=True)['estimates']
        summary = arm_statistics(pops, 'Qz')
        self.assertIsNone(summary['estimates'][PRIMARY[0]]['log_Q'])
        self.assertIn('unresolved', summary['contrasts']['native_vs_contact_noentry'])
        self.assertIsNone(summary['covariance_of_mean']['log_mass_delta_method_matrix'][CLASSES.index(PRIMARY[0])][0])

    def test_agreement_requires_both_bounds(self):
        for difference, se, expected in ((.1, .1, 'corroborated'), (.3, .2, 'unresolved'), (.1, .001, 'unresolved'), (.5, .01, 'material_disagreement')):
            item = agreement({'log_Q': difference, 'population_relative_SE': se}, {'log_Q': 0., 'population_relative_SE': 0.})
            self.assertEqual(item['state'], expected)
        self.assertEqual(agreement({'log_Q': None}, {'log_Q': 1.})['state'], 'unresolved')

    def test_all_observed_diagnostics_can_pass_without_unseen_mode_certificate(self):
        pops = population_fixture(scales=(1., 1., 1., 1.))
        for population in pops:
            population['estimates'] = estimate_fixture(repeats=100)['estimates']
        result = compare(pops, 'standard')
        self.assertTrue(result['diagnostics']['declared_primary_diagnostics_passed'])
        self.assertFalse(result['full_vessel_unseen_modes_certified'])
        self.assertFalse(result['assembly_stability_established'])
        native = result['arms']['vessel']['Qz']['estimates'][PRIMARY[0]]
        self.assertAlmostEqual(native['importance_ESS'], 1600.)
        self.assertAlmostEqual(native['largest_contribution'], 1 / 1600.)
        self.assertEqual(native['population_relative_SE'], 0.)

    def test_stage_no_pooling_seeds_and_diagnostic_not_certificate(self):
        pops = population_fixture(); result = compare(pops, 'standard')
        self.assertTrue(result['complete']); self.assertFalse(result['stages_pooled'])
        self.assertFalse(result['full_vessel_unseen_modes_certified']); self.assertFalse(result['assembly_stability_established'])
        self.assertFalse(result['diagnostics']['declared_primary_diagnostics_passed'])
        self.assertIn('outside_measured_pockets', result['between_arms']['Qz'])
        with self.assertRaises(ValueError): compare(pops + population_fixture('large'), 'standard')
        with self.assertRaises(ValueError): compare(pops, 'large')
        pops[4]['seed'] = pops[0]['seed']
        with self.assertRaises(ValueError): compare(pops, 'standard')


class StageAuthenticationTests(unittest.TestCase):
    def test_freeze_must_cover_required_and_actual_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); write(root / 'analysis.json', {'complete': True}); freeze(root)
            with self.assertRaisesRegex(ValueError, 'omitted'): frozen_artifact(Ledger(), root, ('analysis.json', 'labels.jsonl'))
            (root / 'extra').write_text('unfrozen')
            with self.assertRaisesRegex(ValueError, 'unfrozen'): frozen_artifact(Ledger(), root, ('analysis.json',))

    def test_frozen_pipeline_and_source_identity_fail_closed_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); prep = base / 'preparation'; plan = artifact_fixture(prep)
            with patch('compare_full_vessel_stage.validate_preparation', return_value=plan):
                result = run(prep, 'standard', base / 'comparison')
                self.assertTrue(result['complete']); self.assertEqual(result['preparation_sha256'], sha(prep / 'plan.json'))
                self.assertEqual(len(result['populations']), 8)
                self.assertTrue((base / 'comparison/freeze.json').is_file())
                first = plan['jobs'][0]
                archive = Path(first['audit_directory']) / 'provenance/audit_full_vessel_baseline.py'
                archive.write_text('# Changed source while retaining complete=true\n'); freeze(archive.parent.parent)
                with self.assertRaisesRegex(ValueError, 'Hash mismatch'): run(prep, 'standard', base / 'changed-source')
                self.assertFalse((base / 'changed-source').exists())

    def test_refrozen_primary_moment_disagreement_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); prep = base / 'preparation'; plan = artifact_fixture(prep); job = plan['jobs'][0]
            audit_root, part_root = Path(job['audit_directory']), Path(job['partition_directory'])
            original = read(audit_root / 'analysis.json')
            for field, value in (('draws', 12), ('nonzero', 11), ('ess', 10.), ('max_fraction', .2)):
                audit = copy.deepcopy(original)
                audit['estimates']['total']['Qz'][field] = value
                write(audit_root / 'analysis.json', audit); freeze(audit_root)
                part = read(part_root / 'analysis.json')
                part['input_sha256'][str(audit_root / 'analysis.json')] = sha(audit_root / 'analysis.json')
                write(part_root / 'analysis.json', part); freeze(part_root)
                out = base / ('changed-' + field)
                with self.subTest(field=field), patch('compare_full_vessel_stage.validate_preparation', return_value=plan):
                    with self.assertRaisesRegex(ValueError, 'Audited primary'):
                        run(prep, 'standard', out)
                self.assertFalse(out.exists())

    def test_executed_identity_is_checked_beyond_flags_and_changed_raw_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); prep = base / 'preparation'; plan = artifact_fixture(prep); job = plan['jobs'][0]
            manifest = read(Path(job['directory']) / 'manifest.json')
            for field, value in (('seed', -1), ('samples', 12), ('executable_sha256', 'wrong'),
                                 ('cloud_replicates', 1), ('atomic_wall', {'radius': 1}), ('uniform_probability', .5)):
                bad = dict(manifest, **{field: value})
                with self.subTest(field=field), self.assertRaises(ValueError): validate_manifest(plan, job, bad)
            (Path(job['directory']) / 'samples.jsonl').write_text('Replaced raw attempts\n')
            with patch('compare_full_vessel_stage.validate_preparation', return_value=plan):
                with self.assertRaisesRegex(ValueError, 'Hash mismatch'): run(prep, 'standard', base / 'changed-raw')
            self.assertFalse((base / 'changed-raw').exists())


if __name__ == '__main__':
    unittest.main()
