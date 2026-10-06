"""Synthetic bookkeeping controls; optional bounded saved-fixture equivalence."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import audit_native_class_vessel_streaming as stream


def manifest(schema=7, draws=7):
    value = dict(schema=schema, outer_mixture_schema=stream.reference.SCHEMA,
        latent_guide_schema=stream.reference.line.SCHEMA, outer_vessel_probability=.5,
        samples=draws, cloud_replicates=2,
        attempt_journal='attempts.jsonl; begin before each attempt; no retries', resume_supported=False,
        density_measure='Lebesgue center volume times normalized SO(3) Haar measure',
        latent_reference_ball_is_target_restriction=False,
        latent_source_capture=dict(restricts_target=False, conditions_guide=True), bath_wall_permeable=True)
    if schema == 8:
        value.update(pre_envelope_schema=7, vessel_uniform_schema='one-atom-wall-envelope-v1', vessel_uniform_envelope={})
    return value


def rows():
    values = []
    for i in range(7):
        valid = i not in (1, 4)
        logq = -.1*i
        first, second = .1*i, .1*i+.2
        value = dict(draw=i, pose={'fixture_index': i}, outer_branch='vessel' if i % 2 else 'latent',
            capture_valid=True, wall_valid=i != 1, hard_valid=valid,
            depletion_contact=bool(i % 2) if valid else None,
            latent_density=dict(in_reference_ball=i < 3, structural_zero=i > 5),
            log_proposal_density=logq,
            log_importance_weight=math.log((math.exp(first)+math.exp(second))/2)-logq if valid else None,
            log_hard_weight=-logq if valid else None,
            clouds=[dict(log_weight=first, raw_points=i), dict(log_weight=second, raw_points=i+1)] if valid else [])
        values.append(value)
    return values


def density_check(config, local, batch, vessel, guide):
    if [r['draw'] for r in batch] != list(range(len(batch))):
        raise ValueError('Local indexing differs')
    return dict(checked_attempts=len(batch), valid_outside_R4=sum(r['hard_valid'] and not r['latent_density']['in_reference_ball'] for r in batch),
        structural_zero_queries=sum(r['latent_density']['structural_zero'] for r in batch), exact_chart_seams=0,
        valid_outside_source_capture=0, maximum_log_density_error=0., maximum_interval_endpoint_error=0.,
        maximum_inverse_CDF_error=0., source_capture=2., vessel_capture=4.,
        chart_factor_validation={'passed': True}, scope='synthetic bookkeeping only',
        outer_branches=dict(stream.Counter(r['outer_branch'] for r in batch)))


def generation_check(config, local, batch, vessel):
    n = sum(r['outer_branch'] == 'vessel' for r in batch)
    return dict(checked_vessel_generation_rows=n, **({'scope':
        'Generation metadata and selected-anchor densities; RNG replay is a separate implementation obligation.'} if n else {}))


def primitive_check(local, summary, batch):
    counts = stream.row_counts(batch)
    if summary != dict(samples=len(batch), **counts):
        raise ValueError('Local denominator differs')
    return counts


class MetadataTests(unittest.TestCase):
    def test_schema7_and_schema8_contracts(self):
        for schema in (7, 8):
            m = manifest(schema)
            stream.validate_manifest(m, dict(complete=True, manifest=m, numerical_nulls=0, samples=7))

    def test_unknown_hidden_or_inconsistent_laws_rejected(self):
        bad = [dict(manifest(), schema=6), dict(manifest(), schema=True),
               dict(manifest(), vessel_uniform_schema='hidden'), dict(manifest(8), pre_envelope_schema=4),
               dict(manifest(), cloud_replicates=1), dict(manifest(), resume_supported=True),
               dict(manifest(), latent_reference_ball_is_target_restriction=True),
               dict(manifest(), latent_source_capture=dict(restricts_target=True, conditions_guide=True)),
               dict(manifest(), samples=False)]
        for m in bad:
            with self.subTest(m=m), self.assertRaises(ValueError):
                stream.validate_manifest(m, dict(complete=True, manifest=m, numerical_nulls=0, samples=m['samples']))

    def test_summary_denominator_and_completion_rejected(self):
        m = manifest()
        for changes in [dict(samples=6), dict(complete=False), dict(numerical_nulls=1), dict(manifest={})]:
            s = dict(complete=True, manifest=m, numerical_nulls=0, samples=7)
            s.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): stream.validate_manifest(m, s)

    def test_all_attempts_in_order_and_missing_tail_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'rows').write_text(''.join(json.dumps(r)+'\n' for r in rows()))
            journal = [dict(draw=i, state='begin') for i in range(7)]
            (root/'journal').write_text(''.join(json.dumps(r)+'\n' for r in journal))
            batches = list(stream.attempt_batches(root/'rows', root/'journal', 7, 3))
            self.assertEqual([len(b) for b in batches], [3, 3, 1])
            self.assertEqual([r for b in batches for r in b], rows())
            for bad in [journal[:-1], [journal[0]]*7, [dict(draw=False, state='begin')]+journal[1:]]:
                (root/'journal').write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.assertRaises(ValueError): list(stream.attempt_batches(root/'rows', root/'journal', 7, 3))


class BatchTests(unittest.TestCase):
    def apply(self, batch_size):
        r = rows()
        m = manifest()
        checks = stream.BatchChecks()
        with patch.object(stream.reference, 'check_rows', side_effect=density_check), \
             patch.object(stream.reference.vessel_reference, 'check_generation_metadata', side_effect=generation_check), \
             patch.object(stream.reference.vessel_reference, 'check_cloud_envelopes_and_counts', side_effect=primitive_check):
            for start in range(0, len(r), batch_size): checks.check({}, m, r[start:start+batch_size], None, None)
        checks.finish(m, dict(samples=7, **stream.row_counts(r)))
        self.assertEqual(r, rows())
        return checks

    def test_batch_sizes_preserve_complete_counters(self):
        first = self.apply(1)
        for size in (3, 64):
            current = self.apply(size)
            self.assertEqual(first.density, current.density)
            self.assertEqual(first.primitive, current.primitive)
            self.assertEqual(first.generation_count, current.generation_count)
            self.assertEqual(current.peak_rows, min(size, 7))

    def test_lost_global_attempt_or_counter_rejected(self):
        checks = self.apply(3)
        with self.assertRaises(ValueError): checks.finish(dict(samples=8), dict(samples=8, **stream.row_counts(rows())))
        for key in stream.row_counts(rows()):
            s = dict(samples=7, **stream.row_counts(rows()))
            s[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError): checks.finish(manifest(), s)

    def test_unhandled_density_fields_rejected(self):
        with patch.object(stream.reference, 'check_rows', return_value=dict(density_check({}, {}, rows(), None, None), unhandled=1)):
            with self.assertRaises(ValueError): stream.BatchChecks().check({}, manifest(), rows(), None, None)

    def test_changed_chart_validation_rejected(self):
        checks = stream.BatchChecks()
        calls = 0
        def changing(*args):
            nonlocal calls
            result = density_check(*args)
            result['chart_factor_validation'] = {'identity': calls}
            calls += 1
            return result
        with patch.object(stream.reference, 'check_rows', side_effect=changing), \
             patch.object(stream.reference.vessel_reference, 'check_generation_metadata', side_effect=generation_check), \
             patch.object(stream.reference.vessel_reference, 'check_cloud_envelopes_and_counts', side_effect=primitive_check):
            checks.check({}, manifest(), rows()[:3], None, None)
            with self.assertRaises(ValueError): checks.check({}, manifest(), rows()[3:], None, None)

    def test_original_index_partition_and_eager_moments_with_invalid_zeros(self):
        result = moment_control(rows())
        self.assertEqual(result['draws'], 7)
        self.assertEqual(result['invalid_zeros'], 2)
        self.assertEqual(result['regions']['total']['Qz']['nonzero'], 5)


def fixture(root):
    root.mkdir()
    p = root/'provenance'
    p.mkdir()
    text = 'inert synthetic audit fixture; no executable is run\n'
    bundle = dict(files={'fixture': dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())})
    for name in ['input-config.json', 'model.json', 'shape.json', 'latent-region.json', 'latent-guide.json']:
        stream.write(p/name, {})
    stream.write(p/'source-bundle.json', bundle)
    binary = root/'inert-binary'
    binary.write_bytes(b'never execute\n'+(p/'source-bundle.json').read_bytes())
    m = manifest()
    m['executable_sha256'] = stream.sha(binary)
    for name, key in [('input-config.json', 'config_sha256'), ('model.json', 'model_sha256'),
                      ('shape.json', 'shape_sha256'), ('latent-region.json', 'latent_region_sha256'),
                      ('latent-guide.json', 'latent_guide_sha256'), ('source-bundle.json', 'source_bundle_sha256')]:
        m[key] = stream.sha(p/name)
    stream.write(root/'manifest.json', m)
    (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows()))
    (root/'attempts.jsonl').write_text(''.join(json.dumps(dict(draw=i, state='begin'))+'\n' for i in range(7)))
    reducer = stream.RegionMoments(7)
    for row in rows(): reducer.add(row, row['hard_valid'])
    totals = reducer.result()
    stream.write(root/'summary.json', dict(complete=True, manifest=m, samples=7, numerical_nulls=0,
        samples_sha256=stream.sha(root/'samples.jsonl'), attempts_sha256=stream.sha(root/'attempts.jsonl'),
        estimates=dict(total={'log_normalizer': totals['Qz']['logQ']}, hard_total={'log_normalizer': totals['Q0']['logQ']}),
        **stream.row_counts(rows())))
    return binary


class FakeWall:
    def __init__(self, *args): self.checked = self.rejected = 0
    def check(self, row):
        self.checked += 1
        self.rejected += int(row['capture_valid'] and not row['wall_valid'])
    def result(self): return dict(checked_poses=self.checked, wall_rejected_inside_capture=self.rejected)


class FakeContact:
    def __init__(self, *args): self.checked = 0
    def classify(self, pose, **kwargs):
        self.checked += 1
        r = rows()[pose['fixture_index']]
        return dict(core_disjoint=r['hard_valid'], exclusion_contact=r['depletion_contact'], near_core_boundary=False)
    def report(self): return dict(attempts=self.checked, synthetic=True)


class EntryTests(unittest.TestCase):
    def run_audit(self, pop, out, binary, size, checker=density_check):
        with patch.object(stream, 'context', return_value=(dict(fixed_poses=[], depletant_radius=1.), {}, None, SimpleNamespace(envelope=None), {})), \
             patch.object(stream, 'WallOracle', FakeWall), \
             patch.object(stream.reference.vessel_reference, 'PrunedExclusionContact', FakeContact), \
             patch.object(stream.reference, 'check_rows', side_effect=checker), \
             patch.object(stream.reference.vessel_reference, 'check_generation_metadata', side_effect=generation_check), \
             patch.object(stream.reference.vessel_reference, 'check_cloud_envelopes_and_counts', side_effect=primitive_check):
            return stream.audit(pop, out, binary, size, synthetic=True)

    def test_complete_entry_point_and_frozen_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pop = root/'population'
            binary = fixture(pop)
            results = [self.run_audit(pop, root/f'audit{size}', binary, size) for size in (1, 3, 64)]
            for size, result in zip((1, 3, 64), results):
                self.assertTrue(result['complete'])
                self.assertEqual(result['geometry_sha256'], results[0]['geometry_sha256'])
                self.assertEqual(result['estimates'], results[0]['estimates'])
                self.assertEqual(result['estimates']['total']['Qz']['draws'], 7)
                self.assertEqual(result['estimates']['total']['Qz']['nonzero'], 5)
                directory = root/f'audit{size}'
                frozen = stream.read(directory/'freeze.json')['files']
                self.assertTrue({'analysis.json', 'geometry.jsonl', 'status.json'} <= set(frozen))
                for name, digest in frozen.items(): self.assertEqual(stream.sha(directory/name), digest)
                self.assertEqual(stream.read(directory/'status.json')['analysis_sha256'], stream.sha(directory/'analysis.json'))
            with self.assertRaises(ValueError): self.run_audit(pop, root/'audit1', binary, 1)

    def test_failure_preserves_completed_prefix_without_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pop = root/'population'
            binary = fixture(pop)
            def fail_second(config, local, batch, vessel, guide):
                if batch[0]['pose']['fixture_index'] == 3: raise ValueError('deliberate second batch failure')
                return density_check(config, local, batch, vessel, guide)
            with self.assertRaisesRegex(ValueError, 'deliberate'):
                self.run_audit(pop, root/'failed', binary, 3, checker=fail_second)
            self.assertEqual(len((root/'failed/geometry.jsonl').read_text().splitlines()), 3)
            self.assertEqual(stream.read(root/'failed/status.json')['phase'], 'failed')
            self.assertFalse((root/'failed/analysis.json').exists())
            self.assertFalse((root/'failed/freeze.json').exists())

    def test_binary_and_source_bundle_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pop = root/'population'
            binary = fixture(pop)
            binary.write_bytes(b'changed')
            with self.assertRaises(ValueError): self.run_audit(pop, root/'bad', binary, 3)
            self.assertFalse((root/'bad').exists())

    def test_truncated_journal_preserves_attempts_without_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pop = root/'population'
            binary = fixture(pop)
            p = pop/'attempts.jsonl'
            p.write_text(''.join(p.read_text().splitlines(keepends=True)[:-1]))
            s = stream.read(pop/'summary.json')
            s['attempts_sha256'] = stream.sha(p)
            stream.write(pop/'summary.json', s)
            with self.assertRaises(ValueError): self.run_audit(pop, root/'failed', binary, 3)
            self.assertEqual(stream.read(root/'failed/status.json')['processed_attempts'], 6)
            self.assertFalse((root/'failed/analysis.json').exists())


def moment_control(batch):
    """Reuse archived labels, not geometry; compare full-index eager reduction."""
    reducers = {name: stream.RegionMoments(len(batch)) for name in stream.CLASSES}
    classes = []
    for row in batch:
        c = stream.memberships(row, dict(core_disjoint=row['hard_valid'], exclusion_contact=row['depletion_contact']))
        stream.require(c['inside_R4']+c['outside_R4'] == c['total']
                       and c['exclusion_contact']+c['unbound'] == c['total'], 'Nonexhaustive saved-region partition')
        classes.append(c)
        for name in reducers: reducers[name].add(row, c[name])
    results = {}
    for name, reducer in reducers.items():
        current = reducer.result()
        for kind, key in [('Qz', 'log_importance_weight'), ('Q0', 'log_hard_weight')]:
            logs = [row[key] if c[name] else -math.inf for row, c in zip(batch, classes)]
            expected = stream.reference.moments(logs)
            for field in expected:
                a, b = current[kind][field], expected[field]
                stream.require((math.isclose(a, b, rel_tol=2e-12, abs_tol=2e-12)
                                if isinstance(a, (float, int)) and isinstance(b, (float, int)) else a == b),
                               'Eager/streaming saved moments differ: '+name+'/'+kind+'/'+field)
        physical = stream.reference.np.asarray([row['log_importance_weight'] if c[name] else -math.inf
                                              for row, c in zip(batch, classes)])
        pairs = stream.reference.np.asarray([[cloud['log_weight']-row['log_proposal_density'] for cloud in row['clouds']]
            if c[name] else [-math.inf, -math.inf] for row, c in zip(batch, classes)])
        expected = stream.reference.paired_noise(physical, pairs)
        stream.require((expected is None) == (current['paired_noise'] is None), 'Paired-noise observation differs')
        if expected is not None:
            for field, value in expected.items():
                if field == 'scope': continue
                actual = current['paired_noise'][field]
                stream.require((actual is None and value is None) or
                    (actual is not None and value is not None and math.isclose(actual, value, rel_tol=2e-12, abs_tol=2e-12)),
                    'Eager/streaming paired noise differs: '+name+'/'+field)
        results[name] = current
    return dict(draws=len(batch), invalid_zeros=sum(not row['hard_valid'] for row in batch), regions=results,
        native_labels='Not classified by this auditor; physical native partition remains separate.')


def saved_fixture_control(repository):
    """Exactly 32 saved-row density reconstructions, no draws or cloud samples.

    Eight existing rows per schema, each checked once by the reference and
    once via three streaming batches. No protein or core/contact query.
    """
    repository = Path(repository)
    roots = [repository/'results/native-class-vessel-cli-validation-20261004/attempt01/populations/class-depletion',
             repository/'results/wall-envelope-cli-validation-20261004/attempt01/class-depletion']
    identities = [
        ('573dad1a15e672dd867fe1f1420c6c50ac7a357d695a47bed709825a303621c8',
         'dbab54f04d50d1bdb0b8976f1a382af19a767c5001bfc232ba10b363b135bf89'),
        ('03e77a814d4af7b373aa10e485a6403ba8a94f5884609e8d6f2ef91f477c31d8',
         '5c9633345d3ddf4f15d7c4f674ce7e2ec831f8dd6a3f878e641931eb2ced1aae')]
    ledger = stream.Ledger()
    for name, digest in [
        ('results/native-class-vessel-cli-validation-20261004/audit-attempt01/analysis/summary.json',
         '1d95b8e42a208fa22298d51ff25887e56989d1697bbe00e09ff62b24649bd1e0'),
        ('results/wall-envelope-cli-validation-20261004/attempt01/class-depletion-audit.json',
         'd4b5245b5cb1493866bb542b0c40ce7b25cba3c3eccc77a45a5d1d94999546ad')]:
        old = stream.read(ledger.bind(repository/name, digest))
        stream.require(old['complete'] is True and old.get('passed', True) is True, 'Historical fixture audit failed')
    results = []
    for root, (manifest_sha, summary_sha) in zip(roots, identities):
        m = stream.read(ledger.bind(root/'manifest.json', manifest_sha))
        summary = stream.read(ledger.bind(root/'summary.json', summary_sha))
        stream.validate_manifest(m, summary)
        for name, key in [('input-config.json', 'config_sha256'), ('model.json', 'model_sha256'),
                          ('shape.json', 'shape_sha256'), ('source-bundle.json', 'source_bundle_sha256'),
                          ('latent-region.json', 'latent_region_sha256'), ('latent-guide.json', 'latent_guide_sha256')]:
            ledger.bind(root/'provenance'/name, m[key])
        samples = ledger.bind(root/'samples.jsonl', summary['samples_sha256'])
        journal = ledger.bind(root/'attempts.jsonl', summary['attempts_sha256'])
        batch = next(stream.attempt_batches(samples, journal, m['samples'], 8))
        config, _, guide, vessel, _ = stream.context(root, m, ledger, None, True)
        local = dict(m, samples=8)
        expected = stream.reference.check_rows(config, local, batch, vessel, guide)
        checks = stream.BatchChecks()
        for start in range(0, 8, 3): checks.check(config, local, batch[start:start+3], vessel, guide)
        checks.finish(local, dict(samples=8, **stream.row_counts(batch)))
        stream.require(set(checks.density) == set(expected), 'Saved reference/batch field inventory differs')
        for key in expected:
            if key in checks.maxima:
                stream.require(math.isclose(checks.density[key], expected[key], rel_tol=0., abs_tol=2e-11),
                               'Saved reference/batched numerical maximum differs: '+key)
            else:
                stream.require(checks.density[key] == expected[key], 'Saved reference/batch accounting differs: '+key)
        results.append(dict(schema=m['schema'], directory=str(root), manifest_sha256=stream.sha(root/'manifest.json'),
            selected_draws=list(range(8)), reference_density_pose_checks=8, streaming_density_pose_checks=8,
            density=checks.density, reference_density=expected,
            maximum_absolute_batch_reduction_difference=max(abs(checks.density[k]-expected[k]) for k in checks.maxima),
            eager_moment_control=moment_control(batch),
            primitive=checks.primitive, peak_rows=checks.peak_rows))
    ledger.recheck()
    return dict(complete=True, passed=True, controls=results, input_sha256=ledger.files,
        unique_saved_poses=16, density_pose_reconstructions=32,
        new_pose_draws=0, new_clouds=0, protein_queries=0, physical_core_contact_queries=0,
        scope='Small archived sphere-fixture equivalence only; interval geometry reconstructed twice per selected row. '
              'No full reference rerun, physical sample, or protein production admission.')


if __name__ == '__main__': unittest.main()
