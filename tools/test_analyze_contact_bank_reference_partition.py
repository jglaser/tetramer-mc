"""Guard the declared old-R5 predicate, fixed-N partition, and source scope."""
import copy
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analyze_contact_bank_reference_partition import (HashLedger, NATIVE, PARTS, PLAN_SHA256, SCOPE,
    r5_contains, partition_masks, summarize_partition, validate_regions, validate_sources,
    read_population, sha, write)


def regions():
    shared = dict(physical_fixed_neighbors=[dict(position=[0., 0., 0.]), dict(position=[1., 0., 0.])],
        capture_center=[0., 0., 0.], capture_radius=170., shape_sha256='shape', activity=.035,
        depletant_radius=1.5, physical_metric={'identity': 'original'}, minimum_mahalanobis_radius=0.)
    current = dict(shared, mahalanobis_radius=4., minimum_original_q=0., minimum_original_q_inclusive=True)
    old = dict(shared, mahalanobis_radius=5., minimum_original_q=1., minimum_original_q_inclusive=False)
    return current, old


def populations(all_zero=False):
    result = []
    for index in range(4):
        z = np.array([math.log(2.), math.log(3.), math.log(7.), -np.inf, -np.inf])
        h = np.array([0., 0., 0., -np.inf, -np.inf])
        native = np.array([True, True, False, False, False])
        if all_zero:
            z[:] = h[:] = -np.inf
            native[:] = False
        valid = np.isfinite(z)
        masks = partition_masks(valid, native, np.array([True, False, True, True, False]))
        result.append(dict(id=f'r{index:02d}', seed=100+index, z=z, h=h,
            pairs=np.column_stack((z, z)), masks=masks))
    return result


class ReferencePartitionTests(unittest.TestCase):
    def test_r5_closed_radius_and_strict_original_q(self):
        radius = [0., 5., np.nextafter(5., np.inf), 5., 5., 1.]
        q = [2., 2., 2., 1., np.nextafter(1., np.inf), 3.]
        capture = np.array([True, True, True, True, True, False])
        np.testing.assert_array_equal(r5_contains(radius, q, capture), [True, True, False, False, True, False])
        with self.assertRaisesRegex(ValueError, 'Invalid R5'):
            r5_contains([float('nan')], [2.], np.array([True]))

    def test_complement_is_within_full_native_and_keeps_every_attempt(self):
        pops = populations(); estimates = summarize_partition(pops)
        for estimate in estimates.values():
            self.assertEqual(estimate['row_uncertainty']['draws'], 20)
            self.assertEqual(estimate['population_uncertainty']['draws'], 4)
            self.assertEqual([p['draws'] for p in estimate['populations']], [5]*4)
        self.assertAlmostEqual(estimates[NATIVE]['row_uncertainty']['log_Qz'], 0.)
        self.assertAlmostEqual(estimates[PARTS[0]]['row_uncertainty']['log_Qz'], math.log(2./5.))
        self.assertAlmostEqual(estimates[PARTS[1]]['row_uncertainty']['log_Qz'], math.log(3./5.))
        self.assertAlmostEqual(estimates[PARTS[0]]['observed_fraction_of_native']['Qz'], 2./5.)
        # The nonnative weight seven and the exterior/invalid zeros enter N only.
        self.assertEqual(estimates[NATIVE]['row_uncertainty']['nonzero'], 8)
        changed = copy.deepcopy(estimates[NATIVE]); changed['row_uncertainty']['draws'] = 8
        with self.assertRaisesRegex(ValueError, 'draws changed'):
            summarize_partition(pops, changed)

    def test_all_zero_populations_remain_unresolved_with_full_N(self):
        estimates = summarize_partition(populations(all_zero=True))
        for estimate in estimates.values():
            self.assertEqual(estimate['row_uncertainty']['draws'], 20)
            self.assertEqual(estimate['row_uncertainty']['nonzero'], 0)
            self.assertIsNone(estimate['row_uncertainty']['log_Qz'])
            self.assertIsNone(estimate['row_uncertainty']['log_Q0'])
            self.assertEqual(estimate['population_uncertainty']['draws'], 4)
            self.assertEqual([p['draws'] for p in estimate['populations']], [5]*4)
        with self.assertRaisesRegex(ValueError, 'Invalid zero'):
            partition_masks(np.array([False]), np.array([True]), np.array([True]))

    def test_unobserved_complement_does_not_become_proven_zero(self):
        pops = populations()
        for p in pops:
            p['masks'][PARTS[0]] = p['masks'][NATIVE].copy()
            p['masks'][PARTS[1]][:] = False
        estimate = summarize_partition(pops)[PARTS[1]]
        self.assertIsNone(estimate['row_uncertainty']['log_Qz'])
        self.assertIn('neither zero mass nor an upper bound', estimate['row_uncertainty']['unresolved'])
        self.assertEqual(estimate['observed_fraction_of_native']['Qz'], 0.)

    def test_full_R4_and_old_R5_target_identities_are_required(self):
        current, old = regions(); validate_regions(current, old)
        for key, value in [('minimum_original_q_inclusive', True), ('minimum_original_q', 0.), ('mahalanobis_radius', 5.1)]:
            changed = copy.deepcopy(old); changed[key] = value
            with self.assertRaisesRegex(ValueError, 'strict original q'):
                validate_regions(current, changed)
        for key, value in [('physical_metric', {'identity': 'other'}), ('capture_radius', 169.), ('shape_sha256', 'other')]:
            changed = copy.deepcopy(old); changed[key] = value
            with self.assertRaisesRegex(ValueError, 'target differs'):
                validate_regions(current, changed)
        changed = copy.deepcopy(current); changed['maximum_original_q'] = 1.
        with self.assertRaisesRegex(ValueError, 'full R4'):
            validate_regions(changed, old)

    def test_hash_ledger_records_checked_files_and_detects_post_read_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root/'raw'; other = root/'not-consumed'
            raw.write_text('all attempted rows\n'); other.write_text('unrelated\n')
            write(root/'freeze.json', {'files': {'raw': sha(raw)}})
            ledger = HashLedger(); ledger.frozen(root)
            self.assertEqual(set(ledger.files), {str(raw), str(root/'freeze.json')})
            other.write_text('unrelated change\n'); ledger.recheck()
            raw.write_text('only valid rows\n')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                ledger.recheck()
            with self.assertRaisesRegex(ValueError, 'Hash changed'):
                HashLedger().frozen(root)

    def test_freeze_paths_cannot_escape_and_plan_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); nested = root/'nested'; nested.mkdir()
            outside = root/'outside'; outside.write_text('data')
            write(nested/'freeze.json', {'../outside': sha(outside)})
            with self.assertRaisesRegex(ValueError, 'escapes'):
                HashLedger().frozen(nested)
            plan = root/'plan.json'; write(plan, {'schema': 'contact-bank-reference-partition-plan-v1'})
            with self.assertRaisesRegex(ValueError, 'Hash changed'):
                validate_sources(plan, HashLedger())
            self.assertEqual(len(PLAN_SHA256), 64)

    def test_no_production_or_training_validation_claim(self):
        self.assertIn('not fresh proposal validation', SCOPE)
        self.assertIn('convergence gates are unchanged', SCOPE)
        self.assertIn('cannot certify unseen mass or authorize production', SCOPE)

    def test_raw_records_labels_and_all_zero_rows_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); campaign = root/'campaign'; comparison = root/'comparison'; comparison.mkdir()
            arm = dict(id='bank', alpha=.5, component_count=1)
            current, old = regions(); provenance = campaign/'bank/provenance'; provenance.mkdir(parents=True)
            write(provenance/'region.json', current)
            folder = campaign/'bank/runs/r00'; folder.mkdir(parents=True)
            rows = []
            for i in range(3):
                zero = i > 0; latent = [5. if i == 1 else 0., 0., 0., 0., 0., 0.]
                rows.append(dict(draw=i, pose=dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.]),
                    hard_valid=i < 2, capture_valid=True, region_valid=True, shell_valid=i != 1,
                    latent=latent, latent_radius=latent[0], q=2., log_physical_jacobian=0., log_proposal_density=0.,
                    proposal_branch='gaussian', proposal_component=0, log_hard_weight=None if zero else 0.,
                    log_importance_weight=None if zero else math.log(2.), clouds=[] if zero else [{'log_weight': math.log(2.)}]*2))
            raw = folder/'samples.jsonl'; raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest = dict(samples=3, seed=1, region_sha256=sha(provenance/'region.json'), shape_sha256='shape',
                cloud_replicates=2, physical_fixed_neighbors=current['physical_fixed_neighbors'], minimum_original_q=0.,
                minimum_original_q_inclusive=True, maximum_original_q=None, minimum_latent_radius=0., latent_radius=4.,
                proposal_density_measure='Lebesgue measure in the original six-dimensional whitened region chart')
            write(folder/'manifest.json', manifest)
            write(folder/'summary.json', dict(samples=3, complete=True, manifest=manifest))
            output = {key: sha(folder/name) for name, key in [('samples.jsonl', 'samples_sha256'), ('manifest.json', 'manifest_sha256'), ('summary.json', 'summary_sha256')]}
            records = comparison/'records.npz'
            from analyze_contact_bank_pilot import read_weights
            arrays = read_weights(rows, 3, current, arm); arrays.pop('latents')
            np.savez_compressed(records, **arrays, native=np.array([True, False, False]), contact=np.array([True, False, False]))
            labels = comparison/'labels.jsonl.gz'
            with gzip.open(labels, 'wt') as stream:
                for i in range(3):
                    stream.write(json.dumps(dict(draw=i, applicable=i == 0,
                        classification={'native_any': True} if i == 0 else None,
                        contact={'exclusion_contact': True} if i == 0 else None))+'\n')
            job = dict(arm='bank', id='r00', seed=1, samples=3, directory=str(folder))
            record = dict(job, raw_output=output, records='records.npz', records_sha256=sha(records),
                labels='labels.jsonl.gz', labels_sha256=sha(labels))
            terminal = dict(output=output)
            def invoke():
                return read_population(comparison, campaign, record, job, terminal, arm, current, old, HashLedger())
            coords = np.asarray([r['latent'] for r in rows])
            with patch('analyze_contact_bank_reference_partition.chart_coordinates', return_value=coords), \
                    patch('analyze_contact_bank_reference_partition.jacobian', return_value=np.zeros(3)), \
                    patch('analyze_contact_bank_reference_partition.native_q', return_value=2.):
                population, diagnostic = invoke()
                self.assertEqual(len(population['z']), 3)
                self.assertEqual(diagnostic['exterior_attempts'], 1)
                self.assertEqual(diagnostic['invalid_or_exterior_attempts'], 2)
                np.testing.assert_array_equal(population['masks'][PARTS[0]], [True, False, False])
                with gzip.open(labels, 'at') as stream:
                    stream.write('{}\n')
                with self.assertRaisesRegex(ValueError, 'Hash changed'):
                    invoke()
                record['labels_sha256'] = sha(labels)
                with self.assertRaisesRegex(ValueError, 'Too many classifier labels'):
                    invoke()


if __name__ == '__main__':
    unittest.main()
