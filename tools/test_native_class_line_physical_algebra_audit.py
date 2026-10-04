"""Streaming v7 audit controls using six fixed synthetic attempted rows.

The provenance schema follows src/latent_region.rs. All hashes bind actual
temporary bytes; no real result directories, geometry queries, poses or Poisson
clouds are sampled. The deliberately supplied base interval sets test algebra,
not correspondence with the toy atoms. No native observer is constructed.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

import native_class_line_algebra_reference as algebra
import native_class_line_physical_algebra_audit as audit
import native_class_line_physical_reference as physical
import native_class_line_reference as line
from native_contact_regions import CRITERIA
from test_native_class_line_algebra_reference import rebuild_saved_channels, selected_draw


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonlines(path, rows):
    Path(path).write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))


def stream_contract():
    return dict(hash_domain='tetramer-uniform-latent-region-v1',
        key_fields=['master seed u64 little endian', 'draw u64 little endian',
                    'cloud index u64 little endian', 'role UTF-8 bytes'],
        proposal_role='latent', proposal_cloud_index=0, physical_cloud_role='cloud',
        physical_cloud_indices='0..cloud_replicates, separately derived; no proposal RNG consumption')


def recorded_trace(law, u, *, hard=True):
    """Finite, canonical synthetic unions; does not evaluate any atom pair."""
    if law.alpha == 1 or law.beta == 0:
        return {'conditioning_disabled': True}
    logs = law.gaussian_logs(u)
    uniform = math.log(law.alpha)-law.logvolume if np.linalg.norm(u) <= law.radius else -math.inf
    baseline = float(np.logaddexp(uniform, math.log1p(-law.alpha)+logsumexp(logs)))
    axes = []
    for axis in law.axes:
        frame = law.line_frame(u, axis)
        segment = [line.interval(*frame['segment'])] if 'segment' in frame else []
        frame.update(hard_free_intervals=segment if hard else [],
            native_intervals=line.intersection(segment, [line.interval(-1., 1.)]),
            exclusion_contact_intervals=line.intersection(segment, [line.interval(-2., 2.)]))
        axes.append(frame)
    trace = dict(trace_format='class-line-compact-v1', class_scope=algebra.CLASS_SCOPE,
        raw_coordinates=law.raw(u).tolist(), baseline_log_density=baseline, axes=axes)
    return rebuild_saved_channels(law, u, trace)


def fixture(base, *, beta=.5):
    """Build the actual archived v7 record layout with independently bound bytes."""
    root = Path(base)/'population'; root.mkdir(); provenance = root/'provenance'; provenance.mkdir()
    identity = np.eye(3).tolist()
    pose = lambda x: dict(position=list(x), orientation=[1., 0., 0., 0.])
    origin = pose([0., 0., 0.]); fixed = [origin]
    shape = dict(name='synthetic-four-coincident-members', volume=4*math.pi*.1**3/3,
                 atoms=[dict(center=[0., 0., 0.], radius=.1) for _ in range(4)])
    save(provenance/'shape.json', shape); shape_sha = sha(provenance/'shape.json')
    # These files are fixture authorities, not claimed Rust executions.
    save(provenance/'native-definition.json', dict(schema='synthetic-native-definition',
         input_sha256={'tetramer-shape.json': shape_sha}))
    definition_sha = sha(provenance/'native-definition.json')
    compiled = dict(schema='native-entry-compiled-v1', source_definition_sha256=definition_sha,
        source_input_sha256={'tetramer-shape.json': shape_sha}, criteria=CRITERIA, fixed_poses=fixed,
        members=[dict(position=[0., 0., 0.], rotation=identity) for _ in range(4)],
        monomer_atoms=[dict(center=[0., 0., 0.], radius=.1, residue=0)], residue_count=1,
        references=[dict(label='toy', family='toy', position=[.5, 0., 0.], rotation=identity,
                         native_residue_pairs=[0])],
        motifs=[dict(id=0, position=[.5, 0., 0.], rotation=identity,
                     member_contacts=[dict(member_i=i, member_j=i, directed_class='toy') for i in range(4)])])
    save(provenance/'compiled-native.json', compiled); native_sha = sha(provenance/'compiled-native.json')
    metadata = dict(native_poses=[pose([.5, 0., 0.])], rigid_members=[origin]*4,
                    member_error_scale=2., angle_error_scale_deg=15.)
    config = dict(shape=str(provenance/'shape.json'), fixed_poses=fixed, initial_pose=pose([.5, 0., 0.]),
        capture_center=[0., 0., 0.], capture_radius=10., depletant_radius=.5, reservoir_density=.3,
        poisson_lambda_ratio=64., translation_steps=[.1], rotation_steps_deg=[1.], rotation_probability=.5,
        local_attempts_per_cycle=1, uniform_probability=.5, seed=123, metadata=metadata,
        endpoint_gate=dict(max_cells=127, max_depth=6, min_width=0.))
    save(provenance/'input-config.json', config); save(root/'config.json', config)
    chart = dict(shape_sha256=shape_sha, coordinate_convention='anchor-body-relative', angular_length=8.,
        weights=[1.], anchors=[dict(position=[0., 0., 0.], rotation=identity)],
        means=[[0.]*6], covariances=[np.eye(6).tolist()])
    region = dict(shape_sha256=shape_sha, fixed_neighbor=origin, physical_fixed_neighbors=fixed,
        capture_center=[0., 0., 0.], capture_radius=10., activity=.3, depletant_radius=.5,
        physical_metric=metadata, minimum_original_q=0., maximum_original_q=1.,
        minimum_original_q_inclusive=True, maximum_original_q_inclusive=True,
        mahalanobis_radius=4., gaussian_chart=chart)
    save(provenance/'region.json', region)
    guide = dict(schema=physical.GUIDE_SCHEMA, region_sha256=sha(provenance/'region.json'),
        defensive_uniform_shell_probability=.5,
        gaussian_components=[dict(weight=.6, mean=[.5, 0., 0., 0., 0., 0.], covariance=np.eye(6).tolist()),
                             dict(weight=.4, mean=[-.5, 0., 0., 0., 0., 0.], covariance=np.eye(6).tolist())],
        raw_translation_axes=[0, 1, 2], conditional_probability=beta, minimum_conditional_mass=1e-12,
        class_channels=[dict(**{'class': 'hard_free'}, probability=.3),
                        dict(**{'class': 'native'}, probability=.3),
                        dict(**{'class': 'contact_without_native'}, probability=.4)],
        compiled_native=dict(path=str(provenance/'compiled-native.json'), sha256=native_sha),
        shape_sha256=shape_sha, fixed_poses=fixed, capture_center=[0., 0., 0.], capture_radius=10., depletant_radius=.5)
    save(provenance/'importance-guide.json', guide)
    save(provenance/'source-bundle.json', dict(synthetic_fixture=True,
        source_sha256={str(Path(__file__).resolve()): sha(__file__)}))
    (provenance/'synthetic-producer.txt').write_text('Fixed synthetic records; no Rust or Monte Carlo execution.\n')
    law = algebra.AlgebraLaw(region, guide, config)
    report = dict(compiled_sha256=native_sha, expected_shape_sha256=shape_sha,
        native_atoms=4, physical_atoms=4, matched_atoms=4, center_tolerance_a=1e-10, radius_tolerance_a=1e-12,
        matched_max_center_error_a=0., matched_max_radius_error_a=0., physical_index_by_native_atom=list(range(4)),
        unmatched_native_atoms=[], unmatched_physical_atoms=[], compatible=True, pair_overlap_slack_bound_a=0.,
        observer_hard_overlap_tolerance_a=1e-8, hard_valid_implication_within_tolerance=True)
    manifest = dict(schema=physical.SCHEMA, guide_schema=physical.GUIDE_SCHEMA, proposal_kind=physical.PROPOSAL_KIND,
        samples=6, seed=123, cloud_replicates=2, activity=.3, lambda_ratio=64., **{'lambda': 19.2},
        compiled_native=dict(compiled_sha256=native_sha, source_definition_sha256=definition_sha,
                             source_input_sha256=compiled['source_input_sha256'], shape_compatibility=report),
        physical_fixed_neighbors=fixed, chart_anchor=origin, importance_uniform_probability=.5,
        importance_component_count=2, proposal_density_measure=physical.MEASURE, latent_radius=4.,
        minimum_latent_radius=0., log_latent_ball_volume=law.logvolume, log_latent_shell_volume=law.logvolume,
        minimum_original_q=0., maximum_original_q=1., minimum_original_q_inclusive=True,
        maximum_original_q_inclusive=True, resume_supported=False,
        executable_sha256=sha(provenance/'synthetic-producer.txt'),
        density_trace_contract=dict(format='class-line-compact-v1',
            retained='Every axis/class interval set, geometric work counts, raw coordinates and full per-component axis-averaged multipliers',
            omitted='Per-component per-channel masses and fallback records; reconstruct from frozen guide plus retained intervals',
            diagnostic_full_trace_unchanged=True), random_stream_contract=stream_contract(),
        class_scope='complete native and competing contact; optional original latent orthants; no old-R5 restriction',
        attempt_journal='attempts.jsonl; begin record before each draw; no retries',
        failure_trace_contract='Attempt ID and completed rows retained; v7 additionally saves available latent/pose/density/validity fields, completed cloud records and current cloud index; internal progress of a failed cloud is unavailable')
    for name, key in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'),
                      ('shape.json', 'shape_sha256'), ('importance-guide.json', 'importance_guide_sha256'),
                      ('source-bundle.json', 'source_bundle_sha256')]:
        manifest[key] = sha(provenance/name)
    save(root/'manifest.json', manifest)
    rows = []
    for index, x in enumerate([.5, 1.5, 0., 5., 11., 3.]):
        u = np.array([x, 0., 0., 0., 0., 0.]); hard = index not in (2, 4)
        trace = recorded_trace(law, u, hard=hard); evaluated = law.evaluate_saved(u, trace)
        raw, position, rotation, jac = law.decode(u)
        p = dict(position=position.tolist(), orientation=Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].tolist())
        q = abs(x-.5)/2.; valid = index in (0, 1)
        clouds = []
        if valid:
            for k in (1, 3):
                clouds.append(dict(lower_volume=.1, upper_volume=.5, uncertain_volume=.4,
                    raw_points=4, overlap_points=k, retained_cells=2, created_cells=5, certified_cells=1,
                    log_weight=.3*.1+k*math.log1p(.3/19.2)))
        h = jac-evaluated['log_density'] if valid else None
        draw = (selected_draw(law, u, trace, evaluated, channel=1) if index == 0 and beta > 0
                else dict(conditional=False, original_latent=u.tolist()))
        row = dict(draw=index, latent=u.tolist(), latent_radius=float(np.linalg.norm(u)), pose=p,
            backmapped_latent=u.tolist(), backmapped_radius=float(np.linalg.norm(u)), q=q,
            log_physical_jacobian=jac, physical_jacobian=math.exp(jac), shell_valid=bool(np.linalg.norm(u) <= 4),
            capture_valid=x <= 10, hard_valid=hard, region_valid=q <= 1.,
            log_proposal_density=evaluated['log_density'], log_hard_weight=h,
            log_importance_weight=h+float(logsumexp([c['log_weight'] for c in clouds]))-math.log(2) if valid else None,
            clouds=clouds, proposal_branch='uniform-shell' if index == 1 else 'native-class-line',
            proposal_component=None if index == 1 else 0,
            native_class_line_draw=draw, native_class_line_density=trace)
        rows.append(row)
    jsonlines(root/'samples.jsonl', rows)
    jsonlines(root/'attempts.jsonl', [dict(draw=i, state='begin') for i in range(6)])
    z = [r['log_importance_weight'] if r['log_importance_weight'] is not None else -math.inf for r in rows]
    h = [r['log_hard_weight'] if r['log_hard_weight'] is not None else -math.inf for r in rows]
    summary = dict(complete=True, manifest=manifest, samples=6, samples_sha256=sha(root/'samples.jsonl'),
        attempts_sha256=sha(root/'attempts.jsonl'), capture_rejected=1, hard_rejected=1, region_rejected=2,
        shell_rejected=2, raw_points=16, sampler_cpu_seconds=.1, wall_seconds=.2, maximum_backmap_error=0.,
        estimates=dict(region=physical.physical.statistics.moments(z), hard_region=physical.physical.statistics.moments(h)))
    save(root/'summary.json', summary)
    return root, rows, manifest, summary, law


def rebind_rows(root, rows, summary):
    jsonlines(root/'samples.jsonl', rows); summary['samples_sha256'] = sha(root/'samples.jsonl')
    save(root/'summary.json', summary)


class StreamingAlgebraAuditTests(unittest.TestCase):
    def run_fixture(self, root, *, name='audit.json'):
        output = root.parent/name
        result = audit.audit(root, output=output)
        self.assertEqual(json.loads(output.read_text()), result)
        return result, output

    def test_complete_rows_include_zeros_and_receipt_disclaims_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root, rows, manifest, summary, law = fixture(Path(temp))
            forbidden = AssertionError('All-row algebra must not query atom geometry')
            with ExitStack() as stack:
                for target, name in [(line.NativeContactRegions, '__init__'), (line, 'Reconstructor'),
                    (line, 'leaf_contact_intervals'), (line.hard, 'hard_free_intervals'),
                    (line, 'cKDTree'), (line.hard, 'cKDTree'), (line.NativeLineReference, 'all_anchors')]:
                    stack.enter_context(mock.patch.object(target, name, side_effect=forbidden))
                result, output = self.run_fixture(root)
            self.assertTrue(result['complete'] and result['passed'])
            self.assertEqual(result['schema'], 'native-class-physical-all-row-algebra-audit-v1')
            self.assertEqual(result['all_rows_algebra'], 6)
            self.assertEqual(result['independently_reconstructed_geometry_rows'], 0)
            self.assertFalse(result['geometry_certified'])
            self.assertEqual(result['estimate']['draws'], 6)
            self.assertEqual(result['estimate']['nonzero'], 2)
            self.assertAlmostEqual(result['estimate']['logQ'], summary['estimates']['region']['logQ'])
            self.assertEqual(result['counts']['attempted'], 6)
            self.assertEqual(result['counts']['capture_rejected'], 1)
            self.assertEqual(result['counts']['hard_rejected'], 1)
            self.assertEqual(result['counts']['shell_rejected'], 2)
            self.assertEqual(result['counts']['region_rejected'], 2)
            self.assertEqual(result['independently_checked_distinct_role_keys'], 18)
            self.assertEqual(result['new_pose_draws'], 0)
            self.assertEqual(result['new_Poisson_clouds'], 0)
            events = [json.loads(s) for s in output.with_suffix('.journal.jsonl').read_text().splitlines()]
            self.assertEqual([e['draw'] for e in events if e['state'] == 'begin'], list(range(6)))
            self.assertEqual([e['draw'] for e in events if e['state'] == 'complete'], list(range(6)))

    def test_disabled_conditioning_retains_same_wrapper_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root, *_ = fixture(Path(temp), beta=0.)
            result, _ = self.run_fixture(root)
            self.assertEqual(result['counts']['conditioned_draws'], 0)
            self.assertEqual(result['all_rows_algebra'], 6)

    def test_beta_one_cannot_record_an_unconditioned_gaussian_draw(self):
        with tempfile.TemporaryDirectory() as temp:
            root, rows, manifest, summary, law = fixture(Path(temp), beta=1.)
            config = json.loads((root/'provenance/input-config.json').read_text())
            # The genuinely conditional first row is supported under beta=1.
            accounting, _, _ = audit._check_row(rows[0], 0, manifest, law.region, config, law)
            self.assertEqual(accounting['counters']['conditioned_draws'], 1)
            # Merely leaving its original coordinate unchanged cannot explain
            # a Gaussian draw on a branch with zero unconditional probability.
            rows[0]['native_class_line_draw'] = dict(conditional=False, original_latent=rows[0]['latent'])
            rebind_rows(root, rows, summary)
            output = Path(temp)/'audit.json'
            with self.assertRaisesRegex(ValueError, 'conditional'):
                audit.audit(root, output=output)
            failed = json.loads(output.with_suffix('.failure.json').read_text())
            self.assertEqual((failed['draw'], failed['completed_rows']), (0, 0))

    def test_truncated_extra_duplicate_reordered_rows_and_halfline_fail(self):
        for mode in ('missing_last', 'extra', 'duplicate', 'reordered', 'halfline', 'no_newline'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root, rows, manifest, summary, law = fixture(Path(temp))
                if mode == 'missing_last': rows.pop()
                if mode == 'extra': rows.append(copy.deepcopy(rows[-1]))
                if mode == 'duplicate': rows[3] = copy.deepcopy(rows[2])
                if mode == 'reordered': rows[2], rows[3] = rows[3], rows[2]
                rebind_rows(root, rows, summary)
                if mode in ('halfline', 'no_newline'):
                    path = root/'samples.jsonl'; data = path.read_bytes()
                    path.write_bytes(data[:-20] if mode == 'halfline' else data[:-1])
                    summary['samples_sha256'] = sha(path); save(root/'summary.json', summary)
                output = Path(temp)/'audit.json'
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    audit.audit(root, output=output)
                self.assertFalse(output.exists())
                self.assertTrue(output.with_suffix('.failure.json').exists())

    def test_attempt_journal_inventory_and_summary_denominator_fail(self):
        for mode in ('missing', 'extra', 'wrong_state', 'bool_id', 'denominator', 'summary_count'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root, rows, manifest, summary, law = fixture(Path(temp))
                attempts = [dict(draw=i, state='begin') for i in range(6)]
                if mode == 'missing': attempts.pop()
                if mode == 'extra': attempts.append(dict(draw=6, state='begin'))
                if mode == 'wrong_state': attempts[2]['state'] = 'complete'
                if mode == 'bool_id': attempts[0]['draw'] = False
                if mode == 'denominator': summary['samples'] = 5
                if mode == 'summary_count': summary['hard_rejected'] = 0
                jsonlines(root/'attempts.jsonl', attempts)
                summary['attempts_sha256'] = sha(root/'attempts.jsonl'); save(root/'summary.json', summary)
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    audit.audit(root, output=Path(temp)/'audit.json')

    def test_weight_density_jacobian_geometry_trace_and_draw_sabotage(self):
        changes = {
            'J': lambda row: row.update(log_physical_jacobian=row['log_physical_jacobian']+.1),
            'q': lambda row: row.update(log_proposal_density=row['log_proposal_density']+.1),
            'hard_weight': lambda row: row.update(log_hard_weight=row['log_hard_weight']+.1),
            'physical_weight': lambda row: row.update(log_importance_weight=row['log_importance_weight']+.1),
            'metric': lambda row: row.update(q=.1),
            'backmap': lambda row: row['backmapped_latent'].__setitem__(0, 2.),
            'pose': lambda row: row['pose']['position'].__setitem__(0, 2.),
            'density_multiplier': lambda row: row['native_class_line_density']['component_mixture_multipliers'].__setitem__(0, 99.),
            'class_interval': lambda row: row['native_class_line_density']['axes'][0]['channels'][1].update(intervals=[]),
            'selected_mass': lambda row: row['native_class_line_draw'].update(effective_mass=.9),
            'draw_cdf': lambda row: row['native_class_line_draw'].update(uniform_within_interval=.99),
            'component': lambda row: row['native_class_line_draw'].update(component=1),
            'cloud_count': lambda row: row['clouds'][0].update(overlap_points=9),
            'cloud_count_integer': lambda row: row['clouds'][0].update(raw_points=True),
        }
        for mode, change in changes.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root, rows, manifest, summary, law = fixture(Path(temp)); change(rows[0]); rebind_rows(root, rows, summary)
                output = Path(temp)/'audit.json'
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    audit.audit(root, output=output)
                self.assertFalse(output.exists())
                failure = json.loads(output.with_suffix('.failure.json').read_text())
                self.assertFalse(failure.get('complete', False))
                self.assertIn(str((root/'samples.jsonl').resolve()), failure['input_sha256'])
                self.assertEqual(failure['input_sha256'][str((root/'samples.jsonl').resolve())], sha(root/'samples.jsonl'))

    def test_failure_and_changed_provenance_cannot_be_admitted(self):
        for mode in ('population_failure', 'manifest_summary', 'shape', 'guide', 'compiled', 'source', 'samples'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root, rows, manifest, summary, law = fixture(Path(temp))
                if mode == 'population_failure': save(root/'failure.json', dict(complete=False, draw=5))
                elif mode == 'manifest_summary': manifest['seed'] += 1; save(root/'manifest.json', manifest)
                else:
                    path = {'shape': root/'provenance/shape.json', 'guide': root/'provenance/importance-guide.json',
                            'compiled': root/'provenance/compiled-native.json', 'source': root/'provenance/source-bundle.json',
                            'samples': root/'samples.jsonl'}[mode]
                    path.write_bytes(path.read_bytes()+b' ')
                output = Path(temp)/'audit.json'
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    audit.audit(root, output=output)
                self.assertFalse(output.exists())

    def test_json_duplicates_nonfinite_constants_and_nonobjects_fail(self):
        for replacement in ('{"draw":0,"draw":0}\n', '{"draw":NaN}\n', '[]\n', '\n'):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as temp:
                root, rows, manifest, summary, law = fixture(Path(temp))
                path = root/'samples.jsonl'; lines = path.read_text().splitlines(keepends=True)
                lines[0] = replacement; path.write_text(''.join(lines))
                summary['samples_sha256'] = sha(path); save(root/'summary.json', summary)
                output = Path(temp)/'audit.json'
                with self.assertRaises(ValueError): audit.audit(root, output=output)
                failed = json.loads(output.with_suffix('.failure.json').read_text())
                self.assertEqual(failed['draw'], 0)
                self.assertEqual(failed['begun_rows'], 1)
                self.assertEqual(failed['completed_rows'], 0)
                self.assertFalse(output.exists())

    def test_late_failure_retains_prefix_draw_ids_hashes_and_byte_offsets(self):
        with tempfile.TemporaryDirectory() as temp:
            root, rows, manifest, summary, law = fixture(Path(temp))
            rows[4]['backmapped_latent'][0] = 12.; rebind_rows(root, rows, summary)
            output = Path(temp)/'audit.json'
            with self.assertRaises(ValueError): audit.audit(root, output=output)
            failure = json.loads(output.with_suffix('.failure.json').read_text())
            self.assertEqual((failure['begun_rows'], failure['completed_rows'], failure['draw']), (5, 4, 4))
            journal = output.with_suffix('.journal.jsonl')
            self.assertEqual(failure['journal']['sha256'], sha(journal))
            events = [json.loads(s) for s in journal.read_text().splitlines()]
            self.assertEqual([e['draw'] for e in events if e['state'] == 'begin'], list(range(5)))
            self.assertEqual([e['draw'] for e in events if e['state'] == 'complete'], list(range(4)))
            raw = (root/'samples.jsonl').read_bytes().splitlines(keepends=True)
            for e in events:
                if e['state'] != 'complete': continue
                record = e['sample_record']; i = e['draw']
                self.assertEqual(record['offset'], sum(map(len, raw[:i])))
                self.assertEqual(record['sha256'], hashlib.sha256(raw[i]).hexdigest())
            self.assertFalse(output.exists())

    def test_mutating_input_or_new_producer_failure_during_audit_invalidates_receipt(self):
        for mode in ('input', 'producer_failure'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root, *_ = fixture(Path(temp)); output = Path(temp)/'audit.json'
                original = audit._check_row
                def changed(row, index, *args):
                    result = original(row, index, *args)
                    if index == 0:
                        if mode == 'input':
                            p = root/'provenance/input-config.json'; p.write_bytes(p.read_bytes()+b' ')
                        else: save(root/'failure.json', dict(complete=False, draw=5))
                    return result
                with mock.patch.object(audit, '_check_row', side_effect=changed), self.assertRaises(ValueError):
                    audit.audit(root, output=output)
                failed = json.loads(output.with_suffix('.failure.json').read_text())
                self.assertEqual(failed['completed_rows'], 6)
                self.assertEqual(failed['phase'], 'final_validation')
                self.assertFalse(output.exists())

    def test_sigterm_writes_failure_and_preserves_begun_row_in_bounded_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root, *_ = fixture(Path(temp)); output = Path(temp)/'audit.json'
            program = '''
import os, signal, sys
from pathlib import Path
import native_class_line_physical_algebra_audit as audit
def interrupted(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGTERM)
    raise AssertionError('SIGTERM did not interrupt')
audit._check_row = interrupted
audit.audit(Path(sys.argv[1]), output=Path(sys.argv[2]))
'''
            environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent),
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            child = subprocess.run([sys.executable, '-c', program, str(root), str(output)],
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            self.assertEqual(child.returncode, 128+signal.SIGTERM, child.stderr)
            self.assertFalse(output.exists())
            failed = json.loads(output.with_suffix('.failure.json').read_text())
            self.assertEqual((failed['draw'], failed['begun_rows'], failed['completed_rows']), (0, 1, 0))
            self.assertEqual(failed['error_type'], 'SystemExit')
            self.assertEqual(failed['journal']['sha256'], sha(output.with_suffix('.journal.jsonl')))

    def test_existing_signal_handler_is_restored_after_success_or_failure(self):
        previous = signal.getsignal(signal.SIGTERM)
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as temp:
                root, rows, manifest, summary, law = fixture(Path(temp))
                if fail: rows[0]['q'] = .1; rebind_rows(root, rows, summary)
                try:
                    if fail:
                        with self.assertRaises(ValueError): audit.audit(root, output=Path(temp)/'audit.json')
                    else: self.run_fixture(root)
                finally:
                    self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_exclusive_output_and_journal_prevent_repeat_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            root, *_ = fixture(Path(temp)); result, output = self.run_fixture(root)
            original = output.read_bytes()
            with self.assertRaises((ValueError, FileExistsError)):
                audit.audit(root, output=output)
            self.assertEqual(output.read_bytes(), original)
        with tempfile.TemporaryDirectory() as temp:
            root, *_ = fixture(Path(temp)); output = Path(temp)/'audit.json'
            journal = output.with_suffix('.journal.jsonl'); journal.write_text('previous attempt\n')
            with self.assertRaises((ValueError, FileExistsError)):
                audit.audit(root, output=output)
            self.assertEqual(journal.read_text(), 'previous attempt\n')
            self.assertFalse(output.exists())


class HardFreeDispatchTests(unittest.TestCase):
    def make_population(self, root):
        from test_hard_free_line_physical_reference import fixture as hard_fixture
        root.mkdir()
        region, guide, config, shape, rows, manifest, summary = hard_fixture(root)
        manifest.update(resume_supported=False,
            attempt_journal='attempts.jsonl; begin record before each draw; no retries')
        save(root/'manifest.json', manifest)
        summary['manifest'] = manifest; save(root/'summary.json', summary)
        return region, rows, manifest, summary

    def test_v6_audit_keeps_real_schema_and_all_zeros_without_atom_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'population'; _, rows, manifest, summary = self.make_population(root)
            original = (root/'manifest.json').read_bytes(); samples = (root/'samples.jsonl').read_bytes()
            with (mock.patch.object(audit.physical.line.Reconstructor, '__init__',
                                    side_effect=AssertionError('No H atom/tree constructor')),
                  mock.patch.object(audit.line.NativeLineReference, '__init__',
                                    side_effect=AssertionError('No native line constructor'))):
                result = audit.audit(root, output=Path(tmp)/'audit.json')
            self.assertTrue(result['passed']); self.assertEqual(result['source_schema'], audit.hard_input.SCHEMA)
            self.assertEqual(result['all_rows_algebra'], 3); self.assertEqual(result['finite_count'], 1)
            self.assertEqual(result['estimate']['draws'], 3)
            self.assertEqual(result['native_identity_origin'], 'not_present_in_v6_producer')
            self.assertIsNone(result['native_source_identity'])
            self.assertFalse(result['stream_contract']['declared_in_producer_manifest'])
            self.assertEqual(result['independently_checked_distinct_role_keys'], 9)
            self.assertAlmostEqual(result['estimate']['logQ'], summary['estimates']['region']['logQ'])
            self.assertEqual((root/'manifest.json').read_bytes(), original)
            self.assertEqual((root/'samples.jsonl').read_bytes(), samples)
            self.assertFalse(result['geometry_certified'])

    def test_v6_failure_retains_original_begun_draw_and_unknown_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'population'; _, rows, manifest, summary = self.make_population(root)
            rows[1]['backmapped_latent'][0] += .3
            jsonlines(root/'samples.jsonl', rows); summary['samples_sha256'] = sha(root/'samples.jsonl')
            save(root/'summary.json', summary); output = Path(tmp)/'audit.json'
            with self.assertRaises(ValueError): audit.audit(root, output=output)
            failure = audit.read(output.with_suffix('.failure.json'))
            self.assertEqual((failure['draw'], failure['begun_rows'], failure['completed_rows']), (1, 2, 1))
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError): audit.audit(root, output=output)
        with self.assertRaisesRegex(ValueError, 'Unsupported physical producer schema'):
            audit.validate_manifest({'schema': 'invented-v8'}, {})


if __name__ == '__main__':
    unittest.main()
