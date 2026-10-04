#!/usr/bin/env python3
"""Stream saved v6 hard-free or v7 class attempts through their own algebra.

No poses/clouds are generated and no atom overlap, native classifier or atom
search tree is evaluated. Saved H/native/exclusion unions are inputs to the
proposal algebra, not independently measured physical labels. For v7, static
shape bijection validation is performed once without overlap queries; v6 has
no producer-native geometry to certify. This receipt
cannot replace independently allocated geometry or final contact-label audits.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import sys
import threading
import time

import numpy as np
import scipy

import native_class_line_algebra_reference as algebra
import native_class_line_physical_reference as full
import native_class_line_weight_row as weights
import hard_free_physical_input as hard_input

physical, line = full.physical, algebra.line
require, close = physical.require, physical.close
SCHEMA = 'native-class-physical-all-row-algebra-audit-v1'
SOURCE_NAMES = (
    'native_class_line_physical_algebra_audit.py',
    'native_class_line_algebra_reference.py', 'native_class_line_weight_row.py',
    'native_class_line_physical_reference.py', 'hard_free_line_physical_reference.py',
    'native_class_line_reference.py', 'hard_free_line_reference.py',
    'analyze_contact_line_audit.py', 'native_contact_regions.py',
    'analyze_native_region_reference.py', 'analyze_basin_normalizers.py',
    'hard_free_physical_input.py',
)
OBLIGATIONS = [
    'Saved H/native/exclusion base unions are not independently reconstructed here.',
    'Physical hard validity and complete native/contact labels require separate geometry evidence; guide memberships are not physical labels.',
    'Normalized guide conditionals require a measurable deterministic retained-five-coordinate set function shared by generator and scorer; per-row algebra does not prove that global property.',
    'Distinct derived role keys do not establish actual RNG consumption or independence, spatial Poisson uniformity, exact thinning, or containment/nonoverlap of envelope cells.',
    'Pinned producer source/executable hashes are provenance identities, not an execution or floating-point correctness proof.',
    'No equilibrium native-weight, unseen-mode coverage, full-vessel or assembly conclusion follows from this audit.',
]


def sha(path):
    """Bound memory hashing, including potentially large attempted-row files."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _pairs_no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate JSON object key '+str(key))
        value[key] = item
    return value


def _bad_constant(value):
    raise ValueError('Nonfinite JSON constant '+value)


def loads(raw):
    return json.loads(raw, object_pairs_hook=_pairs_no_duplicates, parse_constant=_bad_constant)


def read(path):
    return loads(Path(path).read_bytes())


def _write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def _digest(value, label):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'Invalid SHA256 '+label)
    return value


class JsonLines:
    """Exact byte stream: no blank, unterminated, reordered or extra records."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open('rb')
        self.hash = hashlib.sha256()
        self.lines = self.bytes = 0
        self.last = None

    def next(self):
        offset = self.bytes
        raw = self.stream.readline()
        if not raw:
            self.last = dict(path=str(self.path), offset=offset, eof=True)
            return None
        self.hash.update(raw); self.bytes += len(raw); self.lines += 1
        self.last = dict(path=str(self.path), offset=offset, bytes=len(raw),
                         line=self.lines, sha256=hashlib.sha256(raw).hexdigest())
        require(raw.endswith(b'\n'), 'Unterminated JSONL record '+str(self.path))
        require(raw.strip(), 'Blank JSONL record '+str(self.path))
        value = loads(raw)
        require(type(value) is dict, 'JSONL record is not an object '+str(self.path))
        return value

    def close(self):
        self.stream.close()


def runtime_identity():
    paths = [Path(sys.executable).resolve(), Path(np.__file__).resolve(), Path(scipy.__file__).resolve()]
    return dict(python=sys.version, executable=str(paths[0]), platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__,
                longdouble=dict(bits=int(np.finfo(np.longdouble).bits), nmant=int(np.finfo(np.longdouble).nmant),
                                epsilon=str(np.finfo(np.longdouble).eps)),
                thread_environment={key: os.environ.get(key) for key in
                                    ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                file_sha256={str(p): sha(p) for p in paths},
                scope='Python executable and package initializer bytes plus versions/platform; not a hash of every system/shared library')


def input_backend(manifest):
    """Select the actual producer schema; never rewrite rows or manifests."""
    if manifest.get('schema') == full.SCHEMA:
        return weights
    require(manifest.get('schema') == hard_input.SCHEMA, 'Unsupported physical producer schema')
    return hard_input


def validate_manifest(manifest, region):
    return input_backend(manifest).validate_manifest(manifest, region)


def validate_row(row, *, expected_draw, manifest, region):
    return input_backend(manifest).validate_row(row, expected_draw=expected_draw,
                                               manifest=manifest, region=region)


def stream_contract(manifest):
    if manifest['schema'] == full.SCHEMA:
        return full.validate_stream_contract(manifest)
    require(manifest['schema'] == hard_input.SCHEMA, 'Unsupported stream schema')
    return dict(hash_domain='tetramer-uniform-latent-region-v1',
        key_fields=['master seed u64 little endian', 'draw u64 little endian',
                    'cloud index u64 little endian', 'role UTF-8 bytes'],
        proposal_role='latent', proposal_cloud_index=0, physical_cloud_role='cloud',
        physical_cloud_indices=[0, 1], declared_in_producer_manifest=False,
        scope='Reconstruct the existing v6 source convention; derived role keys do not prove actual RNG execution or independence.')


def _provenance(root, bind):
    manifest = read(bind(root/'manifest.json'))
    if manifest.get('schema') == hard_input.SCHEMA:
        return hard_input.provenance(root, bind)
    summary = read(bind(root/'summary.json'))
    require(manifest['schema'] == full.SCHEMA and manifest['guide_schema'] == full.GUIDE_SCHEMA
            and manifest['proposal_kind'] == full.PROPOSAL_KIND, 'Wrong v7 physical class-guide schema')
    require(summary['complete'] is True and summary['manifest'] == manifest and not (root/'failure.json').exists(),
            'Incomplete or failed physical population')
    require(manifest.get('resume_supported') is False, 'Unsupported continuation contract')
    _digest(manifest['executable_sha256'], 'producer executable')
    for name, key in [('input-config.json', 'config_sha256'), ('region.json', 'region_sha256'),
                      ('shape.json', 'shape_sha256'), ('importance-guide.json', 'importance_guide_sha256'),
                      ('source-bundle.json', 'source_bundle_sha256')]:
        bind(root/'provenance'/name, manifest[key])
    region, guide, config, shape = [read(root/'provenance'/name) for name in
                                   ('region.json', 'importance-guide.json', 'input-config.json', 'shape.json')]
    provenance = manifest['compiled_native']
    compiled = read(bind(root/'provenance/compiled-native.json', provenance['compiled_sha256']))
    _digest(provenance['source_definition_sha256'], 'native definition')
    require(type(provenance['source_input_sha256']) is dict, 'Invalid native source digest map')
    for name, digest in provenance['source_input_sha256'].items(): _digest(digest, name)
    witness = full.validate_shape_witness(compiled, shape, provenance['shape_compatibility'],
                                        provenance['compiled_sha256'], manifest['shape_sha256'])
    require(guide['compiled_native']['sha256'] == provenance['compiled_sha256']
            and compiled['source_definition_sha256'] == provenance['source_definition_sha256']
            and compiled['source_input_sha256'] == provenance['source_input_sha256'], 'Native provenance differs')
    require(guide['region_sha256'] == manifest['region_sha256'] and region['shape_sha256'] == manifest['shape_sha256']
            and region['gaussian_chart']['shape_sha256'] == manifest['shape_sha256']
            and guide['shape_sha256'] == manifest['shape_sha256'], 'Guide/chart/shape binding differs')
    fixed = region.get('physical_fixed_neighbors', [region['fixed_neighbor']])
    require(fixed == config['fixed_poses'] == manifest['physical_fixed_neighbors'] == guide['fixed_poses'] == compiled['fixed_poses']
            and region['fixed_neighbor'] == manifest['chart_anchor'], 'Changed fixed scaffold or chart anchor')
    for rk, ck in [('capture_center', 'capture_center'), ('capture_radius', 'capture_radius'),
                   ('depletant_radius', 'depletant_radius'), ('activity', 'reservoir_density'), ('physical_metric', 'metadata')]:
        require(region[rk] == config[ck], 'Changed physical '+rk)
        if rk in ('capture_center', 'capture_radius', 'depletant_radius'):
            require(guide[rk] == config[ck], 'Changed guide geometry '+rk)
    weights.validate_manifest(manifest, region)
    law = algebra.AlgebraLaw(region, guide, config)
    require(manifest['importance_uniform_probability'] == law.alpha and manifest['importance_component_count'] == len(law.weights),
            'Proposal mixture differs')
    close(manifest['log_latent_ball_volume'], law.logvolume, 'Latent ball volume differs')
    close(manifest['log_latent_shell_volume'], law.logvolume, 'Latent shell volume differs')
    require(type(summary['samples']) is int and summary['samples'] == manifest['samples'], 'Changed unconditional denominator')
    bind(root/'samples.jsonl', summary['samples_sha256'])
    bind(root/'attempts.jsonl', summary['attempts_sha256'])
    return manifest, summary, region, config, law, witness


def _check_row(row, index, manifest, region, config, law):
    if manifest.get('schema') == hard_input.SCHEMA:
        return hard_input.check_row(row, index, manifest, region, config, law)
    accounting = weights.validate_row(row, expected_draw=index, manifest=manifest, region=region)
    u = algebra.finite_array(row['latent'], (6,), 'latent coordinate')
    result = law.evaluate_saved(u, row['native_class_line_density'])
    position, rotation, jac = result['position'], result['rotation'], result['log_physical_jacobian']
    saved_position, saved_rotation = line.pose_arrays(row['pose'])
    close(saved_position, position, 'Physical position differs')
    close(saved_rotation, rotation, 'Physical orientation differs')
    backmap = law.inverse_pose(saved_position, saved_rotation)
    close(backmap, u, 'Independent inverse chart differs')
    close(row['backmapped_latent'], backmap, 'Saved inverse chart differs')
    close(row['backmapped_radius'], np.linalg.norm(backmap), 'Inverse radius differs')
    close(row['log_physical_jacobian'], jac, 'Physical Jacobian differs')
    close(row['physical_jacobian'], math.exp(jac), 'Exponentiated Jacobian differs')
    require(math.isfinite(result['log_density']), 'Generated row has zero/nonfinite proposal density')
    close(row['log_proposal_density'], result['log_density'], 'Complete proposal density differs', atol=2e-7, rtol=1e-11)
    if row['proposal_branch'] == 'native-class-line' and law.beta == 1.:
        require(row['native_class_line_draw']['conditional'] is True,
                'Gaussian branch with beta=1 must record conditional processing, including fallback')
    generated = law.verify_draw(row['native_class_line_draw'], u, result)
    capture = bool(np.linalg.norm(position-law.capture_center) <= law.capture_radius)
    require(row['capture_valid'] == capture, 'Capture predicate differs')
    original_q = physical.native.native_q(region['physical_metric'], row['pose'])
    close(row['q'], original_q, 'Original physical q differs')
    q_region = dict(region)
    if q_region.get('maximum_original_q') is None: q_region.pop('maximum_original_q', None)
    require(row['region_valid'] == physical.q_contains(original_q, q_region), 'Original q-window differs')
    require(row['shell_valid'] == physical.shell_contains(float(np.linalg.norm(u)), region), 'Independent shell predicate differs')
    # These are consistency checks of recorded guide geometry, not fresh atom
    # predicates or physical native labels. N/E are deliberately not reported.
    hard_memberships = []
    for axis in result['axes']:
        s = result['raw_coordinates'][axis['axis']]
        if 'segment' in axis and axis['segment'][0] <= s <= axis['segment'][1]:
            hard_memberships.append(line.contains(axis['hard_free_intervals'], s))
    if hard_memberships:
        require(all(v == hard_memberships[0] for v in hard_memberships), 'Saved H membership differs across axes')
        require(hard_memberships[0] == row['hard_valid'], 'Saved H membership differs from recorded physical hard flag')
    maxima = dict(accounting.get('maxima', {}))
    maxima.update(inverse_cdf=generated['inverse_error'],
                  interval_endpoints=max(result['interval_error'], generated['endpoint_error']),
                  log_proposal_density=abs(row['log_proposal_density']-result['log_density']),
                  log_physical_jacobian=abs(row['log_physical_jacobian']-jac),
                  latent_backmap=float(np.max(np.abs(backmap-u))), original_q=abs(row['q']-original_q))
    return accounting, maxima, dict(hard_membership_axes=len(hard_memberships),
        interval_selection_ambiguous=int(generated.get('interval_selection_ambiguous', False)),
        component_branches=result.get('component_branches', 0),
        fallback_component_branches=result.get('fallback_component_branches', 0))


def audit(root, *, output, journal=None):
    """Audit one completed, frozen population once; never resume/retry rows.

    Writes an exclusive success receipt or a separate failure receipt. The
    fsynced journal starts before metadata validation and preserves every begun
    row, including malformed/truncated input lines. Only four float64 log-weight
    columns (z,h,W1,W2) persist across rows; no JSON populations are retained.
    """
    require(threading.current_thread() is threading.main_thread(), 'Audit must run in one main thread')
    root, output = Path(root).resolve(), Path(output).resolve()
    journal = output.with_suffix('.journal.jsonl') if journal is None else Path(journal).resolve()
    failure = output.with_suffix('.failure.json')
    require(len({output, journal, failure}) == 3, 'Audit output paths must differ')
    for path in (output, journal, failure):
        require(not path.exists(), 'Audit output already exists '+str(path))
        require(path.parent.is_dir(), 'Audit output parent must exist '+str(path.parent))
    started = time.process_time(); wall = time.monotonic()
    bindings = {}; source_bindings = {}; readers = []; completed = begun = 0
    active = None; phase = 'metadata'; runtime = None
    ledger = journal.open('x')
    def emit(value):
        ledger.write(json.dumps(value, allow_nan=False)+'\n'); ledger.flush(); os.fsync(ledger.fileno())
    def bind(path, expected=None):
        path = Path(path).resolve(); actual = sha(path)
        if expected is not None:
            _digest(expected, str(path)); require(actual == expected, 'Changed bound input '+str(path))
        bindings[str(path)] = actual
        return path
    def interrupted(signum, _frame):
        raise SystemExit(128+signum)
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        emit(dict(state='started', schema=SCHEMA, root=str(root), output=str(output), new_pose_draws=0, new_Poisson_clouds=0))
        source_root = Path(__file__).resolve().parent
        for name in SOURCE_NAMES:
            expected = (source_root/name).resolve()
            module = sys.modules.get(Path(name).stem)
            if name == Path(__file__).name:
                actual = Path(__file__).resolve()
            else:
                require(module is not None and getattr(module, '__file__', None), 'Missing source module '+name)
                actual = Path(module.__file__).resolve()
            require(actual == expected, 'Imported source is outside bound closure '+name)
            p = bind(actual); source_bindings[str(p)] = bindings[str(p)]
        runtime = runtime_identity()
        for path, digest in runtime['file_sha256'].items(): bind(path, digest)
        manifest, summary, region, config, law, witness = _provenance(root, bind)
        count = manifest['samples']
        emit(dict(state='inputs_bound', input_sha256=bindings, source_sha256=source_bindings,
                  runtime=runtime, samples=count, setup_static_validation=witness,
                  independently_reconstructed_geometry_rows=0, geometry_certified=False))
        z = np.full(count, -np.inf); h = np.full(count, -np.inf); pairs = np.full((count, 2), -np.inf)
        counts, maxima, diagnostics = Counter(), Counter(), Counter(); raw_points = 0
        samples, attempts = JsonLines(root/'samples.jsonl'), JsonLines(root/'attempts.jsonl')
        readers.extend((samples, attempts)); phase = 'rows'
        for index in range(count):
            active = index; begun += 1; tick = time.process_time()
            emit(dict(state='begin', draw=index, sample_offset=samples.bytes, attempt_offset=attempts.bytes))
            attempt = attempts.next()
            require(attempt is not None and type(attempt.get('draw')) is int
                    and attempt == dict(draw=index, state='begin'), 'Missing/repeated/reordered attempt record')
            row = samples.next()
            require(row is not None, 'Missing attempted sample row')
            accounting, errors, extra = _check_row(row, index, manifest, region, config, law)
            z[index], h[index], pairs[index] = accounting['z'], accounting['h'], accounting['pairs']
            counts.update(accounting['counters']); diagnostics.update(extra); raw_points += accounting['raw_points']
            for key, value in errors.items(): maxima[key] = max(maxima[key], value)
            completed += 1
            emit(dict(state='complete', draw=index, sample_record=samples.last, attempt_record=attempts.last,
                      cpu_seconds=time.process_time()-tick, finite_weight=math.isfinite(accounting['z']),
                      role_keys=accounting['role_keys'], maximum_errors=errors, algebra_diagnostics=extra))
            active = None
        phase = 'final_validation'
        require(attempts.next() is None, 'Extra attempt record outside declared allocation')
        require(samples.next() is None, 'Extra sample record outside declared allocation')
        for reader, key in ((samples, 'samples_sha256'), (attempts, 'attempts_sha256')):
            require(reader.lines == count and reader.hash.hexdigest() == summary[key], 'Consumed stream digest/count differs '+reader.path.name)
        require(counts['attempted'] == count, 'Changed attempted denominator')
        for key in ('capture_rejected', 'hard_rejected', 'region_rejected', 'shell_rejected'):
            require(type(summary[key]) is int and summary[key] == counts[key], 'Changed rejection summary '+key)
        require(type(summary['raw_points']) is int and summary['raw_points'] == raw_points, 'Raw cloud total differs')
        estimate, hard_estimate = physical.statistics.moments(z), physical.statistics.moments(h)
        physical.check_estimate(summary['estimates']['region'], estimate)
        physical.check_estimate(summary['estimates']['hard_region'], hard_estimate)
        cpu = summary['sampler_cpu_seconds']
        require(type(cpu) in (int, float) and math.isfinite(cpu) and cpu >= 0, 'Invalid sampler CPU')
        for path, digest in bindings.items(): require(sha(path) == digest, 'Input/source/runtime changed during audit '+path)
        require(not (root/'failure.json').exists(), 'Producer failure appeared during audit')
        result = dict(schema=SCHEMA, source_schema=manifest['schema'], complete=True, passed=True, root=str(root), seed=manifest['seed'],
            samples=count, all_rows_algebra=count, independently_reconstructed_geometry_rows=0,
            geometry_certified=False, physical_contact_labels_certified=False, full_geometry_row_ids=[],
            finite_count=estimate['nonzero'], estimate=estimate, hard_region=hard_estimate,
            paired_noise=physical.statistics.paired_noise(z, pairs), counts=dict(counts),
            maximum_errors=dict(maxima), algebra_diagnostics=dict(diagnostics), raw_points=raw_points,
            input_sha256=bindings, source_sha256=source_bindings, runtime=runtime,
            samples_sha256=summary['samples_sha256'], attempts_sha256=summary['attempts_sha256'],
            executable_sha256=manifest['executable_sha256'], source_bundle_sha256=manifest['source_bundle_sha256'],
            native_source_identity=manifest.get('compiled_native'), setup_static_validation=witness,
            native_identity_origin='producer_compiled_native' if manifest['schema'] == full.SCHEMA else 'not_present_in_v6_producer',
            native_source_files_reopened=False, chart_factor_validation=law.chart_factor_validation,
            sampler_cpu_seconds=cpu, analysis_cpu_seconds=time.process_time()-started,
            wall_seconds=time.monotonic()-wall, retained_numeric_bytes=z.nbytes+h.nbytes+pairs.nbytes,
            stream_contract=stream_contract(manifest),
            independently_checked_distinct_role_keys=3*count,
            role_key_scope='Three distinct derived role keys per exact attempted ID; unique input tuples across rows, without claiming actual random-stream independence.',
            new_pose_draws=0, new_Poisson_clouds=0, input_unchanged_after_audit=True,
            interruption_contract='Python exceptions, SIGINT and SIGTERM record failure without retry; SIGKILL or machine failure can only leave the fsynced begun-row journal.',
            scope='All attempted-row proposal algebra and physical-weight arithmetic conditional on recorded geometry; no independent atom geometry or final native/contact-label audit.',
            implementation_obligations=OBLIGATIONS)
        emit(dict(state='finished', complete=True, passed=True, all_rows_algebra=count, counts=dict(counts)))
        ledger.close(); result['journal'] = dict(path=str(journal), sha256=sha(journal))
        _write(output, result)
        return result
    except BaseException as exc:
        event = dict(state='failed', phase=phase, draw=active, begun_rows=begun, completed_rows=completed,
                     error_type=type(exc).__name__, error=str(exc),
                     input_stream_context=[r.last for r in readers], cpu_seconds=time.process_time()-started)
        if not ledger.closed: emit(event); ledger.close()
        failed = dict(schema=SCHEMA, complete=False, passed=False, root=str(root), **event,
                      all_rows_algebra=completed, independently_reconstructed_geometry_rows=0,
                      geometry_certified=False, physical_contact_labels_certified=False,
                      input_sha256=bindings, source_sha256=source_bindings, runtime=runtime,
                      journal=dict(path=str(journal), sha256=sha(journal)), new_pose_draws=0, new_Poisson_clouds=0)
        if not failure.exists(): _write(failure, failed)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        for reader in readers: reader.close()
        if not ledger.closed: ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', '--root', dest='root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--journal', type=Path)
    args = parser.parse_args()
    result = audit(args.root, output=args.out, journal=args.journal)
    print(json.dumps(dict(complete=True, passed=True, all_rows_algebra=result['all_rows_algebra'],
                         independently_reconstructed_geometry_rows=0, geometry_certified=False)))


if __name__ == '__main__':
    main()
