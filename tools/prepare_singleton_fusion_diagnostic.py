#!/usr/bin/env python3
"""Prepare the fixed context-0 passive fusion diagnostic; never execute it.

The first original predeclared context is used, with its shared source and
four original prepared starts. No live trajectory or posterior audit is read.
The completed initial native audit supplies only the stated initial labels.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import hashlib
import math
from pathlib import Path
import signal
import sys

from analyze_mobile_native_pocket import local_sources
from bind_conditional_native_audit import Bindings, _initial_receipt
import run_native_class_physical_campaign as driver

read, require, sha, write = driver.read, driver.require, driver.sha, driver.write
SCHEMA = 'singleton-fusion-diagnostic-preparation-v1'
CONTEXT = dict(name='whole_27_132', root=27, child=132, anchor=228)
CAPS = dict(constructions=10, fits=40960, center_checks=2560, core_overlap_calls=673280)
LIMITS = dict(cpu_limit_seconds=600, wall_limit_seconds=1200, address_space_limit_bytes=8*2**30)
OLIGOMER = dict(multi_contact_mass=.8, max_mismatch=12., pair_distance_A=8., pair_angle_degrees=60.,
                max_candidates=4096, max_hard_checks=256, max_components=32)
CHOICE = ('Context 0 is the first original predeclared conditional context, not selected using partial '
          'native trajectories or posterior outcomes. The completed initial audit finds no mobile-related '
          'native bond here or in the four original prepared starts. This is one context without initial native mobile contacts; '
          'context 2 is a separate native-retention control. No general mixing or assembly conclusion follows.')


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def recheck(bindings):
    for path, expected in bindings.files.items(): require(sha(path) == expected, 'Input changed: '+path)


def pose(value):
    require(type(value) is dict and set(value) == {'position', 'orientation'}, 'Expected complete pose')
    for field, size in [('position', 3), ('orientation', 4)]:
        require(type(value[field]) is list and len(value[field]) == size
                and all(type(v) in (int, float) and math.isfinite(v) for v in value[field]), 'Invalid finite pose')
    require(abs(math.fsum(v*v for v in value['orientation'])-1.) <= 1e-8, 'Unnormalized pose quaternion')


def initial_states(original, reference_config, source_config, source_frame, initial, starts):
    """Pure original-label substitution; no geometry or recentering."""
    require(original['schema'] == 'evolving-dimer-benchmark-v1' and len(original['contexts']) == 4,
            'Wrong original context inventory')
    context = original['contexts'][0]
    require(all(context[k] == v for k, v in CONTEXT.items()), 'First predeclared context changed')
    require(reference_config['source_config']['sha256'] == original['source_config']['sha256']
            and reference_config['source_frame']['sha256'] == original['source_frame']['sha256']
            and reference_config['shape']['sha256'] == original['shape']['sha256'], 'Reference/source identity differs')
    source = source_frame['poses']
    require(len(source) == 264 and source == source_config['initial_poses']
            and source_frame['boundary'] == 'spherical'
            and source_config['boundary'] == dict(kind='spherical', radius=original['physical']['wall_radius'])
            and source_frame['spherical_wall_radius'] == original['physical']['wall_radius'], 'Source poses/wall differ')
    require('Stored poses use sphere-centered coordinates' in source_config['coordinate_frame_convention'],
            'Missing explicit stored sphere-center convention')
    require(original['physical']['activity'] == source_config['reservoir_density'] == .0275
            and original['physical']['depletant_radius'] == source_config['depletant_radius'] == 1.4,
            'Conditional physical target differs')
    for p in source: pose(p)
    members = [CONTEXT['root'], CONTEXT['child']]
    require(initial['states'][0]['state_id'] == 'source' and not any(
        a in members or b in members for a, b, _ in initial['states'][0]['reference_native_keys']),
        'Completed source audit does not establish absence of initial native mobile contacts')
    require(len(initial['states']) == 17 and all(s['reference_native_keys'] == [] for s in initial['states'][1:]),
            'Completed sixteen-start initial native evidence differs')
    require([v['stream'] for v in starts] == list(range(4)), 'Exactly four original ordered starts required')
    result = [dict(id='source', kind='source', poses=copy.deepcopy(source), provenance=original['source_frame'])]
    for stream, item in enumerate(starts):
        record, value = item['record'], item['value']
        require(item['context_index'] == value['context_index'] == 0 and value['stream'] == stream
                and value['status'] == 'prepared' and value['is_equilibrium_sample'] is False
                and value['source'] == [source[i] for i in members] and len(value['selected']) == 2,
                'Prepared source, labels or status differs')
        for p in value['selected']: pose(p)
        positions = copy.deepcopy(source)
        for member, selected in zip(members, value['selected']): positions[member] = copy.deepcopy(selected)
        require(all(positions[i] == source[i] for i in range(264) if i not in members), 'A spectator changed')
        result.append(dict(id=f'prepared-{stream}', kind='prepared', poses=positions, provenance=record))
    return result


def bind_build(bindings, witness_ref):
    witness = bindings.load(witness_ref)
    require(witness['schema'] == 'singleton-fusion-build-witness-v1' and witness['complete'] is True
            and witness['passed'] is True and witness['production_executable_unchanged'] is True,
            'Missing completed isolated build witness')
    for key in ('executable', 'source_bundle', 'cli_source', 'validation'):
        ref = witness[key]
        require(set(ref) == {'path', 'sha256'}, 'Exact build BoundFile required')
        bindings.bind(ref['path'], ref['sha256'])
    validation = read(witness['validation']['path'])
    require(validation['complete'] is True and validation['passed'] is True, 'Build/test validation failed')
    executable = Path(witness['executable']['path'])
    require(executable.name == 'singleton-fusion-diagnostic' and os.access(executable, os.X_OK), 'Wrong/non-executable diagnostic binary')
    source = Path(witness['cli_source']['path']).resolve()
    require(source.name == 'singleton-fusion-diagnostic.rs' and source.parent.name == 'bin' and source.parent.parent.name == 'src',
            'Wrong diagnostic source identity')
    repository = source.parents[2]
    bundle = read(witness['source_bundle']['path'])
    require(bundle['schema'] == 1 and bundle['files']['src/bin/singleton-fusion-diagnostic.rs']['sha256'] == witness['cli_source']['sha256'],
            'Binary source bundle does not contain this CLI')
    for name, entry in bundle['files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts
                and hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'Invalid compiled source bundle entry')
        bindings.bind(repository/path, entry['sha256'])
    return witness


def validate_tests(bindings, validation_ref, paths):
    receipt = bindings.load(validation_ref)
    require(receipt['complete'] is True and receipt['passed'] is True
            and receipt['source_before'] == receipt['source_after'], 'Preparer validation failed or changed sources')
    for path in paths:
        path = Path(path).resolve()
        expected = receipt['source_before'].get(str(path))
        require(expected is not None, 'Preparer validation omitted source: '+str(path))
        bindings.bind(path, expected)
    return receipt


def admit(original_path, initial_root, witness_path, validation_path):
    """Read completed metadata and immutable initial states only; create nothing."""
    b = Bindings(); original_ref = b.bind(original_path); original = b.load(original_ref)
    initial_plan = read(Path(initial_root).resolve()/'audit-plan.json')
    initial = _initial_receipt(b, initial_root, initial_plan['config'])
    inherited = b.load(initial['plan']['config'])
    require(inherited['inherited_campaign']['config'] == original_ref
            and initial['plan']['contexts'] == original['contexts']
            and initial['plan']['source_frame'] == original['source_frame']
            and initial['plan']['shape'] == original['shape'], 'Completed initial audit belongs to another benchmark')
    source_frame, source_config, ref_config = (b.load(original[k]) for k in ('source_frame', 'source_config', 'reference_config'))
    require(b.load(ref_config['source_config']) == source_config, 'Reference Config bytes/poses differ')
    b.load(original['shape']); b.load(original['atlas']); b.load(original['source_freeze_manifest'])
    starts = []
    for item in initial['plan']['starts']:
        if item['context_index'] == 0:
            require(item['members'] == [27, 132], 'Initial prepared labels changed')
            starts.append(dict(item, value=b.load(item['record'])))
    states = initial_states(original, ref_config, source_config, source_frame, initial['result'], starts)
    witness_ref = b.bind(witness_path); witness = bind_build(b, witness_ref)
    validation_ref = b.bind(validation_path)
    sources = {}
    for entry in (Path(__file__), Path(driver.__file__)):
        for name, path in local_sources(entry).items():
            require(name not in sources or sources[name] == path, 'Ambiguous preparation/controller source')
            sources[name] = path; b.bind(path)
    validate_tests(b, validation_ref, [*sources.values(), Path(__file__).with_name('test_prepare_singleton_fusion_diagnostic.py')])
    python = str(Path(sys.executable).absolute()); b.bind(Path(python).resolve())
    recheck(b)
    return dict(bindings=b, original=original, original_ref=original_ref, source_config=source_config,
        states=states, initial=initial, witness=witness, witness_ref=witness_ref,
        validation=validation_ref, sources=sources, python=python)


def materialize(root, context):
    b, original = context['bindings'], context['original']
    common, code, states_dir = root/'common', root/'code', root/'states'
    for directory in (common, code, states_dir): directory.mkdir()
    copied = {}
    def copy_file(path, destination, executable=False):
        path = Path(path).resolve(); expected = b.files[str(path)]
        data = path.read_bytes(); require(hashlib.sha256(data).hexdigest() == expected, 'Input changed before copy')
        with destination.open('xb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        if executable: destination.chmod(0o755)
        require(sha(destination) == expected, 'Copied bytes differ'); copied[str(destination)] = expected
        return dict(path=str(destination), sha256=expected)
    for name, path in context['sources'].items(): copy_file(path, code/name)
    binary = copy_file(context['witness']['executable']['path'], code/'singleton-fusion-diagnostic', executable=True)
    bundle = copy_file(context['witness']['source_bundle']['path'], common/'source-bundle.json')
    shape = copy_file(original['shape']['path'], common/'shape.json')
    model = copy_file(original['atlas']['path'], common/'atlas.json')
    config = copy.deepcopy(context['source_config']); config['shape'] = shape['path']
    write(common/'config.json', config)
    require({k: v for k, v in config.items() if k != 'shape'} == {k: v for k, v in context['source_config'].items() if k != 'shape'},
            'Materialized Config changed beyond the bound shape path')
    states = []; constructions = []
    for value in context['states']:
        path = states_dir/(value['id']+'.json')
        write(path, dict(schema='singleton-fusion-fixed-state-v1', coordinate_frame='sphere_centered',
                         source=original['source_frame'], original_record=value['provenance'], poses=value['poses']))
        states.append(dict(id=value['id'], kind=value['kind'], file=reference(path), poses_pointer='/poses'))
        for moving, other in ((27, 132), (132, 27)):
            constructions.append(dict(id=f'{value["id"]}-{moving}', state=value['id'], moving=moving, neighbors=[other, 228]))
    manifest = dict(schema='singleton-fusion-diagnostic-manifest-v1', context=CONTEXT['name'],
        coordinate_frame='sphere_centered', wall_center=[0., 0., 0.], output=str(root/'diagnostic'),
        config=reference(common/'config.json'), model=model, shape=shape, members=[27,132], anchor=228,
        states=states, constructions=constructions, oligomer=OLIGOMER, caps=CAPS,
        witness=dict(executable_sha256=binary['sha256'], compiled_source_bundle_sha256=bundle['sha256'],
                     cli_source_sha256=context['witness']['cli_source']['sha256']))
    write(root/'manifest.json', manifest); manifest_ref = reference(root/'manifest.json')
    recheck(b)
    for path, expected in copied.items(): require(sha(path) == expected, 'Frozen copy changed')
    files = dict(b.files)
    files.update({str(path): sha(path) for path in root.rglob('*') if path.is_file()})
    protocol = dict(schema=SCHEMA, complete=True, launched=False, context_index=0, choice=CHOICE,
        original_config=context['original_ref'], initial_audit=context['initial']['bindings'],
        build_witness=context['witness_ref'], validation=context['validation'],
        original_starts=[v['provenance'] for v in context['states'][1:]], source_frame=original['source_frame'],
        manifest=manifest_ref, binary=binary, source_bundle=bundle, files=files,
        limits=dict(LIMITS, maximum_workers=1, threads=1), caps=CAPS, python=context['python'],
        new_pose_draws=0, new_native_queries=0, geometry_queries_during_preparation=0,
        coordinate_note='Stored sphere-centered poses retained exactly; coordinate_wall_center is not subtracted.',
        config_transformation='Only shape path rewritten to the identical bound physical shape bytes.')
    write(root/'protocol.json', protocol); protocol_ref = reference(root/'protocol.json')
    files = dict(files); files[protocol_ref['path']] = protocol_ref['sha256']
    job = dict(id='context0-passive-fusion', population='whole_27_132', phase='geometry',
        argv=[binary['path'], '--manifest', manifest_ref['path'], '--manifest-sha256', manifest_ref['sha256'], '--out', manifest['output']],
        terminal=dict(path=str(root/'diagnostic/summary.json'), success_contract='complete_and_passed'), **LIMITS)
    execution = dict(schema='native-class-physical-execution-v1', root=str(root), maximum_workers=1, threads=1,
        files=files, executable_resolutions={binary['path']:binary['path']}, jobs=[job], preparation_receipt=str(root/'preparation.json'),
        scope=CHOICE+' Passive catalogue construction only; no proposals, bath, native classifier or trajectory replay.')
    write(root/'execution-plan.json', execution)
    recheck(b)
    for path, expected in files.items(): require(sha(path) == expected, 'Frozen input changed before publication')
    receipt = dict(schema=SCHEMA, complete=True, launched=False, protocol=protocol_ref,
        execution_plan=reference(root/'execution-plan.json'), manifest=manifest_ref, context_index=0, caps=CAPS,
        geometry_queries_during_preparation=0, new_pose_draws=0)
    write(root/'preparation.json', receipt)
    return receipt


def prepare(original_path, initial_root, witness_path, validation_path, *, out):
    root = Path(out).resolve(); require(not root.exists(), 'Fresh preparation root required')
    context = admit(original_path, initial_root, witness_path, validation_path)
    recheck(context['bindings']); root.mkdir()
    write(root/'preparation-attempt.json', dict(schema=SCHEMA, launched=False, input_sha256=context['bindings'].files))
    handlers = {}
    def interrupted(signum, _frame): raise InterruptedError('Preparation signal '+str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT): handlers[sig] = signal.signal(sig, interrupted)
        return materialize(root, context)
    except BaseException as error:
        write(root/'preparation-failure.json', dict(schema=SCHEMA, complete=False, launched=False,
            error_type=type(error).__name__, error=str(error), retries=0,
            scope='Partial metadata directory retained. Never launch or reuse.'))
        raise
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-config', type=Path, required=True)
    parser.add_argument('--initial-audit', type=Path, required=True)
    parser.add_argument('--build-witness', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.original_config, args.initial_audit, args.build_witness, args.validation, out=args.out)
