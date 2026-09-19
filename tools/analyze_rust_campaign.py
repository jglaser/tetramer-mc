#!/usr/bin/env python3
"""Optional independent audit using an explicitly supplied Python reference.

This is not a simulation runtime dependency. NumPy/SciPy and the unchanged
AtomicAssembly/TetramerOrder modules come from --reference. The small GSD2 reader
below uses struct/NumPy directly and never calls the Rust GSD implementation.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import mmap
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def gsd_audit(path, frames, cfg, shape):
    """Read the emitted GSD2 subset independently, with strict bounds checks.

    This is a validation reader, not a general GSD package: no frame-inheritance
    fallback is needed because this writer emits every checked field each frame.
    Layout: 256-byte header, 32-byte index entries, null-delimited name list.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    types = {1:'u1', 2:'u2', 3:'u4', 4:'u8', 5:'i1', 6:'i2', 7:'i4', 8:'i8', 9:'f4', 10:'f8', 11:'u1'}
    with Path(path).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        assert len(raw) >= 256
        magic = 0x65df65df65df65df
        endian = '<' if struct.unpack_from('<Q', raw)[0] == magic else '>'
        header = struct.unpack_from(endian+'5Q2I', raw)
        assert header[0] == magic and header[6] >> 16 == 2
        _, index_start, index_count, names_start, names_blocks, schema, version = header
        assert index_start+32*index_count <= len(raw) and names_start+64*names_blocks <= len(raw)
        names = []
        for name in raw[names_start:names_start+64*names_blocks].split(b'\0'):
            if not name: break
            names.append(name.decode('utf-8'))
        assert len(names) == len(set(names))
        chunks = {}
        for offset in range(index_start, index_start+32*index_count, 32):
            frame, n, location, m, name_id, type_id, flags = struct.unpack_from(endian+'QQQIHBB', raw, offset)
            if location == 0: break
            assert name_id < len(names) and type_id in types and flags == 0 and m > 0
            dtype = np.dtype(endian+types[type_id])
            assert location+n*m*dtype.itemsize <= len(raw)
            key = (frame, names[name_id]); assert key not in chunks
            chunks[key] = (location, int(n), int(m), dtype)
        assert {f for f, _ in chunks} == set(range(len(frames)))
        def array(frame, name):
            location, n, m, dtype = chunks[(frame, name)]
            return np.frombuffer(raw, dtype=dtype, count=n*m, offset=location).reshape(n, m).copy()
        atoms = np.asarray([a['center'] for a in shape['atoms']])
        radii = np.asarray([a['radius'] for a in shape['atoms']])
        box = np.asarray(cfg['box_lengths']); maximum_error = 0.; atom_checks = 0
        for index, frame in enumerate(frames):
            p = np.asarray([pose['position'] for pose in frame['poses']])
            q = np.asarray([pose['orientation'] for pose in frame['poses']])
            assert array(index, 'configuration/step').item() == frame['sweep']
            assert array(index, 'configuration/dimensions').item() == 3
            np.testing.assert_array_equal(array(index, 'configuration/box').ravel(), np.r_[box, [0, 0, 0]].astype('f4'))
            if (index, 'log/tetramer_mc/box_lengths') in chunks:
                assert chunks[(index, 'log/tetramer_mc/box_lengths')][3].itemsize == 8
                np.testing.assert_array_equal(array(index, 'log/tetramer_mc/box_lengths').ravel(), box)
            assert chunks[(index, 'log/tetramer_mc/body_position')][3].kind == 'f'
            assert chunks[(index, 'log/tetramer_mc/body_position')][3].itemsize == 8
            assert chunks[(index, 'log/tetramer_mc/body_orientation')][3].itemsize == 8
            np.testing.assert_array_equal(array(index, 'log/tetramer_mc/body_position'), p)
            np.testing.assert_array_equal(array(index, 'log/tetramer_mc/body_orientation'), q)
            matrices = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
            world = (np.einsum('bij,aj->bai', matrices, atoms)+p[:, None]-box/2).reshape(-1, 3)
            images = np.floor(world/box+.5).astype('i4'); wrapped = world-images*box
            actual = array(index, 'particles/position')
            assert actual.shape == wrapped.shape and actual.dtype.itemsize == 4
            err = float(np.max(np.abs(actual-wrapped))); maximum_error = max(maximum_error, err)
            assert err <= 4e-5, (path, index, err)
            np.testing.assert_array_equal(array(index, 'particles/image'), images)
            assert array(index, 'particles/N').item() == len(world)
            np.testing.assert_array_equal(array(index, 'particles/diameter').ravel(), np.tile(2*radii, len(p)).astype('f4'))
            np.testing.assert_array_equal(array(index, 'log/tetramer_mc/body_id').ravel(), np.repeat(np.arange(len(p)), len(atoms)))
            assert not np.any(array(index, 'particles/typeid'))
            assert bytes(array(index, 'particles/types').ravel()) == b'atom\0'
            atom_checks += len(world)
        return dict(passed=True, frames=len(frames), atom_positions_checked=atom_checks,
            fp64_body_poses_exactly_equal_json=True, fp32_maximum_atom_error_A=maximum_error,
            file_format_version=[version >> 16, version & 65535],
            schema_version=[schema >> 16, schema & 65535],
            scope='Independent struct/NumPy GSD2 subset reader; checks every emitted frame, atom/image/diameter/body ID and FP64 body pose. Not a general third-party application compatibility test.')


def audit_one(task):
    reference_str, campaign_str, out_str, job = task
    reference, campaign, out = map(Path, (reference_str, campaign_str, out_str))
    sys.path.insert(0, str(reference/'scripts'))
    import numpy as np
    from audit_tetramer_assembly import AtomicAssembly, pose_arrays
    from tetramer_order import TetramerOrder, graph_summary
    from analyze_precursor_exchange import prepare_templates
    folder = Path(job['directory']); cfg = read(folder/'config.json')
    manifest, summary = read(folder/'manifest.json'), read(folder/'summary.json')
    campaign_manifest = read(campaign/'manifest.json')
    assert summary['complete'] and summary['completed_sweeps'] == campaign_manifest['sweeps']
    assert summary['method'] == job['method']
    shape = read(folder/'provenance/shape.json')
    assert sha(folder/'provenance/shape.json') == manifest['shape_sha256'] == sha(cfg['shape'])
    assert sha(folder/'provenance/input-config.json') == manifest['config_sha256'] == sha(job['config'])
    assert summary['config_sha256'] == manifest['config_sha256']
    assert summary['shape_sha256'] == manifest['shape_sha256']
    assert manifest['executable_sha256'] == campaign_manifest['binary_sha256']
    if job['method'] == 'learned':
        assert sha(folder/'provenance/frozen-relative-model.json') == manifest['model_sha256'] == campaign_manifest['model_sha256']
    else: assert manifest['model_sha256'] is None
    assert summary['model_sha256'] == manifest['model_sha256']
    # Rigid member metadata is display/analysis geometry, not a kernel input.
    # Rust's effective Config excludes unused extras; use the archived shape.
    analysis_cfg = copy.deepcopy(cfg)
    if not analysis_cfg.get('rigid_members'): analysis_cfg['rigid_members'] = shape['rigid_members']
    templates = prepare_templates(cfg['monomer_shape'], reference/'results/c1c3-scaffold/motifs.json',
        reference/'results/native-neighbor-classes/classification.json')
    order, atomic = TetramerOrder(analysis_cfg, folder, templates=templates), AtomicAssembly(analysis_cfg, folder)
    frames = [json.loads(s) for s in (folder/'trajectory.jsonl').read_text().splitlines()]
    assert frames[0]['sweep'] == 0 and frames[-1]['sweep'] == summary['completed_sweeps']
    n = len(cfg['initial_poses']); assert n == 12
    active = set(); rows = []; initial = None; formed = broken = 0
    for frame in frames:
        record = order.classify(frame)
        key = lambda r: (*r['bodies'], r['motif_id'])
        entry = {key(r) for r in record['registered_tetramer_motifs'] if r['entry']}
        stay = {key(r) for r in record['registered_tetramer_motifs']}
        previous = active; active = (active&stay)|entry
        if initial is None: initial = set(active)
        else: formed += len(active-previous); broken += len(previous-active)
        g = graph_summary(n, {k[:2] for k in active}, cfg['seed_labels'])
        rows.append(dict(sweep=frame['sweep'], graph=g, registered_keys=sorted(active),
            original_motifs_retained=len(initial&active)))
    endpoints = []
    for frame in (frames[0], frames[-1]):
        p, q = pose_arrays(frame); result = atomic.frame(p, q)
        assert result['hard_valid'], (job['id'], frame['sweep'], result)
        endpoints.append(dict(sweep=frame['sweep'], **result))
    def same(a, b):
        return np.array_equal(a['position'], b['position']) and np.array_equal(a['orientation'], b['orientation'])
    assert all(same(a, b) for a, b in zip(frames[0]['poses'], cfg['initial_poses']))
    state = copy.deepcopy(frames[0]['poses']); selected = Counter(); counts = {'local': Counter(), 'global': Counter()}
    saved = {f['sweep']: f for f in frames}; permutations = set(); replay_frames = 1
    with (folder/'moves.jsonl').open() as stream:
        for serial, line in enumerate(stream, 1):
            move = json.loads(line); i = move['moving_index']; kind = move['kind']
            assert move['sweep'] == (serial-1)//n+1 and move['update_in_sweep'] == (serial-1)%n
            assert i not in permutations; permutations.add(i); selected[i] += 1
            assert same(move['old_pose'], state[i])
            counts[kind]['attempted'] += 1
            if move['hard_valid']: counts[kind]['hard_valid'] += 1
            if move['accepted']:
                assert move['hard_valid'] and move['proposed_pose'] is not None
                assert same(move['retained_pose'], move['proposed_pose'])
                counts[kind]['accepted'] += 1; state[i] = move['retained_pose']
            else: assert same(move['retained_pose'], state[i])
            if serial % n == 0:
                assert permutations == set(range(n)); permutations.clear()
                if move['sweep'] in saved:
                    assert all(same(a, b) for a, b in zip(state, saved[move['sweep']]['poses']))
                    replay_frames += 1
    assert serial == summary['counts']['selected_body_updates'] == n*summary['completed_sweeps']
    assert replay_frames == len(frames)
    assert [selected[i] for i in range(n)] == summary['counts']['selected_body_updates_by_body']
    for kind in counts:
        for key, value in counts[kind].items(): assert value == summary['counts'][kind][key]
    checkpoint = read(folder/'checkpoint.json')
    assert all(same(a, b) for a, b in zip(state, checkpoint['poses']))
    assert checkpoint['counts'] == summary['counts']
    gsd = gsd_audit(folder/'trajectory.gsd', frames, cfg, shape)
    sources = [folder/name for name in ('config.json', 'manifest.json', 'summary.json', 'trajectory.jsonl', 'trajectory.gsd', 'moves.jsonl', 'checkpoint.json')]
    sources += [Path(cfg['monomer_shape']), Path(cfg['metadata']['native_pair_motifs'])]
    result = dict(job=job, passed=True, initial_component=rows[0]['graph']['largest_component_size'],
        final_component=rows[-1]['graph']['largest_component_size'], maximum_component=max(r['graph']['largest_component_size'] for r in rows),
        rows=rows, stored_frame_motif_formations=formed, stored_frame_motif_breakages=broken,
        endpoint_hard_audit=endpoints, move_replay=dict(passed=True, updates=serial,
            frames=replay_frames, every_old_retained_saved_checkpoint_pose_exactly_equal=True,
            every_sweep_is_a_permutation=True, counts_match=True), gsd=gsd,
        sampler_cpu_seconds=summary['sampler_cpu_seconds'], wall_seconds=summary['wall_seconds'],
        timings_include_gsd=True, counts=summary['counts'], cost=summary['cost'],
        provenance=dict(passed=True, model_sha256=manifest['model_sha256'], shape_sha256=manifest['shape_sha256'],
            executable_sha256=manifest['executable_sha256'], config_sha256=manifest['config_sha256']),
        native_protocol=order.protocol,
        source_sha256={str(p):sha(p) for p in sources})
    save(out/'runs'/job['id']/'analysis.json', result)
    print(json.dumps(dict(id=job['id'], passed=True, final_component=result['final_component'],
        cpu_seconds=result['sampler_cpu_seconds'])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=ROOT/'runs/rust-validation-400')
    parser.add_argument('--out', type=Path, default=ROOT/'results/rust-validation')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error('Use1–8workers')
    reference, campaign, out = args.reference.resolve(), args.campaign.resolve(), args.out.resolve()
    manifest = read(campaign/'manifest.json'); completed = read(campaign/'summary.json')
    assert completed['complete'] and all(r['returncode'] == 0 for r in completed['records'])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [(str(reference), str(campaign), str(out), job) for job in manifest['jobs']]
        for future in as_completed([pool.submit(audit_one, task) for task in tasks]): results.append(future.result())
    results.sort(key=lambda r:(r['job']['arm'], r['job']['replicate'], r['job']['method']))
    starts = {}
    for r in results:
        cfg = read(Path(r['job']['directory'])/'config.json')
        digest = hashlib.sha256(json.dumps(cfg['initial_poses'], sort_keys=True).encode()).hexdigest()
        starts.setdefault(digest, []).append(r['job']['id'])
    for arm in ('seeded', 'seed-free'):
        for rep in (0, 1):
            pair = [r for r in results if r['job']['arm']==arm and r['job']['replicate']==rep]
            assert len(pair)==2
            configs = [read(Path(r['job']['directory'])/'config.json') for r in pair]
            assert configs[0]['initial_poses']==configs[1]['initial_poses'] and configs[0]['seed']==configs[1]['seed']
    reference_sources = [reference/'scripts'/name for name in ('audit_tetramer_assembly.py','tetramer_order.py','analyze_precursor_exchange.py')]
    source_paths = [Path(__file__), ROOT/'README.md', ROOT/'src/trajectory.rs', ROOT/'vendor/hoomd-rs/hoomd-gsd/src/file_layer.rs',
        reference/'results/c1c3-scaffold/motifs.json', reference/'results/native-neighbor-classes/classification.json', *reference_sources]
    result = dict(passed=True, runs=results, unique_initial_configurations=len(starts), initial_groups=starts,
        total_sampler_cpu_seconds=sum(r['sampler_cpu_seconds'] for r in results),
        provenance_match=True, campaign_manifest=manifest,
        source_sha256={str(p):sha(p) for p in source_paths},
        scope='Structural/integration audit. Repeated stochastic runs from two fixed starts; no equilibrium, kinetics, or cross-implementation mixing-speedup inference. Rust source files are the inspection-time snapshot; recorded executable hashes identify the actual campaign binary.')
    save(out/'analysis.json', result)
    text = ['# Independent Rust campaign audit', '',
        f'All {len(results)} runs passed: independent periodic atomic audits of initial/final states; unchanged native registry classification; exact replay of {sum(r["move_replay"]["updates"] for r in results):,} move records to every saved frame and final checkpoint; provenance hash agreement; independently decoded GSD data.', '',
        '| Start | Replicate | Method | Final / maximum registered component | CPU seconds | Wall seconds |',
        '|---|---:|---|---:|---:|---:|']
    for r in results:
        j=r['job']; text.append(f'| {j["arm"]} | {j["replicate"]} | {j["method"]} | {r["final_component"]} / {r["maximum_component"]} | {r["sampler_cpu_seconds"]:.3f} | {r["wall_seconds"]:.3f} |')
    text += ['', 'These timings include enabled GSD output. The Rust/NumPy RNGs and preparations differ from the earlier Python campaign, so these numbers do not establish a sampling or mixing speedup. A same-endpoint gate benchmark is a separate measurement.', '',
        f'The campaign uses {len(starts)} distinct initial configurations: one supplied seeded state and one supplied fluid state, each repeated with new paired RNG seeds. Replicates are stochastic repeats, not independently equilibrated preparations.', '',
        'Native classification uses the unchanged TetramerOrder proper-pose, member-center, atomic-gap and residue-patch criteria. Intrinsic tetramer contacts are excluded. Registered histories are evaluated at the stored ten-sweep cadence; intermediate formation/breakage events can be missed. Move replay verifies state bookkeeping exactly but does not independently reevaluate every acceptance decision.', '',
        f'The independent GSD2 subset reader checked {sum(r["gsd"]["frames"] for r in results)} frames and {sum(r["gsd"]["atom_positions_checked"] for r in results):,} atom positions. Every FP64 body position/orientation equals JSON exactly. Standard FP32 atom positions, images, diameters, type IDs and display body IDs agree with reconstruction. Largest FP32 position error: {max(r["gsd"]["fp32_maximum_atom_error_A"] for r in results):.6g} Å. No Python GSD package or new installation was used. This is an independently implemented reader of the emitted format subset, not a general external-viewer certification.', '',
        'All input/model/shape/config hashes agree with the archived provenance, and every run records the campaign executable hash. Analysis sources and inspected Rust/GSD layout sources are hashed separately; inspection-time source hashes alone do not certify which source revision built an executable.', '',
        'The hard audit is independent SciPy geometry with its existing 2×10⁻⁸ Å audit tolerance; production uses strict geometry. Sixteen initial/final configurations were audited. Intermediate saved configurations were classified but not separately hard-audited.', '',
        'README review: its claims of frozen proposals, all-mobile periodic runs, FP64 authoritative poses, repeated supplied starts, separate gate timing, and lack of equilibrium/nucleation conclusions agree with this audit. The README test-suite list and exact-resume claim are covered by the repository tests, not independently reproved by this campaign audit.', '',
        'Reproduce with the existing reference environment:', '', '```bash',
        f'{reference}/.venv/bin/python tools/analyze_rust_campaign.py --reference {reference} --campaign {campaign} --out {out}', '```', '']
    (out/'report.md').write_text('\n'.join(text))
    shutil.copy2(__file__, out/'analyze_rust_campaign.py')
    print(json.dumps(dict(passed=True, out=str(out), runs=len(results))))


if __name__ == '__main__': main()
