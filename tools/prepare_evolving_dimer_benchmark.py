#!/usr/bin/env python3
"""Freeze a conditional two-mobile-body benchmark; never launch or sample it.

Three exclusive operations freeze inputs, bind a validated executable, and bind
completed geometric preparations. No energy, native label, or pilot outcome is
used to choose the held-out panel. Files are created exclusively, never replaced.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
MASTER = 6100300601
ARMS = ['local', 'unguided', 'm4']
INITIALIZATIONS = ['source', 'proposal_prepared']
PREVIOUS_PAIRS = [[13, 215], [18, 234], [0, 1], [8, 255]]
EXPECTED_PAIRS = [[27, 132], [32, 110], [9, 24], [11, 246]]
PANEL_SHA = 'a9f758a3c7a1411931584a355d8bf0dbc29c9a0460aef46c49e51678667db6c1'
REFERENCE_SHA = '0bde358181834b1848b2b6dbebe84d1b7747bb9faeb6d7a0ac5ce79b3b018237'
SHAPE_SHA = 'c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
FRAME_SHA = 'e69e6c4696eb7d8fece5c2ef025364620dc21d7db7a5d92ff5ff6a896ed59b2e'
SOURCE_CONFIG_SHA = 'f310ec8cf3ff4aa4b915be7dec16643b36d59e9293acf504eeb9f2dfbec1ddea'
SOURCE_FREEZE_SHA = '0561abfd14eb5f5134907490ebbb8195dee8b281d25cca965d377b32607097fb'
FFT_SHA = 'c460dc61fb7bf9e76f1d4dca77987b5886ea82cdda5d703ea5de6a25733fcb08'
SCALED_FFT_SHA = 'f6be7889418ae39c078719c64af989aaa7c7bbbe004989badaa56d34121b8e49'
PATCH_SHA = 'a1dee6d26d86852afadd21de231271ac73617bdfff884b07f129700e5c25e2bc'
EXAMPLE = 'examples/evolving_dimer_benchmark.rs'
TOOLS = ['tools/prepare_evolving_dimer_benchmark.py',
         'tools/test_prepare_evolving_dimer_benchmark.py',
         'tools/analyze_evolving_dimer_benchmark.py',
         'tools/test_analyze_evolving_dimer_benchmark.py',
         'tools/analyze_contact_efficiency.py', 'tools/mobile_posterior_metrics.py',
         'tools/run_evolving_dimer_benchmark.py', 'tools/test_evolving_dimer_runner.py',
         'tools/audit_evolving_dimer_preparation.py', 'tools/test_audit_evolving_dimer_preparation.py']


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def checked_file(value):
    path = Path(value['path']).resolve()
    require(path.is_file() and sha(path) == value['sha256'], 'Changed bound input: '+str(path))
    if 'bytes' in value:
        require(path.stat().st_size == value['bytes'], 'Bound byte count differs: '+str(path))
    return path


def runtime_seed(master, context, initialization, stream, block, role):
    text = f'evolving-dimer-v1/{master}/{context}/{initialization}/{stream}/{block}/{role}'
    return hashlib.sha256(text.encode()).digest()


def seed(master, context, initialization, stream, block, role):
    # This u64 is ONLY a queue-order key; never the Rust runtime seed.
    return int.from_bytes(runtime_seed(master, context, initialization, stream, block, role)[:8], 'big')


def allocation():
    blocks = 512+4096
    return dict(schema='evolving-dimer-allocation-v1', contexts=4, arms=ARMS[:],
        initializations=INITIALIZATIONS[:], streams=4, chains=96,
        warmup_blocks=512, production_blocks=4096, blocks_per_chain=blocks,
        local_attempts_per_block=4, local_member_order=[0, 1, 0, 1],
        extra_dimer_per_block=dict(local=0, unguided=1, m4=1),
        local_attempts=96*blocks*4, dimer_attempts=64*blocks,
        retained_block_observations=96*blocks, production_observations=96*4096,
        initial_observations=96, preparation_streams=16, prep_outer_cap=256,
        maximum_preparation_outers=16*256, cloud_banks=32,
        raw_points_per_cloud=16384, raw_cloud_points=32*16384,
        raw_cloud_bytes=32*16384*24, master_seed=MASTER,
        extension=False, replacement=False, native_classification=False,
        preparation_energy_evaluations=0, python_geometry_evaluations=0,
        physical_draws_during_python_preparation=0,
        limits_scope='Every raw/retained campaign and CPU limit applies separately to each chain; never a replacement budget.')


def select_held_out(table, previous=PREVIOUS_PAIRS, per_stratum=2):
    """Read only pair, parent, whole-component, and witness metadata."""
    require(type(per_stratum) is int and per_stratum > 0, 'Invalid stratum count')
    indexed = {}
    for row in table:
        pair, parent, whole = row['members'], row['parent_component_members'], row['whole_component']
        require(len(pair) == 2 and all(type(i) is int and i >= 0 for i in pair)
                and pair == sorted(set(pair)), 'Invalid unordered pair')
        require(parent == sorted(set(parent)) and all(type(i) is int and i >= 0 for i in parent)
                and set(pair) <= set(parent) and type(whole) is bool
                and whole == (pair == parent), 'Invalid archived parent')
        require(tuple(pair) not in indexed and row['witnesses'], 'Duplicate pair or missing witness')
        indexed[tuple(pair)] = row
    require(all(tuple(p) in indexed for p in previous), 'Previous pair missing from archive')
    excluded = {tuple(indexed[tuple(p)]['parent_component_members']) for p in previous}
    eligible = [indexed[p] for p in sorted(indexed)
                if tuple(indexed[p]['parent_component_members']) not in excluded]
    selected = []
    for whole in (True, False):
        parents, group = set(), []
        for row in eligible:
            parent = tuple(row['parent_component_members'])
            if row['whole_component'] == whole and parent not in parents:
                group.append(copy.deepcopy(row))
                parents.add(parent)
                if len(group) == per_stratum:
                    break
        require(len(group) == per_stratum, 'Insufficient held-out parent components')
        selected.extend(group)
    return selected, dict(excluded_parents=[list(p) for p in sorted(excluded)],
        eligible_counts={str(w).lower(): sum(r['whole_component'] == w for r in eligible)
                         for w in (True, False)})


def make_contexts(state, selected):
    """Only squared center distance chooses an anchor; no atomic predicates."""
    require(len(state) >= 3, 'Insufficient source bodies')
    for pose in state:
        require(len(pose['position']) == 3 and len(pose['orientation']) == 4
                and all(type(v) in (int, float) and math.isfinite(v)
                        for v in pose['position']+pose['orientation'])
                and sum(v*v for v in pose['orientation']) > 0, 'Invalid source pose')
    result = []
    for row in selected:
        root, child = row['members']
        require(child < len(state), 'Selected body outside source')
        # Explicit accumulation preserves archived selector arithmetic.
        def order(i):
            distance = 0.
            for a, b in zip(state[root]['position'], state[i]['position']):
                distance += (a-b)**2
            return distance, i
        anchor = min((i for i in range(len(state)) if i not in (root, child)), key=order)
        result.append(dict(name=f"{'whole' if row['whole_component'] else 'embedded'}_{root}_{child}",
            root=root, child=child, anchor=anchor,
            parent_component_members=row['parent_component_members'],
            whole_component=row['whole_component'], witness=row['witnesses'][0]))
    return result


def verify_covariance_transform(original, scaled):
    require(original.get('schema') == 'reciprocal-pose-mixture-v1'
            and original.get('base_model', {}).get('schema') == 'weighted-pose-mixture-v1',
            'Unexpected FFT model schema')
    expected = copy.deepcopy(original)
    covariances = expected['base_model']['covariances']
    require(covariances and all(len(c) == 6 and all(len(row) == 6 for row in c)
                               for c in covariances), 'Expected 6x6 covariances')
    require(all(type(v) in (int, float) and math.isfinite(v)
                for c in covariances for row in c for v in row), 'Nonfinite covariance')
    expected['base_model']['covariances'] = [[[v*.0625 for v in row] for row in c]
                                           for c in covariances]
    require(scaled == expected, 'Scaled atlas changed outside exact tau=.25 covariance transform')


def jobs():
    entries = []
    for context in range(4):
        for initialization in INITIALIZATIONS:
            for stream in range(4):
                family = dict(context_index=context, initialization=initialization, stream=stream)
                for arm in ARMS:
                    key = seed(MASTER, context, initialization, stream, -1, 'queue/'+arm)
                    entries.append(dict(context_index=context, initialization=initialization,
                        stream=stream, arm=arm, seed_family=family.copy(), queue_key=key))
    entries.sort(key=lambda j: (j['queue_key'], j['context_index'], j['initialization'], j['stream'], j['arm']))
    return [dict(id=i, **j) for i, j in enumerate(entries)]


def configuration(base, inputs, contexts, compiled_source):
    """The complete runner-facing schema; also used by synthetic interface tests."""
    base = Path(base).resolve()
    return dict(schema='evolving-dimer-benchmark-v1',
        scientific_allocation=record(base/'common/scientific-allocation.json'),
        protocol=record(base/'protocol.json'), master_seed=MASTER,
        **inputs, contexts=contexts, jobs=jobs(), allocation=allocation(),
        local=dict(translation_std_A=.2, rotation_std_degrees=1., member_order=[0, 1, 0, 1],
                   pair_contact_required=False),
        factorized=dict(root_cap=32, internal_cap=32, joint_cap=1, order='root_first',
            uniform_probability=.5, uniform_half_width=160., tau=.25,
            pair_contact_required=True, outside_pair_contact='self_loop'),
        preparation=dict(kind='first_geometrically_feasible_unguided', outer_cap=256,
            minimum_max_center_displacement_A=5., minimum_max_body_orientation_degrees=10.,
            criterion='OR', energy_filter=False, native_filter=False,
            orientation='Absolute body quaternion geodesic angle from its own source pose: 2*acos(min(1,abs(dot(normalized_old,normalized_new)))).',
            failure='Fatal after the fixed cap; preserve every null/failure; no fallback, refill, or redraw.'),
        cloud=dict(raw_count=16384, banks=32, scope='fixed_per_chain_family',
            generate_all_before_preparation=True, shared_across_arms=True,
            refresh_threshold_each_dimer_attempt=True),
        physical=dict(depletant_radius=1.4, activity=.0275, lambda_ratio=64.,
            wall_radius=593.742500239952),
        envelope=dict(max_cells=255, max_depth=8, min_width=0.),
        limits=dict(raw_per_leg=20_000_000, raw_per_outer=40_000_000,
            raw_campaign=20_000_000_000, retained_per_leg=20_000_000,
            retained_per_outer=40_000_000, retained_campaign=20_000_000_000,
            cpu_seconds=1800.),
        compiled_source_sha256=compiled_source,
        preparation_output=str(base/'prepared'), output=str(base/'execution'))


def protocol(inputs, contexts, selection):
    return dict(schema='evolving-dimer-protocol-v1', allocation=allocation(),
        inputs=inputs, contexts=contexts, selection=selection,
        panel_rule='Exclude every archived parent component of 13/215,18/234,0/1,8/255. In whole then embedded strata take first two lexicographic pairs with distinct parents. Root is smaller label. Anchor is nearest root nonmember by squared center distance, label tie-break, then fixed for the entire trajectory.',
        conditional_target='Only root and child move. Every other labeled body, including anchor, stays exactly at its source pose. All three arms share the same unrestricted hard-valid two-body physical target; local moves may break the pair contact. Global factorized attempts are self-loops outside that contact subset.',
        initializations='Source is identical across streams; proposal_prepared uses 16 independent preparation RNG streams from those same four sources. First bounded feasible unguided candidate satisfying the frozen displacement OR body-angle criterion; shared across three arms. Geometry-conditioned starts are not equilibrium draws or independent physical preparations.',
        cloud_law='32 fixed-size root-body AABB clouds, 16384 raw points each, thinned by root exclusion membership. Generate every bank before any alternative-start search, with separate RNG roles. Each bank is shared across the three arms of one context/initialization/stream family; only m4 uses it. Refresh exact integer m4 threshold for each attempt and include the auxiliary count correction once. Fixed P preserves the conditional target but differs from fresh-cloud pilot dynamics.',
        seed_rule='StdRng::from_seed with the complete 32-byte SHA256 digest of evolving-dimer-v1/{master}/{context}/{initialization}/{stream}/{block}/{role}. Roles are separate for cloud, preparation, each local attempt proposal/bath/accept, and global proposal/threshold/bath/accept. Global roles include arm; local roles do not. Each block/attempt starts its own role stream so geometry-dependent consumption cannot leak into later updates. Queue ordering alone uses the first16hex u64 of the same digest with block=-1 and role=queue/{arm}.',
        observation='Retain every local and global attempt, including nulls/rejections/fatals. Record the actual retained endpoint after every block, during warmup and production, plus the initial endpoint. Never concatenate independent chains or manufacture intermediate states from sparse frames.',
        timing='Full sampler CPU per chain includes local/global proposals, density scoring, thresholds, physical gates, and rejected/null work. Preparation/cloud setup and passive observation CPU are reported separately; setup-inclusive campaign cost is also reported.',
        stop_rule='Fixed allocation. Any fatal or limit exhaustion is retained and stops that chain; no replacement or extension. Checkpoint stop is allowed only for a preproduction validation, never an intentional production stopping rule.',
        observer='Frozen native-free atom patch map; canonical body/patch contact tokens and partner graph. Apparent finite-record ESS uses all production block endpoints and full sampler CPU. Constant fingerprints yield null ESS. No stationarity, unseen-mode, assembly-equilibrium, or physical-rate inference.',
        limitations=['One inherited snapshot; held out by pair and parent component, not independent preparation or equilibrium sampling.',
            'Selection uses archived event-1 availability and geometry metadata, never pilot outcomes or native labels; historical source generation was not native-blind.',
            'Whole/embedded are initial strata only. Evolving two mobile bodies against 262 fixed spectators measures conditional relaxation, not finite-system assembly.',
            'Physical conditions 1.4 A/.0275 A^-3/500 uM remain distinct from the original 1.5 A/.035 A^-3/106.8 uM objective.'])


def source_dependencies(paths):
    result={};pending=list(paths)
    while pending:
        path=Path(pending.pop()).resolve()
        if path in result:continue
        result[path]=True
        for node in ast.walk(ast.parse(path.read_text(),filename=str(path))):
            names=([v.name for v in node.names] if isinstance(node,ast.Import) else
                   [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                local=path.parent/(name.split('.')[0]+'.py')
                if local.is_file():pending.append(local)
    return sorted(result)


def prepare(base):
    base = Path(base).resolve()
    require(not base.exists(), 'Preparation destination already exists')
    reference_path = ROOT/'results/dimer-destination-probe-20261002/config.json'
    require(sha(reference_path) == REFERENCE_SHA, 'Reference configuration changed')
    reference = read(reference_path)
    originals = dict(reference_config=record(reference_path), panel=reference['panel'],
        shape=reference['shape'], source_config=reference['source_config'],
        source_frame=reference['source_frame'], source_freeze_manifest=reference['source_freeze_manifest'],
        original_atlas=reference['atlases'][1]['model'],
        atlas=record(ROOT/'results/fft-width-probe-20261003/common/blind_fft512slots_tau0p25.json'),
        patch_map=record(ROOT/'runs/finite-assembly-observer-periodic-20260924/patch-map.json'))
    expected = dict(panel=PANEL_SHA, shape=SHAPE_SHA, source_config=SOURCE_CONFIG_SHA,
        source_frame=FRAME_SHA, source_freeze_manifest=SOURCE_FREEZE_SHA,
        original_atlas=FFT_SHA, atlas=SCALED_FFT_SHA, patch_map=PATCH_SHA)
    for name, value in originals.items():
        checked_file(value)
        if name in expected:
            require(value['sha256'] == expected[name], 'Unexpected source identity: '+name)
    panel, source, frame = (read(originals[k]['path']) for k in ['panel', 'source_config', 'source_frame'])
    require(frame['sweep'] == 7400 and source['initial_poses'] == frame['poses']
            and len(frame['poses']) == 264, 'Source state mismatch')
    require(reference['depletant_radius'] == source['depletant_radius'] == 1.4
            and reference['activity'] == source['reservoir_density'] == .0275
            and source['boundary'] == dict(kind='spherical', radius=reference['wall_radius'])
            and reference['wall_radius'] == 593.742500239952, 'Physical source identity differs')
    verify_covariance_transform(read(originals['original_atlas']['path']), read(originals['atlas']['path']))
    patch = read(originals['patch_map']['path'])
    require(patch['shape_sha256'] == SHAPE_SHA and len(patch['atom_patch_ids']) == 4004,
            'Patch map physical identity differs')
    selected, selection = select_held_out(panel['complete_eligible_pair_table'])
    require([r['members'] for r in selected] == EXPECTED_PAIRS, 'Held-out panel changed')
    contexts = make_contexts(frame['poses'], selected)
    source_files = [ROOT/p for p in ['Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md']]
    source_files += sorted((ROOT/'src').rglob('*.rs'))
    compiled = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    closure = source_files+[ROOT/EXAMPLE]+source_dependencies([ROOT/p for p in TOOLS])
    closure += sorted(p for p in (ROOT/'vendor').rglob('*') if p.is_file())
    require(all(p.is_file() for p in closure), 'Runner/preparation source closure incomplete')
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(closure)}
    # The scientific allocation is the first artifact; no scientific execution
    # occurs in this tool, and all source/input hashes precede any runner output.
    (base/'common/inputs').mkdir(parents=True)
    write(base/'common/scientific-allocation.json', allocation())
    from analyze_evolving_dimer_benchmark import analysis_plan
    write(base/'analysis-plan.json', analysis_plan())
    inputs = {}
    for name, value in originals.items():
        path = base/'common/inputs'/(name+'.json')
        shutil.copyfile(checked_file(value), path)
        require(sha(path) == value['sha256'], 'Input changed during copy')
        inputs[name] = record(path)
    for relative, digest in source_hashes.items():
        target = base/'common/source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, target)
        require(sha(target) == digest, 'Source changed during copy')
    details = protocol(inputs, contexts, selection)
    details['analysis_plan'] = record(base/'analysis-plan.json')
    details['source_files'] = source_hashes
    details['original_input_bindings'] = originals
    write(base/'protocol.json', details)
    config = configuration(base, inputs, contexts, compiled)
    write(base/'config.json', config)
    frozen = {str(p.relative_to(base)): sha(p) for p in sorted(base.rglob('*')) if p.is_file()}
    write(base/'freeze.json', dict(schema='evolving-dimer-preparation-freeze-v1', complete=True,
        scientific_execution_started=False, files=frozen, original_input_bindings=originals,
        python_version=sys.version, python_executable=sys.executable))
    return dict(base=str(base), config=record(base/'config.json'), freeze=record(base/'freeze.json'),
        chains=96, status='frozen_without_launch_or_binary_binding')


def verify_freeze(base):
    base = Path(base).resolve()
    frozen = read(base/'freeze.json')
    require(frozen['schema'] == 'evolving-dimer-preparation-freeze-v1'
            and frozen['complete'] is True and frozen['scientific_execution_started'] is False,
            'Invalid preparation freeze')
    for name, digest in frozen['files'].items():
        path = (base/name).resolve()
        require(path.is_relative_to(base) and path.is_file() and sha(path) == digest,
                'Frozen file changed: '+name)
    for value in frozen['original_input_bindings'].values():
        checked_file(value)
    config = read(base/'config.json')
    require(config['schema'] == 'evolving-dimer-benchmark-v1' and config['allocation'] == allocation()
            and config['jobs'] == jobs(), 'Runner allocation changed')
    return config


def bind_executable(base, executable, source_bundle=None):
    base = Path(base).resolve()
    config = verify_freeze(base)
    require(not (base/'binding.json').exists(), 'Executable already bound')
    matches = []
    paths = [Path(source_bundle)] if source_bundle else sorted(
        (ROOT/'target-validation-line-guide/release/build').glob('tetramer-mc-*/out/source-bundle.json'))
    for path in paths:
        bundle = read(path)
        for item in bundle['files'].values():
            require(hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256'], 'Corrupt source bundle')
        if {k: v['sha256'] for k, v in bundle['files'].items()} == config['compiled_source_sha256']:
            matches.append(path)
    require(matches and len({sha(p) for p in matches}) == 1, 'No unique matching compiled source bundle')
    require(sha(ROOT/EXAMPLE) == sha(base/'common/source'/EXAMPLE), 'Runner source changed after freeze')
    target = base/'common/evolving_dimer_benchmark'
    require(not target.exists() and not (base/'common/source-bundle.json').exists(), 'Partial prior binary binding')
    executable = Path(executable).resolve()
    digest = sha(executable)
    shutil.copyfile(executable, target)
    require(sha(target) == digest, 'Executable changed during copy')
    target.chmod(0o755)
    shutil.copyfile(matches[0], base/'common/source-bundle.json')
    binding = dict(schema='evolving-dimer-binding-v1', config_sha256=sha(base/'config.json'),
        protocol_sha256=sha(base/'protocol.json'), freeze_sha256=sha(base/'freeze.json'),
        example_source_sha256=sha(base/'common/source'/EXAMPLE),
        compiled_source_bundle_sha256=sha(base/'common/source-bundle.json'), executable_sha256=digest)
    write(base/'binding.json', binding)
    return dict(binding=record(base/'binding.json'), executable=record(target),
        status='bound_without_launch')


def preparation_distance_passes(old, new, definition):
    require(len(old) == len(new) == 2, 'Invalid preparation pose count')
    displacement = 0.; angle = 0.
    for a,b in zip(old,new):
        require(len(a['position']) == len(b['position']) == 3 and len(a['orientation']) == len(b['orientation']) == 4,
                'Malformed preparation pose')
        require(all(type(v) in (int,float) and math.isfinite(v) for v in a['position']+b['position']+a['orientation']+b['orientation']), 'Nonfinite preparation pose')
        displacement=max(displacement, math.sqrt(sum((x-y)**2 for x,y in zip(a['position'],b['position']))))
        na=math.sqrt(sum(x*x for x in a['orientation'])); nb=math.sqrt(sum(x*x for x in b['orientation']))
        require(na>0 and nb>0, 'Zero quaternion in preparation')
        dot=sum(x*y for x,y in zip(a['orientation'],b['orientation']))/(na*nb)
        angle=max(angle,math.degrees(2*math.acos(min(1.,abs(dot)))))
    return displacement >= definition['minimum_max_center_displacement_A'] or angle >= definition['minimum_max_body_orientation_degrees']


def validate_prepared_manifest(base, manifest_path):
    base, path = Path(base).resolve(), Path(manifest_path).resolve()
    require(path.is_relative_to(base/'prepared'), 'Manifest outside preparation directory')
    manifest = read(path)
    require(manifest['schema'] == 'evolving-dimer-prepared-starts-v1'
            and manifest['complete'] is True and manifest['passed'] is True
            and manifest['all_attempts_retained'] is True, 'Prepared-start manifest incomplete or failed')
    require(manifest['config_sha256'] == sha(base/'config.json')
            and manifest['binding_sha256'] == sha(base/'binding.json'), 'Prepared-start provenance differs')
    files = {}
    for value in manifest['files']:
        bound_path = checked_file(value)
        require(bound_path.is_relative_to(base/'prepared') and bound_path != path
                and str(bound_path) not in files, 'Invalid or duplicate prepared file')
        files[str(bound_path)] = value['sha256']
    require(files, 'No prepared files bound')
    config = read(base/'config.json')
    source = read(checked_file(config['source_frame']))['poses']
    clouds = manifest['cloud_banks']
    require(all(type(c['context_index']) is int and type(c['stream']) is int
                and isinstance(c['raw'], dict) and isinstance(c['metadata'], dict) for c in clouds),
            'Cloud family or file references malformed')
    keys = [(c['context_index'], c['initialization'], c['stream']) for c in clouds]
    expected = {(c, initialization, s) for c in range(4) for initialization in INITIALIZATIONS for s in range(4)}
    require(len(keys) == 32 and set(keys) == expected, 'Cloud family allocation differs')
    starts = manifest['alternative_starts']
    require(all(type(s['context_index']) is int and type(s['stream']) is int
                and isinstance(s['record'], dict) and isinstance(s['ledger'], dict) for s in starts),
            'Alternative-start family or file references malformed')
    keys = [(s['context_index'], s['stream']) for s in starts]
    require(len(keys) == 16 and set(keys) == {(c, s) for c in range(4) for s in range(4)},
            'Alternative-start allocation differs')
    require(all(s['status'] == 'prepared' and type(s['attempts']) is int
                and 1 <= s['attempts'] <= 256 for s in starts), 'Failed, missing, or extended alternative start')
    # Every nested BoundFile must belong to the complete authenticated inventory.
    def visit(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                p = checked_file(value)
                require(files.get(str(p)) == value['sha256'], 'Prepared reference omitted from inventory')
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(clouds)
    visit(starts)
    actual = {str(p.resolve()) for p in (base/'prepared').rglob('*') if p.is_file() and p.resolve() != path}
    require(actual == set(files), 'Prepared directory has omitted or extra inventory files')
    raw_count = config['cloud']['raw_count']
    for bank in clouds:
        raw_path, metadata_path = checked_file(bank['raw']), checked_file(bank['metadata'])
        metadata = read(metadata_path)
        require(metadata['raw_count'] == raw_count and raw_path.stat().st_size == raw_count*24,
                'Cloud raw allocation differs')
        require(len(metadata['low']) == len(metadata['high']) == 3 and all(
            type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)
            and a < b for a,b in zip(metadata['low'],metadata['high'])), 'Invalid cloud bounds')
        indices = metadata['kept_indices']
        require(indices == sorted(set(indices)) and all(type(i) is int and 0 <= i < raw_count for i in indices),
                'Invalid retained cloud indices')
        require(math.isfinite(metadata['cpu_seconds']) and metadata['cpu_seconds'] >= 0, 'Invalid cloud CPU')
        require(all(math.isfinite(u) and 0 <= u < 1 for (u,) in struct.iter_unpack('<d', raw_path.read_bytes())),
                'Invalid saved raw cloud variate')
    for start in starts:
        saved = read(checked_file(start['record']))
        case = config['contexts'][start['context_index']]
        require(saved['context_index'] == start['context_index'] and saved['stream'] == start['stream']
                and saved['status'] == 'prepared' and saved['attempts'] == start['attempts']
                and saved['source'] == [source[case['root']], source[case['child']]],
                'Prepared start/source correspondence differs')
        with checked_file(start['ledger']).open() as handle:
            rows = [json.loads(line) for line in handle]
        require(len(rows) == start['attempts'] and all(type(r['attempt']) is int
                and r['attempt'] == i for i, r in enumerate(rows, 1)), 'Preparation attempt ledger has gaps')
        require(all('failure' not in r and r.get('selected') is (i == len(rows))
                    for i, r in enumerate(rows, 1)), 'Preparation ledger did not retain first selected success')
        for row in rows:
            candidate = row['outcome']['candidate']
            selected = candidate is not None and preparation_distance_passes(saved['source'],
                [candidate['root'], candidate['child']], config['preparation'])
            require(row['selected'] is selected, 'Preparation first-feasible displacement criterion differs')
        candidate = rows[-1]['outcome']['candidate']
        require(candidate is not None and saved['selected'] == [candidate['root'], candidate['child']],
                'Saved preparation differs from selected ledger candidate')
    return manifest, files


def bind_prepared(base, manifest_path=None):
    base = Path(base).resolve()
    verify_freeze(base)
    require(not (base/'run-binding.json').exists(), 'Prepared starts already bound')
    binding = read(base/'binding.json')
    require(binding['schema'] == 'evolving-dimer-binding-v1', 'Unknown prelaunch binding')
    for key, path in [('config_sha256', base/'config.json'), ('protocol_sha256', base/'protocol.json'),
                      ('freeze_sha256', base/'freeze.json'),
                      ('example_source_sha256', base/'common/source'/EXAMPLE),
                      ('compiled_source_bundle_sha256', base/'common/source-bundle.json'),
                      ('executable_sha256', base/'common/evolving_dimer_benchmark')]:
        require(binding[key] == sha(path), 'Prelaunch binding changed: '+key)
    path = Path(manifest_path).resolve() if manifest_path else base/'prepared/manifest.json'
    manifest, files = validate_prepared_manifest(base, path)
    result = dict(schema='evolving-dimer-run-binding-v1', complete=True,
        config_sha256=sha(base/'config.json'), protocol_sha256=sha(base/'protocol.json'),
        prelaunch_binding=record(base/'binding.json'), prepared_manifest=record(path),
        prepared_files=files, cloud_banks=len(manifest['cloud_banks']),
        alternative_starts=len(manifest['alternative_starts']))
    write(base/'run-binding.json', result)
    return dict(run_binding=record(base/'run-binding.json'), status='prepared_inputs_bound_without_launch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--bind', action='store_true')
    mode.add_argument('--bind-prepared', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--source-bundle', type=Path)
    parser.add_argument('--prepared-manifest', type=Path)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.output)
    elif args.bind:
        parser.error('--bind requires --executable') if args.executable is None else None
        result = bind_executable(args.output, args.executable, args.source_bundle)
    else:
        result = bind_prepared(args.output, args.prepared_manifest)
    print(json.dumps(result, indent=2, allow_nan=False))
