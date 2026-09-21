#!/usr/bin/env python3
"""Prepare, but never launch, the fixed AB conditional-shoulder docking pilot.

The input atlas is frozen. This is a benchmark of the target restricted to
1 < original q < 2, not native escape, assembly, or physical kinetics.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
from pathlib import Path
import shlex
import shutil

from prepare_smc_normalizer_atlas import ROOT, Density, read, relative_poses, sha, write
from prepare_deep_far_normalizer_atlas import registration

SOURCE = ROOT / 'runs/ab-shoulder-contact-atlas-preparation-20260921'
REFERENCE = ROOT / 'runs/ab-shoulder-contact-atlas-assessment-20260921/analysis.json'
MODEL_SHA = 'daf2d3adb096453b6e91f1dfe8f614a614fb053f139e2bde386b36d9b8b2c901'
LABELS = ['direct', 'mixture', 'geometry', 'inner_remainder', 'outer_shoulder']
METRIC_KEYS = ('native_poses', 'rigid_members', 'member_error_scale', 'angle_error_scale_deg')
WINDOW = dict(minimum=1., maximum=2., lower_inclusive=False, upper_inclusive=False)
DEFAULT_CYCLES = 5000
DEFAULT_BURN_CYCLES = 1000
DEFAULT_SEED_BASE = 115101010
SEED_STRIDE = 1009
JOB_COUNT = 12
ATTEMPTS_PER_CYCLE = 3


def validate_design(cycles, burn_cycles, seed_base, workers):
    """Validate the fixed allocation before reading inputs or writing output."""
    if type(cycles) is not int or not 1 <= cycles <= (2**64-1)//ATTEMPTS_PER_CYCLE:
        raise ValueError('cycles must be a positive integer with the attempt count fitting in u64')
    if type(burn_cycles) is not int or not 0 <= burn_cycles < cycles:
        raise ValueError('burn_cycles must be an integer with 0 <= burn_cycles < cycles')
    if cycles-burn_cycles < 2:
        raise ValueError('Require at least two retained cycle endpoints for half diagnostics')
    if type(seed_base) is not int or not 0 <= seed_base <= 2**64-1-SEED_STRIDE*(JOB_COUNT-1):
        raise ValueError('seed_base and every offset seed must fit in u64')
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('Require one to 32 workers')


def validate_plan(manifest, campaign):
    """Require the exact frozen allocation and executable CLI in every job."""
    jobs, seeds = manifest['jobs'], manifest['seeds']
    assert len(jobs) == len(seeds) == JOB_COUNT
    # Historical pilot manifests predate the explicit seed_base field.
    seed_base = manifest.get('seed_base', seeds[0])
    validate_design(manifest['cycles'], manifest['burn_cycles'], seed_base, manifest['workers'])
    assert manifest['sample_every'] == 1 and manifest['attempts_per_cycle'] == ATTEMPTS_PER_CYCLE
    assert type(manifest['total_attempts']) is int
    assert manifest['total_attempts'] == len(jobs)*manifest['cycles']*ATTEMPTS_PER_CYCLE
    assert seeds == [seed_base+SEED_STRIDE*i for i in range(JOB_COUNT)]
    assert all(type(seed) is int for seed in seeds)
    assert len({job['id'] for job in jobs}) == JOB_COUNT
    allocation = [(start, replicate, mode, method, correlation)
        for start in ('direct', 'geometry') for replicate in range(2)
        for mode, method, correlation in (('local', 'local', 0.),
            ('c0', 'posterior-involution', 0.), ('c09', 'posterior-involution', .9))]
    for index, (job, design) in enumerate(zip(jobs, allocation)):
        assert (job['start'], job['replicate'], job['mode'], job['method'], job['correlation']) == design
        assert job['index'] == index and job['seed'] == seeds[index]
        expected = [str(Path(campaign)/'provenance/docking-mc'), '--config', job['config'],
            '--model', str(Path(campaign)/'provenance/model.json'), '--out', job['directory'],
            '--cycles', str(manifest['cycles']), '--sample-every', '1', '--method', job['method'],
            '--correlation', str(job['correlation'])]
        assert job['command'] == expected, 'Job command differs from frozen allocation'


def local_dependencies(paths):
    """Resolve the local Python import closure, including nested imports."""
    result, pending = {}, list(paths)
    while pending:
        path = Path(pending.pop()).resolve()
        if path.name in result:
            assert result[path.name] == path, 'Ambiguous flat-archive import'
            continue
        result[path.name] = path
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = ([item.name for item in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                dependency = path.parent/(name.split('.')[0]+'.py')
                if dependency.is_file():
                    pending.append(dependency)
    return result


def physical_reference(path, source):
    """Freeze observed five-region probabilities and independent-population errors."""
    assessment = read(path)
    assert assessment['complete'] and assessment['original_q_window'] == WINDOW
    assert assessment['inner_q_window'] == dict(WINDOW, maximum=1.1)
    assert assessment['center_order'] == LABELS[:3]
    assert assessment['source_sha256'][str(source/'config.json')] == sha(source/'config.json')
    assert assessment['source_sha256'][str(source/'model-broad.json')] == MODEL_SHA
    for name in LABELS[:3]:
        assert assessment['source_sha256'][str(source/f'geometric-{name}.json')] == sha(source/f'geometric-{name}.json')
    broad = next(campaign for campaign in assessment['campaigns'] if campaign['arm'] == 'broad')['physical']
    names = [f'inner_{name}_assigned_ball0p5' for name in LABELS[:3]]+['inner_outside_union', 'outer']
    base = broad['full']['logQ']
    denominator = [math.exp(value-base) if value is not None else 0.
                   for value in broad['full']['population_logQ_values']]
    n = len(denominator)
    rows = {}
    for label, name in zip(LABELS, names):
        region = broad[name]
        probability = region['fraction_of_observed_full_shoulder']
        numerator = [math.exp(value-base) if value is not None else 0.
                     for value in region['population_logQ_values']]
        residual = [x-probability*y for x, y in zip(numerator, denominator)]
        mean_residual = sum(residual)/n
        variance = sum((value-mean_residual)**2 for value in residual)/(n-1)
        error = math.sqrt(variance/n)/(sum(denominator)/n)
        rows[label] = dict(probability=probability, observed_delta_method_SE=error, source_region=name)
    assert abs(sum(row['probability'] for row in rows.values())-1.) < 1e-10
    return dict(source_path=str(path), source_sha256=sha(path), arm='broad', regions=rows,
        independent_populations=n, scope='Observed conditional probabilities from independent importance populations; correlated numerator/denominator errors retained by population delta method. Observed errors do not bound unseen mass.')


def prepare(out: Path, source: Path = SOURCE, binary: Path | None = None, workers: int = 4,
            reference: Path = REFERENCE, *, cycles: int = DEFAULT_CYCLES,
            burn_cycles: int = DEFAULT_BURN_CYCLES, seed_base: int = DEFAULT_SEED_BASE):
    """Freeze an inert plan into a new empty directory; no subprocesses run."""
    validate_design(cycles, burn_cycles, seed_base, workers)
    source, out = source.resolve(), out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be new or empty')
    protocol, cfg = read(source / 'protocol.json'), read(source / 'config.json')
    assert sha(source / 'config.json') == protocol['config_sha256']
    model_path = source / 'model-broad.json'
    if sha(model_path) != MODEL_SHA:
        raise ValueError('The predeclared broad five-component atlas has changed')
    model = read(model_path)
    assert len(model['weights']) == 5 and model['coordinate_convention'] == 'anchor-body-relative'
    assert len(cfg['fixed_poses']) == 2 and protocol['proposal_anchor_index'] == 0
    assert cfg['capture_radius'] == 18. and cfg['depletant_radius'] == 1.5
    assert cfg['reservoir_density'] == .035 and cfg['poisson_lambda_ratio'] == 64.
    assert cfg['translation_steps'] == [.2, 2.] and cfg['rotation_steps_deg'] == [1.5, 15.]
    assert cfg['rotation_probability'] == .5 and cfg['local_attempts_per_cycle'] == 2
    assert protocol['q_window'] == WINDOW
    metric = {key: copy.deepcopy(cfg['metadata'][key]) for key in METRIC_KEYS}
    assert metric['member_error_scale'] == 2. and metric['angle_error_scale_deg'] == 15.
    centers = {entry['name']: entry for entry in protocol['analysis']['centers']}
    for entry in centers.values():
        assert sha(entry['model_path']) == entry['model_sha256']
    shape = Path(cfg['shape'])
    assert sha(shape) == model['shape_sha256'] == protocol['shape_sha256']
    if binary is not None and not binary.resolve().is_file():
        raise ValueError('An explicitly supplied binary must already exist')
    reference = reference.resolve()
    reference_summary = physical_reference(reference, source)

    for name in ('provenance', 'configs', 'logs', 'runs'):
        (out / name).mkdir(parents=True, exist_ok=True)
    archive = out / 'provenance'
    inputs = {'model.json': model_path, 'input-config.json': source / 'config.json',
              'input-protocol.json': source / 'protocol.json', 'shape.json': shape,
              'physical-reference-assessment.json': reference,
              'prepare_shoulder_docking_benchmark.py': Path(__file__).resolve()}
    for name in ('direct', 'mixture', 'geometry'):
        inputs[f'geometric-{name}.json'] = Path(centers[name]['model_path'])
    for name in ('analyze_shoulder_docking_benchmark.py', 'prepare_smc_normalizer_atlas.py',
                 'prepare_deep_far_normalizer_atlas.py', 'analyze_posterior_docking_pilot.py',
                 'analyze_involution_docking_campaign.py', 'analyze_free_tetramer_campaign.py',
                 'run_shoulder_docking_benchmark.py', 'test_shoulder_docking_benchmark.py'):
        inputs[name] = ROOT / 'tools' / name
    inputs.update(local_dependencies([path for path in inputs.values() if path.suffix == '.py']))
    for name in ('Cargo.toml', 'Cargo.lock', 'build.rs'):
        inputs[name] = ROOT / name
    inputs['source/vendor/README.md'] = ROOT/'vendor/README.md'
    # The source archive records dirty working-tree code without building or
    # touching target/release or any running experiment's executable.
    for path in sorted((ROOT / 'src').rglob('*.rs')):
        inputs[str(Path('source') / path.relative_to(ROOT))] = path
    inputs['source/tests/docking_region.rs'] = ROOT/'tests/docking_region.rs'
    if binary is not None:
        inputs['docking-mc'] = binary.resolve()
    for name, path in inputs.items():
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    density, jobs, starts = Density(model), [], []
    for start in ('direct', 'geometry'):
        pose = centers[start]['pose']
        q = float(registration([pose], metric)[0])
        assert 1. < q < 2.
        logg, _, _ = density.evaluate(relative_poses([pose], cfg['fixed_poses'][0]))
        starts.append(dict(name=start, pose=pose, original_q=q, gaussian_log_density=float(logg[0]),
                           source='input-protocol.json:analysis.centers', initialization_equilibrated=False))
        for replicate in range(2):
            for mode, method, correlation in (('local', 'local', 0.), ('c0', 'posterior-involution', 0.),
                                               ('c09', 'posterior-involution', .9)):
                index, seed = len(jobs), seed_base + SEED_STRIDE * len(jobs)
                identifier = f'AB-shoulder-{start}-r{replicate:02d}-{mode}'
                config = copy.deepcopy(cfg)
                config.update(shape=str(archive / 'shape.json'), initial_pose=copy.deepcopy(pose),
                              seed=seed, uniform_probability=.05, proposal_anchor_index=0,
                              target_region=dict(metric=metric, window=WINDOW))
                config['metadata'].update(mode=mode, start_basin=start, replicate=replicate,
                    initialization_equilibrated=False, target_scope='Strict original 1 < q < 2',
                    initial_pose_source=dict(file='input-protocol.json', center=start),
                    conditional_atlas=True, atlas_frozen=True, native_informed_atlas=True,
                    source_config_sha256=sha(source / 'config.json'))
                config_path = out / 'configs' / f'{identifier}.json'
                write(config_path, config)
                directory = out / 'runs' / identifier
                command = [str(archive / 'docking-mc'), '--config', str(config_path), '--model',
                           str(archive / 'model.json'), '--out', str(directory), '--cycles', str(cycles),
                           '--sample-every', '1', '--method', method, '--correlation', str(correlation)]
                jobs.append(dict(id=identifier, index=index, start=start, replicate=replicate, mode=mode,
                    method=method, correlation=correlation, seed=seed, config=str(config_path),
                    config_sha256=sha(config_path), directory=str(directory), command=command,
                    log=str(out / 'logs' / f'{identifier}.log')))

    hashes = {name: sha(archive / name) for name in inputs}
    source_bundle_files = {name: name for name in ('Cargo.toml', 'Cargo.lock', 'build.rs')}
    source_bundle_files['vendor/README.md'] = 'source/vendor/README.md'
    source_bundle_files.update({name.removeprefix('source/'): name for name in inputs
                                if name.startswith('source/src/') and name.endswith('.rs')})
    manifest = dict(schema='conditional-shoulder-docking-v1', preparation_only=True, launched=False,
        binary_supplied=binary is not None, binary_sha256=hashes.get('docking-mc'),
        model_sha256=MODEL_SHA, shape_sha256=hashes['shape.json'], input_sha256=hashes,
        source_paths={name: str(path) for name, path in inputs.items()}, jobs=jobs, starts=starts,
        source_bundle_files=source_bundle_files,
        cycles=cycles, burn_cycles=burn_cycles, seed_base=seed_base,
        sample_every=1, workers=workers, attempts_per_cycle=ATTEMPTS_PER_CYCLE,
        physical_reference=reference_summary,
        total_attempts=len(jobs)*cycles*ATTEMPTS_PER_CYCLE,
        seeds=[job['seed'] for job in jobs], target_region=dict(metric=metric, window=WINDOW),
        physical_labels=LABELS, physical_mask_rule='Within original q<1.1 assign direct, mixture, geometry in that priority using frozen geometric radius<=0.5; then inner remainder; q>=1.1 is outer shoulder.',
        proposal='Two common local slots and a third local, posterior c=0, or posterior c=0.9 slot. Frozen five Gaussian weights; separate uniform branch probability0.05; fixed proposal anchor A; both A and B interact physically.',
        atlas_distinction='The source integration hybrid/geometric-cover allocation0.99 is not an MCMC proposal weight. No importance-sampling denominator is transported.',
        inference_scope='Conditional shoulder sampling diagnostic only; starts are not equilibrated. Retain all repeats. Apparent ESS is descriptive until initialization and occupancy agreement are established. No native escape, assembly, or kinetic-rate inference.')
    validate_plan(manifest, out)
    write(out / 'manifest.json', manifest)
    write(out / 'commands.json', [dict(id=j['id'], argv=j['command'], stdout=j['log']) for j in jobs])
    # This is a reviewable plan, intentionally not an automatic supervisor.
    plan = ['#!/usr/bin/env bash', 'set -euo pipefail', '# INERT PREPARATION: nothing has been launched.',
            '# Run only after reviewing the source and explicitly freezing a compatible binary.',
            f'# Sequential commands below satisfy the maximum concurrency of {workers}.']
    if binary is None:
        plan += ["echo 'No binary was supplied; create a fresh preparation with --binary after review.' >&2", 'exit 2']
    else:
        plan += [f"test -x {shlex.quote(str(archive / 'docking-mc'))}",
                 f"printf '%s  %s\\n' {shlex.quote(hashes['docking-mc'])} {shlex.quote(str(archive / 'docking-mc'))} | sha256sum --check --status"]
    plan += [shlex.join(job['command']) + ' > ' + shlex.quote(job['log']) + ' 2>&1' for job in jobs]
    terminal_status = dict(running=False, complete=True,
        jobs=[dict(id=job['id'], exit_code=0, status='complete') for job in jobs])
    plan += ['# Written only after every command above returns zero (set -e).',
             "cat > " + shlex.quote(str(out/'status.json.tmp')) + " <<'SHOULDER_TERMINAL_STATUS_JSON'",
             json.dumps(terminal_status, indent=2), 'SHOULDER_TERMINAL_STATUS_JSON',
             'mv -- '+shlex.quote(str(out/'status.json.tmp'))+' '+shlex.quote(str(out/'status.json'))]
    (out / 'commands.sh').write_text('\n'.join(plan) + '\n')
    (out / 'README.md').write_text(
        '# Conditional shoulder pilot — preparation only\n\n'
        f'No simulation was launched. Twelve fixed runs, {cycles:,} cycles each, three attempts per cycle; '
        f'discard the first {burn_cycles:,} cycle endpoints and retain cycles {burn_cycles+1}–{cycles} including repeats. '
        'The target is strict original 1 < q < 2.\n\n'
        'Review `manifest.json`, `commands.json`, and `commands.sh`. The shell plan is sequential and does not build anything. '
        'No binary is inferred from target/release: pass an explicitly reviewed existing binary into a fresh preparation.\n\n'
        'The observer requires terminal `status.json` with `running:false`, `complete:true`, and exactly the 12 manifest job IDs, '
        'each with `exit_code:0` and `status:"complete"`. The sequential plan atomically writes this only after all commands succeed; '
        'an external controller may write the same schema. Missing, running, incomplete, failed, duplicate, or extra jobs block assessment.\n\n'
        'The observer requires completed outputs and writes to a separate empty output directory. '
        'Use `python tools/analyze_shoulder_docking_benchmark.py CAMPAIGN --out NEW_REPORT`. '
        'Contact descriptors retain A and B separately. Source chart labels are distinct from physical regions; '
        'finite observed physical-region occupancies do not by themselves establish equilibrium probabilities.\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--binary', type=Path, help='Explicit existing reviewed binary; never inferred or built')
    parser.add_argument('--reference', type=Path, default=REFERENCE, help='Frozen broad five-region assessment')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cycles', type=int, default=DEFAULT_CYCLES)
    parser.add_argument('--burn-cycles', type=int, default=DEFAULT_BURN_CYCLES)
    parser.add_argument('--seed-base', type=int, default=DEFAULT_SEED_BASE)
    args = parser.parse_args()
    manifest = prepare(args.out, args.source, args.binary, args.workers, args.reference,
                       cycles=args.cycles, burn_cycles=args.burn_cycles, seed_base=args.seed_base)
    print(json.dumps(dict(prepared=True, launched=False, jobs=len(manifest['jobs']),
                         binary_supplied=manifest['binary_supplied'], out=str(args.out.resolve()))))


if __name__ == '__main__':
    main()
