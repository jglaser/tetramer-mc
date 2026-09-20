#!/usr/bin/env python3
"""Matched fixed-neighbor docking with frozen-mixture and involution controls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_free_tetramer_campaign import ROOT, sha, write


MODES = {'mixture': None, 'c0': 0., 'c05': .5, 'c09': .9, 'c1': 1.}


def convert_conditional_atlas(original, environment, shape_hash):
    """Change a laboratory-frame Cayley atlas into the fixed-neighbor frame.

    Conjugating a proper rotation sends its Cayley vector c to R_f^T c.
    Consequently B=diag(R_f^T,R_f^T) transforms means and covariances;
    the rigid frame change preserves the joint translation/Haar measure.
    """
    import numpy as np
    from analyze_involution_docking_campaign import ChartAudit
    reference = ChartAudit(original)
    fixed = environment['fixed_poses'][0]
    t, r = reference.arrays(fixed)
    block = np.zeros((6, 6))
    block[:3, :3] = block[3:, 3:] = r.T
    converted = copy.deepcopy(original)
    converted['anchors'] = [dict(position=(r.T@(np.asarray(a['position'])-t)).tolist(),
        rotation=(r.T@np.asarray(a['rotation'])).tolist()) for a in original['anchors']]
    converted['means'] = (np.asarray(original['means'])@block.T).tolist()
    converted['covariances'] = (block@np.asarray(original['covariances'])@block.T).tolist()
    converted['shape_sha256'] = shape_hash
    converted['coordinate_convention'] = 'anchor-body-relative'
    converted['frame_conversion'] = dict(kind='proper_fixed_anchor_rigid_frame', fixed_pose=fixed,
        component_count_unchanged=len(original['weights']), refitted=False, pruned=False)
    # Ten fresh poses, two from each of the five unchanged Gaussian components.
    # Evaluate the complete densities, so this also checks covariance coupling
    # and the chart Haar Jacobian, rather than just matching chart centers.
    audit = ChartAudit(converted)
    rng = np.random.default_rng(739012)
    checks = []
    for label in range(len(original['weights'])):
        for repeat in range(2):
            pose, _ = reference.decode(label, rng.normal(size=6))
            relative = reference.relative(pose, fixed)
            old = reference.full_gaussian(pose)
            new = audit.full_gaussian(relative)
            assert abs(old-new) < 2e-6, (label, repeat, old, new)
            checks.append(dict(label=label, repeat=repeat, laboratory_pose=pose,
                relative_pose=relative, original_log_density=old, converted_log_density=new,
                absolute_log_density_error=abs(old-new)))
    return converted, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path,
                        default=Path('/home/xvg/protein-nucleation/results/coordination-mc/narrow-water'))
    parser.add_argument('--binary', type=Path, default=ROOT/'target/release/docking-mc')
    parser.add_argument('--model', type=Path, default=ROOT/'examples/frozen-relative-mixture.json')
    parser.add_argument('--conditional-atlas', type=Path,
        help='Existing laboratory-frame site0 atlas; convert all components without fitting')
    parser.add_argument('--covered-starts', type=Path,
        help='Directory containing heldout-r2-f0.5.json and heldout-r3-f0.5.json competing poses')
    parser.add_argument('--sites', nargs='+', default=['site0-m1', 'site1-m1'])
    parser.add_argument('--starts', nargs='+', choices=['native', 'other'], default=['native', 'other'])
    parser.add_argument('--modes', nargs='+', choices=list(MODES), default=list(MODES))
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--cycles', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--activity', type=float, default=.035)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if min(args.replicates, args.cycles, args.workers) < 1 or args.replicates > 8:
        parser.error('Positive run sizes required; at most eight supplied starts are available')
    conditional = args.conditional_atlas is not None
    if conditional and (args.sites != ['site0-m1'] or args.replicates != 2 or args.covered_starts is None):
        parser.error('Conditional coverage control requires --sites site0-m1 --replicates 2 and --covered-starts')
    if args.covered_starts is not None and not conditional:
        parser.error('--covered-starts requires --conditional-atlas')
    out, source = args.out.resolve(), args.source.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error('Output directory must be empty')
    for name in ('configs', 'logs', 'runs', 'provenance'):
        (out/name).mkdir()
    archive = out/'provenance'
    inputs = {'docking-mc': args.binary.resolve(), 'model.json': args.model.resolve(),
              'shape.json': source/'inputs/tetramer-shape.json',
              'source-manifest.json': source/'manifest.json',
              'source-analysis.json': source/'analysis.json',
              'launcher.py': Path(__file__).resolve(),
              'run_free_tetramer_campaign.py': ROOT/'tools/run_free_tetramer_campaign.py',
              'analyze_involution_docking_campaign.py': ROOT/'tools/analyze_involution_docking_campaign.py'}
    if conditional:
        inputs['original-conditional-model.json'] = args.conditional_atlas.resolve()
        for replicate in range(2):
            inputs[f'covered-other-{replicate}.json'] = args.covered_starts.resolve()/f'heldout-r{replicate+2}-f0.5.json'
    for site in args.sites:
        inputs[f'environment-{site}.json'] = source/'inputs'/f'environment-{site}.json'
        for start in args.starts:
            for replicate in range(args.replicates):
                name = f'{site}-r1.5-z{args.activity:g}-start-{start}-{replicate:02d}.json'
                inputs['source-'+name] = source/'configs'/name
    for name, path in inputs.items():
        shutil.copy2(path, archive/name)
    model = json.loads((archive/'model.json').read_text())
    conversion_checks = None
    if conditional:
        original = json.loads((archive/'original-conditional-model.json').read_text())
        env_path = archive/'environment-site0-m1.json'
        assert sha(archive/'shape.json') in original['input_sha256'].values()
        assert sha(env_path) in original['input_sha256'].values()
        assert len(original['weights']) == 5
        model, conversion_checks = convert_conditional_atlas(original,
            json.loads(env_path.read_text()), sha(archive/'shape.json'))
        write(archive/'model.json', model)
        write(archive/'frame-conversion-audit.json', conversion_checks)
    assert model['shape_sha256'] == sha(archive/'shape.json')
    assert len(model['anchors']) == (5 if conditional else 28)
    jobs, preparations = [], []
    for site in args.sites:
        env = json.loads((archive/f'environment-{site}.json').read_text())
        assert len(env['fixed_poses']) == 1
        for start in args.starts:
            for replicate in range(args.replicates):
                name = f'{site}-r1.5-z{args.activity:g}-start-{start}-{replicate:02d}.json'
                old = json.loads((archive/('source-'+name)).read_text())
                pose = old['trajectory']['initial_pose']
                pose_source = dict(kind='selected_smc_endpoint', path=str(archive/('source-'+name)))
                if conditional and start == 'other':
                    from scipy.spatial.transform import Rotation
                    covered_path = archive/f'covered-other-{replicate}.json'
                    covered = json.loads(covered_path.read_text())
                    pose = dict(position=covered['position'],
                        orientation=Rotation.from_matrix(covered['rotation']).as_quat()[[3, 0, 1, 2]].tolist())
                    pose_source = dict(kind='previously_inspected_conditional_basin_snapshot',
                        path=str(covered_path), sha256=sha(covered_path), source=covered['source'])
                preparations.append(dict(site=site, start=start, replicate=replicate,
                    seed=old['seed'], initial_pose=pose, source_config_sha256=sha(archive/('source-'+name)),
                    source_smc=old['metadata']['source_smc'], initial_pose_source=pose_source))
                for mode in args.modes:
                    identifier = f'{site}-{start}-r{replicate:02d}-{mode}'
                    cfg = dict(shape=str(archive/'shape.json'), fixed_poses=env['fixed_poses'],
                        initial_pose=copy.deepcopy(pose), capture_center=env['capture_center'],
                        capture_radius=env['capture_radius'], depletant_radius=old['depletant_radius'],
                        reservoir_density=old['reservoir_density'], poisson_lambda_ratio=16.,
                        translation_steps=[.2, 2.], rotation_steps_deg=[1.5, 15.],
                        rotation_probability=.5, local_attempts_per_cycle=2, uniform_probability=.1,
                        seed=old['seed'], endpoint_gate=dict(max_cells=2047, max_depth=14, min_width=.5),
                        metadata=dict(site=site, start_basin=start, replicate=replicate, mode=mode,
                            native_informed_atlas=True, atlas_frozen=True,
                            conditional_atlas=conditional, initial_pose_source=pose_source,
                            source_config_sha256=sha(archive/('source-'+name)),
                            source_smc=old['metadata']['source_smc'], environment=str(archive/f'environment-{site}.json'),
                            native_poses=env.get('native_poses', [env['native_pose']]),
                            rigid_members=env['rigid_members'], member_error_scale=old.get('member_error_scale', 2.),
                            angle_error_scale_deg=old.get('angle_error_scale_deg', 15.),
                            source_periodic_box=old['box_lengths'],
                            native_core_q=.8, other_core_q=2., strict_other_core_q=5.,
                            initialization_equilibrated=False))
                    path = out/'configs'/f'{identifier}.json'
                    write(path, cfg)
                    jobs.append(dict(id=identifier, site=site, start=start, replicate=replicate, mode=mode,
                        method='mixture' if mode == 'mixture' else 'involution', correlation=MODES[mode],
                        config=str(path), config_sha256=sha(path), directory=str(out/'runs'/identifier)))
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    manifest = dict(schema=1, created_unix_time=time.time(), jobs=jobs, preparations=preparations,
        cycles=args.cycles, sample_every=1, workers=args.workers, sites=args.sites, starts=args.starts,
        modes=args.modes, source=str(source), git_head=head, binary_sha256=sha(archive/'docking-mc'),
        model_sha256=sha(archive/'model.json'), input_sha256={name: sha(archive/name) for name in inputs},
        component_count=len(model['weights']), conditional_coverage_control=conditional,
        frame_conversion_checks=conversion_checks,
        target='One mobile rigid tetramer in the fixed capture ball with one fixed neighbor and implicit ideal depletants. All proper rotations. Native is an observable only.',
        proposal_scope='Two identical local attempts plus one global attempt per cycle. Involutions use the static symmetric directed product w_a*w_b including self pairs. Mixture uses its complete normalized proposal density; c=0 is a componentwise auxiliary-MH control, not the same transition kernel.',
        inference_scope='Matched selected SMC endpoints, not equilibrated basin samples; finite MC passages are algorithmic sampling diagnostics, not physical kinetic rates or a proof of equilibrium.')
    write(out/'manifest.json', manifest)
    print(json.dumps(dict(prepared=True, jobs=len(jobs), out=str(out))), flush=True)
    if args.prepare_only:
        return

    def run(job):
        command = [str(archive/'docking-mc'), '--config', job['config'], '--model', str(archive/'model.json'),
                   '--out', job['directory'], '--cycles', str(args.cycles), '--sample-every', '1',
                   '--method', job['method']]
        if job['correlation'] is not None:
            command += ['--correlation', str(job['correlation'])]
        started = time.monotonic()
        with (out/'logs'/f'{job["id"]}.log').open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            write(out/'logs'/f'{job["id"]}-process.json', dict(pid=process.pid, command=command))
            code = process.wait()
        path = Path(job['directory'])/'summary.json'
        summary = json.loads(path.read_text()) if path.exists() else {}
        return dict(id=job['id'], returncode=code, complete=summary.get('complete', False),
                    cpu_seconds=summary.get('sampler_cpu_seconds'), wall_seconds=time.monotonic()-started)

    records = []
    started = time.monotonic()
    write(out/'status.json', dict(running=True, completed=0, total=len(jobs), supervisor_pid=os.getpid()))
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            records.append(future.result())
            print(json.dumps(records[-1]), flush=True)
            write(out/'status.json', dict(running=True, completed=len(records), total=len(jobs), records=records))
    assert all(sha(archive/name) == digest for name, digest in manifest['input_sha256'].items())
    complete = all(r['returncode'] == 0 and r['complete'] for r in records)
    result = dict(complete=complete, records=records, wall_seconds=time.monotonic()-started,
                  total_sampler_cpu_seconds=sum(r['cpu_seconds'] or 0. for r in records))
    write(out/'summary.json', result)
    write(out/'status.json', dict(result, running=False, completed=len(records), total=len(jobs)))
    if not complete:
        raise SystemExit('At least one docking run failed')


if __name__ == '__main__':
    main()
