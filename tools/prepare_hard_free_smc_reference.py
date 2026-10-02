#!/usr/bin/env python3
"""Freeze fresh analytic sphere/Haar inputs for guided SMC; never launch jobs."""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import shutil
from prepare_hard_free_line_score import read, require, sha, write
from prepare_hard_free_line_sensitivity_reference import PYTHON, runtime
from prepare_hard_free_line_fresh import ROOTS
from run_hard_free_line_physical_pilot import inventory
from prepare_shoulder_docking_benchmark import local_dependencies

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(610017201+i for i in range(10))
R, ST, SR, ELL, CAPTURE, RD = 4., .6, 1.3, 1.1, 2.2, .4


def reference_integral(activity, upper=CAPTURE, subdivisions=32768):
    """Physical radial integral times exact normalized SO(3) Haar cap."""
    require(type(subdivisions) is int and subdivisions > 0 and subdivisions % 2 == 0,
            'Simpson subdivisions must be a positive even integer')
    require(math.isfinite(activity) and activity >= 0 and math.isfinite(upper) and upper >= .6,
            'Invalid analytic reference bounds')
    def f(radius):
        cayley = SR/ELL*math.sqrt(max(0., R*R-(radius/ST)**2))
        lens = math.pi*(2.8+radius)*(1.4-radius)**2/12 if radius < 1.4 else 0.
        return 8*radius**2*(math.atan(cayley)-cayley/(1+cayley*cayley))*math.exp(activity*lens)
    hi = min(upper, CAPTURE, ST*R); step = (hi-.6)/subdivisions
    return (f(.6)+f(hi)+math.fsum((4 if i % 2 else 2)*f(.6+i*step)
            for i in range(1, subdivisions)))*step/3


def diagonal(scales):
    return [[scale*scale if i == j else 0. for j in range(6)] for i, scale in enumerate(scales)]


def fixture(directory, activity, core=.3):
    directory.mkdir(parents=True)
    fixed = dict(position=[0., 0., 0.], orientation=[1., 0., 0., 0.])
    native = dict(position=[1., 0., 0.], orientation=[1., 0., 0., 0.])
    metric = dict(native_poses=[native], rigid_members=[fixed], member_error_scale=1., angle_error_scale_deg=15.)
    write(directory/'shape.json', dict(name='Guided SMC analytic sphere', volume=4*math.pi*core**3/3,
        atoms=[dict(center=[0., 0., 0.], radius=core)]))
    shape_hash = sha(directory/'shape.json')
    write(directory/'config.json', dict(shape=str(directory/'shape.json'), fixed_poses=[fixed], initial_pose=native,
        capture_center=[0., 0., 0.], capture_radius=CAPTURE, depletant_radius=RD, reservoir_density=activity,
        poisson_lambda_ratio=8., translation_steps=[.15, .4], rotation_steps_deg=[8., 25.], rotation_probability=.5,
        local_attempts_per_cycle=1, uniform_probability=.1, seed=1, metadata=metric,
        endpoint_gate=dict(max_cells=31, max_depth=8, min_width=.3)))
    write(directory/'region.json', dict(fixed_neighbor=fixed, physical_fixed_neighbors=[fixed],
        capture_center=[0., 0., 0.], capture_radius=CAPTURE, shape_sha256=shape_hash, activity=activity,
        depletant_radius=RD, physical_metric=metric, minimum_original_q=0., minimum_mahalanobis_radius=0.,
        mahalanobis_radius=R, gaussian_chart=dict(shape_sha256=shape_hash, angular_length=ELL,
        coordinate_convention='anchor-body-relative', anchors=[dict(position=[0., 0., 0.],
        rotation=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])], means=[[0.]*6],
        covariances=[diagonal([ST]*3+[SR]*3)], weights=[1.])) )
    guide = dict(schema='defensive-hard-free-line-guide-v1', region_sha256=sha(directory/'region.json'),
        defensive_uniform_shell_probability=.5, conditional_probability=1., raw_translation_axes=[0, 1, 2],
        minimum_conditional_mass=1e-12, gaussian_components=[
            dict(weight=.6, mean=[.8, -.2, .1, 0., 0., 0.], covariance=diagonal([1., 1., 1., .75, .75, .75])),
            dict(weight=.4, mean=[-.7, .3, -.1, .2, -.1, .15], covariance=diagonal([.8, 1.2, 1., .9, .8, 1.1]))])
    write(directory/'guide.json', guide)
    write(directory/'guide-uniform.json', dict(guide, defensive_uniform_shell_probability=1.))


def jobs(out):
    values = [dict(id=f'z{z}-r{i:02}', fixture=f'z{z}', activity=z, seed=SEEDS[4*a+i],
        initial_draws=4096, population=192, stages=8, sweeps=6, guide='guide.json', purpose='analytic')
        for a, z in enumerate([0, 4]) for i in range(4)]
    values += [dict(id='zero-hit', fixture='zero', activity=0, seed=SEEDS[8], initial_draws=256,
        population=192, stages=8, sweeps=6, guide='guide.json', purpose='zero-hit')]
    values += [dict(id=label, fixture='z4', activity=4, seed=SEEDS[9], initial_draws=256,
        population=32, stages=4, sweeps=2, guide=guide, purpose='paired-uniform-parity')
        for label, guide in [('uniform-guided', 'guide-uniform.json'), ('uniform-legacy', None)]]
    for job in values:
        inputs = out/'inputs'/job['fixture']; job['directory'] = str(out/'populations'/job['id'])
        argv = [str(out/'common/latent-region-smc'), '--config', str(inputs/'config.json'),
            '--region', str(inputs/'region.json'), '--out', job['directory'], '--seed', str(job['seed']),
            '--bridge', 'proposal-density', '--initial-draws', str(job['initial_draws']),
            '--population', str(job['population']), '--stages', str(job['stages']),
            '--sweeps-per-stage', str(job['sweeps']), '--cloud-replicates', '2', '--lambda-ratio', '8']
        job['argv'] = argv+(['--initial-guide', str(inputs/job['guide'])] if job['guide'] else [])
    return values


def prepare(binary, bundle, out):
    environment = runtime(); binary, bundle, out = map(lambda p: Path(p).resolve(), (binary, bundle, out))
    require(not out.exists() and binary.is_file() and bundle.is_file(), 'Fresh destination and completed build required')
    require(bundle.read_bytes() in binary.read_bytes(), 'Exact source bundle is not embedded in executable')
    source = read(bundle); require(isinstance(source.get('files'), dict) and source['files'], 'Missing source closure')
    for name, entry in source['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe embedded source path')
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'Corrupt embedded source: '+name)
    oracle = ROOT/'tests/latent_region_smc.rs'; sources = local_dependencies([
        Path(__file__), ROOT/'tools/test_prepare_hard_free_smc_reference.py', ROOT/'tools/audit_hard_free_smc.py',
        ROOT/'tools/test_audit_hard_free_smc.py'])
    inputs = {str(p): sha(p) for p in [binary, bundle, oracle, *sources.values()]}
    roots = list(dict.fromkeys([str(ROOT/'runs'), str(ROOT/'results'), *ROOTS]))
    require(all(Path(p).is_dir() for p in roots), 'Missing seed inventory root')
    seeds = inventory(roots, SEEDS); require(seeds['files'], 'Empty seed inventory')
    out.mkdir(parents=True); common = out/'common'; common.mkdir()
    shutil.copy2(binary, common/'latent-region-smc'); shutil.copy2(bundle, common/'source-bundle.json')
    for name, entry in source['files'].items():
        destination = common/'rust-source'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(entry['text']); require(sha(destination) == entry['sha256'], 'Extracted source bytes differ')
    for name, path in sources.items(): shutil.copy2(path, common/name)
    shutil.copy2(oracle, common/'latent_region_smc.rs'); write(out/'seed-inventory.json', seeds)
    for label, activity, core in [('z0', 0., .3), ('z4', 4., .3), ('zero', 0., 2.)]:
        fixture(out/'inputs'/label, activity, core)
    exact = {str(z): {name: reference_integral(z, bound) for name, bound in [('total', CAPTURE), ('contact', 1.4)]}
        for z in [0, 4]}
    for z, values in exact.items():
        values['unbound'] = values['total']-values['contact']
        require(all(abs(values[k]-reference_integral(float(z), b, 65536)) < 1e-10
            for k, b in [('total', CAPTURE), ('contact', 1.4)]), 'Analytic quadrature has not converged')
    allocation = jobs(out)
    plan = dict(schema='hard-free-smc-reference-preparation-v1', state='frozen_unlaunched', preparation_only=True,
        python=PYTHON, runtime=environment, binary_sha256=sha(common/'latent-region-smc'),
        source_bundle_sha256=sha(common/'source-bundle.json'), input_sha256=inputs, exact=exact, jobs=allocation,
        populations_per_activity=4, main_initial_attempts=32768, all_initial_attempts=sum(j['initial_draws'] for j in allocation),
        seeds=list(SEEDS), intentional_seed_reuse=dict(seed=SEEDS[9], jobs=['uniform-guided', 'uniform-legacy'],
        purpose='Same-stream alpha=1 implementation parity control, not independent evidence.',
        exact_fields=['generated poses', 'parent indices', 'accepted decisions', 'retained endpoints'],
        scalar_density_jacobian_logZ_absolute_tolerance=2e-10, latent_roundtrip_absolute_tolerance=2e-8),
        alpha=.5, conditioning_probability=1., raw_translation_axes=[0, 1, 2], minimum_conditional_mass=1e-12,
        cloud_replicates=2, lambda_ratio=8., reference_standard_error_multiplier=6.,
        reference_absolute_tolerance=1e-6,
        reference_check='Absolute difference from analytic integral <= 6 independent-population SE + 1e-6.',
        analytic_regions=dict(total='0.6 <= r <= 2.2 within R4', contact='0.6 <= r < 1.4 within R4',
            unbound='1.4 <= r <= 2.2 within R4'), analytic_measure='d3t times normalized SO(3) Haar',
        zero_hit_reference=dict(core_radius=2., capture_radius=CAPTURE, hard_separation_minimum=4., exact_mass=0.),
        maximum_workers=4, maximum_global_physical_workers=8, maximum_total_workers=32,
        thread_environment={key: '1' for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS']},
        no_retries=True, no_autoextension=True, no_optional_stopping=True, all_attempts_retained=True,
        launched=False, physical_gate_open=False, scope='Software sphere/Haar references only; no protein, convergence, '
        'mixing-speedup, full-vessel or finite-assembly evidence. Preparation starts no sampler jobs.')
    for path, digest in inputs.items(): require(sha(path) == digest, 'Input changed during preparation: '+path)
    plan['files_sha256'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'plan.json', plan)
    write(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(prepared=True, launched=False, out=str(out), jobs=len(allocation), plan_sha256=sha(out/'plan.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['binary', 'bundle', 'out']: parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); print(prepare(args.binary, args.bundle, args.out))
