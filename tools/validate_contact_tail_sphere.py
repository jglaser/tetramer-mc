#!/usr/bin/env python3
"""Two fresh analytic sphere runs through the exact 92-component Rust guide."""
from __future__ import annotations
import argparse
import copy
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'

from prepare_contact_tail_expansion import read, sha, require, write_new
from prepare_shoulder_docking_benchmark import local_dependencies
from run_mobile_posterior_pilot import verify_bundle
from validate_conditional_ray_sphere import reference

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [610019101, 610019102]


def validate(out, preparation, repository=ROOT, reuse_zero=None):
    out, preparation, repository = map(lambda p: Path(p).resolve(), (out, preparation, repository))
    require(not out.exists(), 'Fresh sphere reference directory required')
    if reuse_zero is not None:
        reuse_zero = Path(reuse_zero).resolve()
        old_summary = read(reuse_zero/'runs/r00/summary.json')
        require(old_summary['complete'] and old_summary['samples'] == 16384 and
                old_summary['manifest']['seed'] == SEEDS[0] and
                old_summary['manifest']['activity'] == 0. and
                old_summary['samples_sha256'] == sha(reuse_zero/'runs/r00/samples.jsonl'),
                'Only the exact completed zero-activity reference may be reused')
    completed = read(preparation/'completion.json')
    for name, digest in completed['files'].items():
        require(sha(preparation/name) == digest, 'Prepared guide evidence changed')
    guide = read(preparation/'expanded-guide.json')
    require(len(guide['gaussian_components']) == 92 and guide['defensive_uniform_shell_probability'] == .5,
            'Candidate law differs')
    pilot = read(preparation/'prospective-pilot-plan.json')
    previous = set()
    from run_protected_guide_validation import declared_seeds
    for path in pilot['earlier_protocol_sha256']:
        require(sha(path) == pilot['earlier_protocol_sha256'][path], 'Earlier declaration changed')
        previous.update(declared_seeds(read(path)))
    require(not set(SEEDS)&(previous|set(pilot['seeds'])), 'Sphere seed collision')
    old = repository/'runs/protected-guide-validation-20260923/common'
    binary, bundle_path = old/'latent-region-normalizer', old/'source-bundle.json'
    bundle, rust = verify_bundle(binary, bundle_path, old/'source')
    require(sha(binary) == pilot['source_binary_sha256'] and sha(bundle_path) == pilot['source_bundle_sha256'],
            'Candidate validation uses another executable')
    out.mkdir(parents=True); common = out/'common'; common.mkdir()
    sources = local_dependencies([Path(__file__)])
    for name, path in sources.items(): shutil.copy2(path, common/name)
    shutil.copy2(binary, common/'latent-region-normalizer'); shutil.copy2(bundle_path, common/'source-bundle.json')
    fixtures = repository/'runs/conditional-ray-sphere-validation-20260921'
    declaration = dict(schema='contact-tail-sphere-reference-v1', seeds=SEEDS, samples_per_run=16384,
        activities=[0., 2.], independent_clouds=2, auxiliary_intensity_ratio=128,
        candidate_sha256=sha(preparation/'expanded-guide.json'), candidate_preparation_sha256=sha(preparation/'completion.json'),
        binary_sha256=sha(binary), source_bundle_sha256=sha(bundle_path), rust_sources=rust,
        guide_change='Only region_sha256 rebound to analytic sphere fixture; all92 means/covariances/weights unchanged',
        reference='Existing independent AO plus normalized proper-rotation Haar one-dimensional quadrature',
        criteria='Each Q0/Qz within6.5 observed row SE plus1e-7; every attempted density/Jacobian independently reconstructed',
        source_sha256={name: sha(common/name) for name in sources}, physical_target='spheres only, no protein draws')
    declaration['reused_zero_activity_output'] = None if reuse_zero is None else dict(
        source=str(reuse_zero), summary_sha256=sha(reuse_zero/'runs/r00/summary.json'),
        reason='Original harness used conditional-ray campaign schema for Gaussian guide; physical rows unchanged; new Gaussian-schema audit wrapper')
    write_new(out/'declaration.json', declaration)
    results = []
    for index, activity in enumerate((0., 2.)):
        base = out/f'z{index}'; archive = base/'provenance'; archive.mkdir(parents=True)
        old_archive = (reuse_zero/'provenance' if index == 0 and reuse_zero is not None
                       else fixtures/f'z{index}'/'provenance')
        for name in ('shape.json', 'region.json'):
            shutil.copy2(old_archive/name, archive/name)
        if index == 0 and reuse_zero is not None:
            shutil.copy2(old_archive/'config.json', archive/'config.json')
        else:
            config = read(old_archive/'config.json'); config['shape'] = str(archive/'shape.json')
            write_new(archive/'config.json', config)
        rebound = copy.deepcopy(guide); rebound['region_sha256'] = sha(archive/'region.json')
        write_new(archive/'importance-guide.json', rebound)
        for name, path in [('latent-region-normalizer', binary), ('source-bundle.json', bundle_path)]:
            shutil.copy2(path, archive/name)
        for name, path in sources.items(): shutil.copy2(path, archive/name)
        job = dict(id='r00', seed=SEEDS[index], samples=16384,
            directory=str(reuse_zero/'runs/r00' if index == 0 and reuse_zero is not None else base/'runs/r00'))
        manifest = dict(schema='importance-latent-region-campaign-v1', jobs=[job], workers=1,
            lambda_ratio=128., cloud_replicates=2, region_sha256=sha(archive/'region.json'),
            importance_guide_sha256=sha(archive/'importance-guide.json'),
            archive_sha256={p.name: sha(p) for p in archive.iterdir() if p.is_file()})
        write_new(base/'manifest.json', manifest)
        command = [str(common/'latent-region-normalizer'), '--config', str(archive/'config.json'),
            '--region', str(archive/'region.json'), '--importance-guide', str(archive/'importance-guide.json'),
            '--out', job['directory'], '--samples', '16384', '--seed', str(job['seed']),
            '--cloud-replicates', '2', '--lambda-ratio', '128']
        if index == 0 and reuse_zero is not None:
            write_new(base/'status.json', dict(returncode=0, reused=str(reuse_zero/'runs/r00'),
                source_status_sha256=sha(reuse_zero/'status.json'), new_physical_draws=0))
        else:
            with (base/'physical.log').open('xb') as stream:
                physical = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            write_new(base/'status.json', dict(returncode=physical.returncode, command=command))
            physical.check_returncode()
        with (base/'audit.log').open('xb') as stream:
            audited = subprocess.run([sys.executable, '-B', str(common/'analyze_latent_region.py'),
                '--root', str(base)], stdout=stream, stderr=subprocess.STDOUT)
        audited.check_returncode()
        analysis = read(base/'assessment/analysis.json'); checks = []
        require(analysis['independently_reconstructed_poses'] == 16384, 'Missing attempted sphere rows')
        for key, act in [('estimate', activity), ('hard_region', 0.)]:
            estimate = analysis[key]; exact, quadrature_error = reference(act)
            mean = math.exp(estimate['logQ']); se = mean*estimate['relative_se']
            require(abs(mean-exact) < 6.5*se+1e-7, 'Analytic sphere integral failed')
            checks.append(dict(estimand=key, estimate=mean, exact=exact, observed_row_SE=se,
                standardized_error=(mean-exact)/se, quadrature_error=quadrature_error, passed=True))
        sampling = analysis['importance_sampling']
        require(sampling['gaussian_component_count'] == 92 and sampling['uniform_shell_probability'] == .5,
                'Wrong Rust proposal law')
        results.append(dict(activity=activity, seed=SEEDS[index], attempts=16384,
            checks=checks, proposal_audit=sampling, analysis_sha256=sha(base/'assessment/analysis.json'),
            sampler_cpu_seconds=analysis['sampler_cpu_seconds']))
        print('Sphere validation passed', activity, flush=True)
    value = dict(schema=declaration['schema'], complete=True, results=results,
        total_attempted_draws=32768, protein_draws=0, candidate_sha256=declaration['candidate_sha256'],
        binary_sha256=sha(binary), source_bundle_sha256=sha(bundle_path),
        declaration_sha256=sha(out/'declaration.json'), all_checks_passed=True)
    write_new(out/'validation.json', value)
    write_new(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--reuse-zero', type=Path)
    args = parser.parse_args(); validate(args.out, args.preparation, reuse_zero=args.reuse_zero)
