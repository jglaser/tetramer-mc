#!/usr/bin/env python3
"""Freeze the reviewed full-wall sphere law and fixed references; never launch it."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

from analyze_mobile_native_pocket import local_sources
from run_native_class_physical_campaign import read, require

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'results/native-class-vessel-cli-validation-20261004/attempt01/populations'
SCHEMA = 'native-class-vessel-sphere-preparation-v1'
CENTER = [7., -4., 3.]
CORE, RD, WALL, SOURCE_CAPTURE = .3, .5, 3., 1.4
ACTIVITIES = (0., .4)
POPULATIONS, DRAWS, TOTAL, MAXIMUM_CLOUDS = 4, 2048, 16384, 32768
REGIONS = dict(total=(.6, 2.7), contact=(.6, 1.6), unbound=(1.6, 2.7),
               inside_source=(.6, 1.4), outside_source=(1.4, 2.7))
CRITERIA = dict(population_SE_multiplier=4., maximum_absolute_log_discrepancy=.1,
    floating_point_relative_allowance=5e-12, cloud_diagnostic_z_limit=5., paired_residual_sigma_multiplier=6.,
    interpretation='Correlated fixed implementation-reference checks, not simultaneous confidence or protein convergence.')
CHECKED = {
    'config.json':'09299eac7cec8b91555fa683acc09b54e36ab8f3f4c7f025582e962eecc252cc',
    'shape.json':'86f17f107a484208e90063043122e58ebe91968f91378515397eb4bb6c63bc6e',
    'model.json':'1a4c0745c8740f2c1533d9add46e3c56e49c4fce6b4e5f1c8cbfe50623ad06f7',
    'region.json':'392dd7a899068afcc7c1c90bb48a89f9268756f56eb4df55012cf21dff8521da',
    'class-guide.json':'61615059ccd5ebf65616eea5d53f078a7716019cf3f83e22d8810d9e044aa3b6',
    'compiled-native.json':'fc34d60ab65591476659ff722c31334e33958c8d0c9d401a1f38e16def844430',
    'reference.json':'8936f6325b4aabd408088ae2605fd76e8227eaa43a9118ef5bcbdb896fc37601',
    'allocation.json':'2317a43658f4f8af22cef6aaca8a927d8f80cbee77c640520c45a2d4bb125110',
}
VALIDATION_SHA = '45af45cdad928861917b4715257adbd98ac5a0ba2940bd2586aa9a93122af475'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def overlap(r):
    require(math.isfinite(r) and r >= 0., 'Invalid sphere separation')
    a = CORE+RD
    return math.pi*(4*a+r)*(2*a-r)**2/12 if r < 2*a else 0.


def reference_mass(activity, lower, upper, subdivisions=4096):
    """Radial integral and real-arithmetic Simpson remainder bound, FP separate."""
    require(activity in ACTIVITIES and .6 <= lower < upper <= 2.7, 'Wrong reference domain')
    require(type(subdivisions) is int and subdivisions >= 2 and subdivisions % 2 == 0,
            'Positive even Simpson count required')
    volume = lambda lo, hi: 4*math.pi*(hi**3-lo**3)/3
    if activity == 0. or lower >= 1.6:
        return dict(mass=volume(lower, upper), quadrature_error_bound=0., method='closed-form shell volume')
    hi = min(upper, 1.6); step = (hi-lower)/subdivisions
    f = lambda r: 4*math.pi*r*r*math.exp(activity*overlap(r))
    value = step/3*math.fsum([f(lower), f(hi), *(
        (4 if i % 2 else 2)*f(lower+i*step) for i in range(1, subdivisions))])
    if upper > 1.6: value += volume(1.6, upper)
    a = CORE+RD
    first, second, third = activity*math.pi*a*a, activity*math.pi*a, activity*math.pi/2
    g2 = second+first*first
    g3 = third+3*first*second+first**3
    g4 = 4*first*third+3*second**2+6*first*first*second+first**4
    fourth = 4*math.pi*math.exp(activity*overlap(lower))*(hi*hi*g4+8*hi*g3+12*g2)
    bound = (hi-lower)*step**4*fourth/180
    return dict(mass=value, quadrature_error_bound=bound, subdivisions=subdivisions,
        fourth_derivative_absolute_bound=fourth, method='split composite Simpson; analytic unbound shell',
        obligation='Remainder bound is in real arithmetic; FP evaluation uses the separately declared relative allowance.')


def references():
    return {str(z): {name: reference_mass(z, *bounds) for name, bounds in REGIONS.items()} for z in ACTIVITIES}


def jobs_for(root, python):
    root = Path(root).resolve(); common, inputs = root/'common', root/'inputs'
    jobs = []
    for zi, z in enumerate(ACTIVITIES):
        for p in range(POPULATIONS):
            identity = f'z{z:g}-p{p}'; destination = root/'populations'/identity
            seed = 6100410001+1000*zi+p
            argv = [str(common/'basin-normalizer'), '--config', str(inputs/'config.json'),
                '--model', str(inputs/'model.json'), '--latent-region', str(inputs/'region.json'),
                '--latent-guide', str(inputs/'class-guide.json'), '--out', str(destination),
                '--samples', str(DRAWS), '--seed', str(seed), '--cloud-replicates', '2',
                '--covariance-scale', '1', '--activity', str(z), '--uniform-probability', '.4',
                '--wall-radius', str(WALL), '--wall-center', *map(str, CENTER)]
            audit = root/'audits'/(identity+'.json')
            audit_argv = [python, '-B', str(common/'physical_native_class_line_vessel.py'),
                '--directory', str(destination), '--out', str(audit), '--synthetic']
            jobs.append(dict(id=identity, activity=z, population=p, samples=DRAWS, seed=seed,
                argv=argv, directory=str(destination), audit_argv=audit_argv, audit=str(audit),
                expected_outputs=['manifest.json','config.json','summary.json','samples.jsonl','attempts.jsonl',
                    'provenance/compiled-native.json','provenance/source-bundle.json'],
                producer_terminal=dict(path=str(destination/'summary.json'), success_contract='complete'),
                audit_terminal=dict(path=str(audit), success_contract='complete')))
    require(len(jobs) == 8 and len({j['seed'] for j in jobs}) == 8, 'Changed sphere allocation')
    return jobs


def bind_bundle(binary, bundle):
    raw = Path(bundle).read_bytes(); source = read(bundle)
    require(Path(binary).is_file() and os.access(binary, os.X_OK) and raw in Path(binary).read_bytes(),
            'Exact source bundle is not embedded in supplied executable')
    require(source['schema'] == 1 and source['files'], 'Empty/unsupported source bundle')
    for name, entry in source['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe Rust source path')
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], 'Corrupt Rust source text')
    return source


def prepare(binary, bundle, out):
    binary, bundle, out = (Path(p).resolve() for p in (binary, bundle, out))
    require(not out.exists(), 'Fresh inert sphere preparation directory required')
    source_bindings = {str(SOURCE/name): digest for name, digest in CHECKED.items()}
    source_bindings[str(SOURCE.parent/'validation.json')] = VALIDATION_SHA
    for name, digest in source_bindings.items(): require(sha(name) == digest, 'Checked fixture changed '+name)
    completed = read(SOURCE.parent/'validation.json')
    require(completed['complete'] is True and completed['passed'] is True and completed['child_drained'] is True
        and completed['returncode'] == 0 and read(SOURCE/'reference.json')['passed'] is True,
        'Reviewed CLI fixture was not drained and passed')
    source = bind_bundle(binary, bundle)
    old_bundle = SOURCE/'class-depletion/provenance/source-bundle.json'
    require(read(old_bundle) == source, 'Release Rust closure differs from reviewed CLI implementation')
    source_bindings.update({str(p): sha(p) for p in (binary, bundle, old_bundle)})
    closure = {}
    for entry in (__file__, Path(__file__).with_name('analyze_native_class_vessel_sphere.py'),
                  Path(__file__).with_name('physical_native_class_line_vessel.py'),
                  Path(__file__).with_name('run_native_class_physical_campaign.py')):
        closure.update(local_sources(entry))
    for p in closure.values(): source_bindings[str(p)] = sha(p)
    out.mkdir(parents=True); common, inputs = out/'common', out/'inputs'; common.mkdir(); inputs.mkdir()
    for name, path in closure.items(): shutil.copy2(path, common/name)
    shutil.copy2(binary, common/'basin-normalizer'); shutil.copy2(bundle, common/'source-bundle.json')
    for name, entry in source['files'].items():
        path = common/'rust-source'/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry['text'].encode()); require(sha(path) == entry['sha256'], 'Rust archive changed')
    for name in CHECKED: shutil.copy2(SOURCE/name, inputs/('source-'+name))
    for name in ('shape.json','model.json','region.json','compiled-native.json'):
        shutil.copy2(SOURCE/name, inputs/name)
    config = read(SOURCE/'config.json'); config['shape'] = str(inputs/'shape.json'); write(inputs/'config.json', config)
    guide = read(SOURCE/'class-guide.json'); guide['compiled_native']['path'] = str(inputs/'compiled-native.json')
    write(inputs/'class-guide.json', guide)
    # No region, native geometry, weights, covariance, source capture or target
    # edit is permitted. Only these two path fields change.
    original = read(SOURCE/'config.json'); original['shape'] = config['shape']; require(original == config, 'Physical law changed')
    original = read(SOURCE/'class-guide.json'); original['compiled_native']['path'] = guide['compiled_native']['path']
    require(original == guide, 'Guide law changed')
    analytic = dict(schema='full-wall-sphere-analytic-reference-v1', center=CENTER, core_radius=CORE,
        exclusion_radius=CORE+RD, wall_radius=WALL, source_capture=SOURCE_CAPTURE,
        radial_regions={k:list(v) for k,v in REGIONS.items()}, activities=list(ACTIVITIES), references=references(),
        measure='d3t times normalized SO(3) Haar; the full orientation integral is one',
        geometry='Four coincident atomic members form one union sphere, with no factor four or sixteen.',
        formula='Qz(A)=4*pi*integral_A r^2*exp[z*pi*(3.2+r)*(1.6-r)^2/12]dr below1.6; overlap is zero above1.6.',
        Q0_formula='4*pi*(upper^3-lower^3)/3',
        partitions='Contact/unbound and inside/outside source each exhaust the physical radial shell. R4 is a proposal coordinate region, not a target restriction.')
    write(out/'analytic-reference.json', analytic)
    jobs = jobs_for(out, sys.executable)
    for name in ('populations','audits'): (out/name).mkdir()
    plan = dict(schema=SCHEMA, root=str(out), preparation_only=True, launched=False,
        total_attempts=TOTAL, maximum_clouds=MAXIMUM_CLOUDS, samples_per_population=DRAWS,
        populations_per_activity=POPULATIONS, activities=list(ACTIVITIES), jobs=jobs,
        criteria=CRITERIA, maximum_workers=1, threads=1, retries=0, replacements=0, extensions=0,
        python=sys.executable, python_sha256=sha(Path(sys.executable).resolve()),
        executable=dict(path=str(common/'basin-normalizer'), sha256=sha(common/'basin-normalizer')),
        source_bundle=dict(path=str(common/'source-bundle.json'), sha256=sha(common/'source-bundle.json')),
        source_bindings=source_bindings, archived_source_sha256={name:sha(common/name) for name in closure},
        physical=dict(core_radius=CORE, depletant_radius=RD, wall_center=CENTER, wall_radius=WALL,
            physical_capture_radius=5., source_capture_radius=SOURCE_CAPTURE, cloud_replicates=2, lambda_ratio=16.),
        proposal=dict(outer_vessel_probability=.5, guide_alpha=.5, guide_beta=1., source_law_unchanged=True),
        analytic_reference=dict(path=str(out/'analytic-reference.json'),sha256=sha(out/'analytic-reference.json')),
        analysis_contract='Only after all producer/audit jobs drain under run_native_class_physical_campaign; preserve exact per-attempt sphere companion weights and all null/hard zeros.',
        statistical_contract='Four independent population linear means; unbiased sample variance/4. Each primary exact/noisy/hard regional mass passes four-SE plus deterministic error plus FP allowance AND .1 absolute log discrepancy. No optional extension.',
        scope='Software full-wall reference only; no new protein draws, native stability, equilibration or assembly inference.')
    for path,digest in source_bindings.items(): require(sha(path) == digest, 'Input changed during preparation '+path)
    plan['files_sha256'] = {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'plan.json', plan)
    write(out/'freeze.json', dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary','bundle','out'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); value = prepare(args.binary, args.bundle, args.out)
    print(json.dumps(dict(prepared=True, launched=False, jobs=len(value['jobs']), total_attempts=TOTAL)))
