#!/usr/bin/env python3
"""Freeze a fresh schema-8/native-class vessel comparison; never dispatch.

The completed sphere reference validates implementation, not protein convergence.
The old cube/hard-free preparation and its fixed inputs remain unchanged.
"""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import copy
import hashlib
from fractions import Fraction
from pathlib import Path
import shutil
import sys

from analyze_r4_smc_control import Ledger, read, require, sha, write
from analyze_mobile_native_pocket import local_sources
from prepare_full_vessel_comparison import ROOT, SOURCES, NATIVE, STAGES, ARMS, TOTAL, WALL, THREADS
from prepare_streaming_vessel_comparison import runtime
from partition_vessel_streaming import SUPPORTS, validate_supports
from certify_vessel_region_containment import certify, read_exact
from native_class_line_physical_reference import validate_shape_witness

SCHEMA = 'native-class-streaming-vessel-preparation-v1'
SEED_BASE = 610061001
LAMBDA_RATIO = 128.
REFERENCE_ROOT = Path('/vast/xvg/tetramer-mc-runs/wall-envelope-vessel-sphere-20261004')
PRIMARY_ROOT = Path('/vast/xvg/tetramer-mc-runs/native-class-physical-primary-20261004')
BINARY_SHA = 'b9a549274929eff5c84a5242784a5db14adfdef58dcf936a5b31f65163ca8c18'
BUNDLE_SHA = '7e2fb223452ebb6cc15ccd59984a6b83c0ec45621a4a7642eaa22e1b67cd5c5f'
GUIDE = (ROOT/'results/native-class-support-pilot-20261004/common/guide.json',
         '13e5c31774c77169d7f4604b2a46b80e84b366ff2172fcef3911fab93286a179')
COMPILED_SHA = 'dbb3c3e32259f506b9c979ebb9d1fd773cb78507c1f808fd8dd0ed319e2eade4'
WITNESS_SHA = 'deb5bd014d95cf8813823e94b3bf590f9c1cb0b6c020ad90d3165cd3e9f85abb'
REFERENCES = {
    'analysis/summary.json': '6164ede77f8a7456eabc38db69e0644f976d3eeb14caac79eae21a01ef1d8366',
    'execution/summary.json': '352a0070e9e53ae2e28f25c60173367cae31847483ca0b9beac5d88ccd24cf53',
    'execution-plan.json': '4fa24d933702fd8a991c3bec057d52bae628b0fa317564f6cdf8dc889db62f27',
    'plan.json': 'dbff991dc467d063aa88dff462657714ee9e4a8f7b3a9a902e59474500e4f2af',
}
AUDIT_ENTRIES = {'vessel': 'audit_vessel_baseline_streaming.py',
                 'half_mixture': 'audit_native_class_vessel_streaming.py'}
AUDIT_SCHEMAS = {'vessel': 'full-vessel-baseline-streaming-audit-v1',
                 'half_mixture': 'full-vessel-native-class-line-streaming-audit-v1'}
CHANNELS = [dict(**{'class': c}, probability=.2, **({} if o is None else dict(orthant=o)))
            for c, o in [('hard_free', None), ('contact_without_native', None),
                         ('contact_without_native', 22), ('contact_without_native', 62), ('native', 55)]]


def physical():
    return dict(depletant_radius=1.5, activity=.035, lambda_ratio=LAMBDA_RATIO, cloud_replicates=2,
        wall_center=[0., 0., 0.], wall_radius=WALL, capture_radius=273., bath_wall_permeable=True,
        measure='Lebesgue center volume times normalized SO(3) Haar measure')


def proposal_contracts():
    common = dict(schema=8, vessel_uniform_schema='one-atom-wall-envelope-v1')
    return {'vessel': dict(common, pre_envelope_schema=4),
        'half_mixture': dict(common, pre_envelope_schema=7,
            outer_mixture_schema='full-vessel-native-class-line-half-mixture-v1', outer_vessel_probability=.5,
            latent_guide_schema='defensive-native-class-line-guide-v1', latent_gaussian_component_count=116,
            latent_defensive_uniform_probability=.5)}


def jobs_for(out, python=None):
    out = Path(out).resolve(); common, inputs = out/'common', out/'inputs'; jobs = []
    python = python or sys.executable
    for stage, count in STAGES:
        for population in range(4):
            for arm in ARMS:
                base = out/stage/arm; destination = base/'runs'/f'r{population:02}'
                audit, partition = base/'audits'/f'r{population:02}', base/'partitions'/f'r{population:02}'
                seed = SEED_BASE+len(jobs)
                command = [str(common/'basin-normalizer'), '--config', str(inputs/'config.json'),
                    '--model', str(inputs/'model.json'), '--out', str(destination), '--samples', str(count),
                    '--seed', str(seed), '--cloud-replicates', '2', '--covariance-scale', '1',
                    '--uniform-probability', '0.1', '--wall-radius', str(WALL), '--wall-center', '0', '0', '0',
                    '--wall-uniform-envelope']
                if arm == 'half_mixture':
                    command += ['--latent-region', str(inputs/'current_R4.json'), '--latent-guide', str(inputs/'guide.json')]
                audit_command = [python, '-B', str(common/AUDIT_ENTRIES[arm]), '--directory', str(destination),
                    '--out', str(audit), '--binary', str(common/'basin-normalizer'), '--batch-size', '64']
                audit_command += (['--region', str(inputs/'current_R4.json')] if arm == 'vessel' else
                                  ['--definition', str(inputs/'native-region/definition.json')])
                partition_command = [python, '-B', str(common/'partition_vessel_streaming.py'),
                    '--audit', str(audit/'analysis.json'), '--native-definition', str(inputs/'native-region/definition.json'),
                    '--out', str(partition)]
                for name in SUPPORTS:
                    partition_command += ['--'+name.replace('_', '-'), str(inputs/(name+'.json'))]
                jobs.append(dict(id=f'{stage}-{arm}-r{population:02}', stage=stage, arm=arm,
                    population=population, samples=count, seed=seed, directory=str(destination), command=command,
                    log=str(base/'logs'/f'r{population:02}.log'), audit_directory=str(audit), audit_command=audit_command,
                    partition_directory=str(partition), partition_command=partition_command))
    require(len(jobs) == 16 and len({j['seed'] for j in jobs}) == 16
            and sum(j['samples'] for j in jobs) == TOTAL, 'Changed fixed allocation')
    return jobs


def validate_guide(guide, source, compiled_path):
    expected = copy.deepcopy(source); expected['compiled_native']['path'] = str(Path(compiled_path).resolve())
    require(guide == expected, 'Class guide changed beyond compiled-native path')
    require(guide['schema'] == proposal_contracts()['half_mixture']['latent_guide_schema']
            and len(guide['gaussian_components']) == 116 and guide['class_channels'] == CHANNELS
            and guide['raw_translation_axes'] == [0, 1, 2] and guide['conditional_probability'] == 1.
            and guide['defensive_uniform_shell_probability'] == .5
            and guide['compiled_native']['sha256'] == COMPILED_SHA
            and guide['region_sha256'] == SOURCES['current_R4.json'][1], 'Changed frozen class guide')


def validate_contract(plan, out):
    require(plan['schema'] == SCHEMA and plan['preparation_only'] is True
            and plan['physical_jobs_launched'] == 0 and plan['dispatch_ready'] is False,
            'Not an inert native-class preparation')
    require(plan['jobs'] == jobs_for(out) and plan['total_unconditional_draws'] == TOTAL
            and plan['stages'] == [dict(name=s, draws_per_population=n, independent_populations_per_arm=4) for s, n in STAGES]
            and plan['arms'] == list(ARMS), 'Changed fixed stage allocation')
    require(plan['physical'] == physical() and plan['proposal_contracts'] == proposal_contracts(),
            'Changed physical target or explicit proposal law')
    require(plan['maximum_physical_workers'] == 8 and plan['maximum_audit_workers'] == 4
            and plan['maximum_all_workers'] == 32 and plan['thread_environment'] == THREADS,
            'Changed worker limits')
    require(plan['native_definition_sha256'] == NATIVE[1]
            and plan['input_sha256']['compiled-native.json'] == COMPILED_SHA
            and plan['input_sha256']['shape-compatibility.json'] == WITNESS_SHA
            and plan['input_sha256']['source-guide.json'] == GUIDE[1], 'Changed native/guide identity')
    require(plan['input_sha256']['native-region/definition.json'] == NATIVE[1], 'Changed archived native definition')
    for name in ('source-config.json', 'shape.json', 'model.json', *[k+'.json' for k in SUPPORTS]):
        require(plan['input_sha256'][name] == SOURCES[name][1], 'Changed fixed vessel input: '+name)


def verify_embedded(binary, bundle_path):
    raw = Path(bundle_path).read_bytes()
    require(raw in Path(binary).read_bytes(), 'Source bundle absent from executable')
    bundle = read(bundle_path)
    for name, item in bundle['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Unsafe source bundle path')
        require(hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256'], 'Invalid embedded source digest')
    return bundle


def validate_inventory(out, frozen, jobs):
    """Exact immutable payload; only declared per-job outputs may appear later."""
    out = Path(out).resolve()
    immutable = {'plan.json'}
    for prefix in ('common', 'inputs'):
        immutable.update(str(p.relative_to(out)) for p in (out/prefix).rglob('*') if p.is_file())
    require(set(frozen) == immutable, 'Unfrozen or missing immutable preparation file')
    mutable = [Path(j[k]).resolve() for j in jobs for k in ('directory', 'audit_directory', 'partition_directory')]
    logs = {Path(j['log']).resolve() for j in jobs}
    require(all(p.is_relative_to(out) and p != out for p in [*mutable, *logs]), 'Mutable output escapes preparation')
    for p in out.rglob('*'):
        if not p.is_file() or p == out/'freeze.json' or str(p.relative_to(out)) in immutable:
            continue
        require(p.resolve() in logs or any(p.resolve().is_relative_to(d) for d in mutable),
                'Undeclared mutable preparation output: '+str(p))


def freeze(out):
    out = Path(out).resolve(); require(not out.exists(), 'Fresh inert preparation required')
    require(sys.flags.optimize == 0, 'Unoptimized Python required')
    ledger = Ledger()
    reports = {name: read(ledger.bind(REFERENCE_ROOT/name, digest)) for name, digest in REFERENCES.items()}
    report = reports['analysis/summary.json']
    require(report['complete'] is True and report['passed'] is True and report['attempts'] == 131072
            and len(report['checks']) == 47 and all(c['passed'] is True for c in report['checks'])
            and report['failed_checks'] == [] and report['plan_sha256'] == REFERENCES['plan.json'],
            'Incomplete analytic wall-envelope reference')
    reference_plan = reports['plan.json']
    execution = reports['execution/summary.json']
    require(execution['complete'] is True and execution['passed'] is True
            and execution['plan_sha256'] == REFERENCES['execution-plan.json']
            and len(execution['completed']) == 17 and execution['active'] is None
            and execution['unstarted'] == [] and execution['failure'] is None
            and execution['retries'] == execution['replacements'] == 0,
            'Wall-envelope reference execution incomplete')
    for key, digest in [('executable', BINARY_SHA), ('source_bundle', BUNDLE_SHA)]:
        require(reference_plan[key]['sha256'] == digest, 'Reference binary identity differs')
    binary = ledger.bind(REFERENCE_ROOT/'common/basin-normalizer', BINARY_SHA)
    bundle_path = ledger.bind(REFERENCE_ROOT/'common/source-bundle.json', BUNDLE_SHA)
    bundle = verify_embedded(binary, bundle_path)
    sources = {name: (ROOT/path, digest) for name, (path, digest) in SOURCES.items() if name != 'guide.json'}
    sources.update({'source-guide.json': GUIDE,
        'compiled-native.json': (PRIMARY_ROOT/'common/compiled-native.json', COMPILED_SHA),
        'shape-compatibility.json': (PRIMARY_ROOT/'common/shape-compatibility.json', WITNESS_SHA)})
    for path, digest in sources.values(): ledger.bind(path, digest)
    definition_path = ledger.bind(ROOT/NATIVE[0], NATIVE[1]); definition = read(definition_path)
    compiled = read(sources['compiled-native.json'][0])
    require(compiled['source_definition_sha256'] == NATIVE[1]
            and compiled['source_input_sha256'] == definition['input_sha256'], 'Compiled definition provenance differs')
    native_inputs = {}
    for name, digest in definition['input_sha256'].items():
        path = (definition_path.parent/'inputs'/name).resolve()
        require(path.is_relative_to((definition_path.parent/'inputs').resolve()), 'Native input escapes archive')
        native_inputs[name] = ledger.bind(path, digest)
    witness = validate_shape_witness(compiled, read(sources['shape.json'][0]), read(sources['shape-compatibility.json'][0]),
                                     COMPILED_SHA, SOURCES['shape.json'][1])
    closure = {}
    for entry in (Path(__file__).name, *AUDIT_ENTRIES.values(), 'partition_vessel_streaming.py',
                  'analyze_streaming_vessel_stage.py'):
        closure.update(local_sources(Path(__file__).with_name(entry)))
    for path in closure.values(): ledger.bind(path)
    out.mkdir(parents=True); common, inputs = out/'common', out/'inputs'; common.mkdir(); inputs.mkdir()
    for name, (path, _) in sources.items(): shutil.copy2(path, inputs/name)
    native_root = inputs/'native-region'; (native_root/'inputs').mkdir(parents=True)
    shutil.copy2(definition_path, native_root/'definition.json')
    for name, path in native_inputs.items():
        destination = native_root/'inputs'/name; destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    config = read(inputs/'source-config.json'); config.update(shape=str(inputs/'shape.json'), poisson_lambda_ratio=LAMBDA_RATIO)
    write(inputs/'config.json', config)
    guide = read(inputs/'source-guide.json'); guide['compiled_native']['path'] = str(inputs/'compiled-native.json')
    validate_guide(guide, read(inputs/'source-guide.json'), inputs/'compiled-native.json'); write(inputs/'guide.json', guide)
    validate_supports({k: read(inputs/(k+'.json')) for k in SUPPORTS}, config, dict(shape_sha256=SOURCES['shape.json'][1]))
    certificate = certify(read_exact(inputs/'current_R4.json'), read_exact(inputs/'shape.json'), [0, 0, 0],
        Fraction(str(WALL)), [0, 0, 0], Fraction(273))
    require(certificate['wall_containment_proven'] and certificate['capture_containment_proven'], 'Unresolved R4 inclusion')
    write(inputs/'r4-vessel-containment.json', certificate)
    for name, path in closure.items(): shutil.copy2(path, common/name)
    shutil.copy2(binary, common/'basin-normalizer'); shutil.copy2(bundle_path, common/'source-bundle.json')
    for name, item in bundle['files'].items():
        path = common/'source'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(item['text'])
    ledger.recheck()
    plan = dict(schema=SCHEMA, preparation_only=True, physical_jobs_launched=0, dispatch_ready=False,
        stages=[dict(name=s, draws_per_population=n, independent_populations_per_arm=4) for s, n in STAGES],
        arms=list(ARMS), jobs=jobs_for(out), total_unconditional_draws=TOTAL,
        maximum_physical_workers=8, maximum_audit_workers=4, maximum_all_workers=32, thread_environment=THREADS,
        binary_sha256=BINARY_SHA, source_bundle_sha256=BUNDLE_SHA, runtime=runtime(),
        sources={name: sha(common/name) for name in closure},
        input_sha256={str(p.relative_to(inputs)): sha(p) for p in inputs.rglob('*') if p.is_file()},
        source_bindings=ledger.files, native_definition_sha256=NATIVE[1], shape_witness=witness,
        physical=physical(), proposal_contracts=proposal_contracts(),
        baseline_choice='Schema 8 one-atom wall envelope replaces the legacy cube component; target unchanged, proposal changed.',
        prerequisites=dict(regional='All original regional/sensitivity gates and matching SMC evidence require separate admission.',
            execution='Reviewed bounded dispatcher with runtime/resource/receipt authentication remains required.',
            failures='No replacement draws or retries; retain every attempted draw and drain all started children.'),
        scope='Fresh immutable preparation only. No physical launch, no convergence admission, no assembly inference.')
    write(out/'plan.json', plan)
    write(out/'freeze.json', dict(files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
    return validate(out)


def validate(out):
    out = Path(out).resolve(); ledger = Ledger(); frozen = ledger.frozen(out)
    plan = read(ledger.bind(out/'plan.json')); validate_contract(plan, out)
    validate_inventory(out, frozen, plan['jobs'])
    require(plan['runtime'] == runtime(), 'Frozen runtime differs')
    require(plan['binary_sha256'] == BINARY_SHA and plan['source_bundle_sha256'] == BUNDLE_SHA,
            'Changed validated executable')
    for prefix, key in [('common', 'sources'), ('inputs', 'input_sha256')]:
        for name, digest in plan[key].items():
            path = (out/prefix/name).resolve(); require(path.is_relative_to(out/prefix), 'Archive path escapes preparation')
            require(prefix+'/'+name in frozen, 'Prepared input omitted from freeze'); ledger.bind(path, digest)
    ledger.bind(out/'common/basin-normalizer', BINARY_SHA); ledger.bind(out/'common/source-bundle.json', BUNDLE_SHA)
    verify_embedded(out/'common/basin-normalizer', out/'common/source-bundle.json')
    inputs = out/'inputs'; source = read(inputs/'source-config.json')
    source.update(shape=str(inputs/'shape.json'), poisson_lambda_ratio=LAMBDA_RATIO)
    require(read(inputs/'config.json') == source, 'Config changed beyond archived paths/cloud intensity')
    validate_guide(read(inputs/'guide.json'), read(inputs/'source-guide.json'), inputs/'compiled-native.json')
    definition = read(inputs/'native-region/definition.json'); compiled = read(inputs/'compiled-native.json')
    require(compiled['source_definition_sha256'] == NATIVE[1]
            and compiled['source_input_sha256'] == definition['input_sha256'], 'Compiled original source differs')
    for name, digest in definition['input_sha256'].items():
        require(plan['input_sha256'].get('native-region/inputs/'+name) == digest, 'Incomplete native source closure')
    witness = validate_shape_witness(compiled, read(inputs/'shape.json'), read(inputs/'shape-compatibility.json'),
                                     COMPILED_SHA, SOURCES['shape.json'][1])
    require(witness == plan['shape_witness'], 'Changed complete shape correspondence')
    ledger.recheck(); return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('freeze', 'validate'))
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    result = freeze(args.out) if args.action == 'freeze' else validate(args.out)
    print(dict(preparation_only=True, jobs=len(result['jobs']), draws=result['total_unconditional_draws'], dispatch_ready=False))
