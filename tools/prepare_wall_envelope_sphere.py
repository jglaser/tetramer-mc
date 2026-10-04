#!/usr/bin/env python3
"""Freeze a new wall-envelope sphere reference; never execute or extend a run."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys

from analyze_mobile_native_pocket import local_sources
import prepare_native_class_vessel_sphere as original
from prepare_native_class_vessel_sphere import (
    ACTIVITIES, CENTER, CORE, RD, WALL, SOURCE_CAPTURE, REGIONS, CRITERIA,
    read, require, sha, write, references, bind_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'wall-envelope-sphere-preparation-v1'
ALLOCATION = ROOT/'results/wall-envelope-allocation-design-20261004/allocation.json'
ALLOCATION_SHA = '4a1cda778d92c0fb3951ba7921c1493a3501315e66d5e76da2faf523b72a1d24'
CLI = ROOT/'results/wall-envelope-cli-validation-20261004/attempt01'
CLI_RECEIPT_SHA = '2abeb134b88f4fcf4818e6c1a13c4bdeaa56d16c839dd30fe210330de31e5888'
AUDIT_RECEIPT = ROOT/'results/wall-envelope-saved-audit-20261004/attempt02/validation.json'
AUDIT_RECEIPT_SHA = 'ddb03ea8036a735767f8560516b039f2a8ba7c06cb4a0a7a870fceb3b22eb18e'
POPULATIONS = 4
DRAWS_BY_ACTIVITY = {'0.0': 12288, '0.4': 20480}
TOTAL, MAXIMUM_CLOUDS = 131072, 262144
# Generous finite limits, not an invitation to extend a scientific allocation.
# The old 2,048-row audit used about ten wall seconds; the largest new job is 10x.
RESOURCE_LIMITS = {
    'producer': dict(cpu_limit_seconds=120, wall_limit_seconds=300, address_space_limit_bytes=4*1024**3),
    'geometry': dict(cpu_limit_seconds=600, wall_limit_seconds=1200, address_space_limit_bytes=4*1024**3),
    'statistics': dict(cpu_limit_seconds=300, wall_limit_seconds=600, address_space_limit_bytes=4*1024**3),
}
PHYSICAL = dict(core_radius=CORE, depletant_radius=RD, wall_center=CENTER, wall_radius=WALL,
    physical_capture_radius=5., source_capture_radius=SOURCE_CAPTURE, cloud_replicates=2, lambda_ratio=16.)
PROPOSAL = dict(manifest_schema=8, pre_envelope_schema=7,
    vessel_uniform_schema='one-atom-wall-envelope-v1', outer_vessel_probability=.5,
    inner_uniform_probability=.4, guide_alpha=.5, guide_beta=1., source_law_unchanged=True)


def validate_allocation(value):
    require(value['schema'] == 'prospective-wall-envelope-reference-allocation-v1'
        and value['prepared_execution'] is False and value['launched'] is False
        and value['total_attempts'] == TOTAL and value['populations_per_activity'] == POPULATIONS
        and value['draws_per_population'] == DRAWS_BY_ACTIVITY
        and value['retries'] == value['replacements'] == value['extensions'] == 0,
        'Changed prospective wall-envelope allocation')
    expected = dict(physical_core=CORE, rd=RD, wall=WALL, source_capture=SOURCE_CAPTURE,
        capture=5., outer_vessel_probability=.5, inner_uniform_probability=.4,
        cloud_replicates=2, lambda_ratio=16.)
    require(all(value[k] == v for k,v in expected.items()), 'Changed allocation physical/proposal law')
    require([r['region'] for r in value['regions']] == list(REGIONS)
        and 0 < value['joint_point_gate_failure_bound'] < .05,
        'Changed five-region point-error design')
    return value


def passed_receipt(path, expected_sha=None):
    path = Path(path).resolve()
    require(expected_sha is None or sha(path) == expected_sha, 'Validation receipt hash differs')
    result = read(path)
    require(result['complete'] is True and result['passed'] is True
        and result.get('error') is None, 'Validation is not complete and passed')
    if 'jobs' in result:
        require(result['jobs'] and all(j['returncode'] == 0 and j['child_drained'] is True
            for j in result['jobs']), 'Validation children were not drained successfully')
    else:
        require(result['returncode'] == 0 and result['child_drained'] is True,
            'Validation child was not drained successfully')
    return result


def jobs_for(root, python):
    root = Path(root).resolve(); common, inputs = root/'common', root/'inputs'
    jobs = []
    for zi,z in enumerate(ACTIVITIES):
        draws = DRAWS_BY_ACTIVITY[str(z)]
        for population in range(POPULATIONS):
            identity = f'z{z:g}-p{population}'; destination = root/'populations'/identity
            seed = 6900410001+1000*zi+population
            argv = [str(common/'basin-normalizer'), '--config', str(inputs/'config.json'),
                '--model', str(inputs/'model.json'), '--latent-region', str(inputs/'region.json'),
                '--latent-guide', str(inputs/'class-guide.json'), '--out', str(destination),
                '--samples', str(draws), '--seed', str(seed), '--cloud-replicates', '2',
                '--covariance-scale', '1', '--activity', str(z), '--uniform-probability', '.4',
                '--wall-radius', str(WALL), '--wall-center', *map(str,CENTER), '--wall-uniform-envelope']
            audit = root/'audits'/(identity+'.json')
            jobs.append(dict(id=identity, activity=z, population=population, samples=draws, seed=seed,
                argv=argv, directory=str(destination), audit_argv=[python, '-B',
                    str(common/'physical_native_class_line_vessel.py'), '--directory', str(destination),
                    '--out', str(audit), '--synthetic'], audit=str(audit),
                expected_outputs=['manifest.json','config.json','summary.json','samples.jsonl','attempts.jsonl',
                    'provenance/compiled-native.json','provenance/source-bundle.json'],
                producer_terminal=dict(path=str(destination/'summary.json'), success_contract='complete'),
                audit_terminal=dict(path=str(audit), success_contract='complete')))
    require(len(jobs) == 8 and sum(j['samples'] for j in jobs) == TOTAL
        and len({j['seed'] for j in jobs}) == 8, 'Changed fixed wall-envelope allocation')
    return jobs


def analytic_reference():
    return dict(schema='full-wall-sphere-analytic-reference-v1', center=CENTER, core_radius=CORE,
        exclusion_radius=CORE+RD, wall_radius=WALL, source_capture=SOURCE_CAPTURE,
        radial_regions={k:list(v) for k,v in REGIONS.items()}, activities=list(ACTIVITIES), references=references(),
        measure='d3t times normalized SO(3) Haar; the full orientation integral is one',
        geometry='Four coincident atomic members form one union sphere, with no multiplicity factor.',
        formula='Qz(A)=4*pi*integral_A r^2*exp[z*pi*(3.2+r)*(1.6-r)^2/12]dr below1.6; overlap is zero above1.6.',
        Q0_formula='4*pi*(upper^3-lower^3)/3',
        partitions='Contact/unbound and inside/outside source each exhaust the full physical radial shell. R4 is descriptive, not a target restriction.')


def check_input_copies(inputs):
    inputs = Path(inputs)
    for name in original.CHECKED:
        require(sha(inputs/('source-'+name)) == original.CHECKED[name], 'Original fixture archive changed '+name)
    for name in ('shape.json','model.json','region.json','compiled-native.json'):
        require(sha(inputs/name) == original.CHECKED[name], 'Physical input changed '+name)
    expected = read(inputs/'source-config.json'); expected['shape'] = str(inputs/'shape.json')
    require(read(inputs/'config.json') == expected, 'Physical config changed beyond shape path')
    expected = read(inputs/'source-class-guide.json')
    expected['compiled_native']['path'] = str(inputs/'compiled-native.json')
    require(read(inputs/'class-guide.json') == expected, 'Class guide changed beyond native path')


def prepare(binary, bundle, out, build_receipt):
    binary, bundle, out, build_receipt = (Path(p).resolve() for p in (binary,bundle,out,build_receipt))
    require(not out.exists(), 'Fresh inert wall-envelope preparation directory required')
    require(sha(ALLOCATION) == ALLOCATION_SHA, 'Prospective allocation hash differs')
    allocation = validate_allocation(read(ALLOCATION))
    allocation_source = ALLOCATION.with_name('bounds.py')
    require(sha(allocation_source) == allocation['source_sha256'], 'Allocation derivation source differs')
    cli = passed_receipt(CLI/'validation.json', CLI_RECEIPT_SHA)
    audit = passed_receipt(AUDIT_RECEIPT, AUDIT_RECEIPT_SHA)
    release = passed_receipt(build_receipt)
    for key,path in [('binary',binary),('bundle',bundle)]:
        require(release['artifacts'][key] == dict(path=str(path),sha256=sha(path)), 'Release artifact identity differs')
    source = bind_bundle(binary,bundle)
    checked_bundle = CLI/'class-hard/provenance/source-bundle.json'
    require(read(checked_bundle) == source, 'Release differs from passed schema-8 CLI Rust closure')
    for name,entry in source['files'].items():
        # The validated build harnesses enumerate Rust/Cargo/build sources, not
        # this documentation file. Exact whole-bundle equality and bind_bundle's
        # text hashes already bind it; do not amend the historical receipts.
        if name == 'vendor/README.md':
            require(str(ROOT/name) not in release['source_sha256']
                and str(ROOT/name) not in cli['source_sha256'], 'Unexpected vendor documentation receipt entry')
            continue
        require(release['source_sha256'].get(str(ROOT/name)) == entry['sha256']
            and cli['source_sha256'].get(str(ROOT/name)) == entry['sha256'], 'Rust receipt source differs '+name)
    fixture_receipt = original.SOURCE.parent/'validation.json'
    passed_receipt(fixture_receipt, original.VALIDATION_SHA)
    require(read(original.SOURCE/'reference.json')['passed'] is True, 'Original physical fixture failed')
    source_bindings = {str(original.SOURCE/name):digest for name,digest in original.CHECKED.items()}
    source_bindings[str(fixture_receipt)] = original.VALIDATION_SHA
    for path in (ALLOCATION,allocation_source,CLI/'validation.json',AUDIT_RECEIPT,build_receipt,binary,bundle,checked_bundle):
        source_bindings[str(path)] = sha(path)
    # Only the audit's actual import closure must match that saved-row validation.
    # This new preparation/analysis code has its own validation receipt.
    auditor = Path(__file__).with_name('physical_native_class_line_vessel.py')
    for path in local_sources(auditor).values():
        require(audit['source_sha256'].get(str(path)) == sha(path), 'Saved-row audit implementation changed '+str(path))
    closure = {}
    for entry in (__file__,Path(__file__).with_name('analyze_wall_envelope_sphere.py'),auditor,
                  Path(__file__).with_name('run_native_class_physical_campaign.py')):
        closure.update(local_sources(entry))
    for path in closure.values(): source_bindings[str(path)] = sha(path)
    for path,digest in source_bindings.items(): require(sha(path) == digest, 'Frozen source changed '+path)
    out.mkdir(parents=True); common,inputs = out/'common',out/'inputs'; common.mkdir(); inputs.mkdir()
    receipts = common/'receipts'; receipts.mkdir()
    for name,path in closure.items(): shutil.copy2(path,common/name)
    shutil.copy2(binary,common/'basin-normalizer'); shutil.copy2(bundle,common/'source-bundle.json')
    for label,path in [('release',build_receipt),('schema8-cli',CLI/'validation.json'),
                       ('saved-row-audit',AUDIT_RECEIPT),('original-fixture',fixture_receipt)]:
        shutil.copy2(path,receipts/(label+'.json'))
    for name,entry in source['files'].items():
        path = common/'rust-source'/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(entry['text'].encode()); require(sha(path) == entry['sha256'], 'Rust archive changed')
    for name in original.CHECKED: shutil.copy2(original.SOURCE/name,inputs/('source-'+name))
    for name in ('shape.json','model.json','region.json','compiled-native.json'):
        shutil.copy2(original.SOURCE/name,inputs/name)
    config = read(original.SOURCE/'config.json'); config['shape'] = str(inputs/'shape.json'); write(inputs/'config.json',config)
    guide = read(original.SOURCE/'class-guide.json'); guide['compiled_native']['path'] = str(inputs/'compiled-native.json')
    write(inputs/'class-guide.json',guide); check_input_copies(inputs)
    shutil.copy2(ALLOCATION,out/'allocation.json'); shutil.copy2(allocation_source,common/'allocation-bounds.py')
    write(out/'analytic-reference.json',analytic_reference())
    for name in ('populations','audits'): (out/name).mkdir()
    plan = dict(schema=SCHEMA,root=str(out),preparation_only=True,launched=False,
        total_attempts=TOTAL,maximum_clouds=MAXIMUM_CLOUDS,populations_per_activity=POPULATIONS,
        samples_per_population_by_activity=DRAWS_BY_ACTIVITY,activities=list(ACTIVITIES),
        jobs=jobs_for(out,sys.executable),criteria=CRITERIA,maximum_workers=1,threads=1,
        retries=0,replacements=0,extensions=0,resource_limits=RESOURCE_LIMITS,
        python=sys.executable,python_sha256=sha(Path(sys.executable).resolve()),
        executable=dict(path=str(common/'basin-normalizer'),sha256=sha(common/'basin-normalizer')),
        source_bundle=dict(path=str(common/'source-bundle.json'),sha256=sha(common/'source-bundle.json')),
        source_bindings=source_bindings,archived_source_sha256={name:sha(common/name) for name in closure},
        physical=PHYSICAL,proposal=PROPOSAL,
        allocation=dict(path=str(out/'allocation.json'),sha256=ALLOCATION_SHA,
            joint_point_gate_failure_bound=allocation['joint_point_gate_failure_bound']),
        analytic_reference=dict(path=str(out/'analytic-reference.json'),sha256=sha(out/'analytic-reference.json')),
        analysis_contract='Seventeen sequential stages: eight producer/audit pairs, then authenticated whole-allocation analysis. Every attempted draw is retained; no source/R4 conditioning.',
        statistical_contract='Same four-SE and 0.1-log checks as original reference; four independent equal-sized populations within each activity. Unequal activity allocations are never pooled. Allocation tail bound covers only point-log gates.',
        scope='New optional proposal law on unchanged analytic sphere target. Completed cube reference remains failed and separate; no imported rows, extensions, protein draws or assembly inference.')
    for path,digest in source_bindings.items(): require(sha(path) == digest, 'Input changed during preparation '+path)
    plan['files_sha256'] = {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'plan.json',plan)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary','bundle','out','build-receipt'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(); result = prepare(args.binary,args.bundle,args.out,args.build_receipt)
    import json
    print(json.dumps(dict(prepared=True,launched=False,jobs=len(result['jobs']),total_attempts=TOTAL)))
