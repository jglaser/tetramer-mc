#!/usr/bin/env python3
"""Freeze one explicit format-correction continuation, retaining all original draws.

The --adopt mode only copies authenticated completed r00 bytes. It does not
parse sample/attempt streams or invoke a producer. The original failed root
and its frozen sources remain immutable; r01-r03 keep their original seeds.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sys

import prepare_native_class_support_pilot as base
import audit_native_class_support_pilot as worker
from analyze_mobile_native_pocket import local_sources
from native_class_physical_stage import Inputs, verify_live_authority

read, require, sha, write, bound = base.read, base.require, base.sha, base.write, base.bound
SCHEMA = 'native-class-support-format-recovery-v1'
ORIGINAL_PROTOCOL = 'bcfa66107a371f7db7aad3067f862ddac87d51febaaf8dfd03897387e648f982'
ORIGINAL_PLAN = '61613b40549091e46d5496a9e72559841f613753b34fc6b937419cd212e0be76'
ORIGINAL_SUMMARY = '6063a2e21f1515a82aa97a788f5c997bdb31a0844c9f259c16fcf834faf913a8'
ORIGINAL_FAILURE = '4b2f4f311b231bf9dad2d12bafef2cd7d79c6416c03efd4bb7cc76887bc2b855'
ORIGINAL_ALGEBRA_FAILURE = '648cfd1b7856bae27321a907116e33b02c8f201a9ecfc903c523c35599a850e8'
CORRECTED_WORKER = '9020e96dd3983f7c990f2270a5e9c80057b7f4ae1d7fd15b9a05981cb6bbb666'
POPULATION_FILES = frozenset(('manifest.json', 'summary.json', 'samples.jsonl', 'attempts.jsonl', 'probes.jsonl',
    'provenance/config.json', 'provenance/region.json', 'provenance/importance-guide.json',
    'provenance/shape.json', 'provenance/compiled-native.json', 'provenance/source-bundle.json'))


def original_driver(root, bindings):
    """Retain verify_plan's actual archived __file__ identity."""
    plan = read(bindings.bind(root/'execution-plan.json', ORIGINAL_PLAN))
    path = root/'code/run_native_class_physical_campaign.py'
    bindings.bind(path, plan['files'][str(path)])
    spec = importlib.util.spec_from_file_location('completed_original_support_driver', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, plan


def population_inventory(root, summary, manifest, bindings):
    """Hash exact full streams as opaque files; do not decode a scientific row."""
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == POPULATION_FILES, 'Original producer output inventory changed')
    require(summary['manifest'] == manifest and summary['complete'] is True
            and summary['samples'] == manifest['samples'] == 128 and summary['probes'] == 0
            and manifest['schema'] == base.PRODUCER_SCHEMA and manifest['physical_jobs'] == 0
            and manifest['probes_sha256'] is None, 'Original population is not complete proposal-only r00')
    expected = {name+'.jsonl': summary[name+'_sha256'] for name in ('samples', 'attempts', 'probes')}
    for name, field in [('config', 'config'), ('region', 'region'), ('importance-guide', 'guide'),
                        ('shape', 'shape'), ('compiled-native', 'compiled_native'), ('source-bundle', 'source_bundle')]:
        expected['provenance/'+name+'.json'] = manifest[field+'_sha256']
    result = {}
    for name in sorted(actual):
        path = root/name; require(not path.is_symlink(), 'Symlink in original population')
        bindings.bind(path, expected.get(name))
        result[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    require(result['probes.jsonl']['bytes'] == 0, 'Saved probes are not allowed')
    return result


def original_evidence(root, bindings):
    root = Path(root).resolve()
    protocol = read(bindings.bind(root/'protocol.json', ORIGINAL_PROTOCOL))
    driver, plan = original_driver(root, bindings)
    terminal = driver.completed_terminal(root, plan, 'r00-producer')
    require(terminal == dict(path=str(root/'queries/r00/summary.json'), sha256=ORIGINAL_SUMMARY), 'Wrong retained producer')
    for ordinal, identity in [(0, 'r00-producer'), (1, 'r00-algebra')]:
        folder = root/'execution/jobs'/f'{ordinal:03d}-{identity}'
        for name in ('attempt', 'process', 'exit'):
            bindings.bind(folder/(name+'.json'))
        if ordinal == 0: bindings.bind(folder/'success.json')
    bindings.bind(root/'execution/claim.json'); bindings.bind(root/'preparation.json')
    failure = read(bindings.bind(root/'execution/failure.json', ORIGINAL_FAILURE))
    algebra_failure = read(bindings.bind(root/'analysis/r00/algebra.failure.json', ORIGINAL_ALGEBRA_FAILURE))
    status = read(bindings.bind(root/'execution/status.json'))
    failed_exit = read(root/'execution/jobs/001-r00-algebra/exit.json')
    require(failure['plan_sha256'] == ORIGINAL_PLAN and failure['complete'] is False
            and failure['failed_job'] == dict(ordinal=1, id='r00-algebra', population='r00', phase='algebra')
            and failure['completed'] == [read(root/'execution/jobs/000-r00-producer/success.json')]
            and failure['unstarted'] == plan['jobs'][2:] and status['active'] is None and status['failure'] is not None
            and status['completed'] == failure['completed'] and status['unstarted'] == failure['unstarted']
            and failed_exit['child_drained'] is True and failed_exit['returncode'] == 1
            and not (root/'execution/summary.json').exists(), 'Original stop boundary changed')
    require(algebra_failure['phase'] == 'algebra' and algebra_failure['population'] == 'r00'
            and algebra_failure['active_id'] == 0 and algebra_failure['completed_ids'] == []
            and algebra_failure['counts'] == algebra_failure['setup_counts'] == {}
            and algebra_failure['error'] == 'Wrong proposal density/contact format', 'Different original failure')
    for population in protocol['populations'][1:]:
        require(not Path(population['directory']).exists(), 'A remaining producer already began')
    for job in plan['jobs'][2:]: require(not Path(job['terminal']['path']).exists(), 'A remaining stage already completed')
    summary = read(bindings.bind(terminal['path'], terminal['sha256']))
    manifest = read(bindings.bind(root/'queries/r00/manifest.json'))
    require(manifest['seed'] == protocol['populations'][0]['seed'], 'Retained seed differs')
    for key, field in [('config', 'config'), ('region', 'region'), ('shape', 'shape'), ('guide', 'guide'),
                       ('producer', 'executable'), ('producer_bundle', 'source_bundle'), ('compiled_native', 'compiled_native'),
                       ('definition', 'native_definition')]:
        require(manifest[field+'_sha256'] == protocol[key]['sha256'], 'Retained physical/proposal identity differs')
    inventory = population_inventory(root/'queries/r00', summary, manifest, bindings)
    for path, digest in protocol['files'].items(): bindings.bind(path, digest)
    return protocol, dict(schema=SCHEMA, original_root=str(root), original_protocol=bound(root/'protocol.json'),
        original_plan=bound(root/'execution-plan.json'), original_failure=bound(root/'execution/failure.json'),
        original_algebra_failure=bound(root/'analysis/r00/algebra.failure.json'), retained_summary=terminal,
        source_directory=str(root/'queries/r00'), retained_files=inventory,
        retained_attempts=128, new_draws=384, total_attempts=512, scientific_replacements=0,
        no_outcome_adaptation=True, allocation_changed=False,
        correction='Match genuine producer width_contacts=[[]], the empty contact list for its sole zero-width sentinel. '
                   'No proposal law, sample, label, density tolerance or geometry algorithm changes.',
        old_worker_sha256=protocol['source_sha256']['audit_native_class_support_pilot.py'],
        corrected_worker_sha256=CORRECTED_WORKER)


def source_paths():
    result = base.source_paths()
    for name, path in local_sources(__file__).items():
        require(name not in result or result[name] == path, 'Ambiguous recovery source')
        result[name] = path
    return result


def continuation_protocol(original, out, files, sources, recovery, validation):
    value = copy.deepcopy(original)
    value.update(root=str(out), code_directory=str(out/'code'), files=files,
        source_sha256=sources, recovery=recovery, validation=validation,
        launched=False, launch_review_complete=False, new_draws=384, retained_attempts=128)
    for population in value['populations']: population['directory'] = str(out/'queries'/population['id'])
    return value


def execution_plan(protocol, reference):
    plan = base.execution_plan(protocol, reference)
    first = plan['jobs'][0]
    require(first['id'] == 'r00-producer' and first['phase'] == 'producer', 'Wrong adoption slot')
    first['argv'] = [protocol['python'], '-B', str(Path(protocol['code_directory'])/Path(__file__).name),
        '--adopt', '--protocol', reference['path'], '--protocol-sha256', reference['sha256']]
    plan['executable_resolutions'] = {job['argv'][0]: str(Path(job['argv'][0]).resolve()) for job in plan['jobs']}
    for path in plan['executable_resolutions'].values(): plan['files'][path] = sha(path)
    plan['scope'] = 'Explicit format-correction continuation: adopt completed128 bytes, draw only remaining384 original seeds; no replacement or adaptive extension.'
    return plan


def prepare(original, out, validation_path):
    out = Path(out).resolve(); require(not out.exists(), 'Recovery preparation requires a fresh root')
    bindings = base.Bindings(); old, recovery = original_evidence(original, bindings)
    source = source_paths(); current = {str(path): sha(path) for path in source.values()}
    require(sha(source['audit_native_class_support_pilot.py']) == CORRECTED_WORKER, 'Different worker correction')
    for name, path in source.items():
        require(name not in old['source_sha256'] or name == 'audit_native_class_support_pilot.py'
                or sha(path) == old['source_sha256'][name], 'Unrelated analysis source changed: '+name)
        bindings.bind(path)
    validation_path = bindings.bind(validation_path); validation = read(validation_path)
    require(validation['complete'] is True and validation['passed'] is True and validation['returncode'] == 0
            and validation['source_before'] == validation['source_after']
            and all(validation['source_after'].get(path) == digest for path, digest in current.items()),
            'Passing focused validation of exact recovery closure required')
    require(old['runtime'] == base.runtime_identity(), 'Runtime changed since original freeze')
    require(shutil.disk_usage(out.parent).free >= old['storage_reserve_bytes'], 'Insufficient unchanged storage reserve')
    bindings.recheck(); out.mkdir(); (out/'code').mkdir(); (out/'queries').mkdir(); (out/'analysis').mkdir()
    write(out/'preparation-attempt.json', dict(schema=SCHEMA, launched=False, input_sha256=bindings.files))
    handlers = {}
    def interrupted(signum, frame): raise InterruptedError('Recovery preparation signal '+str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT): handlers[sig] = signal.signal(sig, interrupted)
        for name, path in source.items():
            destination = out/'code'/name; shutil.copy2(path, destination)
            require(sha(destination) == current[str(path)], 'Recovery source copy differs')
        shutil.copy2(validation_path, out/'validation.json')
        require(sha(out/'validation.json') == sha(validation_path), 'Recovery validation copy differs')
        for population in old['populations']: (out/'analysis'/population['id']).mkdir()
        write(out/'correction.json', recovery)
        # Admission authenticates working sources; execution uses only their
        # checked archive copies. Original lifecycle/output evidence stays bound.
        files = dict(old['files'])
        files.update({path:digest for path,digest in bindings.files.items() if path not in current})
        files.update({str(path): sha(path) for path in out.rglob('*') if path.is_file()})
        protocol = continuation_protocol(old, out, files, {name: current[str(path)] for name,path in source.items()},
                                         recovery, bound(out/'validation.json'))
        write(out/'protocol.json', protocol)
        plan = execution_plan(protocol, bound(out/'protocol.json'))
        bindings.recheck()
        for path,digest in files.items(): require(sha(path) == digest, 'Recovery frozen closure changed')
        write(out/'execution-plan.json', plan)
        receipt = dict(schema=SCHEMA, complete=True, launched=False, root=str(out), protocol=bound(out/'protocol.json'),
            execution_plan=bound(out/'execution-plan.json'), retained_attempts=128, new_draws=384,
            total_attempts=512, selected_reference_rows=64, new_Poisson_clouds=0, launch_review_complete=False)
        write(out/'preparation.json', receipt)
        return receipt
    except BaseException as error:
        write(out/'preparation-failure.json', dict(schema=SCHEMA, complete=False, launched=False,
              error_type=type(error).__name__, error=str(error), retries=0))
        raise
    finally:
        for sig, previous in handlers.items(): signal.signal(sig, previous)


def copy_population(source, destination, inventory, emit):
    source, destination = Path(source), Path(destination)
    require(not destination.exists(), 'Adoption requires a fresh population directory')
    require(set(inventory) == POPULATION_FILES, 'Adoption inventory changed')
    destination.mkdir()
    # Publish the original producer terminal last. A failed child is never admitted.
    ordered = sorted(set(inventory)-{'summary.json'})+['summary.json']
    for name in ordered:
        spec = inventory[name]; origin = source/name; target = destination/name
        require(origin.is_file() and not origin.is_symlink() and sha(origin) == spec['sha256']
                and origin.stat().st_size == spec['bytes'], 'Original retained file changed')
        emit(dict(state='copy_begin', member=name, sha256=spec['sha256'], bytes=spec['bytes']))
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(origin, target)
        require(sha(target) == spec['sha256'] and target.stat().st_size == spec['bytes'], 'Adopted bytes differ')
        emit(dict(state='copy_complete', member=name, sha256=spec['sha256'], bytes=spec['bytes']))


def adopt(protocol_path, digest):
    bindings = Inputs(); protocol, execution = worker.load_protocol(protocol_path, digest, bindings)
    verify_live_authority(protocol, execution, 'r00', 'producer')
    root = Path(protocol['root']); recovery = protocol['recovery']; folder = root/'analysis/r00'
    journal_path, receipt_path, failure_path = [folder/('adoption'+suffix) for suffix in ('.journal.jsonl', '.json', '.failure.json')]
    require(all(not p.exists() for p in (journal_path, receipt_path, failure_path)), 'Adoption is single-use')
    journal = journal_path.open('x'); handlers = {}
    def emit(value):
        journal.write(json.dumps(value, allow_nan=False)+'\n'); journal.flush(); os.fsync(journal.fileno())
    def interrupted(signum, frame): raise InterruptedError('Adoption signal '+str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU): handlers[sig] = signal.signal(sig, interrupted)
        emit(dict(state='started', protocol_sha256=digest, new_draws=0, retained_attempts=128))
        _, authenticated = original_evidence(recovery['original_root'], bindings)
        require(authenticated == recovery, 'Original recovery evidence changed')
        destination = Path(protocol['populations'][0]['directory'])
        copy_population(recovery['source_directory'], destination, recovery['retained_files'], emit)
        bindings.recheck(); verify_live_authority(protocol, execution, 'r00', 'producer')
        emit(dict(state='complete', retained_attempts=128, new_draws=0)); journal.close()
        result = dict(schema=SCHEMA, complete=True, passed=True, protocol_sha256=digest, new_draws=0,
            retained_attempts=128, retained_files=recovery['retained_files'], original_summary=recovery['retained_summary'],
            adopted_summary=bound(destination/'summary.json'), input_sha256=bindings.files,
            journal=bound(journal_path), sample_rows_parsed=0, retries=0)
        write(receipt_path, result)
        return result
    except BaseException as error:
        value = dict(schema=SCHEMA, complete=False, passed=False, protocol_sha256=digest,
            error_type=type(error).__name__, error=str(error), new_draws=0, retries=0)
        if not journal.closed: emit(dict(state='failed', **value)); journal.close()
        write(failure_path, value); raise
    finally:
        if not journal.closed: journal.close()
        for sig, previous in handlers.items(): signal.signal(sig, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt', action='store_true'); parser.add_argument('--protocol', type=Path)
    parser.add_argument('--protocol-sha256'); parser.add_argument('--original', type=Path)
    parser.add_argument('--out', type=Path); parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    if args.adopt:
        require(args.protocol is not None and args.protocol_sha256 is not None, 'Bound protocol required')
        result = adopt(args.protocol, args.protocol_sha256)
    else:
        require(all(x is not None for x in (args.original, args.out, args.validation)), 'Original root, fresh out and validation required')
        result = prepare(args.original, args.out, args.validation)
    print(json.dumps(result, indent=2))
