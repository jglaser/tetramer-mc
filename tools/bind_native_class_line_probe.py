#!/usr/bin/env python3
"""Bind validated code to the already frozen proposal diagnostic; never launch."""
import argparse
import json
from pathlib import Path
import shutil
from run_native_class_line_probe import read, require, sha, verify, write


def validate_evidence(executable, validations, repository):
    require(set(validations) == {'rust', 'python', 'cli'}, 'All three validation roles are required')
    require(len({Path(p).resolve() for p in validations.values()}) == 3, 'Validation roles require distinct receipts')
    require(len({sha(p) for p in validations.values()}) == 3, 'Validation receipts must have distinct content')
    receipts = {}; evidence = {}
    executable_hash = sha(executable)
    for role, path in validations.items():
        path = Path(path).absolute(); data = read(path)
        require(data.get('complete') is True and data.get('passed') is True, 'Validation did not pass: '+str(path))
        if role in ('rust', 'cli'):
            require(data['executable_sha256'] == executable_hash, 'Validation used a different executable: '+str(path))
        receipts[str(path)] = sha(path)
        evidence[role] = data
    rust = evidence['rust']
    require(rust['source_closure_before'] == rust['source_closure_after'] and
            rust['test_count'] > 0 and rust['exit_code'] == 0, 'Rust validation source closure changed')
    for name, digest in rust['source_closure_after'].items():
        require(sha(repository/name) == digest, 'Rust source changed after validation: '+name)
    reference_sources = evidence['cli']['reference_source_sha256']
    required_modules = {'native_class_line_reference.py', 'hard_free_line_reference.py',
                        'analyze_contact_line_audit.py', 'native_contact_regions.py'}
    require(required_modules.issubset({Path(p).name for p in reference_sources}), 'Incomplete independent reference closure')
    for path, digest in reference_sources.items():
        require(Path(path).is_absolute() and sha(path) == digest, 'Reference source changed after CLI validation: '+path)
    for path, digest in evidence['python']['source_sha256'].items():
        require(Path(path).is_absolute() and sha(path) == digest, 'Python source changed after tests: '+path)
    return receipts


def bind(root, executable, rust_validation, python_validation, cli_validation, repository):
    root, executable, repository = map(lambda p: Path(p).absolute(), (root, executable, repository))
    require(not (root/'execution-plan.json').exists() and not (root/'execution-code').exists(), 'Already bound')
    frozen = read(root/'freeze.json')
    for name, digest in frozen['files'].items(): require(sha(root/name) == digest, 'Changed frozen input: '+name)
    validations = dict(rust=rust_validation, python=python_validation, cli=cli_validation)
    receipts = validate_evidence(executable, validations, repository)
    output = root/'execution-code'; (output/'tools').mkdir(parents=True)
    shutil.copy2(executable, output/'contact-line-guide-audit')
    # Snapshot the Python module namespace; imports cannot silently follow later
    # working-tree edits. The user's unrelated analysis scripts are excluded.
    exclude = {'analyze_growth_moves.py', 'analyze_growth_native.py', 'replay_growth_native_events.py'}
    for path in sorted((repository/'tools').glob('*.py')):
        if path.name not in exclude: shutil.copy2(path, output/'tools'/path.name)
    native_source = repository/'runs/contact-confirmation-campaign-20260922/common/reference-package/native-region'
    native = output/'native-definition'
    shutil.copytree(native_source, native, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    compiled = read(root/'common/compiled-native.json'); definition = read(native/'definition.json')
    require(sha(native/'definition.json') == compiled['source_definition_sha256'], 'Original native definition differs')
    require(definition['input_sha256'] == compiled['source_input_sha256'], 'Original native input inventory differs')
    for name, digest in definition['input_sha256'].items(): require(sha(native/'inputs'/name) == digest, 'Native input changed')
    python = '/home/xvg/protein-nucleation/.venv/bin/python'
    allocation = read(root/'allocation.json'); queries = []
    source_jobs = allocation['jobs']+[dict(id='saved', arm='class', samples=0, seed=0)]
    for item in source_jobs:
        destination = root/'queries'/item['id']; count = 40 if item['id'] == 'saved' else 0
        argv = [str(output/'contact-line-guide-audit'), '--config', str(root/'common/config.json'),
            '--region', str(root/'common/region.json'), '--importance-guide', str(root/'guides'/f'{item["arm"]}.json'),
            '--out', str(destination), '--samples', str(item['samples']), '--seed', str(item['seed'])]
        if count: argv += ['--probes', str(root/'probes.jsonl')]
        queries.append(dict(item, phase='query', probes=count, argv=argv,
            terminal=str(destination/'summary.json'), cpu_limit_seconds=1800, wall_limit_seconds=3600))
    audits = []
    (root/'audits').mkdir()
    for query in queries:
        ident = query['id']; destination = root/'audits'/f'{ident}.json'
        audits.append(dict(id=ident+'-audit', query_id=ident, phase='audit',
            argv=[python, str(output/'tools/native_class_line_reference.py'), str(root/'queries'/ident),
                '--definition', str(native/'definition.json'), '--output', str(destination),
                '--journal', str(root/'audits'/f'{ident}.journal.jsonl')],
            terminal=str(destination), cpu_limit_seconds=7200, wall_limit_seconds=14400))
    files = {str(root/name): digest for name, digest in frozen['files'].items()}
    files[str(root/'freeze.json')] = sha(root/'freeze.json'); files.update(receipts)
    files.update({str(p): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    files[python] = sha(python)
    plan = dict(schema='native-class-line-reviewed-execution-v1', ready=True, maximum_workers=1,
        physical_clouds=0, repository=str(repository), files=files, jobs=queries+audits,
        validation_receipts=receipts, validation_roles={k:str(Path(v).absolute()) for k,v in validations.items()},
        allocation_sha256=sha(root/'allocation.json'),
        initial_freeze_sha256=sha(root/'freeze.json'), executable_sha256=sha(executable),
        source_note='Original preparation remains immutable and execution_ready=false there; this separately hashed binding authorizes only its fixed diagnostic allocation.',
        resource_note='Unpruned independent geometry audits may be slow; no retry or extension if their fixed limits fail.',
        cpu_comparison_note='H-only new-schema control constructs full class geometry for audit. Timings are not a speedup against the optimized legacy H guide.')
    verify(root, plan)
    write(root/'execution-plan.json', plan)
    return dict(complete=True, bound=True, execution_plan_sha256=sha(root/'execution-plan.json'),
                executable_sha256=sha(executable), jobs=18, new_Poisson_clouds=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    for role in ('rust', 'python', 'cli'):
        parser.add_argument('--'+role+'-validation', type=Path, required=True)
    parser.add_argument('--repository', type=Path, default=Path(__file__).absolute().parents[1])
    args = parser.parse_args()
    print(json.dumps(bind(args.root, args.executable, args.rust_validation,
                         args.python_validation, args.cli_validation, args.repository), indent=2))
