#!/usr/bin/env python3
"""Freeze 56 existing Rust rows for optional projected-reference validation.

Preparation reads JSON, bytes and runtime metadata only. It never constructs a
protein observer or evaluates geometry. Execution requires a separate reviewed
freeze, owns exactly one bounded child, and cannot resume or add rows.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

SOURCE_NAMES = ('prepare_native_class_projected_probe.py', 'run_native_class_projected_probe.py',
    'run_evolving_dimer_analysis.py', 'native_class_line_reference.py',
    'hard_free_line_reference.py', 'analyze_contact_line_audit.py', 'native_contact_regions.py')
PROVENANCE = dict(config='config_sha256', region='region_sha256',
    **{'importance-guide': 'guide_sha256', 'shape': 'shape_sha256',
       'source-bundle': 'source_bundle_sha256', 'compiled-native': 'compiled_native_sha256'})
SCHEMA = 'saved-native-class-projected-probe-v1'


def read(path): return json.loads(Path(path).read_text())
def digest(data): return hashlib.sha256(data).hexdigest()
def sha(path): return digest(Path(path).read_bytes())
def require(ok, message):
    if not ok: raise ValueError(message)
def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
def safe_relative(name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and name not in ('', '.'), 'Unsafe archive path')
    return p


def versions():
    # No random draws or geometric queries. Set threads before loading NumPy.
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    info = np.finfo(np.longdouble)
    return dict(python=sys.version, python_executable=str(Path(sys.executable).absolute()),
        python_sha256=sha(sys.executable), packages={n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
        longdouble=dict(nmant=int(info.nmant), maxexp=int(info.maxexp), eps=str(info.eps),
                        itemsize=np.dtype(np.longdouble).itemsize))


def prefix_bytes(path):
    """Read only started + 16 exact begin/complete pairs, never the live suffix."""
    lines = []
    with Path(path).open('rb') as stream:
        for _ in range(33):
            value = stream.readline()
            require(value.endswith(b'\n'), 'Unpruned prefix is not complete through ordinal 15')
            lines.append(value)
    return b''.join(lines)


def check_prefix(blob, fresh_rows):
    records = [json.loads(s) for s in blob.splitlines()]
    require(len(records) == 33, 'Wrong unpruned prefix length')
    start = records[0]
    require(start['state'] == 'started' and start['schema'] == 'native-class-line-independent-audit-journal-v1'
            and start['synthetic'] is False and start['samples'] == 128 and start['probes'] == 0,
            'Wrong unpruned audit prefix scope')
    receipts = []
    for ordinal, row in enumerate(fresh_rows):
        begin, done = records[1+2*ordinal:3+2*ordinal]
        require(begin == dict(state='begin', ordinal=ordinal, kind='fresh', id=ordinal),
                'Missing/reordered unpruned begin')
        require(done['state'] == 'complete' and done['ordinal'] == ordinal
                and done['row']['kind'] == 'fresh' and done['row']['id'] == ordinal,
                'Missing/reordered unpruned completion')
        require(math.isfinite(done['cpu_seconds']) and done['cpu_seconds'] >= 0
                and all(math.isfinite(v) and v >= 0 for v in done['maxima'].values()),
                'Invalid prior audit diagnostics')
        expected = row['log_proposal_density']
        require((expected is None and done['row']['log_density'] is None)
                or (expected is not None and done['row']['log_density'] is not None
                    and math.isclose(expected, done['row']['log_density'], rel_tol=2e-10, abs_tol=2e-8)),
                'Prior receipt does not match selected Rust row')
        receipts.append(done)
    return start, receipts


def read_query(root, label):
    directory = root/'queries'/label
    manifest, summary = read(directory/'manifest.json'), read(directory/'summary.json')
    require(manifest['schema'] == 'native-class-line-guide-audit-v1' and manifest['physical_jobs'] == 0
            and summary['complete'] is True and summary['manifest'] == manifest
            and not (directory/'failure.json').exists(), 'Query is incomplete or has wrong scope')
    files = {directory/'manifest.json': sha(directory/'manifest.json'),
             directory/'summary.json': sha(directory/'summary.json')}
    for name, key in PROVENANCE.items(): files[directory/'provenance'/(name+'.json')] = manifest[key]
    samples, probes = [], []
    for name, target in [('samples', samples), ('probes', probes)]:
        path = directory/(name+'.jsonl'); files[path] = summary[name+'_sha256']
        require(sha(path) == files[path], 'Changed Rust row file')
        target.extend(path.read_bytes().splitlines(keepends=True))
        require(len(target) == summary[name], 'Missing Rust query rows')
        if name == 'samples':
            require(len(target) == manifest['samples'], 'Fresh manifest count differs')
    files[directory/'attempts.jsonl'] = summary['attempts_sha256']
    attempts = read_jsonl(directory/'attempts.jsonl')
    rows = [json.loads(s) for s in samples+probes]
    require(attempts == [dict(ordinal=i, state='begin', kind=r['kind'], id=r['id'])
                        for i, r in enumerate(rows)], 'Rust attempt inventory differs')
    require([json.loads(s)['id'] for s in samples] == list(range(len(samples))), 'Fresh IDs differ')
    if manifest['probes_sha256'] is not None:
        path = directory/'provenance/probes.jsonl'; files[path] = manifest['probes_sha256']
        declarations = read_jsonl(path)
        require([(r['id'], r['latent']) for r in declarations] ==
                [(json.loads(s)['id'], json.loads(s)['latent']) for s in probes], 'Saved query declaration differs')
    else:
        require(not probes, 'Saved queries have no bound declarations')
    for path, expected in files.items(): require(sha(path) == expected, 'Changed query provenance: '+str(path))
    return dict(manifest=manifest, files=files, samples=samples, probes=probes)


def read_jsonl(path): return [json.loads(s) for s in Path(path).read_bytes().splitlines()]


def authority_bindings(base, queries, prefix_start):
    """Tie selected Rust outputs and the old reference to their reviewed plan."""
    plan = read(base/'execution-plan.json')
    require(plan['schema'] == 'native-class-line-reviewed-execution-v1' and plan['ready'] is True
            and plan['maximum_workers'] == 1 and plan['physical_clouds'] == 0,
            'Original execution authority is not ready or has changed scope')
    bindings = {}
    def bound(path, expected=None):
        path = Path(path).resolve()
        value = plan['files'].get(str(path))
        require(value is not None and value == sha(path) and (expected is None or value == expected),
                'Original execution plan does not bind '+str(path))
        require(path.is_relative_to(base), 'Selected authority file is outside original archive')
        bindings[path] = value
        return value
    executable = base/'execution-code/contact-line-guide-audit'
    bound(executable, plan['executable_sha256'])
    definition = base/'execution-code/native-definition/definition.json'
    for label, query in queries.items():
        selected = [j for j in plan['jobs'] if j.get('id') == label]
        require(len(selected) == 1 and selected[0]['phase'] == 'query', 'Missing/duplicate selected query authority')
        job, manifest = selected[0], query['manifest']
        arm, samples, probes = ('hard_free', 128, 0) if label == 'hard_free-r00' else ('class', 0, 40)
        require(job['arm'] == arm and job['samples'] == samples and job['probes'] == probes
                and len(query['samples']) == samples and len(query['probes']) == probes
                and manifest['seed'] == job['seed'] and manifest['executable_sha256'] == plan['executable_sha256'],
                'Selected query differs from declared allocation or executable')
        config, region, guide = base/'common/config.json', base/'common/region.json', base/'guides'/(arm+'.json')
        for path, key in [(config, 'config_sha256'), (region, 'region_sha256'), (guide, 'guide_sha256')]:
            bound(path, manifest[key])
        bound(definition, manifest['native_definition_sha256'])
        argv = [str(executable), '--config', str(config), '--region', str(region),
                '--importance-guide', str(guide), '--out', str(base/'queries'/label),
                '--samples', str(samples), '--seed', str(job['seed'])]
        if probes:
            declaration = base/'probes.jsonl'; bound(declaration, manifest['probes_sha256'])
            argv += ['--probes', str(declaration)]
        else:
            require(manifest['probes_sha256'] is None, 'Fresh authority unexpectedly declares probes')
        require(job['argv'] == argv and job['terminal'] == str(base/'queries'/label/'summary.json'),
                'Selected query command/terminal differs from authority')
    # Every non-output input in the earlier started record must come from the
    # reviewed source closure; output files are separately linked above.
    fresh_outputs = {str(p) for p in queries['hard_free-r00']['files']}
    for name, expected in prefix_start['input_sha256'].items():
        if name not in fresh_outputs: bound(name, expected)
    old_source = base/'execution-code/tools/native_class_line_reference.py'
    for name in ('native_class_line_reference.py', 'hard_free_line_reference.py',
                 'analyze_contact_line_audit.py', 'native_contact_regions.py'):
        path = base/'execution-code/tools'/name
        require(prefix_start['input_sha256'].get(str(path)) == bound(path),
                'Prior prefix lacks its frozen reference source')
    audits = [j for j in plan['jobs'] if j.get('query_id') == 'hard_free-r00' and j.get('phase') == 'audit']
    require(len(audits) == 1, 'Missing/duplicate prior audit authority')
    audit = audits[0]
    require(audit['argv'][1:] == [str(old_source), str(base/'queries/hard_free-r00'), '--definition', str(definition),
            '--output', str(base/'audits/hard_free-r00.json'), '--journal', str(base/'audits/hard_free-r00.journal.jsonl')]
            and plan['files'].get(audit['argv'][0]) == sha(audit['argv'][0]), 'Prior audit command/interpreter differs')
    return bindings


def setup_inventory(definition):
    """Metadata-only upper bounds for unchanged NativeContactRegions setup."""
    definition = Path(definition); data = read(definition); inputs = definition.parent/'inputs'
    classification_name = 'reference/results/native-neighbor-classes/classification.json'
    for name in (classification_name, 'native-pair-motifs.json'):
        require(data['input_sha256'].get(name) == sha(inputs/name), 'Setup catalogue input is not bound')
    classes = read(inputs/classification_name)['classes']
    require(len({v['label'] for v in classes}) == len(classes) and bool(classes), 'Duplicate/empty setup classes')
    motifs = read(inputs/'native-pair-motifs.json')['motifs']
    triples = sorted({(c['member_i'], c['member_j'], c['directed_class']) for m in motifs for c in m['member_contacts']})
    require(all(c[2] in {v['label'] for v in classes} for c in triples), 'Setup motif refers to unknown class')
    require(len(data['fixed_poses']) == 2, 'Setup needs the bound two-body scaffold')
    return dict(observer_instances=1, guide_instances=2,
        reference_contact_queries=[dict(label=v['label'], position=v['body_delta'], rotation=v['body_relative_rotation']) for v in classes],
        reference_contact_calls=len(classes), fixed_scaffold_classifier_calls=1,
        fixed_scaffold_poses=data['fixed_poses'], catalogue_member_triples=[list(t) for t in triples],
        maximum_scaffold_contact_calls=len(triples), maximum_total_contact_calls=len(classes)+len(triples),
        setup_scope='Original observer initialization only: directed reference contact patches and one fixed-scaffold classifier. '
                    'Bound member triples cap internal scaffold contact calls. These are separate from the 56 saved-row density checks.')


def check_setup_counts(inventory, counts):
    for key, expected in [('reference_contacts', inventory['reference_contact_calls']),
                          ('scaffold_classifier', inventory['fixed_scaffold_classifier_calls'])]:
        require(counts.get(key+'_started', 0) == counts.get(key+'_completed', 0) == expected,
                'Incomplete or changed observer setup '+key)
    scaffold = counts.get('scaffold_contacts_started', 0)
    require(scaffold == counts.get('scaffold_contacts_completed', 0)
            and scaffold <= inventory['maximum_scaffold_contact_calls']
            and scaffold+inventory['reference_contact_calls'] <= inventory['maximum_total_contact_calls'],
            'Changed observer scaffold contact count')


def prepare(root, base, repository):
    root, base, repository = [Path(p).resolve() for p in (root, base, repository)]
    require(not root.exists(), 'Fresh projected-probe directory required')
    fresh, saved = read_query(base, 'hard_free-r00'), read_query(base, 'saved')
    require(len(fresh['samples']) == 128 and not fresh['probes'] and len(saved['probes']) == 40
            and not saved['samples'], 'Fixed 128-fresh/40-saved source inventory differs')
    selected = fresh['samples'][:16]+saved['probes']
    require(len({json.loads(s)['id'] for s in saved['probes']}) == 40, 'Duplicate saved query IDs')
    fresh_rows = [json.loads(s) for s in selected[:16]]
    journal = base/'audits/hard_free-r00.journal.jsonl'
    prefix = prefix_bytes(journal); start, receipts = check_prefix(prefix, fresh_rows)
    for path, expected in start['input_sha256'].items():
        require(sha(path) == expected, 'Changed prior-audit input: '+path)
    for path, expected in fresh['files'].items():
        require(start['input_sha256'].get(str(path)) == expected, 'Prior audit is not bound to this Rust query')
    authority = authority_bindings(base, {'hard_free-r00': fresh, 'saved': saved}, start)
    definition = base/'execution-code/native-definition/definition.json'
    definition_data = read(definition)
    require(sha(definition) == fresh['manifest']['native_definition_sha256']
            == saved['manifest']['native_definition_sha256'], 'Changed native definition')
    original = {**fresh['files'], **saved['files'], **authority, definition: sha(definition),
                base/'execution-plan.json': sha(base/'execution-plan.json')}
    for name, expected in definition_data['input_sha256'].items():
        original[definition.parent/'inputs'/safe_relative(name)] = expected
    for path, expected in original.items(): require(sha(path) == expected, 'Changed bound input '+str(path))
    # Finish all inexpensive consistency checks before creating the archive.
    for name in SOURCE_NAMES: require((repository/'tools'/name).is_file(), 'Missing frozen source '+name)
    require(sha(repository/'tools/native_contact_regions.py') ==
            definition_data['input_sha256']['source/native_contact_regions.py'], 'Observer source differs')
    initialization = setup_inventory(definition)
    root.mkdir(parents=True)
    bindings = []
    def copy_file(source, target, expected):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(sha(source) == expected == sha(target), 'Input changed while archiving')
        bindings.append(dict(source=str(source), archive=str(target.relative_to(root)), sha256=expected))
    for path, expected in original.items():
        copy_file(path, root/'inputs'/path.relative_to(base), expected)
    # Archive the earlier reference modules cited by the prefix for provenance;
    # they are never imported or rerun in this optional comparison.
    for path, expected in start['input_sha256'].items():
        p = Path(path)
        if p not in original:
            target = root/'prior-audit-inputs'/p.relative_to(base)
            copy_file(p, target, expected)
    for name in SOURCE_NAMES:
        p = repository/'tools'/name; copy_file(p, root/'code'/name, sha(p))
    (root/'selected-rows.jsonl').write_bytes(b''.join(selected))
    (root/'prior-unpruned-prefix.jsonl').write_bytes(prefix)
    write(root/'prior-unpruned-receipts.json', receipts)
    inventory = [dict(ordinal=i, query='hard_free-r00' if i < 16 else 'saved',
        kind=json.loads(s)['kind'], id=json.loads(s)['id'], row_sha256=digest(s),
        prior_unpruned_correspondence='completed_per_row' if i < 16 else 'pending') for i, s in enumerate(selected)]
    runtime = versions()
    allocation = dict(schema=SCHEMA, rows=inventory, row_count=56, maximum_workers=1,
        new_poses=0, new_clouds=0, full_unpruned_reruns=0, retries=0,
        cpu_limit_seconds=3600, wall_limit_seconds=7200, address_space_limit_bytes=16*1024**3,
        original_base=str(base), live_journal=str(journal), journal_prefix_sha256=digest(prefix),
        definition='inputs/execution-code/native-definition/definition.json', runtime=runtime,
        input_bindings=bindings, fresh_prior_rows=16, saved_pending_rows=40,
        observer_setup=initialization,
        scope='Optional q/J/full interval and existing draw-trace correspondence to frozen Rust rows only. '
              'Hard-core intervals remain unpruned. Original observer setup is explicitly inventoried separately; '
              'no selected-row native point-classifier, bath, physical weights or new poses.',
        equivalence='Fresh rows: two independent implementations agree with the same Rust traces within existing tolerances; '
                    'prior journals do not store full expected intervals. Saved rows remain pending unpruned correspondence.',
        timing='New q/J/interval/draw work and prior full-audit per-row CPU have different scope; no speedup ratio.',
        full_independent_audit_complete=False, physical_campaign_gate_open=False)
    write(root/'allocation.json', allocation)
    require(prefix_bytes(journal) == prefix, 'Live unpruned prefix changed during freeze')
    files = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write(root/'freeze.json', dict(schema=SCHEMA, complete=True, metadata_only=True,
        geometry_queries=0, files=files, allocation_sha256=sha(root/'allocation.json')))
    return dict(complete=True, metadata_only=True, geometry_queries=0, rows=56,
                allocation_sha256=sha(root/'allocation.json'), freeze_sha256=sha(root/'freeze.json'))


def verify(root):
    root = Path(root).resolve(); frozen = read(root/'freeze.json'); plan = read(root/'allocation.json')
    require(frozen['schema'] == plan['schema'] == SCHEMA and frozen['complete'] is True
            and frozen['metadata_only'] is True and frozen['geometry_queries'] == 0,
            'Wrong projected archive schema/scope')
    require(sha(root/'allocation.json') == frozen['allocation_sha256'], 'Allocation changed')
    require(plan['row_count'] == len(plan['rows']) == 56 and plan['maximum_workers'] == 1
            and all(plan[k] == 0 for k in ('new_poses', 'new_clouds', 'full_unpruned_reruns', 'retries'))
            and plan['cpu_limit_seconds'] == 3600 and plan['wall_limit_seconds'] == 7200
            and plan['address_space_limit_bytes'] == 16*1024**3
            and plan['full_independent_audit_complete'] is False and plan['physical_campaign_gate_open'] is False,
            'Changed fixed allocation or limits')
    required = {'allocation.json', 'selected-rows.jsonl', 'prior-unpruned-prefix.jsonl',
                'prior-unpruned-receipts.json'} | {'code/'+n for n in SOURCE_NAMES}
    required |= {b['archive'] for b in plan['input_bindings']}
    require(set(frozen['files']) == required, 'Incomplete frozen file closure')
    for name, expected in frozen['files'].items():
        require(sha(root/safe_relative(name)) == expected, 'Changed archive '+name)
    for b in plan['input_bindings']:
        require(frozen['files'][b['archive']] == b['sha256'], 'Changed source-to-archive binding')
    require(plan['observer_setup'] == setup_inventory(root/plan['definition']), 'Observer setup inventory changed')
    raw = (root/'selected-rows.jsonl').read_bytes().splitlines(keepends=True)
    require(len(raw) == 56, 'Missing/repeated selected rows')
    expected = ((root/'inputs/queries/hard_free-r00/samples.jsonl').read_bytes().splitlines(keepends=True)[:16]
                +(root/'inputs/queries/saved/probes.jsonl').read_bytes().splitlines(keepends=True))
    require(raw == expected, 'Selected rows differ from the fixed archived Rust inventory')
    for i, (entry, value) in enumerate(zip(plan['rows'], raw)):
        row = json.loads(value)
        require(entry['ordinal'] == i and entry['row_sha256'] == digest(value)
                and entry['kind'] == row['kind'] and entry['id'] == row['id']
                and entry['query'] == ('hard_free-r00' if i < 16 else 'saved')
                and row['kind'] == ('fresh' if i < 16 else 'probe')
                and entry['prior_unpruned_correspondence'] == ('completed_per_row' if i < 16 else 'pending'),
                'Changed selected row identity/scope')
        if i < 16: require(row['id'] == i, 'Fresh prefix changed')
    require(len({r['id'] for r in plan['rows'][16:]}) == 40, 'Duplicate saved row')
    prefix = (root/'prior-unpruned-prefix.jsonl').read_bytes()
    require(digest(prefix) == plan['journal_prefix_sha256'] and prefix_bytes(plan['live_journal']) == prefix,
            'Unpruned prefix changed; appended suffix is intentionally unbound')
    check_prefix(prefix, [json.loads(v) for v in raw[:16]])
    require(plan['runtime'] == versions(), 'Python/runtime/longdouble environment changed')
    return plan


def run(root, review):
    import run_evolving_dimer_analysis as controller
    from run_evolving_dimer_analysis import owned_child, terminate_requested
    root = Path(root).resolve(); review = Path(review).resolve()
    plan = verify(root); freeze_hash, allocation_hash, review_hash = sha(root/'freeze.json'), sha(root/'allocation.json'), sha(review)
    checked = read(review)
    frozen = read(root/'freeze.json')
    require(sha(__file__) == frozen['files']['code/prepare_native_class_projected_probe.py']
            and sha(controller.__file__) == frozen['files']['code/run_evolving_dimer_analysis.py'],
            'Loaded launcher/controller differs from the archive')
    require(checked['complete'] is True and checked['passed'] is True
            and checked['freeze_sha256'] == freeze_hash and checked['allocation_sha256'] == allocation_hash,
            'Separate review does not approve this freeze')
    write(root/'execution-claim.json', dict(freeze_sha256=freeze_hash, allocation_sha256=allocation_hash,
        review=dict(path=str(review), sha256=review_hash), retries=0, pid=os.getpid(), started=time.time()))
    execution = root/'execution'; execution.mkdir()
    previous = signal.signal(signal.SIGTERM, terminate_requested)
    try:
        owned_child([plan['runtime']['python_executable'], '-B', str(root/'code/run_native_class_projected_probe.py'),
            '--root', str(root)], str(root), execution, 3600, 7200, plan['address_space_limit_bytes'])
        result = read(root/'result.json')
        require(result['observer_setup_inventory'] == plan['observer_setup'], 'Changed reported setup inventory')
        check_setup_counts(plan['observer_setup'], result['observer_setup_counts'])
        require(result['complete'] is True and result['passed'] is True and result['completed_rows'] == 56
                and result['counts']['density_queries'] == 56 and result['counts']['axis_geometries'] <= 168
                and result['saved_unpruned_correspondence_pending_rows'] == 40
                and result['full_independent_audit_complete'] is False and result['physical_campaign_gate_open'] is False,
                'Incomplete projected worker result')
        require([{k: row[k] for k in entry} for row, entry in zip(result['rows'], plan['rows'])] == plan['rows']
                and len(result['rows']) == 56, 'Worker completed a different row inventory')
        verify(root)
        require(sha(root/'freeze.json') == freeze_hash and sha(root/'allocation.json') == allocation_hash
                and sha(review) == review_hash, 'Execution binding changed')
        write(root/'completion.json', dict(complete=True, passed=True, completed_rows=56,
            result_sha256=sha(root/'result.json'), journal_sha256=sha(root/'journal.jsonl'),
            freeze_sha256=freeze_hash, review_sha256=review_hash, retries=0,
            full_independent_audit_complete=False, physical_campaign_gate_open=False))
    except BaseException as error:
        write(root/'failure.json', dict(complete=False, passed=False, error=repr(error),
            traceback=traceback.format_exc(), retries=0, partial_outputs_retained=True,
            freeze_sha256=freeze_hash, review_sha256=review_hash))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    if args.run:
        if args.review is None: parser.error('--review is required for execution')
        run(args.root, args.review)
    else:
        if args.base is None: parser.error('--base is required for metadata-only preparation')
        print(json.dumps(prepare(args.root, args.base, args.repository), indent=2))
