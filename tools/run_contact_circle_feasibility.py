#!/usr/bin/env python3
"""Freeze or execute the two sequential, geometry-only saved-circle audits."""
import argparse
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

from diagnose_fitted_kernel_shear import read, sha, require, write_new
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_confirmation import worker_environment, runtime
from run_contact_tail_pilot import verify_frozen
from run_full_vessel_comparison import execute_group
from run_mobile_posterior_pilot import verify_bundle

CASES_SHA = 'dc4743a8dc06ea51f86e508cc1a3a631371a844ff2c50c28d3983b5c0236ca85'


def make_jobs(out, common, preparation):
    jobs = [dict(id='rust', command=[str(common/'contact-circle-feasibility'),
        '--cases', str(preparation/'cases.json'), '--expected-cases-sha256', CASES_SHA,
        '--out', str(out/'rust')]), dict(id='python', command=[sys.executable, '-B',
        str(common/'contact_circle_reference.py'), '--shape', str(preparation/'shape.json'),
        '--config', str(preparation/'config.json'), '--cases', str(preparation/'cases.json'),
        '--out', str(out/'python.json')])]
    for j in jobs:
        j.update(status='pending', kind='geometry', log=str(out/(j['id']+'.log')),
            directory=str(out/('rust' if j['id']=='rust' else 'python.json')))
    return jobs


def freeze(out, preparation, binary, bundle, reference, repository):
    out, preparation, binary, bundle, reference, repository = map(
        lambda x: Path(x).resolve(), (out, preparation, binary, bundle, reference, repository))
    require(not out.exists(), 'Fresh execution directory required')
    verify_frozen(preparation)
    require(sha(preparation/'cases.json') == CASES_SHA, 'Frozen circle allocation changed')
    source, _ = verify_bundle(binary, bundle, repository)
    sources = local_dependencies([Path(__file__)])
    refs = ['results/circle-geometry-validation-20261001/validation.json',
        'results/circle-geometry-validation-20261001/source-receipt.json',
        'results/circle-geometry-validation-20261001/final-test-output.log',
        'tools/test_contact_circle_controller.py',
        'results/contact-circle-reference-validation-20261001/validation.json']
    require(all((repository/name).is_file() for name in refs), 'Missing reviewed reference')
    validation = read(repository/refs[-1])
    require(validation['complete'] and validation['unittest']['passed'] and
        validation['source_sha256']['tools/contact_circle_reference.py'] == sha(reference),
        'Independent reference differs from validation')
    require(read(repository/refs[0])['returncode'] == 0, 'Rust tests did not pass')
    out.mkdir(); common = out/'common'; common.mkdir()
    shutil.copy2(binary, common/'contact-circle-feasibility')
    shutil.copy2(bundle, common/'source-bundle.json')
    shutil.copy2(reference, common/'contact_circle_reference.py')
    for name, path in sources.items(): shutil.copy2(path, common/name)
    receipts = []
    for i, name in enumerate(refs):
        path = repository/name
        require(path.is_file(), 'Missing reviewed reference: '+name)
        dest = common/f'reference-{i}-{path.name}'; shutil.copy2(path, dest)
        receipts.append(dict(original=str(path), sha256=sha(path), archive=dest.name))
    jobs = make_jobs(out, common, preparation)
    protocol = dict(schema='contact-circle-feasibility-controller-v1', cases=240,
        preparation=str(preparation), cases_sha256=CASES_SHA,
        binary_sha256=sha(binary), source_bundle_sha256=sha(bundle),
        source_hashes={name: data['sha256'] for name, data in source['files'].items()},
        independent_python_sha256=sha(reference), reference_receipts=receipts,
        controller_sha256=sha(__file__), python_sources={name:sha(path) for name,path in sources.items()},
        repository=str(repository), runtime=runtime(), jobs=jobs,
        maximum_CPU_workers=1, new_pose_draws=0, new_Poisson_clouds=0,
        scope='Deterministic audits of the same 240 saved circles, no retries or new sampling')
    write_new(out/'protocol.json', protocol)
    write_new(out/'freeze.json', dict(files={str(p.relative_to(out)):sha(p)
        for p in sorted(out.rglob('*')) if p.is_file()}))
    return sha(out/'protocol.json')


def run(out, expected):
    out = Path(out).resolve(); verify_frozen(out)
    require(sha(out/'protocol.json') == expected, 'Protocol changed')
    p = read(out/'protocol.json'); verify_frozen(Path(p['preparation']))
    require(runtime() == p['runtime'] and sys.flags.optimize == 0, 'Runtime changed')
    require(sha(__file__) == p['controller_sha256'] and
        {n:sha(v) for n,v in local_dependencies([Path(__file__)]).items()} == p['python_sources'],
        'Controller source/import closure changed')
    require(not (out/'status.json').exists(), 'No repeated launch')
    state = dict(complete=False, jobs=p['jobs'], protocol_sha256=expected, started=time.time())
    def snapshot():
        (out/'status.tmp').write_text(json.dumps(state, indent=2)+'\n')
        (out/'status.tmp').replace(out/'status.json')
    with (out/'launch-claim.json').open('x') as f: json.dump(dict(protocol_sha256=expected), f)
    snapshot()
    def interrupted(signum, frame):
        raise InterruptedError('Circle controller signal '+str(signum))
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        for job in state['jobs']:
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            execute_group([job], snapshot, 1, Path(p['repository']), worker_environment(),
                before_launch=lambda:verify_frozen(out))
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            job['child_CPU_seconds'] = after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
            snapshot()
        require(read(out/'rust/summary.json')['complete'] and read(out/'python.json')['complete'],
            'Incomplete geometry output')
        state.update(complete=True, finished=time.time(), output_sha256={name:sha(out/name) for name in
            ['rust/summary.json','rust/circles.jsonl','rust/attempts.jsonl','python.json']})
        snapshot()
    except BaseException as error:
        state.update(error=repr(error), complete=False, finished=time.time())
        for j in state['jobs']:
            if j['status']=='pending': j['status']='not_started'
        snapshot(); raise
    finally:
        signal.signal(signal.SIGTERM, previous)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    f = sub.add_parser('freeze')
    for name in ['out','preparation','binary','bundle','reference','repository']:
        f.add_argument('--'+name, type=Path, required=True)
    r = sub.add_parser('run'); r.add_argument('--out', type=Path, required=True)
    r.add_argument('--expected-protocol-sha256', required=True)
    a = parser.parse_args()
    if a.action=='freeze': print(freeze(a.out,a.preparation,a.binary,a.bundle,a.reference,a.repository))
    else: print(run(a.out,a.expected_protocol_sha256)['complete'])
