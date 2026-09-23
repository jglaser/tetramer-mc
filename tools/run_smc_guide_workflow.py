#!/usr/bin/env python3
"""Run one frozen SMC-guide pilot, audits, then first-pass classification in order."""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import time

NAME = 'run_smc_guide_workflow.py'
SCHEMA = 'smc-guide-pilot-workflow-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(campaign, expected):
    require(sha(campaign / 'protocol.json') == expected, 'Protocol checksum changed')
    protocol = read(campaign / 'protocol.json'); common = campaign / 'common'
    sources = protocol['python_sources']
    require({NAME, 'run_smc_guide_pilot.py', 'analyze_smc_guide_pilot.py'} <= set(sources),
        'Missing frozen workflow source closure')
    for name, digest in sources.items():
        require(Path(name).name == name and name.endswith('.py'), 'Unsafe frozen source path')
        require(sha(common / name) == digest, 'Frozen Python source changed: ' + name)
    require(Path(__file__).resolve() == common / NAME and sha(__file__) == sources[NAME],
        'Use the exact archived workflow driver')
    return protocol


def load_archived(campaign, expected):
    protocol = verify_sources(campaign, expected); common = campaign / 'common'
    for name in protocol['python_sources']:
        loaded = sys.modules.get(Path(name).stem)
        if loaded is not None:
            require(Path(getattr(loaded, '__file__', '')).resolve() == common / name,
                'A nonarchived module is already loaded: ' + name)
    sys.dont_write_bytecode = True  # Frozen source trees must never gain bytecode files.
    sys.path.insert(0, str(common))
    runner = importlib.import_module('run_smc_guide_pilot')
    analyzer = importlib.import_module('analyze_smc_guide_pilot')
    for module in (runner, analyzer):
        require(Path(module.__file__).resolve().parent == common, 'Nonarchived workflow import')
    require(runner.runtime() == protocol['runtime'], 'Frozen audit runtime changed')
    require(runner.preflight(campaign) == protocol, 'Pilot preflight protocol differs')
    return runner, analyzer, protocol


def run(campaign, expected):
    campaign = Path(campaign)
    require(campaign.is_absolute(), 'Campaign path must be absolute')
    campaign = campaign.resolve()
    require(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected),
        'Expected protocol checksum must be lowercase SHA256')
    require(not (campaign / 'status.json').exists() and not (campaign / 'comparison').exists(),
        'Existing physical or classification output; no retries')
    runner, analyzer, protocol = load_archived(campaign, expected)
    path = campaign / 'workflow-status.json'
    state = dict(schema=SCHEMA, complete=False, phase='claimed', protocol_sha256=expected,
        pid=os.getpid(), process_birth=runner.process_token(os.getpid()), started=time.time(),
        workflow_sha256=protocol['python_sources'][NAME], analysis_workers=4)
    with path.open('x') as stream:
        stream.write(json.dumps(state, indent=2, allow_nan=False) + '\n')
    def snapshot():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    def interrupted(signum, frame):
        raise InterruptedError('Workflow received signal ' + str(signum))
    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        state['phase'] = 'physical_and_audits'; snapshot()
        physical = runner.run(campaign)
        require(physical['complete'] is True and physical['phase'] == 'complete',
            'Physical pilot and audits did not complete')
        physical_sha = sha(campaign / 'status.json')
        verify_sources(campaign, expected)
        state.update(phase='first_pass_classification', physical_status_sha256=physical_sha); snapshot()
        result = analyzer.analyze(campaign, campaign / 'comparison', workers=4)
        terminal = read(campaign / 'comparison/status.json')
        require(result['complete'] is True and terminal['complete'] is True and terminal['phase'] == 'complete',
            'Classification did not complete')
        require(result['protocol_sha256'] == expected and result['status_sha256'] == physical_sha
            and sha(campaign / 'status.json') == physical_sha, 'Physical terminal status changed during classification')
        verify_sources(campaign, expected)
        state.update(complete=True, phase='complete', finished=time.time(),
            comparison_sha256=sha(campaign / 'comparison/analysis.json')); snapshot()
    except BaseException as error:
        state.update(complete=False, phase=state['phase'] + '_failed', exception=repr(error), finished=time.time())
        snapshot()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha256', required=True)
    args = parser.parse_args()
    run(args.campaign, args.expected_protocol_sha256)
