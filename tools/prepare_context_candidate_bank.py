#!/usr/bin/env python3
"""Extract every rho=0 global candidate using completed, hash-bound audits.

No geometry, new poses, or Poisson draws. A failed extraction retains its
prefix and has no complete inventory. Accepted/rejected status never selects
rows. Local slots are outside the independent proposal law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x') as dest:
        json.dump(value, dest, indent=2, allow_nan=False)
        dest.write('\n')


def pose_valid(pose):
    require(isinstance(pose, dict) and set(pose) == {'position', 'orientation'}, 'Missing physical pose')
    for key, size in (('position', 3), ('orientation', 4)):
        require(len(pose[key]) == size and all(type(x) in (int, float) and math.isfinite(x)
                for x in pose[key]), 'Malformed/nonfinite physical pose')
    require(abs(sum(x*x for x in pose['orientation'])-1) < 2e-10, 'Unnormalized quaternion')


def extract_rows(source, identity, expected_cycles=2304):
    """Read a previously authenticated complete journal, preserving slot order.

    expected_cycles is a small-fixture hook for tests. Production always uses
    2304. This is an extraction check, not a rerun of the completed MC audit.
    """
    events = 0
    candidates = 0
    pending = None
    completed = 0
    for line in source:
        event = json.loads(line)
        require(event.get('event_index') == events, 'Missing/reordered journal event')
        events += 1
        kind = event['kind']
        if kind == 'invocation_begun':
            require(events == 1 and event['method'] == 'posterior_involution'
                    and event['resume'] is None and event['certify'] is None,
                    'Bank must be a fresh rho=0 atlas invocation')
        if kind == 'attempt_decision' and event['slot'] == 4:
            require(pending is None, 'Missing complete event for preceding candidate')
            cycle = candidates+1
            require(cycle <= expected_cycles and event['cycle'] == cycle
                    and event['attempt_index'] == 5*(cycle-1)+4, 'Candidate inventory changed')
            step = event['step']
            pose_valid(step['proposed_pose'])
            proposal = step['proposal']
            branch = proposal['branch']
            require(branch in ('uniform', 'involution'), 'Non-independent proposal in bank')
            require(type(step['accepted']) is bool, 'Invalid acceptance provenance')
            status = step['status']
            require(status in ('wall_rejected', 'core_rejected', 'bath_rejected', 'accepted'),
                    'Fatal/null candidate must not become a zero')
            require(step['accepted'] == (status == 'accepted'), 'Changed accepted status')
            if status == 'wall_rejected':
                require(step['wall_valid'] is False and step['core_valid'] is None, 'Changed wall verdict')
            elif status == 'core_rejected':
                require(step['wall_valid'] is True and step['core_valid'] is False, 'Changed core verdict')
            else:
                require(step['wall_valid'] is True and step['core_valid'] is True, 'Changed valid verdict')
            saved_g = None
            if branch == 'involution':
                require(proposal['source_law'] == 'posterior' and proposal['identity'] is False,
                        'Changed learned source law')
                saved_g = proposal['full_new_gaussian_log_density']
                require(type(saved_g) in (int, float) and math.isfinite(saved_g), 'Invalid saved density')
            pending = dict(ordinal=candidates, cycle=cycle, slot=4,
                attempt_index=event['attempt_index'], event_index=event['event_index'], identity=identity,
                proposed_pose=step['proposed_pose'], branch=branch, status=status,
                wall_valid=step['wall_valid'], core_valid=step['core_valid'], saved_log_g=saved_g,
                production=cycle > 256, accepted=step['accepted'])
            candidates += 1
        elif kind == 'attempt_complete' and event['slot'] == 4:
            require(pending is not None, 'Complete candidate without decision')
            for key in ('cycle', 'slot', 'attempt_index', 'identity', 'proposed_pose',
                        'wall_valid', 'core_valid', 'accepted', 'production'):
                require(event[key] == pending[key], 'Decision/complete candidate differs: '+key)
            require(event['global'] is True and event['proposal']['branch'] == pending['branch'],
                    'Changed tested-slot law')
            yield pending
            completed += 1
            pending = None
    require(pending is None and candidates == completed == expected_cycles,
            'Incomplete candidate bank; no replacement or truncated denominator')
    require(events == 4+16*expected_cycles, 'Unexpected full journal length')


def success(receipt, terminal, digest):
    require(receipt['success'] is True and receipt['child_drained'] is True and receipt['returncode'] == 0
            and Path(receipt['terminal']['path']).resolve() == Path(terminal).resolve()
            and receipt['terminal']['sha256'] == digest, 'Missing completed/drained authority')


def source_reference(reports):
    require(len(reports) == 12, 'Require all twelve source-start reports, without selection')
    initial = reports[0]['initial_observation']
    require(all(r['initial_observation'] == initial for r in reports), 'Inconsistent initial source contacts')
    require(initial['neighbor_labels'] == [16, 217], 'Wrong source environment')
    tokens = [t for t in initial['patch_tokens'] if t[:2] == [77, 217]]
    require(len(tokens) == 16 and tokens == sorted(tokens) and len({tuple(t) for t in tokens}) == 16,
            'Wrong complete secondary source pattern')
    return dict(a_neighbors=[16, 217], b_neighbors=[16, 56], secondary_label=217,
        source_secondary_tokens=tokens, inclusion_boundaries=[0., .25, .5, .75, 1.])


def prepare(protocol_path, output):
    started = time.process_time()
    protocol = read(protocol_path)
    require(protocol['schema'] == 'context-candidate-bank-extraction-v1', 'Wrong extraction protocol')
    require(protocol['chains'] == 16 and protocol['records_per_chain'] == 2304
            and protocol['total_records'] == 36864, 'Changed fixed bank allocation')
    require(not output.exists(), 'Fresh preparation directory required')
    output.mkdir(parents=True)
    bindings = {str(protocol_path.resolve()): sha(protocol_path), str(Path(__file__).resolve()): sha(__file__)}
    verified = {}

    def bind(path, digest=None):
        key = str(Path(path).resolve())
        actual = verified.get(key)
        if actual is None:
            actual = sha(key)
            verified[key] = actual
        require(digest is None or actual == digest, 'Changed frozen input: '+key)
        require(key not in bindings or bindings[key] == actual, 'Inconsistent authority')
        bindings[key] = actual
        return read(key)

    try:
        for path, digest in protocol['input_sha256'].items():
            bind(path, digest)
        comparison = bind(protocol['comparison'], protocol['input_sha256'][protocol['comparison']])
        require(comparison['complete'] and comparison['passed']
                and comparison['schema'] == 'fixed-context-prior-pilot-comparison-v1', 'Incomplete old comparison')
        status = bind(protocol['comparison_status'])
        require(status['complete'] and status['passed'] and len(status['completed']) == 1,
                'Incomplete old comparison execution')
        success(status['completed'][0], protocol['comparison'], sha(protocol['comparison']))
        for path, digest in comparison['input_sha256'].items():
            bind(path, digest)
        inventory = bind(protocol['observer_inventory'])
        require(len(inventory) == 24 and len({r['id'] for r in inventory}) == 24, 'Old inventory incomplete')
        expected = {(s, i, a) for s in ('saved_body77', 'highest_original_prior_valid_neighbor_distinct_center')
                    for i in range(4) for a in ('local', 'original', 'context')}
        require({(r['start'], r['stream'], r['arm']) for r in inventory} == expected, 'Unexpected bank identities')
        source_reports = []
        selected = []
        for entry in inventory:
            root = Path(entry['physical_result'])
            report_path = Path(entry['result'])/'report.json'
            report = bind(report_path, comparison['input_sha256'][str(report_path)])
            require(report['complete'] and report['passed'] and report['new_poses'] == 0
                    and report['physical_draws'] == 0, 'Invalid reused observation')
            observer_receipt = bind(entry['execution_receipt'])
            success(observer_receipt, report_path, sha(report_path))
            physical_receipt = bind(entry['physical_execution_receipt'],
                report['input_sha256'][entry['physical_execution_receipt']])
            summary = bind(root/'summary.json', report['input_sha256'][str(root/'summary.json')])
            success(physical_receipt, root/'summary.json', sha(root/'summary.json'))
            cfg = bind(entry['config'], entry['config_sha256'])
            require(summary['identity'] == cfg['identity'] == report['identity']
                    and cfg['identity']['start'] == entry['start'] and cfg['identity']['arm'] == entry['arm']
                    and cfg['identity']['stream'] == entry['stream'] and cfg['seed'] == entry['seed'],
                    'Actual execution identity differs from inventory')
            require(summary['correlation'] == 0. and summary['completed_cycles'] == 2304
                    and summary['completed_attempts'] == 11520 and summary['bindings'] == report['bindings']
                    and summary['bindings']['config'] == entry['config_sha256'], 'Changed compiled invocation')
            require(cfg['uniform_probability'] == .5 and cfg['depletant_radius'] == 1.5
                    and cfg['reservoir_density'] == .035, 'Changed physical/proposal definition')
            for key in ('config', 'model', 'shape', 'fixed_context', 'source_state'):
                bind(root/'provenance'/(key+'.json'), summary['bindings'][key])
            bind(root/'provenance/source-bundle.json', summary['bindings']['source_bundle'])
            source = bind(root/'provenance/source_state.json')
            if entry['start'] == 'saved_body77':
                require(cfg['initial_pose'] == source['pose'], 'Source labels not from certified initial geometry')
                source_reports.append(report)
            if entry['arm'] == 'local':
                continue
            require(summary['method'] == 'posterior_involution', 'Non-atlas bank source')
            if entry['arm'] == 'context':
                bind(root/'provenance/prior.json', summary['bindings']['prior'])
            else:
                require('prior' not in summary['bindings'], 'Unexpected original-arm prior')
            selected.append((entry, report, summary, cfg))
        regions = source_reference(source_reports)
        write(output/'regions.json', dict(schema='context-candidate-bank-regions-v1', **regions,
            source_reference='All12 immutable saved_body77 initial observations; no production-token selection',
            interpretation='Retrospective coverage diagnostics; zero hits are not a physical mass bound'))
        write(output/'region-authority.json', dict(initial_observation=source_reports[0]['initial_observation'],
            comparison_sha256=sha(protocol['comparison']), source_start_reports=12,
            fixed_before_bank_extraction=True, physical_draws=0, geometry_queries=0))
        banks = []
        for entry, report, summary, cfg in selected:
            require(time.process_time()-started < protocol['cpu_limit_seconds'], 'Extraction CPU cap reached')
            run = Path(entry['physical_result'])
            journal = run/'events.jsonl'
            digest = report['input_sha256'][str(journal)]
            require(sha(journal) == digest, 'Reused journal changed after its audit')
            bindings[str(journal)] = digest
            dest = output/(entry['id']+'.jsonl')
            with journal.open() as source, dest.open('x') as stream:
                count = 0
                for row in extract_rows(source, cfg['identity']):
                    stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                    count += 1
            require(sha(journal) == digest, 'Journal changed during extraction')
            require(count == 2304, 'Wrong unconditional candidate denominator')
            banks.append(dict(**entry, bank=str(dest.resolve()), bank_sha256=sha(dest), records=count,
                journal=str(journal), journal_sha256=digest,
                model=str(run/'provenance/model.json'), model_sha256=summary['bindings']['model'],
                prior=str(run/'provenance/prior.json') if entry['arm'] == 'context' else None,
                prior_sha256=summary['bindings'].get('prior')))
        require(len(banks) == 16 and sum(r['records'] for r in banks) == 36864, 'Incomplete bank')
        grouped = {}
        for row in banks:
            grouped.setdefault((row['start'], row['stream']), []).append(row)
        require(len(grouped) == 8 and all(len(v) == 2 and len({r['seed'] for r in v}) == 1
                                        for v in grouped.values()), 'Lost paired populations')
        require(len({v[0]['seed'] for v in grouped.values()}) == 8, 'Repeated independent population seed')
        write(output/'inventory.json', banks)
        write(output/'summary.json', dict(schema='context-candidate-bank-extracted-v1', complete=True, passed=True,
            input_sha256=bindings, protocol_sha256=sha(protocol_path), regions_sha256=sha(output/'regions.json'),
            inventory_sha256=sha(output/'inventory.json'), chains=16, records=36864,
            warmup_records=4096, new_poses=0, physical_draws=0, geometry_queries=0,
            cpu_seconds=time.process_time()-started, initial_reference_reports=12,
            statistical_units='Eight distinct start/stream populations, paired across two proposal arms'))
    except BaseException as error:
        write(output/'failure.json', dict(complete=False, passed=False, error=repr(error),
            input_sha256=bindings, physical_draws=0, geometry_queries=0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.protocol, args.out)


if __name__ == '__main__':
    main()
