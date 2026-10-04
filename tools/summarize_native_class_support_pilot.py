#!/usr/bin/env python3
"""Reduce authenticated pilot receipts; never reopen proposals or query geometry."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import signal
import time

import audit_native_class_support_pilot as worker
from analyze_mobile_native_pocket import local_sources
from native_class_physical_stage import Inputs, predecessor, verify_live_authority
from run_native_class_physical_campaign import read, require, sha, write

SCHEMA = 'native-class-support-pilot-summary-v1'


def _finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Expected finite numeric diagnostic')
    return float(value)


def population_report(population, algebra, labels, geometry, component_inventory):
    """The small per-attempt receipts preserve all unconditional denominators."""
    size = population['samples']; require(size == 128, 'Wrong prospective population size')
    expected = list(range(size))
    for phase, receipt, ids in [('algebra', algebra, expected), ('labels', labels, expected),
                              ('geometry', geometry, population['selected_ids'])]:
        require(receipt['complete'] is True and receipt['passed'] is True, 'Failed '+phase+' receipt')
        require(receipt['schema'] == 'native-class-support-pilot-'+phase+'-v1'
                and receipt['phase'] == phase and receipt['population'] == population['id'], 'Wrong '+phase+' identity')
        require([row['id'] for row in receipt['rows']] == ids, 'Missing, repeated or reordered '+phase+' IDs')
    require(len(population['selected_ids']) == len(set(population['selected_ids'])) == 16, 'Wrong reference allocation')
    counts, orthants, chosen, fallbacks, branch_groups, cpu = (Counter() for _ in range(6))
    masses = defaultdict(list); hits = Counter(); selected_denominators = Counter()
    component_map = {entry['index']: entry for entry in component_inventory}
    require(set(component_map) == set(range(116)), 'Incomplete component inventory')
    critical = Counter()
    hashes = {row['id']: row['sample_record_sha256'] for row in algebra['rows']}
    require(all(row['sample_record_sha256'] == hashes[row['id']] for row in labels['rows']+geometry['rows']),
            'Audits did not bind the same exact row bytes')
    for algebra_row, label in zip(algebra['rows'], labels['rows']):
        valid, native, contact = (label[key] for key in ('valid', 'native', 'contact'))
        require(type(valid) is bool, 'Invalid validity flag')
        require((valid and type(native) is bool and type(contact) is bool)
                or (not valid and native is None and contact is None), 'Invalid point-label applicability')
        if not valid: native = contact = False
        kind = 'invalid_or_exterior' if not valid else 'native' if native else 'competing' if contact else 'unbound'
        counts[kind] += 1
        orthant = label['orthant']; require(type(orthant) is int and 0 <= orthant < 64, 'Invalid original orthant')
        orthants[f'{kind}:{orthant}'] += 1
        if valid and native and not contact: counts['native_without_contact_anomaly'] += 1
        if kind == 'competing' and orthant in (22, 62): critical[f'competing{orthant}'] += 1
        if kind == 'native' and orthant == 55: critical['native55'] += 1
        for key, value in algebra_row['producer_cpu_seconds'].items():
            value = _finite(value); require(value >= 0, 'Negative producer CPU'); cpu[key] += value
        selected = algebra_row['selected']
        if selected is None:
            chosen['uniform_or_unconditioned'] += 1; branch_groups['uniform_or_unconditioned'] += 1
            continue
        component = selected['component']; require(type(component) is int and component in component_map, 'Invalid component')
        channel = selected['channel']; require(type(channel) is int and 0 <= channel < 5, 'Invalid channel')
        info = component_map[component]
        groups = [f'channel:{channel}', f'bank:{info["bank"]}:channel:{channel}']
        if info['bank'] == 'added':
            groups.append(f'center:{info["training_id"]}:sigma:{info["latent_sigma"]}:channel:{channel}')
        chosen[str(channel)] += 1
        fallback = selected['fallback']; require(fallback in ('class', 'hard_free', 'unconditional'), 'Unknown fallback')
        empty = selected['empty_class']; require(type(empty) is bool, 'Missing empty-class distinction')
        class_mass = _finite(selected['class_mass']); hard_mass = _finite(selected['hard_mass'])
        effective_mass = _finite(selected['effective_mass'])
        require(all(0 <= v <= 1+1e-12 for v in (class_mass, hard_mass, effective_mass)), 'Invalid conditional probability')
        reason = 'empty_class' if empty else 'class_mass_below_floor' if class_mass <= 1e-12 else 'class_usable'
        fallbacks[f'{channel}:{fallback}'] += 1; fallbacks[f'{channel}:{reason}'] += 1
        for group in groups:
            selected_denominators[group] += 1; branch_groups[group+':'+fallback] += 1
            branch_groups[group+':'+reason] += 1
            masses[group+':class'].append(class_mass); masses[group+':hard'].append(hard_mass)
            if not empty and class_mass > 1e-12: hits[group] += 1
        require((valid and type(label['selected_channel_match']) is bool)
                or (not valid and label['selected_channel_match'] is None), 'Missing selected-channel endpoint result')
        if label['selected_channel_match']: branch_groups[f'channel:{channel}:endpoint_match'] += 1
    primary = ('native', 'competing', 'unbound', 'invalid_or_exterior')
    require(sum(counts[key] for key in primary) == size, 'Lost unconditional denominator')
    summaries = {key: dict(count=len(values), minimum=min(values), mean=math.fsum(values)/len(values), maximum=max(values))
                 for key, values in masses.items()}
    return dict(id=population['id'], attempted=size, selected_reference_rows=16, counts=dict(counts),
        original_orthants=dict(orthants), critical_endpoints={key: critical[key] for key in ('competing22', 'competing62', 'native55')},
        chosen_channels=dict(chosen), selected_fallbacks=dict(fallbacks), selected_branch_diagnostics=dict(branch_groups),
        target_line_hit_rates={key: dict(selected=value, hits=hits[key], hit_fraction=hits[key]/value)
                               for key, value in selected_denominators.items()},
        conditional_masses=summaries, producer_cpu_seconds=dict(cpu),
        analysis_cpu_seconds={phase: receipt.get('analysis_cpu_seconds') for phase, receipt in
                              [('algebra', algebra), ('labels', labels), ('geometry', geometry)]},
        query_counts={phase: dict(counts=receipt.get('counts'), setup_counts=receipt.get('setup_counts'))
                      for phase, receipt in [('labels', labels), ('geometry', geometry)]})


def aggregate(reports):
    require(len(reports) == 4 and [v['id'] for v in reports] == [f'r{i:02}' for i in range(4)]
            and all(v['attempted'] == 128 for v in reports), 'Four complete independent populations required')
    critical = {key: sum(r['critical_endpoints'][key] for r in reports) for key in ('competing22', 'competing62', 'native55')}
    selected = {str(channel): dict(selected=sum(r['target_line_hit_rates'].get(f'channel:{channel}', {}).get('selected', 0) for r in reports),
        hits=sum(r['target_line_hit_rates'].get(f'channel:{channel}', {}).get('hits', 0) for r in reports)) for channel in range(5)}
    for value in selected.values(): value['hit_fraction'] = value['hits']/value['selected'] if value['selected'] else None
    fractions = {}
    for key in ('native', 'competing', 'unbound', 'invalid_or_exterior'):
        values = [r['counts'].get(key, 0)/128 for r in reports]; mean = math.fsum(values)/4
        fractions[key] = dict(count=sum(r['counts'].get(key, 0) for r in reports),
            attempted_denominator=512, fraction=mean, population_SE=math.sqrt(math.fsum((x-mean)**2 for x in values)/12))
    access = all(critical[key] > 0 for key in ('competing22', 'competing62', 'native55')) \
        and all(selected[str(c)]['hits'] > 0 for c in (2, 3, 4))
    return dict(attempted_denominator=512, selected_reference_rows=64, fractions=fractions,
        critical_endpoints=critical, selected_target_line_access=selected,
        access_screen='access_observed_requires_review' if access else 'unsuccessful_or_inconclusive',
        physical_campaign_gate_open=False, full_vessel_gate_open=False, assembly_gate_open=False,
        inference='Prospective proposal access only. No physical mass, ESS, free energy, matched speedup or generalization conclusion.',
        zero_count_rule='Unobserved endpoints/selected branches do not imply zero physical mass.',
        training_critical_points=8, nontraining_critical_holdouts=0)


def run(protocol_path, digest):
    started = time.process_time(); bindings = Inputs()
    protocol, execution = worker.load_protocol(protocol_path, digest, bindings)
    for name, source in local_sources(__file__).items():
        require(source.parent == Path(protocol['code_directory']), 'Reducer has a mixed source directory')
        bindings.bind(source, protocol['source_sha256'][name])
    verify_live_authority(protocol, execution, 'all', 'statistics')
    root = Path(protocol['root']); output = root/'analysis/statistics.json'; failure = root/'analysis/statistics.failure.json'
    attempt = root/'analysis/statistics.attempt.json'
    require(not output.exists() and not failure.exists() and not attempt.exists(), 'Statistics is single-use')
    write(attempt, dict(schema=SCHEMA, protocol_sha256=digest, started=True, raw_rows_read=0, new_geometry_queries=0))
    handlers = {}
    def interrupted(signum, frame): raise InterruptedError('Statistics signal '+str(signum))
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGXCPU): handlers[sig] = signal.signal(sig, interrupted)
        reports = []; references = {}
        for population in protocol['populations']:
            receipts = {}
            for phase in ('algebra', 'labels', 'geometry'):
                identity = population['id']+'-'+phase
                reference, receipts[phase] = predecessor(root, execution, identity,
                    root/'analysis'/population['id']/(phase+'.json'), bindings)
                require(receipts[phase]['protocol_sha256'] == digest, 'Receipt came from another protocol')
                references[identity] = reference
            reports.append(population_report(population, receipts['algebra'], receipts['labels'], receipts['geometry'], protocol['component_inventory']))
        result = dict(schema=SCHEMA, complete=True, passed=True, protocol_sha256=digest,
            populations=reports, combined=aggregate(reports), predecessor_receipts=references,
            input_sha256=bindings.files, analysis_cpu_seconds=time.process_time()-started,
            new_Poisson_clouds=0, raw_rows_read=0, new_geometry_queries=0, retries=0,
            scope='Complete receipt reduction; passed means accounting checks passed, not physical convergence or an open campaign gate.')
        bindings.recheck(); verify_live_authority(protocol, execution, 'all', 'statistics')
        write(output, result)
        return result
    except BaseException as error:
        write(failure, dict(schema=SCHEMA, complete=False, passed=False, protocol_sha256=digest,
            error_type=type(error).__name__, error=str(error), input_sha256=bindings.files, retries=0,
            analysis_cpu_seconds=time.process_time()-started))
        raise
    finally:
        for sig, previous in handlers.items(): signal.signal(sig, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.protocol_sha256)['combined'], indent=2))
