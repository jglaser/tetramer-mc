#!/usr/bin/env python3
"""Independent fixed-allocation capped-dimer audit; never sample or update poses."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from dimer_destination_density import DimerDestinationDensity, compose_pose, relative_pose, pose_arrays
from dimer_destination_geometry import DimerGeometry
from analyze_dimer_destination_probe import Checks, full_density, fingerprints, quantiles, log_value, serial

MAP_FILES = ['src/docking.rs', 'src/proposal.rs', 'src/basin_involution.rs', 'src/math.rs',
             'src/dimer_tree_proposal.rs', 'src/defensive_dimer_proposal.rs', 'src/capped_dimer.rs',
             'src/geometry.rs', 'src/spherical.rs']
REFERENCE_CONFIG_SHA256 = '0bde358181834b1848b2b6dbebe84d1b7747bb9faeb6d7a0ac5ce79b3b018237'
PANEL_SHA256 = 'a9f758a3c7a1411931584a355d8bf0dbc29c9a0460aef46c49e51678667db6c1'
CAPS, ATTEMPTS = [1, 8, 32], 32


def require(ok, message):
    if not ok: raise ValueError(message)


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bound(record):
    require(sha(record['path']) == record['sha256'], 'Changed input: '+record['path'])
    return read(record['path'])


def stream_seed(master, atlas, case, attempt):
    return int(hashlib.sha256(f'capped-dimer-probe-v1/{master}/{atlas}/{case}/{attempt}'.encode()).hexdigest()[:16], 16)


def bind_current_source(run, config, binding, protocol):
    require(sha(run/'source-bundle.json') == binding['compiled_source_bundle_sha256'], 'Source bundle binding differs')
    bundle = read(run/'source-bundle.json')
    require(set(bundle['files']) == set(config['compiled_source_sha256']), 'Source list differs')
    for name, item in bundle['files'].items():
        actual = hashlib.sha256(item['text'].encode()).hexdigest()
        require(actual == item['sha256'] == config['compiled_source_sha256'][name], 'Corrupt source: '+name)
    require(set(protocol['map_density_source_sha256']) == set(MAP_FILES), 'Incomplete map source binding')
    for name in MAP_FILES:
        require(protocol['map_density_source_sha256'][name] == bundle['files'][name]['sha256'], 'Unbound map implementation: '+name)
    # This source closure is frozen before proposals. The old constructor binder
    # is deliberately not called: this experiment scores the actual map factors.
    require(sha(__file__) == protocol['audit_files']['analyze_capped_dimer_probe.py']['sha256'], 'Executing auditor differs from frozen source')
    for record in protocol['audit_files'].values():
        require(sha(record['path']) == record['sha256'], 'Changed frozen audit source')
    return {name: bundle['files'][name]['sha256'] for name in MAP_FILES}


def feasibility(oracle, case, selected, geometry):
    labels = [case['root'], case['child']]
    pair = sorted(labels)
    hard = geometry['hard_overlap_edges']
    collisions = [sorted(e[1] if e[0] == label else e[0]
                         for e in hard if label in e and e != pair) for label in labels]
    wall = [bool(np.all(np.sum(oracle.placed(p)**2, axis=1) <= (oracle.wall_radius-oracle.radii)**2)) for p in selected]
    return dict(internal_core_overlap=pair in hard, spectator_core_collisions=collisions,
                wall_valid=wall, internal_exclusion_contact=pair in geometry['contacts'])


def is_feasible(f):
    return not f['internal_core_overlap'] and not any(f['spectator_core_collisions']) and all(f['wall_valid']) and f['internal_exclusion_contact']


def verify_stopping(outcome, cap, source, checks, tag):
    trials = outcome['trials']
    checks.exact(outcome['trial_cap'], cap, 'trial cap', tag)
    checks.exact(outcome['source_feasibility'], source, 'source feasibility', tag)
    require(len(trials) <= cap, 'Too many raw trials')
    if not source['internal_exclusion_contact']:
        checks.exact(outcome['status'], 'source_outside_contact', 'ineligible source status', tag)
        checks.exact(trials, [], 'ineligible source trials', tag)
        checks.exact(outcome['candidate'], None, 'ineligible source candidate', tag)
        return
    for j, trial in enumerate(trials):
        checks.exact(trial['index'], j+1, 'raw index', tag)
        require(trial['draw'] is not None and trial['draw']['candidate'] is not None and trial['feasibility'] is not None,
                'Completed outer event contains numerical/null failure')
        if j+1 < len(trials): checks.exact(is_feasible(trial['feasibility']), False, 'first-success stopping', tag)
    if trials and is_feasible(trials[-1]['feasibility']):
        checks.exact(outcome['status'], 'candidate', 'successful status', tag)
        checks.exact(outcome['candidate'], trials[-1]['draw']['candidate'], 'selected first success', tag)
    else:
        checks.exact(outcome['status'], 'cap_exhausted', 'exhaustion status', tag)
        checks.exact(len(trials), cap, 'exhausted all trials', tag)
        checks.exact(outcome['candidate'], None, 'exhausted candidate', tag)


def verify_prefixes(rows, checks, tag):
    require([r['cap'] for r in rows] == CAPS, 'Missing/reordered paired caps')
    longest = rows[-1]['outcome']['trials']
    for row in rows:
        trials = row['outcome']['trials']
        checks.exact(trials, longest[:len(trials)], 'identical paired raw prefix', tag)
        checks.exact(row['raw_contacts'], rows[-1]['raw_contacts'][:len(trials)], 'paired raw contacts', tag)
        if len(trials) == len(longest):
            checks.exact(row['rng_after_fingerprint'], rows[-1]['rng_after_fingerprint'], 'paired terminal RNG', tag)
    return longest


def decode_edge(helper, edge, alpha, half_width, checks, tag):
    trace = edge['trace']
    coin = trace['branch_uniform']
    require(type(coin) in (float, int) and 0 <= coin < 1, 'Invalid mixture coin')
    branch = 'uniform' if coin < alpha else 'learned'
    checks.exact(edge['branch'], branch, 'mixture branch', tag)
    checks.exact(edge['null_reason'], None, 'raw edge null', tag)
    if branch == 'uniform':
        u, normals = np.asarray(trace['translation_uniforms']), np.asarray(trace['quaternion_normals'])
        require(u.shape == (3,) and np.isfinite(u).all() and np.all((u >= 0) & (u < 1)), 'Invalid uniform coordinates')
        require(normals.shape == (4,) and np.isfinite(normals).all() and np.linalg.norm(normals) > 0, 'Invalid Haar normals')
        pose = dict(position=((2*u-1)*half_width).tolist(), orientation=(normals/np.linalg.norm(normals)).tolist())
    else:
        label = trace['target_label']
        checks.exact({k:label[k] for k in ('kind','member','anchor')}, dict(kind='single',member=0,anchor=0), 'independent branch label', tag)
        pose = helper.decode(label['branch'], trace['target_latent'])
        require(helper.map_density.weights[label['branch']] > 0, 'Zero-weight generated component')
    checks.pose(edge['proposed_relative_pose'], pose, 'raw generation', tag)
    return pose


def audit_unique_trials(helper, oracle, case, old, anchor, trials, raw_contacts, alpha, half_width, checks, tag):
    old_edges = [relative_pose(anchor,old[0]), relative_pose(old[0],old[1])]
    # Score only the actual map-factor density. Complete reciprocal/component
    # mixtures are independently evaluated, not inferred from selected labels.
    poses = list(old_edges)
    endpoints = []
    for trial in trials:
        c = trial['draw']['candidate']; new = [c['root'],c['child']]
        edges = [relative_pose(anchor,new[0]),relative_pose(new[0],new[1])]
        endpoints.append((new,edges)); poses.extend(edges)
    log_g = helper.map_density.evaluate(poses)[0]
    densities = [full_density(float(g),p,alpha,half_width) for p,g in zip(poses,log_g)]
    results = []
    for j,(trial,(new,new_edges)) in enumerate(zip(trials,endpoints)):
        where = f'{tag}/raw{j+1}'; draw = trial['draw']; diag = draw['candidate']['diagnostics']
        checks.exact(draw['old_root'],old[0],'fixed root source',where)
        checks.exact(draw['old_child'],old[1],'fixed child source',where)
        checks.exact(draw['spectator'],anchor,'fixed anchor',where)
        checks.exact(draw['null_reason'],None,'joint null',where)
        generated = []
        for e,edge in enumerate(draw['edges']):
            checks.pose(edge['old_relative_pose'],old_edges[e],'old tree coordinates',where)
            generated.append(decode_edge(helper,edge,alpha,half_width,checks,where))
        root = compose_pose(anchor,generated[0]); child = compose_pose(root,generated[1])
        checks.pose(new[0],root,'new root decoding',where); checks.pose(new[1],child,'new child decoding',where)
        expected_old, expected_new = densities[:2], densities[2+2*j:4+2*j]
        for name, expected in [('old_edges',expected_old),('new_edges',expected_new)]:
            for actual,want in zip(diag[name],expected):
                for key in ('log_uniform','log_learned','log_full'):
                    checks.close(log_value(actual[key]),want[key],'map-factor '+name+' '+key,where)
        old_full=sum(d['log_full'] for d in expected_old); new_full=sum(d['log_full'] for d in expected_new)
        correction=old_full-new_full
        for key,want in [('full_old_log_density',old_full),('full_new_log_density',new_full),
                         ('log_reverse_forward',correction),('log_tree_coordinate_jacobian',0.),('selection_log_reverse_forward',0.)]:
            checks.close(log_value(diag[key]),want,'joint '+key,where)
        geometry = oracle.fingerprint([case['root'],case['child']],new)
        predicted = feasibility(oracle,case,new,geometry)
        checks.exact(trial['feasibility'],predicted,'independent raw geometry',where)
        checks.exact(raw_contacts[j],geometry['contacts'],'independent raw contacts',where)
        pair=sorted([case['root'],case['child']]); anchor_edge=sorted([case['root'],case['anchor']])
        results.append(dict(index=j+1,hard_valid=geometry['hard_valid'],feasible=is_feasible(predicted),
            internal_contact=pair in geometry['contacts'],both_intended_contacts=pair in geometry['contacts'] and anchor_edge in geometry['contacts'],
            external_contacts=len(geometry['contacts'])-int(pair in geometry['contacts']),contacts=geometry['contacts'],
            log_reverse_forward=correction,branches=[e['branch'] for e in draw['edges']],
            threshold_ambiguity=geometry['threshold_ambiguity']))
    return results


def summarize(rows):
    candidates=[r for r in rows if r['status']=='candidate']
    return dict(outer_attempts=len(rows),raw_trials=sum(r['raw_trials'] for r in rows),
                candidate_count=len(candidates),candidate_fraction=len(candidates)/len(rows) if rows else None,
                candidates_per_raw_trial=len(candidates)/sum(r['raw_trials'] for r in rows) if sum(r['raw_trials'] for r in rows) else None,
                candidates_per_proposal_cpu_second=len(candidates)/sum(r['proposal_cpu_seconds'] for r in rows) if sum(r['proposal_cpu_seconds'] for r in rows) else None,
                raw_hard_valid=sum(r.get('raw_hard_valid',0) for r in rows),
                raw_hard_valid_without_internal_contact=sum(r.get('raw_hard_valid_without_internal_contact',0) for r in rows),
                status_counts=dict(Counter(r['status'] for r in rows)),
                candidates_with_external_contacts=sum(r['candidate']['external_contacts']>0 for r in candidates),
                both_intended_contacts=sum(r['candidate']['both_intended_contacts'] for r in candidates),
                log_reverse_forward=quantiles([r['candidate']['log_reverse_forward'] for r in candidates]),
                proposal_cpu_seconds=sum(r['proposal_cpu_seconds'] for r in rows),
                contact_diagnostic_cpu_seconds=sum(r['contact_diagnostic_cpu_seconds'] for r in rows))


def analyze(run):
    clock=time.process_time(); run=Path(run)
    config,binding,terminal,protocol=[read(run/p) for p in ['config.json','binding.json','terminal.json','protocol.json']]
    require(config['schema']=='capped-dimer-screen-v1' and binding['schema']=='capped-dimer-probe-binding-v1','Unknown schema')
    for name,key in [('config.json','config_sha256'),('protocol.json','protocol_sha256'),('example.rs','example_source_sha256')]:
        require(sha(run/name)==binding[key],'Binding mismatch: '+name)
    require(sha(run/'attempts.jsonl')==terminal['attempts_sha256'],'Attempt ledger changed')
    require(terminal['summary']['complete'] is True,'Probe failed; retain original fatal ledger and do not substitute draws')
    require(config['density_law']=='map-factor-full-mixture-v1','Wrong scoring law')
    require(config['caps']==CAPS and config['attempts_per_context']==ATTEMPTS,'Allocation changed')
    require(config['reference_config']['sha256']==REFERENCE_CONFIG_SHA256,'Reference source configuration differs')
    ref=bound(config['reference_config']); panel=bound(ref['panel'])
    require(ref['panel']['sha256']==PANEL_SHA256 and ref['cases']==panel['cases'],'Panel differs')
    require(len(ref['cases'])==8 and len(ref['atlases'])==3,'Panel size differs')
    require(ref['uniform_probability']==.5 and ref['uniform_half_width']==160. and ref['depletant_radius']==1.4 and ref['activity']==.0275,'Model settings differ')
    source=bound(ref['source_config']); frame=bound(ref['source_frame']); archive=bound(ref['source_freeze_manifest']); shape=bound(ref['shape'])
    require(source['initial_poses']==frame['poses'] and archive['frame_sha256']==ref['source_frame']['sha256'] and archive['shape_sha256']==ref['shape']['sha256'],'Source mismatch')
    source_binding=bind_current_source(run,config,binding,protocol)
    state=source['initial_poses']; oracle=DimerGeometry(shape,state,ref['depletant_radius'],ref['wall_radius'])
    checks=Checks(); results=[]; unique_raw=0; raw_ambiguities=0; source_records=[]
    with (run/'attempts.jsonl').open() as stream:
        for ai,atlas in enumerate(ref['atlases']):
            helper=DimerDestinationDensity(bound(atlas['model']))  # no obsolete constructor source binder
            for ci,case in enumerate(ref['cases']):
                old=[state[case['root']],state[case['child']]]; anchor=state[case['anchor']]
                source_geometry=oracle.fingerprint([case['root'],case['child']],old)
                source_feasibility=feasibility(oracle,case,old,source_geometry)
                require(source_geometry['hard_valid'],'Frozen source is hard-invalid')
                source_records.append(dict(atlas=atlas['name'],case=case['name'],feasibility=source_feasibility))
                for attempt in range(ATTEMPTS):
                    group=[]; tag=f'{ai}/{ci}/{attempt}'
                    for cap in CAPS:
                        raw=stream.readline(); require(bool(raw),'Missing outer attempt')
                        row=json.loads(raw); group.append(row)
                        require(row['status']=='completed','Fatal proposal retained; screen is not complete')
                        for key,want in [('atlas',atlas['name']),('atlas_index',ai),('case',case),('case_index',ci),('attempt',attempt),('cap',cap),
                                         ('seed',stream_seed(config['master_seed'],ai,ci,attempt)),('old',old),('anchor_pose',anchor),('old_contacts',source_geometry['contacts'])]:
                            checks.exact(row[key],want,'row '+key,tag)
                        outcome=row['outcome']
                        for key,want in [('members',[case['root'],case['child']]),('anchor_label',case['anchor']),('old',old)]:
                            checks.exact(outcome[key],want,'outcome '+key,tag)
                        verify_stopping(outcome,cap,source_feasibility,checks,tag)
                        checks.exact(len(row['raw_contacts']),len(outcome['trials']),'raw contact ledger length',tag)
                    trials=verify_prefixes(group,checks,tag); unique_raw+=len(trials)
                    audited=audit_unique_trials(helper,oracle,case,old,anchor,trials,group[-1]['raw_contacts'],.5,160.,checks,tag)
                    raw_ambiguities+=sum(r['threshold_ambiguity'] for r in audited)
                    for row in group:
                        outcome=row['outcome']; count=len(outcome['trials'])
                        candidate=audited[count-1] if outcome['candidate'] is not None else None
                        result=dict(atlas=atlas['name'],case=case['name'],attempt=attempt,cap=row['cap'],status=outcome['status'],
                            raw_trials=count,candidate=candidate,proposal_cpu_seconds=row['proposal_cpu_seconds'],
                            contact_diagnostic_cpu_seconds=row['contact_diagnostic_cpu_seconds'],
                            raw_hard_valid=sum(r['hard_valid'] for r in audited[:count]),
                            raw_hard_valid_without_internal_contact=sum(r['hard_valid'] and not r['internal_contact'] for r in audited[:count]))
                        if candidate: result['contact_change']=fingerprints(source_geometry['contacts'],candidate['contacts'])
                        results.append(result)
        require(not stream.readline(),'Extra unallocated attempt')
    total=summarize(results)
    for key,want in [('outer_attempts',len(results)),('raw_trials',total['raw_trials']),('candidates',total['candidate_count']),('physical_draws',0),('state_updates',0)]:
        checks.exact(terminal['summary']['result'][key],want,'terminal '+key)
    require(len(results)==2304 and total['raw_trials']<=31488,'Allocation violated')
    comparisons=[dict(atlas=a['name'],cap=k,summary=summarize([r for r in results if r['atlas']==a['name'] and r['cap']==k]),
                 contexts=[dict(case=c['name'],summary=summarize([r for r in results if r['atlas']==a['name'] and r['cap']==k and r['case']==c['name']])) for c in ref['cases']])
                 for a in ref['atlases'] for k in CAPS]
    return dict(schema='capped-dimer-independent-analysis-v1',complete=True,passed=not checks.failures,
        checks=checks.count,failures=checks.failures,maximum_absolute_errors=dict(checks.maximum_absolute_errors),
        source_binding=source_binding,input_hashes={p:sha(run/p) for p in ['config.json','binding.json','protocol.json','source-bundle.json','example.rs','attempts.jsonl','terminal.json']},
        summary=total,comparisons=comparisons,source_contexts=source_records,rows=results,
        pairing=dict(independent_source_seed_slots=768,key=['atlas','case','attempt'],caps_share_prefixes=True,
                     statistical_intervals='None: fixed purposive contexts; cap arms are dependent paired controls.'),
        unique_raw_trials_independently_reconstructed=unique_raw,raw_prefix_duplicates_verified=total['raw_trials']-unique_raw,
        unique_raw_threshold_ambiguities=raw_ambiguities,analyzer_cpu_seconds=time.process_time()-clock,
        probe_cpu_seconds=terminal['cpu_seconds'],density_tolerance=dict(absolute=2e-7,relative=2e-10),
        limitations=['Passive reset-source proposal comparison, not physical acceptance, equilibrium, native registry or assembly.',
                     'Caps share exact raw prefixes; they are paired controls, not independent populations.',
                     'Every raw endpoint audited; duplicate prefixes checked bitwise and counted in every outer denominator.',
                     'No source or native-label filtering and no allocation extensions. Original physical decision conditions remain separate.',
                     'Independent reconstruction does not certify floating-point arithmetic or unseen modes; prior stationarity failure remains unresolved.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as out:
        try: report=analyze(args.run)
        except Exception as error: report=dict(schema='capped-dimer-independent-analysis-v1',complete=False,passed=False,error=f'{type(error).__name__}: {error}')
        json.dump(serial(report),out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:report.get(k) for k in ['complete','passed','checks','analyzer_cpu_seconds','error']}))
    raise SystemExit(0 if report.get('passed') else 1)
