#!/usr/bin/env python3
"""Audit the frozen passive dimer destination screen; never draw or update poses."""
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
from scipy.spatial.transform import Rotation
from dimer_destination_density import DimerDestinationDensity, compose_pose, relative_pose, pose_errors
from dimer_destination_geometry import DimerGeometry
from prepare_dimer_destination_panel import collect_pairs, select_pairs, make_cases

ABS_TOL, REL_TOL = 2e-7, 2e-10


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bound(record):
    require(sha(record['path']) == record['sha256'], f"Changed input: {record['path']}")
    return read(record['path'])


def log_value(value):
    if value == '-inf':
        return -math.inf
    value = float(value)
    require(math.isfinite(value), 'Invalid serialized log density')
    return value


def serial(value):
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return '-inf' if value < 0 else ('inf' if value > 0 else 'nan')
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(v) for v in value]
    return value


def full_density(log_g, pose, alpha, half_width):
    log_u = -3*math.log(2*half_width) if all(abs(x) <= half_width for x in pose['position']) else -math.inf
    if alpha == 0:
        full = log_g
    elif alpha == 1:
        full = log_u
    else:
        full = float(np.logaddexp(math.log(alpha)+log_u, math.log1p(-alpha)+log_g))
    return dict(log_uniform=log_u, log_learned=log_g, log_full=full)


def log_difference(a, b):
    """Keep a zero/zero comparison visible instead of creating a NaN."""
    if a == b == -math.inf:
        return 'undefined_both_zero_density'
    return a-b


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []
        self.maximum_absolute_errors = defaultdict(float)

    def exact(self, actual, expected, kind, row=None):
        self.count += 1
        if actual != expected:
            self.failures.append(dict(row=row, check=kind, actual=actual, expected=expected))

    def close(self, actual, expected, kind, row=None):
        self.count += 1
        a, b = float(actual), float(expected)
        if a == b:
            return
        error = abs(a-b)
        self.maximum_absolute_errors[kind] = max(self.maximum_absolute_errors[kind], error)
        if not (math.isfinite(a) and math.isfinite(b) and error <= ABS_TOL+REL_TOL*abs(b)):
            self.failures.append(dict(row=row, check=kind, actual=a, expected=b, absolute_error=error))

    def pose(self, actual, expected, kind, row=None):
        # Compare rotation matrices, never quaternion sign representatives.
        for i, (a, b) in enumerate(zip(actual['position'], expected['position'])):
            self.close(a, b, kind+'.translation', row)
        a = Rotation.from_quat(np.asarray(actual['orientation'])[[1, 2, 3, 0]]).as_matrix()
        b = Rotation.from_quat(np.asarray(expected['orientation'])[[1, 2, 3, 0]]).as_matrix()
        for x, y in zip(a.flat, b.flat):
            self.close(x, y, kind+'.rotation', row)


def fingerprints(old, new):
    a, b = {tuple(e) for e in old}, {tuple(e) for e in new}
    union = a | b
    return dict(old_contacts=len(a), new_contacts=len(b), gained_contacts=len(b-a), lost_contacts=len(a-b),
                fingerprint_jaccard=len(a & b)/len(union) if union else 1.,
                empty_union=not union)


def quantiles(values):
    x = np.asarray(values, float)
    finite = x[np.isfinite(x)]
    return dict(count=len(x), negative_infinity=int(np.isneginf(x).sum()), finite_count=len(finite),
                quantiles=dict(zip(('min', 'p05', 'median', 'p95', 'max'), np.quantile(finite, [0, .05, .5, .95, 1]).tolist())) if len(finite) else None)


def summarize(rows):
    finite = [r for r in rows if r['status'] == 'finite_endpoint']
    valid = [r for r in finite if r.get('hard_valid')]
    result = dict(attempts=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
                  finite_endpoints=len(finite), hard_valid=len(valid), hard_valid_all_attempt_fraction=len(valid)/len(rows),
                  branch_pairs=dict(Counter('/'.join(r.get('branches', [])) for r in finite)))
    for name, group in [('all_finite', finite), ('hard_valid_only', valid)]:
        result[name] = dict(count=len(group))
        for metric in ('log_reverse_forward', 'fingerprint_jaccard', 'gained_contacts', 'lost_contacts',
                       'external_contacts', 'proposal_cpu_seconds', 'hard_cpu_seconds', 'contact_cpu_seconds'):
            result[name][metric] = quantiles([r[metric] for r in group if metric in r])
        result[name]['internal_contact_count'] = sum(r.get('internal_contact', False) for r in group)
    return result


def authenticate(run):
    config, binding, terminal = (read(run/name) for name in ('config.json', 'binding.json', 'terminal.json'))
    require(sha(run/'config.json') == binding['config_sha256'], 'Config binding mismatch')
    require(sha(run/'protocol.json') == binding['protocol_sha256'], 'Protocol binding mismatch')
    require(sha(run/'source-bundle.json') == binding['compiled_source_bundle_sha256'], 'Source bundle binding mismatch')
    require(sha(run/'example.rs') == binding['example_source_sha256'], 'Example source binding mismatch')
    require(sha(run/'attempts.jsonl') == terminal['attempts_sha256'], 'Attempt terminal hash mismatch')
    require(terminal['summary']['complete'] is True, 'Probe did not complete; retain terminal failure separately')
    bundle = read(run/'source-bundle.json')
    for name, entry in bundle['files'].items():
        require(hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256'], f'Source entry mismatch {name}')
    require({k: v['sha256'] for k,v in bundle['files'].items()} == config['compiled_source_sha256'], 'Compiled source closure differs')
    for name in ('protocol', 'source_config', 'source_frame', 'source_freeze_manifest', 'shape', 'panel'):
        bound(config[name])
    frame, physical, archive = (bound(config[k]) for k in ('source_frame', 'source_config', 'source_freeze_manifest'))
    require(frame['poses'] == physical['initial_poses'], 'Source poses changed')
    require(archive['frame_sha256'] == config['source_frame']['sha256'], 'Source freeze frame mismatch')
    require(archive['shape_sha256'] == config['shape']['sha256'], 'Source freeze shape mismatch')
    require(physical['depletant_radius'] == config['depletant_radius'] and physical['reservoir_density'] == config['activity']
            and physical['boundary']['radius'] == config['wall_radius'], 'Physical settings mismatch')
    panel = bound(config['panel'])
    require(panel['cases'] == config['cases'] and panel['uniform_half_width'] == config['uniform_half_width'], 'Panel/config mismatch')
    for record in panel['source_bindings'].values():
        require(sha(record['path']) == record['sha256'], 'Panel source changed')
    streams = [(f'events-p{i}.jsonl', [json.loads(line) for line in Path(panel['source_bindings'][f'events-p{i}.jsonl']['path']).read_text().splitlines()]) for i in range(4)]
    table, first = collect_pairs(frame['poses'], streams)
    selected = select_pairs(table)
    cases, details = make_cases(frame['poses'], selected, config['uniform_half_width'])
    require(table == panel['complete_eligible_pair_table'] and first == panel['complete_first_event_table']
            and selected == panel['selected_pairs'] and cases == panel['cases'] and details == panel['case_details'], 'Panel rule reconstruction mismatch')
    require(config['attempts_per_cell'] == 64 and len(cases) == 8 and len(config['atlases']) == 3
            and config['laws'] == ['tree', 'independent_learned', 'defensive'], 'Frozen allocation differs')
    return config, terminal, frame, panel


def evaluate_tasks(helper, tasks, batch=128):
    """Evaluate whole mixtures once per batch; retain requested branch scores."""
    results = []
    for start in range(0, len(tasks), batch):
        part = tasks[start:start+batch]
        poses = [t['pose'] for t in part]
        score, _, components = helper.score.evaluate(poses)
        generated, _, _ = helper.map_density.evaluate(poses)
        for i, task in enumerate(part):
            results.append(dict(log_g=float(score[i]), map_log_g=float(generated[i]),
                                branch_log=float(components[i, task['branch']]) if task.get('branch') is not None else None))
    return results


def analyze(run):
    clock = time.process_time()
    run = Path(run).resolve()
    config, terminal, frame, panel = authenticate(run)
    rows = [json.loads(line) for line in (run/'attempts.jsonl').read_text().splitlines()]
    expected = [(a, c, law, attempt) for a in config['atlases'] for c in config['cases']
                for law in config['laws'] for attempt in range(config['attempts_per_cell'])]
    require(len(rows) == len(expected) == 4608, 'Incomplete or expanded attempt allocation')
    checks, summaries, result_rows, factor_reports = Checks(), [], [], {}
    state = frame['poses']
    oracle = DimerGeometry(bound(config['shape']), state, config['depletant_radius'], config['wall_radius'])
    old_geometry = {}
    for case in config['cases']:
        members = [case['root'], case['child']]
        old_geometry[case['name']] = oracle.fingerprint(members, [state[i] for i in members])
    atlas_data = {}
    for index, (row, (atlas, case, law, attempt)) in enumerate(zip(rows, expected)):
        checks.exact((row['atlas'], row['case'], row['law'], row['attempt']), (atlas['name'], case, law, attempt), 'allocation/order', index)
        ai = config['atlases'].index(atlas);ci = config['cases'].index(case);li = config['laws'].index(law)
        seed_text = f"protein-dimer-destinations-v1/{config['master_seed']}/{ai}/{ci}/{li}"
        checks.exact(row['cell_seed'], int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16), 'cell seed', index)
        old = [state[case['root']], state[case['child']]];anchor = state[case['anchor']]
        checks.exact(row['old_selected'], old, 'exact source copies', index)
        checks.exact(row['anchor_pose'], anchor, 'exact anchor copy', index)
        derived = [relative_pose(anchor, old[0]), relative_pose(old[0], old[1])]
        for e in range(2):checks.pose(row['old_edges'][e], derived[e], 'old physical tree coordinates', index)
        checks.exact(row['old_contacts'], old_geometry[case['name']]['contacts'], 'old contact fingerprint', index)
        require(row['status'] in ('finite_endpoint', 'proposal_null', 'proposal_error'), 'Unknown attempted status')
        data = atlas_data.setdefault(atlas['name'], dict(tasks=[], references={}))
        def task(key, pose, branch=None):
            if key not in data['references']:
                data['references'][key] = len(data['tasks']);data['tasks'].append(dict(pose=pose, branch=branch))
        for e in range(2):task(('old', case['name'], e), row['old_edges'][e])
        if row['status'] != 'finite_endpoint':continue
        checks.exact(row.get('diagnostic_error'), None, 'finite endpoint diagnostic error', index)
        checks.exact('new_edges' in row and 'new_densities' in row, True, 'finite endpoint diagnostics present', index)
        candidate = row['outcome']['candidate']
        checks.exact(row['proposed'], [candidate['root'], candidate['child']], 'candidate endpoint copies', index)
        checks.exact(row['outcome']['old_root'], old[0], 'trace old root copy', index)
        checks.exact(row['outcome']['old_child'], old[1], 'trace old child copy', index)
        checks.exact(row['outcome']['spectator'], anchor, 'trace anchor copy', index)
        world = [relative_pose(anchor, row['proposed'][0]), relative_pose(row['proposed'][0], row['proposed'][1])]
        for e, edge in enumerate(row['outcome']['edges']):
            checks.pose(edge['old_relative_pose'], row['old_edges'][e], 'trace old tree edge', index)
            if 'new_edges' in row:
                checks.pose(row['new_edges'][e], world[e], 'new physical tree coordinates', index)
                task(('world', index, e), row['new_edges'][e])
            else:task(('world', index, e), world[e])
            local = edge['step']['handle'] if law == 'tree' else edge['proposed_relative_pose']
            task(('local', index, e), local, edge['trace']['target'] if law == 'tree' else None)
            if law == 'tree':task(('source', index, e), row['old_edges'][e], edge['trace']['source'])
        local = [e['step']['handle'] if law == 'tree' else e['proposed_relative_pose'] for e in row['outcome']['edges']]
        root = compose_pose(anchor, local[0]);child = compose_pose(root, local[1])
        for a,b in zip(row['proposed'], [root, child]):checks.pose(a,b,'physical tree composition',index)
    for atlas in config['atlases']:
        data = atlas_data[atlas['name']]
        helper = DimerDestinationDensity(bound(atlas['model']), source_bundle=run/'source-bundle.json')
        data['helper'] = helper
        data['values'] = evaluate_tasks(helper, data['tasks'])
        factor_reports[atlas['name']] = dict(factors=helper.factor_report, source_binding=helper.source_binding)
    for index, row in enumerate(rows):
        case, law = row['case'], row['law'];data = atlas_data[row['atlas']];helper=data['helper']
        value = lambda key: data['values'][data['references'][key]]
        old = [value(('old', case['name'], e)) for e in range(2)]
        for e in range(2):
            reference = full_density(old[e]['log_g'], row['old_edges'][e], config['uniform_probability'], config['uniform_half_width'])
            for key, expected_log in reference.items():checks.close(log_value(row['old_densities'][e][key]), expected_log, 'old '+key, index)
        result = dict(index=index, atlas=row['atlas'], case=case['name'], law=law, attempt=row['attempt'], status=row['status'],
                      proposal_cpu_seconds=row['proposal_cpu_seconds'])
        if row['status'] != 'finite_endpoint':
            result.update(error=row.get('error'), outcome=row.get('outcome'));result_rows.append(result);continue
        world = [value(('world', index, e)) for e in range(2)]
        local = [value(('local', index, e)) for e in range(2)]
        diagnostics = row['outcome']['candidate']['diagnostics']
        alpha = config['uniform_probability'] if law == 'defensive' else 0.
        new_poses = [data['tasks'][data['references'][('world', index, e)]]['pose'] for e in range(2)]
        old_logs = [full_density(old[e]['log_g'], row['old_edges'][e], alpha, config['uniform_half_width'])['log_full'] for e in range(2)]
        new_logs = ([v['log_g'] for v in local] if law == 'tree' else
                    [full_density(world[e]['log_g'], new_poses[e], alpha, config['uniform_half_width'])['log_full'] for e in range(2)])
        correction = sum(old_logs)-sum(new_logs)
        checks.close(log_value(row['log_reverse_forward']), correction, 'joint proposal correction', index)
        checks.close(log_value(diagnostics['log_reverse_forward']), correction, 'candidate correction', index)
        checks.close(log_value(diagnostics['full_old_log_density']), sum(old_logs), 'joint old density', index)
        checks.close(log_value(diagnostics['full_new_log_density']), sum(new_logs), 'joint new density', index)
        checks.exact(diagnostics['log_tree_coordinate_jacobian'], 0., 'tree coordinate Jacobian', index)
        checks.exact(diagnostics['selection_log_reverse_forward'], 0., 'fixed label correction', index)
        for e in range(2):
            reference = full_density(world[e]['log_g'], new_poses[e], config['uniform_probability'], config['uniform_half_width'])
            if 'new_densities' in row:
                for key, expected_log in reference.items():checks.close(log_value(row['new_densities'][e][key]), expected_log, 'new '+key, index)
            if law != 'tree':
                for when, g, pose in [('old', old[e]['log_g'], row['old_edges'][e]), ('new', world[e]['log_g'], new_poses[e])]:
                    for key, expected_log in full_density(g, pose, alpha, config['uniform_half_width']).items():
                        checks.close(log_value(diagnostics[when+'_edges'][e][key]), expected_log, 'candidate '+when+' '+key, index)
        # Generation is reconstructed from the saved raw variables, with no RNG.
        branches = []
        for e, edge in enumerate(row['outcome']['edges']):
            if law == 'tree':
                trace=edge['trace'];step=edge['step'];raw=step['step']
                source = helper.encode(trace['source'], edge['old_relative_pose'])
                noise=np.asarray(trace['noise']);sine=math.sqrt(1-config['correlation']**2)
                target=config['correlation']*source+sine*noise
                inverse=sine*source-config['correlation']*noise
                for a,b in zip(raw['source_latent'],source):checks.close(a,b,'tree source latent',index)
                for a,b in zip(raw['target_latent'],target):checks.close(a,b,'tree target latent',index)
                for a,b in zip(raw['inverse_trace']['noise'],inverse):checks.close(a,b,'tree inverse noise',index)
                checks.exact([raw['inverse_trace']['source'],raw['inverse_trace']['target']], [trace['target'],trace['source']], 'reversed tree labels', index)
                checks.pose(step['handle'],helper.decode(trace['target'],target),'tree generated edge',index)
                aux=.5*float(noise@noise-inverse@inverse)
                jac=helper.log_volume(trace['target'],target)-helper.log_volume(trace['source'],source)
                source_value=value(('source', index, e))
                label=(local[e]['branch_log']-local[e]['log_g'])-(source_value['branch_log']-source_value['log_g'])+math.log(helper.score.weights[trace['source']])-math.log(helper.score.weights[trace['target']])
                for actual,reference,kind in [(raw['log_auxiliary_ratio'],aux,'tree auxiliary'),(raw['log_extended_jacobian'],jac,'tree Jacobian'),
                    (step['label_log_reverse_forward'],label,'tree posterior labels'),(step['expanded_log_reverse_forward'],aux+jac+label,'tree expanded correction'),
                    (step['log_reverse_forward'],old[e]['log_g']-local[e]['log_g'],'tree edge correction'),
                    (step['full_old_member_log_density'],old[e]['log_g'],'tree old edge density'),
                    (step['full_new_member_log_density'],local[e]['log_g'],'tree new edge density')]:checks.close(actual,reference,kind,index)
                checks.exact(row['outcome']['candidate']['inverse_trace']['edges'][e],raw['inverse_trace'],'candidate inverse trace copy',index)
                branches.append('correlated_map')
            else:
                trace=edge['trace'];branches.append(edge['branch'])
                checks.exact(edge['branch'], 'uniform' if trace['branch_uniform'] < alpha else 'learned', 'defensive branch coin', index)
                if edge['branch'] == 'uniform':
                    q=np.asarray(trace['quaternion_normals']);q=q/np.linalg.norm(q)
                    proposed=dict(position=((2*np.asarray(trace['translation_uniforms'])-1)*config['uniform_half_width']).tolist(),orientation=q.tolist())
                else:
                    checks.exact(trace['target_label']['anchor'],0,'independent virtual anchor',index)
                    proposed=helper.decode(trace['target_label']['branch'],trace['target_latent'])
                checks.pose(edge['proposed_relative_pose'],proposed,'independent generated edge',index)
        if law == 'tree':
            for key in ('log_extended_jacobian','log_auxiliary_ratio','label_log_reverse_forward','expanded_log_reverse_forward'):
                total=sum(edge['step']['step'][key] if key in ('log_extended_jacobian','log_auxiliary_ratio') else edge['step'][key]
                          for edge in row['outcome']['edges'])
                checks.close(diagnostics[key],total,'joint '+key,index)
        geom = oracle.fingerprint([case['root'],case['child']],row['proposed'])
        for name in ('hard_overlap_edges','wall_valid','hard_valid'):checks.exact(row[name],geom[name],'independent '+name,index)
        checks.exact(row['new_contacts'],geom['contacts'],'independent new contacts',index)
        contact = fingerprints(row['old_contacts'],geom['contacts']);internal=sorted([case['root'],case['child']]) in geom['contacts']
        result.update(contact, hard_valid=geom['hard_valid'], geometry=geom, internal_contact=internal,
                      external_contacts=len(geom['contacts'])-int(internal), branches=branches,
                      log_reverse_forward=correction, hard_cpu_seconds=row['hard_cpu_seconds'],contact_cpu_seconds=row['contact_cpu_seconds'],
                      old_log_g=[v['log_g'] for v in old],new_log_g=[v['log_g'] for v in world],
                      old_log_f=[full_density(old[e]['log_g'],row['old_edges'][e],config['uniform_probability'],config['uniform_half_width'])['log_full'] for e in range(2)],
                      new_log_f=[full_density(world[e]['log_g'],new_poses[e],config['uniform_probability'],config['uniform_half_width'])['log_full'] for e in range(2)],
                      old_law_log_density=old_logs,new_law_log_density=new_logs,
                      world_minus_local_log_g=[log_difference(world[e]['log_g'],local[e]['log_g']) for e in range(2)],
                      old_map_minus_score_log_g=[log_difference(v['map_log_g'],v['log_g']) for v in old],
                      new_map_minus_score_log_g=[log_difference(v['map_log_g'],v['log_g']) for v in world],
                      new_map_minus_score_log_f=[log_difference(full_density(world[e]['map_log_g'],new_poses[e],config['uniform_probability'],config['uniform_half_width'])['log_full'],
                                                full_density(world[e]['log_g'],new_poses[e],config['uniform_probability'],config['uniform_half_width'])['log_full']) for e in range(2)],
                      diagnostic_error=row.get('diagnostic_error'))
        result_rows.append(result)
    totals = terminal['summary']['result']
    for key, actual in [('attempts',len(rows)),('finite_endpoints',sum(r['status']=='finite_endpoint' for r in rows)),
                        ('hard_valid',sum(r.get('hard_valid',False) for r in rows)),('proposal_errors',sum(r['status']=='proposal_error' for r in rows)),
                        ('nulls',sum(r['status']=='proposal_null' for r in rows)),('physical_draws',0),('state_updates',0)]:
        checks.exact(totals[key],actual,'terminal '+key)
    for atlas in config['atlases']:
        for law in config['laws']:
            group=[r for r in result_rows if r['atlas']==atlas['name'] and r['law']==law]
            summaries.append(dict(atlas=atlas['name'],law=law,summary=summarize(group),
                                  contexts=[dict(case=c['name'],summary=summarize([r for r in group if r['case']==c['name']])) for c in config['cases']]))
    return dict(schema='dimer-destination-independent-analysis-v1',complete=True,passed=not checks.failures,
                input_hashes={p.name:sha(p) for p in [run/'attempts.jsonl',run/'terminal.json',run/'binding.json',run/'config.json',run/'source-bundle.json',run/'example.rs',run/'protocol.json']},
                checks=checks.count,failures=checks.failures,maximum_absolute_errors=dict(checks.maximum_absolute_errors),
                density_tolerance=dict(absolute=ABS_TOL,relative=REL_TOL),factor_reports=factor_reports,
                jaccard_empty_union=1.,summary=summarize(result_rows),comparisons=summaries,rows=result_rows,
                analyzer_cpu_seconds=time.process_time()-clock,probe_cpu_seconds=terminal['cpu_seconds'],
                limitations=['All 4608 attempted draws retained; diagnostic failures never trigger redraw or allocation extension.',
                             'Eight contexts from four selected pairs in one saved frame; conditional source feasibility only.',
                             'No physical Poisson clouds, acceptance decisions, state updates, native classifier, overlap volume or assembly-rate inference.',
                             'Map-versus-score numerical differences are reported; observed agreement does not prove exact coefficient closure or unseen-tail coverage.',
                             'Proposal-only log corrections are not physical acceptance probabilities or bounds.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Claim output once, retaining terminal failure if authentication or audit fails.
    with args.output.open('x') as output:
        try:report=analyze(args.run)
        except Exception as error:
            report=dict(schema='dimer-destination-independent-analysis-v1',complete=False,passed=False,error=f'{type(error).__name__}: {error}')
        json.dump(serial(report),output,indent=2,allow_nan=False);output.write('\n')
    print(json.dumps({k:report.get(k) for k in ('complete','passed','checks','analyzer_cpu_seconds','error')},allow_nan=False))
    raise SystemExit(0 if report.get('passed') else 1)
