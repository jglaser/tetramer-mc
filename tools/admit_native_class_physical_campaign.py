#!/usr/bin/env python3
"""Join completed physical evidence without parsing rows or observing geometry.

This is a post-controller metadata admission, never a scientific worker. It
authenticates existing all-row, endpoint, selected-geometry and statistics
receipts. It cannot establish unseen-support bounds or assembly stability.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import analyze_native_class_physical_populations as statistics
import native_class_selected_geometry as selected
import run_native_class_physical_campaign as runner
from analyze_mobile_native_pocket import local_sources

streaming = statistics.streaming
require = streaming.require
PLAN_SCHEMA = 'native-class-physical-admission-plan-v1'
SCHEMA = 'native-class-physical-admission-v1'
PROTOCOL_SCHEMA = 'native-class-physical-protocol-v1'
EXECUTION_SCHEMA = runner.SCHEMA
V6, V7 = streaming.hard_input.SCHEMA, streaming.full.SCHEMA
PRIMARY = {'hard_free': (V6, 16384), 'class': (V7, 16384)}
FULL = dict(PRIMARY, class_large=(V7, 65536), class_defensive=(V7, 16384),
            class_intensity=(V7, 16384))


class Bindings:
    """Union closures first; hash each large immutable stream only once.

    Metadata read here is checked immediately and again at final admission.
    JSONL files are only hashed as bytes, never decoded or rescored.
    """
    def __init__(self): self.files = {}; self.metadata = set()

    def bind(self, path, digest=None):
        path = Path(path)
        require(path.is_absolute(), 'Bound path must be absolute')
        path = path.resolve()
        if digest is None: digest = streaming.sha(path)
        streaming._digest(digest, str(path))
        require(str(path) not in self.files or self.files[str(path)] == digest,
                'Conflicting input binding '+str(path))
        self.files[str(path)] = digest
        return path

    def reference(self, value):
        require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Expected a BoundFile')
        return self.bind(value['path'], value['sha256'])

    def add_map(self, values):
        require(type(values) is dict and values, 'Missing input/source closure')
        for path, digest in values.items(): self.bind(path, digest)

    def read(self, reference):
        path = self.reference(reference); raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'Changed metadata '+str(path))
        self.metadata.add(str(path))
        return streaming.loads(raw)

    def finish(self):
        for path, digest in self.files.items():
            require(streaming.sha(path) == digest, 'Changed bound input '+path)


def require_int(value, minimum, label):
    require(type(value) is int and value >= minimum, 'Invalid '+label)
    return value


def require_ref_in_map(reference, files, label):
    require(files.get(str(Path(reference['path']).resolve())) == reference['sha256'], label+' is unbound')


def protocol_inventory(protocol):
    require(protocol['schema'] == PROTOCOL_SCHEMA, 'Wrong scientific protocol')
    scope = protocol['study_scope']
    require(scope in ('primary_only', 'full_declared_campaign'), 'Unknown study scope')
    expected = PRIMARY if scope == 'primary_only' else FULL
    require(protocol['strata'] == statistics.STRATA and protocol['gates'] == statistics.GATES,
            'Changed strata or diagnostic thresholds')
    arms = protocol['arms']
    require(len(arms) == len(expected) and {a['id'] for a in arms} == set(expected),
            'Missing, extra or repeated scientific arms')
    inventory = {}; seeds = []; audit_seeds = []; roots = []
    for arm in arms:
        schema, samples = expected[arm['id']]
        require(arm['source_schema'] == schema and arm['samples'] == samples,
                'Changed schema or draw allocation')
        populations = arm['populations']; ids = [p['id'] for p in populations]
        require(len(populations) == len(set(ids)) == 8 and ids == sorted(ids),
                'Exactly eight canonical independent populations required per arm')
        for population in populations:
            seed = require_int(population['seed'], 0, 'population seed')
            audit_seed = require_int(population['audit_seed'], 0, 'audit seed')
            require(seed < 2**64 and audit_seed < 2**64, 'Seed exceeds u64')
            root = Path(population['directory'])
            require(root.is_absolute(), 'Population directory must be absolute')
            roots.append(str(root.resolve())); seeds.append(seed); audit_seeds.append(audit_seed)
            inventory[(arm['id'], population['id'])] = (arm, population)
    require(len(set(roots)) == len(roots) and len(set(seeds+audit_seeds)) == 2*len(seeds),
            'Reused population directory or scientific/audit seed')
    comparisons = protocol['comparisons']
    expected_pairs = {('hard_free', 'class')}
    if scope == 'full_declared_campaign':
        expected_pairs |= {('class', name) for name in ('class_large','class_defensive','class_intensity')}
    pairs = [(c['left'], c['right']) for c in comparisons]
    require(len(pairs) == len(expected_pairs) and set(pairs) == expected_pairs,
            'Missing, extra or repeated fixed comparisons')
    for comparison in comparisons:
        require(type(comparison['historical_failed_strata']) is list, 'Missing historical stratum inventory')
    return inventory


def completed_execution(plan, status, plan_ref, protocol_ref, bindings):
    require(plan['schema'] == EXECUTION_SCHEMA and plan['maximum_workers'] == 1 and plan['threads'] == 1,
            'Changed execution scope')
    require(status['schema'] == EXECUTION_SCHEMA and status['complete'] is True and status['passed'] is True
            and status['plan_sha256'] == plan_ref['sha256'] and status['active'] is None
            and status['unstarted'] == [] and status['failure'] is None
            and status['maximum_workers'] == status['threads'] == 1
            and status['retries'] == status['replacements'] == 0,
            'Controller is not completely drained and successful')
    root = Path(plan['root']).resolve()
    require(not (root/'execution/failure.json').exists(), 'Controller failure is preserved; admission refused')
    require(status['files'] == plan['files'], 'Controller input closure differs')
    bindings.add_map(plan['files']); require_ref_in_map(protocol_ref, plan['files'], 'Scientific protocol')
    def lifecycle(path):
        return bindings.read(dict(path=str(path),sha256=streaming.sha(path)))
    claim=lifecycle(root/'execution/claim.json')
    require(claim['schema']==EXECUTION_SCHEMA and claim['plan_sha256']==plan_ref['sha256']
            and claim['maximum_workers']==claim['threads']==1 and claim['retries']==claim['replacements']==0,
            'Controller claim differs')
    if 'preparation_receipt' in plan:
        prepared_path=Path(plan['preparation_receipt']).resolve()
        require(prepared_path==root/'preparation.json' and str(prepared_path) not in plan['files']
                and not (root/'preparation-failure.json').exists(), 'Wrong or failed preparation receipt')
        prepared=bindings.read(claim['preparation_receipt'])
        require(Path(claim['preparation_receipt']['path']).resolve()==prepared_path
                and prepared['complete'] is True and prepared['launched'] is False
                and prepared['execution_plan']==plan_ref and prepared['protocol']==protocol_ref,
                'Preparation changed or does not bind this execution')
    jobs, completed = plan['jobs'], status['completed']
    require(jobs and len(jobs) == len(completed) and len({j['id'] for j in jobs}) == len(jobs),
            'Missing or repeated completed execution jobs')
    terminals = {}
    for ordinal,(job, done) in enumerate(zip(jobs, completed)):
        require(all(done[key] == job[key] for key in ('id','population','phase','argv')),
                'Completed job differs from fixed execution order')
        require(done['child_started'] is True and done['child_drained'] is True
                and type(done['returncode']) is int and done['returncode'] == 0 and done['error'] is None
                and done['success'] is True and done['timeout'] is False
                and done['retries'] == done['replacements'] == 0, 'Failed or undrained child')
        # Same published lifecycle records as runner.completed_terminal, with
        # the immutable input closure checked once rather than once per job.
        directory=runner.job_directory(root,ordinal,job)
        require(lifecycle(directory/'success.json')==lifecycle(directory/'exit.json')==done
                and lifecycle(directory/'attempt.json')['job']==job, 'Published lifecycle records differ')
        process=lifecycle(directory/'process.json')
        require(type(done['pid']) is int and done['pid']>0
                and all(process[k]==done[k] for k in ('id','pid','birth_ticks','argv')), 'Owned child identity differs')
        terminal = job['terminal']; path = str(Path(terminal['path']).resolve())
        require(path not in terminals, 'Repeated job terminal')
        reference = done['terminal']
        require(reference['path'] == terminal['path'] and done['success_contract'] == terminal['success_contract'],
                'Completed terminal identity differs')
        value = bindings.read(reference)
        require(value['complete'] is True, 'Incomplete job terminal')
        require(terminal['success_contract'] in ('complete','complete_and_passed'), 'Unknown success contract')
        if terminal['success_contract'] == 'complete_and_passed':
            require(value['passed'] is True, 'Failed job terminal')
        terminals[path] = reference['sha256']
    return terminals


def sources_and_inputs(receipt, execution, bindings):
    bindings.add_map(receipt['input_sha256']); bindings.add_map(receipt['source_sha256'])
    for path, digest in receipt['source_sha256'].items():
        require(execution['files'].get(str(Path(path).resolve())) == digest,
                'Evidence source is outside frozen execution closure')


def check_selection(selection, declaration, population, samples):
    require(selection['schema'] == selected.SELECTION_SCHEMA
            and selection['population'] == population and selection['unconditional_denominator'] == samples
            and selection['maximum_full_geometry_rows'] == 20 and selection['replacements'] == 0,
            'Changed selected population/allocation')
    maxima = selection['decision_maxima']
    require(set(maxima) == set(statistics.DECISIONS), 'Changed maximum-contributor regions')
    for value in maxima.values():
        if value is None: continue
        require(set(value) == {'draw','log_weight'}, 'Malformed maximum-contributor identity')
        require(0 <= require_int(value['draw'], 0, 'maximum draw') < samples
                and type(value['log_weight']) in (int,float) and math.isfinite(value['log_weight']),
                'Invalid selected maximum')
    reasons = selected.selected_inventory(declaration['draw_ids'], maxima)
    rows = selection['rows']; ids = [r['draw'] for r in rows]
    require(ids == sorted(reasons) and selection['full_geometry_rows'] == len(ids),
            'Selected IDs are not exactly unconditional plus regional maxima')
    for row in rows:
        require(row['reasons'] == reasons[row['draw']], 'Changed selection reasons')
        for name in ('sample_record','label_record'):
            record = row[name]
            require(record['line'] == row['draw']+1 and not record.get('eof',False),
                    'Wrong original record line')
            require_int(record['offset'],0,'record offset'); require_int(record['bytes'],1,'record bytes')
            streaming._digest(record['sha256'],'original record')
    return rows


def join_population(arm, population, slot, stat_slot, stat_record, target, target_id,
                    descriptor, execution, terminals, bindings):
    root = Path(population['directory']).resolve(); samples = arm['samples']
    require(stat_slot['id'] == population['id'] and stat_slot['seed'] == population['seed']
            and Path(stat_slot['directory']).resolve() == root, 'Statistics population differs')
    require(all(Path(stat_slot[name]['path']).resolve() == root/(name+'.json') for name in ('manifest','summary')),
            'Statistics producer file path differs')
    manifest, summary = [bindings.read(stat_slot[k]) for k in ('manifest','summary')]
    require(manifest['schema'] == arm['source_schema'] and manifest['samples'] == samples
            and manifest['seed'] == population['seed'] and summary['complete'] is True
            and summary['manifest'] == manifest and not (root/'failure.json').exists(),
            'Producer population identity differs')
    for field, key in [('alpha','importance_uniform_probability'),('lambda_ratio','lambda_ratio')]:
        if field in arm: require(manifest[key] == arm[field], 'Changed proposal '+field)
    if 'guide' in arm:
        require(manifest['importance_guide_sha256'] == arm['guide']['sha256'], 'Changed frozen proposal guide')
    require_ref_in_map(stat_slot['summary'], terminals, 'Producer terminal')
    for name, expected in [('id',population['id']),('seed',population['seed']),('draws',samples),
                          ('unconditional_denominator',samples),('source_schema',arm['source_schema'])]:
        require(stat_record[name] == expected, 'Statistics record identity differs')
    for kind in ('algebra','labels'):
        receipt = bindings.read(stat_slot[kind]); require_ref_in_map(stat_slot[kind],terminals,kind+' terminal')
        sources_and_inputs(receipt,execution,bindings)
        statistics.check_receipt(receipt,root,manifest,summary,target_id,kind,bindings,descriptor)
        require(receipt['source_schema'] == arm['source_schema'], 'Explicit producer schema required')
        record_key = 'algebra_receipt_sha256' if kind == 'algebra' else 'label_receipt_sha256'
        require(stat_record[record_key] == stat_slot[kind]['sha256'], 'Statistics uses another receipt')
        if kind == 'labels':
            endpoint = receipt; counts = endpoint['counts']
            require(counts['attempted'] == samples and endpoint['query_counts'] == dict(capture=samples,
                atomic=samples-counts['capture_rejected'],native=counts['contributing'],contact=counts['contributing']),
                'Independent endpoint coverage differs')
            for name in ('capture_rejected','hard_rejected','region_rejected','shell_rejected'):
                require(counts[name] == summary[name], 'Endpoint rejection accounting differs')
    plan, receipt = [bindings.read(slot[k]) for k in ('selected_plan','selected_receipt')]
    require_ref_in_map(slot['selected_receipt'],terminals,'Selected terminal')
    require(plan['schema'] == selected.PLAN_SCHEMA and receipt['schema'] == selected.SCHEMA
            and receipt['complete'] is True and receipt['passed'] is True
            and receipt['execution_plan_sha256'] == slot['selected_plan']['sha256'], 'Wrong selected completion')
    sources_and_inputs(receipt,execution,bindings)
    require_ref_in_map(slot['selected_plan'],receipt['input_sha256'],'Selected execution plan')
    require({Path(p).name:d for p,d in receipt['source_sha256'].items()} == plan['source_sha256']
            and len(receipt['source_sha256']) == len(plan['source_sha256'])
            and receipt['runtime'] == plan['runtime'], 'Selected source/runtime closure differs')
    require(plan['target'] == target and plan['definition'] == target['native_definition']
            and plan['reference_region'] == target['old_r5_region'], 'Changed selected target references')
    require(receipt['target_and_regions_sha256'] == target_id and receipt['source_schema'] == arm['source_schema'],
            'Changed selected target or schema')
    expected_population = dict(id=population['id'],root=str(root),seed=population['seed'],samples=samples,
        manifest_sha256=stat_slot['manifest']['sha256'],summary_sha256=stat_slot['summary']['sha256'],
        samples_sha256=summary['samples_sha256'],attempts_sha256=summary['attempts_sha256'])
    require(plan['population'] == expected_population and receipt['population'] == expected_population,
            'Selected population or denominator differs')
    require(plan['algebra'] == stat_slot['algebra'] and plan['labels'] == stat_slot['labels']
            and plan['preselection'] == population['preselection'], 'Selected evidence references differ')
    for name in ('algebra','labels','preselection','selection','review'):
        require(receipt[name] == plan[name], 'Selected receipt reference differs: '+name)
        require_ref_in_map(plan[name],receipt['input_sha256'],'Selected '+name)
    declaration = bindings.read(plan['preselection'])
    require_ref_in_map(plan['preselection'],execution['files'],'Pre-draw unconditional selection')
    require(declaration == selected.preselection(population['id'],samples,population['audit_seed']),
            'Unconditional IDs differ from frozen audit seed')
    selection = bindings.read(plan['selection'])
    require(selection['source_schema'] == arm['source_schema']
            and selection['target_and_regions_sha256'] == target_id, 'Selected inventory scope differs')
    for name in ('preselection','algebra','labels'):
        require(selection[name] == plan[name], 'Selection uses different evidence')
    rows = check_selection(selection,declaration,expected_population,samples)
    for row in rows:
        require(Path(row['sample_record']['path']).resolve() == root/'samples.jsonl'
                and Path(row['label_record']['path']).resolve() == Path(endpoint['labels']['path']).resolve(),
                'Original selected record path differs')
    review = bindings.read(plan['review'])
    require_ref_in_map(plan['review'],terminals,'Selection review terminal')
    require(review['complete'] is True and review['passed'] is True
            and review['predeclared_before_first_draw'] is True
            and review['no_previous_full_geometry_audit_of_population'] is True,
            'Missing external chronology/uniqueness review')
    for name in ('preselection','selection','population','limits'):
        require(review[name] == plan[name], 'Selected external review differs')
    count = len(rows); results = receipt['rows']
    require([r['draw'] for r in results] == [r['draw'] for r in rows]
            and receipt['full_geometry_selected_rows'] == count
            and receipt['original_attempt_denominator'] == samples
            and receipt['query_counts'] == dict(full_geometry=count,axis=3*count), 'Selected query accounting differs')
    require(count <= plan['limits']['max_full_geometry_queries'] <= 20
            and 3*count <= plan['limits']['max_axis_queries'] <= 60, 'Selected budget exceeded')
    for value in [receipt,*results]:
        require(value['source_schema'] == arm['source_schema'] and value['hard_line_geometry_certified'] is True
                and value['native_line_geometry_certified'] is (arm['source_schema'] == V7)
                and value['exclusion_contact_line_geometry_certified'] is (arm['source_schema'] == V7),
                'Selected certification overclaims its source schema')
    for name in ('native_identity_origin','compiled_native_binding','shape_compatibility_binding'):
        require(receipt[name] == endpoint[name], 'Selected and endpoint native identities differ')
    if arm['source_schema'] == V6:
        require(plan['compiled_native'] == endpoint['compiled_native_binding']
                and plan['shape_compatibility'] == endpoint['shape_compatibility_binding'], 'Changed external native identity')
    selected.endpoints.check_setup_counts(plan['observer_setup'],receipt['observer_setup_counts'])
    require(receipt['all_row_full_geometry_certified'] is False
            and all(receipt[k] == 0 for k in ('new_pose_draws','new_Poisson_clouds','retries','replacements')),
            'Selected audit changed its allocation/trust boundary')
    return dict(arm=arm['id'],id=population['id'],source_schema=arm['source_schema'],samples=samples,
        selected_rows=count,unconditional_rows=16,selected_receipt_sha256=slot['selected_receipt']['sha256'])


def summarize_diagnostics(report):
    """Reduce already calculated diagnostics; no new estimates or row scans."""
    regional=[]; precision=[]; comparisons=[]; strata=[]
    for arm_id,arm in report['arms'].items():
        for kind in ('Qz','Q0'):
            primary=arm['estimates'][kind]['primary']
            require(set(primary['quality'])==set(statistics.DECISIONS),'Incomplete regional quality inventory')
            for region in statistics.DECISIONS:
                quality=primary['quality'][region]; observed=primary['row_diagnostics'][region]['observed']
                require(type(observed) is bool and type(quality['passed']) is bool,'Malformed regional diagnostic')
                if not observed or not quality['passed']:
                    regional.append(dict(arm=arm_id,kind=kind,region=region,observed=observed,quality=quality))
            contrast=primary['population_statistics']['free_energy_contrast']
            require(type(contrast['observed']) is bool,'Malformed free-energy observation flag')
            halfwidth=contrast['halfwidth_95']
            if contrast['observed']:
                require(type(halfwidth) in (int,float) and math.isfinite(halfwidth) and halfwidth>=0,
                        'Malformed free-energy interval')
            if not contrast['observed'] or halfwidth>statistics.GATES['deltaF_95_halfwidth_max']:
                precision.append(dict(arm=arm_id,kind=kind,observed=contrast['observed'],halfwidth_95=halfwidth))
    expected={(family,index,region) for family,count in statistics.BINS.items()
              for index in range(count) for region in statistics.DECISIONS}
    for pair in report['comparisons']:
        identity=dict(left=pair['left'],right=pair['right'])
        for kind in ('Qz','Q0'):
            block=pair['kinds'][kind]
            require(set(block['regions'])==set(statistics.REGIONS),'Incomplete fixed regional comparisons')
            for region,value in [*block['regions'].items(),('native_minus_competing',block['free_energy_contrast'])]:
                require(type(value['observed']) is bool and type(value['passed']) is bool,'Malformed comparison diagnostic')
                if not value['observed'] or not value['passed']:
                    comparisons.append(dict(**identity,kind=kind,region=region,comparison=value))
            entries=block['strata']; keys=[(s['family'],s['bin'],s['region']) for s in entries]
            require(len(keys)==len(expected) and set(keys)==expected,'Incomplete stratum comparison inventory')
            for entry in entries:
                require(type(entry['material']) is bool and type(entry['historical_failure']) is bool,
                        'Malformed stratum relevance flags')
                value=entry['comparison']
                require(type(value['observed']) is bool and type(value['passed']) is bool,'Malformed stratum comparison')
                if entry['material'] or entry['historical_failure']:
                    item=dict(**identity,kind=kind,**entry)
                    for side in ('left','right'):
                        saved=report['arms'][pair[side]]['estimates'][kind]['strata'][entry['family']][entry['bin']]
                        item[side+'_quality']=saved['quality'][entry['region']]
                        item[side+'_observed']=saved['row_diagnostics'][entry['region']]['observed']
                        item[side+'_parent_mass_fraction']=saved['observed_parent_mass_fraction'][entry['region']]
                    strata.append(item)
    return dict(aggregate_reducer_applied=True,regional_quality_issues=regional,
        free_energy_precision_issues=precision,fixed_comparison_issues=comparisons,
        material_or_historical_stratum_comparisons=strata,
        unstable_stratum_comparisons=[s for s in strata if not s['comparison']['observed'] or not s['comparison']['passed']
            or any(not s[side+'_observed'] or not s[side+'_quality']['passed'] for side in ('left','right'))],
        scope='Existing Qz/Q0 diagnostics only. Empty observations are unresolved, and passing diagnostics do not bound unseen mass.')


def join(plan_path):
    bindings = Bindings(); path = Path(plan_path).resolve()
    plan = bindings.read(dict(path=str(path),sha256=streaming.sha(path)))
    require(plan['schema'] == PLAN_SCHEMA,'Wrong admission plan')
    protocol, execution, status, stat_plan, report = [bindings.read(plan[k]) for k in
        ('protocol','execution_plan','execution_status','statistics_plan','statistics_result')]
    inventory = protocol_inventory(protocol)
    expected_jobs={(f'{arm}-{population}',phase) for arm,population in inventory
                   for phase in ('producer','algebra','labels','selection','geometry')} | {('all','statistics')}
    jobs=[(j['population'],j['phase']) for j in execution['jobs']]
    require(len(jobs)==len(expected_jobs) and set(jobs)==expected_jobs,'Execution phase inventory differs')
    require(Path(plan['execution_status']['path']).resolve() == Path(execution['root']).resolve()/'execution/summary.json',
            'Admission needs the final controller summary')
    terminals = completed_execution(execution,status,plan['execution_plan'],plan['protocol'],bindings)
    source_files = {}
    for source in local_sources(__file__).values():
        source = Path(source).resolve(); digest = streaming.sha(source)
        require(execution['files'].get(str(source)) == digest, 'Admission source is outside frozen execution closure')
        bindings.bind(source,digest); source_files[str(source)] = digest
    require_ref_in_map(plan['statistics_result'],terminals,'Statistics terminal')
    require(stat_plan['schema'] == statistics.PLAN_SCHEMA and report['schema'] == statistics.SCHEMA
            and report['complete'] is True and report['plan_sha256'] == plan['statistics_plan']['sha256'],
            'Wrong completed statistics artifact')
    for name in ('target','target_and_regions_sha256','strata','gates','comparisons'):
        require(stat_plan[name] == protocol[name], 'Statistics declaration differs: '+name)
    target_id = protocol['target_and_regions_sha256']
    require(report['target_and_regions_sha256'] == target_id and report['gates'] == protocol['gates'],
            'Statistics target or gates differ')
    sources_and_inputs(report,execution,bindings)
    require_ref_in_map(plan['statistics_plan'],report['input_sha256'],'Statistics plan')
    require({a['id'] for a in stat_plan['arms']} == {a['id'] for a in protocol['arms']}
            and len(stat_plan['arms']) == len(protocol['arms']) == len(report['arms']), 'Statistics arm inventory differs')
    slots = {(s['arm'],s['id']):s for s in plan['populations']}
    require(len(slots) == len(plan['populations']) and set(slots) == set(inventory), 'Admission population inventory differs')
    stats = {a['id']:a for a in stat_plan['arms']}; accepted = []
    for arm in protocol['arms']:
        stat_arm = stats[arm['id']]; records = report['arms'][arm['id']]['populations']
        require(stat_arm['samples'] == arm['samples'] and len(stat_arm['populations']) == len(records) == 8,
                'Statistics population allocation differs')
        require([p['id'] for p in stat_arm['populations']] == [p['id'] for p in arm['populations']]
                == [p['id'] for p in records], 'Statistics population order/inventory differs')
        for population, stat_slot, record in zip(arm['populations'],stat_arm['populations'],records):
            accepted.append(join_population(arm,population,slots[(arm['id'],population['id'])],stat_slot,record,
                protocol['target'],target_id,report['target_descriptor'],execution,terminals,bindings))
    require([(c['left'],c['right']) for c in report['comparisons']] ==
            [(c['left'],c['right']) for c in protocol['comparisons']], 'Statistics comparison inventory differs')
    diagnostics=summarize_diagnostics(report)
    bindings.finish()
    return dict(schema=SCHEMA,complete=True,implementation_admission_passed=True,
        execution_lifecycle_passed=True,selected_full_geometry_gate_satisfied=True,
        plan_sha256=bindings.files[str(path)],protocol=plan['protocol'],statistics_result=plan['statistics_result'],
        target_and_regions_sha256=target_id,study_scope=protocol['study_scope'],populations=accepted,
        total_unconditional_attempts=sum(p['samples'] for p in accepted),
        selected_full_geometry_rows=sum(p['selected_rows'] for p in accepted),input_sha256=bindings.files,
        source_sha256=source_files,
        statistical_diagnostics=dict(source=plan['statistics_result'],**diagnostics),
        missing_studies=(['population_size','defensive_probability','cloud_intensity']
                         if protocol['study_scope']=='primary_only' else []),
        regional_convergence_established=False,unseen_support_verdict='unresolved',
        physical_campaign_gate_open=False,full_vessel_gate_open=False,assembly_gate_open=False,
        new_pose_draws=0,new_Poisson_clouds=0,new_atom_geometry_queries=0,parsed_scientific_rows=0,
        scope='Bound implementation evidence only. Selected maxima rely on the frozen worker all-attempt scan; '
              'pre-draw chronology relies on external review. Neither sampling convergence, unseen mass nor assembly follows.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),'Admission output must be new')
    streaming._write(args.out,join(args.plan))
