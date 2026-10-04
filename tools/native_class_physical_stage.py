#!/usr/bin/env python3
"""Materialize and execute frozen physical-analysis stages after their inputs exist.

Future output hashes are not knowable at campaign preparation. This worker fills
only those bindings, using successful predecessor receipts under the immutable
controller plan. Proposal, allocation, classifier, strata, limits and the 16
unconditional audit IDs were fixed before sampling. Selection of at most four
additional contributors follows the existing, frozen rule. No failed population
is removed or replaced. The final campaign admission is a separate operation.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
from pathlib import Path

import native_class_line_physical_algebra_audit as streaming
import native_class_physical_labels as endpoints
import native_class_selected_geometry as selected
import analyze_native_class_physical_populations as statistics
import admit_native_class_physical_campaign as admission
import run_native_class_physical_campaign as controller
from analyze_mobile_native_pocket import local_sources

require, read, sha, write = streaming.require, streaming.read, streaming.sha, streaming._write
PROTOCOL_SCHEMA = 'native-class-physical-protocol-v1'
POLICY = 'bound-predecessors-v1'
PHASES = ('labels', 'selection', 'geometry', 'statistics', 'admission')


def paths(root, arm=None, population=None):
    root = Path(root).resolve()
    if arm is None:
        return dict(statistics_plan=root/'analysis/statistics-plan.json',
                    statistics=root/'analysis/statistics.json',
                    admission_plan=root/'analysis/admission-plan.json', admission=root/'analysis/admission.json')
    folder = root/'analysis'/f'{arm}-{population}'
    return {key: folder/name for key, name in dict(
        algebra='algebra.json', labels_plan='labels-plan.json', labels='labels.json',
        labels_data='labels.jsonl', labels_journal='labels.journal.jsonl', labels_failure='labels.failure.json',
        selection_plan='selection-plan.json', selection='selection.json', selection_review='selection-review.json',
        geometry_plan='geometry-plan.json', geometry='geometry.json',
        geometry_journal='geometry.journal.jsonl', geometry_failure='geometry.failure.json').items()}


class Inputs:
    def __init__(self): self.files = {}

    def bind(self, path, digest=None):
        path = Path(path).resolve(); value = sha(path)
        require(digest is None or digest == value, 'Bound stage input changed: '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == value, 'Input changed within stage')
        self.files[str(path)] = value
        return path

    def ref(self, path):
        path = self.bind(path)
        return dict(path=str(path), sha256=self.files[str(path)])

    def reference(self, item):
        require(type(item) is dict and set(item) == {'path', 'sha256'}, 'Expected exact BoundFile')
        require(Path(item['path']).is_absolute(), 'Stage input reference must be absolute')
        return self.bind(item['path'], item['sha256'])

    def recheck(self):
        for path, digest in self.files.items(): require(sha(path) == digest, 'Stage input changed: '+path)


def load_protocol(path, digest, bindings):
    protocol_path = bindings.bind(path, digest); protocol = read(protocol_path)
    require(protocol['schema'] == PROTOCOL_SCHEMA and protocol['stage_materialization_policy'] == POLICY,
            'Unsupported frozen campaign/materialization contract')
    root = Path(protocol['root']).resolve()
    require(protocol_path == root/'protocol.json', 'Protocol is outside its campaign root')
    require(Path(__file__).resolve().parent == Path(protocol['code_directory']).resolve(),
            'Stage must use the frozen code directory')
    for name, value in protocol['files'].items(): bindings.bind(name, value)
    require(protocol['runtime'] == streaming.runtime_identity(), 'Frozen runtime differs')
    closure = local_sources(__file__)
    require(set(closure) <= set(protocol['source_sha256']), 'Missing stage source closure')
    for name, source in closure.items():
        require(source.parent == Path(protocol['code_directory']).resolve(), 'Mixed source directories')
        bindings.bind(source, protocol['source_sha256'][name])
    execution_path = bindings.bind(root/'execution-plan.json'); execution = read(execution_path)
    require(execution['schema'] == 'native-class-physical-execution-v1'
            and execution['root'] == str(root) and execution['maximum_workers'] == execution['threads'] == 1,
            'Wrong campaign controller allocation')
    require(execution['files'].get(str(protocol_path)) == digest, 'Protocol was not frozen by controller')
    for name, value in protocol['files'].items():
        require(execution['files'].get(name) == value, 'Controller omitted frozen scientific input')
    claim = read(bindings.bind(root/'execution/claim.json'))
    require(claim['plan_sha256'] == sha(execution_path), 'No matching pre-draw execution authority')
    return protocol, execution


def find_population(protocol, key):
    found = [(arm, pop) for arm in protocol['arms'] for pop in arm['populations']
             if f"{arm['id']}-{pop['id']}" == key]
    require(len(found) == 1, 'Unknown/repeated population key')
    return found[0]


def predecessor(root, execution, job_id, terminal, bindings):
    """Consume the controller's successful exit, not a loose output file."""
    matches = [(i, j) for i, j in enumerate(execution['jobs']) if j['id'] == job_id]
    require(len(matches) == 1, 'Missing/repeated predecessor stage')
    ordinal, job = matches[0]
    folder = Path(root)/'execution/jobs'/f'{ordinal:03d}-{job_id}'
    for name in ('attempt', 'process', 'exit', 'success'):
        bindings.bind(folder/(name+'.json'))
    reference = controller.completed_terminal(root, execution, job_id)
    path = bindings.reference(reference)
    require(path == Path(terminal).resolve() == Path(job['terminal']['path']).resolve(), 'Predecessor terminal differs')
    value = read(path)
    require(value['complete'] is True, 'Predecessor terminal incomplete')
    if job['terminal']['success_contract'] == 'complete_and_passed':
        require(value['passed'] is True, 'Predecessor validation failed')
    return reference, value


def verify_live_authority(protocol, execution, population, phase):
    """A physical-analysis worker must be the current frozen controller child."""
    root = Path(protocol['root']); claim = read(root/'execution/claim.json')
    require(type(claim['pid']) is int and claim['pid'] == os.getppid(),
            'Stage was not launched by its frozen controller')
    stat = Path(f"/proc/{claim['pid']}/stat").read_text().rsplit(')', 1)[1].split()
    require(int(stat[19]) == claim['birth_ticks'] and stat[0] not in ('Z', 'X'),
            'Controller process identity changed')
    status = read(root/'execution/status.json')
    require(status['plan_sha256'] == claim['plan_sha256'] and status['failure'] is None,
            'Controller status is failed or belongs to another plan')
    active = status['active']
    require(type(active) is dict and active['population'] == population and active['phase'] == phase,
            'This stage is not the active frozen allocation')
    ordinal = active['ordinal']
    require(type(ordinal) is int and 0 <= ordinal < len(execution['jobs']), 'Invalid active job index')
    job = execution['jobs'][ordinal]
    require(all(active[key] == job[key] for key in ('id', 'population', 'phase')), 'Active job differs from frozen order')


def population_binding(protocol, execution, arm, pop, bindings):
    directory = Path(pop['directory']).resolve(); key = f"{arm['id']}-{pop['id']}"
    _, summary = predecessor(protocol['root'], execution, key+'-producer', directory/'summary.json', bindings)
    manifest_path = bindings.bind(directory/'manifest.json'); manifest = read(manifest_path)
    require(summary['manifest'] == manifest and summary['samples'] == manifest['samples'] == arm['samples']
            and manifest['seed'] == pop['seed'] and manifest['schema'] == arm['source_schema'],
            'Producer population/format differs from frozen allocation')
    require(not (directory/'failure.json').exists(), 'Producer failure is present')
    require(manifest['region_sha256'] == protocol['target']['region']['sha256']
            and manifest['activity'] == .035 and manifest['cloud_replicates'] == 2
            and manifest['importance_uniform_probability'] == arm['alpha']
            and manifest['lambda_ratio'] == arm['lambda_ratio']
            and manifest['importance_guide_sha256'] == arm['guide']['sha256'],
            'Producer changed target, guide, defensive probability or cloud allocation')
    result = dict(id=pop['id'], root=str(directory), samples=arm['samples'], seed=pop['seed'],
                  manifest_sha256=sha(manifest_path), summary_sha256=sha(directory/'summary.json'))
    for name in ('samples', 'attempts'):
        result[name+'_sha256'] = summary[name+'_sha256']
        bindings.bind(directory/(name+'.jsonl'), result[name+'_sha256'])
    return result


def source_map(protocol, module):
    sources = local_sources(module.__file__)
    require(set(sources) <= set(protocol['source_sha256']), 'Worker sources absent from pre-draw freeze')
    return {name: protocol['source_sha256'][name] for name in sources}


def worker_limits(protocol, phase, count):
    declared = protocol['phase_limits'][phase]
    result = dict(cpu_seconds=declared['cpu_limit_seconds'], wall_seconds=declared['wall_limit_seconds'],
                  max_rows=count, max_record_bytes=declared['max_record_bytes'])
    result.update({f'max_{name}_queries': count for name in endpoints.QUERY_ROLES})
    if phase == 'geometry': result.update(max_full_geometry_queries=20, max_axis_queries=60)
    return result


def base_plan(protocol, population):
    return dict(population=population, definition=protocol['target']['native_definition'],
                reference_region=protocol['target']['old_r5_region'], strata=protocol['strata_file'],
                observer_setup=protocol['observer_setup'], runtime=protocol['runtime'],
                compiled_native=protocol['native_identity']['compiled_native'],
                shape_compatibility=protocol['native_identity']['shape_compatibility'])


def materialize_labels(protocol, execution, arm, pop, bindings):
    key = f"{arm['id']}-{pop['id']}"; output = paths(protocol['root'], arm['id'], pop['id'])
    population = population_binding(protocol, execution, arm, pop, bindings)
    algebra_ref, _ = predecessor(protocol['root'], execution, key+'-algebra', output['algebra'], bindings)
    plan = dict(base_plan(protocol, population), schema=endpoints.PLAN_SCHEMA,
        limits=worker_limits(protocol, 'labels', arm['samples']),
        source_sha256=source_map(protocol, endpoints),
        output={key: str(output[name]) for key, name in [('receipt', 'labels'), ('labels', 'labels_data'),
                   ('journal', 'labels_journal'), ('failure', 'labels_failure')]},
        predecessor_algebra=algebra_ref, materialization_policy=POLICY)
    bindings.recheck(); write(output['labels_plan'], plan)
    return output['labels_plan']


def materialize_selection(protocol, execution, arm, pop, bindings):
    key = f"{arm['id']}-{pop['id']}"; output = paths(protocol['root'], arm['id'], pop['id'])
    for name in ('geometry_plan', 'geometry', 'geometry_journal', 'geometry_failure'):
        require(not output[name].exists(), 'Prior full-geometry stage output is present')
    population = population_binding(protocol, execution, arm, pop, bindings)
    algebra, _ = predecessor(protocol['root'], execution, key+'-algebra', output['algebra'], bindings)
    labels, _ = predecessor(protocol['root'], execution, key+'-labels', output['labels'], bindings)
    declaration = read(bindings.reference(pop['preselection']))
    require(declaration == selected.preselection(pop['id'], arm['samples'], pop['audit_seed']),
            'Unconditional audit selection changed')
    require(execution['files'].get(pop['preselection']['path']) == pop['preselection']['sha256'],
            'Audit IDs were not frozen before sampling')
    plan = dict(base_plan(protocol, population), schema=selected.PLAN_SCHEMA,
        target=protocol['target'], algebra=algebra, labels=labels, preselection=pop['preselection'],
        limits=worker_limits(protocol, 'geometry', arm['samples']), source_sha256=source_map(protocol, selected),
        output={key: str(output[name]) for key, name in [('receipt', 'geometry'),
                   ('journal', 'geometry_journal'), ('failure', 'geometry_failure')]},
        materialization_policy=POLICY)
    bindings.recheck(); write(output['selection_plan'], plan)
    selection = selected.select(plan)
    # The selected worker compares exact selection objects. Do not add execution
    # fields to that object; keep a distinct terminal envelope instead.
    source_plan = bindings.ref(output['selection_plan'])
    write(output['selection'], selection)
    review = dict(schema='native-class-deterministic-selection-review-v1', complete=True, passed=True,
        preselection=pop['preselection'], selection=bindings.ref(output['selection']), population=population,
        limits=plan['limits'], predeclared_before_first_draw=True,
        no_previous_full_geometry_audit_of_population=True,
        authority='Exclusive frozen controller; deterministic materialization allowed before first draw.',
        materialization_policy=POLICY, selection_plan=source_plan,
        protocol=bindings.ref(Path(protocol['root'])/'protocol.json'),
        execution_plan=bindings.ref(Path(protocol['root'])/'execution-plan.json'))
    bindings.recheck(); write(output['selection_review'], review)
    return review


def materialize_geometry(protocol, execution, arm, pop, bindings):
    key = f"{arm['id']}-{pop['id']}"; output = paths(protocol['root'], arm['id'], pop['id'])
    review_ref, review = predecessor(protocol['root'], execution, key+'-selection', output['selection_review'], bindings)
    plan_path = bindings.reference(review['selection_plan']); plan = read(plan_path)
    require(plan_path == output['selection_plan'] and review['protocol'] == bindings.ref(Path(protocol['root'])/'protocol.json'),
            'Selection belongs to another frozen protocol')
    selection_path = bindings.reference(review['selection'])
    require(selection_path == output['selection'] and review['preselection'] == pop['preselection'],
            'Selection changed between stages')
    require(plan['population']['seed'] == pop['seed'] and plan['population']['samples'] == arm['samples'],
            'Selection allocation changed')
    plan.update(selection=review['selection'], review=review_ref)
    bindings.recheck(); write(output['geometry_plan'], plan)
    return output['geometry_plan']


def materialize_statistics(protocol, execution, bindings):
    arms = []
    for arm in protocol['arms']:
        slots = []
        for pop in arm['populations']:
            key = f"{arm['id']}-{pop['id']}"; output = paths(protocol['root'], arm['id'], pop['id'])
            population_binding(protocol, execution, arm, pop, bindings)
            refs = {}
            for phase, name in [('algebra', 'algebra'), ('labels', 'labels'), ('geometry', 'geometry')]:
                refs[phase], _ = predecessor(protocol['root'], execution, key+'-'+phase, output[name], bindings)
            root = Path(pop['directory'])
            slots.append(dict(id=pop['id'], seed=pop['seed'], directory=str(root),
                manifest=bindings.ref(root/'manifest.json'), summary=bindings.ref(root/'summary.json'),
                algebra=refs['algebra'], labels=refs['labels']))
        arms.append(dict(id=arm['id'], samples=arm['samples'], populations=slots))
    plan = dict(schema=statistics.PLAN_SCHEMA, target=protocol['target'],
        target_and_regions_sha256=protocol['target_and_regions_sha256'], strata=protocol['strata'],
        gates=protocol['gates'], arms=arms, comparisons=protocol['comparisons'])
    bindings.recheck(); path = paths(protocol['root'])['statistics_plan']; write(path, plan)
    return path


def materialize_admission(protocol, execution, bindings):
    """Post-controller handoff; never put this phase inside that controller."""
    root = Path(protocol['root']); status_path = bindings.bind(root/'execution/summary.json')
    status = read(status_path)
    require(status['complete'] is True and status['passed'] is True
            and status['active'] is None and status['unstarted'] == [] and status['failure'] is None
            and status['plan_sha256'] == sha(root/'execution-plan.json'), 'Controller must finish and drain first')
    require(not any(job['phase'] == 'admission' for job in execution['jobs']), 'Circular final-admission dependency')
    require(len(status['completed']) == len(execution['jobs']), 'Missing completed execution stages')
    output = paths(root); populations = []
    for arm in protocol['arms']:
        for pop in arm['populations']:
            per = paths(root, arm['id'], pop['id'])
            populations.append(dict(arm=arm['id'], id=pop['id'],
                selected_plan=bindings.ref(per['geometry_plan']), selected_receipt=bindings.ref(per['geometry'])))
    plan = dict(schema=admission.PLAN_SCHEMA, protocol=bindings.ref(root/'protocol.json'),
        execution_plan=bindings.ref(root/'execution-plan.json'), execution_status=bindings.ref(status_path),
        statistics_plan=bindings.ref(output['statistics_plan']), statistics_result=bindings.ref(output['statistics']),
        populations=populations)
    bindings.recheck(); write(output['admission_plan'], plan)
    return output['admission_plan']


def run(protocol_path, *, protocol_sha256, population, phase):
    require(phase in PHASES, 'Unsupported analysis phase')
    bindings = Inputs(); protocol, execution = load_protocol(protocol_path, protocol_sha256, bindings)
    if phase == 'admission':
        require(population == 'all', 'Final admission must retain every population')
        plan_path = materialize_admission(protocol, execution, bindings)
        result = admission.join(plan_path)
        write(paths(protocol['root'])['admission'], result)
        return result
    verify_live_authority(protocol, execution, population, phase)
    if phase == 'statistics':
        require(population == 'all', 'Statistics must retain the full frozen population inventory')
        plan_path = materialize_statistics(protocol, execution, bindings)
        result = statistics.analyze(plan_path)
        write(paths(protocol['root'])['statistics'], result)
        return result
    arm, pop = find_population(protocol, population)
    if phase == 'labels':
        plan_path = materialize_labels(protocol, execution, arm, pop, bindings)
        return endpoints.run(plan_path, plan_sha256=sha(plan_path))
    if phase == 'selection': return materialize_selection(protocol, execution, arm, pop, bindings)
    plan_path = materialize_geometry(protocol, execution, arm, pop, bindings)
    return selected.run(plan_path, plan_sha256=sha(plan_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--population', required=True)
    parser.add_argument('--phase', choices=PHASES, required=True)
    args = parser.parse_args()
    result = run(args.protocol, protocol_sha256=args.protocol_sha256, population=args.population, phase=args.phase)
    print(json.dumps(dict(complete=result['complete'], phase=args.phase)))
