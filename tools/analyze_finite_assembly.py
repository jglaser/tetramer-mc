#!/usr/bin/env python3
"""Audit one completed frozen, unbiased run and measure finite assembly.

No run is launched and no production window is chosen here. Every saved state
is classified, including repeated retained poses and unresolved native graphs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from analyze_contact_efficiency import (MATCHED_SCHEDULE, digest, load_bound_file, native_input_bindings,
    physical_identity, read, recheck_bindings, require, save, sha,
    validate_effective_config, validate_frames)
from finite_assembly_observer import FiniteAssemblyObserver, REGIONS, summarize_series
from native_contact_regions import NativeContactRegions


SCHEMA = 'finite-assembly-observation-v1'
PLAN_SCHEMA = 'finite-assembly-observation-window-v1'
RUNTIME_SOURCES = ('analyze_finite_assembly.py', 'finite_assembly_observer.py',
    'analyze_contact_efficiency.py', 'mobile_posterior_metrics.py',
    'contact_benchmark_contract.py', 'native_contact_regions.py', 'native_graph_consistency.py')


def observe(run, definition_path, plan_path, out):
    require(__debug__, 'Run with PYTHONOPTIMIZE=0 for checked native predicates')
    run, definition_path, plan_path, out = map(lambda p: Path(p).resolve(),
                                             (run, definition_path, plan_path, out))
    require(not out.exists(), 'Fresh analysis directory required')
    started = time.process_time()
    definition, plan = read(definition_path), read(plan_path)
    require(definition['schema'] == 'finite-assembly-observer-definition-v1'
            and definition['reference_size'] == 8, 'Unsupported assembly observer definition')
    require(plan['schema'] == PLAN_SCHEMA and plan['definition_sha256'] == sha(definition_path),
            'Window plan does not bind the observer definition')
    require(set(plan) <= {'schema', 'definition_sha256', 'burn_sweep', 'end_sweep',
                         'preparation_id', 'proposal_arm', 'benchmark_contract'}, 'Unsupported window plan field')
    for key in ('preparation_id', 'proposal_arm'):
        require(isinstance(plan[key], str) and plan[key], 'Missing '+key)
    # The finite-system definition is a frozen law, not an arbitrary expression
    # evaluator. Authenticate the rules against the preparation implementation.
    from prepare_finite_assembly_observer import REGION_RULES
    require(definition['regions'] == REGION_RULES, 'Region definitions differ from implementation')
    dependencies = {str(definition_path): sha(definition_path), str(plan_path): sha(plan_path)}
    implementation = {}
    for relative, expected in definition['source_sha256'].items():
        path = (definition_path.parent/relative).resolve()
        require(sha(path) == expected, 'Frozen observer source changed: '+str(path))
        dependencies[str(path)] = expected
        if path.name in RUNTIME_SOURCES+('prepare_finite_assembly_observer.py',):
            actual = Path(__file__).with_name(path.name).resolve()
            require(sha(actual) == expected, 'Runtime observer differs from frozen source: '+str(actual))
            dependencies[str(actual)] = expected
            implementation[path.name] = expected
    require(set(RUNTIME_SOURCES+('prepare_finite_assembly_observer.py',)) <= set(implementation),
            'Incomplete observer implementation closure')
    names = ('config.json', 'manifest.json', 'summary.json', 'checkpoint.json',
             'trajectory.jsonl', 'provenance/input-config.json', 'provenance/shape.json',
             'provenance/source-bundle.json')
    bindings = {name: sha(run/name) for name in names}
    config, manifest, summary, checkpoint = [read(run/name) for name in names[:4]]
    for field, name in [('config_sha256', 'provenance/input-config.json'),
                        ('shape_sha256', 'provenance/shape.json')]:
        require(manifest[field] == summary[field] == checkpoint[field] == bindings[name],
                'Run input provenance mismatch: '+field)
    require(manifest['source_bundle_sha256'] == bindings['provenance/source-bundle.json'],
            'Run source bundle changed')
    require(manifest.get('model_sha256') == summary.get('model_sha256') == checkpoint.get('model_sha256'),
            'Run proposal identity differs')
    if manifest.get('model_sha256') is not None:
        name = 'provenance/frozen-relative-model.json'
        require(sha(run/name) == manifest['model_sha256'], 'Run frozen proposal changed')
        bindings[name] = sha(run/name)
    shape = read(run/'provenance/shape.json')
    validate_effective_config(config, read(run/'provenance/input-config.json'), manifest, summary, shape)
    frames = [json.loads(line) for line in (run/'trajectory.jsonl').read_text().splitlines()]
    window = validate_frames(frames, config, summary, manifest, checkpoint,
                             plan['burn_sweep'], plan['end_sweep'])
    identity = physical_identity(config, manifest['shape_sha256'])
    require(identity == definition['physical_identity']
            and digest(identity) == definition['physical_identity_sha256'],
            'Different physical target, mobility, boundary or size')
    patch_path = load_bound_file(definition['patch_map'], definition_path.parent)
    native_path = load_bound_file(definition['native_definition'], definition_path.parent)
    patch = read(patch_path)
    require(patch['schema'] == 'body-frame-atom-patch-map-v1'
            and patch['shape_sha256'] == manifest['shape_sha256'], 'Patch/shape identity differs')
    dependencies[str(patch_path)] = sha(patch_path)
    dependencies.update(native_input_bindings(native_path))
    native = NativeContactRegions(native_path)
    require(native.shape_sha256 == manifest['shape_sha256'], 'Different native shape')
    observer = FiniteAssemblyObserver(shape, patch['atom_patch_ids'], config, native)
    observations = [observer.classify(frame['poses']) for frame in frames]
    result = summarize_series(observations, frames, window)
    result.update(schema=SCHEMA, complete=True, run=str(run),
        physical_identity=identity, definition_sha256=sha(definition_path),
        region_definition_sha256=digest(definition['regions']),
        observer_definition_sha256=digest(definition),
        plan_sha256=sha(plan_path), implementation_sha256=implementation,
        source_sha256=bindings, dependency_sha256=dependencies,
        initialization=dict(master_seed=config['seed'], preparation_id=plan['preparation_id'],
            proposal_arm=plan['proposal_arm'], initial_poses_sha256=digest(frames[0]['poses']),
            invocation_initial_sweep=summary['initial_sweep'], resume=manifest.get('resume')),
        schedule={key: config[key] for key in MATCHED_SCHEDULE},
        proposal=dict(method=config['method'], model_sha256=manifest.get('model_sha256'),
            frozen_posterior=config.get('frozen_posterior'),
            learned_uniform_weight=config.get('learned_uniform_weight')),
        observer_cpu_seconds=time.process_time()-started,
        frozen_bias_supported=False,
        physical_reference=None,
        equilibrium_scope='No independent finite-system probabilities supplied. Region returns remain unassessed; '
                         'conditional scaffold weights do not supply this reference.')
    result['window_attempts'] = {kind: frames[window['indices'][-1]]['counts'][kind]['attempted']
        -frames[window['indices'][0]]['counts'][kind]['attempted'] for kind in ('local', 'global')}
    result['benchmark_contract'] = None
    if plan.get('benchmark_contract') is not None:
        from contact_benchmark_contract import validate_report
        path = load_bound_file(plan['benchmark_contract'], plan_path.parent)
        contract = read(path)
        dependencies[str(path)] = sha(path)
        result['benchmark_contract'] = dict(path=str(path), sha256=sha(path),
            content=contract, content_sha256=digest(contract))
        validate_report(result, contract)
    recheck_bindings({str(run/name): value for name, value in bindings.items()},
                    'Run changed during observation')
    recheck_bindings(dependencies, 'Definition/source changed during observation')
    out.mkdir(parents=True)
    save(out/'analysis.json', result)
    with (out/'observations.jsonl').open('x') as stream:
        for frame, obs in zip(frames, observations):
            stream.write(json.dumps(dict(sweep=frame['sweep'], **obs), allow_nan=False)+'\n')
    save(out/'freeze.json', dict(schema='finite-assembly-observation-freeze-v1',
        implementation_sha256=implementation,
        files={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'definition', 'plan', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = observe(args.run, args.definition, args.plan, args.out)
    print(json.dumps(dict(complete=result['complete'], samples=result['fingerprint']['samples'],
                         observer_cpu_seconds=result['observer_cpu_seconds']), indent=2))
