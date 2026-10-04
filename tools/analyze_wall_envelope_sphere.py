#!/usr/bin/env python3
"""Analyze one completed wall-envelope allocation without draws or geometry."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys

import analyze_native_class_vessel_sphere as common
import prepare_wall_envelope_sphere as preparation
from prepare_wall_envelope_sphere import read, require, sha, write


def check_preparation(root, plan, execution, ledger):
    ledger.frozen(root)
    require(plan['schema'] == preparation.SCHEMA and plan['root'] == str(root)
        and plan['launched'] is False and plan['preparation_only'] is True
        and plan['criteria'] == preparation.CRITERIA and plan['resource_limits'] == preparation.RESOURCE_LIMITS
        and plan['physical'] == preparation.PHYSICAL and plan['proposal'] == preparation.PROPOSAL
        and plan['total_attempts'] == preparation.TOTAL
        and plan['maximum_clouds'] == preparation.MAXIMUM_CLOUDS
        and plan['populations_per_activity'] == preparation.POPULATIONS
        and plan['samples_per_population_by_activity'] == preparation.DRAWS_BY_ACTIVITY
        and plan['activities'] == list(preparation.ACTIVITIES)
        and plan['maximum_workers'] == plan['threads'] == 1
        and plan['retries'] == plan['replacements'] == plan['extensions'] == 0,
        'Changed wall-envelope allocation, physical law or criteria')
    require(plan['jobs'] == preparation.jobs_for(root,plan['python']), 'Changed fixed commands/seeds')
    require(plan['python'] == sys.executable
        and plan['python_sha256'] == sha(Path(sys.executable).resolve()), 'Python runtime changed')
    expected_paths = dict(executable=root/'common/basin-normalizer',source_bundle=root/'common/source-bundle.json',
        analytic_reference=root/'analytic-reference.json',allocation=root/'allocation.json')
    for name,path in expected_paths.items():
        require(plan[name]['path'] == str(path), 'Bound artifact path differs '+name)
        ledger.bind(path,plan[name]['sha256'])
    require(plan['allocation']['sha256'] == preparation.ALLOCATION_SHA, 'Allocation binding differs')
    allocation = preparation.validate_allocation(read(root/'allocation.json'))
    require(plan['allocation']['joint_point_gate_failure_bound'] == allocation['joint_point_gate_failure_bound'],
        'Allocation assurance differs')
    require(read(root/'analytic-reference.json') == preparation.analytic_reference(), 'Analytic target or quadrature changed')
    preparation.check_input_copies(root/'inputs')
    expected = []
    for job in plan['jobs']:
        expected.extend([(job['argv'],job['producer_terminal'],'producer'),
                         (job['audit_argv'],job['audit_terminal'],'geometry')])
    require(len(execution['jobs']) == 17, 'Wrong execution stage count')
    for (argv,terminal,phase),actual in zip(expected,execution['jobs'][:16]):
        require(actual['argv'] == argv and actual['terminal'] == terminal and actual['phase'] == phase,
            'Execution order/commands differ from frozen wall-envelope job')
    for actual in execution['jobs']:
        require(actual['phase'] in preparation.RESOURCE_LIMITS
            and all(actual[k] == v for k,v in preparation.RESOURCE_LIMITS[actual['phase']].items()),
            'Execution resource limit differs')
    for relative,digest in plan['files_sha256'].items():
        path = (root/relative).resolve(); require(path.is_relative_to(root), 'Unsafe preparation path')
        ledger.bind(path,digest)


def check_manifest(manifest, job):
    require(manifest['schema'] == 8 and manifest['pre_envelope_schema'] == 7
        and manifest['vessel_uniform_schema'] == 'one-atom-wall-envelope-v1'
        and manifest['outer_mixture_schema'] == 'full-vessel-native-class-line-half-mixture-v1'
        and manifest['outer_vessel_probability'] == .5 and manifest['uniform_probability'] == .4
        and manifest['latent_defensive_uniform_probability'] == .5
        and manifest['latent_guide_schema'] == 'defensive-native-class-line-guide-v1'
        and manifest['cloud_replicates'] == 2 and manifest['samples'] == job['samples']
        and manifest['activity'] == job['activity'] and manifest['seed'] == job['seed'],
        'Changed schema-8 mixture or population')
    expected = dict(atom_index=0,atom_center=[0.,0.,0.],atom_radius=preparation.CORE,
        wall_center=preparation.CENTER,wall_radius=preparation.WALL,envelope_radius=2.7)
    witness = manifest['vessel_uniform_envelope']
    require(set(witness) == set(expected)|{'log_volume'} and all(witness[k] == v for k,v in expected.items()),
        'Changed one-atom sphere envelope witness')
    common.close(witness['log_volume'],math.log(4*math.pi*2.7**3/3),'Envelope normalization differs')
    common.close(manifest['lambda'],16*job['activity'] if job['activity'] > 0 else 1.,'Auxiliary intensity differs')
    require(manifest['atomic_wall'] == dict(center=preparation.CENTER,radius=preparation.WALL)
        and manifest['density_measure'] == 'Lebesgue center volume times normalized SO(3) Haar measure'
        and manifest['latent_reference_ball_is_target_restriction'] is False
        and manifest['latent_source_capture'] == dict(center=preparation.CENTER,radius=preparation.SOURCE_CAPTURE,
            conditions_guide=True,restricts_target=False)
        and manifest['bath_wall_permeable'] is True, 'Changed physical domain or source capture')


def analyze(root, output):
    root,output = Path(root).resolve(),Path(output).resolve()
    require(not output.exists(), 'Fresh statistics output required')
    ledger = common.Ledger()
    execution,terminals = common.authenticated_predecessors(root,output,ledger,analyzer_file=__file__)
    plan = read(ledger.bind(root/'plan.json')); check_preparation(root,plan,execution,ledger)
    reference = read(ledger.bind(plan['analytic_reference']['path'],plan['analytic_reference']['sha256']))
    output.parent.mkdir(parents=True,exist_ok=True); companions = output.parent/'exact-weights'; companions.mkdir()
    populations = []
    for i,job in enumerate(plan['jobs']):
        require(terminals[2*i]['path'] == job['producer_terminal']['path']
            and terminals[2*i+1]['path'] == job['audit'], 'Predecessor order differs')
        audit = read(ledger.bind(job['audit'],terminals[2*i+1]['sha256']))
        manifest = read(ledger.bind(Path(job['directory'])/'manifest.json')); check_manifest(manifest,job)
        populations.append(common.population(job,plan,audit,ledger,companions/(job['id']+'.jsonl'),
            expected_manifest_schema=8))
    require(sum(p['samples'] for p in populations) == preparation.TOTAL
        and sum(p['counts']['clouds'] for p in populations) <= preparation.MAXIMUM_CLOUDS, 'Allocation exceeded')
    for z in preparation.ACTIVITIES:
        selected = [p for p in populations if p['activity'] == z]
        require(len(selected) == preparation.POPULATIONS
            and {p['samples'] for p in selected} == {preparation.DRAWS_BY_ACTIVITY[str(z)]},
            'Activity population denominator differs')
    groups,checks = common.aggregate(populations,reference)
    exercised = all(sum(p['counts'][key] for p in populations) > 0 for key in ('invalid_zeros','outside_R4'))
    checks.append(dict(name='invalid-and-exterior-accounting-exercised',passed=exercised))
    ledger.recheck()
    result = dict(schema='wall-envelope-sphere-analysis-v1',complete=True,passed=all(c['passed'] for c in checks),
        plan_sha256=sha(root/'plan.json'),execution_plan_sha256=sha(root/'execution-plan.json'),
        allocation_sha256=preparation.ALLOCATION_SHA,populations=populations,groups=groups,checks=checks,
        failed_checks=[c for c in checks if not c['passed']],attempts=preparation.TOTAL,
        attempts_by_activity={str(z):sum(p['samples'] for p in populations if p['activity'] == z) for z in preparation.ACTIVITIES},
        maximum_clouds=preparation.MAXIMUM_CLOUDS,actual_clouds=sum(p['counts']['clouds'] for p in populations),
        point_gate_assurance=plan['allocation'],input_sha256=ledger.files,
        new_pose_draws=0,new_clouds=0,new_geometry_queries=0,
        scope='Fixed new wall-envelope reference; same analytic sphere target, unconditional attempted denominators and four-population SE checks. The prospective joint bound applies to point-log errors only. Earlier failed cube rows are not pooled or replaced. R4 is descriptive; no protein or assembly inference. Failure authorizes no retries, replacements or extensions.')
    write(output,result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); result = analyze(args.root,args.out)
    import json
    print(json.dumps(dict(complete=True,passed=result['passed'],attempts=result['attempts'],failed_checks=result['failed_checks'])))
