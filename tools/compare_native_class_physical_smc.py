#!/usr/bin/env python3
"""Compare completed, authenticated population summaries; never read sample rows.

The plan binds current admission + its plan, both historical unrestricted SMC
analyses + their authentication/review, and the restricted campaign plan/status/
analysis. Historical verdicts are copied unchanged. This is a new comparison,
not a rerun of those audits, and cannot establish unseen mass or assembly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import analyze_native_class_physical_populations as current
import contact_population_statistics as population
from analyze_mobile_native_pocket import local_sources

PLAN_SCHEMA = 'native-class-physical-smc-comparison-plan-v1'
SCHEMA = 'native-class-physical-smc-comparison-v1'
OLD_SHA = '76ea65088e302d6b6478ac033af9b67b7cf21cb7cea1f854f70cdcd6db0473ae'
NATIVE_SHA = '5cf55ebe0c0f78ec6b8cdc745cac938b0fadfa3732e1e1c730864c62c651d7a9'
PARTITION_SHA = '8620fc2932bd6583a7571e36b44c71eaf95bd8032720806be31cf5c5c10f49dd'
# Reviewed historical Chart.bins: radius edges 2/3, covariance-derived angular
# projection squared edges 4/9, right-side assignment and six >=0 sign bits.
# These are receipt bindings, not imports or re-executions of historical code.
STRATA_SOURCES = {
    'analyze_r4_smc_control.py': '0196a0fb2e967e15d9da17a0c58951fe0df67cb7f1bc99712b00c0a947aedfb8',
    'analyze_native_excluded_smc.py': '8b74fa74385347af6801d55406e94dbdf3d3432dc0f6f25e3ba9ae4da3c41aae',
}
STRATA_DEFINITION = dict(radial_edges=[0., 2., 3., 4.], angular_projection_squared_edges=[0., 4., 9., 16.],
                         latent_orthants='six signs, zero assigned positive; all 64 bins retained')
MEASURE = 'd3t in Angstrom cubed times normalized proper SO(3) Haar'
MAP = dict(total='total', native='registered_native_entry', competing='contact_no_native_entry',
           unbound='unbound_no_native_entry', native_old_r5='old_R5_intersection_native',
           native_remainder='remaining_R4_native')
RESTRICTED = ('competing', 'unbound')
require = current.require


class Metadata:
    """Only opened JSON metadata is hashed. Upstream raw-row evidence is inherited."""
    def __init__(self): self.files = {}

    def read(self, ref):
        require(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'Expected BoundFile')
        path = Path(ref['path'])
        require(path.is_absolute() and path.suffix == '.json', 'Only absolute JSON metadata inputs allowed')
        path = path.resolve(); raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'Changed metadata '+str(path))
        require(str(path) not in self.files or self.files[str(path)] == ref['sha256'], 'Conflicting binding')
        self.files[str(path)] = ref['sha256']
        return current.streaming.loads(raw)

    def finish(self):
        for path, digest in self.files.items():
            require(current.streaming.sha(path) == digest, 'Metadata changed during comparison')


def bound(ref, closure):
    require(closure.get(str(Path(ref['path']).resolve())) == ref['sha256'], 'Metadata absent from upstream closure')


def named_binding(closure, name, digest):
    matches = [value for path, value in closure.items() if Path(path).name == name]
    require(matches and all(value == digest for value in matches), 'Unbound or changed historical '+name)


def strata_binding(closure, restricted=False):
    require(current.STRATA == STRATA_DEFINITION, 'Current stratum definition changed')
    names = tuple(STRATA_SOURCES) if restricted else ('analyze_r4_smc_control.py',)
    for name in names: named_binding(closure, name, STRATA_SOURCES[name])


def logsum(values):
    values = [population._log(v) for v in values]
    values = [v for v in values if v is not None]
    if not values: return None
    top = max(values)
    return top+math.log(math.fsum(math.exp(v-top) for v in values))


def equal_log(left, right, message):
    left, right = population._log(left), population._log(right)
    require((left is None) == (right is None), message)
    if left is not None: current.close(left, right, message)


def partition(masses, restricted=False):
    keys = ('total', MAP['competing'], MAP['unbound']) if restricted else tuple(MAP.values())
    require(set(masses) == set(keys), 'Incomplete historical region inventory')
    children = [MAP[n] for n in (RESTRICTED if restricted else ('native','competing','unbound'))]
    equal_log(masses['total'], logsum(masses[k] for k in children), 'Physical class partition differs')
    if not restricted:
        equal_log(masses[MAP['native']], logsum(masses[MAP[k]] for k in ('native_old_r5','native_remainder')),
                  'Native partition differs')


def terminal_check(p, particles, restricted=False):
    t = p['terminal']; n = t['particles']; counts = t['counts']; masses = t['log_masses']
    require(type(p['zero_estimate']) is bool and type(n) is int
            and n == (0 if p['zero_estimate'] else particles) and t['stage'] == 'terminal'
            and t['measure'] == ('restricted final physical target' if restricted else 'final physical target'),
            'Incomplete or changed terminal allocation')
    stages = p['stages']; hits = p['initial_hits']
    require(type(hits) is int and 0 <= hits <= p['initial_draws'] and (hits == 0) == p['zero_estimate'],
            'Initial-hit and zero-estimate accounting differs')
    # A failed initialization is a retained zero estimator, with no annealing.
    # Synthetic empty profile snapshots must not be mistaken for actual stages.
    if p['zero_estimate']:
        require(len(stages) == 1 and stages[0]['stage'] == 0 and stages[0]['beta'] == 0.
                and stages[0]['log_Z'] is None and stages[0]['zero_estimate'] is True,
                'Zero estimate must retain only the actual initialization stage')
    else:
        require(len(stages) == 129 and all(type(s['stage']) is int and s['stage'] == i
                and s['beta'] == i/128 and population._log(s['log_Z']) is not None
                for i, s in enumerate(stages)), 'Incomplete or changed 128-stage beta bridge')
        equal_log(stages[-1]['log_Z'], masses['total'], 'Terminal mass differs from final beta-one normalizer')
    require((population._log(masses['total']) is None) == p['zero_estimate'], 'Terminal zero-estimate flag differs')
    require(set(counts) == set(masses), 'Terminal count inventory differs')
    require(all(type(v) is int and v >= 0 for v in counts.values()) and counts['total'] == n,
            'Invalid terminal counts')
    partition(masses, restricted)
    children = RESTRICTED if restricted else ('native','competing','unbound')
    require(n == sum(counts[MAP[k]] for k in children), 'Terminal count partition differs')
    if not restricted:
        require(counts[MAP['native']] == counts[MAP['native_old_r5']]+counts[MAP['native_remainder']],
                'Native terminal counts differ')
    for key, value in masses.items():
        expected = None if n == 0 or counts[key] == 0 or masses['total'] is None else masses['total']+math.log(counts[key]/n)
        equal_log(value, expected, 'SMC mass must be normalizer times terminal indicator')
    partition(p['initial_physical_hard_log_masses'], restricted)
    require((population._log(p['initial_physical_hard_log_masses']['total']) is None) == p['zero_estimate'],
            'Initial physical hard mass and zero-estimate flag differ')
    require(set(t['log_strata']) == set(current.BINS), 'Missing historical strata')
    for family, size in current.BINS.items():
        bins = t['log_strata'][family]
        require(set(bins) == set(masses) and all(len(v) == size for v in bins.values()), 'Historical stratum dimensions differ')
        for key in masses:
            equal_log(masses[key], logsum(bins[key]), 'Strata do not sum to parent mass')
        for index in range(size): partition({key: bins[key][index] for key in masses}, restricted)


def config_matches(config, region):
    for key, rkey in [('fixed_poses','physical_fixed_neighbors'),('capture_center','capture_center'),
                     ('capture_radius','capture_radius'),('reservoir_density','activity'),
                     ('depletant_radius','depletant_radius'),('metadata','physical_metric')]:
        require(config[key] == region[rkey], 'Different physical '+key)


def scope_id(target, family=None, index=None):
    return target if family is None else current.fingerprint(dict(target_and_regions_sha256=target,
                                                                  stratum=f'{family}:{index}:'))


def batch(name, records, regions, target):
    records = sorted(records, key=lambda r:r['id'])
    draws = records[0]['draws']
    declaration = dict(arm_id=name, population_count=len(records), draws_per_population=draws,
        total_unconditional_draws=len(records)*draws, regions=list(regions), target_and_regions_sha256=target,
        populations=[dict(id=r['id'],seed=r['seed']) for r in records])
    population._validate(declaration, records)
    return dict(declaration=declaration, population_records=records)


def historical_batch(name, pops, kind, target, restricted=False, family=None, index=None):
    require(kind in ('Qz','Q0') and not (kind == 'Q0' and family is not None), 'Q0 strata were not archived')
    regions = RESTRICTED if restricted else tuple(MAP)
    rows = []
    for p in pops:
        if kind == 'Q0': masses = p['initial_physical_hard_log_masses']
        elif family is None: masses = p['terminal']['log_masses']
        else: masses = {k:v[index] for k,v in p['terminal']['log_strata'][family].items()}
        rows.append(dict(id=p['id'],seed=p['seed'],draws=p['initial_draws'],
            unconditional_denominator=p['initial_draws'],log_masses={k:masses[MAP[k]] for k in regions}))
    return batch(name, rows, regions, scope_id(target,family,index))


def load_current(spec, metadata):
    admission, plan = [metadata.read(spec[k]) for k in ('admission','admission_plan')]
    require(plan['schema'] == 'native-class-physical-admission-plan-v1'
            and admission['schema'] == 'native-class-physical-admission-v1' and admission['complete'] is True
            and admission['implementation_admission_passed'] is True
            and admission['execution_lifecycle_passed'] is True
            and admission['selected_full_geometry_gate_satisfied'] is True
            and admission['plan_sha256'] == spec['admission_plan']['sha256'], 'Completed implementation admission required')
    for key in ('protocol','statistics_result','statistics_plan'):
        bound(plan[key],admission['input_sha256'])
    require(admission['protocol'] == plan['protocol'] and admission['statistics_result'] == plan['statistics_result'],
            'Admission references differ')
    protocol, stat_plan, result = [metadata.read(plan[k]) for k in ('protocol','statistics_plan','statistics_result')]
    require(protocol['schema'] == 'native-class-physical-protocol-v1' and stat_plan['schema'] == current.PLAN_SCHEMA
            and result['schema'] == current.SCHEMA and result['complete'] is True
            and result['plan_sha256'] == plan['statistics_plan']['sha256'], 'Completed current statistics required')
    bound(plan['statistics_plan'],result['input_sha256'])
    for key in ('target','target_and_regions_sha256','strata','gates'):
        require(protocol[key] == stat_plan[key], 'Changed current protocol '+key)
    target = stat_plan['target']; region = metadata.read(target['region'])
    old = metadata.read(target['old_r5_region']); metadata.read(target['native_definition'])
    for ref in target.values():
        bound(ref,admission['input_sha256']); bound(ref,result['input_sha256'])
    require(target['region']['sha256'] == current.ORIGINAL_REGION_SHA
            and region['shape_sha256'] == current.ORIGINAL_SHAPE_SHA
            and target['old_r5_region']['sha256'] == OLD_SHA
            and target['native_definition']['sha256'] == NATIVE_SHA, 'Original target hashes differ')
    current.validate_regions(region, old)
    require(region['capture_radius'] == 170. and region['capture_center'] == [0.,0.,0.]
            and region['activity'] == .035 and region['depletant_radius'] == 1.5
            and region['mahalanobis_radius'] == 4. and region.get('minimum_mahalanobis_radius',0.) == 0.,
            'Original physical region differs')
    _, descriptor, target_id = current.target_identity(target, current.Bindings())
    require(result['target_descriptor'] == descriptor and result['target_and_regions_sha256'] == target_id
            and admission['target_and_regions_sha256'] == target_id == stat_plan['target_and_regions_sha256']
            and stat_plan['strata'] == current.STRATA and result['gates'] == stat_plan['gates'] == current.GATES,
            'Current target descriptor or diagnostics changed')
    require(len(result['arms']) == len(stat_plan['arms']) == len(protocol['arms'])
            and {a['id'] for a in stat_plan['arms']} == {a['id'] for a in protocol['arms']} == set(result['arms']),
            'Dropped current arm')
    admitted = {(p['arm'],p['id'],p['samples']) for p in admission['populations']}
    expected = set(); seeds = []
    for arm in stat_plan['arms']:
        pops = arm['populations']; expected.update((arm['id'],p['id'],arm['samples']) for p in pops)
        required = sorted((p['id'],p['seed']) for p in pops)
        require(len(pops) == 8 and len(set(required)) == 8, 'Eight current populations required')
        seeds.extend(p['seed'] for p in pops)
        proto = next(a for a in protocol['arms'] if a['id'] == arm['id'])
        require(proto['samples'] == arm['samples'] and sorted((p['id'],p['seed']) for p in proto['populations']) == required,
                'Current population allocation changed')
        for kind in ('Qz','Q0'):
            blocks = result['arms'][arm['id']]['estimates'][kind]
            for family, size in [(None,1), *current.BINS.items()]:
                entries = [blocks['primary']] if family is None else blocks['strata'][family]
                require(len(entries) == size, 'Missing current stratum')
                for index, item in enumerate(entries):
                    dec = item['declaration']; rows = item['population_records']
                    population._validate(dec,rows)
                    require(sorted((p['id'],p['seed']) for p in rows) == required
                            and dec['draws_per_population'] == arm['samples']
                            and dec['target_and_regions_sha256'] == scope_id(target_id,family,index)
                            and tuple(dec['regions']) == current.REGIONS, 'Current block population/scope differs')
                    for row in rows: partition({MAP[k]:v for k,v in row['log_masses'].items()})
    require(len(admission['populations']) == len(expected) and admitted == expected
            and len(seeds) == len(set(seeds)), 'Current admission inventory or independence differs')
    return result, region, target_id, admission


def load_unrestricted(spec, region, metadata):
    require(set(spec['analyses']) == {'broad','narrow'}, 'Both historical unrestricted controls required')
    auth, review = [metadata.read(spec[k]) for k in ('authentication','review')]
    require(auth['schema'] == 'completed-smc-bridge-evidence-authentication-v1' and auth['complete'] is True
            and review['schema'] == 'completed-smc-evidence-summary-v1' and review['complete'] is True,
            'Completed historical authentication/review required')
    bound(spec['authentication'],review['input_sha256'])
    arms = {}
    for name, ref in spec['analyses'].items():
        bound(ref,auth['input_sha256']); bound(ref,review['input_sha256'])
        value = metadata.read(ref); p = value['protocol_snapshot']; target = p['physical_target']
        require(value['schema'] == 'smc-r4-control-analysis-v1' and value['complete'] is True,
                'Incomplete unrestricted SMC')
        require(target['region_sha256'] == current.ORIGINAL_REGION_SHA
                and target['shape_sha256'] == current.ORIGINAL_SHAPE_SHA
                and p['proposal']['reference_region_sha256'] == OLD_SHA
                and value['native_definition_sha256'] == NATIVE_SHA and target['measure'] == MEASURE,
                'Historical target hashes/measure differ')
        config_matches(value['physical_config'],region)
        strata_binding(value['analyzer_source_sha256'])
        named_binding(p['source_and_input_sha256'], 'native-partition-definition.json', PARTITION_SHA)
        for k,rk in [('fixed_poses','physical_fixed_neighbors'),('capture_center','capture_center'),
                     ('capture_radius','capture_radius'),('activity','activity'),('depletant_radius','depletant_radius')]:
            require(target[k] == region[rk], 'Historical physical target differs: '+k)
        jobs = p['jobs']; pops = value['populations']; a = p['allocation']
        require(a['stages_each'] == 128 and len(jobs) == len(pops) == a['independent_populations'] == 4
                and len({j['id'] for j in jobs}) == 4
                and sorted((j['id'],j['seed']) for j in jobs) == sorted((v['id'],v['seed']) for v in pops),
                'Dropped historical population')
        for pop in pops:
            require(pop['initial_draws'] == a['unconditional_initialization_draws_each'], 'Historical denominator changed')
            terminal_check(pop,a['particles_each'])
        arms['unrestricted_'+name] = dict(populations=pops,restricted=False)
    return arms, review


def load_restricted(spec, region, metadata):
    plan,status,result = [metadata.read(spec[k]) for k in ('plan','status','analysis')]
    require(plan['schema'] == result['schema'] == 'native-excluded-smc-fixed-control-v1'
            and result['complete'] is True and status['complete'] is True and status['phase'] == 'complete'
            and status['plan_sha256'] == spec['plan']['sha256']
            and status['analysis_sha256'] == spec['analysis']['sha256'], 'Restricted campaign is not completed')
    t = plan['physical_target']
    require(t['shape_sha256'] == current.ORIGINAL_SHAPE_SHA and t['region_sha256'] == current.ORIGINAL_REGION_SHA
            and t['definition_sha256'] == NATIVE_SHA
            and t['measure'] == 'Cartesian Angstrom cubed times normalized proper SO(3) Haar'
            and t['target'] == 'Hcapture Hhard IR4 (1-Inative) exp(zC) d3t dHaar', 'Restricted target changed')
    require(plan['allocation']['stages'] == 128
            and plan['analysis']['class_order'] == ['total', MAP['competing'], MAP['unbound']]
            and plan['analysis']['strata'] == current.BINS, 'Restricted stage/stratum allocation differs')
    strata_binding(plan['sources'], True)
    jobs=plan['jobs']; expected={f'{arm}-r{i:02}' for arm in ('narrow','large','broad') for i in range(4)}
    require(len(jobs) == 12 and {j['id'] for j in jobs} == expected and set(result['audits']) == expected
            and set(result['arms']) == {'narrow','large','broad'}, 'Restricted allocation incomplete')
    for kind in ('jobs','audits'):
        records=status[kind]
        require(len(records) == 12 and {r['id'] for r in records} == expected
                and all(r['status'] == 'complete' and r['returncode'] == 0 for r in records), 'Restricted lifecycle incomplete')
    groups={name:dict(populations=[],restricted=True) for name in ('restricted_narrow','restricted_large','restricted_broad')}
    for job in jobs:
        arm=job['arm']; ref=result['audits'][job['id']]; bound(ref,result['audited_source_sha256'])
        p=metadata.read(ref); closure=p['source_sha256']; directory=Path(job['output'])
        strata_binding(closure, True)
        def read_child(path):
            path=str(path.resolve()); return metadata.read(dict(path=path,sha256=closure[path]))
        config=read_child(directory/'provenance/config.json'); config_matches(config,region)
        for file,digest in [('region.json',current.ORIGINAL_REGION_SHA),('shape.json',current.ORIGINAL_SHAPE_SHA),
                            ('initial-reference-region.json',OLD_SHA)]:
            require(closure[str((directory/'provenance'/file).resolve())] == digest, 'Restricted target binding differs')
        require(p['schema'] == 'native-excluded-smc-population-audit-v1' and p['complete'] is True
                and p['id'] == job['id'] and p['seed'] == job['seed']
                and p['initial_draws'] == plan['allocation']['initial_draws_per_population']
                and p['independent_analytic_test_adapter'] is False
                and p['native_definition'] == dict(definition_sha256=NATIVE_SHA,compiled_sha256=t['compiled_sha256'],
                    shape_sha256=current.ORIGINAL_SHAPE_SHA,fixed_poses=region['physical_fixed_neighbors']),
                'Restricted audit identity differs')
        terminal_check(p,plan['allocation']['arms'][arm]['population'],True)
        groups['restricted_'+arm]['populations'].append(p)
    return groups, {k:result[k] for k in ('quality','comparisons','strata','matching_importance_contact_comparisons',
                                          'physical_conclusion','finite_R4_unbound_bound')}


def compare_reports(report, historical, target_id):
    """Pure population-mass adapter. No source population or stratum is dropped."""
    require(set(historical) == {'unrestricted_broad','unrestricted_narrow','restricted_narrow','restricted_large','restricted_broad'},
            'All five historical controls required')
    seeds=[p['seed'] for group in historical.values() for p in group['populations']]
    require(len(seeds) == 20 and len(seeds) == len(set(seeds))
            and all(len(g['populations']) == 4 for g in historical.values()), 'Historical inventory or seeds differ')
    comparisons=[]
    for arm, values in report['arms'].items():
        for method, group in historical.items():
            regions=RESTRICTED if group['restricted'] else tuple(MAP)
            item=dict(current_arm=arm,historical_arm=method,restricted=group['restricted'],kinds={})
            for kind in ('Qz','Q0'):
                left=values['estimates'][kind]['primary']
                right=historical_batch(method,group['populations'],kind,target_id,group['restricted'])
                masses={key:population.compare_population_masses(left['declaration'],left['population_records'],
                    right['declaration'],right['population_records'],key) for key in regions}
                strata=[]
                if kind == 'Qz':
                    for family,size in current.BINS.items():
                        for index in range(size):
                            a=values['estimates'][kind]['strata'][family][index]
                            b=historical_batch(method,group['populations'],kind,target_id,group['restricted'],family,index)
                            for key in regions:
                                result=population.compare_population_masses(a['declaration'],a['population_records'],
                                    b['declaration'],b['population_records'],key)
                                fractions=[]
                                for side in ('left','right'):
                                    part=result[side]['log_linear_mean']; whole=masses[key][side]['log_linear_mean']
                                    fractions.append(None if whole is None else 0. if part is None else math.exp(part-whole))
                                strata.append(dict(family=family,bin=index,region=key,comparison=result,
                                    observed_parent_mass_fractions=fractions,material=any(v is not None and v>=.01 for v in fractions)))
                item['kinds'][kind]=dict(regions=masses,strata=strata,strata_available=kind=='Qz',
                    unavailable_reason=None if kind=='Qz' else 'Historical unresampled H/g stratum masses were not archived; not reconstructed.')
            comparisons.append(item)
    return comparisons


def compare(plan_path):
    metadata=Metadata(); path=Path(plan_path).resolve()
    plan=metadata.read(dict(path=str(path),sha256=current.streaming.sha(path)))
    require(plan['schema'] == PLAN_SCHEMA, 'Wrong bridge plan schema')
    sources={str(p):current.streaming.sha(p) for p in local_sources(__file__).values()}
    report,region,target_id,admission=load_current(plan['current'],metadata)
    historical,old=load_unrestricted(plan['unrestricted'],region,metadata)
    restricted,old_restricted=load_restricted(plan['restricted'],region,metadata); historical.update(restricted)
    result=compare_reports(report,historical,target_id)
    metadata.finish()
    require(all(current.streaming.sha(p)==h for p,h in sources.items()), 'Adapter source changed')
    return dict(schema=SCHEMA,complete=True,plan_sha256=metadata.files[str(path)],
        target_and_regions_sha256=target_id,region_mapping=MAP,comparisons=result,
        historical_strata_binding=dict(definition=STRATA_DEFINITION, source_sha256=STRATA_SOURCES,
            native_partition_sha256=PARTITION_SHA, edge_assignment='right; zero sign positive',
            scope='Reviewed source identities inherited through completed metadata; historical code not executed.'),
        historical_population_batches={name:{kind:historical_batch(name,group['populations'],kind,target_id,group['restricted'])
            for kind in ('Qz','Q0')} for name,group in historical.items()},
        historical_verdicts_unchanged=dict(unrestricted=old,restricted=old_restricted),
        current_admission_diagnostics=admission['statistical_diagnostics'],
        missing_current_studies=admission['missing_studies'],
        input_sha256=metadata.files,source_sha256=sources,
        regional_convergence_established=False,unseen_support_verdict='unresolved',
        physical_campaign_gate_open=False,full_vessel_gate_open=False,assembly_gate_open=False,
        parsed_scientific_rows=0,new_pose_draws=0,new_Poisson_clouds=0,new_geometry_queries=0,
        scope='New independent linear-population comparisons from inherited completed audits. No row replay, '
              'label queries, descendant IID errors, population pooling, or revisions of historical verdicts. '
              'Current admission and historical audit execution are trusted receipt evidence, not re-executed here. '
              'Restricted total is not unrestricted total; native is not inferred zero from restriction. '
              'Agreement and finite-R4 bounds do not certify unseen contact modes, vessel remainder or assembly.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(), 'Fresh output required')
    result=compare(args.plan)
    with args.out.open('x') as handle: handle.write(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
