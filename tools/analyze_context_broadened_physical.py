"""Audit all-attempt Poisson importance weights for frozen broadened guides.

The two complete positive weights are averaged before division by the FULL
deterministic-mixture arm density. Invalid poses remain zeros in denominator4096.
This is a conditional frozen-environment calculation, not an assembly decision.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.stats import t as student_t

from analyze_context_overlap_panel import audit_rows, close
from analyze_context_covariance_balance import importance_summary, across_populations

ARMS = ('baseline', 'broadened')
CHARTS = ('full', 'diagonal', 'broad_full')
REGIONS = ('A_patch_0_0.25', 'A_patch_0.25_0.5', 'A_patch_0.5_0.75',
           'A_patch_0.75_1', 'A_patch_complete', 'B', 'other_contact', 'unbound', 'hard_invalid')
GROUPS = ('full_domain', 'A_T', 'T_any', 'T_outside_A', 'remaining_without_A_T',
          'contact_without_A_T', 'A_partial', 'B', 'other_contact', 'unbound')
BINS = (0., 6., 12., 24., 48., 96., math.inf)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def jsonl(path):
    text = Path(path).read_text()
    require(not text or text.endswith('\n'), 'Incomplete JSONL: '+str(path))
    return [json.loads(line) for line in text.splitlines()]


def selectors(row):
    valid, region = row['physical_valid'], row['region']
    require(type(valid) is bool and region in REGIONS, 'Invalid geometry partition')
    require(valid == (region != 'hard_invalid'), 'Hard-invalid partition disagreement')
    target = valid and region == 'A_patch_complete'
    any_t = row['source_T_complete_all_regions']
    require(type(any_t) is bool and (not any_t or valid), 'Invalid T-any indicator')
    require(not target or any_t, 'A_T is not contained in T-any')
    return dict(full_domain=valid, A_T=target, T_any=any_t,
                T_outside_A=any_t and not target, remaining_without_A_T=valid and not target,
                contact_without_A_T=valid and region != 'unbound' and not target,
                A_partial=valid and region.startswith('A_patch_') and not target,
                B=region == 'B', other_contact=region == 'other_contact', unbound=region == 'unbound')


def validate_inventory(jobs):
    require(len(jobs) == 20 and len({j['stratum_id'] for j in jobs}) == 20
            and len({j['id'] for j in jobs}) == 20, 'Expected20 distinct stratum jobs')
    expected = {'baseline': {'full': 2048, 'diagonal': 2048},
                'broadened': {'full': 1536, 'diagonal': 1536, 'broad_full': 1024}}
    found = set()
    for job in jobs:
        key = (job['comparison_arm'], job['population_index'], job['component'])
        require(key not in found and key[0] in expected and key[1] in range(4)
                and job['attempts'] == expected[key[0]].get(key[2]), 'Changed fixed stratum allocation')
        found.add(key)
    require(found == {(a,p,c) for a in ARMS for p in range(4) for c in expected[a]},
            'Missing or extra generating component')


def validate_panel_coverage(job, panel, geometry, source_rows):
    require(len(geometry) == len(source_rows) == job['attempts'], 'Lost unconditional attempts')
    require([r['ordinal'] for r in geometry] == list(range(job['attempts'])), 'Changed draw order')
    expected = []
    for record, saved in zip(geometry, source_rows):
        require(saved['input']['ordinal'] == record['ordinal'] and saved['complete']
                and saved['actual']['physical_valid'] == record['physical_valid']
                and saved['input']['proposed_pose'] == record['proposed_pose'], 'Saved draw identity differs')
        selectors(record)
        require(record['region'] == (saved['region'] if record['physical_valid'] else 'hard_invalid'),
                'Saved region differs')
        require(saved['clouds'] == [] and saved['physical_weight_status'] == 'not_estimated',
                'Geometry rows are not untouched zero-cloud draws')
        if record['physical_valid']:
            expected.append(record['ordinal'])
    entries = panel['entries']
    require([entry['ordinal'] for entry in entries] == expected, 'Panel is not ALL hard-valid draws in order')
    require(bool(entries) != job['empty'], 'Empty-panel status differs')
    for entry in entries:
        record = geometry[entry['ordinal']]
        require(entry['source_rows'] == record['source_rows'] and entry['pose'] == record['proposed_pose'],
                'Panel source identity differs')
        close(entry['metadata']['log_q_balanced'], record['log_q_arm'], 'full_arm_density')
        require(entry['metadata'].get('original_row') == source_rows[entry['ordinal']],
                'Missing complete authenticated original row in panel')
    return expected


def validate_empty(job, cfg, panel, receipt):
    require(not panel['entries'] and receipt['schema'] == 'context-broadened-empty-physical-v1'
            and receipt['complete'] and receipt['passed'] and receipt['id'] == job['id']
            and receipt['stratum_id'] == job['stratum_id'] and receipt['attempts'] == job['attempts']
            and receipt['all_hard_invalid'] and receipt['all_attempts_preserved']
            and receipt['config_sha256'] == job['config']['sha256']
            and receipt['panel_sha256'] == job['panel']['sha256'], 'Invalid explicit empty-job receipt')
    for key in ('valid_poses', 'clouds_begun', 'clouds_completed', 'raw_points', 'processed_points',
                'new_poses_generated', 'retries', 'replacements'):
        require(receipt[key] == 0, 'Empty stratum performed or omitted work: '+key)
    require(math.isfinite(receipt['cpu_seconds']) and receipt['cpu_seconds'] >= 0,
            'Missing measured empty-job CPU')


def radial_index(value):
    value = math.inf if value is None else value
    require(type(value) in (int, float) and value >= 0 and not math.isnan(value), 'Invalid chart radius')
    return min(int(np.searchsorted(BINS, value, side='right'))-1, len(BINS)-2)


def concentration_gate(summary):
    return summary['importance_ess'] >= 200 and summary['largest_fraction'] is not None \
        and summary['largest_fraction'] <= .02


def population_statistics(records, geometry_cpu, physical_cpu):
    require(len(records) == 4096, 'Every population denominator must be4096')
    groups = {}
    regions = {}
    for name, predicate in [(g, lambda r,g=g: selectors(r)[g]) for g in GROUPS] \
            + [(g, lambda r,g=g: r['physical_valid'] and r['region'] == g) for g in REGIONS]:
        chosen = [r for r in records if predicate(r)]
        value = dict(physical=importance_summary([r['log_physical_weight'] for r in chosen], 4096),
                     hard=importance_summary([-r['log_q_arm'] for r in chosen], 4096))
        value['concentration_gate'] = concentration_gate(value['physical'])
        (groups if name in GROUPS else regions)[name] = value
    # Some group names equal partition names; retain every partition explicitly.
    for name in REGIONS:
        chosen = [r for r in records if r['physical_valid'] and r['region'] == name]
        regions[name] = dict(physical=importance_summary([r['log_physical_weight'] for r in chosen],4096),
                            hard=importance_summary([-r['log_q_arm'] for r in chosen],4096),
                            attempted_count=sum(r['region'] == name for r in records))
    for measure in ('physical','hard'):
        expected = groups['full_domain'][measure]['mass']
        actual = math.fsum(r[measure]['mass'] for r in regions.values())
        require(abs(actual-expected) <= 1e-11*max(expected,1e-300), 'Region masses fail closure')
    radial = {}
    for chart in CHARTS:
        radial[chart] = {}
        for group in ('full_domain','A_T','remaining_without_A_T'):
            radial[chart][group] = [dict(bin_index=b, lower=BINS[b], upper=None if b==5 else BINS[b+1],
                **importance_summary([r['log_physical_weight'] for r in records
                    if selectors(r)[group] and radial_index(r['mahalanobis_squared'][chart])==b],4096))
                for b in range(6)]
    total_cpu = geometry_cpu+physical_cpu
    require(math.isfinite(total_cpu) and total_cpu > 0, 'Missing geometry plus scoring CPU')
    return dict(attempts=4096, groups=groups, regions=regions, radial_partitions=radial,
                geometry_cpu_seconds=geometry_cpu, physical_cpu_seconds=physical_cpu,
                total_cpu_seconds=total_cpu,
                importance_ess_per_cpu={g:groups[g]['physical']['importance_ess']/total_cpu for g in GROUPS},
                efficiency_scope='Independent importance-draw concentration per geometry plus scoring CPU; not Markov-chain contact-fingerprint ESS or mixing speed.')


def compare_means(left, right):
    a,b = left['mean_mass'],right['mean_mass']
    se = math.hypot(left['standard_error'],right['standard_error'])
    difference = b-a
    log_ratio = math.log(b/a) if a>0 and b>0 else None
    return dict(broadened_minus_baseline=difference, combined_standard_error=se,
                difference_in_combined_se=abs(difference)/se if se else (0. if difference==0 else None),
                log_mass_ratio=log_ratio,
                within_three_combined_se=abs(difference)<=3*se,
                within_point_two_kbt=log_ratio is not None and abs(log_ratio)<=.2,
                comparison_scope='Fresh independent four-population arm means of the same conditional region; both agreement diagnostics required, not proof of unseen-mode coverage.')


def within_arm_free_energy(populations, numerator='A_T', denominator='remaining_without_A_T'):
    a=np.asarray([p['groups'][numerator]['physical']['mass'] for p in populations]); b=np.asarray([p['groups'][denominator]['physical']['mass'] for p in populations])
    if a.mean()<=0 or b.mean()<=0:
        return dict(available=False, reason='At least one region has zero estimated mean mass; no finite log ratio or confidence interval.')
    estimate=-math.log(float(a.mean()/b.mean()))
    influence=-(a/a.mean()-b/b.mean())
    se=float(influence.std(ddof=1)/2)
    half=float(student_t.ppf(.975,3)*se)
    return dict(available=True, numerator=numerator, denominator=denominator, delta_free_energy_kbt=estimate,
                population_delta_method_standard_error=se, interval95=[estimate-half,estimate+half],
                interval95_half_width=half, half_width_gate=half<=.5,
                caveat='Paired four-population delta-method Student-t interval. Approximate; unreliable for dominant tails or zero individual region masses; never a missing-mode bound.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); protocol=read(args.protocol); started=time.process_time()
    require(protocol['schema']=='context-broadened-physical-audit-v1'
            and protocol['all_attempts']==32768 and protocol['population_denominator']==4096
            and protocol['bins']==[0,6,12,24,48,96,None], 'Changed allocation or bins')
    conditions=protocol['physical_conditions']
    require(conditions=={'depletant_radius':1.5,'activity':.035,'lambda':2.24}, 'Changed physical conditions')
    bindings=protocol['input_sha256']
    for path,digest in bindings.items():require(sha(path)==digest,'Changed input: '+path)
    def bound(path):
        path=str(Path(path).resolve());require(path in bindings,'Unbound input: '+path);return read(path)
    def asset(spec):
        require(bindings.get(str(Path(spec['path']).resolve()))==spec['sha256'],'Conflicting asset binding')
        return bound(spec['path'])
    geometry=asset(protocol['geometry_audit']);asset(protocol['geometry_manifest']);asset(protocol['geometry_inventory']);asset(protocol['mixture_manifest'])
    require(geometry['complete'] and geometry['passed'] and geometry['all_attempts']==32768
            and geometry['independent_panel_size']==512,'Incomplete independent geometry audit')
    for path,digest in geometry['input_sha256'].items():
        require(bindings.get(path)==digest,'Unbound geometry-audit dependency: '+path)
    contribution_path=protocol['geometry_contributions'];require(bindings.get(str(Path(contribution_path['path']).resolve()))==contribution_path['sha256']
        and geometry['contributions_sha256']==contribution_path['sha256'],'Unbound or unaudited contributions')
    contributions=jsonl(contribution_path['path']);require(len(contributions)==32768,'Lost geometry attempts')
    jobs=protocol['jobs'];validate_inventory(jobs)
    strata={j['stratum_id']:[] for j in jobs}
    for row in contributions:
        require(row['stratum_id'] in strata,'Unknown geometry stratum');strata[row['stratum_id']].append(row)
    geometry_cpu={r['stratum_id']:r['geometry_cpu_seconds'] for r in geometry['stratum_summaries']}
    require(not args.out.exists(),'Fresh output required');args.out.mkdir()
    results=[];populations={(a,p):[] for a in ARMS for p in range(4)};costs={(a,p):[0.,0.] for a in ARMS for p in range(4)}
    with (args.out/'events.jsonl').open('x') as events,(args.out/'contributions.jsonl').open('x') as output:
        def emit(value):events.write(json.dumps(value,allow_nan=False)+'\n');events.flush()
        try:
            for job in jobs:
                sid=job['stratum_id'];emit(dict(kind='stratum_begun',id=job['id'],stratum_id=sid))
                records=strata[sid];cfg=asset(job['config']);panel=asset(job['panel'])
                require(cfg['schema']=='fixed-saved-pose-overlap-v1' and cfg['lambda']==2.24 and cfg['activity']==.035
                        and cfg['panel']==job['panel'],'Changed fixed scorer configuration')
                cloud_caps=cfg['cloud_limits']
                require(cloud_caps==dict(raw_per_cloud=1000000,raw_per_pose=2000000,
                    raw_total=2000000*job['attempts'],processed_per_cloud=1000000,
                    processed_per_pose=2000000,processed_total=2000000*job['attempts'],callback_interval=8192),
                    'Changed predeclared full-envelope cloud caps')
                require(all((r['comparison_arm'],r['population_index'],r['component'])==(job['comparison_arm'],job['population_index'],job['component']) for r in records),'Changed stratum identity')
                source_assets={r['source_rows']['path']:r['source_rows']['sha256'] for r in records};require(len(source_assets)==1,'Mixed saved source files')
                source_path,source_digest=next(iter(source_assets.items()));require(bindings.get(str(Path(source_path).resolve()))==source_digest,'Unbound original draws')
                originals=jsonl(source_path);valid=validate_panel_coverage(job,panel,records,originals)
                result=Path(job['result']);summary=bound(result/'summary.json');receipt=bound(job['execution_receipt'])
                require(receipt['success'] and receipt['child_drained'] and receipt['returncode']==0
                        and not receipt['timeout'] and receipt['retries']==receipt['replacements']==0,
                        'Undrained/failed physical or explicit empty job')
                require(receipt['id']==job['id']
                        and Path(receipt['terminal']['path']).resolve()==(result/'summary.json').resolve()
                        and receipt['terminal']['sha256']==bindings[str((result/'summary.json').resolve())],
                        'Execution receipt terminal or identity differs')
                scored=[];totals=dict(raw_points=0,processed_points=0,overlap_points=0,clouds=0)
                if job['empty']:
                    validate_empty(job,cfg,panel,summary)
                    require(summary['source_rows']==records[0]['source_rows'],'Empty source mismatch')
                else:
                    producer=bound(result/'protocol.json')
                    require(producer['config']==cfg and producer['config_sha256']==job['config']['sha256'],'Producer configuration mismatch')
                    for path,digest in producer['input_sha256'].items():require(bindings.get(path)==digest,'Unbound producer input')
                    row_path=result/'rows.jsonl';require(str(row_path.resolve()) in bindings,'Unbound physical rows')
                    physical_rows=jsonl(row_path)
                    scored,totals=audit_rows(cfg,panel,physical_rows)
                    require(all(c['weight']['raw_points']<=1000000 for r in physical_rows for c in r['clouds'])
                            and totals['raw_points']<=cloud_caps['raw_total']
                            and totals['processed_points']<=cloud_caps['processed_total'],'Cloud caps exceeded')
                    require(summary['complete'] and summary['passed']
                            and summary['poses_begun']==summary['poses_completed']==len(valid)
                            and summary['clouds_begun']==summary['clouds_completed']==2*len(valid)
                            and summary['retries']==summary['replacements']==summary['new_poses_generated']==0
                            and not summary['normalizer_estimated'],'Incomplete physical stratum')
                    for key in ('raw_points','processed_points'):require(summary[key]==totals[key],'Cloud summary differs')
                scores=dict(zip(valid,scored));key=(job['comparison_arm'],job['population_index'])
                for row in records:
                    enriched=dict(row);record=scores.get(row['ordinal'])
                    enriched['log_physical_weight']=record['log_positive_importance_weight'] if record else None
                    enriched['conditional_log_weight_score']=record['z_overlap'] if record else None
                    require(bool(record)==row['physical_valid'],'Missing or extra physical weight')
                    populations[key].append(enriched);output.write(json.dumps(enriched,allow_nan=False)+'\n')
                output.flush();costs[key][0]+=geometry_cpu[sid];costs[key][1]+=summary['cpu_seconds']
                results.append(dict(id=job['id'],stratum_id=sid,attempts=job['attempts'],valid_poses=len(valid),
                    geometry_cpu_seconds=geometry_cpu[sid],physical_cpu_seconds=summary['cpu_seconds'],totals=totals))
                emit(dict(kind='stratum_complete',**results[-1]))
            per_pop=[dict(comparison_arm=a,population_index=p,**population_statistics(populations[a,p],*costs[a,p])) for a in ARMS for p in range(4)]
            arm_summaries={a:{g:across_populations([r['groups'][g]['physical'] for r in per_pop if r['comparison_arm']==a]) for g in GROUPS} for a in ARMS}
            hard_arm_summaries={a:{g:across_populations([r['groups'][g]['hard'] for r in per_pop if r['comparison_arm']==a]) for g in GROUPS} for a in ARMS}
            gates={a:{g:dict(concentration_all_populations=all(r['groups'][g]['concentration_gate'] for r in per_pop if r['comparison_arm']==a),
                population_relative_error_at_most_point_one=arm_summaries[a][g]['population_relative_standard_error'] is not None and arm_summaries[a][g]['population_relative_standard_error']<=.1) for g in GROUPS} for a in ARMS}
            for path,digest in bindings.items():require(sha(path)==digest,'Input changed during reduction')
            report=dict(schema='context-broadened-physical-report-v1',complete=True,passed=True,
                protocol_sha256=sha(args.protocol),source_sha256=sha(__file__),input_sha256=bindings,
                all_attempts=32768,population_denominator=4096,strata=results,populations=per_pop,
                arm_summaries=arm_summaries,hard_arm_summaries=hard_arm_summaries,
                comparison={g:compare_means(arm_summaries['baseline'][g],arm_summaries['broadened'][g]) for g in GROUPS},
                hard_comparison={g:compare_means(hard_arm_summaries['baseline'][g],hard_arm_summaries['broadened'][g]) for g in GROUPS},
                pilot_gates=gates,free_energy={a:within_arm_free_energy([r for r in per_pop if r['comparison_arm']==a]) for a in ARMS},
                contributions_sha256=sha(args.out/'contributions.jsonl'),cpu_seconds=time.process_time()-started,
                new_geometry_queries=0,new_cloud_draws=0,new_poses=0,finite_stability_conclusion=False,
                scope='Historical500uM frozen263-body environment with one mobile tetramer; source-informed conditional contact patterns, not a native registry classifier. No finite-system assembly, instability or equilibrium concentration conclusion.',
                estimator='Sum of hard-valid (W0+W1)/(2 q_arm), divided by ALL4096 attempts per population; hard-invalid zeros retained. Chart Jacobian is already inside q_arm. No exponentiation of overlap score.',
                groups={'A_T':'A_patch_complete: exact neighbor set16,217 and all16 source patch tokens, extra patch tokens allowed.',
                    'T_any':'All16 source patch tokens at any neighbor set; overlaps A_T and is reported separately.',
                    'remaining_without_A_T':'Every physically valid full-wall-domain pose outside A_T, including unbound.',
                    'contact_without_A_T':'Every physically valid contact pose outside A_T.',
                    'radial_partitions':'Separate six-bin partitions under each frozen chart; never summed across charts.'},
                limitations='Four populations and ESS/concentration/stratum diagnostics cannot exclude unseen modes. A failed campaign is an unresolved sampling limitation, not evidence against assembly. Positive-weight identities apply to uncapped law; failed prefixes are never replaced, clipped, treated as zero, or success-conditioned.')
            (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        except BaseException as error:
            emit(dict(kind='fatal',error=repr(error)))
            (args.out/'failure.json').write_text(json.dumps(dict(complete=False,passed=False,error=repr(error),prefix_preserved=True),indent=2)+'\n')
            raise


if __name__=='__main__':main()
