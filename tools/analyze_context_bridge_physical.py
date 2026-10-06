"""Independent audit of the exploratory saved-bridge physical extension.

The bank was selected after inspecting geometry. Conditional cloud expectations
remain exact, but population SEs/intervals and threshold checks are descriptive;
this extension is not fresh confirmation or evidence of finite-system stability.
All16384 attempts, including invalid zeros, remain in2048-draw populations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from analyze_context_multicage_physical import (
    require, sha, read, jsonl, selectors, validate_empty, radial_index,
    concentration_gate, within_arm_free_energy, paired_weights, audit_ledger,
    audit_rows, close, importance_summary, across_populations, REGIONS, GROUPS, BINS)
from context_multicage_noise_budget import moment_terms, logsum, linear
from audit_context_bridge_guide import ARMS, COMPONENTS as CHARTS, STRATA, COEFFICIENTS, arm_log_q
from audit_context_candidate_bank import canonical_tokens

ALLOCATIONS={arm:dict(STRATA[arm]) for arm in ARMS}
PHYSICAL={'depletant_radius':1.5,'activity':.035,'lambda':2.24}
CLOUD_RULE=dict(clouds_per_valid_pose=2,intensity=2.24,
    primary_estimator='pooled-count-rao-blackwell',secondary_estimator='arithmetic-pair-positive-weights',
    primary_log_formula='z*L+(K0+K1)*log1p(z/(2*lambda))',
    common_certified_lower_volume=True,independent_equal_intensity_clouds=True,
    score_every_valid_pose=True,zero_volume_clouds_audited=True,
    retain_hard_invalid_zeros=True,population_denominator=2048,
    maximum_cloud_count=32768,new_clouds_only=True)
EXTENSION_SCOPE=dict(geometry_selected_after_outcomes=True,fresh_confirmation=False,
    guide_refit=False,every_valid_saved_pose=True,failed_geometry_gate_preserved=True)
DENOMINATOR=2048
ALL_ATTEMPTS=16384
RADIAL_CHARTS=('full',)



def validate_density(record):
    """All thirteen frozen source charts plus uniform/context, each exactly once."""
    require(record['comparison_arm'] in ARMS and set(record['log_source'])==set(CHARTS),
            'Incomplete bridge density inventory')
    logs=[record['log_u'] if record['log_u'] is not None else -math.inf,
          record['log_g'] if record['log_g'] is not None else -math.inf]
    logs.extend(record['log_source'][c] if record['log_source'][c] is not None else -math.inf
                for c in CHARTS)
    expected=float(arm_log_q(np.asarray(logs)[:,None],COEFFICIENTS[record['comparison_arm']])[0])
    close(record['log_q_arm'],expected,'complete fifteen-term bridge arm density')


def tail_statistics(records,denominator):
    """Physical and hard mass shares in the unchanged original full-chart bins."""
    require(len(records)==denominator,'Lost unconditional tail denominator')
    selected=[r for r in records if selectors(r)['A_T']]
    def summary(rows):
        return dict(physical=importance_summary([r['log_physical_weight'] for r in rows],denominator),
                    hard=importance_summary([-r['log_q_arm'] for r in rows],denominator))
    total=summary(selected)
    def fraction(row,measure):
        value=row['log_physical_weight'] if measure=='physical' else -row['log_q_arm']
        return math.exp(value-total[measure]['log_weight_sum'])
    bins=[]
    for b in range(6):
        rows=[r for r in selected if radial_index(r['mahalanobis_squared']['full'])==b]
        entry=dict(bin_index=b,lower=BINS[b],upper=None if b==5 else BINS[b+1],**summary(rows))
        for measure in ('physical','hard'):
            entry[measure+'_fraction_of_A_T']=math.exp(entry[measure]['log_weight_sum']-
                total[measure]['log_weight_sum']) if rows else 0.
        bins.append(entry)
    for measure in ('physical','hard'):
        require(sum(b[measure]['hits'] for b in bins)==len(selected),'Lost A_T tail rows')
        require(not selected or abs(math.fsum(b[measure+'_fraction_of_A_T'] for b in bins)-1)<2e-12,
                'A_T tail fractions do not close')
    largest={}
    for measure in ('physical','hard'):
        if not selected:
            largest[measure]=None
            continue
        row=max(selected,key=lambda r:r['log_physical_weight'] if measure=='physical' else -r['log_q_arm'])
        largest[measure]=dict(row_id=row['row_id'],comparison_arm=row['comparison_arm'],
            population_index=row['population_index'],component=row['component'],branch=row['branch'],
            original_full_squared_radius=row['mahalanobis_squared']['full'],
            log_q_arm=row['log_q_arm'],log_physical_weight=row['log_physical_weight'],
            physical_fraction_of_A_T=fraction(row,'physical'),hard_fraction_of_A_T=fraction(row,'hard'))
    return dict(denominator=denominator,chart='full',totals=total,bins=bins,largest=largest,
        scope='Observed-bank contribution shares, not coverage or confidence bounds; source-contact labels are not native registry.')


def cloud_variance_statistics(records,denominator):
    """Exact conditional cloud-variance identity, no random variance-share ratios."""
    require(len(records)==denominator,'Lost unconditional cloud-variance denominator')
    result={}
    for group in GROUPS:
        values=[r['log_auxiliary_variance'] for r in records if selectors(r)[group]]
        total=logsum(values)
        estimate=None if total is None else total-2*math.log(denominator)
        result[group]=dict(log_estimate=estimate,estimate=0. if estimate is None else linear(estimate))
    return dict(denominator=denominator,groups=result,
        estimator='sum_x [Y_x^2-P_x]/N^2; P_x=exp(2*z*L)*(1+z/lambda)^S/q_arm^2',
        scope='Unbiased conditional variance contribution for fresh independent clouds at this fixed selected pose bank. It excludes pose uncertainty and does not supply a confidence interval or an unbiased variance fraction.')


def validate_inventory(jobs):
    require(len(jobs)==72 and len({j['stratum_id'] for j in jobs})==72
            and len({j['id'] for j in jobs})==72, 'Expected72 distinct stratum jobs')
    found=set()
    for job in jobs:
        key=(job['comparison_arm'],job['population_index'],job['component'])
        require(key not in found and key[0] in ALLOCATIONS and type(key[1]) is int
            and key[1] in range(4) and key[2] in ALLOCATIONS[key[0]]
            and type(job['attempts']) is int and job['attempts']==ALLOCATIONS[key[0]][key[2]] and job['attempts']>0,
            'Changed fixed stratum allocation')
        found.add(key)
    require(found=={(a,p,c) for a in ARMS for p in range(4)
            for c,n in ALLOCATIONS[a].items() if n}, 'Missing or extra generating component')

def validate_panel_coverage(job, panel, geometry, source_rows, reference):
    require(len(geometry) == len(source_rows) == job['attempts'], 'Lost unconditional attempts')
    require([r['ordinal'] for r in geometry] == list(range(job['attempts'])), 'Changed draw order')
    source_tokens=canonical_tokens(reference)
    require(len(source_tokens)==16 and all(t[:2]==(77,217) for t in source_tokens),
            'Changed original source-token reference')
    expected = []
    for record, saved in zip(geometry, source_rows):
        require(saved['input']['ordinal'] == record['ordinal'] and saved['complete']
                and saved['actual']['physical_valid'] == record['physical_valid']
                and saved['input']['proposed_pose'] == record['proposed_pose'], 'Saved draw identity differs')
        selectors(record)
        validate_density(record)
        canonical=json.dumps(saved,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        require(record['canonical_source_row_sha256']==hashlib.sha256(canonical).hexdigest(), 'Canonical geometry row digest differs')
        require(record['region'] == (saved['region'] if record['physical_valid'] else 'hard_invalid'),
                'Saved region differs')
        require(saved['clouds'] == [] and saved['physical_weight_status'] == 'not_estimated',
                'Geometry rows are not untouched zero-cloud draws')
        wall,core=saved['actual']['wall_valid'],saved['actual']['core_valid']
        require(type(wall) is bool and ((wall and type(core) is bool) or (not wall and core is None))
                and record['physical_valid']==(wall and core is True)
                and record['wall_valid']==wall and record['core_valid']==core,
                'Changed delayed wall/core verdict')
        if record['physical_valid']:
            tokens=canonical_tokens(saved['patches']['tokens']);count=len(tokens&source_tokens)
            neighbors=sorted({b if a==77 else a for a,b,_,_ in tokens})
            require(record['exact_source_intersection_count']==count
                    and saved['patches']['source_fraction']==record['source_fraction']==count/16
                    and saved['patches']['neighbor_labels']==record['neighbor_labels']==neighbors
                    and record['source_T_complete_all_regions']==(count==16)
                    and record['exact_A']==(neighbors==[16,217]), 'Changed exact source-token classification')
            expected.append(record['ordinal'])
        else:
            require(record['exact_source_intersection_count'] is None
                    and record['source_fraction'] is None and record['neighbor_labels'] is None,
                    'Hard-invalid pose has physical contact labels')
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

def population_statistics(records, geometry_cpu, physical_cpu, include_cloud_variance=True):
    require(len(records) == 2048, 'Every population denominator must be2048')
    groups = {}
    regions = {}
    for name, predicate in [(g, lambda r,g=g: selectors(r)[g]) for g in GROUPS] \
            + [(g, lambda r,g=g: r['physical_valid'] and r['region'] == g) for g in REGIONS]:
        chosen = [r for r in records if predicate(r)]
        value = dict(physical=importance_summary([r['log_physical_weight'] for r in chosen], 2048),
                     hard=importance_summary([-r['log_q_arm'] for r in chosen], 2048))
        value['concentration_gate'] = concentration_gate(value['physical'])
        (groups if name in GROUPS else regions)[name] = value
    # Some group names equal partition names; retain every partition explicitly.
    for name in REGIONS:
        chosen = [r for r in records if r['physical_valid'] and r['region'] == name]
        regions[name] = dict(physical=importance_summary([r['log_physical_weight'] for r in chosen],2048),
                            hard=importance_summary([-r['log_q_arm'] for r in chosen],2048),
                            attempted_count=sum(r['region'] == name for r in records))
    for measure in ('physical','hard'):
        expected = groups['full_domain'][measure]['mass']
        actual = math.fsum(r[measure]['mass'] for r in regions.values())
        require(abs(actual-expected) <= 1e-11*max(expected,1e-300), 'Region masses fail closure')
    radial = {}
    for chart in RADIAL_CHARTS:
        radial[chart] = {}
        for group in ('full_domain','A_T','remaining_without_A_T'):
            radial[chart][group] = [dict(bin_index=b, lower=BINS[b], upper=None if b==5 else BINS[b+1],
                **importance_summary([r['log_physical_weight'] for r in records
                    if selectors(r)[group] and radial_index(r['mahalanobis_squared'][chart])==b],2048))
                for b in range(6)]
    total_cpu = geometry_cpu+physical_cpu
    require(math.isfinite(total_cpu) and total_cpu > 0, 'Missing geometry plus scoring CPU')
    return dict(attempts=2048, groups=groups, regions=regions, radial_partitions=radial,
                A_T_tail=tail_statistics(records,2048),
                fixed_bank_cloud_variance=cloud_variance_statistics(records,2048) if include_cloud_variance else None,
                geometry_cpu_seconds=geometry_cpu, physical_cpu_seconds=physical_cpu,
                total_cpu_seconds=total_cpu,
                importance_ess_per_cpu={g:groups[g]['physical']['importance_ess']/total_cpu for g in GROUPS},
                efficiency_scope='Independent importance-draw concentration per geometry plus scoring CPU; not Markov-chain contact-fingerprint ESS or mixing speed.')

def compare_means(left, right):
    a,b = left['mean_mass'],right['mean_mass']
    se = math.hypot(left['standard_error'],right['standard_error'])
    difference = b-a
    log_ratio = math.log(b/a) if a>0 and b>0 else None
    return dict(bridge_minus_baseline=difference, combined_standard_error=se,
                difference_in_combined_se=abs(difference)/se if se else (0. if difference==0 else None),
                log_mass_ratio=log_ratio,
                within_three_combined_se=abs(difference)<=3*se,
                within_point_two_kbt=log_ratio is not None and abs(log_ratio)<=.2,
                comparison_scope='Descriptive comparison of four population means from a geometry-selected saved bank. Not fresh confirmatory uncertainty or a missing-mode bound.')

def validate_extension_contract(protocol):
    require(protocol['schema']=='context-bridge-physical-audit-v1'
            and protocol['all_attempts']==ALL_ATTEMPTS and protocol['population_denominator']==DENOMINATOR
            and protocol['bins']==[0,6,12,24,48,96,None] and protocol['estimator']==CLOUD_RULE
            and protocol['physical_conditions']==PHYSICAL and protocol['extension_scope']==EXTENSION_SCOPE,
            'Changed exploratory allocation, weight law, conditions or selection scope')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); protocol=read(args.protocol); started=time.process_time()
    validate_extension_contract(protocol)
    bindings=protocol['input_sha256']
    for path,digest in bindings.items():require(sha(path)==digest,'Changed input: '+path)
    def bound(path):
        path=str(Path(path).resolve());require(path in bindings,'Unbound input: '+path);return read(path)
    def asset(spec):
        require(bindings.get(str(Path(spec['path']).resolve()))==spec['sha256'],'Conflicting asset binding')
        return bound(spec['path'])
    geometry=asset(protocol['geometry_audit']);asset(protocol['geometry_manifest']);asset(protocol['geometry_inventory']);mixture=asset(protocol['mixture_manifest'])
    require(mixture['physical_conditions']==PHYSICAL and mixture['mode']=='geometry' and mixture['physical_weight_status']=='not_estimated', 'Changed geometry-only source bank')
    require(protocol['extension_scope']==EXTENSION_SCOPE, 'Exploratory selection scope changed')
    campaign=asset(protocol['physical_campaign']);manifest=asset(protocol['physical_manifest'])
    physical_inventory=asset(protocol['physical_inventory']);controller=asset(protocol['physical_controller_receipt']);drain=asset(protocol['physical_host_drained'])
    scorer_bundle=asset(protocol['scorer_source_bundle']);scorer_build=asset(protocol['scorer_build_receipt'])
    require(campaign['geometry_audit']==protocol['geometry_audit'] and campaign['mixture_manifest']==protocol['mixture_manifest']
        and campaign['estimator']==CLOUD_RULE and campaign['scorer_source_bundle']==protocol['scorer_source_bundle']
        and campaign['scorer_build_receipt']==protocol['scorer_build_receipt']
        and campaign['extension_scope']==EXTENSION_SCOPE,'Physical campaign provenance differs')
    policy=asset(campaign['policy'])
    require(policy['estimator']==CLOUD_RULE and policy['extension_scope']==EXTENSION_SCOPE, 'Unbound separate physical policy')
    build_manifest=asset(campaign['scorer_build_manifest'])
    require(scorer_build['manifest_sha256']==campaign['scorer_build_manifest']['sha256'],'Scorer build manifest differs')
    for path,digest in manifest['files'].items():require(bindings.get(path)==digest,'Unbound physical execution input: '+path)
    for record in scorer_bundle['files'].values():require(hashlib.sha256(record['text'].encode()).hexdigest()==record['sha256'],'Corrupt scorer source bundle')
    require(scorer_build['complete'] and scorer_build['passed'] and scorer_build['source_unchanged'] and scorer_build['protected_unchanged']
        and campaign['scorer']['sha256']=='10c1be09018226745506dfbdd9ef3894ebc83c4d11029a1d41da2801a8fcd8a1'
        and scorer_build['binaries_sha256'].get(campaign['scorer']['path'])==campaign['scorer']['sha256']
        and bindings.get(campaign['scorer']['path'])==campaign['scorer']['sha256'],'Unbound scorer executable')
    require(controller['complete'] and controller['passed'] and controller['manifest_sha256']==protocol['physical_manifest']['sha256']
        and controller['source_unchanged'] and controller['error'] is None and controller['drain_error'] is None
        and drain['complete'] and drain['passed'] and drain['manifest_sha256']==protocol['physical_manifest']['sha256']
        and drain['controller_receipt_sha256']==protocol['physical_controller_receipt']['sha256']
        and len(drain['owned_groups'])==77 and all(not g['same_process'] and not g['group_exists'] for g in drain['owned_groups']),
        'Incomplete or unrelated physical controller/drain')
    require(geometry['complete'] and geometry['passed'] and geometry['all_attempts']==16384
            and geometry['independent_panel_size']==512 and geometry['schema']=='context-bridge-guide-independent-audit-v1'
            and geometry['mixture_manifest']==protocol['mixture_manifest'],'Incomplete independent geometry audit')
    for path,digest in geometry['input_sha256'].items():
        require(bindings.get(path)==digest,'Unbound geometry-audit dependency: '+path)
    contribution_path=protocol['geometry_contributions'];require(bindings.get(str(Path(contribution_path['path']).resolve()))==contribution_path['sha256']
        and geometry['contributions_sha256']==contribution_path['sha256'],'Unbound or unaudited contributions')
    contributions=jsonl(contribution_path['path']);require(len(contributions)==16384,'Lost geometry attempts')
    require(sum(r['physical_valid'] for r in contributions)==3727, 'Changed all-valid pose allocation')
    jobs=protocol['jobs'];validate_inventory(jobs)
    require(jobs==physical_inventory,'Final audit inventory differs from complete physical allocation')
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
                geometry_config=bound(job['geometry_config'])
                require(cfg['inputs']==geometry_config['inputs'] and cfg['envelope']==geometry_config['envelope'],
                    'Physical inputs/envelope changed from audited geometry')
                cloud_caps=cfg['cloud_limits']
                require(cloud_caps==dict(raw_per_cloud=1000000,raw_per_pose=2000000,
                    raw_total=2000000*job['attempts'],processed_per_cloud=1000000,
                    processed_per_pose=2000000,processed_total=2000000*job['attempts'],callback_interval=8192),
                    'Changed predeclared full-envelope cloud caps')
                require(all((r['comparison_arm'],r['population_index'],r['component'])==(job['comparison_arm'],job['population_index'],job['component']) for r in records),'Changed stratum identity')
                source_assets={r['source_rows']['path']:r['source_rows']['sha256'] for r in records};require(len(source_assets)==1,'Mixed saved source files')
                source_path,source_digest=next(iter(source_assets.items()));require(bindings.get(str(Path(source_path).resolve()))==source_digest,'Unbound original draws')
                originals=jsonl(source_path);valid=validate_panel_coverage(job,panel,records,originals,geometry_config['inputs']['regions']['source_secondary_tokens'])
                result=Path(job['result']);summary=bound(result/'summary.json');receipt=bound(job['execution_receipt'])
                require(receipt['success'] and receipt['child_drained'] and receipt['returncode']==0
                        and not receipt['timeout'] and receipt['retries']==receipt['replacements']==0,
                        'Undrained/failed physical or explicit empty job')
                require(receipt['id']==job['id']
                        and Path(receipt['terminal']['path']).resolve()==(result/'summary.json').resolve()
                        and receipt['terminal']['sha256']==bindings[str((result/'summary.json').resolve())],
                        'Execution receipt terminal or identity differs')
                scored=[];pairs=[];totals=dict(raw_points=0,processed_points=0,overlap_points=0,clouds=0)
                if job['empty']:
                    validate_empty(job,cfg,panel,summary)
                    require(summary['source_rows']==records[0]['source_rows'],'Empty source mismatch')
                    empty_rows=result/'rows.jsonl';require(str(empty_rows.resolve()) in bindings and jsonl(empty_rows)==[],
                        'Explicit empty job has unexpected physical rows')
                else:
                    producer=bound(result/'protocol.json')
                    require(producer['config']==cfg and producer['config_sha256']==job['config']['sha256']
                        and producer['source_bundle_sha256']==protocol['scorer_source_bundle']['sha256'],'Producer configuration/source bundle mismatch')
                    for path,digest in producer['input_sha256'].items():require(bindings.get(path)==digest,'Unbound producer input')
                    row_path=result/'rows.jsonl';require(str(row_path.resolve()) in bindings,'Unbound physical rows')
                    physical_rows=jsonl(row_path)
                    scored,totals=audit_rows(cfg,panel,physical_rows)
                    event_path=result/'events.jsonl';require(str(event_path.resolve()) in bindings,'Unbound physical journal')
                    ledger=audit_ledger(jsonl(event_path),cfg,job['config']['sha256'],panel,physical_rows,Path(source_path).read_text().splitlines())
                    require(all(summary[k]==v for k,v in ledger.items()),'Physical journal/summary differs')
                    pairs=[paired_weights(r,records[ordinal]['log_q_arm'],cfg) for ordinal,r in zip(valid,physical_rows)]
                    for physical_row,pair,ordinal in zip(physical_rows,pairs,valid):
                        terms=moment_terms(physical_row['envelope']['lower_volume'],pair['count_sum'],cfg['lambda'],cfg['activity'],records[ordinal]['log_q_arm'])
                        close(pair['log_rb'],terms['log_y'],'saved-count conditional noise mean')
                        pair['log_auxiliary_variance']=terms['log_auxiliary_variance']
                    for a,b in zip(scored,pairs):close(a['log_positive_importance_weight'],b['log_arithmetic'],'arithmetic control')
                    require(all(c['weight']['raw_points']<=1000000 for r in physical_rows for c in r['clouds'])
                            and totals['raw_points']<=cloud_caps['raw_total']
                            and totals['processed_points']<=cloud_caps['processed_total'],'Cloud caps exceeded')
                    require(summary['complete'] and summary['passed']
                            and summary['poses_begun']==summary['poses_completed']==len(valid)
                            and summary['clouds_begun']==summary['clouds_completed']==2*len(valid)
                            and summary['retries']==summary['replacements']==summary['new_poses_generated']==0
                            and not summary['normalizer_estimated'],'Incomplete physical stratum')
                    for key in ('raw_points','processed_points'):require(summary[key]==totals[key],'Cloud summary differs')
                scores=dict(zip(valid,scored));pair_scores=dict(zip(valid,pairs));key=(job['comparison_arm'],job['population_index'])
                for row in records:
                    enriched=dict(row);record=scores.get(row['ordinal'])
                    pair=pair_scores.get(row['ordinal'])
                    enriched['log_physical_weight']=pair['log_rb'] if pair else None
                    enriched['log_arithmetic_physical_weight']=pair['log_arithmetic'] if pair else None
                    enriched['cloud_count_sum']=pair['count_sum'] if pair else None
                    enriched['log_auxiliary_variance']=pair['log_auxiliary_variance'] if pair else None
                    enriched['arithmetic_cloud_noise_fraction']=pair['cloud_noise_fraction'] if pair else None
                    enriched['conditional_log_weight_score']=record['z_overlap'] if record else None
                    require(bool(record)==row['physical_valid'],'Missing or extra physical weight')
                    populations[key].append(enriched);output.write(json.dumps(enriched,allow_nan=False)+'\n')
                output.flush();costs[key][0]+=geometry_cpu[sid];costs[key][1]+=summary['cpu_seconds']
                results.append(dict(id=job['id'],stratum_id=sid,attempts=job['attempts'],valid_poses=len(valid),
                    geometry_cpu_seconds=geometry_cpu[sid],physical_cpu_seconds=summary['cpu_seconds'],totals=totals))
                emit(dict(kind='stratum_complete',**results[-1]))
            per_pop=[dict(comparison_arm=a,population_index=p,**population_statistics(populations[a,p],*costs[a,p])) for a in ARMS for p in range(4)]
            arithmetic_pop=[dict(comparison_arm=a,population_index=p,**population_statistics(
                [dict(r,log_physical_weight=r['log_arithmetic_physical_weight']) for r in populations[a,p]],*costs[a,p],include_cloud_variance=False)) for a in ARMS for p in range(4)]
            arithmetic_summaries={a:{g:across_populations([r['groups'][g]['physical'] for r in arithmetic_pop if r['comparison_arm']==a]) for g in GROUPS} for a in ARMS}
            arm_summaries={a:{g:across_populations([r['groups'][g]['physical'] for r in per_pop if r['comparison_arm']==a]) for g in GROUPS} for a in ARMS}
            hard_arm_summaries={a:{g:across_populations([r['groups'][g]['hard'] for r in per_pop if r['comparison_arm']==a]) for g in GROUPS} for a in ARMS}
            gates={a:{g:dict(concentration_all_populations=all(r['groups'][g]['concentration_gate'] for r in per_pop if r['comparison_arm']==a),
                population_relative_error_at_most_point_one=arm_summaries[a][g]['population_relative_standard_error'] is not None and arm_summaries[a][g]['population_relative_standard_error']<=.1) for g in GROUPS} for a in ARMS}
            for path,digest in bindings.items():require(sha(path)==digest,'Input changed during reduction')
            report=dict(schema='context-bridge-physical-report-v1',complete=True,passed=True,
                protocol_sha256=sha(args.protocol),source_sha256=sha(__file__),input_sha256=bindings,
                all_attempts=16384,population_denominator=2048,strata=results,populations=per_pop,
                arm_summaries=arm_summaries,hard_arm_summaries=hard_arm_summaries,
                pooled_A_T_tail={a:tail_statistics([r for p in range(4) for r in populations[a,p]],8192) for a in ARMS},
                pooled_fixed_bank_cloud_variance={a:cloud_variance_statistics([r for p in range(4) for r in populations[a,p]],8192) for a in ARMS},
                extension_scope=EXTENSION_SCOPE,confirmatory_gates_admitted=False,
                estimator=CLOUD_RULE,secondary_arithmetic_populations=arithmetic_pop,secondary_arithmetic_arm_summaries=arithmetic_summaries,
                comparison={g:compare_means(arm_summaries['baseline'][g],arm_summaries['bridge'][g]) for g in GROUPS},
                hard_comparison={g:compare_means(hard_arm_summaries['baseline'][g],hard_arm_summaries['bridge'][g]) for g in GROUPS},
                pilot_gates=gates,free_energy={a:within_arm_free_energy([r for r in per_pop if r['comparison_arm']==a]) for a in ARMS},
                contributions_sha256=sha(args.out/'contributions.jsonl'),cpu_seconds=time.process_time()-started,
                new_geometry_queries=0,new_cloud_draws=0,new_poses=0,finite_stability_conclusion=False,
                scope='Exploratory postgeometry-selected bank. Historical500uM frozen263-body environment with one mobile tetramer; source-informed conditional contact patterns, not a native registry classifier. No finite-system assembly, instability or equilibrium concentration conclusion.',
                estimator_description='Primary sum exp(zL)*(1+z/(2lambda))^(K0+K1)/q_arm, divided by ALL2048 attempts per population; arithmetic pair secondary. Hard-invalid zeros retained. Chart Jacobian is inside q_arm. No exponentiation of overlap score.',
                groups={'A_T':'A_patch_complete: exact neighbor set16,217 and all16 source patch tokens, extra patch tokens allowed.',
                    'T_any':'All16 source patch tokens at any neighbor set; overlaps A_T and is reported separately.',
                    'remaining_without_A_T':'Every physically valid full-wall-domain pose outside A_T, including unbound.',
                    'contact_without_A_T':'Every physically valid contact pose outside A_T.',
                    'radial_partitions':'Original full-chart six-bin partitions; all13 chart radii remain in each contribution.',
                    'A_T_tail':'Physical and hard A_T weight fractions by original full-chart radius, plus pooled leading row identities.',
                    'fixed_bank_cloud_variance':'Sum of unbiased per-pose auxiliary variances divided by the full denominator squared; conditional on the selected bank, not pose uncertainty.'},
                uncertainty_scope='All population SEs, delta-method intervals and numerical gates are descriptive because this bank was selected after its geometry results. No threshold can promote it to fresh confirmation.',
                limitations='Four populations and ESS/concentration/stratum diagnostics cannot exclude unseen modes. A failed campaign is an unresolved sampling limitation, not evidence against assembly. Positive-weight identities apply to uncapped law; failed prefixes are never replaced, clipped, treated as zero, or success-conditioned.')
            (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        except BaseException as error:
            emit(dict(kind='fatal',error=repr(error)))
            (args.out/'failure.json').write_text(json.dumps(dict(complete=False,passed=False,error=repr(error),prefix_preserved=True),indent=2)+'\n')
            raise

if __name__=='__main__':main()
