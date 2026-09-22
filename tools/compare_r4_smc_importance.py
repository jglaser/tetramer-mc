#!/usr/bin/env python3
"""Compare frozen R4 SMC and importance analyses without replaying any audit.

Arithmetic means/covariances use independent whole-population region masses.
The SMC terminal mass is Zhat times a terminal indicator; its initial bridge
mass is NEVER substituted for the physical hard-only integral.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import shutil
from analyze_r4_smc_control import (CLASSES, PRIMARY, STRATA, Ledger, close, nullable_log,
    population_statistics, read, require, sha, validate_config_identity, write)
from analyze_mobile_native_pocket import local_sources


STRATA_DEFINITION=dict(radial_edges=[0.,2.,3.,4.],angular_projection_squared_edges=[0.,4.,9.,16.],
    latent_orthants='six signs, zero assigned positive; all 64 bins retained')


def verify_log_sum(total,parts,message):
    finite=[nullable_log(v) for v in parts if v is not None]
    wanted=max(finite)+math.log(sum(math.exp(v-max(finite)) for v in finite)) if finite else -math.inf
    actual=nullable_log(total)
    if wanted==-math.inf:require(actual==-math.inf,message)
    else:close(actual,wanted,message)


def validate_strata(smc,importance):
    require(importance['strata_definition']==STRATA_DEFINITION,'Predeclared stratum geometry differs')
    for pop in smc['populations']:
        terminal=pop['terminal']
        for family,size in STRATA.items():
            for region in CLASSES:
                parts=terminal['log_strata'][family][region]
                require(len(parts)==size,'SMC stratum array length differs')
                verify_log_sum(terminal['log_masses'][region],parts,'SMC strata do not sum to physical class mass')
    for arm in importance['arms'].values():
        for family,size in STRATA.items():
            for region in CLASSES:
                parts=arm['strata'][family][region]
                require(len(parts)==size,'Importance stratum array length differs')
                total={p['id']:p for p in arm['estimates'][region]['populations']}
                for kind in ('Qz','Q0'):
                    by_bin=[{p['id']:p for p in item['populations']} for item in parts]
                    require(all(set(bin)==set(total) for bin in by_bin),'Missing importance stratum population')
                    for pid,value in total.items():
                        verify_log_sum(value['log_'+kind],[bin[pid]['log_'+kind] for bin in by_bin],
                                       'Importance strata do not sum to physical class mass')


def compare_mass(left,right):
    if left['log_Q'] is None or right['log_Q'] is None:
        return dict(passed=False,unresolved='At least one contribution unobserved; no epsilon or upper-bound substitution.')
    difference=left['log_Q']-right['log_Q']
    se=math.hypot(left['population_relative_SE'],right['population_relative_SE'])
    return dict(log_mass_difference_SMC_minus_importance=difference,combined_population_SE=se,
        absolute_passed=abs(difference)<=.2,three_SE_passed=abs(difference)<=3*se+1e-12,
        passed=abs(difference)<=.2 and abs(difference)<=3*se+1e-12)


def importance_rows(arm,kind,family=None,index=None):
    require(kind in ('Qz','Q0'),'Physical importance mass kind required')
    records=sorted(arm['populations'],key=lambda p:p['id'])
    require(len(records)==4 and len({p['id'] for p in records})==4
        and len({p['seed'] for p in records})==4,'Incomplete importance population allocation')
    rows=[{} for _ in records]
    for name in CLASSES:
        estimate=arm['estimates'][name] if family is None else arm['strata'][family][name][index]
        values=sorted(estimate['populations'],key=lambda p:p['id'])
        require(len(values)==len(records),'Dropped importance population')
        for i,(record,value) in enumerate(zip(records,values)):
            require(record['id']==value['id'] and record['seed']==value['seed']
                and record['samples']==value['draws']==arm['allocation']['samples'],'Changed unconditional importance denominator')
            rows[i][name]=value['log_'+kind];nullable_log(rows[i][name])
    for row in rows:
        for total,parts in [('registered_native_entry',('old_R5_intersection_native','remaining_R4_native')),
                            ('total',('registered_native_entry','contact_no_native_entry','unbound_no_native_entry'))]:
            verify_log_sum(row[total],[row[c] for c in parts],'Importance class masses do not partition their parent')
    return rows


def smc_rows(smc,kind,family=None,index=None):
    require(kind in ('Qz','Q0'),'Physical SMC mass kind required')
    allocation=smc['protocol_snapshot']['allocation'];pops=smc['populations']
    expected={p['id']:p for p in smc['protocol_snapshot']['jobs']}
    require(len(pops)==allocation['independent_populations']==len(expected)==4
        and {p['id'] for p in pops}==set(expected),'Dropped independent SMC population')
    rows=[]
    for pop in sorted(pops,key=lambda p:p['id']):
        require(pop['seed']==expected[pop['id']]['seed'] and pop['initial_draws']==allocation['unconditional_initialization_draws_each'],
                'Changed SMC population identity or initialization denominator')
        terminal=pop['terminal'];n=terminal['particles'];counts=terminal['counts'];total=terminal['log_masses']['total']
        require((pop['zero_estimate'] and n==0) or (not pop['zero_estimate'] and n==allocation['particles_each']), 'Terminal population count differs')
        require(counts['total']==n and counts['registered_native_entry']==counts['old_R5_intersection_native']+counts['remaining_R4_native']
            and n==sum(counts[k] for k in ('registered_native_entry','contact_no_native_entry','unbound_no_native_entry')),'SMC terminal partition differs')
        for name in CLASSES:
            wanted=-math.inf if n==0 or counts[name]==0 or total is None else total+math.log(counts[name]/n)
            actual=nullable_log(terminal['log_masses'][name])
            if wanted==-math.inf: require(actual==-math.inf,'Nonzero mass for zero terminal count')
            else:close(actual,wanted,'Terminal mass must be normalizer times indicator, not endpoint fraction')
        if kind=='Q0':
            require(family is None,'No initial hard-only stratum estimator was archived')
            row=pop['initial_physical_hard_log_masses']
        elif family is None:row=terminal['log_masses']
        else:row={name:terminal['log_strata'][family][name][index] for name in CLASSES}
        require(set(row)==set(CLASSES),'Missing physical region mass')
        rows.append({k:row[k] for k in CLASSES})
    return rows


def identity(smc,importance,importance_config):
    require(smc['schema']=='smc-r4-control-analysis-v1' and smc['complete']
        and importance['schema']=='contact-confirmation-comparison-v1' and importance['complete'],'Completed matching analysis schemas required')
    target=smc['protocol_snapshot']['physical_target']
    for key in ('shape_sha256','region_sha256'):
        require(target[key]==importance[key],'Different physical '+key)
    require(smc['native_definition_sha256']==importance['native_definition']['definition_sha256'],'Complete native observer differs')
    require(smc['protocol_snapshot']['proposal']['reference_region_sha256']==importance['reference_region_sha256'],'Old R5 reporting chart differs')
    require(target['measure']=='d3t in Angstrom cubed times normalized proper SO(3) Haar','Unexpected physical measure')
    allowed=set(smc['protocol_snapshot'].get('proposal_control_difference',{}).get('changed_config_fields',{}))
    require(allowed<=set(('translation_steps','rotation_steps_deg')),'Undeclared physical exclusions')
    validate_config_identity(smc['physical_config'],importance_config,allowed)
    for target_key,config_key in [('fixed_poses','fixed_poses'),('capture_center','capture_center'),('capture_radius','capture_radius'),
        ('activity','reservoir_density'),('depletant_radius','depletant_radius')]:
        require(target[target_key]==importance_config[config_key],'Different physical '+target_key)
    seeds={p['seed'] for p in smc['populations']}
    require(all(seeds.isdisjoint(p['seed'] for p in arm['populations']) for arm in importance['arms'].values()),'Methods share population seeds')
    return sorted(allowed)


def compare(smc,importance,importance_config):
    exclusions=identity(smc,importance,importance_config)
    validate_strata(smc,importance)
    smc_summary={kind:population_statistics(smc_rows(smc,kind)) for kind in ('Qz','Q0')}
    arms={}
    for name,arm in importance['arms'].items():
        independent={kind:population_statistics(importance_rows(arm,kind)) for kind in ('Qz','Q0')}
        masses={kind:{c:compare_mass(smc_summary[kind]['estimates'][c],independent[kind]['estimates'][c]) for c in CLASSES} for kind in ('Qz','Q0')}
        x,y=smc_summary['Qz']['free_energy_contrast'],independent['Qz']['free_energy_contrast']
        contrast=dict(passed=False,unresolved='Native or no-entry terminal mass unobserved; independent importance remains necessary.')
        if 'SE' in x and 'SE' in y:
            difference=x['beta_F_native_minus_noentry']-y['beta_F_native_minus_noentry'];se=math.hypot(x['SE'],y['SE'])
            contrast=dict(difference=difference,combined_population_SE=se,passed=abs(difference)<=.2 and abs(difference)<=3*se+1e-12)
        strata={}
        for family,size in STRATA.items():
            entries=[]
            for i in range(size):
                a=population_statistics(smc_rows(smc,'Qz',family,i))['estimates']
                b=population_statistics(importance_rows(arm,'Qz',family,i))['estimates'];parts={}
                for region in CLASSES:
                    comparison=compare_mass(a[region],b[region]);fractions=[]
                    for piece,whole in ((a[region],smc_summary['Qz']['estimates'][region]),(b[region],independent['Qz']['estimates'][region])):
                        fractions.append(math.exp(piece['log_Q']-whole['log_Q']) if piece['log_Q'] is not None and whole['log_Q'] is not None else 0.)
                    comparison.update(observed_class_fractions=fractions,decision_relevant=region in PRIMARY and max(fractions)>=.01)
                    parts[region]=comparison
                entries.append(parts)
            strata[family]=entries
        arms[name]=dict(importance_population_statistics=independent,masses=masses,direct_free_energy_contrast=contrast,strata=strata,
            primary_Qz_mass_agreement=all(masses['Qz'][c]['passed'] for c in PRIMARY),
            significant_primary_strata_agree=all(c['passed'] for entries in strata.values() for entry in entries for c in entry.values() if c['decision_relevant']))
    return dict(schema='r4-smc-importance-comparison-v1',complete=True,smc_population_statistics=smc_summary,arms=arms,
        proposal_only_config_exclusions=exclusions,primary_regions=list(PRIMARY),populations_pooled=False,
        scope='Independent linear region masses and whole-population covariance. Qz uses SMC terminal Zhat times indicator; Q0 uses unresampled initial H/g. Bridge normalizers, terminal fractions alone, and descendant IID errors are excluded. Zero observations do not bound missing mass; no finite-system conclusion follows.',
        assembly_conclusion='unresolved')


def run(smc_path,importance_path,out):
    smc_path=Path(smc_path).resolve();importance_path=Path(importance_path).resolve();out=Path(out).resolve()
    require(not out.exists(),'Fresh comparison directory required');ledger=Ledger()
    ledger.frozen(smc_path.parent);ledger.frozen(importance_path.parent)
    smc=read(ledger.bind(smc_path));importance=read(ledger.bind(importance_path))
    protocol_path=importance_path.parent/'provenance/campaign-protocol.json'
    protocol=read(ledger.bind(protocol_path,importance['protocol_sha256']))
    config_path=Path(importance['campaign'])/'bank/provenance/config.json';config=read(ledger.bind(config_path))
    region_path=Path(importance['campaign'])/'bank/provenance/region.json'
    region=read(ledger.bind(region_path,importance['region_sha256']))
    for rk,ck in [('physical_fixed_neighbors','fixed_poses'),('capture_center','capture_center'),('capture_radius','capture_radius'),
        ('activity','reservoir_density'),('depletant_radius','depletant_radius'),('physical_metric','metadata')]:
        require(region[rk]==config[ck],'Importance physical config differs from frozen region')
    require(protocol['region_sha256']==importance['region_sha256'] and protocol['supplemental_definition_sha256']==importance['supplemental_definition_sha256'],'Importance protocol/reporting partition differs')
    ledger.bind(config['shape'],importance['shape_sha256'])
    source={str(path):sha(path) for path in local_sources(__file__).values()}
    result=compare(smc,importance,config);ledger.recheck()
    require(all(sha(p)==h for p,h in source.items()),'Comparison source changed')
    result.update(inputs=dict(smc=str(smc_path),importance=str(importance_path)),input_sha256=ledger.files,source_sha256=source)
    out.mkdir(parents=True);(out/'provenance').mkdir()
    for path in source:shutil.copy2(path,out/'provenance'/Path(path).name)
    write(out/'analysis.json',result)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smc',type=Path,required=True)
    p.add_argument('--importance',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=run(a.smc,a.importance,a.out)
    print({name:arm['primary_Qz_mass_agreement'] for name,arm in r['arms'].items()},flush=True)


if __name__=='__main__':main()
