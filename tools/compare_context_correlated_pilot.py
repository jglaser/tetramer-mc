#!/usr/bin/env python3
"""Admit16 new rho.95 reports against immutable24 completed controls.

No old sampler or observer is rerun. Independent stream traces remain separate.
"""
import argparse
import itertools
import json
from pathlib import Path
from compare_context_prior_pilot import compare,efficiency_pair,sha,read,require,ARMS,STARTS

BASELINE_COMPARISON_SHA='9ef364c36de0d10c4099e37df9cbae56dec2831f9932c8dfa6914009d57fb397'
RHO=.95


def index(reports):
    result={(r['identity']['start'],r['identity']['stream'],r['identity']['arm']):r for r in reports}
    require(len(result)==len(reports),'Duplicate report identities')
    return result


def compare_correlations(baseline,new):
    old_by,new_by=index(baseline),index(new)
    require(set(old_by)==set(itertools.product(STARTS,range(4),ARMS)),'All24 frozen baseline reports required')
    require(set(new_by)==set(itertools.product(STARTS,range(4),('original','context'))),'All16 correlated reports required')
    keys=('shape','model','fixed_context','source_state','executable','source_bundle')
    paired=[]
    for key,r in new_by.items():
        old=old_by[key]
        require(r['schema']=='context-correlated095-pilot-contact-analysis-v1' and r['correlation']==RHO
            and r['method']=='posterior_involution','Mixed correlation or unsupported new method')
        require(old['schema']=='context-prior-pilot-contact-analysis-v1','Unreviewed baseline observer')
        require(all(r['bindings'][k]==old['bindings'][k] for k in keys),'Changed physical context/model/compiled kernel')
        require(r['bindings'].get('prior')==old['bindings'].get('prior'),'Changed normalized branch prior')
        require(r['initial_observation']==old['initial_observation'],'Changed certified initial patch geometry')
        for cadence in ('all_attempts','cycle_endpoints'):
            for descriptor in ('patch_ess','fingerprint_ess','neighbor_ess'):
                paired.append(dict(efficiency_pair(old,r,cadence,descriptor),baseline_correlation=0.,tested_correlation=RHO))
    # Reuse the already-validated24-way descriptive reduction with8 old local
    # controls and16 new reports; label correlations explicitly in its result.
    combined=list(new)+[r for k,r in old_by.items() if k[2]=='local']
    result=compare(combined)
    result.update(correlation_by_arm=dict(local=0.,original=RHO,context=RHO),
        paired_against_same_prior_rho0=paired,new_chains=16,reused_local_controls=8,reused_rho0_atlas_controls=16,
        new_sampler_cpu_seconds=sum(r['costs']['invocation_cpu_seconds'] for r in new),
        new_offline_observer_cpu_seconds=sum(r['offline_observer_cpu_seconds'] for r in new),
        compared_total_note='Combined descriptive panels reuse8 local controls; their existing costs are not newly incurred. Same-prior rho0 comparisons reuse16 more completed controls without concatenation.',
        correlated_mechanism_diagnostics=[dict(identity=r['identity'],correlation=RHO,
            proposal_mechanisms=r['metrics']['proposal_mechanisms']) for r in new])
    return result


def load_inventory(path,baseline_binding=None):
    inventory=read(path);reports=[];pins={str(path.resolve()):sha(path)}
    if baseline_binding is not None:
        require(baseline_binding[str(path.resolve())]==sha(path),'Changed baseline inventory')
    for item in inventory:
        report=Path(item['result'])/'report.json';receipt_path=Path(item['execution_receipt']);receipt=read(receipt_path)
        require(receipt['success'] is True and receipt['child_drained'] is True and receipt['returncode']==0
            and Path(receipt['terminal']['path']).resolve()==report.resolve() and receipt['terminal']['sha256']==sha(report),
            'Missing completed observer authority')
        if baseline_binding is not None:require(baseline_binding[str(report.resolve())]==sha(report),'Changed completed baseline report')
        r=read(report);require(all(r['identity'][k]==item[k] for k in ('start','stream','arm')),'Inventory/report identity differs')
        config_path=Path(item['config']);validate_config(item,r)
        pins[str(config_path.resolve())]=sha(config_path)
        reports.append(r);pins[str(report.resolve())]=sha(report);pins[str(receipt_path.resolve())]=sha(receipt_path)
    return inventory,reports,pins


def validate_config(item,report):
    """Connect a declared paired seed to the bytes actually used by the run."""
    config=Path(item['config']);digest=sha(config)
    require(digest==item['config_sha256']==report['bindings']['config'],'Changed executed configuration')
    value=read(config)
    require(type(value['seed']) is int and value['seed']==item['seed'],'Declared seed differs from executed configuration')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('inventory','baseline-inventory','baseline-comparison','out'):p.add_argument('--'+k,type=Path,required=True)
    args=p.parse_args();require(not args.out.exists(),'Fresh output required')
    require(sha(args.baseline_comparison)==BASELINE_COMPARISON_SHA,'Unreviewed baseline comparison')
    baseline=read(args.baseline_comparison);require(baseline['complete'] is True and baseline['passed'] is True,'Incomplete baseline')
    old_inventory,old,old_pins=load_inventory(args.baseline_inventory,baseline['input_sha256'])
    new_inventory,new,new_pins=load_inventory(args.inventory)
    old_id={(i['start'],i['stream'],i['arm']):i for i in old_inventory}
    for i in new_inventory:
        k=i['start'],i['stream'],i['arm']
        require(k in old_id and i['seed']==old_id[k]['seed'],'Changed matched start/stream seed')
    result=dict(schema='context-correlated095-pilot-comparison-v1',complete=True,passed=True,
        input_sha256={**old_pins,**new_pins,str(args.baseline_comparison.resolve()):sha(args.baseline_comparison)},
        **compare_correlations(old,new),geometry_queries=0,physical_draws=0)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
