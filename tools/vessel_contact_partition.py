#!/usr/bin/env python3
"""Exhaustive vessel contact/pocket accounting from already audited labels.

This cheap pass never repeats native classification or atom-pair searches. Old
capture and q cuts are reporting predicates only. Overlapping pocket witnesses
are combined by a Boolean union, never by adding their masses.
"""
from __future__ import annotations
import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t
from analyze_r4_smc_control import Chart, Ledger, close, read, require, sha, write
from analyze_mobile_native_pocket import local_sources
from prepare_deep_far_normalizer_atlas import registration

PRIMARY = ('registered_native_entry', 'contact_no_native_entry', 'unbound_no_native_entry')
SUPPORTS = ('current_R4', 'old_native_R4', 'old_alternative_R5', 'old_alternative_R32')
PARTITIONS = ('inside_current_R4', 'outside_current_R4', 'inside_measured_pockets', 'outside_measured_pockets')


class RegionSupport:
    def __init__(self, region):
        self.region = region; self.chart = Chart(region)

    def contains(self, pose):
        value = self.chart.evaluate(pose); region = self.region
        inner=region.get('minimum_mahalanobis_radius', 0.)
        if not value['inside'] or (inner>0 and value['radius']<=inner): return False
        if math.dist(pose['position'], region['capture_center']) > region['capture_radius']: return False
        q = float(registration([pose], region['physical_metric'])[0])
        lower = region.get('minimum_original_q', 0.); upper = region.get('maximum_original_q')
        if q < lower or (q == lower and not region.get('minimum_original_q_inclusive', False)): return False
        return upper is None or q < upper or (q == upper and region.get('maximum_original_q_inclusive', True))


def partition(valid, primary, supports):
    require(type(valid) is bool and set(primary) == set(PRIMARY) and set(supports) == set(SUPPORTS), 'Missing partition labels')
    require(all(type(v) is bool for v in (*primary.values(), *supports.values())), 'Non-Boolean partition label')
    require(sum(primary.values()) == int(valid), 'Primary labels do not partition all valid poses')
    flags = dict(inside_current_R4=valid and supports['current_R4'], outside_current_R4=valid and not supports['current_R4'],
        inside_measured_pockets=valid and any(supports.values()), outside_measured_pockets=valid and not any(supports.values()))
    classes = dict(total=valid, **primary, **flags)
    classes.update({'witness:'+key: valid and value for key, value in supports.items()})
    classes.update({name+':'+key: primary[name] and flag for name in PRIMARY for key, flag in flags.items()})
    for name in ('total', *PRIMARY):
        for left, right in (PARTITIONS[:2], PARTITIONS[2:]):
            keys = (left, right) if name == 'total' else (name+':'+left, name+':'+right)
            require(sum(classes[k] for k in keys) == int(classes[name]), 'Nonexhaustive cross partition')
    return classes


class Mass:
    """Unconditional streaming sum, square sum and maximum in log arithmetic."""
    def __init__(self): self.count=0; self.logsum=self.logsum2=self.logmax=-math.inf
    def add(self, value):
        require(value is not None and math.isfinite(value), 'Selected valid weight must be finite')
        self.count += 1; self.logsum=float(np.logaddexp(self.logsum, value))
        self.logsum2=float(np.logaddexp(self.logsum2, 2*value)); self.logmax=max(self.logmax, value)
    def result(self, n):
        require(type(n) is int and n>0 and self.count<=n, 'Unconditional denominator differs')
        return dict(draws=n, nonzero=self.count, logQ=self.logsum-math.log(n) if self.count else None,
            log_sum_squared_weights=self.logsum2 if self.count else None,
            log_max_weight=self.logmax if self.count else None,
            ess=math.exp(2*self.logsum-self.logsum2) if self.count else 0.,
            max_fraction=math.exp(self.logmax-self.logsum) if self.count else None)


def independent_mass(values):
    require(len(values)==4, 'Four independent populations required')
    logs=np.asarray([-math.inf if v['logQ'] is None else v['logQ'] for v in values])
    require(not np.isnan(logs).any() and not np.isposinf(logs).any(), 'Invalid population mass')
    n={v['draws'] for v in values}; require(len(n)==1 and next(iter(n))>0, 'Population denominators differ')
    finite=np.isfinite(logs)
    if not finite.any():
        return dict(log_Q=None, population_relative_SE=None, importance_ESS=0., largest_contribution=None,
            nonzero_populations=0, unresolved='Unobserved mass, not a zero-mass result or upper bound.')
    offset=float(logs[finite].max()); x=np.exp(logs-offset)
    require((x[finite]>0).all(), 'Population scaling underflowed a positive mass')
    mean=float(x.mean()); se=float(x.std(ddof=1)/math.sqrt(len(x))/mean)
    total=math.log(next(iter(n)))+float(logsumexp(logs))
    square=float(logsumexp([v['log_sum_squared_weights'] for v in values if v['nonzero']]))
    largest=max(v['log_max_weight'] for v in values if v['nonzero'])
    return dict(log_Q=offset+math.log(mean), population_relative_SE=se,
        importance_ESS=math.exp(2*total-square), largest_contribution=math.exp(largest-total),
        nonzero_populations=int(finite.sum()), independent_populations=4, draws=4*next(iter(n)))


def contrast(populations, kind='Qz'):
    columns=[]; means=[]
    for name in PRIMARY[:2]:
        v=np.asarray([-math.inf if p[name][kind]['logQ'] is None else p[name][kind]['logQ'] for p in populations])
        if not np.isfinite(v).any(): return dict(unresolved='Native or competing contribution unobserved; no finite contrast.')
        scale=float(v[np.isfinite(v)].max()); x=np.exp(v-scale); m=float(x.mean())
        columns.append(x/m); means.append(scale+math.log(m))
    difference=columns[1]-columns[0]
    se=float(difference.std(ddof=1)/math.sqrt(len(difference)))
    return dict(beta_F_native_minus_competing=means[1]-means[0], population_SE=se,
        halfwidth_95=float(student_t.ppf(.975,len(difference)-1))*se,
        scope='Paired whole-population covariance of linear masses; delta method with Student-t interval.')


def enrich(audit_path, region_paths, out):
    audit_path=Path(audit_path).resolve(); out=Path(out).resolve(); require(not out.exists(), 'Fresh partition output required')
    ledger=Ledger(); ledger.frozen(audit_path.parent); audit=read(ledger.bind(audit_path))
    require(audit['schema'] in ('full-vessel-baseline-audit-v1','full-vessel-latent-audit-v1')
        and audit['complete'] and audit.get('native_binding'), 'Completed independent full-native audit required')
    root=Path(audit['population']); raw=ledger.bind(root/'samples.jsonl',audit['raw_sample_binding']['sha256'])
    manifest=read(ledger.bind(root/'manifest.json')); require(manifest==audit['manifest'], 'Population manifest changed')
    config_path=(root/'config.json').resolve()
    config=read(ledger.bind(config_path,audit['source_sha256'][str(config_path)])); supports={}
    require(set(region_paths)==set(SUPPORTS), 'Freeze all current and historical reporting supports')
    current_hash=(audit['reporting_guide_binding']['region_sha256'] if audit['schema']=='full-vessel-baseline-audit-v1'
        else manifest['latent_region_sha256'])
    ledger.bind(region_paths['current_R4'],current_hash)
    for name,path in region_paths.items():
        region=read(ledger.bind(path)); require(region['shape_sha256']==manifest['shape_sha256'], 'Reporting shape differs')
        require(region['physical_fixed_neighbors']==config['fixed_poses'] and region['fixed_neighbor'] in config['fixed_poses'], 'Reporting scaffold differs')
        require(region['physical_metric']==config['metadata'], 'Reporting physical metric differs')
        supports[name]=RegionSupport(region)
    sources=local_sources(__file__)
    for path in sources.values(): ledger.bind(path)
    labels=ledger.bind(audit_path.parent/'labels.jsonl'); estimates={}; count=0; invalid=0
    out.mkdir(parents=True); (out/'provenance').mkdir()
    with raw.open() as raws, labels.open() as saved, (out/'labels.jsonl').open('x') as output:
        for rtext,ltext in zip_longest(raws,saved):
            require(rtext is not None and ltext is not None, 'Lost raw or audited draw')
            row,label=json.loads(rtext),json.loads(ltext)
            require(row['draw']==label['draw']==count, 'Reordered or lost attempted draw'); count+=1
            valid=row['hard_valid']; require(label['classes']['total']==valid, 'Audit validity changed')
            primary={k:label['classes'][k] for k in PRIMARY}
            witnesses={k:s.contains(row['pose']) for k,s in supports.items()} if valid else {k:False for k in supports}
            classes=partition(valid,primary,witnesses); invalid+=int(not valid)
            if not valid: require(row['log_importance_weight'] is None and row['log_hard_weight'] is None, 'Invalid weight not zero')
            for name,selected in classes.items():
                pair=estimates.setdefault(name, {kind:Mass() for kind in ('Qz','Q0')})
                if selected:
                    pair['Qz'].add(row['log_importance_weight']); pair['Q0'].add(row['log_hard_weight'])
            output.write(json.dumps(dict(draw=row['draw'],classes=classes),allow_nan=False)+'\n')
    require(count==manifest['samples'], 'Changed original N')
    estimates={name:{kind:m.result(count) for kind,m in pair.items()} for name,pair in estimates.items()}
    for name in ('total',*PRIMARY):
        for kind in ('Qz','Q0'):
            old=audit['estimates'][name][kind]['logQ']; new=estimates[name][kind]['logQ']
            require((old is None)==(new is None), 'Changed zero/nonzero class mass')
            if old is not None: close(old,new,'Changed audited primary mass')
    ledger.recheck()
    for name,path in sources.items(): shutil.copy2(path,out/'provenance'/name)
    ledger.recheck()
    result=dict(schema='full-vessel-contact-partition-v1',complete=True,population=str(root),audit=str(audit_path),
        manifest=manifest,estimates=estimates,samples=count,invalid_draws=invalid,input_sha256=ledger.files,
        labels_sha256=sha(out/'labels.jsonl'),region_paths={k:str(Path(v).resolve()) for k,v in region_paths.items()},
        native_definition_sha256=audit['native_binding']['definition_sha256'],
        scope='Same audited all-attempt weights. Current ball, original capture and q cuts define reporting supports only; pocket union counted once. Native/core classification is reused, not repeated.',
        coverage_conclusion='unresolved until independent population, proposal and remainder diagnostics are assessed')
    write(out/'analysis.json',result); write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--audit',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    for name in SUPPORTS: p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    a=p.parse_args(); r=enrich(a.audit,{k:getattr(a,k) for k in SUPPORTS},a.out); print(json.dumps(dict(complete=r['complete'],samples=r['samples'])))


if __name__=='__main__': main()
