#!/usr/bin/env python3
"""Passive explicit-index diagnostics from complete, independently audited scores.

No geometry, proposal draw, Poisson process, or physical-density evaluation.
Only Gaussian arithmetic and sums of saved component logs are performed.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='1'
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.special import logsumexp

ARMS=('uniform_phi92','localized_phi92')
CLASSES=('native_R5','native_complement','competing','invalid')
def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,(float,np.floating)) and not math.isfinite(x):return None
    if isinstance(x,np.generic):return x.item()
    return x
def distribution(values):
    a=np.asarray(values,float);require(len(a)>0 and np.isfinite(a).all(),'Invalid distribution')
    return dict(count=len(a),minimum=float(a.min()),p10=float(np.quantile(a,.1)),median=float(np.median(a)),mean=float(a.mean()),p90=float(np.quantile(a,.9)),maximum=float(a.max()))


def index_factors(logrho,logold,lognew):
    """All terms are normalized single-component latent densities."""
    logrho=np.asarray(logrho);logold=np.asarray(logold);lognew=np.asarray(lognew)
    require(logrho.shape==logold.shape==lognew.shape,'Incomplete component vector')
    require(np.isfinite(logold).all() and np.isfinite(lognew).all(),'Lost Gaussian defensive support')
    Qo=float(logsumexp(logrho+logold));Qn=float(logsumexp(logrho+lognew))
    logS=float(logsumexp(logrho+2*logold-lognew)-2*Qo)
    psi_old=np.exp(logrho+logold-Qo);psi_new=np.exp(logrho+lognew-Qn)
    logpenalty=Qn+logS;require(logpenalty>=-2e-12,'Cauchy-Schwarz bound failed')
    direct_chi=float(np.sum((psi_old-psi_new)**2/psi_new))
    require(math.isclose(math.expm1(logpenalty),direct_chi,abs_tol=2e-10,rel_tol=2e-9),'Chi-square identity failed')
    return dict(log_old_mixture=Qo,log_new_mixture=Qn,log_S=logS,
        index_over_marginal_second_moment=math.exp(logpenalty),index_over_old_second_moment=math.exp(Qo+logS),
        marginal_over_old_second_moment=math.exp(Qo-Qn),chi_squared_old_vs_new_responsibilities=direct_chi,
        psi_old=psi_old,psi_new=psi_new)


class SavedComponentDecoder:
    def __init__(self,guide,region):
        self.guide=guide;self.alpha=guide['defensive_uniform_shell_probability'];self.beta=guide['conditional_probability']
        require(self.alpha==.5 and self.beta==.5,'This retrospective comparison fixes alpha=beta=.5')
        comps=guide['gaussian_components'];weights=np.array([c['weight'] for c in comps]);self.rho=weights/weights.sum();self.logrho=np.log(self.rho)
        self.means=np.array([c['mean'] for c in comps]);self.lowers=[np.linalg.cholesky(c['covariance']) for c in comps]
        self.norms=np.array([-3*math.log(2*math.pi)-np.log(np.diag(L)).sum() for L in self.lowers])
        self.widths=guide['contact_widths_A'];self.radius=region['mahalanobis_radius'];self.logvolume=3*math.log(math.pi)+6*math.log(self.radius)-math.log(6)
        self.logchartdet=.5*np.linalg.slogdet(region['gaussian_chart']['covariances'][0])[1]

    def decode(self,row,arm_index):
        arm=row['arms'][arm_index];u=np.array(row['latent'])
        z=np.array([np.linalg.solve(L,u-m) for L,m in zip(self.lowers,self.means)])
        g=self.norms-.5*np.sum(z*z,axis=1)
        U=-self.logvolume if np.linalg.norm(u)<=self.radius else -math.inf
        require(len(arm['components'])==len(g),'Missing saved component')
        old=[];new=[];geometry=[];cost=[];cache={e['id']:e['cpu_seconds'] for e in row['geometry_cache']['circles']}
        for k,c in enumerate(arm['components']):
            require(c['component']==k and len(c['widths'])==len(self.widths),'Changed saved component/width order')
            ho=[];hn=[];ids=set()
            for wi,w in enumerate(c['widths']):
                require(w['width_index']==wi and w['width_A']==self.widths[wi],'Changed width')
                if w['fallback']:ho.append(g[k]);hn.append(g[k])
                else:
                    for key,target in [('old_translation_log_density',ho),('translation_log_density',hn)]:
                        value=w[key];target.append(-math.inf if value is None else value+c['angular_log_density']+self.logchartdet)
                if 'geometry_id' in w:ids.add(w['geometry_id'])
            require(len(ids)<=1,'A selected component unexpectedly requires multiple circles at one query')
            geometry.append(sorted(ids));cost.append(math.fsum(cache[i] for i in ids))
            for terms,target in [(ho,old),(hn,new)]:
                h=float(logsumexp(terms)-math.log(len(self.widths)))
                target.append(float(logsumexp([math.log(.5)+U,math.log(.25)+g[k],math.log(.25)+h])))
        factors=index_factors(self.logrho,old,new)
        require(abs(factors['log_old_mixture']-arm['old_distance_log_density'])<2e-9,'Old full mixture reconstruction failed')
        require(abs(factors['log_new_mixture']-arm['log_proposal_density'])<2e-9,'New full mixture reconstruction failed')
        if row['hard_valid']:
            require(np.min(np.array(new)-old)>=-2e-9,'Componentwise hard-valid monotonicity failed')
            require(factors['index_over_old_second_moment']<=1+2e-9,'Hard-valid second moment bound failed')
        ids=set(i for group in geometry for i in group);counts=np.array([len(g) for g in geometry]);cpu=np.array(cost)
        previous={w['geometry_id'] for a in row['arms'][:arm_index] for c in a['components'] for w in c['widths'] if 'geometry_id' in w}
        incremental=math.fsum(cache[i] for i in ids-previous)
        nongeometry=arm['score_cpu_seconds']-incremental;require(nongeometry>=-1e-9,'Geometry exceeds score timer')
        accounting=dict(full_required_distinct_circles=len(ids),full_required_geometry_CPU_seconds=math.fsum(cache[i] for i in ids),
            measured_arm_score_CPU_seconds=arm['score_cpu_seconds'],incremental_geometry_CPU_seconds=incremental,nongeometry_CPU_seconds=nongeometry)
        for name,p in [('rho',self.rho),('psi_old',factors['psi_old']),('psi_new',factors['psi_new'])]:
            accounting[name]=dict(expected_cold_query_circles=float(p@counts),expected_saved_circle_CPU_seconds=float(p@cpu))
        return dict(**factors,log_old_components=old,log_new_components=new,rho=self.rho,geometry_ids_by_component=geometry,
            saved_geometry_CPU_by_component=cost,accounting=accounting,
            old_mixture_reconstruction_error=factors['log_old_mixture']-arm['old_distance_log_density'],
            new_mixture_reconstruction_error=factors['log_new_mixture']-arm['log_proposal_density'])


def moment(logterms):
    if not logterms:return dict(rows=0,log_selected_second_moment=-math.inf,contribution_ESS=0.,largest_contribution=0.)
    s=float(logsumexp(logterms));return dict(rows=len(logterms),log_selected_second_moment=s,
        contribution_ESS=math.exp(2*s-float(logsumexp(2*np.array(logterms)))),largest_contribution=math.exp(max(logterms)-s))


def summarize(records):
    metrics=('index_over_marginal_second_moment','index_over_old_second_moment','marginal_over_old_second_moment')
    groups={}
    for label,selected in [('all',records),*[(kind,[r for r in records if r['group']=='breadth' and r['coverage_class']==kind]) for kind in CLASSES],
            *[('critical_'+source,[r for r in records if r['group']=='critical' and r['source']['arm']==source]) for source in ['baseline','expanded']]]:
        groups[label]={}
        for arm in ARMS:
            arms=[r['arms'][arm] for r in selected]
            groups[label][arm]=dict(rows=len(arms),hard_valid=sum(r['hard_valid'] for r in selected),
                ratios={m:distribution([a[m] for a in arms]) for m in metrics},
                full_circle_counts=distribution([a['accounting']['full_required_distinct_circles'] for a in arms]),
                accounting={prior:{key:distribution([a['accounting'][prior][key] for a in arms]) for key in ['expected_cold_query_circles','expected_saved_circle_CPU_seconds']} for prior in ['rho','psi_old','psi_new']})
    critical={}
    for source in ['baseline','expanded']:
        selected=[r for r in records if r['group']=='critical' and r['source']['arm']==source];critical[source]={}
        for arm in ARMS:
            all_terms={name:[] for name in ['old_distance','indexed_arc','marginal_arc']};populations={}
            for population in ['r00','r01','r02','r03']:
                selected_pop=[r for r in selected if r['source']['id']==population];terms={name:[] for name in all_terms}
                for r in selected_pop:
                    src=r['source'];a=r['arms'][arm]
                    numerator=sum(src['paired_log_weights'])+src['log_q']-math.log(src['source_attempted_draws'])
                    for key,correction in [('old_distance',-a['log_old_mixture']),('indexed_arc',a['log_S']),('marginal_arc',-a['log_new_mixture'])]:
                        terms[key].append(numerator+correction);all_terms[key].append(numerator+correction-math.log(4))
                populations[population]={name:moment(v) for name,v in terms.items()}
            pooled={name:moment(v) for name,v in all_terms.items()}
            critical[source][arm]=dict(selected_rows=len(selected),independent_source_populations=4,populations=populations,selected_moments=pooled,
                index_over_old_second_moment=math.exp(pooled['indexed_arc']['log_selected_second_moment']-pooled['old_distance']['log_selected_second_moment']),
                index_over_marginal_second_moment=math.exp(pooled['indexed_arc']['log_selected_second_moment']-pooled['marginal_arc']['log_selected_second_moment']))
    return dict(groups=groups,critical_source_diagnostics=critical)


def analyze(execution):
    root=Path(execution).resolve();status=read(root/'status.json');audit=read(root/'independent-audit.json')
    require(status['complete'] and audit['complete'],'Completed independent audit is required before retrospective analysis')
    require(audit['schema']=='independent-contact-arc-density-audit-v1','Unexpected independent audit')
    paths=[root/'status.json',root/'independent-audit.json',root/'protocol.json',root/'freeze.json',root/'rust/summary.json',root/'rust/manifest.json',root/'rust/scores.jsonl',root/'common/preparation/saved-scores.json',root/'common/region.json',*[root/'common'/f'{a}.json' for a in ARMS],Path(__file__).resolve()]
    bindings={str(p):sha(p) for p in paths}
    scorespath=str(root/'rust/scores.jsonl');require(bindings[scorespath]==audit['input_sha256'][scorespath],'Saved scores changed after independent audit')
    frozen=read(root/'freeze.json')['files']
    for p in paths:
        if p.is_relative_to(root) and str(p.relative_to(root)).startswith('common/'):
            require(sha(p)==frozen[str(p.relative_to(root))],'Frozen scoring input changed')
    rows=[json.loads(s) for s in (root/'rust/scores.jsonl').read_text().splitlines()]
    saved=read(root/'common/preparation/saved-scores.json')['rows'];region=read(root/'common/region.json')
    require(len(rows)==len(saved)==206,'Exact 206-query allocation required')
    decoders=[SavedComponentDecoder(read(root/'common'/f'{a}.json'),region) for a in ARMS];records=[]
    for s,r in zip(saved,rows):
        require(s['id']==r['id'] and s['latent']==r['latent'],'Probe identity changed')
        for flag in ['hard_valid','shell_valid','capture_valid']:require(s['saved_geometry'][flag]==r[flag],'Saved flags changed')
        item=dict(id=r['id'],latent=r['latent'],group=s['group'],hard_valid=r['hard_valid'],shell_valid=r['shell_valid'],capture_valid=r['capture_valid'],
            coverage_class=s['metadata'].get('coverage_class'),arms={arm:decoder.decode(r,i) for i,(arm,decoder) in enumerate(zip(ARMS,decoders))})
        if s['group']=='critical':item['source']=s['metadata']['source']
        else:item['source_metadata']=s['metadata']
        records.append(item)
    for kind in CLASSES:require(sum(r['coverage_class']==kind for r in records)==32,'Breadth allocation changed')
    require(sum(r['group']=='critical' for r in records)==78,'Critical allocation changed')
    result=dict(schema='contact-arc-explicit-index-retrospective-v1',complete=True,records=records,**summarize(records),input_sha256=bindings,
        queries=206,densities=412,new_pose_draws=0,new_Poisson_clouds=0,new_geometry_queries=0,
        scope='Passive reconstruction of saved component densities. Critical paired moments cover a previously selected subset only; no integrated variance, speedup, effective sample size or equilibrium conclusion.',
        accounting_scope='Cold query geometry counts and saved geometry CPU weighted by rho, old responsibilities or new responsibilities; excludes draw costs and is not a measured indexed sampler.')
    for p,h in bindings.items():require(sha(p)==h,'Input changed during retrospective analysis')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(not a.out.exists(),'New output directory required');started=time.process_time();result=analyze(a.execution)
    a.out.mkdir(parents=True);result['postprocessing_CPU_seconds']=time.process_time()-started
    (a.out/'analysis.json').write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n');shutil.copyfile(__file__,a.out/Path(__file__).name)
    lines=['# Retrospective explicit-index diagnostic','',result['scope'],'',result['accounting_scope'],'','## Selected critical-subset moments','',
        '| Source | Azimuth | Indexed / old M2 | Indexed / full arc M2 | Selected-contribution ESS (indexed) |','|---|---|---:|---:|---:|']
    for source,arms in result['critical_source_diagnostics'].items():
        for arm,v in arms.items():lines.append(f"| {source} | {arm} | {v['index_over_old_second_moment']:.5g} | {v['index_over_marginal_second_moment']:.5g} | {v['selected_moments']['indexed_arc']['contribution_ESS']:.3g} |")
    lines+=['','The original attempted-draw denominators are retained per population; the source summary averages four independent populations, including zero selected contributions. Selection by previously observed critical weights prevents interpreting this as a converged integrated second moment.','',
        '## All breadth probes','', '| Class | Azimuth | Indexed / full arc M2, min / median / max | Indexed / old M2, min / median / max |','|---|---|---:|---:|']
    for kind in CLASSES:
        for arm,v in result['groups'][kind].items():
            values=[]
            for metric in ['index_over_marginal_second_moment','index_over_old_second_moment']:
                d=v['ratios'][metric];values.append(' / '.join(f'{d[k]:.4g}' for k in ['minimum','median','maximum']))
            lines.append(f"| {kind} | {arm} | {values[0]} | {values[1]} |")
    lines+=['','Invalid probes are retained; their physical integrand is zero and the hard-valid monotonicity bound does not apply. All per-probe components, responsibilities, accounting costs and source identities are in `analysis.json`.','',
        '## Geometry accounting','', '| Azimuth | Full circles/query (mean) | Chosen circles/query, prior / old responsibility | Saved circle ms/query, prior / old responsibility |','|---|---:|---:|---:|']
    for arm,v in result['groups']['all'].items():
        c=v['accounting'];lines.append(f"| {arm} | {v['full_circle_counts']['mean']:.4g} | {c['rho']['expected_cold_query_circles']['mean']:.4g} / {c['psi_old']['expected_cold_query_circles']['mean']:.4g} | {1000*c['rho']['expected_saved_circle_CPU_seconds']['mean']:.4g} / {1000*c['psi_old']['expected_saved_circle_CPU_seconds']['mean']:.4g} |")
    lines+=['','These averages describe the fixed probe set, not draws from an indexed sampler. Proposal generation, index responsibility calculation, target evaluation and cache reuse must be measured in a subsequent bounded experiment.']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n');print(a.out/'analysis.json')


if __name__=='__main__':main()
