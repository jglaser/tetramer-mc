#!/usr/bin/env python3
"""Retrospective index-estimator arithmetic on the 206 saved line-guide scores.

No poses, geometry queries, depletant clouds or native classifications are
generated. This does not alter any ongoing campaign's full-mixture estimator.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.special import logsumexp
import report_contact_arc_index as common

require, read, sha = common.require, common.read, common.sha
MODES = ('old_gaussian', 'full_mixture', 'primitive_index', 'mixed_index')
CLASSES = ('native_R5', 'native_complement', 'competing', 'invalid')


def factors(alpha, weights, log_uniform, log_gaussian, log_conditioned):
    """Physical-support second moments for primitive and grouped indices.

    Arrays have shape [axis, Gaussian]. Each h and g is a normalized component
    density, and h>=g at this physical target point. The full q is their actual
    mixture; neither index variant replaces its sampling law.
    """
    w=np.asarray(weights,float);g=np.asarray(log_gaussian,float);h=np.asarray(log_conditioned,float)
    require(0<alpha<1 and h.ndim==2 and h.shape[1:]==g.shape==w.shape,'Changed indexed component shapes')
    require(np.isfinite(g).all() and np.isfinite(h).all() and np.all(w>0) and abs(w.sum()-1)<1e-12,
            'Invalid physical-support component density')
    require(np.all(h>=g[None,:]-2e-12),'Conditioned component lost physical support')
    m=h.shape[0];logrho=np.log(w)[None,:]-math.log(m)
    lu=math.log(alpha)+log_uniform;lg=math.log1p(-alpha)+g
    old=float(np.logaddexp(lu,logsumexp(np.log(w)+lg)))
    new=float(np.logaddexp(lu,logsumexp(logrho+math.log1p(-alpha)+h)))
    # Primitive index is b=uniform or (Gaussian k, axis j).
    primitive=float(np.logaddexp(lu,logsumexp(logrho+math.log1p(-alpha)+2*g[None,:]-h))-2*old)
    # Group one copy of the uniform channel with each Gaussian/axis pair.
    loga=np.logaddexp(lu,lg)[None,:]
    logb=np.logaddexp(lu,math.log1p(-alpha)+h)
    mixed=float(logsumexp(logrho+2*loga-logb)-2*old)
    require(-new<=mixed+2e-12 and mixed<=primitive+2e-12 and primitive<=-old+2e-12,
            'Rao-Blackwell/monotonicity ordering failed')
    logs=dict(old_gaussian=-old,full_mixture=-new,primitive_index=primitive,mixed_index=mixed)
    return dict(log_old_mixture=old,log_new_mixture=new,log_second_moment_factors=logs,
        ratios={mode:dict(over_old=math.exp(value+old),over_full=math.exp(value+new)) for mode,value in logs.items()},
        primitive_chi_squared_responsibility_penalty=math.expm1(max(0.,new+primitive)),
        mixed_chi_squared_responsibility_penalty=math.expm1(max(0.,new+mixed)))


def decode(row, guide, region):
    require(guide['conditional_probability']==1. and guide['schema']=='defensive-hard-free-line-guide-v1',
            'This exploratory comparison is beta=1 only')
    axes=row['density_details']['axes'];components=guide['gaussian_components']
    require([a['axis'] for a in axes]==guide['raw_translation_axes'],'Saved axis order differs')
    w=np.array([c['weight'] for c in components]);w=w/w.sum();g=None;logs=[]
    for axis in axes:
        require([c['component'] for c in axis['components']]==list(range(len(w))),'Missing saved Gaussian component')
        localg=np.array([c['gaussian_log_density'] for c in axis['components']])
        if g is None:g=localg
        require(np.array_equal(g,localg),'Gaussian density differs by axis')
        h=[]
        for c,logg in zip(axis['components'],g):
            mass=c['conditional_mass'];require(0<=mass<=1+1e-12 and c['fallback']==(mass<=guide['minimum_conditional_mass']),
                                              'Saved floor/fallback differs')
            h.append(logg if c['fallback'] else logg-math.log(mass) if c['query_coordinate_allowed'] else -math.inf)
        logs.append(h)
    logu=-(3*math.log(math.pi)+6*math.log(region['mahalanobis_radius'])-math.log(6)) if row['shell_valid'] else -math.inf
    alpha=guide['defensive_uniform_shell_probability'];h=np.asarray(logs)
    old=float(np.logaddexp(math.log(alpha)+logu,math.log1p(-alpha)+logsumexp(np.log(w)+g)))
    new=float(np.logaddexp(math.log(alpha)+logu,math.log1p(-alpha)+logsumexp(np.log(w)[None,:]-math.log(len(axes))+h)))
    require(abs(old-row['baseline_log_density'])<2e-9 and abs(new-row['log_proposal_density'])<2e-9,
            'Saved complete density reconstruction failed')
    valid=row['hard_valid'] and row['capture_valid'] and row['shell_valid']
    result=factors(alpha,w,logu,g,h) if valid else None
    return dict(valid=valid,factors=result,log_old_mixture=old,log_new_mixture=new,
        geometry_CPU_seconds=sum(a['geometry_cpu_seconds'] for a in axes),
        full_score_CPU_seconds=row['density_cpu_seconds'],
        invalid_scope=None if valid else 'Retained with identically zero physical integrand; indexed variance factor is inapplicable.')


def critical_summary(records):
    answer={}
    for source in ('baseline','expanded'):
        selected=[r for r in records if r['group']=='critical' and r['source']['arm']==source]
        allterms={mode:[] for mode in MODES};populations={}
        for population in ('r00','r01','r02','r03'):
            terms={mode:[] for mode in MODES}
            for r in selected:
                if r['source']['id']!=population:continue
                require(r['valid'],'Critical source has zero physical support')
                s=r['source'];numerator=sum(s['paired_log_weights'])+s['log_q']-math.log(s['source_attempted_draws'])
                for mode,correction in r['factors']['log_second_moment_factors'].items():
                    terms[mode].append(numerator+correction);allterms[mode].append(numerator+correction-math.log(4))
            populations[population]={mode:common.moment(values) for mode,values in terms.items()}
        pooled={mode:common.moment(values) for mode,values in allterms.items()}
        answer[source]=dict(selected_rows=len(selected),source_populations=4,populations=populations,selected_moments=pooled,
            ratios={mode:dict(over_old=math.exp(pooled[mode]['log_selected_second_moment']-pooled['old_gaussian']['log_selected_second_moment']),
                             over_full=math.exp(pooled[mode]['log_selected_second_moment']-pooled['full_mixture']['log_selected_second_moment'])) for mode in MODES})
    return answer


def fresh_cost_inputs(root, bindings):
    """Authenticate the saved timing rows without rerunning any audit or geometry."""
    root=Path(root).resolve();analysis=read(root/'analysis.json');status=read(root/'status.json')
    require(status['complete'] and analysis['complete'] and analysis['total_attempted_draws']==2048,'Expected completed fixed fresh proposal comparison')
    require(status['analysis_sha256']==sha(root/'analysis.json'),'Fresh analysis differs from completed status')
    require(analysis['protocol_sha256']==status['protocol_sha256']==sha(root/'protocol.json'),'Fresh protocol binding differs')
    for p in (root/'analysis.json',root/'status.json',root/'protocol.json'):bindings[str(p)]=sha(p)
    populations=analysis['populations']
    require(sorted((p['arm'],p['population']) for p in populations)==
        [(arm,f'r{i:02}') for arm in ('baseline','xyz') for i in range(4)],'Fresh population identities changed')
    inputs=[]
    for population in sorted((p for p in populations if p['arm']=='xyz'),key=lambda p:p['population']):
        path=root/'xyz'/population['population']/'samples.jsonl';summary_path=path.parent/'summary.json'
        for source in (path,summary_path):
            digest=sha(source)
            require(population['input_sha256'].get(str(source))==digest,'Fresh population timing input changed after complete audit')
            bindings[str(source)]=digest
        summary=read(summary_path)
        require(summary['complete'] and summary['samples']==256 and summary['samples_sha256']==sha(path),'Changed saved fresh score rows')
        inputs.append((path,summary))
    return inputs


def fresh_costs(root, bindings):
    inputs=fresh_cost_inputs(root,bindings)
    c=dict(rows=0,invalid=0,conditional=0,selected_axis_bitidentical=0);times={k:0. for k in
        ('full_proposal','draw','full_density','invalid_density','geometry','selected_axis','selected_valid_axis','selected_bitidentical_axis')}
    for path,summary in inputs:
        times['full_proposal']+=summary['fresh_total_cpu_seconds']
        rows=[json.loads(s) for s in path.read_text().splitlines()];require(len(rows)==256,'Fresh allocation changed')
        for r in rows:
            valid=r['hard_valid'] and r['capture_valid'] and r['shell_valid'];draw=r['draw'];c['rows']+=1;c['invalid']+=not valid;c['conditional']+=draw['conditional']
            times['draw']+=r['draw_cpu_seconds'];times['full_density']+=r['density_cpu_seconds']
            if not valid:times['invalid_density']+=r['density_cpu_seconds']
            for axis in r['density_details']['axes']:
                cost=axis['geometry_cpu_seconds'];times['geometry']+=cost
                if draw['conditional'] and axis['axis']==draw['axis']:
                    times['selected_axis']+=cost
                    if valid:times['selected_valid_axis']+=cost
                    if all(axis.get(k)==draw['geometry'].get(k) for k in ('origin','direction','segment','hard_free_intervals','empty_reason')):
                        c['selected_axis_bitidentical']+=1;times['selected_bitidentical_axis']+=cost
    require(c['rows']==1024,'Exactly four saved xyz populations required')
    total=times['full_proposal']
    return dict(counts=c,measured_CPU_seconds=times,
        zero_replacement_cost_ceiling={name:total/(total-times[key]) for name,key in
            [('eliminate_all_density','full_density'),('eliminate_invalid_density','invalid_density'),('reuse_selected_axis','selected_axis')]},
        primitive_index_geometry_requests_per_fresh_draw=c['conditional']/c['rows'],
        current_geometry_requests_per_fresh_draw=3+c['conditional']/c['rows'],
        scope='Saved proposal-only instrumentation; these Amdahl ceilings assume zero replacement overhead on that workload. They are not a physical sampler speedup, an index-sampler timing or an equilibrium ESS estimate. Primitive index needs cheap Q_old plus its already-sampled conditional mass; fullq geometry is absent.')


def analyze(execution, fresh=None):
    root=Path(execution).resolve();status=read(root/'status.json');audit=read(root/'independent-audit.json')
    require(status['complete'] and audit['complete'] and audit['schema']=='independent-hard-free-line-audit-v1','Completed independent full-score audit required')
    paths=[root/'status.json',root/'independent-audit.json',root/'protocol.json',root/'freeze.json',root/'scored/probes.jsonl',
           root/'scored/summary.json',root/'common/preparation/saved-scores.json',root/'common/xyz.json',root/'common/region.json',Path(__file__).resolve(),Path(common.__file__).resolve()]
    bindings={str(p):sha(p) for p in paths};frozen=read(root/'freeze.json')['files']
    for p in paths:
        if p.is_relative_to(root/'common'):require(sha(p)==frozen[str(p.relative_to(root))],'Frozen score input changed')
    require(bindings[str(root/'scored/probes.jsonl')]==audit['input_sha256'][str(root/'scored/probes.jsonl')],'Scores changed after independent audit')
    rows=[json.loads(s) for s in (root/'scored/probes.jsonl').read_text().splitlines()];saved=read(root/'common/preparation/saved-scores.json')['rows']
    require(len(rows)==len(saved)==206,'Exact 206-query allocation required');guide=read(root/'common/xyz.json');region=read(root/'common/region.json');records=[]
    for r,s in zip(rows,saved):
        require(r['id']==s['id'] and r['latent']==s['latent'],'Saved query identity changed')
        for flag in ('hard_valid','capture_valid','shell_valid'):require(r[flag]==s['saved_geometry'][flag],'Saved physical flag changed')
        item=dict(id=r['id'],group=s['group'],coverage_class=s['metadata'].get('coverage_class'),**decode(r,guide,region))
        if s['group']=='critical':item['source']=s['metadata']['source']
        else:item['source_metadata']=s['metadata']
        records.append(item)
    require(sum(r['group']=='critical' for r in records)==78,'Critical allocation changed')
    breadth={}
    for name in CLASSES:
        selected=[r for r in records if r['coverage_class']==name];require(len(selected)==32,'Breadth allocation changed')
        valid=[r for r in selected if r['valid']]
        breadth[name]=dict(rows=32,valid=len(valid),ratios={mode:{ratio:common.distribution([r['factors']['ratios'][mode][ratio] for r in valid])
            for ratio in ('over_old','over_full')} for mode in MODES} if valid else None)
    result=dict(schema='hard-free-line-index-retrospective-v1',complete=True,records=records,queries=206,critical=critical_summary(records),breadth=breadth,
        full_score_CPU_seconds=sum(r['full_score_CPU_seconds'] for r in records),full_geometry_CPU_seconds=sum(r['geometry_CPU_seconds'] for r in records),
        fresh_proposal_cost_accounting=fresh_costs(fresh,bindings) if fresh else None,input_sha256=bindings,
        new_pose_draws=0,new_Poisson_clouds=0,new_geometry_queries=0,new_native_classifier_calls=0,
        scope='Exploratory arithmetic on previously selected critical4/74 rows and fixed breadth controls, not a converged integrated second moment or physical speedup. Primitive/mixed index estimates are not substituted into any current physical campaign. All32 invalid controls remain visible with zero physical integrand; important-region coverage is unresolved.')
    for p,h in bindings.items():require(sha(p)==h,'Input changed during passive comparison')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execution',type=Path,required=True);p.add_argument('--fresh',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(not a.out.exists(),'New output directory required');started=time.process_time();result=analyze(a.execution,a.fresh)
    a.out.mkdir(parents=True);result['postprocessing_CPU_seconds']=time.process_time()-started
    (a.out/'analysis.json').write_text(json.dumps(common.safe(result),indent=2,allow_nan=False)+'\n');shutil.copyfile(__file__,a.out/Path(__file__).name)
    lines=['# Hard-free-line index retrospective','',result['scope'],'',
        'For the primitive index, the actual draw label is uniform or (Gaussian, axis). Its weight is `J W Z_eff / Q_old`, with `Z_eff=1` for the uniform/fallback branches. The mixed index groups a copy of the uniform channel with each Gaussian/axis. On physical support, `1/Q_new <= S_mixed <= S_primitive <= 1/Q_old`.','',
        '| Critical source | Rows | Full / old M2 | Primitive / old M2 | Primitive / full M2 | Mixed / old M2 | Mixed / full M2 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in result['critical'].items():
        r=v['ratios'];lines.append(f"| {name} | {v['selected_rows']} | {r['full_mixture']['over_old']:.6g} | {r['primitive_index']['over_old']:.6g} | {r['primitive_index']['over_full']:.6g} | {r['mixed_index']['over_old']:.6g} | {r['mixed_index']['over_full']:.6g} |")
    lines+=['','These sums retain each source population denominator and the four-population average. The paired product of the two old cloud weights estimates the squared physical integrand; it is not the second moment of a new noisy weight. The same index ordering holds for that noisy second moment when cloud noise is conditionally independent of the index.','',
        '| Breadth class | Primitive/full min / median / max | Mixed/full min / median / max |','|---|---:|---:|']
    for name,v in result['breadth'].items():
        if not v['valid']:lines.append(f'| {name} | physical weight zero | physical weight zero |');continue
        text=[]
        for mode in ('primitive_index','mixed_index'):
            d=v['ratios'][mode]['over_full'];text.append(' / '.join(f'{d[k]:.5g}' for k in ('minimum','median','maximum')))
        lines.append(f"| {name} | {text[0]} | {text[1]} |")
    if result['fresh_proposal_cost_accounting']:
        c=result['fresh_proposal_cost_accounting'];t=c['measured_CPU_seconds']
        lines+=['',f"Saved fresh xyz draws spent {t['full_density']:.4f} s scoring complete density out of {t['full_proposal']:.4f} s total proposal CPU. Eliminating all that stage with zero replacement cost has an Amdahl ceiling of {c['zero_replacement_cost_ceiling']['eliminate_all_density']:.3f}x on this proposal-only workload. The primitive estimator still computes cheap Q_old; physical clouds and independent validation are additional costs."]
    (a.out/'report.md').write_text('\n'.join(lines)+'\n');print(a.out/'analysis.json')


if __name__=='__main__':main()
