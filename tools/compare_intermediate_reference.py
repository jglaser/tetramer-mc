#!/usr/bin/env python3
"""Compare complete 2<=q<5 reference mass with original global-width estimators.

Every campaign retains its original proposal density and unconditional sample
count. There is no retrospective mixture correction or pooling across widths.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from analyze_basin_normalizers import audit_selected_anchor
from analyze_native_region_reference import analyze as audit_native_population,native_q
from compare_cayley_qwindow import load_latent,PHYSICAL_KEYS
from prepare_cayley_rms_cover import derive_model,read,require,sha,write
from run_shoulder_mis_campaign import local_dependencies
from shoulder_mis import LogMoments,quota_summary

ROOT=Path(__file__).resolve().parents[1]
BANDS=((2.,2.5),(2.5,3.),(3.,4.),(4.,5.))
KEYS=('all','0','1','2','3')
DEFAULT_GLOBALS=(ROOT/'runs/ab-global-contact-4x8192-l64-20260920',
    ROOT/'runs/ab-global-widths-2-4-4x16384-l64-20260920')


def band_index(q):
    return next((str(i) for i,(a,b) in enumerate(BANDS) if a<=q<b),None)


def population_rse(values):
    finite=[x for x in values if x is not None]
    if len(values)<2 or not finite:return None
    offset=max(finite);means=np.asarray([0. if x is None else math.exp(x-offset) for x in values])
    return float(means.std(ddof=1)/math.sqrt(len(means))/means.mean())


class BandMoments:
    def __init__(self):
        self.physical={key:LogMoments() for key in KEYS}
        self.hard={key:LogMoments() for key in KEYS}
        self.lower_boundary=LogMoments()

    def add(self,q=None,valid=False,physical=None,hard=None,cloud_pair=None):
        band=band_index(q) if valid else None
        if valid:require(math.isfinite(q) and math.isfinite(physical) and math.isfinite(hard),'Invalid contributing row')
        for key in KEYS:
            selected=band is not None and (key=='all' or key==band)
            self.physical[key].add(physical if selected else -math.inf,cloud_pair if selected else None)
            self.hard[key].add(hard if selected else -math.inf)
        self.lower_boundary.add(physical if valid and q==2. else -math.inf,cloud_pair if valid and q==2. else None)

    def merge(self,other):
        for key in KEYS:self.physical[key].merge(other.physical[key]);self.hard[key].merge(other.hard[key])
        self.lower_boundary.merge(other.lower_boundary)

    def report(self):
        def stats(moment):
            result=quota_summary([moment]);result['row_RSE']=result.pop('stratified_RSE')
            result['variance_rule']='IID row variance over the original full unconditional N, including all zeros'
            return result
        return dict(physical={k:stats(v) for k,v in self.physical.items()},
            hard={k:stats(v) for k,v in self.hard.items()},q_equals_2=stats(self.lower_boundary))


def finish_group(root,aggregate,populations,cpu,physical_signature,shape_sha,extra=None):
    require(populations and len({p['samples'] for p in populations})==1,'Population errors require equal fixed budgets')
    result=dict(root=str(root),**aggregate.report(),populations=populations,CPU_seconds=cpu,
        physical_signature=physical_signature,shape_sha256=shape_sha)
    for key in KEYS:
        for kind in ('physical','hard'):
            result[kind][key]['independent_population_RSE']=population_rse([p[kind][key]['logQ'] for p in populations])
    if extra:result.update(extra)
    return result


def hash_audit(root):
    paths=[root/'assessment/atomic-geometry-audit/results.json',root/'assessment/contact-geometry-audit/results.json']
    existing=[p for p in paths if p.is_file()]
    require(bool(existing),'Global source requires its earlier independent sample-hash audit')
    saved=read(existing[0]);require(saved['complete'],'Incomplete historical hash audit')
    return existing[0],saved['input_sha256']


def load_global(campaign,batch_size=4096):
    root=Path(campaign).resolve();master=read(root/'manifest.json');cfg=read(root/'provenance/config.json')
    require(batch_size>=1,'Positive density-audit batch size required')
    for name,digest in master['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,f'Changed global input: {name}')
    shape_sha=sha(root/'provenance/shape.json');physical={k:cfg[k] for k in PHYSICAL_KEYS}
    prior_path,prior_hashes=hash_audit(root)
    require(prior_hashes[str(root/'manifest.json')]==sha(root/'manifest.json'),'Historical global manifest changed')
    groups={}
    for width in sorted({j['covariance_std_scale'] for j in master['jobs']}):
        aggregate=BandMoments();populations=[];cpu=0.;sample_hashes={};density_error=0.;q_error=0.;checked=0
        jobs=[j for j in master['jobs'] if j['covariance_std_scale']==width]
        for job in jobs:
            directory=Path(job['directory']);manifest=read(directory/'manifest.json');summary=read(directory/'summary.json')
            require(summary['complete'] and summary['manifest']==manifest,'Incomplete or inconsistent global population')
            require(summary['numerical_nulls']==0 and manifest['schema']==2,'Require the audited selected-anchor no-null proposal')
            require(manifest['seed']==job['seed'] and manifest['covariance_scale']==width,'Original seed or width changed')
            require(manifest['samples']==job['samples']==summary['samples'],'Original global draw count changed')
            require(manifest['shape_sha256']==shape_sha and manifest['config_sha256']==sha(root/'provenance/config.json'),'Global physical bytes changed')
            require(manifest['model_sha256']==sha(root/'provenance/model.json'),'Original atlas changed')
            require(manifest['executable_sha256']==master['archive_sha256']['basin-normalizer'],'Original global executable changed')
            require(manifest['activity']==cfg['reservoir_density'] and manifest['cloud_replicates']==job['cloud_replicates']==2,'Global bath/cloud law changed')
            for filename,field in [('input-config.json','config_sha256'),('model.json','model_sha256'),
                ('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
                require(sha(directory/'provenance'/filename)==manifest[field],'Global population provenance changed')
            require(sha(directory/'summary.json')==prior_hashes[str(directory/'summary.json')],'Historical global summary changed')
            local=BandMoments();total=LogMoments();digest=hashlib.sha256();count=0
            def consume(rows):
                nonlocal density_error,q_error,checked
                audit=audit_selected_anchor(directory,job,manifest,rows)
                density_error=max(density_error,audit['maximum_log_density_error']);q_error=max(q_error,audit['maximum_q_error'])
                checked+=audit['checked_actual_poses']
                for row in rows:
                    if not row['hard_valid']:
                        require(row['log_importance_weight'] is None and not row['clouds'],'Invalid row has physical weight')
                        local.add();total.add();continue
                    require(row['capture_valid'] and len(row['clouds'])==2,'Missing capture or independent cloud pair')
                    q=native_q(cfg['metadata'],row['pose']);require(abs(q-row['q'])<2e-8,'Original scalar q mismatch')
                    label=('native_core' if row['q']<=.8 else 'native_shell' if row['q']<=1 else
                        'shoulder' if row['q']<2 else 'intermediate' if row['q']<5 else 'distant')
                    require(row['region']==label+('_bound' if row['depletion_contact'] else '_unbound'),'Historical region label changed')
                    clouds=[]
                    for cloud in row['clouds']:
                        expected=manifest['activity']*cloud['lower_volume']+cloud['overlap_points']*math.log1p(manifest['activity']/manifest['lambda'])
                        require(abs(expected-cloud['log_weight'])<2e-10,'Original Poisson factor changed')
                        clouds.append(cloud['log_weight'])
                    logh=-row['log_proposal_density'];logw=float(logsumexp(clouds))-math.log(2)+logh
                    require(abs(logw-row['log_importance_weight'])<2e-10,'Original global importance density changed')
                    # Use the actual recorded original metric after independently
                    # checking it; no tolerance-expanded region or density swap.
                    local.add(row['q'],True,row['log_importance_weight'],logh,[c+logh for c in clouds])
                    total.add(row['log_importance_weight'])
            with (directory/'samples.jsonl').open('rb') as handle:
                batch=[]
                for line in handle:
                    digest.update(line);row=json.loads(line);require(row['draw']==count,'Missing global trial');count+=1;batch.append(row)
                    if len(batch)==batch_size:consume(batch);batch=[]
                if batch:consume(batch)
            require(count==job['samples'],'Unconditional sample denominator changed')
            require(digest.hexdigest()==prior_hashes[str(directory/'samples.jsonl')],'Historical global samples changed')
            original=summary['estimates']['total']['log_normalizer'];computed=quota_summary([total])['logQ']
            require(original==computed if computed is None else abs(original-computed)<2e-8,'Whole original normalizer no longer agrees')
            population=dict(id=job['id'],seed=job['seed'],samples=count,**local.report(),samples_sha256=digest.hexdigest())
            populations.append(population);aggregate.merge(local);cpu+=summary['sampler_cpu_seconds']
            sample_hashes[str(directory/'samples.jsonl')]=digest.hexdigest()
        groups[str(width)]=finish_group(root,aggregate,populations,cpu,physical,shape_sha,dict(width=width,
            sample_sha256=sample_hashes,independently_audited_poses=checked,maximum_original_density_error=density_error,
            maximum_original_q_error=q_error,hash_audit_path=str(prior_path),hash_audit_sha256=sha(prior_path),
            model_sha256=sha(root/'provenance/model.json'),manifest_sha256=sha(root/'manifest.json'),
            proposal_rule='Every weight divides by its original width-specific full Gaussian+cube/Haar density. No retrospective MIS.'))
    return groups


def load_reference(campaign):
    root=Path(campaign).resolve();audited=load_latent(root,2.,5.)
    master=read(root/'manifest.json');region=read(root/'provenance/region.json');cfg=read(root/'provenance/config.json')
    require(region['minimum_original_q']<=2. and region['maximum_original_q']>=5.,'Reference does not cover intermediate window')
    require(region.get('minimum_mahalanobis_radius',0.)==0.,'Complete reference cannot have an inner hole')
    expected,proof=derive_model(cfg['metadata'],region['fixed_neighbor'],audited['shape_sha256'],
        region['maximum_original_q'],region['gaussian_chart']['angular_length'])
    for key in ('means','covariances','anchors','weights','angular_length','shape_sha256','coordinate_convention'):
        require(expected[key]==region['gaussian_chart'][key],f'Reference geometric chart mismatch: {key}')
    require(region['mahalanobis_radius']==proof['mahalanobis_radius'],'Reference geometric radius mismatch')
    assessment=read(root/'assessment/analysis.json');logv=assessment['log_latent_volume'];aggregate=BandMoments();populations=[]
    for job in master['jobs']:
        path=Path(job['directory']);summary=read(path/'summary.json');local=BandMoments();digest=hashlib.sha256();n=0
        with (path/'samples.jsonl').open('rb') as handle:
            for line in handle:
                digest.update(line);row=json.loads(line);require(row['draw']==n,'Missing reference trial');n+=1
                valid=row['capture_valid'] and row['hard_valid'] and row['region_valid']
                if not valid:local.add();continue
                q=native_q(cfg['metadata'],row['pose']);require(abs(q-row['q'])<2e-8,'Original reference q mismatch')
                logh=logv+row['log_physical_jacobian'];require(abs(logh-row['log_hard_weight'])<2e-10,'Reference hard weight changed')
                clouds=[]
                for cloud in row['clouds']:
                    expected_weight=summary['manifest']['activity']*cloud['lower_volume']+cloud['overlap_points']*math.log1p(summary['manifest']['activity']/summary['manifest']['lambda'])
                    require(abs(expected_weight-cloud['log_weight'])<2e-10,'Reference Poisson factor changed');clouds.append(cloud['log_weight'])
                require(len(clouds)==2,'Reference requires two independent clouds')
                logw=float(logsumexp(clouds))-math.log(2)+logh
                require(abs(logw-row['log_importance_weight'])<2e-10,'Reference original weight changed')
                local.add(row['q'],True,row['log_importance_weight'],logh,[c+logh for c in clouds])
        require(n==job['samples'] and digest.hexdigest()==audited['sample_sha256'][str(path/'samples.jsonl')],'Reference sample bytes or denominator changed')
        populations.append(dict(id=job['id'],seed=job['seed'],samples=n,**local.report(),samples_sha256=digest.hexdigest()));aggregate.merge(local)
    require(aggregate.lower_boundary.nonzero==0,'Strict load_latent audit would omit observed q==2 weight; audit this boundary separately before comparison')
    result=finish_group(root,aggregate,populations,audited['CPU_seconds'],audited['physical_signature'],audited['shape_sha256'],
        dict(existing_reference_audit=audited,complete_cover_reconstruction=proof,
            endpoint_audit='The reused strict load_latent(2,5) audit and inclusive comparison select exactly the same observed contributions: no valid q==2 rows.'))
    for kind in ('physical','hard'):
        a=result[kind]['all']['logQ'];b=audited[kind]['logQ']
        require(a==b if a is None else b is not None and abs(a-b)<2e-8,'Independent reference reaggregation differs')
    return result


def load_guided(campaign):
    """Optional fresh native-format campaign, with its original hybrid density."""
    root=Path(campaign).resolve();master=read(root/'manifest.json');cfg=read(root/'provenance/config.json')
    for name,digest in master['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,f'Changed guide input: {name}')
    window=master['q_window'];require(window['minimum']<=2. and window['maximum']>=5.,'Guided campaign does not cover the comparison band')
    shape_sha=sha(root/'provenance/shape.json');aggregate=BandMoments();populations=[];cpu=0.;hashes={}
    for job in master['jobs']:
        path=Path(job['output']);audited=audit_native_population(path);manifest=read(path/'manifest.json')
        require(manifest['schema']==4 and manifest['q_window']==window,'Guide original window changed')
        require(manifest['seed']==job['seed'] and manifest['config_sha256']==master['config_sha256']==sha(root/'provenance/config.json'),'Guide seed/config changed')
        require(manifest['shape_sha256']==shape_sha==master['shape_sha256'],'Guide physical shape changed')
        require(manifest['activity']==cfg['reservoir_density'] and manifest['executable_sha256']==master['archive_sha256']['native-region-normalizer'],'Guide bath/executable changed')
        local=BandMoments();digest=hashlib.sha256();n=0
        with (path/'samples.jsonl').open('rb') as handle:
            for line in handle:
                digest.update(line);row=json.loads(line);require(row['draw']==n,'Missing guided trial');n+=1
                if 'zero' in row:local.add();continue
                logh=-row['log_proposal_density'];clouds=[x+logh for x in row['cloud_log_weights']]
                require(abs(float(logsumexp(clouds))-math.log(2)-row['log_importance_weight'])<2e-10,'Guide original density/weight changed')
                local.add(row['q'],True,row['log_importance_weight'],logh,clouds)
        require(n==audited['samples'] and digest.hexdigest()==audited['sample_sha256'],'Guided audit bytes or denominator changed')
        populations.append(dict(id=path.name,seed=job['seed'],samples=n,**local.report(),samples_sha256=digest.hexdigest()))
        aggregate.merge(local);cpu+=audited['cpu_seconds'];hashes[str(path/'samples.jsonl')]=digest.hexdigest()
    require(aggregate.physical['all'].count==master['total_unconditional_draws'],'Guided campaign denominator changed')
    return finish_group(root,aggregate,populations,cpu,{k:cfg[k] for k in PHYSICAL_KEYS},shape_sha,
        dict(sample_sha256=hashes,original_integration_window=window,manifest_sha256=sha(root/'manifest.json'),
            proposal_rule='Original frozen product-cover/Gaussian/cube hybrid density on every draw; separately estimated, no MIS reweighting.'))


def compare(reference,globals_,out,guided_roots=()):
    out=Path(out).resolve();require(not out.exists(),'Use a fresh comparison output directory')
    ref=load_reference(reference);sources={}
    for path in globals_:
        for width,result in load_global(path).items():require(width not in sources,'Duplicate global width');sources[width]=result
    require(set(sources)=={'1.0','2.0','4.0'},'Require the three width controls 1,2,4')
    guided={}
    for path in guided_roots:
        name=Path(path).name;require(name not in guided,'Guided campaign labels must be distinct');guided[name]=load_guided(path)
    all_seeds=set()
    for item in [ref,*sources.values(),*guided.values()]:
        require(item['physical_signature']==ref['physical_signature'] and item['shape_sha256']==ref['shape_sha256'],'Comparison physical target changed')
        for p in item['populations']:require(p['seed'] not in all_seeds,'Independent comparisons require distinct seeds');all_seeds.add(p['seed'])
    out.mkdir(parents=True);archive=out/'provenance';archive.mkdir()
    dependencies=local_dependencies([Path(__file__)])
    for name,path in dependencies.items():(archive/name).write_bytes(path.read_bytes())
    result=dict(complete=True,q_window=dict(minimum=2.,maximum=5.,lower_inclusive=True,upper_inclusive=False),
        bands=BANDS,reference=ref,global_widths=sources,guided_campaigns=guided,physical=ref['physical_signature'],shape_sha256=ref['shape_sha256'],
        source_sha256={name:sha(archive/name) for name in dependencies},
        scope='Separate independent original-proposal estimates of the same fixed physical2<=q<5 region. Every hard/capture/q-invalid trial remains in full N; widths are never pooled or reweighted. All-zero subbands remain unresolved. Importance ESS and observed errors do not establish unobserved-mass coverage or MCMC mixing.')
    write(out/'comparison.json',result)
    lines=['# Intermediate contact-region reference comparison','','| Source | Band | N | Nonzero | log Q | Row / population RSE | Largest weight | Paired-cloud variance |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def number(value):return 'unresolved' if value is None else f'{value:.6g}'
    for label,item in [('Complete reference',ref)]+[(f'Global width {w}',v) for w,v in sources.items()]+list(guided.items()):
        for key in KEYS:
            r=item['physical'][key];band='2<=q<5' if key=='all' else f'{BANDS[int(key)][0]}<=q<{BANDS[int(key)][1]}'
            lines.append(f"| {label} | {band} | {r['draws']} | {r['nonzero']} | {number(r['logQ'])} | {number(r['row_RSE'])} / {number(r['independent_population_RSE'])} | {number(r['maximum_fraction'])} | {number(r['paired_cloud_variance_fraction'])} |")
    lines+=['',result['scope'],'','Hard-volume estimates, all population results, physical checks, hashes and endpoint counts are in comparison.json.','']
    (out/'report.md').write_text('\n'.join(lines));return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--global-root',type=Path,action='append');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--guided-root',type=Path,action='append',default=[],help='Additional fresh native-format campaign covering 2<=q<5; keep its original density and separate estimate')
    args=ap.parse_args();result=compare(args.reference,args.global_root or DEFAULT_GLOBALS,args.out,args.guided_root)
    print(json.dumps(dict(complete=result['complete'],reference=result['reference']['physical']['all'],
        global_widths={k:v['physical']['all'] for k,v in result['global_widths'].items()},
        guided_campaigns={k:v['physical']['all'] for k,v in result['guided_campaigns'].items()}),indent=2))


if __name__=='__main__':main()
