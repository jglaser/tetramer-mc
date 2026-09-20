#!/usr/bin/env python3
"""Audit separate radial references, then sum their disjoint physical integrals."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

from audit_shoulder_mis_independently import chart_values, near
from compare_intermediate_local_reference import load_reference, PartitionMoments, finish, read_batches
from compare_intermediate_reference import population_rse
from prepare_cayley_rms_cover import derive_model, read, write, sha, require
from run_shoulder_mis_campaign import local_dependencies

NAMES = ('r4','r8','r16','r32','remainder')


def total_independent(parts):
    """Sum independent regional estimators, not samples or log normalizers."""
    observed=[p for p in parts if p['logQ'] is not None]
    partial=float(logsumexp([p['logQ'] for p in observed])) if observed else None
    if len(observed)!=len(parts):
        return dict(logQ=None,observed_partial_logQ=partial,all_pieces_observed=False,
                    scope='A supported piece has no nonzero observations; its mass is unresolved, not bounded by zero.')
    logq=partial
    logvar=float(logsumexp([p['log_variance_of_mean'] for p in parts if p['log_variance_of_mean'] is not None]))
    fractions=[math.exp(p['logQ']-logq) for p in parts]
    cloud_terms=[p['log_variance_of_mean']+math.log(p['paired_cloud_variance_fraction']) for p in parts
                 if p['log_variance_of_mean'] is not None and p['paired_cloud_variance_fraction'] is not None
                 and p['paired_cloud_variance_fraction']>0]
    cloud=float(math.exp(logsumexp(cloud_terms)-logvar)) if cloud_terms else None
    return dict(logQ=logq,observed_partial_logQ=logq,all_pieces_observed=True,
                log_variance_of_mean=logvar if math.isfinite(logvar) else None,row_RSE=math.exp(.5*logvar-logq),
                piece_estimated_mass_fractions=fractions,
                largest_single_draw_fraction=max(f*p['maximum_fraction'] for f,p in zip(fractions,parts)),
                paired_cloud_variance_fraction=cloud,
                variance_rule='Variance of a sum of independent regional estimates; every summand retains its own original N. No pooled proposal or independence assumption between bins from the same rows.',
                scope='Observed uncertainty only, not a bound on unseen tails or proof of equilibrium mixing.')


def audit_remainder(root,plan,source,weighted,raw_audit):
    """Direct positive outside-mask contributions under the original cover law."""
    radius=32.; aggregate=PartitionMoments(radius); populations=[]; errors=0.
    master=read(root/'manifest.json'); bound=plan['complete_cover_bound']['complete_weighted_radius_bound']
    expected_model,_=derive_model(source['physical_metric'],source['fixed_neighbor'],source['shape_sha256'],
                                  q_max=5.,ell=source['gaussian_chart']['angular_length'])
    require(source['gaussian_chart']==expected_model,'Complete geometric cover was not reconstructed exactly')
    require(source['mahalanobis_radius']==10.,'Unexpected geometric-cover radius')
    for job in master['jobs']:
        path=Path(job['directory']); local=PartitionMoments(radius); digest=hashlib.sha256(); count=0
        for lines,rows in read_batches(path/'samples.jsonl'):
            for line in lines: digest.update(line)
            t=np.array([r['pose']['position'] for r in rows]); q=np.array([r['pose']['orientation'] for r in rows])
            rotations=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
            _,radii,_=chart_values(weighted,t,rotations,source['fixed_neighbor'])
            # Independent affine reconstruction from the recorded source latent
            # coordinates, which the original reference audit already verified.
            affine=plan['complete_cover_bound']
            mapped=np.asarray([r['latent'] for r in rows])@np.array(affine['affine_matrix']).T+affine['affine_shift']
            errors=max(errors,near(radii[:,0],np.linalg.norm(mapped,axis=1)))
            require(np.max(radii)<=bound,'A source draw exceeds the complete enclosing bound')
            for row,r in zip(rows,radii[:,0]):
                require(row['draw']==count,'Missing source row');count+=1
                if row['log_importance_weight'] is None:local.add();continue
                hard=row['log_hard_weight']; pair=[hard+c['log_weight'] for c in row['clouds']]
                local.add(r,row['log_importance_weight'],hard,pair)
        expected=raw_audit['sample_sha256'][str(path/'samples.jsonl')]
        require(digest.hexdigest()==expected and count==job['samples'],'Source rows changed after full audit')
        populations.append(dict(id=job['id'],seed=job['seed'],samples=count,**local.report(),samples_sha256=expected))
        aggregate.merge(local)
    result=finish(aggregate,populations)
    for kind in ('physical','hard'):
        near(result[kind]['full']['logQ'],raw_audit[kind]['ball']['logQ'])
    result['affine_radius_max_error']=errors
    return result


def component(plan_path,name,out):
    require(not out.exists(),'Use a fresh component output directory')
    plan=read(plan_path); weighted_path=Path(plan['chart_model']); weighted=read(weighted_path)
    require(sha(weighted_path)==plan['chart_model_sha256'],'Weighted target chart changed')
    seal=read(plan_path.parent/'freeze.json')
    require(sha(plan_path)==seal['protocol_sha256'],'Changed allocation plan')
    if name=='r4':
        root=Path(plan['existing_inner_reference']); expected_hash=plan['existing_inner_manifest_sha256'];selected='ball'
        require(sha(root/'manifest.json')==expected_hash,'Changed original inner campaign')
        allocation=None
    elif name=='remainder':
        allocation=plan['remainder'];root=Path(allocation['campaign']);selected='complement'
    else:
        allocation=next(p for p in plan['local_references'] if name==f"r{p['radius']}")
        root=Path(allocation['campaign']);selected='outer_half'
    master=read(root/'manifest.json'); source=read(root/'provenance/region.json')
    if allocation:
        require(master['region_sha256']==allocation['region_sha256'],'Region differs from frozen allocation')
        require([j['seed'] for j in master['jobs']]==allocation['seeds'],'Seeds differ from allocation')
        require(all(j['samples']==allocation['samples_per_population'] for j in master['jobs']),'Budget differs from allocation')
    require(master['archive_sha256']['latent-region-normalizer']==plan['executable_sha256'],'Changed reference executable')
    require(master['lambda_ratio']==plan['lambda_ratio'] and master['cloud_replicates']==plan['cloud_replicates'],'Cloud law changed')
    if name!='remainder':require(source['gaussian_chart']==weighted,'Changed local chart')
    archive=out/'provenance';archive.mkdir(parents=True)
    dependencies=local_dependencies([Path(__file__)])
    for filename,path in dependencies.items():(archive/filename).write_bytes(path.read_bytes())
    (archive/'plan.json').write_bytes(plan_path.read_bytes())
    (archive/'weighted-chart.json').write_bytes(weighted_path.read_bytes())
    raw,_,_=load_reference(root,source['gaussian_chart'])
    assessed=audit_remainder(root,plan,source,weighted,raw) if name=='remainder' else raw
    result=dict(complete=True,name=name,plan_sha256=sha(plan_path),chart_sha256=sha(weighted_path),
        physical_signature=raw['physical_signature'],shape_sha256=raw['shape_sha256'],
        region=dict(minimum=32.,maximum=None,lower_inclusive=False) if name=='remainder' else
               dict(minimum=0. if name=='r4' else int(name[1:])/2,maximum=float(name[1:]),lower_inclusive=name=='r4',upper_inclusive=True),
        selected_key=selected,physical=assessed['physical'][selected],hard=assessed['hard'][selected],
        populations=[dict(id=p['id'],seed=p['seed'],samples=p['samples'],physical=p['physical'][selected],hard=p['hard'][selected]) for p in assessed['populations']],
        sampler_CPU_seconds=raw['CPU_seconds'],source_audit=raw,
        partition_audit=assessed if name=='remainder' else None,
        source_sha256={filename:sha(archive/filename) for filename in dependencies},
        scope='One disjoint piece, original zero-inclusive estimator and original source proposal; no subtraction of total estimates.')
    write(out/'analysis.json',result)
    print(json.dumps(dict(name=name,physical=result['physical'],hard=result['hard'],CPU_seconds=result['sampler_CPU_seconds'])),flush=True)


def combine(plan_path,components,out):
    require(not out.exists(),'Use a fresh combined output directory')
    plan=read(plan_path); pieces={p['name']:p for p in [read(Path(root)/'analysis.json') for root in components]}
    require(set(pieces)==set(NAMES) and len(components)==len(NAMES),'Need exactly one of each disjoint piece')
    ordered=[pieces[name] for name in NAMES];seeds=set()
    expected_regions=[(0.,4.),(4.,8.),(8.,16.),(16.,32.),(32.,None)]
    for p,(lo,hi) in zip(ordered,expected_regions):
        require(p['complete'] and p['plan_sha256']==sha(plan_path),'Incomplete or differently planned component')
        require(p['chart_sha256']==plan['chart_model_sha256'],'Different analysis chart')
        require(p['physical_signature']==ordered[0]['physical_signature'] and p['shape_sha256']==ordered[0]['shape_sha256'],'Different physical targets')
        require(p['region']['minimum']==lo and p['region']['maximum']==hi and p['region']['lower_inclusive']==(lo==0.),'Wrong radial partition')
        if hi is not None:require(p['region']['upper_inclusive'],'Wrong upper inclusion')
        require(len(p['populations'])==4,'Four populations required')
        for population in p['populations']:
            require(population['seed'] not in seeds,'Repeated reference random stream');seeds.add(population['seed'])
    result=dict(complete=True,plan_sha256=sha(plan_path),pieces=ordered,
                sampler_CPU_seconds=sum(p['sampler_CPU_seconds'] for p in ordered),
                coverage='Original [2,5), unchanged AB hard/capture masks, disjoint weighted-chart pieces [0,4],(4,8],(8,16],(16,32],(32,infinity). Remainder support follows the reconstructed complete geometric cover; no finite piece is silently omitted.')
    for kind in ('physical','hard'):
        result[kind]=total_independent([p[kind] for p in ordered])
        groups=[]
        for i in range(4):
            values=[p['populations'][i][kind]['logQ'] for p in ordered if p['populations'][i][kind]['logQ'] is not None]
            groups.append(float(logsumexp(values)) if values else None)
        result[kind]['population_group_logQ']=groups
        result[kind]['population_group_RSE']=population_rse(groups)
    out.mkdir(parents=True)
    write(out/'analysis.json',result)
    write(out/'provenance.json',dict(plan=str(plan_path),plan_sha256=sha(plan_path),
                                   components={str(Path(root)/'analysis.json'):sha(Path(root)/'analysis.json') for root in components},
                                   helper_sha256=sha(Path(__file__))))
    (out/'analyzer.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k!='pieces'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--component',choices=NAMES);parser.add_argument('--combine',type=Path,nargs='+')
    args=parser.parse_args();require(bool(args.component)!=bool(args.combine),'Select component audit or combine')
    if args.component:component(args.plan.resolve(),args.component,args.out.resolve())
    else:combine(args.plan.resolve(),args.combine,args.out.resolve())
