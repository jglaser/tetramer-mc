#!/usr/bin/env python3
"""Audit atoms and q locations of the new outer-reference extremes; no refit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from analyze_native_region_reference import native_q
from prepare_cayley_rms_cover import read,write,sha,require
from prepare_native_confirmation_atlas import AtomUnionAudit
from shoulder_mis import LogMoments,quota_summary
from run_shoulder_mis_campaign import local_dependencies

EDGES=(2.,2.25,2.5,3.,4.,5.)


def diagnose(total_path,out):
    require(not out.exists(),'Use a fresh diagnostic output directory')
    total=read(total_path);plan=read(Path(total_path.parent/'provenance.json'))
    protocol=read(Path(plan['plan']))
    require(sha(Path(plan['plan']))==total['plan_sha256']==plan['plan_sha256'],'Changed allocation')
    require(total['complete'],'Missing complete partition audit')
    archive=out/'provenance';archive.mkdir(parents=True)
    for name,path in local_dependencies([Path(__file__)]).items():(archive/name).write_bytes(path.read_bytes())
    (archive/'total.json').write_bytes(total_path.read_bytes())
    reports=[];all_selected=[]
    for piece in total['pieces']:
        name=piece['name']; source=piece['source_audit'];root=Path(source['root'])
        cfg=read(root/'provenance/config.json');shape=read(root/'provenance/shape.json')
        atom=AtomUnionAudit(shape,cfg['fixed_poses']);master=read(root/'manifest.json')
        bins=[LogMoments() for _ in EDGES[:-1]];heap=[];n=0;valid=0;ordinal=0;hashes={}
        affine=protocol['complete_cover_bound'];matrix=np.array(affine['affine_matrix']);shift=np.array(affine['affine_shift'])
        lo=piece['region']['minimum'];hi=piece['region']['maximum']
        for job in master['jobs']:
            path=Path(job['directory'])/'samples.jsonl';digest=hashlib.sha256();count=0
            with path.open('rb') as handle:
                for line in handle:
                    digest.update(line);row=json.loads(line)
                    require(row['draw']==count,'Missing draw');count+=1;n+=1
                    if row['log_importance_weight'] is None:continue
                    radius=float(np.linalg.norm(matrix@np.array(row['latent'])+shift)) if name=='remainder' else row['latent_radius']
                    selected=(radius>=lo if lo==0 else radius>lo) and (hi is None or radius<=hi)
                    if not selected:continue
                    valid+=1;q=row['q'];require(2<=q<5,'Changed original window')
                    index=next(i for i in range(len(EDGES)-1) if EDGES[i]<=q<EDGES[i+1])
                    hard=row['log_hard_weight'];pair=[hard+c['log_weight'] for c in row['clouds']]
                    bins[index].add(row['log_importance_weight'],pair)
                    point=dict(population=job['id'],seed=job['seed'],**row,weighted_radius=radius)
                    entry=(row['log_importance_weight'],ordinal,point);ordinal+=1
                    if len(heap)<8:heapq.heappush(heap,entry)
                    elif entry[:2]>heap[0][:2]:heapq.heapreplace(heap,entry)
            require(count==job['samples'] and digest.hexdigest()==source['sample_sha256'][str(path)],'Changed audited rows')
            hashes[str(path)]=digest.hexdigest()
        require(n==piece['physical']['draws'] and valid==piece['physical']['nonzero'],'Selected count disagrees with audited partition')
        band_results=[]
        for i,moment in enumerate(bins):
            # Bulk-retain all zero contributions; count is never the hit count.
            moment.count+=n-moment.count
            result=quota_summary([moment]);result['row_RSE']=result.pop('stratified_RSE')
            result['variance_rule']='IID full-N row variance, including all off-band and invalid zeros'
            band_results.append(dict(q_minimum=EDGES[i],q_maximum=EDGES[i+1],**result))
        observed=[b['logQ'] for b in band_results if b['logQ'] is not None]
        require(abs(float(logsumexp(observed))-piece['physical']['logQ'])<2e-8,'q bands do not reconstruct selected mass')
        points=[]
        for _,_,row in sorted(heap,reverse=True):
            q=native_q(cfg['metadata'],row['pose']);require(abs(q-row['q'])<2e-8,'q reconstruction failed')
            gaps=atom.gaps(row['pose']);require(min(gaps)>=0,'Selected pose fails independent atom-union hard check')
            logf=float(logsumexp([c['log_weight'] for c in row['clouds']])-math.log(2))
            require(abs(logf+row['log_hard_weight']-row['log_importance_weight'])<2e-8,'Original numerator changed')
            point=dict(source=name,population=row['population'],seed=row['seed'],draw=row['draw'],pose=row['pose'],
                       q=q,weighted_radius=row['weighted_radius'],minimum_atomic_gap_by_neighbor_A=gaps,
                       original_log_importance_weight=row['log_importance_weight'],original_clouds=row['clouds'],
                       log_boltzmann_mean=logf,log_mean_contribution=row['log_importance_weight']-math.log(n),
                       fraction_of_piece=math.exp(row['log_importance_weight']-math.log(n)-piece['physical']['logQ']))
            points.append(point);all_selected.append(point)
        require(abs(points[0]['fraction_of_piece']-piece['physical']['maximum_fraction'])<2e-8,'Largest fraction mismatch')
        reports.append(dict(name=name,draws=n,valid=valid,bands=band_results,top_poses=points,sample_sha256=hashes))
        print(json.dumps(dict(name=name,top_q=points[0]['q'],top_weighted_radius=points[0]['weighted_radius'],
                              minimum_checked_gap_A=min(min(p['minimum_atomic_gap_by_neighbor_A']) for p in points),
                              q_band_logQ=[b['logQ'] for b in band_results])),flush=True)
    write(out/'analysis.json',dict(complete=True,total_sha256=sha(total_path),plan_sha256=total['plan_sha256'],
        pieces=reports,selected_poses=all_selected,q_edges=EDGES,
        archived_sha256={p.name:sha(p) for p in archive.iterdir()},
        scope='Retrospective point and q-band diagnosis of unchanged original estimators. Independent atomic checks at selected extremes; no new clouds, proposal fits, physical draws or tail bounds.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--total',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();diagnose(args.total.resolve(),args.out.resolve())
