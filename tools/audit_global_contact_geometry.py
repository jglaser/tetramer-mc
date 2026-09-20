#!/usr/bin/env python3
"""Independent atomic checks of fixed random and leading global contributions."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree

from check_contact_neighborhood_survival import first_clash,placed
from prepare_deep_far_normalizer_atlas import registration
from prepare_smc_normalizer_atlas import read,sha,write


def audit(root,out,seed=99341010):
    if out.exists():raise ValueError('Use a fresh audit directory')
    manifest=read(root/'manifest.json');cfg=read(root/'provenance/config.json')
    shape_path=root/'provenance/shape.json';shape=read(shape_path)
    for name,digest in manifest['archive_sha256'].items():assert sha(root/'provenance'/name)==digest
    centers=np.asarray([a['center'] for a in shape['atoms']]);radii=np.asarray([a['radius'] for a in shape['atoms']])
    fixed=[placed(centers,p) for p in cfg['fixed_poses']];trees=[cKDTree(p) for p in fixed]
    for i in range(len(fixed)):
        for j in range(i):assert first_clash(fixed[i],radii,trees[j],fixed[j]) is None
    records=[];sources={str(root/'manifest.json'):sha(root/'manifest.json'),str(shape_path):sha(shape_path)}
    for job in manifest['jobs']:
        directory=Path(job['directory']);summary=read(directory/'summary.json')
        assert summary['complete'] and summary['samples']==job['samples']
        assert summary['manifest']['shape_sha256']==sha(shape_path)
        path=directory/'samples.jsonl';rows=[json.loads(line) for line in path.open()]
        assert len(rows)==job['samples'] and [r['draw'] for r in rows]==list(range(job['samples']))
        for row in rows:records.append(dict(row,job_id=job['id'],population_seed=job['seed']))
        sources[str(path)]=sha(path);sources[str(directory/'summary.json')]=sha(directory/'summary.json')
    rng=np.random.default_rng(seed)
    random_indices=sorted(int(i) for i in rng.choice(len(records),min(128,len(records)),replace=False))
    top_indices=sorted(range(len(records)),key=lambda i:records[i]['log_importance_weight'] if records[i]['log_importance_weight'] is not None else -np.inf,reverse=True)[:16]
    selected=sorted(set(random_indices+top_indices));actual=[records[i]['pose'] for i in selected]
    assert all(p is not None for p in actual)
    qs=registration(actual,cfg['metadata']);results=[];begun=time.process_time()
    for index,q in zip(selected,qs):
        row=records[index];p=row['pose'];points=placed(centers,p)
        clashes=[first_clash(points,radii,trees[j],fixed[j]) for j in range(len(fixed))]
        capture=bool(np.linalg.norm(np.asarray(p['position'])-cfg['capture_center'])<=cfg['capture_radius'])
        hard=capture and all(c is None for c in clashes)
        assert row['capture_valid']==capture and row['hard_valid']==hard
        contacts=[first_clash(points,radii+cfg['depletant_radius'],trees[j],fixed[j]) is not None for j in range(len(fixed))]
        if hard:
            assert abs(row['q']-q)<2e-8 and row['depletion_contact']==any(contacts)
        else:
            assert row['q'] is None and row['depletion_contact'] is None
        results.append(dict(global_index=index,job_id=row['job_id'],draw=row['draw'],pose=p,
            random_selected=index in random_indices,top_weight_selected=index in top_indices,
            original_q=float(q),capture_valid=capture,hard_valid=hard,clash_witnesses=clashes,
            depletion_contact_neighbors=contacts,log_importance_weight=row['log_importance_weight'],
            recorded_region=row['region'],proposal=row['proposal']))
    out.mkdir(parents=True);archive=out/'provenance';archive.mkdir()
    for path in [Path(__file__).resolve(),Path(__file__).with_name('check_contact_neighborhood_survival.py'),Path(__file__).with_name('prepare_deep_far_normalizer_atlas.py'),Path(__file__).with_name('prepare_smc_normalizer_atlas.py')]:
        shutil.copy2(path,archive/path.name);sources[str(path)]=sha(path)
    report=dict(complete=True,seed=seed,unconditional_source_draws=len(records),random_requested=128,top_requested=16,
        unique_checked=len(selected),random_and_top_overlap=len(random_indices)+len(top_indices)-len(selected),
        hard_valid_checked=sum(r['hard_valid'] for r in results),
        checked_valid_region_counts=dict(Counter(r['recorded_region'] for r in results if r['hard_valid'])),
        sampler_labels_agree=True,cpu_seconds=time.process_time()-begun,input_sha256=sources,rows=results,
        scope='Fixed random128 unconditional rows plus16 largest original full-proposal weights, duplicates removed. Exact atom-radius inequalities against every fixed neighbor; KD tree only enumerates possible pairs. No new physical sampling or equilibrium occupancy inference.')
    write(out/'results.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','input_sha256')},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=99341010);args=p.parse_args();audit(args.root.resolve(),args.out.resolve(),args.seed)
