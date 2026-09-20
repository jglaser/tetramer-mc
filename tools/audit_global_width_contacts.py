#!/usr/bin/env python3
"""Freeze newly observed shoulder and leading contact contributions for review."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from analyze_basin_normalizers import moments,paired_noise
from analyze_global_fixed_region import region_latent
from check_contact_neighborhood_survival import first_clash,placed
from prepare_deep_far_normalizer_atlas import registration
from prepare_smc_normalizer_atlas import Density,read,relative_poses,sha,write


def audit(root,out):
    if out.exists():raise ValueError('Use a fresh immutable audit directory')
    manifest=read(root/'manifest.json');cfg=read(root/'provenance/config.json');model=read(root/'provenance/model.json')
    assert manifest['proposal_anchor_index']==0 and cfg['uniform_probability']==.1
    for name,digest in manifest['archive_sha256'].items():assert sha(root/'provenance'/name)==digest
    shape=read(root/'provenance/shape.json');centers=np.asarray([a['center'] for a in shape['atoms']]);radii=np.asarray([a['radius'] for a in shape['atoms']])
    fixed=[placed(centers,p) for p in cfg['fixed_poses']];trees=[cKDTree(p) for p in fixed]
    sources={str(root/'manifest.json'):sha(root/'manifest.json')};groups={};cpu=time.process_time()
    for scale in sorted({j['covariance_std_scale'] for j in manifest['jobs']}):
        records=[];jobs=[j for j in manifest['jobs'] if j['covariance_std_scale']==scale]
        for job in jobs:
            directory=Path(job['directory']);summary=read(directory/'summary.json');assert summary['complete']
            path=directory/'samples.jsonl';rows=[json.loads(line) for line in path.open()]
            assert len(rows)==job['samples'] and [r['draw'] for r in rows]==list(range(job['samples']))
            records.extend(dict(r,population=job['id'],seed=job['seed']) for r in rows)
            sources[str(path)]=sha(path);sources[str(directory/'summary.json')]=sha(directory/'summary.json')
        other=[i for i,r in enumerate(records) if r['hard_valid'] and r['q']>1]
        shoulder=[i for i in other if records[i]['q']<2]
        native=[i for i,r in enumerate(records) if r['hard_valid'] and r['q']<=1]
        topother=sorted(other,key=lambda i:records[i]['log_importance_weight'],reverse=True)[:16]
        topnative=sorted(native,key=lambda i:records[i]['log_importance_weight'],reverse=True)[:16]
        selected=sorted(set(shoulder+topother+topnative))
        poses=[records[i]['pose'] for i in selected];q=registration(poses,cfg['metadata'])
        relative=relative_poses(poses,cfg['fixed_poses'][0]);full_densities={}
        for width in (1.,2.,4.):
            alternative=copy.deepcopy(model);alternative['covariances']=(np.asarray(model['covariances'])*width**2).tolist()
            g=Density(alternative).evaluate(relative)[0]
            # Every selected pose is physically captured and therefore within
            # the cube defense, including its correct absolute normalization.
            full_densities[width]=np.logaddexp(np.log(.1)-3*np.log(36),np.log(.9)+g)
        native_chart={k:copy.deepcopy(model[k]) for k in ('schema','shape_sha256','coordinate_convention','angular_length')}
        for name in ('anchors','means','covariances'):native_chart[name]=[copy.deepcopy(model[name][0])]
        native_chart['weights']=[1.]
        native_region={'gaussian_chart':native_chart,'fixed_neighbor':cfg['fixed_poses'][0]}
        latent,md,diagnostic=region_latent(poses,native_region)
        total_other=logsumexp([records[i]['log_importance_weight'] for i in other]) if other else -np.inf
        total_shoulder=logsumexp([records[i]['log_importance_weight'] for i in shoulder]) if shoulder else -np.inf
        checked=[]
        for k,index in enumerate(selected):
            row=records[index];p=row['pose'];points=placed(centers,p)
            assert np.linalg.norm(np.asarray(p['position'])-cfg['capture_center'])<=cfg['capture_radius']
            assert all(first_clash(points,radii,trees[j],fixed[j]) is None for j in range(len(fixed)))
            contacts=[first_clash(points,radii+cfg['depletant_radius'],trees[j],fixed[j]) is not None for j in range(len(fixed))]
            assert row['depletion_contact']==any(contacts) and abs(row['q']-q[k])<2e-8
            logs=[c['log_weight'] for c in row['clouds']];assert len(logs)==2
            mean=float(logsumexp(logs)-np.log(2));recorded=row['log_proposal_density']
            assert abs(mean-recorded-row['log_importance_weight'])<1e-10
            assert abs(full_densities[scale][k]-recorded)<2e-8
            is_shoulder=index in shoulder
            checked.append(dict(row,width=scale,all_atom_hard_valid=True,independent_contact_neighbors=contacts,
                selected_as_all_shoulder=is_shoulder,selected_as_top_other=index in topother,selected_as_top_native=index in topnative,
                original_q_independent=float(q[k]),unscaled_native_chart_latent=latent[k].tolist(),unscaled_native_chart_radius=float(md[k]),
                log_boltzmann_mean=mean,cloud_log_weight_difference=logs[0]-logs[1],
                full_log_proposal_by_width={str(w):float(full_densities[w][k]) for w in full_densities},
                actual_minus_nominal_log_proposal=float(recorded-full_densities[1.][k]),
                other_mass_fraction=float(np.exp(row['log_importance_weight']-total_other)) if row['q']>1 else None,
                shoulder_mass_fraction=float(np.exp(row['log_importance_weight']-total_shoulder)) if is_shoulder else None))
        statistics={}
        for label,indices in [('shoulder',shoulder),('other',other),('native',native)]:
            included=set(indices)
            weights=[r['log_importance_weight'] if i in included else -np.inf for i,r in enumerate(records)]
            pairs=[[c['log_weight']-r['log_proposal_density'] for c in r['clouds']] if i in included else [-np.inf,-np.inf] for i,r in enumerate(records)]
            statistics[label]=moments(weights);statistics[label]['paired_noise']=paired_noise(np.asarray(weights),np.asarray(pairs))
        groups[str(scale)]=dict(unconditional_draws=len(records),all_shoulder_count=len(shoulder),unique_atomic_audits=len(selected),
            shoulder_generator_components=dict(Counter(records[i]['proposal']['component_index'] for i in shoulder)),
            all_atom_and_density_labels_agree=True,statistics=statistics,coordinate_diagnostic=diagnostic,rows=checked)
    out.mkdir(parents=True);archive=out/'provenance';archive.mkdir()
    for name in ('audit_global_width_contacts.py','check_contact_neighborhood_survival.py','analyze_global_fixed_region.py','prepare_smc_normalizer_atlas.py','prepare_deep_far_normalizer_atlas.py','analyze_basin_normalizers.py'):
        path=Path(__file__).with_name(name);shutil.copy2(path,archive/name);sources[str(path)]=sha(path)
    report=dict(complete=True,groups=groups,input_sha256=sources,cpu_seconds=time.process_time()-cpu,
        scope='All newly observed shoulder poses,16 top other and16 top native per width, unchanged original contributions. Finite-density diagnostics compare frozen proposal laws at the same existing poses; these are not counterfactual normalizer weights or equilibrium occupations. No refit, new geometry-generated pose, or physical simulation.')
    write(out/'results.json',report)
    for width,g in groups.items():write(out/f'width-{width}-shoulder-poses.json',[r for r in g['rows'] if r['selected_as_all_shoulder']])
    print(json.dumps({width:{k:v for k,v in group.items() if k!='rows'} for width,group in groups.items()},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();audit(args.root.resolve(),args.out.resolve())
