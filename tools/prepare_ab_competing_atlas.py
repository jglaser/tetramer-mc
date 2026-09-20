#!/usr/bin/env python3
"""Freeze a two-contact competitor guide from an existing geometry-only screen.

Screen survivors describe proposal geometry, never equilibrium component mass.
No new poses, geometry queries, Poisson points, or physical trajectories occur.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from prepare_smc_normalizer_atlas import (
    ROOT, Density, arrays, convert_model, fit_population, model_from_components,
    read, relative_poses, sha, write,
)


def prepare(screen_path, native_path, parent_path, config_path, out):
    screen, native, parent, cfg = map(read, (screen_path, native_path, parent_path, config_path))
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    archive = out/'provenance';archive.mkdir()
    shape=Path(cfg['shape'])
    assert len(cfg['fixed_poses']) == 2 and cfg['capture_radius'] == 18
    assert cfg['depletant_radius'] == 1.5 and cfg['reservoir_density'] == .035
    assert native['shape_sha256'] == parent['shape_sha256'] == sha(shape)
    assert native['coordinate_convention'] == parent['coordinate_convention'] == 'anchor-body-relative'
    assert native['angular_length'] == parent['angular_length']
    rows=[r for r in screen['rows'] if r['cohort']=='fresh_far_endpoints'
          and r['id'].startswith('site0-far5-r1.5-z0.035-n512-r2/') and r['hard_valid']]
    assert len(rows) == 30 and all(r['q']>=5 and r['depletion_contact_neighbors']==[True,True] for r in rows)
    source=ROOT/'runs/explicit-far-smc-plan-20260920/production/site0-far5-r1.5-z0.035-n512-r2/summary.json'
    assert screen['source_sha256'][str(source)] == sha(source)
    endpoint=read(source);assert endpoint['complete']
    selected=[]
    for row in rows:
        index=int(row['id'].rsplit('/',1)[1]);original=endpoint['final_particles'][index]
        assert row['pose'] == original['pose']
        selected.append(dict(screen_id=row['id'],endpoint_index=index,family_id=original['family_id'],pose=original['pose'],q=row['q']))
    poses=[r['pose'] for r in selected]
    component,fit=fit_population(poses,2,parent['angular_length'],.05,1e-8,1.)
    assert fit['maximum_chart_angle_deg'] < 60, 'Use multiple charts for a broad or near-seam cohort'
    laboratory=model_from_components([component],parent['angular_length'],parent['shape_sha256'])
    competitor=convert_model(laboratory,cfg['fixed_poses'][0])
    laboratory_log=Density(laboratory).evaluate(poses)[0]
    relative=relative_poses(poses,cfg['fixed_poses'][0])
    relative_log=Density(competitor).evaluate(relative)[0]
    frame_error=float(np.max(np.abs(laboratory_log-relative_log)));assert frame_error < 2e-8
    models=[native,parent,competitor];fractions=[.5,.25,.25]
    mixture={k:copy.deepcopy(parent[k]) for k in ('schema','angular_length','coordinate_convention','shape_sha256')}
    for key in ('anchors','means','covariances'):
        mixture[key]=sum([copy.deepcopy(m[key]) for m in models],[])
    mixture['weights']=sum([[fraction*w for w in m['weights']] for fraction,m in zip(fractions,models)],[])
    assert abs(sum(mixture['weights'])-1)<1e-12
    # Check complete mixture sums in its exact deployment frame using only the
    # existing screened poses, without drawing any additional candidates.
    calculated=Density(mixture).evaluate(relative)[0]
    expected=np.logaddexp.reduce(np.asarray([np.log(f)+Density(m).evaluate(relative)[0] for f,m in zip(fractions,models)]),axis=0)
    density_error=float(np.max(np.abs(calculated-expected)));assert density_error<2e-10
    region=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        fixed_neighbor=cfg['fixed_poses'][0],physical_fixed_neighbors=cfg['fixed_poses'],
        capture_center=cfg['capture_center'],capture_radius=cfg['capture_radius'],
        shape_sha256=parent['shape_sha256'],activity=cfg['reservoir_density'],depletant_radius=cfg['depletant_radius'],
        physical_metric=cfg['metadata'],gaussian_chart=competitor,mahalanobis_radius=3.,
        minimum_original_q=5.,
        definition='Fixed radius-three ellipsoid of the frozen geometric competitor chart, intersect unchanged capture/hard and original q>=5. No equilibrium basin boundary claim.')
    write(out/'model-competitor.json',competitor)
    write(out/'model-mixture.json',mixture)
    write(out/'region-competitor-r3.json',region)
    write(out/'selected-endpoints.json',selected)
    cfgcopy=copy.deepcopy(cfg);cfgcopy['shape']=str(archive/'shape.json')
    write(out/'config.json',cfgcopy)
    # Metadata equality is the regional integrator's physical identity check.
    assert region['physical_metric'] == cfgcopy['metadata']
    unique=len({json.dumps(r['pose'],sort_keys=True) for r in selected})
    families=Counter(r['family_id'] for r in selected)
    inputs={'screen.json':screen_path,'native-model.json':native_path,'parent-model.json':parent_path,
            'input-config.json':config_path,'shape.json':shape,'source-summary.json':source,
            'prepare_ab_competing_atlas.py':Path(__file__).resolve(),
            'prepare_smc_normalizer_atlas.py':ROOT/'tools/prepare_smc_normalizer_atlas.py'}
    for name,path in inputs.items():shutil.copy2(path,archive/name)
    report=dict(selected_count=len(selected),unique_exact_poses=unique,duplicate_exact_poses=len(selected)-unique,
        family_counts=dict(families),fit=fit,frame_density_max_error=frame_error,mixture_density_max_error=density_error,
        mixture_allocation=dict(native=.5,unchanged_parent=.25,competitor=.25),
        component_ranges=dict(native=[0,len(native['weights'])],parent=[len(native['weights']),len(native['weights'])+len(parent['weights'])],
                              competitor=[len(mixture['weights'])-1,len(mixture['weights'])]),
        proposal_anchor_index=0,uniform_probability=cfg['uniform_probability'],
        minimum_original_q=float(min(r['q'] for r in selected)),maximum_original_q=float(max(r['q'] for r in selected)),
        input_sha256={name:sha(archive/name) for name in inputs},
        outputs={name:sha(out/name) for name in ('model-competitor.json','model-mixture.json','region-competitor-r3.json','config.json','selected-endpoints.json')},
        geometry_budget='No new generated poses or atomic checks; this fit uses only the prior fixed2048-pose screen.',
        inference='Equal endpoint weights and .5/.25/.25 mixture allocation define a proposal only. Endpoints were selected by AB hard geometry from a correlated one-neighbor SMC population. Neither component weights nor covariance estimate AB equilibrium probability. Fresh independent integration and proposal sensitivity are required.')
    write(out/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,default=ROOT/'runs/ab-competing-geometry-screen-20260920/results.json')
    parser.add_argument('--native-model',type=Path,default=ROOT/'runs/native-ab-refined-covariance-guide-20260920/model.json')
    parser.add_argument('--parent-model',type=Path,default=ROOT/'runs/outside-r8-atlas-extension-20260920/site0/model.json')
    parser.add_argument('--config',type=Path,default=ROOT/'runs/native-factorial-screen-20260920/AB/provenance/config.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.screen.resolve(),args.native_model.resolve(),args.parent_model.resolve(),args.config.resolve(),args.out.resolve())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
