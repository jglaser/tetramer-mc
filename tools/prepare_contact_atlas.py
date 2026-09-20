#!/usr/bin/env python3
"""Freeze two contact atlases before independent full-window physical draws."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import numpy as np

from analyze_native_region_reference import GaussianGuide
from prepare_cayley_rms_cover import read,write,sha,require
from prepare_intermediate_guides import geometry_probe,PHYSICAL
from prepare_intermediate_local_region import ROOT,GUIDES
from prepare_native_confirmation_atlas import AtomUnionAudit
from prepare_peak_neighborhood import geometry_model
from prepare_smc_normalizer_atlas import Density,arrays,relative_poses
from run_shoulder_mis_campaign import local_dependencies

CENTERS=ROOT/'runs/ab-intermediate-contact-centers-20260920/selection.json'
FIT=ROOT/'runs/ab-intermediate-local-fit-diagnostic-20260920'
BINARY=ROOT/'runs/ab-shoulder-mixture-4x16384-l64-20260920/provenance/native-region-normalizer'
BINARY_SHA='3a6a2dbba0ec5234c36cf66d7c40877336f22358a6a1ea91ea8366b344ee027b'
ARMS=(('narrow',.2,101401010),('broad',.4,101501010))


def assemble(cfg,centers,local,std):
    require(std>0 and math.isfinite(std),'Invalid geometric width')
    require(local['weights']==[1.] and len(local['means'])==len(local['covariances'])==1,'Require one local fitted component')
    model={key:copy.deepcopy(local[key]) for key in ('schema','angular_length','coordinate_convention','shape_sha256')}
    model.update(anchors=[],means=[],covariances=[],weights=[])
    for point in centers:
        geometric,_=geometry_model(cfg['metadata'],cfg['fixed_poses'][0],point['pose'],local['shape_sha256'],local['angular_length'])
        model['anchors']+=geometric['anchors'];model['means']+=geometric['means']
        model['covariances'].append((np.asarray(geometric['covariances'][0])*std**2).tolist())
        model['weights'].append(.8/len(centers))
    for scale,weight in ((1.,.16),(2.,.04)):
        model['anchors']+=copy.deepcopy(local['anchors']);model['means']+=copy.deepcopy(local['means'])
        model['covariances'].append((np.asarray(local['covariances'][0])*scale**2).tolist())
        model['weights'].append(weight)
    require(abs(sum(model['weights'])-1)<2e-12,'Unnormalized mixture')
    return model


def prepare(out):
    require(not out.exists(),'Use a fresh preparation directory')
    centers=read(CENTERS);fit_report=read(FIT/'report.json');local=read(FIT/'model-weighted.json')
    require(centers['complete'] and centers['no_physical_draws'],'Incomplete center selection')
    require(fit_report['complete'] and fit_report['no_new_geometry_or_physical_draws'],'Incomplete local fit')
    for name,digest in centers['archived_sha256'].items():require(sha(CENTERS.parent/'provenance'/name)==digest,'Changed center archive')
    for name,digest in read(FIT/'artifact-hashes.json').items():require(sha(FIT/name)==digest,'Changed exploratory fit artifact')
    for name,entry in read(FIT/'provenance-sha256.json').items():require(sha(FIT/'provenance'/name)==entry['sha256'],'Changed fit source')
    old_freeze=read(GUIDES/'freeze.json');require(sha(GUIDES/'config.json')==old_freeze['config_sha256'],'Changed original config')
    cfg=read(GUIDES/'config.json');shape=Path(cfg['shape'])
    require(sha(shape)==local['shape_sha256'],'Changed physical shape')
    require(sha(BINARY)==BINARY_SHA,'Changed archived physical kernel')
    for key in PHYSICAL:require(cfg[key]==fit_report['physical'][key],'Local fit has a different target')
    center_cfg=read(CENTERS.parent/'provenance/config.json')
    for key in PHYSICAL:require(cfg[key]==center_cfg[key],'Center selection has a different target')
    selected=centers['selected_centers'];require(len(selected)==36 and len(centers['candidates'])==64,'Unexpected frozen center set')
    # Independently reconstruct finite-set coverage from actual member positions.
    members,_,_=arrays(cfg['metadata']['rigid_members'])
    poses=[p['pose'] for p in centers['candidates']];t,_,r=arrays(poses)
    world=np.einsum('nij,mj->nmi',r,members)+t[:,None,:]
    indexes=centers['selected_indices']
    matrix=np.sqrt(np.mean(np.sum((world[:,None]-world[np.asarray(indexes)][None,:])**2,axis=3),axis=2))
    require(np.max(np.min(matrix,axis=1))<=.5,'Selected centers do not cover the declared observed set')
    for point,index in zip(selected,indexes):require(point['pose']==poses[index],'Center index/pose mismatch')
    models={name:assemble(cfg,selected,local,std) for name,std,_ in ARMS}
    archive=out/'provenance';archive.mkdir(parents=True)
    sources={'input-config.json':GUIDES/'config.json','source-region.json':GUIDES/'provenance/source-region.json',
             'shape.json':shape,'center-selection.json':CENTERS,'center-selector.py':CENTERS.parent/'provenance/select_centers.py',
             'local-fit-model.json':FIT/'model-weighted.json','local-fit-report.json':FIT/'report.json',
             'local-fit-artifact-hashes.json':FIT/'artifact-hashes.json','local-fit-provenance-hashes.json':FIT/'provenance-sha256.json',
             'local-fit-diagnostic.py':FIT/'diagnostic.py',**local_dependencies([Path(__file__)])}
    for name,path in sources.items():shutil.copy2(path,archive/name)
    frozen_cfg=copy.deepcopy(cfg);frozen_cfg['shape']=str(archive/'shape.json');write(out/'config.json',frozen_cfg)
    charts={name:dict(path=str(path),sha256=sha(path)) for name,path in (
        ('peak',ROOT/'runs/ab-intermediate-new-peak-geometry-20260920/model.json'),
        ('old_weighted',GUIDES/'model-weighted.json'))}
    protocol=dict(schema='intermediate-contact-atlas-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        physical={key:copy.deepcopy(cfg[key]) for key in PHYSICAL},
        q_window=dict(minimum=2.,maximum=5.,lower_inclusive=True,upper_inclusive=False),
        center_selection=dict(path=str(CENTERS),sha256=sha(CENTERS),count=36,observed_candidate_count=64,
                              observed_maximum_nearest_member_RMS_A=float(matrix.min(axis=1).max())),
        local_fit=dict(path=str(FIT/'model-weighted.json'),sha256=sha(FIT/'model-weighted.json'),
                       source=str(FIT),source_scope='Conditional radius-.5 fit; not a full-basin covariance or occupancy estimate.'),
        assembly='80% equal geometric centers +16% local fitted covariance +4% local covariance broadened by a factor four.',
        hybrid='25% complete q<=5 product cover +75%*(95% frozen atlas +5% capture cube with normalized Haar). Every draw divides by this full density.',
        analysis_charts=charts,
        arms=[dict(name=name,geometric_latent_SD_A=std,components=len(models[name]['weights']),samples_per_population=16384,
                   populations=4,seeds=[seed+1009*i for i in range(4)],
                   output=str(ROOT/f'runs/ab-intermediate-contact-atlas-{name}-4x16384-l64-20260920')) for name,std,seed in ARMS],
        probe_count_per_arm=256,probe_seeds=[101301010,101302019],
        executable=str(BINARY),executable_sha256=BINARY_SHA,lambda_ratio=64.,cloud_replicates=2,
        selection_scope='All center/fit decisions use historical data. Existing local references are calibration data, not untouched holdouts. Fresh fixed-N populations validate the frozen proposal; no refitting or optional stopping during production.',
        coverage_scope='Finite observed-point coverage is not physical basin coverage. Product cover and cube preserve complete target support. Unseen weight remains a statistical question.',
        archived_sha256={name:sha(archive/name) for name in sources})
    write(out/'protocol.json',protocol)
    for name,model in models.items():
        model['proposal_provenance']=dict(kind='Frozen multiscale contact atlas',arm=name,protocol_sha256=sha(out/'protocol.json'),
                                          physical=protocol['physical'],proposal_anchor_index=0,
                                          inference=protocol['selection_scope'])
        write(out/f'model-{name}.json',model)
    seal=dict(protocol_sha256=sha(out/'protocol.json'),config_sha256=sha(out/'config.json'),
              model_sha256={name:sha(out/f'model-{name}.json') for name in models},archived_sha256=protocol['archived_sha256'])
    write(out/'freeze.json',seal)
    atom=AtomUnionAudit(read(shape),cfg['fixed_poses']);reports={}
    for index,(name,_,_) in enumerate(ARMS):
        report,rows=geometry_probe(models[name],cfg,atom,256,protocol['probe_seeds'][index])
        world=[row['pose'] for row in rows];pure=Density(models[name]).evaluate(relative_poses(world,cfg['fixed_poses'][0]))[0]
        description=dict(weight=.75,uniform_probability=.05,anchor_index=0,anchor_pose=cfg['fixed_poses'][0],
                         capture_center=cfg['capture_center'],cube_lengths=[36.]*3)
        scalar=GaussianGuide(description,models[name],cfg,local['shape_sha256'])
        inside=np.array([all(-18<=pose['position'][k]-cfg['capture_center'][k]<18 for k in range(3)) for pose in world])
        uniform=np.where(inside,math.log(.05)-3*math.log(36.),-math.inf)
        expected=np.logaddexp(math.log(.95)+pure,uniform)
        error=float(np.max(np.abs([scalar.log_density(p)[0]-e for p,e in zip(world,expected)])))
        require(error<2e-9,'Scalar and vectorized physical mixture densities disagree')
        report['scalar_vectorized_density_max_error']=error
        with (out/f'geometry-probes-{name}.jsonl').open('w') as handle:
            for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
        reports[name]=report;print(json.dumps(dict(arm=name,valid=report['valid_original_window'],draws=256,density_error=error)),flush=True)
    for name,digest in seal['model_sha256'].items():require(sha(out/f'model-{name}.json')==digest,'Frozen model changed')
    require(sha(out/'protocol.json')==seal['protocol_sha256'],'Frozen protocol changed')
    write(out/'report.json',dict(complete=True,probes=reports,freeze_sha256=sha(out/'freeze.json'),
                               probe_sha256={name:sha(out/f'geometry-probes-{name}.jsonl') for name in models},
                               scope='Geometry probes and independent normalized-density checks only; no new depletant weights or physical acceptance evidence.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare(args.out.resolve())
