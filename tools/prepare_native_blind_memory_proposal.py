#!/usr/bin/env python3
"""Freeze every historical native-blind pair-memory slot without fitting/filtering.

The exporter reads only bank poses for model construction. Their historical
pair sampler's reviewed source and complete terminal checkpoints are archived.
This is a proposal asset, not an equilibrium or assembly calculation.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from prepare_geometry_reciprocal_proposal import read, write, sha, require, SHAPE_SHA, BASE_SHA
from prepare_mobile_reciprocal_benchmark import reciprocal_envelope, density_preflight
from analyze_involution_docking_campaign import ChartAudit
from prepare_shoulder_docking_benchmark import local_dependencies

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/contact-memory-free-1000'
SHAPE=ROOT/'examples/tetramer-shape.json'
CHECKPOINT_HASHES=[
    'c01fa74db7e23604a2c9f81802a2240ab2f23011b735b28c5a738fe2cd4e689a',
    'ff4e22386d523fa111a1989e1bbc3fa3395d7ae3ef1ceae1904eda7de5cf5a98',
    '91ebc9029cb295a7948bba54ca2aff51dc3ce7cc8c562c68ba5ccb10de9dddb7',
    '64b54beef1df47269f92b7e9aadef28377ab17274e98ffc395725e9896eefe3c']
BUNDLE_SHA='da9aaeb5401113b4cbaba5c0c6cfa17b3d0ddb1c5d49470daef1ed6654381746'
BINARY_SHA='9ea570c1918321f90781856a0e4dd7d484dd19ce6b1542187b081d3c7a9e50e5'
MODULES={'src/contact_memory.rs':'03ac0571dee123e3b5cabd07c25f16e68ea966f889e32dcb562a3e51152f6d7f',
         'src/simulation.rs':'11ef30f90cd76e0c3011e49cb493682a742b0db6d1f89292643b0f1e08677499'}
TRANSLATION_STD=1.
ANGLE_STD_DEGREES=3.


def construct(checkpoints, shape):
    """No production poses, native label, likelihood or source weight is used."""
    require(len(checkpoints)==4,'Require all four terminal banks')
    xyz=np.asarray([a['center'] for a in shape['atoms']],float)
    radii=np.asarray([a['radius'] for a in shape['atoms']],float)
    require(len(xyz)>0 and np.isfinite(xyz).all() and np.isfinite(radii).all() and (radii>0).all(),'Bad shape')
    ell=2*max(float(np.sqrt(np.mean(np.sum((xyz-xyz.mean(axis=0))**2,axis=1)))),float(radii.max()))
    covariance=np.diag([TRANSLATION_STD**2]*3+[(ell*np.deg2rad(ANGLE_STD_DEGREES)/2)**2]*3)
    anchors=[];order=[]
    for replicate,checkpoint in enumerate(checkpoints):
        require(checkpoint['completed_sweeps']==1000 and checkpoint['shape_sha256']==SHAPE_SHA
                and checkpoint['model_sha256']==BASE_SHA,'Historical bank identity differs')
        poses=checkpoint['contact_memory_state']['poses']
        require(len(poses)==16,'Require every one of the 16 slots')
        for slot,p in enumerate(poses):
            t=np.asarray(p['position'],float);q=np.asarray(p['orientation'],float)
            require(t.shape==(3,) and q.shape==(4,) and np.isfinite(t).all() and np.isfinite(q).all()
                    and abs(float(q@q)-1)<1e-10,'Invalid bank pose')
            anchors.append(dict(position=t.tolist(),rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix().tolist()))
            order.append(dict(replicate=replicate,slot=slot))
    base=dict(schema='weighted-pose-mixture-v1',coordinate_convention='anchor-body-relative',angular_length=ell,
        shape_sha256=SHAPE_SHA,anchors=anchors,means=[[0.]*6 for _ in anchors],
        covariances=[covariance.tolist() for _ in anchors],weights=[1/64.]*64,dfs=[None]*64)
    return base,order


def validate_hard_centers(base,shape):
    xyz=np.asarray([a['center'] for a in shape['atoms']],float);r=np.asarray([a['radius'] for a in shape['atoms']],float)
    fixed=cKDTree(xyz);pairs_checked=0
    for index,anchor in enumerate(base['anchors']):
        mobile=xyz@np.asarray(anchor['rotation']).T+anchor['position']
        neighbors=fixed.query_ball_point(mobile,r+r.max()+1e-9)
        for i,near in enumerate(neighbors):
            if near:
                d=xyz[near]-mobile[i];radii=r[near]+r[i];pairs_checked+=len(near)
                require(not np.any(np.einsum('ij,ij->i',d,d)<radii*radii),f'Hard-invalid bank slot {index}; do not filter')
    return dict(all_centers_hard_valid=True,atomic_pairs_checked=pairs_checked)


def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Fresh preparation directory required')
    require(sha(SHAPE)==SHAPE_SHA,'Physical shape changed')
    paths=[SOURCE/f'runs/free-r{i:02d}-rj-memory-on' for i in range(4)]
    binary=SOURCE/'provenance/tetramer-mc';bundle=paths[0]/'provenance/source-bundle.json'
    require(sha(binary)==BINARY_SHA and sha(bundle)==BUNDLE_SHA,'Historical executable/source differs')
    require(bundle.read_bytes() in binary.read_bytes(),'Historical source bytes absent from executable')
    historical=read(bundle)
    for name,item in historical['files'].items():
        require(hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256'],'Historical source text/hash mismatch')
    for name,digest in MODULES.items():require(historical['files'][name]['sha256']==digest,'Reviewed native-blind sampler source changed')
    checkpoints=[];configs=[];training=[]
    for i,path in enumerate(paths):
        checkpoint=read(path/'checkpoint.json');config=read(path/'provenance/input-config.json');summary=read(path/'summary.json')
        require(sha(path/'checkpoint.json')==CHECKPOINT_HASHES[i],'Historical bank checkpoint changed')
        require(sha(path/'provenance/input-config.json')==checkpoint['config_sha256'],'Historical configuration binding differs')
        require(sha(path/'provenance/shape.json')==SHAPE_SHA and sha(path/'provenance/frozen-relative-model.json')==BASE_SHA
                and sha(path/'provenance/source-bundle.json')==BUNDLE_SHA,'Historical shape/model/source differs')
        require(config['contact_memory']=={} and config['depletant_radius']==1.5 and config['reservoir_density']==.035,
                'Historical default bank or physical bath changed')
        require(summary['complete'] is True and summary['completed_sweeps']==1000
                and summary['shape_sha256']==SHAPE_SHA and summary['config_sha256']==checkpoint['config_sha256'],
                'Historical terminal summary differs')
        checkpoints.append(checkpoint);configs.append(config)
        training.append(dict(replicate=i,reported_whole_run_cpu_seconds=summary['sampler_cpu_seconds'],
            reported_bank_cpu_seconds=summary['cost']['contact_memory_cpu_seconds'],
            interpretation='Whole-run timing includes production. Bank-only timing excludes initialization; archived summary retains all costs.'))
    base,order=construct(checkpoints,read(SHAPE));geometry=validate_hard_centers(base,read(SHAPE))
    wrapped=reciprocal_envelope(base);density=density_preflight(base,wrapped,[],seed=131200011)
    audit=ChartAudit(wrapped);latent=np.array([.2,-.3,.4,.1,-.2,.3]);errors=[]
    for label in range(128):
        pose,jac=audit.decode(label,latent);errors.append(float(np.max(np.abs(audit.encode(label,pose)-latent))))
        require(abs(audit.gaussian(label,pose)+jac+3*np.log(2*np.pi)+.5*latent@latent)<1e-10,'Chart measure changed')
    require(max(errors)<1e-10,'Chart round trip failed')
    archive=out/'provenance';archive.mkdir(parents=True)
    for i,path in enumerate(paths):
        target=archive/f'bank-r{i:02d}';target.mkdir()
        for name in ('checkpoint.json','summary.json','manifest.json'):shutil.copy2(path/name,target/name)
        shutil.copy2(path/'provenance/input-config.json',target/'input-config.json')
    for name,path in {'shape.json':SHAPE,'historical-binary':binary,'historical-source-bundle.json':bundle,
        'historical-geometry-model.json':paths[0]/'provenance/frozen-relative-model.json',
        **local_dependencies([Path(__file__),Path(__file__).with_name('test_prepare_native_blind_memory_proposal.py')])}.items():
        shutil.copy2(path,archive/name)
    write(out/'base-model.json',base);write(out/'model.json',wrapped)
    preflight=dict(complete=True,**geometry,reciprocal_density=density,maximum_encode_decode_error=max(errors),
                   new_physical_updates=0,new_depletant_clouds=0,all_slots_retained=True)
    write(out/'preflight.json',preflight)
    manifest=dict(schema='native-blind-memory-reciprocal-preparation-v1',model='model.json',model_sha256=sha(out/'model.json'),
        base_model='base-model.json',base_model_sha256=sha(out/'base-model.json'),shape_sha256=SHAPE_SHA,
        base_components=64,virtual_components=128,source_slot_order=order,
        historical_checkpoint_sha256=CHECKPOINT_HASHES,historical_binary_sha256=BINARY_SHA,historical_source_bundle_sha256=BUNDLE_SHA,
        shape_derived_angular_length_A=base['angular_length'],translation_std_A=TRANSLATION_STD,
        small_angle_std_degrees=ANGLE_STD_DEGREES,covariance='uncoupled full-rank geometric scale; no fitted covariance',
        recommended_uniform_weight=.1,native_informed_proposal=False,native_intertetramer_labels_used=False,
        selection='All 64 terminal slots, in replicate then slot order. Equal weights. No filtering, deduplication or compression.',
        historical_training=training,preflight_sha256=sha(out/'preflight.json'),production_launched=False,
        scope='Frozen native-blind physically sampled isolated-pair bank. Native intratetramer structure supplied. Historical banks are not claimed equilibrated, independent or representative of cooperative neighborhoods. No assembly verdict from this export.')
    write(out/'manifest.json',manifest)
    write(out/'freeze.json',dict(files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(dict(complete=True,out=str(out),model_sha256=manifest['model_sha256'],base_components=64,virtual_components=128,
               physical_updates=0,maximum_encode_decode_error=max(errors)))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    prepare(parser.parse_args().out)
