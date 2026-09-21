#!/usr/bin/env python3
"""Archive an unchanged shape-only contact atlas in its exact reciprocal envelope.

No native docking catalogue, production configuration, bath weight or learned
covariance is read. Historical generator source and shape-derived covariances
are checked before a new model is written. Proposal-law probes are not MC.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_mobile_reciprocal_benchmark import reciprocal_envelope, density_preflight
from prepare_shoulder_docking_benchmark import local_dependencies
from analyze_involution_docking_campaign import ChartAudit

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/free-tetramer-atlases/geometry.json'
GENERATION=BASE.with_name('geometry.provenance.json')
SHAPE=ROOT/'examples/tetramer-shape.json'
BASE_SHA='1a39c8cc0d2977016d5dcd2072328de99612495eab76f6aac8a4aca5086e1519'
GENERATION_SHA='0e18a3511b8e2c2e6ab9c79001e70d7d885263315353c2b5dd32ba5e1ee16161'
SHAPE_SHA='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'
GENERATOR_SHA='b2e08464bb4ec5e2688906e46c775d38e7c1a117b186676fd923e2ed372b010d'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_construction(base,generation,shape):
    construction=base['construction'];args=generation['arguments']
    require(construction==generation['construction'] and generation['physical_shape']==shape,'Geometry provenance differs')
    require(args==dict(angle_width_degrees=3.,angular_length=None,count=96,gap=.25,
        out='runs/free-tetramer-atlases/geometry.json',seed=2026091907,
        shape='examples/tetramer-shape.json',translation_width=1.,uncoupled=False),'Generator inputs changed')
    require(construction['kind']=='geometry-only-outermost-ray-contact-v1' and
        construction['native_information'] is False and construction['attempted_rays']==96
        and construction['missed_rays']==0,'Shape-only construction changed')
    require(base['shape_sha256']==construction['shape_sha256']==SHAPE_SHA,'Different shape')
    require(base['coordinate_convention']=='anchor-body-relative' and all(x is None for x in base['dfs']),
        'Expected ordinary body-relative Gaussians')
    require(all(len(base[k])==96 for k in ('anchors','means','covariances','weights','contact_witnesses','dfs')),'Component count differs')
    require(base['means']==[[0.]*6]*96 and base['weights']==[1/96.]*96,'Means or equal weights changed')
    xyz=np.asarray([a['center']for a in shape['atoms']],float)
    radii=np.asarray([a['radius']for a in shape['atoms']],float)
    ell=2*max(float(np.sqrt(np.mean(np.sum((xyz-xyz.mean(axis=0))**2,axis=1)))),float(radii.max()))
    require(abs(ell-base['angular_length'])<1e-10,'Angular scale is not shape derived')
    angle=np.deg2rad(args['angle_width_degrees']);errors=[]
    for a,c,w in zip(base['anchors'],base['covariances'],base['contact_witnesses']):
        contact=w['radial_contact'];direction=np.asarray(contact['direction']);rotation=np.asarray(a['rotation'])
        require(abs(np.linalg.norm(direction)-1)<1e-10 and np.allclose(rotation.T@rotation,np.eye(3),atol=1e-12)
            and abs(np.linalg.det(rotation)-1)<1e-10,'Nonproper geometric anchor')
        q=np.asarray(contact['orientation'])[[1,2,3,0]]
        require(np.allclose(rotation,Rotation.from_quat(q).as_matrix(),atol=1e-12),'Contact rotation differs')
        require(np.allclose(a['position'],direction*(contact['distance']+construction['effective_gap']),atol=1e-12),
            'Contact anchor differs from recorded outward offset')
        lever=np.asarray(w['surface_point'])-direction*contact['distance']
        require(np.allclose(lever,w['mobile_surface_lever'],atol=1e-12),'Contact lever changed')
        u,v,z=lever;cross=np.array([[0.,-z,v],[z,0.,-u],[-v,u,0.]])
        factor=np.zeros((6,6));factor[:3,:3]=np.eye(3)*args['translation_width']
        factor[:3,3:]=angle*cross;factor[3:,3:]=np.eye(3)*ell*angle/2
        expected=factor@factor.T;error=float(np.max(np.abs(np.asarray(c)-expected)))
        require(error<1e-10,'Covariance is not recorded shape-pivot rolling construction')
        np.linalg.cholesky(c);errors.append(error)
    return dict(passed=True,shape_derived_angular_length_A=ell,
        maximum_covariance_reconstruction_error=max(errors),
        minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(base['covariances']).min()),
        native_geometry_inputs=0,bath_queries=0,physical_updates=0)


def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Fresh output required; no overwrite')
    require(sha(BASE)==BASE_SHA and sha(GENERATION)==GENERATION_SHA and sha(SHAPE)==SHAPE_SHA,'Historical input changed')
    base,generation,shape=read(BASE),read(GENERATION),read(SHAPE)
    item=generation['source_bundle']['files']['src/bin/contact_atlas.rs']
    require(item['sha256']==GENERATOR_SHA and hashlib.sha256(item['text'].encode()).hexdigest()==GENERATOR_SHA
        and sha(ROOT/'src/bin/contact_atlas.rs')==GENERATOR_SHA,'Historical generator source differs')
    geometric=validate_construction(base,generation,shape)
    wrapped=reciprocal_envelope(base)
    density=density_preflight(base,wrapped,[],seed=129100011)
    audit=ChartAudit(wrapped);errors=[]
    latent=np.array([.2,-.3,.4,.1,-.2,.3])
    for label in range(192):
        pose,jac=audit.decode(label,latent);errors.append(float(np.max(np.abs(audit.encode(label,pose)-latent))))
        gaussian=-3*np.log(2*np.pi)-.5*latent@latent
        require(abs(audit.gaussian(label,pose)+jac-gaussian)<1e-10,'Normalized chart measure differs')
    require(max(errors)<1e-10,'Geometry chart round trip failed')
    provenance=out/'provenance';provenance.mkdir(parents=True)
    for name,path in {'source-model.json':BASE,'generation.json':GENERATION,'shape.json':SHAPE,
        'generator.rs':ROOT/'src/bin/contact_atlas.rs',**local_dependencies([Path(__file__),
            Path(__file__).with_name('test_prepare_geometry_reciprocal_proposal.py')])}.items():
        shutil.copy2(path,provenance/name)
    write(out/'model.json',wrapped)
    preflight=dict(geometry=geometric,reciprocal_density=density,all_virtual_branches=192,
        maximum_encode_decode_error=max(errors),physical_updates=0,bath_queries=0)
    write(out/'preflight.json',preflight)
    manifest=dict(schema='geometry-reciprocal-proposal-preparation-v1',model='model.json',model_sha256=sha(out/'model.json'),
        base_model='provenance/source-model.json',base_model_sha256=BASE_SHA,
        generation_provenance='provenance/generation.json',generation_provenance_sha256=GENERATION_SHA,
        generator_source='provenance/generator.rs',generator_source_sha256=GENERATOR_SHA,shape_sha256=SHAPE_SHA,
        base_components=96,virtual_components=192,reciprocal_components=[True]*96,
        native_informed_proposal=False,native_geometry_used_in_preparation=False,
        preflight_sha256=sha(out/'preflight.json'),preparer_sha256=sha(__file__),production_launched=False,
        scope='Unchanged historical shape-only atlas with exact reciprocal branches. Proposal distribution only; no native data, fitting, bath weights or production poses used. Full Gaussian tails remain; any hard rejection belongs to subsequent MC.')
    write(out/'manifest.json',manifest)
    write(out/'freeze.json',dict(files={p.relative_to(out).as_posix():sha(p)for p in sorted(out.rglob('*'))if p.is_file()}))
    print(json.dumps(dict(complete=True,out=str(out),model_sha256=manifest['model_sha256'],preflight=preflight)))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();prepare(args.out)
