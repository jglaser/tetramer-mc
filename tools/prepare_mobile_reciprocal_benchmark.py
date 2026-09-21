#!/usr/bin/env python3
"""Prepare a frozen exact-reciprocal control of the existing 150-chart atlas.

No covariance fit, contact chart, physical trajectory or bath cloud is generated.
The reciprocal component is the exact nonlinear pose inverse of its Gaussian
base, with unit physical Haar Jacobian; it is not a Gaussian approximation.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_mobile_posterior_pilot import validate_model
from prepare_shoulder_docking_benchmark import local_dependencies
from prepare_smc_normalizer_atlas import (
    Density, arrays, read, reciprocal_arrays, relative_poses, sha,
    unwrap_proposal_model, write,
)
from run_mobile_posterior_pilot import require, safe_relative

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'runs/mobile-competing-atlas-benchmark-20260921/legacy'
SOURCE_MANIFEST_SHA='219a01f139916e6ea2df2590393e17ae5447d9aadffa4f5fe79a455343198648'
SOURCE_MODEL_SHA='d0f3f82960c218ecbb0617b78412bb8072a3f33a726e128465d9418e0690af68'
SNAPSHOT_SHA='4f74942180010463fbefb51c6b116fad675428c27533d6558e8bb0b87e1029b9'
CONFIG_SHAS=dict(c0='ad8325b07da15803a9aea784cbd48e2fe7213522370d750a37dca9ed01815c5e',
    c09='d6b5cadfa0983f3a447cdece654976ea7804c6caa3f5ef34f202a8a172475cde')
SEED_BASE=119201010
SCHEMA='mobile-reciprocal-atlas-preparation-v1'


def reciprocal_envelope(base):
    envelope=dict(schema='reciprocal-pose-mixture-v1',base_model=copy.deepcopy(base),
        reciprocal_components=[True]*len(base['weights']))
    unwrapped,flags=unwrap_proposal_model(envelope)
    require(unwrapped==base and all(flags),'Reciprocal envelope changed its base')
    return envelope


def inverse_poses(poses):
    t,_,r=arrays(poses);t,r=reciprocal_arrays(t,r)
    q=Rotation.from_matrix(r).as_quat()[:,[3,0,1,2]]
    return [dict(position=x.tolist(),orientation=y.tolist()) for x,y in zip(t,q)]


def density_preflight(base,wrapped,physical_poses,seed=119200011):
    legacy,reciprocal=Density(base),Density(wrapped)
    rng=np.random.default_rng(seed)
    # One representative Gaussian draw per base component, plus all six
    # ordered physical pair poses. These checks use no target weights.
    probes=[legacy.draw_component(rng,k,1)[0] for k in range(len(base['weights']))]
    pairs=[]
    for moving in range(len(physical_poses)):
        for anchor in range(len(physical_poses)):
            if moving!=anchor:
                pairs.append(dict(moving=moving,anchor=anchor,
                    pose=relative_poses([physical_poses[moving]],physical_poses[anchor])[0]))
    probes += [p['pose'] for p in pairs]
    reverse=inverse_poses(probes)
    old=legacy.evaluate(probes)[0];reverse_old=legacy.evaluate(reverse)[0]
    actual=reciprocal.evaluate(probes)[0];inverse_actual=reciprocal.evaluate(reverse)[0]
    expected=np.logaddexp(old,reverse_old)-np.log(2.)
    error=float(np.max(np.abs(actual-expected)))
    symmetry=float(np.max(np.abs(actual-inverse_actual)))
    require(error<2e-8 and symmetry<2e-8,'Exact reciprocal density identity failed')
    require(len(reciprocal.weights)==2*len(base['weights']),'Virtual branch count differs')
    np.testing.assert_array_equal(reciprocal.weights[::2],.5*np.asarray(base['weights']))
    np.testing.assert_array_equal(reciprocal.weights[1::2],.5*np.asarray(base['weights']))
    for i,pair in enumerate(pairs):
        k=len(probes)-len(pairs)+i
        pair.update(legacy_log_G=float(old[k]),reciprocal_log_G=float(actual[k]),
            log_density_gain=float(actual[k]-old[k]))
    return dict(passed=True,probe_count=len(probes),seed=seed,
        maximum_log_density_identity_error=error,maximum_reciprocal_symmetry_error=symmetry,
        physical_pair_proposal_densities=pairs,physical_updates=0,bath_clouds=0,
        scope='Proposal-law validation only. Density gain does not predict acceptance, stationary weights or speedup.')


def configured(source,variant,mode,provenance):
    require(variant in ('legacy','reciprocal') and mode in ('c0','c09'),'Unknown control')
    result=copy.deepcopy(source)
    result['shape']=str(provenance/'shape.json')
    result['monomer_shape']=str(provenance/'monomer-shape.json')
    index=2*('legacy','reciprocal').index(variant)+('c0','c09').index(mode)
    result['seed']=SEED_BASE+1009*index
    result['metadata'].update(atlas=variant,native_pair_motifs=str(provenance/'native-pair-motifs.json'),
        added_contact_charts=0,reciprocal_virtual_components=(300 if variant=='reciprocal' else 0),
        exact_reciprocal_proposal=(variant=='reciprocal'),
        scope='All three tetramers remain mobile from the same selected, non-equilibrated competing snapshot. The unchanged 150-chart atlas is native-informed. Reciprocal mode exactly symmetrizes its physical pose density; no charts, fitted parameters or physical prior are added.')
    return result


def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Use a fresh preparation directory')
    require(sha(SOURCE/'manifest.json')==SOURCE_MANIFEST_SHA,'Source campaign manifest changed')
    manifest=read(SOURCE/'manifest.json');archive=SOURCE/'provenance'
    for name,digest in manifest['input_sha256'].items():
        require(sha(archive/safe_relative(name))==digest,'Source archive changed: '+name)
    require(sha(archive/'model.json')==SOURCE_MODEL_SHA,'Require the unchanged 150-chart legacy atlas')
    require(sha(archive/'fixed-snapshot.json')==SNAPSHOT_SHA,'Exact competing snapshot changed')
    base=read(archive/'model.json');shape_hash=sha(archive/'tetramer-shape.json')
    require(validate_model(base,shape_hash)==150,'Require exactly 150 base components')
    wrapped=reciprocal_envelope(base)
    snapshot=read(archive/'fixed-snapshot.json');poses=snapshot['original_global']['poses']
    sourceconfigs={mode:read(archive/f'sourceconfigs/{mode}.json') for mode in ('c0','c09')}
    for mode,config in sourceconfigs.items():
        require(sha(archive/f'sourceconfigs/{mode}.json')==CONFIG_SHAS[mode],'Source config changed')
        require(config['initial_poses']==poses and len(poses)==3,'Source snapshot mismatch')
        require(not config['fixed_body_indices'] and not config['seed_labels'],'Source contains a fixed body')
        require(config['boundary']['kind']=='spherical','Reciprocal mode requires the open spherical boundary')
        require(config['frozen_posterior']==dict(probability=.5,correlation=0. if mode=='c0' else .9),'Wrong posterior control')
    checks=density_preflight(base,wrapped,poses)
    provenance=out/'provenance';provenance.mkdir(parents=True)
    sources={'shape.json':archive/'tetramer-shape.json','monomer-shape.json':archive/'monomer-shape.json',
        'native-pair-motifs.json':archive/'native-pair-motifs.json','fixed-snapshot.json':archive/'fixed-snapshot.json',
        'source-model.json':archive/'model.json','source-manifest.json':SOURCE/'manifest.json'}
    sources.update({f'source-{mode}.json':archive/f'sourceconfigs/{mode}.json' for mode in ('c0','c09')})
    sources.update(local_dependencies([Path(__file__).resolve()]))
    for name,path in sources.items():shutil.copy2(path,provenance/name)
    shutil.copy2(archive/'model.json',out/'model-legacy.json')
    write(out/'model-reciprocal.json',wrapped)
    (out/'configs').mkdir();controls=[]
    for variant in ('legacy','reciprocal'):
        for mode in ('c0','c09'):
            cfg=configured(sourceconfigs[mode],variant,mode,provenance)
            path=out/'configs'/f'{variant}-{mode}.json';write(path,cfg)
            model=out/f'model-{variant}.json'
            controls.append(dict(id=f'{variant}-{mode}',atlas=variant,mode=mode,
                correlation=cfg['frozen_posterior']['correlation'],seed=cfg['seed'],
                config=str(path),config_sha256=sha(path),model=str(model),model_sha256=sha(model)))
    write(out/'preflight.json',checks)
    plan=dict(schema=SCHEMA,production_launched=False,controls=controls,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,legacy_model_sha256=SOURCE_MODEL_SHA,
        source_snapshot_sha256=SNAPSHOT_SHA,base_components=150,reciprocal_virtual_components=300,
        reciprocal_components=[True]*150,
        seed_policy='Fresh r0 seeds 119201010 + 1009*j for four atlas/mode templates; r1 adds 4036. All eight are distinct from the completed comparison.',
        construction='G_R(g) = [G(g) + G(g^-1)]/2 with unchanged base Gaussian anchors, means, covariances and component weights. Every base chart has ordinary and exact nonlinear inverse virtual branches of equal mass. Haar inversion has unit Jacobian.',
        physical_scope='One shared, deliberately selected non-equilibrium trapped start; all three tetramers mobile. Both proposals remain native-informed. No geometry fit, density-dependent training, changed target, or added contact chart; this short comparison does not establish equilibrium speedup or crystal assembly.',
        balance='Capture evaluates the full uniform-plus-G_R density; posterior source responsibilities and reverse correction use the full G_R. Virtual source/target traces are wrapped in exact physical pose inversion, leaving the Gaussian latent involution Jacobian unchanged.',
        shapes=dict(tetramer_sha256=shape_hash,monomer_sha256=sha(provenance/'monomer-shape.json')),
        input_sha256={name:sha(provenance/name) for name in sources},
        outputs_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*.json')) if not p.is_relative_to(provenance)})
    write(out/'plan.json',plan)
    write(out/'freeze.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return dict(output=str(out),plan_sha256=sha(out/'plan.json'),base_components=150,
        virtual_components=300,controls=4,preflight_passed=True,physical_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(__import__('json').dumps(prepare(args.out),indent=2))


if __name__=='__main__':main()
