#!/usr/bin/env python3
"""Freeze a full-vessel contact coverage proposal, without physical sampling.

Saved rows supply geometry only. The normalized proposal is not a model of
physical basin probabilities; every chart remains untruncated and the runtime
must evaluate the complete two-anchor mixture including its uniform floor.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from diagnose_mobile_native_remainder import AtomicGeometry, compose, pose, read, relative, rt, sha, write
from normalizer_proposal_density import NormalizerProposalDensity, scalar_cholesky
from prepare_mobile_posterior_pilot import validate_model
from prepare_shoulder_docking_benchmark import local_dependencies
from prepare_smc_normalizer_atlas import Density, unwrap_proposal_model

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'runs/mobile-competing-reference-preparation-20260921'
CAMPAIGN=ROOT/'runs/mobile-full-capture-campaign-20260921'
COMPARISON=ROOT/'runs/mobile-full-capture-comparison-20260921'
DEFINITION=CAMPAIGN/'provenance/native-region/definition.json'
INPUTS=DEFINITION.parent/'inputs'
ALLOCATION={'original_reciprocal':.20,'known_competitor':.15,'diverse_unregistered_contacts':.30,
            'confirmed_native_core':.15,'original_native':.10,'outside_d170_native':.10}
SCALES=(1.,4.)
SOURCE_MODEL_SHA='dc9218c9706e1691af9336ef1c0e87a75a86994933dfd41cc6e1d1cd3dc52aa3'
SHAPE_SHA='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'


def require(ok,message):
    if not ok:raise ValueError(message)


def contact_candidates(rows,labels):
    """No cloud, density or weight appears in the eligibility predicate."""
    result=[]
    for value in rows:
        row=value['sample'];key=(value['arm'],value['population'],row['draw'])
        if not row['hard_valid']:continue
        require(key in labels,'Missing saved native classification')
        label=labels[key]
        require(type(label['native_any']) is bool,'Invalid saved entry predicate')
        if row['depletion_contact'] is True and not label['native_any']:
            result.append(dict(value,classification=label))
    return sorted(result,key=lambda v:(v['arm'],v['population'],v['sample']['draw']))


def member_embedding(value,members):
    t,r=rt(value)
    return (np.asarray(members)@r.T+t).reshape(-1)/math.sqrt(len(members))


def geometric_maximin(candidates,members,initial_pose,count):
    """Greedy maximum minimum labeled-member RMS distance from prior centers.

    The initial center is the already retained competitor. Lexical source ID
    breaks ties. A common isometry leaves every distance unchanged.
    """
    require(type(count) is int and 0<count<=len(candidates),'Invalid center count')
    x=np.asarray([member_embedding(v['sample']['pose'],members)for v in candidates])
    distances=np.linalg.norm(x-member_embedding(initial_pose,members),axis=1)
    selected=[]
    for _ in range(count):
        index=int(np.argmax(distances));distance=float(distances[index])
        require(distance>0,'Insufficient geometrically distinct contacts')
        selected.append(dict(candidates[index],selection_distance_A=distance,candidate_index=index))
        distances=np.minimum(distances,np.linalg.norm(x-x[index],axis=1));distances[index]=-np.inf
    return selected


def one_component(base,index):
    model=copy.deepcopy(base)
    for key in ('anchors','means','covariances'):model[key]=[copy.deepcopy(base[key][index])]
    model['weights']=[1.]
    return model


def recenter(template,relative_pose):
    model=copy.deepcopy(template);t,r=rt(relative_pose)
    model['anchors']=[dict(position=t.tolist(),rotation=r.tolist())];model['means']=[[0.]*6]
    return model


def chart_center(model,index=0):
    a=model['anchors'][index];x=np.asarray(model['means'][index])
    r=Rotation.from_quat(np.r_[x[3:]/model['angular_length'],1.]).as_matrix()@np.asarray(a['rotation'])
    return pose(np.asarray(a['position'])+x[:3],r)


def build_model(original,sites):
    base,flags=unwrap_proposal_model(original)
    require(len(base['weights'])==150 and all(flags),'Require unchanged reciprocal 150 atlas')
    result=copy.deepcopy(base);result['weights']=[.2*w for w in base['weights']]
    flags=list(flags);blocks=[dict(name='original_reciprocal',base_indices=list(range(150)),
        learned_mass=.2,intended_anchor=None,intended_anchor_learned_mass=None)]
    counts={name:sum(s['block']==name for s in sites)for name in ALLOCATION if name!='original_reciprocal'}
    require(all(n>0 for n in counts.values()),'Empty coverage block')
    require(set(counts)=={s['block']for s in sites},'Unknown coverage block')
    for site in sites:
        require(site['anchor_index'] in (0,1),'Unknown intended fixed anchor')
        model=site['model'];validate_model(model,base['shape_sha256'])
        require(len(model['weights'])==1 and model['angular_length']==base['angular_length'],'Incompatible single chart')
        indices=[];mass=ALLOCATION[site['block']]/counts[site['block']]
        for scale in SCALES:
            indices.append(len(result['weights']))
            result['anchors'].append(copy.deepcopy(model['anchors'][0]));result['means'].append(copy.deepcopy(model['means'][0]))
            result['covariances'].append((np.asarray(model['covariances'][0])*scale**2).tolist())
            result['weights'].append(mass/len(SCALES));flags.append(False)
        blocks.append(dict(name=site['block'],site_id=site['site_id'],base_indices=indices,
            learned_mass=mass,intended_anchor=site['anchor_index'],intended_anchor_learned_mass=mass/2,
            incidental_anchor_learned_mass=mass/2,standard_deviation_scales=list(SCALES)))
    validate_model(result,base['shape_sha256'])
    return dict(schema='reciprocal-pose-mixture-v1',base_model=result,reciprocal_components=flags),blocks


def effective_ordinary(model):
    density=Density(model)
    density.lower=np.array([scalar_cholesky(c)for c in model['covariances']])
    density.logdet=np.log(np.diagonal(density.lower,axis1=1,axis2=2)).sum(axis=1)
    return density


def validate_density(original,coverage,sites,fixed,bundle):
    old=NormalizerProposalDensity(original,source_bundle=bundle)
    new=NormalizerProposalDensity(coverage,source_bundle=bundle)
    base=coverage['base_model'];oldbase=original['base_model']
    require(all(base[k][:150]==oldbase[k]for k in ('anchors','means','covariances')),'Original charts changed')
    # Mathematical Gaussian probes only: no overlaps, bath or physical Markov chain.
    rng=np.random.default_rng(126601010)
    probes=[chart_center(base,k)for k in range(150,len(base['weights']))]
    probes += old.draw_component(rng,0,3)+old.draw_component(rng,1,3)
    for k in range(300,len(new.weights)):probes += new.draw_component(rng,k,1)
    observed=new.evaluate(probes)[0]
    pieces=[old.evaluate(probes)[0]+math.log(.2)]
    for k in range(150,len(base['weights'])):
        pieces.append(effective_ordinary(one_component(base,k)).evaluate(probes)[0]+math.log(base['weights'][k]))
    expected=logsumexp(np.array(pieces),axis=0)
    require(np.isfinite(observed).all(),'Nonfinite mixture at component probes')
    error=float(np.max(abs(observed-expected)));require(error<2e-10,'Block mixture density mismatch')
    # Marginalize BOTH anchors and combine a cube/Haar floor, including outside
    # the cube where only untruncated Gaussian tails remain.
    lab=[compose(fixed[0],p)for p in probes]+[dict(position=[400.,0.,0.],orientation=[1.,0.,0.,0.])]
    logs=np.array([new.evaluate([relative(p,a)for p in lab])[0]for a in fixed])
    logg=logsumexp(logs,axis=0)-math.log(2.)
    direct=[]
    for a in fixed:
        ps=[relative(p,a)for p in lab]
        terms=[old.evaluate(ps)[0]+math.log(.2)]
        for k in range(150,len(base['weights'])):
            terms.append(effective_ordinary(one_component(base,k)).evaluate(ps)[0]+math.log(base['weights'][k]))
        direct.append(logsumexp(terms,axis=0))
    directg=logsumexp(direct,axis=0)-math.log(2.)
    inside=np.all(abs(np.array([p['position']for p in lab]))<=273.,axis=1)
    uniform=np.where(inside,-3*math.log(546.),-np.inf)
    full=np.logaddexp(math.log(.9)+logg,math.log(.1)+uniform)
    reference=np.logaddexp(math.log(.9)+directg,math.log(.1)+uniform)
    fullerror=float(np.max(abs(full-reference)));require(fullerror<2e-9,'Anchor-marginal full density mismatch')
    return dict(passed=True,base_components=len(base['weights']),virtual_components=len(new.weights),
        reciprocal_base_components=sum(coverage['reciprocal_components']),ordinary_added_components=len(base['weights'])-150,
        normalized_virtual_weight_sum=float(new.weights.sum()),maximum_block_density_log_error=error,
        maximum_two_anchor_uniform_full_density_log_error=fullerror,deterministic_math_probe_count=len(probes),
        scalar_factor_certificates=new.factor_diagnostics,factor_source_binding=new.algorithm_metadata['source_binding'],
        normalization_argument='Each Gaussian is normalized in R6, Cayley maps onto SO(3) except its Haar-null seam; physical density divides by the exact translation-Haar Jacobian. Inverse SE(3) branches preserve that measure. Positive component weights sum to one, anchor averaging preserves one, and the normalized cube/Haar floor completes q.',
        scope='Untruncated proposal law only. Wall/core/capture invalidity is a target zero, never a proposal truncation or conditioning.')


def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Require a fresh preparation directory')
    for name,digest in read(COMPARISON/'freeze.json').items():require(sha(COMPARISON/name)==digest,'Saved comparison changed: '+name)
    analysis=read(COMPARISON/'analysis.json');require(analysis['complete'],'Incomplete saved assessment')
    definition=read(DEFINITION)
    for name,digest in definition['input_sha256'].items():require(sha(INPUTS/name)==digest,'Native input changed: '+name)
    original_path=CAMPAIGN/'epsilon-0p1/provenance/model.json'
    require(sha(original_path)==SOURCE_MODEL_SHA,'Original reciprocal model changed')
    original=read(original_path);base=original['base_model']
    cfg=read(REFERENCE/'config.json');shape=read(INPUTS/'tetramer-shape.json');snapshot=read(INPUTS/'fixed-snapshot.json')
    require(sha(INPUTS/'tetramer-shape.json')==SHAPE_SHA and base['shape_sha256']==SHAPE_SHA,'Shape mismatch')
    require(cfg==read(INPUTS/'physical-config.json'),'Scaffold config mismatch')
    require(cfg['depletant_radius']==1.5 and cfg['reservoir_density']==.035,'Physical bath mismatch')
    fixed=cfg['fixed_poses'];members=np.array([p['position']for p in cfg['metadata']['rigid_members']])
    wall=dict(center=snapshot['original_global']['wall_center'],radius=snapshot['original_global']['wall_radius'])
    require(wall==dict(center=[0.,0.,0.],radius=223.32617672378387),'Original wall differs')
    geometry=AtomicGeometry(shape,fixed,wall)
    source_paths=[original_path,REFERENCE/'config.json',DEFINITION,COMPARISON/'analysis.json',COMPARISON/'freeze.json']
    rows=[];labels={}
    for arm in analysis['arms']:
        lp=COMPARISON/f"{arm['arm']}-native-labels.jsonl";require(sha(lp)==arm['native_labels_sha256'],'Saved labels changed');source_paths.append(lp)
        for v in map(json.loads,lp.read_text().splitlines()):labels[(arm['arm'],v['population'],v['draw'])]=v['classification']
        for pop in arm['populations']:
            path=CAMPAIGN/arm['arm']/'runs'/pop['id']/'samples.jsonl'
            require(sha(path)==arm['source_sha256'][str(path)],'Audited row hash changed');source_paths.append(path)
            population=list(map(json.loads,path.read_text().splitlines()));require(len(population)==8192,'Draw allocation changed')
            rows.extend(dict(arm=arm['arm'],population=pop['id'],sample=r)for r in population)
    candidates=contact_candidates(rows,labels);trap=snapshot['original_global']['poses'][0]
    selected=geometric_maximin(candidates,members,trap,8)
    geometric=read(REFERENCE/'model-competitor-geometric.json');native=read(REFERENCE/'model-native-original.json')
    require(np.max(abs(np.array(chart_center(geometric)['position'])-np.array(relative(trap,fixed[0])['position'])))<1e-12,'Trap chart mismatched')
    motifs={m['id']:dict(position=m['relative_position'],orientation=m['relative_orientation'])for m in read(INPUTS/'native-pair-motifs.json')['motifs']}
    sites=[]
    def add(block,site_id,model,anchor_index,covariance_source,**extra):
        center=chart_center(model);compositions=[]
        for i,a in enumerate(fixed):
            physical=compose(a,center);check=geometry.check(physical)
            compositions.append(dict(anchor_index=i,fixed_body_id=(2,1)[i],role='intended'if i==anchor_index else'incidental',
                pose=physical,center_norm_A=float(np.linalg.norm(physical['position'])),atomic_geometry=check))
        sites.append(dict(block=block,site_id=site_id,model=model,anchor_index=anchor_index,
            covariance_source=covariance_source,compositions=compositions,**extra))
    add('known_competitor','observed-trap',geometric,0,'inputs/model-competitor-geometric.json:covariances[0]')
    for n,item in enumerate(selected):
        check=geometry.check(item['sample']['pose']);require(check['hard_valid_at_zero_tolerance']and check['original_atomic_wall_valid'],'Selected saved contact fails independent geometry')
        anchor_index=min(range(2),key=lambda i:check['fixed_neighbor_gaps'][i]['minimum_surface_gap_A'])
        require(check['fixed_neighbor_gaps'][anchor_index]['minimum_surface_gap_A']<3.,'Selected no-entry pose is unbound')
        model=recenter(geometric,relative(item['sample']['pose'],fixed[anchor_index]))
        add('diverse_unregistered_contacts',f'geometry-{n:02}',model,anchor_index,'inputs/model-competitor-geometric.json:covariances[0]',
            saved_identity=dict(arm=item['arm'],population=item['population'],draw=item['sample']['draw']),
            selection_distance_A=item['selection_distance_A'],candidate_index=item['candidate_index'],saved_native_classification=item['classification'])
    core=recenter(one_component(base,19),motifs[7])
    add('confirmed_native_core','motif7-4-core',core,0,'original-reciprocal.json:base_model.covariances[19]',ideal_motif_id=7)
    add('original_native','original-native-r4',native,0,'inputs/model-native-original.json:covariances[0]')
    for i,motif in ((0,0),(1,6),(1,13)):
        add('outside_d170_native',f'body{(2,1)[i]}-motif{motif}',recenter(geometric,motifs[motif]),i,
            'inputs/model-competitor-geometric.json:covariances[0]',ideal_motif_id=motif)
        check=sites[-1]['compositions'][i]
        require(check['center_norm_A']>170 and check['atomic_geometry']['hard_valid_at_zero_tolerance']and check['atomic_geometry']['original_atomic_wall_valid'],'Outside-capture native witness differs')
    coverage,blocks=build_model(original,sites)
    bundle=ROOT/'runs/mobile-full-capture-source-review-20260921/source-bundle.json'
    validation=validate_density(original,coverage,sites,fixed,bundle);source_paths.append(bundle)
    out.mkdir();(out/'inputs').mkdir();(out/'source').mkdir()
    shutil.copy2(original_path,out/'original-reciprocal.json')
    sources={'physical-config.json':REFERENCE/'config.json','tetramer-shape.json':INPUTS/'tetramer-shape.json',
        'fixed-snapshot.json':INPUTS/'fixed-snapshot.json','native-pair-motifs.json':INPUTS/'native-pair-motifs.json',
        'model-competitor-geometric.json':REFERENCE/'model-competitor-geometric.json',
        'model-native-original.json':REFERENCE/'model-native-original.json','source-bundle.json':bundle}
    for name,path in sources.items():shutil.copy2(path,out/'inputs'/name);source_paths.append(path)
    for name,path in local_dependencies([Path(__file__)]).items():shutil.copy2(path,out/'source'/name)
    test=ROOT/'tools/test_mobile_wall_proposal.py';shutil.copy2(test,out/'source'/test.name)
    write(out/'coverage-reciprocal.json',coverage);write(out/'validation.json',validation)
    write(out/'sites.json',[{k:v for k,v in s.items()if k!='model'}for s in sites])
    with (out/'selected-saved-rows.jsonl').open('x')as stream:
        for v in selected:stream.write(json.dumps(v,allow_nan=False)+'\n')
    # Largest saved weight is an inspection witness only, never a center score.
    held=max(candidates,key=lambda v:v['sample']['log_importance_weight'])
    held_id=dict(arm=held['arm'],population=held['population'],draw=held['sample']['draw'])
    write(out/'held-out-inspection.json',dict(identity=held_id,sample=held['sample'],classification=held['classification'],
        selected_by_geometry=any((v['arm'],v['population'],v['sample']['draw'])==(held['arm'],held['population'],held['sample']['draw'])for v in selected),
        scope='Largest-weight no-entry contact is inspection only; its random cloud factors did not select or tune the proposal. Discovery rows cannot validate this guide independently.'))
    manifest=dict(schema='mobile-wall-proposal-preparation-v1',complete=True,physical_simulations_launched=False,
        arms={'original':dict(model='original-reciprocal.json',sha256=sha(out/'original-reciprocal.json'),base_components=150,virtual_components=300),
              'coverage':dict(model='coverage-reciprocal.json',sha256=sha(out/'coverage-reciprocal.json'),base_components=178,virtual_components=328)},
        requested_runtime=dict(uniform_weight=.1,covariance_scale=1.,anchor_indices=[0,1],anchor_probabilities=[.5,.5],
            capture_center=[0.,0.,0.],capture_radius=273.,atomic_wall=wall,depletant_radius=1.5,reservoir_density=.035),
        learned_allocation=ALLOCATION,blocks=blocks,sites='sites.json',validation='validation.json',
        selection=dict(eligible_contacts=len(candidates),selected_count=8,source_draws=len(rows),
            predicate='hard_valid AND depletion_contact AND NOT stateless native_any; unbound poses excluded',
            score='Greedy maximum minimum RMS displacement of four labeled member centers in the common laboratory frame, initialized by known trap; lexical (arm,population,draw) tie break.',
            score_uses_bath_or_density_or_weight=False,covariance_fitting=False,standard_deviation_scales=list(SCALES),
            highest_weight_inspection='held-out-inspection.json'),
        coordinate_semantics='Body-relative charts are applied to either fixed neighbor with probability 1/2. Every targeted site has half its learned mass on its intended anchor and half on the incidental anchor. Incidental centers may clash; retain those unconditional target zeros. Covariance copying/recentering/scaling defines normalized Gaussian proposals, not exact transported Gaussian laws.',
        normalization='Active partial reciprocal envelope: original 150 components retain exact inverse branches; all 28 added components are ordinary. Weights denote learned-mixture masses before the 0.1 global uniform floor. No target filter, truncation, wall conditioning, native prior added to physical target, or bath refitting.',
        inference='Geometry-derived guide using old discovery rows, native-informed legacy atlas and prescribed catalogue sites. Independent fresh importance populations are required. Classify their physical poses with the frozen entry definition, never with selected component labels.',
        sources={str(p):sha(p)for p in sorted(set(source_paths))},classifier_definition_sha256=sha(DEFINITION),
        physical_config_sha256=sha(REFERENCE/'config.json'),shape_sha256=SHAPE_SHA)
    require(len(coverage['base_model']['weights'])==178 and len(validation['scalar_factor_certificates'])==178,'Unexpected component count')
    write(out/'manifest.json',manifest)
    write(out/'freeze.json',{str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()})
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'runs/mobile-wall-proposal-preparation-20260921')
    args=parser.parse_args();result=prepare(args.out)
    print(json.dumps(dict(directory=str(args.out.resolve()),arms=result['arms'],selection=result['selection']),indent=2))
