#!/usr/bin/env python3
"""Freeze a 99:1 atlas extension from the existing geometric contact chart.

Preparation and fixed-candidate diagnostics only. This does not launch a chain,
refit a covariance, or interpret trajectory occupancy as mixture probability.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import numpy as np
from prepare_smc_normalizer_atlas import ROOT,Density,arrays,read,relative_poses,sha,write
from analyze_trapped_docking_moves import classify,describe


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',type=Path,default=ROOT/'runs/posterior-docking-mis-5000')
    ap.add_argument('--region',type=Path,default=ROOT/'runs/outside-r8-local-region-20260920/site0/region-r3.json')
    ap.add_argument('--out',type=Path,default=ROOT/'runs/outside-r8-atlas-extension-20260920/site0')
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing existing frozen output directory')
    parent_path=args.campaign/'provenance/model.json'
    source=args.campaign/'runs/site0-m1-deep-r01-c09'
    cfg=read(source/'config.json');parent=read(parent_path);region=read(args.region)
    chart=region['gaussian_chart']
    assert len(parent['weights'])==116 and chart['weights']==[1.]
    for key in ('schema','coordinate_convention','angular_length','shape_sha256'):
        assert parent[key]==chart[key]
    assert chart['coordinate_convention']=='anchor-body-relative'
    assert cfg['fixed_poses']==[region['fixed_neighbor']]
    assert cfg['capture_center']==region['capture_center'] and cfg['capture_radius']==region['capture_radius']
    assert cfg['reservoir_density']==region['activity'] and cfg['depletant_radius']==region['depletant_radius']
    for key in ('native_poses','rigid_members','member_error_scale','angle_error_scale_deg'):
        assert cfg['metadata'][key]==region['physical_metric'][key]
    # This is exact reuse, not a fit or empirical-occupancy reweighting.
    model=copy.deepcopy(parent)
    for key in ('anchors','means','covariances'):model[key].append(copy.deepcopy(chart[key][0]))
    model['weights']=[.99*w for w in parent['weights']]+[.01]
    assert abs(sum(model['weights'])-1)<1e-12 and min(model['weights'])>0
    for key in ('anchors','means','covariances'):assert model[key][:-1]==parent[key]
    eigenvalues=np.linalg.eigvalsh(chart['covariances'][0]);assert min(eigenvalues)>0
    old_region_path=args.region.parent/'provenance/old-region.json'
    old=read(old_region_path)
    assert sha(old_region_path)==region['old_region_sha256']
    old_density,new_density=Density(old['gaussian_chart']),Density(chart)
    parent_density,extended_density=Density(parent),Density(model)
    frames=[json.loads(line) for line in (source/'trajectory.jsonl').open()]
    rows=[json.loads(line) for line in (source/'moves.jsonl').open()]
    q,ar,br,labels=classify([r['pose'] for r in frames],old['fixed_neighbor'],old['physical_metric'],old_density,new_density)
    relative=relative_poses([r['pose'] for r in frames],old['fixed_neighbor'])
    log_parent=parent_density.evaluate(relative)[0]
    log_extended=extended_density.evaluate(relative)[0]
    log_added=new_density.evaluate(relative)[0]
    mixture_identity_error=float(np.max(np.abs(log_extended-np.logaddexp(np.log(.99)+log_parent,np.log(.01)+log_added))))
    responsibility=np.exp(np.log(.01)+log_added-log_extended)
    assert mixture_identity_error<1e-10
    record_groups={}
    for name,mask in [('all',np.ones(len(frames),dtype=bool)),('old_R8',labels=='old_R8'),
                     ('new_B5',np.isin(labels,['new_B3','new_B5_shell']))]:
        record_groups[name]=dict(frames=int(mask.sum()),parent_log_G=describe(log_parent[mask]),
            extended_log_G=describe(log_extended[mask]),log_G_increase=describe((log_extended-log_parent)[mask]),
            added_component_source_responsibility=describe(responsibility[mask]))
    candidates=[r for r in rows if r['branch']=='involution' and r['hard_valid']]
    oq,oa,ob,ol=classify([r['old_pose'] for r in candidates],old['fixed_neighbor'],old['physical_metric'],old_density,new_density)
    nq,na,nb,nl=classify([r['proposed_pose'] for r in candidates],old['fixed_neighbor'],old['physical_metric'],old_density,new_density)
    selected=np.isin(ol,['new_B3','new_B5_shell'])&(nl=='old_R8')
    candidates=[r for r,yes in zip(candidates,selected) if yes]
    assert len(candidates)==261
    oldposes=relative_poses([r['old_pose'] for r in candidates],old['fixed_neighbor'])
    newposes=relative_poses([r['proposed_pose'] for r in candidates],old['fixed_neighbor'])
    go=parent_density.evaluate(oldposes)[0];gn=parent_density.evaluate(newposes)[0]
    eo=extended_density.evaluate(oldposes)[0];en=extended_density.evaluate(newposes)[0]
    old_correction=go-gn;new_correction=eo-en
    gate=np.asarray([r['gate']['log_weight'] for r in candidates])
    correction_error=float(np.max(np.abs(old_correction-np.asarray([r['proposal']['log_reverse_forward'] for r in candidates]))))
    assert correction_error<1e-9
    candidate_groups={}
    for name,mask in [('all',np.ones(len(candidates),dtype=bool)),('target_113',np.asarray([r['proposal']['trace']['target']==113 for r in candidates])),
                     ('final_dwell_cycle_ge_2770',np.asarray([r['cycle']>=2770 for r in candidates]))]:
        candidate_groups[name]=dict(count=int(mask.sum()),old_correction=describe(old_correction[mask]),
            new_correction=describe(new_correction[mask]),correction_increase=describe((new_correction-old_correction)[mask]),
            same_recorded_gate_log_factor=describe(gate[mask]),
            old_sum_conditional_acceptance=float(np.exp(np.minimum(0,gate[mask]+old_correction[mask])).sum()),
            new_sum_conditional_acceptance=float(np.exp(np.minimum(0,gate[mask]+new_correction[mask])).sum()),
            actual_old_acceptances=sum(r['accepted'] for r,yes in zip(candidates,mask) if yes))
    # An independent change of fixed-neighbor coordinate frame must preserve
    # normalized physical density: translation and proper rotation have unit Jacobian.
    ft,_,fr=arrays([old['fixed_neighbor']]);rotation=fr[0]
    block=np.zeros((6,6));block[:3,:3]=block[3:,3:]=rotation
    laboratory=copy.deepcopy(model)
    laboratory['coordinate_convention']='laboratory'
    laboratory['anchors']=[dict(position=(rotation@np.asarray(a['position'])+ft[0]).tolist(),
        rotation=(rotation@np.asarray(a['rotation'])).tolist()) for a in model['anchors']]
    laboratory['means']=(np.asarray(model['means'])@block.T).tolist()
    laboratory['covariances']=(block@np.asarray(model['covariances'])@block.T).tolist()
    chosen=np.unique(np.linspace(0,len(frames)-1,256,dtype=int))
    lab_log=Density(laboratory).evaluate([frames[i]['pose'] for i in chosen])[0]
    frame_error=float(np.max(np.abs(lab_log-log_extended[chosen])))
    assert frame_error<1e-8
    cohort_path=args.campaign/'assessment/outside-R8-cohort.json';cohort=read(cohort_path)
    assert sha(cohort_path)==region['cohort_sha256']
    starts=[]
    for i in (8,24):
        selected_pose=cohort['poses'][i]
        assert frames[selected_pose['cycle']]['pose']==selected_pose['pose']
        assert selected_pose['contact_descriptor']['hard_valid']
        assert labels[selected_pose['cycle']] in ('new_B3','new_B5_shell')
        starts.append(dict(cohort_index=i,source_cycle=selected_pose['cycle'],pose=selected_pose['pose'],
            original_q=selected_pose['q'],new_latent_radius=float(br[selected_pose['cycle']]),
            old_latent_radius=float(ar[selected_pose['cycle']])))
    args.out.mkdir(parents=True);provenance=args.out/'provenance';provenance.mkdir()
    inputs={'parent-model.json':parent_path,'region.json':args.region,'old-region.json':old_region_path,
            'source-config.json':source/'config.json','cohort.json':cohort_path,
            'prepare_outside_r8_atlas_extension.py':Path(__file__),
            'prepare_smc_normalizer_atlas.py':Path(__file__).with_name('prepare_smc_normalizer_atlas.py'),
            'analyze_trapped_docking_moves.py':Path(__file__).with_name('analyze_trapped_docking_moves.py')}
    for name,path in inputs.items():shutil.copyfile(path,provenance/name)
    write(args.out/'model.json',model)
    with (args.out/'fixed-candidates.jsonl').open('w') as out:
        for i,r in enumerate(candidates):
            out.write(json.dumps(dict(cycle=r['cycle'],attempt=r['attempt'],source=r['proposal']['trace']['source'],
                target=r['proposal']['trace']['target'],old_pose=r['old_pose'],proposed_pose=r['proposed_pose'],
                accepted_old=r['accepted'],gate_log=float(gate[i]),old_log_proposal_ratio=float(old_correction[i]),
                extended_log_proposal_ratio=float(new_correction[i])),allow_nan=False)+'\n')
    report=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),status='prepared_not_launched',
        source_hashes={name:sha(path) for name,path in inputs.items()},
        trajectory_sha256=sha(source/'trajectory.jsonl'),moves_sha256=sha(source/'moves.jsonl'),
        model_sha256=sha(args.out/'model.json'),parent_components=116,new_components=117,
        rule='G_new=0.99 G_parent+0.01 g_existing_geometric_chart; full untruncated Gaussians. The uniform branch stays separate and unchanged.',
        normalization='Every component uses its existing full-rank normalized 6D Gaussian and exact Cayley/Haar Jacobian. Positive weights sum to one. No capture/hard rejection renormalization is introduced. The old atlas support and uniform defense remain.',
        no_fit='Mean/covariance/anchor are copied exactly from the already frozen outside-R8 geometric cohort region. Weight .01 is specified as a proposal-control choice, not estimated from occupancy or physical mass.',
        covariance_eigenvalues=eigenvalues.tolist(),covariance_condition=float(eigenvalues[-1]/eigenvalues[0]),
        mixture_identity_max_log_error=mixture_identity_error,lab_body_density_max_log_error=frame_error,
        recorded_correction_max_log_error=correction_error,trajectory_density=record_groups,
        fixed_candidate_diagnostics=candidate_groups,
        caveat='These are the same recorded candidates and the same recorded Poisson factors, with only full-G corrections reevaluated. New source responsibilities would change actual candidate generation; acceptance sums are not predicted fresh-chain rates or new accepted events.')
    write(args.out/'report.json',report)
    plan=dict(status='proposed_not_launched',purpose='Test filling a known proposal-coverage hole with all physical and local kernels unchanged.',
        models={'parent':str((provenance/'parent-model.json').resolve()),'extension':str((args.out/'model.json').resolve())},
        model_sha256={'parent':sha(provenance/'parent-model.json'),'extension':sha(args.out/'model.json')},
        starts=starts,start_rule='Previously frozen chronological cohort entries8 and24. No new covariance or weight optimization.',
        correlations=[0.,.9],cycles=2000,jobs=8,workers=4,
        seeds_by_start=[98641010,98642019],seed_pairing='Same seed for all four parent/extension and c=0/.9 comparisons at each identical start; paired random streams do not imply identical later proposals.',
        physical_configuration=str((provenance/'source-config.json').resolve()),
        unchanged=['hard shape','capture ball','fixed neighbor','activity0.035','depletant radius1.5','lambda ratio16','endpoint envelope2047','two local attempts per cycle','local move sizes','uniform probability0.1'],
        prespecified_diagnostics=['First exit from B5, and first entry to oldR8, with censoring at2000','B5↔oldR8 crossings and full residence','Hard-valid oldR8 proposals and acceptance factors','Source/target component concentration','Every retained frame; contact observable at fixed cadence; CPU separated from observer work'],
        limitations='This is a deliberately initialized escape control, not equilibrium kinetics, unbiased basin discovery, template-free assembly, or a sufficient rate-precision study.',
        estimated_budget='Historical8x5000 benchmark cost about770 sampler CPU seconds;8x2000 extrapolates to~310, with one added Gaussian. Actual geometry/contact costs may differ. No chain is launched by this preparer.')
    write(args.out/'benchmark-plan.json',plan)
    print(json.dumps(dict(model=str(args.out/'model.json'),model_sha256=report['model_sha256'],checks={k:report[k] for k in ('mixture_identity_max_log_error','lab_body_density_max_log_error','recorded_correction_max_log_error')},
        density_B5=record_groups['new_B5'],fixed_candidate_diagnostics=candidate_groups,plan=str(args.out/'benchmark-plan.json')),indent=2))


if __name__=='__main__':main()
