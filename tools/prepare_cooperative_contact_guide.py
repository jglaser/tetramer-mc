#!/usr/bin/env python3
"""Archive saved cooperative contact geometry and freeze a normalized R4 guide.

No physical sampling, geometry/classifier replay, or holdout-dependent fitting.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import math
import os
from pathlib import Path
import shutil
import sys
import time

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree

from diagnose_fitted_kernel_shear import (read, sha, require, write_new, bind,
    load_rows, group_masks, evaluate_group, combine_group)
from prepare_contact_tail_expansion import (PINS, expand_guide, allocation_control, validate_guide)
from prepare_contact_bank_guides import latent_geometric_floor, nearest_distinct, capped_weights, log_proposal
from prepare_native_confirmation_atlas import weighted_fit
from prepare_shoulder_docking_benchmark import local_dependencies
from run_protected_guide_validation import declared_seeds

PILOT_ANALYSIS = '4dc2a930c66ad9b2e5eb9863c1f20da5a7b02e9fe4f52d1b849ea1b59974bdb0'
GUIDE92 = 'a00a4470d898e06f66b78dcb7f1c1b42d7c4dc76f9a111dff9ce6268e09a96c0'
SEEDS = [610031001+1009*i for i in range(8)]
FIT = dict(component_count=4, gaussian_fraction=.1, total_proposal_fraction=.05,
    neighbors=128, weight_cap=1/16, region_class_id=2, orthant=55,
    training='original bank/protected r00/r01 only; fresh pilot excluded',
    eligibility='hard valid, no native entry, orthant55, exclusion contacts with both scaffold anchors',
    anchor='one per training population: greatest paired physical second moment under frozen92',
    covariance='128 nearest distinct eligible poses, full 6D covariance about actual anchor plus existing floor',
    duplicates='exact duplicate u removed deterministically in source-population/draw order',
    geometric_floor='existing 0.05 A and 0.1 degree raw-chart floor; no bandwidth search')


def cooperative(label):
    if not label.get('applicable') or label['classification']['native_any']:
        return False
    contacts = label.get('contact', {}).get('anchors', [])
    return len(contacts) == 2 and {x['anchor_index'] for x in contacts} == {0, 1} and all(
        x['exclusion_contact'] for x in contacts)


def selected_labels(path, draws):
    import json
    wanted = set(map(int, draws)); result = {}
    if not wanted:
        return result
    with gzip.open(path, 'rt') as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                row = json.loads(line)
                require(row['draw'] == index, 'Label/draw identity changed')
                result[index] = row
            if len(result) == len(wanted):
                break
    require(set(result) == wanted, 'Missing saved labels')
    return result


def records(analysis, comparison, arms, role):
    result = []
    for arm in arms:
        for row in sorted(analysis['arms'][arm]['populations'], key=lambda x: x['id']):
            result.append(dict(arm=arm, id=row['id'], seed=row['seed'], samples=row['samples'],
                role=role(row['id']), records=str((comparison/row['records']).resolve()),
                records_sha256=row['records_sha256'], labels=str((comparison/row['labels']).resolve()),
                labels_sha256=row['labels_sha256']))
    return result


def inspect_pilot(pilot, destination):
    require(not destination.exists(), 'Inspection already exists')
    destination.mkdir()
    bindings = {}; analysis = read(bind(pilot/'comparison/analysis.json', bindings, PILOT_ANALYSIS))
    status = read(bind(pilot/'status.json', bindings))
    bind(pilot/'protocol.json', bindings, analysis['protocol_sha256'])
    require(status['complete'] and status['comparison_sha256'] == PILOT_ANALYSIS, 'Pilot not complete')
    populations = records(analysis, pilot/'comparison', ('baseline','expanded'), lambda _: 'diagnostic_only')
    saved = []; counts = []
    for source in populations:
        arrays = load_rows(source)
        bind(source['records'], bindings, source['records_sha256'])
        bind(source['labels'], bindings, source['labels_sha256'])
        mask = (arrays['class_id']==2) & (arrays['bin_orthant']==55) & np.isfinite(arrays['z'])
        indices = np.flatnonzero(mask); labels = selected_labels(source['labels'], indices)
        job = next(x for x in status['jobs'] if x['arm']==source['arm'] and x['id']==source['id'])
        raw = Path(job['directory'])/'samples.jsonl'
        bind(raw, bindings, job['output']['samples_sha256'])
        for i in indices:
            label = labels[int(i)]
            saved.append(dict(arm=source['arm'], id=source['id'], seed=source['seed'],
                draw=int(i), source_attempted_draws=source['samples'], raw_samples=str(raw),
                records_sha256=source['records_sha256'], labels_sha256=source['labels_sha256'],
                u=arrays['u'][i].tolist(), log_J=float(arrays['log_physical_jacobian'][i]),
                log_q=float(arrays['log_q'][i]), log_weight=float(arrays['z'][i]),
                paired_log_weights=arrays['pairs'][i].tolist(), component=int(arrays['component'][i]),
                branch=int(arrays['branch'][i]), both_scaffold_contacts=cooperative(label),
                saved_classification=label))
        counts.append(dict(arm=source['arm'], id=source['id'], nonzero_competing55=len(indices)))
    summaries = {}
    for arm in ('baseline','expanded'):
        rows = [r for r in saved if r['arm']==arm]; total = logsumexp([r['log_weight'] for r in rows])
        for row in rows:
            row['fraction_of_observed_orthant55_mass'] = float(math.exp(row['log_weight']-total))
        components = {}
        for component in sorted({r['component'] for r in rows}):
            subset = [r for r in rows if r['component']==component]
            components[str(component)] = dict(rows=len(subset), observed_mass_fraction=float(math.exp(
                logsumexp([r['log_weight'] for r in subset])-total)))
        summaries[arm] = dict(rows=len(rows), both_scaffold_contact_rows=sum(r['both_scaffold_contacts'] for r in rows),
            components=components, top_draws=[dict(id=r['id'],draw=r['draw'],component=r['component'],
            fraction=r['fraction_of_observed_orthant55_mass'],
            gaps_A=[c['minimum_surface_gap_A'] for c in r['saved_classification']['contact']['anchors']])
            for r in sorted(rows,key=lambda x:-x['log_weight'])[:5]])
    write_new(destination/'inspection.json', dict(schema='saved-cooperative-competing55-inspection-v1',
        source_sha256=bindings, populations=counts, rows=saved, summary=summaries,
        scope='Saved-row geometry only; no new geometry calls or physical mass estimates; competing55 is not native-complement55'))
    (destination/'source.py').write_bytes(Path(__file__).read_bytes())
    write_new(destination/'manifest.json',dict(files={p.name:sha(p) for p in destination.iterdir() if p.is_file()}))
    return populations


def prepare(out, repository, inspection):
    require(not out.exists(), 'Fresh preparation output required')
    comparison = repository/'runs/protected-guide-validation-20260923/comparison'
    bindings={}; old=read(bind(comparison/'analysis.json',bindings,PINS['analysis']))
    base_path=repository/'runs/contact-tail-expansion-20261001/expanded-guide.json'
    bind(base_path,bindings,GUIDE92)
    region_path=repository/'runs/contact-tail-expansion-20261001/region.json'
    bind(region_path,bindings,PINS['region'])
    datasets=records(old,comparison,('bank','protected'),lambda rid:'training' if rid in ('r00','r01') else 'retrospective_holdout')
    out.mkdir();(out/'source').mkdir()
    for name, source in local_dependencies([Path(__file__)]).items():
        bind(source,bindings);shutil.copy2(source,out/'source'/name)
    shutil.copy2(base_path,out/'baseline92.json');shutil.copy2(region_path,out/'region.json')
    write_new(out/'plan.json',dict(schema='cooperative-contact-guide-preparation-v1',fit=FIT,datasets=datasets,
        source_and_input_sha256=bindings,physical_samples_launched=0,pilot_fit_rows=0,
        holdout_caveat='Previously inspected original r02/r03, disjoint from this fit; retrospective only',
        scope='Native-label-aware integration design, not a blind assembly proposal',
        target=dict(shape_sha256=PINS['shape'],region_sha256=PINS['region'],
            native_definition_sha256=old['native_definition']['definition_sha256'],radius_A=1.5,activity_A_minus3=.035),
        density='Normalized mixture on R6: uniform R4 plus untruncated Gaussians; never condition on hard validity, contacts or being inside R4',
        diagnostic_go_gate=dict(target_paired_M2_ratio_max=.5,geometry_beats_allocation_in_both_old_source_arms=True,
            target_M2_contribution_ESS_min=20,target_M2_largest_fraction_max=.2,
            comparison='per original source arm; pilot remains diagnostic only; failure means no recommended launch')))
    print('Declaration frozen; reading original training only',flush=True)
    training=[r for r in datasets if r['role']=='training']; pieces=[]; labels_by_population=[]; counts=[]
    start=time.monotonic()
    for pi,source in enumerate(training):
        arrays=load_rows(source);bind(source['records'],bindings,source['records_sha256'])
        bind(source['labels'],bindings,source['labels_sha256'])
        eligible=np.flatnonzero((arrays['class_id']==2)&(arrays['bin_orthant']==55)&np.isfinite(arrays['z']))
        labels=selected_labels(source['labels'],eligible)
        eligible=np.asarray([i for i in eligible if cooperative(labels[int(i)])],int)
        pieces.append({key:arrays[key][eligible] for key in ('u','z','pairs','log_q','draw')})
        pieces[-1]['population']=np.full(len(eligible),pi,int)
        labels_by_population.append(labels)
        counts.append(dict(arm=source['arm'],id=source['id'],source_attempts=source['samples'],
            competing55_rows=len(labels),both_scaffold_contact_rows=len(eligible)))
    data={key:np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    floor,rawfloor=latent_geometric_floor(read(out/'region.json')['gaussian_chart'])
    baseline=read(out/'baseline92.json');tree=cKDTree(data['u']);eligible=np.arange(len(data['u']))
    components=[];details=[]
    for pi,source in enumerate(training):
        local=np.flatnonzero(data['population']==pi)
        require(len(local)>0,'No eligible cooperative anchor in training population')
        score=data['pairs'][local].sum(axis=1)+data['log_q'][local]-log_proposal(data['u'][local],baseline)
        ai=int(local[np.argmax(score)]);anchor=data['u'][ai]
        neighbors=nearest_distinct(data,eligible,tree,anchor,FIT['neighbors'])
        ns=np.asarray([training[int(p)]['samples'] for p in data['population'][neighbors]])
        w,cap=capped_weights(data['z'][neighbors]-np.log(ns),FIT['weight_cap'])
        keep=w>0;fit=weighted_fit(data['u'][neighbors][keep],np.log(w[keep]),floor)
        offset=fit['mean']-anchor;cov=fit['covariance']+np.outer(offset,offset);cov=.5*(cov+cov.T)
        np.linalg.cholesky(cov)
        components.append(dict(weight=.25,mean=anchor.tolist(),covariance=cov.tolist()))
        draw=int(data['draw'][ai]); label=labels_by_population[pi][draw]
        details.append(dict(source=dict(arm=source['arm'],id=source['id'],seed=source['seed'],draw=draw),
            selected_paired_moment=float(score.max()),anchor=anchor.tolist(),saved_label=label,
            capped_fit_weights=cap,covariance_eigenvalues=np.linalg.eigvalsh(cov).tolist(),
            neighbors=[dict(arm=training[int(data['population'][i])]['arm'],
                id=training[int(data['population'][i])]['id'],draw=int(data['draw'][i]),weight=float(weight))
                for i,weight in zip(neighbors,w)]))
    candidate=expand_guide(baseline,components,.1)
    control,assignments=allocation_control(baseline,[c['mean'] for c in components],.1)
    checks={}
    for name,guide in [('candidate96',candidate),('allocation92',control)]:
        checks[name]=validate_guide(guide,baseline,[c['mean'] for c in components])
        rng=np.random.default_rng(610010001)
        probes=np.vstack([np.asarray([c['mean'] for c in components]),np.zeros((1,6)),
                          rng.normal(size=(256,6))*2,rng.normal(size=(64,6))*8])
        margin=float(np.min(log_proposal(probes,guide)-log_proposal(probes,baseline)-math.log(.9)))
        require(margin>=-1e-12,
            'Nine-tenths baseline lower bound fails')
        checks[name]['old_proposal_retention_lower_bound']=.9
        checks[name]['smallest_observed_log_bound_margin']=margin
        checks[name]['exact_second_moment_increase_bound']=1/.9
    write_new(out/'candidate96.json',candidate);write_new(out/'allocation-control92.json',control)
    write_new(out/'fit.json',dict(components=details,counts=counts,floor_u=floor.tolist(),floor_raw_chart=rawfloor.tolist(),
        training_attempts=sum(r['samples'] for r in training),fit_eligible_rows=len(data['u']),
        holdout_rows_read=0,pilot_rows_used=0,allocation_control_assignments=assignments,
        density_identity='q96 = .5 U_R4 + .45 G92 + .05 G4; q96 >= .9 q92 pointwise',
        geometry_source_bindings=bindings))
    write_new(out/'guide-validation.json',checks)
    write_new(out/'model-freeze.json',dict(heldout_rows_read=0,pilot_rows_used=0,
        files={name:sha(out/name) for name in ('plan.json','baseline92.json','region.json','candidate96.json',
            'allocation-control92.json','fit.json','guide-validation.json')}))
    print('Candidate and allocation-only control frozen; opening retrospective holdouts',flush=True)
    pilot=repository/'runs/contact-tail-pilot-20261001'
    pilot_datasets=inspect_pilot(pilot,inspection)
    reports=[]
    for source in [r for r in datasets if r['role']=='retrospective_holdout']+pilot_datasets:
        arrays=load_rows(source);valid=np.isfinite(arrays['z']);densities={}
        for name,guide in [('baseline',baseline),('candidate',candidate),('allocation',control)]:
            densities[name]=np.zeros(source['samples']);densities[name][valid]=log_proposal(arrays['u'][valid],guide)
        comparisons={name:{key:evaluate_group(arrays,densities['baseline'],densities[name],mask)
            for key,mask in group_masks(arrays)} for name in ('candidate','allocation')}
        reports.append(dict(arm=source['arm'],id=source['id'],role=source['role'],
            records=source['records'],records_sha256=source['records_sha256'],controls=comparisons))
        print('Scored saved population',source['role'],source['arm'],source['id'],flush=True)
    combined={}
    for role,arms in [('retrospective_holdout',('bank','protected')),('diagnostic_only',('baseline','expanded'))]:
        for arm in arms:
            rows=[r for r in reports if r['role']==role and r['arm']==arm]
            combined[f'{role}:{arm}']={name:{key:combine_group([r['controls'][name][key] for r in rows])
                for key in rows[0]['controls'][name]} for name in ('candidate','allocation')}
    gates=[];target='contact_no_native_entry:orthant:55'
    for arm in ('bank','protected'):
        controls=combined['retrospective_holdout:'+arm]
        c=controls['candidate'][target]['second_moments']['paired_physical']
        w=controls['allocation'][target]['second_moments']['paired_physical']
        checks=dict(target_M2_ratio=math.exp(c['log_warp_to_baseline_ratio'])<=.5,
            beats_allocation=c['log_warp_to_baseline_ratio']<w['log_warp_to_baseline_ratio'],
            contribution_ESS=c['warped']['contribution_ESS']>=20,
            largest_fraction=c['warped']['largest_fraction']<=.2)
        gates.append(dict(arm=arm,checks=checks,passed=all(checks.values())))
    write_new(out/'retrospective-analysis.json',dict(complete=True,populations=reports,combined=combined,
        launch_recommendation_diagnostics=gates,recommend_fresh_pilot=all(g['passed'] for g in gates),
        model_freeze_sha256=sha(out/'model-freeze.json'),physical_samples_launched=0,
        scope='Paired physical second moments under source importance distributions; not prospective speedups or new thermodynamic estimates'))
    previous=set();seed_inventory={}
    for path in sorted((repository/'runs').rglob('protocol.json')):
        previous.update(declared_seeds(read(path)));seed_inventory[str(path.resolve())]=sha(path)
    previous.update(r['seed'] for r in datasets+pilot_datasets)
    require(not previous.intersection(SEEDS),'Future seeds already used')
    original=read(pilot/'protocol.json');jobs=[]
    for arm,guide in [('baseline92','baseline92.json'),('candidate96','candidate96.json')]:
        for i in range(4):
            cmd=list(original['jobs'][0]['command']);seed=SEEDS[len(jobs)]
            changes={'--importance-guide':str(out/guide),'--out':str(out/'future-pilot'/arm/f'r{i:02}'),
                '--samples':'16384','--seed':str(seed),'--region':str(out/'region.json')}
            for flag,value in changes.items():cmd[cmd.index(flag)+1]=value
            jobs.append(dict(arm=arm,id=f'r{i:02}',seed=seed,samples=16384,command=cmd))
    write_new(out/'prospective-plan.json',dict(preparation_only=True,physical_jobs_launched=0,
        candidate_sha256=sha(out/'candidate96.json'),guide_pair=['baseline92','candidate96'],
        jobs=jobs,total_attempted_draws=131072,maximum_physical_jobs=2,maximum_overall_physical_jobs=8,
        maximum_total_workers=32,two_clouds_per_valid_pose=True,lambda_ratio=128,uniform_probability=.5,
        future_seed_inventory=seed_inventory,old_samples_pooled=False,retries=False,optional_extension=False,
        run_recommendation=all(g['passed'] for g in gates),
        prerequisites=['candidate Rust sphere/hard-only limits and independent output density reconstruction',
            'new source/input-authenticated failure-draining dispatcher; archived raw commands alone are not production approval'],
        retained_estimator='J*mean(W1,W2)/q; full mixture density; all invalid draws zero; original N denominator',
        gates={'pilot_utility':'target paired physical second moment improves >=2x; compare geometry and allocation-only archived controls',
            'regional_convergence':original['convergence'],
            'all_material_strata':'original radial/angular/orthant strata, including native55 and competing55/58/63; no relabeling or deleted bins',
            'assembly':'closed regardless of this small feasibility pilot; full-vessel, size/intensity controls and many-body finite-system evidence still required'},
        scope='Minimum new feasibility control only if retrospective support diagnostics justify it; not a promise that this allocation reaches thermodynamic convergence'))
    write_new(out/'completion.json',dict(complete=True,physical_samples_launched=0,elapsed_seconds=time.monotonic()-start,
        files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print('Prepared; no physical jobs launched',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inspection-out',type=Path,required=True)
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    prepare(args.out.resolve(),args.repository.resolve(),args.inspection_out.resolve())


if __name__=='__main__':
    main()
