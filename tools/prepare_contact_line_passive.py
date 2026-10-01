#!/usr/bin/env python3
"""Freeze a proposal-only contact-line pilot and saved-pair axis diagnostics.

No new Poisson clouds, classifier replay, physical normalizer, or jobs are run.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil

for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np
from scipy.spatial.transform import Rotation

from diagnose_fitted_kernel_shear import read, sha, require, bind, write_new, load_rows
from prepare_contact_bank_guides import log_proposal
from run_protected_guide_validation import declared_seeds

SEEDS=[610041001+1009*i for i in range(8)]
COVERAGE_SEED=610049001
COVERAGE_SELECTION_SEED=610040701
WIDTHS=[.02,.1,.5]
GUIDE_SHA='a00a4470d898e06f66b78dcb7f1c1b42d7c4dc76f9a111dff9ce6268e09a96c0'
REGION_SHA='924648f4d9db473045397c300b3b4af7ccde8cfda1b1703a6899239ec12e2f02'
SHAPE_SHA='c7442034e7ffa4627b67c57a81117f0e9bc2d389ecf956a83dac2a5e915554c9'


def intersect(first, second):
    return [[max(a,c),min(b,d)] for a,b in first for c,d in second if max(a,c)<min(b,d)]


def remove(first, second):
    result=list(first)
    for a,b in second:
        updated=[]
        for c,d in result:
            if b<=c or a>=d: updated.append([c,d])
            else:
                if c<a: updated.append([c,a])
                if b<d: updated.append([b,d])
        result=updated
    return result


def sphere_interval(delta,direction,radius):
    aa=float(direction@direction); bb=float(delta@direction)
    cc=float(delta@delta-radius*radius); discriminant=bb*bb-aa*cc
    if discriminant<=0: return []
    root=math.sqrt(discriminant)
    return [[(-bb-root)/aa,(-bb+root)/aa]]


def prepare(out,repository):
    require(not out.exists(),'Fresh passive preparation directory required')
    pilot=repository/'runs/contact-tail-pilot-20261001'
    archive=repository/'runs/contact-tail-pilot-review-v2-20261001/geometry-inspection-v2'
    bindings={}
    inspection=read(bind(archive/'inspection.json',bindings))
    for name,digest in read(archive/'manifest.json')['files'].items():
        require(sha(archive/name)==digest,'Inspection archive changed')
    guide_path=bind(repository/'runs/contact-tail-expansion-20261001/expanded-guide.json',bindings,GUIDE_SHA)
    region_path=bind(pilot/'baseline/provenance/region.json',bindings,REGION_SHA)
    shape_path=bind(pilot/'baseline/provenance/shape.json',bindings,SHAPE_SHA)
    config_path=bind(pilot/'baseline/provenance/config.json',bindings)
    region,shape,config,guide=map(read,(region_path,shape_path,config_path,guide_path))
    require(config['fixed_poses']==region['physical_fixed_neighbors'],'Scaffold mismatch')
    require(config['depletant_radius']==region['depletant_radius']==1.5 and
            config['reservoir_density']==region['activity']==.035,'Physical conditions changed')
    out.mkdir(parents=True);(out/'common').mkdir();(out/'guides').mkdir()
    shutil.copy2(region_path,out/'common/region.json');shutil.copy2(shape_path,out/'common/shape.json')
    config['shape']=str(out/'common/shape.json');write_new(out/'common/config.json',config)
    shutil.copy2(guide_path,out/'common/original92.json')
    for name,beta in [('baseline92',0.),('line92',.5)]:
        candidate=copy.deepcopy(guide);candidate['schema']='defensive-contact-line-guide-v1'
        candidate.update(raw_translation_axes=[0],contact_widths_A=WIDTHS,contact_neighbor_indices=[0,1],
                         conditional_probability=beta,minimum_conditional_mass=1e-12)
        write_new(out/'guides'/f'{name}.json',candidate)

    chart=region['gaussian_chart'];lower=np.linalg.cholesky(chart['covariances'][0]);inverse=np.linalg.inv(lower)
    mean=np.array(chart['means'][0]);anchor=chart['anchors'][0];ap=np.array(anchor['position']);ar=np.array(anchor['rotation'])
    ell=chart['angular_length'];fixed=region['fixed_neighbor']
    rotation=lambda pose:Rotation.from_quat(np.roll(pose['orientation'],-1)).as_matrix()
    body=rotation(fixed);bp=np.array(fixed['position'])
    centers=np.array([x['center'] for x in shape['atoms']]);radii=np.array([x['radius'] for x in shape['atoms']])
    neighbors=[(rotation(p),np.array(p['position'])) for p in region['physical_fixed_neighbors']]
    rows=[];probes=[];gap_error=0.
    for saved in inspection['rows']:
        u=np.array(saved['u']);x=mean+lower@u;c=x[3:]/ell;q=np.r_[c,1.];q/=np.linalg.norm(q)
        mr=body@Rotation.from_quat(q).as_matrix()@ar;mp=bp+body@(ap+x[:3])
        pairs=[]
        for entry in saved['saved_classification']['contact']['anchors']:
            i,j=entry['moving_fixed_atom_indices'];nr,np_=neighbors[entry['anchor_index']]
            delta=mr@centers[i]+mp-(nr@centers[j]+np_);radius=radii[i]+radii[j];gap=np.linalg.norm(delta)-radius
            gap_error=max(gap_error,abs(gap-entry['minimum_surface_gap_A']))
            pairs.append(dict(anchor_index=entry['anchor_index'],atom_indices=[i,j],delta_world=delta.tolist(),
                core_radius_sum=float(radius),gap_A=float(gap),raw_translation_gap_gradient=((delta/np.linalg.norm(delta))@body).tolist()))
        axes={}
        for axis in range(3):
            world=body[:,axis];du=inverse[:,axis]
            shell=sphere_interval(u,du,float(region['mahalanobis_radius']))
            conditions={}
            for width in WIDTHS+[3.]:
                permitted=shell
                for pair in pairs:
                    delta=np.array(pair['delta_world']);radius=pair['core_radius_sum']
                    band=remove(sphere_interval(delta,world,radius+width),sphere_interval(delta,world,radius))
                    permitted=intersect(permitted,band)
                conditions[str(width)]=dict(intervals_delta_raw_translation_A=permitted,
                    total_length_A=sum(b-a for a,b in permitted),positive_measure=bool(permitted))
            axes[str(axis)]=conditions
        identifier=f"{saved['arm']}/{saved['id']}/{saved['draw']}"
        probes.append(dict(id=identifier,latent=u.tolist()))
        worldq=Rotation.from_matrix(mr).as_quat()
        rows.append(dict(id=identifier,source=saved,raw_chart=x.tolist(),
            world_pose=dict(position=mp.tolist(),orientation=np.roll(worldq,1).tolist()),
            original92_log_q=float(log_proposal(u[None,:],guide)[0]),
            log_physical_J=float(np.log(np.diag(lower)).sum()-3*np.log(ell)-2*np.log(np.pi)-2*np.log1p(c@c)),
            saved_pair_geometry=pairs,axes=axes))
    require(len(rows)==78 and gap_error<1e-9,'Saved geometry reconstruction failed')
    require(max(abs(r['log_physical_J']-r['source']['log_J']) for r in rows)<1e-10,'Physical Jacobian reconstruction failed')
    with (out/'probes.jsonl').open('w') as stream:
        for row in probes:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    summaries={}
    for arm in ('baseline','expanded'):
        for split,ids in [('all',('r00','r01','r02','r03')),('earlier',('r00','r01')),('later',('r02','r03'))]:
            subset=[r for r in rows if r['source']['arm']==arm and r['source']['id'] in ids]
            weights=np.array([r['source']['fraction_of_observed_orthant55_mass'] for r in subset]);weights/=weights.sum()
            summaries[f'{arm}:{split}']={str(axis):{str(width):dict(rows=len(subset),
                positive_rows=sum(r['axes'][str(axis)][str(width)]['positive_measure'] for r in subset),
                observed_weighted_positive_fraction=float(weights@np.array([r['axes'][str(axis)][str(width)]['positive_measure'] for r in subset])),
                observed_weighted_interval_length_A=float(weights@np.array([r['axes'][str(axis)][str(width)]['total_length_A'] for r in subset])))
                for width in WIDTHS+[3.]} for axis in range(3)}
    write_new(out/'axis-diagnostics.json',dict(rows=rows,summaries=summaries,maximum_reconstructed_gap_error_A=gap_error,
        scope='Exact intervals for the TWO SAVED atom pairs only, intersected with R4-line; all other core pairs omitted. These are neither lower nor upper bounds on full-union admissible intervals. Not a physical mass estimate.',
        split='Both earlier and later pilot rows inspected historically. These are descriptive subsets, not pristine validation or fit data.',
        choice='Fixed raw x chosen from retrospective geometry; fresh proposal-only streams below provide independent cost/support observations. No new fit.'))
    # Separate breadth control: uniform within predeclared classes, never selected by weight.
    old_root=repository/'runs/protected-guide-validation-20260923/comparison'
    old=read(bind(old_root/'analysis.json',bindings,'471f61cacf7253bf5746d8fc04765ea71ddfacfb07e205e12e50bf43742737dd'))
    rng=np.random.default_rng(COVERAGE_SELECTION_SEED);coverage=[];coverage_sources=[]
    for arm in ('bank','protected'):
        for source in sorted(old['arms'][arm]['populations'],key=lambda r:r['id']):
            if source['id'] not in ('r02','r03'):continue
            record=dict(source,records=str((old_root/source['records']).resolve()))
            arrays=load_rows(record);bind(record['records'],bindings,source['records_sha256'])
            residual=-1 if np.count_nonzero(arrays['class_id']==-1)>=8 else -2
            groups=[('native_R5',0),('native_complement',1),('competing',2),
                    ('unbound_or_residual' if residual==-1 else 'invalid',residual)]
            for label,identity in groups:
                eligible=np.flatnonzero(arrays['class_id']==identity);require(len(eligible)>=8,'Missing declared breadth stratum')
                selected=np.sort(rng.choice(eligible,8,replace=False))
                for index in selected:
                    coverage.append(dict(id=f"coverage/{arm}/{source['id']}/{int(index)}",latent=arrays['u'][index].tolist(),
                        source_arm=arm,source_population=source['id'],source_seed=source['seed'],source_attempted_draws=source['samples'],
                        source_records=record['records'],source_records_sha256=source['records_sha256'],source_draw=int(index),
                        coverage_class=label,class_id=int(identity),class_population_count=len(eligible),
                        original_log_q=float(arrays['log_q'][index]),original_log_J=float(arrays['log_physical_jacobian'][index]),
                        original92_log_q=float(log_proposal(arrays['u'][index][None,:],guide)[0])))
            coverage_sources.append(dict(arm=arm,id=source['id'],records_sha256=source['records_sha256'],
                residual_class='unbound_or_residual' if residual==-1 else 'invalid'))
    require(len(coverage)==128,'Breadth allocation differs')
    with (out/'coverage-probes.jsonl').open('w') as stream:
        for row in coverage:stream.write(json.dumps(dict(id=row['id'],latent=row['latent']),separators=(',',':'))+'\n')
    write_new(out/'coverage-selection.json',dict(selection_seed=COVERAGE_SELECTION_SEED,rows=coverage,
        source_populations=coverage_sources,uniform_within_class=True,rows_per_class_per_population=8,
        scope='Previously inspected original holdouts; random breadth diagnostic, no physical mass inference or weight filtering'))
    previous=set();inventory={}
    for path in sorted((repository/'runs').rglob('protocol.json')):
        previous.update(declared_seeds(read(path)));inventory[str(path.resolve())]=sha(path)
    previous.update(r['source']['seed'] for r in rows)
    require(not previous.intersection(SEEDS+[COVERAGE_SEED]),'Fresh passive seed collision')
    executable=repository/'target-validation-line-guide/release/contact-line-guide-audit';jobs=[]
    for arm in ('baseline92','line92'):
        for index in range(4):
            seed=SEEDS[len(jobs)];cmd=[str(executable),'--config',str(out/'common/config.json'),
                '--region',str(out/'common/region.json'),'--importance-guide',str(out/'guides'/f'{arm}.json'),
                '--out',str(out/'audit'/arm/f'r{index:02}'),'--samples','64','--seed',str(seed)]
            if index==0:cmd+=['--probes',str(out/'probes.jsonl')]
            jobs.append(dict(arm=arm,id=f'r{index:02}',seed=seed,fresh_proposal_draws=64,
                archived_probe_queries=78 if index==0 else 0,command=cmd))
    coverage_job=dict(arm='line92',id='coverage',seed=COVERAGE_SEED,fresh_proposal_draws=0,archived_probe_queries=128,
        probe_file='coverage-probes.jsonl',command=[str(executable),'--config',str(out/'common/config.json'),
            '--region',str(out/'common/region.json'),'--importance-guide',str(out/'guides/line92.json'),
            '--out',str(out/'audit/line92/coverage'),'--samples','0','--seed',str(COVERAGE_SEED),
            '--probes',str(out/'coverage-probes.jsonl')])
    write_new(out/'plan.json',dict(schema='contact-line-proposal-only-preparation-v1',source_sha256=bindings,
        jobs=jobs,coverage_job=coverage_job,physical_jobs_launched=0,new_Poisson_clouds=0,proposal_jobs_launched=0,
        fresh_draws=512,archived_probe_queries=156,additional_coverage_queries=128,total_output_pose_rows=796,maximum_CPU_workers=1,
        widths_A=WIDTHS,axes=[0],conditional_probability=.5,minimum_conditional_mass=1e-12,
        support='50% uniform R4; untruncated original Gaussian part and normalized one-coordinate conditioning; no outer5 redraw; old-q retention at least0.5',
        fallback='Empty geometry or each component conditional mass at or below fixed 1e-12: use its original conditional Normal; identical component/axis/width decision in full q evaluation',
        correction='Full mixture over both conditional/unconditional branches, all widths and all92 Gaussian components; exact raw-chart-to-u transformation and physical J',
        geometry='R4 line intersect both scaffold inflated unions (ri+rj+width), subtract ALL core intervals; widths are near-core gaps, not native2A registration bands',
        provenance='No fitted parameters. Axis choice used previously inspected saved geometry; no prospective claim for archived poses. Eight distinct fresh streams have never contributed to design.',
        seed_inventory=inventory,keep_all_draws=True,optional_stopping=False,retries=False,larger_autoextension=False,
        independent_audit=['reconstruct every component conditional mean/variance, stable interval mass and fallback',
            'full q from saved per-width interval endpoints; qline >= .5q92 within stated floating tolerance',
            'raw coordinate, physical pose and Haar-Jacobian reconstruction',
            'geometry interval midpoint/endpoints checked against primitive independent reference tests',
            'baseline conditional_probability0 agrees with original92 full density'],
        diagnostics=['hard-valid and R4 retention with all attempted draws as denominator; between-stream variability',
            'joint-contact and per-width retention, unconditional uniform/Gaussian branch counts, empty/low-mass fallback counts',
            'geometry node/leaf counts; construction/draw/full-density/audit CPU separately and total',
            'saved-cloud second-moment diagnostics separately by original source; no newly estimated physical masses'],
        gates={'correctness':'all density/Jacobian/interval/reference checks pass and everyattempt retained; otherwise software failure',
            'proposal_utility':'report valid joint-contact attempts per total CPU with population variation; no accepted-move or mass-stability inference',
            'physical_campaign':'not authorized by passive counts alone; contact-weight and unseen-mode gates remain unchanged'},
        prerequisites=['bind exact compiled binary/source closure and independent density auditor before diagnostic execution'],
        scope='Passive sampling-law and geometry/cost study, no depletion estimators, physical integrals, equilibrium conclusion or assembly launch'))
    shutil.copy2(__file__,out/'source.py')
    write_new(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(dict(out=str(out),prepared=True,new_Poisson_clouds=0,
        maximum_reconstructed_gap_error_A=gap_error,planned_fresh_draws=512,planned_archived_queries=156,additional_coverage_queries=128),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();prepare(args.out.resolve(),args.repository.resolve())
