#!/usr/bin/env python3
"""Freeze two additional geometric inner-shoulder references and union masks.

Preparation/probes only. Original guided maxima choose centers; no likelihood
fit, physical-weight draw, Gaussian reweighting or production launch occurs.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import copy
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from analyze_native_region_reference import native_q
from prepare_cayley_rms_cover import read,write,sha,require,uniform_draws,check_coordinates
from prepare_far_atlas_repeat import validate_command_options
from prepare_far_contact_candidates import member_distances
from prepare_intermediate_local_region import BINARY,BINARY_SHA
from prepare_peak_neighborhood import geometry_model,verify_member_geometry
from prepare_native_confirmation_atlas import AtomUnionAudit
from run_shoulder_mis_campaign import local_dependencies

ROOT=Path(__file__).resolve().parents[1]
DISCOVERY=ROOT/'runs/ab-inner-shoulder-local-candidates-20260921/analysis.json'
DISCOVERY_SHA='cb776845bd955b78d1e93cbbc5206c4f59c8d9d32350b911f8314ad6833c9881'
GUIDED=ROOT/'runs/ab-shoulder-confirmation-with-cover-comparison-20260920/comparison.json'
GUIDES=ROOT/'runs/ab-shoulder-guide-preparation-20260920'
DIRECT_PREP=ROOT/'runs/ab-inner-shoulder-peak-reference-preparation-20260921'
DIRECT_ANALYSIS=ROOT/'runs/ab-inner-shoulder-peak-reference-assessment-20260921/analysis.json'
DIRECT_ANALYSIS_SHA='a5b341daa73b8eabd2a9f9752c52055aeedfd0b96d26be5210699ba86999fb65'
WINDOW=dict(minimum=1.,maximum=1.1,lower_inclusive=False,upper_inclusive=False)
PHYSICAL=('metadata','fixed_poses','capture_center','capture_radius','depletant_radius','reservoir_density')
RADII=(.25,.5)
NAMES=('mixture','geometry')
SEED_BASES=((110001010,110101010),(111001010,111101010))
PROBE_BASES=((110201010,110301010),(111201010,111301010))


def union_assignment(radii):
    """Assign each point to one of three closed balls, in fixed priority."""
    require(len(radii)==3 and all(r>=0 and not math.isnan(r) for r in radii),'Need three nonnegative chart radii')
    return next((i for i,r in enumerate(radii) if r<=.5),None)


def selected_peaks(diagnostic,guided):
    require(diagnostic['complete'],'Discovery diagnostic incomplete')
    entries=diagnostic['comparison_peaks']
    require(len(entries)==2 and {e['source_campaign'] for e in entries}=={'mixture-confirmation','geometry-confirmation'},'Frozen peak sources changed')
    entries=sorted(entries,key=lambda e:('mixture-confirmation','geometry-confirmation').index(e['source_campaign']))
    result=[]
    for entry,identity in zip(entries,(('r03',30939,1.038236897099948),('r01',33280,1.0232210223101685))):
        name=entry['source_campaign'];row=entry['original_row'];source=guided['campaigns'][name]
        require(row==source['top_16'][0],'Stored maximum differs from original guide comparison')
        require((row['population'],row['draw'],row['q'])==identity,'Selected source identity/window changed')
        require(row['log_importance_weight']==max(r['log_importance_weight'] for r in source['top_16']),'Selected row is not the recorded maximum')
        result.append(copy.deepcopy(entry))
    return result


def original_record(entry,cfg):
    source=read(GUIDED)['campaigns'][entry['source_campaign']];root=Path(source['root'])
    master=read(root/'manifest.json');row=entry['original_row'];population=row['population']
    require(sha(root/'manifest.json')==source['manifest_sha256'],'Guided source manifest changed')
    require(sha(root/'assessment-streaming.json')==source['assessment_sha256'],'Original full-density audit changed')
    original_cfg=read(root/'provenance/config.json')
    require(all(original_cfg[k]==cfg[k] for k in PHYSICAL),'Original physical target changed')
    for name,digest in master['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,'Original source archive changed')
    job=next(j for j in master['jobs'] if Path(j['output']).name==population)
    sample=Path(job['output'])/'samples.jsonl';manifest=Path(job['output'])/'manifest.json'
    require(sha(sample)==source['sample_sha256'][str(sample)],'Selected original source rows changed')
    run=read(manifest);require(run['seed']==job['seed'],'Selected source seed changed')
    original=None
    with sample.open() as handle:
        for line in handle:
            current=json.loads(line)
            if current['draw']==row['draw']:original=current;break
    require(original is not None and dict(original,population=population)==row,'Original selected row differs')
    require('zero' not in original and original['log_importance_weight']==row['log_importance_weight'],'Selected physical contribution changed')
    return dict(source_campaign=entry['source_campaign'],population=population,seed=job['seed'],draw=row['draw'],
        pose=row['pose'],q=row['q'],original_log_importance_weight=row['log_importance_weight'],original_row=original,
        source_original_N=source['physical']['draws'],source_samples_path=str(sample),source_samples_sha256=sha(sample),
        source_manifest_path=str(manifest),source_manifest_sha256=sha(manifest),
        source_campaign_manifest_path=str(root/'manifest.json'),source_campaign_manifest_sha256=sha(root/'manifest.json'),
        source_config_path=str(root/'provenance/config.json'),source_config_sha256=sha(root/'provenance/config.json'),
        source_comparison_path=str(GUIDED),source_comparison_sha256=sha(GUIDED),
        source_full_density_audit_path=str(root/'assessment-streaming.json'),source_full_density_audit_sha256=source['assessment_sha256'])


def prepare(out):
    out=Path(out).resolve();require(not out.exists(),'Fresh preparation directory required')
    require(sha(DISCOVERY)==DISCOVERY_SHA and sha(DIRECT_ANALYSIS)==DIRECT_ANALYSIS_SHA,'Frozen discovery/direct calibration changed')
    diagnostic=read(DISCOVERY);guided=read(GUIDED);direct=read(DIRECT_ANALYSIS);direct_protocol=read(DIRECT_PREP/'protocol.json')
    require(direct['complete'] and direct['original_q_window']==direct_protocol['original_q_window']==WINDOW,'Prior direct local reference differs')
    for path,digest in diagnostic['source_sha256'].items():require(sha(path)==digest,'Discovery source changed')
    for path,digest in direct['source_sha256'].items():require(sha(path)==digest,'Completed direct calibration input changed')
    for name,digest in direct['archived_sha256'].items():require(sha(DIRECT_ANALYSIS.parent/'provenance'/name)==digest,'Completed direct calibration archive changed')
    cfg=read(GUIDES/'config.json');shape=Path(cfg['shape']);shape_hash=sha(shape);fixed=cfg['fixed_poses'][0]
    require({k:cfg[k] for k in PHYSICAL}==diagnostic['source_physical_signature']==direct_protocol['physical'],'AB model/bath/metric differs')
    require(cfg['capture_radius']==18. and cfg['depletant_radius']==1.5 and cfg['reservoir_density']==.035 and len(cfg['fixed_poses'])==2,'Wrong AB bath')
    require(sha(BINARY)==BINARY_SHA,'Reviewed latent executable changed')
    selected=[original_record(e,cfg) for e in selected_peaks(diagnostic,guided)]
    direct_model=read(DIRECT_PREP/'model.json');direct_peak=read(DIRECT_PREP/'selected-pose.json')
    prior_seeds={j['seed'] for h in direct_protocol['historical_campaigns'] for j in read(Path(h['root'])/'manifest.json')['jobs']}
    prior_seeds.update(j['seed'] for c in direct_protocol['campaigns'] for j in read(Path(c['output'])/'manifest.json')['jobs'])
    proposed_seeds=[base+1009*i for bases in SEED_BASES for base in bases for i in range(16)]
    probe_seeds=[base for bases in PROBE_BASES for base in bases]
    require(len(set(proposed_seeds+probe_seeds))==len(proposed_seeds+probe_seeds) and not set(proposed_seeds+probe_seeds)&prior_seeds,'Fresh streams overlap')
    archive=out/'provenance';archive.mkdir(parents=True)
    runner=ROOT/'tools/run_latent_region_campaign.py'
    auditor_files=[ROOT/'tools'/name for name in ('analyze_latent_region.py','analyze_latent_region_shells.py')]
    sources={'discovery.json':DISCOVERY,'guided-comparison.json':GUIDED,'direct-analysis.json':DIRECT_ANALYSIS,
        'direct-protocol.json':DIRECT_PREP/'protocol.json','direct-model.json':DIRECT_PREP/'model.json',
        'direct-selected-pose.json':DIRECT_PREP/'selected-pose.json','input-config.json':GUIDES/'config.json','shape.json':shape,
        **local_dependencies([Path(__file__),runner,*auditor_files])}
    for name,path in sources.items():shutil.copy2(path,archive/name)
    frozen_cfg=copy.deepcopy(cfg);frozen_cfg['shape']=str(archive/'shape.json');write(out/'config.json',frozen_cfg)
    atom=AtomUnionAudit(read(shape),cfg['fixed_poses']);centers=[];campaigns=[];commands={};model_geometry=[]
    for index,(name,record) in enumerate(zip(NAMES,selected)):
        model,geometry=geometry_model(cfg['metadata'],fixed,record['pose'],shape_hash,direct_model['angular_length'])
        # This construction uses rigid-member second moments only, no source
        # statistical weights, weighted covariance fit or likelihood update.
        gaps=atom.gaps(record['pose']);require(min(gaps)>=0 and abs(native_q(cfg['metadata'],record['pose'])-record['q'])<2e-8,'Selected AB geometry differs')
        geometry['maximum_rotation_angle_deg_by_radius']={str(r):math.degrees(2*math.atan(r/(2*math.sqrt(min(geometry['A_eigenvalues_A2']))))) for r in RADII}
        write(out/f'model-{name}.json',model);write(out/f'geometry-{name}.json',geometry);write(out/f'selected-{name}.json',record)
        centers.append(dict(name=name,selected_pose=str(out/f'selected-{name}.json'),selected_pose_sha256=sha(out/f'selected-{name}.json'),
            model=str(out/f'model-{name}.json'),model_sha256=sha(out/f'model-{name}.json'),geometry=str(out/f'geometry-{name}.json'),
            selection_identity=dict(population=record['population'],seed=record['seed'],draw=record['draw']),atomic_gaps_A=gaps))
        model_geometry.append((model,geometry))
        for ri,radius in enumerate(RADII):
            label=f'{radius:g}'.replace('.','p');base=SEED_BASES[index][ri]
            region=dict(fixed_neighbor=fixed,physical_fixed_neighbors=cfg['fixed_poses'],capture_center=cfg['capture_center'],capture_radius=18.,
                activity=.035,depletant_radius=1.5,physical_metric=cfg['metadata'],shape_sha256=shape_hash,gaussian_chart=model,
                minimum_original_q=1.,maximum_original_q=1.1,minimum_original_q_inclusive=False,maximum_original_q_inclusive=False,
                minimum_mahalanobis_radius=0.,mahalanobis_radius=radius,
                definition='Fixed geometric ball at an existing guided maximum, intersected with original strict1<q<1.1, capture and full AB hard/depletion interactions. No refit.')
            region_path=out/f'region-{name}-r{label}.json';write(region_path,region)
            output=ROOT/f'runs/ab-shoulder-guide-peak-reference-20260921/{name}/r{label}';require(not output.exists(),'Preserve existing production output')
            campaigns.append(dict(name=name,radius_A=radius,region=str(region_path),region_sha256=sha(region_path),output=str(output),
                populations=16,samples_per_population=16384,workers=8,seed_base=base,seeds=[base+1009*i for i in range(16)]))
            commands[f'{name}-r{label}']=[sys.executable,str(archive/runner.name),'--out',str(output),'--config',str(out/'config.json'),
                '--region',str(region_path),'--binary',str(BINARY),'--samples','16384','--replicates','16','--workers','8',
                '--seed',str(base),'--lambda-ratio','64','--cloud-replicates','2']
    help_text=subprocess.run([sys.executable,str(archive/runner.name),'--help'],capture_output=True,text=True,check=True).stdout
    for command in commands.values():validate_command_options(command,help_text)
    (archive/'runner-help.txt').write_text(help_text);write(out/'commands.json',commands)
    distances=member_distances([direct_peak['pose']]+[s['pose'] for s in selected],np.asarray([m['position'] for m in cfg['metadata']['rigid_members']]))
    # Membership always uses each frozen geometric chart, not fitted Gaussian
    # distances or these pairwise RMS diagnostics.
    allocation=dict(center_order=['direct','mixture','geometry'],closed_ball_radius_A=.5,
        memberships=['rho_direct<=.5','rho_mixture<=.5 AND rho_direct>.5','rho_geometry<=.5 AND rho_direct>.5 AND rho_mixture>.5'],
        direct_source=dict(analysis_path=str(DIRECT_ANALYSIS),analysis_sha256=DIRECT_ANALYSIS_SHA,
            field='independent_shell_sum',mask='ball0p5',pieces=direct['independent_shell_sum']['pieces'],
            rule='Previously frozen independent [0,.25]+(.25,.5] direct reference; keep both original denominators and summed independent variances.'),
        new_center_sources=[dict(name=name,assigned_region_index=i+1,shell_sources=[
            dict(campaign_radius_A=.25,own_shell=[0.,.25],lower_inclusive=True,upper_inclusive=True),
            dict(campaign_radius_A=.5,own_shell=[.25,.5],lower_inclusive=False,upper_inclusive=True)]) for i,name in enumerate(NAMES)],
        outside_union='rho_direct>.5 AND rho_mixture>.5 AND rho_geometry>.5 within original strict1<q<1.1, capture and AB hard target',
        complement_rule='Retain positive outside-union masks from each original historical inner-window stream with its full original N and same-row covariance. Do not combine historical complement with fresh local estimates into a stitched whole-window estimate.',
        sum_rule='Sum the three geometrically disjoint assigned-region means and independent variances only. Each new assigned region sums its own prespecified two disjoint shells after exclusion of earlier balls. Never sum overlapping balls or pool historical/fresh observations.',
        comparison_rule='Keep local whole-ball references and same-mask historical calibrations separate. Center selection is discovery, not independent evidence of basin weight; no zero observation or finite agreement bounds unseen tails.')
    protocol=dict(schema='AB-inner-shoulder-guided-peak-references-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        physical={k:cfg[k] for k in PHYSICAL},original_q_window=WINDOW,config_sha256=sha(out/'config.json'),shape_sha256=shape_hash,
        selection_source=dict(path=str(DISCOVERY),sha256=DISCOVERY_SHA),centers=centers,campaigns=campaigns,
        historical_campaigns=direct_protocol['historical_campaigns'],direct_reference_preparation=str(DIRECT_PREP),
        analysis_plan=allocation,center_pairwise_member_RMS_A=distances.tolist(),
        executable=str(BINARY),executable_sha256=BINARY_SHA,lambda_ratio=64.,cloud_replicates=2,
        samples_total=sum(c['populations']*c['samples_per_population'] for c in campaigns),maximum_simultaneous_workers=32,
        probe_count_per_radius=256,probe_seeds={name:list(PROBE_BASES[i]) for i,name in enumerate(NAMES)},
        commands_sha256=sha(out/'commands.json'),input_sha256={str(p):sha(p) for p in sources.values()},
        archived_sha256={p.name:sha(p) for p in archive.iterdir()},
        scope='Preparation and fixed geometric probes only. No new Gaussian fit or physical integration. Later priority-union calibration remains finite-region sampling, not a full shoulder, native-tail, far-tail, mixing or assembly result.')
    write(out/'protocol.json',protocol);write(out/'freeze.json',{p.name:sha(p) for p in out.glob('*.json')})
    reports=[]
    for ci,name in enumerate(NAMES):
        model,geometry=model_geometry[ci]
        for ri,radius in enumerate(RADII):
            seed=PROBE_BASES[ci][ri];poses,latent,x,logj=uniform_draws(model,fixed,radius,256,seed)
            checks=check_coordinates(model,fixed,poses,latent,x,logj)
            checks.update(verify_member_geometry(cfg['metadata'],fixed,selected[ci]['pose'],poses,latent,x,model,geometry,logj))
            path=out/f'geometry-probes-{name}-r{radius:g}.jsonl';valid=0
            with path.open('x') as handle:
                for draw,pose in enumerate(poses):
                    q=native_q(cfg['metadata'],pose);gaps=atom.gaps(pose);capture=bool(np.linalg.norm(np.asarray(pose['position'])-cfg['capture_center'])<=18.)
                    ok=bool(capture and min(gaps)>=0 and 1<q<1.1);valid+=ok
                    handle.write(json.dumps(dict(draw=draw,pose=pose,q=q,atomic_gaps_A=gaps,capture_valid=capture,valid=ok,log_jacobian=float(logj[draw])),allow_nan=False)+'\n')
            reports.append(dict(name=name,radius_A=radius,seed=seed,valid=valid,unconditional_draws=256,geometry_checks=checks,probe_path=str(path),probe_sha256=sha(path)))
    report=dict(complete=True,prepared_only=True,no_physical_weight_draws=True,protocol_sha256=sha(out/'protocol.json'),freeze_sha256=sha(out/'freeze.json'),
        production_allocation_unchanged_after_probes=True,probes=reports,analysis_driver_requires_extension=True,
        analyzer_extension='Existing original latent audit and generic geometric remasking can be reused; a new driver must validate these two source identities/charts and three-center priority-union/shell masks. Keep previous executed analyzer immutable.')
    write(out/'report.json',report)
    print(dict(complete=True,prepared_only=True,protocol=str(out/'protocol.json'),protocol_sha256=sha(out/'protocol.json'),
        commands_sha256=sha(out/'commands.json'),probes=[{k:p[k] for k in ('name','radius_A','valid','unconditional_draws')} for p in reports]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();prepare(args.out)
