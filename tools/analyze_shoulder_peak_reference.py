#!/usr/bin/env python3
"""Independent local inner-shoulder calibration with original full-N histories.

The local chart is chosen from discovery data. Fresh balls and historical
full-window complements stay separate; only fixed disjoint shells are added.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import heapq
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from analyze_far_peak_reference import add_population_errors, sum_independent_estimates
from analyze_latent_region import original_q_window
from analyze_global_fixed_region import region_latent
from analyze_peak_neighborhood import geometry_model_audit, geometry_rows
from audit_shoulder_mis_independently import near
from compare_far_local_history import independent_comparison, validate_runtime_metric
from compare_intermediate_local_reference import read_batches
from prepare_cayley_rms_cover import read, write, sha, require
from prepare_native_confirmation_atlas import AtomUnionAudit
from run_shoulder_mis_campaign import local_dependencies
from shoulder_peak_moments import ShoulderMoments, MASKS, inner_contains

ROOT = Path(__file__).resolve().parents[1]
WINDOW = dict(minimum=1., maximum=1.1, lower_inclusive=False, upper_inclusive=False)
SHOULDER_WINDOW = dict(minimum=1., maximum=2., lower_inclusive=False, upper_inclusive=False)
RADII = (.25, .5)
PHYSICAL = ('fixed_poses', 'capture_center', 'capture_radius', 'depletant_radius', 'reservoir_density')
DEFAULT_PREPARATION = ROOT/'runs/ab-inner-shoulder-peak-reference-preparation-20260921'


def compare_physical(actual, expected):
    require(all(actual[k] == expected[k] for k in PHYSICAL), 'Physical AB bath/capture differs')
    validate_runtime_metric({k: actual['metadata'][k] for k in ('native_poses', 'rigid_members', 'member_error_scale', 'angle_error_scale_deg')}, expected['metadata'])


def finish(aggregate, populations):
    result = add_population_errors(aggregate.report(), populations, aggregate.keys)
    for kind in ('physical', 'hard'):
        full = result[kind]['full']['logQ']
        for row in result[kind].values():
            row['fraction_of_observed_inner_mass'] = math.exp(row['logQ']-full) if row['logQ'] is not None else None
            row['scope'] = 'Within this source proposal support and original strict inner window; all original N and zero draws retained.'
    return result


def combine_shells(campaigns):
    require(sorted(c['radius_A'] for c in campaigns) == list(RADII), 'Need both frozen radii')
    ordered = sorted(campaigns, key=lambda c: c['radius_A'])
    seeds = [p['seed'] for c in ordered for p in c['populations']]
    require(len(seeds) == len(set(seeds)), 'Independent shell streams overlap')
    result = {kind: {'ball0p5': sum_independent_estimates([c[kind][f'radial_{i}'] for i,c in enumerate(ordered)])} for kind in ('physical','hard')}
    return dict(**result, pieces=[dict(root=c['root'], radius_A=c['radius_A'], selected_mask=f'radial_{i}',
        lower_A=(0.,.25)[i], upper_A=RADII[i], lower_inclusive=i==0, upper_inclusive=True) for i,c in enumerate(ordered)],
        scope='Independent sum of [0,.25] from radius.25 and (.25,.5] from radius.5 only. Whole balls are never added. Outside.5 is not estimated by these local references.')


def geometry_batch(rows, geometry, model, cfg):
    poses = [r['pose'] for r in rows]
    positions = np.asarray([p['position'] for p in poses]); quats = np.asarray([p['orientation'] for p in poses])
    near(np.sum(quats*quats,axis=1),np.ones(len(rows)))
    rotations = Rotation.from_quat(quats[:,[1,2,3,0]]).as_matrix()
    radii, logj, errors, _ = geometry_rows(positions,rotations,geometry)
    _, other, independent = region_latent(poses,dict(gaussian_chart=model,fixed_neighbor=cfg['fixed_poses'][0]))
    finite = np.isfinite(radii) & np.isfinite(other)
    require(np.array_equal(np.isfinite(radii),np.isfinite(other)), 'Geometric chart seam differs')
    error = float(np.max(abs(radii[finite]-other[finite])/(1+other[finite]))) if finite.any() else 0.
    require(error < 2e-7, 'Independent geometric radii differ')
    for radius in RADII: require(np.array_equal(radii <= radius,other <= radius), 'Frozen radius masks differ between reconstructions')
    return radii,logj,dict(**errors,geometric_radius_relative=error,matrix_radius_relative=independent['matrix_radius_relative_error'])


def original_weights(row, kind):
    """Remask without altering a source's original proposal denominator."""
    if kind == 'latent':
        weight = row['log_importance_weight']
        valid = row['capture_valid'] and row['hard_valid'] and row['region_valid']
        require((weight is not None) == valid, 'Original latent positive mask differs')
        if not valid: return None,None,None
        hard = row['log_hard_weight']; pair = [hard+c['log_weight'] for c in row['clouds']]
    else:
        if 'zero' in row:
            require(row.get('log_importance_weight') is None, 'Rejected guide row has weight')
            return None,None,None
        require(1 < row['q'] < 2, 'Original guided shoulder window differs')
        weight, hard = row['log_importance_weight'],row['log_hard_weight']
        near(hard,-row['log_proposal_density'])
        pair = [hard+w for w in row['cloud_log_weights']]
    require(len(pair)==2 and all(math.isfinite(x) for x in pair), 'Original cloud pair missing')
    near(float(np.logaddexp(*pair))-math.log(2),weight)
    return (weight,hard,pair) if inner_contains(row['q']) else (None,None,None)


def remask(root, manifest, cfg, model, geometry, original, kind, expected_hashes, source_name, radius=None):
    aggregate=ShoulderMoments(); populations=[]; hashes={}; top=[]; errors={}; seen=set()
    atom=AtomUnionAudit(read(root/'provenance/shape.json'),cfg['fixed_poses'])
    for pi,job in enumerate(manifest['jobs']):
        directory=Path(job['directory'] if kind=='latent' else job['output']); path=directory/'samples.jsonl'
        summary=read(directory/'summary.json'); run=read(directory/'manifest.json')
        count_expected=job['samples'] if kind=='latent' else run['samples']
        require(summary['complete'] and summary['samples']==count_expected and run['seed']==job['seed'], 'Original population incomplete or changed')
        require(job['seed'] not in seen,'Duplicate source population seed');seen.add(job['seed'])
        if kind=='latent':
            prior=next(p for p in original['populations'] if p['id']==job['id'])
            digest_expected=prior['samples_sha256']
            require(run['lambda_ratio']==manifest['lambda_ratio']==64 and run['cloud_replicates']==manifest['cloud_replicates']==2,'Latent cloud law changed')
        else:
            prior=next(p for p in original['populations'] if p['replicate']==directory.name)
            digest_expected=prior['sample_sha256']
            require(sha(directory/'summary.json')==prior['summary_sha256'],'Guided summary changed after audit')
            require(run['q_window']==original['q_window']==SHOULDER_WINDOW,'Guided source target differs')
            validate_runtime_metric(run['metric'],cfg['metadata'])
            require(run['guide']==original['guide'] and run['cover_mixture']==original['cover_mixture'],'Guided original full density differs')
            require(run['executable_sha256']==manifest['binary_sha256'] and run['shape_sha256']==model['shape_sha256'],'Guided runtime executable/shape differs')
        if expected_hashes: require(expected_hashes[str(path)]==digest_expected,'Previously audited source hash differs')
        local=ShoulderMoments(); digest=hashlib.sha256(); count=0
        for lines,rows in read_batches(path):
            for line in lines:digest.update(line)
            radii,logj,current=geometry_batch(rows,geometry,model,cfg)
            if radius is not None:
                require(np.all(radii<=radius*(1+1e-10)),'Local draw outside frozen chart ball')
                current['latent_radius']=near(radii,[r['latent_radius'] for r in rows])
                current['jacobian']=near(logj,[r['log_physical_jacobian'] for r in rows])
            for key,value in current.items(): errors[key]=max(errors.get(key,0.),value)
            for i,row in enumerate(rows):
                require(row['draw']==count,'Original draw ordering differs');count+=1
                weight,hard,pair=original_weights(row,kind)
                local.add(float(radii[i]),weight,hard,pair)
                if weight is not None:
                    record=dict(population=directory.name,seed=job['seed'],draw=row['draw'],pose=row['pose'],q=row['q'],
                        geometric_radius_A=float(radii[i]),original_log_importance_weight=weight,original_row=row,source_samples_path=str(path))
                    entry=(weight,-pi,-row['draw'],record)
                    if len(top)<8:heapq.heappush(top,entry)
                    elif entry[:3]>top[0][:3]:heapq.heapreplace(top,entry)
        require(count==count_expected,'Original full N differs')
        require(digest.hexdigest()==digest_expected==summary['samples_sha256'],'Rows changed after original audit')
        hashes[str(path)]=digest.hexdigest()
        for name in ('manifest.json','summary.json'):hashes[str(directory/name)]=sha(directory/name)
        aggregate.merge(local);populations.append(dict(id=directory.name,seed=job['seed'],samples=count,**local.report()))
    result=finish(aggregate,populations)
    extrema=[]
    for _,_,_,row in sorted(top,key=lambda t:t[:3],reverse=True):
        gaps=atom.gaps(row['pose']);require(min(gaps)>=0,'Positive extreme clashes atomic AB')
        row['minimum_AB_atomic_gaps_A']=gaps;row['source_samples_sha256']=hashes[row['source_samples_path']];extrema.append(row)
    return dict(root=str(root),name=source_name,radius_A=radius,**result,populations=populations,
        top8_atomic_checks=extrema,input_sha256=hashes,geometry_reconstruction_errors=errors,
        original_unconditional_draws=sum(p['samples'] for p in populations),
        support_scope='Finite local chart ball only' if radius is not None else 'Original source full inner window, including direct same-row local complements')


def verify_reference_manifest(root,declared,protocol,cfg,model):
    master=read(root/'manifest.json');region=read(root/'provenance/region.json');local_cfg=read(root/'provenance/config.json')
    compare_physical(local_cfg,cfg)
    require(master['region_sha256']==declared['region_sha256']==sha(root/'provenance/region.json'),'Frozen local region changed')
    require(original_q_window(region)==WINDOW and region['gaussian_chart']==model,'Strict inner window/chart changed')
    require(region['mahalanobis_radius']==declared['radius_A'] and region.get('minimum_mahalanobis_radius',0.)==0.,'Local ball support changed')
    require(region['physical_fixed_neighbors']==cfg['fixed_poses'] and region['fixed_neighbor']==cfg['fixed_poses'][0],'Physical AB/chart anchor differs')
    require(region['physical_metric']==local_cfg['metadata'],'Local original q definition changed')
    require(master['archive_sha256']['latent-region-normalizer']==protocol['executable_sha256'],'Frozen executable changed')
    require(master['lambda_ratio']==protocol['lambda_ratio']==64 and master['cloud_replicates']==protocol['cloud_replicates']==2,'Frozen cloud law changed')
    require(len(master['jobs'])==declared['populations'] and [j['seed'] for j in master['jobs']]==declared['seeds'],'Frozen populations/seeds changed')
    require(all(j['samples']==declared['samples_per_population'] for j in master['jobs']),'Frozen draw counts changed')
    for name,digest in master['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,'Archived runtime source/input changed')
    require(master['archive_sha256']['analyze_latent_region.py']==protocol['archived_sha256']['analyze_latent_region.py'],'Frozen original auditor source changed')
    return master


def ensure_audits(requests,out,allow_run):
    state=dict(phase='original_latent_audits',processes=[]);live=[];state_path=out/'runner-state.json'
    for declared,root,_ in requests:
        audit=root/'assessment/analysis.json'
        if audit.exists():
            state['processes'].append(dict(radius_A=declared['radius_A'],reused=True,path=str(audit),sha256=sha(audit)));continue
        require(allow_run,'Original audit missing: launch only after physical terminal authorization')
        command=[sys.executable,str(root/'provenance/analyze_latent_region.py'),'--root',str(root)]
        logpath=out/f"original-audit-r{str(declared['radius_A']).replace('.','p')}.log";log=logpath.open('x')
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        record=dict(radius_A=declared['radius_A'],pid=process.pid,command=command,log=str(logpath),output=str(audit),terminal=False)
        state['processes'].append(record);live.append((process,log,record));write(state_path,state)
        print(f"Original latent audit PID{process.pid}: {root}",flush=True)
    failures=[]
    for process,log,record in live:
        code=process.wait();log.close();record.update(terminal=True,exit_code=code);write(state_path,state)
        if code:failures.append(record)
    state['phase']='original_audits_failed' if failures else 'original_audits_complete';write(state_path,state)
    require(not failures,'Original audit failed; diagnose without replaying successful audits')
    return state


def check_equal_mass(actual,expected):
    if actual is None: require(expected is None,'Original zero mass differs')
    else: require(expected is not None,'Original positive mass missing');near(actual,expected,2e-8)


def source_history(declared,cfg,model,geometry):
    """Reuse a hash-bound previous full audit; never run new clouds/audit jobs."""
    root=Path(declared['root']);comparison_path=Path(declared['comparison_path'])
    require(sha(comparison_path)==declared['comparison_sha256'],'Historical comparison changed')
    require(sha(root/'manifest.json')==declared['manifest_sha256'],'Historical manifest changed')
    comparison=read(comparison_path);master=read(root/'manifest.json');kind='latent' if declared['name']=='direct-inner' else 'guided'
    cfgpath=root/'provenance/config.json';old_cfg=read(cfgpath);compare_physical(old_cfg,cfg)
    require(sha(root/'provenance/shape.json')==model['shape_sha256'],'Historical protein shape differs')
    for name,digest in master['archive_sha256'].items():require(sha(root/'provenance'/name)==digest,'Historical archive changed')
    if kind=='latent':
        require(comparison['complete'],'Historical direct comparison incomplete')
        record=comparison['focused_band'];auditpath=root/'assessment/analysis.json';region=read(root/'provenance/region.json')
        require(original_q_window(region)==declared['original_q_window']==record['integration_window']==WINDOW,'Historical strict inner target changed')
        for path,digest in record['input_sha256'].items():require(sha(path)==digest,'Historical direct audit input changed')
        require(region['physical_fixed_neighbors']==cfg['fixed_poses'] and region['fixed_neighbor']==cfg['fixed_poses'][0],'Historical direct AB/frame differs')
        require(record['physical']['draws']==declared['original_N'],'Historical direct N changed')
        original=read(auditpath)
        require(original['independently_reconstructed_poses']==declared['original_N'],'Historical direct all-row audit missing')
        physical_expected=record['physical']['logQ'];hard_expected=record['hard']['logQ']
    else:
        require(comparison['q_window']==declared['original_q_window']==SHOULDER_WINDOW,'Historical guide target changed')
        record=comparison['campaigns'][declared['comparison_key']];auditpath=root/'assessment-streaming.json'
        require(sha(auditpath)==record['assessment_sha256'],'Historical guided all-row audit changed')
        require(sha(root/'manifest.json')==record['manifest_sha256'] and sha(cfgpath)==record['config_sha256'],'Historical guide configuration differs')
        original=read(auditpath)
        require(original['complete'] and original['all_rows_and_hashes_validated'] and original['independent_complete_cover_validated'],'Historical guide audit incomplete')
        require(original['analysis_script_sha256']==master['archive_sha256']['analyze_native_region_reference.py'],'Historical original auditor source differs')
        require(original['samples']==original['independent_q_and_density_rows']==declared['original_N'],'Historical all-row N changed')
        physical_expected=record['bands']['0']['logQ'];hard_expected=record['hard_bands']['0']['logQ']
    require(record['root']==str(root),'Historical source identity differs')
    result=remask(root,master,cfg,model,geometry,original,kind,record['sample_sha256'],declared['name'])
    require(result['original_unconditional_draws']==declared['original_N'],'Historical full N changed')
    check_equal_mass(result['physical']['full']['logQ'],physical_expected)
    check_equal_mass(result['hard']['full']['logQ'],hard_expected)
    result.update(original_audit_path=str(auditpath),original_audit_sha256=sha(auditpath),CPU_seconds=record['CPU_seconds'],
        source_original_q_window=declared['original_q_window'],remasked_q_window=WINDOW,
        calibration_scope='Data-selected region. Historical and fresh observations stay separate; selection precludes interpreting their observed-error contrasts as nominal significance tests.')
    result['input_sha256'].update({str(p):sha(p) for p in (comparison_path,root/'manifest.json',cfgpath,auditpath)})
    return result


def validate_plan(protocol):
    require(protocol['original_q_window']==WINDOW,'Target is not the original strict inner shoulder')
    plan=protocol['analysis_plan']
    require(plan['radial_edges_A']==[0.,.25,.5],'Frozen radial edges changed')
    require(plan['disjoint_shell_sources']==[dict(radius_A=.25,shell=[0.,.25],lower_inclusive=True),dict(radius_A=.5,shell=[.25,.5],lower_inclusive=False)],'Frozen independent shell allocation changed')
    require(sorted(c['radius_A'] for c in protocol['campaigns'])==list(RADII),'Unplanned reference radius')
    require(all(c['populations']==16 and c['samples_per_population']==16384 for c in protocol['campaigns']),'Fixed local reference budgets changed')
    require(len(protocol['historical_campaigns'])==3 and {h['name'] for h in protocol['historical_campaigns']}=={'direct-inner','mixture-confirmation','geometry-confirmation'},'Historical controls changed')


def compare_references(histories,campaigns,combined):
    lookup={c['radius_A']:c for c in campaigns}
    comparisons={}
    for history in histories:
        kinds={}
        for kind in ('physical','hard'):
            entries={}
            for radius in RADII:
                key='ball0p25' if radius==.25 else 'ball0p5'
                entries[key]=independent_comparison(history[kind][key],lookup[radius][kind]['full'])
            entries['radial_1']=independent_comparison(history[kind]['radial_1'],lookup[.5][kind]['radial_1'])
            entries['ball0p5_independent_shell_sum']=independent_comparison(history[kind]['ball0p5'],combined[kind]['ball0p5'])
            kinds[kind]=entries
        comparisons[history['name']]=kinds
    nested={kind:independent_comparison(lookup[.25][kind]['full'],lookup[.5][kind]['ball0p25']) for kind in ('physical','hard')}
    for entry in nested.values():entry['scope']='Same fixed local ball under independently drawn radius.25 and radius.5 references. Keep independent original budgets; observed errors do not certify unseen tails.'
    return dict(historical_vs_fresh=comparisons,fresh_nested_region_control=nested)


def analyze(preparation,out,run_audits=False):
    preparation=Path(preparation).resolve();out=Path(out).resolve();require(not out.exists(),'Use a fresh derived audit directory')
    protocol=read(preparation/'protocol.json');freeze=read(preparation/'freeze.json');prepared=read(preparation/'report.json')
    require(prepared['complete'] and prepared['freeze_sha256']==sha(preparation/'freeze.json') and prepared['frozen_protocol_sha256']==sha(preparation/'protocol.json'),'Preparation not complete/frozen')
    for name,digest in freeze.items():require(sha(preparation/name)==digest,'Preparation freeze changed')
    for name,digest in protocol['archived_sha256'].items():require(sha(preparation/'provenance'/name)==digest,'Preparation archive changed')
    validate_plan(protocol)
    cfg=read(preparation/'config.json');model=read(preparation/'model.json');selected=read(preparation/'selected-pose.json')
    require(sha(preparation/'config.json')==protocol['config_sha256'] and sha(preparation/'model.json')==protocol['model_sha256'],'Frozen config/chart changed')
    compare_physical(cfg,protocol['physical'])
    require(model['shape_sha256']==sha(preparation/'provenance/shape.json'),'Frozen physical shape differs')
    source=protocol['selection_source'];discovery=read(preparation/'provenance/discovery.json')
    require(source['sha256']==sha(preparation/'provenance/discovery.json')==sha(source['path']),'Selected discovery artifact changed')
    require(source['selected_peak']==discovery['selected_peak']==0 and selected==discovery['candidates'][0],'Selected original peak changed')
    require(sha(preparation/'selected-pose.json')==protocol['selected_pose_sha256'],'Selected pose snapshot changed')
    require((selected['population'],selected['seed'],selected['draw'])==('r02',99633028,211501),'Peak source identity changed')
    require(model==discovery['geometric_model'],'Geometric chart refitted after selection')
    geometry_report,geometry=geometry_model_audit(model,dict(fixed_neighbor=cfg['fixed_poses'][0]),cfg,selected['pose'])
    requests=[]
    for declared in protocol['campaigns']:
        root=Path(declared['output']);master=verify_reference_manifest(root,declared,protocol,cfg,model)
        requests.append((declared,root,master))
    newseeds=[j['seed'] for _,_,m in requests for j in m['jobs']]
    oldseeds=[j['seed'] for h in protocol['historical_campaigns'] for j in read(Path(h['root'])/'manifest.json')['jobs']]
    require(len(set(newseeds))==len(newseeds) and not set(newseeds)&set(oldseeds),'Independent local/historical seeds overlap')
    out.mkdir(parents=True);ensure_audits(requests,out,run_audits)
    campaigns=[]
    for declared,root,master in requests:
        auditpath=root/'assessment/analysis.json';original=read(auditpath)
        require(original['original_q_window']==WINDOW and original['independently_reconstructed_poses']==declared['populations']*declared['samples_per_population'],'New original all-row audit incomplete')
        result=remask(root,master,cfg,model,geometry,original,'latent',{},f"R{declared['radius_A']}",declared['radius_A'])
        check_equal_mass(result['physical']['full']['logQ'],original['estimate']['logQ'])
        check_equal_mass(result['hard']['full']['logQ'],original['hard_region']['logQ'])
        require(result['original_unconditional_draws']==declared['populations']*declared['samples_per_population'],'New full N differs')
        result.update(original_audit_path=str(auditpath),original_audit_sha256=sha(auditpath),CPU_seconds=original['sampler_cpu_seconds'])
        result['input_sha256'].update({str(p):sha(p) for p in (root/'manifest.json',root/'provenance/config.json',root/'provenance/region.json',auditpath)})
        for kind in ('physical','hard'):
            for key,row in result[kind].items():
                row['scope']='Integral only within this original local proposal ball AND the inner target window; unsupported complements cannot estimate the global inner complement.'
        campaigns.append(result)
        print(f"R{declared['radius_A']}: logQ={result['physical']['full']['logQ']}, rowRSE={result['physical']['full']['row_RSE']}",flush=True)
    histories=[source_history(h,cfg,model,geometry) for h in protocol['historical_campaigns']]
    combined=combine_shells(campaigns);comparisons=compare_references(histories,campaigns,combined)
    sources={str(preparation/name):sha(preparation/name) for name in freeze}
    for result in campaigns+histories:sources.update(result['input_sha256'])
    for path,digest in sources.items():require(sha(path)==digest,'Input changed during audit/remasking')
    archive=out/'provenance';archive.mkdir()
    archived={**local_dependencies([Path(__file__)]),'protocol.json':preparation/'protocol.json','freeze.json':preparation/'freeze.json',
        'model.json':preparation/'model.json','selected-pose.json':preparation/'selected-pose.json','discovery.json':preparation/'provenance/discovery.json'}
    for name,path in archived.items():(archive/name).write_bytes(Path(path).read_bytes())
    report=dict(complete=True,original_q_window=WINDOW,radial_edges_A=[0.,.25,.5],campaigns=campaigns,histories=histories,
        independent_shell_sum=combined,comparisons=comparisons,geometry_model_audit=geometry_report,
        selection=dict(source_sha256=source['sha256'],selected_pose_sha256=sha(preparation/'selected-pose.json'),
            source_population=selected['population'],source_seed=selected['seed'],source_draw=selected['draw'],
            discovery_three_point_fraction=discovery['selected_three_fraction_of_entire_source_estimate'],
            scope='Discovery only, not an independent physical estimate or fixed basin probability.'),
        source_sha256=sources,archived_sha256={p.name:sha(p) for p in archive.iterdir()},
        scope='Fixed local inner-shoulder calibration, original strict1<q<1.1 and AB bath. Historical original Ns and full proposal densities remain intact, including all outer-shoulder and invalid zeros. Historical complements are direct positive masks with same-row covariance. Fresh whole balls stay separate; only prespecified independent disjoint shells are summed. Data selection and unknown local/outside tails preclude a convergence or mixing claim. No historical/fresh pooling or substitution of local mass for full shoulder.')
    write(out/'analysis.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preparation',type=Path,default=DEFAULT_PREPARATION)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--run-original-audits',action='store_true')
    args=parser.parse_args();result=analyze(args.preparation,args.out,args.run_original_audits)
    print(dict(complete=result['complete'],output=str(args.out/'analysis.json')))
