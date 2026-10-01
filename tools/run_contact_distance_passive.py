#!/usr/bin/env python3
"""Freeze and drain the fixed proposal-only contact-distance diagnostic, one CPU.

No Poisson clouds or physical contact-mass estimates are generated.
"""
from __future__ import annotations
from collections import Counter
import argparse
import copy
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy.special import logsumexp

from diagnose_fitted_kernel_shear import read, sha, require, write_new
from prepare_contact_distance_passive import SEEDS, COVERAGE_SEEDS, GUIDE_SHA, REGION_SHA, SHAPE_SHA, WIDTHS, ARMS
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_confirmation import runtime, worker_environment
from run_contact_tail_pilot import verify_frozen, copy_frozen
from run_mobile_posterior_pilot import verify_bundle
from run_full_vessel_comparison import execute_group
from run_smc_importance_bridge import process_token

SCHEMA='contact-distance-passive-controller-v1'


def validate_design(plan):
    require(plan['fresh_draws']==1536 and plan['archived_probe_queries']==234 and
            plan['additional_coverage_queries']==384 and plan['total_archived_queries']==618 and
            plan['total_output_pose_rows']==2154 and plan['maximum_CPU_workers']==1 and
            plan['new_Poisson_clouds']==0,'Passive budget changed')
    require(plan['widths_A']==WIDTHS and plan['contact_neighbor_indices']==[0,1] and
            plan['minimum_center_distance']==1e-8 and plan['minimum_polygon_area']==1e-16 and
            plan['azimuth_floors']==dict(radius_floor=1e-8,projection_floor=1e-10,gamma_min=.01,gamma_max=math.pi),
            'Frozen proposal choices changed')
    require(plan['arms']==[dict(name=a,conditional_probability=beta,localized_probability=b) for a,beta,b in ARMS],
            'Frozen comparison arms changed')
    require(plan['keep_all_draws'] and not plan['optional_stopping'] and not plan['retries'] and
            not plan['larger_autoextension'],'No-retry attempted-draw contract changed')
    jobs=plan['jobs']
    require(len(jobs)==12 and [j['seed'] for j in jobs]==SEEDS,'Passive streams changed')
    for index,job in enumerate(jobs):
        require(job['arm']==ARMS[index//4][0] and job['id']==f'r{index%4:02}',
                'Passive stream order changed')
        require(job['fresh_proposal_draws']==128 and job['archived_probe_queries']==(78 if index%4==0 else 0) and
                job['probe_file']==('probes.jsonl' if index%4==0 else None),
                'Passive per-stream allocation changed')
    require(len(plan['coverage_jobs'])==3,'Missing breadth controls')
    for index,coverage in enumerate(plan['coverage_jobs']):
        require(coverage['arm']==ARMS[index][0] and coverage['id']=='coverage' and
                coverage['fresh_proposal_draws']==0 and coverage['archived_probe_queries']==128 and
                coverage['seed']==COVERAGE_SEEDS[index] and coverage['probe_file']=='coverage-probes.jsonl',
                'Breadth control changed')
    require(len({j['seed'] for j in all_jobs(plan)})==15,'Repeated passive stream seed')


def all_jobs(plan):
    return [*plan['jobs'],*plan['coverage_jobs']]


def command(out, job):
    common=out/'common'
    cmd=[str(common/'contact-distance-guide-audit'),'--config',str(common/'config.json'),
        '--region',str(common/'region.json'),'--importance-guide',str(common/(job['arm']+'.json')),
        '--out',job['directory'],'--samples',str(job['fresh_proposal_draws']),'--seed',str(job['seed'])]
    if job['archived_probe_queries']:cmd+=['--probes',str(common/job.get('probe_file','probes.jsonl'))]
    return cmd


def freeze(out, preparation, binary, bundle, repository, references):
    out,preparation,binary,bundle,repository=map(lambda p:Path(p).resolve(),(out,preparation,binary,bundle,repository))
    require(not out.exists(),'Fresh passive output directory required')
    references=[Path(path).resolve() for path in references]
    require(references and all(path.is_file() for path in references),'Reviewed reference receipts required')
    verify_frozen(preparation);plan=read(preparation/'plan.json');validate_design(plan)
    source,source_hashes=verify_bundle(binary,bundle,repository)
    from contact_distance_reference import audit  # dependency closure; no execution here
    sources=local_dependencies([Path(__file__)])
    out.mkdir();common=out/'common';common.mkdir();(out/'logs').mkdir();(out/'runs').mkdir()
    copy_frozen(preparation,common/'preparation')
    for name,path in sources.items():shutil.copy2(path,common/name)
    shutil.copy2(binary,common/'contact-distance-guide-audit');shutil.copy2(bundle,common/'source-bundle.json')
    for name,entry in source['files'].items():
        path=common/'rust-source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(entry['text'])
    (common/'reference-receipts').mkdir()
    receipts=[]
    for index,path in enumerate(references):
        name=f'{index:02}-{path.name}';shutil.copy2(path,common/'reference-receipts'/name)
        receipts.append(dict(original=str(path),sha256=sha(path),archive='common/reference-receipts/'+name,
            scope='Reviewed reference evidence; frozen by content, not automatically inferred from file naming'))
    for name in ('region.json','shape.json','original92.json'):
        shutil.copy2(preparation/'common'/name,common/name)
    shutil.copy2(preparation/'probes.jsonl',common/'probes.jsonl')
    shutil.copy2(preparation/'coverage-probes.jsonl',common/'coverage-probes.jsonl')
    shutil.copy2(preparation/'coverage-selection.json',common/'coverage-selection.json')
    labels=read(preparation/'component-labels.json')
    require(len(labels['components'])==92 and not labels['probe_data_used'] and not labels['fit_data_used'],
            'Frozen label provenance changed')
    expected_labels=[entry['contact_pairs'] for entry in labels['components']]
    for arm,beta,b in ARMS:
        guide=read(preparation/'guides'/f'{arm}.json')
        require(guide['schema']=='defensive-contact-distance-guide-v1' and guide['contact_neighbor_indices']==[0,1] and
                guide['conditional_probability']==beta and guide['contact_widths_A']==WIDTHS and
                guide['minimum_center_distance']==1e-8 and guide['minimum_polygon_area']==1e-16 and
                guide['azimuth']==dict(localized_probability=b,**plan['azimuth_floors']) and
                guide['defensive_uniform_shell_probability']==.5 and guide['component_contact_pairs']==expected_labels,
                'Conditioning guide differs from frozen design')
        base=read(common/'original92.json')
        require(guide['gaussian_components']==base['gaussian_components'] and len(base['gaussian_components'])==92,
                'Gaussian components changed')
        shutil.copy2(preparation/'guides'/f'{arm}.json',common/f'{arm}.json')
    cfg=read(preparation/'common/config.json');cfg['shape']=str(common/'shape.json');write_new(common/'config.json',cfg)
    require(sha(common/'shape.json')==SHAPE_SHA and sha(common/'region.json')==REGION_SHA and
            sha(common/'original92.json')==GUIDE_SHA,'Original target/guide identity changed')
    def bind_job(source_job):
        job={k:copy.deepcopy(v) for k,v in source_job.items() if k!='command'}
        job.update(directory=str(out/'runs'/job['arm']/job['id']),log=str(out/'logs'/(job['arm']+'-'+job['id']+'.log')),
            kind='proposal_only',status='pending')
        (out/'runs'/job['arm']).mkdir(exist_ok=True);job['command']=command(out,job)
        return job
    jobs=[bind_job(j) for j in plan['jobs']]
    coverage=[bind_job(j) for j in plan['coverage_jobs']]
    protocol=copy.deepcopy(plan);protocol.update(schema=SCHEMA,jobs=jobs,repository=str(repository),
        coverage_jobs=coverage,
        preparation=str(preparation),preparation_sha256=sha(preparation/'plan.json'),
        preparation_freeze_sha256=sha(preparation/'freeze.json'),binary_sha256=sha(binary),
        source_bundle_sha256=sha(bundle),rust_sources=source_hashes,python_sources={n:sha(common/n) for n in sources},
        reference_receipts=receipts,
        runtime=runtime(),controller_sha256=sha(__file__),scope='Fixed passive diagnostic; no new Poisson clouds or physical masses')
    write_new(out/'protocol.json',protocol)
    write_new(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    validate(out,sha(out/'protocol.json'))
    return protocol


def validate(out, expected):
    out=Path(out).resolve();require(sha(out/'protocol.json')==expected,'Protocol hash mismatch')
    p=read(out/'protocol.json');validate_design(p);verify_frozen(out)
    require(p['schema']==SCHEMA and sys.flags.optimize==0 and runtime()==p['runtime'],'Controller schema/runtime differs')
    require(sha(__file__)==p['controller_sha256'] and
            {name:sha(path) for name,path in local_dependencies([Path(__file__)]).items()}==p['python_sources'],
            'Executing controller/import closure differs from frozen Python source')
    common=out/'common';verify_bundle(common/'contact-distance-guide-audit',common/'source-bundle.json',common/'rust-source')
    require(sha(common/'contact-distance-guide-audit')==p['binary_sha256'],'Frozen executable changed')
    preparation=Path(p['preparation']);verify_frozen(preparation)
    require(sha(preparation/'plan.json')==p['preparation_sha256'] and
            sha(preparation/'freeze.json')==p['preparation_freeze_sha256'],'Preparation binding changed')
    for job in all_jobs(p):require(job['command']==command(out,job),'Frozen command changed')
    return p


def cpu_children():
    usage=resource.getrusage(resource.RUSAGE_CHILDREN);return usage.ru_utime+usage.ru_stime


def verify_output(directory,job,p):
    manifest=read(directory/'manifest.json');summary=read(directory/'summary.json')
    common=directory.parents[2]/'common'
    expected=dict(schema='contact-distance-guide-audit-v1',samples=job['fresh_proposal_draws'],seed=job['seed'],
        executable_sha256=p['binary_sha256'],source_bundle_sha256=p['source_bundle_sha256'],
        config_sha256=sha(common/'config.json'),region_sha256=REGION_SHA,shape_sha256=SHAPE_SHA,
        guide_sha256=sha(common/(job['arm']+'.json')),probes_sha256=sha(common/job.get('probe_file','probes.jsonl')) if job['archived_probe_queries'] else None,
        physical_jobs=0)
    require(all(manifest.get(k)==v for k,v in expected.items()),'Output manifest differs from planned source/inputs')
    require(summary['complete'] and summary['samples']==job['fresh_proposal_draws'] and summary['probes']==job['archived_probe_queries'] and
            summary['manifest']==manifest,'Incomplete output/attempt allocation')
    for name in ('samples','probes'):
        require(sha(directory/(name+'.jsonl'))==summary[name+'_sha256'],'Output row hash mismatch')
    for name,key in [('config.json','config_sha256'),('region.json','region_sha256'),('shape.json','shape_sha256'),
                     ('importance-guide.json','guide_sha256'),('source-bundle.json','source_bundle_sha256')]:
        require(sha(directory/'provenance'/name)==manifest[key],'Saved provenance mismatch')
    verify_attempts(directory,common,job)
    return summary


def verify_attempts(directory,common,job):
    samples=read_rows(directory/'samples.jsonl')
    require(len(samples)==job['fresh_proposal_draws'] and
            all(type(r['id']) is int and r['id']==i and r['kind']=='fresh' for i,r in enumerate(samples)),
            'Missing, reordered or duplicated attempted draw')
    probes=read_rows(directory/'probes.jsonl')
    original=read_rows(common/job['probe_file']) if job['archived_probe_queries'] else []
    require(len(probes)==len(original)==job['archived_probe_queries'] and
            all(r['kind']=='probe' and r['id']==s['id'] and r['latent']==s['latent'] for r,s in zip(probes,original)),
            'Missing, reordered or altered archived probe')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def summarize(out,state):
    arms={};probes={};names=[a[0] for a in ARMS];maximum_errors={}
    for job in all_jobs(state):
        receipt=read(Path(job['directory'])/'independent-audit.json')
        for key,value in receipt.get('maximum_errors',{}).items():
            maximum_errors[key]=max(maximum_errors.get(key,0.),value)
    for arm in names:
        populations=[]
        for job in state['jobs']:
            if job['arm']!=arm:continue
            directory=Path(job['directory']);rows=read_rows(directory/'samples.jsonl')
            total=read(directory/'summary.json')
            valid=[r['hard_valid'] and r['shell_valid'] and r['capture_valid'] for r in rows]
            joint=[sum(bool(ok and all(r['width_contacts'][i])) for r,ok in zip(rows,valid)) for i in range(3)]
            conditional=[r for r in rows if r['draw']['conditional']]
            successful=[r for r in conditional if not r['draw']['fallback']]
            reasons=Counter(r['draw']['fallback_reason'] for r in conditional if r['draw']['fallback'])
            selected_by_width=[]
            for wi in range(3):
                subset=[r for r in successful if r['draw']['width_index']==wi]
                selected_by_width.append(dict(width_A=WIDTHS[wi],draws=len(subset),
                    hard_valid=sum(r['hard_valid'] for r in subset),shell_valid=sum(r['shell_valid'] for r in subset),
                    hard_shell_capture=sum(r['hard_valid'] and r['shell_valid'] and r['capture_valid'] for r in subset)))
            require(len(conditional)==total['conditioned_draws'] and sum(reasons.values())==total['fallback_draws'],
                    'Summary selected-branch counts differ from every recorded draw')
            fresh_cpu=total['fresh_total_cpu_seconds'];probe_cpu=total['probe_total_cpu_seconds']
            setup_overhead=job['child_cpu_seconds']-fresh_cpu-probe_cpu
            populations.append(dict(id=job['id'],seed=job['seed'],draws=len(rows),hard_shell_capture=sum(valid),
                shell=sum(r['shell_valid'] for r in rows),hard=sum(r['hard_valid'] for r in rows),
                capture=sum(r['capture_valid'] for r in rows),joint_contact_by_width=joint,
                conditional=len(conditional),fallback=sum(reasons.values()),fallback_reasons=dict(reasons),
                successful_conditioning=len(successful),selected_success_by_width=selected_by_width,
                localized_projection_fallback=sum(r['draw']['azimuth_law']['localized_probability']==0 for r in successful)
                    if arm=='localized_phi92' else 0,
                fresh_cpu_seconds=fresh_cpu,draw_cpu_seconds=total['draw_cpu_seconds'],density_cpu_seconds=total['density_cpu_seconds'],
                probe_cpu_seconds=probe_cpu,full_child_cpu_seconds=job['child_cpu_seconds'],setup_and_unattributed_cpu_seconds=setup_overhead,
                independent_audit_cpu_seconds=job['audit_cpu_seconds'],output_sha256=job['output_sha256']))
            for row in read_rows(directory/'probes.jsonl'):
                probes.setdefault(row['id'],{})[arm]=row['log_proposal_density']
        cpu=sum(r['fresh_cpu_seconds'] for r in populations)
        fractions={}
        for label,index in [('hard_shell_capture',None),*[(f'joint_width_{w}',i) for i,w in enumerate(WIDTHS)]]:
            values=np.array([(r['hard_shell_capture'] if index is None else r['joint_contact_by_width'][index])/r['draws'] for r in populations])
            fractions[label]=dict(mean=float(values.mean()),population_SE=float(values.std(ddof=1)/math.sqrt(len(values))),
                populations=values.tolist(),scope='Four-population descriptive SE; no physical occupancy inference')
        arms[arm]=dict(populations=populations,fresh_attempts=sum(r['draws'] for r in populations),
            fractions=fractions,total_fresh_cpu_seconds=cpu,total_probe_cpu_seconds=sum(r['probe_cpu_seconds'] for r in populations),
            total_child_cpu_seconds=sum(r['full_child_cpu_seconds'] for r in populations),
            hard_shell_capture_per_fresh_cpu=sum(r['hard_shell_capture'] for r in populations)/cpu,
            joint_by_width_per_fresh_cpu=[sum(r['joint_contact_by_width'][i] for r in populations)/cpu for i in range(3)],
            scope='Fresh CPU includes draw, complete density and endpoint geometry; startup, probes and independent audits separately recorded. No physical-speed claim.')
    archived=read(out/'common/preparation/axis-diagnostics.json')['rows'];moments={}
    require(len(probes)==78 and all(set(row)==set(names) for row in probes.values()),'Incomplete critical cross-arm density matrix')
    for source_arm in ('baseline','expanded'):
        source=[r for r in archived if r['source']['arm']==source_arm]
        # Source log weights already include original J/q and each cloud. The
        # factor oldq/newq converts their product to the candidate M2 integrand.
        # Keep original source arms separate; per-population denominator remains.
        terms={name:np.array([sum(r['source']['paired_log_weights'])+r['source']['log_q']-probes[r['id']][name]
               -math.log(r['source']['source_attempted_draws']) for r in source]) for name in names}
        logs={name:float(logsumexp(x)) for name,x in terms.items()}
        moments[source_arm]=dict(rows=len(source),
            to_baseline92_paired_M2_ratio={name:math.exp(logs[name]-logs['baseline92']) for name in names},
            contribution_ESS={name:float(math.exp(2*logs[name]-logsumexp(2*x))) for name,x in terms.items()},
            largest_fraction={name:float(math.exp(max(x)-logs[name])) for name,x in terms.items()},
            scope='Retrospective paired-M2 contribution ratios on saved competing55 rows, original source arms separate; no new normalizer or unseen-tail bound.')
    selection={r['id']:r for r in read(out/'common/coverage-selection.json')['rows']}
    coverage={}
    for job in state['coverage_jobs']:
        rows=read_rows(Path(job['directory'])/'probes.jsonl')
        require(len(rows)==128 and {r['id'] for r in rows}==set(selection),'Coverage identities changed')
        arm={}
        for name in sorted({r['coverage_class'] for r in selection.values()}):
            chosen=[r for r in rows if selection[r['id']]['coverage_class']==name]
            gains=np.array([r['log_proposal_density']-selection[r['id']]['original92_log_q'] for r in chosen])
            arm[name]=dict(rows=len(chosen),log_q_ratio_min=float(min(gains)),log_q_ratio_median=float(np.median(gains)),
                log_q_ratio_max=float(max(gains)),hard_shell_capture=sum(r['hard_valid'] and r['shell_valid'] and r['capture_valid'] for r in chosen),
                scope='Same frozen uniform saved-row subset within each original class/population; not a physical mass estimator')
        coverage[job['arm']]=dict(classes=arm,child_cpu_seconds=job['child_cpu_seconds'],audit_cpu_seconds=job['audit_cpu_seconds'])
    return dict(schema=SCHEMA,complete=True,arms=arms,archived_moment_diagnostics=moments,coverage=coverage,
        coverage_jobs=state['coverage_jobs'],fresh_attempts=1536,archived_queries=234,additional_coverage_queries=384,
        total_archived_queries=618,maximum_independent_audit_errors=maximum_errors,
        new_Poisson_clouds=0,physical_mass_estimates_generated=0,
        full_vessel_gate_open=False,assembly_gate_open=False,protocol_sha256=state['protocol_sha256'])


def run(out,expected):
    out=Path(out).resolve();p=validate(out,expected)
    require(not (out/'status.json').exists(),'No passive retries or repeated launch')
    state=dict(schema=SCHEMA,complete=False,phase='starting',protocol_sha256=expected,started=time.time(),
        jobs=copy.deepcopy(p['jobs']),coverage_jobs=copy.deepcopy(p['coverage_jobs']),physical_jobs_launched=0,new_Poisson_clouds=0)
    def snapshot():
        (out/'status.tmp').write_text(json.dumps(state,indent=2)+'\n');(out/'status.tmp').replace(out/'status.json')
    with (out/'launch-claim.json').open('x') as stream:json.dump(dict(pid=os.getpid(),process_birth=process_token(os.getpid()),protocol_sha256=expected),stream)
    with (Path(p['preparation'])/'execution-claim.json').open('x') as stream:json.dump(dict(output=str(out),protocol_sha256=expected),stream)
    snapshot()
    def interrupted(signum,frame):raise InterruptedError('Passive controller signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        from contact_distance_reference import audit
        for job in all_jobs(state):
            state['phase']=job['arm']+'/'+job['id'];snapshot();before=cpu_children()
            execute_group([job],snapshot,1,Path(p['repository']),worker_environment(),before_launch=lambda:verify_frozen(out))
            job['child_cpu_seconds']=cpu_children()-before
            directory=Path(job['directory']);verify_output(directory,job,p)
            tick=time.process_time();result=audit(directory);job['audit_cpu_seconds']=time.process_time()-tick
            require(result.get('complete',False),'Independent audit incomplete')
            write_new(directory/'independent-audit.json',result)
            job['output_sha256']={name:sha(directory/name) for name in ('manifest.json','summary.json','samples.jsonl','probes.jsonl','independent-audit.json')}
            snapshot()
        verify_frozen(out)
        for job in all_jobs(state):
            require(all(sha(Path(job['directory'])/name)==digest for name,digest in job['output_sha256'].items()),
                    'Completed audited rows changed before aggregation')
        analysis=summarize(out,state);write_new(out/'analysis.json',analysis)
        state.update(complete=True,phase='complete',finished=time.time(),analysis_sha256=sha(out/'analysis.json'));snapshot()
    except BaseException as error:
        state.update(complete=False,phase='failed',error=repr(error),finished=time.time())
        for job in all_jobs(state):
            if job['status']=='pending':job['status']='not_started'
        snapshot();raise
    finally:signal.signal(signal.SIGTERM,previous)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze')
    for name in ('out','preparation','binary','source-bundle','repository'):f.add_argument('--'+name,type=Path,required=True)
    f.add_argument('--reference-receipt',type=Path,action='append',required=True)
    for action in ('preflight','run'):
        x=sub.add_parser(action);x.add_argument('--out',type=Path,required=True);x.add_argument('--expected-protocol-sha256',required=True)
    a=parser.parse_args()
    if a.action=='freeze':
        freeze(a.out,a.preparation,a.binary,a.source_bundle,a.repository,a.reference_receipt);print(sha(a.out/'protocol.json'))
    elif a.action=='preflight':validate(a.out,a.expected_protocol_sha256);print('preflight passed; no jobs launched')
    else:print(run(a.out,a.expected_protocol_sha256)['phase'])


if __name__=='__main__':main()
