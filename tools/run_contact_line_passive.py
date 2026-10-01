#!/usr/bin/env python3
"""Freeze and drain the fixed proposal-only contact-line diagnostic, one CPU.

No Poisson clouds or physical contact-mass estimates are generated.
"""
from __future__ import annotations
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
from prepare_contact_line_passive import SEEDS, COVERAGE_SEED, GUIDE_SHA, REGION_SHA, SHAPE_SHA, WIDTHS
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_confirmation import runtime, worker_environment
from run_contact_tail_pilot import verify_frozen, copy_frozen
from run_mobile_posterior_pilot import verify_bundle
from run_full_vessel_comparison import execute_group
from run_smc_importance_bridge import process_token

SCHEMA='contact-line-passive-controller-v1'


def validate_design(plan):
    require(plan['fresh_draws']==512 and plan['archived_probe_queries']==156 and
            plan['maximum_CPU_workers']==1 and plan['new_Poisson_clouds']==0,'Passive budget changed')
    require(plan['axes']==[0] and plan['widths_A']==WIDTHS and plan['conditional_probability']==.5 and
            plan['minimum_conditional_mass']==1e-12,'Frozen proposal choices changed')
    jobs=plan['jobs']
    require(len(jobs)==8 and [j['seed'] for j in jobs]==SEEDS,'Passive streams changed')
    for index,job in enumerate(jobs):
        require(job['arm']==('baseline92' if index<4 else 'line92') and job['id']==f'r{index%4:02}',
                'Passive stream order changed')
        require(job['fresh_proposal_draws']==64 and job['archived_probe_queries']==(78 if index%4==0 else 0),
                'Passive per-stream allocation changed')
    coverage=plan['coverage_job']
    require(plan['additional_coverage_queries']==128 and coverage['arm']=='line92' and
            coverage['id']=='coverage' and coverage['fresh_proposal_draws']==0 and
            coverage['archived_probe_queries']==128 and coverage['seed']==COVERAGE_SEED and
            coverage['probe_file']=='coverage-probes.jsonl','Breadth control changed')


def command(out, job):
    common=out/'common'
    cmd=[str(common/'contact-line-guide-audit'),'--config',str(common/'config.json'),
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
    from analyze_contact_line_audit import audit  # dependency closure; no execution here
    sources=local_dependencies([Path(__file__)])
    out.mkdir();common=out/'common';common.mkdir();(out/'logs').mkdir();(out/'runs').mkdir()
    copy_frozen(preparation,common/'preparation')
    for name,path in sources.items():shutil.copy2(path,common/name)
    shutil.copy2(binary,common/'contact-line-guide-audit');shutil.copy2(bundle,common/'source-bundle.json')
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
    for arm,beta in [('baseline92',0.),('line92',.5)]:
        guide=read(preparation/'guides'/f'{arm}.json')
        require(guide['contact_neighbor_indices']==[0,1] and guide['raw_translation_axes']==[0] and
                guide['conditional_probability']==beta and guide['contact_widths_A']==WIDTHS and
                guide['minimum_conditional_mass']==1e-12 and guide['defensive_uniform_shell_probability']==.5,
                'Conditioning guide differs from frozen design')
        base=read(common/'original92.json')
        require(guide['gaussian_components']==base['gaussian_components'] and len(base['gaussian_components'])==92,
                'Gaussian components changed')
        shutil.copy2(preparation/'guides'/f'{arm}.json',common/f'{arm}.json')
    cfg=read(preparation/'common/config.json');cfg['shape']=str(common/'shape.json');write_new(common/'config.json',cfg)
    require(sha(common/'shape.json')==SHAPE_SHA and sha(common/'region.json')==REGION_SHA and
            sha(common/'original92.json')==GUIDE_SHA,'Original target/guide identity changed')
    jobs=[]
    for source_job in plan['jobs']:
        job={k:copy.deepcopy(v) for k,v in source_job.items() if k!='command'}
        job.update(directory=str(out/'runs'/job['arm']/job['id']),log=str(out/'logs'/(job['arm']+'-'+job['id']+'.log')),
            kind='proposal_only',status='pending')
        (out/'runs'/job['arm']).mkdir(exist_ok=True);job['command']=command(out,job);jobs.append(job)
    coverage={k:copy.deepcopy(v) for k,v in plan['coverage_job'].items() if k!='command'}
    coverage.update(directory=str(out/'runs/line92/coverage'),log=str(out/'logs/line92-coverage.log'),
        kind='proposal_only',status='pending');coverage['command']=command(out,coverage)
    protocol=copy.deepcopy(plan);protocol.update(schema=SCHEMA,jobs=jobs,repository=str(repository),
        coverage_job=coverage,
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
    common=out/'common';verify_bundle(common/'contact-line-guide-audit',common/'source-bundle.json',common/'rust-source')
    require(sha(common/'contact-line-guide-audit')==p['binary_sha256'],'Frozen executable changed')
    preparation=Path(p['preparation']);verify_frozen(preparation)
    require(sha(preparation/'plan.json')==p['preparation_sha256'] and
            sha(preparation/'freeze.json')==p['preparation_freeze_sha256'],'Preparation binding changed')
    for job in p['jobs']:require(job['command']==command(out,job),'Frozen command changed')
    require(p['coverage_job']['command']==command(out,p['coverage_job']),'Frozen coverage command changed')
    return p


def cpu_children():
    usage=resource.getrusage(resource.RUSAGE_CHILDREN);return usage.ru_utime+usage.ru_stime


def verify_output(directory,job,p):
    manifest=read(directory/'manifest.json');summary=read(directory/'summary.json')
    common=directory.parents[2]/'common'
    expected=dict(schema='contact-line-guide-audit-v1',samples=job['fresh_proposal_draws'],seed=job['seed'],
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
    return summary


def summarize(out,state):
    arms={};probes={}
    for arm in ('baseline92','line92'):
        populations=[]
        for job in state['jobs']:
            if job['arm']!=arm:continue
            directory=Path(job['directory']);rows=[json.loads(x) for x in (directory/'samples.jsonl').read_text().splitlines()]
            total=read(directory/'summary.json')
            valid=[r['hard_valid'] and r['shell_valid'] and r['capture_valid'] for r in rows]
            joint=[sum(bool(ok and all(r['width_contacts'][i])) for r,ok in zip(rows,valid)) for i in range(3)]
            fresh_cpu=total['fresh_total_cpu_seconds'];probe_cpu=total['probe_total_cpu_seconds']
            setup_overhead=job['child_cpu_seconds']-fresh_cpu-probe_cpu
            populations.append(dict(id=job['id'],seed=job['seed'],draws=len(rows),hard_shell_capture=sum(valid),
                shell=sum(r['shell_valid'] for r in rows),hard=sum(r['hard_valid'] for r in rows),joint_contact_by_width=joint,
                conditional=total['conditioned_draws'],fallback=total['fallback_draws'],
                fresh_cpu_seconds=fresh_cpu,draw_cpu_seconds=total['draw_cpu_seconds'],density_cpu_seconds=total['density_cpu_seconds'],
                probe_cpu_seconds=probe_cpu,full_child_cpu_seconds=job['child_cpu_seconds'],setup_and_unattributed_cpu_seconds=setup_overhead,
                independent_audit_cpu_seconds=job['audit_cpu_seconds'],output_sha256=job['output_sha256']))
            for line in (directory/'probes.jsonl').read_text().splitlines():
                row=json.loads(line);probes.setdefault(row['id'],{})[arm]=row['log_proposal_density']
        cpu=sum(r['fresh_cpu_seconds'] for r in populations)
        arms[arm]=dict(populations=populations,fresh_attempts=sum(r['draws'] for r in populations),
            total_fresh_cpu_seconds=cpu,total_probe_cpu_seconds=sum(r['probe_cpu_seconds'] for r in populations),
            total_child_cpu_seconds=sum(r['full_child_cpu_seconds'] for r in populations),
            hard_shell_capture_per_fresh_cpu=sum(r['hard_shell_capture'] for r in populations)/cpu,
            joint_by_width_per_fresh_cpu=[sum(r['joint_contact_by_width'][i] for r in populations)/cpu for i in range(3)],
            joint_fraction_populations=[[r['joint_contact_by_width'][i]/r['draws'] for r in populations] for i in range(3)],
            scope='Fresh CPU includes draw, complete density and endpoint geometry; construction and probes are reported separately, not hidden in the rate.')
    archived=read(out/'common/preparation/axis-diagnostics.json')['rows'];moments={}
    for arm in ('baseline','expanded'):
        source=[r for r in archived if r['source']['arm']==arm]
        # Only observed saved-row contributions, never a new physical mass estimate.
        terms={name:np.array([sum(r['source']['paired_log_weights'])+r['source']['log_q']-probes[r['id']][name]
               for r in source]) for name in ('baseline92','line92')}
        logs={name:float(logsumexp(x)) for name,x in terms.items()}
        moments[arm]=dict(rows=len(source),line_to_baseline_paired_M2_ratio=math.exp(logs['line92']-logs['baseline92']),
            contribution_ESS={name:float(math.exp(2*logs[name]-logsumexp(2*x))) for name,x in terms.items()},
            largest_fraction={name:float(math.exp(max(x)-logs[name])) for name,x in terms.items()},
            scope='Retrospective contribution ratio on the 78 saved competing55 rows; not a new normalizer or unseen-tail bound.')
    selection={r['id']:r for r in read(out/'common/coverage-selection.json')['rows']}
    coverage_rows=[json.loads(x) for x in (Path(state['coverage_job']['directory'])/'probes.jsonl').read_text().splitlines()]
    require(len(coverage_rows)==128 and {r['id'] for r in coverage_rows}==set(selection),'Coverage identities changed')
    coverage={}
    for name in sorted({r['coverage_class'] for r in selection.values()}):
        rows=[r for r in coverage_rows if selection[r['id']]['coverage_class']==name]
        gains=np.array([r['log_proposal_density']-selection[r['id']]['original92_log_q'] for r in rows])
        coverage[name]=dict(rows=len(rows),log_q_ratio_min=float(min(gains)),log_q_ratio_median=float(np.median(gains)),
            log_q_ratio_max=float(max(gains)),hard_shell_capture=sum(r['hard_valid'] and r['shell_valid'] and r['capture_valid'] for r in rows),
            scope='Uniform saved-row subset within each original class/population; not an equilibrium mass estimator')
    return dict(schema=SCHEMA,complete=True,arms=arms,archived_moment_diagnostics=moments,coverage=coverage,
        coverage_job=state['coverage_job'],fresh_attempts=512,archived_queries=156,additional_coverage_queries=128,
        new_Poisson_clouds=0,physical_mass_estimates_generated=0,
        full_vessel_gate_open=False,assembly_gate_open=False,protocol_sha256=state['protocol_sha256'])


def run(out,expected):
    out=Path(out).resolve();p=validate(out,expected)
    require(not (out/'status.json').exists(),'No passive retries or repeated launch')
    state=dict(schema=SCHEMA,complete=False,phase='starting',protocol_sha256=expected,started=time.time(),
        jobs=copy.deepcopy(p['jobs']),coverage_job=copy.deepcopy(p['coverage_job']),physical_jobs_launched=0,new_Poisson_clouds=0)
    def snapshot():
        (out/'status.tmp').write_text(json.dumps(state,indent=2)+'\n');(out/'status.tmp').replace(out/'status.json')
    with (out/'launch-claim.json').open('x') as stream:json.dump(dict(pid=os.getpid(),process_birth=process_token(os.getpid()),protocol_sha256=expected),stream)
    with (Path(p['preparation'])/'execution-claim.json').open('x') as stream:json.dump(dict(output=str(out),protocol_sha256=expected),stream)
    snapshot()
    def interrupted(signum,frame):raise InterruptedError('Passive controller signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:
        from analyze_contact_line_audit import audit
        for job in [*state['jobs'],state['coverage_job']]:
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
        for job in [*state['jobs'],state['coverage_job']]:
            require(all(sha(Path(job['directory'])/name)==digest for name,digest in job['output_sha256'].items()),
                    'Completed audited rows changed before aggregation')
        analysis=summarize(out,state);write_new(out/'analysis.json',analysis)
        state.update(complete=True,phase='complete',finished=time.time(),analysis_sha256=sha(out/'analysis.json'));snapshot()
    except BaseException as error:
        state.update(complete=False,phase='failed',error=repr(error),finished=time.time())
        for job in [*state['jobs'],state['coverage_job']]:
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
