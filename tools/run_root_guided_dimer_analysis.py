#!/usr/bin/env python3
"""Admit the completed root_m4 extension and its independent audit, then observe once.

Preparation binds metadata/bytes only. The single explicitly launched child is
the campaign's frozen observer: 32 new trajectories, 32 cached m4 controls.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

import run_evolving_dimer_analysis as launcher
import run_evolving_dimer_root_guidance_audit as guidance

require,read,sha=launcher.require,launcher.read,launcher.sha
SCHEMA='root-guided-dimer-observer-execution-v1'
SCOPE=dict(maximum_workers=1,cpu_limit_seconds=1800,wall_limit_seconds=3600,
    address_space_limit_bytes=16*1024**3,new_chains=32,reused_control_chains=32,
    expected_new_retained_endpoints=147488,expected_new_production_endpoints=131072,
    maximum_new_pair_classifications=77431200,new_physical_draws=0,old_geometry_queries=0,
    retries=0,partial_analysis_allowed=False,native_observer=False)


def write(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def record(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha(p))


def runtime():
    return dict(python=sys.version,executable=record(sys.executable),
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy')},
        threads={n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')})


def observer_contract(base):
    p=read(Path(base)/'analysis-plan.json')
    require(p['schema']=='root-guided-dimer-analysis-plan-v1'
        and p['new_chains']==32 and p['reused_control_chains']==32
        and p['new_retained_initial_observations']==147488 and p['new_production_observations']==131072
        and p['maximum_new_pair_classifications']==77431200
        and p['new_geometry_only'] is True and p['new_physical_draws']==0
        and p['native_observer'] is False and p['complete_inventory_required'] is True,
        'Frozen observer allocation differs')
    return p


def admit(base,audit_root):
    """Metadata/hash gate; no saved row or atom geometry is interpreted."""
    base,audit_root=Path(base).resolve(),Path(audit_root).resolve()
    current=guidance.campaign_bindings(base);files=dict(current['files'])
    def bind(path,expected=None):
        item=record(path);require(expected is None or item['sha256']==expected,'Changed audit evidence '+str(path))
        require(item['path'] not in files or files[item['path']]==item['sha256'],'Input changed during admission')
        files[item['path']]=item['sha256'];return Path(item['path'])
    require(not any((audit_root/n).exists() for n in ('failure.json','worker-failure.json','preparation-failure.json')),
        'Failed or interrupted guidance audit')
    plan_path=bind(audit_root/'execution-plan.json');plan=read(plan_path)
    require(plan['base']==str(base),'Guidance audit belongs to a different campaign')
    guidance.verify_plan(audit_root,plan)
    require(plan['jobs']==current['jobs'] and all(plan['files'].get(p)==h for p,h in current['files'].items()),
        'Guidance audit does not cover these completed trajectories')
    summary=read(bind(audit_root/'summary.json'))
    require(summary.get('schema')=='evolving-dimer-root-guidance-audit-summary-v1'
        and summary.get('base')==str(base) and summary.get('complete') is True and summary.get('passed') is True
        and summary.get('plan_sha256')==files[str(plan_path)] and summary.get('chains')==32
        and summary.get('scalar_dimer_attempts')==147456 and summary.get('scalar_local_attempts')==589824
        and summary.get('independent_geometry_events')==32
        and summary.get('new_poses')==summary.get('new_clouds')==0,
        'Matching complete independent guidance audit required')
    result=read(bind(audit_root/'audit.json',summary['audit_sha256']))
    require(result['complete'] is True and result['passed'] is True and result['plan_sha256']==files[str(plan_path)]
        and [r['job'] for r in result['chains']]==[j['job'] for j in current['jobs']]
        and result['scalar_dimer_attempts']==147456 and result['scalar_local_attempts']==589824
        and result['independent_geometry_events']==32
        and result['new_poses']==result['new_clouds']==0,
        'Guidance result inventory or scope differs')
    require(result['count_queries_started']==result['count_queries_completed']==summary['count_queries']<=2560
        and 0<=result['count_queries_completed'] and 0<=result['point_membership_tests_started']<=41943040,
        'Guidance count allocation differs')
    bind(audit_root/'attempts.jsonl',result['journal_sha256'])
    require(summary['journal_sha256']==result['journal_sha256'],'Guidance journal evidence differs')
    claim=read(bind(audit_root/'claim.json'));review=read(bind(audit_root/'review.json'))
    require(claim['plan_sha256']==files[str(plan_path)] and claim['review_sha256']==files[str(audit_root/'review.json')]
        and review['complete'] is True and review['passed'] is True and review['plan_sha256']==files[str(plan_path)],
        'Guidance launch review/claim differs')
    exit_status=read(bind(audit_root/'child/exit.json'))
    require(exit_status['child_started'] is True and exit_status['child_drained'] is True
        and exit_status['returncode']==0 and exit_status['error'] is None,'Guidance child did not complete cleanly')
    for p,h in plan['files'].items():bind(p,h)
    for p,h in files.items():require(sha(p)==h,'Admission input changed '+p)
    return files


def prepare(base,root,guidance_audit_root):
    base,root,audit_root=map(lambda p:Path(p).resolve(),(base,root,guidance_audit_root))
    require(not root.exists(),'Fresh postrun observer directory required; no retry')
    files=admit(base,audit_root)
    observer_contract(base)
    sources=source_closure()
    root.mkdir();(root/'code').mkdir()
    for name,path in sources.items():
        target=root/'code'/name;shutil.copyfile(path,target)
        require(sha(target)==sha(path),'Wrapper source changed during archive')
        files[str(target)]=sha(target)
    python=str(Path(sys.executable).absolute());files[python]=sha(python)
    plan=dict(schema=SCHEMA,base=str(base),root=str(root),guidance_audit_root=str(audit_root),
        argv=[python,'-B',str(base/'common/source/tools/analyze_root_guided_dimer_benchmark.py'),
            '--base',str(base),'--output',str(root/'analysis')],
        source_sha256={n:sha(root/'code'/n) for n in sources},files=files,runtime=runtime(),**SCOPE)
    write(root/'execution-plan.json',plan)
    return plan


def source_closure():
    """Only wrapper/metadata launcher modules; scientific code remains archived."""
    paths=[Path(__file__).resolve(),Path(launcher.__file__).resolve(),Path(guidance.__file__).resolve(),
        Path(__file__).resolve().parent/'test_run_root_guided_dimer_analysis.py']
    # Guidance's metadata controller explicitly declares its local helper closure.
    # Its campaign worker is NOT imported or executed during preparation.
    import ast
    result={};pending=paths[:]
    while pending:
        p=pending.pop()
        if p.name in result:
            require(result[p.name]==p,'Ambiguous wrapper module name');continue
        result[p.name]=p
        for node in ast.walk(ast.parse(p.read_text(),filename=str(p))):
            names=([a.name for a in node.names] if isinstance(node,ast.Import) else
                [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                candidate=p.parent/(name.split('.')[0]+'.py')
                if candidate.is_file():pending.append(candidate.resolve())
    return result


def verify_plan(root,plan):
    root=Path(root).resolve();base=Path(plan['base']).resolve()
    require(plan['schema']==SCHEMA and plan['root']==str(root) and all(plan.get(k)==v for k,v in SCOPE.items()),
        'Changed observer scope or budget')
    python=str(Path(sys.executable).absolute())
    require(plan['runtime']==runtime(),'Changed runtime')
    require(plan['argv']==[python,'-B',str(base/'common/source/tools/analyze_root_guided_dimer_benchmark.py'),
        '--base',str(base),'--output',str(root/'analysis')],'Changed observer command')
    current=admit(base,Path(plan['guidance_audit_root']))
    require(all(plan['files'].get(p)==h for p,h in current.items()),'Omitted or changed admission input')
    sources=source_closure()
    require(set(plan['source_sha256'])==set(sources),'Changed wrapper dependency inventory')
    for name,path in sources.items():
        archived=root/'code'/name
        require(plan['files'].get(str(archived))==plan['source_sha256'][name]==sha(archived)==sha(path),
            'Changed wrapper source '+name)
    require(plan['files'].get(python)==sha(python),'Unbound Python executable')
    for p,h in plan['files'].items():require(sha(p)==h,'Changed execution input '+p)
    observer_contract(base)


def verify_output(base,root,plan):
    result_path=root/'analysis/analysis.json';r=read(result_path)
    require(r['schema']=='root-guided-dimer-analysis-v1' and r['complete'] is True
        and r['new_chains']==32 and r['reused_control_chains']==32 and len(r['chains'])==64
        and r['new_geometry_endpoints']==147488 and r['new_physical_draws']==0 and r['native_observer'] is False,
        'Incomplete or out-of-scope observer result')
    c=read(base/'config.json');new=[v for v in r['chains'] if v['reused_control'] is False]
    old=[v for v in r['chains'] if v['reused_control'] is True]
    require([v['job'] for v in new]==c['jobs'] and [v['job'] for v in old]==[dict(j,arm='m4') for j in c['jobs']],
        'Observer changed matched chain inventory')
    require(sum(v['metrics']['production_samples'] for v in new)==131072
        and all(v['metrics']['production_samples']==4096 for v in new+old)
        and all(v['new_geometry_queries']==0 for v in old),'Lost production samples or reused-control geometry')
    require(sum(v['pair_classifications'] for v in new)<=77431200,'Exceeded geometry allocation')
    require(r['analysis_plan']==read(base/'analysis-plan.json'),'Observer plan differs')
    manifest=read(root/'analysis/manifest.json')
    expected={'analysis.json','input-binding.json'}|{f"job-{j['id']:03}-observations.jsonl" for j in c['jobs']}
    require(manifest['complete'] is True and set(manifest['files'])==expected,'Observer artifact inventory differs')
    for name,h in manifest['files'].items():require(sha(root/'analysis'/name)==h,'Changed observer artifact '+name)
    return record(result_path),record(root/'analysis/manifest.json')


def run(root):
    root=Path(root).resolve();plan=read(root/'execution-plan.json');digest=sha(root/'execution-plan.json')
    verify_plan(root,plan)
    require(sha(root/'execution-plan.json')==digest,'Plan changed during admission')
    write(root/'claim.json',dict(pid=os.getpid(),started=time.time(),plan_sha256=digest,retries=0))
    handlers={s:signal.signal(s,launcher.terminate_requested) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        require(not(root/'analysis').exists() and not(root/'failure.json').exists(),'Prior observer outputs exist; no restart')
        write(root/'begin.json',dict(argv=plan['argv'],plan_sha256=digest,started=time.time(),**SCOPE))
        launcher.owned_child(plan['argv'],str(Path(plan['base'])/'common/source'),root,1800,3600,16*1024**3)
        result,manifest=verify_output(Path(plan['base']),root,plan)
        verify_plan(root,plan);require(sha(root/'execution-plan.json')==digest,'Plan changed during observation')
        write(root/'summary.json',dict(schema='root-guided-dimer-observer-completion-v1',complete=True,passed=True,
            base=plan['base'],analysis=result,manifest=manifest,plan_sha256=digest,
            guidance_audit_root=plan['guidance_audit_root'],**SCOPE))
    except BaseException as error:
        write(root/'failure.json',dict(complete=False,passed=False,error=repr(error),traceback=traceback.format_exc(),
            plan_sha256=digest,retries=0,partial_outputs_retained=True))
        raise
    finally:
        for s,h in handlers.items():signal.signal(s,h)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path);p.add_argument('--guidance-audit-root',type=Path);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:run(a.root)
    else:
        p.error('--base and --guidance-audit-root required for preparation') if a.base is None or a.guidance_audit_root is None else None
        plan=prepare(a.base,a.root,a.guidance_audit_root)
        print(json.dumps(dict(prepared=True,files=len(plan['files']),new_geometry_started=False)))
if __name__=='__main__':main()
