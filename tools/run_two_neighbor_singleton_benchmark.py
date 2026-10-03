#!/usr/bin/env python3
"""Bounded, exclusive three-process controller for the matched fused/unfused singleton benchmark."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from prepare_two_neighbor_singleton_benchmark import LIMITS, require, read, record, runtime, sha, verify, write
from run_evolving_dimer_analysis import drain


def save(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def checked(x):
    p=Path(x['path']).resolve();require(sha(p)==x['sha256'],'Changed binding '+str(p));return p


def review_inputs(base,review_path):
    c=verify(base);review=read(review_path)
    require(review.get('complete') is True and review.get('passed') is True,'Review incomplete or failed')
    required={str((base/p).resolve()) for p in ('config.json','freeze.json','protocol.json','binding.json','run-binding.json',
        'analysis-plan.json','common/evolving_dimer_benchmark',
        'common/source/tools/run_two_neighbor_singleton_benchmark.py','common/source/tools/prepare_two_neighbor_singleton_benchmark.py',
        'common/source/tools/run_evolving_dimer_analysis.py')}
    require(required<=set(review['input_sha256']),'Review omits execution closure')
    for p,h in review['input_sha256'].items():require(sha(p)==h,'Reviewed input changed '+p)
    b=read(base/'binding.json');r=read(base/'run-binding.json')
    protocol=read(base/'protocol.json')
    require(b['isolated_build_receipt']==protocol['validation_receipts']['build'],
        'Isolated build receipt differs from frozen protocol')
    build=read(checked(b['isolated_build_receipt']))
    require(build['passed'] is True and build['source_unchanged'] is True
        and build['production_unchanged'] is True
        and build['executable']['sha256']==b['executable_sha256']
        and build['compiled_source_bundle']['sha256']==b['compiled_source_bundle_sha256']
        and build['source_sha256']['examples/evolving_dimer_benchmark.rs']==b['example_source_sha256'],
        'Bound executable differs from validated isolated build')
    for k,p in [('config_sha256','config.json'),('protocol_sha256','protocol.json'),('freeze_sha256','freeze.json'),
                ('executable_sha256','common/evolving_dimer_benchmark'),('compiled_source_bundle_sha256','common/source-bundle.json'),
                ('example_source_sha256','common/source/examples/evolving_dimer_benchmark.rs')]:
        require(b[k]==sha(base/p),'Executable binding mismatch '+k)
    require(r['complete'] is True and r['config_sha256']==sha(base/'config.json') and r['prelaunch_binding']==record(base/'binding.json')
        and r['prepared_manifest']==c['inherited_campaign']['prepared_manifest'],'Run binding differs')
    checked(r['prepared_manifest'])
    for p,h in r['prepared_files'].items():require(sha(p)==h,'Prepared file changed '+p)
    require(read(base/'freeze.json')['runtime']==runtime(),'Preparation and launch runtime differ')
    return c


def launch(base,review_path,workers=3):
    base,review_path=Path(base).resolve(),Path(review_path).resolve()
    require(type(workers) is int and 1<=workers<=3,'At most three workers')
    c=review_inputs(base,review_path)
    directory=base/'dispatch';directory.mkdir()
    frozen=base/'common/source/tools/run_two_neighbor_singleton_benchmark.py'
    require(sha(frozen)==sha(__file__),'Live/frozen controller differ')
    require(not Path(c['output']).exists(),'Execution output already exists; no restart')
    plan=dict(schema='two-neighbor-singleton-dispatch-v1',base=str(base),jobs=c['jobs'],workers=workers,
        limits=LIMITS,review=record(review_path),runtime=runtime(),source=record(frozen),
        input_sha256={str(base/p):sha(base/p) for p in ('config.json','binding.json','run-binding.json','freeze.json','common/evolving_dimer_benchmark')},
        retries=False,replacements=False,physical_chains=64,
        failure='Stop new launches; kill and reap every owned active process group; preserve all partial outputs and unstarted jobs.')
    write(directory/'plan.json',plan)
    child=None
    handlers={s:signal.signal(s,terminate_requested) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        with (directory/'controller.log').open('xb') as log:
            # A stop must reach this process only after it owns the detached
            # controller handle. Restore the launcher's prior mask in the child
            # before exec, so the controller can receive its cleanup signal.
            prior_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
            try:
                child=subprocess.Popen([sys.executable,'-B',str(frozen),'--run','--root',str(base)],
                    cwd=base/'common/source',stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,prior_mask))
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,prior_mask)
        write(directory/'launch.json',dict(pid=child.pid,started=time.time(),plan_sha256=sha(directory/'plan.json')))
        return dict(pid=child.pid,chains=64,workers=workers,plan=record(directory/'plan.json'))
    except BaseException:
        if child is not None:
            # Workers have separate process groups. Killing the controller
            # group would orphan them: request its existing graceful drain and
            # reap it before surfacing the launch/publication failure.
            for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,signal.SIG_IGN)
            child.terminate()
            while True:
                try:
                    child.wait()
                    break
                except (KeyboardInterrupt,InterruptedError):
                    continue
        raise
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)


def terminate_requested(signum,_frame):
    raise SystemExit('Controller received signal '+str(signum))


def run(base):
    base=Path(base).resolve();out=base/'dispatch';plan=read(out/'plan.json');plan_hash=sha(out/'plan.json')
    require(plan['schema']=='two-neighbor-singleton-dispatch-v1' and plan['limits']==LIMITS
        and plan['runtime']==runtime() and plan['retries'] is False and plan['replacements'] is False,'Dispatch contract changed')
    require(checked(plan['source'])==Path(__file__).resolve(),'Executed controller differs')
    c=review_inputs(base,checked(plan['review']))
    require(plan['jobs']==c['jobs'] and len(c['jobs'])==64 and 1<=plan['workers']<=3,'Dispatch job allocation differs')
    for p,h in plan['input_sha256'].items():require(sha(p)==h,'Dispatch input changed '+p)
    write(out/'claimed.json',dict(pid=os.getpid(),started=time.time(),plan_sha256=plan_hash,retry=False))
    queue=list(plan['jobs']);active={};completed=[];failure=None
    handlers={s:signal.signal(s,terminate_requested) for s in (signal.SIGINT,signal.SIGTERM)}
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RAYON_NUM_THREADS='1')
    def status(done=False):
        save(out/'status.json',dict(schema='two-neighbor-singleton-dispatch-status-v1',complete=done,
            passed=done and failure is None and len(completed)==64 and all(v['success'] for v in completed),
            failure=failure,failure_draining=failure is not None and bool(active),completed=completed,
            active=[dict(job=a['job'],pid=a['child'].pid,started=a['started']) for a in active.values()],
            unstarted=queue,updated=time.time(),plan_sha256=plan_hash))
    def finish(identity,error=None):
        a=active[identity];child=a['child'];terminal=Path(c['output'])/f'job-{identity:03}'/'terminal.json'
        termsha=None;success=False
        try:
            if terminal.is_file():
                termsha=sha(terminal);t=read(terminal)
                counts=t.get('counts',{})
                valid=(t.get('complete') is True and t.get('conditional_target') is True and t.get('job')==a['job']
                    and t.get('blocks')==4608 and t.get('config_sha256')==plan['input_sha256'][str(base/'config.json')]
                    and t.get('binding_sha256')==plan['input_sha256'][str(base/'run-binding.json')]
                    and counts.get('local_attempted')==18432 and counts.get('singleton_attempted')==4608
                    and all(type(counts.get(k)) is int and 0<=counts[k]<=n for k,n in
                        [('local_accepted',18432),('singleton_accepted',4608),('singleton_self_loop',4608)])
                    and counts.get('singleton_accepted',4609)+counts.get('singleton_self_loop',4609)<=4608
                    and not (terminal.parent/'failure.json').exists())
                success=error is None and child.returncode==0 and valid
                if not valid:error='Terminal provenance, schedule or completion differs'
        except Exception as e:error=repr(e)
        a['log'].close()
        result=dict(job=a['job'],pid=child.pid,started=a['started'],finished=time.time(),returncode=child.returncode,
            success=success,terminal_sha256=termsha,error=error,child_drained=child.returncode is not None)
        write(out/f'job-{identity:03}-exit.json',result);completed.append(result);del active[identity]
        return success
    try:
        status()
        while queue or active:
            while queue and len(active)<plan['workers']:
                job=queue[0];identity=job['id'];log=None;child=None;started=time.time();prior_mask=None
                argv=[str(base/'common/evolving_dimer_benchmark'),'--mode','run','--config',str(base/'config.json'),
                    '--binding',str(base/'run-binding.json'),'--job',str(identity)]
                write(out/f'job-{identity:03}-begin.json',dict(job=job,argv=argv,started=started,limits=LIMITS))
                queue.pop(0)
                try:
                    log=(out/f'job-{identity:03}.log').open('xb')
                    def limits():
                        resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['cpu_seconds']+1))
                        resource.setrlimit(resource.RLIMIT_AS,(LIMITS['address_space_bytes'],LIMITS['address_space_bytes']))
                        os.nice(10);signal.pthread_sigmask(signal.SIG_SETMASK,prior_mask)
                    prior_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
                    try:
                        child=subprocess.Popen(argv,cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT,
                            start_new_session=True,preexec_fn=limits)
                        active[identity]=dict(child=child,job=job,log=log,started=started,monotonic=time.monotonic())
                    finally:signal.pthread_sigmask(signal.SIG_SETMASK,prior_mask)
                except BaseException as e:
                    if child is None:
                        if log is not None:log.close()
                        result=dict(job=job,pid=None,started=started,finished=time.time(),returncode=None,success=False,
                            error=repr(e),child_drained=True)
                        write(out/f'job-{identity:03}-exit.json',result);completed.append(result)
                    raise
                status()
            for identity,a in list(active.items()):
                code=a['child'].poll()
                if code is None:
                    require(time.monotonic()-a['monotonic']<=LIMITS['wall_seconds'],'Chain wall limit exceeded '+str(identity))
                elif not finish(identity):raise RuntimeError('Chain failed '+str(identity))
            status()
            if active:time.sleep(.2)
        require(sha(out/'plan.json')==plan_hash,'Dispatch plan changed')
        review_inputs(base,checked(plan['review']))
    except BaseException as e:
        failure=repr(e);status()
        raise
    finally:
        # No exception, including a second stop signal, may orphan an owned group.
        saved={s:signal.signal(s,signal.SIG_IGN) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            errors=[]
            for a in active.values():
                try:drain(a['child'])
                except BaseException as e:errors.append(repr(e))
            # A transient cleanup failure must not prevent every other owned
            # child from being drained. Retry cleanup only for still-live
            # children; this never retries or replaces a scientific chain.
            for a in active.values():
                if a['child'].returncode is None:
                    try:drain(a['child'])
                    except BaseException as e:errors.append(repr(e))
            if errors:
                failure=failure or 'Owned child drain error: '+repr(errors)
            if any(a['child'].returncode is None for a in active.values()):
                status()
                raise RuntimeError('Owned child remains undrained: '+repr(errors))
            # Drain every child before any receipt write can fail.
            for identity in list(active):
                finish(identity,error=failure or 'Controller final drain')
            status(done=True)
        finally:
            for a in active.values():a['log'].close()
            for s,h in handlers.items():signal.signal(s,h)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--review',type=Path);p.add_argument('--workers',type=int,default=3);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.run:run(a.root)
    else:
        p.error('--review required') if a.review is None else None
        print(json.dumps(launch(a.root,a.review,a.workers),indent=2))
if __name__=='__main__':main()
