#!/usr/bin/env python3
"""Execute a frozen toy export/audit sequence after its physical prerequisite.

The paired SMC workflow reserves at most four jobs after this same prerequisite;
this workflow adds one sequential toy job, staying below eight physical jobs.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import scipy
from run_r4_smc_controls import THREAD_ENV, prerequisite_state, process_token
from analyze_r4_smc_control import Ledger, read, require, sha, write


def run(root):
    root=Path(root).resolve();ledger=Ledger();ledger.frozen(root);plan=read(root/'plan.json')
    require(plan['schema']=='full-vessel-reference-bridge-plan-v1','Unknown frozen toy bridge')
    require(plan['python_version']==sys.version and plan['python_sha256']==sha(sys.executable)
        and sys.flags.optimize==0 and plan['numerical_runtime']==dict(numpy=np.__version__,scipy=scipy.__version__),'Frozen audit runtime differs')
    require(sha(__file__)==plan['sources']['run_vessel_reference_bridge.py'],'Use the frozen runner')
    for name,digest in plan['sources'].items():ledger.bind(root/'common'/name,digest)
    dependency=plan['dependency']
    for name,digest in dependency['source_bindings'].items():ledger.bind(name,digest)
    require(plan['physical_workers']==1 and plan['known_other_followup_physical_cap']==4,'Worker allocation changed')
    for step in plan['steps']:require(not Path(step['output']).exists(),'Existing reference output; no replay')
    with (root/'status.json').open('x') as stream:stream.write('{}\n')
    state=dict(schema='full-vessel-reference-bridge-status-v1',complete=False,phase='waiting_for_prerequisite',
        pid=os.getpid(),started=time.time(),plan_sha256=sha(root/'plan.json'),steps=[])
    def snapshot():
        write(root/'status.tmp',state);(root/'status.tmp').replace(root/'status.json')
    snapshot();env=dict(os.environ,**THREAD_ENV,PYTHONOPTIMIZE='0')
    try:
        status=Path(dependency['directory'])/'status.json'
        while prerequisite_state(read(status),process_token(dependency['pid'])==dependency['process_birth'])=='waiting':time.sleep(30)
        actual=read(status)['steps']
        require(len(actual)==len(dependency['steps']) and all(all(a[k]==v for k,v in s.items()) for a,s in zip(actual,dependency['steps'])),
                'Prerequisite command identities changed')
        ledger.recheck();state.update(phase='references',prerequisite_status_sha256=sha(status));snapshot()
        for step in plan['steps']:
            require(not Path(step['output']).exists(),'Existing step output; no replay')
            item=dict(step,started=time.time(),returncode=None);state['steps'].append(item);snapshot()
            with (root/(step['id']+'.log')).open('xb') as log:
                result=subprocess.run(step['command'],cwd=plan['repository'],env=dict(env,**step.get('environment',{})),stdout=log,stderr=subprocess.STDOUT)
            item.update(returncode=result.returncode,finished=time.time());snapshot();result.check_returncode()
        fixtures=read(Path(plan['export'])/'fixture-index.json')
        require(fixtures['complete'] and fixtures['draws']==768 and len(fixtures['fixtures'])==6,'Reference export allocation differs')
        require(fixtures['test_executable_sha256']==sha(root/'test-executable'),'Reference executable differs')
        audits=[]
        for record in fixtures['fixtures']:
            output=Path(record['output']);require(sha(output/'samples.jsonl')==record['samples_sha256']
                and sha(output/'manifest.json')==record['manifest_sha256'],'Fixture raw outputs changed')
            report=Path(plan['audit_root'])/record['name']/'analysis.json';ledger.frozen(report.parent);a=read(report)
            require(a['complete'] and a['density_audit']['checked_attempts']==record['samples'],'Incomplete independent reference audit')
            require(a['source_sha256'][str(output/'samples.jsonl')]==record['samples_sha256'],'Audit did not bind original fixture bytes')
            audits.append(dict(name=record['name'],analysis_sha256=sha(report),density_audit=a['density_audit']))
        ledger.recheck();write(root/'validation.json',dict(schema='full-vessel-integrated-bridge-validation-v1',complete=True,
            physical_system='synthetic spheres only',attempts=768,audits=audits,plan_sha256=sha(root/'plan.json'),
            scope='Integrated reconstruction reference, not protein convergence or assembly evidence.'))
        state.update(complete=True,phase='complete',validation_sha256=sha(root/'validation.json'),finished=time.time());snapshot()
    except BaseException as error:
        state.update(phase='failed',complete=False,error=repr(error),finished=time.time());snapshot();raise
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(run(args.root)['phase'],flush=True)
