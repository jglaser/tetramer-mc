#!/usr/bin/env python3
"""Analytic, no-refit moment-Gaussian control on the preserved pilot evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from fit_kernel_shear import KernelMixture
from kernel_shear_moments import moment_matched_mixture
from diagnose_fitted_kernel_shear import (bind, combine_group, evaluate_group, group_masks,
    load_rows, read, recheck, require, sha, status, write_new)
from prepare_shoulder_docking_benchmark import local_dependencies

PILOT_SHA='8ade00d34dffcfc5beee68d0fc1fdac774f478a8720d20e544cd26e89f900992'
MODEL_SHA='d780ad5eccb1c7078d43bf1367d80fb400fd7d9dcd038e84c05b1e5bc4689268'
SCOPE=('Analytic Gaussian first-two-moment control of a frozen nonlinear proposal. '
    'No data fitting, physical sampling, classifier call, candidate selection or gate promotion. '
    'Eight original pilot populations retained; heldout results are retrospective. '
    'The moment Gaussian optimizes KL for the fitted component, not for the physical target.')


def run(out, repository):
    out=Path(out).resolve(); repository=Path(repository).resolve()
    require(not out.exists(),'Fresh control output required')
    require(sys.flags.optimize==0,'Disable Python optimization')
    root=repository/'runs/fitted-kernel-shear-pilot-20260924';bindings={}
    pilot=read(bind(root/'analysis.json',bindings,PILOT_SHA))
    model_path=bind(root/'fitted-model.json',bindings,MODEL_SHA)
    require(pilot['complete'] is True and pilot['all_unconditional_draws_preserved']==524288,
        'Completed original pilot required')
    frozen=read(bind(root/'model-freeze.json',bindings))
    for name,digest in frozen['files'].items():bind(root/name,bindings,digest)
    records=[]
    for p in pilot['populations']:
        record={key:p[key] for key in ('arm','id','role','seed','samples')}
        path=bind(root/p['evaluated_rows'],bindings,p['evaluated_rows_sha256'])
        record.update(records=str(path),records_sha256=p['evaluated_rows_sha256'])
        records.append(record)
    require(len(records)==8 and len({r['seed'] for r in records})==8
        and sum(r['samples'] for r in records)==524288,'Original allocation changed')
    sources=local_dependencies([Path(__file__),Path(__file__).with_name('test_kernel_shear_moments.py')])
    for path in sources.values():bind(path,bindings)
    out.mkdir();(out/'source').mkdir()
    for name,path in sources.items():shutil.copy2(path,out/'source'/name)
    shutil.copy2(model_path,out/'frozen-shear-model.json')
    plan=dict(schema='kernel-moment-control-v1',scope=SCOPE,datasets=records,
        physical_draws=0,refit_calls=0,random_draws=0,model_sha256=MODEL_SHA,
        pilot_sha256=PILOT_SHA,input_and_source_sha256=bindings,
        construction='Exact Gaussian expectations of the fixed finite RBF displacement',
        comparisons=['original_affine_to_moment_gaussian','moment_gaussian_to_full_shear'],
        all_unconditional_attempts=524288,python=sys.executable,python_sha256=sha(sys.executable),python_version=sys.version)
    write_new(out/'plan.json',plan)
    write_new(out/'declaration-freeze.json',dict(files={str(p.relative_to(out)):sha(p)
        for p in out.rglob('*') if p.is_file()}))
    state=dict(complete=False,phase='analytic_construction');status(out/'status.json',state)
    started=time.monotonic()
    try:
        moment,diagnostics=moment_matched_mixture(KernelMixture.from_dict(read(model_path)))
        write_new(out/'moment-gaussian-model.json',moment.to_dict());write_new(out/'moments.json',diagnostics)
        write_new(out/'model-freeze.json',dict(files={name:sha(out/name) for name in
            ('plan.json','frozen-shear-model.json','moment-gaussian-model.json','moments.json')}))
        # Construction has used no evaluation rows. Retain all subsequent scores.
        state['phase']='evaluating';status(out/'status.json',state);populations=[];(out/'densities').mkdir()
        for record in records:
            arrays=load_rows(record); q0=arrays['log_baseline_density']; qs=arrays['log_warped_density']
            qm=moment.log_density(arrays['u'])
            require(qm.shape==q0.shape==qs.shape==(record['samples'],) and np.isfinite(qm).all(),
                'Incomplete/nonfinite moment-control densities')
            comparisons={name:{} for name in plan['comparisons']}
            for group,mask in group_masks(arrays):
                comparisons['original_affine_to_moment_gaussian'][group]=evaluate_group(arrays,q0,qm,mask)
                comparisons['moment_gaussian_to_full_shear'][group]=evaluate_group(arrays,qm,qs,mask)
            target=out/'densities'/f"{record['arm']}-{record['id']}.npz"
            np.savez_compressed(target,draw=arrays['draw'],log_moment_gaussian_density=qm)
            populations.append(dict(**record,comparisons=comparisons,attempts_preserved=len(qm),
                moment_density_file=str(target.relative_to(out)),moment_density_sha256=sha(target)))
        aggregates={}
        for role in ('training','heldout'):
            for arm in ('bank','smc'):
                group=[p for p in populations if p['role']==role and p['arm']==arm]
                require(len(group)==2,'Fixed source split changed')
                aggregates[role+'_'+arm]={comparison:{name:combine_group([p['comparisons'][comparison][name] for p in group])
                    for name in group[0]['comparisons'][comparison]} for comparison in plan['comparisons']}
        recheck(bindings)
        for name,digest in read(out/'model-freeze.json')['files'].items():require(sha(out/name)==digest,'Control model changed')
        result=dict(schema=plan['schema'],complete=True,scope=SCOPE,plan_sha256=sha(out/'plan.json'),
            populations=populations,aggregates=aggregates,physical_draws=0,refit_calls=0,random_draws=0,
            attempts_preserved=sum(p['attempts_preserved'] for p in populations),
            source_and_input_sha256=bindings,wall_seconds=time.monotonic()-started)
        write_new(out/'analysis.json',result)
        state.update(complete=True,phase='complete',analysis_sha256=sha(out/'analysis.json'));status(out/'status.json',state)
        write_new(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
        return result
    except BaseException as error:
        state.update(phase='failed',error=repr(error));status(out/'status.json',state);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--repository',type=Path,required=True)
    args=parser.parse_args();run(args.out,args.repository)
