#!/usr/bin/env python3
"""Proposal-only analytic R4 volume and independent density controls.

No proteins, hard predicates, Jacobian weights or depletants are evaluated.
The known integral is E_q[1_R4/(V_R4 q)] = 1, with every weight in [0,2].
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
import scipy
from analyze_latent_region import LatentImportanceGuide, shell_log_volume
from prepare_contact_bank_guides import log_proposal, local_source_closure

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'runs/smc-geometry-guide-preparation-20260923'
OLD=ROOT/'runs/refined-contact-bank-preparation-20260922'
N=16384
SEEDS=tuple(137001010+1009*i for i in range(8))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def require(value,message):
    if not value:raise ValueError(message)


def draw_proposal(rng,guide,n):
    """Untruncated mixture, independent of the Rust sampler implementation."""
    alpha=guide['defensive_uniform_shell_probability']
    branch=rng.random(n)>=alpha
    u=np.empty((n,6));component=np.full(n,-1,dtype=np.int16)
    m=int((~branch).sum())
    normals=rng.normal(size=(m,6));lengths=np.linalg.norm(normals,axis=1)
    require(np.all(lengths>0),'Zero Gaussian direction; do not silently drop a draw')
    u[~branch]=normals/lengths[:,None]*(4*rng.random(m)**(1/6))[:,None]
    cs=guide['gaussian_components'];weights=np.array([c['weight'] for c in cs]);weights/=weights.sum()
    component[branch]=rng.choice(len(cs),size=int(branch.sum()),p=weights)
    for i,c in enumerate(cs):
        selected=component==i;count=int(selected.sum())
        u[selected]=np.asarray(c['mean'])+rng.normal(size=(count,6))@np.linalg.cholesky(c['covariance']).T
    return u,branch,component


def run(out):
    require(not out.exists(),'Refuse to overwrite reference control')
    plan=read(PREP/'plan.json');frozen=read(PREP/'freeze.json')['files']
    for name,digest in frozen.items():require(sha(PREP/name)==digest,'Prepared guide changed')
    region=read(OLD/'region.json')
    require(sha(OLD/'region.json')==plan['input_sha256'][str(OLD/'region.json')], 'Region changed')
    paths={'bank':OLD/'guide-bank.json','smc':PREP/'guide-r5-cov1.json'}
    guides={k:read(p) for k,p in paths.items()}
    require(sha(paths['bank'])==plan['input_sha256'][str(paths['bank'])]
        and sha(paths['smc'])==plan['guides']['r5-cov1']['sha256'],'Guide identity changed')
    out.mkdir(parents=True)
    declaration=dict(schema='smc-guide-analytic-volume-reference-v1',samples_per_population=N,
        seeds=SEEDS,arms=list(guides),populations_per_arm=4,alpha=.5,
        expected_mean=1.,weight_bounds=[0.,2.],family_failure_probability=1e-6,
        source_sha256={str(p):sha(p) for p in [PREP/'plan.json',PREP/'freeze.json',OLD/'region.json',*paths.values()]},
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        scope='Independent proposal-only normalization control; not new physical data or Rust execution validation.')
    closure=local_source_closure([Path(__file__)])
    (out/'source').mkdir()
    for name,path in closure.items():shutil.copyfile(path,out/'source'/name)
    declaration['code_sha256']={str(p):sha(p) for p in closure.values()}
    write(out/'declaration.json',declaration)
    logvolume=float(shell_log_volume(region));arms={}
    # Hoeffding P(|mean-1|>=e)<=2 exp(-N_total*e²/2), union bound for two arms.
    epsilon=math.sqrt(2*math.log(4/declaration['family_failure_probability'])/(4*N))
    for arm_index,(arm,guide) in enumerate(guides.items()):
        require(guide['defensive_uniform_shell_probability']==.5,'Uniform floor changed')
        independent=LatentImportanceGuide(guide,sha(OLD/'region.json'));results=[]
        for p in range(4):
            seed=SEEDS[4*arm_index+p];u,branch,component=draw_proposal(np.random.default_rng(seed),guide,N)
            support=np.sum(u*u,axis=1)<=16
            q=independent.log_density(u,support,logvolume)
            error=float(np.max(np.abs(q-log_proposal(u,guide))))
            require(error<2e-10,'Independent proposal densities differ')
            weights=np.where(support,np.exp(-logvolume-q),0.)
            require(np.isfinite(weights).all() and weights.min()>=0 and weights.max()<=2+2e-12,
                'Analytic defensive weight bound failed')
            archive=out/f'{arm}-r{p:02d}.npz'
            np.savez_compressed(archive,u=u,branch=branch,component=component,log_q=q,
                support=support,volume_weight=weights,draw=np.arange(N))
            results.append(dict(id=f'r{p:02d}',seed=seed,samples=N,mean=float(weights.mean()),
                maximum_weight=float(weights.max()),exterior_zeros=int((~support).sum()),
                maximum_density_difference=error,archive_sha256=sha(archive)))
        mean=float(np.mean([r['mean'] for r in results]))
        # Deterministic density witnesses include every Gaussian center and points
        # immediately on either side of each coordinate-axis R4 boundary.
        witnesses=[c['mean'] for c in guide['gaussian_components']]
        for axis in range(6):
            for sign in (-1,1):
                for radius in (math.nextafter(4.,0.),4.,math.nextafter(4.,math.inf),8.):
                    v=np.zeros(6);v[axis]=sign*radius;witnesses.append(v.tolist())
        witnesses=np.array(witnesses);inside=np.sum(witnesses*witnesses,axis=1)<=16
        difference=float(np.max(abs(independent.log_density(witnesses,inside,logvolume)-log_proposal(witnesses,guide))))
        require(difference<2e-10,'Boundary density reconstruction failed')
        passed=abs(mean-1)<=epsilon
        arms[arm]=dict(populations=results,mean=mean,analytic_mean=1.,absolute_error=abs(mean-1),
            familywise_Hoeffding_halfwidth=epsilon,passed=passed,witness_count=len(witnesses),
            maximum_witness_density_difference=difference)
        print(f'{arm}: E[volume weight]={mean:.8f}, bound ±{epsilon:.8f}, pass={passed}',flush=True)
    require(all(a['passed'] for a in arms.values()),'Analytic normalization control failed')
    for path,digest in {**declaration['source_sha256'],**declaration['code_sha256']}.items():
        require(sha(path)==digest,'Reference source changed during execution')
    result=dict(schema=declaration['schema'],complete=True,arms=arms,total_proposal_only_draws=8*N,
        physical_draws=0,classifiers_rerun=0,audits_replayed=0,
        declaration_sha256=sha(out/'declaration.json'),scope=declaration['scope'])
    write(out/'validation.json',result)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();run(args.out.resolve())
