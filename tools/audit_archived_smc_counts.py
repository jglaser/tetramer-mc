#!/usr/bin/env python3
"""Run the actual archived SMC Poisson overlap-count law at three audited poses."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/home/xvg/protein-nucleation/results/coordination-test/narrow-water/high-activity/implementation/src')
def read(p): return json.loads(Path(p).read_text())
def save(p, x): Path(p).write_text(json.dumps(x, indent=2)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    archive=out/'provenance';archive.mkdir();crate=out/'probe';(crate/'src').mkdir(parents=True)
    ref=args.reference.resolve();reference=read(ref/'input.json');saved=read(ref/'results.json')
    wanted=['site0-m1-r1.5-z0.035-native-07-p263',
            'site0-m1-r1.5-z0.035-other_adsorbed-01-p463',
            'site1-m1-r1.5-z0.035-other_adsorbed-02-p352']
    cases=[next(c.copy() for c in reference['cases'] if c['id']==k) for k in wanted]
    for i,c in enumerate(cases):c['seed']=args.seed+10000000*i
    shutil.copy2(reference['shape'],archive/'shape.json')
    sources=[OLD/'geometry.rs',OLD/'single_body_depletion.rs',ROOT/'tools/archived_smc_count_probe.rs',Path(__file__)]
    for name in ['geometry.rs','single_body_depletion.rs']:shutil.copy2(OLD/name,crate/'src'/name)
    shutil.copy2(sources[2],crate/'src/main.rs');shutil.copy2(__file__,archive/Path(__file__).name)
    save(out/'input.json',dict(shape=str(archive/'shape.json'),rd=reference['rd'],activity=reference['activity'],
        clouds=args.clouds,intensity=args.intensity,cases=cases))
    save(out/'reference-results.json',[r for r in saved['results'] if r['id'] in wanted])
    cargo='[package]\nname="archived-smc-count-probe"\nversion="0.1.0"\nedition="2024"\n[dependencies]\n'
    cargo+='anyhow="1"\nserde={version="1",features=["derive"]}\nserde_json="1"\nrand="0.10"\n'
    vendor=ROOT/'vendor/hoomd-rs';old_vendor=Path('/home/xvg/protein-nucleation/vendor/hoomd-rs')
    for dep in ['hoomd-vector','hoomd-interaction','hoomd-microstate']:
        for p in (vendor/dep).rglob('*'):
            if p.is_file():assert sha(p)==sha(old_vendor/dep/p.relative_to(vendor/dep))
        cargo+=f'{dep}={{path="{vendor/dep}"}}\n'
    (crate/'Cargo.toml').write_text(cargo)
    save(out/'provenance.json',dict(source_sha256={str(p):sha(p) for p in sources},
        reference_sha256={str(ref/n):sha(ref/n) for n in ['input.json','results.json','provenance.json']},
        shape_sha256=sha(archive/'shape.json'),dependency_source_match=True,
        selection='Preselected three poses from previous geometry audit: largest-native-q site0, strongest old/current residual site0 other, site1 other. No selection using new counts.',
        algorithm='Archived geometry.rs and single_body_depletion.rs compiled unchanged; mirror module supplies only P,V aliases.',
        limit='This tests the archived incremental count law at fixed poses, not the complete SMC path, resampling or equilibrium coverage.'))
    env=os.environ.copy();env['CARGO_TARGET_DIR']=str(ROOT/'target/geometry-audit')
    with (out/'build.log').open('w') as log:
        subprocess.run(['/home/xvg/.cargo/bin/cargo','build','--offline','--release','--manifest-path',str(crate/'Cargo.toml')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    binary=Path(env['CARGO_TARGET_DIR'])/'release/archived-smc-count-probe';shutil.copy2(binary,archive/binary.name)
    def cap_cpu():resource.setrlimit(resource.RLIMIT_CPU,(90,90))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
    with (out/'probe.log').open('w') as log:
        subprocess.run([str(archive/binary.name),str(out/'input.json'),str(out/'results.json')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180,preexec_fn=cap_cpu)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    results=read(out/'results.json');references={r['id']:r for r in saved['results']}
    for r in results['results']:
        old=references[r['id']];df=r['clouds']-1;dispersion=df*r['variance_to_mean']
        r['poisson_dispersion_two_sided_p_approx']=min(1.,2*min(chi2.cdf(dispersion,df),chi2.sf(dispersion,df)))
        r['poisson_variance_mean_95pct_reference_interval_approx']=[chi2.ppf(.025,df)/df,chi2.ppf(.975,df)/df]
        for name,value,se in [('common',old['common_C_A3'],old['common_C_standard_error_A3']),
                              ('current',old['current_cloud_C_A3'],old['current_cloud_C_standard_error_A3'])]:
            delta=r['C_A3']-value;combined=(r['C_poisson_standard_error_A3']**2+se**2)**.5
            r[name+'_reference']=dict(C_A3=value,standard_error_A3=se,delta_A3=delta,combined_standard_error_A3=combined,
                delta_standard_errors=delta/combined,delta_kBT=reference['activity']*delta,combined_standard_error_kBT=reference['activity']*combined)
    results['elapsed_seconds']=time.monotonic()-start
    results['cpu_seconds']=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
    save(out/'assessment.json',results)
    save(out/'artifacts.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifacts.json'})
    print(json.dumps(results,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'runs/archived-smc-count-audit')
    p.add_argument('--reference',type=Path,default=ROOT/'runs/old-new-geometry-audit-v2')
    p.add_argument('--clouds',type=int,default=64);p.add_argument('--intensity',type=float,default=.125)
    p.add_argument('--seed',type=int,default=671203901)
    main(p.parse_args())
