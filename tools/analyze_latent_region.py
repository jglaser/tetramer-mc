#!/usr/bin/env python3
"""Independent fixed-N aggregation of direct regional integration."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from prepare_smc_normalizer_atlas import Density,relative_poses,read,write,sha
from analyze_basin_normalizers import moments,paired_noise

def analyze(root):
    manifest=read(root/"manifest.json");region=read(root/"provenance/region.json")
    all_logs=[];all_hard=[];all_pairs=[];populations=[];cpu=0.;backmap=0.;density_error=0.
    chart=Density(region["gaussian_chart"])
    log_volume=3*np.log(np.pi)+6*np.log(region["mahalanobis_radius"])-np.log(6)
    for job in manifest["jobs"]:
        directory=Path(job["directory"]);summary=read(directory/"summary.json")
        assert summary["complete"] and summary["samples"]==job["samples"]
        assert summary["manifest"]["region_sha256"]==manifest["region_sha256"]
        assert summary["manifest"]["executable_sha256"]==manifest["archive_sha256"]["latent-region-normalizer"]
        rows=[json.loads(line) for line in (directory/"samples.jsonl").open()]
        assert len(rows)==job["samples"] and [r["draw"] for r in rows]==list(range(job["samples"]))
        logs=[];hard_logs=[];pairs=[]
        for r in rows:
            valid=r["hard_valid"] and r["region_valid"]
            assert r["latent_radius"]<=region["mahalanobis_radius"]*(1+1e-12)
            if valid:
                assert r["capture_valid"] and r["q"]>=region["minimum_original_q"]
                assert len(r["clouds"])==2
                p=[log_volume+r["log_physical_jacobian"]+c["log_weight"] for c in r["clouds"]]
                assert abs(logsumexp(p)-np.log(2)-r["log_importance_weight"])<1e-10
                logs.append(r["log_importance_weight"]);hard_logs.append(r["log_hard_weight"]);pairs.append(p)
            else:
                assert r["log_importance_weight"] is None and not r["clouds"]
                logs.append(-np.inf);hard_logs.append(-np.inf);pairs.append([-np.inf,-np.inf])
        estimate=moments(logs)
        recorded=summary["estimates"]["region"]["logQ"]
        assert recorded is None if estimate["logQ"] is None else abs(estimate["logQ"]-recorded)<1e-10
        populations.append(dict(id=job["id"],seed=job["seed"],estimate=estimate,hard_region=moments(hard_logs)))
        all_logs.extend(logs);all_hard.extend(hard_logs);all_pairs.extend(pairs);cpu+=summary["sampler_cpu_seconds"]
        backmap=max(backmap,summary["maximum_backmap_error"])
        selected=[rows[i] for i in np.linspace(0,len(rows)-1,32,dtype=int)]
        poses=relative_poses([r["pose"] for r in selected],region["fixed_neighbor"])
        gaussian,norms,_=chart.evaluate(poses)
        for r,logg,md in zip(selected,gaussian,norms[:,0]):
            latent=np.array(r["latent"])
            assert abs(md-np.linalg.norm(latent))<2e-8
            independent_logj=-3*np.log(2*np.pi)-.5*np.sum(latent**2)-logg
            error=abs(independent_logj-r["log_physical_jacobian"])
            assert error<2e-8
            density_error=max(density_error,error)
    estimate=moments(all_logs)
    estimate["paired_noise"]=paired_noise(np.array(all_logs),np.array(all_pairs))
    estimate["independent_populations"]=moments([-np.inf if p["estimate"]["logQ"] is None else p["estimate"]["logQ"] for p in populations])
    estimate["independent_populations"]["logQ_values"]=[p["estimate"]["logQ"] for p in populations]
    hard=moments(all_hard)
    enhancement=None if estimate["logQ"] is None or hard["logQ"] is None else estimate["logQ"]-hard["logQ"]
    result=dict(region_sha256=manifest["region_sha256"],estimate=estimate,hard_region=hard,
        log_regional_depletion_enhancement=enhancement,
        enhancement_scope="Correlated ratio Qz/Q0 from the same poses, relative to uniform physical measure in this fixed valid region. Point value only; no independent-error assumption or substitution of mean overlap.",populations=populations,
        sampler_cpu_seconds=cpu,maximum_backmap_error=backmap,independent_jacobian_reconstruction_max_error=density_error,
        scope="ONLY the predeclared fixed ellipsoid physical mass, not a global normalizer; no conditioning away invalid zeros")
    out=root/"assessment";out.mkdir(exist_ok=True)
    write(out/"analysis.json",result)
    observed=(f"log Q = {estimate['logQ']:.6f}, ESS {estimate['ess']:.1f}, "
        f"largest contribution {estimate['max_fraction']:.2%}, observed relative SE {estimate['relative_se']:.2%}."
        if estimate["logQ"] is not None else "no nonzero observations: unresolved regional mass, not a physical zero or upper bound.")
    report=(f"# Direct integration of the frozen region\n\n"
        f"{len(all_logs):,} unconditional uniform six-ball draws; {observed}\n\n"
        f"Only the predeclared ellipsoid is integrated. This is not a global normalizer. "
        f"The region hash is `{manifest['region_sha256']}`. Invalid draws remain zeros.\n\n"
        f"Independent population log Q values: "+", ".join("unresolved" if p['estimate']['logQ'] is None else f"{p['estimate']['logQ']:.6f}" for p in populations)+".\n\n"
        f"CPU {cpu:.1f} s. Independent pose/Jacobian reconstruction maximum log error {density_error:.3g}.\n\n"
        + (f"Hard regional log Q0={hard['logQ']:.6f}; log(Qz/Q0)={enhancement:.6f}. This is regional depletion enhancement relative to uniform physical measure; the correlated ratio is reported without independent-error bars.\n" if enhancement is not None else "Hard-region mass or enhancement remains unresolved.\n"))
    (out/"report.md").write_text(report)
    print(report)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,required=True)
    analyze(parser.parse_args().root.resolve())
