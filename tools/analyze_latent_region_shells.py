#!/usr/bin/env python3
"""Poststratify uniform-ball draws into prespecified latent shells, keeping zeros."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from prepare_smc_normalizer_atlas import read,write,sha
from analyze_basin_normalizers import moments,paired_noise

def analyze(root,plan_path):
    manifest=read(root/"manifest.json");plan=read(plan_path);region=read(root/"provenance/region.json")
    radius=region["mahalanobis_radius"]
    assert radius in plan["boundaries"]
    assert manifest["region_sha256"]==plan["regions"][str(int(radius))]["sha256"]
    boundaries=[0.]+plan["boundaries"]
    names=[f"{lo:g}<r<={hi:g}" for lo,hi in zip(boundaries[:-1],boundaries[1:])]
    supported=[hi<=radius for hi in boundaries[1:]]
    weights={name:[] for name,ok in zip(names,supported) if ok};pairs={name:[] for name in weights}
    hard_weights={name:[] for name in weights};populations=[]
    for job in manifest["jobs"]:
        directory=Path(job["directory"]);summary=read(directory/"summary.json")
        assert summary["complete"] and summary["samples"]==job["samples"]
        assert summary["manifest"]["region_sha256"]==manifest["region_sha256"]
        rows=[json.loads(line) for line in (directory/"samples.jsonl").open()]
        assert len(rows)==job["samples"] and [r["draw"] for r in rows]==list(range(len(rows)))
        all_logs={name:[] for name in weights};all_pairs={name:[] for name in weights};all_hard={name:[] for name in weights}
        for row in rows:
            r=float(np.linalg.norm(row["latent"]))
            assert abs(r-row["backmapped_radius"])<2e-7*(1+radius)
            assert r<=radius*(1+1e-12)
            for name,lo,hi,ok in zip(names,boundaries[:-1],boundaries[1:],supported):
                if not ok:continue
                selected=row["log_importance_weight"] is not None and r<=hi and (r>lo or lo==0.)
                if selected:
                    log_base=row["log_hard_weight"]
                    p=[log_base+c["log_weight"] for c in row["clouds"]]
                    assert len(p)==2 and abs(logsumexp(p)-np.log(2)-row["log_importance_weight"])<1e-10
                    all_logs[name].append(row["log_importance_weight"]);all_hard[name].append(log_base);all_pairs[name].append(p)
                else:
                    all_logs[name].append(-np.inf);all_hard[name].append(-np.inf);all_pairs[name].append([-np.inf,-np.inf])
        per={}
        for name in weights:
            per[name]=moments(all_logs[name]);per[name]["hard_region"]=moments(all_hard[name])
            weights[name].extend(all_logs[name]);hard_weights[name].extend(all_hard[name]);pairs[name].extend(all_pairs[name])
        finite=[s["logQ"] for s in per.values() if s["logQ"] is not None]
        recorded=summary["estimates"]["region"]["logQ"]
        assert recorded is None if not finite else abs(logsumexp(finite)-recorded)<1e-10
        populations.append(dict(id=job["id"],seed=job["seed"],shells=per))
    estimates={}
    for name,ok in zip(names,supported):
        if not ok:
            estimates[name]=dict(logQ=None,ess=None,status="Outside this campaign's sampled domain; not estimated, not a zero mass")
            continue
        e=moments(weights[name]);e["paired_noise"]=paired_noise(np.array(weights[name]),np.array(pairs[name]))
        e["hard_region"]=moments(hard_weights[name])
        e["log_regional_depletion_enhancement"]=None if e["logQ"] is None or e["hard_region"]["logQ"] is None else e["logQ"]-e["hard_region"]["logQ"]
        logs=[p["shells"][name]["logQ"] for p in populations]
        e["independent_populations"]=moments([-np.inf if x is None else x for x in logs]);e["independent_populations"]["logQ_values"]=logs
        estimates[name]=e
    result=dict(plan_sha256=sha(plan_path),region_sha256=manifest["region_sha256"],campaign_radius=radius,
        boundaries=plan["boundaries"],estimates=estimates,populations=populations,
        scope="All supported shell masses use every unconditional draw in this campaign's denominator. No pooling across radii. Unsupported shells are not estimated; zero observed supported mass is unresolved. Depletion enhancement Qz/Q0 is a correlated same-pose ratio, point value only, not exp(z mean overlap).")
    out=root/"assessment";out.mkdir(exist_ok=True);write(out/"shells.json",result)
    print(json.dumps(dict(output=str(out/"shells.json"),estimates=estimates),indent=2))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,required=True);parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args();analyze(args.root.resolve(),args.plan.resolve())
