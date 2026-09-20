#!/usr/bin/env python3
"""Classify fresh full-support importance draws in unchanged frozen deep shells."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from prepare_smc_normalizer_atlas import Density,read,relative_poses,sha,write
from analyze_basin_normalizers import moments,paired_noise


def analyze(root,plan_path):
    manifest=read(root/"manifest.json");plan=read(plan_path)
    region=read(plan["regions"]["3"]["path"])
    assert sha(plan["regions"]["3"]["path"])==plan["regions"]["3"]["sha256"]
    cfg=read(root/"provenance/config.json")
    assert cfg["fixed_poses"]==[region["fixed_neighbor"]] and cfg["metadata"]==region["physical_metric"]
    assert cfg["capture_center"]==region["capture_center"] and cfg["capture_radius"]==region["capture_radius"]
    assert cfg["depletant_radius"]==region["depletant_radius"] and cfg["reservoir_density"]==region["activity"]
    assert sha(root/"provenance/shape.json")==region["shape_sha256"]
    assert plan["boundaries"]==[3.,5.,8.]
    chart=Density(region["gaussian_chart"])
    density=Density(read(root/"provenance/model.json"))
    names=["r_le_3","3_lt_r_le_5","5_lt_r_le_8","far_outside_r8","not_far","r_le_5","r_le_8","all_far","total"]
    partition=names[:5]
    all_logs={n:[] for n in names};all_hard={n:[] for n in names};all_pairs={n:[] for n in names}
    top={n:[] for n in partition};populations=[];density_error=0.
    for job in manifest["jobs"]:
        directory=Path(job["directory"]);summary=read(directory/"summary.json");runtime=read(directory/"manifest.json")
        assert summary["complete"] and summary["samples"]==job["samples"] and runtime["covariance_scale"]==1.
        rows=[json.loads(line) for line in (directory/"samples.jsonl").open()]
        assert len(rows)==job["samples"] and [r["draw"] for r in rows]==list(range(len(rows)))
        valid=np.array([r["hard_valid"] for r in rows]);indices=np.flatnonzero(valid)
        relative=relative_poses([rows[i]["pose"] for i in indices],region["fixed_neighbor"])
        distances=np.full(len(rows),np.inf)
        if len(indices):distances[indices]=chart.evaluate(relative)[1][:,0]
        far=np.array([r["q"] is not None and r["q"]>=region["minimum_original_q"] for r in rows])&valid
        masks={"r_le_3":far&(distances<=3),"3_lt_r_le_5":far&(distances>3)&(distances<=5),
            "5_lt_r_le_8":far&(distances>5)&(distances<=8),"far_outside_r8":far&(distances>8),
            "not_far":valid&~far,"r_le_5":far&(distances<=5),"r_le_8":far&(distances<=8),"all_far":far,"total":valid}
        assert np.array_equal(sum(masks[n].astype(int) for n in partition),valid.astype(int))
        stats={}
        for name,mask in masks.items():
            logs=np.array([r["log_importance_weight"] if hit else -np.inf for r,hit in zip(rows,mask)])
            hard=np.array([r["log_hard_weight"] if hit else -np.inf for r,hit in zip(rows,mask)])
            pairs=np.array([[c["log_weight"]-r["log_proposal_density"] for c in r["clouds"]] if hit else [-np.inf,-np.inf] for r,hit in zip(rows,mask)])
            if mask.any():assert np.max(np.abs(logsumexp(pairs[mask],axis=1)-np.log(2)-logs[mask]))<1e-10
            stats[name]=moments(logs);stats[name]["hard_region"]=moments(hard)
            all_logs[name].extend(logs);all_hard[name].extend(hard);all_pairs[name].extend(pairs)
            if name in top:
                chosen=sorted(np.flatnonzero(mask),key=lambda i:rows[i]["log_importance_weight"],reverse=True)[:8]
                for i in chosen:
                    row=rows[i]
                    top[name].append(dict(population=job["id"],draw=row["draw"],original_q=row["q"],
                        mahalanobis_radius=None if not np.isfinite(distances[i]) else float(distances[i]),
                        pose=row["pose"],log_proposal_density=row["log_proposal_density"],
                        log_importance_weight=row["log_importance_weight"],clouds=row["clouds"],proposal=row["proposal"]))
        values=[stats[n]["logQ"] for n in partition if stats[n]["logQ"] is not None]
        assert stats["total"]["logQ"] is None if not values else abs(logsumexp(values)-stats["total"]["logQ"])<1e-10
        populations.append(dict(id=job["id"],seed=job["seed"],estimates=stats))
        if len(indices):
            picks=np.linspace(0,len(indices)-1,min(32,len(indices)),dtype=int)
            sample=[relative[i] for i in picks];logg=density.evaluate(sample)[0]
            epsilon=runtime["uniform_probability"]
            expected=np.logaddexp(np.log(epsilon)-3*np.log(2*cfg["capture_radius"]),np.log1p(-epsilon)+logg)
            errors=np.abs(expected-np.array([rows[indices[i]]["log_proposal_density"] for i in picks]))
            density_error=max(density_error,float(errors.max()));assert errors.max()<2e-8
    estimates={}
    for name in names:
        e=moments(all_logs[name]);e["hard_region"]=moments(all_hard[name])
        e["paired_noise"]=paired_noise(np.array(all_logs[name]),np.array(all_pairs[name]))
        e["log_regional_depletion_enhancement"]=None if e["logQ"] is None or e["hard_region"]["logQ"] is None else e["logQ"]-e["hard_region"]["logQ"]
        poplogs=[p["estimates"][name]["logQ"] for p in populations]
        e["independent_populations"]=moments([-np.inf if x is None else x for x in poplogs]);e["independent_populations"]["logQ_values"]=poplogs
        estimates[name]=e
    for name in top:top[name]=sorted(top[name],key=lambda r:r["log_importance_weight"],reverse=True)[:8]
    output=dict(plan_sha256=sha(plan_path),chart_region_sha256=plan["regions"]["3"]["sha256"],
        shell_boundaries=plan["boundaries"],estimates=estimates,populations=populations,
        independent_full_proposal_log_density_max_error=density_error,
        scope="Fresh full-mixture importance weights divided by the complete known pose density; every unconditional draw retained. Deep shells require originalq>=5. No training draws reused; observed ESS does not certify coverage. Qz/Q0 are correlated point ratios.")
    out=root/"assessment";out.mkdir(exist_ok=True)
    write(out/"deep-shells.json",output);write(out/"top-deep-shell-contributors.json",top)
    print(json.dumps(dict(output=str(out/"deep-shells.json"),estimates={n:{k:estimates[n][k] for k in ["logQ","ess","nonzero","max_fraction"]} for n in names},density_error=density_error),indent=2))
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args();analyze(args.root.resolve(),args.plan.resolve())
