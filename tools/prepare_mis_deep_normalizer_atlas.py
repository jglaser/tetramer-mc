#!/usr/bin/env python3
"""Freeze deep-contact Gaussian guides fitted with nested-domain MIS weights.

All input data are training data. Only subsequent independent production can
assess physical mass; mixture weights and tempered fit weights are proposals.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation
from prepare_smc_normalizer_atlas import (
    ROOT,Density,arrays,candidate_audit,read,relative_poses,sha,write,quantiles,
)


def log_mixture_density(radius,radii):
    radius=np.asarray(radius)
    logs=3*np.log(np.pi)+6*np.log(radii)-np.log(6.)
    return logsumexp(np.where(radius[...,None]<=radii,-logs,-np.inf),axis=-1)-np.log(len(radii))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,default=ROOT/"runs/smc-normalizer-deep-far/site0/models/std1.json")
    parser.add_argument("--out",type=Path,default=ROOT/"runs/smc-normalizer-mis-refined/site0")
    parser.add_argument("--candidate-probes",type=int,default=256)
    args=parser.parse_args()
    if args.candidate_probes<1:parser.error("Require positive probe count")
    roots=[ROOT/"runs/latent-region-uniform-ball-16384-l64",
           ROOT/"runs/latent-region-radius5-16384-l64",ROOT/"runs/latent-region-radius8-16384-l64"]
    radii=np.array([3.,5.,8.])
    volumes=np.pi**3*radii**6/6.
    parent=read(args.parent);component=len(parent["weights"])-1
    assert len(parent["weights"])==114 and parent["coordinate_convention"]=="anchor-body-relative"
    regions=[read(root/"provenance/region.json") for root in roots]
    assert [r["mahalanobis_radius"] for r in regions]==radii.tolist()
    keys=["gaussian_chart","fixed_neighbor","capture_center","capture_radius","shape_sha256","activity","depletant_radius","physical_metric","minimum_original_q"]
    signature={key:regions[0][key] for key in keys}
    assert all({key:r[key] for key in keys}==signature for r in regions)
    chart=signature["gaussian_chart"]
    for key in ["anchors","means","covariances"]:assert chart[key][0]==parent[key][component]
    assert chart["angular_length"]==parent["angular_length"] and parent["shape_sha256"]==signature["shape_sha256"]
    cfg=read(roots[0]/"provenance/config.json")
    assert cfg["metadata"]==signature["physical_metric"] and cfg["fixed_poses"]==[signature["fixed_neighbor"]]
    out=args.out.resolve()
    if out.exists() and any(out.iterdir()):parser.error("Use a fresh output directory")
    archive=out/"provenance";archive.mkdir(parents=True)
    fixed_sources={"parent-model.json":args.parent,"physical-config.json":roots[0]/"provenance/config.json",
        "shape.json":roots[0]/"provenance/shape.json","prepare_mis_deep_normalizer_atlas.py":Path(__file__),
        "prepare_smc_normalizer_atlas.py":Path(__file__).with_name("prepare_smc_normalizer_atlas.py"),
        "shell-plan.json":ROOT/"runs/latent-region-shell-plan-20260920/plan.json"}
    for name,path in fixed_sources.items():shutil.copy2(path,archive/name)
    sources=[];selected=[];budgets=[];all_radii=[];seeds=set()
    for source,(root,radius,volume) in enumerate(zip(roots,radii,volumes)):
        manifest=read(root/"manifest.json")
        assert sha(root/"provenance/shape.json")==signature["shape_sha256"]
        other=read(root/"provenance/config.json")
        for key in ["capture_center","capture_radius","fixed_poses","depletant_radius","reservoir_density","metadata"]:
            assert other[key]==cfg[key]
        shutil.copy2(root/"manifest.json",archive/f"campaign-{source}-manifest.json")
        shutil.copy2(root/"provenance/region.json",archive/f"campaign-{source}-region.json")
        source_radii=[];draws=0
        for job in manifest["jobs"]:
            assert job["seed"] not in seeds;seeds.add(job["seed"])
            directory=Path(job["directory"]);summary=read(directory/"summary.json")
            assert summary["complete"] and summary["samples"]==job["samples"]
            path=directory/"samples.jsonl";count=0
            for line in path.open():
                row=json.loads(line);assert row["draw"]==count;count+=1;draws+=1
                radial=float(np.linalg.norm(row["latent"]))
                assert radial<=radius*(1+1e-12)
                source_radii.append(radial)
                valid=row["hard_valid"] and row["capture_valid"] and row["region_valid"]
                if not valid:
                    assert row["log_importance_weight"] is None and not row["clouds"]
                    continue
                assert row["q"]>=signature["minimum_original_q"]
                logw=float(logsumexp([c["log_weight"] for c in row["clouds"]])-np.log(len(row["clouds"])))
                logg=float(log_mixture_density(radial,radii))
                log_mis=logw+row["log_physical_jacobian"]-logg
                assert abs(log_mis-(row["log_importance_weight"]-np.log(volume)-logg))<1e-10
                selected.append(dict(source=source,population=job["id"],seed=job["seed"],draw=row["draw"],
                    pose=row["pose"],latent=row["latent"],radius=radial,q=row["q"],
                    log_physical_jacobian=row["log_physical_jacobian"],cloud_log_weights=[c["log_weight"] for c in row["clouds"]],
                    log_cloud_average=logw,log_mixture_latent_density=logg,
                    log_mixture_physical_density=logg-row["log_physical_jacobian"],log_training_mis_weight=log_mis))
            assert count==job["samples"]
            sources.append(dict(campaign=source,population=job["id"],seed=job["seed"],samples=count,
                source_samples=str(path),samples_sha256=sha(path),summary_sha256=sha(directory/"summary.json")))
        budgets.append(draws);all_radii.append(np.asarray(source_radii))
    assert len(set(budgets))==1,"Equal-allocation MIS requires equal unconditional source budgets"
    # Independent measure control: known latent shell volumes, using every draw
    # and variance appropriate to fixed allocation among the three proposals.
    volume_checks=[]
    for lower,upper in [(0.,3.),(3.,5.),(5.,8.),(0.,8.)]:
        ys=[np.where((r<=upper)&((r>lower)|(lower==0.)),np.exp(-log_mixture_density(r,radii)),0.) for r in all_radii]
        mean=float(np.mean([y.mean() for y in ys]))
        se=float(np.sqrt(sum(np.var(y,ddof=1)/len(y) for y in ys))/3.)
        exact=float(np.pi**3*(upper**6-lower**6)/6.)
        assert abs(mean-exact)<6.5*se+1e-8*exact
        volume_checks.append(dict(lower=lower,upper=upper,estimate=mean,reference=exact,standard_error=se))
    boundaries=np.r_[0.,radii];piece_density=np.exp(log_mixture_density((boundaries[:-1]+boundaries[1:])/2,radii))
    density_integral=float(np.sum(piece_density*np.pi**3*np.diff(boundaries**6)/6.))
    assert abs(density_integral-1)<1e-13
    poses=relative_poses([r["pose"] for r in selected],signature["fixed_neighbor"])
    t,_,rot=arrays(poses);anchor=chart["anchors"][0];ell=chart["angular_length"]
    delta=Rotation.from_matrix(rot@np.asarray(anchor["rotation"]).T).as_quat()
    if np.any(delta[:,3]==0):raise ValueError("Nonfinite chart seam; no fit or censoring allowed")
    cayley=delta[:,:3]/delta[:,3,None]
    x=np.column_stack((t-anchor["position"],ell*cayley))
    sigma0=np.asarray(chart["covariances"][0]);chol0=np.linalg.cholesky(sigma0);mu0=np.asarray(chart["means"][0])
    stored_u=np.asarray([r["latent"] for r in selected]);from_latent=stored_u@chol0.T+mu0
    frame_error=float(np.max(np.abs(x-from_latent)));assert frame_error<2e-8 and np.isfinite(x).all()
    # Density/Jacobian identity checks all selected training points independently.
    logg=Density(chart).evaluate(poses)[0]
    independent_logj=-3*np.log(2*np.pi)-.5*np.sum(stored_u**2,axis=1)-logg
    jac_error=float(np.max(np.abs(independent_logj-np.asarray([r["log_physical_jacobian"] for r in selected]))))
    assert jac_error<2e-8
    model={key:copy.deepcopy(parent[key]) for key in ["schema","angular_length","coordinate_convention","shape_sha256","anchors","means","covariances","weights"]}
    model["weights"]=[.5*w for w in parent["weights"]]
    log_weights=np.asarray([r["log_training_mis_weight"] for r in selected])
    fits=[]
    for power in [1.,.5]:
        w=np.exp(power*log_weights-logsumexp(power*log_weights))
        mean=w@x;centered=x-mean;raw=centered.T@(w[:,None]*centered);raw=.5*(raw+raw.T)
        covariance=raw+.25*sigma0;covariance=.5*(covariance+covariance.T)
        np.linalg.cholesky(covariance);eig=np.linalg.eigvalsh(covariance)
        whitened=np.linalg.solve(chol0,np.linalg.solve(chol0,covariance).T).T
        relative_eig=np.linalg.eigvalsh(.5*(whitened+whitened.T))
        assert np.all(eig>0) and relative_eig[0]>=.25-1e-9
        model["anchors"].append(copy.deepcopy(anchor));model["means"].append(mean.tolist())
        model["covariances"].append(covariance.tolist());model["weights"].append(.25)
        fits.append(dict(weight_power=power,training_selected_count=len(w),training_unconditional_draws=sum(budgets),
            normalized_training_ess=float(1/np.sum(w*w)),largest_training_weight=float(w.max()),
            mean=mean.tolist(),mean_shift_in_original_latent_coordinates=np.linalg.solve(chol0,mean-mu0).tolist(),
            raw_weighted_covariance=raw.tolist(),additive_floor=(.25*sigma0).tolist(),final_covariance=covariance.tolist(),
            final_eigenvalues=eig.tolist(),condition_number=float(eig[-1]/eig[0]),
            eigenvalues_relative_to_original_covariance=relative_eig.tolist(),
            inference="Training fit only. Nonlinear tempering of noisy importance weights is a proposal choice, not an equilibrium density.",
            source_normalized_weight_fractions=[float(w[np.array([r['source']==s for r in selected])].sum()) for s in range(3)]))
    for key in ["anchors","means","covariances"]:assert model[key][:-2]==parent[key]
    assert abs(sum(model["weights"])-1)<1e-12
    provenance=dict(parent_model=str(args.parent.resolve()),parent_model_sha256=sha(args.parent),parent_components=114,
        parent_std_choice=1,retained_parent_weight=.5,new_weights=[.25,.25],chart_component=component,
        training_campaigns=[str(r) for r in roots],source_draw_counts=budgets,training_nonzero_count=len(selected),
        radii=radii.tolist(),latent_volumes=volumes.tolist(),allocation=[1/3]*3,
        fitting_denominator="g_mix(u)=sum_r (1/3) I(norm(u)<=r)/V6(r); physical density g_mix/J",
        training_weight="mean independent cloud W * J / g_mix; all invalid draws have zero weight",
        coordinate_check_max_error=frame_error,independent_jacobian_log_error=jac_error,
        maximum_chart_angle_deg=float(np.degrees(2*np.arccos(np.min(np.abs(delta[:,3]))))),
        raw_input_hashes=sources,scope="Frozen proposal-only MIS fit. All completed regional observations are training; only fresh global draws validate. Parent remains native-informed. No SMC normalizer or native pose enters new means.")
    model["proposal_provenance"]=provenance;write(out/"model.json",model)
    new={key:copy.deepcopy(model[key]) for key in ["schema","angular_length","coordinate_convention","shape_sha256"]}
    for key in ["anchors","means","covariances"]:new[key]=copy.deepcopy(model[key][-2:])
    new["weights"]=[.5,.5]
    geometry=candidate_audit(new,cfg,read(archive/"shape.json"),args.candidate_probes,84917210)
    # Proper body-to-lab frame conversion has determinant one, including a
    # correlated translation/rotation covariance. Audit both new guides.
    ft,_,fr=arrays([signature["fixed_neighbor"]]);block=np.zeros((6,6));block[:3,:3]=block[3:,3:]=fr[0]
    lab=copy.deepcopy(new);lab["coordinate_convention"]="laboratory"
    lab["anchors"]=[dict(position=(fr[0]@np.asarray(a["position"])+ft[0]).tolist(),rotation=(fr[0]@a["rotation"]).tolist()) for a in new["anchors"]]
    lab["means"]=(np.asarray(new["means"])@block.T).tolist();lab["covariances"]=(block@np.asarray(new["covariances"])@block.T).tolist()
    frame_density_error=float(np.max(np.abs(Density(lab).evaluate([r["pose"] for r in selected[:512]])[0]-Density(new).evaluate(poses[:512])[0])))
    assert frame_density_error<2e-8
    with (archive/"selected-training.jsonl").open("w") as f:
        for row in selected:f.write(json.dumps(row,allow_nan=False)+"\n")
    report=dict(provenance=provenance,fit_reports=fits,candidate_geometry=geometry,
        analytic_latent_mixture_density_integral=density_integral,independent_latent_volume_controls=volume_checks,
        body_lab_log_density_error=frame_density_error,model_components=len(model["weights"]),model_sha256=sha(out/"model.json"),
        archived_files={p.name:sha(p) for p in archive.iterdir()},generalization="There is no held-out physical estimate in this fit. Highly unequal weights and finite support R<=8 can miss important configurations; full support is retained by the parent plus uniform defense in future production.")
    write(out/"report.json",report)
    print(json.dumps(dict(output=str(out),model_sha256=report["model_sha256"],selected=len(selected),total_draws=sum(budgets),
        fits=[{k:r[k] for k in ["weight_power","normalized_training_ess","largest_training_weight","condition_number","eigenvalues_relative_to_original_covariance"]} for r in fits],candidate_geometry=geometry,volume_controls=volume_checks),indent=2))


if __name__=="__main__":main()
