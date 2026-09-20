#!/usr/bin/env python3
"""Independent fixed-N aggregation of direct regional integration."""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import argparse
import math
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from prepare_smc_normalizer_atlas import Density,relative_poses,read,write,sha
from analyze_basin_normalizers import moments,paired_noise
from analyze_native_region_reference import native_q


def shell_log_volume(region):
    outer=region['mahalanobis_radius'];inner=region.get('minimum_mahalanobis_radius',0.)
    assert math.isfinite(outer) and math.isfinite(inner) and 0<=inner<outer
    result=3*np.log(np.pi)+6*np.log(outer)-np.log(6)
    if inner:
        result+=math.log(-math.expm1(6*math.log1p((inner-outer)/outer)))
    return result


def original_q_window(region):
    """Read original-metric boundaries; omitted flags retain the legacy law."""
    minimum=region['minimum_original_q']
    maximum=region.get('maximum_original_q',math.inf)
    lower=region.get('minimum_original_q_inclusive',True)
    upper=region.get('maximum_original_q_inclusive',True)
    assert type(lower) is bool and type(upper) is bool, 'q inclusion flags must be booleans'
    assert math.isfinite(minimum) and 0<=minimum<=maximum
    assert 'maximum_original_q' not in region or math.isfinite(maximum)
    return dict(minimum=minimum,maximum=maximum,lower_inclusive=lower,upper_inclusive=upper)


def original_q_contains(q,window):
    return ((q>=window['minimum'] if window['lower_inclusive'] else q>window['minimum']) and
        (q<=window['maximum'] if window['upper_inclusive'] else q<window['maximum']))


def validate_manifest_q_window(population_manifest,region,window):
    assert population_manifest['minimum_original_q']==window['minimum']
    assert population_manifest.get('maximum_original_q')==region.get('maximum_original_q')
    assert population_manifest.get('minimum_original_q_inclusive',True)==window['lower_inclusive']
    assert population_manifest.get('maximum_original_q_inclusive',True)==window['upper_inclusive']
    for field in ('minimum_original_q_inclusive','maximum_original_q_inclusive'):
        assert type(population_manifest.get(field,True)) is bool


def analyze(root):
    manifest=read(root/"manifest.json");region=read(root/"provenance/region.json")
    for name,digest in manifest['archive_sha256'].items():
        assert sha(root/'provenance'/name)==digest, f'Changed archived input: {name}'
    assert sha(root/'provenance/region.json')==manifest['region_sha256']
    all_logs=[];all_hard=[];all_pairs=[];populations=[];cpu=0.;backmap=0.;density_error=0.;audited_poses=0
    chart=Density(region["gaussian_chart"])
    log_volume=shell_log_volume(region)
    inner=region.get('minimum_mahalanobis_radius',0.)
    window=original_q_window(region)
    config=read(root/'provenance/config.json')
    fixed=region.get('physical_fixed_neighbors',[region['fixed_neighbor']])
    assert fixed==config['fixed_poses'] and region['fixed_neighbor'] in fixed
    assert region['physical_metric']==config['metadata']
    assert region['capture_center']==config['capture_center'] and region['capture_radius']==config['capture_radius']
    assert region['activity']==config['reservoir_density'] and region['depletant_radius']==config['depletant_radius']
    for job in manifest["jobs"]:
        directory=Path(job["directory"]);summary=read(directory/"summary.json")
        assert summary["complete"] and summary["samples"]==job["samples"]
        assert summary["manifest"]["region_sha256"]==manifest["region_sha256"]
        assert summary["manifest"]["executable_sha256"]==manifest["archive_sha256"]["latent-region-normalizer"]
        assert summary['manifest']['config_sha256']==sha(root/'provenance/config.json')
        assert summary['manifest']['shape_sha256']==region['shape_sha256']==sha(root/'provenance/shape.json')
        assert summary['manifest']['seed']==job['seed']
        assert summary['manifest']['samples']==job['samples']
        assert summary['manifest']['activity']==region['activity']
        assert summary['manifest']['lambda_ratio']==manifest['lambda_ratio']
        expected_lambda=region['activity']*manifest['lambda_ratio'] if region['activity']>0 else 1.
        assert summary['manifest']['lambda']==expected_lambda
        assert summary['manifest']['cloud_replicates']==manifest['cloud_replicates']==2
        for name,field in [('input-config.json','config_sha256'),('region.json','region_sha256'),
                           ('shape.json','shape_sha256'),('source-bundle.json','source_bundle_sha256')]:
            assert sha(directory/'provenance'/name)==summary['manifest'][field]
        extended=summary['manifest']['schema']=='uniform-latent-region-normalizer-v2'
        if not window['lower_inclusive'] or not window['upper_inclusive']:
            assert extended, 'Open q boundaries require the extended manifest'
        sample_hash=sha(directory/'samples.jsonl')
        if extended:
            assert read(directory/'manifest.json')==summary['manifest'], 'Summary and on-disk population manifests disagree'
            assert sample_hash==summary['samples_sha256']
            assert summary['manifest']['physical_fixed_neighbors']==fixed
            assert summary['manifest']['chart_anchor']==region['fixed_neighbor']
            assert summary['manifest']['minimum_latent_radius']==inner
            validate_manifest_q_window(summary['manifest'],region,window)
            assert abs(summary['manifest']['log_latent_shell_volume']-log_volume)<1e-10
        with (directory/'samples.jsonl').open() as sample_file:
            rows=[json.loads(line) for line in sample_file]
        assert len(rows)==job["samples"] and [r["draw"] for r in rows]==list(range(job["samples"]))
        logs=[];hard_logs=[];pairs=[]
        for r in rows:
            valid=r["hard_valid"] and r["region_valid"]
            assert inner*(1-1e-12)<=r["latent_radius"]<=region["mahalanobis_radius"]*(1+1e-12)
            assert r['region_valid']==original_q_contains(r['q'],window)
            if extended:
                assert abs(native_q(region['physical_metric'],r['pose'])-r['q'])<2e-8
                assert r['capture_valid']==(math.dist(r['pose']['position'],region['capture_center'])<=region['capture_radius'])
            if valid:
                assert r['capture_valid'] and original_q_contains(r['q'],window)
                assert len(r["clouds"])==2
                p=[log_volume+r["log_physical_jacobian"]+c["log_weight"] for c in r["clouds"]]
                assert abs(r['log_hard_weight']-log_volume-r['log_physical_jacobian'])<1e-10
                for cloud in r['clouds']:
                    expected=summary['manifest']['activity']*cloud['lower_volume']+cloud['overlap_points']*math.log1p(summary['manifest']['activity']/summary['manifest']['lambda'])
                    assert abs(cloud['log_weight']-expected)<1e-10
                assert abs(logsumexp(p)-np.log(2)-r["log_importance_weight"])<1e-10
                logs.append(r["log_importance_weight"]);hard_logs.append(r["log_hard_weight"]);pairs.append(p)
            else:
                assert r["log_importance_weight"] is None and not r["clouds"]
                logs.append(-np.inf);hard_logs.append(-np.inf);pairs.append([-np.inf,-np.inf])
        estimate=moments(logs)
        recorded=summary["estimates"]["region"]["logQ"]
        assert recorded is None if estimate["logQ"] is None else abs(estimate["logQ"]-recorded)<1e-10
        populations.append(dict(id=job["id"],seed=job["seed"],estimate=estimate,hard_region=moments(hard_logs),samples_sha256=sample_hash))
        all_logs.extend(logs);all_hard.extend(hard_logs);all_pairs.extend(pairs);cpu+=summary["sampler_cpu_seconds"]
        backmap=max(backmap,summary["maximum_backmap_error"])
        selected=rows if extended else [rows[i] for i in np.linspace(0,len(rows)-1,32,dtype=int)]
        audited_poses+=len(selected)
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
        independently_reconstructed_poses=audited_poses,log_latent_volume=log_volume,
        minimum_mahalanobis_radius=inner,maximum_original_q=region.get('maximum_original_q'),physical_fixed_neighbors=fixed,
        scope="ONLY the predeclared fixed ellipsoid or shell physical mass, not a global normalizer; no conditioning away invalid zeros")
    if any(field in region for field in ('minimum_original_q_inclusive','maximum_original_q_inclusive')):
        result['original_q_window']=dict(window,maximum=region.get('maximum_original_q'))
    out=root/"assessment";out.mkdir(exist_ok=True)
    write(out/"analysis.json",result)
    observed=(f"log Q = {estimate['logQ']:.6f}, ESS {estimate['ess']:.1f}, "
        f"largest contribution {estimate['max_fraction']:.2%}, observed relative SE {estimate['relative_se']:.2%}."
        if estimate["logQ"] is not None else "no nonzero observations: unresolved regional mass, not a physical zero or upper bound.")
    report=(f"# Direct integration of the frozen region\n\n"
        f"{len(all_logs):,} unconditional uniform six-dimensional ball/shell draws; {observed}\n\n"
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
