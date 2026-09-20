#!/usr/bin/env python3
"""Stream every native-reference row, retaining only moments and a small top-weight heap."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import time


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(min(a, b)-max(a, b)))


def fresh():
    return {"sum": -math.inf, "square": -math.inf, "max": -math.inf, "nonzero": 0}


def add(moment, weight):
    moment["sum"] = logadd(moment["sum"], weight)
    moment["square"] = logadd(moment["square"], 2*weight)
    moment["max"] = max(moment["max"], weight)
    moment["nonzero"] += 1


def present(moment, count):
    if moment["nonzero"] == 0:
        return {"samples": count, "nonzero": 0, "log_normalizer": None, "ess": 0,
                "relative_SE": None, "maximum_point_fraction": None}
    ess = math.exp(2*moment["sum"]-moment["square"])
    return {"samples": count, "nonzero": moment["nonzero"],
            "log_normalizer": moment["sum"]-math.log(count), "ess": ess,
            "relative_SE": math.sqrt(max(0.0, (count/ess-1)/(count-1))) if count > 1 else None,
            "maximum_point_fraction": math.exp(moment["max"]-moment["sum"])}


def theta_minus_sin(theta):
    # Independent Taylor evaluation avoids cancellation in the small-cap volume.
    if abs(theta) >= .05:
        return theta-math.sin(theta)
    term=theta**3/6
    total=term
    for k in range(2,9):
        term *= -theta*theta/((2*k)*(2*k+1))
        total += term
    return total


def normalized_quaternion(q):
    assert len(q)==4 and all(math.isfinite(x) for x in q)
    length=math.hypot(*q)
    assert length>0
    return [x/length for x in q]


def cross(a,b):
    return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]


def rotate(q,v):
    q=normalized_quaternion(q)
    twice=[2*x for x in cross(q[1:],v)]
    other=cross(q[1:],twice)
    return [v[i]+q[0]*twice[i]+other[i] for i in range(3)]


def cover_coordinates(cover,pose):
    p=normalized_quaternion(pose['orientation'])
    q=normalized_quaternion(cover['reference']['orientation'])
    pq=cross(q[1:],p[1:])
    vector=[q[0]*p[i+1]-p[0]*q[i+1]-pq[i] for i in range(3)]
    scalar=sum(x*y for x,y in zip(p,q))
    theta=2*math.atan2(math.hypot(*vector),abs(scalar))
    pc=rotate(p,cover['centroid']);qc=rotate(q,cover['centroid'])
    w=[pose['position'][i]-cover['reference']['position'][i]+pc[i]-qc[i] for i in range(3)]
    return math.hypot(*w),theta


def proposal_model(manifest,base):
    model=manifest.get('cover_mixture')
    if model is None:
        return None
    scales,weights,covers=(model[k] for k in ['scales','weights','covers'])
    assert len(scales)==len(weights)==len(covers)>0
    assert all(math.isfinite(s) and 0<s<=1 for s in scales)
    assert 1.0 in scales and all(math.isfinite(w) and w>0 for w in weights)
    assert abs(sum(weights)-1)<1e-12
    for scale,cover in zip(scales,covers):
        assert cover['reference']==base['reference'] and cover['centroid']==base['centroid']
        assert abs(cover['ball_radius']-scale*base['ball_radius'])<1e-13*base['ball_radius']
        assert abs(cover['angle_cap']-scale*base['angle_cap'])<1e-13*base['angle_cap']
        volume=4*cover['ball_radius']**3/3*theta_minus_sin(cover['angle_cap'])
        assert math.isfinite(volume) and volume>0
        assert abs(cover['volume']/volume-1)<2e-11
    return model


def proposal_log_density(model,pose):
    radius,angle=cover_coordinates(model['covers'][0],pose)
    log_density=-math.inf
    contains=[]
    for weight,cover in zip(model['weights'],model['covers']):
        inside=radius<=cover['ball_radius'] and angle<=cover['angle_cap']
        contains.append(inside)
        if inside:
            log_density=logadd(log_density,math.log(weight)-math.log(cover['volume']))
    return log_density,contains


def native_q(metric,pose):
    values=[]
    for reference in metric['native_poses']:
        errors=[]
        for member in metric['rigid_members']:
            x=rotate(pose['orientation'],member['position'])
            y=rotate(reference['orientation'],member['position'])
            errors.append(math.hypot(*[pose['position'][i]-reference['position'][i]+x[i]-y[i] for i in range(3)]))
        a=normalized_quaternion(pose['orientation']);b=normalized_quaternion(reference['orientation'])
        # Match the unchanged target's acos metric, independently of proposal support.
        angle=2*math.acos(min(1.,abs(sum(x*y for x,y in zip(a,b)))))
        values.append(max(max(errors)/metric['member_error_scale'],angle/math.radians(metric['angle_error_scale_deg'])))
    return min(values)


def paired_statistics(physical,hard,log_cross,n):
    if n<=1 or not physical['nonzero'] or not hard['nonzero']:
        return {'relative_covariance_of_means':None,'observed_SE_log_depletion_enhancement':None}
    covariance=math.expm1(math.log(n)+log_cross-physical['sum']-hard['sum'])/(n-1)
    vx=present(physical,n)['relative_SE']**2
    vy=present(hard,n)['relative_SE']**2
    return {'relative_covariance_of_means':covariance,
            'observed_SE_log_depletion_enhancement':math.sqrt(max(0.,vx+vy-2*covariance))}


def analyze(path_string):
    path = Path(path_string)
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    cover = json.loads((path / "cover.json").read_text())
    z, lam = manifest["activity"], manifest["lambda"]
    log_volume = math.log(cover["volume"])
    model=proposal_model(manifest,cover)
    mixture_rows=model is not None and len(model['scales'])>1
    config=json.loads((path/'provenance/config.json').read_text()) if mixture_rows else None
    counts = {"q": 0, "capture": 0, "hard": 0, "valid": 0}
    moments = {name: fresh() for name in ["native", "native_core", "native_shell"]}
    hard_moments={name:fresh() for name in moments}
    log_cross=-math.inf
    component_counts=[0]*len(model['scales']) if model else None
    component_valid=[0]*len(model['scales']) if model else None
    cloud_points=0
    top = []
    digest = hashlib.sha256()
    byte_count = 0
    n = 0
    with (path / "samples.jsonl").open("rb") as sample_file:
        for line in sample_file:
            digest.update(line)
            byte_count += len(line)
            row = json.loads(line)
            assert row["draw"] == n
            n += 1
            log_density=-log_volume
            if mixture_rows:
                assert math.isfinite(row['q'])
                assert abs(native_q(manifest['metric'],row['pose'])-row['q'])<2e-8
                log_density,membership=proposal_log_density(model,row['pose'])
                selected=row['proposal_component']
                assert isinstance(selected,int) and 0<=selected<len(membership) and membership[selected]
                assert math.isfinite(log_density) and abs(log_density-row['log_proposal_density'])<2e-10
                component_counts[selected]+=1
                distance=math.dist(row['pose']['position'],config['capture_center'])
                reason=row.get('zero')
                if row['q']<=1:
                    assert (distance>config['capture_radius'])==(reason=='capture')
            elif model:
                component_counts[0]+=1
            if "zero" in row:
                counts[row["zero"]] += 1
                assert (row["q"] > 1) == (row["zero"] == "q")
                continue
            assert row["q"] <= 1
            counts["valid"] += 1
            if model:
                component_valid[row['proposal_component'] if mixture_rows else 0]+=1
            logs = row["cloud_log_weights"]
            assert len(logs) == manifest["cloud_replicates"]
            assert len(row['cloud_overlap_counts'])==len(row['cloud_raw_points'])==len(logs)
            assert all(isinstance(k,int) and 0<=k<=raw for k,raw in zip(row['cloud_overlap_counts'],row['cloud_raw_points']))
            cloud_points+=sum(row['cloud_raw_points'])
            mean = -math.inf
            for k, w in zip(row["cloud_overlap_counts"], logs):
                expected = z*row["lower_volume"] + k*math.log1p(z/lam)
                assert abs(w-expected) < 1e-10
                mean = logadd(mean, w)
            mean -= math.log(len(logs))
            assert abs(mean-row["log_boltzmann_mean"]) < 1e-10
            hard_weight=-log_density
            weight = mean+hard_weight
            assert abs(weight-row["log_importance_weight"]) < 1e-10
            if mixture_rows:
                assert abs(hard_weight-row['log_hard_weight'])<2e-10
            add(moments["native"], weight)
            add(moments["native_core" if row["q"] <= .8 else "native_shell"], weight)
            add(hard_moments['native'],hard_weight)
            add(hard_moments['native_core' if row['q']<=.8 else 'native_shell'],hard_weight)
            log_cross=logadd(log_cross,weight+hard_weight)
            item = (weight, row["draw"], row)
            if len(top) < 12:
                heapq.heappush(top, item)
            elif item[:2] > top[0][:2]:
                heapq.heapreplace(top, item)
    assert n == summary["samples"] == manifest["samples"]
    assert digest.hexdigest() == summary["samples_sha256"]
    assert cloud_points==summary['raw_cloud_points']
    for key in ["q", "capture", "hard"]:
        assert counts[key] == summary[f"{key}_rejected"]
    for key, m in moments.items():
        assert m["nonzero"] == summary[key]["nonzero"]
        if m["nonzero"]:
            assert abs(m["sum"]-summary[key]["log_sum_weights"]) < 1e-8
            assert abs(m["square"]-summary[key]["log_sum_squared_weights"]) < 1e-8
    if 'hard_native' in summary:
        for key,m in hard_moments.items():
            assert m['nonzero']==summary['hard_'+key]['nonzero']
            if m['nonzero']:
                assert abs(m['sum']-summary['hard_'+key]['log_sum_weights'])<1e-8
                assert abs(m['square']-summary['hard_'+key]['log_sum_squared_weights'])<1e-8
        if counts['valid']:
            assert abs(log_cross-summary['physical_hard_log_cross_sum'])<1e-8
        else:
            assert summary['physical_hard_log_cross_sum'] is None
    return {"replicate": path.name, "samples": n, "counts": counts, "moments": moments,
            "regions": {key: present(m, n) for key, m in moments.items()},
            'hard_moments':hard_moments,'hard_regions':{key:present(m,n) for key,m in hard_moments.items()},
            'physical_hard_log_cross_sum':None if log_cross==-math.inf else log_cross,
            'paired_statistics':paired_statistics(moments['native'],hard_moments['native'],log_cross,n),
            'cover_mixture':model,'proposal_component_counts':component_counts,'proposal_component_valid':component_valid,
            "sample_bytes": byte_count, "sample_sha256": digest.hexdigest(),
            "summary_sha256": hashlib.sha256((path / "summary.json").read_bytes()).hexdigest(),
            "physical_signature": {key: manifest[key] for key in ["config_sha256", "shape_sha256",
                                       "activity", "depletant_radius", "metric"]},
            "cover": cover,
            "top_weights": [dict(item[2], replicate=path.name) for item in sorted(top, reverse=True)],
            "cover_volume": cover["volume"], "cpu_seconds": summary["cpu_seconds"],
            "wall_seconds": summary["wall_seconds"], "raw_cloud_points": summary["raw_cloud_points"]}


def require_matching_targets(results):
    assert results, "No populations to combine"
    reference = results[0]
    for result in results[1:]:
        assert result["physical_signature"] == reference["physical_signature"], "Different physical targets"
        assert result["cover"] == reference["cover"], "Different native covers"
        assert result["cover_volume"] == reference["cover_volume"], "Different cover volumes"
        assert result.get('cover_mixture')==reference.get('cover_mixture'), 'Different proposal mixtures; use a separately defined combination'


def replicate_relative_se(regional_results, combined_log_q):
    if combined_log_q is None or len(regional_results) <= 1:
        return None
    ratios = [math.exp(result["log_normalizer"]-combined_log_q)
              if result["log_normalizer"] is not None else 0.0 for result in regional_results]
    k = len(ratios)
    return math.sqrt(sum((x-1)**2 for x in ratios)/(k*(k-1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    manifest = json.loads((args.root / "manifest.json").read_text())
    assert all((Path(job["output"]) / "summary.json").exists() for job in manifest["jobs"])
    paths = [job["output"] for job in manifest["jobs"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(analyze, paths))
    require_matching_targets(results)
    n = sum(r["samples"] for r in results)
    assert n == manifest["total_unconditional_draws"]
    combined = {key: fresh() for key in ["native", "native_core", "native_shell"]}
    combined_hard={key:fresh() for key in combined}
    log_cross=-math.inf
    for result in results:
        for key, m in result["moments"].items():
            c = combined[key]
            c["sum"] = logadd(c["sum"], m["sum"])
            c["square"] = logadd(c["square"], m["square"])
            c["max"] = max(c["max"], m["max"])
            c["nonzero"] += m["nonzero"]
        for key,m in result['hard_moments'].items():
            c=combined_hard[key]
            c['sum']=logadd(c['sum'],m['sum']);c['square']=logadd(c['square'],m['square'])
            c['max']=max(c['max'],m['max']);c['nonzero']+=m['nonzero']
        if result['physical_hard_log_cross_sum'] is not None:
            log_cross=logadd(log_cross,result['physical_hard_log_cross_sum'])
    regions = {key: present(m, n) for key, m in combined.items()}
    hard_regions={key:present(m,n) for key,m in combined_hard.items()}
    # Independent equal-size population means provide a second observed error diagnostic.
    assert len({r["samples"] for r in results}) == 1
    qlog = regions["native"]["log_normalizer"]
    rep_se = replicate_relative_se([r["regions"]["native"] for r in results], qlog)
    nonzero = combined["native"]["nonzero"]
    cover_volume = results[0]["cover_volume"]
    is_mixture=results[0]['cover_mixture'] is not None and len(results[0]['cover_mixture']['scales'])>1
    hard_log=hard_regions['native']['log_normalizer']
    hard_volume=math.exp(hard_log) if hard_log is not None else 0.
    hard_se=hard_volume*hard_regions['native']['relative_SE'] if hard_log is not None and n>1 else None
    if not is_mixture:
        # Retain the legacy arithmetic as well as its sampling law.
        hard_volume=cover_volume*nonzero/n
        hard_se=cover_volume*math.sqrt((nonzero/n)*(1-nonzero/n)/n)
    result = {"complete": True, "all_rows_and_hashes_validated": True,
              "analyzed_utc": datetime.now(timezone.utc).isoformat(), "samples": n,
              "regions": regions, "independent_population_relative_SE": rep_se,
              'hard_regions':hard_regions,
              'hard_independent_population_relative_SE':replicate_relative_se([r['hard_regions']['native'] for r in results],hard_log),
              'paired_statistics':paired_statistics(combined['native'],combined_hard['native'],log_cross,n),
              'physical_hard_log_cross_sum':None if log_cross==-math.inf else log_cross,
              'cover_mixture':results[0]['cover_mixture'],
              "cpu_seconds": sum(r["cpu_seconds"] for r in results),
              "max_population_wall_seconds": max(r["wall_seconds"] for r in results),
              "raw_cloud_points": sum(r["raw_cloud_points"] for r in results),
              "sample_bytes": sum(r["sample_bytes"] for r in results),
              "zero_activity_volume_estimate_A3": hard_volume,
              "zero_activity_volume_SE_A3": hard_se,
              "populations": results,
              "top_weights": sorted([row for r in results for row in r["top_weights"]],
                                    key=lambda row: row["log_importance_weight"], reverse=True)[:20],
              "analysis_wall_seconds": time.monotonic()-started,
              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Complete geometric proposal support; fixed-budget independent estimate. "
                       "Observed errors and ESS do not certify unobserved weight concentration. "
                       "Previous campaign not pooled."}
    (args.root / "assessment-streaming.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ["populations", "top_weights"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
