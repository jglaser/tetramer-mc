#!/usr/bin/env python3
"""Stream every native-reference row, retaining only moments and a small top-weight heap."""

import argparse
import copy
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


def multiply_quaternions(a,b):
    v=cross(a[1:],b[1:])
    return [a[0]*b[0]-sum(x*y for x,y in zip(a[1:],b[1:]))]+[
        a[0]*b[i+1]+b[0]*a[i+1]+v[i] for i in range(3)]


def matrix_quaternion(matrix):
    """Independent stable proper-matrix to scalar-first quaternion conversion."""
    assert len(matrix)==3 and all(len(row)==3 for row in matrix)
    assert all(math.isfinite(x) for row in matrix for x in row)
    for i in range(3):
        for j in range(3):
            assert abs(sum(matrix[k][i]*matrix[k][j] for k in range(3))-(i==j))<1e-10
    determinant=sum(matrix[0][i]*cross(matrix[1],matrix[2])[i] for i in range(3))
    assert abs(determinant-1)<1e-10
    trace=sum(matrix[i][i] for i in range(3))
    if trace>0:
        s=2*math.sqrt(1+trace)
        q=[s/4,(matrix[2][1]-matrix[1][2])/s,(matrix[0][2]-matrix[2][0])/s,(matrix[1][0]-matrix[0][1])/s]
    else:
        i=max(range(3),key=lambda j:matrix[j][j]);j=(i+1)%3;k=(i+2)%3
        s=2*math.sqrt(1+matrix[i][i]-matrix[j][j]-matrix[k][k])
        q=[0.,0.,0.,0.];q[0]=(matrix[k][j]-matrix[j][k])/s
        q[i+1]=s/4;q[j+1]=(matrix[j][i]+matrix[i][j])/s;q[k+1]=(matrix[k][i]+matrix[i][k])/s
    return normalized_quaternion(q)


class GaussianGuide:
    """One fixed-anchor Gaussian atlas plus a laboratory-centered uniform cube.

    Pure-Python scalar reconstruction keeps the audit streaming and independent
    of Rust's matrix inverse and linear algebra. No density is conditioned on
    capture, hard validity, or which family actually generated the row.
    """
    def __init__(self,description,raw_model,config,shape_sha256):
        self.description=description
        self.weight=description['weight'];self.epsilon=description['uniform_probability']
        assert math.isfinite(self.weight) and 0<self.weight<1
        assert math.isfinite(self.epsilon) and 0<self.epsilon<=1
        index=description['anchor_index']
        assert type(index) is int and 0<=index<len(config['fixed_poses'])
        assert description['anchor_pose']==config['fixed_poses'][index]
        assert description['capture_center']==config['capture_center']
        self.center=description['capture_center'];self.lengths=description['cube_lengths']
        assert len(self.lengths)==3 and all(math.isfinite(x) and x>0 for x in self.lengths)
        assert self.lengths==[2*config['capture_radius']]*3
        self.anchor=description['anchor_pose']
        q=normalized_quaternion(self.anchor['orientation']);self.inverse_anchor=[q[0],-q[1],-q[2],-q[3]]
        assert raw_model['coordinate_convention']=='anchor-body-relative'
        assert raw_model['shape_sha256']==shape_sha256
        self.ell=raw_model['angular_length'];assert math.isfinite(self.ell) and self.ell>0
        count=len(raw_model['anchors']);assert count>0 and len(raw_model['means'])==count
        if raw_model.get('dfs') is not None:
            assert len(raw_model['dfs'])==count and all(x is None for x in raw_model['dfs'])
        assert (raw_model.get('covariances') is None)!=(raw_model.get('scales') is None)
        covariances=raw_model.get('covariances',raw_model.get('scales'))
        if covariances is None:covariances=raw_model['scales']
        weights=raw_model.get('weights')
        if weights is None:weights=[1/count]*count
        assert len(weights)==len(covariances)==count
        assert all(math.isfinite(w) and w>0 for w in weights) and abs(sum(weights)-1)<=2e-12
        total=sum(weights);self.components=[]
        for anchor,mean,covariance,weight in zip(raw_model['anchors'],raw_model['means'],covariances,weights):
            assert len(mean)==6 and all(math.isfinite(x) for x in mean)
            assert len(anchor['position'])==3 and all(math.isfinite(x) for x in anchor['position'])
            assert len(covariance)==6 and all(len(row)==6 for row in covariance)
            assert all(math.isfinite(x) for row in covariance for x in row)
            size=max(abs(x) for row in covariance for x in row);assert size>0
            for i in range(6):
                for j in range(6):assert abs(covariance[i][j]-covariance[j][i])<=1e-12*(size+abs(covariance[j][i]))
            lower=[[0.]*6 for _ in range(6)]
            for i in range(6):
                for j in range(i+1):
                    value=(covariance[i][j]+covariance[j][i])/2-sum(lower[i][k]*lower[j][k] for k in range(j))
                    if i==j:
                        assert value>0;lower[i][j]=math.sqrt(value)
                    else:lower[i][j]=value/lower[j][j]
            rotation=matrix_quaternion(anchor['rotation'])
            inverse_rotation=[rotation[0],-rotation[1],-rotation[2],-rotation[3]]
            normalizer=-3*math.log(2*math.pi)-sum(math.log(lower[i][i]) for i in range(6))
            self.components.append((anchor['position'],inverse_rotation,mean,lower,normalizer,math.log(weight/total)))

    def log_density(self,pose):
        displacement=[pose['position'][i]-self.anchor['position'][i] for i in range(3)]
        position=rotate(self.inverse_anchor,displacement)
        relative=normalized_quaternion(multiply_quaternions(self.inverse_anchor,normalized_quaternion(pose['orientation'])))
        component_logs=[]
        for anchor,inverse_rotation,mean,lower,normalizer,log_weight in self.components:
            delta=normalized_quaternion(multiply_quaternions(relative,inverse_rotation))
            if delta[0]==0:
                component_logs.append(-math.inf);continue
            cayley=[x/delta[0] for x in delta[1:]]
            x=[position[i]-anchor[i] for i in range(3)]+[self.ell*c for c in cayley]
            residual=[]
            for i in range(6):residual.append((x[i]-mean[i]-sum(lower[i][j]*residual[j] for j in range(i)))/lower[i][i])
            quadratic=sum(v*v for v in residual)
            if not math.isfinite(quadratic):component_logs.append(-math.inf);continue
            # Haar dc = dc/[pi^2 (1+|c|^2)^2]; x_rot=ell*c.
            haar=2*math.log(math.pi)+4*math.log(math.hypot(1.,*cayley))
            component_logs.append(log_weight+normalizer-.5*quadratic+3*math.log(self.ell)+haar)
        gaussian=-math.inf
        for value in component_logs:gaussian=logadd(gaussian,value)
        inside=all(-length/2<=pose['position'][i]-self.center[i]<length/2 for i,length in enumerate(self.lengths))
        uniform=math.log(self.epsilon)-sum(math.log(x) for x in self.lengths) if inside else -math.inf
        learned=math.log1p(-self.epsilon)+gaussian if self.epsilon<1 else -math.inf
        return logadd(uniform,learned),inside,component_logs


def hybrid_log_density(cover_model,guide,pose):
    cover_density,membership=proposal_log_density(cover_model,pose)
    guide_density,inside_cube,component_logs=guide.log_density(pose)
    density=logadd(math.log1p(-guide.weight)+cover_density,math.log(guide.weight)+guide_density)
    return density,membership,inside_cube,component_logs


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


DEFAULT_Q_WINDOW=dict(minimum=0.,maximum=1.,lower_inclusive=True,upper_inclusive=True)


def validate_q_window(window):
    assert set(window)==set(DEFAULT_Q_WINDOW), 'Unexpected q-window fields'
    lo,hi=window['minimum'],window['maximum']
    assert math.isfinite(lo) and math.isfinite(hi) and 0<=lo<hi
    assert type(window['lower_inclusive']) is bool and type(window['upper_inclusive']) is bool
    return window


def q_in_window(q,window):
    lo,hi=window['minimum'],window['maximum']
    return (q>=lo if window['lower_inclusive'] else q>lo) and (q<=hi if window['upper_inclusive'] else q<hi)


def window_cover_metric(metric,window):
    """Enclose the ORIGINAL q window; never change its membership metric."""
    validate_q_window(window)
    value=copy.deepcopy(metric)
    value['member_error_scale']*=window['maximum']
    value['angle_error_scale_deg']=min(180.,value['angle_error_scale_deg']*window['maximum'])
    assert math.isfinite(value['member_error_scale']) and value['member_error_scale']>0
    return value


def validate_window_cover(manifest,cover,config):
    """Independent covariance/norm calculation checks the complete outer cover.

    For member covariance M, mean squared registration error implies
    |w|² + 4 sin²(theta/2)(tr M - n^T M n) <= a².  The norm bound
    on lambda_max(M), with the recorded outward slack, gives a complete
    angular cap. Recomputing arcsin after scaling a matters; merely scaling
    the previously derived cap can truncate the requested q region.
    """
    window=validate_q_window(manifest['q_window'])
    metric=manifest['metric']
    assert all(config['metadata'][key]==value for key,value in metric.items()), 'Original physical metric changed'
    expected=window_cover_metric(metric,window)
    assert manifest['cover_metric']==expected, 'Cover metric must scale the original metric by qmax'
    members=expected['rigid_members'];assert members and len(expected['native_poses'])==1
    assert cover['reference']==expected['native_poses'][0]
    count=len(members)
    center=[sum(p['position'][i]/count for p in members) for i in range(3)]
    covariance=[[sum((p['position'][i]-center[i])*(p['position'][j]-center[j])/count
        for p in members) for j in range(3)] for i in range(3)]
    trace=sum(covariance[i][i] for i in range(3))
    frobenius=math.sqrt(sum(x*x for row in covariance for x in row))
    row_bound=max(sum(abs(x) for x in row) for row in covariance)
    magnitude=sum(sum(x*x for x in p['position'])/count for p in members)
    slack=4096.*math.ulp(1.)*(1.+magnitude)*count
    upper=min(frobenius,row_bound)+slack
    lower=max(0.,trace-slack-upper)
    radius=expected['member_error_scale'];nominal=math.radians(expected['angle_error_scale_deg'])
    cap=min(nominal,2*math.asin(min(1.,radius/(2*math.sqrt(lower)))) if lower>0 else math.pi)
    volume=4*radius**3/3*theta_minus_sin(cap)
    def close(a,b):
        assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-11*max(1.,abs(b))
    for a,b in zip(cover['centroid'],center):close(a,b)
    for ar,br in zip(cover['member_covariance'],covariance):
        for a,b in zip(ar,br):close(a,b)
    for field,value in [('moment_trace',trace),('lambda_max_upper',upper),('l_lower',lower),
        ('ball_radius',radius),('nominal_angle_cap',nominal),('angle_cap',cap)]:close(cover[field],value)
    assert math.isfinite(volume) and volume>0 and abs(cover['volume']/volume-1)<2e-10
    return window


def analyze(path_string):
    path = Path(path_string)
    summary = json.loads((path / "summary.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    cover = json.loads((path / "cover.json").read_text())
    z, lam = manifest["activity"], manifest["lambda"]
    log_volume = math.log(cover["volume"])
    model=proposal_model(manifest,cover)
    mixture_rows=model is not None and len(model['scales'])>1
    guide_description=manifest.get('guide')
    guided_rows=guide_description is not None
    windowed=manifest.get('schema')==4
    target='region' if windowed else 'native'
    window=DEFAULT_Q_WINDOW
    if manifest.get('schema')==3:assert guided_rows, 'Schema3 requires the hybrid guide description'
    detailed_rows=mixture_rows or guided_rows or windowed
    config=json.loads((path/'provenance/config.json').read_text()) if detailed_rows else None
    if windowed:
        assert model is not None, 'Schema4 requires the normalized complete cover mixture'
        assert hashlib.sha256((path/'provenance/config.json').read_bytes()).hexdigest()==manifest['config_sha256']
        assert hashlib.sha256((path/'provenance/shape.json').read_bytes()).hexdigest()==manifest['shape_sha256']
        assert config['depletant_radius']==manifest['depletant_radius']
        window=validate_window_cover(manifest,cover,config)
    guide=None
    if guided_rows:
        assert manifest['schema'] in (3,4) and model is not None
        model_bytes=(path/'provenance/guide-model.json').read_bytes()
        assert hashlib.sha256(model_bytes).hexdigest()==guide_description['model_sha256']
        guide=GaussianGuide(guide_description,json.loads(model_bytes),config,manifest['shape_sha256'])
    counts = {"q": 0, "capture": 0, "hard": 0, "valid": 0}
    moments = {name: fresh() for name in [target,target+'_core',target+'_shell']}
    hard_moments={name:fresh() for name in moments}
    log_cross=-math.inf
    component_counts=[0]*len(model['scales']) if model else None
    component_valid=[0]*len(model['scales']) if model else None
    family_counts={'cover':0,'guide':0} if guide else None
    family_valid={'cover':0,'guide':0} if guide else None
    guide_component_counts=[0]*len(guide.components) if guide else None
    guide_component_valid=[0]*len(guide.components) if guide else None
    guide_uniform_count=0
    guide_uniform_valid=0
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
            if detailed_rows:
                assert math.isfinite(row['q'])
                assert abs(native_q(manifest['metric'],row['pose'])-row['q'])<2e-8
                if guide:
                    log_density,membership,inside_cube,component_logs=hybrid_log_density(model,guide,row['pose'])
                    family=row['proposal_family'];assert family in family_counts
                    family_counts[family]+=1
                    if family=='cover':
                        assert row['guide_branch'] is None and row['guide_component'] is None
                        selected=row['proposal_component']
                        assert type(selected) is int and 0<=selected<len(membership) and membership[selected]
                        component_counts[selected]+=1
                    else:
                        assert row['proposal_component'] is None
                        branch=row['guide_branch'];assert branch in ('uniform','learned')
                        if branch=='uniform':
                            assert inside_cube and row['guide_component'] is None
                            guide_uniform_count+=1
                        else:
                            selected=row['guide_component']
                            assert guide.epsilon<1 and type(selected) is int and 0<=selected<len(component_logs)
                            assert math.isfinite(component_logs[selected])
                            guide_component_counts[selected]+=1
                else:
                    log_density,membership=proposal_log_density(model,row['pose'])
                    if windowed:
                        assert row['proposal_family']=='cover'
                        assert row.get('guide_branch') is None and row.get('guide_component') is None
                    selected=row['proposal_component']
                    assert type(selected) is int and 0<=selected<len(membership) and membership[selected]
                    component_counts[selected]+=1
                assert math.isfinite(log_density) and abs(log_density-row['log_proposal_density'])<2e-10
                distance=math.dist(row['pose']['position'],config['capture_center'])
                reason=row.get('zero')
                if q_in_window(row['q'],window):
                    assert (distance>config['capture_radius'])==(reason=='capture')
            elif model:
                component_counts[0]+=1
            if "zero" in row:
                counts[row["zero"]] += 1
                assert (not q_in_window(row['q'],window)) == (row["zero"] == "q")
                if guide or windowed:
                    assert row.get('log_importance_weight') is None and row.get('log_hard_weight') is None
                continue
            assert q_in_window(row['q'],window)
            counts["valid"] += 1
            if guide:
                family_valid[row['proposal_family']]+=1
                if row['proposal_family']=='cover':component_valid[row['proposal_component']]+=1
                elif row['guide_branch']=='uniform':guide_uniform_valid+=1
                else:guide_component_valid[row['guide_component']]+=1
            elif model:
                component_valid[row['proposal_component'] if detailed_rows else 0]+=1
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
            if detailed_rows:
                assert abs(hard_weight-row['log_hard_weight'])<2e-10
            add(moments[target], weight)
            add(moments[target+'_core' if row['q']<=.8 else target+'_shell'], weight)
            add(hard_moments[target],hard_weight)
            add(hard_moments[target+'_core' if row['q']<=.8 else target+'_shell'],hard_weight)
            log_cross=logadd(log_cross,weight+hard_weight)
            item = (weight, row["draw"], row)
            if len(top) < 12:
                heapq.heappush(top, item)
            elif item[:2] > top[0][:2]:
                heapq.heapreplace(top, item)
    assert n == summary["samples"] == manifest["samples"]
    assert digest.hexdigest() == summary["samples_sha256"]
    assert cloud_points==summary['raw_cloud_points']
    if 'component_draws' in summary:assert component_counts==summary['component_draws']
    if guide:
        assert family_counts==summary['family_draws']
        assert guide_component_counts==summary['guide_component_draws']
        assert guide_uniform_count==summary['guide_uniform_draws']
        assert sum(component_counts)==family_counts['cover']
        assert sum(guide_component_counts)+guide_uniform_count==family_counts['guide']
    for key in ["q", "capture", "hard"]:
        assert counts[key] == summary[f"{key}_rejected"]
    for key, m in moments.items():
        assert m["nonzero"] == summary[key]["nonzero"]
        if m["nonzero"]:
            assert abs(m["sum"]-summary[key]["log_sum_weights"]) < 1e-8
            assert abs(m["square"]-summary[key]["log_sum_squared_weights"]) < 1e-8
    if 'hard_'+target in summary:
        for key,m in hard_moments.items():
            assert m['nonzero']==summary['hard_'+key]['nonzero']
            if m['nonzero']:
                assert abs(m['sum']-summary['hard_'+key]['log_sum_weights'])<1e-8
                assert abs(m['square']-summary['hard_'+key]['log_sum_squared_weights'])<1e-8
        if counts['valid']:
            assert abs(log_cross-summary['physical_hard_log_cross_sum'])<1e-8
        else:
            assert summary['physical_hard_log_cross_sum'] is None
    result = {"replicate": path.name, "samples": n, "counts": counts, "moments": moments,
            "regions": {key: present(m, n) for key, m in moments.items()},
            'hard_moments':hard_moments,'hard_regions':{key:present(m,n) for key,m in hard_moments.items()},
            'physical_hard_log_cross_sum':None if log_cross==-math.inf else log_cross,
            'paired_statistics':paired_statistics(moments[target],hard_moments[target],log_cross,n),
            'cover_mixture':model,'proposal_component_counts':component_counts,'proposal_component_valid':component_valid,
            'guide':guide_description,'proposal_family_counts':family_counts,'proposal_family_valid':family_valid,
            'guide_component_counts':guide_component_counts,'guide_component_valid':guide_component_valid,
            'guide_uniform_count':guide_uniform_count if guide else None,'guide_uniform_valid':guide_uniform_valid if guide else None,
            "sample_bytes": byte_count, "sample_sha256": digest.hexdigest(),
            "summary_sha256": hashlib.sha256((path / "summary.json").read_bytes()).hexdigest(),
            "physical_signature": {key: manifest[key] for key in ["config_sha256", "shape_sha256",
                                       "activity", "depletant_radius", "metric"]},
            "cover": cover,
            "top_weights": [dict(item[2], replicate=path.name) for item in sorted(top, reverse=True)],
            "cover_volume": cover["volume"], "cpu_seconds": summary["cpu_seconds"],
            "wall_seconds": summary["wall_seconds"], "raw_cloud_points": summary["raw_cloud_points"]}
    if windowed:
        result.update(q_window=window,cover_metric=manifest['cover_metric'],target_key=target,
            independent_q_and_density_rows=n,independent_complete_cover_validated=True)
        result['physical_signature']['q_window']=window
        result['physical_signature']['physical_fixed_neighbors']=config['fixed_poses']
        result['physical_signature']['capture_center']=config['capture_center']
        result['physical_signature']['capture_radius']=config['capture_radius']
    return result


def require_matching_targets(results):
    assert results, "No populations to combine"
    reference = results[0]
    for result in results[1:]:
        assert result["physical_signature"] == reference["physical_signature"], "Different physical targets"
        assert result["cover"] == reference["cover"], "Different native covers"
        assert result["cover_volume"] == reference["cover_volume"], "Different cover volumes"
        assert result.get('cover_mixture')==reference.get('cover_mixture'), 'Different proposal mixtures; use a separately defined combination'
        assert result.get('guide')==reference.get('guide'), 'Different hybrid guides; use a separately defined combination'


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
    target=results[0].get('target_key','native')
    if target=='region':
        assert manifest['q_window']==results[0]['q_window'], 'Campaign window differs from population window'
        assert manifest['config_sha256']==results[0]['physical_signature']['config_sha256']
        assert all(hashlib.sha256((args.root/'provenance'/name).read_bytes()).hexdigest()==digest
            for name,digest in manifest['archive_sha256'].items()), 'Frozen campaign archive changed'
        source_config=json.loads((args.root/'provenance/input-config.json').read_text())
        effective_config=json.loads((args.root/'provenance/config.json').read_text())
        assert {k:v for k,v in source_config.items() if k!='shape'}=={k:v for k,v in effective_config.items() if k!='shape'}
        assert hashlib.sha256((args.root/'provenance/input-config.json').read_bytes()).hexdigest()==manifest['source_config_sha256']
        assert hashlib.sha256((args.root/'provenance/shape.json').read_bytes()).hexdigest()==manifest['shape_sha256']
        assert len({job['seed'] for job in manifest['jobs']})==len(results), 'Independent populations require distinct seeds'
        for job in manifest['jobs']:
            population=json.loads((Path(job['output'])/'manifest.json').read_text())
            assert population['seed']==job['seed'], 'Population seed differs from frozen campaign'
    combined = {key: fresh() for key in [target,target+'_core',target+'_shell']}
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
    qlog = regions[target]["log_normalizer"]
    rep_se = replicate_relative_se([r["regions"][target] for r in results], qlog)
    nonzero = combined[target]["nonzero"]
    cover_volume = results[0]["cover_volume"]
    is_mixture=results[0].get('guide') is not None or (results[0]['cover_mixture'] is not None and len(results[0]['cover_mixture']['scales'])>1)
    hard_log=hard_regions[target]['log_normalizer']
    hard_volume=math.exp(hard_log) if hard_log is not None else 0.
    hard_se=hard_volume*hard_regions[target]['relative_SE'] if hard_log is not None and n>1 else None
    if not is_mixture:
        # Retain the legacy arithmetic as well as its sampling law.
        hard_volume=cover_volume*nonzero/n
        hard_se=cover_volume*math.sqrt((nonzero/n)*(1-nonzero/n)/n)
    result = {"complete": True, "all_rows_and_hashes_validated": True,
              "analyzed_utc": datetime.now(timezone.utc).isoformat(), "samples": n,
              "regions": regions, "independent_population_relative_SE": rep_se,
              'hard_regions':hard_regions,
              'hard_independent_population_relative_SE':replicate_relative_se([r['hard_regions'][target] for r in results],hard_log),
              'paired_statistics':paired_statistics(combined[target],combined_hard[target],log_cross,n),
              'physical_hard_log_cross_sum':None if log_cross==-math.inf else log_cross,
              'cover_mixture':results[0]['cover_mixture'],
              'guide':results[0].get('guide'),
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
    if target=='region':
        result.update(target_key=target,q_window=results[0]['q_window'],cover_metric=results[0]['cover_metric'],
            independent_complete_cover_validated=True,
            independent_q_and_density_rows=sum(r['independent_q_and_density_rows'] for r in results),
            subregion_rule='region_core uses original q <= 0.8; region_shell is the remaining part of the declared window')
    (args.root / "assessment-streaming.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ["populations", "top_weights"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
