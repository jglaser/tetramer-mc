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
from scipy.linalg import solve_triangular
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
from prepare_smc_normalizer_atlas import Density,arrays,relative_poses,read,write,sha
from analyze_basin_normalizers import moments,paired_noise
from analyze_native_region_reference import native_q
from entry_shell_proposal import EntryShellGuide
from conditional_ray_proposal import ConditionalRayGuide


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


IMPORTANCE_SCHEMA='importance-latent-region-normalizer-v1'
ENTRY_IMPORTANCE_SCHEMA='importance-latent-region-normalizer-v2'
RAY_IMPORTANCE_SCHEMA='importance-latent-region-normalizer-v3'
GUIDE_FILE='importance-guide.json'
DENSITY_MEASURE='Lebesgue measure in the original six-dimensional whitened region chart'


def latent_shell_contains(radius,region):
    """The target shell is open at a positive inner radius; the ball includes 0."""
    inner=region.get('minimum_mahalanobis_radius',0.)
    return (math.isfinite(radius) and (radius>inner if inner else radius>=0.)
        and radius<=region['mahalanobis_radius'])


class LatentImportanceGuide:
    """Independent full proposal density in the original whitened d^6u measure.

    Gaussian components are never truncated or renormalized on the target shell.
    The only shell indicator multiplies the defensive uniform contribution.
    """
    def __init__(self,guide,region_sha256):
        assert set(guide)=={'schema','region_sha256','defensive_uniform_shell_probability','gaussian_components'}
        assert guide['schema']=='defensive-latent-shell-guide-v1'
        assert guide['region_sha256']==region_sha256, 'Guide targets another frozen region'
        assert isinstance(region_sha256,str) and len(region_sha256)==64 and all(c in '0123456789abcdefABCDEF' for c in region_sha256)
        self.alpha=guide['defensive_uniform_shell_probability']
        assert type(self.alpha) in (float,int) and math.isfinite(self.alpha) and 0<self.alpha<=1
        components=guide['gaussian_components']
        assert isinstance(components,list) and (components or self.alpha==1.)
        self.count=len(components)
        weights=[];means=[];lowers=[];normalizers=[]
        for component in components:
            assert set(component)=={'weight','mean','covariance'}
            weight=component['weight']
            assert type(weight) in (int,float) and math.isfinite(weight) and weight>0
            assert isinstance(component['mean'],list) and all(type(x) in (int,float) for x in component['mean'])
            assert isinstance(component['covariance'],list) and all(isinstance(row,list) and all(type(x) in (int,float) for x in row) for row in component['covariance'])
            mean=np.asarray(component['mean'],dtype=float)
            covariance=np.asarray(component['covariance'],dtype=float)
            assert mean.shape==(6,) and covariance.shape==(6,6)
            assert np.isfinite(mean).all() and np.isfinite(covariance).all()
            assert np.all(np.abs(covariance-covariance.T)<=1e-12*(1+np.maximum(np.abs(covariance),np.abs(covariance.T))))
            lower=np.linalg.cholesky(.5*(covariance+covariance.T))
            normalizer=-3*np.log(2*np.pi)-np.log(np.diag(lower)).sum()
            assert np.isfinite(lower).all() and math.isfinite(normalizer)
            weights.append(weight);means.append(mean);lowers.append(lower);normalizers.append(normalizer)
        total=sum(weights)
        assert not weights or (math.isfinite(total) and total>0)
        self.weights=np.asarray(weights)/total if weights else np.empty(0)
        assert np.isfinite(self.weights).all() and np.all(self.weights>0)
        self.means,self.lowers,self.normalizers=means,lowers,normalizers

    def log_density(self,latents,shell_valid,log_volume):
        latents=np.asarray(latents,dtype=float);shell_valid=np.asarray(shell_valid,dtype=bool)
        assert latents.ndim==2 and latents.shape[1]==6 and shell_valid.shape==(len(latents),)
        assert np.isfinite(latents).all() and math.isfinite(log_volume)
        total=np.where(shell_valid,math.log(self.alpha)-log_volume,-np.inf)
        if self.alpha<1.:
            gaussian=[]
            for weight,mean,lower,normalizer in zip(self.weights,self.means,self.lowers,self.normalizers):
                residual=solve_triangular(lower,(latents-mean).T,lower=True).T
                gaussian.append(math.log(weight)+normalizer-.5*np.einsum('ij,ij->i',residual,residual))
            total=np.logaddexp(total,math.log1p(-self.alpha)+logsumexp(gaussian,axis=0))
        return total


def importance_guide_binding(root,manifest,region):
    digest=manifest['archive_sha256'].get(GUIDE_FILE)
    if digest is None:
        assert 'importance_guide_sha256' not in manifest, 'Unarchived importance guide'
        assert manifest.get('schema')not in ('importance-latent-region-campaign-v1','importance-latent-region-campaign-v2','importance-latent-region-campaign-v3'), 'Importance campaign has no archived guide'
        return None,None
    assert manifest['importance_guide_sha256']==digest
    path=root/'provenance'/GUIDE_FILE
    assert sha(path)==digest
    if read(path).get('schema')=='defensive-conditional-ray-guide-v1':
        assert manifest['schema']=='importance-latent-region-campaign-v3', 'Conditional ray requires explicit v3 campaign'
        return ConditionalRayGuide(read(path),region,manifest['region_sha256']),digest
    if read(path).get('schema')=='defensive-entry-shell-guide-v1':
        assert manifest['schema']=='importance-latent-region-campaign-v2', 'Entry shell requires an explicit v2 campaign'
        return EntryShellGuide(read(path),region,manifest['region_sha256']),digest
    assert manifest['schema']=='importance-latent-region-campaign-v1', 'Gaussian guide requires its explicit importance campaign'
    return LatentImportanceGuide(read(path),manifest['region_sha256']),digest


def validate_population_guide(directory,population_manifest,guide,digest):
    importance=population_manifest['schema'] in (IMPORTANCE_SCHEMA,ENTRY_IMPORTANCE_SCHEMA,RAY_IMPORTANCE_SCHEMA)
    assert importance==(guide is not None), 'Campaign and population sampling laws differ'
    if not importance:
        assert 'importance_guide_sha256' not in population_manifest
        return False
    assert sha(directory/'provenance'/GUIDE_FILE)==population_manifest['importance_guide_sha256']==digest
    assert population_manifest['importance_uniform_probability']==guide.alpha
    assert type(population_manifest['importance_component_count']) is int and population_manifest['importance_component_count']==guide.count
    assert population_manifest['proposal_density_measure']==DENSITY_MEASURE
    if isinstance(guide,(EntryShellGuide,ConditionalRayGuide)):
        assert population_manifest['schema']==guide.population_schema
        assert population_manifest['guide_schema']==guide.schema and population_manifest['proposal_kind']==guide.proposal_kind
    else:
        assert population_manifest['schema']==IMPORTANCE_SCHEMA
    return True


def audit_importance_rows(rows,region,guide,chart):
    """Reconstruct every actual latent point, shell flag, Jacobian and full q."""
    assert len(chart.weights)==1 and not chart.inverted.any(), 'Region chart must be a single unwrapped Gaussian'
    for row in rows:
        assert all(type(row[key]) is bool for key in ('shell_valid','capture_valid','hard_valid','region_valid'))
        assert abs(np.linalg.norm(row['pose']['orientation'])-1.)<1e-8
    poses=relative_poses([row['pose'] for row in rows],region['fixed_neighbor'])
    translation,_,rotation=arrays(poses)
    anchor=chart.anchors[0]
    quaternion=Rotation.from_matrix(rotation@np.asarray(anchor['rotation']).T).as_quat()
    assert np.all(quaternion[:,3]!=0.), 'A chart inverse failure is an error, not a rejected draw'
    cayley=quaternion[:,:3]/quaternion[:,3,None]
    coordinates=np.column_stack((translation-anchor['position'],chart.ell*cayley))
    reconstructed=solve_triangular(chart.lower[0],(coordinates-chart.mean[0]).T,lower=True).T
    latent=np.asarray([row['latent'] for row in rows],dtype=float)
    assert latent.shape==(len(rows),6) and np.isfinite(latent).all() and np.isfinite(reconstructed).all()
    latent_error=float(np.max(np.abs(latent-reconstructed)))
    assert latent_error<2e-8, 'Saved latent vector does not reconstruct the saved physical pose'
    radii=np.linalg.norm(latent,axis=1)
    # Radius is logged by the draw, independently of the chart backmap. Allow
    # roundoff between Euclidean norm implementations, never a target-size cap.
    for row,radius,actual in zip(rows,radii,reconstructed):
        assert math.isfinite(row['latent_radius']) and abs(row['latent_radius']-radius)<2e-8
        backmapped=np.asarray(row['backmapped_latent'])
        assert backmapped.shape==(6,) and np.max(np.abs(backmapped-actual))<2e-8
        assert abs(row['backmapped_radius']-np.linalg.norm(actual))<2e-8
    shell=np.asarray([latent_shell_contains(row['latent_radius'],region) for row in rows])
    assert all(type(row['shell_valid']) is bool and row['shell_valid']==inside for row,inside in zip(rows,shell)), 'Wrong saved shell indicator'
    independent_logj=chart.logdet[0]-3*np.log(chart.ell)-2*np.log(np.pi)-2*np.log1p(np.einsum('ij,ij->i',cayley,cayley))
    jacobian_error=float(np.max(np.abs(independent_logj-[row['log_physical_jacobian'] for row in rows])))
    assert math.isfinite(jacobian_error) and jacobian_error<2e-8
    expected=guide.log_density(latent,shell,shell_log_volume(region))
    recorded=np.asarray([row['log_proposal_density'] for row in rows],dtype=float)
    errors=np.abs(expected-recorded)
    assert np.isfinite(recorded).all() and np.isfinite(errors).all() and np.max(errors)<2e-8, 'Full defensive mixture density differs'
    guided_branch=getattr(guide,'branch','gaussian')
    counts={'uniform-shell':0,guided_branch:0};component_counts=[0]*guide.count
    entry_support=guide.entry_support(latent) if isinstance(guide,EntryShellGuide) else None
    ray=guide.diagnostics(latent) if isinstance(guide,ConditionalRayGuide) else None
    ray_fallbacks=[0]*guide.count
    for row_number,(row,inside) in enumerate(zip(rows,shell)):
        branch,index=row['proposal_branch'],row['proposal_component']
        assert branch in counts
        counts[branch]+=1
        if branch=='uniform-shell':
            assert index is None and inside, 'A uniform-shell draw lies outside its support'
            if ray is not None:
                assert row.get('selected_ray_fallback') is None, 'Uniform branch gained ray fallback state'
        else:
            assert guide.alpha<1. and type(index) is int and 0<=index<guide.count, 'Invalid guided branch label'
            component_counts[index]+=1
            if entry_support is not None:
                assert entry_support[row_number,index], 'Selected entry shell does not contain proposed pose'
            if ray is not None:
                fallback=row['selected_ray_fallback']
                assert type(fallback) is bool and fallback==bool(ray['empty'][row_number,index]), 'Ray fallback differs'
                assert inside and ray['support'][row_number,index], 'Selected radial interval does not contain pose'
                ray_fallbacks[index]+=int(fallback)
        if not inside:
            assert row['log_importance_weight'] is None and row['log_hard_weight'] is None and not row['clouds'], 'Out-of-shell draw was conditioned away or assigned nonzero weight'
    result=dict(draws=len(rows),shell_rejected=int((~shell).sum()),branch_counts=counts,
        maximum_log_proposal_density_error=float(np.max(errors)),
        maximum_latent_vector_reconstruction_error=latent_error,maximum_log_jacobian_error=jacobian_error,
        scope='Every unconditional draw is retained. Branch labels are checked against the frozen law and support; random numbers are not regenerated. Full q includes all untruncated Gaussians, regardless of the selected branch.')
    component_key='ray_component_counts' if ray is not None else ('entry_shell_component_counts' if entry_support is not None else 'gaussian_component_counts')
    result[component_key]=component_counts
    if entry_support is not None:
        result['scope']='Every unconditional draw retained. Exact angular marginal times all overlapping world-space member-shell densities, plus full-ball uniform support; no conditioning on the physical target or hard validity. Saved source-branch membership checked independently.'
    if ray is not None:
        assert shell.all(), 'Conditional-ray proposal left original latent ball'
        result['ray_fallback_counts']=ray_fallbacks
        result['scope']='Exact original angular and directional marginals; full radial interval-mixture density includes every width and empty-ray fallback. No orientation/direction retry or conditioning on hard/native validity. Every attempted draw remains in N.'
    return result


def analyze(root):
    manifest=read(root/"manifest.json");region=read(root/"provenance/region.json")
    for name,digest in manifest['archive_sha256'].items():
        assert sha(root/'provenance'/name)==digest, f'Changed archived input: {name}'
    assert sha(root/'provenance/region.json')==manifest['region_sha256']
    guide,guide_sha256=importance_guide_binding(root,manifest,region)
    importance_audits=[]
    if guide is not None:
        assert len(manifest['jobs'])==len({j['seed'] for j in manifest['jobs']})>0, 'Independent importance populations require distinct seeds'
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
        importance=validate_population_guide(directory,summary['manifest'],guide,guide_sha256)
        extended=summary['manifest']['schema']=='uniform-latent-region-normalizer-v2' or importance
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
        if importance:
            proposal_audit=audit_importance_rows(rows,region,guide,chart)
            assert summary['shell_rejected']==proposal_audit['shell_rejected']
            importance_audits.append(proposal_audit)
        logs=[];hard_logs=[];pairs=[]
        for r in rows:
            valid=r["hard_valid"] and r["region_valid"] and (not importance or r['shell_valid'])
            if not importance:
                assert inner*(1-1e-12)<=r["latent_radius"]<=region["mahalanobis_radius"]*(1+1e-12)
            assert r['region_valid']==original_q_contains(r['q'],window)
            if extended:
                assert abs(native_q(region['physical_metric'],r['pose'])-r['q'])<2e-8
                assert r['capture_valid']==(math.dist(r['pose']['position'],region['capture_center'])<=region['capture_radius'])
            if valid:
                assert r['capture_valid'] and original_q_contains(r['q'],window)
                assert len(r["clouds"])==2
                geometric_weight=(r['log_physical_jacobian']-r['log_proposal_density'] if importance
                    else log_volume+r['log_physical_jacobian'])
                p=[geometric_weight+c["log_weight"] for c in r["clouds"]]
                assert abs(r['log_hard_weight']-geometric_weight)<1e-10
                for cloud in r['clouds']:
                    expected=summary['manifest']['activity']*cloud['lower_volume']+cloud['overlap_points']*math.log1p(summary['manifest']['activity']/summary['manifest']['lambda'])
                    assert abs(cloud['log_weight']-expected)<1e-10
                assert abs(logsumexp(p)-np.log(2)-r["log_importance_weight"])<1e-10
                logs.append(r["log_importance_weight"]);hard_logs.append(r["log_hard_weight"]);pairs.append(p)
            else:
                assert r["log_importance_weight"] is None and not r["clouds"]
                if importance:assert r['log_hard_weight'] is None
                logs.append(-np.inf);hard_logs.append(-np.inf);pairs.append([-np.inf,-np.inf])
        estimate=moments(logs)
        recorded=summary["estimates"]["region"]["logQ"]
        assert recorded is None if estimate["logQ"] is None else abs(estimate["logQ"]-recorded)<1e-10
        population=dict(id=job["id"],seed=job["seed"],estimate=estimate,hard_region=moments(hard_logs),samples_sha256=sample_hash)
        if importance:
            hard_estimate=population['hard_region']['logQ']
            recorded_hard=summary['estimates']['hard_region']['logQ']
            assert recorded_hard is None if hard_estimate is None else abs(hard_estimate-recorded_hard)<1e-10
            population['importance_sampling_audit']=proposal_audit
        populations.append(population)
        all_logs.extend(logs);all_hard.extend(hard_logs);all_pairs.extend(pairs);cpu+=summary["sampler_cpu_seconds"]
        backmap=max(backmap,summary["maximum_backmap_error"])
        if importance:
            audited_poses+=len(rows)
            density_error=max(density_error,proposal_audit['maximum_log_jacobian_error'])
            continue
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
    if guide is not None:
        guided_branch=getattr(guide,'branch','gaussian')
        result['importance_sampling']=dict(guide_sha256=guide_sha256,
            uniform_shell_probability=guide.alpha,
            draws=len(all_logs),shell_rejected=sum(p['shell_rejected'] for p in importance_audits),
            branch_counts={key:sum(p['branch_counts'][key] for p in importance_audits) for key in ('uniform-shell',guided_branch)},
            maximum_log_proposal_density_error=max(p['maximum_log_proposal_density_error'] for p in importance_audits),
            maximum_latent_vector_reconstruction_error=max(p['maximum_latent_vector_reconstruction_error'] for p in importance_audits),
            density_measure=DENSITY_MEASURE,
            scope='Fixed frozen proposal: q=alpha*1_shell/V+(1-alpha)*untruncated Gaussian mixture. Physical weights use J/q and all outside-shell, hard-invalid or q-invalid draws remain zeros in the original attempted denominator. Independence from guide-construction data is a campaign requirement. No bound on mass outside the target shell.')
        count_key='conditional_ray_component_count' if isinstance(guide,ConditionalRayGuide) else ('entry_shell_component_count'if isinstance(guide,EntryShellGuide)else'gaussian_component_count')
        result['importance_sampling'][count_key]=guide.count
        if isinstance(guide,(EntryShellGuide,ConditionalRayGuide)):
            result['importance_sampling'].update(guide_schema=guide.schema,proposal_kind=guide.proposal_kind,
                scope='Frozen mixture of full latent-ball uniform support and exact angular-marginal/member-shell draws. Full density sums every overlapping shell. Physical weights use J/q; all out-of-ball, hard-invalid and q-invalid draws remain unconditional zeros. No change to physical region, native criterion or bath; no bound on unmeasured regions.')
        if isinstance(guide,ConditionalRayGuide):
            result['importance_sampling'].update(
                ray_fallback_counts=[sum(p['ray_fallback_counts'][i]for p in importance_audits)for i in range(guide.count)],
                scope='Normalized conditional-ray mixture inside the unchanged latent ball; exact angular and directional marginals, radial r^3 interval masses and empty-ray fallback, plus positive uniform floor. Physical weights J/q preserve every invalid zero. No whole-vessel convergence claim.')
    out=root/"assessment";out.mkdir(exist_ok=True)
    write(out/"analysis.json",result)
    observed=(f"log Q = {estimate['logQ']:.6f}, ESS {estimate['ess']:.1f}, "
        f"largest contribution {estimate['max_fraction']:.2%}, observed relative SE {estimate['relative_se']:.2%}."
        if estimate["logQ"] is not None else "no nonzero observations: unresolved regional mass, not a physical zero or upper bound.")
    sampling=('defensive-mixture six-dimensional draws (including outside-shell zeros)' if guide is not None
        else 'uniform six-dimensional ball/shell draws')
    report=(f"# Direct integration of the frozen region\n\n"
        f"{len(all_logs):,} unconditional {sampling}; {observed}\n\n"
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
