"""Split observed importance second moments using two independent saved clouds.

For A=(w0+w1)/2 and P=w0*w1, Q=A^2-P=(w0-w1)^2/4.
Conditional on a pose, E[P]=m^2 and E[Q]=Var(w)/2. The normalized
ratios below are noisy diagnostics, not unbiased variances or corrected ESS.
No cloud is sampled, no geometry is queried, and primary results are unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ARMS=('baseline','broadened')
GROUPS=('full_domain','A_T','T_any','T_outside_A','remaining_without_A_T',
        'contact_without_A_T','A_partial','B','other_contact','unbound')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def logsum(values):
    if not values:return -math.inf
    high=max(values)
    return high+math.log(math.fsum(math.exp(x-high) for x in values)) if math.isfinite(high) else high


def close(a,b,name):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-9+2e-11*max(abs(a),abs(b)),
            'Saved arithmetic differs: '+name)


def cloud_pair(log_w0,log_w1):
    require(math.isfinite(log_w0) and math.isfinite(log_w1),'Nonfinite positive cloud weight')
    mean=logsum([log_w0,log_w1])-math.log(2)
    delta=.5*(log_w0-log_w1)
    fraction=math.tanh(delta)**2
    return dict(log_mean=mean,log_square=2*mean,log_product=log_w0+log_w1,
                log_noise=2*mean+math.log(fraction) if fraction else None,
                pair_noise_fraction=fraction,cloud_log_weight_difference=abs(log_w0-log_w1))


def rao_blackwell_log_weight(lower,total,intensity,activity,log_q):
    require(type(total) is int and total>=0 and all(math.isfinite(x) for x in (lower,intensity,activity,log_q))
            and lower>=0 and intensity>0 and activity>=0,'Invalid Rao-Blackwell inputs')
    return activity*lower+total*math.log1p(activity/(2*intensity))-log_q


def rb_summary(rows,denominator):
    if not rows:return dict(hits=0,denominator=denominator,mass=0.,log_mass=None,importance_ess=0.,largest_contribution=None)
    logs=[r['log_rb_weight'] for r in rows];total=logsum(logs);log_mass=total-math.log(denominator)
    return dict(hits=len(logs),denominator=denominator,mass=math.exp(log_mass),log_mass=log_mass,
        importance_ess=math.exp(2*total-logsum([2*x for x in logs])),largest_contribution=math.exp(max(logs)-total))


def memberships(row):
    valid=row['physical_valid'];region=row['region'];target=valid and region=='A_patch_complete'
    any_t=row['source_T_complete_all_regions']
    require(type(valid) is bool and type(any_t) is bool and (not any_t or valid)
            and (not target or any_t),'Inconsistent orthogonal contact labels')
    return dict(full_domain=valid,A_T=target,T_any=any_t,T_outside_A=any_t and not target,
                remaining_without_A_T=valid and not target,
                contact_without_A_T=valid and region!='unbound' and not target,
                A_partial=valid and region.startswith('A_patch_') and not target,
                B=region=='B',other_contact=region=='other_contact',unbound=region=='unbound')


def summarize(rows,denominator):
    require(type(denominator) is int and denominator>0 and len(rows)<=denominator,'Lost attempt denominator')
    if not rows:return dict(hits=0,denominator=denominator,log_mass=None,importance_ess=0.,
        largest_contribution=None,cloud_noise_fraction_of_observed_second_moment=None,
        conditional_product_ratio_not_ess=None,top=[])
    total=logsum([r['log_mean'] for r in rows]);squares=logsum([r['log_square'] for r in rows])
    products=logsum([r['log_product'] for r in rows]);noise=logsum([r['log_noise'] for r in rows if r['log_noise'] is not None])
    share=math.exp(noise-squares) if math.isfinite(noise) else 0.
    product_share=math.exp(products-squares)
    require(abs(share+product_share-1)<2e-10,'Second-moment decomposition does not close')
    leaders=sorted(rows,key=lambda r:r['log_mean'],reverse=True)[:8]
    return dict(hits=len(rows),denominator=denominator,log_mass=total-math.log(denominator),
        importance_ess=math.exp(2*total-squares),largest_contribution=math.exp(leaders[0]['log_mean']-total),
        cloud_noise_fraction_of_observed_second_moment=share,
        conditional_product_ratio_not_ess=math.exp(2*total-products),
        maximum_cloud_log_weight_difference=max(r['cloud_log_weight_difference'] for r in rows),
        top=[dict(stratum_id=r['stratum_id'],ordinal=r['ordinal'],region=r['region'],
            source_T_complete_all_regions=r['source_T_complete_all_regions'],
            mahalanobis_squared=r['mahalanobis_squared'],weight_fraction=math.exp(r['log_mean']-total),
            second_moment_fraction=math.exp(r['log_square']-squares),
            pair_noise_fraction=r['pair_noise_fraction'],
            cloud_log_weight_difference=r['cloud_log_weight_difference']) for r in leaders])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();started=time.process_time();protocol=read(args.protocol)
    require(protocol['schema']=='context-broadened-cloud-noise-v1','Wrong diagnostic schema')
    bindings=protocol['input_sha256']
    for path,digest in bindings.items():require(sha(path)==digest,'Changed input: '+path)
    def bound(path):
        path=str(Path(path).resolve());require(path in bindings,'Unbound input: '+path);return read(path)
    report=bound(protocol['physical_report']);inventory=bound(protocol['physical_inventory'])
    require(report['complete'] and report['passed'] and report['all_attempts']==32768
            and report['finite_stability_conclusion'] is False,'Primary report incomplete or wrong scope')
    contributions=Path(protocol['physical_contributions']).resolve()
    require(bindings.get(str(contributions))==report['contributions_sha256'],'Wrong audited contributions')
    indexed={}
    with contributions.open() as stream:
        for line in stream:
            row=json.loads(line);key=(row['stratum_id'],row['ordinal']);require(key not in indexed,'Duplicate attempt')
            indexed[key]=row
    require(len(indexed)==32768 and len(inventory)==20,'Changed complete allocation')
    all_rows=[];seen=set();clouds=0
    for job in inventory:
        cfg=bound(job['config']['path']);require(cfg['activity']==.035 and cfg['lambda']==2.24,'Changed cloud conditions')
        path=(Path(job['result'])/'rows.jsonl').resolve()
        require(str(path) in bindings and report['input_sha256'].get(str(path))==bindings[str(path)],'Physical rows lack primary audit binding')
        count=0
        with path.open() as stream:
            for line in stream:
                row=json.loads(line);ordinal=row['saved_row_identity']['ordinal'];key=(job['stratum_id'],ordinal)
                require(key in indexed and key not in seen,'Unknown or repeated scored attempt');seen.add(key)
                geometry=indexed[key];require(geometry['physical_valid'] and row['complete']
                    and row['pose']==geometry['proposed_pose'] and row['region']==geometry['region']
                    and len(row['clouds'])==2,'Physical/geometry identity mismatch')
                q=geometry['log_q_arm'];logs=[];total_hits=0;common_lower=row['envelope']['lower_volume']
                for cloud in row['clouds']:
                    w=cloud['weight'];progress=cloud['progress'];require(progress['complete'] and progress['begun'],'Incomplete cloud')
                    require(w['overlap_points']==progress['overlap_points'] and w['raw_points']==progress['processed_points']
                        and type(w['overlap_points']) is int and 0<=w['overlap_points']<=w['raw_points'],'Invalid count accounting')
                    expected=.035*w['lower_volume']+w['overlap_points']*math.log1p(.035/2.24)
                    close(w['lower_volume'],common_lower,'common two-cloud lower volume')
                    close(w['log_weight'],expected,'Poisson positive weight');close(progress['log_weight'],expected,'cloud terminal weight')
                    logs.append(expected-q);total_hits+=w['overlap_points'];clouds+=1
                pair=cloud_pair(*logs);close(pair['log_mean'],geometry['log_physical_weight'],'primary contribution')
                all_rows.append(dict(geometry,**pair,log_rb_weight=rao_blackwell_log_weight(common_lower,total_hits,2.24,.035,q)));count+=1
        require(count==job['valid_poses'] and bool(count)!=job['empty'],'Incomplete valid-pose stratum')
    require(seen=={key for key,row in indexed.items() if row['physical_valid']},'Not every valid draw has two clouds')
    population=[]
    for arm in ARMS:
        for pop in range(4):
            rows=[r for r in all_rows if r['comparison_arm']==arm and r['population_index']==pop]
            attempted=[r for r in indexed.values() if r['comparison_arm']==arm and r['population_index']==pop]
            require(len(attempted)==4096,'Changed population denominator')
            primary=next(r for r in report['populations'] if r['comparison_arm']==arm and r['population_index']==pop)
            groups={g:summarize([r for r in rows if memberships(r)[g]],4096) for g in GROUPS}
            for g,value in groups.items():close(value['importance_ess'],primary['groups'][g]['physical']['importance_ess'],'primary ESS '+g)
            rb_groups={g:rb_summary([r for r in rows if memberships(r)[g]],4096) for g in GROUPS}
            population.append(dict(comparison_arm=arm,population_index=pop,groups=groups,secondary_rao_blackwell=rb_groups,
                geometry_plus_scoring_cpu_seconds=primary['total_cpu_seconds']))
    pooled={arm:{g:summarize([r for r in all_rows if r['comparison_arm']==arm and memberships(r)[g]],16384)
                 for g in GROUPS} for arm in ARMS}
    rb_pooled={arm:{g:rb_summary([r for r in all_rows if r['comparison_arm']==arm and memberships(r)[g]],16384)
                 for g in GROUPS} for arm in ARMS}
    rb_population_means={}
    for arm in ARMS:
        rb_population_means[arm]={}
        for group in GROUPS:
            masses=[p['secondary_rao_blackwell'][group]['mass'] for p in population if p['comparison_arm']==arm]
            mean=statistics.mean(masses);se=statistics.stdev(masses)/2
            rb_population_means[arm][group]=dict(mean_mass=mean,standard_error=se,relative_standard_error=se/mean if mean else None)
    for path,digest in bindings.items():require(sha(path)==digest,'Input changed during diagnostic')
    output=dict(schema='context-broadened-cloud-noise-report-v1',complete=True,passed=True,
        protocol_sha256=sha(args.protocol),source_sha256=sha(__file__),input_sha256=bindings,
        all_attempts=32768,valid_poses=len(all_rows),audited_clouds=clouds,populations=population,pooled=pooled,
        secondary_rao_blackwell_pooled=rb_pooled,secondary_rao_blackwell_population_means=rb_population_means,
        cpu_seconds=time.process_time()-started,new_clouds=0,new_geometry_queries=0,new_poses=0,
        finite_stability_conclusion=False,primary_results_changed=False,
        identity='A=(w0+w1)/2,P=w0*w1,Q=A^2-P=(w0-w1)^2/4; conditional E[P]=m^2,E[Q]=Var(w)/2.',
        secondary_rao_blackwell_identity='For S=K0+K1, E[(W0+W1)/2|S]=exp(zL)*(1+z/(2lambda))^S. This unbiased positive secondary estimator uses identical saved point work, removes the random cloud split, and has relative variance exp(z^2(O-L)/(2lambda))-1, versus (exp(z^2(O-L)/lambda)-1)/2 for the original arithmetic pair. It does not remove pose-coverage error; primary allocation, weights and convergence results stay unchanged.',
        caveat='The normalized cloud-noise share and conditional-product ratio are noisy realized diagnostics. The product ratio is NOT corrected or true ESS and can exceed the number of poses. Physical mass estimates and convergence gates remain those of the unchanged primary report. A_T and T-any are source contact-token patterns, not native registry classifiers.')
    with args.out.open('x') as stream:json.dump(output,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(complete=True,passed=True,valid_poses=len(all_rows),clouds=clouds,cpu_seconds=output['cpu_seconds'])))


if __name__=='__main__':main()
