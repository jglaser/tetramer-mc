#!/usr/bin/env python3
"""Saved learned-branch marginal-Q counterfactual. No new physical evaluations.

Uniform candidates lack saved G scores and remain explicitly unevaluated. These
are one-step decisions on old trajectories, never a simulated marginal-Q chain.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

COMPARISON_SHA='9ef364c36de0d10c4099e37df9cbae56dec2831f9932c8dfa6914009d57fb397'
QUANTILES=(0.,.01,.05,.25,.5,.75,.95,.99,1.)
ARMS=('local','original','context')
STARTS=('saved_body77','highest_original_prior_valid_neighbor_distinct_center')


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def logadd(a,b):
    top=max(a,b)
    if top==-math.inf:return top
    return top+math.log1p(math.exp(min(a,b)-top))


def close(a,b,scale=1.):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-8+2e-10*(scale+abs(a)+abs(b)),
        'Saved arithmetic differs')


def scalar_distribution(values):
    ordered=sorted(values);n=len(ordered)
    require(all(math.isfinite(x) for x in ordered),'Nonfinite diagnostic')
    def quantile(p):
        x=(n-1)*p;i=int(x);j=min(i+1,n-1)
        return ordered[i]+(x-i)*(ordered[j]-ordered[i])
    return dict(n=n,quantile_probabilities=QUANTILES,quantiles=[quantile(p) for p in QUANTILES] if n else None,
        mean=math.fsum(ordered)/n if n else None)


def learned_counterfactual(row,halfwidth):
    info=row['proposal'];gx=info['full_old_gaussian_log_density'];gy=info['full_new_gaussian_log_density']
    require(math.isfinite(gx) and math.isfinite(gy) and math.isfinite(halfwidth) and halfwidth>0,'Invalid saved density/domain')
    def inside(pose):return all(math.isfinite(v) and abs(v)<=halfwidth for v in pose['position'])
    require(inside(row['old_pose']),'Old physical pose lacks uniform reverse support')
    new_inside=inside(row['proposed_pose']);logu=-3*math.log(2*halfwidth)
    qg=gx-gy;qmix=logadd(logu,gx)-logadd(logu if new_inside else -math.inf,gy)
    close(info['log_reverse_forward'],qg,1+abs(gx)+abs(gy))
    close(row['log_proposal_reverse_forward'],qg,1+abs(gx)+abs(gy))
    result=dict(old_G_minus_logU=gx-logu,new_G_minus_logU=gy-logu if new_inside else None,
        old_logG=gx,new_logG=gy,log_uniform_density=logu,candidate_inside_uniform_cube=new_inside,
        original_logq=qg,marginal_logq=qmix,logq_change=qmix-qg,
        original_accepted=row['accepted'],counterfactual_accepted=None,acceptance_changed=None)
    gate=row['gate']
    if gate is None:
        require(not row['accepted'],'Accepted hard-invalid candidate')
        result['counterfactual_accepted']=False
        return dict(result,status='hard_rejected',physical_gate_evaluated=False)
    require(row['wall_valid'] is True and row['core_valid'] is True and new_inside,'Physical candidate lacks full uniform support')
    w=gate['log_weight'];u=row['acceptance_uniform'];require(0<=u<1,'Invalid saved uniform')
    logu_draw=math.log(u) if u else -math.inf
    if u:close(row['log_uniform'],logu_draw)
    else:require(row['log_uniform'] is None,'Wrong zero-uniform schema')
    raw=qg+w;counter=qmix+w
    close(row['raw_log_acceptance'],raw,1+abs(qg)+abs(w));close(row['log_acceptance'],min(0.,raw),1+abs(qg)+abs(w))
    take=logu_draw<min(0.,raw);other=logu_draw<min(0.,counter)
    require(row['accepted'] is take,'Saved acceptance differs')
    return dict(result,status='accepted' if take else 'bath_rejected',physical_gate_evaluated=True,
        bath_log_weight=w,original_raw_log_acceptance=raw,counterfactual_raw_log_acceptance=counter,
        counterfactual_accepted=other,acceptance_changed=other!=take,
        original_acceptance_probability=math.exp(min(0.,raw)),counterfactual_acceptance_probability=math.exp(min(0.,counter)))


def reduction(rows):
    learned=[r for r in rows if r['branch']=='involution'];gated=[r for r in learned if r['physical_gate_evaluated']]
    fields=('old_G_minus_logU','new_G_minus_logU','original_logq','marginal_logq','logq_change',
        'bath_log_weight','original_raw_log_acceptance','counterfactual_raw_log_acceptance',
        'original_acceptance_probability','counterfactual_acceptance_probability')
    def summarize(selected):
        return {key:scalar_distribution([r[key] for r in selected if r.get(key) is not None]) for key in fields}
    return dict(global_attempts=len(rows),branches=dict(Counter(r['branch'] for r in rows)),
        all_branch_dispositions=[dict(branch=b,status=s,count=n) for (b,s),n in sorted(Counter((r['branch'],r['status']) for r in rows).items())],
        learned_attempts=len(learned),learned_candidate_inside_cube=sum(r['candidate_inside_uniform_cube'] for r in learned),
        learned_dispositions=dict(Counter(r['status'] for r in learned)),gated_learned=len(gated),
        original_gated_accepts=sum(r['original_accepted'] for r in gated),
        counterfactual_gated_accepts=sum(r['counterfactual_accepted'] for r in gated),
        acceptance_gains=sum(not r['original_accepted'] and r['counterfactual_accepted'] for r in gated),
        acceptance_losses=sum(r['original_accepted'] and not r['counterfactual_accepted'] for r in gated),
        all_learned=summarize(learned),gated_learned_distributions=summarize(gated),
        gated_acceptance_cross_table=[dict(original=a,counterfactual=b,count=sum(r['original_accepted']==a and r['counterfactual_accepted']==b for r in gated))
            for a,b in itertools.product((False,True),repeat=2)],
        missing_uniform_counterfactual=sum(r['branch']=='uniform' for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','comparison','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),'Fresh output required')
    require(sha(args.comparison)==COMPARISON_SHA,'Unreviewed comparison authority')
    comparison=read(args.comparison);require(comparison['complete'] is True and comparison['passed'] is True,'Incomplete primary comparison')
    inventory=read(args.inventory)
    expected=set(itertools.product(STARTS,range(4),ARMS))
    require(len(inventory)==24 and {(i['start'],i['stream'],i['arm']) for i in inventory}==expected,'All24 inventory required')
    require(comparison['input_sha256'][str(args.inventory.resolve())]==sha(args.inventory),'Different primary inventory')
    args.out.mkdir();records=[];audit=[];started=time.process_time();wall=time.monotonic()
    with (args.out/'attempts.jsonl').open('x') as output:
        def emit(row):output.write(json.dumps(row,allow_nan=False)+'\n');output.flush()
        try:
            for item in inventory:
                emit(dict(kind='chain_begun',id=item['id']))
                report_path=Path(item['result'])/'report.json';report=read(report_path)
                require(comparison['input_sha256'][str(report_path.resolve())]==sha(report_path),'Observer report changed')
                path=Path(item['physical_result'])/'events.jsonl'
                require(report['input_sha256'][str(path)]==sha(path),'Physical journal differs from audited observer input')
                count=global_count=0;halfwidth=None
                with path.open() as journal:
                    for line in journal:
                        # Rust's frozen Journal uses compact serde_json lines.
                        # Full row ordering already passed the primary observer.
                        if '"kind":"prepared"' in line:
                            prepared=json.loads(line);halfwidth=prepared['uniform_cube_half_width'];continue
                        if '"kind":"attempt_complete"' not in line:continue
                        row=json.loads(line);require(row['kind']=='attempt_complete' and row['attempt_index']==count,'Saved attempt gap')
                        require(row['cycle']==count//5+1 and row['slot']==count%5 and row['production']==(count//5+1>256),'Schedule changed')
                        expected_global=item['arm']!='local' and row['slot']==4
                        require(row['global'] is expected_global,'Changed global inventory');count+=1
                        if not expected_global:continue
                        require(time.process_time()-started<30 and time.monotonic()-wall<60,'Saved-record reduction cap')
                        emit(dict(kind='global_begun',id=item['id'],attempt_index=row['attempt_index']))
                        branch=row['proposal']['branch'];require(branch in ('uniform','involution'),'Unexpected global branch')
                        common=dict(id=item['id'],start=item['start'],stream=item['stream'],arm=item['arm'],
                            attempt_index=row['attempt_index'],cycle=row['cycle'],phase='production' if row['production'] else 'warmup',branch=branch)
                        if branch=='involution':
                            require(row['proposal']['source_law']=='posterior' and row['proposal']['identity'] is False,'Wrong learned kernel')
                            details=learned_counterfactual(row,halfwidth)
                        else:details=dict(original_accepted=row['accepted'],counterfactual_accepted=None,
                            status='accepted' if row['accepted'] else ('wall_rejected' if row['wall_valid'] is False
                                else ('core_rejected' if row['core_valid'] is False else 'bath_rejected')),
                            unavailable_reason='Uniform candidates have no saved learned endpoint densities; no new density evaluation allocated.')
                        record=dict(common,**details);records.append(record);global_count+=1
                        emit(dict(kind='global_complete',**record))
                require(count==11520 and global_count==(0 if item['arm']=='local' else 2304),'Wrong attempted denominator')
                item_audit=dict(id=item['id'],all_attempts=count,global_attempts=global_count,physical_journal_sha256=sha(path))
                audit.append(item_audit);emit(dict(kind='chain_complete',**item_audit))
            require(len(records)==36864 and sum(r['phase']=='production' for r in records)==32768,'Changed total global allocation')
            groups=defaultdict(list);pooled=defaultdict(list)
            for row in records:
                groups[(row['start'],row['stream'],row['arm'],row['phase'])].append(row)
                pooled[(row['start'],row['arm'],row['phase'])].append(row)
            report=dict(complete=True,passed=True,schema='saved-learned-marginal-Q-counterfactual-v1',
                input_sha256={str(p.resolve()):sha(p) for p in (args.inventory,args.comparison)},chain_audit=audit,
                global_attempts=len(records),physical_attempts_audited=sum(r['all_attempts'] for r in audit),
                per_stream=[dict(start=k[0],stream=k[1],arm=k[2],phase=k[3],**reduction(v)) for k,v in sorted(groups.items())],
                descriptive_pooled=[dict(start=k[0],arm=k[1],phase=k[2],**reduction(v)) for k,v in sorted(pooled.items())],
                records_sha256=sha(args.out/'attempts.jsonl'),cpu_seconds=time.process_time()-started,
                geometry_queries=0,new_density_evaluations=0,new_clouds=0,new_poses=0,physical_draws=0,
                scope='Same saved learned candidates, bath estimates and acceptance uniforms on old trajectories only. Uniform-branch counterfactuals are unavailable. No marginal-Q trajectory, complete-kernel acceptance/speedup, equilibrium weight or assembly claim.')
            with (args.out/'report.json').open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
        except BaseException as error:
            emit(dict(kind='failed',error=repr(error),complete=False));raise


if __name__=='__main__':main()
