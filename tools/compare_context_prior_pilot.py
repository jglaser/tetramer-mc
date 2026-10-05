#!/usr/bin/env python3
"""All-24 completed-summary comparison; no raw replay or geometry queries.

Report each independent stream and retain undefined ESS. Empirical occupancy
agreement is descriptive: autocorrelated finite traces do not establish weights.
"""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ARMS=('local','original','context')
STARTS=('saved_body77','highest_original_prior_valid_neighbor_distinct_center')


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probability_comparison(left,right):
    """Explicit support and total variation, including both exclusive supports."""
    for value in (left,right):
        require(value and all(math.isfinite(p) and p>=0 for p in value.values())
            and abs(sum(value.values())-1)<1e-9,'Invalid complete categorical occupancy')
    a={key for key,p in left.items() if p>0};b={key for key,p in right.items() if p>0}
    common=a&b;union=a|b
    return dict(left_support=len(a),right_support=len(b),intersection=len(common),union=len(union),
        support_jaccard=len(common)/len(union),
        total_variation=.5*sum(abs(left.get(k,0)-right.get(k,0)) for k in union),
        left_probability_in_shared_support=sum(left[k] for k in common),
        right_probability_in_shared_support=sum(right[k] for k in common),
        complete_support_partition=dict(shared=sorted(common),left_only=sorted(a-b),right_only=sorted(b-a)))


def average_probabilities(tables):
    result=defaultdict(float)
    for table in tables:
        for k,p in table.items():result[k]+=p/len(tables)
    return dict(result)


def patch_marginal_comparison(left,right):
    def dictionary(table):return {tuple(x['token']):x['fraction'] for x in table}
    a,b=dictionary(left),dictionary(right);keys=sorted(a.keys()|b.keys())
    rows=[dict(token=k,left=a.get(k,0),right=b.get(k,0),right_minus_left=b.get(k,0)-a.get(k,0)) for k in keys]
    differences=[abs(row['right_minus_left']) for row in rows]
    return dict(tokens=rows,mean_absolute_difference=statistics.mean(differences) if differences else 0.,
        maximum_absolute_difference=max(differences,default=0.),
        scope='Patch marginal differences; these overlapping event probabilities do not sum to one and have no categorical TV interpretation.')


def pair_comparison(left,right,cadence):
    a,b=left['metrics'][cadence],right['metrics'][cadence]
    output=dict(left=left['identity'],right=right['identity'],cadence=cadence,
        fingerprint=probability_comparison(a['fingerprint']['occupancy'],b['fingerprint']['occupancy']),
        neighbor_set=probability_comparison(a['neighbor_sets']['occupancy'],b['neighbor_sets']['occupancy']),
        patch_marginals=patch_marginal_comparison(a['patch_occupancy'],b['patch_occupancy']))
    output['empty_contact_fraction']=[a['empty_contact_fraction'],b['empty_contact_fraction']]
    output['singleton_fingerprint_fraction']=[a['singleton_fingerprint_fraction'],b['singleton_fingerprint_fraction']]
    return output


def efficiency_pair(left,right,cadence,descriptor):
    a=left['metrics'][cadence][descriptor]['apparent_ess_per_sampling_CPU_second']
    b=right['metrics'][cadence][descriptor]['apparent_ess_per_sampling_CPU_second']
    valid=a is not None and b is not None
    if valid:require(a>0 and b>0 and math.isfinite(a+b),'Invalid finite efficiency')
    return dict(start=left['identity']['start'],stream=left['identity']['stream'],cadence=cadence,descriptor=descriptor,
        baseline_arm=left['identity']['arm'],tested_arm=right['identity']['arm'],baseline=a,tested=b,
        difference=b-a if valid else None,ratio=b/a if valid else None,
        undefined_reason=None if valid else 'At least one constant/undefined observed descriptor; no imputation or efficiency rank.')


def compare(reports):
    expected=set(itertools.product(STARTS,range(4),ARMS))
    indexed={(r['identity']['start'],r['identity']['stream'],r['identity']['arm']):r for r in reports}
    require(len(reports)==len(indexed)==24 and set(indexed)==expected,'Expected every frozen stream/start/arm exactly once')
    for key,r in indexed.items():
        require(r['complete'] is True and r['passed'] is True and r['journal_audit']['attempts']==11520,'Incomplete chain analysis')
        require(r['metrics']['all_attempts']['samples']==10240 and r['metrics']['cycle_endpoints']['samples']==2048,'Filtered or changed production denominator')
        require(r['prior_active']==(key[2]=='context'),'Arm/prior identity differs')
        require(r['native_classifier_calls']==r['new_poses']==r['physical_draws']==0,'Observer performed undeclared work')
    paired=[];initializations=[];within_start=[];pooled=[];streams=[]
    for start,stream,arm in sorted(indexed):
        r=indexed[start,stream,arm];m=r['metrics'];a=m['all_attempts'];ab=m['certified_environment_exchanges']
        streams.append(dict(identity=r['identity'],costs=r['costs'],
            offline_observer_cpu_seconds=r['offline_observer_cpu_seconds'],
            patch_ess=a['patch_ess'],fingerprint_ess=a['fingerprint_ess'],neighbor_ess=a['neighbor_ess'],
            singleton_fingerprint_fraction=a['singleton_fingerprint_fraction'],unique_fingerprints=a['unique_fingerprints'],
            empty_contact_fraction=a['empty_contact_fraction'],ABOther=ab['instantaneous_unconditional_occupancy'],
            AB_directional_counts=ab['directional_counts'],AB_returns=len(ab['completed_returns']),
            AB_nonoverlapping_roundtrips=len(ab['nonoverlapping_roundtrips']),
            persistent_partner_exchanges=m['persistent_neighbors']['completed_partner_exchanges'],
            proposal_and_bath_factors=m['proposal_and_bath_factors']))
    for cadence in ('all_attempts','cycle_endpoints'):
        for start,stream in itertools.product(STARTS,range(4)):
            for a,b in (('local','original'),('local','context'),('original','context')):
                for descriptor in ('patch_ess','fingerprint_ess','neighbor_ess'):
                    paired.append(efficiency_pair(indexed[start,stream,a],indexed[start,stream,b],cadence,descriptor))
        for arm in ARMS:
            # All sixteen between-start pairs plus the six pairs within each
            # start reveal initialization disagreement against stream variability.
            for i,j in itertools.product(range(4),repeat=2):
                initializations.append(pair_comparison(indexed[STARTS[0],i,arm],indexed[STARTS[1],j,arm],cadence))
            for start in STARTS:
                for i,j in itertools.combinations(range(4),2):
                    within_start.append(pair_comparison(indexed[start,i,arm],indexed[start,j,arm],cadence))
            averaged=[]
            for start in STARTS:
                rs=[indexed[start,s,arm]['metrics'][cadence] for s in range(4)]
                fp=average_probabilities([r['fingerprint']['occupancy'] for r in rs])
                ns=average_probabilities([r['neighbor_sets']['occupancy'] for r in rs])
                pm=average_probabilities([{tuple(p['token']):p['fraction'] for p in r['patch_occupancy']} for r in rs])
                averaged.append(dict(fp=fp,ns=ns,pm=[dict(token=k,fraction=v) for k,v in sorted(pm.items())]))
            pooled.append(dict(arm=arm,cadence=cadence,streams_per_start=4,
                fingerprint=probability_comparison(averaged[0]['fp'],averaged[1]['fp']),
                neighbor_set=probability_comparison(averaged[0]['ns'],averaged[1]['ns']),
                patch_marginals=patch_marginal_comparison(averaged[0]['pm'],averaged[1]['pm']),
                scope='Equal-stream empirical occupancy average only. Trajectories were not concatenated and no pooled ESS is calculated.'))
    grouped=defaultdict(list)
    for row in paired:grouped[(row['start'],row['cadence'],row['descriptor'],row['baseline_arm'],row['tested_arm'])].append(row)
    ratios=[]
    for key,rows in sorted(grouped.items()):
        values=[r['ratio'] for r in rows if r['ratio'] is not None]
        ratios.append(dict(start=key[0],cadence=key[1],descriptor=key[2],baseline_arm=key[3],tested_arm=key[4],
            streams=4,defined_streams=len(values),undefined_streams=4-len(values),
            median_defined_ratio=statistics.median(values) if values else None,
            improved_defined_streams=sum(x>1 for x in values),
            limitation='Defined-subset descriptive ratio; omitted constants remain explicitly undefined, not evidence of improvement.'))
    return dict(streams=streams,paired_efficiency=paired,efficiency_ratio_summaries=ratios,
        between_start_comparisons=initializations,within_start_comparisons=within_start,
        equal_stream_average_start_comparisons=pooled,
        costs=dict(total_full_sampler_cpu_seconds=sum(r['costs']['invocation_cpu_seconds'] for r in reports),
            total_offline_observer_cpu_seconds=sum(r['offline_observer_cpu_seconds'] for r in reports),
            shared_prior_construction_cpu_seconds=None,total_end_to_end_cpu_seconds=None,
            missing_cost_note='Shared prior construction, certification and execution-controller overhead are not allocated among chains. Sampler preparation is already included per invocation; no fictitious amortization is used.'),
        scope='Conditional fixed263-body neighborhood inherited from500uM, rd1.5/activity.035. No equilibrium weight, native-registry, finite-system106.8uM assembly, stability or physical-time conclusion. Apparent ESS and occupancy agreement may miss unsampled modes. Unequal equilibrium occupancies do not require equal transition rates or frequent returns.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),'Fresh output required')
    inventory=read(args.inventory);reports=[];pins={str(args.inventory.resolve()):sha(args.inventory)}
    for item in inventory:
        path=Path(item['result'])/'report.json';receipt=read(item['execution_receipt'])
        require(receipt['success'] is True and receipt['child_drained'] is True and receipt['returncode']==0
            and Path(receipt['terminal']['path']).resolve()==path.resolve()
            and receipt['terminal']['sha256']==sha(path),'Missing completed/drained observer authority')
        r=read(path)
        require(all(r['identity'][k]==item[k] for k in ('start','stream','arm')),'Observer identity differs')
        reports.append(r);pins[str(path.resolve())]=sha(path)
        pins[str(Path(item['execution_receipt']).resolve())]=sha(item['execution_receipt'])
    result=dict(schema='fixed-context-prior-pilot-comparison-v1',complete=True,passed=True,
        input_sha256=pins,**compare(reports),geometry_queries=0,physical_draws=0)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
