"""Select the predeclared saved-pose core/tail panel without geometry or draws."""
import argparse
import hashlib
import json
import math
from pathlib import Path

SALT = 'context-overlap-core-tail-20261005-v1'
BINS = (0.,6.,12.,24.,48.,math.inf)
EXPECTED_COUNTS = (1118,458,99,6,0)
LIMITS = dict(raw_per_cloud=500000,raw_per_pose=1000000,raw_total=24000000,
    processed_per_cloud=500000,processed_per_pose=1000000,processed_total=24000000,callback_interval=8192)


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def canonical_id(row):
    require(row['arm'] in ('full','diagonal') and type(row['stream']) is int and 0<=row['stream']<4
            and type(row['ordinal']) is int and 0<=row['ordinal']<2048,'Invalid saved identity')
    return json.dumps({k:row[k] for k in ('arm','stream','ordinal')},sort_keys=True,separators=(',',':'))


def rank_digest(row):
    return hashlib.sha256(SALT.encode('utf8')+canonical_id(row).encode('utf8')).hexdigest()


def bin_index(value):
    require(type(value) in (float,int) and math.isfinite(value) and value>=0,'Invalid saved radius')
    return next(i for i in range(5) if BINS[i]<=value<BINS[i+1])


def select_panel(records,expected_counts=EXPECTED_COUNTS):
    groups=[[] for _ in range(5)];seen=set()
    for row in records:
        identity=canonical_id(row);require(identity not in seen,'Duplicate attempted identity');seen.add(identity)
        if not row['is_A_T']:continue
        require(row['physical_valid'] and row['region']=='A_patch_complete','Inconsistent saved A_T indicator')
        index=bin_index(row['mahalanobis_squared']['full'])
        groups[index].append(row)
    require(tuple(map(len,groups))==tuple(expected_counts),'Changed frozen stratum inventory')
    require(not groups[4],'Unallocated outer stratum')
    selected=[];proof=[]
    for index,group in enumerate(groups[:4]):
        ordered=sorted(group,key=lambda r:(rank_digest(r),canonical_id(r)))
        count=len(group);take=min(6,count)
        require(take>0,'Empty occupied stratum')
        identities=[]
        for rank,row in enumerate(ordered):
            item=dict(identity=json.loads(canonical_id(row)),hash_sha256=rank_digest(row),rank=rank,
                      selected=rank<take)
            identities.append(item)
            if rank<take:selected.append(dict(record=row,bin_index=index,rank=rank,hash_sha256=item['hash_sha256'],
                population=count,take=take,inclusion_probability=take/count))
        proof.append(dict(bin_index=index,lower=BINS[index],upper=BINS[index+1],population=count,take=take,
                          inclusion_probability=take/count,ranked_identities=identities))
    return selected,proof


def bound_cost(lower,uncertain,z=.035,lam=2.24):
    require(all(math.isfinite(v) and v>=0 for v in (lower,uncertain,z)) and math.isfinite(lam) and lam>0,'Invalid saved envelope')
    exponent=z*z*uncertain/lam
    return dict(lower_volume=lower,uncertain_volume=uncertain,upper_volume=lower+uncertain,
        log_physical_weight_bounds=[z*lower,z*(lower+uncertain)],
        two_cloud_relative_variance_upper_bound=math.expm1(exponent)/2 if exponent<710 else None,
        relative_variance_bound_log_exponent=exponent,
        two_cloud_score_variance_upper=z*z*uncertain/(2*lam),
        expected_complete_allocation_raw_points=2*lam*uncertain,
        bound_scope='Saved geometric envelope floating-point obligations remain; expectation is not a realized-count cap.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--balance-report',type=Path,required=True);p.add_argument('--expected-balance-report-sha256',required=True)
    p.add_argument('--audit-report',type=Path,required=True);p.add_argument('--expected-audit-report-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    require(not args.out.exists(),'Fresh panel output required');bindings={}
    def bind(path,expected=None):
        path=Path(path).resolve();actual=sha(path)
        require(expected is None or actual==expected,'Changed input: '+str(path));bindings[str(path)]=actual
        return dict(path=str(path),sha256=actual)
    balance_asset=bind(args.balance_report,args.expected_balance_report_sha256);balance=read(args.balance_report)
    audit_asset=bind(args.audit_report,args.expected_audit_report_sha256);audit=read(args.audit_report)
    require(balance['complete'] and balance['passed'] and balance['all_attempts']==16384
            and audit['complete'] and audit['passed'] and audit['decoded_candidates']==16384,'Incomplete evidence')
    balance_root=args.balance_report.resolve().parents[1]
    bind(balance_root/'protocol.json',balance['protocol_sha256']);protocol=read(balance_root/'protocol.json')
    bind(balance_root/'execution/status.json');status=read(balance_root/'execution/status.json')
    require(status['complete'] and status['passed'] and len(status['completed'])==1 and status['active'] is None,'Balance job incomplete')
    completed=status['completed'][0]
    require(completed['success'] and completed['child_drained'] and completed['returncode']==0
            and completed['terminal']['sha256']==balance_asset['sha256'],'Balance result not drained/authenticated')
    bind(balance_root/'execution-plan.json',status['plan_sha256'])
    receipt=balance_root/'execution/jobs'/('000-'+completed['id'])/'success.json';bind(receipt)
    require(read(receipt)==completed,'Balance receipt mismatch')
    for path,digest in balance['input_sha256'].items():bind(path,digest)
    contributions=balance_root/'result/contributions.jsonl';contributions_asset=bind(contributions,balance['contributions_sha256'])
    records=[json.loads(line) for line in contributions.read_text().splitlines()]
    require(len(records)==16384,'Missing attempted records')
    selected,proof=select_panel(records)
    require(len(selected)==24 and proof[3]['take']==proof[3]['population']==6,'Changed fixed panel allocation')
    source_files={};source_data={};population_index={(r['arm'],r['stream']):r for r in protocol['populations']}
    for key,entry in population_index.items():
        path=Path(entry['result'])/'rows.jsonl'
        source_files[key]=bind(path,audit['input_sha256'][str(path.resolve())])
        source_data[key]=[json.loads(line) for line in path.read_text().splitlines()]
        require(len(source_data[key])==2048,'Incomplete original pose bank')
    entries=[]
    for item in selected:
        row=item['record'];key=(row['arm'],row['stream']);saved=source_data[key][row['ordinal']]
        require(saved['input']['ordinal']==row['ordinal'] and saved['region']=='A_patch_complete'
                and saved['actual']['physical_valid'] and saved['patches']['source_fraction']==1,'Selected source row differs')
        require(math.isfinite(row['log_q_balanced']),'Invalid balanced proposal density')
        envelope=saved['envelope'];cost=bound_cost(envelope['lower_volume'],envelope['uncertain_volume'])
        require(math.isclose(envelope['upper_volume'],cost['upper_volume'],rel_tol=1e-14,abs_tol=1e-9),'Saved envelope arithmetic differs')
        metadata=dict(full_chart_bin_index=item['bin_index'],mahalanobis_squared=row['mahalanobis_squared'],
            log_q_balanced=row['log_q_balanced'],inclusion_probability=item['inclusion_probability'],
            stratum_population=item['population'],arm=row['arm'],stream=row['stream'],ordinal=row['ordinal'],
            selection_rank=item['rank'],selection_hash_sha256=item['hash_sha256'],selection_salt=SALT,
            original_row=saved,balance_row=row,saved_bounds=cost)
        entries.append(dict(id=f"r2bin{item['bin_index']}-{row['arm']}-s{row['stream']}-o{row['ordinal']}",
            source_rows=source_files[key],ordinal=row['ordinal'],pose=saved['input']['proposed_pose'],metadata=metadata))
    bind(__file__)
    panel=dict(schema='context-fixed-pose-overlap-panel-v1',entries=entries,
        selection=dict(salt=SALT,rank_rule='SHA256(salt UTF8 + canonical sorted compact JSON {arm,stream,ordinal}); ascending digest, then canonical ID for ties.',
            bins=[0,6,12,24,48,None],take_per_occupied_bin=6,stratum_counts=list(EXPECTED_COUNTS),proof=proof,
            probability_scope='Conditional finite-bank inclusion min(6,M)/M under exchangeable uniform hash priorities. The actual reproducible SHA256 ranking is fixed, not a fresh random sample once the salt is fixed; no survey or normalizer inference is made.'),
        balance_report=balance_asset,audit_report=audit_asset,contributions=contributions_asset,input_sha256=bindings,
        source_control=None,source_control_reason='The certified source has no saved absolute overlap envelope in this bank.',
        prospective_scoring=dict(lambda_intensity=2.24,depletant_activity=.035,clouds_per_pose=2,clouds=48,
            cloud_limits=LIMITS,cpu_seconds=600,wall_seconds=1200,memory_bytes=4*1024**3,threads=1,
            patch_node_visits_per_candidate=20000000,patch_leaf_tests_per_candidate=2000000,
            patch_node_visits_total=1000000000,patch_leaf_tests_total=100000000),
        geometry_queries_during_preparation=0,clouds_during_preparation=0,new_poses=0,
        interpretation='Fixed conditional core/tail diagnostic only. Score z*O-log(qbar) needs overlap counts and cannot be inferred from hard-only weights. Panel selection is not a regional-normalizer calculation.')
    for path,digest in bindings.items():require(sha(path)==digest,'Input changed during metadata selection')
    args.out.mkdir();path=args.out/'panel.json'
    path.write_text(json.dumps(panel,indent=2,allow_nan=False)+'\n')
    summary=dict(complete=True,passed=True,source_sha256=sha(__file__),panel_sha256=sha(path),selected=24,
        stratum_counts=list(EXPECTED_COUNTS),inclusion_probabilities=[min(6,n)/n for n in EXPECTED_COUNTS[:4]],
        expected_complete_allocation_raw_points=sum(e['metadata']['saved_bounds']['expected_complete_allocation_raw_points'] for e in entries),
        maximum_two_cloud_relative_variance_upper_bound=max(e['metadata']['saved_bounds']['two_cloud_relative_variance_upper_bound'] for e in entries),
        input_sha256=bindings,geometry_queries=0,cloud_draws=0,new_poses=0,scorer_launched=False)
    (args.out/'report.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'}))

if __name__=='__main__':main()
