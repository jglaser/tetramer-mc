#!/usr/bin/env python3
"""Post-hoc saved-count overlap diagnostic; never changes a physical decision."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from analyze_factorized_dimer_physical import distribution, require, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--passive-analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    physical=json.loads(args.analysis.read_text())
    passive=json.loads(args.passive_analysis.read_text())
    require(physical['complete'] and physical['passed'] and passive['complete'] and passive['passed'],'Complete audited inputs required')
    require(physical['checked_input_hashes'][str(args.passive_analysis.resolve())]==sha(args.passive_analysis),'Changed passive analysis')
    require(len(physical['rows'])==len(passive['rows'])==1536,'Lost unconditional rows')
    z,lam=.0275,1.76
    rows=[]
    for row,prior in zip(physical['rows'],passive['rows']):
        require(all(row[k]==prior[k] for k in ('atlas','case','attempt','method')),'Wrong source/contact join')
        if not row['candidate']:
            continue
        g,l=row['gained'],row['lost']
        ell=z*(g/lam-l/(lam+z))
        variance=z*z*(g/(lam*lam)+l/((lam+z)**2))
        penalty=ell-row['log_depletion_factor']
        require(penalty>=-1e-10 and variance>=0,'Impossible count diagnostic')
        internal_only=prior['contact_change']['old_contacts']==1 and row['external_contacts']==0
        rows.append(dict(index=row['index'],atlas=row['atlas'],case=row['case'],method=row['method'],
            log_physical_bath_ratio_estimate=ell,variance_estimate=variance,standard_error_estimate=math.sqrt(variance),
            estimated_auxiliary_log_penalty=penalty,internal_only_endpoints=internal_only,
            pair_overlap_volume_change_estimate_A3=ell/z if internal_only else None))
    def summary(rs):
        return dict(candidates=len(rs),negative_bath_estimates=sum(r['log_physical_bath_ratio_estimate']<0 for r in rs),
            **{key:distribution([r[key] for r in rs]) for key in ('log_physical_bath_ratio_estimate','standard_error_estimate','estimated_auxiliary_log_penalty')})
    groups=defaultdict(list)
    for row in rows:groups[(row['atlas'],row['method'])].append(row)
    result=dict(schema='factorized-dimer-saved-count-diagnostic-v1',complete=True,passed=True,
        exploratory=True,new_clouds=0,new_pose_draws=0,unconditional_outer_count=1536,candidates=len(rows),proposal_nulls=565,
        input_hashes={str(p.resolve()):sha(p) for p in (args.analysis,args.passive_analysis,Path(__file__))},
        formulas=dict(log_ratio='z*(G/lambda-L/(lambda+z))',variance='z^2*(G/lambda^2+L/(lambda+z)^2)'),
        summary=summary(rows),internal_only_summary=summary([r for r in rows if r['internal_only_endpoints']]),
        comparisons=[dict(atlas=a,method=m,summary=summary(rs),internal_only=summary([r for r in rs if r['internal_only_endpoints']])) for (a,m),rs in sorted(groups.items())],
        rows=rows,limitations=[
            'Exploratory arithmetic on all saved candidates; not an equilibrium or free-energy-of-association calculation.',
            'Formula unbiasedness assumes the intended untruncated independent Poisson thinning law conditional on pose and path order; floating-point geometry remains an obligation.',
            'The estimated variance is conditional on pose/order. Zero counts do not prove zero volume or uncertainty.',
            'The nonnegative difference from realized gate logweight estimates an auxiliary Jensen penalty and is correlated with the logratio estimate.',
            'No plug-in exponential is used as an unbiased weight, certified acceptance bound or replacement MH criterion.',
            'Internal-only stratum is selected solely from audited endpoint geometry; null proposals have no energy estimate.'])
    with args.output.open('x') as out:json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','comparisons')},indent=2))


if __name__=='__main__':main()
