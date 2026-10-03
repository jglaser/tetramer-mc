#!/usr/bin/env python3
"""Fixed analytic-sphere physical validation; no new poses or cloud points."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from native_class_line_physical_reference import equal_sphere_overlap, poisson_weight_moments


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def zscore(value,variance):
    if variance==0:
        require(abs(value)<1e-10,'Nonzero residual with zero analytic variance');return 0.
    return value/math.sqrt(variance)


def analyze(root):
    root=Path(root).resolve();plan=read(root/'allocation.json');reference=read(plan['analytic_reference']);execution=read(root/'execution-summary.json')
    binding=read(root/'execution-binding.json')
    require(execution['binding_sha256']==sha(root/'execution-binding.json') and execution['allocation_sha256']==sha(root/'allocation.json'), 'Execution binding changed')
    for path,digest in binding['files'].items():require(sha(path)==digest,'Frozen execution source/input changed '+path)
    require(execution['executable_sha256']==binding['executable_sha256']==sha(binding['executable']), 'Physical executable changed')
    require(execution['complete'] and execution['passed'],'Incomplete or failed toy execution')
    require(plan['total_attempts']==32768 and len(plan['jobs'])==16,'Changed analytic allocation')
    for path,digest in plan['source_and_input_sha256'].items():require(sha(path)==digest,'Frozen input changed '+path)
    tests=[];population=[];groups={};bindings={str(root/'allocation.json'):sha(root/'allocation.json'),str(root/'execution-summary.json'):sha(root/'execution-summary.json')}
    for job in plan['jobs']:
        directory=Path(job['out']);audit_path=root/'audits'/(job['id']+'.json');audit=read(audit_path)
        require(audit['complete'] and audit['passed'] and audit['synthetic'],'Missing independent synthetic physical audit')
        require(audit['samples']==job['samples'] and audit['seed']==job['seed'],'Different analytic population')
        require(audit['executable_sha256']==binding['executable_sha256'], 'Audit executable differs')
        for path,digest in audit['input_sha256'].items():
            if Path(path).suffix=='.py':require(binding['files'].get(path)==digest,'Audit source outside frozen execution closure')
        for path,digest in audit['input_sha256'].items():require(sha(path)==digest,'Audit input changed '+path)
        bindings[str(audit_path)]=sha(audit_path)
        summary=read(directory/'summary.json');manifest=summary['manifest'];require(manifest['activity']==job['activity'] and manifest['cloud_replicates']==2,'Different physical law')
        rows=[json.loads(s) for s in (directory/'samples.jsonl').read_text().splitlines()]
        require(len(rows)==job['samples'] and [r['draw']for r in rows]==list(range(job['samples'])),'All-attempt denominator lost')
        mass=math.exp(audit['estimate']['logQ']);hard=math.exp(audit['hard_region']['logQ'])
        groups.setdefault((job['activity'],job['method']),[]).append(dict(mass=mass,hard=hard))
        sums=dict(raw_residual=0.,raw_variance=0.,overlap_residual=0.,overlap_variance=0.,
                  weight_residual=0.,weight_variance=0.,pair_difference=0.,pair_difference_variance=0.,
                  replica_product=0.,replica_product_variance=0.)
        counts=dict(attempts=len(rows),valid=0,hard_invalid=0,exterior=0,clouds=0)
        maximum_envelope_slack=0.
        for row in rows:
            counts['hard_invalid']+=int(not row['hard_valid']);counts['exterior']+=int(not row['shell_valid'])
            if not row['clouds']:continue
            counts['valid']+=1;counts['clouds']+=2
            distance=math.hypot(*row['pose']['position']);true_volume=equal_sphere_overlap(1.5,distance)
            require(distance>=2.-1e-12,'Hard-valid analytic sphere overlaps')
            weights=[];variance=None;expected_weight=math.exp(job['activity']*true_volume)
            for cloud in row['clouds']:
                lower,upper,uncertain=(cloud[k]for k in ['lower_volume','upper_volume','uncertain_volume'])
                slack=max(lower-true_volume,true_volume-upper,0.);maximum_envelope_slack=max(maximum_envelope_slack,slack)
                require(slack<=1e-9,'Analytic overlap outside certified envelope')
                require(abs(upper-lower-uncertain)<1e-9,'Envelope volumes inconsistent')
                overlap_uncertain=max(0.,true_volume-lower)
                mean,variance=poisson_weight_moments(job['activity'],manifest['lambda'],lower,overlap_uncertain)
                require(abs(mean-expected_weight)<1e-10,'PGF mean differs from analytic overlap weight')
                weight=math.exp(cloud['log_weight']);weights.append(weight)
                sums['weight_residual']+=weight-expected_weight;sums['weight_variance']+=variance
                if job['activity']>0:
                    raw_mean=manifest['lambda']*uncertain;overlap_mean=manifest['lambda']*overlap_uncertain
                    sums['raw_residual']+=cloud['raw_points']-raw_mean;sums['raw_variance']+=raw_mean
                    sums['overlap_residual']+=cloud['overlap_points']-overlap_mean;sums['overlap_variance']+=overlap_mean
            sums['pair_difference']+=weights[0]-weights[1];sums['pair_difference_variance']+=2*variance
            sums['replica_product']+=(weights[0]-expected_weight)*(weights[1]-expected_weight)
            sums['replica_product_variance']+=variance*variance
        scores={name:zscore(sums[a],sums[b])for name,a,b in[
            ('raw_count','raw_residual','raw_variance'),('overlap_count','overlap_residual','overlap_variance'),
            ('weight','weight_residual','weight_variance'),('replica_difference','pair_difference','pair_difference_variance'),
            ('replica_centered_product','replica_product','replica_product_variance')]}
        for name,z in scores.items():tests.append(dict(name=job['id']+':'+name,z=z,passed=abs(z)<=plan['criteria']['cloud_count_or_weight_z_limit']))
        population.append(dict(id=job['id'],mass=mass,hard=hard,counts=counts,cloud_scores=scores,
            maximum_envelope_violation_A3=maximum_envelope_slack,physical_cpu_seconds=audit['sampler_cpu_seconds'],audit_cpu_seconds=audit['analysis_cpu_seconds']))
    aggregate={}
    for (activity,method),items in groups.items():
        require(len(items)==4,'Incomplete independent analytic populations')
        key=f'z{activity:g}-{method}';aggregate[key]={}
        for kind in ['mass','hard']:
            values=[p[kind]for p in items];mean=statistics.mean(values);se=statistics.stdev(values)/2
            expected=reference['references'][str(activity) if kind=='mass' else 'hard']['mass']
            z=zscore(mean-expected,se*se)
            aggregate[key][kind]=dict(mean=mean,population_SE=se,analytic=expected,z=z,per_population=values)
            tests.append(dict(name=key+':'+kind+'-analytic',z=z,passed=abs(z)<=plan['criteria']['reference_mass_combined_SE_limit']))
    for activity in [0.,.3]:
        for kind in ['mass','hard']:
            a=aggregate[f'z{activity:g}-hard-free'][kind];b=aggregate[f'z{activity:g}-class'][kind]
            z=zscore(a['mean']-b['mean'],a['population_SE']**2+b['population_SE']**2)
            tests.append(dict(name=f'z{activity:g}:{kind}-proposal-agreement',z=z,passed=abs(z)<=plan['criteria']['proposal_difference_combined_SE_limit']))
    for method in ['hard-free','class']:
        a=aggregate[f'z0-{method}']['hard'];b=aggregate[f'z0.3-{method}']['hard']
        z=zscore(a['mean']-b['mean'],a['population_SE']**2+b['population_SE']**2)
        tests.append(dict(name=method+':hard-activity-agreement',z=z,passed=abs(z)<=plan['criteria']['hard_depletion_difference_combined_SE_limit']))
    require(sum(p['counts']['attempts']for p in population)==32768,'Changed complete denominator')
    require(sum(p['counts']['hard_invalid']for p in population)>0 and sum(p['counts']['exterior']for p in population)>0,'Declared zero-accounting controls not exercised')
    for path,digest in bindings.items():require(sha(path)==digest,'Audit evidence changed during analysis')
    for path,digest in binding['files'].items():require(sha(path)==digest,'Frozen execution changed during analysis')
    return dict(schema='native-class-line-physical-analytic-sphere-validation-v1',complete=True,passed=all(t['passed']for t in tests),
        executable_sha256=execution['executable_sha256'],populations=population,aggregate=aggregate,tests=tests,
        failed_tests=[t for t in tests if not t['passed']],attempts=32768,protein_queries=0,new_pose_draws=0,new_Poisson_clouds=0,
        input_sha256=bindings,allocation_sha256=sha(root/'allocation.json'),analytic_reference_sha256=sha(plan['analytic_reference']),
        scope='Fixed sphere reference limits, full J W/q denominator and Poisson expectation/variance checks. No native thermodynamic or protein assembly conclusion; five-SE diagnostics are not formal convergence guarantees.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();require(not a.out.exists(),'Fresh analysis output required');result=analyze(a.root)
    with a.out.open('x')as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=True,passed=result['passed'],tests=len(result['tests']),failed_tests=result['failed_tests'])))
