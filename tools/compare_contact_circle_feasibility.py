#!/usr/bin/env python3
"""Compare completed Rust/Python saved-circle records; no geometry or sampling.

Only interval algebra, identities, existing witnesses and saved masses are
checked. No atomic overlap query or new angular pose is evaluated.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

TAU=2*math.pi
TOLERANCE=1e-10
CASES_SHA='dc4743a8dc06ea51f86e508cc1a3a631371a844ff2c50c28d3983b5c0236ca85'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def write(p,data):
    with Path(p).open('x') as stream:json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines()]
def length(intervals):return math.fsum(i['upper']-i['lower'] for i in intervals)
def contains(intervals,phi):
    phi=phi%TAU
    return any((phi>i['lower'] or phi==i['lower'] and i['lower_closed']) and
               (phi<i['upper'] or phi==i['upper'] and i['upper_closed']) for i in intervals)


def interval_errors(intervals):
    errors=[]
    for k,i in enumerate(intervals):
        if not all(math.isfinite(i[x]) for x in ('lower','upper')) or not 0<=i['lower']<=i['upper']<=TAU:
            errors.append(dict(index=k,error='nonfinite_or_noncanonical_bounds'));continue
        if any(type(i[x]) is not bool for x in ('lower_closed','upper_closed')):
            errors.append(dict(index=k,error='invalid_endpoint_closure'))
        if i['upper']==TAU and i['upper_closed']:errors.append(dict(index=k,error='2pi_endpoint_must_be_open'))
        if i['lower']==i['upper'] and not(i['lower_closed'] and i['upper_closed']):
            errors.append(dict(index=k,error='empty_interval_record'))
        if k:
            previous=intervals[k-1]
            if i['lower']<previous['upper'] or (i['lower']==previous['upper'] and
                    (i['lower_closed'] or previous['upper_closed'])):
                errors.append(dict(index=k,error='not_canonical_disjoint_union'))
    return errors


def partition_errors(allowed,forbidden):
    endpoints=sorted({0.,TAU,*[i[k] for seq in (allowed,forbidden) for i in seq for k in ('lower','upper')]})
    probes=[p for p in endpoints if p<TAU]
    probes += [a+(b-a)/2 for a,b in zip(endpoints,endpoints[1:]) if a<b]
    return [dict(phi=p,allowed=contains(allowed,p),forbidden=contains(forbidden,p))
            for p in probes if contains(allowed,p)==contains(forbidden,p)]


def max_difference(a,b):
    if isinstance(a,list):
        require(isinstance(b,list) and len(a)==len(b),'Array shape mismatch')
        return max((max_difference(x,y) for x,y in zip(a,b)),default=0.)
    require(math.isfinite(a) and math.isfinite(b),'Nonfinite compared scalar')
    return abs(a-b)


def compare_case(case,rust,python):
    errors={};failures=[];checks={}
    def check(name,passed,detail=None):
        checks[name]=bool(passed)
        if not passed:failures.append(dict(check=name,detail=detail))
    def difference(name,a,b):
        error=max_difference(a,b);errors[name]=error
        check(name,error<=TOLERANCE,dict(error=error,tolerance=TOLERANCE))
    check('identity',case['id']==rust['id']==python['id'])
    check('arm_population',case['arm']==rust['arm']==python['arm'] and
          case['population']==rust['population']==python['population'])
    check('width',case['width_index']==rust['width_index'])
    check('law',case['azimuth_law']==rust['azimuth_law'])
    difference('circle_center',case['circle_center'],rust['circle']['center'])
    difference('circle_radius',case['circle_radius'],rust['circle']['radius'])
    difference('independent_basis',rust['circle']['basis'],python['basis'])
    difference('moving_orientation',case['pose']['orientation'],rust['moving_orientation'])
    difference('original_phi',case['phi'],python['original_phi'])
    sets={}
    for name in ('allowed','forbidden'):
        a=rust['geometry'][name]['intervals'];b=python[name];sets[name]=(a,b)
        check(name+'_rust_canonical',not interval_errors(a),interval_errors(a))
        check(name+'_python_canonical',not interval_errors(b),interval_errors(b))
        check(name+'_interval_count',len(a)==len(b),dict(rust=len(a),python=len(b)))
        if len(a)==len(b):
            topology=all(x['lower_closed']==y['lower_closed'] and x['upper_closed']==y['upper_closed'] for x,y in zip(a,b))
            check(name+'_endpoint_topology',topology)
            difference(name+'_endpoints',[[i['lower'],i['upper']] for i in a],[[i['lower'],i['upper']] for i in b])
        difference(name+'_length',length(a),length(b))
    a,b=sets['allowed'];fa,fb=sets['forbidden']
    check('rust_complement_partition',not partition_errors(a,fa),partition_errors(a,fa))
    check('python_complement_partition',not partition_errors(b,fb),partition_errors(b,fb))
    difference('allowed_length_saved',length(b),python['allowed_length'])
    difference('uniform_probability',rust['uniform_allowed_mass'],python['uniform_allowed_probability'])
    difference('rust_uniform_from_length',rust['uniform_allowed_mass'],length(a)/TAU)
    difference('python_uniform_from_length',python['uniform_allowed_probability'],length(b)/TAU)
    difference('original_law_mass',rust['original_law_allowed_mass'],python['azimuth_allowed_probability'])
    check('probabilities_in_unit_interval',all(0<=x<=1+TOLERANCE for x in
        [rust['original_law_allowed_mass'],rust['uniform_allowed_mass'],python['azimuth_allowed_probability'],python['uniform_allowed_probability']]))
    check('python_nonempty_flags',python['has_positive_allowed_length']==(length(b)>0) and
          python['has_any_allowed_point']==bool(b))
    check('original_hard_predicate',case['hard_valid']==rust['original_hard_valid']==python['original_hard_valid'])
    check('original_phi_membership',contains(a,case['phi'])==contains(b,case['phi'])==case['hard_valid'])
    witnesses=rust['witnesses']
    check('original_phi_witness_present',bool(witnesses) and witnesses[0]['phi']==case['phi'] and witnesses[0]['hard_free']==case['hard_valid'])
    mismatches=[]
    for i,w in enumerate(witnesses):
        if not (contains(a,w['phi'])==contains(b,w['phi'])==w['hard_free']):
            mismatches.append(dict(index=i,**w,rust_contains=contains(a,w['phi']),python_contains=contains(b,w['phi'])))
    check('all_saved_witness_memberships',not mismatches,mismatches)
    return dict(id=case['id'],arm=case['arm'],population=case['population'],width_index=case['width_index'],
        passed=not failures,checks=checks,failures=failures,absolute_errors=errors,witnesses_checked=len(witnesses),
        has_positive_allowed_length=length(a)>0,has_any_allowed_point=bool(a),allowed_length=length(a),
        original_law_allowed_mass=rust['original_law_allowed_mass'],uniform_allowed_mass=rust['uniform_allowed_mass'],
        original_hard_valid=case['hard_valid'],original_shell_valid=case['shell_valid'],original_capture_valid=case['capture_valid'],
        original_hard_and_domain_valid=case['hard_valid'] and case['shell_valid'] and case['capture_valid'],
        rust_geometry_CPU_seconds=rust['geometry_cpu_seconds'],python_geometry_CPU_seconds=python['analysis_cpu_seconds'],
        rust_tree_counts=rust['geometry']['counts'],python_tree_counts=python['counts'])


def grouped_summary(records):
    def average(key):return math.fsum(r[key] for r in records)/len(records)
    rust_counts=Counter();python_counts=Counter()
    for r in records:
        rust_counts.update(r['rust_tree_counts']);python_counts.update(r['python_tree_counts'])
    return dict(cases=len(records),positive_length_circles=sum(r['has_positive_allowed_length'] for r in records),
        any_allowed_point_circles=sum(r['has_any_allowed_point'] for r in records),
        original_hard_valid=sum(r['original_hard_valid'] for r in records),
        original_hard_and_domain_valid=sum(r['original_hard_and_domain_valid'] for r in records),
        mean_allowed_length=average('allowed_length'),mean_original_law_allowed_mass=average('original_law_allowed_mass'),
        mean_uniform_allowed_mass=average('uniform_allowed_mass'),
        rust_geometry_CPU_seconds=sum(r['rust_geometry_CPU_seconds'] for r in records),
        python_geometry_CPU_seconds=sum(r['python_geometry_CPU_seconds'] for r in records),
        rust_tree_counts=dict(rust_counts),python_tree_counts=dict(python_counts),
        scope='Unfiltered saved-circle average under original azimuth laws; no R4 arc restriction or physical weight.')


def compare(cases_path,execution,out,expected_execution_sha256,prior_prelaunch_failure=None):
    cases_path,execution,out=map(lambda p:Path(p).resolve(),(cases_path,execution,out))
    require(not out.exists(),'Fresh comparison directory required')
    require(sha(cases_path)==CASES_SHA,'Frozen all-240 cases changed')
    require(sha(execution/'protocol.json')==expected_execution_sha256,'Frozen execution protocol changed')
    protocol=read(execution/'protocol.json');state=read(execution/'status.json')
    require(state['complete'] and state['protocol_sha256']==expected_execution_sha256,'Execution controller not complete')
    require(all(j['status']=='complete' and j['returncode']==0 for j in state['jobs']),
            'Geometry execution job did not complete')
    case_data=read(cases_path);cases=case_data['cases'];rust_root=execution/'rust'
    manifest=read(rust_root/'manifest.json');summary=read(rust_root/'summary.json');python=read(execution/'python.json')
    require(case_data['new_pose_draws']==case_data['new_Poisson_clouds']==0 and len(cases)==240,'Wrong input allocation')
    require(summary['complete'] and summary['manifest']==manifest and summary['cases']==240 and
            manifest['schema']=='contact-circle-feasibility-v1' and manifest['cases_sha256']==CASES_SHA and
            manifest['new_pose_draws']==manifest['new_Poisson_clouds']==0,'Incomplete or mismatched Rust output')
    require(python['complete'] and python['schema']=='independent-contact-circle-diagnostic-v1' and python['cases']==240,
            'Incomplete Python output')
    bindings={str(p):sha(p) for p in [cases_path,rust_root/'manifest.json',rust_root/'summary.json',execution/'python.json',
                                    execution/'protocol.json',execution/'status.json']}
    earlier=None
    if prior_prelaunch_failure is not None:
        prior=Path(prior_prelaunch_failure).resolve();failed=read(prior/'status.json')
        require(not failed['complete'] and all(j['status']=='not_started' for j in failed['jobs']) and
                not (prior/'rust').exists() and not (prior/'python.json').exists(),
                'Earlier event was not a prelaunch-only failure')
        for name in ('status.json','protocol.json'):bindings[str(prior/name)]=sha(prior/name)
        earlier=dict(directory=str(prior),error=failed.get('error'),actual_geometry_jobs=0,
            scope='Preserved scheduling failure before either geometry implementation started; no repeated evaluations')
    for name,digest in state['output_sha256'].items():
        require(sha(execution/name)==digest,'Completed controller output changed');bindings[str(execution/name)]=digest
    for name,digest in read(execution/'freeze.json')['files'].items():
        require(sha(execution/name)==digest,'Frozen execution source changed');bindings[str(execution/name)]=digest
    require(manifest['executable_sha256']==protocol['binary_sha256'] and
            manifest['source_bundle_sha256']==protocol['source_bundle_sha256'],'Executed Rust source/binary differs')
    for name in ('circles','attempts'):
        path=rust_root/(name+'.jsonl');require(sha(path)==summary[name+'_sha256'],'Rust row/journal hash changed');bindings[str(path)]=sha(path)
    for name,key in [('cases.json','cases_sha256'),('config.json','config_sha256'),('shape.json','shape_sha256'),
                     ('source-bundle.json','source_bundle_sha256')]:
        path=rust_root/'provenance'/name;require(sha(path)==manifest[key],'Rust source input changed');bindings[str(path)]=sha(path)
    require(manifest['config_sha256']==case_data['config_sha256'] and manifest['shape_sha256']==case_data['shape_sha256'],
            'Rust physical definition changed')
    for path,digest in {**case_data['sources'],**python['input_sha256']}.items():
        require(sha(path)==digest,'Independent source input changed: '+path);bindings[path]=digest
    require(CASES_SHA in python['input_sha256'].values(),'Python did not bind exact frozen cases')
    require(protocol['independent_python_sha256'] in python['input_sha256'].values(),'Executed Python reference source differs')
    rust=rows(rust_root/'circles.jsonl');journal=rows(rust_root/'attempts.jsonl');pyrows=python['rows'];ids=[r['id'] for r in cases]
    require(len(set(ids))==len(ids)==len(rust)==len(pyrows)==len(journal)==240,'Missing/duplicate cases')
    require([r['id'] for r in rust]==[r['id'] for r in pyrows]==[r['id'] for r in journal]==ids,
            'Reordered or filtered case identities')
    require(all(r['ordinal']==i and j['ordinal']==i and j['state']=='begin' for i,(r,j) in enumerate(zip(rust,journal))),
            'Incomplete ordered attempt journal')
    records=[compare_case(c,r,p) for c,r,p in zip(cases,rust,pyrows)]
    maxima={key:max(r['absolute_errors'].get(key,0.) for r in records)
            for key in sorted({key for r in records for key in r['absolute_errors']})}
    checks=Counter()
    for r in records:
        checks.update({name:int(not passed) for name,passed in r['checks'].items()})
    groups={}
    for arm in sorted({r['arm'] for r in records}):
        chosen=[r for r in records if r['arm']==arm]
        groups[arm]=dict(all=grouped_summary(chosen),
            by_width={str(w):grouped_summary([r for r in chosen if r['width_index']==w]) for w in sorted({r['width_index'] for r in chosen})},
            by_population={p:grouped_summary([r for r in chosen if r['population']==p]) for p in sorted({r['population'] for r in chosen})})
    result=dict(schema='independent-contact-circle-comparison-v1',complete=True,passed=all(r['passed'] for r in records),
        cases=240,cases_sha256=CASES_SHA,execution_protocol_sha256=expected_execution_sha256,
        absolute_tolerance=TOLERANCE,source_sha256=bindings,
        maximum_absolute_errors=maxima,failed_checks=dict(checks),failed_cases=[r['id'] for r in records if not r['passed']],
        witnesses_checked=sum(r['witnesses_checked'] for r in records),summary=grouped_summary(records),arms=groups,rows=records,
        rust_total_CPU_seconds=summary['cpu_seconds'],controller_wall_seconds=state['finished']-state['started'],
        child_CPU_seconds={j['id']:j['child_CPU_seconds'] for j in state['jobs']},prior_prelaunch_failure=earlier,
        new_pose_draws=0,new_Poisson_clouds=0,
        scope='Independent interval/mass/identity comparison on all saved circles; no new geometry execution, normalized sampler, R4/capture arc survival or physical inference.')
    out.mkdir(parents=True);write(out/'analysis.json',result);shutil.copy2(__file__,out/'source.py')
    write(out/'freeze.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k:result[k] for k in ('passed','cases','maximum_absolute_errors','failed_cases','witnesses_checked','summary')},indent=2))
    require(result['passed'],'Circle comparison failed; all discrepancies preserved')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','execution','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-execution-sha256',required=True)
    parser.add_argument('--prior-prelaunch-failure',type=Path)
    args=parser.parse_args();compare(args.cases,args.execution,args.out,args.expected_execution_sha256,args.prior_prelaunch_failure)
