#!/usr/bin/env python3
"""Frozen, one-pass native/contact labels for completed passive proposal rows.

No poses, depletants or physical weights are generated. Archived physical rows
and the 618 passive probe queries are not reclassified.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import scipy

SCHEMA='contact-distance-passive-native-observer-v1'
DISPOSITIONS=['outside_R4_or_capture','inside_domain_hard_invalid','valid_native_entry',
              'valid_contact_without_native_entry','valid_unbound_without_native_entry']
BRANCHES=['uniform','original_gaussian','conditioned_fallback','conditioned_success']


def require(condition,message):
    if not condition:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def runtime():return dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__)


def verified_files(directory):
    directory=Path(directory);files=read(directory/'freeze.json')['files']
    for name,digest in files.items():
        p=(directory/name).resolve()
        require(p.is_relative_to(directory.resolve()) and sha(p)==digest,'Frozen file changed: '+name)
    return files


def freeze(out,declaration,passive):
    out,declaration,passive=map(lambda x:Path(x).resolve(),(out,declaration,passive))
    require(not out.exists(),'Fresh observer output required')
    files=verified_files(declaration);verified_files(passive)
    d=read(declaration/'declaration.json');p=read(passive/'protocol.json')
    require(d['fresh_attempts_to_account']==p['fresh_draws']==1536 and p['new_Poisson_clouds']==0,
            'Observer allocation differs')
    require(p['preparation_sha256']==d['passive_preparation_plan_sha256'],'Declared passive preparation differs')
    require([{k:j[k] for k in ('arm','id','seed','fresh_proposal_draws')} for j in p['jobs']]==d['population_identities'],
            'Declared population identities differ')
    out.mkdir();common=out/'common';common.mkdir()
    for name in [*files,'freeze.json']:
        target=common/'declaration'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(declaration/name,target)
    for name in ('config.json','region.json','shape.json'):
        shutil.copy2(passive/'common'/name,common/name)
    shutil.copy2(__file__,common/'analyze_contact_distance_native.py')
    tests=Path(__file__).with_name('test_contact_distance_native.py')
    require(tests.is_file(),'Observer bookkeeping test source required')
    shutil.copy2(tests,common/tests.name)
    protocol=dict(schema=SCHEMA,passive=str(passive),passive_protocol_sha256=sha(passive/'protocol.json'),
        passive_freeze_sha256=sha(passive/'freeze.json'),declaration=str(declaration),
        declaration_sha256=sha(declaration/'declaration.json'),declaration_freeze_sha256=sha(declaration/'freeze.json'),
        adapter_sha256=sha(__file__),bookkeeping_test_source_sha256=sha(tests),runtime=runtime(),fresh_attempts=1536,
        populations=d['population_identities'],maximum_CPU_workers=1,physical_jobs=0,new_Poisson_clouds=0,
        scope='Unconditional native/contact geometry of new passive proposals only; no physical weights or equilibrium occupancy.')
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    validate(out,sha(out/'protocol.json'))
    return sha(out/'protocol.json')


def validate(out,expected):
    out=Path(out).resolve();verified_files(out);p=read(out/'protocol.json')
    require(sha(out/'protocol.json')==expected and p['schema']==SCHEMA,'Observer protocol differs')
    require(p['adapter_sha256']==sha(__file__) and p['runtime']==runtime(),'Frozen adapter/runtime differs')
    declaration=Path(p['declaration']);verified_files(declaration)
    require(sha(declaration/'declaration.json')==p['declaration_sha256'] and
            sha(declaration/'freeze.json')==p['declaration_freeze_sha256'],'Original declaration differs')
    passive=Path(p['passive']);verified_files(passive)
    require(sha(passive/'protocol.json')==p['passive_protocol_sha256'] and
            sha(passive/'freeze.json')==p['passive_freeze_sha256'],'Passive source protocol differs')
    return p


def proposal_branch(row):
    draw=row['draw']
    if draw['component'] is None:return 'uniform'
    if not draw['conditional']:return 'original_gaussian'
    return 'conditioned_fallback' if draw['fallback'] else 'conditioned_success'


def classify_rows(rows,classifier,contact,stream,identity):
    counts=Counter();branches={b:Counter() for b in BRANCHES};cross=[Counter() for _ in range(3)]
    anomalies=[];calls=0;native=cooperative=triangles=0
    for index,row in enumerate(rows):
        require(type(row['id']) is int and row['id']==index and row['kind']=='fresh','Attempt identities changed')
        flags={key:row[key] for key in ('hard_valid','shell_valid','capture_valid')}
        require(all(type(v) is bool for v in flags.values()),'Invalid physical flags')
        in_domain=flags['shell_valid'] and flags['capture_valid'];valid=in_domain and flags['hard_valid']
        label=classifier.classify(row['pose']) if valid else None
        measured=contact.classify(row['pose']) if valid else None
        branch=proposal_branch(row)
        if not in_domain:disposition=DISPOSITIONS[0]
        elif not flags['hard_valid']:disposition=DISPOSITIONS[1]
        elif label['native_any']:disposition=DISPOSITIONS[2]
        elif measured['exclusion_contact']:disposition=DISPOSITIONS[3]
        else:disposition=DISPOSITIONS[4]
        counts[disposition]+=1;branches[branch]['attempts']+=1;branches[branch][disposition]+=1
        if valid:
            calls+=1
            require(label['native_any']==(label['native_anchor_count']>0),'Native anchor count differs')
            require(not label['registry_consistent_triangle'] or label['native_anchor_count']==2,
                    'Registry triangle lacks both anchors')
            native+=int(label['native_any']);cooperative+=int(label['cooperative_entry'])
            triangles+=int(label['registry_consistent_triangle'])
            branches[branch]['native_both_anchors']+=int(label['cooperative_entry'])
            branches[branch]['registry_triangle']+=int(label['registry_consistent_triangle'])
            for i,pair in enumerate(row['width_contacts']):
                if all(pair):cross[i][disposition]+=1
            if label['native_any'] and not measured['exclusion_contact']:anomalies.append(index)
        record=dict(**identity,attempt_index=index,applicable=valid,proposal_branch=branch,
            disposition=disposition,physical_flags=flags,classification=label,contact=measured,
            width_contacts=row['width_contacts'])
        stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n');stream.flush()
    require(sum(counts.values())==len(rows),'Attempt partition has a gap')
    return dict(attempted=len(rows),valid=calls,native_entry=native,native_both_anchors=cooperative,
        registry_triangle=triangles,partition={k:counts[k] for k in DISPOSITIONS},
        branches={b:{k:c[k] for k in ['attempts',*DISPOSITIONS,'native_both_anchors','registry_triangle']} for b,c in branches.items()},
        joint_contact_cross_counts=[{k:c[k] for k in DISPOSITIONS[2:]} for c in cross],
        native_without_exclusion_contact_attempts=anomalies,native_classifier_calls=calls,contact_classifier_calls=calls)


def run(out,expected):
    out=Path(out).resolve();protocol=validate(out,expected);passive=Path(protocol['passive'])
    require(not (out/'launch-claim.json').exists(),'No observer retry or duplicate classification')
    state=read(passive/'status.json');source_protocol=read(passive/'protocol.json')
    require(state['complete'] and state['phase']=='complete' and state['protocol_sha256']==protocol['passive_protocol_sha256'],
            'Passive sampling and audits must be complete')
    require(sha(passive/'analysis.json')==state['analysis_sha256'] and read(passive/'analysis.json')['complete'],
            'Passive completed analysis changed')
    jobs=state['jobs']
    require(len(jobs)==12 and all(j['status']=='complete' for j in [*jobs,*state['coverage_jobs']]),
            'Not all passive jobs complete')
    input_hashes={str(passive/name):sha(passive/name) for name in ('protocol.json','status.json','analysis.json')}
    for job in [*jobs,*state['coverage_jobs']]:
        for name,digest in job['output_sha256'].items():
            path=Path(job['directory'])/name
            require(sha(path)==digest,'Completed passive row/audit changed')
            input_hashes[str(path)]=digest
        require(read(Path(job['directory'])/'independent-audit.json')['complete'],'Independent passive audit failed')
    write(out/'launch-claim.json',dict(pid=os.getpid(),started=time.time(),protocol_sha256=expected))
    start=time.process_time();completed=[]
    try:
        common=out/'common';frozen=common/'declaration';d=read(frozen/'declaration.json')
        sys.path.insert(0,str(frozen/'source'))
        loader=importlib.import_module('analyze_mobile_native_pocket')
        contact_module=importlib.import_module('analyze_mobile_threshold_reference')
        for name,digest in d['source_sha256'].items():
            require(sha(frozen/'source'/name)==digest,'Frozen classifier helper changed')
            module=sys.modules.get(name[:-3])
            if module is not None:require(Path(module.__file__).resolve()==(frozen/'source'/name).resolve(),
                                         'Unfrozen classifier helper already imported')
        definition=frozen/d['native_definition']
        classifier,binding=loader.load_classifier(definition)
        require(binding['definition_sha256']==d['native_definition_sha256'],'Full native definition changed')
        config=read(common/'config.json');shape=read(common/'shape.json')
        contact_module.validate_classifier_target(config,sha(common/'shape.json'),classifier.definition,definition)
        contact=contact_module.ExclusionContact(shape,config['fixed_poses'],config['depletant_radius'])
        setup_cpu=time.process_time()-start
        for job,identity in zip(jobs,protocol['populations']):
            require(all(job[k]==identity[k] for k in ('arm','id','seed','fresh_proposal_draws')),'Fresh population mismatch')
            directory=Path(job['directory']);path=directory/'samples.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            require(len(rows)==identity['fresh_proposal_draws']==128,'Missing passive attempted rows')
            target=out/job['arm']/job['id'];target.mkdir(parents=True,exist_ok=False)
            tick=time.process_time()
            with (target/'labels.jsonl').open('x') as stream:
                result=classify_rows(rows,classifier,contact,stream,dict(arm=job['arm'],population=job['id'],seed=job['seed'],
                    input_samples_sha256=job['output_sha256']['samples.jsonl']))
            cpu=time.process_time()-tick
            require(sha(path)==job['output_sha256']['samples.jsonl'],'Passive rows changed during classification')
            result.update(arm=job['arm'],population=job['id'],seed=job['seed'],classifier_CPU_seconds=cpu,
                passive_fresh_CPU_seconds=read(directory/'summary.json')['fresh_total_cpu_seconds'],
                labels_sha256=sha(target/'labels.jsonl'),input_samples_sha256=sha(path))
            write(target/'summary.json',result);completed.append(result)
        require(sum(r['attempted'] for r in completed)==1536,'Incomplete all-attempt classification')
        arms={}
        for arm in [a['name'] for a in source_protocol['arms']]:
            rows=[r for r in completed if r['arm']==arm];require(len(rows)==4,'Missing independent populations')
            fractions={}
            for category in DISPOSITIONS:
                values=np.array([r['partition'][category]/r['attempted'] for r in rows])
                fractions[category]=dict(count=sum(r['partition'][category] for r in rows),
                    mean=float(values.mean()),population_SE=float(values.std(ddof=1)/2))
            proposal_cpu=sum(r['passive_fresh_CPU_seconds'] for r in rows);classifier_cpu=sum(r['classifier_CPU_seconds'] for r in rows)
            arms[arm]=dict(populations=rows,attempted=512,fractions=fractions,
                passive_fresh_CPU_seconds=proposal_cpu,classifier_CPU_seconds=classifier_cpu,
                native_events_per_passive_CPU=sum(r['native_entry'] for r in rows)/proposal_cpu,
                native_events_per_combined_diagnostic_CPU=sum(r['native_entry'] for r in rows)/(proposal_cpu+classifier_cpu),
                scope='Geometric proposal events, not physical samples or equilibrium occupancies')
        require(all(sha(Path(p))==digest for p,digest in input_hashes.items()),'Passive inputs changed before aggregation')
        result=dict(schema=SCHEMA,complete=True,protocol_sha256=expected,arms=arms,total_attempted=1536,
            input_sha256=input_hashes,native_definition_sha256=binding['definition_sha256'],
            startup_CPU_seconds=setup_cpu,total_observer_CPU_seconds=time.process_time()-start,
            archived_rows_reclassified=0,new_Poisson_clouds=0,physical_mass_estimates_generated=0,
            scope=protocol['scope'])
        write(out/'analysis.json',result)
        write(out/'status.json',dict(complete=True,phase='complete',analysis_sha256=sha(out/'analysis.json'),
            protocol_sha256=expected,finished=time.time()))
        return result
    except BaseException as error:
        write(out/'status.json',dict(complete=False,phase='failed',error=repr(error),completed_populations=completed,
            protocol_sha256=expected,finished=time.time()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze')
    for name in ('out','declaration','passive'):f.add_argument('--'+name,type=Path,required=True)
    for action in ('preflight','run'):
        p=sub.add_parser(action);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-protocol-sha256',required=True)
    a=parser.parse_args()
    if a.action=='freeze':print(freeze(a.out,a.declaration,a.passive))
    elif a.action=='preflight':validate(a.out,a.expected_protocol_sha256);print('observer preflight passed; no classification')
    else:print(run(a.out,a.expected_protocol_sha256)['complete'])


if __name__=='__main__':main()
