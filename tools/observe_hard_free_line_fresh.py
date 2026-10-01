#!/usr/bin/env python3
"""Classify each valid newly proposed pose once, using the frozen observer."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'

PARTS=['outside_R4_or_capture','inside_domain_hard_invalid','valid_native_entry',
       'valid_contact_without_native_entry','valid_unbound_without_native_entry']
BRANCHES=['uniform','original_gaussian','conditioned_fallback','conditioned_success']
def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def branch(row):
    d=row['draw']
    if d['component'] is None:return 'uniform'
    if not d['conditional']:return 'original_gaussian'
    return 'conditioned_fallback' if d['fallback'] else 'conditioned_success'


def classify_rows(records,classifier,contact,stream,identity):
    counts=Counter();branches={name:Counter() for name in BRANCHES};calls=0;anomalies=[]
    for i,row in enumerate(records):
        require(type(row['id']) is int and row['id']==i and row['kind']=='fresh','Attempt identity changed')
        flags={k:row[k] for k in ('hard_valid','shell_valid','capture_valid')}
        require(all(type(v) is bool for v in flags.values()),'Invalid geometry flags')
        valid=all(flags.values());native=classifier.classify(row['pose']) if valid else None
        physical=contact.classify(row['pose']) if valid else None;b=branch(row)
        if not flags['shell_valid'] or not flags['capture_valid']:part=PARTS[0]
        elif not flags['hard_valid']:part=PARTS[1]
        elif native['native_any']:part=PARTS[2]
        elif physical['exclusion_contact']:part=PARTS[3]
        else:part=PARTS[4]
        counts[part]+=1;branches[b]['attempts']+=1;branches[b][part]+=1
        if valid:
            calls+=1
            require(native['native_any']==(native['native_anchor_count']>0),'Native count inconsistent')
            require(not native['registry_consistent_triangle'] or native['native_anchor_count']==2,'Triangle lacks both native anchors')
            events=dict(exclusion_contact=int(physical['exclusion_contact']),
                both_scaffold_contacts=int(all(p['exclusion_contact'] for p in physical['anchors'])),
                native_both_anchors=int(native['native_anchor_count']==2),
                cooperative_entry=int(native['cooperative_entry']),
                registry_triangle=int(native['registry_consistent_triangle']))
            counts.update(events);branches[b].update(events)
            if native['native_any'] and not physical['exclusion_contact']:anomalies.append(i)
        stream.write(json.dumps(dict(**identity,attempt_index=i,applicable=valid,physical_flags=flags,
            proposal_branch=b,disposition=part,classification=native,contact=physical),allow_nan=False)+'\n')
        stream.flush()
    require(sum(counts[p] for p in PARTS)==len(records),'Partition missing attempted rows')
    extra=['exclusion_contact','both_scaffold_contacts','native_both_anchors','cooperative_entry','registry_triangle']
    return dict(attempted=len(records),valid=calls,partition={k:counts[k] for k in PARTS},
        events={k:counts[k] for k in extra},
        branches={b:{k:c[k] for k in ['attempts',*PARTS,*extra]} for b,c in branches.items()},
        native_classifier_calls=calls,contact_classifier_calls=calls,
        native_without_exclusion_contact_attempts=anomalies,width_contacts_used=False)


def run(root,frozen,config_path,shape_path,out,arm,population,seed):
    root,frozen,config_path,shape_path,out=map(lambda p:Path(p).resolve(),(root,frozen,config_path,shape_path,out))
    require(not out.exists(),'No repeated native/contact classification')
    bindings={}
    def bind(path,digest=None):
        actual=sha(path);require(digest is None or actual==digest,'Frozen observer input changed')
        bindings[str(path)]=actual
    for name,digest in read(frozen/'freeze.json')['files'].items():bind(frozen/name,digest)
    d=read(frozen/'declaration.json');bind(frozen/'freeze.json');bind(config_path);bind(shape_path)
    manifest=read(root/'manifest.json');summary=read(root/'summary.json');audit=read(root/'independent-audit.json')
    require(summary['complete'] and audit['complete'] and manifest['samples']==summary['samples']==256 and
            manifest['seed']==seed and summary['probes']==0,'Fresh proposal and independent audit must finish')
    for name in ('manifest.json','summary.json','independent-audit.json'):bind(root/name)
    bind(root/'samples.jsonl',summary['samples_sha256'])
    require(audit['input_sha256'].get(str(root/'samples.jsonl'))==sha(root/'samples.jsonl'),'Audit does not bind these fresh poses')
    started=time.process_time();sys.path.insert(0,str(frozen/'source'))
    loader=importlib.import_module('analyze_mobile_native_pocket')
    module=importlib.import_module('analyze_mobile_threshold_reference')
    for name,digest in d['source_sha256'].items():
        require(sha(frozen/'source'/name)==digest,'Archived classifier helper changed')
        loaded=sys.modules.get(name[:-3])
        if loaded is not None:require(Path(loaded.__file__).resolve()==frozen/'source'/name,'Nonfrozen classifier import')
    definition=frozen/d['native_definition'];classifier,binding=loader.load_classifier(definition)
    require(binding['definition_sha256']==d['native_definition_sha256'],'Native definition changed')
    config=read(config_path);module.validate_classifier_target(config,sha(shape_path),classifier.definition,definition)
    contact=module.ExclusionContact(read(shape_path),config['fixed_poses'],config['depletant_radius'])
    startup=time.process_time()-started
    records=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    require(len(records)==256,'Fresh attempted rows missing')
    out.mkdir();identity=dict(arm=arm,population=population,seed=seed,input_samples_sha256=sha(root/'samples.jsonl'))
    started=time.process_time()
    with (out/'labels.jsonl').open('x') as stream:result=classify_rows(records,classifier,contact,stream,identity)
    result.update(identity,complete=True,classifier_CPU_seconds=time.process_time()-started,
        classifier_startup_CPU_seconds=startup,input_sha256=bindings,native_definition_sha256=binding['definition_sha256'],
        labels_sha256=sha(out/'labels.jsonl'),new_Poisson_clouds=0,scope='Unconditional fresh-proposal labels, not physical occupancies or weights.')
    require(all(sha(p)==digest for p,digest in bindings.items()),'Observer inputs changed during classification')
    write(out/'summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','frozen','config','shape','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--population',required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();print(json.dumps({k:v for k,v in run(a.root,a.frozen,a.config,a.shape,a.out,a.arm,a.population,a.seed).items()
        if k not in ('input_sha256','branches')},indent=2))
