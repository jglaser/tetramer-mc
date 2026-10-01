#!/usr/bin/env python3
"""Reuse audited SMC population summaries; never replay poses or classifiers."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
from compare_r4_smc_importance import (smc_rows,importance_rows,verify_log_sum,validate_strata,STRATA_DEFINITION)
from analyze_r4_smc_control import CLASSES,STRATA,validate_config_identity
from run_native_excluded_smc_campaign import mass_statistics,mass_comparison
from prepare_hard_free_line_fresh import read,sha,write,require,NATIVE_SHA,SHAPE_SHA,REGION_SHA
from prepare_shoulder_docking_benchmark import local_dependencies
from run_contact_tail_pilot import verify_frozen

R5_SHA='76ea65088e302d6b6478ac033af9b67b7cf21cb7cea1f854f70cdcd6db0473ae'
PARTITION_SHA='8620fc2932bd6583a7571e36b44c71eaf95bd8032720806be31cf5c5c10f49dd'
RESTRICTED_CLASSES=('contact_no_native_entry','unbound_no_native_entry')
CONTROLS=['unrestricted_broad','unrestricted_narrow','excluded_narrow','excluded_large','excluded_broad']


class Inputs:
    def __init__(self):self.files={}
    def bind(self,path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        require(expected is None or digest==expected,'Summary/provenance hash mismatch: '+str(path))
        self.files[str(path)]=digest;return path
    def json(self,path,expected=None):return read(self.bind(path,expected))
    def recheck(self):
        for path,digest in self.files.items():require(sha(path)==digest,'Bound input changed: '+path)


def target(campaign,ledger):
    c=Path(campaign)/'common';p=ledger.json(Path(campaign)/'protocol.json')
    region=ledger.json(c/'reference-package/region.json',REGION_SHA)
    config=ledger.json(c/'config.json')
    ledger.bind(c/'reference-package/shape.json',SHAPE_SHA)
    ledger.bind(c/'reference-package/old-r5-region.json',R5_SHA)
    ledger.bind(c/'reference-package/native-region/definition.json',NATIVE_SHA)
    ledger.bind(c/'reference-package/native-partition-definition.json',PARTITION_SHA)
    require(p['shape_sha256']==SHAPE_SHA and p['region_sha256']==REGION_SHA and p['supplemental_definition_sha256']==PARTITION_SHA,
        'Physical campaign target/partition mismatch')
    require(p['strata']==STRATA_DEFINITION,'Different radial/angular/orthant partition')
    for rk,ck in [('physical_fixed_neighbors','fixed_poses'),('capture_center','capture_center'),('capture_radius','capture_radius'),
        ('activity','reservoir_density'),('depletant_radius','depletant_radius'),('physical_metric','metadata')]:
        require(region[rk]==config[ck],'Config differs from exact region: '+rk)
    require(config['reservoir_density']==.035 and config['depletant_radius']==1.5,'Original bath changed')
    return dict(shape_sha256=SHAPE_SHA,region_sha256=REGION_SHA,reference_region_sha256=R5_SHA,
        native_definition_sha256=NATIVE_SHA,supplemental_definition_sha256=PARTITION_SHA,
        measure='Cartesian translation in Angstrom cubed times normalized proper SO(3) Haar',
        config=config,protocol_sha256=sha(Path(campaign)/'protocol.json'),seeds=[j['seed'] for j in p['jobs']],strata=p['strata'])


def validate_unrestricted(smc,t):
    require(smc['schema']=='smc-r4-control-analysis-v1' and smc['complete'],'Incomplete unrestricted SMC')
    p=smc['protocol_snapshot'];physical=p['physical_target']
    require(physical['shape_sha256']==t['shape_sha256'] and physical['region_sha256']==t['region_sha256']
        and smc['native_definition_sha256']==t['native_definition_sha256']
        and p['proposal']['reference_region_sha256']==t['reference_region_sha256'],'Unrestricted SMC region/model/native mismatch')
    require(physical['measure']=='d3t in Angstrom cubed times normalized proper SO(3) Haar','Unrecognized unrestricted measure')
    source=p['source_and_input_sha256']
    require(any(Path(path).name=='native-partition-definition.json' and digest==t['supplemental_definition_sha256']
        for path,digest in source.items()),'Different/unbound R5 native complement definition')
    allowed=set(p.get('proposal_control_difference',{}).get('changed_config_fields',{}))
    require(allowed<=set(('translation_steps','rotation_steps_deg')),'Physical differences cannot be excluded')
    validate_config_identity(smc['physical_config'],t['config'],allowed)
    require(set(t['seeds']).isdisjoint(pop['seed'] for pop in smc['populations']),'SMC/importance streams overlap')
    return sorted(allowed)


def restricted_rows(populations,kind,family=None,index=None):
    require(len(populations)==4 and len({p['id'] for p in populations})==4 and len({p['seed'] for p in populations})==4,
        'All four independent restricted populations required')
    rows=[]
    for p in sorted(populations,key=lambda p:p['id']):
        terminal=p['terminal'];counts=terminal['counts'];n=terminal['particles'];total=terminal['log_masses']['total']
        require(counts['total']==n and n==sum(counts[k] for k in RESTRICTED_CLASSES),'Restricted partition incomplete')
        for key in ('total',*RESTRICTED_CLASSES):
            actual=terminal['log_masses'][key]
            expected=None if n==0 or counts[key]==0 or total is None else total+math.log(counts[key]/n)
            require(actual is None if expected is None else actual is not None and abs(actual-expected)<1e-9,
                'Restricted mass is not Zhat times terminal indicator')
            for fam,size in STRATA.items():
                require(len(terminal['log_strata'][fam][key])==size,'Missing restricted stratum')
                verify_log_sum(actual,terminal['log_strata'][fam][key],'Restricted strata do not sum')
        if kind=='Q0':
            require(family is None,'Initial physical H/g strata were not archived')
            value=p['initial_physical_hard_log_masses']
        elif family is None:value=terminal['log_masses']
        else:value={key:terminal['log_strata'][family][key][index] for key in ('total',*RESTRICTED_CLASSES)}
        rows.append({key:value[key] for key in RESTRICTED_CLASSES})
    return rows


def summarize(rows,classes):return {k:mass_statistics([r[k] for r in rows]) for k in classes}


def freeze(out,campaign,repository):
    out,campaign,repository=map(lambda p:Path(p).resolve(),(out,campaign,repository));require(not out.exists(),'Fresh bridge preparation required')
    ledger=Inputs();t=target(campaign,ledger);runs=repository/'runs'
    authentication=ledger.json(runs/'smc-completed-evidence-review-20260923/authentication.json')
    require(authentication['complete'] and authentication['schema']=='completed-smc-bridge-evidence-authentication-v1','Missing completed SMC authentication')
    controls={};historical={}
    for name in ('broad','narrow'):
        path=(runs/f'smc-r4-controls-workflow-20260922/{name}-analysis/analysis.json').resolve()
        smc=ledger.json(path,authentication['input_sha256'][str(path)])
        excluded=validate_unrestricted(smc,t)
        # Reuse the previously authenticated derived population records. No histories or labels are reread.
        controls['unrestricted_'+name]=dict(kind='unrestricted',analysis=smc,proposal_only_exclusions=excluded,
            populations=smc['populations'],classes=list(CLASSES))
        for kind in ('Qz','Q0'):smc_rows(smc,kind)
    old_comparison=(runs/'smc-r4-controls-workflow-20260922/comparison/comparison.json').resolve()
    historical['unrestricted_control_comparison']=ledger.json(old_comparison,authentication['input_sha256'][str(old_comparison)])
    restricted=runs/'native-excluded-smc-control-20260924'
    status=ledger.json(restricted/'status.json');require(status['complete'] and status['phase']=='complete','Restricted SMC incomplete')
    plan=ledger.json(restricted/'plan.json',status['plan_sha256']);analysis=ledger.json(restricted/'analysis.json',status['analysis_sha256'])
    physical=plan['physical_target']
    require(physical['shape_sha256']==SHAPE_SHA and physical['region_sha256']==REGION_SHA and physical['definition_sha256']==NATIVE_SHA
        and physical['target']=='Hcapture Hhard IR4 (1-Inative) exp(zC) d3t dHaar'
        and physical['measure']=='Cartesian Angstrom cubed times normalized proper SO(3) Haar','Restricted physical target differs')
    ledger.bind(restricted/'inputs/region.json',REGION_SHA);ledger.bind(restricted/'inputs/reference-region.json',R5_SHA)
    ledger.bind(restricted/'inputs/shape.json',SHAPE_SHA)
    historical['restricted_control_checks']={k:analysis[k] for k in ['quality','comparisons','strata','physical_conclusion']}
    for name in ('narrow','large','broad'):
        config=ledger.json(restricted/'inputs'/f'{name}-config.json')
        validate_config_identity(config,t['config'],('translation_steps','rotation_steps_deg'))
        expected={j['id']:j for j in plan['jobs'] if j['arm']==name};populations=[]
        for jobid,j in expected.items():
            receipt=analysis['audits'][jobid];p=ledger.json(receipt['path'],receipt['sha256'])
            require(p['complete'] and p['schema']=='native-excluded-smc-population-audit-v1' and p['id']==jobid and p['seed']==j['seed']
                and p['initial_draws']==plan['allocation']['initial_draws_per_population'],'Restricted audit allocation/identity mismatch')
            require(p['native_definition']==dict(definition_sha256=NATIVE_SHA,compiled_sha256=physical['compiled_sha256'],
                shape_sha256=SHAPE_SHA,fixed_poses=t['config']['fixed_poses']),'Restricted native predicate mismatch')
            require(p['terminal']['particles']==(0 if p['zero_estimate'] else plan['allocation']['arms'][name]['population']),
                'Restricted terminal allocation mismatch')
            require(p['seed'] not in t['seeds'],'Restricted/importance seed overlap');populations.append(p)
        for kind in ('Qz','Q0'):restricted_rows(populations,kind)
        controls['excluded_'+name]=dict(kind='restricted',populations=populations,classes=list(RESTRICTED_CLASSES),
            proposal_only_exclusions=['translation_steps','rotation_steps_deg'])
    require(list(controls)==CONTROLS,'Historical controls omitted or reordered')
    allseeds=[p['seed'] for c in controls.values() for p in c['populations']]
    require(len(allseeds)==len(set(allseeds))==20,'Historical controls reuse population streams')
    ledger.recheck();out.mkdir(parents=True);(out/'source').mkdir()
    sources=local_dependencies([Path(__file__)])
    for name,path in sources.items():shutil.copy2(path,out/'source'/name)
    write(out/'historical.json',dict(target=t,controls=controls,historical_checks=historical,input_sha256=ledger.files,
        authentication_reused=True,raw_history_rehashed=False,raw_history_replayed=False,old_classifier_calls=0))
    write(out/'plan.json',dict(schema='hard-free-line-smc-bridge-plan-v1',campaign=str(campaign),controls=CONTROLS,
        fresh_arms=['baseline','conditioned'],comparison_kinds=['Qz','Q0'],strata=STRATA_DEFINITION,
        absolute_log_tolerance=.2,combined_linear_SE_multiplier=3.,material_stratum_fraction=.01,
        require_complete_current_comparison=True,missing_Q0_strata='Initial unresampled H/g stratum masses were not archived; no substitute estimator.',
        physical_target=t,historical_sha256=sha(out/'historical.json'),sources={n:sha(p) for n,p in sources.items()},
        new_physical_draws=0,new_classifier_calls=0,audits_replayed=0,physical_gate_open=False,
        scope='Summary-only independent linear population comparison. All five historical controls and zeros retained; no selected-control pooling or finite-system conclusion.'))
    write(out/'freeze.json',dict(files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    return sha(out/'plan.json')


def compare_control(control,importance):
    classes=control['classes']
    def source(kind,family=None,index=None):
        return smc_rows(control['analysis'],kind,family,index) if control['kind']=='unrestricted' else restricted_rows(control['populations'],kind,family,index)
    smc={kind:summarize(source(kind),classes) for kind in ('Qz','Q0')}
    new={kind:summarize(importance_rows(importance,kind),classes) for kind in ('Qz','Q0')}
    result=dict(SMC=smc,fresh=new,masses={kind:{k:mass_comparison(smc[kind][k],new[kind][k]) for k in classes} for kind in ('Qz','Q0')},strata={})
    for family,size in STRATA.items():
        entries=[]
        for i in range(size):
            a=summarize(source('Qz',family,i),classes);b=summarize(importance_rows(importance,'Qz',family,i),classes)
            for name in classes:
                fractions=[None if whole['log_Q'] is None else 0. if part['log_Q'] is None else math.exp(part['log_Q']-whole['log_Q'])
                    for part,whole in [(a[name],smc['Qz'][name]),(b[name],new['Qz'][name])]]
                entries.append(dict(bin=i,region=name,SMC=a[name],fresh=b[name],observed_class_fractions=fractions,
                    material=max(v or 0 for v in fractions)>=.01,comparison=mass_comparison(a[name],b[name])))
        result['strata'][family]=entries
    material=[r for rows in result['strata'].values() for r in rows if r['material']]
    result['material_strata']=dict(total=len(material),failed=sum(not r['comparison']['passed'] for r in material))
    result['missing_Q0_strata']='Not archived; terminal strata cannot replace unresampled initial H/g.'
    return result


def run(preparation,out):
    preparation,out=Path(preparation).resolve(),Path(out).resolve();require(not out.exists(),'No repeated bridge output')
    verify_frozen(preparation);plan=read(preparation/'plan.json');historical=read(preparation/'historical.json')
    require(sha(preparation/'historical.json')==plan['historical_sha256'] and plan['controls']==CONTROLS,'Historical selection changed')
    sources=local_dependencies([Path(__file__)])
    require({n:sha(p) for n,p in sources.items()}==plan['sources'],'Use frozen bridge source closure')
    ledger=Inputs();campaign=Path(plan['campaign']);t=target(campaign,ledger)
    require(t==plan['physical_target']==historical['target'],'Fresh physical target changed')
    status=ledger.json(campaign/'status.json');require(status['complete'] and status['phase']=='complete','Physical campaign/audit/classifier not complete')
    importance=ledger.json(campaign/'comparison/analysis.json',status['comparison_sha256'])
    require(importance['schema']=='hard-free-line-physical-comparison-v1' and importance['complete']
        and importance['protocol_sha256']==t['protocol_sha256'] and importance['native_definition']['definition_sha256']==NATIVE_SHA,
        'Fresh completed summary/native binding differs')
    require(set(importance['arms'])==set(plan['fresh_arms']),'Fresh arm missing or added')
    for path,digest in historical['input_sha256'].items():ledger.bind(path,digest)
    for control in historical['controls'].values():
        if control['kind']=='unrestricted':
            validate_strata(control['analysis'],dict(strata_definition=plan['strata'],arms=importance['arms']))
    comparisons={arm:{name:compare_control(historical['controls'][name],importance['arms'][arm]) for name in CONTROLS}
        for arm in plan['fresh_arms']}
    ledger.recheck();out.mkdir(parents=True)
    result=dict(schema='hard-free-line-matching-smc-comparison-v1',complete=True,comparisons=comparisons,
        historical_checks=historical['historical_checks'],fresh_regional_diagnostics=importance['diagnostics'],
        preparation=str(preparation),plan_sha256=sha(preparation/'plan.json'),input_sha256=ledger.files,
        source_sha256=plan['sources'],populations_pooled=False,new_physical_draws=0,new_classifier_calls=0,audits_replayed=0,
        raw_history_replayed=False,physical_gate_open=False,finite_system_conclusion='unresolved',
        scope='Arithmetic means of four independent linear masses, zeros included. SMC Qz is Zhat times terminal indicator; Q0 is unresampled initial H/g. Three combined linear-SE and 0.2-log criteria are separate. Historical control failures remain; agreement does not bound unseen modes, the vessel remainder or finite-system stability.')
    write(out/'analysis.json',result);write(out/'freeze.json',dict(files={'analysis.json':sha(out/'analysis.json')}));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    f=s.add_parser('freeze')
    for name in ('out','campaign','repository'):f.add_argument('--'+name,type=Path,required=True)
    r=s.add_parser('run')
    for name in ('preparation','out'):r.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.action=='freeze':print(freeze(a.out,a.campaign,a.repository))
    else:print(run(a.preparation,a.out)['complete'])
