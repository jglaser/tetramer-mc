"""Predeclared local/redraw/transport comparison under one physical target.

Only the explicitly declared global branch can differ. This is a measurement
contract, not a trajectory generator or evidence that the chain equilibrated.
"""
from __future__ import annotations
import hashlib
import json
import math

SCHEMA='matched-contact-kernel-benchmark-v1'
COMMON=('local_translation_std_A','local_small_angle_std_degrees','gca_probability',
        'center_shift_probability','poisson_lambda_ratio','endpoint_gate')
PROPOSAL=('method','model_sha256','frozen_posterior','learned_uniform_weight')
ROLES=('local','independent-redraw','correlated-transport')
ATTEMPTS='one single-body update per mobile body per sweep'


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def validate_contract(contract):
    require(set(contract)=={'schema','physical_identity_sha256','observer_definition_sha256','region_definition_sha256',
        'common_schedule','window','preparations','streams_per_preparation','arms','single_body_attempt_budget'},'Unknown/incomplete benchmark contract')
    require(contract['schema']==SCHEMA and contract['single_body_attempt_budget']==ATTEMPTS,'Benchmark schema or attempt budget differs')
    for key in ('physical_identity_sha256','observer_definition_sha256','region_definition_sha256'):
        require(isinstance(contract[key],str) and len(contract[key])==64 and all(c in '0123456789abcdef' for c in contract[key]),'Invalid benchmark identity: '+key)
    common=contract['common_schedule']; require(set(common)==set(COMMON),'Common proposal/noise/cost controls must be exhaustive')
    require(all(finite(common[k]) and common[k]>0 for k in ('local_translation_std_A','local_small_angle_std_degrees','poisson_lambda_ratio')),'Invalid common local/noise scale')
    require(all(finite(common[k]) and 0<=common[k]<=1 for k in ('gca_probability','center_shift_probability')),'Invalid collective probability')
    gate=common['endpoint_gate'];require(set(gate)=={'max_cells','max_depth','min_width'} and
        type(gate['max_cells']) is int and gate['max_cells']>0 and type(gate['max_depth']) is int and gate['max_depth']>=0 and finite(gate['min_width']) and gate['min_width']>=0,'Invalid geometric work budget')
    window=contract['window'];require(set(window)=={'burn_sweep','end_sweep','cadence_sweeps'} and
        all(type(v) is int for v in window.values()) and window['end_sweep']>window['burn_sweep']>=0 and window['cadence_sweeps']>0,'Invalid matched observation window')
    starts=contract['preparations'];require(isinstance(starts,list) and starts and len(set(starts))==len(starts) and all(isinstance(s,str) and s for s in starts),'Unique initial-preparation labels required')
    require(contract['streams_per_preparation']==4 and type(contract['streams_per_preparation']) is int,'Production contract requires four independent streams per preparation and arm')
    arms=contract['arms'];require(isinstance(arms,dict) and len(arms)==3 and all(isinstance(k,str) and k for k in arms),'Three named proposal arms required')
    by_role={}
    for name,arm in arms.items():
        require(set(arm)=={'role','global_probability',*PROPOSAL},'Incomplete or undeclared arm settings: '+name)
        role=arm['role'];require(role in ROLES and role not in by_role,'Exactly one local/redraw/transport role required');by_role[role]=arm
        g=arm['global_probability'];require(finite(g) and 0<=g<=1,'Invalid declared global branch probability')
        if role=='local':
            require(g==0 and arm['method']=='local-uniform' and arm['model_sha256'] is None and arm['frozen_posterior'] is None,'Local arm must have no global/model/posterior moves')
        else:
            require(0<g<1 and arm['method']=='learned','Learned comparison retains both local and global updates')
            h=arm['model_sha256'];require(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),'Frozen model hash required')
        u=arm['learned_uniform_weight'];require(finite(u) and 0<u<1,'Retain positive capture/uniform proposal support')
        posterior=arm['frozen_posterior']
        if role=='correlated-transport':
            require(isinstance(posterior,dict) and set(posterior)=={'probability','correlation'} and
                finite(posterior['probability']) and 0<posterior['probability']<1 and finite(posterior['correlation']) and 0<abs(posterior['correlation'])<=1,
                'Transport requires nonzero correlation and retained independent capture branch')
        else:require(posterior is None,'Independent redraw is the capture-mixture law, not a zero-correlation posterior-component update')
    redraw,transport=by_role['independent-redraw'],by_role['correlated-transport']
    require(all(redraw[k]==transport[k] for k in ('global_probability','model_sha256','learned_uniform_weight')),'Redraw and transport must share global frequency, frozen model and defensive floor')
    return contract


def validate_report(report,contract):
    validate_contract(contract)
    require(digest(report['physical_identity'])==contract['physical_identity_sha256'] and
        report['observer_definition_sha256']==contract['observer_definition_sha256'] and
        report['region_definition_sha256']==contract['region_definition_sha256'],'Benchmark physical target or measured regions differ')
    require(set(report['schedule'])==set(COMMON)|{'global_probability'} and
        {k:report['schedule'][k] for k in COMMON}==contract['common_schedule'],'Benchmark common local/collective/noise/cost settings differ')
    initialization=report['initialization'];name=initialization['proposal_arm']
    require(name in contract['arms'] and initialization['preparation_id'] in contract['preparations'],'Undeclared benchmark arm or preparation')
    arm=contract['arms'][name]
    require(report['schedule']['global_probability']==arm['global_probability'] and
        report['proposal']=={k:arm[k] for k in PROPOSAL},'Actual global kernel differs from its predeclared arm')
    require({k:report['window'][k] for k in ('burn_sweep','end_sweep','cadence_sweeps')}==contract['window'],'Benchmark observation window differs')
    require(initialization.get('invocation_initial_sweep',0)==0 and initialization.get('resume') is None,'Independent production arms cannot substitute resumed segments')
    counts=report['window_attempts']; total=report['physical_identity']['bodies']*(report['window']['end_sweep']-report['window']['burn_sweep'])
    require(set(counts)=={'local','global'} and all(type(v) is int and v>=0 for v in counts.values()) and sum(counts.values())==total,'Matched single-body attempt budget differs')
    if arm['role']=='local':require(counts['global']==0,'Pure-local control contains global attempts')


def validate_comparison(reports):
    bindings=[r.get('benchmark_contract') for r in reports]
    if not any(b is not None for b in bindings):return None
    require(all(isinstance(b,dict) for b in bindings),'Contracted and undeclared benchmarks cannot be mixed')
    first=bindings[0];contract=first['content'];validate_contract(contract)
    for report,binding in zip(reports,bindings):
        require(binding['sha256']==first['sha256'] and binding['content']==contract and
            binding['content_sha256']==digest(contract),'Benchmark contract changed across streams')
        require(report['dependency_sha256'].get(binding['path'])==binding['sha256'],'Report does not bind its benchmark contract')
        validate_report(report,contract)
    expected={(arm,start):[] for arm in contract['arms'] for start in contract['preparations']}
    for r in reports:expected[(r['initialization']['proposal_arm'],r['initialization']['preparation_id'])].append(r['initialization']['master_seed'])
    require(all(len(seeds)==contract['streams_per_preparation'] for seeds in expected.values()),'Incomplete or enlarged frozen arm/preparation allocation')
    seeds=[s for group in expected.values() for s in group]
    require(len(set(seeds))==len(seeds),'Independent benchmark streams reuse seeds')
    return dict(contract_sha256=first['sha256'],content_sha256=digest(contract),
        population_matrix={arm:{start:expected[(arm,start)] for start in contract['preparations']} for arm in contract['arms']},
        common_schedule=contract['common_schedule'],single_body_attempt_budget=ATTEMPTS,
        scope='All planned independent populations retained. Pure-local single-body arm and declared global variants share local sizes, collective opportunities, noise/cost controls and CPU accounting. This verifies design, not equilibrium or speedup.')
