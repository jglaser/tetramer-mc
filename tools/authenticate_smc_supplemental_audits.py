#!/usr/bin/env python3
"""Authenticate supplemental numerical receipts without rewriting audit history.

validate_plan is preparation-safe: it binds immutable plans, metadata inputs,
original failure records and frozen source bytes, but neither reads statistical
estimates nor binds the supplemental controller's mutable status. Large raw
physical-file hashes are declared in that immutable plan and checked at runtime.

authenticate requires terminal supplemental success. It reuses the two original
successful audit receipts through the existing physical checker, authenticates
new numerical receipts for exactly the two previously failed populations, and
returns per-population input bindings. It never launches sampling, reruns an
audit, edits a status, or changes classifiers, physical estimators or scientific
convergence criteria. The original campaign remains independent_audit_failed.
"""
from __future__ import annotations
import ast
import math
from pathlib import Path
import re
from analyze_r4_smc_control import read, require, sha
from analyze_mobile_native_pocket import local_sources
import run_hard_free_protein_smc as control

IDS = ('r00','r01','r02','r03')
REUSED = ('r00','r01')
SUPPLEMENTED = ('r02','r03')
PLAN_SCHEMA = 'protein-smc-supplemental-numerical-audits-v1'
AUDIT_SCHEMA = 'hard-free-initial-guide-smc-independent-audit-v1'
CONDITIONING_SCHEMA = 'split-conditioned-density-numerical-audit-v1'
NUMERICAL_CONTRACT = ('Strict same-input arithmetic and independent topology/endpoints, plus per-component '
                     'support/fallback agreement and log envelope cap2e-7. No global FP guarantee or scientific gate release.')
CONSTANTS = dict(LOG_ATOL=2e-8,LOG_RTOL=2e-11,ENDPOINT_ATOL=1e-9,
                 LOG_ENVELOPE_CAP=2e-7,ENVELOPE_ROUNDING=2e-11,SCHEMA=CONDITIONING_SCHEMA)
AUDIT_SOURCE_NAMES = frozenset('''audit_hard_free_smc.py conditioned_density_audit.py
hard_free_line_reference.py analyze_contact_line_audit.py analyze_mobile_native_pocket.py
analyze_mobile_competing_reference.py hard_free_line_physical_reference.py analyze_basin_normalizers.py
normalizer_proposal_density.py prepare_smc_normalizer_atlas.py prepare_deep_far_normalizer_atlas.py
analyze_native_region_reference.py physical_hard_free_line_vessel.py analyze_r4_smc_control.py
analyze_mobile_threshold_reference.py analyze_mobile_wall_contacts.py analyze_mobile_full_capture.py
recover_mobile_full_capture_audit.py prepare_shoulder_docking_benchmark.py audit_full_vessel_latent.py
physical_latent_guide.py review_conditional_ray_extremes.py diagnose_conditional_ray_extremes.py
analyze_latent_region.py conditional_ray_proposal.py entry_shell_proposal.py'''.split())
MAXIMUM_FIELDS = frozenset('''same_input_log_error coordinate_log_error interval_endpoint_error
independent_geometry_log_difference log_envelope_width latent_coordinate_error
raw_coordinate_error jacobian_error'''.split())


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _digest(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def _mapping(value, label):
    require(isinstance(value,dict) and all(isinstance(p,str) and _digest(h) for p,h in value.items()),label)
    return value


def _rows(rows, ids, label):
    require(isinstance(rows,list) and len(rows)==len(ids),label)
    require(all(isinstance(j,dict) and isinstance(j.get('id'),str) for j in rows),label)
    require({j['id'] for j in rows}==set(ids),label)
    return {j['id']:j for j in rows}


def _relative(root, name):
    require(isinstance(name,str) and not Path(name).is_absolute(),'Absolute frozen source path')
    path=(root/name).resolve()
    require(path.is_relative_to(root) and path!=root,'Frozen source path escapes supplemental root')
    return path


def _bind_map(ledger, values):
    for path,digest in _mapping(values,'Invalid input digest map').items():
        require(Path(path).is_absolute(),'Input binding must be absolute')
        ledger.bind(path,digest)


def _adopt_checked(ledger, values):
    """Existing checker just hashed these bytes; avoid a redundant full raw scan."""
    for path,digest in _mapping(values,'Invalid checked digest map').items():
        canonical=str(Path(path).resolve())
        require(path==canonical,'Noncanonical checked path')
        require(path not in ledger.files or ledger.files[path]==digest,'Conflicting checked binding: '+path)
        ledger.files[path]=digest


def _original_state(campaign, physical_plan, plan, ledger):
    digest=plan['input_sha256'].get(str(campaign/'status.json'))
    require(_digest(digest),'Missing immutable original failure-state hash')
    state=read(ledger.bind(campaign/'status.json',digest))
    require(state['complete'] is False and state['phase']=='independent_audit_failed',
            'Original campaign failure was changed or waived')
    require(state['plan_sha256']==plan['input_sha256'].get(str(campaign/'plan.json')),
            'Original campaign plan binding differs')
    jobs=_rows(state['jobs'],IDS,'Original physical job identities differ')
    expected=_rows(physical_plan['jobs'],IDS,'Physical plan identities differ')
    for identity in IDS:
        j,e=jobs[identity],expected[identity]
        require(j['kind']=='physical' and j['status']=='complete' and type(j['returncode']) is int and j['returncode']==0,
                'Original physical job not successfully completed')
        require(all(j.get(k)==e.get(k) for k in ['id','seed','kind','directory','log','command']),
                'Original physical job command or path differs')
        require(e['directory']==str(campaign/'populations'/identity),'Physical output path differs')
    audits=_rows(state['audits'],IDS,'Original audit identities differ')
    for identity,j in audits.items():
        require(j['kind']=='audit' and j['directory']==str(campaign/(identity+'-audit.json'))
                and j['log']==str(campaign/(identity+'-audit.log')),'Original audit paths differ')
        expected_command=[physical_plan['python'],'-B',str(campaign/'common/audit_hard_free_smc.py'),
            '--directory',str(campaign/'populations'/identity),'--binary',str(campaign/'common/latent-region-smc'),
            '--out',j['directory']]
        require(j['command']==expected_command,'Original audit command differs')
        if identity in REUSED:
            require(j['status']=='complete' and type(j['returncode']) is int and j['returncode']==0,
                    'Original successful audit is not reusable')
        else:
            require(j['status']=='failed' and _integer(j['returncode'],minimum=1),'Original failed audit was relabeled')
            require(not Path(j['directory']).exists(),'Original failed receipt path was overwritten')
            failure_hash=plan['input_sha256'].get(j['log'])
            require(_digest(failure_hash),'Missing immutable original failure-log hash')
            ledger.bind(j['log'],failure_hash)
    return state


def validate_plan(supplemental_root, expected_plan_sha, campaign_root, physical_plan, ledger):
    """Return immutable supplemental plan; deliberately do not bind status.json."""
    root,campaign=Path(supplemental_root).resolve(),Path(campaign_root).resolve()
    require(root!=campaign,'Supplemental directory must preserve original campaign')
    require(_digest(expected_plan_sha),'Missing pinned supplemental plan digest')
    plan=read(ledger.bind(root/'plan.json',expected_plan_sha))
    require(plan['schema']==PLAN_SCHEMA and plan['original_campaign']==str(campaign),
            'Supplemental plan belongs to a different protocol/campaign')
    require(plan['repository']==physical_plan['repository'],'Supplemental repository differs')
    require(type(plan['workers']) is int and plan['workers']==2,'Supplemental worker allocation differs')
    require(all(type(plan[k]) is int and plan[k]==0 for k in ['physical_jobs','new_pose_draws','new_clouds']),
            'Supplemental plan authorizes new physical work')
    require(plan['numerical_contract']==NUMERICAL_CONTRACT,'Supplemental numerical contract differs')
    _bind_map(ledger,plan['input_sha256'])
    physical_hash=plan['input_sha256'].get(str(campaign/'plan.json'))
    require(_digest(physical_hash),'Missing original physical plan binding')
    require(read(ledger.bind(campaign/'plan.json',physical_hash))==physical_plan,'Supplied physical plan differs from frozen bytes')
    _original_state(campaign,physical_plan,plan,ledger)
    sources=_mapping(plan['source_sha256'],'Invalid frozen source hashes')
    require({'run.py','prepare.py','validation.json','frozen-tests.log'}<=sources.keys(),'Missing frozen audit controller/validation sources')
    for name,digest in sources.items():ledger.bind(_relative(root,name),digest)
    audit_sources=_mapping(plan['audit_sources'],'Invalid audit source hashes')
    require(set(audit_sources)==AUDIT_SOURCE_NAMES and len(audit_sources)==26,'Audit source names/closure differ')
    closure=local_sources(root/'common/audit_hard_free_smc.py')
    require(set(closure)==AUDIT_SOURCE_NAMES,'Frozen audit import closure differs')
    for name,digest in audit_sources.items():
        require(Path(name).name==name and closure[name]==root/'common'/name,'Relabeled audit source')
        require(sources.get('common/'+name)==digest,'Audit and frozen-source digests disagree')
    assignments={}
    for node in ast.parse((root/'common/conditioned_density_audit.py').read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in CONSTANTS:
                    require(target.id not in assignments,'Duplicate numerical-contract assignment')
                    assignments[target.id]=ast.literal_eval(node.value)
    require(assignments==CONSTANTS,'Frozen density tolerances or numerical schema changed')
    jobs=_rows(plan['jobs'],SUPPLEMENTED,'Only the two failed populations may be supplemented')
    for identity,j in jobs.items():
        destination=str(root/(identity+'-audit.json'))
        command=[physical_plan['python'],'-B',str(root/'common/audit_hard_free_smc.py'),
            '--directory',str(campaign/'populations'/identity),'--binary',str(campaign/'common/latent-region-smc'),
            '--out',destination,'--conditioned-density-audit']
        require(j['kind']=='audit' and j['status']=='pending' and j['directory']==destination
                and j['log']==str(root/(identity+'-audit.log')) and j['command']==command,
                'Supplemental immutable job command/path differs')
    reused=_rows(plan['reused_audits'],REUSED,'Successful original audit reuse changed')
    for identity,j in reused.items():
        require(j['path']==str(campaign/(identity+'-audit.json'))
                and j['sha256']==plan['input_sha256'].get(j['path']),'Reused receipt binding differs')
        ledger.bind(j['path'],j['sha256'])
    require(set(plan['physical'])==set(IDS),'Physical population binding set differs')
    binary=str(campaign/'common/latent-region-smc')
    require(plan['input_sha256'].get(binary)==physical_plan['binary_sha256'],'Physical executable binding differs')
    metadata=['status.json','summary.json','manifest.json']+[f'provenance/{name}.json'
        for name in ['config','region','shape','source-bundle','initial-guide']]
    for identity,binding in plan['physical'].items():
        directory=campaign/'populations'/identity
        require(binding['directory']==str(directory),'Physical population path relabeled')
        required=_mapping(binding['required_audit_inputs'],'Invalid required physical audit inputs')
        expected_paths={str(directory/name) for name in metadata+['initialization.jsonl','stages.jsonl','attempts.jsonl']}|{binary}
        require(set(required)==expected_paths,'Missing or extra required physical audit input')
        for path in [str(directory/name) for name in metadata]+[binary]:
            require(plan['input_sha256'].get(path)==required[path],'Physical metadata pin differs')
    return plan


def _receipt_counters(receipt, summary, physical_plan):
    allocation=physical_plan['allocation']; draws=allocation['initial_draws']; stages=allocation['stages']; n=allocation['population']
    require(receipt['schema']==AUDIT_SCHEMA and receipt['complete'] is True
            and receipt['density_audit_mode']=='split_conditioned','Wrong supplemental audit schema/mode/completion')
    for name,expected in [('initial_draws',draws),('initial_hits',summary['initial_hits']),
        ('completed_stage',summary['completed_stage']),('terminal_particles',len(summary['terminal_particles']))]:
        require(_integer(receipt[name]) and receipt[name]==expected,'Supplemental receipt counter differs: '+name)
    require(receipt['initial_draws']==summary['initial_draws'] and receipt['completed_stage']==stages
            and receipt['terminal_particles']==n and receipt['initial_hits']>0,'Supplemental receipt does not cover completed positive population')
    attempted=draws+stages*n*(1+allocation['sweeps'])
    require(type(receipt['attempted_events']) is int and receipt['attempted_events']==attempted,'Supplemental attempt denominator differs')
    branches=receipt['initial_branches']
    require(isinstance(branches,dict) and set(branches)<= {'uniform','gaussian'}
            and all(_integer(v) for v in branches.values()) and sum(branches.values())==draws,'Initialization branch denominator differs')
    for name in ['new_pose_draws','new_Poisson_clouds','new_classifier_calls']:
        require(type(receipt[name]) is int and receipt[name]==0,'Supplemental receipt generated new work: '+name)
    for name in ['guide_evaluations','analytic_sphere_cloud_bounds_checked']:
        require(_integer(receipt[name]),'Invalid supplemental counter: '+name)
    for name in ['audit_cpu_seconds','maximum_interval_error','maximum_inverse_CDF_error']:
        require(type(receipt[name]) in (int,float) and math.isfinite(receipt[name]) and receipt[name]>=0,'Invalid supplemental diagnostic: '+name)
    require(receipt['maximum_interval_error']<=CONSTANTS['ENDPOINT_ATOL']
            and receipt['maximum_inverse_CDF_error']<2e-6,'Supplemental geometry/generation bound exceeded')
    require(type(receipt['log_Z']) in (int,float) and math.isfinite(receipt['log_Z'])
            and type(summary['log_Z']) in (int,float) and math.isfinite(summary['log_Z'])
            and abs(receipt['log_Z']-summary['log_Z'])<=2e-10,'Supplemental audited normalizer differs from unchanged output')
    conditioned=receipt['conditioned_density_audit']
    require(conditioned['schema']==CONDITIONING_SCHEMA,'Supplemental density audit contract differs')
    require(_integer(conditioned['queries']) and draws<=conditioned['queries']<=draws+stages*n*allocation['sweeps'],
            'Supplemental conditioning-query count differs')
    unique=len({tuple(p['pose']['position'])+tuple(p['pose']['orientation']) for p in summary['terminal_particles']})
    require(type(conditioned['certificates_retained']) is int and conditioned['certificates_retained']==unique,
            'Supplemental retained certificate count differs')
    maxima=conditioned['maxima']
    require(set(maxima)==MAXIMUM_FIELDS and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in maxima.values()),
            'Incomplete/nonfinite conditioning maxima')
    require(maxima['log_envelope_width']<=CONSTANTS['LOG_ENVELOPE_CAP']
            and maxima['interval_endpoint_error']<=CONSTANTS['ENDPOINT_ATOL']
            and maxima['latent_coordinate_error']<CONSTANTS['LOG_ATOL']
            and maxima['independent_geometry_log_difference']<=maxima['log_envelope_width']+2*CONSTANTS['ENVELOPE_ROUNDING'],
            'Supplemental conditioning bounds exceed frozen contract')
    require(abs(maxima['interval_endpoint_error']-receipt['maximum_interval_error'])<=1e-15,'Supplemental geometry maxima disagree')


def authenticate(supplemental_root, expected_plan_sha, campaign_root, physical_plan, ledger):
    """Authenticate terminal receipts, preserving original failure state verbatim."""
    root,campaign=Path(supplemental_root).resolve(),Path(campaign_root).resolve()
    plan=validate_plan(root,expected_plan_sha,campaign,physical_plan,ledger)
    state=read(ledger.bind(root/'status.json'))
    require(state['complete'] is True and state['phase']=='supplemental_audits_complete'
            and state['plan_sha256']==expected_plan_sha and type(state['physical_jobs']) is int
            and state['physical_jobs']==0,'Supplemental numerical audits are not successfully complete')
    completed=_rows(state['jobs'],SUPPLEMENTED,'Supplemental completion job identities differ')
    planned={j['id']:j for j in plan['jobs']}
    for identity,j in completed.items():
        require(all(j.get(k)==planned[identity][k] for k in ['id','kind','directory','log','command'])
                and j['status']=='complete' and type(j['returncode']) is int and j['returncode']==0
                and _digest(j.get('receipt_sha256')),'Supplemental completed job differs from frozen allocation')
    physical_jobs={j['id']:j for j in physical_plan['jobs']}
    outputs={}
    for identity in IDS:
        checked=control.check_output(campaign,physical_plan,physical_jobs[identity],identity in REUSED)
        _adopt_checked(ledger,checked['input_sha256'])
        if identity in REUSED:
            outputs[identity]=checked
            continue
        required=dict(checked['input_sha256'])
        binary=str(campaign/'common/latent-region-smc');required[binary]=physical_plan['binary_sha256']
        require(required==plan['physical'][identity]['required_audit_inputs'],'Raw physical byte bindings differ from supplemental plan')
        path=Path(completed[identity]['directory'])
        receipt=read(ledger.bind(path,completed[identity]['receipt_sha256']))
        require(receipt['directory']==str(campaign/'populations'/identity),'Supplemental receipt population relabeled')
        source_inputs={str(root/'common'/name):digest for name,digest in plan['audit_sources'].items()}
        all_required=dict(required,**source_inputs)
        require(receipt['input_sha256']==all_required,'Supplemental receipt omits, adds or relabels physical/source bindings')
        # Physical checker and validate_plan already hashed these same inputs.
        _adopt_checked(ledger,all_required)
        summary=read(campaign/'populations'/identity/'summary.json')
        _receipt_counters(receipt,summary,physical_plan)
        outputs[identity]=dict(input_sha256=dict(all_required,**{str(path):completed[identity]['receipt_sha256']}),
            initial_draws=receipt['initial_draws'],initial_hits=receipt['initial_hits'],
            audit_mode='split_conditioned',supplemental_audit=str(path))
    return outputs
