#!/usr/bin/env python3
"""Freeze exactly 32 matched root_m4 chains, reusing authenticated preparations.

Metadata and hashes only: never construct geometry, sample, thin, or select starts.
The old prepared manifest keeps its original config/binding identity.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys

import prepare_evolving_dimer_benchmark as old

ROOT = Path(__file__).resolve().parents[1]
require, read, checked = old.require, old.read, old.checked_file
EXAMPLE = old.EXAMPLE
ROLE_MAP = dict(proposal='m4/proposal', internal_threshold='m4/threshold',
    root_threshold='root_m4/root_threshold', bath='m4/bath', accept='m4/accept',
    local='local/{attempt}/{proposal,bath,accept}')
LIMITS = dict(max_workers=3, cpu_seconds=3600, wall_seconds=7200, address_space_bytes=16*1024**3)


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    p=Path(path).resolve()
    return dict(path=str(p), sha256=sha(p))


def runtime():
    return dict(python=sys.version, executable=record(sys.executable), platform=platform.platform(),
        packages={name:version(name) for name in ('numpy','scipy')},
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS')})


def allocation():
    return dict(schema='evolving-dimer-root-extension-allocation-v1', contexts=4,
        arms=['root_m4'], initializations=old.INITIALIZATIONS, streams=4, chains=32,
        warmup_blocks=512, production_blocks=4096, blocks_per_chain=4608,
        local_attempts_per_block=4, local_member_order=[0,1,0,1],
        extra_dimer_per_block={'root_m4':1}, local_attempts=589824, dimer_attempts=147456,
        retained_block_observations=147456, production_observations=131072,
        initial_observations=32, inherited_cloud_banks=32, inherited_alternative_starts=16,
        new_cloud_banks=0, new_preparation_attempts=0, new_raw_cloud_points=0,
        master_seed=old.MASTER, extension=True, replacement=False,
        native_classification=False, python_geometry_evaluations=0,
        physical_draws_during_python_preparation=0, execution_limits=LIMITS,
        limits_scope='Raw, retained and CPU limits apply separately to each new chain. No refill, restart or replacement.')


def inherited(original, analysis_root):
    """Authenticate metadata authority; do not inspect cloud/start/trajectory data."""
    original,analysis_root=Path(original).resolve(),Path(analysis_root).resolve()
    authority={k:record(original/p) for k,p in dict(config='config.json',binding='binding.json',
        run_binding='run-binding.json',freeze='freeze.json',protocol='protocol.json',
        prepared_manifest='prepared/manifest.json',preparation_audit='preparation-audit-v2/result.json',
        dispatch_status='dispatch/status.json').items()}
    config=old.verify_freeze(original)
    require(config['jobs']==old.jobs() and config['allocation']==old.allocation(), 'Original allocation differs')
    status=read(authority['dispatch_status']['path'])
    require(status['complete'] is True and status['passed'] is True and not status['active'] and not status['unstarted'], 'Original campaign incomplete')
    require(len(status['completed'])==96 and {x['job']['id'] for x in status['completed']}==set(range(96))
            and all(x['success'] is True and x['job']==config['jobs'][x['job']['id']] for x in status['completed']), 'Original terminal inventory differs')
    selected=[j for j in config['jobs'] if j['arm']=='m4']
    require(len(selected)==32 and {(j['context_index'],j['initialization'],j['stream']) for j in selected}
        =={(c,i,s) for c in range(4) for i in old.INITIALIZATIONS for s in range(4)}, 'Control family inventory differs')
    binding,run_binding,prepared,audit=[read(authority[k]['path']) for k in ('binding','run_binding','prepared_manifest','preparation_audit')]
    require(binding['config_sha256']==authority['config']['sha256'] and binding['freeze_sha256']==authority['freeze']['sha256']
        and binding['protocol_sha256']==authority['protocol']['sha256'], 'Original prelaunch binding differs')
    require(run_binding['complete'] is True and run_binding['config_sha256']==authority['config']['sha256']
        and run_binding['prelaunch_binding']==authority['binding'] and run_binding['prepared_manifest']==authority['prepared_manifest'], 'Original run binding differs')
    require(prepared['complete'] is True and prepared['passed'] is True and prepared['all_attempts_retained'] is True
        and prepared['config_sha256']==authority['config']['sha256'] and prepared['binding_sha256']==authority['binding']['sha256'], 'Inherited preparation identity differs')
    require(audit['schema']=='evolving-dimer-preparation-independent-audit-v1' and audit['complete'] is True and audit['passed'] is True
        and audit['physical']==config['physical'] and audit['source_contexts']==config['contexts'], 'Independent preparation audit failed or mismatched')
    for k in ('config','binding','prepared_manifest'):
        a=authority[k];require(audit['input_sha256'].get(a['path'])==a['sha256'],'Preparation audit authority differs '+k)
    checked(audit['protocol'])
    files={str(Path(f['path']).resolve()):f['sha256'] for f in prepared['files']}
    require(len(files)==len(prepared['files']) and files==run_binding['prepared_files'], 'Prepared inventory differs')
    for p,h in files.items():require(sha(p)==h, 'Changed inherited prepared file '+p)
    require(len(prepared['cloud_banks'])==32 and {(x['context_index'],x['initialization'],x['stream']) for x in prepared['cloud_banks']}
        =={(c,i,s) for c in range(4) for i in old.INITIALIZATIONS for s in range(4)}
        and len(prepared['alternative_starts'])==16 and {(x['context_index'],x['stream']) for x in prepared['alternative_starts']}
        =={(c,s) for c in range(4) for s in range(4)}, 'Prepared family counts differ')
    for group,keys in [(prepared['cloud_banks'],('raw','metadata')),(prepared['alternative_starts'],('record','ledger'))]:
        for item in group:
            for k in keys:
                ref=item[k];require(files.get(str(Path(ref['path']).resolve()))==ref['sha256'],'Preparation nested binding omitted')
    for j in selected:
        terminal=original/'execution'/f"job-{j['id']:03}"/'terminal.json'
        prior=next(x for x in status['completed'] if x['job']['id']==j['id'])
        require(sha(terminal)==prior['terminal_sha256'] and read(terminal)['complete'] is True, 'Changed old control terminal')
    summary=read(analysis_root/'summary.json'); manifest=read(analysis_root/'analysis/manifest.json')
    require(summary['complete'] is True and summary['passed'] is True and summary['chains']==96 and manifest['complete'] is True,
        'Completed control analysis missing')
    controls=dict(summary=record(analysis_root/'summary.json'), manifest=record(analysis_root/'analysis/manifest.json'),
        analysis=record(analysis_root/'analysis/analysis.json'), input_binding=record(analysis_root/'analysis/input-binding.json'), observations={})
    original_analysis_binding=read(controls['input_binding']['path'])
    require(manifest['files']['input-binding.json']==controls['input_binding']['sha256']
        and original_analysis_binding['config_sha256']==authority['config']['sha256']
        and original_analysis_binding['run_binding_sha256']==authority['run_binding']['sha256'],'Control analysis input authority differs')
    for name,digest in original_analysis_binding['source_files'].items():
        require(sha(ROOT/name)==digest,'Inherited observer implementation changed '+name)
    require(summary['analysis']==controls['analysis'] and summary['manifest_sha256']==controls['manifest']['sha256']
        and manifest['files']['analysis.json']==controls['analysis']['sha256'], 'Control analysis binding differs')
    for j in selected:
        name=f"job-{j['id']:03}-observations.jsonl"
        item=record(analysis_root/'analysis'/name)
        require(manifest['files'][name]==item['sha256'], 'Cached control observations changed')
        controls['observations'][str(j['id'])]=item
    return config,selected,authority,controls,files


def validation_bindings(receipts):
    require(set(receipts)=={'integration','root_factor','balance','adapter','auditor','observer'},'Six validation receipt labels required')
    bindings={};source_names=set()
    for label,path in receipts.items():
        item=record(path);v=read(path)
        require(v.get('passed') is True and v.get('complete',True) is True,'Failed validation '+label)
        sources=v.get('source_after',v.get('source_sha256'))
        if 'source_before' in v:require(v['source_before']==v['source_after'],'Changed tested source '+label)
        if label=='balance':sources={'tools/test_external_overlap_balance.py':sources}
        require(isinstance(sources,dict) and sources,'Missing validated source hashes '+label)
        if label=='root_factor':
            # This historical receipt precedes the new evolving_dimer integration;
            # authenticate exactly its unchanged building blocks and tests.
            names=['src/factorized_dimer.rs','src/auxiliary_overlap_threshold.rs',
                'tests/factorized_root_guidance.rs','tests/factorized_dimer.rs','tests/auxiliary_overlap_threshold.rs']
            sources={n:sources[n] for n in names}
        for name,digest in sources.items():
            p=ROOT/name;require(p.is_file() and sha(p)==digest,'Changed validated source '+label+': '+name)
            source_names.add(name)
        bindings[label]=item
    return bindings,source_names


def prepare(base, original, control_analysis, validation_receipts):
    base=Path(base).resolve()
    require(not base.exists(), 'Destination exists; no overwrite or retry')
    original_config, controls, authority, analysis, prepared_files=inherited(original,control_analysis)
    validations,validated_sources=validation_bindings(validation_receipts)
    from analyze_root_guided_dimer_benchmark import analysis_plan
    from audit_evolving_dimer_root_guidance import audit_protocol
    ap,aup=analysis_plan(),audit_protocol()
    compiled_paths=[ROOT/p for p in ('Cargo.toml','Cargo.lock','build.rs','vendor/README.md')]+sorted((ROOT/'src').rglob('*.rs'))
    compiled={str(p.relative_to(ROOT)):sha(p) for p in compiled_paths}
    toolnames=['tools/prepare_evolving_dimer_root_extension.py','tools/run_evolving_dimer_root_extension.py',
        'tools/test_evolving_dimer_root_extension.py','tools/audit_evolving_dimer_root_guidance.py',
        'tools/test_audit_evolving_dimer_root_guidance.py']+ap['source_files']
    closure=compiled_paths+[ROOT/EXAMPLE]+old.source_dependencies([ROOT/p for p in toolnames])
    closure += [ROOT/p for p in validated_sources]
    closure+=sorted(p for p in (ROOT/'vendor').rglob('*') if p.is_file())
    sources={str(p.relative_to(ROOT)):sha(p) for p in dict.fromkeys(closure)}
    base.mkdir(parents=True)
    (base/'common').mkdir()
    write(base/'common/scientific-allocation.json',allocation())
    write(base/'analysis-plan.json',ap);write(base/'root-guidance-audit-plan.json',aup)
    for name,h in sources.items():
        p=base/'common/source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,p);require(sha(p)==h,'Source changed while freezing '+name)
    mapping=[dict(job_id=j['id'],control_job_id=j['id'],context_index=j['context_index'],initialization=j['initialization'],stream=j['stream']) for j in controls]
    protocol=dict(schema='evolving-dimer-root-extension-protocol-v1',allocation=allocation(),
        inherited_campaign=authority,control_analysis=analysis,control_mapping=mapping,roles=ROLE_MAP,validation_receipts=validations,
        analysis_plan=record(base/'analysis-plan.json'),root_guidance_audit_plan=record(base/'root-guidance-audit-plan.json'),
        guidance=dict(root_m=4,internal_m=4,cloud='Same fixed body points reused in anchor/root frames; threshold streams independent.',
            old_preparation='Original manifest/config/binding identities retained; no new preparation or thinning.'),
        changed=['root_m4 anchor-root guide and its independent threshold role','CPU budget only: 1800 to 3600 seconds'],
        interpretation='Conditional two-mobile relaxation at radius1.4/activity0.0275; not full-system assembly or original-condition stability.',
        source_files=sources,runtime=runtime(),retry=False,replacement=False)
    write(base/'protocol.json',protocol)
    config=copy.deepcopy(original_config)
    config.update(scientific_allocation=record(base/'common/scientific-allocation.json'),protocol=record(base/'protocol.json'),
        jobs=[dict(j,arm='root_m4') for j in controls],allocation=allocation(),compiled_source_sha256=compiled,
        output=str(base/'execution'),inherited_campaign=authority,control_analysis=analysis,control_mapping=mapping)
    config['limits']['cpu_seconds']=LIMITS['cpu_seconds']
    # preparation_output intentionally retains the old path, and is never invoked.
    write(base/'config.json',config)
    bindings={}
    def collect(x):
        if isinstance(x,dict):
            if set(x)=={'path','sha256'}:bindings[x['path']]=x['sha256']
            else:
                for v in x.values():collect(v)
        elif isinstance(x,list):
            for v in x:collect(v)
    collect(config);collect(validations);bindings.update(prepared_files)
    frozen={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()}
    write(base/'freeze.json',dict(schema='evolving-dimer-root-extension-freeze-v1',complete=True,
        scientific_execution_started=False,files=frozen,input_sha256=bindings,runtime=runtime()))
    verify(base)
    return dict(root=str(base),config=record(base/'config.json'),freeze=record(base/'freeze.json'),chains=32,launched=False)


def verify(base):
    base=Path(base).resolve();f=read(base/'freeze.json')
    require(f['schema']=='evolving-dimer-root-extension-freeze-v1' and f['complete'] is True and f['scientific_execution_started'] is False,'Invalid freeze')
    for n,h in f['files'].items():
        p=(base/n).resolve();require(p.is_relative_to(base) and sha(p)==h,'Frozen source/input changed '+n)
    for p,h in f['input_sha256'].items():require(sha(p)==h,'Inherited input changed '+p)
    c=read(base/'config.json');o=read(checked(c['inherited_campaign']['config']))
    selected=[j for j in o['jobs'] if j['arm']=='m4']
    expected=copy.deepcopy(o)
    for k in ('scientific_allocation','protocol','jobs','allocation','compiled_source_sha256','output','inherited_campaign','control_analysis','control_mapping'):
        expected[k]=c[k]
    expected['limits']['cpu_seconds']=3600
    require(c==expected and c['jobs']==[dict(j,arm='root_m4') for j in selected] and len(selected)==32
        and c['allocation']==allocation() and c['output']==str(base/'execution'),'New config changes exceed matched scope')
    require(read(base/'protocol.json')['roles']==ROLE_MAP,'RNG role contract differs')
    return c


def bind(base, executable, source_bundle=None):
    base=Path(base).resolve();config=verify(base)
    require(not(base/'binding.json').exists() and not(base/'run-binding.json').exists(),'Already bound')
    paths=[Path(source_bundle)] if source_bundle else sorted((ROOT/'target-validation-line-guide/release/build').glob('tetramer-mc-*/out/source-bundle.json'))
    matches=[]
    for p in paths:
        b=read(p)
        require(all(hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256'] for v in b['files'].values()),'Corrupt bundle')
        if {k:v['sha256'] for k,v in b['files'].items()}==config['compiled_source_sha256']:matches.append(p)
    require(matches and len({sha(p) for p in matches})==1,'No unique matching compiled bundle')
    require(sha(ROOT/EXAMPLE)==sha(base/'common/source'/EXAMPLE),'Example changed since freeze')
    target=base/'common/evolving_dimer_benchmark';require(not target.exists(),'Partial binding')
    digest=sha(executable);shutil.copyfile(executable,target);require(sha(target)==digest,'Executable changed');target.chmod(0o755)
    shutil.copyfile(matches[0],base/'common/source-bundle.json')
    binding=dict(schema='evolving-dimer-binding-v1',config_sha256=sha(base/'config.json'),protocol_sha256=sha(base/'protocol.json'),
        freeze_sha256=sha(base/'freeze.json'),example_source_sha256=sha(base/'common/source'/EXAMPLE),
        compiled_source_bundle_sha256=sha(base/'common/source-bundle.json'),executable_sha256=digest)
    write(base/'binding.json',binding)
    authority=config['inherited_campaign'];old_run=read(checked(authority['run_binding']))
    run=dict(schema='evolving-dimer-run-binding-v1',complete=True,config_sha256=sha(base/'config.json'),
        protocol_sha256=sha(base/'protocol.json'),prelaunch_binding=record(base/'binding.json'),
        prepared_manifest=authority['prepared_manifest'],prepared_files=old_run['prepared_files'],
        cloud_banks=32,alternative_starts=16,inherited_preparation_authority=authority)
    write(base/'run-binding.json',run)
    return dict(binding=record(base/'binding.json'),run_binding=record(base/'run-binding.json'),executable=record(target),launched=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--original',type=Path);p.add_argument('--control-analysis',type=Path)
    p.add_argument('--validation-receipt',action='append',default=[],help='label=path for integration/root_factor/balance/adapter/auditor/observer');p.add_argument('--bind-executable',type=Path);p.add_argument('--source-bundle',type=Path);a=p.parse_args()
    if a.bind_executable:r=bind(a.root,a.bind_executable,a.source_bundle)
    else:
        p.error('Preparation requires --original and --control-analysis') if a.original is None or a.control_analysis is None else None
        r=prepare(a.root,a.original,a.control_analysis,dict(x.split('=',1) for x in a.validation_receipt))
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
