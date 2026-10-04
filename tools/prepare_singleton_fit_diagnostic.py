#!/usr/bin/env python3
"""Freeze a seedless paired fit-policy diagnostic; never run constructors.

All five anchors, five original states, two moving members and two policies
are retained. Historical mutable Rust paths can be authenticated only through
their exact old compiled-source archive entries, for the two declared edits.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import signal
import sys

import run_native_class_physical_campaign as driver

read, require, sha, write = driver.read, driver.require, driver.sha, driver.write
REPO = Path(__file__).resolve().parents[1]
SCHEMA = 'singleton-fit-diagnostic-preparation-v1'
POLICIES = ['early_abort', 'full_iterations']
ANCHORS = [228,46,106,178,179]
CAPS = dict(constructions=10,fits=40960,center_checks=2560,core_overlap_calls=673280)
LIMITS = dict(cpu_limit_seconds=600,wall_limit_seconds=1200,address_space_limit_bytes=8*2**30)
OLIGOMER = dict(multi_contact_mass=.8,max_mismatch=12.,pair_distance_A=8.,pair_angle_degrees=60.,
                max_candidates=4096,max_hard_checks=256,max_components=32)
HISTORICAL_SOURCE_EDITS = ('src/oligomer_proposal.rs','src/bin/singleton-fusion-diagnostic.rs')
REVIEW_SHA = {'original':'a777aad74a9be5c276698c0e099fa5452740d6a176114252b9fa7f9a8a0e3eb8',
              'alternatives':'9bcc5904326ce8cd50f61768c5a731c105a19a50f10242e33ad0cd4008ca490e'}
SCOPE = ('Prospective paired fit-policy diagnostic on all fifty previously declared anchor/state/member '
         'contexts, without selecting successful mean screens or fits. Early-abort runs supply explicit '
         'fit-exit reasons and must reproduce every saved old diagnostic field. Full-iterations disables '
         'only the iteration>=2 and chi2>120 abort; the 16-iteration, 12-line-search and final chi2<=12 '
         'rules remain unchanged. No proposal draws, Poisson clouds, native queries or trajectory replay. '
         'Results concern these fixed contexts, not sampling efficiency or equilibrium.')


class Inputs:
    def __init__(self): self.files = {}

    def bind(self,path,expected=None):
        path=Path(path).resolve(); digest=sha(path)
        require(expected is None or digest==expected,'Changed bound input: '+str(path))
        require(str(path) not in self.files or self.files[str(path)]==digest,'Contradictory input binding')
        self.files[str(path)]=digest
        return dict(path=str(path),sha256=digest)

    def load(self,ref):
        require(set(ref)=={'path','sha256'},'Exact BoundFile required')
        self.bind(ref['path'],ref['sha256']); return read(ref['path'])

    def recheck(self):
        for path,digest in self.files.items(): require(sha(path)==digest,'Input changed during preparation: '+path)


def source_paths():
    return [Path(__file__).resolve(),Path(driver.__file__).resolve(),
            Path(__file__).with_name('test_prepare_singleton_fit_diagnostic.py').resolve()]


def bundle_entries(inputs,ref):
    bundle=inputs.load(ref); require(bundle['schema']==1 and type(bundle['files']) is dict,'Wrong source bundle')
    for name,entry in bundle['files'].items():
        path=Path(name)
        require(not path.is_absolute() and '..' not in path.parts
                and hashlib.sha256(entry['text'].encode()).hexdigest()==entry['sha256'],'Malformed compiled source entry')
    return bundle['files']


def historical_inputs(inputs,files,bundle_ref,entries,*,repo=REPO):
    """Never excuse changed scientific inputs or arbitrary historical sources."""
    substitutions=[]; allowed={str((repo/name).resolve()):name for name in HISTORICAL_SOURCE_EDITS}
    for path,old_digest in files.items():
        actual=sha(path)
        if actual==old_digest:
            inputs.bind(path,old_digest); continue
        name=allowed.get(path)
        require(name is not None and name in entries and entries[name]['sha256']==old_digest,
                'Historical input changed outside exact compiled-source allowlist: '+path)
        require(hashlib.sha256(entries[name]['text'].encode()).hexdigest()==old_digest,
                'Historical archive entry bytes differ')
        substitutions.append(dict(path=path,historical_sha256=old_digest,current_sha256=actual,
            archived_source_bundle=bundle_ref,entry=name,
            reason='Completed old run source authenticated from immutable compiled text; current workspace differs.'))
    return substitutions


def completed_history(inputs,root,role):
    """Authenticate completed lifecycle with only the narrow archived-source join.

    Ordinary driver plan verification rehashes all old workspace sources. Here those
    same input checks are performed explicitly, permitting only the declared
    two Rust edits when their old bytes survive in the bound compiled bundle.
    """
    root=Path(root).resolve(); review_ref=inputs.bind(root/'review.json',REVIEW_SHA[role]); review=read(review_ref['path'])
    require(review['complete'] is True and review['passed'] is True,'Unpassed historical review')
    plan_ref=inputs.bind(root/'execution-plan.json'); plan=read(plan_ref['path'])
    protocol_ref=inputs.bind(root/'protocol.json'); protocol=read(protocol_ref['path'])
    require(plan['schema']==driver.SCHEMA and plan['root']==str(root)
            and plan['maximum_workers']==plan['threads']==1,'Wrong historical controller')
    bundle_ref=protocol['source_bundle']; entries=bundle_entries(inputs,bundle_ref)
    substitutions=historical_inputs(inputs,plan['files'],bundle_ref,entries)
    claim=read(root/'execution/claim.json'); status=read(root/'execution/status.json'); summary=read(root/'execution/summary.json')
    require(claim['schema']==driver.SCHEMA and claim['plan_sha256']==plan_ref['sha256']
            and claim['maximum_workers']==claim['threads']==1 and claim['retries']==claim['replacements']==0,
            'Historical controller claim differs')
    prepared=read(root/'preparation.json')
    require(prepared['complete'] is True and prepared['launched'] is False
            and prepared['execution_plan']==plan_ref and prepared['protocol']==protocol_ref
            and not (root/'preparation-failure.json').exists(),'Historical preparation differs')
    require(summary['complete'] is True and summary['passed'] is True and summary['plan_sha256']==plan_ref['sha256']
            and status['complete'] is True and status['passed'] is True and status['failure'] is None
            and status['active'] is None and status['unstarted']==[] and status['plan_sha256']==plan_ref['sha256']
            and status['completed']==summary['completed'] and len(summary['completed'])==len(plan['jobs']),
            'Incomplete historical lifecycle')
    for path in (root/'preparation.json',root/'execution/claim.json',root/'execution/status.json',root/'execution/summary.json'):
        inputs.bind(path)
    require(claim['preparation_receipt']==inputs.bind(root/'preparation.json'),'Historical preparation claim changed')
    results=[]
    for ordinal,job in enumerate(plan['jobs']):
        directory=driver.job_directory(root,ordinal,job)
        attempt=read(directory/'attempt.json'); process=read(directory/'process.json'); result=read(directory/'success.json')
        require(attempt['job']==job and read(directory/'exit.json')==result and summary['completed'][ordinal]==result,
                'Historical lifecycle records differ')
        require(all(result[k]==job[k] for k in ('id','population','phase','argv'))
                and result['success'] is True and result['error'] is None and result['timeout'] is False
                and result['child_started'] is True and result['child_drained'] is True
                and type(result['returncode']) is int and result['returncode']==0
                and result['retries']==result['replacements']==0,'Historical child did not complete cleanly')
        require(type(result['pid']) is int and result['pid']>0
                and all(process[k]==result[k] for k in ('id','pid','birth_ticks','argv')),'Historical child identity differs')
        require(result['terminal']['path']==job['terminal']['path']
                and result['success_contract']==job['terminal']['success_contract']=='complete_and_passed','Historical terminal contract differs')
        terminal=inputs.load(result['terminal']); require(terminal['complete'] is True and terminal['passed'] is True,'Historical terminal failed')
        for name in ('attempt.json','process.json','exit.json','success.json'): inputs.bind(directory/name)
        argv=job['argv']; require(len(argv)==7 and argv[1]=='--manifest' and argv[3]=='--manifest-sha256' and argv[5]=='--out','Wrong historical command')
        manifest_ref=inputs.bind(argv[2],argv[4]); manifest=read(manifest_ref['path'])
        require(manifest['output']==argv[6] and result['terminal']['path']==str(Path(argv[6])/'summary.json')
                and terminal['manifest_sha256']==manifest_ref['sha256'],'Historical manifest/output differs')
        journal=inputs.bind(Path(argv[6])/'attempts.jsonl',terminal['attempts_sha256'])
        results.append(dict(manifest=manifest,manifest_ref=manifest_ref,terminal=result['terminal'],journal=journal))
    terminal_refs=[r['terminal'] for r in results]; summary_ref=inputs.bind(root/'execution/summary.json')
    if role=='original':
        require(len(results)==1 and review['execution_plan']==plan_ref and review['execution_summary']==summary_ref
                and review['diagnostic']==terminal_refs[0],'Original review does not join completed lifecycle')
    else:
        require(len(results)==4 and review['plan_sha256']==plan_ref['sha256']
                and review['execution_summary_sha256']==summary_ref['sha256'] and review['terminals']==terminal_refs,
                'Alternative review does not join completed lifecycle')
    return dict(review=review_ref,execution_plan=plan_ref,protocol=protocol_ref,execution_summary=summary_ref,
                terminals=terminal_refs,source_substitutions=substitutions),results


def validate_originals(results):
    require(len(results)==5 and [r['manifest']['anchor'] for r in results]==ANCHORS,'Wrong complete five-anchor inventory')
    original=results[0]['manifest']; ids=['source']+[f'prepared-{i}' for i in range(4)]
    expected=[dict(id=f'{s}-{i}',state=s,moving=i,neighbors=[j,228]) for s in ids for i,j in ((27,132),(132,27))]
    require(original['schema']=='singleton-fusion-diagnostic-manifest-v1' and original['members']==[27,132]
            and original['coordinate_frame']=='sphere_centered' and original['wall_center']==[0.,0.,0.]
            and original['oligomer']==OLIGOMER and original['caps']==CAPS
            and original['constructions']==expected and [s['id'] for s in original['states']]==ids
            and [s['kind'] for s in original['states']]==['source']+['prepared']*4,'Original state/frame/policy changed')
    baseline={}
    for result in results:
        manifest=result['manifest']; restored=copy.deepcopy(manifest)
        for key in ('context','anchor','output'): restored[key]=original[key]
        for construction in restored['constructions']: construction['neighbors'][1]=228
        require(restored==original,'Saved alternative changed more than anchor context')
        rows={}
        with Path(result['journal']['path']).open() as stream:
            for line in stream:
                require(len(line)<=2**22,'Oversized historical diagnostic record')
                import json
                row=json.loads(line)
                if row['state']!='complete': continue
                ordinal=len(rows)
                require(ordinal<10 and row['ordinal']==ordinal and row['construction']==manifest['constructions'][ordinal]
                        and row['error'] is None and row['diagnostics']['complete'] is True,'Incomplete/reordered old constructor')
                rows[row['construction']['id']]=row['diagnostics']
        require(list(rows)==[c['id'] for c in expected],'Missing original default baseline')
        baseline[str(manifest['anchor'])]=dict(manifest=result['manifest_ref'],journal=result['journal'],diagnostics=rows)
    return original,baseline


def validate_build(inputs,witness_ref,*,repo=REPO):
    witness=inputs.load(witness_ref)
    require(witness['schema']=='singleton-fusion-build-witness-v1' and witness['complete'] is True
            and witness['passed'] is True and witness['production_executable_unchanged'] is True
            and witness.get('fit_policies')==POLICIES and witness.get('default_kernel_unchanged') is True
            and witness.get('per_fit_trace') is True,'Missing passed fit-capable isolated build witness')
    for key in ('executable','cli_source','validation'): inputs.bind(witness[key]['path'],witness[key]['sha256'])
    require(os.access(witness['executable']['path'],os.X_OK),'Diagnostic binary is not executable')
    entries=bundle_entries(inputs,witness['source_bundle'])
    validation=inputs.load(witness['validation'])
    require(validation['complete'] is True and validation['passed'] is True
            and validation['compiled_source_bundle']['sha256']==witness['source_bundle']['sha256'], 'Build validation/source bundle differs')
    for name in HISTORICAL_SOURCE_EDITS:
        require(name in entries and validation['sources'][name]==dict(path=str(repo/name),sha256=entries[name]['sha256']),
                'Build validation omits current fit source')
    require(witness['cli_source']==dict(path=str(repo/HISTORICAL_SOURCE_EDITS[1]),sha256=entries[HISTORICAL_SOURCE_EDITS[1]]['sha256']),
            'CLI source witness differs')
    for name,entry in entries.items(): inputs.bind(repo/name,entry['sha256'])
    return witness


def validate_tests(inputs,ref):
    value=inputs.load(ref)
    require(value['complete'] is True and value['passed'] is True and value['source_before']==value['source_after'],
            'Preparer test receipt is not passed/source-stable')
    for path in source_paths():
        expected=value['source_before'].get(str(path)); require(expected is not None,'Missing preparer validation source')
        inputs.bind(path,expected)


def paired_manifests(original,witness,root):
    """Pure Cartesian transformation; never inspect fitted or mean-screen counts."""
    result=[]
    for policy in POLICIES:
        for anchor in ANCHORS:
            manifest=copy.deepcopy(original); manifest['fit_policy']=policy
            manifest['context']=f'whole_27_132_anchor_{anchor}_{policy}'
            manifest['anchor']=anchor; manifest['output']=str(root/'diagnostics'/f'{policy}-anchor-{anchor}')
            manifest['witness']=dict(executable_sha256=witness['executable']['sha256'],
                compiled_source_bundle_sha256=witness['source_bundle']['sha256'],cli_source_sha256=witness['cli_source']['sha256'])
            for construction in manifest['constructions']: construction['neighbors'][1]=anchor
            restored=copy.deepcopy(manifest); restored.pop('fit_policy')
            for key in ('context','anchor','output','witness'): restored[key]=original[key]
            for construction in restored['constructions']: construction['neighbors'][1]=original['anchor']
            require(restored==original,'Paired manifest changed undeclared fields')
            result.append(manifest)
    return result


def admit(original_root,alternatives_root,witness_path,validation_path):
    inputs=Inputs(); old,first=completed_history(inputs,original_root,'original')
    alternatives,rest=completed_history(inputs,alternatives_root,'alternatives')
    original,baseline=validate_originals(first+rest)
    witness_ref=inputs.bind(witness_path); witness=validate_build(inputs,witness_ref)
    validation_ref=inputs.bind(validation_path); validate_tests(inputs,validation_ref)
    inputs.bind(Path(sys.executable).resolve()); inputs.recheck()
    return dict(inputs=inputs,original=original,baseline=baseline,witness=witness,witness_ref=witness_ref,
                validation=validation_ref,historical=[old,alternatives])


def materialize(root,context):
    inputs=context['inputs']; witness=context['witness']; manifests=paired_manifests(context['original'],witness,root)
    (root/'manifests').mkdir(); (root/'diagnostics').mkdir(); jobs=[]; refs=[]
    for manifest in manifests:
        identity=f'{manifest["fit_policy"]}-anchor-{manifest["anchor"]}'
        path=root/'manifests'/(identity+'.json'); write(path,manifest); ref=inputs.bind(path); refs.append(ref)
        jobs.append(dict(id=identity,population=manifest['context'],phase='geometry',**LIMITS,
            argv=[witness['executable']['path'],'--manifest',str(path),'--manifest-sha256',ref['sha256'],'--out',manifest['output']],
            terminal=dict(path=str(Path(manifest['output'])/'summary.json'),success_contract='complete_and_passed')))
    totals={k:10*v for k,v in CAPS.items()}
    require(totals==dict(constructions=100,fits=409600,center_checks=25600,core_overlap_calls=6732800),'Allocation changed')
    protocol=dict(schema=SCHEMA,complete=True,launched=False,root=str(root),scope=SCOPE,
        anchors=ANCHORS,fit_policies=POLICIES,manifests=refs,historical=context['historical'],
        baseline_diagnostics=context['baseline'],default_equivalence_required=True,
        default_equivalence_contract='Compare every saved old diagnostics key/value for each of fifty early_abort constructions; new fit trace fields are additional.',
        build_witness=context['witness_ref'],preparer_validation=context['validation'],
        maximum_workers=1,threads=1,per_job_caps=CAPS,aggregate_caps=totals,per_job_limits=LIMITS,
        total_cpu_limit_seconds=6000,total_wall_limit_seconds=12000,peak_address_space_limit_bytes=8*2**30,
        retries=0,replacements=0,geometry_queries_during_preparation=0,new_pose_draws=0,input_sha256=dict(inputs.files))
    path=root/'protocol.json'; write(path,protocol); protocol_ref=inputs.bind(path)
    executable=witness['executable']['path']
    execution=dict(schema=driver.SCHEMA,root=str(root),maximum_workers=1,threads=1,files=dict(inputs.files),jobs=jobs,
        executable_resolutions={executable:str(Path(executable).resolve())},preparation_receipt=str(root/'preparation.json'),scope=SCOPE)
    write(root/'execution-plan.json',execution); inputs.recheck()
    receipt=dict(schema=SCHEMA,complete=True,launched=False,protocol=protocol_ref,aggregate_caps=totals,
        execution_plan=dict(path=str(root/'execution-plan.json'),sha256=sha(root/'execution-plan.json')),
        geometry_queries_during_preparation=0,new_pose_draws=0)
    write(root/'preparation.json',receipt)
    driver.verify_plan(root/'execution-plan.json',execution,receipt['execution_plan']['sha256'],fresh=True)
    return receipt


def prepare(original_root,alternatives_root,witness_path,validation_path,*,out):
    root=Path(out).resolve(); require(not root.exists(),'Fresh preparation root required')
    context=admit(original_root,alternatives_root,witness_path,validation_path)
    root.mkdir(); write(root/'preparation-claim.json',dict(schema=SCHEMA,pid=os.getpid(),launched=False,input_sha256=context['inputs'].files))
    handlers={}
    def stop(sig,_): raise InterruptedError('Preparation signal '+str(sig))
    try:
        for sig in (signal.SIGTERM,signal.SIGINT): handlers[sig]=signal.signal(sig,stop)
        return materialize(root,context)
    except BaseException as error:
        write(root/'preparation-failure.json',dict(schema=SCHEMA,complete=False,launched=False,error=repr(error),retries=0)); raise
    finally:
        for sig,handler in handlers.items(): signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root',type=Path,default=REPO/'results/singleton-fusion-context0-20261004')
    parser.add_argument('--alternatives-root',type=Path,default=REPO/'results/singleton-alternative-fusion-context0-20261004')
    parser.add_argument('--build-witness',type=Path,required=True); parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    import json
    print(json.dumps(prepare(args.original_root,args.alternatives_root,args.build_witness,args.validation,out=args.out),indent=2))
