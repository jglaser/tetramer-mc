#!/usr/bin/env python3
"""Once-only contact profiles after all four frozen protein SMC audits pass.

Preparation reads completed reference summaries, not live trajectories. Run
requires terminal audit receipts and uses no new poses, clouds or density audit.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RAYON_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import time
import analyze_r4_smc_control as old
import run_hard_free_protein_smc as control
from compare_r4_smc_importance import importance_rows, smc_rows, validate_strata, STRATA_DEFINITION
from run_native_excluded_smc_campaign import mass_statistics, mass_comparison
from prepare_shoulder_docking_benchmark import local_dependencies

read, write, sha, require = old.read, old.write, old.sha, old.require
SCOPE = ('Independent whole-population fixed-R4 SMC masses, with rejected/retained descendants '
    'and completed zero populations included. No descendant IID error or contact-mixing ESS. '
    'Intermediate profiles refer to this guide-dependent bridge, not physical equilibrium. '
    'All completed IID convergence failures remain; no full-vessel or assembly conclusion.')


def prepare(campaign, campaign_sha, out):
    campaign, out = Path(campaign).resolve(), Path(out).resolve()
    require(not out.exists(), 'Fresh analysis preparation required')
    p = control.validate(campaign,campaign_sha)
    require(sha(control.__file__) == sha(campaign/'common/run_hard_free_protein_smc.py'), 'Use the checked campaign controller contract')
    context = old.validate_control(p['historical_protocol'],p['historical_protocol_sha256'])
    require(context['config'] == read(campaign/'common/config.json')
        and context['current'] == read(campaign/'common/region.json'), 'Endpoint differs from historical physical target')
    ledger = old.Ledger()
    for path,digest in context['bindings'].items(): ledger.bind(path,digest)
    historical = Path(p['historical_analysis'])
    h = read(ledger.bind(historical,p['preparation_input_sha256'][str(historical)]))
    require(h['complete'] and h['protocol_sha256'] == p['historical_protocol_sha256']
        and h['native_definition_sha256'] == context['definition_sha256'], 'Historical completed summary differs')
    paths = [Path(path) for path in p['preparation_input_sha256'] if Path(path).name == 'population-size-comparison.json']
    require(len(paths) == 1,'Ambiguous larger IID reference')
    iid = paths[0].parent/'comparison/analysis.json'
    d = read(ledger.bind(iid,p['preparation_input_sha256'][str(iid)]))
    iid_protocol = paths[0].parent/'protocol.json'
    ip = read(ledger.bind(iid_protocol,p['preparation_input_sha256'][str(iid_protocol)]))
    require(d['complete'] and d['schema'] == 'hard-free-line-physical-comparison-v1'
        and d['protocol_sha256'] == sha(iid_protocol)
        and d['native_definition']['definition_sha256'] == context['definition_sha256'], 'IID observer/summary differs')
    require(ip['shape_sha256'] == context['protocol']['physical_target']['shape_sha256']
        and ip['region_sha256'] == context['protocol']['physical_target']['region_sha256']
        and ip['strata'] == STRATA_DEFINITION, 'IID physical target/strata differ')
    validate_strata(h,dict(strata_definition=ip['strata'],arms=d['arms']))
    seeds = [j['seed'] for j in p['jobs']]+[j['seed'] for j in h['populations']]
    seeds += [j['seed'] for arm in d['arms'].values() for j in arm['populations']]
    require(len(seeds) == len(set(seeds)) == 16,'Reference/new populations share streams')
    sources = local_dependencies([Path(__file__),Path(__file__).with_name('test_analyze_hard_free_protein_smc.py')])
    (out/'source').mkdir(parents=True)
    for name,path in sources.items(): shutil.copy2(path,out/'source'/name)
    shutil.copy2(historical,out/'historical.json'); shutil.copy2(iid,out/'iid.json')
    write(out/'context.json',context)
    ledger.recheck()
    # The complete context includes historical analyzer source provenance. Only
    # classifier data/code reached through its frozen definition are executed
    # externally; local analysis modules are copied into this preparation.
    repo_tools = Path(p['repository'])/'tools'
    external = {path:digest for path,digest in context['bindings'].items()
        if not Path(path).is_relative_to(repo_tools)}
    plan = dict(schema='hard-free-protein-smc-analysis-preparation-v1',campaign=str(campaign),
        campaign_plan_sha256=campaign_sha,analysis=p['analysis'],workers=2,
        preparation_input_sha256=ledger.files,external_required_sha256=external,
        historical_summary_sha256=sha(historical),iid_summary_sha256=sha(iid),
        native_definition_sha256=context['definition_sha256'],runtime=control.runtime(),
        no_retries=True,scope=SCOPE,full_vessel_gate_open=False,assembly_gate_open=False,
        files_sha256={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
    write(out/'plan.json',plan)
    write(out/'freeze.json',dict(files=dict(plan['files_sha256'],**{'plan.json':sha(out/'plan.json')})))
    return dict(prepared=True,launched=False,analysis_plan_sha256=sha(out/'plan.json'))


def validate_preparation(out, expected):
    ledger = old.Ledger(); plan = read(ledger.bind(out/'plan.json',expected)); files = ledger.frozen(out)
    require(files == dict(plan['files_sha256'],**{'plan.json':expected}), 'Analysis freeze catalog differs')
    require(plan['schema'] == 'hard-free-protein-smc-analysis-preparation-v1'
        and plan['workers'] == 2 and plan['analysis'] == control.ANALYSIS
        and not plan['full_vessel_gate_open'] and not plan['assembly_gate_open'], 'Analysis allocation/scope differs')
    for path,digest in plan['external_required_sha256'].items(): ledger.bind(path,digest)
    require(control.runtime() == plan['runtime'],'Analysis runtime changed')
    physical = control.validate(Path(plan['campaign']),plan['campaign_plan_sha256'])
    return plan,physical,ledger


def require_audited_status(state, expected):
    require(state['complete'] is True and state['phase'] == 'audited_awaiting_classification'
        and state['plan_sha256'] == expected,'All independent SMC audits must finish before classification')
    for name in ('jobs','audits'):
        rows = state[name]
        require(len(rows) == 4 and {r['id'] for r in rows} == {'r00','r01','r02','r03'}
            and all(r['status'] == 'complete' and r['returncode'] == 0 for r in rows), 'Incomplete/duplicated '+name)


def stage_rows(path, expected):
    """Hash the same bytes parsed for classification, bounded to one stage."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for line in stream:
            digest.update(line); yield json.loads(line)
    require(digest.hexdigest() == expected,'Stage history differs from independent audit')


def profile_stages(stages, summary, observer, labels, selected):
    """Counts retain every descendant; only expensive deterministic labels cache."""
    profiles = {}; diagnostics = []; initial_classes = {}; last = None
    for index,stage in enumerate(stages):
        require(stage['stage'] == index,'Missing/reordered classified stage')
        particles = stage['particles']; last = stage
        if index == 0:
            for particle in particles:
                ancestor = particle['initial_ancestor']
                classes = observer.classify(particle['pose'],True)['classes']
                require(ancestor not in initial_classes or initial_classes[ancestor] == classes,'Stage-zero ancestor identity differs')
                initial_classes[ancestor] = classes
        require('ancestry' in stage or (summary['zero_estimate'] and index == 0 and not particles),
            'Missing nonempty stage genealogy')
        diagnostics.append(dict(stage=index,ancestry=stage.get('ancestry',summary.get('ancestry')),
            pre_resampling_weight_ESS=stage.get('pre_resampling_weight_ESS')))
        if index in selected:
            profiles[str(index)] = old.profile(particles,stage['log_Z'],observer,index,labels,initial_classes)
    require(last is not None and last['stage'] == summary['completed_stage']
        and last['particles'] == summary['terminal_particles'] and last['log_Z'] == summary['log_Z'],
        'Classified endpoint differs from audited summary')
    require(set(map(int,profiles)) == ({0} if summary['zero_estimate'] else set(selected)), 'Missing fixed contact profile')
    terminal = copy.deepcopy(profiles[str(last['stage'])])
    terminal.update(stage='terminal',measure='final physical target')
    return dict(profiles=profiles,terminal=terminal,stages=diagnostics,
        terminal_label_alias_stage=last['stage'],initial_ancestor_labels=len(initial_classes))


def classify_population(context, job, receipt, destination, selected):
    started = time.process_time(); destination = Path(destination)
    require(not destination.exists(),'No replay of a classified population'); destination.mkdir(parents=True)
    root = Path(job['directory']); bindings = receipt['input_sha256']
    summary_path = root/'summary.json'; require(sha(summary_path) == bindings[str(summary_path)],'Summary changed after audit')
    summary = read(summary_path); observer = old.Observer(context)
    label_path = destination/'labels.jsonl.gz'
    with label_path.open('xb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=1) as packed, io.TextIOWrapper(packed) as stream:
        result = profile_stages(stage_rows(root/'stages.jsonl',bindings[str(root/'stages.jsonl')]),summary,observer,stream,selected)
    result.update(schema='hard-free-protein-smc-contact-population-v1',complete=True,id=job['id'],seed=job['seed'],
        zero_estimate=summary['zero_estimate'],initial_draws=summary['initial_draws'],initial_hits=summary['initial_hits'],
        sampler_cpu_seconds=summary['sampler_cpu_seconds'],classifier_cpu_seconds=time.process_time()-started,
        classifier_calls=observer.classifier_calls,contact_calls=observer.contact_calls,
        source_sha256=bindings,label_sha256=sha(label_path),scope=SCOPE,
        initial_Q0_available=False,initial_labels_scope='Only stage-zero retained ancestors, not all unconditional attempts.',
        initial_attempt_audit_reused=True,density_audit_replayed=False,new_poses=0,new_clouds=0)
    write(destination/'analysis.json',result)
    return result


def smc_view(populations, physical):
    """A minimal arithmetic adapter; never relabel the guided protocol as legacy."""
    a = physical['allocation']
    return dict(populations=populations,protocol_snapshot=dict(jobs=physical['jobs'],allocation=dict(
        independent_populations=a['populations'],particles_each=a['population'],
        unconditional_initialization_draws_each=a['initial_draws'])))


def summarize_rows(rows):
    require(len(rows) == 4,'Four independent population estimates required')
    return {name:mass_statistics([row[name] for row in rows]) for name in old.CLASSES}


def compare_rows(left, right):
    a,b = summarize_rows(left),summarize_rows(right)
    return dict(left=a,right=b,comparisons={name:mass_comparison(a[name],b[name]) for name in old.CLASSES})


def compare(populations, physical, historical, iid):
    require(len(populations) == 4 and len({p['seed'] for p in populations}) == 4, 'Missing/repeated new population')
    view = smc_view(populations,physical); current = smc_rows(view,'Qz')
    validate_strata(view,dict(strata_definition=STRATA_DEFINITION,arms=iid['arms']))
    refs = {'historical_broad':lambda family=None,index=None:smc_rows(historical,'Qz',family,index)}
    for arm in ('baseline','conditioned'):
        refs['larger_iid_'+arm] = lambda family=None,index=None,arm=arm:importance_rows(iid['arms'][arm],'Qz',family,index)
    comparisons = {}; failures = []
    for label,rows in refs.items():
        result = compare_rows(current,rows()); result['strata'] = []
        for family,size in old.STRATA.items():
            for index in range(size):
                item = compare_rows(smc_rows(view,'Qz',family,index),rows(family,index))
                for region in old.CLASSES:
                    fractions = [None if whole[region]['log_Q'] is None else 0. if part[region]['log_Q'] is None
                        else math.exp(part[region]['log_Q']-whole[region]['log_Q'])
                        for whole,part in [(result['left'],item['left']),(result['right'],item['right'])]]
                    value = dict(family=family,bin=index,region=region,observed_parent_fractions=fractions,
                        material=max(v or 0. for v in fractions)>=.01,comparison=item['comparisons'][region],
                        new_population_estimates=item['left'][region],reference_population_estimates=item['right'][region])
                    result['strata'].append(value)
                    if value['material'] and not value['comparison']['passed']:failures.append(dict(reference=label,**value))
        comparisons[label] = result
    return dict(summary=old.population_statistics(current),comparisons=comparisons,failed_material_strata=failures,
        primary_regions=physical['analysis']['primary_regions'],initial_physical_Q0='Not classified; initial Hg cannot replace physical hard-only masses.',
        full_vessel_gate_open=False,assembly_gate_open=False,scope=SCOPE)


def run(out,expected):
    out = Path(out).resolve(); plan,physical,ledger = validate_preparation(out,expected)
    require(Path(__file__).resolve() == out/'source'/Path(__file__).name,'Use frozen analysis source')
    for module in tuple(sys.modules.values()):
        path = getattr(module,'__file__',None)
        if path and 'source/'+Path(path).name in plan['files_sha256']:
            require(Path(path).resolve() == out/'source'/Path(path).name,'Unfrozen analysis import')
    campaign = Path(plan['campaign']); state = read(ledger.bind(campaign/'status.json'))
    require_audited_status(state,plan['campaign_plan_sha256'])
    require(not (out/'status.json').exists(),'No retry/reclassification of a claimed analysis')
    counts = control.capacity()
    require(not counts['private_pid_namespace'] and counts['workers'] <= 29,'Host capacity unavailable for two classifiers and controller')
    # Authentication reuses receipts; it never executes a previous density audit.
    receipts = {job['id']:control.check_output(campaign,physical,job,True) for job in physical['jobs']}
    status = dict(complete=False,phase='classification',plan_sha256=expected,pid=os.getpid(),jobs=[],started=time.time())
    with (out/'status.json').open('x') as stream:json.dump(status,stream)
    def snapshot():
        write(out/'status.tmp',status); (out/'status.tmp').replace(out/'status.json')
    try:
        context = read(out/'context.json'); populations = []
        with ProcessPoolExecutor(max_workers=2) as pool:
            # Two bounded batches avoid enqueuing later classifications before
            # the current pair succeeds. Every started peer drains on failure.
            for start in range(0,4,2):
                counts = control.capacity()
                require(not counts['private_pid_namespace'] and counts['workers'] <= 30,'Classifier capacity changed')
                pending = {}; error = None
                for job in physical['jobs'][start:start+2]:
                    record = dict(id=job['id'],complete=False,phase='started')
                    status['jobs'].append(record);snapshot()
                    future = pool.submit(classify_population,context,job,receipts[job['id']],out/'populations'/job['id'],
                        plan['analysis']['profile_stages'])
                    pending[future] = record
                for future in as_completed(pending):
                    record = pending[future]
                    try:
                        result = future.result();populations.append(result)
                        record.update(complete=True,phase='complete',sha256=sha(out/'populations'/record['id']/'analysis.json'))
                    except BaseException as exc:
                        record.update(phase='failed',error=repr(exc));error = error or exc
                    snapshot()
                if error is not None:raise error
        populations.sort(key=lambda p:p['id'])
        status['phase'] = 'comparison';snapshot()
        result = compare(populations,physical,read(out/'historical.json'),read(out/'iid.json'))
        ledger.recheck()
        result.update(schema='hard-free-protein-smc-contact-analysis-v1',complete=True,populations=populations,
            input_sha256=ledger.files,analysis_plan_sha256=expected,campaign_plan_sha256=plan['campaign_plan_sha256'],
            native_definition_sha256=plan['native_definition_sha256'])
        write(out/'analysis.json',result)
        status.update(complete=True,phase='complete',finished=time.time(),analysis_sha256=sha(out/'analysis.json'));snapshot()
    except BaseException as error:
        status.update(complete=False,phase=status['phase']+'_failed',error=repr(error),finished=time.time());snapshot();raise
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action',required=True)
    f = sub.add_parser('freeze'); f.add_argument('--campaign',type=Path,required=True); f.add_argument('--campaign-plan-sha256',required=True); f.add_argument('--out',type=Path,required=True)
    for action in ('validate','run'):
        p = sub.add_parser(action);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-plan-sha256',required=True)
    args = parser.parse_args()
    value = prepare(args.campaign,args.campaign_plan_sha256,args.out) if args.action == 'freeze' else (
        validate_preparation(args.out.resolve(),args.expected_plan_sha256)[0] if args.action == 'validate' else run(args.out,args.expected_plan_sha256))
    print(json.dumps({k:value[k] for k in ('prepared','launched','analysis_plan_sha256','phase') if k in value}))
