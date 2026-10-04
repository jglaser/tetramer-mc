"""Metadata-only campaign admission/materialization tests; no science runs."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_native_class_physical_campaign as prepare


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')
    return prepare.bound(path)


def limits():
    result = {name: dict(cpu_limit_seconds=60, wall_limit_seconds=120, address_space_limit_bytes=2**30)
              for name in prepare.PHASES}
    for name in ('labels', 'geometry'): result[name]['max_record_bytes'] = 2**20
    return result


def resources():
    return dict(maximum_workers=1, threads_per_worker=1, global_scientific_slots=8,
                global_scientific_threads=32, estimated_output_bytes=1000, minimum_free_bytes=2000)


def diagnostic(root, completed=18):
    """An ordered synthetic 18-job execution, with no sample/attempt files."""
    root = Path(root); jobs = []; success = []
    for ordinal in range(18):
        query = ordinal < 9; index = ordinal if query else ordinal-9
        ident = f'query{index}' if query else f'query{index}-audit'
        samples, probes = (0, 40) if index == 8 else (128, 0)
        path = root/('queries' if query else 'audits')/(ident+'.json')
        job = dict(id=ident, terminal=str(path), phase='query' if query else 'audit',
                   samples=samples, probes=probes, seed=index)
        jobs.append(job)
        value = dict(complete=True, passed=True, samples=samples, probes=probes,
                     manifest=dict(executable_sha256='a'*64, seed=index, samples=samples))
        ref = save(path, value)
        record = dict(id=ident, terminal=str(path), sha256=ref['sha256']); success.append(record)
        folder = root/'execution'/f'{ordinal:02}-{ident}'
        save(folder/'exit.json', dict(returncode=0, error=None, child_started=True, child_drained=True))
        save(folder/'success.json', record)
    plan = dict(schema='native-class-line-reviewed-execution-v1', ready=True, physical_clouds=0,
                executable_sha256='a'*64, jobs=jobs)
    plan_ref = save(root/'execution-plan.json', plan)
    done = dict(complete=True, passed=True, completed=success[:completed], plan_sha256=plan_ref['sha256'],
                fresh_draws=1024, saved_development_probes=40, new_physical_clouds=0)
    done_ref = save(root/'execution/summary.json', done)
    pop = dict(attempted=128, cpu_seconds={k: .1 for k in ('draw_cpu_seconds', 'density_cpu_seconds', 'observer_cpu_seconds')})
    summary = dict(schema='native-class-line-proposal-diagnostic-summary-v1', complete=True,
        execution_plan_sha256=plan_ref['sha256'], execution_summary_sha256=done_ref['sha256'], new_Poisson_clouds=0,
        full_vessel_gate_open=False, assembly_gate_open=False,
        arms={name: dict(populations=[copy.deepcopy(pop) for _ in range(4)]) for name in ('hard_free', 'class')},
        saved_development_probes=[dict(id=i) for i in range(40)])
    summary_ref = save(root/'proposal-summary.json', summary)
    return dict(proposal_plan=plan, proposal_execution=done, proposal_summary=summary,
        _refs=dict(proposal_plan=plan_ref, proposal_execution=done_ref, proposal_summary=summary_ref))


def reference_evidence():
    h, c = prepare.PRODUCERS['hard_free'], prepare.PRODUCERS['class']
    ref = dict(binary_sha256=h['executable'], source_bundle_sha256=h['source_bundle'], complete=True)
    moment = dict(passed=True, attempted_denominator=8192)
    job = dict(samples=8192, manifest=dict(schema=h['schema']), **{key: dict(moment) for key in ('hard', 'depletion', 'latent_volume')})
    allocation_ref = dict(path='/synthetic/allocation.json', sha256='b'*64)
    binding_ref = dict(path='/synthetic/binding.json', sha256='c'*64)
    class_common = dict(executable_sha256=c['executable'], allocation_sha256=allocation_ref['sha256'])
    return dict(
        hard_prerequisites=dict(ref, schema='hard-free-line-physical-prerequisites-v1', all_checks_passed=True, fresh_proposal_validated=True),
        hard_reference=dict(ref, reference=dict(complete=True, primary_attempts=16384, jobs=[copy.deepcopy(job), copy.deepcopy(job)])),
        hard_independent=dict(complete=True, total_audited_rows=16704, fresh_independent_reference_draws=16384),
        class_rust=dict(complete=True, passed=True, test_count=94, executable_sha256=c['executable'], source_bundle_sha256=c['source_bundle']),
        class_sphere_allocation=dict(total_attempts=32768), class_sphere_binding=dict(class_common),
        class_sphere_execution=dict(class_common, complete=True, passed=True, failure=None, retries=0,
            remaining_jobs_not_started=0, completed=[{} for _ in range(32)], binding_sha256=binding_ref['sha256']),
        class_sphere_analytic=dict(class_common, complete=True, passed=True, attempts=32768, populations=[{} for _ in range(16)], failed_tests=[]),
        sphere_algebra=dict(complete=True, passed=True, all_rows_algebra=32768, populations=[{} for _ in range(16)], retries=0),
        bridge_validation=dict(complete=True, passed=True, returncode=0, source_before={}, source_after={}),
        bridge_review=dict(complete=True, passed=True),
        _refs=dict(class_sphere_allocation=allocation_ref, class_sphere_binding=binding_ref))


def materialization_context(base, scope='primary_only'):
    """Already-admitted toy metadata for testing the independent copy/plan layer."""
    base = Path(base); bindings = prepare.statistics.Bindings()
    def put(name, value):
        ref = save(base/name, value); bindings.reference(ref); return ref
    config = dict(shape=str(base/'shape.json'), capture_radius=170., depletant_radius=1.5, reservoir_density=.035)
    shape = put('shape.json', dict(synthetic_shape=True))
    region = put('region.json', dict(synthetic_region=True)); old = put('old.json', dict(synthetic_old=True))
    definition = dict(input_sha256={}); definition_ref = put('native/definition.json', definition)
    compiled = put('compiled.json', dict(synthetic_compiled=True))
    config_ref = put('config.json', config)
    history = dict(complete=True, policy='retention_only', entries=[dict(family='orthant', bin=22, region='competing')])
    history_ref = put('history.json', history)
    proof = put('proof.json', dict(complete=True, synthetic=True))
    guide = dict(schema='defensive-hard-free-line-guide-v1', defensive_uniform_shell_probability=.5,
                 conditional_probability=1., gaussian_components=[{}]*92, raw_translation_axes=[0,1,2], minimum_conditional_mass=1e-12)
    guides = dict(hard_free=guide, **{'class': dict(guide, schema='defensive-native-class-line-guide-v1',
        class_channels=prepare.CHANNELS, compiled_native=compiled)})
    producers = {}
    for name in ('hard_free', 'class'):
        archive = base/name/'rust-source'; files = {}
        for filename in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'vendor/README.md', 'src/lib.rs'):
            path = archive/filename; path.parent.mkdir(parents=True, exist_ok=True)
            text = '// synthetic '+name+'/'+filename+'\n'; path.write_text(text); bindings.bind(path)
            files[filename] = dict(text=text, sha256=prepare.sha(path))
        bundle_path = base/name/'source-bundle.json'; bundle_ref = put(name+'/source-bundle.json', dict(schema=1, files=files))
        executable = base/name/'latent-region-normalizer'; executable.write_bytes(b'NEVER EXECUTE\n'+bundle_path.read_bytes()); executable.chmod(0o700)
        bindings.bind(executable)
        producers[name] = dict(input=dict(executable=prepare.bound(executable), source_bundle=bundle_ref,
            source_archive=str(archive)), bundle=prepare.read(bundle_path))
    source = prepare._sources()
    for path in source.values(): bindings.bind(path)
    runtime = prepare.streaming.runtime_identity()
    for path, digest in runtime['file_sha256'].items(): bindings.bind(path, digest)
    inputs = dict(schema=prepare.INPUT_SCHEMA, study_scope=scope, seed_namespace='synthetic-never-run',
        phase_limits=limits(), resources=resources(), target=dict(region=region, old_r5_region=old, native_definition=definition_ref),
        config=config_ref, shape=shape, compiled_native=compiled, historical_failed_strata=history_ref, prerequisites={'toy': proof})
    inputs_ref = put('inputs.json', inputs); review_ref = put('review.json', dict(synthetic=True))
    return dict(inputs=inputs, review={}, bindings=bindings, region={}, target_id='d'*64, descriptor={}, config=config,
        definition=definition, definition_path=Path(definition_ref['path']), witness=dict(synthetic=True),
        guides=guides, producers=producers, history=history, source=source, runtime=runtime,
        arms=prepare.arm_design(scope), inputs_path=Path(inputs_ref['path']), review_path=Path(review_ref['path']))


def candidate_fixture(base):
    """Seventeen synthetic terminals and a real archived-helper lifecycle join."""
    old_root, root = base/'original', base/'recovery'
    shared = save(base/'common.json', dict(synthetic=True))
    common = {key:shared for key in ('guide','config','shape','region','compiled_native','definition')}
    populations = [dict(id=f'r{i:02}', samples=128, seed=10+i, audit_seed=20+i,
        selected_ids=list(range(16)), directory=str(root/'queries'/f'r{i:02}')) for i in range(4)]
    old = dict(common, populations=[dict(p,directory=str(old_root/'queries'/p['id'])) for p in populations])
    old_ref = save(old_root/'protocol.json', old)
    driver = old_root/'code/run_native_class_physical_campaign.py'; driver.parent.mkdir(parents=True)
    driver.write_bytes(Path(prepare.admission.runner.__file__).read_bytes())
    python = str(Path(sys.executable).absolute())
    base_files = {r['path']:r['sha256'] for r in (shared,prepare.bound(driver),prepare.bound(Path(python).resolve()))}

    def job(root, identity, pop, phase):
        path = root/'queries'/pop/'summary.json' if phase == 'producer' else root/'analysis'/pop/(phase+'.json')
        if phase == 'statistics': path=root/'analysis/statistics.json'
        return dict(id=identity, population=pop, phase=phase, argv=[python,'-B','synthetic-never-execute',identity],
            terminal=dict(path=str(path),success_contract='complete' if phase in ('producer','statistics') else 'complete_and_passed'),
            cpu_limit_seconds=1,wall_limit_seconds=1,address_space_limit_bytes=1024)
    old_jobs = [job(old_root,f'r{i:02}-{phase}',f'r{i:02}',phase) for i in range(4)
                for phase in ('producer','algebra','labels','geometry')]+[job(old_root,'statistics','all','statistics')]
    old_plan = dict(schema=prepare.admission.EXECUTION_SCHEMA,root=str(old_root),maximum_workers=1,threads=1,
        files=base_files,jobs=old_jobs,executable_resolutions={python:str(Path(python).resolve())})
    old_plan_ref = save(old_root/'execution-plan.json',old_plan)

    def completed(root, plan_ref, jobs, terminals, files):
        records=[]
        for index,(entry,ref) in enumerate(zip(jobs,terminals)):
            record=dict(**{k:entry[k] for k in ('id','population','phase','argv')},terminal=ref,
                success_contract=entry['terminal']['success_contract'],child_started=True,child_drained=True,returncode=0,
                success=True,error=None,timeout=False,retries=0,replacements=0,pid=100+index,birth_ticks=1000+index)
            folder=prepare.admission.runner.job_directory(root,index,entry)
            save(folder/'success.json',record);save(folder/'exit.json',record);save(folder/'attempt.json',dict(job=entry))
            save(folder/'process.json',{k:record[k] for k in ('id','pid','birth_ticks','argv')});records.append(record)
        save(root/'execution/claim.json',dict(schema=prepare.admission.EXECUTION_SCHEMA,plan_sha256=plan_ref['sha256'],
            maximum_workers=1,threads=1,retries=0,replacements=0))
        return dict(schema=prepare.admission.EXECUTION_SCHEMA,complete=True,passed=True,plan_sha256=plan_ref['sha256'],
            files=files,completed=records,active=None,unstarted=[],failure=None,retries=0,replacements=0,maximum_workers=1,threads=1)
    summary = dict(complete=True,samples=128)
    old_summary=save(old_root/'queries/r00/summary.json',summary)
    adopted=save(root/'queries/r00/summary.json',summary)
    old_status=completed(old_root,old_plan_ref,old_jobs[:1],[old_summary],base_files)
    failure=save(old_root/'execution/failure.json',dict(plan_sha256=old_plan_ref['sha256'],complete=False,
        failed_job=dict(ordinal=1,id='r00-algebra',population='r00',phase='algebra'),
        completed=old_status['completed'],unstarted=old_jobs[2:]))
    algebra_failure=save(old_root/'analysis/r00/algebra.failure.json',dict(active_id=0,completed_ids=[],counts={},setup_counts={},
        error='Wrong proposal density/contact format'))
    inventory={'summary.json':dict(sha256=old_summary['sha256'],bytes=Path(old_summary['path']).stat().st_size)}
    recovery=dict(schema='native-class-support-format-recovery-v1',original_root=str(old_root),original_protocol=old_ref,
        original_plan=old_plan_ref,original_failure=failure,original_algebra_failure=algebra_failure,
        retained_summary=old_summary,retained_files=inventory,source_directory=str(old_root/'queries/r00'),
        retained_attempts=128,new_draws=384,total_attempts=512,scientific_replacements=0,no_outcome_adaptation=True,allocation_changed=False)
    source={'synthetic.py':shared['sha256']}; runtime=dict(synthetic=True)
    protocol=dict(common,schema='native-class-support-pilot-v1',root=str(root),populations=populations,
        total_attempts=512,new_draws=384,retained_attempts=128,new_Poisson_clouds=0,physical_weight_estimates=0,
        selected_geometry_rows=64,recovery=recovery,source_sha256=source,runtime=runtime,
        binary64_weight_retention=dict(original_component_retention_min=.75))
    protocol_ref=save(root/'protocol.json',protocol)
    jobs=[job(root,f'r{i:02}-{phase}',f'r{i:02}',phase) for i in range(4)
          for phase in ('producer','algebra','labels','geometry')]+[job(root,'statistics','all','statistics')]
    files=dict(base_files); files[protocol_ref['path']]=protocol_ref['sha256']
    plan=dict(schema=prepare.admission.EXECUTION_SCHEMA,root=str(root),maximum_workers=1,threads=1,files=files,jobs=jobs)
    plan_ref=save(root/'execution-plan.json',plan)
    refs={}; terminals=[]
    for entry in jobs[:-1]:
        pop,phase=entry['population'],entry['phase']
        if phase=='producer': value=summary
        else:
            ids=range(16 if phase=='geometry' else 128)
            value=dict(schema='native-class-support-pilot-'+phase+'-v1',complete=True,passed=True,phase=phase,population=pop,
                protocol_sha256=protocol_ref['sha256'],attempted_records=128,
                rows=[dict(id=i,sample_record_sha256=f'{i:064x}') for i in ids],source_sha256=source,runtime=runtime,
                input_sha256={shared['path']:shared['sha256']},journal=shared,new_pose_draws=0,new_Poisson_clouds=0,physical_weight_estimates=0)
        ref=save(entry['terminal']['path'],value);terminals.append(ref)
        if phase!='producer':refs[entry['id']]=ref
    reports=[dict(id=p['id'],attempted=128,selected_reference_rows=16,
        critical_endpoints=dict(competing22=1,competing62=1,native55=1),
        target_line_hit_rates={f'channel:{c}':dict(selected=2,hits=1) for c in (2,3,4)}) for p in populations]
    combined=dict(attempted_denominator=512,selected_reference_rows=64,critical_endpoints=dict(competing22=4,competing62=4,native55=4),
        selected_target_line_access={str(c):dict(selected=8,hits=4) for c in (2,3,4)},
        access_screen='access_observed_requires_review',physical_campaign_gate_open=False,full_vessel_gate_open=False,
        assembly_gate_open=False,training_critical_points=8,nontraining_critical_holdouts=0)
    statistics=save(root/'analysis/statistics.json',dict(schema='native-class-support-pilot-summary-v1',complete=True,passed=True,
        protocol_sha256=protocol_ref['sha256'],new_Poisson_clouds=0,raw_rows_read=0,new_geometry_queries=0,retries=0,
        populations=reports,combined=combined,predecessor_receipts=refs,input_sha256={shared['path']:shared['sha256']}))
    terminals.append(statistics)
    status=save(root/'execution/summary.json',completed(root,plan_ref,jobs,terminals,files))
    adoption=save(root/'analysis/r00/adoption.json',dict(schema=recovery['schema'],complete=True,passed=True,
        protocol_sha256=protocol_ref['sha256'],new_draws=0,retained_attempts=128,sample_rows_parsed=0,retries=0,
        retained_files=inventory,original_summary=old_summary,adopted_summary=adopted,
        input_sha256={shared['path']:shared['sha256']},journal=shared))
    inputs=dict(guides={'class':shared},target=dict(region=shared,native_definition=shared),shape=shared,compiled_native=shared,
        candidate_pilot=dict(protocol=protocol_ref,execution_plan=plan_ref,execution_summary=status,statistics=statistics,adoption=adoption))
    review=dict(class_guide_profile=prepare.REVISED_PROFILE,cost_review=dict(candidate_statistics_sha256=statistics['sha256']))
    pins=dict(guide=shared['sha256'],protocol=protocol_ref['sha256'],execution_plan=plan_ref['sha256'],
        original_plan=old_plan_ref['sha256'],original_failure=failure['sha256'],original_algebra_failure=algebra_failure['sha256'],
        retained_summary=old_summary['sha256'])
    return inputs,review,pins


class PreparationTests(unittest.TestCase):
    def test_candidate_profile_requires_exact_retained_and_added_design(self):
        original=[dict(weight=1/92,mean=[float(i)]*6,covariance=[[float(a==b) for b in range(6)] for a in range(6)]) for i in range(92)]
        common=dict(region_sha256='a'*64,raw_translation_axes=[0,1,2],conditional_probability=1.,
                    minimum_conditional_mass=1e-12,defensive_uniform_shell_probability=.5,gaussian_components=original)
        guides=dict(hard_free=dict(common,schema='defensive-hard-free-line-guide-v1'),
            **{'class':dict(common,schema='defensive-native-class-line-guide-v1',class_channels=prepare.CHANNELS,
                           compiled_native=dict(sha256=prepare.COMPILED_SHA))})
        inputs=dict(guides={k:dict(sha256=v) for k,v in prepare.GUIDES.items()},target=dict(region=dict(sha256='a'*64)))
        self.assertEqual(prepare._guide_profile(inputs,guides),prepare.ORIGINAL_PROFILE)
        components=copy.deepcopy(original)
        for c in components:c['weight']=.75*(c['weight']/math.fsum(v['weight'] for v in original))
        components += [dict(weight=.25/24,mean=[float(i)]*6,
            covariance=[[width**2 if a==b else 0. for b in range(6)] for a in range(6)]) for i in range(8) for width in (.05,.15,.45)]
        guides['class']['gaussian_components']=components
        with self.assertRaisesRegex(ValueError,'Changed Gaussian'):prepare._guide_profile(inputs,guides)
        inputs.update(class_guide_profile=prepare.REVISED_PROFILE);inputs['guides']['class']['sha256']=prepare.REVISED_PINS['guide']
        self.assertEqual(prepare._guide_profile(inputs,guides),prepare.REVISED_PROFILE)
        for index,field in [(0,'weight'),(0,'mean'),(94,'covariance'),(115,'weight')]:
            bad=copy.deepcopy(guides);bad['class']['gaussian_components'][index][field]=0
            with self.subTest(index=index,field=field),self.assertRaises(ValueError):prepare._guide_profile(inputs,bad)

    def test_completed_candidate_authenticates_all_terminals_and_retained_bytes(self):
        for change in (None,'missing_job','process','row_hash','adoption','raw_copy'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                base=Path(d);inputs,review,pins=candidate_fixture(base);refs=inputs['candidate_pilot']
                if change=='missing_job':
                    value=prepare.read(refs['execution_summary']['path']);value['completed'].pop()
                    refs['execution_summary']=save(refs['execution_summary']['path'],value)
                if change=='process':
                    path=base/'recovery/execution/jobs/005-r01-algebra/process.json';value=prepare.read(path);value['birth_ticks']+=1;save(path,value)
                if change=='row_hash':
                    path=base/'recovery/analysis/r01/labels.json';value=prepare.read(path);value['rows'][0]['sample_record_sha256']='f'*64;save(path,value)
                if change=='adoption':
                    value=prepare.read(refs['adoption']['path']);value['retained_attempts']=127;refs['adoption']=save(refs['adoption']['path'],value)
                if change=='raw_copy':save(base/'recovery/queries/r00/summary.json',dict(complete=True,samples=127))
                with patch.dict(prepare.REVISED_PINS,pins,clear=True),patch.object(prepare.subprocess,'Popen',side_effect=AssertionError('science')):
                    if change:
                        with self.assertRaises(ValueError):prepare._candidate_evidence(inputs,review,prepare.statistics.Bindings())
                    else:
                        result=prepare._candidate_evidence(inputs,review,prepare.statistics.Bindings())
                        self.assertEqual(result['attempted_denominator'],512);self.assertEqual(result['selected_reference_rows'],64)
                        self.assertFalse(result['physical_campaign_gate_open']);self.assertFalse(result['regional_convergence_established'])

    def test_candidate_access_requires_each_endpoint_and_line_without_promoting_precision(self):
        with tempfile.TemporaryDirectory() as d:
            inputs,_,pins=candidate_fixture(Path(d));summary=prepare.read(inputs['candidate_pilot']['statistics']['path'])
            with patch.dict(prepare.REVISED_PINS,pins,clear=True):
                prepare._candidate_access(summary)
                for region in ('competing22','competing62','native55'):
                    bad=copy.deepcopy(summary)
                    for p in bad['populations']:p['critical_endpoints'][region]=0
                    bad['combined']['critical_endpoints'][region]=0
                    with self.subTest(region=region),self.assertRaisesRegex(ValueError,'endpoint access'):prepare._candidate_access(bad)
                for channel in (2,3,4):
                    bad=copy.deepcopy(summary)
                    for p in bad['populations']:p['target_line_hit_rates'][f'channel:{channel}']['hits']=0
                    bad['combined']['selected_target_line_access'][str(channel)]['hits']=0
                    with self.subTest(channel=channel),self.assertRaisesRegex(ValueError,'remain empty'):prepare._candidate_access(bad)
                for key,value in [('attempted_denominator',511),('physical_campaign_gate_open',True),('nontraining_critical_holdouts',1)]:
                    bad=copy.deepcopy(summary);bad['combined'][key]=value
                    with self.subTest(key=key),self.assertRaises(ValueError):prepare._candidate_access(bad)

    def test_historical_inventory_requires_complete_retention_only_and_binds_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); target_id = 'd'*64
            history = dict(schema='native-class-historical-failed-strata-v1', complete=True, policy='retention_only',
                target_and_regions_sha256=target_id,
                entries=[dict(family='orthant', bin=index, region=region) for index, region in
                         [(22, 'competing'), (62, 'competing'), (55, 'native'), (55, 'native_remainder')]],
                source_receipts=[save(base/'source.json', dict(synthetic=True))],
                derived_from=save(base/'original.json', dict(synthetic=True)),
                provenance_review=save(base/'review.json', dict(synthetic=True)))
            bindings = prepare.statistics.Bindings()
            prepare._historical_evidence(history, target_id, bindings)
            self.assertEqual(set(bindings.files), {str(base/name) for name in ('source.json', 'original.json', 'review.json')})
            for field, values, message in [('complete', [None, False, 1, 'true'], 'must be complete'),
                                            ('policy', [None, 'gate', 'retention'], 'retention_only')]:
                for value in values:
                    with self.subTest(field=field, value=value):
                        bad = copy.deepcopy(history)
                        if value is None: bad.pop(field)
                        else: bad[field] = value
                        with self.assertRaisesRegex(ValueError, message):
                            prepare._historical_evidence(bad, target_id, prepare.statistics.Bindings())
            save(base/'review.json', dict(synthetic='changed'))
            with self.assertRaises(ValueError):
                prepare._historical_evidence(history, target_id, prepare.statistics.Bindings())

    def test_original_eighteen_jobs_and_one_time_summary_are_required(self):
        for count in (0, 14, 17, 18):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as d:
                evidence = diagnostic(Path(d)/'pilot', count); bindings = prepare.statistics.Bindings()
                if count < 18:
                    with self.assertRaisesRegex(ValueError, 'All 18'): prepare._proposal_evidence(evidence, bindings)
                else:
                    prepare._proposal_evidence(evidence, bindings)
                    self.assertEqual(len(bindings.files), 54)
        for mode in ('exit', 'order', 'summary', 'allocation'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                evidence = diagnostic(Path(d)/'pilot'); root = Path(d)/'pilot'
                if mode == 'exit': save(root/'execution/17-query8-audit/exit.json', dict(returncode=1, error=None, child_started=True, child_drained=True))
                if mode == 'order': evidence['proposal_execution']['completed'].reverse()
                if mode == 'summary': evidence['proposal_summary']['execution_summary_sha256'] = '0'*64
                if mode == 'allocation': evidence['proposal_summary']['arms']['class']['populations'][0]['attempted'] = 127
                with self.assertRaises(ValueError): prepare._proposal_evidence(evidence, prepare.statistics.Bindings())

    def test_reference_counts_binary_links_and_failed_checks_cannot_be_opaque_passes(self):
        evidence = reference_evidence(); prepare._reference_evidence(evidence)
        mutations = [lambda e: e['hard_reference'].update(binary_sha256='0'*64),
            lambda e: e['hard_reference']['reference']['jobs'][1]['depletion'].update(passed=False),
            lambda e: e['hard_independent'].update(total_audited_rows=1),
            lambda e: e['class_sphere_execution']['completed'].pop(),
            lambda e: e['class_sphere_analytic'].update(failed_tests=['failed']),
            lambda e: e['class_sphere_binding'].update(executable_sha256='0'*64),
            lambda e: e['sphere_algebra'].update(all_rows_algebra=32767),
            lambda e: e['bridge_validation'].update(source_after={'changed': '0'*64})]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                bad = copy.deepcopy(evidence); mutate(bad)
                with self.assertRaises(ValueError): prepare._reference_evidence(bad)

    def test_incomplete_real_admission_stops_before_creating_output(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); evidence = diagnostic(base/'pilot', 14); refs = evidence['_refs']
            for name in prepare.PINS:
                if name not in refs: refs[name] = save(base/(name+'.json'), dict(complete=True, passed=True))
            inputs = dict(schema=prepare.INPUT_SCHEMA, study_scope='primary_only', prerequisites=refs,
                          resources=resources(), phase_limits=limits())
            inp = save(base/'inputs.json', inputs)
            review = dict(schema=prepare.REVIEW_SCHEMA, complete=True, passed=True, inputs=inp, study_scope='primary_only',
                stage_materialization_policy=prepare.POLICY, deterministic_materialization_authorized=True,
                historical_inventory_complete=True, physical_gates_remain_closed=True, resources=resources(), phase_limits=limits())
            save(base/'review.json', review)
            with patch.dict(prepare.PINS, {name: refs[name]['sha256'] for name in prepare.PINS}, clear=True):
                with self.assertRaisesRegex(ValueError, 'All 18'): prepare.prepare(base/'campaign', base/'inputs.json', base/'review.json')
            self.assertFalse((base/'campaign').exists())

    def test_primary_and_full_exact_inventory_and_predeclared_ids(self):
        for scope, populations, attempts in [('primary_only', 16, 262144), ('full_declared_campaign', 40, 1048576)]:
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as d:
                base = Path(d); context = materialization_context(base/'source', scope); out = base/'campaign'
                with patch.object(prepare, 'validate', return_value=context), patch.object(prepare, 'seed_inventory', return_value={'seeds': []}), \
                     patch.object(prepare.labels, 'observer_setup_inventory', return_value={'synthetic_metadata_only': True}), \
                     patch.object(prepare.labels, 'load_bounded_classifier', side_effect=AssertionError('geometry')), \
                     patch.object(prepare.subprocess, 'Popen', side_effect=AssertionError('scientific launch')):
                    result = prepare.prepare(out, context['inputs_path'], context['review_path'])
                protocol = prepare.read(out/'protocol.json'); plan = prepare.read(out/'execution-plan.json')
                self.assertTrue(result['complete']); self.assertFalse(result['launched'])
                self.assertEqual(result['populations'], populations); self.assertEqual(protocol['total_attempts'], attempts)
                self.assertEqual(protocol['maximum_clouds'], 2*attempts); self.assertEqual(len(plan['jobs']), populations*5+1)
                self.assertEqual(protocol['selected_geometry_max_rows'], populations*20)
                self.assertEqual(len(protocol['comparisons']), 1 if scope == 'primary_only' else 4)
                self.assertEqual(protocol['historical_failed_strata_policy'], 'retention_only')
                self.assertEqual(protocol['historical_failed_strata']['sha256'], context['inputs']['historical_failed_strata']['sha256'])
                self.assertEqual(prepare.read(protocol['historical_failed_strata']['path']), context['history'])
                self.assertEqual(protocol['gates'], prepare.statistics.GATES)
                self.assertNotIn('class_guide_profile',protocol)
                self.assertNotIn('candidate_pilot',protocol)
                for comparison in protocol['comparisons']:
                    self.assertEqual(comparison['historical_failed_strata'], context['history']['entries'])
                seeds = []
                for arm in protocol['arms']:
                    for pop in arm['populations']:
                        seeds += [pop['seed'], pop['audit_seed']]
                        declaration = prepare.read(pop['preselection']['path'])
                        self.assertEqual(declaration, prepare.selected.preselection(pop['id'], arm['samples'], pop['audit_seed']))
                        self.assertEqual(len(declaration['draw_ids']), 16); self.assertFalse(Path(pop['directory']).exists())
                self.assertEqual(len(set(seeds)), populations*2)
                self.assertEqual(plan['jobs'][-1]['phase'], 'statistics'); self.assertEqual(plan['jobs'][-1]['terminal']['success_contract'], 'complete')
                self.assertNotIn('admission', [j['phase'] for j in plan['jobs']])
                for job in plan['jobs']:
                    if job['phase'] == 'selection': self.assertTrue(job['terminal']['path'].endswith('selection-review.json'))
                    if job['phase'] == 'algebra': self.assertIn('--out', job['argv'])
                    if job['phase'] != 'producer': self.assertEqual(job['argv'][0], str(Path(sys.executable).absolute()))
                self.assertEqual(plan['executable_resolutions'][protocol['python']], str(Path(sys.executable).resolve()))
                self.assertEqual(result['execution_plan'], prepare.bound(out/'execution-plan.json'))
                self.assertEqual(plan['preparation_receipt'], str(out/'preparation.json'))
                self.assertNotIn(str(out/'protocol.json'), protocol['files'])
                self.assertFalse((out/'execution').exists())
                with self.assertRaisesRegex(ValueError, 'Fresh campaign'): prepare.prepare(out, context['inputs_path'], context['review_path'])

    def test_revised_profile_materializes_evidence_without_changing_physical_gates(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);context=materialization_context(base/'source');out=base/'campaign'
            candidate=save(base/'candidate.json',dict(synthetic=True,complete=True))
            context['bindings'].reference(candidate)
            context['inputs'].update(class_guide_profile=prepare.REVISED_PROFILE,guides={'class':candidate})
            context['candidate']=dict(profile=prepare.REVISED_PROFILE,evidence={k:candidate for k in
                ('protocol','execution_plan','execution_summary','statistics','adoption')},
                attempted_denominator=512,selected_reference_rows=64,physical_campaign_gate_open=False,
                regional_convergence_established=False,access_screen='access_observed_requires_review')
            with patch.object(prepare,'validate',return_value=context),patch.object(prepare,'seed_inventory',return_value={'seeds':[]}), \
                 patch.object(prepare.labels,'observer_setup_inventory',return_value={'synthetic_metadata_only':True}), \
                 patch.object(prepare.subprocess,'Popen',side_effect=AssertionError('scientific launch')):
                prepare.prepare(out,context['inputs_path'],context['review_path'])
            protocol=prepare.read(out/'protocol.json')
            self.assertEqual(protocol['class_guide_profile'],prepare.REVISED_PROFILE)
            self.assertEqual(protocol['class_guide_source'],candidate)
            self.assertEqual(protocol['total_attempts'],262144)
            self.assertEqual(protocol['gates'],prepare.statistics.GATES)
            self.assertEqual(protocol['selected_geometry_max_rows'],320)
            self.assertFalse(protocol['physical_campaign_gate_open'])
            self.assertFalse(protocol['candidate_pilot']['regional_convergence_established'])
            for ref in protocol['candidate_pilot']['evidence'].values():
                self.assertEqual(ref['sha256'],candidate['sha256'])
                self.assertEqual(protocol['files'][ref['path']],ref['sha256'])

    def test_embedded_source_archive_validation_is_metadata_only(self):
        with tempfile.TemporaryDirectory() as d:
            context = materialization_context(Path(d)/'source'); value = context['producers']['hard_free']['input']
            with patch.object(prepare.subprocess, 'Popen', side_effect=AssertionError('execute')):
                prepare.verify_bundle(value['executable']['path'], value['source_bundle']['path'], value['source_archive'])
                (Path(value['source_archive'])/'src/lib.rs').write_text('changed source')
                with self.assertRaisesRegex(ValueError, 'Reviewed source differs'):
                    prepare.verify_bundle(value['executable']['path'], value['source_bundle']['path'], value['source_archive'])

    def test_copy_corruption_preserves_failed_preparation_without_plan(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); context = materialization_context(base/'source'); original = prepare.shutil.copy2
            def corrupt(source, destination):
                result = original(source, destination); Path(destination).write_bytes(b'corrupt'); return result
            out = base/'campaign'
            with patch.object(prepare, 'validate', return_value=context), patch.object(prepare, 'seed_inventory', return_value={'seeds': []}), \
                 patch.object(prepare.shutil, 'copy2', side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, 'Copied bytes differ'): prepare.prepare(out, context['inputs_path'], context['review_path'])
            self.assertTrue((out/'preparation-claim.json').exists())
            failure = prepare.read(out/'preparation-failure.json'); self.assertFalse(failure['complete']); self.assertFalse(failure['launched'])
            self.assertFalse((out/'execution-plan.json').exists()); self.assertFalse((out/'preparation.json').exists())

    def test_seed_collision_and_invalid_budgets_stop_before_materialization(self):
        for change in ('seed', 'budget'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                base = Path(d); context = materialization_context(base/'source'); out = base/'campaign'
                collision = prepare.stream_seed(context['inputs']['seed_namespace'], 'hard_free', 'r00', 'physical')
                if change == 'seed':
                    with patch.object(prepare, 'validate', return_value=context), patch.object(prepare, 'seed_inventory', return_value={'seeds': [collision]}):
                        with self.assertRaisesRegex(ValueError, 'Seed collision'): prepare.prepare(out, context['inputs_path'], context['review_path'])
                    self.assertFalse(out.exists())
                else:
                    bad = limits(); bad['labels']['max_record_bytes'] = True
                    with self.assertRaisesRegex(ValueError, 'positive integers'): prepare.validate_limits(bad, resources())
                    bad_resources = resources(); bad_resources['maximum_workers'] = True
                    with self.assertRaises(ValueError): prepare.validate_limits(limits(), bad_resources)


if __name__ == '__main__': unittest.main()
