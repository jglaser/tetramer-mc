"""Synthetic receipt joins only; no producer, row parser or geometry executes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import admit_native_class_physical_campaign as admission


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else (json.dumps(value,allow_nan=False)+'\n').encode())
    return ref(path)


def ref(path): return dict(path=str(path.resolve()),sha256=admission.streaming.sha(path))


def diagnostic_arm():
    primary=dict(quality={r:dict(passed=False) for r in admission.statistics.DECISIONS},
        row_diagnostics={r:dict(observed=False) for r in admission.statistics.DECISIONS},
        population_statistics=dict(free_energy_contrast=dict(observed=False,halfwidth_95=None)))
    bins={family:[dict(copy.deepcopy(primary),observed_parent_mass_fraction={r:None for r in admission.statistics.DECISIONS})
                  for _ in range(count)] for family,count in admission.statistics.BINS.items()}
    return dict(estimates={k:dict(primary=copy.deepcopy(primary),strata=copy.deepcopy(bins)) for k in ('Qz','Q0')})


def diagnostic_comparison(declaration):
    block=dict(regions={r:dict(observed=False,passed=False) for r in admission.statistics.REGIONS},
        free_energy_contrast=dict(observed=False,passed=False),
        strata=[dict(family=family,bin=index,region=region,material=False,historical_failure=False,
                     comparison=dict(observed=False,passed=False))
            for family,count in admission.statistics.BINS.items() for index in range(count)
            for region in admission.statistics.DECISIONS])
    return dict(left=declaration['left'],right=declaration['right'],
                kinds={k:copy.deepcopy(block) for k in ('Qz','Q0')})


def fixture(base, full=False):
    """Lifecycle/receipt stand-ins with real hashes and deliberately inert rows."""
    source=Path(admission.__file__).resolve(); source_map={str(source):admission.streaming.sha(source)}
    shape=write(base/'shape.json',dict(synthetic=True)); definition=write(base/'definition.json',dict(synthetic=True))
    region=write(base/'region.json',dict(shape_sha256=shape['sha256']))
    old=write(base/'old.json',dict(synthetic=True))
    strata=write(base/'strata.json',admission.statistics.STRATA)
    target=dict(region=region,old_r5_region=old,native_definition=definition); target_id='a'*64
    descriptor=dict(region_sha256=region['sha256'],shape_sha256=shape['sha256'],
        native_definition_sha256=definition['sha256'],old_r5_region_sha256=old['sha256'],
        strata_sha256=admission.statistics.fingerprint(admission.statistics.STRATA))
    protocol=dict(schema=admission.PROTOCOL_SCHEMA,study_scope='full_declared_campaign' if full else 'primary_only',
        target=target,target_and_regions_sha256=target_id,strata=admission.statistics.STRATA,
        gates=admission.statistics.GATES,arms=[],comparisons=[dict(left='hard_free',right='class',historical_failed_strata=[])])
    if full:
        protocol['comparisons'] += [dict(left='class',right=name,historical_failed_strata=[])
            for name in ('class_large','class_defensive','class_intensity')]
    stats_plan=dict(schema=admission.statistics.PLAN_SCHEMA,arms=[],**{k:copy.deepcopy(protocol[k]) for k in
        ('target','target_and_regions_sha256','strata','gates','comparisons')})
    report=dict(schema=admission.statistics.SCHEMA,complete=True,target_and_regions_sha256=target_id,
        target_descriptor=descriptor,gates=admission.statistics.GATES,arms={},
        comparisons=[diagnostic_comparison(c) for c in protocol['comparisons']],
        input_sha256=dict(source_map),source_sha256=dict(source_map),runtime=dict(synthetic=True),
        physical_campaign_gate_open=False,full_vessel_gate_open=False,assembly_gate_open=False)
    jobs=[];completed=[];slots=[];files=dict(source_map); objects={}
    setup=dict(reference_contact_calls=0,fixed_scaffold_classifier_calls=0,
               maximum_scaffold_contact_calls=0,maximum_total_contact_calls=0)

    def terminal(path,value,identity,population,phase,contract='complete_and_passed'):
        bound=write(path,value);job=dict(id=identity,population=population,phase=phase,argv=['synthetic',identity],
            terminal=dict(path=bound['path'],success_contract=contract),cpu_limit_seconds=1,
            wall_limit_seconds=1,address_space_limit_bytes=1024)
        jobs.append(job);completed.append(dict(**{k:job[k] for k in ('id','population','phase','argv')},
            terminal=bound,success_contract=contract,child_started=True,child_drained=True,returncode=0,
            success=True,error=None,timeout=False,retries=0,replacements=0,pid=1,birth_ticks=1))
        return bound

    specifications=admission.FULL if full else admission.PRIMARY
    for ai,(arm_id,(schema,n)) in enumerate(specifications.items()):
        arm=dict(id=arm_id,source_schema=schema,samples=n,populations=[])
        stat_arm=dict(id=arm_id,samples=n,populations=[]);records=[]
        for pi in range(8):
            pop_id=f'r{pi:02}';root=base/'populations'/f'{arm_id}-{pop_id}';root.mkdir(parents=True)
            seed=1000+ai*100+pi;audit_seed=10000+ai*100+pi
            pre=write(base/'preselection'/f'{arm_id}-{pop_id}.json',admission.selected.preselection(pop_id,n,audit_seed))
            files[pre['path']]=pre['sha256']
            population=dict(id=pop_id,seed=seed,directory=str(root),audit_seed=audit_seed,preselection=pre)
            arm['populations'].append(population)
            fixed=[dict(position=[0.,0.,0.],orientation=[1.,0.,0.,0.])]
            config=write(root/'provenance/input-config.json',dict(fixed_poses=fixed))
            region_copy=write(root/'provenance/region.json',Path(region['path']).read_bytes())
            manifest=dict(schema=schema,samples=n,seed=seed,shape_sha256=shape['sha256'],region_sha256=region['sha256'],
                config_sha256=config['sha256'],physical_fixed_neighbors=fixed)
            compiled=write(root/'external-compiled.json',dict(source_definition_sha256=definition['sha256'],
                source_input_sha256={'tetramer-shape.json':shape['sha256']},fixed_poses=fixed))
            witness=write(root/'external-witness.json',dict(compiled_sha256=compiled['sha256'],
                expected_shape_sha256=shape['sha256'],compatible=True))
            if schema==admission.V7:
                manifest['compiled_native']=dict(source_definition_sha256=definition['sha256'])
            manifest_ref=write(root/'manifest.json',manifest)
            samples=write(root/'samples.jsonl',b'{}\n'*n);attempts=write(root/'attempts.jsonl',b'{}\n'*n)
            counts=dict(attempted=n,contributing=n,capture_rejected=0,hard_rejected=0,region_rejected=0,shell_rejected=0)
            summary=dict(complete=True,manifest=manifest,samples=n,samples_sha256=samples['sha256'],
                attempts_sha256=attempts['sha256'],**{k:counts[k] for k in
                ('capture_rejected','hard_rejected','region_rejected','shell_rejected')})
            prefix=f'{arm_id}-{pop_id}'
            summary_ref=terminal(root/'summary.json',summary,prefix+'-producer',prefix,'producer','complete')
            inputs={r['path']:r['sha256'] for r in (config,region_copy,manifest_ref,samples,attempts,summary_ref,compiled,witness)}
            common=dict(complete=True,passed=True,samples=n,seed=seed,root=str(root),source_schema=schema,
                samples_sha256=samples['sha256'],attempts_sha256=attempts['sha256'],input_sha256=dict(inputs),
                source_sha256=dict(source_map),runtime=dict(synthetic=True))
            algebra=terminal(root/'algebra.json',dict(common,schema=admission.streaming.SCHEMA,
                all_rows_algebra=n,geometry_certified=False,physical_contact_labels_certified=False),prefix+'-algebra',prefix,'algebra')
            label_rows=write(root/'labels.jsonl',b'{}\n'*n)
            endpoint=dict(common,schema=admission.statistics.LABEL_SCHEMA,target_and_regions_sha256=target_id,
                manifest_sha256=manifest_ref['sha256'],labels=label_rows,
                physical_hard_validity_scope='every_attempt_capture_and_atomic',physical_native_contact_scope='every_contributing_pose',
                region_sha256=region['sha256'],reference_region_sha256=old['sha256'],shape_sha256=shape['sha256'],
                definition_sha256=definition['sha256'],strata_sha256=descriptor['strata_sha256'],counts=counts,
                query_counts=dict(capture=n,atomic=n,native=n,contact=n),
                native_identity_origin='external_analysis_plan' if schema==admission.V6 else 'producer_compiled_native',
                compiled_native_binding=compiled,shape_compatibility_binding=witness if schema==admission.V6 else None)
            labels=terminal(root/'labels.json',endpoint,prefix+'-labels',prefix,'labels')
            stat_slot=dict(id=pop_id,seed=seed,directory=str(root),manifest=manifest_ref,summary=summary_ref,algebra=algebra,labels=labels)
            stat_arm['populations'].append(stat_slot)
            records.append(dict(id=pop_id,seed=seed,source_schema=schema,draws=n,unconditional_denominator=n,
                algebra_receipt_sha256=algebra['sha256'],label_receipt_sha256=labels['sha256']))
            p=dict(id=pop_id,root=str(root),seed=seed,samples=n,manifest_sha256=manifest_ref['sha256'],
                summary_sha256=summary_ref['sha256'],samples_sha256=samples['sha256'],attempts_sha256=attempts['sha256'])
            declaration=admission.streaming.read(pre['path']); maxima={k:None for k in admission.statistics.DECISIONS}
            # One outcome-selected maximum outside the unconditional set.
            maximum=next(i for i in range(n) if i not in declaration['draw_ids'])
            maxima['native']=dict(draw=maximum,log_weight=1.)
            reasons=admission.selected.selected_inventory(declaration['draw_ids'],maxima)
            row_hash=hashlib.sha256(b'{}\n').hexdigest()
            rows=[dict(draw=i,reasons=reasons[i],**{name:dict(path=bound['path'],line=i+1,
                offset=3*i,bytes=3,sha256=row_hash) for name,bound in [('sample_record',samples),('label_record',label_rows)]})
                for i in sorted(reasons)]
            selection=write(root/'selection.json',dict(schema=admission.selected.SELECTION_SCHEMA,source_schema=schema,
                population=p,target_and_regions_sha256=target_id,unconditional_denominator=n,preselection=pre,
                algebra=algebra,labels=labels,decision_maxima=maxima,rows=rows,full_geometry_rows=len(rows),
                maximum_full_geometry_rows=20,replacements=0))
            limits=dict(max_full_geometry_queries=20,max_axis_queries=60)
            review=terminal(root/'review.json',dict(complete=True,passed=True,predeclared_before_first_draw=True,
                no_previous_full_geometry_audit_of_population=True,preselection=pre,selection=selection,population=p,limits=limits),
                prefix+'-selection',prefix,'selection')
            selected_plan=dict(schema=admission.selected.PLAN_SCHEMA,target=target,definition=definition,
                reference_region=old,strata=strata,population=p,algebra=algebra,labels=labels,preselection=pre,
                selection=selection,review=review,limits=limits,observer_setup=setup,
                source_sha256={source.name:source_map[str(source)]},runtime=dict(synthetic=True))
            if schema==admission.V6:selected_plan.update(compiled_native=compiled,shape_compatibility=witness)
            selected_plan_ref=write(root/'geometry-plan.json',selected_plan)
            selected_inputs=dict(inputs,**{r['path']:r['sha256'] for r in
                (algebra,labels,pre,selection,review,selected_plan_ref,label_rows)})
            flags=dict(source_schema=schema,hard_line_geometry_certified=True,native_line_geometry_certified=schema==admission.V7,
                exclusion_contact_line_geometry_certified=schema==admission.V7)
            selected_receipt=dict(schema=admission.selected.SCHEMA,complete=True,passed=True,**flags,
                execution_plan_sha256=selected_plan_ref['sha256'],population=p,target_and_regions_sha256=target_id,
                **{k:selected_plan[k] for k in ('algebra','labels','preselection','selection','review')},
                full_geometry_selected_rows=len(rows),original_attempt_denominator=n,
                query_counts=dict(full_geometry=len(rows),axis=3*len(rows)),
                rows=[dict(draw=r['draw'],axes=3,membership_axes=3,**flags) for r in rows],
                observer_setup_counts={},input_sha256=selected_inputs,source_sha256=dict(source_map),runtime=dict(synthetic=True),
                **{k:endpoint[k] for k in ('native_identity_origin','compiled_native_binding','shape_compatibility_binding')},
                all_row_full_geometry_certified=False,new_pose_draws=0,new_Poisson_clouds=0,retries=0,replacements=0)
            selected_ref=terminal(root/'geometry.json',selected_receipt,prefix+'-geometry',prefix,'geometry')
            slots.append(dict(arm=arm_id,id=pop_id,selected_plan=selected_plan_ref,selected_receipt=selected_ref))
            report['input_sha256'].update(selected_inputs)
            objects[(arm_id,pop_id)]=dict(root=root,stat_slot=stat_slot,slot=slots[-1])
        protocol['arms'].append(arm);stats_plan['arms'].append(stat_arm)
        report['arms'][arm_id]=dict(populations=records,free_energy_precision_passed=False,**diagnostic_arm())
    protocol_ref=write(base/'protocol.json',protocol);files[protocol_ref['path']]=protocol_ref['sha256']
    stats_plan_ref=write(base/'statistics-plan.json',stats_plan)
    report['input_sha256'][stats_plan_ref['path']]=stats_plan_ref['sha256'];report['plan_sha256']=stats_plan_ref['sha256']
    stats_ref=terminal(base/'statistics.json',report,'statistics','all','statistics','complete')
    execution=dict(schema=admission.EXECUTION_SCHEMA,root=str(base),maximum_workers=1,threads=1,files=files,jobs=jobs)
    execution_ref=write(base/'execution-plan.json',execution)
    write(base/'execution/claim.json',dict(schema=admission.EXECUTION_SCHEMA,plan_sha256=execution_ref['sha256'],
        maximum_workers=1,threads=1,retries=0,replacements=0))
    for ordinal,(job,done) in enumerate(zip(jobs,completed)):
        directory=admission.runner.job_directory(base,ordinal,job)
        write(directory/'success.json',done);write(directory/'exit.json',done)
        write(directory/'attempt.json',dict(job=job))
        write(directory/'process.json',{k:done[k] for k in ('id','pid','birth_ticks','argv')})
    status=dict(schema=admission.EXECUTION_SCHEMA,complete=True,passed=True,plan_sha256=execution_ref['sha256'],
        files=files,completed=completed,active=None,unstarted=[],failure=None,retries=0,replacements=0,
        maximum_workers=1,threads=1)
    status_ref=write(base/'execution/summary.json',status)
    plan=dict(schema=admission.PLAN_SCHEMA,protocol=protocol_ref,execution_plan=execution_ref,
        execution_status=status_ref,statistics_plan=stats_plan_ref,statistics_result=stats_ref,populations=slots)
    path=base/'admission-plan.json';write(path,plan)
    return path,plan,objects


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.path,self.plan,self.objects=fixture(self.base)
        self.sources=mock.patch.object(admission,'local_sources',return_value={Path(admission.__file__).name:Path(admission.__file__)})
        self.sources.start()

    def tearDown(self):self.sources.stop();self.temp.cleanup()

    def join(self):return admission.join(self.path)

    def test_primary_complete_with_statistical_failures_keeps_scientific_gates_closed(self):
        with mock.patch.object(admission.statistics,'analyze',side_effect=AssertionError('No repeated statistics scan')), \
             mock.patch.object(admission.selected,'_select',side_effect=AssertionError('No selected row rescan')), \
             mock.patch.object(admission.streaming,'JsonLines',side_effect=AssertionError('No JSONL parser')), \
             mock.patch.object(admission.selected,'build_reconstructor',side_effect=AssertionError('No geometry')):
            result=self.join()
        self.assertTrue(result['implementation_admission_passed']);self.assertEqual(len(result['populations']),16)
        self.assertEqual(result['total_unconditional_attempts'],262144);self.assertEqual(result['selected_full_geometry_rows'],272)
        self.assertEqual(result['parsed_scientific_rows'],0);self.assertEqual(len(result['missing_studies']),3)
        self.assertFalse(result['full_vessel_gate_open']);self.assertFalse(result['assembly_gate_open'])
        self.assertEqual(result['unseen_support_verdict'],'unresolved')

    def test_full_declared_inventory_has_forty_populations_without_assembly_claim(self):
        other=self.base/'full';other.mkdir();path,_,_=fixture(other,full=True)
        result=admission.join(path)
        self.assertEqual(len(result['populations']),40);self.assertEqual(result['total_unconditional_attempts'],1048576)
        self.assertEqual(result['missing_studies'],[]);self.assertFalse(result['regional_convergence_established'])

    def test_missing_duplicate_population_and_changed_bound_file_are_fatal(self):
        original=copy.deepcopy(self.plan)
        for mode in ('missing','duplicate'):
            plan=copy.deepcopy(original)
            if mode=='missing':plan['populations'].pop()
            else:plan['populations'][1]=plan['populations'][0]
            write(self.path,plan)
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.join()
        write(self.path,original)
        self.objects[('hard_free','r00')]['root'].joinpath('samples.jsonl').write_bytes(b'changed original bytes')
        with self.assertRaisesRegex(ValueError,'Changed bound input'):self.join()

    def test_incomplete_undrained_failed_and_repeated_terminal_inventory_rejected(self):
        path=Path(self.plan['execution_status']['path']); original=admission.streaming.read(path)
        for mode in ('incomplete','active','unstarted','failure','undrained','missing','repeated','terminal'):
            value=copy.deepcopy(original)
            if mode=='incomplete':value['complete']=False
            elif mode=='active':value['active']={}
            elif mode=='unstarted':value['unstarted']=[{}]
            elif mode=='failure':value['failure']='preserved'
            elif mode=='undrained':value['completed'][0]['child_drained']=False
            elif mode=='missing':value['completed'].pop()
            elif mode=='repeated':value['completed'][1]=value['completed'][0]
            else:value['completed'][0]['terminal']['sha256']='0'*64
            self.plan['execution_status']=write(path,value);write(self.path,self.plan)
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.join()

    def test_scientific_protocol_cannot_silently_omit_comparisons_or_reuse_seeds(self):
        protocol=admission.streaming.read(self.plan['protocol']['path'])
        for mode in ('comparison','arm','allocation','seed','audit_seed','schema'):
            value=copy.deepcopy(protocol)
            if mode=='comparison':value['comparisons']=[]
            elif mode=='arm':value['arms'].pop()
            elif mode=='allocation':value['arms'][0]['samples']+=1
            elif mode=='seed':value['arms'][1]['populations'][0]['seed']=value['arms'][0]['populations'][0]['seed']
            elif mode=='audit_seed':value['arms'][0]['populations'][0]['audit_seed']=value['arms'][0]['populations'][0]['seed']
            else:value['arms'][0]['source_schema']=admission.V7
            with self.subTest(mode=mode),self.assertRaises(ValueError):admission.protocol_inventory(value)

    def test_optional_preparation_receipt_binds_plan_and_claim_without_a_cycle(self):
        path=Path(self.plan['execution_plan']['path']);execution=admission.streaming.read(path)
        execution['preparation_receipt']=str(self.base/'preparation.json')
        self.plan['execution_plan']=write(path,execution)
        prepared=write(self.base/'preparation.json',dict(complete=True,launched=False,
            execution_plan=self.plan['execution_plan'],protocol=self.plan['protocol']))
        claim_path=self.base/'execution/claim.json';claim=admission.streaming.read(claim_path)
        claim.update(plan_sha256=self.plan['execution_plan']['sha256'],preparation_receipt=prepared);write(claim_path,claim)
        status_path=Path(self.plan['execution_status']['path']);status=admission.streaming.read(status_path)
        status['plan_sha256']=self.plan['execution_plan']['sha256']
        self.plan['execution_status']=write(status_path,status);write(self.path,self.plan)
        self.assertTrue(self.join()['implementation_admission_passed'])
        write(self.base/'preparation.json',dict(complete=False))
        with self.assertRaisesRegex(ValueError,'Changed metadata'):self.join()

    def test_selection_exact_unconditional_maxima_reasons_and_original_ids(self):
        obj=self.objects[('hard_free','r00')];plan=admission.streaming.read(obj['slot']['selected_plan']['path'])
        selection=admission.streaming.read(plan['selection']['path']);pre=admission.streaming.read(plan['preselection']['path'])
        for mode in ('missing','duplicate','reason','maximum','line','offset','hash','region'):
            value=copy.deepcopy(selection)
            if mode=='missing':value['rows'].pop()
            elif mode=='duplicate':value['rows'][1]=value['rows'][0]
            elif mode=='reason':value['rows'][0]['reasons']=[]
            elif mode=='maximum':value['decision_maxima']['native']['draw']=16384
            elif mode=='line':value['rows'][0]['sample_record']['line']=1
            elif mode=='offset':value['rows'][0]['sample_record']['offset']=-1
            elif mode=='hash':value['rows'][0]['sample_record']['sha256']='broken'
            else:value['decision_maxima'].pop('competing')
            # The first sorted selected row is the added maximum at draw 0.
            if mode=='line':value['rows'][0]['sample_record']['line']=2
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                admission.check_selection(value,pre,plan['population'],16384)

    def test_selected_scope_and_identity_are_checked_after_authenticating_receipt(self):
        obj=self.objects[('hard_free','r00')]; slot=obj['slot'];path=Path(slot['selected_receipt']['path'])
        original=admission.streaming.read(path)
        for mode in ('native_scope','identity','count','axis','full_rows','review','plan_hash'):
            value=copy.deepcopy(original)
            if mode=='native_scope':value['native_line_geometry_certified']=True
            elif mode=='identity':value['native_identity_origin']='producer_compiled_native'
            elif mode=='count':value['full_geometry_selected_rows']-=1
            elif mode=='axis':value['query_counts']['axis']-=1
            elif mode=='full_rows':value['all_row_full_geometry_certified']=True
            elif mode=='review':value['review']=value['selection']
            else:value['execution_plan_sha256']='0'*64
            bound=write(path,value);slot['selected_receipt']=bound
            # Rebind only the synthetic lifecycle terminal, so this test reaches
            # semantic receipt checks instead of merely a stale hash rejection.
            status_path=Path(self.plan['execution_status']['path']);status=admission.streaming.read(status_path)
            execution=admission.streaming.read(self.plan['execution_plan']['path'])
            for ordinal,done in enumerate(status['completed']):
                if done['terminal']['path']==str(path):
                    done['terminal']=bound
                    directory=admission.runner.job_directory(self.base,ordinal,execution['jobs'][ordinal])
                    write(directory/'success.json',done);write(directory/'exit.json',done)
            self.plan['execution_status']=write(status_path,status);write(self.path,self.plan)
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.join()

    def test_diagnostics_retain_failed_and_unobserved_relevant_strata_without_new_statistics(self):
        report=admission.streaming.read(self.plan['statistics_result']['path'])
        strata=report['comparisons'][0]['kinds']['Qz']['strata']
        strata[0]['material']=True
        strata[1].update(historical_failure=True,comparison=dict(observed=True,passed=True))
        primary=report['arms']['class']['estimates']['Qz']['primary']
        primary['population_statistics']['free_energy_contrast']=dict(observed=True,halfwidth_95=.7)
        result=admission.summarize_diagnostics(report)
        self.assertTrue(result['aggregate_reducer_applied'])
        self.assertEqual(len(result['material_or_historical_stratum_comparisons']),2)
        # The second comparison agrees, but its inherited within-arm ESS/quality
        # is unresolved; it must remain unstable rather than disappearing.
        self.assertEqual(len(result['unstable_stratum_comparisons']),2)
        self.assertTrue(result['unstable_stratum_comparisons'][1]['comparison']['passed'])
        self.assertFalse(result['unstable_stratum_comparisons'][1]['left_quality']['passed'])
        self.assertIsNone(result['unstable_stratum_comparisons'][1]['left_parent_mass_fraction'])
        self.assertTrue(any(v['arm']=='class' and v['halfwidth_95']==.7 for v in result['free_energy_precision_issues']))
        self.assertTrue(result['regional_quality_issues']);self.assertTrue(result['fixed_comparison_issues'])
        del report['comparisons'][0]['kinds']['Qz']['strata'][-1]
        with self.assertRaisesRegex(ValueError,'Incomplete stratum'):admission.summarize_diagnostics(report)


if __name__=='__main__':unittest.main()
