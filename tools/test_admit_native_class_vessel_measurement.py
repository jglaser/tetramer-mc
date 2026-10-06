"""Synthetic prerequisite failures; no physical sampling or geometry queries."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import admit_native_class_vessel_measurement as gate


def arm(name, seed_base, samples=16384):
    records = [dict(id=f'r{i:02}', seed=seed_base+i, draws=samples,
                    unconditional_denominator=samples, classifier_contact_consistency_passed=True,
                    native_entry_unbound_anomalies=0) for i in range(8)]
    parent_diagnostics = {}
    def block(family=None, index=None):
        fraction = 1. if family is None else 1./gate.stats.BINS[family]
        declaration = dict(arm_id=name, population_count=8, draws_per_population=samples,
            total_unconditional_draws=8*samples, regions=list(gate.stats.REGIONS),
            target_and_regions_sha256=gate.smc.scope_id('a'*64, family, index),
            populations=[dict(id=r['id'], seed=r['seed']) for r in records])
        rows = []
        for i, record in enumerate(records):
            values = {r: None if r == 'unbound' else math.log(
                fraction*(1+i/10)*(3 if r == 'total' else 2 if r == 'native' else 1))
                for r in gate.stats.REGIONS}
            rows.append(dict(record, log_masses=values))
        summary = gate.stats.population.summarize_populations(declaration, rows)
        diagnostics = {r: dict(logQ=v['log_linear_mean'], observed=v['observed_positive'],
            ess=512. if v['observed_positive'] else 0.,
            max_fraction=.002 if v['observed_positive'] else None)
            for r, v in summary['estimates'].items()}
        if family is None: parent_diagnostics.update(diagnostics)
        return dict(declaration=declaration, population_records=rows, population_statistics=summary,
            row_diagnostics=diagnostics,
            quality={r:gate.stats.quality(diagnostics[r],summary['estimates'][r]) for r in gate.stats.DECISIONS},
            observed_parent_mass_fraction={r:None if r=='unbound' else
                math.exp(diagnostics[r]['logQ']-parent_diagnostics[r]['logQ']) for r in gate.stats.REGIONS})
    return dict(populations=records, classifier_contact_consistency_passed=True, estimates={kind:dict(primary=block(),
        strata={f:[block(f,i) for i in range(n)] for f,n in gate.stats.BINS.items()}) for kind in ('Qz','Q0')})


def fixture():
    names = ['hard_free','class',*(a['id'] for a in gate.followup.ARMS)]
    arms = {name:arm(name,100+100*i) for i,name in enumerate(names)}
    comparison = dict(schema=gate.followup.COMPARISON_SCHEMA, complete=True,
        target_and_regions_sha256='a'*64, target_descriptor=dict(region_sha256='b'*64),
        gates=gate.stats.GATES, missing_studies=[], arms=arms,
        retained_primary_comparisons=[gate.followup.comparison_pair('hard_free','class',arms,[])],
        comparisons=[gate.followup.comparison_pair('class',name,arms,[]) for name in names[2:]])
    return comparison


class PrerequisiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.base=fixture()

    def test_contact_diagnostics_do_not_invent_unbound_observations(self):
        result=gate.regional_diagnostics(self.base)
        self.assertTrue(result['contact_checks_passed'])
        self.assertEqual(len(result['unbound_comparison_issues']),8)
        self.assertTrue(all(not x['comparison']['observed'] for x in result['unbound_comparison_issues']))

    def test_primary_comparison_cannot_be_omitted_by_fresh_sensitivity_success(self):
        value=copy.deepcopy(self.base);value['retained_primary_comparisons']=[]
        with self.assertRaisesRegex(ValueError,'primary'):gate.regional_diagnostics(value)

    def test_stored_pass_flags_cannot_override_population_comparison(self):
        value=copy.deepcopy(self.base)
        value['comparisons'][0]['kinds']['Qz']['regions']['native']['passed']=False
        with self.assertRaisesRegex(ValueError,'Comparison differs'):gate.regional_diagnostics(value)

    def test_saved_quality_must_match_ess_maximum_and_population_error(self):
        value=copy.deepcopy(self.base)
        value['arms']['class']['estimates']['Qz']['primary']['row_diagnostics']['native']['ess']=1.
        with self.assertRaisesRegex(ValueError,'quality'):gate.regional_diagnostics(value)

    def test_attempted_denominators_and_seed_independence_are_preserved(self):
        for mutation in ('denominator','seed'):
            value=copy.deepcopy(self.base)
            if mutation=='denominator':
                value['arms']['class']['estimates']['Qz']['primary']['population_records'][0]['unconditional_denominator']-=1
            else: value['arms']['class']['populations'][0]['seed']=value['arms']['hard_free']['populations'][0]['seed']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):gate.regional_diagnostics(value)

    def test_population_summary_and_region_partition_cannot_be_relabelled(self):
        value=copy.deepcopy(self.base)
        value['arms']['class']['estimates']['Qz']['primary']['population_statistics']['free_energy_contrast']['halfwidth_95']=0.
        with self.assertRaisesRegex(ValueError,'linear population'):gate.regional_diagnostics(value)
        value=copy.deepcopy(self.base)
        value['arms']['class']['estimates']['Qz']['primary']['population_records'][0]['log_masses']['total']+=.1
        with self.assertRaisesRegex(ValueError,'partition'):gate.regional_diagnostics(value)

    def test_historical_stratum_still_requires_resolution(self):
        value=copy.deepcopy(self.base);family,index,region='orthant',5,'native'
        for name in ('class','class_large'):
            b=value['arms'][name]['estimates']['Qz']['strata'][family][index]
            b['row_diagnostics'][region]['ess']=1.
            b['quality'][region]=gate.stats.quality(b['row_diagnostics'][region],b['population_statistics']['estimates'][region])
        # Flags have to come from the complete comparison, not an aggregate list.
        value['comparisons'][0]=gate.followup.comparison_pair('class','class_large',value['arms'],
            [dict(family=family,bin=index,region=region)])
        result=gate.regional_diagnostics(value)
        self.assertFalse(result['contact_checks_passed'])
        self.assertTrue(any(s['historical_failure'] and s['bin']==index
                            for s in result['diagnostics']['unstable_stratum_comparisons']))

    def test_material_stratum_cannot_be_hidden_by_altering_its_fraction(self):
        value=copy.deepcopy(self.base)
        value['arms']['class']['estimates']['Qz']['strata']['orthant'][5]['observed_parent_mass_fraction']['native']=0.
        with self.assertRaisesRegex(ValueError,'mass fraction'):gate.regional_diagnostics(value)

    def test_classifier_contact_inconsistency_is_not_a_sampling_failure(self):
        value=copy.deepcopy(self.base)
        value['arms']['class']['populations'][0]['native_entry_unbound_anomalies']=1
        with self.assertRaisesRegex(ValueError,'classifier/contact'):gate.regional_diagnostics(value)

    def test_linear_se_and_absolute_discrepancy_have_distinct_states(self):
        for a,b,expected in [(True,True,'corroborated'),(False,False,'material_contradiction'),
                              (False,True,'unresolved'),(True,False,'unresolved')]:
            result=dict(observed=True,passed=a and b,SE_passed=a,absolute_passed=b,
                        SE_multiplier=3.,absolute_limit=.2)
            self.assertEqual(gate.comparison_state(result),expected)
        empty=dict(observed=False,passed=False,SE_passed=None,absolute_passed=None,
                   SE_multiplier=3.,absolute_limit=.2)
        self.assertEqual(gate.comparison_state(empty),'unresolved')
        empty['passed']=True
        with self.assertRaises(ValueError):gate.comparison_state(empty)

    def test_bound_is_absolute_and_never_claims_hard_volume_negligibility(self):
        region=dict(gaussian_chart=dict(covariances=[[[float(i==j) for j in range(6)] for i in range(6)]],
                                       angular_length=2.),mahalanobis_radius=4.)
        comparison=copy.deepcopy(self.base)
        comparison['target_descriptor'].update(shape_sha256='c'*64,native_definition_sha256='d'*64,
                                                physical_measure='Lebesgue times normalized Haar')
        report=dict(schema='conditional-ray-reference-comparison-v1',complete=True,
            region_sha256='b'*64,shape_sha256='c'*64,native_definition=dict(definition_sha256='d'*64),
            unbound_finite_region_bound=gate.unbound_volume_bound(region))
        result=gate.unbound_evidence(report,comparison,region,dict(path='synthetic',sha256='e'*64))
        self.assertFalse(result['relative_negligibility_proven'])
        self.assertFalse(result['Q0_unbound_estimate_resolved'])
        report['unbound_finite_region_bound']['log_Qz_upper']-=1.
        with self.assertRaisesRegex(ValueError,'bound changed'):gate.unbound_evidence(report,comparison,region,{})

    def test_pinned_plan_bytes_are_authenticated_before_any_dependent_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';path.write_text(json.dumps(dict(schema=gate.PLAN_SCHEMA)))
            with self.assertRaisesRegex(ValueError,'Changed metadata'):gate.load(path,'0'*64)


def completed_followup_fixture(root, mutation=None):
    """Synthetic 121-phase JSON lifecycle; no producer or sample stream exists.

    Mutations precede serialization and rebinding, so semantic failures cannot
    pass merely because a downstream receipt has internally consistent hashes.
    The actual Metadata reader authenticates all JSON inputs consumed by the
    helper; terminal row-audit contents remain inherited upstream evidence.
    """
    root = Path(root).resolve()
    protocol = dict(root=str(root),arms=[dict(id=a['id'],populations=[dict(id=f'r{i:02}') for i in range(8)])
        for a in gate.followup.ARMS],target=dict(shape='synthetic-shape'),
        target_and_regions_sha256='a'*64,strata=dict(radial_edges=[0.,2.,3.,4.]),
        gates=copy.deepcopy(gate.stats.GATES),comparisons=[],
        external_primary=dict(admission=dict(path=str(root/'prior.json'),sha256='b'*64)))
    statistical_plan = dict(schema=gate.stats.PLAN_SCHEMA,
        **{k:copy.deepcopy(protocol[k]) for k in ('target','target_and_regions_sha256','strata','gates','comparisons')})
    jobs = []
    for arm in protocol['arms']:
        for population in arm['populations']:
            identity = f'{arm["id"]}-{population["id"]}'
            for phase in ('producer','algebra','labels','selection','geometry'):
                jobs.append(dict(id=f'{identity}-{phase}',population=identity,phase=phase,
                                 argv=['synthetic-never-executed',identity,phase]))
    jobs.append(dict(id='statistics',population='all',phase='statistics',argv=['synthetic-never-executed','statistics']))
    execution = dict(schema=gate.followup.admission.EXECUTION_SCHEMA,root=str(root),
                     maximum_workers=1,threads=1,jobs=copy.deepcopy(jobs))
    done = dict(schema=gate.followup.admission.EXECUTION_SCHEMA,complete=True,passed=True,failure=None,
        active=None,unstarted=[],retries=0,replacements=0,maximum_workers=1,threads=1,
        completed=[dict(job,success=True,returncode=0,child_drained=True,error=None,timeout=False,
                        terminal_hash_error=None,retries=0,replacements=0) for job in jobs])
    fresh = dict(schema=gate.stats.SCHEMA,complete=True,comparisons=[])
    admitted = dict(target_and_regions_sha256=protocol['target_and_regions_sha256'],
        total_unconditional_attempts=gate.followup.TOTAL_ATTEMPTS,
        external_primary=copy.deepcopy(protocol['external_primary']),primary_is_external_control=True)
    state = dict(protocol=protocol,statistical_plan=statistical_plan,execution=execution,
                 done=done,fresh=fresh,admitted=admitted)
    if mutation is not None: mutation(state)

    def put(name,value):
        path = root/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True))
        return dict(path=str(path),sha256=gate.stats.streaming.sha(path))

    protocol_ref = put('protocol.json',protocol)
    execution['files'] = {protocol_ref['path']:protocol_ref['sha256']}
    execution_ref = put('execution-plan.json',execution)
    statistical_ref = put('analysis/statistics-plan.json',statistical_plan)
    fresh.update(plan_sha256=statistical_ref['sha256'],input_sha256={statistical_ref['path']:statistical_ref['sha256']})
    fresh_ref = put('analysis/statistics.json',fresh)
    closure = {}
    for i,receipt in enumerate(done['completed']):
        ref = fresh_ref if receipt['phase'] == 'statistics' else put(f'terminals/{i:03}.json',dict(complete=True,phase=receipt['phase']))
        receipt['terminal'] = ref; closure[ref['path']] = ref['sha256']
    done['plan_sha256'] = execution_ref['sha256']
    done_ref = put('execution/summary.json',done)
    plan = dict(schema=gate.followup.admission.PLAN_SCHEMA,protocol=protocol_ref,execution_plan=execution_ref,
                execution_status=done_ref,statistics_plan=statistical_ref,statistics_result=fresh_ref)
    for key in ('protocol','execution_plan','execution_status','statistics_plan','statistics_result'):
        ref = plan[key]; closure[ref['path']] = ref['sha256']
    admitted.update(plan_sha256=put('analysis/admission-plan.json',plan)['sha256'],
                    input_sha256=closure,statistics_result=fresh_ref)
    metadata = gate.smc.Metadata()
    # Match load(): these two already decoded reports are authenticated before
    # completed_followup consumes their linked admission/execution metadata.
    protocol = metadata.read(protocol_ref); fresh = metadata.read(fresh_ref)
    return protocol,protocol_ref,admitted,fresh,metadata


class CompletedFollowupTests(unittest.TestCase):
    def test_full121_phase_receipts_and_metadata_links_pass_without_sample_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            args = completed_followup_fixture(directory)
            gate.completed_followup(*args); args[-1].finish()
            self.assertEqual(len(args[-1].files),6)
            self.assertTrue(all(Path(path).suffix == '.json' for path in args[-1].files))
            self.assertFalse(list(Path(directory).rglob('*.jsonl')))

    def test_incomplete_or_not_drained_summary_and_preserved_failure_are_rejected(self):
        for field,value in [('complete',False),('passed',False),('active',{'id':'live'}),
                            ('unstarted',['remaining']),('retries',1),('replacements',1)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                args = completed_followup_fixture(directory,lambda s:s['done'].update({field:value}))
                with self.assertRaisesRegex(ValueError,'completed and drained'):gate.completed_followup(*args)
        with tempfile.TemporaryDirectory() as directory:
            args = completed_followup_fixture(directory)
            (Path(directory)/'execution/failure.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Preserved execution failure'):gate.completed_followup(*args)

    def test_missing_or_duplicate_phase_fails_with_internally_rehashed_metadata(self):
        def remove(state):
            state['execution']['jobs'].pop(19); state['done']['completed'].pop(19)
        def duplicate(state):
            state['execution']['jobs'][19] = copy.deepcopy(state['execution']['jobs'][18])
            state['done']['completed'][19] = copy.deepcopy(state['done']['completed'][18])
        for mutation in (remove,duplicate):
            with self.subTest(mutation=mutation.__name__),tempfile.TemporaryDirectory() as directory:
                args = completed_followup_fixture(directory,mutation)
                with self.assertRaisesRegex(ValueError,'Missing follow-up execution phase'):gate.completed_followup(*args)

    def test_changed_terminal_receipt_or_unbound_receipt_is_rejected(self):
        for field,value in [('child_drained',False),('success',False),('returncode',1),
                            ('timeout',True),('terminal_hash_error','changed'),('argv',['different'])]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as directory:
                args = completed_followup_fixture(directory,lambda s:s['done']['completed'][17].update({field:value}))
                with self.assertRaisesRegex(ValueError,'terminal phase is incomplete or differs'):gate.completed_followup(*args)
        with tempfile.TemporaryDirectory() as directory:
            args = completed_followup_fixture(directory)
            args[2]['input_sha256'].pop(str(Path(directory).resolve()/'terminals/017.json'))
            with self.assertRaisesRegex(ValueError,'absent from upstream closure'):gate.completed_followup(*args)

    def test_changed_statistics_target_strata_or_gates_cannot_reuse_receipts(self):
        for key,value in [('target',{'shape':'other'}),('target_and_regions_sha256','c'*64),
                          ('strata',{'radial_edges':[0.,4.]}),('gates',{}),('comparisons',[{'unplanned':True}])]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                args = completed_followup_fixture(directory,lambda s:s['statistical_plan'].update({key:value}))
                with self.assertRaisesRegex(ValueError,'frozen target or diagnostics'):gate.completed_followup(*args)

    def test_changed_attempt_denominator_or_external_primary_cannot_be_admitted(self):
        for key,value in [('total_unconditional_attempts',gate.followup.TOTAL_ATTEMPTS-1),
                          ('target_and_regions_sha256','c'*64),('external_primary',{}),
                          ('primary_is_external_control',False)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                args = completed_followup_fixture(directory,lambda s:s['admitted'].update({key:value}))
                with self.assertRaisesRegex(ValueError,'target, allocation or external control'):gate.completed_followup(*args)

    def test_changed_statistics_metadata_bytes_or_closure_fail_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            args = completed_followup_fixture(directory)
            (Path(directory)/'analysis/statistics-plan.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Changed metadata'):gate.completed_followup(*args)
        with tempfile.TemporaryDirectory() as directory:
            args = completed_followup_fixture(directory)
            args[3]['input_sha256'][str(Path(directory).resolve()/'analysis/statistics-plan.json')] = '0'*64
            with self.assertRaisesRegex(ValueError,'absent from upstream closure'):gate.completed_followup(*args)


if __name__=='__main__':unittest.main()
