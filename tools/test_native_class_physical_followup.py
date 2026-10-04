"""Synthetic protocol/lifecycle/summary checks; never a protein sample or build."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import native_class_physical_followup as followup
import test_prepare_native_class_physical_campaign as preparation_fixture
import test_admit_native_class_physical_campaign as admission_fixture


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False)+'\n')
    return followup.base.bound(path)


def protocol(base):
    arms = []
    for j, declaration in enumerate(followup.ARMS):
        arms.append(dict(declaration, producer='class', source_schema=followup.admission.V7,
            populations=[dict(id=f'r{i:02}', seed=1000+100*j+i, audit_seed=10000+100*j+i,
                directory=str(base/declaration['id']/f'r{i:02}')) for i in range(8)]))
    return dict(schema=followup.base.SCHEMA, study_scope=followup.SCOPE,
        arms=arms, strata=followup.statistics.STRATA, gates=followup.statistics.GATES,
        cloud_replicates=2, total_attempts=786432, maximum_clouds=1572864,
        comparisons=[], external_comparisons=followup.external_comparisons([]), primary_is_external_control=True,
        phase_limits=preparation_fixture.limits(), resources=preparation_fixture.resources())


def materialization_context(base):
    old = preparation_fixture.materialization_context(base)
    guide = copy.deepcopy(old['guides']['class']); guide['gaussian_components'] = [{}]*116
    guide_ref = save(base/'inherited-guide.json', guide); old['bindings'].reference(guide_ref)
    old['inputs']['schema'] = followup.INPUT_SCHEMA
    old['inputs']['primary'] = {name: save(base/(name+'.json'), dict(synthetic=True)) for name in followup.PRIMARY_PINS}
    original = dict(schema=followup.base.SCHEMA, root=str(base), stage_materialization_policy=followup.base.POLICY,
        runtime=old['runtime'], target=old['inputs']['target'], target_and_regions_sha256=old['target_id'],
        strata=followup.statistics.STRATA, strata_file=save(base/'strata.json', followup.statistics.STRATA),
        gates=followup.statistics.GATES, config=old['inputs']['config'], shape=old['inputs']['shape'],
        native_identity={}, observer_setup={}, historical_failed_strata=old['inputs']['historical_failed_strata'],
        historical_failed_strata_policy='retention_only', prerequisites={}, class_guide_profile=followup.base.REVISED_PROFILE,
        files=dict(old['bindings'].files), producers={'class': old['producers']['class']['input']})
    sources = followup.source_closure()
    for p in sources.values(): old['bindings'].bind(p)
    return dict(inputs=old['inputs'], review={}, bindings=old['bindings'], primary={'protocol':original},
        guide=guide, guide_ref=guide_ref, target_id=old['target_id'], descriptor={}, sources=sources,
        runtime=old['runtime'], history=old['history'], inputs_path=old['inputs_path'], review_path=old['review_path'])


def join_fixture(root):
    """Fresh synthetic 121-stage lifecycle, reusing the archived receipt fixture."""
    original_write = admission_fixture.write
    lookup = {a['id']: a for a in followup.ARMS}
    def intercept(path, value):
        if isinstance(value, dict):
            if value.get('schema') == followup.admission.V7 and 'physical_fixed_neighbors' in value:
                arm = next(a for a in lookup if path.parent.name.startswith(a+'-'))
                value.update(importance_uniform_probability=lookup[arm]['alpha'], lambda_ratio=lookup[arm]['lambda_ratio'])
            if value.get('schema') == followup.base.SCHEMA:
                value.update({k:v for k,v in protocol(root).items() if k != 'arms'})
                value['external_primary'] = {'synthetic': True}
                for a in value['arms']: a.update(lookup[a['id']], producer='class')
            if value.get('schema') in (followup.statistics.PLAN_SCHEMA, followup.statistics.SCHEMA):
                value['comparisons'] = []
        return original_write(path, value)
    with patch.object(admission_fixture.admission, 'FULL', {a['id']:(followup.admission.V7,a['samples']) for a in followup.ARMS}), \
         patch.object(admission_fixture, 'write', side_effect=intercept):
        path, plan, objects = admission_fixture.fixture(root, full=True)
    p = followup.read(plan['protocol']['path'])
    old = dict(target=p['target'], target_and_regions_sha256=p['target_and_regions_sha256'], arms=[])
    return path, plan, objects, {'protocol': old}


def toy_arm(name, samples, offset):
    """Saved population columns with unequal trial counts and explicit zero mass."""
    records = [dict(id=f'r{i:02}', seed=offset+i, draws=samples, unconditional_denominator=samples,
                    source_schema=followup.admission.V7) for i in range(8)]
    estimates = {}
    for kind in ('Qz','Q0'):
        moments = []
        for i in range(8):
            m = {}
            for prefix in ['']+[f'{family}:{j}:' for family,n in followup.statistics.BINS.items() for j in range(n)]:
                for region in followup.statistics.REGIONS:
                    obj = followup.statistics.Moment()
                    if region != 'unbound':
                        # Native=2, competing=1; total=3; two native children=1 each.
                        import math
                        amount = 3 if region=='total' else 2 if region=='native' else 1
                        obj.add(math.log(amount*(1+i/10)), dict(draw=0))
                    m[prefix+region] = obj
            moments.append(m)
        primary = followup.statistics.statistics_block(name,records,moments,samples,'a'*64)
        strata = {}
        for family,n in followup.statistics.BINS.items():
            strata[family] = []
            for index in range(n):
                x = followup.statistics.statistics_block(name,records,moments,samples,'a'*64,f'{family}:{index}:')
                x['observed_parent_mass_fraction'] = {r:None if r=='unbound' else 1. for r in followup.statistics.REGIONS}
                strata[family].append(x)
        estimates[kind] = dict(primary=primary,strata=strata)
    return dict(populations=records,estimates=estimates)


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_exact_allocation_and_independent_controls(self):
        p = protocol(self.root); inventory = followup.protocol_inventory(p)
        self.assertEqual(len(inventory),24)
        self.assertEqual(sum(a['samples']*8 for a in p['arms']),786432)
        self.assertEqual(p['comparisons'],[])
        self.assertEqual([x['left'] for x in p['external_comparisons']],['class']*3)
        self.assertNotIn('hard_free',[a['id'] for a in p['arms']])

    def test_reject_allocation_proposal_population_seed_and_budget_drift(self):
        original = protocol(self.root)
        mutations = [lambda p:p['arms'].pop(), lambda p:p['arms'].append(copy.deepcopy(p['arms'][0])),
            lambda p:p['arms'][0].update(samples=16384), lambda p:p['arms'][1].update(alpha=.5),
            lambda p:p['arms'][2].update(lambda_ratio=128), lambda p:p['arms'][0]['populations'].pop(),
            lambda p:p['arms'][0]['populations'][1].update(seed=p['arms'][0]['populations'][0]['seed']),
            lambda p:p['arms'][0]['populations'][0].update(audit_seed=p['arms'][0]['populations'][0]['seed']),
            lambda p:p.update(comparisons=[dict(left='class',right='class_large')]),
            lambda p:p.update(external_comparisons=[]), lambda p:p['resources'].update(maximum_workers=2),
            lambda p:p.update(maximum_clouds=786432),lambda p:p.update(primary_is_external_control=False)]
        for mutate in mutations:
            p=copy.deepcopy(original);mutate(p)
            with self.subTest(mutation=mutate),self.assertRaises((ValueError,KeyError)):followup.protocol_inventory(p)

    def test_materialization_has_only_121_new_jobs_and_preserves_guide(self):
        context=materialization_context(self.root/'old'); out=self.root/'new';out.mkdir()
        result=followup.materialize(out,context,dict(seeds=[],files={},roots=['synthetic']))
        p=followup.read(out/'protocol.json');plan=followup.read(out/'execution-plan.json')
        self.assertEqual(len(plan['jobs']),121);self.assertEqual(result['primary_jobs_reexecuted'],0)
        self.assertEqual(sum(j['phase']=='producer' for j in plan['jobs']),24)
        self.assertFalse(any(j['population'].startswith('hard_free-') or j['population'].startswith('class-r') for j in plan['jobs']))
        self.assertEqual(p['target'],context['primary']['protocol']['target'])
        self.assertEqual(p['producers']['class'],context['primary']['protocol']['producers']['class'])
        for arm in p['arms']:
            got=followup.read(arm['guide']['path']);wanted=copy.deepcopy(context['guide'])
            wanted['defensive_uniform_shell_probability']=arm['alpha'];self.assertEqual(got,wanted)
            for pop in arm['populations']:
                self.assertEqual(followup.read(pop['preselection']['path']),
                    followup.base.selected.preselection(pop['id'],arm['samples'],pop['audit_seed']))
        # Verify the real unchanged controller accepts the frozen manifest only.
        with patch.object(followup.admission.runner,'__file__',str(out/'code/run_native_class_physical_campaign.py')):
            followup.admission.runner.verify_plan(out/'execution-plan.json',plan,followup.sha(out/'execution-plan.json'),fresh=True)
        self.assertFalse((out/'execution').exists())
        # Synthetic authority record only: invoke the real stage loader against
        # the copied frozen closure, without a child, geometry or row stream.
        save(out/'execution/claim.json',dict(plan_sha256=followup.sha(out/'execution-plan.json')))
        with patch.object(followup.stage,'__file__',str(out/'code/native_class_physical_stage.py')):
            loaded,_=followup.stage.load_protocol(out/'protocol.json',followup.sha(out/'protocol.json'),followup.stage.Inputs())
        self.assertEqual(loaded['source_sha256'],p['source_sha256'])
        geometry_sources=followup.stage.source_map(p,followup.base.selected)
        self.assertTrue(set(geometry_sources)<=set(p['source_sha256']))
        self.assertEqual(len(geometry_sources),46)

    def test_seed_derivation_disjoint_roles_arms_and_namespaces(self):
        def seeds(namespace):return [followup.base.stream_seed(namespace,a['id'],f'r{i:02}',role)
            for a in followup.ARMS for i in range(8) for role in ('physical','audit')]
        a,b=seeds('synthetic-A'),seeds('synthetic-B')
        self.assertEqual(len(set(a)),48);self.assertEqual(a,seeds('synthetic-A'));self.assertTrue(set(a).isdisjoint(b))

    def test_historical_collision_rejected_before_output_creation(self):
        context=materialization_context(self.root/'old');out=self.root/'new'
        seed=followup.base.stream_seed(context['inputs']['seed_namespace'],'class_large','r00','physical')
        with patch.object(followup,'validate',return_value=context), \
             patch.object(followup.base,'seed_inventory',return_value=dict(seeds=[seed])):
            with self.assertRaisesRegex(ValueError,'historical'):followup.prepare(out,'unused','unused')
        self.assertFalse(out.exists())

    def test_partial_preparation_failure_is_preserved_and_cannot_resume(self):
        context=materialization_context(self.root/'old');out=self.root/'new'
        with patch.object(followup,'validate',return_value=context), \
             patch.object(followup.base,'seed_inventory',return_value=dict(seeds=[])), \
             patch.object(followup,'materialize',side_effect=ValueError('synthetic failure')):
            with self.assertRaisesRegex(ValueError,'synthetic failure'):followup.prepare(out,'unused','unused')
            self.assertFalse(followup.read(out/'preparation-failure.json')['launched'])
            with self.assertRaisesRegex(ValueError,'Fresh'):followup.prepare(out,'unused','unused')

    def test_admission_joins_all_121_synthetic_terminals_without_row_replay(self):
        path,plan,objects,old=join_fixture(self.root/'done')
        with patch.object(followup,'primary_evidence',return_value=old), \
             patch.object(followup,'local_sources',return_value={Path(followup.admission.__file__).name:Path(followup.admission.__file__)}), \
             patch.object(followup.statistics,'analyze',side_effect=AssertionError('No repeated reduction')), \
             patch.object(followup.base.streaming,'JsonLines',side_effect=AssertionError('No row parser')):
            result=followup.join(path)
        self.assertEqual(result['total_unconditional_attempts'],786432);self.assertEqual(len(result['populations']),24)
        self.assertTrue(result['selected_full_geometry_gate_satisfied']);self.assertFalse(result['regional_convergence_established'])
        self.assertEqual(result['missing_studies'],['external_primary_comparison'])

    def test_incomplete_followup_cannot_be_admitted(self):
        path,plan,_,old=join_fixture(self.root/'done')
        status=followup.read(plan['execution_status']['path']);status['completed'].pop()
        plan['execution_status']=save(plan['execution_status']['path'],status);save(path,plan)
        with patch.object(followup,'primary_evidence',return_value=old),self.assertRaisesRegex(ValueError,'Missing or repeated'):
            followup.join(path)

    def test_comparison_uses_linear_unconditional_masses_not_normalized_fractions(self):
        arms={'class':toy_arm('class',16384,100),'class_large':toy_arm('class_large',65536,200)}
        r=followup.comparison_pair('class','class_large',arms,[])
        import math
        self.assertAlmostEqual(r['kinds']['Qz']['regions']['native']['log_left_minus_right'],math.log(4))
        self.assertFalse(r['kinds']['Qz']['regions']['unbound']['observed'])
        self.assertTrue(r['kinds']['Qz']['free_energy_contrast']['passed'])
        self.assertEqual(len(r['kinds']['Qz']['strata']),280)

    def test_comparison_rejects_reused_population_seeds(self):
        arms={'class':toy_arm('class',16384,100),'class_large':toy_arm('class_large',65536,100)}
        with self.assertRaises(ValueError):followup.comparison_pair('class','class_large',arms,[])

    def test_report_inventory_keeps_every_attempt_and_rejects_duplicates(self):
        p=protocol(self.root);report=dict(arms={})
        for arm in p['arms']:
            report['arms'][arm['id']]=dict(populations=[dict(id=pop['id'],seed=pop['seed'],draws=arm['samples'],
                unconditional_denominator=arm['samples'],source_schema=arm['source_schema']) for pop in arm['populations']])
        followup.check_report_inventory(report,p['arms'])
        report['arms']['class_large']['populations'][0]['unconditional_denominator']-=1
        with self.assertRaisesRegex(ValueError,'denominator'):followup.check_report_inventory(report,p['arms'])

    def test_primary_pins_fail_closed_before_any_scientific_read(self):
        refs={name:dict(path='/nonexistent/'+name+'.json',sha256='0'*64) for name in followup.PRIMARY_PINS}
        with self.assertRaisesRegex(ValueError,'Changed external primary'):
            followup.primary_evidence(refs,followup.statistics.Bindings())

    def test_union_keeps_failed_q0_and_qz_strata_after_mass_drops(self):
        history=dict(complete=True,target_and_regions_sha256='a'*64,
            entries=[dict(family='orthant',bin=22,region='competing')])
        row=dict(left='hard_free',right='class',family='orthant',bin=49,region='native',
            material=True,historical_failure=False,left_quality=dict(passed=False),right_quality=dict(passed=False),
            left_observed=True,right_observed=True)
        primary=dict(complete=True,target_and_regions_sha256='a'*64,statistical_diagnostics=dict(
            unstable_stratum_comparisons=[dict(row,kind='Q0'),dict(row,kind='Qz')]))
        got=followup.history_union(history,primary,{'old':'bound'},{'primary':'bound'})
        self.assertEqual(len(got['entries']),2);self.assertEqual(len(got['primary_unstable_provenance']),2)
        self.assertEqual(got['added_keys'],[dict(family='orthant',bin=49,region='native')])
        self.assertEqual({x['kind'] for x in got['primary_unstable_provenance']},{'Q0','Qz'})
        primary['target_and_regions_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'target'):followup.history_union(history,primary,{}, {})

    def test_metadata_comparison_retains_all_external_and_fresh_populations(self):
        p=protocol(self.root);p.update(target_and_regions_sha256='a'*64,external_primary={'synthetic':True})
        original=[]
        for j,name in enumerate(('hard_free','class')):
            original.append(dict(id=name,samples=16384,source_schema=followup.admission.V7,
                populations=[dict(id=f'r{i:02}',seed=100+100*j+i,directory=str(self.root/'old'/name/f'r{i:02}')) for i in range(8)]))
        old=dict(schema=followup.statistics.SCHEMA,complete=True,target_and_regions_sha256='a'*64,target_descriptor={},
            gates=followup.statistics.GATES,comparisons=[dict(retained='primary comparison')],
            arms={a['id']:toy_arm(a['id'],a['samples'],a['populations'][0]['seed']) for a in original})
        fresh=dict(schema=followup.statistics.SCHEMA,complete=True,target_and_regions_sha256='a'*64,target_descriptor={},
            gates=followup.statistics.GATES,comparisons=[],
            arms={a['id']:toy_arm(a['id'],a['samples'],a['populations'][0]['seed']) for a in p['arms']})
        protocol_ref=save(self.root/'protocol.json',p);report_ref=save(self.root/'statistics.json',fresh)
        admitted=dict(schema=followup.ADMISSION_SCHEMA,complete=True,implementation_admission_passed=True,
            protocol=protocol_ref,total_unconditional_attempts=786432,selected_full_geometry_gate_satisfied=True,
            statistics_result=report_ref,populations=[dict(arm=a['id'],id=x['id'],samples=a['samples'],source_schema=a['source_schema'])
                for a in p['arms'] for x in a['populations']])
        accepted_ref=save(self.root/'admission.json',admitted)
        with patch.object(followup,'primary_evidence',return_value=dict(statistics=old,protocol=dict(arms=original))), \
             patch.object(followup.statistics,'analyze',side_effect=AssertionError('No row reduction')):
            result=followup.compare(protocol_ref,accepted_ref)
        self.assertEqual(result['arms'],dict(old['arms'],**fresh['arms']))
        self.assertEqual(len(result['comparisons']),3);self.assertEqual(result['parsed_scientific_rows'],0)
        self.assertEqual(result['retained_primary_comparisons'],old['comparisons'])
        self.assertFalse(result['regional_convergence_established']);self.assertFalse(result['assembly_gate_open'])


if __name__ == '__main__':unittest.main()
