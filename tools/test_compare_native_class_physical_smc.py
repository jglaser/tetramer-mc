"""Synthetic summary-only bridge fixtures. No poses, sample rows or geometry.

Receipt objects below are fabricated attestations for interface tests. Original
target SHA constants are patched only in the test; production has no bypass.
"""
from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import compare_native_class_physical_smc as bridge


def save(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
    return dict(path=str(path.resolve()),sha256=bridge.current.streaming.sha(path))


def read(ref): return json.loads(Path(ref['path']).read_text())


def patches(target,shape):
    context=ExitStack()
    context.enter_context(mock.patch.object(bridge.current,'ORIGINAL_REGION_SHA',target['region']['sha256']))
    context.enter_context(mock.patch.object(bridge.current,'ORIGINAL_SHAPE_SHA',shape))
    context.enter_context(mock.patch.object(bridge,'OLD_SHA',target['old_r5_region']['sha256']))
    context.enter_context(mock.patch.object(bridge,'NATIVE_SHA',target['native_definition']['sha256']))
    return context


def toy_population(index,seed,restricted=False,zero=False):
    n=0 if zero else 100
    weights={'total':100,'registered_native_entry':50,'old_R5_intersection_native':20,
             'remaining_R4_native':30,'contact_no_native_entry':40,'unbound_no_native_entry':10}
    if restricted: weights={'total':100,'contact_no_native_entry':80,'unbound_no_native_entry':20}
    masses={k:None if zero else math.log((8+2*index)*v/100) for k,v in weights.items()}
    hard={k:None if zero else math.log(2*v/100) for k,v in weights.items()}
    # The real historical documents separate diagnostic stages 0..128 from
    # the named terminal profile. A zero estimate records only stage 0.
    stages=[dict(stage=0,beta=0.,log_Z=None,zero_estimate=True)] if zero else [
        dict(stage=i,beta=i/128,log_Z=math.log(.1)+(masses['total']-math.log(.1))*i/128)
        for i in range(129)]
    return dict(id=f'r{index:02}',seed=seed,initial_draws=1000,initial_hits=0 if zero else 100,
        zero_estimate=zero,stages=stages,
        initial_physical_hard_log_masses=hard,initial_bridge_log_masses={k:999. for k in weights},
        terminal=dict(stage='terminal',measure='restricted final physical target' if restricted else 'final physical target',
            particles=n,counts={k:0 if zero else v for k,v in weights.items()},
            log_masses=masses,log_strata={family:{k:[v]+[None]*(size-1) for k,v in masses.items()}
                for family,size in bridge.current.BINS.items()}))


def fixture(root):
    root=Path(root).resolve(); shape=save(root/'shape.json',dict(synthetic=True))
    region=dict(activity=.035,depletant_radius=1.5,mahalanobis_radius=4.,minimum_mahalanobis_radius=0.,
        minimum_original_q=0.,minimum_original_q_inclusive=True,maximum_original_q_inclusive=True,
        shape_sha256=shape['sha256'],physical_fixed_neighbors=[dict(synthetic=1),dict(synthetic=2)],
        capture_center=[0.,0.,0.],capture_radius=170.,physical_metric=dict(synthetic=True),
        gaussian_chart=dict(covariances=[[[float(i==j) for j in range(6)] for i in range(6)]]))
    target=dict(region=save(root/'region.json',region),
        old_r5_region=save(root/'old.json',dict(region,mahalanobis_radius=5.,minimum_original_q=1.,minimum_original_q_inclusive=False)),
        native_definition=save(root/'native.json',dict(synthetic=True)))
    with patches(target,shape['sha256']):
        _,descriptor,target_id=bridge.current.target_identity(target,bridge.current.Bindings())
    config=dict(fixed_poses=region['physical_fixed_neighbors'],capture_center=region['capture_center'],capture_radius=170.,
                reservoir_density=.035,depletant_radius=1.5,metadata=region['physical_metric'])
    historical={}
    for arm_index,name in enumerate(['unrestricted_broad','unrestricted_narrow','restricted_narrow','restricted_large','restricted_broad']):
        restricted=name.startswith('restricted')
        historical[name]=dict(restricted=restricted,populations=[toy_population(i,100+100*arm_index+i,restricted) for i in range(4)])
    target_physics=dict(shape_sha256=shape['sha256'],region_sha256=target['region']['sha256'],measure=bridge.MEASURE,
        fixed_poses=config['fixed_poses'],capture_center=[0.,0.,0.],capture_radius=170.,activity=.035,depletant_radius=1.5)
    analyses={}
    for name in ('broad','narrow'):
        pops=historical['unrestricted_'+name]['populations']
        protocol=dict(physical_target=target_physics,proposal=dict(reference_region_sha256=target['old_r5_region']['sha256']),
            source_and_input_sha256={str(root/'native-partition-definition.json'):bridge.PARTITION_SHA},
            jobs=[dict(id=p['id'],seed=p['seed']) for p in pops],
            allocation=dict(independent_populations=4,unconditional_initialization_draws_each=1000,particles_each=100,stages_each=128))
        analyses[name]=save(root/(name+'.json'),dict(schema='smc-r4-control-analysis-v1',complete=True,
            analyzer_source_sha256={'analyze_r4_smc_control.py':bridge.STRATA_SOURCES['analyze_r4_smc_control.py']},
            protocol_snapshot=protocol,physical_config=config,native_definition_sha256=target['native_definition']['sha256'],populations=pops))
    auth=save(root/'authentication.json',dict(schema='completed-smc-bridge-evidence-authentication-v1',complete=True,
        input_sha256={r['path']:r['sha256'] for r in analyses.values()}))
    review=save(root/'review.json',dict(schema='completed-smc-evidence-summary-v1',complete=True,
        physical_conclusion='unresolved historical failure',matched_mass_comparisons=[dict(passed=False)],
        input_sha256={r['path']:r['sha256'] for r in [auth,*analyses.values()]}))
    jobs=[]; audits={}; allocation=dict(initial_draws_per_population=1000,arms={},stages=128); compiled='a'*64
    for arm in ('narrow','large','broad'):
        allocation['arms'][arm]=dict(population=100)
        for original in historical['restricted_'+arm]['populations']:
            p=copy.deepcopy(original); ident=arm+'-'+p['id']; p['id']=ident
            directory=root/'restricted-populations'/ident
            cfg=save(directory/'provenance/config.json',config)
            closure={cfg['path']:cfg['sha256']}
            closure.update({str(root/'common'/name):digest for name,digest in bridge.STRATA_SOURCES.items()})
            for file,ref in [('region.json',target['region']),('shape.json',shape),('initial-reference-region.json',target['old_r5_region'])]:
                closure[str(directory/'provenance'/file)]=ref['sha256']
            p.update(schema='native-excluded-smc-population-audit-v1',complete=True,
                source_sha256=closure,independent_analytic_test_adapter=False,
                native_definition=dict(definition_sha256=target['native_definition']['sha256'],compiled_sha256=compiled,
                                       shape_sha256=shape['sha256'],fixed_poses=config['fixed_poses']))
            audits[ident]=save(root/'audits'/ident/'population.json',p)
            jobs.append(dict(id=ident,arm=arm,seed=p['seed'],output=str(directory)))
    restricted_plan=save(root/'restricted-plan.json',dict(schema='native-excluded-smc-fixed-control-v1',
        sources=bridge.STRATA_SOURCES,analysis=dict(class_order=['total',bridge.MAP['competing'],bridge.MAP['unbound']],
                                                  strata=bridge.current.BINS),
        allocation=allocation,jobs=jobs,physical_target=dict(shape_sha256=shape['sha256'],region_sha256=target['region']['sha256'],
            definition_sha256=target['native_definition']['sha256'],compiled_sha256=compiled,
            measure='Cartesian Angstrom cubed times normalized proper SO(3) Haar',
            target='Hcapture Hhard IR4 (1-Inative) exp(zC) d3t dHaar')))
    restricted_result=save(root/'restricted-analysis.json',dict(schema='native-excluded-smc-fixed-control-v1',complete=True,
        arms={arm:{} for arm in ('narrow','large','broad')},audits=audits,
        audited_source_sha256={r['path']:r['sha256'] for r in audits.values()},
        quality={'broad':dict(passed=False)},comparisons={'historical':dict(passed=False)},strata={},
        matching_importance_contact_comparisons={},physical_conclusion='unresolved',finite_R4_unbound_bound={'scope':'synthetic'}))
    exits=[dict(id=j['id'],status='complete',returncode=0) for j in jobs]
    restricted_status=save(root/'restricted-status.json',dict(complete=True,phase='complete',
        plan_sha256=restricted_plan['sha256'],analysis_sha256=restricted_result['sha256'],jobs=exits,audits=exits))
    cpops=[dict(id=f'p{i:02}',seed=1000+i) for i in range(8)]
    arm=dict(id='current',samples=1000,populations=cpops)
    stat_plan=dict(schema=bridge.current.PLAN_SCHEMA,target=target,target_and_regions_sha256=target_id,
                   strata=bridge.current.STRATA,gates=bridge.current.GATES,arms=[arm])
    statref=save(root/'statistics-plan.json',stat_plan)
    blocks={}
    for kind in ('Qz','Q0'):
        blocks[kind]={}
        for family,size in [(None,1),*bridge.current.BINS.items()]:
            entries=[]
            for index in range(size):
                pops=[toy_population(i%4,1000+i) for i in range(8)]
                for i,p in enumerate(pops): p['id']=f'p{i:02}'
                # Current Q0 has independently archived strata; historical Q0 does not.
                b=bridge.historical_batch('current',pops,kind,target_id,family=family if kind=='Qz' else None,
                                          index=index if kind=='Qz' else None)
                b['declaration']['target_and_regions_sha256']=bridge.scope_id(target_id,family,index)
                if kind=='Q0' and family is not None and index>0:
                    for row in b['population_records']:row['log_masses']={k:None for k in bridge.MAP}
                entries.append(b)
            if family is None: blocks[kind]['primary']=entries[0]
            else: blocks[kind].setdefault('strata',{})[family]=entries
    report=dict(schema=bridge.current.SCHEMA,complete=True,plan_sha256=statref['sha256'],
        target_descriptor=descriptor,target_and_regions_sha256=target_id,gates=bridge.current.GATES,
        arms={'current':dict(estimates=blocks)},input_sha256={r['path']:r['sha256'] for r in [statref,*target.values()]})
    reportref=save(root/'statistics.json',report)
    protocol=save(root/'protocol.json',dict(stat_plan,schema='native-class-physical-protocol-v1'))
    admission_plan=save(root/'admission-plan.json',dict(schema='native-class-physical-admission-plan-v1',
        protocol=protocol,statistics_plan=statref,statistics_result=reportref))
    admission=save(root/'admission.json',dict(schema='native-class-physical-admission-v1',complete=True,
        implementation_admission_passed=True,execution_lifecycle_passed=True,selected_full_geometry_gate_satisfied=True,
        plan_sha256=admission_plan['sha256'],protocol=protocol,statistics_result=reportref,
        target_and_regions_sha256=target_id,populations=[dict(arm='current',id=p['id'],samples=1000) for p in cpops],
        statistical_diagnostics=dict(failed_regional_checks=['synthetic']),missing_studies=['population_size'],
        input_sha256={r['path']:r['sha256'] for r in [protocol,statref,reportref,*target.values()]}))
    plan=dict(schema=bridge.PLAN_SCHEMA,current=dict(admission=admission,admission_plan=admission_plan),
        unrestricted=dict(authentication=auth,review=review,analyses=analyses),
        restricted=dict(plan=restricted_plan,status=restricted_status,analysis=restricted_result))
    return save(root/'plan.json',plan),target,shape['sha256'],report,historical


class BridgeTests(unittest.TestCase):
    def test_complete_summary_join_unequal_populations_and_historical_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            ref,target,shape,_,_=fixture(temp)
            with patches(target,shape): result=bridge.compare(ref['path'])
            self.assertEqual(len(result['comparisons']),5)
            self.assertEqual(len(result['historical_population_batches']),5)
            self.assertFalse(result['historical_verdicts_unchanged']['unrestricted']['matched_mass_comparisons'][0]['passed'])
            for pair in result['comparisons']:
                self.assertFalse(pair['kinds']['Q0']['strata_available'])
                self.assertEqual(pair['kinds']['Q0']['strata'],[])
                expected=set(bridge.RESTRICTED if pair['restricted'] else bridge.MAP)
                self.assertEqual(set(pair['kinds']['Qz']['regions']),expected)
                self.assertEqual(len(pair['kinds']['Qz']['strata']),70*len(expected))
                for value in pair['kinds']['Qz']['regions'].values():
                    self.assertEqual((value['left_population_count'],value['right_population_count']),(8,4))
            self.assertFalse(result['assembly_gate_open'])
            self.assertEqual(result['parsed_scientific_rows'],0)
            self.assertTrue(all(Path(p).suffix=='.json' for p in result['input_sha256']))

    def test_linear_estimator_zeros_and_physical_hard_mass(self):
        pops=[toy_population(i,10+i,zero=i==0) for i in range(4)]
        for p in pops:bridge.terminal_check(p,100)
        b=bridge.historical_batch('example',pops,'Qz','0'*64)
        stats=bridge.population.summarize_populations(b['declaration'],b['population_records'])
        self.assertAlmostEqual(stats['estimates']['native']['log_linear_mean'],math.log(4.5))
        self.assertEqual(stats['estimates']['native']['observed_zero_populations'],1)
        h=bridge.historical_batch('example',pops,'Q0','0'*64)
        self.assertEqual(h['population_records'][0]['log_masses']['total'],None)
        self.assertAlmostEqual(h['population_records'][1]['log_masses']['total'],math.log(2.))
        self.assertEqual(len(b['population_records']),4)
        with self.assertRaisesRegex(ValueError,'Q0 strata'):bridge.historical_batch('x',pops,'Q0','0'*64,family='radial',index=0)

    def test_real_terminal_profile_marker_and_actual_beta_bridge_are_distinct(self):
        for restricted in (False,True):
            good=toy_population(0,1,restricted)
            bridge.terminal_check(good,100,restricted)
            changes=[lambda p:p['terminal'].update(stage=128),
                     lambda p:p['terminal'].update(measure='fixed beta bridge target; not final physical mass'),
                     lambda p:p['stages'].pop(),
                     lambda p:p['stages'][64].update(stage=63),
                     lambda p:p['stages'][-1].update(beta=.99),
                     lambda p:p['stages'][-1].update(log_Z=1.),
                     lambda p:p.update(initial_hits=0)]
            for change in changes:
                with self.subTest(restricted=restricted,change=change):
                    bad=copy.deepcopy(good);change(bad)
                    with self.assertRaises(ValueError):bridge.terminal_check(bad,100,restricted)

    def test_zero_estimate_retains_initialization_without_invented_annealing(self):
        for restricted in (False,True):
            good=toy_population(0,1,restricted,zero=True)
            # Historical auditors may emit empty profile snapshots at scheduled
            # stages; these are not completed annealing transitions.
            good['profiles']={str(i):dict(stage=i,particles=0) for i in (0,16,32,128)}
            bridge.terminal_check(good,100,restricted)
            for change in [lambda p:p['stages'].append(dict(stage=128,beta=1.,log_Z=None)),
                           lambda p:p['stages'][0].update(beta=1.),
                           lambda p:p['stages'][0].update(zero_estimate=False),
                           lambda p:p.update(initial_hits=1),
                           lambda p:p.update(initial_physical_hard_log_masses=toy_population(0,1,restricted)['initial_physical_hard_log_masses'])]:
                bad=copy.deepcopy(good);change(bad)
                with self.assertRaises(ValueError):bridge.terminal_check(bad,100,restricted)

    def test_historical_stratum_and_native_partition_bindings_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            ref,target,shape,_,_=fixture(temp);plan=read(ref)
            uref=plan['unrestricted']['analyses']['broad'];original=read(uref)
            for change in [lambda d:d['analyzer_source_sha256'].clear(),
                           lambda d:d['analyzer_source_sha256'].update({'analyze_r4_smc_control.py':'0'*64}),
                           lambda d:d['protocol_snapshot']['source_and_input_sha256'].clear(),
                           lambda d:d['protocol_snapshot']['allocation'].update(stages_each=64)]:
                bad=copy.deepcopy(original);change(bad)
                with patches(target,shape),mock.patch.object(bridge.Metadata,'read',side_effect=lambda r:
                    bad if r==uref else read(r)):
                    with self.assertRaises(ValueError):bridge.load_unrestricted(plan['unrestricted'],read(target['region']),bridge.Metadata())
            pref=plan['restricted']['plan'];original=read(pref)
            for change in [lambda d:d['analysis'].update(strata={'radial':3,'angular':3,'orthant':32}),
                           lambda d:d['sources'].update({'analyze_native_excluded_smc.py':'0'*64}),
                           lambda d:d['allocation'].update(stages=64)]:
                bad=copy.deepcopy(original);change(bad)
                with patches(target,shape),mock.patch.object(bridge.Metadata,'read',side_effect=lambda r:
                    bad if r==pref else read(r)):
                    with self.assertRaises(ValueError):bridge.load_restricted(plan['restricted'],read(target['region']),bridge.Metadata())
            aref=next(iter(read(plan['restricted']['analysis'])['audits'].values()));bad=read(aref)
            bad['source_sha256']={k:v for k,v in bad['source_sha256'].items() if not k.endswith('analyze_r4_smc_control.py')}
            with patches(target,shape),mock.patch.object(bridge.Metadata,'read',side_effect=lambda r:
                bad if r==aref else read(r)):
                with self.assertRaisesRegex(ValueError,'historical analyze_r4'):
                    bridge.load_restricted(plan['restricted'],read(target['region']),bridge.Metadata())

    def test_exact_target_hash_scaffold_domain_measure_and_native_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            ref,target,shape,_,_=fixture(temp); plan=read(ref)
            original=read(plan['unrestricted']['analyses']['broad'])
            mutations=[lambda d:d['protocol_snapshot']['physical_target'].update(capture_radius=18.),
                       lambda d:d['physical_config'].update(fixed_poses=[]),
                       lambda d:d['protocol_snapshot']['physical_target'].update(measure='flat quaternion'),
                       lambda d:d.update(native_definition_sha256='0'*64),
                       lambda d:d['protocol_snapshot']['proposal'].update(reference_region_sha256='1'*64),
                       lambda d:d['protocol_snapshot']['physical_target'].update(region_sha256='2'*64)]
            for mutate in mutations:
                bad=copy.deepcopy(original);mutate(bad)
                with patches(target,shape), mock.patch.object(bridge.Metadata,'read',side_effect=lambda r:
                    bad if r==plan['unrestricted']['analyses']['broad'] else read(r)):
                    with self.assertRaises(ValueError):bridge.load_unrestricted(plan['unrestricted'],read(target['region']),bridge.Metadata())

    def test_dropped_population_reused_seed_and_terminal_fraction_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,_,report,historical=fixture(temp)
            for mutation in ('drop','duplicate_seed'):
                bad=copy.deepcopy(historical)
                if mutation=='drop':bad['unrestricted_broad']['populations'].pop()
                else:bad['restricted_narrow']['populations'][0]['seed']=bad['unrestricted_broad']['populations'][0]['seed']
                with self.assertRaises(ValueError):bridge.compare_reports(report,bad,report['target_and_regions_sha256'])
            p=toy_population(0,1);p['terminal']['log_masses']['total']=math.log(1.)
            with self.assertRaises(ValueError):bridge.terminal_check(p,100)
            p=toy_population(0,1);p['terminal']['log_strata']['orthant']['total'][0]+=.2
            with self.assertRaisesRegex(ValueError,'Strata'):bridge.terminal_check(p,100)

    def test_current_completion_tampered_metadata_and_seed_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ref,target,shape,report,historical=fixture(temp);plan=read(ref)
            admission=read(plan['current']['admission']);admission['execution_lifecycle_passed']=False
            with patches(target,shape),mock.patch.object(bridge.Metadata,'read',side_effect=lambda r:
                admission if r==plan['current']['admission'] else read(r)):
                with self.assertRaisesRegex(ValueError,'admission'):bridge.load_current(plan['current'],bridge.Metadata())
            Path(plan['current']['admission']['path']).write_text('{}\n')
            with patches(target,shape),self.assertRaisesRegex(ValueError,'Changed metadata'):bridge.compare(ref['path'])
            historical['unrestricted_broad']['populations'][0]['seed']=1000
            with self.assertRaisesRegex(ValueError,'seeds'):bridge.compare_reports(report,historical,report['target_and_regions_sha256'])

    def test_restricted_total_not_misidentified_as_full_total_and_unobserved_remains_unresolved(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,_,report,historical=fixture(temp)
            for group in historical.values():
                for p in group['populations']:
                    p['terminal']['log_masses']['unbound_no_native_entry']=None
                    for bins in p['terminal']['log_strata'].values():bins['unbound_no_native_entry']=[None]*len(bins['unbound_no_native_entry'])
            pairs=bridge.compare_reports(report,historical,report['target_and_regions_sha256'])
            for p in pairs:
                self.assertFalse(p['kinds']['Qz']['regions']['unbound']['observed'])
                if p['restricted']:
                    self.assertEqual(set(p['kinds']['Qz']['regions']),{'competing','unbound'})
                    self.assertNotIn('native',p['kinds']['Qz']['regions'])
                    self.assertNotIn('total',p['kinds']['Qz']['regions'])


if __name__=='__main__': unittest.main()
