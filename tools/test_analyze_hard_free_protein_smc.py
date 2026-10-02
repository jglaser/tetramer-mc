"""Synthetic estimator/accounting tests: no protein or Poisson sampling."""
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import analyze_hard_free_protein_smc as analysis


class Observer:
    def __init__(self):self.seen=set();self.classifier_calls=0
    def classify(self,pose,valid):
        assert valid
        label=pose['label']
        if label not in self.seen:self.classifier_calls+=1;self.seen.add(label)
        classes=dict(total=True,registered_native_entry=label in ('old','remaining'),
            old_R5_intersection_native=label=='old',remaining_R4_native=label=='remaining',
            contact_no_native_entry=label=='competing',unbound_no_native_entry=label=='unbound')
        return dict(classes=classes,strata=dict(radial=0,angular=0,orthant=0))


def particle(label,ancestor):return dict(pose=dict(label=label),initial_ancestor=ancestor)


def fixture(z=16.):
    first=[particle('old',0),particle('old',0),particle('remaining',1),particle('competing',2)]
    last=[particle('old',0),particle('remaining',0),particle('remaining',1),particle('competing',2)]
    stages=[dict(stage=0,particles=first,log_Z=0.,ancestry={}),
        dict(stage=1,particles=last,log_Z=math.log(z),ancestry={})]
    summary=dict(zero_estimate=False,completed_stage=1,terminal_particles=last,log_Z=math.log(z))
    return stages,summary


def population(index,z=16.,zero=False):
    if zero:
        summary=dict(zero_estimate=True,completed_stage=0,terminal_particles=[],log_Z=None,ancestry={})
        stages=[dict(stage=0,particles=[],log_Z=None)]
    else:stages,summary=fixture(z)
    result=analysis.profile_stages(iter(stages),summary,Observer(),io.StringIO(),[0,1])
    result.update(id=f'r{index:02}',seed=100+index,zero_estimate=zero,initial_draws=32)
    return result


def iid_arm(pops,seed_base):
    records=[dict(id=p['id'],seed=seed_base+i,samples=32) for i,p in enumerate(pops)]
    def estimate(region,family=None,index=None):
        values=[]
        for p,r in zip(pops,records):
            log=(p['terminal']['log_masses'][region] if family is None
                else p['terminal']['log_strata'][family][region][index])
            values.append(dict(id=r['id'],seed=r['seed'],draws=32,log_Qz=log,log_Q0=log))
        return dict(populations=values)
    return dict(populations=records,allocation=dict(samples=32),
        estimates={r:estimate(r) for r in analysis.old.CLASSES},
        strata={f:{r:[estimate(r,f,i) for i in range(n)] for r in analysis.old.CLASSES}
            for f,n in analysis.old.STRATA.items()})


class ContactAnalysisTests(unittest.TestCase):
    def test_statistical_code_binding_ignores_authentication_but_rejects_estimator_changes(self):
        with tempfile.TemporaryDirectory() as name:
            a,b = Path(name)/'a.py',Path(name)/'b.py'
            functions = '\n'.join('def '+key+'():\n    return 3\n' for key in analysis.STATISTICAL_FUNCTIONS)
            a.write_text(functions+'\ndef authenticate():\n    return False\n')
            b.write_text(functions+'\ndef authenticate():\n    return True\n')
            self.assertEqual(analysis.statistical_code(a),analysis.statistical_code(b))
            b.write_text(functions.replace('def compare():\n    return 3','def compare():\n    return 4'))
            self.assertNotEqual(analysis.statistical_code(a),analysis.statistical_code(b))
            b.write_text(functions.replace('def compare():','def omitted_compare():'))
            with self.assertRaisesRegex(ValueError,'Incomplete statistical implementation'):
                analysis.statistical_code(b)

    def test_multiplicity_joint_normalizer_and_replenished_family_counts(self):
        stages,summary=fixture();observer=Observer();stream=io.StringIO()
        result=analysis.profile_stages(iter(stages),summary,observer,stream,[0,1]);t=result['terminal']
        self.assertEqual(t['particles'],4)
        self.assertEqual(t['counts']['registered_native_entry'],3)
        self.assertAlmostEqual(math.exp(t['log_masses']['registered_native_entry']),12.)
        self.assertAlmostEqual(math.exp(t['log_masses']['contact_no_native_entry']),4.)
        self.assertEqual(t['descendants_from_initially_other_class']['remaining_R4_native'],1)
        self.assertEqual(t['class_distinct_initial_families']['remaining_R4_native'],2)
        self.assertEqual(observer.classifier_calls,3)
        self.assertEqual(len(stream.getvalue().splitlines()),8)
        self.assertEqual(result['terminal_label_alias_stage'],1)

    def test_completed_zero_has_no_refill_or_classifier_calls(self):
        observer=Observer();summary=dict(zero_estimate=True,completed_stage=0,terminal_particles=[],log_Z=None,ancestry={})
        result=analysis.profile_stages(iter([dict(stage=0,particles=[],log_Z=None)]),summary,observer,io.StringIO(),[0,1])
        self.assertEqual(set(result['profiles']),{'0'})
        self.assertTrue(all(v is None for v in result['terminal']['log_masses'].values()))
        self.assertEqual(observer.classifier_calls,0)

    def test_missing_profile_changed_endpoint_or_missing_genealogy_refused(self):
        for mode in ('profile','endpoint','genealogy'):
            with self.subTest(mode=mode):
                stages,summary=fixture()
                selected=[0,1,2] if mode=='profile' else [0,1]
                if mode=='endpoint':summary['log_Z']+=.1
                if mode=='genealogy':stages[1].pop('ancestry')
                with self.assertRaises(ValueError):analysis.profile_stages(iter(stages),summary,Observer(),io.StringIO(),selected)

    def test_hash_binds_exact_parsed_stage_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'stages.jsonl';stages,_=fixture()
            path.write_text(''.join(json.dumps(v)+'\n' for v in stages));digest=analysis.sha(path)
            self.assertEqual(list(analysis.stage_rows(path,digest)),stages)
            path.write_text(path.read_text()+'\n')
            with self.assertRaises((ValueError,json.JSONDecodeError)):list(analysis.stage_rows(path,digest))

    def test_all_audits_must_complete_and_ids_cannot_be_duplicated(self):
        state=dict(complete=True,phase='audited_awaiting_classification',plan_sha256='x',
            jobs=[dict(id=f'r{i:02}',status='complete',returncode=0) for i in range(4)],
            audits=[dict(id=f'r{i:02}',status='complete',returncode=0) for i in range(4)])
        analysis.require_audited_status(state,'x')
        for mode in ('unfinished','failed','duplicate','hash'):
            bad=copy.deepcopy(state)
            if mode=='unfinished':bad['complete']=False
            if mode=='failed':bad['audits'][0]['returncode']=1
            if mode=='duplicate':bad['jobs'][1]['id']='r00'
            if mode=='hash':bad['plan_sha256']='y'
            with self.subTest(mode=mode),self.assertRaises(ValueError):analysis.require_audited_status(bad,'x')

    def test_linear_population_estimator_retains_zero_and_rejects_endpoint_fraction(self):
        pops=[population(i,z,zero=i==2) for i,z in enumerate([8.,16.,1.,4.])]
        physical=dict(allocation=dict(populations=4,population=4,initial_draws=32),
            jobs=[dict(id=p['id'],seed=p['seed']) for p in pops])
        view=analysis.smc_view(pops,physical);rows=analysis.smc_rows(view,'Qz')
        stats=analysis.summarize_rows(rows)
        self.assertAlmostEqual(math.exp(stats['total']['log_Q']),7.)
        self.assertAlmostEqual(math.exp(stats['registered_native_entry']['log_Q']),5.25)
        self.assertEqual(stats['total']['nonzero_populations'],3)
        empty=analysis.compare_rows(rows,rows)['comparisons']['unbound_no_native_entry']
        self.assertFalse(empty['passed']);self.assertIn('Unobserved',empty['unresolved'])
        bad=copy.deepcopy(view);bad['populations'][0]['terminal']['log_masses']['registered_native_entry']=math.log(.75)
        with self.assertRaisesRegex(ValueError,'normalizer'):analysis.smc_rows(bad,'Qz')

    def test_full_comparison_retains_all_three_references_and_all_strata(self):
        pops=[population(i,z,zero=i==2) for i,z in enumerate([8.,16.,1.,4.])]
        physical=dict(allocation=dict(populations=4,population=4,initial_draws=32),
            jobs=[dict(id=p['id'],seed=p['seed']) for p in pops],analysis=analysis.control.ANALYSIS)
        historical=analysis.smc_view(copy.deepcopy(pops),copy.deepcopy(physical))
        for p,j in zip(historical['populations'],historical['protocol_snapshot']['jobs']):
            p['seed']+=300;j['seed']+=300
        iid=dict(arms=dict(baseline=iid_arm(pops,500),conditioned=iid_arm(pops,600)))
        result=analysis.compare(pops,physical,historical,iid)
        self.assertEqual(set(result['comparisons']),{'historical_broad','larger_iid_baseline','larger_iid_conditioned'})
        for ref in result['comparisons'].values():
            self.assertEqual(len(ref['strata']),(3+3+64)*6)
            self.assertTrue(ref['comparisons']['registered_native_entry']['passed'])
            self.assertFalse(ref['comparisons']['unbound_no_native_entry']['passed'])
        self.assertEqual(result['failed_material_strata'],[])
        self.assertFalse(result['assembly_gate_open']);self.assertFalse(result['full_vessel_gate_open'])
        # Moving a stratum mass without its counterpart must fail partitioning,
        # even though the region's aggregate mass has not changed.
        bad=copy.deepcopy(iid)
        bad['arms']['baseline']['strata']['orthant']['registered_native_entry'][0]['populations'][0]['log_Qz']+=.3
        with self.assertRaisesRegex(ValueError,'strata'):analysis.compare(pops,physical,historical,bad)


if __name__=='__main__':unittest.main()
