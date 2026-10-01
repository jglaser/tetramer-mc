import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import compare_hard_free_line_smc as bridge


def population(i,zero=False):
    counts=dict(total=0 if zero else 4,contact_no_native_entry=0 if zero else 3,unbound_no_native_entry=0 if zero else 1)
    logs={k:None if n==0 else math.log(8*n/4) for k,n in counts.items()}
    return dict(id='r'+str(i),seed=100+i,terminal=dict(particles=counts['total'],counts=counts,log_masses=logs,
        log_strata={f:{k:[value]+[None]*(size-1) for k,value in logs.items()} for f,size in bridge.STRATA.items()}),
        initial_physical_hard_log_masses=logs)


def identity_fixture():
    t=dict(shape_sha256='shape',region_sha256='R4',native_definition_sha256='native',reference_region_sha256='R5',
        supplemental_definition_sha256='partition',config=dict(shape='new/path',fixed_poses=[1,2],capture_radius=170,activity=.035),seeds=[9])
    s=dict(schema='smc-r4-control-analysis-v1',complete=True,native_definition_sha256='native',
        physical_config=dict(t['config'],shape='old/path'),populations=[dict(seed=10)],
        protocol_snapshot=dict(physical_target=dict(shape_sha256='shape',region_sha256='R4',
            measure='d3t in Angstrom cubed times normalized proper SO(3) Haar'),
            proposal=dict(reference_region_sha256='R5'),source_and_input_sha256={'/old/native-partition-definition.json':'partition'}))
    return s,t


class MatchingSMC(unittest.TestCase):
    def test_linear_population_mean_and_independent_SE_include_zeros(self):
        value=bridge.mass_statistics([0.,math.log(3),None,math.log(4)])
        self.assertAlmostEqual(math.exp(value['log_Q']),2.)
        se=math.sqrt(sum((x-2)**2 for x in [1,3,0,4])/(4*3))
        self.assertAlmostEqual(value['population_relative_SE'],se/2)
        self.assertEqual(value['populations'],4);self.assertEqual(value['nonzero_populations'],3)

    def test_unobserved_is_unresolved_not_zero_or_bound(self):
        empty=bridge.mass_statistics([None]*4);positive=bridge.mass_statistics([0.]*4)
        result=bridge.mass_comparison(empty,positive)
        self.assertFalse(result['passed']);self.assertIn('Unobserved',result['unresolved'])
        self.assertIsNone(empty['log_Q']);self.assertIsNone(empty['population_relative_SE'])
        self.assertNotIn('upper_bound',result)

    def test_linear_combined_SE_and_absolute_rule_are_separate(self):
        a=dict(log_Q=math.log(2),population_relative_SE=.3)
        b=dict(log_Q=0.,population_relative_SE=.4)
        value=bridge.mass_comparison(a,b)
        self.assertAlmostEqual(value['scaled_linear_difference'],.5)
        self.assertAlmostEqual(value['scaled_linear_SE'],math.hypot(.3,.2))
        self.assertTrue(value['within_three_SE']);self.assertFalse(value['within_absolute_limit']);self.assertFalse(value['passed'])

    def test_terminal_Zhat_times_indicator_and_all_strata_conserve(self):
        pops=[population(i,i==3) for i in range(4)]
        rows=bridge.restricted_rows(pops,'Qz')
        self.assertAlmostEqual(math.exp(rows[0]['contact_no_native_entry']),6)
        self.assertIsNone(rows[3]['contact_no_native_entry'])
        bad=copy.deepcopy(pops);bad[0]['terminal']['log_masses']['contact_no_native_entry']=math.log(.75)
        with self.assertRaisesRegex(ValueError,'Zhat'):bridge.restricted_rows(bad,'Qz')
        with self.assertRaisesRegex(ValueError,'not archived'):bridge.restricted_rows(pops,'Q0','radial',0)
        bad=copy.deepcopy(pops);bad[0]['terminal']['log_strata']['orthant']['total'][0]=0.
        with self.assertRaisesRegex(ValueError,'strata'):bridge.restricted_rows(bad,'Qz')

    def test_model_domain_measure_native_partition_mismatch_rejected(self):
        s,t=identity_fixture();self.assertEqual(bridge.validate_unrestricted(s,t),[])
        cases=[('shape_sha256','other'),('region_sha256','other'),('reference_region_sha256','other'),
               ('native_definition_sha256','other'),('supplemental_definition_sha256','other')]
        for key,value in cases:
            bad=copy.deepcopy(t);bad[key]=value
            with self.assertRaises(ValueError):bridge.validate_unrestricted(s,bad)
        bad=copy.deepcopy(s);bad['protocol_snapshot']['physical_target']['measure']='unnormalized Haar'
        with self.assertRaisesRegex(ValueError,'measure'):bridge.validate_unrestricted(bad,t)
        bad=copy.deepcopy(s);bad['physical_config']['capture_radius']=180
        with self.assertRaisesRegex(ValueError,'Physical configuration'):bridge.validate_unrestricted(bad,t)
        bad=copy.deepcopy(s);bad['populations'][0]['seed']=9
        with self.assertRaisesRegex(ValueError,'streams'):bridge.validate_unrestricted(bad,t)

    def test_summary_hash_mismatch_fails_without_history_replay(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'population.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):bridge.Inputs().json(p,'0'*64)
            ledger=bridge.Inputs();self.assertEqual(ledger.json(p),{})
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'changed'):ledger.recheck()

    def test_restricted_strata_source_definition_and_population_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'common').mkdir()
            p=root/'common/chart.py';p.write_text('reviewed edges and >=0 signs')
            hashes={'chart.py':bridge.sha(p)}
            plan=dict(schema='native-excluded-smc-fixed-control-v1',sources=hashes,
                analysis=dict(class_order=['total',*bridge.RESTRICTED_CLASSES],strata=bridge.STRATA))
            analysis=dict(schema=plan['schema'],complete=True);target=dict(strata=bridge.STRATA_DEFINITION)
            with patch.object(bridge,'RESTRICTED_STRATA_SOURCES',hashes):
                binding=bridge.restricted_definition(root,plan,analysis,target,bridge.Inputs())
                bridge.restricted_population_strata_binding(dict(source_sha256=binding['source_sha256']),binding)
                with self.assertRaisesRegex(ValueError,'population stratum'):
                    bridge.restricted_population_strata_binding(dict(source_sha256={}),binding)
                for key,value in [('schema','unknown'),('complete',False)]:
                    with self.assertRaisesRegex(ValueError,'restricted summary'):
                        bridge.restricted_definition(root,plan,dict(analysis,**{key:value}),target,bridge.Inputs())
                bad=copy.deepcopy(plan);bad['analysis']['class_order'].reverse()
                with self.assertRaisesRegex(ValueError,'class order'):
                    bridge.restricted_definition(root,bad,analysis,target,bridge.Inputs())
                bad=copy.deepcopy(target);bad['strata']['radial_edges'][1]=1.9
                with self.assertRaisesRegex(ValueError,'stratum definitions'):
                    bridge.restricted_definition(root,plan,analysis,bad,bridge.Inputs())
                p.write_text('changed edges or signs')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    bridge.restricted_definition(root,plan,analysis,target,bridge.Inputs())


if __name__=='__main__':unittest.main()
