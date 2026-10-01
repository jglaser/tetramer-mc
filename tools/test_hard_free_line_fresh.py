import io
import json
import unittest
from pathlib import Path
import tempfile
from prepare_hard_free_line_fresh import seeds_in,SEEDS,jobs
from observe_hard_free_line_fresh import classify_rows,PARTS
from report_hard_free_line_fresh import fraction,fallback_reason
from run_hard_free_line_fresh import steps
from run_full_vessel_comparison import execute_group


class Native:
    def __init__(self):self.calls=0
    def classify(self,pose):
        self.calls+=1;native=pose=='native';triangle=pose=='triangle'
        return dict(native_any=native or triangle,native_anchor_count=2 if triangle else int(native),
                    cooperative_entry=triangle,registry_consistent_triangle=triangle)
class Contact:
    def __init__(self):self.calls=0
    def classify(self,pose):
        self.calls+=1;contact=pose in ('native','triangle','contact')
        return dict(exclusion_contact=contact,anchors=[dict(exclusion_contact=contact)]*2)


def row(i,pose='unbound',valid=True,shell=True,branch='original_gaussian'):
    return dict(id=i,kind='fresh',pose=pose,hard_valid=valid,shell_valid=shell,capture_valid=True,
        draw=dict(component=None if branch=='uniform' else 0,conditional=branch.startswith('conditioned'),
                  fallback=branch=='conditioned_fallback'),width_contacts=[[]])


class FreshBookkeeping(unittest.TestCase):
    def test_fixed_independent_allocation_and_seed_extraction(self):
        plan=jobs();self.assertEqual(len(plan),8);self.assertEqual(sum(j['samples'] for j in plan),2048)
        self.assertEqual(len(set(SEEDS)),8)
        self.assertEqual(seeds_in(dict(seed=10,rng_seed='20',seeds=[30,40],command=['--seed','50'],a=[dict(seed=60)])),
                         {10,20,30,40,50,60})

    def test_valid_rows_classified_exactly_once_invalid_retained(self):
        n,c,stream=Native(),Contact(),io.StringIO()
        records=[row(0,valid=False,branch='uniform'),row(1,shell=False),row(2,'native'),
                 row(3,'contact',branch='conditioned_success'),row(4),row(5,'triangle',branch='conditioned_fallback')]
        result=classify_rows(records,n,c,stream,{})
        saved=[json.loads(s) for s in stream.getvalue().splitlines()]
        self.assertEqual(result['partition'],dict(zip(PARTS,[1,1,2,1,1])))
        self.assertEqual((n.calls,c.calls,result['valid']),(4,4,4));self.assertEqual(len(saved),6)
        self.assertIsNone(saved[0]['classification']);self.assertIsNone(saved[1]['contact'])
        self.assertEqual(result['events']['exclusion_contact'],3)
        self.assertEqual(result['events']['registry_triangle'],1)
        self.assertFalse(result['width_contacts_used']);self.assertNotIn('width_contacts',saved[0])
        self.assertEqual(result['branches']['conditioned_success']['attempts'],1)

    def test_empty_width_sentinel_cannot_create_contact(self):
        n,c,s=Native(),Contact(),io.StringIO()
        result=classify_rows([row(0)],n,c,s,{})
        self.assertEqual(result['events']['exclusion_contact'],0)
        self.assertEqual(result['partition']['valid_unbound_without_native_entry'],1)

    def test_attempt_and_flag_tampering_rejected(self):
        for bad in (dict(row(0),id=1),dict(row(0),id=False),dict(row(0),hard_valid=1),dict(row(0),kind='probe')):
            with self.assertRaises(ValueError):classify_rows([bad],Native(),Contact(),io.StringIO(),{})

    def test_population_SE_keeps_unconditional_denominators(self):
        value=fraction([0,1,2,3],[256]*4)
        self.assertEqual(value['count'],6);self.assertEqual(value['denominator'],1024)
        self.assertAlmostEqual(value['fraction'],6/1024)
        self.assertGreater(value['population_SE'],0)
        with self.assertRaises(ValueError):fraction([1,2],[256,256])

    def test_selected_fallback_reasons_not_contact_sentinel(self):
        self.assertIsNone(fallback_reason(dict(fallback=False)))
        self.assertEqual(fallback_reason(dict(fallback=True,geometry=dict(empty_reason='outside_R4'))),'outside_R4')
        self.assertEqual(fallback_reason(dict(fallback=True,geometry=dict(hard_free_intervals=[]))),'empty_hard_free_set')
        self.assertEqual(fallback_reason(dict(fallback=True,geometry=dict(hard_free_intervals=[dict(lower=0,upper=1)]))),
            'positive_geometry_mass_at_or_below_floor')


class FreshDispatch(unittest.TestCase):
    def exercise(self,root,fail_at=None):
        (root/'logs').mkdir();plan=steps(root,'/python');launched=[]
        class Child:
            pid=999999
            def __init__(self,code):self.code=code
            def poll(self):return self.code
            def wait(self):return self.code
        def launch(command,**kwargs):
            item=plan[len(launched)];launched.append(command);destination=Path(item['directory'])
            if item['kind']=='audit':
                self.assertTrue(Path(item['population_directory']).is_dir());destination.write_text('{}')
            else:destination.mkdir(parents=True)
            return Child(int(len(launched)-1==fail_at))
        try:
            execute_group(plan,lambda:None,1,root,{},popen=launch,pause=lambda _:None,
                capacity=lambda _:dict(workers=0,physical_pids=[]),birth=lambda _:0)
        except RuntimeError:
            if fail_at is None:raise
        return plan,launched

    def test_actual_executor_unique_absent_destinations_and_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,launched=self.exercise(Path(directory))
            self.assertEqual(len(launched),24);self.assertTrue(all(j['status']=='complete' for j in plan))
            self.assertEqual(len({j['directory'] for j in plan}),24)
            proposals=[s for s in plan if s['kind']=='proposal']
            self.assertEqual([int(s['command'][s['command'].index('--seed')+1]) for s in proposals],SEEDS)
            self.assertTrue(all(s['command'][s['command'].index('--samples')+1]=='256' and '--probes' not in s['command'] for s in proposals))

    def test_failure_preserves_started_attempt_and_drains_without_refill(self):
        with tempfile.TemporaryDirectory() as directory:
            plan,launched=self.exercise(Path(directory),1)
            self.assertEqual(len(launched),2);self.assertEqual(plan[0]['status'],'complete')
            self.assertEqual(plan[1]['status'],'failed');self.assertTrue(all(j['status']=='not_started' for j in plan[2:]))
            self.assertTrue(Path(plan[1]['directory']).exists());self.assertTrue(Path(plan[1]['log']).exists())


if __name__=='__main__':unittest.main()
