import copy
import json
import math
import unittest
from prepare_auxiliary_overlap_physical import cache_rows


def row(candidate=False):
    aux=math.log(11)-math.log(5)
    return dict(status='completed',atlas='blind',atlas_index=0,case=dict(name='case',root=0,child=1,anchor=2),case_index=0,attempt=0,method='m1',m=1,old=[{},{}],anchor_pose={},proposal_cpu_seconds=.1,contact_diagnostic_cpu_seconds=.2,raw_edge_draws=64,
        cloud_construction_cpu_seconds=.3,guidance_setup_cpu_seconds=.05,standalone_proposal_cpu_seconds=.45,
        complete_log_correction=-3+aux if candidate else None,
        outcome=dict(candidate=dict(root='r',child='c',diagnostics=dict(log_reverse_forward=-3.)) if candidate else None,status='candidate' if candidate else 'cap_exhausted',
            guidance=dict(m=1,point_count=20,old_count=10,new_count=4 if candidate else None,threshold=1,integer_draws=[1],aux_log_correction=aux if candidate else None)))


class CacheTests(unittest.TestCase):
    def test_nulls_retained_and_separate_factors_preserved(self):
        raw=[row(),row(True),row()];cached=cache_rows([json.dumps(r) for r in raw])
        self.assertEqual(len(cached),3);self.assertEqual([r['index'] for r in cached],[0,1,2])
        self.assertEqual([r['candidate'] for r in cached],[r['outcome']['candidate'] for r in raw]);self.assertEqual(cached[1]['guidance'],raw[1]['outcome']['guidance'])
        self.assertEqual(cached[1]['standalone_proposal_cpu_seconds'],.45)

    def test_fatal_or_inconsistent_rows_cannot_become_rejections(self):
        for kind in ['fatal','status','method','timing','count','integer','omitted','double','null_aux','m']:
            r=row(True)
            if kind=='fatal':r['status']='fatal'
            elif kind=='status':r['outcome']['status']='cap_exhausted'
            elif kind=='method':r['method']='selected_best'
            elif kind=='timing':r['standalone_proposal_cpu_seconds']=-1.
            elif kind=='count':r['outcome']['guidance']['new_count']=21
            elif kind=='integer':r['outcome']['guidance']['integer_draws']=[11]
            elif kind=='omitted':r['complete_log_correction']=-3.
            elif kind=='double':r['complete_log_correction']=-3+2*r['outcome']['guidance']['aux_log_correction']
            elif kind=='null_aux':r=row();r['outcome']['guidance']['aux_log_correction']=0.
            else:r['m']=4
            with self.subTest(kind=kind),self.assertRaises(ValueError):cache_rows([json.dumps(r)])

    def test_zero_reverse_density_is_a_retained_valid_candidate(self):
        r=row(True);r['complete_log_correction']='-inf';r['outcome']['candidate']['diagnostics']['log_reverse_forward']='-inf'
        cached=cache_rows([json.dumps(r)])[0];self.assertEqual(cached['complete_log_correction'],'-inf')


if __name__=='__main__':unittest.main()
