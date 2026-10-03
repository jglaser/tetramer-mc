import copy
import json
import unittest
from prepare_factorized_dimer_physical import cache_rows


def row(candidate=None):
    return dict(status='completed',atlas='blind',atlas_index=0,case=dict(name='case',root=0,child=1,anchor=2),
                case_index=0,attempt=0,method='whole_joint',old=[{},{}],anchor_pose={},proposal_cpu_seconds=.1,
                contact_diagnostic_cpu_seconds=.2,raw_edge_draws=64,
                outcome=dict(candidate=candidate,status='cap_exhausted' if candidate is None else 'candidate'))


class CacheTests(unittest.TestCase):
    def test_all_rows_and_candidates_retained_without_outcome_selection(self):
        rows=[row(),row(dict(root='newroot',child='newchild',diagnostics={'log_reverse_forward':'-inf'})),row()]
        rows[1]['method']='factorized'
        result=cache_rows([json.dumps(r) for r in rows])
        self.assertEqual(len(result),3)
        self.assertEqual([r['index'] for r in result],[0,1,2])
        self.assertEqual([r['candidate'] for r in result],[r['outcome']['candidate'] for r in rows])
        self.assertEqual(result[1]['candidate']['diagnostics']['log_reverse_forward'],'-inf')

    def test_fatal_rows_never_become_physical_rejections(self):
        for status in ('fatal','fatal_diagnostic'):
            r=row();r['status']=status
            with self.assertRaises(ValueError):cache_rows([json.dumps(r)])

    def test_inconsistent_null_and_unknown_method_fail_closed(self):
        for change in ('candidate','method','timing'):
            r=copy.deepcopy(row())
            if change=='candidate':r['outcome']['status']='candidate'
            elif change=='method':r['method']='selected_best'
            else:r['proposal_cpu_seconds']=-1
            with self.assertRaises(ValueError):cache_rows([json.dumps(r)])


if __name__=='__main__':unittest.main()
