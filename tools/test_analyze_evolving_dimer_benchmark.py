"""Synthetic geometric and journal controls; no protein sampling."""
import copy
import unittest
import analyze_evolving_dimer_benchmark as a


def pose(x): return dict(position=[float(x),0.,0.],orientation=[1.,0.,0.,0.])


def fixture(arm='local'):
    c=dict(contexts=[dict(root=1,child=3,anchor=2)],allocation=dict(warmup_blocks=1,production_blocks=3))
    j=dict(id=0,context_index=0,initialization='source',stream=0,arm=arm)
    initial=[pose(0),pose(2)]
    rows=[dict(kind='initial',block=0,job=j,selected=copy.deepcopy(initial),conditional_target=True,sampler_cpu_seconds=0.)]
    counts=dict(local_attempted=0,local_accepted=0,dimer_attempted=0,dimer_accepted=0,dimer_self_loop=0)
    for block in range(1,5):
        for attempt,slot in enumerate([0,1,0,1]):
            rows.append(dict(kind='local',block=block,attempt=attempt,member=[1,3][slot],old=copy.deepcopy(initial[slot]),
                proposed=pose(999),accepted=False,status='hard_rejected',retained=copy.deepcopy(initial),sampler_cpu_seconds=float(len(rows))))
            counts['local_attempted']+=1
        if arm!='local':
            rows.append(dict(kind='factorized_dimer',block=block,members=[1,3],anchor=2,old=copy.deepcopy(initial),
                retained=copy.deepcopy(initial),accepted=False,status='proposal_self_loop',proposal=dict(candidate=None),sampler_cpu_seconds=float(len(rows))))
            counts['dimer_attempted']+=1;counts['dimer_self_loop']+=1
        rows.append(dict(kind='retained_block',block=block,production=block>1,selected=copy.deepcopy(initial),raw=0,
            retained=0,counts=copy.deepcopy(counts),sampler_cpu_seconds=float(len(rows))))
    t=dict(complete=True,conditional_target=True,job=j,blocks=4,counts=counts,raw=0,retained=0,cpu_seconds=float(len(rows)))
    return c,j,initial,rows,t


class ConditionalTests(unittest.TestCase):
    def observer(self):
        return a.ConditionalObserver(dict(atoms=[dict(center=[0.,0.,0.],radius=.5)]),['p'],
            dict(boundary=dict(kind='spherical',radius=50.),box_lengths=[100.]*3,depletant_radius=.6),
            [pose(30),pose(0),pose(8),pose(2),pose(20)],[1,3])

    def test_joint_pose_internal_once_global_ids_exact_cache(self):
        o=self.observer();first=o.classify([pose(0),pose(2)])
        self.assertEqual(first['patch_tokens'],[(1,3,'p','p')]);self.assertEqual(o.calls,1)
        self.assertEqual(o.classify([pose(0),pose(2)]),first);self.assertEqual(o.calls,1);self.assertEqual(o.hits,1)
        second=o.classify([pose(6),pose(10)])
        self.assertEqual(second['patch_tokens'],[(1,2,'p','p'),(2,3,'p','p')]);self.assertFalse(second['internal_contact'])
        self.assertEqual(second['partners_by_member'],{'1':[2],'3':[2]});self.assertEqual(o.source[1],pose(0))

    def test_wall_and_hard_overlap(self):
        with self.assertRaisesRegex(ValueError,'wall'):self.observer().classify([pose(-51),pose(2)])
        with self.assertRaisesRegex(ValueError,'hard overlap'):self.observer().classify([pose(0),pose(.5)])

    def test_all_rejections_nulls_warmup_retained(self):
        for arm in ('local','unguided','m4'):
            c,j,s,rows,t=fixture(arm);points=a.validate_journal(rows,c,j,s,t)
            self.assertEqual([p['block'] for p in points],list(range(5)))
            self.assertTrue(all(p['selected']==s for p in points))

    def test_journal_defects_rejected(self):
        for defect in ('missing','duplicate','state','count','fatal','extra','cpu','source'):
            c,j,s,rows,t=fixture('m4')
            if defect=='missing':rows.pop(1)
            elif defect=='duplicate':rows.insert(2,copy.deepcopy(rows[1]))
            elif defect=='state':rows[1]['retained'][0]['position'][0]=.1
            elif defect=='count':rows[-1]['counts']['local_accepted']=1
            elif defect=='fatal':rows[1]['fatal_error']='test'
            elif defect=='extra':rows.append(copy.deepcopy(rows[-1]))
            elif defect=='cpu':rows[2]['sampler_cpu_seconds']=-1
            else:rows[1]['old']['position'][0]=.1
            with self.subTest(defect=defect),self.assertRaises(ValueError):a.validate_journal(rows,c,j,s,t)

    def test_mh_and_counts(self):
        c,j,s,rows,t=fixture();row=rows[1]
        row.update(status='completed',accepted=True,proposed=copy.deepcopy(s[0]),log_u=-2.,log_acceptance_ratio=-1.,bath=dict(raw_points=3,retained_points=1))
        for r in rows:
            if r['kind']=='retained_block':r['counts']['local_accepted']=1;r['raw']=3;r['retained']=1
        t['counts']['local_accepted']=1;t['raw']=3;t['retained']=1
        a.validate_journal(rows,c,j,s,t);row['log_u']=-.5
        with self.assertRaisesRegex(ValueError,'Acceptance'):a.validate_journal(rows,c,j,s,t)

    def test_constant_ess_null(self):
        o=self.observer();trace=[dict(o.classify([pose(0),pose(2)]),block=i) for i in range(5)]
        summary=a.summarize_trace(trace,1,20.);self.assertEqual(summary['production_samples'],3)
        for name in ('fingerprint_ess','partner_ess','patch_ess'):self.assertIsNone(summary[name]['apparent_ess'])
        self.assertEqual(summary['fingerprint']['completed_passages'],0)

    def test_categorical_returns_and_label_invariance(self):
        labels=['a','a','b','b','a','c','a'];summary=a.categorical_summary(labels,list(range(7)))
        self.assertEqual(summary['completed_passages'],4);self.assertEqual(len(summary['completed_returns']),2)
        self.assertEqual(summary['occupancy'],{'a':4/7,'b':2/7,'c':1/7})
        x,_=a.presence_matrix([[(v,)] for v in labels]);y,_=a.presence_matrix([[(dict(a='z',b='x',c='y')[v],)] for v in labels])
        self.assertAlmostEqual(a.apparent_effective_count(x,1)['apparent_ess'],a.apparent_effective_count(y,1)['apparent_ess'])

    def test_comparisons_never_pool_targets(self):
        rows=[]
        for context in (0,1):
            for arm in ('local','m4'):
                for init in ('source','proposal_prepared'):
                    for stream in (0,1):
                        rows.append(dict(job=dict(context_index=context,arm=arm,initialization=init,stream=stream),metrics=dict(
                            fingerprint_ess=dict(apparent_ess=None,apparent_ess_per_sampling_CPU_second=None),internal_contact_fraction=1.,
                            fingerprint=dict(completed_passages=0,completed_returns=[],occupancy={'x':1.}))))
        result=a.comparison_summaries(rows);self.assertEqual(len(result['groups']),8)
        self.assertTrue(all(len(r['streams'])==2 for r in result['groups']))
        self.assertTrue(all(r['left']['context_index']==r['right']['context_index'] and r['left']['stream']==r['right']['stream'] for r in result['descriptive_paired_comparisons']))

if __name__=='__main__':unittest.main()
