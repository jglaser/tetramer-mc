"""Bridge extension controls: synthetic poses/counts only, no physical queries."""
import copy
import hashlib
import json
import math
import unittest

import numpy as np

import analyze_context_bridge_physical as m
from test_context_broadened_physical import score_fixture
from test_context_multicage_physical import ledger_fixture


def reference():
    return [[77,217,f'p{i}','q'] for i in range(16)]


def jobs():
    return [dict(id=f'{a}-{p}-{c}',stratum_id=f'{a}-{p}-{c}',comparison_arm=a,population_index=p,
        component=c,attempts=n) for a,counts in m.ALLOCATIONS.items() for p in range(4) for c,n in counts.items()]


def record(ordinal, valid=True, region='A_patch_complete', weight=1., hard=1., radius=0., population=0):
    logs=np.linspace(-1.,1.,15)
    q=float(m.arm_log_q(logs[:,None],m.COEFFICIENTS['bridge'])[0])
    position=dict(position=[float(ordinal),0.,0.],orientation=[1.,0.,0.,0.])
    tokens=reference() if region=='A_patch_complete' else []
    if region=='A_patch_complete':tokens=[[16,77,'a','b']]+tokens
    neighbors=[16,217] if region=='A_patch_complete' else []
    original=dict(input=dict(ordinal=ordinal,proposed_pose=position),complete=True,
        actual=dict(physical_valid=valid,wall_valid=valid,core_valid=True if valid else None),
        region=region if valid else None,clouds=[],physical_weight_status='not_estimated',
        patches=dict(tokens=tokens,neighbor_labels=neighbors,source_fraction=1. if tokens else 0.) if valid else None)
    digest=hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    row=dict(ordinal=ordinal,comparison_arm='bridge',population_index=population,component='bridge_a0_b4',branch='source',
        row_id=dict(stratum_id=f'bridge-{population}',ordinal=ordinal),physical_valid=valid,
        wall_valid=valid,core_valid=True if valid else None,region=region if valid else 'hard_invalid',
        source_T_complete_all_regions=valid and region=='A_patch_complete',exact_A=valid and region=='A_patch_complete',
        exact_source_intersection_count=(16 if tokens else 0) if valid else None,
        source_fraction=(1. if tokens else 0.) if valid else None,neighbor_labels=neighbors if valid else None,
        proposed_pose=position,log_q_arm=q,log_physical_weight=math.log(weight) if valid else None,
        log_auxiliary_variance=None,log_u=float(logs[0]),log_g=float(logs[1]),
        log_source={c:float(v) for c,v in zip(m.CHARTS,logs[2:])},
        source_rows=dict(path='/synthetic/rows.jsonl',sha256='a'*64),canonical_source_row_sha256=digest,
        mahalanobis_squared={c:radius for c in m.CHARTS})
    return row,original


def panel_fixture():
    pairs=[record(0),record(1,False),record(2,region='unbound')]
    rows,originals=map(list,zip(*pairs))
    entries=[dict(ordinal=i,source_rows=rows[i]['source_rows'],pose=rows[i]['proposed_pose'],
        metadata=dict(log_q_balanced=rows[i]['log_q_arm'],original_row=originals[i])) for i in (0,2)]
    return dict(attempts=3,empty=False),dict(entries=entries),rows,originals,reference()


class BridgePhysicalTests(unittest.TestCase):
    def test_exact72_strata_and2048_denominators(self):
        inventory=jobs();m.validate_inventory(inventory)
        self.assertEqual(sum(j['attempts'] for j in inventory),16384)
        for change in ('missing','duplicate','count','arm'):
            bad=copy.deepcopy(inventory)
            if change=='missing':bad.pop()
            elif change=='duplicate':bad[-1]=bad[0]
            elif change=='count':bad[-1]['attempts']-=1
            else:bad[-1]['comparison_arm']='multicage'
            with self.assertRaises(ValueError):m.validate_inventory(bad)

    def test_complete15_term_density_reconstruction_and_seams(self):
        row,_=record(0);m.validate_density(row)
        row['log_source']['cage0']=None;row['log_g']=None
        logs=[row['log_u'],-math.inf]+[row['log_source'][c] if row['log_source'][c] is not None else -math.inf for c in m.CHARTS]
        row['log_q_arm']=float(m.arm_log_q(np.array(logs)[:,None],m.COEFFICIENTS['bridge'])[0]);m.validate_density(row)
        bad=copy.deepcopy(row);bad['log_source'].pop('bridge_a3_b4')
        with self.assertRaisesRegex(ValueError,'inventory'):m.validate_density(bad)
        row['log_q_arm']+=.01
        with self.assertRaisesRegex(ValueError,'fifteen-term'):m.validate_density(row)

    def test_mixture_coefficients_normalization_defense_and_stratified_expectation(self):
        laws=np.arange(1,61,dtype=float).reshape(15,4);laws/=laws.sum(axis=1)[:,None]
        target=np.array([0.,3.,5.,1.])
        values={a:np.exp(m.arm_log_q(np.log(laws),m.COEFFICIENTS[a])) for a in m.ARMS}
        self.assertTrue(np.all(values['bridge']>=.5*values['baseline']))
        for arm,counts in m.ALLOCATIONS.items():
            q=values[arm];self.assertAlmostEqual(q.sum(),1.)
            value=sum(n/2048*np.sum((.5*laws[0]+.25*laws[1]+.25*laws[2+m.CHARTS.index(c)])*target/q) for c,n in counts.items())
            self.assertAlmostEqual(value,target.sum())

    def test_all_valid_panel_canonical_rows_and_token_counts(self):
        args=panel_fixture();self.assertEqual(m.validate_panel_coverage(*args),[0,2])
        for mutation in ('missing','digest','count','pose'):
            bad=copy.deepcopy(args)
            if mutation=='missing':bad[1]['entries'].pop()
            elif mutation=='digest':bad[2][0]['canonical_source_row_sha256']='0'*64
            elif mutation=='count':bad[2][0]['exact_source_intersection_count']=15
            else:bad[1]['entries'][0]['pose']['position'][0]+=.1
            with self.assertRaises(ValueError):m.validate_panel_coverage(*bad)

    def test_wall_failure_preserves_unknown_core_and_zero(self):
        args=panel_fixture();m.validate_panel_coverage(*args)
        args[2][1]['core_valid']=False
        with self.assertRaisesRegex(ValueError,'wall/core'):m.validate_panel_coverage(*args)

    def test_t_any_extra_neighbor_is_not_exact_A(self):
        args=panel_fixture();r,s=args[2][0],args[3][0]
        s['patches']['tokens']+=[[56,77,'x','y']];s['patches']['neighbor_labels']=[16,56,217];s['region']='other_contact'
        r.update(region='other_contact',neighbor_labels=[16,56,217],exact_A=False)
        r['canonical_source_row_sha256']=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(m.validate_panel_coverage(*args),[0,2])
        selected=m.selectors(r);self.assertFalse(selected['A_T']);self.assertTrue(selected['T_outside_A'])

    def test_population_zero_attempts_and_original_radial_mass_closure(self):
        rows=[record(i,False)[0] for i in range(2048)]
        rows[0]=record(0,weight=3)[0];rows[1]=record(1,region='unbound',weight=5)[0]
        result=m.population_statistics(rows,1.,2.)
        self.assertAlmostEqual(result['groups']['full_domain']['physical']['mass'],8/2048)
        self.assertEqual(result['regions']['hard_invalid']['attempted_count'],2046)
        self.assertEqual(set(result['radial_partitions']),{'full'})
        self.assertAlmostEqual(sum(b['mass'] for b in result['radial_partitions']['full']['full_domain']),8/2048)
        self.assertEqual(result['total_cpu_seconds'],3.)
        with self.assertRaisesRegex(ValueError,'denominator'):m.population_statistics(rows[:-1],1.,2.)

    def test_tail_physical_vs_hard_fractions_and_distinct_largest_rows(self):
        rows=[record(i,weight=(i+1),radius=r,population=i%4)[0] for i,r in enumerate([0.,6.,12.,24.,48.,96.])]
        for i,r in enumerate(rows):r['log_q_arm']=-math.log(6-i)
        result=m.tail_statistics(rows,6)
        self.assertEqual([b['physical']['hits'] for b in result['bins']],[1]*6)
        self.assertAlmostEqual(sum(b['physical_fraction_of_A_T'] for b in result['bins']),1.)
        self.assertAlmostEqual(result['bins'][5]['physical_fraction_of_A_T'],6/21)
        self.assertAlmostEqual(result['bins'][5]['hard_fraction_of_A_T'],1/21)
        self.assertEqual(result['largest']['physical']['row_id']['ordinal'],5)
        self.assertEqual(result['largest']['hard']['row_id']['ordinal'],0)
        self.assertEqual(result['largest']['physical']['original_full_squared_radius'],96.)

    def test_tail_empty_and_explicit_seam(self):
        row,_=record(0,False);empty=m.tail_statistics([row],1)
        self.assertIsNone(empty['largest']['physical']);self.assertEqual(empty['totals']['physical']['mass'],0.)
        row,_=record(1);row['mahalanobis_squared']['full']=None
        self.assertEqual(m.tail_statistics([row],1)['bins'][5]['physical']['hits'],1)

    def test_pooled_rb_arithmetic_and_overlap_score_remain_distinct(self):
        cfg,_,rows=score_fixture();r=rows[0];v=m.paired_weights(r,-4.,cfg)
        self.assertAlmostEqual(v['log_rb'],.035*2+8*math.log1p(.035/(2*2.24))+4)
        self.assertAlmostEqual(v['log_arithmetic'],r['log_mean_positive_weight']+4)
        self.assertNotAlmostEqual(v['log_rb'],v['log_arithmetic'],places=8)
        self.assertNotAlmostEqual(v['log_rb'],r['score']['z_overlap']+4,places=6)
        terms=m.moment_terms(2.,8,2.24,.035,-4.)
        self.assertAlmostEqual(v['log_rb'],terms['log_y'])
        self.assertGreater(math.exp(terms['log_auxiliary_variance']),0.)

    def test_fixed_bank_cloud_variance_uses_full_squared_denominator(self):
        rows=[record(i,False)[0] for i in range(4)];rows[0]=record(0)[0];rows[0]['log_auxiliary_variance']=math.log(9.)
        out=m.cloud_variance_statistics(rows,4)
        self.assertAlmostEqual(out['groups']['full_domain']['estimate'],9/16)
        self.assertAlmostEqual(out['groups']['A_T']['estimate'],9/16)
        self.assertEqual(out['groups']['unbound']['estimate'],0.)
        terms=m.moment_terms(2.,0,2.24,.035,-4.)
        self.assertIsNone(terms['log_auxiliary_variance'])

    def test_reused_ledger_checks_fresh_roles_zero_clouds_and_rejects_missing_progress(self):
        for zero in (False,True):
            args=ledger_fixture(zero);value=m.audit_ledger(*args)
            self.assertEqual(value['clouds_completed'],2)
            self.assertEqual(value['raw_points'],0 if zero else 40)
        args=list(ledger_fixture());args[0][-1]['kind']='pose_failed'
        with self.assertRaises(ValueError):m.audit_ledger(*args)

    def test_scope_contract_cannot_become_fresh_confirmation(self):
        p=dict(schema='context-bridge-physical-audit-v1',all_attempts=16384,population_denominator=2048,
            bins=[0,6,12,24,48,96,None],estimator=copy.deepcopy(m.CLOUD_RULE),physical_conditions=m.PHYSICAL,
            extension_scope=copy.deepcopy(m.EXTENSION_SCOPE))
        m.validate_extension_contract(p)
        for key in m.EXTENSION_SCOPE:
            bad=copy.deepcopy(p);bad['extension_scope'][key]=not bad['extension_scope'][key]
            with self.assertRaises(ValueError):m.validate_extension_contract(bad)
        p['population_denominator']=4096
        with self.assertRaises(ValueError):m.validate_extension_contract(p)

    def test_arm_comparison_is_descriptive_and_named_for_bridge(self):
        a=dict(mean_mass=2.,standard_error=.1);b=dict(mean_mass=3.,standard_error=.2)
        result=m.compare_means(a,b)
        self.assertEqual(result['bridge_minus_baseline'],1.)
        self.assertIn('Descriptive',result['comparison_scope'])
        self.assertAlmostEqual(result['log_mass_ratio'],math.log(1.5))

    def test_arithmetic_control_does_not_relabel_rb_noise_as_its_variance(self):
        rows=[record(i,False)[0] for i in range(2048)];rows[0]=record(0)[0]
        result=m.population_statistics(rows,1.,2.,include_cloud_variance=False)
        self.assertIsNone(result['fixed_bank_cloud_variance'])


if __name__=='__main__':unittest.main()
