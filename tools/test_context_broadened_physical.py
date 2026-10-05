"""Small synthetic controls; no protein queries, pose draws, or point clouds."""
import copy
import math
import unittest

import analyze_context_broadened_physical as m
from analyze_context_overlap_panel import audit_rows


def record(ordinal, valid=True, region='A_patch_complete', log_q=0., log_w=0.):
    pose={'position':[0.,0.,0.], 'orientation':[1.,0.,0.,0.]}
    return dict(ordinal=ordinal, physical_valid=valid, region=region if valid else 'hard_invalid',
                source_T_complete_all_regions=valid and region=='A_patch_complete',
                proposed_pose=pose, log_q_arm=log_q, log_physical_weight=log_w if valid else None,
                source_rows={'path':'/synthetic/rows.jsonl','sha256':'synthetic'},
                mahalanobis_squared={'full':6.,'diagonal':12.,'broad_full':1.5})


def saved(row):
    return dict(input={'ordinal':row['ordinal'],'proposed_pose':row['proposed_pose']},complete=True,
                actual={'physical_valid':row['physical_valid']},region=row['region'] if row['physical_valid'] else None,
                clouds=[],physical_weight_status='not_estimated')


def panel_entry(row, original):
    return dict(id=str(row['ordinal']),ordinal=row['ordinal'],source_rows=row['source_rows'],
                pose=row['proposed_pose'],metadata={'log_q_balanced':row['log_q_arm'],'original_row':original})


def score_fixture(uncertain=10., counts=(3,5)):
    z=.035; intensity=2.24; lower=2.; total=sum(counts); q=-4.
    envelope=dict(lower_volume=lower,upper_volume=lower+uncertain,uncertain_volume=uncertain)
    weights=[z*lower+k*math.log1p(z/intensity) for k in counts]
    log_mean=max(weights)+math.log(sum(math.exp(x-max(weights)) for x in weights)/2)
    entry=dict(id='pose0',pose={'position':[0.,0.,0.],'orientation':[1.,0.,0.,0.]},metadata={'log_q_balanced':q})
    clouds=[]
    for k,w in zip(counts,weights):
        n=20 if uncertain else 0
        clouds.append(dict(progress=dict(begun=True,complete=True,planned_points=n if uncertain else None,
                                        processed_points=n,overlap_points=k,log_weight=w),
                           weight=dict(**envelope,raw_points=n,overlap_points=k,log_weight=w)))
    row=dict(id='pose0',pose_index=0,complete=True,pose=entry['pose'],envelope=envelope,clouds=clouds,
             log_q_balanced=q,log_mean_positive_weight=log_mean,
             score=dict(overlap_volume=lower+total/(2*intensity),z_overlap=z*(lower+total/(2*intensity)),
                        variance_estimate=z*z*total/(4*intensity*intensity),variance_upper=z*z*uncertain/(2*intensity)))
    return {'lambda':intensity,'activity':z},{'entries':[entry]},[row]


class Tests(unittest.TestCase):
    def test_two_positive_weights_not_exponentiated_score(self):
        cfg,panel,rows=score_fixture()
        result,counts=audit_rows(cfg,panel,rows)
        expected=math.log(sum(math.exp(c['weight']['log_weight']) for c in rows[0]['clouds'])/2)+4
        self.assertAlmostEqual(result[0]['log_positive_importance_weight'],expected)
        self.assertNotAlmostEqual(result[0]['log_positive_importance_weight'],result[0]['log_physical_importance_score'],places=6)
        self.assertEqual(counts['clouds'],2)
        rows[0]['clouds'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete pair'):audit_rows(cfg,panel,rows)

    def test_zero_envelope_still_two_complete_weights(self):
        cfg,panel,rows=score_fixture(0.,(0,0));result,counts=audit_rows(cfg,panel,rows)
        self.assertEqual(counts['raw_points'],0);self.assertEqual(counts['clouds'],2)
        self.assertAlmostEqual(result[0]['log_positive_importance_weight'],.07+4)
        rows[0]['clouds'][0]['progress']['complete']=False
        with self.assertRaises(ValueError):audit_rows(cfg,panel,rows)

    def test_all_valid_panel_and_invalid_zeros_preserved(self):
        geometry=[record(0),record(1,False),record(2,region='unbound')];originals=list(map(saved,geometry))
        panel={'entries':[panel_entry(geometry[i],originals[i]) for i in (0,2)]};job={'attempts':3,'empty':False}
        self.assertEqual(m.validate_panel_coverage(job,panel,geometry,originals),[0,2])
        panel['entries'].pop()
        with self.assertRaisesRegex(ValueError,'ALL hard-valid'):m.validate_panel_coverage(job,panel,geometry,originals)

    def test_full_arm_density_alias_cannot_change(self):
        geometry=[record(0,log_q=-7.)];originals=list(map(saved,geometry));entry=panel_entry(geometry[0],originals[0])
        entry['metadata']['log_q_balanced']=-8.
        with self.assertRaisesRegex(ValueError,'full_arm_density'):
            m.validate_panel_coverage({'attempts':1,'empty':False},{'entries':[entry]},geometry,originals)

    def test_mass_denominator_region_closure_and_orthogonal_T(self):
        rows=[record(i,False) for i in range(4096)]
        rows[0]=record(0,log_w=math.log(3));rows[1]=record(1,region='other_contact',log_w=math.log(5))
        rows[1]['source_T_complete_all_regions']=True
        result=m.population_statistics(rows,1.,2.)
        self.assertAlmostEqual(result['groups']['full_domain']['physical']['mass'],8/4096)
        self.assertAlmostEqual(result['groups']['A_T']['physical']['mass'],3/4096)
        self.assertAlmostEqual(result['groups']['T_any']['physical']['mass'],8/4096)
        self.assertAlmostEqual(result['groups']['T_outside_A']['physical']['mass'],5/4096)
        self.assertAlmostEqual(result['groups']['full_domain']['physical']['importance_ess'],64/34)
        self.assertEqual(result['regions']['hard_invalid']['attempted_count'],4094)
        self.assertEqual(result['total_cpu_seconds'],3.)
        with self.assertRaisesRegex(ValueError,'denominator'):m.population_statistics(rows[:-1],1.,2.)

    def test_fixed_stratum_inventory_rejects_missing_or_unequal_allocation(self):
        jobs=[]
        for arm,counts in [('baseline',{'full':2048,'diagonal':2048}),('broadened',{'full':1536,'diagonal':1536,'broad_full':1024})]:
            for pop in range(4):
                for comp,n in counts.items():
                    identity=f'{arm}-{pop}-{comp}'
                    jobs.append(dict(id=identity,stratum_id=identity,comparison_arm=arm,population_index=pop,component=comp,attempts=n))
        m.validate_inventory(jobs)
        wrong=copy.deepcopy(jobs);wrong[-1]['attempts']=1023
        with self.assertRaises(ValueError):m.validate_inventory(wrong)
        with self.assertRaises(ValueError):m.validate_inventory(jobs[:-1])

    def test_explicit_empty_receipt_does_not_erase_failed_job(self):
        job=dict(id='j',stratum_id='s',attempts=2,config={'sha256':'cfg'},panel={'sha256':'panel'})
        receipt=dict(schema='context-broadened-empty-physical-v1',complete=True,passed=True,id='j',stratum_id='s',attempts=2,
            all_hard_invalid=True,all_attempts_preserved=True,config_sha256='cfg',panel_sha256='panel',
            valid_poses=0,clouds_begun=0,clouds_completed=0,raw_points=0,processed_points=0,new_poses_generated=0,
            retries=0,replacements=0,cpu_seconds=.001)
        m.validate_empty(job,{}, {'entries':[]},receipt)
        receipt['replacements']=1
        with self.assertRaises(ValueError):m.validate_empty(job,{}, {'entries':[]},receipt)

    def test_radial_seams_and_zero_region_comparison(self):
        self.assertEqual([m.radial_index(x) for x in (0,6,12,24,48,96,None)],[0,1,2,3,4,5,5])
        zero=dict(mean_mass=0.,standard_error=0.)
        result=m.compare_means(zero,zero)
        self.assertIsNone(result['log_mass_ratio']);self.assertFalse(result['within_point_two_kbt'])
        self.assertFalse(m.concentration_gate({'importance_ess':0.,'largest_fraction':None}))


if __name__=='__main__':unittest.main()
