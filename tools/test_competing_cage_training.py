"""Synthetic-only controls for deterministic competing-cage metadata."""
import copy
import math
import unittest
import prepare_competing_cage_training as p


LIMITS=dict(cpu_seconds=600,wall_seconds=1200,memory_bytes=4*1024**3,
    raw_per_leg=20_000_000,raw_per_outer=40_000_000,raw_campaign=2_000_000_000,
    retained_per_leg=20_000_000,retained_per_outer=40_000_000,retained_campaign=2_000_000_000)


def row(name,ordinal,score,patches,region='B',valid=True):
    return dict(stratum_id=name,ordinal=ordinal,log_rb_importance=score,
        patch_tokens=[[1,2,str(x),'p'] for x in patches],region=region,physical_valid=valid,
        proposed_pose=dict(position=[1.,2.,3.],orientation=[1.,0.,0.,0.]))


class TrainingControls(unittest.TestCase):
    def test_stable_rank_tie_and_distinct_fingerprint(self):
        a=row('a',1,9,[1,2]);b=row('b',1,9,[1,2]);c=row('z',8,8,[3,4])
        self.assertEqual(p.choose_seeds([c,b,a])[0],[a,c])
        self.assertEqual(p.choose_seeds([a,c,b])[0],[a,c])

    def test_exact_jaccard_boundary(self):
        a=row('a',0,2,[1,2]);b=row('b',0,1,[1])
        self.assertEqual(p.jaccard(a,b),.5)
        self.assertEqual(p.choose_seeds([a,b])[0],[a,b])

    def test_eligibility_excludes_source_complete_unbound_invalid(self):
        eligible=[row('a',0,2,[1]),row('b',0,1,[2])]
        extras=[row('x',0,100,[9],'A_patch_complete'),row('y',0,101,[8],'unbound'),row('z',0,102,[7],valid=False)]
        self.assertEqual(p.choose_seeds(extras+eligible)[0],eligible)

    def test_missing_second_fails_without_substitution(self):
        with self.assertRaises(ValueError):p.choose_seeds([row('a',0,2,[1]),row('b',0,1,[1])])
        with self.assertRaises(ValueError):p.choose_seeds([row('a',0,float('nan'),[1])])

    def test_duplicate_token_rejected(self):
        with self.assertRaises(ValueError):p.tokens(row('a',0,2,[1,1]))

    def test_rotation_sign_and_seam(self):
        q=[math.cos(math.pi/8),math.sin(math.pi/8),0.,0.]
        self.assertAlmostEqual(p.angle_degrees([1.,0.,0.,0.],q),45.)
        self.assertAlmostEqual(p.angle_degrees(q,[-x for x in q]),0.)
        self.assertAlmostEqual(p.angle_degrees([1.,0.,0.,0.],[0.,1.,0.,0.]),180.)

    def test_seed_roles_unique_and_deterministic(self):
        seeds=[p.seed_number('ab'*32,c,s) for c in range(2) for s in range(4)]
        self.assertEqual(len(set(seeds)),8)
        self.assertEqual(seeds,[p.seed_number('ab'*32,c,s) for c in range(2) for s in range(4)])

    def test_only_initial_pose_and_execution_metadata_change(self):
        base=dict(translation_steps=[.2],rotation_steps_deg=[1.],rotation_probability=.5,
            local_attempts_per_cycle=4,expected_fixed_body_count=263,depletant_radius=1.5,
            reservoir_density=.035,poisson_lambda_ratio=64.,source_state='immutable-source',
            fixed_context='immutable-context',shape='immutable-shape',expected_sha256={'source_state':'abc'},
            endpoint_gate={'max_cells':255},initial_pose={'old':True},identity={'old':True})
        before=copy.deepcopy(base);selected=row('a',2,3,[1])
        cfg=p.make_config(base,selected,0,2,'ab'*32,LIMITS)
        self.assertEqual(base,before)
        mutable={'initial_pose','seed','maximum_attempts','warmup_cycles','wall_seconds','limits','identity'}
        self.assertEqual({k:v for k,v in cfg.items() if k not in mutable},
                         {k:v for k,v in base.items() if k not in mutable})
        self.assertEqual(cfg['initial_pose'],selected['proposed_pose'])
        self.assertEqual(cfg['identity']['split'],'heldout')
        self.assertEqual(cfg['maximum_attempts'],2304*(cfg['local_attempts_per_cycle']+1))
        self.assertEqual(cfg['warmup_cycles']*5,1280)
        self.assertEqual(cfg['maximum_attempts']-cfg['warmup_cycles']*5,10240)

    def test_limits_cannot_silently_expand(self):
        p.validate_limits(LIMITS)
        with self.assertRaises(ValueError):p.validate_limits(dict(LIMITS,raw_campaign=20_000_000_000))
        with self.assertRaises(ValueError):p.validate_limits(dict(LIMITS,cpu_seconds=1200))


if __name__=='__main__':unittest.main()
