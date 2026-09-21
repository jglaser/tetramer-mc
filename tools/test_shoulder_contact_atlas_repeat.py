"""Fixed-law repeat guards and independent contrasts, with no physical draws."""
import copy
import math
import unittest

from analyze_shoulder_contact_atlas import KEYS
from analyze_shoulder_contact_atlas_repeat import compare_campaigns, independent_repeat_comparison
from prepare_cayley_rms_cover import read
from prepare_shoulder_contact_atlas_repeat import (ORIGINAL, POPULATIONS, PREFIXES, SAMPLES,
    SEEDS, validate_repeat_target)


def row(mean, variance, draws):
    return dict(logQ=math.log(mean) if mean else None,
        log_variance_of_mean=math.log(variance) if variance else None,
        row_RSE=math.sqrt(variance)/mean if mean else None,
        independent_population_RSE=math.sqrt(variance)/mean if mean else None,
        draws=draws, weight_ESS=10. if mean else 0.)


class ShoulderRepeatTests(unittest.TestCase):
    def plans(self):
        original=read(ORIGINAL/'protocol.json'); repeat=copy.deepcopy(original)
        repeat['prefix_counts']=PREFIXES; repeat['total_unconditional_draws']=2*POPULATIONS*SAMPLES
        for arm in repeat['arms']:
            arm.update(populations=POPULATIONS,samples_per_population=SAMPLES,
                       seeds=[SEEDS[arm['name']]+1009*i for i in range(POPULATIONS)])
        models={a['name']:a['model_sha256'] for a in original['arms']}
        return original,repeat,models

    def test_dynamic_double_budget_preserves_all_original_targets(self):
        a,b,models=self.plans();validate_repeat_target(a,b,models,models)
        for key,value in [('hybrid',dict(model_weight=.75,uniform_probability=.05,cover_scales=[1.])),
                          ('q_window',dict(a['q_window'],upper_inclusive=True)),
                          ('config_sha256','changed'),('prefix_counts',[8192,32768,65536])]:
            changed=copy.deepcopy(b);changed[key]=value
            with self.assertRaises(ValueError):validate_repeat_target(a,changed,models,models)
        changed=copy.deepcopy(b);changed['analysis']['radii']=[.25,.6]
        with self.assertRaises(ValueError):validate_repeat_target(a,changed,models,models)

    def test_fixed_streams_and_byte_hashes_are_enforced(self):
        a,b,models=self.plans(); changed=copy.deepcopy(b)
        changed['arms'][0]['seeds'][0]=a['arms'][0]['seeds'][0]
        with self.assertRaises(ValueError):validate_repeat_target(a,changed,models,models)
        changed=copy.deepcopy(b);changed['arms'][0]['new_covariance_scale']=2.
        with self.assertRaises(ValueError):validate_repeat_target(a,changed,models,models)
        with self.assertRaises(ValueError):validate_repeat_target(a,b,models,dict(models,narrow='other-bytes'))

    def test_independent_difference_has_explicit_repeat_direction(self):
        result=independent_repeat_comparison(row(10.,4.,100),row(12.,1.,400))
        self.assertEqual((result['left_label'],result['right_label']),('repeat','original'))
        self.assertAlmostEqual(result['left_to_right_ratio'],1.2)
        self.assertAlmostEqual(result['linear_left_minus_right_in_combined_row_SE'],2/math.sqrt(5))
        self.assertAlmostEqual(math.exp(result['estimated_log_variance_of_difference']),5.)
        self.assertAlmostEqual(result['N_scaled_observed_variance_ratio'],1.)
        self.assertEqual(result['unconditional_draw_count_ratio'],4.)
        zero=independent_repeat_comparison(row(0.,0.,100),row(12.,1.,400))
        self.assertIsNone(zero['left_to_right_ratio'])
        self.assertIsNone(zero['linear_left_minus_right_in_combined_row_SE'])

    def test_every_mask_and_prefix_is_compared_without_pooling(self):
        a=dict(arm='narrow',CPU_seconds=10.,populations=[dict(seed=11)],prefix_comparisons=[dict(source='old')],
               **{kind:{k:row(10.,4.,100) for k in KEYS} for kind in ('physical','hard')})
        b=dict(arm='narrow',CPU_seconds=20.,populations=[dict(seed=21)],prefix_comparisons=[dict(source='repeat')],
               **{kind:{k:row(12.,1.,400) for k in KEYS} for kind in ('physical','hard')})
        b['prefixes']=[dict(samples_per_population=n,**{kind:{k:row(11.,2.,n) for k in KEYS} for kind in ('physical','hard')}) for n in [100,200,400]]
        result=compare_campaigns(a,b)
        self.assertEqual(set(result['physical']),set(KEYS))
        self.assertEqual(len(result['repeat_prefix_against_original_final']),3)
        self.assertEqual(result['repeat_prefix_comparisons'],[dict(source='repeat')])
        self.assertIn('correlated diagnostics',result['prefix_scope'])
        self.assertAlmostEqual(result['physical']['full']['observed_variance_times_CPU_ratio'],.5)
        self.assertAlmostEqual(result['physical']['full']['weight_ESS_per_CPU_second_ratio'],.5)
        b['populations'][0]['seed']=11
        with self.assertRaises(ValueError):compare_campaigns(a,b)


if __name__=='__main__': unittest.main()
