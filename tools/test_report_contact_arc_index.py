import json
import math
from pathlib import Path
import unittest
import numpy as np
from report_contact_arc_index import SavedComponentDecoder,index_factors,moment


class IndexPostprocessingTests(unittest.TestCase):
    def test_finite_state_factors_match_direct_algebra(self):
        rho=np.array([.3,.7]);old=np.array([.15,.3]);new=np.array([.25,.75])
        result=index_factors(np.log(rho),np.log(old),np.log(new));Qo=rho@old;Qn=rho@new
        psi=rho*old/Qo;S=np.sum(psi*psi/(rho*new))
        self.assertAlmostEqual(result['log_S'],math.log(S),places=14)
        self.assertAlmostEqual(result['index_over_marginal_second_moment'],Qn*S,places=14)
        self.assertLessEqual(result['index_over_old_second_moment'],1.)

    def test_identity_and_normalized_moment_denominator(self):
        rho=np.array([.2,.8]);logs=np.array([-4.,-2.]);r=index_factors(np.log(rho),logs,logs)
        self.assertAlmostEqual(r['index_over_old_second_moment'],1.,places=14)
        self.assertAlmostEqual(r['chi_squared_old_vs_new_responsibilities'],0.,places=14)
        m=moment([math.log(3/100),math.log(1/100)])
        self.assertAlmostEqual(math.exp(m['log_selected_second_moment']),.04,places=14)
        self.assertAlmostEqual(m['largest_contribution'],.75,places=14)
        self.assertEqual(moment([])['rows'],0)

    def test_saved_toy_components_reconstruct_complete_mixtures(self):
        root=Path(__file__).resolve().parents[1];base=root/'results/contact-arc-reference-validation-20261001'
        allocation=root/'results/contact-arc-reference-inputs-20261001/allocation.json'
        if not allocation.exists() or not (base/'two_spheres/rust/scores.jsonl').exists():
            self.skipTest('Archived fixed toy-score validation artifacts are absent')
        jobs=json.loads(allocation.read_text())['jobs'];count=0
        for job in jobs:
            region=json.loads(Path(job['region']).read_text())
            decoders=[SavedComponentDecoder(json.loads(Path(p).read_text()),region) for p in job['guides']]
            rows=[json.loads(s) for s in (base/job['name']/'rust/scores.jsonl').read_text().splitlines()]
            for row in rows:
                for i,decoder in enumerate(decoders):
                    decoded=decoder.decode(row,i);count+=1
                    self.assertLess(abs(decoded['old_mixture_reconstruction_error']),1e-12)
                    self.assertLess(abs(decoded['new_mixture_reconstruction_error']),1e-12)
                    self.assertLessEqual(decoded['accounting']['rho']['expected_cold_query_circles'],1.)
        self.assertEqual(count,120)


if __name__=='__main__':unittest.main()
