import copy
import unittest
from prepare_hard_free_line_score import candidate,validate_inputs


class HardFreeLinePreparation(unittest.TestCase):
    def test_unchanged_gaussians_no_contact_restriction(self):
        old=dict(schema='defensive-latent-shell-guide-v1',region_sha256='fixed',
                 defensive_uniform_shell_probability=.5,gaussian_components=[dict(weight=1/92,mean=[0]*6,covariance=[])]*92)
        original=copy.deepcopy(old);guide=candidate(old,[0,1,2])
        self.assertEqual(old,original);self.assertEqual(guide['gaussian_components'],old['gaussian_components'])
        self.assertEqual(guide['conditional_probability'],1)
        self.assertNotIn('contact_widths_A',guide)
        self.assertNotIn('contact_neighbor_indices',guide)
        old['component_contact_pairs']=[]
        with self.assertRaisesRegex(ValueError,'contact labels'):candidate(old,[0])

    def allocation(self):
        probes=[dict(id=str(i),latent=[i,0,0,0,0,0]) for i in range(206)]
        saved=[];classes=['native_R5','native_complement','competing','invalid']
        for i,p in enumerate(probes):
            saved.append(dict(**p,group='critical' if i<78 else 'breadth',
                metadata={'source':{'arm':'baseline' if i<4 else 'expanded'}} if i<78 else {'coverage_class':classes[(i-78)//32]},
                saved_geometry=dict(hard_valid=i<174,shell_valid=True,capture_valid=True)))
        return probes,saved

    def test_exact_predeclared_groups_and_invalid_rows(self):
        validate_inputs(*self.allocation())
        for mutation in (lambda p,s:s.pop(),lambda p,s:s[0]['saved_geometry'].update(hard_valid=False),
                         lambda p,s:s[0]['metadata']['source'].update(arm='expanded'),
                         lambda p,s:p[0].update(id='1')):
            p,s=self.allocation();mutation(p,s)
            with self.assertRaises(ValueError):validate_inputs(p,s)


if __name__=='__main__':unittest.main()
