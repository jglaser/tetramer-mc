import copy
import unittest
from prepare_contact_arc_score import validate_probe_allocation, combine_old_scores, OLD_ARMS


class FrozenArcAllocation(unittest.TestCase):
    def allocation(self):
        critical = [dict(id=f'critical/{i}', latent=[float(i),0,0,0,0,0]) for i in range(78)]
        breadth = [dict(id=f'breadth/{i}', latent=[float(i+78),0,0,0,0,0]) for i in range(128)]
        cm = [dict(id=p['id'], source=dict(arm='baseline' if i<4 else 'expanded',
              u=p['latent'], paired_log_weights=[1.,2.], source_attempted_draws=16384))
              for i,p in enumerate(critical)]
        classes=['native_R5','native_complement','competing','invalid']
        bm=[dict(**p,coverage_class=classes[i//32]) for i,p in enumerate(breadth)]
        return critical,breadth,cm,bm

    def test_allocation_retains_invalid_and_separate_sources(self):
        result=validate_probe_allocation(*self.allocation())
        self.assertEqual(result['invalid'],32)

    def test_identity_coordinate_and_metadata_changes_rejected(self):
        for mutate in (
            lambda a:a[1].pop(),
            lambda a:a[1][0].update(id=a[0][0]['id']),
            lambda a:a[1][0].update(latent=a[0][0]['latent']),
            lambda a:a[2][0]['source'].update(arm='expanded'),
            lambda a:a[3][0].update(coverage_class='invalid'),
        ):
            args=self.allocation();mutate(args)
            with self.assertRaises(ValueError):validate_probe_allocation(*args)

    def scores(self):
        probes=[dict(id='one',latent=[0.]*6)]
        row=dict(**probes[0],hard_valid=False,shell_valid=True,capture_valid=True,
            pose={},raw_coordinates=[0.]*6,baseline_log_density=2.,log_proposal_density=2.,
            log_physical_jacobian=-1.,density_cpu_seconds=.1,width_contacts=[False]*3)
        arms={a:[copy.deepcopy(row)] for a in OLD_ARMS}
        arms['uniform_phi92'][0]['log_proposal_density']=1.5
        return probes,arms,[dict(coverage_class='invalid')]

    def test_saved_invalid_scores_and_all_densities_retained(self):
        result=combine_old_scores(*self.scores())
        self.assertFalse(result[0]['saved_geometry']['hard_valid'])
        self.assertEqual(result[0]['old_log_q']['uniform_phi92'],1.5)

    def test_missing_rows_and_inconsistent_baseline_rejected(self):
        args=self.scores();args[1]['localized_phi92']=[]
        with self.assertRaises(ValueError):combine_old_scores(*args)
        args=self.scores();args[1]['localized_phi92'][0]['baseline_log_density']=3.
        with self.assertRaises(ValueError):combine_old_scores(*args)


if __name__=='__main__':unittest.main()
