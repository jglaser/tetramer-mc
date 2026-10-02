import copy
from pathlib import Path
import tempfile
import unittest
import report_hard_free_line_sensitivity as report


class DisplayTests(unittest.TestCase):
    def test_running_or_failed_campaign_is_not_a_completed_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for phase,complete in [('classification',False),('independent_audit_failed',False),('classification',True)]:
                report.write(root/'status.json',dict(phase=phase,complete=complete))
                with self.assertRaisesRegex(ValueError,'Wait for all'): report.load(root)

    def test_preserves_all_failures_and_does_not_call_unobserved_mass_zero_efficiency(self):
        arms = {name:dict(observed_importance_ESS_per_cpu_second={r:float(i+1) for r in report.REGIONS},
                         estimates={r:dict(row_uncertainty=dict(log_Qz=1.)) for r in report.REGIONS})
                for i,name in enumerate(report.LABEL)}
        arms['alpha02']['estimates'][report.REGIONS[0]]['row_uncertainty']['log_Qz'] = None
        failures = [dict(control='alpha02',family='orthant',region=report.REGIONS[0],bin=i) for i in range(64)]
        comparison = dict(checks=dict(unchanged=False),sensitivity_checks_passed=False,stage_free_energy_intervals={},
            stage_quality={},regional_comparisons={},free_energy_contrast_comparisons={},
            failed_material_strata=failures,all_stratum_comparisons=failures+[dict(material=False)])
        saved = copy.deepcopy(comparison); summary = report.summarize(arms,comparison)
        self.assertEqual(comparison,saved)
        self.assertEqual(summary['failed_material_strata'],failures)
        self.assertEqual(len(summary['all_stratum_comparisons']),65)
        self.assertIsNone(summary['importance_efficiency']['alpha02'][report.REGIONS[0]]['ratio'])
        self.assertEqual(summary['importance_efficiency']['lambda64'][report.REGIONS[0]]['ratio'],3.)
        self.assertFalse(summary['sensitivity_checks_passed'])


if __name__ == '__main__': unittest.main()
