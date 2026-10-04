"""Synthetic controls for independent direct/map density comparison."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import audit_proposal_density_probe as audit


class DensityAuditTests(unittest.TestCase):
    def test_status_and_threshold_decisions(self):
        good=dict(status='finite',log_density=1.,error=None)
        self.assertTrue(audit.compare_row(good,1.+1e-7,2e-6)['passed'])
        self.assertFalse(audit.compare_row(good,1.+1e-4,2e-6)['passed'])
        zero=dict(status='negative_infinity',log_density=None,error=None)
        self.assertTrue(audit.compare_row(zero,-float('inf'),2e-6)['passed'])
        self.assertFalse(audit.compare_row(zero,-100.,2e-6)['passed'])
        self.assertFalse(audit.compare_row(good,-float('inf'),2e-6)['passed'])
        self.assertFalse(audit.compare_row(dict(status='error',log_density=None,error='bad'),1.,2e-6)['passed'])

    def test_completed_synthetic_rust_fixture_and_deliberate_mismatch(self):
        folder=Path(os.environ['DENSITY_PROBE_SYNTHETIC_FIXTURE'])
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);report=json.loads((folder/'synthetic-report.json').read_bytes())
            def run(right,out):
                argv=['audit','--baseline-model',str(folder/'synthetic-model.json'),
                    '--exported-model',str(folder/'synthetic-model.json'),
                    '--baseline-report',str(folder/'synthetic-report.json'),'--exported-report',str(right),
                    '--probes',str(folder/'synthetic-probes.json'),'--expected-probes','2',
                    '--source-bundle',str(folder/'source-bundle.json'),'--out',str(out)]
                with patch.object(sys,'argv',argv):return audit.main()
            good=tmp/'good.json'
            self.assertEqual(run(folder/'synthetic-report.json',good),0)
            result=json.loads(good.read_bytes())
            self.assertTrue(result['passed']);self.assertTrue(result['complete'])
            self.assertEqual(result['arms']['baseline']['mismatches'],dict(direct=0,transport=0))
            self.assertEqual(len(result['arms']['exported']['probes']),2)
            altered=copy.deepcopy(report);altered['probes'][0]['direct']['log_density']+=.01
            wrong=tmp/'wrong-rust.json';wrong.write_text(json.dumps(altered))
            bad=tmp/'bad.json';self.assertEqual(run(wrong,bad),1)
            result=json.loads(bad.read_bytes())
            self.assertFalse(result['passed']);self.assertTrue(result['complete'])
            self.assertEqual(result['arms']['exported']['mismatches'],dict(direct=1,transport=0))
            self.assertEqual(len(result['arms']['exported']['probes']),2)
            altered=copy.deepcopy(report);altered['probes'][0]['probe']['label']='substituted'
            wrong.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'Changed probe identities'):
                run(wrong,tmp/'should-not-exist.json')


if __name__=='__main__':unittest.main()
