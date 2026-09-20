#!/usr/bin/env python3
"""Small synthetic controls for analyzer boundary cases and pooling guards."""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import analyze_native_region_reference as analysis


class AnalyzerControls(unittest.TestCase):
    def test_one_nonzero_draw_has_no_variance_estimate(self):
        moment = analysis.fresh()
        analysis.add(moment, math.log(2))
        result = analysis.present(moment, 1)
        self.assertIsNone(result["relative_SE"])
        self.assertAlmostEqual(result["log_normalizer"], math.log(2))

    def test_zero_and_mixed_populations(self):
        zero = {"log_normalizer": None}
        self.assertIsNone(analysis.replicate_relative_se([zero, zero], None))
        self.assertEqual(analysis.replicate_relative_se([zero, {"log_normalizer": math.log(2)}], 0), 1)
        self.assertIsNone(analysis.present(analysis.fresh(), 1)["relative_SE"])

    def test_pooling_requires_same_target_and_cover(self):
        first = {"physical_signature": {"shape_sha256": "a", "activity": .035},
                 "cover": {"volume": 2.0}, "cover_volume": 2.0}
        analysis.require_matching_targets([first, copy.deepcopy(first)])
        for key in ["physical_signature", "cover", "cover_volume"]:
            changed = copy.deepcopy(first)
            changed[key] = "different"
            with self.assertRaises(AssertionError):
                analysis.require_matching_targets([first, changed])

    def test_all_zero_pipeline(self):
        with tempfile.TemporaryDirectory(prefix="native-analyzer-zero-") as directory:
            root = Path(directory)
            jobs = []
            for i in range(2):
                output = root/f"r{i:02d}"
                output.mkdir()
                line = json.dumps({"draw": 0, "q": 2.0, "zero": "q"})+"\n"
                (output/"samples.jsonl").write_text(line)
                summary = {"samples": 1, "samples_sha256": hashlib.sha256(line.encode()).hexdigest(),
                           "q_rejected": 1, "capture_rejected": 0, "hard_rejected": 0,
                           "cpu_seconds": 0, "wall_seconds": 0, "raw_cloud_points": 0,
                           **{name: {"nonzero": 0} for name in ["native", "native_core", "native_shell"]}}
                manifest = {"samples": 1, "activity": 0, "lambda": 1, "depletant_radius": 1.5,
                            "cloud_replicates": 2, "config_sha256": "same", "shape_sha256": "same", "metric": {}}
                (output/"summary.json").write_text(json.dumps(summary))
                (output/"manifest.json").write_text(json.dumps(manifest))
                (output/"cover.json").write_text(json.dumps({"volume": 1.0}))
                jobs.append({"output": str(output)})
            (root/"manifest.json").write_text(json.dumps({"jobs": jobs, "total_unconditional_draws": 2}))
            subprocess.run([sys.executable, str(Path(analysis.__file__)), str(root), "--workers", "1"],
                           check=True, capture_output=True, text=True)
            result = json.loads((root/"assessment-streaming.json").read_text())
            self.assertIsNone(result["regions"]["native"]["log_normalizer"])
            self.assertIsNone(result["independent_population_relative_SE"])
            self.assertIsNone(result["regions"]["native"]["relative_SE"])
            self.assertEqual(result["samples"], 2)


if __name__ == "__main__":
    unittest.main()
