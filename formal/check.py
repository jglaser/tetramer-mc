#!/usr/bin/env python3
"""Build proofs, reject unexpected axiom dependencies, and record source hashes."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


root = Path(__file__).resolve().parent
env = os.environ.copy()
local_lake = root / ".elan/bin/lake"
if local_lake.is_file():
    lake = str(local_lake)
    env["ELAN_HOME"] = str(root / ".elan")
else:
    lake = shutil.which("lake")
    if lake is None:
        raise SystemExit("Install the pinned Lean toolchain first; see README.md")

checks = []
for args in [[lake, "build"], [lake, "env", "lean", "Audit.lean"]]:
    result = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, check=True)
    checks.append({"command": args, "exit_code": result.returncode,
                   "stdout": result.stdout, "stderr": result.stderr})

audit = checks[1]["stdout"]
axiom_lists = re.findall(r"depends on axioms: \[([^]]*)\]", audit)
expected = (root / "Audit.lean").read_text().count("#print axioms")
assert len(axiom_lists) == expected, audit
allowed = {"propext", "Classical.choice", "Quot.sound"}
for axioms in axiom_lists:
    assert {name.strip() for name in axioms.split(",") if name.strip()} <= allowed, axioms

files = ["lean-toolchain", "lakefile.toml", "lake-manifest.json",
         "ReversibleSampling.lean", "ReversibleSampling/Balance.lean",
         "ReversibleSampling/MetropolisHastings.lean", "ReversibleSampling/Involution.lean",
         "ReversibleSampling/ImportanceSampling.lean",
         "ReversibleSampling/Poisson.lean", "ReversibleSampling/CountGate.lean",
         "ReversibleSampling/ConditionalPoisson.lean",
         "Audit.lean", "check.py"]
record = {
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "lean": "4.24.0",
    "lean_commit": "797c613eb9b6d4ec95db23e3e00af9ac6657f24b",
    "mathlib_commit": "f897ebcf72cd16f89ab4577d0c826cd14afaafc7",
    "checks": checks,
    "theorems_axiom_audited": expected,
    "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files},
    "scope": "General measurable-state accept/reject balance; finite-density MH and "
             "measure-preserving deterministic-involution specializations; conditional "
             "auxiliary marginal and hybrid invariance; nonnegative importance-weight "
             "expectation, unbiased auxiliary-weight marginal, and randomized importance "
             "expectation identities; Poisson generating function, depletion mean/second "
             "moment/relative variance and concrete importance/marginal corollaries; "
             "normalized auxiliary-count MH and the simplified gained/lost Poisson gate, "
             "including zero-activity and zero-volume limits. Not a proof "
             "of concrete geometric proposal densities, Rust code, "
             "numerical arithmetic, ergodicity, or mixing.",
}
(root / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
print(f"Build and {expected}-theorem axiom audit passed; wrote {root / 'validation.json'}")
