# Completed SMC versus importance comparisons

`tools/run_smc_importance_bridge.py` freezes a dependent, read-only analysis
workflow. It waits for the original SMC scheduler to finish its eight physical
populations, both terminal audits, and the broad-versus-narrow comparison. It
then runs `compare_r4_smc_importance.py` sequentially for the completed broad and
narrow analyses against the exact completed confirmation analysis.

This wrapper launches zero physical workers and one comparison process at a
time. It does not classify poses, replay audits, inspect unfinished weights,
pool populations, or dispatch full-vessel work. `confirmation_passed=false` and
scientific disagreements do not prevent the independent diagnosis from running.
Successful execution is recorded separately from the comparator's agreement
flags; neither implies convergence or assembly stability.

The frozen plan records the complete local Python source closure, interpreter
hash/version, NumPy/SciPy versions, single-thread numerical environment, exact
comparison commands, dependency plan checksum, original PID/start ticks, and
immutable importance inputs. The dependent status remains mutable while waiting;
its terminal bytes and both completed analysis hashes are bound before dispatch.
The scheduler's physical, audit, and comparison command identities must match
the original plan. A dead or recycled original process with unfinished status
is a failure, as is any unsuccessful dependency command or changed frozen byte.

The September22 workflow is already frozen and running as controller PID278609. Its [status](../runs/smc-importance-bridge-workflow-20260922/status.json) is authoritative; it is waiting for the original SMC dependency. Plan SHA256 is `be0a3d190cbc317b857ae7a5551d6410b2f190f47f8a7826837f2c35f406febe`. Do not launch a second copy. The commands below record its preparation and launch procedure.

Freeze the workflow with the pinned evidence:

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B tools/run_smc_importance_bridge.py freeze \
  --out runs/smc-importance-bridge-workflow-20260922 \
  --dependency runs/smc-r4-controls-workflow-20260922 \
  --expected-dependency-plan-sha256 80ad403b5315a6ae25cd81633e74be81ca0c9c74e713fac383ee5d403922a86a \
  --dependency-pid 103563 --dependency-birth 99949842 \
  --importance runs/contact-confirmation-comparison-20260922/analysis.json \
  --expected-importance-sha256 963b2b32d8b4dc5fe4acbaf3aed4b4a9afdf8ec224d61306dd1ab55e059c0b05
```

Freezing creates an inert new directory and prints its plan checksum. Launch the
archived runner separately, using that exact checksum:

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B \
  runs/smc-importance-bridge-workflow-20260922/common/run_smc_importance_bridge.py run \
  --out runs/smc-importance-bridge-workflow-20260922 \
  --expected-plan-sha256 PLAN_SHA256_FROM_FREEZE
```

The outputs are `broad-comparison/analysis.json` and
`narrow-comparison/analysis.json`, each with the existing comparator's provenance
and freeze manifest. The bridge retains separate logs and a `status.json`
containing its terminal dependency hashes, exact commands, return codes, output
hashes, and agreement summaries. A first comparison failure stops the sequence;
existing and partial outputs remain intact. A bridge status file is claimed
exclusively, so completed, failed, or interrupted runs cannot be resumed or
overwritten through this wrapper.

`tools/test_run_smc_importance_bridge.py` uses synthetic metadata and a mocked
comparison process. It checks terminal command completeness, original-process
identity, checksum and runtime rejection, source changes during waiting,
sequential execution despite scientific disagreements, failure preservation,
and refusal to overwrite. It creates no physical population or native labels.

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B -m unittest discover \
  -s tools -p 'test_run_smc_importance_bridge.py'
```
