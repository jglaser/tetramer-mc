# Gated full-vessel execution

`tools/run_full_vessel_comparison.py` consumes the unchanged preparation made by
`prepare_full_vessel_comparison.py`. It provides an inert `freeze`, read-only
repeatable `preflight`, and one-shot `run`. It never launches assembly sampling.
The current regional confirmation fails its convergence gate; this controller
must not be used to bypass that result.

The allocation remains 4 populations of 65,536 attempts per arm, followed by
4 fresh populations of 262,144 attempts per arm: 2,621,440 total. One arm uses the
original vessel proposal; the other uses its normalized half-mixture with the
frozen regional guide. Both have the full atomic-wall physical domain. There is
no guide refitting or adaptive allocation between stages.

Before any physical child, the controller authenticates:

- The preparation, exact binary/source bundle, all inputs and archived code.
- The completed optimized-release sphere references using that same binary.
- The frozen regional confirmation and all its required convergence checks.
- The successful SMC comparison workflow, its exact commands, terminal output
  and source bindings, distinct broad/narrow protocol identities, and matching
  regional masses. Material contradictions fail the gate; unresolved precision
  remains explicit under the existing gate contract.

A frozen failing confirmation cannot later become passing without changed bytes.
Freezing a workflow is therefore preparation, not permission to launch or a
promise that its scientific gate will eventually pass. A different evidence
binding requires a separately reviewed freeze; it cannot silently replace one.

Every standard population must finish, then every independent audit, exhaustive
contact/pocket partition, and authenticated stage summary must finish before any
large population starts. Stage quality flags remain diagnostic: failure of a
precision threshold does not discard the already allocated larger comparison.
Stage and proposal populations are never pooled to manufacture agreement.
See [stage comparison](full-vessel-stage-comparison.md).

There are at most eight physical children, followed by at most four audit or
partition children and one aggregator. Single-thread numerical environments are
set for every child. Before each launch, `/proc` accounting checks the eight
physical-job and 32 scientific-worker limits, including other same-user jobs in
this workspace. Uncoordinated external launches can still race with that check;
other campaigns should use the same caps. OS and browser threads are excluded.
Launch the controller itself with all numerical thread limits set to one.

`run` creates an exclusive immutable launch claim. A second claim on the
preparation prevents another workflow from consuming the same allocation.
Controller and child process-birth identities are recorded. Failures are polled
before refilling slots, stop new launches, and drain all started children.
Catchable SIGTERM and keyboard interruption also drain; uncatchable process or
machine death cannot be repaired by a Python exception handler. Logs, retained
attempts and return codes are preserved. No retry, deletion or overwrite occurs.

`basin-normalizer` has no resumable checkpoint interface. This workflow therefore
refuses run continuation after a launch claim, even if the original controller
has disappeared. A missing process is evidence requiring review, never permission
to regenerate attempts. Repeating `preflight` before a first launch is supported.
The existing conditional-normalizer continuation validations are unchanged.

After each child, its outputs are bound. Executable/source/input bytes are
rechecked before each next launch; accumulated completed evidence is checked
before the next stage and at completion. Each audit independently checks every
attempt, proposal density, invalid zero and complete native classifier. The
stage aggregator authenticates full population/audit/partition lineage and all
reporting-region definitions. Original preparation frozen files are unchanged;
the claim and physical outputs are separate added artifacts.

The CLI deliberately requires explicit expected SHA256 values:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/xvg/protein-nucleation/.venv/bin/python \
  /home/xvg/tetramer-mc/tools/run_full_vessel_comparison.py freeze \
  --out /absolute/new-workflow \
  --preparation /absolute/frozen-preparation --expected-preparation-sha256 PLAN_SHA \
  --release /absolute/release/validation.json --expected-release-sha256 VALIDATION_SHA \
  --bridge /absolute/frozen-bridge --expected-bridge-sha256 BRIDGE_PLAN_SHA
```

Use the resulting workflow's `common/run_full_vessel_comparison.py` for
`preflight` or `run`, with `--out` and `--expected-plan-sha256`. No production run
is warranted while the fixed contact-weight checks fail. Successful execution
would supply conditional full-vessel evidence, not a finite-system assembly or
bulk-stability conclusion.
