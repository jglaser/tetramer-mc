# Independent-stream finite-assembly comparison

`tools/compare_finite_assembly.py` compares one complete block of frozen,
unbiased assembly observations. It requires **36 reports**: local, independent
redraw and correlated transport, each from dispersed, competing-aggregate and
native-seeded preparations, with four independent streams in every group.
It does not launch simulations, choose an observation window or declare
equilibrium. This is a measurement capability; no such physical block has yet
passed the upstream contact-weight gates.

Every report must use `finite-assembly-observation-v1` and the same predeclared
benchmark contract. The complete physical target, boundary, particle count,
measurement definitions, observer implementation, saved cadence and window must
match. The existing contract checks proposal laws, local widths, collective
schedules, auxiliary intensity and geometric work budgets. Independent redraw
has no posterior component selection. The local arm has no global attempts.
All seeds are distinct uint64 values, run directories are distinct, and resumed
segments cannot substitute for independent streams. Within each preparation,
the initial-pose hash multisets must match across arms; this does not imply
paired-stream statistical inference.

The two sizes, boundaries and model families remain separate blocks. They are
not pooled into a common physical distribution. The current executable's
[boundary limitations](finite-assembly-runtime-and-boundaries.md) must be
resolved before declaring a compatible learned periodic block.

## Authentication and retained-state accounting

The file interface authenticates both frozen observer outputs, the observer
definition and implementation dependencies, the bound analysis plan and
benchmark contract, and the complete original run inputs. It reuses the existing
effective-configuration and retained-frame validators. The report's CPU interval,
attempt counters, initial poses, seed, proposal and schedule must match the
original trajectory and configuration. Merely rescaling a reported CPU rate and
resealing its output files therefore fails.

Cached observations must contain every original saved frame in order, including
frames outside the selected window. Within the window, the initial frame is a
baseline; each subsequent retained endpoint contributes once, including repeated
states. The comparator checks region counts, half-window counts, dictionaries
and constant-token counts, component and joint-size histograms, passages and
censored episodes against the saved observations. It also checks the recorded
graph/region bookkeeping. It does **not** repeat atomic geometry, native-entry
classification, catalogue-cycle solving or FFT autocorrelation calculations.
Thus authentication and bookkeeping checks reuse the original observer's
validated geometric and statistical implementation; they do not independently
prove every physical predicate or estimated autocorrelation.

## Population statistics

Each arm/preparation group retains its four individual stream values. Occupancy
means use equal weight per independent stream, with standard error `s/sqrt(4)`.
Descriptive 95% intervals use Student-t with three degrees of freedom. These
intervals are reported without clipping to physical bounds and are not
simultaneous confidence intervals over all observables. If the observed variance
is zero, the interval is `null` and the zero-variance flag remains explicit.
All-zero native observations do not become an upper bound on native probability.

Initialization contrasts compare preparations within one proposal arm. Proposal
contrasts compare arms within one preparation. Their errors combine independent
population standard errors; descriptive difference intervals use Welch degrees
of freedom. Three-observed-SE agreement is only a diagnostic. When both estimated
errors vanish, the agreement field is `null`, even if the observed means are
identical. Common trapping, unseen modes and nonstationarity remain possible.

Patch-contact and native-motif fingerprint ESS per full sampler CPU are reported
separately. A null ESS in any stream leaves the group's efficiency and associated
speedup unresolved; that stream is never dropped or converted to zero. Reported
ratios compare population mean rates. They have no claimed ratio confidence
interval and establish neither an equilibrium speedup nor physical attachment
kinetics. Observer CPU is excluded from the sampler denominator.

All component-size bins are retained. Exclusion and raw-native component
histograms must each conserve the full number of bodies. Certified-native
histograms may represent less than the full body count when registry is
unresolved. Their represented body fraction is reported without renormalizing
away the missing mass. Joint size bins include unresolved states; absent bins in
one stream receive zero *empirical* occupancy, not an equilibrium-zero claim.

Passages and completed roundtrips are counted within each stream. No stream's
terminal state is joined to another's initial state. `remaining` retains its
occupancy and censored episodes and may lie between two named environments;
a third named environment interrupts a pair roundtrip. The comparator preserves
per-stream directional counts and histories. The current observer supplies no
independent finite-system probability reference, so missing returns remain
thermodynamically unassessed. Conditional scaffold masses cannot supply that
reference, and unequal occupancies impose no equal-rate requirement.

## Interface

The pure `compare_reports(reports)` API validates the complete declared block
and performs aggregation. `compare_report_files(paths)` additionally authenticates
the file and original-run bindings and checks cached records. The command-line
interface uses the latter:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  /home/xvg/protein-nucleation/.venv/bin/python tools/compare_finite_assembly.py \
  --reports /path/to/completed-block/observers/*/analysis.json \
  --out /path/to/fresh-comparison
```

The glob must resolve to all 36 reports from one actual matched block. Outputs
include all group values, descriptive
contrasts, apparent-efficiency ratios, input hashes, source files and runtime
versions. `convergence_established` remains false: completing this comparison
does not by itself satisfy the physical convergence and stability requirements.

## Validation

[Twenty-six focused tests passed](../runs/finite-assembly-comparison-validation-20260924/validation.json).
They cover the complete synthetic 36-stream matrix, four-population uncertainty,
null efficiency, unresolved native mass, and per-stream exchange accounting.
Separate file tests create synthetic observer records and isolate authentication;
they reject resealed CPU/rate changes, dropped frames, altered counters and
misidentified proposals, preparations or initial poses. The comparator itself
performs no geometry, native-classification or FFT replay. These tests are not a
completed physical assembly comparison.
