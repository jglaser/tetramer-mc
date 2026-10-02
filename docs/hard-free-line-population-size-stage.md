# Fixed population-size sensitivity stage

**Completed:** all sampling, independent audits and classification finished.
See the [results](hard-free-line-population-size-results.md). Aggregate masses
agree, but the fixed convergence gate fails material-stratum checks; no
full-vessel or assembly conclusion follows. The launch notes below are a record
of the existing campaign, not instructions to repeat it.

This stage adds four fresh 65,536-draw populations per arm: **524,288 new
unconditional draws**, compared separately with the completed four-by-16,384
pilot. The schema is `hard-free-line-physical-population-size-v1`; the old
`hard-free-line-physical-pilot-v1` allocation and CLI remain supported. There
is no unrestricted sample-count option or automatic extension.

Both stages use the identical repaired hard shape, radius 1.5 Å, activity
0.035 Å⁻³, R4 target, original chart/Jacobian, two independent clouds at
auxiliary intensity ratio 128, defensive probability 0.5, and frozen
92-component baseline/conditioned guides. The new stage uses seeds
610013301–610013308 after checking the existing declaration inventory. It
allows two physical workers, followed by four audit workers, within the
32-worker overall cap. It starts no other simulations.

## What is frozen

`freeze-population-size` requires the original pilot's terminal controller,
eight successful physical jobs, eight complete independent geometry audits,
and hash-bound final classification summary. No old raw rows, geometry,
Poisson clouds or classifiers are replayed.

The new directory copies the exact old physical binary, source bundle, Rust
source, config, reference package, guides and prerequisite receipt. The config
bytes stay unchanged, including the existing absolute shape path into the
archived pilot; that shape remains hash-bound, and the new directory also
contains its identical copy. The validated audit/classifier/statistics Python
files must still match the pilot byte for byte. Only the controller and the
new aggregate comparison helper differ. Pilot evidence and the comparison
plan are frozen before any new job starts.

The controller exclusively creates `status.json`, so concurrent invocations
cannot overwrite accounting. Failure drains started children; there are no
retries or continuation. Every attempted draw is preserved. All physical
populations must finish before independent auditing, and all audits must
finish before the single classification pass on the **new** draws.

## Predeclared comparisons

Each stage retains its own four independent linear population means. The
stages are never pooled, and no pilot weights are substituted into the new
estimates. For each arm and decision-driving region, the size comparison
reports both physical and hard-only masses, all four means from each stage,
the difference of the linear means, and its standard error

\[
\mathrm{SE}(\overline Q_{65k}-\overline Q_{16k})
=\sqrt{s^2_{65k}/4+s^2_{16k}/4}.
\]

The statistical agreement check uses the **linear difference** within three
of these standard errors. A separate check requires the absolute log mass
ratio to be at most 0.2. The delta-method log-ratio SE is reported only as a
diagnostic. The scaled linear comparison admits 32 machine eps times the
larger scaled mean for arithmetic roundoff; it does not introduce a physical
tolerance. Log masses and ratios of estimated masses have finite-sample bias.

All radial, angular and 64 orthant comparisons remain visible. A stratum is
material when its observed fraction of its region's physical mass reaches
1% in either stage of that arm; every failed material stratum is listed.
The original per-stage quality checks and native-versus-competing free-energy
intervals are retained without changing their definitions. Unobserved masses
remain unresolved, not physical zeros or upper bounds.

Completion writes `population-size-comparison.json`, hashes it in terminal
status, and leaves full-vessel and assembly gates closed. The result is a
population-size sensitivity check, not an independent coverage guarantee or
evidence of finite-system assembly or instability.

## Executed commands and live status

The stage was frozen and launched after fourteen controller/comparison tests
and an independent source review passed. The
[validation receipt](../results/hard-free-line-population-size-controller-validation-20261001/validation.json)
binds the source and test logs. The physical binary remains the archived
`752c5d7aa36249cb923ca0959b0872b4f96204101090ac1cfca4606732554c7b` executable;
all 54 reused Python dependencies match the pilot archive.

Live directory:
`/vast/xvg/tetramer-mc-runs/hard-free-line-population-size-20261001`.
Protocol SHA-256:
`cbcf1400a38365f2fa24585110da389b64174fc7e3a563e26cf1618a161c588e`.
Controller PID: 541958. Startup verified two running physical workers and six
preexisting user simulations. Read `status.json` for subsequent progress.
The following commands document the launch; **do not rerun a started stage**.

Freezing performs only copying and validation:

```bash
python -B tools/run_hard_free_line_physical_pilot.py freeze-population-size \
  --pilot /vast/xvg/tetramer-mc-runs/hard-free-line-physical-pilot-20261001 \
  --out /vast/xvg/tetramer-mc-runs/hard-free-line-population-size-20261001
```

The command prints the frozen protocol hash. A later explicit invocation of
the copied controller starts the declared jobs:

```bash
python -B /vast/xvg/tetramer-mc-runs/hard-free-line-population-size-20261001/common/run_hard_free_line_physical_pilot.py run \
  --out /vast/xvg/tetramer-mc-runs/hard-free-line-population-size-20261001 \
  --expected-protocol-sha256 PRINTED_HASH
```

The controller was launched detached with output in `controller.log` and its
launch record in `launch.json`. No job is restarted when a status poll expires.
Physical and assembly conclusions remain unresolved while this stage runs.
