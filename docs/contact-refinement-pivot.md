# Frozen material-pivot refinement

The FFT search identifies candidate contact centers. A covariance estimated
from a refinement trace containing one repeated pose describes only the imposed
regularizer. This opt-in refinement changes the local proposal geometry and
learns a step scale during discarded preparation; it does not change the hard
shape, pair target, depletant bath, or production assembly kernel.

## Symmetric proposal and balance

Choose a material point \(c\) in the moving body's coordinates before the chain
starts. Discovery obtains it from the optimized contact witness:

\[
c=R_0^T(s_0-t_0),\qquad p=t+Rc.
\]

The witness and \(c\) remain fixed, rather than following the instantaneous
nearest contact. The coordinate map \((t,R)\mapsto(p,R)\) has a block-triangular
derivative with identity translation block, so it preserves Cartesian
translation times normalized proper Haar rotation. Draw independent centered
isotropic Gaussian vectors \(d\) and \(a\), then propose

\[
R'=C(a)R,\qquad p'=p+d,
\qquad t'=t+Rc-R'c+d,
\]

where \(C\) is the Cayley rotation. The inverse uses \((-d,-a)\), since
\(C(-a)=C(a)^{-1}\). Both Gaussian densities and the Cayley-to-Haar Jacobian
are invariant under negation. Thus the full pose proposal is symmetric with
respect to the physical translation/Haar measure. The transformation couples
rotation and translation so that a pure rotation holds the selected material
point fixed in the laboratory frame.

Rejecting hard overlaps or origins outside the existing memory ball preserves
this symmetry. Valid endpoints use the unchanged exact conditional-Poisson
gained/lost-volume gate from
[`contact-memory-balance.md`](contact-memory-balance.md). No approximate FFT or
overlap score enters its acceptance. Each frozen proposal therefore preserves
the existing anchored-pair target, including its many-body volume construction
specialized to one fixed partner. This statement does not assert that a finite
trajectory has equilibrated or visited every contact basin.

The original kernel is recovered with no optional pivot. Its arithmetic and
random-number stream remain unchanged when the new options are disabled.

## Discarded adaptation and retained observations

`--refine-adapt` tunes a multiplier \(h\), initially one, shared by the original
translation and angular widths. At the end of each discarded warmup window,

\[
\log h_{next}=\operatorname{clip}
 [\log h+g(\hat a-a_*),\log h_{min},\log h_{max}].
\]

Defaults are window 16 attempts, target acceptance 0.3, gain 2, and scale range
\([10^{-4},10]\). All attempted moves, including hard rejection, enter
\(\hat a\). The final partial warmup window is included. No update occurs after
the configured burn step. This is preparation, not an adaptive equilibrium
claim. Freeze the material point and widths throughout retained sampling.

Every update records its scale, acceptance, and whether it reaches a scale
bound. The output separately records warmup and retained-phase move counts,
the final resolved kernel, and the number of distinct retained poses. Sampling
at a fixed attempted-step stride keeps rejected repeated states. A small step
with high acceptance can still explore slowly, so compare covariance rank,
displacements, autocorrelation, and independent chains instead of acceptance
alone.

## Fixed paired pilot, 2026-09-28

The frozen allocation uses the first **64 source slots in recorded order** from
`fft-union-discovery-768-20260926/frozen-fit-tempered-z0005`. This selection uses
no native label, overlap threshold, covariance quality, or success filter. Each
source pose initializes two independent streams in each of two arms:

| Setting | Legacy | Adaptive material pivot |
|---|---:|---:|
| Source poses | Same 64 | Same 64 |
| Independent streams per source | 2 | 2 |
| Discarded attempts | 512 | 512 |
| Total attempts per stream | 4,608 | 4,608 |
| Retained stride | 8 | 8 |
| Retained poses per stream | 512 | 512 |
| New greedy/FFT search | None | None |
| Radius / activity | 1.4 Å / 0.0275 Å⁻³ | Same |

The resulting 1,179,648 attempts are organized into 32 jobs of eight slots,
with at most four concurrent new jobs and one CPU thread per job. The paired
arms share their master seed and starts; their random-number consumption can
diverge. Replicate/group seeds are `202609280701 + 1009*i`, for `i=0..15`.
Two endpoint-validation clouds per endpoint retain their fixed 8,192-point
allocation and never influence the sampler; the unused search-stage diagnostic
uses 2,048 points. Every slot and attempt remains in the output.

The controller freezes source metrics, shape, executable, Python dependency
closure, commands, and exact source-to-new-slot mappings before any job starts.
It refuses existing output, prevents retries, drains running children after
failure, and reports per-slot/runtime estimates from completed slot logs.
The original fit's first 64 masses and other 704 slots can consequently be
preserved exactly in a later controlled covariance replacement. Any such
replacement still needs frozen-density validation and a matched proposal audit.

Preparation and execution are separate:

```sh
PY=/home/xvg/protein-nucleation/.venv/bin/python
OUT=/vast/xvg/tetramer-mc-runs/contact-refinement-20260928
$PY tools/run_contact_refinement_campaign.py prepare \
  --source-fit runs/fft-union-discovery-768-20260926/frozen-fit-tempered-z0005 \
  --shape examples/tetramer-shape.json \
  --binary target/release/depletion-contact-discovery --out "$OUT"
$PY "$OUT/inputs/python/run_contact_refinement_campaign.py" run --out "$OUT"
$PY "$OUT/inputs/python/run_contact_refinement_campaign.py" status --out "$OUT"
```

Do not recreate or relaunch an existing campaign. A failed or underexplored
slot is a diagnostic outcome, not an excuse to remove that slot. Failure to
sample a basin does not establish instability of native assembly.

## Validation

`tests/refinement_pivot.rs` checks the inverse transform, material-point
displacement, numerical unit Jacobian, zero-bath acceptance and hard rejection,
and bitwise replay of every adaptation step followed by a frozen production
phase. Existing contact-memory and discovery tests check old-stream replay,
held-out cloud independence, sphere overlap references, and complete traces.

## Frozen continuation

`tools/continue_contact_refinement_campaign.py` prepares a separate immutable
continuation package, then runs it only through an explicit `run --out ...`
invocation. A persistent exclusive lock prevents duplicate controllers or
automatic restarts. The controller waits for the existing campaign completion
receipt and release of its workers; it does not start or rerun refinement.

Its fixed sequence is:

1. Analyze the complete allocation, with no native labels.
2. Fit both arms in replicate/group/slot order. Use only refined charts,
   shrinkage 0.02 and covariance floor \(10^{-6}\). Require at least 32 distinct
   retained poses and empirical rank six in **every** chain. ESS remains a
   reported diagnostic; no minimum ESS gate is imposed.
3. If the entire adaptive fit passes, replace the corresponding 64 original
   slots with the two fitted chains per slot. Preserve each old slot's exact
   mass and all other 704 slots. Validate the full reciprocal proposal density.
4. Audit 256 unconditional draws per component from the old and new updated
   submodels, first against one partner and then against all eight predeclared
   seed neighborhoods. These evaluation contexts never enter fitting. Run at
   most four audit jobs at once. Any execution or hard-scaffold validation
   failure drains active audits and prevents the assembly benchmark.
5. Run the same 200-sweep, two-stream-per-arm bounded assembly comparison from
   the frozen physical start, using the old complete atlas and replaced atlas.
6. Run the independent frozen 32-case FFT-peak repair allocation, including if
   a whole-fit quality gate rejected either fit. No failed source slot is
   removed to enable a fit or production run.

All scripts, dependency closures, binaries, input configs, source freeze receipts,
commands and stage outputs are hashed. An ordinary failed stage does not cause
a retry; already active children drain. The independent FFT stage only starts
after the main workers and any dependent stage workers have exited. A user
interruption skips further stages. The two-stream benchmark tests proposal
accessibility, not finite-system equilibrium, thermodynamic stability or a
physical growth rate.
