# Independent proposal and Poisson-intensity controls

The next fixed stage tests whether the completed hard-free line guide gives
consistent contact weights when either its defensive probability or its
auxiliary cloud intensity changes. The physical bath remains at radius
1.5 Å and activity 0.035 Å⁻³. The original R4, repaired shape, scaffold,
Jacobian and complete native classifier stay unchanged.

| Arm | Uniform probability α | Auxiliary λ/z | Fresh populations × attempts |
|---|---:|---:|---:|
| `alpha02` | 0.2 | 128 | 4 × 16,384 |
| `lambda64` | 0.5 | 64 | 4 × 16,384 |

Each control is compared with the **completed conditioned pilot**, which used
α=0.5 and λ/z=128. Both retain the same 92 Gaussian components, xyz hard-free
conditioning, mass-floor fallback, complete mixture density and two independent
clouds per valid pose. No baseline populations are repeated. No comparison
between the two new controls is attributed to a single changed parameter.

The allocation is **131,072 new protein attempts**, with seeds
610014301–610014308 checked for collisions at freeze. Combined with the
131,072-draw pilot and separate 524,288-draw population-size check, this makes
786,432 protein attempts for this guide's three stages. Stages remain separate;
there is no pooling, adaptive extension, retry or reclassification of old poses.

## Why these checks are still needed

Earlier campaigns tested similar probabilities and intensities but used
different proposals and had unresolved weighted tails. They do not replace
these controls. In particular, `poisson_lambda_ratio: 64` in the archived config
does not mean the completed line pilot used that intensity: its command line
explicitly selected 128.

The new analyzer compares independent **linear population means** for the four
decision regions' physical and hard-only masses, as well as total hard-only
mass. Total physical mass is also reported. It retains every radial, angular
and orthant comparison; a physical stratum is material at 1% of its region's
observed mass in either arm. No absent contribution is called a physical zero.

The native-minus-competing free-energy difference is compared directly, in
addition to comparing each mass. Opposite changes of +0.15 and −0.15 in log
mass can each satisfy a 0.2 tolerance while shifting their contrast by 0.30 kBT.
The contrast uses covariance within each population and independence between
arms. Its agreement thresholds remain three combined SE and 0.2 kBT; each
population-based 95% interval must have half-width at most 0.5 kBT.

The existing relative-error, ESS and largest-contribution checks remain.
Passing this stage does not open the full-vessel or assembly gates: the separate
population-size evidence, remaining spatial contributions and finite-system
measurements must still be assessed. The current pilot's eight failed material
strata remain recorded.

## Execution and validation

`tools/run_hard_free_line_sensitivity.py` is a separate controller. Existing
pilot and population-size controllers, production kernels and archived outputs
are unchanged. It authenticates the completed pilot, copies the exact physical
binary/config and preserves the audit, classifier and statistical dependency
bytes. Only the α=0.2 guide changes its defensive-probability field. The λ=64
guide is byte-identical to the original conditioned guide.

Before protein controls can be frozen, an exact-binary reference prerequisite
must pass. The preparer
`tools/prepare_hard_free_line_sensitivity_reference.py` reuses the completed
α=0.5/λ64 sphere references. It declares two new α=0.2/λ128 sphere references:
8,192 unconditional attempts each at activities 0 and 2. These are analytic
normalization/hard-volume/depletion checks with independent full geometry
reconstruction, not additional protein populations. Their mixture has two
components; the downstream protein mixture still has 92.

Preparation launches nothing. Reference execution is explicit and requires
two available physical slots in the host process namespace. No partial result
or failed reference can authorize a protein control. The physical controller
runs at most two jobs, then at most four audit workers, within the global
eight-physical-job/32-worker limits. It stops new launches on failure, drains
started children and preserves attempted-draw journals. Analysis classifies
each new population once, after all physical jobs and audits complete.

Freezing the protein controls requires a successful reference receipt:

```bash
/home/xvg/protein-nucleation/.venv/bin/python -B \
  tools/run_hard_free_line_sensitivity.py freeze \
  --pilot /vast/xvg/tetramer-mc-runs/hard-free-line-physical-pilot-20261001 \
  --prerequisites /absolute/completed/reference/prerequisites.json \
  --out /absolute/fresh/sensitivity-stage
```

Use only the copied controller and its printed protocol hash for a subsequent
`run` invocation. Never rerun a started stage. The final
`sensitivity-comparison.json` binds both input summaries and the frozen
protocol. Successful numerical checks still leave the thermodynamic verdict
unresolved until the remaining coverage and assembly evidence is available.

## Prepared status

The 22 new comparison/controller/reference tests passed, including a direct
contrast counterexample, lost denominators, unobserved masses, extreme finite
log weights, changed α/λ metadata, one-shot execution and failure draining.
The [validation receipt](../results/hard-free-line-sensitivity-validation-v2-20261001/validation.json)
binds the source files and test log. Independent review also authenticated the
completed pilot and all 54 reused Python dependencies.

The sphere-reference inputs are frozen in
`results/hard-free-line-sensitivity-reference-preparation-v2-20261001`, allocation
SHA-256 `e531a10572722785ecdfb61e618024cfbf8e8e5bda344727a30b038130a14b99`.
Seeds 610016211 and 610016212 were checked against 30,294 declarations.
No new physical reference or protein-control job has started, and no successful
reference receipt exists yet. The live population-size stage retains the two
available physical slots alongside six existing user assembly jobs. Its first
two populations have completed; no unfinished weights were used for this design.

V2 supersedes the retained, unlaunched v1 preparation. It pins the Python,
NumPy and SciPy runtime, rejects an optimized Python parent, and explicitly
disables optimization in children so validation assertions remain active.
The superseded preparation must not be executed.

A [supplemental summary-only check](../results/hard-free-line-sensitivity-validation-20261001/completed-pilot-contrast.json)
confirms that the completed pilot's direct free-energy contrast agrees:
conditioned minus baseline is −0.00171 kBT, with combined population SE
0.06660 kBT. This calculation uses only saved summaries and does not remove
the eight previously failed material strata.
