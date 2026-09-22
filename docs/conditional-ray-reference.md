# Conditional-ray contact-weight reference

This experiment tests a proposal change inside the **unchanged six-dimensional
R4 integration region**. It does not modify any assembly kernel or classify a
pose as native merely because it satisfied a guide constraint. The repaired
4004-sphere tetramer, two fixed neighbors, normalized proper-rotation Haar
measure, 1.5 Å depletants and activity 0.035 Å⁻³ are unchanged.

The earlier member-shell proposal put only 831 of 52,320 guided draws inside
R4. That completed experiment is preserved in
[the entry-shell report](entry-shell-reference.md). The new proposal keeps
every draw inside R4 by solving the radial problem *conditional on* the
original angular marginal and a uniform direction. No old physical campaign
is replayed or pooled into the new six-arm experiment.

The [completed results](conditional-ray-reference-results.md) fail the
predeclared convergence gate: the larger guided native/no-entry ESS values are
1.1/2.7, with dominant single draws and incompatible stratum contributions.
The subsequent full-vessel and assembly production remain gated. This is an
unresolved sampling outcome, not a conclusion that the model fails to assemble.

## Conditional geometry and complete density

Let the original ordinary chart be `x=m+L u`, with `||u||≤R=4`, and split
`x=(t,ω)` into translation and scaled Cayley coordinates. Write `Σ=LLᵀ`.
For a fixed angular coordinate,

\[
 \mu(\omega)=m_t+\Sigma_{t\omega}\Sigma_{\omega\omega}^{-1}(\omega-m_\omega),
 \qquad
 CC^T=\Sigma_{tt}-\Sigma_{t\omega}\Sigma_{\omega\omega}^{-1}\Sigma_{\omega t}.
\]

If `A Aᵀ=Σωω`, the available conditional radius is
`s²=R²−||A⁻¹(ω−mω)||²`. The conditional translation is

\[
 t=\mu+Cnr,\qquad n\sim U(S^2),\qquad 0\le r\le s.
\]

The baseline conditional radial density is `3r²/s³`. The angular marginal
is obtained by drawing once from the original uniform six-ball and retaining
its angular coordinates; it includes the conditional-volume factor `s³`.
It is not a uniform angular ellipsoid, a Gaussian, or uniform SO(3).

The chart's anchor and fixed-neighbor transform map this ray into world
coordinates. At fixed orientation, each member center is affine in r, so
its squared displacement from a prescribed member center gives a quadratic
inequality. Let `I_A(b)` and `I_B(b)` be the intervals satisfying **all four**
member-distance constraints of interfaces A7 and B4 at threshold b. The
boundary intervals for width δ are

\[
 I_\delta=[I_A(2+\delta)\cup I_B(2+\delta)]
              \setminus[I_A(2)\cup I_B(2)],
 \qquad \delta\in\{0.02,0.1,0.5\}\ {\rm \AA}.
\]

Thus the guide approaches the existing entry threshold from its outer side.
Interval unions prevent double counting the two interfaces. Tangencies have
zero measure. All intervals are clipped to `[0,s]`.

Choose the three widths equally. For disjoint pieces `[a,b]`, choose a piece
in proportion to `b³−a³` and draw uniformly in r³ within it. If the selected
width has no interval, use the baseline radial law on the **same angular
coordinate and direction**. There is no valid-only retry or direction redraw.

Writing `Mδ=Σ(b³−a³)` and `V6=π³R⁶/6`, the complete density with respect to
the original whitened six-dimensional Lebesgue measure is

\[
 q(u)=\frac{1_{\|u\|\le R}}{V_6}
 \left[\alpha+\frac{1-\alpha}{3}\sum_\delta F_\delta(u)\right],
 \quad
 F_\delta=
 \begin{cases}
 s^3 1_{r\in I_\delta}/M_\delta,&M_\delta>0,\\
 1,&M_\delta=0.
 \end{cases}
\]

The denominator sums **every width**, irrespective of the selected branch.
The primary defensive fraction is `α=0.5`. Each radial component integrates
to one conditionally; consequently this is a normalized density on the same
ball. Its uniform floor retains every original pose. At `α=1`, the Rust
physical rows are bitwise identical to the legacy uniform sampler.

For each of N unconditional draws, the retained contribution is

\[
 Y=1_{\rm hard,domain,region}\frac{J(u)}{q(u)}
      \frac{W_1+W_2}{2},\qquad
 J=\frac{|\det L|}{\ell^3\pi^2(1+|\omega/\ell|^2)^2}.
\]

Here the two independent clouds satisfy `Kj~Pois(λ Cphysical)` and
`Wj=(1+z/λ)^Kj`. `Cphysical` is the overlap with the complete neighbor
exclusion union, including its many-body shielding. The estimator is
`ΣY/N`, with a zero for every hard-invalid draw. Member constraints guide
proposals only. The complete frozen native/patch classifier is applied
separately, including all angle, atomic and residue requirements.

## Fixed allocation and gates

| Arm | Populations × unconditional draws | Uniform fraction | λ/z |
|---|---:|---:|---:|
| Pilot uniform | 4 × 16,384 | 1 | 64 |
| Pilot ray | 4 × 16,384 | 0.5 | 64 |
| Larger uniform | 4 × 65,536 | 1 | 64 |
| Larger ray | 4 × 65,536 | 0.5 | 64 |
| Defensive-fraction sensitivity | 4 × 16,384 | 0.2 | 64 |
| Cloud-intensity sensitivity | 4 × 16,384 | 0.5 | 128 |

The total is **786,432 draws**, 24 fresh independently seeded populations,
two clouds per valid pose and at most eight simultaneous physical jobs.
Stages remain separate. The frozen protocol records all commands, seeds,
input hashes, native definition and complete Rust/Python source closures.
Preflight rejects existing outputs. A physical failure stops further launches,
drains already started children and preserves their outputs without retries.
All physical populations finish before any of the six independent density
audits. The controller binds audited branch/component/fallback counts to each
population's original rows and unconditional denominator.

The decision gates require ≤10% between-population relative SE, importance
ESS≥200, largest row≤2%, and a population-based 95% free-energy interval with
half-width≤0.5 kBT. Proposal, population-size and cloud-intensity comparisons
must agree both within three combined SE and 0.2 kBT. Fixed radial bins
`[0,2,3,4]`, angular projected-squared-radius bins `[0,4,9,16]`, and all 64
original latent orthants are retained. No stratum is omitted because it has
zero observed weight. Significant stratum disagreement is a failed diagnostic.

These criteria do not prove coverage of unseen modes. In particular, a finite
R4 comparison does not bound the full-vessel contact remainder. A passed
regional gate permits a separate full-vessel mixture comparison; a failed
gate leaves physical assembly unresolved and stops that dependent campaign.

## Implementation and validation

- Rust: `src/latent_region/conditional_ray.rs`, dispatched through the existing
  `latent-region-normalizer --importance-guide` option. Previous Gaussian and
  member-shell schemas are retained.
- Python: `tools/conditional_ray_proposal.py` independently reconstructs the
  covariance, Schur complement, world geometry, interval mixture and density.
  Its quadratic-root implementation differs from Rust's projected-line roots.
- Controller: `tools/run_conditional_ray_campaign.py` freezes and executes
  the allocation. Population and campaign schemas are versioned separately.
- Analysis: `tools/analyze_conditional_ray_campaign.py` uses the original
  classifier, linear regional masses and independent-population uncertainty.

The cross-language sphere validation uses 32,768 fresh toy draws. Its maximum
log-density discrepancy is 6.22×10⁻¹⁵; all proposals remain inside their ball.
Both hard-only Haar volumes and analytic pair-depletion integrals agree within
1.03 observed SE. Tests also cover interval set algebra, tangencies, empty-ray
fallback, covariance factorization, normalization, full-mixture reconstruction,
global isometries, the pure-uniform limit and corrupted provenance. Hard-only
protein measures are recorded alongside every physical arm.

The runner's assembly checkpoints have an independently tested exact
continuation path. The regional reference controller does **not** resume a
partially written population: per-draw RNG streams are deterministic, but
interrupted physical populations remain failed rather than being silently
replaced. This distinction is retained in the campaign record.

Completed toy evidence is in
[`conditional-ray-sphere-validation-20260921`](../runs/conditional-ray-sphere-validation-20260921/validation.json).
The exact protein allocation is in
[`mobile-conditional-ray-campaign-20260921`](../runs/mobile-conditional-ray-campaign-20260921/protocol.json).

## Relationship to the remaining assembly plan

The first physical endpoint is equilibrium **finite-system** native assembly
at approximately 106.8 μM tetramers. N=12 and N=24, three preparations, four
streams per preparation/arm, native-informed controls and an equal-volume
spherical/periodic comparison remain conditional on reliable contact weights.
Common-boundary comparisons must use only kernels valid for both boundaries.
No accepted-move throughput or short conditional-scaffold trajectory decides
that endpoint. Contact-fingerprint ESS per CPU, initial-condition agreement,
completed exchanges and independent free-energy evidence are required.

The [frozen cluster-size bias](frozen-assembly-bias.md) is implemented for those
later measurements, with a correction after each elementary reversible kernel
and physical reweighting by exp(B). It does not supply an equilibrium result.
The [Lean bridge](../formal/README.md) proves the Poisson mean/variance and
the implemented conditional gained/lost-count gate under explicit count-law
and proposal hypotheses. Geometric thinning, coordinate Jacobians, finite
precision and convergence remain implementation obligations.

Earlier SMC comparisons are reused from the
[same-target reconciliation](smc-reference-reconciliation.md). A regional SMC
estimate is `Zhat × terminal mean(indicator)`, not an endpoint fraction alone.
Only matching shape, scaffold, capture/wall domain, Haar normalization and
complete region labels can be compared. The old A-only q≤2 control was
reconciled; the present observed-AB R4 is a different target. Its result cannot
be subtracted from an old single-neighbor capture normalizer and called an SMC
discrepancy. The full one-neighbor far remainder remains unresolved.
