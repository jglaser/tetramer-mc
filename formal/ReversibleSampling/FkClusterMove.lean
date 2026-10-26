import ReversibleSampling.Involution
import ReversibleSampling.PairFlowPoisson

/-! Fortuin-Kasteleyn (FK) cluster boundary cancellation and involution cluster balance
for rigid-cluster Monte Carlo and proper half-turn GCA under a general multi-body
Hamiltonian `E_total = E_surrogate + E_residual` (such as pairwise square-well +
like-charge + chemistry residual plus many-body morphometric solvation).

1. **FK Boundary Cancellation (`fk_boundary_flow_balance`)**:
   Suppose a subset `C` of mobile bodies is recruited via Fortuin-Kasteleyn bond
   probabilities `p_ij(x) = 1 - exp(min(0, E_cut(x)))` where `E_cut(x)` is the
   attractive part (`min 0 E_surr(x)`) of the surrogate cut energy between `C` and its
   complement `V \ C`, so the non-bonding probability across the cut is
   `P_cut(x) = exp(min 0 (E_surr(x)))`.
   When the internal surrogate energy of `C` and of `V \ C` are preserved by the rigid
   move (`E_total(y) - E_total(x) = (E_surr(y) - E_surr(x)) + (E_res(y) - E_res(x))`),
   accepting the rigid move with the residual + repulsive-cut Metropolis ratio
   `min 1 ((q(y,x) / q(x,y)) * exp(-(E_res(y) - E_res(x)) - (max 0 (E_surr(y)) - max 0 (E_surr(x)))))`
   satisfies exact detailed balance with respect to the full Boltzmann weight
   `exp(-(E_surr + E_res))`: the attractive cut bond-breaking penalty cancels identically
   against the forward/reverse FK cut non-bonding probabilities `P_cut(x)` and `P_cut(y)`,
   enabling barrier-free cluster merging and splitting.

2. **Measure-Preserving Involution GCA (`fk_involution_residual_correct`)**:
   For a measure-preserving involution `T` (such as a centered proper half-turn
   `HalfTurn` flipping an FK component while keeping the fixed seed unflipped),
   gating by the residual energy change `min 1 (p(T x) / p(x))` yields a Markov,
   reversible, and invariant kernel on the general measurable state space.
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace ReversibleSampling.FkClusterMove

/-- Attractive part `min(0, e)` of a cut energy `e` (in units of `kT`). -/
noncomputable def attractivePart (e : ℝ) : ℝ := min 0 e

/-- Repulsive part `max(0, e)` of a cut energy `e` (in units of `kT`). -/
noncomputable def repulsivePart (e : ℝ) : ℝ := max 0 e

/-- Every real cut energy decomposes into its attractive and repulsive parts:
`min(0, e) + max(0, e) = e`. -/
theorem attractive_add_repulsive (e : ℝ) :
    attractivePart e + repulsivePart e = e := by
  unfold attractivePart repulsivePart
  simpa using min_add_max 0 e

/-- Fortuin-Kasteleyn cut non-bonding probability `P_cut(e) = exp(min(0, e))`.
When `e < 0` (attractive cut), the bond probability is `1 - exp(e) ∈ (0, 1)` and the
non-bonding probability is `exp(e)`. When `e ≥ 0` (repulsive/non-interacting cut),
the bond probability is `0` and `P_cut(e) = 1`. -/
noncomputable def fkCutSurvival (e : ℝ) : ℝ := Real.exp (attractivePart e)

theorem fkCutSurvival_pos (e : ℝ) : 0 < fkCutSurvival e :=
  Real.exp_pos _

theorem fkCutSurvival_le_one (e : ℝ) : fkCutSurvival e ≤ 1 := by
  unfold fkCutSurvival attractivePart
  exact Real.exp_le_one_iff.mpr (min_le_left 0 e)

/-- Full Boltzmann weight `π(e_surr, e_res) = exp(-(e_surr + e_res))`. -/
noncomputable def boltzmannWeight (e_surr e_res : ℝ) : ℝ :=
  Real.exp (-(e_surr + e_res))

/-- Effective weight appearing in the FK-cancelled Metropolis acceptance ratio:
only the repulsive part of the cut surrogate energy and the residual multi-body energy
(e.g. morphometric solvation residual) enter the acceptance gate! The attractive cut
energy cancels against `fkCutSurvival`. -/
noncomputable def fkEffectiveWeight (e_surr e_res : ℝ) : ℝ :=
  Real.exp (-(repulsivePart e_surr + e_res))

/-- Fundamental FK identity: the full Boltzmann weight times the FK cut non-bonding
probability equals the effective residual weight (`exp(-(max(0, e_surr) + e_res))`). -/
theorem boltzmann_mul_fkCutSurvival (e_surr e_res : ℝ) :
    boltzmannWeight e_surr e_res * fkCutSurvival e_surr =
      fkEffectiveWeight e_surr e_res := by
  unfold boltzmannWeight fkCutSurvival fkEffectiveWeight
  rw [← Real.exp_add]
  congr 1
  have h := attractive_add_repulsive e_surr
  linarith

/-- Exact pointwise detailed balance for FK-recruited rigid cluster moves (including
rigid cluster translation/rotation, reinsertion, and cluster docking with asymmetric
proposal densities `q_fwd`, `q_rev` ≥ 0):
`π(x) P_cut(x) q_fwd min(1, (fkEffectiveWeight(y) q_rev) / (fkEffectiveWeight(x) q_fwd))`
is symmetric under swapping `(x, q_fwd) ↔ (y, q_rev)`. -/
theorem fk_boundary_flow_balance
    (es_x er_x es_y er_y q_fwd q_rev : ℝ)
    (hq_fwd : 0 < q_fwd) (hq_rev : 0 < q_rev) :
    let w_x := fkEffectiveWeight es_x er_x
    let w_y := fkEffectiveWeight es_y er_y
    boltzmannWeight es_x er_x * fkCutSurvival es_x * q_fwd *
        min 1 ((w_y * q_rev) / (w_x * q_fwd)) =
      boltzmannWeight es_y er_y * fkCutSurvival es_y * q_rev *
        min 1 ((w_x * q_fwd) / (w_y * q_rev)) := by
  intro w_x w_y
  have hwx : 0 < w_x := Real.exp_pos _
  have hwy : 0 < w_y := Real.exp_pos _
  have hfx : 0 < w_x * q_fwd := mul_pos hwx hq_fwd
  have hfy : 0 < w_y * q_rev := mul_pos hwy hq_rev
  have h1 : boltzmannWeight es_x er_x * fkCutSurvival es_x * q_fwd = w_x * q_fwd := by
    rw [boltzmann_mul_fkCutSurvival]
  have h2 : boltzmannWeight es_y er_y * fkCutSurvival es_y * q_rev = w_y * q_rev := by
    rw [boltzmann_mul_fkCutSurvival]
  rw [h1, h2]
  rw [mul_min_of_nonneg _ _ (le_of_lt hfx), mul_min_of_nonneg _ _ (le_of_lt hfy)]
  rw [mul_one, mul_one, mul_div_cancel₀ _ (ne_of_gt hfx),
    mul_div_cancel₀ _ (ne_of_gt hfy), min_comm]

/-- Measure-theoretic FK cluster balance on a general measurable state space `X`:
when the FK-recruited cluster proposal kernel `Q` has symmetric reference pair flow
under the surrogate weight `w_surr` (i.e. `((μ.withDensity w_surr) ⊗ₘ Q).map Prod.swap =
(μ.withDensity w_surr) ⊗ₘ Q`), gating by the residual multi-body weight `w_res`
(`involutionAcceptance w_res`) produces a Markov, reversible, and invariant kernel for
the full physical target `(μ.withDensity w_surr).withDensity w_res`. -/
theorem fk_surrogate_residual_correct {X : Type*} [MeasurableSpace X]
    (μ : Measure X) (w_surr w_res : X → ℝ≥0∞)
    [SFinite (μ.withDensity w_surr)]
    (hsurr : Measurable w_surr) (hres : Measurable w_res)
    (hres_finite : ∀ x, w_res x ≠ ⊤)
    (Q : Kernel X X) [IsMarkovKernel Q]
    (hswap : ((μ.withDensity w_surr) ⊗ₘ Q).map Prod.swap = (μ.withDensity w_surr) ⊗ₘ Q) :
    let K := complete (accepted Q (involutionAcceptance w_res))
    let π := μ.withDensity (fun x => w_surr x * w_res x)
    IsMarkovKernel K ∧ Kernel.IsReversible K π ∧ Kernel.Invariant K π := by
  intro K π
  have hα := involutionAcceptance_measurable w_res hres
  have hpt : ∀ x y, w_res x * involutionAcceptance w_res x y =
      w_res y * involutionAcceptance w_res y x := by
    intro x y
    unfold involutionAcceptance
    rw [mass_mul_mh _ _ (hres_finite x), mass_mul_mh _ _ (hres_finite y), min_comm]
  have hsym := pair_flow_accepted_flow_symmetric (μ.withDensity w_surr) Q hswap
    w_res (involutionAcceptance w_res) hres hα hres_finite hpt
  have hcorr := accept_reject_correct Q (involutionAcceptance w_res)
    ((μ.withDensity w_surr).withDensity w_res) hα (fun _ _ => min_le_left _ _) hsym
  have hmul : (μ.withDensity w_surr).withDensity w_res = π := by
    rw [← withDensity_mul μ hsurr hres]
    rfl
  simpa only [hmul] using hcorr

/-- General inverse-temperature (`fk_beta`) Fortuin-Kasteleyn cut non-bonding probability
`P_cut(β_fk, e) = exp(β_fk * min(0, e))` used in `MorphMc::step_cluster` (`cs.fk_beta ≥ 0`). -/
noncomputable def fkBetaCutSurvival (β_fk e : ℝ) : ℝ :=
  Real.exp (β_fk * attractivePart e)

/-- Effective weight appearing in the `fk_beta`-scaled cluster Metropolis ratio:
`exp(-(e_surr - β_fk * min(0, e_surr) + e_res))`. When `β_fk = 1`, this reduces to
`fkEffectiveWeight`; when `β_fk = 0`, it reduces to `boltzmannWeight`. -/
noncomputable def fkBetaEffectiveWeight (β_fk e_surr e_res : ℝ) : ℝ :=
  Real.exp (-(e_surr - β_fk * attractivePart e_surr + e_res))

/-- Fundamental `fk_beta` identity: `boltzmannWeight * fkBetaCutSurvival = fkBetaEffectiveWeight`
for any `β_fk ∈ ℝ`. -/
theorem boltzmann_mul_fkBetaCutSurvival (β_fk e_surr e_res : ℝ) :
    boltzmannWeight e_surr e_res * fkBetaCutSurvival β_fk e_surr =
      fkBetaEffectiveWeight β_fk e_surr e_res := by
  unfold boltzmannWeight fkBetaCutSurvival fkBetaEffectiveWeight
  rw [← Real.exp_add]
  congr 1
  ring

/-- Exact pointwise detailed balance for `fk_beta`-recruited rigid cluster moves
(`MorphMc::step_cluster` with general `cs.fk_beta` and `fk_log_survival_diff =
β_fk * (min(0, e_new) - min(0, e_old))`). -/
theorem fk_beta_boundary_flow_balance
    (β_fk es_x er_x es_y er_y q_fwd q_rev : ℝ)
    (hq_fwd : 0 < q_fwd) (hq_rev : 0 < q_rev) :
    let w_x := fkBetaEffectiveWeight β_fk es_x er_x
    let w_y := fkBetaEffectiveWeight β_fk es_y er_y
    boltzmannWeight es_x er_x * fkBetaCutSurvival β_fk es_x * q_fwd *
        min 1 ((w_y * q_rev) / (w_x * q_fwd)) =
      boltzmannWeight es_y er_y * fkBetaCutSurvival β_fk es_y * q_rev *
        min 1 ((w_x * q_fwd) / (w_y * q_rev)) := by
  intro w_x w_y
  have hwx : 0 < w_x := Real.exp_pos _
  have hwy : 0 < w_y := Real.exp_pos _
  have hfx : 0 < w_x * q_fwd := mul_pos hwx hq_fwd
  have hfy : 0 < w_y * q_rev := mul_pos hwy hq_rev
  have h1 : boltzmannWeight es_x er_x * fkBetaCutSurvival β_fk es_x * q_fwd = w_x * q_fwd := by
    rw [boltzmann_mul_fkBetaCutSurvival]
  have h2 : boltzmannWeight es_y er_y * fkBetaCutSurvival β_fk es_y * q_rev = w_y * q_rev := by
    rw [boltzmann_mul_fkBetaCutSurvival]
  rw [h1, h2]
  rw [mul_min_of_nonneg _ _ (le_of_lt hfx), mul_min_of_nonneg _ _ (le_of_lt hfy)]
  rw [mul_one, mul_one, mul_div_cancel₀ _ (ne_of_gt hfx),
    mul_div_cancel₀ _ (ne_of_gt hfy), min_comm]

/-- Forward GCA cross-energy non-bonding probability `P_gca(β_fk, e_old, e_cross) =
exp(-β_fk * max(0, e_cross - e_old))` used in `MorphMc::step_gca` when pair `(u, v)`
has `u ∈ C`, `v ∉ C`, and `de = e_cross - e_old`. -/
noncomputable def gcaCutSurvival (β_fk e_old e_cross : ℝ) : ℝ :=
  Real.exp (-β_fk * repulsivePart (e_cross - e_old))

/-- Ratio identity for `MorphMc::step_gca`: because `max(0, -de) - max(0, de) = -de`,
the ratio of reverse-to-forward GCA cut survival probabilities is
`P_gca(β_fk, e_cross, e_old) / P_gca(β_fk, e_old, e_cross) = exp(β_fk * (e_cross - e_old))`,
which is exactly `exp(fk_log_survival_diff)` in `step_gca`. When `β_fk = 1`, this cancels
the pairwise surrogate cut energy change `-(e_cross - e_old)` identically. -/
theorem gcaCutSurvival_ratio (β_fk e_old e_cross : ℝ) :
    gcaCutSurvival β_fk e_cross e_old / gcaCutSurvival β_fk e_old e_cross =
      Real.exp (β_fk * (e_cross - e_old)) := by
  unfold gcaCutSurvival repulsivePart
  rw [← Real.exp_sub]
  congr 1
  rcases le_total e_old e_cross with h | h
  · rw [max_eq_left (sub_nonpos.mpr h), max_eq_right (sub_nonneg.mpr h)]; ring
  · rw [max_eq_right (sub_nonneg.mpr h), max_eq_left (sub_nonpos.mpr h)]; ring

/-- Exact pointwise detailed balance for `MorphMc::step_gca` with general `fk_beta` and
many-body residual `er_x`, `er_y` (3-body surface-field junction + morphometric solvation +
Poisson memory). -/
theorem gca_cross_flow_balance
    (β_fk e_old er_x e_cross er_y : ℝ) :
    let w_x := Real.exp (-((1 - β_fk) * e_old + er_x))
    let w_y := Real.exp (-((1 - β_fk) * e_cross + er_y))
    boltzmannWeight e_old er_x * gcaCutSurvival β_fk e_old e_cross *
        min 1 (w_y / w_x) =
      boltzmannWeight e_cross er_y * gcaCutSurvival β_fk e_cross e_old *
        min 1 (w_x / w_y) := by
  intro w_x w_y
  have hwx : 0 < w_x := Real.exp_pos _
  have hwy : 0 < w_y := Real.exp_pos _
  have hcommon :
      boltzmannWeight e_old er_x * gcaCutSurvival β_fk e_old e_cross =
        w_x * Real.exp (-β_fk * max e_old e_cross) := by
    unfold boltzmannWeight gcaCutSurvival repulsivePart
    dsimp [w_x]
    rw [← Real.exp_add, ← Real.exp_add]
    congr 1
    rcases le_total e_old e_cross with h | h
    · rw [max_eq_right (sub_nonneg.mpr h), max_eq_right h]; ring
    · rw [max_eq_left (sub_nonpos.mpr h), max_eq_left h]; ring
  have hcommon_rev :
      boltzmannWeight e_cross er_y * gcaCutSurvival β_fk e_cross e_old =
        w_y * Real.exp (-β_fk * max e_old e_cross) := by
    unfold boltzmannWeight gcaCutSurvival repulsivePart
    dsimp [w_y]
    rw [← Real.exp_add, ← Real.exp_add]
    congr 1
    rcases le_total e_old e_cross with h | h
    · rw [max_eq_left (sub_nonpos.mpr h), max_eq_right h]; ring
    · rw [max_eq_right (sub_nonneg.mpr h), max_eq_left h]; ring
  rw [hcommon, hcommon_rev]
  have h1 : w_x * Real.exp (-β_fk * max e_old e_cross) * min 1 (w_y / w_x) =
      Real.exp (-β_fk * max e_old e_cross) * (w_x * min 1 (w_y / w_x)) := by ring
  have h2 : w_y * Real.exp (-β_fk * max e_old e_cross) * min 1 (w_x / w_y) =
      Real.exp (-β_fk * max e_old e_cross) * (w_y * min 1 (w_x / w_y)) := by ring
  rw [h1, h2]
  congr 1
  rw [mul_min_of_nonneg _ _ (le_of_lt hwx), mul_min_of_nonneg _ _ (le_of_lt hwy)]
  rw [mul_one, mul_one, mul_div_cancel₀ _ (ne_of_gt hwx),
    mul_div_cancel₀ _ (ne_of_gt hwy), min_comm]

end ReversibleSampling.FkClusterMove
