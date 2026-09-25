import ReversibleSampling.Balance
import ReversibleSampling.Poisson

/-!
# Internal-geometry subset rates and fixed-duration uniformization

This file proves the finite-state, exact-arithmetic rate construction. States may include
auxiliaries. It does not discretize or certify the Rust continuous pose space: its connection
to the existing measurable-kernel theorems and exact event-clock execution is an explicit
implementation obligation. A fixed-subset accepted/rejected kernel is assumed reversible.
- internal rigid geometry supplies equality of the rate at the two endpoints;
- summing such rates yields a reversible transition-rate matrix;
- a fixed global bound gives a stochastic invariant uniformization;
- a Poisson mixture of its powers preserves the same physical target;
- selecting events without their clock instead preserves the rate-biased target.
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal

namespace ReversibleSampling.ClusterRates

variable {X I : Type*} [Fintype X] [Fintype I]

set_option linter.unusedSectionVars false

/-- A nonnegative finite-state transition matrix. Targets need not be normalized. -/
def Stochastic (K : X → X → ℝ≥0∞) : Prop := ∀ x, ∑ y, K x y = 1

def Reversible (π : X → ℝ≥0∞) (K : X → X → ℝ≥0∞) : Prop :=
  ∀ x y, π x * K x y = π y * K y x

def Invariant (π : X → ℝ≥0∞) (K : X → X → ℝ≥0∞) : Prop :=
  ∀ y, ∑ x, π x * K x y = π y

/-- A fixed subset can be weighted by a rate unchanged along its supported move.
The support condition also allows zero target or zero proposal probability. -/
theorem weighted_rate_reversible (π a : X → ℝ≥0∞) (K : X → X → ℝ≥0∞)
    (hK : Reversible π K)
    (ha : ∀ x y, π x * K x y ≠ 0 → a x = a y) :
    Reversible π (fun x y => a x * K x y) := by
  intro x y
  by_cases h : π x * K x y = 0
  · calc
      π x * (a x * K x y) = a x * (π x * K x y) := by ac_rfl
      _ = 0 := by rw [h, mul_zero]
      _ = a y * (π y * K y x) := by rw [← hK x y, h, mul_zero]
      _ = π y * (a y * K y x) := by ac_rfl
  · calc
      π x * (a x * K x y) = a x * (π x * K x y) := by ac_rfl
      _ = a y * (π y * K y x) := by rw [ha x y h, hK x y]
      _ = π y * (a y * K y x) := by ac_rfl

/-- Rates depend only on an invariant internal descriptor. -/
theorem descriptor_rate_equal {D : Type*} (d : X → D) (w : D → ℝ≥0∞)
    {x y : X} (h : d x = d y) : w (d x) = w (d y) := congrArg w h

/-- A finite sum over eligible labelled subsets preserves flow symmetry. -/
theorem sum_reversible (π : X → ℝ≥0∞) (R : I → X → X → ℝ≥0∞)
    (hR : ∀ i, Reversible π (R i)) :
    Reversible π (fun x y => ∑ i, R i x y) := by
  intro x y
  simp only [Finset.mul_sum]
  exact Finset.sum_congr rfl (fun i _ => hR i x y)

/-- Total attempt rate, including proposals subsequently rejected by a subset kernel. -/
noncomputable def rateMatrix (a : I → X → ℝ≥0∞) (K : I → X → X → ℝ≥0∞)
    (x y : X) : ℝ≥0∞ := ∑ i, a i x * K i x y

def totalRate (a : I → X → ℝ≥0∞) (x : X) : ℝ≥0∞ := ∑ i, a i x

theorem rateMatrix_row_mass (a : I → X → ℝ≥0∞) (K : I → X → X → ℝ≥0∞)
    (hK : ∀ i, Stochastic (K i)) (x : X) :
    ∑ y, rateMatrix a K x y = totalRate a x := by
  simp only [rateMatrix, totalRate]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  rw [← Finset.mul_sum, hK i x, mul_one]

theorem rateMatrix_reversible (π : X → ℝ≥0∞)
    (a : I → X → ℝ≥0∞) (K : I → X → X → ℝ≥0∞)
    (hK : ∀ i, Reversible π (K i))
    (ha : ∀ i x y, π x * K i x y ≠ 0 → a i x = a i y) :
    Reversible π (rateMatrix a K) :=
  sum_reversible π _ (fun i => weighted_rate_reversible π (a i) (K i) (hK i) (ha i))

/-- Missing row mass is put on the diagonal, exactly as for measurable completed kernels. -/
noncomputable def completeMatrix [DecidableEq X] (A : X → X → ℝ≥0∞)
    (x y : X) : ℝ≥0∞ := A x y + if x = y then 1 - ∑ z, A x z else 0

theorem completeMatrix_stochastic [DecidableEq X] (A : X → X → ℝ≥0∞)
    (hA : ∀ x, ∑ y, A x y ≤ 1) : Stochastic (completeMatrix A) := by
  intro x
  simp only [completeMatrix, Finset.sum_add_distrib]
  simp only [Finset.sum_ite_eq, Finset.mem_univ, if_true]
  exact add_tsub_cancel_of_le (hA x)

theorem completeMatrix_reversible [DecidableEq X] (π : X → ℝ≥0∞)
    (A : X → X → ℝ≥0∞) (hA : Reversible π A) :
    Reversible π (completeMatrix A) := by
  intro x y
  by_cases h : x = y
  · subst y; rfl
  · simpa [completeMatrix, h, Ne.symm h] using hA x y

theorem reversible_invariant (π : X → ℝ≥0∞) (K : X → X → ℝ≥0∞)
    (hK : Stochastic K) (hR : Reversible π K) : Invariant π K := by
  intro y
  calc
    ∑ x, π x * K x y = ∑ x, π y * K y x :=
      Finset.sum_congr rfl (fun x _ => hR x y)
    _ = π y := by rw [← Finset.mul_sum, hK y, mul_one]

/-- Uniformization uses a *state-independent* positive finite bound `M`.
Its rejection probability includes both physical rejection and thinning of the attempt clock. -/
noncomputable def uniformized [DecidableEq X] (R : X → X → ℝ≥0∞)
    (M : ℝ≥0∞) : X → X → ℝ≥0∞ := completeMatrix (fun x y => R x y / M)

theorem uniformized_correct [DecidableEq X] (π : X → ℝ≥0∞)
    (R : X → X → ℝ≥0∞) (M : ℝ≥0∞) (hM0 : M ≠ 0) (hMfin : M ≠ ⊤)
    (hbound : ∀ x, ∑ y, R x y ≤ M) (hR : Reversible π R) :
    Stochastic (uniformized R M) ∧ Reversible π (uniformized R M) ∧
      Invariant π (uniformized R M) := by
  have hsub : ∀ x, ∑ y, R x y / M ≤ 1 := by
    intro x
    simp only [ENNReal.div_eq_inv_mul]
    rw [← Finset.mul_sum, ← ENNReal.div_eq_inv_mul]
    exact (ENNReal.div_le_iff hM0 hMfin).2 (by simpa using hbound x)
  have hrev : Reversible π (fun x y => R x y / M) := by
    intro x y
    simp only [← mul_div_assoc]
    rw [hR x y]
  have hs := completeMatrix_stochastic _ hsub
  have hr := completeMatrix_reversible π _ hrev
  exact ⟨hs, hr, reversible_invariant π _ hs hr⟩

/-- Composition order: first `A`, then `B`. -/
noncomputable def compose (A B : X → X → ℝ≥0∞) (x y : X) : ℝ≥0∞ := ∑ z, A x z * B z y

noncomputable def identity [DecidableEq X] (x y : X) : ℝ≥0∞ := if x = y then 1 else 0

theorem identity_stochastic [DecidableEq X] : Stochastic (identity : X → X → ℝ≥0∞) := by
  intro x; simp [identity]

theorem identity_invariant [DecidableEq X] (π : X → ℝ≥0∞) : Invariant π identity := by
  intro y; simp [identity]

theorem compose_stochastic (A B : X → X → ℝ≥0∞)
    (hA : Stochastic A) (hB : Stochastic B) : Stochastic (compose A B) := by
  intro x
  simp only [Stochastic] at hB
  simp only [compose]
  rw [Finset.sum_comm]
  simp_rw [← Finset.mul_sum, hB, mul_one]
  exact hA x

theorem compose_invariant (π : X → ℝ≥0∞) (A B : X → X → ℝ≥0∞)
    (hA : Invariant π A) (hB : Invariant π B) : Invariant π (compose A B) := by
  intro y
  simp only [Invariant] at hA
  simp only [compose, Finset.mul_sum]
  rw [Finset.sum_comm]
  simp_rw [← mul_assoc, ← Finset.sum_mul, hA]
  exact hB y

noncomputable def iterate [DecidableEq X] (K : X → X → ℝ≥0∞) : ℕ → X → X → ℝ≥0∞
  | 0 => identity
  | n + 1 => compose (iterate K n) K

theorem iterate_stochastic [DecidableEq X] (K : X → X → ℝ≥0∞)
    (hK : Stochastic K) (n : ℕ) : Stochastic (iterate K n) := by
  induction n with
  | zero => exact identity_stochastic
  | succ n hn => exact compose_stochastic _ K hn hK

theorem iterate_invariant [DecidableEq X] (π : X → ℝ≥0∞) (K : X → X → ℝ≥0∞)
    (hK : Invariant π K) (n : ℕ) : Invariant π (iterate K n) := by
  induction n with
  | zero => exact identity_invariant π
  | succ n hn => exact compose_invariant π _ K hn hK

/-- An external normalized count mixture. Weights must not depend on the starting state. -/
noncomputable def countMixture (w : ℕ → ℝ≥0∞) (K : ℕ → X → X → ℝ≥0∞)
    (x y : X) : ℝ≥0∞ := ∑' n, w n * K n x y

lemma sum_tsum_swap (f : X → ℕ → ℝ≥0∞) :
    ∑ x, ∑' n, f x n = ∑' n, ∑ x, f x n := by
  simpa only [tsum_fintype] using (ENNReal.tsum_comm (f := f))

theorem countMixture_stochastic (w : ℕ → ℝ≥0∞) (hw : ∑' n, w n = 1)
    (K : ℕ → X → X → ℝ≥0∞) (hK : ∀ n, Stochastic (K n)) :
    Stochastic (countMixture w K) := by
  intro x
  simp only [Stochastic] at hK
  simp only [countMixture]
  rw [sum_tsum_swap]
  simp_rw [← Finset.mul_sum, hK, mul_one]
  exact hw

theorem countMixture_invariant (π : X → ℝ≥0∞) (w : ℕ → ℝ≥0∞)
    (hw : ∑' n, w n = 1) (K : ℕ → X → X → ℝ≥0∞)
    (hK : ∀ n, Invariant π (K n)) : Invariant π (countMixture w K) := by
  intro y
  simp only [countMixture, ← ENNReal.tsum_mul_left]
  rw [sum_tsum_swap]
  have heq : ∀ n, (∑ x, π x * (w n * K n x y)) = w n * π y := by
    intro n
    calc
      ∑ x, π x * (w n * K n x y) = ∑ x, w n * (π x * K n x y) := by
        apply Finset.sum_congr rfl; intro x _; ac_rfl
      _ = w n * π y := by rw [← Finset.mul_sum, hK n y]
  simp_rw [heq]
  rw [ENNReal.tsum_mul_right, hw, one_mul]

/-- Fixed algorithmic duration `t`: the uniformized clock count is Pois(M*t).
The input `rateTime` is the exact nonnegative product; zero duration is allowed. -/
theorem poisson_phase_correct [DecidableEq X] (π : X → ℝ≥0∞)
    (K : X → X → ℝ≥0∞) (hS : Stochastic K) (hI : Invariant π K)
    (rateTime : ℝ≥0) :
    Stochastic (countMixture (poissonPMF rateTime) (iterate K)) ∧
      Invariant π (countMixture (poissonPMF rateTime) (iterate K)) := by
  exact ⟨countMixture_stochastic _ (PMF.tsum_coe _) _ (iterate_stochastic K hS),
    countMixture_invariant π _ (PMF.tsum_coe _) _ (iterate_invariant π K hI)⟩

/-- Omitting holding times generally changes the invariant target to π(x)*Λ(x).
This normalized event kernel includes rejected attempts. Positive finite total rate is
assumed at each state; zero-rate states are handled by the fixed-time construction. -/
theorem event_chain_rate_bias (π Λ : X → ℝ≥0∞) (R : X → X → ℝ≥0∞)
    (hΛ0 : ∀ x, Λ x ≠ 0) (hΛfin : ∀ x, Λ x ≠ ⊤)
    (hrow : ∀ x, ∑ y, R x y = Λ x) (hR : Reversible π R) :
    Stochastic (fun x y => R x y / Λ x) ∧
      Reversible (fun x => π x * Λ x) (fun x y => R x y / Λ x) ∧
      Invariant (fun x => π x * Λ x) (fun x y => R x y / Λ x) := by
  have hs : Stochastic (fun x y => R x y / Λ x) := by
    intro x
    simp only [ENNReal.div_eq_inv_mul]
    rw [← Finset.mul_sum, ← ENNReal.div_eq_inv_mul, hrow x]
    exact ENNReal.div_self (hΛ0 x) (hΛfin x)
  have hr : Reversible (fun x => π x * Λ x) (fun x y => R x y / Λ x) := by
    intro x y
    rw [mul_assoc, ENNReal.mul_div_cancel (hΛ0 x) (hΛfin x),
      mul_assoc, ENNReal.mul_div_cancel (hΛ0 y) (hΛfin y)]
    exact hR x y
  exact ⟨hs, hr, reversible_invariant _ _ hs hr⟩

/-- The complete finite-state bridge from internally invariant subset rates to a
fixed-duration Poisson phase. All physical acceptance corrections are already inside K. -/
theorem internal_subset_phase_correct [DecidableEq X] (π : X → ℝ≥0∞)
    (a : I → X → ℝ≥0∞) (K : I → X → X → ℝ≥0∞)
    (hS : ∀ i, Stochastic (K i)) (hR : ∀ i, Reversible π (K i))
    (ha : ∀ i x y, π x * K i x y ≠ 0 → a i x = a i y)
    (M : ℝ≥0∞) (hM0 : M ≠ 0) (hMfin : M ≠ ⊤)
    (hbound : ∀ x, totalRate a x ≤ M) (rateTime : ℝ≥0) :
    Stochastic (countMixture (poissonPMF rateTime)
      (iterate (uniformized (rateMatrix a K) M))) ∧
    Invariant π (countMixture (poissonPMF rateTime)
      (iterate (uniformized (rateMatrix a K) M))) := by
  have h := uniformized_correct π (rateMatrix a K) M hM0 hMfin
    (fun x => by rw [rateMatrix_row_mass a K hS x]; exact hbound x)
    (rateMatrix_reversible π a K hR ha)
  exact poisson_phase_correct π _ h.1 h.2.2 rateTime

end ReversibleSampling.ClusterRates
