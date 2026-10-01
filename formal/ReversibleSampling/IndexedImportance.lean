import ReversibleSampling.ImportanceSampling
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-! Pointwise algebra for retaining a proposal-component index.

At a fixed physical pose, `a i = p_i q_old_i` and `b i = p_i q_new_i`.
The target is split using the old responsibilities `a i / sum a`.
These are algebraic identities, not a proof of the geometric sampler's
normalization, implementation, finite variance or Monte Carlo convergence.
-/

namespace ReversibleSampling.IndexedImportance

open scoped BigOperators

variable {I : Type*} [Fintype I]

/-- Summing over the retained index recovers the unsplit target at each pose. -/
theorem indexed_mean (a b : I → ℝ) (f : ℝ)
    (hQ : (∑ i, a i) ≠ 0) (hb : ∀ i, b i ≠ 0) :
    (∑ i, b i * (f * a i / ((∑ j, a j) * b i))) = f := by
  calc
    _ = ∑ i, f * a i / (∑ j, a j) := by
      apply Finset.sum_congr rfl
      intro i _
      field_simp [hb i]
    _ = f := by
      rw [← Finset.sum_div, ← Finset.mul_sum]
      exact mul_div_cancel_right₀ f hQ

/-- Conditioning each component to retain the physical support can only reduce
the pointwise second moment relative to the old full-mixture estimator. -/
theorem indexed_second_moment_le (a b : I → ℝ) (f : ℝ)
    (hQ : 0 < ∑ i, a i) (ha : ∀ i, 0 ≤ a i)
    (hb : ∀ i, 0 < b i) (hab : ∀ i, a i ≤ b i) :
    (∑ i, b i * (f * a i / ((∑ j, a j) * b i)) ^ 2) ≤
      f ^ 2 / (∑ i, a i) := by
  have hterm : ∀ i, b i * (f * a i / ((∑ j, a j) * b i)) ^ 2 ≤
      f ^ 2 / (∑ j, a j) ^ 2 * a i := by
    intro i
    have hd : a i ^ 2 / b i ≤ a i := by
      apply (div_le_iff₀ (hb i)).2
      nlinarith [mul_nonneg (ha i) (sub_nonneg.mpr (hab i))]
    have heq : b i * (f * a i / ((∑ j, a j) * b i)) ^ 2 =
        (f ^ 2 / (∑ j, a j) ^ 2) * (a i ^ 2 / b i) := by
      field_simp
    rw [heq]
    exact mul_le_mul_of_nonneg_left hd (div_nonneg (sq_nonneg f) (sq_nonneg _))
  calc
    _ ≤ ∑ i, f ^ 2 / (∑ j, a j) ^ 2 * a i := Finset.sum_le_sum (fun i _ => hterm i)
    _ = f ^ 2 / (∑ i, a i) := by
      rw [← Finset.mul_sum]
      field_simp

/-- Algebraic weighted-variance identity. The gap is nonnegative when the
weights are nonnegative, as used in `full_mixture_second_moment_le` below. -/
theorem weighted_variance_identity (b w : I → ℝ) (f : ℝ)
    (hB : (∑ i, b i) ≠ 0) (hmean : (∑ i, b i * w i) = f) :
    (∑ i, b i * (w i - f / (∑ j, b j)) ^ 2) =
      (∑ i, b i * w i ^ 2) - f ^ 2 / (∑ i, b i) := by
  calc
    _ = ∑ i, (b i * w i ^ 2 - (2 * f / (∑ j, b j)) * (b i * w i) +
        (f / (∑ j, b j)) ^ 2 * b i) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = (∑ i, b i * w i ^ 2) - (2 * f / (∑ j, b j)) * f +
        (f / (∑ j, b j)) ^ 2 * (∑ i, b i) := by
      rw [Finset.sum_add_distrib, Finset.sum_sub_distrib,
        ← Finset.mul_sum, ← Finset.mul_sum, hmean]
    _ = _ := by
      field_simp
      ring

/-- The complete new mixture is at least as efficient per draw as the retained
index estimator. This is a variance statement, not a runtime comparison. -/
theorem full_mixture_second_moment_le (a b : I → ℝ) (f : ℝ)
    (hQ : (∑ i, a i) ≠ 0) (hB : 0 < ∑ i, b i) (hb : ∀ i, 0 < b i) :
    f ^ 2 / (∑ i, b i) ≤
      ∑ i, b i * (f * a i / ((∑ j, a j) * b i)) ^ 2 := by
  have hmean := indexed_mean a b f hQ (fun i => ne_of_gt (hb i))
  have hid := weighted_variance_identity b
    (fun i => f * a i / ((∑ j, a j) * b i)) f (ne_of_gt hB) hmean
  have hn : 0 ≤ ∑ i, b i *
      (f * a i / ((∑ j, a j) * b i) - f / (∑ j, b j)) ^ 2 :=
    Finset.sum_nonneg (fun i _ => mul_nonneg (le_of_lt (hb i)) (sq_nonneg _))
  linarith

end ReversibleSampling.IndexedImportance
