import ReversibleSampling.MetropolisHastings
import ReversibleSampling.ClusterRates

/-!
Finite-cap, independent hard-conditioned proposals. The renewal identity is over
real probabilities; the flow identity is over nonnegative extended masses.
The same context, success probability, and cap must hold at both endpoints.
This file does not certify geometry, normalized pose densities or Rust execution.
-/

open scoped ENNReal

namespace ReversibleSampling.CappedIndependent

/-- Sum of the probabilities of reaching each of at most `T` independent trials. -/
def retryFactor (z : ℝ) : ℕ → ℝ
  | 0 => 0
  | t + 1 => 1 + (1 - z) * retryFactor z t

/-- Successful candidate mass plus the all-failed self-loop is one, including z=0. -/
theorem capped_mass_identity (z : ℝ) (t : ℕ) :
    z * retryFactor z t + (1 - z) ^ t = 1 := by
  induction t with
  | zero => simp [retryFactor]
  | succ t ih =>
    simp only [retryFactor, pow_succ]
    calc
      z * (1 + (1 - z) * retryFactor z t) + (1 - z) ^ t * (1 - z) =
          z + (1 - z) * (z * retryFactor z t + (1 - z) ^ t) := by ring
      _ = 1 := by rw [ih]; ring

theorem retryFactor_nonnegative (z : ℝ) (hz : z ≤ 1) (t : ℕ) :
    0 ≤ retryFactor z t := by
  induction t with
  | zero => simp [retryFactor]
  | succ t ih =>
    exact add_nonneg (by norm_num) (mul_nonneg (sub_nonneg.mpr hz) ih)

/-- The candidate subdensity has mass at most one; no division by z is needed. -/
theorem capped_mass_bounds (z : ℝ) (hz0 : 0 ≤ z) (hz1 : z ≤ 1) (t : ℕ) :
    0 ≤ z * retryFactor z t ∧ z * retryFactor z t ≤ 1 := by
  refine ⟨mul_nonneg hz0 (retryFactor_nonnegative z hz1 t), ?_⟩
  have h := capped_mass_identity z t
  have hp := pow_nonneg (sub_nonneg.mpr hz1) t
  linarith

theorem empty_success_set (t : ℕ) : retryFactor 0 t = (t : ℝ) := by
  induction t with
  | zero => simp [retryFactor]
  | succ t ih => simp [retryFactor, ih, Nat.cast_add, Nat.cast_one]; ring

/-- A shared retry factor multiplies both accepted flows, including a zero
factor. `a` and `b` include target, guide and complete ordered-pool probabilities.
The acceptance uses their ratio; the unknown retry factor is never evaluated. -/
theorem shared_factor_accepted_flow (c a b : ℝ≥0∞)
    (ha : a ≠ ⊤) (hb : b ≠ ⊤) :
    c * a * min 1 (b / a) = c * b * min 1 (a / b) := by
  rw [mul_assoc c a, mass_mul_mh a b ha,
      mul_assoc c b, mass_mul_mh b a hb, min_comm a b]

/-- This also applies directly to the already marginalized Poisson accepted
flow, so no expectation of a random log acceptance is substituted for the gate. -/
theorem shared_factor_preserves_balance (c forward reverse : ℝ≥0∞)
    (h : forward = reverse) : c * forward = c * reverse := congrArg (c * ·) h

end ReversibleSampling.CappedIndependent
