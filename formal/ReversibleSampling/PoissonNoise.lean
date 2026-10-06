import ReversibleSampling.Poisson

/-! An exact fixed-pose auxiliary-noise diagnostic from a saved Poisson count.

For `S ~ Poisson (r * V)`, the estimator is `a * (1 + z/r)^S`.
The same saved count gives the nonnegative unbiased variance estimator
`a^2 * ((1 + z/r)^(2*S) - (1 + 2*z/r)^S)`.
In the pooled two-cloud application, `r = 2*lambda`, `V = overlap - L`, and
`a = exp (z*L) / q`.  The hypotheses require the exact conditional count law and
a fixed (count-independent) prefactor. No ratio of random variance estimates is
asserted to be unbiased, and these results do not measure missing pose coverage.
-/

open ProbabilityTheory
open scoped NNReal Nat

namespace ReversibleSampling

/-- Physical importance weight at a fixed pose, including an arbitrary fixed prefactor. -/
noncomputable def savedPoissonWeight (r z : ℝ≥0) (a : ℝ) (n : ℕ) : ℝ :=
  a * (1 + (z : ℝ) / r) ^ n

/-- A same-count unbiased estimator of the square of the conditional mean. -/
noncomputable def savedPoissonMeanSquare (r z : ℝ≥0) (a : ℝ) (n : ℕ) : ℝ :=
  a ^ 2 * (1 + 2 * (z : ℝ) / r) ^ n

theorem savedPoisson_mean_hasSum (r z V : ℝ≥0) (hr : 0 < r) (a : ℝ) :
    HasSum (fun n => poissonPMFReal (r * V) n * savedPoissonWeight r z a n)
      (a * Real.exp ((z : ℝ) * V)) := by
  have h := (poisson_pgf_hasSum (r * V) (1 + (z : ℝ) / r)).mul_left a
  have hr' : (r : ℝ) ≠ 0 := ne_of_gt (show (0 : ℝ) < r from hr)
  convert h using 1
  · ext n
    simp only [savedPoissonWeight]
    ring
  · congr 2
    simp only [NNReal.coe_mul]
    field_simp
    ring

theorem savedPoisson_meanSquare_hasSum (r z V : ℝ≥0) (hr : 0 < r) (a : ℝ) :
    HasSum (fun n => poissonPMFReal (r * V) n * savedPoissonMeanSquare r z a n)
      (a ^ 2 * Real.exp (2 * (z : ℝ) * V)) := by
  have h := (poisson_pgf_hasSum (r * V) (1 + 2 * (z : ℝ) / r)).mul_left (a ^ 2)
  have hr' : (r : ℝ) ≠ 0 := ne_of_gt (show (0 : ℝ) < r from hr)
  convert h using 1
  · ext n
    simp only [savedPoissonMeanSquare]
    ring
  · congr 2
    simp only [NNReal.coe_mul]
    field_simp
    ring

theorem savedPoisson_secondMoment_hasSum (r z V : ℝ≥0) (hr : 0 < r) (a : ℝ) :
    HasSum (fun n => poissonPMFReal (r * V) n * (savedPoissonWeight r z a n) ^ 2)
      (a ^ 2 * Real.exp (2 * (z : ℝ) * V + (z : ℝ) ^ 2 * V / r)) := by
  have h := (poisson_pgf_hasSum (r * V) ((1 + (z : ℝ) / r) ^ 2)).mul_left (a ^ 2)
  have hr' : (r : ℝ) ≠ 0 := ne_of_gt (show (0 : ℝ) < r from hr)
  convert h using 1
  · ext n
    simp only [savedPoissonWeight, mul_pow, ← pow_mul]
    rw [Nat.mul_comm n 2]
    ring
  · congr 2
    simp only [NNReal.coe_mul]
    field_simp
    ring

/-- Nonnegativity is pointwise, including zero count, zero activity and zero prefactor. -/
theorem savedPoisson_noise_nonneg (r z : ℝ≥0) (a : ℝ) (n : ℕ) :
    0 ≤ (savedPoissonWeight r z a n) ^ 2 - savedPoissonMeanSquare r z a n := by
  have hc : 0 ≤ (z : ℝ) / r := by positivity
  have hb : 1 + 2 * ((z : ℝ) / r) ≤ (1 + (z : ℝ) / r) ^ 2 := by
    nlinarith [sq_nonneg ((z : ℝ) / r)]
  have hp := pow_le_pow_left₀ (by positivity : 0 ≤ 1 + 2 * ((z : ℝ) / r)) hb n
  have hp' : (1 + 2 * (z : ℝ) / r) ^ n ≤ ((1 + (z : ℝ) / r) ^ n) ^ 2 := by
    calc
      _ = (1 + 2 * ((z : ℝ) / r)) ^ n := by congr 1; ring
      _ ≤ ((1 + (z : ℝ) / r) ^ 2) ^ n := hp
      _ = ((1 + (z : ℝ) / r) ^ n) ^ 2 := by
        rw [← pow_mul, ← pow_mul, Nat.mul_comm 2 n]
  simp only [savedPoissonWeight, savedPoissonMeanSquare, mul_pow]
  exact sub_nonneg.mpr (mul_le_mul_of_nonneg_left hp' (sq_nonneg a))

/-- Expectation of the saved-count difference: the exact conditional variance.
All series here are finite real expectations, not a ratio-based approximation. -/
theorem savedPoisson_noise_hasSum (r z V : ℝ≥0) (hr : 0 < r) (a : ℝ) :
    HasSum (fun n => poissonPMFReal (r * V) n *
      ((savedPoissonWeight r z a n) ^ 2 - savedPoissonMeanSquare r z a n))
      (a ^ 2 * (Real.exp (2 * (z : ℝ) * V + (z : ℝ) ^ 2 * V / r) -
        Real.exp (2 * (z : ℝ) * V))) := by
  convert (savedPoisson_secondMoment_hasSum r z V hr a).sub
    (savedPoisson_meanSquare_hasSum r z V hr a) using 1
  · ext n
    ring
  · ring

/-- The centered second moment has the same value as the saved-count diagnostic. -/
theorem savedPoisson_centeredSecondMoment_hasSum (r z V : ℝ≥0) (hr : 0 < r) (a : ℝ) :
    HasSum (fun n => poissonPMFReal (r * V) n *
      (savedPoissonWeight r z a n - a * Real.exp ((z : ℝ) * V)) ^ 2)
      (a ^ 2 * (Real.exp (2 * (z : ℝ) * V + (z : ℝ) ^ 2 * V / r) -
        Real.exp (2 * (z : ℝ) * V))) := by
  have hn : HasSum (fun n => poissonPMFReal (r * V) n) (1 : ℝ) := by
    simpa using poisson_pgf_hasSum (r * V) 1
  have h := ((savedPoisson_secondMoment_hasSum r z V hr a).sub
    ((savedPoisson_mean_hasSum r z V hr a).mul_left
      (2 * (a * Real.exp ((z : ℝ) * V))))).add
        (hn.mul_left ((a * Real.exp ((z : ℝ) * V)) ^ 2))
  convert h using 1
  · ext n
    ring
  · rw [show Real.exp (2 * (z : ℝ) * V) = Real.exp ((z : ℝ) * V) ^ 2 by
      rw [← Real.exp_nat_mul]; congr 1; ring]
    ring

/-- Exact equality of the two finite expectations, making the variance interpretation explicit. -/
theorem savedPoisson_noise_eq_variance (r z V : ℝ≥0) (hr : 0 < r) (a : ℝ) :
    (∑' n, poissonPMFReal (r * V) n *
      ((savedPoissonWeight r z a n) ^ 2 - savedPoissonMeanSquare r z a n)) =
    ∑' n, poissonPMFReal (r * V) n *
      (savedPoissonWeight r z a n - a * Real.exp ((z : ℝ) * V)) ^ 2 := by
  rw [(savedPoisson_noise_hasSum r z V hr a).tsum_eq,
    (savedPoisson_centeredSecondMoment_hasSum r z V hr a).tsum_eq]

end ReversibleSampling
