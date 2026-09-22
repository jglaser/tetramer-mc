import ReversibleSampling.ImportanceSampling
import ReversibleSampling.MetropolisHastings
import Mathlib.Probability.Distributions.Poisson
import Mathlib.MeasureTheory.Integral.Lebesgue.Countable

/-! Poisson depletion weights and the gained/lost-count acceptance bridge. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal Nat

namespace ReversibleSampling

/-- The probability-generating function, proved from the exponential series.
The rate may vanish. This real series also supplies finite moments. -/
theorem poisson_pgf_hasSum (r : ℝ≥0) (t : ℝ) :
    HasSum (fun n : ℕ => poissonPMFReal r n * t ^ n)
      (Real.exp ((r : ℝ) * (t - 1))) := by
  have h := (NormedSpace.expSeries_div_hasSum_exp ℝ ((r : ℝ) * t)).mul_left
    (Real.exp (-(r : ℝ)))
  convert h using 1
  · ext n
    simp only [poissonPMFReal, mul_pow]
    ring
  · rw [← Real.exp_eq_exp_ℝ, ← Real.exp_add]
    congr 1
    ring

/-- Exact nonnegative expectation under the actual Poisson probability measure. -/
theorem poisson_pgf_lintegral (r : ℝ≥0) (t : ℝ) (ht : 0 ≤ t) :
    (∫⁻ n : ℕ, ENNReal.ofReal (t ^ n) ∂poissonMeasure r) =
      ENNReal.ofReal (Real.exp ((r : ℝ) * (t - 1))) := by
  rw [lintegral_countable']
  have h := poisson_pgf_hasSum r t
  rw [← h.tsum_eq, ENNReal.ofReal_tsum_of_nonneg
    (fun n => mul_nonneg poissonPMFReal_nonneg (pow_nonneg ht n)) h.summable]
  apply tsum_congr
  intro n
  rw [poissonMeasure, PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton n)]
  change ENNReal.ofReal (t ^ n) * ENNReal.ofReal (poissonPMFReal r n) = _
  rw [← ENNReal.ofReal_mul (pow_nonneg ht n)]
  congr 1
  exact mul_comm _ _

/-- An unbiased nonnegative ideal-depletion estimator, including zero overlap/activity. -/
theorem poisson_depletion_mean (lam z C : ℝ≥0) (hlam : 0 < lam) :
    (∫⁻ n : ℕ, ENNReal.ofReal ((1 + (z : ℝ) / lam) ^ n)
      ∂poissonMeasure (lam * C)) = ENNReal.ofReal (Real.exp ((z : ℝ) * C)) := by
  rw [poisson_pgf_lintegral _ _ (by positivity)]
  congr 2
  have hlam' : (lam : ℝ) ≠ 0 := ne_of_gt (show (0 : ℝ) < lam from hlam)
  simp only [NNReal.coe_mul]
  field_simp
  ring

/-- Second moment of the same estimator; finite for every finite rate/volume. -/
theorem poisson_depletion_second_moment (lam z C : ℝ≥0) (hlam : 0 < lam) :
    (∫⁻ n : ℕ, ENNReal.ofReal (((1 + (z : ℝ) / lam) ^ n) ^ 2)
      ∂poissonMeasure (lam * C)) =
      ENNReal.ofReal (Real.exp (2 * (z : ℝ) * C + (z : ℝ) ^ 2 * C / lam)) := by
  simp_rw [← pow_mul, Nat.mul_comm _ 2, pow_mul]
  rw [poisson_pgf_lintegral _ _ (by positivity)]
  congr 2
  have hlam' : (lam : ℝ) ≠ 0 := ne_of_gt (show (0 : ℝ) < lam from hlam)
  simp only [NNReal.coe_mul]
  field_simp
  ring

/-- Relative variance expressed as the second-moment/mean-squared ratio minus one.
The expectation and second moment on its left are independently proved above. -/
theorem poisson_depletion_relative_variance (lam z C : ℝ≥0) (hlam : 0 < lam) :
    (∫⁻ n : ℕ, ENNReal.ofReal (((1 + (z : ℝ) / lam) ^ n) ^ 2)
      ∂poissonMeasure (lam * C)).toReal /
    (∫⁻ n : ℕ, ENNReal.ofReal ((1 + (z : ℝ) / lam) ^ n)
      ∂poissonMeasure (lam * C)).toReal ^ 2 - 1 =
      Real.exp ((z : ℝ) ^ 2 * C / lam) - 1 := by
  rw [poisson_depletion_second_moment lam z C hlam, poisson_depletion_mean lam z C hlam]
  simp only [ENNReal.toReal_ofReal (Real.exp_nonneg _)]
  rw [← Real.exp_nat_mul, ← Real.exp_sub]
  congr 2
  ring

variable {X : Type*} [MeasurableSpace X]

/-- Conditional expectation when the actual auxiliary kernel is the overlap-count law. -/
lemma poisson_auxiliary_conditional_mean (lam z : ℝ≥0) (hlam : 0 < lam)
    (C : X → ℝ≥0) (base : X → ℝ≥0∞) (hfinite : ∀ x, base x ≠ ⊤)
    (η : Kernel X ℕ) (hη : ∀ x, η x = poissonMeasure (lam * C x)) (x : X) :
    (∫⁻ n, base x * ENNReal.ofReal ((1 + (z : ℝ) / lam) ^ n) ∂η x) =
      base x * ENNReal.ofReal (Real.exp ((z : ℝ) * C x)) := by
  rw [hη x, lintegral_const_mul' _ _ (hfinite x), poisson_depletion_mean lam z (C x) hlam]

/-- Concrete Poisson nonnegative weights recover the physical depletion marginal.
The hypothesis on η is a precise obligation for the geometric count sampler. -/
theorem poisson_auxiliary_weight_marginal (μ : Measure X) [SFinite μ]
    (lam z : ℝ≥0) (hlam : 0 < lam) (C : X → ℝ≥0)
    (base : X → ℝ≥0∞) (hb : Measurable base) (hfinite : ∀ x, base x ≠ ⊤)
    (η : Kernel X ℕ) [IsMarkovKernel η]
    (hη : ∀ x, η x = poissonMeasure (lam * C x)) :
    ((μ ⊗ₘ η).withDensity (fun a : X × ℕ =>
      base a.1 * ENNReal.ofReal ((1 + (z : ℝ) / lam) ^ a.2))).fst =
    μ.withDensity (fun x => base x * ENNReal.ofReal (Real.exp ((z : ℝ) * C x))) := by
  apply unbiased_auxiliary_weight_marginal
  · exact (hb.comp measurable_fst).mul
      ((measurable_of_countable (fun n : ℕ =>
        ENNReal.ofReal ((1 + (z : ℝ) / lam) ^ n))).comp measurable_snd)
  · exact Filter.Eventually.of_forall
      (poisson_auxiliary_conditional_mean lam z hlam C base hfinite η hη)

/-- Exact physical normalizer expectation for a normalized pose proposal and fresh Poisson
counts. This is not a finite-variance or finite-sample convergence assertion. -/
theorem poisson_importance_sampling (μ : Measure X)
    (lam z : ℝ≥0) (hlam : 0 < lam) (C : X → ℝ≥0) (hC : Measurable C)
    (base g : X → ℝ≥0∞) (hb : Measurable base) (hg : Measurable g)
    (hfinite : ∀ x, base x ≠ ⊤) (hnormalized : (∫⁻ x, g x ∂μ) = 1)
    (hsupport : ∀ᵐ x ∂μ, base x ≠ 0 → g x ≠ 0 ∧ g x ≠ ⊤)
    (η : Kernel X ℕ) [IsMarkovKernel η]
    (hη : ∀ x, η x = poissonMeasure (lam * C x)) :
    (∫⁻ a : X × ℕ,
      (base a.1 * ENNReal.ofReal ((1 + (z : ℝ) / lam) ^ a.2)) / g a.1
      ∂((μ.withDensity g) ⊗ₘ η)) =
      ∫⁻ x, base x * ENNReal.ofReal (Real.exp ((z : ℝ) * C x)) ∂μ := by
  apply randomized_importance_sampling μ η
  · exact hb.mul ((measurable_const.mul hC.coe_nnreal_real).exp.ennreal_ofReal)
  · exact hg
  · exact (hb.comp measurable_fst).mul
      ((measurable_of_countable (fun n : ℕ =>
        ENNReal.ofReal ((1 + (z : ℝ) / lam) ^ n))).comp measurable_snd)
  · exact hnormalized
  · filter_upwards [hsupport] with x hx hnonzero
    exact hx (fun hbzero => hnonzero (by simp [hbzero]))
  · exact Filter.Eventually.of_forall
      (poisson_auxiliary_conditional_mean lam z hlam C base hfinite η hη)

end ReversibleSampling
