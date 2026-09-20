import ReversibleSampling.Balance

/-! Exact nonnegative importance-sampling identities on general measurable spaces. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace ReversibleSampling

variable {X : Type*} [MeasurableSpace X]

/-- A proposal density cancels from the nonnegative importance weight when it is positive
and finite wherever the target is nonzero, up to a reference-measure null set.
Neither normalization nor a finite target integral is needed for this identity. -/
theorem importance_sampling_lintegral (μ : Measure X) (f g : X → ℝ≥0∞)
    (hf : Measurable f) (hg : Measurable g)
    (hsupport : ∀ᵐ x ∂μ, f x ≠ 0 → g x ≠ 0 ∧ g x ≠ ⊤) :
    (∫⁻ x, f x / g x ∂μ.withDensity g) = ∫⁻ x, f x ∂μ := by
  rw [lintegral_withDensity_eq_lintegral_mul μ hg (hf.div hg)]
  apply lintegral_congr_ae
  filter_upwards [hsupport] with x hx
  change g x * (f x / g x) = f x
  by_cases hzero : f x = 0
  · simp [hzero]
  · exact ENNReal.mul_div_cancel (hx hzero).1 (hx hzero).2

/-- With a normalized proposal density, the same identity is an expectation under a
probability law. The nonnegative expectation can be infinite; finite mean requires
a finite target integral. No variance or Monte Carlo convergence claim is made. -/
theorem normalized_importance_sampling (μ : Measure X) (f g : X → ℝ≥0∞)
    (hf : Measurable f) (hg : Measurable g)
    (hnormalized : (∫⁻ x, g x ∂μ) = 1)
    (hsupport : ∀ᵐ x ∂μ, f x ≠ 0 → g x ≠ 0 ∧ g x ≠ ⊤) :
    IsProbabilityMeasure (μ.withDensity g) ∧
    (∫⁻ x, f x / g x ∂μ.withDensity g) = ∫⁻ x, f x ∂μ := by
  constructor
  · constructor
    simpa only [withDensity_apply _ MeasurableSet.univ, Measure.restrict_univ]
      using hnormalized
  · exact importance_sampling_lintegral μ f g hf hg hsupport

/-- A measurable nonnegative auxiliary weight with the right conditional mean defines an
extended target with exactly the desired physical marginal. The kernel is a declared
normalized auxiliary law; changing that law requires re-establishing the mean condition. -/
theorem unbiased_auxiliary_weight_marginal {U : Type*} [MeasurableSpace U]
    (μ : Measure X) [SFinite μ] (η : Kernel X U) [IsMarkovKernel η]
    (f : X → ℝ≥0∞) (w : X × U → ℝ≥0∞) (hw : Measurable w)
    (hmean : ∀ᵐ x ∂μ, (∫⁻ u, w (x, u) ∂η x) = f x) :
    ((μ ⊗ₘ η).withDensity w).fst = μ.withDensity f := by
  ext s hs
  rw [Measure.fst_apply hs, withDensity_apply _ (measurable_fst hs),
    withDensity_apply _ hs, ← Set.prod_univ,
    Measure.setLIntegral_compProd hw hs MeasurableSet.univ]
  simp only [Measure.restrict_univ]
  exact lintegral_congr_ae (ae_restrict_of_ae hmean)

/-- Successive conditional averaging and importance weighting are exact. This combines
the two expectation steps without asserting a concrete estimator's conditional mean
or a concrete proposal's density and normalization. -/
theorem randomized_importance_sampling {U : Type*} [MeasurableSpace U]
    (μ : Measure X) (η : Kernel X U) [IsMarkovKernel η]
    (f g : X → ℝ≥0∞) (w : X × U → ℝ≥0∞)
    (hf : Measurable f) (hg : Measurable g) (hw : Measurable w)
    (hnormalized : (∫⁻ x, g x ∂μ) = 1)
    (hsupport : ∀ᵐ x ∂μ, f x ≠ 0 → g x ≠ 0 ∧ g x ≠ ⊤)
    (hmean : ∀ᵐ x ∂μ, (∫⁻ u, w (x, u) ∂η x) = f x) :
    (∫⁻ z, w z / g z.1 ∂((μ.withDensity g) ⊗ₘ η)) = ∫⁻ x, f x ∂μ := by
  have himportance := normalized_importance_sampling μ f g hf hg hnormalized hsupport
  letI : IsProbabilityMeasure (μ.withDensity g) := himportance.1
  have hratio : Measurable (fun z : X × U => w z / g z.1) :=
    hw.div (hg.comp measurable_fst)
  rw [Measure.lintegral_compProd hratio]
  calc
    (∫⁻ x, ∫⁻ u, w (x, u) / g x ∂η x ∂μ.withDensity g) =
        ∫⁻ x, f x / g x ∂μ.withDensity g := by
      apply lintegral_congr_ae
      filter_upwards [(withDensity_absolutelyContinuous μ g).ae_le hmean] with x hx
      have hwx : Measurable (fun u => w (x, u)) := hw.comp measurable_prodMk_left
      simp only [div_eq_mul_inv]
      rw [lintegral_mul_const _ hwx, hx]
    _ = ∫⁻ x, f x ∂μ := himportance.2

end ReversibleSampling
