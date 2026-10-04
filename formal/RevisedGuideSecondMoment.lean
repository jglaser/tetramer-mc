import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
An exact-arithmetic defensive-mixture and second-moment bound.

For randomized importance sampling, use the SAME conditional auxiliary law in
both schemes and put A(u) = E[(J(u) W(u, noise))^2 | u]. The integral theorem
accepts this common nonnegative conditional second moment as a real-valued
function; it does not construct a Poisson law, prove its moments, or verify a
sampler. The support, measurability and finite-old-moment premises are explicit.

The final inequality concerns the population moment ESS fraction Z^2 / M2.
An iid strong-law argument, density normalization, common mean Z, implementation
roundoff, measured finite-record ESS, CPU efficiency and unseen-mode coverage
are NOT established here. This isolated file changes no default build target.
-/

open MeasureTheory

namespace ReversibleSampling.RevisedGuideSecondMoment

/-- Preserve half uniform mass, retain three quarters of the old conditional
mixture, and allocate the remaining learned mass to an added normalized law.
The component laws themselves must be unchanged by the weight modification. -/
theorem density_dominance (u old added : ℝ) (hu : 0 ≤ u) (ha : 0 ≤ added) :
    (3 / 4 : ℝ) * ((1 / 2 : ℝ) * u + (1 / 2 : ℝ) * old) ≤
      (1 / 2 : ℝ) * u + (3 / 8 : ℝ) * old + (1 / 8 : ℝ) * added := by
  linarith

/-- Pointwise domination of a common nonnegative second-moment numerator. -/
theorem second_moment_ratio_le (a qold qnew : ℝ)
    (ha : 0 ≤ a) (ho : 0 < qold) (hd : (3 / 4 : ℝ) * qold ≤ qnew) :
    a / qnew ≤ (4 / 3 : ℝ) * (a / qold) := by
  have hn : 0 < qnew := lt_of_lt_of_le (mul_pos (by norm_num) ho) hd
  apply (div_le_iff₀ hn).2
  calc
    a = ((4 / 3 : ℝ) * (a / qold)) * ((3 / 4 : ℝ) * qold) := by
      field_simp
      <;> ring
    _ ≤ ((4 / 3 : ℝ) * (a / qold)) * qnew :=
      mul_le_mul_of_nonneg_left hd (by positivity)

/-- A finite old second moment dominates the new one, including its
integrability. No positivity of the proposal is needed where A is zero.
For importance sampling, the quotient is interpreted as zero on that set. -/
theorem second_moment_integral_bound {X : Type*} [MeasurableSpace X]
    (μ : Measure X) (A qold qnew : X → ℝ)
    (hA : ∀ᵐ x ∂μ, 0 ≤ A x)
    (hsupport : ∀ᵐ x ∂μ, A x ≠ 0 → 0 < qold x)
    (hdom : ∀ᵐ x ∂μ, (3 / 4 : ℝ) * qold x ≤ qnew x)
    (hnew_meas : AEStronglyMeasurable (fun x => A x / qnew x) μ)
    (hold : Integrable (fun x => A x / qold x) μ) :
    Integrable (fun x => A x / qnew x) μ ∧
      (∫ x, A x / qnew x ∂μ) ≤ (4 / 3 : ℝ) * ∫ x, A x / qold x ∂μ := by
  have hpoint : ∀ᵐ x ∂μ, 0 ≤ A x / qnew x ∧
      A x / qnew x ≤ (4 / 3 : ℝ) * (A x / qold x) := by
    filter_upwards [hA, hsupport, hdom] with x ha hs hd
    by_cases hz : A x = 0
    · simp [hz]
    · have ho := hs hz
      have hn : 0 < qnew x := lt_of_lt_of_le (mul_pos (by norm_num) ho) hd
      exact ⟨div_nonneg ha hn.le, second_moment_ratio_le _ _ _ ha ho hd⟩
  have hscaled := hold.const_mul (4 / 3 : ℝ)
  have hnew : Integrable (fun x => A x / qnew x) μ := by
    apply hscaled.mono' hnew_meas
    filter_upwards [hpoint] with x hx
    simpa only [Real.norm_eq_abs, abs_of_nonneg hx.1] using hx.2
  refine ⟨hnew, ?_⟩
  calc
    (∫ x, A x / qnew x ∂μ) ≤ ∫ x, (4 / 3 : ℝ) * (A x / qold x) ∂μ :=
      integral_mono_ae hnew hscaled (hpoint.mono fun _ hx => hx.2)
    _ = (4 / 3 : ℝ) * ∫ x, A x / qold x ∂μ := integral_const_mul _ _

/-- Population moment ESS fraction retention for a common mean Z and positive
finite second moments. For a nontrivial ESS ratio also require Z nonzero. -/
theorem moment_ess_retention (Z oldMoment newMoment : ℝ)
    (ho : 0 < oldMoment) (hn : 0 < newMoment)
    (hbound : newMoment ≤ (4 / 3 : ℝ) * oldMoment) :
    (3 / 4 : ℝ) * (Z ^ 2 / oldMoment) ≤ Z ^ 2 / newMoment := by
  apply (le_div_iff₀ hn).2
  calc
    ((3 / 4 : ℝ) * (Z ^ 2 / oldMoment)) * newMoment ≤
        ((3 / 4 : ℝ) * (Z ^ 2 / oldMoment)) * ((4 / 3 : ℝ) * oldMoment) :=
      mul_le_mul_of_nonneg_left hbound (by positivity)
    _ = Z ^ 2 := by
      field_simp
      <;> ring

end ReversibleSampling.RevisedGuideSecondMoment

#print axioms ReversibleSampling.RevisedGuideSecondMoment.density_dominance
#print axioms ReversibleSampling.RevisedGuideSecondMoment.second_moment_ratio_le
#print axioms ReversibleSampling.RevisedGuideSecondMoment.second_moment_integral_bound
#print axioms ReversibleSampling.RevisedGuideSecondMoment.moment_ess_retention
