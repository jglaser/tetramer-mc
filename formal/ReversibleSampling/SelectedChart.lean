import ReversibleSampling.PosteriorContext

/-!
Original-atlas label selection with a different, spectator-fixed transport map.
These are finite real-density identities. They do not identify the selected
transport charts with the densities used for selecting labels.

The implemented map/noise Jacobian, its inverse, measurable change of variables,
normalization of the Gaussian chart densities, deterministic fixed context,
physical Poisson gate and floating-point arithmetic remain implementation
obligations. Kernel completion below explicitly assumes integrated balance.
-/

namespace ReversibleSampling.SelectedChart

open scoped BigOperators
open PosteriorContext

variable {S X : Type*} [Fintype S]

/-- Ordered retained labels: posterior source and an independent prior target. -/
noncomputable def selection (w : S → ℝ) (G : S → X → ℝ)
    (x : X) (i j : S) : ℝ := posterior w G x i * w j

theorem selection_nonnegative (w : S → ℝ) (G : S → X → ℝ)
    (x : X) (i j : S) (hwi : 0 ≤ w i) (hwj : 0 ≤ w j)
    (hGi : 0 ≤ G i x) (hQ : 0 < mixture w G x) :
    0 ≤ selection w G x i j :=
  mul_nonneg (posterior_nonnegative w G x i hwi hGi hQ) hwj

/-- No successful-label filtering enters either normalization. -/
theorem selection_sum_one (w : S → ℝ) (G : S → X → ℝ) (x : X)
    (hw : ∑ j, w j = 1) (hQ : 0 < mixture w G x) :
    (∑ i, ∑ j, selection w G x i j) = 1 := by
  simp only [selection, ← Finset.mul_sum, hw, mul_one]
  exact posterior_sum_one w G x hQ

/-- Swapping retained labels cancels the prior weights. Reverse component
density may be zero; forward selected density and both mixtures are nonzero. -/
theorem selected_label_ratio (wi wj gx gy Qx Qy : ℝ)
    (hwi : wi ≠ 0) (hwj : wj ≠ 0) (hgx : gx ≠ 0)
    (hQx : Qx ≠ 0) (hQy : Qy ≠ 0) :
    ((wj * gy / Qy) * wi) / ((wi * gx / Qx) * wj) =
      (gy / gx) * (Qx / Qy) := by
  field_simp
  <;> ring

/-- The implementation's map correction is noise ratio times extended
Jacobian. Label correction uses ORIGINAL atlas densities, even if map charts
have different centers or covariances. -/
theorem full_acceptance_ratio (px py wi wj gx gy Qx Qy eta_x eta_y J : ℝ)
    (hpx : px ≠ 0) (hwi : wi ≠ 0) (hwj : wj ≠ 0)
    (hgx : gx ≠ 0) (hQx : Qx ≠ 0) (hQy : Qy ≠ 0)
    (heta : eta_x ≠ 0) :
    (py * ((wj * gy / Qy) * wi) * eta_y * J) /
        (px * ((wi * gx / Qx) * wj) * eta_x) =
      (py / px) * ((eta_y / eta_x) * J) *
        ((gy / gx) * (Qx / Qy)) := by
  field_simp
  <;> ring

/-- Exact cancellation to Qx/Qy requires the transport's density ratio to
match the selected ORIGINAL chart densities. This is not assumed generally. -/
theorem matched_chart_cancellation (gx gy Qx Qy : ℝ)
    (hgx : gx ≠ 0) (hgy : gy ≠ 0) :
    (gx / gy) * ((gy / gx) * (Qx / Qy)) = Qx / Qy := by
  have hcancel : (gx / gy) * (gy / gx) = 1 := by field_simp
  rw [← mul_assoc, hcancel, one_mul]

/-- Local accepted-flow equality after expressing the inverse flow in the
forward volume element. This includes zero reverse mass. A true invertible
map additionally needs a positive Jacobian and change-of-variables theorem. -/
theorem jacobian_weighted_flow (a b J : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hJ : 0 ≤ J) :
    a * min 1 (b * J / a) = b * J * min 1 (a / (b * J)) := by
  rw [real_mass_mul_mh a (b * J) ha (mul_nonneg hb hJ),
    real_mass_mul_mh (b * J) a (mul_nonneg hb hJ) ha, min_comm]

/-- A zero inverse flow is a rejected move, not an undefined likelihood ratio. -/
theorem zero_reverse_acceptance (a : ℝ) : min 1 ((0 : ℝ) / a) = 0 := by
  simp

/-- Summing/integrating the concrete traces must establish these premises.
Accepted-flow symmetry plus rejection gives the invariant physical kernel;
the proposal and acceptance need not each preserve the physical measure. -/
theorem complete_selected_kernel {Y : Type*} [MeasurableSpace Y]
    (A : ProbabilityTheory.Kernel Y Y) (π : MeasureTheory.Measure Y)
    (hmass : ∀ x, A x Set.univ ≤ 1)
    (hbalance : ProbabilityTheory.Kernel.IsReversible A π) :
    ProbabilityTheory.IsMarkovKernel (ReversibleSampling.complete A) ∧
      ProbabilityTheory.Kernel.IsReversible (ReversibleSampling.complete A) π ∧
      ProbabilityTheory.Kernel.Invariant (ReversibleSampling.complete A) π :=
  complete_context_kernel A π hmass hbalance

end ReversibleSampling.SelectedChart
