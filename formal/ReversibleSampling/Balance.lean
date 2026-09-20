import Mathlib.Probability.Kernel.CompProdEqIff
import Mathlib.Probability.Kernel.Invariance

/-! General measurable-state balance arguments for completed accept/reject kernels. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal ProbabilityTheory

namespace ReversibleSampling

variable {X : Type*} [MeasurableSpace X]

/-- Rejected mass stays at its starting point. -/
noncomputable def rejection (A : Kernel X X) : Kernel X X :=
  Kernel.id.withDensity (fun x _ => 1 - A x Set.univ)

lemma rejection_density_measurable (A : Kernel X X) :
    Measurable (Function.uncurry (fun x (_ : X) => 1 - A x Set.univ)) := by
  exact measurable_const.sub ((A.measurable_coe MeasurableSet.univ).comp measurable_fst)

lemma rejection_apply (A : Kernel X X) (x : X) {s : Set X} (hs : MeasurableSet s) :
    rejection A x s = s.indicator (fun x => 1 - A x Set.univ) x := by
  classical
  rw [rejection, Kernel.withDensity_apply' _ (rejection_density_measurable A)]
  simp only [Kernel.id_apply]
  rw [setLIntegral_dirac' measurable_const hs]
  rfl

/-- A state-dependent rejection self-loop is reversible for every measure. -/
theorem rejection_reversible (A : Kernel X X) (π : Measure X) :
    Kernel.IsReversible (rejection A) π := by
  intro s t hs ht
  simp_rw [rejection_apply A _ ht, rejection_apply A _ hs]
  rw [setLIntegral_indicator ht, setLIntegral_indicator hs, Set.inter_comm]

/-- Adding two symmetric flows preserves symmetry. Neither summand must be Markov. -/
theorem reversible_add {A B : Kernel X X} {π : Measure X}
    (hA : Kernel.IsReversible A π) (hB : Kernel.IsReversible B π) :
    Kernel.IsReversible (A + B) π := by
  intro s t hs ht
  simp only [Kernel.add_apply, Measure.add_apply]
  rw [lintegral_add_left (A.measurable_coe ht),
    lintegral_add_left (A.measurable_coe hs), hA hs ht, hB hs ht]

/-- Complete a subprobability kernel with its missing mass on the diagonal. -/
noncomputable def complete (A : Kernel X X) : Kernel X X := A + rejection A

theorem complete_markov (A : Kernel X X) (hA : ∀ x, A x Set.univ ≤ 1) :
    IsMarkovKernel (complete A) := by
  constructor
  intro x
  constructor
  simp only [complete, Kernel.add_apply, Measure.add_apply,
    rejection_apply A x MeasurableSet.univ, Set.indicator_univ]
  exact add_tsub_cancel_of_le (hA x)

/-- Symmetric accepted flow implies detailed balance after adding rejections. -/
theorem complete_reversible (A : Kernel X X) (π : Measure X)
    (hbal : Kernel.IsReversible A π) : Kernel.IsReversible (complete A) π :=
  reversible_add hbal (rejection_reversible A π)

/-- A balanced subprobability transition, completed by rejection, preserves its target. -/
theorem complete_invariant (A : Kernel X X) (π : Measure X)
    (hA : ∀ x, A x Set.univ ≤ 1) (hbal : Kernel.IsReversible A π) :
    Kernel.Invariant (complete A) π := by
  letI := complete_markov A hA
  exact (complete_reversible A π hbal).invariant

/-- The accepted part of an arbitrary proposal, with no invariance requirement on `Q`. -/
noncomputable def accepted (Q : Kernel X X) [IsMarkovKernel Q]
    (α : X → X → ℝ≥0∞) : Kernel X X := Q.withDensity α

theorem accepted_mass_le_one (Q : Kernel X X) [IsMarkovKernel Q]
    (α : X → X → ℝ≥0∞) (hα : Measurable (Function.uncurry α))
    (hbound : ∀ x y, α x y ≤ 1) (x : X) : accepted Q α x Set.univ ≤ 1 := by
  rw [accepted, Kernel.withDensity_apply' Q hα]
  calc
    (∫⁻ y in Set.univ, α x y ∂Q x) ≤ ∫⁻ _y in Set.univ, 1 ∂Q x :=
      lintegral_mono (hbound x)
    _ = 1 := by simp

/-- Detailed balance for acceptance weighting, written directly in terms of proposal integrals. -/
def AcceptedFlowSymmetric (Q : Kernel X X) (α : X → X → ℝ≥0∞) (π : Measure X) : Prop :=
  ∀ ⦃s t⦄, MeasurableSet s → MeasurableSet t →
    (∫⁻ x in s, ∫⁻ y in t, α x y ∂Q x ∂π) =
    ∫⁻ y in t, ∫⁻ x in s, α y x ∂Q y ∂π

theorem accepted_reversible (Q : Kernel X X) [IsMarkovKernel Q]
    (α : X → X → ℝ≥0∞) (π : Measure X)
    (hα : Measurable (Function.uncurry α)) (hbal : AcceptedFlowSymmetric Q α π) :
    Kernel.IsReversible (accepted Q α) π := by
  intro s t hs ht
  simpa only [accepted, Kernel.withDensity_apply' Q hα] using hbal hs ht

/-- A general measurable accept/reject proposal is reversible and invariant when its accepted
flow is symmetric. A proposal that is not itself target-invariant is explicitly allowed. -/
theorem accept_reject_correct (Q : Kernel X X) [IsMarkovKernel Q]
    (α : X → X → ℝ≥0∞) (π : Measure X)
    (hα : Measurable (Function.uncurry α)) (hbound : ∀ x y, α x y ≤ 1)
    (hbal : AcceptedFlowSymmetric Q α π) :
    IsMarkovKernel (complete (accepted Q α)) ∧
    Kernel.IsReversible (complete (accepted Q α)) π ∧
    Kernel.Invariant (complete (accepted Q α)) π := by
  have hmass := accepted_mass_le_one Q α hα hbound
  have hrev := accepted_reversible Q α π hα hbal
  exact ⟨complete_markov _ hmass, complete_reversible _ π hrev,
    complete_invariant _ π hmass hrev⟩

/-- Any physical marginal of an invariant extended-state measure is preserved after a step.
This does not assert that the projected process is Markov or that adaptation may ignore balance. -/
theorem invariant_preserves_projection {Y : Type*} [MeasurableSpace Y]
    (K : Kernel X X) (μjoint : Measure X) (p : X → Y) (π : Measure Y)
    (_hp : Measurable p)
    (hK : Kernel.Invariant K μjoint) (hmarginal : μjoint.map p = π) :
    (μjoint.bind K).map p = π := by
  rw [hK.def, hmarginal]

/-- In particular, memory, mixture size, and proposal parameters may live in an auxiliary state,
provided the joint kernel is invariant for a joint measure with the required physical marginal. -/
theorem invariant_preserves_physical_marginal {U : Type*} [MeasurableSpace U]
    (K : Kernel (X × U) (X × U)) (μjoint : Measure (X × U)) (π : Measure X)
    (hK : Kernel.Invariant K μjoint) (hmarginal : μjoint.map Prod.fst = π) :
    (μjoint.bind K).map Prod.fst = π :=
  invariant_preserves_projection K μjoint Prod.fst π measurable_fst hK hmarginal

/-- A normalized independent auxiliary reservoir provides the desired marginal. This is one
sufficient joint target; configuration-dependent auxiliary laws require their own normalization. -/
theorem product_auxiliary_marginal {U : Type*} [MeasurableSpace U]
    (π : Measure X) (ρ : Measure U) [SFinite π] [IsProbabilityMeasure ρ] :
    (π.prod ρ).map Prod.fst = π := by
  simp only [Measure.map_fst_prod, measure_univ, one_smul]

/-- A normalized, configuration-dependent auxiliary law also preserves the physical marginal.
The kernel may encode memory parameters or a variable-size mixture state. -/
theorem conditional_auxiliary_marginal {U : Type*} [MeasurableSpace U]
    (π : Measure X) [SFinite π] (η : Kernel X U) [IsMarkovKernel η] :
    (π ⊗ₘ η).fst = π := Measure.fst_compProd π η

/-- Any update invariant for this declared conditional-auxiliary joint target preserves the
physical marginal. This does not permit changing the defining kernel `η` without justification. -/
theorem conditional_auxiliary_update_preserves_marginal {U : Type*} [MeasurableSpace U]
    (π : Measure X) [SFinite π] (η : Kernel X U) [IsMarkovKernel η]
    (K : Kernel (X × U) (X × U)) (hK : Kernel.Invariant K (π ⊗ₘ η)) :
    ((π ⊗ₘ η).bind K).fst = π := by
  rw [hK.def, conditional_auxiliary_marginal π η]

/-- Alternating two kernels preserving the same joint target preserves that target. Individual
reversibility need not imply reversibility of the ordered composition. -/
theorem invariant_hybrid (A B : Kernel X X) (π : Measure X)
    (hA : Kernel.Invariant A π) (hB : Kernel.Invariant B π) :
    Kernel.Invariant (A ∘ₖ B) π := hA.comp hB

end ReversibleSampling
