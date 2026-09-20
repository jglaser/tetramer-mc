import ReversibleSampling.MetropolisHastings

/-! Deterministic involutions, including singular proposals on augmented state spaces. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace ReversibleSampling

variable {X : Type*} [MeasurableSpace X]

/-- Acceptance for a reference-measure-preserving deterministic proposal. -/
noncomputable def involutionAcceptance (p : X → ℝ≥0∞) (x y : X) : ℝ≥0∞ :=
  min 1 (p y / p x)

lemma involutionAcceptance_measurable (p : X → ℝ≥0∞) (hp : Measurable p) :
    Measurable (Function.uncurry (involutionAcceptance p)) :=
  measurable_const.min ((hp.comp measurable_snd).div (hp.comp measurable_fst))

/-- The deterministic accepted flow reduces to a one-state integral of the smaller target
density. This step only needs a measurable map; reversibility is established separately. -/
lemma deterministic_accepted_flow_density (μ : Measure X) (T : X → X)
    (hT : Measurable T) (p : X → ℝ≥0∞) (hp : Measurable p)
    (hpfinite : ∀ x, p x ≠ ⊤) {s t : Set X}
    (hs : MeasurableSet s) (ht : MeasurableSet t) :
    (∫⁻ x in s, ∫⁻ y in t, involutionAcceptance p x y
      ∂Kernel.deterministic T hT x ∂μ.withDensity p) =
    ∫⁻ x in s ∩ T ⁻¹' t, min (p x) (p (T x)) ∂μ := by
  classical
  have hα := involutionAcceptance_measurable p hp
  rw [setLIntegral_withDensity_eq_setLIntegral_mul μ hp
    (hα.setLIntegral_kernel_prod_right ht) hs]
  calc
    (∫⁻ x in s, p x * (∫⁻ y in t, involutionAcceptance p x y
      ∂Kernel.deterministic T hT x) ∂μ) =
        ∫⁻ x in s, (T ⁻¹' t).indicator (fun x => min (p x) (p (T x))) x ∂μ := by
      apply lintegral_congr
      intro x
      have hαx : Measurable (involutionAcceptance p x) :=
        hα.comp (measurable_const.prodMk measurable_id)
      rw [Kernel.setLIntegral_deterministic' hT hαx ht]
      by_cases hx : T x ∈ t
      · rw [if_pos hx, Set.indicator_of_mem (show x ∈ T ⁻¹' t from hx)]
        exact mass_mul_mh _ _ (hpfinite x)
      · simp [hx]
    _ = ∫⁻ x in s ∩ T ⁻¹' t, min (p x) (p (T x)) ∂μ := by
      rw [setLIntegral_indicator (hT ht), Set.inter_comm]

/-- An involution that preserves the reference measure gives symmetric accepted flow under
Metropolis weighting. Reference-measure preservation is an explicit geometric hypothesis;
mere invertibility is insufficient. -/
theorem involution_accepted_flow_symmetric (μ : Measure X) (T : X → X)
    (hpres : MeasurePreserving T μ μ) (hinv : Function.Involutive T)
    (p : X → ℝ≥0∞) (hp : Measurable p) (hpfinite : ∀ x, p x ≠ ⊤) :
    AcceptedFlowSymmetric (Kernel.deterministic T hpres.measurable)
      (involutionAcceptance p) (μ.withDensity p) := by
  intro s t hs ht
  rw [deterministic_accepted_flow_density μ T hpres.measurable p hp hpfinite hs ht,
    deterministic_accepted_flow_density μ T hpres.measurable p hp hpfinite ht hs]
  have hpre : T ⁻¹' (t ∩ T ⁻¹' s) = s ∩ T ⁻¹' t := by
    ext x
    simp [hinv x, and_comm]
  have hw : ∀ x, min (p (T x)) (p (T (T x))) = min (p x) (p (T x)) := by
    intro x
    rw [hinv x, min_comm]
  have hchange := hpres.setLIntegral_comp_preimage (ht.inter (hpres.measurable hs))
    (hp.min (hp.comp hpres.measurable))
  simpa only [hpre, Function.comp_apply, hw] using hchange

/-- The completed deterministic-involution Metropolis kernel is Markov, reversible and
invariant. The theorem applies to augmented spaces, but does not verify a concrete code map. -/
theorem involution_metropolis_correct (μ : Measure X) (T : X → X)
    (hpres : MeasurePreserving T μ μ) (hinv : Function.Involutive T)
    (p : X → ℝ≥0∞) (hp : Measurable p) (hpfinite : ∀ x, p x ≠ ⊤) :
    IsMarkovKernel (complete (accepted (Kernel.deterministic T hpres.measurable)
      (involutionAcceptance p))) ∧
    Kernel.IsReversible (complete (accepted (Kernel.deterministic T hpres.measurable)
      (involutionAcceptance p))) (μ.withDensity p) ∧
    Kernel.Invariant (complete (accepted (Kernel.deterministic T hpres.measurable)
      (involutionAcceptance p))) (μ.withDensity p) := by
  exact accept_reject_correct _ _ _ (involutionAcceptance_measurable p hp)
    (fun _ _ => min_le_left _ _)
    (involution_accepted_flow_symmetric μ T hpres hinv p hp hpfinite)

end ReversibleSampling
