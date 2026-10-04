import ReversibleSampling.ClusterMeasureRates

/-! State-dependent selection of retained labels after the selection ratio has
already been included in each label's final acceptance. The corrected component
is reversible for `π.withDensity p`, not necessarily for `π` itself.

For a fixed surrogate proposal, `twoLeg_poisson_correct` in `TwoLegPoisson.lean` supplies
this premise by using the reference `exp(S) μ` and `base = p * exp(-S)`; cancelling
the surrogate factors identifies the resulting measure as `π.withDensity p`.
That concrete density substitution is not repeated here. These lemmas establish
the selection/mixture step on a general measurable state space, allowing zero
selection probabilities and singular kernels. No equality of the selection
probability at the two endpoints is assumed.

The label catalogue is finite and selection is normalized at every state. This
file does not assert proposal symmetry, the Poisson trace law, geometry, RNG or
floating-point correctness, ergodicity, or equivalence to an event-clock executor.
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal ProbabilityTheory

namespace ReversibleSampling

variable {X : Type*} [MeasurableSpace X]

/-- Move a selection weight from the target measure to the source row of the
kernel. Reversibility for the weighted target is sufficient; reversibility for
the unweighted target and invariance of the weight along transitions are not
assumptions. Zero weights are included without dividing by them. -/
theorem corrected_selection_reversible (K : Kernel X X) [IsSFiniteKernel K]
    (π : Measure X) (p : X → ℝ≥0∞) (hp : Measurable p)
    (hK : Kernel.IsReversible K (π.withDensity p)) :
    Kernel.IsReversible (K.withDensity (fun x _ => p x)) π := by
  have hf : Measurable (Function.uncurry (fun x (_ : X) => p x)) :=
    hp.comp measurable_fst
  intro s t hs ht
  have htarget := hK hs ht
  rw [setLIntegral_withDensity_eq_setLIntegral_mul π hp (K.measurable_coe ht) hs,
    setLIntegral_withDensity_eq_setLIntegral_mul π hp (K.measurable_coe hs) ht] at htarget
  simpa only [Kernel.withDensity_apply' K hf, setLIntegral_const, Pi.mul_apply] using htarget

/-- Choose a retained label with its actual normalized state-dependent probability,
then apply that label's completed, selection-corrected kernel. Their sum is a
Markov kernel reversible and invariant for the unweighted physical target.
Component rejection mass is already included in each `K i`. -/
theorem normalized_corrected_selection_correct {I : Type*} [Fintype I]
    (K : I → Kernel X X) [∀ i, IsMarkovKernel (K i)]
    (π : Measure X) (p : I → X → ℝ≥0∞) (hp : ∀ i, Measurable (p i))
    (hsum : ∀ x, ∑ i, p i x = 1)
    (hK : ∀ i, Kernel.IsReversible (K i) (π.withDensity (p i))) :
    IsMarkovKernel (∑ i, (K i).withDensity (fun x _ => p i x)) ∧
      Kernel.IsReversible (∑ i, (K i).withDensity (fun x _ => p i x)) π ∧
      Kernel.Invariant (∑ i, (K i).withDensity (fun x _ => p i x)) π := by
  have hrev := measurable_finite_sum_reversible Finset.univ
    (fun i => (K i).withDensity (fun x _ => p i x)) π (fun i _ =>
      corrected_selection_reversible (K i) π (p i) (hp i) (hK i))
  have hmass : ∀ x, (∑ i, (K i).withDensity (fun x _ => p i x)) x Set.univ = 1 := by
    intro x
    rw [Kernel.finset_sum_apply, Measure.finset_sum_apply]
    have heq : ∀ i, ((K i).withDensity (fun x _ => p i x)) x Set.univ = p i x := by
      intro i
      have hf : Measurable (Function.uncurry (fun x (_ : X) => p i x)) :=
        (hp i).comp measurable_fst
      rw [Kernel.withDensity_apply' _ hf]
      simp
    simp_rw [heq]
    exact hsum x
  have hmarkov : IsMarkovKernel (∑ i, (K i).withDensity (fun x _ => p i x)) := by
    constructor
    intro x
    constructor
    exact hmass x
  letI := hmarkov
  exact ⟨hmarkov, hrev, hrev.invariant⟩

end ReversibleSampling
