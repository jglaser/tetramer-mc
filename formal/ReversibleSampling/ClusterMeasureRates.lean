import ReversibleSampling.Balance
import ReversibleSampling.Poisson

/-! General measurable-state rate weighting. Finite stationary measures and uniformly bounded
rates suffice; no coordinate density is assumed, so singular rigid maps are permitted. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal ProbabilityTheory

namespace ReversibleSampling

variable {X : Type*} [MeasurableSpace X]

/-- Detailed balance of a finite kernel is precisely symmetry of its stationary pair flow.
Only the forward implication is needed below. -/
theorem reversible_pair_flow_swap (K : Kernel X X) [IsFiniteKernel K]
    (π : Measure X) [IsFiniteMeasure π] (hK : Kernel.IsReversible K π) :
    (π ⊗ₘ K).map Prod.swap = π ⊗ₘ K := by
  apply Measure.ext_prod
  intro s t hs ht
  rw [Measure.map_apply measurable_swap (hs.prod ht), Set.preimage_swap_prod,
    Measure.compProd_apply_prod ht hs, Measure.compProd_apply_prod hs ht]
  exact hK ht hs

/-- A measurable bounded rate unchanged almost everywhere along stationary transitions
preserves accepted-flow symmetry. No pair-density or nonsingular proposal is required. -/
theorem measurable_internal_rate_reversible (K : Kernel X X) [IsFiniteKernel K]
    (π : Measure X) [IsFiniteMeasure π] (hK : Kernel.IsReversible K π)
    (a : X → ℝ≥0∞) (ha : Measurable a) (B : ℝ≥0∞) (hB : B ≠ ⊤)
    (habound : ∀ x, a x ≤ B)
    (hinternal : ∀ᵐ p : X × X ∂(π ⊗ₘ K), a p.1 = a p.2) :
    Kernel.IsReversible (K.withDensity (fun x _ => a x)) π := by
  letI : IsFiniteKernel (K.withDensity (fun x (_ : X) => a x)) :=
    Kernel.isFiniteKernel_withDensity_of_bounded K hB (fun x _ => habound x)
  have haf : Measurable (Function.uncurry (fun x (_ : X) => a x)) :=
    ha.comp measurable_fst
  have hflow := reversible_pair_flow_swap K π hK
  intro s t hs ht
  rw [← Measure.compProd_apply_prod hs ht, ← Measure.compProd_apply_prod ht hs,
    Measure.compProd_withDensity haf,
    withDensity_apply _ (hs.prod ht), withDensity_apply _ (ht.prod hs)]
  calc
    (∫⁻ p in s ×ˢ t, a p.1 ∂π ⊗ₘ K) =
        ∫⁻ p in s ×ˢ t, a p.1 ∂(π ⊗ₘ K).map Prod.swap := by rw [hflow]
    _ = ∫⁻ p in t ×ˢ s, a p.2 ∂π ⊗ₘ K := by
      simpa only [Set.preimage_swap_prod, Function.comp_apply, Prod.fst_swap] using
        (setLIntegral_map (μ := π ⊗ₘ K) (hs.prod ht)
          (ha.comp measurable_fst) measurable_swap)
    _ = ∫⁻ p in t ×ˢ s, a p.1 ∂π ⊗ₘ K :=
      lintegral_congr_ae (ae_restrict_of_ae (hinternal.mono (fun _ h => h.symm)))

/-- Bounded internal weighting of an already balanced accepted kernel, followed by rejection,
preserves π whenever its row mass is at most one. This is a continuous-state bridge into the
existing completion theorem, including deterministic or singular accepted kernels. -/
theorem measurable_internal_rate_complete_correct (K : Kernel X X) [IsFiniteKernel K]
    (π : Measure X) [IsFiniteMeasure π] (hK : Kernel.IsReversible K π)
    (a : X → ℝ≥0∞) (ha : Measurable a) (B : ℝ≥0∞) (hB : B ≠ ⊤)
    (habound : ∀ x, a x ≤ B)
    (hinternal : ∀ᵐ p : X × X ∂(π ⊗ₘ K), a p.1 = a p.2)
    (hmass : ∀ x, (K.withDensity (fun x _ => a x)) x Set.univ ≤ 1) :
    IsMarkovKernel (complete (K.withDensity (fun x _ => a x))) ∧
      Kernel.IsReversible (complete (K.withDensity (fun x _ => a x))) π ∧
      Kernel.Invariant (complete (K.withDensity (fun x _ => a x))) π := by
  have hrev := measurable_internal_rate_reversible K π hK a ha B hB habound hinternal
  exact ⟨complete_markov _ hmass, complete_reversible _ π hrev,
    complete_invariant _ π hmass hrev⟩

/-- A finite sum of reversible measurable kernels remains reversible. -/
theorem measurable_finite_sum_reversible {I : Type*} (s : Finset I)
    (K : I → Kernel X X) (π : Measure X) (hK : ∀ i ∈ s, Kernel.IsReversible (K i) π) :
    Kernel.IsReversible (∑ i ∈ s, K i) π := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [Kernel.IsReversible]
  | @insert i s hi ih =>
    rw [Finset.sum_insert hi]
    exact reversible_add (hK i (Finset.mem_insert_self _ _))
      (ih (fun j hj => hK j (Finset.mem_insert_of_mem hj)))

/-- General-state uniformization for finitely many labelled subsets. The normalized rates
`b_i=a_i/M` have sum at most one and are internally invariant. Completion supplies the
remaining clock-thinning mass on the diagonal. -/
theorem measurable_subset_uniformization_correct {I : Type*} [Fintype I]
    (K : I → Kernel X X) [∀ i, IsMarkovKernel (K i)]
    (π : Measure X) [IsFiniteMeasure π] (hK : ∀ i, Kernel.IsReversible (K i) π)
    (b : I → X → ℝ≥0∞) (hb : ∀ i, Measurable (b i))
    (hbi : ∀ i x, b i x ≤ 1) (hsum : ∀ x, ∑ i, b i x ≤ 1)
    (hinternal : ∀ i, ∀ᵐ p : X × X ∂(π ⊗ₘ K i), b i p.1 = b i p.2) :
    IsMarkovKernel (complete (∑ i, (K i).withDensity (fun x _ => b i x))) ∧
      Kernel.IsReversible (complete (∑ i, (K i).withDensity (fun x _ => b i x))) π ∧
      Kernel.Invariant (complete (∑ i, (K i).withDensity (fun x _ => b i x))) π := by
  have hrev := measurable_finite_sum_reversible Finset.univ
    (fun i => (K i).withDensity (fun x _ => b i x)) π (fun i _ =>
      measurable_internal_rate_reversible (K i) π (hK i) (b i) (hb i) 1
        ENNReal.one_ne_top (hbi i) (hinternal i))
  have hmass : ∀ x, (∑ i, (K i).withDensity (fun x _ => b i x)) x Set.univ ≤ 1 := by
    intro x
    rw [Kernel.finset_sum_apply, Measure.finset_sum_apply]
    have heq : ∀ i, ((K i).withDensity (fun x _ => b i x)) x Set.univ = b i x := by
      intro i
      have hf : Measurable (Function.uncurry (fun x (_ : X) => b i x)) :=
        (hb i).comp measurable_fst
      rw [Kernel.withDensity_apply' _ hf]
      simp
    simp_rw [heq]
    exact hsum x
  exact ⟨complete_markov _ hmass, complete_reversible _ π hrev,
    complete_invariant _ π hmass hrev⟩

/-- A countable mixture of kernels with a state-independent PMF. -/
noncomputable def measurableCountMixture (w : PMF ℕ) (K : ℕ → Kernel X X)
    [∀ n, IsMarkovKernel (K n)] : Kernel X X :=
  Kernel.sum (fun n => (K n).withDensity (fun _ _ => w n))

lemma measurableCountMixture_apply (w : PMF ℕ) (K : ℕ → Kernel X X)
    [∀ n, IsMarkovKernel (K n)] (x : X) {s : Set X} (hs : MeasurableSet s) :
    measurableCountMixture w K x s = ∑' n, w n * K n x s := by
  rw [measurableCountMixture, Kernel.sum_apply' _ _ hs]
  apply tsum_congr
  intro n
  rw [Kernel.withDensity_apply' _ measurable_const]
  simp

/-- Count mixtures preserve invariance on a general measurable state space. -/
theorem measurable_count_mixture_correct (w : PMF ℕ) (K : ℕ → Kernel X X)
    [∀ n, IsMarkovKernel (K n)] (π : Measure X)
    (hK : ∀ n, Kernel.Invariant (K n) π) :
    IsMarkovKernel (measurableCountMixture w K) ∧
      Kernel.Invariant (measurableCountMixture w K) π := by
  constructor
  · constructor
    intro x
    constructor
    rw [measurableCountMixture_apply w K x MeasurableSet.univ]
    simp
  · ext s hs
    rw [Measure.bind_apply hs (Kernel.aemeasurable _)]
    simp_rw [measurableCountMixture_apply w K _ hs]
    rw [lintegral_tsum (fun n => (measurable_const.mul ((K n).measurable_coe hs)).aemeasurable)]
    have heq : ∀ n, (∫⁻ x, w n * K n x s ∂π) = w n * π s := by
      intro n
      rw [lintegral_const_mul' _ _ (PMF.apply_ne_top _ _),
        ← Measure.bind_apply hs (Kernel.aemeasurable _), (hK n).def]
    simp_rw [heq]
    rw [ENNReal.tsum_mul_right, PMF.tsum_coe, one_mul]

/-- General measurable-state kernel iterates. -/
noncomputable def measurableKernelIterate (K : Kernel X X) : ℕ → Kernel X X
  | 0 => Kernel.id
  | n + 1 => K ∘ₖ measurableKernelIterate K n

instance measurableKernelIterate_markov (K : Kernel X X) [IsMarkovKernel K] (n : ℕ) :
    IsMarkovKernel (measurableKernelIterate K n) := by
  induction n with
  | zero => exact inferInstanceAs (IsMarkovKernel Kernel.id)
  | succ n hn =>
    letI := hn
    exact inferInstanceAs (IsMarkovKernel (K ∘ₖ measurableKernelIterate K n))

theorem measurableKernelIterate_invariant (K : Kernel X X) (π : Measure X)
    (hK : Kernel.Invariant K π) (n : ℕ) :
    Kernel.Invariant (measurableKernelIterate K n) π := by
  induction n with
  | zero => exact Measure.id_comp
  | succ n hn => exact hK.comp hn

/-- Once uniformization supplies an invariant Markov kernel, the fixed-duration Poisson
mixture of its powers preserves π even for continuous or singular-kernel state spaces.
This proves the mixture construction, not equivalence of an exponential-race implementation. -/
theorem measurable_poisson_phase_correct (K : Kernel X X) [IsMarkovKernel K]
    (π : Measure X) (hK : Kernel.Invariant K π) (rateTime : NNReal) :
    IsMarkovKernel (measurableCountMixture (poissonPMF rateTime) (measurableKernelIterate K)) ∧
      Kernel.Invariant
        (measurableCountMixture (poissonPMF rateTime) (measurableKernelIterate K)) π :=
  measurable_count_mixture_correct _ _ π (measurableKernelIterate_invariant K π hK)

end ReversibleSampling
