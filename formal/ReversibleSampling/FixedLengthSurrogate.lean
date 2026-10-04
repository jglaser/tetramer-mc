import ReversibleSampling.PairFlowPoisson

/-! Fixed-length powers of one reversible kernel and a surrogate endpoint correction.

`fixedSteps K m` counts every completed transition, including its rejection self-loop.
Its length is fixed independently of the realized path. The power theorem holds on an
arbitrary measurable state space and does not require a finite reference measure.
The endpoint theorem uses a finite surrogate measure to derive pair-measure symmetry.
It permits singular proposals and imposes no density on the endpoint kernel.

These are mathematical kernel statements, not a certification of Rust, floating-point
arithmetic, Gaussian/Cayley geometry, random-number generation, or convergence.
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal ProbabilityTheory

namespace ReversibleSampling

variable {X : Type*} [MeasurableSpace X]

/-- Detailed balance transfers a restricted initial measure through the kernel. -/
theorem reversible_restrict_bind (K : Kernel X X) (π : Measure X)
    (hK : Kernel.IsReversible K π) {s : Set X} (hs : MeasurableSet s) :
    (π.restrict s).bind K = π.withDensity (fun x => K x s) := by
  ext t ht
  rw [Measure.bind_apply ht K.aemeasurable, withDensity_apply _ ht]
  exact hK hs ht

/-- Weighted detailed balance without a finiteness or countability assumption. -/
theorem reversible_setLIntegral (K : Kernel X X) (π : Measure X)
    (hK : Kernel.IsReversible K π) {s : Set X} (hs : MeasurableSet s)
    (f : X → ℝ≥0∞) (hf : Measurable f) :
    (∫⁻ x in s, ∫⁻ y, f y ∂K x ∂π) = ∫⁻ y, K y s * f y ∂π := by
  rw [← Measure.lintegral_bind K.aemeasurable hf.aemeasurable,
    reversible_restrict_bind K π hK hs,
    lintegral_withDensity_eq_lintegral_mul π (K.measurable_coe hs) hf]
  rfl

/-- Reversing a two-kernel path reverses the order of the reversible transitions. -/
theorem reversible_comp_reverse (A B : Kernel X X) (π : Measure X)
    (hA : Kernel.IsReversible A π) (hB : Kernel.IsReversible B π)
    {s t : Set X} (hs : MeasurableSet s) (ht : MeasurableSet t) :
    (∫⁻ x in s, (A ∘ₖ B) x t ∂π) = ∫⁻ y in t, (B ∘ₖ A) y s ∂π := by
  simp_rw [Kernel.comp_apply' _ _ _ ht, Kernel.comp_apply' _ _ _ hs]
  rw [reversible_setLIntegral B π hB hs _ (A.measurable_coe ht),
    reversible_setLIntegral A π hA ht _ (B.measurable_coe hs)]
  apply lintegral_congr
  intro y
  exact mul_comm _ _

/-- Commutativity is required: an arbitrary ordered composition need not be reversible. -/
theorem reversible_comp_of_commute (A B : Kernel X X) (π : Measure X)
    (hA : Kernel.IsReversible A π) (hB : Kernel.IsReversible B π)
    (hcomm : A ∘ₖ B = B ∘ₖ A) : Kernel.IsReversible (A ∘ₖ B) π := by
  intro s t hs ht
  rw [reversible_comp_reverse A B π hA hB hs ht, ← hcomm]

/-- Exactly `m` transitions of the same kernel, with each rejection consuming one step. -/
noncomputable def fixedSteps (K : Kernel X X) : ℕ → Kernel X X
  | 0 => Kernel.id
  | n + 1 => K ∘ₖ fixedSteps K n

instance fixedSteps_markov (K : Kernel X X) [IsMarkovKernel K] (m : ℕ) :
    IsMarkovKernel (fixedSteps K m) := by
  induction m with
  | zero => change IsMarkovKernel Kernel.id; infer_instance
  | succ m ih =>
    letI := ih
    change IsMarkovKernel (K ∘ₖ fixedSteps K m)
    infer_instance

theorem fixedSteps_commute (K : Kernel X X) (m : ℕ) :
    K ∘ₖ fixedSteps K m = fixedSteps K m ∘ₖ K := by
  induction m with
  | zero => simp [fixedSteps]
  | succ m ih =>
    change K ∘ₖ (K ∘ₖ fixedSteps K m) = (K ∘ₖ fixedSteps K m) ∘ₖ K
    calc
      K ∘ₖ (K ∘ₖ fixedSteps K m) = K ∘ₖ (fixedSteps K m ∘ₖ K) := by rw [ih]
      _ = (K ∘ₖ fixedSteps K m) ∘ₖ K := (Kernel.comp_assoc _ _ _).symm

/-- Every fixed power of a reversible kernel is reversible, including power zero. -/
theorem fixedSteps_reversible (K : Kernel X X) (π : Measure X)
    (hK : Kernel.IsReversible K π) (m : ℕ) :
    Kernel.IsReversible (fixedSteps K m) π := by
  induction m with
  | zero =>
    intro s t hs ht
    simp only [fixedSteps, Kernel.id_apply, Measure.dirac_apply' _ ht,
      Measure.dirac_apply' _ hs]
    rw [setLIntegral_indicator ht, setLIntegral_indicator hs, Set.inter_comm]
  | succ m ih =>
    exact reversible_comp_of_commute K (fixedSteps K m) π hK ih (fixedSteps_commute K m)

theorem fixedSteps_invariant (K : Kernel X X) [IsMarkovKernel K] (π : Measure X)
    (hK : Kernel.IsReversible K π) (m : ℕ) :
    Kernel.Invariant (fixedSteps K m) π := (fixedSteps_reversible K π hK m).invariant

/-- Finite reference mass lets rectangle balance identify the full reversed pair measure.
The state space itself may be uncountable. -/
theorem reversible_pair_swap (K : Kernel X X) [IsMarkovKernel K]
    (π : Measure X) [IsFiniteMeasure π] (hK : Kernel.IsReversible K π) :
    (π ⊗ₘ K).map Prod.swap = π ⊗ₘ K := by
  apply Measure.ext_prod
  intro s t hs ht
  rw [Measure.map_apply measurable_swap (hs.prod ht), Set.preimage_swap_prod,
    Measure.compProd_apply_prod ht hs, Measure.compProd_apply_prod hs ht]
  exact hK ht hs

/-- A finite, strictly positive exponential weight on the chosen hard-support state measure. -/
noncomputable def surrogateWeight (S : X → ℝ) (x : X) : ℝ≥0∞ :=
  ENNReal.ofReal (Real.exp (S x))

lemma surrogateWeight_measurable (S : X → ℝ) (hS : Measurable S) :
    Measurable (surrogateWeight S) := hS.exp.ennreal_ofReal

omit [MeasurableSpace X] in
/-- The endpoint acceptance factor is exactly the exponential of `S_old - S_new`. -/
theorem surrogate_correction_ratio (S : X → ℝ) (x y : X) :
    surrogateWeight (fun a => -S a) y / surrogateWeight (fun a => -S a) x =
      ENNReal.ofReal (Real.exp (S x - S y)) := by
  unfold surrogateWeight
  rw [← ENNReal.ofReal_div_of_pos (Real.exp_pos _), ← Real.exp_sub]
  congr 2
  ring

/-- The inverse surrogate correction removes the surrogate from the target measure.
Hard support can be encoded by taking `μ` to be a restricted measure. -/
theorem surrogate_target_cancel (μ : Measure X) (S C : X → ℝ) (z : ℝ≥0)
    (hS : Measurable S) (hC : Measurable C) :
    (μ.withDensity (surrogateWeight S)).withDensity
        (depletionDensity z (surrogateWeight (fun x => -S x)) C) =
      μ.withDensity (surrogateWeight (fun x => (z : ℝ) * C x)) := by
  have hd : Measurable (depletionDensity z (surrogateWeight (fun x => -S x)) C) :=
    (surrogateWeight_measurable _ hS.neg).mul ((measurable_const.mul hC).exp.ennreal_ofReal)
  rw [← withDensity_mul μ (surrogateWeight_measurable S hS) hd]
  congr 1
  funext x
  simp only [Pi.mul_apply, surrogateWeight, depletionDensity]
  rw [← mul_assoc, ← ENNReal.ofReal_mul (Real.exp_nonneg _), ← Real.exp_add]
  simp

/-- One exact gained/lost Poisson endpoint gate after a fixed surrogate chain is
Markov, reversible, and invariant for the physical depletion target. The reference
measure `μ` carries hard support; `K` must already be the completed reversible inner
kernel for `μ.withDensity (exp S)`. The surrogate measure must be finite. No proposal
density, state-countability, or independence of the endpoint from the start is assumed. -/
theorem fixed_surrogate_poisson_correct (μ : Measure X)
    (S C : X → ℝ) (hS : Measurable S) (hC : Measurable C)
    [IsFiniteMeasure (μ.withDensity (surrogateWeight S))]
    (K : Kernel X X) [IsMarkovKernel K]
    (hK : Kernel.IsReversible K (μ.withDensity (surrogateWeight S))) (m : ℕ)
    (lam z : ℝ≥0) (hlam : 0 < lam) (v : X → X → ℝ≥0)
    (hv : Measurable (Function.uncurry v))
    (hdelta : ∀ x y, C y - C x = (v x y : ℝ) - v y x) :
    let Q := fixedSteps K m
    let α := poissonGateAcceptance lam z v (surrogateWeight (fun x => -S x)) (fun _ _ => 1)
    let π := μ.withDensity (surrogateWeight (fun x => (z : ℝ) * C x))
    IsMarkovKernel (complete (accepted Q α)) ∧
      Kernel.IsReversible (complete (accepted Q α)) π ∧
      Kernel.Invariant (complete (accepted Q α)) π := by
  have hcorrect := pair_flow_conditional_poisson_correct
    (μ.withDensity (surrogateWeight S)) (fixedSteps K m)
    (reversible_pair_swap _ _ (fixedSteps_reversible K _ hK m))
    lam z hlam v (surrogateWeight (fun x => -S x)) C hv
    (surrogateWeight_measurable _ hS.neg) hC (fun _ => ENNReal.ofReal_ne_top) hdelta
  rw [surrogate_target_cancel μ S C z hS hC] at hcorrect
  exact hcorrect

end ReversibleSampling
