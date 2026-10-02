import ReversibleSampling.ConditionalPoisson

/-! A count-gate bridge for singular proposals. The reference pair measure is assumed
swap invariant explicitly; neither a proposal density nor a finite reference measure is
required. S-finiteness is used only for composition-product integration. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal ProbabilityTheory

namespace ReversibleSampling

variable {X : Type*} [MeasurableSpace X]

/-- Reweight only the initial marginal, retaining the actual (possibly singular) proposal. -/
lemma accepted_flow_pair_measure (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (p : X → ℝ≥0∞) (α : X → X → ℝ≥0∞)
    (hp : Measurable p) (hα : Measurable (Function.uncurry α))
    (hpfinite : ∀ x, p x ≠ ⊤)
    {s t : Set X} (hs : MeasurableSet s) (ht : MeasurableSet t) :
    (∫⁻ x in s, ∫⁻ y in t, α x y ∂Q x ∂μ.withDensity p) =
      ∫⁻ a in s ×ˢ t, p a.1 * α a.1 a.2 ∂(μ ⊗ₘ Q) := by
  have hm : Measurable (fun a : X × X => p a.1 * α a.1 a.2) :=
    (hp.comp measurable_fst).mul hα
  rw [Measure.setLIntegral_compProd hm hs ht,
    setLIntegral_withDensity_eq_setLIntegral_mul μ hp
      (hα.setLIntegral_kernel_prod_right ht) hs]
  apply lintegral_congr
  intro x
  exact (lintegral_const_mul' (p x) (fun y => α x y) (hpfinite x)).symm

/-- Pointwise weighted-acceptance symmetry integrates against a swap-invariant pair
measure. The assumption is about μ(dx)Q(x,dy), not Q having a density against μ(dy). -/
theorem pair_flow_accepted_flow_symmetric (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (hswap : (μ ⊗ₘ Q).map Prod.swap = μ ⊗ₘ Q)
    (p : X → ℝ≥0∞) (α : X → X → ℝ≥0∞)
    (hp : Measurable p) (hα : Measurable (Function.uncurry α))
    (hpfinite : ∀ x, p x ≠ ⊤)
    (hbalance : ∀ x y, p x * α x y = p y * α y x) :
    AcceptedFlowSymmetric Q α (μ.withDensity p) := by
  intro s t hs ht
  rw [accepted_flow_pair_measure μ Q p α hp hα hpfinite hs ht,
    accepted_flow_pair_measure μ Q p α hp hα hpfinite ht hs]
  have hm : Measurable (fun a : X × X => p a.1 * α a.1 a.2) :=
    (hp.comp measurable_fst).mul hα
  calc
    (∫⁻ a in s ×ˢ t, p a.1 * α a.1 a.2 ∂(μ ⊗ₘ Q)) =
        ∫⁻ a in s ×ˢ t, p a.1 * α a.1 a.2 ∂(μ ⊗ₘ Q).map Prod.swap := by rw [hswap]
    _ = ∫⁻ a in t ×ˢ s, p a.2 * α a.2 a.1 ∂(μ ⊗ₘ Q) := by
      simpa only [Set.preimage_swap_prod, Function.comp_apply, Prod.fst_swap,
        Prod.snd_swap] using (setLIntegral_map (μ := μ ⊗ₘ Q) (hs.prod ht) hm measurable_swap)
    _ = ∫⁻ a in t ×ˢ s, p a.1 * α a.1 a.2 ∂(μ ⊗ₘ Q) := by
      apply lintegral_congr
      intro a
      exact hbalance a.2 a.1

/-- A normalized auxiliary count gate for a reference-symmetric, possibly singular
proposal. The constant scalar q=1 cancels no fictitious six-dimensional density. -/
theorem pair_flow_count_gate_correct (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (hswap : (μ ⊗ₘ Q).map Prod.swap = μ ⊗ₘ Q)
    (p : X → ℝ≥0∞) (k : X → X → ℕ × ℕ → ℝ≥0∞)
    (hp : Measurable p)
    (hk : ∀ n, Measurable (fun a : X × X => k a.1 a.2 n))
    (hpfinite : ∀ x, p x ≠ ⊤) (hkfinite : ∀ x y n, k x y n ≠ ⊤)
    (hnorm : ∀ x y, ∑' n, k x y n = 1) :
    IsMarkovKernel (complete (accepted Q (countGateAcceptance p (fun _ _ => 1) k))) ∧
      Kernel.IsReversible (complete (accepted Q (countGateAcceptance p (fun _ _ => 1) k)))
        (μ.withDensity p) ∧
      Kernel.Invariant (complete (accepted Q (countGateAcceptance p (fun _ _ => 1) k)))
        (μ.withDensity p) := by
  have hα := countGateAcceptance_measurable p (fun _ _ => 1) k hp measurable_const hk
  apply accept_reject_correct Q _ _ hα
    (countGateAcceptance_le_one p (fun _ _ => 1) k hnorm)
  apply pair_flow_accepted_flow_symmetric μ Q hswap p _ hp hα hpfinite
  intro x y
  simpa only [mul_one] using
    countGate_pointwise_balance p (fun _ _ => 1) k hpfinite
      (fun _ _ => ENNReal.one_ne_top) hkfinite x y

/-- Concrete gained/lost Poisson acceptance on a swap-invariant reference pair measure.
This applies to singular local moves once their reference-flow symmetry is established.
Zero base density and zero changed volumes are allowed; λ is strictly positive. -/
theorem pair_flow_conditional_poisson_correct (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (hswap : (μ ⊗ₘ Q).map Prod.swap = μ ⊗ₘ Q)
    (lam z : ℝ≥0) (hlam : 0 < lam)
    (v : X → X → ℝ≥0) (base : X → ℝ≥0∞) (C : X → ℝ)
    (hv : Measurable (Function.uncurry v)) (hb : Measurable base) (hC : Measurable C)
    (hbfinite : ∀ x, base x ≠ ⊤)
    (hdelta : ∀ x y, C y - C x = (v x y : ℝ) - v y x) :
    IsMarkovKernel (complete (accepted Q (poissonGateAcceptance lam z v base (fun _ _ => 1)))) ∧
      Kernel.IsReversible (complete (accepted Q (poissonGateAcceptance lam z v base (fun _ _ => 1))))
        (μ.withDensity (depletionDensity z base C)) ∧
      Kernel.Invariant (complete (accepted Q (poissonGateAcceptance lam z v base (fun _ _ => 1))))
        (μ.withDensity (depletionDensity z base C)) := by
  have heq : poissonGateAcceptance lam z v base (fun _ _ => 1) =
      countGateAcceptance (depletionDensity z base C) (fun _ _ => 1)
        (fun x y => poissonCountMass lam z (v x y) (v y x)) := by
    funext x y
    exact poissonGateAcceptance_eq_countGate lam z hlam v base (fun _ _ => 1) C hdelta x y
  rw [heq]
  apply pair_flow_count_gate_correct μ Q hswap
  · exact hb.mul (((measurable_const.mul hC).exp).ennreal_ofReal)
  · exact poissonCountMass_measurable lam z v hv
  · intro x
    exact ENNReal.mul_ne_top (hbfinite x) ENNReal.ofReal_ne_top
  · intro x y n
    exact poissonCountMass_finite lam z (v x y) (v y x) n
  · intro x y
    exact poissonCountMass_normalized lam z (v x y) (v y x)

end ReversibleSampling
