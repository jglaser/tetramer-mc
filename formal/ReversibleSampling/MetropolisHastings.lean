import ReversibleSampling.Balance

/-! Constructive Metropolis--Hastings balance for densities on a general measurable space. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace ReversibleSampling

variable {X : Type*} [MeasurableSpace X]

/-- The ordinary MH acceptance rule, including proposal-density asymmetry. -/
noncomputable def mhAcceptance (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞) (x y : X) : ℝ≥0∞ :=
  min 1 ((p y * q y x) / (p x * q x y))

lemma mhAcceptance_measurable (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (Function.uncurry q)) :
    Measurable (Function.uncurry (mhAcceptance p q)) := by
  exact measurable_const.min (((hp.comp measurable_snd).mul (hq.comp measurable_swap)).div
    ((hp.comp measurable_fst).mul hq))

/-- Multiplying a finite forward mass by MH acceptance gives the smaller directed mass.
Zero forward mass is allowed, so hard exclusions do not require strict positivity. -/
lemma mass_mul_mh (a b : ℝ≥0∞) (ha : a ≠ ⊤) :
    a * min 1 (b / a) = min a b := by
  by_cases hzero : a = 0
  · simp [hzero]
  · rw [mul_min, mul_one, ENNReal.mul_div_cancel hzero ha]

lemma accepted_flow_density (μ : Measure X) (Q : Kernel X X) [IsMarkovKernel Q]
    (p : X → ℝ≥0∞) (q α : X → X → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (Function.uncurry q))
    (hα : Measurable (Function.uncurry α)) (hfinite : ∀ x, p x ≠ ⊤)
    (hQ : ∀ x, Q x = μ.withDensity (q x))
    {s t : Set X} (hs : MeasurableSet s) (ht : MeasurableSet t) :
    (∫⁻ x in s, ∫⁻ y in t, α x y ∂Q x ∂μ.withDensity p) =
    ∫⁻ x in s, ∫⁻ y in t, p x * q x y * α x y ∂μ ∂μ := by
  rw [setLIntegral_withDensity_eq_setLIntegral_mul μ hp
    (hα.setLIntegral_kernel_prod_right ht) hs]
  apply lintegral_congr
  intro x
  change p x * (∫⁻ y in t, α x y ∂Q x) = _
  have hqx : Measurable (q x) := hq.comp (measurable_const.prodMk measurable_id)
  have hαx : Measurable (α x) := hα.comp (measurable_const.prodMk measurable_id)
  rw [hQ x, setLIntegral_withDensity_eq_setLIntegral_mul μ
    hqx hαx ht]
  rw [← lintegral_const_mul' (p x) _ (hfinite x)]
  apply lintegral_congr
  intro y
  exact (mul_assoc (p x) (q x y) (α x y)).symm

/-- With a common s-finite reference measure and finite densities, MH constructs a symmetric
accepted flow. The proposal need not preserve the target and may have zero-density regions. -/
theorem mh_accepted_flow_symmetric (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (Function.uncurry q))
    (hpfinite : ∀ x, p x ≠ ⊤) (hqfinite : ∀ x y, q x y ≠ ⊤)
    (hQ : ∀ x, Q x = μ.withDensity (q x)) :
    AcceptedFlowSymmetric Q (mhAcceptance p q) (μ.withDensity p) := by
  intro s t hs ht
  have hα := mhAcceptance_measurable p q hp hq
  rw [accepted_flow_density μ Q p q _ hp hq hα hpfinite hQ hs ht,
    accepted_flow_density μ Q p q _ hp hq hα hpfinite hQ ht hs]
  simp_rw [mhAcceptance, mass_mul_mh _ _ (ENNReal.mul_ne_top (hpfinite _) (hqfinite _ _))]
  have hmin : Measurable (fun z : X × X => min (p z.1 * q z.1 z.2) (p z.2 * q z.2 z.1)) :=
    ((hp.comp measurable_fst).mul hq).min ((hp.comp measurable_snd).mul (hq.comp measurable_swap))
  rw [lintegral_lintegral_swap hmin.aemeasurable]
  apply lintegral_congr
  intro y
  apply lintegral_congr
  intro x
  exact min_comm _ _

/-- The concrete asymmetric-proposal MH rule gives a Markov, reversible, invariant kernel. -/
theorem metropolis_hastings_correct (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (Function.uncurry q))
    (hpfinite : ∀ x, p x ≠ ⊤) (hqfinite : ∀ x y, q x y ≠ ⊤)
    (hQ : ∀ x, Q x = μ.withDensity (q x)) :
    IsMarkovKernel (complete (accepted Q (mhAcceptance p q))) ∧
    Kernel.IsReversible (complete (accepted Q (mhAcceptance p q))) (μ.withDensity p) ∧
    Kernel.Invariant (complete (accepted Q (mhAcceptance p q))) (μ.withDensity p) := by
  exact accept_reject_correct Q _ _ (mhAcceptance_measurable p q hp hq)
    (fun _ _ => min_le_left _ _)
    (mh_accepted_flow_symmetric μ Q p q hp hq hpfinite hqfinite hQ)

end ReversibleSampling
