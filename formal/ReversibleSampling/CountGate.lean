import ReversibleSampling.Poisson

/-! Count-averaged auxiliary Metropolis gates. The reverse trace swaps the two counts. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace ReversibleSampling

variable {X : Type*} [MeasurableSpace X]

/-- Draw a normalized pair of auxiliary counts and accept using its full reverse law.
The marginalized acceptance is a deterministic number in [0,1]. -/
noncomputable def countGateAcceptance (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞)
    (k : X → X → ℕ × ℕ → ℝ≥0∞) (x y : X) : ℝ≥0∞ :=
  ∑' n, k x y n * min 1
    ((p y * q y x * k y x n.swap) / (p x * q x y * k x y n))

lemma countGateAcceptance_measurable (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞)
    (k : X → X → ℕ × ℕ → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (Function.uncurry q))
    (hk : ∀ n, Measurable (fun a : X × X => k a.1 a.2 n)) :
    Measurable (Function.uncurry (countGateAcceptance p q k)) := by
  apply Measurable.ennreal_tsum
  intro n
  exact (hk n).mul (measurable_const.min
    ((((hp.comp measurable_snd).mul (hq.comp measurable_swap)).mul
      ((hk n.swap).comp measurable_swap)).div
     (((hp.comp measurable_fst).mul hq).mul (hk n))))

omit [MeasurableSpace X] in
lemma countGateAcceptance_le_one (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞)
    (k : X → X → ℕ × ℕ → ℝ≥0∞) (hnorm : ∀ x y, ∑' n, k x y n = 1)
    (x y : X) : countGateAcceptance p q k x y ≤ 1 := by
  calc
    countGateAcceptance p q k x y ≤ ∑' n, k x y n := by
      apply ENNReal.tsum_le_tsum
      intro n
      simpa only [mul_one] using mul_le_mul_left' (min_le_left 1 _) (k x y n)
    _ = 1 := hnorm x y

omit [MeasurableSpace X] in
lemma countGate_flow_density (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞)
    (k : X → X → ℕ × ℕ → ℝ≥0∞)
    (hpfinite : ∀ x, p x ≠ ⊤) (hqfinite : ∀ x y, q x y ≠ ⊤)
    (hkfinite : ∀ x y n, k x y n ≠ ⊤) (x y : X) :
    p x * q x y * countGateAcceptance p q k x y =
      ∑' n, min (p x * q x y * k x y n) (p y * q y x * k y x n.swap) := by
  rw [countGateAcceptance, ← ENNReal.tsum_mul_left]
  apply tsum_congr
  intro n
  rw [← mul_assoc]
  exact mass_mul_mh _ _ (ENNReal.mul_ne_top
    (ENNReal.mul_ne_top (hpfinite x) (hqfinite x y)) (hkfinite x y n))

omit [MeasurableSpace X] in
/-- Swapping old/new states and gained/lost counts leaves the integrated accepted flow equal.
Zero target, proposal and auxiliary masses are allowed. -/
theorem countGate_pointwise_balance (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞)
    (k : X → X → ℕ × ℕ → ℝ≥0∞)
    (hpfinite : ∀ x, p x ≠ ⊤) (hqfinite : ∀ x y, q x y ≠ ⊤)
    (hkfinite : ∀ x y n, k x y n ≠ ⊤) (x y : X) :
    p x * q x y * countGateAcceptance p q k x y =
      p y * q y x * countGateAcceptance p q k y x := by
  rw [countGate_flow_density p q k hpfinite hqfinite hkfinite x y,
    countGate_flow_density p q k hpfinite hqfinite hkfinite y x]
  rw [← (Equiv.prodComm ℕ ℕ).tsum_eq
    (fun n => min (p y * q y x * k y x n) (p x * q x y * k x y n.swap))]
  apply tsum_congr
  intro n
  simp only [Equiv.prodComm_apply, Prod.swap_swap]
  exact min_comm _ _

/-- Count averaging supplies the symmetry premise required by the general balance theorem. -/
theorem countGate_accepted_flow_symmetric (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞) (k : X → X → ℕ × ℕ → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (Function.uncurry q))
    (hk : ∀ n, Measurable (fun a : X × X => k a.1 a.2 n))
    (hpfinite : ∀ x, p x ≠ ⊤) (hqfinite : ∀ x y, q x y ≠ ⊤)
    (hkfinite : ∀ x y n, k x y n ≠ ⊤)
    (hQ : ∀ x, Q x = μ.withDensity (q x)) :
    AcceptedFlowSymmetric Q (countGateAcceptance p q k) (μ.withDensity p) := by
  intro s t hs ht
  have hα := countGateAcceptance_measurable p q k hp hq hk
  rw [accepted_flow_density μ Q p q _ hp hq hα hpfinite hQ hs ht,
    accepted_flow_density μ Q p q _ hp hq hα hpfinite hQ ht hs]
  have hm : Measurable (fun a : X × X =>
      p a.1 * q a.1 a.2 * countGateAcceptance p q k a.1 a.2) :=
    ((hp.comp measurable_fst).mul hq).mul hα
  rw [lintegral_lintegral_swap hm.aemeasurable]
  apply lintegral_congr
  intro y
  apply lintegral_congr
  intro x
  exact countGate_pointwise_balance p q k hpfinite hqfinite hkfinite x y

/-- A normalized count gate, completed with rejection, preserves the physical target.
The counts are integrated out; they need not be retained as trajectory state. -/
theorem countGate_metropolis_correct (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (p : X → ℝ≥0∞) (q : X → X → ℝ≥0∞) (k : X → X → ℕ × ℕ → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (Function.uncurry q))
    (hk : ∀ n, Measurable (fun a : X × X => k a.1 a.2 n))
    (hpfinite : ∀ x, p x ≠ ⊤) (hqfinite : ∀ x y, q x y ≠ ⊤)
    (hkfinite : ∀ x y n, k x y n ≠ ⊤)
    (hnorm : ∀ x y, ∑' n, k x y n = 1)
    (hQ : ∀ x, Q x = μ.withDensity (q x)) :
    IsMarkovKernel (complete (accepted Q (countGateAcceptance p q k))) ∧
    Kernel.IsReversible (complete (accepted Q (countGateAcceptance p q k)))
      (μ.withDensity p) ∧
    Kernel.Invariant (complete (accepted Q (countGateAcceptance p q k)))
      (μ.withDensity p) := by
  exact accept_reject_correct Q _ _ (countGateAcceptance_measurable p q k hp hq hk)
    (countGateAcceptance_le_one p q k hnorm)
    (countGate_accepted_flow_symmetric μ Q p q k hp hq hk hpfinite hqfinite hkfinite hQ)

end ReversibleSampling
