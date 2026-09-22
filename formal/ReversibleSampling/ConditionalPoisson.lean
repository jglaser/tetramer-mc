import ReversibleSampling.CountGate

/-! The concrete gained/lost Poisson law and its exact simplified depletion correction.
Volumes may be zero; no likelihood ratio divides by a volume or a Poisson mass. -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal Nat

namespace ReversibleSampling

/-- Independent gained/lost counts with intensities lam and lam+z. -/
noncomputable def poissonCountMass (lam z vp vm : ℝ≥0) (n : ℕ × ℕ) : ℝ≥0∞ :=
  poissonPMF (lam * vp) n.1 * poissonPMF ((lam + z) * vm) n.2

lemma poissonCountMass_finite (lam z vp vm : ℝ≥0) (n : ℕ × ℕ) :
    poissonCountMass lam z vp vm n ≠ ⊤ := by
  change ENNReal.ofReal _ * ENNReal.ofReal _ ≠ ⊤
  exact ENNReal.mul_ne_top ENNReal.ofReal_ne_top ENNReal.ofReal_ne_top

/-- Count normalization, including degenerate zero-volume Poisson laws. -/
theorem poissonCountMass_normalized (lam z vp vm : ℝ≥0) :
    ∑' n, poissonCountMass lam z vp vm n = 1 := by
  rw [ENNReal.tsum_prod']
  simp only [poissonCountMass, ENNReal.tsum_mul_left, PMF.tsum_coe, mul_one]

/-- Exponential tilting of a Poisson mass, with no positive-rate restriction. -/
lemma poisson_pmf_tilt (r t : ℝ≥0) (n : ℕ) :
    poissonPMFReal (r * t) n =
      Real.exp ((r : ℝ) * (1 - t)) * poissonPMFReal r n * (t : ℝ) ^ n := by
  simp only [poissonPMFReal, NNReal.coe_mul, mul_pow]
  rw [show -((r : ℝ) * t) = (r : ℝ) * (1 - t) + -(r : ℝ) by ring,
    Real.exp_add]
  ring

lemma poisson_depletion_pmf_tilt (lam z v : ℝ≥0) (hlam : 0 < lam) (n : ℕ) :
    poissonPMFReal ((lam + z) * v) n =
      Real.exp (-(z : ℝ) * v) * poissonPMFReal (lam * v) n *
        (1 + (z : ℝ) / lam) ^ n := by
  have hlam0 : lam ≠ 0 := ne_of_gt hlam
  have hz : lam * (z / lam) = z := by
    rw [mul_comm, div_mul_cancel₀ _ hlam0]
  have rate : (lam + z) * v = (lam * v) * (1 + z / lam) := by
    calc
      (lam + z) * v = (lam + lam * (z / lam)) * v := by rw [hz]
      _ = (lam * v) * (1 + z / lam) := by ring
  rw [rate, poisson_pmf_tilt]
  simp only [NNReal.coe_mul, NNReal.coe_add, NNReal.coe_one, NNReal.coe_div]
  have hlam' : (lam : ℝ) ≠ 0 := ne_of_gt (show (0 : ℝ) < lam from hlam)
  have hz' : (lam : ℝ) * ((z : ℝ) / lam) = z := by
    rw [mul_comm, div_mul_cancel₀ _ hlam']
  have he : (lam : ℝ) * v * (1 - (1 + (z : ℝ) / lam)) = -(z : ℝ) * v := by
    calc
      _ = -((lam : ℝ) * ((z : ℝ) / lam)) * v := by ring
      _ = _ := by rw [hz']
  rw [he]

/-- Real-valued count mass, useful for finite exponential algebra. -/
noncomputable def poissonCountMassReal (lam z vp vm : ℝ≥0) (n : ℕ × ℕ) : ℝ :=
  poissonPMFReal (lam * vp) n.1 * poissonPMFReal ((lam + z) * vm) n.2

lemma poissonCountMass_eq_ofReal (lam z vp vm : ℝ≥0) (n : ℕ × ℕ) :
    poissonCountMass lam z vp vm n = ENNReal.ofReal (poissonCountMassReal lam z vp vm n) := by
  exact (ENNReal.ofReal_mul poissonPMFReal_nonneg).symm

/-- Zero-safe forward/reverse flow identity. The right powers are positive even when
one or both changed overlap regions are empty. -/
theorem poisson_depletion_count_flow (lam z vp vm : ℝ≥0) (hlam : 0 < lam)
    (cx cy : ℝ) (hdelta : cy - cx = (vp : ℝ) - vm) (n : ℕ × ℕ) :
    Real.exp ((z : ℝ) * cy) * poissonCountMassReal lam z vm vp n.swap *
        (1 + (z : ℝ) / lam) ^ n.2 =
      Real.exp ((z : ℝ) * cx) * poissonCountMassReal lam z vp vm n *
        (1 + (z : ℝ) / lam) ^ n.1 := by
  simp only [poissonCountMassReal, Prod.fst_swap, Prod.snd_swap]
  rw [poisson_depletion_pmf_tilt lam z vp hlam n.1,
    poisson_depletion_pmf_tilt lam z vm hlam n.2]
  have he : Real.exp ((z : ℝ) * cy) * Real.exp (-(z : ℝ) * vp) =
      Real.exp ((z : ℝ) * cx) * Real.exp (-(z : ℝ) * vm) := by
    rw [← Real.exp_add, ← Real.exp_add]
    congr 1
    have hc : cy = cx + (vp : ℝ) - vm := by linarith
    rw [hc]
    ring
  calc
    _ = (Real.exp ((z : ℝ) * cy) * Real.exp (-(z : ℝ) * vp)) *
      (poissonPMFReal (lam * vm) n.2 * poissonPMFReal (lam * vp) n.1 *
        (1 + (z : ℝ) / lam) ^ n.1 * (1 + (z : ℝ) / lam) ^ n.2) := by ring
    _ = _ := by rw [he]; ring

/-- The physical density cancels the exponential normalization of the reverse count law.
There is no division by a zero overlap volume or zero count probability. -/
theorem poisson_depletion_count_factor (lam z vp vm : ℝ≥0) (hlam : 0 < lam)
    (cx cy : ℝ) (hdelta : cy - cx = (vp : ℝ) - vm) (n : ℕ × ℕ) :
    ENNReal.ofReal (Real.exp ((z : ℝ) * cy)) * poissonCountMass lam z vm vp n.swap =
      ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) * poissonCountMass lam z vp vm n *
        ENNReal.ofReal ((1 + (z : ℝ) / lam) ^ n.1 / (1 + (z : ℝ) / lam) ^ n.2) := by
  have ht : 0 < 1 + (z : ℝ) / lam := by positivity
  have hp : (1 + (z : ℝ) / lam) ^ n.2 ≠ 0 := ne_of_gt (pow_pos ht _)
  have he := poisson_depletion_count_flow lam z vp vm hlam cx cy hdelta n
  have hreal : Real.exp ((z : ℝ) * cy) * poissonCountMassReal lam z vm vp n.swap =
      Real.exp ((z : ℝ) * cx) * poissonCountMassReal lam z vp vm n *
        ((1 + (z : ℝ) / lam) ^ n.1 / (1 + (z : ℝ) / lam) ^ n.2) := by
    rw [← mul_div_assoc]
    exact (eq_div_iff hp).2 he
  rw [poissonCountMass_eq_ofReal, poissonCountMass_eq_ofReal,
    ← ENNReal.ofReal_mul (Real.exp_nonneg _),
    ← ENNReal.ofReal_mul (Real.exp_nonneg _)]
  have hc : 0 ≤ Real.exp ((z : ℝ) * cx) * poissonCountMassReal lam z vp vm n :=
    mul_nonneg (Real.exp_nonneg _) (mul_nonneg poissonPMFReal_nonneg poissonPMFReal_nonneg)
  rw [← ENNReal.ofReal_mul hc]
  exact congrArg ENNReal.ofReal hreal

/-- The actual count-wise rule equals augmented MH after multiplying by its draw mass.
This identity includes impossible count pairs and zero forward/reverse proposal weights. -/
theorem poisson_gate_term_correction (lam z vp vm : ℝ≥0) (hlam : 0 < lam)
    (cx cy : ℝ) (hdelta : cy - cx = (vp : ℝ) - vm)
    (a b : ℝ≥0∞) (n : ℕ × ℕ) :
    poissonCountMass lam z vp vm n * min 1
      ((b * ENNReal.ofReal (Real.exp ((z : ℝ) * cy)) *
          poissonCountMass lam z vm vp n.swap) /
       (a * ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) *
          poissonCountMass lam z vp vm n)) =
    poissonCountMass lam z vp vm n * min 1
      ((b / a) * ENNReal.ofReal
        ((1 + (z : ℝ) / lam) ^ n.1 / (1 + (z : ℝ) / lam) ^ n.2)) := by
  by_cases hk : poissonCountMass lam z vp vm n = 0
  · simp only [hk, zero_mul]
  · have he0 : ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) ≠ 0 :=
      ne_of_gt (ENNReal.ofReal_pos.mpr (Real.exp_pos _))
    have hf0 := mul_ne_zero he0 hk
    have hffin : ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) *
        poissonCountMass lam z vp vm n ≠ ⊤ :=
      ENNReal.mul_ne_top ENNReal.ofReal_ne_top (poissonCountMass_finite lam z vp vm n)
    have hfactor := poisson_depletion_count_factor lam z vp vm hlam cx cy hdelta n
    have hn : b * ENNReal.ofReal (Real.exp ((z : ℝ) * cy)) *
        poissonCountMass lam z vm vp n.swap =
      (b * ENNReal.ofReal
        ((1 + (z : ℝ) / lam) ^ n.1 / (1 + (z : ℝ) / lam) ^ n.2)) *
      (ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) * poissonCountMass lam z vp vm n) := by
      rw [mul_assoc, hfactor]
      ac_rfl
    rw [hn, mul_assoc a, ENNReal.mul_div_mul_right _ _ hf0 hffin]
    congr 2
    simp only [div_eq_mul_inv]
    ac_rfl

variable {X : Type*} [MeasurableSpace X]

/-- The physical target includes an arbitrary nonnegative base density, e.g. hard support. -/
noncomputable def depletionDensity (z : ℝ≥0) (base : X → ℝ≥0∞) (C : X → ℝ) (x : X) : ℝ≥0∞ :=
  base x * ENNReal.ofReal (Real.exp ((z : ℝ) * C x))

/-- The executable gate averaged over its exact Poisson count law.
The signed integer exponent is written as a ratio of natural powers. -/
noncomputable def poissonGateAcceptance (lam z : ℝ≥0) (v : X → X → ℝ≥0)
    (base : X → ℝ≥0∞) (q : X → X → ℝ≥0∞) (x y : X) : ℝ≥0∞ :=
  ∑' n, poissonCountMass lam z (v x y) (v y x) n * min 1
    (((base y * q y x) / (base x * q x y)) * ENNReal.ofReal
      ((1 + (z : ℝ) / lam) ^ n.1 / (1 + (z : ℝ) / lam) ^ n.2))

omit [MeasurableSpace X] in
lemma poissonGateAcceptance_eq_countGate (lam z : ℝ≥0) (hlam : 0 < lam)
    (v : X → X → ℝ≥0) (base : X → ℝ≥0∞) (q : X → X → ℝ≥0∞) (C : X → ℝ)
    (hdelta : ∀ x y, C y - C x = (v x y : ℝ) - v y x) (x y : X) :
    poissonGateAcceptance lam z v base q x y =
      countGateAcceptance (depletionDensity z base C) q
        (fun x y => poissonCountMass lam z (v x y) (v y x)) x y := by
  unfold poissonGateAcceptance countGateAcceptance
  apply tsum_congr
  intro n
  have h := poisson_gate_term_correction lam z (v x y) (v y x) hlam
    (C x) (C y) (hdelta x y) (base x * q x y) (base y * q y x) n
  simpa only [depletionDensity, mul_right_comm (base x), mul_right_comm (base y)] using h.symm

lemma poissonCountMass_measurable (lam z : ℝ≥0) (v : X → X → ℝ≥0)
    (hv : Measurable (Function.uncurry v)) (n : ℕ × ℕ) :
    Measurable (fun a : X × X => poissonCountMass lam z (v a.1 a.2) (v a.2 a.1) n) := by
  change Measurable (fun a : X × X =>
    ENNReal.ofReal (poissonPMFReal (lam * v a.1 a.2) n.1) *
    ENNReal.ofReal (poissonPMFReal ((lam + z) * v a.2 a.1) n.2))
  have hv' : Measurable (fun a : X × X => (v a.1 a.2 : ℝ)) := hv.coe_nnreal_real
  have hvrev : Measurable (fun a : X × X => (v a.2 a.1 : ℝ)) :=
    hv'.comp measurable_swap
  simp only [poissonPMFReal, NNReal.coe_mul, NNReal.coe_add]
  fun_prop

/-- Exact implicit depletion gate on a measurable physical state space.
The only geometric premise is the gained-minus-lost volume identity; the raw proposal
need not preserve the target. Zero overlap changes, hard-zero base density and zero proposal
density are covered. Correct Poisson thinning remains an implementation obligation. -/
theorem conditional_poisson_metropolis_correct (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (lam z : ℝ≥0) (hlam : 0 < lam)
    (v : X → X → ℝ≥0) (base : X → ℝ≥0∞) (q : X → X → ℝ≥0∞) (C : X → ℝ)
    (hv : Measurable (Function.uncurry v)) (hb : Measurable base) (hC : Measurable C)
    (hq : Measurable (Function.uncurry q))
    (hbfinite : ∀ x, base x ≠ ⊤) (hqfinite : ∀ x y, q x y ≠ ⊤)
    (hdelta : ∀ x y, C y - C x = (v x y : ℝ) - v y x)
    (hQ : ∀ x, Q x = μ.withDensity (q x)) :
    IsMarkovKernel (complete (accepted Q (poissonGateAcceptance lam z v base q))) ∧
    Kernel.IsReversible (complete (accepted Q (poissonGateAcceptance lam z v base q)))
      (μ.withDensity (depletionDensity z base C)) ∧
    Kernel.Invariant (complete (accepted Q (poissonGateAcceptance lam z v base q)))
      (μ.withDensity (depletionDensity z base C)) := by
  have heq : poissonGateAcceptance lam z v base q =
      countGateAcceptance (depletionDensity z base C) q
        (fun x y => poissonCountMass lam z (v x y) (v y x)) := by
    funext x y
    exact poissonGateAcceptance_eq_countGate lam z hlam v base q C hdelta x y
  rw [heq]
  apply countGate_metropolis_correct μ Q
  · exact hb.mul (((measurable_const.mul hC).exp).ennreal_ofReal)
  · exact hq
  · exact poissonCountMass_measurable lam z v hv
  · intro x
    exact ENNReal.mul_ne_top (hbfinite x) ENNReal.ofReal_ne_top
  · exact hqfinite
  · intro x y n
    exact poissonCountMass_finite lam z (v x y) (v y x) n
  · intro x y
    exact poissonCountMass_normalized lam z (v x y) (v y x)
  · exact hQ

omit [MeasurableSpace X] in
/-- At zero activity the random gate is ordinary MH, even if lam also vanishes. -/
theorem poissonGateAcceptance_zero_activity (lam : ℝ≥0) (v : X → X → ℝ≥0)
    (base : X → ℝ≥0∞) (q : X → X → ℝ≥0∞) (x y : X) :
    poissonGateAcceptance lam 0 v base q x y = mhAcceptance base q x y := by
  unfold poissonGateAcceptance
  simp only [NNReal.coe_zero, zero_div, add_zero, one_pow, div_one,
    ENNReal.ofReal_one, mul_one]
  rw [ENNReal.tsum_mul_right, poissonCountMass_normalized, one_mul]
  rfl

/-- Empty gained and lost regions generate only the (0,0) count pair. -/
theorem poissonCountMass_zero_changes (lam z : ℝ≥0) (n : ℕ × ℕ) :
    poissonCountMass lam z 0 0 n = if n = (0, 0) then 1 else 0 := by
  rw [poissonCountMass_eq_ofReal]
  obtain ⟨g, l⟩ := n
  cases g <;> cases l <;>
    simp [poissonCountMassReal, poissonPMFReal]

omit [MeasurableSpace X] in
/-- Empty gained/lost regions leave only the base/proposal MH correction. -/
theorem poissonGateAcceptance_zero_changes (lam z : ℝ≥0) (v : X → X → ℝ≥0)
    (base : X → ℝ≥0∞) (q : X → X → ℝ≥0∞) (x y : X)
    (hxy : v x y = 0) (hyx : v y x = 0) :
    poissonGateAcceptance lam z v base q x y = mhAcceptance base q x y := by
  classical
  unfold poissonGateAcceptance
  simp only [hxy, hyx, poissonCountMass_zero_changes, ite_mul, zero_mul]
  rw [tsum_eq_single (0, 0)]
  · simp [mhAcceptance]
  · intro n hn
    simp [hn]

/-- The implementation's zero-activity shortcut is correct even at lam=0. -/
theorem conditional_poisson_zero_activity_correct (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (lam : ℝ≥0) (v : X → X → ℝ≥0)
    (base : X → ℝ≥0∞) (q : X → X → ℝ≥0∞)
    (hb : Measurable base) (hq : Measurable (Function.uncurry q))
    (hbfinite : ∀ x, base x ≠ ⊤) (hqfinite : ∀ x y, q x y ≠ ⊤)
    (hQ : ∀ x, Q x = μ.withDensity (q x)) :
    IsMarkovKernel (complete (accepted Q (poissonGateAcceptance lam 0 v base q))) ∧
    Kernel.IsReversible (complete (accepted Q (poissonGateAcceptance lam 0 v base q)))
      (μ.withDensity base) ∧
    Kernel.Invariant (complete (accepted Q (poissonGateAcceptance lam 0 v base q)))
      (μ.withDensity base) := by
  have heq : poissonGateAcceptance lam 0 v base q = mhAcceptance base q := by
    funext x y
    exact poissonGateAcceptance_zero_activity lam v base q x y
  rw [heq]
  exact metropolis_hastings_correct μ Q base q hb hq hbfinite hqfinite hQ

end ReversibleSampling
