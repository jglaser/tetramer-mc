import ReversibleSampling.FixedLengthSurrogate

/-! Countable auxiliary traces with an involutive reverse, including two independently
sampled Poisson legs and a fair choice of leg order. Mathematical laws only: geometry,
thinning, floating point, RNG execution and resource failures remain implementation obligations.
-/

open MeasureTheory ProbabilityTheory
open scoped ENNReal NNReal ProbabilityTheory

namespace ReversibleSampling

variable {X A : Type*} [MeasurableSpace X] [Countable A]

/-- MH averaged over a normalized auxiliary trace; the reverse need not be a count swap. -/
noncomputable def traceGate (r : A ≃ A) (p : X → ℝ≥0∞)
    (k : X → X → A → ℝ≥0∞) (x y : X) : ℝ≥0∞ :=
  ∑' a, k x y a * min 1 ((p y * k y x (r a)) / (p x * k x y a))

lemma traceGate_measurable (r : A ≃ A) (p : X → ℝ≥0∞)
    (k : X → X → A → ℝ≥0∞) (hp : Measurable p)
    (hk : ∀ a, Measurable (fun xy : X × X => k xy.1 xy.2 a)) :
    Measurable (Function.uncurry (traceGate r p k)) := by
  apply Measurable.ennreal_tsum
  intro a
  exact (hk a).mul (measurable_const.min
    (((hp.comp measurable_snd).mul ((hk (r a)).comp measurable_swap)).div
      ((hp.comp measurable_fst).mul (hk a))))

omit [MeasurableSpace X] [Countable A] in
lemma traceGate_le_one (r : A ≃ A) (p : X → ℝ≥0∞)
    (k : X → X → A → ℝ≥0∞) (hnorm : ∀ x y, ∑' a, k x y a = 1)
    (x y : X) : traceGate r p k x y ≤ 1 := by
  calc
    traceGate r p k x y ≤ ∑' a, k x y a := by
      apply ENNReal.tsum_le_tsum
      intro a
      simpa only [mul_one] using mul_le_mul_left' (min_le_left 1 _) (k x y a)
    _ = 1 := hnorm x y

omit [MeasurableSpace X] [Countable A] in
lemma traceGate_flow (r : A ≃ A) (p : X → ℝ≥0∞)
    (k : X → X → A → ℝ≥0∞) (hp : ∀ x, p x ≠ ⊤)
    (hk : ∀ x y a, k x y a ≠ ⊤) (x y : X) :
    p x * traceGate r p k x y =
      ∑' a, min (p x * k x y a) (p y * k y x (r a)) := by
  rw [traceGate, ← ENNReal.tsum_mul_left]
  apply tsum_congr
  intro a
  rw [← mul_assoc]
  exact mass_mul_mh _ _ (ENNReal.mul_ne_top (hp x) (hk x y a))

omit [MeasurableSpace X] [Countable A] in
theorem traceGate_balance (r : A ≃ A) (hr : Function.Involutive r)
    (p : X → ℝ≥0∞) (k : X → X → A → ℝ≥0∞)
    (hp : ∀ x, p x ≠ ⊤) (hk : ∀ x y a, k x y a ≠ ⊤) (x y : X) :
    p x * traceGate r p k x y = p y * traceGate r p k y x := by
  rw [traceGate_flow r p k hp hk x y, traceGate_flow r p k hp hk y x]
  rw [← r.tsum_eq (fun a => min (p y * k y x a) (p x * k x y (r a)))]
  apply tsum_congr
  intro a
  rw [hr a, min_comm]

/-- Marginalize the complete auxiliary trace and add rejection. Singular proposals allowed. -/
theorem traceGate_correct (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (hswap : (μ ⊗ₘ Q).map Prod.swap = μ ⊗ₘ Q)
    (r : A ≃ A) (hr : Function.Involutive r)
    (p : X → ℝ≥0∞) (k : X → X → A → ℝ≥0∞)
    (hp : Measurable p) (hk : ∀ a, Measurable (fun xy : X × X => k xy.1 xy.2 a))
    (hpfinite : ∀ x, p x ≠ ⊤) (hkfinite : ∀ x y a, k x y a ≠ ⊤)
    (hnorm : ∀ x y, ∑' a, k x y a = 1) :
    IsMarkovKernel (complete (accepted Q (traceGate r p k))) ∧
      Kernel.IsReversible (complete (accepted Q (traceGate r p k))) (μ.withDensity p) ∧
      Kernel.Invariant (complete (accepted Q (traceGate r p k))) (μ.withDensity p) := by
  have hα := traceGate_measurable r p k hp hk
  apply accept_reject_correct Q _ _ hα (traceGate_le_one r p k hnorm)
  exact pair_flow_accepted_flow_symmetric μ Q hswap p _ hp hα hpfinite
    (traceGate_balance r hr p k hpfinite hkfinite)

abbrev TwoLegTrace := Bool × ((ℕ × ℕ) × (ℕ × ℕ))

/-- Reverse the leg order and swap gained/lost on each leg. -/
def reverseTwoLeg : TwoLegTrace ≃ TwoLegTrace where
  toFun a := (!a.1, a.2.2.swap, a.2.1.swap)
  invFun a := (!a.1, a.2.2.swap, a.2.1.swap)
  left_inv a := by rcases a with ⟨b, ⟨g, l⟩, ⟨h, j⟩⟩; cases b <;> rfl
  right_inv a := by rcases a with ⟨b, ⟨g, l⟩, ⟨h, j⟩⟩; cases b <;> rfl

theorem reverseTwoLeg_involutive : Function.Involutive reverseTwoLeg := by
  intro a
  exact reverseTwoLeg.left_inv a

/-- One order's two independent count pairs. Zero rates are allowed. -/
noncomputable def twoLegMass (lam z ap am bp bm : ℝ≥0)
    (n : (ℕ × ℕ) × (ℕ × ℕ)) : ℝ≥0∞ :=
  poissonCountMass lam z ap am n.1 * poissonCountMass lam z bp bm n.2

theorem twoLegMass_normalized (lam z ap am bp bm : ℝ≥0) :
    ∑' n, twoLegMass lam z ap am bp bm n = 1 := by
  rw [ENNReal.tsum_prod']
  simp only [twoLegMass, ENNReal.tsum_mul_left, poissonCountMass_normalized, mul_one]

lemma twoLegMass_finite (lam z ap am bp bm : ℝ≥0) (n) :
    twoLegMass lam z ap am bp bm n ≠ ⊤ :=
  ENNReal.mul_ne_top (poissonCountMass_finite _ _ _ _ _) (poissonCountMass_finite _ _ _ _ _)

/-- Empty legs contribute only zero counts, including when λ or z also vanish. -/
theorem twoLegMass_zero_changes (lam z : ℝ≥0) (n : (ℕ × ℕ) × (ℕ × ℕ)) :
    twoLegMass lam z 0 0 0 0 n = if n = ((0, 0), (0, 0)) then 1 else 0 := by
  rcases n with ⟨a, b⟩
  by_cases ha : a = (0, 0) <;> by_cases hb : b = (0, 0) <;>
    simp [twoLegMass, poissonCountMass_zero_changes, ha, hb]

noncomputable def countFactor (lam z : ℝ≥0) (n : ℕ × ℕ) : ℝ≥0∞ :=
  ENNReal.ofReal ((1 + (z : ℝ) / lam) ^ n.1 / (1 + (z : ℝ) / lam) ^ n.2)

noncomputable def twoLegFactor (lam z : ℝ≥0) (n : (ℕ × ℕ) × (ℕ × ℕ)) : ℝ≥0∞ :=
  countFactor lam z n.1 * countFactor lam z n.2

/-- Summed gained/lost counts give the same factor as multiplying the leg factors. -/
theorem twoLegFactor_aggregate (lam z : ℝ≥0) (n : (ℕ × ℕ) × (ℕ × ℕ)) :
    twoLegFactor lam z n = countFactor lam z (n.1.1 + n.2.1, n.1.2 + n.2.2) := by
  unfold twoLegFactor countFactor
  rw [← ENNReal.ofReal_mul (by positivity), div_mul_div_comm, ← pow_add, ← pow_add]

/-- Telescope the physical change across two legs, with no midpoint hard-support factor.
The intermediate value is arbitrary real geometry, not an accepted physical state. -/
theorem twoLeg_count_factor (lam z ap am bp bm : ℝ≥0) (hlam : 0 < lam)
    (cx cy : ℝ) (hdelta : cy - cx = ((ap : ℝ) - am) + ((bp : ℝ) - bm))
    (n : (ℕ × ℕ) × (ℕ × ℕ)) :
    ENNReal.ofReal (Real.exp ((z : ℝ) * cy)) *
        twoLegMass lam z bm bp am ap (n.2.swap, n.1.swap) =
      ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) * twoLegMass lam z ap am bp bm n *
        twoLegFactor lam z n := by
  let da : ℝ := (ap : ℝ) - am
  let db : ℝ := (bp : ℝ) - bm
  have ha := poisson_depletion_count_factor lam z ap am hlam 0 da (by simp [da]) n.1
  have hb := poisson_depletion_count_factor lam z bp bm hlam 0 db (by simp [db]) n.2
  simp only [mul_zero, Real.exp_zero, ENNReal.ofReal_one, one_mul] at ha hb
  have he : ENNReal.ofReal (Real.exp ((z : ℝ) * cy)) =
      ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) *
      ENNReal.ofReal (Real.exp ((z : ℝ) * da)) *
      ENNReal.ofReal (Real.exp ((z : ℝ) * db)) := by
    rw [← ENNReal.ofReal_mul (Real.exp_nonneg _),
      ← ENNReal.ofReal_mul (mul_nonneg (Real.exp_nonneg _) (Real.exp_nonneg _)),
      ← Real.exp_add, ← Real.exp_add]
    congr 2
    have hc : cy = cx + da + db := by dsimp [da, db]; linarith
    rw [hc]
    ring
  unfold twoLegMass twoLegFactor countFactor
  rw [he]
  calc
    _ = ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) *
        (ENNReal.ofReal (Real.exp ((z : ℝ) * da)) * poissonCountMass lam z am ap n.1.swap) *
        (ENNReal.ofReal (Real.exp ((z : ℝ) * db)) * poissonCountMass lam z bm bp n.2.swap) := by ac_rfl
    _ = _ := by rw [ha, hb]; ac_rfl

/-- The implemented single MH decision equals augmented-trace MH, including impossible
count combinations and zero gained/lost volumes. There is no separate leg acceptance. -/
theorem twoLeg_gate_term (lam z ap am bp bm : ℝ≥0) (hlam : 0 < lam)
    (cx cy : ℝ) (hdelta : cy - cx = ((ap : ℝ) - am) + ((bp : ℝ) - bm))
    (a b : ℝ≥0∞) (n : (ℕ × ℕ) × (ℕ × ℕ)) :
    twoLegMass lam z ap am bp bm n * min 1
      ((b * ENNReal.ofReal (Real.exp ((z : ℝ) * cy)) *
          twoLegMass lam z bm bp am ap (n.2.swap, n.1.swap)) /
       (a * ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) * twoLegMass lam z ap am bp bm n)) =
    twoLegMass lam z ap am bp bm n * min 1 ((b / a) * twoLegFactor lam z n) := by
  by_cases hk : twoLegMass lam z ap am bp bm n = 0
  · simp only [hk, zero_mul]
  · have he0 : ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) ≠ 0 :=
      ne_of_gt (ENNReal.ofReal_pos.mpr (Real.exp_pos _))
    have hf0 := mul_ne_zero he0 hk
    have hffin : ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) * twoLegMass lam z ap am bp bm n ≠ ⊤ :=
      ENNReal.mul_ne_top ENNReal.ofReal_ne_top (twoLegMass_finite _ _ _ _ _ _ _)
    have hfactor := twoLeg_count_factor lam z ap am bp bm hlam cx cy hdelta n
    have hn : b * ENNReal.ofReal (Real.exp ((z : ℝ) * cy)) *
        twoLegMass lam z bm bp am ap (n.2.swap, n.1.swap) =
      (b * twoLegFactor lam z n) *
      (ENNReal.ofReal (Real.exp ((z : ℝ) * cx)) * twoLegMass lam z ap am bp bm n) := by
      rw [mul_assoc, hfactor]
      ac_rfl
    rw [hn, mul_assoc a, ENNReal.mul_div_mul_right _ _ hf0 hffin]
    congr 2
    simp only [div_eq_mul_inv]
    ac_rfl

variable (v : X → X → Bool → Bool → ℝ≥0)

/-- `v x y order leg` is that forward leg's gained volume. Its lost volume is
the gained volume of the reversed path's opposite leg. -/
noncomputable def orderedTwoLegMass (lam z : ℝ≥0) (x y : X)
    (b : Bool) (n : (ℕ × ℕ) × (ℕ × ℕ)) : ℝ≥0∞ :=
  twoLegMass lam z (v x y b false) (v y x (!b) true)
    (v x y b true) (v y x (!b) false) n

/-- Fair order followed by independent Poisson counts on both auxiliary legs. -/
noncomputable def twoLegLaw (lam z : ℝ≥0) (x y : X) (a : TwoLegTrace) : ℝ≥0∞ :=
  (2 : ℝ≥0∞)⁻¹ * orderedTwoLegMass v lam z x y a.1 a.2

omit [MeasurableSpace X] in
theorem twoLegLaw_normalized (lam z : ℝ≥0) (x y : X) :
    ∑' a, twoLegLaw v lam z x y a = 1 := by
  rw [ENNReal.tsum_prod']
  simp only [twoLegLaw, orderedTwoLegMass, ENNReal.tsum_mul_left, twoLegMass_normalized, mul_one]
  rw [tsum_fintype]
  norm_num
  exact ENNReal.mul_inv_cancel (by norm_num) (by norm_num)

omit [MeasurableSpace X] in
lemma twoLegLaw_finite (lam z : ℝ≥0) (x y : X) (a) :
    twoLegLaw v lam z x y a ≠ ⊤ := by
  exact ENNReal.mul_ne_top (by norm_num) (twoLegMass_finite _ _ _ _ _ _ _)

lemma twoLegLaw_measurable (lam z : ℝ≥0)
    (hv : ∀ b i, Measurable (fun xy : X × X => v xy.1 xy.2 b i)) (a : TwoLegTrace) :
    Measurable (fun xy : X × X => twoLegLaw v lam z xy.1 xy.2 a) := by
  have ha := (hv a.1 false).coe_nnreal_real
  have hb := (hv a.1 true).coe_nnreal_real
  have hc := ((hv (!a.1) false).comp measurable_swap).coe_nnreal_real
  have hd := ((hv (!a.1) true).comp measurable_swap).coe_nnreal_real
  simp only [twoLegLaw, orderedTwoLegMass, twoLegMass, poissonCountMass_eq_ofReal,
    poissonCountMassReal, poissonPMFReal, NNReal.coe_mul, NNReal.coe_add]
  fun_prop

/-- The executable gate makes one decision from the sum of the two bath log factors. -/
noncomputable def twoLegAcceptance (lam z : ℝ≥0) (base : X → ℝ≥0∞) (x y : X) : ℝ≥0∞ :=
  ∑' a, twoLegLaw v lam z x y a * min 1 ((base y / base x) * twoLegFactor lam z a.2)

omit [MeasurableSpace X] in
theorem twoLegAcceptance_eq_traceGate (lam z : ℝ≥0) (hlam : 0 < lam)
    (base : X → ℝ≥0∞) (C : X → ℝ)
    (hdelta : ∀ x y b, C y - C x =
      ((v x y b false : ℝ) - v y x (!b) true) +
      ((v x y b true : ℝ) - v y x (!b) false)) (x y : X) :
    twoLegAcceptance v lam z base x y =
      traceGate reverseTwoLeg (depletionDensity z base C) (twoLegLaw v lam z) x y := by
  unfold twoLegAcceptance traceGate
  apply tsum_congr
  rintro ⟨b, n⟩
  simp only [twoLegLaw, orderedTwoLegMass, reverseTwoLeg, Equiv.coe_fn_mk,
    Bool.not_not, depletionDensity]
  let f := twoLegMass lam z (v x y b false) (v y x (!b) true)
    (v x y b true) (v y x (!b) false) n
  let r := twoLegMass lam z (v y x (!b) false) (v x y b true)
    (v y x (!b) true) (v x y b false) (n.2.swap, n.1.swap)
  have hhalf0 : (2 : ℝ≥0∞)⁻¹ ≠ 0 := by norm_num
  have hhalffin : (2 : ℝ≥0∞)⁻¹ ≠ ⊤ := by norm_num
  have hratio :
      (base y * ENNReal.ofReal (Real.exp ((z : ℝ) * C y)) * ((2 : ℝ≥0∞)⁻¹ * r)) /
        (base x * ENNReal.ofReal (Real.exp ((z : ℝ) * C x)) * ((2 : ℝ≥0∞)⁻¹ * f)) =
      (base y * ENNReal.ofReal (Real.exp ((z : ℝ) * C y)) * r) /
        (base x * ENNReal.ofReal (Real.exp ((z : ℝ) * C x)) * f) := by
    rw [mul_comm ((2 : ℝ≥0∞)⁻¹) r, ← mul_assoc,
      mul_comm ((2 : ℝ≥0∞)⁻¹) f, ← mul_assoc]
    exact ENNReal.mul_div_mul_right _ _ hhalf0 hhalffin
  change ((2 : ℝ≥0∞)⁻¹ * f) * min 1 ((base y / base x) * twoLegFactor lam z n) =
    ((2 : ℝ≥0∞)⁻¹ * f) * min 1 _
  rw [hratio, mul_assoc, mul_assoc]
  congr 1
  exact (twoLeg_gate_term lam z _ _ _ _ hlam (C x) (C y) (hdelta x y b)
    (base x) (base y) n).symm

/-- A fair-order, two-leg count bath preserves the physical endpoint target. Only the
endpoint hard support enters μ/base; no acceptance or hard test occurs at a midpoint. -/
theorem twoLeg_poisson_correct (μ : Measure X) [SFinite μ]
    (Q : Kernel X X) [IsMarkovKernel Q]
    (hswap : (μ ⊗ₘ Q).map Prod.swap = μ ⊗ₘ Q)
    (lam z : ℝ≥0) (hlam : 0 < lam) (base : X → ℝ≥0∞) (C : X → ℝ)
    (hv : ∀ b i, Measurable (fun xy : X × X => v xy.1 xy.2 b i))
    (hb : Measurable base) (hC : Measurable C) (hbfinite : ∀ x, base x ≠ ⊤)
    (hdelta : ∀ x y b, C y - C x =
      ((v x y b false : ℝ) - v y x (!b) true) +
      ((v x y b true : ℝ) - v y x (!b) false)) :
    IsMarkovKernel (complete (accepted Q (twoLegAcceptance v lam z base))) ∧
      Kernel.IsReversible (complete (accepted Q (twoLegAcceptance v lam z base)))
        (μ.withDensity (depletionDensity z base C)) ∧
      Kernel.Invariant (complete (accepted Q (twoLegAcceptance v lam z base)))
        (μ.withDensity (depletionDensity z base C)) := by
  have heq : twoLegAcceptance v lam z base =
      traceGate reverseTwoLeg (depletionDensity z base C) (twoLegLaw v lam z) := by
    funext x y
    exact twoLegAcceptance_eq_traceGate v lam z hlam base C hdelta x y
  rw [heq]
  apply traceGate_correct μ Q hswap reverseTwoLeg reverseTwoLeg_involutive
  · exact hb.mul (((measurable_const.mul hC).exp).ennreal_ofReal)
  · exact twoLegLaw_measurable v lam z hv
  · intro x
    exact ENNReal.mul_ne_top (hbfinite x) ENNReal.ofReal_ne_top
  · exact twoLegLaw_finite v lam z
  · exact twoLegLaw_normalized v lam z

/-- Fixed surrogate proposals with one two-leg correction preserve exp(z C), independently
of the surrogate score. Actual inner-kernel reversibility is an explicit premise. -/
theorem fixed_surrogate_twoLeg_correct (μ : Measure X)
    (S C : X → ℝ) (hS : Measurable S) (hC : Measurable C)
    [IsFiniteMeasure (μ.withDensity (surrogateWeight S))]
    (K : Kernel X X) [IsMarkovKernel K]
    (hK : Kernel.IsReversible K (μ.withDensity (surrogateWeight S))) (m : ℕ)
    (lam z : ℝ≥0) (hlam : 0 < lam)
    (hv : ∀ b i, Measurable (fun xy : X × X => v xy.1 xy.2 b i))
    (hdelta : ∀ x y b, C y - C x =
      ((v x y b false : ℝ) - v y x (!b) true) +
      ((v x y b true : ℝ) - v y x (!b) false)) :
    let Q := fixedSteps K m
    let α := twoLegAcceptance v lam z (surrogateWeight (fun x => -S x))
    let π := μ.withDensity (surrogateWeight (fun x => (z : ℝ) * C x))
    IsMarkovKernel (complete (accepted Q α)) ∧
      Kernel.IsReversible (complete (accepted Q α)) π ∧
      Kernel.Invariant (complete (accepted Q α)) π := by
  have hcorrect := twoLeg_poisson_correct v
    (μ.withDensity (surrogateWeight S)) (fixedSteps K m)
    (reversible_pair_swap _ _ (fixedSteps_reversible K _ hK m))
    lam z hlam (surrogateWeight (fun x => -S x)) C hv
    (surrogateWeight_measurable _ hS.neg) hC (fun _ => ENNReal.ofReal_ne_top) hdelta
  rw [surrogate_target_cancel μ S C z hS hC] at hcorrect
  exact hcorrect

end ReversibleSampling
