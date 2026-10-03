import ReversibleSampling.FactorizedIndependent

/-!
Scalar bridge for a fixed root-frame guidance cloud and threshold. The count K
is an input, not formalized geometry. The independent cloud law, rigid-frame
transport, endpoint-independent conditioning normalizers, exact bounded-integer
sampling and floating-point predicates remain implementation obligations.
-/

open scoped ENNReal BigOperators

namespace ReversibleSampling.AuxiliaryOverlapThreshold

def increment (k m : ℕ) : ℝ := ((k + 1 : ℕ) : ℝ) ^ m - (k : ℝ) ^ m

/-- Threshold PMF on naturals, with explicit zero mass beyond the current count. -/
noncomputable def weight (K m k : ℕ) : ℝ :=
  if k ≤ K then increment k m / ((K + 1 : ℕ) : ℝ) ^ m else 0

theorem denominator_positive (K m : ℕ) : 0 < ((K + 1 : ℕ) : ℝ) ^ m := by
  exact pow_pos (by positivity) m

theorem increment_positive (k m : ℕ) (hm : 0 < m) : 0 < increment k m := by
  apply sub_pos.mpr
  exact pow_lt_pow_left₀ (by exact_mod_cast Nat.lt_succ_self k)
    (by positivity) (Nat.ne_of_gt hm)

theorem weight_nonnegative (K m k : ℕ) (hm : 0 < m) : 0 ≤ weight K m k := by
  unfold weight
  split_ifs
  · exact (div_pos (increment_positive k m hm) (denominator_positive K m)).le
  · exact le_refl 0

theorem weight_positive (K m k : ℕ) (hm : 0 < m) (hk : k ≤ K) :
    0 < weight K m k := by
  rw [weight, if_pos hk]
  exact div_pos (increment_positive k m hm) (denominator_positive K m)

theorem weight_outside (K m k : ℕ) (hk : K < k) : weight K m k = 0 := by
  simp [weight, Nat.not_le_of_lt hk]

theorem increment_sum (n m : ℕ) (hm : 0 < m) :
    ∑ k ∈ Finset.range n, increment k m = (n : ℝ) ^ m := by
  simpa [increment, zero_pow (Nat.ne_of_gt hm)] using
    Finset.sum_range_sub (fun k : ℕ => (k : ℝ) ^ m) n

/-- Exact CDF; the right side is also the maximum-of-m-uniform-integers CDF.
The independence and integer generator are not supplied by this scalar lemma. -/
theorem weight_cdf (K m j : ℕ) (hm : 0 < m) (hj : j ≤ K) :
    ∑ k ∈ Finset.range (j + 1), weight K m k =
      (((j + 1 : ℕ) : ℝ) / ((K + 1 : ℕ) : ℝ)) ^ m := by
  calc
    ∑ k ∈ Finset.range (j + 1), weight K m k =
        ∑ k ∈ Finset.range (j + 1), increment k m / ((K + 1 : ℕ) : ℝ) ^ m := by
      apply Finset.sum_congr rfl
      intro k hk
      have hkj : k ≤ j := Nat.lt_succ_iff.mp (Finset.mem_range.mp hk)
      simp [weight, hkj.trans hj]
    _ = (∑ k ∈ Finset.range (j + 1), increment k m) /
        ((K + 1 : ℕ) : ℝ) ^ m := (Finset.sum_div _ _ _).symm
    _ = _ := by rw [increment_sum (j + 1) m hm, div_pow]

theorem weight_normalized (K m : ℕ) (hm : 0 < m) :
    ∑ k ∈ Finset.range (K + 1), weight K m k = 1 := by
  rw [weight_cdf K m K hm le_rfl, div_self (by positivity), one_pow]

theorem zero_count (m : ℕ) (hm : 0 < m) : weight 0 m 0 = 1 := by
  simp [weight, increment, zero_pow (Nat.ne_of_gt hm)]

/-- Summing the auxiliary threshold leaves an arbitrary physical/cloud mass unchanged. -/
theorem physical_mass_marginal (w : ℝ) (K m : ℕ) (hm : 0 < m) :
    ∑ k ∈ Finset.range (K + 1), w * weight K m k = w := by
  rw [← Finset.mul_sum, weight_normalized K m hm, mul_one]

/-- The threshold increment is common to both endpoints and cancels exactly. -/
theorem weight_ratio (KX KY m k : ℕ) (hm : 0 < m)
    (hx : k ≤ KX) (hy : k ≤ KY) :
    weight KY m k / weight KX m k =
      (((KX + 1 : ℕ) : ℝ) / ((KY + 1 : ℕ) : ℝ)) ^ m := by
  have hi := ne_of_gt (increment_positive k m hm)
  have hdx := ne_of_gt (denominator_positive KX m)
  have hdy := ne_of_gt (denominator_positive KY m)
  simp only [weight, if_pos hx, if_pos hy, div_pow]
  field_simp

/-- Exact real logarithm corresponding to m*(log1p(KX)-log1p(KY)). -/
theorem log_weight_ratio (KX KY m k : ℕ) (hm : 0 < m)
    (hx : k ≤ KX) (hy : k ≤ KY) :
    Real.log (weight KY m k / weight KX m k) =
      (m : ℝ) * (Real.log ((KX + 1 : ℕ) : ℝ) - Real.log ((KY + 1 : ℕ) : ℝ)) := by
  rw [weight_ratio KX KY m k hm hx hy, Real.log_pow,
    Real.log_div (by positivity) (by positivity)]

theorem augmented_ratio (a b : ℝ≥0∞) (ha : a ≠ ⊤)
    (KX KY m k : ℕ) (hm : 0 < m) (hx : k ≤ KX) (hy : k ≤ KY) :
    (b * ENNReal.ofReal (weight KY m k)) / (a * ENNReal.ofReal (weight KX m k)) =
      (b / a) * ENNReal.ofReal
        ((((KX + 1 : ℕ) : ℝ) / ((KY + 1 : ℕ) : ℝ)) ^ m) := by
  rw [ENNReal.mul_div_mul_comm (Or.inr ENNReal.ofReal_ne_top) (Or.inl ha),
    ← ENNReal.ofReal_div_of_pos (weight_positive KX m k hm hx),
    weight_ratio KX KY m k hm hx hy]

/-- Augmented accepted-flow symmetry with the implemented auxiliary correction.
The common coefficient includes a fixed cloud density and any shared retry factor.
Zero forward/reverse proposal masses are allowed; infinite ones are excluded. -/
theorem threshold_accepted_flow (c a b : ℝ≥0∞) (ha : a ≠ ⊤) (hb : b ≠ ⊤)
    (KX KY m k : ℕ) (hm : 0 < m) (hx : k ≤ KX) (hy : k ≤ KY) :
    c * (a * ENNReal.ofReal (weight KX m k)) * min 1
      ((b / a) * ENNReal.ofReal ((((KX + 1 : ℕ) : ℝ) / ((KY + 1 : ℕ) : ℝ)) ^ m)) =
    c * (b * ENNReal.ofReal (weight KY m k)) * min 1
      ((a / b) * ENNReal.ofReal ((((KY + 1 : ℕ) : ℝ) / ((KX + 1 : ℕ) : ℝ)) ^ m)) := by
  rw [← augmented_ratio a b ha KX KY m k hm hx hy,
    ← augmented_ratio b a hb KY KX m k hm hy hx]
  exact CappedIndependent.shared_factor_accepted_flow _ _ _
    (ENNReal.mul_ne_top ha ENNReal.ofReal_ne_top)
    (ENNReal.mul_ne_top hb ENNReal.ofReal_ne_top)

/-- Conditional reuse of the existing capped law: P and k are fixed when its
probabilities are defined. Endpoint independence remains an explicit obligation. -/
theorem fixed_threshold_capped_mass (p₀ p₁ d : ℝ) (k₀ k₁ j : ℕ) :
    d * FactorizedIndependent.endpointFactor p₀ p₁ d k₀ k₁ j +
      (1 - FactorizedIndependent.jointMass p₀ p₁ d k₀ k₁) ^ j = 1 :=
  FactorizedIndependent.factorized_mass_identity p₀ p₁ d k₀ k₁ j

end ReversibleSampling.AuxiliaryOverlapThreshold
