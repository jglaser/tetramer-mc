import ReversibleSampling.CappedIndependent

/-! Scalar finite-cap composition. The final-region inclusion d ≤ p₀*p₁ is
an explicit premise; pose geometry, independence and Rust execution are not
formalized here. No division by a success probability is used. -/
open scoped ENNReal
namespace ReversibleSampling.FactorizedIndependent
open CappedIndependent

def edgeFactor (p₀ p₁ : ℝ) (k₀ k₁ : ℕ) : ℝ :=
  retryFactor p₀ k₀ * retryFactor p₁ k₁

def jointMass (p₀ p₁ d : ℝ) (k₀ k₁ : ℕ) : ℝ :=
  d * edgeFactor p₀ p₁ k₀ k₁

def endpointFactor (p₀ p₁ d : ℝ) (k₀ k₁ j : ℕ) : ℝ :=
  edgeFactor p₀ p₁ k₀ k₁ * retryFactor (jointMass p₀ p₁ d k₀ k₁) j

theorem edgeFactor_nonnegative (p₀ p₁ : ℝ) (h₀ : p₀ ≤ 1) (h₁ : p₁ ≤ 1)
    (k₀ k₁ : ℕ) : 0 ≤ edgeFactor p₀ p₁ k₀ k₁ :=
  mul_nonneg (retryFactor_nonnegative p₀ h₀ k₀) (retryFactor_nonnegative p₁ h₁ k₁)

/-- Preliminary subprobability bounds and final-region inclusion imply q≤1. -/
theorem jointMass_bounds (p₀ p₁ d : ℝ)
    (h₀lo : 0 ≤ p₀) (h₀hi : p₀ ≤ 1) (h₁lo : 0 ≤ p₁) (h₁hi : p₁ ≤ 1)
    (hdlo : 0 ≤ d) (hdhi : d ≤ p₀ * p₁) (k₀ k₁ : ℕ) :
    0 ≤ jointMass p₀ p₁ d k₀ k₁ ∧ jointMass p₀ p₁ d k₀ k₁ ≤ 1 := by
  have hf := edgeFactor_nonnegative p₀ p₁ h₀hi h₁hi k₀ k₁
  have hb₀ := capped_mass_bounds p₀ h₀lo h₀hi k₀
  have hb₁ := capped_mass_bounds p₁ h₁lo h₁hi k₁
  refine ⟨mul_nonneg hdlo hf, ?_⟩
  calc
    jointMass p₀ p₁ d k₀ k₁ ≤ (p₀ * p₁) * edgeFactor p₀ p₁ k₀ k₁ :=
      mul_le_mul_of_nonneg_right hdhi hf
    _ = (p₀ * retryFactor p₀ k₀) * (p₁ * retryFactor p₁ k₁) := by
      unfold edgeFactor; ring
    _ ≤ (p₀ * retryFactor p₀ k₀) * 1 :=
      mul_le_mul_of_nonneg_left hb₁.2 hb₀.1
    _ ≤ 1 := by simpa using hb₀.2

/-- The common endpoint-density multiplier is nonnegative on valid inputs. -/
theorem endpointFactor_nonnegative (p₀ p₁ d : ℝ)
    (h₀lo : 0 ≤ p₀) (h₀hi : p₀ ≤ 1) (h₁lo : 0 ≤ p₁) (h₁hi : p₁ ≤ 1)
    (hdlo : 0 ≤ d) (hdhi : d ≤ p₀ * p₁) (k₀ k₁ j : ℕ) :
    0 ≤ endpointFactor p₀ p₁ d k₀ k₁ j := by
  have hb := jointMass_bounds p₀ p₁ d h₀lo h₀hi h₁lo h₁hi hdlo hdhi k₀ k₁
  exact mul_nonneg (edgeFactor_nonnegative p₀ p₁ h₀hi h₁hi k₀ k₁)
    (retryFactor_nonnegative (jointMass p₀ p₁ d k₀ k₁) hb.2 j)

/-- Successful candidate mass plus joint-cap exhaustion is exactly one. -/
theorem factorized_mass_identity (p₀ p₁ d : ℝ) (k₀ k₁ j : ℕ) :
    d * endpointFactor p₀ p₁ d k₀ k₁ j
      + (1 - jointMass p₀ p₁ d k₀ k₁) ^ j = 1 := by
  unfold endpointFactor
  rw [← mul_assoc]
  exact capped_mass_identity (jointMass p₀ p₁ d k₀ k₁) j

theorem factorized_mass_bounds (p₀ p₁ d : ℝ)
    (h₀lo : 0 ≤ p₀) (h₀hi : p₀ ≤ 1) (h₁lo : 0 ≤ p₁) (h₁hi : p₁ ≤ 1)
    (hdlo : 0 ≤ d) (hdhi : d ≤ p₀ * p₁) (k₀ k₁ j : ℕ) :
    0 ≤ d * endpointFactor p₀ p₁ d k₀ k₁ j ∧
      d * endpointFactor p₀ p₁ d k₀ k₁ j ≤ 1 := by
  have hb := jointMass_bounds p₀ p₁ d h₀lo h₀hi h₁lo h₁hi hdlo hdhi k₀ k₁
  simpa only [endpointFactor, ← mul_assoc, jointMass] using
    capped_mass_bounds (jointMass p₀ p₁ d k₀ k₁) hb.1 hb.2 j

theorem zero_root_cap (p₀ p₁ d : ℝ) (k₁ j : ℕ) :
    d * endpointFactor p₀ p₁ d 0 k₁ j = 0 ∧
      (1 - jointMass p₀ p₁ d 0 k₁) ^ j = 1 := by
  simp [endpointFactor, jointMass, edgeFactor, retryFactor]

theorem zero_child_cap (p₀ p₁ d : ℝ) (k₀ j : ℕ) :
    d * endpointFactor p₀ p₁ d k₀ 0 j = 0 ∧
      (1 - jointMass p₀ p₁ d k₀ 0) ^ j = 1 := by
  simp [endpointFactor, jointMass, edgeFactor, retryFactor]

theorem zero_joint_cap (p₀ p₁ d : ℝ) (k₀ k₁ : ℕ) :
    d * endpointFactor p₀ p₁ d k₀ k₁ 0 = 0 ∧
      (1 - jointMass p₀ p₁ d k₀ k₁) ^ 0 = 1 := by
  simp [endpointFactor, retryFactor]

theorem zero_final_mass (p₀ p₁ : ℝ) (k₀ k₁ j : ℕ) :
    0 * endpointFactor p₀ p₁ 0 k₀ k₁ j = 0 ∧
      (1 - jointMass p₀ p₁ 0 k₀ k₁) ^ j = 1 := by
  simp [jointMass]

/-- Directed masses retain all target, guide and selection terms. -/
theorem factorized_accepted_flow (p₀ p₁ d : ℝ) (k₀ k₁ j : ℕ) (a b : ℝ≥0∞)
    (ha : a ≠ ⊤) (hb : b ≠ ⊤) :
    ENNReal.ofReal (endpointFactor p₀ p₁ d k₀ k₁ j) * a * min 1 (b / a) =
      ENNReal.ofReal (endpointFactor p₀ p₁ d k₀ k₁ j) * b * min 1 (a / b) :=
  shared_factor_accepted_flow _ a b ha hb

/-- Applies to an already count-marginalized Poisson flow, not to a noisy-log
substitution for the physical MH ratio. -/
theorem factorized_preserves_balance (p₀ p₁ d : ℝ) (k₀ k₁ j : ℕ)
    (forward reverse : ℝ≥0∞) (h : forward = reverse) :
    ENNReal.ofReal (endpointFactor p₀ p₁ d k₀ k₁ j) * forward =
      ENNReal.ofReal (endpointFactor p₀ p₁ d k₀ k₁ j) * reverse :=
  shared_factor_preserves_balance _ forward reverse h
end ReversibleSampling.FactorizedIndependent
