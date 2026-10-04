import Mathlib.Analysis.Normed.Module.Basic

/-!
Exact-arithmetic necessary condition for the native member-center predicate.

The result holds in any real normed vector space, hence in Euclidean three-space.
Apply it to residuals `e k = actual_member k - ideal_member k`.  Averaging affine
member positions gives the centroid expressions in the conditional native audit.
This file makes no claim about IEEE arithmetic, outward rounding, spatial trees,
candidate implementation, or the full orientation/residue-contact classifier.
It is intentionally isolated from the project's default library build.
-/

open scoped BigOperators

namespace ReversibleSampling.NativeCentroidBound

variable {ι E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Arithmetic mean over an explicit finite set. The bound below requires it
to be nonempty; no empty-set centroid is used by the theorem. -/
noncomputable def mean (s : Finset ι) (e : ι → E) : E :=
  (s.card : ℝ)⁻¹ • ∑ k ∈ s, e k

/-- If every member residual has norm at most `ε`, its mean has norm at most
`ε`. Thus violating the centroid bound excludes the all-member predicate. -/
theorem norm_mean_le (s : Finset ι) (hs : s.Nonempty) (e : ι → E) (ε : ℝ)
    (he : ∀ k ∈ s, ‖e k‖ ≤ ε) : ‖mean s e‖ ≤ ε := by
  have hc : 0 < (s.card : ℝ) := Nat.cast_pos.mpr (Finset.card_pos.mpr hs)
  have hi : 0 < (s.card : ℝ)⁻¹ := inv_pos.mpr hc
  have hsum : ‖∑ k ∈ s, e k‖ ≤ (s.card : ℝ) * ε := by
    calc
      ‖∑ k ∈ s, e k‖ ≤ ∑ k ∈ s, ‖e k‖ := norm_sum_le s e
      _ ≤ ∑ _k ∈ s, ε := Finset.sum_le_sum he
      _ = (s.card : ℝ) * ε := by simp
  calc
    ‖mean s e‖ = (s.card : ℝ)⁻¹ * ‖∑ k ∈ s, e k‖ := by
      rw [mean, norm_smul, Real.norm_eq_abs, abs_of_pos hi]
    _ ≤ (s.card : ℝ)⁻¹ * ((s.card : ℝ) * ε) :=
      mul_le_mul_of_nonneg_left hsum hi.le
    _ = ε := by rw [← mul_assoc, inv_mul_cancel₀ (ne_of_gt hc), one_mul]

/-- The residual mean is exactly the difference of the actual and ideal
centroids, so the same necessary bound applies to those two centroids. -/
theorem norm_mean_sub_le (s : Finset ι) (hs : s.Nonempty)
    (actual ideal : ι → E) (ε : ℝ)
    (he : ∀ k ∈ s, ‖actual k - ideal k‖ ≤ ε) :
    ‖mean s actual - mean s ideal‖ ≤ ε := by
  have h := norm_mean_le s hs (fun k => actual k - ideal k) ε he
  simpa only [mean, Finset.sum_sub_distrib, smul_sub] using h

end ReversibleSampling.NativeCentroidBound

#print axioms ReversibleSampling.NativeCentroidBound.norm_mean_le
#print axioms ReversibleSampling.NativeCentroidBound.norm_mean_sub_le
