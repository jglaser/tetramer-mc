import Mathlib

/-! Count-level detailed balance of the Poisson-memory refresh.

The memory refresh keeps each of the `n` current slots independently with probability `1 - δ`
and adds `Poisson(δ Λ)` new slots. On counts this is the chain
`n → n'` with probability `Σ_k Binom(n, 1 - δ)(k) Poi(δ Λ)(n' - k)` (`k` kept slots). Each term
of this sum, weighted by the stationary `Poi(Λ)(n)`, is symmetric in `n` and `n'`, so the count
chain is reversible for `Poi(Λ)`. (Given the counts, the kept slots are a uniformly random
subset and the new slots are independent draws from `μ / Λ`; the symmetry of that mark part
follows from exchangeability and is not formalized here.) -/

open Finset

namespace ReversibleSampling.PoissonMemory

/-- Poisson probability mass `e^{-a} a^n / n!`. -/
noncomputable def poi (a : ℝ) (n : ℕ) : ℝ := Real.exp (-a) * a ^ n / n.factorial

/-- Probability of keeping exactly `k` of `n` slots, each kept with probability `1 - δ`. -/
noncomputable def keep (δ : ℝ) (n k : ℕ) : ℝ := n.choose k * (1 - δ) ^ k * δ ^ (n - k)

/-- Count-level Poisson-memory intensity `Λ(x) = lam_floor + lam_prior + lam * N(x)`, including
the fixed monomer surface-prior rate `lam_prior`. -/
noncomputable def memoryRate (lam_floor lam_prior lam : ℝ) (entries : ℕ) : ℝ :=
  lam_floor + lam_prior + lam * (entries : ℝ)

/-- Under a physical move changing the directed contact count from `entries` to `entries'`,
the ratio of Poisson-memory count normalization factors `exp(-Λ(x')) / exp(-Λ(x))` depends
only on the contact-count change `entries' - entries`; both `lam_floor` and `lam_prior` cancel
identically (`memory_delta` in `src/morph_mc.rs`). -/
theorem memoryRate_exp_ratio (lam_floor lam_prior lam : ℝ) (entries entries' : ℕ) :
    Real.exp (-memoryRate lam_floor lam_prior lam entries') /
        Real.exp (-memoryRate lam_floor lam_prior lam entries) =
      Real.exp (-lam * ((entries' : ℝ) - (entries : ℝ))) := by
  rw [← Real.exp_sub]
  congr 1
  unfold memoryRate
  ring

/-- Joint physical-plus-memory slot weight `π(x) * exp(-Λ(x)) * ∏_k μ_x(m_k)` times the
Metropolis acceptance with `dm = -lam * (N' - N) + ∑_k (log μ_{x'}(m_k) - log μ_x(m_k))` and
proposal density ratio `q_rev / q_fwd` (which may depend on the frozen memory slots `M`
and spectator contacts, as in `--memory-only` or Boltzmann-weighted library docking)
is symmetric under swapping `(x, q_fwd) ↔ (x', q_rev)`. -/
theorem memory_joint_flow_balance
    (e_x e_x' lam_floor lam_prior lam prod_mu_x prod_mu_x' q_fwd q_rev : ℝ)
    (entries entries' : ℕ)
    (hmu_x : 0 < prod_mu_x) (hmu_x' : 0 < prod_mu_x')
    (hq_fwd : 0 < q_fwd) (hq_rev : 0 < q_rev) :
    let w_x := Real.exp (-e_x) * Real.exp (-memoryRate lam_floor lam_prior lam entries) * prod_mu_x
    let w_x' := Real.exp (-e_x') * Real.exp (-memoryRate lam_floor lam_prior lam entries') * prod_mu_x'
    let dm_ratio := Real.exp (-lam * ((entries' : ℝ) - (entries : ℝ))) * (prod_mu_x' / prod_mu_x)
    let dm_ratio_rev := Real.exp (-lam * ((entries : ℝ) - (entries' : ℝ))) * (prod_mu_x / prod_mu_x')
    w_x * q_fwd * min 1 (Real.exp (-(e_x' - e_x)) * dm_ratio * (q_rev / q_fwd)) =
      w_x' * q_rev * min 1 (Real.exp (-(e_x - e_x')) * dm_ratio_rev * (q_fwd / q_rev)) := by
  intro w_x w_x' dm_ratio dm_ratio_rev
  have hwx : 0 < w_x := mul_pos (mul_pos (Real.exp_pos _) (Real.exp_pos _)) hmu_x
  have hwx' : 0 < w_x' := mul_pos (mul_pos (Real.exp_pos _) (Real.exp_pos _)) hmu_x'
  have hfx : 0 < w_x * q_fwd := mul_pos hwx hq_fwd
  have hfy : 0 < w_x' * q_rev := mul_pos hwx' hq_rev
  have hrat_fwd :
      Real.exp (-(e_x' - e_x)) * dm_ratio * (q_rev / q_fwd) = (w_x' * q_rev) / (w_x * q_fwd) := by
    dsimp [w_x, w_x', dm_ratio]
    have hexp_e : Real.exp (-(e_x' - e_x)) = Real.exp (-e_x') / Real.exp (-e_x) := by
      rw [← Real.exp_sub]
      congr 1
      ring
    have hexp_m := (memoryRate_exp_ratio lam_floor lam_prior lam entries entries').symm
    rw [hexp_e, hexp_m]
    have he1 : Real.exp (-e_x) ≠ 0 := ne_of_gt (Real.exp_pos _)
    have he2 : Real.exp (-memoryRate lam_floor lam_prior lam entries) ≠ 0 := ne_of_gt (Real.exp_pos _)
    have hm1 : prod_mu_x ≠ 0 := ne_of_gt hmu_x
    have hq1 : q_fwd ≠ 0 := ne_of_gt hq_fwd
    field_simp
  have hrat_rev :
      Real.exp (-(e_x - e_x')) * dm_ratio_rev * (q_fwd / q_rev) = (w_x * q_fwd) / (w_x' * q_rev) := by
    dsimp [w_x, w_x', dm_ratio_rev]
    have hexp_e : Real.exp (-(e_x - e_x')) = Real.exp (-e_x) / Real.exp (-e_x') := by
      rw [← Real.exp_sub]
      congr 1
      ring
    have hexp_m := (memoryRate_exp_ratio lam_floor lam_prior lam entries' entries).symm
    rw [hexp_e, hexp_m]
    have he1 : Real.exp (-e_x') ≠ 0 := ne_of_gt (Real.exp_pos _)
    have he2 : Real.exp (-memoryRate lam_floor lam_prior lam entries') ≠ 0 := ne_of_gt (Real.exp_pos _)
    have hm1 : prod_mu_x' ≠ 0 := ne_of_gt hmu_x'
    have hq1 : q_rev ≠ 0 := ne_of_gt hq_rev
    field_simp
  rw [hrat_fwd, hrat_rev]
  rw [mul_min_of_nonneg _ _ (le_of_lt hfx), mul_min_of_nonneg _ _ (le_of_lt hfy)]
  rw [mul_one, mul_one, mul_div_cancel₀ _ (ne_of_gt hfx),
    mul_div_cancel₀ _ (ne_of_gt hfy), min_comm]

/-- One term of the flow `n → n'` through `k` kept slots, in symmetric closed form. -/
theorem flow_term_closed_form (Λ δ : ℝ) {n n' k : ℕ} (hk : k ≤ n) (hk' : k ≤ n') :
    poi Λ n * (keep δ n k * poi (δ * Λ) (n' - k)) =
      Real.exp (-Λ) * Real.exp (-(δ * Λ)) * Λ ^ (n + n' - k) * (1 - δ) ^ k
        * δ ^ (n + n' - 2 * k) / (k.factorial * (n - k).factorial * (n' - k).factorial) := by
  have hfac : (n.choose k : ℝ) = n.factorial / (k.factorial * (n - k).factorial) := by
    rw [Nat.choose_eq_factorial_div_factorial hk, Nat.cast_div
      (Nat.factorial_mul_factorial_dvd_factorial hk) (by positivity)]
    push_cast; ring
  have e1 : n + n' - k = n + (n' - k) := by omega
  have e2 : n + n' - 2 * k = (n - k) + (n' - k) := by omega
  rw [e1, e2]
  simp only [poi, keep, hfac, pow_add, mul_pow]
  field_simp

/-- Termwise detailed balance: the flow `n → n'` through `k` kept slots equals the flow
`n' → n` through `k` kept slots. -/
theorem flow_term_symmetric (Λ δ : ℝ) {n n' k : ℕ} (hk : k ≤ n) (hk' : k ≤ n') :
    poi Λ n * (keep δ n k * poi (δ * Λ) (n' - k)) =
      poi Λ n' * (keep δ n' k * poi (δ * Λ) (n - k)) := by
  rw [flow_term_closed_form Λ δ hk hk', flow_term_closed_form Λ δ hk' hk,
    Nat.add_comm n n']
  ring

/-- Transition probability of the count chain. -/
noncomputable def countStep (Λ δ : ℝ) (n n' : ℕ) : ℝ :=
  ∑ k ∈ range (min n n' + 1), keep δ n k * poi (δ * Λ) (n' - k)

/-- Detailed balance of the count chain for `Poi(Λ)`. -/
theorem count_detailed_balance (Λ δ : ℝ) (n n' : ℕ) :
    poi Λ n * countStep Λ δ n n' = poi Λ n' * countStep Λ δ n' n := by
  simp only [countStep, mul_sum, min_comm n n']
  apply sum_congr rfl
  intro k hk
  have hk : k ≤ min n' n := Nat.lt_succ_iff.mp (mem_range.mp hk)
  exact (flow_term_symmetric Λ δ (le_of_le_of_eq (hk.trans (min_le_right _ _)) rfl)
    (hk.trans (min_le_left _ _)))

end ReversibleSampling.PoissonMemory
