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
