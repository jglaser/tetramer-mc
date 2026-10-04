import ReversibleSampling.CappedIndependent
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
Exact finite-context algebra for posterior selection followed by a retained,
capped independent proposal. The context catalogue, its weights/densities,
hard-feasible domain, success probabilities and caps are fixed by spectators,
and therefore the SAME at both endpoints. Successful-path formulas are on the
common hard-feasible domain; the target is zero outside it.

Normalized context densities, iid complete redraws, geometric predicates,
the physical acceptance/count law and IEEE execution are implementation
premises, not proved here. Real scalars are finite. The final kernel corollary
requires its own measurable-kernel mass and integrated balance premises.
No existing default build target is changed by this isolated module.
-/

namespace ReversibleSampling.PosteriorContext

open scoped BigOperators
open CappedIndependent

variable {S X : Type*} [Fintype S]

noncomputable def mixture (w : S → ℝ) (G : S → X → ℝ) (x : X) : ℝ :=
  ∑ s, w s * G s x

noncomputable def posterior (w : S → ℝ) (G : S → X → ℝ) (x : X) (s : S) : ℝ :=
  w s * G s x / mixture w G x

/-- Positive full mixture mass normalizes posterior context selection, even
when individual contexts have zero weight or zero density at the source. -/
theorem posterior_sum_one (w : S → ℝ) (G : S → X → ℝ) (x : X)
    (hQ : 0 < mixture w G x) : (∑ s, posterior w G x s) = 1 := by
  simp only [posterior]
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt hQ)

theorem posterior_nonnegative (w : S → ℝ) (G : S → X → ℝ) (x : X) (s : S)
    (hw : 0 ≤ w s) (hG : 0 ≤ G s x) (hQ : 0 < mixture w G x) :
    0 ≤ posterior w G x s := div_nonneg (mul_nonneg hw hG) hQ.le

/-- Select a context once, then retain it through all independent hard trials.
`z s` is the hard-feasible mass of the normalized density `G s`; this equation
uses that interpretation as a premise rather than deriving it from geometry. -/
theorem posterior_capped_mass_identity (w : S → ℝ) (G : S → X → ℝ)
    (x : X) (z : S → ℝ) (K : S → ℕ) (hQ : 0 < mixture w G x) :
    (∑ s, posterior w G x s * (z s * retryFactor (z s) (K s))) +
      (∑ s, posterior w G x s * (1 - z s) ^ K s) = 1 := by
  rw [← Finset.sum_add_distrib]
  calc
    _ = ∑ s, posterior w G x s *
        (z s * retryFactor (z s) (K s) + (1 - z s) ^ K s) := by
      apply Finset.sum_congr rfl
      intro s _
      ring
    _ = ∑ s, posterior w G x s := by
      simp only [capped_mass_identity, mul_one]
    _ = 1 := posterior_sum_one w G x hQ

theorem posterior_capped_mass_bounds (w : S → ℝ) (G : S → X → ℝ)
    (x : X) (z : S → ℝ) (K : S → ℕ)
    (hw : ∀ s, 0 ≤ w s) (hG : ∀ s, 0 ≤ G s x)
    (hz0 : ∀ s, 0 ≤ z s) (hz1 : ∀ s, z s ≤ 1)
    (hQ : 0 < mixture w G x) :
    0 ≤ (∑ s, posterior w G x s * (z s * retryFactor (z s) (K s))) ∧
      (∑ s, posterior w G x s * (z s * retryFactor (z s) (K s))) ≤ 1 := by
  have hp := fun s => posterior_nonnegative w G x s (hw s) (hG s) hQ
  have hc := fun s => capped_mass_bounds (z s) (hz0 s) (hz1 s) (K s)
  constructor
  · exact Finset.sum_nonneg (fun s _ => mul_nonneg (hp s) (hc s).1)
  · calc
      _ ≤ ∑ s, posterior w G x s * 1 :=
        Finset.sum_le_sum (fun s _ => mul_le_mul_of_nonneg_left (hc s).2 (hp s))
      _ = 1 := by simpa using posterior_sum_one w G x hQ

/-- Successful candidate density on the common feasible domain. `C s` is
source-independent for the retained spectator-fixed context. -/
noncomputable def successfulPath (w : S → ℝ) (G : S → X → ℝ)
    (C : S → ℝ) (s : S) (x y : X) : ℝ :=
  posterior w G x s * G s y * C s

noncomputable def acceptance (π Q : X → ℝ) (x y : X) : ℝ :=
  min 1 ((π y / Q y) / (π x / Q x))

noncomputable def acceptedPath (π : X → ℝ) (w : S → ℝ) (G : S → X → ℝ)
    (C : S → ℝ) (s : S) (x y : X) : ℝ :=
  π x * successfulPath w G C s x y * acceptance π (mixture w G) x y

noncomputable def acceptedPathWithGate (π : X → ℝ) (w : S → ℝ) (G : S → X → ℝ)
    (C : S → ℝ) (A : X → X → ℝ) (s : S) (x y : X) : ℝ :=
  π x * successfulPath w G C s x y * A x y

/-- The actual (possibly already auxiliary-integrated) physical gate can be
used directly if its count/physical law supplies this weighted flow identity.
That identity is a premise, not a replacement by an ideal marginal MH gate. -/
theorem weighted_gate_path_symmetric (π : X → ℝ) (w : S → ℝ) (G : S → X → ℝ)
    (C : S → ℝ) (A : X → X → ℝ) (s : S) (x y : X)
    (hgate : (π x / mixture w G x) * A x y = (π y / mixture w G y) * A y x) :
    acceptedPathWithGate π w G C A s x y = acceptedPathWithGate π w G C A s y x := by
  unfold acceptedPathWithGate successfulPath posterior
  calc
    _ = (w s * G s x * G s y * C s) * ((π x / mixture w G x) * A x y) := by ring
    _ = (w s * G s x * G s y * C s) * ((π y / mixture w G y) * A y x) := by rw [hgate]
    _ = _ := by ring

theorem weighted_gate_context_sum_symmetric (π : X → ℝ) (w : S → ℝ) (G : S → X → ℝ)
    (z : S → ℝ) (K : S → ℕ) (A : X → X → ℝ) (x y : X)
    (hgate : (π x / mixture w G x) * A x y = (π y / mixture w G y) * A y x) :
    (∑ s, acceptedPathWithGate π w G (fun s => retryFactor (z s) (K s)) A s x y) =
      ∑ s, acceptedPathWithGate π w G (fun s => retryFactor (z s) (K s)) A s y x := by
  exact Finset.sum_congr rfl (fun s _ => weighted_gate_path_symmetric π w G _ A s x y hgate)

/-- The physical ratio is corrected by FULL mixture Q(x)/Q(y), not by the
selected context density. A zero target at y is allowed. -/
theorem full_mixture_correction (π Q : X → ℝ) (x y : X)
    (hπ : π x ≠ 0) (hx : Q x ≠ 0) (hy : Q y ≠ 0) :
    acceptance π Q x y = min 1 ((π y / π x) * (Q x / Q y)) := by
  unfold acceptance
  congr 1
  field_simp

/-- Real-valued MH flow identity including zero forward/target mass. -/
theorem real_mass_mul_mh (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    a * min 1 (b / a) = min a b := by
  by_cases hzero : a = 0
  · simp [hzero, min_eq_left hb]
  · have hab : a * (b / a) = b := by field_simp
    rw [mul_min_of_nonneg 1 (b / a) ha, mul_one, hab]

/-- The unknown retry factor remains a shared multiplier; it is never divided
out. Thus zero context weights/densities or a zero retry factor are included. -/
theorem accepted_path_factorization (π : X → ℝ) (w : S → ℝ) (G : S → X → ℝ)
    (C : S → ℝ) (s : S) (x y : X)
    (hπx : 0 ≤ π x) (hπy : 0 ≤ π y)
    (hQx : 0 < mixture w G x) (hQy : 0 < mixture w G y) :
    acceptedPath π w G C s x y =
      (w s * G s x * G s y * C s) * min (π x / mixture w G x) (π y / mixture w G y) := by
  unfold acceptedPath successfulPath posterior acceptance
  calc
    _ = (w s * G s x * G s y * C s) *
        ((π x / mixture w G x) * min 1 ((π y / mixture w G y) / (π x / mixture w G x))) := by ring
    _ = _ := by rw [real_mass_mul_mh _ _ (div_nonneg hπx hQx.le) (div_nonneg hπy hQy.le)]

theorem accepted_path_symmetric (π : X → ℝ) (w : S → ℝ) (G : S → X → ℝ)
    (C : S → ℝ) (s : S) (x y : X)
    (hπx : 0 ≤ π x) (hπy : 0 ≤ π y)
    (hQx : 0 < mixture w G x) (hQy : 0 < mixture w G y) :
    acceptedPath π w G C s x y = acceptedPath π w G C s y x := by
  rw [accepted_path_factorization π w G C s x y hπx hπy hQx hQy,
      accepted_path_factorization π w G C s y x hπy hπx hQy hQx, min_comm (π y / _) (π x / _)]
  ring

/-- The same endpoint-independent cap and hard mass are retained for each
context. Summing these path flows proves finite-context pointwise balance. -/
theorem capped_context_sum_symmetric (π : X → ℝ) (w : S → ℝ) (G : S → X → ℝ)
    (z : S → ℝ) (K : S → ℕ) (x y : X)
    (hπx : 0 ≤ π x) (hπy : 0 ≤ π y)
    (hQx : 0 < mixture w G x) (hQy : 0 < mixture w G y) :
    (∑ s, acceptedPath π w G (fun s => retryFactor (z s) (K s)) s x y) =
      ∑ s, acceptedPath π w G (fun s => retryFactor (z s) (K s)) s y x := by
  exact Finset.sum_congr rfl (fun s _ =>
    accepted_path_symmetric π w G _ s x y hπx hπy hQx hQy)

theorem zero_cap_path (w : S → ℝ) (G : S → X → ℝ) (z : S → ℝ)
    (s : S) (x y : X) : successfulPath w G (fun t => retryFactor (z t) 0) s x y = 0 := by
  simp [successfulPath, retryFactor]

/-- A posterior context with zero source mass contributes no successful path. -/
theorem zero_context_source_path (w : S → ℝ) (G : S → X → ℝ) (C : S → ℝ)
    (s : S) (x y : X) (hz : w s = 0 ∨ G s x = 0) :
    successfulPath w G C s x y = 0 := by
  rcases hz with hw | hG
  · simp [successfulPath, posterior, hw]
  · simp [successfulPath, posterior, hG]

/-- Q=0 cannot normalize posterior selection. The wrapper must self-loop;
the zero-extension here has no successful candidate mass to accept. -/
theorem zero_mixture_source_path (w : S → ℝ) (G : S → X → ℝ) (C : S → ℝ)
    (s : S) (x y : X) (hz : mixture w G x = 0) :
    successfulPath w G C s x y = 0 := by
  simp [successfulPath, posterior, hz]

/-- z=0 means zero integrated successful mass, not necessarily a pointwise
zero density at every null-set endpoint. The retry factor itself equals K. -/
theorem zero_hard_mass (K : ℕ) :
    (0 : ℝ) * retryFactor 0 K = 0 ∧ retryFactor 0 K = (K : ℝ) := by
  exact ⟨zero_mul _, empty_success_set K⟩

/-- Existing measure-theoretic completion is conditional on an actual
measurable accepted kernel with subprobability mass and integrated balance.
This theorem deliberately does not infer those premises from Rust geometry. -/
theorem complete_context_kernel {Y : Type*} [MeasurableSpace Y]
    (A : ProbabilityTheory.Kernel Y Y) (π : MeasureTheory.Measure Y)
    (hmass : ∀ x, A x Set.univ ≤ 1) (hbalance : ProbabilityTheory.Kernel.IsReversible A π) :
    ProbabilityTheory.IsMarkovKernel (ReversibleSampling.complete A) ∧
      ProbabilityTheory.Kernel.IsReversible (ReversibleSampling.complete A) π ∧
      ProbabilityTheory.Kernel.Invariant (ReversibleSampling.complete A) π :=
  ⟨complete_markov A hmass, complete_reversible A π hbalance,
   complete_invariant A π hmass hbalance⟩

end ReversibleSampling.PosteriorContext
