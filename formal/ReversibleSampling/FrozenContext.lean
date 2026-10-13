import ReversibleSampling.MetropolisHastings

/-! Metropolis--Hastings for one coordinate block with a proposal that depends on the frozen
remainder of the state (context-dependent proposals, e.g. docking from the current contact
library of the other bodies).

The state is `Y × Z`: `Y` is the moved block (one rigid body), `Z` the context (all other
bodies). The proposal draws a new `y'` from `κ (y, z)`, whose density `q z y` with respect to
`μ` may depend on the context `z` in an arbitrary measurable way, and copies `z`. The MH
acceptance uses the proposal density of the *same* context in both directions, which is
legitimate because the move does not change `z`. The proposal may lose mass (a sub-Markov
`κ`, e.g. a truncated draw that is treated as a rejection). -/

open MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace ReversibleSampling

variable {Y Z : Type*} [MeasurableSpace Y] [MeasurableSpace Z]

/-- Propose a new moved block from `κ` and copy the context. -/
noncomputable def frozenProposal (κ : Kernel (Y × Z) Y) [IsSFiniteKernel κ] :
    Kernel (Y × Z) (Y × Z) :=
  κ ×ₖ Kernel.deterministic Prod.snd measurable_snd

instance (κ : Kernel (Y × Z) Y) [IsSFiniteKernel κ] : IsSFiniteKernel (frozenProposal κ) := by
  unfold frozenProposal; infer_instance

/-- MH acceptance with the proposal density of the current context `x.2`. -/
noncomputable def frozenAcceptance (p : Y × Z → ℝ≥0∞) (q : Z → Y → Y → ℝ≥0∞)
    (x x' : Y × Z) : ℝ≥0∞ :=
  min 1 ((p x' * q x.2 x'.1 x.1) / (p x * q x.2 x.1 x'.1))

lemma frozenAcceptance_measurable (p : Y × Z → ℝ≥0∞) (q : Z → Y → Y → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (fun w : Z × Y × Y => q w.1 w.2.1 w.2.2)) :
    Measurable (Function.uncurry (frozenAcceptance p q)) := by
  have hback : Measurable (fun w : (Y × Z) × (Y × Z) => q w.1.2 w.2.1 w.1.1) :=
    hq.comp ((measurable_snd.comp measurable_fst).prodMk
      ((measurable_fst.comp measurable_snd).prodMk (measurable_fst.comp measurable_fst)))
  have hfwd : Measurable (fun w : (Y × Z) × (Y × Z) => q w.1.2 w.1.1 w.2.1) :=
    hq.comp ((measurable_snd.comp measurable_fst).prodMk
      ((measurable_fst.comp measurable_fst).prodMk (measurable_fst.comp measurable_snd)))
  exact measurable_const.min (((hp.comp measurable_snd).mul hback).div
    ((hp.comp measurable_fst).mul hfwd))

/-- Integrating against the frozen proposal integrates over the new block at fixed context. -/
lemma lintegral_frozenProposal (κ : Kernel (Y × Z) Y) [IsSFiniteKernel κ]
    (f : Y × Z → ℝ≥0∞) (hf : Measurable f) (x : Y × Z) :
    ∫⁻ x', f x' ∂(frozenProposal κ x) = ∫⁻ y', f (y', x.2) ∂(κ x) := by
  have hm : Measurable (fun y : Y => (y, x.2)) := measurable_id.prodMk measurable_const
  rw [frozenProposal, Kernel.prod_apply, Kernel.deterministic_apply, Measure.prod_dirac,
    lintegral_map hf hm]

/-- The accepted flow from `s` to `t` as an iterated integral: context outermost, then the
old and the new moved block, all with respect to the reference measures. -/
lemma frozen_flow (μ : Measure Y) [SFinite μ] (ν : Measure Z) [SFinite ν]
    (κ : Kernel (Y × Z) Y) [IsSFiniteKernel κ]
    (p : Y × Z → ℝ≥0∞) (q : Z → Y → Y → ℝ≥0∞) (α : Y × Z → Y × Z → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (fun w : Z × Y × Y => q w.1 w.2.1 w.2.2))
    (hα : Measurable (Function.uncurry α)) (hpfinite : ∀ x, p x ≠ ⊤)
    (hκ : ∀ y z, κ (y, z) = μ.withDensity (q z y))
    {s t : Set (Y × Z)} (hs : MeasurableSet s) (ht : MeasurableSet t) :
    (∫⁻ x in s, ∫⁻ x' in t, α x x' ∂(frozenProposal κ x) ∂((μ.prod ν).withDensity p)) =
    ∫⁻ z, ∫⁻ y, ∫⁻ y', s.indicator 1 (y, z) * t.indicator 1 (y', z) *
      (p (y, z) * q z y y' * α (y, z) (y', z)) ∂μ ∂μ ∂ν := by
  have hinner : Measurable (fun x => ∫⁻ x' in t, α x x' ∂(frozenProposal κ x)) :=
    hα.setLIntegral_kernel_prod_right ht
  rw [setLIntegral_withDensity_eq_setLIntegral_mul _ hp hinner hs,
    ← lintegral_indicator hs]
  have hfun : Measurable
      (s.indicator (p * fun x => ∫⁻ x' in t, α x x' ∂(frozenProposal κ x))) :=
    (hp.mul hinner).indicator hs
  rw [lintegral_prod_symm _ hfun.aemeasurable]
  apply lintegral_congr
  intro z
  apply lintegral_congr
  intro y
  have hαx : Measurable (α (y, z)) := hα.comp (measurable_const.prodMk measurable_id)
  have hqzy : Measurable (q z y) :=
    hq.comp (measurable_const.prodMk (measurable_const.prodMk measurable_id))
  have hti : Measurable (fun x' : Y × Z => t.indicator (α (y, z)) x') := hαx.indicator ht
  have hslice : Measurable (fun y' : Y => t.indicator (α (y, z)) (y', z)) :=
    hti.comp (measurable_id.prodMk measurable_const)
  by_cases hyz : (y, z) ∈ s
  · rw [Set.indicator_of_mem hyz, Pi.mul_apply, ← lintegral_indicator ht,
      lintegral_frozenProposal κ _ hti, hκ y z,
      lintegral_withDensity_eq_lintegral_mul _ hqzy hslice,
      ← lintegral_const_mul' _ _ (hpfinite _)]
    apply lintegral_congr
    intro y'
    by_cases hyz' : (y', z) ∈ t
    · simp only [Set.indicator_of_mem hyz, Set.indicator_of_mem hyz', Pi.one_apply, one_mul,
        Pi.mul_apply]
      ring
    · simp [Set.indicator_of_notMem hyz']
  · simp [Set.indicator_of_notMem hyz]

/-- With finite densities and the context-frozen MH acceptance, the accepted flow is symmetric.
The proposal density may depend on the context in any measurable way. -/
theorem frozen_mh_accepted_flow_symmetric (μ : Measure Y) [SFinite μ] (ν : Measure Z)
    [SFinite ν] (κ : Kernel (Y × Z) Y) [IsSFiniteKernel κ]
    (p : Y × Z → ℝ≥0∞) (q : Z → Y → Y → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (fun w : Z × Y × Y => q w.1 w.2.1 w.2.2))
    (hpfinite : ∀ x, p x ≠ ⊤) (hqfinite : ∀ z y y', q z y y' ≠ ⊤)
    (hκ : ∀ y z, κ (y, z) = μ.withDensity (q z y))
    {s t : Set (Y × Z)} (hs : MeasurableSet s) (ht : MeasurableSet t) :
    (∫⁻ x in s, ∫⁻ x' in t, frozenAcceptance p q x x' ∂(frozenProposal κ x)
      ∂((μ.prod ν).withDensity p)) =
    ∫⁻ x in t, ∫⁻ x' in s, frozenAcceptance p q x x' ∂(frozenProposal κ x)
      ∂((μ.prod ν).withDensity p) := by
  have hα := frozenAcceptance_measurable p q hp hq
  rw [frozen_flow μ ν κ p q _ hp hq hα hpfinite hκ hs ht,
    frozen_flow μ ν κ p q _ hp hq hα hpfinite hκ ht hs]
  apply lintegral_congr
  intro z
  -- The directed mass times the acceptance is the smaller of the two directed masses.
  have hmin : ∀ y y', p (y, z) * q z y y' * frozenAcceptance p q (y, z) (y', z) =
      min (p (y, z) * q z y y') (p (y', z) * q z y' y) := by
    intro y y'
    exact mass_mul_mh _ _ (ENNReal.mul_ne_top (hpfinite _) (hqfinite _ _ _))
  simp_rw [hmin]
  have hpz : Measurable (fun y : Y => p (y, z)) := hp.comp (measurable_id.prodMk measurable_const)
  have hqz : Measurable (fun w : Y × Y => q z w.1 w.2) :=
    hq.comp (measurable_const.prodMk measurable_id)
  have hsz : Measurable (fun y : Y => s.indicator (1 : Y × Z → ℝ≥0∞) (y, z)) :=
    (measurable_one.indicator hs).comp (measurable_id.prodMk measurable_const)
  have htz : Measurable (fun y : Y => t.indicator (1 : Y × Z → ℝ≥0∞) (y, z)) :=
    (measurable_one.indicator ht).comp (measurable_id.prodMk measurable_const)
  have hF : Measurable (fun w : Y × Y => s.indicator 1 (w.1, z) * t.indicator 1 (w.2, z) *
      min (p (w.1, z) * q z w.1 w.2) (p (w.2, z) * q z w.2 w.1)) :=
    ((hsz.comp measurable_fst).mul (htz.comp measurable_snd)).mul
      (((hpz.comp measurable_fst).mul hqz).min ((hpz.comp measurable_snd).mul
        (hqz.comp measurable_swap)))
  rw [lintegral_lintegral_swap hF.aemeasurable]
  apply lintegral_congr
  intro y'
  apply lintegral_congr
  intro y
  rw [min_comm, mul_comm (s.indicator 1 (y, z))]

/-- The context-frozen MH kernel, completed by rejection, is Markov, reversible and invariant
for the joint target `(μ ⊗ ν).withDensity p`. The proposal kernel may be sub-Markov. -/
theorem frozen_metropolis_hastings_correct (μ : Measure Y) [SFinite μ] (ν : Measure Z)
    [SFinite ν] (κ : Kernel (Y × Z) Y) [IsSFiniteKernel κ]
    (p : Y × Z → ℝ≥0∞) (q : Z → Y → Y → ℝ≥0∞)
    (hp : Measurable p) (hq : Measurable (fun w : Z × Y × Y => q w.1 w.2.1 w.2.2))
    (hpfinite : ∀ x, p x ≠ ⊤) (hqfinite : ∀ z y y', q z y y' ≠ ⊤)
    (hκ : ∀ y z, κ (y, z) = μ.withDensity (q z y))
    (hmass : ∀ x, κ x Set.univ ≤ 1) :
    IsMarkovKernel (complete ((frozenProposal κ).withDensity (frozenAcceptance p q))) ∧
    Kernel.IsReversible (complete ((frozenProposal κ).withDensity (frozenAcceptance p q)))
      ((μ.prod ν).withDensity p) ∧
    Kernel.Invariant (complete ((frozenProposal κ).withDensity (frozenAcceptance p q)))
      ((μ.prod ν).withDensity p) := by
  have hα := frozenAcceptance_measurable p q hp hq
  set A := (frozenProposal κ).withDensity (frozenAcceptance p q)
  have hAmass : ∀ x, A x Set.univ ≤ 1 := by
    intro x
    rw [Kernel.withDensity_apply' _ hα, Measure.restrict_univ]
    calc
      (∫⁻ x', frozenAcceptance p q x x' ∂(frozenProposal κ x))
          ≤ ∫⁻ _x', 1 ∂(frozenProposal κ x) := lintegral_mono (fun _ => min_le_left _ _)
      _ = frozenProposal κ x Set.univ := by simp
      _ = κ x Set.univ := by
          rw [← lintegral_one, lintegral_frozenProposal κ _ measurable_const x, lintegral_one]
      _ ≤ 1 := hmass x
  have hrev : Kernel.IsReversible A ((μ.prod ν).withDensity p) := by
    intro s t hs ht
    simp only [A, Kernel.withDensity_apply' _ hα]
    exact frozen_mh_accepted_flow_symmetric μ ν κ p q hp hq hpfinite hqfinite hκ hs ht
  exact ⟨complete_markov A hAmass, complete_reversible A _ hrev,
    complete_invariant A _ hAmass hrev⟩

/-! ## Composition: relabelling the moved block, and random-scan mixtures

For a configuration of several bodies, the update of body `b` acts on `X` through a measurable
equivalence `X ≃ᵐ Y × Z` (pose of `b`, poses of the others). Reversibility transports along
such an equivalence, and choosing the body (or the move type) at random with fixed
probabilities is a weighted sum of reversible kernels, which is reversible. -/

section Composition

variable {X W : Type*} [MeasurableSpace X] [MeasurableSpace W]

/-- Reversibility is preserved by a measurable relabelling of the state space. -/
theorem reversible_transport (e : X ≃ᵐ W) (K : Kernel W W) (π : Measure W)
    (h : Kernel.IsReversible K π) :
    Kernel.IsReversible ((K.map e.symm).comap e e.measurable) (π.map e.symm) := by
  intro s t hs ht
  have hpre : ∀ {u : Set X}, MeasurableSet u → MeasurableSet (e.symm ⁻¹' u) :=
    fun hu => e.symm.measurable hu
  have happ : ∀ (x : X) {u : Set X}, MeasurableSet u →
      ((K.map e.symm).comap e e.measurable) x u = K (e x) (e.symm ⁻¹' u) := by
    intro x u hu
    rw [Kernel.comap_apply, Kernel.map_apply' _ e.symm.measurable _ hu]
  simp_rw [happ _ ht, happ _ hs]
  have hmt : Measurable (fun x : X => K (e x) (e.symm ⁻¹' t)) :=
    (K.measurable_coe (hpre ht)).comp e.measurable
  have hms : Measurable (fun x : X => K (e x) (e.symm ⁻¹' s)) :=
    (K.measurable_coe (hpre hs)).comp e.measurable
  rw [setLIntegral_map hs hmt e.symm.measurable, setLIntegral_map ht hms e.symm.measurable]
  simp only [MeasurableEquiv.apply_symm_apply]
  exact h (hpre hs) (hpre ht)

/-- Weighting a kernel by a constant keeps it reversible. -/
theorem reversible_const_weight (K : Kernel X X) [IsSFiniteKernel K] (π : Measure X)
    (c : ℝ≥0∞) (h : Kernel.IsReversible K π) :
    Kernel.IsReversible (K.withDensity (fun _ _ => c)) π := by
  intro s t hs ht
  have hc : Measurable (Function.uncurry (fun (_ : X) (_ : X) => c)) := measurable_const
  simp_rw [Kernel.withDensity_apply' _ hc, setLIntegral_const]
  rw [lintegral_const_mul _ (K.measurable_coe ht), lintegral_const_mul _ (K.measurable_coe hs),
    h hs ht]

/-- A finite sum of reversible kernels is reversible (random scan over bodies or move types,
with the selection probabilities as constant weights). -/
theorem reversible_finset_sum {ι : Type*} (S : Finset ι) (K : ι → Kernel X X) (π : Measure X)
    (h : ∀ i ∈ S, Kernel.IsReversible (K i) π) :
    Kernel.IsReversible (∑ i ∈ S, K i) π := by
  classical
  induction S using Finset.induction_on with
  | empty =>
    intro s t _ _
    simp
  | insert a S ha ih =>
    rw [Finset.sum_insert ha]
    exact reversible_add (h a (Finset.mem_insert_self a S))
      (ih (fun i hi => h i (Finset.mem_insert_of_mem hi)))

/-- Random scan: choosing update `i` (a body, or a move type) with fixed weight `w i` and then
applying a kernel that is reversible for `π` gives a reversible kernel; if the mixture is
Markov (weights summing to one over Markov updates), it leaves `π` invariant. -/
theorem random_scan_correct {ι : Type*} (S : Finset ι) (K : ι → Kernel X X)
    [∀ i, IsSFiniteKernel (K i)] (w : ι → ℝ≥0∞) (π : Measure X)
    (h : ∀ i ∈ S, Kernel.IsReversible (K i) π)
    [IsMarkovKernel (∑ i ∈ S, (K i).withDensity (fun _ _ => w i))] :
    Kernel.IsReversible (∑ i ∈ S, (K i).withDensity (fun _ _ => w i)) π ∧
    Kernel.Invariant (∑ i ∈ S, (K i).withDensity (fun _ _ => w i)) π := by
  have hrev := reversible_finset_sum S (fun i => (K i).withDensity (fun _ _ => w i)) π
    (fun i hi => reversible_const_weight (K i) π (w i) (h i hi))
  exact ⟨hrev, hrev.invariant⟩

end Composition

end ReversibleSampling
