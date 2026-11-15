import ActionCompensationEvidence

/-!
# Finite-window recovery on the causal rollout law

Prediction errors, commands and incoming beliefs share one probability space.
The union bound does not require independent times, heads or coordinates.
Moment and excitation bounds concern this law, not the training-data law.
-/

noncomputable section

namespace BAPR.ActionCompensation

open Finset MeasureTheory

section ErrorTails

variable {Ω Index : Type} [m₀ : MeasurableSpace Ω]
variable (μ : Measure Ω) [IsProbabilityMeasure μ]

theorem inverse_error_square_integrable (p u : Ω → ℝ)
    (hp : Measurable p) (hu : Measurable u)
    (hpb : ∀ ω, -1 ≤ p ω ∧ p ω ≤ 1)
    (hub : ∀ ω, -1 ≤ u ω ∧ u ω ≤ 1) :
    Integrable (fun ω => (p ω - u ω) ^ 2) μ := by
  apply (integrable_const (4 : ℝ)).mono' ((hp.sub hu).pow_const 2).aestronglyMeasurable
  filter_upwards [] with ω
  rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
  nlinarith [(hpb ω).1, (hpb ω).2, (hub ω).1, (hub ω).2]

theorem inverse_error_tail_bound (err : Ω → ℝ) (ε M : ℝ)
    (hε : 0 < ε) (herr : Integrable (fun ω => err ω ^ 2) μ)
    (hM : (∫ ω, err ω ^ 2 ∂μ) ≤ M) :
    μ.real {ω | ε < |err ω|} ≤ M / ε ^ 2 := by
  have hsub : {ω | ε < |err ω|} ⊆ {ω | ε ^ 2 ≤ err ω ^ 2} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at *
    nlinarith [sq_abs (err ω)]
  have hmarkov := mul_meas_ge_le_integral_of_nonneg
    (Filter.Eventually.of_forall fun ω => sq_nonneg (err ω)) herr (ε ^ 2)
  apply (le_div_iff₀ (sq_pos_of_pos hε)).mpr
  calc
    _ ≤ μ.real {ω | ε ^ 2 ≤ err ω ^ 2} * ε ^ 2 :=
      mul_le_mul_of_nonneg_right (measureReal_mono hsub) (sq_nonneg ε)
    _ ≤ ∫ ω, err ω ^ 2 ∂μ := by simpa [mul_comm] using hmarkov
    _ ≤ M := hM

def predictionErrorFailure (err : Index → Ω → ℝ) (ε : Index → ℝ) : Set Ω :=
  ⋃ i, {ω | ε i < |err i ω|}

theorem measurableSet_predictionErrorFailure [Fintype Index]
    (err : Index → Ω → ℝ) (ε : Index → ℝ) (herr : ∀ i, Measurable (err i)) :
    MeasurableSet (predictionErrorFailure err ε) := by
  exact MeasurableSet.iUnion fun i => measurableSet_lt measurable_const (herr i).abs

/-- A finite union bound allows arbitrary dependencies within the rollout. -/
theorem prediction_error_failure_bound [Fintype Index]
    (err : Index → Ω → ℝ) (ε M : Index → ℝ)
    (hε : ∀ i, 0 < ε i)
    (herr : ∀ i, Integrable (fun ω => err i ω ^ 2) μ)
    (hM : ∀ i, (∫ ω, err i ω ^ 2 ∂μ) ≤ M i) :
    μ.real (predictionErrorFailure err ε) ≤ ∑ i, M i / ε i ^ 2 := by
  apply (measureReal_iUnion_fintype_le _).trans
  exact sum_le_sum fun i _ => inverse_error_tail_bound μ
    (err i) (ε i) (M i) (hε i) (herr i) (hM i)

/-- History-conditional moments imply marginal moments by the tower identity. -/
theorem inverse_error_moment_of_conditional {m : MeasurableSpace Ω}
    (hm : m ≤ m₀) (err : Ω → ℝ) (M : ℝ)
    (herr : Integrable (fun ω => err ω ^ 2) μ)
    (hM : ∀ᵐ ω ∂μ, μ[(fun ω => err ω ^ 2) | m] ω ≤ M) :
    (∫ ω, err ω ^ 2 ∂μ) ≤ M := by
  have htower := setIntegral_condExp (μ := μ) hm herr MeasurableSet.univ
  simp only [setIntegral_univ] at htower
  calc
    _ = ∫ ω, μ[(fun ω => err ω ^ 2) | m] ω ∂μ := htower.symm
    _ ≤ ∫ _ : Ω, M ∂μ := integral_mono_ae integrable_condExp (integrable_const M) hM
    _ = M := by simp

end ErrorTails

section Measurability

variable {Ω Head Mode : Type} [MeasurableSpace Ω] [Fintype Head] [Fintype Mode]

omit [Fintype Mode] in
theorem measurable_inverseEvidence {d : ℕ}
    (candidate : Ω → Mode → Action d) (prediction : Ω → Head → Action d)
    (variance : Action d) (z : Mode)
    (hu : ∀ w i, Measurable (fun ω => candidate ω w i))
    (hp : ∀ j i, Measurable (fun ω => prediction ω j i)) :
    Measurable (fun ω => inverseEvidence (candidate ω) (prediction ω) variance z) := by
  unfold inverseEvidence logMeanExp inverseHeadScore
  fun_prop

theorem measurable_temperedFilterPath (a k δ c : ℝ)
    (initial : Ω → Mode → ℝ) (evidence : ℕ → Ω → Mode → ℝ)
    (hi : ∀ z, Measurable (fun ω => initial ω z))
    (he : ∀ t z, Measurable (fun ω => evidence t ω z)) (t : ℕ) (z : Mode) :
    Measurable (fun ω => temperedFilterPath a k δ c (initial ω)
      (fun s => evidence s ω) t z) := by
  induction t generalizing z with
  | zero => exact hi z
  | succ t ih =>
    simp only [temperedFilterPath, temperedFilterStep, modeSoftmax, stickyPrior]
    have hs (w : Mode) : Measurable (fun ω => Real.exp
        (δ * Real.log (a * temperedFilterPath a k δ c (initial ω)
          (fun s => evidence s ω) t w + k) + c * evidence t ω w)) := by
      have hprev := ih w
      fun_prop
    exact (hs z).div (by fun_prop)

def inverseRecoveryEvent {d : ℕ} (a k δ c : ℝ) (initial : Ω → Mode → ℝ)
    (candidate : ℕ → Ω → Mode → Action d) (prediction : ℕ → Ω → Head → Action d)
    (variance : ℕ → Action d) (z : Mode) (N : ℕ) : Set Ω :=
  {ω | ∀ w, w ≠ z →
    temperedFilterPath a k δ c (initial ω)
      (fun t => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) N w <
    temperedFilterPath a k δ c (initial ω)
      (fun t => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) N z}

theorem measurableSet_inverseRecoveryEvent {d : ℕ} (a k δ c : ℝ)
    (initial : Ω → Mode → ℝ) (candidate : ℕ → Ω → Mode → Action d)
    (prediction : ℕ → Ω → Head → Action d) (variance : ℕ → Action d)
    (hi : ∀ z, Measurable (fun ω => initial ω z))
    (hu : ∀ t w i, Measurable (fun ω => candidate t ω w i))
    (hp : ∀ t j i, Measurable (fun ω => prediction t ω j i)) (z : Mode) (N : ℕ) :
    MeasurableSet (inverseRecoveryEvent a k δ c initial candidate prediction variance z N) := by
  have hb := measurable_temperedFilterPath a k δ c initial
    (fun t ω => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) hi
    (fun t w => measurable_inverseEvidence (candidate t) (prediction t) (variance t) w
      (hu t) (hp t)) N
  simp only [inverseRecoveryEvent, Set.setOf_forall]
  apply MeasurableSet.iInter
  intro w
  apply MeasurableSet.iInter
  intro _
  exact measurableSet_lt (hb w) (hb z)

def inverseExcitationEvent {d : ℕ} (candidate : ℕ → Ω → Mode → Action d)
    (ε variance : ℕ → Action d) (z : Mode) (m : ℝ) (N : ℕ) : Set Ω :=
  {ω | ∀ t : Fin N, ∀ w, w ≠ z → m ≤
    inverseExcitationMargin (candidate t ω z) (candidate t ω w) (ε t) (variance t)}

theorem measurableSet_inverseExcitationEvent {d : ℕ}
    (candidate : ℕ → Ω → Mode → Action d) (ε variance : ℕ → Action d)
    (hu : ∀ t w i, Measurable (fun ω => candidate t ω w i)) (z : Mode) (m : ℝ) (N : ℕ) :
    MeasurableSet (inverseExcitationEvent candidate ε variance z m N) := by
  simp only [inverseExcitationEvent, Set.setOf_forall]
  apply MeasurableSet.iInter
  intro t
  apply MeasurableSet.iInter
  intro w
  apply MeasurableSet.iInter
  intro _
  apply measurableSet_le measurable_const
  unfold inverseExcitationMargin
  fun_prop

end Measurability

/-- The recovery probability includes prediction tails and insufficient excitation.
    The initial belief may be stale; the mode must persist through this window. -/
theorem inverse_filter_recovery_probability {Ω Head Mode : Type} {d : ℕ}
    [MeasurableSpace Ω] [Fintype Head] [Nonempty Head] [Fintype Mode] [Nonempty Mode]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (hd : 0 < d)
    (a k δ c m α : ℝ) (initial : Ω → Mode → ℝ)
    (candidate : ℕ → Ω → Mode → Action d) (prediction : ℕ → Ω → Head → Action d)
    (ε variance : ℕ → Action d) (z : Mode) (n : ℕ)
    (M : Fin (n + 1) → Head → Action d)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 < c) (hm : 0 < m)
    (hb : ∀ ω, IsModeMass (initial ω))
    (hi : ∀ w, Measurable (fun ω => initial ω w))
    (hu : ∀ t w i, Measurable (fun ω => candidate t ω w i))
    (hp : ∀ t j i, Measurable (fun ω => prediction t ω j i))
    (hv : ∀ t, t < n + 1 → ∀ i, 0 < variance t i)
    (hε : ∀ t : Fin (n + 1), ∀ i, 0 < ε t i)
    (hint : ∀ t : Fin (n + 1), ∀ j i,
      Integrable (fun ω => (prediction t ω j i - candidate t ω z i) ^ 2) μ)
    (hM : ∀ t : Fin (n + 1), ∀ j i,
      (∫ ω, (prediction t ω j i - candidate t ω z i) ^ 2 ∂μ) ≤ M t j i)
    (hexc : μ.real (inverseExcitationEvent candidate ε variance z m (n + 1))ᶜ ≤ α)
    (hcert : 0 < filterMarginEnvelope δ (c * m)
      (Real.log (a + k) - Real.log k) (n + 1)) :
    μ.real (inverseRecoveryEvent a k δ c initial candidate prediction variance z (n + 1))ᶜ ≤
        α + ∑ t, ∑ j, ∑ i, M t j i / (ε t i) ^ 2 ∧
      1 - (α + ∑ t, ∑ j, ∑ i, M t j i / (ε t i) ^ 2) ≤
        μ.real (inverseRecoveryEvent a k δ c initial candidate prediction variance z (n + 1)) := by
  let err : (Fin (n + 1) × Head × Fin d) → Ω → ℝ :=
    fun q ω => prediction q.1 ω q.2.1 q.2.2 - candidate q.1 ω z q.2.2
  let tol : (Fin (n + 1) × Head × Fin d) → ℝ := fun q => ε q.1 q.2.2
  let moments : (Fin (n + 1) × Head × Fin d) → ℝ := fun q => M q.1 q.2.1 q.2.2
  have htail := prediction_error_failure_bound μ err tol moments
    (fun q => hε q.1 q.2.2) (fun q => hint q.1 q.2.1 q.2.2)
    (fun q => hM q.1 q.2.1 q.2.2)
  simp only [Fintype.sum_prod_type] at htail
  have hsub : (inverseRecoveryEvent a k δ c initial candidate prediction variance z (n + 1))ᶜ ⊆
      (inverseExcitationEvent candidate ε variance z m (n + 1))ᶜ ∪
        predictionErrorFailure err tol := by
    intro ω hfail
    by_cases he : ω ∈ inverseExcitationEvent candidate ε variance z m (n + 1)
    · apply Or.inr
      by_contra htube
      have herr : ∀ t, t < n + 1 → ∀ j i,
          |prediction t ω j i - candidate t ω z i| ≤ ε t i := by
        intro t ht j i
        apply le_of_not_gt
        intro hgt
        apply htube
        exact Set.mem_iUnion.mpr ⟨(⟨t, ht⟩, j, i), hgt⟩
      have hsep : ∀ t, t < n + 1 → ∀ w, w ≠ z → m ≤
          inverseExcitationMargin (candidate t ω z) (candidate t ω w) (ε t) (variance t) := by
        intro t ht w hw
        exact he ⟨t, ht⟩ w hw
      exact hfail (inverse_filter_switch_recovery hd a k δ c m (initial ω)
        (fun t => candidate t ω) (fun t => prediction t ω) ε variance
        ha hk hδ hc hm (hb ω) z n hv herr hsep hcert)
    · exact Or.inl he
  have hbad : μ.real (inverseRecoveryEvent a k δ c initial candidate prediction variance z (n + 1))ᶜ ≤
      α + ∑ t, ∑ j, ∑ i, M t j i / (ε t i) ^ 2 := by
    calc
      _ ≤ μ.real ((inverseExcitationEvent candidate ε variance z m (n + 1))ᶜ ∪
          predictionErrorFailure err tol) := measureReal_mono hsub
      _ ≤ μ.real (inverseExcitationEvent candidate ε variance z m (n + 1))ᶜ +
          μ.real (predictionErrorFailure err tol) := measureReal_union_le _ _
      _ ≤ _ := add_le_add hexc htail
  refine ⟨hbad, ?_⟩
  rw [probReal_compl_eq_one_sub
    (measurableSet_inverseRecoveryEvent a k δ c initial candidate prediction variance hi hu hp z
      (n + 1))] at hbad
  linarith

end BAPR.ActionCompensation
