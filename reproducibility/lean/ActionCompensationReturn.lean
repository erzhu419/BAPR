import ActionCompensationRecovery
import ActionCompensationClosedLoop

/-!
# Recovery probability to closed-loop compensation loss

Mode-error probabilities are measured on the causal occupancy, not the oracle
occupancy. A change after the observed dwell is charged separately. Correct MAP
selection gives zero same-state error; no posterior-calibration premise is used.
-/

noncomputable section

namespace BAPR.ActionCompensation

open Finset MeasureTheory

/-- Strict dominance forces any maximizing selector to return that mode. -/
theorem map_mismatch_probability {Ω Mode : Type} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (b : Ω → Mode → ℝ) (selected current identified : Ω → Mode)
    (hmap : ∀ ω w, b ω w ≤ b ω (selected ω)) :
    μ.real {ω | selected ω ≠ current ω} ≤
      μ.real {ω | ∀ w, w ≠ identified ω → b ω w < b ω (identified ω)}ᶜ +
        μ.real {ω | current ω ≠ identified ω} := by
  have hsub : {ω | selected ω ≠ current ω} ⊆
      {ω | ∀ w, w ≠ identified ω → b ω w < b ω (identified ω)}ᶜ ∪
        {ω | current ω ≠ identified ω} := by
    intro ω hwrong
    by_cases hc : current ω = identified ω
    · apply Or.inl
      intro hstrict
      have hs : selected ω ≠ identified ω := by
        simpa only [Set.mem_setOf_eq, hc] using hwrong
      exact (not_lt_of_ge (hmap ω (identified ω))) (hstrict (selected ω) hs)
    · exact Or.inr hc
  exact (measureReal_mono hsub).trans (measureReal_union_le _ _)

/-- Integrate a bounded loss that is zero off a measurable error event. -/
theorem integral_abs_bound_on_event {Ω : Type} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (bad : Set Ω) (f : Ω → ℝ) (C : ℝ)
    (hbad : MeasurableSet bad) (hf : Integrable f μ)
    (hzero : ∀ ω, ω ∉ bad → f ω = 0) (hbound : ∀ᵐ ω ∂μ, |f ω| ≤ C) :
    |∫ ω, f ω ∂μ| ≤ C * μ.real bad := by
  apply abs_integral_le_integral_abs.trans
  calc
    _ ≤ ∫ ω, bad.indicator (fun _ => C) ω ∂μ := by
      apply integral_mono_ae hf.abs ((integrable_const C).indicator hbad)
      filter_upwards [hbound] with ω hωbound
      by_cases hω : ω ∈ bad
      · simpa only [Set.indicator_of_mem hω] using hωbound
      · simp only [Set.indicator_of_notMem hω, hzero ω hω, abs_zero, le_refl]
    _ = _ := by rw [integral_indicator_const C hbad, smul_eq_mul, mul_comm]

theorem boxed_action_energy {d : ℕ} (q : Action d) (hq : inBox q) :
    (∑ i, q i ^ 2) ≤ (d : ℝ) := by
  calc
    _ ≤ ∑ _ : Fin d, (1 : ℝ) := sum_le_sum fun i _ => by
      nlinarith [(hq i).1, (hq i).2]
    _ = _ := by simp

theorem modeError_self {d : ℕ} (g r q : Action d) (hg : IsPolarity g) :
    modeError g g r q = 0 := by
  funext i
  rcases hg i with h | h <;> simp [modeError, h]

theorem modeError_squared_bound {d : ℕ} {Mode : Type} [DecidableEq Mode]
    (g : Mode → Action d) (q : Action d) (reference selected z : Mode)
    (hg : ∀ w, IsPolarity (g w)) (hq : inBox q) :
    (∑ i, modeError (g z) (g selected) (g reference) q i ^ 2) ≤ 4 * (d : ℝ) := by
  rw [action_weighted_mode_error _ _ _ _ (hg z) (hg selected) (hg reference)]
  exact mul_le_mul_of_nonneg_left
    ((modeActionCost_le_energy g q selected z).trans (boxed_action_energy q hq))
    (by norm_num)

theorem modeError_l2_bound {d : ℕ} {Mode : Type} [DecidableEq Mode]
    (g : Mode → Action d) (q : Action d) (reference selected z : Mode)
    (hg : ∀ w, IsPolarity (g w)) (hq : inBox q) :
    l2Norm (modeError (g z) (g selected) (g reference) q) ≤ 2 * Real.sqrt d := by
  calc
    _ ≤ Real.sqrt (4 * (d : ℝ)) :=
      Real.sqrt_le_sqrt (modeError_squared_bound g q reference selected z hg hq)
    _ = _ := by rw [Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 4)]; norm_num

theorem compensated_executed_squared_bound {d : ℕ} {Mode : Type} [DecidableEq Mode]
    (g : Mode → Action d) (q e : Action d) (reference selected z : Mode)
    (hg : ∀ w, IsPolarity (g w)) (hq : inBox q) :
    (∑ i, (executed (applyGain (g z) (compensate (g selected) (g reference) q)) e i -
      executed (applyGain (g reference) q) e i) ^ 2) ≤ 4 * (d : ℝ) := by
  apply le_trans _ (modeError_squared_bound g q reference selected z hg hq)
  apply sum_le_sum
  intro i _
  have h := clipScalar_nonexpansive
    (applyGain (g z) (compensate (g selected) (g reference) q) i + e i)
    (applyGain (g reference) q i + e i)
  have hid : applyGain (g z) (compensate (g selected) (g reference) q) i -
      applyGain (g reference) q i = modeError (g z) (g selected) (g reference) q i := by
    simp only [applyGain, compensate, modeError]
    ring
  change |executed _ e i - executed _ e i| ≤ _ at h
  rw [add_sub_add_right_eq_sub, hid] at h
  have hsq := mul_self_le_mul_self (abs_nonneg _) h
  simpa only [← pow_two, sq_abs] using hsq

/-- Same-state nominal and executed MSE are at most `4*d` times mode-error probability. -/
theorem compensation_error_moments {Ω Mode : Type} {d : ℕ}
    [MeasurableSpace Ω] [MeasurableSpace Mode] [Fintype Mode]
    [DecidableEq Mode] [MeasurableSingletonClass Mode]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (g : Mode → Action d)
    (q e : Ω → Action d) (reference : Mode) (selected current : Ω → Mode)
    (hg : ∀ w, IsPolarity (g w)) (hq : ∀ ω, inBox (q ω))
    (hmq : ∀ i, Measurable (fun ω => q ω i)) (hme : ∀ i, Measurable (fun ω => e ω i))
    (hs : Measurable selected) (hz : Measurable current) :
    (∫ ω, ∑ i, modeError (g (current ω)) (g (selected ω)) (g reference) (q ω) i ^ 2 ∂μ) ≤
        4 * (d : ℝ) * μ.real {ω | selected ω ≠ current ω} ∧
      (∫ ω, ∑ i,
        (executed (applyGain (g (current ω)) (compensate (g (selected ω)) (g reference) (q ω)))
            (e ω) i - executed (applyGain (g reference) (q ω)) (e ω) i) ^ 2 ∂μ) ≤
        4 * (d : ℝ) * μ.real {ω | selected ω ≠ current ω} := by
  classical
  have hbad : MeasurableSet {ω | selected ω ≠ current ω} := (measurableSet_eq_fun hs hz).compl
  have hgain (i : Fin d) : Measurable (fun z : Mode => g z i) := measurable_of_finite _
  have hgsel (i : Fin d) := (hgain i).comp hs
  have hgcur (i : Fin d) := (hgain i).comp hz
  let f : Ω → ℝ := fun ω => ∑ i,
    modeError (g (current ω)) (g (selected ω)) (g reference) (q ω) i ^ 2
  let v : Ω → ℝ := fun ω => ∑ i,
    (executed (applyGain (g (current ω)) (compensate (g (selected ω)) (g reference) (q ω)))
      (e ω) i - executed (applyGain (g reference) (q ω)) (e ω) i) ^ 2
  have hmf : Measurable f := by unfold f modeError; fun_prop
  have hmv : Measurable v := by
    unfold v executed clipAction clipScalar applyGain compensate
    fun_prop
  have hfbound (ω : Ω) : |f ω| ≤ 4 * (d : ℝ) := by
    rw [abs_of_nonneg (sum_nonneg fun i _ => sq_nonneg _)]
    exact modeError_squared_bound g (q ω) reference (selected ω) (current ω) hg (hq ω)
  have hvbound (ω : Ω) : |v ω| ≤ 4 * (d : ℝ) := by
    rw [abs_of_nonneg (sum_nonneg fun i _ => sq_nonneg _)]
    exact compensated_executed_squared_bound g (q ω) (e ω) reference (selected ω)
      (current ω) hg (hq ω)
  have hfint : Integrable f μ := (integrable_const (4 * (d : ℝ))).mono'
    hmf.aestronglyMeasurable (Filter.Eventually.of_forall fun ω => by
      simpa only [Real.norm_eq_abs] using hfbound ω)
  have hvint : Integrable v μ := (integrable_const (4 * (d : ℝ))).mono'
    hmv.aestronglyMeasurable (Filter.Eventually.of_forall fun ω => by
      simpa only [Real.norm_eq_abs] using hvbound ω)
  have hfzero (ω : Ω) (hω : ω ∉ {ω | selected ω ≠ current ω}) : f ω = 0 := by
    have heq : selected ω = current ω := not_ne_iff.mp hω
    simp [f, heq, modeError_self _ _ _ (hg (current ω))]
  have hvzero (ω : Ω) (hω : ω ∉ {ω | selected ω ≠ current ω}) : v ω = 0 := by
    have heq : selected ω = current ω := not_ne_iff.mp hω
    have hid := polarity_compensates (g (current ω)) (g reference) (q ω) (hg (current ω))
    simp [v, heq, hid]
  exact ⟨(le_abs_self _).trans (integral_abs_bound_on_event μ _ f _ hbad hfint hfzero
      (Filter.Eventually.of_forall hfbound)),
    (le_abs_self _).trans (integral_abs_bound_on_event μ _ v _ hbad hvint hvzero
      (Filter.Eventually.of_forall hvbound))⟩

/-- Finite-horizon return loss from actual causal mode-error probabilities.
    The sensitivity envelope is uniform over visited states, and the occupancies
    must be the forward flow of this controller, including its true mode. -/
theorem closed_loop_mode_error_return_bound {Y E Mode : Type} {d : ℕ}
    [MeasurableSpace Y] [MeasurableSpace E] [MeasurableSpace Mode]
    [Fintype Mode] [DecidableEq Mode] [MeasurableSingletonClass Mode]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (ρ : ℕ → Measure (Y × Mode)) [∀ t, IsProbabilityMeasure (ρ t)]
    (step : (Y × Mode) → Action d → E → (Y × Mode))
    (reward : (Y × Mode) → Action d → E → ℝ) (noise : E → Action d)
    (g : Mode → Action d) (reference : Mode) (q : Y → Action d)
    (selected : ℕ → Y → Mode) (L β : ℕ → ℝ)
    (hg : ∀ z, IsPolarity (g z)) (hq : ∀ y, inBox (q y))
    (hsel : ∀ t, Measurable (selected t))
    (hC : BellmanIntegrable μ step reward noise (beliefCausalSignal g reference q selected))
    (hO : BellmanIntegrable μ step reward noise (beliefOracleSignal g reference q)) (H : ℕ)
    (hnext : ∀ t < H,
      Measurable (causalNext step noise (beliefCausalSignal g reference q selected) t))
    (hflow : ∀ t < H, ρ (t + 1) = ((ρ t).prod μ).map
      (causalNext step noise (beliefCausalSignal g reference q selected) t))
    (hval : ∀ t ≤ H, Integrable (horizonValueDifference μ step reward noise
      (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q) H t) (ρ t))
    (hiC : Integrable (measureValue μ step reward noise
      (beliefCausalSignal g reference q selected) H 0) (ρ 0))
    (hiO : Integrable (measureValue μ step reward noise
      (beliefOracleSignal g reference q) H 0) (ρ 0))
    (hL : ∀ t < H, 0 ≤ L t)
    (hsens : ∀ t < H, ∀ᵐ x ∂ρ t,
      |horizonOracleGap μ step reward noise
        (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q)
        H t x| ≤ L t * l2Norm (modeError (g x.2) (g (selected t x.1)) (g reference) (q x.1)))
    (herror : ∀ t < H, (ρ t).real {x | selected t x.1 ≠ x.2} ≤ β t) :
    |(∫ x, measureValue μ step reward noise
        (beliefCausalSignal g reference q selected) H 0 x ∂ρ 0) -
      (∫ x, measureValue μ step reward noise
        (beliefOracleSignal g reference q) H 0 x ∂ρ 0)| ≤
      2 * Real.sqrt d * ∑ t ∈ Finset.range H, L t * β t := by
  rw [occupancy_return_difference μ ρ step reward noise
    (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q)
    hC hO H hnext hflow hval hiC hiO]
  calc
    _ ≤ ∑ t ∈ Finset.range H, (2 * L t * Real.sqrt d) * β t := by
      apply (Finset.abs_sum_le_sum_abs _ _).trans
      apply Finset.sum_le_sum
      intro t ht
      have htH := Finset.mem_range.mp ht
      have htL := hL t htH
      let gap := horizonOracleGap μ step reward noise
        (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q) H t
      let bad : Set (Y × Mode) := {x | selected t x.1 ≠ x.2}
      have hbad : MeasurableSet bad :=
        (measurableSet_eq_fun ((hsel t).comp measurable_fst) measurable_snd).compl
      have hgap : Integrable gap (ρ t) := horizonOracleGap_integrable μ ρ step reward noise
        (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q)
        hC hO H t htH (hnext t htH) (hflow t htH) (hval t (by omega))
        (hval (t + 1) (by omega))
      have hzero (x : Y × Mode) (hx : x ∉ bad) : gap x = 0 := by
        have heq : selected t x.1 = x.2 := not_ne_iff.mp hx
        have hid := polarity_compensates (g x.2) (g reference) (q x.1) (hg x.2)
        simp only [gap, horizonOracleGap, beliefCausalSignal, beliefOracleSignal, heq, hid,
          sub_self]
      have hbound : ∀ᵐ x ∂ρ t, |gap x| ≤ 2 * L t * Real.sqrt d := by
        filter_upwards [hsens t htH] with x hx
        calc
          _ ≤ L t * l2Norm (modeError (g x.2) (g (selected t x.1)) (g reference) (q x.1)) :=
            hx
          _ ≤ L t * (2 * Real.sqrt d) := mul_le_mul_of_nonneg_left
            (modeError_l2_bound g (q x.1) reference (selected t x.1) x.2 hg (hq x.1))
            (hL t htH)
          _ = _ := by ring
      exact (integral_abs_bound_on_event (ρ t) bad gap _ hbad hgap hzero hbound).trans
        (mul_le_mul_of_nonneg_left (herror t htH) (by positivity))
    _ = _ := by
      rw [mul_sum]
      apply sum_congr rfl
      intro t _
      ring

end BAPR.ActionCompensation
