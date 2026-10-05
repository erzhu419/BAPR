import ActionCompensationWindows

/-!
# Recovery from cumulative excitation and prediction error

The margin recursion retains the sticky mixer's truncation at zero. Error is
charged through one discounted budget, rather than a simultaneous error tube.
-/

noncomputable section

namespace BAPR.ActionCompensation

open Finset MeasureTheory

def cumulativeMargin (δ c B : ℝ) (gap : ℕ → ℝ) : ℕ → ℝ
  | 0 => -B
  | n + 1 => δ * min (cumulativeMargin δ c B gap n) 0 + c * gap n

def discountedEvidenceError (δ c : ℝ) (η : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => δ * discountedEvidenceError δ c η n + c * η n

theorem cumulativeMargin_zero_last (δ c B : ℝ) (gap : ℕ → ℝ) (n : ℕ)
    (hδ : 0 ≤ δ) (hgap : gap n = 0) : cumulativeMargin δ c B gap (n + 1) ≤ 0 := by
  simp only [cumulativeMargin, hgap, mul_zero, add_zero]
  exact mul_nonpos_of_nonneg_of_nonpos hδ (min_le_right _ _)

theorem measurable_cumulativeMargin {Ω : Type} [MeasurableSpace Ω]
    (δ c B : ℝ) (gap : ℕ → Ω → ℝ) (hg : ∀ t, Measurable (gap t)) (N : ℕ) :
    Measurable (fun ω => cumulativeMargin δ c B (fun t => gap t ω) N) := by
  induction N with
  | zero => exact measurable_const
  | succ N ih =>
    exact (measurable_const.mul (ih.min measurable_const)).add
      (measurable_const.mul (hg N))

theorem discountedEvidenceError_nonneg (δ c : ℝ) (η : ℕ → ℝ)
    (hδ : 0 ≤ δ) (hc : 0 ≤ c) (hη : ∀ t, 0 ≤ η t) (N : ℕ) :
    0 ≤ discountedEvidenceError δ c η N := by
  induction N with
  | zero => rfl
  | succ N ih => exact add_nonneg (mul_nonneg hδ ih) (mul_nonneg hc (hη N))

theorem discountedEvidenceError_sum (δ c : ℝ) (η : ℕ → ℝ) (N : ℕ) :
    discountedEvidenceError δ c η N = c * ∑ t ∈ range N, δ ^ (N - 1 - t) * η t := by
  induction N with
  | zero => simp [discountedEvidenceError]
  | succ N ih =>
    have hshift : (∑ t ∈ range N, δ ^ (N - t) * η t) =
        δ * ∑ t ∈ range N, δ ^ (N - 1 - t) * η t := by
      rw [mul_sum]
      apply sum_congr rfl
      intro t ht
      have he : N - t = (N - 1 - t) + 1 := by
        have := mem_range.mp ht
        omega
      rw [he, pow_succ]
      ring
    simp only [discountedEvidenceError, ih, sum_range_succ,
      Nat.add_sub_cancel, Nat.sub_self, pow_zero, one_mul]
    rw [hshift]
    ring

theorem min_zero_deficit (x y D : ℝ) (hD : 0 ≤ D) (h : x - y ≤ D) :
    min x 0 - min y 0 ≤ D := by
  rcases le_total x 0 with hx | hx <;> rcases le_total y 0 with hy | hy
  · simpa [min_eq_left hx, min_eq_left hy] using h
  · simp only [min_eq_left hx, min_eq_right hy, sub_zero]
    exact hx.trans hD
  · simp only [min_eq_right hx, min_eq_left hy, zero_sub]
    linarith
  · simpa only [min_eq_right hx, min_eq_right hy, sub_self] using hD

/-- The one-sided deficit survives the mixer's nonlinear positive-odds truncation. -/
theorem cumulativeMargin_deficit (δ c B : ℝ) (ideal actual η : ℕ → ℝ)
    (hδ : 0 ≤ δ) (hc : 0 ≤ c) (hη : ∀ t, 0 ≤ η t)
    (hgap : ∀ t, ideal t - actual t ≤ η t) (N : ℕ) :
    cumulativeMargin δ c B ideal N - cumulativeMargin δ c B actual N ≤
      discountedEvidenceError δ c η N := by
  induction N with
  | zero => simp [cumulativeMargin, discountedEvidenceError]
  | succ N ih =>
    have hm := mul_le_mul_of_nonneg_left (min_zero_deficit _ _ _
      (discountedEvidenceError_nonneg δ c η hδ hc hη N) ih) hδ
    have hg := mul_le_mul_of_nonneg_left (hgap N) hc
    simp only [cumulativeMargin, discountedEvidenceError]
    nlinarith

/-- The initial sticky floor permits stale or zero-mass incoming beliefs. -/
theorem temperedFilter_cumulative_margin {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ)
    (gap : ℕ → ℝ) (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c)
    (hb : IsModeMass initial) (z w : Mode)
    (hgap : ∀ t, gap t ≤ evidence t z - evidence t w) (n : ℕ) :
    cumulativeMargin δ c (Real.log (a + k) - Real.log k) gap (n + 1) ≤
      modeLogOdds (temperedFilterPath a k δ c initial evidence (n + 1)) z w := by
  have hB : 0 ≤ Real.log (a + k) - Real.log k := by
    have h := Real.log_le_log hk (show k ≤ a + k by linarith)
    linarith
  induction n with
  | zero =>
    have hs := mul_le_mul_of_nonneg_left
      (stickyPrior_log_odds_floor a k initial ha hk hb z w) hδ
    have hg := mul_le_mul_of_nonneg_left (hgap 0) hc
    simp only [cumulativeMargin, temperedFilterPath, temperedFilter_log_odds,
      min_eq_left (neg_nonpos.mpr hB)]
    linarith
  | succ n ih =>
    have hs := stickyPrior_log_odds_lower a k _ ha hk
      (temperedFilterPath_mass a k δ c initial evidence hb (n + 1))
      (fun v => by simpa only [temperedFilterPath, temperedFilterStep] using
        modeSoftmax_pos _ v) z w
    have hm := mul_le_mul_of_nonneg_left (min_le_min ih (le_refl (0 : ℝ))) hδ
    have hprior := mul_le_mul_of_nonneg_left hs hδ
    have hg := mul_le_mul_of_nonneg_left (hgap (n + 1)) hc
    change δ * min (cumulativeMargin δ c (Real.log (a + k) - Real.log k) gap (n + 1)) 0 +
      c * gap (n + 1) ≤ modeLogOdds (temperedFilterStep a k δ c
        (temperedFilterPath a k δ c initial evidence (n + 1)) (evidence (n + 1))) z w
    rw [temperedFilter_log_odds]
    exact add_le_add (hm.trans hprior) hg

section Probability

variable {Ω : Type} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]

theorem discountedEvidenceError_integrable (δ c : ℝ) (η : ℕ → Ω → ℝ)
    (hη : ∀ t, Integrable (η t) P) (N : ℕ) :
    Integrable (fun ω => discountedEvidenceError δ c (fun t => η t ω) N) P := by
  induction N with
  | zero => exact integrable_const 0
  | succ N ih => exact (ih.const_mul δ).add ((hη N).const_mul c)

theorem discountedEvidenceError_expectation (δ c : ℝ) (η : ℕ → Ω → ℝ)
    (A : ℕ → ℝ) (hδ : 0 ≤ δ) (hc : 0 ≤ c)
    (hη : ∀ t, Integrable (η t) P) (hA : ∀ t, (∫ ω, η t ω ∂P) ≤ A t) (N : ℕ) :
    (∫ ω, discountedEvidenceError δ c (fun t => η t ω) N ∂P) ≤
      discountedEvidenceError δ c A N := by
  induction N with
  | zero => simp [discountedEvidenceError]
  | succ N ih =>
    simp only [discountedEvidenceError]
    rw [integral_add ((discountedEvidenceError_integrable P δ c η hη N).const_mul δ)
      ((hη N).const_mul c), integral_const_mul, integral_const_mul]
    exact add_le_add (mul_le_mul_of_nonneg_left ih hδ)
      (mul_le_mul_of_nonneg_left (hA N) hc)

/-- Only a cumulative excitation certificate and the mean discounted deficit are
    needed. Dependencies across times, heads and the excitation event are retained. -/
theorem cumulative_filter_recovery_probability {Mode : Type}
    [Fintype Mode] [Nonempty Mode]
    (a k δ c θ α : ℝ) (initial : Ω → Mode → ℝ)
    (evidence ideal : ℕ → Ω → Mode → ℝ) (η : ℕ → Ω → ℝ) (A : ℕ → ℝ)
    (z : Mode) (n : ℕ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c) (hθ : 0 < θ)
    (hb : ∀ ω, IsModeMass (initial ω)) (hη : ∀ t ω, 0 ≤ η t ω)
    (hint : ∀ t, Integrable (η t) P) (hA : ∀ t, (∫ ω, η t ω ∂P) ≤ A t)
    (hgap : ∀ t ω w, w ≠ z →
      ideal t ω w - (evidence t ω z - evidence t ω w) ≤ η t ω)
    (hexc : P.real {ω | ∀ w, w ≠ z → θ <
      cumulativeMargin δ c (Real.log (a + k) - Real.log k)
        (fun t => ideal t ω w) (n + 1)}ᶜ ≤ α) :
    P.real {ω | ∀ w, w ≠ z →
      temperedFilterPath a k δ c (initial ω) (fun t => evidence t ω) (n + 1) w <
      temperedFilterPath a k δ c (initial ω) (fun t => evidence t ω) (n + 1) z}ᶜ ≤
      α + discountedEvidenceError δ c A (n + 1) / θ := by
  let D := fun ω => discountedEvidenceError δ c (fun t => η t ω) (n + 1)
  have hD (ω : Ω) : 0 ≤ D ω :=
    discountedEvidenceError_nonneg δ c _ hδ hc (fun t => hη t ω) (n + 1)
  have hDi : Integrable D P := discountedEvidenceError_integrable P δ c η hint (n + 1)
  have htail : P.real {ω | θ ≤ D ω} ≤ discountedEvidenceError δ c A (n + 1) / θ := by
    apply (le_div_iff₀ hθ).mpr
    have ht := mul_meas_ge_le_integral_of_nonneg
      (Filter.Eventually.of_forall hD) hDi θ
    calc
      _ ≤ ∫ ω, D ω ∂P := by simpa only [mul_comm] using ht
      _ ≤ _ := discountedEvidenceError_expectation P δ c η A hδ hc hint hA (n + 1)
  have hsub : {ω | ∀ w, w ≠ z →
      temperedFilterPath a k δ c (initial ω) (fun t => evidence t ω) (n + 1) w <
      temperedFilterPath a k δ c (initial ω) (fun t => evidence t ω) (n + 1) z}ᶜ ⊆
    {ω | ∀ w, w ≠ z → θ < cumulativeMargin δ c (Real.log (a + k) - Real.log k)
      (fun t => ideal t ω w) (n + 1)}ᶜ ∪ {ω | θ ≤ D ω} := by
    intro ω hfail
    by_cases he : ∀ w, w ≠ z → θ < cumulativeMargin δ c
        (Real.log (a + k) - Real.log k) (fun t => ideal t ω w) (n + 1)
    · apply Or.inr
      by_contra hsmall
      have hlt : D ω < θ := lt_of_not_ge hsmall
      apply hfail
      intro w hw
      have hd := cumulativeMargin_deficit δ c (Real.log (a + k) - Real.log k)
        (fun t => ideal t ω w) (fun t => evidence t ω z - evidence t ω w)
        (fun t => η t ω) hδ hc (fun t => hη t ω) (fun t => hgap t ω w hw) (n + 1)
      have hm := temperedFilter_cumulative_margin a k δ c (initial ω)
        (fun t => evidence t ω) (fun t => evidence t ω z - evidence t ω w)
        ha hk hδ hc (hb ω) z w (fun _ => le_rfl) n
      have hx : 0 < modeLogOdds
          (temperedFilterPath a k δ c (initial ω) (fun t => evidence t ω) (n + 1)) z w := by
        have := he w hw
        change _ ≤ D ω at hd
        linarith
      have hpos (v : Mode) : 0 <
          temperedFilterPath a k δ c (initial ω) (fun t => evidence t ω) (n + 1) v := by
        simpa only [temperedFilterPath, temperedFilterStep] using modeSoftmax_pos _ v
      apply (Real.log_lt_log_iff (hpos w) (hpos z)).mp
      unfold modeLogOdds at hx
      exact sub_pos.mp hx
    · exact Or.inl he
  exact (measureReal_mono hsub).trans ((measureReal_union_le _ _).trans (add_le_add hexc htail))

end Probability

def inverseAbsoluteError {Head : Type} [Fintype Head] {d : ℕ}
    (prediction : Head → Action d) (truth : Action d) : Action d :=
  fun i => ∑ j, |prediction j i - truth i|

theorem inverseAbsoluteError_nonneg {Head : Type} [Fintype Head] {d : ℕ}
    (prediction : Head → Action d) (truth : Action d) (i : Fin d) :
    0 ≤ inverseAbsoluteError prediction truth i := sum_nonneg fun _ _ => abs_nonneg _

theorem inverseAbsoluteError_head_bound {Head : Type} [Fintype Head] {d : ℕ}
    (prediction : Head → Action d) (truth : Action d) (j : Head) (i : Fin d) :
    |prediction j i - truth i| ≤ inverseAbsoluteError prediction truth i := by
  unfold inverseAbsoluteError
  exact single_le_sum (fun h _ => abs_nonneg (prediction h i - truth i)) (mem_univ j)

/-- Separation minus an error budget is valid without a simultaneous error tube. -/
theorem inverseExcitationMargin_deficit {d : ℕ} (hd : 0 < d)
    (truth competitor ε variance : Action d) (hu : inBox truth) (hw : inBox competitor)
    (hε : ∀ i, 0 ≤ ε i) (hv : ∀ i, 0 < variance i) :
    inverseExcitationMargin truth competitor (fun _ => 0) variance -
      inverseExcitationMargin truth competitor ε variance ≤ inverseScoreBudget ε variance := by
  have hid : inverseExcitationMargin truth competitor (fun _ => 0) variance -
      inverseExcitationMargin truth competitor ε variance =
      (∑ i, 2 * ε i * |competitor i - truth i| / variance i) / (2 * d) := by
    simp only [inverseExcitationMargin, mul_zero, zero_mul, sub_zero]
    rw [← sub_div, ← sum_sub_distrib]
    congr 1
    apply sum_congr rfl
    intro i _
    ring
  rw [hid]
  calc
    _ ≤ (∑ i, 4 * ε i / variance i) / (2 * d) := by
      apply div_le_div_of_nonneg_right _ (by positivity)
      apply sum_le_sum
      intro i _
      have hs : |competitor i - truth i| ≤ 2 := by
        apply abs_le.mpr
        constructor <;> linarith [(hu i).1, (hu i).2, (hw i).1, (hw i).2]
      have he : 2 * ε i * |competitor i - truth i| ≤ 4 * ε i := by
        calc
          _ ≤ (2 * ε i) * 2 := mul_le_mul_of_nonneg_left hs
            (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) (hε i))
          _ = _ := by ring
      exact div_le_div_of_nonneg_right he (hv i).le
    _ = _ := by
      simp only [inverseScoreBudget, mul_div_assoc, ← mul_sum]
      ring

theorem inverseEvidence_cumulative_deficit {Head Mode : Type} {d : ℕ}
    [Fintype Head] [Nonempty Head] (hd : 0 < d)
    (candidate : Mode → Action d) (prediction : Head → Action d) (variance : Action d)
    (hu : ∀ w, inBox (candidate w)) (hv : ∀ i, 0 < variance i) (z w : Mode) :
    inverseExcitationMargin (candidate z) (candidate w) (fun _ => 0) variance -
      (inverseEvidence candidate prediction variance z -
        inverseEvidence candidate prediction variance w) ≤
      inverseScoreBudget (inverseAbsoluteError prediction (candidate z)) variance := by
  have he := inverseExcitationMargin_deficit hd (candidate z) (candidate w)
    (inverseAbsoluteError prediction (candidate z)) variance (hu z) (hu w)
    (inverseAbsoluteError_nonneg prediction (candidate z)) hv
  have hs := inverseEvidence_signal_margin hd candidate prediction
    (inverseAbsoluteError prediction (candidate z)) variance hv z w
    (inverseAbsoluteError_head_bound prediction (candidate z))
  linarith

section Moments

variable {Ω : Type} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]

theorem absolute_moment_le_sqrt (err : Ω → ℝ) (M : ℝ)
    (ha : Integrable (fun ω => |err ω|) P) (hs : Integrable (fun ω => err ω ^ 2) P)
    (hM : (∫ ω, err ω ^ 2 ∂P) ≤ M) : (∫ ω, |err ω| ∂P) ≤ Real.sqrt M := by
  let A := ∫ ω, |err ω| ∂P
  have hidentity : (fun ω => (|err ω| - A) ^ 2) =
      (fun ω => err ω ^ 2 - 2 * A * |err ω| + A ^ 2) := by
    funext ω
    nlinarith [sq_abs (err ω)]
  have hvariance : 0 ≤ (∫ ω, err ω ^ 2 ∂P) - A ^ 2 := by
    have h : 0 ≤ (∫ ω, (|err ω| - A) ^ 2 ∂P) :=
      integral_nonneg (fun ω => sq_nonneg (|err ω| - A))
    have hsub : (∫ ω, err ω ^ 2 - 2 * A * |err ω| ∂P) =
        (∫ ω, err ω ^ 2 ∂P) - 2 * A * A := by
      calc
        _ = (∫ ω, err ω ^ 2 ∂P) - (∫ ω, 2 * A * |err ω| ∂P) :=
          integral_sub hs (ha.const_mul (2 * A))
        _ = _ := by rw [integral_const_mul]
    have hadd : (∫ ω, err ω ^ 2 - 2 * A * |err ω| + A ^ 2 ∂P) =
        (∫ ω, err ω ^ 2 - 2 * A * |err ω| ∂P) + A ^ 2 := by
      calc
        _ = (∫ ω, err ω ^ 2 - 2 * A * |err ω| ∂P) + (∫ _ : Ω, A ^ 2 ∂P) :=
          integral_add (hs.sub (ha.const_mul (2 * A))) (integrable_const (A ^ 2))
        _ = _ := by simp
    rw [hidentity, hadd, hsub] at h
    nlinarith
  have hMn : 0 ≤ M := (integral_nonneg (fun ω => sq_nonneg (err ω))).trans hM
  have hAn : 0 ≤ A := integral_nonneg (fun ω => abs_nonneg (err ω))
  have hroot := Real.sq_sqrt hMn
  have hrn := Real.sqrt_nonneg M
  change A ≤ Real.sqrt M
  nlinarith

omit [IsProbabilityMeasure P] in
theorem inverseScoreBudget_integrable {d : ℕ} (ε : Ω → Action d) (variance : Action d)
    (hε : ∀ i, Integrable (fun ω => ε ω i) P) :
    Integrable (fun ω => inverseScoreBudget (ε ω) variance) P := by
  have hs := integrable_finset_sum univ (fun i _ => (hε i).div_const (variance i))
  exact (hs.const_mul 2).div_const (d : ℝ)

omit [IsProbabilityMeasure P] in
theorem integral_inverseScoreBudget {d : ℕ} (ε : Ω → Action d) (variance : Action d)
    (hε : ∀ i, Integrable (fun ω => ε ω i) P) :
    (∫ ω, inverseScoreBudget (ε ω) variance ∂P) =
      inverseScoreBudget (fun i => ∫ ω, ε ω i ∂P) variance := by
  simp only [inverseScoreBudget, integral_div, integral_const_mul]
  rw [integral_finset_sum univ (fun i _ => (hε i).div_const (variance i))]
  simp only [integral_div]

theorem inverseScoreBudget_mono {d : ℕ} (ε η variance : Action d)
    (hv : ∀ i, 0 < variance i) (h : ∀ i, ε i ≤ η i) :
    inverseScoreBudget ε variance ≤ inverseScoreBudget η variance := by
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
  apply mul_le_mul_of_nonneg_left _ (by norm_num)
  exact sum_le_sum fun i _ => div_le_div_of_nonneg_right (h i) (hv i).le

/-- A second-moment budget yields a mean score deficit on this same rollout law. -/
theorem inverseScoreBudget_moment_bound {Head : Type} [Fintype Head] {d : ℕ}
    (prediction : Ω → Head → Action d) (truth : Ω → Action d) (variance : Action d)
    (M : Head → Action d) (hv : ∀ i, 0 < variance i)
    (ha : ∀ j i, Integrable (fun ω => |prediction ω j i - truth ω i|) P)
    (hs : ∀ j i, Integrable (fun ω => (prediction ω j i - truth ω i) ^ 2) P)
    (hM : ∀ j i, (∫ ω, (prediction ω j i - truth ω i) ^ 2 ∂P) ≤ M j i) :
    (∫ ω, inverseScoreBudget (inverseAbsoluteError (prediction ω) (truth ω)) variance ∂P) ≤
      inverseScoreBudget (fun i => ∑ j, Real.sqrt (M j i)) variance := by
  have hint (i : Fin d) : Integrable
      (fun ω => inverseAbsoluteError (prediction ω) (truth ω) i) P :=
    integrable_finset_sum univ (fun j _ => ha j i)
  rw [integral_inverseScoreBudget P _ variance hint]
  apply inverseScoreBudget_mono _ _ variance hv
  intro i
  rw [show (fun ω => inverseAbsoluteError (prediction ω) (truth ω) i) =
    (fun ω => ∑ j, |prediction ω j i - truth ω i|) from rfl,
    integral_finset_sum univ (fun j _ => ha j i)]
  exact sum_le_sum fun j _ => absolute_moment_le_sqrt P
    (fun ω => prediction ω j i - truth ω i) (M j i) (ha j i) (hs j i) (hM j i)

end Moments

/-- The deployed inverse ensemble supplies the cumulative score-deficit budget.
    Boxed outputs establish integrability; only the population moment bounds remain inputs. -/
theorem inverse_cumulative_filter_recovery_probability {Ω Head Mode : Type} {d : ℕ}
    [MeasurableSpace Ω] [Fintype Head] [Nonempty Head] [Fintype Mode] [Nonempty Mode]
    (P : Measure Ω) [IsProbabilityMeasure P] (hd : 0 < d)
    (a k δ c θ α : ℝ) (initial : Ω → Mode → ℝ)
    (candidate : ℕ → Ω → Mode → Action d) (prediction : ℕ → Ω → Head → Action d)
    (variance : ℕ → Action d) (M : ℕ → Head → Action d) (z : Mode) (n : ℕ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c) (hθ : 0 < θ)
    (hb : ∀ ω, IsModeMass (initial ω))
    (hi : ∀ w, Measurable (fun ω => initial ω w))
    (hu : ∀ t w i, Measurable (fun ω => candidate t ω w i))
    (hp : ∀ t j i, Measurable (fun ω => prediction t ω j i))
    (hub : ∀ t ω w, inBox (candidate t ω w)) (hpb : ∀ t ω j, inBox (prediction t ω j))
    (hv : ∀ t i, 0 < variance t i)
    (hM : ∀ t j i, (∫ ω, (prediction t ω j i - candidate t ω z i) ^ 2 ∂P) ≤ M t j i)
    (hexc : P.real {ω | ∀ w, w ≠ z → θ < cumulativeMargin δ c
      (Real.log (a + k) - Real.log k)
      (fun t => inverseExcitationMargin (candidate t ω z) (candidate t ω w)
        (fun _ => 0) (variance t)) (n + 1)}ᶜ ≤ α) :
    P.real (inverseRecoveryEvent a k δ c initial candidate prediction variance z (n + 1))ᶜ ≤
        α + discountedEvidenceError δ c
          (fun t => inverseScoreBudget (fun i => ∑ j, Real.sqrt (M t j i)) (variance t))
          (n + 1) / θ ∧
      1 - (α + discountedEvidenceError δ c
          (fun t => inverseScoreBudget (fun i => ∑ j, Real.sqrt (M t j i)) (variance t))
          (n + 1) / θ) ≤
        P.real (inverseRecoveryEvent a k δ c initial candidate prediction variance z (n + 1)) := by
  let η := fun t ω => inverseScoreBudget
    (inverseAbsoluteError (prediction t ω) (candidate t ω z)) (variance t)
  have habs (t : ℕ) (j : Head) (i : Fin d) :
      Integrable (fun ω => |prediction t ω j i - candidate t ω z i|) P := by
    apply (integrable_const (2 : ℝ)).mono'
      (((hp t j i).sub (hu t z i)).abs.aestronglyMeasurable)
    filter_upwards [] with ω
    rw [Real.norm_eq_abs, abs_abs]
    apply abs_le.mpr
    constructor <;> linarith [(hpb t ω j i).1, (hpb t ω j i).2,
      (hub t ω z i).1, (hub t ω z i).2]
  have hsq (t : ℕ) (j : Head) (i : Fin d) :
      Integrable (fun ω => (prediction t ω j i - candidate t ω z i) ^ 2) P :=
    inverse_error_square_integrable P _ _ (hp t j i) (hu t z i)
      (fun ω => hpb t ω j i) (fun ω => hub t ω z i)
  have hint (t : ℕ) : Integrable (η t) P :=
    inverseScoreBudget_integrable P _ (variance t)
      (fun i => integrable_finset_sum univ (fun j _ => habs t j i))
  have hnonneg (t : ℕ) (ω : Ω) : 0 ≤ η t ω := by
    unfold η inverseScoreBudget
    apply div_nonneg
    · apply mul_nonneg (by norm_num)
      exact sum_nonneg fun i _ => div_nonneg
        (inverseAbsoluteError_nonneg _ _ i) (hv t i).le
    · exact Nat.cast_nonneg _
  have hmean (t : ℕ) : (∫ ω, η t ω ∂P) ≤
      inverseScoreBudget (fun i => ∑ j, Real.sqrt (M t j i)) (variance t) :=
    inverseScoreBudget_moment_bound P _ _ (variance t) (M t) (hv t)
      (habs t) (hsq t) (hM t)
  have hbad := cumulative_filter_recovery_probability P a k δ c θ α initial
    (fun t ω => inverseEvidence (candidate t ω) (prediction t ω) (variance t))
    (fun t ω w => inverseExcitationMargin (candidate t ω z) (candidate t ω w)
      (fun _ => 0) (variance t)) η
    (fun t => inverseScoreBudget (fun i => ∑ j, Real.sqrt (M t j i)) (variance t)) z n
    ha hk hδ hc hθ hb hnonneg hint hmean
    (fun t ω w _ => inverseEvidence_cumulative_deficit hd
      (candidate t ω) (prediction t ω) (variance t) (hub t ω) (hv t) z w) hexc
  change P.real (inverseRecoveryEvent a k δ c initial candidate prediction variance
    z (n + 1))ᶜ ≤ _ at hbad
  refine ⟨hbad, ?_⟩
  rw [probReal_compl_eq_one_sub
    (measurableSet_inverseRecoveryEvent a k δ c initial candidate prediction variance hi hu hp
      z (n + 1))] at hbad
  linarith

end BAPR.ActionCompensation
