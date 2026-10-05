import ActionCompensationFilter

/-!
# Inverse-ensemble scores and actuator excitation

The score is the deployed dimension-normalized Gaussian residual score and
log-mean-exp aggregation, with the same fixed diagonal variance for all modes.
Prediction accuracy is a hypothesis on completed transitions, not a consequence
of fitting the ensemble or a forward-likelihood calibration assumption.
-/

noncomputable section

namespace BAPR.ActionCompensation

open Finset

def logMeanExp {Head : Type} [Fintype Head] (scores : Head → ℝ) : ℝ :=
  Real.log (∑ j, Real.exp (scores j)) - Real.log (Fintype.card Head)

theorem logMeanExp_mono {Head : Type} [Fintype Head] [Nonempty Head]
    (p q : Head → ℝ) (h : ∀ j, p j ≤ q j) : logMeanExp p ≤ logMeanExp q := by
  have hs := sum_le_sum (s := univ) fun j _ => Real.exp_le_exp.mpr (h j)
  have hl := Real.log_le_log (modeSoftmax_sum_pos p) hs
  unfold logMeanExp
  linarith

theorem logMeanExp_shift {Head : Type} [Fintype Head] [Nonempty Head]
    (p : Head → ℝ) (offset : ℝ) :
    logMeanExp (fun j => p j + offset) = logMeanExp p + offset := by
  simp only [logMeanExp, Real.exp_add, ← sum_mul]
  rw [Real.log_mul (ne_of_gt (modeSoftmax_sum_pos p))
    (ne_of_gt (Real.exp_pos offset)), Real.log_exp]
  ring

theorem logMeanExp_lipschitz {Head : Type} [Fintype Head] [Nonempty Head]
    (p q : Head → ℝ) (η : ℝ) (h : ∀ j, |p j - q j| ≤ η) :
    |logMeanExp p - logMeanExp q| ≤ η := by
  have hu := logMeanExp_mono p (fun j => q j + η)
    (fun j => by have hj := (abs_le.mp (h j)).2; linarith)
  have hl := logMeanExp_mono q (fun j => p j + η)
    (fun j => by have hj := (abs_le.mp (h j)).1; linarith)
  rw [logMeanExp_shift] at hu hl
  apply abs_le.mpr
  constructor <;> linarith

theorem logMeanExp_contrast {Head : Type} [Fintype Head] [Nonempty Head]
    (p q : Head → ℝ) (γ : ℝ) (h : ∀ j, γ ≤ p j - q j) :
    γ ≤ logMeanExp p - logMeanExp q := by
  have hl := logMeanExp_mono (fun j => q j + γ) p
    (fun j => by linarith [h j])
  rw [logMeanExp_shift] at hl
  linarith

def inverseHeadScore {d : ℕ} (candidate prediction variance : Action d) : ℝ :=
  -((∑ j, (candidate j - prediction j) ^ 2 / variance j) +
    ∑ j, Real.log (2 * Real.pi * variance j)) / (2 * d)

def inverseEvidence {d : ℕ} {Head Mode : Type} [Fintype Head]
    (candidate : Mode → Action d) (prediction : Head → Action d)
    (variance : Action d) (z : Mode) : ℝ :=
  logMeanExp (fun j => inverseHeadScore (candidate z) (prediction j) variance)

def inverseScoreBudget {d : ℕ} (ε variance : Action d) : ℝ :=
  2 * (∑ j, ε j / variance j) / d

theorem clippedAction_inBox {d : ℕ} (u : Action d) : inBox (clipAction u) := by
  intro j
  constructor
  · exact le_max_left _ _
  · exact max_le (by norm_num) (min_le_right _ _)

theorem tanhAction_inBox {d : ℕ} (u : Action d) : inBox (fun j => Real.tanh (u j)) := by
  intro j
  exact ⟨(Real.neg_one_lt_tanh _).le, (Real.tanh_lt_one _).le⟩

def effectiveInverseVariance {d : ℕ} (v : Action d) : Action d :=
  fun j => max (1 / (10 ^ 6) : ℝ) (min (v j) 1)

theorem effectiveInverseVariance_bounds {d : ℕ} (v : Action d) (j : Fin d) :
    (1 / (10 ^ 6) : ℝ) ≤ effectiveInverseVariance v j ∧
      effectiveInverseVariance v j ≤ 1 := by
  constructor
  · exact le_max_left _ _
  · exact max_le (by norm_num) (min_le_right _ _)

theorem effectiveInverseVariance_pos {d : ℕ} (v : Action d) (j : Fin d) :
    0 < effectiveInverseVariance v j := by
  exact lt_of_lt_of_le (by norm_num) (effectiveInverseVariance_bounds v j).1

theorem squared_residual_error (u x y ε : ℝ)
    (hu : -1 ≤ u ∧ u ≤ 1) (hx : -1 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 1)
    (herr : |x - y| ≤ ε) : |(u - x) ^ 2 - (u - y) ^ 2| ≤ 4 * ε := by
  have hf : |2 * u - x - y| ≤ 4 := by
    apply abs_le.mpr
    constructor <;> linarith
  have hid : (u - x) ^ 2 - (u - y) ^ 2 = (y - x) * (2 * u - x - y) := by ring
  rw [hid, abs_mul]
  have he : |y - x| ≤ ε := by simpa only [abs_sub_comm] using herr
  calc
    _ ≤ |y - x| * 4 := mul_le_mul_of_nonneg_left hf (abs_nonneg _)
    _ ≤ ε * 4 := mul_le_mul_of_nonneg_right he (by norm_num)
    _ = _ := by ring

/-- Common Gaussian normalizers cancel before bounding the prediction error. -/
theorem inverseHeadScore_prediction_error {d : ℕ} (hd : 0 < d)
    (u p q ε variance : Action d) (hu : inBox u) (hp : inBox p) (hq : inBox q)
    (hv : ∀ j, 0 < variance j) (herr : ∀ j, |p j - q j| ≤ ε j) :
    |inverseHeadScore u p variance - inverseHeadScore u q variance| ≤
      inverseScoreBudget ε variance := by
  have hdpos : 0 < (d : ℝ) := Nat.cast_pos.mpr hd
  have hid : inverseHeadScore u p variance - inverseHeadScore u q variance =
      -(∑ j, ((u j - p j) ^ 2 - (u j - q j) ^ 2) / variance j) / (2 * d) := by
    simp only [inverseHeadScore, sub_div, sum_sub_distrib]
    ring
  rw [hid, abs_div, abs_neg, abs_of_pos (by positivity : 0 < (2 * d : ℝ))]
  calc
    _ ≤ (∑ j, 4 * ε j / variance j) / (2 * d) := by
      apply div_le_div_of_nonneg_right _ (by positivity)
      apply (abs_sum_le_sum_abs _ _).trans
      apply sum_le_sum
      intro j _
      rw [abs_div, abs_of_pos (hv j)]
      exact div_le_div_of_nonneg_right
        (squared_residual_error (u j) (p j) (q j) (ε j)
          (hu j) (hp j) (hq j) (herr j)) (hv j).le
    _ = _ := by
      simp only [inverseScoreBudget, mul_div_assoc, ← mul_sum]
      ring

theorem inverseEvidence_prediction_error {d : ℕ} {Head Mode : Type}
    [Fintype Head] [Nonempty Head] (hd : 0 < d)
    (candidate : Mode → Action d) (p q : Head → Action d) (ε variance : Action d)
    (hu : ∀ z, inBox (candidate z)) (hp : ∀ j, inBox (p j)) (hq : ∀ j, inBox (q j))
    (hv : ∀ j, 0 < variance j) (herr : ∀ j i, |p j i - q j i| ≤ ε i) (z : Mode) :
    |inverseEvidence candidate p variance z - inverseEvidence candidate q variance z| ≤
      inverseScoreBudget ε variance := by
  exact logMeanExp_lipschitz _ _ _ (fun j => inverseHeadScore_prediction_error hd
    (candidate z) (p j) (q j) ε variance (hu z) (hp j) (hq j) hv (herr j))

theorem inverseScoreBudget_uniform {d : ℕ} (hd : 0 < d)
    (ε variance : Action d) (e vmin : ℝ) (he : 0 ≤ e) (hmin : 0 < vmin)
    (hε : ∀ j, 0 ≤ ε j ∧ ε j ≤ e) (hv : ∀ j, vmin ≤ variance j) :
    inverseScoreBudget ε variance ≤ 2 * e / vmin := by
  have hdpos : 0 < (d : ℝ) := Nat.cast_pos.mpr hd
  have hsum : (∑ j, ε j / variance j) ≤ d * (e / vmin) := by
    calc
      _ ≤ ∑ j : Fin d, e / vmin := sum_le_sum fun j _ => by
        calc
          _ ≤ e / variance j := div_le_div_of_nonneg_right (hε j).2 (hmin.le.trans (hv j))
          _ ≤ e / vmin := div_le_div_of_nonneg_left he hmin (hv j)
      _ = _ := by simp
  unfold inverseScoreBudget
  apply (div_le_iff₀ hdpos).mpr
  have h := mul_le_mul_of_nonneg_left hsum (by norm_num : (0 : ℝ) ≤ 2)
  exact h.trans_eq (by ring)

def inverseExcitationMargin {d : ℕ} (truth competitor ε variance : Action d) : ℝ :=
  (∑ j, ((competitor j - truth j) ^ 2 -
    2 * ε j * |competitor j - truth j|) / variance j) / (2 * d)

theorem squared_residual_contrast (u w p ε : ℝ) (herr : |p - u| ≤ ε) :
    (w - u) ^ 2 - 2 * ε * |w - u| ≤ (w - p) ^ 2 - (u - p) ^ 2 := by
  have hprod : (w - u) * (p - u) ≤ |w - u| * ε := by
    calc
      _ ≤ |(w - u) * (p - u)| := le_abs_self _
      _ = |w - u| * |p - u| := abs_mul _ _
      _ ≤ _ := mul_le_mul_of_nonneg_left herr (abs_nonneg _)
  nlinarith

theorem inverseHeadScore_signal_margin {d : ℕ} (hd : 0 < d)
    (truth competitor prediction ε variance : Action d)
    (hv : ∀ j, 0 < variance j) (herr : ∀ j, |prediction j - truth j| ≤ ε j) :
    inverseExcitationMargin truth competitor ε variance ≤
      inverseHeadScore truth prediction variance - inverseHeadScore competitor prediction variance := by
  have hdpos : 0 < (d : ℝ) := Nat.cast_pos.mpr hd
  have hid : inverseHeadScore truth prediction variance -
      inverseHeadScore competitor prediction variance =
      (∑ j, ((competitor j - prediction j) ^ 2 -
        (truth j - prediction j) ^ 2) / variance j) / (2 * d) := by
    simp only [inverseHeadScore, sub_div, sum_sub_distrib]
    ring
  rw [hid]
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact sum_le_sum fun j _ => div_le_div_of_nonneg_right
    (squared_residual_contrast (truth j) (competitor j) (prediction j) (ε j) (herr j))
    (hv j).le

/-- The mixture preserves a common contrast bound for every ensemble head. -/
theorem inverseEvidence_signal_margin {d : ℕ} {Head Mode : Type}
    [Fintype Head] [Nonempty Head] (hd : 0 < d)
    (candidate : Mode → Action d) (prediction : Head → Action d) (ε variance : Action d)
    (hv : ∀ j, 0 < variance j) (z w : Mode)
    (herr : ∀ j i, |prediction j i - candidate z i| ≤ ε i) :
    inverseExcitationMargin (candidate z) (candidate w) ε variance ≤
      inverseEvidence candidate prediction variance z -
        inverseEvidence candidate prediction variance w := by
  exact logMeanExp_contrast _ _ _ fun j => inverseHeadScore_signal_margin hd
    (candidate z) (candidate w) (prediction j) ε variance hv (herr j)

theorem inverseExcitationMargin_same {d : ℕ} (u ε variance : Action d) :
    inverseExcitationMargin u u ε variance = 0 := by
  simp [inverseExcitationMargin]

theorem inverseEvidence_same_candidates {d : ℕ} {Head Mode : Type} [Fintype Head]
    (candidate : Mode → Action d) (prediction : Head → Action d)
    (variance : Action d) (z w : Mode) (h : candidate z = candidate w) :
    inverseEvidence candidate prediction variance z =
      inverseEvidence candidate prediction variance w := by
  simp only [inverseEvidence, h]

theorem polarity_gain_inBox {d : ℕ} (g command : Action d)
    (hg : IsPolarity g) (hc : inBox command) : inBox (applyGain g command) := by
  intro j
  rcases hg j with h | h
  · simpa [applyGain, h] using hc j
  · simp only [applyGain, h, neg_one_mul]
    constructor <;> linarith [(hc j).1, (hc j).2]

theorem polarity_candidate_unclipped {d : ℕ} (g command : Action d)
    (hg : IsPolarity g) (hc : inBox command) :
    clipAction (applyGain g command) = applyGain g command := by
  exact (clipAction_self_iff _).mpr (polarity_gain_inBox g command hg hc)

theorem polarity_coordinate_separation (g h a ε : ℝ)
    (hg : g = 1 ∨ g = -1) (hh : h = 1 ∨ h = -1) :
    (h * a - g * a) ^ 2 - 2 * ε * |h * a - g * a| =
      4 * (if g = h then 0 else a ^ 2 - ε * |a|) := by
  rcases hg with rfl | rfl <;> rcases hh with rfl | rfl
  · simp
  · have heq : -1 * a - 1 * a = -2 * a := by ring
    rw [heq, abs_mul]
    norm_num
    ring
  · have heq : 1 * a - -1 * a = 2 * a := by ring
    rw [heq, abs_mul]
    norm_num
    ring
  · simp

/-- Only excited coordinates on which the two polarity modes differ contribute. -/
theorem inverseExcitationMargin_polarity {d : ℕ} (g h command ε variance : Action d)
    (hg : IsPolarity g) (hh : IsPolarity h) :
    inverseExcitationMargin (applyGain g command) (applyGain h command) ε variance =
      (2 / (d : ℝ)) * ∑ j, if g j = h j then 0 else
        ((command j) ^ 2 - ε j * |command j|) / variance j := by
  have hs : (∑ j, ((applyGain h command j - applyGain g command j) ^ 2 -
      2 * ε j * |applyGain h command j - applyGain g command j|) / variance j) =
      4 * ∑ j, if g j = h j then 0 else
        ((command j) ^ 2 - ε j * |command j|) / variance j := by
    rw [mul_sum]
    apply sum_congr rfl
    intro j _
    rw [show applyGain h command j = h j * command j from rfl,
      show applyGain g command j = g j * command j from rfl,
      polarity_coordinate_separation (g j) (h j) (command j) (ε j) (hg j) (hh j)]
    split_ifs <;> ring
  rw [inverseExcitationMargin, hs]
  ring

theorem inverse_filter_prediction_error {d : ℕ} {Head Mode : Type}
    [Fintype Head] [Nonempty Head] [Fintype Mode] [Nonempty Mode] [DecidableEq Mode]
    (hd : 0 < d) (a k δ c D η : ℝ) (p b : Mode → ℝ)
    (candidate : ℕ → Mode → Action d) (reference actual : ℕ → Head → Action d)
    (ε variance : ℕ → Action d)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c)
    (hp : IsModeMass p) (hb : IsModeMass b) (hpp : ∀ z, 0 < p z) (hbp : ∀ z, 0 < b z)
    (hinit : LogOddsClose p b D)
    (hu : ∀ t z, inBox (candidate t z))
    (href : ∀ t j, inBox (reference t j)) (hact : ∀ t j, inBox (actual t j))
    (hv : ∀ t i, 0 < variance t i)
    (herr : ∀ t j i, |reference t j i - actual t j i| ≤ ε t i)
    (hbudget : ∀ t, inverseScoreBudget (ε t) (variance t) ≤ η) (t : ℕ) :
    LogOddsClose
      (temperedFilterPath a k δ c p
        (fun s => inverseEvidence (candidate s) (reference s) (variance s)) t)
      (temperedFilterPath a k δ c b
        (fun s => inverseEvidence (candidate s) (actual s) (variance s)) t)
      (filterErrorEnvelope δ c D η t) := by
  apply temperedFilter_error_envelope a k δ c D η p b _ _
    ha hk hδ hc hp hb hpp hbp hinit _ t
  intro s z
  exact (inverseEvidence_prediction_error hd (candidate s) (reference s) (actual s)
    (ε s) (variance s) (hu s) (href s) (hact s) (hv s) (herr s) z).trans (hbudget s)

/-- Prediction accuracy and actuator excitation supply the filter's separation hypothesis. -/
theorem inverse_filter_switch_recovery {d : ℕ} {Head Mode : Type}
    [Fintype Head] [Nonempty Head] [Fintype Mode] [Nonempty Mode]
    (hd : 0 < d) (a k δ c m : ℝ) (initial : Mode → ℝ)
    (candidate : ℕ → Mode → Action d) (prediction : ℕ → Head → Action d)
    (ε variance : ℕ → Action d)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 < c) (hm : 0 < m)
    (hb : IsModeMass initial) (z : Mode) (n : ℕ)
    (hv : ∀ t, t < n + 1 → ∀ i, 0 < variance t i)
    (herr : ∀ t, t < n + 1 → ∀ j i, |prediction t j i - candidate t z i| ≤ ε t i)
    (hsep : ∀ t, t < n + 1 → ∀ w, w ≠ z → m ≤
      inverseExcitationMargin (candidate t z) (candidate t w) (ε t) (variance t))
    (hcert : 0 < filterMarginEnvelope δ (c * m)
      (Real.log (a + k) - Real.log k) (n + 1)) :
    ∀ w, w ≠ z →
      temperedFilterPath a k δ c initial
        (fun t => inverseEvidence (candidate t) (prediction t) (variance t)) (n + 1) w <
      temperedFilterPath a k δ c initial
        (fun t => inverseEvidence (candidate t) (prediction t) (variance t)) (n + 1) z := by
  apply temperedFilter_switch_recovery a k δ c m initial _ ha hk hδ hc hm hb z n _ hcert
  intro t ht w hw
  exact (hsep t ht w hw).trans (inverseEvidence_signal_margin hd
    (candidate t) (prediction t) (ε t) (variance t) (hv t ht) z w (herr t ht))

end BAPR.ActionCompensation
