import ActionCompensationBelief

/-!
# Stability and conditional recovery of the deployed tempered sticky filter

The comparison filter has the same forgetting and hazard, not an exact Bayes
posterior. Both filters see the same completed-transition sequence. Evidence
accuracy and persistent score separation are explicit hypotheses.
-/

noncomputable section

namespace BAPR.ActionCompensation

open Finset

def modeLogOdds {Mode : Type} (b : Mode → ℝ) (z w : Mode) : ℝ :=
  Real.log (b z) - Real.log (b w)

def LogOddsClose {Mode : Type} (p b : Mode → ℝ) (D : ℝ) : Prop :=
  ∀ z w, |modeLogOdds p z w - modeLogOdds b z w| ≤ D

def OddsRatioBound {Mode : Type} (p b : Mode → ℝ) (D : ℝ) : Prop :=
  ∀ z w, p z * b w ≤ Real.exp D * (b z * p w)

def modeSoftmax {Mode : Type} [Fintype Mode] (scores : Mode → ℝ) (z : Mode) : ℝ :=
  Real.exp (scores z) / ∑ w, Real.exp (scores w)

theorem modeSoftmax_sum_pos {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (scores : Mode → ℝ) : 0 < ∑ z, Real.exp (scores z) := by
  exact sum_pos (fun z _ => Real.exp_pos _) univ_nonempty

theorem modeSoftmax_pos {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (scores : Mode → ℝ) (z : Mode) : 0 < modeSoftmax scores z := by
  exact div_pos (Real.exp_pos _) (modeSoftmax_sum_pos scores)

theorem modeSoftmax_isModeMass {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (scores : Mode → ℝ) : IsModeMass (modeSoftmax scores) := by
  constructor
  · exact fun z => (modeSoftmax_pos scores z).le
  · simp only [modeSoftmax, ← sum_div]
    exact div_self (ne_of_gt (modeSoftmax_sum_pos scores))

theorem modeSoftmax_log_odds {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (scores : Mode → ℝ) (z w : Mode) :
    modeLogOdds (modeSoftmax scores) z w = scores z - scores w := by
  simp only [modeLogOdds, modeSoftmax,
    Real.log_div (ne_of_gt (Real.exp_pos _))
      (ne_of_gt (modeSoftmax_sum_pos scores)), Real.log_exp]
  ring

theorem modeSoftmax_shift {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (scores : Mode → ℝ) (offset : ℝ) :
    modeSoftmax (fun z => scores z + offset) = modeSoftmax scores := by
  funext z
  simp only [modeSoftmax, Real.exp_add, ← sum_mul]
  field_simp

theorem logOddsClose_iff_ratio {Mode : Type} (p b : Mode → ℝ) (D : ℝ)
    (hp : ∀ z, 0 < p z) (hb : ∀ z, 0 < b z) :
    LogOddsClose p b D ↔ OddsRatioBound p b D := by
  have upper (z w : Mode) :
      modeLogOdds p z w - modeLogOdds b z w ≤ D ↔
        p z * b w ≤ Real.exp D * (b z * p w) := by
    have hlog : modeLogOdds p z w - modeLogOdds b z w ≤ D ↔
        Real.log (p z * b w) ≤ D + Real.log (b z * p w) := by
      rw [Real.log_mul (ne_of_gt (hp z)) (ne_of_gt (hb w)),
        Real.log_mul (ne_of_gt (hb z)) (ne_of_gt (hp w))]
      unfold modeLogOdds
      constructor <;> intro h <;> linarith
    rw [hlog, ← Real.exp_le_exp, Real.exp_add,
      Real.exp_log (mul_pos (hp z) (hb w)),
      Real.exp_log (mul_pos (hb z) (hp w))]
  constructor
  · intro h z w
    exact (upper z w).mp ((abs_le.mp (h z w)).2)
  · intro h z w
    apply abs_le.mpr
    constructor
    · have hw := (upper w z).mpr (h w z)
      unfold modeLogOdds at *
      linarith
    · exact (upper z w).mpr (h z w)

def modeLinear {Mode : Type} [Fintype Mode]
    (A : Mode → Mode → ℝ) (b : Mode → ℝ) (z : Mode) : ℝ :=
  ∑ w, A z w * b w

/-- A common nonnegative mixing matrix cannot enlarge multiplicative odds spread. -/
theorem modeLinear_ratio_bound {Mode : Type} [Fintype Mode]
    (A : Mode → Mode → ℝ) (p b : Mode → ℝ) (D : ℝ)
    (hA : ∀ z w, 0 ≤ A z w) (h : OddsRatioBound p b D) :
    OddsRatioBound (modeLinear A p) (modeLinear A b) D := by
  intro z w
  have hid (v u : Mode → ℝ) : modeLinear A v z * modeLinear A u w =
      ∑ i, ∑ j, (A z i * A w j) * (v i * u j) := by
    unfold modeLinear
    rw [sum_mul]
    simp only [mul_sum]
    apply sum_congr rfl
    intro i _
    apply sum_congr rfl
    intro j _
    ring
  calc
    _ = ∑ i, ∑ j, (A z i * A w j) * (p i * b j) := hid p b
    _ ≤ ∑ i, ∑ j, (A z i * A w j) * (Real.exp D * (b i * p j)) := by
      apply sum_le_sum
      intro i _
      apply sum_le_sum
      intro j _
      exact mul_le_mul_of_nonneg_left (h i j) (mul_nonneg (hA z i) (hA w j))
    _ = _ := by
      rw [hid b p, mul_sum]
      apply sum_congr rfl
      intro i _
      rw [mul_sum]
      apply sum_congr rfl
      intro j _
      ring

def stickyPrior {Mode : Type} (a k : ℝ) (b : Mode → ℝ) (z : Mode) : ℝ :=
  a * b z + k

theorem stickyPrior_pos {Mode : Type} [Fintype Mode] (a k : ℝ) (b : Mode → ℝ)
    (ha : 0 ≤ a) (hk : 0 < k) (hb : IsModeMass b) (z : Mode) :
    0 < stickyPrior a k b z := by
  unfold stickyPrior
  exact add_pos_of_nonneg_of_pos (mul_nonneg ha (hb.1 z)) hk

theorem stickyPrior_isModeMass {Mode : Type} [Fintype Mode]
    (a k : ℝ) (b : Mode → ℝ) (ha : 0 ≤ a) (hk : 0 ≤ k)
    (hnorm : a + Fintype.card Mode * k = 1) (hb : IsModeMass b) :
    IsModeMass (stickyPrior a k b) := by
  constructor
  · intro z
    exact add_nonneg (mul_nonneg ha (hb.1 z)) hk
  · simpa only [stickyPrior, sum_add_distrib, ← mul_sum, hb.2, mul_one,
      sum_const, card_univ, nsmul_eq_mul] using hnorm

theorem stickyPrior_log_stability {Mode : Type} [Fintype Mode] [DecidableEq Mode]
    (a k : ℝ) (p b : Mode → ℝ) (D : ℝ) (ha : 0 ≤ a) (hk : 0 < k)
    (hp : IsModeMass p) (hb : IsModeMass b)
    (hpp : ∀ z, 0 < p z) (hbp : ∀ z, 0 < b z) (h : LogOddsClose p b D) :
    LogOddsClose (stickyPrior a k p) (stickyPrior a k b) D := by
  let A : Mode → Mode → ℝ := fun z w => (if z = w then a else 0) + k
  have hA (z w : Mode) : 0 ≤ A z w := by
    dsimp [A]
    split_ifs <;> linarith
  have hid (v : Mode → ℝ) (hv : IsModeMass v) :
      modeLinear A v = stickyPrior a k v := by
    funext z
    simp only [modeLinear, A, add_mul, sum_add_distrib, ite_mul,
      zero_mul, sum_ite_eq, mem_univ, if_true, ← mul_sum, hv.2, mul_one,
      stickyPrior]
  apply (logOddsClose_iff_ratio _ _ D
    (stickyPrior_pos a k p ha hk hp) (stickyPrior_pos a k b ha hk hb)).mpr
  rw [← hid p hp, ← hid b hb]
  exact modeLinear_ratio_bound A p b D hA
    ((logOddsClose_iff_ratio p b D hpp hbp).mp h)

def temperedFilterStep {Mode : Type} [Fintype Mode]
    (a k δ c : ℝ) (b evidence : Mode → ℝ) : Mode → ℝ :=
  modeSoftmax (fun z => δ * Real.log (stickyPrior a k b z) + c * evidence z)

theorem temperedFilter_centered_evidence {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c offset : ℝ) (b evidence : Mode → ℝ) :
    temperedFilterStep a k δ c b (fun z => evidence z - offset) =
      temperedFilterStep a k δ c b evidence := by
  unfold temperedFilterStep
  have h : (fun z => δ * Real.log (stickyPrior a k b z) + c * (evidence z - offset)) =
      (fun z => (δ * Real.log (stickyPrior a k b z) + c * evidence z) + -(c * offset)) := by
    funext z
    ring
  rw [h, modeSoftmax_shift]

/-- The frozen four-mode hazard has an inactive numerical prior clamp in real arithmetic. -/
theorem frozenSticky_hazard_formula (x : ℝ) :
    (1 - (1 / 500 : ℝ)) * x + ((1 / 500 : ℝ) / 3) * (1 - x) =
      (374 / 375 : ℝ) * x + 1 / 1500 := by
  ring

theorem frozenSticky_prior_bounds {Mode : Type} [Fintype Mode]
    (b : Mode → ℝ) (hb : IsModeMass b) (z : Mode) :
    (1 / 1500 : ℝ) ≤ stickyPrior (374 / 375) (1 / 1500) b z ∧
      stickyPrior (374 / 375) (1 / 1500) b z ≤ 499 / 500 := by
  have hz : b z ≤ 1 := (single_le_sum (fun i _ => hb.1 i) (mem_univ z)).trans_eq hb.2
  unfold stickyPrior
  constructor <;> nlinarith [hb.1 z]

theorem frozenSticky_prior_clip {Mode : Type} [Fintype Mode]
    (b : Mode → ℝ) (hb : IsModeMass b) (z : Mode) :
    min 1 (max (1 / (10 ^ 12) : ℝ) (stickyPrior (374 / 375) (1 / 1500) b z)) =
      stickyPrior (374 / 375) (1 / 1500) b z := by
  have h := frozenSticky_prior_bounds b hb z
  rw [max_eq_right (by linarith [h.1] :
    (1 / (10 ^ 12) : ℝ) ≤ stickyPrior (374 / 375) (1 / 1500) b z),
    min_eq_right (by linarith [h.2])]

theorem temperedFilter_log_odds {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c : ℝ) (b evidence : Mode → ℝ) (z w : Mode) :
    modeLogOdds (temperedFilterStep a k δ c b evidence) z w =
      δ * modeLogOdds (stickyPrior a k b) z w + c * (evidence z - evidence w) := by
  rw [temperedFilterStep, modeSoftmax_log_odds]
  unfold modeLogOdds
  ring

/-- The factor two is the error in a pairwise evidence difference. -/
theorem temperedFilter_score_stability {Mode : Type} [Fintype Mode]
    [Nonempty Mode] [DecidableEq Mode]
    (a k δ c : ℝ) (p b ideal actual : Mode → ℝ) (D η : ℝ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c)
    (hp : IsModeMass p) (hb : IsModeMass b)
    (hpp : ∀ z, 0 < p z) (hbp : ∀ z, 0 < b z)
    (h : LogOddsClose p b D) (herr : ∀ z, |ideal z - actual z| ≤ η) :
    LogOddsClose (temperedFilterStep a k δ c p ideal)
      (temperedFilterStep a k δ c b actual) (δ * D + 2 * c * η) := by
  intro z w
  rw [temperedFilter_log_odds, temperedFilter_log_odds]
  have hprior := (abs_le.mp (stickyPrior_log_stability a k p b D
    ha hk hp hb hpp hbp h z w))
  have hz := abs_le.mp (herr z)
  have hw := abs_le.mp (herr w)
  have hlo := mul_le_mul_of_nonneg_left hprior.1 hδ
  have hhi := mul_le_mul_of_nonneg_left hprior.2 hδ
  have hel : -2 * η ≤ (ideal z - ideal w) - (actual z - actual w) := by linarith
  have heu : (ideal z - ideal w) - (actual z - actual w) ≤ 2 * η := by linarith
  have hcl := mul_le_mul_of_nonneg_left hel hc
  have hcu := mul_le_mul_of_nonneg_left heu hc
  apply abs_le.mpr
  constructor <;> nlinarith

def temperedFilterPath {Mode : Type} [Fintype Mode]
    (a k δ c : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ) : ℕ → Mode → ℝ
  | 0 => initial
  | t + 1 => temperedFilterStep a k δ c (temperedFilterPath a k δ c initial evidence t)
      (evidence t)

theorem temperedFilterPath_mass {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ)
    (h : IsModeMass initial) (t : ℕ) :
    IsModeMass (temperedFilterPath a k δ c initial evidence t) := by
  cases t with
  | zero => exact h
  | succ t => exact modeSoftmax_isModeMass _

theorem temperedFilterPath_pos {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ)
    (h : ∀ z, 0 < initial z) (t : ℕ) (z : Mode) :
    0 < temperedFilterPath a k δ c initial evidence t z := by
  cases t with
  | zero => exact h z
  | succ t =>
    simp only [temperedFilterPath, temperedFilterStep]
    exact modeSoftmax_pos _ z

/-- A uniform score-error floor for forgetting 0 <= delta < 1. -/
theorem temperedFilter_uniform_error {Mode : Type} [Fintype Mode]
    [Nonempty Mode] [DecidableEq Mode]
    (a k δ c η R : ℝ) (p b : Mode → ℝ) (ideal actual : ℕ → Mode → ℝ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c)
    (hp : IsModeMass p) (hb : IsModeMass b)
    (hpp : ∀ z, 0 < p z) (hbp : ∀ z, 0 < b z)
    (hinit : LogOddsClose p b R) (hfloor : δ * R + 2 * c * η ≤ R)
    (herr : ∀ t z, |ideal t z - actual t z| ≤ η) (t : ℕ) :
    LogOddsClose (temperedFilterPath a k δ c p ideal t)
      (temperedFilterPath a k δ c b actual t) R := by
  induction t with
  | zero => exact hinit
  | succ t ih =>
    have hs := temperedFilter_score_stability a k δ c
      _ _ (ideal t) (actual t) R η ha hk hδ hc
      (temperedFilterPath_mass a k δ c p ideal hp t)
      (temperedFilterPath_mass a k δ c b actual hb t)
      (temperedFilterPath_pos a k δ c p ideal hpp t)
      (temperedFilterPath_pos a k δ c b actual hbp t) ih (herr t)
    exact fun z w => (hs z w).trans hfloor

def filterErrorEnvelope (δ c D η : ℝ) (t : ℕ) : ℝ :=
  δ ^ t * D + 2 * c * η * ∑ i ∈ range t, δ ^ i

theorem filterErrorEnvelope_succ (δ c D η : ℝ) (t : ℕ) :
    filterErrorEnvelope δ c D η (t + 1) =
      δ * filterErrorEnvelope δ c D η t + 2 * c * η := by
  simp only [filterErrorEnvelope, pow_succ, geom_sum_succ]
  ring

theorem temperedFilter_error_envelope {Mode : Type} [Fintype Mode]
    [Nonempty Mode] [DecidableEq Mode]
    (a k δ c D η : ℝ) (p b : Mode → ℝ) (ideal actual : ℕ → Mode → ℝ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c)
    (hp : IsModeMass p) (hb : IsModeMass b)
    (hpp : ∀ z, 0 < p z) (hbp : ∀ z, 0 < b z)
    (hinit : LogOddsClose p b D) (herr : ∀ t z, |ideal t z - actual t z| ≤ η)
    (t : ℕ) :
    LogOddsClose (temperedFilterPath a k δ c p ideal t)
      (temperedFilterPath a k δ c b actual t) (filterErrorEnvelope δ c D η t) := by
  induction t with
  | zero => simpa [filterErrorEnvelope, temperedFilterPath] using hinit
  | succ t ih =>
    rw [filterErrorEnvelope_succ]
    exact temperedFilter_score_stability a k δ c _ _ (ideal t) (actual t)
      _ η ha hk hδ hc
      (temperedFilterPath_mass a k δ c p ideal hp t)
      (temperedFilterPath_mass a k δ c b actual hb t)
      (temperedFilterPath_pos a k δ c p ideal hpp t)
      (temperedFilterPath_pos a k δ c b actual hbp t) ih (herr t)

theorem temperedFilter_error_floor {Mode : Type} [Fintype Mode]
    [Nonempty Mode] [DecidableEq Mode]
    (a k δ c η : ℝ) (p b : Mode → ℝ) (ideal actual : ℕ → Mode → ℝ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hδ1 : δ < 1) (hc : 0 ≤ c)
    (hp : IsModeMass p) (hb : IsModeMass b)
    (hpp : ∀ z, 0 < p z) (hbp : ∀ z, 0 < b z)
    (hinit : LogOddsClose p b (2 * c * η / (1 - δ)))
    (herr : ∀ t z, |ideal t z - actual t z| ≤ η) (t : ℕ) :
    LogOddsClose (temperedFilterPath a k δ c p ideal t)
      (temperedFilterPath a k δ c b actual t) (2 * c * η / (1 - δ)) := by
  apply temperedFilter_uniform_error a k δ c η _ p b ideal actual
    ha hk hδ hc hp hb hpp hbp hinit _ herr t
  have hd : 1 - δ ≠ 0 := ne_of_gt (by linarith)
  apply le_of_eq
  field_simp [hd]
  ring

theorem modeTotalVariation_triangle {Mode : Type} [Fintype Mode]
    (p b v : Mode → ℝ) :
    modeTotalVariation p b ≤ modeTotalVariation p v + modeTotalVariation v b := by
  unfold modeTotalVariation
  have h := sum_le_sum (s := univ) (fun z _ =>
    abs_sub_le (p z) (v z) (b z))
  rw [sum_add_distrib] at h
  linarith

/-- Converts log-odds spread into a conservative TV bound on normalized masses. -/
theorem logOddsClose_totalVariation {Mode : Type} [Fintype Mode]
    (p b : Mode → ℝ) (D : ℝ) (hp : IsModeMass p) (hb : IsModeMass b)
    (hpp : ∀ z, 0 < p z) (hbp : ∀ z, 0 < b z) (h : LogOddsClose p b D) :
    modeTotalVariation p b ≤ Real.exp D - 1 := by
  have hr := (logOddsClose_iff_ratio p b D hpp hbp).mp h
  have bound (z : Mode) : p z ≤ Real.exp D * b z := by
    have hs := sum_le_sum (s := univ) (fun w _ => hr z w)
    simpa only [← mul_sum, hb.2, hp.2, mul_one, mul_assoc] using hs
  have reverse (z : Mode) : b z ≤ Real.exp D * p z := by
    have hs := sum_le_sum (s := univ) (fun w _ => hr w z)
    have hid : (∑ w, Real.exp D * (b w * p z)) = Real.exp D * p z := by
      rw [← mul_sum, ← sum_mul, hb.2, one_mul]
    simpa only [← sum_mul, hp.2, one_mul, hid] using hs
  have hD : 1 ≤ Real.exp D := by
    have hs := sum_le_sum (s := univ) (fun z _ => bound z)
    simpa only [← mul_sum, hp.2, hb.2, mul_one] using hs
  have he (z : Mode) : |p z - b z| ≤ (Real.exp D - 1) * (p z + b z) := by
    apply abs_le.mpr
    have hn1 := mul_nonneg (sub_nonneg.mpr hD) (hp.1 z)
    have hn2 := mul_nonneg (sub_nonneg.mpr hD) (hb.1 z)
    constructor <;> nlinarith [bound z, reverse z]
  have hs := sum_le_sum (s := univ) (fun z _ => he z)
  rw [← mul_sum, sum_add_distrib, hp.2, hb.2] at hs
  unfold modeTotalVariation
  linarith

/-- Reference-filter bias is retained; score robustness does not imply calibration. -/
theorem calibration_of_reference_filter {Mode : Type} [Fintype Mode]
    (truth reference actual : Mode → ℝ) (D : ℝ)
    (href : IsModeMass reference) (hact : IsModeMass actual)
    (hrp : ∀ z, 0 < reference z) (hap : ∀ z, 0 < actual z)
    (h : LogOddsClose reference actual D) :
    modeTotalVariation truth actual ≤ modeTotalVariation truth reference +
      (Real.exp D - 1) := by
  exact (modeTotalVariation_triangle truth actual reference).trans
    (add_le_add (le_refl _) (logOddsClose_totalVariation reference actual D href hact hrp hap h))

/-- Positive common mixing moves each pair's odds towards one. -/
theorem stickyPrior_log_odds_lower {Mode : Type} [Fintype Mode]
    (a k : ℝ) (b : Mode → ℝ) (ha : 0 ≤ a) (hk : 0 < k)
    (hb : IsModeMass b) (hbp : ∀ z, 0 < b z) (z w : Mode) :
    min (modeLogOdds b z w) 0 ≤ modeLogOdds (stickyPrior a k b) z w := by
  have hpz := stickyPrior_pos a k b ha hk hb z
  have hpw := stickyPrior_pos a k b ha hk hb w
  by_cases hzw : b z ≤ b w
  · have hmul : b z * stickyPrior a k b w ≤ stickyPrior a k b z * b w := by
      have h := mul_le_mul_of_nonneg_left hzw hk.le
      unfold stickyPrior
      nlinarith
    have hl := Real.log_le_log (mul_pos (hbp z) hpw) hmul
    rw [Real.log_mul (ne_of_gt (hbp z)) (ne_of_gt hpw),
      Real.log_mul (ne_of_gt hpz) (ne_of_gt (hbp w))] at hl
    have hmin := min_le_left (modeLogOdds b z w) 0
    unfold modeLogOdds at *
    linarith
  · have hprior : stickyPrior a k b w ≤ stickyPrior a k b z := by
      unfold stickyPrior
      exact add_le_add (mul_le_mul_of_nonneg_left (le_of_not_ge hzw) ha) le_rfl
    have hl := Real.log_le_log hpw hprior
    have hmin := min_le_right (modeLogOdds b z w) 0
    unfold modeLogOdds at *
    linarith

theorem stickyPrior_log_odds_floor {Mode : Type} [Fintype Mode]
    (a k : ℝ) (b : Mode → ℝ) (ha : 0 ≤ a) (hk : 0 < k)
    (hb : IsModeMass b) (z w : Mode) :
    Real.log k - Real.log (a + k) ≤ modeLogOdds (stickyPrior a k b) z w := by
  have hw : b w ≤ 1 := (single_le_sum (fun i _ => hb.1 i) (mem_univ w)).trans_eq hb.2
  have hl : k ≤ stickyPrior a k b z := by
    unfold stickyPrior
    nlinarith [mul_nonneg ha (hb.1 z)]
  have hu : stickyPrior a k b w ≤ a + k := by
    unfold stickyPrior
    nlinarith [mul_le_mul_of_nonneg_left hw ha]
  have h1 := Real.log_le_log hk hl
  have h2 := Real.log_le_log (stickyPrior_pos a k b ha hk hb w) hu
  unfold modeLogOdds
  linarith

def filterMarginEnvelope (δ s B : ℝ) (t : ℕ) : ℝ :=
  -(δ ^ t) * B + s * ∑ i ∈ range t, δ ^ i

theorem filterMarginEnvelope_succ (δ s B : ℝ) (t : ℕ) :
    filterMarginEnvelope δ s B (t + 1) = δ * filterMarginEnvelope δ s B t + s := by
  simp only [filterMarginEnvelope, pow_succ, geom_sum_succ]
  ring

theorem filter_margin_lower_step (δ s x E : ℝ) (hδ : 0 ≤ δ) (hs : 0 ≤ s)
    (h : min s E ≤ x) : min s (δ * E + s) ≤ δ * min x 0 + s := by
  by_cases hE : E ≤ 0
  · have hEx : E ≤ x := by simpa [min_eq_right (hE.trans hs)] using h
    have hm : E ≤ min x 0 := le_min hEx hE
    exact (min_le_right s _).trans (add_le_add
      (mul_le_mul_of_nonneg_left hm hδ) le_rfl)
  · have hx : 0 ≤ x := (le_min hs (le_of_not_ge hE)).trans h
    simp [min_eq_right hx]

/-- A finite-dwell recovery certificate under persistent pairwise evidence margins. -/
theorem temperedFilter_margin_envelope {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c m B : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c) (hm : 0 ≤ m)
    (hb : IsModeMass initial) (hbp : ∀ z, 0 < initial z)
    (z w : Mode) (hinit : -B ≤ modeLogOdds initial z w) (n : ℕ)
    (hgap : ∀ t, t < n → m ≤ evidence t z - evidence t w) :
    min (c * m) (filterMarginEnvelope δ (c * m) B n) ≤
      modeLogOdds (temperedFilterPath a k δ c initial evidence n) z w := by
  suffices h : ∀ t, t ≤ n → min (c * m) (filterMarginEnvelope δ (c * m) B t) ≤
      modeLogOdds (temperedFilterPath a k δ c initial evidence t) z w from h n le_rfl
  intro t
  induction t with
  | zero =>
    intro _
    simpa [filterMarginEnvelope, temperedFilterPath] using
      (min_le_right (c * m) (-B)).trans hinit
  | succ t ih =>
    intro ht
    have hi := ih (Nat.le_of_succ_le ht)
    have hs := stickyPrior_log_odds_lower a k _ ha hk
      (temperedFilterPath_mass a k δ c initial evidence hb t)
      (temperedFilterPath_pos a k δ c initial evidence hbp t) z w
    have he := mul_le_mul_of_nonneg_left (hgap t (Nat.lt_of_succ_le ht)) hc
    have hp := mul_le_mul_of_nonneg_left hs hδ
    rw [filterMarginEnvelope_succ, temperedFilterPath, temperedFilter_log_odds]
    exact (filter_margin_lower_step δ (c * m) _ _ hδ (mul_nonneg hc hm) hi).trans
      (add_le_add hp he)

theorem temperedFilter_mode_recovery {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c m B : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 < c) (hm : 0 < m)
    (hb : IsModeMass initial) (hbp : ∀ z, 0 < initial z) (z : Mode)
    (hinit : ∀ w, w ≠ z → -B ≤ modeLogOdds initial z w) (n : ℕ)
    (hgap : ∀ t, t < n → ∀ w, w ≠ z → m ≤ evidence t z - evidence t w)
    (hcert : 0 < filterMarginEnvelope δ (c * m) B n) :
    ∀ w, w ≠ z → temperedFilterPath a k δ c initial evidence n w <
      temperedFilterPath a k δ c initial evidence n z := by
  intro w hw
  have h := temperedFilter_margin_envelope a k δ c m B initial evidence
    ha hk hδ hc.le hm.le hb hbp z w (hinit w hw) n (fun t ht => hgap t ht w hw)
  have hl : 0 < modeLogOdds (temperedFilterPath a k δ c initial evidence n) z w :=
    (lt_min (mul_pos hc hm) hcert).trans_le h
  apply (Real.log_lt_log_iff
    (temperedFilterPath_pos a k δ c initial evidence hbp n w)
    (temperedFilterPath_pos a k δ c initial evidence hbp n z)).mp
  unfold modeLogOdds at hl
  linarith

/-- An inverse-score approximation preserves the reference separation only if gamma > 2 eta. -/
theorem evidence_margin_of_score_error {Mode : Type}
    (ideal actual : Mode → ℝ) (γ η : ℝ) (z w : Mode)
    (hgap : γ ≤ ideal z - ideal w) (herr : ∀ v, |ideal v - actual v| ≤ η) :
    γ - 2 * η ≤ actual z - actual w := by
  have hz := abs_le.mp (herr z)
  have hw := abs_le.mp (herr w)
  linarith

theorem temperedFilterPath_shift {Mode : Type} [Fintype Mode]
    (a k δ c : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ) (n : ℕ) :
    temperedFilterPath a k δ c initial evidence (n + 1) =
      temperedFilterPath a k δ c (temperedFilterStep a k δ c initial (evidence 0))
        (fun t => evidence (t + 1)) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change temperedFilterStep a k δ c (temperedFilterPath a k δ c initial evidence (n + 1))
        (evidence (n + 1)) =
      temperedFilterStep a k δ c
        (temperedFilterPath a k δ c (temperedFilterStep a k δ c initial (evidence 0))
          (fun t => evidence (t + 1)) n) (evidence (n + 1))
    rw [ih]

/-- The sticky floor makes recovery independent of a possibly stale incoming belief. -/
theorem temperedFilter_switch_recovery {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c m : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 < c) (hm : 0 < m)
    (hb : IsModeMass initial) (z : Mode) (n : ℕ)
    (hgap : ∀ t, t < n + 1 → ∀ w, w ≠ z → m ≤ evidence t z - evidence t w)
    (hcert : 0 < filterMarginEnvelope δ (c * m)
      (Real.log (a + k) - Real.log k) (n + 1)) :
    ∀ w, w ≠ z → temperedFilterPath a k δ c initial evidence (n + 1) w <
      temperedFilterPath a k δ c initial evidence (n + 1) z := by
  let B := Real.log (a + k) - Real.log k
  let first := temperedFilterStep a k δ c initial (evidence 0)
  have hfirst : ∀ w, w ≠ z → -(δ * B - c * m) ≤ modeLogOdds first z w := by
    intro w hw
    have hp := mul_le_mul_of_nonneg_left
      (stickyPrior_log_odds_floor a k initial ha hk hb z w) hδ
    have he := mul_le_mul_of_nonneg_left (hgap 0 (by omega) w hw) hc.le
    dsimp [first, B]
    rw [temperedFilter_log_odds]
    nlinarith
  have hcert' : 0 < filterMarginEnvelope δ (c * m) (δ * B - c * m) n := by
    have heq : filterMarginEnvelope δ (c * m) (δ * B - c * m) n =
        filterMarginEnvelope δ (c * m) B (n + 1) := by
      simp only [filterMarginEnvelope, pow_succ, geom_sum_succ']
      ring
    rw [heq]
    exact hcert
  have h := temperedFilter_mode_recovery a k δ c m (δ * B - c * m) first
    (fun t => evidence (t + 1)) ha hk hδ hc hm
    (modeSoftmax_isModeMass _) (modeSoftmax_pos _) z hfirst n
    (fun t ht w hw => hgap (t + 1) (by omega) w hw) hcert'
  simpa only [temperedFilterPath_shift, first] using h

theorem beliefActionRisk_reference_filter {d : ℕ} {Mode : Type}
    [Fintype Mode] [DecidableEq Mode]
    (truth reference actual : Mode → ℝ) (D : ℝ) (g : Mode → Action d)
    (q : Action d) (selected : Mode)
    (htruth : IsModeMass truth) (href : IsModeMass reference) (hact : IsModeMass actual)
    (hrp : ∀ z, 0 < reference z) (hap : ∀ z, 0 < actual z)
    (h : LogOddsClose reference actual D) :
    beliefActionRisk truth g q selected ≤ beliefActionRisk actual g q selected +
      (∑ j, (q j) ^ 2) * (modeTotalVariation truth reference + (Real.exp D - 1)) := by
  have hcal := beliefActionRisk_calibration truth actual g q selected htruth hact
  have htv := calibration_of_reference_filter truth reference actual D href hact hrp hap h
  have hmul := mul_le_mul_of_nonneg_left htv
    (sum_nonneg (s := univ) fun j _ => sq_nonneg (q j))
  have hu := (le_abs_self
    (beliefActionRisk truth g q selected - beliefActionRisk actual g q selected)).trans hcal
  linarith

end BAPR.ActionCompensation
