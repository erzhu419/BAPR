import ActionCompensation

/-!
# Belief calibration and BAPR compensation risk

The mode space is finite; the state, history, transition law, and actuator
noise need not be. `p` is a true conditional mode distribution and `b` is the
implemented filter's belief at the same information set. No Bayesian
correctness or calibration is assumed for the tempered sticky filter.
This module has no simulator-specific dependencies.
-/

noncomputable section

namespace BAPR.ActionCompensation

open Finset

def IsModeMass {Mode : Type} [Fintype Mode] (p : Mode → ℝ) : Prop :=
  (∀ z, 0 ≤ p z) ∧ ∑ z, p z = 1

def modeActionCost {d : ℕ} {Mode : Type} [DecidableEq Mode]
    (g : Mode → Action d) (q : Action d) (selected z : Mode) : ℝ :=
  ∑ j, if g z j = g selected j then 0 else (q j) ^ 2

def beliefActionRisk {d : ℕ} {Mode : Type} [Fintype Mode] [DecidableEq Mode]
    (p : Mode → ℝ) (g : Mode → Action d) (q : Action d) (selected : Mode) : ℝ :=
  ∑ z, p z * modeActionCost g q selected z

def modeTotalVariation {Mode : Type} [Fintype Mode] (p b : Mode → ℝ) : ℝ :=
  (∑ z, |p z - b z|) / 2

theorem modeActionCost_nonneg {d : ℕ} {Mode : Type} [DecidableEq Mode]
    (g : Mode → Action d) (q : Action d) (selected z : Mode) :
    0 ≤ modeActionCost g q selected z := by
  exact sum_nonneg fun j _ => by split_ifs <;> positivity

theorem modeActionCost_le_energy {d : ℕ} {Mode : Type} [DecidableEq Mode]
    (g : Mode → Action d) (q : Action d) (selected z : Mode) :
    modeActionCost g q selected z ≤ ∑ j, (q j) ^ 2 := by
  apply sum_le_sum
  intro j _
  split_ifs <;> nlinarith [sq_nonneg (q j)]

theorem modeActionCost_self {d : ℕ} {Mode : Type} [DecidableEq Mode]
    (g : Mode → Action d) (q : Action d) (selected : Mode) :
    modeActionCost g q selected selected = 0 := by
  simp [modeActionCost]

theorem beliefActionRisk_nonneg {d : ℕ} {Mode : Type}
    [Fintype Mode] [DecidableEq Mode]
    (p : Mode → ℝ) (g : Mode → Action d) (q : Action d) (selected : Mode)
    (hp : IsModeMass p) : 0 ≤ beliefActionRisk p g q selected := by
  exact sum_nonneg fun z _ => mul_nonneg (hp.1 z)
    (modeActionCost_nonneg g q selected z)

/-- Exact conditional mean squared nominal-signal error, not classification loss. -/
theorem belief_squared_signal_error {d : ℕ} {Mode : Type}
    [Fintype Mode] [DecidableEq Mode]
    (p : Mode → ℝ) (g : Mode → Action d) (q : Action d) (reference selected : Mode)
    (hg : ∀ z, IsPolarity (g z)) :
    (∑ z, p z * ∑ j,
      (modeError (g z) (g selected) (g reference) q j) ^ 2) =
      4 * beliefActionRisk p g q selected := by
  simp_rw [action_weighted_mode_error _ _ _ _ (hg _) (hg _) (hg _)]
  unfold beliefActionRisk modeActionCost
  rw [mul_sum]
  apply sum_congr rfl
  intro z _
  ring

/-- The bound survives common actuator noise and the actual action clipping. -/
theorem belief_squared_executed_error_bound {d : ℕ} {Mode : Type}
    [Fintype Mode] [DecidableEq Mode]
    (p : Mode → ℝ) (g : Mode → Action d) (q noise : Action d)
    (reference selected : Mode) (hp : IsModeMass p)
    (hg : ∀ z, IsPolarity (g z)) :
    (∑ z, p z * ∑ j,
      (executed (applyGain (g z) (compensate (g selected) (g reference) q)) noise j -
        executed (applyGain (g reference) q) noise j) ^ 2) ≤
      4 * beliefActionRisk p g q selected := by
  rw [← belief_squared_signal_error p g q reference selected hg]
  apply sum_le_sum
  intro z _
  apply mul_le_mul_of_nonneg_left _ (hp.1 z)
  apply sum_le_sum
  intro j _
  have hd : applyGain (g z) (compensate (g selected) (g reference) q) j -
      applyGain (g reference) q j = modeError (g z) (g selected) (g reference) q j := by
    simp only [applyGain, compensate, modeError]
    ring
  have h := clipScalar_nonexpansive
    (applyGain (g z) (compensate (g selected) (g reference) q) j + noise j)
    (applyGain (g reference) q j + noise j)
  rw [add_sub_add_right_eq_sub, hd] at h
  change |executed (applyGain (g z) (compensate (g selected) (g reference) q)) noise j -
      executed (applyGain (g reference) q) noise j| ≤
      |modeError (g z) (g selected) (g reference) q j| at h
  nlinarith [sq_abs (modeError (g z) (g selected) (g reference) q j),
    sq_abs (executed (applyGain (g z) (compensate (g selected) (g reference) q)) noise j -
      executed (applyGain (g reference) q) noise j), abs_nonneg
      (executed (applyGain (g z) (compensate (g selected) (g reference) q)) noise j -
        executed (applyGain (g reference) q) noise j)]

/-- Centering the bounded cost gives the sharp total-variation coefficient. -/
theorem beliefActionRisk_calibration {d : ℕ} {Mode : Type}
    [Fintype Mode] [DecidableEq Mode]
    (p b : Mode → ℝ) (g : Mode → Action d) (q : Action d) (selected : Mode)
    (hp : IsModeMass p) (hb : IsModeMass b) :
    |beliefActionRisk p g q selected - beliefActionRisk b g q selected| ≤
      (∑ j, (q j) ^ 2) * modeTotalVariation p b := by
  let E : ℝ := ∑ j, (q j) ^ 2
  have hE : 0 ≤ E := sum_nonneg fun j _ => sq_nonneg (q j)
  have hmass : ∑ z, (p z - b z) = 0 := by
    rw [sum_sub_distrib, hp.2, hb.2, sub_self]
  have hcenter (z : Mode) :
      |modeActionCost g q selected z - E / 2| ≤ E / 2 := by
    rw [abs_le]
    have hlo := modeActionCost_nonneg g q selected z
    have hhi := modeActionCost_le_energy g q selected z
    change modeActionCost g q selected z ≤ E at hhi
    constructor <;> linarith
  have hid : beliefActionRisk p g q selected - beliefActionRisk b g q selected =
      ∑ z, (p z - b z) * (modeActionCost g q selected z - E / 2) := by
    calc
      _ = ∑ z, (p z - b z) * modeActionCost g q selected z := by
        simp only [beliefActionRisk, ← sum_sub_distrib]
        apply sum_congr rfl
        intro z _
        ring
      _ = _ := by
        simp only [mul_sub, sum_sub_distrib, ← sum_mul, hmass, zero_mul, sub_zero]
  rw [hid]
  calc
    _ ≤ ∑ z, |(p z - b z) * (modeActionCost g q selected z - E / 2)| :=
      abs_sum_le_sum_abs _ _
    _ ≤ ∑ z, |p z - b z| * (E / 2) := by
      apply sum_le_sum
      intro z _
      rw [abs_mul]
      exact mul_le_mul_of_nonneg_left (hcenter z) (abs_nonneg _)
    _ = E * modeTotalVariation p b := by
      rw [← sum_mul, modeTotalVariation]
      ring

theorem beliefActionRisk_le_error_mass {d : ℕ} {Mode : Type}
    [Fintype Mode] [DecidableEq Mode]
    (b : Mode → ℝ) (g : Mode → Action d) (q : Action d) (selected : Mode)
    (hb : IsModeMass b) :
    beliefActionRisk b g q selected ≤ (∑ j, (q j) ^ 2) * (1 - b selected) := by
  let E : ℝ := ∑ j, (q j) ^ 2
  have hterm (z : Mode) : b z * modeActionCost g q selected z ≤
      b z * E - (if z = selected then b z * E else 0) := by
    by_cases h : z = selected
    · subst z
      simp [modeActionCost_self]
    · simp only [h, if_false, sub_zero]
      exact mul_le_mul_of_nonneg_left
        (modeActionCost_le_energy g q selected z) (hb.1 z)
  calc
    _ ≤ ∑ z, (b z * E - (if z = selected then b z * E else 0)) :=
      sum_le_sum fun z _ => hterm z
    _ = E * (1 - b selected) := by
      rw [sum_sub_distrib, ← sum_mul, hb.2]
      simp only [sum_ite_eq', mem_univ, if_true]
      ring

/-- A high-confidence MAP route is safe only to the extent that belief is calibrated. -/
theorem beliefActionRisk_confidence {d : ℕ} {Mode : Type}
    [Fintype Mode] [DecidableEq Mode]
    (p b : Mode → ℝ) (g : Mode → Action d) (q : Action d) (selected : Mode)
    (hp : IsModeMass p) (hb : IsModeMass b) :
    beliefActionRisk p g q selected ≤
      (∑ j, (q j) ^ 2) * (1 - b selected + modeTotalVariation p b) := by
  have hcal := beliefActionRisk_calibration p b g q selected hp hb
  have hmass := beliefActionRisk_le_error_mass b g q selected hb
  have hu := (le_abs_self
    (beliefActionRisk p g q selected - beliefActionRisk b g q selected)).trans hcal
  nlinarith

theorem belief_sqrt_cost_bound {d : ℕ} {Mode : Type}
    [Fintype Mode] [DecidableEq Mode]
    (p : Mode → ℝ) (g : Mode → Action d) (q : Action d) (selected : Mode)
    (hp : IsModeMass p) :
    (∑ z, p z * Real.sqrt (modeActionCost g q selected z)) ≤
      Real.sqrt (beliefActionRisk p g q selected) := by
  have hsq : (∑ z, p z * Real.sqrt (modeActionCost g q selected z)) ^ 2 ≤
      beliefActionRisk p g q selected := by
    have h := sum_sq_le_sum_mul_sum_of_sq_eq_mul univ
      (f := p) (g := fun z => p z * modeActionCost g q selected z)
      (r := fun z => p z * Real.sqrt (modeActionCost g q selected z))
      (fun z _ => hp.1 z)
      (fun z _ => mul_nonneg (hp.1 z) (modeActionCost_nonneg g q selected z))
      (fun z _ => by
        rw [mul_pow, Real.sq_sqrt (modeActionCost_nonneg g q selected z)]
        ring)
    simpa only [hp.2, one_mul, beliefActionRisk] using h
  exact (Real.le_sqrt (sum_nonneg fun z _ => mul_nonneg (hp.1 z)
    (Real.sqrt_nonneg _)) (beliefActionRisk_nonneg p g q selected hp)).mpr hsq

/-- A mode-conditioned action-value sensitivity bound, at one causal decision.
    The sensitivity must hold uniformly over the conditional mode support. -/
theorem belief_action_value_bound {d : ℕ} {Mode : Type}
    [Fintype Mode] [DecidableEq Mode]
    (p b : Mode → ℝ) (g : Mode → Action d) (q : Action d) (reference selected : Mode)
    (Q : Mode → Action d → ℝ) (L : ℝ)
    (hp : IsModeMass p) (hb : IsModeMass b) (hg : ∀ z, IsPolarity (g z))
    (hL : 0 ≤ L)
    (hsens : ∀ z,
      |Q z (applyGain (g z) (compensate (g selected) (g reference) q)) -
        Q z (applyGain (g reference) q)| ≤
      L * l2Norm (modeError (g z) (g selected) (g reference) q)) :
    |∑ z, p z *
      (Q z (applyGain (g z) (compensate (g selected) (g reference) q)) -
        Q z (applyGain (g reference) q))| ≤
      2 * L * Real.sqrt (beliefActionRisk b g q selected +
        (∑ j, (q j) ^ 2) * modeTotalVariation p b) := by
  have hrisk : beliefActionRisk p g q selected ≤ beliefActionRisk b g q selected +
      (∑ j, (q j) ^ 2) * modeTotalVariation p b := by
    have hcal := beliefActionRisk_calibration p b g q selected hp hb
    have hu := (le_abs_self
      (beliefActionRisk p g q selected - beliefActionRisk b g q selected)).trans hcal
    linarith
  calc
    _ ≤ ∑ z, |p z *
        (Q z (applyGain (g z) (compensate (g selected) (g reference) q)) -
          Q z (applyGain (g reference) q))| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ z, p z * (L * (2 * Real.sqrt (modeActionCost g q selected z))) := by
      apply sum_le_sum
      intro z _
      rw [abs_mul, abs_of_nonneg (hp.1 z)]
      apply mul_le_mul_of_nonneg_left _ (hp.1 z)
      simpa only [modeError_l2Norm _ _ _ _ (hg _) (hg _) (hg _), modeActionCost]
        using hsens z
    _ = 2 * L * ∑ z, p z * Real.sqrt (modeActionCost g q selected z) := by
      rw [mul_sum]
      apply sum_congr rfl
      intro z _
      ring
    _ ≤ 2 * L * Real.sqrt (beliefActionRisk p g q selected) :=
      mul_le_mul_of_nonneg_left (belief_sqrt_cost_bound p g q selected hp)
        (by positivity)
    _ ≤ _ := mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt hrisk) (by positivity)

end BAPR.ActionCompensation
