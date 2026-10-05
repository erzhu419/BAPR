import Mathlib

/-!
# BAPR actuator-polarity mechanism

This file formalizes the mechanism used by the current BAPR paper, not the
legacy belief-weighted Bellman operator in `BAPR.lean`. The actuator map is
diagonal, actions are bounded componentwise, and all modes share the same
physics, reward, and exogenous noise. Both pathwise and finite-noise expected
returns are modeled below. The conditional-KL chain step is stated for finite
positive kernels, matching the algebra of a fixed-mode dwell interval.

The pathwise results allow arbitrary real-valued noise sequences, including
realizations of the experimental Gaussian noise. The expectation results use
finite sums: their probability interpretation requires nonnegative normalized
weights. `ActionCompensationMeasure.lean` supplies the Bochner-integral
counterparts, explicit Gaussian noise, and measurable causal execution.
For a history-dependent controller, `X` includes its belief/history as well as
the physical state; `expectedValue_project` below justifies ignoring that
memory when evaluating the frozen oracle.
-/

noncomputable section

namespace BAPR.ActionCompensation

abbrev Action (d : ℕ) := Fin d → ℝ

def clipScalar (x : ℝ) : ℝ := max (-1) (min x 1)

def inBox {d : ℕ} (a : Action d) : Prop :=
  ∀ j, -1 ≤ a j ∧ a j ≤ 1

def clipAction {d : ℕ} (a : Action d) : Action d :=
  fun j => clipScalar (a j)

def applyGain {d : ℕ} (g a : Action d) : Action d :=
  fun j => g j * a j

def inverseCommand {d : ℕ} (g target : Action d) : Action d :=
  fun j => target j / g j

theorem clipScalar_self_iff (x : ℝ) :
    clipScalar x = x ↔ -1 ≤ x ∧ x ≤ 1 := by
  constructor
  · intro h
    constructor
    · have hlo : (-1 : ℝ) ≤ max (-1) (min x 1) := le_max_left _ _
      dsimp [clipScalar] at h
      linarith
    · have hhi : max (-1 : ℝ) (min x 1) ≤ 1 :=
        max_le (by norm_num) (min_le_right _ _)
      dsimp [clipScalar] at h
      linarith
  · rintro ⟨hlo, hhi⟩
    simp [clipScalar, min_eq_left hhi, max_eq_right hlo]

theorem clipAction_self_iff {d : ℕ} (a : Action d) :
    clipAction a = a ↔ inBox a := by
  constructor
  · intro h j
    exact (clipScalar_self_iff (a j)).mp (congrFun h j)
  · intro h
    funext j
    exact (clipScalar_self_iff (a j)).mpr (h j)

theorem applyGain_inverseCommand {d : ℕ} (g target : Action d)
    (hg : ∀ j, g j ≠ 0) :
    applyGain g (inverseCommand g target) = target := by
  funext j
  dsimp [applyGain, inverseCommand]
  field_simp [hg j]

/-- Invertibility is not enough: exact nominal compensation is equivalent to
    feasibility of the inverse command in the actuator box. -/
theorem exact_compensation_iff_feasible {d : ℕ} (g target : Action d)
    (hg : ∀ j, g j ≠ 0) :
    applyGain g (clipAction (inverseCommand g target)) = target ↔
      inBox (inverseCommand g target) := by
  constructor
  · intro h
    have hsame : applyGain g (clipAction (inverseCommand g target)) =
        applyGain g (inverseCommand g target) := by
      rw [h, applyGain_inverseCommand g target hg]
    apply (clipAction_self_iff (inverseCommand g target)).mp
    funext j
    have hj := congrFun hsame j
    have hzero : g j * (clipScalar (inverseCommand g target j) -
        inverseCommand g target j) = 0 := by
      dsimp [applyGain, clipAction] at hj
      nlinarith
    rcases mul_eq_zero.mp hzero with hbad | hclip
    · exact False.elim ((hg j) hbad)
    · exact sub_eq_zero.mp hclip
  · intro h
    rw [(clipAction_self_iff (inverseCommand g target)).mpr h]
    exact applyGain_inverseCommand g target hg

/-- A polarity matrix is a diagonal gain with entries in `{+1,-1}`. -/
def IsPolarity {d : ℕ} (g : Action d) : Prop :=
  ∀ j, g j = 1 ∨ g j = -1

theorem polarity_square {d : ℕ} {g : Action d} (hg : IsPolarity g)
    (j : Fin d) : g j * g j = 1 := by
  rcases hg j with h | h <;> simp [h]

def compensate {d : ℕ} (gCurrent gReference q : Action d) : Action d :=
  fun j => gCurrent j * gReference j * q j

theorem polarity_compensates {d : ℕ} (gCurrent gReference q : Action d)
    (hg : IsPolarity gCurrent) :
    applyGain gCurrent (compensate gCurrent gReference q) =
      applyGain gReference q := by
  funext j
  dsimp [applyGain, compensate]
  calc
    gCurrent j * (gCurrent j * gReference j * q j) =
        (gCurrent j * gCurrent j) * (gReference j * q j) := by ring
    _ = gReference j * q j := by rw [polarity_square hg j]; ring

theorem polarity_compensated_inBox {d : ℕ}
    (gCurrent gReference q : Action d)
    (hc : IsPolarity gCurrent) (hr : IsPolarity gReference)
    (hq : inBox q) : inBox (compensate gCurrent gReference q) := by
  intro j
  rcases hc j with hc | hc <;> rcases hr j with hr | hr
  all_goals
    simp [compensate, hc, hr]
    constructor <;> nlinarith [(hq j).1, (hq j).2]

def executed {d : ℕ} (signal noise : Action d) : Action d :=
  clipAction (fun j => signal j + noise j)

theorem polarity_executed_equal {d : ℕ}
    (gCurrent gReference q noise : Action d)
    (hg : IsPolarity gCurrent) :
    executed (applyGain gCurrent (compensate gCurrent gReference q)) noise =
      executed (applyGain gReference q) noise := by
  rw [polarity_compensates gCurrent gReference q hg]

/-- A finite-horizon simulator driven by a nominal pre-noise signal. Its
    action cost, if any, is represented by `reward` on the executed signal. -/
def rollout {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (signal : ℕ → X → Action d) (noise : ℕ → Action d) (event : ℕ → E)
    (n t : ℕ) (x : X) : X × ℝ :=
  match n with
  | 0 => (x, 0)
  | n + 1 =>
      let u := executed (signal t x) (noise t)
      let x' := step x u (event t)
      let tail := rollout step reward signal noise event n (t + 1) x'
      (tail.1, reward x u (event t) + tail.2)

theorem rollout_congr {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (signal₁ signal₂ : ℕ → X → Action d)
    (noise : ℕ → Action d) (event : ℕ → E)
    (h : ∀ t x, signal₁ t x = signal₂ t x)
    (n t : ℕ) (x : X) :
    rollout step reward signal₁ noise event n t x =
      rollout step reward signal₂ noise event n t x := by
  induction n generalizing t x with
  | zero => rfl
  | succ n ih =>
      simp [rollout, h, ih]

/-- Exact pathwise state and return equality for an arbitrary switch schedule. -/
theorem arbitrary_polarity_switching {d : ℕ} {X Obs E Mode : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (observe : X → Obs) (policy : Obs → Action d)
    (g : Mode → Action d) (r : Mode) (schedule : ℕ → Mode)
    (hg : ∀ z, IsPolarity (g z))
    (noise : ℕ → Action d) (event : ℕ → E)
    (n t : ℕ) (x : X) :
    rollout step reward
      (fun k s => applyGain (g (schedule k))
        (compensate (g (schedule k)) (g r) (policy (observe s))))
      noise event n t x =
    rollout step reward
      (fun _ s => applyGain (g r) (policy (observe s)))
      noise event n t x := by
  apply rollout_congr
  intro k s
  exact polarity_compensates _ _ _ (hg (schedule k))

/-- The same causal command under two modes cannot expose a difference if
    their physical actuator signals coincide on every visited state. -/
theorem no_excitation_same_rollout {d : ℕ} {X E Mode : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (command : ℕ → X → Action d) (g : Mode → Action d)
    (mode₁ mode₂ : ℕ → Mode) (noise : ℕ → Action d) (event : ℕ → E)
    (h : ∀ t x, applyGain (g (mode₁ t)) (command t x) =
                  applyGain (g (mode₂ t)) (command t x))
    (n t : ℕ) (x : X) :
    rollout step reward (fun k s => applyGain (g (mode₁ k)) (command k s))
      noise event n t x =
    rollout step reward (fun k s => applyGain (g (mode₂ k)) (command k s))
      noise event n t x := by
  apply rollout_congr
  exact h

/-- An action selected from the first `t` completed transitions is unaffected
    by replacing any later transitions. -/
def causalCommand {B S T : Type} {d : ℕ}
    (update : B → T → B) (initial : B) (select : B → S → Action d)
    (stream : List T) (t : ℕ) (state : S) : Action d :=
  select ((stream.take t).foldl update initial) state

theorem causalCommand_prefix {B S T : Type} {d : ℕ}
    (update : B → T → B) (initial : B) (select : B → S → Action d)
    (stream₁ stream₂ : List T) (t : ℕ) (state : S)
    (h : stream₁.take t = stream₂.take t) :
    causalCommand update initial select stream₁ t state =
      causalCommand update initial select stream₂ t state := by
  simp [causalCommand, h]

def modeError {d : ℕ} (gTrue gSelected gReference q : Action d) : Action d :=
  fun j => (gTrue j * gSelected j - 1) * gReference j * q j

theorem modeError_square {d : ℕ}
    (gTrue gSelected gReference q : Action d)
    (ht : IsPolarity gTrue) (hs : IsPolarity gSelected)
    (hr : IsPolarity gReference) (j : Fin d) :
    (modeError gTrue gSelected gReference q j) ^ 2 =
      if gTrue j = gSelected j then 0 else 4 * (q j) ^ 2 := by
  rcases ht j with ht | ht <;>
    rcases hs j with hs | hs <;>
    rcases hr j with hr | hr <;>
    simp [modeError, ht, hs, hr] <;> norm_num <;> ring

/-- The squared Euclidean signal error is action-weighted mode disagreement. -/
theorem action_weighted_mode_error {d : ℕ}
    (gTrue gSelected gReference q : Action d)
    (ht : IsPolarity gTrue) (hs : IsPolarity gSelected)
    (hr : IsPolarity gReference) :
    (∑ j : Fin d, (modeError gTrue gSelected gReference q j) ^ 2) =
      4 * ∑ j : Fin d, if gTrue j = gSelected j then 0 else (q j) ^ 2 := by
  calc
    _ = ∑ j : Fin d, (if gTrue j = gSelected j then 0 else 4 * (q j) ^ 2) := by
      apply Finset.sum_congr rfl
      intro j _
      exact modeError_square gTrue gSelected gReference q ht hs hr j
    _ = 4 * ∑ j : Fin d, if gTrue j = gSelected j then 0 else (q j) ^ 2 := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      split_ifs <;> ring

def l2Norm {d : ℕ} (a : Action d) : ℝ :=
  Real.sqrt (∑ j : Fin d, (a j) ^ 2)

/-- The actual actuator clipping cannot amplify a same-state signal error. -/
theorem clipScalar_nonexpansive (x y : ℝ) :
    |clipScalar x - clipScalar y| ≤ |x - y| := by
  have hmin : |min x 1 - min y 1| ≤ |x - y| := by
    simpa using abs_min_sub_min_le_max x (1 : ℝ) y 1
  have hmax : |clipScalar x - clipScalar y| ≤ |min x 1 - min y 1| := by
    simpa [clipScalar] using
      abs_max_sub_max_le_max (-1 : ℝ) (min x 1) (-1) (min y 1)
  exact hmax.trans hmin

theorem executed_l2_nonexpansive {d : ℕ} (v w noise : Action d) :
    l2Norm (fun j => executed v noise j - executed w noise j) ≤
      l2Norm (fun j => v j - w j) := by
  apply Real.sqrt_le_sqrt
  apply Finset.sum_le_sum
  intro j _
  have h : |executed v noise j - executed w noise j| ≤ |v j - w j| := by
    simpa [executed, clipAction] using
      clipScalar_nonexpansive (v j + noise j) (w j + noise j)
  nlinarith [sq_abs (executed v noise j - executed w noise j),
    sq_abs (v j - w j), abs_nonneg (executed v noise j - executed w noise j),
    abs_nonneg (v j - w j)]

theorem modeError_l2Norm {d : ℕ}
    (gTrue gSelected gReference q : Action d)
    (ht : IsPolarity gTrue) (hs : IsPolarity gSelected)
    (hr : IsPolarity gReference) :
    l2Norm (modeError gTrue gSelected gReference q) =
      2 * Real.sqrt
        (∑ j : Fin d, if gTrue j = gSelected j then 0 else (q j) ^ 2) := by
  rw [l2Norm, action_weighted_mode_error gTrue gSelected gReference q ht hs hr]
  rw [Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 4)]
  norm_num

def oracleNominal {d : ℕ} {X Obs Mode : Type}
    (observe : X → Obs) (policy : Obs → Action d)
    (g : Mode → Action d) (reference : Mode) (_t : ℕ) (x : X) : Action d :=
  applyGain (g reference) (policy (observe x))

def causalNominal {d : ℕ} {X Obs Mode : Type}
    (observe : X → Obs) (policy : Obs → Action d)
    (g : Mode → Action d) (reference : Mode)
    (trueMode : ℕ → Mode) (selected : ℕ → X → Mode)
    (t : ℕ) (x : X) : Action d :=
  applyGain (g (trueMode t))
    (compensate (g (selected t x)) (g reference) (policy (observe x)))

theorem causal_nominal_difference {d : ℕ} {X Obs Mode : Type}
    (observe : X → Obs) (policy : Obs → Action d)
    (g : Mode → Action d) (reference : Mode)
    (trueMode : ℕ → Mode) (selected : ℕ → X → Mode)
    (t : ℕ) (x : X) :
    (fun j => causalNominal observe policy g reference trueMode selected t x j -
      oracleNominal observe policy g reference t x j) =
      modeError (g (trueMode t)) (g (selected t x)) (g reference)
        (policy (observe x)) := by
  funext j
  dsimp [causalNominal, oracleNominal, applyGain, compensate, modeError]
  ring

/-- With two hypotheses producing the same observed outcome for every noise
    realization, an equal-prior binary decision rule succeeds with probability
    exactly one half. The mass function is finite and normalized; it need not
    be uniform. This is the finite-sample no-excitation consequence, not a
    formalization of the paper's conditional-KL chain rule. -/
theorem indistinguishable_binary_accuracy {Ω Y : Type}
    [Fintype Ω] (mass : Ω → ℝ) (observed₁ observed₂ : Ω → Y)
    (select : Y → Bool)
    (hsame : ∀ ω, observed₁ ω = observed₂ ω)
    (hnorm : ∑ ω : Ω, mass ω = 1) :
    ((∑ ω : Ω, mass ω * (if select (observed₁ ω) then 1 else 0)) +
     (∑ ω : Ω, mass ω * (if select (observed₂ ω) then 0 else 1))) / 2 =
      (1 / 2 : ℝ) := by
  have hf : observed₁ = observed₂ := funext hsame
  rw [← hf]
  rw [← Finset.sum_add_distrib]
  have hsum :
      (∑ ω : Ω,
        (mass ω * (if select (observed₁ ω) then (1 : ℝ) else 0) +
         mass ω * (if select (observed₁ ω) then (0 : ℝ) else 1))) =
        ∑ ω : Ω, mass ω := by
    apply Finset.sum_congr rfl
    intro ω _
    by_cases h : select (observed₁ ω) = true <;> simp [h]
  rw [hsum, hnorm]

/-- One step of the conditional-KL chain rule. The history distributions
    under two mode hypotheses are `p` and `q`; `k` and `l` are their next
    observation kernels. Iterating this equality over a finite dwell interval
    gives the paper's trajectory KL decomposition. Strict positivity avoids
    extended-real conventions for zero-probability observations. -/
theorem finite_conditional_kl_chain_step {H Y : Type}
    [Fintype H] [Fintype Y]
    (p q : H → ℝ) (k l : H → Y → ℝ)
    (hp : ∀ h, 0 < p h) (hq : ∀ h, 0 < q h)
    (hk : ∀ h y, 0 < k h y) (hl : ∀ h y, 0 < l h y)
    (hknorm : ∀ h, ∑ y : Y, k h y = 1) :
    (∑ h : H, ∑ y : Y,
      p h * k h y * Real.log ((p h * k h y) / (q h * l h y))) =
      (∑ h : H, p h * Real.log (p h / q h)) +
      (∑ h : H, p h * ∑ y : Y,
        k h y * Real.log (k h y / l h y)) := by
  have hlog (h : H) (y : Y) :
      Real.log ((p h * k h y) / (q h * l h y)) =
        Real.log (p h / q h) + Real.log (k h y / l h y) := by
    rw [Real.log_div
        (mul_ne_zero (ne_of_gt (hp h)) (ne_of_gt (hk h y)))
        (mul_ne_zero (ne_of_gt (hq h)) (ne_of_gt (hl h y))),
      Real.log_mul (ne_of_gt (hp h)) (ne_of_gt (hk h y)),
      Real.log_mul (ne_of_gt (hq h)) (ne_of_gt (hl h y)),
      Real.log_div (ne_of_gt (hp h)) (ne_of_gt (hq h)),
      Real.log_div (ne_of_gt (hk h y)) (ne_of_gt (hl h y))]
    ring
  have hinner (h : H) :
      (∑ y : Y, p h * k h y *
        (Real.log (p h / q h) + Real.log (k h y / l h y))) =
        p h * Real.log (p h / q h) +
        p h * ∑ y : Y, k h y * Real.log (k h y / l h y) := by
    calc
      _ = (∑ y : Y, p h * k h y * Real.log (p h / q h)) +
          (∑ y : Y, p h * (k h y * Real.log (k h y / l h y))) := by
            rw [← Finset.sum_add_distrib]
            apply Finset.sum_congr rfl
            intro y _
            ring
      _ = p h * Real.log (p h / q h) +
          p h * ∑ y : Y, k h y * Real.log (k h y / l h y) := by
            have hfirst :
                (∑ y : Y, p h * k h y * Real.log (p h / q h)) =
                  p h * Real.log (p h / q h) := by
              calc
                _ = (∑ y : Y, k h y) *
                    (p h * Real.log (p h / q h)) := by
                      rw [Finset.sum_mul]
                      apply Finset.sum_congr rfl
                      intro y _
                      ring
                _ = _ := by rw [hknorm h]; ring
            rw [hfirst, Finset.mul_sum]
  calc
    _ = ∑ h : H, ∑ y : Y, p h * k h y *
        (Real.log (p h / q h) + Real.log (k h y / l h y)) := by
          apply Finset.sum_congr rfl
          intro h _
          apply Finset.sum_congr rfl
          intro y _
          rw [hlog h y]
    _ = ∑ h : H, (p h * Real.log (p h / q h) +
          p h * ∑ y : Y, k h y * Real.log (k h y / l h y)) := by
            apply Finset.sum_congr rfl
            intro h _
            exact hinner h
    _ = _ := by rw [Finset.sum_add_distrib]

/-- Return from following the true-mode oracle for `n` remaining steps under
    a fixed exogenous noise and event sequence. -/
def oracleValue {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (oracle : ℕ → X → Action d) (noise : ℕ → Action d) (event : ℕ → E)
    (n t : ℕ) (x : X) : ℝ :=
  (rollout step reward oracle noise event n t x).2

/-- One-step counterfactual action value followed by the oracle. -/
def oracleQ {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (oracle : ℕ → X → Action d) (noise : ℕ → Action d) (event : ℕ → E)
    (remaining t : ℕ) (x : X) (signal : Action d) : ℝ :=
  let u := executed signal (noise t)
  reward x u (event t) +
    oracleValue step reward oracle noise event remaining (t + 1)
      (step x u (event t))

/-- The sum of oracle action-value differences along the causal trajectory. -/
def oracleGapSum {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (causal oracle : ℕ → X → Action d)
    (noise : ℕ → Action d) (event : ℕ → E)
    (n t : ℕ) (x : X) : ℝ :=
  match n with
  | 0 => 0
  | remaining + 1 =>
      let u := executed (causal t x) (noise t)
      (oracleQ step reward oracle noise event remaining t x (causal t x) -
       oracleQ step reward oracle noise event remaining t x (oracle t x)) +
        oracleGapSum step reward causal oracle noise event remaining (t + 1)
          (step x u (event t))

/-- A pathwise finite-horizon performance-difference identity. The oracle
    continuation is evaluated at states visited by the causal controller;
    later trajectories need not be paired. Averaging this identity over an
    independent exogenous-noise law gives the paper's expected-return identity. -/
theorem pathwise_return_difference {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (causal oracle : ℕ → X → Action d)
    (noise : ℕ → Action d) (event : ℕ → E)
    (n t : ℕ) (x : X) :
    (rollout step reward causal noise event n t x).2 -
      (rollout step reward oracle noise event n t x).2 =
        oracleGapSum step reward causal oracle noise event n t x := by
  induction n generalizing t x with
  | zero => simp [rollout, oracleGapSum]
  | succ n ih =>
      simp only [rollout, oracleGapSum, oracleQ, oracleValue]
      rw [← ih (t + 1)
        (step x (executed (causal t x) (noise t)) (event t))]
      ring

/-- Cumulative local action-value sensitivity along the causal trajectory. -/
def gapBoundSum {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (causal : ℕ → X → Action d)
    (noise : ℕ → Action d) (event : ℕ → E)
    (localBound : ℕ → X → ℝ) (n t : ℕ) (x : X) : ℝ :=
  match n with
  | 0 => 0
  | remaining + 1 =>
      localBound t x +
        gapBoundSum step causal noise event localBound remaining (t + 1)
          (step x (executed (causal t x) (noise t)) (event t))

/-- A conditional return bound: any valid per-state oracle action-value
    difference bound sums along the causal path. No global regularity of
    contact dynamics is asserted by this theorem. -/
theorem pathwise_return_bound {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (causal oracle : ℕ → X → Action d)
    (noise : ℕ → Action d) (event : ℕ → E)
    (localBound : ℕ → X → ℝ)
    (hlocal : ∀ remaining t x,
      |oracleQ step reward oracle noise event remaining t x (causal t x) -
       oracleQ step reward oracle noise event remaining t x (oracle t x)| ≤
        localBound t x)
    (n t : ℕ) (x : X) :
    |(rollout step reward causal noise event n t x).2 -
      (rollout step reward oracle noise event n t x).2| ≤
        gapBoundSum step causal noise event localBound n t x := by
  rw [pathwise_return_difference step reward causal oracle noise event n t x]
  induction n generalizing t x with
  | zero => simp [oracleGapSum, gapBoundSum]
  | succ n ih =>
      simp only [oracleGapSum, gapBoundSum]
      exact (abs_add_le _ _).trans
        (add_le_add (hlocal n t x)
          (ih (t + 1)
            (step x (executed (causal t x) (noise t)) (event t))))

/-! A finite-noise model makes the performance-difference identity an exact
expected-return theorem, rather than merely a pathwise telescoping identity.
The same probability mass is used at every step, so successive draws are
independent; time-dependent laws can be included in the state or time index. -/

def expectedQ {d : ℕ} {X E : Type} [Fintype E]
    (mass : E → ℝ) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ)
    (nextValue : X → ℝ) (noise : E → Action d)
    (x : X) (signal : Action d) : ℝ :=
  ∑ e : E,
    mass e *
      (let u := executed signal (noise e)
       reward x u e + nextValue (step x u e))

def expectedValue {d : ℕ} {X E : Type} [Fintype E]
    (mass : E → ℝ) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ)
    (policy : ℕ → X → Action d) (noise : E → Action d)
    (n t : ℕ) (x : X) : ℝ :=
  match n with
  | 0 => 0
  | remaining + 1 =>
      expectedQ mass step reward
        (expectedValue mass step reward policy noise remaining (t + 1))
        noise x (policy t x)

def expectedGapSum {d : ℕ} {X E : Type} [Fintype E]
    (mass : E → ℝ) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ)
    (causal oracle : ℕ → X → Action d) (noise : E → Action d)
    (n t : ℕ) (x : X) : ℝ :=
  match n with
  | 0 => 0
  | remaining + 1 =>
      let oracleNext :=
        expectedValue mass step reward oracle noise remaining (t + 1)
      (expectedQ mass step reward oracleNext noise x (causal t x) -
       expectedQ mass step reward oracleNext noise x (oracle t x)) +
        ∑ e : E, mass e *
          expectedGapSum mass step reward causal oracle noise remaining (t + 1)
            (step x (executed (causal t x) (noise e)) e)

/-- Exact expected-return difference under a finite exogenous-noise law. The
    recursion `expectedGapSum` is the expectation along causal state occupancy
    of the current oracle-Q difference. No Lipschitz assumption is needed. -/
theorem expected_return_difference {d : ℕ} {X E : Type} [Fintype E]
    (mass : E → ℝ) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ)
    (causal oracle : ℕ → X → Action d) (noise : E → Action d)
    (n t : ℕ) (x : X) :
    expectedValue mass step reward causal noise n t x -
      expectedValue mass step reward oracle noise n t x =
        expectedGapSum mass step reward causal oracle noise n t x := by
  induction n generalizing t x with
  | zero => simp [expectedValue, expectedGapSum]
  | succ n ih =>
      simp only [expectedValue, expectedGapSum, expectedQ]
      rw [← Finset.sum_sub_distrib]
      -- The causal continuation differs from the oracle continuation by
      -- exactly the induction hypothesis at each reachable next state.
      rw [← Finset.sum_sub_distrib]
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro e _
      rw [← ih (t + 1)
        (step x (executed (causal t x) (noise e)) e)]
      ring

/-- Expected accumulation of a local action-value discrepancy bound under
    the causal controller's state distribution. -/
def expectedBoundSum {d : ℕ} {X E : Type} [Fintype E]
    (mass : E → ℝ) (step : X → Action d → E → X)
    (causal : ℕ → X → Action d) (noise : E → Action d)
    (localBound : ℕ → X → ℝ) (n t : ℕ) (x : X) : ℝ :=
  match n with
  | 0 => 0
  | remaining + 1 =>
      localBound t x +
        ∑ e : E, mass e *
          expectedBoundSum mass step causal noise localBound remaining (t + 1)
            (step x (executed (causal t x) (noise e)) e)

/-- The paper's conditional expected-return bound in a finite-noise model.
    The caller must supply the per-state oracle-Q sensitivity bound. No
    Lipschitz property of simulator contact dynamics is silently assumed. -/
theorem expected_return_bound {d : ℕ} {X E : Type} [Fintype E]
    (mass : E → ℝ) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ)
    (causal oracle : ℕ → X → Action d) (noise : E → Action d)
    (localBound : ℕ → X → ℝ)
    (hmass : ∀ e, 0 ≤ mass e)
    (hlocal : ∀ remaining t x,
      |expectedQ mass step reward
        (expectedValue mass step reward oracle noise remaining (t + 1))
        noise x (causal t x) -
       expectedQ mass step reward
        (expectedValue mass step reward oracle noise remaining (t + 1))
        noise x (oracle t x)| ≤ localBound t x)
    (n t : ℕ) (x : X) :
    |expectedValue mass step reward causal noise n t x -
      expectedValue mass step reward oracle noise n t x| ≤
        expectedBoundSum mass step causal noise localBound n t x := by
  rw [expected_return_difference mass step reward causal oracle noise n t x]
  induction n generalizing t x with
  | zero => simp [expectedGapSum, expectedBoundSum]
  | succ n ih =>
      simp only [expectedGapSum, expectedBoundSum]
      have hsum :
          |∑ e : E, mass e *
            expectedGapSum mass step reward causal oracle noise n (t + 1)
              (step x (executed (causal t x) (noise e)) e)| ≤
          ∑ e : E, mass e *
            expectedBoundSum mass step causal noise localBound n (t + 1)
              (step x (executed (causal t x) (noise e)) e) := by
        calc
          _ ≤ ∑ e : E, |mass e *
              expectedGapSum mass step reward causal oracle noise n (t + 1)
                (step x (executed (causal t x) (noise e)) e)| :=
              Finset.abs_sum_le_sum_abs _ _
          _ = ∑ e : E, mass e *
              |expectedGapSum mass step reward causal oracle noise n (t + 1)
                (step x (executed (causal t x) (noise e)) e)| := by
              apply Finset.sum_congr rfl
              intro e _
              rw [abs_mul, abs_of_nonneg (hmass e)]
          _ ≤ _ := by
              apply Finset.sum_le_sum
              intro e _
              exact mul_le_mul_of_nonneg_left
                (ih (t + 1)
                  (step x (executed (causal t x) (noise e)) e)) (hmass e)
      exact (abs_add_le _ _).trans (add_le_add (hlocal n t x) hsum)

def weightedLocalBound {d : ℕ} {X Obs Mode : Type}
    (observe : X → Obs) (policy : Obs → Action d)
    (g : Mode → Action d) (trueMode : ℕ → Mode)
    (selected : ℕ → X → Mode) (L : ℕ → X → ℝ)
    (t : ℕ) (x : X) : ℝ :=
  2 * L t x * Real.sqrt
    (∑ j : Fin d,
      if g (trueMode t) j = g (selected t x) j then 0
      else (policy (observe x) j) ^ 2)

/-- The polarity-specific version of the conditional expected-return bound.
    `hLip` supplies oracle-Q sensitivity for all states and remaining horizons
    in this finite-noise model, a sufficient condition for the paper's weaker
    assumption restricted to causal occupancy at the given horizon.
    The factor of two and action-weighted mismatch are proved above, not
    added as independent hypotheses. -/
theorem polarity_expected_return_bound
    {d : ℕ} {X Obs E Mode : Type} [Fintype E]
    (mass : E → ℝ) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ) (noise : E → Action d)
    (observe : X → Obs) (policy : Obs → Action d)
    (g : Mode → Action d) (reference : Mode)
    (trueMode : ℕ → Mode) (selected : ℕ → X → Mode)
    (L : ℕ → X → ℝ)
    (hg : ∀ z, IsPolarity (g z))
    (hmass : ∀ e, 0 ≤ mass e)
    (hLip : ∀ remaining t x,
      |expectedQ mass step reward
        (expectedValue mass step reward
          (oracleNominal observe policy g reference) noise remaining (t + 1))
        noise x (causalNominal observe policy g reference trueMode selected t x) -
       expectedQ mass step reward
        (expectedValue mass step reward
          (oracleNominal observe policy g reference) noise remaining (t + 1))
        noise x (oracleNominal observe policy g reference t x)| ≤
        L t x * l2Norm
          (fun j => causalNominal observe policy g reference trueMode selected t x j -
            oracleNominal observe policy g reference t x j))
    (n t : ℕ) (x : X) :
    |expectedValue mass step reward
       (causalNominal observe policy g reference trueMode selected) noise n t x -
     expectedValue mass step reward
       (oracleNominal observe policy g reference) noise n t x| ≤
      expectedBoundSum mass step
        (causalNominal observe policy g reference trueMode selected) noise
        (weightedLocalBound observe policy g trueMode selected L) n t x := by
  apply expected_return_bound mass step reward
    (causalNominal observe policy g reference trueMode selected)
    (oracleNominal observe policy g reference) noise
    (weightedLocalBound observe policy g trueMode selected L) hmass
  intro remaining t x
  have hnorm :
      l2Norm
        (fun j => causalNominal observe policy g reference trueMode selected t x j -
          oracleNominal observe policy g reference t x j) =
        2 * Real.sqrt
          (∑ j : Fin d,
            if g (trueMode t) j = g (selected t x) j then 0
            else (policy (observe x) j) ^ 2) := by
    rw [causal_nominal_difference observe policy g reference trueMode selected t x]
    exact modeError_l2Norm _ _ _ _
      (hg (trueMode t)) (hg (selected t x)) (hg reference)
  have h := hLip remaining t x
  rw [hnorm] at h
  simpa [weightedLocalBound, mul_assoc, mul_left_comm, mul_comm] using h

/-- Lifting the simulator state with controller memory leaves oracle values
    unchanged if physical transitions, rewards, and oracle actions factor
    through the physical-state projection. `Y` can contain the entire causal
    history; its update may differ between actions. -/
theorem expectedValue_project
    {d : ℕ} {X Y E : Type} [Fintype E]
    (mass : E → ℝ) (noise : E → Action d) (project : Y → X)
    (stepX : X → Action d → E → X) (stepY : Y → Action d → E → Y)
    (rewardX : X → Action d → E → ℝ) (rewardY : Y → Action d → E → ℝ)
    (oracleX : ℕ → X → Action d) (oracleY : ℕ → Y → Action d)
    (hstep : ∀ y u e, project (stepY y u e) = stepX (project y) u e)
    (hreward : ∀ y u e, rewardY y u e = rewardX (project y) u e)
    (horacle : ∀ t y, oracleY t y = oracleX t (project y))
    (n t : ℕ) (y : Y) :
    expectedValue mass stepY rewardY oracleY noise n t y =
      expectedValue mass stepX rewardX oracleX noise n t (project y) := by
  induction n generalizing t y with
  | zero => rfl
  | succ n ih =>
      simp only [expectedValue, expectedQ]
      apply Finset.sum_congr rfl
      intro e _
      rw [ih, hstep, hreward, horacle]

end BAPR.ActionCompensation
