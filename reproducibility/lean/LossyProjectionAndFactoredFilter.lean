import ActionCompensation
import ActionCompensationFilter
import ActionCompensationMeasure

/-!
# Lossy Gain Projection, Tempered Log-Odds Caps, and Channel-Factored Inference

This module extends the BAPR formalization in three directions needed for
non-bijective actuator degradation and high-dimensional polarity spaces:

1. **Weighted box-constrained projection optimality and exact infeasibility residual**:
   for any invertible diagonal gain vector `g` (including lossy attenuation
   `|g i| < 1` where exact inverse commands leave `[-1, 1]^d`) and any positive
   diagonal sensitivity weights `w`, the clipped inverse command
   `clipAction (inverseCommand g v)` uniquely minimizes the weighted nominal
   mismatch `∑ i, w i * (g i * a i - v i)^2` over all feasible `a ∈ [-1, 1]^d`,
   leaving the exact squared infeasibility residual
   `∑ i, (max 0 (|v i| - |g i|))^2`.
2. **Two-term return decomposition (Infeasibility Gap + Causal Inference Loss)**:
   decomposes the finite-horizon return difference relative to ideal reference
   execution into an unavoidable physical saturation gap plus a causal mode
   inference gap.
3. **Dwell-independent switch recovery under tempering (`δ < 1` vs. `δ = 1`)
   and `O(d)` channel-factored MAP exactness on `{−1, +1}^d` (`M = 2^d`)**.
-/

noncomputable section

open scoped BigOperators
open Finset MeasureTheory

namespace BAPR.ActionCompensation

/-- Euclidean projection property of scalar clipping onto `[-1, 1]`. -/
theorem clipScalar_dist_le (x a : ℝ) (ha : -1 ≤ a ∧ a ≤ 1) :
    |clipScalar x - x| ≤ |a - x| := by
  by_cases hlo : x < -1
  · have hclip : clipScalar x = -1 := by
      simp [clipScalar, min_eq_left (by linarith : x ≤ 1),
        max_eq_left (by linarith : x ≤ -1)]
    rw [hclip, abs_of_pos (by linarith), abs_of_pos (by linarith [ha.1])]
    linarith [ha.1]
  · by_cases hhi : 1 < x
    · have hclip : clipScalar x = 1 := by
        simp [clipScalar, min_eq_right (by linarith : 1 ≤ x)]
      rw [hclip, abs_of_neg (by linarith), abs_of_neg (by linarith [ha.2])]
      linarith [ha.2]
    · have hbox : -1 ≤ x ∧ x ≤ 1 := ⟨not_lt.mp hlo, not_lt.mp hhi⟩
      rw [(clipScalar_self_iff x).mpr hbox, sub_self, abs_zero]
      exact abs_nonneg _

/-- Exact scalar distance from `x` to `[-1, 1]`. -/
theorem clipScalar_abs_error (x : ℝ) :
    |clipScalar x - x| = max 0 (|x| - 1) := by
  by_cases hlo : x < -1
  · have hclip : clipScalar x = -1 := by
      simp [clipScalar, min_eq_left (by linarith : x ≤ 1),
        max_eq_left (by linarith : x ≤ -1)]
    rw [hclip, abs_of_pos (by linarith), abs_of_neg (by linarith),
      max_eq_right (by linarith)]
    ring
  · by_cases hhi : 1 < x
    · have hclip : clipScalar x = 1 := by
        simp [clipScalar, min_eq_right (by linarith : 1 ≤ x)]
      rw [hclip, abs_of_neg (by linarith), abs_of_pos (by linarith),
        max_eq_right (by linarith)]
      ring
    · have hbox : -1 ≤ x ∧ x ≤ 1 := ⟨not_lt.mp hlo, not_lt.mp hhi⟩
      have habs : |x| ≤ 1 := abs_le.mpr hbox
      rw [(clipScalar_self_iff x).mpr hbox, sub_self, abs_zero,
        max_eq_left (by linarith)]

/-- Projected compensation command `clip(G⁻¹ v, -1, 1)`. -/
def projectedCommand {d : ℕ} (g target : Action d) : Action d :=
  clipAction (inverseCommand g target)

theorem projectedCommand_inBox {d : ℕ} (g target : Action d) :
    inBox (projectedCommand g target) := by
  intro j
  dsimp [projectedCommand, clipAction, clipScalar]
  exact ⟨le_max_left _ _, max_le (by norm_num) (min_le_right _ _)⟩

/-- Weighted box-projection optimality: for any invertible diagonal gain `g` and
    nonnegative sensitivity weights `w`, `projectedCommand g target` minimizes
    `∑ i, w i * (g i * a i - target i)^2` over all feasible `a ∈ [-1, 1]^d`. -/
theorem weightedBoxProjection_optimal {d : ℕ} (g target w a : Action d)
    (hg : ∀ i, g i ≠ 0) (hw : ∀ i, 0 ≤ w i) (ha : inBox a) :
    (∑ i, w i * (applyGain g (projectedCommand g target) i - target i) ^ 2) ≤
      ∑ i, w i * (applyGain g a i - target i) ^ 2 := by
  apply sum_le_sum
  intro i _
  have hdist := clipScalar_dist_le (inverseCommand g target i) (a i) (ha i)
  have hsq : (clipScalar (inverseCommand g target i) - inverseCommand g target i) ^ 2 ≤
      (a i - inverseCommand g target i) ^ 2 := by
    have hmul := mul_self_le_mul_self (abs_nonneg _) hdist
    simpa only [← pow_two, sq_abs] using hmul
  have hid1 : applyGain g (projectedCommand g target) i - target i =
      g i * (clipScalar (inverseCommand g target i) - inverseCommand g target i) := by
    dsimp [applyGain, projectedCommand, clipAction, inverseCommand]
    field_simp [hg i]
  have hid2 : applyGain g a i - target i =
      g i * (a i - inverseCommand g target i) := by
    dsimp [applyGain, inverseCommand]
    field_simp [hg i]
  rw [hid1, hid2, mul_pow, mul_pow, ← mul_assoc, ← mul_assoc]
  exact mul_le_mul_of_nonneg_left hsq (mul_nonneg (hw i) (sq_nonneg (g i)))

/-- Exact coordinatewise infeasibility residual under projected compensation. -/
theorem scalar_infeasibility_abs (g v : ℝ) (hg : g ≠ 0) :
    |g * clipScalar (v / g) - v| = max 0 (|v| - |g|) := by
  have hgpos : 0 < |g| := abs_pos.mpr hg
  have hfac : g * clipScalar (v / g) - v = g * (clipScalar (v / g) - v / g) := by
    field_simp [hg]
  rw [hfac, abs_mul, clipScalar_abs_error (v / g), abs_div]
  by_cases hex : |v| ≤ |g|
  · have hle : |v| / |g| - 1 ≤ 0 := by
      have hdiv : |v| / |g| ≤ 1 := (div_le_one hgpos).mpr hex
      linarith
    rw [max_eq_left hle, mul_zero, max_eq_left (by linarith)]
  · have hgt : |g| < |v| := not_le.mp hex
    have hge : 0 ≤ |v| / |g| - 1 := by
      have hdiv : 1 ≤ |v| / |g| := (one_le_div hgpos).mpr hgt.le
      linarith
    rw [max_eq_right hge, max_eq_right (by linarith)]
    field_simp [ne_of_gt hgpos]

/-- Exact squared Euclidean infeasibility residual across all `d` channels. -/
theorem lossyProjection_infeasibility_residual {d : ℕ} (g target : Action d)
    (hg : ∀ i, g i ≠ 0) :
    (∑ i, (applyGain g (projectedCommand g target) i - target i) ^ 2) =
      ∑ i, (max 0 (|target i| - |g i|)) ^ 2 := by
  apply sum_congr rfl
  intro i _
  have h := scalar_infeasibility_abs (g i) (target i) (hg i)
  dsimp [applyGain, projectedCommand, clipAction, inverseCommand]
  rw [← sq_abs (g i * clipScalar (target i / g i) - target i), h]

/-- When every channel satisfies `|target i| ≤ |g i|` (in particular whenever
    `|g i| ≥ 1` and `target ∈ [-1, 1]^d`), the infeasibility residual vanishes. -/
theorem lossyProjection_zero_of_sufficient_gain {d : ℕ} (g target : Action d)
    (hg : ∀ i, g i ≠ 0) (hsuff : ∀ i, |target i| ≤ |g i|) :
    (∑ i, (applyGain g (projectedCommand g target) i - target i) ^ 2) = 0 := by
  rw [lossyProjection_infeasibility_residual g target hg]
  apply sum_eq_zero
  intro i _
  rw [max_eq_left (by linarith [hsuff i])]
  norm_num

/-- Exact two-term decomposition of finite-horizon expected return under lossy
    actuator regimes into Infeasibility Gap + Causal Inference Loss. -/
theorem lossy_two_term_return_decomposition {d : ℕ} {X E : Type}
    [MeasurableSpace X] [MeasurableSpace E]
    (μ : Measure E) (ρ₀ : Measure X)
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d)
    (causal projOracle refOracle : ℕ → X → Action d) (H : ℕ)
    (hiC : Integrable (measureValue μ step reward noise causal H 0) ρ₀)
    (hiP : Integrable (measureValue μ step reward noise projOracle H 0) ρ₀)
    (hiR : Integrable (measureValue μ step reward noise refOracle H 0) ρ₀) :
    (∫ x, (measureValue μ step reward noise causal H 0 x -
        measureValue μ step reward noise refOracle H 0 x) ∂ρ₀) =
      (∫ x, (measureValue μ step reward noise projOracle H 0 x -
        measureValue μ step reward noise refOracle H 0 x) ∂ρ₀) +
      (∫ x, (measureValue μ step reward noise causal H 0 x -
        measureValue μ step reward noise projOracle H 0 x) ∂ρ₀) := by
  rw [integral_sub hiC hiR, integral_sub hiP hiR, integral_sub hiC hiP]
  ring

/-!
## Untempered (`δ = 1`) Linear Drift vs. Tempered (`δ < 1`) Uniform Log-Odds Cap
-/

def linearLogOdds : ℕ → ℝ → (ℕ → ℝ) → ℝ
  | 0, L₀, _ => L₀
  | t + 1, L₀, s => linearLogOdds t L₀ s + s t

theorem untempered_log_odds_dwell (T : ℕ) (L₀ m : ℝ) :
    linearLogOdds T L₀ (fun _ => m) = L₀ + (T : ℝ) * m := by
  induction T with
  | zero => simp [linearLogOdds]
  | succ t ih =>
    simp only [linearLogOdds, ih, Nat.cast_add, Nat.cast_one]
    ring

/-- Under `δ = 1` linear log-odds accumulation, a stationary dwell of length `T`
    with positive per-step margin `m > 0` requires strictly more than
    `(L₀ + T * m) / m'` opposing steps of magnitude `m' > 0` to flip sign. -/
theorem untempered_switch_delay_scales_with_dwell (T N : ℕ) (L₀ m m' : ℝ)
    (hm' : 0 < m')
    (hflip : linearLogOdds N (linearLogOdds T L₀ (fun _ => m)) (fun _ => -m') < 0) :
    (L₀ + (T : ℝ) * m) / m' < (N : ℝ) := by
  rw [untempered_log_odds_dwell, untempered_log_odds_dwell] at hflip
  rw [div_lt_iff₀ hm']
  linarith

def discountedLogOdds (δ : ℝ) : ℕ → ℝ → (ℕ → ℝ) → ℝ
  | 0, L₀, _ => L₀
  | t + 1, L₀, s => δ * discountedLogOdds δ t L₀ s + s t

/-- Under `0 ≤ δ < 1`, bounded single-step evidence `|s t| ≤ S` uniformly caps
    `|L_t| ≤ S / (1 - δ)` for ALL dwell durations `t`, preventing history lock-in. -/
theorem tempered_log_odds_uniform_cap (δ S L₀ : ℝ) (s : ℕ → ℝ)
    (hδ0 : 0 ≤ δ) (hδ1 : δ < 1)
    (h0 : |L₀| ≤ S / (1 - δ)) (hs : ∀ t, |s t| ≤ S) (t : ℕ) :
    |discountedLogOdds δ t L₀ s| ≤ S / (1 - δ) := by
  have h1m : 0 < 1 - δ := by linarith
  induction t with
  | zero => simpa [discountedLogOdds] using h0
  | succ k ih =>
    simp only [discountedLogOdds]
    calc
      |δ * discountedLogOdds δ k L₀ s + s k|
        ≤ |δ * discountedLogOdds δ k L₀ s| + |s k| := abs_add_le _ _
      _ = δ * |discountedLogOdds δ k L₀ s| + |s k| := by
        rw [abs_mul, abs_of_nonneg hδ0]
      _ ≤ δ * (S / (1 - δ)) + S := add_le_add (mul_le_mul_of_nonneg_left ih hδ0) (hs k)
      _ = S / (1 - δ) := by
        field_simp [ne_of_gt h1m]
        ring

/-!
## `O(d)` Channel-Factored MAP Exactness over the `2^d` Polarity Hypercube
-/

abbrev SignAssignment (d : ℕ) := Fin d → Bool

def jointChannelScore {d : ℕ} (φ : Fin d → Bool → ℝ) (z : SignAssignment d) : ℝ :=
  ∑ i, φ i (z i)

def channelFactoredMAP {d : ℕ} (φ : Fin d → Bool → ℝ) : SignAssignment d :=
  fun i => decide (φ i false ≤ φ i true)

/-- `O(d)` independent binary decisions achieve the exact global maximum of any
    coordinate-additive log-posterior over all `2^d` joint polarity modes. -/
theorem channelFactored_map_maximizes_joint {d : ℕ} (φ : Fin d → Bool → ℝ)
    (z : SignAssignment d) :
    jointChannelScore φ z ≤ jointChannelScore φ (channelFactoredMAP φ) := by
  apply sum_le_sum
  intro i _
  dsimp [channelFactoredMAP]
  by_cases h : φ i false ≤ φ i true
  · rw [decide_eq_true h]
    cases z i <;> simp [h]
  · rw [decide_eq_false h]
    have hgt : φ i true < φ i false := not_le.mp h
    cases z i <;> simp [hgt.le]

/-- When each channel's true sign has strict score superiority, the `O(d)`
    channel-factored selector recovers the unique global mode across `2^d` candidates. -/
theorem channelFactored_map_unique_recovery {d : ℕ} (φ : Fin d → Bool → ℝ)
    (zStar : SignAssignment d)
    (hmargin : ∀ i, φ i (!zStar i) < φ i (zStar i)) :
    channelFactoredMAP φ = zStar ∧
      ∀ z, z ≠ zStar → jointChannelScore φ z < jointChannelScore φ zStar := by
  refine ⟨?_, ?_⟩
  · funext i
    dsimp [channelFactoredMAP]
    have hi := hmargin i
    cases hz : zStar i
    · simp only [hz, Bool.not_false] at hi
      exact decide_eq_false (not_le.mpr hi)
    · simp only [hz, Bool.not_true] at hi
      exact decide_eq_true hi.le
  · intro z hne
    have hex : ∃ i, z i ≠ zStar i := by
      by_contra hall
      push_neg at hall
      exact hne (funext hall)
    rcases hex with ⟨i₀, hi₀⟩
    apply sum_lt_sum
    · intro i _
      by_cases heq : z i = zStar i
      · rw [heq]
      · have hnot : z i = !zStar i := by
          cases h1 : z i <;> cases h2 : zStar i <;> simp_all
        rw [hnot]
        exact (hmargin i).le
    · refine ⟨i₀, mem_univ i₀, ?_⟩
      have hnot : z i₀ = !zStar i₀ := by
        cases h1 : z i₀ <;> cases h2 : zStar i₀ <;> simp_all
      rw [hnot]
      exact hmargin i₀

end BAPR.ActionCompensation
