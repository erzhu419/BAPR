import ActionCompensationBelief
import ActionCompensationMeasure

/-!
# Closed-loop accumulation of BAPR belief risk

Occupancy measures must follow the causal controller's actual transition map.
The true conditional mode kernel is separate from the implemented belief.
States can contain continuous physical state and pre-action controller history.
No simulator-specific certificate or posterior-correctness assumption is used.
-/

noncomputable section

open MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace BAPR.ActionCompensation

def causalNext {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (noise : E → Action d)
    (causal : ℕ → X → Action d) (t : ℕ) (xe : X × E) : X :=
  step xe.1 (executed (causal t xe.1) (noise xe.2)) xe.2

def horizonValueDifference {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ) (noise : E → Action d)
    (causal oracle : ℕ → X → Action d) (H t : ℕ) (x : X) : ℝ :=
  measureValue μ step reward noise causal (H - t) t x -
    measureValue μ step reward noise oracle (H - t) t x

def horizonOracleGap {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ) (noise : E → Action d)
    (causal oracle : ℕ → X → Action d) (H t : ℕ) (x : X) : ℝ :=
  measureQ μ step reward noise
      (measureValue μ step reward noise oracle (H - (t + 1)) (t + 1)) x (causal t x) -
    measureQ μ step reward noise
      (measureValue μ step reward noise oracle (H - (t + 1)) (t + 1)) x (oracle t x)

theorem horizon_difference_step {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (causal oracle : ℕ → X → Action d)
    (hC : BellmanIntegrable μ step reward noise causal)
    (hO : BellmanIntegrable μ step reward noise oracle)
    (H t : ℕ) (ht : t < H) (x : X) :
    horizonValueDifference μ step reward noise causal oracle H t x =
      horizonOracleGap μ step reward noise causal oracle H t x +
        ∫ e, horizonValueDifference μ step reward noise causal oracle H (t + 1)
          (causalNext step noise causal t (x, e)) ∂μ := by
  have hn : H - t = (H - (t + 1)) + 1 := by omega
  unfold horizonValueDifference horizonOracleGap causalNext
  rw [hn, measure_return_difference μ step reward noise causal oracle hC hO,
    measureGapSum]
  simp_rw [← measure_return_difference μ step reward noise causal oracle hC hO]

/-- Actual causal occupancy transport, followed by Fubini. -/
theorem causal_occupancy_transport {d : ℕ} {X E : Type}
    [MeasurableSpace X] [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (ρ : ℕ → Measure X) [∀ t, IsProbabilityMeasure (ρ t)]
    (step : X → Action d → E → X) (noise : E → Action d)
    (causal : ℕ → X → Action d) (t : ℕ) (f : X → ℝ)
    (hnext : Measurable (causalNext step noise causal t))
    (hflow : ρ (t + 1) = (ρ t |>.prod μ).map (causalNext step noise causal t))
    (hf : Integrable f (ρ (t + 1))) :
    Integrable (fun x => ∫ e, f (causalNext step noise causal t (x, e)) ∂μ) (ρ t) ∧
      (∫ x, f x ∂ρ (t + 1)) =
        ∫ x, ∫ e, f (causalNext step noise causal t (x, e)) ∂μ ∂ρ t := by
  have hfmap : Integrable f
      ((ρ t |>.prod μ).map (causalNext step noise causal t)) := hflow ▸ hf
  have hprod : Integrable (fun xe => f (causalNext step noise causal t xe))
      (ρ t |>.prod μ) :=
    (integrable_map_measure hfmap.aestronglyMeasurable hnext.aemeasurable).mp hfmap
  refine ⟨hprod.integral_prod_left, ?_⟩
  rw [hflow, integral_map hnext.aemeasurable hfmap.aestronglyMeasurable]
  exact integral_prod _ hprod

/-- Only visited-state value differences need occupancy integrability here. -/
theorem horizonOracleGap_integrable {d : ℕ} {X E : Type}
    [MeasurableSpace X] [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (ρ : ℕ → Measure X) [∀ t, IsProbabilityMeasure (ρ t)]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (causal oracle : ℕ → X → Action d)
    (hC : BellmanIntegrable μ step reward noise causal)
    (hO : BellmanIntegrable μ step reward noise oracle)
    (H t : ℕ) (ht : t < H)
    (hnext : Measurable (causalNext step noise causal t))
    (hflow : ρ (t + 1) = (ρ t |>.prod μ).map (causalNext step noise causal t))
    (htval : Integrable
      (horizonValueDifference μ step reward noise causal oracle H t) (ρ t))
    (hnextval : Integrable
      (horizonValueDifference μ step reward noise causal oracle H (t + 1)) (ρ (t + 1))) :
    Integrable (horizonOracleGap μ step reward noise causal oracle H t) (ρ t) := by
  have htrans := causal_occupancy_transport μ ρ step noise causal t _ hnext hflow hnextval
  apply (htval.sub htrans.1).congr
  filter_upwards [] with x
  have h := horizon_difference_step μ step reward noise causal oracle hC hO H t ht x
  change horizonValueDifference μ step reward noise causal oracle H t x - _ = _
  linarith

/-- Performance difference expressed on the actual causal occupancy sequence. -/
theorem occupancy_return_difference {d : ℕ} {X E : Type}
    [MeasurableSpace X] [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (ρ : ℕ → Measure X) [∀ t, IsProbabilityMeasure (ρ t)]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (causal oracle : ℕ → X → Action d)
    (hC : BellmanIntegrable μ step reward noise causal)
    (hO : BellmanIntegrable μ step reward noise oracle) (H : ℕ)
    (hnext : ∀ t < H, Measurable (causalNext step noise causal t))
    (hflow : ∀ t < H,
      ρ (t + 1) = (ρ t |>.prod μ).map (causalNext step noise causal t))
    (hval : ∀ t ≤ H, Integrable
      (horizonValueDifference μ step reward noise causal oracle H t) (ρ t))
    (hiC : Integrable (measureValue μ step reward noise causal H 0) (ρ 0))
    (hiO : Integrable (measureValue μ step reward noise oracle H 0) (ρ 0)) :
    (∫ x, measureValue μ step reward noise causal H 0 x ∂ρ 0) -
      (∫ x, measureValue μ step reward noise oracle H 0 x ∂ρ 0) =
        ∑ t ∈ Finset.range H,
          ∫ x, horizonOracleGap μ step reward noise causal oracle H t x ∂ρ t := by
  let D (t : ℕ) := ∫ x, horizonValueDifference μ step reward noise causal oracle H t x ∂ρ t
  have hstep (t : ℕ) (ht : t < H) :
      (∫ x, horizonOracleGap μ step reward noise causal oracle H t x ∂ρ t) =
        D t - D (t + 1) := by
    have htrans := causal_occupancy_transport μ ρ step noise causal t _
      (hnext t ht) (hflow t ht) (hval (t + 1) (by omega))
    have hgap := horizonOracleGap_integrable μ ρ step reward noise causal oracle
      hC hO H t ht (hnext t ht) (hflow t ht) (hval t (by omega))
      (hval (t + 1) (by omega))
    have hpoint : D t =
        (∫ x, horizonOracleGap μ step reward noise causal oracle H t x ∂ρ t) +
          D (t + 1) := by
      dsimp [D]
      calc
        _ = ∫ x, horizonOracleGap μ step reward noise causal oracle H t x +
            (∫ e, horizonValueDifference μ step reward noise causal oracle H (t + 1)
              (causalNext step noise causal t (x, e)) ∂μ) ∂ρ t := by
          apply integral_congr_ae
          filter_upwards [] with x
          exact horizon_difference_step μ step reward noise causal oracle hC hO H t ht x
        _ = _ := by rw [integral_add hgap htrans.1, ← htrans.2]
    linarith
  rw [← integral_sub hiC hiO]
  change D 0 = _
  calc
    _ = D 0 - D H := by simp [D, horizonValueDifference, measureValue]
    _ = ∑ t ∈ Finset.range H, (D t - D (t + 1)) := by
      simpa using (Finset.sum_range_sub' D H).symm
    _ = _ := Finset.sum_congr rfl fun t ht => (hstep t (Finset.mem_range.mp ht)).symm

def kernelModeMass {Y Mode : Type} [MeasurableSpace Y] [MeasurableSpace Mode]
    (K : Kernel Y Mode) (y : Y) (z : Mode) : ℝ := (K y).real {z}

theorem kernelModeMass_isModeMass {Y Mode : Type}
    [MeasurableSpace Y] [MeasurableSpace Mode] [Fintype Mode] [MeasurableSingletonClass Mode]
    (K : Kernel Y Mode) [IsMarkovKernel K] (y : Y) :
    IsModeMass (kernelModeMass K y) := by
  refine ⟨fun z => measureReal_nonneg, ?_⟩
  simp [kernelModeMass]

/-- The mode sum is an actual conditional-kernel integral, not an arbitrary belief average. -/
theorem conditional_mode_integral {Y Mode : Type}
    [MeasurableSpace Y] [MeasurableSpace Mode] [Fintype Mode] [MeasurableSingletonClass Mode]
    (ν : Measure Y) [IsProbabilityMeasure ν]
    (K : Kernel Y Mode) [IsMarkovKernel K] (f : Y × Mode → ℝ)
    (hf : Integrable f (ν ⊗ₘ K)) :
    (∫ x, f x ∂(ν ⊗ₘ K)) =
      ∫ y, ∑ z, kernelModeMass K y z * f (y, z) ∂ν := by
  rw [Measure.integral_compProd hf]
  apply integral_congr_ae
  filter_upwards [] with y
  simpa [kernelModeMass, smul_eq_mul] using
    (integral_fintype (fun z => f (y, z)) (Integrable.of_finite) (μ := K y))

theorem conditional_mode_average_integrable {Y Mode : Type}
    [MeasurableSpace Y] [MeasurableSpace Mode] [Fintype Mode] [MeasurableSingletonClass Mode]
    (ν : Measure Y) [IsProbabilityMeasure ν]
    (K : Kernel Y Mode) [IsMarkovKernel K] (f : Y × Mode → ℝ)
    (hf : Integrable f (ν ⊗ₘ K)) :
    Integrable (fun y => ∑ z, kernelModeMass K y z * f (y, z)) ν := by
  have hint : Integrable (fun y => ∫ z, f (y, z) ∂K y) ν := by
    have h := (show Integrable f
      ((Kernel.const Unit ν ⊗ₖ Kernel.prodMkLeft Unit K) ()) from hf).integral_compProd
    simpa only [Kernel.const_apply, Kernel.prodMkLeft_apply'] using h
  apply hint.congr
  filter_upwards [] with y
  simpa [kernelModeMass, smul_eq_mul] using
    (integral_fintype (fun z => f (y, z)) (Integrable.of_finite) (μ := K y))

abbrev conditionalOccupancy {Y Mode : Type} [MeasurableSpace Y] [MeasurableSpace Mode]
    (ν : ℕ → Measure Y) (K : ℕ → Kernel Y Mode) (t : ℕ) : Measure (Y × Mode) :=
  ν t ⊗ₘ K t

def beliefCausalSignal {d : ℕ} {Y Mode : Type}
    (g : Mode → Action d) (reference : Mode) (q : Y → Action d)
    (selected : ℕ → Y → Mode) (t : ℕ) (x : Y × Mode) : Action d :=
  applyGain (g x.2) (compensate (g (selected t x.1)) (g reference) (q x.1))

def beliefOracleSignal {d : ℕ} {Y Mode : Type}
    (g : Mode → Action d) (reference : Mode) (q : Y → Action d)
    (_t : ℕ) (x : Y × Mode) : Action d := applyGain (g reference) (q x.1)

def calibratedLocalBudget {d : ℕ} {Y Mode : Type}
    [MeasurableSpace Y] [MeasurableSpace Mode] [Fintype Mode] [DecidableEq Mode]
    (K : ℕ → Kernel Y Mode) (b : ℕ → Y → Mode → ℝ)
    (g : Mode → Action d) (q : Y → Action d) (selected : ℕ → Y → Mode)
    (L : ℕ → Y → ℝ) (t : ℕ) (y : Y) : ℝ :=
  2 * L t y * Real.sqrt (beliefActionRisk (b t y) g (q y) (selected t y) +
    (∑ j, (q y j) ^ 2) * modeTotalVariation (kernelModeMass (K t) y) (b t y))

/-- Closed-loop return loss from belief-weighted action risk and calibration error.
    `K` is the true conditional kernel in the occupancy law, not the filter `b`. -/
theorem closed_loop_belief_return_bound {d : ℕ} {Y E Mode : Type}
    [MeasurableSpace Y] [MeasurableSpace E] [MeasurableSpace Mode]
    [Fintype Mode] [DecidableEq Mode] [MeasurableSingletonClass Mode]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (ν : ℕ → Measure Y) [∀ t, IsProbabilityMeasure (ν t)]
    (K : ℕ → Kernel Y Mode) [∀ t, IsMarkovKernel (K t)]
    (step : (Y × Mode) → Action d → E → (Y × Mode))
    (reward : (Y × Mode) → Action d → E → ℝ) (noise : E → Action d)
    (g : Mode → Action d) (reference : Mode) (q : Y → Action d)
    (selected : ℕ → Y → Mode) (b : ℕ → Y → Mode → ℝ) (L : ℕ → Y → ℝ)
    (hg : ∀ z, IsPolarity (g z))
    (hC : BellmanIntegrable μ step reward noise (beliefCausalSignal g reference q selected))
    (hO : BellmanIntegrable μ step reward noise (beliefOracleSignal g reference q)) (H : ℕ)
    (hnext : ∀ t < H,
      Measurable (causalNext step noise (beliefCausalSignal g reference q selected) t))
    (hflow : ∀ t < H, conditionalOccupancy ν K (t + 1) =
      ((conditionalOccupancy ν K t).prod μ).map
        (causalNext step noise (beliefCausalSignal g reference q selected) t))
    (hval : ∀ t ≤ H, Integrable (horizonValueDifference μ step reward noise
      (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q) H t)
      (conditionalOccupancy ν K t))
    (hiC : Integrable (measureValue μ step reward noise
      (beliefCausalSignal g reference q selected) H 0) (conditionalOccupancy ν K 0))
    (hiO : Integrable (measureValue μ step reward noise
      (beliefOracleSignal g reference q) H 0) (conditionalOccupancy ν K 0))
    (hb : ∀ t < H, ∀ y, IsModeMass (b t y))
    (hL : ∀ t < H, ∀ y, 0 ≤ L t y)
    (hsens : ∀ t < H, ∀ y z,
      |horizonOracleGap μ step reward noise
        (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q)
        H t (y, z)| ≤ L t y * l2Norm (modeError (g z) (g (selected t y)) (g reference) (q y)))
    (hbudget : ∀ t < H, Integrable (calibratedLocalBudget K b g q selected L t) (ν t)) :
    |(∫ x, measureValue μ step reward noise
        (beliefCausalSignal g reference q selected) H 0 x ∂conditionalOccupancy ν K 0) -
      (∫ x, measureValue μ step reward noise
        (beliefOracleSignal g reference q) H 0 x ∂conditionalOccupancy ν K 0)| ≤
      ∑ t ∈ Finset.range H, ∫ y, calibratedLocalBudget K b g q selected L t y ∂ν t := by
  rw [occupancy_return_difference μ (conditionalOccupancy ν K) step reward noise
    (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q)
    hC hO H hnext hflow hval hiC hiO]
  apply (Finset.abs_sum_le_sum_abs _ _).trans
  apply Finset.sum_le_sum
  intro t ht
  have htH := Finset.mem_range.mp ht
  let gap := horizonOracleGap μ step reward noise
    (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q) H t
  have hgap : Integrable gap (conditionalOccupancy ν K t) :=
    horizonOracleGap_integrable μ (conditionalOccupancy ν K) step reward noise
      (beliefCausalSignal g reference q selected) (beliefOracleSignal g reference q)
      hC hO H t htH (hnext t htH) (hflow t htH) (hval t (by omega))
      (hval (t + 1) (by omega))
  rw [conditional_mode_integral (ν t) (K t) gap hgap]
  apply abs_integral_le_integral_abs.trans
  apply integral_mono (conditional_mode_average_integrable (ν t) (K t) gap hgap).abs
    (hbudget t htH)
  intro y
  have h := belief_action_value_bound (kernelModeMass (K t) y) (b t y) g (q y)
    reference (selected t y)
    (fun z v => measureQ μ step reward noise
      (measureValue μ step reward noise (beliefOracleSignal g reference q)
        (H - (t + 1)) (t + 1)) (y, z) v) (L t y)
    (kernelModeMass_isModeMass (K t) y) (hb t htH y) hg (hL t htH y)
    (by
      intro z
      simpa only [horizonOracleGap, beliefCausalSignal, beliefOracleSignal] using hsens t htH y z)
  simpa only [calibratedLocalBudget, gap, horizonOracleGap,
    beliefCausalSignal, beliefOracleSignal] using h

end BAPR.ActionCompensation
