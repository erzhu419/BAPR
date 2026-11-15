import ActionCompensationReturn
import Mathlib.Probability.ConditionalProbability

/-!
# Random dwell windows on the actual rollout law

Window conditioning retains the joint dependence of incoming beliefs, commands
and prediction errors. The next, unobserved mode change is charged separately.
-/

noncomputable section

namespace BAPR.ActionCompensation

open Finset MeasureTheory ProbabilityTheory

/-- Start of the last constant-mode suffix among completed transitions `0..n`. -/
def dwellStart {Mode : Type} [DecidableEq Mode] (mode : ℕ → Mode) :
    (n : ℕ) → Fin (n + 1)
  | 0 => 0
  | n + 1 => if mode (n + 1) = mode n then (dwellStart mode n).castSucc
      else Fin.last (n + 1)

theorem dwellStart_persistent {Mode : Type} [DecidableEq Mode]
    (mode : ℕ → Mode) (n j : ℕ)
    (hstart : (dwellStart mode n).val ≤ j) (hend : j ≤ n) : mode j = mode n := by
  induction n with
  | zero =>
    have : j = 0 := by omega
    subst j
    rfl
  | succ n ih =>
    by_cases h : mode (n + 1) = mode n
    · simp only [dwellStart, if_pos h, Fin.val_castSucc] at hstart
      by_cases hj : j = n + 1
      · subst j; rfl
      · exact (ih hstart (by omega)).trans h.symm
    · simp only [dwellStart, if_neg h, Fin.val_last] at hstart
      have : j = n + 1 := by omega
      subst j
      rfl

theorem measurable_dwellStart {Ω Mode : Type}
    [MeasurableSpace Ω] [MeasurableSpace Mode] [DecidableEq Mode] [MeasurableEq Mode]
    (mode : ℕ → Ω → Mode) (hm : ∀ t, Measurable (mode t)) (n : ℕ) :
    Measurable (fun ω => dwellStart (fun t => mode t ω) n) := by
  induction n with
  | zero => exact measurable_const
  | succ n ih =>
    exact Measurable.ite (measurableSet_eq_fun (hm (n + 1)) (hm n))
      ((measurable_of_finite Fin.castSucc).comp ih) measurable_const

def observedWindowLabel {Ω Mode : Type} [DecidableEq Mode]
    (mode : ℕ → Ω → Mode) (n : ℕ) (ω : Ω) : Fin (n + 1) × Mode :=
  (dwellStart (fun t => mode t ω) n, mode n ω)

theorem measurable_observedWindowLabel {Ω Mode : Type}
    [MeasurableSpace Ω] [MeasurableSpace Mode] [DecidableEq Mode] [MeasurableEq Mode]
    (mode : ℕ → Ω → Mode) (hm : ∀ t, Measurable (mode t)) (n : ℕ) :
    Measurable (observedWindowLabel mode n) :=
  (measurable_dwellStart mode hm n).prodMk (hm n)

theorem observedWindowLabel_persistent {Ω Mode : Type} [DecidableEq Mode]
    (mode : ℕ → Ω → Mode) (n : ℕ) (l : Fin (n + 1) × Mode) (ω : Ω)
    (hl : observedWindowLabel mode n ω = l) (j : ℕ)
    (hs : l.1.val ≤ j) (hj : j ≤ n) : mode j ω = l.2 := by
  have hstart := congrArg (fun q : Fin (n + 1) × Mode => q.1.val) hl
  have hz := congrArg Prod.snd hl
  change (dwellStart (fun t => mode t ω) n).val = l.1.val at hstart
  exact (dwellStart_persistent (fun t => mode t ω) n j (by rw [hstart]; exact hs) hj).trans hz

/-- A continuing filter restarts from its actual incoming belief, not a fresh prior. -/
theorem temperedFilterPath_restart {Mode : Type} [Fintype Mode]
    (a k δ c : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ) (s N : ℕ) :
    temperedFilterPath a k δ c initial evidence (s + N) =
      temperedFilterPath a k δ c (temperedFilterPath a k δ c initial evidence s)
        (fun j => evidence (s + j)) N := by
  induction N with
  | zero => simp [temperedFilterPath]
  | succ N ih =>
    change temperedFilterStep a k δ c (temperedFilterPath a k δ c initial evidence (s + N))
      (evidence (s + N)) = _
    rw [ih]
    rfl

section Conditioning

variable {Ω Label : Type} [MeasurableSpace Ω] [MeasurableSpace Label]
variable [Fintype Label] [DiscreteMeasurableSpace Label]
variable (P : Measure Ω) [IsProbabilityMeasure P]

/-- Exact total probability, including zero-mass window cells. -/
theorem real_probability_by_fibers (label : Ω → Label) (hl : Measurable label)
    (bad : Set Ω) (_hb : MeasurableSet bad) :
    P.real bad = ∑ l, P.real (label ⁻¹' {l}) * (P[|label ⁻¹' {l}]).real bad := by
  have hsum := congrArg (fun μ : Measure Ω => μ bad) (sum_meas_smul_cond_fiber hl P)
  simp only [Measure.coe_finset_sum, Measure.coe_smul, Finset.sum_apply,
    Pi.smul_apply, smul_eq_mul] at hsum
  have hfin (l : Label) : P (label ⁻¹' {l}) * (P[|label ⁻¹' {l}]) bad ≠ ⊤ := by
    rw [mul_comm, cond_mul_eq_inter (hl (.singleton l))]
    exact measure_ne_top _ _
  have hreal := congrArg ENNReal.toReal hsum
  simpa only [measureReal_def, ENNReal.toReal_sum (fun l _ => hfin l),
    ENNReal.toReal_mul] using hreal.symm

/-- Cell-specific events need agree only on their own fiber. -/
theorem conditional_fiber_failure_bound (label : Ω → Label) (hl : Measurable label)
    (bad : Set Ω) (hb : MeasurableSet bad) (cellBad : Label → Set Ω) (β : Label → ℝ)
    (hfiber : ∀ l ω, label ω = l → (ω ∈ bad ↔ ω ∈ cellBad l))
    (hβ : ∀ l, P (label ⁻¹' {l}) ≠ 0 → (P[|label ⁻¹' {l}]).real (cellBad l) ≤ β l) :
    P.real bad ≤ ∑ l, P.real (label ⁻¹' {l}) * β l := by
  rw [real_probability_by_fibers P label hl bad hb]
  apply sum_le_sum
  intro l _
  by_cases hzero : P (label ⁻¹' {l}) = 0
  · simp [measureReal_def, hzero]
  · have heq : (label ⁻¹' {l}) ∩ bad = (label ⁻¹' {l}) ∩ cellBad l := by
      ext ω
      simp only [Set.mem_inter_iff, Set.mem_preimage, Set.mem_singleton_iff]
      exact and_congr_right fun h => hfiber l ω h
    have hcond : (P[|label ⁻¹' {l}]).real bad =
        (P[|label ⁻¹' {l}]).real (cellBad l) := by
      simp only [measureReal_def, cond_apply (hl (.singleton l)), heq]
    rw [hcond]
    exact mul_le_mul_of_nonneg_left (hβ l hzero) measureReal_nonneg

end Conditioning

/-- Recovery of a suffix implies recovery of the continuing filter at its endpoint. -/
theorem inverse_recovery_restart_event {Ω Head Mode : Type} {d : ℕ}
    [Fintype Head] [Fintype Mode] (a k δ c : ℝ)
    (initial : Ω → Mode → ℝ) (candidate : ℕ → Ω → Mode → Action d)
    (prediction : ℕ → Ω → Head → Action d) (variance : ℕ → Action d)
    (s N : ℕ) (z : Mode) :
    inverseRecoveryEvent a k δ c
      (fun ω => temperedFilterPath a k δ c (initial ω)
        (fun t => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) s)
      (fun t => candidate (s + t)) (fun t => prediction (s + t))
      (fun t => variance (s + t)) z N =
    inverseRecoveryEvent a k δ c initial candidate prediction variance z (s + N) := by
  ext ω
  simp only [inverseRecoveryEvent, Set.mem_setOf_eq]
  rw [temperedFilterPath_restart a k δ c _ _ s N]

/-- Moment bounds are on the actual conditional rollout law. Positive-mass
    conditioning supplies a probability law without an independent-window assumption. -/
theorem conditional_inverse_filter_restart_probability {Ω Head Mode : Type} {d : ℕ}
    [MeasurableSpace Ω] [Fintype Head] [Nonempty Head] [Fintype Mode] [Nonempty Mode]
    (P : Measure Ω) [IsProbabilityMeasure P] (cell : Set Ω) (hcell : P cell ≠ 0)
    (hd : 0 < d)
    (a k δ c m α : ℝ) (initial : Ω → Mode → ℝ)
    (candidate : ℕ → Ω → Mode → Action d) (prediction : ℕ → Ω → Head → Action d)
    (ε variance : ℕ → Action d) (s n : ℕ) (z : Mode)
    (M : Fin (n + 1) → Head → Action d)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 < c) (hm : 0 < m)
    (hb : ∀ ω, IsModeMass (initial ω))
    (hi : ∀ w, Measurable (fun ω => initial ω w))
    (hu : ∀ t w i, Measurable (fun ω => candidate t ω w i))
    (hp : ∀ t j i, Measurable (fun ω => prediction t ω j i))
    (hv : ∀ t, t < n + 1 → ∀ i, 0 < variance (s + t) i)
    (hε : ∀ t : Fin (n + 1), ∀ i, 0 < ε (s + t) i)
    (hint : ∀ t : Fin (n + 1), ∀ j i,
      Integrable (fun ω => (prediction (s + t) ω j i - candidate (s + t) ω z i) ^ 2)
        (P[|cell]))
    (hM : ∀ t : Fin (n + 1), ∀ j i,
      (∫ ω, (prediction (s + t) ω j i - candidate (s + t) ω z i) ^ 2 ∂P[|cell]) ≤ M t j i)
    (hexc : (P[|cell]).real (inverseExcitationEvent (fun t => candidate (s + t))
      (fun t => ε (s + t)) (fun t => variance (s + t)) z m (n + 1))ᶜ ≤ α)
    (hcert : 0 < filterMarginEnvelope δ (c * m)
      (Real.log (a + k) - Real.log k) (n + 1)) :
    (P[|cell]).real (inverseRecoveryEvent a k δ c initial candidate prediction variance
      z (s + (n + 1)))ᶜ ≤ α + ∑ t, ∑ j, ∑ i, M t j i / (ε (s + t) i) ^ 2 := by
  letI : IsProbabilityMeasure (P[|cell]) := cond_isProbabilityMeasure hcell
  let incoming := fun ω => temperedFilterPath a k δ c (initial ω)
    (fun t => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) s
  have hmass (ω : Ω) : IsModeMass (incoming ω) :=
    temperedFilterPath_mass a k δ c _ _ (hb ω) s
  have hmeas (w : Mode) : Measurable (fun ω => incoming ω w) :=
    measurable_temperedFilterPath a k δ c initial
      (fun t ω => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) hi
      (fun t w => measurable_inverseEvidence (candidate t) (prediction t) (variance t)
        w (hu t) (hp t)) s w
  have h := (inverse_filter_recovery_probability (P[|cell]) hd a k δ c m α incoming
    (fun t => candidate (s + t)) (fun t => prediction (s + t))
    (fun t => ε (s + t)) (fun t => variance (s + t)) z n M ha hk hδ hc hm
    hmass hmeas (fun t => hu (s + t)) (fun t => hp (s + t)) hv hε hint hM hexc hcert).1
  rw [inverse_recovery_restart_event] at h
  exact h

/-- Mix conditional recovery certificates on the actual random suffixes, then
    charge changes after the last observed transition. Uncertified cells may use `β = 1`. -/
theorem random_dwell_map_error_bound {Ω Head Mode : Type} {d : ℕ}
    [MeasurableSpace Ω] [MeasurableSpace Mode] [Fintype Head] [Fintype Mode]
    [DecidableEq Mode] [MeasurableSingletonClass Mode]
    (P : Measure Ω) [IsProbabilityMeasure P] (a k δ c : ℝ)
    (initial : Ω → Mode → ℝ) (candidate : ℕ → Ω → Mode → Action d)
    (prediction : ℕ → Ω → Head → Action d) (variance : ℕ → Action d)
    (mode : ℕ → Ω → Mode) (selected : Ω → Mode) (n : ℕ)
    (β : Fin (n + 1) × Mode → ℝ)
    (hi : ∀ w, Measurable (fun ω => initial ω w))
    (hu : ∀ t w i, Measurable (fun ω => candidate t ω w i))
    (hp : ∀ t j i, Measurable (fun ω => prediction t ω j i))
    (hm : ∀ t, Measurable (mode t))
    (hmap : ∀ ω w, temperedFilterPath a k δ c (initial ω)
        (fun t => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) (n + 1) w ≤
      temperedFilterPath a k δ c (initial ω)
        (fun t => inverseEvidence (candidate t ω) (prediction t ω) (variance t))
        (n + 1) (selected ω))
    (hβ : ∀ l : Fin (n + 1) × Mode,
      P ((observedWindowLabel mode n) ⁻¹' {l}) ≠ 0 →
      (P[|(observedWindowLabel mode n) ⁻¹' {l}]).real
        (inverseRecoveryEvent a k δ c
          (fun ω => temperedFilterPath a k δ c (initial ω)
            (fun t => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) l.1)
          (fun t => candidate (l.1 + t)) (fun t => prediction (l.1 + t))
          (fun t => variance (l.1 + t)) l.2 (n - l.1 + 1))ᶜ ≤ β l) :
    P.real {ω | selected ω ≠ mode (n + 1) ω} ≤
      (∑ l, P.real ((observedWindowLabel mode n) ⁻¹' {l}) * β l) +
        P.real {ω | mode (n + 1) ω ≠ mode n ω} := by
  let b := fun ω => temperedFilterPath a k δ c (initial ω)
    (fun t => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) (n + 1)
  let bad : Set Ω := {ω | ∀ w, w ≠ mode n ω → b ω w < b ω (mode n ω)}ᶜ
  have hb (w : Mode) : Measurable (fun ω => b ω w) :=
    measurable_temperedFilterPath a k δ c initial
      (fun t ω => inverseEvidence (candidate t ω) (prediction t ω) (variance t)) hi
      (fun t w => measurable_inverseEvidence (candidate t) (prediction t) (variance t)
        w (hu t) (hp t)) (n + 1) w
  have hbad : MeasurableSet bad := by
    have hcoord : Measurable (fun ω => b ω (mode n ω)) := by
      have heq : (fun ω => b ω (mode n ω)) =
          (fun ω => ∑ w, if mode n ω = w then b ω w else 0) := by
        funext ω
        simp
      rw [heq]
      exact Finset.measurable_sum _ fun w _ => Measurable.ite
        (measurableSet_eq_fun (hm n) measurable_const) (hb w) measurable_const
    simp only [bad, Set.setOf_forall]
    apply MeasurableSet.compl
    apply MeasurableSet.iInter
    intro w
    convert (measurableSet_eq_fun measurable_const (hm n)).union
      (measurableSet_lt (hb w) hcoord) using 1
    ext ω
    simp only [Set.mem_setOf_eq, Set.mem_union]
    tauto
  have hmix : P.real bad ≤ ∑ l, P.real ((observedWindowLabel mode n) ⁻¹' {l}) * β l := by
    apply conditional_fiber_failure_bound P (observedWindowLabel mode n)
      (measurable_observedWindowLabel mode hm n) bad hbad
      (fun l => (inverseRecoveryEvent a k δ c initial candidate prediction variance
        l.2 (n + 1))ᶜ) β
    · intro l ω hlabel
      have hz : mode n ω = l.2 := congrArg Prod.snd hlabel
      simp only [bad, inverseRecoveryEvent, Set.mem_compl_iff, Set.mem_setOf_eq, b, hz]
    · intro l hl
      have hend : l.1.val + (n - l.1.val + 1) = n + 1 := by omega
      simpa only [inverse_recovery_restart_event, hend] using hβ l hl
  exact (map_mismatch_probability P b selected (mode (n + 1)) (mode n) hmap).trans
    (add_le_add hmix le_rfl)

/-- The probability interface used by the return theorem is the pushforward
    occupancy of this same path law. -/
theorem occupancy_mode_error_eq {Ω Y Mode : Type}
    [MeasurableSpace Ω] [MeasurableSpace Y] [MeasurableSpace Mode]
    [MeasurableEq Mode] (P : Measure Ω) (decision : Ω → Y × Mode)
    (selected : Y → Mode) (hd : Measurable decision) (hs : Measurable selected) :
    (P.map decision).real {x | selected x.1 ≠ x.2} =
      P.real {ω | selected (decision ω).1 ≠ (decision ω).2} := by
  exact map_measureReal_apply hd
    (measurableSet_eq_fun (hs.comp measurable_fst) measurable_snd).compl

theorem occupancy_mode_error_bound_of_path {Ω Y Mode : Type}
    [MeasurableSpace Ω] [MeasurableSpace Y] [MeasurableSpace Mode] [MeasurableEq Mode]
    (P : Measure Ω) (ρ : Measure (Y × Mode)) (decision : Ω → Y × Mode)
    (selected : Y → Mode) (current : Ω → Mode) (β : ℝ)
    (hd : Measurable decision) (hs : Measurable selected) (hρ : ρ = P.map decision)
    (hc : ∀ ω, (decision ω).2 = current ω)
    (hp : P.real {ω | selected (decision ω).1 ≠ current ω} ≤ β) :
    ρ.real {x | selected x.1 ≠ x.2} ≤ β := by
  rw [hρ, occupancy_mode_error_eq P decision selected hd hs]
  simpa only [hc] using hp

end BAPR.ActionCompensation
