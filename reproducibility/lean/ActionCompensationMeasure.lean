import ActionCompensation

/-!
# Continuous-noise action compensation

Bochner-integral counterparts of the finite-noise return identities. No
discretization of actuator noise is used. `X` may include causal controller
memory. Integrability is an explicit hypothesis, not inferred from Gaussianity.
-/

noncomputable section

open MeasureTheory
open scoped NNReal

namespace BAPR.ActionCompensation

def measureQ {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ) (noise : E → Action d)
    (nextValue : X → ℝ) (x : X) (v : Action d) : ℝ :=
  ∫ e, reward x (executed v (noise e)) e +
    nextValue (step x (executed v (noise e)) e) ∂μ

def measureValue {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ) (noise : E → Action d)
    (policy : ℕ → X → Action d) (n t : ℕ) (x : X) : ℝ :=
  match n with
  | 0 => 0
  | remaining + 1 => measureQ μ step reward noise
      (measureValue μ step reward noise policy remaining (t + 1)) x (policy t x)

/-- Integrability of the Bellman integrands, including counterfactual first
    signals. This sufficient condition is stronger than integrability only
    along the particular causal and oracle trajectories. -/
def BellmanIntegrable {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ) (noise : E → Action d)
    (policy : ℕ → X → Action d) : Prop :=
  ∀ n t x v, Integrable (fun e => reward x (executed v (noise e)) e +
    measureValue μ step reward noise policy n (t + 1)
      (step x (executed v (noise e)) e)) μ

def measureGapSum {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) (step : X → Action d → E → X)
    (reward : X → Action d → E → ℝ) (noise : E → Action d)
    (causal oracle : ℕ → X → Action d) (n t : ℕ) (x : X) : ℝ :=
  match n with
  | 0 => 0
  | remaining + 1 =>
      let next := measureValue μ step reward noise oracle remaining (t + 1)
      (measureQ μ step reward noise next x (causal t x) -
       measureQ μ step reward noise next x (oracle t x)) +
      ∫ e, measureGapSum μ step reward noise causal oracle remaining (t + 1)
        (step x (executed (causal t x) (noise e)) e) ∂μ

/-- Continuous-noise expected performance difference, using oracle-Q values
    averaged over fresh noise rather than pathwise clairvoyant Q values. -/
theorem measure_return_difference {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (causal oracle : ℕ → X → Action d)
    (hC : BellmanIntegrable μ step reward noise causal)
    (hO : BellmanIntegrable μ step reward noise oracle)
    (n t : ℕ) (x : X) :
    measureValue μ step reward noise causal n t x -
      measureValue μ step reward noise oracle n t x =
        measureGapSum μ step reward noise causal oracle n t x := by
  induction n generalizing t x with
  | zero => simp [measureValue, measureGapSum]
  | succ n ih =>
      have hdiff :
          (∫ e, measureValue μ step reward noise causal n (t + 1)
              (step x (executed (causal t x) (noise e)) e) -
            measureValue μ step reward noise oracle n (t + 1)
              (step x (executed (causal t x) (noise e)) e) ∂μ) =
          measureQ μ step reward noise
            (measureValue μ step reward noise causal n (t + 1)) x (causal t x) -
          measureQ μ step reward noise
            (measureValue μ step reward noise oracle n (t + 1)) x (causal t x) := by
        rw [measureQ, measureQ, ← integral_sub (hC n t x (causal t x))
          (hO n t x (causal t x))]
        apply integral_congr_ae
        filter_upwards [] with e
        ring
      simp only [measureValue, measureGapSum]
      simp_rw [← ih]
      rw [hdiff]
      ring

def measureBoundSum {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) (step : X → Action d → E → X) (noise : E → Action d)
    (causal : ℕ → X → Action d) (bound : ℕ → X → ℝ)
    (n t : ℕ) (x : X) : ℝ :=
  match n with
  | 0 => 0
  | remaining + 1 => bound t x +
      ∫ e, measureBoundSum μ step noise causal bound remaining (t + 1)
        (step x (executed (causal t x) (noise e)) e) ∂μ

theorem measure_return_bound {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (causal oracle : ℕ → X → Action d)
    (hC : BellmanIntegrable μ step reward noise causal)
    (hO : BellmanIntegrable μ step reward noise oracle)
    (bound : ℕ → X → ℝ)
    (hbound : ∀ n t x, Integrable (fun e =>
      measureBoundSum μ step noise causal bound n (t + 1)
        (step x (executed (causal t x) (noise e)) e)) μ)
    (hlocal : ∀ n t x,
      |measureQ μ step reward noise
          (measureValue μ step reward noise oracle n (t + 1)) x (causal t x) -
        measureQ μ step reward noise
          (measureValue μ step reward noise oracle n (t + 1)) x (oracle t x)| ≤
        bound t x)
    (n t : ℕ) (x : X) :
    |measureValue μ step reward noise causal n t x -
      measureValue μ step reward noise oracle n t x| ≤
        measureBoundSum μ step noise causal bound n t x := by
  rw [measure_return_difference μ step reward noise causal oracle hC hO]
  induction n generalizing t x with
  | zero => simp [measureGapSum, measureBoundSum]
  | succ n ih =>
      have hint : Integrable (fun e =>
          measureGapSum μ step reward noise causal oracle n (t + 1)
            (step x (executed (causal t x) (noise e)) e)) μ := by
        apply ((hC n t x (causal t x)).sub (hO n t x (causal t x))).congr
        filter_upwards [] with e
        simp only [Pi.sub_apply]
        rw [← measure_return_difference μ step reward noise causal oracle hC hO]
        ring
      simp only [measureGapSum, measureBoundSum]
      apply (abs_add_le _ _).trans
      apply add_le_add (hlocal n t x)
      exact abs_integral_le_integral_abs.trans
        (integral_mono hint.abs (hbound n t x) (fun e => ih (t + 1) _))

/-- Initial-state distributions can also be continuous. -/
theorem measure_initial_return_difference
    {d : ℕ} {X E : Type} [MeasurableSpace X] [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (ρ : Measure X) [IsProbabilityMeasure ρ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (causal oracle : ℕ → X → Action d)
    (hC : BellmanIntegrable μ step reward noise causal)
    (hO : BellmanIntegrable μ step reward noise oracle)
    (n t : ℕ)
    (hiC : Integrable (measureValue μ step reward noise causal n t) ρ)
    (hiO : Integrable (measureValue μ step reward noise oracle n t) ρ) :
    (∫ x, measureValue μ step reward noise causal n t x ∂ρ) -
      (∫ x, measureValue μ step reward noise oracle n t x ∂ρ) =
        ∫ x, measureGapSum μ step reward noise causal oracle n t x ∂ρ := by
  rw [← integral_sub hiC hiO]
  apply integral_congr_ae
  filter_upwards [] with x
  exact measure_return_difference μ step reward noise causal oracle hC hO n t x

/-- The experimental action-weighted bound, now under an arbitrary probability
    measure on noise, including a continuous Gaussian measure. -/
theorem measure_polarity_return_bound
    {d : ℕ} {X Obs E Mode : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (observe : X → Obs) (policy : Obs → Action d)
    (g : Mode → Action d) (reference : Mode)
    (trueMode : ℕ → Mode) (selected : ℕ → X → Mode) (L : ℕ → X → ℝ)
    (hg : ∀ z, IsPolarity (g z))
    (hC : BellmanIntegrable μ step reward noise
      (causalNominal observe policy g reference trueMode selected))
    (hO : BellmanIntegrable μ step reward noise
      (oracleNominal observe policy g reference))
    (hbound : ∀ n t x, Integrable (fun e =>
      measureBoundSum μ step noise
        (causalNominal observe policy g reference trueMode selected)
        (weightedLocalBound observe policy g trueMode selected L) n (t + 1)
        (step x (executed
          (causalNominal observe policy g reference trueMode selected t x)
          (noise e)) e)) μ)
    (hLip : ∀ n t x,
      |measureQ μ step reward noise
          (measureValue μ step reward noise (oracleNominal observe policy g reference)
            n (t + 1)) x (causalNominal observe policy g reference trueMode selected t x) -
        measureQ μ step reward noise
          (measureValue μ step reward noise (oracleNominal observe policy g reference)
            n (t + 1)) x (oracleNominal observe policy g reference t x)| ≤
        L t x * l2Norm (fun j =>
          causalNominal observe policy g reference trueMode selected t x j -
          oracleNominal observe policy g reference t x j))
    (n t : ℕ) (x : X) :
    |measureValue μ step reward noise
        (causalNominal observe policy g reference trueMode selected) n t x -
      measureValue μ step reward noise
        (oracleNominal observe policy g reference) n t x| ≤
      measureBoundSum μ step noise
        (causalNominal observe policy g reference trueMode selected)
        (weightedLocalBound observe policy g trueMode selected L) n t x := by
  apply measure_return_bound μ step reward noise
    (causalNominal observe policy g reference trueMode selected)
    (oracleNominal observe policy g reference) hC hO
    (weightedLocalBound observe policy g trueMode selected L) hbound
  intro remaining k s
  have h := hLip remaining k s
  rw [causal_nominal_difference observe policy g reference trueMode selected k s,
    modeError_l2Norm _ _ _ _ (hg (trueMode k)) (hg (selected k s)) (hg reference)] at h
  simpa [weightedLocalBound, mul_assoc, mul_left_comm, mul_comm] using h

/-- Averaging a valid pointwise bound over a common continuous reset law. -/
theorem measure_initial_return_bound
    {d : ℕ} {X E : Type} [MeasurableSpace X] [MeasurableSpace E]
    (μ : Measure E) (ρ : Measure X)
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (causal oracle : ℕ → X → Action d)
    (bound : ℕ → X → ℝ) (n t : ℕ)
    (hiC : Integrable (measureValue μ step reward noise causal n t) ρ)
    (hiO : Integrable (measureValue μ step reward noise oracle n t) ρ)
    (hiB : Integrable (measureBoundSum μ step noise causal bound n t) ρ)
    (hlocal : ∀ᵐ x ∂ρ,
      |measureValue μ step reward noise causal n t x -
        measureValue μ step reward noise oracle n t x| ≤
        measureBoundSum μ step noise causal bound n t x) :
    |(∫ x, measureValue μ step reward noise causal n t x ∂ρ) -
      (∫ x, measureValue μ step reward noise oracle n t x ∂ρ)| ≤
        ∫ x, measureBoundSum μ step noise causal bound n t x ∂ρ := by
  rw [← integral_sub hiC hiO]
  exact abs_integral_le_integral_abs.trans
    (integral_mono_ae (hiC.sub hiO).abs hiB hlocal)

theorem measureValue_project
    {d : ℕ} {X Y E : Type} [MeasurableSpace E]
    (μ : Measure E) (noise : E → Action d) (project : Y → X)
    (stepX : X → Action d → E → X) (stepY : Y → Action d → E → Y)
    (rewardX : X → Action d → E → ℝ) (rewardY : Y → Action d → E → ℝ)
    (oracleX : ℕ → X → Action d) (oracleY : ℕ → Y → Action d)
    (hstep : ∀ y u e, project (stepY y u e) = stepX (project y) u e)
    (hreward : ∀ y u e, rewardY y u e = rewardX (project y) u e)
    (horacle : ∀ t y, oracleY t y = oracleX t (project y))
    (n t : ℕ) (y : Y) :
    measureValue μ stepY rewardY noise oracleY n t y =
      measureValue μ stepX rewardX noise oracleX n t (project y) := by
  induction n generalizing t y with
  | zero => rfl
  | succ n ih =>
      simp only [measureValue, measureQ]
      apply integral_congr_ae
      filter_upwards [] with e
      rw [ih, hstep, hreward, horacle]

/-- Independent actuator noise with exactly the paper's standard deviation
    0.02; `gaussianReal` takes variance, hence 1/2500 rather than 1/50. -/
def actuatorGaussian (d : ℕ) : Measure (Action d) :=
  Measure.pi (fun _ : Fin d => ProbabilityTheory.gaussianReal 0 (1 / 2500))

instance actuatorGaussian_probability (d : ℕ) :
    IsProbabilityMeasure (actuatorGaussian d) := by
  unfold actuatorGaussian
  infer_instance

theorem actuatorGaussian_marginal {d : ℕ} (j : Fin d) :
    (actuatorGaussian d).map (fun v => v j) =
      ProbabilityTheory.gaussianReal 0 (1 / 2500) := by
  change (Measure.pi (fun _ : Fin d =>
    ProbabilityTheory.gaussianReal 0 (1 / 2500))).map (Function.eval j) = _
  simp [Measure.pi_map_eval]

theorem gaussian_return_difference {d : ℕ} {X : Type}
    (step : X → Action d → Action d → X)
    (reward : X → Action d → Action d → ℝ)
    (causal oracle : ℕ → X → Action d)
    (hC : BellmanIntegrable (actuatorGaussian d) step reward id causal)
    (hO : BellmanIntegrable (actuatorGaussian d) step reward id oracle)
    (n t : ℕ) (x : X) :
    measureValue (actuatorGaussian d) step reward id causal n t x -
      measureValue (actuatorGaussian d) step reward id oracle n t x =
        measureGapSum (actuatorGaussian d) step reward id causal oracle n t x :=
  measure_return_difference _ step reward id causal oracle hC hO n t x

/-- The pathwise equivalence also gives equality of probability laws under
    continuous noise. Measurability ensures that the pushforward is a genuine
    probability law, not a map of a nonmeasurable function. -/
theorem polarity_return_distribution
    {d : ℕ} {X Obs E Mode Ω : Type} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (observe : X → Obs) (policy : Obs → Action d)
    (g : Mode → Action d) (r : Mode) (schedule : ℕ → Mode)
    (hg : ∀ z, IsPolarity (g z))
    (noise : Ω → ℕ → Action d) (event : Ω → ℕ → E) (initial : Ω → X)
    (n t : ℕ)
    (hm : AEMeasurable (fun ω => (rollout step reward
      (fun _ s => applyGain (g r) (policy (observe s)))
      (noise ω) (event ω) n t (initial ω)).2) μ) :
    μ.map (fun ω => (rollout step reward
      (fun k s => applyGain (g (schedule k))
        (compensate (g (schedule k)) (g r) (policy (observe s))))
      (noise ω) (event ω) n t (initial ω)).2) =
    μ.map (fun ω => (rollout step reward
      (fun _ s => applyGain (g r) (policy (observe s)))
      (noise ω) (event ω) n t (initial ω)).2) ∧
    IsProbabilityMeasure (μ.map (fun ω => (rollout step reward
      (fun _ s => applyGain (g r) (policy (observe s)))
      (noise ω) (event ω) n t (initial ω)).2)) := by
  constructor
  · apply Measure.map_congr
    filter_upwards [] with ω
    exact congrArg Prod.snd (arbitrary_polarity_switching step reward observe
      policy g r schedule hg (noise ω) (event ω) n t (initial ω))
  · exact μ.isProbabilityMeasure_map hm

/-- Finite independent noise paths, represented recursively to expose the
    first draw and the remaining draws to Fubini's theorem. -/
def NoisePath (E : Type) : ℕ → Type
  | 0 => Unit
  | n + 1 => E × NoisePath E n

instance noisePathMeasurable {E : Type} [MeasurableSpace E] (n : ℕ) :
    MeasurableSpace (NoisePath E n) := by
  induction n with
  | zero => exact inferInstanceAs (MeasurableSpace Unit)
  | succ n ih =>
      letI := ih
      exact inferInstanceAs (MeasurableSpace (E × NoisePath E n))

def noisePathLaw {E : Type} [MeasurableSpace E] (μ : Measure E) :
    (n : ℕ) → Measure (NoisePath E n)
  | 0 => Measure.dirac ()
  | n + 1 => μ.prod (noisePathLaw μ n)

instance noisePathLaw_probability {E : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ] (n : ℕ) :
    IsProbabilityMeasure (noisePathLaw μ n) := by
  induction n with
  | zero => exact inferInstanceAs (IsProbabilityMeasure (Measure.dirac ()))
  | succ n ih =>
      letI := ih
      exact inferInstanceAs (IsProbabilityMeasure (μ.prod (noisePathLaw μ n)))

def noisePathReturn {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (policy : ℕ → X → Action d) :
    (n : ℕ) → ℕ → X → NoisePath E n → ℝ
  | 0, _, _, _ => 0
  | n + 1, t, x, ω =>
      let u := executed (policy t x) (noise ω.1)
      reward x u ω.1 + noisePathReturn step reward noise policy
        n (t + 1) (step x u ω.1) ω.2

/-- The recursive value is the actual integral of cumulative rollout reward
    under independently drawn noise, not merely a formal Bellman expression. -/
theorem measureValue_eq_path_integral {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (policy : ℕ → X → Action d)
    (hint : ∀ n t x, Integrable (noisePathReturn step reward noise policy n t x)
      (noisePathLaw μ n)) (n t : ℕ) (x : X) :
    measureValue μ step reward noise policy n t x =
      ∫ ω, noisePathReturn step reward noise policy n t x ω ∂noisePathLaw μ n := by
  induction n generalizing t x with
  | zero => simp [measureValue, noisePathReturn]
  | succ n ih =>
      change measureValue μ step reward noise policy (n + 1) t x =
        ∫ ω : E × NoisePath E n,
          noisePathReturn step reward noise policy (n + 1) t x ω
            ∂μ.prod (noisePathLaw μ n)
      rw [integral_prod (fun ω : E × NoisePath E n =>
        noisePathReturn step reward noise policy (n + 1) t x ω)
        (hint (n + 1) t x)]
      simp only [measureValue, measureQ]
      apply integral_congr_ae
      filter_upwards [] with e
      simp only [noisePathReturn]
      rw [integral_add (integrable_const _)
        (hint n (t + 1) (step x (executed (policy t x) (noise e)) e))]
      simp only [integral_const, probReal_univ, one_smul]
      rw [ih]

def beliefProcess {Ω B T : Type} (update : B → T → B) (initial : B)
    (transition : ℕ → Ω → T) : ℕ → Ω → B
  | 0, _ => initial
  | t + 1, ω => update (beliefProcess update initial transition t ω) (transition t ω)

/-- Transition `t` is available in `F (t+1)`, not before action `t`.
    The belief used for that action is measurable in `F t`. -/
theorem measurable_beliefProcess {Ω B T : Type}
    [MeasurableSpace B] [MeasurableSpace T]
    (F : ℕ → MeasurableSpace Ω) (hF : Monotone F)
    (update : B → T → B) (initial : B) (transition : ℕ → Ω → T)
    (hupdate : Measurable (fun p : B × T => update p.1 p.2))
    (htransition : ∀ t, Measurable[F (t + 1)] (transition t)) (t : ℕ) :
    Measurable[F t] (beliefProcess update initial transition t) := by
  induction t with
  | zero => exact measurable_const
  | succ t ih =>
      exact hupdate.comp
        ((ih.mono (hF (Nat.le_succ t)) le_rfl).prodMk (htransition t))

theorem measurable_causal_action {d : ℕ} {Ω B T S : Type}
    [MeasurableSpace B] [MeasurableSpace T] [MeasurableSpace S]
    (F : ℕ → MeasurableSpace Ω) (hF : Monotone F)
    (update : B → T → B) (initial : B) (transition : ℕ → Ω → T)
    (state : ℕ → Ω → S) (select : B → S → Action d)
    (hupdate : Measurable (fun p : B × T => update p.1 p.2))
    (htransition : ∀ t, Measurable[F (t + 1)] (transition t))
    (hstate : ∀ t, Measurable[F t] (state t))
    (hselect : Measurable (fun p : B × S => select p.1 p.2)) (t : ℕ) :
    Measurable[F t] (fun ω =>
      select (beliefProcess update initial transition t ω) (state t ω)) := by
  exact hselect.comp ((measurable_beliefProcess F hF update initial transition
    hupdate htransition t).prodMk (hstate t))

/-- A uniform reward bound controls finite-horizon values for every probability
    noise law. This is a sufficient condition, not a claim about Brax rewards. -/
theorem measureValue_abs_le_of_bounded_reward
    {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (policy : ℕ → X → Action d)
    (R : ℝ) (hr : ∀ x u e, |reward x u e| ≤ R)
    (n t : ℕ) (x : X) :
    |measureValue μ step reward noise policy n t x| ≤ n * R := by
  induction n generalizing t x with
  | zero => simp [measureValue]
  | succ n ih =>
      change |∫ e, reward x (executed (policy t x) (noise e)) e +
        measureValue μ step reward noise policy n (t + 1)
          (step x (executed (policy t x) (noise e)) e) ∂μ| ≤ (n + 1 : ℕ) * R
      have hb : ∀ᵐ e ∂μ, ‖reward x (executed (policy t x) (noise e)) e +
          measureValue μ step reward noise policy n (t + 1)
            (step x (executed (policy t x) (noise e)) e)‖ ≤ (n + 1 : ℕ) * R := by
        filter_upwards [] with e
        rw [Real.norm_eq_abs]
        calc
          _ ≤ |reward x (executed (policy t x) (noise e)) e| +
              |measureValue μ step reward noise policy n (t + 1)
                (step x (executed (policy t x) (noise e)) e)| := abs_add_le _ _
          _ ≤ R + n * R := add_le_add (hr _ _ _) (ih _ _)
          _ = _ := by push_cast; ring
      simpa only [Real.norm_eq_abs, probReal_univ, mul_one] using
        norm_integral_le_of_norm_le_const hb

theorem bellmanIntegrable_of_bounded_reward
    {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (policy : ℕ → X → Action d)
    (R : ℝ) (hr : ∀ x u e, |reward x u e| ≤ R)
    (hm : ∀ n t x v, AEStronglyMeasurable (fun e =>
      reward x (executed v (noise e)) e +
        measureValue μ step reward noise policy n (t + 1)
          (step x (executed v (noise e)) e)) μ) :
    BellmanIntegrable μ step reward noise policy := by
  intro n t x v
  apply (integrable_const (R + n * R)).mono' (hm n t x v)
  filter_upwards [] with e
  rw [Real.norm_eq_abs]
  exact (abs_add_le _ _).trans (add_le_add (hr _ _ _)
    (measureValue_abs_le_of_bounded_reward μ step reward noise policy R hr n (t + 1) _))

theorem noisePathReturn_abs_le_of_bounded_reward
    {d : ℕ} {X E : Type}
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (policy : ℕ → X → Action d)
    (R : ℝ) (hr : ∀ x u e, |reward x u e| ≤ R)
    (n t : ℕ) (x : X) (ω : NoisePath E n) :
    |noisePathReturn step reward noise policy n t x ω| ≤ n * R := by
  induction n generalizing t x with
  | zero => simp [noisePathReturn]
  | succ n ih =>
      simp only [noisePathReturn]
      apply (abs_add_le _ _).trans
      calc
        _ ≤ R + n * R := add_le_add (hr _ _ _) (ih _ _ ω.2)
        _ = _ := by push_cast; ring

theorem noisePathReturn_integrable_of_bounded_reward
    {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (policy : ℕ → X → Action d)
    (R : ℝ) (hr : ∀ x u e, |reward x u e| ≤ R)
    (n t : ℕ) (x : X)
    (hm : AEStronglyMeasurable (noisePathReturn step reward noise policy n t x)
      (noisePathLaw μ n)) :
    Integrable (noisePathReturn step reward noise policy n t x) (noisePathLaw μ n) := by
  apply (integrable_const ((n : ℝ) * R)).mono' hm
  filter_upwards [] with ω
  simpa only [Real.norm_eq_abs] using
    noisePathReturn_abs_le_of_bounded_reward step reward noise policy R hr n t x ω

/-- Executed-action sensitivity passes through clipping and noise averaging.
    The simulator/continuation sensitivity is still an explicit assumption. -/
theorem measureQ_sensitivity_of_executed_sensitivity
    {d : ℕ} {X E : Type} [MeasurableSpace E]
    (μ : Measure E) [IsProbabilityMeasure μ]
    (step : X → Action d → E → X) (reward : X → Action d → E → ℝ)
    (noise : E → Action d) (V : X → ℝ) (x : X) (v w : Action d)
    (L : ℝ) (hL : 0 ≤ L)
    (hv : Integrable (fun e => reward x (executed v (noise e)) e +
      V (step x (executed v (noise e)) e)) μ)
    (hw : Integrable (fun e => reward x (executed w (noise e)) e +
      V (step x (executed w (noise e)) e)) μ)
    (hsens : ∀ e u u',
      |(reward x u e + V (step x u e)) - (reward x u' e + V (step x u' e))| ≤
        L * l2Norm (fun j => u j - u' j)) :
    |measureQ μ step reward noise V x v - measureQ μ step reward noise V x w| ≤
      L * l2Norm (fun j => v j - w j) := by
  unfold measureQ
  rw [← integral_sub hv hw]
  have hb : ∀ᵐ e ∂μ,
      ‖(reward x (executed v (noise e)) e + V (step x (executed v (noise e)) e)) -
        (reward x (executed w (noise e)) e + V (step x (executed w (noise e)) e))‖ ≤
        L * l2Norm (fun j => v j - w j) := by
    filter_upwards [] with e
    rw [Real.norm_eq_abs]
    exact (hsens e _ _).trans (mul_le_mul_of_nonneg_left
      (executed_l2_nonexpansive v w (noise e)) hL)
  simpa only [Real.norm_eq_abs, probReal_univ, mul_one] using
    norm_integral_le_of_norm_le_const hb

end BAPR.ActionCompensation
