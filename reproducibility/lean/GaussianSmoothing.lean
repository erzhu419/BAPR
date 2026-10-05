import ActionCompensationMeasure

/-!
# Sensitivity from Gaussian averaging, not smooth simulator dynamics

The integrand may be merely measurable and bounded. The proof uses the
affinity of two translated Gaussian densities and a quadratic inequality.
The bound is in Euclidean action distance, with constant B / sigma.
-/

noncomputable section
open MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace BAPR.ActionCompensation

def gaussianActionDensity {d : ℕ} (v : Action d) (q : ℝ≥0) (x : Action d) : ℝ :=
  ∏ j, gaussianPDFReal (v j) q (x j)

theorem gaussianActionDensity_nonneg {d : ℕ} (v : Action d) (q : ℝ≥0) (x) :
    0 ≤ gaussianActionDensity v q x :=
  Finset.prod_nonneg (fun _ _ => gaussianPDFReal_nonneg _ _ _)

theorem gaussianActionDensity_measurable {d : ℕ} (v : Action d) (q : ℝ≥0) :
    Measurable (gaussianActionDensity v q) := by
  unfold gaussianActionDensity
  fun_prop

theorem gaussianActionDensity_integrable {d : ℕ} (v : Action d) (q : ℝ≥0) :
    Integrable (gaussianActionDensity v q) :=
  Integrable.fintype_prod (fun _ => integrable_gaussianPDFReal _ _)

theorem gaussianActionDensity_integral {d : ℕ} (v : Action d) (q : ℝ≥0)
    (hq : q ≠ 0) : ∫ x, gaussianActionDensity v q x = 1 := by
  unfold gaussianActionDensity
  rw [integral_fintype_prod_volume_eq_prod]
  simp [integral_gaussianPDFReal_eq_one _ hq]

private theorem scalar_affinity (v w x : ℝ) (q : ℝ≥0) (hq : q ≠ 0) :
    Real.sqrt (gaussianPDFReal v q x) * Real.sqrt (gaussianPDFReal w q x) =
      Real.exp (-(v - w)^2 / (8 * q)) * gaussianPDFReal ((v+w)/2) q x := by
  have hq' : (q : ℝ) ≠ 0 := by exact_mod_cast hq
  have hp := gaussianPDFReal_nonneg v q x
  have hr := gaussianPDFReal_nonneg w q x
  have hi : gaussianPDFReal v q x * gaussianPDFReal w q x =
      (Real.exp (-(v-w)^2 / (8*q)) * gaussianPDFReal ((v+w)/2) q x)^2 := by
    simp only [gaussianPDFReal, mul_pow, ← Real.exp_nat_mul]
    have he : -(x-v)^2 / (2*q) + -(x-w)^2 / (2*q) =
        2 * (-(v-w)^2 / (8*q)) + 2 * (-(x-(v+w)/2)^2 / (2*q)) := by
      field_simp [hq']
      ring
    rw [mul_mul_mul_comm, ← Real.exp_add, he, Real.exp_add]
    norm_num
    ring
  have hl : 0 ≤ Real.sqrt (gaussianPDFReal v q x) *
      Real.sqrt (gaussianPDFReal w q x) := by positivity
  have hb : 0 ≤ Real.exp (-(v-w)^2 / (8*q)) *
      gaussianPDFReal ((v+w)/2) q x :=
    mul_nonneg (Real.exp_pos _).le (gaussianPDFReal_nonneg _ _ _)
  have hs : (Real.sqrt (gaussianPDFReal v q x) *
      Real.sqrt (gaussianPDFReal w q x))^2 =
      gaussianPDFReal v q x * gaussianPDFReal w q x := by
    rw [mul_pow, Real.sq_sqrt hp, Real.sq_sqrt hr]
  nlinarith

def actionSqDistance {d : ℕ} (v w : Action d) : ℝ := ∑ j, (v j-w j)^2

theorem actionSqDistance_nonneg {d : ℕ} (v w : Action d) :
    0 ≤ actionSqDistance v w := Finset.sum_nonneg (fun _ _ => sq_nonneg _)

theorem gaussianActionDensity_affinity {d : ℕ} (v w : Action d) (q : ℝ≥0)
    (hq : q ≠ 0) (x) :
    Real.sqrt (gaussianActionDensity v q x) * Real.sqrt (gaussianActionDensity w q x) =
      Real.exp (-actionSqDistance v w / (8*q)) *
        gaussianActionDensity (fun j => (v j+w j)/2) q x := by
  unfold gaussianActionDensity
  rw [Real.sqrt_prod _ (fun _ _ => gaussianPDFReal_nonneg _ _ _),
    Real.sqrt_prod _ (fun _ _ => gaussianPDFReal_nonneg _ _ _),
    ← Finset.prod_mul_distrib]
  simp_rw [scalar_affinity _ _ _ q hq]
  rw [Finset.prod_mul_distrib]
  congr 1
  rw [← Real.exp_sum]
  congr 1
  rw [← Finset.sum_div, Finset.sum_neg_distrib]
  rfl

/-- An L1 estimate from Gaussian affinity. No regularity of an integrand is used. -/
theorem gaussianActionDensity_l1 {d : ℕ} (v w : Action d) (q : ℝ≥0)
    (hq : q ≠ 0) :
    ∫ x, |gaussianActionDensity v q x - gaussianActionDensity w q x| ≤
      Real.sqrt (actionSqDistance v w) / Real.sqrt q := by
  let p := gaussianActionDensity v q
  let r := gaussianActionDensity w q
  let a := Real.exp (-actionSqDistance v w / (8*q))
  let h := fun x => Real.sqrt (p x) * Real.sqrt (r x)
  have hp := gaussianActionDensity_integrable v q
  have hr := gaussianActionDensity_integrable w q
  have hh : Integrable h := by
    simpa only [h, p, r, gaussianActionDensity_affinity v w q hq] using
      (gaussianActionDensity_integrable (fun j => (v j+w j)/2) q).const_mul a
  have hhi : ∫ x, h x = a := by
    simp only [h, p, r, gaussianActionDensity_affinity v w q hq,
      integral_const_mul, gaussianActionDensity_integral _ q hq, mul_one, a]
  have hpi : ∫ x, p x = 1 := gaussianActionDensity_integral v q hq
  have hri : ∫ x, r x = 1 := gaussianActionDensity_integral w q hq
  have hqpos : 0 < (q : ℝ) := by exact_mod_cast (pos_iff_ne_zero.mpr hq)
  have hD := actionSqDistance_nonneg v w
  by_cases hz : actionSqDistance v w = 0
  · have he : v = w := by
      funext j
      have hj : (v j-w j)^2 ≤ actionSqDistance v w :=
        Finset.single_le_sum (fun i _ => sq_nonneg (v i-w i)) (Finset.mem_univ j)
      rw [hz] at hj
      nlinarith [sq_nonneg (v j-w j)]
    simp [he, actionSqDistance]
  have hdpos : 0 < actionSqDistance v w := lt_of_le_of_ne hD (Ne.symm hz)
  let b := Real.sqrt (actionSqDistance v w) / Real.sqrt q
  have hb : 0 < b := by dsimp [b]; positivity
  have hb2 : b^2 = actionSqDistance v w / q := by
    dsimp [b]
    rw [div_pow, Real.sq_sqrt hD, Real.sq_sqrt hqpos.le]
  have ha : a ≤ 1 := by
    apply Real.exp_le_one_iff.mpr
    exact div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr hD) (by positivity)
  have hal : 1 - b^2 / 8 ≤ a := by
    have he := Real.add_one_le_exp (-actionSqDistance v w / (8*q))
    dsimp [a]
    rw [hb2]
    convert he using 1
    ring
  have hi : ∫ x, |p x-r x| ≤
      ∫ x, (p x+r x-2*h x) / (b/2) + (b/2)*(p x+r x+2*h x)/4 := by
    apply integral_mono (hp.sub hr).abs
      (((hp.add hr).sub (hh.const_mul 2)).div_const (b/2) |>.add
        (((hp.add hr).add (hh.const_mul 2)).const_mul (b/2) |>.div_const 4))
    intro x
    have hps := Real.sq_sqrt (gaussianActionDensity_nonneg v q x)
    have hrs := Real.sq_sqrt (gaussianActionDensity_nonneg w q x)
    have hs := sq_nonneg ((Real.sqrt (p x)-Real.sqrt (r x)) -
      (b/4)*(Real.sqrt (p x)+Real.sqrt (r x)))
    have ht := sq_nonneg ((Real.sqrt (p x)-Real.sqrt (r x)) +
      (b/4)*(Real.sqrt (p x)+Real.sqrt (r x)))
    rw [abs_le]
    dsimp [h]
    constructor <;> apply (mul_le_mul_iff_right₀ (show 0 < b/2 by positivity)).mp
    all_goals
      field_simp
      change (Real.sqrt (p x))^2 = p x at hps
      change (Real.sqrt (r x))^2 = r x at hrs
      nlinarith
  have heq : (∫ x, (p x+r x-2*h x) / (b/2) + (b/2)*(p x+r x+2*h x)/4) =
      (2-2*a)/(b/2) + (b/2)*(2+2*a)/4 := by
    rw [integral_add, integral_div, integral_div, integral_const_mul,
      integral_sub (by exact hp.add hr) (hh.const_mul 2),
      integral_add (by exact hp.add hr) (hh.const_mul 2), integral_add hp hr,
      integral_const_mul, hpi, hri, hhi]
    · ring
    · exact ((hp.add hr).sub (hh.const_mul 2)).div_const _
    · exact (((hp.add hr).add (hh.const_mul 2)).const_mul _).div_const _
  rw [heq] at hi
  change _ ≤ b
  refine hi.trans ?_
  apply (mul_le_mul_iff_right₀ hb).mp
  field_simp
  nlinarith [mul_nonneg (show 0 ≤ b^2 by positivity) (show 0 ≤ 1-a by linarith)]

def gaussianSmoothed {d : ℕ} (q : ℝ≥0) (f : Action d → ℝ) (v : Action d) : ℝ :=
  ∫ x, gaussianActionDensity v q x * f x

theorem independentGaussian_density {d : ℕ} (v : Action d) (q : ℝ≥0) (hq : q ≠ 0) :
    Measure.pi (fun j => gaussianReal (v j) q) =
      volume.withDensity (fun x => ENNReal.ofReal (gaussianActionDensity v q x)) := by
  apply Measure.pi_eq
  intro s hs
  rw [withDensity_apply _ (MeasurableSet.univ_pi hs),
    ← ofReal_integral_eq_lintegral_ofReal
      (gaussianActionDensity_integrable v q).restrict
      (ae_of_all _ (gaussianActionDensity_nonneg v q))]
  change ENNReal.ofReal (∫ x, gaussianActionDensity v q x
    ∂(Measure.pi (fun _ : Fin d => (volume : Measure ℝ))).restrict (Set.univ.pi s)) = _
  rw [Measure.restrict_pi_pi]
  unfold gaussianActionDensity
  rw [integral_fintype_prod_eq_prod,
    ENNReal.ofReal_prod_of_nonneg (fun _ _ => integral_nonneg (gaussianPDFReal_nonneg _ _))]
  simp_rw [gaussianReal_apply_eq_integral _ hq]

theorem gaussianSmoothed_eq_expectation {d : ℕ} (q : ℝ≥0) (hq : q ≠ 0)
    (f : Action d → ℝ) (v : Action d) :
    gaussianSmoothed q f v = ∫ x, f x ∂Measure.pi (fun j => gaussianReal (v j) q) := by
  rw [independentGaussian_density v q hq,
    integral_withDensity_eq_integral_toReal_smul
      (gaussianActionDensity_measurable v q).ennreal_ofReal
      (ae_of_all _ (fun _ => ENNReal.ofReal_lt_top))]
  simp only [ENNReal.toReal_ofReal (gaussianActionDensity_nonneg v q _),
    smul_eq_mul, gaussianSmoothed]

theorem gaussianSmoothed_eq_noise_integral {d : ℕ} (q : ℝ≥0) (hq : q ≠ 0)
    (f : Action d → ℝ) (hf : Measurable f) (v : Action d) :
    gaussianSmoothed q f v =
      ∫ e, f (fun j => v j + e j) ∂Measure.pi (fun _ : Fin d => gaussianReal 0 q) := by
  have hm : (Measure.pi (fun _ : Fin d => gaussianReal 0 q)).map
      (fun e j => v j + e j) = Measure.pi (fun j => gaussianReal (v j) q) := by
    rw [Measure.pi_map_pi (fun _ => by fun_prop)]
    simp_rw [gaussianReal_map_const_add, zero_add]
  rw [gaussianSmoothed_eq_expectation q hq f v, ← hm,
    integral_map (Measurable.aemeasurable (by fun_prop)) hf.aestronglyMeasurable]

/-- Bounded measurable integrands, including discontinuous contact responses. -/
theorem gaussianSmoothed_sensitivity {d : ℕ} (q : ℝ≥0) (hq : q ≠ 0)
    (f : Action d → ℝ) (hf : Measurable f) (B : ℝ)
    (hB : 0 ≤ B) (hb : ∀ x, |f x| ≤ B) (v w : Action d) :
    |gaussianSmoothed q f v - gaussianSmoothed q f w| ≤
      B * (Real.sqrt (actionSqDistance v w) / Real.sqrt q) := by
  have hi (a : Action d) : Integrable (fun x => gaussianActionDensity a q x * f x) := by
    apply ((gaussianActionDensity_integrable a q).mul_const B).mono'
      ((gaussianActionDensity_measurable a q).mul hf).aestronglyMeasurable
    filter_upwards [] with x
    rw [Real.norm_eq_abs, abs_mul,
      abs_of_nonneg (gaussianActionDensity_nonneg a q x)]
    exact mul_le_mul_of_nonneg_left (hb x) (gaussianActionDensity_nonneg a q x)
  unfold gaussianSmoothed
  rw [← integral_sub (hi v) (hi w)]
  calc
    _ ≤ ∫ x, |gaussianActionDensity v q x * f x -
        gaussianActionDensity w q x * f x| := abs_integral_le_integral_abs
    _ ≤ ∫ x, B * |gaussianActionDensity v q x - gaussianActionDensity w q x| := by
      apply integral_mono ((hi v).sub (hi w)).abs
        (((gaussianActionDensity_integrable v q).sub
          (gaussianActionDensity_integrable w q)).abs.const_mul B)
      intro x
      simp only [Pi.sub_apply]
      rw [← sub_mul, abs_mul, mul_comm B]
      exact mul_le_mul_of_nonneg_left (hb x) (abs_nonneg _)
    _ ≤ _ := by
      rw [integral_const_mul]
      exact mul_le_mul_of_nonneg_left (gaussianActionDensity_l1 v w q hq) hB

/-- Clipping and arbitrary measurable downstream physics are inside the
    integrand, so neither needs a derivative or a Lipschitz certificate. -/
theorem gaussianSmoothed_clipped_sensitivity {d : ℕ} (q : ℝ≥0) (hq : q ≠ 0)
    (f : Action d → ℝ) (hf : Measurable f) (B : ℝ) (hB : 0 ≤ B)
    (hb : ∀ u, inBox u → |f u| ≤ B) (v w : Action d) :
    |gaussianSmoothed q (fun x => f (clipAction x)) v -
      gaussianSmoothed q (fun x => f (clipAction x)) w| ≤
      B * (Real.sqrt (actionSqDistance v w) / Real.sqrt q) := by
  apply gaussianSmoothed_sensitivity q hq _ (hf.comp (by
    unfold clipAction clipScalar
    fun_prop)) B hB
  intro x
  apply hb
  intro j
  exact ⟨le_max_left _ _, max_le (by norm_num) (min_le_right _ _)⟩

/-- The current actuator-noise Q definition, for physics/reward whose noise
    dependence is entirely through the executed action. -/
theorem actuatorGaussian_Q_sensitivity {d : ℕ} {X : Type}
    (step : X → Action d → X) (reward : X → Action d → ℝ) (V : X → ℝ)
    (x : X) (B : ℝ) (hB : 0 ≤ B)
    (hm : Measurable (fun u => reward x u + V (step x u)))
    (hb : ∀ u, inBox u → |reward x u + V (step x u)| ≤ B) (v w : Action d) :
    |measureQ (actuatorGaussian d) (fun x u _ => step x u)
        (fun x u _ => reward x u) id V x v -
      measureQ (actuatorGaussian d) (fun x u _ => step x u)
        (fun x u _ => reward x u) id V x w| ≤
      50 * B * Real.sqrt (actionSqDistance v w) := by
  have hm' : Measurable (fun u => reward x (clipAction u) + V (step x (clipAction u))) :=
    hm.comp (by unfold clipAction clipScalar; fun_prop)
  have h := gaussianSmoothed_clipped_sensitivity (1/2500) (by norm_num)
    (fun u => reward x u + V (step x u)) hm B hB hb v w
  rw [gaussianSmoothed_eq_noise_integral _ (by norm_num) _ hm' v,
    gaussianSmoothed_eq_noise_integral _ (by norm_num) _ hm' w] at h
  norm_num [Real.sqrt_div] at h
  simpa [measureQ, actuatorGaussian, executed, id_eq, div_eq_mul_inv,
    mul_assoc, mul_comm, mul_left_comm] using h

end BAPR.ActionCompensation
