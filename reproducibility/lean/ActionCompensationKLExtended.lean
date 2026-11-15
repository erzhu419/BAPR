import ActionCompensationKL

/-!
# Extended-valued trajectory information

The nonnegative KL integrand supports infinite divergence without subtracting
infinite integrals. Conditional absolute continuity is retained; integrability
of log likelihoods is not assumed.
-/

noncomputable section
open MeasureTheory ProbabilityTheory InformationTheory
open scoped ENNReal

namespace BAPR.ActionCompensation

private theorem klFun_product_balance (r s : ℝ) :
    klFun (r * s) + |r * Real.log r| * s + r =
      r * klFun s + (r * Real.log r + |r * Real.log r|) * s + 1 := by
  by_cases h0 : r = 0
  · simp [h0, klFun]
  by_cases hs0 : s = 0
  · simp [hs0, klFun, add_comm]
  rw [klFun, klFun, Real.log_mul h0 hs0]
  ring

/-- Conditional integration of the KL product identity remains valid at infinity. -/
theorem lintegral_klFun_mul
    {E : Type*} [MeasurableSpace E] (μ : Measure E) [IsProbabilityMeasure μ]
    (s : E → ℝ) (hs : Measurable s) (hs0 : ∀ x, 0 ≤ s x)
    (hs1 : ∫⁻ x, ENNReal.ofReal (s x) ∂μ = 1) (r : ℝ) (hr : 0 ≤ r) :
    (∫⁻ x, ENNReal.ofReal (klFun (r * s x)) ∂μ) =
      ENNReal.ofReal (klFun r) +
        ENNReal.ofReal r * ∫⁻ x, ENNReal.ofReal (klFun (s x)) ∂μ := by
  let c := |r * Real.log r|
  have hc : 0 ≤ c := abs_nonneg _
  have he : 0 ≤ r * Real.log r + c := by dsimp [c]; linarith [neg_le_abs (r * Real.log r)]
  have hpoint : ∀ x,
      ENNReal.ofReal (klFun (r * s x)) +
        ENNReal.ofReal c * ENNReal.ofReal (s x) + ENNReal.ofReal r =
      ENNReal.ofReal r * ENNReal.ofReal (klFun (s x)) +
        ENNReal.ofReal (r * Real.log r + c) * ENNReal.ofReal (s x) + 1 := by
    intro x
    have h := congrArg ENNReal.ofReal (klFun_product_balance r (s x))
    change ENNReal.ofReal (klFun (r * s x) + c * s x + r) =
      ENNReal.ofReal (r * klFun (s x) + (r * Real.log r + c) * s x + 1) at h
    simpa only [ENNReal.ofReal_add (klFun_nonneg (mul_nonneg hr (hs0 x))) (mul_nonneg hc (hs0 x)),
      ENNReal.ofReal_add (add_nonneg (klFun_nonneg (mul_nonneg hr (hs0 x)))
        (mul_nonneg hc (hs0 x))) hr,
      ENNReal.ofReal_add (mul_nonneg hr (klFun_nonneg (hs0 x))) (mul_nonneg he (hs0 x)),
      ENNReal.ofReal_add (add_nonneg (mul_nonneg hr (klFun_nonneg (hs0 x)))
        (mul_nonneg he (hs0 x))) zero_le_one,
      ENNReal.ofReal_mul hc, ENNReal.ofReal_mul hr, ENNReal.ofReal_mul he,
      ENNReal.ofReal_one] using h
  have hint := congrArg (fun f : E → ℝ≥0∞ => ∫⁻ x, f x ∂μ) (funext hpoint)
  simp only [lintegral_add_left (by fun_prop : Measurable (fun x =>
    ENNReal.ofReal (klFun (r * s x)) + ENNReal.ofReal c * ENNReal.ofReal (s x))),
    lintegral_add_left (by fun_prop : Measurable (fun x =>
      ENNReal.ofReal r * ENNReal.ofReal (klFun (s x)) +
        ENNReal.ofReal (r * Real.log r + c) * ENNReal.ofReal (s x))),
    lintegral_add_left (by fun_prop : Measurable (fun x => ENNReal.ofReal (klFun (r * s x)))),
    lintegral_add_left (by fun_prop : Measurable (fun x => ENNReal.ofReal r * ENNReal.ofReal (klFun (s x)))),
    lintegral_const_mul _ (by fun_prop : Measurable (fun x => ENNReal.ofReal (s x))),
    lintegral_const_mul _ (by fun_prop : Measurable (fun x => ENNReal.ofReal (klFun (s x)))),
    lintegral_const, measure_univ, mul_one, hs1] at hint
  have hconst : ENNReal.ofReal (r * Real.log r + c) + 1 =
      ENNReal.ofReal (klFun r) + ENNReal.ofReal c + ENNReal.ofReal r := by
    rw [← ENNReal.ofReal_one, ← ENNReal.ofReal_add he zero_le_one,
      ← ENNReal.ofReal_add (klFun_nonneg hr) hc,
      ← ENNReal.ofReal_add (add_nonneg (klFun_nonneg hr) hc) hr]
    congr 1
    simp only [klFun]
    ring
  rw [add_assoc (ENNReal.ofReal r * _) _ _, hconst] at hint
  have heq : (∫⁻ x, ENNReal.ofReal (klFun (r * s x)) ∂μ) +
      (ENNReal.ofReal c + ENNReal.ofReal r) =
      (ENNReal.ofReal (klFun r) + ENNReal.ofReal r * ∫⁻ x, ENNReal.ofReal (klFun (s x)) ∂μ) +
        (ENNReal.ofReal c + ENNReal.ofReal r) := by
    simpa only [add_assoc, add_left_comm, add_comm] using hint
  exact (ENNReal.add_left_inj (by finiteness)).mp heq

section Kernels

variable {H Y : Type*} [MeasurableSpace H] [MeasurableSpace Y]
  [MeasurableSpace.CountablyGenerated Y]

def kernelKL (K L : Kernel H Y) (h : H) : ℝ≥0∞ :=
  ∫⁻ y, ENNReal.ofReal (klFun (K.rnDeriv L h y).toReal) ∂L h

theorem measurable_kernelKL (K L : Kernel H Y) [IsMarkovKernel L] :
    Measurable (kernelKL K L) := by
  exact (by fun_prop : Measurable (fun p : H × Y =>
    ENNReal.ofReal (klFun (K.rnDeriv L p.1 p.2).toReal))).lintegral_kernel_prod_right'

theorem kernelKL_eq (K L : Kernel H Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hKL : ∀ h, K h ≪ L h) (h : H) :
    kernelKL K L h = klDiv (K h) (L h) := by
  rw [klDiv_eq_lintegral_klFun, if_pos (hKL h)]
  apply lintegral_congr_ae
  filter_upwards [K.rnDeriv_eq_rnDeriv_measure (η := L) (a := h)] with y hy
  rw [hy]

theorem kernelRN_normalized (K L : Kernel H Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hKL : ∀ h, K h ≪ L h) (h : H) :
    ∫⁻ y, ENNReal.ofReal (K.rnDeriv L h y).toReal ∂L h = 1 := by
  calc
    _ = ∫⁻ y, K.rnDeriv L h y ∂L h := by
      apply lintegral_congr_ae
      filter_upwards [K.rnDeriv_ne_top L (a := h)] with y hy
      exact ENNReal.ofReal_toReal hy
    _ = 1 := by rw [Kernel.lintegral_rnDeriv (hKL h), measure_univ]

/-- Extended-valued chain rule: no log-integrability assumption and no
    conversion of the divergence to `Real`. -/
theorem extended_kl_chain_step
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (K L : Kernel H Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hPQ : P ≪ Q) (hKL : ∀ h, K h ≪ L h) :
    klDiv (P ⊗ₘ K) (Q ⊗ₘ L) =
      klDiv P Q + ∫⁻ h, klDiv (K h) (L h) ∂P := by
  have hac := hPQ.compProd (Filter.Eventually.of_forall hKL)
  have hm : Measurable (fun p : H × Y => P.rnDeriv Q p.1 * K.rnDeriv L p.1 p.2) :=
    ((Measure.measurable_rnDeriv P Q).comp measurable_fst).mul (K.measurable_rnDeriv L)
  have hder := (Q ⊗ₘ L).rnDeriv_withDensity hm
  rw [compProd_density_factor P Q K L hPQ hKL] at hder
  rw [klDiv_eq_lintegral_klFun, if_pos hac]
  calc
    _ = ∫⁻ p : H × Y, ENNReal.ofReal (klFun
        ((P.rnDeriv Q p.1).toReal * (K.rnDeriv L p.1 p.2).toReal)) ∂Q ⊗ₘ L := by
      apply lintegral_congr_ae
      filter_upwards [hder] with p hp
      rw [hp, ENNReal.toReal_mul]
    _ = ∫⁻ h, ENNReal.ofReal (klFun (P.rnDeriv Q h).toReal) +
        ENNReal.ofReal (P.rnDeriv Q h).toReal * kernelKL K L h ∂Q := by
      rw [Measure.lintegral_compProd (by fun_prop)]
      apply lintegral_congr
      intro h
      exact lintegral_klFun_mul (L h) (fun y => (K.rnDeriv L h y).toReal)
        (by fun_prop) (fun _ => ENNReal.toReal_nonneg)
        (kernelRN_normalized K L hKL h) (P.rnDeriv Q h).toReal ENNReal.toReal_nonneg
    _ = (∫⁻ h, ENNReal.ofReal (klFun (P.rnDeriv Q h).toReal) ∂Q) +
        ∫⁻ h, P.rnDeriv Q h * kernelKL K L h ∂Q := by
      rw [lintegral_add_left (by fun_prop)]
      congr 1
      apply lintegral_congr_ae
      filter_upwards [P.rnDeriv_ne_top Q] with h hh
      rw [ENNReal.ofReal_toReal hh]
    _ = klDiv P Q + ∫⁻ h, klDiv (K h) (L h) ∂P := by
      rw [lintegral_rnDeriv_mul hPQ (measurable_kernelKL K L).aemeasurable,
        klDiv_eq_lintegral_klFun, if_pos hPQ]
      congr 1
      exact lintegral_congr (kernelKL_eq K L hKL)

theorem extended_same_policy_kl
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (policy : Kernel H Y) [IsMarkovKernel policy] (hPQ : P ≪ Q) :
    klDiv (P ⊗ₘ policy) (Q ⊗ₘ policy) = klDiv P Q := by
  rw [extended_kl_chain_step P Q policy policy hPQ
    (fun _ => Measure.AbsolutelyContinuous.rfl)]
  simp

theorem extended_controlled_kl_chain_step
    {A : Type*} [MeasurableSpace A] [MeasurableSpace.CountablyGenerated A]
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (policy : Kernel H A) [IsMarkovKernel policy]
    (K L : Kernel (H × A) Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hPQ : P ≪ Q) (hKL : ∀ ha, K ha ≪ L ha) :
    klDiv ((P ⊗ₘ policy) ⊗ₘ K) ((Q ⊗ₘ policy) ⊗ₘ L) =
      klDiv P Q + ∫⁻ h, ∫⁻ a, klDiv (K (h, a)) (L (h, a)) ∂policy h ∂P := by
  rw [extended_kl_chain_step _ _ K L (hPQ.compProd_left policy) hKL,
    extended_same_policy_kl P Q policy hPQ]
  have hm : Measurable (fun ha => klDiv (K ha) (L ha)) := by
    simpa only [← kernelKL_eq K L hKL] using measurable_kernelKL K L
  rw [Measure.lintegral_compProd hm]

end Kernels

/-- Full finite-horizon controlled trajectory identity, allowing infinite KL. -/
theorem extended_controlled_trajectory_kl
    {X A E : Type} [MeasurableSpace X] [MeasurableSpace A] [MeasurableSpace E]
    [MeasurableSpace.CountablyGenerated A] [MeasurableSpace.CountablyGenerated E]
    (ρ : Measure X) [IsProbabilityMeasure ρ]
    (policy : (n : ℕ) → Kernel (ControlledHistory X A E n) A)
    (K L : (n : ℕ) → Kernel (ControlledHistory X A E n × A) E)
    [∀ n, IsMarkovKernel (policy n)]
    [∀ n, IsMarkovKernel (K n)] [∀ n, IsMarkovKernel (L n)]
    (T : ℕ) (hKL : ∀ n < T, ∀ ha, K n ha ≪ L n ha) :
    klDiv (controlledTrajectoryLaw ρ policy K T) (controlledTrajectoryLaw ρ policy L T) =
      ∑ n ∈ Finset.range T, ∫⁻ h, ∫⁻ a,
        klDiv (K n (h, a)) (L n (h, a)) ∂policy n h
          ∂controlledTrajectoryLaw ρ policy K n := by
  have hac : ∀ n ≤ T,
      controlledTrajectoryLaw ρ policy K n ≪ controlledTrajectoryLaw ρ policy L n := by
    intro n
    induction n with
    | zero => intro _; exact Measure.AbsolutelyContinuous.rfl
    | succ n ih =>
        intro hn
        exact ((ih (by omega)).compProd_left (policy n)).compProd
          (Filter.Eventually.of_forall (hKL n (by omega)))
  have hsum : ∀ n ≤ T,
      klDiv (controlledTrajectoryLaw ρ policy K n) (controlledTrajectoryLaw ρ policy L n) =
        ∑ k ∈ Finset.range n, ∫⁻ h, ∫⁻ a,
          klDiv (K k (h, a)) (L k (h, a)) ∂policy k h
            ∂controlledTrajectoryLaw ρ policy K k := by
    intro n
    induction n with
    | zero => intro _; simp [controlledTrajectoryLaw]
    | succ n ih =>
        intro hn
        rw [controlledTrajectoryLaw, controlledTrajectoryLaw,
          extended_controlled_kl_chain_step _ _ _ _ _ (hac n (by omega))
            (hKL n (by omega)), ih (by omega), Finset.sum_range_succ]
  exact hsum T le_rfl

end BAPR.ActionCompensation
