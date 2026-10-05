import ActionCompensationMeasure

/-!
# Information in a continuous, causally controlled trajectory

Probability kernels, rather than Lebesgue densities, accommodate deterministic
actions and clipped actuator noise. Finite KL requires absolute continuity and
integrable log likelihoods; it does not follow from Gaussian actuator noise alone.
-/

noncomputable section

open MeasureTheory ProbabilityTheory InformationTheory
open scoped ENNReal

namespace BAPR.ActionCompensation

variable {H Y : Type*} [MeasurableSpace H] [MeasurableSpace Y]
  [MeasurableSpace.CountablyGenerated Y]

/-- The joint likelihood factorization is derived from conditional kernels,
    not assumed as a trajectory-level hypothesis. -/
theorem compProd_density_factor
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (K L : Kernel H Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hPQ : P ≪ Q) (hKL : ∀ h, K h ≪ L h) :
    (Q ⊗ₘ L).withDensity
      (fun p => P.rnDeriv Q p.1 * K.rnDeriv L p.1 p.2) = P ⊗ₘ K := by
  apply Measure.ext_of_lintegral
  intro f hf
  have hm : Measurable (fun p : H × Y => P.rnDeriv Q p.1 * K.rnDeriv L p.1 p.2) :=
    ((Measure.measurable_rnDeriv P Q).comp measurable_fst).mul (K.measurable_rnDeriv L)
  rw [lintegral_withDensity_eq_lintegral_mul _ hm hf,
    Measure.lintegral_compProd (by fun_prop), Measure.lintegral_compProd hf]
  simp only [Pi.mul_apply, mul_assoc]
  have hpull : ∀ h, (∫⁻ y, P.rnDeriv Q h * (K.rnDeriv L h y * f (h, y)) ∂L h) =
      P.rnDeriv Q h * ∫⁻ y, K.rnDeriv L h y * f (h, y) ∂L h := by
    intro h
    exact lintegral_const_mul _ (by fun_prop)
  simp_rw [hpull]
  have hinner : ∀ h, (∫⁻ y, K.rnDeriv L h y * f (h, y) ∂L h) =
      ∫⁻ y, f (h, y) ∂K h := by
    intro h
    calc
      _ = ∫⁻ y, (K h).rnDeriv (L h) y * f (h, y) ∂L h := by
        apply lintegral_congr_ae
        filter_upwards [K.rnDeriv_eq_rnDeriv_measure (η := L) (a := h)] with y hy
        rw [hy]
      _ = _ := lintegral_rnDeriv_mul (hKL h)
        (hf.comp measurable_prodMk_left).aemeasurable
  simp_rw [hinner]
  exact lintegral_rnDeriv_mul hPQ (hf.lintegral_kernel_prod_right').aemeasurable

def kernelLLR (K L : Kernel H Y) (p : H × Y) : ℝ :=
  Real.log (K.rnDeriv L p.1 p.2).toReal

theorem measurable_kernelLLR (K L : Kernel H Y) : Measurable (kernelLLR K L) := by
  unfold kernelLLR
  fun_prop

theorem kernelLLR_ae (K L : Kernel H Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hKL : ∀ h, K h ≪ L h) (h : H) :
    (fun y => kernelLLR K L (h, y)) =ᵐ[K h] llr (K h) (L h) := by
  filter_upwards [(hKL h).ae_le
    (K.rnDeriv_eq_rnDeriv_measure (η := L) (a := h))] with y hy
  simp [kernelLLR, llr, hy]

theorem compProd_llr
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (K L : Kernel H Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hPQ : P ≪ Q) (hKL : ∀ h, K h ≪ L h) :
    llr (P ⊗ₘ K) (Q ⊗ₘ L) =ᵐ[P ⊗ₘ K]
      (fun p => llr P Q p.1 + kernelLLR K L p) := by
  have hac := hPQ.compProd (Filter.Eventually.of_forall hKL)
  have hm : Measurable (fun p : H × Y => P.rnDeriv Q p.1 * K.rnDeriv L p.1 p.2) :=
    ((Measure.measurable_rnDeriv P Q).comp measurable_fst).mul (K.measurable_rnDeriv L)
  have hder := (Q ⊗ₘ L).rnDeriv_withDensity hm
  rw [compProd_density_factor P Q K L hPQ hKL] at hder
  have hpos : ∀ᵐ p ∂(P ⊗ₘ K),
      0 < (P.rnDeriv Q p.1).toReal ∧ 0 < (K.rnDeriv L p.1 p.2).toReal := by
    apply Measure.ae_compProd_of_ae_ae
    · exact (measurableSet_lt measurable_const
        ((Measure.measurable_rnDeriv P Q).comp measurable_fst).ennreal_toReal).inter
        (measurableSet_lt measurable_const (K.measurable_rnDeriv L).ennreal_toReal)
    filter_upwards [Measure.rnDeriv_pos hPQ, hPQ.ae_le (P.rnDeriv_ne_top Q)] with h hp ht
    filter_upwards [K.rnDeriv_toReal_pos (hKL h)] with y hy
    exact ⟨ENNReal.toReal_pos hp.ne' ht, hy⟩
  filter_upwards [hac.ae_le hder, hpos] with p hd hp
  simp only [llr, hd, ENNReal.toReal_mul, kernelLLR]
  exact Real.log_mul hp.1.ne' hp.2.ne'

omit [MeasurableSpace.CountablyGenerated Y] in
theorem integrable_history_lift
    (P : Measure H) [IsProbabilityMeasure P] (K : Kernel H Y) [IsMarkovKernel K]
    {f : H → ℝ} (hf : Integrable f P) :
    Integrable (fun p : H × Y => f p.1) (P ⊗ₘ K) := by
  have hmap : (P ⊗ₘ K).map Prod.fst = P := Measure.fst_compProd P K
  have hi : Integrable f ((P ⊗ₘ K).map Prod.fst) := by rwa [hmap]
  exact hi.comp_measurable measurable_fst

theorem compProd_llr_integrable
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (K L : Kernel H Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hPQ : P ≪ Q) (hKL : ∀ h, K h ≪ L h)
    (hbase : Integrable (llr P Q) P)
    (hstep : Integrable (kernelLLR K L) (P ⊗ₘ K)) :
    Integrable (llr (P ⊗ₘ K) (Q ⊗ₘ L)) (P ⊗ₘ K) :=
  ((integrable_history_lift P K hbase).add hstep).congr
    (compProd_llr P Q K L hPQ hKL).symm

/-- Finite-KL chain step on continuous history and transition spaces. -/
theorem continuous_kl_chain_step
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (K L : Kernel H Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hPQ : P ≪ Q) (hKL : ∀ h, K h ≪ L h)
    (hbase : Integrable (llr P Q) P)
    (hstep : Integrable (kernelLLR K L) (P ⊗ₘ K)) :
    (klDiv (P ⊗ₘ K) (Q ⊗ₘ L)).toReal =
      (klDiv P Q).toReal + ∫ h, (klDiv (K h) (L h)).toReal ∂P := by
  have hjoint := compProd_llr_integrable P Q K L hPQ hKL hbase hstep
  rw [toReal_klDiv (hPQ.compProd (Filter.Eventually.of_forall hKL)) hjoint]
  simp only [probReal_univ, add_sub_cancel_right]
  rw [integral_congr_ae (compProd_llr P Q K L hPQ hKL),
    integral_add (integrable_history_lift P K hbase) hstep,
    Measure.integral_compProd (integrable_history_lift P K hbase),
    Measure.integral_compProd hstep]
  simp only [integral_const, probReal_univ, one_smul]
  congr 1
  · symm
    simpa using toReal_klDiv hPQ hbase
  · apply integral_congr_ae
    filter_upwards [] with h
    rw [integral_congr_ae (kernelLLR_ae K L hKL h)]
    exact (toReal_klDiv_of_measure_eq (hKL h) (by simp)).symm

/-- A shared behavior-policy kernel contributes no KL, even for a Dirac policy. -/
theorem same_policy_llr_integrable
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (policy : Kernel H Y) [IsMarkovKernel policy]
    (hPQ : P ≪ Q) (hbase : Integrable (llr P Q) P) :
    Integrable (llr (P ⊗ₘ policy) (Q ⊗ₘ policy)) (P ⊗ₘ policy) := by
  have hzero : kernelLLR policy policy =ᵐ[P ⊗ₘ policy] 0 := by
    apply Measure.ae_compProd_of_ae_ae
    · exact measurableSet_eq_fun (measurable_kernelLLR policy policy) measurable_const
    filter_upwards [] with h
    exact (kernelLLR_ae policy policy (fun _ => Measure.AbsolutelyContinuous.rfl) h).trans
      (llr_self (policy h))
  exact compProd_llr_integrable P Q policy policy hPQ
    (fun _ => Measure.AbsolutelyContinuous.rfl) hbase
    ((integrable_zero _ _ _).congr hzero.symm)

theorem same_policy_kl
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (policy : Kernel H Y) [IsMarkovKernel policy]
    (hPQ : P ≪ Q) (hbase : Integrable (llr P Q) P) :
    (klDiv (P ⊗ₘ policy) (Q ⊗ₘ policy)).toReal = (klDiv P Q).toReal := by
  have hzero : kernelLLR policy policy =ᵐ[P ⊗ₘ policy] 0 := by
    apply Measure.ae_compProd_of_ae_ae
    · exact measurableSet_eq_fun (measurable_kernelLLR policy policy) measurable_const
    filter_upwards [] with h
    exact (kernelLLR_ae policy policy (fun _ => Measure.AbsolutelyContinuous.rfl) h).trans
      (llr_self (policy h))
  have hint : Integrable (kernelLLR policy policy) (P ⊗ₘ policy) :=
    (integrable_zero _ _ _).congr hzero.symm
  rw [continuous_kl_chain_step P Q policy policy hPQ
    (fun _ => Measure.AbsolutelyContinuous.rfl) hbase hint]
  simp

theorem integrable_conditional_kl
    (P : Measure H) [IsProbabilityMeasure P]
    (K L : Kernel H Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hKL : ∀ h, K h ≪ L h)
    (hstep : Integrable (kernelLLR K L) (P ⊗ₘ K)) :
    Integrable (fun h => (klDiv (K h) (L h)).toReal) P := by
  have hi : Integrable (fun h => ∫ y, kernelLLR K L (h, y) ∂K h) P := by
    simpa [Measure.compProd, Kernel.prodMkLeft_apply] using hstep.integral_compProd
  apply hi.congr
  filter_upwards [] with h
  rw [integral_congr_ae (kernelLLR_ae K L hKL h)]
  exact (toReal_klDiv_of_measure_eq (hKL h) (by simp)).symm

/-- One controlled observation: average over the same causal action law under
    both mode hypotheses. No positive density for the action law is required. -/
theorem controlled_kl_chain_step
    {A : Type*} [MeasurableSpace A] [MeasurableSpace.CountablyGenerated A]
    (P Q : Measure H) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (policy : Kernel H A) [IsMarkovKernel policy]
    (K L : Kernel (H × A) Y) [IsMarkovKernel K] [IsMarkovKernel L]
    (hPQ : P ≪ Q) (hKL : ∀ ha, K ha ≪ L ha)
    (hbase : Integrable (llr P Q) P)
    (hstep : Integrable (kernelLLR K L) ((P ⊗ₘ policy) ⊗ₘ K)) :
    (klDiv ((P ⊗ₘ policy) ⊗ₘ K) ((Q ⊗ₘ policy) ⊗ₘ L)).toReal =
      (klDiv P Q).toReal +
        ∫ h, ∫ a, (klDiv (K (h, a)) (L (h, a))).toReal ∂policy h ∂P := by
  rw [continuous_kl_chain_step (P ⊗ₘ policy) (Q ⊗ₘ policy) K L
    (hPQ.compProd_left policy) hKL
    (same_policy_llr_integrable P Q policy hPQ hbase) hstep,
    same_policy_kl P Q policy hPQ hbase,
    Measure.integral_compProd (integrable_conditional_kl (P ⊗ₘ policy) K L hKL hstep)]

/-- Nested prefixes retain every observation, rather than only the final state. -/
def KLHistory (X E : Type) : ℕ → Type
  | 0 => X
  | n + 1 => KLHistory X E n × E

instance historyMeasurable (X E : Type) [MeasurableSpace X] [MeasurableSpace E] :
    (n : ℕ) → MeasurableSpace (KLHistory X E n)
  | 0 => inferInstanceAs (MeasurableSpace X)
  | n + 1 => @MeasurableSpace.prod _ _ (historyMeasurable X E n)
      (inferInstanceAs (MeasurableSpace E))

def trajectoryLaw {X E : Type} [MeasurableSpace X] [MeasurableSpace E]
    (ρ : Measure X) (K : (n : ℕ) → Kernel (KLHistory X E n) E) :
    (n : ℕ) → Measure (KLHistory X E n)
  | 0 => ρ
  | n + 1 => trajectoryLaw ρ K n ⊗ₘ K n

instance trajectoryLaw_probability {X E : Type} [MeasurableSpace X] [MeasurableSpace E]
    (ρ : Measure X) [IsProbabilityMeasure ρ]
    (K : (n : ℕ) → Kernel (KLHistory X E n) E) [∀ n, IsMarkovKernel (K n)] :
    (n : ℕ) → IsProbabilityMeasure (trajectoryLaw ρ K n)
  | 0 => inferInstanceAs (IsProbabilityMeasure ρ)
  | n + 1 => by
      letI := trajectoryLaw_probability ρ K n
      exact inferInstanceAs (IsProbabilityMeasure (trajectoryLaw ρ K n ⊗ₘ K n))

/-- Multi-step KL identity on continuous path space with a common reset law.
    Finiteness is part of the conclusion, so `toReal` cannot conceal infinity. -/
theorem continuous_trajectory_kl
    {X E : Type} [MeasurableSpace X] [MeasurableSpace E]
    [MeasurableSpace.CountablyGenerated E]
    (ρ : Measure X) [IsProbabilityMeasure ρ]
    (K L : (n : ℕ) → Kernel (KLHistory X E n) E)
    [∀ n, IsMarkovKernel (K n)] [∀ n, IsMarkovKernel (L n)]
    (hKL : ∀ n h, K n h ≪ L n h)
    (hstep : ∀ n, Integrable (kernelLLR (K n) (L n))
      (trajectoryLaw ρ K n ⊗ₘ K n)) (T : ℕ) :
    klDiv (trajectoryLaw ρ K T) (trajectoryLaw ρ L T) ≠ ∞ ∧
      (klDiv (trajectoryLaw ρ K T) (trajectoryLaw ρ L T)).toReal =
        ∑ n ∈ Finset.range T,
          ∫ h, (klDiv (K n h) (L n h)).toReal ∂trajectoryLaw ρ K n := by
  have hdata : ∀ n, trajectoryLaw ρ K n ≪ trajectoryLaw ρ L n ∧
      Integrable (llr (trajectoryLaw ρ K n) (trajectoryLaw ρ L n))
        (trajectoryLaw ρ K n) := by
    intro n
    induction n with
    | zero =>
        exact ⟨Measure.AbsolutelyContinuous.rfl,
          (integrable_zero _ _ _).congr (llr_self ρ).symm⟩
    | succ n ih =>
        exact ⟨ih.1.compProd (Filter.Eventually.of_forall (hKL n)),
          compProd_llr_integrable _ _ _ _ ih.1 (hKL n) ih.2 (hstep n)⟩
  refine ⟨klDiv_ne_top_iff.mpr (hdata T), ?_⟩
  induction T with
  | zero => simp [trajectoryLaw]
  | succ n ih =>
      rw [trajectoryLaw, trajectoryLaw,
        continuous_kl_chain_step _ _ _ _ (hdata n).1 (hKL n) (hdata n).2 (hstep n),
        ih, Finset.sum_range_succ]

def ControlledHistory (X A E : Type) : ℕ → Type
  | 0 => X
  | n + 1 => (ControlledHistory X A E n × A) × E

instance controlledHistoryMeasurable (X A E : Type)
    [MeasurableSpace X] [MeasurableSpace A] [MeasurableSpace E] :
    (n : ℕ) → MeasurableSpace (ControlledHistory X A E n)
  | 0 => inferInstanceAs (MeasurableSpace X)
  | n + 1 => @MeasurableSpace.prod _ _
      (@MeasurableSpace.prod _ _ (controlledHistoryMeasurable X A E n)
        (inferInstanceAs (MeasurableSpace A))) (inferInstanceAs (MeasurableSpace E))

def controlledTrajectoryLaw {X A E : Type}
    [MeasurableSpace X] [MeasurableSpace A] [MeasurableSpace E]
    (ρ : Measure X) (policy : (n : ℕ) → Kernel (ControlledHistory X A E n) A)
    (K : (n : ℕ) → Kernel (ControlledHistory X A E n × A) E) :
    (n : ℕ) → Measure (ControlledHistory X A E n)
  | 0 => ρ
  | n + 1 => (controlledTrajectoryLaw ρ policy K n ⊗ₘ policy n) ⊗ₘ K n

instance controlledTrajectoryLaw_probability {X A E : Type}
    [MeasurableSpace X] [MeasurableSpace A] [MeasurableSpace E]
    (ρ : Measure X) [IsProbabilityMeasure ρ]
    (policy : (n : ℕ) → Kernel (ControlledHistory X A E n) A)
    (K : (n : ℕ) → Kernel (ControlledHistory X A E n × A) E)
    [∀ n, IsMarkovKernel (policy n)] [∀ n, IsMarkovKernel (K n)] :
    (n : ℕ) → IsProbabilityMeasure (controlledTrajectoryLaw ρ policy K n)
  | 0 => inferInstanceAs (IsProbabilityMeasure ρ)
  | n + 1 => by
      letI := controlledTrajectoryLaw_probability ρ policy K n
      exact inferInstanceAs (IsProbabilityMeasure
        ((controlledTrajectoryLaw ρ policy K n ⊗ₘ policy n) ⊗ₘ K n))

/-- Paper's controlled trajectory chain rule. Both hypotheses use the same
    history-dependent policy and reset law; only transition kernels differ.
    Likelihood integrability is needed only before the specified horizon. -/
theorem controlled_trajectory_kl
    {X A E : Type} [MeasurableSpace X] [MeasurableSpace A] [MeasurableSpace E]
    [MeasurableSpace.CountablyGenerated A] [MeasurableSpace.CountablyGenerated E]
    (ρ : Measure X) [IsProbabilityMeasure ρ]
    (policy : (n : ℕ) → Kernel (ControlledHistory X A E n) A)
    (K L : (n : ℕ) → Kernel (ControlledHistory X A E n × A) E)
    [∀ n, IsMarkovKernel (policy n)]
    [∀ n, IsMarkovKernel (K n)] [∀ n, IsMarkovKernel (L n)]
    (T : ℕ) (hKL : ∀ n < T, ∀ ha, K n ha ≪ L n ha)
    (hstep : ∀ n < T, Integrable (kernelLLR (K n) (L n))
      ((controlledTrajectoryLaw ρ policy K n ⊗ₘ policy n) ⊗ₘ K n)) :
    klDiv (controlledTrajectoryLaw ρ policy K T)
      (controlledTrajectoryLaw ρ policy L T) ≠ ∞ ∧
    (klDiv (controlledTrajectoryLaw ρ policy K T)
      (controlledTrajectoryLaw ρ policy L T)).toReal =
      ∑ n ∈ Finset.range T, ∫ h, ∫ a,
        (klDiv (K n (h, a)) (L n (h, a))).toReal ∂policy n h
          ∂controlledTrajectoryLaw ρ policy K n := by
  have hdata : ∀ n ≤ T,
      controlledTrajectoryLaw ρ policy K n ≪ controlledTrajectoryLaw ρ policy L n ∧
      Integrable (llr (controlledTrajectoryLaw ρ policy K n)
        (controlledTrajectoryLaw ρ policy L n)) (controlledTrajectoryLaw ρ policy K n) := by
    intro n
    induction n with
    | zero =>
        intro _
        exact ⟨Measure.AbsolutelyContinuous.rfl,
          (integrable_zero _ _ _).congr (llr_self ρ).symm⟩
    | succ n ih =>
        intro hn
        have hp := ih (by omega)
        have hac := hp.1.compProd_left (policy n)
        exact ⟨hac.compProd (Filter.Eventually.of_forall (hKL n (by omega))),
          compProd_llr_integrable _ _ _ _ hac (hKL n (by omega))
            (same_policy_llr_integrable _ _ _ hp.1 hp.2) (hstep n (by omega))⟩
  refine ⟨klDiv_ne_top_iff.mpr (hdata T le_rfl), ?_⟩
  have hsum : ∀ n ≤ T,
      (klDiv (controlledTrajectoryLaw ρ policy K n)
        (controlledTrajectoryLaw ρ policy L n)).toReal =
        ∑ k ∈ Finset.range n, ∫ h, ∫ a,
          (klDiv (K k (h, a)) (L k (h, a))).toReal ∂policy k h
            ∂controlledTrajectoryLaw ρ policy K k := by
    intro n
    induction n with
    | zero => intro _; simp [controlledTrajectoryLaw]
    | succ n ih =>
        intro hn
        have hp := hdata n (by omega)
        rw [controlledTrajectoryLaw, controlledTrajectoryLaw,
          controlled_kl_chain_step _ _ _ _ _ hp.1 (hKL n (by omega)) hp.2
            (hstep n (by omega)), ih (by omega), Finset.sum_range_succ]
  exact hsum T le_rfl

end BAPR.ActionCompensation
