import ActionCompensationCumulative

/-!
# Retaining a positive mode margin through quiet transitions

The bound retains positive log odds rather than truncating them to zero. Flat
evidence preserves a strict mode ordering on every finite quiet prefix; this
does not establish that the retained mode is still the hidden current mode.
-/

noncomputable section

namespace BAPR.ActionCompensation

open Finset

def stickyRetention (K a k m : ℝ) : ℝ :=
  Real.log (a + K * k) - Real.log (a * Real.exp (-m) + K * k)

theorem stickyRetention_zero (K a k : ℝ) : stickyRetention K a k 0 = 0 := by
  simp [stickyRetention]

theorem stickyRetention_nonneg (K a k m : ℝ)
    (hK : 0 < K) (ha : 0 ≤ a) (hk : 0 < k) (hm : 0 ≤ m) :
    0 ≤ stickyRetention K a k m := by
  have he : Real.exp (-m) ≤ 1 := by
    simpa using Real.exp_le_exp.mpr (neg_nonpos.mpr hm)
  have hd : 0 < a * Real.exp (-m) + K * k :=
    add_pos_of_nonneg_of_pos (mul_nonneg ha (Real.exp_pos _).le) (mul_pos hK hk)
  have hl := Real.log_le_log hd (show a * Real.exp (-m) + K * k ≤ a + K * k by
    nlinarith [mul_le_mul_of_nonneg_left he ha])
  exact sub_nonneg.mpr hl

theorem stickyRetention_pos (K a k m : ℝ)
    (hK : 0 < K) (ha : 0 < a) (hk : 0 < k) (hm : 0 < m) :
    0 < stickyRetention K a k m := by
  have he : Real.exp (-m) < 1 := by
    simpa using Real.exp_lt_exp.mpr (neg_neg_of_pos hm)
  have hd : 0 < a * Real.exp (-m) + K * k :=
    add_pos (mul_pos ha (Real.exp_pos _)) (mul_pos hK hk)
  exact sub_pos.mpr (Real.log_lt_log hd (by
    nlinarith [mul_lt_mul_of_pos_left he ha]))

theorem stickyRetention_le_self (K a k m : ℝ)
    (hK : 0 < K) (ha : 0 ≤ a) (hk : 0 < k) (hm : 0 ≤ m) :
    stickyRetention K a k m ≤ m := by
  have he : Real.exp (-m) ≤ 1 := by
    simpa using Real.exp_le_exp.mpr (neg_nonpos.mpr hm)
  have hn : 0 < a + K * k := add_pos_of_nonneg_of_pos ha (mul_pos hK hk)
  have hcross : Real.exp (-m) * (a + K * k) ≤ a * Real.exp (-m) + K * k := by
    nlinarith [mul_le_mul_of_nonneg_right he (mul_pos hK hk).le]
  have hl := Real.log_le_log (mul_pos (Real.exp_pos _) hn) hcross
  rw [Real.log_mul (ne_of_gt (Real.exp_pos _)) (ne_of_gt hn), Real.log_exp] at hl
  unfold stickyRetention
  linarith

theorem modeMass_card_lower_of_margin {Mode : Type} [Fintype Mode]
    (b : Mode → ℝ) (z : Mode) (m : ℝ) (hb : IsModeMass b)
    (hbp : ∀ w, 0 < b w) (hm : 0 ≤ m)
    (hmargin : ∀ w, w ≠ z → m ≤ modeLogOdds b z w) :
    1 ≤ (Fintype.card Mode : ℝ) * b z := by
  have hmax (w : Mode) : b w ≤ b z := by
    by_cases hw : w = z
    · simp [hw]
    · apply (Real.log_le_log_iff (hbp w) (hbp z)).mp
      have h := hmargin w hw
      unfold modeLogOdds at h
      linarith
  have hs := sum_le_sum (s := univ) (fun w _ => hmax w)
  simpa only [hb.2, sum_const, card_univ, nsmul_eq_mul] using hs

/-- A normalized positive mode margin survives common sticky mixing. -/
theorem stickyPrior_retention {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k m : ℝ) (b : Mode → ℝ) (z : Mode)
    (ha : 0 ≤ a) (hk : 0 < k) (hm : 0 ≤ m)
    (hb : IsModeMass b) (hbp : ∀ v, 0 < b v)
    (hmargin : ∀ v, v ≠ z → m ≤ modeLogOdds b z v)
    (w : Mode) (hw : w ≠ z) :
    stickyRetention (Fintype.card Mode) a k m ≤
      modeLogOdds (stickyPrior a k b) z w := by
  let K : ℝ := Fintype.card Mode
  let e : ℝ := Real.exp (-m)
  have hK : 0 < K := by
    change 0 < (Fintype.card Mode : ℝ)
    exact_mod_cast (Fintype.card_pos (α := Mode))
  have he : 0 < e := Real.exp_pos _
  have he1 : e ≤ 1 := by
    simpa [e] using Real.exp_le_exp.mpr (neg_nonpos.mpr hm)
  have hfloor := modeMass_card_lower_of_margin b z m hb hbp hm hmargin
  have hsep : b w ≤ e * b z := by
    have hlog : Real.log (b w) ≤ -m + Real.log (b z) := by
      have h := hmargin w hw
      unfold modeLogOdds at h
      linarith
    have h := Real.exp_le_exp.mpr hlog
    simpa only [Real.exp_add, Real.exp_log (hbp w), Real.exp_log (hbp z)] using h
  have hn : 0 < a + K * k := add_pos_of_nonneg_of_pos ha (mul_pos hK hk)
  have hd : 0 < a * e + K * k :=
    add_pos_of_nonneg_of_pos (mul_nonneg ha he.le) (mul_pos hK hk)
  have hpz := stickyPrior_pos a k b ha hk hb z
  have hpw := stickyPrior_pos a k b ha hk hb w
  have hcross : (a + K * k) * stickyPrior a k b w ≤
      (a * e + K * k) * stickyPrior a k b z := by
    apply sub_nonneg.mp
    calc
      _ = a * (a + K * k) * (e * b z - b w) +
          a * k * ((1 - e) * (K * b z - 1)) := by
            unfold stickyPrior
            ring
      _ ≥ 0 := add_nonneg
        (mul_nonneg (mul_nonneg ha hn.le) (sub_nonneg.mpr hsep))
        (mul_nonneg (mul_nonneg ha hk.le)
          (mul_nonneg (sub_nonneg.mpr he1) (sub_nonneg.mpr hfloor)))
  have hl := Real.log_le_log (mul_pos hn hpw) hcross
  rw [Real.log_mul (ne_of_gt hn) (ne_of_gt hpw),
    Real.log_mul (ne_of_gt hd) (ne_of_gt hpz)] at hl
  change Real.log (a + K * k) - Real.log (a * e + K * k) ≤
    Real.log (stickyPrior a k b z) - Real.log (stickyPrior a k b w)
  linarith

theorem temperedFilter_retention_step {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c m η : ℝ) (b evidence : Mode → ℝ) (z : Mode)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c) (hm : 0 ≤ m)
    (hb : IsModeMass b) (hbp : ∀ v, 0 < b v)
    (hmargin : ∀ v, v ≠ z → m ≤ modeLogOdds b z v)
    (hscore : ∀ v, v ≠ z → -η ≤ evidence z - evidence v)
    (w : Mode) (hw : w ≠ z) :
    δ * stickyRetention (Fintype.card Mode) a k m - c * η ≤
      modeLogOdds (temperedFilterStep a k δ c b evidence) z w := by
  rw [temperedFilter_log_odds a k δ c b evidence z w]
  have hp := mul_le_mul_of_nonneg_left
    (stickyPrior_retention a k m b z ha hk hm hb hbp hmargin w hw) hδ
  have he := mul_le_mul_of_nonneg_left (hscore w hw) hc
  linarith

def retainedMargin (K a k δ c m : ℝ) (η : ℕ → ℝ) : ℕ → ℝ
  | 0 => m
  | n + 1 => δ * stickyRetention K a k (retainedMargin K a k δ c m η n) - c * η n

/-- Nonnegative intermediate certificates suffice; evidence may be adverse. -/
theorem temperedFilter_retained_margin {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c m : ℝ) (η : ℕ → ℝ) (initial : Mode → ℝ)
    (evidence : ℕ → Mode → ℝ) (z : Mode) (N : ℕ)
    (ha : 0 ≤ a) (hk : 0 < k) (hδ : 0 ≤ δ) (hc : 0 ≤ c)
    (hb : IsModeMass initial) (hbp : ∀ v, 0 < initial v)
    (hinit : ∀ v, v ≠ z → m ≤ modeLogOdds initial z v)
    (hbudget : ∀ n < N, 0 ≤ retainedMargin (Fintype.card Mode) a k δ c m η n)
    (hscore : ∀ n < N, ∀ v, v ≠ z → -η n ≤ evidence n z - evidence n v) :
    ∀ w, w ≠ z → retainedMargin (Fintype.card Mode) a k δ c m η N ≤
      modeLogOdds (temperedFilterPath a k δ c initial evidence N) z w := by
  have bound (n : ℕ) : n ≤ N → ∀ w, w ≠ z →
      retainedMargin (Fintype.card Mode) a k δ c m η n ≤
        modeLogOdds (temperedFilterPath a k δ c initial evidence n) z w := by
    induction n with
    | zero => simpa only [retainedMargin, temperedFilterPath] using fun _ => hinit
    | succ n ih =>
      intro hn w hw
      have hnN : n < N := Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hn
      change δ * stickyRetention (Fintype.card Mode) a k
          (retainedMargin (Fintype.card Mode) a k δ c m η n) - c * η n ≤
        modeLogOdds (temperedFilterStep a k δ c
          (temperedFilterPath a k δ c initial evidence n) (evidence n)) z w
      exact temperedFilter_retention_step a k δ c _ (η n) _ (evidence n) z
        ha hk hδ hc (hbudget n hnN)
        (temperedFilterPath_mass a k δ c initial evidence hb n)
        (temperedFilterPath_pos a k δ c initial evidence hbp n)
        (ih (Nat.le_of_lt hnN)) (hscore n hnN) w hw
  exact bound N le_rfl

theorem retainedMargin_flat_pos (K a k δ c m : ℝ)
    (hK : 0 < K) (ha : 0 < a) (hk : 0 < k) (hδ : 0 < δ) (hm : 0 < m)
    (n : ℕ) : 0 < retainedMargin K a k δ c m (fun _ => 0) n := by
  induction n with
  | zero => exact hm
  | succ n ih =>
    simpa only [retainedMargin, mul_zero, sub_zero] using
      mul_pos hδ (stickyRetention_pos K a k _ hK ha hk ih)

theorem retainedMargin_flat_upper (K a k δ c m : ℝ)
    (hK : 0 < K) (ha : 0 < a) (hk : 0 < k) (hδ : 0 < δ) (hm : 0 < m)
    (n : ℕ) : retainedMargin K a k δ c m (fun _ => 0) n ≤ δ ^ n * m := by
  induction n with
  | zero => simp [retainedMargin]
  | succ n ih =>
    have hp := retainedMargin_flat_pos K a k δ c m hK ha hk hδ hm n
    calc
      _ = δ * stickyRetention K a k (retainedMargin K a k δ c m (fun _ => 0) n) := by
        simp only [retainedMargin, mul_zero, sub_zero]
      _ ≤ δ * retainedMargin K a k δ c m (fun _ => 0) n :=
        mul_le_mul_of_nonneg_left (stickyRetention_le_self K a k _ hK ha.le hk hp.le) hδ.le
      _ ≤ δ * (δ ^ n * m) := mul_le_mul_of_nonneg_left ih hδ.le
      _ = δ ^ (n + 1) * m := by rw [pow_succ]; ring

theorem frozenSticky_retention_formula (m : ℝ) :
    stickyRetention 4 (374 / 375) (1 / 1500) m =
      -Real.log ((374 * Real.exp (-m) + 1) / 375) := by
  have hn : (374 / 375 : ℝ) + 4 * (1 / 1500) = 1 := by norm_num
  have hd : (374 / 375 : ℝ) * Real.exp (-m) + 4 * (1 / 1500) =
      (374 * Real.exp (-m) + 1) / 375 := by ring
  rw [stickyRetention, hn, hd, Real.log_one, zero_sub]

theorem positive_mode_margin_forces_map {Mode : Type}
    (b : Mode → ℝ) (z selected : Mode) (m : ℝ)
    (hbp : ∀ w, 0 < b w) (hm : 0 < m)
    (hmargin : ∀ w, w ≠ z → m ≤ modeLogOdds b z w)
    (hmap : ∀ w, b w ≤ b selected) : selected = z := by
  by_contra hs
  have hl := hmargin selected hs
  have hmax := Real.log_le_log (hbp z) (hmap z)
  unfold modeLogOdds at hl
  linarith

/-- Flat scores preserve a strict MAP on every finite real-arithmetic prefix. -/
theorem temperedFilter_quiet_retention {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c m : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ)
    (z : Mode) (ha : 0 < a) (hk : 0 < k) (hδ : 0 < δ) (hc : 0 ≤ c)
    (hm : 0 < m) (hb : IsModeMass initial) (hbp : ∀ v, 0 < initial v)
    (hinit : ∀ v, v ≠ z → m ≤ modeLogOdds initial z v) (N : ℕ)
    (hflat : ∀ n < N, ∀ w, evidence n z = evidence n w) :
    ∀ w, w ≠ z → 0 < modeLogOdds
      (temperedFilterPath a k δ c initial evidence N) z w := by
  have hK : 0 < (Fintype.card Mode : ℝ) := by exact_mod_cast Fintype.card_pos
  have hpos := retainedMargin_flat_pos (Fintype.card Mode) a k δ c m hK ha hk hδ hm
  have hbnd := temperedFilter_retained_margin a k δ c m (fun _ => 0) initial
    evidence z N ha.le hk hδ.le hc hb hbp hinit
    (fun n _ => (hpos n).le) (fun n hn w _ => by simp [hflat n hn w])
  intro w hw
  exact (hpos N).trans_le (hbnd w hw)

theorem temperedFilter_quiet_window {Mode : Type} [Fintype Mode] [Nonempty Mode]
    (a k δ c m : ℝ) (initial : Mode → ℝ) (evidence : ℕ → Mode → ℝ)
    (z : Mode) (s N : ℕ) (ha : 0 < a) (hk : 0 < k) (hδ : 0 < δ) (hc : 0 ≤ c)
    (hm : 0 < m) (hb : IsModeMass initial) (hbp : ∀ v, 0 < initial v)
    (hinit : ∀ v, v ≠ z → m ≤
      modeLogOdds (temperedFilterPath a k δ c initial evidence s) z v)
    (hflat : ∀ n < N, ∀ w, evidence (s + n) z = evidence (s + n) w) :
    ∀ w, w ≠ z → 0 < modeLogOdds
      (temperedFilterPath a k δ c initial evidence (s + N)) z w := by
  rw [temperedFilterPath_restart a k δ c initial evidence s N]
  exact temperedFilter_quiet_retention a k δ c m _ _ z ha hk hδ hc hm
    (temperedFilterPath_mass a k δ c initial evidence hb s)
    (temperedFilterPath_pos a k δ c initial evidence hbp s) hinit N hflat

theorem inverse_quiet_retains_mode {d : ℕ} {Head Mode : Type}
    [Fintype Head] [Fintype Mode] [Nonempty Mode]
    (a k δ c m : ℝ) (initial : Mode → ℝ)
    (candidate : ℕ → Mode → Action d) (prediction : ℕ → Head → Action d)
    (variance : ℕ → Action d) (z : Mode)
    (ha : 0 < a) (hk : 0 < k) (hδ : 0 < δ) (hc : 0 ≤ c) (hm : 0 < m)
    (hb : IsModeMass initial) (hbp : ∀ v, 0 < initial v)
    (hinit : ∀ v, v ≠ z → m ≤ modeLogOdds initial z v) (N : ℕ)
    (hquiet : ∀ n < N, ∀ w, candidate n z = candidate n w) :
    ∀ w, w ≠ z → 0 < modeLogOdds
      (temperedFilterPath a k δ c initial
        (fun n => inverseEvidence (candidate n) (prediction n) (variance n)) N) z w := by
  exact temperedFilter_quiet_retention a k δ c m initial _ z ha hk hδ hc hm hb hbp
    hinit N (fun n hn w => inverseEvidence_same_candidates _ _ _ z w (hquiet n hn w))

end BAPR.ActionCompensation
