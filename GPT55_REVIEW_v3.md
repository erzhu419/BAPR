================================================================================
GPT-5.5 ROUND 3: ALGORITHMIC REDESIGN
================================================================================
# Executive recommendation

If you want novelty strong enough to change a NeurIPS decision, do **not** merely tune BOCD or add another scalar schedule. The key redesign should be:

> **Replace “BOCD → scalar λ → scalar β” with a full Bayesian regime belief used as an input, a replay index, and a posterior gate over mode-conditioned value functions.**

The deepest issue is not only scalar collapse. It is that current BAPR’s theory is written over **modes** \(m\), while the implementation uses BOCD over **run-lengths** \(h\). Run-length answers *when did something change?*; it does not answer *which dynamics regime am I in?* ESCP already learns a context representation for “which”. BAPR needs to fuse the two into a real posterior over:

\[
b_t(h,z) \approx P(\text{run-length}=h,\ \text{regime}=z \mid \mathcal H_t)
\]

and then use this belief everywhere important.

My top recommendation is to reframe the paper around:

> **Bayesian Regime Belief SAC:** a belief-state, posterior-gated robust SAC method for piecewise-stationary control.

The three most promising contributions are:

1. **Joint when/which Bayesian regime posterior** \(b_t(h,z)\), not scalar \(\lambda_w\).
2. **Posterior-gated robust actor-critic / low-rank MoE**, implementing the paper’s mixture-operator theorem rather than approximating it away.
3. **Posterior-kernel replay and belief-conditioned Bellman learning**, giving a clean stale-replay theory and an empirically useful mechanism.

These are implementable in 1–2 weeks if you use shared trunks + vectorized heads rather than fully independent networks.

---

# Task A — Critical assessment of A–H

## A. Per-mode \(\beta\) / per-mode policy

### Verdict

Promising, but only if reformulated as **posterior-gated low-rank experts**. Full \(M\)-independent critics/policies for \(M=20\) or \(40\) is expensive and risks looking like a generic MoE. The theoretical hook is real because it finally matches the theorem:

\[
\mathcal T_b Q = \sum_m b(m)\mathcal T_m Q.
\]

But the current BOCD posterior is over run-lengths \(h\), not physical modes \(m\). So per-mode policies require an additional mode posterior \(p(z)\), from context clustering or task labels in simulation.

### Theoretical hook

Strong if stated as:

> A posterior-gated mixture of frozen mode-conditional Bellman operators remains a \(\gamma\)-contraction, and scalar-\(\beta\) BAPR is a degenerate rank-1 approximation.

This is much stronger than “adaptive beta”.

### Minimal experiment

Implement \(Z=4\) or \(Z=8\) mode heads:

\[
Q_{i,z}(s,a), \qquad \pi_z(a\mid s)
\]

with shared trunks and vectorized heads. Gate using either:

- oracle task ID as an upper bound;
- context cluster posterior \(p(z\mid e_t)\);
- BOCD/context joint posterior \(b_t(h,z)\).

Compare:

1. ESCP,
2. v14 scalar BAPR,
3. BAPR + per-mode critic only,
4. BAPR + per-mode critic + per-mode actor,
5. oracle-gated experts.

Metrics: final return, post-switch recovery AUC, expert specialization \(I(z;\text{task})\), compute overhead.

### Failure mode

- Expert collapse: all data goes to one head.
- Wrong mode posterior: gating noise hurts more than scalar BAPR.
- Full independent policies are costly and may be perceived as “just MoE”.
- If regimes are continuous gravity values rather than discrete clusters, hard mode heads may be too coarse.

### Scores

| Criterion | Score |
|---|---:|
| Novelty | 7/10 |
| Feasibility in 2 weeks | 5/10 for full independent, 7/10 for shared-trunk heads |
| Expected empirical lift | 6.5/10 |

---

## B. BOCD belief-conditioned \(Q(s,a,\rho)\)

### Verdict

This is one of the best low-cost redesigns. It directly addresses scalar collapse by feeding the full posterior into the actor/critic:

\[
Q_\phi(s,a,e,\rho), \qquad \pi_\theta(a\mid s,e,\rho).
\]

But it is only theoretically clean if you store the belief in replay. Otherwise, off-policy updates incorrectly evaluate old transitions under the current belief.

### Theoretical hook

Strong:

> In a hidden piecewise-stationary MDP, the Bayesian belief is a sufficient statistic. The belief-state soft Bellman operator is a \(\gamma\)-contraction.

Define belief-state Bellman operator:

\[
(\mathcal T Q)(s,b,a)
=
\bar r_b(s,a)
+
\gamma
\mathbb E_{s',y\sim b}
\left[
V_Q(s',\Phi(b,y))
\right],
\]

where \(b\) is the BOCD/regime posterior and \(\Phi\) is the Bayesian filter update. This is more principled than adaptive scalar \(\beta\).

### Minimal experiment

Add belief vector to replay:

\[
(s_t,a_t,r_t,s_{t+1},d_t,e_t,\rho_t,\rho_{t+1}).
\]

Train:

\[
Q_\phi(s,a,e,\rho), \qquad \pi_\theta(a\mid s,e,\rho).
\]

Ablations:

- \(Q(s,a,e)\): ESCP.
- \(Q(s,a,e,\mathbb E[h])\): scalar run-length mean.
- \(Q(s,a,e,\rho)\): full BOCD vector.
- \(Q(s,a,e,\rho)\) + belief-kernel replay.

### Failure mode

- BOCD run-length posterior alone may not identify mode.
- \(\rho\) may be low-dimensional but semantically weak if detector is bad.
- If belief is updated only once per 4000-step rollout, it is coarse.
- Without replay storage, off-policy inconsistency can erase gains.

### Scores

| Criterion | Score |
|---|---:|
| Novelty | 8/10 |
| Feasibility in 2 weeks | 8/10 |
| Expected empirical lift | 6/10 |

---

## C. Risk-sensitive RL via distributional \(Q\)

### Verdict

Good theoretical story, but implementation risk is higher. This makes BAPR different from ESCP because the BOCD posterior controls a **risk functional**, not just an LCB coefficient.

Instead of:

\[
\bar Q + \beta \sigma_Q,
\]

use a distributional critic \(Z_\phi(s,a,e,b)\) and a posterior-dependent spectral risk measure:

\[
\rho_b[Z]
=
\int_0^1 F_Z^{-1}(\tau)\, w_b(\tau)\,d\tau.
\]

Early after a change, \(w_b\) emphasizes lower quantiles; late in stable regimes, \(w_b\) approaches median/mean.

### Theoretical hook

Moderately strong to strong:

> Posterior-dependent spectral risk Bellman operators remain contractions under frozen belief/risk distortion.

For quantile critic \(Z_\phi\):

\[
\mathcal L_{\text{QR}}
=
\sum_{i,j}
\rho_{\tau_i}^{\kappa}
\left(
r+\gamma z_j'(s',a') - z_i(s,a)
\right).
\]

Policy objective:

\[
\mathcal L_\pi
=
\mathbb E
\left[
\alpha \log \pi(a\mid s,e,b)
-
\rho_b[Z_\phi(s,a,e,b)]
\right].
\]

### Minimal experiment

Use \(N_q=25\) quantiles, ensemble size maybe reduce from 10 to 5. Define:

\[
q_b
=
q_{\min}
+
(q_{\text{med}}-q_{\min})
\cdot
\frac{\mathbb E[h]}{H-1}
\]

or better:

\[
w_b(\tau)
=
\operatorname{BetaPDF}(\tau;\alpha_b,\beta_b),
\]

where high change probability gives \(\alpha_b<1,\beta_b>1\), concentrating weight near low quantiles.

Compare:

- LCB BAPR,
- fixed CVaR,
- BOCD-adaptive quantile,
- ESCP.

### Failure mode

- Distributional SAC in continuous action is less stable.
- Quantile critic adds compute and hyperparameters.
- It addresses risk sensitivity, not mode identification.
- If current performance is limited by representation/gating, quantiles may not lift much.

### Scores

| Criterion | Score |
|---|---:|
| Novelty | 8/10 |
| Feasibility in 2 weeks | 5/10 |
| Expected empirical lift | 6.5/10 |

---

## D. Information-theoretic objective

### Verdict

High novelty, high risk. This could be a very interesting paper if framed as **active regime identification**, but it may hurt locomotion returns if probing actions are unsafe or unnecessary.

A possible objective:

\[
J(\pi)
=
\mathbb E
\sum_t
\gamma^t
\left[
r_t
+
\eta I(z_{t+1};a_t\mid s_t,b_t)
-
\lambda H(z_t\mid \mathcal H_t)
\right].
\]

Equivalent one-step intrinsic reward:

\[
r_t^{\text{info}}
=
\eta
\left[
H(b_t^z)
-
\mathbb E_{s_{t+1}}
H(b_{t+1}^z)
\right].
\]

### Theoretical hook

Strong if you can show improved identification delay or value-of-information bounds:

> The agent selects actions that maximize expected reduction in posterior regime entropy, yielding faster adaptation after switches.

This differentiates sharply from ESCP.

### Minimal experiment

Train a mode discriminator:

\[
q_\omega(z\mid s,a,s')
\]

or transition-prediction ensemble. Add intrinsic reward for first \(K\) steps after high change probability:

\[
r'_t
=
r_t
+
\eta \left[
H(q_\omega(z\mid \mathcal H_t))
-
H(q_\omega(z\mid \mathcal H_t,s_t,a_t,s_{t+1}))
\right].
\]

Metrics:

- detection delay,
- posterior entropy decay,
- recovery AUC,
- return drop due to exploration.

### Failure mode

- Active probing can destabilize control.
- Requires a good mode posterior.
- Hard to show return lift in 1–2 weeks.
- Reviewers may ask why information gain is needed if passive signals suffice.

### Scores

| Criterion | Score |
|---|---:|
| Novelty | 9/10 |
| Feasibility in 2 weeks | 5/10 |
| Expected empirical lift | 5/10 |

---

## E. Replace BOCD with simpler/stronger detection

### Verdict

Not enough by itself. Detector replacement does not solve scalar collapse. CUSUM may improve robustness but reduces novelty. Neural changepoint detection is publishable only if deeply integrated with control.

### Sub-options

#### CUSUM

\[
S_t = \max(0, S_{t-1} + \xi_t - \nu),
\qquad
\text{change if } S_t > c.
\]

- Novelty low.
- Feasibility high.
- Good ablation baseline.

#### GP changepoint

Better for smooth parameter drift, but expensive and mismatched to high-throughput JAX RL.

#### Neural changepoint

Train:

\[
p_\omega(c_t=1\mid \tau_{t-L:t})
\]

on rollout windows. Could be useful as auxiliary detection but not core novelty.

#### VAE / contrastive novelty detection

Potentially good if tied to context embeddings and regime posterior.

### Minimal experiment

Hold the RL algorithm fixed and compare detector quality:

- BOCD,
- CUSUM,
- reward-only,
- Q-std-only,
- neural CP head.

Metrics:

- CP F1,
- detection delay,
- false positives,
- return recovery.

### Failure mode

- Better detection does not necessarily produce better policy.
- CUSUM may be dismissed as engineering.
- Neural detector can overfit simulator schedule.

### Scores

| Criterion | Score |
|---|---:|
| Novelty | 4/10 overall |
| Feasibility in 2 weeks | 8/10 |
| Expected empirical lift | 3.5/10 |

---

## F. Joint training of BOCD likelihood with the agent

### Verdict

Very novel, but too risky for this timeline if you literally meta-gradient through policy performance. However, a simplified learnable likelihood is feasible:

\[
p_\eta(\xi_t\mid h,z)
=
\mathcal N(\xi_t;\mu_{\eta,h,z},\Sigma_{\eta,h,z})
\]

trained by predictive likelihood or pseudo-labels.

### Theoretical hook

Potentially strong:

> Differentiable Bayesian filtering trained end-to-end for control.

But if you meta-gradient through RL updates, complexity explodes.

### Minimal experiment

Do not meta-gradient through the entire RL loop. Instead learn likelihood parameters \(\eta\) by maximizing online predictive likelihood:

\[
\mathcal L_{\text{filter}}
=
-\log
\sum_{h,z}
b_{t-1}(h,z)
p_\eta(y_t\mid h,z).
\]

Then compare fixed BOCD likelihood vs learned likelihood.

### Failure mode

- End-to-end gradients through BOCD and SAC are unstable.
- Learned detector may overfit reward scale.
- Theory becomes messy if belief depends on current Q through learned likelihood.

### Scores

| Criterion | Score |
|---|---:|
| Novelty | 9/10 |
| Feasibility in 2 weeks | 3/10 |
| Expected empirical lift | 5.5/10 |

---

## G. Changepoint as auxiliary task

### Verdict

Good practical add-on, but not enough as the main contribution. It can improve context embeddings and detector calibration.

Add head:

\[
g_\omega(e_t) \to \hat p(c_t=1), \quad \hat \rho_t(h)
\]

and train with BOCD posterior pseudo-labels:

\[
\mathcal L_{\text{CP}}
=
\operatorname{CE}\left(
\operatorname{stopgrad}(\rho_t),
\hat \rho_t
\right)
\]

or hazard label:

\[
\mathcal L_{\text{hazard}}
=
-\hat c_t \log g_\omega(e_t)
-
(1-\hat c_t)\log(1-g_\omega(e_t)).
\]

### Theoretical hook

Weak to moderate:

> Auxiliary prediction aligns representation with the sufficient statistic for non-stationarity.

Useful but likely not NeurIPS-novel alone.

### Minimal experiment

Train context net with:

\[
\mathcal L_{\text{ctx}}
=
\mathcal L_{\text{RMDM}}
+
\lambda_{\text{CP}}\mathcal L_{\text{CP}}.
\]

Measure:

- CP prediction accuracy,
- embedding clustering,
- return,
- adaptation speed.

### Failure mode

- BOCD pseudo-labels are noisy.
- The context network is currently stateless \(e=\phi(s)\); changepoint needs history.
- It may just regularize ESCP, not differentiate BAPR enough.

### Scores

| Criterion | Score |
|---|---:|
| Novelty | 5/10 |
| Feasibility in 2 weeks | 8/10 |
| Expected empirical lift | 4/10 |

---

## H. BAPR as MoE-of-RESACs

### Verdict

This is probably the strongest listed direction, provided you avoid full independent networks. Use shared trunk + posterior-gated heads/adapters. This directly addresses the scalar collapse and matches the mixture-operator theorem.

### Theoretical hook

Strong:

\[
\mathcal T_b Q
=
\sum_{z=1}^Z b(z)\mathcal T_z Q_z
\]

with frozen \(b\) is a contraction. Also gives a clean uncertainty decomposition:

\[
\operatorname{Var}_{z,i}[Q_{i,z}]
=
\underbrace{\mathbb E_z \operatorname{Var}_i[Q_{i,z}]}_{\text{within-mode epistemic}}
+
\underbrace{\operatorname{Var}_z(\mathbb E_i Q_{i,z})}_{\text{between-mode ambiguity}}.
\]

That second term is a genuinely Bayesian regime-uncertainty term. ESCP does not have this.

### Minimal experiment

Implement \(Z=4\) expert heads:

\[
Q_{i,z}(s,e,a), \quad \pi_z(a\mid s,e).
\]

Gate with:

\[
\mu_t(z)=\sum_h b_t(h,z).
\]

Policy loss:

\[
\mathcal L_\pi
=
\sum_z
\mu_t(z)
\mathbb E_{a_z\sim\pi_z}
\left[
\alpha \log \pi_z(a_z\mid s,e)
-
\operatorname{LCB}_b(s,a_z)
\right].
\]

Compare:

- scalar BAPR,
- MoE with uniform gates,
- MoE with oracle gates,
- MoE with learned posterior gates.

### Failure mode

- More compute.
- Gate collapse.
- Continuous regimes may not cluster cleanly.
- Need a credible non-oracle posterior over \(z\).

### Scores

| Criterion | Score |
|---|---:|
| Novelty | 8.5/10 |
| Feasibility in 2 weeks | 6.5/10 |
| Expected empirical lift | 7/10 |

---

# Task B — Additional proposals not in A–H

## Proposal 1 — Joint run-length/regime posterior \(b_t(h,z)\)

Current BAPR uses \(\rho(h)\), a posterior over run-length. But the theorem and policy need a posterior over modes. Introduce:

\[
b_t(h,z)
=
P(h_t=h,z_t=z\mid \mathcal H_t).
\]

Here:

- \(h\): time since last changepoint.
- \(z\): latent regime cluster.

Use context embeddings \(e_t=\phi(s_t)\), reward residuals, TD residuals, and Q-ensemble disagreement as observations.

A simple online filter:

\[
\tilde b_{t+1}(h+1,z)
=
(1-H)
b_t(h,z)
\ell_z(y_{t+1}),
\]

\[
\tilde b_{t+1}(0,z')
=
H
\pi_0(z'\mid e_{t+1})
\sum_{h,z}
b_t(h,z)
\ell_{z'}(y_{t+1}),
\]

\[
b_{t+1}
=
\frac{\tilde b_{t+1}}{\sum_{h,z}\tilde b_{t+1}(h,z)}.
\]

The likelihood can be:

\[
\ell_z(y)
=
\mathcal N(y;\mu_z,\Sigma_z),
\]

where:

\[
y_t
=
[
r\text{-zscore},
\ Q\text{-std ratio},
\ \|e_t-c_z\|^2,
\ |\delta_t|
].
\]

The categorical regime posterior is:

\[
\mu_t(z)
=
\sum_h b_t(h,z).
\]

The run-length posterior is:

\[
\rho_t(h)
=
\sum_z b_t(h,z).
\]

### Why this is a real contribution

It resolves the theoretical mismatch:

- BOCD detects **when**.
- Context identifies **which**.
- BAPR needs both.

This can become the core paper idea.

### Implementation cost

Moderate. You can implement a `RegimeBeliefTracker` in NumPy in 1–2 days. Start with \(Z=4\), \(H=20\). The belief vector has only \(80\) entries.

---

## Proposal 2 — Posterior-kernel replay

Current belief-aware replay is still scalar:

\[
\text{recent\_frac}=\lambda_w.
\]

Replace it with full posterior compatibility. Store belief \(b_i\) with every transition. At update time with current belief \(b_t\), sample transition \(i\) with probability:

\[
p_t(i)
\propto
\epsilon
+
\exp
\left(
-
\frac{D_{\mathrm{JS}}(b_i,b_t)}{\tau_b}
\right).
\]

Or factorized:

\[
p_t(i)
\propto
\epsilon
+
\exp
\left(
-
\frac{D_{\mathrm{JS}}(\mu_i,\mu_t)}{\tau_z}
-
\frac{D_{\mathrm{JS}}(\rho_i,\rho_t)}{\tau_h}
\right).
\]

This is more principled than recency. Recency assumes old data is stale. Posterior-kernel replay says old data is useful if it came from a similar inferred regime.

### Theoretical hook

Stale replay induces Bellman bias. If the Bellman operator is Lipschitz in belief:

\[
\|\mathcal T_{b_i}Q-\mathcal T_{b_t}Q\|_\infty
\le
L_b
\|b_i-b_t\|_1,
\]

then the replay-induced bias is bounded by:

\[
\left\|
\mathbb E_{i\sim p_t}\mathcal T_{b_i}Q
-
\mathcal T_{b_t}Q
\right\|_\infty
\le
L_b
\mathbb E_{i\sim p_t}
\|b_i-b_t\|_1.
\]

Posterior-kernel replay directly minimizes the upper bound while preserving sample diversity.

### Implementation cost

Low to moderate. Add `belief` to replay buffer and implement weighted sampling. If fully weighted JAX sampling is annoying, sample a larger candidate pool and resample with NumPy probabilities.

---

## Proposal 3 — Between-mode epistemic uncertainty decomposition

Existing ensemble uncertainty only measures disagreement among critics:

\[
\sigma_{\text{ens}}^2(s,a)
=
\operatorname{Var}_{i}
Q_i(s,a).
\]

But in non-stationary RL, uncertainty has two components:

\[
\operatorname{Var}_{z,i}[Q_{z,i}]
=
\underbrace{
\mathbb E_{z\sim \mu}
\operatorname{Var}_i[Q_{z,i}]
}_{\text{within-mode epistemic}}
+
\underbrace{
\operatorname{Var}_{z\sim \mu}
\left[
\mathbb E_i Q_{z,i}
\right]
}_{\text{between-mode ambiguity}}.
\]

Use the posterior-gated LCB:

\[
\operatorname{LCB}_b(s,a)
=
\bar Q_b(s,a)
-
c_{\text{in}}
\sqrt{
\mathbb E_z \operatorname{Var}_i[Q_{z,i}]
}
-
c_{\text{btw}}
\sqrt{
\operatorname{Var}_z(\bar Q_z)
}.
\]

This is much more publishable than scalar \(\beta\). It says BAPR is not just “more conservative after surprise”; it is robust to **posterior ambiguity over regimes**.

### Implementation cost

Moderate if you already have expert heads. No new detector required.

---

# Task C — Top-3 picks for paper

I would build the paper around the following three. They are synergistic and feasible.

---

# Top Pick 1 — Joint Bayesian regime belief \(b_t(h,z)\)

## Core idea

Replace the scalar mapping:

\[
\rho(h)
\rightarrow
\lambda_w
\rightarrow
\beta_{\mathrm{eff}}
\]

with a joint Bayesian belief:

\[
b_t(h,z)
=
P(h_t=h,z_t=z\mid \mathcal H_t).
\]

This belief represents both:

- **when** the regime changed: \(h\),
- **which** regime is active: \(z\).

The marginal posteriors are:

\[
\rho_t(h)=\sum_z b_t(h,z),
\qquad
\mu_t(z)=\sum_h b_t(h,z).
\]

The key shift is that BAPR no longer treats BOCD as a scalar alarm. It becomes a belief-state RL method.

---

## Mathematical formulation

Let \(z_t\in\{1,\dots,Z\}\) denote a latent regime cluster. Let \(h_t\in\{0,\dots,H-1\}\) be the run-length. Let \(y_t\) be an observable surprise/context statistic:

\[
y_t
=
\left[
z_r(t),
\frac{\sigma_Q(t)}{\bar\sigma_Q(t)},
\|\phi(s_t)-c_1\|^2,
\dots,
\|\phi(s_t)-c_Z\|^2,
|\delta_t|
\right].
\]

Mode likelihood:

\[
\ell_z(y_t)
=
p(y_t\mid z_t=z)
=
\mathcal N(y_t;\mu_z,\Sigma_z).
\]

BOCD-style recursion:

\[
\tilde b_{t+1}(h+1,z)
=
(1-H_{\mathrm{cp}})
b_t(h,z)
\ell_z(y_{t+1}),
\]

\[
\tilde b_{t+1}(0,z')
=
H_{\mathrm{cp}}
\pi_0(z'\mid e_{t+1})
\sum_{h,z}
b_t(h,z)
\ell_{z'}(y_{t+1}),
\]

\[
b_{t+1}(h,z)
=
\frac{
\tilde b_{t+1}(h,z)
}{
\sum_{h',z'}
\tilde b_{t+1}(h',z')
}.
\]

Here \(\pi_0(z'\mid e_{t+1})\) is a prior over the new regime. Minimal implementation:

\[
\pi_0(z\mid e)
\propto
\exp
\left(
-\frac{\|e-c_z\|^2}{\tau_c}
\right).
\]

The regime prototypes \(c_z\) can be updated online:

\[
c_z
\leftarrow
(1-\eta_z)c_z
+
\eta_z
\frac{
\sum_t \mu_t(z)e_t
}{
\sum_t \mu_t(z)
}.
\]

---

## Pseudocode

```python
# New: common/regime_belief_tracker.py

class RegimeBeliefTracker:
    def __init__(self, H=20, Z=4, hazard=0.05, y_dim=None):
        self.H = H
        self.Z = Z
        self.hazard = hazard
        self.b = np.ones((H, Z), dtype=np.float32) / (H * Z)

        self.mu_y = np.zeros((Z, y_dim), dtype=np.float32)
        self.var_y = np.ones((Z, y_dim), dtype=np.float32)
        self.ctx_proto = np.random.randn(Z, ep_dim).astype(np.float32)
        self.ctx_proto /= np.linalg.norm(self.ctx_proto, axis=-1, keepdims=True) + 1e-8

    def mode_prior(self, ctx):
        # ctx: [ep_dim]
        dist2 = ((self.ctx_proto - ctx[None]) ** 2).sum(-1)
        logits = -dist2 / self.ctx_tau
        return softmax(logits)

    def likelihood(self, y):
        # diagonal Gaussian likelihood per mode
        diff2 = (y[None] - self.mu_y) ** 2
        logp = -0.5 * (diff2 / self.var_y + np.log(2 * np.pi * self.var_y)).sum(-1)
        return softmax_logsafe(logp)  # or exp(logp) with clipping

    def update(self, y, ctx):
        ell = self.likelihood(y)            # [Z]
        pi0 = self.mode_prior(ctx)          # [Z]

        old = self.b
        new = np.zeros_like(old)

        # Growth: h -> h+1, same z
        for h in range(self.H - 1):
            new[h + 1, :] += (1 - self.hazard) * old[h, :] * ell

        # Changepoint: reset h=0, sample new z
        evidence = np.sum(old * ell[None, :])
        new[0, :] += self.hazard * evidence * pi0

        new_sum = new.sum()
        if new_sum < 1e-10:
            new = np.ones_like(new) / new.size
        else:
            new /= new_sum
        self.b = new

        # responsibilities for mode update
        mu_z = self.b.sum(axis=0)
        self.update_prototypes(y, ctx, mu_z)

    @property
    def runlength_posterior(self):
        return self.b.sum(axis=1)

    @property
    def mode_posterior(self):
        return self.b.sum(axis=0)

    @property
    def flat_belief(self):
        return self.b.reshape(-1)
```

Training loop change:

```python
# After rollout, compute recent context/surprise feature y_t
ctx_recent = np.array(agent.context_net(jnp.array(last_obs)[None]))[0]
y_t = build_filter_observation(
    recent_rewards=ep_rewards,
    q_std=q_std,
    td_error=td_error,
    ctx=ctx_recent,
)

agent.regime_belief.update(y_t, ctx_recent)

belief = agent.regime_belief.flat_belief
mode_posterior = agent.regime_belief.mode_posterior
runlength_posterior = agent.regime_belief.runlength_posterior

# Pass full belief into update and store in replay
metrics = agent.multi_update(
    stacked,
    current_iter=iteration,
    belief=belief,
    mode_posterior=mode_posterior,
    runlength_posterior=runlength_posterior,
)
```

---

## Headline theoretical claim

**Theorem: Belief-state contraction for piecewise-stationary MDPs.**

Let \(b_t(h,z)\) be the Bayesian posterior over run-length and latent regime. Define the belief-state soft Bellman operator:

\[
(\mathcal T Q)(s,b,a)
=
\bar r_b(s,a)
+
\gamma
\mathbb E_{z\sim \mu_b}
\mathbb E_{s'\sim P_z(\cdot\mid s,a)}
\left[
V_Q(s',\Phi(b,y'))
\right],
\]

where:

\[
\bar r_b(s,a)
=
\sum_z \mu_b(z) r_z(s,a),
\qquad
\mu_b(z)=\sum_h b(h,z),
\]

and \(\Phi\) is the Bayesian filtering update. Then:

\[
\|\mathcal T Q_1-\mathcal T Q_2\|_\infty
\le
\gamma
\|Q_1-Q_2\|_\infty.
\]

This gives you a much cleaner paper claim:

> BAPR converts a non-stationary control problem into a stationary belief-MDP using a Bayesian run-length/regime posterior.

---

## Experimental design

### Environments

Use current four:

- HalfCheetah-v2,
- Hopper-v2,
- Walker2d-v2,
- Ant-v2.

Add harder variants:

1. **Multi-parameter non-stationarity**: gravity + body mass + damping.
2. **Random switching intervals**: Poisson or geometric around 40k steps.
3. **Recurring regimes**: same physical parameters reappear later.
4. **Novel held-out regimes**: test tasks sampled from wider log-scale.

### Metrics

- final return,
- post-switch recovery AUC,
- detection delay,
- mode-posterior calibration,
- entropy decay \(H(\mu_t)\),
- run-length posterior accuracy,
- false-positive rate,
- context/mode mutual information \(I(z;\text{task id})\).

### Ablations

- scalar BAPR \( \lambda_w \),
- run-length-only belief \( \rho(h) \),
- mode-only belief \( \mu(z) \),
- joint belief \(b(h,z)\),
- oracle mode posterior upper bound,
- random posterior control.

---

## Risk assessment

| Risk | Mitigation |
|---|---|
| Posterior over \(z\) collapses | entropy regularizer, prototype DPP loss, soft responsibilities |
| Continuous regimes do not form discrete modes | use \(Z=8\), soft gating, report clustering as approximation |
| Detector remains noisy | compare fixed BOCD, CUSUM, and learned likelihood |
| Implementation complexity | first implement factorized \(b(h,z)=\rho(h)\mu(z)\), then joint recursion |

---

# Top Pick 2 — Posterior-gated robust experts with uncertainty decomposition

## Core idea

Implement the theorem rather than approximating it with scalar \(\beta\). Use \(Z\) mode-conditioned expert heads:

\[
Q_{i,z}(s,a,e), \qquad \pi_z(a\mid s,e),
\]

and gate them using the posterior mode belief:

\[
\mu_t(z)
=
\sum_h b_t(h,z).
\]

This can be done cheaply with shared trunks and vectorized heads.

---

## Mathematical formulation

For each ensemble member \(i\) and regime expert \(z\):

\[
Q_{i,z}(s,a,e)
=
g_{i,z}(f_\phi(s,a,e)).
\]

The posterior-averaged value is:

\[
\bar Q_b(s,a)
=
\sum_z
\mu_b(z)
\bar Q_z(s,a),
\]

where:

\[
\bar Q_z(s,a)
=
\frac{1}{K}
\sum_{i=1}^K
Q_{i,z}(s,a).
\]

Now define the total posterior epistemic uncertainty:

\[
\sigma_b^2(s,a)
=
\underbrace{
\sum_z
\mu_b(z)
\operatorname{Var}_{i}
[
Q_{i,z}(s,a)
]
}_{\text{within-mode epistemic}}
+
\underbrace{
\sum_z
\mu_b(z)
\left(
\bar Q_z(s,a)-\bar Q_b(s,a)
\right)^2
}_{\text{between-mode ambiguity}}.
\]

Then use a posterior LCB:

\[
\operatorname{LCB}_b(s,a)
=
\bar Q_b(s,a)
+
\beta_{\text{in}}
\sqrt{\sigma_{\text{within}}^2(s,a)}
+
\beta_{\text{btw}}
\sqrt{\sigma_{\text{between}}^2(s,a)}.
\]

Because \(\beta_{\text{in}},\beta_{\text{btw}}<0\), the agent becomes conservative both when critics disagree within a mode and when the posterior remains ambiguous across modes.

This is far stronger than scalar \(\beta_{\mathrm{eff}}\). It creates a new uncertainty decomposition specific to non-stationary RL.

---

## Actor objective

Option 1: posterior mixture policy:

\[
\pi_b(a\mid s,e)
=
\sum_z
\mu_b(z)
\pi_z(a\mid s,e).
\]

Exact mixture log-prob is:

\[
\log \pi_b(a\mid s,e)
=
\log
\sum_z
\mu_b(z)
\pi_z(a\mid s,e).
\]

Policy loss:

\[
\mathcal L_\pi
=
\mathbb E_{s\sim \mathcal D,\ a\sim \pi_b}
\left[
\alpha \log \pi_b(a\mid s,e)
-
\operatorname{LCB}_b(s,a)
\right].
\]

Option 2, easier and more stable: weighted expert policy loss:

\[
\mathcal L_\pi
=
\sum_z
\mu_b(z)
\mathbb E_{a_z\sim \pi_z}
\left[
\alpha\log\pi_z(a_z\mid s,e)
-
\operatorname{LCB}_b(s,a_z)
\right].
\]

Deployment action can be:

\[
a_t
=
\sum_z
\mu_t(z)
\tanh m_z(s_t,e_t)
\]

for deterministic eval, or sample \(z\sim \mu_t\), then sample \(a\sim\pi_z\) for stochastic rollout.

---

## Critic objective

For transition \(j\), store posterior \(\mu_j\). Train expert \(z\) with responsibility weight \(w_{j,z}\). Minimal:

\[
w_{j,z}
=
\mu_j(z).
\]

Critic target:

\[
y_j
=
r_j
+
\gamma(1-d_j)
\left[
\operatorname{LCB}_{b_j'}^{-}(s_j',a_j')
-
\alpha \log \pi_{b_j'}(a_j'\mid s_j',e_j')
\right].
\]

Critic loss:

\[
\mathcal L_Q
=
\mathbb E_{j\sim\mathcal D}
\sum_z
w_{j,z}
\sum_i
\left(
Q_{i,z}(s_j,a_j,e_j)-y_j
\right)^2.
\]

For a more mode-pure version, use per-mode targets:

\[
y_{j,z}
=
r_j
+
\gamma(1-d_j)
\left[
\bar Q_z^{-}(s_j',a_j')
-
\alpha\log\pi_z(a_j'\mid s_j',e_j')
\right].
\]

But the shared posterior target is simpler.

---

## Pseudocode

```python
# networks/regime_critic.py

class RegimeEnsembleCritic(nnx.Module):
    """
    Outputs Q-values with shape [ensemble_size, num_modes, batch].
    Shared trunk + vectorized mode heads.
    """
    def __init__(self, obs_aug_dim, act_dim, hidden_dim, ensemble_size, num_modes, rngs):
        self.ensemble_size = ensemble_size
        self.num_modes = num_modes
        self.trunk = SharedCriticTrunk(obs_aug_dim + act_dim, hidden_dim, rngs=rngs)
        self.heads = VectorizedModeHeads(
            hidden_dim, 1,
            ensemble_size=ensemble_size,
            num_modes=num_modes,
            rngs=rngs
        )

    def __call__(self, obs_aug, act):
        x = jnp.concatenate([obs_aug, act], axis=-1)
        h = self.trunk(x)
        q = self.heads(h)  # [K, Z, B, 1]
        return q.squeeze(-1)
```

Inside scan update:

```python
# qv: [K, Z, B]
qv = critic(obs_aug, action)

# mode posterior: [B, Z] or [Z] broadcasted
mu = batch_mode_posterior

q_mean_z = qv.mean(axis=0)          # [Z, B]
q_var_within_z = qv.var(axis=0)     # [Z, B]

q_mean_b = (mu.T * q_mean_z).sum(axis=0)  # [B]

within = (mu.T * q_var_within_z).sum(axis=0)

between = (mu.T * (q_mean_z - q_mean_b[None, :]) ** 2).sum(axis=0)

lcb = (
    q_mean_b
    + beta_in * jnp.sqrt(within + 1e-6)
    + beta_btw * jnp.sqrt(between + 1e-6)
)

policy_loss = (alpha * log_prob - lcb).mean()
```

---

## Headline theoretical claim

**Theorem: Posterior-gated expert Bellman contraction.**

Let each mode-conditional robust soft Bellman operator \(\mathcal T_z\) be a \(\gamma\)-contraction. For any frozen posterior \(\mu\in\Delta_Z\), define:

\[
\mathcal T_\mu Q
=
\sum_z
\mu(z)
\mathcal T_z Q_z.
\]

Then:

\[
\|\mathcal T_\mu Q-\mathcal T_\mu Q'\|_\infty
\le
\gamma
\max_z
\|Q_z-Q_z'\|_\infty.
\]

Moreover, the posterior variance decomposes as:

\[
\operatorname{Var}_{z,i}[Q_{i,z}]
=
\mathbb E_z\operatorname{Var}_i[Q_{i,z}]
+
\operatorname{Var}_z(\mathbb E_iQ_{i,z}),
\]

yielding a principled separation between epistemic uncertainty and regime ambiguity.

This gives the paper a genuinely new robust RL object.

---

## Experimental design

### Main comparison

- ESCP,
- RESAC,
- v14 scalar BAPR,
- belief-conditioned BAPR,
- posterior-gated expert BAPR,
- oracle-gated expert BAPR.

### Key plots

1. Posterior \(\mu_t(z)\) over time with true mode shading.
2. Within-mode vs between-mode uncertainty after switches.
3. Expert specialization heatmap:
   \[
   P(z_{\text{expert}}\mid \text{true task id}).
   \]
4. Post-switch recovery curves aligned at changepoints.
5. Performance vs number of experts \(Z\in\{2,4,8\}\).

### Strong ablation

Remove between-mode uncertainty term:

\[
\operatorname{LCB}_b
=
\bar Q_b
+
\beta_{\text{in}}\sigma_{\text{within}}.
\]

If full method wins, you can claim regime ambiguity matters.

---

## Risk assessment

| Risk | Mitigation |
|---|---|
| Compute grows by \(Z\) | shared trunk, vectorized heads, \(Z=4\) first |
| MoE collapse | entropy regularization on \(\mu\), balanced prototype updates |
| Policy mixture log-prob hard | use weighted expert loss instead |
| Continuous task parameters | soft posterior, more experts, report approximation |

---

# Top Pick 3 — Posterior-kernel replay and belief-conditioned Bellman learning

## Core idea

Current replay is belief-aware only through a scalar recent fraction. Replace it with posterior-indexed replay.

Each replay transition stores:

\[
(s_i,a_i,r_i,s_i',d_i,e_i,b_i,b_i').
\]

At update time with current belief \(b_t\), sample or weight transitions according to posterior similarity:

\[
p_t(i)
\propto
\epsilon
+
\exp
\left(
-
\frac{D_{\mathrm{JS}}(b_i,b_t)}{\tau_b}
\right).
\]

This allows old but relevant regimes to be reused and stale but recent transitions to be downweighted.

---

## Mathematical formulation

Define a belief-conditioned Bellman operator:

\[
\mathcal T_b Q
=
r_b
+
\gamma P_b V_Q.
\]

Assume Lipschitz continuity in belief:

\[
\|\mathcal T_b Q-\mathcal T_{b'}Q\|_\infty
\le
L_b
D(b,b').
\]

If replay samples are drawn from distribution \(p_t(i)\), the replay Bellman operator is:

\[
\widehat{\mathcal T}_{p_t}Q
=
\mathbb E_{i\sim p_t}
[
\mathcal T_{b_i}Q
].
\]

Then:

\[
\left\|
\widehat{\mathcal T}_{p_t}Q
-
\mathcal T_{b_t}Q
\right\|_\infty
\le
L_b
\mathbb E_{i\sim p_t}
D(b_i,b_t).
\]

Thus posterior-kernel replay directly controls replay-induced non-stationarity bias.

This is a very clean theoretical contribution because it formalizes the stale replay problem.

---

## Sampling rule

Use factorized posterior distance:

\[
D(b_i,b_t)
=
\lambda_z
D_{\mathrm{JS}}(\mu_i,\mu_t)
+
\lambda_h
D_{\mathrm{JS}}(\rho_i,\rho_t)
+
\lambda_e
\|e_i-e_t\|^2.
\]

Sampling probability:

\[
p_t(i)
=
(1-\eta)\frac{1}{|\mathcal D|}
+
\eta
\frac{
\exp(-D(b_i,b_t)/\tau)
}{
\sum_j\exp(-D(b_j,b_t)/\tau)
}.
\]

Alternatively, avoid custom sampling and use loss weights:

\[
w_i
=
\exp(-D(b_i,b_t)/\tau),
\]

\[
\mathcal L_Q
=
\frac{
\sum_i w_i
(Q(s_i,a_i,b_i)-y_i)^2
}{
\sum_i w_i
}.
\]

Loss weighting is easier to implement in JAX.

---

## Pseudocode

Replay buffer additions:

```python
self.belief = jnp.zeros((capacity, belief_dim), dtype=jnp.float32)
self.next_belief = jnp.zeros((capacity, belief_dim), dtype=jnp.float32)
self.mode_post = jnp.zeros((capacity, num_modes), dtype=jnp.float32)
self.run_post = jnp.zeros((capacity, max_run_length), dtype=jnp.float32)
```

Push:

```python
belief_batch = jnp.broadcast_to(current_belief[None], (n_steps, belief_dim))
mode_batch = jnp.broadcast_to(current_mu[None], (n_steps, num_modes))
run_batch = jnp.broadcast_to(current_rho[None], (n_steps, max_H))

replay_buffer.push_batch_jax(
    obs, act, rew, nobs, done, task_ids,
    belief=belief_batch,
    mode_post=mode_batch,
    run_post=run_batch,
)
```

Weighted loss:

```python
def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

# batch_mu: [B, Z]
# current_mu: [Z]
D_z = vmap(lambda mu_i: js_div(mu_i, current_mu))(batch_mu)

# batch_rho: [B, H]
D_h = vmap(lambda rho_i: js_div(rho_i, current_rho))(batch_rho)

w = jnp.exp(-(lambda_z * D_z + lambda_h * D_h) / tau_replay)
w = w / (w.mean() + 1e-6)

critic_loss = (w[None, :] * (pred_q - target_q[None]) ** 2).mean()
```

---

## Headline theoretical claim

**Theorem: Posterior-kernel replay bounds stale Bellman bias.**

In a piecewise-stationary MDP with belief-indexed Bellman operators \(\{\mathcal T_b\}\), if \(\mathcal T_b\) is \(L_b\)-Lipschitz in belief, then the bias induced by replay distribution \(p_t\) is bounded by:

\[
\operatorname{Bias}(p_t)
=
\left\|
\mathbb E_{i\sim p_t}\mathcal T_{b_i}Q
-
\mathcal T_{b_t}Q
\right\|_\infty
\le
L_b
\mathbb E_{i\sim p_t}D(b_i,b_t).
\]

Scalar recent replay is a special case where \(D\) depends only on time index. Posterior-kernel replay strictly generalizes recency.

---

## Experimental design

### Main experiment

Compare:

1. uniform replay,
2. recent replay,
3. scalar-\(\lambda\) replay,
4. posterior-kernel replay with \(\rho(h)\),
5. posterior-kernel replay with \(\mu(z)\),
6. posterior-kernel replay with full \(b(h,z)\).

### Metrics

- replay belief mismatch:
  \[
  \mathbb E_{i\sim p_t}D(b_i,b_t),
  \]
- critic TD error after switches,
- recovery AUC,
- final return,
- sample reuse from recurring regimes.

### Special environment

Use recurring regimes:

\[
A \rightarrow B \rightarrow C \rightarrow A.
\]

Recent replay discards old \(A\) data. Posterior-kernel replay should retrieve it. This is a clean “money plot”.

---

## Risk assessment

| Risk | Mitigation |
|---|---|
| Weighted replay reduces diversity | mixture with uniform floor \(\epsilon\) |
| Belief posterior inaccurate | compare oracle belief upper bound |
| JAX weighted sampling complexity | use loss weights first |
| Extra memory | belief dim small; \(H=20,Z=4\Rightarrow80\) floats/transition |

---

# Practical 1–2 week implementation plan

## Week 1: belief-conditioned + posterior replay

1. Add `RegimeBeliefTracker`.
2. Add belief/mode/run posterior to replay buffer.
3. Concatenate belief to actor/critic inputs:
   \[
   e_{\text{aug}}=[e,\rho,\mu] \quad \text{or} \quad [e,\operatorname{vec}(b)].
   \]
4. Implement posterior-kernel loss weights.
5. Run HalfCheetah + Hopper, 3 seeds.

This already gives a differentiated method.

## Week 2: posterior-gated expert heads

1. Replace `EnsembleCritic` with `RegimeEnsembleCritic` outputting `[K,Z,B]`.
2. Add between-mode uncertainty term.
3. Add simple multi-head policy or keep shared actor conditioned on belief.
4. Run:
   - scalar BAPR,
   - belief BAPR,
   - posterior replay,
   - expert BAPR,
   - oracle-gated upper bound.

If time is short, make the critic expert-gated first and keep one shared belief-conditioned actor. That is already publishable.

---

# Task D — Paper narrative reframe

## Recommended new title

**Bayesian Regime Belief SAC: Posterior-Gated Robust Reinforcement Learning for Piecewise-Stationary Control**

Alternative stronger titles:

1. **From Change Detection to Belief-State Control: Posterior-Gated Robust RL under Regime Shifts**
2. **Posterior-Gated Robust Actor-Critic for Piecewise-Stationary Reinforcement Learning**
3. **Bayesian Regime Belief Reinforcement Learning with Posterior-Kernel Replay**
4. **Beyond Adaptive Conservatism: Belief-Indexed Robust Control for Non-Stationary RL**

My favorite:

> **Bayesian Regime Belief SAC: Posterior-Gated Robust RL for Piecewise-Stationary Control**

It clearly signals the new contribution: not scalar BAPR, but belief-state posterior-gated RL.

---

## New abstract

Real-world control systems often exhibit piecewise-stationary dynamics: the environment remains stable for extended periods before abruptly switching to a new regime. Existing robust RL methods are typically either globally conservative, sacrificing performance during stable periods, or context-adaptive without explicitly reasoning about changepoints and stale replay. We propose **Bayesian Regime Belief SAC**, a posterior-gated robust actor-critic method for piecewise-stationary control. Instead of collapsing changepoint uncertainty into a scalar penalty, our method maintains a joint Bayesian belief \(b_t(h,z)\) over both run-length \(h\), indicating when the last change occurred, and latent regime \(z\), indicating which dynamics mode is currently active. This belief is used as a state variable, a replay index, and a gate over mode-conditioned robust value functions.

Our method introduces three algorithmic components. First, a joint run-length/regime filter fuses BOCD-style changepoint inference with context embeddings and critic uncertainty to produce calibrated posterior beliefs. Second, a posterior-gated ensemble critic decomposes uncertainty into within-mode epistemic uncertainty and between-mode regime ambiguity, yielding a principled lower-confidence policy objective. Third, posterior-kernel replay samples transitions by belief similarity rather than recency, directly reducing stale Bellman backup bias. Theoretically, we show that the belief-state robust Bellman operator is a \(\gamma\)-contraction under frozen posterior beliefs, that posterior-gated mixtures of mode-conditioned operators preserve contraction, and that replay-induced non-stationarity bias is bounded by the expected posterior mismatch between sampled transitions and the current belief.

Across non-stationary MuJoCo locomotion benchmarks with abrupt and recurring dynamics shifts, Bayesian Regime Belief SAC improves post-switch recovery, reduces false conservatism during stable periods, and outperforms scalar adaptive-conservatism baselines. Ablations show that full posterior conditioning, between-mode uncertainty, and posterior-kernel replay each contribute to improved adaptation. Our results suggest that effective non-stationary RL requires not only detecting that a change occurred, but maintaining and exploiting a structured belief over when and which regime is active.

---

# Final prioritization

If you only implement one thing:

> **Implement \(Q(s,a,e,\rho,\mu)\) with belief stored in replay and posterior-kernel loss weighting.**

If you implement two things:

> Add **posterior-gated critic heads** and the within/between uncertainty decomposition.

If you implement three things:

> Add the **joint \(b(h,z)\) filter** and make the paper about Bayesian regime belief, not adaptive scalar conservatism.

This is the redesign most likely to make reviewers say:

> “This is no longer ESCP plus a BOCD scalar. This is a principled belief-state robust RL algorithm for piecewise-stationary environments.”

================================================================================
Tokens: prompt=73351, completion=2513
