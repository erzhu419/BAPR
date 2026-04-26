"""BAPR: ESCP + BOCD Belief Tracker + Adaptive β — scan-fused.

Two belief modes (toggle via config.use_regime_belief):
  False (default, legacy): scalar BOCD ρ(h), Q sees [s, a, e, ρ] (20-dim belief)
  True  (Minimum redesign): joint b(h, z), Q sees [s, a, e, ρ, μ]
        (max_run_length + num_regimes dim; default 20+4=24)
        See common/regime_belief_tracker.py + GPT55_REVIEW_v3.md for theory.
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx
from copy import deepcopy

from jax_experiments.networks.policy import GaussianPolicy
from jax_experiments.networks.ensemble_critic import EnsembleCritic
from jax_experiments.networks.context_net import ContextNetwork, compute_rmdm_loss
from jax_experiments.common.belief_tracker import BeliefTracker, SurpriseComputer
from jax_experiments.common.regime_belief_tracker import RegimeBeliefTracker


class BAPR:
    """BAPR = ESCP + BOCD belief-weighted adaptive conservatism — scan-fused."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)

        # Belief conditioning (3 modes):
        #   use_regime_belief=True : Q sees [s, a, e, ρ, μ] — joint b(h, z)
        #   belief_conditioned=True : Q sees [s, a, e, ρ]    — scalar BOCD (v15)
        #   both False              : Q sees [s, a, e]       — pre-v15 BAPR
        self.use_regime_belief = bool(getattr(config, 'use_regime_belief', False))
        if self.use_regime_belief:
            self.belief_dim = config.max_run_length + config.num_regimes
        elif getattr(config, 'belief_conditioned', True):
            self.belief_dim = config.max_run_length
        else:
            self.belief_dim = 0
        self.aug_dim = config.ep_dim + self.belief_dim

        self.context_net = ContextNetwork(
            obs_dim, config.ep_dim, hidden_dim=128, rngs=self.rngs)
        self.policy = GaussianPolicy(
            obs_dim, act_dim, config.hidden_dim,
            ep_dim=self.aug_dim, n_layers=2, rngs=self.rngs)
        self.critic = EnsembleCritic(
            obs_dim + self.aug_dim, act_dim, config.hidden_dim,
            ensemble_size=config.ensemble_size, n_layers=3, rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

        self.log_alpha = jnp.array(jnp.log(config.alpha))
        self.target_entropy = -float(act_dim)

        self.policy_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)
        self.context_opt = optax.adam(config.lr)
        self.alpha_opt = optax.adam(config.lr)

        self.policy_opt_state = self.policy_opt.init(nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(nnx.state(self.critic, nnx.Param))
        self.context_opt_state = self.context_opt.init(nnx.state(self.context_net, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)

        # Both trackers always exist: legacy belief_tracker provides the scalar
        # λ_w (effective_window-based) used by the adaptive β path. The new
        # regime_tracker (when enabled) provides the joint b(h, z) used to
        # build the belief vector fed to Q/π.
        self.belief_tracker = BeliefTracker(
            max_run_length=config.max_run_length,
            hazard_rate=config.hazard_rate,
            base_variance=config.base_variance,
            variance_growth=config.variance_growth)
        self.surprise_computer = SurpriseComputer(ema_alpha=config.surprise_ema_alpha)

        if self.use_regime_belief:
            self.regime_tracker = RegimeBeliefTracker(
                max_run_length=config.max_run_length,
                num_regimes=config.num_regimes,
                hazard_rate=config.regime_hazard_rate,
                warmup_samples=config.regime_warmup_samples,
                ewma_alpha=config.regime_ewma_alpha,
                obs_dim=config.regime_obs_dim,
                seed=seed + 7919)
            # Per-chunk reward EMA → reward_z channel of y. Lives on BAPR
            # because it needs to span multiple iters; tracker itself is
            # regime-conditional and shouldn't carry global reward stats.
            self._chunk_rew_ema_mean: float = None
            self._chunk_rew_ema_var: float = None
            self._chunk_rew_ema_alpha: float = 0.05
        else:
            self.regime_tracker = None

        self.update_count = 0
        self._current_weighted_lambda = 0.0
        self._build_scan_fn()

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        beta_base = self.config.beta
        penalty_scale = self.config.penalty_scale
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        rbf_r = self.config.rbf_radius
        cons_w = self.config.consistency_loss_weight
        div_w = self.config.diversity_loss_weight

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)
        gd_ctx = nnx.graphdef(self.context_net)

        p_opt = self.policy_opt
        c_opt = self.critic_opt
        x_opt = self.context_opt
        a_opt = self.alpha_opt

        belief_dim = self.belief_dim

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         ctx_params, log_alpha,
                         c_opt_state, p_opt_state, x_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         all_task_ids, rng_key, warmup, weighted_lambda,
                         belief_vec):
            """BAPR scan with adaptive beta + belief-conditioned Q (v15).

            belief_vec: shape (max_run_length,) — current BOCD posterior ρ(h),
            broadcast to every batch sample. v15's belief-conditioned critic
            sees [obs, e, ρ] instead of [obs, e].
            """
            effective_beta = beta_base - weighted_lambda * penalty_scale

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, x_p, la, c_os, p_os, x_os, a_os, key) = carry
                (obs, act, rew, next_obs, done, tids) = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                # === Context RMDM ===
                def ctx_loss_fn(xp):
                    xm = nnx.merge(gd_ctx, xp)
                    return compute_rmdm_loss(xm(obs), tids, rbf_r, cons_w, div_w)

                x_loss, x_grads = jax.value_and_grad(ctx_loss_fn)(x_p)
                x_upd, new_x_os = x_opt.update(x_grads, x_os, x_p)
                new_x_p = optax.apply_updates(x_p, x_upd)

                ctx_m = nnx.merge(gd_ctx, new_x_p)
                ep = ctx_m(obs)
                ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
                next_ep = ctx_m(next_obs)
                next_ep = jnp.where(warmup, jnp.zeros_like(next_ep), next_ep)

                # v15: append belief vector ρ to context (broadcast over batch).
                # When belief_dim==0 (config disabled), this concat is a no-op.
                if belief_dim > 0:
                    bcast = jnp.broadcast_to(belief_vec[None, :], (obs.shape[0], belief_dim))
                    ep_full = jnp.concatenate([ep, bcast], axis=-1)
                    next_ep_full = jnp.concatenate([next_ep, bcast], axis=-1)
                else:
                    ep_full = ep
                    next_ep_full = next_ep

                obs_aug = jnp.concatenate([obs, ep_full], axis=-1)
                next_aug = jnp.concatenate([next_obs, next_ep_full], axis=-1)

                # === Critic (belief-conditioned) ===
                def critic_loss_fn(cp):
                    tm = nnx.merge(gd_target, t_p)
                    pm = nnx.merge(gd_policy, p_p)
                    na, nlp = pm.sample(next_obs, k1, next_ep_full)
                    tq = tm(next_aug, na).min(axis=0) - alpha * nlp
                    tv = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq
                    cm = nnx.merge(gd_critic, cp)
                    pq = cm(obs_aug, act)
                    return jnp.mean((pq - tv[None]) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy (adaptive β + belief-conditioned) ===
                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    na, lp = pm.sample(obs, k2, ep_full)
                    qv = cm(obs_aug, na)
                    lcb = qv.mean(axis=0) + effective_beta * qv.std(axis=0)
                    return (jnp.exp(la) * lp - lcb).mean(), lp

                (p_loss, lp), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                p_upd, new_p_os = p_opt.update(p_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, p_upd)

                # === Alpha ===
                a_grad = -(lp.mean() + target_entropy)
                a_upd, new_a_os = a_opt.update(a_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + a_upd, la)
                new_a_os = jax.tree.map(
                    lambda n, o: jnp.where(auto_alpha, n, o), new_a_os, a_os)

                # === Target ===
                new_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1 - tau) + cp * tau, t_p, new_c_p)

                new_carry = (new_c_p, new_t_p, new_p_p, new_x_p, new_la,
                             new_c_os, new_p_os, new_x_os, new_a_os, key)
                metrics = (c_loss, p_loss, x_loss, jnp.exp(new_la),
                           pq.mean(), pq.std(axis=0).mean(), lp.mean())
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    ctx_params, log_alpha,
                    c_opt_state, p_opt_state, x_opt_state, a_opt_state, rng_key)
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done,
                       all_task_ids)
            return jax.lax.scan(body_fn, init, batches)

        # JIT'd Q-std computation for surprise (called once per multi_update).
        # v15: also takes belief_vec so the critic input shape matches scan's.
        @jax.jit
        def _compute_q_stats(critic_params, ctx_params, obs, act, warmup,
                              belief_vec):
            cm = nnx.merge(gd_critic, critic_params)
            xm = nnx.merge(gd_ctx, ctx_params)
            ep = xm(obs)
            ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
            if belief_dim > 0:
                bcast = jnp.broadcast_to(
                    belief_vec[None, :], (obs.shape[0], belief_dim))
                ep_full = jnp.concatenate([ep, bcast], axis=-1)
            else:
                ep_full = ep
            oa = jnp.concatenate([obs, ep_full], axis=-1)
            pq = cm(oa, act)
            return pq.std(axis=0).mean()

        # ── Minimum BAPR redesign: chunked observable signals ─────────────
        # Splits a full rollout into n_chunks pieces, returns per-chunk
        # (q_std, td_resid) for the RegimeBeliefTracker. y_3 = (reward_mean,
        # q_std, td_resid) — reward_mean is computed in numpy outside.
        # n_chunks_static is closure-captured from config so reshape works
        # inside JIT (must be Python int, not traced).
        n_chunks_static = self.config.regime_chunks_per_iter

        @jax.jit
        def _compute_chunked_signals(critic_params, target_params, policy_params,
                                       ctx_params, full_obs, full_act, full_rew,
                                       full_next_obs, full_done, warmup,
                                       belief_vec):
            cm = nnx.merge(gd_critic, critic_params)
            tm = nnx.merge(gd_target, target_params)
            pm = nnx.merge(gd_policy, policy_params)
            xm = nnx.merge(gd_ctx, ctx_params)

            obs_c = full_obs.reshape((n_chunks_static, -1, full_obs.shape[-1]))
            act_c = full_act.reshape((n_chunks_static, -1, full_act.shape[-1]))
            rew_c = full_rew.reshape((n_chunks_static, -1))
            nobs_c = full_next_obs.reshape((n_chunks_static, -1, full_next_obs.shape[-1]))
            done_c = full_done.reshape((n_chunks_static, -1))

            def per_chunk(obs, act, rew, next_obs, done):
                ep = xm(obs)
                ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
                ep_next = xm(next_obs)
                ep_next = jnp.where(warmup, jnp.zeros_like(ep_next), ep_next)
                if belief_dim > 0:
                    bcast = jnp.broadcast_to(
                        belief_vec[None, :], (obs.shape[0], belief_dim))
                    ep_full = jnp.concatenate([ep, bcast], axis=-1)
                    ep_next_full = jnp.concatenate([ep_next, bcast], axis=-1)
                else:
                    ep_full = ep
                    ep_next_full = ep_next

                # Q-std (ensemble disagreement). EnsembleCritic returns (E, B)
                # — already squeezed the per-critic 1-dim head.
                oa = jnp.concatenate([obs, ep_full], axis=-1)
                pq = cm(oa, act)                              # (E, B)
                q_std = pq.std(axis=0).mean()                 # scalar
                q_pred_mean = pq.mean(axis=0)                 # (B,)

                # TD residual: |r + γ(1-d) Q_target(s', π_det(s')) - Q̄(s, a)|
                # Deterministic policy for the bootstrap action — avoids extra
                # sampling overhead and gives a stable signal channel.
                next_act = pm.deterministic(next_obs, ep_next_full)
                noa = jnp.concatenate([next_obs, ep_next_full], axis=-1)
                tq = tm(noa, next_act).mean(axis=0)           # (B,)
                td_target = rew + gamma * (1.0 - done) * tq
                td_err = jnp.abs(td_target - q_pred_mean).mean()

                return jnp.stack([q_std, td_err])             # (2,)

            return jax.vmap(per_chunk)(obs_c, act_c, rew_c, nobs_c, done_c)

        self._scan_update = _scan_update
        self._compute_q_stats = _compute_q_stats
        self._compute_chunked_signals = _compute_chunked_signals

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    def select_action(self, obs, deterministic=False):
        obs_jax = jnp.array(obs)[None] if np.asarray(obs).ndim == 1 else jnp.array(obs)
        ep = self.context_net(obs_jax)
        # Append current belief vector to ep so policy input matches scan.
        # Format selected by _build_belief_jax (use_regime_belief vs legacy).
        if self.belief_dim > 0:
            belief = self._build_belief_jax()
            bcast = jnp.broadcast_to(belief[None, :], (obs_jax.shape[0], self.belief_dim))
            ep = jnp.concatenate([ep, bcast], axis=-1)
        if deterministic:
            return np.array(self.policy.deterministic(obs_jax, ep)[0])
        key = self.rngs.params()
        action, _ = self.policy.sample(obs_jax, key, ep)
        return np.array(action[0])

    def reset_episode(self):
        self.belief_tracker.reset()
        self.surprise_computer.reset()
        if self.regime_tracker is not None:
            self.regime_tracker.reset()

    def _build_belief_jax(self):
        """Build the belief vector to feed Q/π based on current mode.

        - use_regime_belief=True: concat([ρ(h), μ(z)]) — H+K dim
        - belief_conditioned=True: ρ(h) — H dim (legacy v15)
        - both False: empty (legacy pre-v15)
        """
        if self.use_regime_belief:
            rho = self.regime_tracker.rho
            mu = self.regime_tracker.mu_marginal
            return jnp.asarray(np.concatenate([rho, mu]), dtype=jnp.float32)
        return jnp.asarray(self.belief_tracker.belief, dtype=jnp.float32)

    def _observe_chunked_rollout(self, recent_rollout, warmup_jax, belief_jax):
        """Split recent_rollout into N chunks; feed (r_z, q_std, td_resid)
        per chunk to the regime tracker. Runs once per multi_update."""
        full_obs = jnp.asarray(recent_rollout["obs"])         # (rollout_len, obs_dim)
        full_act = jnp.asarray(recent_rollout["act"])
        full_rew = jnp.asarray(recent_rollout["rew"]).reshape(-1)
        full_nobs = jnp.asarray(recent_rollout["next_obs"])
        full_done = jnp.asarray(recent_rollout["done"]).reshape(-1)

        signals = self._compute_chunked_signals(              # (n_chunks, 2)
            nnx.state(self.critic, nnx.Param),
            nnx.state(self.target_critic, nnx.Param),
            nnx.state(self.policy, nnx.Param),
            nnx.state(self.context_net, nnx.Param),
            full_obs, full_act, full_rew, full_nobs, full_done,
            warmup_jax, belief_jax)
        signals_np = np.asarray(signals)                       # (n_chunks, 2)

        n_chunks = self.config.regime_chunks_per_iter
        chunk_size = full_rew.shape[0] // n_chunks
        rew_per_chunk = np.asarray(full_rew).reshape(n_chunks, chunk_size).mean(axis=1)

        a = self._chunk_rew_ema_alpha
        for i in range(n_chunks):
            r_mean = float(rew_per_chunk[i])
            if self._chunk_rew_ema_mean is None:
                self._chunk_rew_ema_mean = r_mean
                self._chunk_rew_ema_var = 1.0
                r_z = 0.0
            else:
                old_mean = self._chunk_rew_ema_mean
                self._chunk_rew_ema_mean = a * r_mean + (1 - a) * old_mean
                self._chunk_rew_ema_var = a * (r_mean - old_mean) ** 2 \
                                          + (1 - a) * self._chunk_rew_ema_var
                std = max(float(np.sqrt(self._chunk_rew_ema_var)), 1e-3)
                r_z = abs(r_mean - self._chunk_rew_ema_mean) / std

            y = np.array([r_z, float(signals_np[i, 0]), float(signals_np[i, 1])],
                         dtype=np.float32)
            self.regime_tracker.observe(y)

    def _compute_weighted_lambda(self):
        """Compute belief-weighted penalty.

        BOCD dynamics: high surprise → belief pushed to high h (high variance
        explains outliers) → effective_window increases.
        So: λ_w = ew / H → high when environment just changed.
        """
        ew = self.belief_tracker.effective_window
        max_h = self.belief_tracker.max_H - 1
        return float(np.clip(ew / max_h, 0.0, 1.0))

    def multi_update(self, stacked_batch, current_iter=0, recent_rewards=None,
                     recent_rollout=None, **kwargs):
        rng_key = self.rngs.params()
        warmup = jnp.array(current_iter < self.config.context_warmup_iters)

        # Change 2 (GPT-5.5 v2): compute surprise from MOST RECENT rollout, not
        # from random replay batch. Random replay mixes old regimes, making
        # BOCD's changepoint detection unreliable.
        # belief_jax is the conditioning vector for Q/π — _build_belief_jax
        # picks the right format (legacy 20-dim ρ vs new 24-dim concat[ρ;μ]).
        belief_jax = self._build_belief_jax()

        if recent_rollout is not None:
            w = getattr(self.config, "surprise_window", 1024)
            robs = jnp.asarray(recent_rollout["obs"][-w:])
            ract = jnp.asarray(recent_rollout["act"][-w:])
            rrew = np.asarray(recent_rollout["rew"][-w:]).reshape(-1)
            q_std = self._compute_q_stats(
                nnx.state(self.critic, nnx.Param),
                nnx.state(self.context_net, nnx.Param),
                robs, ract, warmup, belief_jax)
            reward_signal = rrew
        else:
            # Fallback: replay batch (random exploration phase or compatibility)
            last_obs = jnp.asarray(stacked_batch["obs"][-1])
            last_act = jnp.asarray(stacked_batch["act"][-1])
            q_std = self._compute_q_stats(
                nnx.state(self.critic, nnx.Param),
                nnx.state(self.context_net, nnx.Param),
                last_obs, last_act, warmup, belief_jax)
            if recent_rewards is not None and len(recent_rewards) > 0:
                reward_signal = np.array(recent_rewards, dtype=np.float32)
            else:
                reward_signal = np.asarray(stacked_batch["rew"][-1]).reshape(-1)

        surprise = self.surprise_computer.compute(
            reward_signal, np.asarray([float(q_std)], dtype=np.float32))
        self.belief_tracker.update(surprise)

        # ── Minimum redesign: feed chunked observations to regime tracker.
        # Uses the PRE-update belief_jax (Q-net was conditioned on this for
        # the chunked Q-std / TD computations); fine because the new tracker
        # state is built from this iter's data and read back below.
        if self.use_regime_belief and recent_rollout is not None:
            self._observe_chunked_rollout(recent_rollout, warmup, belief_jax)

        # P0: BAPR mechanism warmup — λ_w=0 for first bapr_warmup_iters.
        if current_iter < self.config.bapr_warmup_iters:
            weighted_lambda = 0.0
        else:
            weighted_lambda = self._compute_weighted_lambda()
        self._current_weighted_lambda = weighted_lambda

        # Refresh belief vector AFTER update so scan uses post-update belief.
        # Format depends on flags: see _build_belief_jax docstring.
        belief_jax = self._build_belief_jax()

        # Run scan
        c_p = nnx.state(self.critic, nnx.Param)
        t_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)
        x_p = nnx.state(self.context_net, nnx.Param)

        obs = jnp.array(stacked_batch["obs"])
        act = jnp.array(stacked_batch["act"])
        rew = jnp.array(stacked_batch["rew"])
        nobs = jnp.array(stacked_batch["next_obs"])
        done = jnp.array(stacked_batch["done"])
        tids = jnp.array(stacked_batch["task_id"])

        final, metrics = self._scan_update(
            c_p, t_p, p_p, x_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state,
            self.context_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done, tids, rng_key, warmup,
            jnp.array(weighted_lambda), belief_jax)

        (new_c, new_t, new_p, new_x, new_la,
         self.critic_opt_state, self.policy_opt_state,
         self.context_opt_state, self.alpha_opt_state, _) = final
        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_t)
        nnx.update(self.policy, new_p)
        nnx.update(self.context_net, new_x)
        self.log_alpha = new_la

        n = obs.shape[0]
        self.update_count += n

        c_loss, p_loss, x_loss, alpha, qm, qs, lp = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "rmdm_loss": float(x_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(qm.mean()),
            "q_std_mean": float(qs.mean()),
            "log_prob": float(lp.mean()),
            "weighted_lambda": weighted_lambda,
            "belief_entropy": float(self.belief_tracker.entropy),
            "effective_window": float(self.belief_tracker.effective_window),
            "warmup": bool(warmup),
        }
