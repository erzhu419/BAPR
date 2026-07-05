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

        self.adaptation_mode = getattr(config, "bapr_adaptation_mode", "gate")

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
        self.ema_policy = deepcopy(self.policy)
        self.critic = EnsembleCritic(
            obs_dim + self.aug_dim, act_dim, config.hidden_dim,
            ensemble_size=config.ensemble_size, n_layers=3, rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

        self.log_alpha = jnp.array(jnp.log(config.alpha))
        self.target_entropy = -float(act_dim)

        def make_opt():
            clip_norm = float(getattr(config, "bapr_grad_clip_norm", 0.0))
            if clip_norm > 0.0:
                return optax.chain(
                    optax.clip_by_global_norm(clip_norm),
                    optax.adam(config.lr))
            return optax.adam(config.lr)

        self.policy_opt = make_opt()
        self.critic_opt = make_opt()
        self.context_opt = make_opt()
        self.alpha_opt = make_opt()

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
        self.surprise_computer = SurpriseComputer(
            ema_alpha=config.surprise_ema_alpha,
            reward_weight=getattr(config, "surprise_reward_weight", 0.5),
            q_weight=getattr(config, "surprise_q_weight", 0.3),
            reg_weight=getattr(config, "surprise_reg_weight", 0.2))

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
        self._adaptation_gate = 0.0
        self._raw_adaptation_gate = 0.0
        self._last_surprise = 0.0
        self._last_q_std_mean = 0.0
        self._last_q_abs_mean = 0.0
        self._last_q_std_ratio = 0.0
        self._current_recent_replay_frac = 0.0
        self._current_reg_scale = 1.0
        self._current_actor_lcb_scale = 0.0
        self._reg_latched = False
        self._last_ema_rollout_active = False
        self._train_reward_ema = None
        self._train_reward_peak = None
        self._eval_reward_ema = None
        self._eval_reward_peak = None
        self._last_train_reward_drop_frac = 0.0
        self._last_eval_reward_drop_frac = 0.0
        self._last_train_reward_improve_frac = 0.0
        self._last_eval_reward_improve_frac = 0.0
        self._last_perf_drop_frac = 0.0
        self._reg_perf_collapse_active = False
        self._actor_lcb_perf_active = False
        self._actor_lcb_qstd_active = False
        self._actor_lcb_active = False
        self._controller_active = False
        self._controller_latched = False
        self._controller_active_iters = 0
        self._controller_cooldown_iters = 0
        self._controller_duration_released = False
        self._controller_exit_reason = 0.0
        self._controller_signal = 0.0
        self._controller_drawdown = 0.0
        self._controller_low_disagreement = False
        self._controller_reg_multiplier = 1.0
        self._controller_recent_multiplier = 1.0
        self._controller_recent_add_frac = 0.0
        self._controller_lcb_multiplier = 1.0
        self._controller_actor_update_multiplier = 1.0
        self._build_scan_fn()

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        ema_tau = float(getattr(self.config, "ema_tau", tau))
        beta_base = self.config.beta
        penalty_scale = self.config.penalty_scale
        weight_reg = float(getattr(self.config, "weight_reg", 0.0))
        beta_ood = float(getattr(self.config, "beta_ood", 0.0))
        actor_objective = getattr(self.config, "actor_objective", "lcb")
        target_mode = getattr(self.config, "critic_target_mode", "min")
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        rbf_r = self.config.rbf_radius
        cons_w = self.config.consistency_loss_weight
        div_w = self.config.diversity_loss_weight
        residual_delta = float(getattr(self.config, "bapr_residual_delta", 0.25))
        residual_gate_scale = float(getattr(
            self.config, "bapr_residual_gate_scale", 1.0))
        residual_adv_margin = float(getattr(
            self.config, "bapr_residual_adv_margin", 0.0))
        residual_adv_temp = max(float(getattr(
            self.config, "bapr_residual_adv_temp", 300.0)), 1e-6)
        residual_qstd_scale = float(getattr(
            self.config, "bapr_residual_qstd_scale", 0.5))
        residual_behavior_weight = float(getattr(
            self.config, "bapr_residual_behavior_weight", 0.05))
        residual_action_penalty = float(getattr(
            self.config, "bapr_residual_action_penalty", 0.0))

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
                         ema_policy_params,
                         ctx_params, log_alpha,
                         c_opt_state, p_opt_state, x_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         all_task_ids, rng_key, warmup, weighted_lambda,
                         belief_vec, all_beliefs, reg_scale, actor_lcb_scale,
                         actor_update_scale):
            """BAPR scan with adaptive beta + belief-conditioned Q.

            belief_vec: shape (belief_dim,) — current BOCD posterior, used as
              fallback when per-transition belief is unavailable (e.g.
              random exploration phase).
            all_beliefs: shape (n_iters, batch_size, belief_dim) — per-
              transition belief stored at rollout time (GPT-5.5 advice #2).
              When belief_dim==0 this is a zero-width tensor and unused.
            """
            effective_beta = beta_base - weighted_lambda * penalty_scale

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, ema_p, x_p, la,
                 c_os, p_os, x_os, a_os, key) = carry
                (obs, act, rew, next_obs, done, tids, b) = batch_data
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

                # GPT-5.5 advice #2: per-transition belief from replay buffer.
                # `b` is [B, belief_dim] holding the belief active when each
                # transition was generated, not the current iter's belief.
                # When belief_dim==0, b is zero-width and the concat is a no-op.
                if belief_dim > 0:
                    ep_full = jnp.concatenate([ep, b], axis=-1)
                    next_ep_full = jnp.concatenate([next_ep, b], axis=-1)
                else:
                    ep_full = ep
                    next_ep_full = next_ep

                obs_aug = jnp.concatenate([obs, ep_full], axis=-1)
                next_aug = jnp.concatenate([next_obs, next_ep_full], axis=-1)

                # === Critic (belief-conditioned) ===
                # GPT-5.5 advice #4: target_mode = "min" (legacy BAPR LCB —
                # ensemble collapses to lowest member) or "independent" (each
                # Q_i learns own target — ESCP-like, preserves disagreement).
                def critic_loss_fn(cp):
                    tm = nnx.merge(gd_target, t_p)
                    pm = nnx.merge(gd_policy, p_p)
                    na, nlp = pm.sample(next_obs, k1, next_ep_full)
                    reg_bonus = reg_scale * weight_reg * tm.compute_reg_norm()[:, None]
                    tq_all = tm(next_aug, na) + reg_bonus  # [K, B]
                    if target_mode == "min":
                        tq = tq_all.min(axis=0) - alpha * nlp
                        tv = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq
                        cm = nnx.merge(gd_critic, cp)
                        pq = cm(obs_aug, act)
                        mse = jnp.mean((pq - tv[None]) ** 2)
                        return mse + reg_scale * beta_ood * pq.std(axis=0).mean(), pq
                    else:                                    # "independent"
                        tv_all = (rew.squeeze(-1) + gamma * (1 - done.squeeze(-1))
                                  * (tq_all - alpha * nlp))   # [K, B]
                        cm = nnx.merge(gd_critic, cp)
                        pq = cm(obs_aug, act)
                        mse = jnp.mean((pq - tv_all) ** 2)
                        return mse + reg_scale * beta_ood * pq.std(axis=0).mean(), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy (adaptive β + belief-conditioned) ===
                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    na, lp = pm.sample(obs, k2, ep_full)
                    target_reg = (
                        reg_scale * weight_reg
                        * nnx.merge(gd_target, t_p).compute_reg_norm()[:, None])
                    residual_gate_mean = jnp.array(0.0, dtype=obs.dtype)
                    residual_adv_mean = jnp.array(0.0, dtype=obs.dtype)
                    residual_norm_mean = jnp.array(0.0, dtype=obs.dtype)
                    if actor_objective in (
                            "conservative_residual",
                            "residual_advantage",
                            "v80_residual"):
                        # v80: treat a zero-context EMA policy as the stable
                        # base. The online context-conditioned actor can only
                        # affect the objective through a bounded residual, and
                        # that residual is stopped unless conservative
                        # Q-advantage over the base is positive enough.
                        base_pm = nnx.merge(gd_policy, ema_p)
                        zero_ep = jnp.zeros_like(ep_full)
                        base_action = jax.lax.stop_gradient(
                            base_pm.deterministic(obs, zero_ep))
                        prop_residual = jnp.clip(
                            na - base_action,
                            -residual_delta, residual_delta)
                        prop_action = jnp.clip(
                            base_action + prop_residual,
                            -1.0 + 1e-6, 1.0 - 1e-6)

                        base_qv = cm(obs_aug, base_action) + target_reg
                        prop_qv = cm(obs_aug, prop_action) + target_reg
                        base_safe = (
                            base_qv.mean(axis=0)
                            - residual_qstd_scale * base_qv.std(axis=0))
                        prop_safe = (
                            prop_qv.mean(axis=0)
                            - residual_qstd_scale * prop_qv.std(axis=0))
                        adv = prop_safe - base_safe
                        gate = residual_gate_scale * jax.nn.sigmoid(
                            (adv - residual_adv_margin) / residual_adv_temp)
                        gate = jnp.clip(gate, 0.0, 1.0)
                        gate = jax.lax.stop_gradient(gate)
                        mixed_action = jnp.clip(
                            base_action + gate[:, None] * prop_residual,
                            -1.0 + 1e-6, 1.0 - 1e-6)
                        qv = cm(obs_aug, mixed_action) + target_reg
                        safe_q = (
                            qv.mean(axis=0)
                            - residual_qstd_scale * qv.std(axis=0))
                        residual_norm = jnp.sum(
                            jnp.square(mixed_action - base_action), axis=-1)
                        behavior_loss = jnp.mean(
                            (1.0 - gate)
                            * jnp.sum(jnp.square(na - base_action), axis=-1))
                        action_penalty = jnp.mean(residual_norm)
                        residual_gate_mean = gate.mean()
                        residual_adv_mean = adv.mean()
                        residual_norm_mean = residual_norm.mean()
                        loss = (
                            (jnp.exp(la) * lp - safe_q).mean()
                            + residual_behavior_weight * behavior_loss
                            + residual_action_penalty * action_penalty)
                        return loss, (lp, residual_gate_mean,
                                      residual_adv_mean, residual_norm_mean)

                    qv = cm(obs_aug, na) + target_reg
                    q_mean = qv.mean(axis=0)
                    q_std = qv.std(axis=0)
                    if actor_objective == "mean":
                        actor_q = q_mean
                    elif actor_objective == "ucb":
                        actor_q = q_mean + jnp.abs(effective_beta) * q_std
                    elif actor_objective == "gated_lcb":
                        actor_q = q_mean + beta_base * weighted_lambda * q_std
                    elif actor_objective == "reg_gated_lcb":
                        actor_q = q_mean + beta_base * reg_scale * q_std
                    elif actor_objective == "qstd_gated_lcb":
                        actor_q = q_mean + beta_base * actor_lcb_scale * q_std
                    else:
                        actor_q = q_mean + effective_beta * q_std
                    loss = (jnp.exp(la) * lp - actor_q).mean()
                    return loss, (lp, residual_gate_mean,
                                  residual_adv_mean, residual_norm_mean)

                (p_loss, p_aux), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                lp, residual_gate_mean, residual_adv_mean, residual_norm_mean = p_aux
                p_grads = jax.tree.map(
                    lambda g: g * actor_update_scale, p_grads)
                p_upd, new_p_os = p_opt.update(p_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, p_upd)
                new_ema_p = jax.tree.map(
                    lambda ep, pp: ep * (1.0 - ema_tau) + pp * ema_tau,
                    ema_p, new_p_p)

                # === Alpha ===
                a_grad = -(lp.mean() + target_entropy)
                a_upd, new_a_os = a_opt.update(a_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + a_upd, la)
                new_a_os = jax.tree.map(
                    lambda n, o: jnp.where(auto_alpha, n, o), new_a_os, a_os)

                # === Target ===
                new_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1 - tau) + cp * tau, t_p, new_c_p)

                new_carry = (new_c_p, new_t_p, new_p_p, new_ema_p,
                             new_x_p, new_la,
                             new_c_os, new_p_os, new_x_os, new_a_os, key)
                metrics = (c_loss, p_loss, x_loss, jnp.exp(new_la),
                           pq.mean(), pq.std(axis=0).mean(), lp.mean(),
                           residual_gate_mean, residual_adv_mean,
                           residual_norm_mean)
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    ema_policy_params,
                    ctx_params, log_alpha,
                    c_opt_state, p_opt_state, x_opt_state, a_opt_state, rng_key)
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done,
                       all_task_ids, all_beliefs)
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
            return pq.std(axis=0).mean(), jnp.abs(pq.mean(axis=0)).mean()

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
        if self.belief_dim == 0:
            return jnp.zeros((0,), dtype=jnp.float32)
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
        """Compute legacy belief-weighted penalty.

        BOCD dynamics: high surprise → belief pushed to high h (high variance
        explains outliers) → effective_window increases.
        So: λ_w = ew / H → high when environment just changed.
        """
        ew = self.belief_tracker.effective_window
        max_h = self.belief_tracker.max_H - 1
        return float(np.clip(ew / max_h, 0.0, 1.0))

    def _update_adaptation_gate(self, surprise: float):
        """Bounded surprise gate for redesigned BAPR.

        The gate is intentionally much simpler than BOCD belief conditioning:
        it is not a critic input and does not change Bellman targets. It only
        controls how much recent replay is sampled and, when explicitly
        requested, how much gated LCB risk the actor sees.
        """
        threshold = float(getattr(self.config, "bapr_surprise_threshold", 0.8))
        gain = float(getattr(self.config, "bapr_gate_gain", 0.5))
        gate_max = float(getattr(self.config, "bapr_gate_max", 0.8))
        alpha = float(getattr(self.config, "bapr_gate_ema_alpha", 0.2))

        raw = np.clip((float(surprise) - threshold) * gain, 0.0, gate_max)
        self._raw_adaptation_gate = float(raw)
        self._adaptation_gate = float(
            np.clip(alpha * raw + (1.0 - alpha) * self._adaptation_gate,
                    0.0, gate_max))
        return self._adaptation_gate

    def _compute_recent_replay_frac(self, weighted_lambda, q_std, q_abs_mean):
        cap = float(getattr(self.config, "bapr_recent_frac_cap", 0.0))
        frac = float(min(cap, max(0.0, float(weighted_lambda))))
        floor = float(min(cap, max(
            0.0, float(getattr(self.config, "bapr_recent_frac_floor", 0.0)))))
        true_floor = bool(getattr(self.config, "bapr_recent_true_floor", False))
        floor_mode = str(getattr(
            self.config, "bapr_recent_floor_mode", "always")).lower()
        q_std = float(q_std)
        q_abs_mean = max(float(q_abs_mean), 1e-6)
        ratio = q_std / q_abs_mean
        self._last_q_std_ratio = float(ratio)

        if cap <= 0.0:
            return 0.0
        if frac <= 0.0 and (not true_floor or floor <= 0.0):
            return self._apply_controller_to_recent_frac(0.0, cap)

        if not getattr(self.config, "bapr_recent_disagreement_gate", False):
            return self._apply_controller_to_recent_frac(max(frac, floor), cap)

        def gated_floor():
            if floor_mode == "always":
                return floor
            if floor_mode == "reg_latched":
                return floor if bool(getattr(self, "_reg_latched", False)) else 0.0
            if floor_mode == "low_disagreement":
                return 0.0
            if floor_mode == "ratio_schedule":
                low = float(getattr(
                    self.config, "bapr_recent_floor_ratio_low", 0.02))
                high = float(getattr(
                    self.config, "bapr_recent_floor_ratio_high", 0.04))
                if ratio <= low:
                    return floor
                if ratio <= high:
                    mid_floor = float(getattr(
                        self.config, "bapr_recent_floor_mid_frac", floor))
                    return float(min(cap, max(0.0, mid_floor)))
                extreme_floor = float(getattr(
                    self.config, "bapr_recent_floor_extreme_frac", 0.0))
                return float(min(cap, max(0.0, extreme_floor)))
            return floor

        if (getattr(self.config, "bapr_recent_open_if_reg_latched", False)
                and bool(getattr(self, "_reg_latched", False))):
            return self._apply_controller_to_recent_frac(
                max(frac, gated_floor()), cap)

        qstd_ok = q_std <= float(getattr(
            self.config, "bapr_recent_qstd_threshold", 5.0))
        ratio_ok = ratio <= float(getattr(
            self.config, "bapr_recent_qstd_ratio_threshold", 0.02))
        if qstd_ok and ratio_ok:
            return self._apply_controller_to_recent_frac(
                max(frac, gated_floor()), cap)
        # Keep a small amount of recent data available under high disagreement.
        # Full shutoff made Walker/Hopper lose recovery signal after switches.
        return self._apply_controller_to_recent_frac(gated_floor(), cap)

    def recent_replay_fraction(self):
        return float(getattr(self, "_current_recent_replay_frac", 0.0))

    def _update_reward_drop(self, value, ema_attr, peak_attr):
        """Track fractional drawdown from the best smoothed reward so far."""
        if value is None:
            return 0.0
        value = float(value)
        alpha = float(getattr(
            self.config, "bapr_actor_lcb_perf_ema_alpha", 0.20))
        ema = getattr(self, ema_attr)
        prev_ema = ema
        if ema is None:
            ema = value
        else:
            ema = alpha * value + (1.0 - alpha) * float(ema)
        peak = getattr(self, peak_attr)
        if peak is None:
            peak = ema
        else:
            peak = max(float(peak), float(ema))
        setattr(self, ema_attr, float(ema))
        setattr(self, peak_attr, float(peak))
        denom = max(abs(float(peak)), 1e-6)
        improve = 0.0 if prev_ema is None else (float(ema) - float(prev_ema)) / denom
        if ema_attr == "_train_reward_ema":
            self._last_train_reward_improve_frac = float(improve)
        elif ema_attr == "_eval_reward_ema":
            self._last_eval_reward_improve_frac = float(improve)
        return float(max(0.0, (float(peak) - float(ema)) / denom))

    def _observe_train_reward(self, recent_rewards):
        if recent_rewards is None or len(recent_rewards) == 0:
            return
        self._last_train_reward_drop_frac = self._update_reward_drop(
            float(np.mean(recent_rewards)),
            "_train_reward_ema",
            "_train_reward_peak")

    def _reset_algorithm_controller(self):
        self._controller_active = False
        self._controller_latched = False
        self._controller_active_iters = 0
        self._controller_cooldown_iters = 0
        self._controller_duration_released = False
        self._controller_exit_reason = 0.0
        self._controller_signal = 0.0
        self._controller_drawdown = 0.0
        self._controller_low_disagreement = False
        self._controller_reg_multiplier = 1.0
        self._controller_recent_multiplier = 1.0
        self._controller_recent_add_frac = 0.0
        self._controller_lcb_multiplier = 1.0
        self._controller_actor_update_multiplier = 1.0

    def _update_algorithm_controller(self, q_std, q_abs_mean, current_iter):
        mode = str(getattr(self.config, "bapr_controller_mode", "off")).lower()
        if mode in ("", "off", "none", "disabled"):
            self._reset_algorithm_controller()
            return

        warmup_iters = int(getattr(
            self.config, "bapr_controller_warmup_iters", 120))
        if current_iter < warmup_iters:
            self._reset_algorithm_controller()
            return

        q_std = float(q_std)
        q_abs_mean = max(float(q_abs_mean), 1e-6)
        ratio = q_std / q_abs_mean
        self._last_q_std_ratio = float(ratio)

        drop_frac = max(
            float(getattr(self, "_last_train_reward_drop_frac", 0.0)),
            float(getattr(self, "_last_eval_reward_drop_frac", 0.0)))
        train_peak = getattr(self, "_train_reward_peak", None)
        eval_peak = getattr(self, "_eval_reward_peak", None)
        peak = max(float(train_peak or 0.0), float(eval_peak or 0.0))
        min_peak = float(getattr(
            self.config, "bapr_controller_min_peak", 100.0))

        low_ratio_th = float(getattr(
            self.config, "bapr_controller_low_qstd_ratio", 0.02))
        low_qstd_th = float(getattr(
            self.config, "bapr_controller_low_qstd_threshold", 0.0))
        ratio_ok = True if low_ratio_th <= 0.0 else ratio <= low_ratio_th
        qstd_ok = True if low_qstd_th <= 0.0 else q_std <= low_qstd_th
        low_disagreement = bool(ratio_ok and qstd_ok)

        drop_threshold = float(getattr(
            self.config, "bapr_controller_drop_frac", 0.9))
        drop_ramp = max(float(getattr(
            self.config, "bapr_controller_drop_ramp", 0.4)), 1e-6)
        signal = float(np.clip(
            (drop_frac - drop_threshold) / drop_ramp, 0.0, 1.0))
        entry_active = bool(peak >= min_peak and low_disagreement and signal > 0.0)
        soft_mode = mode in ("soft", "soft_recovery")
        if soft_mode:
            prev_signal = float(getattr(self, "_controller_signal", 0.0))
            decay = float(np.clip(getattr(
                self.config, "bapr_controller_soft_decay", 0.92), 0.0, 0.999))
            min_signal = float(max(0.0, getattr(
                self.config, "bapr_controller_soft_min_signal", 0.01)))
            signal = float(max(signal if entry_active else 0.0,
                               prev_signal * decay))
            active = bool(signal > min_signal)
            latch_enabled = False
        else:
            active = entry_active
            latch_enabled = bool(getattr(self.config, "bapr_controller_latch", False))
        release_drop = float(getattr(
            self.config, "bapr_controller_release_drop_frac", 0.45))
        if drop_frac <= release_drop:
            self._controller_duration_released = False
        if bool(getattr(self, "_controller_duration_released", False)):
            entry_active = False
            active = False
            signal = 0.0

        if latch_enabled and entry_active:
            self._controller_latched = True
        elif (
            latch_enabled
            and bool(getattr(self, "_controller_latched", False))
            and drop_frac <= release_drop
        ):
            self._controller_latched = False

        if latch_enabled and bool(getattr(self, "_controller_latched", False)):
            active = bool(peak >= min_peak and drop_frac > release_drop)
            if active:
                latched_signal = float(getattr(
                    self.config, "bapr_controller_latched_signal", 1.0))
                signal = max(signal, float(np.clip(latched_signal, 0.0, 1.0)))

        active_iters = int(getattr(self, "_controller_active_iters", 0))
        cooldown_iters = int(getattr(self, "_controller_cooldown_iters", 0))
        exit_reason = 0.0
        if cooldown_iters > 0:
            cooldown_iters -= 1
            active = False
            signal = 0.0
            active_iters = 0
            exit_reason = 1.0
        if active:
            active_iters += 1
            max_active_iters = int(getattr(
                self.config, "bapr_controller_max_active_iters", 0))
            min_active_iters = int(getattr(
                self.config, "bapr_controller_min_active_iters", 0))
            min_active_ok = active_iters >= max(0, min_active_iters)
            cooldown_target = int(getattr(
                self.config, "bapr_controller_exit_cooldown_iters", 0))
            signal_exit_th = float(getattr(
                self.config, "bapr_controller_exit_signal_threshold", 0.0))
            improve_exit_th = float(getattr(
                self.config, "bapr_controller_exit_improve_frac", 0.0))
            drawdown_exit_th = float(getattr(
                self.config, "bapr_controller_exit_drawdown_frac", -1.0))
            reward_improve = max(
                float(getattr(self, "_last_train_reward_improve_frac", 0.0)),
                float(getattr(self, "_last_eval_reward_improve_frac", 0.0)))
            force_exit = False
            if max_active_iters > 0 and active_iters > max_active_iters:
                force_exit = True
                exit_reason = 2.0
            elif min_active_ok and signal_exit_th > 0.0 and signal <= signal_exit_th:
                force_exit = True
                exit_reason = 3.0
            elif min_active_ok and improve_exit_th > 0.0 and reward_improve >= improve_exit_th:
                force_exit = True
                exit_reason = 4.0
            elif min_active_ok and drawdown_exit_th >= 0.0 and drop_frac <= drawdown_exit_th:
                force_exit = True
                exit_reason = 5.0

            if force_exit:
                if cooldown_target > 0:
                    cooldown_iters = cooldown_target
                    self._controller_duration_released = False
                else:
                    self._controller_duration_released = True
                self._controller_latched = False
                active = False
                signal = 0.0
                active_iters = 0
        else:
            active_iters = 0

        self._controller_drawdown = float(drop_frac)
        self._controller_low_disagreement = low_disagreement
        self._controller_signal = float(signal)
        self._controller_active = active
        self._controller_active_iters = active_iters
        self._controller_cooldown_iters = cooldown_iters
        self._controller_exit_reason = float(exit_reason)
        if not active:
            self._controller_reg_multiplier = 1.0
            self._controller_recent_multiplier = 1.0
            self._controller_recent_add_frac = 0.0
            self._controller_lcb_multiplier = 1.0
            self._controller_actor_update_multiplier = 1.0
            return

        reg_target = float(np.clip(getattr(
            self.config, "bapr_controller_reg_multiplier", 0.2), 0.0, 1.0))
        recent_target = float(max(0.0, getattr(
            self.config, "bapr_controller_recent_multiplier", 1.0)))
        recent_add = float(max(0.0, getattr(
            self.config, "bapr_controller_recent_add_frac", 0.0)))
        lcb_target = float(np.clip(getattr(
            self.config, "bapr_controller_lcb_multiplier", 1.0), 0.0, 1.0))
        actor_update_target = float(np.clip(getattr(
            self.config, "bapr_controller_actor_update_multiplier", 1.0),
            0.0, 1.0))

        def recover_target(target, recover_iters):
            recover_iters = int(recover_iters)
            if recover_iters <= 0 or target >= 1.0:
                return float(target)
            progress = float(np.clip(
                (active_iters - 1) / max(float(recover_iters), 1.0),
                0.0, 1.0))
            return float(target + progress * (1.0 - target))

        reg_target = recover_target(
            reg_target,
            getattr(self.config, "bapr_controller_reg_recover_iters", 0))
        actor_update_target = recover_target(
            actor_update_target,
            getattr(self.config, "bapr_controller_actor_recover_iters", 0))

        self._controller_reg_multiplier = float(
            1.0 + signal * (reg_target - 1.0))
        self._controller_recent_multiplier = float(
            1.0 + signal * (recent_target - 1.0))
        self._controller_recent_add_frac = float(signal * recent_add)
        self._controller_lcb_multiplier = float(
            1.0 + signal * (lcb_target - 1.0))
        self._controller_actor_update_multiplier = float(
            1.0 + signal * (actor_update_target - 1.0))

    def _apply_controller_to_recent_frac(self, frac, cap):
        if not bool(getattr(self, "_controller_active", False)):
            return float(frac)
        value = (
            float(frac) * float(getattr(self, "_controller_recent_multiplier", 1.0))
            + float(getattr(self, "_controller_recent_add_frac", 0.0))
        )
        return float(min(float(cap), max(0.0, value)))

    def _apply_controller_to_reg_scale(self, scale):
        return float(scale) * float(getattr(
            self, "_controller_reg_multiplier", 1.0))

    def _apply_controller_to_actor_lcb_scale(self, scale):
        return float(scale) * float(getattr(
            self, "_controller_lcb_multiplier", 1.0))

    def _actor_update_scale(self):
        return float(getattr(
            self, "_controller_actor_update_multiplier", 1.0))

    def report_eval(self, eval_reward):
        self._last_eval_reward_drop_frac = self._update_reward_drop(
            eval_reward,
            "_eval_reward_ema",
            "_eval_reward_peak")

    def _performance_drop_active(self, current_iter):
        warmup_iters = int(getattr(
            self.config, "bapr_actor_lcb_perf_warmup_iters", 100))
        if current_iter < warmup_iters:
            self._last_perf_drop_frac = 0.0
            self._actor_lcb_perf_active = False
            return False
        min_peak = float(getattr(
            self.config, "bapr_actor_lcb_perf_min_peak", 100.0))
        train_peak = getattr(self, "_train_reward_peak", None)
        eval_peak = getattr(self, "_eval_reward_peak", None)
        peak = max(float(train_peak or 0.0), float(eval_peak or 0.0))
        drop_frac = max(
            float(getattr(self, "_last_train_reward_drop_frac", 0.0)),
            float(getattr(self, "_last_eval_reward_drop_frac", 0.0)))
        threshold = float(getattr(
            self.config, "bapr_actor_lcb_perf_drop_frac", 0.25))
        active = peak >= min_peak and drop_frac >= threshold
        self._last_perf_drop_frac = float(drop_frac)
        self._actor_lcb_perf_active = bool(active)
        return bool(active)

    def _adjust_reg_scale_for_perf_collapse(self, scale, current_iter):
        if not getattr(self.config, "bapr_reg_perf_collapse_gate", False):
            self._reg_perf_collapse_active = False
            return float(scale)
        warmup_iters = int(getattr(
            self.config, "bapr_reg_perf_collapse_warmup_iters", 100))
        if current_iter < warmup_iters:
            self._reg_perf_collapse_active = False
            return float(scale)
        drop_frac = max(
            float(getattr(self, "_last_train_reward_drop_frac", 0.0)),
            float(getattr(self, "_last_eval_reward_drop_frac", 0.0)))
        threshold = float(getattr(
            self.config, "bapr_reg_perf_collapse_drop_frac", 1.0))
        active = drop_frac >= threshold and float(scale) > 0.0
        self._reg_perf_collapse_active = bool(active)
        if not active:
            return float(scale)
        collapse_scale = float(getattr(
            self.config, "bapr_reg_perf_collapse_scale", 0.0))
        return float(scale) * float(np.clip(collapse_scale, 0.0, 1.0))

    def _compute_reg_scale(self, q_std, q_abs_mean, current_iter):
        if not getattr(self.config, "bapr_reg_disagreement_gate", False):
            scale = self._adjust_reg_scale_for_perf_collapse(1.0, current_iter)
            return self._apply_controller_to_reg_scale(scale)

        warmup_iters = int(getattr(self.config, "bapr_reg_warmup_iters", 0))
        if current_iter < warmup_iters:
            self._reg_perf_collapse_active = False
            return 0.0

        q_std = float(q_std)
        q_abs_mean = max(float(q_abs_mean), 1e-6)
        ratio = q_std / q_abs_mean
        qstd_high = q_std > float(getattr(
            self.config, "bapr_reg_qstd_threshold", 5.0))
        ratio_high = ratio > float(getattr(
            self.config, "bapr_reg_qstd_ratio_threshold", 0.02))
        high_disagreement = (qstd_high and ratio_high) if getattr(
            self.config, "bapr_reg_require_both", False) else (
            qstd_high or ratio_high)
        emergency_disagreement = False
        if getattr(self.config, "bapr_reg_emergency_gate", False):
            emergency_disagreement = (
                q_std > float(getattr(
                    self.config, "bapr_reg_emergency_qstd_threshold", 20.0))
                and ratio > float(getattr(
                    self.config,
                    "bapr_reg_emergency_qstd_ratio_threshold", 0.05)))

        max_iters = int(getattr(self.config, "bapr_reg_max_iters", 0))
        if getattr(self.config, "bapr_reg_latch", False):
            if (max_iters <= 0 or current_iter < max_iters) and high_disagreement:
                self._reg_latched = True
            if self._reg_latched:
                scale = float(getattr(
                    self.config, "bapr_reg_latched_scale", 1.0))
                scale = self._adjust_reg_scale_for_perf_collapse(
                    scale, current_iter)
                return self._apply_controller_to_reg_scale(scale)
            if emergency_disagreement:
                scale = float(getattr(
                    self.config, "bapr_reg_emergency_scale", 1.0))
                scale = self._adjust_reg_scale_for_perf_collapse(
                    scale, current_iter)
                return self._apply_controller_to_reg_scale(scale)
            self._reg_perf_collapse_active = False
            return 0.0

        if max_iters > 0 and current_iter >= max_iters:
            if emergency_disagreement:
                scale = float(getattr(
                    self.config, "bapr_reg_emergency_scale", 1.0))
                scale = self._adjust_reg_scale_for_perf_collapse(
                    scale, current_iter)
                return self._apply_controller_to_reg_scale(scale)
            self._reg_perf_collapse_active = False
            return 0.0

        if getattr(self.config, "bapr_reg_require_both", False):
            if qstd_high and ratio_high:
                scale = self._adjust_reg_scale_for_perf_collapse(
                    1.0, current_iter)
                return self._apply_controller_to_reg_scale(scale)
            self._reg_perf_collapse_active = False
            return 0.0
        if qstd_high or ratio_high:
            scale = self._adjust_reg_scale_for_perf_collapse(
                1.0, current_iter)
            return self._apply_controller_to_reg_scale(scale)
        self._reg_perf_collapse_active = False
        return 0.0

    def _compute_actor_lcb_scale(self, q_std, q_abs_mean, current_iter):
        if not getattr(self.config, "bapr_actor_lcb_qstd_gate", False):
            self._last_perf_drop_frac = max(
                float(getattr(self, "_last_train_reward_drop_frac", 0.0)),
                float(getattr(self, "_last_eval_reward_drop_frac", 0.0)))
            self._actor_lcb_perf_active = False
            self._actor_lcb_qstd_active = False
            self._actor_lcb_active = False
            return 0.0

        perf_ok = True
        if getattr(self.config, "bapr_actor_lcb_perf_gate", False):
            perf_ok = self._performance_drop_active(current_iter)
        else:
            self._last_perf_drop_frac = max(
                float(getattr(self, "_last_train_reward_drop_frac", 0.0)),
                float(getattr(self, "_last_eval_reward_drop_frac", 0.0)))
            self._actor_lcb_perf_active = False

        q_std = float(q_std)
        q_abs_mean = max(float(q_abs_mean), 1e-6)
        ratio = q_std / q_abs_mean
        qstd_high = q_std > float(getattr(
            self.config, "bapr_actor_lcb_qstd_threshold", 270.0))
        ratio_high = ratio > float(getattr(
            self.config, "bapr_actor_lcb_qstd_ratio_threshold", 0.04))
        if getattr(self.config, "bapr_actor_lcb_require_both", True):
            active = qstd_high and ratio_high
        else:
            active = qstd_high or ratio_high
        self._actor_lcb_qstd_active = bool(active)
        if not active:
            self._actor_lcb_active = False
            return 0.0
        scale = float(getattr(self.config, "bapr_actor_lcb_scale", 1.0))
        if not getattr(self.config, "bapr_actor_lcb_perf_gate", False):
            value = self._apply_controller_to_actor_lcb_scale(scale)
            self._actor_lcb_active = value > 0.0
            return value

        if not perf_ok:
            self._actor_lcb_active = False
            return 0.0
        threshold = max(float(getattr(
            self.config, "bapr_actor_lcb_perf_drop_frac", 0.25)), 1e-6)
        ramp = float(np.clip(self._last_perf_drop_frac / threshold, 0.0, 1.0))
        value = self._apply_controller_to_actor_lcb_scale(scale * ramp)
        self._actor_lcb_active = value > 0.0
        return value

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
            q_std, q_abs_mean = self._compute_q_stats(
                nnx.state(self.critic, nnx.Param),
                nnx.state(self.context_net, nnx.Param),
                robs, ract, warmup, belief_jax)
            reward_signal = rrew
        else:
            # Fallback: replay batch (random exploration phase or compatibility)
            last_obs = jnp.asarray(stacked_batch["obs"][-1])
            last_act = jnp.asarray(stacked_batch["act"][-1])
            q_std, q_abs_mean = self._compute_q_stats(
                nnx.state(self.critic, nnx.Param),
                nnx.state(self.context_net, nnx.Param),
                last_obs, last_act, warmup, belief_jax)
            if recent_rewards is not None and len(recent_rewards) > 0:
                reward_signal = np.array(recent_rewards, dtype=np.float32)
            else:
                reward_signal = np.asarray(stacked_batch["rew"][-1]).reshape(-1)

        surprise = self.surprise_computer.compute(
            reward_signal, np.asarray([float(q_std)], dtype=np.float32))
        self._last_surprise = float(surprise)
        self.belief_tracker.update(surprise)

        # ── Minimum redesign: feed chunked observations to regime tracker.
        # Uses the PRE-update belief_jax (Q-net was conditioned on this for
        # the chunked Q-std / TD computations); fine because the new tracker
        # state is built from this iter's data and read back below.
        if self.use_regime_belief and recent_rollout is not None:
            self._observe_chunked_rollout(recent_rollout, warmup, belief_jax)

        if self.adaptation_mode == "gate":
            # Gate-mode BAPR keeps the stable mean-actor path and only uses
            # this scalar for recent replay / optional gated_lcb. Do not
            # reuse the long legacy BOCD warmup: in the failed v1 runs, critic
            # disagreement exploded before the gate was even allowed to open.
            gate_warmup = int(getattr(self.config, "bapr_gate_warmup_iters", 0))
            if current_iter < gate_warmup:
                weighted_lambda = 0.0
                self._adaptation_gate = 0.0
                self._raw_adaptation_gate = 0.0
            else:
                weighted_lambda = self._update_adaptation_gate(surprise)
        elif current_iter < self.config.bapr_warmup_iters:
            weighted_lambda = 0.0
        else:
            weighted_lambda = self._compute_weighted_lambda()
        self._current_weighted_lambda = weighted_lambda
        self._last_q_std_mean = float(q_std)
        self._last_q_abs_mean = float(q_abs_mean)
        self._observe_train_reward(recent_rewards)
        self._update_algorithm_controller(q_std, q_abs_mean, current_iter)
        self._current_recent_replay_frac = self._compute_recent_replay_frac(
            weighted_lambda, q_std, q_abs_mean)
        self._current_reg_scale = self._compute_reg_scale(
            q_std, q_abs_mean, current_iter)
        self._current_actor_lcb_scale = self._compute_actor_lcb_scale(
            q_std, q_abs_mean, current_iter)

        # Refresh belief vector AFTER update so scan uses post-update belief.
        # Format depends on flags: see _build_belief_jax docstring.
        belief_jax = self._build_belief_jax()

        # Run scan
        c_p = nnx.state(self.critic, nnx.Param)
        t_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)
        ema_p = nnx.state(self.ema_policy, nnx.Param)
        x_p = nnx.state(self.context_net, nnx.Param)

        obs = jnp.array(stacked_batch["obs"])
        act = jnp.array(stacked_batch["act"])
        rew = jnp.array(stacked_batch["rew"])
        nobs = jnp.array(stacked_batch["next_obs"])
        done = jnp.array(stacked_batch["done"])
        tids = jnp.array(stacked_batch["task_id"])
        # GPT-5.5 advice #2: per-transition belief from replay (toggle).
        use_per_trans = getattr(self.config, "use_per_transition_belief", True)
        if use_per_trans and "belief" in stacked_batch:
            beliefs = jnp.array(stacked_batch["belief"])
        else:
            # Fallback / legacy: broadcast current belief_jax to scan shape
            n_iters, batch_size = obs.shape[0], obs.shape[1]
            beliefs = jnp.broadcast_to(
                belief_jax[None, None, :],
                (n_iters, batch_size, self.belief_dim)) if self.belief_dim > 0 \
                else jnp.zeros((n_iters, batch_size, 0), dtype=jnp.float32)

        final, metrics = self._scan_update(
            c_p, t_p, p_p, ema_p, x_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state,
            self.context_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done, tids, rng_key, warmup,
            jnp.array(weighted_lambda), belief_jax, beliefs,
            jnp.array(self._current_reg_scale, dtype=jnp.float32),
            jnp.array(self._current_actor_lcb_scale, dtype=jnp.float32),
            jnp.array(self._actor_update_scale(), dtype=jnp.float32))

        (new_c, new_t, new_p, new_ema_p, new_x, new_la,
         self.critic_opt_state, self.policy_opt_state,
         self.context_opt_state, self.alpha_opt_state, _) = final
        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_t)
        nnx.update(self.policy, new_p)
        nnx.update(self.ema_policy, new_ema_p)
        nnx.update(self.context_net, new_x)
        self.log_alpha = new_la

        n = obs.shape[0]
        self.update_count += n

        (c_loss, p_loss, x_loss, alpha, qm, qs, lp,
         residual_gate, residual_adv, residual_norm) = metrics
        effective_beta = self.config.beta - weighted_lambda * self.config.penalty_scale
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "rmdm_loss": float(x_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(qm.mean()),
            "q_std_mean": float(qs.mean()),
            "log_prob": float(lp.mean()),
            "v80_residual_gate": float(residual_gate.mean()),
            "v80_residual_advantage": float(residual_adv.mean()),
            "v80_residual_norm": float(residual_norm.mean()),
            "weighted_lambda": weighted_lambda,
            "adaptation_gate": float(self._adaptation_gate),
            "raw_adaptation_gate": float(self._raw_adaptation_gate),
            "surprise": float(self._last_surprise),
            "q_std_ratio": float(self._last_q_std_ratio),
            "planned_recent_replay_frac": float(self._current_recent_replay_frac),
            "reg_scale": float(self._current_reg_scale),
            "reg_perf_collapse_active": float(self._reg_perf_collapse_active),
            "actor_lcb_scale": float(self._current_actor_lcb_scale),
            "actor_lcb_perf_active": float(self._actor_lcb_perf_active),
            "actor_lcb_qstd_active": float(self._actor_lcb_qstd_active),
            "actor_lcb_active": float(self._actor_lcb_active),
            "controller_active": float(self._controller_active),
            "controller_latched": float(self._controller_latched),
            "controller_active_iters": float(self._controller_active_iters),
            "controller_cooldown_iters": float(self._controller_cooldown_iters),
            "controller_duration_released": float(
                self._controller_duration_released),
            "controller_exit_reason": float(self._controller_exit_reason),
            "controller_signal": float(self._controller_signal),
            "controller_drawdown": float(self._controller_drawdown),
            "controller_low_disagreement": float(self._controller_low_disagreement),
            "controller_reg_multiplier": float(self._controller_reg_multiplier),
            "controller_recent_multiplier": float(self._controller_recent_multiplier),
            "controller_recent_add_frac": float(self._controller_recent_add_frac),
            "controller_lcb_multiplier": float(self._controller_lcb_multiplier),
            "controller_actor_update_multiplier": float(
                self._controller_actor_update_multiplier),
            "perf_drop_frac": float(self._last_perf_drop_frac),
            "train_reward_drop_frac": float(self._last_train_reward_drop_frac),
            "eval_reward_drop_frac": float(self._last_eval_reward_drop_frac),
            "train_reward_improve_frac": float(
                self._last_train_reward_improve_frac),
            "eval_reward_improve_frac": float(
                self._last_eval_reward_improve_frac),
            "reg_latched": float(self._reg_latched),
            "effective_beta": float(effective_beta),
            "belief_entropy": float(self.belief_tracker.entropy),
            "effective_window": float(self.belief_tracker.effective_window),
            "warmup": bool(warmup),
        }
