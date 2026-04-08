"""BAPR ablation variants for isolating component contributions.

Four ablation classes, each removing one component:
  - BAPRNoBocd:       BOCD removed, beta_eff = beta_base (static), context active
  - BAPRNoRmdm:       RMDM removed, no context embedding, BOCD active
  - BAPRNoAdaptBeta:  BOCD runs but lambda_w NOT fed into LCB (diagnostic only)
  - BAPRFixedDecay:   Heuristic exp-decay penalty replaces BOCD
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


# ---------------------------------------------------------------------------
#  Helper: build common ESCP-like scan (shared by NoBocd, NoAdaptBeta, FixedDecay)
# ---------------------------------------------------------------------------
def _build_escp_scan_with_beta(agent, effective_beta_val):
    """Build ESCP-style scan body with a given (possibly adaptive) beta value.
    Returns JIT-compiled _scan_update function.
    """
    gamma = agent.config.gamma
    tau = agent.config.tau
    auto_alpha = agent.config.auto_alpha
    target_entropy = jnp.array(agent.target_entropy)
    rbf_r = agent.config.rbf_radius
    cons_w = agent.config.consistency_loss_weight
    div_w = agent.config.diversity_loss_weight

    gd_policy = nnx.graphdef(agent.policy)
    gd_critic = nnx.graphdef(agent.critic)
    gd_target = nnx.graphdef(agent.target_critic)
    gd_ctx = nnx.graphdef(agent.context_net)

    p_opt = agent.policy_opt
    c_opt = agent.critic_opt
    x_opt = agent.context_opt
    a_opt = agent.alpha_opt

    @jax.jit
    def _scan_update(critic_params, target_params, policy_params,
                     ctx_params, log_alpha,
                     c_opt_state, p_opt_state, x_opt_state, a_opt_state,
                     all_obs, all_act, all_rew, all_next_obs, all_done,
                     all_task_ids, rng_key, warmup, effective_beta):

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

            obs_aug = jnp.concatenate([obs, ep], axis=-1)
            next_aug = jnp.concatenate([next_obs, next_ep], axis=-1)

            # === Critic ===
            def critic_loss_fn(cp):
                tm = nnx.merge(gd_target, t_p)
                pm = nnx.merge(gd_policy, p_p)
                na, nlp = pm.sample(next_obs, k1, next_ep)
                # Independent targets: each Q_i uses its own target Q_i
                tq_all = tm(next_aug, na) - alpha * nlp  # [K, batch]
                tv_all = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq_all
                cm = nnx.merge(gd_critic, cp)
                pq = cm(obs_aug, act)
                return jnp.mean((pq - tv_all) ** 2), pq

            (c_loss, pq), c_grads = jax.value_and_grad(
                critic_loss_fn, has_aux=True)(c_p)
            c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
            new_c_p = optax.apply_updates(c_p, c_upd)

            # === Policy (with effective_beta) ===
            def policy_loss_fn(pp):
                pm = nnx.merge(gd_policy, pp)
                cm = nnx.merge(gd_critic, new_c_p)
                na, lp = pm.sample(obs, k2, ep)
                oa = jnp.concatenate([obs, ep], axis=-1)
                qv = cm(oa, na)
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

    return _scan_update


# ---------------------------------------------------------------------------
#  Common base for ESCP-like ablation agents (context + ensemble)
# ---------------------------------------------------------------------------
class _BAPRAblationBase:
    """Shared init for all BAPR ablation variants that keep the context network."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)

        self.context_net = ContextNetwork(
            obs_dim, config.ep_dim, hidden_dim=128, rngs=self.rngs)
        self.policy = GaussianPolicy(
            obs_dim, act_dim, config.hidden_dim,
            ep_dim=config.ep_dim, n_layers=2, rngs=self.rngs)
        self.critic = EnsembleCritic(
            obs_dim + config.ep_dim, act_dim, config.hidden_dim,
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

        self.update_count = 0

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    def select_action(self, obs, deterministic=False):
        obs_jax = jnp.array(obs)[None] if np.asarray(obs).ndim == 1 else jnp.array(obs)
        ep = self.context_net(obs_jax)
        if deterministic:
            return np.array(self.policy.deterministic(obs_jax, ep)[0])
        key = self.rngs.params()
        action, _ = self.policy.sample(obs_jax, key, ep)
        return np.array(action[0])

    def _run_scan_and_unpack(self, stacked_batch, rng_key, warmup, effective_beta):
        c_p = nnx.state(self.critic, nnx.Param)
        t_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)
        x_p = nnx.state(self.context_net, nnx.Param)

        # Data is already JAX arrays from GPU-native replay buffer
        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        nobs = stacked_batch["next_obs"]
        done = stacked_batch["done"]
        tids = stacked_batch["task_id"]

        final, metrics = self._scan_update(
            c_p, t_p, p_p, x_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state,
            self.context_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done, tids, rng_key, warmup,
            jnp.array(effective_beta))

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
            "warmup": bool(warmup),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  1. BAPR w/o BOCD  —  beta_eff = beta_base (static), context still active
# ═══════════════════════════════════════════════════════════════════════════
class BAPRNoBocd(_BAPRAblationBase):
    """BAPR minus BOCD: static beta, context conditioning active."""

    def __init__(self, obs_dim, act_dim, config, seed=0):
        super().__init__(obs_dim, act_dim, config, seed)
        self._scan_update = _build_escp_scan_with_beta(self, None)

    def multi_update(self, stacked_batch, current_iter=0, **kwargs):
        rng_key = self.rngs.params()
        warmup = jnp.array(current_iter < self.config.context_warmup_iters)
        result = self._run_scan_and_unpack(
            stacked_batch, rng_key, warmup,
            effective_beta=self.config.beta)  # static beta, no adaptation
        result["weighted_lambda"] = 0.0
        result["belief_entropy"] = 0.0
        result["effective_window"] = 0.0
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  2. BAPR w/o RMDM  —  no context embedding, BOCD + adaptive beta active
# ═══════════════════════════════════════════════════════════════════════════
class BAPRNoRmdm:
    """BAPR minus RMDM: no context embedding, BOCD active, adaptive beta."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)

        # No context network — policy and critic use raw obs
        self.policy = GaussianPolicy(
            obs_dim, act_dim, config.hidden_dim,
            ep_dim=0, n_layers=2, rngs=self.rngs)
        self.critic = EnsembleCritic(
            obs_dim, act_dim, config.hidden_dim,
            ensemble_size=config.ensemble_size, n_layers=3, rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

        self.log_alpha = jnp.array(jnp.log(config.alpha))
        self.target_entropy = -float(act_dim)

        self.policy_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)
        self.alpha_opt = optax.adam(config.lr)

        self.policy_opt_state = self.policy_opt.init(nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(nnx.state(self.critic, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)

        self.belief_tracker = BeliefTracker(
            max_run_length=config.max_run_length,
            hazard_rate=config.hazard_rate,
            base_variance=config.base_variance,
            variance_growth=config.variance_growth)
        self.surprise_computer = SurpriseComputer(ema_alpha=config.surprise_ema_alpha)

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

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)

        p_opt = self.policy_opt
        c_opt = self.critic_opt
        a_opt = self.alpha_opt

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         log_alpha, c_opt_state, p_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         rng_key, weighted_lambda):
            effective_beta = beta_base - weighted_lambda * penalty_scale

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key) = carry
                (obs, act, rew, next_obs, done) = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                def critic_loss_fn(cp):
                    tm = nnx.merge(gd_target, t_p)
                    pm = nnx.merge(gd_policy, p_p)
                    na, nlp = pm.sample(next_obs, k1)
                    # Independent targets: each Q_i uses its own target Q_i
                    tq_all = tm(next_obs, na) - alpha * nlp  # [K, batch]
                    tv_all = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq_all
                    cm = nnx.merge(gd_critic, cp)
                    pq = cm(obs, act)
                    return jnp.mean((pq - tv_all) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    na, lp = pm.sample(obs, k2)
                    qv = cm(obs, na)
                    lcb = qv.mean(axis=0) + effective_beta * qv.std(axis=0)
                    return (jnp.exp(la) * lp - lcb).mean(), lp

                (p_loss, lp), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                p_upd, new_p_os = p_opt.update(p_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, p_upd)

                a_grad = -(lp.mean() + target_entropy)
                a_upd, new_a_os = a_opt.update(a_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + a_upd, la)
                new_a_os = jax.tree.map(
                    lambda n, o: jnp.where(auto_alpha, n, o), new_a_os, a_os)

                new_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1 - tau) + cp * tau, t_p, new_c_p)

                new_carry = (new_c_p, new_t_p, new_p_p, new_la,
                             new_c_os, new_p_os, new_a_os, key)
                metrics = (c_loss, p_loss, jnp.exp(new_la),
                           pq.mean(), pq.std(axis=0).mean(), lp.mean())
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    log_alpha, c_opt_state, p_opt_state, a_opt_state, rng_key)
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done)
            return jax.lax.scan(body_fn, init, batches)

        @jax.jit
        def _compute_q_stats(critic_params, obs, act):
            cm = nnx.merge(gd_critic, critic_params)
            pq = cm(obs, act)
            return pq.std(axis=0).mean()

        self._scan_update = _scan_update
        self._compute_q_stats_normdm = _compute_q_stats

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    def select_action(self, obs, deterministic=False):
        obs_jax = jnp.array(obs)[None] if np.asarray(obs).ndim == 1 else jnp.array(obs)
        if deterministic:
            return np.array(self.policy.deterministic(obs_jax)[0])
        key = self.rngs.params()
        action, _ = self.policy.sample(obs_jax, key)
        return np.array(action[0])

    def reset_episode(self):
        self.belief_tracker.reset()
        self.surprise_computer.reset()

    def _compute_weighted_lambda(self):
        ew = self.belief_tracker.effective_window
        max_h = self.belief_tracker.max_H - 1
        raw_lw = float(np.clip(ew / max_h, 0.0, 1.0))
        if not hasattr(self, '_lw_baseline'):
            self._lw_baseline = raw_lw
        else:
            self._lw_baseline = 0.95 * self._lw_baseline + 0.05 * raw_lw
        return float(np.clip(raw_lw - self._lw_baseline, 0.0, 1.0))

    def multi_update(self, stacked_batch, current_iter=0, recent_rewards=None, **kwargs):
        rng_key = self.rngs.params()

        # Data is already JAX arrays from GPU-native replay buffer
        last_obs = stacked_batch["obs"][-1]
        last_act = stacked_batch["act"][-1]
        q_std = self._compute_q_stats_normdm(
            nnx.state(self.critic, nnx.Param), last_obs, last_act)

        if recent_rewards is not None and len(recent_rewards) > 0:
            reward_signal = np.array(recent_rewards, dtype=np.float32)
        else:
            reward_signal = stacked_batch["rew"][-1].flatten()

        surprise = self.surprise_computer.compute(reward_signal, np.array([float(q_std)]))
        self.belief_tracker.update(surprise)
        weighted_lambda = self._compute_weighted_lambda()
        self._current_weighted_lambda = weighted_lambda

        c_p = nnx.state(self.critic, nnx.Param)
        t_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)

        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        nobs = stacked_batch["next_obs"]
        done = stacked_batch["done"]

        final, metrics = self._scan_update(
            c_p, t_p, p_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done, rng_key,
            jnp.array(weighted_lambda))

        (new_c, new_t, new_p, new_la,
         self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
         _) = final
        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_t)
        nnx.update(self.policy, new_p)
        self.log_alpha = new_la

        n = obs.shape[0]
        self.update_count += n

        c_loss, p_loss, alpha, qm, qs, lp = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(qm.mean()),
            "q_std_mean": float(qs.mean()),
            "log_prob": float(lp.mean()),
            "weighted_lambda": weighted_lambda,
            "belief_entropy": float(self.belief_tracker.entropy),
            "effective_window": float(self.belief_tracker.effective_window),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  3. BAPR w/o adaptive beta  —  BOCD runs (logged) but does NOT affect LCB
# ═══════════════════════════════════════════════════════════════════════════
class BAPRNoAdaptBeta(_BAPRAblationBase):
    """BAPR minus adaptive beta: BOCD runs for diagnostics but beta stays static."""

    def __init__(self, obs_dim, act_dim, config, seed=0):
        super().__init__(obs_dim, act_dim, config, seed)
        self.belief_tracker = BeliefTracker(
            max_run_length=config.max_run_length,
            hazard_rate=config.hazard_rate,
            base_variance=config.base_variance,
            variance_growth=config.variance_growth)
        self.surprise_computer = SurpriseComputer(ema_alpha=config.surprise_ema_alpha)
        self._current_weighted_lambda = 0.0
        self._scan_update = _build_escp_scan_with_beta(self, None)

    def reset_episode(self):
        self.belief_tracker.reset()
        self.surprise_computer.reset()

    def multi_update(self, stacked_batch, current_iter=0, recent_rewards=None, **kwargs):
        rng_key = self.rngs.params()
        warmup = jnp.array(current_iter < self.config.context_warmup_iters)

        # Run BOCD for diagnostic logging only
        if recent_rewards is not None and len(recent_rewards) > 0:
            reward_signal = np.array(recent_rewards, dtype=np.float32)
        else:
            reward_signal = stacked_batch["rew"][-1].flatten()
        surprise = self.surprise_computer.compute(reward_signal, np.array([0.0]))
        self.belief_tracker.update(surprise)

        # Static beta — BOCD does NOT affect policy
        result = self._run_scan_and_unpack(
            stacked_batch, rng_key, warmup,
            effective_beta=self.config.beta)

        result["weighted_lambda"] = 0.0  # not used
        result["belief_entropy"] = float(self.belief_tracker.entropy)
        result["effective_window"] = float(self.belief_tracker.effective_window)
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  4. BAPR-FixedDecay  —  heuristic exp decay replaces BOCD
# ═══════════════════════════════════════════════════════════════════════════
class BAPRFixedDecay(_BAPRAblationBase):
    """BAPR with fixed exponential decay penalty instead of BOCD.

    On each iteration:
      - If reward drops > threshold from EMA: trigger penalty spike to 1.0
      - Otherwise: penalty *= decay_rate
    """

    def __init__(self, obs_dim, act_dim, config, seed=0):
        super().__init__(obs_dim, act_dim, config, seed)
        self._scan_update = _build_escp_scan_with_beta(self, None)
        # Fixed-decay state
        self._decay_lambda = 0.0
        self._decay_rate = 0.99
        self._trigger_threshold = 1.5   # reward z-score to trigger
        self._ema_reward = None
        self._ema_reward_var = None
        self._ema_alpha = 0.3
        self._current_weighted_lambda = 0.0

    def reset_episode(self):
        self._decay_lambda = 0.0
        self._ema_reward = None
        self._ema_reward_var = None

    def multi_update(self, stacked_batch, current_iter=0, recent_rewards=None, **kwargs):
        rng_key = self.rngs.params()
        warmup = jnp.array(current_iter < self.config.context_warmup_iters)

        # Reward signal
        if recent_rewards is not None and len(recent_rewards) > 0:
            r_mean = float(np.mean(recent_rewards))
        else:
            r_mean = float(stacked_batch["rew"][-1].mean())

        # EMA tracking
        if self._ema_reward is None:
            self._ema_reward = r_mean
            self._ema_reward_var = 1.0
        else:
            old_ema = self._ema_reward
            self._ema_reward = self._ema_alpha * r_mean + (1 - self._ema_alpha) * self._ema_reward
            self._ema_reward_var = (self._ema_alpha * (r_mean - old_ema) ** 2
                                    + (1 - self._ema_alpha) * self._ema_reward_var)

        reward_std = max(np.sqrt(self._ema_reward_var), 1e-6)
        reward_z = abs(r_mean - self._ema_reward) / reward_std

        # Trigger or decay
        if reward_z > self._trigger_threshold:
            self._decay_lambda = 1.0
        else:
            self._decay_lambda *= self._decay_rate

        weighted_lambda = self._decay_lambda
        self._current_weighted_lambda = weighted_lambda

        effective_beta = self.config.beta - weighted_lambda * self.config.penalty_scale

        result = self._run_scan_and_unpack(
            stacked_batch, rng_key, warmup,
            effective_beta=effective_beta)

        result["weighted_lambda"] = weighted_lambda
        result["belief_entropy"] = 0.0
        result["effective_window"] = 0.0
        return result
