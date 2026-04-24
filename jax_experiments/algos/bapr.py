"""BAPR: ESCP + BOCD Belief Tracker + Adaptive β — scan-fused."""
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


class BAPR:
    """BAPR = ESCP + BOCD belief-weighted adaptive conservatism — scan-fused."""

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

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         ctx_params, log_alpha,
                         c_opt_state, p_opt_state, x_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         all_task_ids, rng_key, warmup, weighted_lambda):
            """BAPR scan with adaptive beta via weighted_lambda."""
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

                obs_aug = jnp.concatenate([obs, ep], axis=-1)
                next_aug = jnp.concatenate([next_obs, next_ep], axis=-1)

                # === Critic ===
                def critic_loss_fn(cp):
                    tm = nnx.merge(gd_target, t_p)
                    pm = nnx.merge(gd_policy, p_p)
                    na, nlp = pm.sample(next_obs, k1, next_ep)
                    tq = tm(next_aug, na).min(axis=0) - alpha * nlp
                    tv = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq
                    cm = nnx.merge(gd_critic, cp)
                    pq = cm(obs_aug, act)
                    return jnp.mean((pq - tv[None]) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy (adaptive β) ===
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

        # JIT'd Q-std computation for surprise (called once per multi_update)
        @jax.jit
        def _compute_q_stats(critic_params, ctx_params, obs, act, warmup):
            cm = nnx.merge(gd_critic, critic_params)
            xm = nnx.merge(gd_ctx, ctx_params)
            ep = xm(obs)
            ep = jnp.where(warmup, jnp.zeros_like(ep), ep)
            oa = jnp.concatenate([obs, ep], axis=-1)
            pq = cm(oa, act)
            return pq.std(axis=0).mean()

        self._scan_update = _scan_update
        self._compute_q_stats = _compute_q_stats

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

    def reset_episode(self):
        self.belief_tracker.reset()
        self.surprise_computer.reset()

    def _compute_weighted_lambda(self):
        """Compute belief-weighted penalty.

        BOCD dynamics: high surprise → belief pushed to high h (high variance
        explains outliers) → effective_window increases.
        So: λ_w = ew / H → high when environment just changed.
        """
        ew = self.belief_tracker.effective_window
        max_h = self.belief_tracker.max_H - 1
        return float(np.clip(ew / max_h, 0.0, 1.0))

    def multi_update(self, stacked_batch, current_iter=0, recent_rewards=None, **kwargs):
        rng_key = self.rngs.params()
        warmup = jnp.array(current_iter < self.config.context_warmup_iters)

        # Q-std from critic (always from replay batch — evaluates uncertainty)
        last_obs = jnp.array(stacked_batch["obs"][-1])
        last_act = jnp.array(stacked_batch["act"][-1])
        q_std = self._compute_q_stats(
            nnx.state(self.critic, nnx.Param),
            nnx.state(self.context_net, nnx.Param),
            last_obs, last_act, warmup)

        # Reward signal: prefer actual rollout episode rewards (direct task signal)
        # over random replay batch rewards (may be from old tasks)
        if recent_rewards is not None and len(recent_rewards) > 0:
            reward_signal = np.array(recent_rewards, dtype=np.float32)
        else:
            reward_signal = stacked_batch["rew"][-1].flatten()

        surprise = self.surprise_computer.compute(reward_signal, np.array([float(q_std)]))
        self.belief_tracker.update(surprise)
        weighted_lambda = self._compute_weighted_lambda()
        self._current_weighted_lambda = weighted_lambda

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
            jnp.array(weighted_lambda))

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
