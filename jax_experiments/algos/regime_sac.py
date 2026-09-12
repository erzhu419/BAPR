"""Equal-budget SAC controllers for regime-control headroom tests.

Both arms use the same actor and critic architecture. The robust arm receives
an all-zero mode vector, while the privileged oracle arm receives the true
one-hot persistent regime. There is no encoder, residual branch, gate, BOCD,
or RE-SAC regularizer in this diagnostic.
"""
from __future__ import annotations

from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.networks.ensemble_critic import EnsembleCritic
from jax_experiments.networks.policy import GaussianPolicy


class RegimeSAC:
    """SAC conditioned directly on either zero or privileged mode context."""

    uses_regime_context = True
    EVAL_CONTEXT_KINDS = (
        "checkpoint", "true", "zero", "fixed", "cyclic", "shuffled",
        "delayed")

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        if getattr(config, "env_type", None) != "stochastic_mode":
            raise ValueError(
                "regime_sac is restricted to env_type=stochastic_mode")
        source = str(getattr(config, "regime_context_source", "robust"))
        if source not in ("robust", "oracle"):
            raise ValueError(
                "regime_context_source must be 'robust' or 'oracle'")
        mode_count = int(config.task_num)
        if mode_count <= 1:
            raise ValueError("regime_sac requires at least two modes")

        self.config = config
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.context_dim = mode_count
        self.belief_dim = mode_count
        self.context_mode = source
        self.rngs = nnx.Rngs(seed)
        self._current_mode_id = 0
        self._task_metadata_ready = False
        self._eval_context_kind = "checkpoint"
        self._eval_fixed_mode_id = None
        self._eval_shuffled_mode_map = None
        self._eval_delay_steps = 0
        self.reset_eval_context_state()

        # Both experiment arms instantiate this exact architecture. The robust
        # arm keeps the context coordinates at zero, preserving parameter count
        # and optimizer/update budget without exposing hidden mode information.
        self.policy = GaussianPolicy(
            self.obs_dim, self.act_dim, config.hidden_dim,
            ep_dim=self.context_dim, n_layers=2, rngs=self.rngs)
        self.critic = EnsembleCritic(
            self.obs_dim + self.context_dim, self.act_dim,
            config.hidden_dim, ensemble_size=config.ensemble_size,
            n_layers=3, rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

        self.log_alpha = jnp.asarray(jnp.log(config.alpha))
        self.target_entropy = -float(self.act_dim)
        self.policy_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)
        self.alpha_opt = optax.adam(config.lr)
        self.policy_opt_state = self.policy_opt.init(
            nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(
            nnx.state(self.critic, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)
        self.update_count = 0
        self._build_scan_fn()

    def set_task_metadata(self, tasks) -> None:
        mode_ids = [int(task["mode_id"]) for task in tasks]
        expected = list(range(self.context_dim))
        if sorted(mode_ids) != expected:
            raise ValueError(
                "regime_sac requires exactly one task for every mode id; "
                f"got {mode_ids}, expected {expected}")
        self._task_metadata_ready = True

    def context_for_task_id(self, mode_id: int):
        if not self._task_metadata_ready:
            raise RuntimeError("set_task_metadata must run before rollout")
        mode_id = int(mode_id)
        if not 0 <= mode_id < self.context_dim:
            raise ValueError(
                f"mode_id={mode_id} outside [0,{self.context_dim - 1}]")
        if self.context_mode == "robust":
            return jnp.zeros((self.context_dim,), dtype=jnp.float32)
        return jax.nn.one_hot(
            mode_id, self.context_dim, dtype=jnp.float32)

    def set_eval_context_override(
            self, kind: str = "checkpoint",
            fixed_mode_id: int | None = None,
            shuffled_mode_map=None,
            delay_steps: int | None = None) -> None:
        """Override only the context presented during policy evaluation."""
        kind = str(kind)
        if kind not in self.EVAL_CONTEXT_KINDS:
            raise ValueError(
                f"unknown regime eval context {kind!r}; expected one of "
                f"{self.EVAL_CONTEXT_KINDS}")
        if kind == "fixed":
            if fixed_mode_id is None:
                raise ValueError("fixed eval context requires fixed_mode_id")
            fixed_mode_id = int(fixed_mode_id)
            if not 0 <= fixed_mode_id < self.context_dim:
                raise ValueError(
                    f"fixed_mode_id={fixed_mode_id} outside "
                    f"[0,{self.context_dim - 1}]")
        elif fixed_mode_id is not None:
            raise ValueError(
                f"fixed_mode_id is invalid for eval context {kind!r}")
        if kind == "shuffled":
            if shuffled_mode_map is None:
                raise ValueError(
                    "shuffled eval context requires shuffled_mode_map")
            shuffled_mode_map = tuple(
                int(mode_id) for mode_id in shuffled_mode_map)
            expected = tuple(range(self.context_dim))
            if (len(shuffled_mode_map) != self.context_dim
                    or tuple(sorted(shuffled_mode_map)) != expected):
                raise ValueError(
                    "shuffled_mode_map must be a permutation of "
                    f"{expected}; got {shuffled_mode_map}")
        elif shuffled_mode_map is not None:
            raise ValueError(
                f"shuffled_mode_map is invalid for eval context {kind!r}")
        if kind == "delayed":
            if delay_steps is None:
                raise ValueError("delayed eval context requires delay_steps")
            delay_steps = int(delay_steps)
            if delay_steps < 0:
                raise ValueError("delay_steps must be nonnegative")
        elif delay_steps is not None:
            raise ValueError(
                f"delay_steps is invalid for eval context {kind!r}")
        self._eval_context_kind = kind
        self._eval_fixed_mode_id = fixed_mode_id
        self._eval_shuffled_mode_map = shuffled_mode_map
        self._eval_delay_steps = int(delay_steps or 0)
        self.reset_eval_context_state()

    def reset_eval_context_state(self) -> None:
        """Reset evaluation-only temporal context state between episodes."""
        self._eval_context_state_ready = False
        self._eval_context_mode_id = 0
        self._eval_pending_mode_id = None
        self._eval_delay_remaining = 0

    def _update_delayed_context(self, mode_id: int) -> None:
        if not self._eval_context_state_ready:
            self._eval_context_state_ready = True
            self._eval_context_mode_id = mode_id
            self._eval_pending_mode_id = mode_id
            self._eval_delay_remaining = 0
            return
        if mode_id == self._eval_context_mode_id:
            self._eval_pending_mode_id = mode_id
            self._eval_delay_remaining = 0
            return
        if mode_id != self._eval_pending_mode_id:
            self._eval_pending_mode_id = mode_id
            self._eval_delay_remaining = self._eval_delay_steps
            if self._eval_delay_remaining == 0:
                self._eval_context_mode_id = mode_id
            return
        if self._eval_delay_remaining > 1:
            self._eval_delay_remaining -= 1
        else:
            self._eval_context_mode_id = mode_id
            self._eval_delay_remaining = 0

    def eval_context_mode_id(self, mode_id: int | None = None) -> int:
        """Return the mode encoded in the policy input, or -1 for zeros."""
        physical_mode_id = (
            int(self._current_mode_id) if mode_id is None else int(mode_id))
        kind = self._eval_context_kind
        if kind == "checkpoint":
            return (
                -1 if self.context_mode == "robust"
                else physical_mode_id)
        if kind == "zero":
            return -1
        if kind == "fixed":
            return int(self._eval_fixed_mode_id)
        if kind == "cyclic":
            return (physical_mode_id + 1) % self.context_dim
        if kind == "shuffled":
            return int(
                self._eval_shuffled_mode_map[physical_mode_id])
        if kind == "delayed":
            return int(self._eval_context_mode_id)
        return physical_mode_id

    def eval_context_delay_remaining(self) -> int:
        if self._eval_context_kind != "delayed":
            return 0
        return int(self._eval_delay_remaining)

    def eval_context_for_task_id(self, mode_id: int):
        """Return the context selected by the evaluation-only override."""
        if not self._task_metadata_ready:
            raise RuntimeError("set_task_metadata must run before rollout")
        mode_id = int(mode_id)
        if not 0 <= mode_id < self.context_dim:
            raise ValueError(
                f"mode_id={mode_id} outside [0,{self.context_dim - 1}]")
        kind = self._eval_context_kind
        if kind == "checkpoint":
            return self.context_for_task_id(mode_id)
        if kind == "zero":
            return jnp.zeros((self.context_dim,), dtype=jnp.float32)
        context_mode_id = self.eval_context_mode_id(mode_id)
        return jax.nn.one_hot(
            context_mode_id, self.context_dim, dtype=jnp.float32)

    def set_oracle_task_id(self, mode_id: int) -> None:
        # The robust arm records the clock for auditing but never exposes it to
        # actor or critic inputs; context_for_task_id still returns zeros.
        mode_id = int(mode_id)
        if not 0 <= mode_id < self.context_dim:
            raise ValueError(
                f"mode_id={mode_id} outside [0,{self.context_dim - 1}]")
        self._current_mode_id = mode_id
        if self._eval_context_kind == "delayed":
            self._update_delayed_context(mode_id)

    def set_eval_task(self, task) -> None:
        self.set_oracle_task_id(int(task["mode_id"]))

    def rollout_context(self, iteration: int | None = None):
        del iteration
        return self.eval_context_for_task_id(self._current_mode_id)

    def _build_belief_jax(self):
        return self.rollout_context()

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    def _build_scan_fn(self) -> None:
        gamma = float(self.config.gamma)
        tau = float(self.config.tau)
        auto_alpha = bool(self.config.auto_alpha)
        target_entropy = jnp.asarray(self.target_entropy)
        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)
        p_opt = self.policy_opt
        c_opt = self.critic_opt
        a_opt = self.alpha_opt

        @jax.jit
        def scan_update(critic_params, target_params, policy_params,
                        log_alpha, c_opt_state, p_opt_state, a_opt_state,
                        all_obs, all_act, all_rew, all_next_obs, all_done,
                        all_context, all_next_context, rng_key):
            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key) = carry
                (obs, act, rew, next_obs, done,
                 context, next_context) = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)
                critic_obs = jnp.concatenate([obs, context], axis=-1)
                next_critic_obs = jnp.concatenate(
                    [next_obs, next_context], axis=-1)

                def critic_loss_fn(cp):
                    target = nnx.merge(gd_target, t_p)
                    policy = nnx.merge(gd_policy, p_p)
                    next_action, next_log_prob = policy.sample(
                        next_obs, k1, next_context)
                    next_q = target(next_critic_obs, next_action).min(axis=0)
                    target_q = (
                        rew.squeeze(-1)
                        + gamma * (1 - done.squeeze(-1))
                        * (next_q - alpha * next_log_prob)
                    )
                    critic = nnx.merge(gd_critic, cp)
                    predicted_q = critic(critic_obs, act)
                    loss = jnp.mean((predicted_q - target_q[None]) ** 2)
                    return loss, predicted_q

                (critic_loss, predicted_q), critic_grads = (
                    jax.value_and_grad(
                        critic_loss_fn, has_aux=True)(c_p))
                critic_updates, next_c_opt_state = c_opt.update(
                    critic_grads, c_os, c_p)
                next_c_p = optax.apply_updates(c_p, critic_updates)

                def policy_loss_fn(pp):
                    policy = nnx.merge(gd_policy, pp)
                    critic = nnx.merge(gd_critic, next_c_p)
                    action, log_prob = policy.sample(obs, k2, context)
                    q_value = critic(critic_obs, action)
                    loss = (jnp.exp(la) * log_prob
                            - q_value.mean(axis=0)).mean()
                    return loss, log_prob

                (policy_loss, log_prob), policy_grads = (
                    jax.value_and_grad(
                        policy_loss_fn, has_aux=True)(p_p))
                policy_updates, next_p_opt_state = p_opt.update(
                    policy_grads, p_os, p_p)
                next_p_p = optax.apply_updates(p_p, policy_updates)

                alpha_grad = -(log_prob.mean() + target_entropy)
                alpha_updates, next_a_opt_state = a_opt.update(
                    alpha_grad, a_os, la)
                next_la = jnp.where(auto_alpha, la + alpha_updates, la)
                next_a_opt_state = jax.tree.map(
                    lambda new, old: jnp.where(auto_alpha, new, old),
                    next_a_opt_state, a_os)
                next_t_p = jax.tree.map(
                    lambda target, critic: (
                        target * (1 - tau) + critic * tau),
                    t_p, next_c_p)
                next_carry = (
                    next_c_p, next_t_p, next_p_p, next_la,
                    next_c_opt_state, next_p_opt_state,
                    next_a_opt_state, key)
                metrics = (
                    critic_loss, policy_loss, jnp.exp(next_la),
                    predicted_q.mean(), predicted_q.std(axis=0).mean(),
                    log_prob.mean())
                return next_carry, metrics

            initial = (
                critic_params, target_params, policy_params, log_alpha,
                c_opt_state, p_opt_state, a_opt_state, rng_key)
            batches = (
                all_obs, all_act, all_rew, all_next_obs, all_done,
                all_context, all_next_context)
            return jax.lax.scan(body_fn, initial, batches)

        self._scan_update = scan_update

    def select_action(self, obs, deterministic: bool = False):
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        if obs_jax.ndim == 1:
            obs_jax = obs_jax[None]
        context = jnp.broadcast_to(
            self.rollout_context()[None],
            obs_jax.shape[:-1] + (self.context_dim,))
        if deterministic:
            action = self.policy.deterministic(obs_jax, context)
        else:
            action, _ = self.policy.sample(
                obs_jax, self.rngs.params(), context)
        return np.asarray(action[0])

    def multi_update(self, stacked_batch: dict, **kwargs):
        del kwargs
        context = stacked_batch.get("belief")
        next_context = stacked_batch.get("next_belief")
        if context is None or next_context is None:
            raise ValueError("regime_sac replay requires context fields")
        if context.shape[-1] != self.context_dim:
            raise ValueError(
                f"replay context width {context.shape[-1]} does not match "
                f"regime_sac width {self.context_dim}")
        if self.context_mode == "robust":
            context = jnp.zeros_like(context)
            next_context = jnp.zeros_like(next_context)

        final, metrics = self._scan_update(
            nnx.state(self.critic, nnx.Param),
            nnx.state(self.target_critic, nnx.Param),
            nnx.state(self.policy, nnx.Param),
            self.log_alpha,
            self.critic_opt_state,
            self.policy_opt_state,
            self.alpha_opt_state,
            stacked_batch["obs"], stacked_batch["act"],
            stacked_batch["rew"], stacked_batch["next_obs"],
            stacked_batch["done"], context, next_context,
            self.rngs.params())
        (new_critic, new_target, new_policy, self.log_alpha,
         self.critic_opt_state, self.policy_opt_state,
         self.alpha_opt_state, _) = final
        nnx.update(self.critic, new_critic)
        nnx.update(self.target_critic, new_target)
        nnx.update(self.policy, new_policy)
        self.update_count += int(stacked_batch["obs"].shape[0])

        critic_loss, policy_loss, alpha, q_mean, q_std, log_prob = metrics
        return {
            "critic_loss": float(critic_loss.mean()),
            "policy_loss": float(policy_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(log_prob.mean()),
            "regime_context_oracle": float(
                self.context_mode == "oracle"),
        }
