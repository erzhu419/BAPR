"""SAC with a configurable actor update period and validation selection."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.algos.sac_base import SACBase
from jax_experiments.common.checkpoint import (
    _restore_tree_like,
    _to_numpy_tree,
)


class SACPolicyStability(SACBase):
    """Keep critic updates dense while optionally thinning actor updates.

    The class also stores the policy with the best externally reported
    validation return. Training and checkpoint budgets remain unchanged; the
    caller decides whether to publish the final or validation-selected policy.
    """

    CHECKPOINT_SCHEMA = "bapr.sac-policy-stability.v20"

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.actor_update_period = int(
            getattr(config, "sac_actor_update_period", 1))
        if self.actor_update_period <= 0:
            raise ValueError("sac_actor_update_period must be positive")
        self.select_best_eval = bool(
            getattr(config, "sac_select_best_eval", False))
        super().__init__(obs_dim, act_dim, config, seed=seed)
        self.reset_selection_anchor()

    def reset_selection_anchor(self) -> None:
        """Reset selection state after an external warm-start is installed."""
        self._best_policy_params = deepcopy(
            nnx.state(self.policy, nnx.Param))
        self._best_eval_reward = -math.inf
        self._best_eval_update_count = int(self.update_count)
        self._validation_observations = 0

    def report_eval(self, eval_reward: float) -> None:
        """Record the current policy when validation return reaches a new max."""
        value = float(eval_reward)
        if not math.isfinite(value):
            return
        self._validation_observations += 1
        if value > self._best_eval_reward:
            self._best_eval_reward = value
            self._best_eval_update_count = int(self.update_count)
            self._best_policy_params = deepcopy(
                nnx.state(self.policy, nnx.Param))

    def selected_policy_state(self):
        """Return the preregistered final or best-validation policy state."""
        if self.select_best_eval:
            if self._validation_observations <= 0:
                raise RuntimeError(
                    "validation selection requested without an evaluation")
            return deepcopy(self._best_policy_params)
        return deepcopy(nnx.state(self.policy, nnx.Param))

    def selection_record(self) -> dict[str, Any]:
        selected_update = (
            self._best_eval_update_count
            if self.select_best_eval else int(self.update_count)
        )
        return {
            "selection": (
                "best_validation" if self.select_best_eval else "final"
            ),
            "actor_update_period": self.actor_update_period,
            "final_update_count": int(self.update_count),
            "selected_update_count": int(selected_update),
            "best_validation_return": (
                float(self._best_eval_reward)
                if math.isfinite(self._best_eval_reward) else None
            ),
            "validation_observations": int(self._validation_observations),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema": self.CHECKPOINT_SCHEMA,
            "actor_update_period": self.actor_update_period,
            "select_best_eval": self.select_best_eval,
            "best_policy_params": _to_numpy_tree(
                self._best_policy_params),
            "best_eval_reward": float(self._best_eval_reward),
            "best_eval_update_count": int(self._best_eval_update_count),
            "validation_observations": int(self._validation_observations),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if (
            state.get("schema") != self.CHECKPOINT_SCHEMA
            or int(state.get("actor_update_period", -1))
            != self.actor_update_period
            or bool(state.get("select_best_eval"))
            != self.select_best_eval
        ):
            raise ValueError("incompatible SAC policy-stability checkpoint")
        self._best_policy_params = _restore_tree_like(
            nnx.state(self.policy, nnx.Param),
            state["best_policy_params"],
            "best validation policy",
            allow_fallback=False,
        )
        self._best_eval_reward = float(state["best_eval_reward"])
        self._best_eval_update_count = int(
            state["best_eval_update_count"])
        self._validation_observations = int(
            state["validation_observations"])

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        actor_update_period = self.actor_update_period

        policy_graphdef = nnx.graphdef(self.policy)
        critic_graphdef = nnx.graphdef(self.critic)
        target_graphdef = nnx.graphdef(self.target_critic)
        policy_opt = self.policy_opt
        critic_opt = self.critic_opt
        alpha_opt = self.alpha_opt

        @jax.jit
        def _scan_update(
            critic_params,
            target_params,
            policy_params,
            log_alpha,
            critic_opt_state,
            policy_opt_state,
            alpha_opt_state,
            all_obs,
            all_act,
            all_rew,
            all_next_obs,
            all_done,
            rng_key,
            initial_update_count,
        ):
            def body_fn(carry, batch_data):
                (
                    critic_state,
                    target_state,
                    policy_state,
                    current_log_alpha,
                    current_critic_opt,
                    current_policy_opt,
                    current_alpha_opt,
                    key,
                    update_count,
                ) = carry
                obs, act, rew, next_obs, done = batch_data
                key, target_key, policy_key = jax.random.split(key, 3)
                alpha = jnp.exp(current_log_alpha)

                def critic_loss_fn(params):
                    target_model = nnx.merge(
                        target_graphdef, target_state)
                    policy_model = nnx.merge(
                        policy_graphdef, policy_state)
                    next_action, next_log_prob = policy_model.sample(
                        next_obs, target_key)
                    target_q = (
                        target_model(next_obs, next_action).min(axis=0)
                        - alpha * next_log_prob
                    )
                    target_value = (
                        rew.squeeze(-1)
                        + gamma * (1 - done.squeeze(-1)) * target_q
                    )
                    critic_model = nnx.merge(critic_graphdef, params)
                    predicted_q = critic_model(obs, act)
                    loss = jnp.mean(
                        (predicted_q - target_value[None]) ** 2)
                    return loss, predicted_q

                (critic_loss, predicted_q), critic_grads = (
                    jax.value_and_grad(
                        critic_loss_fn, has_aux=True)(critic_state)
                )
                critic_updates, next_critic_opt = critic_opt.update(
                    critic_grads, current_critic_opt, critic_state)
                next_critic = optax.apply_updates(
                    critic_state, critic_updates)

                def policy_loss_fn(params):
                    policy_model = nnx.merge(policy_graphdef, params)
                    critic_model = nnx.merge(
                        critic_graphdef, next_critic)
                    new_action, log_prob = policy_model.sample(
                        obs, policy_key)
                    q_value = critic_model(obs, new_action)
                    loss = (
                        jnp.exp(current_log_alpha) * log_prob
                        - q_value.mean(axis=0)
                    ).mean()
                    return loss, log_prob

                (policy_loss, log_prob), policy_grads = (
                    jax.value_and_grad(
                        policy_loss_fn, has_aux=True)(policy_state)
                )
                update_actor = jnp.equal(
                    jnp.mod(update_count, actor_update_period), 0)

                def apply_policy_update(operand):
                    grads, optimizer_state, params = operand
                    updates, next_optimizer = policy_opt.update(
                        grads, optimizer_state, params)
                    return (
                        optax.apply_updates(params, updates),
                        next_optimizer,
                    )

                def keep_policy(operand):
                    _, optimizer_state, params = operand
                    return params, optimizer_state

                next_policy, next_policy_opt = jax.lax.cond(
                    update_actor,
                    apply_policy_update,
                    keep_policy,
                    (policy_grads, current_policy_opt, policy_state),
                )

                alpha_grad = -(log_prob.mean() + target_entropy)

                def apply_alpha_update(operand):
                    grad, optimizer_state, value = operand
                    updates, next_optimizer = alpha_opt.update(
                        grad, optimizer_state, value)
                    next_value = jnp.where(
                        auto_alpha, value + updates, value)
                    next_optimizer = jax.tree.map(
                        lambda new, old: jnp.where(
                            auto_alpha, new, old),
                        next_optimizer,
                        optimizer_state,
                    )
                    return next_value, next_optimizer

                def keep_alpha(operand):
                    _, optimizer_state, value = operand
                    return value, optimizer_state

                next_log_alpha, next_alpha_opt = jax.lax.cond(
                    update_actor,
                    apply_alpha_update,
                    keep_alpha,
                    (alpha_grad, current_alpha_opt, current_log_alpha),
                )
                next_target = jax.tree.map(
                    lambda target, critic: (
                        target * (1 - tau) + critic * tau),
                    target_state,
                    next_critic,
                )
                next_carry = (
                    next_critic,
                    next_target,
                    next_policy,
                    next_log_alpha,
                    next_critic_opt,
                    next_policy_opt,
                    next_alpha_opt,
                    key,
                    update_count + 1,
                )
                metrics = (
                    critic_loss,
                    policy_loss,
                    jnp.exp(next_log_alpha),
                    predicted_q.mean(),
                    predicted_q.std(axis=0).mean(),
                    log_prob.mean(),
                    update_actor.astype(jnp.float32),
                )
                return next_carry, metrics

            initial_carry = (
                critic_params,
                target_params,
                policy_params,
                log_alpha,
                critic_opt_state,
                policy_opt_state,
                alpha_opt_state,
                rng_key,
                jnp.asarray(initial_update_count, dtype=jnp.int32),
            )
            batches = (
                all_obs,
                all_act,
                all_rew,
                all_next_obs,
                all_done,
            )
            return jax.lax.scan(body_fn, initial_carry, batches)

        self._scan_update = _scan_update

    def multi_update(self, stacked_batch: dict, **kwargs):
        del kwargs
        rng_key = self.rngs.params()
        critic_params = nnx.state(self.critic, nnx.Param)
        target_params = nnx.state(self.target_critic, nnx.Param)
        policy_params = nnx.state(self.policy, nnx.Param)

        final, metrics = self._scan_update(
            critic_params,
            target_params,
            policy_params,
            self.log_alpha,
            self.critic_opt_state,
            self.policy_opt_state,
            self.alpha_opt_state,
            stacked_batch["obs"],
            stacked_batch["act"],
            stacked_batch["rew"],
            stacked_batch["next_obs"],
            stacked_batch["done"],
            rng_key,
            self.update_count,
        )
        (
            new_critic,
            new_target,
            new_policy,
            new_log_alpha,
            self.critic_opt_state,
            self.policy_opt_state,
            self.alpha_opt_state,
            _,
            final_update_count,
        ) = final
        nnx.update(self.critic, new_critic)
        nnx.update(self.target_critic, new_target)
        nnx.update(self.policy, new_policy)
        self.log_alpha = new_log_alpha
        self.update_count = int(final_update_count)

        (
            critic_loss,
            policy_loss,
            alpha,
            q_mean,
            q_std,
            log_prob,
            actor_update_mask,
        ) = metrics
        return {
            "critic_loss": float(critic_loss.mean()),
            "policy_loss": float(policy_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(log_prob.mean()),
            "actor_update_fraction": float(actor_update_mask.mean()),
        }

