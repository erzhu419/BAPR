"""SAC variant that lets a fresh critic calibrate before actor updates."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.algos.sac_base import SACBase


class SACActorDelay(SACBase):
    """Keep actor and temperature fixed until an absolute update count."""

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        actor_update_after = int(self.config.sac_actor_update_after)

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
                         rng_key, initial_update_count):
            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key,
                 update_count) = carry
                obs, act, rew, next_obs, done = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                def critic_loss_fn(cp):
                    t_model = nnx.merge(gd_target, t_p)
                    p_model = nnx.merge(gd_policy, p_p)
                    next_action, next_log_prob = p_model.sample(next_obs, k1)
                    target_q = (
                        t_model(next_obs, next_action).min(axis=0)
                        - alpha * next_log_prob
                    )
                    target_value = (
                        rew.squeeze(-1)
                        + gamma * (1 - done.squeeze(-1)) * target_q
                    )
                    c_model = nnx.merge(gd_critic, cp)
                    predicted_q = c_model(obs, act)
                    loss = jnp.mean(
                        (predicted_q - target_value[None]) ** 2)
                    return loss, predicted_q

                (critic_loss, predicted_q), critic_grads = (
                    jax.value_and_grad(critic_loss_fn, has_aux=True)(c_p)
                )
                critic_updates, new_c_os = c_opt.update(
                    critic_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, critic_updates)

                def policy_loss_fn(pp):
                    policy_model = nnx.merge(gd_policy, pp)
                    critic_model = nnx.merge(gd_critic, new_c_p)
                    new_action, log_prob = policy_model.sample(obs, k2)
                    q_value = critic_model(obs, new_action)
                    loss = (
                        jnp.exp(la) * log_prob - q_value.mean(axis=0)
                    ).mean()
                    return loss, log_prob

                (policy_loss, log_prob), policy_grads = (
                    jax.value_and_grad(policy_loss_fn, has_aux=True)(p_p)
                )
                update_actor = update_count >= actor_update_after

                def apply_policy_update(operand):
                    grads, opt_state, params = operand
                    updates, new_opt_state = p_opt.update(
                        grads, opt_state, params)
                    return optax.apply_updates(params, updates), new_opt_state

                def keep_policy(operand):
                    _, opt_state, params = operand
                    return params, opt_state

                new_p_p, new_p_os = jax.lax.cond(
                    update_actor,
                    apply_policy_update,
                    keep_policy,
                    (policy_grads, p_os, p_p),
                )

                alpha_grad = -(log_prob.mean() + target_entropy)

                def apply_alpha_update(operand):
                    grad, opt_state, value = operand
                    updates, new_opt_state = a_opt.update(
                        grad, opt_state, value)
                    new_value = jnp.where(
                        auto_alpha, value + updates, value)
                    new_opt_state = jax.tree.map(
                        lambda new, old: jnp.where(
                            auto_alpha, new, old),
                        new_opt_state,
                        opt_state,
                    )
                    return new_value, new_opt_state

                def keep_alpha(operand):
                    _, opt_state, value = operand
                    return value, opt_state

                new_la, new_a_os = jax.lax.cond(
                    update_actor,
                    apply_alpha_update,
                    keep_alpha,
                    (alpha_grad, a_os, la),
                )
                new_t_p = jax.tree.map(
                    lambda target, critic: (
                        target * (1 - tau) + critic * tau),
                    t_p,
                    new_c_p,
                )
                new_carry = (
                    new_c_p,
                    new_t_p,
                    new_p_p,
                    new_la,
                    new_c_os,
                    new_p_os,
                    new_a_os,
                    key,
                    update_count + 1,
                )
                metrics = (
                    critic_loss,
                    policy_loss,
                    jnp.exp(new_la),
                    predicted_q.mean(),
                    predicted_q.std(axis=0).mean(),
                    log_prob.mean(),
                    update_actor.astype(jnp.float32),
                )
                return new_carry, metrics

            initial_carry = (
                critic_params,
                target_params,
                policy_params,
                log_alpha,
                c_opt_state,
                p_opt_state,
                a_opt_state,
                rng_key,
                jnp.asarray(initial_update_count, dtype=jnp.int32),
            )
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done)
            return jax.lax.scan(body_fn, initial_carry, batches)

        self._scan_update = _scan_update

    def multi_update(self, stacked_batch: dict, **kwargs):
        rng_key = self.rngs.params()
        critic_params = nnx.state(self.critic, nnx.Param)
        target_params = nnx.state(self.target_critic, nnx.Param)
        policy_params = nnx.state(self.policy, nnx.Param)
        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        next_obs = stacked_batch["next_obs"]
        done = stacked_batch["done"]

        final, metrics = self._scan_update(
            critic_params,
            target_params,
            policy_params,
            self.log_alpha,
            self.critic_opt_state,
            self.policy_opt_state,
            self.alpha_opt_state,
            obs,
            act,
            rew,
            next_obs,
            done,
            rng_key,
            self.update_count,
        )
        (new_critic, new_target, new_policy, new_log_alpha,
         self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
         _, final_update_count) = final
        nnx.update(self.critic, new_critic)
        nnx.update(self.target_critic, new_target)
        nnx.update(self.policy, new_policy)
        self.log_alpha = new_log_alpha
        self.update_count = int(final_update_count)

        (critic_loss, policy_loss, alpha, q_mean, q_std, log_prob,
         actor_update_mask) = metrics
        return {
            "critic_loss": float(critic_loss.mean()),
            "policy_loss": float(policy_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(log_prob.mean()),
            "actor_update_fraction": float(actor_update_mask.mean()),
        }
