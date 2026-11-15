"""Switch-recovery SAC with an explicit termination-risk critic."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from jax_experiments.algos.sac_switch_recovery import SACSwitchRecovery
from jax_experiments.common.checkpoint import _restore_tree_like, _to_numpy_tree
from jax_experiments.networks.ensemble_critic import EnsembleCritic


RISK_OBJECTIVES = ("absolute", "relative")


def risk_constraint(candidate_probability, fallback_probability, objective):
    """Return the per-state risk term used by the actor objective."""
    if objective == "absolute":
        return candidate_probability
    if objective == "relative":
        return jax.nn.relu(candidate_probability - fallback_probability)
    raise ValueError(f"unknown risk objective {objective!r}")


class SACSwitchRecoveryRisk(SACSwitchRecovery):
    """SAC specialist constrained by learned discounted failure probability."""

    CHECKPOINT_SCHEMA = "bapr.sac-switch-recovery-risk.v1"

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        super().__init__(obs_dim, act_dim, config, seed=seed)
        self.risk_objective = str(config.switch_recovery_risk_objective)
        if self.risk_objective not in RISK_OBJECTIVES:
            raise ValueError(f"unknown risk objective {self.risk_objective!r}")
        self.risk_lambda = float(config.switch_recovery_risk_lambda)
        self.risk_actor_start_update = int(
            config.switch_recovery_risk_actor_start_update)
        self.risk_critic = EnsembleCritic(
            obs_dim,
            act_dim,
            config.hidden_dim,
            ensemble_size=config.ensemble_size,
            n_layers=3,
            rngs=self.rngs,
        )
        self.target_risk_critic = deepcopy(self.risk_critic)
        self.risk_opt = optax.adam(config.lr)
        self.risk_opt_state = self.risk_opt.init(
            nnx.state(self.risk_critic, nnx.Param))
        self._build_risk_scan_fn()

    def _build_risk_scan_fn(self) -> None:
        gamma = float(self.config.gamma)
        tau = float(self.config.tau)
        auto_alpha = bool(self.config.auto_alpha)
        target_entropy = jnp.asarray(self.target_entropy)
        risk_lambda = jnp.asarray(self.risk_lambda)
        risk_actor_start = int(self.risk_actor_start_update)
        risk_objective = self.risk_objective

        policy_graph = nnx.graphdef(self.policy)
        fallback_graph = nnx.graphdef(self.fallback_policy)
        critic_graph = nnx.graphdef(self.critic)
        target_graph = nnx.graphdef(self.target_critic)
        risk_graph = nnx.graphdef(self.risk_critic)
        target_risk_graph = nnx.graphdef(self.target_risk_critic)
        policy_opt = self.policy_opt
        critic_opt = self.critic_opt
        alpha_opt = self.alpha_opt
        risk_opt = self.risk_opt

        @jax.jit
        def scan_update(
            critic_params,
            target_params,
            policy_params,
            fallback_params,
            risk_params,
            target_risk_params,
            log_alpha,
            critic_opt_state,
            policy_opt_state,
            alpha_opt_state,
            risk_opt_state,
            all_obs,
            all_act,
            all_rew,
            all_next_obs,
            all_done,
            initial_update_count,
            rng_key,
        ):
            def body(carry, batch):
                (
                    c_p,
                    t_p,
                    p_p,
                    r_p,
                    tr_p,
                    la,
                    c_os,
                    p_os,
                    a_os,
                    r_os,
                    key,
                ) = carry
                obs, act, rew, next_obs, done, step_index = batch
                key, next_key, policy_key = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                def critic_loss_fn(params):
                    target_model = nnx.merge(target_graph, t_p)
                    policy_model = nnx.merge(policy_graph, p_p)
                    next_action, next_log_prob = policy_model.sample(
                        next_obs, next_key)
                    target_q = (
                        target_model(next_obs, next_action).min(axis=0)
                        - alpha * next_log_prob
                    )
                    target_value = (
                        rew.squeeze(-1)
                        + gamma * (1.0 - done.squeeze(-1)) * target_q
                    )
                    critic_model = nnx.merge(critic_graph, params)
                    predicted_q = critic_model(obs, act)
                    loss = jnp.mean(
                        (predicted_q - target_value[None, :]) ** 2)
                    return loss, predicted_q

                (critic_loss, predicted_q), critic_grad = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                critic_updates, new_c_os = critic_opt.update(
                    critic_grad, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, critic_updates)

                def risk_loss_fn(params):
                    policy_model = nnx.merge(policy_graph, p_p)
                    target_risk_model = nnx.merge(
                        target_risk_graph, tr_p)
                    next_action, _ = policy_model.sample(next_obs, next_key)
                    next_probability = jax.nn.sigmoid(
                        target_risk_model(next_obs, next_action))
                    terminal = done.squeeze(-1)[None, :]
                    target_probability = jnp.clip(
                        terminal
                        + gamma * (1.0 - terminal) * next_probability,
                        0.0,
                        1.0,
                    )
                    risk_model = nnx.merge(risk_graph, params)
                    logits = risk_model(obs, act)
                    loss = optax.sigmoid_binary_cross_entropy(
                        logits,
                        jax.lax.stop_gradient(target_probability),
                    ).mean()
                    return loss, jax.nn.sigmoid(logits)

                (risk_loss, replay_risk), risk_grad = jax.value_and_grad(
                    risk_loss_fn, has_aux=True)(r_p)
                risk_updates, new_r_os = risk_opt.update(
                    risk_grad, r_os, r_p)
                new_r_p = optax.apply_updates(r_p, risk_updates)

                def policy_loss_fn(params):
                    policy_model = nnx.merge(policy_graph, params)
                    fallback_model = nnx.merge(
                        fallback_graph, fallback_params)
                    critic_model = nnx.merge(critic_graph, new_c_p)
                    risk_model = nnx.merge(risk_graph, new_r_p)
                    action, log_prob = policy_model.sample(obs, policy_key)
                    q_value = critic_model(obs, action).mean(axis=0)
                    candidate_risk = jax.nn.sigmoid(
                        risk_model(obs, action)).max(axis=0)
                    fallback_action = fallback_model.deterministic(obs)
                    fallback_risk = jax.lax.stop_gradient(
                        jax.nn.sigmoid(
                            risk_model(obs, fallback_action)).max(axis=0))
                    constrained_risk = risk_constraint(
                        candidate_risk,
                        fallback_risk,
                        risk_objective,
                    )
                    loss = (
                        alpha * log_prob
                        - q_value
                        + risk_lambda * constrained_risk
                    ).mean()
                    return loss, (
                        log_prob,
                        candidate_risk,
                        fallback_risk,
                        constrained_risk,
                    )

                (policy_loss, policy_aux), policy_grad = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                policy_updates, candidate_p_os = policy_opt.update(
                    policy_grad, p_os, p_p)
                candidate_p_p = optax.apply_updates(p_p, policy_updates)
                actor_enabled = (
                    initial_update_count + step_index >= risk_actor_start)
                new_p_p = jax.tree.map(
                    lambda new, old: jnp.where(actor_enabled, new, old),
                    candidate_p_p,
                    p_p,
                )
                new_p_os = jax.tree.map(
                    lambda new, old: jnp.where(actor_enabled, new, old),
                    candidate_p_os,
                    p_os,
                )

                log_prob = policy_aux[0]
                alpha_grad = -(log_prob.mean() + target_entropy)
                alpha_updates, candidate_a_os = alpha_opt.update(
                    alpha_grad, a_os, la)
                candidate_la = jnp.where(auto_alpha, la + alpha_updates, la)
                candidate_a_os = jax.tree.map(
                    lambda new, old: jnp.where(auto_alpha, new, old),
                    candidate_a_os,
                    a_os,
                )
                new_la = jnp.where(actor_enabled, candidate_la, la)
                new_a_os = jax.tree.map(
                    lambda new, old: jnp.where(actor_enabled, new, old),
                    candidate_a_os,
                    a_os,
                )

                new_t_p = jax.tree.map(
                    lambda target, online: (
                        target * (1.0 - tau) + online * tau),
                    t_p,
                    new_c_p,
                )
                new_tr_p = jax.tree.map(
                    lambda target, online: (
                        target * (1.0 - tau) + online * tau),
                    tr_p,
                    new_r_p,
                )
                new_carry = (
                    new_c_p,
                    new_t_p,
                    new_p_p,
                    new_r_p,
                    new_tr_p,
                    new_la,
                    new_c_os,
                    new_p_os,
                    new_a_os,
                    new_r_os,
                    key,
                )
                metrics = (
                    critic_loss,
                    policy_loss,
                    risk_loss,
                    jnp.exp(new_la),
                    predicted_q.mean(),
                    predicted_q.std(axis=0).mean(),
                    log_prob.mean(),
                    replay_risk.mean(),
                    policy_aux[1].mean(),
                    policy_aux[2].mean(),
                    policy_aux[3].mean(),
                    actor_enabled.astype(jnp.float32),
                )
                return new_carry, metrics

            initial = (
                critic_params,
                target_params,
                policy_params,
                risk_params,
                target_risk_params,
                log_alpha,
                critic_opt_state,
                policy_opt_state,
                alpha_opt_state,
                risk_opt_state,
                rng_key,
            )
            step_indices = jnp.arange(all_obs.shape[0], dtype=jnp.int32)
            batches = (
                all_obs,
                all_act,
                all_rew,
                all_next_obs,
                all_done,
                step_indices,
            )
            return jax.lax.scan(body, initial, batches)

        self._risk_scan_update = scan_update

    def multi_update(self, stacked_batch: dict, **kwargs):
        del kwargs
        critic_params = nnx.state(self.critic, nnx.Param)
        target_params = nnx.state(self.target_critic, nnx.Param)
        policy_params = nnx.state(self.policy, nnx.Param)
        fallback_params = nnx.state(self.fallback_policy, nnx.Param)
        risk_params = nnx.state(self.risk_critic, nnx.Param)
        target_risk_params = nnx.state(self.target_risk_critic, nnx.Param)
        final, metrics = self._risk_scan_update(
            critic_params,
            target_params,
            policy_params,
            fallback_params,
            risk_params,
            target_risk_params,
            self.log_alpha,
            self.critic_opt_state,
            self.policy_opt_state,
            self.alpha_opt_state,
            self.risk_opt_state,
            stacked_batch["obs"],
            stacked_batch["act"],
            stacked_batch["rew"],
            stacked_batch["next_obs"],
            stacked_batch["done"],
            jnp.asarray(self.update_count, dtype=jnp.int32),
            self.rngs.params(),
        )
        (
            new_critic,
            new_target,
            new_policy,
            new_risk,
            new_target_risk,
            new_log_alpha,
            self.critic_opt_state,
            self.policy_opt_state,
            self.alpha_opt_state,
            self.risk_opt_state,
            _,
        ) = final
        nnx.update(self.critic, new_critic)
        nnx.update(self.target_critic, new_target)
        nnx.update(self.policy, new_policy)
        nnx.update(self.risk_critic, new_risk)
        nnx.update(self.target_risk_critic, new_target_risk)
        self.log_alpha = new_log_alpha
        self.update_count += int(stacked_batch["obs"].shape[0])

        (
            critic_loss,
            policy_loss,
            risk_loss,
            alpha,
            q_mean,
            q_std,
            log_prob,
            replay_risk,
            candidate_risk,
            fallback_risk,
            constrained_risk,
            actor_enabled,
        ) = metrics
        return {
            "critic_loss": float(critic_loss.mean()),
            "policy_loss": float(policy_loss.mean()),
            "risk_critic_loss": float(risk_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(log_prob.mean()),
            "risk_replay_probability": float(replay_risk.mean()),
            "risk_candidate_probability": float(candidate_risk.mean()),
            "risk_fallback_probability": float(fallback_risk.mean()),
            "risk_constraint": float(constrained_risk.mean()),
            "risk_actor_enabled": float(actor_enabled.mean()),
            "switch_recovery_target_mode": float(
                self.switch_recovery_target_mode),
            "switch_recovery_termination_penalty": 0.0,
            "switch_recovery_termination_rate": float(
                self._last_switch_recovery_termination_rate),
            "switch_recovery_raw_reward": float(
                self._last_switch_recovery_raw_reward),
            "frozen_robust_actor": 1.0,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema": self.CHECKPOINT_SCHEMA,
            "base": super().checkpoint_state(),
            "risk_objective": self.risk_objective,
            "risk_lambda": self.risk_lambda,
            "risk_actor_start_update": self.risk_actor_start_update,
            "risk_critic": _to_numpy_tree(
                nnx.state(self.risk_critic, nnx.Param)),
            "target_risk_critic": _to_numpy_tree(
                nnx.state(self.target_risk_critic, nnx.Param)),
            "risk_opt_state": _to_numpy_tree(self.risk_opt_state),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if (
            state.get("schema") != self.CHECKPOINT_SCHEMA
            or state.get("risk_objective") != self.risk_objective
            or float(state.get("risk_lambda", -1.0)) != self.risk_lambda
            or int(state.get("risk_actor_start_update", -1))
            != self.risk_actor_start_update
        ):
            raise ValueError("incompatible switch-recovery risk checkpoint")
        super().load_checkpoint_state(state["base"])
        nnx.update(
            self.risk_critic,
            _restore_tree_like(
                nnx.state(self.risk_critic, nnx.Param),
                state["risk_critic"],
                "switch-recovery risk critic",
                allow_fallback=False,
            ),
        )
        nnx.update(
            self.target_risk_critic,
            _restore_tree_like(
                nnx.state(self.target_risk_critic, nnx.Param),
                state["target_risk_critic"],
                "switch-recovery target risk critic",
                allow_fallback=False,
            ),
        )
        self.risk_opt_state = _restore_tree_like(
            self.risk_opt_state,
            state["risk_opt_state"],
            "switch-recovery risk optimizer",
            allow_fallback=False,
        )
