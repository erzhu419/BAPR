"""RE-SAC with explicit legacy and paper-B0 compatibility controls."""
from collections import deque
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.algos.sac_base import SACBase


class RESAC(SACBase):
    """Ensemble SAC with LCB policy optimization.

    ``resac_independent_ratio=1`` preserves the legacy BAPR fork.  The
    completed MuJoCo B0 artifact used a 0.75 independent/min target blend,
    an EMA evaluation actor, and an anchor to the best evaluated actor.  These
    are opt-in so an old checkpoint never changes semantics on resume.

    The positive target shift used by the bus implementation is intentionally
    preserved.  MuJoCo B0 comparisons disable it with ``weight_reg=0`` rather
    than silently changing its sign.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (bool(getattr(self.config, "use_ema_eval", False))
                or bool(getattr(self.config, "use_ema_rollout", False))):
            self.ema_policy = deepcopy(self.policy)
        self._anchor_params = nnx.state(self.policy, nnx.Param)
        self._best_eval = -float("inf")
        self._eval_history = deque(maxlen=10)
        self._perf_gate_active = False
        self._current_beta = float(self.config.beta)

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        weight_reg = float(getattr(self.config, "weight_reg", 0.0))
        beta_ood = float(getattr(self.config, "beta_ood", 0.0))
        beta_bc = float(getattr(self.config, "resac_beta_bc", 0.0))
        independent_ratio = float(getattr(
            self.config, "resac_independent_ratio", 1.0))
        if not 0.0 <= independent_ratio <= 1.0:
            raise ValueError("resac_independent_ratio must be in [0, 1]")
        anchor_lambda = float(getattr(
            self.config, "resac_anchor_lambda", 0.0))
        critic_actor_ratio = int(getattr(
            self.config, "resac_critic_actor_ratio", 1))
        if critic_actor_ratio <= 0:
            raise ValueError("critic_actor_ratio must be positive")
        clip_norm = float(getattr(
            self.config, "resac_clip_norm", 0.0))
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)

        policy_opt = self.policy_opt
        critic_opt = self.critic_opt
        alpha_opt = self.alpha_opt

        def clip_grads(grads):
            if clip_norm <= 0.0:
                return grads
            grad_norm = optax.global_norm(grads)
            scale = jnp.minimum(
                1.0, jnp.asarray(clip_norm) / (grad_norm + 1e-8))
            return jax.tree.map(lambda grad: grad * scale, grads)

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         log_alpha, critic_opt_state, policy_opt_state,
                         alpha_opt_state, all_obs, all_act, all_rew,
                         all_next_obs, all_done, rng_key, update_offset,
                         beta_lcb, anchor_params):

            def body_fn(carry, batch_data):
                (critic_p, target_p, policy_p, log_a, critic_os, policy_os,
                 alpha_os, key) = carry
                (obs, act, rew, next_obs, done, update_index) = batch_data
                key, target_key, policy_key = jax.random.split(key, 3)
                alpha = jnp.exp(log_a)

                def critic_loss_fn(candidate_critic_p):
                    target_model = nnx.merge(gd_target, target_p)
                    policy_model = nnx.merge(gd_policy, policy_p)
                    next_action, next_log_prob = policy_model.sample(
                        next_obs, target_key)
                    target_q_all = target_model(next_obs, next_action)
                    target_q_min = target_q_all.min(axis=0)
                    target_q = (
                        independent_ratio * target_q_all
                        + (1.0 - independent_ratio) * target_q_min[None])
                    # Do not flip this sign.  It is the empirically validated
                    # bus implementation; deterministic MuJoCo B0 sets the
                    # coefficient to zero.
                    reg_bonus = (
                        weight_reg
                        * target_model.compute_reg_norm()[:, None])
                    bootstrap = target_q + reg_bonus - alpha * next_log_prob
                    target = (
                        rew.squeeze(-1)
                        + gamma * (1 - done.squeeze(-1)) * bootstrap)
                    critic_model = nnx.merge(
                        gd_critic, candidate_critic_p)
                    predicted_q = critic_model(obs, act)
                    loss = jnp.mean((predicted_q - target) ** 2)
                    if beta_ood != 0.0:
                        loss = (
                            loss
                            + beta_ood
                            * predicted_q.std(axis=0).mean())
                    return loss, (predicted_q, reg_bonus)

                (critic_loss, (predicted_q, reg_bonus)), critic_grads = (
                    jax.value_and_grad(
                        critic_loss_fn, has_aux=True)(critic_p))
                critic_grads = clip_grads(critic_grads)
                critic_upd, new_critic_os = critic_opt.update(
                    critic_grads, critic_os, critic_p)
                new_critic_p = optax.apply_updates(critic_p, critic_upd)

                def policy_loss_fn(candidate_policy_p):
                    policy_model = nnx.merge(
                        gd_policy, candidate_policy_p)
                    critic_model = nnx.merge(gd_critic, new_critic_p)
                    new_action, log_prob = policy_model.sample(
                        obs, policy_key)
                    target_reg = (
                        weight_reg
                        * nnx.merge(
                            gd_target, target_p).compute_reg_norm()[:, None])
                    q_values = critic_model(obs, new_action) + target_reg
                    q_mean = q_values.mean(axis=0)
                    q_std = q_values.std(axis=0)
                    lcb = q_mean + beta_lcb * q_std
                    bc_loss = jnp.mean((new_action - act) ** 2)
                    base_loss = (
                        alpha * log_prob - lcb).mean() + beta_bc * bc_loss

                    anchor_sq = jax.tree.map(
                        lambda current, anchor: jnp.sum(
                            (current - anchor) ** 2),
                        candidate_policy_p, anchor_params)
                    anchor_count = jax.tree.map(
                        lambda value: float(value.size), candidate_policy_p)
                    normalized_anchor = (
                        sum(jax.tree.leaves(anchor_sq))
                        / jnp.maximum(
                            sum(jax.tree.leaves(anchor_count)), 1.0))
                    return (
                        base_loss + anchor_lambda * normalized_anchor,
                        (log_prob, bc_loss))

                (policy_loss, (log_prob, bc_loss)), policy_grads = (
                    jax.value_and_grad(
                        policy_loss_fn, has_aux=True)(policy_p))
                policy_grads = clip_grads(policy_grads)
                policy_upd, candidate_policy_os = policy_opt.update(
                    policy_grads, policy_os, policy_p)
                candidate_policy_p = optax.apply_updates(
                    policy_p, policy_upd)
                update_actor = jnp.equal(
                    jnp.mod(update_index, critic_actor_ratio), 0)
                new_policy_p, new_policy_os = jax.lax.cond(
                    update_actor,
                    lambda _: (candidate_policy_p, candidate_policy_os),
                    lambda _: (policy_p, policy_os),
                    operand=None)

                # Match the released JAX SAC/RE-SAC implementation: update
                # log(alpha) after the actor and do not introduce an extra
                # chain-rule factor into the explicitly supplied gradient.
                alpha_grad = -(log_prob.mean() + target_entropy)
                alpha_upd, candidate_alpha_os = alpha_opt.update(
                    alpha_grad, alpha_os, log_a)
                new_log_a = jnp.where(
                    auto_alpha, log_a + alpha_upd, log_a)
                new_alpha_os = jax.tree.map(
                    lambda candidate, old: jnp.where(
                        auto_alpha, candidate, old),
                    candidate_alpha_os, alpha_os)

                new_target_p = jax.tree.map(
                    lambda target_value, critic_value: (
                        target_value * (1 - tau) + critic_value * tau),
                    target_p, new_critic_p)
                new_carry = (
                    new_critic_p, new_target_p, new_policy_p, new_log_a,
                    new_critic_os, new_policy_os, new_alpha_os, key)
                metrics = (
                    critic_loss, policy_loss, jnp.exp(new_log_a),
                    predicted_q.mean(),
                    predicted_q.std(axis=0).mean(), log_prob.mean(),
                    bc_loss, update_actor.astype(jnp.float32),
                    reg_bonus.mean(), reg_bonus.std())
                return new_carry, metrics

            init = (
                critic_params, target_params, policy_params, log_alpha,
                critic_opt_state, policy_opt_state, alpha_opt_state, rng_key)
            update_indices = (
                jnp.arange(all_obs.shape[0], dtype=jnp.int32)
                + jnp.asarray(update_offset, dtype=jnp.int32))
            return jax.lax.scan(
                body_fn, init,
                (all_obs, all_act, all_rew, all_next_obs, all_done,
                 update_indices))

        self._scan_update = _scan_update

    def get_adaptive_beta(self, current_iter: int) -> float:
        if not bool(getattr(self.config, "resac_adaptive_beta", False)):
            return float(self.config.beta)
        max_iters = max(int(self.config.max_iters), 1)
        warmup = int(float(self.config.resac_beta_warmup) * max_iters)
        if current_iter < warmup:
            value = float(self.config.resac_beta_start)
        else:
            progress = min(
                1.0,
                (current_iter - warmup) / max(max_iters - warmup, 1))
            value = (
                float(self.config.resac_beta_start)
                + progress * (
                    float(self.config.resac_beta_end)
                    - float(self.config.resac_beta_start)))
            if len(self._eval_history) >= 6:
                earlier = float(np.mean(list(self._eval_history)[-6:-3]))
                later = float(np.mean(list(self._eval_history)[-3:]))
                if earlier > 0.0 and later < 0.9 * earlier:
                    value = 0.5 * (
                        value + float(self.config.resac_beta_start))
        self._current_beta = value
        return value

    def report_eval(self, eval_reward: float):
        value = float(eval_reward)
        self._eval_history.append(value)
        if value > self._best_eval:
            self._best_eval = value
            self._anchor_params = nnx.state(self.policy, nnx.Param)

    def _update_ema_policy(self):
        if not hasattr(self, "ema_policy"):
            return
        ema_params = nnx.state(self.ema_policy, nnx.Param)
        policy_params = nnx.state(self.policy, nnx.Param)
        ema_tau = float(self.config.ema_tau)
        nnx.update(self.ema_policy, jax.tree.map(
            lambda ema, current: (
                ema * (1.0 - ema_tau) + current * ema_tau),
            ema_params, policy_params))

    def checkpoint_state(self):
        return {
            "kind": "resac_compat_v1",
            "anchor_params": jax.tree.map(
                np.asarray, self._anchor_params),
            "best_eval": float(self._best_eval),
            "current_beta": float(self._current_beta),
        }

    def load_checkpoint_state(self, state):
        if state.get("kind") != "resac_compat_v1":
            return
        if "anchor_params" in state:
            # Checkpoints may be written by a nearby Flax NNX release whose
            # VariableState metadata differs from the current runtime.  The
            # network and optimizer restore paths already rebuild saved leaves
            # with current treedef metadata; the frozen policy anchor needs the
            # same treatment or the first resumed actor update fails inside
            # jax.tree.map despite matching parameter shapes.
            candidate = jax.tree.map(jnp.asarray, state["anchor_params"])
            template_leaves, template_def = jax.tree.flatten(
                self._anchor_params)
            saved_leaves, saved_def = jax.tree.flatten(candidate)
            if template_def == saved_def:
                self._anchor_params = candidate
            elif (len(template_leaves) == len(saved_leaves)
                  and all(
                      getattr(current, "shape", None)
                      == getattr(saved, "shape", None)
                      for current, saved in zip(
                          template_leaves, saved_leaves))):
                self._anchor_params = jax.tree.unflatten(
                    template_def, saved_leaves)
                print(
                    "  Checkpoint compatibility: remapped RE-SAC policy "
                    "anchor leaves to the current Flax pytree metadata")
            else:
                raise ValueError(
                    "Checkpoint RE-SAC policy anchor does not match the "
                    "current policy architecture")
        self._best_eval = float(state.get("best_eval", self._best_eval))
        self._current_beta = float(state.get(
            "current_beta", self._current_beta))

    def multi_update(self, stacked_batch: dict, current_iter=0, **kwargs):
        del kwargs
        rng_key = self.rngs.params()
        critic_params = nnx.state(self.critic, nnx.Param)
        target_params = nnx.state(self.target_critic, nnx.Param)
        policy_params = nnx.state(self.policy, nnx.Param)
        observations = stacked_batch["obs"]
        beta_lcb = jnp.asarray(
            self.get_adaptive_beta(int(current_iter)), dtype=jnp.float32)

        final, metrics = self._scan_update(
            critic_params, target_params, policy_params, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state,
            self.alpha_opt_state, observations, stacked_batch["act"],
            stacked_batch["rew"], stacked_batch["next_obs"],
            stacked_batch["done"], rng_key, self.update_count, beta_lcb,
            self._anchor_params)
        (new_critic, new_target, new_policy, new_log_alpha,
         self.critic_opt_state, self.policy_opt_state,
         self.alpha_opt_state, _) = final
        nnx.update(self.critic, new_critic)
        nnx.update(self.target_critic, new_target)
        nnx.update(self.policy, new_policy)
        self.log_alpha = new_log_alpha
        self.update_count += int(observations.shape[0])
        self._update_ema_policy()

        (critic_loss, policy_loss, alpha, q_mean, q_std, log_prob,
         bc_loss, actor_updated, reg_bonus_mean, reg_bonus_std) = metrics
        return {
            "critic_loss": float(critic_loss.mean()),
            "policy_loss": float(policy_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(log_prob.mean()),
            "bc_loss": float(bc_loss.mean()),
            "actor_update_rate": float(actor_updated.mean()),
            "reg_bonus_mean": float(reg_bonus_mean.mean()),
            "reg_bonus_std": float(reg_bonus_std.mean()),
            "beta_lcb": float(beta_lcb),
            "resac_independent_ratio": float(
                self.config.resac_independent_ratio),
        }
