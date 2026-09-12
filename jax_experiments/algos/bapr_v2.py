"""BAPR-v2: causal transition context with a robust residual fallback."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.networks.ensemble_critic import EnsembleCritic
from jax_experiments.networks.residual_policy import (
    ResidualGaussianPolicy,
    advantage_gated_action,
    modewise_update_diagnostics,
    normalized_conservative_q_advantage,
)
from jax_experiments.networks.transition_context import TransitionContextEncoder


def _gravity_latent(task: dict[str, Any], latent_dim: int,
                    log_scale_limit: float,
                    scale_distribution: str = "exp",
                    scale_mode: str = "legacy_exp") -> np.ndarray:
    """Map privileged synthetic task metadata to a bounded oracle latent."""
    out = np.zeros((latent_dim,), dtype=np.float32)
    gravity = task.get("gravity") if isinstance(task, dict) else None
    if gravity is not None:
        value = np.asarray(gravity, dtype=np.float64)
        scale = max(float(abs(np.min(value)) / 9.81), 1e-8)
        use_task_base = scale_mode == "task_distribution"
        log_base = (
            np.log(1.5)
            if use_task_base and scale_distribution == "pow1p5"
            else 1.0
        )
        denominator = max(float(log_scale_limit) * log_base, 1e-6)
        out[0] = np.clip(np.log(scale) / denominator, -1.0, 1.0)
    return out


def _oracle_context_from_task_ids(task_latents, task_ids):
    """Build unit-gated policy contexts from replay task identifiers."""
    task_latents = jnp.asarray(task_latents)
    task_ids = jnp.asarray(task_ids, dtype=jnp.int32)
    indices = jnp.mod(task_ids, task_latents.shape[0])
    latent = task_latents[indices]
    gate = jnp.ones(latent.shape[:-1] + (1,), dtype=latent.dtype)
    return jnp.concatenate([latent, gate], axis=-1)


def _reduce_critic_target(target_q, mode: str):
    if mode == "independent":
        return target_q
    if mode == "min":
        return target_q.min(axis=0)
    raise ValueError(f"unknown critic target mode {mode!r}")


class BAPRv2:
    """Minimal BAPR redesign without BOCD, Q-std gates, or reg latches."""

    uses_transition_context = True
    uses_per_context_alpha = False
    CONTEXT_ROBUST = 0
    CONTEXT_ORACLE = 1
    CONTEXT_LEARNED = 2

    def _make_context_net(self):
        return TransitionContextEncoder(
            self.obs_dim, self.act_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.config.bapr_v2_context_hidden_dim,
            mode=self.context_mode,
            reward_scale=self.config.bapr_v2_reward_scale,
            delta_scale=self.config.bapr_v2_delta_scale,
            min_history=self.config.bapr_v2_min_history,
            gate_error_threshold=self.config.bapr_v2_gate_error_threshold,
            gate_error_scale=self.config.bapr_v2_gate_error_scale,
            error_ema_alpha=self.config.bapr_v2_error_ema_alpha,
            reset_temperature=self.config.bapr_v2_reset_temperature,
            use_fallback=self.config.bapr_v2_use_fallback,
            rngs=self.rngs)

    def _make_policy(self):
        return ResidualGaussianPolicy(
            self.obs_dim, self.act_dim, self.config.hidden_dim,
            self.latent_dim,
            residual_delta=self.config.bapr_v2_residual_delta,
            policy_mode=self.config.bapr_v2_policy_mode,
            num_experts=self.config.bapr_v2_num_experts,
            policy_gate_init=self.config.bapr_v2_policy_gate_init,
            rngs=self.rngs)

    def _make_critic(self):
        return EnsembleCritic(
            self.obs_dim + self.context_dim, self.act_dim,
            self.config.hidden_dim,
            ensemble_size=self.config.ensemble_size, n_layers=3,
            rngs=self.rngs)

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.rngs = nnx.Rngs(seed)
        self.context_mode = str(config.bapr_v2_mode)
        self.policy_context_source = str(
            config.bapr_v2_policy_context_source)
        if self.policy_context_source not in ("stored", "oracle_task"):
            raise ValueError(
                "bapr_v2_policy_context_source must be stored or oracle_task, "
                f"got {self.policy_context_source!r}")
        self.training_schedule = str(config.bapr_v2_training_schedule)
        if self.training_schedule not in (
                "joint", "teacher_student", "constrained_deploy"):
            raise ValueError(
                "bapr_v2_training_schedule must be joint, teacher_student, "
                "or constrained_deploy, "
                f"got {self.training_schedule!r}")
        self.critic_target_mode = str(
            getattr(config, "bapr_v2_critic_target_mode", "independent"))
        if self.critic_target_mode not in ("independent", "min"):
            raise ValueError(
                "bapr_v2_critic_target_mode must be independent or min, "
                f"got {self.critic_target_mode!r}")
        self.latent_dim = int(config.bapr_v2_latent_dim)
        self.context_dim = self.latent_dim + 1
        # Reuse the existing replay vector storage, but it now contains causal
        # context rather than a BOCD posterior.
        self.belief_dim = self.context_dim

        self.context_net = self._make_context_net()
        self.policy = self._make_policy()
        self.critic = self._make_critic()
        self.target_critic = deepcopy(self.critic)

        initial_log_alpha = jnp.asarray(jnp.log(config.alpha))
        self.log_alpha = (
            jnp.full((self.latent_dim + 1,), initial_log_alpha)
            if self.uses_per_context_alpha else initial_log_alpha)
        self.target_entropy = -float(act_dim)
        self.policy_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)
        self.alpha_opt = optax.adam(config.lr)
        self.context_opt = optax.chain(
            optax.clip_by_global_norm(config.bapr_v2_context_grad_clip),
            optax.adam(config.bapr_v2_context_lr))
        self.policy_opt_state = self.policy_opt.init(
            nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(
            nnx.state(self.critic, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)
        self.context_opt_state = self.context_opt.init(
            nnx.state(self.context_net, nnx.Param))

        self.update_count = 0
        self.adaptation_state = self.context_net.initial_state()
        self.oracle_latent = jnp.zeros((self.latent_dim,), dtype=jnp.float32)
        self.task_latents = jnp.zeros(
            (max(int(config.task_num), 1), self.latent_dim),
            dtype=jnp.float32)
        self.safe_task_targets = jnp.zeros(
            (max(int(config.task_num), 1),), dtype=jnp.float32)
        self.safe_task_gains = jnp.zeros_like(self.safe_task_targets)
        self.safe_task_risk_gaps = jnp.zeros_like(self.safe_task_targets)
        self._safe_targets_calibrated = False
        self._replay_reset_requested = False
        self._context_switch_chunks = 0
        self._last_context_error = 0.0
        self._last_context_gate = 0.0
        self._last_advantage = 0.0
        self._last_advantage_gate = 0.0
        self._training_iteration = 0
        self._training_stage = self.training_stage(0)
        self._conditioned_warmstarted = False
        self._v2_rollout_task_changed = False
        self._v2_last_rollout_task_id = None

        policy_state = nnx.state(self.policy, nnx.Param)

        def base_mask(path, value):
            root = str(getattr(path[0], "key", "")) if path else ""
            return jnp.asarray(root.startswith("base_"), dtype=value.dtype)

        def gate_mask(path, value):
            root = str(getattr(path[0], "key", "")) if path else ""
            return jnp.asarray(
                root.startswith("adaptation_gate"), dtype=value.dtype)

        self._policy_base_mask = jax.tree_util.tree_map_with_path(
            base_mask, policy_state)
        self._policy_gate_mask = jax.tree_util.tree_map_with_path(
            gate_mask, policy_state)
        self._policy_adaptive_mask = jax.tree.map(
            lambda base, gate: 1.0 - base - gate,
            self._policy_base_mask, self._policy_gate_mask)
        self._policy_residual_mask = jax.tree.map(
            lambda value: 1.0 - value, self._policy_base_mask)
        self._build_scan_fn()
        self._build_context_update_fn()
        self._build_online_fns()

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    def training_stage(self, iteration: int | None = None) -> str:
        if self.training_schedule == "joint":
            return "joint"
        iteration = (
            self._training_iteration if iteration is None else int(iteration))
        base_end = int(self.config.bapr_v2_base_pretrain_iters)
        teacher_end = base_end + int(self.config.bapr_v2_teacher_iters)
        if iteration < base_end:
            return "robust"
        if iteration < teacher_end:
            return "teacher"
        if self.training_schedule == "constrained_deploy":
            student_end = teacher_end + int(
                self.config.bapr_v2_student_iters)
            if iteration < student_end:
                return "student"
            return "deployment"
        return "student"

    def set_training_iteration(self, iteration: int) -> None:
        previous = self._training_stage
        self._training_iteration = int(iteration)
        self._training_stage = self.training_stage(iteration)
        if (previous == "robust" and self._training_stage == "teacher"
                and bool(self.config.bapr_v2_warmstart_conditioned)
                and not self._conditioned_warmstarted):
            self._conditioned_warmstarted = bool(
                self.policy.warmstart_conditioned_from_base())
        if previous != self._training_stage:
            self.reset_adaptation()
        if previous != "deployment" and self._training_stage == "deployment":
            self._replay_reset_requested = True

    def consume_replay_reset_request(self) -> bool:
        requested = bool(self._replay_reset_requested)
        self._replay_reset_requested = False
        return requested

    def rollout_context_source(self, iteration: int | None = None) -> int:
        stage = self.training_stage(iteration)
        if stage == "robust":
            return self.CONTEXT_ROBUST
        if stage == "teacher":
            return self.CONTEXT_ORACLE
        return self.CONTEXT_LEARNED

    def controller_update_flags(
            self, iteration: int | None = None) -> tuple[bool, bool, bool]:
        stage = self.training_stage(iteration)
        if stage == "robust":
            return True, False, True
        if stage == "teacher":
            return False, True, True
        if stage == "student":
            return False, False, False
        if stage == "deployment":
            return False, True, True
        return True, True, True

    def train_policy_gate(self, iteration: int | None = None) -> bool:
        stage = self.training_stage(iteration)
        if stage == "teacher":
            return not bool(self.config.bapr_v2_freeze_gate_in_teacher)
        return stage in ("joint", "deployment")

    def advantage_gate_active(self, iteration: int | None = None) -> bool:
        if not bool(self.config.bapr_v2_advantage_gate):
            return False
        # Keep the oracle teacher unconstrained so it can establish adaptation
        # headroom. Learned student/deployment actions must beat the robust
        # action under the conservative critic before they are executed.
        return self.training_stage(iteration) in (
            "joint", "student", "deployment")

    def rollout_context(self, iteration: int | None = None):
        return self.context_for_source(self.rollout_context_source(iteration))

    def context_for_source(self, source: int):
        source = int(source)
        if source == self.CONTEXT_ROBUST:
            return jnp.zeros((self.context_dim,), dtype=jnp.float32)
        if source == self.CONTEXT_ORACLE:
            return jnp.concatenate([
                self.oracle_latent,
                jnp.ones((1,), dtype=self.oracle_latent.dtype),
            ])
        if source == self.CONTEXT_LEARNED:
            return self.current_context()
        raise ValueError(f"unsupported BAPR-v2 context source: {source}")

    def set_task_metadata(self, tasks) -> None:
        values = [
            _gravity_latent(task, self.latent_dim,
                            self.config.log_scale_limit,
                            self.config.task_scale_distribution,
                            self.config.bapr_v2_latent_scale_mode)
            for task in tasks
        ]
        if values:
            self.task_latents = jnp.asarray(np.stack(values), dtype=jnp.float32)

    def set_safe_task_targets(self, targets, gains, risk_gaps) -> None:
        expected = int(self.task_latents.shape[0])
        arrays = [
            jnp.asarray(value, dtype=jnp.float32).reshape(-1)
            for value in (targets, gains, risk_gaps)
        ]
        if any(int(value.shape[0]) != expected for value in arrays):
            raise ValueError(
                f"paired calibration expected {expected} task values")
        (self.safe_task_targets, self.safe_task_gains,
         self.safe_task_risk_gaps) = arrays
        self._safe_targets_calibrated = True

    def set_oracle_task_id(self, task_id: int) -> None:
        if len(self.task_latents):
            idx = int(task_id) % int(self.task_latents.shape[0])
            self.oracle_latent = self.task_latents[idx]

    def set_eval_task(self, task) -> None:
        self.oracle_latent = jnp.asarray(
            _gravity_latent(task, self.latent_dim,
                            self.config.log_scale_limit,
                            self.config.task_scale_distribution,
                            self.config.bapr_v2_latent_scale_mode))

    def reset_adaptation(self) -> None:
        self.adaptation_state = self.context_net.initial_state()
        self._last_context_error = 0.0
        self._last_context_gate = 0.0

    def snapshot_adaptation(self):
        return (
            jax.tree.map(lambda x: np.asarray(x), self.adaptation_state),
            np.asarray(self.oracle_latent),
            self._last_context_error,
            self._last_context_gate,
        )

    def restore_adaptation(self, snapshot) -> None:
        state, oracle, error, gate = snapshot
        self.adaptation_state = jax.tree.map(jnp.asarray, state)
        self.oracle_latent = jnp.asarray(oracle)
        self._last_context_error = float(error)
        self._last_context_gate = float(gate)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "adaptation_state": jax.tree.map(
                lambda x: np.asarray(x), self.adaptation_state),
            "oracle_latent": np.asarray(self.oracle_latent),
            "task_latents": np.asarray(self.task_latents),
            "safe_task_targets": np.asarray(self.safe_task_targets),
            "safe_task_gains": np.asarray(self.safe_task_gains),
            "safe_task_risk_gaps": np.asarray(self.safe_task_risk_gaps),
            "safe_targets_calibrated": self._safe_targets_calibrated,
            "last_context_error": self._last_context_error,
            "last_context_gate": self._last_context_gate,
            "last_advantage": self._last_advantage,
            "last_advantage_gate": self._last_advantage_gate,
            "training_iteration": self._training_iteration,
            "conditioned_warmstarted": self._conditioned_warmstarted,
            "last_rollout_task_id": self._v2_last_rollout_task_id,
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if "adaptation_state" in state:
            self.adaptation_state = jax.tree.map(
                jnp.asarray, state["adaptation_state"])
        if "oracle_latent" in state:
            self.oracle_latent = jnp.asarray(state["oracle_latent"])
        if "task_latents" in state:
            self.task_latents = jnp.asarray(state["task_latents"])
        if "safe_task_targets" in state:
            self.safe_task_targets = jnp.asarray(
                state["safe_task_targets"], dtype=jnp.float32)
        if "safe_task_gains" in state:
            self.safe_task_gains = jnp.asarray(
                state["safe_task_gains"], dtype=jnp.float32)
        if "safe_task_risk_gaps" in state:
            self.safe_task_risk_gaps = jnp.asarray(
                state["safe_task_risk_gaps"], dtype=jnp.float32)
        self._safe_targets_calibrated = bool(
            state.get("safe_targets_calibrated", False))
        self._last_context_error = float(
            state.get("last_context_error", 0.0))
        self._last_context_gate = float(
            state.get("last_context_gate", 0.0))
        self._last_advantage = float(state.get("last_advantage", 0.0))
        self._last_advantage_gate = float(
            state.get("last_advantage_gate", 0.0))
        self._training_iteration = int(state.get("training_iteration", 0))
        self._training_stage = self.training_stage(self._training_iteration)
        self._conditioned_warmstarted = bool(
            state.get("conditioned_warmstarted", False))
        self._v2_last_rollout_task_id = state.get("last_rollout_task_id")

    @staticmethod
    def _effective_context(context):
        gate = jnp.clip(context[..., -1:], 0.0, 1.0)
        return jnp.concatenate([context[..., :-1] * gate, gate], axis=-1)

    def _build_scan_fn(self):
        gamma = float(self.config.gamma)
        tau = float(self.config.tau)
        auto_alpha = bool(self.config.auto_alpha)
        freeze_alpha = bool(getattr(
            self.config, "bapr_v2_freeze_alpha", False))
        target_entropy = jnp.asarray(self.target_entropy)
        beta = float(self.config.beta)
        actor_objective = str(self.config.bapr_v2_actor_objective)
        critic_target_mode = self.critic_target_mode
        dropout = float(np.clip(
            self.config.bapr_v2_context_dropout, 0.0, 1.0))
        base_aux_weight = float(self.config.bapr_v2_base_aux_weight)
        action_deviation_weight = float(
            self.config.bapr_v2_action_deviation_weight)
        train_advantage_constraint = bool(
            self.config.bapr_v2_train_advantage_constraint)
        train_advantage_lcb_scale = float(
            self.config.bapr_v2_train_advantage_lcb_scale)
        train_advantage_margin = float(
            self.config.bapr_v2_train_advantage_margin)
        train_advantage_temperature = float(max(
            self.config.bapr_v2_train_advantage_temperature, 1e-6))
        train_advantage_weight = float(
            self.config.bapr_v2_train_advantage_weight)
        train_update_filter = bool(
            self.config.bapr_v2_train_update_filter)
        train_update_tolerance = float(
            self.config.bapr_v2_train_update_tolerance)
        train_update_floor = float(
            self.config.bapr_v2_train_update_floor)
        gate_supervision_weight = float(
            self.config.bapr_v2_gate_supervision_weight)
        unsafe_deviation_weight = float(
            self.config.bapr_v2_unsafe_deviation_weight)
        beta_ood = float(self.config.bapr_v2_beta_ood)
        # Common, normalized shift: unlike legacy per-head raw norms, this
        # cannot manufacture ensemble disagreement. It is zero for MuJoCo jobs.
        reg_weight = float(self.config.bapr_v2_reg_weight)
        reg_norm_ref = float(max(self.config.bapr_v2_reg_norm_ref, 1.0))
        reg_reward_scale = float(self.config.bapr_v2_reward_scale)
        per_context_alpha = bool(self.uses_per_context_alpha)
        latent_dim = int(self.latent_dim)

        def context_membership(context):
            valid = jnp.clip(context[..., -1:], 0.0, 1.0)
            options = context[..., :latent_dim] * valid
            return jnp.concatenate([1.0 - valid, options], axis=-1)

        def alpha_for_context(log_alpha, context):
            if not per_context_alpha:
                return jnp.exp(log_alpha)
            return jnp.sum(
                context_membership(context) * jnp.exp(log_alpha), axis=-1)

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)
        p_opt, c_opt, a_opt = (
            self.policy_opt, self.critic_opt, self.alpha_opt)
        base_mask = self._policy_base_mask
        adaptive_mask = self._policy_adaptive_mask
        gate_mask = self._policy_gate_mask

        @jax.jit
        def scan_update(critic_params, target_params, policy_params,
                        log_alpha, c_opt_state, p_opt_state, a_opt_state,
                        all_obs, all_act, all_rew, all_next_obs, all_done,
                        all_context, all_next_context, all_safe_target,
                        train_base, train_residual, train_gate,
                        train_critic, safe_constraint_active, rng_key):
            def body(carry, data):
                c_p, t_p, p_p, la, c_os, p_os, a_os, key = carry
                (obs, act, rew, nobs, done, context, next_context,
                 safe_target) = data
                key, k1, k2, k3, kdrop = jax.random.split(key, 5)
                alpha = jnp.exp(la)

                if dropout > 0.0:
                    keep = jax.random.bernoulli(
                        kdrop, 1.0 - dropout,
                        shape=context.shape[:-1] + (1,))
                    context = context.at[..., -1:].set(
                        context[..., -1:] * keep)
                    next_context = next_context.at[..., -1:].set(
                        next_context[..., -1:] * keep)
                context = BAPRv2._effective_context(context)
                next_context = BAPRv2._effective_context(next_context)
                obs_aug = jnp.concatenate([obs, context], axis=-1)
                nobs_aug = jnp.concatenate([nobs, next_context], axis=-1)

                def relative_policy_advantage(pp):
                    pm = nnx.merge(gd_policy, pp)
                    tm = nnx.merge(gd_target, t_p)
                    adaptive_action = pm.deterministic(obs, context)
                    base_action = jax.lax.stop_gradient(
                        pm.base_deterministic(obs))
                    return normalized_conservative_q_advantage(
                        tm(obs_aug, adaptive_action),
                        tm(obs_aug, base_action),
                        lcb_scale=train_advantage_lcb_scale,
                    )

                def critic_loss_fn(cp):
                    tm = nnx.merge(gd_target, t_p)
                    pm = nnx.merge(gd_policy, p_p)
                    next_action, next_lp = pm.sample(nobs, k1, next_context)
                    target_q = tm(nobs_aug, next_action)
                    if reg_weight != 0.0:
                        common_norm = tm.compute_reg_norm().mean()
                        common_shift = (
                            reg_weight * reg_reward_scale
                            * jnp.tanh(common_norm / reg_norm_ref))
                        target_q = target_q + common_shift
                    target_q = _reduce_critic_target(
                        target_q, critic_target_mode)
                    target = (
                        rew.squeeze(-1)
                        + gamma * (1.0 - done.squeeze(-1))
                        * (target_q - alpha_for_context(
                            la, next_context) * next_lp))
                    cm = nnx.merge(gd_critic, cp)
                    pred = cm(obs_aug, act)
                    mse = jnp.mean(jnp.square(pred - target))
                    if beta_ood != 0.0:
                        mse = mse + beta_ood * pred.std(axis=0).mean()
                    return mse, pred

                (critic_loss, pred_q), c_grad = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_update, next_c_os = c_opt.update(c_grad, c_os, c_p)
                updated_c_p = optax.apply_updates(c_p, c_update)
                next_c_p = jax.tree.map(
                    lambda new, old: jnp.where(train_critic, new, old),
                    updated_c_p, c_p)
                next_c_os = jax.tree.map(
                    lambda new, old: jnp.where(train_critic, new, old),
                    next_c_os, c_os)

                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, next_c_p)
                    action, lp = pm.sample(obs, k2, context)
                    q = cm(obs_aug, action)
                    actor_q = q.mean(axis=0)
                    if actor_objective == "lcb":
                        actor_q = actor_q + beta * q.std(axis=0)
                    loss = (alpha_for_context(
                        la, context) * lp - actor_q).mean()

                    if base_aux_weight > 0.0:
                        zero_context = jnp.zeros_like(context)
                        base_action, base_lp = pm.sample(obs, k3, zero_context)
                        base_aug = jnp.concatenate(
                            [obs, zero_context], axis=-1)
                        base_q = cm(base_aug, base_action).mean(axis=0)
                        base_loss = (
                            alpha_for_context(
                                la, zero_context) * base_lp - base_q).mean()
                        loss = (
                            loss + base_aux_weight * base_loss
                        ) / (1.0 + base_aux_weight)
                    if action_deviation_weight > 0.0:
                        adaptive_det = pm.deterministic(obs, context)
                        base_det = pm.base_deterministic(obs)
                        deviation = jnp.mean(jnp.square(
                            adaptive_det - base_det))
                        loss = loss + action_deviation_weight * deviation
                    if train_advantage_constraint:
                        relative_advantage = relative_policy_advantage(pp)
                        adaptive_weight = context[..., -1]
                        shortfall = train_advantage_temperature * jax.nn.softplus(
                            (train_advantage_margin - relative_advantage)
                            / train_advantage_temperature)
                        weighted_shortfall = jnp.sum(
                            adaptive_weight * shortfall
                        ) / jnp.maximum(jnp.sum(adaptive_weight), 1.0)
                        weighted_advantage = jnp.sum(
                            adaptive_weight * relative_advantage
                        ) / jnp.maximum(jnp.sum(adaptive_weight), 1.0)
                        loss = (
                            loss
                            + train_advantage_weight * weighted_shortfall)
                    else:
                        weighted_advantage = jnp.asarray(
                            0.0, dtype=loss.dtype)
                        weighted_shortfall = jnp.asarray(
                            0.0, dtype=loss.dtype)
                    safe_target_clipped = jnp.clip(safe_target, 0.0, 1.0)
                    learned_gate = jnp.clip(
                        pm.learned_adaptation_gate(
                            obs, context).squeeze(-1), 1e-5, 1.0 - 1e-5)
                    gate_loss = -jnp.mean(
                        safe_target_clipped * jnp.log(learned_gate)
                        + (1.0 - safe_target_clipped)
                        * jnp.log(1.0 - learned_gate))
                    adaptive_det = pm.deterministic(obs, context)
                    base_det = pm.base_deterministic(obs)
                    unsafe_deviation = jnp.mean(
                        (1.0 - safe_target_clipped)[..., None]
                        * jnp.square(adaptive_det - base_det))
                    constraint = (
                        gate_supervision_weight * gate_loss
                        + unsafe_deviation_weight * unsafe_deviation)
                    loss = loss + safe_constraint_active.astype(
                        loss.dtype) * constraint
                    return loss, (
                        lp,
                        gate_loss,
                        unsafe_deviation,
                        weighted_advantage,
                        weighted_shortfall,
                    )

                (
                    policy_loss,
                    (
                        lp,
                        gate_loss,
                        unsafe_deviation,
                        training_advantage,
                        training_shortfall,
                    ),
                ), p_grad = (
                    jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                )
                p_grad = jax.tree.map(
                    lambda grad, base, adaptive, gate: grad * (
                        train_base.astype(grad.dtype) * base
                        + train_residual.astype(grad.dtype) * adaptive
                        + train_gate.astype(grad.dtype) * gate),
                    p_grad, base_mask, adaptive_mask, gate_mask)
                p_update, next_p_os = p_opt.update(p_grad, p_os, p_p)
                updated_p_p = optax.apply_updates(p_p, p_update)
                next_p_p = jax.tree.map(
                    lambda new, old, base, adaptive, gate: jnp.where(
                        jnp.logical_or(
                            jnp.logical_and(
                                train_base, base.astype(jnp.bool_)),
                            jnp.logical_or(
                                jnp.logical_and(
                                    train_residual,
                                    adaptive.astype(jnp.bool_)),
                                jnp.logical_and(
                                    train_gate, gate.astype(jnp.bool_)))),
                        new, old),
                    updated_p_p, p_p, base_mask, adaptive_mask, gate_mask)
                if train_update_filter:
                    current_advantage = relative_policy_advantage(p_p)
                    candidate_advantage = relative_policy_advantage(next_p_p)
                    (
                        update_accepted,
                        current_mode_advantage,
                        candidate_mode_advantage,
                        candidate_mode_represented,
                        candidate_advantage_min,
                        candidate_advantage_mean,
                        candidate_regression_margin_min,
                        candidate_floor_margin_min,
                        candidate_nonfinite,
                        update_reject_regression,
                        update_reject_floor,
                    ) = modewise_update_diagnostics(
                            current_advantage,
                            candidate_advantage,
                            context,
                            tolerance=train_update_tolerance,
                            floor=train_update_floor,
                        )
                    candidate_mode_regression_margin = (
                        candidate_mode_advantage
                        - (current_mode_advantage - train_update_tolerance)
                    )
                    candidate_mode_floor_margin = (
                        candidate_mode_advantage - train_update_floor)
                    candidate_mode_finite = jnp.logical_and(
                        jnp.isfinite(current_mode_advantage),
                        jnp.isfinite(candidate_mode_advantage),
                    )
                    candidate_mode_active = jnp.logical_and(
                        train_residual, candidate_mode_represented)
                    candidate_mode_reject_nonfinite = jnp.logical_and(
                        candidate_mode_active,
                        jnp.logical_not(candidate_mode_finite),
                    )
                    candidate_mode_reject_regression = jnp.logical_and(
                        candidate_mode_active,
                        jnp.logical_and(
                            candidate_mode_finite,
                            candidate_mode_regression_margin < 0.0,
                        ),
                    )
                    candidate_mode_reject_floor = jnp.logical_and(
                        candidate_mode_active,
                        jnp.logical_and(
                            jnp.isfinite(candidate_mode_advantage),
                            candidate_mode_floor_margin < 0.0,
                        ),
                    )
                    candidate_mode_advantage = jnp.where(
                        candidate_mode_active,
                        candidate_mode_advantage,
                        0.0,
                    )
                    candidate_mode_regression_margin = jnp.where(
                        candidate_mode_active,
                        candidate_mode_regression_margin,
                        0.0,
                    )
                    candidate_mode_floor_margin = jnp.where(
                        candidate_mode_active,
                        candidate_mode_floor_margin,
                        0.0,
                    )
                    candidate_advantage_min = jnp.where(
                        train_residual, candidate_advantage_min, 0.0)
                    candidate_advantage_mean = jnp.where(
                        train_residual, candidate_advantage_mean, 0.0)
                    candidate_regression_margin_min = jnp.where(
                        train_residual,
                        candidate_regression_margin_min,
                        0.0,
                    )
                    candidate_floor_margin_min = jnp.where(
                        train_residual, candidate_floor_margin_min, 0.0)
                    candidate_nonfinite = jnp.logical_and(
                        train_residual, candidate_nonfinite)
                    update_reject_regression = jnp.logical_and(
                        train_residual, update_reject_regression)
                    update_reject_floor = jnp.logical_and(
                        train_residual, update_reject_floor)
                    update_accepted = jnp.logical_or(
                        jnp.logical_not(train_residual), update_accepted)
                    next_p_p = jax.tree.map(
                        lambda new, old: jnp.where(
                            update_accepted, new, old),
                        next_p_p,
                        p_p,
                    )
                    next_p_os = jax.tree.map(
                        lambda new, old: jnp.where(
                            update_accepted, new, old),
                        next_p_os,
                        p_os,
                    )
                else:
                    update_accepted = jnp.asarray(True, dtype=jnp.bool_)
                    candidate_advantage_min = jnp.asarray(
                        0.0, dtype=training_advantage.dtype)
                    candidate_advantage_mean = jnp.asarray(
                        0.0, dtype=training_advantage.dtype)
                    candidate_regression_margin_min = jnp.asarray(
                        0.0, dtype=training_advantage.dtype)
                    candidate_floor_margin_min = jnp.asarray(
                        0.0, dtype=training_advantage.dtype)
                    candidate_nonfinite = jnp.asarray(
                        False, dtype=jnp.bool_)
                    update_reject_regression = jnp.asarray(
                        False, dtype=jnp.bool_)
                    update_reject_floor = jnp.asarray(
                        False, dtype=jnp.bool_)
                    candidate_mode_advantage = jnp.zeros(
                        (latent_dim,), dtype=training_advantage.dtype)
                    candidate_mode_regression_margin = jnp.zeros(
                        (latent_dim,), dtype=training_advantage.dtype)
                    candidate_mode_floor_margin = jnp.zeros(
                        (latent_dim,), dtype=training_advantage.dtype)
                    candidate_mode_active = jnp.zeros(
                        (latent_dim,), dtype=jnp.bool_)
                    candidate_mode_reject_nonfinite = jnp.zeros(
                        (latent_dim,), dtype=jnp.bool_)
                    candidate_mode_reject_regression = jnp.zeros(
                        (latent_dim,), dtype=jnp.bool_)
                    candidate_mode_reject_floor = jnp.zeros(
                        (latent_dim,), dtype=jnp.bool_)

                if per_context_alpha:
                    membership = context_membership(context)
                    entropy_error = jax.lax.stop_gradient(
                        lp + target_entropy)
                    counts = jnp.maximum(membership.sum(axis=0), 1.0)
                    alpha_grad = -jnp.sum(
                        membership * entropy_error[..., None], axis=0
                    ) / counts
                else:
                    alpha_grad = -(lp.mean() + target_entropy)
                alpha_update, next_a_os = a_opt.update(
                    alpha_grad, a_os, la)
                train_controller = jnp.logical_or(
                    jnp.logical_or(train_base, train_residual), train_gate)
                update_alpha = jnp.logical_and(
                    auto_alpha and not freeze_alpha, train_controller)
                next_la = jnp.where(update_alpha, la + alpha_update, la)
                next_a_os = jax.tree.map(
                    lambda n, o: jnp.where(update_alpha, n, o),
                    next_a_os, a_os)
                updated_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1.0 - tau) + cp * tau,
                    t_p, next_c_p)
                next_t_p = jax.tree.map(
                    lambda new, old: jnp.where(train_critic, new, old),
                    updated_t_p, t_p)

                next_carry = (
                    next_c_p, next_t_p, next_p_p, next_la,
                    next_c_os, next_p_os, next_a_os, key)
                metrics = (
                    critic_loss, policy_loss, jnp.exp(next_la),
                    pred_q.mean(), pred_q.std(axis=0).mean(), lp.mean(),
                    context[..., -1].mean(), gate_loss, unsafe_deviation,
                    training_advantage, training_shortfall,
                    update_accepted.astype(jnp.float32),
                    candidate_advantage_min,
                    candidate_advantage_mean,
                    candidate_regression_margin_min,
                    candidate_floor_margin_min,
                    candidate_nonfinite.astype(jnp.float32),
                    update_reject_regression.astype(jnp.float32),
                    update_reject_floor.astype(jnp.float32),
                    candidate_mode_advantage,
                    candidate_mode_regression_margin,
                    candidate_mode_floor_margin,
                    candidate_mode_active.astype(jnp.float32),
                    candidate_mode_reject_nonfinite.astype(jnp.float32),
                    candidate_mode_reject_regression.astype(jnp.float32),
                    candidate_mode_reject_floor.astype(jnp.float32))
                return next_carry, metrics

            init = (
                critic_params, target_params, policy_params, log_alpha,
                c_opt_state, p_opt_state, a_opt_state, rng_key)
            batches = (
                all_obs, all_act, all_rew, all_next_obs, all_done,
                all_context, all_next_context, all_safe_target)
            return jax.lax.scan(body, init, batches)

        self._scan_update = scan_update

    def _build_context_update_fn(self):
        gd_context = nnx.graphdef(self.context_net)
        context_opt = self.context_opt
        mode = self.context_mode
        pred_weight = float(self.config.bapr_v2_predictive_weight)
        if mode == "supervised":
            supervised_weight = float(
                self.config.bapr_v2_supervised_weight)
        elif mode == "hybrid":
            supervised_weight = float(
                self.config.bapr_v2_hybrid_supervised_weight)
        else:
            supervised_weight = 0.0
        temporal_weight = float(self.config.bapr_v2_temporal_weight)
        burnin = int(self.config.bapr_v2_context_burnin)

        @jax.jit
        def context_update(params, opt_state, obs, act, rew, nobs, done,
                           target_latents):
            def loss_fn(cp):
                model = nnx.merge(gd_context, cp)

                def one_chunk(chunk):
                    c_obs, c_act, c_rew, c_nobs, c_done = chunk

                    def body(state, transition):
                        o, a, r, no, d = transition
                        context = model.policy_context(
                            state, jnp.zeros((model.latent_dim,)))
                        next_state, error, _, _ = model.observe(
                            state, o, a, r, no, d, enable_reset=False)
                        return next_state, (context[:-1], error)

                    _, outputs = jax.lax.scan(
                        body, model.initial_state(),
                        (c_obs, c_act, c_rew, c_nobs, c_done))
                    return outputs

                latents, errors = jax.vmap(one_chunk)(
                    (obs, act, rew, nobs, done))
                tail = latents[:, burnin:, :]
                target = target_latents[:, burnin:, :]
                supervised = jnp.mean(jnp.square(tail - target))
                predictive = errors.mean()
                latent_delta = tail[:, 1:, :] - tail[:, :-1, :]
                target_delta = target[:, 1:, :] - target[:, :-1, :]
                same_task = jnp.all(
                    jnp.abs(target_delta) < 1e-6, axis=-1, keepdims=True)
                temporal = jnp.sum(
                    jnp.square(latent_delta) * same_task
                ) / jnp.maximum(jnp.sum(same_task) * tail.shape[-1], 1.0)
                loss = (
                    pred_weight * predictive
                    + supervised_weight * supervised
                    + temporal_weight * temporal)
                return loss, (predictive, supervised, temporal,
                              tail.std(axis=(0, 1)).max())

            (loss, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(params)
            updates, next_opt_state = context_opt.update(
                grads, opt_state, params)
            next_params = optax.apply_updates(params, updates)
            return next_params, next_opt_state, (loss,) + aux

        self._context_update = context_update

    def _build_online_fns(self):
        gd_policy = nnx.graphdef(self.policy)
        gd_context = nnx.graphdef(self.context_net)
        gd_critic = nnx.graphdef(self.critic)
        advantage_margin = float(self.config.bapr_v2_advantage_margin)
        advantage_lcb = float(self.config.bapr_v2_advantage_lcb_scale)

        @jax.jit
        def deterministic_action(policy_params, critic_params, context, obs,
                                 gate_enabled):
            policy = nnx.merge(gd_policy, policy_params)
            critic = nnx.merge(gd_critic, critic_params)
            effective = BAPRv2._effective_context(context)
            action, advantage, gate = advantage_gated_action(
                policy, critic, obs[None], effective[None],
                enabled=gate_enabled, margin=advantage_margin,
                lcb_scale=advantage_lcb)
            return action[0], context, advantage[0], gate[0]

        @jax.jit
        def stochastic_action(policy_params, critic_params, context, obs, key,
                              gate_enabled):
            policy = nnx.merge(gd_policy, policy_params)
            critic = nnx.merge(gd_critic, critic_params)
            effective = BAPRv2._effective_context(context)
            action, advantage, gate = advantage_gated_action(
                policy, critic, obs[None], effective[None], key=key,
                enabled=gate_enabled, margin=advantage_margin,
                lcb_scale=advantage_lcb)
            return action[0], context, advantage[0], gate[0]

        @jax.jit
        def policy_diagnostics(policy_params, obs, context):
            policy = nnx.merge(gd_policy, policy_params)
            context = BAPRv2._effective_context(context)
            base_action = policy.base_deterministic(obs)
            context_action = policy.deterministic(obs, context)
            shuffled = jnp.concatenate([
                jnp.roll(context[..., :-1], 1, axis=0),
                context[..., -1:],
            ], axis=-1)
            shuffled_action = policy.deterministic(obs, shuffled)
            return (
                jnp.mean(jnp.abs(context_action - base_action)),
                jnp.mean(jnp.abs(context_action - shuffled_action)),
                jnp.mean(policy.adaptation_strength(obs, context)),
            )

        @jax.jit
        def observe(context_params, state, obs, action, reward, next_obs, done):
            model = nnx.merge(gd_context, context_params)
            return model.observe(
                state, obs, action, reward, next_obs, done,
                enable_reset=True)

        self._deterministic_action = deterministic_action
        self._stochastic_action = stochastic_action
        self._policy_diagnostics = policy_diagnostics
        self._online_observe = observe

    def current_context(self):
        return self.context_net.policy_context(
            self.adaptation_state, self.oracle_latent)

    def select_action(self, obs, deterministic=False, *,
                      context_source=None, advantage_enabled=None):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        p_state = nnx.state(self.policy, nnx.Param)
        q_state = nnx.state(self.critic, nnx.Param)
        source = (
            self.rollout_context_source()
            if context_source is None else int(context_source))
        context = self.context_for_source(source)
        enabled = (
            self.advantage_gate_active()
            if advantage_enabled is None else bool(advantage_enabled))
        gate_enabled = jnp.asarray(
            enabled, dtype=jnp.bool_)
        if deterministic:
            action, context, advantage, gate = self._deterministic_action(
                p_state, q_state, context, obs, gate_enabled)
        else:
            action, context, advantage, gate = self._stochastic_action(
                p_state, q_state, context, obs, self.rngs.params(),
                gate_enabled)
        self._last_context_gate = float(context[-1])
        self._last_advantage = float(advantage)
        self._last_advantage_gate = float(gate)
        return np.asarray(action)

    def observe_transition(self, obs, action, reward, next_obs, done):
        c_state = nnx.state(self.context_net, nnx.Param)
        state, error, _, _ = self._online_observe(
            c_state, self.adaptation_state,
            jnp.asarray(obs, dtype=jnp.float32),
            jnp.asarray(action, dtype=jnp.float32),
            jnp.asarray(reward, dtype=jnp.float32),
            jnp.asarray(next_obs, dtype=jnp.float32),
            jnp.asarray(done, dtype=jnp.float32))
        self.adaptation_state = state
        self._last_context_error = float(error)
        self._last_context_gate = float(self.current_context()[-1])
        return np.asarray(self.current_context())

    def _context_training_batch(self, recent_rollout):
        length = int(self.config.bapr_v2_context_length)
        chunks = int(self.config.bapr_v2_context_chunks)
        n_steps = int(recent_rollout["obs"].shape[0])
        if n_steps < length:
            return None
        max_start = n_steps - length
        uniform_count = max(1, chunks // 2)
        starts = list(np.linspace(
            0, max_start, uniform_count, dtype=np.int32))
        task_ids_np = np.asarray(recent_rollout["task_id"], dtype=np.int32)
        boundaries = np.flatnonzero(
            task_ids_np[1:] != task_ids_np[:-1]) + 1
        boundary_slots = chunks - len(starts)
        if boundary_slots > 0 and len(boundaries):
            selected = boundaries[np.linspace(
                0, len(boundaries) - 1,
                min(boundary_slots, len(boundaries)), dtype=np.int32)]
            starts.extend(
                int(np.clip(boundary - int(
                    self.config.bapr_v2_context_burnin), 0, max_start))
                for boundary in selected)
        if len(starts) < chunks:
            filler = np.linspace(0, max_start, chunks, dtype=np.int32)
            starts.extend(int(value) for value in filler[:chunks - len(starts)])
        starts = np.asarray(starts[:chunks], dtype=np.int32)
        index = jnp.asarray(starts)[:, None] + jnp.arange(length)[None, :]
        transitions = tuple(
            jnp.asarray(recent_rollout[key])[index]
            for key in ("obs", "act", "rew", "next_obs", "done"))
        task_ids = jnp.asarray(recent_rollout["task_id"])[index]
        switch_chunks = int(np.sum(np.any(
            task_ids_np[starts[:, None] + np.arange(length)[None, :]][:, 1:]
            != task_ids_np[starts[:, None] + np.arange(length)[None, :]][:, :-1],
            axis=1)))
        return transitions + (task_ids, switch_chunks)

    def multi_update(self, stacked_batch: dict, current_iter=0,
                     recent_rollout=None, **kwargs):
        self.set_training_iteration(current_iter)
        stage = self.training_stage(current_iter)
        context_metrics = (0.0, 0.0, 0.0, 0.0, 0.0)
        if (recent_rollout is not None
                and self.context_mode in ("supervised", "hybrid")
                and stage != "robust"):
            chunks = self._context_training_batch(recent_rollout)
            if chunks is not None:
                *transition_chunks, task_ids, switch_chunks = chunks
                target = self.task_latents[jnp.mod(
                    task_ids, int(self.task_latents.shape[0]))]
                params = nnx.state(self.context_net, nnx.Param)
                params, self.context_opt_state, metrics = self._context_update(
                    params, self.context_opt_state,
                    *transition_chunks, target)
                nnx.update(self.context_net, params)
                extra_metrics = metrics[5:]
                diagnostics_hook = getattr(
                    self, "_record_context_update_diagnostics", None)
                if diagnostics_hook is not None and extra_metrics:
                    diagnostics_hook(*extra_metrics)
                context_metrics = tuple(float(x) for x in metrics[:5])
                self._context_switch_chunks = int(switch_chunks)

        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        nobs = stacked_batch["next_obs"]
        done = stacked_batch["done"]
        context = stacked_batch.get("belief")
        if context is None:
            context = jnp.zeros(
                obs.shape[:-1] + (self.context_dim,), dtype=obs.dtype)
        next_context = stacked_batch.get("next_belief", context)
        if stage == "robust":
            context = jnp.zeros_like(context)
            next_context = jnp.zeros_like(next_context)
        elif (self.policy_context_source == "oracle_task"
              and stage == "teacher"):
            task_ids = stacked_batch.get("task_id")
            if task_ids is None:
                raise ValueError(
                    "oracle_task policy context requires replay task_id")
            context = _oracle_context_from_task_ids(
                self.task_latents, task_ids)
            # Training rollouts hold physics fixed for the whole scan, so a
            # replay transition and its successor share the same task id.
            next_context = context

        task_ids = stacked_batch.get("task_id")
        if task_ids is None:
            safe_target = jnp.zeros(obs.shape[:-1], dtype=obs.dtype)
        else:
            safe_target = self.safe_task_targets[jnp.mod(
                task_ids, int(self.safe_task_targets.shape[0]))]
        safe_constraint_active = (
            stage == "deployment" and self._safe_targets_calibrated
            and (float(self.config.bapr_v2_gate_supervision_weight) > 0.0
                 or float(self.config.bapr_v2_unsafe_deviation_weight) > 0.0)
        )
        if (stage == "deployment"
                and int(self.config.bapr_v2_paired_calibration_episodes) > 0
                and not self._safe_targets_calibrated):
            raise RuntimeError(
                "deployment started before paired safety calibration")

        train_base, train_residual, train_critic = (
            self.controller_update_flags(current_iter))
        train_gate = self.train_policy_gate(current_iter)

        final, metrics = self._scan_update(
            nnx.state(self.critic, nnx.Param),
            nnx.state(self.target_critic, nnx.Param),
            nnx.state(self.policy, nnx.Param),
            self.log_alpha,
            self.critic_opt_state,
            self.policy_opt_state,
            self.alpha_opt_state,
            obs, act, rew, nobs, done, context, next_context, safe_target,
            jnp.asarray(train_base, dtype=jnp.bool_),
            jnp.asarray(train_residual, dtype=jnp.bool_),
            jnp.asarray(train_gate, dtype=jnp.bool_),
            jnp.asarray(train_critic, dtype=jnp.bool_),
            jnp.asarray(safe_constraint_active, dtype=jnp.bool_),
            self.rngs.params())
        (new_critic, new_target, new_policy, self.log_alpha,
         self.critic_opt_state, self.policy_opt_state,
         self.alpha_opt_state, _) = final
        nnx.update(self.critic, new_critic)
        nnx.update(self.target_critic, new_target)
        nnx.update(self.policy, new_policy)
        self.update_count += int(obs.shape[0])

        (context_effect, latent_sensitivity,
         adaptation_strength) = self._policy_diagnostics(
             nnx.state(self.policy, nnx.Param), obs[-1],
             context[-1])

        (
            critic_loss,
            policy_loss,
            alpha,
            q_mean,
            q_std,
            log_prob,
            gate,
            gate_supervision_loss,
            unsafe_deviation,
            training_advantage,
            training_shortfall,
            training_update_accepted,
            candidate_advantage_min,
            candidate_advantage_mean,
            candidate_regression_margin_min,
            candidate_floor_margin_min,
            candidate_nonfinite,
            update_reject_regression,
            update_reject_floor,
            candidate_mode_advantage,
            candidate_mode_regression_margin,
            candidate_mode_floor_margin,
            candidate_mode_active,
            candidate_mode_reject_nonfinite,
            candidate_mode_reject_regression,
            candidate_mode_reject_floor,
        ) = metrics
        (context_loss, predictive_loss, supervised_loss,
         temporal_loss, latent_std) = context_metrics
        result = {
            "critic_loss": float(critic_loss.mean()),
            "policy_loss": float(policy_loss.mean()),
            "alpha": float(jnp.mean(alpha[-1])),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(log_prob.mean()),
            "v2_context_loss": context_loss,
            "v2_predictive_loss": predictive_loss,
            "v2_supervised_loss": supervised_loss,
            "v2_temporal_loss": temporal_loss,
            "v2_latent_std": latent_std,
            "v2_context_gate": float(gate.mean()),
            "v2_policy_context_effect": float(context_effect),
            "v2_policy_latent_sensitivity": float(latent_sensitivity),
            "v2_policy_adaptation_strength": float(adaptation_strength),
            "v2_gate_supervision_loss": float(
                gate_supervision_loss.mean()),
            "v2_unsafe_deviation": float(unsafe_deviation.mean()),
            "v2_train_advantage_lcb": float(training_advantage.mean()),
            "v2_train_advantage_shortfall": float(
                training_shortfall.mean()),
            "v2_train_update_accept_rate": float(
                training_update_accepted.mean()),
            "v2_train_candidate_advantage_min": float(
                candidate_advantage_min.mean()),
            "v2_train_candidate_advantage_mean": float(
                candidate_advantage_mean.mean()),
            "v2_train_candidate_regression_margin_min": float(
                candidate_regression_margin_min.mean()),
            "v2_train_candidate_floor_margin_min": float(
                candidate_floor_margin_min.mean()),
            "v2_train_candidate_nonfinite_rate": float(
                candidate_nonfinite.mean()),
            "v2_train_update_reject_regression_rate": float(
                update_reject_regression.mean()),
            "v2_train_update_reject_floor_rate": float(
                update_reject_floor.mean()),
            "v2_safe_target_mean": float(jnp.mean(
                self.safe_task_targets)),
            "v2_safe_target_std": float(jnp.std(
                self.safe_task_targets)),
            "v2_paired_gain_mean": float(jnp.mean(
                self.safe_task_gains)),
            "v2_paired_risk_gap_mean": float(jnp.mean(
                self.safe_task_risk_gaps)),
            "v2_safe_targets_calibrated": float(
                self._safe_targets_calibrated),
            "v2_context_switch_chunks": float(
                self._context_switch_chunks),
            "v2_task_latent_std": float(jnp.std(
                self.task_latents[..., 0])),
            "v2_teacher_policy_context": float(stage == "teacher"),
            "v2_training_stage": float({
                "joint": 0, "robust": 1, "teacher": 2, "student": 3,
                "deployment": 4, "inference": 5, "adaptation": 6,
            }.get(stage, -1)),
            "v2_train_base": float(train_base),
            "v2_train_residual": float(train_residual),
            "v2_train_gate": float(train_gate),
            "v2_train_critic": float(train_critic),
            "v2_conditioned_warmstarted": float(
                self._conditioned_warmstarted),
            "v2_online_error": float(self._last_context_error),
            "v2_live_gate": float(self._last_context_gate),
            "v2_advantage": float(self._last_advantage),
            "v2_advantage_gate": float(self._last_advantage_gate),
            "v2_rollout_task_changed": float(
                self._v2_rollout_task_changed),
        }
        mode_counts = jnp.maximum(
            candidate_mode_active.sum(axis=0), 1.0)
        for mode in range(self.latent_dim):
            result[
                f"v2_train_candidate_mode_{mode}_advantage"
            ] = float(
                candidate_mode_advantage[:, mode].sum()
                / mode_counts[mode])
            result[
                f"v2_train_candidate_mode_{mode}_regression_margin"
            ] = float(
                candidate_mode_regression_margin[:, mode].sum()
                / mode_counts[mode])
            result[
                f"v2_train_candidate_mode_{mode}_floor_margin"
            ] = float(
                candidate_mode_floor_margin[:, mode].sum()
                / mode_counts[mode])
            result[
                f"v2_train_candidate_mode_{mode}_represented_rate"
            ] = float(candidate_mode_active[:, mode].mean())
            result[
                f"v2_train_candidate_mode_{mode}_nonfinite_rate"
            ] = float(candidate_mode_reject_nonfinite[:, mode].mean())
            result[
                f"v2_train_candidate_mode_{mode}_reject_regression_rate"
            ] = float(candidate_mode_reject_regression[:, mode].mean())
            result[
                f"v2_train_candidate_mode_{mode}_reject_floor_rate"
            ] = float(candidate_mode_reject_floor[:, mode].mean())
        return result
