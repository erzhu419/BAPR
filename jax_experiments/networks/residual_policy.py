"""Gaussian policy with a robust base and a bounded context residual."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.policy import LOG_STD_MAX, LOG_STD_MIN


def _module_list(layers):
    list_cls = getattr(nnx, "List", None)
    return list_cls(layers) if list_cls is not None else layers


def conservative_q_advantage(q_adaptive, q_base, lcb_scale: float = 1.0):
    """Lower-confidence estimate of the adaptive action's Q advantage."""
    delta = q_adaptive - q_base
    scale = jnp.asarray(lcb_scale, dtype=delta.dtype)
    centered = delta - delta.mean(axis=0, keepdims=True)
    variance = jnp.mean(jnp.square(centered), axis=0)
    epsilon = jnp.asarray(1e-6, dtype=delta.dtype)
    # Keep the exact zero-advantage anchor while avoiding sqrt'(0) = inf.
    stable_std = jnp.sqrt(variance + epsilon) - jnp.sqrt(epsilon)
    return delta.mean(axis=0) - scale * stable_std


def normalized_conservative_q_advantage(
    q_adaptive,
    q_base,
    lcb_scale: float = 1.0,
):
    """Scale a conservative action advantage by the base Q magnitude."""
    advantage = conservative_q_advantage(
        q_adaptive, q_base, lcb_scale=lcb_scale)
    q_scale = jnp.maximum(jnp.mean(jnp.abs(q_base), axis=0), 1.0)
    return advantage / jax.lax.stop_gradient(q_scale)


def modewise_update_acceptance(
    current_advantage,
    candidate_advantage,
    context,
    *,
    tolerance: float,
    floor: float,
):
    """Reject an actor update that degrades any represented adaptive mode."""
    diagnostics = modewise_update_diagnostics(
        current_advantage,
        candidate_advantage,
        context,
        tolerance=tolerance,
        floor=floor,
    )
    return diagnostics[:4]


def modewise_update_diagnostics(
    current_advantage,
    candidate_advantage,
    context,
    *,
    tolerance: float,
    floor: float,
):
    """Return acceptance and the mode-level margins behind that decision."""
    context = jnp.asarray(context)
    valid = jnp.clip(context[..., -1:], 0.0, 1.0)
    membership = jnp.clip(context[..., :-1], 0.0) * valid
    counts = jnp.sum(membership, axis=0)
    denominator = jnp.maximum(counts, 1.0)

    def mode_mean(value):
        expanded = value[..., None]
        # A non-finite sample must only contaminate modes it belongs to.
        weighted = membership * jnp.where(membership > 0.0, expanded, 0.0)
        return jnp.sum(weighted, axis=0) / denominator

    current = mode_mean(current_advantage)
    candidate = mode_mean(candidate_advantage)
    represented = counts > 0.0
    tolerance = jnp.asarray(tolerance, dtype=candidate.dtype)
    floor = jnp.asarray(floor, dtype=candidate.dtype)
    finite = jnp.logical_and(
        jnp.isfinite(current),
        jnp.isfinite(candidate),
    )
    regression_margin = candidate - (current - tolerance)
    floor_margin = candidate - floor
    safe = jnp.logical_and(
        finite,
        jnp.logical_and(
            regression_margin >= 0.0,
            floor_margin >= 0.0,
        ),
    )
    accepted = jnp.all(jnp.logical_or(jnp.logical_not(represented), safe))

    def represented_min(value):
        mask = jnp.logical_and(represented, jnp.isfinite(value))
        return jnp.where(
            jnp.any(mask),
            jnp.min(jnp.where(mask, value, jnp.inf)),
            jnp.asarray(jnp.nan, dtype=value.dtype),
        )

    finite_candidate = jnp.logical_and(
        represented, jnp.isfinite(candidate))
    candidate_mean = jnp.where(
        jnp.any(finite_candidate),
        jnp.sum(jnp.where(finite_candidate, candidate, 0.0))
        / jnp.maximum(jnp.sum(finite_candidate), 1),
        jnp.asarray(jnp.nan, dtype=candidate.dtype),
    )
    reject_nonfinite = jnp.any(
        jnp.logical_and(represented, jnp.logical_not(finite)))
    reject_regression = jnp.any(jnp.logical_and(
        represented,
        jnp.logical_and(finite, regression_margin < 0.0),
    ))
    reject_floor = jnp.any(jnp.logical_and(
        represented,
        jnp.logical_and(jnp.isfinite(candidate), floor_margin < 0.0),
    ))
    return (
        accepted,
        current,
        candidate,
        represented,
        represented_min(candidate),
        candidate_mean,
        represented_min(regression_margin),
        represented_min(floor_margin),
        reject_nonfinite,
        reject_regression,
        reject_floor,
    )


def advantage_gated_action(policy, critic, obs, context, *, key=None,
                           enabled=True, margin: float = 0.0,
                           lcb_scale: float = 1.0):
    """Fall back to the robust action unless its residual has positive LCB."""
    adaptive_det = policy.deterministic(obs, context)
    base_det = policy.base_deterministic(obs)
    obs_context = jnp.concatenate([obs, context], axis=-1)
    score = conservative_q_advantage(
        critic(obs_context, adaptive_det),
        critic(obs_context, base_det),
        lcb_scale=lcb_scale,
    )
    gate = jnp.logical_or(
        jnp.logical_not(jnp.asarray(enabled, dtype=jnp.bool_)),
        score > jnp.asarray(margin, dtype=score.dtype),
    )
    if key is None:
        adaptive_action, base_action = adaptive_det, base_det
    else:
        adaptive_action, base_action, _, _ = policy.sample_pair(
            obs, key, context)
    action = jnp.where(gate[..., None], adaptive_action, base_action)
    return action, score, gate.astype(obs.dtype)


class ResidualGaussianPolicy(nnx.Module):
    """SAC actor whose context path can never remove the base controller.

    The final context coordinate is a confidence gate in ``[0, 1]``.  The
    remaining coordinates are the inferred environment latent.  With gate=0
    this is exactly the context-free base actor; with gate>0 a bounded residual
    can modify its pre-tanh mean.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int,
                 latent_dim: int, residual_delta: float = 0.25,
                 policy_mode: str = "residual", num_experts: int = 5,
                 policy_gate_init: float = 0.0, *,
                 rngs: nnx.Rngs):
        if policy_mode not in (
                "residual", "direct", "gated_direct", "expert",
                "categorical_expert"):
            raise ValueError(
                f"unsupported BAPR-v2 policy mode: {policy_mode}")
        self.latent_dim = int(latent_dim)
        self.ep_dim = self.latent_dim + 1
        self.residual_delta = float(residual_delta)
        self.act_dim = int(act_dim)
        self.policy_mode = str(policy_mode)
        self.num_experts = int(max(num_experts, 2))
        self.policy_gate_init = float(policy_gate_init)

        base_layers = []
        in_dim = obs_dim
        for _ in range(2):
            base_layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        self.base_layers = _module_list(base_layers)
        self.base_mean = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.base_log_std = nnx.Linear(hidden_dim, act_dim, rngs=rngs)

        if (self.policy_mode == "categorical_expert"
                and self.num_experts != self.latent_dim):
            raise ValueError(
                "categorical_expert requires num_experts == latent_dim, "
                f"got {self.num_experts} and {self.latent_dim}")

        if self.policy_mode == "categorical_expert":
            expert_layers = []
            expert_means = []
            expert_log_stds = []
            for _ in range(self.num_experts):
                layers = []
                in_dim = obs_dim
                for _ in range(2):
                    layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
                    in_dim = hidden_dim
                expert_layers.append(_module_list(layers))
                expert_means.append(nnx.Linear(
                    hidden_dim, act_dim, rngs=rngs))
                expert_log_stds.append(nnx.Linear(
                    hidden_dim, act_dim, rngs=rngs))
            self.expert_layers = _module_list(expert_layers)
            self.expert_means = _module_list(expert_means)
            self.expert_log_stds = _module_list(expert_log_stds)
            return

        residual_layers = []
        in_dim = (
            obs_dim if self.policy_mode == "expert"
            else obs_dim + self.latent_dim
        )
        for _ in range(2):
            residual_layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        self.residual_layers = _module_list(residual_layers)
        adaptive_outputs = (
            self.num_experts * act_dim
            if self.policy_mode == "expert" else act_dim
        )
        self.residual_mean = nnx.Linear(
            hidden_dim, adaptive_outputs, rngs=rngs)
        if self.policy_mode in ("direct", "gated_direct", "expert"):
            self.conditioned_log_std = nnx.Linear(
                hidden_dim, adaptive_outputs, rngs=rngs)
        if self.policy_mode == "gated_direct":
            self.adaptation_gate = nnx.Linear(hidden_dim, 1, rngs=rngs)
            self.adaptation_gate.kernel.value = jnp.zeros_like(
                self.adaptation_gate.kernel.value)
            self.adaptation_gate.bias.value = jnp.full_like(
                self.adaptation_gate.bias.value, self.policy_gate_init)

    def _split_context(self, obs, context):
        if context is None:
            z = jnp.zeros(obs.shape[:-1] + (self.latent_dim,), obs.dtype)
            gate = jnp.zeros(obs.shape[:-1] + (1,), obs.dtype)
            return z, gate
        z = context[..., :self.latent_dim]
        gate = jnp.clip(context[..., self.latent_dim:self.latent_dim + 1],
                        0.0, 1.0)
        return z, gate

    def __call__(self, obs, ep_tensor=None):
        base = obs
        for layer in self.base_layers:
            base = nnx.relu(layer(base))
        base_mean = self.base_mean(base)
        log_std = jnp.clip(
            self.base_log_std(base), LOG_STD_MIN, LOG_STD_MAX)

        z, gate = self._split_context(obs, ep_tensor)
        if self.policy_mode == "categorical_expert":
            expert_means = []
            expert_log_stds = []
            for layers, mean_head, log_std_head in zip(
                    self.expert_layers, self.expert_means,
                    self.expert_log_stds):
                hidden = obs
                for layer in layers:
                    hidden = nnx.relu(layer(hidden))
                expert_means.append(mean_head(hidden))
                expert_log_stds.append(log_std_head(hidden))
            adaptive_mean = jnp.stack(expert_means, axis=-2)
            adaptive_log_std = jnp.stack(expert_log_stds, axis=-2)
            nonnegative = jnp.clip(z, 0.0)
            total = jnp.sum(nonnegative, axis=-1, keepdims=True)
            normalized = nonnegative / jnp.maximum(total, 1e-8)
            hard_fallback = jax.nn.one_hot(
                jnp.argmax(z, axis=-1), self.num_experts,
                dtype=adaptive_mean.dtype)
            weights = jnp.where(total > 1e-8, normalized, hard_fallback)
            adaptive_mean = jnp.sum(
                adaptive_mean * weights[..., :, None], axis=-2)
            adaptive_log_std = jnp.sum(
                adaptive_log_std * weights[..., :, None], axis=-2)
            adaptive_log_std = jnp.clip(
                adaptive_log_std, LOG_STD_MIN, LOG_STD_MAX)
            mean = base_mean + gate * (adaptive_mean - base_mean)
            log_std = log_std + gate * (adaptive_log_std - log_std)
            return mean, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

        residual = (
            obs if self.policy_mode == "expert"
            else jnp.concatenate([obs, z], axis=-1)
        )
        for layer in self.residual_layers:
            residual = nnx.relu(layer(residual))
        adaptive_mean = self.residual_mean(residual)
        if self.policy_mode == "residual":
            delta = self.residual_delta * jnp.tanh(adaptive_mean)
            return base_mean + gate * delta, log_std

        adaptive_log_std = self.conditioned_log_std(residual)
        if self.policy_mode == "expert":
            shape = adaptive_mean.shape[:-1] + (
                self.num_experts, self.act_dim)
            adaptive_mean = adaptive_mean.reshape(shape)
            adaptive_log_std = adaptive_log_std.reshape(shape)
            centers = jnp.linspace(
                -1.0, 1.0, self.num_experts, dtype=z.dtype)
            expert_index = jnp.argmin(
                jnp.abs(z[..., :1] - centers), axis=-1)
            weights = jax.nn.one_hot(
                expert_index, self.num_experts, dtype=adaptive_mean.dtype)
            adaptive_mean = jnp.sum(
                adaptive_mean * weights[..., :, None], axis=-2)
            adaptive_log_std = jnp.sum(
                adaptive_log_std * weights[..., :, None], axis=-2)

        if self.policy_mode == "gated_direct":
            learned_gate = jax.nn.sigmoid(self.adaptation_gate(residual))
            gate = gate * learned_gate

        adaptive_log_std = jnp.clip(
            adaptive_log_std, LOG_STD_MIN, LOG_STD_MAX)
        mean = base_mean + gate * (adaptive_mean - base_mean)
        log_std = log_std + gate * (adaptive_log_std - log_std)
        return mean, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs, key, ep_tensor=None):
        mean, log_std = self(obs, ep_tensor)
        return self._sample_distribution(mean, log_std, key)

    @staticmethod
    def _sample_distribution(mean, log_std, key, noise=None):
        std = jnp.exp(log_std)
        if noise is None:
            noise = jax.random.normal(key, mean.shape)
        pre_tanh = mean + std * noise
        action = jnp.tanh(pre_tanh)
        log_prob = -0.5 * (
            ((pre_tanh - mean) / std) ** 2
            + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob = log_prob.sum(axis=-1)
        log_prob -= jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
        return action, log_prob

    def sample_pair(self, obs, key, ep_tensor):
        """Sample adaptive and robust actions with shared exploration noise."""
        adaptive_mean, adaptive_log_std = self(obs, ep_tensor)
        base_mean, base_log_std = self(obs, None)
        noise = jax.random.normal(key, adaptive_mean.shape)
        adaptive = self._sample_distribution(
            adaptive_mean, adaptive_log_std, key, noise=noise)
        base = self._sample_distribution(
            base_mean, base_log_std, key, noise=noise)
        return adaptive[0], base[0], adaptive[1], base[1]

    def deterministic(self, obs, ep_tensor=None):
        mean, _ = self(obs, ep_tensor)
        return jnp.tanh(mean)

    def base_deterministic(self, obs):
        return self.deterministic(obs, None)

    def adaptation_strength(self, obs, context):
        """Continuous amount of conditioned control used by the policy."""
        _, history_gate = self._split_context(obs, context)
        return history_gate * self.learned_adaptation_gate(obs, context)

    def learned_adaptation_gate(self, obs, context):
        """Policy-owned gate, excluding causal-history confidence."""
        z, _ = self._split_context(obs, context)
        if self.policy_mode != "gated_direct":
            return jnp.ones(obs.shape[:-1] + (1,), dtype=obs.dtype)
        hidden = jnp.concatenate([obs, z], axis=-1)
        for layer in self.residual_layers:
            hidden = nnx.relu(layer(hidden))
        return jax.nn.sigmoid(self.adaptation_gate(hidden))

    def warmstart_conditioned_from_base(self) -> bool:
        """Make a direct conditioned branch exactly reproduce the robust base."""
        if self.policy_mode == "categorical_expert":
            for layers, mean_head, log_std_head in zip(
                    self.expert_layers, self.expert_means,
                    self.expert_log_stds):
                for base_layer, expert_layer in zip(
                        self.base_layers, layers):
                    expert_layer.kernel.value = jnp.asarray(
                        base_layer.kernel.value)
                    expert_layer.bias.value = jnp.asarray(
                        base_layer.bias.value)
                mean_head.kernel.value = jnp.asarray(
                    self.base_mean.kernel.value)
                mean_head.bias.value = jnp.asarray(
                    self.base_mean.bias.value)
                log_std_head.kernel.value = jnp.asarray(
                    self.base_log_std.kernel.value)
                log_std_head.bias.value = jnp.asarray(
                    self.base_log_std.bias.value)
            return True
        if self.policy_mode not in ("direct", "gated_direct"):
            return False
        for base_layer, conditioned_layer in zip(
                self.base_layers, self.residual_layers):
            base_kernel = base_layer.kernel.value
            conditioned_kernel = jnp.zeros_like(
                conditioned_layer.kernel.value)
            conditioned_kernel = conditioned_kernel.at[
                :base_kernel.shape[0], :base_kernel.shape[1]
            ].set(base_kernel)
            conditioned_layer.kernel.value = conditioned_kernel
            conditioned_layer.bias.value = jnp.asarray(base_layer.bias.value)
        self.residual_mean.kernel.value = jnp.asarray(
            self.base_mean.kernel.value)
        self.residual_mean.bias.value = jnp.asarray(self.base_mean.bias.value)
        self.conditioned_log_std.kernel.value = jnp.asarray(
            self.base_log_std.kernel.value)
        self.conditioned_log_std.bias.value = jnp.asarray(
            self.base_log_std.bias.value)
        if self.policy_mode == "gated_direct":
            self.adaptation_gate.kernel.value = jnp.zeros_like(
                self.adaptation_gate.kernel.value)
            self.adaptation_gate.bias.value = jnp.full_like(
                self.adaptation_gate.bias.value, self.policy_gate_init)
        return True

    def zero_residual_output(self) -> bool:
        """Start a residual branch as an exact copy of the robust policy."""
        if self.policy_mode != "residual":
            return False
        self.residual_mean.kernel.value = jnp.zeros_like(
            self.residual_mean.kernel.value)
        self.residual_mean.bias.value = jnp.zeros_like(
            self.residual_mean.bias.value)
        return True
