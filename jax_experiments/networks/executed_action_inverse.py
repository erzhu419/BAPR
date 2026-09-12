"""Policy-invariant inverse dynamics for actuator-regime identification."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.ensemble_critic import VectorizedLinear


def _module_list(layers):
    list_cls = getattr(nnx, "List", None)
    return list_cls(layers) if list_cls is not None else layers


class ExecutedActionInverse(nnx.Module):
    """Infer the action applied to physics from one observed transition.

    The ensemble contains independent vectorized MLP heads. The commanded
    action is deliberately excluded from the network input so the model cannot
    learn a policy-specific shortcut. It is used only after prediction to
    score candidate actuator transforms.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        ensemble_size: int = 5,
        n_layers: int = 3,
        obs_scale: float = 5.0,
        delta_scale: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        if obs_dim <= 0 or act_dim <= 0:
            raise ValueError("observation and action dimensions must be positive")
        if hidden_dim <= 0 or ensemble_size < 2 or n_layers <= 0:
            raise ValueError("invalid inverse-model capacity")
        if obs_scale <= 0.0 or delta_scale <= 0.0:
            raise ValueError("inverse-model feature scales must be positive")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.ensemble_size = int(ensemble_size)
        self.n_layers = int(n_layers)
        self.obs_scale = float(obs_scale)
        self.delta_scale = float(delta_scale)

        layers = []
        input_dim = 3 * self.obs_dim
        for _ in range(self.n_layers):
            layers.append(VectorizedLinear(
                input_dim, self.hidden_dim, self.ensemble_size, rngs=rngs))
            input_dim = self.hidden_dim
        layers.append(VectorizedLinear(
            input_dim, self.act_dim, self.ensemble_size, rngs=rngs))
        self.layers = _module_list(layers)

    def transition_features(self, obs, next_obs):
        obs = jnp.asarray(obs)
        next_obs = jnp.asarray(next_obs)
        if obs.shape != next_obs.shape or obs.shape[-1] != self.obs_dim:
            raise ValueError("inverse-model observation shape mismatch")
        delta = next_obs - obs
        return jnp.concatenate([
            jnp.tanh(obs / self.obs_scale),
            jnp.tanh(next_obs / self.obs_scale),
            jnp.tanh(delta / self.delta_scale),
        ], axis=-1)

    def _apply_heads(self, features):
        hidden = features
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index < self.n_layers:
                hidden = jax.nn.silu(hidden)
        return jnp.tanh(hidden)

    def predict(self, obs, next_obs):
        """Return independent predictions with shape [heads, batch, action]."""
        features = self.transition_features(obs, next_obs)
        if features.ndim != 2:
            raise ValueError("predict expects batched observations")
        features = jnp.broadcast_to(
            features[None],
            (self.ensemble_size,) + features.shape,
        )
        return self._apply_heads(features)

    def predict_head_batches(self, obs, next_obs):
        """Predict separate bootstrap batches for every ensemble head."""
        features = self.transition_features(obs, next_obs)
        if (features.ndim != 3
                or features.shape[0] != self.ensemble_size):
            raise ValueError(
                "head batches must have shape [ensemble, batch, feature]")
        return self._apply_heads(features)


def candidate_action_evidence(
    predicted_actions,
    commanded_actions,
    gain_vectors,
    residual_variance,
):
    """Score persistent gain patterns under a mixture of inverse-model heads.

    Args:
        predicted_actions: ``[heads, batch, action]``.
        commanded_actions: ``[batch, action]``.
        gain_vectors: ``[modes, action]``.
        residual_variance: empirical inverse residual variance ``[action]``.

    Returns:
        Per-transition mode log likelihood, aleatoric variance, and epistemic
        variance, each with shape ``[batch, modes]``.
    """
    predicted = jnp.asarray(predicted_actions)
    commanded = jnp.asarray(commanded_actions)
    gains = jnp.asarray(gain_vectors)
    variance = jnp.asarray(residual_variance)
    if predicted.ndim != 3:
        raise ValueError("predicted actions must have shape [heads,batch,action]")
    if commanded.shape != predicted.shape[1:]:
        raise ValueError("commanded-action shape mismatch")
    if gains.ndim != 2 or gains.shape[1] != commanded.shape[1]:
        raise ValueError("gain-vector shape mismatch")
    if variance.shape != (commanded.shape[1],):
        raise ValueError("residual-variance shape mismatch")
    variance = jnp.clip(variance, 1e-6, 1.0)

    candidate = jnp.clip(
        commanded[:, None, :] * gains[None, :, :],
        -1.0,
        1.0,
    )
    error = candidate[None, :, :, :] - predicted[:, :, None, :]
    head_log_likelihood = -0.5 * jnp.mean(
        jnp.square(error) / variance[None, None, None, :]
        + jnp.log(2.0 * jnp.pi * variance)[None, None, None, :],
        axis=-1,
    )
    log_likelihood = jax.scipy.special.logsumexp(
        head_log_likelihood, axis=0) - jnp.log(float(predicted.shape[0]))
    aleatoric = jnp.broadcast_to(
        jnp.mean(variance),
        log_likelihood.shape,
    )
    epistemic_scalar = jnp.mean(jnp.var(predicted, axis=0), axis=-1)
    epistemic = jnp.broadcast_to(
        epistemic_scalar[:, None],
        log_likelihood.shape,
    )
    return log_likelihood, aleatoric, epistemic
