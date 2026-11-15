"""Causal transition-history encoder for BAPR-v2."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class TransitionContextEncoder(nnx.Module):
    """GRU-like causal encoder with a dynamics-residual confidence gate.

    Adaptation state is ``(hidden, error_ema, count)``.  The policy context is
    ``concat(latent, gate)`` so the residual actor has an explicit safe fallback.
    Modes:

    * ``robust``: zero latent and zero gate.
    * ``oracle``: privileged task latent and unit gate.
    * ``supervised``: learned history latent with task-parameter supervision.
    * ``hybrid``: learned history latent with predictive and weak supervised loss.
    """

    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int = 4,
                 hidden_dim: int = 64, mode: str = "supervised",
                 reward_scale: float = 10.0, delta_scale: float = 1.0,
                 min_history: int = 32, gate_error_threshold: float = 0.20,
                 gate_error_scale: float = 0.25,
                 error_ema_alpha: float = 0.10,
                 reset_temperature: float = 0.10,
                 use_fallback: bool = True, *, rngs: nnx.Rngs):
        if mode not in ("robust", "oracle", "supervised", "hybrid"):
            raise ValueError(f"unsupported BAPR-v2 context mode: {mode}")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.latent_dim = int(latent_dim)
        self.context_dim = self.latent_dim + 1
        self.hidden_dim = int(hidden_dim)
        self.mode = mode
        self.reward_scale = float(max(reward_scale, 1e-6))
        self.delta_scale = float(max(delta_scale, 1e-6))
        self.min_history = int(max(min_history, 1))
        self.gate_error_threshold = float(gate_error_threshold)
        self.gate_error_scale = float(max(gate_error_scale, 1e-6))
        self.error_ema_alpha = float(error_ema_alpha)
        self.reset_temperature = float(max(reset_temperature, 1e-6))
        self.use_fallback = bool(use_fallback)

        feature_dim = 2 * self.obs_dim + self.act_dim + 2
        self.x_gates = nnx.Linear(
            feature_dim, 3 * self.hidden_dim, rngs=rngs)
        self.h_gates = nnx.Linear(
            self.hidden_dim, 3 * self.hidden_dim,
            use_bias=False, rngs=rngs)
        self.latent_head = nnx.Linear(
            self.hidden_dim, self.latent_dim, rngs=rngs)

        decoder_in = self.obs_dim + self.act_dim + self.latent_dim
        self.decoder_hidden = nnx.Linear(
            decoder_in, self.hidden_dim, rngs=rngs)
        self.decoder_out = nnx.Linear(
            self.hidden_dim, self.obs_dim + 1, rngs=rngs)

    def initial_state(self):
        return (
            jnp.zeros((self.hidden_dim,), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def _latent(self, hidden):
        return jnp.tanh(self.latent_head(hidden[None]))[0]

    def policy_context(self, state, oracle_latent):
        hidden, error_ema, count = state
        if self.mode == "robust":
            return jnp.zeros((self.context_dim,), dtype=hidden.dtype)
        if self.mode == "oracle":
            z = jnp.asarray(oracle_latent, dtype=hidden.dtype)
            z = jnp.pad(z[:self.latent_dim],
                        (0, max(0, self.latent_dim - z.shape[0])))
            z = z[:self.latent_dim]
            return jnp.concatenate([z, jnp.ones((1,), dtype=hidden.dtype)])

        z = self._latent(hidden)
        history_conf = 1.0 - jnp.exp(
            -count.astype(jnp.float32) / float(self.min_history))
        if self.use_fallback:
            excess = jax.nn.relu(
                error_ema - self.gate_error_threshold)
            error_conf = jnp.exp(-excess / self.gate_error_scale)
        else:
            error_conf = jnp.asarray(1.0, dtype=hidden.dtype)
        gate = jnp.clip(history_conf * error_conf, 0.0, 1.0)
        return jnp.concatenate([z, gate[None]])

    def _target(self, obs, reward, next_obs):
        delta = jnp.tanh((next_obs - obs) / self.delta_scale)
        reward = jnp.tanh(jnp.asarray(reward).reshape(()) / self.reward_scale)
        return jnp.concatenate([delta, reward[None]])

    def predict(self, state, obs, action):
        hidden, _, _ = state
        z = self._latent(hidden)
        x = jnp.concatenate([obs, action, z])[None]
        x = nnx.relu(self.decoder_hidden(x))
        return jnp.tanh(self.decoder_out(x))[0]

    def _feature(self, obs, action, reward, next_obs, done):
        return jnp.concatenate([
            jnp.tanh(obs / 5.0),
            jnp.clip(action, -1.0, 1.0),
            jnp.tanh(jnp.asarray(reward).reshape(1) / self.reward_scale),
            jnp.tanh((next_obs - obs) / self.delta_scale),
            jnp.asarray(done).reshape(1),
        ])

    def _gru(self, hidden, feature):
        x_proj = self.x_gates(feature[None])[0]
        h_proj = self.h_gates(hidden[None])[0]
        xr, xz, xn = jnp.split(x_proj, 3)
        hr, hz, hn = jnp.split(h_proj, 3)
        reset = jax.nn.sigmoid(xr + hr)
        update = jax.nn.sigmoid(xz + hz)
        candidate = jnp.tanh(xn + reset * hn)
        return update * hidden + (1.0 - update) * candidate

    def observe(self, state, obs, action, reward, next_obs, done,
                enable_reset=True):
        hidden, error_ema, count = state
        if self.mode in ("robust", "oracle"):
            prediction = jnp.zeros((self.obs_dim + 1,), dtype=hidden.dtype)
            target = self._target(obs, reward, next_obs)
            next_state = (
                hidden,
                error_ema,
                jnp.minimum(
                    count + 1, jnp.asarray(1_000_000, jnp.int32)),
            )
            return next_state, jnp.asarray(0.0), prediction, target
        prediction = self.predict(state, obs, action)
        target = self._target(obs, reward, next_obs)
        error = jnp.mean(jnp.square(prediction - target))
        feature = self._feature(obs, action, reward, next_obs, done)
        updated = self._gru(hidden, feature)

        if self.use_fallback and enable_reset:
            ready = (count >= self.min_history).astype(jnp.float32)
            reset_strength = ready * jax.nn.sigmoid(
                (error - self.gate_error_threshold)
                / self.reset_temperature)
            fresh = self._gru(jnp.zeros_like(hidden), feature)
            updated = ((1.0 - reset_strength) * updated
                       + reset_strength * fresh)

        alpha = self.error_ema_alpha
        new_error_ema = jnp.where(
            count > 0,
            (1.0 - alpha) * error_ema + alpha * error,
            error)
        new_count = jnp.minimum(count + 1, jnp.asarray(1_000_000, jnp.int32))
        return (updated, new_error_ema, new_count), error, prediction, target
