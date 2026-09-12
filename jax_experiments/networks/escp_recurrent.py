"""Recurrent ESCP networks matching the released PyTorch architecture."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.networks.ensemble_critic import VectorizedLinear


class RecurrentEnvironmentProbe(nnx.Module):
    """Causal ``(state, previous action)`` environment probe.

    The released ESCP implementation uses ``FC(128) -> GRU(64) -> FC(ep)``
    and a fixed history of 16 transitions.  History truncation is handled by
    the caller; this module owns only the recurrent computation.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ep_dim: int = 2,
        input_hidden_dim: int = 128,
        recurrent_hidden_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.ep_dim = int(ep_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        self.input_layer = nnx.Linear(
            self.obs_dim + self.act_dim, input_hidden_dim, rngs=rngs)
        self.cell = nnx.GRUCell(
            input_hidden_dim, self.recurrent_hidden_dim, rngs=rngs)
        self.output_layer = nnx.Linear(
            self.recurrent_hidden_dim, self.ep_dim, rngs=rngs)

    def initial_hidden(self, batch_shape=()):
        return jnp.zeros(
            tuple(batch_shape) + (self.recurrent_hidden_dim,),
            dtype=jnp.float32,
        )

    def step(self, hidden, observation, previous_action):
        inputs = jnp.concatenate([observation, previous_action], axis=-1)
        inputs = nnx.leaky_relu(self.input_layer(inputs))
        next_hidden, output = self.cell(hidden, inputs)
        context = jnp.tanh(self.output_layer(output))
        return next_hidden, context

    def sequence(
        self,
        observations,
        previous_actions,
        reset_before=None,
        initial_hidden=None,
    ):
        """Encode a batch of causal histories.

        Args:
            observations: ``[batch, time, obs_dim]``.
            previous_actions: ``[batch, time, act_dim]``.
            reset_before: optional ``[batch, time]`` mask. A true value clears
                recurrent state before consuming that row.
            initial_hidden: optional ``[batch, hidden_dim]`` carry.
        """
        observations = jnp.asarray(observations, dtype=jnp.float32)
        previous_actions = jnp.asarray(previous_actions, dtype=jnp.float32)
        if observations.ndim != 3 or previous_actions.ndim != 3:
            raise ValueError("ESCP histories must have [batch,time,feature]")
        batch_size, time_steps = observations.shape[:2]
        hidden = (
            self.initial_hidden((batch_size,))
            if initial_hidden is None else initial_hidden)
        if reset_before is None:
            reset_before = jnp.zeros(
                (batch_size, time_steps), dtype=jnp.bool_)
        reset_before = jnp.asarray(reset_before, dtype=jnp.bool_)

        def scan_step(carry, values):
            observation, previous_action, reset = values
            carry = jnp.where(reset[:, None], jnp.zeros_like(carry), carry)
            next_hidden, context = self.step(
                carry, observation, previous_action)
            return next_hidden, context

        values = (
            jnp.swapaxes(observations, 0, 1),
            jnp.swapaxes(previous_actions, 0, 1),
            jnp.swapaxes(reset_before, 0, 1),
        )
        final_hidden, contexts = jax.lax.scan(scan_step, hidden, values)
        return final_hidden, jnp.swapaxes(contexts, 0, 1)


class ESCPGaussianPolicy(nnx.Module):
    """Released ESCP universe-policy MLP: 128, 64, squashed Gaussian."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        ep_dim: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.ep_dim = int(ep_dim)
        self.act_dim = int(act_dim)
        self.fc1 = nnx.Linear(obs_dim + ep_dim, 128, rngs=rngs)
        self.fc2 = nnx.Linear(128, 64, rngs=rngs)
        self.output = nnx.Linear(64, 2 * act_dim, rngs=rngs)

    def __call__(self, observations, context=None):
        if context is None:
            context = jnp.zeros(
                observations.shape[:-1] + (self.ep_dim,),
                dtype=observations.dtype,
            )
        values = jnp.concatenate([observations, context], axis=-1)
        values = nnx.leaky_relu(self.fc1(values))
        values = nnx.leaky_relu(self.fc2(values))
        mean, log_std = jnp.split(self.output(values), 2, axis=-1)
        return mean, jnp.clip(log_std, -7.0, 2.0)

    def sample(self, observations, key, context=None):
        mean, log_std = self(observations, context)
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, mean.shape)
        pre_tanh = mean + std * noise
        action = jnp.tanh(pre_tanh)
        log_prob = -0.5 * (
            ((pre_tanh - mean) / std) ** 2
            + 2.0 * log_std
            + jnp.log(2.0 * jnp.pi)
        )
        log_prob = log_prob.sum(axis=-1)
        log_prob -= jnp.log(1.0 - action ** 2 + 1e-6).sum(axis=-1)
        return action, log_prob

    def deterministic(self, observations, context=None):
        mean, _ = self(observations, context)
        return jnp.tanh(mean)


class ESCPTwinCritic(nnx.Module):
    """Two released-size ESCP Q functions in one vectorized module."""

    def __init__(
        self,
        obs_context_dim: int,
        act_dim: int,
        ensemble_size: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        if int(ensemble_size) != 2:
            raise ValueError("recurrent ESCP requires exactly two Q functions")
        self.ensemble_size = 2
        self.fc1 = VectorizedLinear(
            obs_context_dim + act_dim, 128, 2, rngs=rngs)
        self.fc2 = VectorizedLinear(128, 64, 2, rngs=rngs)
        self.output = VectorizedLinear(64, 1, 2, rngs=rngs)

    def __call__(self, observations, actions):
        values = jnp.concatenate([observations, actions], axis=-1)
        values = jnp.broadcast_to(values[None], (2,) + values.shape)
        values = nnx.leaky_relu(self.fc1(values))
        values = nnx.leaky_relu(self.fc2(values))
        return self.output(values).squeeze(-1)

    def compute_reg_norm(self):
        total = jnp.zeros((2,), dtype=jnp.float32)
        for layer in (self.fc1, self.fc2, self.output):
            total += jnp.abs(layer.kernel.value).sum(axis=(1, 2))
            total += jnp.abs(layer.bias.value).sum(axis=(1, 2))
        return total
