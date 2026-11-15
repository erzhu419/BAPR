"""Persistent semi-Markov option state on top of BAPR-v3 inference."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from jax_experiments.networks.probabilistic_regime_context import (
    ProbabilisticRegimeContext,
)


class PersistentOptionRegimeContext(ProbabilisticRegimeContext):
    """Commit posterior decisions only at persistent option boundaries."""

    def __init__(self, *args, option_hold_steps: int = 64,
                 option_confidence_threshold: float = 0.80,
                 option_margin_threshold: float = 0.05,
                 option_hysteresis_margin: float = 0.02, **kwargs):
        super().__init__(*args, **kwargs)
        if int(option_hold_steps) < 1:
            raise ValueError("option_hold_steps must be positive")
        for name, value in (
                ("option_confidence_threshold",
                 option_confidence_threshold),
                ("option_margin_threshold", option_margin_threshold),
                ("option_hysteresis_margin", option_hysteresis_margin)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self.option_hold_steps = int(option_hold_steps)
        self.option_confidence_threshold = float(
            option_confidence_threshold)
        self.option_margin_threshold = float(option_margin_threshold)
        self.option_hysteresis_margin = float(option_hysteresis_margin)

    def initial_state(self):
        base = super().initial_state()
        option_id = jnp.asarray(-1, dtype=jnp.int32)
        option_age = jnp.asarray(0, dtype=jnp.int32)
        return base + (option_id, option_age)

    @staticmethod
    def base_state(state):
        return state[:5]

    def policy_context(self, state, oracle_latent):
        posterior = state[0]
        if self.mode == "robust":
            return jnp.zeros((self.context_dim,), dtype=posterior.dtype)
        if self.mode == "oracle":
            oracle = jnp.asarray(oracle_latent, dtype=posterior.dtype)
            option = jax.nn.one_hot(
                jnp.argmax(oracle[:self.num_modes]), self.num_modes,
                dtype=posterior.dtype)
            return jnp.concatenate([
                option, jnp.ones((1,), dtype=posterior.dtype)])
        option_id = state[5]
        valid = option_id >= 0
        option = jax.nn.one_hot(
            jnp.maximum(option_id, 0), self.num_modes,
            dtype=posterior.dtype) * valid.astype(posterior.dtype)
        return jnp.concatenate([option, valid.astype(posterior.dtype)[None]])

    def _advance_option(self, posterior, count, option_id, option_age):
        candidate = jnp.argmax(posterior).astype(jnp.int32)
        top_probability = posterior[candidate]
        safe_current = jnp.maximum(option_id, 0)
        current_probability = posterior[safe_current]
        competitor = jnp.max(jnp.where(
            jnp.arange(self.num_modes) == safe_current,
            -jnp.inf, posterior))
        keep_current = jnp.logical_and(
            option_id >= 0,
            current_probability + self.option_hysteresis_margin
            >= top_probability)
        selected = jnp.where(keep_current, option_id, candidate)
        selected_probability = jnp.where(
            keep_current, current_probability, top_probability)
        selected_competitor = jnp.where(
            keep_current, competitor,
            jnp.max(jnp.where(
                jnp.arange(self.num_modes) == candidate,
                -jnp.inf, posterior)))
        margin = selected_probability - selected_competitor
        eligible = jnp.logical_and(
            selected_probability >= self.option_confidence_threshold,
            margin >= self.option_margin_threshold)
        proposed = jnp.where(
            eligible, selected, jnp.asarray(-1, dtype=jnp.int32))

        incremented_age = jnp.minimum(
            option_age + 1, jnp.asarray(1_000_000, dtype=jnp.int32))
        initial_due = jnp.logical_and(
            option_id < 0, count >= self.min_history)
        periodic_due = jnp.logical_and(
            option_id >= 0, incremented_age >= self.option_hold_steps)
        decision_due = jnp.logical_or(initial_due, periodic_due)
        next_option = jnp.where(decision_due, proposed, option_id)
        next_age = jnp.where(
            decision_due, jnp.asarray(0, dtype=jnp.int32), incremented_age)
        return next_option, next_age

    def observe(self, state, obs, action, reward, next_obs, done,
                enable_reset=True, stop_variance_grad=False):
        next_base, surprise, prediction, target = super().observe(
            self.base_state(state), obs, action, reward, next_obs, done,
            enable_reset=enable_reset,
            stop_variance_grad=stop_variance_grad)
        option_id, option_age = self._advance_option(
            next_base[0], next_base[4], state[5], state[6])
        return (
            next_base + (option_id, option_age), surprise,
            prediction, target)
