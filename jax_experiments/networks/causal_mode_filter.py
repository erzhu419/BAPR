"""Compact causal filter over frozen per-transition mode likelihoods."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class CausalModeFilter(nnx.Module):
    """Integrate noisy mode evidence with a recurrent causal state."""

    def __init__(
        self,
        num_modes: int = 4,
        hidden_dim: int = 64,
        evidence_clip: float = 6.0,
        *,
        rngs: nnx.Rngs,
    ):
        if num_modes < 2:
            raise ValueError("causal mode filter requires at least two modes")
        if hidden_dim <= 0 or evidence_clip <= 0.0:
            raise ValueError("hidden_dim and evidence_clip must be positive")
        self.num_modes = int(num_modes)
        self.hidden_dim = int(hidden_dim)
        self.evidence_clip = float(evidence_clip)
        self.cell = nnx.GRUCell(
            self.num_modes,
            self.hidden_dim,
            rngs=rngs,
        )
        self.head = nnx.Linear(
            self.hidden_dim,
            self.num_modes,
            rngs=rngs,
        )

    def initial_hidden(self, batch_shape=()):
        return jnp.zeros(
            tuple(batch_shape) + (self.hidden_dim,), dtype=jnp.float32)

    def normalize_evidence(self, mode_log_likelihood):
        evidence = jnp.asarray(mode_log_likelihood, dtype=jnp.float32)
        centered = evidence - jnp.max(evidence, axis=-1, keepdims=True)
        return jnp.clip(
            centered, -self.evidence_clip, 0.0) / self.evidence_clip

    def step(self, hidden, mode_log_likelihood):
        inputs = self.normalize_evidence(mode_log_likelihood)
        next_hidden, output = self.cell(hidden, inputs)
        logits = self.head(output)
        return next_hidden, logits

    def sequence(self, mode_log_likelihood, initial_hidden=None):
        evidence = jnp.asarray(mode_log_likelihood, dtype=jnp.float32)
        hidden = (
            self.initial_hidden()
            if initial_hidden is None else initial_hidden)

        def scan_step(carry, row):
            next_hidden, logits = self.step(carry, row)
            return next_hidden, logits

        final_hidden, logits = jax.lax.scan(scan_step, hidden, evidence)
        return final_hidden, logits
