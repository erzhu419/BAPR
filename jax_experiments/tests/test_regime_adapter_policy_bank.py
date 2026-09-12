"""Tests for evaluation-time routing across independent adapters."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.analysis.regime_adapter_policy_bank import ModePolicyBank
from jax_experiments.networks.residual_policy import ResidualGaussianPolicy


def _policy(seed: int, bias: float) -> ResidualGaussianPolicy:
    policy = ResidualGaussianPolicy(
        3, 2, 8, latent_dim=4, residual_delta=0.5,
        policy_mode="residual", rngs=nnx.Rngs(seed))
    policy.zero_residual_output()
    policy.residual_mean.bias.value = jnp.full_like(
        policy.residual_mean.bias.value, bias)
    return policy


def _context(controller: int | None):
    if controller is None:
        return jnp.zeros((1, 5), dtype=jnp.float32)
    return jnp.concatenate([
        jnp.eye(4, dtype=jnp.float32)[controller:controller + 1],
        jnp.ones((1, 1), dtype=jnp.float32),
    ], axis=-1)


def test_bank_zero_context_is_exact_base() -> None:
    policies = [_policy(index + 1, 0.1 * (index + 1)) for index in range(4)]
    bank = ModePolicyBank(policies)
    obs = jnp.asarray([[0.2, -0.4, 0.3]], dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(bank.deterministic(obs, _context(None))),
        np.asarray(policies[0].base_deterministic(obs)), atol=1e-7)


def test_bank_one_hot_matches_selected_specialist() -> None:
    policies = [_policy(index + 11, 0.2 * (index + 1)) for index in range(4)]
    # Force a shared base so only the selected residual distinguishes policies.
    base = policies[0]
    for policy in policies[1:]:
        for source, destination in zip(base.base_layers, policy.base_layers):
            destination.kernel.value = jnp.asarray(source.kernel.value)
            destination.bias.value = jnp.asarray(source.bias.value)
        for source, destination in (
                (base.base_mean, policy.base_mean),
                (base.base_log_std, policy.base_log_std)):
            destination.kernel.value = jnp.asarray(source.kernel.value)
            destination.bias.value = jnp.asarray(source.bias.value)
    bank = ModePolicyBank(policies)
    obs = jnp.asarray([[0.1, 0.3, -0.2]], dtype=jnp.float32)
    for controller, policy in enumerate(policies):
        specialist_context = jnp.concatenate([
            jnp.eye(4, dtype=jnp.float32)[controller:controller + 1],
            jnp.ones((1, 1), dtype=jnp.float32),
        ], axis=-1)
        np.testing.assert_allclose(
            np.asarray(bank.deterministic(obs, _context(controller))),
            np.asarray(policy.deterministic(obs, specialist_context)),
            atol=1e-7)

