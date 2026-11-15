"""BAPR-v5: robust-isolated persistent options with replay relabelling."""
from __future__ import annotations

import jax.numpy as jnp

from jax_experiments.algos.bapr_v2 import _oracle_context_from_task_ids
from jax_experiments.algos.bapr_v4 import BAPRv4
from jax_experiments.networks.hard_option_critic import (
    HardOptionEnsembleCritic,
)
from jax_experiments.networks.hard_option_policy import HardOptionGaussianPolicy


class BAPRv5(BAPRv4):
    """Hard option heads trained from both robust and oracle relabels."""

    uses_hard_option_heads = True

    def _make_policy(self):
        return HardOptionGaussianPolicy(
            self.obs_dim, self.act_dim, self.config.hidden_dim,
            self.latent_dim, rngs=self.rngs)

    def _make_critic(self):
        return HardOptionEnsembleCritic(
            self.obs_dim, self.act_dim, self.context_dim,
            self.config.hidden_dim, self.latent_dim,
            self.config.ensemble_size, n_layers=3, rngs=self.rngs)

    def context_checkpoint_signature(self) -> dict[str, object]:
        signature = super().context_checkpoint_signature()
        signature.update({
            "kind": "bapr_v5_hard_persistent_option",
            "dual_context_relabel": True,
            "isolated_robust_actor_critic": True,
        })
        return signature

    def _dual_context_batch(self, stacked_batch: dict):
        task_ids = stacked_batch.get("task_id")
        if task_ids is None:
            raise ValueError("BAPR-v5 replay relabelling requires task_id")
        oracle = _oracle_context_from_task_ids(self.task_latents, task_ids)
        robust = jnp.zeros_like(oracle)
        expanded = {
            key: jnp.concatenate([value, value], axis=1)
            for key, value in stacked_batch.items()
            if key not in ("belief", "next_belief")
        }
        expanded["belief"] = jnp.concatenate([robust, oracle], axis=1)
        expanded["next_belief"] = expanded["belief"]
        return expanded

    def multi_update(self, stacked_batch: dict, *args, **kwargs):
        metrics = super().multi_update(
            self._dual_context_batch(stacked_batch), *args, **kwargs)
        metrics.update({
            "v5_dual_context_relabel": 1.0,
            "v5_effective_batch_multiplier": 2.0,
        })
        return metrics
