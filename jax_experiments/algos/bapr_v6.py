"""BAPR-v6: optimizer-equivalent hard persistent options."""
from __future__ import annotations

import jax.numpy as jnp

from jax_experiments.algos.bapr_v2 import _oracle_context_from_task_ids
from jax_experiments.algos.bapr_v5 import BAPRv5


class BAPRv6(BAPRv5):
    """Balance replay draws and entropy temperatures across controller heads."""

    uses_per_context_alpha = True

    def context_checkpoint_signature(self) -> dict[str, object]:
        signature = super().context_checkpoint_signature()
        signature.update({
            "kind": "bapr_v6_optimizer_equivalent_hard_option",
            "balanced_per_head_replay": True,
            "per_context_alpha": True,
        })
        return signature

    def _balanced_context_batch(self, stacked_batch: dict):
        task_ids = stacked_batch.get("task_id")
        if task_ids is None:
            raise ValueError("BAPR-v6 replay balancing requires task_id")
        batch_size = int(task_ids.shape[1])
        if batch_size < self.latent_dim:
            raise ValueError(
                "BAPR-v6 batch size must cover every option: "
                f"batch={batch_size}, options={self.latent_dim}")
        robust_count = batch_size // self.latent_dim
        oracle = _oracle_context_from_task_ids(self.task_latents, task_ids)
        robust = jnp.zeros_like(oracle[:, :robust_count])
        expanded = {
            key: jnp.concatenate(
                [value[:, :robust_count], value], axis=1)
            for key, value in stacked_batch.items()
            if key not in ("belief", "next_belief")
        }
        expanded["belief"] = jnp.concatenate([robust, oracle], axis=1)
        expanded["next_belief"] = expanded["belief"]
        return expanded

    def multi_update(self, stacked_batch: dict, *args, **kwargs):
        balanced = self._balanced_context_batch(stacked_batch)
        metrics = super(BAPRv5, self).multi_update(
            balanced, *args, **kwargs)
        metrics.update({
            "v6_balanced_per_head_replay": 1.0,
            "v6_per_context_alpha": 1.0,
            "v6_effective_batch_multiplier": (
                float(balanced["obs"].shape[1])
                / float(stacked_batch["obs"].shape[1])),
        })
        return metrics
