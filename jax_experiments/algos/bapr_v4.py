"""BAPR-v4: jointly trained persistent option-conditioned SAC."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.algos.bapr_v3 import BAPRv3
from jax_experiments.networks.persistent_option_context import (
    PersistentOptionRegimeContext,
)
from jax_experiments.networks.persistent_option_policy import (
    PersistentOptionGaussianPolicy,
)


class BAPRv4(BAPRv3):
    """Shared low-level controller with semi-Markov posterior options."""

    uses_persistent_options = True

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        period = int(config.bapr_v4_training_source_period)
        robust_slots = int(config.bapr_v4_training_robust_slots)
        if period < 1:
            raise ValueError("bapr_v4_training_source_period must be positive")
        if not 0 <= robust_slots < period:
            raise ValueError(
                "bapr_v4_training_robust_slots must be in [0, period)")
        if config.bapr_v2_training_schedule != "joint":
            raise ValueError("BAPR-v4 currently requires joint training")
        super().__init__(obs_dim, act_dim, config, seed=seed)
        self._v4_context_bootstrapped = False
        self._load_context_bootstrap()

    def _load_context_bootstrap(self) -> None:
        model_value = str(self.config.bapr_v4_context_bootstrap_model)
        manifest_value = str(self.config.bapr_v4_context_bootstrap_manifest)
        if not model_value and not manifest_value:
            return
        if not model_value or not manifest_value:
            raise ValueError(
                "both BAPR-v4 context bootstrap paths are required")
        model_path = Path(model_value).expanduser().resolve()
        manifest_path = Path(manifest_value).expanduser().resolve()
        if not model_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing BAPR-v4 context bootstrap: "
                f"model={model_path}, manifest={manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        metadata = list(manifest.get("parameter_leaves", []))
        template = nnx.state(self.context_net, nnx.Param)
        template_leaves, treedef = jax.tree.flatten(template)
        with np.load(model_path, allow_pickle=False) as archive:
            keys = sorted(archive.files)
            if len(keys) != len(template_leaves) or len(keys) != len(metadata):
                raise ValueError("BAPR-v4 context bootstrap leaf count changed")
            restored = []
            for index, (key, expected, template_value) in enumerate(
                    zip(keys, metadata, template_leaves)):
                wanted = f"leaf_{index:05d}"
                value = np.asarray(archive[key])
                if (key != wanted or key != expected.get("key")
                        or tuple(value.shape) != tuple(template_value.shape)
                        or str(value.dtype) != str(expected.get("dtype"))):
                    raise ValueError(
                        f"BAPR-v4 context bootstrap mismatch at {wanted}")
                restored.append(jnp.asarray(
                    value, dtype=template_value.dtype))
        nnx.update(self.context_net, jax.tree.unflatten(treedef, restored))
        self._v4_context_bootstrapped = True

    def _make_context_net(self):
        return PersistentOptionRegimeContext(
            self.obs_dim, self.act_dim,
            num_modes=self.latent_dim,
            hidden_dim=self.config.bapr_v2_context_hidden_dim,
            ensemble_size=self.config.bapr_v3_context_ensemble_size,
            mode=self.context_mode,
            likelihood=self.config.bapr_v3_likelihood,
            reward_scale=self.config.bapr_v2_reward_scale,
            delta_scale=self.config.bapr_v2_delta_scale,
            min_history=self.config.bapr_v2_min_history,
            hazard_rate=self.config.bapr_v3_hazard_rate,
            evidence_scale=self.config.bapr_v3_evidence_scale,
            fixed_variance=self.config.bapr_v3_fixed_variance,
            logvar_min=self.config.bapr_v3_logvar_min,
            logvar_max=self.config.bapr_v3_logvar_max,
            variance_model=self.config.bapr_v3_variance_model,
            variance_floor=self.config.bapr_v3_variance_floor,
            variance_ceiling=self.config.bapr_v3_variance_ceiling,
            mean_loss_weight=self.config.bapr_v3_mean_loss_weight,
            variance_loss_weight=self.config.bapr_v3_variance_loss_weight,
            variance_prior_weight=self.config.bapr_v3_variance_prior_weight,
            evidence_clip=self.config.bapr_v3_evidence_clip,
            surprise_threshold=self.config.bapr_v3_surprise_threshold,
            surprise_scale=self.config.bapr_v3_surprise_scale,
            posterior_decay=self.config.bapr_v4_posterior_decay,
            change_cusum_threshold=self.config.bapr_v4_cusum_threshold,
            change_cusum_drift=self.config.bapr_v4_cusum_drift,
            option_hold_steps=self.config.bapr_v4_option_hold_steps,
            option_confidence_threshold=(
                self.config.bapr_v4_option_confidence_threshold),
            option_margin_threshold=(
                self.config.bapr_v4_option_margin_threshold),
            option_hysteresis_margin=(
                self.config.bapr_v4_option_hysteresis_margin),
            rngs=self.rngs)

    def _make_policy(self):
        return PersistentOptionGaussianPolicy(
            self.obs_dim, self.act_dim, self.config.hidden_dim,
            self.latent_dim, rngs=self.rngs)

    def context_checkpoint_signature(self) -> dict[str, object]:
        signature = super().context_checkpoint_signature()
        signature.update({
            "kind": "bapr_v4_persistent_option",
            "option_hold_steps": self.config.bapr_v4_option_hold_steps,
            "option_confidence_threshold": (
                self.config.bapr_v4_option_confidence_threshold),
            "option_margin_threshold": (
                self.config.bapr_v4_option_margin_threshold),
            "option_hysteresis_margin": (
                self.config.bapr_v4_option_hysteresis_margin),
            "cusum_threshold": self.config.bapr_v4_cusum_threshold,
            "cusum_drift": self.config.bapr_v4_cusum_drift,
            "context_bootstrap_model": str(
                self.config.bapr_v4_context_bootstrap_model),
        })
        return signature

    def rollout_context_source(self, iteration: int | None = None) -> int:
        iteration = (
            self._training_iteration if iteration is None else int(iteration))
        period = int(self.config.bapr_v4_training_source_period)
        robust_slots = int(self.config.bapr_v4_training_robust_slots)
        if iteration % period < robust_slots:
            return self.CONTEXT_ROBUST
        return self.CONTEXT_ORACLE

    def load_checkpoint_state(self, state: dict[str, object]) -> None:
        super().load_checkpoint_state(state)
        if len(self.adaptation_state) == 5:
            self.adaptation_state = self.adaptation_state + (
                jnp.asarray(-1, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
            )

    def multi_update(self, *args, **kwargs):
        metrics = super().multi_update(*args, **kwargs)
        posterior, _, _, _, count, option_id, option_age = (
            self.adaptation_state)
        target_mode = int(jnp.argmax(self.oracle_latent))
        metrics.update({
            "v4_option_id": float(option_id),
            "v4_option_age": float(option_age),
            "v4_option_active": float(option_id >= 0),
            "v4_option_matches_mode": float(
                (option_id >= 0) & (option_id == target_mode)),
            "v4_option_candidate_probability": float(jnp.max(posterior)),
            "v4_option_context_count": float(count),
            "v4_training_context_source": float(
                self.rollout_context_source()),
            "v4_context_bootstrapped": float(
                self._v4_context_bootstrapped),
        })
        return metrics
