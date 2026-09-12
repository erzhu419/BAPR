"""Minimal BAPR for persistent stochastic regimes.

The controller is deliberately shared: a robust SAC actor remains available
at every step and a bounded residual is trained only after a causal,
heteroscedastic regime model has seen robust-policy transitions.  This module
does not use BOCD, Q disagreement as a mode detector, or independent policy
banks.
"""
from __future__ import annotations

from typing import Any

from jax_experiments.algos.bapr_v3 import BAPRv3


class BAPRRegime(BAPRv3):
    """Robust -> inference -> posterior-conditioned adaptation schedule."""

    uses_shared_regime_residual = True

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        if config.bapr_v2_training_schedule != "joint":
            raise ValueError(
                "bapr_regime owns its three-stage schedule; "
                "bapr_v2_training_schedule must remain joint")
        if config.bapr_v2_mode not in ("supervised", "hybrid"):
            raise ValueError(
                "bapr_regime requires a causal supervised or hybrid "
                "regime estimator")
        if config.bapr_v2_policy_mode != "residual":
            raise ValueError(
                "bapr_regime requires one shared bounded residual policy")
        if config.bapr_v3_variance_model not in (
                "mode_empirical", "mode_shared_empirical",
                "inverse_empirical"):
            raise ValueError(
                "bapr_regime requires an empirical heteroscedastic "
                "variance model")
        if config.bapr_regime_adaptation_source not in ("learned", "oracle"):
            raise ValueError(
                "bapr_regime_adaptation_source must be learned or oracle")
        if int(config.bapr_v2_base_pretrain_iters) < 1:
            raise ValueError(
                "bapr_regime requires bapr_v2_base_pretrain_iters >= 1")
        if int(config.bapr_regime_inference_iters) < 1:
            raise ValueError(
                "bapr_regime_inference_iters must be >= 1")

        self._regime_residual_warmstarted = False
        super().__init__(obs_dim, act_dim, config, seed=seed)

    def context_checkpoint_signature(self) -> dict[str, object]:
        signature = super().context_checkpoint_signature()
        signature.update({
            "kind": "bapr_regime_v1",
            "schedule": "robust_inference_adaptation",
            "adaptation_source": self.config.bapr_regime_adaptation_source,
            "inference_iters": int(self.config.bapr_regime_inference_iters),
            "freeze_context_after_inference": bool(
                self.config.bapr_regime_freeze_context_after_inference),
            "shared_residual": True,
        })
        return signature

    def training_stage(self, iteration: int | None = None) -> str:
        iteration = (
            self._training_iteration if iteration is None else int(iteration))
        robust_end = int(self.config.bapr_v2_base_pretrain_iters)
        inference_end = (
            robust_end + int(self.config.bapr_regime_inference_iters))
        if iteration < robust_end:
            return "robust"
        if iteration < inference_end:
            return "inference"
        return "adaptation"

    def rollout_context_source(self, iteration: int | None = None) -> int:
        if self.training_stage(iteration) != "adaptation":
            return self.CONTEXT_ROBUST
        if self.config.bapr_regime_adaptation_source == "oracle":
            return self.CONTEXT_ORACLE
        return self.CONTEXT_LEARNED

    def controller_update_flags(
            self, iteration: int | None = None) -> tuple[bool, bool, bool]:
        if self.training_stage(iteration) in ("robust", "inference"):
            return True, False, True
        return False, True, True

    def train_policy_gate(self, iteration: int | None = None) -> bool:
        return self.training_stage(iteration) == "adaptation"

    def advantage_gate_active(self, iteration: int | None = None) -> bool:
        return bool(
            self.config.bapr_regime_advantage_fallback
            and self.training_stage(iteration) == "adaptation")

    def set_training_iteration(self, iteration: int) -> None:
        previous = self._training_stage
        self._training_iteration = int(iteration)
        self._training_stage = self.training_stage(iteration)

        entering_adaptation = (
            previous != "adaptation"
            and self._training_stage == "adaptation")
        if (entering_adaptation
                and bool(self.config.bapr_regime_zero_residual_init)
                and not self._regime_residual_warmstarted):
            if not self.policy.zero_residual_output():
                raise RuntimeError(
                    "bapr_regime could not zero-initialize residual policy")
            self._regime_residual_warmstarted = True
        if previous != self._training_stage:
            self.reset_adaptation()
        if (entering_adaptation
                and bool(self.config.bapr_regime_clear_replay_on_adaptation)):
            self._replay_reset_requested = True

    def multi_update(self, stacked_batch: dict, current_iter=0,
                     recent_rollout=None, **kwargs):
        stage = self.training_stage(current_iter)
        if (stage == "adaptation"
                and bool(
                    self.config.bapr_regime_freeze_context_after_inference)):
            recent_rollout = None
        metrics = super().multi_update(
            stacked_batch, current_iter=current_iter,
            recent_rollout=recent_rollout, **kwargs)
        metrics.update({
            "regime_stage_robust": float(stage == "robust"),
            "regime_stage_inference": float(stage == "inference"),
            "regime_stage_adaptation": float(stage == "adaptation"),
            "regime_oracle_training": float(
                self.config.bapr_regime_adaptation_source == "oracle"),
            "regime_context_frozen": float(
                stage == "adaptation"
                and self.config.bapr_regime_freeze_context_after_inference),
            "regime_residual_zero_initialized": float(
                self._regime_residual_warmstarted),
        })
        return metrics

    def checkpoint_state(self) -> dict[str, Any]:
        state = super().checkpoint_state()
        state.update({
            "regime_residual_warmstarted": (
                self._regime_residual_warmstarted),
        })
        return state

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        super().load_checkpoint_state(state)
        self._regime_residual_warmstarted = bool(
            state.get("regime_residual_warmstarted", False))
