"""BAPR-v3: variance-aware sticky mode inference with robust fallback."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_experiments.algos.bapr_v2 import BAPRv2
from jax_experiments.networks.probabilistic_regime_context import (
    ProbabilisticRegimeContext,
)


def _mode_latent(task, num_modes: int) -> np.ndarray:
    mode_id = int(task.get("mode_id", 0)) if isinstance(task, dict) else 0
    if not 0 <= mode_id < num_modes:
        raise ValueError(f"mode_id={mode_id} outside [0, {num_modes})")
    latent = np.zeros((num_modes,), dtype=np.float32)
    latent[mode_id] = 1.0
    return latent


class BAPRv3(BAPRv2):
    """BAPRv2 policy schedule with a probabilistic regime filter."""

    uses_probabilistic_regime_context = True

    def context_checkpoint_signature(self) -> dict[str, object]:
        """Identify context architectures that are safe to resume in place."""
        return {
            "kind": "bapr_v3",
            "mode": self.context_mode,
            "likelihood": self.config.bapr_v3_likelihood,
            "variance_model": self.config.bapr_v3_variance_model,
            "estimator_rollout_source": (
                self.config.bapr_v3_estimator_rollout_source),
            "latent_dim": self.latent_dim,
            "hidden_dim": self.config.bapr_v2_context_hidden_dim,
            "ensemble_size": self.config.bapr_v3_context_ensemble_size,
        }

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        expected_modes = int(config.task_num)
        if int(config.bapr_v2_latent_dim) != expected_modes:
            raise ValueError(
                "BAPR-v3 requires bapr_v2_latent_dim == task_num so the "
                "policy latent is the full mode posterior")
        variance_ema = float(config.bapr_v3_variance_ema)
        if not 0.0 < variance_ema <= 1.0:
            raise ValueError("bapr_v3_variance_ema must be in (0, 1]")
        if float(config.bapr_v3_instant_classifier_weight) < 0.0:
            raise ValueError(
                "bapr_v3_instant_classifier_weight must be nonnegative")
        if config.bapr_v3_estimator_rollout_source not in (
                "learned", "robust"):
            raise ValueError(
                "bapr_v3_estimator_rollout_source must be learned or robust")
        super().__init__(obs_dim, act_dim, config, seed=seed)
        self._v3_calibration_error = 0.0
        self._v3_variance_targets = jnp.zeros(
            (self.latent_dim, self.context_net.output_dim),
            dtype=jnp.float32)
        self._v3_variance_target_counts = jnp.zeros(
            (self.latent_dim,), dtype=jnp.float32)
        self._v3_posterior_accuracy = 0.0
        self._v3_posterior_true_probability = 0.0
        self._v3_instant_classifier_loss = 0.0
        self._v3_posterior_loss = 0.0
        self._v3_empirical_updates = 0

    def _make_context_net(self):
        return ProbabilisticRegimeContext(
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
            rngs=self.rngs)

    def set_task_metadata(self, tasks) -> None:
        if len(tasks) != self.latent_dim:
            raise ValueError(
                f"BAPR-v3 expected {self.latent_dim} modes, got {len(tasks)}")
        values = np.zeros((self.latent_dim, self.latent_dim), dtype=np.float32)
        for task in tasks:
            latent = _mode_latent(task, self.latent_dim)
            values[int(np.argmax(latent))] = latent
        if np.any(np.sum(values, axis=1) != 1.0):
            raise ValueError("BAPR-v3 task metadata must contain every mode once")
        self.task_latents = jnp.asarray(values)

    def set_eval_task(self, task) -> None:
        self.oracle_latent = jnp.asarray(
            _mode_latent(task, self.latent_dim), dtype=jnp.float32)

    def rollout_context_source(self, iteration: int | None = None) -> int:
        stage = self.training_stage(iteration)
        if (stage in ("student", "deployment")
                and self.config.bapr_v3_estimator_rollout_source == "robust"):
            return self.CONTEXT_ROBUST
        return super().rollout_context_source(iteration)

    def controller_update_flags(
            self, iteration: int | None = None) -> tuple[bool, bool, bool]:
        flags = super().controller_update_flags(iteration)
        if (bool(self.config.bapr_v3_freeze_teacher_after_teacher)
                and self.training_stage(iteration) == "deployment"):
            return False, False, False
        return flags

    def train_policy_gate(self, iteration: int | None = None) -> bool:
        if (bool(self.config.bapr_v3_freeze_teacher_after_teacher)
                and self.training_stage(iteration) == "deployment"):
            return False
        return super().train_policy_gate(iteration)

    def _build_context_update_fn(self):
        gd_context = nnx.graphdef(self.context_net)
        context_opt = self.context_opt
        pred_weight = float(self.config.bapr_v2_predictive_weight)
        supervised_weight = float(self.config.bapr_v2_supervised_weight)
        temporal_weight = float(self.config.bapr_v2_temporal_weight)
        instant_classifier_weight = float(
            self.config.bapr_v3_instant_classifier_weight)
        burnin = int(self.config.bapr_v2_context_burnin)

        @jax.jit
        def context_update(params, opt_state, obs, act, rew, nobs, done,
                           target_latents):
            def loss_fn(cp):
                model = nnx.merge(gd_context, cp)

                def one_chunk(chunk):
                    (c_obs, c_act, c_rew, c_nobs, c_done,
                     c_target) = chunk

                    def body(state, transition):
                        o, a, r, no, d, target_mode = transition
                        next_state, _, _, _ = model.observe(
                            state, o, a, r, no, d, enable_reset=False,
                            stop_variance_grad=(
                                model.variance_model in (
                                    "mode_calibrated",
                                    "mode_empirical",
                                    "mode_shared_empirical",
                                    "inverse_empirical")))
                        (predictive, classifier, aleatoric, epistemic,
                         residual_sq) = model.supervised_statistics(
                            o, a, r, no, target_mode)
                        posterior = next_state[0]
                        posterior_ce = -jnp.sum(
                            target_mode * jnp.log(jnp.clip(
                                posterior, 1e-8, 1.0)))
                        return next_state, (
                            posterior, predictive,
                            classifier, posterior_ce,
                            aleatoric, epistemic, residual_sq)

                    _, outputs = jax.lax.scan(
                        body, model.initial_state(),
                        (c_obs, c_act, c_rew, c_nobs, c_done, c_target))
                    return outputs

                (posteriors, predictive_nll, classifier_ce, posterior_ce,
                 aleatoric, epistemic, residual_sq) = jax.vmap(one_chunk)(
                    (obs, act, rew, nobs, done, target_latents))
                tail = posteriors[:, burnin:, :]
                target = target_latents[:, burnin:, :]
                predictive = jnp.mean(predictive_nll[:, burnin:])
                instant_classifier = jnp.mean(
                    classifier_ce[:, burnin:])
                posterior_supervised = jnp.mean(
                    posterior_ce[:, burnin:])
                supervised = (
                    posterior_supervised
                    + instant_classifier_weight * instant_classifier)
                posterior_delta = tail[:, 1:, :] - tail[:, :-1, :]
                target_delta = target[:, 1:, :] - target[:, :-1, :]
                same_mode = jnp.all(
                    jnp.abs(target_delta) < 1e-6, axis=-1, keepdims=True)
                temporal = jnp.sum(
                    jnp.square(posterior_delta) * same_mode
                ) / jnp.maximum(
                    jnp.sum(same_mode) * tail.shape[-1], 1.0)
                loss = (
                    pred_weight * predictive
                    + supervised_weight * supervised
                    + temporal_weight * temporal)
                latent_std = jnp.std(tail, axis=(0, 1)).max()
                variance_targets, variance_counts = (
                    model.empirical_variance_targets(
                        residual_sq[:, burnin:, :, :], target))
                mode_variances = model.mode_variances()
                if mode_variances is None:
                    mode_variances = jnp.full_like(
                        variance_targets, jnp.exp(model.fixed_logvar))
                active = variance_counts[:, None] > 0.0
                calibration_error = jnp.sum(jnp.where(
                    active,
                    jnp.abs(
                        jnp.log(jnp.clip(mode_variances, 1e-8))
                        - jnp.log(jnp.clip(variance_targets, 1e-8))),
                    0.0)) / jnp.maximum(
                        jnp.sum(active) * variance_targets.shape[-1], 1.0)
                true_probability = jnp.sum(tail * target, axis=-1)
                posterior_accuracy = jnp.mean(
                    jnp.argmax(tail, axis=-1)
                    == jnp.argmax(target, axis=-1))
                diagnostics = (
                    predictive, supervised, temporal, latent_std,
                    calibration_error, variance_targets, variance_counts,
                    posterior_accuracy, jnp.mean(true_probability),
                    instant_classifier, posterior_supervised)
                return loss, diagnostics

            (loss, diagnostics), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(params)
            updates, next_opt_state = context_opt.update(
                grads, opt_state, params)
            next_params = jax.tree.map(
                lambda value: jnp.nan_to_num(value),
                optax.apply_updates(params, updates))
            return (
                next_params, next_opt_state,
                (loss,) + diagnostics)

        self._context_update = context_update

    def _record_context_update_diagnostics(
            self, calibration_error, variance_targets, variance_counts,
            posterior_accuracy, posterior_true_probability,
            instant_classifier_loss, posterior_loss) -> None:
        self._v3_variance_targets = jnp.asarray(variance_targets)
        self._v3_variance_target_counts = jnp.asarray(variance_counts)
        self._v3_posterior_accuracy = float(posterior_accuracy)
        self._v3_posterior_true_probability = float(
            posterior_true_probability)
        self._v3_instant_classifier_loss = float(
            instant_classifier_loss)
        self._v3_posterior_loss = float(posterior_loss)
        self._v3_calibration_error = float(calibration_error)

        if self.context_net.variance_model not in (
                "mode_empirical", "mode_shared_empirical",
                "inverse_empirical"):
            return
        current = self.context_net.mode_variances()
        counts = self._v3_variance_target_counts[:, None]
        ema = float(self.config.bapr_v3_variance_ema)
        updated = jnp.where(
            counts > 0.0,
            (1.0 - ema) * current + ema * self._v3_variance_targets,
            current)
        self.context_net.mode_logvar_raw.value = (
            self.context_net.calibrated_raw_from_variance(updated))
        active = counts > 0.0
        self._v3_calibration_error = float(
            jnp.sum(jnp.where(
                active,
                jnp.abs(
                    jnp.log(jnp.clip(updated, 1e-8))
                    - jnp.log(jnp.clip(
                        self._v3_variance_targets, 1e-8))),
                0.0))
            / jnp.maximum(
                jnp.sum(active) * updated.shape[-1], 1.0))
        self._v3_empirical_updates += 1

    def multi_update(self, *args, **kwargs):
        metrics = super().multi_update(*args, **kwargs)
        posterior, surprise, aleatoric, epistemic, count = (
            self.adaptation_state[:5])
        entropy = -jnp.sum(
            posterior * jnp.log(jnp.clip(posterior, 1e-8, 1.0)))
        metrics.update({
            "v3_posterior_entropy": float(entropy),
            "v3_posterior_max": float(jnp.max(posterior)),
            "v3_surprise": float(surprise),
            "v3_aleatoric": float(aleatoric),
            "v3_epistemic": float(epistemic),
            "v3_context_count": float(count),
            "v3_calibration_error": self._v3_calibration_error,
            "v3_posterior_accuracy": self._v3_posterior_accuracy,
            "v3_posterior_true_probability": (
                self._v3_posterior_true_probability),
            "v3_instant_classifier_loss": (
                self._v3_instant_classifier_loss),
            "v3_posterior_loss": self._v3_posterior_loss,
            "v3_empirical_updates": float(self._v3_empirical_updates),
        })
        target_mode = int(jnp.argmax(self.oracle_latent))
        metrics.update({
            "v3_online_target_probability": float(posterior[target_mode]),
            "v3_online_mode_correct": float(
                int(jnp.argmax(posterior)) == target_mode),
        })
        mode_variances = self.context_net.mode_variances()
        if mode_variances is not None:
            mode_means = jnp.mean(mode_variances, axis=-1)
            metrics.update({
                "v3_variance_min": float(jnp.min(mode_variances)),
                "v3_variance_mean": float(jnp.mean(mode_variances)),
                "v3_variance_max": float(jnp.max(mode_variances)),
                "v3_variance_mode_spread": float(jnp.std(mode_means)),
            })
            for mode_index in range(int(mode_means.shape[0])):
                metrics[f"v3_variance_mode_{mode_index}"] = float(
                    mode_means[mode_index])
                metrics[f"v3_variance_target_mode_{mode_index}"] = float(
                    jnp.mean(self._v3_variance_targets[mode_index]))
                metrics[f"v3_variance_target_count_{mode_index}"] = float(
                    self._v3_variance_target_counts[mode_index])
        return metrics
