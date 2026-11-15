"""Probabilistic sticky-regime context for BAPR-v3."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def detached_gaussian_variance_loss(residual_sq, logvar):
    """Calibrate variance without letting it weaken mean-model gradients."""
    residual_sq = jax.lax.stop_gradient(jnp.asarray(residual_sq))
    logvar = jnp.asarray(logvar)
    return 0.5 * (residual_sq * jnp.exp(-logvar) + logvar)


class ProbabilisticRegimeContext(nnx.Module):
    """Filter persistent modes using transition likelihoods.

    The decoder emits an ensemble of conditional means and variances for every
    candidate mode.  The filter uses a sticky transition prior, so one noisy
    sample cannot be interpreted as a mode switch without likelihood evidence.
    """

    def __init__(self, obs_dim: int, act_dim: int, num_modes: int = 4,
                 hidden_dim: int = 128, ensemble_size: int = 5,
                 mode: str = "supervised", likelihood: str = "probabilistic",
                 reward_scale: float = 10.0, delta_scale: float = 1.0,
                 min_history: int = 16, hazard_rate: float = 0.002,
                 evidence_scale: float = 4.0, fixed_variance: float = 0.02,
                 logvar_min: float = -6.0, logvar_max: float = 1.0,
                 variance_model: str = "legacy_state",
                 variance_floor: float = 1e-4,
                 variance_ceiling: float = 0.25,
                 mean_loss_weight: float = 1.0,
                 variance_loss_weight: float = 0.1,
                 variance_prior_weight: float = 0.001,
                 evidence_clip: float = 0.0,
                 surprise_threshold: float = 2.0,
                 surprise_scale: float = 1.0,
                 posterior_decay: float = 1.0,
                 change_reset_threshold: float = 0.0,
                 change_reset_alpha: float = 0.25,
                 change_reset_mix: float = 1.0,
                 change_cusum_threshold: float = 0.0,
                 change_cusum_drift: float = 0.0, *, rngs: nnx.Rngs):
        if mode not in ("robust", "oracle", "supervised", "hybrid"):
            raise ValueError(f"unsupported BAPR-v3 context mode: {mode}")
        if likelihood not in ("point", "probabilistic"):
            raise ValueError(
                f"unsupported BAPR-v3 likelihood: {likelihood}")
        if variance_model not in (
                "legacy_state", "mode_calibrated", "mode_empirical",
                "mode_shared_empirical", "inverse_empirical"):
            raise ValueError(
                "BAPR-v3 variance_model must be legacy_state, "
                "mode_calibrated, mode_empirical, or "
                "mode_shared_empirical, or inverse_empirical, "
                f"got {variance_model!r}")
        if num_modes < 2:
            raise ValueError("BAPR-v3 requires at least two modes")
        if not 0.0 < variance_floor < variance_ceiling:
            raise ValueError(
                "variance_floor must be positive and below variance_ceiling")
        if not 0.0 < posterior_decay <= 1.0:
            raise ValueError("posterior_decay must be in (0, 1]")
        if change_reset_threshold < 0.0:
            raise ValueError("change_reset_threshold must be nonnegative")
        if not 0.0 < change_reset_alpha <= 1.0:
            raise ValueError("change_reset_alpha must be in (0, 1]")
        if not 0.0 <= change_reset_mix <= 1.0:
            raise ValueError("change_reset_mix must be in [0, 1]")
        if change_cusum_threshold < 0.0:
            raise ValueError("change_cusum_threshold must be nonnegative")
        if change_cusum_drift < 0.0:
            raise ValueError("change_cusum_drift must be nonnegative")
        if change_reset_threshold > 0.0 and change_cusum_threshold > 0.0:
            raise ValueError("EMA reset and CUSUM reset are mutually exclusive")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.num_modes = int(num_modes)
        self.latent_dim = self.num_modes
        self.context_dim = self.latent_dim + 1
        self.hidden_dim = int(hidden_dim)
        self.ensemble_size = int(max(ensemble_size, 2))
        self.mode = str(mode)
        self.likelihood = str(likelihood)
        self.variance_model = str(variance_model)
        self.reward_scale = float(max(reward_scale, 1e-6))
        self.delta_scale = float(max(delta_scale, 1e-6))
        self.min_history = int(max(min_history, 1))
        self.hazard_rate = float(jnp.clip(hazard_rate, 1e-6, 0.49))
        self.evidence_scale = float(max(evidence_scale, 1e-6))
        self.fixed_logvar = float(jnp.log(max(fixed_variance, 1e-6)))
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.variance_floor = float(variance_floor)
        self.variance_ceiling = float(variance_ceiling)
        self.calibrated_logvar_min = float(jnp.log(variance_floor))
        self.calibrated_logvar_max = float(jnp.log(variance_ceiling))
        self.mean_loss_weight = float(max(mean_loss_weight, 0.0))
        self.variance_loss_weight = float(max(variance_loss_weight, 0.0))
        self.variance_prior_weight = float(max(variance_prior_weight, 0.0))
        self.evidence_clip = float(max(evidence_clip, 0.0))
        self.surprise_threshold = float(surprise_threshold)
        self.surprise_scale = float(max(surprise_scale, 1e-6))
        self.posterior_decay = float(posterior_decay)
        self.change_reset_threshold = float(change_reset_threshold)
        self.change_reset_alpha = float(change_reset_alpha)
        self.change_reset_mix = float(change_reset_mix)
        self.change_cusum_threshold = float(change_cusum_threshold)
        self.change_cusum_drift = float(change_cusum_drift)
        self.output_dim = (
            self.act_dim if self.variance_model == "inverse_empirical"
            else self.obs_dim + 1)

        if self.variance_model == "inverse_empirical":
            inverse_input_dim = 3 * self.obs_dim
            self.inverse_hidden1 = nnx.Linear(
                inverse_input_dim, self.hidden_dim, rngs=rngs)
            self.inverse_hidden2 = nnx.Linear(
                self.hidden_dim, self.hidden_dim, rngs=rngs)
            self.inverse_action = nnx.Linear(
                self.hidden_dim, self.ensemble_size * self.act_dim,
                rngs=rngs)
        else:
            input_dim = self.obs_dim + self.act_dim
            self.decoder_hidden1 = nnx.Linear(
                input_dim, self.hidden_dim, rngs=rngs)
            self.decoder_hidden2 = nnx.Linear(
                self.hidden_dim, self.hidden_dim, rngs=rngs)
            output_width = (
                self.ensemble_size * self.num_modes * self.output_dim)
            self.decoder_mean = nnx.Linear(
                self.hidden_dim, output_width, rngs=rngs)
            if self.variance_model == "legacy_state":
                self.decoder_logvar = nnx.Linear(
                    self.hidden_dim, output_width, rngs=rngs)
        if (self.likelihood == "probabilistic"
                and self.variance_model != "legacy_state"):
            initial = jnp.full(
                (self.num_modes, self.output_dim),
                jnp.clip(
                    fixed_variance, self.variance_floor,
                    self.variance_ceiling),
                dtype=jnp.float32)
            self.mode_logvar_raw = nnx.Param(
                self.calibrated_raw_from_variance(initial))

    def calibrated_raw_from_variance(self, variance):
        variance = jnp.clip(
            jnp.asarray(variance, dtype=jnp.float32),
            self.variance_floor, self.variance_ceiling)
        logvar = jnp.log(variance)
        fraction = (
            (logvar - self.calibrated_logvar_min)
            / (self.calibrated_logvar_max - self.calibrated_logvar_min))
        fraction = jnp.clip(fraction, 1e-5, 1.0 - 1e-5)
        return jnp.log(fraction) - jnp.log1p(-fraction)

    def _calibrated_mode_logvars(self):
        fraction = jax.nn.sigmoid(self.mode_logvar_raw.value)
        return (
            self.calibrated_logvar_min
            + fraction * (
                self.calibrated_logvar_max
                - self.calibrated_logvar_min))

    def mode_variances(self):
        """Return input-independent mode variances when they are defined."""
        if self.likelihood == "point":
            return jnp.full(
                (self.num_modes, self.output_dim),
                jnp.exp(self.fixed_logvar), dtype=jnp.float32)
        if self.variance_model in (
                "mode_calibrated", "mode_empirical",
                "mode_shared_empirical", "inverse_empirical"):
            return jnp.exp(self._calibrated_mode_logvars())
        return None

    def initial_state(self):
        posterior = jnp.full(
            (self.num_modes,), 1.0 / self.num_modes, dtype=jnp.float32)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        count = jnp.asarray(0, dtype=jnp.int32)
        return posterior, zero, zero, zero, count

    def _target(self, obs, reward, next_obs):
        delta = jnp.tanh((next_obs - obs) / self.delta_scale)
        scaled_reward = jnp.tanh(
            jnp.asarray(reward).reshape(()) / self.reward_scale)
        return jnp.concatenate([delta, scaled_reward[None]])

    def distribution(self, obs, action):
        if self.variance_model == "inverse_empirical":
            raise ValueError(
                "inverse_empirical requires inverse_distribution(obs, next_obs)")
        features = jnp.concatenate([
            jnp.tanh(obs / 5.0), jnp.clip(action, -1.0, 1.0)
        ], axis=-1)
        hidden = nnx.relu(self.decoder_hidden1(features))
        hidden = nnx.relu(self.decoder_hidden2(hidden))
        output_shape = hidden.shape[:-1] + (
            self.ensemble_size, self.num_modes, self.output_dim)
        means = jnp.tanh(self.decoder_mean(hidden)).reshape(output_shape)
        if self.variance_model == "mode_shared_empirical":
            # All modes must explain the same conditional transition mean.
            # This prevents a mode-specific decoder from absorbing stochastic
            # events as mean-model bias and leaves persistent residual
            # distribution differences as the posterior evidence.
            shared_mean = jnp.mean(means, axis=-2, keepdims=True)
            means = jnp.broadcast_to(shared_mean, means.shape)
        if self.likelihood == "point":
            logvars = jnp.full_like(means, self.fixed_logvar)
        elif self.variance_model in (
                "mode_calibrated", "mode_empirical",
                "mode_shared_empirical"):
            mode_logvars = self._calibrated_mode_logvars()
            prefix = (1,) * (means.ndim - 3)
            mode_logvars = mode_logvars.reshape(
                prefix + (1, self.num_modes, self.output_dim))
            logvars = jnp.broadcast_to(mode_logvars, means.shape)
        else:
            logvars = jnp.clip(
                self.decoder_logvar(hidden).reshape(output_shape),
                self.logvar_min, self.logvar_max)
        return means, logvars

    def inverse_distribution(self, obs, next_obs):
        """Predict executed action without exposing the commanded action.

        On clean mode-0 transitions the commanded and executed actions match,
        so that mode supplies an uncorrupted inverse-dynamics target. At
        inference time the residual to the commanded action estimates the
        hidden actuator disturbance.
        """
        if self.variance_model != "inverse_empirical":
            raise ValueError(
                "inverse_distribution is only defined for inverse_empirical")
        delta = (next_obs - obs) / self.delta_scale
        features = jnp.concatenate([
            jnp.tanh(obs / 5.0),
            jnp.tanh(next_obs / 5.0),
            jnp.tanh(delta),
        ], axis=-1)
        hidden = nnx.silu(self.inverse_hidden1(features))
        hidden = nnx.silu(self.inverse_hidden2(hidden))
        base_shape = hidden.shape[:-1]
        predicted = jnp.tanh(self.inverse_action(hidden)).reshape(
            base_shape + (self.ensemble_size, 1, self.act_dim))
        means = jnp.broadcast_to(
            predicted,
            base_shape + (
                self.ensemble_size, self.num_modes, self.act_dim))
        mode_logvars = self._calibrated_mode_logvars()
        prefix = (1,) * len(base_shape)
        mode_logvars = mode_logvars.reshape(
            prefix + (1, self.num_modes, self.act_dim))
        logvars = jnp.broadcast_to(mode_logvars, means.shape)
        return means, logvars

    def _distribution_statistics(self, target, means, logvars):
        error = target[..., None, None, :] - means
        element_nll = 0.5 * (
            jnp.square(error) * jnp.exp(-logvars)
            + logvars + jnp.log(2.0 * jnp.pi))
        head_nll = jnp.mean(element_nll, axis=-1)
        head_log_likelihood = -head_nll
        mode_log_likelihood = jax.scipy.special.logsumexp(
            head_log_likelihood, axis=-2) - jnp.log(
                float(self.ensemble_size))
        mode_nll = -mode_log_likelihood
        aleatoric = jnp.mean(jnp.exp(logvars), axis=(-3, -1))
        epistemic = jnp.mean(jnp.var(means, axis=-3), axis=-1)
        return (
            mode_log_likelihood, mode_nll, aleatoric, epistemic,
            means, logvars)

    def likelihood_statistics(self, obs, action, target,
                              stop_variance_grad=False):
        means, logvars = self.distribution(obs, action)
        effective_logvars = (
            jax.lax.stop_gradient(logvars)
            if stop_variance_grad else logvars)
        return self._distribution_statistics(
            target, means, effective_logvars)

    def transition_statistics(
            self, obs, action, reward, next_obs,
            stop_mean_grad=False, stop_variance_grad=False):
        if self.variance_model == "inverse_empirical":
            target = jnp.clip(jnp.asarray(action), -1.0, 1.0)
            means, logvars = self.inverse_distribution(obs, next_obs)
            if stop_mean_grad:
                means = jax.lax.stop_gradient(means)
            if stop_variance_grad:
                logvars = jax.lax.stop_gradient(logvars)
            return target, self._distribution_statistics(
                target, means, logvars)
        target = self._target(obs, reward, next_obs)
        return target, self.likelihood_statistics(
            obs, action, target,
            stop_variance_grad=stop_variance_grad)

    def _sticky_prior(self, posterior):
        switch_probability = self.hazard_rate / float(self.num_modes - 1)
        return (
            (1.0 - self.hazard_rate) * posterior
            + switch_probability * (1.0 - posterior))

    def _change_point_prior(self, posterior, centered_evidence, change_ema):
        """Reset accumulated belief after persistent contradictory evidence."""
        prior = self._sticky_prior(posterior)
        believed_mode = jnp.argmax(posterior)
        evidence_gap = jnp.maximum(-centered_evidence[believed_mode], 0.0)
        updated_ema = (
            (1.0 - self.change_reset_alpha) * change_ema
            + self.change_reset_alpha * evidence_gap)
        triggered = updated_ema >= self.change_reset_threshold
        uniform = jnp.full_like(prior, 1.0 / float(self.num_modes))
        reset_prior = (
            (1.0 - self.change_reset_mix) * prior
            + self.change_reset_mix * uniform)
        return (
            jnp.where(triggered, reset_prior, prior),
            jnp.where(triggered, 0.0, updated_ema),
            triggered,
        )

    def _cusum_change_point_prior(self, posterior, centered_evidence, score):
        """Reset only after drift-corrected evidence favors another mode."""
        prior = self._sticky_prior(posterior)
        believed_mode = jnp.argmax(posterior)
        alternatives = jnp.where(
            jnp.arange(self.num_modes) == believed_mode,
            -jnp.inf,
            centered_evidence,
        )
        log_likelihood_ratio = (
            jnp.max(alternatives) - centered_evidence[believed_mode])
        updated_score = jnp.maximum(
            0.0, score + log_likelihood_ratio - self.change_cusum_drift)
        triggered = updated_score >= self.change_cusum_threshold
        uniform = jnp.full_like(prior, 1.0 / float(self.num_modes))
        reset_prior = (
            (1.0 - self.change_reset_mix) * prior
            + self.change_reset_mix * uniform)
        return (
            jnp.where(triggered, reset_prior, prior),
            jnp.where(triggered, 0.0, updated_score),
            triggered,
        )

    def policy_context(self, state, oracle_latent):
        posterior, surprise_ema, _, _, count = state
        if self.mode == "robust":
            return jnp.zeros((self.context_dim,), dtype=posterior.dtype)
        if self.mode == "oracle":
            oracle = jnp.asarray(oracle_latent, dtype=posterior.dtype)
            oracle = jnp.pad(
                oracle[:self.num_modes],
                (0, max(0, self.num_modes - oracle.shape[0])))
            return jnp.concatenate([
                oracle[:self.num_modes],
                jnp.ones((1,), dtype=posterior.dtype),
            ])
        entropy = -jnp.sum(
            posterior * jnp.log(jnp.clip(posterior, 1e-8, 1.0)))
        confidence = 1.0 - entropy / jnp.log(float(self.num_modes))
        history = 1.0 - jnp.exp(
            -count.astype(jnp.float32) / float(self.min_history))
        excess_surprise = jax.nn.relu(
            surprise_ema - self.surprise_threshold)
        explained = jnp.exp(-excess_surprise / self.surprise_scale)
        gate = jnp.clip(history * confidence * explained, 0.0, 1.0)
        return jnp.concatenate([posterior, gate[None]])

    def observe(self, state, obs, action, reward, next_obs, done,
                enable_reset=True, stop_variance_grad=False):
        posterior, surprise_ema, aleatoric_ema, epistemic_ema, count = state
        target, statistics = self.transition_statistics(
            obs, action, reward, next_obs,
            stop_mean_grad=(self.variance_model == "inverse_empirical"),
            stop_variance_grad=stop_variance_grad)
        if self.mode in ("robust", "oracle"):
            next_state = (
                posterior, surprise_ema, aleatoric_ema, epistemic_ema,
                jnp.minimum(count + 1, jnp.asarray(1_000_000, jnp.int32)))
            prediction = jnp.zeros_like(target)
            return next_state, jnp.asarray(0.0), prediction, target

        (mode_log_likelihood, mode_nll, aleatoric, epistemic,
         means, _) = statistics
        centered_evidence = mode_log_likelihood - jnp.max(
            mode_log_likelihood)
        if self.evidence_clip > 0.0:
            centered_evidence = jnp.maximum(
                centered_evidence, -self.evidence_clip)
        if self.change_cusum_threshold > 0.0:
            prior, next_change_ema, _ = self._cusum_change_point_prior(
                posterior, centered_evidence, epistemic_ema)
        elif self.change_reset_threshold > 0.0:
            prior, next_change_ema, _ = self._change_point_prior(
                posterior, centered_evidence, epistemic_ema)
        else:
            prior = self._sticky_prior(posterior)
            next_change_ema = epistemic_ema
        next_posterior = jax.nn.softmax(
            self.posterior_decay * jnp.log(jnp.clip(prior, 1e-8, 1.0))
            + self.evidence_scale * centered_evidence)
        surprise = jnp.maximum(jnp.min(mode_nll), 0.0)
        selected_aleatoric = jnp.sum(next_posterior * aleatoric)
        selected_epistemic = jnp.sum(next_posterior * epistemic)
        alpha = 0.10
        next_surprise = jnp.where(
            count > 0,
            (1.0 - alpha) * surprise_ema + alpha * surprise,
            surprise)
        next_aleatoric = jnp.where(
            count > 0,
            (1.0 - alpha) * aleatoric_ema + alpha * selected_aleatoric,
            selected_aleatoric)
        if (self.change_reset_threshold > 0.0
                or self.change_cusum_threshold > 0.0):
            next_epistemic = next_change_ema
        else:
            next_epistemic = jnp.where(
                count > 0,
                (1.0 - alpha) * epistemic_ema + alpha * selected_epistemic,
                selected_epistemic)
        next_count = jnp.minimum(
            count + 1, jnp.asarray(1_000_000, jnp.int32))
        ensemble_mean = jnp.mean(means, axis=0)
        prediction = jnp.sum(
            next_posterior[:, None] * ensemble_mean, axis=0)
        next_state = (
            next_posterior, next_surprise, next_aleatoric,
            next_epistemic, next_count)
        return next_state, surprise, prediction, target

    def supervised_statistics(
            self, obs, action, reward, next_obs, target_mode):
        if self.variance_model == "inverse_empirical":
            target = jnp.clip(jnp.asarray(action), -1.0, 1.0)
            train_means, logvars = self.inverse_distribution(obs, next_obs)
            likelihood_means = jax.lax.stop_gradient(train_means)
            (mode_log_likelihood, mode_nll, aleatoric, epistemic,
             _, _) = self._distribution_statistics(
                target, likelihood_means,
                jax.lax.stop_gradient(logvars))
            means = train_means
        else:
            target = self._target(obs, reward, next_obs)
        bounded_mode_variance = (
            self.likelihood == "probabilistic"
            and self.variance_model in (
                "mode_calibrated", "mode_empirical",
                "mode_shared_empirical", "inverse_empirical"))
        if self.variance_model == "inverse_empirical":
            pass
        elif bounded_mode_variance:
            means, logvars = self.distribution(obs, action)
            (mode_log_likelihood, mode_nll, aleatoric, epistemic,
             _, _) = self._distribution_statistics(
                target, means, jax.lax.stop_gradient(logvars))
        else:
            (mode_log_likelihood, mode_nll, aleatoric, epistemic,
             means, logvars) = self.likelihood_statistics(
                obs, action, target)
        target_mode = jnp.asarray(target_mode, dtype=mode_nll.dtype)
        classification = -jnp.sum(
            target_mode * jax.nn.log_softmax(
                self.evidence_scale * mode_log_likelihood))
        residual_sq = jnp.square(
            target[..., None, None, :] - means)
        ensemble_mean = jnp.mean(means, axis=-3)
        mean_residual_sq = jnp.square(
            target[..., None, :] - ensemble_mean)
        if self.variance_model == "inverse_empirical":
            mean_mse = jnp.mean(residual_sq, axis=(-3, -1))
            # Only nominal mode 0 has executed_action == commanded_action.
            # Posterior losses see a stopped inverse model, so they cannot
            # manufacture discriminative residuals by corrupting this model.
            predictive = (
                target_mode[0] * self.mean_loss_weight * mean_mse[0])
        elif self.variance_model == "mode_calibrated":
            mean_mse = jnp.mean(residual_sq, axis=(-3, -1))
            variance_nll = jnp.mean(
                detached_gaussian_variance_loss(residual_sq, logvars),
                axis=(-3, -1))
            prior = jnp.mean(
                jnp.square(logvars - self.fixed_logvar), axis=(-3, -1))
            predictive = jnp.sum(target_mode * (
                self.mean_loss_weight * mean_mse
                + self.variance_loss_weight * variance_nll
                + self.variance_prior_weight * prior))
        elif self.variance_model in (
                "mode_empirical", "mode_shared_empirical"):
            # The neural optimizer fits only conditional means.  A separate
            # mode-wise residual-moment update owns aleatoric variance, so
            # discriminative losses cannot inflate or collapse it.
            mean_mse = jnp.mean(residual_sq, axis=(-3, -1))
            predictive = jnp.sum(
                target_mode * self.mean_loss_weight * mean_mse)
        else:
            predictive = jnp.sum(target_mode * mode_nll)
            if self.likelihood == "probabilistic":
                predictive = predictive + 1e-4 * jnp.mean(
                    jnp.square(logvars))
        return (
            predictive,
            classification,
            jnp.sum(target_mode * aleatoric),
            jnp.sum(target_mode * epistemic),
            mean_residual_sq,
        )

    def supervised_losses(self, obs, action, reward, next_obs, target_mode):
        return self.supervised_statistics(
            obs, action, reward, next_obs, target_mode)[:4]

    def empirical_variance_targets(self, residual_sq, target_modes):
        """Aggregate ensemble-mean residual second moments by true mode."""
        residual_sq = jax.lax.stop_gradient(jnp.asarray(residual_sq))
        target_modes = jnp.asarray(target_modes, dtype=residual_sq.dtype)
        leading_axes = tuple(range(residual_sq.ndim - 2))
        weights = target_modes[..., :, None]
        weighted_sum = jnp.sum(
            weights * residual_sq, axis=leading_axes)
        counts = jnp.sum(target_modes, axis=leading_axes)
        fallback = self.mode_variances()
        if fallback is None:
            fallback = jnp.full(
                (self.num_modes, self.output_dim),
                jnp.exp(self.fixed_logvar), dtype=residual_sq.dtype)
        targets = weighted_sum / jnp.maximum(counts[:, None], 1.0)
        targets = jnp.where(counts[:, None] > 0.0, targets, fallback)
        targets = jnp.clip(
            targets, self.variance_floor, self.variance_ceiling)
        return jax.lax.stop_gradient(targets), counts
