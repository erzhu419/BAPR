"""BOCD Belief Tracker and Surprise Computer — ported from bapr_components.py."""
import numpy as np


class BeliefTracker:
    """Bayesian Online Change Detection (BOCD) run-length posterior tracker.

    Maintains belief distribution ρ(h) over run-lengths h ∈ {0, ..., H-1}.
    Corresponds to BAPR.lean: update_belief / normalization_const.
    """

    def __init__(self, max_run_length: int = 20, hazard_rate: float = 0.05,
                 base_variance: float = 0.1, variance_growth: float = 0.05):
        self.max_H = max_run_length
        self.hazard = hazard_rate
        self.base_var = base_variance
        self.var_growth = variance_growth
        self.belief = np.ones(max_run_length) / max_run_length

    def reset(self):
        self.belief = np.ones(self.max_H) / self.max_H

    def update(self, surprise: float):
        likelihoods = self._compute_likelihood(surprise)

        # Growth probabilities (no change-point)
        growth = self.belief * likelihoods * (1 - self.hazard)
        # Change-point probability mass
        cp_mass = np.sum(self.belief * likelihoods * self.hazard)

        # Shift and inject change-point mass at h=0
        new_belief = np.zeros(self.max_H)
        new_belief[0] = cp_mass
        new_belief[1:] = growth[:-1]

        # Normalize
        total = new_belief.sum()
        if total > 1e-10:
            new_belief /= total
        else:
            new_belief = np.ones(self.max_H) / self.max_H

        self.belief = new_belief

    def _compute_likelihood(self, surprise: float):
        variances = self.base_var + np.arange(self.max_H) * self.var_growth
        return np.exp(-0.5 * surprise ** 2 / variances) / np.sqrt(2 * np.pi * variances)

    @property
    def entropy(self):
        b = self.belief[self.belief > 1e-10]
        return -np.sum(b * np.log(b))

    @property
    def effective_window(self):
        return np.sum(self.belief * np.arange(self.max_H))


class SurpriseComputer:
    """Multi-signal surprise: reward z-score + Q-std spike + reg-norm divergence."""

    def __init__(self, ema_alpha: float = 0.3):
        self.ema_alpha = ema_alpha
        self.ema_reward = None
        self.ema_reward_var = None
        self.ema_q_std = None

    def reset(self):
        self.ema_reward = None
        self.ema_reward_var = None
        self.ema_q_std = None

    def compute(self, reward_batch, q_std_batch,
                reg_norm_current=None, reg_norm_target=None):
        """Compute scalar surprise from batch statistics."""
        r_mean = float(np.mean(reward_batch))
        q_std_mean = float(np.mean(q_std_batch))

        # Initialize EMAs
        if self.ema_reward is None:
            self.ema_reward = r_mean
            self.ema_reward_var = 1.0
            self.ema_q_std = q_std_mean
            return 0.0

        # Reward z-score — must save old EMA before updating to avoid self-reference
        old_ema_reward = self.ema_reward
        self.ema_reward = self.ema_alpha * r_mean + (1 - self.ema_alpha) * self.ema_reward
        # Variance uses (r_mean - OLD ema), not the just-updated one
        self.ema_reward_var = (self.ema_alpha * (r_mean - old_ema_reward) ** 2
                               + (1 - self.ema_alpha) * self.ema_reward_var)
        reward_std = max(np.sqrt(self.ema_reward_var), 1e-6)
        reward_z = abs(r_mean - self.ema_reward) / reward_std

        # Q-std spike
        self.ema_q_std = self.ema_alpha * q_std_mean + (1 - self.ema_alpha) * self.ema_q_std
        q_std_ratio = q_std_mean / max(self.ema_q_std, 1e-6)

        # Reg norm divergence
        reg_div = 0.0
        if reg_norm_current is not None and reg_norm_target is not None:
            reg_div = float(np.mean(np.abs(reg_norm_current - reg_norm_target)))

        # Combined surprise (weighted sum)
        surprise = 0.5 * reward_z + 0.3 * q_std_ratio + 0.2 * reg_div
        return float(np.clip(surprise, 0.0, 10.0))
