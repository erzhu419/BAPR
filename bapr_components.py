"""
BA-PR SAC Algorithm Components

BeliefTracker: BOCD-style Bayesian belief over run-lengths
SurpriseComputer: surprise signal from reg_norm divergence

Corresponds to BAPR.lean §8: κ-driven belief update
"""

import numpy as np
import torch


class BeliefTracker:
    """
    Bayesian Online Change Detection (Adams & MacKay 2007)
    维护 run-length 后验分布 ρ(h)
    对应 BAPR.lean: update_belief / normalization_const
    """
    def __init__(self, max_run_length: int = 20, hazard_rate: float = 0.05,
                 base_variance: float = 0.1, variance_growth: float = 0.05):
        self.max_H = max_run_length
        self.hazard = hazard_rate
        self.base_var = base_variance
        self.var_growth = variance_growth
        self.belief = np.ones(max_run_length) / max_run_length  # 均匀先验

    def reset(self):
        """每个 episode 开始时重置 belief 为均匀先验"""
        self.belief = np.ones(self.max_H) / self.max_H

    def update(self, surprise: float):
        """
        BAPR.lean: update_belief ρ L Z
        1. 计算似然 L(h, ξ)
        2. 重加权 ρ(h) * L(h)
        3. 归一化
        4. 混入 changepoint 先验 (BOCD hazard shift)
        """
        L = self._compute_likelihood(surprise)      # L ≥ 0 (hL_nn)
        unnorm = self.belief * L
        Z = unnorm.sum()
        if Z > 1e-10:                                 # hZ_pos 保护
            self.belief = unnorm / Z
        else:
            self.belief = np.ones(self.max_H) / self.max_H  # 退化到均匀

        # BOCD: 将 hazard_rate 的质量移到 h=0（新 run-length）
        growth_prob = self.belief * (1 - self.hazard)  # 现有 run-length 增长
        changepoint_prob = self.belief.sum() * self.hazard  # 突变概率

        new_belief = np.zeros(self.max_H)
        new_belief[0] = changepoint_prob  # h=0 获得所有突变质量
        new_belief[1:] = growth_prob[:-1]  # h 右移一步
        total = new_belief.sum()
        self.belief = new_belief / total if total > 1e-10 else np.ones(self.max_H) / self.max_H

    def _compute_likelihood(self, surprise: float) -> np.ndarray:
        """
        L(h, ξ): 指数族似然
        短 run-length → 方差小 → 只有低 surprise 才获得高似然
        长 run-length → 方差大 → 容忍更大 surprise
        """
        variances = self.base_var + self.var_growth * np.arange(self.max_H)
        return np.exp(-surprise**2 / (2 * variances))

    @property
    def effective_window(self) -> float:
        """有效窗口长度 = Σ h * ρ(h)"""
        return np.sum(np.arange(self.max_H) * self.belief)

    @property
    def entropy(self) -> float:
        """belief 分布的熵（用于监控）"""
        p = self.belief[self.belief > 1e-10]
        return -np.sum(p * np.log(p))


class SurpriseComputer:
    """
    Multi-signal surprise detector for BAPR.
    Combines reward z-score and Q-value std change to detect environment shifts.
    对应 BAPR.lean §8: κ 驱动贝叶斯更新 (enhanced signal)
    """
    def __init__(self, ema_alpha: float = 0.3, reward_window: int = 50):
        self.ema_alpha = ema_alpha
        self.ema_surprise = 0.0
        # Reward tracking for z-score
        self.reward_window = reward_window
        self.reward_history = []
        self.reward_ema = 0.0
        self.reward_var_ema = 1.0
        # Q-std tracking
        self.prev_q_std = None

    def reset(self):
        """每个 episode 开始时重置 surprise"""
        self.ema_surprise = 0.0
        self.reward_history = []
        self.reward_ema = 0.0
        self.reward_var_ema = 1.0
        self.prev_q_std = None

    def compute(self, reward_batch: torch.Tensor,
                q_std: torch.Tensor,
                reg_norm_current: torch.Tensor = None,
                reg_norm_target: torch.Tensor = None) -> float:
        """
        reward_batch: batch of rewards from current update [batch, 1]
        q_std:        ensemble Q-value std [batch] or scalar
        reg_norm_*:   optional, kept for backward compat
        """
        signals = []

        # Signal 1: Reward z-score (detects mean-shift in reward distribution)
        batch_reward_mean = reward_batch.mean().item()
        self.reward_history.append(batch_reward_mean)
        if len(self.reward_history) > self.reward_window:
            self.reward_history = self.reward_history[-self.reward_window:]

        # Update reward EMA
        self.reward_ema = 0.9 * self.reward_ema + 0.1 * batch_reward_mean
        deviation = (batch_reward_mean - self.reward_ema) ** 2
        self.reward_var_ema = 0.9 * self.reward_var_ema + 0.1 * deviation
        reward_std = max(self.reward_var_ema ** 0.5, 1e-6)
        reward_zscore = abs(batch_reward_mean - self.reward_ema) / reward_std
        signals.append(reward_zscore)

        # Signal 2: Q-std spike (detects epistemic uncertainty jumps)
        current_q_std = q_std.mean().item() if isinstance(q_std, torch.Tensor) else q_std
        if self.prev_q_std is not None and self.prev_q_std > 1e-6:
            q_std_change = abs(current_q_std - self.prev_q_std) / self.prev_q_std
            signals.append(q_std_change)
        self.prev_q_std = current_q_std

        # Signal 3: reg_norm divergence (original, as a weak supplement)
        if reg_norm_current is not None and reg_norm_target is not None:
            reg_raw = (reg_norm_current - reg_norm_target).abs().mean().item()
            signals.append(reg_raw * 0.5)  # downweight since it's weak

        # Combine signals (max-pooling: any spike triggers surprise)
        raw_surprise = max(signals) if signals else 0.0

        self.ema_surprise = self.ema_alpha * raw_surprise + (1 - self.ema_alpha) * self.ema_surprise
        return self.ema_surprise
