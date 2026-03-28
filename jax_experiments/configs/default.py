"""Hyperparameter configuration dataclass for JAX-based RL experiments."""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Environment
    env_name: str = "Hopper-v2"
    brax_backend: str = "spring"  # 'spring' (fast, 1.5s/4K steps) or 'generalized' (accurate, 14s/4K steps)
    varying_params: List[str] = field(default_factory=lambda: ["gravity"])
    log_scale_limit: float = 3.0
    ood_change_range: float = 4.0
    task_num: int = 40
    test_task_num: int = 40
    changing_period: int = 20000  # task switches every THIS many env steps (~5 iters)
    changing_interval: int = 4000  # align with samples_per_iter (one check per rollout)

    # Algorithm
    algo: str = "resac"  # resac | escp | bapr
    seed: int = 8
    gamma: float = 0.99
    tau: float = 0.005  # soft target update
    alpha: float = 0.2  # SAC entropy weight (initial)
    auto_alpha: bool = True
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 1_000_000
    hidden_dim: int = 256
    max_iters: int = 2000
    samples_per_iter: int = 4000  # env steps collected per iteration
    updates_per_iter: int = 250  # gradient steps per iteration
    start_train_steps: int = 10_000  # random exploration before training
    max_episode_steps: int = 1000

    # Ensemble (RE-SAC / ESCP / BAPR)
    ensemble_size: int = 10
    beta: float = -2.0  # LCB coefficient for policy
    beta_ood: float = 0.01  # OOD regularization weight
    beta_bc: float = 0.001  # behavior cloning weight
    weight_reg: float = 0.01  # critic regularization weight

    # Context / ESCP
    ep_dim: int = 2  # context embedding dimension
    repr_loss_weight: float = 1.0
    rbf_radius: float = 2.0   # tuned for tanh EP embeddings (sq_dist mean≈0.69); original ESCP=80 was for raw physics params
    consistency_loss_weight: float = 50.0
    diversity_loss_weight: float = 0.025
    context_warmup_iters: int = 50  # iterations before injecting context
    rnn_fix_length: int = 16  # history window for context (FC mode = no RNN)

    max_run_length: int = 20
    hazard_rate: float = 0.05
    base_variance: float = 0.1      # variance at h=0 for BOCD likelihood
    variance_growth: float = 0.05   # variance grows with run length
    surprise_ema_alpha: float = 0.3
    # penalty_decay_rate removed: λ_w now = effective_window / H (see bapr.py)
    penalty_scale: float = 5.0      # β_eff = β_base - λ_w × penalty_scale
    belief_warmup_steps: int = 50

    # Logging
    save_root: str = "jax_experiments/results"
    run_name: str = ""
    log_interval: int = 5  # eval every N iterations (saves ~10s per skipped eval)
    save_interval: int = 50  # save model every N iterations
    eval_episodes: int = 5

    # Supported environments
    ENVS: List[str] = field(default_factory=lambda: [
        "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"
    ])
