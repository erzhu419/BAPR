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

    # Env type selector — see jax_experiments/envs/discrete_mode_env.py
    #   'continuous': original ESCP-style continuous task family (uniform log_scale)
    #   'discrete_mode': K=4 semantic modes, exp dwell, random next-mode (BAPR sweet spot)
    env_type: str = "continuous"
    # discrete_mode-specific:
    discrete_mean_dwell_iters: int = 60       # dwell time per mode in iters
    discrete_dwell_distribution: str = "exponential"  # 'exponential' or 'fixed'
    discrete_reward_shaping: bool = False     # Property D explicit penalty

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
    bapr_warmup_iters: int = 100    # P0: pure RESAC (λ_w=0) for early iters
                                     # — BOCD needs valid Q-std + surprise statistics
                                     # before it can gate policy updates meaningfully
    rnn_fix_length: int = 16  # history window for context (FC mode = no RNN)

    max_run_length: int = 20
    hazard_rate: float = 0.05       # legacy BOCD (scalar ρ(h)) — preserved for backcompat
    base_variance: float = 0.1      # variance at h=0 for BOCD likelihood
    variance_growth: float = 0.05   # variance grows with run length
    surprise_ema_alpha: float = 0.3

    # ── Minimum BAPR redesign (joint regime belief b(h, z)) ──────────────
    # Replaces scalar BOCD ρ(h) with joint b(h, z) over (run-length, regime).
    # Toggle with use_regime_belief=True; legacy scalar BOCD stays the default.
    use_regime_belief: bool = False
    num_regimes: int = 4              # K — regime cluster count
    # Critic target operator: "min" (LCB target, BAPR original — collapses
    # ensemble) or "independent" (each Q_i learns its own target Q_i,
    # ESCP-like, preserves ensemble disagreement). GPT-5.5 advice #4: the
    # "min" choice was a confound in the BAPR vs ESCP comparison.
    critic_target_mode: str = "min"   # legacy default for backward compat
    # GPT-5.5 advice #2 toggle: when True, off-policy critic update uses the
    # belief stored at rollout time (replay's "belief" field). When False, it
    # broadcasts the current iter's belief — the v15 (pre-Phase 2) behavior.
    # Toggle exists so we can ablate which advice item helps which env.
    use_per_transition_belief: bool = True
    # Per-CHUNK observation (rollout split into chunks_per_iter pieces).
    # 4000-step rollout / 16 chunks = 250 env steps per chunk → reward + q-std
    # signal per chunk. Warmup samples: 100 iter × 16 chunks = 1600 obs for
    # k-means seeding (vs 100 obs if per-iter).
    regime_chunks_per_iter: int = 16
    regime_warmup_samples: int = 1600  # 100 iter × 16 chunks
    regime_ewma_alpha: float = 0.05   # EMA rate for online (mu_z, var_z) refresh
    regime_obs_dim: int = 3           # 3-channel y: (r_resid, q_std_spike, td_resid)
    # Per-chunk hazard rate. discrete_mode env mean dwell = 60 iter × 16 chunks
    # = 960 chunks → per-chunk hazard ≈ 1/960 ≈ 0.001. Override per env.
    regime_hazard_rate: float = 0.001
    # v15: belief-conditioned Q. Concat full BOCD posterior ρ(h) (max_run_length
    # dim) to context vector e fed to actor + critic. When False, BAPR uses
    # only the scalar λ_w pathway (= v14 behavior).
    belief_conditioned: bool = True
    # Change 2 (GPT-5.5 v2): bounded suffix of recent rollout fed to BOCD.
    # Replay batches mix old regimes — using rollout suffix gives BOCD the
    # current regime's evidence. 1024 ≈ 1 episode × ¼ for HC.
    surprise_window: int = 1024
    # Change 3 (GPT-5.5 v2): when BOCD detects switch, oversample recent
    # transitions in replay. Window = how far back "recent" reaches.
    recent_replay_window: int = 50_000
    # penalty_decay_rate removed: λ_w now = effective_window / H (see bapr.py)
    penalty_scale: float = 2.0      # P1: reduced from 5.0 to avoid over-conservatism
                                     # β_eff = β_base - λ_w × penalty_scale
                                     # With β_base=-2, λ_w∈[0,1]: β_eff ∈ [-4, -2]
                                     # (previously [-7, -2], too aggressive early)
    belief_warmup_steps: int = 50

    # EMA Policy (Polyak-averaged actor for stable evaluation)
    ema_tau: float = 0.005          # EMA smoothing coefficient (same as target critic tau)

    # Performance Gating — DISABLED (bugged; forced λ_w=0.5 early in training
    # when eval is naturally noisy, over-conservatively freezing BAPR policy).
    # Kept for ablation compatibility; always False for production runs.
    perf_gate_enabled: bool = False
    perf_gate_threshold: float = 0.1
    perf_gate_lookback: int = 3

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
