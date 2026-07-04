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
    # BAPR actor objective:
    #   mean      = ensemble mean only (default; v1 survived Ant/HalfCheetah)
    #   gated_lcb = ensemble mean + beta * gate * std (diagnostic; hurt Ant/HC)
    #   reg_gated_lcb = LCB only while RE-SAC regularizer gate is active
    #   qstd_gated_lcb = LCB only under an actor-specific q_std/q_ratio gate
    #   lcb       = ensemble mean + beta_eff * std (diagnostic; too conservative)
    #   ucb       = ensemble mean + abs(beta_eff) * std (optimistic ablation)
    actor_objective: str = "mean"
    beta_ood: float = 0.01  # OOD regularization weight
    beta_bc: float = 0.001  # behavior cloning weight
    weight_reg: float = 0.01  # critic regularization weight
    use_ema_eval: bool = False  # evaluate EMA policy only when explicitly enabled
    use_ema_rollout: bool = False  # collect training rollouts with EMA policy when explicitly enabled
    ema_rollout_start_iter: int = 0
    ema_rollout_require_reg_latched: bool = False

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
    surprise_reward_weight: float = 0.5
    surprise_q_weight: float = 0.3
    surprise_reg_weight: float = 0.2

    # ── Minimum BAPR redesign (joint regime belief b(h, z)) ──────────────
    # Replaces scalar BOCD ρ(h) with joint b(h, z) over (run-length, regime).
    # Toggle with use_regime_belief=True; legacy scalar BOCD stays the default.
    use_regime_belief: bool = False
    num_regimes: int = 4              # K — regime cluster count
    # Critic target operator: "independent" preserves ensemble disagreement.
    # "min" is kept as an ablation/legacy mode; it makes every Q_i chase the
    # same pessimistic target and was a confound in BAPR vs ESCP comparisons.
    critic_target_mode: str = "independent"
    # BAPR-only stability guard. 0 disables clipping.
    bapr_grad_clip_norm: float = 10.0
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
    oracle_reset_on_switch: bool = False

    # BAPR redesign switch.
    #   gate   : do not feed BOCD/belief into actor/critic by default; use a
    #            bounded surprise gate only for recent replay and optional
    #            gated_lcb actor risk. This is the new production path.
    #   legacy : old BOCD run-length path (belief-conditioned Q/π + adaptive β).
    bapr_adaptation_mode: str = "gate"
    bapr_gate_warmup_iters: int = 10
    bapr_surprise_threshold: float = 0.10
    bapr_gate_gain: float = 1.0
    bapr_gate_ema_alpha: float = 0.3
    bapr_gate_max: float = 0.35
    bapr_recent_frac_cap: float = 0.35
    # Optional recent-replay safety gate. Recent replay helped low-disagreement
    # HalfCheetah/Ant but hurt high-disagreement Hopper/Walker in v7; this gate
    # keeps recent replay only when critic disagreement is small on the previous
    # rollout.
    bapr_recent_disagreement_gate: bool = False
    bapr_recent_open_if_reg_latched: bool = False
    bapr_recent_frac_floor: float = 0.0
    # Historical behavior only applied the floor when the surprise gate was
    # already nonzero. Enable this for the recovery-controller path where the
    # floor is meant to be a true low-rate recency bias.
    bapr_recent_true_floor: bool = False
    # How the recent replay floor is applied:
    #   always           - keep the old behavior: floor remains active even
    #                      when disagreement closes normal recent replay.
    #   low_disagreement - apply the floor only when the disagreement gate
    #                      would already allow recent replay.
    #   reg_latched      - apply the floor only after the regularizer latch fires.
    #   ratio_schedule   - use a larger floor under low critic disagreement, a
    #                      smaller floor under moderate disagreement, and shut
    #                      the floor off under extreme disagreement.
    bapr_recent_floor_mode: str = "always"
    bapr_recent_floor_ratio_low: float = 0.02
    bapr_recent_floor_ratio_high: float = 0.04
    bapr_recent_floor_mid_frac: float = 0.015
    bapr_recent_floor_extreme_frac: float = 0.0
    bapr_recent_qstd_threshold: float = 5.0
    bapr_recent_qstd_ratio_threshold: float = 0.02
    # Optional RE-SAC regularizer safety gate. In v6, regularization helped
    # high-disagreement Hopper/Walker but hurt low-disagreement HalfCheetah/Ant.
    bapr_reg_disagreement_gate: bool = False
    bapr_reg_warmup_iters: int = 0
    bapr_reg_max_iters: int = 0
    bapr_reg_latch: bool = False
    bapr_reg_require_both: bool = False
    bapr_reg_qstd_threshold: float = 5.0
    bapr_reg_qstd_ratio_threshold: float = 0.02
    # Late safety valve: if critic disagreement explodes after reg_max_iters,
    # briefly re-enable the RE-SAC regularizer. This targets Ant-like late
    # critic blowups without touching low-disagreement HalfCheetah runs.
    bapr_reg_emergency_gate: bool = False
    bapr_reg_emergency_scale: float = 1.0
    bapr_reg_emergency_qstd_threshold: float = 20.0
    bapr_reg_emergency_qstd_ratio_threshold: float = 0.05
    # Permanent latch scale. This keeps low-disagreement environments on the
    # base setting while letting high-disagreement Hopper/Walker-style runs use
    # a softer RE-SAC regularizer after the safety latch fires.
    bapr_reg_latched_scale: float = 1.0
    # Optional collapse guard for latched regularization. If the regularizer is
    # on but smoothed train/eval return has severely drawn down, temporarily
    # reduce the RE-SAC regularization scale. This targets Ant-like late critic
    # over-regularization without changing the default path.
    bapr_reg_perf_collapse_gate: bool = False
    bapr_reg_perf_collapse_drop_frac: float = 1.0
    bapr_reg_perf_collapse_scale: float = 0.0
    bapr_reg_perf_collapse_warmup_iters: int = 100

    # Second-layer algorithm controller. This does not identify the environment
    # name; it reacts to normalized training state. The intended first use is the
    # Ant-like pattern: large reward drawdown while ensemble disagreement is
    # already low, where more always-on RE-SAC regularization can worsen recovery.
    bapr_controller_mode: str = "off"
    bapr_controller_warmup_iters: int = 120
    bapr_controller_min_peak: float = 100.0
    bapr_controller_drop_frac: float = 0.9
    bapr_controller_drop_ramp: float = 0.4
    bapr_controller_low_qstd_ratio: float = 0.02
    bapr_controller_low_qstd_threshold: float = 0.0
    bapr_controller_latch: bool = False
    bapr_controller_release_drop_frac: float = 0.45
    bapr_controller_latched_signal: float = 1.0
    bapr_controller_reg_multiplier: float = 0.2
    bapr_controller_recent_multiplier: float = 1.0
    bapr_controller_recent_add_frac: float = 0.0
    bapr_controller_lcb_multiplier: float = 1.0
    bapr_controller_actor_update_multiplier: float = 1.0

    # Actor-only LCB gate. This is intentionally separate from the RE-SAC
    # regularizer gate because Hopper wants regularization but not actor LCB,
    # while Walker often needs a conservative actor under extreme disagreement.
    bapr_actor_lcb_qstd_gate: bool = False
    bapr_actor_lcb_qstd_threshold: float = 270.0
    bapr_actor_lcb_qstd_ratio_threshold: float = 0.04
    bapr_actor_lcb_require_both: bool = True
    bapr_actor_lcb_scale: float = 1.0
    # Recovery-controller guard for actor LCB. Critic disagreement alone caused
    # false positives on Hopper; require an actual performance drawdown before
    # conservative actor updates are enabled.
    bapr_actor_lcb_perf_gate: bool = False
    bapr_actor_lcb_perf_warmup_iters: int = 100
    bapr_actor_lcb_perf_drop_frac: float = 0.25
    bapr_actor_lcb_perf_min_peak: float = 100.0
    bapr_actor_lcb_perf_ema_alpha: float = 0.20

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
