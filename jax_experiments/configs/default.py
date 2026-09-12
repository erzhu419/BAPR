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
    task_scale_distribution: str = "exp"  # exp: e**u; pow1p5: ESCP-style 1.5**u
    task_seed_salt: int = 0
    ood_change_range: float = 4.0
    task_num: int = 40
    test_task_num: int = 40
    reserved_test_task_num: int = 0
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
    # stochastic_mode keeps physics fixed during each regime and samples only
    # the hidden actuator transition kernel per step. packet_loss,
    # burst_torque, structured_channel, and actuator_polarity keep morphology
    # fixed; the latter two change spatial actuator regimes.
    stochastic_mode_family: str = "mean_variance"
    stochastic_mode_dwell_steps: int = 500
    stochastic_mode_dwell_distribution: str = "fixed"
    # -1 keeps the normal switching process. Values 0-3 lock training to one
    # persistent mode; strict evaluation may explicitly re-enable switching.
    stochastic_mode_fixed_id: int = -1

    # Equal-budget control-headroom diagnostic. Both arms use the same
    # conditioned actor/critic; robust receives zero context and oracle receives
    # the true persistent stochastic-mode id.
    regime_context_source: str = "robust"  # robust | oracle

    # Algorithm
    algo: str = "resac"  # resac | escp | bapr
    seed: int = 8
    gamma: float = 0.99
    tau: float = 0.005  # soft target update
    alpha: float = 0.2  # SAC entropy weight (initial)
    auto_alpha: bool = True
    lr: float = 3e-4
    clip_norm: float = 1.0
    batch_size: int = 256
    replay_size: int = 1_000_000
    hidden_dim: int = 256
    max_iters: int = 2000
    samples_per_iter: int = 4000  # env steps collected per iteration
    updates_per_iter: int = 250  # gradient steps per iteration
    start_train_steps: int = 10_000  # random exploration before training
    # Optional paper-fidelity warmup collected before iteration 0. When set,
    # start_train_steps must match so the first main-loop rollout uses policy.
    initial_random_steps: int = 0
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
    critic_actor_ratio: int = 2  # RE-SAC critic updates per actor update
    # RE-SAC compatibility controls.  Defaults preserve the legacy BAPR fork;
    # paper-B0 diagnostics opt into the blended target/EMA/anchor explicitly.
    resac_independent_ratio: float = 1.0
    resac_anchor_lambda: float = 0.0
    resac_adaptive_beta: bool = False
    resac_beta_start: float = -2.0
    resac_beta_end: float = -2.0
    resac_beta_warmup: float = 0.2
    # RESAC historically updated its actor every critic step and did not use
    # the generic BAPR BC/gradient-clipping controls.  Keep those semantics
    # explicit instead of inheriting similarly named controller defaults.
    resac_critic_actor_ratio: int = 1
    resac_beta_bc: float = 0.0
    resac_clip_norm: float = 0.0
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
    rmdm_max_tasks: int = 64  # continuous gravity protocol uses 40 tasks
    context_warmup_iters: int = 50  # iterations before injecting context
    # The old JAX ESCP approximation used independent targets, an LCB actor,
    # and trained its state-only RMDM encoder immediately.  Keep that as the
    # checkpoint-compatible default; paper-core audits select twin_min and a
    # delayed representation phase explicitly.
    escp_target_mode: str = "independent"  # independent | twin_min
    escp_actor_mode: str = "lcb"  # lcb | twin_min
    escp_context_min_steps: int = 0
    escp_context_min_tasks: int = 0
    escp_alpha_max: float = -1.0  # <=0 disables the original alpha<=1 clamp
    escp_finite_guard: bool = True
    # ``state_mlp`` preserves every historical JAX checkpoint. ``recurrent``
    # reproduces the released ESCP environment probe: (s_t, a_{t-1}) ->
    # FC(128) -> GRU(64) -> tanh context over a reset-aware fixed history.
    escp_context_mode: str = "state_mlp"  # state_mlp | recurrent
    escp_history_length: int = 16
    # Negative values inherit ``lr`` so legacy configurations stay unchanged.
    escp_policy_lr: float = -1.0
    escp_critic_lr: float = -1.0
    escp_context_lr: float = -1.0
    escp_alpha_lr: float = -1.0
    escp_target_entropy_ratio: float = 1.0
    escp_bottleneck_sigma: float = 0.0
    escp_prototype_tau: float = 0.995
    bapr_warmup_iters: int = 100    # P0: pure RESAC (λ_w=0) for early iters
                                     # — BOCD needs valid Q-std + surprise statistics
                                     # before it can gate policy updates meaningfully
    rnn_fix_length: int = 16  # history window for context (FC mode = no RNN)

    # BAPR-v2: causal transition context + robust residual policy. This is a
    # separate implementation path; legacy BAPR fields below remain untouched.
    bapr_v2_mode: str = "robust"  # robust | oracle | supervised | hybrid
    bapr_v2_latent_dim: int = 4
    # legacy_exp reproduces v82's exp-scale normalization. New experiments
    # must opt into task_distribution to match pow1p5 task generation.
    bapr_v2_latent_scale_mode: str = "legacy_exp"
    # stored reproduces v82 joint training, where replay carries the context
    # emitted by the encoder at collection time. oracle_task is the v84
    # teacher-student path: actor/critic updates reconstruct context from the
    # replay task id; learned rollouts and evaluation still use the encoder.
    bapr_v2_policy_context_source: str = "stored"  # stored | oracle_task
    # joint reproduces v82/v83. teacher_student runs a context-free robust
    # pretrain, an oracle-context residual teacher, then a frozen-controller
    # student phase. constrained_deploy adds a learned-latent deployment phase
    # with paired robust/teacher safety supervision.
    bapr_v2_training_schedule: str = "joint"  # joint | teacher_student | constrained_deploy
    bapr_v2_base_pretrain_iters: int = 0
    bapr_v2_teacher_iters: int = 0
    bapr_v2_student_iters: int = 0
    # Copy a converged robust base into a full conditioned branch exactly when
    # teacher training begins. This avoids a random-policy discontinuity.
    bapr_v2_warmstart_conditioned: bool = False
    bapr_v2_context_hidden_dim: int = 64
    bapr_v2_context_length: int = 64
    bapr_v2_context_chunks: int = 8
    bapr_v2_context_burnin: int = 16
    bapr_v2_context_lr: float = 3e-4
    bapr_v2_context_grad_clip: float = 10.0
    bapr_v2_predictive_weight: float = 1.0
    bapr_v2_supervised_weight: float = 2.0
    bapr_v2_hybrid_supervised_weight: float = 0.2
    bapr_v2_temporal_weight: float = 0.01
    bapr_v2_reward_scale: float = 10.0
    bapr_v2_delta_scale: float = 1.0
    bapr_v2_min_history: int = 32
    bapr_v2_gate_error_threshold: float = 0.20
    bapr_v2_gate_error_scale: float = 0.25
    bapr_v2_error_ema_alpha: float = 0.10
    bapr_v2_reset_temperature: float = 0.10
    bapr_v2_use_fallback: bool = True
    # categorical_expert uses one independent actor per posterior coordinate.
    bapr_v2_policy_mode: str = "residual"  # residual | direct | gated_direct | expert | categorical_expert
    bapr_v2_num_experts: int = 5
    bapr_v2_residual_delta: float = 0.25
    bapr_v2_policy_gate_init: float = 0.0
    bapr_v2_action_deviation_weight: float = 0.0
    # Constrained deployment: train on causal learned contexts and supervise
    # the policy gate from paired robust/teacher return and termination audits.
    bapr_v2_switch_rollout_steps: int = 0
    bapr_v2_freeze_gate_in_teacher: bool = False
    bapr_v2_paired_calibration_episodes: int = 0
    bapr_v2_paired_gain_margin: float = 0.02
    bapr_v2_paired_gain_temperature: float = 0.05
    bapr_v2_paired_risk_tolerance: float = 0.0
    bapr_v2_paired_risk_temperature: float = 0.10
    bapr_v2_paired_return_scale: float = 100.0
    bapr_v2_gate_supervision_weight: float = 0.0
    bapr_v2_unsafe_deviation_weight: float = 0.0
    bapr_v2_context_dropout: float = 0.20
    bapr_v2_base_aux_weight: float = 0.25
    # At deployment, use the adaptive action only when the ensemble LCB of
    # Q(s, a_adapt, z) - Q(s, a_base, z) exceeds this margin.
    bapr_v2_advantage_gate: bool = False
    bapr_v2_advantage_margin: float = 0.0
    bapr_v2_advantage_lcb_scale: float = 1.0
    # Conservative residual training. The critic compares the adaptive and
    # exact base action under the same mode context. The smooth constraint
    # pushes their relative LCB above the margin; the optional update filter
    # rolls back an actor step that degrades any represented mode.
    bapr_v2_train_advantage_constraint: bool = False
    bapr_v2_train_advantage_lcb_scale: float = 1.0
    bapr_v2_train_advantage_margin: float = 0.0
    bapr_v2_train_advantage_temperature: float = 0.01
    bapr_v2_train_advantage_weight: float = 1.0
    bapr_v2_train_update_filter: bool = False
    bapr_v2_train_update_tolerance: float = 0.005
    bapr_v2_train_update_floor: float = -0.01
    bapr_v2_actor_objective: str = "mean"  # mean | lcb
    # Keep legacy independent ensemble bootstrapping by default. Diagnostic
    # continuations may match the robust SAC lower-bound target explicitly.
    bapr_v2_critic_target_mode: str = "independent"  # independent | min
    bapr_v2_freeze_alpha: bool = False
    bapr_v2_beta_ood: float = 0.0
    # Optional common normalized target shift. MuJoCo jobs keep this at zero;
    # the positive sign is retained for later bus-domain calibration.
    bapr_v2_reg_weight: float = 0.0
    bapr_v2_reg_norm_ref: float = 20_000.0

    # BAPR-v3: probabilistic mode evidence over persistent stochastic regimes.
    # The V2 actor/critic schedule is reused, but the context model is separate.
    bapr_v3_likelihood: str = "probabilistic"  # point | probabilistic
    bapr_v3_context_ensemble_size: int = 5
    bapr_v3_hazard_rate: float = 0.002
    bapr_v3_evidence_scale: float = 4.0
    bapr_v3_fixed_variance: float = 0.02
    bapr_v3_logvar_min: float = -6.0
    bapr_v3_logvar_max: float = 1.0
    # legacy_state reproduces the failed state-dependent log-variance head.
    # mode_calibrated learns one bounded residual variance per mode/output via
    # NLL. The empirical variants own variance with an EMA of residual moments.
    # inverse_empirical predicts the executed action from (s, s') using clean
    # mode-0 data, then models mode-specific commanded/executed action residuals.
    bapr_v3_variance_model: str = "legacy_state"
    bapr_v3_variance_floor: float = 1e-4
    bapr_v3_variance_ceiling: float = 0.25
    bapr_v3_variance_ema: float = 0.05
    bapr_v3_mean_loss_weight: float = 1.0
    bapr_v3_variance_loss_weight: float = 0.1
    bapr_v3_variance_prior_weight: float = 0.001
    bapr_v3_instant_classifier_weight: float = 1.0
    bapr_v3_evidence_clip: float = 0.0
    bapr_v3_surprise_threshold: float = 2.0
    bapr_v3_surprise_scale: float = 1.0
    bapr_v3_freeze_teacher_after_teacher: bool = False
    # A robust estimator rollout avoids the feedback loop where a wrong mode
    # posterior changes the policy and therefore changes its own evidence.
    bapr_v3_estimator_rollout_source: str = "learned"  # learned | robust
    # A teacher-only checkpoint may contain an intentionally untrained or
    # architecture-incompatible context model. Preserve the controller and
    # replay state while starting the learned estimator from a clean state.
    bapr_v3_reset_context_on_resume: bool = False
    bapr_v3_eval_context_ladder: bool = False

    # Minimal persistent-regime path. The robust actor keeps training while a
    # causal heteroscedastic estimator is calibrated on robust rollouts; only
    # then is a zero-initialized shared residual trained with learned or oracle
    # posterior context. Legacy BAPR schedules remain unchanged.
    bapr_regime_inference_iters: int = 200
    bapr_regime_adaptation_source: str = "learned"  # learned | oracle
    bapr_regime_freeze_context_after_inference: bool = True
    bapr_regime_clear_replay_on_adaptation: bool = True
    bapr_regime_zero_residual_init: bool = True
    bapr_regime_advantage_fallback: bool = True

    # BAPR-v4: a shared FiLM actor is trained jointly under robust and
    # privileged persistent-option rollouts. At deployment, a CUSUM posterior
    # can change the active option only at fixed semi-Markov boundaries.
    bapr_v4_option_hold_steps: int = 64
    bapr_v4_option_confidence_threshold: float = 0.80
    bapr_v4_option_margin_threshold: float = 0.05
    bapr_v4_option_hysteresis_margin: float = 0.02
    bapr_v4_posterior_decay: float = 1.0
    bapr_v4_cusum_threshold: float = 4.0
    bapr_v4_cusum_drift: float = 0.25
    # One robust iteration followed by three oracle-option iterations. This
    # trains the fallback and all options inside the same replay distribution.
    bapr_v4_training_source_period: int = 4
    bapr_v4_training_robust_slots: int = 1
    # Optional validated estimator initialization. Empty paths mean fresh
    # context parameters. Checkpoint resume always supersedes this bootstrap.
    bapr_v4_context_bootstrap_model: str = ""
    bapr_v4_context_bootstrap_manifest: str = ""

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
    # In soft_recovery mode the instantaneous controller signal is allowed to
    # decay instead of acting as a binary emergency latch.
    bapr_controller_soft_decay: float = 0.92
    bapr_controller_soft_min_signal: float = 0.01
    bapr_controller_low_qstd_ratio: float = 0.02
    bapr_controller_low_qstd_threshold: float = 0.0
    bapr_controller_latch: bool = False
    bapr_controller_release_drop_frac: float = 0.45
    # Optional safety release for recovery latch. 0 disables the cap. When the
    # cap fires, the controller stays suppressed until drawdown recovers below
    # bapr_controller_release_drop_frac, preventing immediate re-latching.
    bapr_controller_max_active_iters: int = 0
    # v79: explicit controller exit/re-entry controls. A cooldown lets recovery
    # return the algorithm to normal training for a window instead of staying
    # active for the whole tail whenever drawdown remains high.
    bapr_controller_min_active_iters: int = 0
    bapr_controller_exit_cooldown_iters: int = 0
    bapr_controller_exit_signal_threshold: float = 0.0
    bapr_controller_exit_improve_frac: float = 0.0
    bapr_controller_exit_drawdown_frac: float = -1.0
    bapr_controller_latched_signal: float = 1.0
    bapr_controller_reg_multiplier: float = 0.2
    # Optional linear recovery of controller multipliers while active. 0 keeps
    # the historical fixed target.
    bapr_controller_reg_recover_iters: int = 0
    bapr_controller_recent_multiplier: float = 1.0
    bapr_controller_recent_add_frac: float = 0.0
    bapr_controller_lcb_multiplier: float = 1.0
    bapr_controller_actor_update_multiplier: float = 1.0
    bapr_controller_actor_recover_iters: int = 0

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

    # v80: conservative residual actor objective. The actor proposes an
    # adaptation action, but the loss evaluates a bounded residual from a
    # slow zero-context EMA policy and gates that residual by conservative
    # Q-advantage. This keeps a stable base policy in charge unless the
    # critic ensemble says the adaptive proposal is worth using.
    bapr_residual_delta: float = 0.25
    bapr_residual_gate_scale: float = 1.0
    bapr_residual_adv_margin: float = 0.0
    bapr_residual_adv_temp: float = 300.0
    bapr_residual_qstd_scale: float = 0.5
    bapr_residual_behavior_weight: float = 0.05
    bapr_residual_action_penalty: float = 0.0

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
    # Optional hard guard for continuation jobs. The value is the next
    # iteration expected after loading, not the saved checkpoint iteration.
    min_resume_iteration: int = -1
    log_interval: int = 5  # eval every N iterations (saves ~10s per skipped eval)
    save_interval: int = 50  # save model every N iterations
    eval_episodes: int = 5
    # stationary: strict-horizon fixed-task eval only, logged as eval_reward
    # full: also logs stationary ID and true-switching online-policy eval
    eval_protocol: str = "stationary"
    eval_switching_episodes: int = 1
    eval_switching_period_steps: int = 500

    # Supported environments
    ENVS: List[str] = field(default_factory=lambda: [
        "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"
    ])
