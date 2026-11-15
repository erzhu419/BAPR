"""Three-arm development intervention; no BAPR tuning or historical pooling."""
from pathlib import Path

from jax_experiments.analysis import regularized_comparison_protocol_v39 as previous

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "v40-resac-norm-causality"
SEEDS = (110003, 110023, 110043)
ARMS = ("per_head", "shared", "no_shift")
RUN_ROOT = ROOT / "jax_experiments/results_resac_norm_causality_v40"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_resac_norm_causality_v40"
PREREG = ROOT / "reports/resac_norm_causality_v40_preregistration_2026-10-02.md"
SMOKE = BUNDLE_ROOT / "smoke/ready.json"
ANALYSIS = RUN_ROOT / "analysis/analysis.json"
RUNTIME, check_runtime = previous.RUNTIME, previous.check_runtime
GPU_PYTHON, CPU_PYTHON = previous.GPU_PYTHON, previous.CPU_PYTHON
GPU_NODES = ("jtl110gpu", "jtl110gpu2", "node007", "jtl311linux")
CPU_NODES = previous.CPU_NODES
ENV, MODES = previous.ENV, previous.MODES
DWELL_STEPS, MAX_EPISODE_STEPS = previous.DWELL_STEPS, previous.MAX_EPISODE_STEPS
STATIONARY_EPISODES_PER_MODE = previous.STATIONARY_EPISODES_PER_MODE
SWITCHING_EPISODES = previous.SWITCHING_EPISODES
STATIONARY_HOLDOUT_EVENT_SEEDS = previous.STATIONARY_HOLDOUT_EVENT_SEEDS
SWITCHING_EVENT_SEEDS = previous.SWITCHING_EVENT_SEEDS
SWITCHING_SCHEDULES = previous.SWITCHING_SCHEDULES
switching_sequence = previous.switching_sequence
require_switching_event_seed = previous.require_switching_event_seed
NORM_DEFINITION = "all-kernels-hidden-biases-excludes-output-bias"
FIXED_CONFIG = dict(algo="resac", ensemble_size=5, hidden_dim=256, lr=0.0003,
                    gamma=0.99, tau=0.005, beta=-2.0, beta_ood=0.01,
                    resac_beta_bc=0.001, resac_independent_ratio=1.0,
                    resac_critic_actor_ratio=1, resac_clip_norm=0.0,
                    resac_anchor_lambda=0.0, resac_adaptive_beta=False,
                    use_ema_eval=False, use_ema_rollout=False, max_iters=2100,
                    samples_per_iter=4000, updates_per_iter=250)


def budget():
    return {"next_iteration": 2100, "total_steps": 8400000, "update_count": 525000}


def settings(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown norm intervention {arm}")
    return {**FIXED_CONFIG, "weight_reg": 0.0 if arm == "no_shift" else 0.01,
            "resac_reg_norm_mode": "shared" if arm == "shared" else "per_head"}


def identity(arm, seed):
    settings(arm)
    if seed not in SEEDS:
        raise ValueError(f"unknown development seed {seed}")
    return {"protocol": PROTOCOL, "arm": arm, "seed": seed}


def run_dir(arm, seed):
    identity(arm, seed)
    return RUN_ROOT / arm / f"seed_{seed}"


def bundle_dir(arm, seed):
    identity(arm, seed)
    return BUNDLE_ROOT / arm / f"seed_{seed}"


def audit_path(seed):
    identity(ARMS[0], seed)
    return RUN_ROOT / "audits" / f"seed_{seed}" / "audit.json"


def validate(metadata, arm, seed):
    if (metadata["identity"] != identity(arm, seed) or metadata["runtime_versions"] != RUNTIME
            or metadata["norm_definition"] != NORM_DEFINITION or metadata["budget"] != budget()):
        raise ValueError("norm-causality identity/runtime/budget differs")
    config = metadata["config"]
    if config["seed"] != seed or any(config[key] != value for key, value in settings(arm).items()):
        raise ValueError("norm-causality treatment or fixed controls differ")
