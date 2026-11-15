"""Matched-runtime comparison of frozen compensation and regularized RE-SAC."""
from importlib.metadata import version
from pathlib import Path

from jax_experiments.analysis import regime_polarity_action_compensation_power_confirmation_v32 as old

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "v39-matched-runtime-regularized-comparison"
SEEDS = (109003, 109023, 109043, 109063, 109083)
TRAIN_ARMS = ("source", "reference", "sac", "escp", "resac_zero", "resac_full")
EVAL_ARMS = ("source", "sac", "escp", "resac_zero", "resac_full", "no_compensation", "oracle", "bapr")
RUNTIME = {"jax": "0.4.38", "brax": "0.10.5", "flax": "0.10.4", "mujoco": "3.1.6"}
RUN_ROOT = ROOT / "jax_experiments/results_regularized_comparison_v39"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_regularized_comparison_v39"
PREREG = ROOT / "reports/regularized_comparison_v39_preregistration_2026-10-01.md"
SMOKE = BUNDLE_ROOT / "smoke/ready.json"
ANALYSIS = RUN_ROOT / "analysis/analysis.json"
ESTIMATOR = ROOT / "reproducibility/v32/estimator"
CPU_NODES = tuple(f"node00{i}" for i in range(1, 8))
GPU_PYTHON = "/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python"
CPU_PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/bapr-v34-eval-0438/bin/python"

# These are the unchanged V32 evaluation windows, not a new environment.
ENV = old.ENV
MODES = old.MODES
DWELL_STEPS = old.DWELL_STEPS
MAX_EPISODE_STEPS = old.MAX_EPISODE_STEPS
STATIONARY_EPISODES_PER_MODE = old.STATIONARY_EPISODES_PER_MODE
SWITCHING_EPISODES = old.SWITCHING_EPISODES
STATIONARY_HOLDOUT_EVENT_SEEDS = old.STATIONARY_HOLDOUT_EVENT_SEEDS
SWITCHING_EVENT_SEEDS = old.SWITCHING_EVENT_SEEDS
SWITCHING_SCHEDULES = old.SWITCHING_SCHEDULES
require_switching_event_seed = old.require_switching_event_seed
switching_sequence = old.switching_sequence
mode_gain_vectors = old.mode_gain_vectors
compensate_action = old.compensate_action


def run_dir(arm, seed):
    if arm not in TRAIN_ARMS or seed not in SEEDS:
        raise ValueError(f"unknown comparison run {arm}/{seed}")
    return RUN_ROOT / arm / f"seed_{seed}"


def bundle_dir(arm, seed):
    run_dir(arm, seed)
    return BUNDLE_ROOT / arm / f"seed_{seed}"


def audit_path(seed):
    if seed not in SEEDS:
        raise ValueError(f"unknown comparison seed {seed}")
    return RUN_ROOT / "audits" / f"seed_{seed}" / "audit.json"


def identity(arm, seed):
    run_dir(arm, seed)
    return {"protocol": PROTOCOL, "arm": arm, "seed": seed}


def budget(arm):
    iterations = 1400 if arm == "source" else 2100
    return {"next_iteration": iterations, "total_steps": iterations * 4000,
            "update_count": iterations * 250}


def check_runtime():
    actual = {name: version(name) for name in RUNTIME}
    if actual != RUNTIME:
        raise ValueError(f"comparison runtime differs: expected={RUNTIME}, actual={actual}")
    return actual
