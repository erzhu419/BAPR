"""Reward-only transfer in the unchanged high-density JAX actuator protocol."""
from jax_experiments.analysis import resac_update_density_protocol_v42 as old

ROOT = old.ROOT
PROTOCOL = "v45-resac-jax-reward-transfer"
SEEDS = (115003, 115023, 115043)
ARMS = ("raw_full", "raw_no_shift", "bus_reward_full", "bus_reward_no_shift")
RUN_ROOT = ROOT / "jax_experiments/results_resac_jax_reward_transfer_v45"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_resac_jax_reward_transfer_v45"
PREREG = ROOT / "reports/resac_jax_reward_transfer_v45_preregistration_2026-10-04.md"
SMOKE = BUNDLE_ROOT / "smoke/ready.json"
ANALYSIS = RUN_ROOT / "analysis/analysis.json"
RUNTIME, check_runtime = old.RUNTIME, old.check_runtime
GPU_PYTHON, CPU_PYTHON = old.GPU_PYTHON, old.CPU_PYTHON
GPU_NODES, CPU_NODES = old.GPU_NODES, old.CPU_NODES
ENV, MODES = old.ENV, old.MODES
DWELL_STEPS, MAX_EPISODE_STEPS = old.DWELL_STEPS, old.MAX_EPISODE_STEPS
STATIONARY_EPISODES_PER_MODE, SWITCHING_EPISODES = old.STATIONARY_EPISODES_PER_MODE, old.SWITCHING_EPISODES
STATIONARY_HOLDOUT_EVENT_SEEDS, SWITCHING_EVENT_SEEDS = old.STATIONARY_HOLDOUT_EVENT_SEEDS, old.SWITCHING_EVENT_SEEDS
SWITCHING_SCHEDULES = old.SWITCHING_SCHEDULES
switching_sequence, require_switching_event_seed = old.switching_sequence, old.require_switching_event_seed
NORM_DEFINITION = old.NORM_DEFINITION


def settings(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown JAX reward-transfer arm {arm}")
    return {**old.settings("high_full"), "weight_reg": 0.0 if arm.endswith("no_shift") else 0.01,
            "resac_reward_transform": "bus_standardized" if arm.startswith("bus_") else "raw"}


def budget(arm):
    settings(arm)
    return old.budget("high_full")


def actor_updates(arm):
    return budget(arm)["update_count"] // settings(arm)["resac_critic_actor_ratio"]


def identity(arm, seed):
    settings(arm)
    if seed not in SEEDS:
        raise ValueError(f"unknown JAX reward-transfer seed {seed}")
    return {"protocol": PROTOCOL, "arm": arm, "seed": seed}


def run_dir(arm, seed):
    identity(arm, seed)
    return RUN_ROOT / arm / f"seed_{seed}"


def bundle_dir(arm, seed):
    identity(arm, seed)
    return BUNDLE_ROOT / arm / f"seed_{seed}"


def audit_path(seed):
    identity(ARMS[0], seed)
    return RUN_ROOT / "audits" / f"seed_{seed}/audit.json"


def validate(metadata, arm, seed):
    if (metadata["identity"] != identity(arm, seed) or metadata["runtime_versions"] != RUNTIME
            or metadata["norm_definition"] != NORM_DEFINITION or metadata["budget"] != budget(arm)
            or metadata["actor_update_count"] != actor_updates(arm)):
        raise ValueError("JAX reward-transfer identity/runtime/budget differs")
    config = metadata["config"]
    if config["seed"] != seed or any(config[key] != value for key, value in settings(arm).items()):
        raise ValueError("JAX reward-transfer controls differ")
