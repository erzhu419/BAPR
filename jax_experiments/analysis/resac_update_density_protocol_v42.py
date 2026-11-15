"""Frozen 2x2 update-density/norm intervention in the unchanged actuator world."""
from jax_experiments.analysis import resac_torch_alignment_protocol_v41 as old

ROOT = old.ROOT
PROTOCOL = "v42-resac-update-density"
SEEDS = (112003, 112023, 112043)
ARMS = ("low_full", "low_no_shift", "high_full", "high_no_shift")
RUN_ROOT = ROOT / "jax_experiments/results_resac_update_density_v42"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_resac_update_density_v42"
PREREG = ROOT / "reports/resac_update_density_v42_preregistration_2026-10-03.md"
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
        raise ValueError(f"unknown update-density arm {arm}")
    return {**old.settings("aligned_full"), "max_iters": 250,
            "updates_per_iter": 4000 if arm.startswith("high_") else 250,
            "weight_reg": 0.0 if arm.endswith("no_shift") else 0.01,
            "initial_random_steps": 10000, "start_train_steps": 10000,
            "env_name": ENV, "brax_backend": "spring", "env_type": "stochastic_mode",
            "stochastic_mode_family": "actuator_polarity",
            "stochastic_mode_dwell_steps": DWELL_STEPS,
            "stochastic_mode_dwell_distribution": "fixed",
            "max_episode_steps": MAX_EPISODE_STEPS, "task_num": 4, "test_task_num": 4}


def budget(arm):
    config = settings(arm)
    return {"next_iteration": config["max_iters"],
            "total_steps": config["initial_random_steps"] + config["max_iters"] * config["samples_per_iter"],
            "update_count": config["max_iters"] * config["updates_per_iter"]}


def actor_updates(arm):
    return budget(arm)["update_count"] // settings(arm)["resac_critic_actor_ratio"]


def identity(arm, seed):
    settings(arm)
    if seed not in SEEDS:
        raise ValueError(f"unknown update-density seed {seed}")
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
        raise ValueError("update-density identity/runtime/budget differs")
    config = metadata["config"]
    if config["seed"] != seed or any(config[key] != value for key, value in settings(arm).items()):
        raise ValueError("update-density profile controls differ")
