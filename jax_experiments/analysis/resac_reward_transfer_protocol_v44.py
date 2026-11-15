"""A fresh-seed reward-processing by weight-shift calibration in native Gym."""
from jax_experiments.analysis import resac_torch_native_protocol_v43 as native

ROOT, SOURCE, PYTHON = native.ROOT, native.SOURCE, native.PYTHON
GPU_NODES, RUNTIME = native.GPU_NODES, native.RUNTIME
budget, write = native.budget, native.write
PROTOCOL = "v44-native-reward-transfer"
SEEDS = (114003, 114023, 114043)
ARMS = ("raw_full", "raw_no_shift", "bus_reward_full", "bus_reward_no_shift")
RUN_ROOT = ROOT / "jax_experiments/results_resac_reward_transfer_v44"
COMPACT = ROOT / "jax_experiments/eval_bundles_resac_reward_transfer_v44"
PREREG = ROOT / "reports/resac_reward_transfer_v44_preregistration_2026-10-03.md"
SMOKE = COMPACT / "smoke/ready.json"


def variant(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    config = native.variant("full" if arm.endswith("_full") else "no_shift")
    config["reward_transform"] = "bus_batch_scale10" if arm.startswith("bus_reward_") else "raw"
    return config


def directory(arm, seed):
    return RUN_ROOT / arm / f"seed_{seed}"


def result(arm, seed):
    return COMPACT / arm / f"seed_{seed}/complete.json"


def validate(row, arm, seed):
    if (row["protocol"] != PROTOCOL or row["arm"] != arm or row["seed"] != seed
            or row["runtime"] != RUNTIME or row["budget"] != budget()
            or row["variant"] != variant(arm)):
        raise ValueError("reward-transfer configuration/runtime/budget mismatch")
    evaluation = row["final_evaluation"]
    if (len(evaluation["returns"]) != 8 or evaluation["lengths"] != [1000] * 8
            or any(evaluation["terminated"])):
        raise ValueError("incomplete native reward-transfer evaluation")
