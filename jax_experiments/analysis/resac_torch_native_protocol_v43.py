"""Original-trainer stationary Gym calibration, separate from BAPR cohorts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "v43-torch-native-stationary"
SEEDS = (113003, 113023, 113043)
ARMS = ("full", "no_shift")
SOURCE = ROOT / "reproducibility/resac_torch_native_v43/source"
RUN_ROOT = ROOT / "jax_experiments/results_resac_torch_native_v43"
COMPACT = ROOT / "jax_experiments/eval_bundles_resac_torch_native_v43"
PREREG = ROOT / "reports/resac_torch_native_v43_preregistration_2026-10-03.md"
SMOKE = COMPACT / "smoke/ready.json"
PYTHON = "/home/erzhu419/.venvs/resac-torch-native-v43/bin/python"
GPU_NODES = ("jtl110gpu", "jtl110gpu2")
RUNTIME = {"torch": "2.4.1+cu121", "gym": "0.10.9", "numpy": "1.24.4",
           "mujoco_py": "2.1.2.14", "mujoco": 210}


def variant(arm):
    import yaml
    if arm not in ARMS:
        raise ValueError(arm)
    config = yaml.safe_load((SOURCE / "halfcheetah.yaml").read_text())
    config["trainer_kwargs"]["weight_reg"] = 0.01 if arm == "full" else 0.0
    return config


def directory(arm, seed):
    return RUN_ROOT / arm / f"seed_{seed}"


def result(arm, seed):
    return COMPACT / arm / f"seed_{seed}/complete.json"


def budget(epochs=1000):
    return {"next_epoch": epochs, "total_steps": 10000 + 1000 * epochs,
            "critic_calls": 1000 * epochs, "actor_opportunities": 500 * epochs}


def validate(row, arm, seed):
    if (row["protocol"] != PROTOCOL or row["arm"] != arm or row["seed"] != seed
            or row["runtime"] != RUNTIME or row["budget"] != budget()
            or row["variant"] != variant(arm)):
        raise ValueError("native Torch configuration/runtime/budget mismatch")
    if (len(row["final_evaluation"]["returns"]) != 8
            or row["final_evaluation"]["lengths"] != [1000] * 8
            or any(row["final_evaluation"]["terminated"])):
        raise ValueError("incomplete native stationary evaluation")


def write(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
