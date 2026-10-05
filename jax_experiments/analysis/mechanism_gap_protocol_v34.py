"""Frozen follow-up experiments for the action-compensation mechanism."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "jax_experiments/results_mechanism_gap_v34"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_mechanism_gap_v34"
CPU_RUN_ROOT = ROOT / "jax_experiments/results_mechanism_gap_v34_reacher_cpu"
CPU_BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_mechanism_gap_v34_reacher_cpu"
CPU_ADMISSION = ROOT / "jax_experiments/results_mechanism_gap_v34/runtime/reacher-cpu-admission/audit.json"
RELEASE = ROOT / "reproducibility/v32"
PREREG = ROOT / "reports/mechanism_gap_v34_preregistration_2026-09-30.md"
POLICY_SEEDS = (92003, 92021, 92039, 92057, 92079)
REACHER_SEEDS = (93003, 93021, 93039)
MODEL_SEEDS = (94103, 94121, 94139)
PROVIDER_SEED = 88003
ABLATION_SEEDS = (88021, 88039, 88057, 88079, 88103, 88121, 88139, 88157, 88179)
EVENT_SEEDS = (202601, 202617, 202633)
TRAIN_EVENT_SEED = 203001
VALIDATION_EVENT_SEED = 203101
EPISODES = 5
SAMPLES = 4000
UPDATES = 250
HC_ITERS = 2100
FORK_ITER = 1400
REACHER_ITERS = 400
REACHER_HORIZON = 200
REACHER_DWELL = 50
TRAIN_ROWS = 178000
VALIDATION_ROWS = 82000
MODEL_UPDATES = 4500
BATCH_SIZE = 256
FROZEN_ROUTES = ("frozen", "no_filter", "single_head", "oracle", "uncompensated")


def run_dir(kind, arm, seed):
    return RUN_ROOT / kind / arm / f"seed_{seed}"


def bundle_dir(kind, arm, seed):
    return BUNDLE_ROOT / kind / arm / f"seed_{seed}"


def audit_path(kind, seed):
    return RUN_ROOT / "audits" / kind / f"seed_{seed}" / "audit.json"


def model_dir(seed):
    return BUNDLE_ROOT / "estimators" / f"seed_{seed}"


def dataset_dir(seed):
    return RUN_ROOT / "identification_data" / f"seed_{seed}"


def analysis_path(kind):
    return RUN_ROOT / "analysis" / kind / "analysis.json"
