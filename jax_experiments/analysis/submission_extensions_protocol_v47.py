"""Frozen estimator robustness, actuator permutation and source-trainer studies."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "submission-extensions-v47"
RUN_ROOT = ROOT / "jax_experiments/results_submission_extensions_v47"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_submission_extensions_v47"
PREREG = ROOT / "reports/submission_extensions_v47_preregistration_2026-10-05.md"
RUNTIME = {"jax": "0.4.38", "brax": "0.10.5", "flax": "0.10.4", "mujoco": "3.1.6"}
REACHER_SEEDS = (99007, 99027, 99047, 99067, 99087)
REACHER_MODELS = (95103, 95121, 95139)
HC_SEEDS = (88021, 88039, 88057, 88079, 88103, 88121, 88139, 88157, 88179)
HC_MODELS = (94103, 94121, 94139)
PERMUTATION_SEEDS = (117003, 117023, 117043, 117063, 117083)
PERMUTATION_MODELS = (118003, 118023, 118043)
PROVIDER_SEED = 116003
EVENT_SEEDS = (211601, 211617, 211633)
STATIONARY_EVENT_SEED = 211701
TRAIN_EVENT_SEED = 210001
VALIDATION_EVENT_SEED = 210101
CONTROLS = (0.002, 1.0, 0.98)
TRANSFORMS = (
    ((1, 0), (0, 1)), ((0, 1), (-1, 0)),
    ((-1, 0), (0, -1)), ((0, -1), (1, 0)),
)
GPU_NODES = ("jtl110gpu", "jtl110gpu2", "jtl311linux", "node007")
CPU_NODES = tuple(f"node00{i}" for i in range(1, 8))
GPU_PYTHON = "/home/erzhu419/.venvs/resac-jax-gpu1-0438/bin/python"
CPU_PYTHON = "/home/zhengliang01/scheduleurm_work/conda_envs/bapr-v34-eval-0438/bin/python"
SOURCE_PYTHON = "/home/erzhu419/.venvs/bapr-source-brax-v47/bin/python"
SOURCE_GPU_NODES = ("jtl110gpu",)
SOURCE_SEEDS = (109003, 109023, 109043, 109063, 109083)
SOURCE = ROOT / "reproducibility/resac_torch_native_v43/source"
SOURCE_ARMS = ("full", "no_shift")


def smoke(suite):
    return BUNDLE_ROOT / suite / "smoke/ready.json"


def actor(suite, arm, seed, model=None):
    root = BUNDLE_ROOT / suite
    if model is not None:
        root = root / f"model_{model}"
    return root / "reacher" / arm / f"seed_{seed}"


def model_dir(seed):
    return BUNDLE_ROOT / "permutation/estimators" / f"seed_{seed}"


def hc_model_dir(seed):
    return BUNDLE_ROOT / "hc-estimator/estimators" / f"seed_{seed}"


def dataset_dir(seed):
    return RUN_ROOT / "permutation/estimators" / f"seed_{seed}/dataset"


def audit_path(suite, seed):
    return RUN_ROOT / "audits" / suite / f"seed_{seed}/audit.json"


def analysis_path(suite):
    return RUN_ROOT / "analysis" / suite / "analysis.json"


def source_budget():
    return {"next_iteration": 2100, "total_steps": 8400000,
            "critic_calls": 525000, "actor_opportunities": 262500}


def source_samples(iteration):
    # Ten thousand warmup transitions count inside the 8.4M interaction budget.
    return 3995 + int(iteration < 500)
