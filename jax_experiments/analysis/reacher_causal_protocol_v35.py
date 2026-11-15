"""Frozen causal-estimator follow-up to the admitted Reacher oracle screen."""
from jax_experiments.analysis import mechanism_gap_protocol_v34 as screen

ROOT = screen.ROOT
RUN_ROOT = ROOT / "jax_experiments/results_reacher_causal_v35"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_reacher_causal_v35"
PREREG = ROOT / "reports/reacher_causal_v35_preregistration_2026-09-30.md"
HEADROOM = screen.CPU_RUN_ROOT / "analysis/reacher/analysis.json"
SMOKE = BUNDLE_ROOT / "smoke/ready.json"
PROVIDER_SEED = 94003
PROVIDER_RUN_ROOT = RUN_ROOT / "provider"
PROVIDER_BUNDLE_ROOT = BUNDLE_ROOT / "provider"
PROVIDER_BUNDLE = PROVIDER_BUNDLE_ROOT / "reacher/reference" / f"seed_{PROVIDER_SEED}"
DATA_DIR = RUN_ROOT / "dataset"
DATA_MANIFEST = BUNDLE_ROOT / "dataset/complete.json"
DATA_NODE = "jtl311linux"
MODEL_SEEDS = (95103, 95121, 95139)
PRIMARY_MODEL = MODEL_SEEDS[0]
POLICY_SEEDS = screen.REACHER_SEEDS
TRAIN_EVENT_SEED = 205001
VALIDATION_EVENT_SEED = 205101
EVENT_SEEDS = (205601, 205617, 205633)
STATIONARY_EVENT_SEED = 205701
TRAIN_ROWS = screen.TRAIN_ROWS
VALIDATION_ROWS = screen.VALIDATION_ROWS
MODEL_UPDATES = screen.MODEL_UPDATES
BATCH_SIZE = screen.BATCH_SIZE
EPISODES = screen.EPISODES
CONTROLS = (0.002, 1.0, 0.98)
RUNTIME = {"jax": "0.4.38", "brax": "0.10.5", "flax": "0.10.4", "mujoco": "3.1.6"}
ANALYSIS = RUN_ROOT / "analysis/analysis.json"


def model_dir(seed):
    return BUNDLE_ROOT / "estimators" / f"seed_{seed}"


def checkpoint_dir(seed):
    return RUN_ROOT / "estimators" / f"seed_{seed}" / "checkpoints"


def audit_path(seed):
    return RUN_ROOT / "audits" / f"seed_{seed}" / "audit.json"


def screen_bundle(arm, seed):
    return screen.CPU_BUNDLE_ROOT / "reacher" / arm / f"seed_{seed}"
