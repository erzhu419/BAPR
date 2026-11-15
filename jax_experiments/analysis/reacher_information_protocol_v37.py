"""Fresh development comparison: compensation versus belief-conditioned SAC."""
from jax_experiments.analysis import reacher_confirmation_protocol_v36 as confirmation

source = confirmation.source
ROOT = source.ROOT
PROTOCOL = "reacher-information-matched-v37"
RUN_ROOT = ROOT / "jax_experiments/results_reacher_information_v37"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_reacher_information_v37"
PREREG = ROOT / "reports/reacher_information_v37_preregistration_2026-10-01.md"
SMOKE = BUNDLE_ROOT / "smoke/ready.json"
FROZEN_METADATA = confirmation.FROZEN_METADATA
POLICY_SEEDS = (97007, 97027, 97047)
ARMS = ("reference", "robust", "conditioned", "conditioned_no_filter")
CONDITIONED_ARMS = ARMS[2:]
PRIMARY_MODEL = confirmation.PRIMARY_MODEL
MODEL_SEEDS = (PRIMARY_MODEL,)
EVENT_SEEDS = (208601, 208617, 208633)
STATIONARY_EVENT_SEED = 208701
EPISODES = source.EPISODES
CONTROLS = source.CONTROLS
RUNTIME = source.RUNTIME
ANALYSIS = RUN_ROOT / "analysis/analysis.json"


def screen_bundle(arm, seed):
    return BUNDLE_ROOT / "reacher" / arm / f"seed_{seed}"


def checkpoint_dir(arm, seed):
    return RUN_ROOT / "reacher" / arm / f"seed_{seed}" / "checkpoints"


def audit_path(seed):
    return RUN_ROOT / "audits" / f"seed_{seed}" / "audit.json"
