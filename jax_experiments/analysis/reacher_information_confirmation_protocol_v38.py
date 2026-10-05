"""Five untouched policy seeds for the frozen information-matched comparison."""
from jax_experiments.analysis import reacher_information_protocol_v37 as development

source = development.source
ROOT = development.ROOT
PROTOCOL = "reacher-information-confirmation-v38"
RUN_ROOT = ROOT / "jax_experiments/results_reacher_information_confirmation_v38"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_reacher_information_confirmation_v38"
PREREG = ROOT / "reports/reacher_information_confirmation_v38_preregistration_2026-10-01.md"
SMOKE = BUNDLE_ROOT / "smoke/ready.json"
FROZEN_METADATA = development.FROZEN_METADATA
POLICY_SEEDS = (99007, 99027, 99047, 99067, 99087)
ARMS = development.ARMS
CONDITIONED_ARMS = development.CONDITIONED_ARMS
PRIMARY_MODEL = development.PRIMARY_MODEL
MODEL_SEEDS = development.MODEL_SEEDS
EVENT_SEEDS = (209601, 209617, 209633)
STATIONARY_EVENT_SEED = 209701
EPISODES = development.EPISODES
CONTROLS = development.CONTROLS
RUNTIME = development.RUNTIME
ANALYSIS = RUN_ROOT / "analysis/analysis.json"


def screen_bundle(arm, seed):
    return BUNDLE_ROOT / "reacher" / arm / f"seed_{seed}"


def checkpoint_dir(arm, seed):
    return RUN_ROOT / "reacher" / arm / f"seed_{seed}" / "checkpoints"


def audit_path(seed):
    return RUN_ROOT / "audits" / f"seed_{seed}" / "audit.json"
