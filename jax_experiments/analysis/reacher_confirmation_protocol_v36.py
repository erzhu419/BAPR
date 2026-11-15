"""Fresh policy-seed confirmation of the frozen Reacher causal estimator."""
from jax_experiments.analysis import reacher_causal_protocol_v35 as source

ROOT = source.ROOT
PROTOCOL = "reacher-causal-confirmation-v36"
RUN_ROOT = ROOT / "jax_experiments/results_reacher_confirmation_v36"
BUNDLE_ROOT = ROOT / "jax_experiments/eval_bundles_reacher_confirmation_v36"
PREREG = ROOT / "reports/reacher_confirmation_v36_preregistration_2026-09-30.md"
SMOKE = BUNDLE_ROOT / "smoke/ready.json"
FROZEN_METADATA = BUNDLE_ROOT / "frozen/metadata.json"
POLICY_SEEDS = (96003, 96021, 96039, 96057, 96079)
PRIMARY_MODEL = source.PRIMARY_MODEL
MODEL_SEEDS = (PRIMARY_MODEL,)
EVENT_SEEDS = (206601, 206617, 206633)
STATIONARY_EVENT_SEED = 206701
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
