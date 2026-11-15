"""Registered survival-valid headroom screen for Hopper and Walker2d."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import regime_polarity_headroom as parent
from jax_experiments.envs.stochastic_mode_env import MODE_FAMILIES


ROOT = parent.ROOT
PROTOCOL_VERSION = "v27-structured-channel-survival-headroom-development"
FAMILY = "structured_channel"
ENVS = ("Hopper-v2", "Walker2d-v2")
ROLES = ("robust", "oracle")
TRAINING_SEEDS = (86_003, 86_021, 86_039)
AUDIT_EVENT_SEEDS = (193_101, 193_117, 193_133)
MODES = (0, 1, 2, 3)

MAX_ITERS = 1_400
FINAL_ITERATION = MAX_ITERS - 1
SAMPLES_PER_ITER = 4_000
UPDATES_PER_ITER = 250
FINAL_TOTAL_STEPS = MAX_ITERS * SAMPLES_PER_ITER
FINAL_UPDATE_COUNT = MAX_ITERS * UPDATES_PER_ITER
DWELL_STEPS = 250
MAX_EPISODE_STEPS = 1_000
EPISODES_PER_TASK = 5
SWITCHING_EPISODES = 5

MIN_RELATIVE_GAIN = 0.15
MIN_MODE_WINS = 3
MIN_SEED_WINS = len(TRAINING_SEEDS)
MAX_TERMINATION_GAP = 0.05
MAX_ABSOLUTE_TERMINATION = 0.10
MIN_PASSING_ENVS = 1

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_channel_survival_headroom_v27"
)
BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_channel_survival_headroom_v27"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_channel_survival_headroom_audit_v27"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_channel_survival_headroom_analysis_v27"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_channel_survival_headroom_v27"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
PRIOR_HEADROOM_ANALYSIS = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_headroom_analysis_v1" / "analysis.json"
)
REPORT = (
    ROOT / "reports"
    / "regime_channel_survival_headroom_v27_2026-09-12.md"
)

BUNDLE_SCHEMA = "bapr.channel-survival-headroom-bundle.v27"
AUDIT_SCHEMA = "bapr.channel-survival-headroom-audit.v27"
ANALYSIS_SCHEMA = "bapr.channel-survival-headroom-analysis.v27"
REGISTRATION_SCHEMA = "bapr.channel-survival-headroom-registration.v27"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
checkpoint_record = parent.checkpoint_record


def require_env(env: str) -> str:
    value = str(env)
    if value not in ENVS:
        raise ValueError(f"unknown V27 environment {value!r}")
    return value


def env_slug(env: str) -> str:
    return require_env(env).replace("-v2", "")


def require_role(role: str) -> str:
    value = str(role)
    if value not in ROLES:
        raise ValueError(f"unknown V27 role {value!r}")
    return value


def require_training_seed(seed: int) -> int:
    value = int(seed)
    if value not in TRAINING_SEEDS:
        raise ValueError(f"unknown V27 training seed {value}")
    return value


def require_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in AUDIT_EVENT_SEEDS:
        raise ValueError(f"unknown V27 event seed {value}")
    return value


def run_dir(env: str, role: str, seed: int) -> Path:
    return (
        RUN_ROOT / env_slug(env) / require_role(role)
        / f"seed_{require_training_seed(seed)}"
    )


def bundle_dir(env: str, role: str, seed: int) -> Path:
    return (
        BUNDLE_ROOT / env_slug(env) / require_role(role)
        / f"seed_{require_training_seed(seed)}"
    )


def bundle_manifest(env: str, role: str, seed: int) -> Path:
    return bundle_dir(env, role, seed) / "bundle_manifest.json"


def bundle_required_paths(
    env: str, role: str, seed: int,
) -> tuple[Path, ...]:
    directory = bundle_dir(env, role, seed)
    return (
        directory / "bundle_manifest.json",
        directory / "checkpoints" / "params.pkl",
        directory / "checkpoints" / "train_state.pkl",
        directory / "logs" / "protocol_signature.json",
    )


def audit_dir(env: str, role: str, seed: int) -> Path:
    return (
        AUDIT_ROOT / env_slug(env) / require_role(role)
        / f"seed_{require_training_seed(seed)}"
    )


def audit_event_dir(
    env: str, role: str, seed: int, event_seed: int,
) -> Path:
    return (
        audit_dir(env, role, seed)
        / f"event_seed_{require_event_seed(event_seed)}"
    )


def audit_manifest(env: str, role: str, seed: int) -> Path:
    return audit_dir(env, role, seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(env: str, role: str, seed: int) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "env": require_env(env),
        "family": FAMILY,
        "role": require_role(role),
        "training_seed": require_training_seed(seed),
        "algo": "regime_sac",
        "benchmark_role": "survival_valid_oracle_headroom_screen",
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "jax_experiments/analysis/run_regime_channel_survival_headroom_controller_v27.py",
        ROOT / "jax_experiments/analysis/run_regime_channel_survival_headroom_audit_v27.py",
        ROOT / "jax_experiments/analysis/analyze_regime_channel_survival_headroom_v27.py",
        ROOT / "scripts/submit_regime_channel_survival_headroom_v27.py",
        ROOT / "jax_experiments/analysis/run_regime_control_headroom_controller.py",
        ROOT / "jax_experiments/analysis/run_regime_control_headroom_audit.py",
        ROOT / "jax_experiments/analysis/analyze_regime_control_headroom.py",
        ROOT / "jax_experiments/analysis/final_task_sweep.py",
        ROOT / "jax_experiments/algos/regime_sac.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/train.py",
        PRIOR_HEADROOM_ANALYSIS,
    )
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    prior = read_json(PRIOR_HEADROOM_ANALYSIS)
    for env in ENVS:
        row = prior["environments"][env]
        if (
            row.get("env_gate_pass") is not False
            or row["summaries"]["robust"]["stationary_termination_mean"]
            != 1.0
            or row["summaries"]["oracle"]["stationary_termination_mean"]
            != 1.0
        ):
            raise ValueError(
                f"prior full-polarity result does not authorize {env}")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V27 registration sources missing: {missing}")
    profiles = MODE_FAMILIES[FAMILY]
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "environments": list(ENVS),
            "family": FAMILY,
            "roles": list(ROLES),
            "training_seeds": list(TRAINING_SEEDS),
            "event_seeds": list(AUDIT_EVENT_SEEDS),
            "modes": list(MODES),
        },
        "environment": {
            "fixed_physics_within_dwell": True,
            "dwell_steps": DWELL_STEPS,
            "affected_channel_patterns": [
                profile["action_gain_pattern"] for profile in profiles
            ],
            "nominal_gain": 1.0,
            "impaired_gain": 0.45,
            "action_noise_std": 0.04,
            "all_gains_positive_and_invertible": True,
            "severity_selected_before_hopper_walker_results": True,
        },
        "training": {
            "equal_budget_roles": True,
            "iterations": MAX_ITERS,
            "environment_steps": FINAL_TOTAL_STEPS,
            "gradient_updates": FINAL_UPDATE_COUNT,
            "robust_context": "all-zero vector",
            "oracle_context": "true current one-hot mode",
        },
        "gate": {
            "minimum_switching_gain": MIN_RELATIVE_GAIN,
            "minimum_worst_mode_gain": MIN_RELATIVE_GAIN,
            "required_switching_seed_wins": MIN_SEED_WINS,
            "required_worst_mode_seed_wins": MIN_SEED_WINS,
            "minimum_stationary_mode_wins": MIN_MODE_WINS,
            "maximum_termination_gap": MAX_TERMINATION_GAP,
            "maximum_absolute_stationary_termination": (
                MAX_ABSOLUTE_TERMINATION
            ),
            "maximum_absolute_switching_termination": (
                MAX_ABSOLUTE_TERMINATION
            ),
            "decision": "each environment is judged independently",
        },
        "stopping_rule": (
            "Only a passing environment may receive the frozen V21 BAPR "
            "recipe. A failed environment is retained without tuning channel "
            "gain, noise, dwell, or the BAPR algorithm on these seeds."
        ),
        "sync_policy": (
            "minimal evaluation bundles and JSON only; never sync replay or "
            "raw run checkpoints"
        ),
        "source_records": {
            _relative(path): file_record(path) for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.is_file():
        if read_json(REGISTRATION_PATH) != payload:
            raise ValueError("existing V27 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing V27 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("V27 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    profiles = MODE_FAMILIES[FAMILY]
    if len(profiles) != len(MODES):
        raise ValueError("V27 requires four structured-channel modes")
    if any(
        profile.get("impaired_gain") != 0.45
        or profile.get("action_noise_std") != 0.04
        for profile in profiles
    ):
        raise ValueError("V27 structured-channel severity changed")
    if set(TRAINING_SEEDS) & set(AUDIT_EVENT_SEEDS):
        raise ValueError("V27 training and event seeds overlap")
    if len(set(TRAINING_SEEDS)) != 3 or len(set(AUDIT_EVENT_SEEDS)) != 3:
        raise ValueError("V27 requires three distinct seeds per split")
    if FINAL_TOTAL_STEPS != 5_600_000 or FINAL_UPDATE_COUNT != 350_000:
        raise ValueError("V27 equal training budget changed")


assert_protocol_integrity()
