"""Ant development protocol for correct matching-mode checkpoint selection."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_ant_full_state_headroom_v22 as parent,
)


ROOT = parent.ROOT
PROTOCOL_VERSION = "v23-ant-matching-checkpoint-development"
ENV = parent.ENV
FAMILY = parent.FAMILY
MODES = parent.MODES
TRAINED_MODES = (0, 1)
FROZEN_MODES = (2, 3)
VARIANTS = ("matching_best", "period2_matching_best")
TRAINING_SEEDS = parent.TRAINING_SEEDS

CALIBRATION_EVENT_SEEDS = (188_001, 188_017, 188_033)
STATIONARY_HOLDOUT_EVENT_SEEDS = (188_101, 188_117, 188_133)
SWITCHING_EVENT_SEEDS = (188_201, 188_217, 188_233)
SWITCHING_SCHEDULES = {
    188_201: (3, 0, 2, 1),
    188_217: (1, 2, 0, 3),
    188_233: (2, 1, 3, 0),
}

SOURCE_NEXT_ITERATION = parent.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = parent.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = parent.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = parent.SOURCE_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = parent.SOURCE_START_TRAIN_STEPS
FINETUNE_ITERS = parent.FINETUNE_ITERS
FINAL_NEXT_ITERATION = parent.FINAL_NEXT_ITERATION
FINAL_ITERATION = parent.FINAL_ITERATION
SAMPLES_PER_ITER = parent.SAMPLES_PER_ITER
UPDATES_PER_ITER = parent.UPDATES_PER_ITER
FINAL_TOTAL_STEPS = parent.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = parent.FINAL_UPDATE_COUNT

DWELL_STEPS = parent.DWELL_STEPS
MAX_EPISODE_STEPS = parent.MAX_EPISODE_STEPS
EPISODES_PER_TASK = parent.EPISODES_PER_TASK
SWITCHING_EPISODES = parent.SWITCHING_EPISODES

MIN_CALIBRATION_GAIN = parent.MIN_CALIBRATION_GAIN
MIN_HOLDOUT_MODE_GAIN = parent.MIN_HOLDOUT_MODE_GAIN
MIN_HOLDOUT_MODE_WINS = parent.MIN_HOLDOUT_MODE_WINS
MIN_SWITCHING_GAIN = parent.MIN_SWITCHING_GAIN
REQUIRED_CELL_PASSES = len(TRAINING_SEEDS)

ACTOR_UPDATE_PERIOD = {
    "matching_best": 1,
    "period2_matching_best": 2,
}

RUN_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_matching_checkpoint_v23"
)
SPECIALIST_RUN_ROOT = RUN_ROOT / "specialists"
SPECIALIST_BUNDLE_ROOT = (
    ROOT / "jax_experiments"
    / "eval_bundles_regime_polarity_ant_matching_checkpoint_v23"
)
AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_matching_checkpoint_audit_v23"
)
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_ant_matching_checkpoint_analysis_v23"
)
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_ant_matching_checkpoint_v23"
)
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
REPORT = (
    ROOT / "reports"
    / "regime_polarity_ant_matching_checkpoint_v23_2026-09-11.md"
)

POLICY_NAME = parent.POLICY_NAME
CONTROLLER_STATE_NAME = parent.CONTROLLER_STATE_NAME
BOOTSTRAP_NAME = parent.BOOTSTRAP_NAME
SELECTION_NAME = parent.SELECTION_NAME
SOURCE_BUNDLE_SCHEMA = parent.SOURCE_BUNDLE_SCHEMA
BUNDLE_SCHEMA = "bapr.ant-matching-checkpoint-bundle.v23"
BOOTSTRAP_SCHEMA = "bapr.ant-matching-checkpoint-bootstrap.v23"
AUDIT_SCHEMA = "bapr.ant-matching-checkpoint-audit.v23"
ANALYSIS_SCHEMA = "bapr.ant-matching-checkpoint-analysis.v23"
REGISTRATION_SCHEMA = "bapr.ant-matching-checkpoint-registration.v23"

file_record = parent.file_record
read_json = parent.read_json
write_json_atomic = parent.write_json_atomic
write_text_atomic = parent.write_text_atomic
checkpoint_record = parent.checkpoint_record


def require_variant(variant: str) -> str:
    value = str(variant)
    if value not in VARIANTS:
        raise ValueError(f"unknown Ant v23 variant {value!r}")
    return value


def require_training_seed(seed: int) -> int:
    return parent.require_training_seed(seed)


require_seed = require_training_seed


def require_mode(mode: int) -> int:
    return parent.require_mode(mode)


def require_trained_mode(mode: int) -> int:
    value = require_mode(mode)
    if value not in TRAINED_MODES:
        raise ValueError(f"Ant v23 trains only modes {TRAINED_MODES}, got {value}")
    return value


def require_switching_event_seed(seed: int) -> int:
    value = int(seed)
    if value not in SWITCHING_EVENT_SEEDS:
        raise ValueError(f"unknown Ant v23 switching event seed {value}")
    return value


def switching_sequence(event_seed: int, episode: int) -> tuple[int, ...]:
    base = SWITCHING_SCHEDULES[require_switching_event_seed(event_seed)]
    shift = int(episode) % len(base)
    return tuple(base[shift:] + base[:shift])


def actor_update_period(variant: str) -> int:
    return ACTOR_UPDATE_PERIOD[require_variant(variant)]


def select_best_validation(variant: str) -> bool:
    require_variant(variant)
    return True


def source_bundle(seed: int) -> Path:
    return parent.source_bundle(require_training_seed(seed))


def source_manifest(seed: int) -> Path:
    return parent.source_manifest(require_training_seed(seed))


def source_required_paths(seed: int) -> tuple[Path, ...]:
    return parent.source_required_paths(require_training_seed(seed))


def run_dir(variant: str, seed: int, mode: int) -> Path:
    return (
        SPECIALIST_RUN_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}"
        / f"mode_{require_trained_mode(mode)}"
    )


def bundle_dir(variant: str, seed: int, mode: int) -> Path:
    variant = require_variant(variant)
    seed = require_training_seed(seed)
    mode = require_mode(mode)
    if mode in FROZEN_MODES:
        return parent.bundle_dir(parent.CONTROL_VARIANT, seed, mode)
    return SPECIALIST_BUNDLE_ROOT / variant / f"seed_{seed}" / f"mode_{mode}"


def bundle_manifest(variant: str, seed: int, mode: int) -> Path:
    return bundle_dir(variant, seed, mode) / "bundle_manifest.json"


def bundle_required_paths(
    variant: str, seed: int, mode: int,
) -> tuple[Path, ...]:
    mode = require_mode(mode)
    if mode in FROZEN_MODES:
        return parent.bundle_required_paths(
            parent.CONTROL_VARIANT, require_training_seed(seed), mode)
    directory = bundle_dir(variant, seed, mode)
    return (
        directory / "bundle_manifest.json",
        directory / "policy" / POLICY_NAME,
        directory / "logs" / "protocol_signature.json",
        directory / "provenance" / BOOTSTRAP_NAME,
        directory / "provenance" / SELECTION_NAME,
    )


def audit_dir(variant: str, seed: int) -> Path:
    return (
        AUDIT_ROOT / require_variant(variant)
        / f"seed_{require_training_seed(seed)}"
    )


def audit_result(variant: str, seed: int) -> Path:
    return audit_dir(variant, seed) / "audit.json"


def audit_manifest(variant: str, seed: int) -> Path:
    return audit_dir(variant, seed) / "audit_manifest.json"


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def identity(variant: str, seed: int, mode: int) -> dict[str, Any]:
    variant = require_variant(variant)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "ant_matching_mode_checkpoint_specialist",
        "variant": variant,
        "env": ENV,
        "family": FAMILY,
        "training_seed": require_training_seed(seed),
        "fixed_mode": require_trained_mode(mode),
        "algo": "sac",
        "source_next_iteration": SOURCE_NEXT_ITERATION,
        "final_next_iteration": FINAL_NEXT_ITERATION,
        "final_total_steps": FINAL_TOTAL_STEPS,
        "final_update_count": FINAL_UPDATE_COUNT,
        "actor_update_period": actor_update_period(variant),
        "select_best_validation": True,
        "selection_task": require_trained_mode(mode),
    }


def expected_source_checkpoint() -> dict[str, Any]:
    return parent.expected_source_checkpoint()


def expected_checkpoint() -> dict[str, Any]:
    return parent.expected_checkpoint()


def bundle_records(variant: str, seed: int) -> dict[str, Any]:
    return {
        str(mode): file_record(bundle_manifest(variant, seed, mode))
        for mode in MODES
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "jax_experiments/analysis/train_regime_polarity_ant_matching_checkpoint_v23.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_matching_checkpoint_v23.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_ant_matching_checkpoint_audit_v23.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_ant_matching_checkpoint_v23.py",
        ROOT / "scripts/submit_regime_polarity_ant_matching_checkpoint_v23.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_policy_stability_v20.py",
        ROOT / "jax_experiments/analysis/run_regime_polarity_specialist_stability_audit_v19.py",
        ROOT / "jax_experiments/analysis/analyze_regime_polarity_specialist_stability_v19.py",
        ROOT / "jax_experiments/algos/sac_policy_stability.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/train.py",
        parent.REGISTRATION_PATH,
        parent.analysis_json(),
    ]
    for seed in TRAINING_SEEDS:
        paths.append(source_manifest(seed))
        for mode in FROZEN_MODES:
            paths.append(parent.bundle_manifest(
                parent.CONTROL_VARIANT, seed, mode))
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    parent.validate_registration()
    prior = read_json(parent.analysis_json())
    if (
        prior.get("gate_pass") is not False
        or prior.get("mean_safe_oracle_switching", 0.0)
        <= 1.2 * prior.get("mean_robust_switching", float("inf"))
        or all(
            cell.get("stationary", {}).get("mode_wins", 0) >= 3
            for cell in prior.get("cells", {}).values()
        )
    ):
        raise ValueError("Ant v22 does not authorize the v23 diagnostic")
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Ant v23 registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "registered",
        "created_before_training": True,
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "variants": list(VARIANTS),
            "training_seeds": list(TRAINING_SEEDS),
            "trained_modes": list(TRAINED_MODES),
            "frozen_modes": list(FROZEN_MODES),
            "calibration_event_seeds": list(CALIBRATION_EVENT_SEEDS),
            "stationary_holdout_event_seeds": list(
                STATIONARY_HOLDOUT_EVENT_SEEDS),
            "switching_event_seeds": list(SWITCHING_EVENT_SEEDS),
            "switching_schedules": {
                str(seed): list(sequence)
                for seed, sequence in SWITCHING_SCHEDULES.items()
            },
        },
        "question": (
            "whether correct matching-mode checkpoint selection, optionally "
            "with period-2 actor updates, stabilizes Ant modes 0 and 1"
        ),
        "frozen_boundary": {
            "v22_training_and_audits_unchanged": True,
            "robust_sources_reused": True,
            "mode_2_and_3_policies_reused": True,
            "environment_and_training_budget_unchanged": True,
            "estimator_or_router_training": False,
        },
        "training": {
            variant: {
                "actor_update_period": actor_update_period(variant),
                "selection": "best matching-fixed-mode trainer evaluation",
                "trained_modes": list(TRAINED_MODES),
            }
            for variant in VARIANTS
        },
        "gate": {
            "all_three_policy_seeds_pass": True,
            "minimum_stationary_mode_wins_per_seed": MIN_HOLDOUT_MODE_WINS,
            "minimum_per_mode_gain": MIN_HOLDOUT_MODE_GAIN,
            "minimum_switching_gain": MIN_SWITCHING_GAIN,
            "all_three_switching_events_win": True,
            "zero_termination": True,
            "selection": (
                "highest mean safe switching return among passing variants; "
                "registered variant order breaks exact ties"
            ),
        },
        "sync_policy": (
            "compact policy/provenance bundles and JSON only; never sync "
            "replay or full checkpoints"
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
            raise ValueError("existing Ant v23 registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    return validate_registration()


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise FileNotFoundError(f"missing Ant v23 registration: {REGISTRATION_PATH}")
    payload = read_json(REGISTRATION_PATH)
    if payload != registration_payload():
        raise ValueError("Ant v23 registration or source closure changed")
    return payload


def assert_protocol_integrity() -> None:
    if set(TRAINED_MODES) & set(FROZEN_MODES):
        raise ValueError("Ant v23 trained and frozen modes overlap")
    if set(TRAINED_MODES) | set(FROZEN_MODES) != set(MODES):
        raise ValueError("Ant v23 mode partition is incomplete")
    splits = (
        set(CALIBRATION_EVENT_SEEDS),
        set(STATIONARY_HOLDOUT_EVENT_SEEDS),
        set(SWITCHING_EVENT_SEEDS),
    )
    if any(
        left & right
        for index, left in enumerate(splits)
        for right in splits[index + 1:]
    ):
        raise ValueError("Ant v23 evaluation splits overlap")
    if set(SWITCHING_SCHEDULES) != set(SWITCHING_EVENT_SEEDS):
        raise ValueError("Ant v23 switching schedules are incomplete")
    if len(set(SWITCHING_SCHEDULES.values())) != len(SWITCHING_SCHEDULES):
        raise ValueError("Ant v23 switching schedules are not distinct")
    if any(
        len(sequence) != len(MODES) or set(sequence) != set(MODES)
        for sequence in SWITCHING_SCHEDULES.values()
    ):
        raise ValueError("each Ant v23 schedule must use every mode")


assert_protocol_integrity()
