"""Frozen checkpoint-only mechanism audit for the final BAPR stack."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from jax_experiments.analysis import (
    regime_polarity_fallback_final_comparison_v1 as frozen,
)


ROOT = frozen.ROOT
PROTOCOL_VERSION = "v2-final-stack-mechanism-audit"
ENV = frozen.ENV
FAMILY = frozen.FAMILY
MODES = frozen.MODES
STUDENT_SEEDS = frozen.STUDENT_SEEDS

# These event streams were not used by estimator training, DAgger, model
# selection, the final comparison, or earlier mechanism diagnostics.
EVENT_SEEDS = (107_503, 107_537, 107_579, 107_621, 107_659)
MAX_EPISODE_STEPS = frozen.MAX_EPISODE_STEPS
DWELL_STEPS = frozen.DWELL_STEPS
STATIONARY_EPISODES = frozen.AUDIT_EPISODES_PER_TASK
SWITCHING_EPISODES = frozen.AUDIT_SWITCHING_EPISODES

ROBUST_ARM = "robust_719"
LEARNED_ARM = "student_learned_no_fallback"
FALLBACK_ARM = "student_learned_with_fallback"
TRUE_ARM = "student_true_context"
ZERO_ARM = "student_zero_context"
UNIFORM_ARM = "student_uniform_context"
FIXED_ARMS = tuple(f"student_fixed_{mode}" for mode in MODES)
CYCLIC_ARM = "student_cyclic_wrong"
SHUFFLED_ARM = "student_shuffled_wrong"
DELAY_STEPS = (1, 5, 10, 25, 50)
DELAY_ARMS = tuple(f"student_true_delay_{delay}" for delay in DELAY_STEPS)
SWITCHING_ARMS = (
    ROBUST_ARM,
    LEARNED_ARM,
    FALLBACK_ARM,
    TRUE_ARM,
    ZERO_ARM,
    UNIFORM_ARM,
    *FIXED_ARMS,
    CYCLIC_ARM,
    SHUFFLED_ARM,
    *DELAY_ARMS,
)
STATIONARY_ARMS = (
    ROBUST_ARM,
    LEARNED_ARM,
    FALLBACK_ARM,
    TRUE_ARM,
    ZERO_ARM,
    UNIFORM_ARM,
    *FIXED_ARMS,
)
TRANSIENT_BINS = ((0, 1), (1, 5), (5, 10), (10, 25),
                  (25, 50), (50, 100), (100, 250))

AUDIT_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_final_mechanism_audit_v2")
ANALYSIS_ROOT = (
    ROOT / "jax_experiments"
    / "results_regime_polarity_final_mechanism_analysis_v2")
REGISTRATION_ROOT = (
    ROOT / "jax_experiments" / "deployments"
    / "regime_polarity_final_mechanism_audit_v2")
REGISTRATION_PATH = REGISTRATION_ROOT / "registration.json"
REPORT = (
    ROOT / "reports"
    / "regime_polarity_final_mechanism_audit_v2_2026-08-09.md")

AUDIT_SCHEMA = "bapr.regime-polarity-final-mechanism-audit.v2"
EVENT_SCHEMA = "bapr.regime-polarity-final-mechanism-event.v2"
ANALYSIS_SCHEMA = "bapr.regime-polarity-final-mechanism-analysis.v2"
REGISTRATION_SCHEMA = "bapr.regime-polarity-final-mechanism-registration.v2"

file_record = frozen.file_record
read_json = frozen.read_json
write_json_atomic = frozen.write_json_atomic
write_text_atomic = frozen.write_text_atomic


def require_student_seed(seed: int) -> int:
    return frozen.require_student_seed(seed)


def require_event_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in EVENT_SEEDS:
        raise ValueError(f"unknown final mechanism event seed {seed}")
    return seed


def require_arm(arm: str) -> str:
    arm = str(arm)
    if arm not in SWITCHING_ARMS:
        raise ValueError(f"unknown final mechanism arm {arm!r}")
    return arm


def delay_for_arm(arm: str) -> int | None:
    arm = require_arm(arm)
    if arm not in DELAY_ARMS:
        return None
    return int(arm.rsplit("_", 1)[1])


def shuffled_mode_map(event_seed: int) -> tuple[int, ...]:
    event_seed = require_event_seed(event_seed)
    modes = list(MODES)
    rng = random.Random(event_seed + 8_209_731)
    while True:
        candidate = modes.copy()
        rng.shuffle(candidate)
        if all(left != right for left, right in zip(modes, candidate)):
            return tuple(candidate)


def audit_dir(student_seed: int, event_seed: int) -> Path:
    return (
        AUDIT_ROOT / f"student_seed_{require_student_seed(student_seed)}"
        / f"event_seed_{require_event_seed(event_seed)}")


def audit_manifest(student_seed: int, event_seed: int) -> Path:
    return audit_dir(student_seed, event_seed) / "audit_manifest.json"


def all_audit_manifests() -> tuple[Path, ...]:
    return tuple(
        audit_manifest(student_seed, event_seed)
        for student_seed in STUDENT_SEEDS
        for event_seed in EVENT_SEEDS
    )


def analysis_json() -> Path:
    return ANALYSIS_ROOT / "analysis.json"


def analysis_markdown() -> Path:
    return ANALYSIS_ROOT / "analysis.md"


def pass_marker() -> Path:
    return ANALYSIS_ROOT / "MECHANISM_PASS"


def audit_identity(student_seed: int, event_seed: int) -> dict[str, Any]:
    event_seed = require_event_seed(event_seed)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_role": "post_confirmation_checkpoint_only_mechanism_audit",
        "confirmatory": False,
        "selection_forbidden": True,
        "env": ENV,
        "family": FAMILY,
        "student_seed": require_student_seed(student_seed),
        "event_seed": event_seed,
        "switching_arms": list(SWITCHING_ARMS),
        "stationary_arms": list(STATIONARY_ARMS),
        "delay_steps": list(DELAY_STEPS),
        "transient_bins": [list(value) for value in TRANSIENT_BINS],
        "shuffled_mode_map": list(shuffled_mode_map(event_seed)),
        "max_episode_steps": MAX_EPISODE_STEPS,
        "dwell_steps": DWELL_STEPS,
        "online_inputs_for_deployable_arms": [
            "observation", "commanded_action", "reward",
            "next_observation", "causal_mode_posterior",
        ],
        "forbidden_online_inputs_for_deployable_arms": [
            "mode_id", "action_gain", "executed_action", "switch_clock",
        ],
        "diagnostic_only_inputs": [
            "true_mode_context", "delayed_true_mode_context",
        ],
    }


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def registration_source_paths() -> tuple[Path, ...]:
    robust_bundle = frozen.development.ensemble.final.bundle_dir(
        ENV, "robust", frozen.ROBUST_SEED)
    paths = [
        ROOT / "jax_experiments/analysis/"
        "regime_polarity_final_mechanism_audit_v1.py",
        ROOT / "jax_experiments/analysis/"
        "run_regime_polarity_final_mechanism_audit_v1.py",
        ROOT / "jax_experiments/analysis/"
        "analyze_regime_polarity_final_mechanism_audit_v1.py",
        ROOT / "scripts/submit_regime_polarity_final_mechanism_audit_v1.py",
        REPORT,
        ROOT / "jax_experiments/analysis/"
        "regime_polarity_fallback_final_comparison_v1.py",
        ROOT / "jax_experiments/analysis/"
        "run_regime_polarity_fallback_final_audit_v1.py",
        ROOT / "jax_experiments/analysis/regime_polarity_final_confirmation.py",
        ROOT / "jax_experiments/analysis/"
        "regime_polarity_policy_distillation_control_model.py",
        ROOT / "jax_experiments/analysis/"
        "regime_polarity_policy_distillation_control_v2.py",
        ROOT / "jax_experiments/analysis/"
        "regime_polarity_policy_distillation_model.py",
        ROOT / "jax_experiments/analysis/train_regime_polarity_posterior.py",
        ROOT / "jax_experiments/analysis/regime_polarity_posterior.py",
        ROOT / "jax_experiments/analysis/regime_polarity_posterior_model.py",
        ROOT / "jax_experiments/analysis/regime_polarity_headroom.py",
        ROOT / "jax_experiments/analysis/final_task_sweep.py",
        ROOT / "jax_experiments/train.py",
        ROOT / "jax_experiments/configs/default.py",
        ROOT / "jax_experiments/common/causal_fallback.py",
        ROOT / "jax_experiments/common/checkpoint.py",
        ROOT / "jax_experiments/common/logging.py",
        ROOT / "jax_experiments/common/replay_buffer.py",
        ROOT / "jax_experiments/envs/brax_env.py",
        ROOT / "jax_experiments/envs/stochastic_mode_env.py",
        ROOT / "jax_experiments/algos/regime_sac.py",
        ROOT / "jax_experiments/algos/bapr_v2.py",
        ROOT / "jax_experiments/networks/policy.py",
        ROOT / "jax_experiments/networks/ensemble_critic.py",
        ROOT / "jax_experiments/networks/probabilistic_regime_context.py",
        frozen.REGISTRATION_PATH,
        frozen.analysis_json(),
        ROOT / "jax_experiments/results_regime_polarity_"
        "corrected_baseline_analysis_v2/analysis.json",
        frozen.ensemble.final.MODEL_MANIFEST,
        frozen.ensemble.final.MODEL_PATH,
        *frozen.development.ensemble.final.bundle_required_paths(
            ENV, "robust", frozen.ROBUST_SEED),
    ]
    for seed in STUDENT_SEEDS:
        paths.extend([
            frozen.model_manifest("mode_heads", seed),
            frozen.model_path("mode_heads", seed),
        ])
    # Retain the directory marker in case the bundle API grows another
    # required file; current records remain file-level and immutable.
    if not robust_bundle.is_dir():
        raise ValueError(f"missing frozen robust bundle: {robust_bundle}")
    return tuple(dict.fromkeys(path.resolve() for path in paths))


def registration_payload() -> dict[str, Any]:
    paths = registration_source_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"mechanism registration sources missing: {missing}")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "frozen",
        "identity": {
            "protocol_version": PROTOCOL_VERSION,
            "env": ENV,
            "family": FAMILY,
            "student_seeds": list(STUDENT_SEEDS),
            "event_seeds": list(EVENT_SEEDS),
            "selection_forbidden": True,
        },
        "source_records": {
            _relative(path): file_record(path) for path in paths
        },
    }


def create_registration() -> dict[str, Any]:
    payload = registration_payload()
    REGISTRATION_ROOT.mkdir(parents=True, exist_ok=True)
    if REGISTRATION_PATH.exists():
        current = read_json(REGISTRATION_PATH)
        if current != payload:
            raise ValueError("existing mechanism registration changed")
    else:
        write_json_atomic(REGISTRATION_PATH, payload)
    validate_registration()
    return payload


def validate_registration() -> dict[str, Any]:
    if not REGISTRATION_PATH.is_file():
        raise ValueError("mechanism registration has not been created")
    payload = read_json(REGISTRATION_PATH)
    expected = registration_payload()
    if payload != expected:
        raise ValueError("mechanism registration or frozen sources changed")
    return payload


def registration_record() -> dict[str, Any]:
    validate_registration()
    return file_record(REGISTRATION_PATH)


def assert_split_integrity() -> None:
    if len(EVENT_SEEDS) != len(set(EVENT_SEEDS)):
        raise ValueError("mechanism event seeds are duplicated")
    prior = {
        *frozen.TRAIN_EVENT_SEEDS,
        *frozen.SUPERVISED_VALIDATION_EVENT_SEEDS,
        *frozen.DAGGER_EVENT_SEEDS,
        *frozen.CONTROL_VALIDATION_EVENT_SEEDS,
        *frozen.FINAL_EVENT_SEEDS,
        102_301, 102_331, 102_367, 102_397, 102_451,
        105_301, 105_331, 105_367, 105_399, 105_451,
    }
    overlap = set(EVENT_SEEDS) & prior
    if overlap:
        raise ValueError(f"mechanism event seeds were reused: {overlap}")


assert_split_integrity()


if __name__ == "__main__":
    print(json.dumps(create_registration(), indent=2, sort_keys=True))
