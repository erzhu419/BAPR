"""V20-runner compatibility view for the V31 canonical reference."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_action_compensation_confirmation_v31 as final,
)


ROOT = final.ROOT
PROTOCOL_VERSION = final.PROTOCOL_VERSION
ENV = final.ENV
FAMILY = final.FAMILY
MODES = final.MODES
VARIANTS = final.VARIANTS
CONTROL_VARIANT = final.CONTROL_VARIANT
SPECIALIST_VARIANT = final.SPECIALIST_VARIANT
TRAINING_SEEDS = final.TRAINING_SEEDS

CALIBRATION_EVENT_SEEDS = ()
STATIONARY_HOLDOUT_EVENT_SEEDS = final.STATIONARY_HOLDOUT_EVENT_SEEDS
SWITCHING_EVENT_SEEDS = final.SWITCHING_EVENT_SEEDS
SWITCHING_SCHEDULES = final.SWITCHING_SCHEDULES

SOURCE_NEXT_ITERATION = final.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = final.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = final.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = final.SOURCE_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = final.SOURCE_START_TRAIN_STEPS
FINETUNE_ITERS = final.FINETUNE_ITERS
FINAL_NEXT_ITERATION = final.FINAL_NEXT_ITERATION
FINAL_ITERATION = final.FINAL_ITERATION
FINAL_TOTAL_STEPS = final.FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = final.FINAL_UPDATE_COUNT

SAMPLES_PER_ITER = final.SAMPLES_PER_ITER
UPDATES_PER_ITER = final.UPDATES_PER_ITER
DWELL_STEPS = final.DWELL_STEPS
MAX_EPISODE_STEPS = final.MAX_EPISODE_STEPS
EPISODES_PER_TASK = final.EPISODES_PER_TASK
SWITCHING_EPISODES = final.SWITCHING_EPISODES
MIN_CALIBRATION_GAIN = final.MIN_CALIBRATION_GAIN

SOURCE_RUN_ROOT = final.SOURCE_RUN_ROOT
SOURCE_BUNDLE_ROOT = final.SOURCE_BUNDLE_ROOT
SPECIALIST_RUN_ROOT = final.REFERENCE_RUN_ROOT
SPECIALIST_BUNDLE_ROOT = final.REFERENCE_BUNDLE_ROOT
REGISTRATION_ROOT = final.REGISTRATION_ROOT
REGISTRATION_PATH = final.REGISTRATION_PATH

POLICY_NAME = final.POLICY_NAME
CONTROLLER_STATE_NAME = final.CONTROLLER_STATE_NAME
BOOTSTRAP_NAME = final.BOOTSTRAP_NAME
SELECTION_NAME = final.SELECTION_NAME
SOURCE_BUNDLE_SCHEMA = final.SOURCE_BUNDLE_SCHEMA
BUNDLE_SCHEMA = final.REFERENCE_BUNDLE_SCHEMA
BOOTSTRAP_SCHEMA = final.REFERENCE_BOOTSTRAP_SCHEMA

file_record = final.file_record
read_json = final.read_json
write_json_atomic = final.write_json_atomic
write_text_atomic = final.write_text_atomic
checkpoint_record = final.checkpoint_record
require_training_seed = final.require_training_seed
require_seed = final.require_seed
require_mode = final.require_mode
require_switching_event_seed = final.require_switching_event_seed
switching_sequence = final.switching_sequence
source_run_dir = final.source_run_dir
source_bundle = final.source_bundle
source_manifest = final.source_manifest
source_required_paths = final.source_required_paths
source_identity = final.source_identity
expected_source_checkpoint = final.expected_source_checkpoint
validate_registration = final.validate_registration


def require_variant(variant: str) -> str:
    return final.require_variant(variant)


def actor_update_period(variant: str) -> int:
    return final.actor_update_period(variant)


def select_best_validation(variant: str) -> bool:
    return final.select_best_validation(variant)


def run_dir(variant: str, seed: int, mode: int):
    require_variant(variant)
    if require_mode(mode) != final.REFERENCE_MODE:
        raise ValueError("V31 trains only the preregistered canonical mode")
    return final.reference_run_dir(seed)


def bundle_dir(variant: str, seed: int, mode: int):
    require_variant(variant)
    if require_mode(mode) != final.REFERENCE_MODE:
        raise ValueError("V31 bundles only the preregistered canonical mode")
    return final.reference_bundle_dir(seed)


def bundle_manifest(variant: str, seed: int, mode: int):
    return bundle_dir(variant, seed, mode) / "bundle_manifest.json"


def bundle_required_paths(variant: str, seed: int, mode: int):
    require_variant(variant)
    if require_mode(mode) != final.REFERENCE_MODE:
        raise ValueError("V31 requires only the canonical reference bundle")
    return final.reference_required_paths(seed)


def bundle_records(variant: str, seed: int):
    return {
        str(final.REFERENCE_MODE): file_record(
            bundle_manifest(variant, seed, final.REFERENCE_MODE))
    }


def identity(variant: str, seed: int, mode: int):
    require_variant(variant)
    if require_mode(mode) != final.REFERENCE_MODE:
        raise ValueError("V31 identity is fixed to canonical mode 0")
    return final.reference_identity(seed)


def expected_checkpoint():
    return final.expected_reference_checkpoint()
