"""V20-runner compatibility view for the frozen V21 specialist recipe."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as final,
)


ROOT = final.ROOT
PROTOCOL_VERSION = final.PROTOCOL_VERSION
ENV = final.ENV
FAMILY = final.FAMILY
MODES = final.MODES
VARIANTS = (final.SPECIALIST_VARIANT,)
CONTROL_VARIANT = final.SPECIALIST_VARIANT
SPECIALIST_VARIANT = final.SPECIALIST_VARIANT
TRAINING_SEEDS = final.TRAINING_SEEDS

CALIBRATION_EVENT_SEEDS = final.CALIBRATION_EVENT_SEEDS
STATIONARY_HOLDOUT_EVENT_SEEDS = final.STATIONARY_HOLDOUT_EVENT_SEEDS
SWITCHING_EVENT_SEEDS = final.SWITCHING_EVENT_SEEDS
SWITCHING_SCHEDULES = final.SWITCHING_SCHEDULES

SOURCE_NEXT_ITERATION = final.SOURCE_NEXT_ITERATION
SOURCE_ITERATION = final.SOURCE_ITERATION
SOURCE_TOTAL_STEPS = final.SOURCE_TOTAL_STEPS
SOURCE_UPDATE_COUNT = final.SOURCE_UPDATE_COUNT
SOURCE_START_TRAIN_STEPS = final.SOURCE_START_TRAIN_STEPS
FINETUNE_ITERS = final.SPECIALIST_FINETUNE_ITERS
FINAL_NEXT_ITERATION = final.SPECIALIST_FINAL_NEXT_ITERATION
FINAL_ITERATION = final.SPECIALIST_FINAL_ITERATION
FINAL_TOTAL_STEPS = final.SPECIALIST_FINAL_TOTAL_STEPS
FINAL_UPDATE_COUNT = final.SPECIALIST_FINAL_UPDATE_COUNT

SAMPLES_PER_ITER = final.SAMPLES_PER_ITER
UPDATES_PER_ITER = final.UPDATES_PER_ITER
DWELL_STEPS = final.DWELL_STEPS
MAX_EPISODE_STEPS = final.MAX_EPISODE_STEPS
EPISODES_PER_TASK = final.EPISODES_PER_TASK
SWITCHING_EPISODES = final.SWITCHING_EPISODES
MIN_CALIBRATION_GAIN = final.MIN_CALIBRATION_GAIN

SOURCE_RUN_ROOT = final.SOURCE_RUN_ROOT
SOURCE_BUNDLE_ROOT = final.SOURCE_BUNDLE_ROOT
SPECIALIST_RUN_ROOT = final.SPECIALIST_RUN_ROOT
SPECIALIST_BUNDLE_ROOT = final.SPECIALIST_BUNDLE_ROOT
REGISTRATION_ROOT = final.REGISTRATION_ROOT
REGISTRATION_PATH = final.REGISTRATION_PATH

POLICY_NAME = final.POLICY_NAME
CONTROLLER_STATE_NAME = final.CONTROLLER_STATE_NAME
BOOTSTRAP_NAME = final.BOOTSTRAP_NAME
SELECTION_NAME = final.SELECTION_NAME
SOURCE_BUNDLE_SCHEMA = final.SOURCE_BUNDLE_SCHEMA
BUNDLE_SCHEMA = final.SPECIALIST_BUNDLE_SCHEMA
BOOTSTRAP_SCHEMA = final.SPECIALIST_BOOTSTRAP_SCHEMA
AUDIT_SCHEMA = final.AUDIT_SCHEMA

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
    value = str(variant)
    if value != final.SPECIALIST_VARIANT:
        raise ValueError(f"unknown v21 specialist variant {value!r}")
    return value


def actor_update_period(variant: str) -> int:
    require_variant(variant)
    return 1


def select_best_validation(variant: str) -> bool:
    require_variant(variant)
    return False


def run_dir(variant: str, seed: int, mode: int):
    require_variant(variant)
    return final.specialist_run_dir(seed, mode)


def bundle_dir(variant: str, seed: int, mode: int):
    require_variant(variant)
    return final.specialist_bundle_dir(seed, mode)


def bundle_manifest(variant: str, seed: int, mode: int):
    require_variant(variant)
    return final.specialist_bundle_manifest(seed, mode)


def bundle_required_paths(variant: str, seed: int, mode: int):
    require_variant(variant)
    return final.specialist_required_paths(seed, mode)


def bundle_records(variant: str, seed: int):
    require_variant(variant)
    return final.specialist_bundle_records(seed)


def identity(variant: str, seed: int, mode: int):
    require_variant(variant)
    return final.specialist_identity(seed, mode)


def expected_checkpoint():
    return final.expected_specialist_checkpoint()
