from __future__ import annotations

import argparse

from jax_experiments.analysis import regime_damping_headroom_v2 as protocol
from jax_experiments.analysis import train_regime_damping_headroom_entry_v2


def test_amendment_does_not_change_scientific_contract():
    old = protocol.predecessor
    assert protocol.FAMILY == old.FAMILY
    assert protocol.ENVS == old.ENVS
    assert protocol.TRAINING_SEEDS == old.TRAINING_SEEDS
    assert protocol.AUDIT_EVENT_SEEDS == old.AUDIT_EVENT_SEEDS
    assert protocol.FINAL_TOTAL_STEPS == old.FINAL_TOTAL_STEPS
    assert protocol.FINAL_UPDATE_COUNT == old.FINAL_UPDATE_COUNT


def test_registered_family_is_accepted_by_isolated_parser_patch():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stochastic_mode_family",
        choices=["actuator_polarity"], required=True)
    parsed = parser.parse_args([
        "--stochastic_mode_family", protocol.FAMILY])
    assert parsed.stochastic_mode_family == protocol.FAMILY


def test_predecessor_produced_no_checkpoint():
    root = protocol.predecessor.RUN_ROOT
    assert not root.exists() or not tuple(root.rglob("train_state.pkl"))


def test_registration_source_closure_exists():
    assert all(path.is_file() for path in protocol.registration_source_paths())

