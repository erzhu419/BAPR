from collections.abc import Mapping
import importlib

import numpy as np

from jax_experiments.analysis import bapr_v3_independent_specialists as protocol


class _StateLike(Mapping):
    def __init__(self, values, label):
        self._values = dict(values)
        self._label = label

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return f"unstable-state-repr:{self._label}"


class _VariableLike:
    def __init__(self, value, label):
        self._value = value
        self._label = label

    def get_raw_value(self):
        return self._value

    def __repr__(self):
        return f"unstable-variable-repr:{self._label}"


def test_component_hash_ignores_mapping_repr_and_insertion_order():
    first = _StateLike({
        "weight": _VariableLike(np.arange(6, dtype=np.float32), "gpu"),
        "bias": _VariableLike(np.array([1.0], dtype=np.float32), "gpu"),
    }, "gpu")
    second = _StateLike({
        "bias": _VariableLike(np.array([1.0], dtype=np.float32), "cpu"),
        "weight": _VariableLike(np.arange(6, dtype=np.float32), "cpu"),
    }, "cpu")

    assert protocol.value_sha256(first) == protocol.value_sha256(second)


def test_component_hash_changes_when_variable_value_changes():
    first = _VariableLike(np.array([1.0, 2.0], dtype=np.float32), "one")
    second = _VariableLike(np.array([1.0, 3.0], dtype=np.float32), "two")

    assert protocol.value_sha256(first) != protocol.value_sha256(second)


def test_stochastic_headroom_profile_uses_env_scoped_roots():
    try:
        protocol.configure_stochastic_headroom("HalfCheetah-v2")

        assert protocol.FAMILIES == ("packet_loss", "burst_torque")
        assert protocol.ENV == "HalfCheetah-v2"
        assert protocol.source_pair("burst_torque").name == (
            "budget_fork_v2_burst_torque_HalfCheetah_s0"
        )
        assert protocol.family_run_root("burst_torque").parts[-2:] == (
            "burst_torque", "HalfCheetah"
        )
        assert protocol.family_bundle_root("burst_torque").parts[-2:] == (
            "burst_torque", "HalfCheetah"
        )
    finally:
        importlib.reload(protocol)
