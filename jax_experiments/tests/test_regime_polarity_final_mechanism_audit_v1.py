"""Structural tests for the frozen final-stack mechanism audit."""
from __future__ import annotations

import numpy as np

from jax_experiments.analysis import (
    analyze_regime_polarity_final_mechanism_audit_v1 as analyze,
)
from jax_experiments.analysis import (
    regime_polarity_final_mechanism_audit_v1 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_final_mechanism_audit_v1 as runner,
)
from scripts import submit_regime_polarity_final_mechanism_audit_v1 as submit


def test_mechanism_split_is_new_and_balanced():
    assert len(protocol.STUDENT_SEEDS) == 5
    assert len(protocol.EVENT_SEEDS) == len(set(protocol.EVENT_SEEDS)) == 5
    protocol.assert_split_integrity()
    assert len(protocol.STATIONARY_ARMS) == 10
    assert len(protocol.SWITCHING_ARMS) == 17


def test_wrong_context_map_is_a_deterministic_derangement():
    for seed in protocol.EVENT_SEEDS:
        mapping = protocol.shuffled_mode_map(seed)
        assert set(mapping) == set(protocol.MODES)
        assert all(source != target
                   for source, target in zip(protocol.MODES, mapping))
        assert mapping == protocol.shuffled_mode_map(seed)


def test_context_arms_have_declared_semantics():
    posterior = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    event_seed = protocol.EVENT_SEEDS[0]
    learned = runner._context_for_arm(
        protocol.LEARNED_ARM, 2, posterior, event_seed, None)
    true = runner._context_for_arm(
        protocol.TRUE_ARM, 2, posterior, event_seed, None)
    zero = runner._context_for_arm(
        protocol.ZERO_ARM, 2, posterior, event_seed, None)
    delayed = runner._context_for_arm(
        "student_true_delay_10", 2, posterior, event_seed, 1)
    assert np.array_equal(learned, posterior)
    assert np.array_equal(true, np.asarray([0, 0, 1, 0], dtype=np.float32))
    assert np.array_equal(zero, np.zeros(4, dtype=np.float32))
    assert np.array_equal(
        delayed, np.asarray([0, 1, 0, 0], dtype=np.float32))


def test_registration_sources_and_scheduler_dag_are_complete():
    paths = protocol.registration_source_paths()
    assert paths and all(path.is_file() for path in paths)
    payload = protocol.registration_payload()
    assert len(payload["source_records"]) == len(paths)
    rows = submit.candidates("all", "high")
    assert len(rows) == 26
    assert len({signature for signature, _, _ in rows}) == 26
    audits = [spec for signature, spec, _ in rows if "/audit/" in signature]
    assert len(audits) == 25
    assert all(spec["vram"] == 0 for spec in audits)
    assert all(spec["allowed_nodes"] == submit.CPU_NODES for spec in audits)
    assert all(spec["cpu"] == 32 for spec in audits)
    analysis = [spec for signature, spec, _ in rows
                if signature.endswith("/analysis")]
    assert len(analysis) == 1
    assert len(analysis[0]["wait_for_files"]) == 25


def test_clustered_pairing_uses_registered_event_clusters():
    reference = {seed: 100.0 for seed in protocol.EVENT_SEEDS}
    candidate = {seed: 110.0 + index
                 for index, seed in enumerate(protocol.EVENT_SEEDS)}
    row = analyze._paired(candidate, reference)
    assert row["event_wins"] == 5
    assert row["mean_delta"] == 12.0
    assert row["clustered_95pct_interval"][0] > 0.0


if __name__ == "__main__":
    test_mechanism_split_is_new_and_balanced()
    test_wrong_context_map_is_a_deterministic_derangement()
    test_context_arms_have_declared_semantics()
    test_registration_sources_and_scheduler_dag_are_complete()
    test_clustered_pairing_uses_registered_event_clusters()
