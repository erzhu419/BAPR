"""Focused checks for the V21 export-only recovery."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_full_state_final_confirmation_v21 as protocol,
)
from jax_experiments.analysis import (
    run_regime_polarity_full_state_confirmation_baseline_export_v21_amendment1
    as exporter,
)
from scripts import (
    recover_regime_polarity_full_state_confirmation_v21_amendment1 as recover,
)


def test_amendment_binds_registered_baseline_schema():
    exporter._bind()
    assert protocol.BUNDLE_SCHEMA == protocol.BASELINE_BUNDLE_SCHEMA
    assert exporter.trainer.base.protocol is protocol


def test_recovery_is_export_only_and_fail_closed_on_checkpoint_scan():
    seed = protocol.TRAINING_SEEDS[0]
    spec = recover.export_spec(
        "sac_replica", seed, 1, "high", "jtl110gpu")
    assert recover.EXPORT_MODULE in spec["cmd"]
    assert "jax_experiments.train" not in spec["cmd"]
    assert spec["skip_resume_scan"] is True
    assert "ckpt_dir" not in spec
    assert "ckpt_glob" not in spec
    assert spec["require_node"] == "jtl110gpu"
    assert "local" not in spec["allowed_nodes"]


def test_recovery_graph_has_thirty_exports_and_five_audits():
    known = []
    for seed in protocol.TRAINING_SEEDS:
        for slot in protocol.SAC_REPLICA_SLOTS:
            known.append({
                "signature": recover.original.baseline_signature(
                    "sac_replica", seed, slot),
                "resume_locations": [{
                    "node": "jtl110gpu", "mtime": 1.0, "size": 1,
                }],
            })
        for kind in protocol.TRAINED_METHODS:
            known.append({
                "signature": recover.original.baseline_signature(
                    kind, seed, None),
                "resume_locations": [{
                    "node": "jtl110gpu", "mtime": 1.0, "size": 1,
                }],
            })
    rows = recover.candidates("high", known)
    assert len(rows) == 35
    assert sum(spec["vram"] > 0 for _, spec, _ in rows) == 30
    assert sum(spec["vram"] == 0 for _, spec, _ in rows) == 5
    assert all(str(recover.AMENDMENT) in spec["wait_for_files"]
               for _, spec, _ in rows)


def test_checkpoint_node_falls_back_only_to_completed_progress():
    seed = protocol.TRAINING_SEEDS[0]
    signature = recover.original.baseline_signature(
        "sac_replica", seed, 1)
    tasks = [{
        "signature": signature,
        "node": "node007",
        "last_progress_line": "Resumed from checkpoint: iter=1400, steps=5600000",
        "finished_at": 2.0,
    }]
    assert recover._checkpoint_node(
        tasks, "sac_replica", seed, 1) == "node007"
