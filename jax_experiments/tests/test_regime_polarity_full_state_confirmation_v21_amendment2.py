"""Focused checks for the V21 audit artifact-path amendment."""
from __future__ import annotations

from jax_experiments.analysis import (
    run_regime_polarity_full_state_confirmation_audit_v21_amendment2
    as amendment,
)
from scripts import (
    recover_regime_polarity_full_state_confirmation_v21_amendment2 as recover,
)


def test_amendment_restores_all_reused_v18_module_aliases(monkeypatch):
    monkeypatch.delattr(amendment.protocol, "BUNDLE_SCHEMA", raising=False)
    monkeypatch.delattr(amendment.v5_model, "MODEL_MANIFEST", raising=False)
    monkeypatch.delattr(amendment.v5_model, "MODEL_PATH", raising=False)
    amendment._bind()
    assert amendment.protocol.BUNDLE_SCHEMA == amendment.protocol.BASELINE_BUNDLE_SCHEMA
    assert (
        amendment.v5_model.MODEL_MANIFEST
        == amendment.v5_model.protocol.MODEL_MANIFEST
    )
    assert amendment.v5_model.MODEL_PATH == amendment.v5_model.protocol.MODEL_PATH


def test_recovery_submits_only_cpu_audits():
    spec = recover.audit_spec(recover.protocol.TRAINING_SEEDS[0], "high")
    assert spec["vram"] == 0
    assert recover.AUDIT_MODULE in spec["cmd"]
    assert spec["signature"].endswith("/amendment2")
    assert str(recover.AMENDMENT) in spec["wait_for_files"]
