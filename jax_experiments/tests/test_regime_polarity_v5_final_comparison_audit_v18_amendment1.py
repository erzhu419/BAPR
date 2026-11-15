"""Regression tests for the provenance-only v18 audit amendment."""
from __future__ import annotations

from jax_experiments.analysis import (
    run_regime_polarity_v5_final_comparison_audit_v18_amendment1 as amendment,
)


def test_amendment_restores_the_loader_artifact_path_api(monkeypatch):
    monkeypatch.delattr(amendment.v5_model, "MODEL_MANIFEST", raising=False)
    monkeypatch.delattr(amendment.v5_model, "MODEL_PATH", raising=False)
    amendment.install_v5_artifact_path_aliases()
    assert (
        amendment.v5_model.MODEL_MANIFEST
        == amendment.protocol.v5_model.MODEL_MANIFEST
    )
    assert amendment.v5_model.MODEL_PATH == amendment.protocol.v5_model.MODEL_PATH


def test_amendment_declares_no_scientific_protocol_change():
    payload = amendment.amendment_registration_payload()
    assert payload["scientific_protocol_changes"] == []
    assert payload["parent_registration"] == amendment.protocol.file_record(
        amendment.protocol.REGISTRATION_PATH)
