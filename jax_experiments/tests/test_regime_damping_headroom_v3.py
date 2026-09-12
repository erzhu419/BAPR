from __future__ import annotations

from jax_experiments.analysis import regime_damping_headroom_v3 as protocol


def test_amendment_changes_only_the_robust_trace_contract():
    old = protocol.predecessor
    assert protocol.FAMILY == old.FAMILY
    assert protocol.ENVS == old.ENVS
    assert protocol.TRAINING_SEEDS == old.TRAINING_SEEDS
    assert protocol.AUDIT_EVENT_SEEDS == old.AUDIT_EVENT_SEEDS
    assert protocol.FINAL_TOTAL_STEPS == old.FINAL_TOTAL_STEPS
    assert protocol.FINAL_UPDATE_COUNT == old.FINAL_UPDATE_COUNT
    assert protocol.RUN_ROOT == old.RUN_ROOT
    assert protocol.BUNDLE_ROOT == old.BUNDLE_ROOT
    assert protocol.ROBUST_TRACE_CONTEXT_MODE_ID == -1


def test_v3_reuses_frozen_oracle_audits_but_not_robust_audits():
    env = protocol.ENVS[0]
    seed = protocol.TRAINING_SEEDS[0]
    assert (protocol.audit_dir(env, "oracle", seed)
            == protocol.predecessor.audit_dir(env, "oracle", seed))
    assert (protocol.audit_dir(env, "robust", seed)
            != protocol.predecessor.audit_dir(env, "robust", seed))


def test_failed_v2_task_inventory_is_complete():
    assert len(protocol.FAILED_V2_ROBUST_AUDITS) == 12
    assert len(set(protocol.FAILED_V2_ROBUST_AUDITS)) == 12


def test_registration_source_closure_exists():
    assert all(path.is_file() for path in protocol.registration_source_paths())
