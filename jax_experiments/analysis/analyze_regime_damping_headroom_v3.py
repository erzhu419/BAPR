"""Aggregate corrected robust audits with frozen v2 oracle audits."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_control_headroom as common,
)
from jax_experiments.analysis import regime_damping_headroom_v2 as v2_protocol
from jax_experiments.analysis import regime_damping_headroom_v3 as protocol
from jax_experiments.analysis import (
    run_regime_damping_headroom_audit_v2 as v2_audit,
)
from jax_experiments.analysis import (
    run_regime_damping_headroom_audit_v3 as v3_audit,
)


def validate_mixed_audit(env: str, role: str, seed: int):
    if role == "robust":
        v3_audit.common.protocol = protocol
        return v3_audit.validate_audit(env, role, seed)
    previous = v2_audit.common.protocol
    try:
        v2_audit.common.protocol = v2_protocol
        return v2_audit.validate_audit(env, role, seed)
    finally:
        v2_audit.common.protocol = previous


common.protocol = protocol
common.validate_audit = validate_mixed_audit
common.T_CRITICAL_95_DF4 = 4.302652729696142
common.MIN_RELATIVE_GAIN = protocol.MIN_RELATIVE_GAIN
common.MAX_TERMINATION_GAP = protocol.MAX_TERMINATION_GAP
common.MIN_MODE_WINS = protocol.MIN_MODE_WINS
common.MIN_PASSING_ENVS = protocol.MIN_PASSING_ENVS


if __name__ == "__main__":
    protocol.validate_registration()
    common.main()
