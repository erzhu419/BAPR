"""Aggregate the v2 persistent-damping oracle-headroom screen."""
from __future__ import annotations

from jax_experiments.analysis import (
    analyze_regime_control_headroom as common,
)
from jax_experiments.analysis import regime_damping_headroom_v2 as protocol
from jax_experiments.analysis import (
    run_regime_damping_headroom_audit_v2 as audit,
)


common.protocol = protocol
common.validate_audit = audit.validate_audit
common.T_CRITICAL_95_DF4 = 4.302652729696142
common.MIN_RELATIVE_GAIN = protocol.MIN_RELATIVE_GAIN
common.MAX_TERMINATION_GAP = protocol.MAX_TERMINATION_GAP
common.MIN_MODE_WINS = protocol.MIN_MODE_WINS
common.MIN_PASSING_ENVS = protocol.MIN_PASSING_ENVS


if __name__ == "__main__":
    protocol.validate_registration()
    common.main()

