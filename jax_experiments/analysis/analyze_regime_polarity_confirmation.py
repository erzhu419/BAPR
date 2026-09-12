"""Aggregate the fresh five-seed HalfCheetah polarity confirmation."""
from __future__ import annotations

from jax_experiments.analysis import regime_polarity_confirmation as protocol
from jax_experiments.analysis import (
    run_regime_control_headroom_controller as controller,
)
from jax_experiments.analysis import (
    run_regime_control_headroom_audit as audit,
)
from jax_experiments.analysis import (
    analyze_regime_control_headroom as common,
)


controller.protocol = protocol
audit.protocol = protocol
common.protocol = protocol
common.validate_audit = audit.validate_audit
common.T_CRITICAL_95_DF4 = 2.7764451051977987
common.MIN_RELATIVE_GAIN = protocol.MIN_RELATIVE_GAIN
common.MAX_TERMINATION_GAP = protocol.MAX_TERMINATION_GAP
common.MIN_MODE_WINS = protocol.MIN_MODE_WINS
common.MIN_PASSING_ENVS = protocol.MIN_PASSING_ENVS


if __name__ == "__main__":
    common.main()
