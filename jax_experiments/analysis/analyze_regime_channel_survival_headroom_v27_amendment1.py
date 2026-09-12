"""Aggregate V27 after declaring the registered robust-context sentinel."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_channel_survival_headroom_v27 as protocol,
)
from jax_experiments.analysis import (
    analyze_regime_channel_survival_headroom_v27 as original,
)


def main() -> None:
    protocol.ROBUST_TRACE_CONTEXT_MODE_ID = -1
    original.main()


if __name__ == "__main__":
    main()
