"""Train one preregistered final mode-head student initialization."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_fallback_final_comparison_v1 as protocol,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation_control_v2 as trainer,
)


# The generic trainer is deliberately rebound to the frozen final protocol.
# Its collector and model helper modules keep module-level protocol references.
trainer.protocol = protocol
trainer.base_trainer.protocol = protocol
trainer.model_lib.protocol = protocol


def main() -> None:
    protocol.validate_registration()
    trainer.main()


if __name__ == "__main__":
    main()
