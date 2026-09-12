"""Train one fresh-cohort frozen-pipeline BAPR deployment student."""
from __future__ import annotations

from jax_experiments.analysis import (
    regime_polarity_deployment_confirmation_v1 as protocol,
)
from jax_experiments.analysis import (
    train_regime_polarity_policy_distillation_control_v2 as trainer,
)


trainer.protocol = protocol
trainer.base_trainer.protocol = protocol
trainer.model_lib.protocol = protocol


def main() -> None:
    protocol.validate_mechanism_release()
    trainer.main()


if __name__ == "__main__":
    main()

