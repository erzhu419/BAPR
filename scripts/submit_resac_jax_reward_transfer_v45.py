"""Submit only the frozen JAX reward-transfer smoke/train/audit batch."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts import submit_resac_update_density_v42 as shared
from jax_experiments.analysis import resac_jax_reward_transfer_protocol_v45 as p

PREFIX = "BAPR/resac-jax-reward-transfer/v45"
MODULE = "jax_experiments.analysis.run_resac_jax_reward_transfer_v45"


def candidates():
    return shared.candidates(protocol=p, module=MODULE, prefix=PREFIX)


if __name__ == "__main__":
    shared.main(protocol=p, module=MODULE, prefix=PREFIX)
