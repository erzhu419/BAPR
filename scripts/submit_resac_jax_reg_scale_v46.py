"""Submit the positive-sign reward-standardized scale pilot."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts import submit_resac_update_density_v42 as shared
from jax_experiments.analysis import resac_jax_reg_scale_protocol_v46 as p

PREFIX = "BAPR/resac-jax-reg-scale/v46"
MODULE = "jax_experiments.analysis.run_resac_jax_reg_scale_v46"


def candidates():
    return shared.candidates(protocol=p, module=MODULE, prefix=PREFIX)


if __name__ == "__main__":
    shared.main(protocol=p, module=MODULE, prefix=PREFIX)
