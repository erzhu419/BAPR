"""Export existing NumPy-2 model PKLs as portable arrays, without retraining."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_experiments.analysis import run_mechanism_gap_v34 as common
from jax_experiments.analysis import run_submission_extensions_v47 as run
from jax_experiments.analysis import submission_extensions_protocol_v47 as p


def main():
    for seed in p.HC_MODELS:
        original = common.p.model_dir(seed)
        metadata = json.loads((original / "complete.json").read_text())
        model = common.inverse_model(17, 6, seed)
        params = common._restore_tree_like(nnx.state(model, nnx.Param),
            common.read_pickle(original / "inverse_params.pkl"), "historical inverse", allow_fallback=False)
        directory = p.hc_model_dir(seed)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "inverse_params.npz", **{
            f"arr_{index}": np.asarray(value) for index, value in enumerate(jax.tree.leaves(params))})
        common.write_json(directory / "complete.json", {
            **metadata, "model_seed": seed, "obs_dim": 17, "act_dim": 6,
            "exported_without_training": True, "original_training_runtime_recorded": False})
        release = json.loads((common.p.RELEASE / "estimator/model_manifest.json").read_text())
        gains = jnp.asarray(release["mode_gain_vectors"], jnp.float32)
        expected = common.inverse_evidence(model, params, gains, jnp.asarray(metadata["residual_variance"]))
        restored = run.hc_evidence(seed, gains)
        arguments = (jnp.arange(17, dtype=jnp.float32)*.01, jnp.arange(6, dtype=jnp.float32)*.02,
                     jnp.arange(17, dtype=jnp.float32)*.011)
        error = float(jnp.max(jnp.abs(expected(*arguments)-restored(*arguments))))
        if error != 0:
            raise ValueError("compact export changed estimator evidence")
        print(f"EXPORTED model={seed} evidence_max_error={error}", flush=True)


if __name__ == "__main__":
    main()
