"""Compare the actual JAX scan against a fixed-noise original-Torch trace."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.algos.resac_torch import TorchEquationRESAC
from jax_experiments.configs.default import Config


def modules(agent):
    for name in ("policy", "critic", "target_critic"):
        model = getattr(agent, name)
        fields = {}
        for index, layer in enumerate(model.layers):
            fields[f"layers/{index}/kernel"] = layer.kernel
            fields[f"layers/{index}/bias"] = layer.bias
        if name == "policy":
            for head in ("mean_head", "log_std_head"):
                layer = getattr(model, head)
                fields[head + "/kernel"], fields[head + "/bias"] = layer.kernel, layer.bias
        else:
            fields["norm/scale"], fields["norm/bias"] = model.norm.scale, model.norm.bias
        yield name, fields


def optimizer_arrays(state):
    result = {}
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        names = [str(getattr(item, "key", getattr(item, "name", ""))) for item in path]
        result["/".join(names)] = np.asarray(value)
    return result


def check(reference, destination=None):
    largest = {"parameters": 0.0, "moments": 0.0}
    with np.load(reference) as trace:
        metadata = json.loads(str(trace["metadata"]))
        for arm, weight in (("full", 0.01), ("no_shift", 0.0)):
            config = Config(hidden_dim=8, ensemble_size=3, weight_reg=weight,
                            resac_critic_actor_ratio=2, resac_clip_norm=1.0, resac_beta_bc=0.001)
            agent = TorchEquationRESAC(3, 2, config, seed=2027)
            for name, fields in modules(agent):
                for key, variable in fields.items():
                    variable.value = jnp.asarray(trace[f"{arm}/initial/{name}/{key}"])
            agent.log_alpha = jnp.asarray(trace[arm + "/initial/log_alpha"])
            # Optimizer templates predate replacement values, but their zero moments have identical shapes.
            for step in range(metadata["steps"]):
                batch = {name: jnp.asarray(trace[name][step:step + 1])
                         for name in ("obs", "act", "rew", "next_obs", "done")}
                final, metrics = agent._scan_update(
                    nnx.state(agent.critic, nnx.Param), nnx.state(agent.target_critic, nnx.Param),
                    nnx.state(agent.policy, nnx.Param), agent.log_alpha,
                    agent.critic_opt_state, agent.policy_opt_state, agent.alpha_opt_state,
                    batch, jnp.asarray(trace["policy_noise"][step:step + 1]),
                    jnp.asarray(trace["next_noise"][step:step + 1]), step)
                cp, tp, pp, agent.log_alpha, agent.critic_opt_state, agent.policy_opt_state, agent.alpha_opt_state = final
                nnx.update(agent.critic, cp)
                nnx.update(agent.target_critic, tp)
                nnx.update(agent.policy, pp)
                prefix = f"{arm}/step{step}"
                for name, fields in modules(agent):
                    for key, variable in fields.items():
                        actual, expected = np.asarray(variable.value), trace[f"{prefix}/{name}/{key}"]
                        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-6,
                                                   err_msg=f"{prefix}/{name}/{key}")
                        largest["parameters"] = max(largest["parameters"], float(abs(actual - expected).max()))
                    if name != "target_critic":
                        state = getattr(agent, name + "_opt_state")[0]
                        for label, moment in (("m", state.mu), ("v", state.nu)):
                            arrays = optimizer_arrays(moment)
                            for key in fields:
                                actual = arrays[key + "/value"]
                                expected = trace[f"{prefix}/{name}_adam_{label}/{key}"]
                                np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-6,
                                    err_msg=f"{prefix}/{name}_adam_{label}/{key}")
                                largest["moments"] = max(largest["moments"], float(abs(actual - expected).max()))
                np.testing.assert_allclose(agent.log_alpha, trace[prefix + "/log_alpha"], atol=1e-7)
                for label, moment in (("m", agent.alpha_opt_state[0].mu), ("v", agent.alpha_opt_state[0].nu)):
                    np.testing.assert_allclose(moment, trace[prefix + "/alpha_adam_" + label], rtol=3e-5, atol=1e-6)
                assert float(metrics[7][0]) == float(step % 2 == 0)
    result = {"pass": True, "source": metadata["source"], "torch": metadata["torch"],
              "jax": jax.__version__, "steps_per_arm": metadata["steps"], "arms": ["full", "no_shift"],
              "precision": metadata["precision"], "largest_abs_error": largest}
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check(args.reference, args.output)
