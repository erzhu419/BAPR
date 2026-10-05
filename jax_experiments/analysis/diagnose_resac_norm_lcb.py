"""Read remote checkpoints in place; emit small JSON, with no updates or rollout."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_experiments.common.checkpoint import (
    _patch_flax_variablestate_unpickle, _to_jax_tree,
)
from jax_experiments.networks.ensemble_critic import EnsembleCritic
from jax_experiments.networks.policy import GaussianPolicy


def disagreement(q_values):
    q = np.asarray(q_values, dtype=np.float64)
    means = q.mean(axis=1)
    centered = q - means[:, None]
    variance = q.var(axis=0).mean()
    return {"head_q_mean": means.tolist(), "offset_std": float(means.std()),
            "raw_std_mean": float(q.std(axis=0).mean()),
            "centered_std_mean": float(centered.std(axis=0).mean()),
            "offset_variance_fraction": float(means.var() / variance) if variance else 0.0}


def layer_norms(critic):
    return [{"layer": index,
             "kernel_l1": np.asarray(jnp.abs(layer.kernel.value).sum((1, 2))).tolist(),
             "bias_l1": np.asarray(jnp.abs(layer.bias.value).sum((1, 2))).tolist()}
            for index, layer in enumerate(critic.layers)]


def diagnose(directory, count=256):
    _patch_flax_variablestate_unpickle()
    config = json.loads((directory / "logs/protocol_signature.json").read_text())["config"]
    with (directory / "checkpoints/params.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    with np.load(directory / "checkpoints/replay_buffer.npz") as replay:
        size = int(replay["size"])
        indices = np.linspace(0, size - 1, min(count, size), dtype=int)
        obs, act = (jnp.asarray(replay[name][indices]) for name in ("obs", "act"))
        rewards, done = (replay[name][indices] for name in ("rew", "done"))
    kwargs = dict(hidden_dim=config["hidden_dim"], ensemble_size=config["ensemble_size"])
    critic = EnsembleCritic(obs.shape[-1], act.shape[-1], **kwargs, rngs=nnx.Rngs(0))
    target = EnsembleCritic(obs.shape[-1], act.shape[-1], **kwargs, rngs=nnx.Rngs(0))
    policy = GaussianPolicy(obs.shape[-1], act.shape[-1], config["hidden_dim"], rngs=nnx.Rngs(0))
    for name, model in (("critic", critic), ("target_critic", target), ("policy", policy)):
        nnx.update(model, _to_jax_tree(saved[name]))

    layers = layer_norms(target)
    legacy_norm = sum(np.asarray(row["kernel_l1"]) + row["bias_l1"] for row in layers)
    reference_norm = legacy_norm - np.asarray(layers[-1]["bias_l1"])
    bonus = config["weight_reg"] * legacy_norm
    q = np.asarray(critic(obs, act))
    head_means = q.mean(axis=1)
    gamma = config["gamma"]
    # This continuing-task approximation diagnoses offsets; it is not a new target.
    predicted_offsets = gamma / (1 - gamma) * bonus
    beta = config["resac_beta_end"] if config["resac_adaptive_beta"] else config["beta"]
    graph, params = nnx.split(policy, nnx.Param)
    alpha = jnp.exp(jnp.asarray(saved["log_alpha"]))
    key = jax.random.PRNGKey(7)

    def loss(candidate, extra, subtract):
        model = nnx.merge(graph, candidate)
        actions, log_prob = model.sample(obs, key)
        values = critic(obs, actions) + extra[:, None] - subtract[:, None]
        objective = values.mean(axis=0) + beta * values.std(axis=0)
        bc = config["resac_beta_bc"] * jnp.mean((actions - act) ** 2)
        return (alpha * log_prob - objective).mean() + bc

    zero = np.zeros_like(bonus)
    probes = (("legacy", bonus, zero), ("no_actor_bonus", zero, zero),
              ("shared_actor_bonus", np.full_like(bonus, bonus.mean()), zero),
              ("bootstrap_offset_removed", bonus, predicted_offsets),
              ("head_mean_centered", zero, head_means))
    gradients = {}
    for name, extra, subtract in probes:
        value, grad = jax.value_and_grad(loss)(params, jnp.asarray(extra), jnp.asarray(subtract))
        vector = np.concatenate([np.asarray(leaf).ravel() for leaf in jax.tree.leaves(grad)])
        if name == "legacy":
            reference_grad = vector
        norm, reference_size = np.linalg.norm(vector), np.linalg.norm(reference_grad)
        gradients[name] = {"loss": float(value), "norm": float(norm),
                           "cosine_vs_legacy": float(vector @ reference_grad / (norm * reference_size)),
                           "relative_difference": float(np.linalg.norm(vector - reference_grad) / reference_size)}

    metrics = {}
    for name in ("iteration", "total_steps", "q_mean", "q_std_mean", "reg_bonus_mean",
                 "reg_bonus_std", "critic_loss", "alpha", "eval_reward"):
        values = np.load(directory / "logs" / f"{name}.npy")
        positions = sorted({i for i in (0, 1, 10, 50, 100, 250, 500, 1000, len(values) - 1)
                            if i < len(values)})
        metrics[name] = {"count": len(values), "finite": bool(np.isfinite(values).all()),
                         "samples": [[i, float(values[i])] for i in positions]}
    correlation = float(np.corrcoef(bonus, head_means)[0, 1]) if bonus.std() else None
    return {"run": str(directory.resolve()), "update_count": int(saved["update_count"]),
            "training_updates_executed": 0, "sample_count": len(indices),
            "config": {name: config[name] for name in (
                "gamma", "weight_reg", "beta_ood", "resac_beta_bc", "resac_independent_ratio",
                "resac_critic_actor_ratio", "resac_clip_norm", "resac_adaptive_beta", "beta")},
            "layers": layers, "legacy_bonus": bonus.tolist(),
            "reference_bonus": (config["weight_reg"] * reference_norm).tolist(),
            "predicted_continuing_offsets": predicted_offsets.tolist(),
            "disagreement": disagreement(q),
            "after_predicted_offset_removal": disagreement(q - predicted_offsets[:, None]),
            "bonus_q_correlation": correlation,
            "batch_reward": {"mean": float(rewards.mean()), "abs_mean": float(abs(rewards).mean()),
                             "std": float(rewards.std()), "done_fraction": float(done.mean())},
            "effective_beta": beta, "actor_gradient": gradients, "logs": metrics,
            "scope": "Frozen V39 weights and own final replay; centering is diagnostic only, not an evaluated policy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [diagnose(directory) for directory in args.run]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"RE-SAC zero-update diagnosis: {args.output}", flush=True)


if __name__ == "__main__":
    main()
